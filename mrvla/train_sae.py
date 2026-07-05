"""Train Sparse Autoencoders on collected OpenVLA residual-stream activations.

Implements the TopK + AuxK architecture and hyperparameters from:
  "Sparse Autoencoders Reveal Interpretable and Steerable Features in VLA Models"
  (arXiv 2603.19183), Appendix B.1, Table 6.

One SAE is trained per layer. For OpenVLA (d=4096) the paper uses expansion
ratio 0.5, giving a dictionary of 2048 features — the same size as the pi0.5
PaliGemma SAEs — to keep comparisons meaningful at robotics dataset scales.

Usage
-----
# Train on all captured layers, one at a time:
python train_sae.py \
    --acts-dir ./activations/libero_spatial \
    --out-dir  ./checkpoints/sae_libero_spatial \
    --layer    all

# Train on a single layer:
python train_sae.py \
    --acts-dir ./activations/libero_spatial \
    --out-dir  ./checkpoints/sae_libero_spatial \
    --layer    24

Output
------
checkpoints/sae_libero_spatial/
  layer_00/  final.pt  train_log.json
  layer_08/  ...
  layer_16/  ...
  layer_24/  ...
  layer_31/  ...

Each final.pt contains:
  {
    "W_enc":   [d, n_features]   encoder weight
    "W_dec":   [n_features, d]   decoder weight  (unit-norm columns)
    "b_pre":   [d]               pre-bias (geometric median)
    "config":  { ... }           all hyperparameters + manifest info
  }
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# ---------------------------------------------------------------------------
# Hyperparameters (Table 6 + OpenVLA-specific notes from Appendix B.1)
# ---------------------------------------------------------------------------
DEFAULTS = dict(
    expansion_ratio=0.5,   # OpenVLA: ER=0.5 → 2048 features for d=4096
    k=100,                 # active features per forward pass
    k_aux=512,             # AuxK: top-k dead latents for auxiliary loss
    aux_coeff=1 / 32,      # α in Eq. 5
    lr=1e-4,
    beta1=0.9,
    beta2=0.999,
    batch_size=4096,
    epochs=100,
    geo_median_samples=10_000,  # samples used to init b_pre
    dead_steps_threshold=500,   # steps before a latent is considered dead
    grad_clip=1.0,
)


# ---------------------------------------------------------------------------
# Geometric median (simple Weiszfeld, fast enough for 10k samples at d=4096)
# ---------------------------------------------------------------------------
def geometric_median(X: torch.Tensor, n_iter: int = 100, eps: float = 1e-6) -> torch.Tensor:
    """Approximate geometric median via Weiszfeld's algorithm. X: [N, d]."""
    y = X.mean(dim=0)
    for _ in range(n_iter):
        dists = (X - y).norm(dim=1, keepdim=True).clamp(min=eps)  # [N, 1]
        weights = 1.0 / dists                                       # [N, 1]
        y_new = (weights * X).sum(dim=0) / weights.sum()
        if (y_new - y).norm() < eps:
            break
        y = y_new
    return y


# ---------------------------------------------------------------------------
# SAE model
# ---------------------------------------------------------------------------
class TopKSAE(nn.Module):
    """TopK sparse autoencoder with AuxK auxiliary loss (Gao et al. 2024).

    Architecture (Eq. 3-4 from paper):
      x_norm = (x - b_pre - mu) / ||(x - b_pre - mu)||_2
      z      = ReLU(TopK(W_enc @ x_norm))          -- sparse codes
      x_hat  = un_normalize(W_dec @ z)              -- reconstruction

    Decoder columns are kept on the unit sphere via gradient projection.
    No bias in encoder or decoder.
    """

    def __init__(self, d: int, n_features: int, k: int, k_aux: int,
                 dead_steps_threshold: int = 500):
        super().__init__()
        self.d = d
        self.n_features = n_features
        self.k = k
        self.k_aux = k_aux
        self.dead_steps_threshold = dead_steps_threshold

        # Learned pre-bias (initialised externally from geometric median)
        self.b_pre = nn.Parameter(torch.zeros(d))

        # Encoder / decoder (no bias terms per paper)
        self.W_dec = nn.Parameter(torch.randn(n_features, d))
        self._normalize_decoder()

        # Init encoder as scaled transpose of decoder: W_enc = W_dec^T * sqrt(k/n)
        with torch.no_grad():
            scale = (k / n_features) ** 0.5
            self.W_enc = nn.Parameter(self.W_dec.data.T.clone() * scale)  # [d, n_features]

        # Step counter for dead-latent tracking
        self.register_buffer("steps_since_active", torch.zeros(n_features, dtype=torch.long))

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _normalize_decoder(self):
        """Project decoder columns onto the unit sphere."""
        norms = self.W_dec.data.norm(dim=1, keepdim=True).clamp(min=1e-8)
        self.W_dec.data = self.W_dec.data / norms

    @torch.no_grad()
    def _project_decoder_grad(self):
        """Project decoder gradient onto tangent plane of unit-norm constraint."""
        if self.W_dec.grad is None:
            return
        # grad_tangent = grad - (grad · col) * col  for each column
        dot = (self.W_dec.grad * self.W_dec.data).sum(dim=1, keepdim=True)
        self.W_dec.grad.data -= dot * self.W_dec.data

    # ------------------------------------------------------------------
    # normalisation helpers
    # ------------------------------------------------------------------
    def _prenorm(self, x: torch.Tensor):
        """Subtract b_pre, then per-sample mean + L2 normalise. Returns (x_norm, mu, l2)."""
        x_shifted = x - self.b_pre                               # [B, d]
        mu = x_shifted.mean(dim=1, keepdim=True)                 # [B, 1]
        x_centered = x_shifted - mu                              # [B, d]
        l2 = x_centered.norm(dim=1, keepdim=True).clamp(min=1e-8)
        x_norm = x_centered / l2                                 # [B, d]
        return x_norm, mu, l2

    def _unnorm(self, x_hat_norm: torch.Tensor, mu: torch.Tensor, l2: torch.Tensor) -> torch.Tensor:
        """Reverse per-sample norm, add b_pre back."""
        return x_hat_norm * l2 + mu + self.b_pre

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor):
        """
        Args:
            x: [B, d]  raw activations (float32)

        Returns:
            x_hat:      [B, d]   reconstruction (original activation space)
            x_norm:     [B, d]   normalised input (loss is computed here)
            x_hat_norm: [B, d]   normalised reconstruction
            z:          [B, n_features]  sparse codes
            aux_loss:   scalar   AuxK auxiliary loss (0 if no dead latents)
        """
        x_norm, mu, l2 = self._prenorm(x)                        # [B, d]

        # Encode
        pre_act = x_norm @ self.W_enc                            # [B, n_features]

        # TopK masking
        topk_vals, topk_idx = pre_act.topk(self.k, dim=1)
        z = torch.zeros_like(pre_act)
        z.scatter_(1, topk_idx, torch.relu(topk_vals))           # [B, n_features]

        # Decode in normalised space
        x_hat_norm = z @ self.W_dec                              # [B, d]
        x_hat = self._unnorm(x_hat_norm, mu, l2)                 # [B, d]

        # ------ AuxK loss on dead latents --------------------------------
        dead_mask = self.steps_since_active >= self.dead_steps_threshold
        n_dead = dead_mask.sum().item()
        if n_dead > 0:
            k_aux_eff = min(self.k_aux, n_dead)
            # residual in normalised space
            residual_norm = x_norm - x_hat_norm.detach()        # [B, d]
            dead_pre = pre_act[:, dead_mask]                     # [B, n_dead]
            aux_topk_vals, aux_topk_idx = dead_pre.topk(k_aux_eff, dim=1)
            z_aux = torch.zeros(x.shape[0], n_dead, device=x.device)
            z_aux.scatter_(1, aux_topk_idx, torch.relu(aux_topk_vals))
            W_dec_dead = self.W_dec[dead_mask]                   # [n_dead, d]
            x_hat_aux_norm = z_aux @ W_dec_dead                  # [B, d]
            aux_loss = (residual_norm - x_hat_aux_norm).pow(2).sum(dim=1).mean()
        else:
            aux_loss = x.new_zeros(1).squeeze()

        # Update dead-latent counter
        with torch.no_grad():
            fired = (z > 0).any(dim=0)                          # [n_features]
            self.steps_since_active[fired] = 0
            self.steps_since_active[~fired] += 1

        return x_hat, x_norm, x_hat_norm, z, aux_loss


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------
def sae_loss(x_norm: torch.Tensor, x_hat_norm: torch.Tensor,
             aux_loss: torch.Tensor, c_mse: float, aux_coeff: float):
    """Eq. 5: normalised MSE + α * AuxK loss.

    Both terms are computed in the unit-norm space so they share the same scale
    and ``aux_coeff`` (1/32) behaves as intended.
    """
    recon = (x_norm - x_hat_norm).pow(2).sum(dim=1).mean() / c_mse
    return recon + aux_coeff * aux_loss / c_mse, recon


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_all_activations(acts_dir: str) -> torch.Tensor:
    """Load every shard once into a single [N, L, H] float16 tensor.

    Kept in float16 to halve RAM (~2.6 GB for 63k×5×4096); per-layer slices are
    cast to float32 on demand by ``slice_layer``.
    """
    import glob
    shards = sorted(glob.glob(os.path.join(acts_dir, "shard_*.npz")))
    if not shards:
        raise FileNotFoundError(f"No shards found in {acts_dir!r}")
    chunks = [np.load(p)["acts"] for p in shards]   # each [n_i, L, H] float16
    full = np.concatenate(chunks, axis=0)
    return torch.from_numpy(full)                   # [N, L, H] float16


def slice_layer(all_acts: torch.Tensor, layer_pos: int) -> torch.Tensor:
    """Return [N, d] float32 view for one captured layer position."""
    return all_acts[:, layer_pos, :].to(torch.float32).contiguous()


def load_layer_activations(acts_dir: str, layer_pos: int) -> torch.Tensor:
    """Load all shard activations for one layer index position and return [N, d] float32."""
    import glob
    shards = sorted(glob.glob(os.path.join(acts_dir, "shard_*.npz")))
    if not shards:
        raise FileNotFoundError(f"No shards found in {acts_dir!r}")

    chunks = []
    for path in shards:
        d = np.load(path)
        # acts shape: [N, L, H] — pick the layer by position in captured set
        chunks.append(d["acts"][:, layer_pos, :].astype(np.float32))
    return torch.from_numpy(np.concatenate(chunks, axis=0))  # [N, d]


def load_manifest(acts_dir: str) -> dict:
    with open(os.path.join(acts_dir, "manifest.json")) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Training loop for one layer
# ---------------------------------------------------------------------------
def train_one_layer(
    X: torch.Tensor,
    layer_idx: int,
    layer_pos: int,
    out_dir: str,
    cfg: dict,
    device: str,
    resume_ckpt: str | None = None,
) -> dict:
    """Train a single SAE. Returns a log dict.

    If ``resume_ckpt`` is provided, loads W_enc/W_dec/b_pre/steps_since_active from
    that .pt file and skips geometric-median init. The Adam optimizer is
    re-initialised (we don't checkpoint optimizer state); a few warmup steps are
    needed for momentum to rebuild but with lr=1e-4 this is fine.
    """
    os.makedirs(out_dir, exist_ok=True)
    d = X.shape[1]
    n_features = max(1, int(d * cfg["expansion_ratio"]))

    print(f"\n[SAE] layer {layer_idx} (pos {layer_pos}) | d={d} | n_features={n_features} "
          f"| N={len(X):,} | device={device}")

    model = TopKSAE(d, n_features, k=cfg["k"], k_aux=cfg["k_aux"],
                    dead_steps_threshold=cfg["dead_steps_threshold"]).to(device)

    start_epoch = 0
    if resume_ckpt is not None:
        print(f"[SAE]   resuming from {resume_ckpt}")
        ck = torch.load(resume_ckpt, map_location=device, weights_only=False)
        with torch.no_grad():
            model.W_enc.data.copy_(ck["W_enc"].to(device))
            model.W_dec.data.copy_(ck["W_dec"].to(device))
            model.b_pre.data.copy_(ck["b_pre"].to(device))
            if "steps_since_active" in ck:
                model.steps_since_active.copy_(ck["steps_since_active"].to(device))
        start_epoch = int(ck.get("epoch", -1)) + 1
        print(f"[SAE]   loaded; resuming at epoch {start_epoch}")
    else:
        # ------ Init b_pre from geometric median ------------------------
        n_geo = min(cfg["geo_median_samples"], len(X))
        idx = torch.randperm(len(X))[:n_geo]
        geo_sample = X[idx].to(device)
        with torch.no_grad():
            model.b_pre.data = geometric_median(geo_sample)
        del geo_sample

    # ------ c_mse: variance of the *normalised* input ------------------
    # The loss is computed in unit-norm space, so the normaliser must be too.
    # Use a random subset (shards are ordered by task, so a head slice is biased).
    with torch.no_grad():
        n_c = min(10_000, len(X))
        sample = X[torch.randperm(len(X))[:n_c]].to(device)
        x_norm, _mu, _l2 = model._prenorm(sample)
        c_mse = float((x_norm - x_norm.mean(dim=0)).pow(2).mean())
        del sample, x_norm
    print(f"[SAE]   c_mse = {c_mse:.6f}")

    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg["lr"],
        betas=(cfg["beta1"], cfg["beta2"])
    )

    dataset = TensorDataset(X)
    loader = DataLoader(dataset, batch_size=cfg["batch_size"], shuffle=True,
                        pin_memory=(device != "cpu"), num_workers=0)

    log = {"layer_idx": layer_idx, "n_features": n_features, "d": d,
           "n_samples": len(X), "epochs": [], "config": cfg}

    for epoch in range(start_epoch, cfg["epochs"]):
        model.train()
        epoch_recon = 0.0
        epoch_loss = 0.0
        n_batches = 0
        t0 = time.time()

        for (xb,) in loader:
            xb = xb.to(device)
            optimizer.zero_grad()

            x_hat, x_norm, x_hat_norm, _z, aux_loss = model(xb)
            loss, recon = sae_loss(x_norm, x_hat_norm, aux_loss, c_mse, cfg["aux_coeff"])
            loss.backward()

            # Decoder gradient projection (unit-norm constraint)
            model._project_decoder_grad()
            nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            optimizer.step()

            # Re-normalise decoder columns after update
            model._normalize_decoder()

            epoch_recon += recon.item()
            epoch_loss += loss.item()
            n_batches += 1

        mean_recon = epoch_recon / n_batches
        mean_loss = epoch_loss / n_batches
        dead_frac = (model.steps_since_active >= model.dead_steps_threshold).float().mean().item()
        elapsed = time.time() - t0

        epoch_log = {
            "epoch": epoch,
            "loss": round(mean_loss, 6),
            "recon_loss": round(mean_recon, 6),
            "dead_frac": round(dead_frac, 4),
            "elapsed_s": round(elapsed, 1),
        }
        log["epochs"].append(epoch_log)

        if epoch % 10 == 0 or epoch == cfg["epochs"] - 1:
            print(f"  [layer {layer_idx:2d}] epoch {epoch:4d}/{cfg['epochs']} | "
                  f"loss={mean_loss:.4f} recon={mean_recon:.4f} "
                  f"dead={dead_frac:.3f} | {elapsed:.1f}s", flush=True)
        else:
            print(f"  [layer {layer_idx:2d}] epoch {epoch:4d} | "
                  f"loss={mean_loss:.4f}", flush=True)

        # Periodic checkpoint every 100 epochs so a killed job loses at most ~100 epochs.
        # Atomic write: save to .tmp, then rename.
        if epoch % 100 == 0 or epoch == cfg["epochs"] - 1:
            model.eval()
            ckpt = {
                "W_enc": model.W_enc.data.cpu(),
                "W_dec": model.W_dec.data.cpu(),
                "b_pre": model.b_pre.data.cpu(),
                "steps_since_active": model.steps_since_active.cpu(),
                "epoch": epoch,
                "config": {**cfg, "layer_idx": layer_idx, "layer_pos": layer_pos,
                           "d": d, "n_features": n_features},
            }
            tmp = os.path.join(out_dir, "checkpoint.tmp.pt")
            dst = os.path.join(out_dir, "checkpoint.pt")
            torch.save(ckpt, tmp)
            os.replace(tmp, dst)
            with open(os.path.join(out_dir, "train_log.json"), "w") as f:
                json.dump(log, f, indent=2)
            print(f"  [layer {layer_idx:2d}] checkpoint @ epoch {epoch} -> {dst}", flush=True)
            model.train()

    # ------ Final checkpoint (stable name) ----------------------------
    model.eval()
    ckpt = {
        "W_enc": model.W_enc.data.cpu(),        # [d, n_features]
        "W_dec": model.W_dec.data.cpu(),        # [n_features, d]
        "b_pre": model.b_pre.data.cpu(),        # [d]
        "steps_since_active": model.steps_since_active.cpu(),
        "epoch": cfg["epochs"] - 1,
        "config": {**cfg, "layer_idx": layer_idx, "layer_pos": layer_pos,
                   "d": d, "n_features": n_features},
    }
    torch.save(ckpt, os.path.join(out_dir, "final.pt"))

    with open(os.path.join(out_dir, "train_log.json"), "w") as f:
        json.dump(log, f, indent=2)

    print(f"[SAE]   saved to {out_dir}")
    return log


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--acts-dir", required=True,
                   help="Directory produced by collect_activations.py")
    p.add_argument("--out-dir", required=True,
                   help="Root output directory; one subdirectory per layer.")
    p.add_argument("--layer", default="all",
                   help='"all" or a single layer INDEX (not position), e.g. "24".')
    p.add_argument("--device", default="cuda:0")

    # Allow overriding key hyperparams from CLI
    p.add_argument("--expansion-ratio", type=float, default=DEFAULTS["expansion_ratio"])
    p.add_argument("--k", type=int, default=DEFAULTS["k"])
    p.add_argument("--k-aux", type=int, default=DEFAULTS["k_aux"])
    p.add_argument("--aux-coeff", type=float, default=DEFAULTS["aux_coeff"])
    p.add_argument("--lr", type=float, default=DEFAULTS["lr"])
    p.add_argument("--batch-size", type=int, default=DEFAULTS["batch_size"])
    p.add_argument("--epochs", type=int, default=DEFAULTS["epochs"])
    p.add_argument("--resume-from", default=None,
                   help="Optional path to a previous --out-dir; loads each layer's final.pt "
                        "and continues training to the new --epochs target. Skip layers "
                        "whose final.pt epoch is already >= --epochs.")
    return p.parse_args()


def main():
    args = parse_args()
    manifest = load_manifest(args.acts_dir)
    layer_indices = manifest["layer_indices"]   # e.g. [0, 8, 16, 24, 31]
    print(f"[SAE] manifest: model={manifest['model_name']} "
          f"layers={layer_indices} hidden={manifest['hidden_dim']} "
          f"total_samples={manifest['total_samples']}")

    cfg = dict(
        expansion_ratio=args.expansion_ratio,
        k=args.k,
        k_aux=args.k_aux,
        aux_coeff=args.aux_coeff,
        lr=args.lr,
        beta1=DEFAULTS["beta1"],
        beta2=DEFAULTS["beta2"],
        batch_size=args.batch_size,
        epochs=args.epochs,
        geo_median_samples=DEFAULTS["geo_median_samples"],
        dead_steps_threshold=DEFAULTS["dead_steps_threshold"],
        grad_clip=DEFAULTS["grad_clip"],
    )

    # Determine which layers to train
    if args.layer.strip().lower() == "all":
        targets = list(enumerate(layer_indices))   # [(pos, idx), ...]
    else:
        target_idx = int(args.layer)
        if target_idx not in layer_indices:
            raise ValueError(f"Layer {target_idx} not in captured layers {layer_indices}")
        pos = layer_indices.index(target_idx)
        targets = [(pos, target_idx)]

    os.makedirs(args.out_dir, exist_ok=True)

    # Load all shards ONCE into a [N, L, H] float16 tensor, then slice per layer.
    # Avoids re-reading every shard for each layer (5x I/O savings).
    print(f"[SAE] Loading all activations from {args.acts_dir} ...")
    all_acts = load_all_activations(args.acts_dir)
    print(f"[SAE]   all_acts shape: {tuple(all_acts.shape)}  dtype: {all_acts.dtype}  "
          f"size: {all_acts.numel() * all_acts.element_size() / 1e9:.2f} GB")

    all_logs = []
    for layer_pos, layer_idx in targets:
        X = slice_layer(all_acts, layer_pos)
        print(f"\n[SAE] layer {layer_idx} (pos {layer_pos}) | X shape: {X.shape}  dtype: {X.dtype}")

        layer_out = os.path.join(args.out_dir, f"layer_{layer_idx:02d}")

        # Resume support: look for an existing checkpoint at --resume-from/layer_NN/final.pt.
        resume_ckpt = None
        if args.resume_from is not None:
            cand = os.path.join(args.resume_from, f"layer_{layer_idx:02d}", "final.pt")
            if os.path.exists(cand):
                ck = torch.load(cand, map_location="cpu", weights_only=False)
                prev_epoch = int(ck.get("epoch", -1))
                if prev_epoch + 1 >= args.epochs:
                    print(f"[SAE] layer {layer_idx}: prev epoch {prev_epoch} >= target "
                          f"{args.epochs - 1}, copying checkpoint without further training")
                    os.makedirs(layer_out, exist_ok=True)
                    torch.save(ck, os.path.join(layer_out, "final.pt"))
                    del X
                    torch.cuda.empty_cache()
                    continue
                resume_ckpt = cand
            else:
                print(f"[SAE] layer {layer_idx}: no checkpoint at {cand}; training from scratch")

        log = train_one_layer(
            X=X,
            layer_idx=layer_idx,
            layer_pos=layer_pos,
            out_dir=layer_out,
            cfg=cfg,
            device=args.device,
            resume_ckpt=resume_ckpt,
        )
        all_logs.append(log)

        # Free memory before next layer
        del X
        torch.cuda.empty_cache()

    del all_acts

    # Write summary
    summary_path = os.path.join(args.out_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump({
            "manifest": manifest,
            "cfg": cfg,
            "layers": [{"layer_idx": l["layer_idx"],
                        "final_recon": l["epochs"][-1]["recon_loss"],
                        "final_dead_frac": l["epochs"][-1]["dead_frac"]}
                       for l in all_logs]
        }, f, indent=2)
    print(f"\n[SAE] All done. Summary: {summary_path}")


if __name__ == "__main__":
    main()