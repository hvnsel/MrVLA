"""Run trained SAEs over collected activations and compute per-feature generality metrics.

For every captured layer:
  1. Load layer_NN/final.pt and reconstruct the TopK SAE forward.
  2. Encode every timestep -> sparse code z of shape [N, n_features].
  3. Compute per-feature metrics across episodes:
       - coverage          : fraction of episodes where the feature fires at least once
       - mean_onsets       : mean # of distinct activation bursts per episode (over episodes
                             where it fired at all)
       - mean_run_length   : mean burst length (total active steps / total bursts)
       - fire_count        : total number of bursts across the dataset
  4. Save z + metadata + metrics to layer_NN.npz, and a summary.json.

Usage
-----
python extract_codes_and_metrics.py \\
    --acts-dir E:/libero_goal_demos/libero_goal_demos \\
    --sae-dir  ./checkpoints/sae_libero_goal_500 \\
    --out-dir  ./codes/sae_libero_goal_500

GPU is used when available; falls back to CPU automatically.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import time

import numpy as np
import torch


# ---------------------------------------------------------------------------
# SAE forward (matches train_sae.py exactly)
# ---------------------------------------------------------------------------
@torch.no_grad()
def sae_forward(
    W_enc: torch.Tensor,
    b_pre: torch.Tensor,
    k: int,
    X: np.ndarray,
    batch_size: int,
    device: str,
) -> np.ndarray:
    """Encode X [N, d] -> sparse codes z [N, n_features] (dense, float32).

    Exactly mirrors TopKSAE.forward() up to the sparse code (we don't need x_hat):
        x_norm = (x - b_pre - mu) / ||x - b_pre - mu||_2
        z      = ReLU(TopK(x_norm @ W_enc))
    """
    N, d = X.shape
    n_features = W_enc.shape[1]
    out = np.empty((N, n_features), dtype=np.float32)

    W_enc_d = W_enc.to(device)
    b_pre_d = b_pre.to(device)

    n_batches = (N + batch_size - 1) // batch_size
    for bi in range(n_batches):
        i = bi * batch_size
        j = min(i + batch_size, N)
        xb = torch.from_numpy(X[i:j]).to(device, dtype=torch.float32)

        x_shifted = xb - b_pre_d                          # [b, d]
        mu = x_shifted.mean(dim=1, keepdim=True)          # [b, 1]
        x_centered = x_shifted - mu
        l2 = x_centered.norm(dim=1, keepdim=True).clamp(min=1e-8)
        x_norm = x_centered / l2

        pre_act = x_norm @ W_enc_d                        # [b, F]
        topk_vals, topk_idx = pre_act.topk(k, dim=1)
        z = torch.zeros_like(pre_act)
        z.scatter_(1, topk_idx, torch.relu(topk_vals))    # [b, F]

        out[i:j] = z.detach().cpu().numpy()

        if bi % 5 == 0 or bi == n_batches - 1:
            print(f"    encode batch {bi+1}/{n_batches}  ({j}/{N} samples)", flush=True)

    return out


# ---------------------------------------------------------------------------
# Shard loading
# ---------------------------------------------------------------------------
def load_shards(acts_dir: str):
    paths = sorted(glob.glob(os.path.join(acts_dir, "shard_*.npz")))
    if not paths:
        raise FileNotFoundError(f"No shards in {acts_dir!r}")
    print(f"[codes] found {len(paths)} shard files", flush=True)

    acts, episode, timestep, task_id = [], [], [], []
    for p in paths:
        d = np.load(p)
        acts.append(d["acts"])                # [n, L, H] float16
        episode.append(d["episode"])
        timestep.append(d["timestep"])
        task_id.append(d["task_id"])
        print(f"  {os.path.basename(p)}: N={len(d['episode']):>6}  "
              f"acts={d['acts'].shape}", flush=True)

    return (
        np.concatenate(acts, axis=0),
        np.concatenate(episode, axis=0),
        np.concatenate(timestep, axis=0),
        np.concatenate(task_id, axis=0),
    )


# ---------------------------------------------------------------------------
# Per-feature generality metrics
# ---------------------------------------------------------------------------
def compute_metrics(z: np.ndarray, episode: np.ndarray, timestep: np.ndarray) -> dict:
    """For each feature, compute coverage, mean onsets, mean run length, total bursts.

    Notes on the definitions
    ------------------------
    * A *burst* (a.k.a. onset) is a contiguous run of timesteps where the feature is
      active (z > 0) within a single episode. We pad each episode with a zero row at
      both ends so that activity starting at t=0 or ending at the last step still
      counts as a complete burst.
    * ``coverage[f]``        = |{episodes where f fires at least once}| / |episodes|
    * ``mean_onsets[f]``     = mean over *those* episodes of (# bursts for f in episode)
    * ``mean_run_length[f]`` = total active timesteps for f / total bursts for f
    """
    active = (z > 0)                                              # [N, F] bool
    F = active.shape[1]

    unique_eps = np.unique(episode)
    n_episodes = len(unique_eps)
    print(f"[metrics] {n_episodes} unique episodes, F={F} features", flush=True)

    onsets_ef = np.zeros((n_episodes, F), dtype=np.int32)         # [E, F]
    active_steps_ef = np.zeros((n_episodes, F), dtype=np.int32)   # [E, F]

    t0 = time.time()
    for e_idx, ep_id in enumerate(unique_eps):
        mask = (episode == ep_id)
        order = np.argsort(timestep[mask])
        active_ep = active[mask][order]                            # [T_ep, F]
        T_ep = active_ep.shape[0]

        # Pad with False rows at both ends so onsets at t=0 / endings at T_ep-1 count.
        padded = np.vstack([
            np.zeros((1, F), dtype=bool),
            active_ep,
            np.zeros((1, F), dtype=bool),
        ])
        diff = padded[1:].astype(np.int8) - padded[:-1].astype(np.int8)  # [T_ep+1, F]
        onsets_ef[e_idx] = (diff == 1).sum(axis=0)                  # # of 0->1 transitions
        active_steps_ef[e_idx] = active_ep.sum(axis=0)

        if e_idx % 50 == 0 or e_idx == n_episodes - 1:
            dt = time.time() - t0
            print(f"    metrics ep {e_idx+1}/{n_episodes}  "
                  f"(T_ep={T_ep})  {dt:.1f}s elapsed", flush=True)

    fired_in_ep = onsets_ef > 0                                    # [E, F]
    n_fired = fired_in_ep.sum(axis=0)                              # [F]

    coverage = n_fired.astype(np.float64) / n_episodes             # [F]

    # mean onsets over episodes where it fired (avoid /0)
    onset_sum_when_fired = (onsets_ef * fired_in_ep).sum(axis=0)   # [F]
    mean_onsets = np.where(
        n_fired > 0,
        onset_sum_when_fired.astype(np.float64) / np.maximum(n_fired, 1),
        0.0,
    )

    # mean run length = total active steps / total bursts (across all episodes)
    total_active = active_steps_ef.sum(axis=0)                     # [F]
    total_bursts = onsets_ef.sum(axis=0)                           # [F]
    mean_run_length = np.where(
        total_bursts > 0,
        total_active.astype(np.float64) / np.maximum(total_bursts, 1),
        0.0,
    )

    return {
        "coverage": coverage.astype(np.float32),
        "mean_onsets": mean_onsets.astype(np.float32),
        "mean_run_length": mean_run_length.astype(np.float32),
        "fire_count": total_bursts.astype(np.int64),
        "n_episodes": int(n_episodes),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--acts-dir", required=True,
                   help="Directory produced by collect_activations.py")
    p.add_argument("--sae-dir", required=True,
                   help="Root directory of trained SAEs (expects layer_NN/final.pt).")
    p.add_argument("--out-dir", required=True,
                   help="Where to write layer_NN.npz + summary.json")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--batch-size", type=int, default=4096)
    args = p.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"[codes] device={device}  torch={torch.__version__}", flush=True)

    manifest_path = os.path.join(args.acts_dir, "manifest.json")
    with open(manifest_path) as f:
        manifest = json.load(f)
    layer_indices = manifest["layer_indices"]
    print(f"[codes] manifest: model={manifest['model_name']}", flush=True)
    print(f"[codes]           layers={layer_indices}  hidden={manifest['hidden_dim']}", flush=True)
    print(f"[codes]           total_samples={manifest['total_samples']}", flush=True)

    print("\n[codes] loading shards...", flush=True)
    acts, episode, timestep, task_id = load_shards(args.acts_dir)
    print(f"[codes]   acts {acts.shape} dtype={acts.dtype}", flush=True)
    print(f"[codes]   N={len(episode)}  episodes={len(np.unique(episode))}  "
          f"tasks={len(np.unique(task_id))}", flush=True)

    os.makedirs(args.out_dir, exist_ok=True)
    summary: dict = {"manifest": manifest, "layers": {}}

    for layer_pos, layer_idx in enumerate(layer_indices):
        print(f"\n[codes] ====== layer {layer_idx} (pos {layer_pos}) ======", flush=True)

        ckpt_path = os.path.join(args.sae_dir, f"layer_{layer_idx:02d}", "final.pt")
        if not os.path.exists(ckpt_path):
            # fall back to interim checkpoint if final.pt isn't there
            alt = os.path.join(args.sae_dir, f"layer_{layer_idx:02d}", "checkpoint.pt")
            if os.path.exists(alt):
                ckpt_path = alt
            else:
                raise FileNotFoundError(f"No SAE checkpoint at {ckpt_path} or {alt}")
        print(f"[codes] loading SAE: {ckpt_path}", flush=True)

        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        W_enc = ckpt["W_enc"]                                       # [d, F]
        b_pre = ckpt["b_pre"]                                       # [d]
        k = int(ckpt["config"]["k"])
        F = W_enc.shape[1]
        print(f"[codes]   d={W_enc.shape[0]}  F={F}  k={k}  "
              f"(expected active frac ~ {k/F:.4f})", flush=True)

        X = acts[:, layer_pos, :].astype(np.float32, copy=False)    # [N, d] float32 view
        print(f"[codes] running SAE forward on {X.shape}...", flush=True)
        t0 = time.time()
        z = sae_forward(W_enc, b_pre, k, X, batch_size=args.batch_size, device=device)
        print(f"[codes]   forward done in {time.time()-t0:.1f}s", flush=True)

        active_frac = float((z > 0).mean())
        print(f"[codes]   actual active frac: {active_frac:.4f}", flush=True)

        print(f"[codes] computing metrics...", flush=True)
        m = compute_metrics(z, episode, timestep)

        # Save per-layer artefact (sparse codes stored as float16 to halve disk size).
        out_path = os.path.join(args.out_dir, f"layer_{layer_idx:02d}.npz")
        np.savez_compressed(
            out_path,
            z=z.astype(np.float16),
            episode=episode.astype(np.int32),
            timestep=timestep.astype(np.int32),
            task_id=task_id.astype(np.int32),
            coverage=m["coverage"],
            mean_onsets=m["mean_onsets"],
            mean_run_length=m["mean_run_length"],
            fire_count=m["fire_count"],
        )
        size_mb = os.path.getsize(out_path) / 1e6
        print(f"[codes]   saved {out_path}  ({size_mb:.1f} MB)", flush=True)

        n_dead = int((m["fire_count"] == 0).sum())
        print(f"[codes]   dead features: {n_dead}/{F} ({100*n_dead/F:.1f}%)", flush=True)
        print(f"[codes]   coverage   mean={m['coverage'].mean():.4f}  "
              f"max={m['coverage'].max():.4f}", flush=True)
        print(f"[codes]   onsets/ep  mean={m['mean_onsets'].mean():.3f}  "
              f"max={m['mean_onsets'].max():.3f}", flush=True)
        print(f"[codes]   run_length mean={m['mean_run_length'].mean():.3f}  "
              f"max={m['mean_run_length'].max():.3f}", flush=True)

        summary["layers"][f"layer_{layer_idx:02d}"] = {
            "n_features": int(F),
            "k": int(k),
            "active_frac": active_frac,
            "n_episodes": m["n_episodes"],
            "n_dead_features": n_dead,
            "coverage_mean": float(m["coverage"].mean()),
            "coverage_max": float(m["coverage"].max()),
            "mean_onsets_mean": float(m["mean_onsets"].mean()),
            "mean_run_length_mean": float(m["mean_run_length"].mean()),
        }

    out_summary = os.path.join(args.out_dir, "summary.json")
    with open(out_summary, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[codes] done. summary -> {out_summary}", flush=True)


if __name__ == "__main__":
    main()
