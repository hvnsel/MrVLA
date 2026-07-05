"""LoRA fine-tuning with ablative steering and episode-level contrastive ratio loss.

Three training signals run jointly:

  BC loss       — cross-entropy on the 7 expert action tokens (standard BC).

  Ablation      — probabilistically removes memorized SAE feature directions from
                  layer 31's residual stream before the LM head.  Forces the model
                  to predict the correct action without those directions available.
                  Whether it reroutes toward *general* features or finds new
                  memorized directions is an open empirical question — answered by
                  the post-training re-measurement step.

  Contrastive   — episode-level ratio:
                      L_c = -log( A / (A + B + ε) )
                  where A = Σ_t Σ_j P(general_j) · z²_j,t  (general energy)
                        B = Σ_t Σ_j P(memorized_j) · z²_j,t (memorized energy)
                  Gradient w.r.t. z_g,t < 0 → minimizing L_c increases z_g,t.
                  Gradient w.r.t. z_m,t > 0 → minimizing L_c decreases z_m,t.
                  Flows back through the frozen SAE encoder to the LoRA weights.
                  Uses soft P(general) weights (no hard threshold) so classifier
                  uncertainty is propagated rather than discarded.

  The BC loss acts as an implicit burstiness guard: firing a general feature at
  the wrong timestep hurts action accuracy, so the joint optimum is "fire when
  useful, not always."

Architecture
------------
  - LoRA applied to q_proj, k_proj, v_proj, o_proj across all decoder layers.
  - Single forward hook on layers[31]:
      1. Mean-pool h over prompt positions (matches ActivationCollector's pooling,
         which is what the SAE was trained on) → SAE encode (differentiable;
         W_enc frozen) → accumulate A, B for the contrastive loss
      2. Apply ablation to h (grad-mode projection) → h_ablated (returned to model)
  - LM head sees h_ablated → BC loss.
  - Contrastive loss built from A, B (from hook storage) after forward pass.
  - total_loss = BC_loss + contrastive_weight * contrastive_loss

Post-training verification
--------------------------
  Re-run collect_activations.py, extract_codes_and_metrics.py, and
  generality_classifier.py on the LoRA checkpoint to check whether
  general-feature activity increased.  The per-step A/(A+B) ratio logged
  during training is a proxy but NOT a substitute for the full measurement.

Usage
-----
  python mrvla/train_lora.py \\
      --model       openvla/openvla-7b-finetuned-libero-goal \\
      --task-suite  libero_goal \\
      --unnorm-key  libero_goal \\
      --sae-dir     ./checkpoints/sae_libero_goal_v3 \\
      --gen-dir     E:/libero_goal_demos/generality_v3 \\
      --out-dir     ./lora_checkpoints/run_001 \\
      [--ablation-layer 31] \\
      [--lora-rank 32] [--lora-alpha 16] \\
      [--ablate-frac 0.5] [--contrastive-weight 0.1] \\
      [--epochs 3] [--batch-size 4] [--lr 2e-4] \\
      [--no-flash-attn]
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import glob
import h5py
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from .libero_collect import (
    _center_crop,
    _resize_libero_image,
)
from .model_utils import PROMPT_TEMPLATE, locate_decoder_layers, load_openvla


# ---------------------------------------------------------------------------
# Action tokenization helpers
# ---------------------------------------------------------------------------

def _get_norm_stats(model, unnorm_key: str) -> dict:
    """Return {q01, q99} normalization arrays for the given key.

    Tries model.config.norm_stats first (standard OpenVLA), then falls back to
    model.norm_stats.  Raises clearly if neither is present.
    """
    for attr in ("config", "norm_stats"):
        cfg = getattr(model, attr, None)
        if cfg is None:
            continue
        stats = (
            cfg.norm_stats.get(unnorm_key)
            if hasattr(cfg, "norm_stats")
            else cfg.get(unnorm_key)
        )
        if stats is not None:
            # norm_stats[key] = {"action": {q01, q99, mask, ...}, "proprio": ...}
            return stats["action"] if "action" in stats else stats
    raise AttributeError(
        f"Cannot find norm_stats[{unnorm_key!r}] on the model.  "
        f"Make sure you loaded the fine-tuned checkpoint, not the base openvla-7b."
    )


def _action_to_token_ids(
    action: np.ndarray,
    norm_stats: dict,
    vocab_size: int,
    n_bins: int = 256,
) -> list[int]:
    """Normalise a continuous action and convert it to OpenVLA action token IDs.

    Faithful to OpenVLA's ActionTokenizer + RLDS normalization:
      - dims where norm_stats['mask'] is True: normalize to [-1, 1] via q01/q99
      - masked-out dims (gripper): kept raw (already in [0, 1])
      - digitize over np.linspace(-1, 1, n_bins)  → bin index in [1, n_bins]
      - token_id = vocab_size - bin_index          (NB: reversed order!)

    This must exactly mirror predict_action()'s decode:
        discretized = vocab_size - token_id
        action      = bin_centers[discretized - 1]
    """
    q01  = np.asarray(norm_stats["q01"], dtype=np.float32)
    q99  = np.asarray(norm_stats["q99"], dtype=np.float32)
    mask = np.asarray(norm_stats.get("mask", np.ones_like(q01)), dtype=bool)
    span = np.maximum(q99 - q01, 1e-8)

    normalized = 2.0 * (action.astype(np.float32) - q01) / span - 1.0
    normalized = np.where(mask, np.clip(normalized, -1.0, 1.0), action.astype(np.float32))

    bins = np.linspace(-1.0, 1.0, n_bins)
    disc = np.clip(np.digitize(normalized, bins), 1, n_bins)   # [1, n_bins]
    return (int(vocab_size) - disc).tolist()


def preprocess_demo_action(action: np.ndarray) -> np.ndarray:
    """Convert a raw LIBERO demo action to OpenVLA's training-target convention.

    This is the INVERSE of the eval-time postprocessing in libero_collect.py
    (_normalize_gripper_action + _invert_gripper_action).  Eval maps model
    output g_m ∈ [0,1] → env action g_env = -sign(2·g_m - 1) ∈ {-1,+1}.
    Inverting: g_m = (1 - sign(g_env)) / 2, i.e. env +1 → 0, env -1 → 1.
    The pose dims (0-5) pass through unchanged (normalized later via q01/q99).
    """
    action = action.copy().astype(np.float32)
    action[..., -1] = (1.0 - np.sign(action[..., -1])) / 2.0
    return action


# ---------------------------------------------------------------------------
# LIBERO demo dataset
# ---------------------------------------------------------------------------

class LiberoActionDataset(Dataset):
    """Lazily loads (image_pil, instruction, action_token_ids) from LIBERO HDF5s.

    All HDF5 files matching ``<demos_dir>/<suite>/*_demo.hdf5`` are included.
    Images and actions are read from disk on each __getitem__ call.

    Parameters
    ----------
    demos_dir         : path to the LIBERO dataset root (contains <suite>/<task>_demo.hdf5)
    suite             : e.g. "libero_goal"
    model             : loaded OpenVLA model (for norm_stats lookup)
    processor         : loaded OpenVLA processor (for tokenizer)
    unnorm_key        : normalization key, e.g. "libero_goal"
    center_crop       : whether to apply the 90%-area centre crop
    max_demos_per_file: cap on demos loaded per HDF5 (None = all)
    max_steps_per_demo: cap on timesteps per demo (None = all)
    """

    def __init__(
        self,
        demos_dir: str,
        suite: str,
        model,
        processor,
        unnorm_key: str,
        center_crop: bool = True,
        max_demos_per_file: int | None = None,
        max_steps_per_demo: int | None = None,
    ):
        self.center_crop = center_crop
        self.processor   = processor

        # Gather all HDF5 paths
        pattern = os.path.join(demos_dir, suite, "*_demo.hdf5")
        hdf5_paths = sorted(glob.glob(pattern))
        if not hdf5_paths:
            raise FileNotFoundError(
                f"No *_demo.hdf5 files found under {os.path.join(demos_dir, suite)!r}"
            )

        # Normalization
        norm_stats  = _get_norm_stats(model, unnorm_key)
        vocab_size  = processor.tokenizer.vocab_size

        # Build flat index: list of (hdf5_path, task_description, demo_key, step_idx)
        self._index: list[tuple[str, str, str, int]] = []
        # Keep a map from hdf5_path → task description (embedded in file attrs or filename)
        self._task_desc: dict[str, str] = {}
        self._norm_stats  = norm_stats
        self._vocab_size  = vocab_size

        for hdf5_path in hdf5_paths:
            with h5py.File(hdf5_path, "r") as f:
                # Language instruction lives in the data group's problem_info JSON
                task_desc = ""
                raw_info = f["data"].attrs.get("problem_info", "")
                if isinstance(raw_info, bytes):
                    raw_info = raw_info.decode()
                if raw_info:
                    try:
                        task_desc = json.loads(raw_info).get("language_instruction", "")
                        task_desc = task_desc.strip().strip('"')
                    except (json.JSONDecodeError, TypeError):
                        task_desc = ""
                # Fallback: derive from filename
                if not task_desc:
                    stem = Path(hdf5_path).stem.replace("_demo", "")
                    task_desc = stem.replace("_", " ").lower()
                self._task_desc[hdf5_path] = task_desc

                demo_keys = sorted(f["data"].keys(), key=lambda k: int(k.split("_")[-1]))
                if max_demos_per_file is not None:
                    demo_keys = demo_keys[:max_demos_per_file]

                for demo_key in demo_keys:
                    # Infer number of steps
                    actions = f["data"][demo_key]["actions"]
                    n_steps = actions.shape[0]
                    if max_steps_per_demo is not None:
                        n_steps = min(n_steps, max_steps_per_demo)
                    for t in range(n_steps):
                        self._index.append((hdf5_path, task_desc, demo_key, t))

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int):
        hdf5_path, task_desc, demo_key, t = self._index[idx]

        with h5py.File(hdf5_path, "r") as f:
            demo = f["data"][demo_key]

            # Image
            obs = demo["obs"]
            for img_key in ("agentview_rgb", "agentview_image"):
                if img_key in obs:
                    frame = np.asarray(obs[img_key][t])
                    break
            else:
                raise KeyError(f"No agentview image key in {hdf5_path}:{demo_key}/obs")

            # Action
            action_raw = np.asarray(demo["actions"][t], dtype=np.float32)

        # Image pre-processing (matches closed-loop eval pipeline)
        img = frame[::-1, ::-1]                             # 180° flip
        pil = Image.fromarray(np.ascontiguousarray(img))
        pil = _resize_libero_image(pil, size=224)
        if self.center_crop:
            pil = _center_crop(pil, crop_scale=0.9)

        # Action pre-processing + tokenization
        action = preprocess_demo_action(action_raw)
        action_token_ids = _action_to_token_ids(
            action, self._norm_stats, self._vocab_size
        )

        prompt = PROMPT_TEMPLATE.format(
            instruction=task_desc.lower().strip().rstrip(".")
        )

        return pil, prompt, action_token_ids


# Llama SentencePiece "empty" token; predict_action() appends it to the prompt
# before generating action tokens, so training sequences must include it too.
_EMPTY_TOKEN = 29871


def _collate_fn(batch, processor, device):
    """Collate a list of (pil, prompt, action_token_ids) into model inputs + labels.

    Builds the full teacher-forcing sequence per sample:
        [BOS, prompt tokens..., 29871, act_1..act_7, EOS, pad...]
    RIGHT padding is mandatory: OpenVLA's multimodal fusion inserts the 256
    image-patch embeddings immediately after the BOS token, which must sit at
    position 0 of every row.

    Labels are -100 everywhere except the 7 action tokens and the EOS
    (matching OpenVLA's official finetune.py supervision).
    """
    pils, prompts, token_id_lists = zip(*batch)
    tok        = processor.tokenizer
    eos_id     = tok.eos_token_id
    pad_id     = tok.pad_token_id if tok.pad_token_id is not None else 0
    action_len = 7

    processed = [processor(p, img) for p, img in zip(prompts, pils)]

    seqs, labels_list = [], []
    for x, tok_ids in zip(processed, token_id_lists):
        base_ids = x["input_ids"][0].to(torch.long)              # [S], starts with BOS
        act      = torch.tensor(tok_ids, dtype=torch.long)
        full_ids = torch.cat([
            base_ids,
            torch.tensor([_EMPTY_TOKEN], dtype=torch.long),
            act,
            torch.tensor([eos_id], dtype=torch.long),
        ])
        lbl = torch.full_like(full_ids, -100)
        lbl[-(action_len + 1):] = full_ids[-(action_len + 1):]   # actions + EOS
        seqs.append(full_ids)
        labels_list.append(lbl)

    max_len   = max(s.shape[0] for s in seqs)
    B         = len(seqs)
    input_ids = torch.full((B, max_len), pad_id, dtype=torch.long)
    labels    = torch.full((B, max_len), -100,  dtype=torch.long)
    attn      = torch.zeros((B, max_len), dtype=torch.long)
    for i, (s, l) in enumerate(zip(seqs, labels_list)):
        S = s.shape[0]
        input_ids[i, :S] = s        # RIGHT padding: real tokens first, pads last
        labels[i, :S]    = l
        attn[i, :S]      = 1

    pixel_values = torch.stack(
        [x["pixel_values"][0] for x in processed]
    ).to(device, dtype=torch.bfloat16)

    return {
        "input_ids":      input_ids.to(device),
        "pixel_values":   pixel_values,
        "labels":         labels.to(device),
        "attention_mask": attn.to(device),
    }


# ---------------------------------------------------------------------------
# Combined ablation + contrastive forward hook
# ---------------------------------------------------------------------------

class TrainingHook:
    """Forward hook on a single decoder layer that does two things per pass:

    1. Mean-pools the layer output over the prompt positions (BOS + image
       patches + instruction + empty token — the same pooling the SAE was
       trained on via ActivationCollector), then runs the SAE encoder (frozen
       W_enc) to produce one sparse code z per sample.  Accumulates
       soft-weighted general energy A and memorized energy B for the
       contrastive loss.  The SAE computation stays in the computation graph
       (W_enc frozen, not detached) so gradients flow back through
       z → pooled h → LoRA weights.

       NB: with shuffled per-timestep sampling the ratio is aggregated over
       the mini-batch rather than a literal episode; the key property is
       preserved either way — ∂L/∂z_g,t = 0 wherever z_g,t = 0, so quiet
       timesteps receive no gradient pressure.

    2. Applies ablation: probabilistically projects out memorized decoder
       directions from h (all token positions), returning h_ablated to the
       rest of the model so the LM head (and BC loss) sees an ablated
       representation.  The projection is done IN grad mode — only the
       Bernoulli mask sampling is no-grad — so BC gradients flow through
       (I - VᵀV) back to all upstream LoRA weights.

    The SAE codes are computed from h *before* ablation so that A and B are
    meaningful (ablating first would trivially zero B, defeating the contrastive
    term).

    Usage in training loop
    ----------------------
        hook = TrainingHook(...)
        with hook:
            hook.zero()
            hook.set_pool_mask((batch["attention_mask"] == 1) & (batch["labels"] == -100))
            outputs = model(**inputs)          # forward pass triggers hook
            bc_loss  = outputs.loss
            c_loss   = hook.contrastive_loss() # uses accumulated A, B
            loss     = bc_loss + w * c_loss
            loss.backward()
    """

    def __init__(
        self,
        layer: nn.Module,
        W_enc: torch.Tensor,       # [d, F]  — frozen SAE encoder weights
        b_pre: torch.Tensor,       # [d]     — SAE pre-bias (geometric median)
        sae_k: int,                # TopK k
        prob_general: torch.Tensor,  # [F]   — P(general) per feature
        W_dec: torch.Tensor,       # [F, d]  — SAE decoder directions (for ablation)
        ablate_frac: float = 0.5,
        contrastive_eps: float = 1e-8,
        num_patches: int = 256,    # image-patch embeddings OpenVLA inserts after BOS
    ):
        # Freeze SAE weights (defensive — they should already not be parameters)
        self._W_enc = W_enc.float().detach().requires_grad_(False)
        self._b_pre = b_pre.float().detach().requires_grad_(False)
        self._k     = sae_k

        self._p_gen = prob_general.float().detach().requires_grad_(False)  # [F]
        self._p_mem = (1.0 - prob_general).float().detach().requires_grad_(False)

        # Unit-norm decoder directions for ablation
        norms = W_dec.norm(dim=1, keepdim=True).clamp(min=1e-8)
        self._V_full = (W_dec / norms).float().detach().requires_grad_(False)  # [F, d]

        self._ablate_frac     = float(ablate_frac)
        self._contrastive_eps = float(contrastive_eps)
        self._num_patches     = int(num_patches)

        # Set per-batch by the training loop; text-token mask [B, S_text]
        self._pool_mask_text: torch.Tensor | None = None

        # Storage populated by hook, consumed by training loop
        self._A: torch.Tensor | None = None  # general energy  (has grad)
        self._B: torch.Tensor | None = None  # memorized energy (has grad)

        self._handle = layer.register_forward_hook(self._hook)

    def set_pool_mask(self, mask_text: torch.Tensor) -> None:
        """Set the per-batch pooling mask over TEXT token positions [B, S_text].

        True = include in the pooled SAE input.  Pass
        ``(attention_mask == 1) & (labels == -100)`` — that selects BOS, the
        instruction prompt, and the empty token while excluding the supervised
        action tokens, EOS, and padding.  The hook expands the mask internally
        to account for the image-patch embeddings inserted after BOS.
        """
        self._pool_mask_text = mask_text.bool()

    def _hook(self, _module, _inputs, output):
        h = output[0] if isinstance(output, tuple) else output
        # h: [B, S, d] — retains grad from LoRA weights
        B_size, S, d = h.shape
        device = h.device

        W_enc = self._W_enc.to(device)
        b_pre = self._b_pre.to(device)
        p_gen = self._p_gen.to(device)
        p_mem = self._p_mem.to(device)

        # ── 1. Build the pooling mask over the multimodal sequence ─────────
        # OpenVLA inserts num_patches patch embeddings after BOS:
        #   multimodal seq = [BOS, patch_1..patch_P, text_2..text_S]
        if (
            self._pool_mask_text is not None
            and S == self._pool_mask_text.shape[1] + self._num_patches
        ):
            m = self._pool_mask_text.to(device)
            pool_mask = torch.cat(
                [
                    m[:, :1],                                                # BOS
                    torch.ones(B_size, self._num_patches,
                               dtype=torch.bool, device=device),            # patches
                    m[:, 1:],                                                # rest of text
                ],
                dim=1,
            )
        else:
            pool_mask = torch.ones(B_size, S, dtype=torch.bool, device=device)

        # ── 2. Mean-pool then SAE encode (differentiable through h) ────────
        # Pooling first matches the SAE's training distribution: it was trained
        # on per-step token-mean-pooled activations (see ActivationCollector).
        w      = pool_mask.float().unsqueeze(-1)                     # [B, S, 1]
        pooled = (h.float() * w).sum(dim=1) / w.sum(dim=1).clamp(min=1.0)  # [B, d]

        x       = pooled - b_pre                                     # [B, d]
        x_norm  = x / x.norm(dim=-1, keepdim=True).clamp(min=1e-8)   # [B, d]
        pre_act = x_norm @ W_enc                                     # [B, F]

        # TopK selection is discrete; gradients flow through the selected values
        topk_vals, topk_idx = pre_act.topk(self._k, dim=-1)
        z = torch.zeros_like(pre_act)
        z.scatter_(-1, topk_idx, topk_vals.clamp(min=0.0))           # [B, F]

        # Soft-weighted energies — aggregated over the batch
        z2      = z ** 2                                             # [B, F]
        self._A = (z2 * p_gen).sum()                                 # scalar, has grad
        self._B = (z2 * p_mem).sum()                                 # scalar, has grad

        # ── 3. Ablation ── mask sampling no-grad, projection IN grad mode ──
        # The projection must stay in the graph: BC gradients flow through
        # (I - VᵀV) back to upstream LoRA weights.  Wrapping the projection in
        # no_grad would detach the graph at this layer and silently disable BC
        # training for everything upstream.
        with torch.no_grad():
            probs = p_mem * self._ablate_frac
            mask  = torch.bernoulli(probs).bool()                    # [F]

        if mask.any():
            V         = self._V_full.to(device)[mask]                # [K, d]
            h_f       = h.float()
            coeffs    = h_f @ V.T                                    # [B, S, K]
            h_ablated = (h_f - coeffs @ V).to(h.dtype)               # [B, S, d]
        else:
            h_ablated = h

        if isinstance(output, tuple):
            return (h_ablated,) + output[1:]
        return h_ablated

    def contrastive_loss(self) -> torch.Tensor:
        """Compute -log(A / (A + B + ε)).  Call after the forward pass."""
        if self._A is None or self._B is None:
            raise RuntimeError("contrastive_loss() called before a forward pass.")
        A, B = self._A, self._B
        return -torch.log(A / (A + B + self._contrastive_eps))

    def zero(self) -> None:
        """Reset accumulated energies.  Call at the start of each training step."""
        self._A = None
        self._B = None

    def remove(self) -> None:
        self._handle.remove()

    def __enter__(self) -> "TrainingHook":
        return self

    def __exit__(self, *exc) -> None:
        self.remove()


# ---------------------------------------------------------------------------
# SAE / generality loaders
# ---------------------------------------------------------------------------

def load_sae_weights(sae_dir: str, layer_idx: int) -> dict:
    """Load W_enc, W_dec, b_pre, and k from final.pt for the given layer."""
    pt = os.path.join(sae_dir, f"layer_{layer_idx:02d}", "final.pt")
    if not os.path.exists(pt):
        raise FileNotFoundError(f"SAE checkpoint not found: {pt}")
    ckpt = torch.load(pt, map_location="cpu", weights_only=True)
    return {
        "W_enc": ckpt["W_enc"].float(),    # [d, F]
        "W_dec": ckpt["W_dec"].float(),    # [F, d]
        "b_pre": ckpt["b_pre"].float(),    # [d]
        "k":     int(ckpt["config"]["k"]),
    }


def load_generality_probs(gen_dir: str, layer_idx: int) -> torch.Tensor:
    """Load P(general) per feature for the given layer.  Returns [F] float32 tensor."""
    path = os.path.join(gen_dir, f"layer_{layer_idx:02d}_generality.npz")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Generality file not found: {path}")
    return torch.from_numpy(np.load(path)["prob_general"].astype(np.float32))


# ---------------------------------------------------------------------------
# LoRA setup
# ---------------------------------------------------------------------------

def apply_lora(model, rank: int = 32, alpha: int = 16):
    """Wrap model with LoRA adapters on all attention projections.

    Returns a peft.PeftModel wrapping the original model.
    """
    try:
        from peft import LoraConfig, get_peft_model, TaskType
    except ImportError as _e:
        raise ImportError(
            f"peft import failed: {_e}\n"
            "Make sure peft is installed in the active Python environment."
        ) from _e

    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=rank,
        lora_alpha=alpha,
        target_modules=target_modules,
        lora_dropout=0.0,
        bias="none",
    )
    model = get_peft_model(model, config)
    model.print_trainable_parameters()
    return model


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(
    model,
    hook: TrainingHook,
    dataloader: DataLoader,
    processor,
    optimizer: torch.optim.Optimizer,
    device: str,
    contrastive_weight: float,
    log_every: int = 20,
    save_every_steps: int | None = None,
    out_dir: str | None = None,
    epoch: int = 0,
) -> dict:
    """Run one epoch.  Returns a dict of mean losses."""
    model.train()
    total_bc = 0.0
    total_c  = 0.0
    total    = 0.0
    n_steps  = 0

    for batch_idx, raw_batch in enumerate(tqdm(dataloader, desc="  train")):
        # Collate is done inside DataLoader via a partial, so raw_batch is already
        # the dict returned by _collate_fn.
        hook.zero()
        # Pool over prompt positions only (BOS + patches + instruction + empty
        # token); excludes supervised action tokens, EOS, and padding.
        hook.set_pool_mask(
            (raw_batch["attention_mask"] == 1) & (raw_batch["labels"] == -100)
        )

        outputs  = model(
            input_ids      = raw_batch["input_ids"],
            pixel_values   = raw_batch["pixel_values"],
            labels         = raw_batch["labels"],
            attention_mask = raw_batch["attention_mask"],
        )
        bc_loss = outputs.loss  # standard cross-entropy on action tokens

        c_loss  = hook.contrastive_loss()
        loss    = bc_loss + contrastive_weight * c_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            (p for p in model.parameters() if p.requires_grad), max_norm=1.0
        )
        optimizer.step()

        total_bc += bc_loss.item()
        total_c  += c_loss.item()
        total    += loss.item()
        n_steps  += 1

        if (batch_idx + 1) % log_every == 0:
            ratio = hook._A.item() / (hook._A.item() + hook._B.item() + 1e-8) \
                    if hook._A is not None else float("nan")
            print(
                f"    step {batch_idx+1:5d}  "
                f"bc={bc_loss.item():.4f}  "
                f"c={c_loss.item():.4f}  "
                f"A/(A+B)={ratio:.4f}"
            )

        if save_every_steps and out_dir and (batch_idx + 1) % save_every_steps == 0:
            ckpt = os.path.join(out_dir, f"step_e{epoch:02d}s{batch_idx+1:06d}")
            model.save_pretrained(ckpt)
            print(f"    [checkpoint] {ckpt}")

    return {
        "bc_loss":          total_bc / max(n_steps, 1),
        "contrastive_loss": total_c  / max(n_steps, 1),
        "total_loss":       total    / max(n_steps, 1),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--model",      required=True,
                   help="HF model ID or local path (fine-tuned checkpoint).")
    p.add_argument("--task-suite", required=True,
                   help="LIBERO task suite name, e.g. libero_goal.")
    p.add_argument("--unnorm-key", required=True,
                   help="Action normalization key, e.g. libero_goal.")
    p.add_argument("--sae-dir",    required=True,
                   help="SAE checkpoint dir (contains layer_NN/final.pt).")
    p.add_argument("--gen-dir",    required=True,
                   help="Generality dir (layer_NN_generality.npz files).")
    p.add_argument("--out-dir",    required=True,
                   help="Where to save LoRA checkpoints and training log.")

    # Data
    p.add_argument("--demos-dir",  default=None,
                   help="LIBERO dataset root (default: LIBERO get_libero_path('datasets')).")
    p.add_argument("--max-demos-per-file", type=int, default=None)
    p.add_argument("--max-steps-per-demo", type=int, default=None)
    p.add_argument("--no-center-crop",     action="store_true")

    # Model
    p.add_argument("--ablation-layer",  type=int, default=31,
                   help="Decoder layer index to hook (default: 31).")
    p.add_argument("--no-flash-attn",   action="store_true")
    p.add_argument("--device",          default="cuda:0")

    # LoRA
    p.add_argument("--lora-rank",  type=int,   default=32)
    p.add_argument("--lora-alpha", type=int,   default=16)

    # Training
    p.add_argument("--epochs",              type=int,   default=3)
    p.add_argument("--batch-size",          type=int,   default=4)
    p.add_argument("--lr",                  type=float, default=2e-4)
    p.add_argument("--contrastive-weight",  type=float, default=0.1,
                   help="λ in total_loss = BC + λ * contrastive.")
    p.add_argument("--ablate-frac",         type=float, default=0.5,
                   help="Scales P(memorized) before Bernoulli sampling. "
                        "0 = no ablation; 1 = ablate with probability = P(memorized).")
    p.add_argument("--log-every",           type=int,   default=20)
    p.add_argument("--save-every-epoch",    action="store_true",
                   help="Save a checkpoint after each epoch in addition to the final.")
    p.add_argument("--save-every-steps",    type=int,   default=None,
                   help="Also save a mid-epoch checkpoint every N optimizer steps.")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # ── Model ──────────────────────────────────────────────────────────────
    print(f"[train_lora] loading {args.model} ...")
    model, processor = load_openvla(
        args.model, device=args.device,
        use_flash_attn=not args.no_flash_attn,
    )

    # ── LoRA ───────────────────────────────────────────────────────────────
    print(f"[train_lora] applying LoRA  rank={args.lora_rank}  alpha={args.lora_alpha}")
    model = apply_lora(model, rank=args.lora_rank, alpha=args.lora_alpha)
    model.train()

    # ── SAE + generality ───────────────────────────────────────────────────
    layer_idx = args.ablation_layer
    print(f"[train_lora] loading SAE for layer {layer_idx} ...")
    sae = load_sae_weights(args.sae_dir, layer_idx)
    prob_general = load_generality_probs(args.gen_dir, layer_idx)
    print(
        f"  features={len(prob_general)}  "
        f"general={int((prob_general >= 0.5).sum())}  "
        f"memorized={int((prob_general < 0.5).sum())}"
    )

    # ── Hook ───────────────────────────────────────────────────────────────
    layers = locate_decoder_layers(model)
    hook = TrainingHook(
        layer        = layers[layer_idx],
        W_enc        = sae["W_enc"],
        b_pre        = sae["b_pre"],
        sae_k        = sae["k"],
        prob_general = prob_general,
        W_dec        = sae["W_dec"],
        ablate_frac  = args.ablate_frac,
    )
    print(
        f"[train_lora] TrainingHook registered on layer {layer_idx}  "
        f"ablate_frac={args.ablate_frac}  contrastive_weight={args.contrastive_weight}"
    )

    # ── Dataset ────────────────────────────────────────────────────────────
    if args.demos_dir is None:
        try:
            from libero.libero import get_libero_path
            args.demos_dir = get_libero_path("datasets")
        except ImportError:
            raise SystemExit(
                "--demos-dir is required when LIBERO is not installed."
            )

    print(f"[train_lora] loading dataset from {args.demos_dir} / {args.task_suite} ...")
    dataset = LiberoActionDataset(
        demos_dir         = args.demos_dir,
        suite             = args.task_suite,
        model             = model,
        processor         = processor,
        unnorm_key        = args.unnorm_key,
        center_crop       = not args.no_center_crop,
        max_demos_per_file= args.max_demos_per_file,
        max_steps_per_demo= args.max_steps_per_demo,
    )
    print(f"  total samples: {len(dataset)}")

    # Collate inline with device + processor captured in closure
    _device    = args.device
    _processor = processor

    def collate(batch):
        return _collate_fn(batch, _processor, _device)

    dataloader = DataLoader(
        dataset,
        batch_size  = args.batch_size,
        shuffle     = True,
        collate_fn  = collate,
        num_workers = 0,   # HDF5 + PIL don't play well with multiprocessing
    )

    # ── Optimiser (LoRA params only) ───────────────────────────────────────
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.0)

    # ── Training ───────────────────────────────────────────────────────────
    log = {
        "args":   vars(args),
        "epochs": [],
    }

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        print(f"\n[train_lora] epoch {epoch}/{args.epochs}")
        metrics = train(
            model              = model,
            hook               = hook,
            dataloader         = dataloader,
            processor          = processor,
            optimizer          = optimizer,
            device             = args.device,
            contrastive_weight = args.contrastive_weight,
            log_every          = args.log_every,
            save_every_steps   = args.save_every_steps,
            out_dir            = args.out_dir,
            epoch              = epoch,
        )
        elapsed = time.time() - t0
        metrics["epoch"]   = epoch
        metrics["elapsed"] = elapsed
        log["epochs"].append(metrics)

        print(
            f"  epoch {epoch} done  "
            f"bc={metrics['bc_loss']:.4f}  "
            f"c={metrics['contrastive_loss']:.4f}  "
            f"total={metrics['total_loss']:.4f}  "
            f"({elapsed:.0f}s)"
        )

        if args.save_every_epoch:
            ckpt_path = os.path.join(args.out_dir, f"epoch_{epoch:02d}")
            model.save_pretrained(ckpt_path)
            print(f"  checkpoint saved → {ckpt_path}")

    # ── Final save ─────────────────────────────────────────────────────────
    final_path = os.path.join(args.out_dir, "final")
    model.save_pretrained(final_path)
    print(f"\n[train_lora] final LoRA adapter saved → {final_path}")

    log_path = os.path.join(args.out_dir, "train_log.json")
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)
    print(f"[train_lora] training log → {log_path}")

    hook.remove()
    print(
        "\n[train_lora] done.  To verify rerouting, re-run:\n"
        "  collect_activations.py --model <lora_checkpoint> ...\n"
        "  extract_codes_and_metrics.py ...\n"
        "  generality_classifier.py ...\n"
        "and compare general-feature coverage and onset counts against the base model."
    )


if __name__ == "__main__":
    main()
