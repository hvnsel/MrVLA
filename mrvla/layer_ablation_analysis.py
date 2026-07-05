"""Rank SAE layers by fraction of residual-stream reconstruction attributable to
memorized features.

Use this to decide which layer to target with ablative-steering LoRA fine-tuning.
The layer where memorized features account for the largest fraction of the
reconstructed activation is where ablation during training has the most direct
effect on the memorized pathway.

Two metrics per layer
---------------------
energy_frac
    mean_t( Σ_{j∈M} z²_j,t  /  Σ_j z²_j,t )
    Share of activation energy from memorized features.  Cheap: only needs z and
    prob_general from the generality classifier.

norm_frac
    mean_t( ‖Σ_{j∈M} z_j,t v_j‖₂  /  ‖Σ_j z_j,t v_j‖₂ )
    Fraction of the SAE reconstruction vector norm that is memorized.  Slower
    (loads W_dec, does matrix multiplications) but accounts for constructive /
    destructive interference between decoder directions.  Often similar to
    energy_frac but diverges when memorized decoder directions cluster together.

Usage
-----
python mrvla/layer_ablation_analysis.py \\
    --sae-dir        ./checkpoints/sae_libero_goal_v3 \\
    --codes-dir      E:/libero_goal_demos/codes/sae_libero_goal_v3 \\
    --generality-dir E:/libero_goal_demos/generality/sae_libero_goal_v3 \\
    [--out-json      ./layer_ablation_rank.json] \\
    [--skip-norm-frac]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re

import numpy as np
import torch

# How many rows of z to process at once when computing norm_frac.
# Reduce if you hit OOM; increase for speed on machines with lots of RAM.
_BATCH = 4096


# ---------------------------------------------------------------------------
# Per-layer metric helpers
# ---------------------------------------------------------------------------

def _energy_frac(z: np.ndarray, is_memorized: np.ndarray) -> float:
    """mean_t( Σ_{j∈M} z²_j,t / Σ_j z²_j,t ).  NaN-safe."""
    z2    = z.astype(np.float32) ** 2          # [N, F]
    total = z2.sum(axis=1)                      # [N]
    mem   = z2[:, is_memorized].sum(axis=1)     # [N]
    valid = total > 0
    if not valid.any():
        return float("nan")
    return float((mem[valid] / total[valid]).mean())


def _norm_frac(
    z: np.ndarray,
    is_memorized: np.ndarray,
    W_dec: torch.Tensor,
) -> float:
    """mean_t( ‖x_mem_t‖₂ / ‖x_hat_t‖₂ ).  Batched to keep peak memory bounded."""
    W_all = W_dec.numpy().astype(np.float32)        # [F, d]
    W_mem = W_all[is_memorized]                     # [M, d]

    numerators   = []
    denominators = []

    for start in range(0, len(z), _BATCH):
        zb    = z[start : start + _BATCH].astype(np.float32)   # [B, F]
        x_hat = zb @ W_all                                      # [B, d]
        x_mem = zb[:, is_memorized] @ W_mem                     # [B, d]
        nhat  = np.linalg.norm(x_hat, axis=1)                  # [B]
        nmem  = np.linalg.norm(x_mem, axis=1)                  # [B]
        valid = nhat > 0
        if valid.any():
            numerators.append(nmem[valid])
            denominators.append(nhat[valid])

    if not numerators:
        return float("nan")

    num = np.concatenate(numerators)
    den = np.concatenate(denominators)
    return float((num / den).mean())


# ---------------------------------------------------------------------------
# SAE checkpoint loader
# ---------------------------------------------------------------------------

def _load_W_dec(sae_dir: str, layer_idx: int) -> torch.Tensor | None:
    pt = os.path.join(sae_dir, f"layer_{layer_idx:02d}", "final.pt")
    if not os.path.exists(pt):
        return None
    ckpt = torch.load(pt, map_location="cpu", weights_only=True)
    return ckpt["W_dec"].float()   # [F, d]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--sae-dir",        required=True,
                   help="SAE checkpoint dir (contains layer_NN/final.pt)")
    p.add_argument("--codes-dir",      required=True,
                   help="Codes dir (layer_NN.npz from extract_codes_and_metrics.py)")
    p.add_argument("--generality-dir", required=True,
                   help="Generality dir (layer_NN_generality.npz from generality_classifier.py)")
    p.add_argument("--out-json",       default=None,
                   help="Optional path to write JSON results")
    p.add_argument("--skip-norm-frac", action="store_true",
                   help="Skip norm_frac (no W_dec needed; faster for quick ranking)")
    args = p.parse_args()

    code_files = sorted(glob.glob(os.path.join(args.codes_dir, "layer_*.npz")))
    if not code_files:
        raise FileNotFoundError(f"No layer_*.npz files in {args.codes_dir!r}")

    results: dict[str, dict] = {}

    header = f"{'layer':<10} {'n_feat':>7} {'n_mem':>7} {'mem%':>6}  {'energy_frac':>12}  {'norm_frac':>10}"
    print()
    print(header)
    print("-" * len(header))

    for fpath in code_files:
        basename = os.path.basename(fpath).replace(".npz", "")   # "layer_08"
        m = re.match(r"layer_(\d+)", basename)
        if not m:
            continue
        layer_idx = int(m.group(1))

        # ── codes ──────────────────────────────────────────────────────────
        d = np.load(fpath)
        z = d["z"].astype(np.float32)   # [N, F]

        # ── generality ─────────────────────────────────────────────────────
        gen_path = os.path.join(args.generality_dir, f"{basename}_generality.npz")
        if not os.path.exists(gen_path):
            print(f"  WARNING: {gen_path} not found, skipping {basename}")
            continue
        prob_general  = np.load(gen_path)["prob_general"].astype(np.float32)
        is_memorized  = prob_general < 0.5

        F     = z.shape[1]
        n_mem = int(is_memorized.sum())
        n_gen = F - n_mem

        efrac = _energy_frac(z, is_memorized)

        if args.skip_norm_frac:
            nfrac     = None
            nfrac_str = "skipped"
        else:
            W_dec = _load_W_dec(args.sae_dir, layer_idx)
            if W_dec is None:
                nfrac     = None
                nfrac_str = "no ckpt"
            else:
                nfrac     = _norm_frac(z, is_memorized, W_dec)
                nfrac_str = f"{nfrac:.4f}"

        print(
            f"layer_{layer_idx:02d}   {F:>7} {n_mem:>7} {100*n_mem/F:>5.1f}%"
            f"  {efrac:>12.4f}  {nfrac_str:>10}"
        )

        results[f"layer_{layer_idx:02d}"] = dict(
            layer_idx   = layer_idx,
            n_features  = F,
            n_memorized = n_mem,
            n_general   = n_gen,
            mem_pct     = float(100.0 * n_mem / F),
            energy_frac = float(efrac),
            norm_frac   = float(nfrac) if nfrac is not None else None,
        )

    print()

    # ── ranking ────────────────────────────────────────────────────────────
    ranked = sorted(
        results.items(),
        key=lambda kv: kv[1]["energy_frac"],
        reverse=True,
    )
    print("Ranked by energy_frac (highest → most load-bearing memorized computation):")
    for rank, (name, r) in enumerate(ranked, 1):
        extra = (
            f"  norm_frac={r['norm_frac']:.4f}" if r["norm_frac"] is not None else ""
        )
        print(f"  {rank}. {name}  energy_frac={r['energy_frac']:.4f}{extra}")

    best = ranked[0][0] if ranked else "N/A"
    print(f"\n  → Suggested ablation target: {best}")
    print(
        "    (verify norm_frac agrees before committing — they can diverge when\n"
        "     memorized decoder directions are highly collinear.)"
    )

    if args.out_json:
        payload = {
            "layers": results,
            "ranked_by_energy_frac": [k for k, _ in ranked],
            "suggested_target": best,
        }
        with open(args.out_json, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"\nResults written to {args.out_json}")


if __name__ == "__main__":
    main()
