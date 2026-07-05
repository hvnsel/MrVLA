"""Apply the paper's logistic-regression generality classifier to SAE features.

Formula (arXiv 2603.19183, Appendix B):
    P(general | m) = sigmoid(β₀ + β₁*ō + β₂*c + β₃*ā + β₄*ℓ̄ᵣ)

    β₀ = -4.20   (intercept)
    β₁ =  1.89   (mean onset count, ō)
    β₂ =  1.80   (episode coverage, c)
    β₃ =  0.52   (mean activation magnitude when firing, ā)
    β₄ = -0.36   (relative run length, ℓ̄ᵣ = mean_run_length / mean_episode_length)

    Threshold: P >= 0.5 → "general", else "memorized"

The four metrics are loaded directly from the layer_NN.npz files produced by
extract_codes_and_metrics.py; ā and ℓ̄ᵣ are derived on the fly from the stored
sparse codes z and metadata.

Usage
-----
python generality_classifier.py \\
    --codes-dir E:/libero_goal_demos/codes \\
    --out-dir   E:/libero_goal_demos/generality
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np


# ---------------------------------------------------------------------------
# Classifier constants from the paper (fit on 30 hand-labeled features)
# ---------------------------------------------------------------------------
BETA = dict(
    intercept=-4.20,
    mean_onsets=1.89,
    coverage=1.80,
    mean_act_magnitude=0.52,
    rel_run_length=-0.36,
)
THRESHOLD = 0.5
TAU_ON = 0.1   # paper's activation threshold for defining "feature fires"


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


# ---------------------------------------------------------------------------
# Derived metric helpers
# ---------------------------------------------------------------------------
def mean_activation_magnitude(z: np.ndarray) -> np.ndarray:
    """ā[f] = mean of z[:, f] over rows where it fires (z > 0).

    Returns [F] float32; 0.0 for dead features.
    """
    z = z.astype(np.float32)
    F = z.shape[1]
    out = np.zeros(F, dtype=np.float32)
    for f in range(F):
        col = z[:, f]
        active = col[col > 0]
        if len(active) > 0:
            out[f] = float(active.mean())
    return out


def mean_activation_magnitude_fast(z: np.ndarray, tau: float = TAU_ON) -> np.ndarray:
    """Vectorized version — faster for large N."""
    z = z.astype(np.float32, copy=False)
    active_sum = np.where(z > tau, z, 0.0).sum(axis=0)   # [F]
    active_cnt = (z > tau).sum(axis=0).astype(np.float32) # [F]
    return np.where(active_cnt > 0, active_sum / np.maximum(active_cnt, 1), 0.0)


def recompute_metrics_with_tau(
    z: np.ndarray, episode: np.ndarray, timestep: np.ndarray, tau: float = TAU_ON
) -> tuple:
    """Recompute coverage, mean_onsets, mean_run_length with activation threshold tau."""
    z = z.astype(np.float32, copy=False)
    active = (z > tau)                          # [N, F]
    F = z.shape[1]
    unique_eps, ep_counts = np.unique(episode, return_counts=True)
    n_eps = len(unique_eps)

    onsets_ef = np.zeros((n_eps, F), dtype=np.int32)
    active_steps_ef = np.zeros((n_eps, F), dtype=np.int32)

    for e_idx, ep_id in enumerate(unique_eps):
        mask = (episode == ep_id)
        order = np.argsort(timestep[mask])
        active_ep = active[mask][order]          # [T, F]
        padded = np.vstack([
            np.zeros((1, F), dtype=bool),
            active_ep,
            np.zeros((1, F), dtype=bool),
        ])
        diff = padded[1:].astype(np.int8) - padded[:-1].astype(np.int8)
        onsets_ef[e_idx] = (diff == 1).sum(axis=0)
        active_steps_ef[e_idx] = active_ep.sum(axis=0)

    fired_in_ep = onsets_ef > 0
    n_fired = fired_in_ep.sum(axis=0)           # [F]
    coverage = n_fired / n_eps
    mean_onsets = np.where(
        n_fired > 0,
        (onsets_ef * fired_in_ep).sum(axis=0) / np.maximum(n_fired, 1),
        0.0,
    )
    total_bursts = onsets_ef.sum(axis=0)
    total_active = active_steps_ef.sum(axis=0)
    mean_run_length = np.where(
        total_bursts > 0,
        total_active / np.maximum(total_bursts, 1),
        0.0,
    )
    ep_mean_len = float(ep_counts.mean())
    return (
        coverage.astype(np.float32),
        mean_onsets.astype(np.float32),
        mean_run_length.astype(np.float32),
        ep_mean_len,
    )


def mean_episode_length(episode: np.ndarray) -> float:
    """Average number of timesteps per episode."""
    _, counts = np.unique(episode, return_counts=True)
    return float(counts.mean())


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------
def classify_features(
    coverage: np.ndarray,
    mean_onsets: np.ndarray,
    mean_run_length: np.ndarray,
    mean_act_mag: np.ndarray,
    ep_mean_len: float,
    verbose: bool = True,
) -> dict:
    """Return per-feature P(general) and binary label using paper's β values."""
    # Relative run length: run length normalised by mean episode length
    rel_run_length = mean_run_length / max(ep_mean_len, 1.0)

    logit = (
        BETA["intercept"]
        + BETA["mean_onsets"]     * mean_onsets
        + BETA["coverage"]        * coverage
        + BETA["mean_act_magnitude"] * mean_act_mag
        + BETA["rel_run_length"]  * rel_run_length
    )
    prob_general = sigmoid(logit)
    is_general = prob_general >= THRESHOLD

    n_features = len(coverage)
    n_general = int(is_general.sum())
    n_memorized = n_features - n_general
    frac_general = n_general / n_features

    if verbose:
        print(f"  features={n_features}  general={n_general} ({100*frac_general:.2f}%)  "
              f"memorized={n_memorized} ({100*(1-frac_general):.2f}%)")
        print(f"  paper target: ~0.45% general")
        print(f"  metric summary:")
        print(f"    coverage       mean={coverage.mean():.4f}  std={coverage.std():.4f}")
        print(f"    mean_onsets    mean={mean_onsets.mean():.4f}  std={mean_onsets.std():.4f}")
        print(f"    mean_act_mag   mean={mean_act_mag.mean():.4f}  std={mean_act_mag.std():.4f}")
        print(f"    rel_run_length mean={rel_run_length.mean():.4f}  std={rel_run_length.std():.4f}")
        print(f"    P(general)     mean={prob_general.mean():.4f}  "
              f"median={np.median(prob_general):.4f}  max={prob_general.max():.4f}")
        print(f"    mean episode length: {ep_mean_len:.1f} steps")

    return {
        "prob_general": prob_general.astype(np.float32),
        "is_general": is_general,
        "n_features": n_features,
        "n_general": n_general,
        "n_memorized": n_memorized,
        "frac_general": float(frac_general),
        "ep_mean_len": float(ep_mean_len),
        "rel_run_length": rel_run_length.astype(np.float32),
        "mean_act_mag": mean_act_mag.astype(np.float32),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--codes-dir", required=True,
                   help="Directory produced by extract_codes_and_metrics.py")
    p.add_argument("--out-dir", required=True,
                   help="Where to write per-layer results + summary.json")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Discover layers from npz files
    import glob
    layer_files = sorted(glob.glob(os.path.join(args.codes_dir, "layer_*.npz")))
    if not layer_files:
        raise FileNotFoundError(f"No layer_*.npz files in {args.codes_dir!r}")
    print(f"[gen] found {len(layer_files)} layer files", flush=True)
    print(f"[gen] β = {BETA}", flush=True)

    summary = {"beta": BETA, "threshold": THRESHOLD, "layers": {}}

    for fpath in layer_files:
        layer_name = os.path.basename(fpath).replace(".npz", "")
        print(f"\n[gen] ====== {layer_name} ======", flush=True)

        d = np.load(fpath)
        episode        = d["episode"]
        timestep       = d["timestep"]
        z              = d["z"]   # [N, F] float16

        F = z.shape[1]
        print(f"  recomputing metrics with tau_on={TAU_ON} ...", flush=True)
        coverage, mean_onsets, mean_run_length, ep_mean_len = recompute_metrics_with_tau(
            z.astype(np.float32), episode, timestep
        )
        mean_act_mag = mean_activation_magnitude_fast(z.astype(np.float32))

        result = classify_features(
            coverage=coverage,
            mean_onsets=mean_onsets,
            mean_run_length=mean_run_length,
            mean_act_mag=mean_act_mag,
            ep_mean_len=ep_mean_len,
        )

        # Save per-layer results
        out_path = os.path.join(args.out_dir, f"{layer_name}_generality.npz")
        np.savez_compressed(
            out_path,
            prob_general=result["prob_general"],
            is_general=result["is_general"].astype(np.uint8),
            coverage=coverage,
            mean_onsets=mean_onsets,
            mean_act_mag=result["mean_act_mag"],
            rel_run_length=result["rel_run_length"],
        )
        print(f"  saved {out_path}", flush=True)

        # Top-10 most general features
        top10_idx = np.argsort(result["prob_general"])[-10:][::-1]
        print(f"  top-10 general features (by P(general)):", flush=True)
        print(f"  {'feat':>6}  {'P(gen)':>7}  {'coverage':>9}  "
              f"{'onsets':>7}  {'act_mag':>8}  {'rel_rl':>7}")
        for fi in top10_idx:
            print(f"  {fi:>6}  {result['prob_general'][fi]:>7.4f}  "
                  f"{coverage[fi]:>9.4f}  {mean_onsets[fi]:>7.3f}  "
                  f"{result['mean_act_mag'][fi]:>8.4f}  "
                  f"{result['rel_run_length'][fi]:>7.4f}")

        summary["layers"][layer_name] = {
            "n_features": result["n_features"],
            "n_general": result["n_general"],
            "n_memorized": result["n_memorized"],
            "frac_general": result["frac_general"],
            "ep_mean_len": result["ep_mean_len"],
            "top10_general_features": top10_idx.tolist(),
        }

    out_summary = os.path.join(args.out_dir, "summary.json")
    with open(out_summary, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[gen] === FINAL SUMMARY ===", flush=True)
    for layer_name, s in summary["layers"].items():
        print(f"  {layer_name}: {s['n_general']}/{s['n_features']} general "
              f"({100*s['frac_general']:.2f}%)")
    print(f"  paper target: ~0.45%")
    print(f"\n[gen] done. {out_summary}", flush=True)


if __name__ == "__main__":
    main()
