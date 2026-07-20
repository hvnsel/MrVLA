"""Tiebreaker 3b — coverage-gated endgame ramp.

TB3 showed L31's endgame ramp is broad-based (23/36 features) with no single-onset
mislabels, BUT the general features have pathologically low per-episode coverage
(~1-2%). A canonical "general" feature should have HIGH coverage (fire across many
episodes). This asks the decision-flipping question directly:

    Does L31's endgame mass ramp survive if we keep ONLY high-coverage general
    features (the ones whose "general" label we actually trust)?

If the ramp survives high-coverage gating -> L31's signal is genuine general
computation concentrated in the endgame (L31 wins). If the ramp is carried only by
the low-coverage features and collapses when gated -> L31 is the low-coverage-label
artifact the critic feared (fall back to L08 / null).

Reuses the exact code path. HARD labels. Compares L08 and L31.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mrvla.check2_timestep_variance import load_layer, EPS, N_PHASE  # noqa: E402
from mrvla.check2_tiebreakers import phase_per_row  # noqa: E402

COV_THRESHOLDS = [0.0, 0.10, 0.30]


def mean_ratio_vs_phase(z, gen_cols, total_mass, phase, n_bins=N_PHASE):
    if len(gen_cols) == 0:
        return np.zeros(n_bins), 0
    gmass = z[:, gen_cols].astype(np.float64).sum(axis=1)
    ratio = np.where(total_mass > EPS, gmass / np.maximum(total_mass, EPS), 0.0)
    bins = np.clip((phase * n_bins).astype(int), 0, n_bins - 1)
    counts = np.bincount(bins, minlength=n_bins).astype(np.float64)
    prof = np.bincount(bins, weights=ratio, minlength=n_bins) / np.maximum(counts, 1)
    return prof, len(gen_cols)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--codes-dir", default="E:/libero_goal_demos/codes_v3")
    ap.add_argument("--generality-dir", default="E:/libero_goal_demos/generality_v3")
    ap.add_argument("--out-dir",
                    default="E:/libero_goal_demos/check2_timestep_variance/tiebreakers")
    ap.add_argument("--layers", default="8,31")
    args = ap.parse_args()

    layers = [int(x) for x in args.layers.split(",") if x.strip() != ""]
    os.makedirs(args.out_dir, exist_ok=True)
    x = (np.arange(N_PHASE) + 0.5) / N_PHASE
    result = {}

    for layer_idx in layers:
        print(f"\n=== layer {layer_idx:02d} ===", flush=True)
        z, episode, timestep, is_general, _ = load_layer(
            args.codes_dir, args.generality_dir, layer_idx)
        g = np.load(os.path.join(args.generality_dir,
                                 f"layer_{layer_idx:02d}_generality.npz"))
        coverage = g["coverage"]
        total_mass = z.astype(np.float64).sum(axis=1)
        phase = phase_per_row(episode, timestep)

        gen_cols_all = np.where(is_general)[0]
        cov_gen = coverage[gen_cols_all]
        print(f"  general features: {len(gen_cols_all)}  "
              f"coverage min/median/max = "
              f"{cov_gen.min():.3f}/{np.median(cov_gen):.3f}/{cov_gen.max():.3f}", flush=True)
        for thr in COV_THRESHOLDS:
            print(f"    coverage >= {thr:.2f}: "
                  f"{int((cov_gen >= thr).sum())}/{len(gen_cols_all)} features", flush=True)

        fig, ax = plt.subplots(figsize=(9, 5.5))
        layer_res = {"n_general": int(len(gen_cols_all)),
                     "coverage_min": float(cov_gen.min()),
                     "coverage_median": float(np.median(cov_gen)),
                     "coverage_max": float(cov_gen.max()),
                     "gated": {}}
        for thr in COV_THRESHOLDS:
            gen_cols = gen_cols_all[cov_gen >= thr]
            prof, n = mean_ratio_vs_phase(z, gen_cols, total_mass, phase)
            early = float(prof[: N_PHASE // 2].mean())
            late = float(prof[N_PHASE // 2:].mean())
            ramp = late / (early + EPS)
            ax.plot(x, prof, marker="o", ms=3, lw=1.8,
                    label=f"cov>={thr:.2f}  (n={n}, ramp x{ramp:.1f})")
            print(f"    cov>={thr:.2f}: n={n:>2}  early={early:.4f} late={late:.4f}  "
                  f"ramp=x{ramp:.2f}", flush=True)
            layer_res["gated"][f"cov_ge_{thr:.2f}"] = {
                "n_features": int(n), "early": early, "late": late, "ramp": float(ramp)}

        ax.set_title(f"Layer {layer_idx:02d} — endgame ramp under coverage gating (hard)",
                     fontsize=11)
        ax.set_xlabel("trajectory phase [0=start, 1=end]")
        ax.set_ylabel("mean ratio_t")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(args.out_dir,
                                 f"layer_{layer_idx:02d}_tb3b_coverage_gated_ramp.png"), dpi=130)
        plt.close(fig)
        result[f"layer_{layer_idx:02d}"] = layer_res

    with open(os.path.join(args.out_dir, "tiebreaker3b.json"), "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n[done] -> {os.path.join(args.out_dir, 'tiebreaker3b.json')}")


if __name__ == "__main__":
    main()
