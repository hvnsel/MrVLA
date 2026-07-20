"""Tiebreakers for the L08-vs-L31 disagreement (timestep-level generality signal).

Two analyses read the same Check-2 summary table and disagreed on whether layer 31's
phase coherence is real structure or the same zero-inflation artifact that disqualifies
layers 16/24. Summary statistics cannot settle it; these three cheap checks can.

Reuses the exact Check-1/Check-2 code path (same v3 codes, same TopK coefficients, same
HARD general/memorized labels, same resample_profile). Nothing is re-extracted.

Tiebreaker 1 — shuffle null for phase_coherence.
    Permute ratio_t across timesteps *within each episode* (destroys temporal order,
    keeps the value distribution), recompute phase_coherence N_PERM times -> null.
    Is the observed coherence significantly above its own shuffle null? Do this for
    L08 and L31 so the 0.431-vs-0.508 comparison gets a baseline.

Tiebreaker 2 — structured vs random zeros.
    Bin every timestep by normalized trajectory phase. Plot (a) fraction of zero-ratio
    timesteps per phase bin and (b) mean ratio_t per phase bin. If L31's zeros are
    front-loaded and its mass ramps to the endgame, the "coherent ramp" is real. If the
    zeros are flat across phase, the ramp is an averaging artifact.

Tiebreaker 3 — per-feature late-mass decomposition at L31.
    Is the endgame mass broad-based across the general features (genuine) or carried by
    a few possibly-mislabeled single-onset features? Rank general features by their mass
    in the late phase, show the cumulative-contribution curve, and print each top
    contributor's coverage / mean_onsets / prob_general so single-onset mislabels stand
    out.

Usage
-----
python mrvla/check2_tiebreakers.py \
    --codes-dir      E:/libero_goal_demos/codes_v3 \
    --generality-dir E:/libero_goal_demos/generality_v3 \
    --out-dir        E:/libero_goal_demos/check2_timestep_variance/tiebreakers
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
from mrvla.check2_timestep_variance import (  # noqa: E402
    load_layer, per_timestep_ratios, resample_profile, group_by_episode,
    EPS, N_PHASE, MIN_TS_FOR_STRUCTURE,
)

N_PERM = 200
LATE_PHASE = 0.75          # timesteps with phase >= this count as "endgame"


# ---------------------------------------------------------------------------
# shared: ratios + masses with explicit zero handling
# ---------------------------------------------------------------------------
def masses(z, is_general, prob_general):
    _, _, total_mass, gmass_hard, _ = per_timestep_ratios(z, is_general, prob_general)
    zero_mass = total_mass <= EPS
    ratio = np.full(total_mass.shape, np.nan, dtype=np.float64)
    valid = ~zero_mass
    ratio[valid] = gmass_hard[valid] / total_mass[valid]
    return ratio, total_mass, gmass_hard, int(zero_mass.sum())


def phase_per_row(episode, timestep):
    """Normalized [0,1] trajectory phase for every row (timestep/ep_max)."""
    ep_ids = np.unique(episode)
    idx = np.searchsorted(ep_ids, episode)
    emax = np.zeros(len(ep_ids), dtype=np.float64)
    np.maximum.at(emax, idx, timestep.astype(np.float64))
    row_max = np.maximum(emax[idx], 1.0)
    return timestep.astype(np.float64) / row_max


def coherence_from_profiles(profiles):
    P = np.asarray(profiles)
    mean = P.mean(axis=0)
    std = P.std(axis=0)
    return float(np.std(mean) / (np.mean(std) + EPS)), mean, std


def profiles_from_groups(groups, ratio_override=None):
    """Resample each episode to N_PHASE bins. ratio_override lets us feed shuffled r."""
    profs = []
    for ep, ts, r in groups:
        rr = r if ratio_override is None else ratio_override[ep]
        p = resample_profile(ts, rr)
        if p is not None:
            profs.append(p)
    return profs


# ---------------------------------------------------------------------------
# Tiebreaker 1
# ---------------------------------------------------------------------------
def tiebreaker1(layer_idx, groups, out_dir, rng):
    obs_profs = profiles_from_groups(groups)
    obs_coh, _, _ = coherence_from_profiles(obs_profs)

    null = np.empty(N_PERM, dtype=np.float64)
    for i in range(N_PERM):
        shuffled = {}
        for ep, ts, r in groups:
            rr = r.copy()
            valid = ~np.isnan(rr)
            rr[valid] = rng.permutation(rr[valid])
            shuffled[ep] = rr
        profs = profiles_from_groups(groups, ratio_override=shuffled)
        null[i], _, _ = coherence_from_profiles(profs)

    null_mean = float(null.mean())
    null_std = float(null.std())
    z = (obs_coh - null_mean) / (null_std + EPS)
    p = float((np.sum(null >= obs_coh) + 1) / (N_PERM + 1))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(null, bins=40, color="#bbbbbb", edgecolor="black", linewidth=0.2,
            label=f"shuffle null (n={N_PERM})")
    ax.axvline(obs_coh, color="red", lw=2.0,
               label=f"observed={obs_coh:.3f}")
    ax.axvline(null_mean, color="black", ls="--", lw=1.0,
               label=f"null mean={null_mean:.3f}")
    ax.set_title(f"Layer {layer_idx:02d} — phase_coherence vs shuffle null\n"
                 f"z={z:.2f}, p={p:.4f}", fontsize=11)
    ax.set_xlabel("phase_coherence")
    ax.set_ylabel("count")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"layer_{layer_idx:02d}_tb1_shuffle_null.png"), dpi=130)
    plt.close(fig)

    print(f"  [TB1 L{layer_idx:02d}] observed={obs_coh:.3f}  "
          f"null={null_mean:.3f}±{null_std:.3f}  z={z:.2f}  p={p:.4f}  "
          f"{'REAL' if p < 0.05 else 'NOT above null'}", flush=True)
    return {"observed": obs_coh, "null_mean": null_mean, "null_std": null_std,
            "z": z, "p": p, "n_perm": N_PERM}


# ---------------------------------------------------------------------------
# Tiebreaker 2
# ---------------------------------------------------------------------------
def tiebreaker2(layer_idx, ratio, gmass_hard, phase, out_dir, n_bins=N_PHASE):
    bins = np.clip((phase * n_bins).astype(int), 0, n_bins - 1)
    is_zero = (gmass_hard <= EPS).astype(np.float64)     # general mass exactly 0

    counts = np.bincount(bins, minlength=n_bins).astype(np.float64)
    zero_frac = np.bincount(bins, weights=is_zero, minlength=n_bins) / np.maximum(counts, 1)
    r = np.nan_to_num(ratio, nan=0.0)
    mean_ratio = np.bincount(bins, weights=r, minlength=n_bins) / np.maximum(counts, 1)

    x = (np.arange(n_bins) + 0.5) / n_bins
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    ax1.bar(x, zero_frac, width=1.0 / n_bins * 0.9, color="#c44e52", edgecolor="black",
            linewidth=0.3)
    ax1.set_title(f"L{layer_idx:02d} — fraction of ZERO-ratio timesteps vs phase")
    ax1.set_xlabel("trajectory phase [0=start, 1=end]")
    ax1.set_ylabel("fraction with general_mass == 0")
    ax1.set_ylim(0, 1)

    ax2.plot(x, mean_ratio, color="#55a868", lw=2.0, marker="o", ms=3)
    ax2.set_title(f"L{layer_idx:02d} — mean ratio_t vs phase")
    ax2.set_xlabel("trajectory phase [0=start, 1=end]")
    ax2.set_ylabel("mean ratio_t")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"layer_{layer_idx:02d}_tb2_zero_and_mass_vs_phase.png"),
                dpi=130)
    plt.close(fig)

    early = zero_frac[: n_bins // 2].mean()
    late = zero_frac[n_bins // 2:].mean()
    print(f"  [TB2 L{layer_idx:02d}] zero-frac early(0-0.5)={early:.3f}  "
          f"late(0.5-1)={late:.3f}  drop={early - late:+.3f}   "
          f"mean-ratio early={mean_ratio[:n_bins//2].mean():.4f} "
          f"late={mean_ratio[n_bins//2:].mean():.4f}", flush=True)
    return {"zero_frac_bins": zero_frac.tolist(), "mean_ratio_bins": mean_ratio.tolist(),
            "zero_frac_early": float(early), "zero_frac_late": float(late),
            "mean_ratio_early": float(mean_ratio[:n_bins // 2].mean()),
            "mean_ratio_late": float(mean_ratio[n_bins // 2:].mean())}


# ---------------------------------------------------------------------------
# Tiebreaker 3
# ---------------------------------------------------------------------------
def tiebreaker3(layer_idx, z, is_general, phase, gen_metrics, out_dir):
    gen_cols = np.where(is_general)[0]                      # [G]
    late_mask = phase >= LATE_PHASE
    z_late = z[late_mask][:, gen_cols].astype(np.float64)   # [n_late, G]
    contrib = z_late.sum(axis=0)                            # [G] total late mass per feature
    total_late = contrib.sum() + EPS
    frac = contrib / total_late

    order = np.argsort(frac)[::-1]
    gen_cols_sorted = gen_cols[order]
    frac_sorted = frac[order]
    cum = np.cumsum(frac_sorted)
    n_for_90 = int(np.searchsorted(cum, 0.90) + 1)

    # cumulative contribution curve
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(np.arange(1, len(cum) + 1), cum, marker="o", ms=4, color="#4c72b0")
    ax.axhline(0.90, color="red", ls="--", lw=1.0, label="90% of late mass")
    ax.axvline(n_for_90, color="red", ls=":", lw=1.0,
               label=f"{n_for_90}/{len(gen_cols)} features")
    ax.set_title(f"L{layer_idx:02d} — cumulative late-phase (>= {LATE_PHASE}) mass\n"
                 f"across {len(gen_cols)} general features")
    ax.set_xlabel("general features ranked by late-mass contribution")
    ax.set_ylabel("cumulative fraction of late mass")
    ax.set_ylim(0, 1.02)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"layer_{layer_idx:02d}_tb3_feature_contrib.png"), dpi=130)
    plt.close(fig)

    cov = gen_metrics["coverage"]
    onsets = gen_metrics["mean_onsets"]
    prob = gen_metrics["prob_general"]

    rows = []
    print(f"  [TB3 L{layer_idx:02d}] {n_for_90}/{len(gen_cols)} features carry 90% of "
          f"late (>= {LATE_PHASE}) mass. Top contributors:", flush=True)
    print(f"      {'feat':>5} {'late_frac':>9} {'coverage':>9} {'mean_onsets':>11} "
          f"{'P(gen)':>7}", flush=True)
    topn = min(10, len(gen_cols_sorted))
    for i in range(topn):
        f = int(gen_cols_sorted[i])
        rows.append({"feature": f, "late_frac": float(frac_sorted[i]),
                     "coverage": float(cov[f]), "mean_onsets": float(onsets[f]),
                     "prob_general": float(prob[f])})
        print(f"      {f:>5} {frac_sorted[i]:>9.3f} {cov[f]:>9.3f} "
              f"{onsets[f]:>11.3f} {prob[f]:>7.3f}", flush=True)

    # suspect single-onset mislabels among the general set
    suspect = int(np.sum((cov[gen_cols] < 0.10) & (onsets[gen_cols] <= 1.2)))
    print(f"      suspect single-onset mislabels in general set "
          f"(coverage<0.10 & mean_onsets<=1.2): {suspect}/{len(gen_cols)}", flush=True)
    return {"n_general": int(len(gen_cols)), "n_for_90pct_late_mass": n_for_90,
            "top_contributors": rows, "suspect_single_onset": suspect}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--codes-dir", default="E:/libero_goal_demos/codes_v3")
    ap.add_argument("--generality-dir", default="E:/libero_goal_demos/generality_v3")
    ap.add_argument("--out-dir",
                    default="E:/libero_goal_demos/check2_timestep_variance/tiebreakers")
    ap.add_argument("--layers", default="8,31")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    layers = [int(x) for x in args.layers.split(",") if x.strip() != ""]
    os.makedirs(args.out_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    result = {}
    for layer_idx in layers:
        print(f"\n=== layer {layer_idx:02d} ===", flush=True)
        z, episode, timestep, is_general, prob_general = load_layer(
            args.codes_dir, args.generality_dir, layer_idx)
        g = np.load(os.path.join(args.generality_dir,
                                 f"layer_{layer_idx:02d}_generality.npz"))
        gen_metrics = {"coverage": g["coverage"], "mean_onsets": g["mean_onsets"],
                       "prob_general": g["prob_general"]}

        ratio, total_mass, gmass_hard, n_zero = masses(z, is_general, prob_general)
        phase = phase_per_row(episode, timestep)
        groups = group_by_episode(episode, timestep, ratio)

        print(f"  general_features={int(is_general.sum())}  "
              f"zero-mass(total==0) timesteps={n_zero}  "
              f"zero-RATIO(general==0) timesteps={int((gmass_hard <= EPS).sum())}"
              f"/{len(gmass_hard)} ({(gmass_hard <= EPS).mean():.1%})", flush=True)

        tb1 = tiebreaker1(layer_idx, groups, args.out_dir, rng)
        tb2 = tiebreaker2(layer_idx, ratio, gmass_hard, phase, args.out_dir)
        tb3 = tiebreaker3(layer_idx, z, is_general, phase, gen_metrics, args.out_dir)
        result[f"layer_{layer_idx:02d}"] = {"tb1_shuffle_null": tb1,
                                            "tb2_zero_vs_phase": tb2,
                                            "tb3_feature_contrib": tb3}

    with open(os.path.join(args.out_dir, "tiebreakers.json"), "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n[done] -> {os.path.join(args.out_dir, 'tiebreakers.json')}")


if __name__ == "__main__":
    main()
