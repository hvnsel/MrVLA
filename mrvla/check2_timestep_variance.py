"""Check 2 / 2b / 3 — Timestep-level generality-score variance.

Episode-level averaging washed the general/memorized signal out (Check 1). This drops
to the *timestep* level to ask whether usable variance lives there instead, and whether
it is semantically aligned across trajectories.

It REUSES the exact Check-1 extraction path (mrvla.episode_generality_variance):
same v3 codes (post-normalization TopK encoder coefficients), same general/memorized
labels. Nothing is re-extracted or retrained. Primary label = HARD (P(general) >= 0.5).

Per timestep t:
    general_mass_t = sum of coefficients of active features labelled general
    total_mass_t   = sum of coefficients of all active features
    ratio_t        = general_mass_t / total_mass_t          (NaN if total_mass_t == 0)

Checks
------
Check 2  : pool ratio_t over every timestep of every episode; stats + histogram/layer.
Check 2b : 6 representative episodes/layer (high/median/low episode-avg); ratio_t vs t.
Check 3  : time-normalize each episode to [0,1], resample ratio_t to N_PHASE bins,
           average the profile across all episodes; plot mean +/- std per phase bin.

Usage
-----
python mrvla/check2_timestep_variance.py \
    --codes-dir      E:/libero_goal_demos/codes_v3 \
    --generality-dir E:/libero_goal_demos/generality_v3 \
    --out-dir        E:/libero_goal_demos/check2_timestep_variance \
    --layers 0,8,16,24,31
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Reuse the exact Check-1 loader + mass computation (guarantees consistency with the
# per-episode table we already have).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mrvla.episode_generality_variance import load_layer, per_timestep_ratios  # noqa: E402


EPS = 1e-8
N_PHASE = 20                       # Check 3 phase bins
MIN_TS_FOR_STRUCTURE = 10          # min valid timesteps to use an episode in 2b / 3
EXPECTED_GENERAL = {0: 3, 8: 9, 16: 12, 24: 13, 31: 36}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def hard_ratio_with_nan(z, is_general, prob_general):
    """ratio_t using HARD labels, with explicit NaN where total_mass_t == 0.

    Returns (ratio_hard [N] float64 w/ NaN, total_mass [N], n_zero_mass int).
    """
    _, _, total_mass, gmass_hard, _ = per_timestep_ratios(z, is_general, prob_general)
    zero_mask = total_mass <= EPS
    ratio = np.full(total_mass.shape, np.nan, dtype=np.float64)
    valid = ~zero_mask
    ratio[valid] = gmass_hard[valid] / total_mass[valid]
    return ratio, total_mass, int(zero_mask.sum())


def group_by_episode(episode, timestep, ratio):
    """Return ordered dict-like list of (ep_id, ts_sorted, ratio_sorted).

    Rows are sorted by (episode, timestep). ratio may contain NaN.
    """
    order = np.lexsort((timestep, episode))
    ep_s = episode[order]
    ts_s = timestep[order]
    r_s = ratio[order]
    uniq, starts = np.unique(ep_s, return_index=True)
    starts = list(starts) + [len(ep_s)]
    groups = []
    for i, ep in enumerate(uniq):
        a, b = starts[i], starts[i + 1]
        groups.append((int(ep), ts_s[a:b], r_s[a:b]))
    return groups


def episode_nanmean(groups):
    """Per-episode mean of valid ratio_t. Returns (ep_ids [E], scores [E] w/ NaN)."""
    ep_ids = np.array([g[0] for g in groups], dtype=np.int64)
    scores = np.array(
        [np.nanmean(g[2]) if np.any(~np.isnan(g[2])) else np.nan for g in groups],
        dtype=np.float64,
    )
    return ep_ids, scores


def pooled_stats(x):
    """Stats over a 1-D array ignoring NaN."""
    v = x[~np.isnan(x)]
    if v.size == 0:
        return {k: float("nan") for k in
                ("n", "mean", "std", "cv", "min", "p5", "p25", "p50", "p75", "p95", "max")}
    pct = np.percentile(v, [5, 25, 50, 75, 95])
    mean = float(v.mean())
    std = float(v.std())
    return {
        "n": int(v.size),
        "mean": mean,
        "std": std,
        "cv": float(std / (abs(mean) + EPS)),
        "min": float(v.min()),
        "p5": float(pct[0]),
        "p25": float(pct[1]),
        "p50": float(pct[2]),
        "p75": float(pct[3]),
        "p95": float(pct[4]),
        "max": float(v.max()),
    }


def resample_profile(ts, ratio, n_bins=N_PHASE):
    """Resample one episode's ratio_t onto a common [0,1] phase axis (n_bins points).

    NaN timesteps are dropped before interpolation. Returns [n_bins] or None if too few
    valid points.
    """
    valid = ~np.isnan(ratio)
    if valid.sum() < MIN_TS_FOR_STRUCTURE:
        return None
    t = ts[valid].astype(np.float64)
    r = ratio[valid].astype(np.float64)
    order = np.argsort(t)
    t, r = t[order], r[order]
    if t[-1] == t[0]:
        return None
    phase = (t - t[0]) / (t[-1] - t[0])          # -> [0, 1]
    bin_centers = np.linspace(0.0, 1.0, n_bins)
    return np.interp(bin_centers, phase, r)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def plot_check2_hist(layer_idx, ratio, stats, ep_cv, out_path):
    v = ratio[~np.isnan(ratio)]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.hist(v, bins=80, color="#4c72b0", edgecolor="black", linewidth=0.2)
    ax.axvline(stats["mean"], color="red", ls="--", lw=1.2, label=f"mean={stats['mean']:.4f}")
    ax.axvline(stats["p50"], color="orange", ls=":", lw=1.2, label=f"median={stats['p50']:.4f}")
    ax.set_title(
        f"Layer {layer_idx:02d} — pooled per-TIMESTEP ratio_t (hard)\n"
        f"n={stats['n']}  std={stats['std']:.4f}  timestep-CV={stats['cv']:.3f}  "
        f"(episode-CV={ep_cv:.3f})",
        fontsize=11,
    )
    ax.set_xlabel("ratio_t = general_mass / total_mass")
    ax.set_ylabel("count")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_check2b_episodes(layer_idx, picks, groups_by_id, out_path):
    fig, axes = plt.subplots(3, 2, figsize=(13, 10), sharex=False)
    axes = axes.ravel()
    for ax, (ep_id, tier, ep_score) in zip(axes, picks):
        ts, r = groups_by_id[ep_id]
        ax.plot(ts, r, color="#c44e52", lw=1.0, marker=".", ms=3)
        # mark zero-mass (NaN) timesteps along the bottom
        nan_ts = ts[np.isnan(r)]
        if nan_ts.size:
            ax.scatter(nan_ts, np.zeros_like(nan_ts, dtype=float),
                       marker="x", color="gray", s=12, label=f"zero-mass ({nan_ts.size})")
        ax.set_title(f"ep {ep_id}  [{tier}]  avg={ep_score:.4f}", fontsize=9)
        ax.set_xlabel("timestep t")
        ax.set_ylabel("ratio_t")
        if nan_ts.size:
            ax.legend(fontsize=7)
    fig.suptitle(f"Layer {layer_idx:02d} — within-episode ratio_t vs t (hard)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_check3_phase(layer_idx, phase_mean, phase_std, n_used, out_path):
    x = np.linspace(0.0, 1.0, N_PHASE)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(x, phase_mean, color="#55a868", lw=2.0, label="mean profile")
    ax.fill_between(x, phase_mean - phase_std, phase_mean + phase_std,
                    color="#55a868", alpha=0.25, label="±1 std across episodes")
    ax.set_title(
        f"Layer {layer_idx:02d} — phase-aligned ratio_t "
        f"(mean±std over {n_used} episodes, hard)",
        fontsize=11,
    )
    ax.set_xlabel("normalized trajectory phase [0=start, 1=end]")
    ax.set_ylabel("ratio_t")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--codes-dir", default="E:/libero_goal_demos/codes_v3")
    ap.add_argument("--generality-dir", default="E:/libero_goal_demos/generality_v3")
    ap.add_argument("--out-dir", default="E:/libero_goal_demos/check2_timestep_variance")
    ap.add_argument("--layers", default="0,8,16,24,31")
    args = ap.parse_args()

    layers = [int(x) for x in args.layers.split(",") if x.strip() != ""]
    os.makedirs(args.out_dir, exist_ok=True)

    csv_rows = []
    summary = {"codes_dir": args.codes_dir, "generality_dir": args.generality_dir, "layers": {}}

    for layer_idx in layers:
        print(f"\n=== layer {layer_idx:02d} ===", flush=True)
        z, episode, timestep, is_general, prob_general = load_layer(
            args.codes_dir, args.generality_dir, layer_idx)

        n_features = z.shape[1]
        n_general = int(is_general.sum())
        n_episodes = int(len(np.unique(episode)))
        exp = EXPECTED_GENERAL.get(layer_idx)
        flag = "" if (exp is None or exp == n_general) else f"  <-- MISMATCH (expected {exp})"
        print(f"  z={z.shape}  episodes={n_episodes}  "
              f"general_features={n_general}/{n_features}{flag}", flush=True)

        # ---- ratios (hard, NaN on zero-mass) ------------------------------
        ratio, total_mass, n_zero = hard_ratio_with_nan(z, is_general, prob_general)
        frac_zero = n_zero / ratio.size
        print(f"  zero-mass timesteps: {n_zero}/{ratio.size} ({frac_zero:.2%}) -> NaN", flush=True)

        groups = group_by_episode(episode, timestep, ratio)
        groups_by_id = {ep: (ts, r) for ep, ts, r in groups}
        ep_ids, ep_scores = episode_nanmean(groups)
        ep_valid = ep_scores[~np.isnan(ep_scores)]
        ep_cv = float(ep_valid.std() / (abs(ep_valid.mean()) + EPS))

        # ---- Check 2: pooled timestep stats ------------------------------
        st = pooled_stats(ratio)
        st["n_zero_mass"] = n_zero
        st["frac_zero_mass"] = frac_zero
        st["episode_cv"] = ep_cv
        st["cv_ratio_ts_over_ep"] = float(st["cv"] / (ep_cv + EPS))
        hist_path = os.path.join(args.out_dir, f"layer_{layer_idx:02d}_check2_pooled_hist.png")
        plot_check2_hist(layer_idx, ratio, st, ep_cv, hist_path)
        print(f"  [check2] pooled ts: mean={st['mean']:.4f} std={st['std']:.4f} "
              f"CV={st['cv']:.3f}  min={st['min']:.4f} max={st['max']:.4f}", flush=True)
        print(f"           pct 5/25/50/75/95 = "
              f"{st['p5']:.4f}/{st['p25']:.4f}/{st['p50']:.4f}/{st['p75']:.4f}/{st['p95']:.4f}",
              flush=True)

        # ---- Check 2b: 6 representative episodes -------------------------
        # only episodes with enough valid timesteps qualify
        ok = np.array([np.sum(~np.isnan(groups_by_id[ep][1])) >= MIN_TS_FOR_STRUCTURE
                       for ep in ep_ids])
        cand_ids = ep_ids[ok & ~np.isnan(ep_scores)]
        cand_scores = ep_scores[ok & ~np.isnan(ep_scores)]
        order = np.argsort(cand_scores)
        cand_ids, cand_scores = cand_ids[order], cand_scores[order]
        m = len(cand_ids)
        mid = m // 2
        pick_idx = [m - 1, m - 2, mid, mid - 1, 0, 1]           # high, median, low
        tiers = ["high", "high", "median", "median", "low", "low"]
        picks = []
        for pi, tier in zip(pick_idx, tiers):
            pi = int(np.clip(pi, 0, m - 1))
            picks.append((int(cand_ids[pi]), tier, float(cand_scores[pi])))
        b_path = os.path.join(args.out_dir, f"layer_{layer_idx:02d}_check2b_episodes.png")
        plot_check2b_episodes(layer_idx, picks, groups_by_id, b_path)
        # within-episode burstiness metric: median over episodes of (std/mean of ratio_t)
        within_cv = []
        for ep in cand_ids:
            r = groups_by_id[ep][1]
            rv = r[~np.isnan(r)]
            if rv.size >= MIN_TS_FOR_STRUCTURE and rv.mean() > EPS:
                within_cv.append(rv.std() / rv.mean())
        med_within_cv = float(np.median(within_cv)) if within_cv else float("nan")
        print(f"  [check2b] median within-episode CV = {med_within_cv:.3f}  "
              f"({'bursty' if med_within_cv > 0.5 else 'flat'})", flush=True)

        # ---- Check 3: phase-aligned profile ------------------------------
        profiles = []
        for ep, ts, r in groups:
            p = resample_profile(ts, r)
            if p is not None:
                profiles.append(p)
        profiles = np.array(profiles) if profiles else np.zeros((0, N_PHASE))
        n_used = profiles.shape[0]
        phase_mean = np.nanmean(profiles, axis=0) if n_used else np.full(N_PHASE, np.nan)
        phase_std = np.nanstd(profiles, axis=0) if n_used else np.full(N_PHASE, np.nan)
        c_path = os.path.join(args.out_dir, f"layer_{layer_idx:02d}_check3_phase.png")
        plot_check3_phase(layer_idx, phase_mean, phase_std, n_used, c_path)
        # coherence: spread of the mean profile vs typical across-episode spread per bin
        between = float(np.nanstd(phase_mean))
        within = float(np.nanmean(phase_std))
        coherence = float(between / (within + EPS))
        print(f"  [check3] phase profile: between-bin std={between:.4f}  "
              f"mean within-bin std={within:.4f}  coherence={coherence:.3f}  "
              f"({'coherent' if coherence > 0.5 else 'scattered'})", flush=True)

        # ---- verdict -----------------------------------------------------
        cv_gain = st["cv_ratio_ts_over_ep"]
        print(f"  VERDICT L{layer_idx:02d}: "
              f"(a) timestep-CV {st['cv']:.3f} vs episode-CV {ep_cv:.3f} "
              f"(x{cv_gain:.2f}); "
              f"(b) within-episode {'BURSTY' if med_within_cv > 0.5 else 'FLAT'}; "
              f"(c) phase profile {'COHERENT' if coherence > 0.5 else 'SCATTERED'}",
              flush=True)

        csv_rows.append({
            "layer": layer_idx,
            "n_general_features": n_general,
            "n_general_expected": exp,
            "n_episodes": n_episodes,
            "n_timesteps_valid": st["n"],
            "n_zero_mass": n_zero,
            "frac_zero_mass": round(frac_zero, 6),
            "ts_mean": round(st["mean"], 6),
            "ts_std": round(st["std"], 6),
            "ts_cv": round(st["cv"], 6),
            "ts_min": round(st["min"], 6),
            "ts_p5": round(st["p5"], 6),
            "ts_p25": round(st["p25"], 6),
            "ts_p50": round(st["p50"], 6),
            "ts_p75": round(st["p75"], 6),
            "ts_p95": round(st["p95"], 6),
            "ts_max": round(st["max"], 6),
            "episode_cv": round(ep_cv, 6),
            "ts_cv_over_ep_cv": round(cv_gain, 4),
            "median_within_episode_cv": round(med_within_cv, 6),
            "phase_coherence": round(coherence, 6),
        })
        summary["layers"][f"layer_{layer_idx:02d}"] = {
            "stats": st,
            "n_general_features": n_general,
            "n_general_expected": exp,
            "median_within_episode_cv": med_within_cv,
            "phase_between_bin_std": between,
            "phase_mean_within_bin_std": within,
            "phase_coherence": coherence,
            "representative_episodes": [
                {"episode": p[0], "tier": p[1], "avg_ratio": p[2]} for p in picks
            ],
        }

    # ---- CSV + JSON --------------------------------------------------------
    csv_path = os.path.join(args.out_dir, "check2_summary.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
        w.writeheader()
        w.writerows(csv_rows)
    json_path = os.path.join(args.out_dir, "check2_summary.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n================ SUMMARY (Check 2 pooled timestep, hard) ================")
    hdr = f"{'layer':>5} {'ts_CV':>7} {'ep_CV':>7} {'x':>5} {'mean':>7} {'std':>7} {'p50':>7} {'zero%':>7} {'nGen':>5}"
    print(hdr)
    for r in csv_rows:
        print(f"{r['layer']:>5} {r['ts_cv']:>7.3f} {r['episode_cv']:>7.3f} "
              f"{r['ts_cv_over_ep_cv']:>5.2f} {r['ts_mean']:>7.4f} {r['ts_std']:>7.4f} "
              f"{r['ts_p50']:>7.4f} {r['frac_zero_mass']*100:>6.2f}% {r['n_general_features']:>5}")
    print(f"\n[done] csv -> {csv_path}\n       json -> {json_path}")


if __name__ == "__main__":
    main()
