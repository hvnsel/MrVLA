"""Step 3 — Confound audit for the per-episode generality score.

The per-episode (soft, trimmed) generality score has non-trivial spread.  This
script asks the question that decides whether reweighting on it means anything:
is that spread GENERALITY, or is it nuisance structure wearing a generality
costume?

Two score constructions are audited (--score-mode, default 'both'):

    mass   sum_j z_j p_j / sum_j z_j   magnitude-weighted (the check-1 soft
           score).  Its numerator is dominated by the near-constant always-on
           high-P features, so it behaves like const/total_mass and inherits
           total-mass variance mechanically.
    count  mean of p over the ACTIVE features (z > 0) — identity only.  With
           a TopK SAE the denominator is the constant K, so total activation
           mass cannot enter by construction.

For each layer and mode we regress the score on every episode property we can
measure that is *not* generality:

    T_full      episode length in frames
    task_id     goal identity (LIBERO-Goal task) — as one-hot factors
    mean_mass   mean total SAE activation mass per frame (trimmed window)
    mean_l0     mean # active features per frame (trimmed window)
    home_frac   fraction of trimmed-window frames whose code is near-identical
                (cosine) to the episode's home/start frame — idle-frame proxy

and report:

    * univariate Pearson/Spearman of score vs each continuous confound
    * eta^2 of task_id (between-goal variance fraction)
    * full-OLS R^2 (all confounds together) + drop-one partial R^2
    * the RESIDUAL score (confounds regressed out), saved per episode
    * reliability: split-half (even/odd frames and first/second half) with
      Spearman-Brown correction and ICC, for BOTH the raw score and the
      residual score.  The residual split-half reliability is the decisive
      number: a reliable, confound-free component is what earns episode-level
      reweighting.  (Even/odd overestimates reliability via temporal
      autocorrelation; first/second underestimates it via nonstationarity —
      the truth lives between the two.)

Verdict heuristics printed per layer (thresholds are advisory, not gates):
    confound R^2 >= 0.8            -> score is mostly nuisance
    residual even/odd r_SB >= 0.5  -> a reliable residual component survives

Usage
-----
python mrvla/confound_audit.py \
    --codes-dir      E:/libero_goal_demos/codes_v3 \
    --generality-dir E:/libero_goal_demos/generality_v3 \
    --out-dir        E:/libero_goal_demos/check3_confound_audit \
    --layers 0,8,16,24,31
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from mrvla.episode_generality_variance import (
        EPS,
        apply_coverage_floor,
        load_coverage,
        per_timestep_count_ratios,
        per_timestep_ratios,
    )
except ImportError:  # run directly as `python mrvla/confound_audit.py`
    from episode_generality_variance import (
        EPS,
        apply_coverage_floor,
        load_coverage,
        per_timestep_count_ratios,
        per_timestep_ratios,
    )


HOME_COSINE_THRESHOLD = 0.9   # frame counts as "at home" above this similarity
HOME_REF_FRAMES = 3           # home reference = mean code of first N frames


# ---------------------------------------------------------------------------
# Small stats helpers (numpy only — no scipy dependency)
# ---------------------------------------------------------------------------
def rankdata(x: np.ndarray) -> np.ndarray:
    """Average ranks (1-based), ties get their mean rank."""
    order = np.argsort(x, kind="stable")
    ranks = np.empty(len(x), dtype=np.float64)
    ranks[order] = np.arange(1, len(x) + 1, dtype=np.float64)
    # average ties
    vals, inv, counts = np.unique(x, return_inverse=True, return_counts=True)
    sums = np.bincount(inv, weights=ranks)
    return (sums / counts)[inv]


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    xc, yc = x - x.mean(), y - y.mean()
    denom = np.sqrt((xc ** 2).sum() * (yc ** 2).sum())
    if denom <= 0:
        return 0.0
    return float((xc * yc).sum() / denom)


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    return pearson(rankdata(x), rankdata(y))


def spearman_brown(r: float) -> float:
    """Reliability of the full-length score from a half-length correlation."""
    if r <= -1.0:
        return -1.0
    return 2.0 * r / (1.0 + r)


def icc_two_halves(a: np.ndarray, b: np.ndarray) -> float:
    """ICC(1) for k=2 'raters' (the two half-scores) per episode."""
    x = np.stack([a, b], axis=1).astype(np.float64)   # [E, 2]
    grand = x.mean()
    row_means = x.mean(axis=1)
    msb = 2.0 * ((row_means - grand) ** 2).sum() / max(len(x) - 1, 1)
    msw = ((x - row_means[:, None]) ** 2).sum() / len(x)
    denom = msb + msw
    if denom <= 0:
        return 0.0
    return float((msb - msw) / denom)


def eta_squared(y: np.ndarray, groups: np.ndarray) -> float:
    """Fraction of variance of y explained by group membership."""
    y = np.asarray(y, dtype=np.float64)
    ss_tot = ((y - y.mean()) ** 2).sum()
    if ss_tot <= 0:
        return 0.0
    ss_between = 0.0
    for g in np.unique(groups):
        yg = y[groups == g]
        ss_between += len(yg) * (yg.mean() - y.mean()) ** 2
    return float(ss_between / ss_tot)


def zscore(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    sd = x.std()
    return (x - x.mean()) / (sd + EPS)


def ols_fit(X: np.ndarray, y: np.ndarray):
    """Least squares y ~ X. Returns (r2, adj_r2, residuals, coef)."""
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ coef
    resid = y - pred
    ss_res = float((resid ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / max(ss_tot, EPS)
    n, p = X.shape
    dof = max(n - p, 1)
    adj_r2 = 1.0 - (1.0 - r2) * (n - 1) / dof
    return r2, adj_r2, resid, coef


def build_design(continuous: dict[str, np.ndarray], task: np.ndarray):
    """Design matrix: intercept + z-scored continuous confounds + task one-hots
    (first task dropped as reference).  Returns (X [E, p], column names,
    slices per named block for drop-one partial R^2)."""
    E = len(task)
    cols = [np.ones(E, dtype=np.float64)]
    names = ["intercept"]
    blocks: dict[str, list[int]] = {}
    for name, v in continuous.items():
        blocks[name] = [len(cols)]
        cols.append(zscore(v))
        names.append(name)
    tasks = np.unique(task)
    blocks["task_id"] = []
    for t in tasks[1:]:
        blocks["task_id"].append(len(cols))
        cols.append((task == t).astype(np.float64))
        names.append(f"task_{t}")
    return np.stack(cols, axis=1), names, blocks


# ---------------------------------------------------------------------------
# Per-episode score + confound extraction
# ---------------------------------------------------------------------------
def episode_table(
    z: np.ndarray,
    ratio_t: np.ndarray,
    episode: np.ndarray,
    timestep: np.ndarray,
    task_id: np.ndarray,
    trim_frac: float,
) -> dict[str, np.ndarray]:
    """One row per episode: trimmed score, split-half scores, and confounds.

    All quantities that feed the score (score, halves, mean_mass, mean_l0,
    home_frac) are computed on the same trimmed window; T_full is the raw
    episode length.  home_frac uses the mean code of the episode's first
    HOME_REF_FRAMES frames as the home reference, so it measures idle frames
    that SURVIVE the trim.
    """
    ep_ids = np.unique(episode)
    E = len(ep_ids)
    out = {k: np.zeros(E, dtype=np.float64) for k in (
        "score", "half_even", "half_odd", "half_first", "half_second",
        "T_full", "mean_mass", "mean_l0", "home_frac")}
    out["episode"] = ep_ids.astype(np.int64)
    out["task"] = np.zeros(E, dtype=np.int64)

    for i, ep in enumerate(ep_ids):
        mask = episode == ep
        order = np.argsort(timestep[mask])
        r = ratio_t[mask][order]
        z_ep = z[mask][order].astype(np.float32)
        tids = task_id[mask]
        out["task"][i] = int(np.bincount(tids).argmax())

        T = len(r)
        out["T_full"][i] = T
        k = int(np.floor(trim_frac * T))
        if T - 2 * k < 2:                       # need >=2 frames to split
            k = 0
        w = slice(k, T - k)
        rw, zw = r[w], z_ep[w]
        n = len(rw)

        out["score"][i] = rw.mean()
        idx = np.arange(n)
        out["half_even"][i] = rw[idx % 2 == 0].mean()
        out["half_odd"][i] = rw[idx % 2 == 1].mean()
        out["half_first"][i] = rw[: n // 2].mean()
        out["half_second"][i] = rw[n // 2 :].mean()

        mass = zw.sum(axis=1)
        out["mean_mass"][i] = mass.mean()
        out["mean_l0"][i] = (zw > 0).sum(axis=1).mean()

        home = z_ep[:HOME_REF_FRAMES].mean(axis=0)
        hnorm = np.linalg.norm(home)
        fnorm = np.linalg.norm(zw, axis=1)
        cos = (zw @ home) / (np.maximum(fnorm * hnorm, EPS))
        out["home_frac"][i] = float((cos > HOME_COSINE_THRESHOLD).mean())

    return out


# ---------------------------------------------------------------------------
# Audit for one layer
# ---------------------------------------------------------------------------
CONTINUOUS_CONFOUNDS = ("T_full", "mean_mass", "mean_l0", "home_frac")


def audit_layer(tab: dict[str, np.ndarray]) -> dict:
    score = tab["score"]
    task = tab["task"]
    continuous = {k: tab[k] for k in CONTINUOUS_CONFOUNDS}

    stats: dict = {"n_episodes": int(len(score)),
                   "score_mean": float(score.mean()),
                   "score_std": float(score.std()),
                   "univariate": {}, "task_eta2": eta_squared(score, task)}

    for name, v in continuous.items():
        stats["univariate"][name] = {
            "pearson": pearson(score, v),
            "spearman": spearman(score, v),
            "r2": pearson(score, v) ** 2,
        }

    X, names, blocks = build_design(continuous, task)
    r2, adj_r2, resid, _ = ols_fit(X, score)
    stats["ols"] = {"r2": r2, "adj_r2": adj_r2,
                    "n_predictors": int(X.shape[1] - 1),
                    "residual_std": float(resid.std())}

    # drop-one partial R^2 per confound block
    partial = {}
    for name, cols in blocks.items():
        keep = [j for j in range(X.shape[1]) if j not in cols]
        r2_wo, _, _, _ = ols_fit(X[:, keep], score)
        partial[name] = r2 - r2_wo
    stats["partial_r2"] = partial

    # reliability of raw and residual score
    def _rel(a: np.ndarray, b: np.ndarray) -> dict:
        r = pearson(a, b)
        return {"r_half": r, "r_spearman_brown": spearman_brown(r),
                "icc": icc_two_halves(a, b)}

    _, _, res_even, _ = ols_fit(X, tab["half_even"])
    _, _, res_odd, _ = ols_fit(X, tab["half_odd"])
    _, _, res_first, _ = ols_fit(X, tab["half_first"])
    _, _, res_second, _ = ols_fit(X, tab["half_second"])

    stats["reliability"] = {
        "raw_even_odd": _rel(tab["half_even"], tab["half_odd"]),
        "raw_first_second": _rel(tab["half_first"], tab["half_second"]),
        "residual_even_odd": _rel(res_even, res_odd),
        "residual_first_second": _rel(res_first, res_second),
    }

    # NOTE on the residual reliability flag: when the confounds explain most
    # of the full score, the first- and second-half residuals are FORCED to
    # be nearly equal-and-opposite (they must average to the tiny full-score
    # residual), and even/odd residuals agree trivially because interleaved
    # halves both track the full score.  A genuinely stable confound-free
    # trait therefore has to show BOTH high even/odd reliability AND a
    # non-negative first/second ICC — a strongly negative first/second value
    # is the fingerprint of the forced anticorrelation, not of signal.
    stats["verdict"] = {
        "mostly_confound": bool(r2 >= 0.8),
        "reliable_residual": bool(
            stats["reliability"]["residual_even_odd"]["r_spearman_brown"]
            >= 0.5
            and stats["reliability"]["residual_first_second"]["icc"] > 0.0),
    }
    return stats, resid


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_layer(layer_idx: int, tab: dict, stats: dict, resid: np.ndarray,
               out_path: str, mode: str = "mass"):
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    score = tab["score"]
    ylabel = f"{mode} soft trimmed score"

    for ax, name in zip(axes.flat, CONTINUOUS_CONFOUNDS):
        u = stats["univariate"][name]
        ax.scatter(tab[name], score, s=8, alpha=0.5, color="#4c72b0")
        ax.set_xlabel(name)
        ax.set_ylabel(ylabel)
        ax.set_title(f"score vs {name}\n"
                     f"pearson={u['pearson']:.3f}  spearman={u['spearman']:.3f}",
                     fontsize=10)

    ax = axes.flat[4]
    tasks = np.unique(tab["task"])
    ax.boxplot([score[tab["task"] == t] for t in tasks])
    ax.set_xticks(range(1, len(tasks) + 1))
    ax.set_xticklabels([str(t) for t in tasks])
    ax.set_xlabel("task_id")
    ax.set_ylabel(ylabel)
    ax.set_title(f"score by goal   eta^2={stats['task_eta2']:.3f}", fontsize=10)

    ax = axes.flat[5]
    rel = stats["reliability"]
    ax.scatter(tab["half_even"], tab["half_odd"], s=8, alpha=0.5,
               color="#55a868", label="raw halves")
    ax.set_xlabel("even-frame half score")
    ax.set_ylabel("odd-frame half score")
    ax.set_title(
        f"split-half   raw r_SB={rel['raw_even_odd']['r_spearman_brown']:.3f}"
        f"   residual r_SB="
        f"{rel['residual_even_odd']['r_spearman_brown']:.3f}", fontsize=10)
    ax.legend(fontsize=8)

    fig.suptitle(
        f"Layer {layer_idx:02d} confound audit ({mode} score)  |  OLS R^2="
        f"{stats['ols']['r2']:.3f} (adj {stats['ols']['adj_r2']:.3f})  |  "
        f"residual std={stats['ols']['residual_std']:.5f}",
        fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
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
    ap.add_argument("--out-dir", default="E:/libero_goal_demos/check3_confound_audit")
    ap.add_argument("--layers", default="0,8,16,24,31")
    ap.add_argument("--coverage-floor", type=float, default=0.1)
    ap.add_argument("--trim-frac", type=float, default=0.10)
    ap.add_argument("--score-mode", default="both",
                    choices=("mass", "count", "both"),
                    help="mass  = magnitude-weighted soft ratio (sum z*p / "
                         "sum z; mechanically coupled to total mass); "
                         "count = mass-robust soft ratio (mean p over active "
                         "features; identity only). 'both' audits the two "
                         "side by side.")
    args = ap.parse_args()

    layers = [int(x) for x in args.layers.split(",") if x.strip() != ""]
    modes = ["mass", "count"] if args.score_mode == "both" else [args.score_mode]
    os.makedirs(args.out_dir, exist_ok=True)

    summary = {"codes_dir": args.codes_dir,
               "generality_dir": args.generality_dir,
               "score": "soft trimmed (coverage-floored)",
               "score_modes": modes,
               "coverage_floor": args.coverage_floor,
               "trim_frac": args.trim_frac,
               "home_cosine_threshold": HOME_COSINE_THRESHOLD,
               "layers": {}}

    for layer_idx in layers:
        print(f"\n=== layer {layer_idx:02d} ===", flush=True)
        c = np.load(os.path.join(args.codes_dir, f"layer_{layer_idx:02d}.npz"))
        g = np.load(os.path.join(
            args.generality_dir, f"layer_{layer_idx:02d}_generality.npz"))

        z = c["z"].astype(np.float32)
        episode = c["episode"].astype(np.int64)
        timestep = c["timestep"].astype(np.int64)
        task_id = c["task_id"].astype(np.int64)
        is_general = g["is_general"].astype(bool)
        prob_general = g["prob_general"].astype(np.float32)

        if args.coverage_floor > 0:
            coverage = load_coverage(args.generality_dir, layer_idx)
            is_general, prob_general, n_floored = apply_coverage_floor(
                is_general, prob_general, coverage, args.coverage_floor)
            print(f"  coverage floor rho={args.coverage_floor}: "
                  f"zeroed soft mass of {n_floored} + "
                  f"{int((~is_general).sum() - n_floored)} features already "
                  f"non-general", flush=True)

        ratios = {}
        if "mass" in modes:
            _, ratios["mass"], *_ = per_timestep_ratios(
                z, is_general, prob_general)
        if "count" in modes:
            _, ratios["count"] = per_timestep_count_ratios(
                z, is_general, prob_general)

        layer_stats = {}
        for mode in modes:
            print(f"  --- score mode: {mode} ---", flush=True)
            tab = episode_table(z, ratios[mode], episode, timestep, task_id,
                                args.trim_frac)
            stats, resid = audit_layer(tab)

            print(f"  episodes={stats['n_episodes']}  "
                  f"score mean={stats['score_mean']:.4f} "
                  f"std={stats['score_std']:.4f}", flush=True)
            print(f"  univariate r (pearson/spearman):", flush=True)
            for name in CONTINUOUS_CONFOUNDS:
                u = stats["univariate"][name]
                print(f"    {name:<10} {u['pearson']:+.3f} / "
                      f"{u['spearman']:+.3f}", flush=True)
            print(f"  task eta^2 = {stats['task_eta2']:.3f}", flush=True)
            print(f"  OLS all-confounds R^2 = {stats['ols']['r2']:.3f} "
                  f"(adj {stats['ols']['adj_r2']:.3f})", flush=True)
            print(f"  partial R^2:", flush=True)
            for name, v in stats["partial_r2"].items():
                if name != "intercept":
                    print(f"    {name:<10} {v:+.4f}", flush=True)
            rel = stats["reliability"]
            print(f"  reliability (r_SB / ICC):", flush=True)
            for key in ("raw_even_odd", "raw_first_second",
                        "residual_even_odd", "residual_first_second"):
                r = rel[key]
                print(f"    {key:<22} {r['r_spearman_brown']:+.3f} / "
                      f"{r['icc']:+.3f}", flush=True)
            v = stats["verdict"]
            print(f"  VERDICT: mostly_confound={v['mostly_confound']}  "
                  f"reliable_residual={v['reliable_residual']}", flush=True)

            out_png = os.path.join(
                args.out_dir,
                f"layer_{layer_idx:02d}_confound_audit_{mode}.png")
            plot_layer(layer_idx, tab, stats, resid, out_png, mode=mode)
            print(f"  wrote {out_png}", flush=True)

            out_npz = os.path.join(
                args.out_dir,
                f"layer_{layer_idx:02d}_confound_audit_{mode}.npz")
            np.savez_compressed(
                out_npz,
                episode=tab["episode"].astype(np.int32),
                task=tab["task"].astype(np.int32),
                score=tab["score"].astype(np.float32),
                score_residual=resid.astype(np.float32),
                half_even=tab["half_even"].astype(np.float32),
                half_odd=tab["half_odd"].astype(np.float32),
                half_first=tab["half_first"].astype(np.float32),
                half_second=tab["half_second"].astype(np.float32),
                T_full=tab["T_full"].astype(np.int32),
                mean_mass=tab["mean_mass"].astype(np.float32),
                mean_l0=tab["mean_l0"].astype(np.float32),
                home_frac=tab["home_frac"].astype(np.float32),
            )
            print(f"  wrote {out_npz}", flush=True)
            layer_stats[mode] = stats

        summary["layers"][f"layer_{layer_idx:02d}"] = layer_stats

    out_summary = os.path.join(args.out_dir, "confound_audit_summary.json")
    with open(out_summary, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[done] summary -> {out_summary}", flush=True)


if __name__ == "__main__":
    main()
