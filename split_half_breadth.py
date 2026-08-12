"""Path A reliability gate: is causal breadth a STABLE property of a feature, or an artefact
of which tasks we happened to measure it on?

Breadth is a participation ratio over a model's OWN task set. That makes it task-set-relative
by construction, and at G = 10 tasks it is a coarse statistic (PR ranges 1..10). Before
spending rollout budget on ablation, we need to know whether the ranking it induces is
reproducible at all.

The test is the psychometric split-half. Per suite:

  1. split the G task ROWS of C into two disjoint halves A and B,
  2. recompute breadth INDEPENDENTLY on each half (PR and magnitude are both functions of C,
     so both are recomputed -- reusing the full-set magnitude would leak the held-out half),
  3. Spearman-correlate the two breadth rankings over features active in BOTH halves,
  4. repeat over many random splits and report the distribution.

Two things come out of it:

  * rho_split -- how much the ranking agrees with itself across disjoint task sets.
  * Spearman-Brown r_full = 2*rho/(1+rho) -- the implied reliability of the FULL G-task
    breadth. Half-length measurements are pessimistic; this corrects for that, and it is the
    number needed to disattenuate any downstream correlation involving breadth.

A label-shuffle control (permute feature identity in half B) pins the floor at rho ~ 0.

WHAT THE OUTCOMES MEAN
  * high everywhere        -> breadth is a stable feature property; ablation is worth running.
  * high in a homogeneous suite (spatial: layout-only variation), low in a diverse one
    (libero-10: ten distinct long-horizon tasks) -> commonality is genuinely TASK-SET-
    RELATIVE. That is a result, not a caveat: it says the quantity is "common within this
    distribution", not "general" in any transfer sense.
  * low everywhere         -> breadth is largely noise at G = 10, and the expensive
    experiments downstream are not worth running. This gate is meant to be able to fail.

KNOWN LIMITATION (stated, not hidden). PR and magnitude are recomputed per half, but
base_rate is a global firing rate accumulated over all tasks in A1 and cannot be split
without re-streaming the shards. So the ADJUSTED curve carries a mild base-rate leak. The RAW
PR curve uses no confound at all and is completely leak-free -- report both, and treat raw as
the conservative number.

Pure re-analysis of existing attribution npz. No GPU.

Usage
-----
python split_half_breadth.py \
    --attr goal=$B/ATTR/goal_k100/layer_31_attribution.npz \
    --attr spatial=$B/ATTR/spatial_k100/layer_31_attribution.npz \
    --attr object=$B/ATTR/object_k100/layer_31_attribution.npz \
    --attr 10=$B/ATTR/10_k100/layer_31_attribution.npz \
    --n-splits 200 --out $B/ATTR/split_half_breadth.png
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

from identify_features import adjusted_breadth
from mrvla.attribution import participation_ratio, total_magnitude


def _avg_ranks(x: np.ndarray) -> np.ndarray:
    """Ranks with TIES AVERAGED (fractional ranking).

    This matters a great deal here: breadth is a participation ratio, and every feature that
    drives exactly one task has PR == 1.0 exactly, so ties are common. Ordinal ranking would
    break those ties in arbitrary order, and arbitrary order in half A is uncorrelated with
    arbitrary order in half B -- which would deflate the reliability estimate for a purely
    numerical reason. It also makes a constant vector rank as constant, so a degenerate input
    correctly yields NaN rather than a spurious correlation.
    """
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = np.arange(n, dtype=np.float64)
    xs = x[order]
    i = 0
    while i < n:
        j = i
        while j + 1 < n and xs[j + 1] == xs[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = 0.5 * (i + j)
        i = j + 1
    return ranks


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Rank correlation with tied ranks averaged. NaN if fewer than 3 usable pairs, or if
    either side is constant (no variance to correlate)."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3:
        return float("nan")
    ra = _avg_ranks(a[m])
    rb = _avg_ranks(b[m])
    ra -= ra.mean(); rb -= rb.mean()
    den = np.sqrt((ra * ra).sum() * (rb * rb).sum())
    return float((ra * rb).sum() / den) if den > 0 else float("nan")


def spearman_brown(rho: float, n: float = 2.0) -> float:
    """Reliability of an n-times-longer measurement given the half-length correlation.

    r_full = n*rho / (1 + (n-1)*rho); with n=2 this is the classic 2*rho/(1+rho). Undefined
    (NaN) at rho <= -1/(n-1), and we do not report corrections for negative rho -- a negative
    split-half correlation means "no reliable signal", and the correction is meaningless there.
    """
    if not np.isfinite(rho) or rho <= 0:
        return float("nan")
    return float(n * rho / (1.0 + (n - 1.0) * rho))


def half_breadth(C_half: np.ndarray, base_rate: np.ndarray, active: np.ndarray):
    """Breadth computed from one half of the tasks only.

    Returns (PR, magnitude, adjusted) where BOTH PR and magnitude come from C_half -- carrying
    the full-set magnitude across would leak the held-out tasks into the confound control.
    """
    PR = participation_ratio(C_half)
    mag = total_magnitude(C_half)
    adj = adjusted_breadth(PR, mag, base_rate, active & (mag > 0) & np.isfinite(PR))
    return PR, mag, adj


def split_half_rho(C, base_rate, active, n_splits=200, seed=0,
                   adjusted=True, shuffle=False):
    """Distribution of split-half rank agreement over `n_splits` random task partitions.

    C is [G, F]. Each split partitions the G task rows into two disjoint halves, recomputes
    breadth on each, and correlates the two rankings over features active in BOTH halves.
    `shuffle=True` permutes feature identity on the B side -> the chance floor.
    Returns an array [n_splits] of Spearman rho (NaN entries possible and expected to be rare).
    """
    C = np.asarray(C, dtype=np.float64)
    G = C.shape[0]
    if G < 4:
        raise SystemExit(f"need at least 4 tasks to split, got {G}")
    rng = np.random.default_rng(seed)
    out = np.full(n_splits, np.nan)
    for i in range(n_splits):
        idx = rng.permutation(G)
        A, B = idx[:G // 2], idx[G // 2:]
        PR_a, mag_a, adj_a = half_breadth(C[A], base_rate, active)
        PR_b, mag_b, adj_b = half_breadth(C[B], base_rate, active)
        # a feature must be load-bearing in BOTH halves for its two scores to be comparable
        m = (active & (mag_a > 0) & (mag_b > 0)
             & np.isfinite(PR_a) & np.isfinite(PR_b))
        if m.sum() < 3:
            continue
        x = (adj_a if adjusted else PR_a)[m]
        y = (adj_b if adjusted else PR_b)[m]
        if shuffle:
            y = rng.permutation(y)
        out[i] = spearman(x, y)
    return out


def summarize(rhos: np.ndarray) -> dict:
    r = rhos[np.isfinite(rhos)]
    if r.size == 0:
        return {"n": 0, "median": None, "q25": None, "q75": None, "sb_median": None}
    med = float(np.median(r))
    return {"n": int(r.size), "median": med,
            "q25": float(np.percentile(r, 25)), "q75": float(np.percentile(r, 75)),
            "sb_median": spearman_brown(med)}


def load_attr(path: str) -> dict:
    A = np.load(path)
    C = A["C"].astype(np.float64)                     # [G, F]
    return {"C": C, "base_rate": A["base_rate"].astype(np.float64),
            "active": A["is_active"].astype(bool), "G": int(C.shape[0]),
            "F": int(C.shape[1])}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--attr", action="append", required=True,
                   help="label=attribution_npz ; repeat per suite")
    p.add_argument("--n-splits", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True, help="output png (json written alongside)")
    args = p.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    suites = {}
    for spec in args.attr:
        label, path = spec.split("=", 1)
        suites[label] = load_attr(path)

    summary = {"n_splits": args.n_splits, "seed": args.seed, "suites": {}}
    raw_box, adj_box, shuf_box, labels = [], [], [], []

    for label, d in suites.items():
        kw = dict(n_splits=args.n_splits, seed=args.seed)
        r_raw = split_half_rho(d["C"], d["base_rate"], d["active"], adjusted=False, **kw)
        r_adj = split_half_rho(d["C"], d["base_rate"], d["active"], adjusted=True, **kw)
        r_shf = split_half_rho(d["C"], d["base_rate"], d["active"], adjusted=True,
                               shuffle=True, **kw)
        # Collinearity governs how to read the ADJUSTED panel: adjusted breadth is PR with the
        # magnitude/base-rate component projected out, so if PR and magnitude are nearly
        # collinear there is little left to be reliable about and a low adjusted rho says
        # "the confounds explain breadth here", NOT "the measurement is broken".
        PR_full = participation_ratio(d["C"])
        mag_full = total_magnitude(d["C"])
        am = d["active"] & np.isfinite(PR_full)
        rho_pm = spearman(PR_full[am], mag_full[am])
        summary["suites"][label] = {
            "G": d["G"], "F": d["F"], "n_active": int(d["active"].sum()),
            "rho_PR_vs_magnitude": rho_pm,
            "raw_PR": summarize(r_raw), "adjusted": summarize(r_adj),
            "shuffle_floor": summarize(r_shf),
        }
        raw_box.append(r_raw[np.isfinite(r_raw)])
        adj_box.append(r_adj[np.isfinite(r_adj)])
        shuf_box.append(r_shf[np.isfinite(r_shf)])
        labels.append(label)
        s = summary["suites"][label]
        print(f"[split-half] {label:9s} G={d['G']:2d}  "
              f"raw rho={s['raw_PR']['median']:+.3f} (SB {s['raw_PR']['sb_median'] or float('nan'):.3f})  "
              f"adj rho={s['adjusted']['median']:+.3f} (SB {s['adjusted']['sb_median'] or float('nan'):.3f})  "
              f"floor={s['shuffle_floor']['median']:+.3f}  "
              f"[PR~mag {rho_pm:+.2f}]")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
    pos = np.arange(len(labels))
    for ax, data, title in ((ax1, raw_box, "Raw PR  (no confound, leak-free)"),
                            (ax2, adj_box, "Adjusted breadth  (the ranking we select on)")):
        ax.boxplot(data, positions=pos, widths=0.55, showfliers=False)
        ax.boxplot(shuf_box, positions=pos, widths=0.25, showfliers=False,
                   boxprops=dict(color="#bbb"), medianprops=dict(color="#bbb"),
                   whiskerprops=dict(color="#bbb"), capprops=dict(color="#bbb"))
        ax.axhline(0, ls="--", color="#888", lw=1.0)
        ax.set_xticks(pos); ax.set_xticklabels(labels)
        ax.set_title(title, fontsize=10); ax.grid(alpha=0.25, axis="y")
    ax1.set_ylabel("split-half Spearman rho")
    fig.suptitle("Is causal breadth stable across disjoint task halves?  "
                 "(grey = feature-label-shuffle floor)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(args.out, dpi=140); plt.close(fig)

    jout = os.path.splitext(args.out)[0] + ".json"
    with open(jout, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[split-half] wrote {args.out} and {jout}")
    print("[split-half] READ: high rho => breadth is a stable feature property, ablation is\n"
          "[split-half] worth running. High in a homogeneous suite but low in a diverse one\n"
          "[split-half] => commonality is task-set-relative. Low everywhere => breadth is\n"
          "[split-half] mostly noise at this G, and the GPU budget should not be spent.")


if __name__ == "__main__":
    main()
