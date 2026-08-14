"""Path B / Part 1 final figure: does cross-model recurrence concentrate in the most
causally-GENERAL features? Rank ALL eligible features by breadth, bin, plot mean recurrence.

The threshold-free, un-shoppable version of the top-N sweep. For each target model we:
  1. rank its features by confound-adjusted breadth (Path A),
  2. bin them into equal-count bins by breadth percentile,
  3. plot mean recurrence (q_cross) per bin, with a +-1 SE band and the CHANCE FLOOR
     (mean permutation q) drawn as a horizontal line,
  4. overlay all target models on one axis,
  5. and report a PERMUTATION p-value per model: how often does the top-decile recurrence
     beat the rest by as much as observed, when breadth labels are shuffled?

Two panels: recurrence (q_cross) vs breadth, and recurrence-BEYOND-inheritance vs breadth
(so a reviewer cannot say the tick-up is the base-rate/inheritance confound sneaking back).

If the curve rises above the floor at the high-breadth end AND the permutation p is small for
all models -> "the most causally-general features recur across models above chance" is real.
If flat / large p -> the top-N effect was noise, and Path B stays the null. Either is clean.

Pure re-analysis of existing npz. No GPU.

Usage
-----
python recurrence_vs_breadth.py \
    --pair goal=$B/ATTR/goal_k100/layer_31_attribution.npz=$B/RECURRENCE_ACTION/layer_31_target_goal.npz \
    --pair spatial=$B/ATTR/spatial_k100/layer_31_attribution.npz=$B/RECURRENCE_ACTION/layer_31_target_spatial.npz \
    --pair object=$B/ATTR/object_k100/layer_31_attribution.npz=$B/RECURRENCE_ACTION/layer_31_target_object.npz \
    --pair 10=$B/ATTR/10_k100/layer_31_attribution.npz=$B/RECURRENCE_ACTION/layer_31_target_libero10.npz \
    --out $B/RECURRENCE_ACTION/recurrence_vs_breadth.png
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

from identify_features import adjusted_breadth
from identify_recurrent_features import recurrence_beyond


def bin_stats(breadth, y, active, n_bins=50):
    """Bin active features into n_bins EQUAL-COUNT bins by ascending breadth; return per-bin
    (percentile center, mean y, standard error). Equal-count bins => none empty, stable means."""
    m = active & np.isfinite(breadth) & np.isfinite(y)
    br, yy = breadth[m], y[m]
    order = np.argsort(br)                       # ascending breadth
    br, yy = br[order], yy[order]
    n = len(br)
    edges = np.linspace(0, n, n_bins + 1).astype(int)
    centers, means, ses = [], [], []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        if hi <= lo:
            continue
        seg = yy[lo:hi]
        centers.append(100.0 * (lo + hi) / 2.0 / n)     # breadth percentile of bin center
        means.append(float(seg.mean()))
        ses.append(float(seg.std(ddof=1) / np.sqrt(len(seg))) if len(seg) > 1 else 0.0)
    return np.array(centers), np.array(means), np.array(ses)


def perm_pvalue(breadth, y, active, top_frac=0.10, n_perm=5000, seed=0):
    """Permutation test: is the top-`top_frac`-breadth mean y elevated above the rest beyond
    chance? Observed gap = mean(y | top breadth) - mean(y | rest). Null shuffles which
    features are 'top' (breadth unrelated to y). Returns (observed_gap, p_value)."""
    m = active & np.isfinite(breadth) & np.isfinite(y)
    br, yy = breadth[m], y[m]
    n = len(br)
    k = max(1, int(round(top_frac * n)))
    order = np.argsort(-br)                       # descending breadth
    top = order[:k]
    rest = order[k:]
    obs = float(yy[top].mean() - yy[rest].mean())
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    ge = 0
    for _ in range(n_perm):
        perm = rng.permutation(idx)
        ge += (yy[perm[:k]].mean() - yy[perm[k:]].mean()) >= obs
    return obs, float((ge + 1) / (n_perm + 1))


def load_pair(attr_path, rec_path):
    A = np.load(attr_path); B = np.load(rec_path)
    if A["PR"].shape != B["q_cross"].shape:
        raise SystemExit(f"feature mismatch {A['PR'].shape} vs {B['q_cross'].shape} "
                         f"({attr_path} vs {rec_path}) -- same action-position SAE?")
    active = A["is_active"].astype(bool) & B["is_active"].astype(bool)
    adj = adjusted_breadth(A["PR"].astype(np.float64), A["magnitude"].astype(np.float64),
                           A["base_rate"].astype(np.float64), active)
    inh = B["inheritance"].astype(np.float64) if "inheritance" in B else None
    rb = recurrence_beyond(B["q_cross"].astype(np.float64), B["base_rate"].astype(np.float64),
                           inh, active)
    floor = float(np.nanmean(B["q_perm"][active])) if "q_perm" in B else None
    return {"breadth": adj, "q_cross": B["q_cross"].astype(np.float64),
            "rec_beyond": rb, "active": active, "floor": floor}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pair", action="append", required=True,
                   help="label=attr_npz=rec_npz ; repeat for each model")
    p.add_argument("--n-bins", type=int, default=50)
    p.add_argument("--top-frac", type=float, default=0.10)
    p.add_argument("--n-perm", type=int, default=5000)
    p.add_argument("--out", required=True, help="output png (json written alongside)")
    args = p.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pairs = {}
    for spec in args.pair:
        label, attr, rec = spec.split("=", 2)
        pairs[label] = load_pair(attr, rec)

    summary = {"n_bins": args.n_bins, "top_frac": args.top_frac, "models": {}}
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6))
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(pairs), 3)))

    for (label, d), col in zip(pairs.items(), colors):
        cx, my, se = bin_stats(d["breadth"], d["q_cross"], d["active"], args.n_bins)
        ax1.plot(cx, my, "-o", color=col, ms=3, label=label)
        ax1.fill_between(cx, my - se, my + se, color=col, alpha=0.15)
        cx2, my2, se2 = bin_stats(d["breadth"], d["rec_beyond"], d["active"], args.n_bins)
        ax2.plot(cx2, my2, "-o", color=col, ms=3, label=label)
        ax2.fill_between(cx2, my2 - se2, my2 + se2, color=col, alpha=0.15)

        obs_q, p_q = perm_pvalue(d["breadth"], d["q_cross"], d["active"],
                                 args.top_frac, args.n_perm)
        obs_r, p_r = perm_pvalue(d["breadth"], d["rec_beyond"], d["active"],
                                 args.top_frac, args.n_perm)
        summary["models"][label] = {
            "floor": d["floor"],
            "top_decile_gap_qcross": obs_q, "perm_p_qcross": p_q,
            "top_decile_gap_recbeyond": obs_r, "perm_p_recbeyond": p_r,
        }
        print(f"[curve] {label:10s} floor={d['floor']:.3f}  "
              f"top-{int(args.top_frac*100)}% q_cross gap={obs_q:+.3f} (p={p_q:.4f})  "
              f"rec_beyond gap={obs_r:+.2f} (p={p_r:.4f})")

    floors = [d["floor"] for d in pairs.values() if d["floor"] is not None]
    if floors:
        ax1.axhline(float(np.mean(floors)), ls="--", color="#888", lw=1.2,
                    label=f"chance floor ≈ {np.mean(floors):.3f}")
    ax2.axhline(0, ls="--", color="#888", lw=1.2)
    ax1.set_xlabel("breadth percentile  (0 = most specialist → 100 = most general)")
    ax1.set_ylabel("mean recurrence  q_cross"); ax1.set_title("Recurrence vs breadth")
    ax2.set_xlabel("breadth percentile")
    ax2.set_ylabel("mean recurrence beyond inheritance (rank-resid)")
    ax2.set_title("Confound-controlled (beyond activity + inheritance)")
    ax1.legend(fontsize=7); ax1.grid(alpha=0.25); ax2.grid(alpha=0.25)
    fig.suptitle("Does cross-model recurrence concentrate in the most causally-general features?",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(args.out, dpi=140); plt.close(fig)

    jout = os.path.splitext(args.out)[0] + ".json"
    with open(jout, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[curve] wrote {args.out} and {jout}")
    print("[curve] READ: rising toward the right AND small perm_p in ALL models => the most\n"
          "[curve] causally-general features recur across models above chance. Flat / large p\n"
          "[curve] => the top-N effect was noise; Path B stays the null.")


if __name__ == "__main__":
    main()
