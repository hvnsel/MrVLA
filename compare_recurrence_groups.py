"""Path B / Part 1: do Path A's GENERAL features recur across models more than its SPECIALISTS?

Instead of asking "do all 2048 features recur" (blind, uninterpretable), we restrict to the
features Path A already identified and compare two named groups:

    top-N causally-GENERAL   (highest confound-adjusted breadth)
    top-N causally-SPECIALIST (lowest, among load-bearing features)

and ask whether the general group recurs across the other fine-tuned models more than the
specialist group. Pure re-analysis of existing outputs -- Path A's attribution npz plus the
per-feature q_cross/inheritance the recurrence run already saved (action-position, same SAE,
so feature indices align). No re-encoding, no GPU.

Reports, per group: mean/median q_cross, mean inheritance, mean recurrence-beyond-inheritance;
the permutation floor for reference; and a common-language effect size
    P(random general recurs more than random specialist)
    0.5 = no difference ; >0.5 = general recurs more ; <0.5 = specialist recurs more.

Usage
-----
python compare_recurrence_groups.py \
    --attr $BASE/ATTR/goal_k100/layer_31_attribution.npz \
    --rec  $BASE/RECURRENCE_ACTION/layer_31_target_goal.npz \
    --top 100 --target goal
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

from identify_features import adjusted_breadth, select_general_specialist
from identify_recurrent_features import recurrence_beyond


def common_language_effect(a: np.ndarray, b: np.ndarray) -> float:
    """P(random a > random b) + 0.5*P(tie), the Mann-Whitney common-language effect size.
    0.5 = groups equal; >0.5 = a tends to exceed b; <0.5 = b tends to exceed a. No scipy."""
    a = np.asarray(a, float); b = np.asarray(b, float)
    a = a[np.isfinite(a)]; b = b[np.isfinite(b)]
    if a.size == 0 or b.size == 0:
        return float("nan")
    gt = (a[:, None] > b[None, :]).sum()
    eq = (a[:, None] == b[None, :]).sum()
    return float((gt + 0.5 * eq) / (a.size * b.size))


def _stats(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    return {"n": int(x.size), "mean": float(np.mean(x)) if x.size else float("nan"),
            "median": float(np.median(x)) if x.size else float("nan")}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--attr", required=True, help="Path A layer_NN_attribution.npz")
    p.add_argument("--rec", required=True, help="Path B action-position layer_NN_target_<m>.npz")
    p.add_argument("--top", type=int, default=100, help="group size N (general and specialist each)")
    p.add_argument("--target", default="", help="model name, for labelling only")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    A = np.load(args.attr)
    B = np.load(args.rec)
    if A["PR"].shape != B["q_cross"].shape:
        raise SystemExit(f"feature-count mismatch {A['PR'].shape} vs {B['q_cross'].shape} -- "
                         "are both on the SAME action-position SAE?")

    PR = A["PR"].astype(np.float64)
    magnitude = A["magnitude"].astype(np.float64)
    a_base = A["base_rate"].astype(np.float64)
    active = A["is_active"].astype(bool) & B["is_active"].astype(bool)

    q_cross = B["q_cross"].astype(np.float64)
    inheritance = B["inheritance"].astype(np.float64) if "inheritance" in B else None
    b_base = B["base_rate"].astype(np.float64)
    q_perm = B["q_perm"].astype(np.float64) if "q_perm" in B else None

    adj = adjusted_breadth(PR, magnitude, a_base, active)
    general, specialist = select_general_specialist(adj, magnitude, active, args.top)
    # recurrence residualised on activity + inheritance, computed over ALL active features
    rec_beyond = recurrence_beyond(q_cross, b_base, inheritance, active)

    g, s = np.array(general), np.array(specialist)
    if inheritance is None:
        print("[cmp] WARNING: no `inheritance` in the recurrence npz -- recurrence-beyond-"
              "inheritance falls back to base-rate only; cannot separate re-derived from "
              "inherited. Re-run run_recurrence with --base-key for the full comparison.", flush=True)

    res = {
        "target": args.target, "top": args.top,
        "n_general": int(g.size), "n_specialist": int(s.size),
        "general": {
            "q_cross": _stats(q_cross[g]),
            "inheritance": _stats(inheritance[g]) if inheritance is not None else None,
            "recurrence_beyond_inheritance": _stats(rec_beyond[g]),
        },
        "specialist": {
            "q_cross": _stats(q_cross[s]),
            "inheritance": _stats(inheritance[s]) if inheritance is not None else None,
            "recurrence_beyond_inheritance": _stats(rec_beyond[s]),
        },
        "effect_general_vs_specialist": {
            "q_cross_CLES": common_language_effect(q_cross[g], q_cross[s]),
            "recurrence_beyond_CLES": common_language_effect(rec_beyond[g], rec_beyond[s]),
        },
        "permutation_floor_mean": float(np.nanmean(q_perm[active])) if q_perm is not None else None,
    }

    def line(name, d):
        s_ = f"  {name:11s} q_cross mean={d['q_cross']['mean']:.3f} median={d['q_cross']['median']:.3f}"
        if d["inheritance"] is not None:
            s_ += f"  inheritance={d['inheritance']['mean']:.3f}"
        s_ += f"  rec_beyond(mean rank-resid)={d['recurrence_beyond_inheritance']['mean']:+.2f}"
        return s_

    print(f"[cmp] target={args.target}  top-{args.top} general vs top-{args.top} specialist "
          f"(action-position, same SAE as Path A)")
    if q_perm is not None:
        print(f"[cmp] permutation floor (chance q_cross) = {res['permutation_floor_mean']:.3f}")
    print(line("GENERAL", res["general"]))
    print(line("SPECIALIST", res["specialist"]))
    cg = res["effect_general_vs_specialist"]["q_cross_CLES"]
    print(f"[cmp] EFFECT  P(general recurs > specialist) = {cg:.3f}   "
          + ("(general recurs MORE)" if cg > 0.55 else
             "(specialist recurs MORE)" if cg < 0.45 else "(no meaningful difference)"))
    print(f"[cmp]   (0.50 = no difference; the join's corr(breadth,q_cross) was -0.127, i.e. "
          "expect ~0.5 or slightly below)")

    out = args.out or os.path.join(os.path.dirname(args.rec),
                                   f"compare_groups_{args.target or 'target'}.json")
    with open(out, "w") as f:
        json.dump(res, f, indent=2)
    print(f"[cmp] wrote {out}")


if __name__ == "__main__":
    main()
