#!/usr/bin/env python3
"""What IS the recurring coalition? -- two quantitative questions, no frame inspection.

P2 establishes that causal mass is concentrated and that substantially the same features
recur across tasks. A reader will ask what those features are. The useful answer is not a
catalogue -- A4 already showed that individual features are interpretable, and a published list
of names invites an argument that cannot be settled -- but an answer to the two readings of the
question that the P2 controls leave open.

Q1. IS THE COALITION JUST THE FEATURES THAT FIRE MOST OFTEN?
    P2's activity control compares n_eff(mass) against n_eff(base rate) and shows mass is the
    more concentrated of the two. That is a statement about SHAPE, and it says nothing about
    IDENTITY: mass concentrated on exactly the 100 most frequently firing features would
    produce the same two numbers. So the deflationary reading -- "the shared core is just the
    always-on features" -- survives every control in P2. It is tested here directly, by
    intersecting the top-N by causal mass with the top-N by base firing rate.

    This comparison is NOT circular: base rate is a count over decisions and is never used in
    the construction of C.

Q2. IS THE RECURRING CORE BROAD BEYOND WHAT ITS SIZE PREDICTS?
    Path A's axis is adjusted breadth -- participation ratio residualised, in rank space, on
    magnitude and base rate. Because magnitude is projected out by construction, a set selected
    purely on magnitude has no mechanical reason to score high on it. So asking where the
    coalition sits on the adjusted-breadth distribution is a real question with a real null
    (the median), and a positive answer means P1 and P2 are two views of one object rather than
    two adjacent findings.

CIRCULARITY, AND WHY THE COALITION IS DEFINED BY POOLED MASS
    The tempting definition -- "features appearing in the per-task top-N of every task" -- is
    circular for Q2: such a feature is forced to carry mass on every task, which is close to the
    definition of high participation ratio. The coalition is therefore defined as the top-N by
    POOLED mass, a pure magnitude criterion, which leaves both questions non-trivial. The strict
    per-task intersection is still reported, as a description of how stable the core is.

CPU only, seconds. Runs on the saved attribution npz.

    python coalition_identity.py --attr goal=$B/ATTR/goal_k100/layer_31_attribution.npz \
                                 --top 50 --out $B/DIAGNOSTICS/coalition
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

from identify_features import adjusted_breadth
from mrvla.stats import rankdata_average


def top_set(v: np.ndarray, n: int, eligible: np.ndarray) -> set[int]:
    """Indices of the n largest entries of v, restricted to eligible features."""
    idx = np.flatnonzero(eligible)
    order = idx[np.argsort(v[idx])[::-1]]
    return set(order[:min(n, order.size)].tolist())


def overlap_vs_chance(A: set[int], B: set[int], pool: int) -> dict:
    """Intersection size against the hypergeometric expectation |A||B|/pool."""
    obs = len(A & B)
    exp = len(A) * len(B) / pool if pool else float("nan")
    return {"observed": obs, "expected": exp,
            "ratio": float(obs / exp) if exp > 0 else float("nan"),
            "jaccard": float(len(A & B) / len(A | B)) if (A | B) else float("nan")}


def percentile_profile(members: set[int], v: np.ndarray, eligible: np.ndarray) -> dict:
    """Where the members sit in the distribution of v, as percentiles among eligible features.

    Reported as percentiles rather than raw values so the answer is scale-free and has an
    obvious null: a set unrelated to v averages the 50th percentile.
    """
    idx = np.flatnonzero(eligible)
    r = rankdata_average(v[idx]) / max(len(idx) - 1, 1) * 100.0
    pos = {int(f): float(r[k]) for k, f in enumerate(idx) if int(f) in members}
    if not pos:
        return {"n": 0}
    a = np.array(list(pos.values()))
    return {"n": int(a.size), "mean_pct": float(a.mean()), "median_pct": float(np.median(a)),
            "p10": float(np.percentile(a, 10)), "p90": float(np.percentile(a, 90)),
            "frac_above_50": float((a > 50).mean())}


def analyse(name: str, z, n: int) -> dict:
    C = np.asarray(z["C"], dtype=np.float64)
    PR = np.asarray(z["PR"], dtype=np.float64)
    mag = np.asarray(z["magnitude"], dtype=np.float64)
    base = np.asarray(z["base_rate"], dtype=np.float64)
    active = np.asarray(z["is_active"]).astype(bool)
    G, F = C.shape
    pool = int(active.sum())

    adj = adjusted_breadth(PR, mag, base, active)

    coalition = top_set(mag, n, active)                       # top-N by POOLED causal mass
    by_base = top_set(base, n, active)                        # top-N by firing frequency
    by_adj = top_set(np.nan_to_num(adj, nan=-np.inf), n, active)
    by_pr = top_set(PR, n, active)

    # strict core: present in EVERY task's per-task top-N (descriptive only -- see docstring)
    per_task = [top_set(C[g], n, active) for g in range(G)]
    strict = set.intersection(*per_task) if per_task else set()

    return {
        "name": name, "n_tasks": G, "n_features": F, "n_active": pool, "top_n": n,
        "q1_mass_vs_baserate": overlap_vs_chance(coalition, by_base, pool),
        "q1_baserate_profile": percentile_profile(coalition, base, active),
        "q2_mass_vs_adjbreadth": overlap_vs_chance(coalition, by_adj, pool),
        "q2_adjbreadth_profile": percentile_profile(coalition, adj, active),
        "raw_pr_profile": percentile_profile(coalition, PR, active),
        "mass_vs_rawpr": overlap_vs_chance(coalition, by_pr, pool),
        "strict_core_size": len(strict),
        "per_task_top_n_union": len(set().union(*per_task)) if per_task else 0,
    }


def report(r: dict) -> None:
    n, pool = r["top_n"], r["n_active"]
    print(f"\n=== {r['name']}  (top-{n} by pooled causal mass, {pool} eligible features) ===")
    print(f"  strict core (in EVERY task's top-{n}): {r['strict_core_size']} features   "
          f"union across tasks: {r['per_task_top_n_union']}")

    q1, p1 = r["q1_mass_vs_baserate"], r["q1_baserate_profile"]
    print(f"\n  Q1  is the coalition just the most frequently firing features?")
    print(f"      overlap with top-{n} by base rate: {q1['observed']}/{n}  "
          f"(chance {q1['expected']:.2f}, ratio {q1['ratio']:.1f}x, Jaccard {q1['jaccard']:.3f})")
    print(f"      coalition's base-rate percentile: mean {p1['mean_pct']:.1f}, "
          f"median {p1['median_pct']:.1f}, {100*p1['frac_above_50']:.0f}% above the median")
    print(f"      -> {'DEFLATIONARY READING LIVE' if q1['observed'] > 0.5 * n else 'not the firing set'}"
          f"   (null: overlap {q1['expected']:.1f}, percentile 50)")

    q2, p2 = r["q2_mass_vs_adjbreadth"], r["q2_adjbreadth_profile"]
    print(f"\n  Q2  is the coalition broad beyond what its size predicts?")
    print(f"      overlap with top-{n} by ADJUSTED breadth: {q2['observed']}/{n}  "
          f"(chance {q2['expected']:.2f}, ratio {q2['ratio']:.1f}x)")
    print(f"      coalition's adjusted-breadth percentile: mean {p2['mean_pct']:.1f}, "
          f"median {p2['median_pct']:.1f}, {100*p2['frac_above_50']:.0f}% above the median")
    print(f"      -> {'P1 and P2 pick the same features' if p2['mean_pct'] > 60 else 'largely separate axes'}"
          f"   (null: percentile 50, since magnitude is residualised out)")

    pr = r["raw_pr_profile"]
    print(f"\n  context: coalition's RAW participation-ratio percentile mean {pr['mean_pct']:.1f} "
          f"(expected high -- carrying mass on many tasks is close to the definition)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--attr", action="append", required=True, metavar="NAME=PATH")
    ap.add_argument("--top", type=int, default=50)
    ap.add_argument("--out", default="out/coalition")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    res = []
    for item in args.attr:
        if "=" not in item:
            raise SystemExit(f"--attr expects NAME=PATH, got {item!r}")
        name, path = item.split("=", 1)
        r = analyse(name, np.load(path, allow_pickle=True), args.top)
        report(r)
        res.append(r)

    with open(os.path.join(args.out, "coalition_identity.json"), "w") as f:
        json.dump({"suites": res, "config": vars(args)}, f, indent=2)
    print(f"\n[coalition] wrote {os.path.join(args.out, 'coalition_identity.json')}")
    print("[coalition] Read: Q1 near chance kills the 'it is just the always-on features'\n"
          "[coalition] objection outright. Q2 well above the 50th percentile means P1 and P2\n"
          "[coalition] describe one object and the paper should say so; near 50 means they are\n"
          "[coalition] two separate findings and the paper must not imply otherwise.")


if __name__ == "__main__":
    main()
