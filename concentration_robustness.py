#!/usr/bin/env python3
"""Uncertainty and robustness for the P2 concentration numbers.

WHAT WAS MISSING
----------------
`causal_concentration.py` reports point estimates with no uncertainty and one arbitrary
choice of N, and it scores cross-task overlap against an ANALYTIC chance baseline of N^2/F
with F = 2048. Three consequences, all of which this script addresses:

1. NO ERROR BARS. Nothing licenses comparing suites ("goal is more concentrated than object"),
   and nothing says whether n_eff = 102.3 is 102 +- 3 or 102 +- 40.

2. TOP-N IS AN UNREPORTED KNOB. The overlap is reported at N = 50 only. If the ratio to chance
   collapses at N = 10 or N = 200, that is a fact about the result, not a detail.

3. THE CHANCE POOL MAY BE WRONG. N^2/F assumes each task draws its top-N uniformly from all F
   features. If far fewer than F features are ever causally active, the true pool is smaller,
   chance overlap is larger, and the reported ratio is inflated -- at N = 50 an observed 41.4
   reads as 33.9x against a pool of 2048 but only 9.9x against a pool of 600.

THE FIX FOR (3): STOP ARGUING ABOUT THE POOL, MEASURE IT
--------------------------------------------------------
The column shuffle already used elsewhere in P2 permutes feature identity within each task row.
It preserves every task's marginal distribution of causal mass, its active set, its ties, and
its exact multiset of values, while destroying which feature holds which value. The mean
pairwise overlap of the SHUFFLED matrix is therefore the overlap this data produces when there
is no cross-task feature identity -- an empirical chance baseline that needs no assumption
about the size or membership of the pool, because it inherits both from the data.

That is the ratio this script reports. N^2/F is printed alongside as a reference only.

UNCERTAINTY
-----------
Delete-one-task jackknife: recompute every statistic on the G-1 remaining tasks and use the
standard jackknife variance. Two honest limitations, stated rather than buried:

* It captures BETWEEN-TASK variability only. The saved artefact is the [G, F] matrix, so
  within-task sampling noise (which episodes, which timesteps) is not recoverable here and the
  intervals are correspondingly optimistic. Answering "would a different set of episodes give
  this?" needs a re-run of `run_attribution.py` emitting per-episode partials.
* The jackknife is consistent for smooth functionals -- n_eff, Gini, top-shares all qualify --
  but top-N OVERLAP is a step function of the data (a feature is in the set or it is not), and
  the jackknife is known to understate variance for non-smooth statistics. Its leave-one-out
  RANGE is therefore reported beside the interval, and the shuffle null carries the actual
  "is it above chance" claim.

CPU only, seconds per suite, runs on the saved attribution npz.

    python concentration_robustness.py --attr goal=$B/ATTR/goal_k100/layer_31_attribution.npz \
                                       --out $B/DIAGNOSTICS/concentration_robustness
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

from causal_concentration import gini, n_effective, top_share

TOP_NS = (10, 25, 50, 100, 200, 400)


def active_count(C: np.ndarray) -> int:
    """Features carrying non-zero causal mass anywhere -- the real pool a top-N is drawn from."""
    return int((np.asarray(C, dtype=np.float64).sum(axis=0) > 0).sum())


def overlap_mean(C: np.ndarray, n: int) -> float:
    """Mean pairwise intersection size of per-task top-n feature sets."""
    C = np.asarray(C, dtype=np.float64)
    G, F = C.shape
    n = min(n, F)
    tops = [set(np.argsort(row)[::-1][:n].tolist()) for row in C]
    pairs = [len(tops[a] & tops[b]) for a in range(G) for b in range(a + 1, G)]
    return float(np.mean(pairs)) if pairs else float("nan")


def union_top_n(C: np.ndarray, n: int) -> int:
    """Size of the union of all per-task top-n sets.

    Directly interpretable and assumption-free: G tasks x top-n each spans at most G*n
    features. A union near n means one shared coalition; a union near G*n means G private ones.
    """
    C = np.asarray(C, dtype=np.float64)
    u: set[int] = set()
    for row in C:
        u |= set(np.argsort(row)[::-1][:min(n, C.shape[1])].tolist())
    return len(u)


def column_shuffle(C: np.ndarray, rng: np.random.Generator,
                   restrict_to_active: bool = True) -> np.ndarray:
    """Permute feature identity within each task row (marginals preserved exactly).

    `restrict_to_active` confines the permutation to features that carry mass SOMEWHERE in the
    suite, leaving structurally dead features dead. This is not a detail -- it is the whole
    point. A permutation over all F columns scatters mass into features the SAE never uses,
    which is a state no task could ever produce, and the resulting chance overlap collapses
    back to N^2/F. That reintroduces precisely the unjustified pool assumption the empirical
    baseline exists to avoid: with 300 of 2048 features active it reports chance 1.20 when the
    true value is 8.33, inflating the reported ratio sevenfold.

    Set False only to reproduce the naive baseline for comparison.
    """
    C = np.asarray(C, dtype=np.float64)
    if not restrict_to_active:
        return np.stack([rng.permutation(row) for row in C])
    act = np.flatnonzero(C.sum(axis=0) > 0)
    out = np.zeros_like(C)
    for g, row in enumerate(C):
        out[g, act] = rng.permutation(row[act])
    return out


def shuffled_overlap_null(C, n: int, n_perm: int, seed: int,
                          restrict_to_active: bool = True) -> np.ndarray:
    """Empirical chance overlap: what this data gives with cross-task identity destroyed.

    Because the permutation is confined to the active support, the pool size is inherited from
    the data rather than assumed -- no argument about whether the denominator should be F or
    the active count is needed, and ties and the marginal distribution are handled for free.
    """
    rng = np.random.default_rng(seed)
    return np.array([overlap_mean(column_shuffle(C, rng, restrict_to_active), n)
                     for _ in range(n_perm)], dtype=np.float64)


def jackknife(stat, C: np.ndarray) -> dict:
    """Delete-one-task jackknife of `stat(C_subset)`. Returns estimate, SE, interval, range.

    SE uses the standard inflation factor sqrt((G-1)/G * sum of squared deviations), which
    corrects for the fact that leave-one-out replicates are far less variable than independent
    samples would be. See the module docstring for when this is trustworthy.
    """
    C = np.asarray(C, dtype=np.float64)
    G = C.shape[0]
    full = float(stat(C))
    loo = np.array([float(stat(C[np.arange(G) != g])) for g in range(G)], dtype=np.float64)
    loo = loo[np.isfinite(loo)]
    if loo.size < 3:
        return {"estimate": full, "se": float("nan"), "lo": float("nan"),
                "hi": float("nan"), "loo_min": float("nan"), "loo_max": float("nan")}
    se = float(np.sqrt((loo.size - 1) / loo.size * ((loo - loo.mean()) ** 2).sum()))
    return {"estimate": full, "se": se, "lo": full - 1.96 * se, "hi": full + 1.96 * se,
            "loo_min": float(loo.min()), "loo_max": float(loo.max())}


def analyse(name: str, C: np.ndarray, args) -> dict:
    C = np.asarray(C, dtype=np.float64)
    G, F = C.shape
    n_act = active_count(C)

    conc = {
        "n_eff": jackknife(lambda X: n_effective(X.sum(axis=0)), C),
        "gini": jackknife(lambda X: gini(X.sum(axis=0)), C),
        "share_top10": jackknife(lambda X: top_share(X.sum(axis=0), 10), C),
        "share_top50": jackknife(lambda X: top_share(X.sum(axis=0), 50), C),
    }

    sweep = []
    for n in TOP_NS:
        if n > n_act:
            continue
        obs = overlap_mean(C, n)
        null = shuffled_overlap_null(C, n, args.n_perm, args.seed)
        nm, ns = float(null.mean()), float(null.std())
        jk = jackknife(lambda X, _n=n: overlap_mean(X, _n), C)
        sweep.append({
            "n": n, "observed": obs,
            "jk_se": jk["se"], "jk_lo": jk["lo"], "jk_hi": jk["hi"],
            "loo_min": jk["loo_min"], "loo_max": jk["loo_max"],
            "shuffle_chance": nm, "shuffle_sd": ns,
            "ratio_vs_shuffle": float(obs / nm) if nm > 0 else float("nan"),
            "z_vs_shuffle": float((obs - nm) / ns) if ns > 0 else float("inf"),
            "analytic_chance_F": n * n / F,
            "ratio_vs_analytic_F": float(obs / (n * n / F)),
            "analytic_chance_active": n * n / n_act,
            "ratio_vs_analytic_active": float(obs / (n * n / n_act)),
            "union": union_top_n(C, n), "union_max": G * n,
        })

    return {"name": name, "n_tasks": G, "n_features": F, "n_active": n_act,
            "active_fraction": n_act / F, "sae_k": args.sae_k,
            "concentration": conc, "top_n_sweep": sweep}


def report(res: dict) -> None:
    print(f"\n=== {res['name']}  ({res['n_tasks']} tasks x {res['n_features']} features) ===")
    print(f"  causally active features: {res['n_active']} of {res['n_features']} "
          f"({100*res['active_fraction']:.1f}%)   [SAE k = {res['sae_k']}]")
    ne = res["concentration"]["n_eff"]["estimate"]
    if res["n_active"] < 4 * ne:
        print(f"  !! n_active ({res['n_active']}) is close to n_eff ({ne:.1f}): the dictionary "
              f"is barely used, and concentration is near-trivial.")

    print(f"\n  {'statistic':12s} {'estimate':>10s} {'jk SE':>8s} {'95% interval':>20s}")
    for k, v in res["concentration"].items():
        print(f"  {k:12s} {v['estimate']:10.3f} {v['se']:8.3f} "
              f"  [{v['lo']:8.3f}, {v['hi']:8.3f}]")

    print(f"\n  top-N overlap sweep (chance measured by column shuffle, not assumed):")
    print(f"  {'N':>4s} {'obs':>7s} {'jk SE':>7s} {'shuffled':>9s} {'ratio':>7s} {'z':>7s}"
          f" {'N^2/F ratio':>12s} {'union':>12s}")
    for s in res["top_n_sweep"]:
        print(f"  {s['n']:4d} {s['observed']:7.1f} {s['jk_se']:7.2f} {s['shuffle_chance']:9.2f} "
              f"{s['ratio_vs_shuffle']:6.1f}x {s['z_vs_shuffle']:7.1f} "
              f"{s['ratio_vs_analytic_F']:11.1f}x {s['union']:6d}/{s['union_max']:<5d}")
    print("  ratio  = observed / column-shuffled chance  <- the defensible number")
    print("  N^2/F  = the analytic baseline currently in results.md, for comparison")
    print("  union  = distinct features across all per-task top-N sets, out of G*N possible")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--attr", action="append", required=True, metavar="NAME=PATH")
    ap.add_argument("--out", default="out/concentration_robustness")
    ap.add_argument("--n-perm", type=int, default=200,
                    help="column shuffles for the empirical chance baseline")
    ap.add_argument("--sae-k", type=int, default=100,
                    help="the SAE's TopK k, printed for comparison against n_eff")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    all_res = []
    for item in args.attr:
        if "=" not in item:
            raise SystemExit(f"--attr expects NAME=PATH, got {item!r}")
        name, path = item.split("=", 1)
        C = np.load(path, allow_pickle=True)["C"].astype(np.float64)
        res = analyse(name, C, args)
        report(res)
        all_res.append(res)

    with open(os.path.join(args.out, "concentration_robustness.json"), "w") as f:
        json.dump({"suites": all_res, "config": vars(args)}, f, indent=2)
    print(f"\n[conc-rob] wrote {os.path.join(args.out, 'concentration_robustness.json')}")
    print("[conc-rob] Read: if ratio_vs_shuffle stays high across the whole N sweep and the\n"
          "[conc-rob] jackknife intervals are tight, the P2 reproducibility claim holds and\n"
          "[conc-rob] should be restated against the shuffle baseline rather than N^2/F.")


if __name__ == "__main__":
    main()
