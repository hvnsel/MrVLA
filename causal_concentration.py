"""How concentrated, and how reproducible across tasks, is causal influence over features?

results.md §A6 answers the "of course hundreds of features influence the action" objection
with: *"Influence is not the claim; CONCENTRATION and REPRODUCIBILITY of influence is."*
Neither word currently has a number attached to it. This script supplies both from the saved
attribution matrix. Pure re-analysis, no GPU, seconds.

CONCENTRATION -- how few features carry the decision.
  * `n_eff` = (sum_j m_j)^2 / sum_j m_j^2 over per-feature total causal magnitude: the
    effective number of features carrying the action, on the same participation-ratio scale
    Path A already uses for tasks. Read against F = 2048 and against the SAE's k.
  * Lorenz shares: the fraction of total causal mass held by the top 1%, top 10, top 50, and
    top 100 features, plus a Gini coefficient.
  * The same statistics computed per task, so "concentrated" is not an artefact of averaging
    ten differently-shaped tasks together.

REPRODUCIBILITY -- whether it is the SAME few features across tasks.
  Concentration alone is compatible with every task recruiting its own private top-50. We
  measure the mean pairwise overlap of per-task top-N sets and express it as a RATIO TO
  CHANCE (two independent top-N draws from F features overlap by N^2/F in expectation). A
  ratio of 1 means a different coalition per task; high ratios mean one shared coalition.

CONTROLS. Every statistic is also computed for a base-firing-rate ranking (the prior work's
activity proxy) and for a column-shuffled matrix. If causal mass concentrates no more than
firing rate does, "concentration" is an activity statement, not a causal one -- the same
confound-first discipline the rest of the project applies.

Usage
-----
python causal_concentration.py --attr goal=$B/ATTR/goal_k100/layer_31_attribution.npz \
                               --attr spatial=$B/ATTR/spatial_k100/layer_31_attribution.npz \
                               --top 50 --out $B/ATTR/concentration.json
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np


def n_effective(v: np.ndarray) -> float:
    """Participation ratio over features: the effective number carrying the mass."""
    v = np.clip(np.asarray(v, dtype=np.float64), 0, None)
    s1, s2 = v.sum(), (v * v).sum()
    return float(s1 * s1 / s2) if s2 > 0 else float("nan")


def top_share(v: np.ndarray, n: int) -> float:
    """Fraction of the total held by the n largest entries."""
    v = np.clip(np.asarray(v, dtype=np.float64), 0, None)
    tot = v.sum()
    if tot <= 0 or n <= 0:
        return float("nan")
    n = min(n, v.size)
    return float(np.sort(v)[::-1][:n].sum() / tot)


def gini(v: np.ndarray) -> float:
    """Gini coefficient of a non-negative vector. 0 = perfectly even, ->1 = all in one."""
    v = np.sort(np.clip(np.asarray(v, dtype=np.float64), 0, None))
    n = v.size
    tot = v.sum()
    if n == 0 or tot <= 0:
        return float("nan")
    idx = np.arange(1, n + 1)
    return float((2 * (idx * v).sum()) / (n * tot) - (n + 1) / n)


def topn_overlap_vs_chance(C: np.ndarray, n: int) -> tuple[float, float, float]:
    """(mean pairwise top-n overlap, chance overlap, ratio) across task rows of C.

    Chance is n^2/F, the expected intersection of two independent uniform n-subsets of F
    features -- the right baseline because both sets are the same size by construction.
    """
    C = np.asarray(C, dtype=np.float64)
    G, F = C.shape
    n = min(n, F)
    tops = [set(np.argsort(row)[::-1][:n].tolist()) for row in C]
    pairs = [len(tops[a] & tops[b]) for a in range(G) for b in range(a + 1, G)]
    if not pairs:
        return float("nan"), float("nan"), float("nan")
    obs = float(np.mean(pairs))
    chance = n * n / F
    return obs, chance, float(obs / chance) if chance > 0 else float("nan")


def profile(v: np.ndarray, tops: list[int]) -> dict:
    return {"n_eff": n_effective(v), "gini": gini(v),
            **{f"share_top{n}": top_share(v, n) for n in tops},
            "share_top1pct": top_share(v, max(1, v.size // 100))}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--attr", action="append", required=True,
                    help="layer_NN_attribution.npz, optionally 'name=path'. Repeatable.")
    ap.add_argument("--top", type=int, default=50, help="N for the cross-task overlap test")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    tops = [10, 50, 100]
    results: dict = {"top_n_overlap": args.top, "suites": {}}

    for spec in args.attr:
        name, sep, path = spec.partition("=")
        if not sep:
            name, path = os.path.basename(os.path.dirname(path or name)), name
        A = np.load(path)
        C = A["C"].astype(np.float64)                    # [G, F]
        base_rate = A["base_rate"].astype(np.float64)
        active = A["is_active"].astype(bool)
        G, F = C.shape
        mag = C.sum(axis=0)

        causal = profile(mag, tops)
        firing = profile(np.where(active, base_rate, 0.0), tops)
        rng = np.random.default_rng(args.seed)
        Cs = np.stack([rng.permutation(row) for row in C])
        shuffled = profile(Cs.sum(axis=0), tops)

        obs, chance, ratio = topn_overlap_vs_chance(C, args.top)
        per_task = [profile(C[g], tops) for g in range(G)]

        entry = {"path": path, "n_tasks": G, "n_features": F,
                 "n_active": int(active.sum()),
                 "causal": causal, "firing_rate_control": firing,
                 "column_shuffled_control": shuffled,
                 "per_task_n_eff": [p["n_eff"] for p in per_task],
                 "per_task_share_top50": [p["share_top50"] for p in per_task],
                 "topn_overlap": {"n": args.top, "observed": obs, "chance": chance,
                                  "ratio_to_chance": ratio}}
        results["suites"][name] = entry

        print(f"\n[conc] {name}: {G} tasks x {F} features ({int(active.sum())} active)")
        print(f"[conc] CONCENTRATION (per-feature total causal magnitude)")
        print(f"[conc]   effective #features carrying the action : {causal['n_eff']:.1f} "
              f"of {F}  ({100*causal['n_eff']/F:.1f}%)")
        print(f"[conc]   top-10 / top-50 / top-100 share of mass : "
              f"{causal['share_top10']:.3f} / {causal['share_top50']:.3f} / "
              f"{causal['share_top100']:.3f}")
        print(f"[conc]   gini                                    : {causal['gini']:.3f}")
        print(f"[conc]   CONTROL base firing rate  n_eff={firing['n_eff']:.1f} "
              f"top50={firing['share_top50']:.3f} gini={firing['gini']:.3f}")
        print(f"[conc]   CONTROL column-shuffled   n_eff={shuffled['n_eff']:.1f} "
              f"top50={shuffled['share_top50']:.3f} gini={shuffled['gini']:.3f}")
        if np.isfinite(causal["gini"]) and np.isfinite(firing["gini"]):
            print("[conc]   -> causal mass is "
                  + ("MORE" if causal["gini"] > firing["gini"] else "NOT more")
                  + " concentrated than firing rate is.")
        print(f"[conc] REPRODUCIBILITY across tasks (top-{args.top} sets)")
        print(f"[conc]   mean pairwise overlap {obs:.1f} of {args.top}   "
              f"chance {chance:.2f}   ratio {ratio:.1f}x")
        pt = np.array([p["n_eff"] for p in per_task], dtype=np.float64)
        print(f"[conc]   per-task n_eff  min {np.nanmin(pt):.1f}  median "
              f"{np.nanmedian(pt):.1f}  max {np.nanmax(pt):.1f}   "
              f"(concentration is not an averaging artefact if these are all small)")

    if args.out:
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n[conc] wrote {args.out}")


if __name__ == "__main__":
    main()
