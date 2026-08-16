"""Negative control for the Path A headline: is `partial | both` = +0.493 above its floor?

A4's claim is that a feature's causal BREADTH on G-1 tasks predicts its causal importance on
the held-out task, beyond causal magnitude and beyond base firing rate. The standing
objection is mechanical: C is a non-negative [tasks x features] matrix, PR and magnitude are
both functions of it, so perhaps ANY such matrix yields a positive partial and the number
says nothing about the model. This script answers that with permutation floors computed from
the saved attribution npz. Pure re-analysis, no GPU, seconds to a minute.

WHICH PERMUTATION IS VALID (this matters -- the obvious one is a no-op)
----------------------------------------------------------------------
EXPERIMENT_PLAN.md §3.2b prescribes "permute task labels and recompute the A4 partial",
expecting a collapse to ~0. Permuting task labels WITHIN a feature (independently shuffling
each column of C over the task rows) does NOT do that. It is very nearly an identity
operation on this statistic, and here is why:

  LOTO already evaluates EVERY fold. For feature j, holding out task g contributes the pair
  (PR of C[!=g, j], C[g, j]). Permuting column j by pi_j makes fold g contribute the pair
  that fold pi_j(g) contributed in the real data -- the same G pairs, dealt into different
  folds. Since the fold-level partials are all positive and similar in size, mixing folds
  reproduces the statistic almost exactly.

Measured on synthetic data with real breadth structure: real +0.164, task-label permutation
+0.164. Running the prescribed control would therefore have returned "the null equals the
result" and looked like a catastrophic failure, when in fact the permutation destroys
nothing. Two floors that DO test something:

  * `column_shuffle` (the one that matters). Independently permute FEATURE identity within
    each task row. Task marginals are preserved exactly, and so is the purely mechanical
    within-column link (a column whose values are evenly spread still has a predictable
    held-out entry), but a feature no longer has an identity across tasks. Whatever survives
    is arithmetic, not biology. On the same synthetic data this drops +0.164 -> +0.018.
    Note base_rate stays attached to its original index and so becomes an uninformative
    control under this null; partialling out a useless covariate removes LESS, which biases
    the floor UPWARD. The floor is therefore conservative.

  * `feature_shuffle` (estimator floor). Permute feature identity of the held-out vector
    only, leaving the predictor and both controls intact. Nothing links predictor to target,
    so anything but ~0 would indicate a bug in the estimator itself.

Both nulls call `mrvla.attribution.loto_partial_both`, the exact function `run_attribution.py`
uses for the reported number, so there is no chance of scoring a re-implementation.

Usage
-----
python permutation_null.py --attr $B/ATTR/goal_k100/layer_31_attribution.npz --n-perm 1000
python permutation_null.py --attr goal=$B/ATTR/goal_k100/layer_31_attribution.npz \
                           --attr spatial=$B/ATTR/spatial_k100/layer_31_attribution.npz
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

from mrvla.attribution import loto_partial_both, participation_ratio, total_magnitude


def observed_statistic(C: np.ndarray, base_rate: np.ndarray) -> dict:
    """The reported A4 numbers, recomputed from the saved matrix."""
    folds = loto_partial_both(C, base_rate)
    return {"partial_both": float(folds.mean()) if folds.size else float("nan"),
            "n_folds": int(folds.size),
            "n_positive": int((folds > 0).sum()),
            "min_fold": float(folds.min()) if folds.size else float("nan"),
            "folds": [float(v) for v in folds]}


def column_shuffle(C: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Permute feature identity independently within each task row.

    Preserves every task's marginal distribution of causal mass and the mechanical
    within-column spread/held-out relationship; destroys a feature's identity across tasks.
    """
    return np.stack([rng.permutation(row) for row in np.asarray(C, dtype=np.float64)])


def row_shuffle(C: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Permute task labels independently within each feature (the plan's control).

    Provided so the no-op can be DEMONSTRATED rather than asserted; see the module docstring.
    Do not report this as a negative control.
    """
    C = np.asarray(C, dtype=np.float64)
    return np.stack([rng.permutation(C[:, j]) for j in range(C.shape[1])], axis=1)


def feature_shuffle_partials(C: np.ndarray, base_rate: np.ndarray,
                             rng: np.random.Generator) -> np.ndarray:
    """LOTO partials with the held-out vector's feature identity permuted (estimator floor).

    Predictor (training PR) and both controls keep their true values; only the target is
    decoupled, which is the cleanest possible "no relationship" reference.
    """
    from mrvla.attribution import rank_partial_both
    C = np.asarray(C, dtype=np.float64)
    base_rate = np.asarray(base_rate, dtype=np.float64)
    G = C.shape[0]
    vals = []
    for gi in range(G):
        keep = np.arange(G) != gi
        PR_tr = participation_ratio(C[keep])
        mag_tr = total_magnitude(C[keep])
        m = (mag_tr > 0) & np.isfinite(PR_tr)
        if m.sum() > 4:
            held = C[gi][m]
            v = rank_partial_both(rng.permutation(held), PR_tr[m], mag_tr[m], base_rate[m])
            if np.isfinite(v):
                vals.append(v)
    return np.array(vals, dtype=np.float64)


def run_null(C, base_rate, kind: str, n_perm: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = np.empty(n_perm, dtype=np.float64)
    for i in range(n_perm):
        if kind == "feature_shuffle":
            folds = feature_shuffle_partials(C, base_rate, rng)
        else:
            Cp = column_shuffle(C, rng) if kind == "column_shuffle" else row_shuffle(C, rng)
            folds = loto_partial_both(Cp, base_rate)
        out[i] = folds.mean() if folds.size else np.nan
    return out[np.isfinite(out)]


def summarize(name: str, obs: float, null: np.ndarray) -> dict:
    """One-sided p (how often the null reaches the observed value) plus a z-score."""
    if null.size == 0:
        return {"name": name, "n": 0}
    p = float((null >= obs).mean())
    sd = float(null.std())
    return {"name": name, "n": int(null.size), "null_mean": float(null.mean()),
            "null_sd": sd, "null_p95": float(np.percentile(null, 95)),
            "p_one_sided": p,
            # (obs - floor) / floor sd: how many null sds the result clears the floor by
            "z": float((obs - null.mean()) / sd) if sd > 0 else float("inf")}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--attr", action="append", required=True,
                    help="layer_NN_attribution.npz, optionally 'name=path'. Repeatable.")
    ap.add_argument("--n-perm", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--show-invalid-null", action="store_true",
                    help="also run the task-label (row) permutation, to demonstrate on THIS "
                         "data that it is a no-op and must not be used as a control")
    ap.add_argument("--out", default=None, help="output json")
    args = ap.parse_args()

    results: dict = {"n_perm": args.n_perm, "suites": {}}
    for spec in args.attr:
        name, sep, path = spec.partition("=")
        if not sep:
            name, path = os.path.basename(os.path.dirname(path or name)), name
        A = np.load(path)
        C = A["C"].astype(np.float64)
        base_rate = A["base_rate"].astype(np.float64)
        obs = observed_statistic(C, base_rate)

        print(f"\n[null] {name}  C={C.shape[0]} tasks x {C.shape[1]} features")
        print(f"[null] OBSERVED partial|both = {obs['partial_both']:+.4f}   "
              f"({obs['n_positive']}/{obs['n_folds']} folds positive, "
              f"min {obs['min_fold']:+.3f})")

        entry = {"path": path, "observed": obs, "nulls": {}}
        kinds = ["column_shuffle", "feature_shuffle"]
        if args.show_invalid_null:
            kinds.append("row_shuffle")
        for kind in kinds:
            null = run_null(C, base_rate, kind, args.n_perm, args.seed)
            s = summarize(kind, obs["partial_both"], null)
            entry["nulls"][kind] = s
            label = {"column_shuffle": "column shuffle (mechanical floor)",
                     "feature_shuffle": "feature shuffle (estimator floor)",
                     "row_shuffle": "task-label perm (INVALID -- expect a no-op)"}[kind]
            print(f"[null]   {label:44s} mean={s['null_mean']:+.4f} "
                  f"sd={s['null_sd']:.4f}  p95={s['null_p95']:+.4f}  "
                  f"p={s['p_one_sided']:.4f}  z={s['z']:+.1f}")
        if args.show_invalid_null:
            rs = entry["nulls"]["row_shuffle"]["null_mean"]
            print(f"[null]   NOTE task-label permutation reproduces the observed value "
                  f"({rs:+.4f} vs {obs['partial_both']:+.4f}) -- it permutes nothing this "
                  f"statistic depends on. See the module docstring.")
        results["suites"][name] = entry

    if args.out:
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n[null] wrote {args.out}")


if __name__ == "__main__":
    main()
