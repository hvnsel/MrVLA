#!/usr/bin/env python3
"""Is the A4 control plane rich enough? -- the linearity check on the headline statistic.

THE CONCERN
-----------
`partial | both` residualises ranked breadth and the ranked held-out row on the PLANE spanned
by ranked causal magnitude and ranked base firing rate. A plane removes only confounds that are
linear in each control and additive between them. Participation ratio is bounded above by the
task count while magnitude is not, so a ceiling effect bending the relationship in rank space is
a concrete possibility. Any curvature the plane cannot represent survives residualisation and is
counted as signal -- the bias runs TOWARD the reported result, so it has to be ruled out rather
than assumed away.

The tests in tests/test_rankbasis.py show this is not hypothetical: a fixture whose ONLY
structure is a curved confound produces `partial | both` = +0.40 under linear control, the same
order as the +0.493 this paper reports.

WHAT IS RUN
-----------
1. The identical estimator under progressively richer control bases (linear -> tensor4, the
   last of which can absorb any smooth surface, additive or not).
2. A PLACEBO at matched degrees of freedom -- the same column count drawn at random. This is
   the load-bearing control: extra columns lower an in-sample partial even when they are pure
   noise, so a drop only implicates curvature if it exceeds what junk columns already cost.
3. `curvature_gain`: how much extra variance the nonlinear terms explain when predicting the
   predictor and the target. Near zero means the plane was already adequate and no basis could
   have changed the answer.
4. A feature-shuffle floor under the RICHEST basis, confirming the enriched estimator is still
   unbiased (a richer basis with a nonzero floor would be its own problem).
5. `loto_stratified`: plain rank correlation inside 2-D magnitude x base-rate cells. Assumes no
   functional form whatsoever. Attenuated by construction, so it corroborates the sign.

READING THE OUTPUT
------------------
    excess(basis) = placebo(matched df) - partial(basis)

i.e. how far this basis pushes the number BELOW what the same number of junk columns already
costs. Scored over every basis and reported at its maximum, because a rich basis has enough
degrees of freedom to partly re-fit the very curvature it removed -- in the curved smoke
fixture `quad` moves the number by 0.18 while `tensor4` moves it by only 0.02, so trusting the
richest rung alone would have missed it.

  max excess ~ 0    -> the plane was adequate; report the linear number, cite this as a footnote.
  max excess large  -> curvature was inflating the headline; the most conservative number over
                       the ladder is the honest one and the paper must be rewritten around it.

The POWER of this check -- that it fires when curvature is genuinely present -- is established
in tests/test_rankbasis.py, where a purely curved confound yields +0.40 under the linear plane
and under 0.12 under every enriched basis.

CPU only. Runs on a login node in seconds per suite.

    python control_linearity.py --attr goal=out/goal/layer_31_attribution.npz \
                                --attr spatial=out/spatial/layer_31_attribution.npz \
                                --out out/linearity
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

from mrvla.rankbasis import (
    SPECS, control_design, curvature_gain, loto_partial_design, loto_partial_placebo,
    loto_stratified, orthonormal_basis, rank_partial_design, rank_unit,
)
from mrvla.stats import tie_fraction


def _load(path: str):
    z = np.load(path, allow_pickle=True)
    return np.asarray(z["C"], dtype=np.float64), np.asarray(z["base_rate"], dtype=np.float64)


def _extra_df(C, base_rate, spec) -> int:
    """Effective columns the spec adds beyond linear, on this data (post rank-deficiency drop)."""
    return int(curvature_gain(C, base_rate, spec)["extra_df"])


def _placebo(C, base_rate, n_extra: int, n_rep: int, seed: int) -> tuple[float, float]:
    vals = [float(loto_partial_placebo(C, base_rate, n_extra,
                                       np.random.default_rng(seed + i)).mean())
            for i in range(n_rep)]
    return float(np.mean(vals)), float(np.std(vals))


def _feature_shuffle_floor(C, base_rate, spec, n_perm, seed) -> tuple[float, float]:
    """Estimator floor under `spec`: decouple the target only, keep predictor and controls."""
    from mrvla.attribution import participation_ratio, total_magnitude
    rng = np.random.default_rng(seed)
    C = np.asarray(C, dtype=np.float64)
    G = C.shape[0]
    out = []
    for _ in range(n_perm):
        vals = []
        for gi in range(G):
            keep = np.arange(G) != gi
            PR_tr, mag_tr = participation_ratio(C[keep]), total_magnitude(C[keep])
            m = (mag_tr > 0) & np.isfinite(PR_tr)
            if m.sum() > 4:
                v = rank_partial_design(rng.permutation(C[gi][m]), PR_tr[m],
                                        [mag_tr[m], base_rate[m]], spec)
                if np.isfinite(v):
                    vals.append(v)
        if vals:
            out.append(float(np.mean(vals)))
    a = np.array(out, dtype=np.float64)
    return (float(a.mean()), float(a.std())) if a.size else (float("nan"), float("nan"))


def analyse(name: str, C, base_rate, args) -> dict:
    rows, richest = [], args.richest
    for spec in SPECS:
        folds = loto_partial_design(C, base_rate, spec)
        cg = curvature_gain(C, base_rate, spec)
        df = cg["extra_df"]
        pm, ps = _placebo(C, base_rate, df, args.n_placebo, args.seed) if df else (
            float(folds.mean()), 0.0)
        rows.append({
            "spec": spec, "extra_df": df,
            "partial": float(folds.mean()) if folds.size else float("nan"),
            "n_folds": int(folds.size), "n_positive": int((folds > 0).sum()),
            "min_fold": float(folds.min()) if folds.size else float("nan"),
            "placebo_partial": pm, "placebo_sd": ps,
            "delta_r2_predictor": cg["delta_r2_predictor"],
            "delta_r2_target": cg["delta_r2_target"],
            "folds": [float(v) for v in folds],
        })

    lin = rows[0]["partial"]
    rich = next(r for r in rows if r["spec"] == richest)
    # Excess is scored across the WHOLE ladder, not just the richest rung. A basis can both
    # remove curvature and, with more degrees of freedom, partly re-fit it, so the richest
    # basis is not reliably the most revealing one -- in the curved smoke fixture `quad` moves
    # the number by 0.18 while `tensor4` moves it by 0.02. The conservative reading is the
    # largest disagreement any reasonable control basis produces.
    for r in rows:
        r["excess"] = r["placebo_partial"] - r["partial"]
    worst = max(rows, key=lambda r: r["excess"])
    excess = worst["excess"]
    floor_m, floor_s = _feature_shuffle_floor(C, base_rate, richest, args.n_perm, args.seed)
    strat = loto_stratified(C, base_rate, args.n_bins, args.min_cell)

    # tie exposure on the controls, for the record (the shipped estimator breaks ties by index)
    ties = {"base_rate": tie_fraction(base_rate),
            "magnitude": tie_fraction(C.sum(axis=0))}

    verdict = ("PLANE ADEQUATE" if excess < args.tol else
               "CURVATURE MATERIAL -- report the conservative number")
    return {"name": name, "n_tasks": int(C.shape[0]), "n_features": int(C.shape[1]),
            "rows": rows, "richest": richest, "linear": lin, "richest_partial": rich["partial"],
            "max_excess": float(excess), "max_excess_spec": worst["spec"],
            "min_partial": float(min(r["partial"] for r in rows)),
            "min_partial_spec": min(rows, key=lambda r: r["partial"])["spec"],
            "tolerance": args.tol, "verdict": verdict,
            "floor_richest_mean": floor_m, "floor_richest_sd": floor_s,
            "z_vs_floor_richest": (float((rich["partial"] - floor_m) / floor_s)
                                   if floor_s > 0 else float("nan")),
            "stratified": strat, "tie_fraction": ties}


def report(res: dict) -> None:
    print(f"\n=== {res['name']}  ({res['n_tasks']} tasks x {res['n_features']} features) ===")
    print(f"{'basis':9s} {'+df':>4s} {'partial':>9s} {'placebo':>9s} {'excess':>8s} "
          f"{'dR2(pred)':>10s} {'dR2(targ)':>10s}  folds+")
    lin = res["linear"]
    for r in res["rows"]:
        print(f"{r['spec']:9s} {r['extra_df']:4d} {r['partial']:+9.4f} "
              f"{r['placebo_partial']:+9.4f} {r['excess']:+8.4f} "
              f"{r['delta_r2_predictor']:10.5f} {r['delta_r2_target']:10.5f}  "
              f"{r['n_positive']}/{r['n_folds']}")
    print(f"\n  linear {lin:+.4f}  ->  {res['richest']} {res['richest_partial']:+.4f}   "
          f"| worst basis {res['max_excess_spec']}: excess {res['max_excess']:+.4f} "
          f"(tolerance {res['tolerance']:.3f})")
    print(f"  most conservative partial over the ladder: {res['min_partial']:+.4f} "
          f"({res['min_partial_spec']})")
    print(f"  feature-shuffle floor under {res['richest']}: "
          f"{res['floor_richest_mean']:+.4f} (sd {res['floor_richest_sd']:.4f})  "
          f"z = {res['z_vs_floor_richest']:+.1f}")
    s = res["stratified"]
    print(f"  stratified (no functional form): {s['mean']:+.4f}  "
          f"{s['n_positive']}/{s['n_folds']} folds +, worst {s['min_fold']:+.4f}, "
          f"{s['mean_cells_used']:.0f} cells/fold")
    print(f"  tie fraction  base_rate={res['tie_fraction']['base_rate']:.3f}  "
          f"magnitude={res['tie_fraction']['magnitude']:.3f}")
    print(f"  VERDICT: {res['verdict']}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--attr", action="append", required=True, metavar="NAME=PATH",
                    help="repeatable: suite name = path to layer_XX_attribution.npz")
    ap.add_argument("--out", default="out/linearity")
    ap.add_argument("--richest", default="tensor4", choices=list(SPECS))
    ap.add_argument("--n-placebo", type=int, default=25,
                    help="random-column draws averaged for the df calibration")
    ap.add_argument("--n-perm", type=int, default=200,
                    help="feature shuffles for the enriched estimator's floor")
    ap.add_argument("--n-bins", type=int, default=5, help="quantile bins per control")
    ap.add_argument("--min-cell", type=int, default=20, help="min features for a cell to count")
    ap.add_argument("--tol", type=float, default=0.05,
                    help="excess drop below this counts the plane as adequate")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    all_res = []
    for item in args.attr:
        if "=" not in item:
            raise SystemExit(f"--attr expects NAME=PATH, got {item!r}")
        name, path = item.split("=", 1)
        C, br = _load(path)
        res = analyse(name, C, br, args)
        report(res)
        all_res.append(res)

    with open(os.path.join(args.out, "linearity.json"), "w") as f:
        json.dump({"suites": all_res, "config": vars(args)}, f, indent=2)

    worst = max(all_res, key=lambda r: r["max_excess"])
    print(f"\n[linearity] wrote {os.path.join(args.out, 'linearity.json')}")
    print(f"[linearity] largest excess anywhere: {worst['name']} / {worst['max_excess_spec']} "
          f"{worst['max_excess']:+.4f}  -> {worst['verdict']}")
    print("[linearity] Read: 'excess' is the part of a basis's drop below the LINEAR number\n"
          "[linearity] that random columns of the same count do NOT explain -- i.e. curvature\n"
          "[linearity] the plane could not see. Scored over every basis, not just the richest,\n"
          "[linearity] because extra degrees of freedom can re-fit what they just removed.\n"
          "[linearity] All excesses near zero => the control plane was already adequate.")


if __name__ == "__main__":
    main()
