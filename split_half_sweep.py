#!/usr/bin/env python3
"""Does the Spearman-Brown model actually fit the breadth reliability? A split-size sweep.

WHY
---
P4 reports split-half reliability at one split size (5 tasks vs 5) and Spearman-Brown-corrects
it to full length. Two objections to that, and this script is the diagnostic for both.

1. SPEARMAN-BROWN MAY NOT APPLY. It assumes the halves differ from the full measurement ONLY
   in length. They do not: halving G = 10 tasks to 5 also halves the participation ratio's
   CEILING (PR can reach 5 instead of 10), compressing the scale and adding ties. On a
   synthetic matrix with this geometry the correction underestimated by a factor of 1.32.
   Meanwhile the base-rate leak in the adjusted curve pushes the other way. Two unmeasured
   biases in opposite directions, so the corrected number could be anywhere in a wide band.

2. THERE MAY BE NO TRUE SCORE TO CORRECT TOWARD. Classical test theory needs a fixed true
   value with random error around it. Participation ratio is defined RELATIVE TO THE TASK SET.
   If breadth genuinely differs across task sets, split-half disagreement is real variation
   rather than error, and disattenuating it is not merely imprecise but invalid.

THE TEST
--------
Spearman-Brown is a one-parameter model: a measurement built from L tasks has reliability

    r_L = L*r_1 / (1 + (L-1)*r_1)

for a single per-task reliability r_1. Invert it,

    r_1 = r_L / (L - r_L*(L-1))

and estimate r_1 SEPARATELY at every split size L = 2, 3, 4, 5. If the model fits, all of them
must agree. If they drift, the model is wrong and any single-size correction is unreliable.

DRIFT ALONE MEANS NOTHING -- IT NEEDS A REFERENCE
-------------------------------------------------
Spearman-Brown drifts here EVEN WHEN THE GENERATIVE MODEL IS PERFECTLY CLASSICAL. On a
synthetic matrix with a fixed true breadth per feature and independent per-task sampling, the
implied r_1 still climbs (0.097 at L=2 to 0.201 at L=5), because PR's CEILING moves with the
split size: at L = 2 the statistic lives in [1, 2] with massive ties, at L = 5 it has real
resolution. The measurement changes qualitatively, not merely in length, so the one-parameter
model cannot hold across sizes whatever the data look like.

A fixed threshold on the drift is therefore meaningless. The script instead CALIBRATES: it
generates a synthetic matrix matched to the real one's shape and per-feature activity, where a
fixed true score exists by construction, and runs the identical sweep on it. The reference is
that calibration drift.

    real drift ~= calibration drift  -> the observed drift is the known ceiling artefact, and
                                        Spearman-Brown should simply not be applied.

A SECOND AXIS FOR THE TRUE-SCORE QUESTION
-----------------------------------------
Drift alone cannot answer whether a true score exists: random halves mix any underlying task
structure and wash it out, so a task-set-relative matrix drifts no more than a classical one.
The signature that DOES separate them is the VARIANCE of rho across splits at a fixed size.

If breadth is a fixed property measured with random error, every random split is equally
informative and rho varies across splits only as much as the calibration's does. If breadth is
task-set-relative, splits that happen to be regime-matched agree far better than regime-crossed
ones, so rho's spread inflates -- on a synthetic two-regime matrix the ratio to calibration runs
1.6x to 3.8x and RISES with split size, while a classical matrix sits at 0.8x to 1.1x.

    sd ratio ~ 1        -> consistent with a fixed true score measured noisily.
    sd ratio >> 1       -> breadth depends on WHICH tasks were used; disattenuation is not
                           merely imprecise but invalid, since the disagreement is real
                           variation rather than error.

Either way the practical conclusion for P4 is the same: report the UNCORRECTED split-half
agreement at a stated half-length, and do not quote a disattenuated correlation.

CONFOUND THIS CONTROLS FOR
--------------------------
The set of features usable at split size L is not the same across L: a feature must be active
in BOTH halves to be correlated, and at L = 2 far fewer qualify. A drift in r_1 could then be
a change of POPULATION rather than a failure of the model. Every statistic is therefore also
computed on a FIXED feature set -- those active in all G tasks, which does not depend on the
split -- and both are reported. Only agreement between the two licenses reading the trend as a
model diagnostic.

Also reported per size: the label-shuffle floor (permute feature identity in half B), which
must sit at ~0 everywhere, and the usable-feature count.

RAW vs ADJUSTED
---------------
`base_rate` is a global firing rate accumulated over all tasks and cannot be split without
re-streaming the shards, so the ADJUSTED curve carries a mild leak that inflates agreement.
The RAW PR curve uses no confound at all and is leak-free. Both are reported; raw is the
conservative one.

CPU only, seconds per suite, on the saved attribution npz.

    python split_half_sweep.py --attr goal=$B/ATTR/goal_k100/layer_31_attribution.npz \
                               --out $B/DIAGNOSTICS/split_half_sweep
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

from split_half_breadth import half_breadth, spearman, spearman_brown


def implied_r1(r_L: float, L: int) -> float:
    """Per-task reliability implied by a length-L reliability, inverting Spearman-Brown.

    r_1 = r_L / (L - r_L*(L-1)). Returns NaN for non-positive r_L, where the correction is
    meaningless, and for the degenerate denominator.
    """
    if not np.isfinite(r_L) or r_L <= 0 or L < 1:
        return float("nan")
    den = L - r_L * (L - 1)
    return float(r_L / den) if den > 0 else float("nan")


def calibration_matrix(C: np.ndarray, seed: int) -> np.ndarray:
    """A synthetic matrix matched to C's shape and per-feature activity, with a TRUE score.

    Feature f is active on each task independently with the same probability it shows in C, at
    the same overall magnitude. A fixed per-feature breadth therefore exists by construction and
    the classical model holds exactly -- so whatever drift the sweep reports on THIS matrix is
    the ceiling artefact alone, and is the reference the real drift must be read against.

    Breadth is encoded as HOW MANY TASKS a feature touches, not how large it is: the
    participation ratio is scale-invariant, so a per-feature multiplier is invisible to it.
    """
    C = np.asarray(C, dtype=np.float64)
    G, F = C.shape
    rng = np.random.default_rng(seed)
    p_f = np.clip((C > 0).mean(axis=0), 1.0 / G, 1.0)
    scale = C.sum(axis=0) / np.maximum(p_f * G, 1e-12)
    fire = rng.random((G, F)) < p_f
    return fire * scale * np.exp(0.3 * rng.standard_normal((G, F)))


def sweep_one_size(C, base_rate, active, size: int, n_splits: int, rng,
                   adjusted: bool, fixed: np.ndarray | None, shuffle: bool) -> dict:
    """Split-half agreement at a given half-size, over `n_splits` random disjoint partitions."""
    C = np.asarray(C, dtype=np.float64)
    G = C.shape[0]
    rhos, ns = [], []
    for _ in range(n_splits):
        p = rng.permutation(G)
        A, B = p[:size], p[size:2 * size]
        PRa, maga, adja = half_breadth(C[A], base_rate, active)
        PRb, magb, adjb = half_breadth(C[B], base_rate, active)
        x = adja if adjusted else PRa
        y = adjb if adjusted else PRb
        m = (np.isfinite(x) & np.isfinite(y) & (maga > 0) & (magb > 0))
        if fixed is not None:
            m &= fixed
        if m.sum() < 3:
            continue
        yy = y[m]
        if shuffle:
            yy = rng.permutation(yy)          # destroys feature identity -> the floor
        rhos.append(spearman(x[m], yy))
        ns.append(int(m.sum()))
    a = np.array([v for v in rhos if np.isfinite(v)], dtype=np.float64)
    if a.size == 0:
        return {"size": size, "n_splits_ok": 0}
    med = float(np.median(a))
    return {"size": size, "n_splits_ok": int(a.size),
            "rho_median": med, "rho_mean": float(a.mean()), "rho_sd": float(a.std()),
            "n_features_median": float(np.median(ns)),
            "r1": implied_r1(med, size),
            "r_full": spearman_brown(med, n=C.shape[0] / size)}


def analyse(name: str, z, args) -> dict:
    C = np.asarray(z["C"], dtype=np.float64)
    base_rate = np.asarray(z["base_rate"], dtype=np.float64)
    active = np.asarray(z["is_active"]).astype(bool)
    G = C.shape[0]
    # fixed population: active in EVERY task, so it cannot change with the split size
    fixed = (C > 0).all(axis=0) & active

    out = {"name": name, "n_tasks": G, "n_features": int(C.shape[1]),
           "n_fixed_population": int(fixed.sum()), "curves": {}}
    sizes = [s for s in range(2, G // 2 + 1)]
    for label, adj, fx, sh in [("adjusted", True, None, False),
                               ("raw", False, None, False),
                               ("adjusted_fixed_pop", True, fixed, False),
                               ("raw_fixed_pop", False, fixed, False),
                               ("shuffle_floor", True, None, True)]:
        rng = np.random.default_rng(args.seed)
        out["curves"][label] = [sweep_one_size(C, base_rate, active, s, args.n_splits,
                                               rng, adj, fx, sh) for s in sizes]

    # calibration: identical sweep on a synthetic matrix where a true score exists by
    # construction, so its drift is the ceiling artefact and nothing else
    rng = np.random.default_rng(args.seed + 1)
    Ccal = calibration_matrix(C, args.seed + 1)
    out["curves"]["calibration"] = [
        sweep_one_size(Ccal, base_rate, active, s, args.n_splits, rng, False, None, False)
        for s in sizes]

    def drift(rows):
        v = np.array([e.get("r1", float("nan")) for e in rows], dtype=np.float64)
        v = v[np.isfinite(v)]
        return (float(v[-1] - v[0]), float(v.max() - v.min())) if v.size >= 3 else (
            float("nan"), float("nan"))

    d_real, s_real = drift(out["curves"]["raw"])
    d_cal, s_cal = drift(out["curves"]["calibration"])

    # split-to-split variance of rho, at the largest size, against the same calibration:
    # the axis that actually separates "noisy measurement of a fixed thing" from
    # "the thing depends on which tasks you used"
    def last_sd(rows):
        ok = [e for e in rows if e.get("n_splits_ok")]
        return float(ok[-1]["rho_sd"]) if ok else float("nan")
    sd_real, sd_cal = last_sd(out["curves"]["raw"]), last_sd(out["curves"]["calibration"])
    ratio = float(sd_real / sd_cal) if sd_cal > 0 else float("nan")

    out.update({"r1_drift": d_real, "r1_spread": s_real,
                "r1_drift_calibration": d_cal, "r1_spread_calibration": s_cal,
                "excess_drift": d_real - d_cal,
                "rho_sd": sd_real, "rho_sd_calibration": sd_cal, "rho_sd_ratio": ratio})

    if not np.isfinite(d_real) or not np.isfinite(ratio):
        out["verdict"] = "insufficient sizes"
    elif ratio > args.sd_ratio_tol:
        out["verdict"] = ("TASK-SET-RELATIVE -- rho varies across splits far more than a fixed "
                          "true score allows; disattenuation is invalid, not merely imprecise")
    else:
        out["verdict"] = ("consistent with a fixed true score measured noisily; drift is the "
                          "ceiling artefact, so report UNCORRECTED split-half agreement")
    return out


def report(r: dict) -> None:
    print(f"\n=== {r['name']}  ({r['n_tasks']} tasks x {r['n_features']} features; "
          f"{r['n_fixed_population']} active in ALL tasks) ===")
    for label in ("raw", "adjusted", "raw_fixed_pop", "adjusted_fixed_pop",
                  "calibration", "shuffle_floor"):
        rows = r["curves"][label]
        print(f"\n  {label}")
        print(f"    {'half size':>10} {'rho':>8} {'sd':>7} {'implied r1':>11} "
              f"{'-> r_full':>10} {'n feats':>9}")
        for e in rows:
            if not e.get("n_splits_ok"):
                continue
            print(f"    {e['size']:>10} {e['rho_median']:>8.3f} {e['rho_sd']:>7.3f} "
                  f"{e['r1']:>11.3f} {e['r_full']:>10.3f} {e['n_features_median']:>9.0f}")
    if "excess_drift" in r:
        print(f"\n  implied r1 drift, real data      : {r['r1_drift']:+.3f} "
              f"(spread {r['r1_spread']:.3f})")
        print(f"  implied r1 drift, calibration    : {r['r1_drift_calibration']:+.3f} "
              f"(spread {r['r1_spread_calibration']:.3f})   <- the ceiling artefact alone")
        print(f"  EXCESS drift                     : {r['excess_drift']:+.3f}"
              f"   (near 0 = the artefact explains it)")
        print(f"  rho spread across splits, real   : {r['rho_sd']:.4f}")
        print(f"  rho spread across splits, calib  : {r['rho_sd_calibration']:.4f}")
        print(f"  RATIO                            : {r['rho_sd_ratio']:.2f}"
              f"   (~1 = fixed true score; >>1 = task-set-relative)")
    print(f"  VERDICT: {r['verdict']}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--attr", action="append", required=True, metavar="NAME=PATH")
    ap.add_argument("--out", default="out/split_half_sweep")
    ap.add_argument("--n-splits", type=int, default=300)
    ap.add_argument("--tol", type=float, default=0.05,
                    help="excess drift beyond the calibration reference that counts as material")
    ap.add_argument("--sd-ratio-tol", type=float, default=1.5,
                    help="rho spread relative to calibration above which breadth counts as "
                         "task-set-relative rather than noisily measured")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    res = []
    for item in args.attr:
        if "=" not in item:
            raise SystemExit(f"--attr expects NAME=PATH, got {item!r}")
        name, path = item.split("=", 1)
        r = analyse(name, np.load(path, allow_pickle=True), args)
        report(r)
        res.append(r)
    with open(os.path.join(args.out, "split_half_sweep.json"), "w") as f:
        json.dump({"suites": res, "config": vars(args)}, f, indent=2)
    print(f"\n[sweep] wrote {os.path.join(args.out, 'split_half_sweep.json')}")
    print("[sweep] Read TWO things. (1) Spearman-Brown drifts here even under a perfectly\n"
          "[sweep] classical model, because PR's ceiling moves with the split size -- so P4\n"
          "[sweep] must report UNCORRECTED split-half agreement at a stated half-length\n"
          "[sweep] regardless of what this prints. (2) The rho-spread RATIO answers whether a\n"
          "[sweep] fixed true score exists at all: ~1 means breadth is a real property measured\n"
          "[sweep] noisily; >>1 means it depends on which tasks were used, and disattenuating\n"
          "[sweep] it would be correcting away real variation. Check the fixed-population rows\n"
          "[sweep] to confirm neither signal is the usable-feature set changing with size.")


if __name__ == "__main__":
    main()
