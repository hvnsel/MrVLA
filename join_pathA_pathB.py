"""Join Path A (causal breadth) with Path B (cross-model recurrence + inheritance), per feature.

Both paths must be computed on the SAME action-position SAE, so feature index j is the same
object in both -- run_attribution's layer_NN_attribution.npz (Path A) and run_recurrence's
layer_NN_target_<model>.npz on the --pool last probe with the ACTION-POSITION SAEs (Path B).

Tests the dissociation hypothesis:
    GENERAL features (high adjusted breadth) recur via INHERITANCE (high inheritance)
    SPECIALIST features (low adjusted breadth) recur via RE-DERIVATION (beyond inheritance)

so the predictions are:
    corr(adjusted_breadth, inheritance)                 > 0   (general = inherited)
    corr(adjusted_breadth, recurrence_beyond_inheritance) < 0   (specialists = re-derived)

Also prints the 2x2 (breadth x inheritance) with mean q_cross per cell, and, if the
hypothesis holds, the honest reading: recurrence decomposes into inherited general primitives
+ re-derived shared-scene memorisations -- NOT independent rediscovery of general skills.

Usage
-----
python join_pathA_pathB.py \
    --attr $BASE/ATTR/goal_k100/layer_31_attribution.npz \
    --rec  $BASE/RECURRENCE_ACTION/layer_31_target_goal.npz \
    --out  $BASE/RECURRENCE_ACTION/join_goal.json
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

from identify_features import adjusted_breadth
from identify_recurrent_features import recurrence_beyond
from mrvla.structural_generality import _spearman


def _ranks(x):
    r = np.argsort(np.argsort(x)).astype(np.float64)
    return r - r.mean()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--attr", required=True, help="Path A layer_NN_attribution.npz")
    p.add_argument("--rec", required=True, help="Path B layer_NN_target_<model>.npz (action-position)")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    A = np.load(args.attr)
    B = np.load(args.rec)

    PR = A["PR"].astype(np.float64)
    magnitude = A["magnitude"].astype(np.float64)
    a_base = A["base_rate"].astype(np.float64)
    a_active = A["is_active"].astype(bool)

    q_cross = B["q_cross"].astype(np.float64)
    b_base = B["base_rate"].astype(np.float64)
    b_active = B["is_active"].astype(bool)
    if "inheritance" not in B:
        raise SystemExit("Path B npz has no `inheritance` -- re-run run_recurrence with --base-key. "
                         "Without it the hypothesis (general=inherited) cannot be tested.")
    inheritance = B["inheritance"].astype(np.float64)

    if PR.shape != q_cross.shape:
        raise SystemExit(f"feature-count mismatch {PR.shape} vs {q_cross.shape} -- are both "
                         "paths on the SAME action-position SAE?")

    active = a_active & b_active
    adj = adjusted_breadth(PR, magnitude, a_base, active)
    rec_beyond = recurrence_beyond(q_cross, b_base, inheritance, active)

    m = active & np.isfinite(adj) & np.isfinite(rec_beyond) & np.isfinite(inheritance)
    n = int(m.sum())

    c_inh = _spearman(adj[m], inheritance[m])            # predict > 0
    c_rederived = _spearman(adj[m], rec_beyond[m])       # predict < 0
    c_qcross = _spearman(adj[m], q_cross[m])             # context

    # 2x2: split by median adjusted breadth and median inheritance; report mean q_cross
    br_hi = adj[m] > np.median(adj[m])
    inh_hi = inheritance[m] > np.median(inheritance[m])
    q = q_cross[m]
    cell = {}
    for bn, bmask in [("general", br_hi), ("specialist", ~br_hi)]:
        for inn, imask in [("inherited", inh_hi), ("re_derived", ~inh_hi)]:
            sel = bmask & imask
            cell[f"{bn}/{inn}"] = {"n": int(sel.sum()),
                                   "mean_qcross": float(np.mean(q[sel])) if sel.any() else float("nan")}

    hyp1 = c_inh > 0.05
    hyp2 = c_rederived < -0.05
    verdict = ("SUPPORTED" if (hyp1 and hyp2) else
               "PARTIAL" if (hyp1 or hyp2) else "NOT supported")

    print(f"[join] {n} features active in both paths (same action-position SAE)")
    print(f"[join] HYPOTHESIS: general features recur via inheritance; specialists re-derived")
    print(f"[join]   corr(breadth, inheritance)              = {c_inh:+.3f}   (predict > 0)  "
          f"{'OK' if hyp1 else 'no'}")
    print(f"[join]   corr(breadth, recurrence-beyond-inherit)= {c_rederived:+.3f}   (predict < 0)  "
          f"{'OK' if hyp2 else 'no'}")
    print(f"[join]   corr(breadth, raw q_cross)              = {c_qcross:+.3f}   (context)")
    print(f"[join]   => {verdict}")
    print(f"[join] 2x2 mean q_cross (breadth x inheritance):")
    for k, v in cell.items():
        print(f"[join]     {k:24s} n={v['n']:4d}  mean_qcross={v['mean_qcross']:.3f}")
    if verdict == "SUPPORTED":
        print("[join] Reading: recurrence decomposes into INHERITED general primitives + "
              "RE-DERIVED specialist structure (shared-scene memorisation), NOT independent "
              "rediscovery of general skills.")

    out = {
        "n_features": n,
        "corr_breadth_inheritance": c_inh,
        "corr_breadth_recurrence_beyond_inheritance": c_rederived,
        "corr_breadth_qcross": c_qcross,
        "hypothesis_verdict": verdict,
        "cells_2x2": cell,
        "attr": os.path.abspath(args.attr), "rec": os.path.abspath(args.rec),
    }
    out_path = args.out or os.path.join(os.path.dirname(args.rec), "join_pathA_pathB.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[join] wrote {out_path}")


if __name__ == "__main__":
    main()
