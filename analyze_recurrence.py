"""Interrogate saved cross-model recurrence arrays: spread + confound-free signal.

`run_recurrence.py` prints only mean/median q_cross, which cannot tell a real
general-vs-specific axis (wide spread) from "everything recurs equally" (a null),
and does not remove the base-rate / base-inheritance confounds it reports.  This
script reads the saved per-(layer, target) npz files and, for each, reports:

  * the DISTRIBUTION of q_cross over active features (percentiles + std) -- is there
    spread to discriminate on at all?
  * how much of q_cross is explained by the two confounds together (OLS R^2 of
    q_cross ~ base_rate + inheritance on ranks) -- if ~all of it, recurrence is just
    activity + inheritance; if a chunk survives, there is a learned-generality signal.
  * the confound-free recurrence residual (q_cross with base_rate AND inheritance
    projected out) and its top features -- the candidate independently-learned
    general features.

Pure numpy on saved arrays; no model, no re-encoding.

Usage
-----
python analyze_recurrence.py --rec-dir ./recurrence_v1 --layers 0,8,16,24,31
"""

from __future__ import annotations

import argparse
import glob
import os

import numpy as np


def _ranks(x):
    r = np.argsort(np.argsort(x)).astype(np.float64)
    return r - r.mean()


def _ols_resid_r2(y, X):
    """Return (residual, R^2) for OLS of y on columns of X (both mean-centred)."""
    y = y - y.mean()
    Xc = X - X.mean(axis=0, keepdims=True)
    # add intercept implicitly via centring; solve least squares
    beta, *_ = np.linalg.lstsq(Xc, y, rcond=None)
    yhat = Xc @ beta
    resid = y - yhat
    ss_tot = (y ** 2).sum()
    r2 = 1.0 - (resid ** 2).sum() / ss_tot if ss_tot > 0 else float("nan")
    return resid, r2, beta


def analyze_file(path: str, topn: int = 12) -> dict:
    d = np.load(path)
    q = d["q_cross"].astype(np.float64)
    br = d["base_rate"].astype(np.float64)
    active = d["is_active"].astype(bool)
    has_inh = "inheritance" in d
    inh = d["inheritance"].astype(np.float64) if has_inh else None

    qa = q[active]
    pcts = np.percentile(qa, [10, 25, 50, 75, 90])
    out = {
        "n_active": int(active.sum()),
        "q_mean": float(qa.mean()), "q_std": float(qa.std()),
        "q_p10": pcts[0], "q_p25": pcts[1], "q_p50": pcts[2],
        "q_p75": pcts[3], "q_p90": pcts[4],
        "spread_p90_p10": float(pcts[4] - pcts[0]),
    }

    # confound regression on ranks (monotone, scale-free)
    ry = _ranks(q[active])
    cols = [_ranks(br[active])]
    names = ["base_rate"]
    if has_inh:
        cols.append(_ranks(inh[active]))
        names.append("inheritance")
    X = np.stack(cols, axis=1)
    resid, r2, beta = _ols_resid_r2(ry, X)
    out["confounds"] = names
    out["confound_r2"] = float(r2)          # variance of q_cross explained by confounds
    # fraction of the recurrence ranking that is confound-free (0..1) = sqrt(1 - R2)
    out["confound_free_frac"] = float(resid.std() / ry.std()) if ry.std() > 0 else float("nan")

    # map residual back to feature indices for the top confound-free features
    idx_active = np.where(active)[0]
    order = np.argsort(-resid)
    out["top_confound_free_feats"] = idx_active[order[:topn]].tolist()
    out["top_confound_free_q"] = q[idx_active[order[:topn]]].round(3).tolist()
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--rec-dir", required=True)
    p.add_argument("--layers", default="0,8,16,24,31")
    p.add_argument("--topn", type=int, default=12)
    args = p.parse_args()

    layers = [int(x) for x in args.layers.split(",") if x.strip()]
    print(f"{'layer/target':22s} {'nact':>5} {'qmean':>6} {'qstd':>6} "
          f"{'p10':>5} {'p50':>5} {'p90':>5} {'conf_R2':>7} {'free_frac':>9}")
    print("-" * 79)
    for layer in layers:
        files = sorted(glob.glob(os.path.join(args.rec_dir, f"layer_{layer:02d}_target_*.npz")))
        for path in files:
            tag = os.path.basename(path).replace(".npz", "").replace("layer_", "L").replace("_target_", "/")
            r = analyze_file(path, args.topn)
            print(f"{tag:22s} {r['n_active']:5d} {r['q_mean']:6.3f} {r['q_std']:6.3f} "
                  f"{r['q_p10']:5.2f} {r['q_p50']:5.2f} {r['q_p90']:5.2f} "
                  f"{r['confound_r2']:7.3f} {r['confound_free_frac']:9.3f}")
    print("\nHow to read:")
    print("  qstd / (p90-p10): spread. Near 0 => every feature recurs equally => no")
    print("     discriminating axis (null). Wide => a real general-vs-specific axis.")
    print("  conf_R2: fraction of q_cross explained by base_rate + inheritance. ~1.0 =>")
    print("     recurrence IS activity+inheritance (confounded). Lower => a learned-")
    print("     generality signal survives; free_std is how much of it remains.")


if __name__ == "__main__":
    main()
