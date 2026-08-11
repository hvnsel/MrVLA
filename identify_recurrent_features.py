"""Path B / identification: which specific features recur across models BEYOND inheritance.

Path B showed cross-model recurrence is real in aggregate, but ranking features by raw
q_cross conflates two very different things:

  * a feature that recurs because it was ALREADY IN THE BASE model (all four fine-tunes
    inherited it) -- trivial, shared ancestry, not evidence of anything;
  * a feature that recurs because fine-tuning independently RE-DERIVED it -- the interesting
    claim.

`inheritance` (saved by run_recurrence) tells them apart per feature: high = the feature
reads a direction already present in the base (push base activations through the fine-tuned
SAE and it fires the same); low = fine-tuning created/reshaped it. So the honest "which
features recur" is: rank q_cross RESIDUALISED on both base firing rate (the activity
confound) AND inheritance. The top of that ranking are the re-derived recurrent features --
the ones that survive "who's to say it isn't just the base model?".

Pure numpy on run_recurrence's saved npz. No re-encoding, no probe needed for the ranking
(capture_recurrent_frames.py does the cross-model visualisation).

Usage
-----
python identify_recurrent_features.py \
    --rec-dir ./RECURRENCE/goal_vs_all --layer 31 --target goal --top 12 \
    --out ./RECURRENCE/goal_vs_all/recurrent_features.json
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np


def _ranks(x: np.ndarray) -> np.ndarray:
    r = np.argsort(np.argsort(x)).astype(np.float64)
    return r - r.mean()


def recurrence_beyond(q_cross, base_rate, inheritance, active):
    """Rank-residualise q_cross on base firing rate AND (if available) inheritance.

    Returns [F] (NaN for inactive). High = recurs across models MORE than its activity and
    its base-model inheritance predict -- i.e. re-derived shared structure. If `inheritance`
    is None we can only remove the activity confound and CANNOT rule out inheritance; the
    caller must flag that loudly.
    """
    F = len(q_cross)
    out = np.full(F, np.nan)
    cols = [base_rate] if inheritance is None else [base_rate, inheritance]
    finite_cols = np.all(np.stack([np.isfinite(c) for c in cols], axis=0), axis=0)
    m = active & np.isfinite(q_cross) & finite_cols
    if m.sum() < 5:
        return out
    ry = _ranks(q_cross[m])
    C = np.stack([_ranks(c[m]) for c in cols], axis=1)
    beta, *_ = np.linalg.lstsq(C, ry, rcond=None)
    out[m] = ry - C @ beta
    return out


def load_target(rec_dir: str, layer: int, target: str) -> dict:
    path = os.path.join(rec_dir, f"layer_{layer:02d}_target_{target}.npz")
    if not os.path.exists(path):
        raise FileNotFoundError(f"no recurrence npz at {path}")
    d = np.load(path)
    return {k: d[k] for k in d.files} | {"_path": path}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--rec-dir", required=True, help="run_recurrence output dir")
    p.add_argument("--layer", type=int, default=31)
    p.add_argument("--target", required=True, help="target model key (e.g. goal)")
    p.add_argument("--top", type=int, default=12)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    d = load_target(args.rec_dir, args.layer, args.target)
    q_cross = d["q_cross"].astype(np.float64)
    base_rate = d["base_rate"].astype(np.float64)
    active = d["is_active"].astype(bool)
    inheritance = d["inheritance"].astype(np.float64) if "inheritance" in d else None
    retention = d.get("retention")

    if inheritance is None:
        print("[rec-id] WARNING: no `inheritance` field in this npz. The recurrence run did "
              "NOT include a base-model probe (--base-key), so we can remove the activity "
              "confound but CANNOT rule out that recurrence is base-model inheritance. Re-run "
              "run_recurrence with the base probe before trusting the re-derived ranking.",
              flush=True)

    score = recurrence_beyond(q_cross, base_rate, inheritance, active)

    order = np.argsort(np.where(np.isfinite(score), score, -np.inf))
    rederived = order[::-1][:args.top].tolist()          # recur beyond inheritance+activity

    # for contrast: features that recur but ARE inherited (high q_cross AND high inheritance)
    inherited = []
    if inheritance is not None:
        inh_rank = np.where(active, _ranks(np.where(active, inheritance, np.nan)), -np.inf)
        q_rank = np.where(active, _ranks(np.where(active, q_cross, np.nan)), -np.inf)
        inh_recurrent = np.where(active, inh_rank + q_rank, -np.inf)     # both high
        inherited = np.argsort(inh_recurrent)[::-1][:args.top].tolist()

    def rec(j):
        r = {"feature": int(j), "q_cross": float(q_cross[j]), "base_rate": float(base_rate[j]),
             "recurrence_beyond_score": float(score[j])}
        if inheritance is not None:
            r["inheritance"] = float(inheritance[j])
        if retention is not None:
            r["retention"] = float(retention[j])
        return r

    out = {"rec_dir": os.path.abspath(args.rec_dir), "layer": args.layer, "target": args.target,
           "has_inheritance_control": inheritance is not None,
           "note": ("ranked by q_cross residualised on base rate AND inheritance"
                    if inheritance is not None else
                    "NO inheritance control -- base rate only; cannot rule out inheritance"),
           "re_derived_recurrent": [rec(j) for j in rederived],
           "inherited_recurrent": [rec(j) for j in inherited]}

    print(f"[rec-id] target={args.target} layer={args.layer}  "
          f"{'WITH' if inheritance is not None else 'WITHOUT'} inheritance control")
    print(f"[rec-id] top re-derived recurrent features (recur beyond activity + inheritance):")
    for j in rederived:
        s = f"  feat {j:5d}  q_cross={q_cross[j]:.3f}  base_rate={base_rate[j]:.3f}"
        if inheritance is not None:
            s += f"  inheritance={inheritance[j]:.3f}"
        s += f"  score={score[j]:+.1f}"
        print(s)

    out_path = args.out or os.path.join(args.rec_dir, f"recurrent_features_{args.target}.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[rec-id] wrote {out_path}")
    print(f"[rec-id] next: python capture_recurrent_frames.py --probe-dir <probe> "
          f"--sae-map <...> --features {out_path} --target {args.target}")


if __name__ == "__main__":
    main()
