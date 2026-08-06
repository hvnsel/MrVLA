"""Validate a Stage A1 smoke collection: does the captured residual re-decode to the
emitted action token?

This is gate levels 0 and 1 in miniature, on the raw residual (no SAE yet). For each
captured decision it applies the real final RMSNorm + action-unembedding to the stored
layer-31 residual and checks whether the argmax over the 256 action tokens equals the
token the model actually emitted. If this match rate is ~1.0, the collection is wired
correctly: the residual we captured really is the one that produced the action, and the
action-token offset / RMSNorm constants are right. If it is low, STOP -- something in the
capture or the head-constant export is wrong, and the full run would be wasted.

Usage
-----
python validate_a1.py --dir /work/.../ACT_ACTION/goal_smoke
"""

from __future__ import annotations

import argparse
import glob
import os

import numpy as np

from mrvla.attribution import action_logits


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dir", required=True, help="A1 output dir (shards + head_constants.npz)")
    p.add_argument("--max-decisions", type=int, default=200)
    args = p.parse_args()

    hc = np.load(os.path.join(args.dir, "head_constants.npz"))
    W_U_act = hc["W_U_act"].astype(np.float64)      # [256, d]
    g = hc["g"].astype(np.float64)                  # [d]
    eps = float(hc["eps"])
    act_ids = hc["act_ids"] if "act_ids" in hc else np.arange(
        int(hc["action_vocab"]) - int(hc["n_bins"]), int(hc["action_vocab"]))
    id0 = int(act_ids[0])                            # first action token id (= A - 256)
    n_act = W_U_act.shape[0]
    print(f"[val] action tokens {id0}..{id0 + n_act - 1}  d={W_U_act.shape[1]}  eps={eps}")

    shards = sorted(glob.glob(os.path.join(args.dir, "shard_*.npz")))
    if not shards:
        raise FileNotFoundError(f"no shard_*.npz in {args.dir}")

    total = match = in_range = 0
    corrs = []
    for sp in shards:
        d = np.load(sp)
        res = d["residual"].astype(np.float64)      # [n, 7, d]
        toks = d["token_ids"].astype(np.int64)      # [n, 7]
        n, seven, dim = res.shape
        for i in range(n):
            for s in range(seven):
                h = res[i, s]
                tok = int(toks[i, s])
                logits = action_logits(h, W_U_act, g, eps)     # [256]
                pred_row = int(np.argmax(logits))
                pred_id = id0 + pred_row
                true_row = tok - id0
                total += 1
                in_range += int(0 <= true_row < n_act)
                match += int(pred_id == tok)
                if 0 <= true_row < n_act:
                    # rank-agnostic sanity: the true token's logit should be near the top
                    corrs.append(int(logits[true_row] >= np.sort(logits)[-3]))
                if total >= args.max_decisions:
                    break
            if total >= args.max_decisions:
                break
        if total >= args.max_decisions:
            break

    print(f"[val] decisions checked      : {total}")
    print(f"[val] emitted-token in range : {in_range}/{total} "
          f"({100*in_range/total:.1f}%)   <- should be 100%")
    print(f"[val] argmax == emitted token: {match}/{total} "
          f"({100*match/total:.1f}%)      <- GATE L1 (want >= 85%, ideally ~100%)")
    if corrs:
        print(f"[val] emitted token in top-3 : {sum(corrs)}/{len(corrs)} "
              f"({100*sum(corrs)/len(corrs):.1f}%)")
    ok = (in_range == total) and (match / total >= 0.85)
    print(f"\n[val] {'PASS -- pipeline wired correctly, proceed to full A1' if ok else 'FAIL -- do NOT run the full collection; something is mis-wired'}")


if __name__ == "__main__":
    main()
