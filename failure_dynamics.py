"""Does coalition CHURN, or a fitted projection of the codes, predict episode failure?

Four level-based signals came back null (mu_t, share, phi_total, top-2 margin). Each was a
projection of the 2048-dim code onto one axis picked in advance. This asks the two questions
that were never asked: whether the coalition's STABILITY over time carries the signal, and
whether a FITTED projection finds an axis four hand-built ones missed.

WHAT IT REPORTS, AND WHAT EACH RESULT WOULD MEAN

  churn / returns      AUROC(t) for coalition instability and for period-2 recurrence in
                       feature space. A win here says failure is a limit cycle in the
                       dictionary -- but only if it beats the next row.
  action churn/returns THE BASELINE. The same period-2 statistic on the executed action
                       vectors. A dithering arm is visible without any dictionary, so if
                       feature churn does not beat action churn the finding is "the arm is
                       oscillating and so are the features driving it", which needs none of
                       this machinery.
  probe(z) vs probe(h) LOTO ridge on per-episode feature means, against the same probe on
                       the raw residual it was trained to reconstruct. probe(z) ~ probe(h)
                       means the dictionary is a lossy reparameterisation and buys nothing;
                       a probe on hidden states is what SAFECAST already does without an SAE.
  breadth skew         Do the features the probe leans on skew LOW adjusted breadth? The
                       bridge back to Path A: mu_t asked this with a hand-built ratio and
                       came back null, and this asks it of a fitted projection instead.

TWO TRAPS THIS INHERITS, BOTH ALREADY PAID FOR ONCE

  * DURATION. Failures always run to the step cap, so an online curve silently drops
    successes as t grows and starts measuring length again. The grid is capped at the
    shortest episode and every point reports its own n_ok / n_fail.
  * TASK IDENTITY. Tasks differ in base failure rate, so the probe is scored leave-one-TASK-
    out and its null permutes labels WITHIN task -- a global permutation would break that
    structure and hand the probe a free win.

Usage
-----
python failure_dynamics.py --rollout-dir $B/ROLLOUT_ACTION/goal \\
                           --sae-dir     $B/ACT_ACTION_SAE/goal/sae \\
                           --attr        $B/ATTR/goal_k100/layer_31_attribution.npz \\
                           --out         $B/ROLLOUT_ACTION/goal/failure_dynamics.json
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
import torch

from identify_features import adjusted_breadth
from mrvla.dynamics import (
    action_dynamics, auroc_curve, coalition_dynamics, max_unbiased_t, probe_loto,
    topk_sets, weight_breadth_skew,
)
from mrvla.readout import signature_matrix, unnormalized_logits
from run_attribution import load_sae, sae_encode_full


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--rollout-dir", required=True)
    p.add_argument("--sae-dir", required=True)
    p.add_argument("--attr", required=True)
    p.add_argument("--layer", type=int, default=31)
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--batch-size", type=int, default=4096)
    p.add_argument("--block", type=int, default=8192)
    p.add_argument("--top-m", type=int, default=32,
                   help="coalition size by |phi|; smaller sets discriminate more sharply "
                        "than the full k=100 active set, which is the same size every step")
    p.add_argument("--probe-window", type=int, default=50,
                   help="timesteps of each episode the probe sees")
    p.add_argument("--lam", type=float, default=10.0, help="ridge penalty")
    p.add_argument("--n-perm", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)

    hc = np.load(os.path.join(args.rollout_dir, "head_constants.npz"))
    W_U_act = hc["W_U_act"].astype(np.float64)
    g_gain = hc["g"].astype(np.float64)
    n_bins, d = W_U_act.shape

    W_enc, W_dec_t, b_pre_t, k, ck = load_sae(args.sae_dir, args.layer)
    W_dec = W_dec_t.detach().float().cpu().numpy().astype(np.float64)
    S_abs = np.abs(signature_matrix(W_dec, g_gain, W_U_act, center=True))
    F = W_dec.shape[0]

    at = np.load(args.attr)
    adj = adjusted_breadth(at["PR"].astype(np.float64), at["magnitude"].astype(np.float64),
                           at["base_rate"].astype(np.float64), at["is_active"].astype(bool))
    print(f"[dyn] SAE {ck}  F={F}  k={k}  top-m={args.top_m}  "
          f"probe window={args.probe_window} steps", flush=True)

    sets_a, ep_a, sl_a, ts_a, sc_a = [], [], [], [], []
    act_a, aep_a, ats_a, asc_a, atk_a = [], [], [], [], []
    zsum: dict = {}
    hsum: dict = {}
    zcnt: dict = {}

    for sp in sorted(glob.glob(os.path.join(args.rollout_dir, "shard_*.npz"))):
        dd = np.load(sp)
        res = dd["residual"]
        n, n_slot = res.shape[0], res.shape[1]
        H = res.reshape(n * n_slot, d).astype(np.float32)
        z, l2, mu = sae_encode_full(W_enc, b_pre_t, k, H, device, args.batch_size)

        ep_r = np.repeat(dd["episode"].astype(np.int64), n_slot)
        ts_r = np.repeat(dd["timestep"].astype(np.int64), n_slot)
        sl_r = np.tile(np.arange(n_slot, dtype=np.int64), n)
        sc_r = np.repeat(dd["success"].astype(np.int64), n_slot)

        ep_a.append(ep_r); ts_a.append(ts_r); sl_a.append(sl_r); sc_a.append(sc_r)
        act_a.append(dd["action"].astype(np.float64))
        aep_a.append(dd["episode"].astype(np.int64)); ats_a.append(dd["timestep"].astype(np.int64))
        asc_a.append(dd["success"].astype(np.int64)); atk_a.append(dd["task_id"].astype(np.int64))

        for s0 in range(0, H.shape[0], args.block):
            s1 = min(s0 + args.block, H.shape[0])
            L = unnormalized_logits(H[s0:s1].astype(np.float64), g_gain, W_U_act)
            rows = np.argmax(L, axis=1)
            zb = z[s0:s1].astype(np.float64)
            sets_a.append(topk_sets(zb * S_abs[:, rows].T, args.top_m))

            w = ts_r[s0:s1] < args.probe_window
            if w.any():
                for e in np.unique(ep_r[s0:s1][w]):
                    m = w & (ep_r[s0:s1] == e)
                    e = int(e)
                    zsum[e] = zsum.get(e, 0.0) + zb[m].sum(axis=0)
                    hsum[e] = hsum.get(e, 0.0) + H[s0:s1][m].astype(np.float64).sum(axis=0)
                    zcnt[e] = zcnt.get(e, 0) + int(m.sum())
        print(f"[dyn]   {os.path.basename(sp)}: {sum(x.shape[0] for x in sets_a)} rows",
              flush=True)

    sets = np.concatenate(sets_a)
    ep, ts, sl, sc = (np.concatenate(x) for x in (ep_a, ts_a, sl_a, sc_a))
    act = np.concatenate(act_a)
    aep, ats, asc, atk = (np.concatenate(x) for x in (aep_a, ats_a, asc_a, atk_a))

    # ---- dynamics ---------------------------------------------------------
    feat = coalition_dynamics(sets, ep, sl, ts)
    base = action_dynamics(act, aep, ats)

    ep_ids = np.unique(aep)
    lengths = np.array([int(np.unique(ats[aep == e]).size) for e in ep_ids])
    fail = np.array([1 - int(asc[aep == e][0]) for e in ep_ids])
    task = np.array([int(atk[aep == e][0]) for e in ep_ids])
    tmax = max_unbiased_t(lengths, fail)
    grid = [t for t in (5, 10, 20, 30, 40, 50, 60, 70, 80, 100, 150, 200) if t <= tmax]
    print(f"\n[dyn] shortest episode is {tmax} steps; the curve is capped there because past "
          f"it\n[dyn] the surviving sample is nearly all failures and the AUROC becomes a "
          f"duration read.", flush=True)

    curves = {
        "feature_churn": auroc_curve(feat["churn"], ep, ts, sc, grid),
        "feature_returns": auroc_curve(feat["returns"], ep, ts, sc, grid),
        "action_churn": auroc_curve(base["churn"], aep, ats, asc, grid),
        "action_returns": auroc_curve(base["returns"], aep, ats, asc, grid),
    }

    # ---- probes -----------------------------------------------------------
    order = {int(e): i for i, e in enumerate(ep_ids)}
    Xz = np.zeros((ep_ids.size, F))
    Xh = np.zeros((ep_ids.size, d))
    for e, i in order.items():
        c = max(zcnt.get(e, 0), 1)
        Xz[i] = np.asarray(zsum.get(e, np.zeros(F))) / c
        Xh[i] = np.asarray(hsum.get(e, np.zeros(d))) / c

    pz = probe_loto(Xz, fail, task, lam=args.lam, n_perm=args.n_perm, seed=args.seed)
    ph = probe_loto(Xh, fail, task, lam=args.lam, n_perm=args.n_perm, seed=args.seed)
    skew = weight_breadth_skew(pz["weights"], adj, top_n=50)

    summary = {
        "rollout_dir": args.rollout_dir, "top_m": args.top_m,
        "probe_window": args.probe_window, "lam": args.lam,
        "n_episodes": int(ep_ids.size), "n_failures": int(fail.sum()),
        "max_unbiased_t": tmax, "curves": curves,
        "probe_z": {k2: v for k2, v in pz.items() if k2 != "weights"},
        "probe_h": {k2: v for k2, v in ph.items() if k2 != "weights"},
        "breadth_skew": skew,
        "top_weight_features": np.argsort(-np.abs(pz["weights"]))[:50].tolist(),
    }
    json.dump(summary, open(args.out, "w"), indent=2, default=float)

    print(f"\n[dyn] {ep_ids.size} episodes, {int(fail.sum())} failures\n")
    print(f"{'AUROC(t)':<18}" + "".join(f"{t:>8}" for t in grid))
    for name, rows in curves.items():
        print(f"{name:<18}" + "".join(f"{r['auroc']:>8.3f}" for r in rows))
    print(f"{'n_ok/n_fail':<18}"
          + "".join(f"{r['n_ok']}/{r['n_fail']:>3}" for r in curves['feature_churn']))
    print("\nFeature rows must beat the ACTION rows, or a dictionary was not needed.\n")
    print(f"probe(z)  AUROC {pz['auroc']:.3f}   null {pz['null_mean']:.3f} "
          f"(p95 {pz['null_p95']:.3f})  p={pz['p_value']:.3f}")
    print(f"probe(h)  AUROC {ph['auroc']:.3f}   null {ph['null_mean']:.3f} "
          f"(p95 {ph['null_p95']:.3f})  p={ph['p_value']:.3f}")
    print("  probe(z) ~ probe(h) means the dictionary added nothing a dense probe lacks.\n")
    print(f"breadth skew of the top-50 probe weights: rho {skew['rho']:+.3f}, "
          f"mean breadth pct {skew.get('top_w_mean_breadth_percentile', float('nan')):.1f}, "
          f"{skew.get('top_w_frac_below_median_breadth', float('nan')):.0%} below median")
    print("  NEGATIVE rho = failure is carried by LOW-breadth specialists (the mu_t "
          "hypothesis, reached a second way).")
    print(f"\n[dyn] wrote {args.out}")


if __name__ == "__main__":
    main()
