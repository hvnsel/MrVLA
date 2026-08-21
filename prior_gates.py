"""Run the prior-gate ladder (Gates 0-4) on a Path A / A1 collection.

The action margin splits into features (0.531), a `mu*1 + b_pre` bias (0.405) and error
(0.064). Path A characterised the feature half and never looked at the bias. This driver asks,
in five escalating steps, whether that bias is an ACTION PRIOR worth modulating -- see
`mrvla/prior_gates.py` for the derivation, the pre-registered pass bars, and what each gate
kills. It stops reporting at the first failure, because a gate whose predecessor failed is
computing a quantity already shown not to exist.

WHAT THIS CANNOT TELL YOU. A1 replays DEMONSTRATIONS, so `success` is the constant 1
(`mrvla/libero_demos.py`) and the only available target is single-step deviation from the
expert AT EXPERT STATES. That is a necessary condition for "modulating the prior raises
success rate" and it is not evidence for it: the policy never sees its own mistakes compound
here. Gates 5-7 (pivotality, closed-loop transfer, a powered success test) need
action-position residuals from CLOSED-LOOP rollouts carrying real success labels, which do not
exist yet -- `mrvla/libero_collect.py` already has the buffer-until-episode-end pattern that
collection would need.

Gates 0-2 run on every decision and need only the shards and the SAE. Gates 3-4 additionally
need the demo action, so they need the LIBERO demo HDF5s and the checkpoint's norm_stats, and
they are computed on a subsample (`--sample`) because they materialise an [n, 256] score
matrix per lambda.

Usage
-----
python prior_gates.py --acts-dir  $B/ACT_ACTION/libero_goal \\
                      --sae-dir   $B/SAE_ACTION/goal_k100 \\
                      --out       $B/ATTR/goal_k100/prior_gates.json
# with Gates 3-4:
    ... --task-suite libero_goal --norm-stats $B/norm_stats.json --unnorm-key libero_goal

Export norm_stats once from the checkpoint:
    python -c "from mrvla.model_utils import load_openvla; import json; \\
               m,_=load_openvla('openvla/openvla-7b-finetuned-libero-goal'); \\
               json.dump({k:{'action':v['action']} for k,v in m.norm_stats.items()}, \\
                         open('norm_stats.json','w'), default=list)"
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
import torch

from mrvla.prior_gates import (
    CANARY_MEDIAN_FRAC, bias_share, demo_bin_index, gate0_bias_composition,
    gate1_mu_over_r, gate2_prior_vs_marginal, gate3_share_predicts_deviation,
    gate4_lambda_sweep, prior_scores, prior_vectors, verdict_table,
)
from mrvla.readout import signature_matrix, top2_margin, unnormalized_logits
from run_attribution import load_sae, sae_encode_full

def load_demo_actions(task_suite: str, acts_dir: str) -> dict:
    """(task_id, episode, timestep) -> expert action [7], via A1's episode->demo mapping.

    A1 stores `episode` as a running counter incremented once per demo, and processes demos in
    sorted(demo_keys) order, so the demo for a decision is at rank `episode - min_episode(task)`
    within that task's sorted keys -- the same recovery `capture_feature_frames.py` documents,
    which keeps the mapping independent of A1's --max-demos/--max-steps settings.
    """
    import h5py
    from libero.libero import benchmark, get_libero_path

    from mrvla.libero_demos import _find_demo_file

    min_ep: dict[int, int] = {}
    for sp in sorted(glob.glob(os.path.join(acts_dir, "shard_*.npz"))):
        d = np.load(sp)
        for t, e in zip(d["task_id"], d["episode"]):
            t, e = int(t), int(e)
            if t not in min_ep or e < min_ep[t]:
                min_ep[t] = e

    suite = benchmark.get_benchmark_dict()[task_suite]()
    root = get_libero_path("datasets")
    out: dict = {}
    for task_id in sorted(min_ep):
        path = _find_demo_file(root, task_suite, suite.get_task(task_id))
        with h5py.File(path, "r") as fh:
            keys = sorted(fh["data"].keys(), key=lambda k: int(k.split("_")[-1]))
            for rank, key in enumerate(keys):
                acts = np.asarray(fh["data"][key]["actions"])          # [T, 7]
                ep = min_ep[task_id] + rank
                for ts in range(acts.shape[0]):
                    out[(task_id, ep, ts)] = acts[ts]
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--acts-dir", required=True, help="A1 output (shards + head_constants.npz)")
    p.add_argument("--sae-dir", required=True, help="SAE trained on action-position residuals")
    p.add_argument("--layer", type=int, default=31)
    p.add_argument("--out", required=True, help="output JSON path")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--batch-size", type=int, default=4096)
    p.add_argument("--sample", type=int, default=100_000,
                   help="rows kept for Gates 3-4; they hold two [n, 256] float64 matrices, "
                        "so 100k is ~400 MB. Raise it if you have the RAM.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--task-suite", default=None, help="enables Gates 3-4 (needs LIBERO + h5py)")
    p.add_argument("--norm-stats", default=None, help="JSON of the checkpoint's norm_stats")
    p.add_argument("--unnorm-key", default=None, help="key into norm_stats, e.g. libero_goal")
    p.add_argument("--force-34", action="store_true",
                   help="report Gates 3-4 even if the demo-bin canary fails (diagnosis only)")
    args = p.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(args.seed)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)

    hc = np.load(os.path.join(args.acts_dir, "head_constants.npz"))
    W_U_act = hc["W_U_act"].astype(np.float64)
    g_gain = hc["g"].astype(np.float64)
    eps = float(hc["eps"])
    id0 = int(hc["act_ids"].astype(np.int64)[0])
    n_bins, d = W_U_act.shape

    W_enc, W_dec_t, b_pre_t, k, ck = load_sae(args.sae_dir, args.layer)
    W_dec = W_dec_t.detach().float().cpu().numpy().astype(np.float64)
    b_pre = b_pre_t.detach().float().cpu().numpy().astype(np.float64)
    print(f"[gates] SAE {ck}  F={W_dec.shape[0]}  k={k}  d={d}  bins={n_bins}", flush=True)

    A, B = prior_vectors(W_U_act, g_gain, b_pre)
    # center=True is load-bearing, not cosmetic: `prior_vectors` centres U, so an uncentred
    # S would put the feature term on a different scale from the prior and the true margin and
    # silently inflate |feat| in `bias_share`. Centring leaves every argmax unchanged.
    S = signature_matrix(W_dec, g_gain, W_U_act, center=True)  # [F, 256], reused from readout.py

    demos = None
    if args.task_suite:
        if not (args.norm_stats and args.unnorm_key):
            raise SystemExit("--task-suite needs --norm-stats and --unnorm-key for Gates 3-4")
        demos = load_demo_actions(args.task_suite, args.acts_dir)
        ns = json.load(open(args.norm_stats))[args.unnorm_key]["action"]
        q01, q99 = np.asarray(ns["q01"], float), np.asarray(ns["q99"], float)
        dmask = np.asarray(ns.get("mask", [True] * len(q01)), bool)
        print(f"[gates] demo actions loaded: {len(demos)} (task, episode, timestep) triples",
              flush=True)

    shards = sorted(glob.glob(os.path.join(args.acts_dir, "shard_*.npz")))
    if not shards:
        raise FileNotFoundError(f"no shard_*.npz in {args.acts_dir}")

    # scalars for Gates 0-2 (all decisions); sampled rows for Gates 3-4
    MU, RR, ROW, SLOT = [], [], [], []
    s_share, s_dev, s_margin, s_activity, s_demo, s_emit = [], [], [], [], [], []
    s_feat, s_prior = [], []
    n_total = 0

    for sp in shards:
        dd = np.load(sp)
        res = dd["residual"]                                   # [n, 7, d] float16
        toks = dd["token_ids"].astype(np.int64)                # [n, 7]
        n, n_slot = res.shape[0], res.shape[1]
        H = res.reshape(n * n_slot, d).astype(np.float32)
        rows = (toks.reshape(-1) - id0).astype(np.int64)
        slots = np.tile(np.arange(n_slot), n)

        z, l2, mu = sae_encode_full(W_enc, b_pre_t, k, H, device, args.batch_size)
        # rms batched. einsum accumulates in float64 WITHOUT materialising a float64 copy of
        # H -- at 4096 dims and a full shard that copy is several GB, and `H * H` alone is as
        # large again. This is mrvla.attribution.rms, row-wise.
        r_scal = np.sqrt(np.einsum("ij,ij->i", H, H, dtype=np.float64) / d + eps)

        MU.append(mu.astype(np.float64)); RR.append(r_scal)
        ROW.append(rows); SLOT.append(slots)
        n_total += rows.size

        # Gate 1 is the structural kill and it converges on the first shard -- mu-vs-r is a
        # property of the model, not something that needs every decision. Print it running so
        # a clear failure can be killed in minutes instead of after the whole encode pass.
        g1 = gate1_mu_over_r(np.concatenate(MU), np.concatenate(RR))
        print(f"[gates]   {os.path.basename(sp)}: {n_total} decisions | "
              f"gate1(provisional) {'PASS' if g1['pass'] else 'FAIL'} "
              f"mu_retained={g1['mu_retained']:+.4f} cv(mu/r)={g1['cv_mu_over_r']:.4f} "
              f"corr(mu,r)={g1['pearson_mu_r']:+.3f}", flush=True)

        # ---- Gates 3-4 subsample -------------------------------------------------
        if demos is None:
            continue
        keep_n = max(1, args.sample // len(shards))   # A1 writes fixed-size shards
        sel = rng.choice(rows.size, size=min(keep_n, rows.size), replace=False)

        ep_r = np.repeat(dd["episode"].astype(np.int64), n_slot)[sel]
        ts_r = np.repeat(dd["timestep"].astype(np.int64), n_slot)[sel]
        tk_r = np.repeat(dd["task_id"].astype(np.int64), n_slot)[sel]
        sl_r = slots[sel]
        # LIBERO actions are [T, 7] and A1 decodes 7 slots, so action dim == n_slot here.
        acts = np.array([demos.get((int(a), int(b), int(c)), np.full(n_slot, np.nan))
                         for a, b, c in zip(tk_r, ep_r, ts_r)], dtype=np.float64)
        ok = np.isfinite(acts).all(axis=1)
        dbin = np.full(sel.size, np.nan)
        if ok.any():
            dbin[ok] = demo_bin_index(acts[ok], q01, q99, dmask, n_bins)[
                np.arange(ok.sum()), sl_r[ok]]

        feat = l2[sel, None].astype(np.float64) * (z[sel].astype(np.float64) @ S)   # [m, 256]
        pri = prior_scores(mu[sel].astype(np.float64), A, B)                        # [m, 256]
        L = unnormalized_logits(H[sel].astype(np.float64), g_gain, W_U_act)
        _, marg = top2_margin(L)

        at = np.arange(sel.size)
        f_t, p_t = feat[at, rows[sel]], pri[at, rows[sel]]
        true_t = L[at, rows[sel]] - L.mean(axis=1)             # contrast-centred true margin
        s_share.append(bias_share(f_t, p_t, true_t - f_t - p_t))
        s_margin.append(marg)
        s_activity.append(np.abs(z[sel].astype(np.float64)).sum(axis=1))
        s_demo.append(dbin)
        s_emit.append((n_bins - rows[sel]).astype(np.float64))
        s_dev.append(np.abs((n_bins - rows[sel]) - dbin))
        s_feat.append(feat); s_prior.append(pri)
        print(f"[gates]     sampled {sum(x.shape[0] for x in s_feat)} rows for Gates 3-4",
              flush=True)

    mu_all = np.concatenate(MU); r_all = np.concatenate(RR)
    row_all = np.concatenate(ROW); slot_all = np.concatenate(SLOT)

    gates = {
        "gate0": gate0_bias_composition(mu_all, r_all, A[row_all], B[row_all]),
        "gate1": gate1_mu_over_r(mu_all, r_all),
        "gate2": gate2_prior_vs_marginal(A, B, mu_all, row_all, slot_all, n_bins),
    }

    if demos is not None and s_feat:
        share = np.concatenate(s_share); dev = np.concatenate(s_dev)
        emit, dbin = np.concatenate(s_emit), np.concatenate(s_demo)
        finite = np.isfinite(dbin)
        med = float(np.median(np.abs(emit[finite] - dbin[finite]))) if finite.any() else np.inf
        canary_ok = med <= CANARY_MEDIAN_FRAC * n_bins
        gates["demo_bin_canary"] = {
            "median_abs_model_minus_demo_bins": med,
            "threshold": CANARY_MEDIAN_FRAC * n_bins,
            "n_matched": int(finite.sum()),
            "pass": bool(canary_ok),
        }
        if canary_ok or args.force_34:
            gates["gate3"] = gate3_share_predicts_deviation(
                share, dev, np.concatenate(s_margin), np.concatenate(s_activity))
            gates["gate4"] = gate4_lambda_sweep(
                np.concatenate(s_feat), np.concatenate(s_prior), dbin, emit, share,
                n_bins=n_bins)
        else:
            print(f"[gates] REFUSING Gates 3-4: demo-bin canary failed (median {med:.1f} bins "
                  f"> {CANARY_MEDIAN_FRAC * n_bins:.1f}). The discretisation or norm_stats are "
                  f"wrong -- fix that before reading any deviation number. --force-34 overrides.")

    summary = {
        "acts_dir": args.acts_dir, "sae_dir": args.sae_dir, "layer": args.layer,
        "n_decisions": int(n_total), "n_bins": int(n_bins), "k": int(k),
        "gates": gates,
    }
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n[gates] === LADDER ===")
    print(verdict_table(gates))
    print(f"\n[gates] wrote {args.out}")
    print("[gates] SCOPE: demo replay, expert states, success is constant 1. These gates "
          "screen a hypothesis about the readout; none of them is evidence about success rate.")


if __name__ == "__main__":
    main()
