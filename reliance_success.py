"""Does an internal reliance signal predict episode failure? Scores the closed-loop rollouts.

Consumes `collect_action_rollouts.py`'s shards -- the first dataset carrying BOTH the
action-position residual and the episode outcome -- and asks whether a per-decision signal
computed from the residual predicts whether that episode failed.

WHAT IS BEING COMPARED, AND WHAT WOULD COUNT AS A RESULT
-------------------------------------------------------
Five signals, one of which is the point and one of which is the bar:

    mu_t@q25 / mu_t@q50   fraction of |phi| carried by the lowest-breadth quarter / half of
                          features. THE ONE THAT USES PATH A'S AXIS.
    share                 the constant prior's fraction of the margin (Gate 0 showed the
                          bias is a fixed vector, so this is "how weak were the features").
    phi_total             raw feature drive.
    margin                top-2 logit gap. THE BASELINE. A mechanistic measure that cannot
                          beat a single scalar read straight off the logits has not earned
                          its complexity, and every published VLA failure monitor already
                          has a scalar of this kind.

Six aggregates per episode: whole-episode mean and max, and the mean over the first
5/10/20/50 timesteps. The early windows are not decoration -- Section 3.2a requires
measuring BEFORE divergence, or a positive result is just "episodes already going wrong look
like they are going wrong". If only the whole-episode aggregate scores, that IS the reverse
causation and the curve shows it.

Everything is reported pooled AND per task. At ~76% success the failures may concentrate in
two or three hard tasks, in which case a pooled AUROC reads "reliance differs by task" and
calls it prediction.

THE CANARY THAT GATES THE WHOLE REPORT
--------------------------------------
The SAE was trained on demo-replay residuals; these are self-generated states. Sufficiency
was 0.936 on demo replay. It is recomputed here first, and if it falls below 0.80 the
driver refuses to print AUROCs -- because every signal would then be read off a
decomposition that no longer holds, and the numbers would look entirely reasonable.

Usage
-----
python reliance_success.py --rollout-dir $B/ROLLOUT_ACTION/goal \\
                           --sae-dir     $B/ACT_ACTION_SAE/goal/sae \\
                           --attr        $B/ATTR/goal_k100/layer_31_attribution.npz \\
                           --out         $B/ROLLOUT_ACTION/goal/reliance_success.json
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
import torch

from identify_features import adjusted_breadth
from mrvla.prior_gates import prior_vectors
from mrvla.readout import signature_matrix, unnormalized_logits
from mrvla.reliance import (
    DEFAULT_WINDOWS, aggregate_episodes, auroc_boot, reliance_signals, sufficiency,
)
from mrvla.stats import rankdata_average, tie_fraction
from run_attribution import load_sae, sae_encode_full

SIGNALS = ["mu_t@q25", "mu_t@q50", "share", "phi_total", "margin"]


def _spearman(x, y):
    """Rank correlation, ties averaged. Local because it needs no gate-ladder machinery."""
    rx = rankdata_average(np.asarray(x, float))
    ry = rankdata_average(np.asarray(y, float))
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    den = np.sqrt((rx * rx).sum() * (ry * ry).sum())
    return float((rx * ry).sum() / den) if den > 0 else float("nan")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--rollout-dir", required=True, help="collect_action_rollouts.py output")
    p.add_argument("--sae-dir", required=True)
    p.add_argument("--attr", required=True, help="layer_31_attribution.npz, for breadth")
    p.add_argument("--layer", type=int, default=31)
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--batch-size", type=int, default=4096)
    p.add_argument("--block", type=int, default=8192,
                   help="rows per signal block; caps the [block, F] temporaries at ~135 MB")
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--force", action="store_true",
                   help="report AUROCs even if the sufficiency canary fails (diagnosis only)")
    args = p.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)

    hc = np.load(os.path.join(args.rollout_dir, "head_constants.npz"))
    W_U_act = hc["W_U_act"].astype(np.float64)
    g_gain = hc["g"].astype(np.float64)
    eps = float(hc["eps"])
    n_bins, d = W_U_act.shape

    W_enc, W_dec_t, b_pre_t, k, ck = load_sae(args.sae_dir, args.layer)
    W_dec = W_dec_t.detach().float().cpu().numpy().astype(np.float64)
    b_pre = b_pre_t.detach().float().cpu().numpy().astype(np.float64)
    A, B = prior_vectors(W_U_act, g_gain, b_pre)
    S = signature_matrix(W_dec, g_gain, W_U_act, center=True)

    # Breadth comes from the DEMO-REPLAY attribution, deliberately: it is a property of the
    # feature, and reusing the published function keeps mu_t's low-breadth set identical to
    # the one every other analysis and selection in this repo used.
    at = np.load(args.attr)
    adj = adjusted_breadth(at["PR"].astype(np.float64), at["magnitude"].astype(np.float64),
                           at["base_rate"].astype(np.float64), at["is_active"].astype(bool))
    fin = np.isfinite(adj)
    masks = {}
    for q, name in ((0.25, "mu_t@q25"), (0.50, "mu_t@q50")):
        thr = np.quantile(adj[fin], q)
        m = np.zeros(adj.size, bool)
        m[fin] = adj[fin] <= thr
        masks[name] = m
    print(f"[rel] SAE {ck}  F={W_dec.shape[0]}  k={k}  "
          f"low-breadth sets: {masks['mu_t@q25'].sum()} / {masks['mu_t@q50'].sum()} of "
          f"{int(fin.sum())} active", flush=True)
    print(f"[rel] base_rate tie fraction {tie_fraction(at['base_rate']):.3f} -- the P9 "
          f"defect perturbs ordering inside tied blocks of this ranking (results.md S8.3)",
          flush=True)

    acc = {s: [] for s in SIGNALS}
    acc.update({k2: [] for k2 in ("feat", "bias", "true", "episode", "timestep",
                                  "task_id", "success", "slot", "exec_action", "bin")})

    for sp in sorted(glob.glob(os.path.join(args.rollout_dir, "shard_*.npz"))):
        dd = np.load(sp)
        res = dd["residual"]                                  # [n, 7, d]
        n, n_slot = res.shape[0], res.shape[1]
        H = res.reshape(n * n_slot, d).astype(np.float32)
        z, l2, mu = sae_encode_full(W_enc, b_pre_t, k, H, device, args.batch_size)

        for k2, arr in (("episode", dd["episode"]), ("timestep", dd["timestep"]),
                        ("task_id", dd["task_id"]), ("success", dd["success"])):
            acc[k2].append(np.repeat(arr.astype(np.int64), n_slot))
        # residual.reshape(n*7, d) and action.reshape(-1) are both row-major over (n, 7),
        # so slot s of decision i lands at the same flat index in both.
        acc["slot"].append(np.tile(np.arange(n_slot, dtype=np.int64), n))
        acc["exec_action"].append(dd["action"].reshape(-1).astype(np.float64))

        for s0 in range(0, H.shape[0], args.block):
            s1 = min(s0 + args.block, H.shape[0])
            Hb = H[s0:s1].astype(np.float64)
            r_b = np.sqrt(np.einsum("ij,ij->i", H[s0:s1], H[s0:s1], dtype=np.float64) / d + eps)
            L = unnormalized_logits(Hb, g_gain, W_U_act)
            rows = np.argmax(L, axis=1)
            acc["bin"].append((n_bins - rows).astype(np.float64))
            common = dict(z=z[s0:s1], l2=l2[s0:s1].astype(np.float64),
                          mu=mu[s0:s1].astype(np.float64), r=r_b,
                          S=S, A=A, B=B, L=L, rows=rows)
            out25 = reliance_signals(low_mask=masks["mu_t@q25"], **common)
            out50 = reliance_signals(low_mask=masks["mu_t@q50"], **common)
            acc["mu_t@q25"].append(out25["mu_t"])
            acc["mu_t@q50"].append(out50["mu_t"])
            for key in ("share", "phi_total", "margin", "feat", "bias", "true"):
                acc[key].append(out50[key])
        print(f"[rel]   {os.path.basename(sp)}: {sum(x.size for x in acc['margin'])} rows",
              flush=True)

    cat = {k2: np.concatenate(v) for k2, v in acc.items()}

    # The emitted bin is RECOVERED by argmax rather than stored (predict_action returns the
    # action, not the tokens). `action` was stored precisely so that recovery is checkable:
    # bin index and executed action are related by a fixed monotone detokenisation, so a
    # per-slot |rho| near 1 confirms it. This needs no norm_stats and would catch a wrong
    # action-row range, an off-by-one, or a slot/flattening mismatch -- none of which would
    # error, and all of which would quietly scramble every signal.
    recov = {}
    for sl in np.unique(cat["slot"]):
        m = cat["slot"] == sl
        recov[int(sl)] = abs(_spearman(cat["bin"][m], cat["exec_action"][m]))
    worst = min(recov.values()) if recov else float("nan")
    print(f"\n[rel] CANARY  argmax recovery |rho(bin, executed action)| per slot: "
          + " ".join(f"{v:.3f}" for v in recov.values()), flush=True)
    if not (worst >= 0.90):
        print(f"[rel] WARNING: worst slot is {worst:.3f}. The recomputed emitted bin does not "
              f"track the executed action, so `rows` is wrong and EVERY signal below is "
              f"indexed at the wrong bin.", flush=True)

    suff = sufficiency(cat["true"], cat["feat"], cat["bias"])
    print(f"\n[rel] CANARY  sufficiency on ROLLOUT residuals: "
          f"features+bias={suff['features_plus_bias']:.4f} "
          f"(features {suff['features']:.4f}, bias {suff['bias']:.4f}, "
          f"error {suff['error']:.4f}); demo replay was 0.9361", flush=True)
    if not suff["pass"] and not args.force:
        json.dump({"sufficiency": suff, "refused": True}, open(args.out, "w"), indent=2)
        raise SystemExit(
            f"[rel] REFUSING to score: sufficiency {suff['features_plus_bias']:.4f} is below "
            f"0.80, so the decomposition does not transfer to self-generated states and every "
            f"signal would be read off a basis that no longer holds. --force to override.")

    results: dict = {}
    tasks = np.unique(cat["task_id"])
    ep_ids = np.unique(cat["episode"])
    first = {int(e): int(cat["task_id"][np.searchsorted(cat["episode"], e)]) for e in ep_ids} \
        if np.all(np.diff(cat["episode"]) >= 0) else \
        {int(e): int(cat["task_id"][cat["episode"] == e][0]) for e in ep_ids}
    for sig in SIGNALS:
        agg = aggregate_episodes(cat[sig], cat["episode"], cat["timestep"], cat["success"],
                                 windows=DEFAULT_WINDOWS)
        fail = 1 - agg["success"]
        ep_task = np.array([first[int(e)] for e in agg["episodes"]])
        entry: dict = {"pooled": {}, "per_task": {}}
        for w in ["mean", "max"] + [f"first{x}" for x in DEFAULT_WINDOWS]:
            b = auroc_boot(agg[w], fail, n_boot=args.n_boot, seed=args.seed)
            b["discrimination"] = (abs(b["auroc"] - 0.5) + 0.5
                                   if np.isfinite(b["auroc"]) else float("nan"))
            entry["pooled"][w] = b
            entry["per_task"][w] = {
                str(int(t)): auroc_boot(agg[w][ep_task == t], fail[ep_task == t],
                                        n_boot=max(200, args.n_boot // 4), seed=args.seed)
                for t in tasks
            }
        results[sig] = entry

    n_ep = int(np.unique(cat["episode"]).size)
    n_fail = int(results["margin"]["pooled"]["mean"]["n_fail"])
    summary = {"rollout_dir": args.rollout_dir, "sae_dir": args.sae_dir,
               "n_rows": int(cat["margin"].size), "n_episodes": n_ep, "n_failures": n_fail,
               "sufficiency": suff, "signals": results}
    json.dump(summary, open(args.out, "w"), indent=2, default=float)

    print(f"\n[rel] {n_ep} episodes, {n_fail} failures\n")
    cols = ["mean", "max"] + [f"first{x}" for x in DEFAULT_WINDOWS]
    print(f"{'signal':<12}" + "".join(f"{c:>12}" for c in cols) + "   (AUROC vs failure)")
    for sig in SIGNALS:
        row = "".join(f"{results[sig]['pooled'][c]['auroc']:>12.3f}" for c in cols)
        print(f"{sig:<12}{row}")
    print("\n0.5 = no information. BELOW 0.5 is not a null -- it means the signal predicts "
          "SUCCESS.\nCompare every row against `margin`: a mechanistic signal that does not "
          "beat that scalar\nhas not earned its complexity. Check per_task in the JSON before "
          "believing any pooled value.")
    print(f"\n[rel] wrote {args.out}")


if __name__ == "__main__":
    main()
