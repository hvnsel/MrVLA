"""Among SUCCESSFUL episodes, does any internal signal predict how long the episode took?

Episode failure turned out to be a degenerate target: LIBERO's `done` fires only on success,
so a failure always runs to the step cap and duration predicts the label at AUROC 1.000.
Eight signals were scored against it and all came back null, with a two-line statistic on
the robot's own commanded actions beating every one of them.

This drops the failures and asks a question the label cannot answer for free: among the
episodes that DID succeed, does an internal signal measured in the first W steps predict how
long the episode ran? Duration is genuinely variable there and has no definitional link to
anything, which makes it the first uncontaminated target in this line. A rank correlation
against it also uses strictly more information than binarising would.

WHAT IS SCORED

  task_margin   NEW. How far the decision's causal profile sits above the OTHER tasks'
                canonical profiles, using the C matrix Path A built and never reused.
                Mechanism: a policy in an unfamiliar state may fall back on another task's
                machinery, which nothing else here can see. See mrvla/task_match.py for why
                the subtraction and the standardisation are both load-bearing.
  the eight     mu_t, share, phi_total, top-2 margin, coalition churn and period-2 returns,
                already dead against binary failure -- re-scored here because the target
                they were tested on was never fair to them.
  action churn  THE BASELINE, again. It won the last round at 0.678 discrimination. Anything
                mechanistic has to beat it or the dictionary was not needed.

CONTROLS

  within task   Tasks differ in typical duration, so a pooled correlation reads "this task
                is slow" as signal. Everything is computed inside a task and averaged.
  row shuffle   task_margin recomputed against a PERMUTED assignment of C rows. It must
                collapse; if it does not, the statistic is reading the always-on component
                every row shares (results.md P2b) rather than anything task-specific.
  magnitude     every signal is also reported partialled on total causal drive, the base-rate
                control this project applies to every score by standing commitment.

Usage
-----
python duration_efficiency.py --rollout-dir $B/ROLLOUT_ACTION/goal \\
                              --sae-dir     $B/ACT_ACTION_SAE/goal/sae \\
                              --attr        $B/ATTR/goal_k100/layer_31_attribution.npz \\
                              --out         $B/ROLLOUT_ACTION/goal/duration_efficiency.json
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
    action_dynamics, coalition_dynamics, episode_kinematics, topk_sets,
)
from mrvla.prior_gates import prior_vectors
from mrvla.readout import signature_matrix, unnormalized_logits
from mrvla.reliance import aggregate_episodes, reliance_signals
from mrvla.task_match import task_margin, task_similarity, within_task_partial, within_task_spearman
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
    p.add_argument("--window", type=int, default=50,
                   help="timesteps of each episode the signals are averaged over; must be "
                        "<= the shortest episode or short ones drop out and the sample "
                        "silently changes")
    p.add_argument("--top-m", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
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
    A_vec, B_vec = prior_vectors(W_U_act, g_gain, b_pre)
    S = signature_matrix(W_dec, g_gain, W_U_act, center=True)
    S_abs = np.abs(S)
    F = W_dec.shape[0]

    at = np.load(args.attr)
    C = at["C"].astype(np.float64)                      # [G, F], the never-reused artifact
    adj = adjusted_breadth(at["PR"].astype(np.float64), at["magnitude"].astype(np.float64),
                           at["base_rate"].astype(np.float64), at["is_active"].astype(bool))
    fin = np.isfinite(adj)
    low = np.zeros(F, bool)
    low[fin] = adj[fin] <= np.median(adj[fin])
    print(f"[dur] SAE {ck}  F={F}  k={k}  C={C.shape}  window={args.window} steps", flush=True)

    acc = {n: [] for n in ("mu_t", "share", "phi_total", "margin")}
    sim_a, ep_a, ts_a, sl_a, sc_a, tk_a, sets_a = [], [], [], [], [], [], []
    act_a, aep_a, ats_a, asc_a, atk_a = [], [], [], [], []

    for sp in sorted(glob.glob(os.path.join(args.rollout_dir, "shard_*.npz"))):
        dd = np.load(sp)
        res = dd["residual"]
        n, n_slot = res.shape[0], res.shape[1]
        H = res.reshape(n * n_slot, d).astype(np.float32)
        z, l2, mu = sae_encode_full(W_enc, b_pre_t, k, H, device, args.batch_size)

        ep_a.append(np.repeat(dd["episode"].astype(np.int64), n_slot))
        ts_a.append(np.repeat(dd["timestep"].astype(np.int64), n_slot))
        tk_a.append(np.repeat(dd["task_id"].astype(np.int64), n_slot))
        sc_a.append(np.repeat(dd["success"].astype(np.int64), n_slot))
        sl_a.append(np.tile(np.arange(n_slot, dtype=np.int64), n))
        act_a.append(dd["action"].astype(np.float64))
        aep_a.append(dd["episode"].astype(np.int64)); ats_a.append(dd["timestep"].astype(np.int64))
        asc_a.append(dd["success"].astype(np.int64)); atk_a.append(dd["task_id"].astype(np.int64))

        for s0 in range(0, H.shape[0], args.block):
            s1 = min(s0 + args.block, H.shape[0])
            Hb = H[s0:s1].astype(np.float64)
            r_b = np.sqrt(np.einsum("ij,ij->i", H[s0:s1], H[s0:s1], dtype=np.float64) / d + eps)
            L = unnormalized_logits(Hb, g_gain, W_U_act)
            rows = np.argmax(L, axis=1)
            zb = z[s0:s1].astype(np.float64)
            phi_abs = zb * S_abs[:, rows].T                 # scale-free: cosine ignores it
            sim_a.append(task_similarity(phi_abs, C))
            sets_a.append(topk_sets(phi_abs, args.top_m))
            out = reliance_signals(z=zb, l2=l2[s0:s1].astype(np.float64),
                                   mu=mu[s0:s1].astype(np.float64), r=r_b,
                                   S=S, A=A_vec, B=B_vec, L=L, rows=rows, low_mask=low)
            for nm in acc:
                acc[nm].append(out[nm])
        print(f"[dur]   {os.path.basename(sp)}: {sum(x.shape[0] for x in sim_a)} rows",
              flush=True)

    ep, ts, sl, sc, tk = (np.concatenate(x) for x in (ep_a, ts_a, sl_a, sc_a, tk_a))
    sim = np.concatenate(sim_a)
    sets = np.concatenate(sets_a)
    act = np.concatenate(act_a)
    aep, ats, asc, atk = (np.concatenate(x) for x in (aep_a, ats_a, asc_a, atk_a))
    sig = {nm: np.concatenate(v) for nm, v in acc.items()}

    rng = np.random.default_rng(args.seed)
    G = C.shape[0]
    perm = rng.permutation(G)
    while np.any(perm == np.arange(G)):                     # a fixed point is not a shuffle
        perm = rng.permutation(G)
    sig["task_margin"] = task_margin(sim, tk)
    sig["task_margin_SHUF"] = task_margin(sim, tk, row_perm=perm)

    dyn = coalition_dynamics(sets, ep, sl, ts)
    sig["feature_churn"], sig["feature_returns"] = dyn["churn"], dyn["returns"]
    base = action_dynamics(act, aep, ats)

    # ---- per episode, successes only --------------------------------------
    W = args.window
    per_ep = {}
    for nm, v in sig.items():
        per_ep[nm] = aggregate_episodes(v, ep, ts, sc, windows=(W,))
    a_ch = aggregate_episodes(base["churn"], aep, ats, asc, windows=(W,))
    a_rt = aggregate_episodes(base["returns"], aep, ats, asc, windows=(W,))
    per_ep["action_churn"], per_ep["action_returns"] = a_ch, a_rt

    kin = episode_kinematics(act, aep, ats, W)
    assert np.array_equal(kin["episodes"], per_ep["action_churn"]["episodes"])
    for nm in ("path", "net", "straight", "rot"):
        per_ep[f"kin_{nm}"] = {f"first{W}": kin[nm], "episodes": kin["episodes"]}

    ref = per_ep["task_margin"]
    ep_ids = ref["episodes"]
    ok = ref["success"] == 1
    dur = ref["length"].astype(np.float64)
    ep_task = np.array([int(tk[ep == e][0]) for e in ep_ids])
    print(f"\n[dur] {ok.sum()} successful episodes of {ep_ids.size}; "
          f"duration among successes {int(dur[ok].min())}-{int(dur[ok].max())} steps "
          f"(median {int(np.median(dur[ok]))}). Window {W} <= shortest "
          f"{int(dur[ok].min())}: {'OK' if W <= dur[ok].min() else 'TOO LONG'}", flush=True)

    # The two aggregations are keyed on the same episode ids and both sort by np.unique,
    # so the vectors align. Assert it rather than assume: a silent misalignment here would
    # pair each episode's signal with another episode's duration and still look plausible.
    assert np.array_equal(per_ep["action_churn"]["episodes"], ep_ids), \
        "slot-level and timestep-level aggregations disagree on the episode set"

    mag = per_ep["phi_total"][f"first{W}"]
    ach = per_ep["action_churn"][f"first{W}"]
    dist = [per_ep[f"kin_{n}"][f"first{W}"][ok] for n in ("path", "net", "straight")]
    CONTROLS = ("phi_total", "action_churn", "kin_path", "kin_net", "kin_straight", "kin_rot")
    results = {}
    for nm, agg in per_ep.items():
        x, y, t = agg[f"first{W}"][ok], dur[ok], ep_task[ok]
        # A control partialled on itself is degenerate, so those cells stay empty rather
        # than showing a zero that would read as a collapsed effect.
        results[nm] = {
            "raw": within_task_spearman(x, y, t),
            "partial_magnitude": (None if nm == "phi_total"
                                  else within_task_partial(x, y, [mag[ok]], t)),
            "partial_action_churn": (None if nm == "action_churn"
                                     else within_task_partial(x, y, [ach[ok]], t)),
            "partial_distance": (None if nm in CONTROLS
                                 else within_task_partial(x, y, dist, t)),
            "partial_all": (None if nm in CONTROLS
                            else within_task_partial(x, y, [mag[ok], ach[ok]] + dist, t)),
        }

    summary = {"rollout_dir": args.rollout_dir, "window": W, "row_perm": perm.tolist(),
               "n_success": int(ok.sum()), "n_episodes": int(ep_ids.size),
               "duration_range": [float(dur[ok].min()), float(dur[ok].max())],
               "results": results}
    json.dump(summary, open(args.out, "w"), indent=2, default=float)

    def cell(v):
        return f"{v['mean']:+.3f} {v['n_positive']}/{v['n_tasks']}" if v else "  (control)"

    print(f"\n{'signal':<20}{'rho|task':>10}{'+tasks':>8}"
          f"{'|mag':>13}{'|churn':>13}{'|distance':>13}{'|ALL':>13}")
    for nm in ("task_margin", "task_margin_SHUF", "mu_t", "share", "margin",
               "feature_churn", "feature_returns", "action_returns",
               "phi_total", "action_churn", "kin_path", "kin_net", "kin_straight", "kin_rot"):
        r = results[nm]
        print(f"{nm:<20}{r['raw']['mean']:>+10.3f}{r['raw']['n_positive']:>5}/"
              f"{r['raw']['n_tasks']:<3}{cell(r['partial_magnitude']):>13}"
              f"{cell(r['partial_action_churn']):>13}"
              f"{cell(r.get('partial_distance')):>13}{cell(r.get('partial_all')):>13}")
    print("\nPositive rho = the signal is HIGHER in episodes that took LONGER.")
    print("Each partial cell shows the mean and how many of the 10 tasks agreed in sign.")
    print("task_margin must beat task_margin_SHUF, or it is reading what every C row shares.")
    print("\nkin_* are the DISTANCE controls, derived from the commanded delta-poses: how")
    print("far the robot travelled, how far it got, and how straight it went. If an episode")
    print("simply started further from the object it would take longer AND leave the early")
    print("features with less to say, so distance is the leading common cause. Read the")
    print("kin_ rows first: if they predict duration strongly and `share` collapses in")
    print("|distance, the signal was distance. They measure how far it DID move, not how far")
    print("it NEEDED to, so they bound the confound rather than eliminating it.\n")
    print("THE COLUMN THAT DECIDES IT is |action_churn. A stuck robot repeats its command,")
    print("and gate 2 showed the constant prior favours the modal action bins -- so `share`")
    print("and action churn may be one phenomenon read two ways. If `share` survives that")
    print("partial it carries information the commanded actions do not; if it collapses, a")
    print("two-line statistic on the robot's own commands is as good as the dictionary.")
    print(f"\n[dur] wrote {args.out}")


if __name__ == "__main__":
    main()
