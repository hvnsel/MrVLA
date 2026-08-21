"""B1 Stage 1: slot-resolved attribution plus exact per-feature necessity, in one pass.

`run_attribution.py` accumulates C[task, feature] and discards the action slot. This re-streams
the same A1 shards and keeps it, producing two things the project does not currently have:

1. C_slot [7, G, F] -- causal mass per (action channel, task, feature), in BOTH absolute and
   decision-share form. The share form is the comparable one; see mrvla/channels.py for why
   absolute |phi| cannot be compared across slots.

2. Exact counterfactual necessity per (feature, slot). For every candidate feature on every
   decision we compute whether removing it changes the emitted action bin -- exactly, in logit
   space, with no model forward pass (see mrvla/readout.py). At ~446k slot-decisions this gives
   a necessity estimate with a standard error near 0.1%, against the ~9-point minimum detectable
   effect of the 200-episode closed-loop ablation. It answers a narrower question than a rollout
   -- the direct effect on one decode slot, not task success -- but it answers it decisively,
   and the two together are far stronger than either alone.

A speed note that also matters for correctness: run_attribution computes the alignment term with
a per-row matvec inside a Python loop over all 446k rows. It is unnecessary. Since
u_contrast = u_t - mean_s u_s, the alignment <w_j, g (*) u_contrast> is exactly column t of the
CONTRAST-CENTRED signature matrix. So the whole loop collapses to a column gather from a matrix
computed once. Same numbers, vectorised.

VALIDATION BUILT IN. The recomputed argmax over the 256 action bins must equal the token the
model actually emitted, stored in the shard. That agreement rate is printed first and gates the
run: if it is not ~1.0, the residuals, the head constants, or the token-id mapping are
mismatched, and every number downstream would be quietly wrong.

Two ablation semantics are supported (--coeff): `projection` reproduces the closed-loop hook
(h - <h,w>w) and needs no SAE encoder, so it runs on CPU without torch; `coded` reproduces
Path A's phi (remove l2*z_j*w_j) and needs the encoder. Running both and reporting the gap
measures how much of a projection ablation acts on structure the SAE never attributed to the
feature.

Usage
-----
python run_channel_attribution.py \
    --acts-dir $B/ACT_ACTION/goal --sae-dir $B/ACT_ACTION_SAE/goal/sae \
    --attr $B/ATTR/goal_k100/layer_31_attribution.npz \
    --layer 31 --top 100 --out $B/CHANNELS/goal
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np

from mrvla.channels import (
    DEFAULT_CHANNEL_NAMES, accumulate_slot_task, channel_participation_ratio, channel_profile,
    decision_shares, slot_index, transition_mask,
)
from mrvla.readout import (
    assert_unit_rows, coded_coeffs, projection_coeffs, signature_matrix, signed_bin_shift,
    single_feature_flips, top2_margin, unnormalized_logits,
)


def pick_candidates(attr_path: str, top: int) -> dict:
    """Which features get the (quadratic) counterfactual treatment.

    Necessity is computed for a candidate set rather than all F, because cost is
    decisions x candidates x bins. The set is the two ends of the confound-adjusted breadth
    ranking plus matched controls, which is exactly the contrast B1 needs -- and it reuses
    `identify_features`' selection so the features here are the same ones the ablation and
    steering runs target.
    """
    from identify_features import adjusted_breadth, select_general_specialist
    A = np.load(attr_path)
    PR = A["PR"].astype(np.float64)
    mag = A["magnitude"].astype(np.float64)
    base = A["base_rate"].astype(np.float64)
    active = A["is_active"].astype(bool)
    adj = adjusted_breadth(PR, mag, base, active)
    general, specialist = select_general_specialist(adj, mag, active, top)
    rng = np.random.default_rng(0)
    pool = np.where(active & np.isfinite(adj))[0]
    chosen = set(general) | set(specialist)
    cand = np.array([j for j in pool if j not in chosen])
    random_set = rng.choice(cand, size=min(top, cand.size), replace=False).tolist() \
        if cand.size else []
    firing = np.argsort(np.where(active, base, -np.inf))[::-1][:top].tolist()
    groups = {"general": list(map(int, general)), "specialist": list(map(int, specialist)),
              "random": list(map(int, random_set)), "firing": list(map(int, firing))}
    order = sorted({j for v in groups.values() for j in v})
    return {"groups": groups, "features": order,
            "adjusted_breadth": {int(j): float(adj[j]) for j in order},
            "task_PR": {int(j): float(PR[j]) for j in order}}


def per_channel_transition(tok_rows: np.ndarray, n: int, n_sl: int,
                           episode: np.ndarray, timestep: np.ndarray) -> np.ndarray:
    """[n*n_sl] bool: did THIS slot's own emitted bin change from the previous timestep?

    Fixes results.md P5d. The mask was previously built from the gripper's tokens alone and
    broadcast to all seven slots with np.repeat, so `flip_trans` for dx meant "dx flipped at a
    timestep where the GRIPPER moved" -- a control for nothing, since dx changes on ~95% of
    steps regardless. The whole `_trans` family was therefore not comparable across channels,
    and any table mixing the gripper's `_trans` with dx's compared two different conditionings.

    Fed ROW indices (tok - id0), not bin indices. `bin_index_from_row` is n_bins - row, strictly
    monotone, so "changed" is identical either way -- but mrvla/readout.py's row-vs-bin note
    exists because this axis has been got backwards before, so it is said out loud here.

    A row whose token is out of range is excluded downstream by `valid`, but it still acts as
    its neighbour's predecessor. That is arguably right ("the bin did change") and the effect is
    bounded by the invalid fraction, which is ~0.008.
    """
    grid = np.asarray(tok_rows).reshape(n, n_sl)
    out = np.zeros((n, n_sl), dtype=bool)
    for s_i in range(n_sl):
        out[:, s_i] = transition_mask(grid[:, s_i], episode, timestep)
    return out.reshape(-1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--acts-dir", required=True, help="A1 shard dir (+ head_constants.npz)")
    ap.add_argument("--sae-dir", required=True)
    ap.add_argument("--attr", required=True, help="layer_NN_attribution.npz, for candidates")
    ap.add_argument("--layer", type=int, default=31)
    ap.add_argument("--top", type=int, default=100, help="group size for candidate selection")
    ap.add_argument("--all-features", action="store_true",
                    help="run the counterfactual on ALL F features instead of the candidate "
                         "set. Required for any statistic CORRELATED across features: the "
                         "candidate groups are the two ends of the adjusted-breadth ranking, "
                         "so selecting on the predictor inflates the correlation. The groups "
                         "are still computed and recorded, so group contrasts are unaffected. "
                         "Do NOT approximate this by raising --top: once top exceeds the "
                         "eligible count, select_general_specialist returns the same set for "
                         "both ends.")
    ap.add_argument("--coeff", default="both", choices=["projection", "coded", "both"],
                    help="ablation semantics for the counterfactual; see module docstring")
    ap.add_argument("--n-slots", type=int, default=7)
    ap.add_argument("--gripper-slot", type=int, default=6,
                    help="LABELLING ONLY. The transition control is now built per channel from "
                         "each slot's own emitted bins, so it no longer depends on this. Kept "
                         "so downstream scripts know which slot is the gripper (OpenVLA action "
                         "order is assumed [dx,dy,dz,droll,dpitch,dyaw,gripper]).")
    ap.add_argument("--max-decisions", type=int, default=0,
                    help="cap on decisions used for the COUNTERFACTUAL layer (0 = all). The "
                         "C_slot accumulation always uses every decision; only the quadratic "
                         "necessity pass is subsampled, and 100k decisions already give a flip "
                         "rate to ~0.1%%.")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch-size", type=int, default=4096)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    S_SLOTS = args.n_slots

    hc = np.load(os.path.join(args.acts_dir, "head_constants.npz"))
    W_U_act = hc["W_U_act"].astype(np.float64)
    act_ids = hc["act_ids"].astype(np.int64)
    g_gain = hc["g"].astype(np.float64)
    eps = float(hc["eps"]) if "eps" in hc else 1e-5
    id0, n_act = int(act_ids[0]), int(act_ids.size)

    import torch  # noqa: F401  (needed by load_sae; the readout math itself is numpy)
    from run_attribution import load_sae, sae_encode_full
    W_enc, W_dec_t, b_pre_t, k, ck = load_sae(args.sae_dir, args.layer)
    W_dec = W_dec_t.detach().float().cpu().numpy().astype(np.float64)
    F = W_dec.shape[0]
    dev = assert_unit_rows(W_dec)
    print(f"[chan] SAE {ck}  W_dec {W_dec.shape}  k={k}  max|row norm - 1| = {dev:.2e}")
    if dev > 1e-2:
        print("[chan] WARNING decoder rows are not unit norm; projection-mode coefficients "
              "assume they are (the rollout hook re-normalises internally, so the two would "
              "disagree). Investigate before trusting projection numbers.")

    S_raw = signature_matrix(W_dec, g_gain, W_U_act, center=False)     # [F, 256]
    S_cen = S_raw - S_raw.mean(axis=1, keepdims=True)                  # alignment lookup table

    # Per-slot sufficiency needs the SAE's constant term (mu*1 + b_pre) projected on the same
    # contrast direction. Both halves are inner products with u_contrast(t), so each collapses
    # to a 256-vector lookup computed once -- the same trick that turns the alignment matvec
    # into a gather. Nothing here costs a pass over the data.
    b_pre_np = b_pre_t.detach().float().cpu().numpy().astype(np.float64)
    U_c = W_U_act - W_U_act.mean(axis=0, keepdims=True)                # [256, d]
    G1 = U_c @ g_gain                                                  # <1 (*) g, u_c(t)>
    BG = U_c @ (b_pre_np * g_gain)                                     # <b_pre (*) g, u_c(t)>

    # The groups are computed either way: `analyze_channels.py` reads them from summary.json for
    # its general-vs-specialist contrasts, and they cost nothing. --all-features only widens the
    # set the counterfactual runs on. With feats = arange(F) the row index IS the feature index,
    # so analyze_channels' `pos` map becomes the identity and nothing downstream changes.
    cand = pick_candidates(args.attr, args.top)
    if args.all_features:
        feats = np.arange(F, dtype=np.int64)
        print(f"[chan] ALL {feats.size} features (groups still recorded: "
              + ", ".join(f"{k2}={len(v)}" for k2, v in cand["groups"].items()) + ")")
    else:
        feats = np.array(cand["features"], dtype=np.int64)
        print(f"[chan] {feats.size} candidate features "
              + ", ".join(f"{k2}={len(v)}" for k2, v in cand["groups"].items()))

    shards = sorted(glob.glob(os.path.join(args.acts_dir, "shard_*.npz")))
    if not shards:
        raise SystemExit(f"no shard_*.npz in {args.acts_dir}")
    task_ids_seen: set[int] = set()
    for sp in shards:
        task_ids_seen.update(np.unique(np.load(sp)["task_id"]).tolist())
    task_list = sorted(task_ids_seen)
    task_row = {t: i for i, t in enumerate(task_list)}
    G = len(task_list)
    print(f"[chan] {len(shards)} shards, {G} tasks, {S_SLOTS} slots")

    C_abs = np.zeros((S_SLOTS, G, F))
    C_shr = np.zeros((S_SLOTS, G, F))
    n_cell = np.zeros((S_SLOTS, G), dtype=np.int64)
    modes = ["projection", "coded"] if args.coeff == "both" else [args.coeff]
    # `_active` variants restrict to decisions where the feature actually FIRES. Without them
    # the necessity comparison is a base-rate comparison: a feature that fires twice as often
    # has twice the opportunity to change an action, and in coded mode an inactive feature has
    # coefficient zero and literally cannot flip anything. Rate-given-active is the confound-free
    # form and is what the general-vs-specialist contrast must be read from.
    # Accumulated per (feature, TASK, slot). The task axis is what makes a leave-one-task-out
    # causal analysis possible at all: without it the only available necessity statistic is
    # pooled over tasks, which measures average decisiveness rather than SCOPE -- and scope is
    # what breadth claims. The [feature, slot] arrays the rest of the project reads are derived
    # by summing this over tasks at write time, so nothing downstream changes.
    FLIP_KEYS = ("n", "flip", "flip_trans", "n_trans", "n_active", "flip_active",
                 "n_active_trans", "flip_active_trans")
    flips = {m: {**{k2: np.zeros((feats.size, G, S_SLOTS), dtype=np.int64) for k2 in FLIP_KEYS},
                 "bin_shift": np.zeros((feats.size, G, S_SLOTS))} for m in modes}
    agree_n = agree_ok = 0
    n_cf_used = 0
    # Sufficiency, accumulated per slot as through-origin slope sums (see the report below).
    #
    # `ff` = sum(features^2) is what makes the slope interpretable. The slope sum(true*feat) /
    # sum(true^2) is a PROJECTION, and a projection of zero is equally consistent with "the
    # features contribute nothing" and "the features contribute plenty, orthogonally to the
    # margin". Simulated, those two return the same slope while their energies differ by four
    # orders of magnitude. energy = ff/tt separates them and nothing else here can.
    #
    # The `_trans` copies restrict to decisions where THIS channel's command actually changes.
    # Without them a low-entropy channel (the gripper is unchanged on ~95% of timesteps) is
    # scored almost entirely on decisions that were already settled, where a constant term fits
    # a constant target and a varying one cannot. Simulation shows features explaining EVERY
    # transition perfectly would still report a pooled slope near zero.
    suff = {k: np.zeros(S_SLOTS) for k in ("tt", "t_recon", "t_feat", "ff",
                                           "tt_trans", "t_recon_trans", "t_feat_trans",
                                           "ff_trans")}
    suff_n = np.zeros(S_SLOTS, dtype=np.int64)
    suff_n_trans = np.zeros(S_SLOTS, dtype=np.int64)

    for sp in shards:
        dd = np.load(sp)
        res = dd["residual"]; toks = dd["token_ids"].astype(np.int64)
        task_of = dd["task_id"].astype(np.int64)
        ep_of = dd["episode"].astype(np.int64); ts_of = dd["timestep"].astype(np.int64)
        n, n_sl, d = res.shape
        X = res.reshape(n * n_sl, d).astype(np.float64)
        tok_rows = (toks - id0).reshape(-1)
        valid = (tok_rows >= 0) & (tok_rows < n_act)
        slots = slot_index(n, n_sl)
        rows_task = np.array([task_row[int(t)] for t in task_of]).repeat(n_sl)

        L = unnormalized_logits(X, g_gain, W_U_act)                    # [n*7, 256]
        base_arg, base_margin = top2_margin(L)
        agree_n += int(valid.sum())
        agree_ok += int((base_arg[valid] == tok_rows[valid]).sum())

        z, l2, mu = sae_encode_full(W_enc, b_pre_t, k, X.astype(np.float32), args.device,
                                    args.batch_size)
        z = np.asarray(z, dtype=np.float64)
        r_scal = np.sqrt((X * X).mean(axis=1) + eps)
        align = S_cen.T[np.clip(tok_rows, 0, n_act - 1)]               # [n*7, F] column gather
        phi_abs = np.abs((l2 / r_scal)[:, None] * z * align)
        phi_abs[~valid] = 0.0

        # --- PER-CHANNEL transition mask. Each slot is clocked by ITS OWN emitted bins.
        #
        # This was previously built from the gripper's tokens alone and broadcast to all seven
        # slots with np.repeat, which made `flip_trans` for dx mean "dx flipped at a timestep
        # where the GRIPPER moved" -- a control for nothing, since dx changes on ~95% of steps
        # regardless. The `_trans` family was consequently not comparable across channels, and
        # any table mixing the gripper's _trans with dx's _trans compared two different
        # conditionings. --gripper-slot now only labels output.
        trans = per_channel_transition(tok_rows, n, n_sl, ep_of, ts_of)

        # --- per-slot sufficiency: what share of THIS channel's action margin do the
        # features additively recover? The margin decomposes exactly at frozen r as
        # true = features + (mu + b_pre) bias + error, and every term is already in hand.
        tok_safe = np.clip(tok_rows, 0, n_act - 1)
        true_c = (L[np.arange(L.shape[0]), tok_safe] - L.mean(axis=1)) / r_scal
        phi_sum = (l2 / r_scal) * (z * align).sum(axis=1)          # features alone (signed)
        const_c = (mu * G1[tok_safe] + BG[tok_safe]) / r_scal      # the constant bias term
        recon_c = phi_sum + const_c
        for s_i in range(n_sl):
            m = (slots == s_i) & valid
            if m.any():
                suff["tt"][s_i] += float((true_c[m] * true_c[m]).sum())
                suff["t_recon"][s_i] += float((true_c[m] * recon_c[m]).sum())
                suff["t_feat"][s_i] += float((true_c[m] * phi_sum[m]).sum())
                suff["ff"][s_i] += float((phi_sum[m] * phi_sum[m]).sum())
                suff_n[s_i] += int(m.sum())
            mt = m & trans
            if mt.any():
                suff["tt_trans"][s_i] += float((true_c[mt] * true_c[mt]).sum())
                suff["t_recon_trans"][s_i] += float((true_c[mt] * recon_c[mt]).sum())
                suff["t_feat_trans"][s_i] += float((true_c[mt] * phi_sum[mt]).sum())
                suff["ff_trans"][s_i] += float((phi_sum[mt] * phi_sum[mt]).sum())
                suff_n_trans[s_i] += int(mt.sum())

        accumulate_slot_task(C_abs, n_cell, phi_abs, slots, rows_task)
        accumulate_slot_task(C_shr, np.zeros_like(n_cell), decision_shares(phi_abs),
                             slots, rows_task)

        # One bucket id per row, so every (task, slot) cell is filled by a single bincount
        # instead of G x S boolean reductions over the full [n*7] row axis.
        bucket = rows_task * n_sl + slots
        n_bucket = G * n_sl
        use = valid.copy()
        if args.max_decisions and n_cf_used >= args.max_decisions:
            use[:] = False
        n_cf_used += n
        if use.any():
            for mode in modes:
                coeffs = (projection_coeffs(X, W_dec[feats]) if mode == "projection"
                          else coded_coeffs(z[:, feats], l2))
                # `n` and `n_trans` do not depend on the feature, so they are counted once for
                # the whole shard rather than feats.size times.
                cnt = lambda mask, w=None: np.bincount(
                    bucket[mask], weights=None if w is None else w[mask],
                    minlength=n_bucket).reshape(G, n_sl)
                n_all = cnt(use).astype(np.int64)
                n_tr = cnt(use & trans).astype(np.int64)

                for fi in range(feats.size):
                    res_f = single_feature_flips(L, S_raw[feats[fi]], coeffs[:, fi],
                                                 base_arg, base_margin)
                    fl = res_f["flipped"] & use
                    act = z[:, feats[fi]] > 0            # did this feature actually fire here?
                    # report the shift on the ACTION axis, not the row axis: they run opposite
                    shift = signed_bin_shift(res_f["base_argmax"], res_f["new_argmax"], n_act)
                    ma = use & act
                    flips[mode]["n"][fi] += n_all
                    flips[mode]["n_trans"][fi] += n_tr
                    flips[mode]["flip"][fi] += cnt(fl).astype(np.int64)
                    flips[mode]["flip_trans"][fi] += cnt(fl & trans).astype(np.int64)
                    flips[mode]["n_active"][fi] += cnt(ma).astype(np.int64)
                    flips[mode]["flip_active"][fi] += cnt(fl & ma).astype(np.int64)
                    flips[mode]["n_active_trans"][fi] += cnt(ma & trans).astype(np.int64)
                    flips[mode]["flip_active_trans"][fi] += cnt(fl & ma & trans).astype(np.int64)
                    flips[mode]["bin_shift"][fi] += cnt(fl, shift.astype(np.float64))
        print(f"[chan]   {os.path.basename(sp)}: decisions={n}  "
              f"argmax agreement so far={agree_ok / max(agree_n, 1):.4f}", flush=True)

    agreement = agree_ok / max(agree_n, 1)
    print(f"\n[chan] VALIDATION recomputed argmax == emitted token : {agreement:.4f}")
    if agreement < 0.99:
        print("[chan] *** agreement is not ~1.0. The residuals, head constants, or token-id\n"
              "[chan]     mapping do not line up, and every number below is unreliable.\n"
              "[chan]     Do not interpret this run until it is resolved. ***")

    with np.errstate(divide="ignore", invalid="ignore"):
        suff_recon = np.where(suff["tt"] > 0, suff["t_recon"] / suff["tt"], np.nan)
        suff_feat = np.where(suff["tt"] > 0, suff["t_feat"] / suff["tt"], np.nan)
        # energy: how much of the margin's scale the feature term carries, REGARDLESS of
        # direction. A near-zero slope with a large energy means the features are loud and
        # orthogonal, not absent -- the slope alone cannot tell those apart.
        suff_energy = np.where(suff["tt"] > 0, suff["ff"] / suff["tt"], np.nan)
        suff_recon_tr = np.where(suff["tt_trans"] > 0,
                                 suff["t_recon_trans"] / suff["tt_trans"], np.nan)
        suff_feat_tr = np.where(suff["tt_trans"] > 0,
                                suff["t_feat_trans"] / suff["tt_trans"], np.nan)
        suff_energy_tr = np.where(suff["tt_trans"] > 0,
                                  suff["ff_trans"] / suff["tt_trans"], np.nan)
        # cosine between the feature term and the true margin = slope / sqrt(energy). The
        # scale-free version of the same statement: slope conflates "how much" with "which
        # direction", and on a channel where the features are small but systematically opposed
        # those are different findings. Reported because a slope near zero at cosine -0.4 is
        # not the same result as a slope near zero at cosine 0.
        suff_cos = np.where(suff["ff"] > 0, suff["t_feat"] / np.sqrt(suff["tt"] * suff["ff"]),
                            np.nan)
        suff_cos_tr = np.where(suff["ff_trans"] > 0,
                               suff["t_feat_trans"] / np.sqrt(suff["tt_trans"]
                                                              * suff["ff_trans"]), np.nan)

    denom = np.maximum(n_cell, 1)[:, :, None]
    C_abs /= denom
    C_shr /= denom
    pr_chan = channel_participation_ratio(C_abs)
    pr_chan_share = channel_participation_ratio(C_shr)

    np.savez_compressed(os.path.join(args.out, f"layer_{args.layer:02d}_channels.npz"),
                        C_slot_abs=C_abs.astype(np.float32),
                        C_slot_share=C_shr.astype(np.float32),
                        n_cell=n_cell, task_ids=np.array(task_list),
                        channel_pr_abs=pr_chan.astype(np.float32),
                        channel_pr_share=pr_chan_share.astype(np.float32),
                        channel_profile_share=channel_profile(C_shr).astype(np.float32),
                        candidate_features=feats,
                        sufficiency_recon=suff_recon.astype(np.float32),
                        sufficiency_features_only=suff_feat.astype(np.float32),
                        sufficiency_n=suff_n,
                        sufficiency_energy=suff_energy.astype(np.float32),
                        sufficiency_recon_trans=suff_recon_tr.astype(np.float32),
                        sufficiency_features_only_trans=suff_feat_tr.astype(np.float32),
                        sufficiency_energy_trans=suff_energy_tr.astype(np.float32),
                        sufficiency_cosine=suff_cos.astype(np.float32),
                        sufficiency_cosine_trans=suff_cos_tr.astype(np.float32),
                        sufficiency_n_trans=suff_n_trans,
                        # `flip_{mode}_{key}` stays [F, S], summed over tasks, because
                        # analyze_channels.py indexes it two-dimensionally and every Step 5
                        # necessity number is read from it. `_gt` carries the (task, slot)
                        # resolution the leave-one-task-out causal analysis needs.
                        **{f"flip_{m}_{key}": v.sum(axis=1) for m in modes
                           for key, v in flips[m].items()},
                        **{f"flip_{m}_{key}_gt": v for m in modes
                           for key, v in flips[m].items()})

    names = list(DEFAULT_CHANNEL_NAMES)[:S_SLOTS]
    summary = {"acts_dir": args.acts_dir, "sae": ck, "layer": args.layer,
               "n_tasks": G, "n_slots": S_SLOTS, "n_features": int(F),
               "argmax_agreement": agreement, "decoder_unit_norm_dev": dev,
               "channel_names": names, "groups": cand["groups"],
               "decisions_per_slot_task": n_cell.tolist(),
               # provenance: a subsampled or narrowed run was previously not self-describing,
               # so a later reader could not tell how much of the data the counterfactual saw
               "coeff": args.coeff, "top": args.top, "all_features": bool(args.all_features),
               "gripper_slot": args.gripper_slot, "n_candidates": int(feats.size),
               # THE critical provenance field. Before this run the transition mask was the
               # gripper's, broadcast to all seven slots; a reader tabling new `_trans` numbers
               # against the old ones would be comparing two different conditionings without
               # any way to notice. Downstream analyses refuse to run on the wrong value.
               "trans_mask": "per_channel", "task_ids": [int(t) for t in task_list],
               "flip_counter_layout": "feature_task_slot",
               "max_decisions": args.max_decisions, "n_decisions_counterfactual": int(n_cf_used),
               "sufficiency_recon_per_slot": [float(v) for v in suff_recon],
               "sufficiency_features_only_per_slot": [float(v) for v in suff_feat],
               "sufficiency_energy_per_slot": [float(v) for v in suff_energy],
               "sufficiency_recon_trans_per_slot": [float(v) for v in suff_recon_tr],
               "sufficiency_features_only_trans_per_slot": [float(v) for v in suff_feat_tr],
               "sufficiency_energy_trans_per_slot": [float(v) for v in suff_energy_tr],
               "sufficiency_cosine_per_slot": [float(v) for v in suff_cos],
               "sufficiency_cosine_trans_per_slot": [float(v) for v in suff_cos_tr],
               "sufficiency_n_trans_per_slot": suff_n_trans.tolist(),
               "modes": modes}
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\n[chan] per-slot SUFFICIENCY -- fraction of each channel's action margin the")
    print("[chan] features additively recover. Channels differ, so conclusions drawn from them")
    print("[chan] are not equally reliable; a low-sufficiency channel needs discounting.")
    print("[chan] `slope` is a through-origin calibration, so it can only go to zero two ways:")
    print("[chan] the features are inert, or they are LOUD AND ORTHOGONAL to the margin.")
    print("[chan] `energy` = sum(feat^2)/sum(true^2) separates those; the slope alone cannot.")
    for s_i in range(S_SLOTS):
        flag = "" if suff_recon[s_i] >= 0.80 else "   <- below the 0.80 gate bar"
        if suff_feat[s_i] < 0.15:
            flag += ("   <- NO ADDITIVE COMPONENT DETECTED (see energy before reading further)")
        print(f"[chan]   {names[s_i]:8s} recon={suff_recon[s_i]:.4f}  "
              f"features_only={suff_feat[s_i]:.4f}  energy={suff_energy[s_i]:.4f}  "
              f"cos={suff_cos[s_i]:+.4f}  n={suff_n[s_i]}{flag}")
    print("[chan] restricted to decisions where THIS channel's own emitted bin changed:")
    for s_i in range(S_SLOTS):
        print(f"[chan]   {names[s_i]:8s} recon={suff_recon_tr[s_i]:.4f}  "
              f"features_only={suff_feat_tr[s_i]:.4f}  energy={suff_energy_tr[s_i]:.4f}  "
              f"cos={suff_cos_tr[s_i]:+.4f}  n={suff_n_trans[s_i]}")
    # WHAT A NEAR-ZERO features_only DOES AND DOES NOT LICENCE.
    #
    # It is a FAILURE TO DETECT an additive feature component along the margin, not evidence
    # that the channel is decided elsewhere. Three separate reasons, each sufficient on its own:
    #
    #   1. Both the true margin and the bias term are functions of the EMITTED token, so on a
    #      channel emitting few distinct tokens the `recon` slope is partly tautological -- the
    #      constant tracks the margin by tracking which token won, not by explaining why.
    #   2. The slope is pooled over decisions dominated by already-settled ones. Simulation
    #      shows features explaining EVERY transition perfectly still report a pooled slope
    #      near zero on a channel that is unchanged ~95% of the time.
    #   3. A zero slope with a large `energy` is a DIRECTION result, not an amplitude one: the
    #      features write a lot and it does not lie along the margin.
    #
    # The transition block above is the control for (2) and `energy` is the control for (3);
    # both are computed in this script. Neither is available in analyze_channels.py.
    if np.nanmin(suff_feat) < 0.15:
        worst_i = int(np.nanargmin(suff_feat))
        e, e_tr = suff_energy[worst_i], suff_energy_tr[worst_i]
        print(f"[chan]   NOTE {names[worst_i]}: no additive feature component detected along "
              f"the margin.")
        print(f"[chan]   energy={e:.4f} (all decisions), {e_tr:.4f} (transitions only). "
              f"features_only at transitions = {suff_feat_tr[worst_i]:.4f}.")
        if np.isfinite(e) and e > 0.15:
            print("[chan]   The feature term is LARGE and misaligned -- this is a direction "
                  "result. Do NOT")
            print("[chan]   report it as the features being absent or the channel being a "
                  "default.")
        else:
            print("[chan]   The feature term is also small in energy, consistent with "
                  "inertness, but a")
            print("[chan]   pooled slope on a low-entropy channel cannot establish that on its "
                  "own -- read the")
            print("[chan]   transition row, which is the only one restricted to decisions that "
                  "were live.")
    spread = float(np.nanmax(suff_recon) - np.nanmin(suff_recon))
    if spread > 0.15:
        print(f"[chan]   SPREAD {spread:.3f} across channels: cross-channel comparisons must be")
        print("[chan]   weighted or caveated -- the decomposition is not equally valid everywhere.")

    # NB do NOT summarise C_slot_share by summing over features: decision_shares normalises
    # every decision to sum to 1 across features, so sum-over-tasks-mean-over-features is
    # G / F for every slot by construction and says nothing. The share matrix is comparable
    # PER FEATURE across slots (that is its purpose, and what analyze_channels.py uses); the
    # cross-CHANNEL split of causal mass has to come from the absolute matrix.
    tot_abs = C_abs.sum()
    print("\n[chan] causal mass by channel (share of the run's total |phi|, ABSOLUTE):")
    for s_i in range(S_SLOTS):
        frac = C_abs[s_i].sum() / tot_abs if tot_abs > 0 else float("nan")
        print(f"[chan]   {names[s_i]:8s} {frac:.4f}")
    print("[chan]   (absolute mass is NOT comparable across slots on its own -- u_contrast norm")
    print("[chan]   depends on where the emitted bin sits. analyze_channels.py does the")
    print("[chan]   per-feature share comparison that controls for it.)")
    print(f"[chan] channel PR (share) mean {np.nanmean(pr_chan_share):.2f} of {S_SLOTS}")
    for mode in modes:
        tot = flips[mode]["flip"].sum() / max(flips[mode]["n"].sum(), 1)
        act = flips[mode]["flip_active"].sum() / max(flips[mode]["n_active"].sum(), 1)
        print(f"[chan] {mode:10s} flip rate {tot:.4f} over {flips[mode]['n'].sum()} "
              f"feature-decisions;  GIVEN THE FEATURE FIRES {act:.4f} over "
              f"{flips[mode]['n_active'].sum()}")
    print(f"\n[chan] wrote {args.out}  -- feed to analyze_channels.py")


if __name__ == "__main__":
    main()
