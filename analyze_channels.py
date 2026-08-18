"""B1 Stages 2-4: which action channel does causal generality actually live in?

Consumes `run_channel_attribution.py`'s output plus the Path A attribution npz. Pure
re-analysis, no GPU, seconds.

THE HYPOTHESIS. Path A's general features look, in their exemplar frames, like grasp / release /
return-to-home events -- all of which are fundamentally gripper-state and phase events rather
than fine positioning. If breadth turns out to concentrate on the gripper channel, "general"
resolves into "phase-event detector": a narrower claim than the current one, but far more
mechanistic and far more falsifiable. If it does not, the current framing survives a real test
it has not yet faced, and channel breadth becomes a second reported axis.

FOUR WAYS THIS CAN GO WRONG, EACH GIVEN ITS OWN CONTROL:

  1. Scale. Raw |phi| is not comparable across slots -- the gripper emits extreme bins, whose
     u_contrast has the largest norm, so every feature looks strong there for geometric reasons.
     Every headline statistic is computed in BOTH absolute and share form, and any conclusion
     that appears in one and not the other is reported as a confound rather than a finding.
     KNOW WHAT THIS CATCHES. A perfectly UNIFORM per-slot rescale is rank-preserving inside a
     per-feature profile, so it cancels in the rank correlations and those will always agree --
     the absolute/share split does not protect them, and it is not claimed to. It bites where
     LEVELS are compared (the group profile table, "which channel carries the most mass"), and
     it catches the realistic version of the confound, which is not uniform: ||u_contrast||
     depends on the emitted bin, and which bins get emitted covaries with which features are
     active. Report both forms, and read a divergence as the confound.
  2. Degeneracy. The gripper token is constant for most of an episode, so dominating it is easy.
     Necessity is reported on all decisions AND on gripper-transition decisions only.
  3. The usual confounds. corr(breadth, gripper share) is rank-residualised on causal magnitude
     and base firing rate, using the same estimator as the Path A headline.
  4. Unequal validity. If the decomposition recovers 95% of the gripper margin and 70% of yaw,
     the seven channels are not equally trustworthy. The per-slot sufficiency table is printed
     first and the spread gates interpretation.

Usage
-----
python analyze_channels.py --chan $B/CHANNELS/goal/layer_31_channels.npz \
                           --attr $B/ATTR/goal_k100/layer_31_attribution.npz \
                           --out $B/CHANNELS/goal/analysis.json
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

from mrvla.attribution import rank_partial_both
from mrvla.channels import (
    DEFAULT_CHANNEL_NAMES, channel_participation_ratio, channel_profile,
)
from mrvla.stats import rankdata_average, tie_fraction, wilson_interval


def _ranks(x):
    """Tie-averaged ranks, centred. NOT argsort(argsort(x)): that breaks ties by array index,
    which matters here because per-channel causal mass is exactly zero for any feature that
    never fires at that slot, and base_rate is a count over a fixed denominator."""
    r = rankdata_average(x)
    return r - r.mean()


def spearman(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3:
        return float("nan")
    ra, rb = _ranks(a[m]), _ranks(b[m])
    den = np.sqrt((ra * ra).sum() * (rb * rb).sum())
    return float((ra * rb).sum() / den) if den > 0 else float("nan")


def common_language_effect(a, b) -> float:
    """P(random a > random b) + 0.5 P(tie). 0.5 = no difference. Same statistic
    compare_recurrence_groups uses, so group contrasts read on one scale across the project."""
    a = np.asarray(a, float); b = np.asarray(b, float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if a.size == 0 or b.size == 0:
        return float("nan")
    gt = (a[:, None] > b[None, :]).sum()
    eq = (a[:, None] == b[None, :]).sum()
    return float((gt + 0.5 * eq) / (a.size * b.size))


def adjusted_breadth_from_attr(A) -> np.ndarray:
    from identify_features import adjusted_breadth
    return adjusted_breadth(A["PR"].astype(np.float64), A["magnitude"].astype(np.float64),
                            A["base_rate"].astype(np.float64), A["is_active"].astype(bool))


def channel_concentration(C_slot: np.ndarray, slot: int) -> np.ndarray:
    """[F]: the fraction of a feature's causal mass sitting in one channel."""
    return channel_profile(C_slot)[slot]


def flip_rates(chan, mode: str, transitions: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """(rate [n_cand, S], n [n_cand, S]) from the saved counterfactual counts."""
    suffix = ("flip_trans", "n_trans") if transitions else ("flip", "n")
    fl = chan[f"flip_{mode}_{suffix[0]}"].astype(np.float64)
    n = chan[f"flip_{mode}_{suffix[1]}"].astype(np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(n > 0, fl / n, np.nan), n


def agreement_verdict(abs_val: float, share_val: float, tol: float = 0.15) -> str:
    """Does a conclusion hold in both normalisations, or only the uncontrolled one?

    Divergence is the signature of the per-slot scale confound: the absolute number carries the
    ||u_contrast|| geometry, the share number does not. When they part company, believe share.

    Its blind spot, stated so AGREE is not over-read: a uniform per-slot scale factor is a
    monotone transform of every feature's channel profile, so it leaves rank correlations
    untouched and this will report AGREE however large the factor is. What it detects is
    DIFFERENTIAL distortion -- the case where the inflation depends on which features are
    active -- plus every level comparison, where even a uniform factor moves the numbers.
    """
    if not (np.isfinite(abs_val) and np.isfinite(share_val)):
        return "undetermined"
    if abs(abs_val - share_val) <= tol and np.sign(abs_val) == np.sign(share_val):
        return "AGREE"
    if np.sign(abs_val) != np.sign(share_val):
        return "CONTRADICT -- absolute and share disagree in SIGN; believe share"
    return "diverge -- magnitudes differ; believe share"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--chan", required=True, help="layer_NN_channels.npz")
    ap.add_argument("--attr", required=True, help="layer_NN_attribution.npz")
    ap.add_argument("--summary", default=None, help="summary.json (default: beside --chan)")
    ap.add_argument("--gripper-slot", type=int, default=6)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    chan = np.load(args.chan)
    A = np.load(args.attr)
    sum_path = args.summary or os.path.join(os.path.dirname(args.chan), "summary.json")
    meta = json.load(open(sum_path)) if os.path.exists(sum_path) else {}
    names = meta.get("channel_names") or list(DEFAULT_CHANNEL_NAMES)
    modes = meta.get("modes", ["projection"])
    groups = {k: np.array(v, dtype=np.int64) for k, v in meta.get("groups", {}).items()}

    C_abs = chan["C_slot_abs"].astype(np.float64)
    C_shr = chan["C_slot_share"].astype(np.float64)
    S_SLOTS = C_abs.shape[0]
    gslot = args.gripper_slot
    active = A["is_active"].astype(bool)
    mag = A["magnitude"].astype(np.float64)
    base = A["base_rate"].astype(np.float64)
    adj = adjusted_breadth_from_attr(A)
    task_pr = A["PR"].astype(np.float64)

    out: dict = {"chan": args.chan, "attr": args.attr, "n_slots": S_SLOTS,
                 "channel_names": names, "gripper_slot": gslot}

    # ---------------- 0. validity of the decomposition per channel -------------------
    print(f"\n[B1] {args.chan}")
    if "argmax_agreement" in meta:
        ag = meta["argmax_agreement"]
        print(f"[B1] recomputed argmax == emitted token: {ag:.4f}"
              + ("" if ag >= 0.99 else "   *** BELOW 1.0 -- nothing below is trustworthy ***"))
    if "sufficiency_recon" in chan:
        sr = chan["sufficiency_recon"].astype(np.float64)
        sf = chan["sufficiency_features_only"].astype(np.float64)
        spread = float(np.nanmax(sr) - np.nanmin(sr))
        out["sufficiency"] = {"recon": [float(v) for v in sr],
                              "features_only": [float(v) for v in sf], "spread": spread}
        print("\n=== per-channel sufficiency (is the decomposition equally valid?) ===")
        for s in range(S_SLOTS):
            print(f"  {names[s]:8s} recon={sr[s]:.4f}  features_only={sf[s]:.4f}"
                  + ("" if sr[s] >= 0.80 else "   below the 0.80 bar"))
        print(f"  spread {spread:.3f}"
              + ("  -- channels are comparably valid" if spread <= 0.15 else
                 "  -- NOT comparably valid; weight or caveat every cross-channel claim"))

    # ---------------- 0b. tie exposure of the rank statistics ------------------------
    # Ranks elsewhere in this repo (identify_features._ranks, structural_generality._ranks) use
    # argsort(argsort(x)), which breaks ties by ARRAY INDEX rather than averaging them. That is
    # only harmless when ties are absent, so the exposure is measured here rather than assumed.
    # base_rate is a count over a fixed denominator, and a per-channel causal mass is exactly
    # zero for any feature that never fires at that slot, so ties are expected, not hypothetical.
    ties = {"magnitude": tie_fraction(mag[active]), "base_rate": tie_fraction(base[active]),
            "adjusted_breadth": tie_fraction(adj[active]),
            "gripper_share": tie_fraction(channel_concentration(C_shr, gslot)[active])}
    out["tie_fractions"] = ties
    worst = max(ties.values())
    print("\n=== tie exposure of the rank statistics ===")
    for k2, v in ties.items():
        print(f"  {k2:18s} {v:.4f}")
    if worst > 0.02:
        print(f"  Up to {100*worst:.1f}% of values are tied. This analysis averages tied ranks, "
              "but\n  adjusted_breadth itself is built with index-broken ties "
              "(identify_features._ranks),\n  so the feature ranking that selects the ablation "
              "and steering targets carries an\n  arbitrary ordering inside every tied block. "
              "Worth fixing upstream before publication.")
    else:
        print("  Ties are negligible; index-broken and tie-averaged ranks agree here.")

    # ---------------- 1. the two breadth axes ---------------------------------------
    pr_chan_abs = channel_participation_ratio(C_abs)
    pr_chan_shr = channel_participation_ratio(C_shr)
    r_axes_abs = spearman(adj[active], pr_chan_abs[active])
    r_axes_shr = spearman(adj[active], pr_chan_shr[active])
    out["breadth_axes"] = {
        "channel_pr_share_mean": float(np.nanmean(pr_chan_shr[active])),
        "channel_pr_share_p10": float(np.nanpercentile(pr_chan_shr[active], 10)),
        "channel_pr_share_p90": float(np.nanpercentile(pr_chan_shr[active], 90)),
        "corr_taskbreadth_channelbreadth_abs": r_axes_abs,
        "corr_taskbreadth_channelbreadth_share": r_axes_shr,
        "corr_taskPR_channelPR_share": spearman(task_pr[active], pr_chan_shr[active]),
        "agreement": agreement_verdict(r_axes_abs, r_axes_shr)}
    print("\n=== the two breadth axes ===")
    print(f"  channel PR (share): mean {out['breadth_axes']['channel_pr_share_mean']:.2f} "
          f"of {S_SLOTS}   p10 {out['breadth_axes']['channel_pr_share_p10']:.2f}   "
          f"p90 {out['breadth_axes']['channel_pr_share_p90']:.2f}")
    print(f"  corr(adjusted task breadth, channel breadth) = {r_axes_shr:+.3f} (share), "
          f"{r_axes_abs:+.3f} (abs)   [{out['breadth_axes']['agreement']}]")
    print("  near 0 => the axes are independent and channel breadth is a genuinely new axis;")
    print("  strongly positive => 'general' just means 'touches everything', one axis not two.")

    # ---------------- 2. the gripper test -------------------------------------------
    print("\n=== does causal breadth concentrate in the gripper channel? ===")
    grip = {}
    for label, C in (("share", C_shr), ("abs", C_abs)):
        conc = channel_concentration(C, gslot)
        raw = spearman(adj[active], conc[active])
        partial = rank_partial_both(conc[active], adj[active], mag[active], base[active])
        grip[label] = {"raw": raw, "partial_vs_magnitude_and_baserate": partial,
                       "mean_gripper_share": float(np.nanmean(conc[active]))}
        print(f"  [{label:5s}] corr(adjusted breadth, {names[gslot]} concentration) "
              f"raw={raw:+.3f}   partial|magnitude,base_rate={partial:+.3f}")
    grip["agreement"] = agreement_verdict(
        grip["abs"]["partial_vs_magnitude_and_baserate"],
        grip["share"]["partial_vs_magnitude_and_baserate"])
    out["gripper_test"] = grip
    print(f"  -> {grip['agreement']}")

    # per-channel version of the same partial, so the story is not gripper-or-nothing
    per_ch = []
    for s in range(S_SLOTS):
        conc = channel_concentration(C_shr, s)
        per_ch.append(rank_partial_both(conc[active], adj[active], mag[active], base[active]))
    out["per_channel_partial_share"] = per_ch
    print("\n  partial corr(adjusted breadth, channel concentration | magnitude, base rate):")
    for s in range(S_SLOTS):
        bar = "+" * int(round(20 * max(per_ch[s], 0))) or ("-" * int(round(20 * -min(per_ch[s], 0))))
        print(f"    {names[s]:8s} {per_ch[s]:+.3f}  {bar}")

    # ---------------- 3. group contrast ---------------------------------------------
    if {"general", "specialist"} <= set(groups):
        gen, spec = groups["general"], groups["specialist"]
        prof = channel_profile(C_shr)
        rows = []
        print("\n=== channel profile: general vs specialist groups (share) ===")
        print(f"  {'channel':10s} {'general':>9s} {'specialist':>11s} {'diff':>8s} {'P(g>s)':>8s}")
        for s in range(S_SLOTS):
            g_v, s_v = prof[s][gen], prof[s][spec]
            cles = common_language_effect(g_v, s_v)
            rows.append({"channel": names[s], "general_mean": float(np.nanmean(g_v)),
                         "specialist_mean": float(np.nanmean(s_v)),
                         "diff": float(np.nanmean(g_v) - np.nanmean(s_v)),
                         "common_language_effect": cles})
            print(f"  {names[s]:10s} {np.nanmean(g_v):9.4f} {np.nanmean(s_v):11.4f} "
                  f"{np.nanmean(g_v) - np.nanmean(s_v):+8.4f} {cles:8.3f}")
        out["group_channel_profile"] = rows
        print("  P(g>s) is the chance a random general feature is more concentrated in that")
        print("  channel than a random specialist. 0.5 = no difference.")

    # ---------------- 4. necessity, and the transition control ----------------------
    if "candidate_features" in chan:
        feats = chan["candidate_features"].astype(np.int64)
        pos = {int(j): i for i, j in enumerate(feats)}
        out["necessity"] = {}
        for mode in modes:
            key = f"flip_{mode}_flip"
            if key not in chan:
                continue
            rate_all, n_all = flip_rates(chan, mode, transitions=False)
            rate_tr, n_tr = flip_rates(chan, mode, transitions=True)
            entry = {"per_group": {}}
            print(f"\n=== necessity ({mode}): does removing the feature change the action? ===")
            print(f"  {'group':11s} {'channel':9s} {'all':>18s} {'transitions only':>20s}")
            for gname, idx in groups.items():
                rows_i = [pos[int(j)] for j in idx if int(j) in pos]
                if not rows_i:
                    continue
                per_ch_stats = []
                for s in range(S_SLOTS):
                    fa = float(np.nansum(chan[f"flip_{mode}_flip"][rows_i, s]))
                    na = float(np.nansum(chan[f"flip_{mode}_n"][rows_i, s]))
                    ft = float(np.nansum(chan[f"flip_{mode}_flip_trans"][rows_i, s]))
                    nt = float(np.nansum(chan[f"flip_{mode}_n_trans"][rows_i, s]))
                    ra = fa / na if na else float("nan")
                    rt = ft / nt if nt else float("nan")
                    lo, hi = wilson_interval(int(fa), int(na)) if na else (np.nan, np.nan)
                    tlo, thi = wilson_interval(int(ft), int(nt)) if nt else (np.nan, np.nan)
                    per_ch_stats.append({"channel": names[s], "rate_all": ra, "ci_all": [lo, hi],
                                         "n_all": na, "rate_transitions": rt,
                                         "ci_transitions": [tlo, thi], "n_transitions": nt})
                    if s == gslot or s == 0:
                        print(f"  {gname:11s} {names[s]:9s} {ra:8.4f}[{lo:.4f},{hi:.4f}] "
                              f"{rt:9.4f}[{tlo:.4f},{thi:.4f}]")
                entry["per_group"][gname] = per_ch_stats
            out["necessity"][mode] = entry

        if len(modes) > 1 and all(f"flip_{m}_flip" in chan for m in ("projection", "coded")):
            rp, _ = flip_rates(chan, "projection")
            rc, _ = flip_rates(chan, "coded")
            gap = float(np.nanmean(rp - rc))
            out["projection_minus_coded_flip_gap"] = gap
            print(f"\n=== projection vs coded ablation ===")
            print(f"  mean flip-rate gap = {gap:+.4f}")
            print("  Projection removes the WHOLE component along the decoder direction; coded")
            print("  removes only what the SAE attributed to the feature. A large positive gap")
            print("  means the rollout intervention acts substantially on structure the SAE")
            print("  never credited to that feature -- a caveat on reading rollout results as")
            print("  evidence about coded features.")

    if args.out:
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\n[B1] wrote {args.out}")


if __name__ == "__main__":
    main()
