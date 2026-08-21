#!/usr/bin/env python3
"""Does breadth predict CAUSAL DECISIVENESS on a held-out task, or only attributed mass?

A3/P1 is the paper's central claim and it is correlational:

    breadth over 9 tasks  ->  predicts ATTRIBUTED importance on the 10th
    (+0.452 goal, curvature-corrected, floor of zero)

This script runs the same estimator with one thing changed. Predictor, controls, folds, basis
ladder and floors are identical; the TARGET becomes the readout counterfactual flip rate on the
held-out task -- does deleting the feature's coded contribution change the emitted token. That
is a causal quantity with ~446k decisions of power behind it, against the rollout ablation's
200 episodes (results.md P3: MDE 9.7 points pooled, 18 per task AT THE CEILING; there is no
feasible rollout experiment that fixes this).

It also answers the question P6a raised but did not ask. P6a found general ~= random on the
POOLED per-firing flip rate. Pooled flip rate is average strength. Breadth, after
residualisation on magnitude, is a claim about SCOPE. The scope version -- does breadth predict
decisiveness on tasks it never saw -- is what runs here.

WHAT IS LOAD-BEARING, AND WHY EACH GUARD EXISTS

  1. The task join. `run_attribution.py` skips out-of-range tokens with `continue`, so a task
     with zero valid rows never enters its `task_ids`; this script's `task_ids` contains every
     task present in the shards. If they ever differ, every row index past the gap shifts by one
     and `flip[F, G, S]` silently joins the wrong task to the wrong feature. It is an assertion,
     not a comment.

  2. All 2048 features, not `pick_candidates`' ~396. Those are the two extremes of the very
     predictor under test, and extreme-group sampling inflates |r| by construction. The test
     `test_selecting_features_on_the_predictor_inflates_the_correlation` pins the size of it.

  3. The BINOMIAL DENOMINATOR floor, which neither of the standard controls covers. The target
     is a ratio. Under a null where every feature has identical true decisiveness, a feature
     with a small denominator piles up at exactly zero while one with a large denominator sits
     near p-bar -- so rank(y) tracks the denominator, and the denominator tracks breadth. A
     ratio target can manufacture a positive partial out of denominator structure alone. If the
     observed number does not clear this floor, the result IS the denominator.

  4. The positive control. The attributed target is recomputed through this same code path, the
     same fold masks and the same feature set, and must land on the published A3 number. If it
     does not, the join or the masking is wrong and nothing else in the output is readable.

SCOPE. The flip counterfactual is a direct-effect LOWER BOUND on the readout only: it freezes r
and the rest of the sequence. A positive result says "this feature decides this token", not
"this feature determines behaviour". That caveat travels with the number.

Usage:
    python causal_breadth.py --chan CHANNELS/goal_all/layer_31_channels.npz \
                             --attr ATTR/goal_k100/layer_31_attribution.npz \
                             --n-perm 1000 --out causal_breadth.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

from mrvla.attribution import participation_ratio, total_magnitude
from mrvla.rankbasis import _folds_xy, loto_partial_target, rank_partial_design
from mrvla.stats import rankdata_average
from permutation_null import summarize

def spearman(x, y) -> float:
    """Tie-averaged rank correlation. Defined here rather than imported from
    `mrvla.confound_audit`, which pulls in matplotlib at import time and would make this script
    unrunnable on a login node without a plotting stack."""
    a, b = rankdata_average(np.asarray(x, float)), rankdata_average(np.asarray(y, float))
    a, b = a - a.mean(), b - b.mean()
    den = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / den) if den > 0 else float("nan")


# The published attributional numbers, for the positive control to be checked against by eye.
PUBLISHED_A3 = {"goal": (0.493, 0.452), "spatial": (0.436, 0.404),
                "object": (0.393, 0.362), "10": (0.512, 0.473)}


# ---------------------------------------------------------------- loading and the task join

def load_pair(chan_path: str, attr_path: str, allow_mismatch: bool = False) -> dict:
    """Load the channel counterfactual and the attribution matrix, joined on task id.

    Refuses rather than reindexes by position. Positional joins are how this class of bug
    survives: both arrays have G rows, so nothing raises.
    """
    ch = np.load(chan_path)
    at = np.load(attr_path)
    sm_path = os.path.join(os.path.dirname(chan_path), "summary.json")
    summary = json.load(open(sm_path)) if os.path.exists(sm_path) else {}

    if "flip_coded_n_active_gt" not in ch.files:
        raise SystemExit(
            f"{chan_path} has no per-(feature, task, slot) counters. It predates the task axis;\n"
            f"re-run run_channel_attribution.py (--all-features --coeff coded) to produce them.\n"
            f"keys present: {sorted(k for k in ch.files if k.startswith('flip_'))[:6]} ...")

    ct, atk = np.asarray(ch["task_ids"]).ravel(), np.asarray(at["task_ids"]).ravel()
    if not np.array_equal(ct, atk):
        only_c, only_a = sorted(set(ct.tolist()) - set(atk.tolist())), \
                         sorted(set(atk.tolist()) - set(ct.tolist()))
        msg = (f"TASK MISALIGNMENT -- the join would be wrong and nothing would raise.\n"
               f"  channels : {ct.tolist()}\n  attribution: {atk.tolist()}\n"
               f"  only in channels: {only_c}\n  only in attribution: {only_a}\n"
               f"  (run_attribution.py drops a task with zero valid token rows; "
               f"run_channel_attribution.py keeps it.)")
        if not allow_mismatch:
            raise SystemExit(msg + "\n  Pass --allow-task-mismatch to intersect and continue.")
        keep = np.array(sorted(set(ct.tolist()) & set(atk.tolist())))
        ci = np.array([int(np.where(ct == t)[0][0]) for t in keep])
        ai = np.array([int(np.where(atk == t)[0][0]) for t in keep])
        print("[cb] " + msg.replace("\n", "\n[cb] "))
        print(f"[cb] --allow-task-mismatch: continuing on the {keep.size} shared tasks.")
    else:
        keep = ct
        ci = ai = np.arange(ct.size)

    C = np.asarray(at["C"], dtype=np.float64)[ai]                    # [G, F] attributed mass
    n_act = np.asarray(ch["flip_coded_n_active_gt"], dtype=np.float64)[:, ci]      # [F, G, S]
    fl_act = np.asarray(ch["flip_coded_flip_active_gt"], dtype=np.float64)[:, ci]
    n_act_t = np.asarray(ch["flip_coded_n_active_trans_gt"], dtype=np.float64)[:, ci]
    fl_act_t = np.asarray(ch["flip_coded_flip_active_trans_gt"], dtype=np.float64)[:, ci]
    n_dec = np.asarray(ch["flip_coded_n_gt"], dtype=np.float64)[0, ci]             # [G, S]

    if C.shape[1] != n_act.shape[0]:
        raise SystemExit(f"feature-count disagreement: attribution F={C.shape[1]}, "
                         f"channels F={n_act.shape[0]}")
    return {"C": C, "n_active": n_act, "flip_active": fl_act,
            "n_active_trans": n_act_t, "flip_active_trans": fl_act_t, "n_dec": n_dec,
            "base_rate": np.asarray(at["base_rate"], dtype=np.float64),
            "task_ids": keep, "summary": summary,
            "n_features": int(C.shape[1]), "n_tasks": int(C.shape[0]),
            "n_slots": int(n_act.shape[2])}


# ---------------------------------------------------------------- the target

def flip_rate(flip_gts: np.ndarray, n_gts: np.ndarray, slots=None,
              zero_denominator: str = "drop") -> tuple[np.ndarray, np.ndarray]:
    """Pool [F, G, S] counters over slots into the [G, F] target and its denominator.

    RATIO OF SUMS, not mean of per-slot ratios. A mean of ratios gives a slot where the feature
    fired four times the same weight as one where it fired four thousand, which is not the
    quantity anyone means by "how often does deleting this feature change what the model says".
    """
    sl = slice(None) if slots is None else slots
    num = flip_gts[:, :, sl].sum(axis=2).T                              # [G, F]
    den = n_gts[:, :, sl].sum(axis=2).T
    with np.errstate(divide="ignore", invalid="ignore"):
        R = np.where(den > 0, num / np.maximum(den, 1.0), np.nan)
    if zero_denominator == "zero":
        # "never fired on this task" encoded as decisiveness 0. A defensible reading of the
        # scientific question and a worse statistic: it re-imports the base-rate confound
        # wholesale, since a feature's chance of a zero cell IS its base rate.
        R = np.where(den > 0, R, 0.0)
    return R, den


def counter_base_rate(n_active: np.ndarray, n_dec: np.ndarray, exclude: str = "held") -> np.ndarray:
    """[G, F] base rate rebuilt from the channel counters, under one of three exclusion rules.

    `exclude="none"`  -- all G tasks, the same vector in every fold. Differs from the shipped
                         `base_rate` only in DEFINITION, so A-vs-this prices the definition.
    `exclude="held"`  -- fold gi drops task gi. The leak-free control.
    `exclude="other"` -- fold gi drops task (gi+1) % G. THE PLACEBO: same nine-of-ten
                         construction, same definition, same sample size, but the held-out task
                         is still in there. If the placebo moves the number as much as "held"
                         does, the uplift is about sample composition and not about the leak,
                         and the whole comparison has to be dropped.

    Why the placebo is not optional. `run_attribution.py:314` builds base_rate over all tasks
    including the held-out one, so A3's second control has always carried some of its own target
    -- that is real. But swapping it for this one changes TWO things at once (the exclusion and
    the numerator/denominator convention, see the module note on run_attribution.py:291), and a
    +0.10 move on the paper's headline in the flattering direction is exactly the shape of result
    that has to be decomposed before it is believed.
    """
    per_task = n_active.sum(axis=2)                                     # [F, G]
    dec_task = n_dec.sum(axis=1)                                        # [G]
    G = per_task.shape[1]
    out = np.empty((G, per_task.shape[0]), dtype=np.float64)
    for gi in range(G):
        if exclude == "none":
            k = np.ones(G, dtype=bool)
        elif exclude == "held":
            k = np.arange(G) != gi
        elif exclude == "other":
            k = np.arange(G) != ((gi + 1) % G)
        else:
            raise ValueError(f"unknown exclude {exclude!r}")
        tot = dec_task[k].sum()
        out[gi] = per_task[:, k].sum(axis=1) / tot if tot > 0 else np.nan
    return out


def loto_base_rate(n_active: np.ndarray, n_dec: np.ndarray) -> np.ndarray:
    """[G, F] base rate from fold gi's TRAINING tasks only. Thin alias, kept for readability."""
    return counter_base_rate(n_active, n_dec, "held")


# ---------------------------------------------------------------- the three floors

def paired_column_shuffle(mats: list[np.ndarray], rng: np.random.Generator) -> list[np.ndarray]:
    """One permutation per task row, applied to every matrix TOGETHER.

    The pairing is the whole point. Within a task, C[g, j] and R[g, j] are two functionals of
    the same decisions and are mechanically linked -- large |phi| makes a flip likelier, and
    breadth is a participation ratio of |phi|. Permuting them independently would destroy that
    link as well as the cross-task identity, giving an ANTI-CONSERVATIVE floor that the observed
    number would clear for free. base_rate deliberately stays attached to its original index,
    which biases the floor upward and keeps it honest.
    """
    G = mats[0].shape[0]
    out = [np.empty_like(m) for m in mats]
    for g in range(G):
        pi = rng.permutation(mats[0].shape[1])
        for k, m in enumerate(mats):
            out[k][g] = m[g][pi]
    return out


def binomial_denominator_null(den: np.ndarray, p_bar: float,
                              rng: np.random.Generator) -> np.ndarray:
    """Target redrawn as Binomial(N, p_bar) -- every denominator kept EXACTLY.

    Under this null all features are equally decisive, so any partial that survives is
    manufactured by the denominator structure alone: small-N cells pile up at an exact zero
    (a rank tie at the bottom) while large-N cells concentrate near p_bar. Since N tracks base
    rate and base rate tracks breadth, this is not a hypothetical.
    """
    N = np.nan_to_num(den, nan=0.0).astype(np.int64)
    draw = rng.binomial(np.maximum(N, 0), p_bar).astype(np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(N > 0, draw / np.maximum(N, 1), np.nan)


def run_floors(C, R, den, base_rate, spec, n_perm, seed, keep_mask=None) -> dict:
    """All three floors on the same folds and basis as the observed number."""
    rng = np.random.default_rng(seed)
    p_bar = float(np.nansum(R * den) / max(np.nansum(den), 1.0))
    out = {"paired_column_shuffle": [], "feature_shuffle": [], "binomial_denominator": []}
    for _ in range(n_perm):
        Cp, Rp, Dp = paired_column_shuffle([C, R, den], rng)
        km = None if keep_mask is None else (Dp >= keep_mask)
        v = loto_partial_target(Cp, Rp, base_rate, spec, keep_mask=km)
        out["paired_column_shuffle"].append(v.mean() if v.size else np.nan)

        vals = []
        for PR_tr, mag_tr, held, br, _e, _gi, _m in _folds_xy(
                C, R, base_rate, None if keep_mask is None else (den >= keep_mask)):
            f = rank_partial_design(rng.permutation(held), PR_tr, [mag_tr, br], spec)
            if np.isfinite(f):
                vals.append(f)
        out["feature_shuffle"].append(np.mean(vals) if vals else np.nan)

        Rb = binomial_denominator_null(den, p_bar, rng)
        v = loto_partial_target(C, Rb, base_rate, spec,
                                keep_mask=None if keep_mask is None else (den >= keep_mask))
        out["binomial_denominator"].append(v.mean() if v.size else np.nan)
    return {k: np.array([x for x in v if np.isfinite(x)]) for k, v in out.items()}


# ---------------------------------------------------------------- the estimate

def estimate(C, R, base_rate, specs, keep_mask=None, extra=None) -> dict:
    res = {}
    for spec in specs:
        f = loto_partial_target(C, R, base_rate, spec, keep_mask=keep_mask, extra=extra)
        res[spec] = {"partial": float(f.mean()) if f.size else float("nan"),
                     "n_folds": int(f.size), "n_positive": int((f > 0).sum()),
                     "worst_fold": float(f.min()) if f.size else float("nan"),
                     "folds": [float(v) for v in f]}
    return res


def fold_sizes(C, R, base_rate, keep_mask=None) -> dict:
    """Per-fold n, the tie fraction of the target, and spearman(predictor, denominator).

    The tie fraction matters because small denominators produce exact-zero blocks; the
    predictor/denominator correlation is the size of the threat the binomial floor tests, and
    printing it means it is looked at rather than argued about.
    """
    ns, ties = [], []
    for _PR, _mag, held, _br, _e, _gi, _m in _folds_xy(C, R, base_rate, keep_mask):
        ns.append(int(held.size))
        _, cnt = np.unique(held, return_counts=True)
        ties.append(float((cnt[cnt > 1] - 1).sum() / max(held.size, 1)))
    return {"n_per_fold": ns, "tie_fraction": ties,
            "median_n": float(np.median(ns)) if ns else float("nan"),
            "median_tie_fraction": float(np.median(ties)) if ties else float("nan")}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--chan", required=True, help="layer_NN_channels.npz with the _gt counters")
    ap.add_argument("--attr", required=True, help="layer_NN_attribution.npz (the C matrix)")
    ap.add_argument("--suite", default="", help="goal|spatial|object|10, for the printed control")
    ap.add_argument("--min-active", type=int, default=30,
                    help="drop a (feature, held-out task) cell with fewer firings than this")
    ap.add_argument("--ladder", default="0,10,30,100,300,1000",
                    help="min-active sensitivity ladder")
    ap.add_argument("--zero-denominator", choices=("drop", "zero"), default="drop")
    ap.add_argument("--base-rate", choices=("global", "loto", "both"), default="both")
    ap.add_argument("--gripper-slot", type=int, default=6)
    ap.add_argument("--n-perm", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--allow-task-mismatch", action="store_true")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    D = load_pair(args.chan, args.attr, args.allow_task_mismatch)
    C, br = D["C"], D["base_rate"]
    G, F, S = D["n_tasks"], D["n_features"], D["n_slots"]
    sm = D["summary"]
    specs = ["linear", "tensor4"]

    print(f"[cb] {os.path.basename(os.path.dirname(args.chan))}  "
          f"F={F} G={G} S={S}  tasks {D['task_ids'].tolist()}")
    print(f"[cb] alignment: channel task_ids == attribution task_ids   OK")
    print(f"[cb] provenance: feature_set={'all' if sm.get('all_features') else 'candidates'}  "
          f"coeff={sm.get('coeff', '?')}  trans_mask={sm.get('trans_mask', 'UNKNOWN')}  "
          f"n_candidates={sm.get('n_candidates', '?')}")
    if sm.get("all_features") is False:
        print("[cb] *** WARNING: this run covers only the candidate features, which are the two")
        print("[cb]     EXTREMES of the predictor under test. Extreme-group sampling inflates")
        print("[cb]     |r| by construction. Re-run with --all-features before reporting. ***")

    R, den = flip_rate(D["flip_active"], D["n_active"], zero_denominator=args.zero_denominator)
    p_bar = float(np.nansum(R * den) / max(np.nansum(den), 1.0))
    fin = np.isfinite(R)
    print(f"[cb] target = flip_active/n_active, coded, pooled over {S} slots (ratio of sums)")
    print(f"[cb]   pooled rate {p_bar:.4f};  denominator per (feature,task): "
          f"median {np.median(den[den > 0]):.0f}  p10 {np.percentile(den[den > 0], 10):.0f}  "
          f"zero cells {int((den == 0).sum())}/{den.size}")

    keep = (den >= args.min_active) if args.min_active > 0 else None
    out = {"chan": args.chan, "attr": args.attr, "n_features": F, "n_tasks": G, "n_slots": S,
           "task_ids": D["task_ids"].tolist(), "provenance": sm, "pooled_flip_rate": p_bar,
           "min_active": args.min_active, "zero_denominator": args.zero_denominator}

    # --- positive control: the attributed target through this exact code path -------------
    ctrl = estimate(C, C, br, specs)
    pub = PUBLISHED_A3.get(args.suite)
    print("\n=== positive control: attributed target, same folds, same feature set ===")
    for spec in specs:
        c = ctrl[spec]
        ref = "" if not pub else f"   published {pub[0] if spec == 'linear' else pub[1]:+.3f}"
        print(f"[cb]   {spec:8s} {c['partial']:+.4f}  {c['n_positive']}/{c['n_folds']} folds"
              f"{ref}")
    if pub and abs(ctrl["tensor4"]["partial"] - pub[1]) > 0.02:
        print("[cb]   *** the control does NOT reproduce the published number. The join or the")
        print("[cb]       feature set differs from A3's; nothing below is readable. ***")
    out["positive_control"] = ctrl

    # --- A3's control plane, decomposed ------------------------------------------------------
    #
    # Swapping the shipped base_rate for one rebuilt from the channel counters changes TWO things
    # at once, and the first pass attributed the whole move to the second of them:
    #
    #   A -> B   DEFINITION. run_attribution.py:291-314 accumulates the numerator over ALL n*7
    #            rows but the denominator only over rows whose token was in range, so the shipped
    #            base_rate is inflated by ~1/0.992. That is close to a uniform scale factor, which
    #            a rank control ignores -- but only close: where invalid tokens are not spread
    #            evenly across features the inflation is feature-specific and the ranking moves.
    #   B -> C   THE LEAK. base_rate over all G tasks includes the held-out one, so A3's second
    #            control has always carried a little of its own target.
    #
    # And a PLACEBO, because B -> C is not self-evidently the explanation. The two vectors differ
    # by a tenth of a sample estimated from ~45k rows per task; they should be ~0.999 rank
    # correlated, and a control that identical should not move a partial by 0.10. The placebo
    # drops a task that is NOT the held-out one: same definition, same nine-of-ten construction,
    # same sample size, leak intact. If it moves the number too, the effect is sample composition
    # and the comparison is worthless.
    #
    # This is the ATTRIBUTED target -- the paper's headline, not the causal one -- so whatever
    # this table says, A3 inherits.
    _cbr = lambda mode: counter_base_rate(D["n_active"], D["n_dec"], mode)
    br_arms = {"B all tasks (definition only)": _cbr("none"),
               "C training tasks (leak-free)": _cbr("held"),
               "P placebo, drops a NON-held task": _cbr("other")}
    print("[cb]   A3's control plane, decomposed (attributed target throughout):")
    print(f"[cb]     {'A shipped base_rate':34s} "
          + "  ".join(f"{sp}={ctrl[sp]['partial']:+.4f}" for sp in specs))
    out["base_rate_decomposition"] = {"A shipped": {sp: ctrl[sp]["partial"] for sp in specs}}
    for label, brv in br_arms.items():
        e = estimate(C, C, brv, specs)
        out["base_rate_decomposition"][label] = {sp: e[sp]["partial"] for sp in specs}
        print(f"[cb]     {label:34s} "
              + "  ".join(f"{sp}={e[sp]['partial']:+.4f}" for sp in specs)
              + "  (delta " + ", ".join(f"{e[sp]['partial'] - ctrl[sp]['partial']:+.4f}"
                                        for sp in specs) + ")")
    br_b, br_c = br_arms["B all tasks (definition only)"], br_arms["C training tasks (leak-free)"]
    rho_ab = spearman(br, br_b[0])
    rho_bc = float(np.mean([spearman(br_b[gi], br_c[gi]) for gi in range(G)]))
    print(f"[cb]     rho(A shipped, B rebuilt) = {rho_ab:+.5f}   "
          f"rho(B, C) per fold = {rho_bc:+.5f}")
    print("[cb]     (read B-A as the definition effect, C-B as the leak, and P-A as what the")
    print("[cb]      same construction gives with the leak left IN. If P tracks C, neither.)")
    out["base_rate_decomposition"]["rho_shipped_vs_rebuilt"] = float(rho_ab)
    out["base_rate_decomposition"]["rho_all_vs_loto_per_fold"] = rho_bc

    # --- the primary estimate --------------------------------------------------------------
    prim = estimate(C, R, br, specs, keep_mask=keep)
    sizes = fold_sizes(C, R, br, keep)
    sp_pred_den = []
    for PR_tr, _mag, _held, _br, _e, gi, m in _folds_xy(C, R, br, keep):
        sp_pred_den.append(spearman(PR_tr, den[gi][m]))
    print("\n=== decisiveness target (breadth on 9 tasks -> flip rate on the 10th) ===")
    print(f"[cb]   spec      partial   folds+   worst    n/fold  ties   rho(PR_tr, N_held)")
    for spec in specs:
        v = prim[spec]
        tag = "   <- REPORTED" if spec == "tensor4" else ""
        print(f"[cb]   {spec:8s}  {v['partial']:+.4f}   {v['n_positive']:2d}/{v['n_folds']:<2d}  "
              f"{v['worst_fold']:+.4f}  {sizes['median_n']:7.0f}  "
              f"{sizes['median_tie_fraction']:.3f}  {np.mean(sp_pred_den):+.3f}{tag}")
    out["primary"] = prim
    out["fold_sizes"] = sizes
    out["spearman_predictor_denominator"] = [float(v) for v in sp_pred_den]

    # --- floors ----------------------------------------------------------------------------
    obs = prim["tensor4"]["partial"]
    floors = run_floors(C, R, den, br, "tensor4", args.n_perm, args.seed,
                        keep_mask=args.min_active if args.min_active > 0 else None)
    print(f"\n=== floors ({args.n_perm} permutations, tensor4) ===")
    out["floors"] = {}
    labels = {"paired_column_shuffle": "paired column shuffle (mechanical)",
              "feature_shuffle": "feature shuffle (estimator)",
              "binomial_denominator": "binomial denominator (ratio artefact)"}
    for k, nullv in floors.items():
        s = summarize(k, obs, nullv)
        out["floors"][k] = s
        star = "   <- the one this design needs" if k == "binomial_denominator" else ""
        print(f"[cb]   {labels[k]:38s} mean {s.get('null_mean', float('nan')):+.4f}  "
              f"sd {s.get('null_sd', float('nan')):.4f}  p95 {s.get('null_p95', float('nan')):+.4f}"
              f"  p {s.get('p_one_sided', float('nan')):.4f}  z {s.get('z', float('nan')):+.1f}"
              f"{star}")

    # --- robustness ------------------------------------------------------------------------
    print("\n=== robustness ===")
    lad = []
    for t in [int(x) for x in args.ladder.split(",") if x.strip()]:
        km = (den >= t) if t > 0 else None
        v = loto_partial_target(C, R, br, "tensor4", keep_mask=km)
        n_kept = int((den >= t).sum()) if t > 0 else int(np.isfinite(R).sum())
        # base rate of the features the threshold keeps vs drops, per the concern that a
        # denominator cut preferentially removes narrow features -- the axis under test
        kept_cells = (den >= max(t, 1))
        br_tile = np.tile(br, (G, 1))
        br_in = float(np.mean(br_tile[kept_cells & fin])) if (kept_cells & fin).any() \
            else float("nan")
        br_out = float(np.mean(br_tile[(~kept_cells) & fin])) if (~kept_cells & fin).any() \
            else float("nan")
        lad.append({"min_active": t, "partial": float(v.mean()) if v.size else float("nan"),
                    "n_folds": int(v.size), "cells_kept": n_kept,
                    "base_rate_kept": br_in, "base_rate_dropped": br_out})
        note = "   <- too few cells to form folds" if v.size == 0 else ""
        print(f"[cb]   min-active {t:5d}  partial {lad[-1]['partial']:+.4f}  "
              f"folds {v.size:2d}  cells {n_kept:7d}  "
              f"base rate kept {br_in:.4f} vs dropped {br_out:.4f}{note}")
    out["min_active_ladder"] = lad
    print("[cb]   (a flat ladder means the threshold is not doing the work; a partial that only")
    print("[cb]    appears at a high threshold is a statement about well-measured features.)")

    # denominator as an explicit control. hinge5, NOT tensor4: control_design's tensor branch
    # crosses per[0] against per[1:] only, so at THREE controls it omits the c1 x c2 surface
    # and is not a tensor product. hinge5's _cross covers all pairs. Do not silently pass three
    # controls to tensor4.
    n_held = np.where(den > 0, den, np.nan)
    v3 = loto_partial_target(C, R, br, "hinge5", keep_mask=keep, extra=[n_held])
    v3b = loto_partial_target(C, R, br, "hinge5", keep_mask=keep)
    m3 = float(v3.mean()) if v3.size else float("nan")
    m3b = float(v3b.mean()) if v3b.size else float("nan")
    print(f"[cb]   + n_active_held control (hinge5)  {m3:+.4f}   vs {m3b:+.4f} without it")
    print("[cb]   (LOWER BOUND, not the headline: breadth partly IS firing across many tasks, so")
    print("[cb]    conditioning on opportunities in the held-out task removes part of the")
    print("[cb]    mechanism by which breadth would transfer. Same discipline P1b applied in")
    print("[cb]    refusing to publish the minimum over the basis ladder.)")
    out["denominator_controlled"] = {"with": m3, "without": m3b}

    if args.base_rate in ("loto", "both"):
        br_l = loto_base_rate(D["n_active"], D["n_dec"])
        vl = loto_partial_target(C, R, br_l, "tensor4", keep_mask=keep)
        print(f"[cb]   base_rate from training tasks only  "
              f"{float(vl.mean()) if vl.size else float('nan'):+.4f}   "
              f"(the shipped base_rate is global, so A3's control leaks the held-out task)")
        out["base_rate_loto"] = float(vl.mean()) if vl.size else float("nan")

    Rz, _ = flip_rate(D["flip_active"], D["n_active"], zero_denominator="zero")
    vz = loto_partial_target(C, Rz, br, "tensor4")
    print(f"[cb]   zero-denominator = zero (no threshold)  "
          f"{float(vz.mean()) if vz.size else float('nan'):+.4f}   "
          f"(re-imports the base-rate confound; reported, not headlined)")
    out["zero_denominator_variant"] = float(vz.mean()) if vz.size else float("nan")

    # per slot, and without the gripper -- P5b shows the gripper's features carry -0.046 of that
    # channel's margin, so pooling mixes one channel where no additive component is detectable
    # with six where it is. If the pooled result is carried by slot 6 that is a finding about
    # slot 6, not about breadth.
    per_slot = []
    names = sm.get("channel_names", [f"slot{i}" for i in range(S)])
    for s_i in range(S):
        Rs, ds = flip_rate(D["flip_active"], D["n_active"], slots=slice(s_i, s_i + 1))
        vs = loto_partial_target(C, Rs, br, "tensor4",
                                 keep_mask=(ds >= args.min_active) if args.min_active else None)
        per_slot.append({"slot": s_i, "name": names[s_i] if s_i < len(names) else str(s_i),
                         "partial": float(vs.mean()) if vs.size else float("nan")})
    print("[cb]   per slot: " + "  ".join(f"{p['name']}={p['partial']:+.3f}" for p in per_slot))
    keep_sl = np.array([i for i in range(S) if i != args.gripper_slot])
    Rg, dg = flip_rate(D["flip_active"][:, :, keep_sl], D["n_active"][:, :, keep_sl])
    vg = loto_partial_target(C, Rg, br, "tensor4",
                             keep_mask=(dg >= args.min_active) if args.min_active else None)
    print(f"[cb]   pooled without the gripper slot  "
          f"{float(vg.mean()) if vg.size else float('nan'):+.4f}")
    out["per_slot"] = per_slot
    out["drop_gripper"] = float(vg.mean()) if vg.size else float("nan")

    if sm.get("trans_mask") == "per_channel":
        Rt, dt = flip_rate(D["flip_active_trans"], D["n_active_trans"])
        vt = loto_partial_target(C, Rt, br, "tensor4",
                                 keep_mask=(dt >= args.min_active) if args.min_active else None)
        print(f"[cb]   transitions only, per-channel mask  "
              f"{float(vt.mean()) if vt.size else float('nan'):+.4f}  "
              f"(median denominator {np.median(dt[dt > 0]):.0f})")
        out["transitions_only"] = float(vt.mean()) if vt.size else float("nan")
    else:
        print(f"[cb]   transitions: SKIPPED -- trans_mask={sm.get('trans_mask', 'UNKNOWN')!r}, "
              f"not 'per_channel'.")
        print("[cb]     Before the per-channel fix the mask was the GRIPPER's, broadcast to all")
        print("[cb]     seven slots, so dx's `_trans` meant 'dx flipped when the gripper moved'.")
        print("[cb]     Those counters are not comparable across channels and are not used.")

    # --- secondary: breadth of the flip matrix itself ---------------------------------------
    ok = (den >= max(args.min_active, 1))
    adequate = ok.sum(axis=0) >= 5
    Rf = np.where(ok, np.nan_to_num(R, nan=0.0), 0.0)
    PR_flip = participation_ratio(Rf)
    PR_C = participation_ratio(C)
    both = adequate & np.isfinite(PR_flip) & np.isfinite(PR_C)
    print("\n=== secondary: is decisive breadth the same axis as attributed breadth? ===")
    q = np.nanpercentile(PR_flip[both], [10, 90]) if both.any() else [np.nan, np.nan]
    qc = np.nanpercentile(PR_C[both], [10, 90]) if both.any() else [np.nan, np.nan]
    print(f"[cb]   PR_flip mean {np.nanmean(PR_flip[both]):.2f} of {G} "
          f"[p10 {q[0]:.2f}, p90 {q[1]:.2f}]   PR_C mean {np.nanmean(PR_C[both]):.2f} "
          f"[p10 {qc[0]:.2f}, p90 {qc[1]:.2f}]   over {int(both.sum())} adequate features")
    # A DEGENERATE PREDICTOR IS NOT A NULL. If every feature fires on every task at a similar
    # rate, PR of the flip matrix is ~G for all of them, the self-contained LOTO has no
    # predictor variation left, and it returns ~0 for a reason that has nothing to do with the
    # hypothesis. This repo has produced two meaningless passes that way (split_half_sweep's
    # calibration matrix, and its breadth-as-activity-rate fixture); the spread is printed and
    # checked rather than assumed.
    flat = float(q[1] - q[0]) < 0.05 * G
    if flat:
        print(f"[cb]   *** PR_flip is nearly constant (p90-p10 = {q[1] - q[0]:.2f} of {G}). The")
        print("[cb]       self-contained and reverse numbers below have almost no predictor")
        print("[cb]       variation to work with and must NOT be read as nulls. ***")
    rho = spearman(PR_C[both], PR_flip[both])
    print(f"[cb]   spearman(PR_C, PR_flip) = {rho:+.4f}")
    print("[cb]   (near +1: one axis in two currencies. near 0: attributed breadth and decisive")
    print("[cb]    breadth are different properties -- which would EXPLAIN P6a, and is itself a")
    print("[cb]    finding rather than a failure.)")
    self_c = estimate(Rf, R, br, ["tensor4"], keep_mask=keep)
    print(f"[cb]   self-contained LOTO (flip breadth -> flip rate, no attribution anywhere): "
          f"{self_c['tensor4']['partial']:+.4f}")
    rev = estimate(Rf, C, br, ["tensor4"], keep_mask=None)
    print(f"[cb]   reverse direction (flip breadth -> attributed mass): "
          f"{rev['tensor4']['partial']:+.4f}   (an asymmetry is informative)")
    out["secondary"] = {"PR_flip_mean": float(np.nanmean(PR_flip[both])),
                        "PR_C_mean": float(np.nanmean(PR_C[both])),
                        "n_adequate": int(both.sum()), "PR_flip_is_flat": bool(flat),
                        "spearman_PR_C_PR_flip": float(rho),
                        "self_contained": self_c["tensor4"],
                        "reverse": rev["tensor4"]}

    # --- verdict ---------------------------------------------------------------------------
    z_col = out["floors"]["paired_column_shuffle"].get("z", float("nan"))
    z_bin = out["floors"]["binomial_denominator"].get("z", float("nan"))
    z_fea = out["floors"]["feature_shuffle"].get("z", float("nan"))
    retain = (out["denominator_controlled"]["with"] / obs) if obs > 0 else float("nan")
    print("\n=== verdict ===")
    if abs(out["floors"]["feature_shuffle"].get("null_mean", 0.0)) > 0.05:
        verdict = "INVALID -- the estimator floor is not zero; fix that before reading anything"
    elif obs >= 0.15 and prim["tensor4"]["n_positive"] >= max(G - 1, 1) \
            and z_col > 5 and z_bin > 5 and retain >= 0.6:
        verdict = ("CAUSAL -- breadth transfers to per-decision decisiveness on held-out tasks, "
                   "clearing both the mechanical and the ratio-artefact floors")
    elif obs >= 0.15 and (z_bin <= 5 or retain < 0.6):
        verdict = ("OPPORTUNITY, NOT DECISIVENESS -- the partial does not survive the "
                   "denominator; this is P6a again in a new place")
    elif abs(obs) < 0.05:
        verdict = ("BOUNDED NULL -- attributed mass transfers, per-decision decisiveness does "
                   "not. Breadth says where a feature WRITES, not where it DECIDES")
    else:
        verdict = "WEAK/AMBIGUOUS -- between the pre-registered bands; report the interval"
    resolution = float(out["floors"]["paired_column_shuffle"].get("null_sd", float("nan"))) * 2
    print(f"[cb]   observed {obs:+.4f}   floors z: mechanical {z_col:+.1f}, "
          f"binomial {z_bin:+.1f}, estimator {z_fea:+.1f}")
    print(f"[cb]   retained under the denominator control: {retain:.2f} of the observed")
    print(f"[cb]   {verdict}")
    print(f"[cb]   resolution: this design separates ~{resolution:.3f} from its floor, against "
          f"the rollout ablation's 0.097 pooled / 0.18 per task (results.md P3).")
    out["verdict"] = verdict
    out["resolution"] = resolution

    if args.out:
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\n[cb] wrote {args.out}")


if __name__ == "__main__":
    main()
