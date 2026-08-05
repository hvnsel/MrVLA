"""Label-free structural generality metrics for SAE features.

Motivation
----------
The classifier of arXiv:2603.19183 (Sec. 3.3) is fit to 30 hand-labelled
features whose labels are, by protocol, partly *defined by* the same four
metrics the classifier regresses onto (Stage-1 candidates are screened by
burstiness; Stage-3 requires the global metrics to agree; ambiguous cases are
excluded).  Its reported 100% LOO-CV therefore measures label<->metric
consistency, not construct validity, and the paper documents two feature
classes its metrics get wrong (Appendix A.5.1):

  * F1939 (LIBERO PG5): fires in the first ~20 timesteps of *every* episode
    regardless of scene/task -- the robot "home" pose.  General by inspection
    but a single onset per episode makes obar indistinguishable from a
    memorized feature.  In truth it is a *clock*: it fires at a fixed point in
    the trajectory, not in response to an event.

  * F1381 (DROID PG5): fires on lid grasps across every lid type and scene,
    but lid episodes are only 6.7% of the dataset, so episode coverage = 0.226
    falls below the boundary and it is called memorized.  Raw coverage
    conflates *how often the triggering event occurs in the dataset* with
    *how reliably the feature responds when it does occur*.

The paper (A.5.1) asks for exactly two fixes: "a dataset-diversity-aware
normalization of episode coverage" and "an additional metric that captures
cross-scene consistency independently of activation frequency."  This module
provides both, with **no parameters fit to labels** -- only declared constants
(a firing threshold ``tau`` and a reliability threshold ``rho``), whose choice
is meant to be swept and reported, never tuned to an outcome.

Two metrics
-----------
1. Group-Balanced reliability (fixes F1381).  Partition episodes into groups g
   by a nuisance variable you already have as metadata -- here ``task_id``.
   Within-group firing rate

       p_j(g) = |{e in g : max_t f_j(x_t^(e)) > tau}| / |g|

   isolates *reliability given context* from *how common the context is*.
   From the [G, F] matrix of rates we report:

       max_group_rate_j   = max_g p_j(g)             reliability in its best
                                                     context; rescues F1381,
                                                     which raw coverage buries
       n_reliable_groups_j= |{g : p_j(g) >= rho}|    breadth across contexts
       mean_group_rate_j  = mean_g p_j(g)            coverage made
                                                     group-size-invariant
                                                     (the requested "diversity-
                                                     aware normalization")

   A rare-but-general feature has high max_group_rate (it fires whenever its
   event appears); a memorized feature that fires in a single episode has low
   within-group rate even in its own group (~1/|g|).  Raw coverage cannot tell
   these apart once the boundary is crossed; max_group_rate can.

2. Phase-invariance (fixes F1939, and supplies an axis the paper's four
   metrics lack entirely).  For every onset, record its normalized trajectory
   position phi = t_onset / (T-1) in [0, 1].  Across all active episodes:

       onset_phase_mean_j  where in the trajectory it tends to fire
       onset_phase_std_j   spread of firing position  == the generality signal

   A real event detector fires whenever the event happens, which lands at
   different phases in different episodes -> high phase std.  A clock fires at
   the same trajectory position every time -> phase std ~ 0.  F1939 has
   near-zero phase std; a grasp detector has large phase std.  The paper's
   (obar, c, abar, lr) contain no phase information, so they cannot make this
   distinction at all.

Decision (reported, not fit)
----------------------------
A feature is a *structural general candidate* when it is reliable in context
AND event-driven:

    max_group_rate_j >= rho   AND   onset_phase_std_j >= sigma_min

with ``n_reliable_groups_j`` reporting how broadly general it is.  All raw
metrics are saved so any threshold can be re-applied post hoc.

Validity comes from prediction the metric never saw -- see
``logo_group_prediction`` (leave-one-group-out): does generality computed on a
subset of groups predict firing in a *held-out* group?  Raw coverage has no
held-out-group notion; this metric does.

Usage
-----
python mrvla/structural_generality.py \
    --codes-dir ./codes/sae_libero_goal_100 \
    --out-dir   ./structural_libero_goal_100
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np

from mrvla.generality_classifier import (
    BETA_LIBERO,
    TAU_ON,
    THRESHOLD,
    compute_metrics as paper_metrics,
    onset_state,
    sigmoid,
)

# Declared constants (swept + reported, never fit to labels).
TAU_FIRE = 0.1        # peak > TAU_FIRE => the feature "fires" in that episode
RHO = 0.5             # within-group rate >= RHO => reliable in that group
SIGMA_MIN = 0.15      # onset-phase std >= SIGMA_MIN => event-driven, not a clock


# ---------------------------------------------------------------------------
# Episode -> group map
# ---------------------------------------------------------------------------
def episode_group_map(episode: np.ndarray, group_key: np.ndarray):
    """Return (ep_ids [E], ep_groups [E], group_ids [G]).

    ``group_key`` is a per-row label (e.g. ``task_id``) that is constant within
    an episode; we take its first value per episode.
    """
    ep_ids = np.unique(episode)
    ep_groups = np.empty(len(ep_ids), dtype=np.int64)
    for i, ep in enumerate(ep_ids):
        vals = group_key[episode == ep]
        ep_groups[i] = int(vals[0])
    group_ids = np.unique(ep_groups)
    return ep_ids, ep_groups, group_ids


# ---------------------------------------------------------------------------
# 1. Group-balanced reliability
# ---------------------------------------------------------------------------
def fired_per_episode(z: np.ndarray, episode: np.ndarray, ep_ids: np.ndarray,
                      tau: float = TAU_FIRE) -> np.ndarray:
    """[E, F] bool: did each feature's per-episode peak exceed ``tau``?"""
    F = z.shape[1]
    fired = np.zeros((len(ep_ids), F), dtype=bool)
    for i, ep in enumerate(ep_ids):
        peak = z[episode == ep].max(axis=0)
        fired[i] = peak > tau
    return fired


def group_reliability(fired_EF: np.ndarray, ep_groups: np.ndarray,
                      group_ids: np.ndarray, rho: float = RHO) -> dict:
    """Per-group firing rates and the three group-balanced summaries."""
    F = fired_EF.shape[1]
    G = len(group_ids)
    rate_GF = np.zeros((G, F), dtype=np.float64)
    for gi, g in enumerate(group_ids):
        rows = ep_groups == g
        if rows.any():
            rate_GF[gi] = fired_EF[rows].mean(axis=0)
    return {
        "pergroup_rate": rate_GF.astype(np.float32),               # [G, F]
        "max_group_rate": rate_GF.max(axis=0).astype(np.float32),  # [F]
        "mean_group_rate": rate_GF.mean(axis=0).astype(np.float32),  # [F]
        "n_reliable_groups": (rate_GF >= rho).sum(axis=0).astype(np.int32),  # [F]
        "n_groups": int(G),
    }


# ---------------------------------------------------------------------------
# 2. Phase-invariance
# ---------------------------------------------------------------------------
def onset_phase_stats(z: np.ndarray, episode: np.ndarray, timestep: np.ndarray,
                      ep_ids: np.ndarray, tau_on: float = TAU_ON) -> dict:
    """Mean/std of onset trajectory-phase per feature, over all active episodes.

    Phase of an onset at sorted position p in an episode of length T is
    p / (T - 1) (0 for T == 1).  We accumulate sum and sum-of-squares of onset
    phases per feature so std is a single pass.
    """
    F = z.shape[1]
    sum_phi = np.zeros(F, dtype=np.float64)
    sum_phi2 = np.zeros(F, dtype=np.float64)
    n_obs = np.zeros(F, dtype=np.int64)

    for ep in ep_ids:
        mask = episode == ep
        order = np.argsort(timestep[mask], kind="stable")
        z_ep = z[mask][order]                         # [T, F]
        T = z_ep.shape[0]
        s = onset_state(z_ep, tau_on)                 # [T, F] bool
        s_prev = np.vstack([np.zeros((1, F), dtype=bool), s[:-1]])
        onset = s & ~s_prev                           # [T, F] onset positions
        if not onset.any():
            continue
        phase = (np.arange(T, dtype=np.float64) / (T - 1)) if T > 1 else np.zeros(T)
        # per feature: add phase for each onset row
        contrib = onset * phase[:, None]              # [T, F]
        sum_phi += contrib.sum(axis=0)
        sum_phi2 += (onset * (phase[:, None] ** 2)).sum(axis=0)
        n_obs += onset.sum(axis=0)

    denom = np.maximum(n_obs, 1)
    mean = sum_phi / denom
    var = np.maximum(sum_phi2 / denom - mean ** 2, 0.0)
    std = np.sqrt(var)
    mean[n_obs == 0] = np.nan
    std[n_obs == 0] = np.nan
    return {
        "onset_phase_mean": mean.astype(np.float32),
        "onset_phase_std": std.astype(np.float32),
        "n_onset_obs": n_obs.astype(np.int64),
    }


# ---------------------------------------------------------------------------
# External validity: leave-one-group-out prediction
# ---------------------------------------------------------------------------
def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3:
        return float("nan")
    ra = np.argsort(np.argsort(a[m]))
    rb = np.argsort(np.argsort(b[m]))
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    d = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / d) if d > 0 else float("nan")


def logo_group_prediction(fired_EF: np.ndarray, ep_groups: np.ndarray,
                          group_ids: np.ndarray, rho: float = RHO,
                          min_active: int = 5) -> dict:
    """Does generality on G-1 groups predict firing in the held-out group?

    For each held-out group g*, compute ``max_group_rate`` on the remaining
    groups (the score never sees g*), then correlate it across features with
    the within-g* firing rate.  A positive mean rank correlation means the
    structural score predicts behaviour in a context it was not fit on -- the
    external test raw coverage cannot pose.  Restricted to features active in
    the training groups (max rate over them > 0) so the correlation is not
    dominated by dead features.
    """
    G = len(group_ids)
    if G < 2:
        return {"mean_spearman": float("nan"), "per_group": []}
    per = []
    for gi, g in enumerate(group_ids):
        held = ep_groups == g
        train = ~held
        if not held.any() or not train.any():
            continue
        train_ids = np.unique(ep_groups[train])
        train_rel = group_reliability(fired_EF[train], ep_groups[train],
                                      train_ids, rho)
        train_score = train_rel["max_group_rate"]
        held_rate = fired_EF[held].mean(axis=0)
        active = train_score > (1.0 / max(min_active, 1))
        rho_s = _spearman(train_score[active], held_rate[active])
        per.append({"group": int(g), "n_active": int(active.sum()),
                    "spearman": rho_s})
    vals = [p["spearman"] for p in per if np.isfinite(p["spearman"])]
    return {"mean_spearman": float(np.mean(vals)) if vals else float("nan"),
            "per_group": per}


# ---------------------------------------------------------------------------
# Combined
# ---------------------------------------------------------------------------
def compute_structural(z: np.ndarray, episode: np.ndarray, timestep: np.ndarray,
                       group_key: np.ndarray, tau: float = TAU_FIRE,
                       rho: float = RHO, tau_on: float = TAU_ON,
                       sigma_min: float = SIGMA_MIN) -> dict:
    """All structural metrics + the reported (unfit) general-candidate flag."""
    z = np.asarray(z, dtype=np.float32)
    ep_ids, ep_groups, group_ids = episode_group_map(episode, group_key)
    fired = fired_per_episode(z, episode, ep_ids, tau)
    rel = group_reliability(fired, ep_groups, group_ids, rho)
    ph = onset_phase_stats(z, episode, timestep, ep_ids, tau_on)

    reliable = rel["max_group_rate"] >= rho
    event_driven = np.nan_to_num(ph["onset_phase_std"], nan=0.0) >= sigma_min
    is_clock = reliable & ~event_driven          # broad/reliable but fixed-phase
    is_candidate = reliable & event_driven

    out = {**rel, **ph,
           "is_general_candidate": is_candidate,
           "is_clock": is_clock,
           "n_episodes": len(ep_ids),
           "tau": float(tau), "rho": float(rho),
           "tau_on": float(tau_on), "sigma_min": float(sigma_min)}
    return out


# ---------------------------------------------------------------------------
# Main: run over a codes dir and contrast with the paper classifier
# ---------------------------------------------------------------------------
def _paper_prob(z, episode, timestep, tau_on):
    m = paper_metrics(z, episode, timestep, tau_on=tau_on)
    logit = (BETA_LIBERO["intercept"]
             + BETA_LIBERO["mean_onsets"] * m["mean_onsets"]
             + BETA_LIBERO["coverage"] * m["coverage"]
             + BETA_LIBERO["mean_act_magnitude"] * m["mean_act_mag"]
             + BETA_LIBERO["rel_run_length"] * m["rel_run_length"])
    return sigmoid(logit).astype(np.float32), m


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--codes-dir", required=True,
                   help="Dir of layer_NN.npz from extract_codes_and_metrics.py "
                        "(needs z, episode, timestep, task_id)")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--group-key", default="task_id",
                   help="npz field to group episodes by (default task_id)")
    p.add_argument("--tau", type=float, default=TAU_FIRE)
    p.add_argument("--rho", type=float, default=RHO)
    p.add_argument("--tau-on", type=float, default=TAU_ON)
    p.add_argument("--sigma-min", type=float, default=SIGMA_MIN)
    p.add_argument("--topn", type=int, default=15)
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    layer_files = sorted(glob.glob(os.path.join(args.codes_dir, "layer_*.npz")))
    if not layer_files:
        raise FileNotFoundError(f"No layer_*.npz in {args.codes_dir!r}")

    summary = {"tau": args.tau, "rho": args.rho, "tau_on": args.tau_on,
               "sigma_min": args.sigma_min, "group_key": args.group_key,
               "layers": {}}

    for fpath in layer_files:
        name = os.path.basename(fpath).replace(".npz", "")
        print(f"\n[struct] ====== {name} ======", flush=True)
        d = np.load(fpath)
        z = d["z"].astype(np.float32)
        episode, timestep = d["episode"], d["timestep"]
        if args.group_key not in d:
            raise KeyError(f"{args.group_key!r} not in {fpath}; "
                           f"available: {list(d.keys())}")
        group_key = d[args.group_key]

        st = compute_structural(z, episode, timestep, group_key,
                                tau=args.tau, rho=args.rho,
                                tau_on=args.tau_on, sigma_min=args.sigma_min)
        paper_p, _pm = _paper_prob(z, episode, timestep, args.tau_on)

        ep_ids, ep_groups, group_ids = episode_group_map(episode, group_key)
        fired = fired_per_episode(z, episode, ep_ids, args.tau)
        logo = logo_group_prediction(fired, ep_groups, group_ids, args.rho)

        active = st["max_group_rate"] > 0
        paper_general = paper_p >= THRESHOLD
        struct_general = st["is_general_candidate"]

        # RESCUED: paper says memorized, structure says general (F1381-like)
        rescued = np.where(active & ~paper_general & struct_general)[0]
        # CLOCK: reliable/broad but fixed-phase (F1939-like) -- paper cannot see this
        clocks = np.where(active & st["is_clock"])[0]

        rescued = rescued[np.argsort(-st["max_group_rate"][rescued])][:args.topn]
        clocks = clocks[np.argsort(st["onset_phase_std"][clocks])][:args.topn]

        print(f"  episodes={st['n_episodes']}  groups={st['n_groups']}  "
              f"active={int(active.sum())}", flush=True)
        print(f"  paper-general={int((paper_general & active).sum())}  "
              f"struct-candidates={int((struct_general & active).sum())}  "
              f"clocks={int((st['is_clock'] & active).sum())}", flush=True)
        print(f"  leave-one-group-out mean Spearman(train score -> held-out "
              f"firing) = {logo['mean_spearman']:.3f}", flush=True)

        if len(rescued):
            print(f"  -- rescued (paper memorized -> structural general), "
                  f"top {len(rescued)} by max_group_rate:")
            print(f"     {'feat':>6} {'paperP':>7} {'maxgrp':>7} "
                  f"{'nrelgrp':>7} {'phstd':>6}")
            for fi in rescued:
                print(f"     {fi:>6} {paper_p[fi]:>7.3f} "
                      f"{st['max_group_rate'][fi]:>7.3f} "
                      f"{st['n_reliable_groups'][fi]:>7d} "
                      f"{st['onset_phase_std'][fi]:>6.3f}")
        if len(clocks):
            print(f"  -- clocks (broad/reliable but fixed-phase; invisible to "
                  f"paper metrics), top {len(clocks)} by lowest phase std:")
            print(f"     {'feat':>6} {'paperP':>7} {'maxgrp':>7} "
                  f"{'phmean':>6} {'phstd':>6}")
            for fi in clocks:
                print(f"     {fi:>6} {paper_p[fi]:>7.3f} "
                      f"{st['max_group_rate'][fi]:>7.3f} "
                      f"{st['onset_phase_mean'][fi]:>6.3f} "
                      f"{st['onset_phase_std'][fi]:>6.3f}")

        out_path = os.path.join(args.out_dir, f"{name}_structural.npz")
        np.savez_compressed(
            out_path,
            max_group_rate=st["max_group_rate"],
            mean_group_rate=st["mean_group_rate"],
            n_reliable_groups=st["n_reliable_groups"],
            onset_phase_mean=st["onset_phase_mean"],
            onset_phase_std=st["onset_phase_std"],
            n_onset_obs=st["n_onset_obs"],
            is_general_candidate=st["is_general_candidate"].astype(np.uint8),
            is_clock=st["is_clock"].astype(np.uint8),
            paper_prob_general=paper_p,
            pergroup_rate=st["pergroup_rate"],
        )
        print(f"  saved {out_path}", flush=True)

        summary["layers"][name] = {
            "n_episodes": st["n_episodes"], "n_groups": st["n_groups"],
            "n_active": int(active.sum()),
            "n_paper_general": int((paper_general & active).sum()),
            "n_struct_candidates": int((struct_general & active).sum()),
            "n_clocks": int((st["is_clock"] & active).sum()),
            "n_rescued": int(len(rescued)),
            "logo_mean_spearman": logo["mean_spearman"],
        }

    with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[struct] done -> {os.path.join(args.out_dir, 'summary.json')}",
          flush=True)


if __name__ == "__main__":
    main()
