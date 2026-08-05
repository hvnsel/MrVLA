"""Tests for label-free structural generality metrics (mrvla/structural_generality.py).

The centrepiece is a synthetic dataset with four feature archetypes that
reconstruct the two failures arXiv:2603.19183 documents in Appendix A.5.1:

    F_general   a grasp-like detector: fires as a short burst at a *variable*
                mid-trajectory phase, in every task group, most episodes.
                General under BOTH the paper's classifier and ours.

    F_lid       (F1381-like) fires reliably -- bursty, variable phase -- but
                only in ONE task group.  Raw episode coverage is low, so the
                paper's classifier calls it memorized; max_group_rate rescues
                it because within its group it fires every time.

    F_clock     (F1939-like) the "home pose": fires in the first two timesteps
                of EVERY episode across all groups.  Broad and reliable, so a
                coverage/reliability view alone would pass it -- but its onset
                phase is fixed (~0), so phase-invariance flags it as a clock.
                The paper's four metrics contain no phase axis and cannot.

    F_memorized fires in a SINGLE episode.  Within-group firing rate ~1/|g|,
                so max_group_rate is low: both views agree it is memorized.

Run directly:
    python tests/test_structural_generality.py
or via pytest.
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mrvla.structural_generality import (  # noqa: E402
    RHO,
    SIGMA_MIN,
    compute_structural,
    episode_group_map,
    fired_per_episode,
    group_reliability,
    logo_group_prediction,
    onset_phase_stats,
)
from mrvla.generality_classifier import (  # noqa: E402
    BETA_LIBERO,
    THRESHOLD,
    compute_metrics as paper_metrics,
    sigmoid,
)

# Feature column indices in the synthetic dictionary.
F_GENERAL, F_LID, F_CLOCK, F_MEMORIZED, F_DEAD = 0, 1, 2, 3, 4
N_FEATURES = 5

N_GROUPS = 5
EPS_PER_GROUP = 8
T = 24               # timesteps per episode


def _burst(z_ep, feat, center_phase, width=2, mag=1.0):
    """Write a short activation burst into z_ep[:, feat] centred at a phase."""
    c = int(round(center_phase * (T - 1)))
    lo, hi = max(0, c - width // 2), min(T, c + width // 2 + 1)
    z_ep[lo:hi, feat] = mag


def _two_bursts(z_ep, feat, rng, mag=1.0):
    """Two well-separated bursts at random phases -> bursty (obar~2), wide phase.

    Mirrors the paper's real general features (e.g. F1129 fires 2-4x per
    episode).  The early burst lands in [0.1, 0.45], the late one in
    [0.55, 0.9], so they never merge and their onset phases span a wide range.
    """
    _burst(z_ep, feat, rng.uniform(0.10, 0.45), mag=mag)
    _burst(z_ep, feat, rng.uniform(0.55, 0.90), mag=mag)


def build_synthetic(seed: int = 0):
    """Return z [N, F], episode [N], timestep [N], task_id [N].

    Deterministic: a fresh RNG is seeded on every call so archetype statistics
    are identical regardless of test order.
    """
    rng = np.random.default_rng(seed)
    z_list, ep_list, ts_list, task_list = [], [], [], []
    ep_counter = 0
    for g in range(N_GROUPS):
        for _ in range(EPS_PER_GROUP):
            z_ep = np.zeros((T, N_FEATURES), dtype=np.float32)

            # F_general: two bursts at variable phases, every episode/group
            # (obar~2 -> the paper's classifier also accepts it: the baseline
            #  "both agree general" cell)
            _two_bursts(z_ep, F_GENERAL, rng)

            # F_lid (F1381-like): a SINGLE burst at a variable phase, only in
            # group 0.  Single onset + low coverage is exactly what the paper's
            # boundary buries; the variable phase keeps it event-driven for us.
            if g == 0:
                _burst(z_ep, F_LID, rng.uniform(0.10, 0.90), width=3, mag=1.0)

            # F_clock: fixed early phase, single onset, every episode/group
            _burst(z_ep, F_CLOCK, 0.02, width=2, mag=1.0)

            # F_memorized: a single burst in exactly one episode of the dataset
            if ep_counter == 3:
                _burst(z_ep, F_MEMORIZED, rng.uniform(0.30, 0.70), mag=1.0)

            # F_dead: never fires.

            n = T
            z_list.append(z_ep)
            ep_list.append(np.full(n, ep_counter, dtype=np.int32))
            ts_list.append(np.arange(n, dtype=np.int32))
            task_list.append(np.full(n, g, dtype=np.int32))
            ep_counter += 1

    return (np.vstack(z_list), np.concatenate(ep_list),
            np.concatenate(ts_list), np.concatenate(task_list))


def _paper_prob(z, episode, timestep):
    m = paper_metrics(z, episode, timestep)
    logit = (BETA_LIBERO["intercept"]
             + BETA_LIBERO["mean_onsets"] * m["mean_onsets"]
             + BETA_LIBERO["coverage"] * m["coverage"]
             + BETA_LIBERO["mean_act_magnitude"] * m["mean_act_mag"]
             + BETA_LIBERO["rel_run_length"] * m["rel_run_length"])
    return sigmoid(logit), m


# ---------------------------------------------------------------------------
# Core archetype separation
# ---------------------------------------------------------------------------
def test_group_map_matches_tasks():
    z, ep, ts, task = build_synthetic()
    ep_ids, ep_groups, group_ids = episode_group_map(ep, task)
    assert len(ep_ids) == N_GROUPS * EPS_PER_GROUP
    assert len(group_ids) == N_GROUPS
    # first EPS_PER_GROUP episodes are group 0, etc.
    assert (ep_groups[:EPS_PER_GROUP] == 0).all()


def test_max_group_rate_rescues_lid():
    """Lid fires in only 1 group -> low raw coverage, but max_group_rate ~ 1."""
    z, ep, ts, task = build_synthetic()
    st = compute_structural(z, ep, ts, task)
    # within its group the lid fires every episode
    assert st["max_group_rate"][F_LID] >= 0.99
    # it is reliable in exactly one group
    assert st["n_reliable_groups"][F_LID] == 1
    # a truly memorized feature is NOT reliable in any group
    assert st["max_group_rate"][F_MEMORIZED] < RHO
    assert st["n_reliable_groups"][F_MEMORIZED] == 0
    # the general feature is reliable in every group
    assert st["n_reliable_groups"][F_GENERAL] == N_GROUPS


def test_paper_classifier_buries_the_lid():
    """The rescue is only meaningful if the paper's boundary misses the lid."""
    z, ep, ts, task = build_synthetic()
    paper_p, m = _paper_prob(z, ep, ts)
    # lid coverage is ~1/5 (one of five groups) -> below the paper boundary
    assert m["coverage"][F_LID] < 0.3
    assert paper_p[F_LID] < THRESHOLD          # paper: memorized
    # but structurally it is reliable in context
    st = compute_structural(z, ep, ts, task)
    assert st["is_general_candidate"][F_LID]   # ours: general
    # => this feature is exactly a paper-memorized / structural-general rescue
    assert (paper_p[F_LID] < THRESHOLD) and st["is_general_candidate"][F_LID]


def test_phase_invariance_flags_the_clock():
    """Clock fires at a fixed phase -> ~0 std; general fires at variable phase."""
    z, ep, ts, task = build_synthetic()
    ph = onset_phase_stats(z, ep, ts, np.unique(ep))
    # clock onset phase is pinned near the start with negligible spread
    assert ph["onset_phase_mean"][F_CLOCK] < 0.1
    assert ph["onset_phase_std"][F_CLOCK] < 0.05
    # the general feature fires across a wide phase range
    assert ph["onset_phase_std"][F_GENERAL] > SIGMA_MIN
    # and the clock's spread is far below the general feature's
    assert ph["onset_phase_std"][F_CLOCK] < ph["onset_phase_std"][F_GENERAL]


def test_clock_needs_both_metrics():
    """Clock passes the reliability test but is caught only by phase."""
    z, ep, ts, task = build_synthetic()
    st = compute_structural(z, ep, ts, task)
    # reliability alone would accept the clock (fires in every group)
    assert st["max_group_rate"][F_CLOCK] >= 0.99
    assert st["n_reliable_groups"][F_CLOCK] == N_GROUPS
    # but it is flagged a clock, not a general candidate
    assert st["is_clock"][F_CLOCK]
    assert not st["is_general_candidate"][F_CLOCK]
    # the genuine general feature clears both
    assert st["is_general_candidate"][F_GENERAL]
    assert not st["is_clock"][F_GENERAL]


def test_full_archetype_table():
    """Every archetype lands in its intended cell of (paper x structural)."""
    z, ep, ts, task = build_synthetic()
    paper_p, _ = _paper_prob(z, ep, ts)
    st = compute_structural(z, ep, ts, task)

    def cell(fi):
        return (bool(paper_p[fi] >= THRESHOLD),
                bool(st["is_general_candidate"][fi]),
                bool(st["is_clock"][fi]))

    # general: paper general, structural general, not a clock
    assert cell(F_GENERAL) == (True, True, False)
    # lid: paper memorized, structural general (the rescue)
    assert cell(F_LID) == (False, True, False)
    # clock: structural clock, not a general candidate
    assert st["is_clock"][F_CLOCK] and not st["is_general_candidate"][F_CLOCK]
    # memorized: structural not general, not a clock
    assert not st["is_general_candidate"][F_MEMORIZED]
    assert not st["is_clock"][F_MEMORIZED]


# ---------------------------------------------------------------------------
# External validity
# ---------------------------------------------------------------------------
def test_logo_prediction_positive_for_general():
    """Leave-one-group-out: score on 4 groups predicts firing in the 5th.

    F_general fires in every group, so a score trained on 4 groups should
    predict its high firing in the held-out group -> positive rank correlation.
    """
    z, ep, ts, task = build_synthetic()
    ep_ids, ep_groups, group_ids = episode_group_map(ep, task)
    fired = fired_per_episode(z, ep, ep_ids)
    logo = logo_group_prediction(fired, ep_groups, group_ids)
    assert np.isfinite(logo["mean_spearman"])
    assert logo["mean_spearman"] > 0.0


def test_logo_mean_variant_and_heldout_var():
    """The mean-score variant runs, and held-out variance is reported > 0.

    With distinct archetypes the held-out firing rate varies across features,
    so a ~0 Spearman (were it to occur) could be told apart from a saturated
    no-variance case.  Here variance must be clearly positive.
    """
    z, ep, ts, task = build_synthetic()
    ep_ids, ep_groups, group_ids = episode_group_map(ep, task)
    fired = fired_per_episode(z, ep, ep_ids)
    logo = logo_group_prediction(fired, ep_groups, group_ids, score="mean")
    assert np.isfinite(logo["mean_spearman"])
    assert logo["mean_heldout_var"] > 0.0


def test_logo_clock_exclusion_mask():
    """Passing keep=~is_clock restricts the correlation to non-clock features."""
    z, ep, ts, task = build_synthetic()
    ep_ids, ep_groups, group_ids = episode_group_map(ep, task)
    fired = fired_per_episode(z, ep, ep_ids)
    st = compute_structural(z, ep, ts, task)
    keep = ~st["is_clock"]
    logo = logo_group_prediction(fired, ep_groups, group_ids, score="mean",
                                 keep=keep)
    base = logo_group_prediction(fired, ep_groups, group_ids, score="mean")
    b = {p["group"]: p["n_active"] for p in base["per_group"]}
    # the clock is active in the base set, so masking must drop at least one
    # feature in every fold and never add any
    for p in logo["per_group"]:
        assert p["n_active"] < b[p["group"]]


def test_dead_feature_is_inert():
    z, ep, ts, task = build_synthetic()
    st = compute_structural(z, ep, ts, task)
    assert st["max_group_rate"][F_DEAD] == 0.0
    assert st["n_reliable_groups"][F_DEAD] == 0
    assert not st["is_general_candidate"][F_DEAD]
    assert not st["is_clock"][F_DEAD]


# ---------------------------------------------------------------------------
# Metric sanity
# ---------------------------------------------------------------------------
def test_mean_group_rate_is_group_size_invariant():
    """mean_group_rate must not change if one group is up-sampled with copies.

    Raw episode coverage would shift toward the enlarged group; the group-
    balanced version should not, because it averages per-group rates.
    """
    z, ep, ts, task = build_synthetic()
    ep_ids, ep_groups, group_ids = episode_group_map(ep, task)
    fired = fired_per_episode(z, ep, ep_ids)
    base = group_reliability(fired, ep_groups, group_ids)["mean_group_rate"]

    # duplicate every episode of group 0 five times at the row level
    dup_mask = task == 0
    z2 = np.vstack([z, np.tile(z[dup_mask], (5, 1))])
    # offset duplicated episode ids so they are distinct episodes
    max_ep = ep.max() + 1
    dup_ep = np.tile(ep[dup_mask], 5) + max_ep * (
        np.repeat(np.arange(1, 6), dup_mask.sum()))
    ep2 = np.concatenate([ep, dup_ep])
    ts2 = np.concatenate([ts, np.tile(ts[dup_mask], 5)])
    task2 = np.concatenate([task, np.tile(task[dup_mask], 5)])

    ep_ids2, ep_groups2, group_ids2 = episode_group_map(ep2, task2)
    fired2 = fired_per_episode(z2, ep2, ep_ids2)
    dup = group_reliability(fired2, ep_groups2, group_ids2)["mean_group_rate"]

    assert np.allclose(base, dup, atol=1e-6)


# ---------------------------------------------------------------------------
def main():
    tests = [(name, fn) for name, fn in sorted(globals().items())
             if name.startswith("test_") and callable(fn)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
        except AssertionError as e:
            failed += 1
            print(f"  FAIL  {name}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
