"""Ground-truth tests for the paper-faithful generality classifier.

Every expected value below is derived by hand from the equations in
arXiv:2603.19183 Sections 3.2-3.3, not from the implementation.

Run directly:
    python tests/test_generality_classifier.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mrvla.generality_classifier import (  # noqa: E402
    BETA_DROID,
    BETA_LIBERO,
    TAU_ON,
    classify_features,
    compute_metrics,
    onset_state,
    sigmoid,
)


# ---------------------------------------------------------------------------
# Eq. (5): the onset state machine is STICKY (off only at exact zero)
# ---------------------------------------------------------------------------
def test_state_holds_through_nonzero_dip():
    # rises above tau_on, dips to a small NONZERO value, rises again.
    # Eq. (5) holds the state through the dip -> ONE onset, not two.
    z = np.array([[0.0], [0.5], [0.02], [0.5], [0.0]], dtype=np.float32)
    s = onset_state(z, TAU_ON)[:, 0]
    assert s.tolist() == [False, True, True, True, False]
    onsets = int((s & ~np.r_[False, s[:-1]]).sum())
    assert onsets == 1


def test_state_releases_only_at_exact_zero():
    z = np.array([[0.5], [0.0], [0.5]], dtype=np.float32)
    s = onset_state(z, TAU_ON)[:, 0]
    assert s.tolist() == [True, False, True]
    onsets = int((s & ~np.r_[False, s[:-1]]).sum())
    assert onsets == 2


def test_state_never_on_below_tau_on():
    # nonzero everywhere but never exceeding tau_on -> never ON,
    # so this episode counts toward COVERAGE but contributes zero onsets.
    z = np.full((8, 1), 0.05, dtype=np.float32)
    assert not onset_state(z, TAU_ON).any()


def test_state_initial_off():
    # sub-threshold nonzero at t=0 must hold the initial OFF state (s_0 = 0)
    z = np.array([[0.05], [0.05], [0.3]], dtype=np.float32)
    s = onset_state(z, TAU_ON)[:, 0]
    assert s.tolist() == [False, False, True]


def test_state_rejects_negative():
    try:
        onset_state(np.array([[-1.0]]), TAU_ON)
    except ValueError:
        pass
    else:
        raise AssertionError("negative activations should raise")


# ---------------------------------------------------------------------------
# Eqs. (4), (7), (8), (10): metrics on a hand-computed dataset
# ---------------------------------------------------------------------------
def make_dataset():
    """Two episodes (T=6, T=4), four features, all metrics known by hand.

    feature 0: ep0 two bursts separated by an exact zero; absent in ep1
    feature 1: dead everywhere
    feature 2: fires in both episodes, sustained
    feature 3: nonzero in ep0 but always BELOW tau_on (coverage counts it,
               onsets do not); absent in ep1
    """
    T0, T1, F = 6, 4, 4
    z0 = np.zeros((T0, F), dtype=np.float32)
    z1 = np.zeros((T1, F), dtype=np.float32)

    # f0: [0, .4, 0, .6, .6, 0] -> two onsets, on-steps 1 and 2, peak .6
    z0[:, 0] = [0.0, 0.4, 0.0, 0.6, 0.6, 0.0]
    # f2: ep0 [.3]*2 then 0; ep1 all .5
    z0[0:2, 2] = 0.3
    z1[:, 2] = 0.5
    # f3: sub-threshold nonzero in ep0
    z0[:, 3] = 0.04

    z = np.vstack([z0, z1])
    episode = np.array([0] * T0 + [1] * T1)
    timestep = np.array(list(range(T0)) + list(range(T1)))
    return z, episode, timestep


def test_metrics_ground_truth():
    z, episode, timestep = make_dataset()
    m = compute_metrics(z, episode, timestep, TAU_ON)

    # --- feature 0: only in ep0 ---
    assert np.isclose(m["coverage"][0], 0.5)              # Eq. (4): 1 of 2 eps
    assert np.isclose(m["mean_onsets"][0], 2.0)           # Eq. (7)
    assert np.isclose(m["mean_act_mag"][0], 0.6)          # Eq. (8): peak of ep0
    # Eq. (9): on-steps = 3 (t=1,3,4), onsets = 2 -> run = 1.5
    assert np.isclose(m["mean_run_length"][0], 1.5)
    assert np.isclose(m["rel_run_length"][0], 1.5 / 6.0)  # Eq. (10): /T=6

    # --- feature 1: dead ---
    assert m["coverage"][1] == 0.0
    assert m["mean_onsets"][1] == 0.0
    assert m["mean_act_mag"][1] == 0.0
    assert not m["is_active"][1]

    # --- feature 2: both episodes ---
    assert np.isclose(m["coverage"][2], 1.0)
    assert np.isclose(m["mean_onsets"][2], 1.0)
    # Eq. (8) averages per-episode PEAKS: (0.3 + 0.5)/2 = 0.4,
    # NOT the mean over active timesteps ((.3*2 + .5*4)/6 = 0.4333)
    assert np.isclose(m["mean_act_mag"][2], 0.4)
    assert not np.isclose(m["mean_act_mag"][2], (0.3 * 2 + 0.5 * 4) / 6)
    # ep0 run = 2 over T=6; ep1 run = 4 over T=4
    assert np.isclose(m["mean_run_length"][2], (2 + 4) / 2)
    assert np.isclose(m["rel_run_length"][2], (2 / 6 + 4 / 4) / 2)

    # --- feature 3: nonzero but sub-threshold ---
    # E+ is defined by f > 0, so coverage counts this episode ...
    assert np.isclose(m["coverage"][3], 0.5)
    assert m["is_active"][3]
    # ... but the state machine never turns on, so obar = 0 (< 1).
    # The paper asserts obar >= 1 whenever c > 0; we measure instead.
    assert np.isclose(m["mean_onsets"][3], 0.0)
    assert m["n_active_no_onset"][3] == 1
    assert np.isclose(m["mean_act_mag"][3], 0.04)

    assert m["n_active"] == 3          # features 0, 2, 3
    assert m["n_episodes"] == 2
    assert np.isclose(m["ep_mean_len"], 5.0)


def test_coverage_uses_nonzero_not_threshold():
    """Coverage (Eq. 4, f > 0) must be decoupled from tau_on."""
    z = np.full((5, 1), 0.02, dtype=np.float32)     # all sub-threshold
    episode = np.zeros(5, dtype=int)
    timestep = np.arange(5)
    m = compute_metrics(z, episode, timestep, TAU_ON)
    assert np.isclose(m["coverage"][0], 1.0)        # counted
    assert np.isclose(m["mean_onsets"][0], 0.0)     # but no onsets


def test_metrics_invariant_to_row_order():
    z, episode, timestep = make_dataset()
    m1 = compute_metrics(z, episode, timestep, TAU_ON)
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(episode))
    m2 = compute_metrics(z[perm], episode[perm], timestep[perm], TAU_ON)
    for k in ("coverage", "mean_onsets", "mean_act_mag",
              "rel_run_length", "mean_run_length"):
        assert np.allclose(m1[k], m2[k]), k


# ---------------------------------------------------------------------------
# Eq. (11): the classifier
# ---------------------------------------------------------------------------
def test_classifier_matches_equation_11():
    cov = np.array([0.99], dtype=np.float32)
    ons = np.array([2.0], dtype=np.float32)
    mag = np.array([1.90], dtype=np.float32)
    rrl = np.array([0.10], dtype=np.float32)
    res = classify_features(cov, ons, mag, rrl, beta=BETA_LIBERO, verbose=False)
    expected = (BETA_LIBERO["intercept"]
                + BETA_LIBERO["mean_onsets"] * 2.0
                + BETA_LIBERO["coverage"] * 0.99
                + BETA_LIBERO["mean_act_magnitude"] * 1.90
                + BETA_LIBERO["rel_run_length"] * 0.10)
    assert np.isclose(res["prob_general"][0], sigmoid(np.array(expected)))


def test_paper_reported_probabilities_are_reachable():
    """Consistency check on Eq. (8).

    The paper reports general LIBERO features with episode coverage > 0.99 and
    P(general) of 0.91, 0.89, 0.92 (F1129, F1902, F128).  Solving Eq. (11) for
    abar at c = 0.99 and a bursty obar = 2 requires abar ~ 1.9.  Such values
    are attainable only if abar is a per-episode PEAK (Eq. 8); a mean over
    active timesteps would be far smaller and could not reach these
    probabilities.  This test pins that reading of Eq. (8).
    """
    cov = np.array([0.99], dtype=np.float32)
    ons = np.array([2.0], dtype=np.float32)
    rrl = np.array([0.10], dtype=np.float32)

    peak_like = classify_features(cov, ons, np.array([1.90], dtype=np.float32),
                                  rrl, verbose=False)["prob_general"][0]
    assert 0.88 < peak_like < 0.94, peak_like

    # a small "mean over ON timesteps"-scale magnitude cannot get there
    mean_like = classify_features(cov, ons, np.array([0.30], dtype=np.float32),
                                  rrl, verbose=False)["prob_general"][0]
    assert mean_like < 0.85


def test_dead_features_are_not_general():
    """A dead feature has all-zero metrics; sigma(beta_0) must be < 0.5."""
    zeros = np.zeros(1, dtype=np.float32)
    for beta in (BETA_LIBERO, BETA_DROID):
        res = classify_features(zeros, zeros, zeros, zeros, beta=beta,
                                verbose=False)
        assert res["prob_general"][0] < 0.5
        assert not res["is_general"][0]


def test_active_fraction_reporting():
    """Paper Table 2 denominators are ACTIVE features, not dictionary width."""
    cov = np.array([0.99, 0.0, 0.5], dtype=np.float32)
    ons = np.array([3.0, 0.0, 0.0], dtype=np.float32)
    mag = np.array([1.9, 0.0, 0.1], dtype=np.float32)
    rrl = np.array([0.1, 0.0, 0.2], dtype=np.float32)
    is_active = np.array([True, False, True])
    res = classify_features(cov, ons, mag, rrl, is_active=is_active,
                            verbose=False)
    assert res["n_features"] == 3
    assert res["n_active"] == 2
    assert res["n_general_active"] == 1
    assert np.isclose(res["frac_general_active"], 0.5)
    assert np.isclose(res["frac_memorized_active"], 0.5)


def test_droid_coefficients_weight_coverage_more():
    """Paper Sec 4.2: in DROID, coverage dominates onsets (b2 = 3.2 x b1)."""
    assert BETA_DROID["coverage"] / BETA_DROID["mean_onsets"] > 3.0
    assert abs(BETA_LIBERO["coverage"] / BETA_LIBERO["mean_onsets"] - 1.0) < 0.1
    assert BETA_DROID["rel_run_length"] < BETA_LIBERO["rel_run_length"] < 0


# ---------------------------------------------------------------------------
def main():
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f)]
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
