"""Synthetic-data tests for the corrected generality pipeline.

Covers the three classifier fixes (hysteresis onset detection, unified
mean-activation-magnitude, episode-weighted coverage/run-length estimators),
the rho coverage floor, the temporal trimmed mean, and the score->weight maps.

Run directly (no pytest needed):
    python tests/test_generality_fixes.py
or via pytest:
    pytest tests/test_generality_fixes.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mrvla.generality_classifier import (  # noqa: E402
    BETA,
    classify_features,
    compute_metrics_hysteresis,
    hysteresis_state,
    sigmoid,
)
from mrvla.episode_generality_variance import (  # noqa: E402
    apply_coverage_floor,
    per_episode_trimmed_mean,
)
from mrvla.episode_weights import (  # noqa: E402
    build_arms,
    effective_sample_size,
    power_weights,
    reflect,
)


TAU_ON, TAU_OFF = 0.1, 0.05


# ---------------------------------------------------------------------------
# Synthetic dataset: 2 episodes (T=7 and T=5), 4 features
# ---------------------------------------------------------------------------
def make_dataset():
    """Hand-crafted codes with known ground-truth metrics.

    feature 0: flickers around tau_on in ep0 only
               ep0 z = [0, .2, .08, .2, .04, .2, 0]
               single-threshold(z > .1) sees 3 onsets; hysteresis sees 2
               (runs of length 3 and 1), a-bar over ON steps = .17
    feature 1: dead everywhere
    feature 2: ep0 burst of 2 (z=.3), ep1 always on (z=.5)
    feature 3: ep0 three isolated 1-step bursts, ep1 always on
               (episode-weighted run length 3.0 vs pooled 2.0)
    """
    T0, T1, F = 7, 5, 4
    z0 = np.zeros((T0, F), dtype=np.float32)
    z1 = np.zeros((T1, F), dtype=np.float32)

    z0[:, 0] = [0.0, 0.2, 0.08, 0.2, 0.04, 0.2, 0.0]
    z0[0:2, 2] = 0.3
    z1[:, 2] = 0.5
    z0[[0, 2, 4], 3] = 0.2
    z1[:, 3] = 0.5

    z = np.vstack([z0, z1])
    episode = np.array([0] * T0 + [1] * T1)
    timestep = np.array(list(range(T0)) + list(range(T1)))
    return z, episode, timestep


# ---------------------------------------------------------------------------
# Fix 2: hysteresis
# ---------------------------------------------------------------------------
def test_hysteresis_debounces_flicker():
    z_ep = np.array([[0.0], [0.2], [0.08], [0.2], [0.04], [0.2], [0.0]],
                    dtype=np.float32)
    state = hysteresis_state(z_ep, TAU_ON, TAU_OFF)[:, 0]
    # dead zone (.08) holds ON; below tau_off (.04) releases
    assert state.tolist() == [False, True, True, True, False, True, False]

    # single-threshold comparison: 3 onsets vs hysteresis's 2
    single = (z_ep[:, 0] > TAU_ON)
    n_onsets_single = int(np.diff(np.r_[0, single.astype(int), 0]).clip(0).sum())
    n_onsets_hyst = int(np.diff(np.r_[0, state.astype(int), 0]).clip(0).sum())
    assert n_onsets_single == 3
    assert n_onsets_hyst == 2


def test_hysteresis_degenerates_to_single_threshold():
    rng = np.random.default_rng(0)
    z_ep = rng.uniform(0, 0.3, size=(50, 8)).astype(np.float32)
    state = hysteresis_state(z_ep, TAU_ON, TAU_ON)
    assert np.array_equal(state, z_ep > TAU_ON)


def test_hysteresis_rejects_bad_taus():
    try:
        hysteresis_state(np.zeros((3, 1)), tau_on=0.05, tau_off=0.1)
    except ValueError:
        pass
    else:
        raise AssertionError("tau_off > tau_on should raise")


def test_hysteresis_never_on_below_tau_on():
    # a feature that never exceeds tau_on must never turn ON,
    # even while above tau_off
    z_ep = np.full((10, 1), 0.08, dtype=np.float32)
    assert not hysteresis_state(z_ep, TAU_ON, TAU_OFF).any()


# ---------------------------------------------------------------------------
# Fixes 1+3: unified metrics
# ---------------------------------------------------------------------------
def test_metrics_ground_truth():
    z, episode, timestep = make_dataset()
    m = compute_metrics_hysteresis(z, episode, timestep, TAU_ON, TAU_OFF)

    # feature 0 — fires only in ep0
    assert np.isclose(m["coverage"][0], 0.5)
    assert np.isclose(m["mean_onsets"][0], 2.0)           # debounced: 2 not 3
    assert np.isclose(m["mean_run_length"][0], 2.0)       # (3+1)/2 within ep0
    assert np.isclose(m["rel_run_length"][0], 2.0 / 7.0)  # ep0 has T=7
    # a-bar over ON steps INCLUDES the dead-zone value .08
    assert np.isclose(m["mean_act_mag"][0], (0.2 + 0.08 + 0.2 + 0.2) / 4)

    # feature 1 — dead
    for key in ("coverage", "mean_onsets", "mean_run_length",
                "rel_run_length", "mean_act_mag"):
        assert m[key][1] == 0.0

    # feature 2 — fires in both episodes
    assert np.isclose(m["coverage"][2], 1.0)
    assert np.isclose(m["mean_onsets"][2], 1.0)
    assert np.isclose(m["mean_run_length"][2], (2 + 5) / 2)
    assert np.isclose(m["rel_run_length"][2], (2 / 7 + 5 / 5) / 2)
    assert np.isclose(m["mean_act_mag"][2], (0.3 * 2 + 0.5 * 5) / 7)

    assert np.isclose(m["ep_mean_len"], 6.0)              # (7+5)/2


def test_run_length_is_episode_weighted_not_burst_weighted():
    z, episode, timestep = make_dataset()
    m = compute_metrics_hysteresis(z, episode, timestep, TAU_ON, TAU_OFF)
    # feature 3: ep0 has 3 one-step bursts (run 1), ep1 one 5-step burst (run 5)
    episode_weighted = (1.0 + 5.0) / 2                    # = 3.0 (each ep once)
    burst_weighted = (3 + 5) / (3 + 1)                    # = 2.0 (old pooled)
    assert np.isclose(m["mean_run_length"][3], episode_weighted)
    assert not np.isclose(m["mean_run_length"][3], burst_weighted)


def test_metrics_invariant_to_row_order():
    z, episode, timestep = make_dataset()
    m1 = compute_metrics_hysteresis(z, episode, timestep, TAU_ON, TAU_OFF)
    rng = np.random.default_rng(7)
    perm = rng.permutation(len(episode))
    m2 = compute_metrics_hysteresis(z[perm], episode[perm], timestep[perm],
                                    TAU_ON, TAU_OFF)
    for key in ("coverage", "mean_onsets", "mean_run_length",
                "rel_run_length", "mean_act_mag"):
        assert np.allclose(m1[key], m2[key]), key


def test_classifier_uses_rel_run_length_as_given():
    cov = np.array([0.8], dtype=np.float32)
    ons = np.array([1.5], dtype=np.float32)
    rrl = np.array([0.25], dtype=np.float32)
    mag = np.array([0.4], dtype=np.float32)
    res = classify_features(cov, ons, rrl, mag, ep_mean_len=100.0, verbose=False)
    expected_logit = (BETA["intercept"] + BETA["mean_onsets"] * 1.5
                      + BETA["coverage"] * 0.8
                      + BETA["mean_act_magnitude"] * 0.4
                      + BETA["rel_run_length"] * 0.25)
    assert np.isclose(res["prob_general"][0], sigmoid(np.array(expected_logit)))
    # ep_mean_len must NOT rescale rel_run_length any more
    res2 = classify_features(cov, ons, rrl, mag, ep_mean_len=1.0, verbose=False)
    assert np.isclose(res["prob_general"][0], res2["prob_general"][0])


# ---------------------------------------------------------------------------
# Step 2: coverage floor + trimmed mean
# ---------------------------------------------------------------------------
def test_coverage_floor():
    is_general = np.array([True, True, False])
    prob = np.array([0.9, 0.8, 0.7], dtype=np.float32)
    coverage = np.array([0.02, 0.5, 0.05], dtype=np.float32)
    g, p, n_killed = apply_coverage_floor(is_general, prob, coverage, rho=0.1)
    assert g.tolist() == [False, True, False]
    assert n_killed == 1
    # soft mass is stripped for EVERY low-coverage feature
    assert np.allclose(p, [0.0, 0.8, 0.0])


def test_trimmed_mean_drops_ends_temporally():
    # junk at both temporal ends, plateau in the middle
    ratio = np.array([0.0, .5, .5, .5, .5, .5, .5, .5, .5, 0.0])
    episode = np.zeros(10, dtype=int)
    timestep = np.arange(10)
    _, trimmed, kept = per_episode_trimmed_mean(ratio, episode, timestep, 0.10)
    assert np.isclose(trimmed[0], 0.5)                    # junk excluded
    assert kept[0] == 8
    _, plain, _ = per_episode_trimmed_mean(ratio, episode, timestep, 0.0)
    assert np.isclose(plain[0], 0.4)                      # junk included

    # the trim is TEMPORAL: junk placed mid-episode is NOT removed
    ratio_mid = np.array([.5, .5, .5, .5, 0.0, 0.0, .5, .5, .5, .5])
    _, trimmed_mid, _ = per_episode_trimmed_mean(ratio_mid, episode, timestep, 0.10)
    assert trimmed_mid[0] < 0.5


def test_trimmed_mean_short_episode_fallback():
    # trimming would leave nothing -> full mean
    ratio = np.array([0.1, 0.9])
    episode = np.zeros(2, dtype=int)
    timestep = np.arange(2)
    _, trimmed, kept = per_episode_trimmed_mean(ratio, episode, timestep, 0.5)
    assert np.isclose(trimmed[0], 0.5)
    assert kept[0] == 2


# ---------------------------------------------------------------------------
# Step 4: weight maps
# ---------------------------------------------------------------------------
def test_weight_arms_properties():
    rng = np.random.default_rng(3)
    scores = rng.uniform(0.05, 0.6, size=428)
    arms = build_arms(scores, seed=0)

    expected = {"uniform", "mild", "medium", "sharp",
                "inverted_mild", "inverted_medium", "inverted_sharp",
                "random_mild", "random_medium", "random_sharp"}
    assert set(arms) == expected

    for name, w in arms.items():
        assert np.isclose(w.mean(), 1.0), name            # mean-1 normalisation
        assert (w > 0).all(), name                        # no episode deleted

    # aggressiveness is monotone in gamma
    assert arms["mild"].std() < arms["medium"].std() < arms["sharp"].std()
    # real arms preserve the score ordering
    order = np.argsort(scores)
    for name in ("mild", "medium", "sharp"):
        assert (np.diff(arms[name][order]) >= -1e-12).all(), name
    # inverted arms reverse it
    assert (np.diff(arms["inverted_sharp"][order]) <= 1e-12).all()
    # random arms are permutations of the real arms
    for name in ("mild", "medium", "sharp"):
        assert np.allclose(np.sort(arms[f"random_{name}"]), np.sort(arms[name]))

    # ESS: uniform = E; sharper maps concentrate mass -> lower ESS
    E = len(scores)
    assert np.isclose(effective_sample_size(arms["uniform"]), E)
    assert effective_sample_size(arms["sharp"]) < effective_sample_size(arms["mild"]) <= E


def test_weight_edge_cases():
    # all-zero scores degrade gracefully to uniform
    w = power_weights(np.zeros(10), gamma=2.0)
    assert np.allclose(w, 1.0)
    # near-zero scores are floored, not deleted
    scores = np.array([1e-9, 0.3, 0.3, 0.3])
    w = power_weights(scores, gamma=4.0, min_weight=0.1, max_weight=10.0)
    assert w.min() > 0.05
    assert np.isclose(w.mean(), 1.0)
    # reflection keeps range, reverses order
    s = np.array([0.1, 0.2, 0.5])
    r = reflect(s)
    assert np.isclose(r.min(), s.min()) and np.isclose(r.max(), s.max())
    assert (np.argsort(r) == np.argsort(s)[::-1]).all()


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
