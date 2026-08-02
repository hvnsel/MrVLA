"""Synthetic-data tests for the episode-scoring and weighting pipeline.

Covers the rho coverage floor, the temporal trimmed mean, and the
score->weight maps.  Feature-level generality metrics are tested separately
in test_generality_classifier.py against the paper's equations.

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
