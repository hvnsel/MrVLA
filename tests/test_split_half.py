"""Tests for split_half_breadth: the Path A reliability gate.

Pins the five things that must be right:
  1. Spearman-Brown is the textbook formula, and refuses to "correct" a non-positive rho;
  2. a feature set whose breadth is genuinely stable across tasks reproduces at high rho;
  3. task-set-relative breadth (each half driven by unrelated tasks) does NOT reproduce --
     the gate can fail, which is the whole point of running it;
  4. the label-shuffle control sits at ~0, so the floor is calibrated;
  5. magnitude is recomputed PER HALF -- reusing the full-set magnitude would leak the
     held-out tasks into the confound control.

Run directly:
    python tests/test_split_half.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mrvla.attribution import total_magnitude
from split_half_breadth import (
    half_breadth, spearman, spearman_brown, split_half_rho, summarize,
)


def test_spearman_brown_formula_and_guard():
    assert abs(spearman_brown(0.5) - 2 / 3) < 1e-12          # 2*.5/(1+.5)
    assert abs(spearman_brown(0.8) - 8 / 9) < 1e-12
    assert spearman_brown(1.0) == 1.0
    # a non-positive split-half rho means "no reliable signal"; correcting it is meaningless
    assert np.isnan(spearman_brown(0.0))
    assert np.isnan(spearman_brown(-0.3))
    assert np.isnan(spearman_brown(float("nan")))


def _stable_C(rng, G=10, F=300):
    """Each feature has a fixed breadth: the first third drive every task, the middle third
    drive half, the last third drive one task. Any half of the tasks reveals the same order."""
    C = np.zeros((G, F))
    for j in range(F):
        if j % 3 == 0:
            C[:, j] = rng.uniform(0.8, 1.2, G)               # broad
        elif j % 3 == 1:
            C[::2, j] = rng.uniform(0.8, 1.2, len(C[::2]))   # every other task
        else:
            C[j % G, j] = 1.0                                # single task
    return C


def test_stable_breadth_reproduces():
    rng = np.random.default_rng(0)
    C = _stable_C(rng)
    F = C.shape[1]
    base_rate = rng.uniform(0.01, 0.5, F)
    active = np.ones(F, bool)
    r = split_half_rho(C, base_rate, active, n_splits=40, seed=1, adjusted=False)
    assert np.nanmedian(r) > 0.7, np.nanmedian(r)


def test_task_set_relative_breadth_does_not_reproduce():
    """Half the tasks and the other half drive DISJOINT feature sets: breadth measured on one
    half says nothing about the other. The gate must report this, not paper over it."""
    rng = np.random.default_rng(1)
    G, F = 10, 300
    C = np.zeros((G, F))
    # tasks 0-4 only ever excite features 0-149, tasks 5-9 only features 150-299
    C[:5, :150] = rng.uniform(0.5, 1.5, (5, 150))
    C[5:, 150:] = rng.uniform(0.5, 1.5, (5, 150))
    base_rate = rng.uniform(0.01, 0.5, F)
    active = np.ones(F, bool)
    r = split_half_rho(C, base_rate, active, n_splits=40, seed=2, adjusted=False)
    # features are active in both halves only by chance; agreement must not look stable
    assert np.nanmedian(r) < 0.5, np.nanmedian(r)


def test_shuffle_floor_is_zero():
    rng = np.random.default_rng(2)
    C = _stable_C(rng)
    F = C.shape[1]
    base_rate = rng.uniform(0.01, 0.5, F)
    active = np.ones(F, bool)
    r = split_half_rho(C, base_rate, active, n_splits=60, seed=3,
                       adjusted=True, shuffle=True)
    assert abs(np.nanmedian(r)) < 0.15, np.nanmedian(r)


def test_half_breadth_recomputes_magnitude_from_the_half():
    """The magnitude used in the confound residualisation must come from the half, not the
    full task set -- otherwise the held-out tasks leak into the control."""
    rng = np.random.default_rng(3)
    G, F = 10, 50
    C = rng.uniform(0, 1, (G, F))
    base_rate = rng.uniform(0.01, 0.5, F)
    active = np.ones(F, bool)
    half = np.array([0, 1, 2, 3, 4])
    _PR, mag, _adj = half_breadth(C[half], base_rate, active)
    assert np.allclose(mag, total_magnitude(C[half]))
    assert not np.allclose(mag, total_magnitude(C))          # NOT the full-set magnitude


def test_ties_are_averaged_not_broken_arbitrarily():
    """PR == 1.0 exactly for every single-task feature, so ties are everywhere in real data.
    Tied values must receive the SAME rank; breaking them in arbitrary order would make two
    halves disagree for a purely numerical reason and deflate the reliability estimate."""
    from split_half_breadth import _avg_ranks
    r = _avg_ranks(np.array([1.0, 1.0, 1.0, 2.0]))
    assert np.allclose(r[:3], 1.0) and r[3] == 3.0            # mean of ranks 0,1,2 == 1
    assert np.allclose(_avg_ranks(np.full(6, 4.2)), 2.5)      # all tied -> all identical
    # two vectors tied in the same places correlate perfectly; arbitrary tie-breaking would not
    a = np.array([1.0, 1.0, 2.0, 3.0, 3.0, 4.0])
    b = np.array([5.0, 5.0, 6.0, 7.0, 7.0, 8.0])
    assert abs(spearman(a, b) - 1.0) < 1e-12


def test_spearman_and_summarize_edges():
    assert np.isnan(spearman(np.array([1.0, 2.0]), np.array([1.0, 2.0])))   # <3 pairs
    assert np.isnan(spearman(np.ones(5), np.arange(5.0)))                   # constant side
    assert abs(spearman(np.arange(10.0), np.arange(10.0)) - 1.0) < 1e-12
    assert abs(spearman(np.arange(10.0), -np.arange(10.0)) + 1.0) < 1e-12
    s = summarize(np.array([np.nan, 0.4, 0.6, 0.5]))
    assert s["n"] == 3 and abs(s["median"] - 0.5) < 1e-12
    assert summarize(np.array([np.nan, np.nan]))["median"] is None


def main():
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f)]
    failed = 0
    for n, f in tests:
        try:
            f(); print(f"  PASS  {n}")
        except AssertionError as e:
            failed += 1; print(f"  FAIL  {n}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
