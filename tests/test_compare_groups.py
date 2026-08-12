"""Tests for compare_recurrence_groups (Path B Part 1).

Confirms the common-language effect size behaves (0.5 when equal, >0.5 when the first group
is larger, <0.5 when smaller) and handles ties/empties.

Run directly:
    python tests/test_compare_groups.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from compare_recurrence_groups import common_language_effect


def test_cles_equal_distributions_is_half():
    rng = np.random.default_rng(0)
    a = rng.normal(0, 1, 500); b = rng.normal(0, 1, 500)
    assert abs(common_language_effect(a, b) - 0.5) < 0.05


def test_cles_a_larger_is_above_half():
    rng = np.random.default_rng(1)
    a = rng.normal(1.0, 1, 400); b = rng.normal(0.0, 1, 400)
    assert common_language_effect(a, b) > 0.6


def test_cles_a_smaller_is_below_half():
    rng = np.random.default_rng(2)
    a = rng.normal(0.0, 1, 400); b = rng.normal(1.0, 1, 400)
    assert common_language_effect(a, b) < 0.4


def test_cles_ties_and_empty():
    # all identical -> every pair is a tie -> 0.5
    a = np.full(10, 3.0); b = np.full(10, 3.0)
    assert abs(common_language_effect(a, b) - 0.5) < 1e-9
    # empty group -> nan, no crash
    assert np.isnan(common_language_effect(np.array([]), np.array([1.0, 2.0])))
    # NaNs are dropped, not counted
    a2 = np.array([1.0, np.nan, 5.0]); b2 = np.array([0.0, 0.0])
    assert common_language_effect(a2, b2) == 1.0


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
