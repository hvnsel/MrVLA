"""Tests for recurrence_vs_breadth: equal-count binning and the permutation test.

The binning must produce ordered bins covering all active features; the permutation test must
return a SMALL p when the top-breadth features genuinely have higher y, and a LARGE p (~uniform)
when breadth is unrelated to y.

Run directly:
    python tests/test_recurrence_curve.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from recurrence_vs_breadth import bin_stats, perm_pvalue


def test_bin_stats_shapes_and_order():
    rng = np.random.default_rng(0)
    F = 500
    breadth = rng.normal(0, 1, F)
    y = 0.3 + 0.1 * breadth + rng.normal(0, 0.05, F)   # y rises with breadth
    active = np.ones(F, bool)
    cx, my, se = bin_stats(breadth, y, active, n_bins=10)
    assert len(cx) == len(my) == len(se) == 10
    assert cx[0] < cx[-1] and cx[0] > 0 and cx[-1] < 100     # percentile centers, ordered
    assert my[-1] > my[0]                                    # high-breadth bin has higher y
    assert (se >= 0).all()


def test_perm_pvalue_detects_real_signal():
    rng = np.random.default_rng(1)
    F = 600
    breadth = rng.normal(0, 1, F)
    # top-breadth features have clearly higher y
    y = rng.normal(0, 0.05, F)
    top = breadth > np.quantile(breadth, 0.9)
    y[top] += 0.2
    active = np.ones(F, bool)
    obs, p = perm_pvalue(breadth, y, active, top_frac=0.10, n_perm=2000, seed=0)
    assert obs > 0.1, obs
    assert p < 0.01, p


def test_perm_pvalue_null_is_not_significant():
    rng = np.random.default_rng(2)
    F = 600
    breadth = rng.normal(0, 1, F)
    y = rng.normal(0, 0.05, F)                # unrelated to breadth
    active = np.ones(F, bool)
    obs, p = perm_pvalue(breadth, y, active, top_frac=0.10, n_perm=2000, seed=0)
    assert p > 0.05, (obs, p)


def test_inactive_features_excluded():
    F = 100
    breadth = np.arange(F, dtype=float)
    y = np.zeros(F); y[:50] = np.nan          # first 50 have nan y
    active = np.ones(F, bool)
    cx, my, se = bin_stats(breadth, y, active, n_bins=5)
    assert np.isfinite(my).all()              # nan-y features dropped, no nan means


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
