"""Tests for permutation_null.py -- the Path A negative control.

The claim this file has to defend is unusual: that the control EXPERIMENT_PLAN.md §3.2b
prescribes (permute task labels) is a no-op on this statistic, and that the column shuffle is
the floor that actually tests something. Both directions are pinned here, because getting
this backwards means either publishing a fake collapse or discarding a real result after
running the wrong control.

Run directly:
    python tests/test_permutation_null.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mrvla.attribution import loto_partial_both, participation_ratio, total_magnitude  # noqa: E402
from permutation_null import (  # noqa: E402
    column_shuffle, feature_shuffle_partials, observed_statistic, row_shuffle, run_null,
    summarize,
)


def structured_C(G=10, F=400, seed=0):
    """A causal matrix with a real breadth spectrum: each feature spreads its mass over tasks
    according to its own concentration parameter, so PR genuinely varies across features."""
    rng = np.random.default_rng(seed)
    mass = rng.gamma(1.5, 1.0, size=F)
    spread = rng.uniform(0.05, 1.0, size=F)
    C = np.stack([rng.dirichlet(np.full(G, s * 5)) for s in spread], axis=1) * mass * G
    return C, rng.uniform(0, 1, F)


def test_observed_statistic_reports_folds():
    C, br = structured_C()
    obs = observed_statistic(C, br)
    assert obs["n_folds"] == C.shape[0]
    assert len(obs["folds"]) == obs["n_folds"]
    assert obs["min_fold"] <= obs["partial_both"]
    assert obs["partial_both"] > 0.05          # the synthetic matrix has real structure


def test_row_shuffle_is_a_noop_which_is_why_it_is_not_a_control():
    """LOTO evaluates every fold, so permuting a feature's task labels only re-deals the same
    G (train-PR, held-out) pairs into different folds. The statistic barely moves."""
    C, br = structured_C()
    real = loto_partial_both(C, br).mean()
    rng = np.random.default_rng(1)
    perm = np.mean([loto_partial_both(row_shuffle(C, rng), br).mean() for _ in range(8)])
    assert abs(perm - real) < 0.02             # indistinguishable from the real value
    assert perm > 0.5 * real                   # nowhere near a collapse to zero


def test_column_shuffle_collapses_the_statistic():
    """The valid floor: destroying a feature's identity across tasks must remove most of it."""
    C, br = structured_C()
    real = loto_partial_both(C, br).mean()
    rng = np.random.default_rng(2)
    null = np.array([loto_partial_both(column_shuffle(C, rng), br).mean() for _ in range(20)])
    assert null.mean() < 0.25 * real
    assert real > np.percentile(null, 95)      # the real value clears the floor


def test_column_shuffle_preserves_task_marginals():
    """It has to be a within-row permutation: each task keeps its own distribution of causal
    mass, so the floor is not contaminated by changing how much any task contributes."""
    C, _ = structured_C(G=6, F=50)
    Cs = column_shuffle(C, np.random.default_rng(0))
    assert np.allclose(np.sort(C, axis=1), np.sort(Cs, axis=1))
    assert not np.allclose(C, Cs)


def test_feature_shuffle_is_the_estimator_floor():
    """Decoupling only the target must give ~0; anything else means the estimator is biased."""
    C, br = structured_C()
    rng = np.random.default_rng(3)
    vals = np.array([feature_shuffle_partials(C, br, rng).mean() for _ in range(30)])
    assert abs(vals.mean()) < 0.03


def test_feature_shuffle_leaves_predictor_and_controls_intact():
    """Only the held-out vector is permuted -- training PR and magnitude must be untouched,
    otherwise the floor would be testing a different quantity than the reported statistic."""
    C, br = structured_C(G=5, F=60)
    keep = np.arange(5) != 0
    pr_before = participation_ratio(C[keep]).copy()
    mag_before = total_magnitude(C[keep]).copy()
    feature_shuffle_partials(C, br, np.random.default_rng(0))
    assert np.allclose(participation_ratio(C[keep]), pr_before)
    assert np.allclose(total_magnitude(C[keep]), mag_before)


def test_run_null_dispatch_and_summary():
    C, br = structured_C(G=6, F=120)
    obs = observed_statistic(C, br)["partial_both"]
    null = run_null(C, br, "column_shuffle", n_perm=25, seed=0)
    s = summarize("column_shuffle", obs, null)
    assert s["n"] == 25
    assert 0.0 <= s["p_one_sided"] <= 1.0
    assert s["z"] > 0                                # the real value sits above the floor
    assert summarize("empty", obs, np.array([]))["n"] == 0


def test_nulls_are_reproducible_under_a_fixed_seed():
    C, br = structured_C(G=6, F=120)
    a = run_null(C, br, "column_shuffle", n_perm=10, seed=5)
    b = run_null(C, br, "column_shuffle", n_perm=10, seed=5)
    assert np.allclose(a, b)


if __name__ == "__main__":
    for name, fn in sorted(list(globals().items())):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("all permutation_null tests passed")
