"""Tests for concentration_robustness.py.

Two things carry the P2 reproducibility claim and both are pinned here:

  * the SHUFFLE baseline must behave like a chance baseline -- close to N^2/pool when the pool
    really is the whole dictionary, and correctly LARGER when only part of the dictionary is
    active. That second case is the entire reason for measuring chance instead of assuming it:
    the analytic N^2/F overstates the ratio exactly when the active set is small.

  * the jackknife must widen when tasks disagree and tighten when they agree, and must not
    claim precision on a statistic driven by one outlier task.

Run directly:
    python tests/test_concentration_robustness.py
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from concentration_robustness import (  # noqa: E402
    active_count, analyse, column_shuffle, jackknife, overlap_mean, shuffled_overlap_null,
    union_top_n,
)
from causal_concentration import n_effective  # noqa: E402


def _args(**kw):
    d = dict(n_perm=60, sae_k=100, seed=0)
    d.update(kw)
    return argparse.Namespace(**d)


def _shared(G=10, F=2048, core=50, seed=0):
    """One shared coalition: the same `core` features are large in every task."""
    rng = np.random.default_rng(seed)
    C = rng.lognormal(0, 0.3, (G, F)) * 0.01
    C[:, :core] *= 300.0
    return C


def _private(G=10, F=2048, core=50, seed=1):
    """A DIFFERENT coalition per task -- same marginals, no shared identity."""
    rng = np.random.default_rng(seed)
    C = _shared(G, F, core, seed)
    return np.stack([rng.permutation(row) for row in C])


def _partly_active(G=10, F=2048, active=300, core=50, seed=2):
    """Only `active` of F features carry any mass at all -- the case N^2/F gets wrong."""
    rng = np.random.default_rng(seed)
    C = np.zeros((G, F))
    C[:, :active] = rng.lognormal(0, 0.3, (G, active)) * 0.01
    C[:, :core] *= 300.0
    return C


def test_active_count_sees_the_real_pool():
    assert active_count(_shared()) == 2048
    assert active_count(_partly_active(active=300)) == 300


def test_shuffle_baseline_matches_the_analytic_one_when_all_features_are_active():
    """Sanity: where N^2/F is the right answer, the measured baseline should agree with it."""
    C = _shared()
    n = 50
    null = shuffled_overlap_null(C, n, 60, 0)
    assert abs(null.mean() - n * n / 2048) < 0.5, (null.mean(), n * n / 2048)


def test_shuffle_baseline_beats_the_analytic_one_when_the_pool_is_small():
    """THE test. With only 300 of 2048 features active, N^2/F understates chance sevenfold and
    inflates the reported ratio by the same factor (41.5x becomes 6.0x). The restricted shuffle
    infers the pool from the data and needs no assumption at all."""
    C = _partly_active(active=300)
    n = 50
    measured = shuffled_overlap_null(C, n, 60, 0).mean()
    analytic_F, analytic_active = n * n / 2048, n * n / 300
    assert measured > 4 * analytic_F, (measured, analytic_F)
    assert abs(measured - analytic_active) < 0.15 * analytic_active, (measured, analytic_active)


def test_unrestricted_shuffle_reproduces_the_flaw_it_exists_to_fix():
    """The regression guard for the bug this design originally had. Permuting across ALL F
    columns scatters mass into structurally dead features -- a state no task can produce -- and
    the baseline collapses back to N^2/F, reintroducing the very pool assumption the empirical
    null was built to avoid."""
    C = _partly_active(active=300)
    n = 50
    loose = shuffled_overlap_null(C, n, 60, 0, restrict_to_active=False).mean()
    tight = shuffled_overlap_null(C, n, 60, 0, restrict_to_active=True).mean()
    assert abs(loose - n * n / 2048) < 0.3, (loose, n * n / 2048)     # the flaw
    assert tight > 5 * loose, (tight, loose)                          # the fix


def test_overlap_separates_a_shared_coalition_from_private_ones():
    shared, private = _shared(), _private()
    n = 50
    assert overlap_mean(shared, n) > 45
    obs_p = overlap_mean(private, n)
    chance = shuffled_overlap_null(private, n, 60, 0).mean()
    assert obs_p < 3 * chance, (obs_p, chance)


def test_union_is_interpretable_at_both_extremes():
    n = 50
    assert union_top_n(_shared(), n) <= 60          # one coalition: union ~ n
    assert union_top_n(_private(), n) > 300         # private: union approaches G*n
    assert union_top_n(_private(), n) <= 10 * n


def test_jackknife_is_tight_when_tasks_agree_and_wide_when_one_dissents():
    agree = _shared()
    tight = jackknife(lambda X: n_effective(X.sum(axis=0)), agree)
    dissent = agree.copy()
    dissent[0] = 0.0
    dissent[0, 1000:1005] = 1e4                     # one task with a totally different profile
    wide = jackknife(lambda X: n_effective(X.sum(axis=0)), dissent)
    assert wide["se"] > 5 * tight["se"], (wide["se"], tight["se"])
    assert wide["loo_max"] - wide["loo_min"] > tight["loo_max"] - tight["loo_min"]


def test_jackknife_estimate_is_the_full_sample_value():
    C = _shared()
    jk = jackknife(lambda X: n_effective(X.sum(axis=0)), C)
    assert abs(jk["estimate"] - n_effective(C.sum(axis=0))) < 1e-9
    assert jk["lo"] < jk["estimate"] < jk["hi"]


def test_analyse_flags_an_underused_dictionary():
    """If barely more features are active than n_eff says are carrying the mass, the
    'concentration' is near-trivial and the report must say so."""
    C = _partly_active(active=120, core=100)
    res = analyse("underused", C, _args())
    assert res["n_active"] == 120
    assert res["n_active"] < 4 * res["concentration"]["n_eff"]["estimate"]


def test_analyse_sweeps_n_and_skips_impossible_sizes():
    C = _partly_active(active=300)
    res = analyse("small", C, _args())
    ns = [s["n"] for s in res["top_n_sweep"]]
    assert 400 not in ns and 200 in ns             # N must not exceed the active pool
    for s in res["top_n_sweep"]:
        assert s["union"] <= s["union_max"]
        assert s["ratio_vs_analytic_F"] > s["ratio_vs_shuffle"]   # N^2/F inflates here


def test_column_shuffle_preserves_every_row_marginal():
    C = _shared()
    Cs = column_shuffle(C, np.random.default_rng(0))
    for g in range(C.shape[0]):
        assert np.allclose(np.sort(C[g]), np.sort(Cs[g]))


if __name__ == "__main__":
    for name, fn in sorted(list(globals().items())):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("all concentration_robustness tests passed")
