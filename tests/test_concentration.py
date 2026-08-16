"""Tests for causal_concentration.py and the compare_recurrence_groups --target guard.

Concentration statistics are easy to write and easy to get subtly wrong (a Gini that does not
reach its endpoints, an overlap "ratio to chance" whose chance term is misderived). Since
these numbers are destined to become the sentence "the action is carried by N of 2048
features", each is pinned against a hand-computable case.

Run directly:
    python tests/test_concentration.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from causal_concentration import (  # noqa: E402
    gini, n_effective, top_share, topn_overlap_vs_chance,
)
from compare_recurrence_groups import (  # noqa: E402
    check_target_matches_paths, target_in_rec_path,
)


def test_n_effective_endpoints():
    assert abs(n_effective(np.ones(100)) - 100.0) < 1e-9      # perfectly even
    one = np.zeros(100); one[7] = 3.0
    assert abs(n_effective(one) - 1.0) < 1e-9                 # all in one feature
    half = np.zeros(100); half[:10] = 1.0
    assert abs(n_effective(half) - 10.0) < 1e-9


def test_top_share_and_gini_endpoints():
    v = np.array([4.0, 3.0, 2.0, 1.0])
    assert abs(top_share(v, 1) - 0.4) < 1e-12
    assert abs(top_share(v, 2) - 0.7) < 1e-12
    assert abs(top_share(v, 99) - 1.0) < 1e-12                # n larger than the vector
    assert abs(gini(np.ones(50))) < 1e-9                      # even -> 0
    spike = np.zeros(1000); spike[0] = 1.0
    assert gini(spike) > 0.99                                 # all in one -> ~1
    assert np.isnan(gini(np.zeros(10)))


def test_topn_overlap_chance_term_is_correct():
    """Two independent top-n draws from F features share n^2/F on average. Verified by
    simulation so the reported 'ratio to chance' means what it claims."""
    F, n, G = 500, 50, 12
    rng = np.random.default_rng(0)
    C = rng.random((G, F))                    # unstructured: no feature is broadly important
    obs, chance, ratio = topn_overlap_vs_chance(C, n)
    assert abs(chance - n * n / F) < 1e-12
    assert 0.6 < ratio < 1.6                  # unstructured data sits at chance


def test_topn_overlap_detects_a_shared_coalition():
    """The reproducibility claim: if every task recruits the same 50 features, the ratio must
    be far above 1."""
    F, n, G = 500, 50, 12
    rng = np.random.default_rng(1)
    C = rng.random((G, F)) * 0.1
    C[:, :n] += 10.0                          # one shared coalition drives every task
    obs, chance, ratio = topn_overlap_vs_chance(C, n)
    assert obs == n
    assert ratio > 5.0


def test_concentration_separates_causal_mass_from_firing_rate():
    """The control that matters: a matrix whose causal mass is concentrated while firing rate
    is flat must show a higher Gini for causal mass than for firing."""
    F = 1000
    rng = np.random.default_rng(2)
    mass = np.zeros(F); mass[:20] = 100.0; mass[20:] = rng.random(F - 20) * 0.1
    firing = rng.uniform(0.2, 0.8, F)
    assert gini(mass) > gini(firing) + 0.4
    assert n_effective(mass) < 0.1 * n_effective(firing)


def test_target_extraction_from_rec_filename():
    assert target_in_rec_path("/x/RECURRENCE_ACTION/layer_31_target_goal.npz") == "goal"
    assert target_in_rec_path("layer_08_target_libero10.npz") == "libero10"
    assert target_in_rec_path("layer_31_attribution.npz") == ""


def test_target_guard_rejects_a_mislabelled_comparison():
    """The bug this replaces: --target only labelled the output, so `--target spatial` with
    the goal npz produced a goal comparison filed under 'spatial'."""
    rec = "/w/RECURRENCE_ACTION/layer_31_target_goal.npz"
    attr = "/w/ATTR/goal_k100/layer_31_attribution.npz"
    check_target_matches_paths("goal", rec, attr)          # consistent -> no raise
    check_target_matches_paths("", rec, attr)              # unlabelled -> no opinion
    raised = False
    try:
        check_target_matches_paths("spatial", rec, attr)
    except SystemExit:
        raised = True
    assert raised, "a target that disagrees with both input paths must not run"
    # the escape hatch still works, for deliberately unconventional naming
    check_target_matches_paths("spatial", rec, attr, allow_mismatch=True)


def test_target_guard_catches_an_attr_path_from_the_wrong_suite():
    """Half-switched arguments are the realistic failure: --rec updated, --attr left behind."""
    raised = False
    try:
        check_target_matches_paths("spatial",
                                   "/w/RECURRENCE_ACTION/layer_31_target_spatial.npz",
                                   "/w/ATTR/goal_k100/layer_31_attribution.npz")
    except SystemExit:
        raised = True
    assert raised


if __name__ == "__main__":
    for name, fn in sorted(list(globals().items())):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("all concentration tests passed")
