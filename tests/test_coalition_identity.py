"""Tests for coalition_identity.py.

The script exists to answer two questions that P2's controls leave open, and each has an error
direction that would be worse than not asking:

  * Q1 must FIRE when the coalition really is the always-on features (otherwise the check gives
    false reassurance on the one deflationary reading P2 cannot exclude), and must stay SILENT
    when mass and firing are unrelated.

  * Q2 must not be circular. Adjusted breadth is participation ratio with magnitude projected
    out, so a set chosen purely on magnitude has no mechanical route to a high score. The test
    builds a fixture where mass and breadth are decoupled and checks the profile sits at the
    50th percentile -- if it did not, every result would be an artefact of the selection.

Run directly:
    python tests/test_coalition_identity.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from coalition_identity import (  # noqa: E402
    analyse, overlap_vs_chance, percentile_profile, top_set,
)


class _Z(dict):
    """Stand-in for the saved npz."""


def _suite(G=10, F=2048, coupling=0.0, seed=0, spread_top=True):
    """C plus the derived arrays the npz carries.

    `coupling` in [0, 1] slides base rate from independent of mass (0) to identical ranking (1),
    which is what Q1 has to detect. `spread_top` controls whether the high-mass features also
    spread evenly across tasks, which is what Q2 has to detect.
    """
    rng = np.random.default_rng(seed)
    scale = np.exp(rng.standard_normal(F))                    # per-feature size
    if spread_top:
        share = np.full((G, F), 1.0 / G)                      # everything spreads evenly
    else:
        share = rng.dirichlet(np.full(G, 0.15), size=F).T     # everything is task-locked
    C = share * scale
    mag = C.sum(axis=0)
    pr = (C.sum(0) ** 2) / (C ** 2).sum(0)
    # base rate: blend an independent ranking with mass's own ranking
    indep = rng.random(F)
    r_mag = np.argsort(np.argsort(mag)) / (F - 1)
    base = (1 - coupling) * indep + coupling * r_mag
    return _Z(C=C, PR=pr, magnitude=mag, base_rate=base, is_active=np.ones(F, bool))


def test_top_set_respects_eligibility_and_size():
    v = np.arange(10.0)
    elig = np.array([True] * 5 + [False] * 5)
    assert top_set(v, 3, elig) == {4, 3, 2}          # 5..9 are ineligible
    assert len(top_set(v, 99, elig)) == 5            # cannot exceed the eligible pool


def test_overlap_vs_chance_matches_the_hypergeometric_mean():
    A, B = set(range(50)), set(range(25, 75))
    r = overlap_vs_chance(A, B, 2048)
    assert r["observed"] == 25
    assert abs(r["expected"] - 50 * 50 / 2048) < 1e-12
    assert abs(r["jaccard"] - 25 / 75) < 1e-12
    # a disjoint pair sits at zero, not at chance
    assert overlap_vs_chance(set(range(10)), set(range(10, 20)), 2048)["observed"] == 0


def test_percentile_profile_is_centred_for_an_unrelated_set():
    rng = np.random.default_rng(0)
    v = rng.standard_normal(2000)
    elig = np.ones(2000, bool)
    p = percentile_profile(set(rng.choice(2000, 50, replace=False).tolist()), v, elig)
    assert 40 < p["mean_pct"] < 60, p
    # and saturated for the set that IS the top of v
    top = percentile_profile(top_set(v, 50, elig), v, elig)
    assert top["mean_pct"] > 97, top


def test_q1_fires_when_the_coalition_IS_the_always_on_features():
    """The failure mode the whole script exists for. P2's controls cannot see this."""
    r = analyse("coupled", _suite(coupling=1.0), 50)
    q1 = r["q1_mass_vs_baserate"]
    assert q1["observed"] > 40, q1                        # near-total overlap
    assert q1["ratio"] > 20, q1
    assert r["q1_baserate_profile"]["mean_pct"] > 90


def test_q1_stays_silent_when_mass_and_firing_are_unrelated():
    r = analyse("indep", _suite(coupling=0.0), 50)
    q1 = r["q1_mass_vs_baserate"]
    assert q1["observed"] <= 5, q1                        # at or near chance (1.22)
    assert 35 < r["q1_baserate_profile"]["mean_pct"] < 65, r["q1_baserate_profile"]


def test_q2_is_not_circular_when_mass_and_breadth_are_decoupled():
    """THE guard. The coalition is chosen on magnitude alone, and adjusted breadth has magnitude
    residualised out, so a fixture where size says nothing about spread must land at the median.
    A high score here would mean every Q2 result is a selection artefact."""
    for spread in (True, False):
        r = analyse(f"spread={spread}", _suite(coupling=0.0, spread_top=spread, seed=3), 50)
        p = r["q2_adjbreadth_profile"]
        assert 30 < p["mean_pct"] < 70, (spread, p)


def test_q2_detects_a_coalition_that_really_is_broader_than_its_size_predicts():
    rng = np.random.default_rng(5)
    G, F = 10, 2048
    scale = np.exp(rng.standard_normal(F))
    share = rng.dirichlet(np.full(G, 0.15), size=F).T          # most features task-locked
    big = np.argsort(scale)[::-1][:50]
    share[:, big] = 1.0 / G                                    # but the biggest ones spread
    C = share * scale
    z = _Z(C=C, PR=(C.sum(0) ** 2) / (C ** 2).sum(0), magnitude=C.sum(0),
           base_rate=rng.random(F), is_active=np.ones(F, bool))
    p = analyse("broad", z, 50)["q2_adjbreadth_profile"]
    assert p["mean_pct"] > 75, p
    assert p["frac_above_50"] > 0.8, p


def test_strict_core_is_bounded_by_the_union_and_by_n():
    r = analyse("s", _suite(coupling=0.0), 50)
    assert r["strict_core_size"] <= 50
    assert r["strict_core_size"] <= r["per_task_top_n_union"]
    assert r["per_task_top_n_union"] <= 10 * 50


if __name__ == "__main__":
    for name, fn in sorted(list(globals().items())):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("all coalition_identity tests passed")
