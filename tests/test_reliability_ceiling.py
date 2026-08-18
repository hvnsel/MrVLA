"""Tests for reliability_ceiling.py -- the attenuation correction on the A x B null.

The trap this file guards is a direction error. Substituting r_yy = 1 for an unknown
reliability feels like "being conservative", but it yields a LOWER bound on |r_true|, which
argues against a null rather than for it. Getting that backwards would let the project
publish "the null survives correction for measurement reliability" on the strength of an
inequality pointing the other way. Each bound's direction is pinned against the attenuation
identity it comes from.

Run directly:
    python tests/test_reliability_ceiling.py
"""

from __future__ import annotations

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from reliability_ceiling import (  # noqa: E402
    breakeven_reliability, disattenuate, lower_bound_one_reliability,
)


def test_disattenuation_inverts_the_attenuation_identity():
    """Generate an observed r from a known truth, then recover the truth."""
    r_true, ra, rb = 0.40, 0.7, 0.5
    r_obs = r_true * math.sqrt(ra * rb)
    d = disattenuate(r_obs, ra, rb)
    assert abs(d["r_corrected"] - r_true) < 1e-12
    assert abs(d["ceiling"] - math.sqrt(ra * rb)) < 1e-12
    assert not d["saturated"]


def test_perfect_reliability_is_a_no_op():
    d = disattenuate(-0.127, 1.0, 1.0)
    assert abs(d["r_corrected"] + 0.127) < 1e-12
    assert abs(d["ceiling"] - 1.0) < 1e-12


def test_saturation_is_flagged():
    """An observed correlation above the ceiling is impossible under the model, so one of the
    reliability estimates must be wrong -- that must be surfaced, not silently corrected to
    |r| > 1."""
    d = disattenuate(0.9, 0.3, 0.3)
    assert d["saturated"]
    assert abs(d["r_corrected"]) > 1.0


def test_disattenuate_rejects_impossible_reliabilities():
    assert "error" in disattenuate(0.2, 0.0, 0.5)
    assert "error" in disattenuate(0.2, 0.5, 1.5)
    assert "error" in lower_bound_one_reliability(0.2, -0.1)


def test_unknown_reliability_gives_a_LOWER_bound_not_an_upper_one():
    """THE direction test. With r_yy unknown, r_obs / sqrt(r_xx) sits at or BELOW the true
    correlation for every admissible r_yy -- so it can strengthen a positive result and can
    never establish a null."""
    r_true, ra = 0.25, 0.6
    lb = lower_bound_one_reliability(r_true * math.sqrt(ra * 1.0), ra)
    assert abs(lb["abs_r_true_lower_bound"] - r_true) < 1e-12       # tight at r_yy = 1
    for rb in (0.9, 0.6, 0.3, 0.1):
        obs = r_true * math.sqrt(ra * rb)
        bound = lower_bound_one_reliability(obs, ra)["abs_r_true_lower_bound"]
        assert bound <= r_true + 1e-12                              # never overstates


def test_ceiling_upper_bound_shrinks_with_the_unknown_reliability():
    """sqrt(r_xx) is the most the ceiling can be; any real r_yy < 1 lowers it. This is why a
    high 'max possible ceiling' alone does not defend the null."""
    lb = lower_bound_one_reliability(-0.127, 0.72)
    assert abs(lb["max_possible_ceiling"] - math.sqrt(0.72)) < 1e-12
    for rb in (0.8, 0.4):
        assert disattenuate(-0.127, 0.72, rb)["ceiling"] < lb["max_possible_ceiling"]


def test_breakeven_reliability_is_the_threshold_boundary():
    """At exactly the breakeven r_yy the corrected correlation equals the threshold; above it
    the null holds, below it the null is not established."""
    r_obs, ra, thr = -0.127, 0.72, 0.30
    be = breakeven_reliability(r_obs, ra, thr)
    rb = be["rel_b_needed"]
    assert abs(abs(disattenuate(r_obs, ra, rb)["r_corrected"]) - thr) < 1e-9
    assert abs(disattenuate(r_obs, ra, min(1.0, rb * 1.5))["r_corrected"]) < thr
    assert abs(disattenuate(r_obs, ra, rb * 0.5)["r_corrected"]) > thr


def test_breakeven_flags_an_unattainable_requirement():
    """A large observed correlation can demand r_yy > 1, i.e. no reliability could make it a
    null. That has to be reported rather than returned as a plausible-looking number."""
    be = breakeven_reliability(0.6, 0.5, 0.30)
    assert be["rel_b_needed"] > 1.0
    assert not be["rel_b_needed_feasible"]
    be = breakeven_reliability(-0.127, 0.72, 0.30)
    assert be["rel_b_needed_feasible"]


def test_saturating_reliability_marks_where_the_measurement_says_nothing():
    """Below rel_b_saturating the corrected |r| exceeds 1: the data constrain nothing."""
    r_obs, ra = -0.127, 0.72
    be = breakeven_reliability(r_obs, ra, 0.30)
    assert abs(abs(disattenuate(r_obs, ra, be["rel_b_saturating"])["r_corrected"]) - 1.0) < 1e-9
    assert be["rel_b_saturating"] < be["rel_b_needed"]


if __name__ == "__main__":
    for name, fn in sorted(list(globals().items())):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("all reliability_ceiling tests passed")
