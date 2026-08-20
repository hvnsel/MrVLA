"""Tests for channels_raw.py -- the uncorrected channel view.

The point of the script is that a correction which corrects nothing must be invisible. So the
tests construct both worlds and check the comparison reports them correctly:

  * where the slots genuinely differ in scale, the absolute and share correlations diverge and
    the script must show it;
  * where they do not, the two agree and no conclusion can depend on which is reported.

Run directly:
    python tests/test_channels_raw.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from channels_raw import rank_partial  # noqa: E402
from mrvla.channels import channel_profile, decision_shares  # noqa: E402


def test_rank_partial_matches_a_plain_correlation_when_controls_are_noise():
    rng = np.random.default_rng(0)
    n = 800
    x = rng.standard_normal(n)
    y = 0.7 * x + rng.standard_normal(n)
    r = rank_partial(y, x, rng.standard_normal(n), rng.standard_normal(n))
    assert 0.4 < r < 0.75, r


def test_rank_partial_removes_a_control_that_explains_everything():
    rng = np.random.default_rng(1)
    n = 800
    c = rng.standard_normal(n)
    x = c + 0.05 * rng.standard_normal(n)
    y = c + 0.05 * rng.standard_normal(n)
    assert abs(rank_partial(y, x, c, rng.standard_normal(n))) < 0.25


def test_rank_partial_guards_short_input():
    assert np.isnan(rank_partial(np.arange(3.0), np.arange(3.0),
                                 np.arange(3.0), np.arange(3.0)))


def _decisions(S=7, G=10, F=400, n_per=60, scale=None, seed=0):
    """Simulate at the DECISION level, which is the only way to build the share matrix honestly.

    `decision_shares` normalises each decision BEFORE accumulation, so the share matrix cannot be
    recovered by post-hoc normalising C_abs -- an earlier version of this test tried exactly that
    and compared two things that were never the same object.

    Feature f puts a fixed fraction of its influence on each slot (the planted profile). `scale`
    multiplies every decision at a given slot, standing in for a per-slot ||u_contrast||
    difference.
    """
    rng = np.random.default_rng(seed)
    prof = rng.dirichlet(np.full(S, 0.7), size=F).T          # [S, F], sums to 1 per feature
    size = np.exp(rng.standard_normal(F))
    mult = np.ones(S) if scale is None else np.asarray(scale, dtype=np.float64)

    C_abs = np.zeros((S, G, F))
    C_shr = np.zeros((S, G, F))
    for s in range(S):
        for g in range(G):
            phi = (prof[s] * size)[None, :] * np.exp(0.3 * rng.standard_normal((n_per, F)))
            phi *= mult[s]                                    # the per-slot scale
            C_abs[s, g] = phi.sum(axis=0) / n_per
            C_shr[s, g] = decision_shares(phi).sum(axis=0) / n_per
    return C_abs, C_shr, prof


def test_share_is_right_when_the_confound_is_real():
    """What the correction is for. With one slot inflated tenfold, the absolute profile is
    dominated by it and lands far from the planted truth, while the share profile is unaffected."""
    C_abs, C_shr, prof = _decisions(scale=[1, 1, 1, 1, 1, 1, 10.0])
    pa, ps = channel_profile(C_abs), channel_profile(C_shr)
    assert np.abs(pa - prof).max() > 0.4, np.abs(pa - prof).max()
    assert np.abs(ps - prof).max() < 0.10, np.abs(ps - prof).max()


def test_ABSOLUTE_is_more_accurate_when_the_confound_is_NOT_real():
    """The cost of correcting for something that is not there, which is the situation P5c
    reports on this data (every slot within 8% of an even mass share).

    The normalisation carries a distortion of its own -- it divides each slot by ITS OWN typical
    decision total, so slots whose features happen to carry more total influence get scaled down
    whether or not any ||u_contrast|| difference exists. With no confound to remove, that leaves
    the share profile roughly six times further from the planted truth than the raw one.

    This is why the uncorrected numbers must be reported alongside, not replaced."""
    C_abs, C_shr, prof = _decisions()
    pa, ps = channel_profile(C_abs), channel_profile(C_shr)
    err_abs = np.abs(pa - prof).max()
    err_shr = np.abs(ps - prof).max()
    assert err_abs < err_shr, (err_abs, err_shr)
    assert err_shr > 3 * err_abs, (err_abs, err_shr)


def test_the_gap_between_the_two_forms_signals_which_regime_you_are_in():
    """The diagnostic the script prints. A large absolute-vs-share gap means a real scale
    difference; a small one means the correction is inert and either form may be reported."""
    gap_clean = np.abs(channel_profile(_decisions()[0])
                       - channel_profile(_decisions()[1])).max()
    Ca, Cs, _ = _decisions(scale=[1, 1, 1, 1, 1, 1, 10.0])
    gap_conf = np.abs(channel_profile(Ca) - channel_profile(Cs)).max()
    assert gap_conf > 5 * gap_clean, (gap_clean, gap_conf)


def test_mass_share_is_even_when_scales_are_even():
    C_abs, _, _ = _decisions()
    tot = C_abs.sum()
    fr = np.array([C_abs[s].sum() / tot for s in range(C_abs.shape[0])])
    assert np.abs(fr - 1 / 7).max() < 0.05, fr


def test_mass_share_detects_an_inflated_slot():
    """What the P5c check would have shown had the confound been real: the inflated slot takes
    far more than its 1/7 share. On the real data it takes 1.08x, which is why it is not real."""
    C_abs, _, _ = _decisions(scale=[1, 1, 1, 1, 1, 1, 10.0])
    assert C_abs[6].sum() / C_abs.sum() > 0.4


if __name__ == "__main__":
    for name, fn in sorted(list(globals().items())):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("all channels_raw tests passed")
