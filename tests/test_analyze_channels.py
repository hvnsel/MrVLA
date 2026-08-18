"""Tests for analyze_channels.py -- the B1 statistics layer.

The load-bearing piece is `agreement_verdict`: it is the mechanism that stops the per-slot scale
confound being reported as a finding. The gripper emits extreme bins, whose u_contrast has the
largest norm, so a spurious "generality lives in the gripper" result shows up in absolute |phi|
and vanishes in shares. A verdict function that failed to flag that divergence would let the
confound through, so it is tested in both directions.

Everything else is pinned against constructions whose answer is known: a planted gripper
signal must be recovered, and a planted uniform one must NOT produce a gripper story.

Run directly:
    python tests/test_analyze_channels.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analyze_channels import (  # noqa: E402
    agreement_verdict, channel_concentration, common_language_effect, flip_rates, spearman,
)
from mrvla.attribution import rank_partial_both  # noqa: E402
from mrvla.channels import channel_participation_ratio  # noqa: E402

S_SLOTS, G, F, GRIP = 7, 8, 300, 6


def planted(gripper_signal: float, seed=0):
    """C_slot [7,G,F] where high-breadth features put `gripper_signal` of their mass on the
    gripper channel and the rest spread evenly. gripper_signal=0 gives no channel story."""
    rng = np.random.default_rng(seed)
    breadth = rng.random(F)                       # the "adjusted breadth" of each feature
    C = np.zeros((S_SLOTS, G, F))
    for j in range(F):
        w = np.full(S_SLOTS, (1.0 - gripper_signal * breadth[j]) / S_SLOTS)
        w[GRIP] += gripper_signal * breadth[j]
        C[:, :, j] = np.outer(w, rng.gamma(2.0, 1.0, G))
    return C, breadth


def test_agreement_verdict_flags_a_sign_contradiction():
    """THE confound detector. Absolute says one thing, share says the opposite -> must shout."""
    v = agreement_verdict(+0.40, -0.05)
    assert v.startswith("CONTRADICT")
    assert "share" in v


def test_agreement_verdict_flags_a_magnitude_divergence():
    v = agreement_verdict(+0.60, +0.05)
    assert v.startswith("diverge")


def test_agreement_verdict_accepts_a_consistent_result():
    assert agreement_verdict(+0.42, +0.38) == "AGREE"
    assert agreement_verdict(-0.42, -0.38) == "AGREE"


def test_agreement_verdict_handles_missing_values():
    assert agreement_verdict(float("nan"), 0.3) == "undetermined"


def test_uniform_per_slot_rescale_is_rank_invariant():
    """Documents the control's blind spot rather than pretending it has none.

    Multiplying one channel by a constant across ALL features maps each feature's profile share
    p to kp/(1+(k-1)p), which is monotone in p -- so every rank correlation is unchanged and the
    absolute/share comparison necessarily reports AGREE. That is a real limit: this split
    catches DIFFERENTIAL distortion and level shifts, not a uniform scale. Anyone reading AGREE
    as "no scale confound possible" would be wrong, so the behaviour is pinned here.
    """
    C, breadth = planted(gripper_signal=0.5, seed=7)
    C_infl = C * (1.0 + 2.0 * (np.arange(S_SLOTS) == GRIP))[:, None, None]
    r_share = spearman(breadth, channel_concentration(C, GRIP))
    r_abs = spearman(breadth, channel_concentration(C_infl, GRIP))
    assert abs(r_abs - r_share) < 1e-9
    assert agreement_verdict(r_abs, r_share) == "AGREE"
    # but the LEVELS do move, which is where the split earns its keep
    assert channel_concentration(C_infl, GRIP).mean() > channel_concentration(C, GRIP).mean() + 0.1


def test_differential_distortion_is_caught():
    """The realistic confound: inflation that depends on which features are active. Here the
    absolute view manufactures a gripper story that the share view does not support."""
    C, breadth = planted(gripper_signal=0.0, seed=8)
    bump = 1.0 + 5.0 * breadth                       # covaries with breadth, unlike a constant
    C_infl = C.copy()
    C_infl[GRIP] *= bump[None, :]
    r_abs = spearman(breadth, channel_concentration(C_infl, GRIP))
    r_share = spearman(breadth, channel_concentration(C, GRIP))
    assert r_abs > 0.8 and abs(r_share) < 0.15
    assert agreement_verdict(r_abs, r_share).startswith("diverge")


def test_planted_gripper_signal_is_recovered():
    C, breadth = planted(gripper_signal=0.8, seed=1)
    conc = channel_concentration(C, GRIP)
    assert spearman(breadth, conc) > 0.9


def test_no_channel_story_when_none_was_planted():
    """The negative control the whole analysis needs: uniform channel usage must NOT produce a
    correlation between breadth and gripper concentration."""
    C, breadth = planted(gripper_signal=0.0, seed=2)
    conc = channel_concentration(C, GRIP)
    assert abs(spearman(breadth, conc)) < 0.15


def test_gripper_signal_survives_the_confound_partial():
    """A real channel effect must survive rank-residualisation on magnitude and base rate --
    the same control the Path A headline uses."""
    C, breadth = planted(gripper_signal=0.8, seed=3)
    rng = np.random.default_rng(3)
    conc = channel_concentration(C, GRIP)
    mag = C.sum(axis=(0, 1))
    base = rng.random(F)
    assert rank_partial_both(conc, breadth, mag, base) > 0.5


def test_a_magnitude_driven_channel_effect_is_killed_by_the_partial():
    """The other direction: if gripper concentration is really just a magnitude shadow, the
    partial must remove it. Otherwise the control is decorative."""
    rng = np.random.default_rng(4)
    mag = rng.gamma(2.0, 1.0, F)
    conc = _ranklike(mag) + 0.01 * rng.standard_normal(F)     # concentration IS magnitude
    breadth = _ranklike(mag) + 0.01 * rng.standard_normal(F)  # breadth IS magnitude
    base = rng.random(F)
    assert spearman(breadth, conc) > 0.9                      # looks like a huge effect
    assert abs(rank_partial_both(conc, breadth, mag, base)) < 0.3   # and it is not one


def _ranklike(x):
    r = np.argsort(np.argsort(x)).astype(np.float64)
    return r / r.max()


def test_channel_pr_separates_single_channel_from_all_channel_features():
    C = np.zeros((S_SLOTS, G, 2))
    C[GRIP, :, 0] = 1.0
    C[:, :, 1] = 1.0
    pr = channel_participation_ratio(C)
    assert abs(pr[0] - 1.0) < 1e-9 and abs(pr[1] - S_SLOTS) < 1e-9


def test_common_language_effect_endpoints():
    assert abs(common_language_effect(np.array([1.0, 2, 3]), np.array([1.0, 2, 3])) - 0.5) < 1e-12
    assert common_language_effect(np.array([5.0, 6]), np.array([1.0, 2])) == 1.0
    assert np.isnan(common_language_effect(np.array([]), np.array([1.0])))


def test_flip_rates_reads_the_saved_counts_and_guards_empty_cells():
    """A channel with no transition decisions must yield NaN, not a divide-by-zero 0.0 that
    would read as 'this feature never matters there'."""
    chan = {"flip_projection_flip": np.array([[3.0, 0.0]]),
            "flip_projection_n": np.array([[10.0, 0.0]]),
            "flip_projection_flip_trans": np.array([[1.0, 0.0]]),
            "flip_projection_n_trans": np.array([[4.0, 0.0]])}
    rate, n = flip_rates(chan, "projection")
    assert abs(rate[0, 0] - 0.3) < 1e-12 and np.isnan(rate[0, 1])
    rate_t, _ = flip_rates(chan, "projection", transitions=True)
    assert abs(rate_t[0, 0] - 0.25) < 1e-12 and np.isnan(rate_t[0, 1])


if __name__ == "__main__":
    for name, fn in sorted(list(globals().items())):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("all analyze_channels tests passed")
