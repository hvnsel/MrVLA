"""Tests for mrvla.channels -- the per-action-channel decomposition (B1).

Three things would silently corrupt every channel result and are pinned here: the slot/decision
flattening convention (getting it backwards transposes the whole analysis), the share
normalisation that makes slots comparable at all, and the transition mask that controls for the
gripper channel being near-constant.

Run directly:
    python tests/test_channels.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mrvla.channels import (  # noqa: E402
    accumulate_slot_task, channel_mass, channel_participation_ratio, channel_profile,
    decision_shares, slot_index, transition_mask,
)


def test_slot_index_matches_the_reshape_convention():
    """`res.reshape(n*7, d)` puts decision i's seven slots in consecutive rows."""
    s = slot_index(3, 7)
    assert s.size == 21
    assert list(s[:7]) == list(range(7))
    assert list(s[7:14]) == list(range(7))
    # row r belongs to decision r // 7 and slot r % 7
    assert all(s[r] == r % 7 for r in range(21))


def test_accumulate_lands_mass_in_the_right_cell():
    S, G, F = 7, 3, 4
    dsum = np.zeros((S, G, F)); dcount = np.zeros((S, G), dtype=np.int64)
    values = np.array([[1.0, 0, 0, 0], [0, 2.0, 0, 0], [0, 0, 3.0, 0]])
    slots = np.array([0, 6, 0])
    tasks = np.array([1, 2, 1])
    accumulate_slot_task(dsum, dcount, values, slots, tasks)
    assert dsum[0, 1, 0] == 1.0 and dsum[0, 1, 2] == 3.0
    assert dsum[6, 2, 1] == 2.0
    assert dcount[0, 1] == 2 and dcount[6, 2] == 1
    assert dsum.sum() == 6.0


def test_accumulate_is_additive_across_shards():
    """Streaming correctness: two half-shards must equal one whole."""
    rng = np.random.default_rng(0)
    S, G, F, n = 7, 4, 5, 40
    values = rng.random((n, F))
    slots = rng.integers(0, S, n)
    tasks = rng.integers(0, G, n)
    whole_s = np.zeros((S, G, F)); whole_c = np.zeros((S, G), dtype=np.int64)
    accumulate_slot_task(whole_s, whole_c, values, slots, tasks)
    part_s = np.zeros((S, G, F)); part_c = np.zeros((S, G), dtype=np.int64)
    for sl in (slice(0, 17), slice(17, n)):
        accumulate_slot_task(part_s, part_c, values[sl], slots[sl], tasks[sl])
    assert np.allclose(whole_s, part_s) and np.array_equal(whole_c, part_c)


def test_decision_shares_remove_the_per_slot_scale():
    """THE normalisation test. Two decisions with identical relative structure but a 100x scale
    difference -- which is exactly what the gripper slot's larger ||u_contrast|| produces -- must
    come out identical in share space."""
    phi = np.array([[1.0, 2.0, 1.0], [100.0, 200.0, 100.0]])
    sh = decision_shares(phi)
    assert np.allclose(sh[0], sh[1])
    assert np.allclose(sh.sum(axis=1), 1.0)
    assert np.allclose(sh[0], [0.25, 0.5, 0.25])


def test_decision_shares_survive_an_all_zero_decision():
    sh = decision_shares(np.zeros((2, 3)))
    assert np.isfinite(sh).all()


def test_channel_pr_endpoints():
    S, G, F = 7, 2, 3
    C = np.zeros((S, G, F))
    C[3, :, 0] = 1.0                      # feature 0: one channel only
    C[:, :, 1] = 1.0                      # feature 1: all seven, evenly
    C[0, :, 2] = 1.0; C[1, :, 2] = 1.0    # feature 2: two channels
    pr = channel_participation_ratio(C)
    assert abs(pr[0] - 1.0) < 1e-12
    assert abs(pr[1] - 7.0) < 1e-12
    assert abs(pr[2] - 2.0) < 1e-12


def test_channel_pr_is_scale_free():
    """Breadth over channels must not be strength over channels."""
    S, G, F = 7, 2, 2
    C = np.zeros((S, G, F))
    C[:3, :, 0] = 1.0
    C[:3, :, 1] = 1000.0
    pr = channel_participation_ratio(C)
    assert abs(pr[0] - pr[1]) < 1e-9


def test_channel_profile_is_per_feature_normalised():
    S, G, F = 7, 2, 2
    C = np.zeros((S, G, F))
    C[6, :, 0] = 5.0                      # a pure gripper feature
    C[0, :, 1] = 1.0; C[6, :, 1] = 3.0    # mostly gripper, some dx
    p = channel_profile(C)
    assert np.allclose(p.sum(axis=0), 1.0)
    assert abs(p[6, 0] - 1.0) < 1e-12
    assert abs(p[6, 1] - 0.75) < 1e-12
    assert np.allclose(channel_mass(C)[6], [10.0, 6.0])   # 5*2 tasks, 3*2 tasks


def test_transition_mask_finds_changes_within_an_episode_only():
    """A gripper that closes once in an episode gives exactly one transition, and episode
    boundaries never count as one."""
    bins = np.array([10, 10, 10, 200, 200,  10, 10])
    ep = np.array([0, 0, 0, 0, 0,  1, 1])
    ts = np.array([0, 1, 2, 3, 4,  0, 1])
    m = transition_mask(bins, ep, ts)
    assert list(m) == [False, False, False, True, False, False, False]
    assert m.sum() == 1


def test_transition_mask_is_order_independent():
    """Shards need not arrive sorted; the answer must not depend on row order."""
    bins = np.array([10, 10, 200, 200, 10])
    ep = np.array([0, 0, 0, 1, 1])
    ts = np.array([0, 1, 2, 0, 1])
    ref = transition_mask(bins, ep, ts)
    perm = np.array([3, 0, 4, 2, 1])
    got = transition_mask(bins[perm], ep[perm], ts[perm])
    assert list(got) == list(ref[perm])


def test_transition_mask_on_a_constant_channel_is_empty():
    """The degeneracy this control exists for: a channel that never changes yields no
    transition decisions at all, and any 'importance' there is plateau-riding."""
    n = 50
    m = transition_mask(np.full(n, 128), np.zeros(n, int), np.arange(n))
    assert not m.any()


if __name__ == "__main__":
    for name, fn in sorted(list(globals().items())):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("all channels tests passed")
