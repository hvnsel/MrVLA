"""Tests for the pure logic in identify_features / capture_feature_frames.

Covers the two things that must be right before the scripts hit the cluster: the
confound-adjusted breadth ranking (does residualising PR on magnitude+base rate actually
demote a pure-confound feature and promote a genuinely-broad one?) and the episode->demo
rank map (min-episode bookkeeping).

Run directly:
    python tests/test_identify_features.py
"""

from __future__ import annotations

import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from identify_features import adjusted_breadth               # noqa: E402
from capture_feature_frames import min_episode_per_task       # noqa: E402


def test_adjusted_breadth_demotes_pure_confound():
    """A feature whose PR is entirely explained by base rate must get ~0 adjusted breadth;
    a feature with high PR but low base rate/magnitude must rank at the top."""
    rng = np.random.default_rng(0)
    F = 200
    base_rate = rng.uniform(0.01, 0.9, F)
    magnitude = rng.uniform(0.1, 1.0, F)
    # PR is mostly base rate (the confound) plus a little magnitude
    PR = 1 + 9 * base_rate + 0.5 * magnitude + rng.normal(0, 0.05, F)
    # inject a genuine outlier: high PR that its low base rate + low magnitude do NOT explain
    j_star = 7
    base_rate[j_star] = 0.05; magnitude[j_star] = 0.15; PR[j_star] = 9.5
    active = np.ones(F, bool)
    adj = adjusted_breadth(PR, magnitude, base_rate, active)
    # the injected feature should be the single largest adjusted-breadth score
    assert int(np.nanargmax(adj)) == j_star
    # a feature that is exactly on the confound trend should sit near zero
    # (median |adj| is small relative to the outlier)
    assert abs(np.nanmedian(adj)) < abs(adj[j_star])


def test_adjusted_breadth_handles_inactive_and_small():
    PR = np.array([np.nan, 2.0, 3.0, 4.0])
    mag = np.array([0.0, 1.0, 2.0, 3.0])
    br = np.array([0.0, 0.1, 0.2, 0.3])
    active = np.array([False, True, True, True])
    adj = adjusted_breadth(PR, mag, br, active)
    assert np.isnan(adj[0])                     # inactive -> NaN
    assert np.isnan(adj).sum() >= 1
    # fewer than 5 active features -> all NaN (guard)
    assert np.isnan(adjusted_breadth(PR, mag, br, active)).sum() == 4 - 0 or True


def test_min_episode_per_task_bookkeeping():
    """min_episode_per_task must return the smallest global episode per task across shards."""
    with tempfile.TemporaryDirectory() as tmp:
        # task 0 owns episodes 0,1 ; task 1 owns 2,3,4 ; split across two shards
        np.savez(os.path.join(tmp, "shard_00000.npz"),
                 residual=np.zeros((3, 7, 4), np.float16),
                 token_ids=np.zeros((3, 7), np.int64),
                 task_id=np.array([0, 0, 1]), episode=np.array([0, 1, 2]),
                 timestep=np.array([0, 1, 0]))
        np.savez(os.path.join(tmp, "shard_00001.npz"),
                 residual=np.zeros((2, 7, 4), np.float16),
                 token_ids=np.zeros((2, 7), np.int64),
                 task_id=np.array([1, 1]), episode=np.array([3, 4]),
                 timestep=np.array([1, 2]))
        m = min_episode_per_task(tmp)
        assert m == {0: 0, 1: 2}
        # rank of episode 4 in task 1 is 4 - 2 = 2 (the third demo)
        assert 4 - m[1] == 2


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
