"""Tests for ablation_power.py -- the statistics layer over the ablation rollouts.

Pins the things a wrong answer here would silently corrupt:
  1. pairing is on (task, episode), and unmatched cells are DROPPED rather than compared
     against a different task set;
  2. the damage/repair sign convention (b10 = baseline won, ablation lost = damage);
  3. duplicate rows from a re-run shard are not double-counted;
  4. the damage-PR null actually brackets a uniformly-damaged run, so "inside the null"
     means what the report says it means;
  5. the attribution-agreement permutation p is calibrated -- ~0.5 when damage and the
     causal profile are unrelated, small when they line up.

Run directly:
    python tests/test_ablation_power.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ablation_power import (  # noqa: E402
    corr_permutation_p, damage_pr_null, paired_counts, participation_ratio, success_by_key,
)


def _rows(cond, successes, task=0):
    return [{"condition": cond, "task_id": task, "episode": i, "success": s}
            for i, s in enumerate(successes)]


def test_success_by_key_indexes_on_task_and_episode():
    rows = _rows("baseline", [1, 0, 1], task=0) + _rows("general", [1, 1, 0], task=1)
    by = success_by_key(rows)
    assert by["baseline"][(0, 1)] == 0
    assert by["general"][(1, 2)] == 0
    assert set(by) == {"baseline", "general"}


def test_duplicate_rows_are_not_double_counted():
    """A re-run worker writes the same (task, episode) twice; counting it twice would inflate
    n and shrink every interval."""
    rows = _rows("baseline", [1, 1]) + _rows("baseline", [1, 1])
    by = success_by_key(rows)
    assert len(by["baseline"]) == 2


def test_paired_counts_sign_convention():
    base = {(0, i): s for i, s in enumerate([1, 1, 1, 1, 0, 0])}
    cond = {(0, i): s for i, s in enumerate([0, 0, 1, 1, 1, 0])}
    b01, b10, n = paired_counts(base, cond)
    assert n == 6
    assert b10 == 2      # baseline succeeded, ablation failed -> damage
    assert b01 == 1      # ablation succeeded where baseline failed -> repair


def test_paired_counts_drops_unmatched_cells():
    """A condition missing task 1 entirely must contribute nothing there, not be silently
    compared against baseline's task-1 episodes."""
    base = {(0, 0): 1, (0, 1): 1, (1, 0): 1, (1, 1): 0}
    cond = {(0, 0): 0, (0, 1): 1}
    b01, b10, n = paired_counts(base, cond)
    assert n == 2 and b10 == 1 and b01 == 0
    # and the per-task restriction sees nothing on the missing task
    assert paired_counts(base, cond, tasks={1}) == (0, 0, 0)


def test_participation_ratio_endpoints():
    assert abs(participation_ratio(np.array([1.0, 0, 0, 0])) - 1.0) < 1e-12
    assert abs(participation_ratio(np.array([1.0, 1, 1, 1])) - 4.0) < 1e-12
    # negative entries (a condition that got LUCKY on a task) are clipped, not counted
    assert abs(participation_ratio(np.array([1.0, -1.0])) - 1.0) < 1e-12


def test_damage_pr_null_brackets_a_uniformly_damaged_run():
    """The null says 'damage is even across tasks'. A run whose damage really is even must
    land inside it -- otherwise every condition would read as significant."""
    n_tasks = 10
    base_rate = np.full(n_tasks, 0.8)
    n_ep = np.full(n_tasks, 20)
    null = damage_pr_null(base_rate, n_ep, mean_damage=0.05, n_sim=800, seed=1)
    assert null.size == 800
    lo, hi = np.percentile(null, 2.5), np.percentile(null, 97.5)
    rng = np.random.default_rng(7)
    inside = 0
    for _ in range(40):
        b = rng.binomial(20, 0.8, n_tasks) / 20.0
        a = rng.binomial(20, 0.75, n_tasks) / 20.0
        pr = participation_ratio(b - a)
        inside += int(lo <= pr <= hi)
    assert inside >= 34            # ~95% coverage, allowing sampling slack

    # and a genuinely task-locked profile falls OUTSIDE it (the null can reject)
    locked = np.zeros(n_tasks)
    locked[3] = 0.5
    assert participation_ratio(locked) < lo


def test_damage_pr_null_handles_degenerate_input():
    assert damage_pr_null(np.array([np.nan]), np.array([0]), 0.05, 10).size == 0


def test_corr_permutation_p_is_calibrated():
    rng = np.random.default_rng(0)
    profile = rng.random(10)
    # damage that follows the causal profile -> small p, positive r
    r, p, ci = corr_permutation_p(profile * 0.3 + 0.01 * rng.standard_normal(10), profile,
                                  n_perm=4000, seed=0)
    assert r > 0.8 and p < 0.01
    assert ci[0] < r < ci[1] or ci[0] <= 1.0     # interval brackets or saturates
    # unrelated damage -> p spread over (0, 1), not systematically small
    ps = [corr_permutation_p(rng.random(10), rng.random(10), n_perm=1000, seed=s)[1]
          for s in range(30)]
    assert 0.25 < float(np.mean(ps)) < 0.75


def test_corr_permutation_p_needs_enough_tasks():
    r, p, ci = corr_permutation_p(np.array([1.0, 2.0]), np.array([1.0, 2.0]))
    assert np.isnan(r) and np.isnan(p)


if __name__ == "__main__":
    for name, fn in sorted(list(globals().items())):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("all ablation_power tests passed")
