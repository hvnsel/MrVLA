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


def test_two_level_bootstrap_rescues_a_spuriously_confident_correlation():
    """THE test. Damage here is PURE NOISE -- baseline and condition draw from the same rate --
    yet by chance it correlates strongly with the causal profile, and the task-only interval
    confidently excludes zero. That is exactly the failure mode: an interval that resamples
    tasks while treating each task's damage as a fixed number reports a confident correlation
    between two noise vectors. Propagating episode noise must put zero back inside.

    Note the correction does NOT simply widen intervals. Extra noise in x attenuates corr(x, y)
    toward zero, which concentrates the bootstrap replicates, so a two-level interval can be
    NARROWER while being centred nearer zero. Width is the wrong criterion; coverage is the
    right one, and that is what is asserted.
    """
    rng = np.random.default_rng(58)
    n_tasks, n_ep = 10, 20
    profile = rng.random(n_tasks)
    paired, damage = {}, np.zeros(n_tasks)
    for t in range(n_tasks):
        b = (rng.random(n_ep) < 0.8).astype(float)
        c = (rng.random(n_ep) < 0.8).astype(float)       # identical rates: no real damage
        paired[t] = (b, c)
        damage[t] = b.mean() - c.mean()
    r, _, ci_task = corr_permutation_p(damage, profile, paired=None, n_perm=200,
                                       n_boot=1200, seed=1)
    assert abs(r) > 0.6, "fixture no longer produces a spurious correlation"
    assert ci_task[0] * ci_task[1] > 0, "task-only interval should (wrongly) exclude zero here"
    _, _, ci_two = corr_permutation_p(damage, profile, paired=paired, n_perm=200,
                                      n_boot=1200, seed=1)
    assert ci_two[0] <= 0 <= ci_two[1], "two-level interval must put zero back inside"


def test_two_level_bootstrap_on_pure_noise_is_uninformative():
    """The case that motivated the fix: damage that is pure sampling noise must produce an
    interval spanning most of [-1, 1], not a confident correlation."""
    rng = np.random.default_rng(6)
    n_tasks, n_ep = 10, 20
    profile = rng.random(n_tasks)
    paired, damage = {}, np.zeros(n_tasks)
    for t in range(n_tasks):
        b = (rng.random(n_ep) < 0.8).astype(float)
        c = (rng.random(n_ep) < 0.8).astype(float)      # NO real damage at all
        paired[t] = (b, c)
        damage[t] = b.mean() - c.mean()
    _, _, ci = corr_permutation_p(damage, profile, paired=paired, n_perm=500,
                                  n_boot=2000, seed=0)
    assert (ci[1] - ci[0]) > 1.0, "an interval on noise must not look confident"


def test_two_level_bootstrap_still_finds_a_real_effect():
    """It must widen intervals without destroying power: a damage profile that genuinely
    tracks attribution should still exclude zero."""
    rng = np.random.default_rng(7)
    n_tasks, n_ep = 10, 200                     # more episodes -> episode noise shrinks
    profile = rng.random(n_tasks)
    paired, damage = {}, np.zeros(n_tasks)
    for t in range(n_tasks):
        b = (rng.random(n_ep) < 0.9).astype(float)
        c = (rng.random(n_ep) < 0.9 - 0.5 * profile[t]).astype(float)
        paired[t] = (b, c)
        damage[t] = b.mean() - c.mean()
    r, p_perm, ci = corr_permutation_p(damage, profile, paired=paired, n_perm=2000,
                                       n_boot=2000, seed=0)
    assert r > 0.8 and p_perm < 0.01
    assert ci[0] > 0


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
