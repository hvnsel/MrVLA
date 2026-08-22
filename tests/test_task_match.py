"""Pins the task-margin statistic and the within-task correlation.

Two tests carry the weight:

  * `test_margin_collapses_under_the_row_shuffle` -- the shuffle is the control that decides
    whether the margin measures anything task-specific at all. If a permuted C row scores as
    well as the true one, the statistic is reading the always-on component every row shares
    (results.md P2b) and the subtraction has failed to do its job.
  * `test_within_task_ignores_a_pure_between_task_effect` -- tasks differ in typical
    duration, so a POOLED correlation reports task identity as signal. This builds data
    where between-task difference is the only structure and confirms the within-task
    estimator returns nothing.

Run directly:
    python tests/test_task_match.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mrvla.task_match import (  # noqa: E402
    row_normalize, task_margin, task_similarity, within_task_partial, within_task_spearman,
)

G, F = 5, 64


def _world(seed=0, n=400, shared=0.0):
    """Decisions whose |phi| is drawn around their own task's profile.

    `shared` adds a component common to EVERY task row -- the always-on set P2b describes.
    Raising it must not raise the margin, which is the whole point of the subtraction.
    """
    r = np.random.default_rng(seed)
    common = np.abs(r.normal(size=F)) * shared
    C = np.abs(r.normal(size=(G, F))) + common[None, :]
    task = r.integers(0, G, size=n)
    phi = np.abs(C[task] + 0.3 * r.normal(size=(n, F)))
    return phi, C, task


def test_row_normalize_leaves_zero_rows_alone():
    out = row_normalize(np.array([[3.0, 4.0], [0.0, 0.0]]))
    assert np.allclose(out[0], [0.6, 0.8])
    assert np.allclose(out[1], [0.0, 0.0])       # not NaN


def test_similarity_is_a_cosine():
    phi = np.array([[1.0, 0.0, 0.0]])
    C = np.array([[2.0, 0.0, 0.0], [0.0, 5.0, 0.0]])
    s = task_similarity(phi, C)
    assert np.allclose(s, [[1.0, 0.0]])


def test_margin_is_positive_when_the_decision_matches_its_own_task():
    phi, C, task = _world(seed=1)
    m = task_margin(task_similarity(phi, C), task)
    assert m.mean() > 0.5, m.mean()
    assert (m > 0).mean() > 0.9, (m > 0).mean()


def test_margin_collapses_under_the_row_shuffle():
    """The control that decides whether the statistic is task-specific."""
    phi, C, task = _world(seed=2)
    sim = task_similarity(phi, C)
    real = task_margin(sim, task)
    perm = np.array([(g + 1) % G for g in range(G)])
    shuf = task_margin(sim, task, row_perm=perm)
    assert shuf.mean() < 0, shuf.mean()          # a wrong row is beaten by the true one
    assert real.mean() - shuf.mean() > 0.1, (real.mean(), shuf.mean())


def _lo_hi(shared):
    phi, C, task = _world(seed=3, shared=shared)
    return task_similarity(phi, C), task


def test_raw_margin_shrinks_under_a_large_shared_component():
    """Why the default is standardised. P2b's always-on set inflates every C row alike,
    driving all G cosines together and compressing the raw margin -- here about 75x --
    while it stays correctly signed. Rank correlation would survive that; a raw value
    compared across suites or SAEs would not."""
    lo = task_margin(*_lo_hi(0.0), standardize=False)
    hi = task_margin(*_lo_hi(8.0), standardize=False)
    assert lo.mean() > 0 and hi.mean() > 0                 # still correctly signed
    assert hi.mean() < 0.1 * lo.mean(), (lo.mean(), hi.mean())


def test_standardised_margin_survives_a_large_shared_component():
    """Dividing by the spread ACROSS tasks removes the shared component's effect on scale,
    because it inflates every competitor equally."""
    lo = task_margin(*_lo_hi(0.0))
    hi = task_margin(*_lo_hi(8.0))
    assert hi.mean() > 0.5 * lo.mean(), (lo.mean(), hi.mean())


def test_margin_is_near_zero_when_phi_is_unrelated_to_any_task():
    r = np.random.default_rng(4)
    _phi, C, _t = _world(seed=5)
    phi = np.abs(r.normal(size=(400, F)))        # carries no task structure
    task = r.integers(0, G, size=400)
    m = task_margin(task_similarity(phi, C), task)
    assert abs(np.nanmean(m)) < 0.3, np.nanmean(m)


# ---------------------------------------------------------------------------
# within-task correlation
# ---------------------------------------------------------------------------
def test_within_task_recovers_a_planted_within_task_effect():
    r = np.random.default_rng(6)
    task = np.repeat(np.arange(10), 40)
    x = r.normal(size=task.size)
    y = 2.0 * x + r.normal(size=task.size) * 0.5
    out = within_task_spearman(x, y, task)
    assert out["mean"] > 0.8, out["mean"]
    assert out["n_positive"] == 10


def test_within_task_ignores_a_pure_between_task_effect():
    """x and y both track task index and nothing else. Pooled this is a perfect
    correlation; within task there is no variance left and the answer must be ~0."""
    task = np.repeat(np.arange(10), 40)
    r = np.random.default_rng(7)
    x = task + 0.01 * r.normal(size=task.size)
    y = task + 0.01 * r.normal(size=task.size)
    from mrvla.prior_gates import spearman
    assert spearman(x, y) > 0.95                  # pooled: looks perfect
    assert abs(within_task_spearman(x, y, task)["mean"]) < 0.2   # within: nothing


def test_within_task_partial_removes_a_planted_control():
    r = np.random.default_rng(8)
    task = np.repeat(np.arange(10), 60)
    c = r.normal(size=task.size)
    x, y = c + 0.3 * r.normal(size=task.size), c + 0.3 * r.normal(size=task.size)
    assert within_task_spearman(x, y, task)["mean"] > 0.6
    assert abs(within_task_partial(x, y, [c], task)["mean"]) < 0.2


def test_small_tasks_are_skipped_not_scored_on_noise():
    task = np.array([0] * 3 + [1] * 40)
    r = np.random.default_rng(9)
    x = r.normal(size=task.size)
    out = within_task_spearman(x, 2 * x, task, min_n=8)
    assert set(out["per_task"]) == {1}, out["per_task"]
    assert out["n_tasks"] == 1


if __name__ == "__main__":
    for name, fn in sorted(list(globals().items())):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("all task_match tests passed")
