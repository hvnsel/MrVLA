"""Pins the coalition-dynamics signals and the LOTO probe.

The load-bearing tests:

  * `test_returns_fires_on_a_planted_period_2_oscillation` and its negative twin -- the
    period-2 statistic IS the hypothesis, so if it cannot separate an alternating coalition
    from a drifting one, nothing downstream means anything.
  * `test_slots_do_not_contaminate_each_other` -- comparisons are per decode slot. Pooling
    all seven would mix the gripper (bias-driven, per P5b) into the arm channels and blur
    exactly the structure being looked for, silently.
  * `test_probe_null_is_centred_on_chance_with_random_labels` -- the probe has 2048 features
    and 500 episodes. Without LOTO and a within-task permutation null it will report a
    confident number on noise, which is the whole reason both exist.

Run directly:
    python tests/test_dynamics.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mrvla.dynamics import (  # noqa: E402
    action_dynamics, auroc_curve, coalition_dynamics, jaccard_at_lag, max_unbiased_t,
    prefix_means, probe_loto, ridge_dual_loto, topk_sets, weight_breadth_skew,
)

F = 64


def _seq(setlist, slot=0, episode=0):
    """Build (sets, episode, slot, timestep) from a list of feature-index lists."""
    sets = np.array([sorted(s) for s in setlist], dtype=np.int32)
    n = len(setlist)
    return (sets, np.full(n, episode), np.full(n, slot), np.arange(n))


# ---------------------------------------------------------------------------
# coalitions
# ---------------------------------------------------------------------------
def test_topk_sets_picks_the_largest():
    s = np.array([[0.0, 5.0, 1.0, 9.0], [7.0, 0.0, 0.0, 2.0]])
    assert np.array_equal(topk_sets(s, 2), np.array([[1, 3], [0, 3]], dtype=np.int32))


def test_jaccard_identical_and_disjoint():
    sets, e, s, t = _seq([[0, 1, 2], [0, 1, 2], [7, 8, 9]])
    j = jaccard_at_lag(sets, e, s, t, 1)
    assert np.isnan(j[0])                      # no partner at the episode start
    assert np.isclose(j[1], 1.0)               # identical
    assert np.isclose(j[2], 0.0)               # disjoint


def test_jaccard_partial_overlap_is_exact():
    sets, e, s, t = _seq([[0, 1, 2, 3], [2, 3, 4, 5]])
    assert np.isclose(jaccard_at_lag(sets, e, s, t, 1)[1], 2 / 6)


def test_jaccard_respects_episode_boundaries():
    sets = np.array([[0, 1], [0, 1]], dtype=np.int32)
    j = jaccard_at_lag(sets, np.array([0, 1]), np.zeros(2, int), np.array([0, 0]), 1)
    assert np.isnan(j).all(), j                # different episodes are never partners


def test_jaccard_respects_timestep_gaps():
    sets = np.array([[0, 1], [0, 1]], dtype=np.int32)
    j = jaccard_at_lag(sets, np.zeros(2, int), np.zeros(2, int), np.array([0, 5]), 1)
    assert np.isnan(j).all(), j                # a gap is not a lag-1 pair


def test_slots_do_not_contaminate_each_other():
    """Slot 0 alternates, slot 1 is constant. Each must be scored on its own history."""
    sets = np.array([[0, 1], [8, 9], [0, 1],      # slot 0: A B A
                     [4, 5], [4, 5], [4, 5]],     # slot 1: constant
                    dtype=np.int32)
    e = np.zeros(6, int)
    s = np.array([0, 0, 0, 1, 1, 1])
    t = np.array([0, 1, 2, 0, 1, 2])
    d = coalition_dynamics(sets, e, s, t)
    assert np.isclose(d["returns"][2], 1.0)       # slot 0 returned to A
    assert np.isclose(d["returns"][5], 0.0)       # slot 1 never left
    assert np.isclose(d["churn"][5], 0.0)


def test_returns_fires_on_a_planted_period_2_oscillation():
    """A B A B A B -- J(t,t-2)=1 and J(t,t-1)=0, so every scoreable step is a return."""
    A, B = [0, 1, 2], [10, 11, 12]
    sets, e, s, t = _seq([A, B, A, B, A, B])
    r = coalition_dynamics(sets, e, s, t)["returns"]
    assert np.isnan(r[0]) and np.isnan(r[1])      # need two steps of history
    assert np.allclose(r[2:], 1.0), r


def test_returns_stays_low_under_smooth_drift():
    """Each step swaps one member, so t-1 is always closer than t-2. No returns."""
    seq = [[0, 1, 2, 3], [1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6], [4, 5, 6, 7]]
    sets, e, s, t = _seq(seq)
    r = coalition_dynamics(sets, e, s, t)["returns"]
    assert np.allclose(r[2:], 0.0), r


def test_churn_is_one_minus_jaccard():
    sets, e, s, t = _seq([[0, 1], [0, 1]])
    d = coalition_dynamics(sets, e, s, t)
    assert np.isclose(d["churn"][1], 0.0)
    assert np.isclose(d["j1"][1], 1.0)


# ---------------------------------------------------------------------------
# the action-space baseline
# ---------------------------------------------------------------------------
def test_action_dynamics_detects_an_oscillating_command():
    """The control that decides whether feature churn means anything on its own."""
    a = np.array([[1.0, 0, 0, 0, 0, 0, 0], [-1.0, 0, 0, 0, 0, 0, 0]] * 3)
    d = action_dynamics(a, np.zeros(6, int), np.arange(6))
    assert np.allclose(d["returns"][2:], 1.0), d["returns"]


def test_action_dynamics_is_quiet_on_a_steady_command():
    a = np.tile(np.array([[1.0, 0, 0, 0, 0, 0, 0]]), (6, 1))
    d = action_dynamics(a, np.zeros(6, int), np.arange(6))
    assert np.allclose(d["returns"][2:], 0.0), d["returns"]
    assert np.allclose(d["churn"][1:], 0.0, atol=1e-9)


# ---------------------------------------------------------------------------
# online scoring
# ---------------------------------------------------------------------------
def test_prefix_means_uses_only_timesteps_below_t():
    v = np.arange(10.0)
    _u, m, lens = prefix_means(v, np.zeros(10, int), np.arange(10), 4)
    assert np.isclose(m[0], np.arange(4).mean())
    assert lens[0] == 10


def test_max_unbiased_t_is_the_shortest_episode():
    assert max_unbiased_t(np.array([300, 72, 150]), np.array([1, 0, 0])) == 72


def test_auroc_curve_finds_a_planted_effect_and_reports_its_sample():
    r = np.random.default_rng(0)
    ep, ts, sc, val = [], [], [], []
    for e in range(200):
        fail = e % 4 == 0
        n = 60 if fail else 40
        for t in range(n):
            ep.append(e); ts.append(t); sc.append(0 if fail else 1)
            val.append((0.8 if fail else 0.0) + r.normal(0, 0.5))
    ep, ts, sc, val = map(np.asarray, (ep, ts, sc, val))
    rows = auroc_curve(val, ep, ts, sc, grid=[10, 20])
    assert all(x["auroc"] > 0.8 for x in rows), rows
    assert all(x["n_ok"] == 150 and x["n_fail"] == 50 for x in rows), rows


def test_auroc_curve_drops_episodes_that_cannot_fill_the_window():
    """Past the shortest episode the sample changes, and the row must say so."""
    ep = np.concatenate([np.zeros(50, int), np.ones(10, int)])
    ts = np.concatenate([np.arange(50), np.arange(10)])
    sc = np.concatenate([np.zeros(50, int), np.ones(10, int)])
    rows = auroc_curve(np.ones(60), ep, ts, sc, grid=[5, 30])
    assert rows[0]["n_ok"] + rows[0]["n_fail"] == 2
    assert rows[1]["n_ok"] + rows[1]["n_fail"] == 1        # the 10-step episode drops out


# ---------------------------------------------------------------------------
# the probe
# ---------------------------------------------------------------------------
def _probe_data(n_task=10, per=50, p=64, signal=0.0, seed=0):
    r = np.random.default_rng(seed)
    task = np.repeat(np.arange(n_task), per)
    y = (r.random(task.size) < 0.25).astype(int)
    X = r.normal(size=(task.size, p))
    X[:, 0] += signal * y                                  # one feature carries the label
    return X, y, task


def test_ridge_recovers_a_planted_direction_out_of_fold():
    X, y, task = _probe_data(signal=2.0, seed=1)
    s, w = ridge_dual_loto(X, y, task, lam=1.0)
    assert np.isfinite(s).all()
    assert np.argmax(np.abs(w)) == 0, np.argsort(-np.abs(w))[:3]


def test_loto_actually_holds_the_task_out():
    """Every test score must come from a fit that never saw that task."""
    X, y, task = _probe_data(seed=2)
    seen = []
    for gi in np.unique(task):
        s, _ = ridge_dual_loto(X, y, np.where(task == gi, gi, -1), lam=1.0)
        seen.append(np.isfinite(s[task == gi]).all())
    assert all(seen)


def test_probe_beats_its_null_when_signal_is_real():
    X, y, task = _probe_data(signal=2.5, seed=3)
    out = probe_loto(X, y, task, lam=1.0, n_perm=60, seed=0)
    assert out["auroc"] > out["null_p95"], out
    assert out["p_value"] < 0.05, out


def test_probe_null_is_centred_on_chance_with_random_labels():
    """2048 features on 500 episodes will fit noise. LOTO plus a within-task permutation
    null is what stops that from being reported as a result."""
    X, y, task = _probe_data(signal=0.0, seed=4)
    out = probe_loto(X, y, task, lam=1.0, n_perm=80, seed=0)
    assert abs(out["null_mean"] - 0.5) < 0.06, out["null_mean"]
    assert out["p_value"] > 0.05, out


def test_probe_null_cannot_be_beaten_by_task_base_rates():
    """Failure rate varies by task and nothing else. A null that permuted labels GLOBALLY
    would break that structure and hand the probe a fake win; permuting within task keeps
    each task's rate fixed, so there is nothing left to learn."""
    r = np.random.default_rng(5)
    task = np.repeat(np.arange(10), 50)
    rate = np.linspace(0.05, 0.6, 10)[task]
    y = (r.random(task.size) < rate).astype(int)
    X = r.normal(size=(task.size, 64))                      # carries no label information
    out = probe_loto(X, y, task, lam=1.0, n_perm=80, seed=0)
    assert out["p_value"] > 0.05, out


# ---------------------------------------------------------------------------
# the bridge back to Path A
# ---------------------------------------------------------------------------
def test_weight_breadth_skew_recovers_a_planted_skew():
    n = 400
    breadth = np.arange(n, dtype=float)
    w = np.zeros(n)
    w[:40] = 1.0                                            # weight on the LOWEST breadth
    out = weight_breadth_skew(w, breadth, top_n=40)
    assert out["rho"] < -0.3, out
    assert out["top_w_mean_breadth_percentile"] < 20, out
    assert out["top_w_frac_below_median_breadth"] == 1.0, out


def test_weight_breadth_skew_reports_the_opposite_sign_too():
    n = 400
    breadth = np.arange(n, dtype=float)
    w = np.zeros(n)
    w[-40:] = 1.0                                           # weight on the HIGHEST breadth
    out = weight_breadth_skew(w, breadth, top_n=40)
    assert out["rho"] > 0.3, out
    assert out["top_w_mean_breadth_percentile"] > 80, out


def test_weight_breadth_skew_is_flat_when_there_is_no_skew():
    r = np.random.default_rng(6)
    out = weight_breadth_skew(r.normal(size=500), r.normal(size=500), top_n=50)
    assert abs(out["rho"]) < 0.15, out


if __name__ == "__main__":
    for name, fn in sorted(list(globals().items())):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("all dynamics tests passed")
