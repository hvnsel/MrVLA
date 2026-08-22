"""Pins the reliance analysis: signal invariances, window semantics, and the scorer.

Three tests carry the weight, all of them guarding failures that produce a plausible wrong
number rather than an error:

  * `test_mu_t_is_invariant_to_l2_and_r` -- mu_t's whole claim to being frozen-r-free is that
    it is a ratio whose numerator and denominator carry the same (l2/r) prefactor. If a
    refactor ever puts the prefactor on only one side, mu_t silently becomes a magnitude
    measure wearing a ratio's name.
  * `test_windows_are_counted_in_timesteps_not_rows` -- a timestep is SEVEN rows, one per
    action slot. A window counted in rows covers a seventh of the intended episode prefix,
    which would quietly turn "first 20 steps" into "first 3" and make the pre-divergence
    curve meaningless.
  * `test_auroc_below_half_is_a_real_effect_not_a_null` -- an AUROC of 0.2 means the signal
    predicts SUCCESS. Reporting it as "no effect" would discard a finding.

Run directly:
    python tests/test_reliance.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mrvla.prior_gates import prior_vectors  # noqa: E402
from mrvla.readout import signature_matrix  # noqa: E402
from mrvla.reliance import (  # noqa: E402
    aggregate_episodes, auroc, auroc_boot, length_diagnostics, reliance_signals,
    sufficiency, through_origin_slope,
)

D, F, NB = 24, 40, 16


def _world(n=200, seed=0):
    """A synthetic head + dictionary + n decisions, self-consistent by construction."""
    r = np.random.default_rng(seed)
    W_U = r.normal(size=(NB, D))
    g = np.abs(r.normal(size=D)) + 0.5
    b_pre = r.normal(size=D) * 0.1
    W_dec = r.normal(size=(F, D))
    W_dec /= np.linalg.norm(W_dec, axis=1, keepdims=True)
    A, B = prior_vectors(W_U, g, b_pre)
    S = signature_matrix(W_dec, g, W_U, center=True)

    z = np.zeros((n, F))
    idx = r.integers(0, F, size=(n, 6))
    np.put_along_axis(z, idx, r.exponential(size=(n, 6)), axis=1)
    l2 = np.abs(r.normal(2.0, 0.4, size=n)) + 0.2
    mu = r.normal(0.1, 0.3, size=n)
    rr = np.abs(r.normal(3.0, 0.5, size=n)) + 1.0
    h = l2[:, None] * (z @ W_dec) + mu[:, None] + b_pre[None, :]
    L = (h * g[None, :]) @ W_U.T
    rows = np.argmax(L, axis=1)
    low = np.zeros(F, bool)
    low[: F // 2] = True
    return dict(z=z, l2=l2, mu=mu, r=rr, S=S, A=A, B=B, L=L, rows=rows, low_mask=low)


# ---------------------------------------------------------------------------
# the canary
# ---------------------------------------------------------------------------
def test_sufficiency_slopes_sum_to_one():
    """feat + bias + err is exactly `true`, so the three slopes are an identity check."""
    w = _world(seed=1)
    s = reliance_signals(**w)
    out = sufficiency(s["true"], s["feat"], s["bias"])
    assert abs(out["sums_to_one"] - 1.0) < 1e-9, out


def test_sufficiency_recovers_a_planted_split():
    true = np.array([1.0, 2.0, 3.0, 4.0])
    feat, bias = 0.7 * true, 0.3 * true
    out = sufficiency(true, feat, bias)
    assert abs(out["features"] - 0.7) < 1e-12
    assert abs(out["bias"] - 0.3) < 1e-12
    assert abs(out["error"]) < 1e-12
    assert out["pass"]


def test_sufficiency_fails_below_the_floor():
    """If the decomposition does not transfer to rollout states, this must refuse."""
    true = np.array([1.0, 2.0, 3.0, 4.0])
    out = sufficiency(true, 0.3 * true, 0.2 * true)      # 0.5, well under the 0.80 bar
    assert not out["pass"], out


def test_through_origin_slope_handles_an_all_zero_target():
    assert not np.isfinite(through_origin_slope(np.zeros(5), np.ones(5)))


# ---------------------------------------------------------------------------
# signal invariances
# ---------------------------------------------------------------------------
def test_mu_t_is_invariant_to_l2_and_r():
    """mu_t is a ratio; the (l2/r) prefactor must cancel exactly on both sides."""
    w = _world(seed=2)
    base = reliance_signals(**w)["mu_t"]
    r2 = np.random.default_rng(9)
    w2 = dict(w, l2=w["l2"] * (1 + r2.random(w["l2"].size)),
              r=w["r"] * (1 + r2.random(w["r"].size)))
    assert np.allclose(base, reliance_signals(**w2)["mu_t"], equal_nan=True, atol=1e-12)


def test_mu_t_is_bounded_and_hits_both_ends():
    w = _world(seed=3)
    mt = reliance_signals(**w)["mu_t"]
    fin = np.isfinite(mt)
    assert fin.any() and mt[fin].min() >= 0.0 and mt[fin].max() <= 1.0
    all_low = reliance_signals(**dict(w, low_mask=np.ones(F, bool)))["mu_t"]
    assert np.allclose(all_low[np.isfinite(all_low)], 1.0)
    none_low = reliance_signals(**dict(w, low_mask=np.zeros(F, bool)))["mu_t"]
    assert np.allclose(none_low[np.isfinite(none_low)], 0.0)


def test_share_is_bounded():
    s = reliance_signals(**_world(seed=4))["share"]
    fin = np.isfinite(s)
    assert s[fin].min() >= 0.0 and s[fin].max() <= 1.0


def test_margin_is_the_top2_gap_and_non_negative():
    w = _world(seed=5)
    m = reliance_signals(**w)["margin"]
    L = w["L"]
    srt = np.sort(L, axis=1)
    assert np.allclose(m, srt[:, -1] - srt[:, -2])
    assert (m >= 0).all()


# ---------------------------------------------------------------------------
# episode aggregation
# ---------------------------------------------------------------------------
def test_windows_are_counted_in_timesteps_not_rows():
    """A timestep is 7 rows. `first5` must average 35 rows, not 5."""
    ts = np.repeat(np.arange(20), 7)
    ep = np.zeros(ts.size, int)
    sc = np.ones(ts.size, int)
    vals = ts.astype(float)                       # value == timestep, so the mean is checkable
    out = aggregate_episodes(vals, ep, ts, sc, windows=(5,))
    assert np.isclose(out["first5"][0], np.arange(5).mean()), out["first5"]
    assert np.isclose(out["mean"][0], np.arange(20).mean())
    assert np.isclose(out["max"][0], 19.0)


def test_aggregate_carries_the_episode_outcome():
    ep = np.array([0, 0, 1, 1, 2, 2])
    ts = np.array([0, 1, 0, 1, 0, 1])
    out = aggregate_episodes(np.arange(6.0), ep, ts, np.array([1, 1, 0, 0, 1, 1]))
    assert list(out["episodes"]) == [0, 1, 2]
    assert list(out["success"]) == [1, 0, 1]


def test_episode_shorter_than_the_window_is_excluded_not_partially_counted():
    """Changed deliberately. This used to assert that a short episode "contributes what it
    has", which is the duration leak: the column then mixes 3-step and 50-step averages and
    the difference tracks the outcome. Under require_full it is NaN and dropped."""
    ep = np.zeros(3, int)
    out = aggregate_episodes(np.ones(3), ep, np.array([0, 1, 2]), np.ones(3, int),
                             windows=(2, 50))
    assert np.isfinite(out["first2"][0])          # 3 steps can fill a 2-step window
    assert np.isnan(out["first50"][0])            # 3 steps cannot fill a 50-step window
    loose = aggregate_episodes(np.ones(3), ep, np.array([0, 1, 2]), np.ones(3, int),
                               windows=(50,), require_full=False)
    assert np.isfinite(loose["first50"][0])       # the old behaviour, still reachable

    out2 = aggregate_episodes(np.array([np.nan] * 3), ep, np.array([0, 1, 2]),
                              np.ones(3, int), windows=(2,))
    assert np.isnan(out2["first2"][0])            # all-NaN contributes nothing, not zero


# ---------------------------------------------------------------------------
# duration -- the confound that cannot be partialled out
# ---------------------------------------------------------------------------
def test_partial_windows_leak_duration_unless_full_is_required():
    """The reason require_full defaults True.

    Successes are short and failures run to the cap, so with require_full=False a short
    success contributes fewer steps to first50 than every failure does -- the column then
    encodes duration, which is the thing it exists to exclude. Here the signal is pure noise
    with a per-timestep drift, so ANY duration leak shows up as discrimination.
    """
    r = np.random.default_rng(7)
    ep, ts, sc, vals = [], [], [], []
    for e in range(300):
        fail = e % 4 == 0
        n = 100 if fail else int(r.integers(15, 40))     # failures long, successes short
        for t in range(n):
            for _ in range(7):
                ep.append(e); ts.append(t); sc.append(0 if fail else 1)
                vals.append(0.01 * t + r.normal(0, 0.05))   # drifts with time, nothing else
    ep, ts, sc, vals = map(np.asarray, (ep, ts, sc, vals))

    leaky = aggregate_episodes(vals, ep, ts, sc, windows=(50,), require_full=False)
    tight = aggregate_episodes(vals, ep, ts, sc, windows=(50,), require_full=True)
    a_leak = auroc(leaky["first50"], 1 - leaky["success"])
    a_tight = auroc(tight["first50"], 1 - tight["success"])
    assert abs(a_leak - 0.5) > 0.2, a_leak          # duration leaks straight through
    assert np.isnan(a_tight), a_tight               # only failures fill a 50-step window


def test_length_is_reported_per_episode():
    ep = np.repeat([0, 1], [7 * 3, 7 * 5])
    ts = np.concatenate([np.repeat(np.arange(3), 7), np.repeat(np.arange(5), 7)])
    out = aggregate_episodes(np.ones(ep.size), ep, ts, np.ones(ep.size, int), windows=())
    assert list(out["length"]) == [3, 5], out["length"]


def test_length_diagnostics_flags_a_disjoint_duration_split():
    """When done fires only on success, failures all hit the cap and the distributions
    barely overlap -- which is what makes a length-matched comparison unavailable."""
    fail_len = np.full(50, 300.0)
    ok_len = np.random.default_rng(0).integers(60, 200, size=150).astype(float)
    d = length_diagnostics(np.concatenate([fail_len, ok_len]),
                           np.concatenate([np.ones(50, int), np.zeros(150, int)]))
    assert d["auroc"] > 0.99, d["auroc"]
    assert d["overlap_frac_success_ge_min_failure"] == 0.0, d


def test_length_diagnostics_reports_real_overlap_when_it_exists():
    r = np.random.default_rng(1)
    d = length_diagnostics(np.concatenate([r.integers(50, 300, 100).astype(float),
                                           r.integers(50, 300, 100).astype(float)]),
                           np.concatenate([np.ones(100, int), np.zeros(100, int)]))
    assert 0.3 < d["auroc"] < 0.7, d["auroc"]
    assert d["overlap_frac_success_ge_min_failure"] > 0.5, d


# ---------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------
def test_auroc_perfect_and_reversed():
    s = np.array([1.0, 2, 3, 4, 5, 6])
    assert np.isclose(auroc(s, np.array([0, 0, 0, 1, 1, 1])), 1.0)
    assert np.isclose(auroc(s, np.array([1, 1, 1, 0, 0, 0])), 0.0)


def test_auroc_below_half_is_a_real_effect_not_a_null():
    """0.2 means the signal predicts SUCCESS. That is a finding with the opposite sign."""
    s = np.array([5.0, 4, 3, 2, 1])
    a = auroc(s, np.array([0, 0, 1, 1, 1]))
    assert a < 0.5 and np.isfinite(a), a


def test_auroc_is_half_on_constant_scores():
    assert np.isclose(auroc(np.ones(10), np.array([0, 1] * 5)), 0.5)


def test_auroc_nan_when_one_class_is_missing():
    """Demo replay is exactly this case: success is constant, so nothing is scoreable."""
    assert np.isnan(auroc(np.arange(5.0), np.ones(5, int)))
    assert np.isnan(auroc(np.arange(5.0), np.zeros(5, int)))


def test_auroc_averages_ties():
    a = auroc(np.array([1.0, 1.0, 2.0, 2.0]), np.array([0, 1, 0, 1]))
    assert np.isclose(a, 0.5), a


def test_auroc_boot_interval_contains_the_point_and_widens_when_small():
    r = np.random.default_rng(0)
    y = (r.random(400) < 0.3).astype(int)
    s = y + 0.6 * r.normal(size=400)
    big = auroc_boot(s, y, n_boot=400, seed=0)
    assert big["lo"] <= big["auroc"] <= big["hi"]
    small = auroc_boot(s[:40], y[:40], n_boot=400, seed=0)
    assert (small["hi"] - small["lo"]) > (big["hi"] - big["lo"])
    assert small["n_fail"] == int(y[:40].sum())


# ---------------------------------------------------------------------------
# composition
# ---------------------------------------------------------------------------
def test_pipeline_finds_a_planted_effect_and_not_a_null_one():
    """signals -> aggregate_episodes -> auroc_boot, end to end, on both branches.

    The unit tests pin each stage; this pins that they compose, which is where an index
    misalignment between per-row signals and per-episode labels would show up. A null
    signal whose interval excludes 0.5 would mean the episode/label join is wrong.
    """
    r = np.random.default_rng(0)
    n_ep, T, slots = 200, 60, 7
    ep = np.repeat(np.arange(n_ep), T * slots)
    ts = np.tile(np.repeat(np.arange(T), slots), n_ep)
    fail = r.random(n_ep) < 0.25
    succ = np.repeat((~fail).astype(int), T * slots)

    planted = np.repeat(fail.astype(float) * 0.4, T * slots) + 0.5 * r.normal(size=ep.size)
    agg = aggregate_episodes(planted, ep, ts, succ)
    hit = auroc_boot(agg["mean"], 1 - agg["success"], n_boot=300, seed=0)
    assert hit["auroc"] > 0.7 and hit["lo"] > 0.5, hit
    assert hit["n"] == n_ep and hit["n_fail"] == int(fail.sum())

    agg0 = aggregate_episodes(r.normal(size=ep.size), ep, ts, succ)
    miss = auroc_boot(agg0["mean"], 1 - agg0["success"], n_boot=300, seed=0)
    assert miss["lo"] < 0.5 < miss["hi"], miss


if __name__ == "__main__":
    for name, fn in sorted(list(globals().items())):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("all reliance tests passed")
