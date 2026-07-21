"""Tests for the step-3 confound audit (mrvla/confound_audit.py).

Run directly (no pytest needed):
    python tests/test_confound_audit.py
or via pytest:
    pytest tests/test_confound_audit.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mrvla.confound_audit import (  # noqa: E402
    audit_layer,
    build_design,
    episode_table,
    eta_squared,
    icc_two_halves,
    ols_fit,
    pearson,
    rankdata,
    spearman,
    spearman_brown,
)


# ---------------------------------------------------------------------------
# Stats helpers
# ---------------------------------------------------------------------------
def test_rankdata_ties():
    r = rankdata(np.array([10.0, 20.0, 20.0, 30.0]))
    assert np.allclose(r, [1.0, 2.5, 2.5, 4.0])


def test_pearson_spearman():
    x = np.array([1.0, 2.0, 3.0, 4.0])
    assert np.isclose(pearson(x, 2 * x + 1), 1.0)
    assert np.isclose(pearson(x, -x), -1.0)
    # spearman is 1 for any monotone transform
    assert np.isclose(spearman(x, x ** 3), 1.0)
    # constant input degrades to 0, not NaN
    assert pearson(x, np.ones(4)) == 0.0


def test_spearman_brown():
    assert np.isclose(spearman_brown(0.5), 2 * 0.5 / 1.5)
    assert spearman_brown(0.0) == 0.0
    assert np.isclose(spearman_brown(1.0), 1.0)


def test_icc_two_halves():
    rng = np.random.default_rng(0)
    truth = rng.normal(size=300)
    # identical halves -> ICC ~ 1; independent noise -> ICC ~ 0
    assert icc_two_halves(truth, truth) > 0.99
    noise = rng.normal(size=300)
    assert abs(icc_two_halves(truth, noise)) < 0.15
    # halves = truth + noise -> ICC ~ var(truth)/(var(truth)+var(noise))
    a = truth + rng.normal(scale=1.0, size=300)
    b = truth + rng.normal(scale=1.0, size=300)
    assert 0.3 < icc_two_halves(a, b) < 0.7


def test_eta_squared():
    # groups fully explain y
    groups = np.array([0, 0, 1, 1, 2, 2])
    y = np.array([1.0, 1.0, 5.0, 5.0, 9.0, 9.0])
    assert np.isclose(eta_squared(y, groups), 1.0)
    # groups explain nothing (same mean per group)
    y2 = np.array([1.0, 3.0, 1.0, 3.0, 1.0, 3.0])
    assert np.isclose(eta_squared(y2, groups), 0.0)


def test_ols_residualizes_planted_confound():
    rng = np.random.default_rng(1)
    E = 400
    confound = rng.uniform(0, 1, E)
    signal = rng.normal(size=E)
    y = 3.0 * confound + 0.5 * signal
    X, names, blocks = build_design({"c": confound},
                                    np.zeros(E, dtype=np.int64))
    r2, _, resid, _ = ols_fit(X, y)
    # the confound part is fully removed; the residual IS the signal part
    assert abs(pearson(resid, confound)) < 1e-8
    assert pearson(resid, signal) > 0.99


def test_build_design_shapes():
    E = 12
    task = np.array([0] * 4 + [1] * 4 + [2] * 4)
    X, names, blocks = build_design(
        {"a": np.arange(E, dtype=float)}, task)
    # intercept + 1 continuous + (3-1) task dummies
    assert X.shape == (E, 4)
    assert names[0] == "intercept"
    assert set(blocks) == {"a", "task_id"}
    assert len(blocks["task_id"]) == 2


# ---------------------------------------------------------------------------
# End-to-end on synthetic episodes with planted structure
# ---------------------------------------------------------------------------
def _make_synthetic(planted: str):
    """Episodes whose score is driven by a planted cause.

    planted='confound': score is a pure function of episode length
    planted='signal'  : score is an episode-level trait independent of
                        length/task, stable across frames (split-half reliable)
    """
    rng = np.random.default_rng(7)
    E, F = 60, 6
    z_rows, ratio_rows, ep_col, ts_col, task_col = [], [], [], [], []
    for ep in range(E):
        T = int(rng.integers(40, 120))
        task = ep % 3
        if planted == "confound":
            base = T / 120.0                    # score := length, rescaled
        else:
            base = rng.uniform(0.2, 0.8)        # stable per-episode trait
        r = np.clip(base + rng.normal(scale=0.01, size=T), 0, 1)
        z = np.abs(rng.normal(size=(T, F))).astype(np.float32)
        z_rows.append(z)
        ratio_rows.append(r)
        ep_col += [ep] * T
        ts_col += list(range(T))
        task_col += [task] * T
    return (np.vstack(z_rows), np.concatenate(ratio_rows),
            np.array(ep_col), np.array(ts_col), np.array(task_col))


def test_audit_flags_planted_confound():
    z, ratio, episode, timestep, task = _make_synthetic("confound")
    tab = episode_table(z, ratio, episode, timestep, task, trim_frac=0.1)
    stats, resid = audit_layer(tab)
    # length explains (nearly) everything
    assert stats["univariate"]["T_full"]["r2"] > 0.9
    assert stats["ols"]["r2"] > 0.9
    assert stats["verdict"]["mostly_confound"]
    # nothing reliable left after regressing confounds out
    assert stats["reliability"]["residual_even_odd"]["r_spearman_brown"] < 0.5


def test_audit_passes_planted_signal():
    z, ratio, episode, timestep, task = _make_synthetic("signal")
    tab = episode_table(z, ratio, episode, timestep, task, trim_frac=0.1)
    stats, resid = audit_layer(tab)
    # confounds explain little; the trait survives residualization reliably
    assert stats["ols"]["r2"] < 0.5
    assert not stats["verdict"]["mostly_confound"]
    assert stats["reliability"]["raw_even_odd"]["r_spearman_brown"] > 0.9
    assert stats["reliability"]["residual_even_odd"]["r_spearman_brown"] > 0.9
    assert stats["verdict"]["reliable_residual"]
    # residual keeps the episode ordering of the true trait
    assert pearson(resid, tab["score"]) > 0.7


def test_episode_table_confounds():
    rng = np.random.default_rng(3)
    T, F = 50, 4
    z = np.abs(rng.normal(size=(T, F))).astype(np.float32)
    z[:, 2] = 0.0                               # dead feature lowers L0
    ratio = np.full(T, 0.4)
    episode = np.zeros(T, dtype=int)
    timestep = np.arange(T)
    task = np.full(T, 5)
    tab = episode_table(z, ratio, episode, timestep, task, trim_frac=0.0)
    assert tab["T_full"][0] == T
    assert tab["task"][0] == 5
    assert np.isclose(tab["score"][0], 0.4)
    assert np.isclose(tab["mean_l0"][0], (z > 0).sum(axis=1).mean())
    assert np.isclose(tab["mean_mass"][0], z.sum(axis=1).mean())
    # halves of a constant ratio equal the score
    for key in ("half_even", "half_odd", "half_first", "half_second"):
        assert np.isclose(tab[key][0], 0.4)


# ---------------------------------------------------------------------------
def main():
    tests = [(name, fn) for name, fn in sorted(globals().items())
             if name.startswith("test_") and callable(fn)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
        except AssertionError as e:
            failed += 1
            print(f"  FAIL  {name}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
