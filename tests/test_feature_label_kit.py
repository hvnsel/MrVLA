"""Tests for the hand-labelling kit.

Run directly:
    python tests/test_feature_label_kit.py
"""

from __future__ import annotations

import csv
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mrvla.feature_label_kit import (  # noqa: E402
    FrameSource,
    cohens_kappa,
    episode_peaks,
    episode_trace,
    fit_logistic,
    loo_cv_accuracy,
    read_labels,
    render_card,
    select_features,
    top_episodes,
    write_label_sheet,
)
from mrvla.generality_classifier import sigmoid  # noqa: E402


# ---------------------------------------------------------------------------
# Feature selection
# ---------------------------------------------------------------------------
def test_stratified_selection_spans_the_score_range():
    rng = np.random.default_rng(0)
    prob = rng.uniform(0, 1, 500)
    prob[:490] = rng.uniform(0, 0.05, 490)      # realistic: mostly memorized
    active = np.ones(500, dtype=bool)

    strat = select_features(prob, active, 40, "stratified", seed=1)
    rand = select_features(prob, active, 40, "random", seed=1)
    # stratification must surface the sparse high-probability tail that
    # random sampling almost entirely misses
    n_hi_strat = int((prob[strat] > 0.2).sum())
    n_hi_rand = int((prob[rand] > 0.2).sum())
    assert n_hi_strat > n_hi_rand, (n_hi_strat, n_hi_rand)
    assert n_hi_strat >= 3
    assert len(strat) == 40
    assert len(np.unique(strat)) == 40


def test_selection_excludes_dead_features():
    prob = np.linspace(0, 1, 100)
    active = np.zeros(100, dtype=bool)
    active[:20] = True
    sel = select_features(prob, active, 10, "stratified", seed=0)
    assert set(sel).issubset(set(range(20)))


def test_selection_returns_all_when_fewer_than_requested():
    prob = np.linspace(0, 1, 10)
    active = np.ones(10, dtype=bool)
    sel = select_features(prob, active, 50, "stratified", seed=0)
    assert sel.tolist() == list(range(10))


# ---------------------------------------------------------------------------
# Per-feature evidence
# ---------------------------------------------------------------------------
def _toy():
    # 3 episodes of length 4; feature peaks highest in ep1
    zj = np.array([0.1, 0.2, 0.0, 0.0,
                   0.0, 0.9, 0.4, 0.0,
                   0.3, 0.0, 0.0, 0.0], dtype=np.float32)
    episode = np.array([0] * 4 + [1] * 4 + [2] * 4)
    timestep = np.array(list(range(4)) * 3)
    return zj, episode, timestep


def test_episode_peaks_and_top_episodes():
    zj, episode, timestep = _toy()
    ids, peaks = episode_peaks(zj, episode)
    assert ids.tolist() == [0, 1, 2]
    assert np.allclose(peaks, [0.2, 0.9, 0.3])
    top, tp = top_episodes(zj, episode, k=2)
    assert top.tolist() == [1, 2]
    assert np.allclose(tp, [0.9, 0.3])


def test_episode_trace_is_time_ordered():
    zj, episode, timestep = _toy()
    rng = np.random.default_rng(3)
    perm = rng.permutation(len(zj))
    ts, vals = episode_trace(zj[perm], episode[perm], timestep[perm], 1)
    assert ts.tolist() == [0, 1, 2, 3]
    assert np.allclose(vals, [0.0, 0.9, 0.4, 0.0])


# ---------------------------------------------------------------------------
# Logistic fit (Eq. 11)
# ---------------------------------------------------------------------------
def test_fit_recovers_known_coefficients():
    rng = np.random.default_rng(0)
    n = 4000
    X = rng.normal(size=(n, 4))
    true = np.array([-0.5, 1.5, -1.0, 0.8, 0.3])
    p = sigmoid(true[0] + X @ true[1:])
    y = (rng.uniform(size=n) < p).astype(float)
    w = fit_logistic(X, y, l2=1e-6)
    assert np.allclose(w, true, atol=0.15), w


def test_fit_is_stable_on_separable_data():
    """Hand-labelled sets are usually separable; the unpenalised MLE would
    diverge, so the ridge term must keep coefficients finite."""
    X = np.vstack([np.full((15, 4), -1.0), np.full((15, 4), 1.0)])
    y = np.array([0.0] * 15 + [1.0] * 15)
    w = fit_logistic(X, y, l2=1.0)
    assert np.all(np.isfinite(w))
    assert np.abs(w).max() < 50
    pred = sigmoid(np.hstack([np.ones((30, 1)), X]) @ w) >= 0.5
    assert (pred == (y == 1)).all()


def test_loo_cv_perfect_and_chance():
    X = np.vstack([np.full((10, 4), -1.0), np.full((10, 4), 1.0)])
    y = np.array([0.0] * 10 + [1.0] * 10)
    assert loo_cv_accuracy(X, y) == 1.0

    rng = np.random.default_rng(5)
    Xr = rng.normal(size=(24, 4))
    yr = np.array([0.0, 1.0] * 12)
    assert loo_cv_accuracy(Xr, yr) < 0.8


# ---------------------------------------------------------------------------
# Agreement
# ---------------------------------------------------------------------------
def test_cohens_kappa():
    a = ["general"] * 5 + ["memorized"] * 5
    assert np.isclose(cohens_kappa(a, a), 1.0)
    # complete disagreement on a balanced set
    b = ["memorized"] * 5 + ["general"] * 5
    assert cohens_kappa(a, b) < 0
    # chance-level agreement -> kappa near 0
    rng = np.random.default_rng(0)
    x = list(rng.choice(["general", "memorized"], 400))
    y = list(rng.choice(["general", "memorized"], 400))
    assert abs(cohens_kappa(x, y)) < 0.15


# ---------------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------------
def test_read_labels_filters_unclear():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "l.csv")
        with open(p, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["feature_id", "label", "confidence", "notes"])
            w.writerow([1, "general", "high", ""])
            w.writerow([2, "memorized", "low", "sustained"])
            w.writerow([3, "unclear", "low", ""])
            w.writerow([4, "", "", ""])
        got = read_labels(p)
        assert got == {1: "general", 2: "memorized"}


def test_render_card_and_sheet_without_frames():
    zj, episode, timestep = _toy()
    with tempfile.TemporaryDirectory() as d:
        out = os.path.join(d, "feat_0001.png")
        render_card(1, zj, episode, timestep, FrameSource(None), out,
                    k_episodes=3)
        assert os.path.getsize(out) > 1000

        sheet = os.path.join(d, "sheet.html")
        write_label_sheet([1, 2], ["feat_0001.png", "feat_0002.png"], sheet, 8)
        html = open(sheet, encoding="utf-8").read()
        for token in ('data-fid="1"', 'value="general"', 'value="memorized"',
                      'value="unclear"', "Download CSV"):
            assert token in html, token
        # the card must not leak any computed statistic
        for banned in ("coverage", "mean_onsets", "rel_run_length",
                       "P(general)", "prob_general"):
            assert banned not in html, banned


# ---------------------------------------------------------------------------
def main():
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f)]
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
