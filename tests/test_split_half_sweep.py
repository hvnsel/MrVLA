"""Tests for split_half_sweep.py -- the check on whether Spearman-Brown applies at all.

The diagnostic is only worth running if it can tell a well-specified case from a misspecified
one, so both are constructed and both are asserted:

  * a fixture obeying classical test theory -- a fixed true score per feature plus independent
    per-task noise -- must yield a FLAT implied per-task reliability across split sizes, since
    that is exactly the one-parameter model Spearman-Brown assumes;

  * a fixture where breadth is genuinely TASK-SET-RELATIVE -- different task subsets induce
    genuinely different orderings, with no fixed true score to converge on -- must NOT.

If the first failed, every verdict would be a false alarm. If the second failed, the diagnostic
would rubber-stamp the exact situation it exists to detect.

Run directly:
    python tests/test_split_half_sweep.py
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from split_half_sweep import (  # noqa: E402
    analyse, calibration_matrix, implied_r1, sweep_one_size,
)
from split_half_breadth import spearman_brown  # noqa: E402


def _args(**kw):
    d = dict(n_splits=150, tol=0.05, sd_ratio_tol=1.5, seed=0)
    d.update(kw)
    return argparse.Namespace(**d)


class _Z(dict):
    pass


def _wrap(C):
    return _Z(C=C, base_rate=np.random.default_rng(0).random(C.shape[1]),
              is_active=np.ones(C.shape[1], bool))


def _classical(G=10, F=1200, seed=0):
    """Obeys classical test theory: a fixed true breadth per feature, independent per-task draws.

    Feature f is active on each task independently with probability p_f, so the participation
    ratio measured on any subset is an unbiased-ish estimate of the same fixed p_f, with
    sampling error that shrinks as more tasks are averaged. That is exactly the one-parameter
    structure Spearman-Brown assumes.

    NOTE the encoding. PR is SCALE-INVARIANT, so giving a feature a constant multiplier is
    invisible to it -- a true score has to live in HOW MANY TASKS the feature touches, not in
    how large it is. An earlier version of this fixture encoded latent breadth as a magnitude
    and produced rho = 0.005 at every split size, which passed the flatness test vacuously.
    """
    rng = np.random.default_rng(seed)
    p_f = rng.uniform(0.15, 1.0, F)                          # the fixed true score
    scale = np.exp(rng.standard_normal(F))
    fire = rng.random((G, F)) < p_f                          # independent per task
    return fire * scale * np.exp(0.3 * rng.standard_normal((G, F)))


def _task_set_relative(G=10, F=1200, seed=1):
    """No fixed true score: which features look broad depends on WHICH tasks you sample.

    Tasks fall into two regimes, and a feature's activity rate in regime A is independent of
    its rate in regime B. A half drawn mostly from one regime therefore measures a genuinely
    different quantity than a half drawn from the other, and averaging more tasks converges on
    a blend rather than on any feature-intrinsic value.
    """
    rng = np.random.default_rng(seed)
    pa, pb = rng.uniform(0.1, 1.0, F), rng.uniform(0.1, 1.0, F)   # uncorrelated by regime
    scale = np.exp(rng.standard_normal(F))
    C = np.zeros((G, F))
    for g in range(G):
        p = pa if g < G // 2 else pb
        C[g] = (rng.random(F) < p) * scale * np.exp(0.3 * rng.standard_normal(F))
    return C


def test_implied_r1_inverts_spearman_brown_exactly():
    for r1 in (0.05, 0.2, 0.5, 0.8):
        for L in (2, 3, 5, 10):
            r_L = L * r1 / (1 + (L - 1) * r1)
            assert abs(implied_r1(r_L, L) - r1) < 1e-12, (r1, L)
    assert np.isnan(implied_r1(0.0, 5))
    assert np.isnan(implied_r1(-0.3, 5))


def test_spearman_brown_and_its_inverse_round_trip():
    for rho in (0.1, 0.3, 0.6):
        assert abs(spearman_brown(rho, 2.0) - 2 * rho / (1 + rho)) < 1e-12


def test_spearman_brown_drifts_even_under_a_perfectly_classical_model():
    """The finding that forced this script's design. A fixed true breadth per feature plus
    independent per-task sampling is exactly the model Spearman-Brown assumes -- and the implied
    per-task reliability STILL climbs with split size, because the participation ratio's ceiling
    moves with it. So drift cannot be read against zero, and a fixed threshold on it is
    meaningless. This is why the script calibrates instead."""
    res = analyse("classical", _wrap(_classical()), _args())
    assert res["r1_drift"] > 0.05, res["r1_drift"]


def test_calibration_absorbs_the_ceiling_artefact():
    """On classical data the calibration reproduces the real drift, so the excess is ~0."""
    res = analyse("classical", _wrap(_classical()), _args())
    assert abs(res["excess_drift"]) < 0.05, (res["excess_drift"], res["r1_drift"],
                                             res["r1_drift_calibration"])


def test_drift_alone_cannot_answer_the_true_score_question():
    """Why the second axis exists. Random halves mix any underlying task structure and wash it
    out, so a task-set-relative matrix drifts no more than a classical one -- both excesses sit
    near zero. Reading the true-score question off the drift would return the wrong answer."""
    a = analyse("classical", _wrap(_classical()), _args())
    b = analyse("relative", _wrap(_task_set_relative()), _args())
    assert abs(a["excess_drift"]) < 0.05 and abs(b["excess_drift"]) < 0.05, (
        a["excess_drift"], b["excess_drift"])


def test_rho_spread_ratio_separates_a_fixed_true_score_from_a_task_set_relative_one():
    """The axis that works. A fixed property measured noisily makes every random split equally
    informative, so rho varies across splits about as much as the calibration's does. A
    task-set-relative property makes regime-matched splits agree far better than regime-crossed
    ones, inflating the spread."""
    a = analyse("classical", _wrap(_classical()), _args())
    b = analyse("relative", _wrap(_task_set_relative()), _args())
    assert a["rho_sd_ratio"] < 1.5, a["rho_sd_ratio"]
    assert b["rho_sd_ratio"] > 2.0, b["rho_sd_ratio"]
    assert a["verdict"].startswith("consistent with a fixed true score"), a["verdict"]
    assert b["verdict"].startswith("TASK-SET-RELATIVE"), b["verdict"]


def test_calibration_matrix_preserves_activity_and_encodes_breadth_as_task_count():
    """The calibration must match the real matrix on the axis PR can actually see. PR is
    scale-invariant, so a per-feature multiplier is invisible to it and the true score has to
    live in how many tasks a feature touches."""
    C = _classical()
    Ccal = calibration_matrix(C, 0)
    assert Ccal.shape == C.shape
    real_p, cal_p = (C > 0).mean(axis=0), (Ccal > 0).mean(axis=0)
    assert np.corrcoef(real_p, cal_p)[0, 1] > 0.8, np.corrcoef(real_p, cal_p)[0, 1]
    assert abs(real_p.mean() - cal_p.mean()) < 0.05


def test_shuffle_floor_sits_at_zero_at_every_size():
    res = analyse("classical", _wrap(_classical()), _args())
    for e in res["curves"]["shuffle_floor"]:
        if e.get("n_splits_ok"):
            assert abs(e["rho_median"]) < 0.08, e


def test_fixed_population_is_split_independent_and_reported():
    C = _classical()
    C[:, :50] = 0.0
    C[0, :50] = 1.0                       # 50 features active in only ONE task
    res = analyse("pop", _wrap(C), _args())
    assert res["n_fixed_population"] <= C.shape[1] - 50
    for e in res["curves"]["raw_fixed_pop"]:
        if e.get("n_splits_ok"):
            assert e["n_features_median"] <= res["n_fixed_population"]


def test_raw_curve_is_reported_alongside_adjusted():
    """The adjusted curve carries a base-rate leak; the leak-free raw curve must also be there,
    and the verdict must be taken from it."""
    res = analyse("classical", _wrap(_classical()), _args())
    assert {"raw", "adjusted", "raw_fixed_pop", "adjusted_fixed_pop",
            "shuffle_floor"} <= set(res["curves"])
    assert all(e.get("n_splits_ok", 0) > 0 for e in res["curves"]["raw"])


def test_sweep_handles_a_degenerate_half():
    """A size at which almost nothing is usable must degrade to a skipped entry, not a crash."""
    C = np.zeros((10, 300)); C[0] = 1.0
    out = sweep_one_size(C, np.random.default_rng(0).random(300), np.ones(300, bool),
                         2, 10, np.random.default_rng(0), True, None, False)
    assert out["size"] == 2 and out.get("n_splits_ok", 0) == 0


if __name__ == "__main__":
    for name, fn in sorted(list(globals().items())):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("all split_half_sweep tests passed")
