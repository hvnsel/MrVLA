"""Tests for causal_breadth.py -- the leave-one-task-out estimator with a causal target.

Four of these are the reason the experiment is designed the way it is, rather than decoration:

  * `test_task_misalignment_is_refused` -- the join between flip[F, G, S] and C[G, F] is
    positional. Both arrays have G rows, so a task dropped by run_attribution.py's invalid-token
    `continue` shifts every later index by one and NOTHING RAISES. The number would still be
    finite, still be plausible, and still be wrong.

  * `test_selecting_on_the_predictor_inflates_the_correlation` -- pins the size of the
    extreme-group inflation, which is the entire justification for running all 2048 features
    instead of pick_candidates' ~396.

  * `test_binomial_denominator_floor_fires_on_a_pure_ratio_artefact` -- a ratio target with
    unequal denominators can produce a positive partial when every feature is EQUALLY decisive.
    Neither of the two standard floors detects that. This test builds the artefact and confirms
    the floor refuses to certify it.

  * `test_pooling_is_a_ratio_of_sums_not_a_mean_of_ratios` -- these differ whenever slots have
    unequal denominators, which they always do.

Every fixture uses G != S. With G == S == 7 a transposed bincount index passes every assertion
in the file and the misalignment tests are worthless.

Run directly:
    python tests/test_causal_breadth.py
"""

from __future__ import annotations

import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from causal_breadth import (binomial_denominator_null, flip_rate, load_pair,  # noqa: E402
                            loto_base_rate, paired_column_shuffle)
from mrvla.attribution import participation_ratio  # noqa: E402
from mrvla.rankbasis import _folds, _folds_xy, loto_partial_target  # noqa: E402

G, S, F = 10, 7, 400          # G != S, deliberately


def _planted(seed=0, F=F, G=G, S=S, strength=1.0):
    """C with a spread of breadths, and a flip rate that genuinely tracks breadth.

    Decisiveness is built as a per-(feature, task) probability that rises with the feature's
    breadth, then realised as an actual binomial draw against a denominator -- so the target
    carries measurement noise like the real one, rather than being an algebraic function of the
    predictor (a mistake that has produced meaningless passes in this repo before).
    """
    rng = np.random.default_rng(seed)
    spread = rng.uniform(0.05, 1.0, F)                       # low = narrow, high = broad
    mass = rng.gamma(1.2, 1.0, F)
    C = np.stack([rng.dirichlet(np.full(G, s * 4)) for s in spread], axis=1) * mass * G
    base = np.clip(0.02 + 0.5 * mass / mass.max(), 0.005, 1.0)
    n_act = np.maximum(rng.poisson(np.broadcast_to((2000 * base)[:, None], (F, G))), 1)
    p = np.clip(0.05 + strength * 0.06 * (spread[:, None] - 0.5)
                + rng.normal(0, 0.004, (F, G)), 1e-4, 0.9)
    fl = rng.binomial(n_act, p)
    # spread the counts over slots so the [F, G, S] shape is exercised, not just [F, G]
    w = rng.dirichlet(np.full(S, 3.0), size=(F, G))
    n_gts = np.round(n_act[:, :, None] * w).astype(np.int64)
    fr = np.where(n_act[:, :, None] > 0, fl[:, :, None] / np.maximum(n_act[:, :, None], 1), 0.0)
    fl_gts = rng.binomial(np.maximum(n_gts, 0), np.clip(fr, 0, 1))
    return {"C": C, "base_rate": base, "n_gts": n_gts, "fl_gts": fl_gts, "spread": spread}


def _write_pair(td, fx, chan_tasks=None, attr_tasks=None, all_features=True):
    """Write a chan/attr npz pair in exactly the layout run_channel_attribution.py saves."""
    Fx, Gx, Sx = fx["n_gts"].shape
    chan_tasks = np.arange(Gx) if chan_tasks is None else np.asarray(chan_tasks)
    attr_tasks = np.arange(Gx) if attr_tasks is None else np.asarray(attr_tasks)
    cp = os.path.join(td, "layer_31_channels.npz")
    ap = os.path.join(td, "layer_31_attribution.npz")
    n_all = np.broadcast_to(fx["n_gts"].sum(axis=0), (Fx, Gx, Sx)).copy()
    np.savez_compressed(
        cp, task_ids=chan_tasks,
        flip_coded_n_active_gt=fx["n_gts"], flip_coded_flip_active_gt=fx["fl_gts"],
        flip_coded_n_active_trans_gt=(fx["n_gts"] // 2),
        flip_coded_flip_active_trans_gt=(fx["fl_gts"] // 2),
        flip_coded_n_gt=n_all)
    C = fx["C"][:attr_tasks.size] if attr_tasks.size < Gx else fx["C"]
    np.savez_compressed(ap, C=C, task_ids=attr_tasks, base_rate=fx["base_rate"],
                        PR=participation_ratio(fx["C"]), magnitude=fx["C"].sum(0))
    import json
    json.dump({"all_features": all_features, "coeff": "coded", "trans_mask": "per_channel",
               "n_candidates": Fx, "channel_names": [f"s{i}" for i in range(Sx)]},
              open(os.path.join(td, "summary.json"), "w"))
    return cp, ap


# ------------------------------------------------------------------ the join

def test_task_misalignment_is_refused():
    """A task present in the channel run and absent from the attribution run shifts every later
    row index by one. Both matrices still have plausible shapes, so only an explicit check
    catches it."""
    fx = _planted(seed=1)
    with tempfile.TemporaryDirectory() as td:
        cp, ap = _write_pair(td, fx, chan_tasks=np.arange(G),
                             attr_tasks=np.array([0, 1, 2, 4, 5, 6, 7, 8, 9]))
        raised = False
        try:
            load_pair(cp, ap)
        except SystemExit as e:
            raised = True
            assert "TASK MISALIGNMENT" in str(e), str(e)
        assert raised, "a task-id mismatch was accepted silently -- the join would be wrong"
        D = load_pair(cp, ap, allow_mismatch=True)
        assert D["n_tasks"] == 9
        assert D["C"].shape[0] == D["n_active"].shape[1] == 9


def test_a_channel_npz_without_the_task_axis_is_refused():
    """An npz from before the task axis existed must fail loudly, not fall back to a 2-D key."""
    with tempfile.TemporaryDirectory() as td:
        cp = os.path.join(td, "layer_31_channels.npz")
        ap = os.path.join(td, "layer_31_attribution.npz")
        np.savez_compressed(cp, task_ids=np.arange(G),
                            flip_coded_n_active=np.ones((F, S)),
                            flip_coded_flip_active=np.ones((F, S)))
        np.savez_compressed(ap, C=np.ones((G, F)), task_ids=np.arange(G),
                            base_rate=np.ones(F) * 0.1)
        try:
            load_pair(cp, ap)
            raise AssertionError("a task-axis-free npz was accepted")
        except SystemExit as e:
            assert "per-(feature, task, slot)" in str(e)


# ------------------------------------------------------------------ the target

def test_pooling_is_a_ratio_of_sums_not_a_mean_of_ratios():
    """With unequal per-slot denominators the two pooling rules disagree, and the mean of ratios
    gives a slot with four firings the same weight as one with four thousand."""
    n = np.zeros((3, 2, 4), dtype=np.int64)
    fl = np.zeros_like(n)
    n[0, 0] = [4, 4000, 0, 0]
    fl[0, 0] = [4, 40, 0, 0]                     # slot0 rate 1.0, slot1 rate 0.01
    R, den = flip_rate(fl.astype(float), n.astype(float))
    assert den[0, 0] == 4004
    assert abs(R[0, 0] - 44 / 4004) < 1e-12, R[0, 0]
    mean_of_ratios = np.mean([1.0, 0.01])
    assert abs(R[0, 0] - mean_of_ratios) > 0.4, "the two rules must actually differ here"


def test_zero_denominator_modes_differ_where_it_matters():
    """`drop` leaves a never-fired cell as NaN so the fold excludes it; `zero` calls it
    decisiveness 0, which re-imports the base-rate confound."""
    n = np.zeros((2, 2, 3), dtype=float)
    fl = np.zeros_like(n)
    n[1, 0] = [10, 10, 10]
    fl[1, 0] = [1, 1, 1]
    R_drop, _ = flip_rate(fl, n, zero_denominator="drop")
    R_zero, _ = flip_rate(fl, n, zero_denominator="zero")
    assert np.isnan(R_drop[0, 0]) and R_zero[0, 0] == 0.0
    assert abs(R_drop[0, 1] - 0.1) < 1e-12 and abs(R_zero[0, 1] - 0.1) < 1e-12


def test_loto_base_rate_excludes_the_held_out_task():
    """Row gi must be computable without task gi. If it is not, the control leaks the target."""
    F2, G2, S2 = 5, 4, 3
    n_act = np.zeros((F2, G2, S2))
    n_act[0, 0] = 900                       # feature 0 fires ONLY on task 0
    n_dec = np.full((G2, S2), 1000.0)
    br = loto_base_rate(n_act, n_dec)
    assert br[0, 0] == 0.0, "fold 0 still sees task 0's firings"
    assert br[1, 0] > 0.0, "folds that keep task 0 must see them"


# ------------------------------------------------------------------ the estimator

def test_target_folds_reproduce_the_shipped_folds_when_the_target_is_C():
    """`_folds_xy(C, C, ...)` must be the SAME estimator as `_folds(C, ...)`, tuple for tuple.
    Otherwise the causal number is produced by a lookalike and is not comparable to A3."""
    fx = _planted(seed=2)
    a = list(_folds(fx["C"], fx["base_rate"]))
    b = list(_folds_xy(fx["C"], fx["C"], fx["base_rate"]))
    assert len(a) == len(b) == G
    for (p1, m1, h1, r1), (p2, m2, h2, r2, extras, gi, msk) in zip(a, b):
        for u, v in ((p1, p2), (m1, m2), (h1, h2), (r1, r2)):
            assert np.array_equal(u, v)
        assert extras == []


def test_estimator_recovers_a_planted_relationship_and_is_null_when_shuffled():
    """Both error directions. A positive fixture must come out positive; the same data with the
    held-out target permuted must come out at zero, or the estimator itself is biased."""
    fx = _planted(seed=3, strength=1.0)
    R, den = flip_rate(fx["fl_gts"].astype(float), fx["n_gts"].astype(float))
    v = loto_partial_target(fx["C"], R, fx["base_rate"], "linear")
    assert v.mean() > 0.15, f"planted relationship not recovered: {v.mean():+.4f}"

    rng = np.random.default_rng(9)
    Rs = np.stack([rng.permutation(row) for row in R])
    vs = loto_partial_target(fx["C"], Rs, fx["base_rate"], "linear")
    assert abs(vs.mean()) < 0.06, f"estimator floor is not zero: {vs.mean():+.4f}"


def test_null_fixture_gives_no_partial():
    """No relationship planted -> the partial must sit at zero on all 400 features."""
    fx = _planted(seed=4, strength=0.0)
    R, _ = flip_rate(fx["fl_gts"].astype(float), fx["n_gts"].astype(float))
    v = loto_partial_target(fx["C"], R, fx["base_rate"], "linear")
    assert abs(v.mean()) < 0.06, f"false positive on a null fixture: {v.mean():+.4f}"


def test_selecting_on_the_predictor_inflates_the_correlation():
    """THE justification for --all-features. `pick_candidates` keeps the top and bottom of
    adjusted breadth; extreme-group sampling inflates |r| by construction, and the inflation
    grows as the selected fraction shrinks."""
    fx = _planted(seed=5, strength=0.6)
    R, _ = flip_rate(fx["fl_gts"].astype(float), fx["n_gts"].astype(float))
    full = loto_partial_target(fx["C"], R, fx["base_rate"], "linear").mean()
    PR = participation_ratio(fx["C"])
    order = np.argsort(PR)
    prev = None
    for K in (150, 80, 40):
        sel = np.concatenate([order[:K], order[-K:]])
        v = loto_partial_target(fx["C"][:, sel], R[:, sel],
                                fx["base_rate"][sel], "linear").mean()
        assert v > full, f"K={K}: selected {v:+.4f} not above all-features {full:+.4f}"
        if prev is not None:
            assert v > prev - 0.02, f"inflation should not shrink as K shrinks (K={K})"
        prev = v
    assert prev > 1.3 * full, f"tightest selection {prev:+.4f} vs all {full:+.4f}"


def test_min_active_mask_is_per_fold_not_global():
    """A feature measured adequately on nine tasks and thinly on the tenth must contribute to
    the nine folds where it is a PREDICTOR and drop out only where it is the TARGET."""
    fx = _planted(seed=6)
    n = fx["n_gts"].astype(float).copy()
    n[7, 3] = 0.0                                   # feature 7 thin on task 3 only
    R, den = flip_rate(fx["fl_gts"].astype(float), n)
    keep = den >= 30
    seen = {gi: bool(m[7]) for _p, _m2, _h, _b, _e, gi, m in
            _folds_xy(fx["C"], R, fx["base_rate"], keep)}
    assert seen[3] is False, "feature 7 should be dropped in fold 3"
    assert all(v for g, v in seen.items() if g != 3), "dropped from folds it belongs in"


# ------------------------------------------------------------------ the floors

def test_paired_column_shuffle_preserves_within_task_pairing():
    """One permutation per row applied to every matrix TOGETHER. Independent permutations would
    destroy the mechanical C<->R link as well as feature identity, giving an anti-conservative
    floor the observed number clears for free."""
    rng = np.random.default_rng(11)
    A = rng.standard_normal((4, 9))
    B = A * 3.0 + 1.0
    Ap, Bp = paired_column_shuffle([A, B], rng)
    for g in range(4):
        assert np.allclose(np.sort(Ap[g]), np.sort(A[g]))
        assert np.allclose(Bp[g], Ap[g] * 3.0 + 1.0), "pairing broken within a task row"
    assert not np.allclose(Ap, A), "nothing was permuted"


def test_paired_shuffle_is_more_conservative_than_independent_shuffles():
    """WHY the shuffle is paired.

    Within a task, C[g, j] and R[g, j] are two functionals of the same decisions: a large |phi|
    makes a flip likelier, and breadth is a participation ratio of |phi|. One permutation per
    row applied to BOTH keeps that joint intact and kills only cross-task feature identity.
    Permuting them independently kills the joint as well, which does not merely change the
    floor's mean -- it SHRINKS ITS SPREAD, and a smaller null sd inflates every z the observed
    number is scored with. That is the anti-conservative failure, and it is measurable.

    The second assertion is the other half: on a fixture whose signal is genuine cross-task
    structure (the target a monotone rescale of the mass, so its ranks ARE the mass's ranks),
    the paired floor still sits at zero. A floor that ate real structure would be useless.
    """
    fx = _planted(seed=7, strength=0.0)
    C = fx["C"]
    R = C / (C.sum(axis=1, keepdims=True) + 1e-12)
    obs = loto_partial_target(C, R, fx["base_rate"], "linear").mean()
    rng = np.random.default_rng(13)
    pair, indep = [], []
    for _ in range(60):
        Cp, Rp = paired_column_shuffle([C, R], rng)
        pair.append(loto_partial_target(Cp, Rp, fx["base_rate"], "linear").mean())
        Cq = np.stack([rng.permutation(r) for r in C])
        Rq = np.stack([rng.permutation(r) for r in R])
        indep.append(loto_partial_target(Cq, Rq, fx["base_rate"], "linear").mean())
    pair, indep = np.array(pair), np.array(indep)
    assert pair.std() > 1.2 * indep.std(), (
        f"independent shuffles gave a TIGHTER floor ({indep.std():.4f}) than paired "
        f"({pair.std():.4f}); using them would inflate every reported z")
    assert obs > 0.15, f"the fixture should carry real cross-task structure, got {obs:+.4f}"
    assert abs(pair.mean()) < 0.05, (
        f"the paired floor ate genuine cross-task structure ({pair.mean():+.4f}); it must "
        f"destroy feature identity across tasks, not the estimator")


def test_binomial_denominator_floor_fires_on_a_pure_ratio_artefact():
    """Every feature EQUALLY decisive, denominators correlated with breadth. Any partial here is
    manufactured by the ratio, and the floor must reproduce it rather than sit at zero."""
    rng = np.random.default_rng(17)
    Gx, Fx = 10, 400
    spread = rng.uniform(0.05, 1.0, Fx)
    C = np.stack([rng.dirichlet(np.full(Gx, s * 4)) for s in spread], axis=1) * Gx
    # broad features get many opportunities; narrow ones get few -- this alone is the artefact
    den = np.maximum(np.round(rng.gamma(2.0, 1.0, (Gx, Fx)) * (4 + 300 * spread)), 1.0)
    base = np.full(Fx, 0.1)
    Rart = binomial_denominator_null(den, 0.05, rng)
    obs = loto_partial_target(C, Rart, base, "linear").mean()
    null = np.array([loto_partial_target(C, binomial_denominator_null(den, 0.05, rng),
                                         base, "linear").mean() for _ in range(60)])
    z = (obs - null.mean()) / null.std() if null.std() > 0 else np.inf
    assert abs(z) < 4.0, (f"a target drawn FROM the null should not clear it: obs {obs:+.4f}, "
                          f"null {null.mean():+.4f} +- {null.std():.4f}, z={z:+.1f}")


def test_binomial_floor_stays_out_of_the_way_of_a_real_effect():
    """The other error direction: a genuine per-feature decisiveness must clear the floor by a
    wide margin, or the control is simply destroying the experiment."""
    fx = _planted(seed=19, strength=1.2)
    R, den = flip_rate(fx["fl_gts"].astype(float), fx["n_gts"].astype(float))
    obs = loto_partial_target(fx["C"], R, fx["base_rate"], "linear").mean()
    rng = np.random.default_rng(23)
    p_bar = float(np.nansum(R * den) / np.nansum(den))
    null = np.array([loto_partial_target(fx["C"], binomial_denominator_null(den, p_bar, rng),
                                         fx["base_rate"], "linear").mean() for _ in range(60)])
    z = (obs - null.mean()) / null.std()
    assert z > 4.0, f"a real effect was suppressed by the floor: obs {obs:+.4f}, z={z:+.1f}"


def test_denominator_control_enters_as_a_third_control():
    """`extra` must actually reach the estimator: adding the denominator as a control on a
    fixture whose ONLY signal is the denominator must collapse the partial."""
    rng = np.random.default_rng(29)
    Gx, Fx = 10, 400
    spread = rng.uniform(0.05, 1.0, Fx)
    C = np.stack([rng.dirichlet(np.full(Gx, s * 4)) for s in spread], axis=1) * Gx
    den = np.maximum(np.round(rng.gamma(2.0, 1.0, (Gx, Fx)) * (4 + 300 * spread)), 1.0)
    base = np.full(Fx, 0.1)
    R = binomial_denominator_null(den, 0.05, rng)
    without = loto_partial_target(C, R, base, "hinge5").mean()
    with_ = loto_partial_target(C, R, base, "hinge5", extra=[den]).mean()
    assert abs(with_) < abs(without) or abs(with_) < 0.05, \
        f"the denominator control did nothing: {without:+.4f} -> {with_:+.4f}"


if __name__ == "__main__":
    for name, fn in sorted(list(globals().items())):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("all causal-breadth tests passed")
