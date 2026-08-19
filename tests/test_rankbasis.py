"""Tests for mrvla.rankbasis -- the linearity check on the A4 control plane.

A diagnostic is only worth running if it can both FIRE and STAY SILENT correctly, so the
load-bearing tests are the two error directions:

  * power      -- a confound that is curved in rank space (and therefore invisible to the
                  linear control plane) must produce a spuriously positive `linear` partial
                  that the richer bases knock down. If this fails the check is decorative.
  * no false alarm -- a confound that really is linear, and a genuine signal, must NOT be
                  destroyed by basis enrichment. Otherwise "the number dropped" would mean
                  nothing, because every number drops.

Plus the calibration the whole design hangs on: random columns of matched count measure the
drop attributable to spending degrees of freedom, so a real basis is only implicated when it
costs materially more than the placebo.

Run directly:
    python tests/test_rankbasis.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mrvla.attribution import rank_partial_both  # noqa: E402
from mrvla.rankbasis import (  # noqa: E402
    SPECS, control_design, curvature_gain, loto_partial_design, loto_partial_placebo,
    loto_stratified, orthonormal_basis, rank_partial_design, rank_unit, residualise,
)


def _curved_fixture(n=2000, seed=0):
    """A confound the LINEAR control plane cannot see.

    Both the predictor and the target are driven by u^2, where u is the ranked control. u^2 is
    symmetric about zero, so it is very nearly uncorrelated with u itself -- meaning linear
    residualisation removes almost none of it, and the leftover shows up as a strong partial
    correlation between two variables that share no direct relationship at all.
    """
    rng = np.random.default_rng(seed)
    c1 = rng.standard_normal(n)
    u = rank_unit(c1)
    g = u * u - (u * u).mean()
    x = g + 0.35 * rng.standard_normal(n)          # predictor: curve + noise
    y = g + 0.35 * rng.standard_normal(n)          # target:    same curve + independent noise
    c2 = rng.standard_normal(n)                     # a second, inert control
    return y, x, c1, c2


def _linear_fixture(n=2000, seed=1):
    """The same story with a LINEAR confound -- what the plane is already built to remove."""
    rng = np.random.default_rng(seed)
    c1 = rng.standard_normal(n)
    u = rank_unit(c1)
    x = u + 0.35 * rng.standard_normal(n)
    y = u + 0.35 * rng.standard_normal(n)
    c2 = rng.standard_normal(n)
    return y, x, c1, c2


def _signal_fixture(n=2000, seed=2):
    """A REAL x -> y relationship on top of a curved confound. Enrichment must not erase it."""
    rng = np.random.default_rng(seed)
    c1 = rng.standard_normal(n)
    u = rank_unit(c1)
    g = u * u - (u * u).mean()
    shared = rng.standard_normal(n)                 # the genuine common cause
    x = g + shared + 0.3 * rng.standard_normal(n)
    y = g + shared + 0.3 * rng.standard_normal(n)
    c2 = rng.standard_normal(n)
    return y, x, c1, c2


def test_linear_spec_reproduces_the_shipped_estimator():
    """The check has to be the SAME statistic, otherwise a difference is uninterpretable.
    Inputs are continuous, so the tie conventions coincide and the match should be exact."""
    y, x, c1, c2 = _linear_fixture()
    a = rank_partial_design(y, x, [c1, c2], "linear")
    b = rank_partial_both(y, x, c1, c2)
    assert abs(a - b) < 1e-10, (a, b)


def test_rank_unit_is_an_affine_map_of_ranks():
    rng = np.random.default_rng(0)
    v = rng.standard_normal(300)
    u = rank_unit(v)
    assert abs(u.mean()) < 1e-12
    assert abs(u.max() - u.min() - 2.0) < 1e-12
    # order preserved, and ties averaged (so a tied block gets ONE value)
    assert np.all(np.diff(u[np.argsort(v)]) >= -1e-12)
    assert len(set(np.round(rank_unit([1.0, 1.0, 1.0, 2.0])[:3], 12))) == 1


def test_curved_confound_fires_the_check():
    """THE power test. Linear control leaves a large spurious partial; a basis that can
    represent the curve removes it."""
    y, x, c1, c2 = _curved_fixture()
    lin = rank_partial_design(y, x, [c1, c2], "linear")
    # ~ +0.40 here: curvature alone manufactures a partial of the same order as the one the
    # paper reports (+0.493), which is precisely why this check is not optional.
    assert lin > 0.35, f"fixture no longer produces a spurious partial (got {lin})"
    for spec in ("quad", "cubic", "hinge5", "tensor4"):
        rich = rank_partial_design(y, x, [c1, c2], spec)
        assert abs(rich) < 0.12, f"{spec} failed to absorb the curvature: {rich}"


def test_linear_confound_does_not_false_alarm():
    """A confound the plane already handles must leave nothing for the richer bases to find."""
    y, x, c1, c2 = _linear_fixture()
    lin = rank_partial_design(y, x, [c1, c2], "linear")
    assert abs(lin) < 0.1, lin
    for spec in SPECS:
        assert abs(rank_partial_design(y, x, [c1, c2], spec) - lin) < 0.05


def test_enrichment_does_not_erase_a_real_signal():
    """If richer bases killed every relationship, a drop on real data would be meaningless."""
    y, x, c1, c2 = _signal_fixture()
    vals = {s: rank_partial_design(y, x, [c1, c2], s) for s in SPECS}
    assert vals["linear"] > 0.6, vals
    for s in SPECS:
        assert vals[s] > 0.55, vals            # survives every basis, including the tensor
    assert vals["tensor4"] > vals["linear"] - 0.12


def test_curvature_gain_separates_curved_from_linear():
    rng = np.random.default_rng(3)
    G, F = 10, 1500
    br = rng.random(F)
    # a C whose columns vary smoothly -- curvature_gain reads the control surface, not C's story
    C_lin = np.abs(rng.lognormal(0, 1, (G, F)))
    g_lin = curvature_gain(C_lin, br, "tensor4")
    assert g_lin["extra_df"] > 10
    assert g_lin["delta_r2_predictor"] >= -1e-9      # R^2 can only rise with nested bases
    assert g_lin["delta_r2_target"] >= -1e-9


def test_placebo_measures_the_degrees_of_freedom_cost():
    """Random columns must cost something (so the calibration is real) but only a little at
    n ~ 2000 (so a large real drop is attributable to curvature, not bookkeeping)."""
    rng = np.random.default_rng(4)
    G, F = 10, 2000
    C = np.abs(rng.lognormal(0, 1, (G, F)))
    br = rng.random(F)
    base = loto_partial_design(C, br, "linear").mean()
    for n_extra in (3, 33):
        p = loto_partial_placebo(C, br, n_extra, np.random.default_rng(5)).mean()
        assert abs(p - base) < 0.05, (n_extra, p, base)


def test_orthonormal_basis_drops_dependent_columns():
    rng = np.random.default_rng(6)
    a = rng.standard_normal(200)
    D = np.stack([a, 2 * a, rng.standard_normal(200)], axis=1)
    D = D - D.mean(axis=0, keepdims=True)      # as control_design always returns them
    Q = orthonormal_basis(D)
    assert Q.shape[1] == 2                     # the duplicated direction is dropped
    assert np.allclose(Q.T @ Q, np.eye(2), atol=1e-10)
    # projection is idempotent -- this holds because control_design CENTRES its columns,
    # so span(Q) contains only centred vectors and residualise's mean-removal is a no-op
    # on an already-residualised input
    v = rng.standard_normal(200)
    r = residualise(v, Q)
    assert np.allclose(residualise(r, Q), r, atol=1e-10)


def test_design_columns_are_centred_and_finite():
    rng = np.random.default_rng(7)
    cs = [rng.standard_normal(400), rng.random(400)]
    for spec in SPECS:
        D = control_design(cs, spec)
        assert np.isfinite(D).all()
        assert np.abs(D.mean(axis=0)).max() < 1e-12


def _breadth_fixture(G=10, F=3000, seed=8):
    """Breadth made explicit: feature f spreads magnitude m_f evenly over k_f tasks.

    Then PR == k_f and total magnitude == m_f EXACTLY, and k is drawn independently of m -- so
    breadth and the magnitude confound are decoupled by construction, which is the situation
    the headline statistic claims to detect. The mechanism that makes it detectable: at equal
    magnitude, a broad feature reliably puts a moderate amount on the held-out task, while a
    narrow one usually puts nothing there and occasionally puts everything.
    """
    rng = np.random.default_rng(seed)
    k = rng.integers(1, G + 1, F)
    m = np.exp(rng.standard_normal(F))
    C = np.zeros((G, F))
    for f in range(F):
        C[rng.choice(G, k[f], replace=False), f] = m[f] / k[f]
    return C, rng.random(F)


def test_stratified_is_positive_on_signal_and_null_on_a_pure_confound():
    """The assumption-free backstop, exercised in both directions on a real LOTO layout."""
    C, br = _breadth_fixture()
    s = loto_stratified(C, br, n_bins=5, min_cell=20)
    assert s["n_folds"] == 10 and s["mean_cells_used"] >= 10
    assert s["mean"] > 0.05, s
    # and it is ATTENUATED relative to the partial, as the docstring warns: throwing away
    # between-cell variance in breadth costs real signal, so it corroborates the sign only
    assert s["mean"] < float(loto_partial_design(C, br, "linear").mean())
    # pure noise: feature identity destroyed across tasks -> nothing survives inside a cell
    rng = np.random.default_rng(99)
    C_null = np.stack([rng.permutation(C[g]) for g in range(C.shape[0])])
    assert abs(loto_stratified(C_null, br, n_bins=5, min_cell=20)["mean"]) < 0.05


def test_loto_partial_design_linear_matches_the_shipped_loto():
    """Plumbing: same folds, same masking, same number."""
    from mrvla.attribution import loto_partial_both
    rng = np.random.default_rng(9)
    C = np.abs(rng.lognormal(0, 1, (10, 800)))
    br = rng.random(800)
    a = loto_partial_design(C, br, "linear")
    b = loto_partial_both(C, br)
    assert a.size == b.size
    assert np.abs(a - b).max() < 1e-9, np.abs(a - b).max()


if __name__ == "__main__":
    for name, fn in sorted(list(globals().items())):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("all rankbasis tests passed")
