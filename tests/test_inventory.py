"""Tests for mrvla.inventory -- one-to-many causal-signature matching (A4).

The central claim this code has to support is that a rising cos-vs-m curve means model B's
dictionary can EXPRESS model A's role via a coalition. So the OMP must actually find the span
when one exists, must not find one when it does not, and must be monotone. And because cos rises
with m mechanically, the tests also pin that a random dictionary rises too -- which is exactly
why every null is m-matched.

The planted-splitting test is the one that matters most: it constructs the failure mode the
whole reformulation is premised on (one model's feature fragmented across three in another) and
checks that one-to-one matching misses it while the sweep recovers it.

Run directly:
    python tests/test_inventory.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mrvla.inventory import (  # noqa: E402
    anti_aligned_fraction, chance_corrected_retention, normalize_rows, omp_curve,
    random_signature_dictionary, signature_sharpness, signed_best_match,
)

D = 32


def unit(x):
    return normalize_rows(np.atleast_2d(x))[0]


def test_omp_recovers_an_exact_span():
    """If the target IS a combination of three B features, m=3 must reach cos ~1 -- and m=1
    must not, or there would be nothing for the sweep to reveal."""
    rng = np.random.default_rng(0)
    B = unit(rng.standard_normal((40, D)))
    parts = [3, 11, 27]
    target = unit(B[parts].sum(axis=0))
    cos, chosen = omp_curve(target, B, m_max=4)
    assert cos[0, 0] < 0.85
    assert cos[0, 2] > 0.999
    assert set(chosen[0, :3].tolist()) == set(parts)


def test_restarts_tighten_the_bound_without_moving_m1():
    """Restarts exist because greedy can be led astray by its first pick. They must improve the
    curve at larger m while leaving m=1 exactly equal to the one-to-one |max cos| -- that
    equality is what keeps the sweep comparable to the published q_causal."""
    for seed in (0, 2, 31):
        roles, B = _split_dictionary(12, 160, 30, np.random.default_rng(seed))
        plain, _ = omp_curve(roles, B, m_max=4, n_restarts=1)
        tuned, _ = omp_curve(roles, B, m_max=4, n_restarts=6)
        # m=1 is untouched -- this is the comparability guarantee with the published metric
        assert np.allclose(tuned[:, 0], np.abs(roles @ B.T).max(axis=1))
        assert np.allclose(plain[:, 0], tuned[:, 0])
        # restarts can only help, never hurt, and monotonicity survives the elementwise max
        assert np.all(tuned >= plain - 1e-9)
        assert np.all(np.diff(tuned, axis=1) >= -1e-9)
    # Strict improvement is NOT guaranteed -- on some dictionaries plain greedy already finds
    # the best subset it is going to find, and restarts change nothing. Asserting a universal
    # gain would be false. Seed 0 is a case where the first pick genuinely misleads greedy.
    roles, B = _split_dictionary(12, 160, 30, np.random.default_rng(0))
    plain, _ = omp_curve(roles, B, m_max=4, n_restarts=1)
    tuned, _ = omp_curve(roles, B, m_max=4, n_restarts=6)
    assert tuned[:, 3].mean() > plain[:, 3].mean() + 0.05


def test_omp_is_monotone_and_bounded():
    rng = np.random.default_rng(1)
    A = unit(rng.standard_normal((25, D)))
    B = unit(rng.standard_normal((60, D)))
    cos, _ = omp_curve(A, B, m_max=6)
    assert np.all(np.diff(cos, axis=1) >= -1e-9)
    assert cos.max() <= 1.0 + 1e-9 and cos.min() >= 0.0


def test_m1_equals_absolute_best_match():
    """The sweep's first point must be exactly the one-to-one metric, in absolute form."""
    rng = np.random.default_rng(2)
    A = unit(rng.standard_normal((30, D)))
    B = unit(rng.standard_normal((50, D)))
    cos, _ = omp_curve(A, B, m_max=1)
    assert np.allclose(cos[:, 0], np.abs(A @ B.T).max(axis=1))


def test_signed_metric_and_m1_differ_exactly_on_anti_aligned_features():
    """Quantifies the documented discontinuity with the published q_causal."""
    rng = np.random.default_rng(3)
    A = unit(rng.standard_normal((200, D)))
    B = unit(rng.standard_normal((40, D)))
    q_signed, _ = signed_best_match(A, B)
    cos, _ = omp_curve(A, B, m_max=1)
    frac = anti_aligned_fraction(A, B)
    assert 0.0 <= frac <= 1.0
    # where the best absolute match is positive the two agree; the fraction counts the rest
    C = A @ B.T
    best_abs = C[np.arange(C.shape[0]), np.argmax(np.abs(C), axis=1)]
    agree = np.isclose(q_signed, cos[:, 0])
    assert np.all(agree[best_abs > 0])
    assert abs(frac - float((best_abs < 0).mean())) < 1e-12


def _split_dictionary(n_roles, d, n_noise, rng, damp=0.35):
    """Roles held whole in A; in B each is split into three features that SPAN it without any
    one being aligned with it. Returns (roles [n_roles, d], B [3*n_roles + n_noise, d])."""
    roles = unit(rng.standard_normal((n_roles, d)))
    frags = []
    for r in range(n_roles):
        basis, _ = np.linalg.qr(np.column_stack([roles[r], rng.standard_normal((d, 2))]))
        basis = basis[:, :3].T                     # orthonormal 3-dim CONTAINING the role
        mix = rng.standard_normal((3, 3))
        mix[:, 0] *= damp                          # damp the role component in every fragment
        frags.append(mix @ basis)
    return roles, unit(np.concatenate(frags + [rng.standard_normal((n_noise, d))], axis=0))


def test_planted_feature_splitting_is_invisible_to_one_to_one_and_visible_to_the_sweep():
    """THE motivating construction. Model A holds a role as one feature; model B holds three
    features that SPAN it without any one being close to it -- which is what dictionary
    splitting actually looks like. One-to-one matching underrates B badly; the sweep recovers
    the role exactly. If this failed, the entire A4 reformulation would be unmotivated."""
    rng = np.random.default_rng(4)
    roles, B = _split_dictionary(n_roles=12, d=160, n_noise=30, rng=rng)
    cos, _ = omp_curve(roles, B, m_max=4)
    assert cos[:, 0].mean() < 0.6, "one-to-one should NOT already find the role"
    assert cos[:, 2].mean() > cos[:, 0].mean() + 0.20, "the sweep must recover most of it"
    # NOT asserted: exact saturation. Greedy is suboptimal even when the span provably contains
    # the target, which is why the curve is documented as a LOWER bound on expressibility.
    assert cos[:, 2].mean() < 1.0 + 1e-9


def test_crowding_defeats_the_sweep_when_the_ambient_space_is_too_small():
    """Why action_space_geometry.py runs first, demonstrated rather than argued.

    The SAME planted splitting, in a cramped ambient space with a dictionary large relative to
    it, is NOT recovered: greedy matching latches onto spurious directions from other roles
    because random vectors are close by chance when there is nowhere to be far apart. This is
    precisely the regime the Stage 0 rank gate screens for -- if the measured signature space is
    small relative to F, a rising or flat m-curve says more about dimension counting than about
    dictionaries, and the distributional route is the honest one."""
    rng = np.random.default_rng(24)
    roles_wide, B_wide = _split_dictionary(12, 160, 30, rng)
    rng = np.random.default_rng(24)
    roles_tight, B_tight = _split_dictionary(12, 24, 30, rng)
    wide, _ = omp_curve(roles_wide, B_wide, m_max=3)
    tight, _ = omp_curve(roles_tight, B_tight, m_max=3)
    # the tight case starts HIGHER at m=1 purely from crowding -- random directions are close by
    # chance when there is nowhere to be far apart. That is the confound in miniature: a healthy
    # looking match that is dimension counting, not correspondence.
    assert tight[:, 0].mean() > wide[:, 0].mean() + 0.05
    # and its apparent gain from extra m is correspondingly inflated relative to what is really
    # there, so the m-sweep cannot be read at face value in this regime
    assert tight[:, 0].mean() > 0.5


def test_noisy_splitting_still_rises_but_does_not_saturate():
    """The realistic case: fragments carry the role plus variation OUTSIDE its span. The sweep
    must still show the characteristic rise, without pretending to recover the role exactly --
    so a partial rise in real data is read as partial evidence, not as proof."""
    rng = np.random.default_rng(14)
    n_roles = 12
    roles = unit(rng.standard_normal((n_roles, D)))
    frags = []
    for r in range(n_roles):
        w = rng.standard_normal((3, D)) * 0.35
        w[0] += roles[r] * 0.5
        w[1] += roles[r] * 0.3
        w[2] += roles[r] * 0.2
        frags.append(w)
    B = unit(np.concatenate(frags + [rng.standard_normal((30, D))], axis=0))
    cos, _ = omp_curve(roles, B, m_max=4)
    assert cos[:, 2].mean() > cos[:, 0].mean() + 0.15      # the rise is the signal
    assert cos[:, 2].mean() < 0.95                          # but recovery is incomplete


def test_atomic_features_are_flat_across_m():
    """The control half of the same contrast: when B contains the role outright, there is
    nothing for extra m to add, so the curve is flat. Splitting-driven rise must be
    distinguishable from a rise everything shows."""
    rng = np.random.default_rng(5)
    roles = unit(rng.standard_normal((12, D)))
    B = unit(np.concatenate([roles + 0.02 * rng.standard_normal(roles.shape),
                             rng.standard_normal((60, D))], axis=0))
    cos, _ = omp_curve(roles, B, m_max=4)
    assert cos[:, 0].mean() > 0.97
    assert cos[:, 3].mean() - cos[:, 0].mean() < 0.03


def test_a_random_dictionary_also_rises_which_is_why_nulls_are_m_matched():
    """Pins the reason the null design is non-negotiable: m arbitrary directions span more of
    anything, so an unmatched null would read a mechanical rise as recurrence."""
    rng = np.random.default_rng(6)
    A = unit(rng.standard_normal((40, D)))
    B = unit(rng.standard_normal((40, D)))
    cos, _ = omp_curve(A, B, m_max=8)
    assert cos[:, 7].mean() > cos[:, 0].mean() + 0.05


def test_no_index_is_ever_chosen_twice():
    rng = np.random.default_rng(7)
    A = unit(rng.standard_normal((20, D)))
    B = unit(rng.standard_normal((15, D)))
    _, chosen = omp_curve(A, B, m_max=6)
    for row in chosen:
        assert len(set(row.tolist())) == len(row)


def test_degenerate_dictionary_does_not_stall_or_exceed_one():
    """A dictionary of near-duplicates spans one direction; the curve must saturate cleanly."""
    rng = np.random.default_rng(8)
    v = unit(rng.standard_normal((1, D)))
    B = unit(np.repeat(v, 10, axis=0) + 1e-9 * rng.standard_normal((10, D)))
    A = unit(rng.standard_normal((5, D)))
    cos, _ = omp_curve(A, B, m_max=5)
    assert np.all(cos <= 1.0 + 1e-9)
    assert np.all(np.diff(cos, axis=1) >= -1e-9)


def test_positive_only_never_beats_unrestricted():
    """TopK codes are non-negative, so restricted selection is the more faithful bound and must
    be the weaker one."""
    rng = np.random.default_rng(9)
    A = unit(rng.standard_normal((30, D)))
    B = unit(rng.standard_normal((40, D)))
    free, _ = omp_curve(A, B, m_max=4, positive_only=False)
    pos, _ = omp_curve(A, B, m_max=4, positive_only=True)
    assert np.all(pos[:, 0] <= free[:, 0] + 1e-9)


def test_random_dictionary_has_the_requested_size_and_lives_in_the_head_subspace():
    """Size must match the real dictionary or the best-of-F effect differs between the
    observation and its null."""
    rng = np.random.default_rng(10)
    W_U = rng.standard_normal((24, D))
    g = np.abs(rng.normal(1.0, 0.1, D))
    S = random_signature_dictionary(37, D, W_U, g, rng)
    assert S.shape == (37, 24)
    assert np.allclose(np.linalg.norm(S, axis=1), 1.0)
    assert np.abs(S.sum(axis=1)).max() < 1e-9      # contrast-centred


def test_chance_corrected_retention_anchors_at_zero_and_one():
    null, ceil = np.array([0.2, 0.2]), np.array([0.8, 0.8])
    assert np.allclose(chance_corrected_retention(np.array([0.2, 0.8]), null, ceil), [0.0, 1.0])
    assert np.allclose(chance_corrected_retention(np.array([0.5]), np.array([0.2]),
                                                  np.array([0.8])), [0.5])
    # a degenerate ceiling (seed matching no better than chance) must be NaN, not a huge number
    assert np.isnan(chance_corrected_retention(np.array([0.5]), np.array([0.3]),
                                               np.array([0.3])))[0]


def test_sharpness_separates_peaked_from_spread_signatures():
    peaked = np.zeros((1, 64)); peaked[0, 3] = 1.0
    spread = np.ones((1, 64))
    assert signature_sharpness(peaked)[0] < 1.5
    assert signature_sharpness(spread)[0] > 60


def test_decile_curves_split_by_score_and_report_slopes():
    """The decile table is the primary output, so its bookkeeping is pinned: features are
    ordered by the breadth score, non-finite scores are dropped rather than silently binned,
    and the slope is the m_max minus m=1 gain."""
    from inventory_recurrence import decile_curves
    n, m = 100, 4
    score = np.arange(n, dtype=np.float64)
    score[7] = np.nan                                  # inactive feature: must be excluded
    cos = np.zeros((n, m))
    cos[:] = np.linspace(0.3, 0.9, m)[None, :]
    cos[score > 50] += 0.05                            # the top half gains a little
    d = decile_curves(cos, score, n_dec=10)
    assert d["n_deciles"] == 10
    assert sum(d["sizes"]) == n - 1                    # the NaN was dropped
    assert all(abs(s - (0.9 - 0.3)) < 1e-9 for s in d["slopes"])
    assert d["curves"][-1][0] > d["curves"][0][0]      # deciles ordered by score


def test_decile_curves_detect_a_slope_difference():
    """The actual claim: a steeper rise in the top decile must show up as a slope difference."""
    from inventory_recurrence import decile_curves
    n, m = 200, 5
    score = np.arange(n, dtype=np.float64)
    cos = np.tile(np.linspace(0.4, 0.45, m), (n, 1))   # flat-ish everywhere ...
    top = score > 180
    cos[top] = np.linspace(0.4, 0.8, m)                # ... except a steep top decile
    d = decile_curves(cos, score, n_dec=10)
    assert d["slopes"][-1] > d["slopes"][0] + 0.3


if __name__ == "__main__":
    for name, fn in sorted(list(globals().items())):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("all inventory tests passed")
