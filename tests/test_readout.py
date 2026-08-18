"""Tests for mrvla.readout -- the exact readout counterfactual.

Everything here is checked against a BRUTE-FORCE recomputation through the real
`mrvla.attribution` path (perturb h, re-run RMSNorm, re-run the unembedding, take the argmax).
The logit-space shortcut is the whole point of the module -- it is what makes per-feature
necessity affordable on 446k decisions -- so it has to be provably identical to the slow path,
not merely close.

Also pinned: the two ablation semantics stay distinct, the margin-bound pruning is exact rather
than approximate, the coalition path reproduces the hook's own formula including its
over-subtraction on correlated directions, and the reversed bin axis is applied.

Run directly:
    python tests/test_readout.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mrvla.attribution import action_logits  # noqa: E402
from mrvla.readout import (  # noqa: E402
    ablated_argmax, ablated_logits, assert_unit_rows, bin_index_from_row, cannot_flip_mask,
    coalition_ablated_argmax, coalition_coeffs, coalition_overlap, coded_coeffs,
    projection_coeffs, signature_matrix, signed_bin_shift, single_feature_flips, top2_margin,
    unnormalized_logits,
)

D, NBINS, F, N = 48, 32, 20, 60
EPS = 1e-5


def fixture(seed=0):
    """A small readout: unit-norm decoder rows, an ordered-ish bin geometry, real residuals."""
    rng = np.random.default_rng(seed)
    W_dec = rng.standard_normal((F, D))
    W_dec /= np.linalg.norm(W_dec, axis=1, keepdims=True)
    t = np.linspace(-1, 1, NBINS)[:, None]
    W_U = np.concatenate([t ** 0, t ** 1, t ** 2], axis=1) @ rng.standard_normal((3, D))
    W_U += 0.35 * rng.standard_normal((NBINS, D))       # keep margins small so flips happen
    g = np.abs(rng.normal(1.0, 0.1, D))
    H = rng.standard_normal((N, D)) * 0.6
    return rng, W_dec, W_U, g, H


def test_unnormalized_logits_preserve_the_argmax():
    """r is a positive per-row scalar, so dropping it cannot change which bin wins."""
    _, _, W_U, g, H = fixture()
    L = unnormalized_logits(H, g, W_U)
    for i in range(N):
        true = action_logits(H[i], W_U, g, EPS)
        assert np.argmax(L[i]) == np.argmax(true)
        # and the two differ only by the positive scalar 1/r
        ratio = true / L[i]
        assert np.allclose(ratio, ratio[0]) and ratio[0] > 0


def test_projection_ablation_matches_brute_force_through_the_real_readout():
    """THE contract. Logit-space shortcut == perturb h, re-normalise, re-decode."""
    _, W_dec, W_U, g, H = fixture(1)
    S = signature_matrix(W_dec, g, W_U)
    A = projection_coeffs(H, W_dec)
    for j in (0, 3, 11):
        fast = ablated_argmax(unnormalized_logits(H, g, W_U), S[j], A[:, j])
        slow = [np.argmax(action_logits(H[i] - A[i, j] * W_dec[j], W_U, g, EPS))
                for i in range(N)]
        assert np.array_equal(fast, np.array(slow))


def test_coded_ablation_matches_brute_force_and_differs_from_projection():
    """The second semantics: remove l2*z_j*w_j. Must also be exact -- and must NOT coincide
    with projection, or the distinction the module draws would be empty."""
    rng, W_dec, W_U, g, H = fixture(2)
    S = signature_matrix(W_dec, g, W_U)
    z = np.zeros((N, F))
    for i in range(N):                                   # k-sparse codes, as TopK produces
        z[i, rng.choice(F, 5, replace=False)] = rng.gamma(2.0, 0.5, 5)
    l2 = np.abs(rng.normal(1.0, 0.2, N))
    C = coded_coeffs(z, l2)
    j = int(np.argmax((z > 0).sum(axis=0)))              # a feature that is often active
    fast = ablated_argmax(unnormalized_logits(H, g, W_U), S[j], C[:, j])
    slow = [np.argmax(action_logits(H[i] - C[i, j] * W_dec[j], W_U, g, EPS)) for i in range(N)]
    assert np.array_equal(fast, np.array(slow))

    proj = projection_coeffs(H, W_dec)[:, j]
    assert not np.allclose(proj, C[:, j])                # genuinely different interventions


def test_centering_the_signature_changes_no_flip():
    """Contrast-centering shifts every bin of a row equally, so argmax is untouched. This is why
    one signature matrix can serve both the counterfactuals and the cross-model cosines."""
    _, W_dec, W_U, g, H = fixture(3)
    L = unnormalized_logits(H, g, W_U)
    A = projection_coeffs(H, W_dec)
    S_raw = signature_matrix(W_dec, g, W_U, center=False)
    S_cen = signature_matrix(W_dec, g, W_U, center=True)
    for j in (2, 7):
        assert np.array_equal(ablated_argmax(L, S_raw[j], A[:, j]),
                              ablated_argmax(L, S_cen[j], A[:, j]))


def test_top2_margin_is_the_gap_to_the_runner_up():
    L = np.array([[1.0, 5.0, 3.0], [2.0, 2.0, 9.0]])
    arg, margin = top2_margin(L)
    assert list(arg) == [1, 2]
    assert np.allclose(margin, [2.0, 7.0])


def test_margin_pruning_is_exact_not_approximate():
    """Pruning is an optimisation; it must never change an answer. Verified feature by feature
    against the unpruned computation, and it must actually prune something."""
    _, W_dec, W_U, g, H = fixture(4)
    L = unnormalized_logits(H, g, W_U)
    S = signature_matrix(W_dec, g, W_U)
    A = projection_coeffs(H, W_dec)
    total_pruned = 0
    for j in range(F):
        res = single_feature_flips(L, S[j], A[:, j])
        brute = np.argmax(L - A[:, j][:, None] * S[j][None, :], axis=1)
        assert np.array_equal(res["new_argmax"], brute)
        total_pruned += res["n_pruned"]
    assert total_pruned > 0, "the bound never fired; it is not buying anything"


def test_pruned_rows_provably_cannot_flip():
    """Direct check of the bound's logic on the rows it excludes."""
    _, W_dec, W_U, g, H = fixture(5)
    L = unnormalized_logits(H, g, W_U)
    S = signature_matrix(W_dec, g, W_U)
    A = projection_coeffs(H, W_dec)
    base, margin = top2_margin(L)
    j = 6
    safe = cannot_flip_mask(A[:, j], S[j], margin)
    if safe.any():
        got = np.argmax(L[safe] - A[safe, j][:, None] * S[j][None, :], axis=1)
        assert np.array_equal(got, base[safe])


def test_flips_actually_occur_in_the_fixture():
    """A test suite where nothing ever flips would pass vacuously."""
    _, W_dec, W_U, g, H = fixture(6)
    L = unnormalized_logits(H, g, W_U)
    S = signature_matrix(W_dec, g, W_U)
    A = projection_coeffs(H, W_dec)
    rates = [single_feature_flips(L, S[j], A[:, j])["flip_rate"] for j in range(F)]
    assert max(rates) > 0.05
    assert min(rates) < max(rates)          # features differ in necessity, as they must


def test_coalition_matches_the_hooks_own_formula():
    """The rollout hook computes h - (h @ V.T) @ V. We reproduce THAT, not a corrected
    projection, because it is the experiment that was actually run."""
    _, W_dec, W_U, g, H = fixture(7)
    S = signature_matrix(W_dec, g, W_U)
    L = unnormalized_logits(H, g, W_U)
    idx = [1, 4, 9, 12]
    V = W_dec[idx]
    coeffs = coalition_coeffs(H, W_dec, idx)
    fast = coalition_ablated_argmax(L, S, idx, coeffs)
    slow = [np.argmax(action_logits(H[i] - (H[i] @ V.T) @ V, W_U, g, EPS)) for i in range(N)]
    assert np.array_equal(fast, np.array(slow))


def test_coalition_overlap_flags_non_orthogonality():
    """Correlated coalitions are over-subtracted by the hook; the diagnostic has to see it."""
    _, W_dec, _, _, _ = fixture(8)
    W = W_dec.copy()
    W[1] = W[0] + 0.02 * W[1]                     # near-duplicate direction
    W /= np.linalg.norm(W, axis=1, keepdims=True)
    bad = coalition_overlap(W, [0, 1, 2])
    assert bad["max_abs_offdiag"] > 0.9 and not bad["is_near_orthogonal"]
    q, _ = np.linalg.qr(np.random.default_rng(0).standard_normal((D, 3)))
    good = coalition_overlap(q.T, [0, 1, 2])
    assert good["is_near_orthogonal"]


def test_ablated_logits_with_r_gives_true_logit_values():
    """Values, not just argmax: passing r must reproduce the real logits exactly."""
    from mrvla.attribution import rms
    _, W_dec, W_U, g, H = fixture(9)
    S = signature_matrix(W_dec, g, W_U)
    A = projection_coeffs(H, W_dec)
    j = 5
    Hp = H - A[:, j][:, None] * W_dec[j][None, :]
    r = np.array([rms(Hp[i], EPS) for i in range(N)])
    got = ablated_logits(unnormalized_logits(H, g, W_U), S[j], A[:, j], r=r)
    want = np.stack([action_logits(Hp[i], W_U, g, EPS) for i in range(N)])
    assert np.allclose(got, want, atol=1e-9)


def test_bin_axis_is_reversed_and_shift_sign_follows():
    assert bin_index_from_row(np.array([0]), 256)[0] == 256
    assert bin_index_from_row(np.array([255]), 256)[0] == 1
    # moving DOWN a row index means moving UP a bin index
    assert signed_bin_shift(np.array([100]), np.array([90]), 256)[0] == 10
    assert signed_bin_shift(np.array([100]), np.array([110]), 256)[0] == -10


def test_unit_row_check_catches_a_rescaled_decoder():
    _, W_dec, _, _, _ = fixture(10)
    assert assert_unit_rows(W_dec) < 1e-9
    bad = W_dec.copy(); bad[0] *= 2.0
    assert assert_unit_rows(bad) > 0.9


if __name__ == "__main__":
    for name, fn in sorted(list(globals().items())):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("all readout tests passed")
