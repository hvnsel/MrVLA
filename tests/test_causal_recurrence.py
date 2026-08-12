"""Tests for the causal-signature redefinition of Path B.

Pins the four things that must be right:
  1. the signature really is S[j,t] = <w_j, g (*) u_t>  (checked against an explicit loop);
  2. contrast centering kills common-mode features (a direction that lifts every action bin
     equally must get a ZERO signature -- it steers nothing);
  3. a feature in another model with the same causal role is matched at cosine ~1, even if
     its magnitude and its dictionary index differ;
  4. the bin-permutation null: shared causal structure gives q > q_perm, while unrelated
     random signatures give q ~ q_perm (gap ~ 0).

Run directly:
    python tests/test_causal_recurrence.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from run_causal_recurrence import (
    best_match_cosine, causal_q_null, causal_signature,
    cross_model_causal_q, normalize_rows,
)


def test_signature_matches_explicit_definition():
    rng = np.random.default_rng(0)
    F, d, T = 7, 11, 5
    W_dec = rng.standard_normal((F, d))
    W_U = rng.standard_normal((T, d))
    g = rng.standard_normal(d)
    S = causal_signature(W_dec, W_U, g, center=False)
    for j in range(F):
        for t in range(T):
            expect = float(np.dot(W_dec[j], g * W_U[t]))    # <w_j, g (*) u_t>
            assert abs(S[j, t] - expect) < 1e-9, (j, t)


def test_centering_removes_common_mode():
    """If every action bin has the SAME readout direction, no feature can steer the action:
    every signature must centre to zero."""
    rng = np.random.default_rng(1)
    F, d, T = 6, 9, 8
    u = rng.standard_normal(d)
    W_U = np.tile(u, (T, 1))                    # all 256 bins identical
    W_dec = rng.standard_normal((F, d))
    g = rng.standard_normal(d)
    S = causal_signature(W_dec, W_U, g, center=True)
    assert np.abs(S).max() < 1e-9
    # without centering it is NOT zero -- the centering is what does the work
    S_raw = causal_signature(W_dec, W_U, g, center=False)
    assert np.abs(S_raw).max() > 1e-6


def test_same_causal_role_matches_at_cosine_one():
    """A feature in model B that drives the actions the same way as A's feature j must be
    found, regardless of its magnitude or its index in B's dictionary."""
    rng = np.random.default_rng(2)
    F, d, T = 12, 16, 20
    W_U = rng.standard_normal((T, d)); g = rng.standard_normal(d)
    W_dec_a = rng.standard_normal((F, d))
    W_dec_b = rng.standard_normal((F, d))
    j, k = 3, 9
    W_dec_b[k] = 4.7 * W_dec_a[j]               # same direction, different scale + index
    Sa, _ = normalize_rows(causal_signature(W_dec_a, W_U, g))
    Sb, _ = normalize_rows(causal_signature(W_dec_b, W_U, g))
    q, idx = best_match_cosine(Sa, Sb)
    assert idx[j] == k, (idx[j], k)
    assert q[j] > 0.999, q[j]


def test_null_is_calibrated_and_detects_real_structure():
    """The random-decoder null must (a) sit at the same level as genuinely unrelated models
    -- so an unrelated pair yields gap ~ 0, no fake positive -- and (b) be clearly beaten when
    the two models really do share causal roles."""
    rng = np.random.default_rng(3)
    F, d, T = 60, 24, 64
    W_U = rng.standard_normal((T, d)); g = rng.standard_normal(d)
    A = rng.standard_normal((F, d)); A /= np.linalg.norm(A, axis=1, keepdims=True)
    Sa, _ = normalize_rows(causal_signature(A, W_U, g))
    head = (F, d, W_U, g, True)

    # (a) CALIBRATION: an unrelated model must score ~ the null (no manufactured gap)
    C = rng.standard_normal((F, d)); C /= np.linalg.norm(C, axis=1, keepdims=True)
    Sc, _ = normalize_rows(causal_signature(C, W_U, g))
    q_unrel = cross_model_causal_q(Sa, [Sc])
    q_null = causal_q_null(Sa, [head], np.random.default_rng(1), n_perm=5)
    assert abs(q_unrel.mean() - q_null.mean()) < 0.05, (q_unrel.mean(), q_null.mean())

    # (b) POWER: a model sharing half of A's directions must beat the null on those features
    B = rng.standard_normal((F, d)); B /= np.linalg.norm(B, axis=1, keepdims=True)
    B[:30] = A[:30] * rng.uniform(0.5, 2.0, (30, 1))
    Sb, _ = normalize_rows(causal_signature(B, W_U, g))
    q_shared = cross_model_causal_q(Sa, [Sb])
    shared = slice(0, 30)
    assert q_shared[shared].mean() > q_null[shared].mean() + 0.2, \
        (q_shared[shared].mean(), q_null[shared].mean())


def main():
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f)]
    failed = 0
    for n, f in tests:
        try:
            f(); print(f"  PASS  {n}")
        except AssertionError as e:
            failed += 1; print(f"  FAIL  {n}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
