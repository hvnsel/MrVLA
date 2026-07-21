"""Tests for the step-3b cross-layer residual consistency check.

Run directly (no pytest needed):
    python tests/test_residual_consistency.py
or via pytest:
    pytest tests/test_residual_consistency.py
"""

from __future__ import annotations

import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mrvla.residual_consistency import (  # noqa: E402
    correlation_matrices,
    load_residuals,
    mean_off_diagonal,
)


def _write_audit_npz(path, episode, resid, score):
    np.savez_compressed(path,
                        episode=episode.astype(np.int32),
                        score_residual=resid.astype(np.float32),
                        score=score.astype(np.float32))


def test_common_trait_is_consistent():
    rng = np.random.default_rng(0)
    E = 300
    trait = rng.normal(size=E)
    vectors = {layer: trait + rng.normal(scale=0.4, size=E)
               for layer in (0, 8, 24)}
    layers, P, S = correlation_matrices(vectors)
    assert layers == [0, 8, 24]
    assert mean_off_diagonal(P) > 0.7
    assert mean_off_diagonal(S) > 0.7
    assert np.allclose(np.diag(P), 1.0)
    assert np.allclose(P, P.T)


def test_independent_noise_is_inconsistent():
    rng = np.random.default_rng(1)
    vectors = {layer: rng.normal(size=300) for layer in (0, 8, 24, 31)}
    _, P, _ = correlation_matrices(vectors)
    assert abs(mean_off_diagonal(P)) < 0.1


def test_load_residuals_aligns_on_common_episodes():
    rng = np.random.default_rng(2)
    with tempfile.TemporaryDirectory() as d:
        # layer 0: episodes 0..9; layer 8: episodes 5..14 (overlap 5..9)
        ep0 = np.arange(10)
        ep8 = np.arange(5, 15)
        r0, s0 = rng.normal(size=10), rng.normal(size=10)
        r8, s8 = rng.normal(size=10), rng.normal(size=10)
        _write_audit_npz(os.path.join(d, "layer_00_confound_audit_count.npz"),
                         ep0, r0, s0)
        _write_audit_npz(os.path.join(d, "layer_08_confound_audit_count.npz"),
                         ep8, r8, s8)
        episodes, residuals, scores = load_residuals(d, [0, 8], "count")
        assert episodes.tolist() == [5, 6, 7, 8, 9]
        assert np.allclose(residuals[0], r0[5:10].astype(np.float32))
        assert np.allclose(residuals[8], r8[0:5].astype(np.float32))
        assert np.allclose(scores[0], s0[5:10].astype(np.float32))


def test_mean_off_diagonal():
    M = np.array([[1.0, 0.5, 0.3],
                  [0.5, 1.0, 0.1],
                  [0.3, 0.1, 1.0]])
    assert np.isclose(mean_off_diagonal(M), (0.5 + 0.3 + 0.1) / 3)
    assert mean_off_diagonal(np.ones((1, 1))) == 0.0


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
