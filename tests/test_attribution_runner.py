"""Tests for run_attribution.sae_encode_full against the REAL TopKSAE.

The contract that matters: the encoder must return z, l2 and mu such that
attribution.reconstruct(z, W_dec, l2, mu, b_pre) reproduces the SAE's own x_hat. If l2 or
mu were dropped or mis-scaled (the easy mistake -- Path B's encoder discards them), every
attribution phi would be wrong by a per-sample factor and the gate would silently
mis-measure. These tests pin that down against train_sae's actual implementation.

Run directly:
    python tests/test_attribution_runner.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mrvla.attribution import reconstruct  # noqa: E402
from mrvla.train_sae import TopKSAE  # noqa: E402
from run_attribution import sae_encode_full  # noqa: E402


def _sae(d=32, F=16, k=5, seed=0):
    torch.manual_seed(seed)
    sae = TopKSAE(d, F, k=k, k_aux=8)
    with torch.no_grad():
        sae.b_pre.copy_(torch.randn(d) * 0.1)
    sae.eval()
    return sae


def test_encoder_matches_topksae_codes():
    """z from sae_encode_full must equal the SAE's own z."""
    d, F, k = 32, 16, 5
    sae = _sae(d, F, k)
    X = np.random.default_rng(0).standard_normal((25, d)).astype(np.float32)
    z_mine, l2, mu = sae_encode_full(sae.W_enc.data, sae.b_pre.data, k, X, "cpu")
    with torch.no_grad():
        _x_hat, _xn, _xhn, z_true, _aux = sae(torch.from_numpy(X))
    assert np.allclose(z_mine, z_true.numpy(), atol=1e-5)


def test_reconstruct_matches_sae_x_hat():
    """THE contract: reconstruct(z, W_dec, l2, mu, b_pre) == the SAE's x_hat.

    This is what makes the gate's h_hat the real reconstruction, and what makes the
    l2 factor in phi correct.
    """
    d, F, k = 32, 16, 5
    sae = _sae(d, F, k, seed=1)
    X = np.random.default_rng(1).standard_normal((20, d)).astype(np.float32)
    z, l2, mu = sae_encode_full(sae.W_enc.data, sae.b_pre.data, k, X, "cpu")
    W_dec = sae.W_dec.data.numpy().astype(np.float64)
    b_pre = sae.b_pre.data.numpy().astype(np.float64)
    with torch.no_grad():
        x_hat_true, _xn, _xhn, _z, _aux = sae(torch.from_numpy(X))
    for i in range(X.shape[0]):
        h_hat = reconstruct(z[i], W_dec, float(l2[i]), float(mu[i]), b_pre)
        assert np.allclose(h_hat, x_hat_true[i].numpy(), atol=1e-4), i


def test_dropping_l2_would_break_reconstruction():
    """Guard: if l2 were ignored (set to 1), the reconstruction would NOT match.
    Confirms the l2 factor is load-bearing, not decorative."""
    d, F, k = 32, 16, 5
    sae = _sae(d, F, k, seed=2)
    X = np.random.default_rng(2).standard_normal((5, d)).astype(np.float32)
    z, l2, mu = sae_encode_full(sae.W_enc.data, sae.b_pre.data, k, X, "cpu")
    W_dec = sae.W_dec.data.numpy().astype(np.float64)
    b_pre = sae.b_pre.data.numpy().astype(np.float64)
    with torch.no_grad():
        x_hat_true, *_ = sae(torch.from_numpy(X))
    wrong = reconstruct(z[0], W_dec, 1.0, float(mu[0]), b_pre)   # l2 dropped
    assert not np.allclose(wrong, x_hat_true[0].numpy(), atol=1e-3)


def test_encoder_sparsity_and_shapes():
    d, F, k = 24, 12, 4
    sae = _sae(d, F, k, seed=3)
    X = np.random.default_rng(3).standard_normal((11, d)).astype(np.float32)
    z, l2, mu = sae_encode_full(sae.W_enc.data, sae.b_pre.data, k, X, "cpu")
    assert z.shape == (11, F) and l2.shape == (11,) and mu.shape == (11,)
    assert (z >= 0).all()                       # ReLU applied
    assert ((z > 0).sum(axis=1) <= k).all()     # at most k active
    assert (l2 > 0).all()


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
