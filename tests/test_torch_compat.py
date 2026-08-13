"""Tests for mrvla.torch_compat: unpickling LIBERO's numpy init-state files under torch>=2.6.

The real failure this guards is a rollout crash inside LIBERO's get_task_init_states, which
does a bare torch.load on a pickled numpy array. We reproduce that exact shape here: save an
ndarray with torch.save, then load it with weights_only=True.

Skips cleanly when torch is absent, so the CPU-only test suite still runs.

Run directly:
    python tests/test_torch_compat.py
"""

from __future__ import annotations

import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mrvla.torch_compat import allow_numpy_pickles


def _torch():
    try:
        import torch
        return torch
    except ImportError:
        return None


def test_ndarray_roundtrip_under_weights_only():
    """The exact LIBERO case: a float64 ndarray saved by torch.save, loaded weights_only=True."""
    torch = _torch()
    if torch is None:
        print("    (skipped: no torch)"); return
    allow_numpy_pickles()
    arr = np.random.default_rng(0).random((8, 79))          # init states are [n_init, dim]
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "init.pruned_init")
        torch.save(arr, p)
        out = torch.load(p, weights_only=True)
    assert isinstance(out, np.ndarray) and out.shape == (8, 79)
    assert np.allclose(out, arr)


def test_multiple_dtypes_and_containers():
    """Allowlisting numpy.dtype alone is not sufficient -- the concrete dtype classes are
    requested next. Cover several dtypes and a dict container."""
    torch = _torch()
    if torch is None:
        print("    (skipped: no torch)"); return
    allow_numpy_pickles()
    rng = np.random.default_rng(1)
    with tempfile.TemporaryDirectory() as d:
        for dt in (np.float64, np.float32, np.int64, np.uint8, np.bool_):
            p = os.path.join(d, f"{np.dtype(dt).name}.pt")
            arr = (rng.random((3, 4)) * 10).astype(dt)
            torch.save(arr, p)
            assert torch.load(p, weights_only=True).dtype == np.dtype(dt)
        p = os.path.join(d, "dict.pt")
        torch.save({"states": rng.random((2, 2))}, p)
        got = torch.load(p, weights_only=True)
        assert isinstance(got, dict) and got["states"].shape == (2, 2)


def test_idempotent_and_reports_names():
    """Safe to call from every worker; the first call reports what it registered."""
    torch = _torch()
    if torch is None:
        print("    (skipped: no torch)"); return
    import mrvla.torch_compat as tc
    tc._APPLIED = False                                     # force a fresh first call
    names = allow_numpy_pickles()
    assert any("_reconstruct" in n for n in names), names
    assert any(n.endswith(".ndarray") for n in names), names
    assert allow_numpy_pickles() == []                      # second call is a no-op


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
