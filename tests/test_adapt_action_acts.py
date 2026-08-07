"""Tests for the A1->A2 adapter (adapt_action_acts_for_sae.py).

Confirms the flattening (7 action positions -> 7*N samples), that values are preserved,
and that the emitted manifest matches what train_sae expects.

Run directly:
    python tests/test_adapt_action_acts.py
"""

from __future__ import annotations

import json
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from adapt_action_acts_for_sae import adapt  # noqa: E402


def _make_a1(dirpath, shard_sizes=(4, 3), d=8):
    os.makedirs(dirpath, exist_ok=True)
    rng = np.random.default_rng(0)
    all_res = []
    for i, n in enumerate(shard_sizes):
        res = rng.standard_normal((n, 7, d)).astype(np.float16)
        all_res.append(res)
        np.savez(os.path.join(dirpath, f"shard_{i:05d}.npz"),
                 residual=res, token_ids=rng.integers(31744, 32000, (n, 7)))
    json.dump({"model_name": "m", "layer": 31, "hidden_dim": d},
              open(os.path.join(dirpath, "manifest.json"), "w"))
    return np.concatenate(all_res, axis=0)      # [N, 7, d]


def test_flatten_count_and_manifest():
    with tempfile.TemporaryDirectory() as tmp:
        a1, a2 = os.path.join(tmp, "in"), os.path.join(tmp, "out")
        _make_a1(a1, shard_sizes=(4, 3), d=8)
        info = adapt(a1, a2)
        assert info["total_samples"] == (4 + 3) * 7      # 49
        man = json.load(open(os.path.join(a2, "manifest.json")))
        assert man["layer_indices"] == [31]
        assert man["hidden_dim"] == 8
        assert man["total_samples"] == 49


def test_values_preserved_row_order():
    """Flattening must lay out the 7 positions of decision i contiguously and in order."""
    with tempfile.TemporaryDirectory() as tmp:
        a1, a2 = os.path.join(tmp, "in"), os.path.join(tmp, "out")
        res = _make_a1(a1, shard_sizes=(4, 3), d=8)     # [7, 7, 8]
        adapt(a1, a2)
        import glob
        acts = np.concatenate(
            [np.load(p)["acts"] for p in sorted(glob.glob(os.path.join(a2, "shard_*.npz")))],
            axis=0)                                      # [49, 1, 8]
        assert acts.shape == (49, 1, 8)
        # decision 0's 7 slots are rows 0..6; decision 1's are 7..13; etc.
        flat = res.reshape(-1, 8)                        # [49, 8]
        assert np.allclose(acts[:, 0, :].astype(np.float32), flat.astype(np.float32))


def test_dim_mismatch_raises():
    with tempfile.TemporaryDirectory() as tmp:
        a1, a2 = os.path.join(tmp, "in"), os.path.join(tmp, "out")
        os.makedirs(a1)
        np.savez(os.path.join(a1, "shard_00000.npz"),
                 residual=np.zeros((3, 7, 8), np.float16))
        json.dump({"model_name": "m", "layer": 31, "hidden_dim": 16},   # says 16, data is 8
                  open(os.path.join(a1, "manifest.json"), "w"))
        try:
            adapt(a1, a2); raised = False
        except ValueError:
            raised = True
        assert raised


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
