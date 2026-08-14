"""Tests for additive steering: the hook's arithmetic, condition construction, and video I/O.

Pins the things that would silently corrupt a steering run:
  1. the hook really computes h + alpha*w (checked against an explicit expression);
  2. gamma calibration resolves alpha from the ACTUAL residual scale and then FREEZES it --
     a magnitude that drifted between passes would make the intervention uninterpretable;
  3. baseline gets no direction, feature conditions get their own unit-norm row, random
     controls are unit-norm too (norm-matched, else the control is not a control);
  4. duplicate feature ids are rejected -- a repeat would double that feature's magnitude;
  5. the PIL GIF fallback writes a real playable file with no extra dependencies.

Run directly:
    python tests/test_steering.py
"""

from __future__ import annotations

import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mrvla.video import write_video
from run_steering import build_conditions, direction_for, parse_features, shard


def _torch():
    try:
        import torch
        return torch
    except ImportError:
        return None


def test_steerer_adds_alpha_times_direction():
    torch = _torch()
    if torch is None:
        print("    (skipped: no torch)"); return
    from mrvla.hooks import ActivationSteerer
    d, alpha = 8, 3.0
    layer = torch.nn.Identity()
    w = torch.zeros(1, d); w[0, 2] = 5.0                 # non-unit on purpose
    s = ActivationSteerer(layer, w, alphas=torch.tensor([alpha]))
    try:
        h = torch.zeros(1, 4, d)
        out = layer(h)
        expect = torch.zeros(1, 4, d); expect[..., 2] = alpha    # w normalised to unit
        assert torch.allclose(out, expect, atol=1e-6), out[0, 0]
    finally:
        s.remove()


def test_gamma_calibrates_from_residual_scale_then_freezes():
    torch = _torch()
    if torch is None:
        print("    (skipped: no torch)"); return
    from mrvla.hooks import ActivationSteerer
    d = 16
    layer = torch.nn.Identity()
    w = torch.zeros(1, d); w[0, 0] = 1.0
    s = ActivationSteerer(layer, w, gamma=2.0)
    try:
        h = torch.zeros(1, 3, d); h[..., 1] = 4.0        # every token has ||h|| = 4
        layer(h)
        assert abs(s.resolved_scale - 4.0) < 1e-5, s.resolved_scale
        first = float(s.resolved_alphas[0])
        assert abs(first - 8.0) < 1e-5, first            # gamma * ||h||
        # a later pass at a DIFFERENT scale must not move alpha
        h2 = torch.zeros(1, 3, d); h2[..., 1] = 400.0
        layer(h2)
        assert float(s.resolved_alphas[0]) == first, s.resolved_alphas
    finally:
        s.remove()


def test_steer_decode_passes_flag():
    torch = _torch()
    if torch is None:
        print("    (skipped: no torch)"); return
    from mrvla.hooks import ActivationSteerer
    d = 4
    layer = torch.nn.Identity()
    w = torch.zeros(1, d); w[0, 0] = 1.0
    s = ActivationSteerer(layer, w, alphas=torch.tensor([1.0]), steer_decode_passes=False)
    try:
        h = torch.zeros(1, 1, d)
        assert float(layer(h)[0, 0, 0]) == 1.0           # prefill steered
        assert float(layer(h)[0, 0, 0]) == 0.0           # decode pass untouched
        s.reset_step()
        assert float(layer(h)[0, 0, 0]) == 1.0           # counter reset -> steered again
    finally:
        s.remove()


def test_conditions_and_directions():
    rng = np.random.default_rng(0)
    W = rng.standard_normal((50, 12))
    conds = build_conditions([7, 9], n_random=1)
    assert list(conds) == ["baseline", "steer_7", "steer_9", "steer_random0"]
    assert direction_for(conds["baseline"], W, 0) is None
    v7 = direction_for(conds["steer_7"], W, 0)
    assert v7.shape == (1, 12)
    assert abs(np.linalg.norm(v7) - 1.0) < 1e-6
    assert np.allclose(v7[0], W[7] / np.linalg.norm(W[7]), atol=1e-6)
    vr = direction_for(conds["steer_random0"], W, 0)
    assert abs(np.linalg.norm(vr) - 1.0) < 1e-6          # norm-matched control
    assert not np.allclose(vr, v7)


def test_parse_features_rejects_duplicates_and_empty():
    assert parse_features("1167, 1235 ,1628") == [1167, 1235, 1628]
    for bad in ("", "  ", "5,5"):
        try:
            parse_features(bad)
        except SystemExit:
            continue
        raise AssertionError(f"expected SystemExit for {bad!r}")


def test_sharding_covers_every_job_once():
    jobs = [(c, t) for c in ["baseline", "a", "b"] for t in range(10)]
    W = 7
    seen = [j for w in range(W) for j in shard(jobs, w, W)]
    assert sorted(seen) == sorted(jobs)
    assert len(seen) == len(jobs)


def test_gif_fallback_writes_playable_file():
    """PIL is already a dependency, so video needs no new install."""
    from PIL import Image
    rng = np.random.default_rng(0)
    frames = [rng.integers(0, 255, (64, 64, 3), dtype=np.uint8) for _ in range(6)]
    with tempfile.TemporaryDirectory() as d:
        path, backend = write_video(frames, os.path.join(d, "ep"), fps=10,
                                    stride=2, max_side=32, prefer="pil")
        assert backend == "pil" and path.endswith(".gif")
        im = Image.open(path)
        assert im.n_frames == 3                       # stride=2 over 6 frames
        assert max(im.size) == 32                     # downscaled


def test_write_video_rejects_empty():
    try:
        write_video([], "/tmp/nope")
    except ValueError:
        return
    raise AssertionError("expected ValueError on empty frame list")


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
