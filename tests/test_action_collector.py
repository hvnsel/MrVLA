"""Tests for ActionPositionCollector (mrvla/hooks.py), the Path A / Stage A1 hook.

The collector cannot be tested against the real model here, but its capture logic --
skip the prefill pass (sequence length > 1), keep the single new position of each decode
pass (length 1), and guard the expected count -- is exercised directly with fake tensors.

Run directly:
    python tests/test_action_collector.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mrvla.hooks import ActionPositionCollector  # noqa: E402


def _emit(module, S, d):
    """Drive a forward pass whose output is a tuple (hidden [1, S, d], ...)."""
    module((torch.randn(1, S, d),))


def test_captures_seven_passes_prefill_plus_decode():
    """generate(max_new_tokens=7) = 1 prefill (emits token 1) + 6 decode (tokens 2-7).
    We capture the last position of every pass -> exactly 7 residuals."""
    d = 16
    m = torch.nn.Identity()
    col = ActionPositionCollector(m, dtype=torch.float16)
    col.reset()
    _emit(m, 20, d)                 # prefill: last position emits action token 1
    for _ in range(6):
        _emit(m, 1, d)             # 6 decode passes emit tokens 2-7
    h7 = col.stack(expected=7)
    assert h7.shape == (7, d)
    col.remove()


def test_robust_to_uncached_growing_sequence():
    """With use_cache=False the decode passes reprocess a growing sequence, but the last
    position is still the new token, so we still capture exactly 7."""
    d = 8
    m = torch.nn.Identity()
    col = ActionPositionCollector(m)
    col.reset()
    for s in [20, 21, 22, 23, 24, 25, 26]:   # 7 passes, sequence grows each time
        _emit(m, s, d)
    h7 = col.stack(expected=7)
    assert h7.shape == (7, d)
    col.remove()


def test_guard_fires_on_wrong_count():
    d = 8
    m = torch.nn.Identity()
    col = ActionPositionCollector(m)
    col.reset()
    for _ in range(6):             # only 6 passes -> not 7
        _emit(m, 1, d)
    try:
        col.stack(expected=7)
        raised = False
    except RuntimeError:
        raised = True
    assert raised
    col.remove()


def test_empty_capture_errors():
    m = torch.nn.Identity()
    col = ActionPositionCollector(m)
    col.reset()                     # no forward passes at all
    try:
        col.stack(expected=7)
        raised = False
    except RuntimeError:
        raised = True
    assert raised
    col.remove()


def test_values_are_the_last_position():
    """The captured vector must be the LAST position of the decode output."""
    d = 5
    m = torch.nn.Identity()
    col = ActionPositionCollector(m, dtype=torch.float32)
    col.reset()
    known = torch.arange(d, dtype=torch.float32).reshape(1, 1, d)
    m((known,))                    # single decode position with known values
    h = col.stack(expected=1)
    assert np.allclose(h[0], np.arange(d))
    col.remove()


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
