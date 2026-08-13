"""Tests for mrvla.model_utils.build_inputs.

Pins the attention_mask invariant. OpenVLA's predict_action() appends the SentencePiece empty
token (29871) to input_ids but does NOT extend an attention_mask handed to it, so a mask built
before the append is one token short of the multimodal sequence and Llama attention dies with
"The size of tensor a (279) must match the size of tensor b (278)". build_inputs must therefore
never hand a mask downstream. This bug has surfaced twice; the test exists so it cannot a third
time.

Run directly:
    python tests/test_model_utils.py
"""

from __future__ import annotations

import os
import sys
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _stub_transformers() -> None:
    """mrvla.model_utils imports transformers at module level, but build_inputs itself is pure
    prompt/tensor plumbing. Stub the heavy dependency when it is absent so this regression test
    still runs on a CPU-only box instead of silently skipping -- a skipped guard is exactly how
    this bug would slip back in."""
    try:
        import transformers  # noqa: F401
    except ImportError:
        stub = types.ModuleType("transformers")
        stub.AutoModelForVision2Seq = object
        stub.AutoProcessor = object
        sys.modules["transformers"] = stub


_stub_transformers()


class _FakeBatch(dict):
    """Stands in for the processor's BatchFeature: dict-like with a chainable .to()."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.moved = None

    def to(self, device=None, dtype=None):
        self.moved = (device, dtype)
        return self


class _FakeProcessor:
    def __init__(self):
        self.last_prompt = None
        self.last_image = None

    def __call__(self, prompt, image):
        self.last_prompt, self.last_image = prompt, image
        return _FakeBatch(input_ids="IDS", pixel_values="PIX", attention_mask="MASK")


def _build(instruction="Pick up the BLACK bowl."):
    from mrvla.model_utils import build_inputs
    proc = _FakeProcessor()
    out = build_inputs(proc, image="IMG", instruction=instruction, device="cpu")
    return proc, out


def test_attention_mask_is_dropped():
    """The whole point: a mask built before the 29871 append desyncs the sequence length."""
    _proc, out = _build()
    assert "attention_mask" not in out, out.keys()


def test_required_inputs_survive():
    _proc, out = _build()
    assert out["input_ids"] == "IDS"
    assert out["pixel_values"] == "PIX"


def test_prompt_is_normalized_and_device_applied():
    proc, out = _build("Pick up the BLACK bowl.")
    # instruction lowercased, trailing period stripped, wrapped in the OpenVLA template
    assert "pick up the black bowl?" in proc.last_prompt
    assert "BLACK" not in proc.last_prompt
    assert proc.last_prompt.endswith("\nOut:")
    assert out.moved is not None and out.moved[0] == "cpu"      # .to(device, dtype) applied


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
