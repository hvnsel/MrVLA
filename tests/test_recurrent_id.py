"""Tests for identify_recurrent_features.recurrence_beyond.

The one property that matters: a feature that recurs strongly but is INHERITED (high
inheritance) must be demoted, while a feature that recurs strongly and is NOT inherited (low
inheritance) must rank at the top. That is the whole point -- it answers "who's to say it
isn't the base model?".

Run directly:
    python tests/test_recurrent_id.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from identify_recurrent_features import recurrence_beyond    # noqa: E402


def test_inherited_recurrence_is_demoted():
    rng = np.random.default_rng(0)
    F = 200
    base_rate = rng.uniform(0.01, 0.9, F)
    inheritance = rng.uniform(0.0, 1.0, F)
    # q_cross is driven mostly by inheritance + a little base rate (the trivial recurrence)
    q_cross = 0.2 + 0.6 * inheritance + 0.15 * base_rate + rng.normal(0, 0.02, F)
    # inject a RE-DERIVED feature: recurs strongly, but low inheritance and low base rate
    j_star = 11
    inheritance[j_star] = 0.05
    base_rate[j_star] = 0.1
    q_cross[j_star] = 0.85
    # inject an INHERITED-recurrent feature: high q_cross but ALSO high inheritance
    j_inh = 42
    inheritance[j_inh] = 0.98
    q_cross[j_inh] = 0.88
    active = np.ones(F, bool)

    score = recurrence_beyond(q_cross, base_rate, inheritance, active)
    # the re-derived feature should be the single top-scoring feature
    assert int(np.nanargmax(score)) == j_star
    # the inherited-recurrent feature, despite the HIGHEST raw q_cross, must NOT rank top
    top5 = np.argsort(np.where(np.isfinite(score), score, -np.inf))[::-1][:5]
    assert j_inh not in top5, "inherited-recurrent feature leaked into the top re-derived set"


def test_without_inheritance_removes_only_base_rate():
    rng = np.random.default_rng(1)
    F = 100
    base_rate = rng.uniform(0.01, 0.9, F)
    q_cross = 0.5 + 0.4 * base_rate + rng.normal(0, 0.02, F)  # recurrence == activity
    active = np.ones(F, bool)
    score = recurrence_beyond(q_cross, base_rate, None, active)  # no inheritance available
    # with the activity confound removed, the residual should not correlate with base rate
    m = np.isfinite(score)
    r = np.corrcoef(np.argsort(np.argsort(score[m])), np.argsort(np.argsort(base_rate[m])))[0, 1]
    assert abs(r) < 0.2


def test_guard_too_few_active():
    q = np.array([0.5, 0.6, 0.7])
    assert np.isnan(recurrence_beyond(q, q, None, np.ones(3, bool))).all()


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
