"""Test the A x B join recovers the dissociation hypothesis when it's present in the data.

Construct synthetic features where, by design:
  general (high breadth) features are inherited (high inheritance)
  specialist (low breadth) features recur but are NOT inherited (re-derived)
and confirm the two headline correlations come out with the predicted signs.

Run directly:
    python tests/test_join.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from identify_features import adjusted_breadth
from identify_recurrent_features import recurrence_beyond
from mrvla.structural_generality import _spearman


def _synth(seed=0):
    """Build (PR, magnitude, base_rate, q_cross, inheritance, active) obeying the hypothesis."""
    rng = np.random.default_rng(seed)
    F, G = 300, 10
    C = np.zeros((G, F))
    breadth_like = np.zeros(F)
    for j in range(F):
        if j % 2 == 0:                      # broad (general): 10 tasks, total mag ~9
            C[:, j] = rng.uniform(0.85, 0.95, G); breadth_like[j] = 1.0
        else:                               # narrow (specialist): 1 task, total mag ~9 (MATCHED)
            C[rng.integers(0, G), j] = rng.uniform(8.5, 9.5); breadth_like[j] = 0.0
    s1, s2 = C.sum(0), (C * C).sum(0)
    PR = np.where(s2 > 0, s1 * s1 / s2, np.nan)
    magnitude = s1
    base_rate = rng.uniform(0.02, 0.9, F)
    gen = breadth_like > 0.5
    # HYPOTHESIS baked in: general -> high inheritance; specialist -> low inheritance
    inheritance = np.where(gen, rng.uniform(0.55, 1.0, F), rng.uniform(0.0, 0.25, F))
    # generals recur BECAUSE inherited (q tracks inheritance tightly, sits ON the line);
    # specialists recur DESPITE low inheritance (high q at low inh -> ABOVE the line), so
    # after residualising q on inheritance the specialists carry the positive residual.
    q_cross = np.where(gen,
                       0.10 + 0.70 * inheritance,             # on the inheritance line
                       rng.uniform(0.55, 0.75, F))            # high q, low inh -> above it
    q_cross += rng.normal(0, 0.02, F)
    active = np.ones(F, bool)
    return PR, magnitude, base_rate, q_cross, inheritance, active


def test_join_recovers_dissociation():
    PR, mag, br, q, inh, active = _synth()
    adj = adjusted_breadth(PR, mag, br, active)
    rec_beyond = recurrence_beyond(q, br, inh, active)
    m = active & np.isfinite(adj) & np.isfinite(rec_beyond)
    c_inh = _spearman(adj[m], inh[m])
    c_red = _spearman(adj[m], rec_beyond[m])
    # prediction 1: general features are inherited -> positive
    assert c_inh > 0.2, f"corr(breadth, inheritance) should be > 0, got {c_inh}"
    # prediction 2: specialists carry the re-derived recurrence -> negative
    assert c_red < -0.1, f"corr(breadth, recurrence-beyond-inheritance) should be < 0, got {c_red}"


def test_null_when_no_dissociation():
    """If inheritance is unrelated to breadth, the breadth-inheritance corr should be ~0."""
    rng = np.random.default_rng(3)
    F, G = 300, 10
    C = np.zeros((G, F))
    for j in range(F):
        if j % 2 == 0:
            C[:, j] = rng.uniform(0.7, 1.3, G)
        else:
            C[rng.integers(0, G), j] = rng.uniform(6, 10)
    s1, s2 = C.sum(0), (C * C).sum(0)
    PR = np.where(s2 > 0, s1 * s1 / s2, np.nan)
    inheritance = rng.uniform(0, 1, F)      # unrelated to breadth
    br = rng.uniform(0.02, 0.9, F)
    active = np.ones(F, bool)
    adj = adjusted_breadth(PR, C.sum(0), br, active)
    m = np.isfinite(adj)
    assert abs(_spearman(adj[m], inheritance[m])) < 0.2


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
