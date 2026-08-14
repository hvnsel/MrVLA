"""Tests for the offline logic of the ablation experiment.

Covers what must be right BEFORE burning GPU-hours: coalition construction (are the four
coalitions disjoint-where-they-should-be, load-bearing, and correctly ranked?), the
worker-sharding maths (does every job run exactly once across N workers?), and the analysis
statistics (damage participation ratio + the scope test).

Run directly:
    python tests/test_ablation.py
"""

from __future__ import annotations

import json
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from run_ablation import build_coalitions                     # noqa: E402
from analyze_ablation import participation_ratio, _pearson    # noqa: E402


def _make_attr(path, F=40, G=10, seed=0):
    """Synthetic attribution npz: some features broad, some narrow, some weak."""
    rng = np.random.default_rng(seed)
    C = np.zeros((G, F))
    for j in range(F):
        if j < 10:            # broad + strong  (should be 'general')
            C[:, j] = rng.uniform(0.8, 1.2, G)
        elif j < 20:          # narrow + strong (should be 'specialist')
            C[rng.integers(0, G), j] = rng.uniform(9, 11)
        else:                 # weak            (must NOT be selected)
            C[:, j] = rng.uniform(0.0, 0.02, G)
    s1, s2 = C.sum(axis=0), (C * C).sum(axis=0)
    PR = np.where(s2 > 0, s1 * s1 / s2, np.nan)
    magnitude = s1
    base_rate = rng.uniform(0.01, 0.9, F)
    np.savez(path, C=C.astype(np.float32), PR=PR.astype(np.float32),
             magnitude=magnitude.astype(np.float32), base_rate=base_rate.astype(np.float32),
             is_active=np.ones(F, np.uint8), task_ids=np.arange(G))
    return C


def test_coalitions_are_load_bearing_and_sized():
    with tempfile.TemporaryDirectory() as tmp:
        pth = os.path.join(tmp, "attr.npz")
        C = _make_attr(pth)
        built = build_coalitions(pth, top=5, seed=0)
        co = built["coalitions"]
        assert set(co) == {"general", "specialist", "random", "firing"}
        for name in ("general", "specialist", "random"):
            assert len(co[name]) == 5, name
        # general/specialist/random must all be load-bearing (never the weak features 20+)
        for name in ("general", "specialist", "random"):
            for j in co[name]:
                assert j < 20, f"{name} picked weak feature {j}"
        # general and specialist must be disjoint
        assert not (set(co["general"]) & set(co["specialist"]))


def test_general_is_broader_than_specialist():
    with tempfile.TemporaryDirectory() as tmp:
        pth = os.path.join(tmp, "attr.npz")
        _make_attr(pth)
        built = build_coalitions(pth, top=5, seed=0)
        prg = np.mean(built["info"]["general"]["PR"])
        prs = np.mean(built["info"]["specialist"]["PR"])
        assert prg > prs, f"general PR {prg} should exceed specialist PR {prs}"
        # the per-task profile of a specialist coalition must be concentrated
        prof_g = np.array(built["info"]["general"]["per_task_profile"])
        prof_s = np.array(built["info"]["specialist"]["per_task_profile"])
        assert participation_ratio(prof_g) > participation_ratio(prof_s)


def test_worker_sharding_covers_every_job_once():
    """Round-robin shard must partition the job list exactly (no gaps, no duplicates)."""
    conds = ["baseline", "general", "specialist", "random", "firing"]
    n_tasks = 10
    jobs = [(c, t) for c in conds for t in range(n_tasks)]
    for n_workers in (1, 3, 4, 7):
        seen = []
        for w in range(n_workers):
            seen += [j for i, j in enumerate(jobs) if i % n_workers == w]
        assert sorted(seen) == sorted(jobs), n_workers
        assert len(seen) == len(set(seen)) == len(jobs), n_workers


def test_damage_participation_ratio_scope():
    """Broad damage -> high PR; damage confined to one task -> PR = 1."""
    broad = np.array([0.4] * 10)
    narrow = np.array([0.9] + [0.0] * 9)
    assert abs(participation_ratio(broad) - 10.0) < 1e-6
    assert abs(participation_ratio(narrow) - 1.0) < 1e-6
    # negative "damage" (task improved) must not create phantom breadth
    mixed = np.array([0.5, -0.3] + [0.0] * 8)
    assert abs(participation_ratio(np.clip(mixed, 0, None)) - 1.0) < 1e-6


def test_scope_correlation_matches_attribution():
    """If damage lands exactly where attribution said, correlation ~ 1."""
    prof = np.array([5.0, 0.1, 0.1, 3.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0])
    damage = prof * 0.1                      # damage proportional to predicted profile
    assert _pearson(damage, prof) > 0.99
    unrelated = np.array([0.0, 4.0, 0.0, 0.0, 5.0, 0.0, 0.0, 3.0, 0.0, 0.0])
    assert _pearson(unrelated, prof) < 0.5



def test_parse_feature_specs_named_and_singletons():
    """Named sets and per-feature singletons, the two shapes a hand-picked run needs."""
    from run_ablation import parse_feature_specs
    got = parse_feature_specs(["grasp=12,45", "lid=77"], "1167,1140")
    assert got == {"grasp": [12, 45], "lid": [77],
                   "only_1167": [1167], "only_1140": [1140]}


def test_parse_feature_specs_bare_list_is_custom_and_backward_compatible():
    from run_ablation import parse_feature_specs
    assert parse_feature_specs(["12,45,900"], None) == {"custom": [12, 45, 900]}
    assert parse_feature_specs(None, None) == {}
    assert parse_feature_specs(None, " ") == {}


def test_parse_feature_specs_rejects_collisions_and_empties():
    from run_ablation import parse_feature_specs
    for bad in (lambda: parse_feature_specs(["a=1", "a=2"], None),      # duplicate name
                lambda: parse_feature_specs(["only_5=1"], "5"),          # collides with each
                lambda: parse_feature_specs(["a="], None),               # no ids
                lambda: parse_feature_specs(["=1,2"], None)):            # no name
        try:
            bad()
        except SystemExit:
            continue
        raise AssertionError("expected SystemExit")


def test_singleton_conditions_are_one_feature_each():
    """The whole point of --ablate-each: never bundle features that carry separate
    predictions, or the per-feature damage cannot be attributed."""
    from run_ablation import parse_feature_specs
    got = parse_feature_specs(None, "1167,1235,1628,1999,1140,1134")
    assert len(got) == 6
    assert all(len(v) == 1 for v in got.values()), got


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
