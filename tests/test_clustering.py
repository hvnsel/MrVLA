"""Tests for mrvla.clustering -- the distributional half of A4.

The Hungarian solver is the piece most worth pinning: it is written from scratch to avoid a
scipy dependency, an incorrect assignment would quietly inflate or deflate every inventory
number, and EXPERIMENT_PLAN.md §3.1 carries a standing commitment to report it. It is checked
against brute-force enumeration over all permutations on small matrices, including rectangular
and degenerate cases.

Run directly:
    python tests/test_clustering.py
"""

from __future__ import annotations

import itertools
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mrvla.clustering import (  # noqa: E402
    cluster_occupancy, greedy_match, hungarian_match, match_inventories, normalize_rows,
    sliced_wasserstein, spherical_kmeans,
)


def brute_force_min(cost):
    """Optimal assignment by enumeration -- the ground truth for small matrices."""
    n, m = cost.shape
    if n <= m:
        best = min(itertools.permutations(range(m), n), key=lambda c: sum(cost[i, c[i]]
                                                                         for i in range(n)))
        return sum(cost[i, best[i]] for i in range(n))
    best = min(itertools.permutations(range(n), m), key=lambda r: sum(cost[r[j], j]
                                                                      for j in range(m)))
    return sum(cost[best[j], j] for j in range(m))


def test_hungarian_matches_brute_force_on_square_matrices():
    rng = np.random.default_rng(0)
    for _ in range(25):
        n = int(rng.integers(2, 7))
        cost = rng.standard_normal((n, n))
        r, c = hungarian_match(cost)
        assert abs(cost[r, c].sum() - brute_force_min(cost)) < 1e-9


def test_hungarian_matches_brute_force_on_rectangular_matrices():
    """Inventories need not be the same size -- k may differ, or a model may have fewer roles."""
    rng = np.random.default_rng(1)
    for _ in range(25):
        n, m = int(rng.integers(2, 6)), int(rng.integers(2, 7))
        cost = rng.standard_normal((n, m))
        r, c = hungarian_match(cost)
        assert len(r) == min(n, m)
        assert abs(cost[r, c].sum() - brute_force_min(cost)) < 1e-9


def test_hungarian_never_reuses_a_column_or_a_row():
    rng = np.random.default_rng(2)
    cost = rng.standard_normal((5, 8))
    r, c = hungarian_match(cost)
    assert len(set(r.tolist())) == len(r)
    assert len(set(c.tolist())) == len(c)


def test_hungarian_recovers_a_planted_permutation():
    """The case the inventory analysis actually runs: B's roles are A's, reordered."""
    rng = np.random.default_rng(3)
    k = 6
    perm = rng.permutation(k)
    sim = np.full((k, k), 0.1)
    sim[np.arange(k), perm] = 0.95
    r, c = hungarian_match(-sim)
    assert list(c[np.argsort(r)]) == list(perm)


def test_greedy_and_hungarian_diverge_when_one_target_is_popular():
    """Why the solver is worth writing. Greedy lets several rows claim the same centroid, which
    overstates inventory agreement; Hungarian forces a one-to-one reading."""
    sim = np.array([[0.9, 0.2, 0.1],
                    [0.85, 0.3, 0.15],
                    [0.8, 0.25, 0.2]])
    res = match_inventories(np.eye(3), np.eye(3))     # placeholder shapes, recompute directly
    gr, gc = greedy_match(sim)
    assert len(set(gc.tolist())) == 1                 # all three grabbed column 0
    hr, hc = hungarian_match(-sim)
    assert len(set(hc.tolist())) == 3                 # forced apart
    assert sim[gr, gc].mean() > sim[hr, hc].mean()    # greedy looks better than it is
    assert res["n_clusters"] == 3


def test_match_inventories_reports_both_rules_and_the_reuse_count():
    a = normalize_rows(np.array([[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]]))[0]
    b = normalize_rows(np.array([[0, 1.0, 0], [1.0, 0, 0], [0, 0, 1.0]]))[0]
    res = match_inventories(a, b)
    assert abs(res["hungarian_mean"] - 1.0) < 1e-9
    assert res["greedy_distinct_targets"] == 3
    assert len(res["hungarian_pairs"]) == 3


def test_spherical_kmeans_recovers_planted_directions():
    rng = np.random.default_rng(4)
    d, k = 24, 5
    centers = normalize_rows(rng.standard_normal((k, d)))[0]
    X = np.repeat(centers, 40, axis=0) + 0.06 * rng.standard_normal((k * 40, d))
    X = normalize_rows(X)[0]
    C, lab, inertia = spherical_kmeans(X, k, np.random.default_rng(0))
    # every planted centre must be matched by some found centroid
    best = (np.abs(centers @ C.T)).max(axis=1)
    assert best.min() > 0.95
    assert len(np.unique(lab)) == k
    assert inertia > 0.9 * X.shape[0]


def test_spherical_kmeans_is_scale_free():
    """Signature magnitude is feature strength, not role -- rescaling rows must not move the
    clustering."""
    rng = np.random.default_rng(5)
    X = normalize_rows(rng.standard_normal((120, 16)))[0]
    C1, l1, _ = spherical_kmeans(X, 4, np.random.default_rng(1))
    scaled = X * rng.uniform(0.1, 10.0, X.shape[0])[:, None]
    C2, l2, _ = spherical_kmeans(normalize_rows(scaled)[0], 4, np.random.default_rng(1))
    assert np.allclose(C1, C2)
    assert np.array_equal(l1, l2)


def test_spherical_kmeans_keeps_k_clusters_when_asked():
    """Empty clusters are reseeded, not dropped: occupancy comparisons across models depend on
    both sides really having k roles."""
    rng = np.random.default_rng(6)
    X = normalize_rows(np.repeat(rng.standard_normal((2, 12)), 30, axis=0))[0]
    C, lab, _ = spherical_kmeans(X, 6, np.random.default_rng(2))
    assert C.shape[0] == 6
    assert np.allclose(np.linalg.norm(C, axis=1), 1.0)


def test_occupancy_is_a_normalised_share():
    occ = cluster_occupancy(np.array([0, 0, 0, 1, 2, 2]), k=4)
    assert abs(occ.sum() - 1.0) < 1e-12
    assert np.allclose(occ, [0.5, 1 / 6, 1 / 3, 0.0])


def test_occupancy_detects_splitting_as_a_multiplicity_difference():
    """The concrete form of 'same inventory, different multiplicities': one model spends three
    features on a role the other covers with one."""
    a = cluster_occupancy(np.array([0, 1, 2, 3]), k=4)
    b = cluster_occupancy(np.array([0, 0, 0, 1, 2, 3]), k=4)
    assert b[0] > a[0]
    assert abs(a.sum() - b.sum()) < 1e-12


def test_sliced_wasserstein_is_zero_for_identical_clouds_and_grows_with_separation():
    rng = np.random.default_rng(7)
    A = normalize_rows(rng.standard_normal((200, 12)))[0]
    assert sliced_wasserstein(A, A.copy(), np.random.default_rng(0)) < 1e-9
    B = normalize_rows(rng.standard_normal((150, 12)) + 3.0)[0]   # a shifted, smaller cloud
    far = sliced_wasserstein(A, B, np.random.default_rng(0))
    near = sliced_wasserstein(A, normalize_rows(A + 0.05 * rng.standard_normal(A.shape))[0],
                              np.random.default_rng(0))
    assert far > near > 0


def test_sliced_wasserstein_tolerates_unequal_sample_sizes():
    """Dictionaries can differ in size; the distance must not depend on padding them."""
    rng = np.random.default_rng(8)
    A = normalize_rows(rng.standard_normal((300, 10)))[0]
    B = normalize_rows(rng.standard_normal((77, 10)))[0]
    v = sliced_wasserstein(A, B, np.random.default_rng(0))
    assert np.isfinite(v) and v >= 0


if __name__ == "__main__":
    for name, fn in sorted(list(globals().items())):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("all clustering tests passed")
