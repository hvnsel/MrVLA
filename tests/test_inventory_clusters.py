"""Tests for inventory_clusters.py, and an end-to-end simulation of the A4 hypothesis itself.

The centrepiece is `test_planted_splitting_...`: it builds the exact scenario A4 proposes -- one
model holding roles whole, another having SPLIT its general roles across several features while
leaving specialist roles atomic -- and checks that the feature-level metric reproduces Path B's
finding ("generals recur less") while the inventory-level metric does not. If that simulation
did not behave this way, the whole reformulation would be unfalsifiable.

The centrepiece result is a CORRECTION to the natural assumption, established here rather than
guessed: the two kinds of dictionary splitting move the feature-level metric in opposite
directions, so only one of them can explain Path B.

  * DUPLICATION (fragments are noisy copies surrounding the role) gives one-to-one matching a
    best-of-N boost, so split features match BETTER. It cannot produce "generals recur less".
    Clustering recovers these roles, and occupancy carries the signature.
  * SPAN-splitting (fragments span the role without surrounding it) leaves every individual
    fragment a poor match, so `max_j cos` drops. This is the mechanism consistent with Path B --
    and clustering does NOT recover it, because the fragments' centroid is not the role. The
    m-sweep does.

Both directions are pinned below, because assuming the wrong one would mean reading a clustering
null as "the inventory does not recur either" when clustering was never the instrument for that
question.

Run directly:
    python tests/test_inventory_clusters.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inventory_clusters import cluster_breadth, spearman  # noqa: E402
from mrvla.clustering import (  # noqa: E402
    cluster_occupancy, match_inventories, normalize_rows, spherical_kmeans,
)
from mrvla.inventory import omp_curve, signed_best_match  # noqa: E402

D = 64
N_GEN = 6
N_SPEC = 6


def planted_models(rng, n_frag=3, frag_noise=0.12):
    """Model A holds every role in one feature. Model B splits the GENERAL roles into `n_frag`
    noisy copies each and leaves the SPECIALIST roles atomic.

    Fragments are role + noise (a positive role component each), which is what non-negative
    codes make likely: several features that each fire in a subset of contexts while all pushing
    the same direction. Their mean recovers the role; no single one of them does.
    """
    roles = normalize_rows(rng.standard_normal((N_GEN + N_SPEC, D)))[0]
    is_general = np.arange(N_GEN + N_SPEC) < N_GEN

    A = normalize_rows(roles + 0.02 * rng.standard_normal(roles.shape))[0]

    b_rows, b_role = [], []
    for r in range(N_GEN + N_SPEC):
        reps = n_frag if is_general[r] else 1
        for _ in range(reps):
            b_rows.append(roles[r] + frag_noise * rng.standard_normal(D))
            b_role.append(r)
    B = normalize_rows(np.asarray(b_rows))[0]
    return roles, A, B, np.asarray(b_role), is_general


def test_duplication_splitting_makes_split_features_match_BETTER_not_worse():
    """THE correction. Noisy-copy splitting hands one-to-one matching several chances, so the
    max improves. Duplication therefore cannot be the explanation for Path B's "generals recur
    less" -- an assumption worth killing before it is carried into the real analysis."""
    rng = np.random.default_rng(0)
    _roles, A, B, _br, is_general = planted_models(rng)
    q, _ = signed_best_match(A, B)
    assert q[is_general].mean() > q[~is_general].mean() + 0.02


def test_span_splitting_is_what_reproduces_the_path_b_pattern():
    """The other mechanism, and the only one consistent with the observed finding: fragments
    that span a role without surrounding it make every individual match poor."""
    rng = np.random.default_rng(11)
    n = 8
    roles = normalize_rows(rng.standard_normal((n, D)))[0]
    A = normalize_rows(roles + 0.02 * rng.standard_normal(roles.shape))[0]
    split_rows = []
    for r in range(n // 2):                      # first half split, second half atomic
        basis, _ = np.linalg.qr(np.column_stack([roles[r], rng.standard_normal((D, 2))]))
        mix = rng.standard_normal((3, 3))
        mix[:, 0] *= 0.3
        split_rows.append(mix @ basis[:, :3].T)
    B = normalize_rows(np.concatenate(split_rows + [roles[n // 2:]], axis=0))[0]
    q, _ = signed_best_match(A, B)
    is_split = np.arange(n) < n // 2
    assert q[is_split].mean() < q[~is_split].mean() - 0.2


def test_clustering_recovers_duplication_split_roles():
    """Clustering's actual job: when fragments surround a role, their centroid is the role, so
    the two inventories align even though the dictionaries differ in size and composition."""
    rng = np.random.default_rng(0)
    roles, A, B, _br, _is_general = planted_models(rng)
    k = N_GEN + N_SPEC
    C_a, _lab_a, _ = spherical_kmeans(A, k, np.random.default_rng(1))
    C_b, _lab_b, _ = spherical_kmeans(B, k, np.random.default_rng(1))
    res = match_inventories(C_a, C_b)
    assert res["hungarian_mean"] > 0.6
    # B has 50% more features than A, yet the inventories are the same size and align
    assert B.shape[0] > A.shape[0]


def test_occupancy_carries_the_splitting_signature():
    """Step three: 'same inventory, different multiplicities'. B must spend more of its
    dictionary on the roles it split, and that is measurable without any feature matching."""
    rng = np.random.default_rng(2)
    roles, A, B, b_role, is_general = planted_models(rng, n_frag=4)
    k = N_GEN + N_SPEC
    _C_a, lab_a, _ = spherical_kmeans(A, k, np.random.default_rng(3))
    _C_b, lab_b, _ = spherical_kmeans(B, k, np.random.default_rng(3))
    occ_a, occ_b = cluster_occupancy(lab_a, k), cluster_occupancy(lab_b, k)
    assert abs(occ_a.sum() - 1.0) < 1e-12 and abs(occ_b.sum() - 1.0) < 1e-12
    # B's occupancy is far more uneven, because four features sit where A puts one
    assert occ_b.std() > occ_a.std()
    # and the general roles really do carry more of B's budget
    gen_share = float(np.mean([np.mean(b_role[lab_b == j] < N_GEN) for j in range(k)
                               if (lab_b == j).any()]))
    assert gen_share > 0.4


def test_clustering_does_not_rescue_span_only_splitting():
    """The documented boundary. When fragments SPAN a role without surrounding it, their
    centroid is not the role and clustering cannot recover it -- but the m-sweep can. This is
    why inventory_recurrence.py and inventory_clusters.py are both run: they catch different
    splitting mechanisms."""
    rng = np.random.default_rng(4)
    roles = normalize_rows(rng.standard_normal((8, D)))[0]
    frags = []
    for r in range(8):
        basis, _ = np.linalg.qr(np.column_stack([roles[r], rng.standard_normal((D, 2))]))
        mix = rng.standard_normal((3, 3))
        mix[:, 0] *= 0.3                     # damp the role component: span it, do not surround it
        frags.append(mix @ basis[:, :3].T)
    B = normalize_rows(np.concatenate(frags, axis=0))[0]
    A = normalize_rows(roles + 0.02 * rng.standard_normal(roles.shape))[0]

    C_a, _, _ = spherical_kmeans(A, 8, np.random.default_rng(5))
    C_b, _, _ = spherical_kmeans(B, 8, np.random.default_rng(5))
    clustered = match_inventories(C_a, C_b)["hungarian_mean"]
    swept = omp_curve(A, B, m_max=3)[0][:, -1].mean()
    assert swept > clustered + 0.1, "the m-sweep should catch what clustering misses here"


def test_cluster_breadth_averages_members_and_handles_empty_roles():
    labels = np.array([0, 0, 1, 1, 2])
    adj = np.array([1.0, 3.0, 10.0, 20.0, 5.0])
    cb = cluster_breadth(labels, k=4, adj=adj)
    assert abs(cb[0] - 2.0) < 1e-12
    assert abs(cb[1] - 15.0) < 1e-12
    assert abs(cb[2] - 5.0) < 1e-12
    assert np.isnan(cb[3])                          # no members -> NaN, not 0


def test_cluster_breadth_ignores_inactive_features():
    """Inactive features carry NaN breadth; they must not drag a role's mean to NaN."""
    labels = np.array([0, 0, 0])
    adj = np.array([2.0, np.nan, 4.0])
    assert abs(cluster_breadth(labels, 1, adj)[0] - 3.0) < 1e-12


def test_spearman_guards_short_and_degenerate_input():
    assert np.isnan(spearman([1.0, 2.0], [1.0, 2.0]))
    assert abs(spearman([1.0, 2, 3, 4], [1.0, 2, 3, 4]) - 1.0) < 1e-12
    assert abs(spearman([1.0, 2, 3, 4], [4.0, 3, 2, 1]) + 1.0) < 1e-12
    assert np.isnan(spearman([1.0, 1, 1, 1], [1.0, 2, 3, 4]))


def test_the_driver_contrast_statistic_is_computable_end_to_end():
    """The headline comparison, computed exactly the way the driver computes it: role breadth
    against matched-centroid similarity, via the Hungarian pairing. Pins the plumbing -- pair
    indices, cluster breadth, and the correlation -- rather than a particular sign, which
    depends on which splitting mechanism the real data turns out to show."""
    rng = np.random.default_rng(6)
    _roles, A, B, _b_role, is_general = planted_models(rng)
    breadth = np.where(is_general, 1.0, 0.0)
    k = N_GEN + N_SPEC
    C_a, lab_a, _ = spherical_kmeans(A, k, np.random.default_rng(7))
    C_b, _lab_b, _ = spherical_kmeans(B, k, np.random.default_rng(7))
    res = match_inventories(C_a, C_b)
    hr = np.array([p[0] for p in res["hungarian_pairs"]])
    assert len(hr) == len(set(hr.tolist())) == k
    cb = cluster_breadth(lab_a, k, breadth)[hr]
    val = spearman(cb, res["hungarian_similarity"])
    assert np.isfinite(val) and -1.0 <= val <= 1.0


if __name__ == "__main__":
    for name, fn in sorted(list(globals().items())):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("all inventory_clusters tests passed")
