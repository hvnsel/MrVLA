"""Tests for run_channel_attribution.py's candidate selection, and for the vectorisation
identity the streaming pass relies on.

The identity is the load-bearing one. run_attribution computes the alignment term
<w_j, g (*) u_contrast> with a per-row matvec inside a Python loop over every row; this pass
replaces that with a column gather from the contrast-centred signature matrix. If the two are
not identical, the slot-resolved C would not be comparable to the published C and the whole
comparison collapses. It is proved here against `mrvla.attribution.attribute` itself.

Run directly:
    python tests/test_channel_attribution.py
"""

from __future__ import annotations

import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mrvla.attribution import attribute, contrast_direction, rms  # noqa: E402
from mrvla.channels import transition_mask  # noqa: E402
from mrvla.readout import signature_matrix  # noqa: E402
from run_channel_attribution import pick_candidates, per_channel_transition  # noqa: E402


def test_signature_gather_equals_the_per_row_alignment_matvec():
    """THE vectorisation contract: column t of the contrast-centred signature IS the alignment
    term for a decision that emitted token t. Same numbers, no Python loop."""
    rng = np.random.default_rng(0)
    d, F, nbins = 40, 25, 16
    W_dec = rng.standard_normal((F, d))
    W_dec /= np.linalg.norm(W_dec, axis=1, keepdims=True)
    W_U = rng.standard_normal((nbins, d))
    g = np.abs(rng.normal(1.0, 0.1, d))

    S_raw = signature_matrix(W_dec, g, W_U, center=False)
    S_cen = S_raw - S_raw.mean(axis=1, keepdims=True)
    for t in (0, 5, nbins - 1):
        slow = W_dec @ (g * contrast_direction(W_U, t))
        assert np.allclose(S_cen[:, t], slow, atol=1e-12)


def test_gathered_phi_equals_attribute():
    """End to end on phi itself, including the l2/r factor, over many rows at once."""
    rng = np.random.default_rng(1)
    d, F, nbins, n = 40, 25, 16, 30
    W_dec = rng.standard_normal((F, d))
    W_dec /= np.linalg.norm(W_dec, axis=1, keepdims=True)
    W_U = rng.standard_normal((nbins, d))
    g = np.abs(rng.normal(1.0, 0.1, d))
    X = rng.standard_normal((n, d))
    z = np.zeros((n, F))
    for i in range(n):
        z[i, rng.choice(F, 4, replace=False)] = rng.gamma(2.0, 0.5, 4)
    l2 = np.abs(rng.normal(1.0, 0.2, n))
    tok = rng.integers(0, nbins, n)

    S_cen = signature_matrix(W_dec, g, W_U, center=True)
    r_scal = np.sqrt((X * X).mean(axis=1) + 1e-5)
    fast = (l2 / r_scal)[:, None] * z * S_cen.T[tok]

    for i in range(n):
        slow = attribute(z[i], W_dec, l2[i], rms(X[i], 1e-5), g,
                         contrast_direction(W_U, int(tok[i])))
        assert np.allclose(fast[i], slow, atol=1e-12)


def _attr_fixture(path, F=400, G=8, seed=0):
    rng = np.random.default_rng(seed)
    spread = rng.uniform(0.05, 1.0, F)
    mass = rng.gamma(1.2, 1.0, F)
    C = np.stack([rng.dirichlet(np.full(G, s * 4)) for s in spread], axis=1) * mass * G
    PR = (C.sum(0) ** 2) / (C ** 2).sum(0)
    mag = C.sum(0)
    base = np.clip(0.02 + 0.5 * mag / mag.max(), 0, 1)
    np.savez_compressed(path, C=C.astype(np.float32), task_ids=np.arange(G),
                        PR=PR.astype(np.float32), magnitude=mag.astype(np.float32),
                        base_rate=base.astype(np.float32),
                        is_active=(mag > 0).astype(np.uint8))


def test_candidate_groups_are_disjoint_where_they_must_be():
    """general/specialist/random must not overlap -- a feature in two groups would appear on
    both sides of the contrast. `firing` MAY overlap: it is the prior work's ranking and the
    interesting case is precisely when it picks the same features."""
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "attr.npz")
        _attr_fixture(p)
        cand = pick_candidates(p, top=20)
        gr = cand["groups"]
        assert len(gr["general"]) == len(gr["specialist"]) == 20
        for a, b in (("general", "specialist"), ("general", "random"), ("specialist", "random")):
            assert not (set(gr[a]) & set(gr[b])), f"{a} and {b} overlap"
        assert cand["features"] == sorted(set(cand["features"]))
        assert set(cand["features"]) == set().union(*(set(v) for v in gr.values()))


def test_candidates_split_the_breadth_ranking_at_the_two_ends():
    """The contrast has to be a contrast: generals must actually rank above specialists."""
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "attr.npz")
        _attr_fixture(p, seed=3)
        cand = pick_candidates(p, top=25)
        adj = cand["adjusted_breadth"]
        gen = [adj[j] for j in cand["groups"]["general"]]
        spec = [adj[j] for j in cand["groups"]["specialist"]]
        assert min(gen) > max(spec)


def test_candidate_selection_is_deterministic():
    """The random control is seeded: two runs must target the same features, or the counterfactual
    numbers cannot be compared across invocations."""
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "attr.npz")
        _attr_fixture(p, seed=5)
        assert pick_candidates(p, 15)["groups"] == pick_candidates(p, 15)["groups"]


def test_per_channel_transition_is_not_the_gripper_broadcast():
    """PINS THE P5d FIX. Every slot must be clocked by its OWN emitted bins.

    Before this, the mask came from the gripper alone and was broadcast with np.repeat, so dx's
    `_trans` counters answered "did dx flip at a timestep where the GRIPPER moved". dx changes
    on nearly every step and the gripper on ~5%, so that control selected the wrong 5% and the
    `_trans` family was not comparable across channels. The test asserts three things: dx gets
    dx's mask, the gripper's own numbers are UNCHANGED by the fix, and the old broadcast form
    gives a materially different answer for dx (otherwise the test would pass on the bug).
    """
    n, n_sl, grip = 40, 7, 6
    ep = np.zeros(n, dtype=np.int64)
    ts = np.arange(n, dtype=np.int64)
    tok = np.zeros((n, n_sl), dtype=np.int64)
    tok[:, 0] = np.arange(n)                       # dx changes at every timestep
    tok[:, grip] = (ts >= 20).astype(np.int64) + (ts >= 30).astype(np.int64)  # gripper: 2 changes
    tr = per_channel_transition(tok.reshape(-1), n, n_sl, ep, ts).reshape(n, n_sl)

    assert np.array_equal(tr[:, 0], transition_mask(tok[:, 0], ep, ts))
    assert np.array_equal(tr[:, grip], transition_mask(tok[:, grip], ep, ts))
    assert tr[:, 0].sum() == n - 1, tr[:, 0].sum()
    assert tr[:, grip].sum() == 2, tr[:, grip].sum()

    broadcast = np.repeat(transition_mask(tok[:, grip], ep, ts), n_sl).reshape(n, n_sl)
    assert broadcast[:, 0].sum() == 2, "fixture does not exercise the bug"
    assert tr[:, 0].sum() != broadcast[:, 0].sum(), "the fix is indistinguishable from the bug"
    assert np.array_equal(tr[:, grip], broadcast[:, grip]), \
        "the gripper's own transition numbers must not move -- only the other six channels"


def test_per_channel_transition_respects_episode_boundaries_and_row_order():
    """The first decision of an episode has no predecessor, and rows may arrive unsorted."""
    n, n_sl = 6, 3
    ep = np.array([0, 0, 0, 1, 1, 1], dtype=np.int64)
    ts = np.array([0, 1, 2, 0, 1, 2], dtype=np.int64)
    tok = np.tile(np.array([[5], [5], [9], [9], [4], [4]], dtype=np.int64), (1, n_sl))
    tr = per_channel_transition(tok.reshape(-1), n, n_sl, ep, ts).reshape(n, n_sl)
    # row 3 starts episode 1: a change from 9 to 9 across the boundary must NOT count
    assert list(tr[:, 0]) == [False, False, True, False, True, False]
    perm = np.array([4, 0, 5, 2, 1, 3])
    tr_p = per_channel_transition(tok[perm].reshape(-1), n, n_sl, ep[perm], ts[perm])
    assert np.array_equal(tr_p.reshape(n, n_sl)[:, 0], tr[perm, 0]), "order-dependent"


def test_all_features_bypass_keeps_the_groups_and_a_large_top_would_not():
    """WHY the bypass exists rather than just that it works.

    --all-features must widen the counterfactual to every feature while `pick_candidates` still
    returns usable groups, because analyze_channels.py's general/specialist/random tables read
    them out of summary.json. Raising --top instead cannot do this: once top reaches the
    eligible count, select_general_specialist slices the same ordering from both ends and the
    two groups stop being disjoint -- so the contrast silently stops being a contrast.
    """
    F = 400
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "attr.npz")
        _attr_fixture(p, F=F, seed=11)
        cand = pick_candidates(p, top=20)
        feats = np.arange(F, dtype=np.int64)                  # what --all-features produces
        assert feats.size == F and np.array_equal(feats, np.unique(feats))
        assert not (set(cand["groups"]["general"]) & set(cand["groups"]["specialist"]))
        assert set(cand["features"]).issubset(set(feats.tolist()))
        # with feats = arange(F), analyze_channels.py's {feature: row} map is the identity
        pos = {int(f): i for i, f in enumerate(feats)}
        assert all(pos[j] == j for j in cand["groups"]["general"])

        big = pick_candidates(p, top=F)
        assert set(big["groups"]["general"]) & set(big["groups"]["specialist"]), \
            "a large --top no longer breaks disjointness; the bypass rationale needs rewriting"


def test_bincount_bucket_equals_the_explicit_task_slot_loop():
    """THE counter refactor's contract, and its index convention.

    The inner loop used to do 7 slots x ~9 boolean reductions over the full [n*7] row axis per
    feature. Adding a task axis naively would make that 70 (task, slot) cells x 9 counters --
    ten times the work per feature, times five times the features. The replacement is one
    `bincount` over `rows_task * n_sl + slots`. It must produce EXACTLY the integers the loop
    did, and it must land them the right way round.

    G != S deliberately: with G == S a transposed reshape passes every assertion.
    """
    rng = np.random.default_rng(4)
    G, n_sl, n = 4, 7, 300
    rows_task = rng.integers(0, G, n).repeat(n_sl)
    slots = np.tile(np.arange(n_sl), n)
    mask = rng.random(n * n_sl) < 0.3
    shift = rng.normal(0, 3, n * n_sl)

    bucket = rows_task * n_sl + slots
    fast = np.bincount(bucket[mask], minlength=G * n_sl).reshape(G, n_sl)
    fast_w = np.bincount(bucket[mask], weights=shift[mask],
                         minlength=G * n_sl).reshape(G, n_sl)

    slow = np.zeros((G, n_sl), dtype=np.int64)
    slow_w = np.zeros((G, n_sl))
    for g in range(G):
        for s_i in range(n_sl):
            m = (rows_task == g) & (slots == s_i) & mask
            slow[g, s_i] = int(m.sum())
            slow_w[g, s_i] = float(shift[m].sum())
    assert np.array_equal(fast, slow), "bincount counters differ from the loop they replace"
    assert np.allclose(fast_w, slow_w, atol=1e-9)
    assert fast.shape == (G, n_sl) and G != n_sl, "fixture must be able to see a transpose"
    # the OPPOSITE convention -- mrvla.channels.accumulate_slot_task uses slots * G + tasks --
    # would put the same numbers in a transposed array. Two conventions live in one file, so
    # the distinction is pinned rather than commented.
    other = np.bincount((slots * G + rows_task)[mask], minlength=G * n_sl).reshape(n_sl, G)
    assert np.array_equal(other, slow.T)


def test_task_axis_sums_to_the_legacy_slot_marginal():
    """`flip_{mode}_{key}` stays [F, S] for analyze_channels.py and is derived by summing the
    new [F, G, S] counters over tasks. Every Step 5 necessity number is read from the 2-D form,
    so the identity is what keeps them reproducible."""
    rng = np.random.default_rng(6)
    F, G, S = 5, 4, 7
    gt = rng.integers(0, 50, (F, G, S))
    assert np.array_equal(gt.sum(axis=1), np.stack([gt[f].sum(axis=0) for f in range(F)]))
    assert gt.sum(axis=1).shape == (F, S)


if __name__ == "__main__":
    for name, fn in sorted(list(globals().items())):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("all channel-attribution tests passed")
