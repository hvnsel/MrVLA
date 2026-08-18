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
from mrvla.readout import signature_matrix  # noqa: E402
from run_channel_attribution import pick_candidates  # noqa: E402


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


if __name__ == "__main__":
    for name, fn in sorted(list(globals().items())):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("all channel-attribution tests passed")
