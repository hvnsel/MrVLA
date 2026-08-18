"""Tests for action_space_geometry.py -- the A4 Stage 0 gate.

This script's output decides which A4 design is run, so the rank statistics have to be right on
matrices whose rank we know by construction, and the branch recommendation has to flip at the
right place. The shared-head check is pinned too: a false "identical" there would let a
cross-model comparison run without the alignment it needs.

Run directly:
    python tests/test_action_space_geometry.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from action_space_geometry import (  # noqa: E402
    array_digest, bin_order_note, compare_heads, contrast_center, effective_rank,
    random_subspace_cosine, rank_at_energy, recommend_branch, spectrum_report,
)


def test_effective_rank_of_a_flat_spectrum_is_the_true_rank():
    assert abs(effective_rank(np.ones(10)) - 10.0) < 1e-9
    assert abs(effective_rank(np.array([1.0, 0, 0, 0])) - 1.0) < 1e-9
    # one dominant direction plus noise -> close to 1
    assert effective_rank(np.array([10.0] + [0.01] * 50)) < 1.1


def test_rank_at_energy_counts_directions_not_values():
    sv = np.array([2.0, 1.0, 1.0])          # energies 4, 1, 1 -> total 6
    assert rank_at_energy(sv, 0.60) == 1     # 4/6 = .667 >= .60
    assert rank_at_energy(sv, 0.80) == 2     # 5/6 = .833
    assert rank_at_energy(sv, 0.99) == 3
    assert rank_at_energy(np.zeros(5), 0.9) == 0


def test_spectrum_report_recovers_a_planted_rank():
    """A matrix built with exactly 5 independent directions must report ~5, whatever its shape."""
    rng = np.random.default_rng(0)
    basis = rng.standard_normal((5, 64))
    M = rng.standard_normal((256, 5)) @ basis          # 256 rows, true rank 5
    rep = spectrum_report(M)
    assert rep["rank_99pct_energy"] == 5
    assert 3.0 < rep["effective_rank"] < 5.5


def test_contrast_centering_removes_exactly_the_mean_direction():
    """The centered matrix must have zero column means, and centering can cost at most one rank."""
    rng = np.random.default_rng(1)
    basis = rng.standard_normal((6, 32))
    M = rng.standard_normal((256, 6)) @ basis + 5.0    # a big shared offset
    C = contrast_center(M)
    assert np.abs(C.mean(axis=0)).max() < 1e-10
    raw_r, cen_r = spectrum_report(M)["rank_99pct_energy"], spectrum_report(C)["rank_99pct_energy"]
    assert cen_r >= raw_r - 1


def test_ordered_discretization_really_is_low_rank():
    """The motivating intuition, made concrete: unembedding rows generated as a smooth function
    of an ordered bin index span far fewer than 256 directions. If this test's construction did
    NOT come out low-rank, the whole premise of Stage 0 would be wrong."""
    rng = np.random.default_rng(2)
    t = np.linspace(-1, 1, 256)[:, None]
    # a smooth curve through 4096-dim space, plus a little noise
    coeffs = rng.standard_normal((4, 4096))
    M = np.concatenate([t ** 0, t ** 1, t ** 2, t ** 3], axis=1) @ coeffs
    M += 0.001 * rng.standard_normal(M.shape)
    rep = spectrum_report(contrast_center(M))
    assert rep["effective_rank"] < 10
    assert rep["rank_90pct_energy"] <= 4


def test_random_subspace_cosine_is_m_over_r_and_saturates():
    assert abs(random_subspace_cosine(8, 2) - 0.25) < 1e-12
    assert abs(random_subspace_cosine(8, 4) - 0.50) < 1e-12
    assert random_subspace_cosine(8, 20) == 1.0            # cannot exceed the whole space
    assert np.isnan(random_subspace_cosine(0, 2))


def test_branch_recommendation_flips_with_the_geometry():
    """The decision this script exists to make, read off a spectrum report."""
    wide = recommend_branch({"effective_rank": 40.0, "rank_99pct_energy": 60})
    assert wide["max_usable_m"] >= 5 and wide["branch"].startswith("m-sweep")
    assert wide["confidence"] == "high"

    degenerate = recommend_branch({"effective_rank": 1.5, "rank_99pct_energy": 1})
    assert degenerate["max_usable_m"] < 2
    assert "distributional" in degenerate["branch"] and degenerate["confidence"] == "high"

    tight = recommend_branch({"effective_rank": 8.0, "rank_99pct_energy": 8})
    assert 2 <= tight["max_usable_m"] < 5 and tight["confidence"] == "high"


def test_branch_defers_when_the_two_rank_estimates_disagree():
    """The failure mode the bracket exists to prevent: a heavy-headed spectrum drags the energy
    participation ratio toward 1 while the 99%-energy rank is in the hundreds. Routing the whole
    project off either number alone would be a coin flip, so an honest gate must decline."""
    amb = recommend_branch({"effective_rank": 1.9, "rank_99pct_energy": 192})
    assert amb["confidence"] == "low"
    assert "AMBIGUOUS" in amb["branch"]
    assert amb["max_usable_m_pessimistic"] < amb["max_usable_m_optimistic"]
    # and it must still name a concrete m to run to, not just shrug
    assert "m<=" in amb["branch"]


def test_branch_survives_a_missing_spectrum():
    out = recommend_branch({"effective_rank": float("nan"), "rank_99pct_energy": 0})
    assert out["branch"] == "UNKNOWN"


def test_shared_head_check_detects_a_real_difference():
    """LoRA trains only attention projections, so the four heads SHOULD be identical -- but the
    check has to be able to say no, or it is worthless."""
    rng = np.random.default_rng(3)
    W = rng.standard_normal((256, 64))
    g = rng.standard_normal(64)
    ids = np.arange(31744, 32000)
    same = {"goal": {"W_U_act": W, "g": g, "act_ids": ids},
            "spatial": {"W_U_act": W.copy(), "g": g.copy(), "act_ids": ids.copy()}}
    cmp = compare_heads(same)
    assert all(cmp["identical"].values())
    assert cmp["max_abs_diff"]["W_U_act"] == 0.0

    W2 = W.copy()
    W2[0, 0] += 1e-3
    diff = {"goal": {"W_U_act": W, "g": g, "act_ids": ids},
            "spatial": {"W_U_act": W2, "g": g, "act_ids": ids}}
    cmp = compare_heads(diff)
    assert not cmp["identical"]["W_U_act"]
    assert cmp["identical"]["g"]
    assert abs(cmp["max_abs_diff"]["W_U_act"] - 1e-3) < 1e-9


def test_digest_is_content_addressed_not_object_addressed():
    a = np.arange(10.0)
    assert array_digest(a) == array_digest(a.copy())
    b = a.copy(); b[3] += 1e-12
    assert array_digest(a) != array_digest(b)


def test_bin_order_note_flags_the_reversed_axis():
    """token_id = vocab_size - bin_index, act_ids stored ascending -> row axis runs backwards."""
    note = bin_order_note(np.arange(31744, 32000))
    assert note["n_bins"] == 256
    assert note["ascending_ids"]
    assert note["row_axis_is_reversed_vs_bin_index"]


if __name__ == "__main__":
    for name, fn in sorted(list(globals().items())):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("all action_space_geometry tests passed")
