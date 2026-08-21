"""Pins the prior-gate ladder: each gate must PASS on structure and FAIL on its own null.

A gate that cannot fail is decoration, so every gate here is tested twice -- once on a fixture
built to contain the thing it screens for, and once on a fixture built to contain the specific
confound it exists to catch. The three that matter most:

  * `test_gate1_fails_when_mu_tracks_r` -- the structural kill. If mu is proportional to r then
    mu/r is constant, the (l2/r) prefactor cancels the lever, and no later gate can rescue the
    story. This is the cheapest way the whole idea dies and it must be caught.
  * `test_gate3_fails_when_share_is_margin_in_disguise` -- the likeliest soft kill. Weak
    features -> unsure model -> deviates from the expert is close to definitional, so a
    bias-share that is a deterministic function of the top-2 margin must score ~0 once the
    margin is partialled out, or the gate is measuring confidence and calling it a prior.
  * `test_prior_split_reproduces_run_attribution_const` -- the algebra check. Gate 0's A/B
    split has to reproduce the exact `const` term that run_attribution.py:190 computes, since
    that is the line the published 0.405 came from. If this drifts, every number in the module
    is wrong in a way no other test would notice.

Run directly:
    python tests/test_prior_gates.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mrvla.prior_gates import (  # noqa: E402
    bias_share, demo_bin_index, gate0_bias_composition, gate1_mu_over_r,
    gate2_prior_vs_marginal, gate3_share_predicts_deviation, gate4_lambda_sweep,
    prior_scores, prior_vectors, rank_partial, spearman, verdict_table,
)

D, NBINS = 32, 16          # small d and few bins: the algebra is dimension-agnostic
RNG = lambda s=0: np.random.default_rng(s)  # noqa: E731


def _head(seed=0, d=D, n_bins=NBINS):
    """A synthetic readout head: (W_U_act, g, b_pre)."""
    r = RNG(seed)
    return (r.normal(size=(n_bins, d)), np.abs(r.normal(size=d)) + 0.5, r.normal(size=d) * 0.1)


# ---------------------------------------------------------------------------
# algebra
# ---------------------------------------------------------------------------
def test_prior_split_reproduces_run_attribution_const():
    """mu*A[t] + B[t] must equal (mu + b_pre) @ (g * u_c) exactly, for every bin.

    That right-hand side is run_attribution.py:190 verbatim -- the line that produced the
    reported bias share of 0.405. Broadcasting a scalar mu into a d-vector before the dot
    product is the step that makes the two forms identical, and it is exactly the step that a
    reimplementation gets wrong.
    """
    W_U, g, b_pre = _head(1)
    A, B = prior_vectors(W_U, g, b_pre)
    U_c = W_U - W_U.mean(axis=0, keepdims=True)
    for mu in (-0.3, 0.0, 0.7):
        for t in range(NBINS):
            gu = g * U_c[t]
            assert np.isclose(mu * A[t] + B[t], (mu + b_pre) @ gu, rtol=0, atol=1e-11), (mu, t)


def test_prior_scores_matches_elementwise_form():
    A, B = prior_vectors(*_head(2))
    mu = RNG(3).normal(size=25)
    P = prior_scores(mu, A, B)
    assert P.shape == (25, NBINS)
    assert np.allclose(P, mu[:, None] * A[None, :] + B[None, :])


def test_bias_share_is_r_invariant_and_bounded():
    """Every term carries the same 1/r, so the share must not move when r changes."""
    r = RNG(4)
    f, b, e = r.normal(size=200), r.normal(size=200), r.normal(size=200) * 0.1
    s1 = bias_share(f, b, e)
    scale = np.abs(r.normal(size=200)) + 0.5
    s2 = bias_share(f / scale, b / scale, e / scale)
    assert np.allclose(s1, s2, atol=1e-12)
    assert np.nanmin(s1) >= 0.0 and np.nanmax(s1) <= 1.0


def test_bias_share_nan_on_dead_decision():
    """A decision with no margin at all is dropped, not scored as 'all features'."""
    assert np.isnan(bias_share([0.0], [0.0], [0.0])[0])


def test_spearman_averages_ties():
    """P9's tie defect must not reappear here: integer bins tie constantly."""
    x = np.array([1, 1, 1, 2, 2, 3], float)
    assert np.isclose(spearman(x, x), 1.0)
    assert np.isclose(spearman(x, -x), -1.0)


def test_rank_partial_removes_a_planted_control():
    """y = c + noise, x = c + noise: correlated raw, ~0 once c is partialled out."""
    r = RNG(5)
    c = r.normal(size=800)
    x, y = c + 0.3 * r.normal(size=800), c + 0.3 * r.normal(size=800)
    assert spearman(x, y) > 0.7
    assert abs(rank_partial(y, x, [c])) < 0.15


# ---------------------------------------------------------------------------
# GATE 0
# ---------------------------------------------------------------------------
def _g0(n=4000, mu_sd=0.4, a_scale=1.0, b_scale=1.0, seed=0):
    r = RNG(seed)
    mu = r.normal(0.0, mu_sd, size=n)
    rr = np.abs(r.normal(3.0, 0.3, size=n)) + 1.0
    A_t = r.normal(0.0, 1.0, size=n) * a_scale
    B_t = r.normal(0.0, 1.0, size=n) * b_scale
    return mu, rr, A_t, B_t


def test_gate0_passes_when_mu_term_is_real():
    g = gate0_bias_composition(*_g0(mu_sd=1.5, b_scale=0.2, seed=10))
    assert g["pass"], g
    assert g["mu_share"] > 0.5
    assert g["frac_var_from_mu"] > 0.5


def test_gate0_fails_when_b_pre_dominates():
    """The degenerate case: the bias is a genuine constant and the lever is recalibration."""
    g = gate0_bias_composition(*_g0(mu_sd=0.002, b_scale=50.0, seed=11))
    assert not g["pass"], g
    assert g["mu_share"] < MU_TOL, g["mu_share"]


def test_gate0_fails_when_mu_is_constant():
    """mu present but frozen: freezing it changes nothing, so there is nothing to modulate."""
    mu, rr, A_t, B_t = _g0(seed=12)
    g = gate0_bias_composition(np.full_like(mu, 0.5), rr, A_t, B_t)
    assert not g["pass"], g
    assert abs(g["frac_var_from_mu"]) < 1e-9


def test_gate0_variance_attribution_finds_the_planted_factor():
    """With r and the bin frozen in the DATA, all bias variance must trace to mu."""
    n = 3000
    mu = RNG(13).normal(size=n)
    g = gate0_bias_composition(mu, np.ones(n) * 2.0, np.ones(n), np.ones(n))
    assert g["frac_var_from_mu"] > 0.99
    assert abs(g["frac_var_from_r"]) < 1e-9
    assert abs(g["frac_var_from_bin"]) < 1e-9


MU_TOL = 0.15


# ---------------------------------------------------------------------------
# GATE 1 -- the structural kill
# ---------------------------------------------------------------------------
def test_gate1_passes_when_mu_is_independent_of_r():
    r = RNG(20)
    g = gate1_mu_over_r(r.normal(2.0, 0.6, size=5000), np.abs(r.normal(3.0, 0.5, size=5000)) + 1)
    assert g["pass"], g
    assert g["mu_retained"] > 0.5


def test_gate1_fails_when_mu_tracks_r():
    """mu = c*r exactly. mu/r is then a constant, the prefactor cancels it, story over.

    This is the cheapest way the whole idea dies, and it is invisible to every other gate --
    Gate 0 would happily report a large, variable mu term on this same fixture.
    """
    rr = np.abs(RNG(21).normal(3.0, 0.5, size=5000)) + 1.0
    g = gate1_mu_over_r(0.7 * rr, rr)
    assert not g["pass"], g
    assert g["cv_mu_over_r"] < 1e-9
    assert g["pearson_mu_r"] > 0.99


def test_gate1_fails_on_near_proportionality_too():
    """Not just the exact case: mu = c*r + small noise must also fail."""
    r = RNG(22)
    rr = np.abs(r.normal(3.0, 0.5, size=5000)) + 1.0
    g = gate1_mu_over_r(0.7 * rr + 0.005 * r.normal(size=5000), rr)
    assert not g["pass"], g


def test_gate1_anticorrelation_is_favourable_not_an_error():
    """mu_retained > 1 when mu and r move oppositely. That is a pass, not a bug."""
    r = RNG(23)
    rr = np.abs(r.normal(3.0, 0.4, size=5000)) + 1.0
    g = gate1_mu_over_r(5.0 - 0.5 * rr, rr)
    assert g["mu_retained"] > 1.0, g
    assert g["pass"]


# ---------------------------------------------------------------------------
# GATE 2
# ---------------------------------------------------------------------------
def _emissions_from_prior(A, B, mu, n, seed, temp=1.0):
    """Sample emitted bins from a softmax over the prior itself: the marginal IS the prior."""
    r = RNG(seed)
    p = np.exp(temp * (np.median(mu) * A + B))
    p = p / p.sum()
    return r.choice(len(p), size=n, p=p)


def test_gate2_passes_when_marginal_is_generated_from_the_prior():
    W_U, g, b_pre = _head(30)
    A, B = prior_vectors(W_U, g, b_pre)
    mu = RNG(31).normal(0.5, 0.2, size=6000)
    rows = _emissions_from_prior(A, B, mu, 6000, 32, temp=3.0)
    res = gate2_prior_vs_marginal(A, B, mu, rows, np.tile(np.arange(7), 6000 // 7 + 1)[:6000],
                                  n_bins=NBINS)
    assert res["pass"], res
    assert res["median_slot_rho"] > 0.5


def test_gate2_fails_on_a_marginal_unrelated_to_the_prior():
    W_U, g, b_pre = _head(33)
    A, B = prior_vectors(W_U, g, b_pre)
    mu = RNG(34).normal(0.5, 0.2, size=6000)
    rows = RNG(35).integers(0, NBINS, size=6000)          # uniform: carries no prior structure
    res = gate2_prior_vs_marginal(A, B, mu, rows, np.zeros(6000, int), n_bins=NBINS)
    assert not res["pass"], res


def test_gate2_reports_a_single_argmax_when_mu_barely_moves():
    """The one-parameter family collapses to a fixed direction when mu is nearly constant."""
    W_U, g, b_pre = _head(36)
    A, B = prior_vectors(W_U, g, b_pre)
    mu = np.full(2000, 0.4) + RNG(37).normal(0, 1e-6, size=2000)
    res = gate2_prior_vs_marginal(A, B, mu, RNG(38).integers(0, NBINS, size=2000),
                                  np.zeros(2000, int), n_bins=NBINS)
    assert res["n_argmax_over_mu_range"] == 1, res


# ---------------------------------------------------------------------------
# GATE 3 -- the rebranding trap
# ---------------------------------------------------------------------------
def test_gate3_passes_when_share_drives_deviation_beyond_the_controls():
    r = RNG(40)
    n = 4000
    share = r.uniform(0, 1, size=n)
    margin = r.uniform(0, 1, size=n)                       # independent of share by construction
    activity = r.uniform(0, 1, size=n)
    dev = 20 * share + 3 * r.normal(size=n)
    res = gate3_share_predicts_deviation(share, dev, margin, activity)
    assert res["pass"], res
    assert res["partial_both"] > 0.4


def test_gate3_fails_when_share_is_margin_in_disguise():
    """share = f(margin) and deviation is driven by margin alone.

    Raw correlation is strong; the partial must collapse. This is the outcome I would bet on
    for the real data, and the gate is worthless if it cannot produce it.
    """
    r = RNG(41)
    n = 4000
    margin = r.uniform(0, 1, size=n)
    share = 1.0 - margin                                   # deterministic function of the control
    dev = 20 * (1.0 - margin) + 3 * r.normal(size=n)
    res = gate3_share_predicts_deviation(share, dev, margin, r.uniform(0, 1, size=n))
    assert res["raw_rho"] > 0.5, res
    assert abs(res["partial_both"]) < 0.1, res
    assert not res["pass"], res


def test_gate3_fails_when_the_partial_flips_sign():
    """A partial that reverses sign is a different effect, not a weakened one."""
    r = RNG(42)
    n = 3000
    c = r.uniform(0, 1, size=n)
    share = c + 0.05 * r.normal(size=n)
    dev = 10 * c - 4 * share + r.normal(size=n)            # suppressed: raw +, partial -
    res = gate3_share_predicts_deviation(share, dev, c, r.uniform(0, 1, size=n))
    if np.isfinite(res["raw_rho"]) and np.isfinite(res["partial_both"]):
        if np.sign(res["raw_rho"]) != np.sign(res["partial_both"]):
            assert not res["pass"], res


# ---------------------------------------------------------------------------
# GATE 4
# ---------------------------------------------------------------------------
def _g4(n=3000, seed=50, tail_only=True, n_bins=NBINS):
    """Feature/prior score matrices where the prior HELPS the bulk and HURTS the tail.

    That asymmetry is the whole hypothesis, so the fixture has to contain it rather than
    merely contain "the prior is wrong somewhere". Bulk rows get deliberately AMBIGUOUS
    features and a prior pointing at the expert bin, so the prior is carrying them; tail rows
    get confident features and a stronger prior pointing at the wrong bin, so the prior
    overrides them. Then lambda < 1 must hurt the bulk and rescue the tail.

    `tail_only=False` makes the prior wrong everywhere with confident features throughout --
    the global-miscalibration case, where lambda helps bulk and tail alike and the signature
    must NOT hold.
    """
    r = RNG(seed)
    demo_row = r.integers(0, n_bins, size=n)
    wrong_row = (demo_row + n_bins // 2) % n_bins
    tail = np.zeros(n, bool)
    tail[: n // 10] = True
    idx = np.arange(n)

    feat = r.normal(size=(n, n_bins)) * 0.5
    prior = np.zeros((n, n_bins))
    if tail_only:
        # bulk: features barely prefer the expert bin, the prior supplies the rest
        # tail: features know the answer, the prior shouts over them
        feat[idx, demo_row] += np.where(tail, 4.0, 0.3)
        prior[idx[~tail], demo_row[~tail]] = 3.0
        prior[idx[tail], wrong_row[tail]] = 6.0
    else:
        feat[idx, demo_row] += 4.0
        prior[idx, wrong_row] = 6.0

    share = np.where(tail, 0.9, 0.1) + 0.01 * r.normal(size=n)
    emitted_row = np.argmax(feat + prior, axis=1)
    return (feat, prior, n_bins - demo_row, n_bins - emitted_row, share)


def test_gate4_passes_when_the_prior_is_wrong_in_the_tail():
    res = gate4_lambda_sweep(*_g4(seed=51), n_bins=NBINS)
    assert res["pass"], res
    assert res["signature_holds"], res
    assert res["best_lambda"] < 1.0, res


def test_gate4_lambda_one_reproduces_the_emitted_action():
    """By construction the fixture's emitted bin IS argmax at lambda = 1; the canary must see it."""
    res = gate4_lambda_sweep(*_g4(seed=52), n_bins=NBINS)
    assert res["recon_agreement"] > 0.99, res["recon_agreement"]


def test_gate4_fails_when_the_prior_is_already_right():
    """Prior points at the expert bin too: no lambda can improve on lambda = 1."""
    feat, prior, demo_bin, emitted_bin, share = _g4(seed=53)
    demo_row = NBINS - demo_bin
    prior = np.zeros_like(prior)
    prior[np.arange(len(demo_row)), demo_row] = 2.0
    emitted_bin = NBINS - np.argmax(feat + prior, axis=1)
    res = gate4_lambda_sweep(feat, prior, demo_bin, emitted_bin, share, n_bins=NBINS)
    assert not res["pass"], res


def test_gate4_signature_fails_when_the_prior_is_uniformly_wrong():
    """Miscalibrated everywhere, not just in the tail -> global recalibration, not an adaptive
    rule. The sweep may still 'pass' on tail gain, but `signature_holds` must be False, which is
    what distinguishes the interesting story from the boring one."""
    res = gate4_lambda_sweep(*_g4(seed=54, tail_only=False), n_bins=NBINS)
    assert not res["signature_holds"], res


def test_gate4_uses_lambda_one_not_the_model_as_baseline():
    """The reported gains must be measured against lambda = 1, not against the emitted action.

    Shifting every emitted bin by a constant changes `emitted_mean_dev` but must leave the
    lambda comparison untouched, because the model's own argmax is not the baseline.
    """
    feat, prior, demo_bin, emitted_bin, share = _g4(seed=55)
    a = gate4_lambda_sweep(feat, prior, demo_bin, emitted_bin, share, n_bins=NBINS)
    b = gate4_lambda_sweep(feat, prior, demo_bin, emitted_bin * 0 + 1, share, n_bins=NBINS)
    assert np.isclose(a["best_tail_gain"], b["best_tail_gain"])
    assert not np.isclose(a["emitted_mean_dev"], b["emitted_mean_dev"])


# ---------------------------------------------------------------------------
# demo_bin_index -- the only piece matching an EXTERNAL convention we cannot check here
# ---------------------------------------------------------------------------
def _decode(bin_index, n_bins):
    """OpenVLA's ActionTokenizer decode: bin index -> normalised action value.

    `discretized = clip(bin_index - 1, 0, len(bin_centers) - 1); bin_centers[discretized]`,
    with bin_centers the midpoints of linspace(-1, 1, n_bins).
    """
    bins = np.linspace(-1.0, 1.0, n_bins)
    centers = (bins[:-1] + bins[1:]) / 2.0
    return centers[np.clip(np.asarray(bin_index) - 1, 0, centers.size - 1)]


def test_demo_bin_index_inverts_the_openvla_decode():
    """Encode(decode(b)) == b for every usable bin. This is the property the canary protects.

    A bin centre lies strictly inside its own interval, so digitize must return the bin it came
    from. An off-by-one in either direction breaks this for every b at once, which is exactly
    the failure the driver's canary is there to catch on real data.
    """
    n_bins = 256
    b = np.arange(1, n_bins)                       # [1, 255]: the range decode can represent
    norm = _decode(b, n_bins)
    got = demo_bin_index(norm[:, None], np.array([-1.0]), np.array([1.0]),
                         np.array([False]), n_bins)[:, 0]
    assert np.array_equal(got, b), np.flatnonzero(got != b)[:5]


def test_demo_bin_index_normalises_only_masked_dims():
    """Masked dims are rescaled by (q01, q99); unmasked ones (the gripper) pass through raw.

    The range must NOT be (-1, 1) here -- that is the identity transform, so masked and
    unmasked would agree and the test would pass without testing anything.
    """
    q01, q99 = np.array([0.0, 0.0]), np.array([2.0, 2.0])
    a = np.array([[0.5, 0.5]])                     # masked -> -0.5 ; unmasked -> +0.5
    out = demo_bin_index(a, q01, q99, np.array([True, False]), 256)
    assert out[0, 0] < out[0, 1], out
    both = demo_bin_index(a, q01, q99, np.array([True, True]), 256)
    assert both[0, 0] == both[0, 1] == out[0, 0], (both, out)


def test_demo_bin_index_clips_out_of_range_actions():
    """An expert action outside [q01, q99] must land on an edge bin, never out of bounds."""
    q01, q99 = np.array([0.0]), np.array([1.0])
    lo = demo_bin_index(np.array([[-5.0]]), q01, q99, np.array([True]), 256)[0, 0]
    hi = demo_bin_index(np.array([[5.0]]), q01, q99, np.array([True]), 256)[0, 0]
    assert 1 <= lo <= 256 and 1 <= hi <= 256, (lo, hi)
    assert lo < hi


def test_demo_bin_index_degenerate_range_does_not_divide_by_zero():
    """q01 == q99 happens for a constant action dim; it must not produce nan or inf."""
    out = demo_bin_index(np.array([[0.5]]), np.array([1.0]), np.array([1.0]),
                         np.array([True]), 16)
    assert np.isfinite(out).all()


# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------
def test_verdict_table_marks_everything_after_the_first_failure_as_moot():
    gates = {
        "gate0": {"pass": True, "mu_share": 0.4},
        "gate1": {"pass": False, "mu_retained": 0.01},
        "gate2": {"pass": True, "median_slot_rho": 0.5},
    }
    txt = verdict_table(gates)
    assert "moot" in txt.split("\n")[2], txt
    assert "moot" not in txt.split("\n")[0], txt
    assert "NOT RUN" in txt.split("\n")[3]


if __name__ == "__main__":
    for name, fn in sorted(list(globals().items())):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("all prior_gates tests passed")
