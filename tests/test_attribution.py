"""Tests for Path A attribution + gate + participation ratio (mrvla/attribution.py).

The load-bearing test is `test_attribution_sums_to_true_contrast_logit`: it proves the
frozen-r per-feature decomposition is EXACT for a perfectly-reconstructed residual, which
is the mathematical claim Path A rests on. The rest cover the gate logic, the PR values
against hand computation, and the activity-escape property (a hard-firing but orthogonal
feature contributes nothing).

Run directly:
    python tests/test_attribution.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mrvla.attribution import (  # noqa: E402
    action_logits,
    attribute,
    contrast_direction,
    gate_level1,
    gate_level2,
    participation_ratio,
    per_task_importance,
    reconstruct,
    rms,
    rmsnorm,
    total_magnitude,
)


def _toy_model(seed=0, d=16, F=12, n_act=8, k=4):
    """A small synthetic SAE + readout for which we control everything."""
    rng = np.random.default_rng(seed)
    W_dec = rng.standard_normal((F, d))
    W_dec /= np.linalg.norm(W_dec, axis=1, keepdims=True)   # unit-norm decoder rows
    b_pre = rng.standard_normal(d) * 0.1
    W_U_act = rng.standard_normal((n_act, d))
    g = rng.uniform(0.8, 1.2, d)                            # final-norm gain
    return dict(rng=rng, d=d, F=F, n_act=n_act, k=k, W_dec=W_dec, b_pre=b_pre,
                W_U_act=W_U_act, g=g)


def _make_decision(m):
    """Build a residual h that the SAE reconstructs EXACTLY (h == h_hat), plus its code."""
    rng = m["rng"]
    z = np.zeros(m["F"])
    active = rng.choice(m["F"], m["k"], replace=False)
    z[active] = rng.uniform(0.5, 2.0, m["k"])
    l2 = float(rng.uniform(1.0, 3.0))
    mu = float(rng.uniform(-0.3, 0.3))
    # h defined so reconstruct() returns it exactly
    h = l2 * (z @ m["W_dec"]) + mu + m["b_pre"]
    return h, z, l2, mu


# ---------------------------------------------------------------------------
def test_rms_and_rmsnorm_basic():
    h = np.array([1.0, 0.0, 2.0, 1.0])
    assert abs(rms(h, eps=0.0) - np.sqrt(1.5)) < 1e-9
    hn = rmsnorm(h, g=np.ones(4), eps=0.0)
    assert np.allclose(hn, h / np.sqrt(1.5))


def test_attribution_sums_to_true_contrast_logit():
    """THE core claim: for an exactly-reconstructed h, the frozen-r per-feature phi plus
    the constant term reproduce the true contrasted logit of the emitted token."""
    m = _toy_model(seed=1)
    for _ in range(20):
        h, z, l2, mu = _make_decision(m)
        eps = 1e-5
        r = rms(h, eps)
        tok_row = int(m["rng"].integers(m["n_act"]))
        u_c = contrast_direction(m["W_U_act"], tok_row)

        phi = attribute(z, m["W_dec"], l2, r, m["g"], u_c)
        gu = m["g"] * u_c
        const_part = float((mu + m["b_pre"]) @ gu / r)
        recon = float(phi.sum()) + const_part

        hn = rmsnorm(h, m["g"], eps)
        true_contrast = float(hn @ u_c)
        assert abs(recon - true_contrast) < 1e-8, (recon, true_contrast)


def test_activity_escape_orthogonal_feature_contributes_nothing():
    """A feature that fires hard but whose direction is orthogonal to g(*)u_contrast
    contributes exactly zero -- the property firing metrics could never capture."""
    d = 8
    g = np.ones(d)
    # build u_contrast, then a decoder direction orthogonal to g(*)u_contrast
    W_U_act = np.zeros((4, d)); W_U_act[0, 0] = 1.0     # so u_contrast points ~e0
    u_c = contrast_direction(W_U_act, 0)
    gu = g * u_c
    w_busy = np.zeros(d); w_busy[1] = 1.0               # orthogonal to gu (which is ~e0)
    w_busy /= np.linalg.norm(w_busy)
    w_driver = gu / np.linalg.norm(gu)                 # aligned with gu
    W_dec = np.stack([w_busy, w_driver])
    z = np.array([100.0, 1.0])                          # busy fires 100x harder
    phi = attribute(z, W_dec, l2=1.0, r=1.0, g=g, u_contrast=u_c)
    assert abs(phi[0]) < 1e-9                           # busy feature: zero contribution
    assert phi[1] > 0.1                                 # weak driver: real contribution


def test_gate1_passes_on_exact_reconstruction():
    """If h == h_hat exactly, the reconstruction must re-decode to the same action."""
    m = _toy_model(seed=2)
    H, Z, L2, MU, TR = [], [], [], [], []
    for _ in range(30):
        h, z, l2, mu = _make_decision(m)
        H.append(h); Z.append(z); L2.append(l2); MU.append(mu)
        TR.append(int(np.argmax(action_logits(h, m["W_U_act"], m["g"]))))
    res = gate_level1(H, Z, L2, MU, TR, m["W_dec"], m["b_pre"], m["W_U_act"], m["g"])
    assert res["action_match"] == 1.0
    assert res["mean_logit_corr"] > 0.999
    assert res["pass"]


def test_gate1_fails_on_scrambled_reconstruction():
    """If the SAE codes are unrelated to h, the reconstruction should NOT re-decode the
    same action, and the gate should fail."""
    m = _toy_model(seed=3)
    H, Z, L2, MU, TR = [], [], [], [], []
    for _ in range(40):
        h, z, l2, mu = _make_decision(m)
        H.append(h); L2.append(l2); MU.append(mu); TR.append(0)
        # scramble the code so h_hat is unrelated to h
        z2 = np.zeros_like(z); z2[m["rng"].choice(m["F"], m["k"], replace=False)] = \
            m["rng"].uniform(0.5, 2.0, m["k"])
        Z.append(z2)
    res = gate_level1(H, Z, L2, MU, TR, m["W_dec"], m["b_pre"], m["W_U_act"], m["g"])
    assert res["action_match"] < 0.85
    assert not res["pass"]


def test_gate2_high_correlation_on_exact_reconstruction():
    m = _toy_model(seed=4)
    H, Z, L2, MU, TR = [], [], [], [], []
    for _ in range(30):
        h, z, l2, mu = _make_decision(m)
        H.append(h); Z.append(z); L2.append(l2); MU.append(mu)
        TR.append(int(m["rng"].integers(m["n_act"])))
    res = gate_level2(H, Z, L2, MU, TR, m["W_dec"], m["b_pre"], m["W_U_act"], m["g"])
    assert res["logit_recon_corr"] > 0.999
    assert res["mean_abs_discrepancy"] < 1e-8


def test_participation_ratio_hand_values():
    """PR matches the worked examples in the design doc (4 tasks)."""
    C = np.array([
        [10.0, 10.0, 10.0, 10.0],   # feat 0: even over 4 -> PR 4
        [10.0,  0.0,  0.0,  0.0],   # feat 1: one task    -> PR 1
        [10.0, 10.0,  0.0,  0.0],   # feat 2: two tasks   -> PR 2
        [10.0,  5.0,  1.0,  1.0],   # feat 3: mostly one  -> PR ~2.276
    ]).T                             # per_task expects [G, F]
    pr = participation_ratio(C)
    assert abs(pr[0] - 4.0) < 1e-9
    assert abs(pr[1] - 1.0) < 1e-9
    assert abs(pr[2] - 2.0) < 1e-9
    assert abs(pr[3] - 2.276) < 1e-3


def test_participation_ratio_scale_free():
    """Doubling all importances leaves PR unchanged (it measures breadth, not strength)."""
    C = np.array([[3.0, 1.0, 0.0], [2.0, 2.0, 2.0]]).T
    assert np.allclose(participation_ratio(C), participation_ratio(2.0 * C),
                       equal_nan=True)


def test_per_task_importance_aggregation():
    # 4 decisions, 2 features, tasks [0,0,1,1]; |phi| means per task
    phi = np.array([[1.0, -3.0], [3.0, 1.0], [-2.0, 0.0], [4.0, 2.0]])
    task = np.array([0, 0, 1, 1])
    C, ids = per_task_importance(phi, task, n_features=2)
    assert list(ids) == [0, 1]
    assert np.allclose(C[0], [2.0, 2.0])   # task 0: mean(|1|,|3|)=2 ; mean(|3|,|1|)=2
    assert np.allclose(C[1], [3.0, 1.0])   # task 1: mean(|2|,|4|)=3 ; mean(|0|,|2|)=1
    assert np.allclose(total_magnitude(C), [5.0, 3.0])


# ---------------------------------------------------------------------------
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
