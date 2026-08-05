"""Tests for cross-model recurrence (mrvla/cross_model_recurrence.py).

Synthetic setup mirrors the real design: 3 "models" each produce a code matrix on
the SAME shared frames, with three archetypes of feature:

    universal   -- driven by a shared latent signal present in every model, at a
                   moderate base rate.  Should recur across models (high q_cross).
    specific    -- independent random signal per model.  Reproduces within a model
                   across SAE seeds, but NOT across models (low q_cross, big drop).
    busy        -- fires on almost every frame in every model but is otherwise
                   independent.  High base rate makes it correlate with other busy
                   features by accident (inflated raw q_cross); the base-rate
                   control must strip this so it does not masquerade as general.

The point: recurrence separates universal from specific, and the base-rate residual
prevents busy features from being mistaken for general (principle #2).

Run directly:
    python tests/test_cross_model_recurrence.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mrvla.cross_model_recurrence import (  # noqa: E402
    base_rate,
    base_rate_residual,
    column_correlations,
    cross_model_q,
    greedy_q,
    recurrence_report,
    summarize,
)

N_FRAMES = 400
N_UNIV, N_SPEC, N_BUSY = 20, 20, 20
F = N_UNIV + N_SPEC + N_BUSY
UNIV = slice(0, N_UNIV)
SPEC = slice(N_UNIV, N_UNIV + N_SPEC)
BUSY = slice(N_UNIV + N_SPEC, F)


def _relu(x):
    return np.maximum(x, 0.0)


def build_model_codes(shared_latent, spec_latent, busy_latent, rng, extra_noise=0.0):
    """One model's [N_frames, F] code matrix on the shared frames.

    shared_latent : [N_frames, N_UNIV] latents common to ALL models (the reusable
                    computation -> universal features recur across models).
    spec_latent   : [N_frames, N_SPEC] latents PRIVATE to this model (specific
                    features -> reproduce within a model across SAE seeds, but do
                    not recur across models).
    busy_latent   : [N_frames, N_BUSY] private, high-offset -> high base rate.
    extra_noise   : extra SAE-style noise; used to emulate a second SAE seed of the
                    SAME model (same latents, different dictionary noise).
    """
    def read(latent, offset=0.0):
        return _relu(latent + 0.15 * rng.standard_normal(latent.shape)
                     + extra_noise * rng.standard_normal(latent.shape) + offset)
    Z = np.zeros((N_FRAMES, F), dtype=np.float32)
    Z[:, UNIV] = read(shared_latent)
    Z[:, SPEC] = read(spec_latent)
    Z[:, BUSY] = read(busy_latent, offset=2.0)     # active on almost every frame
    return Z


def build_scenario(seed=0):
    """Return (codes_by_model, shared_latent, private_latents, rng).

    ``private_latents[m]`` = (spec_latent, busy_latent) for model m, so a second
    seed of a model can re-extract the SAME underlying features.
    """
    rng = np.random.default_rng(seed)
    shared_latent = rng.standard_normal((N_FRAMES, N_UNIV))     # common to all
    codes, priv = {}, {}
    for m in ("A", "B", "C"):
        spec = rng.standard_normal((N_FRAMES, N_SPEC))          # private per model
        busy = rng.standard_normal((N_FRAMES, N_BUSY))          # private per model
        priv[m] = (spec, busy)
        codes[m] = build_model_codes(shared_latent, spec, busy, rng)
    return codes, shared_latent, priv, rng


def build_second_seed(model, shared_latent, priv, rng):
    """A second SAE seed of ``model``: same latents, extra dictionary noise."""
    spec, busy = priv[model]
    return build_model_codes(shared_latent, spec, busy, rng, extra_noise=0.1)


# ---------------------------------------------------------------------------
def test_column_correlations_shape_and_range():
    rng = np.random.default_rng(0)
    A = rng.standard_normal((100, 5))
    B = rng.standard_normal((100, 7))
    Cc = column_correlations(A, B)
    assert Cc.shape == (5, 7)
    assert np.all(Cc <= 1.0 + 1e-6) and np.all(Cc >= -1.0 - 1e-6)
    # a column correlated with itself gives 1
    Cself = column_correlations(A, A)
    assert np.allclose(np.diag(Cself), 1.0, atol=1e-6)


def test_dead_column_correlates_zero():
    rng = np.random.default_rng(0)
    A = rng.standard_normal((50, 3))
    B = np.zeros((50, 2))                          # constant (dead) columns
    Cc = column_correlations(A, B)
    assert np.allclose(Cc, 0.0)


def test_universal_features_recur_across_models():
    codes, _, _, _ = build_scenario()
    q = cross_model_q(codes["A"], [codes["B"], codes["C"]])
    assert q[UNIV].mean() > q[SPEC].mean() + 0.3     # clear separation
    assert q[UNIV].mean() > 0.6                        # universal genuinely recur
    assert q[SPEC].mean() < 0.4                        # specific do not


def test_recurrence_is_not_fooled_by_base_rate():
    """Unlike firing metrics, continuous-correlation recurrence is not confounded by
    base rate: busy-but-independent features have high base rate yet do NOT recur."""
    codes, _, _, _ = build_scenario()
    rep = recurrence_report(codes, target="A")
    q, br = rep["q_cross"], rep["base_rate"]

    # busy features really are high base rate ...
    assert br[BUSY].mean() > 0.9
    assert br[BUSY].mean() > br[UNIV].mean()
    # ... but they do NOT get high recurrence (correlation ignores the shared offset)
    assert q[BUSY].mean() < 0.4
    assert q[UNIV].mean() > q[BUSY].mean() + 0.3
    # raw recurrence is not positively driven by base rate here
    assert rep["spearman_qcross_baserate"] <= 0.1
    # the base-rate control is reported as a safeguard and does not invert the ranking
    resid = base_rate_residual(q, br, rep["is_active"])
    assert np.nanmean(resid[UNIV]) > np.nanmean(resid[BUSY])


def test_noise_floor_retention_separates_universal_from_specific():
    """With a same-model second seed, retention (cross/seed) is high for universal,
    low for specific -- even though both reproduce well within-model across seeds."""
    codes, shared_latent, priv, rng = build_scenario(seed=1)
    seedA = build_second_seed("A", shared_latent, priv, rng)
    rep = recurrence_report(codes, target="A", seed2=seedA)

    # both universal and specific reproduce within-model across seeds (q_seed high)
    assert rep["q_seed"][UNIV].mean() > 0.6
    assert rep["q_seed"][SPEC].mean() > 0.5
    # but cross-model retention is high for universal, low for specific
    assert np.nanmean(rep["retention"][UNIV]) > np.nanmean(rep["retention"][SPEC]) + 0.3
    # specific features drop far more from seed->cross than universal do
    assert rep["drop"][SPEC].mean() > rep["drop"][UNIV].mean() + 0.2


def test_hungarian_matches_greedy_on_clear_signal():
    codes, _, _, _ = build_scenario()
    qg = cross_model_q(codes["A"], [codes["B"]], method="greedy")
    qh = cross_model_q(codes["A"], [codes["B"]], method="hungarian")
    # on universal features both methods should recover a strong match
    assert qg[UNIV].mean() > 0.6 and qh[UNIV].mean() > 0.5


def test_summarize_runs():
    codes, shared_latent, priv, rng = build_scenario(seed=2)
    seedA = build_second_seed("A", shared_latent, priv, rng)
    rep = recurrence_report(codes, target="A", seed2=seedA)
    s = summarize(rep)
    assert s["n_active"] > 0
    assert np.isfinite(s["q_cross_mean"])
    assert "retention_mean" in s


# ---------------------------------------------------------------------------
def main():
    tests = [(name, fn) for name, fn in sorted(globals().items())
             if name.startswith("test_") and callable(fn)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
        except AssertionError as e:
            failed += 1
            print(f"  FAIL  {name}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
