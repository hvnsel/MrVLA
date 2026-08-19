"""End-to-end test of control_linearity.py's verdict logic.

The unit tests in test_rankbasis.py pin the estimator. This pins the DECISION the driver makes
on top of it, which is the part a reader of the paper actually relies on: given a full LOTO
matrix, does it call the control plane adequate when it is, and inadequate when it is not?

The verdict is scored over the whole basis ladder rather than at its richest rung, and the
curved fixture is why: `quad` moves the number by ~0.18 there while `tensor4` moves it by
~0.02, because a basis with enough degrees of freedom partly re-fits the curvature it just
removed. Judging on the richest basis alone would return the wrong verdict on this fixture.

Run directly:
    python tests/test_control_linearity.py
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from control_linearity import analyse  # noqa: E402


def _args(**kw):
    d = dict(richest="tensor4", n_placebo=8, n_perm=40, n_bins=5, min_cell=20, tol=0.05, seed=0)
    d.update(kw)
    return argparse.Namespace(**d)


def _spread(k, m, G, rng):
    """Give feature f a total mass of m[f] split evenly across k[f] randomly chosen tasks.

    This makes the two quantities under test EXACT rather than approximate: participation ratio
    is k[f] and total magnitude is m[f], with no estimation noise in either.
    """
    C = np.zeros((G, len(m)))
    for f in range(len(m)):
        C[rng.choice(G, k[f], replace=False), f] = m[f] / k[f]
    return C


def _clean(G=10, F=2048, seed=0):
    """Breadth drawn independently of magnitude: real signal, no curvature to find."""
    rng = np.random.default_rng(seed)
    m = np.exp(rng.standard_normal(F))
    return _spread(rng.integers(1, G + 1, F), m, G, rng), np.round(rng.random(F) * 200) / 200


def _curved(G=10, F=2048, seed=1):
    """Breadth is a CUBIC function of the magnitude rank -- monotone, so a linear control plane
    absorbs some of it, but the curvature it cannot represent leaks into the residuals."""
    rng = np.random.default_rng(seed)
    m = np.exp(rng.standard_normal(F))
    u = np.argsort(np.argsort(m)) / (F - 1)
    k = np.clip(np.round(1 + (G - 1) * u ** 3).astype(int), 1, G)
    return _spread(k, m, G, rng), np.round(rng.random(F) * 200) / 200


def test_clean_data_is_called_adequate():
    C, br = _clean()
    res = analyse("clean", C, br, _args())
    assert res["verdict"] == "curvature below tolerance", res["verdict"]
    assert res["max_excess"] < 0.05, res["max_excess"]
    # and no basis moved the answer: the whole ladder agrees to within noise
    partials = [r["partial"] for r in res["rows"]]
    assert max(partials) - min(partials) < 0.02, partials


def test_curved_data_is_flagged():
    C, br = _curved()
    res = analyse("curved", C, br, _args())
    assert res["verdict"] == "CURVATURE MATERIAL", res["verdict"]
    assert res["max_excess"] > 0.05, res["max_excess"]


def test_verdict_uses_the_whole_ladder_not_just_the_richest_basis():
    """The regression guard. On the curved fixture the richest basis under-reports the problem;
    a verdict read off `tensor4` alone would wrongly pass."""
    C, br = _curved()
    res = analyse("curved", C, br, _args())
    rich = next(r for r in res["rows"] if r["spec"] == res["richest"])
    assert rich["excess"] < 0.05, "fixture changed: the richest basis now catches it directly"
    assert res["max_excess_spec"] != res["richest"]
    assert res["max_excess"] > 3 * rich["excess"]


def test_reported_number_is_a_fixed_basis_not_the_ladder_minimum():
    """A minimum over five noisy estimates of one quantity is biased downward by the selection
    itself, so the published figure must come from a basis fixed in advance. Both are emitted;
    only one is labelled for reporting."""
    C, br = _curved()
    res = analyse("curved", C, br, _args())
    rich = next(r for r in res["rows"] if r["spec"] == res["richest"])
    assert res["reported_spec"] == res["richest"]
    assert res["reported_partial"] == rich["partial"]
    assert res["min_partial"] <= res["reported_partial"]
    assert res["enriched_spread"] >= 0.0


def test_placebo_tracks_the_linear_number_not_the_enriched_one():
    """The calibration only works if random columns of matched count barely move the estimate at
    this n; otherwise every excess would be swamped by df bookkeeping."""
    C, br = _clean()
    res = analyse("clean", C, br, _args())
    lin = res["linear"]
    for r in res["rows"]:
        assert abs(r["placebo_partial"] - lin) < 0.02, r


def test_enriched_estimator_is_still_unbiased():
    """A richer basis must not acquire a floor of its own -- otherwise the drop it produces
    would be estimator bias rather than removed confound."""
    C, br = _clean()
    res = analyse("clean", C, br, _args())
    assert abs(res["floor_richest_mean"]) < 0.02, res["floor_richest_mean"]
    assert res["z_vs_floor_richest"] > 5


def test_curvature_gain_is_larger_on_the_curved_fixture():
    """The direct diagnostic must agree with the verdict, independently of the partials."""
    clean = analyse("clean", *_clean(), _args())
    curved = analyse("curved", *_curved(), _args())
    g = lambda r: next(x for x in r["rows"] if x["spec"] == "tensor4")["delta_r2_target"]
    assert g(curved) > 3 * g(clean), (g(curved), g(clean))


if __name__ == "__main__":
    for name, fn in sorted(list(globals().items())):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("all control_linearity tests passed")
