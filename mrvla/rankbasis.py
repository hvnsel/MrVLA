"""Basis expansions of the A4 control plane, for testing the linearity assumption.

WHY THIS EXISTS
---------------
The headline A4 statistic (`mrvla.attribution.rank_partial_both`) residualises the ranked
predictor and the ranked target on the PLANE spanned by two ranked controls -- causal
magnitude and base firing rate -- and correlates what is left. A plane can only absorb a
confound that is (a) LINEAR in each control's ranks and (b) ADDITIVE between them. Neither is
guaranteed here: participation ratio is bounded above by the task count while magnitude is
unbounded, so a ceiling effect producing curvature in rank space is a live possibility rather
than a theoretical one.

If the true confound is curved, the linear projection under-removes it, the leftover confound
lands in the residuals, and it is counted as signal. The bias runs TOWARD a positive partial,
i.e. toward the result we report. That is the direction that has to be ruled out.

THE CHECK
---------
Recompute the identical estimator with progressively richer control bases and see whether the
number moves:

    linear   [u1, u2]                                  -- the current control plane
    quad     + u1^2, u2^2, u1*u2                       -- first-order curvature + interaction
    cubic    + u1^3, u2^3, u1^2*u2, u1*u2^2            -- more curvature
    hinge{K} + K piecewise-linear hinges per control   -- absorbs ANY monotone curvature
    tensor{K}  full tensor product of the two hinge bases -- absorbs any smooth SURFACE,
               including arbitrary non-additivity

THE TRAP THIS MODULE IS BUILT AROUND
------------------------------------
Adding columns can only DECREASE an in-sample partial correlation: p extra columns explain
some of both residuals by chance even when they are pure noise. So "the partial dropped under
a richer basis" is not evidence of curvature -- it is the expected mechanical cost of spending
degrees of freedom. `loto_partial_placebo` supplies the calibration: the same number of RANDOM
columns, which measures the drop attributable to df alone. Curvature is only implicated when
the real basis costs materially more than the placebo of matched rank.

`loto_stratified` is the assumption-free backstop: inside a narrow 2-D cell of magnitude AND
base rate, both controls are near-constant, so a plain rank correlation there is confounded by
neither -- no functional form is assumed anywhere. It is a weaker estimand (discarding
between-cell variance in breadth attenuates it), so it corroborates the SIGN and rules out
curvature as the explanation; it is not expected to reproduce the headline magnitude.

All ranking here is TIE-AVERAGED (`mrvla.stats.rankdata_average`), unlike the shipped
estimator's `argsort(argsort(...))`. The difference has been measured at ~1e-5 on this data
geometry, far below anything that matters, but the correct version is what a check should use.
"""

from __future__ import annotations

import numpy as np

from mrvla.attribution import participation_ratio, total_magnitude
from mrvla.stats import rankdata_average

__all__ = [
    "rank_unit", "control_design", "orthonormal_basis", "residualise",
    "rank_partial_design", "loto_partial_design", "loto_partial_placebo",
    "curvature_gain", "loto_stratified", "SPECS",
]

SPECS = ("linear", "quad", "cubic", "hinge5", "tensor4")


def rank_unit(v: np.ndarray) -> np.ndarray:
    """Tie-averaged ranks rescaled to [-1, 1] and centred.

    Rescaling matters for conditioning only: raw ranks run to n-1, so a cubic term would reach
    ~1e10 and swamp the least-squares solve. It is a monotone affine map, so it leaves the
    linear span -- and hence the `linear` result -- unchanged.
    """
    r = rankdata_average(v)
    if r.size < 2:
        return np.zeros(r.size, dtype=np.float64)
    u = 2.0 * r / (r.size - 1) - 1.0
    return u - u.mean()


def _hinges(u: np.ndarray, n_knots: int) -> list[np.ndarray]:
    """Piecewise-linear (ReLU) basis at interior quantile knots of u.

    Linear combinations of {u, max(0, u - t_k)} are exactly the continuous piecewise-linear
    functions with those breakpoints, which approximate any smooth monotone curve to O(1/K^2).
    Quantile knots rather than equally spaced ones put the resolution where the data are.
    """
    qs = np.linspace(0.0, 1.0, n_knots + 2)[1:-1]
    return [np.maximum(0.0, u - t) for t in np.quantile(u, qs)]


def control_design(controls: list[np.ndarray], spec: str = "linear") -> np.ndarray:
    """Design matrix [n, p] of control basis functions. Columns are centred, not orthonormal.

    `controls` are RAW control vectors; they are ranked here so callers cannot accidentally
    mix ranked and unranked inputs.
    """
    U = [rank_unit(c) for c in controls]
    cols: list[np.ndarray] = list(U)

    if spec == "linear":
        pass
    elif spec == "quad":
        cols += [u * u for u in U]
        cols += _cross(U, 1)
    elif spec == "cubic":
        cols += [u * u for u in U] + [u ** 3 for u in U]
        cols += _cross(U, 1) + _cross(U, 2)
    elif spec.startswith("hinge"):
        k = int(spec[len("hinge"):])
        for u in U:
            cols += _hinges(u, k)
        cols += _cross(U, 1)
    elif spec.startswith("tensor"):
        k = int(spec[len("tensor"):])
        per = [[u] + _hinges(u, k) for u in U]
        cols = [c for group in per for c in group]
        # full tensor product across the two controls: any smooth surface, additive or not
        for a in per[0]:
            for b in per[1:]:
                for bb in b:
                    cols.append(a * bb)
    else:
        raise ValueError(f"unknown spec {spec!r}; expected one of {SPECS}")

    D = np.stack(cols, axis=1).astype(np.float64)
    return D - D.mean(axis=0, keepdims=True)


def _cross(U: list[np.ndarray], degree: int) -> list[np.ndarray]:
    """Interaction columns between the controls at the given total extra degree."""
    out = []
    for i in range(len(U)):
        for j in range(i + 1, len(U)):
            if degree == 1:
                out.append(U[i] * U[j])
            elif degree == 2:
                out += [U[i] ** 2 * U[j], U[i] * U[j] ** 2]
    return out


def orthonormal_basis(D: np.ndarray, tol: float = 1e-8) -> np.ndarray:
    """Orthonormal column basis for span(D), dropping numerically dependent directions.

    Rich bases are collinear by construction (u^2 and a hinge overlap heavily). Projecting via
    an SVD basis is stable where a normal-equations solve is not, and the returned column count
    is the EFFECTIVE degrees of freedom -- which is what the placebo has to match for the
    comparison to be fair.
    """
    if D.size == 0:
        return np.zeros((D.shape[0], 0))
    Q, s, _ = np.linalg.svd(D, full_matrices=False)
    return Q[:, s > tol * (s[0] if s.size else 1.0)]


def residualise(v: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """Remove from v everything span(Q) can explain. Q must be orthonormal-columned."""
    v = v - v.mean()
    return v if Q.shape[1] == 0 else v - Q @ (Q.T @ v)


def rank_partial_design(y, x, controls: list[np.ndarray], spec: str = "linear") -> float:
    """Rank-partial correlation of y and x controlling for `controls` under `spec`.

    `spec="linear"` reproduces `mrvla.attribution.rank_partial_both` up to the tie convention.
    """
    y, x = np.asarray(y, float), np.asarray(x, float)
    cs = [np.asarray(c, float) for c in controls]
    m = np.isfinite(y) & np.isfinite(x)
    for c in cs:
        m &= np.isfinite(c)
    if m.sum() < 5:
        return float("nan")
    Q = orthonormal_basis(control_design([c[m] for c in cs], spec))
    ex = residualise(rank_unit(x[m]), Q)
    ey = residualise(rank_unit(y[m]), Q)
    den = np.sqrt((ex * ex).sum() * (ey * ey).sum())
    return float((ex * ey).sum() / den) if den > 0 else float("nan")


def _folds_xy(C: np.ndarray, Y: np.ndarray, base_rate: np.ndarray,
              keep_mask: np.ndarray | None = None, extra: list[np.ndarray] | None = None):
    """Yield (PR_tr, mag_tr, held, base_rate, extras, gi, m) per leave-one-task-out fold.

    C supplies the PREDICTOR (participation ratio over the training tasks) and the magnitude
    control; Y supplies the TARGET row for the held-out task. Passing Y is C reproduces `_folds`
    exactly, which is pinned by a test -- the point of routing both through one generator is
    that a target swap cannot quietly become a different estimator.

    `keep_mask` is an optional [G, F] boolean of per-(task, feature) admissibility, applied to
    the HELD-OUT row only. It has to be per fold rather than global: a feature measured
    adequately on nine tasks and thinly on the tenth belongs in the nine folds where it sits on
    the predictor side, and should drop out only in the fold where it is the target.

    `extra` is a list of [G, F] arrays whose held-out rows become additional controls, so a
    denominator can be conditioned on without any new estimator code.
    """
    C = np.asarray(C, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    # base_rate may be [F] (the shipped global vector) or [G, F], where row gi is the base rate
    # computed from fold gi's TRAINING tasks only. The global vector is contaminated by the
    # held-out task; that is inherited from run_attribution.py and is why the per-fold form is
    # offered rather than assumed.
    base_rate = np.asarray(base_rate, dtype=np.float64)
    extra = [np.asarray(e, dtype=np.float64) for e in (extra or [])]
    G = C.shape[0]
    for gi in range(G):
        keep = np.arange(G) != gi
        br = base_rate[gi] if base_rate.ndim == 2 else base_rate
        PR_tr = participation_ratio(C[keep])
        mag_tr = total_magnitude(C[keep])
        m = (mag_tr > 0) & np.isfinite(PR_tr) & np.isfinite(Y[gi]) & np.isfinite(br)
        if keep_mask is not None:
            m &= np.asarray(keep_mask, dtype=bool)[gi]
        for e in extra:
            m &= np.isfinite(e[gi])
        if m.sum() > 4:
            yield (PR_tr[m], mag_tr[m], Y[gi][m], br[m],
                   [e[gi][m] for e in extra], gi, m)


def _folds(C: np.ndarray, base_rate: np.ndarray):
    """Yield (PR_tr, mag_tr, held, base_rate) per LOTO fold -- the exact split the headline uses."""
    for PR_tr, mag_tr, held, br, _extras, _gi, _m in _folds_xy(C, C, base_rate):
        yield PR_tr, mag_tr, held, br


def loto_partial_target(C, Y, base_rate, spec: str = "linear",
                        keep_mask=None, extra=None) -> np.ndarray:
    """`partial | both` per fold with an arbitrary held-out target. Y is C == `loto_partial_design`.

    Controls are [mag_tr, base_rate] plus any `extra` held-out rows, all entering the SAME
    `rank_partial_design` the headline uses -- so the causal target is estimated by the
    attributional estimator, not by a lookalike.
    """
    vals = [rank_partial_design(held, PR_tr, [mag_tr, br] + extras, spec)
            for PR_tr, mag_tr, held, br, extras, _gi, _m in
            _folds_xy(C, Y, base_rate, keep_mask, extra)]
    return np.array([v for v in vals if np.isfinite(v)], dtype=np.float64)


def loto_partial_design(C, base_rate, spec: str = "linear") -> np.ndarray:
    """Per-fold `partial | both` under a control basis. spec="linear" == the headline."""
    vals = [rank_partial_design(held, PR_tr, [mag_tr, br], spec)
            for PR_tr, mag_tr, held, br in _folds(C, base_rate)]
    return np.array([v for v in vals if np.isfinite(v)], dtype=np.float64)


def loto_partial_placebo(C, base_rate, n_extra: int, rng: np.random.Generator) -> np.ndarray:
    """The calibration: linear controls plus `n_extra` RANDOM columns.

    Measures how far a partial falls from spending n_extra degrees of freedom on nothing. A
    real basis that costs no more than this has found no curvature to remove.
    """
    vals = []
    for PR_tr, mag_tr, held, br in _folds(C, base_rate):
        D = control_design([mag_tr, br], "linear")
        if n_extra > 0:
            D = np.concatenate([D, rng.standard_normal((D.shape[0], n_extra))], axis=1)
            D = D - D.mean(axis=0, keepdims=True)
        Q = orthonormal_basis(D)
        ex, ey = residualise(rank_unit(PR_tr), Q), residualise(rank_unit(held), Q)
        den = np.sqrt((ex * ex).sum() * (ey * ey).sum())
        if den > 0:
            vals.append(float((ex * ey).sum() / den))
    return np.array(vals, dtype=np.float64)


def curvature_gain(C, base_rate, spec: str) -> dict:
    """Direct diagnostic: how much extra variance the nonlinear terms explain.

    Returns mean delta-R^2 over folds for predicting the ranked PREDICTOR and the ranked TARGET
    from the controls, richer basis minus linear. This answers "is the control surface curved
    at all", independently of what that curvature does to the partial. Near zero means the
    plane was already adequate and no amount of basis enrichment can change the answer.
    """
    gx, gy, dfs = [], [], []

    def r2(v, Q):
        v = v - v.mean()
        ss = float((v * v).sum())
        return 0.0 if ss <= 0 else 1.0 - float((residualise(v, Q) ** 2).sum()) / ss

    for PR_tr, mag_tr, held, br in _folds(C, base_rate):
        Ql = orthonormal_basis(control_design([mag_tr, br], "linear"))
        Qr = orthonormal_basis(control_design([mag_tr, br], spec))
        rx, ry = rank_unit(PR_tr), rank_unit(held)
        gx.append(r2(rx, Qr) - r2(rx, Ql))
        gy.append(r2(ry, Qr) - r2(ry, Ql))
        dfs.append(Qr.shape[1] - Ql.shape[1])
    return {"spec": spec,
            "delta_r2_predictor": float(np.mean(gx)) if gx else float("nan"),
            "delta_r2_target": float(np.mean(gy)) if gy else float("nan"),
            "extra_df": int(np.max(dfs)) if dfs else 0}


def loto_stratified(C, base_rate, n_bins: int = 5, min_cell: int = 20) -> dict:
    """Assumption-free backstop: plain rank correlation INSIDE 2-D control cells.

    Features are binned into `n_bins` magnitude quantiles x `n_bins` base-rate quantiles. Both
    controls are near-constant within a cell, so a rank correlation there cannot be produced by
    either of them under ANY functional form -- linear, curved, or otherwise. Cells are pooled
    by a size-weighted mean.

    Expect attenuation relative to the headline: between-cell variance in breadth is real signal
    and this throws it away. The test is on the SIGN and on clearing its own floor, not on
    matching +0.49.
    """
    per_fold, n_cells = [], []
    for PR_tr, mag_tr, held, br in _folds(C, base_rate):
        bm = np.clip(np.searchsorted(np.quantile(mag_tr, np.linspace(0, 1, n_bins + 1)[1:-1]),
                                     mag_tr), 0, n_bins - 1)
        bb = np.clip(np.searchsorted(np.quantile(br, np.linspace(0, 1, n_bins + 1)[1:-1]),
                                     br), 0, n_bins - 1)
        num, den, k = 0.0, 0.0, 0
        for i in range(n_bins):
            for j in range(n_bins):
                sel = (bm == i) & (bb == j)
                n = int(sel.sum())
                if n < min_cell:
                    continue
                a, b = rank_unit(PR_tr[sel]), rank_unit(held[sel])
                d = np.sqrt((a * a).sum() * (b * b).sum())
                if d > 0:
                    num += n * float((a * b).sum() / d)
                    den += n
                    k += 1
        if den > 0:
            per_fold.append(num / den)
            n_cells.append(k)
    arr = np.array(per_fold, dtype=np.float64)
    return {"mean": float(arr.mean()) if arr.size else float("nan"),
            "n_folds": int(arr.size),
            "n_positive": int((arr > 0).sum()),
            "min_fold": float(arr.min()) if arr.size else float("nan"),
            "mean_cells_used": float(np.mean(n_cells)) if n_cells else 0.0,
            "folds": [float(v) for v in arr]}
