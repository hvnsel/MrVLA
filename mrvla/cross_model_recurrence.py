"""Cross-model recurrence: a label-free, firing-rate-independent generality signal.

Motivation
----------
Firing-based generality metrics reduce to base firing rate (see
`structural_generality.py` and EXPERIMENT_PLAN.md §2.2).  This module measures a
different fingerprint of generality: **does the model rediscover a feature when it
is independently fine-tuned on a different task suite?**  A feature that recurs
across independently fine-tuned models is capturing something the specific suite
did not invent -- a reusable computation.  Recurrence is about the feature's
*activation pattern* matching across models, not how often it fires, so it is not
automatically confounded by activity (but we still check, per principle #2).

Setup
-----
We have OpenVLA fine-tuned on 4 LIBERO suites, each with its own SAE.  To compare
feature *i* of model A with feature *j* of model B we need their activations on the
**same probe frames**: push one fixed frame set through every model, encode each
model's residuals with *that model's* SAE, giving code matrices

    Z^m  of shape [N_frames, F]   (row = shared frame, column = SAE feature)

for each model m.  Two features "match" if their columns correlate across frames.

Scores (per feature i of a chosen target model)
-----------------------------------------------
    q_cross_i   = mean over other models m of  max_j corr(Z^target_i, Z^m_j)
                  -- how well feature i recurs across *models*.  This is the raw
                     recurrence-generality score.

    q_seed_i    = max_j corr(Z^target_i, Z^target-seed2_j)
                  -- how well it recurs across a *second SAE seed of the same
                     model* (features identical; only SAE noise differs).  This is
                     the noise floor / SAE-identifiability ceiling.

    retention_i = q_cross_i / q_seed_i
                  -- fraction of the reproducible signal that survives changing the
                     *model*.  Near 1 = the feature is model-independent (general);
                     small = the feature reproduces within a model but not across
                     models (memorized / suite-specific).  Distinguishes universal
                     from suite-specific features even when both are reproducible.

Confound control (mandatory, principle #2)
------------------------------------------
Busy features (high base rate) can match other busy features trivially, inflating
q_cross.  We therefore always report q_cross **residualised on base firing rate**
(rank-partial): a feature only counts as recurrent if it matches across models
*beyond* what its overall activity explains.  This is the same control that
falsified the firing metrics; it is applied here up front, not after the fact.

Decision (EXPERIMENT_PLAN.md §3.1)
----------------------------------
    base-rate-residualised q_cross clearly > 0, above the seed noise floor
        -> recurrence is a real generality signal; rank features by it.
    ~0 with the noise floor resolvable
        -> features are model-local; generality does not survive re-fine-tuning.
    noise floor unresolvable (seeds match no better than chance)
        -> SAE non-identifiability dominates; a methods result, NOT evidence
           about generality.
"""

from __future__ import annotations

import numpy as np

from mrvla.structural_generality import _ranks, _corr, _spearman, _partial_spearman


# ---------------------------------------------------------------------------
# Column-wise correlation between two code matrices over shared frames
# ---------------------------------------------------------------------------
def column_correlations(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Pearson correlation of every column of A with every column of B.

    A : [N, FA]  code matrix for model A on the shared frames
    B : [N, FB]  code matrix for model B on the *same* frames (row-aligned)

    Returns C [FA, FB] with C[i, j] = corr(A[:, i], B[:, j]) over the N frames.
    Dead / constant columns (zero variance) correlate 0 with everything.
    """
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    if A.shape[0] != B.shape[0]:
        raise ValueError(f"row (frame) counts must match: {A.shape[0]} vs {B.shape[0]}")
    N = A.shape[0]
    Az = A - A.mean(axis=0, keepdims=True)
    Bz = B - B.mean(axis=0, keepdims=True)
    Asd = np.sqrt((Az ** 2).sum(axis=0))          # [FA]
    Bsd = np.sqrt((Bz ** 2).sum(axis=0))          # [FB]
    C = Az.T @ Bz                                  # [FA, FB], unnormalised
    denom = np.outer(Asd, Bsd)
    with np.errstate(divide="ignore", invalid="ignore"):
        C = np.where(denom > 0, C / denom, 0.0)
    return C


def greedy_q(C: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-row best match: q[i] = max_j C[i, j], and the argmax index."""
    if C.shape[1] == 0:
        return np.zeros(C.shape[0]), np.full(C.shape[0], -1)
    idx = np.argmax(C, axis=1)
    q = C[np.arange(C.shape[0]), idx]
    return q, idx


def hungarian_q(C: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Optimal one-to-one assignment (max total correlation), if scipy is present.

    Returns (q_assigned [FA], match_idx [FA]); features left unassigned (when
    FA != FB) get q = 0, idx = -1.  Falls back to greedy if scipy is unavailable.
    """
    try:
        from scipy.optimize import linear_sum_assignment
    except Exception:
        return greedy_q(C)
    rows, cols = linear_sum_assignment(-C)         # maximise
    q = np.zeros(C.shape[0])
    idx = np.full(C.shape[0], -1)
    q[rows] = C[rows, cols]
    idx[rows] = cols
    return q, idx


# ---------------------------------------------------------------------------
# Recurrence scores
# ---------------------------------------------------------------------------
def cross_model_q(Z_target: np.ndarray, Z_others: list[np.ndarray],
                  method: str = "greedy") -> np.ndarray:
    """Mean over other models of the best-match correlation for each target feature."""
    if not Z_others:
        raise ValueError("need at least one other model")
    match = hungarian_q if method == "hungarian" else greedy_q
    qs = []
    for Z in Z_others:
        C = column_correlations(Z_target, Z)
        q, _ = match(C)
        qs.append(q)
    return np.mean(np.stack(qs, axis=0), axis=0)   # [F]


def cross_model_q_permuted(Z_target: np.ndarray, Z_others: list[np.ndarray],
                           rng: np.random.Generator, n_perm: int = 1,
                           method: str = "greedy") -> np.ndarray:
    """Chance floor for q_cross: match against row-PERMUTED other models.

    Shuffling the frame order of each other model destroys the real frame-by-frame
    correspondence while preserving every feature's marginal statistics (base rate,
    activation shape) AND the same best-of-F max-matching.  So this is the baseline
    q_cross you would get with no genuine cross-model correspondence.  Real
    recurrence must sit clearly ABOVE this; the excess (q_cross - q_perm) is the
    signal that is not a max-over-many artifact.
    """
    match = hungarian_q if method == "hungarian" else greedy_q
    qs = []
    for _ in range(max(1, n_perm)):
        for Z in Z_others:
            Zp = Z[rng.permutation(Z.shape[0])]
            q, _ = match(column_correlations(Z_target, Zp))
            qs.append(q)
    return np.mean(np.stack(qs, axis=0), axis=0)


def base_rate(Z: np.ndarray, tau: float = 0.0) -> np.ndarray:
    """Per-feature fraction of frames on which the feature is active (> tau)."""
    return (np.asarray(Z) > tau).mean(axis=0)


def recurrence_report(codes_by_model: dict[str, np.ndarray], target: str,
                      seed2: np.ndarray | None = None,
                      method: str = "greedy", tau: float = 0.0) -> dict:
    """Full recurrence analysis for one target model at one layer.

    codes_by_model : {model_name: Z [N_frames, F]} on the SAME shared frames.
    target         : which model's features to score.
    seed2          : optional [N_frames, F] second-seed SAE codes of the *target*
                     model on the same frames (the noise floor).  If given,
                     retention and a noise-floor-corrected score are reported.

    Returns per-feature arrays plus summary confound diagnostics.
    """
    if target not in codes_by_model:
        raise KeyError(f"{target!r} not among {list(codes_by_model)}")
    Z_t = codes_by_model[target]
    others = [Z for m, Z in codes_by_model.items() if m != target]
    if not others:
        raise ValueError("need >= 2 models to measure cross-model recurrence")

    q_cross = cross_model_q(Z_t, others, method=method)
    br = base_rate(Z_t, tau=tau)
    active = br > 0

    out = {
        "q_cross": q_cross.astype(np.float32),
        "base_rate": br.astype(np.float32),
        "is_active": active,
        # descriptive: does raw recurrence track base rate (the confound)?
        "spearman_qcross_baserate": _spearman(q_cross[active], br[active]),
        "n_models": len(others) + 1,
    }

    if seed2 is not None:
        C = column_correlations(Z_t, seed2)
        q_seed, _ = (hungarian_q if method == "hungarian" else greedy_q)(C)
        with np.errstate(divide="ignore", invalid="ignore"):
            retention = np.where(q_seed > 1e-6, q_cross / q_seed, np.nan)
        out["q_seed"] = q_seed.astype(np.float32)
        out["retention"] = retention.astype(np.float32)          # cross / seed
        out["drop"] = (q_seed - q_cross).astype(np.float32)      # seed - cross
    return out


# ---------------------------------------------------------------------------
# Base-rate-controlled ranking (principle #2)
# ---------------------------------------------------------------------------
def base_rate_residual(score: np.ndarray, br: np.ndarray,
                       active: np.ndarray | None = None) -> np.ndarray:
    """Rank-residual of a recurrence score after removing base firing rate.

    Returns the score's rank with the base-rate rank linearly projected out (over
    the active features).  Ranking features by this isolates recurrence that is NOT
    explained by how busy the feature is -- the control the firing metrics failed.
    Inactive features receive nan.
    """
    score = np.asarray(score, dtype=np.float64)
    br = np.asarray(br, dtype=np.float64)
    F = len(score)
    if active is None:
        active = np.ones(F, dtype=bool)
    m = active & np.isfinite(score) & np.isfinite(br)
    res = np.full(F, np.nan)
    if m.sum() < 4:
        return res
    rs, rb = _ranks(score[m]), _ranks(br[m])
    denom = (rb ** 2).sum()
    rs_res = rs - (rs * rb).sum() / denom * rb if denom > 0 else rs
    res[m] = rs_res
    return res


def summarize(report: dict, active: np.ndarray | None = None) -> dict:
    """Compact summary for logging one target/layer."""
    q = report["q_cross"]
    br = report["base_rate"]
    act = report["is_active"] if active is None else active
    s = {
        "n_active": int(act.sum()),
        "q_cross_mean": float(np.nanmean(q[act])) if act.any() else float("nan"),
        "q_cross_median": float(np.nanmedian(q[act])) if act.any() else float("nan"),
        "spearman_qcross_baserate": report["spearman_qcross_baserate"],
    }
    if "retention" in report:
        r = report["retention"]
        rr = r[act]
        s["retention_mean"] = float(np.nanmean(rr)) if np.isfinite(rr).any() else float("nan")
        # base-rate-controlled: does retention beat base rate?
        s["partial_retention_vs_baserate"] = _partial_spearman(
            report["retention"][act], report["q_cross"][act], report["base_rate"][act])
    return s
