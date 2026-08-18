"""Exact counterfactuals at the action readout: what changes if a feature is removed?

Attribution says a feature CONTRIBUTES phi to the emitted action. It does not say the action
would have been different without it -- a large phi on a decision with a large margin changes
nothing. Necessity is the stronger claim, and until now the only way to get it was a closed-loop
rollout: 200 episodes, a success bit each, and a minimum detectable effect around 9 points.

At layer 31 the readout is the whole remaining computation, so the counterfactual can be
computed EXACTLY, for every feature on every stored decision, with no model forward pass at all.

    logit_t   = ((h / r) (*) g) . u_t  =  (1/r) * L_t ,   L_t = (h (*) g) . u_t
    h'        = h - sum_k coeff_k w_k
    L'_t      = L_t - sum_k coeff_k * S[k, t] ,           S[k, t] = (w_k (*) g) . u_t

Three consequences, all of which make this cheap:

  * r DROPS OUT of the argmax. It is a positive scalar, so it rescales every action logit
    equally and cannot change which bin wins. Flips depend only on L', and the RMSNorm
    recomputation is not needed at all (it is still needed for logit VALUES, which is why
    `ablated_logits` takes an optional r).
  * S is the SAME signature matrix Path B builds for cross-model matching. One object serves
    the channel decomposition and the recurrence analysis.
  * Contrast-centering S is irrelevant here: it subtracts a constant from every bin of a row,
    shifting all logits equally. Centered or not, the flips are identical.

TWO ABLATION SEMANTICS -- they are not the same intervention
------------------------------------------------------------
`mrvla/hooks.py:ActivationAblator` (what the closed-loop rollouts run) computes
`h - (h @ V.T) @ V`: it projects out the whole component of h along the decoder direction,
whatever produced it -- the SAE's coded amount, its reconstruction error, the mean term, and
overlap leaking in from other features. Path A's phi describes something narrower: the coded
contribution `l2 * z_j * w_j` alone.

So `coeff` has two defensible definitions, and this module implements both:

    PROJECTION  coeff_j = <h, w_j>      matches the rollouts. No SAE encoder needed -- W_dec
                                        alone suffices, so this runs without torch.
    CODED       coeff_j = l2 * z_j      matches phi and the attribution story.

Report both. Their gap is not noise: it measures how much of a projection ablation's effect
comes from structure the SAE never attributed to that feature, which is a real caveat on
reading rollout results as evidence about coded features.

A NON-ORTHOGONALITY CAVEAT, INHERITED DELIBERATELY
--------------------------------------------------
For a set of directions the hook computes `h - (h @ V.T) @ V`, which is a true orthogonal
projection only when the rows of V are orthonormal. Decoder rows are unit-norm but not mutually
orthogonal, so for correlated features the hook OVER-SUBTRACTS. `coalition_coeffs` reproduces
the hook's formula rather than a corrected projection, because the point is to model the
experiment that was actually run. `coalition_overlap` reports how far from orthogonal a
coalition is, so the size of the distortion is visible instead of implicit.

SCOPE. This is the exact DIRECT effect on one decode slot. In a rollout, ablation also changes
what slot s+1 attends to and what the next timestep observes, so the full behavioural effect is
larger. Read these numbers as a lower bound on behavioural impact, and as the precise answer to
"does this feature decide this action".
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "unnormalized_logits", "signature_matrix", "projection_coeffs", "coded_coeffs",
    "top2_margin", "cannot_flip_mask", "ablated_logits", "ablated_argmax",
    "single_feature_flips", "coalition_coeffs", "coalition_ablated_argmax",
    "coalition_overlap", "bin_index_from_row", "signed_bin_shift",
]


# ---------------------------------------------------------------------------
# the pieces
# ---------------------------------------------------------------------------
def unnormalized_logits(H: np.ndarray, g: np.ndarray, W_U_act: np.ndarray) -> np.ndarray:
    """L [n, 256] with L[i, t] = (h_i (*) g) . u_t -- the action logits up to the 1/r factor.

    argmax(L) == argmax(true logits) exactly, since r > 0 is a per-row scalar.
    """
    H = np.asarray(H, dtype=np.float64)
    if H.ndim == 1:
        H = H[None, :]
    g = np.asarray(g, dtype=np.float64)
    U = np.asarray(W_U_act, dtype=np.float64)
    return (H * g[None, :]) @ U.T


def signature_matrix(W_dec: np.ndarray, g: np.ndarray, W_U_act: np.ndarray,
                     center: bool = False) -> np.ndarray:
    """S [F, 256] with S[j, t] = (w_j (*) g) . u_t: how feature j moves each action logit.

    Identical to `run_causal_recurrence.causal_signature`. `center` is offered for callers that
    want the contrast-centered form for cross-model cosines; it does not affect any argmax or
    flip computed here, because it subtracts a per-row constant across bins.
    """
    W = np.asarray(W_dec, dtype=np.float64)
    S = (W * np.asarray(g, dtype=np.float64)[None, :]) @ np.asarray(W_U_act, dtype=np.float64).T
    return S - S.mean(axis=1, keepdims=True) if center else S


def projection_coeffs(H: np.ndarray, W_dec: np.ndarray) -> np.ndarray:
    """[n, F] of <h_i, w_j> -- the amount the ROLLOUT ablator removes. Needs no SAE encoder.

    Assumes unit-norm decoder rows (the SAE's convention, and the hook re-normalises anyway);
    `assert_unit_rows` below is the runtime check.
    """
    return np.asarray(H, dtype=np.float64) @ np.asarray(W_dec, dtype=np.float64).T


def coded_coeffs(z: np.ndarray, l2: np.ndarray) -> np.ndarray:
    """[n, F] of l2_i * z_ij -- the amount ATTRIBUTION says feature j wrote into the residual."""
    z = np.asarray(z, dtype=np.float64)
    l2 = np.asarray(l2, dtype=np.float64)
    return z * (l2[:, None] if z.ndim == 2 else l2)


def assert_unit_rows(W_dec: np.ndarray, tol: float = 1e-3) -> float:
    """Max deviation of decoder row norms from 1. Projection semantics assume unit rows."""
    n = np.linalg.norm(np.asarray(W_dec, dtype=np.float64), axis=1)
    return float(np.abs(n - 1.0).max())


def top2_margin(L: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(argmax [n], margin [n]) where margin = best - runner-up, in unnormalised logit units.

    The margin is what a perturbation has to overcome to change the action, so it is both the
    pruning bound below and the natural graded measure of "how close was this decision".
    """
    L = np.atleast_2d(np.asarray(L, dtype=np.float64))
    part = np.argpartition(-L, 1, axis=1)[:, :2]
    vals = np.take_along_axis(L, part, axis=1)
    order = np.argsort(-vals, axis=1)
    top = np.take_along_axis(part, order, axis=1)
    tvals = np.take_along_axis(vals, order, axis=1)
    return top[:, 0], tvals[:, 0] - tvals[:, 1]


def cannot_flip_mask(coeff: np.ndarray, S_row: np.ndarray, margin: np.ndarray) -> np.ndarray:
    """True where removing this feature CANNOT change the argmax. Exact, never approximate.

    Subtracting coeff * S_row moves any two bins apart by at most
    |coeff| * (max_t S_row - min_t S_row). If that is at or below the margin to the runner-up,
    no bin can overtake the winner. Used only to skip work; the surviving rows are computed in
    full, so results are identical with or without it (pinned by a test).
    """
    swing = float(np.max(S_row) - np.min(S_row))
    return np.abs(np.asarray(coeff, dtype=np.float64)) * swing <= np.asarray(margin, float)


def ablated_logits(L: np.ndarray, S_rows: np.ndarray, coeff: np.ndarray,
                   r: np.ndarray | None = None) -> np.ndarray:
    """L' after removing one or more features. `coeff` broadcasts against `S_rows` [k, 256].

    Pass `r` only if you need true logit values; argmax and flips do not need it.
    """
    L = np.atleast_2d(np.asarray(L, dtype=np.float64))
    S_rows = np.atleast_2d(np.asarray(S_rows, dtype=np.float64))   # [k, 256]
    c = np.asarray(coeff, dtype=np.float64)
    if c.ndim == 1:
        # a 1-D coeff is per-DECISION when one feature is removed ([n] -> [n,1]), and
        # per-FEATURE when one decision is ([k] -> [1,k]). atleast_2d guesses wrong on the
        # first, which is the common case, so disambiguate on the number of signature rows.
        c = c[:, None] if S_rows.shape[0] == 1 else c[None, :]
    out = L - c @ S_rows
    return out / np.asarray(r, dtype=np.float64)[:, None] if r is not None else out


def ablated_argmax(L: np.ndarray, S_rows: np.ndarray, coeff: np.ndarray) -> np.ndarray:
    return np.argmax(ablated_logits(L, S_rows, coeff), axis=1)


def single_feature_flips(L: np.ndarray, S_row: np.ndarray, coeff: np.ndarray,
                         base_argmax: np.ndarray | None = None,
                         base_margin: np.ndarray | None = None) -> dict:
    """Per-decision effect of removing ONE feature, over n decisions.

    Returns the flip mask, the new argmax, and the signed change in ROW index. Rows the margin
    bound rules out are skipped and reported, so the saving is auditable.
    """
    L = np.atleast_2d(np.asarray(L, dtype=np.float64))
    S_row = np.asarray(S_row, dtype=np.float64).ravel()
    coeff = np.asarray(coeff, dtype=np.float64).ravel()
    if base_argmax is None or base_margin is None:
        base_argmax, base_margin = top2_margin(L)

    safe = cannot_flip_mask(coeff, S_row, base_margin)
    new = base_argmax.copy()
    idx = np.where(~safe)[0]
    if idx.size:
        new[idx] = np.argmax(L[idx] - coeff[idx, None] * S_row[None, :], axis=1)
    flipped = new != base_argmax
    return {"flipped": flipped, "new_argmax": new, "base_argmax": base_argmax,
            "row_shift": (new.astype(np.int64) - base_argmax.astype(np.int64)),
            "flip_rate": float(flipped.mean()) if flipped.size else float("nan"),
            "n_evaluated": int(idx.size), "n_pruned": int(safe.sum())}


# ---------------------------------------------------------------------------
# coalitions -- reproducing the hook, including its imperfection
# ---------------------------------------------------------------------------
def coalition_coeffs(H: np.ndarray, W_dec: np.ndarray, idx) -> np.ndarray:
    """[n, k] of <h_i, w_j> for j in idx: exactly the `(h @ V.T)` the rollout hook subtracts."""
    V = np.asarray(W_dec, dtype=np.float64)[list(idx)]
    return np.asarray(H, dtype=np.float64) @ V.T


def coalition_ablated_argmax(L: np.ndarray, S: np.ndarray, idx, coeffs: np.ndarray) -> np.ndarray:
    """argmax after `h - (h @ V.T) @ V`, computed in logit space."""
    return np.argmax(ablated_logits(L, np.asarray(S)[list(idx)], coeffs), axis=1)


def coalition_overlap(W_dec: np.ndarray, idx) -> dict:
    """How far a coalition is from orthogonal -- i.e. how much the hook over-subtracts.

    Returns the max and mean absolute off-diagonal of the Gram matrix of its unit directions.
    At ~0 the hook's formula is a true projection and the ablation removes exactly the
    coalition's span. Large values mean shared directions are subtracted more than once, and
    the intervention is stronger than "remove these k features" implies -- worth knowing before
    reading a coalition rollout as evidence about the features themselves.
    """
    V = np.asarray(W_dec, dtype=np.float64)[list(idx)]
    V = V / np.maximum(np.linalg.norm(V, axis=1, keepdims=True), 1e-12)
    G = V @ V.T
    off = G[~np.eye(G.shape[0], dtype=bool)] if G.shape[0] > 1 else np.array([0.0])
    return {"k": int(G.shape[0]), "max_abs_offdiag": float(np.abs(off).max()),
            "mean_abs_offdiag": float(np.abs(off).mean()),
            "is_near_orthogonal": bool(np.abs(off).max() < 0.1)}


# ---------------------------------------------------------------------------
# bins
# ---------------------------------------------------------------------------
def bin_index_from_row(row: np.ndarray, n_bins: int = 256) -> np.ndarray:
    """Row of W_U_act -> OpenVLA bin index.

    OpenVLA decodes with `token_id = vocab_size - bin_index`, bin_index in [1, n_bins], while
    act_ids is stored ascending from vocab_size - n_bins. So row 0 is the highest bin index and
    row n_bins-1 the lowest: bin_index = n_bins - row. Any signed reading of the action axis
    must apply this, and the error is silent if it does not.
    """
    return n_bins - np.asarray(row, dtype=np.int64)


def signed_bin_shift(base_row: np.ndarray, new_row: np.ndarray, n_bins: int = 256) -> np.ndarray:
    """Change in BIN INDEX (not row index) when a feature is removed: positive = higher bin.

    Row shift and bin shift have opposite signs; this is the one to report.
    """
    return bin_index_from_row(new_row, n_bins) - bin_index_from_row(base_row, n_bins)
