"""Path A: per-feature attribution of the emitted action, the viability gate, and the
participation-ratio task-breadth score.

See EXPERIMENT_PLAN.md §3.2 and the Path A design doc for the full derivation. In brief,
OpenVLA emits an action token by a dot product at the final layer,

    logit(t) = RMSNorm(h) . u_t ,     u_t = row t of the unembedding W_U,

which is additive, so with the SAE writing the residual as

    h  ~=  l2 * ( sum_j z_j w_j )  +  mu * 1  +  b_pre

feature j's contribution to the emitted token t is

    phi_j = (l2 / r) * z_j * < w_j , g (*) u_contrast > ,          (Eq. 23/24)

where r = rms(h), g is the final-norm gain, (*) is entrywise product, and
u_contrast = u_t - mean over the 256 action tokens of u_s. Two easy mistakes this module
avoids: (i) it carries the per-sample l2 factor (our SAE normalises each sample), and
(ii) it attributes to the contrast direction, so a direction that lifts every action
logit equally receives no credit.

This module is pure numpy and fully unit-tested. It consumes arrays that the (separate)
action-position re-collection must produce -- see `DATA CONTRACT` below -- so writing it
first fixes that contract precisely.

DATA CONTRACT (what the re-collection + retrained SAE must provide, per decision)
--------------------------------------------------------------------------------
    h        [d]        un-pooled layer-31 residual at the action-token position
    tok      int        the emitted action token id (argmax over the 256 action tokens)
    z        [F]        SAE code for h (k non-zeros); from the retrained SAE
    l2, mu   float      the SAE's per-sample normaliser and mean (from its forward pass)
    task     int        which task this decision belongs to (for aggregation)
Once per model (constants):
    W_dec    [F, d]     SAE decoder rows w_j (unit norm)
    W_U_act  [256, d]   unembedding rows for the 256 action tokens only
    act_ids  [256]      the vocabulary ids of those 256 action tokens, in u-row order
    g        [d]        final-RMSNorm gain
    eps      float      final-RMSNorm epsilon
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Final RMSNorm (matches Llama-2 / the SAE trainer's normalisation conventions)
# ---------------------------------------------------------------------------
def rms(h: np.ndarray, eps: float = 1e-5) -> float:
    """r = sqrt( mean(h^2) + eps ). A single scalar per residual vector."""
    h = np.asarray(h, dtype=np.float64)
    return float(np.sqrt((h * h).mean() + eps))


def rmsnorm(h: np.ndarray, g: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """RMSNorm(h) = (h / rms(h)) * g, entrywise gain g."""
    h = np.asarray(h, dtype=np.float64)
    return (h / rms(h, eps)) * np.asarray(g, dtype=np.float64)


def action_logits(h: np.ndarray, W_U_act: np.ndarray, g: np.ndarray,
                  eps: float = 1e-5) -> np.ndarray:
    """Logits over the 256 action tokens for residual h: RMSNorm(h) @ W_U_act.T."""
    hn = rmsnorm(h, g, eps)
    return np.asarray(W_U_act, dtype=np.float64) @ hn


# ---------------------------------------------------------------------------
# Attribution (Eq. 23/24)
# ---------------------------------------------------------------------------
def contrast_direction(W_U_act: np.ndarray, tok_row: int) -> np.ndarray:
    """u_contrast = u_t - mean over all action-token rows of u_s.

    tok_row is the ROW INDEX into W_U_act of the emitted token (0..255), not the
    vocabulary id. Caller maps vocab id -> row via act_ids.
    """
    U = np.asarray(W_U_act, dtype=np.float64)
    return U[tok_row] - U.mean(axis=0)


def attribute(z: np.ndarray, W_dec: np.ndarray, l2: float, r: float,
              g: np.ndarray, u_contrast: np.ndarray) -> np.ndarray:
    """Per-feature attribution phi_j to the emitted action (Eq. 23/24). Returns [F].

    phi_j = (l2 / r) * z_j * < w_j , g (*) u_contrast >.
    Only the active (z_j != 0) features get nonzero phi, so this is cheap despite F.
    """
    z = np.asarray(z, dtype=np.float64)
    W_dec = np.asarray(W_dec, dtype=np.float64)
    gu = np.asarray(g, dtype=np.float64) * np.asarray(u_contrast, dtype=np.float64)  # [d]
    align = W_dec @ gu                                    # [F] = <w_j, g(*)u_contrast>
    return (l2 / r) * z * align


def reconstruct(z: np.ndarray, W_dec: np.ndarray, l2: float, mu: float,
                b_pre: np.ndarray) -> np.ndarray:
    """SAE reconstruction h_hat = l2 * (z @ W_dec) + mu + b_pre  (Eq. 21)."""
    z = np.asarray(z, dtype=np.float64)
    W_dec = np.asarray(W_dec, dtype=np.float64)
    return l2 * (z @ W_dec) + mu + np.asarray(b_pre, dtype=np.float64)


# ---------------------------------------------------------------------------
# Viability gate
# ---------------------------------------------------------------------------
def gate_level1(h_list, z_list, l2_list, mu_list, tok_row_list,
                W_dec, b_pre, W_U_act, g, eps: float = 1e-5) -> dict:
    """L1: does the SAE reconstruction re-decode to the same action?

    For each decision: compare argmax action-logit from the true h vs from the SAE
    reconstruction h_hat. Returns the agreement fraction (pass >= 0.85) and the mean
    correlation between true and reconstructed action-logit vectors (expect > 0.9).
    """
    agree, corrs = 0, []
    n = len(h_list)
    for h, z, l2, mu, tok_row in zip(h_list, z_list, l2_list, mu_list, tok_row_list):
        true_l = action_logits(h, W_U_act, g, eps)
        h_hat = reconstruct(z, W_dec, l2, mu, b_pre)
        rec_l = action_logits(h_hat, W_U_act, g, eps)
        agree += int(np.argmax(true_l) == np.argmax(rec_l))
        # correlation of the two logit vectors across the 256 action tokens
        a = true_l - true_l.mean(); b = rec_l - rec_l.mean()
        d = np.sqrt((a * a).sum() * (b * b).sum())
        corrs.append(float((a * b).sum() / d) if d > 0 else np.nan)
    return {
        "n": n,
        "action_match": agree / n if n else float("nan"),
        "mean_logit_corr": float(np.nanmean(corrs)) if corrs else float("nan"),
        "pass": (agree / n >= 0.85) if n else False,
    }


def gate_level2(h_list, z_list, l2_list, mu_list, tok_row_list,
                W_dec, b_pre, W_U_act, g, eps: float = 1e-5) -> dict:
    """L2: do the frozen-r per-feature phi sum back to the true logit?

    For each decision, compare the emitted token's true logit against
    (sum_j phi_j) + constant, where the constant is the contribution of the mu*1 + b_pre
    terms through the same frozen-r readout. Returns correlation across decisions and the
    mean absolute discrepancy.
    """
    true_vals, recon_vals = [], []
    for h, z, l2, mu, tok_row in zip(h_list, z_list, l2_list, mu_list, tok_row_list):
        r = rms(h, eps)
        u_contrast = contrast_direction(W_U_act, tok_row)
        # feature part
        phi = attribute(z, W_dec, l2, r, g, u_contrast)
        feat_part = float(phi.sum())
        # constant part: (mu*1 + b_pre) through the same frozen-r readout, contrasted
        gu = np.asarray(g, np.float64) * u_contrast
        const_vec = mu + np.asarray(b_pre, np.float64)
        const_part = float((const_vec @ gu) / r)
        recon_logit_contrast = feat_part + const_part
        # true contrasted logit at the emitted token
        hn = rmsnorm(h, g, eps)
        U = np.asarray(W_U_act, np.float64)
        true_contrast = float(hn @ (U[tok_row] - U.mean(axis=0)))
        true_vals.append(true_contrast)
        recon_vals.append(recon_logit_contrast)
    true_vals = np.array(true_vals); recon_vals = np.array(recon_vals)
    a = true_vals - true_vals.mean(); b = recon_vals - recon_vals.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return {
        "n": len(true_vals),
        "logit_recon_corr": float((a * b).sum() / d) if d > 0 else float("nan"),
        "mean_abs_discrepancy": float(np.abs(true_vals - recon_vals).mean()),
    }


# ---------------------------------------------------------------------------
# Participation-ratio task-breadth score
# ---------------------------------------------------------------------------
def per_task_importance(phi_by_decision: np.ndarray, task_by_decision: np.ndarray,
                        n_features: int) -> tuple[np.ndarray, np.ndarray]:
    """C_j(g) = mean over task-g decisions of |phi_j|. Returns (C [G, F], task_ids [G])."""
    phi = np.asarray(phi_by_decision, dtype=np.float64)          # [N_decisions, F]
    task = np.asarray(task_by_decision)
    task_ids = np.unique(task)
    C = np.zeros((len(task_ids), n_features), dtype=np.float64)
    for gi, t in enumerate(task_ids):
        rows = task == t
        if rows.any():
            C[gi] = np.abs(phi[rows]).mean(axis=0)
    return C, task_ids


def participation_ratio(C: np.ndarray) -> np.ndarray:
    """PR_j = (sum_g C_j(g))^2 / sum_g C_j(g)^2, per feature. C is [G, F]. Returns [F].

    Effective number of tasks a feature drives: 1 (all in one task) to G (spread even).
    Scale-free, so it measures breadth of causal influence, not its strength.
    """
    C = np.asarray(C, dtype=np.float64)
    s1 = C.sum(axis=0)                       # [F]
    s2 = (C * C).sum(axis=0)                 # [F]
    with np.errstate(divide="ignore", invalid="ignore"):
        pr = np.where(s2 > 0, s1 * s1 / s2, np.nan)
    return pr


def total_magnitude(C: np.ndarray) -> np.ndarray:
    """Sum_g C_j(g): the feature's overall causal importance (the confound PR must beat)."""
    return np.asarray(C, dtype=np.float64).sum(axis=0)


# ---------------------------------------------------------------------------
# the A4 statistic: leave-one-task-out prediction of held-out causal importance
# ---------------------------------------------------------------------------
def rank_partial_both(y: np.ndarray, x: np.ndarray, c1: np.ndarray,
                      c2: np.ndarray) -> float:
    """Rank-partial correlation of y and x controlling for BOTH c1 and c2.

    Residualises the rank vectors of x and y on the [c1, c2] rank plane by least squares and
    correlates what is left. This is the estimator behind the headline `partial | both`
    number: it asks whether breadth predicts held-out causal importance beyond causal
    magnitude AND beyond base firing rate, the two confounds that killed the firing metrics.
    """
    y, x = np.asarray(y, float), np.asarray(x, float)
    c1, c2 = np.asarray(c1, float), np.asarray(c2, float)
    m = np.isfinite(y) & np.isfinite(x) & np.isfinite(c1) & np.isfinite(c2)
    if m.sum() < 5:
        return float("nan")

    def rk(v):
        r = np.argsort(np.argsort(v[m])).astype(np.float64)
        return r - r.mean()

    ry, rx = rk(y), rk(x)
    rc = np.stack([rk(c1), rk(c2)], axis=1)
    beta_x, *_ = np.linalg.lstsq(rc, rx, rcond=None)
    beta_y, *_ = np.linalg.lstsq(rc, ry, rcond=None)
    ex, ey = rx - rc @ beta_x, ry - rc @ beta_y
    den = np.sqrt((ex * ex).sum() * (ey * ey).sum())
    return float((ex * ey).sum() / den) if den > 0 else float("nan")


def loto_partial_both(C: np.ndarray, base_rate: np.ndarray) -> np.ndarray:
    """Per-fold `partial | both` over leave-one-task-out folds. C is [G, F]; returns [<=G].

    For each held-out task g: recompute breadth (PR) and causal magnitude on the REMAINING
    tasks only -- recomputing both is what keeps the held-out task out of the confound
    controls as well as out of the predictor -- then rank-partial the training breadth
    against the held-out task's causal importance, controlling training magnitude and base
    rate. Folds with too few usable features are dropped rather than returned as NaN.

    Factored out of `run_attribution.py` so that re-analyses (permutation nulls, per-suite
    replication) score the identical estimator instead of a re-implementation that might
    drift from it.
    """
    C = np.asarray(C, dtype=np.float64)
    base_rate = np.asarray(base_rate, dtype=np.float64)
    G = C.shape[0]
    vals: list[float] = []
    for gi in range(G):
        keep = np.arange(G) != gi
        PR_tr = participation_ratio(C[keep])
        mag_tr = total_magnitude(C[keep])
        held = C[gi]
        m = (mag_tr > 0) & np.isfinite(PR_tr)
        if m.sum() > 4:
            v = rank_partial_both(held[m], PR_tr[m], mag_tr[m], base_rate[m])
            if np.isfinite(v):
                vals.append(v)
    return np.array(vals, dtype=np.float64)
