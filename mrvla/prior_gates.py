"""Gates 0-4: is the readout's ACTION PRIOR a lever, and does modulating it help?

Path A splits the action margin into features (0.531), a `mu*1 + b_pre` bias (0.405) and
error (0.064). The bias half has never been examined. The hypothesis under test is that the
bias is an ACTION PRIOR -- what the policy does knowing nothing about the scene -- that is
applied with the wrong strength on some decisions, and that re-weighting it produces better
actions there. That is a long chain, so this module implements it as a LADDER OF GATES, each
of which can kill the story cheaply and each of which reports a number either way.

    Gate 0  is the bias variable at all, and how does it split into mu*1 vs b_pre?
    Gate 1  does mu's variation survive division by r, or does mu track r?
    Gate 2  does the bias direction over bins look like the marginal action distribution?
    Gate 3  does bias-share predict deviation from the demo action, beyond margin and activity?
    Gate 4  does re-scaling the bias by lambda move the argmax toward the demo action?

WHAT THE LEVER PHYSICALLY IS (established here, and it is richer than "a constant")
----------------------------------------------------------------------------------
Writing u_c(t) for the contrast-centred unembedding row and gu(t) = g (*) u_c(t),

    bias(t) = [ mu * <g, u_c(t)>  +  <b_pre, gu(t)> ] / r  =  [ mu*A(t) + B(t) ] / r

so over the 256 bins the prior is a ONE-PARAMETER FAMILY spanned by two fixed vectors A and B,
mixed by the per-decision scalar mu. It is not a single fixed direction with varying gain: the
preferred bin can genuinely move with mu. Gate 2 measures whether it does.

r DROPS OUT of everything that matters here. It is a positive per-decision scalar dividing both
the feature and the bias term, so it cancels from the share and cannot change an argmax (the
same argument `mrvla/readout.py` makes). Gates 2-4 are therefore r-free and inherit none of the
frozen-r caveat. Gate 0 and Gate 1 are the exception, because there r's variation IS the
question.

WHY THE GATES ARE ORDERED THIS WAY
----------------------------------
Cost, then likelihood of killing it. Gate 1 is the cheapest structural kill in the ladder and
the one nobody would think to check: if mu is proportional to r then mu/r is constant, the
prefactor cancels it, and there is nothing to modulate no matter what the later gates say.
Gate 3's control is the likeliest soft kill: weak features -> unsure model -> deviates from the
expert is close to definitional, so bias-share has to beat the plain top-2 logit margin or it
is a confidence score wearing a new name.

SCOPE, STATED UP FRONT. All of this runs on DEMO REPLAY, where the model is fed expert states
and `success` is the constant 1 (`mrvla/libero_demos.py`). So the target in Gates 3-4 is
single-step deviation from the expert at expert states. That is a NECESSARY condition for the
story and not evidence about episode success: low single-step imitation error at expert states
is exactly the quantity that famously fails to imply a low failure rate under self-generated
states. Gates 5-7 (pivotality, closed-loop transfer, and a powered success test) need
action-position residuals from closed-loop rollouts with success labels, which do not exist
yet. Nothing in this module should be read as bearing on success rate.

Pure numpy, no torch, no h5py: the driver `prior_gates.py` supplies the arrays.
"""

from __future__ import annotations

import numpy as np

from mrvla.stats import rankdata_average

__all__ = [
    "prior_vectors", "prior_scores", "bias_share",
    "gate0_bias_composition", "gate1_mu_over_r", "gate2_prior_vs_marginal",
    "gate3_share_predicts_deviation", "gate4_lambda_sweep",
    "rank_partial", "spearman", "verdict_table",
    "demo_bin_index",
    "MU_SHARE_MIN", "VAR_FRAC_MU_MIN", "CV_RATIO_MIN", "MU_RETAINED_MIN",
    "RHO_MARGINAL_MIN", "RHO_PARTIAL_MIN", "TAIL_GAIN_MIN", "CANARY_MEDIAN_FRAC",
]

# ---------------------------------------------------------------------------
# Pre-registered pass bars.
#
# Fixed before any of this was run against real data, in the spirit of the sufficiency
# gate's 0.80. They are deliberately LOW: each gate is a screen asking "is there anything
# here at all", not a claim, and a screen that passes on noise is worse than one set loosely.
# Every function returns the raw statistic beside its verdict so a reader can re-score at any
# bar they prefer.
# ---------------------------------------------------------------------------
MU_SHARE_MIN = 0.15      # G0: |mu*A| must be >= 15% of the bias magnitude, else it is b_pre
VAR_FRAC_MU_MIN = 0.10   # G0: >= 10% of bias variance must vanish when mu is frozen
CV_RATIO_MIN = 0.05      # G1: mu/r must have >= 5% relative spread in absolute terms
MU_RETAINED_MIN = 0.25   # G1: dividing by r must leave >= 25% of mu's own relative spread
RHO_MARGINAL_MIN = 0.20  # G2: median per-slot rank corr of prior vs marginal action frequency
RHO_PARTIAL_MIN = 0.10   # G3: partial | (margin, activity) must clear this, with the raw sign
TAIL_GAIN_MIN = 0.02     # G4: >= 2% relative drop in tail deviation vs the lambda = 1 baseline

# The demo-bin canary. On demo replay a fine-tuned policy agrees with the expert most of the
# time, so a large median gap means the discretisation or the norm_stats are wrong -- not that
# the policy is bad. A fraction of n_bins, so it survives a different bin count.
CANARY_MEDIAN_FRAC = 0.03


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Rank correlation with ties AVERAGED (mrvla.stats.rankdata_average, per results.md P9).

    The older pipeline's `argsort(argsort(.))` breaks ties by array index. Every quantity here
    has real ties -- bin indices are integers and deviations are integer bin counts -- so the
    averaged version is not optional in this module.
    """
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3:
        return float("nan")
    rx, ry = rankdata_average(x[m]), rankdata_average(y[m])
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    den = np.sqrt((rx * rx).sum() * (ry * ry).sum())
    return float((rx * ry).sum() / den) if den > 0 else float("nan")


def rank_partial(y: np.ndarray, x: np.ndarray, controls: list) -> float:
    """Rank correlation of y and x residualised on the rank plane of `controls`.

    Same estimator as `mrvla.attribution.rank_partial_both` but (a) takes any number of
    controls and (b) averages ties. Returns nan if fewer than 5 usable rows.
    """
    y, x = np.asarray(y, float), np.asarray(x, float)
    cols = [np.asarray(c, float) for c in controls]
    m = np.isfinite(y) & np.isfinite(x)
    for c in cols:
        m &= np.isfinite(c)
    if m.sum() < 5:
        return float("nan")

    def rk(v):
        r = rankdata_average(v[m])
        return r - r.mean()

    ry, rx = rk(y), rk(x)
    if not cols:
        den = np.sqrt((rx * rx).sum() * (ry * ry).sum())
        return float((rx * ry).sum() / den) if den > 0 else float("nan")
    rc = np.stack([rk(c) for c in cols], axis=1)
    bx, *_ = np.linalg.lstsq(rc, rx, rcond=None)
    by, *_ = np.linalg.lstsq(rc, ry, rcond=None)
    ex, ey = rx - rc @ bx, ry - rc @ by
    den = np.sqrt((ex * ex).sum() * (ey * ey).sum())
    return float((ex * ey).sum() / den) if den > 0 else float("nan")


def _cv(v: np.ndarray) -> float:
    """Coefficient of variation sd/|mean|. nan when the mean is ~0 (the ratio is meaningless)."""
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if v.size < 2:
        return float("nan")
    m = float(v.mean())
    return float(v.std(ddof=1) / abs(m)) if abs(m) > 1e-12 else float("nan")


# ---------------------------------------------------------------------------
# the fixed objects
# ---------------------------------------------------------------------------
def prior_vectors(W_U_act: np.ndarray, g: np.ndarray, b_pre: np.ndarray) -> tuple:
    """(A, B), each [256]: the two fixed vectors the action prior is spanned by.

        A[t] = <g, u_c(t)>            -- what the per-decision scalar mu multiplies
        B[t] = <b_pre, g (*) u_c(t)>  -- the genuinely constant part

    with u_c(t) = W_U_act[t] - mean over rows. Contrast-centring matters here (unlike in
    `readout.signature_matrix`, where it cancels): A and B are compared against each other and
    against the marginal, not only used inside an argmax.
    """
    U = np.asarray(W_U_act, dtype=np.float64)
    g = np.asarray(g, dtype=np.float64)
    U_c = U - U.mean(axis=0, keepdims=True)              # [256, d]
    A = U_c @ g                                          # [256]
    B = (U_c * g[None, :]) @ np.asarray(b_pre, dtype=np.float64)
    return A, B


def prior_scores(mu: np.ndarray, A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """[n, 256] of mu_i * A[t] + B[t] -- the prior's push on every bin, r factored out."""
    mu = np.atleast_1d(np.asarray(mu, dtype=np.float64))
    return mu[:, None] * np.asarray(A, float)[None, :] + np.asarray(B, float)[None, :]


def bias_share(feat: np.ndarray, bias: np.ndarray, err: np.ndarray) -> np.ndarray:
    """|bias| / (|feat| + |bias| + |err|) at the emitted bin, per decision.

    r-invariant: all three terms carry the same 1/r, so it cancels. Bounded in [0, 1], which
    the signed alternative bias/(feat+bias) is not -- the terms can have opposite signs and
    the ratio then blows up. Zero denominators return nan rather than 0, so a dead decision is
    dropped by the correlations instead of being scored as "all features".
    """
    a, b, e = (np.abs(np.asarray(v, dtype=np.float64)) for v in (feat, bias, err))
    den = a + b + e
    return np.where(den > 0, b / np.where(den > 0, den, 1.0), np.nan)


# ---------------------------------------------------------------------------
# GATE 0 -- is the bias variable, and what is it made of?
# ---------------------------------------------------------------------------
def gate0_bias_composition(mu: np.ndarray, r: np.ndarray,
                           A_at_t: np.ndarray, B_at_t: np.ndarray) -> dict:
    """Split the bias into its mu-coupled and constant halves, and say what makes it vary.

    `A_at_t`/`B_at_t` are A and B evaluated at each decision's EMITTED bin, so all four inputs
    are [n]. The bias contribution to the margin is (mu*A_at_t + B_at_t) / r.

    Variance attribution is by single-factor freezing: recompute the bias with one of the three
    inputs (mu, r, the emitted bin) held at its mean and see how much variance disappears. The
    three fractions do not sum to 1 -- the factors interact multiplicatively -- so they are
    reported as "how much of the variance needs this factor", not as a partition.

    FAILS when the mu term is a rounding error next to b_pre, or when freezing mu changes
    nothing. Either way the lever is a global constant and re-scaling it is plain recalibration
    that needs none of Path A.
    """
    mu = np.asarray(mu, dtype=np.float64)
    r = np.asarray(r, dtype=np.float64)
    A_t = np.asarray(A_at_t, dtype=np.float64)
    B_t = np.asarray(B_at_t, dtype=np.float64)

    mu_term, b_term = mu * A_t / r, B_t / r
    bias = mu_term + b_term
    mam, bam = float(np.abs(mu_term).mean()), float(np.abs(b_term).mean())
    mu_share = mam / (mam + bam) if (mam + bam) > 0 else float("nan")

    v_full = float(np.var(bias))

    def frozen(which: str) -> float:
        m_, r_, a_, b_ = mu, r, A_t, B_t
        if which == "mu":
            m_ = np.full_like(mu, mu.mean())
        elif which == "r":
            r_ = np.full_like(r, r.mean())
        elif which == "bin":
            a_ = np.full_like(A_t, A_t.mean())
            b_ = np.full_like(B_t, B_t.mean())
        return float(np.var(m_ * a_ / r_ + b_ / r_))

    def drop(which: str) -> float:
        return float(1.0 - frozen(which) / v_full) if v_full > 0 else float("nan")

    out = {
        "n": int(mu.size),
        "mu_term_mean_abs": mam,
        "b_pre_term_mean_abs": bam,
        "mu_share": float(mu_share),
        "bias_mean": float(bias.mean()),
        "bias_sd": float(bias.std(ddof=1)) if bias.size > 1 else float("nan"),
        "cv_bias": _cv(bias),
        "frac_var_from_mu": drop("mu"),
        "frac_var_from_r": drop("r"),
        "frac_var_from_bin": drop("bin"),
    }
    out["pass"] = bool(
        np.isfinite(out["mu_share"]) and out["mu_share"] >= MU_SHARE_MIN
        and np.isfinite(out["frac_var_from_mu"]) and out["frac_var_from_mu"] >= VAR_FRAC_MU_MIN
    )
    return out


# ---------------------------------------------------------------------------
# GATE 1 -- does mu's variation survive division by r?
# ---------------------------------------------------------------------------
def gate1_mu_over_r(mu: np.ndarray, r: np.ndarray) -> dict:
    """The structural kill. phi carries a (l2/r) prefactor and the bias carries 1/r, so the
    quantity that actually reaches the readout is mu/r. If mu is proportional to r then mu/r is
    constant, the bias share is fixed by construction, and no later gate can rescue the story.

    `mu_retained` = CV(mu/r) / CV(mu): the fraction of mu's relative spread that survives. Near
    zero means r explains mu. Note it can exceed 1 -- if mu and r are anti-correlated the ratio
    is MORE variable than mu alone, which is favourable and not an error.
    """
    mu = np.asarray(mu, dtype=np.float64)
    r = np.asarray(r, dtype=np.float64)
    m = np.isfinite(mu) & np.isfinite(r) & (r > 0)
    mu, r = mu[m], r[m]
    ratio = mu / r
    cv_mu, cv_ratio = _cv(mu), _cv(ratio)
    mu_c, r_c = mu - mu.mean(), r - r.mean()
    den = np.sqrt((mu_c * mu_c).sum() * (r_c * r_c).sum())

    out = {
        "n": int(mu.size),
        "cv_mu": cv_mu,
        "cv_r": _cv(r),
        "cv_mu_over_r": cv_ratio,
        "mu_retained": float(cv_ratio / cv_mu) if (np.isfinite(cv_mu) and cv_mu > 0) else float("nan"),
        "pearson_mu_r": float((mu_c * r_c).sum() / den) if den > 0 else float("nan"),
        "spearman_mu_r": spearman(mu, r),
    }
    out["pass"] = bool(
        np.isfinite(out["cv_mu_over_r"]) and out["cv_mu_over_r"] >= CV_RATIO_MIN
        and np.isfinite(out["mu_retained"]) and out["mu_retained"] >= MU_RETAINED_MIN
    )
    return out


# ---------------------------------------------------------------------------
# GATE 2 -- is the prior direction the marginal action?
# ---------------------------------------------------------------------------
def gate2_prior_vs_marginal(A: np.ndarray, B: np.ndarray, mu: np.ndarray,
                            emitted_row: np.ndarray, slot: np.ndarray,
                            n_bins: int = 256) -> dict:
    """Does mu*A + B rank bins the way the policy's own action frequencies do?

    Per slot, because the seven action channels have very different marginals -- pooling them
    would compare the prior against a mixture no single decode position ever draws from.

    This is a NARRATIVE gate, not a validity gate. Failing it costs the word "prior" and the
    explanation that goes with it; the mechanical story (a fixed direction applied with the
    wrong strength) is untouched. Recorded separately for that reason.

    `n_argmax_over_mu_range` is the extra question the one-parameter form makes available: as
    mu sweeps its observed 5th-95th percentile, how many DIFFERENT bins does the prior prefer?
    1 means the prior is effectively a fixed direction after all.
    """
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    mu = np.asarray(mu, dtype=np.float64)
    rows = np.asarray(emitted_row, dtype=np.int64)
    slots = np.asarray(slot, dtype=np.int64)

    mu_ref = float(np.median(mu))
    prior = mu_ref * A + B                                    # [256]

    per_slot = {}
    for s in np.unique(slots):
        cnt = np.bincount(rows[slots == s], minlength=n_bins).astype(np.float64)
        per_slot[int(s)] = spearman(prior, np.log1p(cnt))
    vals = np.array([v for v in per_slot.values() if np.isfinite(v)], dtype=np.float64)

    lo, hi = np.percentile(mu, [5, 95])
    grid = np.linspace(lo, hi, 25)
    argmaxes = {int(np.argmax(m * A + B)) for m in grid}

    out = {
        "n": int(rows.size),
        "mu_ref": mu_ref,
        "per_slot_rho": {str(k): float(v) for k, v in per_slot.items()},
        "median_slot_rho": float(np.median(vals)) if vals.size else float("nan"),
        "min_slot_rho": float(vals.min()) if vals.size else float("nan"),
        "pooled_rho": spearman(prior, np.log1p(np.bincount(rows, minlength=n_bins))),
        "n_argmax_over_mu_range": len(argmaxes),
        "mu_p5_p95": [float(lo), float(hi)],
    }
    out["pass"] = bool(np.isfinite(out["median_slot_rho"])
                       and out["median_slot_rho"] >= RHO_MARGINAL_MIN)
    return out


# ---------------------------------------------------------------------------
# GATE 3 -- does bias-share predict deviation from the expert?
# ---------------------------------------------------------------------------
def gate3_share_predicts_deviation(share: np.ndarray, deviation: np.ndarray,
                                   margin: np.ndarray, activity: np.ndarray) -> dict:
    """Rank correlation of bias-share with |emitted bin - demo bin|, controlled.

    TWO CONTROLS, BOTH MANDATORY (EXPERIMENT_PLAN.md commitment #2):

      * `margin` -- the top-2 logit gap from `readout.top2_margin`. Weak features -> unsure
        model -> deviates from the expert is nearly definitional, so bias-share must beat the
        plain confidence signal or it IS the plain confidence signal.
      * `activity` -- how much the features fired (sum of |z|, or the active count). The
        standing base-rate control that killed the firing metrics in the first place.

    The verdict requires the partial to clear the bar AND to keep the raw correlation's sign:
    a partial that flips sign under control is not a weakened effect, it is a different one.
    """
    share = np.asarray(share, dtype=np.float64)
    dev = np.asarray(deviation, dtype=np.float64)
    out = {
        "n": int(np.isfinite(share).sum()),
        "raw_rho": spearman(share, dev),
        "partial_margin": rank_partial(dev, share, [margin]),
        "partial_activity": rank_partial(dev, share, [activity]),
        "partial_both": rank_partial(dev, share, [margin, activity]),
        "margin_alone_rho": spearman(margin, dev),
    }
    pb, raw = out["partial_both"], out["raw_rho"]
    out["pass"] = bool(
        np.isfinite(pb) and np.isfinite(raw)
        and abs(pb) >= RHO_PARTIAL_MIN and np.sign(pb) == np.sign(raw)
    )
    return out


# ---------------------------------------------------------------------------
# GATE 4 -- does re-scaling the prior move the argmax toward the expert?
# ---------------------------------------------------------------------------
def gate4_lambda_sweep(feat_scores: np.ndarray, prior_scores_: np.ndarray,
                       demo_bin: np.ndarray, emitted_bin: np.ndarray,
                       share: np.ndarray, lambdas=None, tail_q: float = 0.90,
                       n_bins: int = 256) -> dict:
    """Sweep score(t) = feat(t) + lambda * prior(t), re-argmax, measure distance to the expert.

    `feat_scores` is [n, 256] of l2 * (z @ S) and `prior_scores_` is [n, 256] of mu*A + B, both
    with r factored out -- which is exact for the argmax, since r > 0 scales every bin alike.

    THE BASELINE THAT MATTERS IS lambda = 1, NOT THE MODEL. At lambda = 1 this is the SAE
    RECONSTRUCTION's argmax, and the reconstruction does not reproduce the model's own argmax
    -- that is the L1 gate which stalled at 0.72-0.76 and caused the pivot to sufficiency. So
    every lambda is scored against lambda = 1 under the same approximation, and
    `recon_agreement` reports how often lambda = 1 recovers the emitted bin. If that agreement
    is low the whole sweep is being run inside a lossy approximation and the deltas are not
    interpretable; the driver refuses on it rather than reporting a number.

    THE PREDICTED SIGNATURE. The hypothesis is that the prior helps on typical decisions and
    hurts on atypical ones, so a UNIFORM lambda < 1 should hurt on average and help in the
    high-bias-share tail. `signature_holds` checks exactly that. Improvement everywhere
    instead means the readout is globally miscalibrated -- a real finding, but one that needs
    no feature decomposition and does not support an adaptive rule.
    """
    if lambdas is None:
        lambdas = [0.0, 0.25, 0.5, 0.75, 0.9, 1.0, 1.1, 1.25, 1.5, 2.0]
    Fm = np.asarray(feat_scores, dtype=np.float64)
    Pm = np.asarray(prior_scores_, dtype=np.float64)
    demo = np.asarray(demo_bin, dtype=np.float64)
    emit = np.asarray(emitted_bin, dtype=np.float64)
    sh = np.asarray(share, dtype=np.float64)

    ok = np.isfinite(sh) & np.isfinite(demo) & np.isfinite(emit)
    thr = np.nanquantile(sh[ok], tail_q) if ok.any() else np.nan
    tail = ok & (sh >= thr)

    rows = {}
    for lam in lambdas:
        rowarg = np.argmax(Fm + lam * Pm, axis=1)
        bin_ = n_bins - rowarg                      # readout.bin_index_from_row
        d = np.abs(bin_ - demo)
        rows[float(lam)] = {
            "mean_dev": float(np.mean(d[ok])) if ok.any() else float("nan"),
            "tail_dev": float(np.mean(d[tail])) if tail.any() else float("nan"),
            "agree_emitted": float(np.mean(bin_[ok] == emit[ok])) if ok.any() else float("nan"),
        }

    base = rows[1.0]
    gains = {lam: (base["tail_dev"] - v["tail_dev"]) / base["tail_dev"]
             for lam, v in rows.items()
             if lam != 1.0 and np.isfinite(base["tail_dev"]) and base["tail_dev"] > 0}
    best_lam = max(gains, key=gains.get) if gains else float("nan")
    best_tail_gain = gains[best_lam] if gains else float("nan")
    best_mean_gain = (
        (base["mean_dev"] - rows[best_lam]["mean_dev"]) / base["mean_dev"]
        if gains and np.isfinite(base["mean_dev"]) and base["mean_dev"] > 0 else float("nan")
    )

    out = {
        "n": int(ok.sum()),
        "n_tail": int(tail.sum()),
        "tail_quantile": float(tail_q),
        "share_threshold": float(thr) if np.isfinite(thr) else float("nan"),
        "recon_agreement": base["agree_emitted"],
        "emitted_mean_dev": float(np.mean(np.abs(emit[ok] - demo[ok]))) if ok.any() else float("nan"),
        "by_lambda": rows,
        "best_lambda": float(best_lam) if np.isfinite(best_lam) else float("nan"),
        "best_tail_gain": float(best_tail_gain),
        "best_mean_gain": float(best_mean_gain),
    }
    out["signature_holds"] = bool(
        np.isfinite(best_tail_gain) and np.isfinite(best_mean_gain)
        and best_tail_gain > best_mean_gain
    )
    out["pass"] = bool(np.isfinite(best_tail_gain) and best_tail_gain >= TAIL_GAIN_MIN)
    return out


# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------
def verdict_table(gates: dict) -> str:
    """One line per gate: name, verdict, and the statistic the verdict turned on.

    Reports the FIRST failure as the stopping point, because the ladder is sequential: a gate
    whose predecessor failed is being computed on a quantity that has already been shown not to
    exist, and its number should not be read.
    """
    keys = [
        ("gate0", "bias is variable / mu-coupled", "mu_share"),
        ("gate1", "mu survives division by r", "mu_retained"),
        ("gate2", "prior looks like the marginal", "median_slot_rho"),
        ("gate3", "share predicts deviation | controls", "partial_both"),
        ("gate4", "lambda moves argmax toward expert", "best_tail_gain"),
    ]
    lines, stopped = [], False
    for k, label, stat in keys:
        g = gates.get(k)
        if g is None:
            lines.append(f"  {k}  {label:<38}  NOT RUN")
            continue
        v = g.get(stat, float("nan"))
        mark = "PASS" if g.get("pass") else "FAIL"
        if stopped:
            mark += " (moot: an earlier gate failed)"
        lines.append(f"  {k}  {label:<38}  {mark:<32}  {stat}={v:+.4f}")
        if not g.get("pass"):
            stopped = True
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# expert action -> bin index (the one piece here with an external convention to match)
# ---------------------------------------------------------------------------
def demo_bin_index(action: np.ndarray, q01: np.ndarray, q99: np.ndarray,
                   mask: np.ndarray, n_bins: int = 256) -> np.ndarray:
    """Expert action [.., 7] -> OpenVLA bin index [1, n_bins], mirroring ActionTokenizer.

    Normalise to [-1, 1] on the masked dimensions (unmasked ones, typically the gripper, are
    already in range and are passed through), then `np.digitize` against `linspace(-1, 1,
    n_bins)`. The result is directly comparable to the model's `n_bins - row`, which is the
    convention `mrvla.readout.bin_index_from_row` documents.

    NOT VERIFIED AGAINST AN INSTALLED openvla HERE -- this box has no model. The canary in
    `main` is what catches a convention mismatch, and it must be believed over this docstring.
    """
    a = np.asarray(action, dtype=np.float64)
    lo, hi = np.asarray(q01, float), np.asarray(q99, float)
    rng = np.where(np.abs(hi - lo) > 1e-12, hi - lo, 1.0)
    norm = np.where(np.asarray(mask, bool)[None, :], 2.0 * (a - lo) / rng - 1.0, a)
    return np.digitize(np.clip(norm, -1.0, 1.0), np.linspace(-1.0, 1.0, n_bins))
