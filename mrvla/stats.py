"""Small-sample statistics used by the behavioural analyses. No scipy.

The repo deliberately avoids a scipy dependency (see `summarize_success.py`), so the few
distributions we need are implemented here once and imported everywhere rather than being
re-derived per script. Everything is closed-form or a short fixed-point iteration.

What lives here and why it matters for the paper:

* `wilson_interval`  -- an interval on a success rate that is honest at n = 200 episodes,
  where the normal approximation is not.
* `mcnemar_p` / `mcnemar_exact_p` -- the PAIRED test. Ablation replays the same LIBERO init
  states under every condition, so conditions are matched pair-by-pair; an unpaired
  two-proportion test throws that pairing away and loses most of the power the design bought.
* `paired_diff_ci` -- a confidence interval on the ablation damage. A null result is only
  interpretable with one of these: "damage is 0 +- 2 points" and "damage is 0 +- 20 points"
  are completely different claims and the point estimate alone cannot tell them apart.
* `mde_paired` / `required_pairs` -- the design's minimum detectable effect. This is the
  number that converts "we found no damage" into "we can exclude damage larger than X",
  which is a reportable bounded null rather than an absence of evidence.
"""

from __future__ import annotations

import math

__all__ = [
    "norm_cdf", "norm_ppf", "wilson_interval", "mcnemar_p", "mcnemar_exact_p",
    "paired_diff_ci", "mde_paired", "required_pairs", "rankdata_average", "tie_fraction",
]


def rankdata_average(x) -> "np.ndarray":
    """Ranks with TIES AVERAGED -- the textbook definition Spearman assumes.

    The idiom used elsewhere in this repo, `np.argsort(np.argsort(x))`, does not do this: it
    breaks ties by ARRAY INDEX, so a block of equal values receives distinct, arbitrary ranks
    determined by feature ordering. For strictly continuous inputs the two agree, but ties are
    not hypothetical here -- `base_rate` is a count over a fixed denominator, so every rarely
    firing feature ties with its neighbours, and a per-channel causal mass is exactly zero for
    any feature that never fires at that slot.

    Index-broken ties inject an arbitrary ordering into whatever is being correlated or
    residualised. Use this instead wherever ties are possible.
    """
    import numpy as np
    x = np.asarray(x, dtype=np.float64)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(x.size, dtype=np.float64)
    sx = x[order]
    i = 0
    while i < x.size:
        j = i
        while j + 1 < x.size and sx[j + 1] == sx[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j)       # average rank over the tied block
        i = j + 1
    return ranks


def tie_fraction(x) -> float:
    """Fraction of entries that share their value with at least one other entry.

    Report this beside any rank statistic computed with index-broken ranks: it is the size of
    the exposure. Near 0 means the distinction does not matter for that variable.
    """
    import numpy as np
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return float("nan")
    _, counts = np.unique(x, return_counts=True)
    return float((counts[counts > 1].sum()) / x.size)


def norm_cdf(x: float) -> float:
    """Standard normal CDF."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


# Acklam's rational approximation to the standard normal quantile; |error| < 1.15e-9 over
# the whole open interval, which is far tighter than anything downstream needs.
_A = (-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
      1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00)
_B = (-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
      6.680131188771972e+01, -1.328068155288572e+01)
_C = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
      -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00)
_D = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
      3.754408661907416e+00)


def norm_ppf(p: float) -> float:
    """Standard normal quantile (inverse CDF). Raises on p outside (0, 1)."""
    if not 0.0 < p < 1.0:
        raise ValueError(f"norm_ppf needs 0 < p < 1, got {p}")
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((_C[0] * q + _C[1]) * q + _C[2]) * q + _C[3]) * q + _C[4]) * q + _C[5]) / \
               ((((_D[0] * q + _D[1]) * q + _D[2]) * q + _D[3]) * q + 1)
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((_C[0] * q + _C[1]) * q + _C[2]) * q + _C[3]) * q + _C[4]) * q + _C[5]) / \
               ((((_D[0] * q + _D[1]) * q + _D[2]) * q + _D[3]) * q + 1)
    q = p - 0.5
    r = q * q
    return (((((_A[0] * r + _A[1]) * r + _A[2]) * r + _A[3]) * r + _A[4]) * r + _A[5]) * q / \
           (((((_B[0] * r + _B[1]) * r + _B[2]) * r + _B[3]) * r + _B[4]) * r + 1)


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion (reliable at modest n and near 0/1)."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1 + z ** 2 / n
    centre = p + z ** 2 / (2 * n)
    margin = z * math.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2))
    # clamp: at k = 0 or k = n the algebra lands a hair outside [0, 1] in floating point,
    # and a reported success-rate bound must never be negative or above one
    lo, hi = (centre - margin) / denom, (centre + margin) / denom
    return (min(max(lo, 0.0), 1.0), min(max(hi, 0.0), 1.0))


def mcnemar_p(b01: int, b10: int) -> float:
    """Continuity-corrected McNemar test, two-sided, via the chi2(df=1) <-> normal identity.

    Only the discordant pairs carry information: concordant pairs cancel by construction.
    """
    if b01 + b10 == 0:
        return float("nan")
    stat = (abs(b01 - b10) - 1) ** 2 / (b01 + b10)
    return 2 * (1 - norm_cdf(math.sqrt(stat)))


def mcnemar_exact_p(b01: int, b10: int) -> float:
    """Exact (binomial) McNemar, two-sided. Use this when the discordant count is small.

    The continuity-corrected chi2 above is CONSERVATIVE at small m -- it overstates p and so
    loses real power -- while the uncorrected version is anticonservative. Neither is
    acceptable in a per-task ablation cell of 20 episodes, where m is often under 10, so the
    exact binomial is what the per-task table reports.
    """
    m = b01 + b10
    if m == 0:
        return float("nan")
    lo = min(b01, b10)
    # two-sided exact p = 2 * P(X <= lo) for X ~ Bin(m, 0.5), capped at 1
    tail = sum(math.comb(m, i) for i in range(lo + 1)) / (2.0 ** m)
    return float(min(1.0, 2.0 * tail))


def paired_diff_ci(b01: int, b10: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    """CI on the paired proportion difference d = (b10 - b01)/n. Returns (d, lo, hi).

    Sign convention for ablation: pass b10 = "baseline succeeded, ablated failed" so that
    positive d = DAMAGE. Variance is the standard paired-binomial form, which depends only on
    the discordant counts -- concordant pairs contribute nothing.
    """
    if n == 0:
        return (float("nan"), float("nan"), float("nan"))
    d = (b10 - b01) / n
    var = ((b01 + b10) - (b10 - b01) ** 2 / n) / (n * n)
    se = math.sqrt(max(var, 0.0))
    return (d, d - z * se, d + z * se)


def mde_paired(n: int, disc_rate: float, alpha: float = 0.05, power: float = 0.80) -> float:
    """Minimum detectable paired difference for McNemar at `n` pairs.

    `disc_rate` is the fraction of pairs expected to be discordant (estimate it from the
    observed run; it is the only nuisance parameter). Returns the smallest |d| this design
    detects with the requested power -- i.e. the bound a null result actually licenses.

    Derivation: conditional on m = n*disc_rate discordant pairs, b10 ~ Bin(m, p) and the test
    rejects when |p - 0.5|*sqrt(m) > z_alpha*0.5 + z_power*sqrt(p(1-p)). With d = (2p-1)*
    disc_rate this gives |d| = 2*disc_rate*(z_alpha*0.5 + z_power*sqrt(p(1-p)))/sqrt(m).
    p appears on both sides, so we iterate from the worst case p = 0.5; the iteration is
    monotone decreasing and converges in a handful of steps.
    """
    if n <= 0 or not 0.0 < disc_rate <= 1.0:
        return float("nan")
    m = n * disc_rate
    za, zb = norm_ppf(1 - alpha / 2), norm_ppf(power)
    p = 0.5
    d = float("nan")
    for _ in range(50):
        d = 2 * disc_rate * (za * 0.5 + zb * math.sqrt(p * (1 - p))) / math.sqrt(m)
        p_new = min(0.999, 0.5 + d / (2 * disc_rate))
        if abs(p_new - p) < 1e-12:
            p = p_new
            break
        p = p_new
    return float(min(d, 1.0))


def required_pairs(effect: float, disc_rate: float, alpha: float = 0.05,
                   power: float = 0.80) -> float:
    """Pairs needed to detect a paired difference of `effect` at the given power.

    The inverse of `mde_paired` under its worst-case p = 0.5 variance, so it is the
    conservative (slightly generous) sample size: n = disc_rate * ((z_alpha + z_power)/d)^2.
    """
    if effect <= 0 or not 0.0 < disc_rate <= 1.0:
        return float("nan")
    za, zb = norm_ppf(1 - alpha / 2), norm_ppf(power)
    return float(disc_rate * ((za + zb) / effect) ** 2)
