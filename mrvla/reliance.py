"""Do internal reliance signals predict episode failure? The analysis side.

`mrvla/action_rollout.py` produces the missing asset -- closed-loop decisions carrying both
the action-position residual and the episode outcome. This module turns those into
per-decision signals, aggregates them per episode, and scores each against failure.

FOUR SIGNALS, ONE OF WHICH IS THE POINT
---------------------------------------
    mu_t        fraction of |phi| carried by LOW-ADJUSTED-BREADTH features.  <-- the one
                that uses Path A's axis, and the plan's Section 3.2a reliance measure.
    share       the constant prior's fraction of the action margin. Gate 0 showed the bias
                is a fixed vector, so this reduces to "how weak were the features here".
    phi_total   raw feature drive, sum |phi|.
    margin      top-2 logit gap. NOT a finding -- this is the BASELINE. Any signal that
                does not beat it is a confidence score with extra steps, and the whole
                point of a mechanistic measure is to carry information a scalar does not.

`mu_t` is r-free and l2-free by construction: it is a RATIO of two sums that both carry the
same (l2/r) prefactor, so the frozen-r caveat does not touch it.

THE CANARY THAT GATES EVERYTHING
--------------------------------
The SAE was trained on demo-replay action-position residuals; these rollouts visit states
the policy generated itself. Sufficiency was 0.936 there. If it collapses here, every signal
above is read off a decomposition that no longer holds, and no other check in the pipeline
would notice -- the numbers would look perfectly reasonable. `sufficiency` is computed on
the rollout residuals first and the driver refuses to report AUROCs if it falls through the
floor.

TWO DESIGN CONSTRAINTS INHERITED FROM THE PLAN
----------------------------------------------
* EARLY WINDOWS, reported as a curve. Section 3.2a requires measuring before divergence,
  or a positive result is just "episodes already going wrong look like episodes going
  wrong". Rather than pick a window, every window is scored: first 5/10/20/50 steps, plus
  whole-episode mean and max. If only the whole-episode aggregate works, that is reverse
  causation and the curve makes it visible instead of hiding it.
* PER TASK, always. At ~76% success the failures may concentrate in two or three hard
  tasks, in which case a pooled AUROC would be reading "reliance differs by task" and
  calling it prediction. Pooled and per-task are reported together, never pooled alone.
"""

from __future__ import annotations

import numpy as np

from mrvla.stats import rankdata_average

__all__ = [
    "through_origin_slope", "sufficiency", "reliance_signals",
    "aggregate_episodes", "auroc", "auroc_boot", "DEFAULT_WINDOWS", "SUFFICIENCY_FLOOR",
]

DEFAULT_WINDOWS = (5, 10, 20, 50)

# Demo replay gave 0.936. A rollout value far below that means the decomposition does not
# transfer to self-generated states. Set at the same 0.80 bar the original viability gate
# used, so this is the pre-registered threshold and not one chosen after seeing the number.
SUFFICIENCY_FLOOR = 0.80


# ---------------------------------------------------------------------------
# the canary
# ---------------------------------------------------------------------------
def through_origin_slope(true: np.ndarray, comp: np.ndarray) -> float:
    """Slope of `comp` on `true` through the origin: sum(true*comp) / sum(true^2).

    The estimator `run_attribution.py` uses for sufficiency. Because feat + bias + err is
    exactly `true`, the three slopes sum to 1 by identity -- which is also a free check that
    the caller assembled the components correctly.
    """
    t = np.asarray(true, dtype=np.float64)
    c = np.asarray(comp, dtype=np.float64)
    m = np.isfinite(t) & np.isfinite(c)
    den = float((t[m] * t[m]).sum())
    return float((t[m] * c[m]).sum() / den) if den > 0 else float("nan")


def sufficiency(true: np.ndarray, feat: np.ndarray, bias: np.ndarray) -> dict:
    """Decompose the action margin into features / bias / error on THESE decisions."""
    err = np.asarray(true, float) - np.asarray(feat, float) - np.asarray(bias, float)
    s_f = through_origin_slope(true, feat)
    s_b = through_origin_slope(true, bias)
    s_e = through_origin_slope(true, err)
    out = {
        "features": s_f, "bias": s_b, "error": s_e,
        "features_plus_bias": s_f + s_b,
        "sums_to_one": float(s_f + s_b + s_e),
        "n": int(np.isfinite(true).sum()),
    }
    out["pass"] = bool(np.isfinite(s_f + s_b) and (s_f + s_b) >= SUFFICIENCY_FLOOR)
    return out


# ---------------------------------------------------------------------------
# per-decision signals
# ---------------------------------------------------------------------------
def reliance_signals(z: np.ndarray, l2: np.ndarray, mu: np.ndarray, r: np.ndarray,
                     S: np.ndarray, A: np.ndarray, B: np.ndarray, L: np.ndarray,
                     rows: np.ndarray, low_mask: np.ndarray) -> dict:
    """All four signals plus the components the sufficiency canary needs. Arrays are [n].

    `S` must be the CONTRAST-CENTRED signature matrix, matching `A`/`B` from
    `prior_gates.prior_vectors`; an uncentred `S` puts the feature term on a different scale
    from the prior and silently inflates `share`.

    `low_mask` [F] marks the low-adjusted-breadth features whose share of |phi| is `mu_t`.
    Codes are non-negative after TopK+ReLU, so |z_j * S[j,t]| is z_j * |S[j,t]| and the sums
    need no per-feature sign handling.
    """
    z = np.asarray(z, dtype=np.float64)
    rows = np.asarray(rows, dtype=np.int64)
    n = rows.size
    at = np.arange(n)

    Srows = np.asarray(S, dtype=np.float64)[:, rows].T          # [n, F] signature at each bin
    feat = np.asarray(l2, float) * (z * Srows).sum(axis=1)
    bias = np.asarray(mu, float) * np.asarray(A, float)[rows] + np.asarray(B, float)[rows]
    Lc = np.asarray(L, dtype=np.float64)
    true = Lc[at, rows] - Lc.mean(axis=1)
    err = true - feat - bias

    absS = np.abs(Srows)
    tot = (z * absS).sum(axis=1)
    low = (z[:, np.asarray(low_mask, bool)] * absS[:, np.asarray(low_mask, bool)]).sum(axis=1)

    den = np.abs(feat) + np.abs(bias) + np.abs(err)
    prefactor = np.asarray(l2, float) / np.asarray(r, float)

    part = np.argpartition(-Lc, 1, axis=1)[:, :2]
    top2 = np.take_along_axis(Lc, part, axis=1)
    margin = np.abs(top2[:, 0] - top2[:, 1])

    return {
        "mu_t": np.where(tot > 0, low / np.where(tot > 0, tot, 1.0), np.nan),
        "share": np.where(den > 0, np.abs(bias) / np.where(den > 0, den, 1.0), np.nan),
        "phi_total": prefactor * tot,
        "margin": margin,
        "feat": feat, "bias": bias, "true": true,
    }


# ---------------------------------------------------------------------------
# episode aggregation
# ---------------------------------------------------------------------------
def aggregate_episodes(values: np.ndarray, episode: np.ndarray, timestep: np.ndarray,
                       success: np.ndarray, windows=DEFAULT_WINDOWS) -> dict:
    """Per-episode aggregates of a per-decision signal, plus that episode's outcome.

    Returns `episodes`, `success`, and one vector per aggregate: `mean`, `max`, and
    `first{N}` for each window. Windows are in TIMESTEPS, not rows -- a timestep is seven
    rows (one per action slot), and a window measured in rows would silently cover a
    seventh of the intended episode prefix.

    An episode shorter than a window contributes whatever it has; an episode with no rows in
    the window gets NaN and is dropped by the scorer rather than imputed.
    """
    ep = np.asarray(episode, dtype=np.int64)
    ts = np.asarray(timestep, dtype=np.int64)
    v = np.asarray(values, dtype=np.float64)
    sc = np.asarray(success, dtype=np.int64)

    uniq = np.unique(ep)
    out: dict = {"episodes": uniq, "success": np.empty(uniq.size, dtype=np.int64)}
    keys = ["mean", "max"] + [f"first{w}" for w in windows]
    for k in keys:
        out[k] = np.full(uniq.size, np.nan)

    for i, e in enumerate(uniq):
        m = ep == e
        vals, tss = v[m], ts[m]
        out["success"][i] = int(sc[m][0])
        fin = np.isfinite(vals)
        if fin.any():
            out["mean"][i] = float(vals[fin].mean())
            out["max"][i] = float(vals[fin].max())
        for w in windows:
            wm = fin & (tss < w)
            if wm.any():
                out[f"first{w}"][i] = float(vals[wm].mean())
    return out


# ---------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------
def auroc(scores: np.ndarray, failure: np.ndarray) -> float:
    """AUROC for `scores` predicting FAILURE (failure = 1). 0.5 = no information.

    Mann-Whitney U with ties averaged. Above 0.5 means a higher signal goes with failure;
    BELOW 0.5 is not a null -- it means the signal predicts SUCCESS, which is a real finding
    with the opposite sign and must not be reported as "no effect".
    """
    s = np.asarray(scores, dtype=np.float64)
    y = np.asarray(failure, dtype=np.int64)
    m = np.isfinite(s) & np.isfinite(y)
    s, y = s[m], y[m]
    n1 = int((y == 1).sum())
    n0 = int((y == 0).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    # rankdata_average is 0-BASED (mrvla/stats.py: a lone value ranks 0). The Mann-Whitney
    # identity below is written for 1-based ranks, so the +1 is load-bearing -- without it
    # perfect separation scores 0.67 rather than 1.0, which looks like a weak real effect.
    r = rankdata_average(s) + 1.0
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def auroc_boot(scores: np.ndarray, failure: np.ndarray, n_boot: int = 2000,
               seed: int = 0) -> dict:
    """AUROC with a percentile bootstrap CI, resampling EPISODES (the independent unit).

    Decisions within an episode are heavily correlated, so resampling rows would produce an
    interval several times too narrow. Each entry here is already one episode, so a plain
    resample of the vector is the correct unit.
    """
    s = np.asarray(scores, dtype=np.float64)
    y = np.asarray(failure, dtype=np.int64)
    m = np.isfinite(s) & np.isfinite(y)
    s, y = s[m], y[m]
    point = auroc(s, y)
    n = s.size
    if not np.isfinite(point) or n < 8:
        return {"auroc": point, "lo": float("nan"), "hi": float("nan"),
                "n": int(n), "n_fail": int((y == 1).sum())}
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        a = auroc(s[idx], y[idx])
        if np.isfinite(a):
            vals.append(a)
    lo, hi = (np.percentile(vals, [2.5, 97.5]) if vals else (np.nan, np.nan))
    return {"auroc": point, "lo": float(lo), "hi": float(hi),
            "n": int(n), "n_fail": int((y == 1).sum())}
