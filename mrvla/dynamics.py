"""Does the IDENTITY of the driving coalition, or its churn over time, predict failure?

Four level-based signals came back null against episode failure (mu_t, share, phi_total,
top-2 margin), all of them a projection of the 2048-dim code onto one axis chosen in
advance. This module tries the two things that were never tried: whether the coalition's
STABILITY over time carries the signal, and whether a FITTED projection finds an axis that
four hand-built ones missed.

TWO SIGNALS

  churn(t)        1 - Jaccard(S_t, S_{t-1}) over the top-m features by |phi|, per decode
                  slot. Level statistics ask how hard things are pushing; this asks whether
                  the SAME things are pushing. A policy can have entirely normal magnitudes
                  while flip-flopping between two internal states.

  return_rate     the fraction of timesteps where Jaccard(S_t, S_{t-2}) exceeds
                  Jaccard(S_t, S_{t-1}) -- the coalition resembles two steps ago more than
                  one step ago, i.e. a period-2 limit cycle in feature space. This is the
                  tighter hypothesis: on LIBERO "failure" means the episode did not finish
                  inside the step cap, and the canonical way that happens is a reach-retry
                  oscillation, which is exactly what a period-2 cycle looks like.

THE BASELINE THAT DECIDES WHETHER EITHER MEANS ANYTHING. A dithering robot is visible in
its own commanded actions -- no dictionary required. `action_dynamics` computes the same
period-2 statistic in action space from the executed 7-vectors. Feature churn has to beat
action churn or the finding is "the arm is oscillating, and so are the features that drive
it", which needs none of this machinery.

THE PROBE, AND THE COMPARISON THAT MAKES IT DECISIVE. `probe_loto` fits ridge regression on
per-episode feature means, leave-one-TASK-out. Tasks differ in base failure rate, so random
folds let the fit memorise task identity and report a number that will not generalise; LOTO
forces it onto a task it never saw, the same discipline Path A's estimator already uses.
Run it on the SAE codes AND on the raw residual: if a probe on `z` does no better than the
same probe on `h`, the dictionary is a lossy reparameterisation and buys nothing here --
which is the honest answer to whether this machinery earns its keep, and is what a probe on
hidden states (SAFECAST and its relatives) already does without an SAE.

Ridge rather than logistic: closed form, no optimiser, no convergence failure to
misdiagnose, and for RANKING -- which is all AUROC uses -- the difference is immaterial. The
dual form is used because n_episodes (500) is far below n_features (2048/4096), so the
solve is 500x500 rather than 4096x4096.

A LENGTH TRAP INHERITED FROM THE LAST ROUND. Failures always run to the step cap and
successes stop early, so an online AUROC(t) curve silently drops successes as t grows: past
the shortest success, the surviving sample is nearly all failures and the curve measures
duration again. `max_unbiased_t` returns the largest t at which every episode still
contributes, and the curve reports its own n_ok/n_fail at every point so the bias is
visible rather than assumed away.
"""

from __future__ import annotations

import numpy as np

from mrvla.stats import rankdata_average

__all__ = [
    "topk_sets", "jaccard_at_lag", "coalition_dynamics", "action_dynamics",
    "episode_kinematics",
    "prefix_means", "max_unbiased_t", "auroc_curve", "ridge_dual_loto", "probe_loto",
    "weight_breadth_skew",
]


# ---------------------------------------------------------------------------
# coalitions
# ---------------------------------------------------------------------------
def topk_sets(scores: np.ndarray, m: int) -> np.ndarray:
    """[n, F] -> [n, m] SORTED indices of the m largest entries per row.

    Sorted because `jaccard_at_lag` counts intersections by concatenating two rows and
    looking for adjacent duplicates, which needs each row's own entries to be distinct (they
    are: they are feature indices) but not the rows to be ordered relative to each other.
    """
    S = np.asarray(scores)
    m = int(min(m, S.shape[1]))
    idx = np.argpartition(-S, m - 1, axis=1)[:, :m]
    return np.sort(idx, axis=1).astype(np.int32)


def _sorted_lag_view(episode, slot, timestep, lag):
    """Order rows by (episode, slot, timestep) and mark rows with a valid partner `lag` back.

    Returns (order, prev, valid) all in SORTED space. Comparing at the same decode slot
    matters: P5b shows the gripper is bias-driven and behaves nothing like the arm channels,
    so pooling all seven into one set would blur them together.
    """
    e = np.asarray(episode, dtype=np.int64)
    s = np.asarray(slot, dtype=np.int64)
    t = np.asarray(timestep, dtype=np.int64)
    order = np.lexsort((t, s, e))
    es, ss, ts = e[order], s[order], t[order]
    n = order.size
    cur = np.arange(n)
    prev = cur - lag
    valid = np.zeros(n, dtype=bool)
    m = prev >= 0
    valid[m] = ((es[m] == es[prev[m]]) & (ss[m] == ss[prev[m]])
                & ((ts[m] - ts[prev[m]]) == lag))
    return order, prev, valid


def jaccard_at_lag(sets: np.ndarray, episode, slot, timestep, lag: int) -> np.ndarray:
    """[n] Jaccard between each row's coalition and the same slot `lag` timesteps earlier.

    NaN where no valid partner exists (episode start, or a gap in the timestep sequence).
    """
    A = np.asarray(sets)
    order, prev, valid = _sorted_lag_view(episode, slot, timestep, lag)
    As = A[order]
    out_sorted = np.full(order.size, np.nan)
    if valid.any():
        cur_i = np.flatnonzero(valid)
        both = np.concatenate([As[cur_i], As[prev[cur_i]]], axis=1)
        both.sort(axis=1)
        inter = (both[:, 1:] == both[:, :-1]).sum(axis=1).astype(np.float64)
        m2 = 2.0 * A.shape[1]
        out_sorted[cur_i] = inter / (m2 - inter)
    out = np.empty_like(out_sorted)
    out[order] = out_sorted
    return out


def coalition_dynamics(sets: np.ndarray, episode, slot, timestep) -> dict:
    """`churn` = 1 - J(t, t-1); `returns` = 1.0 where J(t, t-2) > J(t, t-1), else 0.0.

    Both are per-row and NaN where the lag is unavailable, so episode boundaries drop out of
    the aggregates instead of contributing a fabricated value.
    """
    j1 = jaccard_at_lag(sets, episode, slot, timestep, 1)
    j2 = jaccard_at_lag(sets, episode, slot, timestep, 2)
    both = np.isfinite(j1) & np.isfinite(j2)
    ret = np.full(j1.size, np.nan)
    ret[both] = (j2[both] > j1[both]).astype(np.float64)
    return {"churn": 1.0 - j1, "returns": ret, "j1": j1, "j2": j2}


def action_dynamics(actions: np.ndarray, episode, timestep) -> dict:
    """The SAME period-2 statistic in ACTION space -- the baseline feature churn must beat.

    `actions` is [n_timesteps, 7] of executed commands. Similarity is cosine, so a repeated
    motion counts as a return regardless of its magnitude. One row per timestep here, not
    per slot, so callers must pass timestep-level arrays.
    """
    A = np.asarray(actions, dtype=np.float64)
    slot0 = np.zeros(A.shape[0], dtype=np.int64)          # no slot axis in action space

    def cos_at(lag):
        order, prev, valid = _sorted_lag_view(episode, slot0, timestep, lag)
        As = A[order]
        out_s = np.full(order.size, np.nan)
        i = np.flatnonzero(valid)
        if i.size:
            a, b = As[i], As[prev[i]]
            den = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
            out_s[i] = np.where(den > 0, (a * b).sum(axis=1) / np.where(den > 0, den, 1), np.nan)
        out = np.empty_like(out_s)
        out[order] = out_s
        return out

    c1, c2 = cos_at(1), cos_at(2)
    both = np.isfinite(c1) & np.isfinite(c2)
    ret = np.full(c1.size, np.nan)
    ret[both] = (c2[both] > c1[both]).astype(np.float64)
    return {"churn": 1.0 - c1, "returns": ret}


# ---------------------------------------------------------------------------
# online scoring
# ---------------------------------------------------------------------------
def prefix_means(values, episode, timestep, t: int) -> tuple:
    """Per-episode mean of `values` over timesteps < t. Returns (episodes, means, lengths)."""
    v = np.asarray(values, dtype=np.float64)
    e = np.asarray(episode, dtype=np.int64)
    ts = np.asarray(timestep, dtype=np.int64)
    uniq = np.unique(e)
    means = np.full(uniq.size, np.nan)
    lens = np.zeros(uniq.size, dtype=np.int64)
    for i, ep in enumerate(uniq):
        m = e == ep
        lens[i] = int(np.unique(ts[m]).size)
        w = m & (ts < t) & np.isfinite(v)
        if w.any():
            means[i] = float(v[w].mean())
    return uniq, means, lens


def max_unbiased_t(lengths: np.ndarray, failure: np.ndarray) -> int:
    """Largest prefix length at which EVERY episode still contributes.

    Past this point the curve silently drops short episodes -- which, since successes are
    the short ones, means it starts measuring duration instead of the signal.
    """
    L = np.asarray(lengths, dtype=np.int64)
    return int(L.min()) if L.size else 0


def auroc_curve(values, episode, timestep, success, grid) -> list:
    """AUROC of the prefix mean against failure, at each t in `grid`.

    Each entry carries its own n_ok / n_fail, because a point computed after short episodes
    have dropped out is not comparable to one computed while they were all present.
    """
    from mrvla.reliance import auroc

    sc = np.asarray(success, dtype=np.int64)
    e = np.asarray(episode, dtype=np.int64)
    ep_succ = {int(ep): int(sc[e == ep][0]) for ep in np.unique(e)}
    rows = []
    for t in grid:
        uniq, means, lens = prefix_means(values, episode, timestep, t)
        keep = np.isfinite(means) & (lens >= t)          # only episodes that filled the window
        y = np.array([1 - ep_succ[int(ep)] for ep in uniq])
        rows.append({
            "t": int(t),
            "auroc": auroc(means[keep], y[keep]),
            "n_ok": int((y[keep] == 0).sum()), "n_fail": int((y[keep] == 1).sum()),
        })
    return rows


# ---------------------------------------------------------------------------
# the probe
# ---------------------------------------------------------------------------
def ridge_dual_loto(X: np.ndarray, y: np.ndarray, task: np.ndarray,
                    lam: float = 1.0) -> tuple:
    """Out-of-fold ridge scores, leave-one-TASK-out. Returns (scores, mean_weights).

    Dual form: with n episodes far below p features, `w = X'(XX' + lam I)^-1 y` costs an
    n x n solve instead of p x p. Labels are centred so the fit has no intercept to carry.
    Weights are averaged over folds and returned for the breadth-skew test; they are a
    by-product, and the score is what the AUROC uses.
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    g = np.asarray(task)
    scores = np.full(y.size, np.nan)
    W = np.zeros(X.shape[1])
    folds = 0
    for gi in np.unique(g):
        te = g == gi
        tr = ~te
        if tr.sum() < 5 or te.sum() < 1:
            continue
        Xtr, ytr = X[tr], y[tr] - y[tr].mean()
        mu = Xtr.mean(axis=0)
        Xc = Xtr - mu
        K = Xc @ Xc.T
        a = np.linalg.solve(K + lam * np.eye(K.shape[0]), ytr)
        w = Xc.T @ a
        scores[te] = (X[te] - mu) @ w
        W += w
        folds += 1
    return scores, (W / max(folds, 1))


def probe_loto(X, y, task, lam: float = 1.0, n_perm: int = 200, seed: int = 0) -> dict:
    """LOTO ridge probe with a label-permutation null. `y` is 1 for FAILURE.

    The null permutes labels WITHIN task, preserving each task's failure rate, so it cannot
    be beaten by simply learning which tasks fail more often -- that is the artefact LOTO is
    there to prevent, and the null has to hold it fixed too or it is not a floor for the
    thing being claimed.
    """
    from mrvla.reliance import auroc

    y = np.asarray(y, dtype=np.int64)
    g = np.asarray(task)
    s, w = ridge_dual_loto(X, y, g, lam)
    obs = auroc(s, y)
    rng = np.random.default_rng(seed)
    null = []
    for _ in range(n_perm):
        yp = y.copy()
        for gi in np.unique(g):
            m = g == gi
            yp[m] = rng.permutation(yp[m])
        sp, _ = ridge_dual_loto(X, yp, g, lam)
        a = auroc(sp, yp)
        if np.isfinite(a):
            null.append(a)
    null = np.asarray(null, dtype=np.float64)
    return {
        "auroc": obs, "n": int(y.size), "n_fail": int(y.sum()),
        "null_mean": float(null.mean()) if null.size else float("nan"),
        "null_p95": float(np.percentile(null, 95)) if null.size else float("nan"),
        "p_value": float((null >= obs).mean()) if null.size else float("nan"),
        "weights": w,
    }


def weight_breadth_skew(weights, adjusted_breadth, top_n: int = 50) -> dict:
    """Do the features the probe leans on skew LOW adjusted breadth?

    The bridge back to Path A. mu_t asked the same question with a hand-built ratio and came
    back null; this asks it of a fitted projection instead. A negative rank correlation, and
    a top-|w| set sitting below the median breadth percentile, would say failure is carried
    by narrow scene-specific machinery -- the original hypothesis, reached by a second and
    independent route. The opposite sign is equally reportable and more surprising.
    """
    w = np.abs(np.asarray(weights, dtype=np.float64))
    b = np.asarray(adjusted_breadth, dtype=np.float64)
    m = np.isfinite(w) & np.isfinite(b)
    if m.sum() < 10:
        return {"rho": float("nan"), "n": int(m.sum())}
    rw, rb = rankdata_average(w[m]), rankdata_average(b[m])
    rw = rw - rw.mean()
    rb = rb - rb.mean()
    den = np.sqrt((rw * rw).sum() * (rb * rb).sum())
    top = np.flatnonzero(m)[np.argsort(-w[m])[:top_n]]
    pct = rankdata_average(b[m]) / max(m.sum() - 1, 1) * 100.0
    lookup = dict(zip(np.flatnonzero(m).tolist(), pct.tolist()))
    return {
        "rho": float((rw * rb).sum() / den) if den > 0 else float("nan"),
        "n": int(m.sum()),
        "top_w_mean_breadth_percentile": float(np.mean([lookup[i] for i in top])),
        "top_w_frac_below_median_breadth": float(
            np.mean([lookup[i] < 50.0 for i in top])),
        "top_n": int(top_n),
    }


# ---------------------------------------------------------------------------
# how far did it actually travel? -- the distance-to-goal control
# ---------------------------------------------------------------------------
def episode_kinematics(actions: np.ndarray, episode, timestep, window: int) -> dict:
    """Per-episode motion over the first `window` timesteps, from the commanded actions.

    The leading alternative explanation for any early-window signal that predicts episode
    duration is DISTANCE: an episode starting with the gripper far from the object takes
    more steps AND may leave the early features with less to say, so distance would cause
    both. Pose is not logged, but the commanded actions ARE delta-poses, so the robot's own
    motion is recoverable without re-running anything.

        path      sum of ||delta xyz||  -- total distance travelled
        net       ||sum of delta xyz||  -- how far it actually got
        straight  net / path            -- ~1 travelling purposefully toward something far,
                                           ~0 milling around near something close
        rot       sum of ||delta rpy||  -- reorientation effort, which a far reach need not
                                           involve and a fiddly close one does

    A caveat worth stating rather than burying: these measure how far the policy DID move,
    not how far it NEEDED to. They are a proxy for the confound, not the confound itself.
    The exact control would parse initial object and end-effector pose out of LIBERO's init
    states, which needs the simulator and per-task body indices.
    """
    A = np.asarray(actions, dtype=np.float64)
    e = np.asarray(episode, dtype=np.int64)
    t = np.asarray(timestep, dtype=np.int64)
    uniq = np.unique(e)
    out = {k: np.full(uniq.size, np.nan) for k in ("path", "net", "straight", "rot")}
    out["episodes"] = uniq
    for i, ep in enumerate(uniq):
        m = (e == ep) & (t < window)
        if not m.any():
            continue
        d = A[m][:, :3]
        path = float(np.linalg.norm(d, axis=1).sum())
        net = float(np.linalg.norm(d.sum(axis=0)))
        out["path"][i] = path
        out["net"][i] = net
        out["straight"][i] = net / path if path > 0 else np.nan
        out["rot"][i] = float(np.linalg.norm(A[m][:, 3:6], axis=1).sum())
    return out
