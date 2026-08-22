"""Is the decision running the RIGHT task's machinery? And does that predict how long it takes?

Two things that have never been tried, and they compose.

TASK MARGIN. `layer_31_attribution.npz` holds `C[g, j]` -- feature j's mean causal
contribution on task g, a [G, F] matrix Path A built for the participation ratio and then
never used again. For any closed-loop decision on task g we have the causal profile |phi|,
so we can ask which task's canonical profile it most resembles:

    margin = cos(|phi|, C[g]) - max over h != g of cos(|phi|, C[h])

Positive means the decision looks like its own task; negative means the policy is running
machinery that belongs to a different one -- a plausible failure mode for a multi-task
policy sharing one base checkpoint, and not measurable without a per-task causal
decomposition.

THE SUBTRACTION IS THE DESIGN, NOT A DETAIL. `cos(|phi|, C[g])` alone would be worthless:
results.md P2b shows the top-C features ARE essentially the always-on set (41/50 overlap
with the top base-rate features), so raw similarity would mostly measure "is the usual stuff
firing" -- base rate, the confound that has closed every metric route in this project.
Subtracting the best competing task cancels the shared always-on component, because it is
common to all G rows. What survives is the task-SPECIFIC part.

DURATION AMONG SUCCESSES. Episode failure turned out to be a degenerate target: `done` fires
only on success, so a failure always runs to the step cap and duration predicts the label at
AUROC 1.000. Restricting to SUCCESSFUL episodes and predicting how long each took removes
that entirely -- duration is then genuinely variable and has no definitional link to
anything. It is the first uncontaminated target in this line, and a rank correlation against
it uses strictly more information than binarising.

WITHIN TASK, ALWAYS. Tasks differ in typical duration, so a pooled correlation can read
"this task is slow" and report it as signal. Everything here is computed inside a task and
then averaged, the same discipline the per-task AUROC check applies.
"""

from __future__ import annotations

import numpy as np

from mrvla.prior_gates import rank_partial, spearman

__all__ = [
    "row_normalize", "task_similarity", "task_margin",
    "within_task_spearman", "within_task_partial",
]


def row_normalize(M: np.ndarray) -> np.ndarray:
    """L2-normalise each row; all-zero rows stay zero rather than becoming NaN."""
    M = np.asarray(M, dtype=np.float64)
    n = np.linalg.norm(M, axis=1, keepdims=True)
    return M / np.where(n > 0, n, 1.0)


def task_similarity(phi_abs: np.ndarray, C: np.ndarray) -> np.ndarray:
    """[n, G] cosine between each decision's |phi| profile and each task's C row."""
    return row_normalize(np.abs(np.asarray(phi_abs, dtype=np.float64))) @ row_normalize(C).T


def task_margin(sim: np.ndarray, task_of_decision: np.ndarray,
                row_perm: np.ndarray | None = None,
                standardize: bool = True) -> np.ndarray:
    """[n] how far the decision's own task stands above the competing tasks.

    `standardize=True` (the default) returns `(own - mean_others) / sd_others`; False
    returns the raw `own - max_others`.

    STANDARDISE, BECAUSE THE RAW MARGIN IS NOT SCALE-STABLE. P2b's always-on set inflates
    every C row alike, which drives all G cosines toward each other and compresses the raw
    margin toward zero -- on a fixture with a large shared component it shrinks about 75x
    while staying correctly SIGNED. Rank correlation would not care, but the raw value stops
    being comparable across suites or SAEs, and a compressed margin is easier to swamp with
    noise. Dividing by the spread ACROSS tasks removes the shared component's effect on the
    scale, because it inflates every competitor equally.

    `row_perm` implements the shuffle control: pass a permutation and task g is scored
    against task perm[g]'s profile instead of its own. The margin MUST collapse under it --
    if it does not, the statistic is reading what every row shares rather than anything
    task-specific, which is the failure the subtraction exists to prevent.
    """
    S = np.asarray(sim, dtype=np.float64)
    g = np.asarray(task_of_decision, dtype=np.int64)
    if row_perm is not None:
        g = np.asarray(row_perm, dtype=np.int64)[g]
    n, G = S.shape
    rows = np.arange(n)
    own = S[rows, g]
    other = S.copy()
    other[rows, g] = np.nan
    if not standardize:
        return own - np.nanmax(other, axis=1)
    mu = np.nanmean(other, axis=1)
    sd = np.nanstd(other, axis=1, ddof=1) if G > 2 else np.zeros(n)
    return np.where(sd > 0, (own - mu) / np.where(sd > 0, sd, 1.0), np.nan)


def within_task_spearman(x: np.ndarray, y: np.ndarray, task: np.ndarray,
                         min_n: int = 8) -> dict:
    """Mean of the per-task rank correlations, plus the per-task values.

    Averaging correlations rather than pooling is deliberate: a pooled correlation over
    tasks with different typical durations measures the between-task difference, which is a
    property of the task set and not of the policy's internal state.
    """
    x, y = np.asarray(x, float), np.asarray(y, float)
    t = np.asarray(task)
    per = {}
    for g in np.unique(t):
        m = (t == g) & np.isfinite(x) & np.isfinite(y)
        if m.sum() >= min_n:
            per[int(g)] = spearman(x[m], y[m])
    vals = np.array([v for v in per.values() if np.isfinite(v)], dtype=np.float64)
    return {
        "mean": float(vals.mean()) if vals.size else float("nan"),
        "per_task": per,
        "n_tasks": int(vals.size),
        "n_positive": int((vals > 0).sum()),
    }


def within_task_partial(x: np.ndarray, y: np.ndarray, controls: list,
                        task: np.ndarray, min_n: int = 8) -> dict:
    """`within_task_spearman` with the controls residualised out inside each task."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    cols = [np.asarray(c, float) for c in controls]
    t = np.asarray(task)
    per = {}
    for g in np.unique(t):
        m = (t == g) & np.isfinite(x) & np.isfinite(y)
        for c in cols:
            m &= np.isfinite(c)
        if m.sum() >= min_n:
            per[int(g)] = rank_partial(y[m], x[m], [c[m] for c in cols])
    vals = np.array([v for v in per.values() if np.isfinite(v)], dtype=np.float64)
    return {
        "mean": float(vals.mean()) if vals.size else float("nan"),
        "per_task": per,
        "n_tasks": int(vals.size),
        "n_positive": int((vals > 0).sum()),
    }
