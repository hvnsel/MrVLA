"""Inventory-level comparison of causal roles: clustering, assignment, and distribution distance.

The distributional half of A4, and the primary route when `action_space_geometry.py` reports a
signature space too small for span-based matching. If the ambient space is cramped, every subset
spans nearly everything and "which subspace covers this feature" stops discriminating -- but
WHERE directions sit inside that small space still does.

So instead of asking whether feature i has a twin (or a coalition) in model B, we ask whether
the two models carve the same space into the same regions:

  * cluster each model's unit signatures on the sphere -> an INVENTORY of causal roles;
  * match the inventories across models by optimal assignment on centroid cosine;
  * compare OCCUPANCY -- how many features each model devotes to a matched role. Multiplicity
    differences are precisely what dictionary splitting is, so "same inventory, different
    multiplicities" is a measurable claim rather than a slogan;
  * and, without any clustering at all, compare the two signature DISTRIBUTIONS directly, so
    the conclusion does not rest on a choice of k.

Assignment is a real Hungarian solver rather than greedy max-matching. That matters twice over:
greedy double-books popular centroids and inflates match quality, and EXPERIMENT_PLAN.md §3.1
step 3 carries an outstanding commitment to report Hungarian alongside greedy at scale, which
`best_match_cosine` currently satisfies only when scipy happens to be installed. This
implementation has no dependency.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "normalize_rows", "spherical_kmeans", "hungarian_match", "greedy_match",
    "match_inventories", "cluster_occupancy", "sliced_wasserstein",
]


def normalize_rows(S: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    S = np.asarray(S, dtype=np.float64)
    n = np.linalg.norm(S, axis=1)
    out = np.zeros_like(S)
    nz = n > 0
    out[nz] = S[nz] / n[nz, None]
    return out, n


# ---------------------------------------------------------------------------
# clustering
# ---------------------------------------------------------------------------
def spherical_kmeans(X_hat: np.ndarray, k: int, rng: np.random.Generator,
                     n_iter: int = 60, n_init: int = 4) -> tuple[np.ndarray, np.ndarray, float]:
    """k-means on the unit sphere under cosine similarity. Returns (centroids, labels, inertia).

    Signatures are directions -- magnitude is the feature's strength, not its role -- so cosine
    is the right geometry and centroids are re-normalised each step. `inertia` is the total
    cosine similarity of points to their own centroid (HIGHER is better), used to pick among
    restarts.

    Empty clusters are reseeded onto the worst-fitting point rather than dropped, so k really is
    k and occupancy comparisons across models are not quietly comparing different numbers of
    roles.
    """
    X_hat = np.asarray(X_hat, dtype=np.float64)
    n = X_hat.shape[0]
    k = int(min(max(1, k), n))
    best = None

    for _ in range(max(1, n_init)):
        # k-means++ seeding under cosine distance: spread the initial roles out
        centers = [X_hat[rng.integers(n)]]
        for _ in range(k - 1):
            sim = X_hat @ np.asarray(centers).T
            d = np.clip(1.0 - sim.max(axis=1), 0.0, None) ** 2
            tot = d.sum()
            idx = int(rng.choice(n, p=d / tot)) if tot > 0 else int(rng.integers(n))
            centers.append(X_hat[idx])
        C = normalize_rows(np.asarray(centers))[0]

        lab = np.zeros(n, dtype=np.int64)
        for _ in range(n_iter):
            sim = X_hat @ C.T
            lab = np.argmax(sim, axis=1)
            fit = sim[np.arange(n), lab]
            newC = np.zeros_like(C)
            for j in range(k):
                m = lab == j
                newC[j] = X_hat[m].sum(axis=0) if m.any() else X_hat[int(np.argmin(fit))]
            newC = normalize_rows(newC)[0]
            if np.allclose(newC, C, atol=1e-10):
                C = newC
                break
            C = newC
        sim = X_hat @ C.T
        lab = np.argmax(sim, axis=1)
        inertia = float(sim[np.arange(n), lab].sum())
        if best is None or inertia > best[2]:
            best = (C, lab, inertia)
    return best


# ---------------------------------------------------------------------------
# assignment
# ---------------------------------------------------------------------------
def hungarian_match(cost: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Optimal one-to-one assignment MINIMISING total cost. Returns (rows, cols).

    Shortest-augmenting-path Hungarian, O(n^2 m), no scipy. Rectangular input is handled by
    transposing so the shorter side is assigned; every row of the shorter side gets exactly one
    column, and no column is used twice.

    Pass `-similarity` to maximise similarity.
    """
    cost = np.asarray(cost, dtype=np.float64)
    flip = cost.shape[0] > cost.shape[1]
    if flip:
        cost = cost.T
    n, m = cost.shape
    INF = float("inf")
    u = np.zeros(n + 1)
    v = np.zeros(m + 1)
    p = np.zeros(m + 1, dtype=np.int64)      # p[j] = row (1-based) assigned to column j
    way = np.zeros(m + 1, dtype=np.int64)

    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = np.full(m + 1, INF)
        used = np.zeros(m + 1, dtype=bool)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta, j1 = INF, -1
            for j in range(1, m + 1):
                if used[j]:
                    continue
                cur = cost[i0 - 1, j - 1] - u[i0] - v[j]
                if cur < minv[j]:
                    minv[j], way[j] = cur, j0
                if minv[j] < delta:
                    delta, j1 = minv[j], j
            for j in range(m + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while j0:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1

    rows, cols = [], []
    for j in range(1, m + 1):
        if p[j]:
            rows.append(p[j] - 1)
            cols.append(j - 1)
    r, c = np.array(rows, dtype=np.int64), np.array(cols, dtype=np.int64)
    order = np.argsort(r)
    r, c = r[order], c[order]
    return (c, r) if flip else (r, c)


def greedy_match(sim: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Each row takes its best column, independently. Columns may be reused.

    Reported alongside Hungarian because it is what `q_causal` does, and because the gap between
    them is informative: a large gap means a few popular centroids are absorbing many matches,
    which inflates apparent inventory agreement.
    """
    sim = np.asarray(sim, dtype=np.float64)
    cols = np.argmax(sim, axis=1)
    return np.arange(sim.shape[0]), cols


def match_inventories(C_a: np.ndarray, C_b: np.ndarray) -> dict:
    """Match two sets of unit centroids and report quality under both assignment rules."""
    S = np.asarray(C_a, dtype=np.float64) @ np.asarray(C_b, dtype=np.float64).T
    hr, hc = hungarian_match(-S)
    gr, gc = greedy_match(S)
    h_sim = S[hr, hc]
    g_sim = S[gr, gc]
    return {"hungarian_pairs": list(zip(hr.tolist(), hc.tolist())),
            "hungarian_similarity": h_sim,
            "hungarian_mean": float(h_sim.mean()) if h_sim.size else float("nan"),
            "greedy_similarity": g_sim,
            "greedy_mean": float(g_sim.mean()) if g_sim.size else float("nan"),
            "greedy_distinct_targets": int(np.unique(gc).size),
            "n_clusters": int(S.shape[0])}


def cluster_occupancy(labels: np.ndarray, k: int) -> np.ndarray:
    """[k] counts, as a fraction of all features -- how much of the dictionary each role gets.

    Fractions rather than counts so two models with different F remain comparable, and because
    the interesting quantity is the SHARE of representational budget a role commands.
    """
    labels = np.asarray(labels, dtype=np.int64)
    counts = np.bincount(labels, minlength=k)[:k].astype(np.float64)
    tot = counts.sum()
    return counts / tot if tot > 0 else counts


# ---------------------------------------------------------------------------
# distribution distance, no clustering required
# ---------------------------------------------------------------------------
def sliced_wasserstein(A: np.ndarray, B: np.ndarray, rng: np.random.Generator,
                       n_proj: int = 256, n_quantiles: int = 128) -> float:
    """Sliced 1-Wasserstein distance between two point clouds of unit signatures.

    Project both onto many random directions and average the 1-D Wasserstein distance between
    the projected distributions, read off matched quantiles so the two clouds need not have the
    same number of features. This is the k-free backup for the whole clustering analysis: if the
    inventory conclusion depends on the choice of k, this says so, and if it does not, this
    corroborates it without one.

    Smaller = the two models' causal roles are distributed more alike. Only meaningful against
    the same floors the rest of A4 uses (random dictionary, seed pair), never on its own.
    """
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    d = A.shape[1]
    P = rng.standard_normal((n_proj, d))
    P /= np.linalg.norm(P, axis=1, keepdims=True).clip(min=1e-12)
    qs = (np.arange(n_quantiles) + 0.5) / n_quantiles
    pa = np.sort(A @ P.T, axis=0)
    pb = np.sort(B @ P.T, axis=0)
    ia = np.quantile(pa, qs, axis=0)
    ib = np.quantile(pb, qs, axis=0)
    return float(np.abs(ia - ib).mean())
