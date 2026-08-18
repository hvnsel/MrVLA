"""One-to-many causal-signature matching: can model B's dictionary EXPRESS model A's roles?

Path B asks `q_causal = max_j cos(S_A_i, S_B_j)` -- does feature i have a twin. That is a
ONE-TO-ONE question, and it is structurally blind to the known failure mode of sparse
dictionaries. SAEs split and merge: §2.5 reports that a feature re-derived under a new seed
matches its twin only ~60% of the time, and the usual reason is that one seed's feature becomes
two or three in another. If model A represents a role as one feature and model B splits it
across three, `max_j cos` returns something mediocre while the SPAN of B's three contains a
direction at cos 0.95.

That blindness is not uniform across features, which is why it matters here. Broad, diffuse,
high-usage general features are the ones most likely to fragment; sharp specialists stay atomic.
So splitting alone could manufacture the entire "generals recur less" result -- the observed
Path B finding -- without any real difference in recurrence.

This module replaces the question with: greedily select m features from B whose signatures best
span S_A_i, and report the cosine to that span. Sweeping m nests the old result at m=1 and turns
the null into a testable curve: if splitting is the explanation, the general features' curve
rises steeply with m while the specialists' stays flat.

SIGN, AND WHY m=1 IS NOT EXACTLY q_causal
------------------------------------------
`run_causal_recurrence.best_match_cosine` takes the SIGNED max. Projection onto a span does not:
coefficients are free, so a span containing -v expresses v, and the m=1 point of this curve is
|max cos|, not the signed max. The two differ exactly for features whose best match is
anti-aligned. `anti_aligned_fraction` reports how often that happens, so the size of the
discontinuity is measured rather than assumed, and the published signed value is reported
alongside m=1 for comparability.

The deeper point: TopK codes are NON-NEGATIVE, so model B can only ever build a role as a
non-negative combination of its features -- it can add w_j but never subtract it. Unrestricted
projection therefore OVERSTATES what B's dictionary can express. That asymmetry is convenient:
a flat curve under an upper bound is a genuinely flat curve, so the NULL conclusion is safe. A
RISING curve is the case that needs the non-negative follow-up before it can be believed, and
`positive_only` gives a first pass at it by restricting selection to positively-correlated
features.

WHAT THE NULLS MUST MATCH
-------------------------
cos rises with m mechanically -- m arbitrary directions span more of anything. Every null here
therefore runs the identical sweep at the identical m against a dictionary of the identical size
(so the best-of-F_B selection effect is matched too). An unmatched null would manufacture a
positive result out of dimension counting alone.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "normalize_rows", "omp_curve", "signed_best_match", "anti_aligned_fraction",
    "random_signature_dictionary", "chance_corrected_retention", "signature_sharpness",
]


def normalize_rows(S: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(unit-norm rows, original norms). Zero rows stay zero -- they match nothing.

    Same convention as run_causal_recurrence, so signatures are interchangeable between them.
    """
    S = np.asarray(S, dtype=np.float64)
    n = np.linalg.norm(S, axis=1)
    out = np.zeros_like(S)
    nz = n > 0
    out[nz] = S[nz] / n[nz, None]
    return out, n


def signed_best_match(A_hat: np.ndarray, B_hat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(q [Fa], idx [Fa]) signed max cosine -- exactly the published q_causal, for comparison."""
    C = np.asarray(A_hat, dtype=np.float64) @ np.asarray(B_hat, dtype=np.float64).T
    idx = np.argmax(C, axis=1)
    return C[np.arange(C.shape[0]), idx], idx


def anti_aligned_fraction(A_hat: np.ndarray, B_hat: np.ndarray) -> float:
    """Fraction of target features whose best ABSOLUTE match is negatively aligned.

    The size of the gap between the published signed metric and this module's m=1 point. Near
    zero means the two are effectively the same number; large means sign is doing real work and
    every comparison must say which convention it used.
    """
    C = np.asarray(A_hat, dtype=np.float64) @ np.asarray(B_hat, dtype=np.float64).T
    best = C[np.arange(C.shape[0]), np.argmax(np.abs(C), axis=1)]
    return float((best < 0).mean())


def _omp_single(A_hat, B_hat, m_max, positive_only, force_first=None):
    """One greedy pass. `force_first` (an int rank) makes step 0 take the rank-th best
    candidate instead of the best, which is how restarts explore past a bad first pick."""
    Fa, D = A_hat.shape
    Q = np.zeros((Fa, m_max, D))
    chosen = np.full((Fa, m_max), -1, dtype=np.int64)
    cos_out = np.zeros((Fa, m_max))
    energy = np.zeros(Fa)
    resid = A_hat.copy()
    rows = np.arange(Fa)

    for step in range(m_max):
        C = resid @ B_hat.T                                   # [Fa, Fb]
        score = C if positive_only else np.abs(C)
        if step:
            score[rows[:, None], chosen[:, :step]] = -np.inf   # never re-pick
        if step == 0 and force_first:
            k = min(force_first, score.shape[1] - 1)
            j = np.argpartition(-score, k, axis=1)[:, :k + 1]
            j = j[rows, np.argsort(-np.take_along_axis(score, j, axis=1), axis=1)[:, k]]
        else:
            j = np.argmax(score, axis=1)
        chosen[:, step] = j

        v = B_hat[j].copy()                                   # [Fa, D]
        for t in range(step):                                 # orthogonalise against the basis
            v -= np.einsum("ij,ij->i", v, Q[:, t])[:, None] * Q[:, t]
        n = np.linalg.norm(v, axis=1)
        ok = n > 1e-10
        q = np.zeros_like(v)
        q[ok] = v[ok] / n[ok, None]
        Q[:, step] = q

        coef = np.einsum("ij,ij->i", A_hat, q)
        energy = np.minimum(energy + coef ** 2, 1.0)
        cos_out[:, step] = np.sqrt(energy)
        coefs = np.einsum("ik,ijk->ij", A_hat, Q[:, :step + 1])
        resid = A_hat - np.einsum("ijk,ij->ik", Q[:, :step + 1], coefs)
    return cos_out, chosen


def omp_curve(A_hat: np.ndarray, B_hat: np.ndarray, m_max: int = 8,
              positive_only: bool = False, n_restarts: int = 4) -> tuple[np.ndarray, np.ndarray]:
    """Greedy orthogonal matching pursuit of each row of A_hat onto rows of B_hat.

    Returns (cos [Fa, m_max], chosen [Fa, m_max]) where cos[:, m-1] is the cosine between the
    target signature and its projection onto the span of the m features selected so far. With
    restarts, cos is the elementwise best over runs (each value still achieved by some real
    subset) while `chosen` is the selection from the run that won at m_max, so intermediate
    columns of cos need not correspond to prefixes of `chosen`.

    Vectorised across all target features at once: each step is one [Fa, Fb] cosine matrix, then
    a modified-Gram-Schmidt update of each target's own orthonormal basis. Projected energy
    accumulates as the sum of squared coefficients on that basis, so no least-squares solve is
    needed. A pick already in the span contributes zero energy and leaves the curve flat, which
    is the correct behaviour; already-chosen indices are masked out so a degenerate step cannot
    stall the sweep by re-picking the same feature.

    GREEDY IS SUBOPTIMAL, AND THE DIRECTION OF THE ERROR MATTERS. Matching pursuit takes the
    locally best direction, which can lead away from the best m-subset: on planted data where
    three features provably span the target, greedy alone recovers only ~0.82-0.98 of it. So the
    curve is a LOWER bound on what B's dictionary can express. That is the opposite bias to the
    non-negativity point in the module docstring, and the two must be read together:

        * a RISING curve is conservative evidence for coalition structure (greedy found it
          despite being suboptimal), but needs the non-negative check before it is believed;
        * a FLAT curve is weaker evidence of absence than it looks, because greedy may simply
          have missed the coalition.

    `n_restarts` narrows the gap: the sweep is re-run forcing the first pick to each of the top-r
    candidates, and the elementwise best curve is kept. Monotonicity survives because a maximum
    of monotone curves is monotone. Cost is linear in r. Set n_restarts=1 for the plain greedy
    result.
    """
    A_hat = np.asarray(A_hat, dtype=np.float64)
    B_hat = np.asarray(B_hat, dtype=np.float64)
    m_max = int(max(1, m_max))
    n_restarts = int(max(1, min(n_restarts, B_hat.shape[0])))

    best_cos, best_chosen = _omp_single(A_hat, B_hat, m_max, positive_only, force_first=None)
    final = best_cos[:, -1].copy()
    for r in range(1, n_restarts):
        cos_r, chosen_r = _omp_single(A_hat, B_hat, m_max, positive_only, force_first=r)
        # Elementwise max PER m, not the whole curve from whichever run wins at m_max. Taking a
        # whole curve would let a run that is better at m=4 but worse at m=1 replace the m=1
        # value, and m=1 has to stay equal to the one-to-one |max cos| for comparability with
        # the published q_causal. Every element remains achievable by a real subset, so the
        # lower-bound reading survives, and a max of monotone curves is still monotone.
        best_cos = np.maximum(best_cos, cos_r)
        take = cos_r[:, -1] > final
        final[take] = cos_r[take, -1]
        best_chosen[take] = chosen_r[take]
    return best_cos, best_chosen


def random_signature_dictionary(n_features: int, d: int, W_U_act: np.ndarray, g: np.ndarray,
                                rng: np.random.Generator, center: bool = True) -> np.ndarray:
    """Unit-norm random decoder rows pushed through a real head -> a null dictionary.

    The same null run_causal_recurrence uses, and for the same reason: permuting bin axes would
    rotate one model out of the shared low-dimensional readout subspace, deflating the floor and
    manufacturing a fake gap. Random decoders destroy feature correspondence while preserving
    the readout geometry every real signature also lives in.

    `n_features` must equal the real dictionary size, or the best-of-F selection effect differs
    between observation and null and the comparison is invalid.
    """
    R = rng.standard_normal((n_features, d))
    R /= np.linalg.norm(R, axis=1, keepdims=True).clip(min=1e-12)
    S = (R * np.asarray(g, dtype=np.float64)[None, :]) @ np.asarray(W_U_act, dtype=np.float64).T
    if center:
        S = S - S.mean(axis=1, keepdims=True)
    return normalize_rows(S)[0]


def chance_corrected_retention(observed: np.ndarray, null: np.ndarray,
                               ceiling: np.ndarray) -> np.ndarray:
    """(observed - null) / (ceiling - null), the §2.4 ret_cc statistic, elementwise.

    The ceiling is the same-model different-seed sweep: how well a dictionary matches a
    re-derivation of ITSELF, which is the most any cross-model comparison could achieve given
    SAE non-identifiability. 1.0 means changing the fine-tuning suite costs nothing beyond
    changing the seed; 0.0 means cross-model matching is no better than random. Reporting the
    raw cosine without this is what makes recurrence numbers uninterpretable.
    """
    observed = np.asarray(observed, dtype=np.float64)
    null = np.asarray(null, dtype=np.float64)
    ceiling = np.asarray(ceiling, dtype=np.float64)
    den = ceiling - null
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(np.abs(den) > 1e-12, (observed - null) / den, np.nan)


def signature_sharpness(S: np.ndarray) -> np.ndarray:
    """[F]: participation ratio over the 256 bins of each signature's squared entries.

    Low = the feature pushes a few bins hard (a sharp, specialist-like signature); high = it
    spreads over many. Cosine matching plausibly favours sharp signatures, which is open thread
    #4 in results.md (explanation #4 for the Path B result). Carried here so every decile
    comparison can be rank-residualised on it and the geometry confound ruled in or out.
    """
    S = np.asarray(S, dtype=np.float64) ** 2
    s1 = S.sum(axis=1)
    s2 = (S * S).sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(s2 > 0, s1 * s1 / s2, np.nan)
