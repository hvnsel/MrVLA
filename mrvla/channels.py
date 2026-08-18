"""Per-action-channel decomposition of causal influence (B1).

Path A aggregates |phi| over tasks to get breadth. It also aggregates over the SEVEN action
slots and throws that axis away (`run_attribution.py` uses `r % 7` only to look up the emitted
token). That discarded axis is a second, orthogonal breadth measure, free from data already
collected: how many of the seven action dimensions does a feature actually drive?

WHY THE SLOT IS THE CHANNEL. OpenVLA emits one token per action dimension and reuses the SAME
256 action tokens at every decode position, so channel identity is POSITIONAL. Bin 200 at slot 0
is a translation in x; at slot 6 it is a gripper command. Two consequences run through this
module: a feature's 256-bin causal signature is semantically ambiguous until conditioned on the
slot, and "which channel does this feature drive" is answered by where its |phi| mass sits
across the seven positions.

THE NORMALISATION TRAP. Raw |phi| is NOT comparable across slots. phi carries
u_contrast = u_t - mean_s u_s, whose norm depends on where the emitted bin sits in the ordered
range. The gripper is near-binary and emits extreme bins, which have the largest ||u_contrast||,
so EVERY feature looks stronger at the gripper slot for a purely geometric reason. `share`
statistics -- a feature's fraction of the total |phi| at that decision and slot -- are
comparable; absolute ones are not. Both are computed, because a result that appears only in the
absolute numbers is the confound rather than the finding.

THE DEGENERACY TRAP. The gripper token is constant for most of an episode, so a feature can
score high gripper share simply by dominating a low-entropy slot. `transition_mask` isolates the
decisions where that channel's command actually changes, and every statistic should be reported
on all decisions AND on transitions only.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "slot_index", "accumulate_slot_task", "decision_shares", "channel_mass",
    "channel_profile", "channel_participation_ratio", "transition_mask",
    "DEFAULT_CHANNEL_NAMES",
]

# OpenVLA's action ordering. Not discoverable from the checkpoint, so it is an assumption --
# stated here, overridable, and used only for labelling, never for arithmetic.
DEFAULT_CHANNEL_NAMES = ("dx", "dy", "dz", "droll", "dpitch", "dyaw", "gripper")


def slot_index(n_decisions: int, n_slots: int = 7) -> np.ndarray:
    """Slot id for each row of a flattened [n, n_slots, d] residual block.

    The flattening in `run_attribution` is `res.reshape(n * 7, d)`, so row r belongs to decision
    r // 7 and slot r % 7. Getting this backwards silently transposes every channel result.
    """
    return np.tile(np.arange(n_slots, dtype=np.int64), n_decisions)


def accumulate_slot_task(dest_sum: np.ndarray, dest_count: np.ndarray, values: np.ndarray,
                         slots: np.ndarray, tasks: np.ndarray) -> None:
    """Accumulate `values` [n, F] into dest_sum [S, G, F] and dest_count [S, G], in place.

    `tasks` is the per-ROW task id (already expanded from per-decision), `slots` the per-row slot.
    Streaming means never materialising phi for the whole run, so the caller adds shard by shard
    and divides at the end.
    """
    values = np.asarray(values, dtype=np.float64)
    slots = np.asarray(slots, dtype=np.int64)
    tasks = np.asarray(tasks, dtype=np.int64)
    S, G = dest_sum.shape[0], dest_sum.shape[1]
    flat = slots * G + tasks                      # one bucket id per row
    for b in np.unique(flat):
        m = flat == b
        s, t = divmod(int(b), G)
        if 0 <= s < S and 0 <= t < G:
            dest_sum[s, t] += values[m].sum(axis=0)
            dest_count[s, t] += int(m.sum())


def decision_shares(phi_abs: np.ndarray, eps: float = 1e-30) -> np.ndarray:
    """Row-normalise |phi| so each decision contributes 1 unit of causal mass.

    This is the cross-slot-comparable form: it asks "what fraction of THIS decision did feature
    j carry", removing the per-slot ||u_contrast|| scale that otherwise makes the gripper slot
    look important for every feature at once.
    """
    v = np.asarray(phi_abs, dtype=np.float64)
    return v / np.maximum(v.sum(axis=1, keepdims=True), eps)


def channel_mass(C_slot: np.ndarray) -> np.ndarray:
    """[S, F]: total causal mass per (slot, feature), summing over tasks."""
    return np.asarray(C_slot, dtype=np.float64).sum(axis=1)


def channel_profile(C_slot: np.ndarray, eps: float = 1e-30) -> np.ndarray:
    """[S, F] normalised per feature: how feature j splits its influence across the channels."""
    M = channel_mass(C_slot)
    return M / np.maximum(M.sum(axis=0, keepdims=True), eps)


def channel_participation_ratio(C_slot: np.ndarray) -> np.ndarray:
    """[F]: effective number of ACTION DIMENSIONS a feature drives, 1 (one channel) to S (all).

    Deliberately the same participation ratio Path A applies to tasks, so the two breadth axes
    are on one scale and can be read as a plane. Scale-free, so it measures spread and not
    strength -- a feature that dominates one channel and a feature that barely touches it both
    score 1 if that is all they touch.
    """
    M = channel_mass(C_slot)
    s1 = M.sum(axis=0)
    s2 = (M * M).sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(s2 > 0, s1 * s1 / s2, np.nan)


def transition_mask(bins: np.ndarray, episode: np.ndarray, timestep: np.ndarray) -> np.ndarray:
    """[n] bool: did this channel's emitted bin change from the previous timestep of the SAME
    episode?

    The control for slot degeneracy. A near-constant channel (the gripper) hands any feature a
    high share for free; restricting to the decisions where the command actually changes asks
    whether the feature is driving the EVENT rather than riding the plateau. The first decision
    of each episode has no predecessor and is False.

    Rows may arrive in any order: they are sorted by (episode, timestep) internally, so this does
    not depend on shard layout.
    """
    bins = np.asarray(bins).ravel()
    episode = np.asarray(episode).ravel()
    timestep = np.asarray(timestep).ravel()
    order = np.lexsort((timestep, episode))
    b, e = bins[order], episode[order]
    changed = np.zeros(b.size, dtype=bool)
    if b.size > 1:
        same_ep = e[1:] == e[:-1]
        changed[1:] = same_ep & (b[1:] != b[:-1])
    out = np.zeros(b.size, dtype=bool)
    out[order] = changed
    return out
