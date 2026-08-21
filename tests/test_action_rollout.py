"""Pins the closed-loop rollout store: shapes, shard bookkeeping, and success labelling.

The rollout loop itself needs a GPU and a simulator, so what is testable here is the part
that can corrupt the dataset SILENTLY -- and that part is worth pinning precisely, because
the failure mode is not a crash. Mislabelled or dropped rows produce a confident, wrong
AUROC downstream with nothing to flag it, after the GPU hours are already spent.

Two tests carry the weight:

  * `test_commit_episode_stamps_failure_on_every_row` -- failures ARE the signal in this
    experiment. A label inverted or defaulted to 1 would make the whole collection useless
    and look fine.
  * `test_writer_rejects_pooled_shaped_activations` -- passing a [n_layers, d] pooled vector
    where a [7, d] action-position residual belongs is exactly the mix-up this module exists
    to prevent, and mrvla.store would have accepted it.

Run directly:
    python tests/test_action_rollout.py
"""

from __future__ import annotations

import glob
import json
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mrvla.action_rollout import RolloutShardWriter, commit_episode  # noqa: E402

D, A = 16, 7


def _writer(tmp, **kw):
    return RolloutShardWriter(tmp, hidden_dim=D, action_dim=A, **kw)


def _manifest(tmp):
    """The manifest is namespaced by prefix, so tests resolve it rather than hardcode it."""
    hits = glob.glob(os.path.join(tmp, "manifest_*.json"))
    assert len(hits) == 1, hits
    return hits[0]


def _row(seed=0):
    r = np.random.default_rng(seed)
    return r.normal(size=(A, D)).astype(np.float32), r.normal(size=A).astype(np.float32)


# ---------------------------------------------------------------------------
# success labelling -- the silent-corruption surface
# ---------------------------------------------------------------------------
def test_commit_episode_stamps_success_on_every_row():
    with tempfile.TemporaryDirectory() as tmp:
        w = _writer(tmp)
        buf = [(*_row(i), i) for i in range(5)]
        assert commit_episode(w, buf, True, episode=3, task_id=2) == 5
        w.close()
        d = np.load(glob.glob(os.path.join(tmp, "shard_*.npz"))[0])
        assert (d["success"] == 1).all()
        assert (d["episode"] == 3).all() and (d["task_id"] == 2).all()
        assert list(d["timestep"]) == [0, 1, 2, 3, 4]


def test_commit_episode_stamps_failure_on_every_row():
    """Failures are the signal. A defaulted or inverted label ruins the collection quietly."""
    with tempfile.TemporaryDirectory() as tmp:
        w = _writer(tmp)
        commit_episode(w, [(*_row(i), i) for i in range(4)], False, episode=0, task_id=0)
        w.close()
        d = np.load(glob.glob(os.path.join(tmp, "shard_*.npz"))[0])
        assert (d["success"] == 0).all(), d["success"]


def test_failures_are_never_dropped():
    """There is no store_only_success here, by design. Both outcomes must survive."""
    with tempfile.TemporaryDirectory() as tmp:
        w = _writer(tmp)
        for ep, ok in enumerate([True, False, False, True]):
            commit_episode(w, [(*_row(ep), 0)], ok, episode=ep, task_id=0)
        w.close()
        d = np.load(glob.glob(os.path.join(tmp, "shard_*.npz"))[0])
        assert sorted(d["success"].tolist()) == [0, 0, 1, 1]
        assert json.load(open(_manifest(tmp)))["total_timesteps"] == 4


def test_commit_episode_handles_an_empty_buffer():
    """An episode that ended during the settle window writes nothing and does not raise."""
    with tempfile.TemporaryDirectory() as tmp:
        w = _writer(tmp)
        assert commit_episode(w, [], False, episode=0, task_id=0) == 0
        w.close()
        assert json.load(open(_manifest(tmp)))["total_timesteps"] == 0


# ---------------------------------------------------------------------------
# shape validation
# ---------------------------------------------------------------------------
def test_writer_rejects_pooled_shaped_activations():
    """A [n_layers, d] pooled vector must not be accepted where a [7, d] residual belongs.

    This is the exact mix-up the module exists to prevent -- mrvla.store's validator would
    have passed it, since it only checks against ITS OWN configured layer count.
    """
    with tempfile.TemporaryDirectory() as tmp:
        w = _writer(tmp)
        try:
            w.add(np.zeros((5, D)), np.zeros(A), 0, 0, 0, 1)
        except ValueError:
            pass
        else:
            raise AssertionError("accepted a [5, d] activation as a [7, d] residual")


def test_writer_rejects_wrong_action_dim():
    with tempfile.TemporaryDirectory() as tmp:
        w = _writer(tmp)
        try:
            w.add(np.zeros((A, D)), np.zeros(3), 0, 0, 0, 1)
        except ValueError:
            pass
        else:
            raise AssertionError("accepted a 3-entry action")


# ---------------------------------------------------------------------------
# shard bookkeeping
# ---------------------------------------------------------------------------
def test_shards_roll_over_and_nothing_is_lost():
    with tempfile.TemporaryDirectory() as tmp:
        w = _writer(tmp, shard_size=10)
        for i in range(25):
            w.add(*_row(i), episode=i // 5, timestep=i % 5, task_id=0, success=i % 2)
        w.close()
        shards = sorted(glob.glob(os.path.join(tmp, "shard_*.npz")))
        assert len(shards) == 3, shards
        total = sum(np.load(s)["residual"].shape[0] for s in shards)
        assert total == 25
        man = json.load(open(_manifest(tmp)))
        assert man["total_timesteps"] == 25 and man["n_shards"] == 3


def test_stored_residual_is_fp16_and_action_is_fp32():
    """Storage dtypes are a size decision: fp16 residuals are ~57 KB/timestep, fp32 double it."""
    with tempfile.TemporaryDirectory() as tmp:
        w = _writer(tmp)
        w.add(*_row(0), episode=0, timestep=0, task_id=0, success=1)
        w.close()
        d = np.load(glob.glob(os.path.join(tmp, "shard_*.npz"))[0])
        assert d["residual"].dtype == np.float16, d["residual"].dtype
        assert d["action"].dtype == np.float32, d["action"].dtype


def test_manifest_records_tasks_and_extra():
    with tempfile.TemporaryDirectory() as tmp:
        w = _writer(tmp)
        w.register_task(0, "put the bowl on the plate")
        w.add(*_row(0), episode=0, timestep=0, task_id=0, success=0)
        w.close(extra={"model": "openvla/x", "success_rate": 0.0})
        man = json.load(open(_manifest(tmp)))
        assert man["tasks"]["0"] == "put the bowl on the plate"
        assert man["model"] == "openvla/x"
        assert man["kind"] == "closed_loop_action_positions"


def test_field_names_match_the_a1_contract():
    """Loaders written against A1 shards must work here unchanged."""
    with tempfile.TemporaryDirectory() as tmp:
        w = _writer(tmp)
        w.add(*_row(0), episode=0, timestep=0, task_id=0, success=1)
        w.close()
        keys = set(np.load(glob.glob(os.path.join(tmp, "shard_*.npz"))[0]).files)
        assert {"residual", "episode", "timestep", "task_id"} <= keys, keys
        assert {"action", "success"} <= keys, keys


# ---------------------------------------------------------------------------
# multi-worker output -- the collision that destroyed a 500-episode run
# ---------------------------------------------------------------------------
def test_workers_sharing_a_directory_do_not_overwrite_each_other():
    """Every writer's shard counter starts at 0, so a shared prefix means the last worker
    wins and the loss is silent: the analysis globs whatever survived and reports confident
    numbers on a fraction of the data. This is not hypothetical -- it cost a full run."""
    with tempfile.TemporaryDirectory() as tmp:
        for w in range(4):
            wr = _writer(tmp, shard_size=2, prefix=f"shard_w{w}")
            for i in range(4):
                wr.add(*_row(w * 10 + i), episode=w, timestep=i, task_id=0, success=w % 2)
            wr.close()
        shards = sorted(glob.glob(os.path.join(tmp, "shard_*.npz")))
        assert len(shards) == 8, shards                      # 4 workers x 2 shards each
        total = sum(np.load(s)["residual"].shape[0] for s in shards)
        assert total == 16, total                            # nothing lost
        assert sorted(np.unique(np.concatenate(
            [np.load(s)["episode"] for s in shards]))) == [0, 1, 2, 3]


def test_a_shared_prefix_raises_instead_of_clobbering():
    """If two writers ever do collide again, it must be an error, not lost data."""
    with tempfile.TemporaryDirectory() as tmp:
        a = _writer(tmp, shard_size=1)
        a.add(*_row(0), episode=0, timestep=0, task_id=0, success=1)
        b = _writer(tmp, shard_size=1)
        try:
            b.add(*_row(1), episode=1, timestep=0, task_id=0, success=0)
        except FileExistsError:
            pass
        else:
            raise AssertionError("second writer silently overwrote the first")


def test_manifests_are_per_worker_too():
    with tempfile.TemporaryDirectory() as tmp:
        for w in range(2):
            wr = _writer(tmp, prefix=f"shard_w{w}")
            wr.add(*_row(w), episode=w, timestep=0, task_id=0, success=1)
            wr.close()
        assert len(glob.glob(os.path.join(tmp, "manifest_*.json"))) == 2


# ---------------------------------------------------------------------------
# episode sharding -- the loop needs a simulator, but the PARTITION is pure arithmetic
# ---------------------------------------------------------------------------
def _shard(n_tasks, trials, n_workers):
    """Mirror of the partition and episode-id arithmetic in rollout_action_positions."""
    n_by_task = [trials] * n_tasks
    ep_base = np.concatenate([[0], np.cumsum(n_by_task)[:-1]]).astype(int)
    out = {}
    for w in range(n_workers):
        got = []
        for task_id in range(n_tasks):
            for t in range(n_by_task[task_id]):
                if (int(ep_base[task_id]) + t) % n_workers == w:
                    got.append((task_id, t, int(ep_base[task_id]) + t))
        out[w] = got
    return out


def test_every_init_state_is_claimed_exactly_once():
    """An init state run twice double-counts an episode; one skipped is silently lost."""
    sh = _shard(10, 50, 4)
    claimed = [(t, tr) for w in sh for (t, tr, _e) in sh[w]]
    assert len(claimed) == 500, len(claimed)
    assert len(set(claimed)) == 500, "an init state was claimed by two workers"


def test_episode_ids_are_unique_and_worker_count_independent():
    """The analysis groups by episode id, so a collision would merge two rollouts into one."""
    for n_workers in (1, 2, 4, 8):
        sh = _shard(10, 50, n_workers)
        ids = [e for w in sh for (_t, _tr, e) in sh[w]]
        assert len(set(ids)) == 500, (n_workers, len(set(ids)))
        assert sorted(ids) == list(range(500)), n_workers


def test_episode_sharding_is_exactly_balanced():
    """The point of the switch. Sharding on the per-task trial instead would give
    130/130/120/120, because 50 trials does not divide by 4 and the remainder lands on the
    same two workers in every task."""
    counts = sorted(len(v) for v in _shard(10, 50, 4).values())
    assert counts == [125, 125, 125, 125], counts


def test_sharding_stays_balanced_for_awkward_worker_counts():
    for nw in (3, 6, 7):
        counts = sorted(len(v) for v in _shard(10, 50, nw).values())
        assert max(counts) - min(counts) <= 1, (nw, counts)


def test_each_worker_sees_every_task():
    """Balance in COUNT is not enough -- per-task difficulty differs, so every worker must
    draw from every task or the hard tasks concentrate on one GPU again."""
    sh = _shard(10, 50, 4)
    for w, got in sh.items():
        assert len({t for (t, _tr, _e) in got}) == 10, (w, got[:3])


def test_a_worker_is_empty_only_when_workers_outnumber_episodes():
    """Global-index sharding keeps every worker busy until there are literally fewer
    episodes than workers -- per-task sharding starved workers much sooner, whenever
    n_workers exceeded the per-task trial count."""
    sh = _shard(2, 3, 5)                      # 6 episodes, 5 workers: nobody idle
    assert sum(len(v) for v in sh.values()) == 6
    assert all(len(v) >= 1 for v in sh.values()), {w: len(v) for w, v in sh.items()}
    sh2 = _shard(2, 3, 8)                     # 6 episodes, 8 workers: two idle
    assert sum(len(v) for v in sh2.values()) == 6
    assert sum(len(v) == 0 for v in sh2.values()) == 2


if __name__ == "__main__":
    for name, fn in sorted(list(globals().items())):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("all action_rollout tests passed")
