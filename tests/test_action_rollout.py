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
        assert json.load(open(os.path.join(tmp, "manifest.json")))["total_timesteps"] == 4


def test_commit_episode_handles_an_empty_buffer():
    """An episode that ended during the settle window writes nothing and does not raise."""
    with tempfile.TemporaryDirectory() as tmp:
        w = _writer(tmp)
        assert commit_episode(w, [], False, episode=0, task_id=0) == 0
        w.close()
        assert json.load(open(os.path.join(tmp, "manifest.json")))["total_timesteps"] == 0


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
        man = json.load(open(os.path.join(tmp, "manifest.json")))
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
        man = json.load(open(os.path.join(tmp, "manifest.json")))
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


if __name__ == "__main__":
    for name, fn in sorted(list(globals().items())):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("all action_rollout tests passed")
