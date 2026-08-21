"""Closed-loop LIBERO rollouts that store ACTION-POSITION residuals AND success labels.

This is the asset the whole success-rate line was blocked on. Two datasets existed and
neither could answer the question:

  * A1 (`collect_action_activations.py`) stores the 7 action-position residuals, but replays
    DEMONSTRATIONS -- so `success` is the constant 1 and there is nothing to correlate with.
  * `mrvla/libero_collect.py` drives the policy closed-loop and carries real success labels,
    but stores the MEAN-POOLED PREFILL activation, which is not the vector that decodes the
    action (EXPERIMENT_PLAN.md §2.3).

This module is the intersection: the closed-loop loop from `libero_collect` with the
action-position capture from A1. Every stored timestep carries the un-pooled layer-31
residual at each of the 7 decode positions, the continuous action actually sent to the
simulator, and the true episode outcome.

WHY THE ACTION IS STORED AND THE TOKENS ARE NOT. `model.predict_action` returns the
unnormalised action but not the emitted token ids, and getting both would cost a second
forward pass. The emitted bin is recoverable from the residual itself -- argmax of
`readout.unnormalized_logits` -- which is the same recomputed-argmax baseline P6's flip
analysis already uses, faithful to the stored tokens at 0.9919 on A1. Storing the executed
action (7 float32, 28 bytes against 57 KB of residual) is what makes that recovery
CHECKABLE rather than assumed: the analysis can detokenise its recomputed bin and compare
against what the simulator actually received.

SUCCESS LABELLING. Timesteps are buffered for the whole episode and committed only once the
outcome is known, so every row carries the true label. `store_only_success` does not exist
here on purpose -- in this experiment the failures ARE the signal, and a flag with that name
sitting in the signature is an invitation to destroy the dataset.

THE DISTRIBUTION CAVEAT, WHICH THE ANALYSIS MUST CHECK. The SAE was trained on
demo-replay action-position residuals. Closed-loop rollouts visit states the policy
generated itself, so encoding them is an extrapolation. Sufficiency was 0.936 on demo
replay; if it collapses here, every downstream signal is read off a decomposition that no
longer holds and nothing else in the pipeline would catch it. Recompute it before trusting
any correlation.
"""

from __future__ import annotations

import json
import os

import numpy as np

__all__ = ["RolloutShardWriter", "commit_episode", "rollout_action_positions"]


# ---------------------------------------------------------------------------
# storage
# ---------------------------------------------------------------------------
class RolloutShardWriter:
    """Shard writer for closed-loop action-position rollouts.

    Field names deliberately match A1's shards (`residual`, `episode`, `timestep`,
    `task_id`) so any loader written against those works here, plus the two fields A1
    cannot have: `action` (what was executed) and `success` (how the episode ended).

    `mrvla.store.ShardedActivationWriter` is not reused because it validates `acts` against
    `(n_layers, hidden_dim)` and carries a layer-indices manifest. Passing 7 decode slots
    through it as if they were 7 layers would validate fine and label the data wrongly.
    """

    def __init__(self, out_dir: str, hidden_dim: int, action_dim: int = 7,
                 shard_size: int = 4000, dtype: str = "float16"):
        self.out_dir = out_dir
        self.hidden_dim = int(hidden_dim)
        self.action_dim = int(action_dim)
        self.shard_size = int(shard_size)
        self.dtype = dtype
        os.makedirs(out_dir, exist_ok=True)
        self._res: list[np.ndarray] = []
        self._act: list[np.ndarray] = []
        self._ep: list[int] = []
        self._ts: list[int] = []
        self._task: list[int] = []
        self._succ: list[int] = []
        self._shard = 0
        self.total = 0
        self.tasks: dict[int, str] = {}

    def register_task(self, task_id: int, name: str) -> None:
        self.tasks[int(task_id)] = name

    def add(self, residual: np.ndarray, action: np.ndarray, episode: int,
            timestep: int, task_id: int, success: int) -> None:
        residual = np.asarray(residual)
        if residual.shape != (self.action_dim, self.hidden_dim):
            raise ValueError(f"residual must be {(self.action_dim, self.hidden_dim)}, "
                             f"got {residual.shape}")
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.size != self.action_dim:
            raise ValueError(f"action must have {self.action_dim} entries, got {action.size}")
        self._res.append(residual.astype(self.dtype, copy=False))
        self._act.append(action)
        self._ep.append(int(episode))
        self._ts.append(int(timestep))
        self._task.append(int(task_id))
        self._succ.append(int(success))
        if len(self._res) >= self.shard_size:
            self.flush()

    def flush(self) -> None:
        if not self._res:
            return
        path = os.path.join(self.out_dir, f"shard_{self._shard:05d}.npz")
        np.savez_compressed(
            path,
            residual=np.stack(self._res, axis=0),
            action=np.stack(self._act, axis=0),
            episode=np.array(self._ep, np.int32),
            timestep=np.array(self._ts, np.int32),
            task_id=np.array(self._task, np.int32),
            success=np.array(self._succ, np.int32),
        )
        self.total += len(self._res)
        self._shard += 1
        self._res.clear(); self._act.clear(); self._ep.clear()
        self._ts.clear(); self._task.clear(); self._succ.clear()

    def close(self, extra: dict | None = None) -> None:
        self.flush()
        manifest = {
            "kind": "closed_loop_action_positions",
            "hidden_dim": self.hidden_dim,
            "action_dim": self.action_dim,
            "dtype": self.dtype,
            "n_shards": self._shard,
            "total_timesteps": self.total,
            "tasks": {str(k): v for k, v in sorted(self.tasks.items())},
        }
        manifest.update(extra or {})
        with open(os.path.join(self.out_dir, "manifest.json"), "w") as f:
            json.dump(manifest, f, indent=2)


# ---------------------------------------------------------------------------
# the one piece of episode logic worth isolating
# ---------------------------------------------------------------------------
def commit_episode(writer, buffer, success: bool, episode: int, task_id: int) -> int:
    """Write a finished episode's buffered timesteps, stamping the outcome on each.

    Split out from the rollout loop because it is the only place a silent data-corrupting
    bug can live: mislabelled rows would produce a confident, wrong AUROC downstream with
    nothing to flag it. `buffer` is a list of (residual [7, d], action [7], timestep).
    Returns the number of rows written.
    """
    flag = 1 if success else 0
    for residual, action, timestep in buffer:
        writer.add(residual, action, episode=episode, timestep=timestep,
                   task_id=task_id, success=flag)
    return len(buffer)


# ---------------------------------------------------------------------------
# the rollout loop
# ---------------------------------------------------------------------------
def rollout_action_positions(model, processor, collector, writer, task_suite_name: str,
                             unnorm_key: str, device: str, trials_per_task: int = 50,
                             max_steps: int | None = None, seed: int = 0,
                             camera_res: int = 256, max_tasks: int | None = None,
                             center_crop: bool = True, action_dim: int = 7,
                             worker_id: int = 0, n_workers: int = 1) -> dict:
    """Drive the policy closed-loop, capturing action-position residuals per decision.

    Mirrors `mrvla.libero_collect.collect_libero`'s structure -- same gripper conventions,
    same warm-up wait, same init-state ordering -- so success rates are comparable with the
    ablation runs. `collector` must be an `ActionPositionCollector` on the target layer.

    Tasks are sharded round-robin across `n_workers` so one process per GPU can be launched
    the way `run_ablation.py` already does. Sharding by TASK rather than by episode keeps
    each worker's init-state ordering identical to a single-process run.
    """
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    from mrvla.libero_collect import (
        _DUMMY_ACTION, _NUM_STEPS_WAIT, _SUITE_MAX_STEPS, _get_libero_image,
        _invert_gripper_action, _normalize_gripper_action,
    )
    from mrvla.model_utils import build_inputs
    from mrvla.torch_compat import allow_numpy_pickles
    allow_numpy_pickles()

    if max_steps is None:
        max_steps = _SUITE_MAX_STEPS.get(task_suite_name, 300)

    suite = benchmark.get_benchmark_dict()[task_suite_name]()
    n_tasks = suite.n_tasks if max_tasks is None else min(suite.n_tasks, max_tasks)

    per_task: dict[int, dict] = {}
    global_episode = 0

    for task_id in range(n_tasks):
        task = suite.get_task(task_id)
        init_states = suite.get_task_init_states(task_id)
        n_trials = min(trials_per_task, len(init_states))
        if task_id % n_workers != worker_id:
            global_episode += n_trials      # keep episode ids identical across workers
            continue

        writer.register_task(task_id, task.language)
        env = OffScreenRenderEnv(
            bddl_file_name=os.path.join(get_libero_path("bddl_files"),
                                        task.problem_folder, task.bddl_file),
            camera_heights=camera_res, camera_widths=camera_res,
        )
        env.seed(seed)
        n_succ = 0

        for trial in range(n_trials):
            env.reset()
            obs = env.set_init_state(init_states[trial])
            buffer, success, step = [], False, 0

            while step < max_steps + _NUM_STEPS_WAIT:
                if step < _NUM_STEPS_WAIT:
                    obs, _r, _done, _info = env.step(_DUMMY_ACTION)
                    step += 1
                    continue

                image = _get_libero_image(obs, center_crop=center_crop)
                inputs = build_inputs(processor, image, task.language, device)
                # build_inputs already omits attention_mask on purpose: predict_action appends
                # token 29871 to input_ids without extending a mask we supply, desyncing
                # lengths against the 256 image patches. The pop is belt-and-braces in case
                # that ever changes, matching model_utils.predict_and_capture.
                inputs.pop("attention_mask", None)
                collector.reset()
                action = np.asarray(
                    model.predict_action(**inputs, unnorm_key=unnorm_key, do_sample=False),
                    dtype=np.float32)
                residual = collector.stack(expected=action_dim)      # [7, d]
                buffer.append((np.asarray(residual, dtype=np.float16), action,
                               step - _NUM_STEPS_WAIT))

                env_action = _invert_gripper_action(
                    _normalize_gripper_action(action, binarize=True))
                obs, _r, done, _info = env.step(env_action.tolist())
                step += 1
                if done:
                    success = True
                    break

            commit_episode(writer, buffer, success, global_episode, task_id)
            n_succ += int(success)
            global_episode += 1

        per_task[task_id] = {"task": task.language, "episodes": n_trials,
                             "successes": n_succ, "success_rate": n_succ / max(1, n_trials)}
        print(f"[rollout] task {task_id} ({task.language[:40]}): "
              f"{n_succ}/{n_trials} success", flush=True)
        env.close()

    return per_task
