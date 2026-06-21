"""Closed-loop LIBERO rollout collection for OpenVLA activations.

Mirrors the official OpenVLA LIBERO evaluation loop (openvla/openvla
``experiments/robot/libero/run_libero_eval.py``) but, instead of only scoring success,
captures pooled residual-stream activations at every control step.

Requires the LIBERO benchmark to be installed:
    git clone https://github.com/Lifelong-Robot-Learning/LIBERO
    cd LIBERO && pip install -e .
"""

from __future__ import annotations

import os

import numpy as np
from PIL import Image
from tqdm import tqdm

from .model_utils import predict_and_capture

# Default per-suite episode length caps used by OpenVLA's LIBERO eval.
_SUITE_MAX_STEPS = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
    "libero_90": 400,
}
_NUM_STEPS_WAIT = 10  # let objects settle before the policy acts
_DUMMY_ACTION = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]


def _normalize_gripper_action(action: np.ndarray, binarize: bool = True) -> np.ndarray:
    """Map gripper channel from [0,1] to [-1,1] (LIBERO/robosuite convention)."""
    action = action.copy()
    action[..., -1] = 2.0 * (action[..., -1] - 0.0) / (1.0 - 0.0) - 1.0
    if binarize:
        action[..., -1] = np.sign(action[..., -1])
    return action


def _invert_gripper_action(action: np.ndarray) -> np.ndarray:
    """Flip gripper sign (OpenVLA outputs the opposite of LIBERO's convention)."""
    action = action.copy()
    action[..., -1] = action[..., -1] * -1.0
    return action


def _get_libero_image(obs) -> Image.Image:
    """Extract and orient the agentview image as OpenVLA expects (224 handled by processor)."""
    img = obs["agentview_image"]
    img = img[::-1, ::-1]  # LIBERO renders upside-down relative to training data
    return Image.fromarray(np.ascontiguousarray(img))


def collect_libero(
    model,
    processor,
    collector,
    writer,
    task_suite_name: str,
    unnorm_key: str,
    device: str,
    trials_per_task: int = 20,
    max_steps: int | None = None,
    num_steps_wait: int = _NUM_STEPS_WAIT,
    seed: int = 0,
    camera_res: int = 256,
    max_tasks: int | None = None,
    store_only_success: bool = False,
) -> int:
    """Run closed-loop rollouts across a LIBERO task suite and store activations.

    Activations for an episode are buffered in memory and committed only after the
    episode ends, so each stored timestep carries the true ``success`` label.
    """
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    if max_steps is None:
        max_steps = _SUITE_MAX_STEPS.get(task_suite_name, 300)

    benchmark_dict = benchmark.get_benchmark_dict()
    if task_suite_name not in benchmark_dict:
        raise ValueError(
            f"Unknown task suite {task_suite_name!r}. Available: {list(benchmark_dict)}"
        )
    task_suite = benchmark_dict[task_suite_name]()
    num_tasks = task_suite.n_tasks
    if max_tasks is not None:
        num_tasks = min(num_tasks, max_tasks)

    global_episode = 0
    total_stored = 0

    for task_id in range(num_tasks):
        task = task_suite.get_task(task_id)
        task_description = task.language
        writer.register_task(task_id, task_description)

        bddl_file = os.path.join(
            get_libero_path("bddl_files"), task.problem_folder, task.bddl_file
        )
        env = OffScreenRenderEnv(
            bddl_file_name=bddl_file,
            camera_heights=camera_res,
            camera_widths=camera_res,
        )
        env.seed(seed)
        init_states = task_suite.get_task_init_states(task_id)

        n_trials = min(trials_per_task, len(init_states))
        for trial in tqdm(range(n_trials), desc=f"task {task_id}: {task_description[:40]}"):
            env.reset()
            obs = env.set_init_state(init_states[trial])

            episode_buffer: list[tuple[np.ndarray, int]] = []  # (acts, timestep)
            success = False
            step = 0
            while step < max_steps + num_steps_wait:
                if step < num_steps_wait:
                    obs, _r, done, _info = env.step(_DUMMY_ACTION)
                    step += 1
                    continue

                image = _get_libero_image(obs)
                action, acts = predict_and_capture(
                    model, processor, collector, image, task_description, device, unnorm_key
                )
                episode_buffer.append((acts, step - num_steps_wait))

                action = _normalize_gripper_action(action, binarize=True)
                action = _invert_gripper_action(action)
                obs, _r, done, _info = env.step(action.tolist())
                step += 1
                if done:
                    success = True
                    break

            if store_only_success and not success:
                global_episode += 1
                continue

            success_flag = 1 if success else 0
            for acts, timestep in episode_buffer:
                writer.add(
                    acts,
                    episode=global_episode,
                    timestep=timestep,
                    task_id=task_id,
                    success=success_flag,
                )
                total_stored += 1
            global_episode += 1

        env.close()

    return total_stored
