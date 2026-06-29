"""Replay LIBERO demonstration HDF5s through OpenVLA to collect activations.

This matches the paper's pipeline (arXiv 2603.19183 §A.1.3): activations are
collected over the **fine-tuning demonstrations**, not closed-loop rollouts.
Each stored episode corresponds to one demo trajectory, so the downstream
generality metrics (episode coverage, mean onset count, relative run length)
are well-defined over the fine-tuning dataset.

Requires the LIBERO benchmark and its bundled demo HDF5s:
    git clone https://github.com/Lifelong-Robot-Learning/LIBERO
    cd LIBERO && pip install -e .
    # then download the per-suite demo datasets per LIBERO's README.
"""

from __future__ import annotations

import glob
import os

import h5py
import numpy as np
from PIL import Image
from tqdm import tqdm

from .libero_collect import _center_crop, _resize_libero_image
from .model_utils import predict_and_capture

# HDF5 obs keys that have been used for the LIBERO agentview camera across
# different releases. Tried in order.
_AGENTVIEW_KEYS = ("agentview_rgb", "agentview_image")


def _find_demo_file(datasets_root: str, suite: str, task) -> str:
    """Locate the HDF5 demo file for a LIBERO task within a suite directory."""
    suite_dir = os.path.join(datasets_root, suite)
    bddl_stem = os.path.splitext(task.bddl_file)[0]
    candidate = os.path.join(suite_dir, f"{bddl_stem}_demo.hdf5")
    if os.path.exists(candidate):
        return candidate
    matches = glob.glob(os.path.join(suite_dir, f"*{bddl_stem}*.hdf5"))
    if matches:
        return matches[0]
    raise FileNotFoundError(
        f"No demo HDF5 for task {task.name!r} in {suite_dir}. "
        f"Expected {candidate}. Did you download the LIBERO demo datasets?"
    )


def _resolve_image_key(obs_group, override: str | None) -> str:
    if override is not None:
        if override not in obs_group:
            raise KeyError(
                f"--image-key {override!r} not in obs group; "
                f"available: {list(obs_group.keys())}"
            )
        return override
    for k in _AGENTVIEW_KEYS:
        if k in obs_group:
            return k
    raise KeyError(
        f"None of {_AGENTVIEW_KEYS} found in obs group; "
        f"available: {list(obs_group.keys())}"
    )


def _demo_image(frame: np.ndarray, center_crop: bool = True) -> Image.Image:
    """Apply the OpenVLA LIBERO image transform to a single recorded frame.

    Same pipeline as the closed-loop path: 180-degree flip, LANCZOS resize to
    224, then a central 90%-area crop for the libero-finetuned checkpoints.
    """
    img = frame[::-1, ::-1]
    pil = Image.fromarray(np.ascontiguousarray(img))
    pil = _resize_libero_image(pil, size=224)
    if center_crop:
        pil = _center_crop(pil, crop_scale=0.9)
    return pil


def collect_libero_demos(
    model,
    processor,
    collector,
    writer,
    task_suite_name: str,
    unnorm_key: str,
    device: str,
    max_tasks: int | None = None,
    max_demos_per_task: int | None = None,
    max_steps_per_demo: int | None = None,
    image_key: str | None = None,
    center_crop: bool = True,
) -> int:
    """Iterate every (task, demo, timestep) triple and capture pooled activations.

    Every stored episode is a successful expert trajectory, so ``success`` is
    written as 1.
    """
    from libero.libero import benchmark, get_libero_path

    benchmark_dict = benchmark.get_benchmark_dict()
    if task_suite_name not in benchmark_dict:
        raise ValueError(
            f"Unknown task suite {task_suite_name!r}. Available: {list(benchmark_dict)}"
        )
    task_suite = benchmark_dict[task_suite_name]()
    num_tasks = task_suite.n_tasks
    if max_tasks is not None:
        num_tasks = min(num_tasks, max_tasks)

    datasets_root = get_libero_path("datasets")

    global_episode = 0
    total_stored = 0

    for task_id in range(num_tasks):
        task = task_suite.get_task(task_id)
        task_description = task.language
        writer.register_task(task_id, task_description)

        demo_path = _find_demo_file(datasets_root, task_suite_name, task)
        with h5py.File(demo_path, "r") as f:
            data_group = f["data"]
            demo_keys = sorted(
                data_group.keys(), key=lambda k: int(k.split("_")[-1])
            )
            if max_demos_per_task is not None:
                demo_keys = demo_keys[:max_demos_per_task]

            for demo_key in tqdm(
                demo_keys, desc=f"task {task_id}: {task_description[:40]}"
            ):
                obs_group = data_group[f"{demo_key}/obs"]
                resolved_key = _resolve_image_key(obs_group, image_key)
                frames = obs_group[resolved_key]
                n_steps = frames.shape[0]
                if max_steps_per_demo is not None:
                    n_steps = min(n_steps, max_steps_per_demo)

                for t in range(n_steps):
                    pil = _demo_image(np.asarray(frames[t]), center_crop=center_crop)
                    _action, acts = predict_and_capture(
                        model, processor, collector, pil,
                        task_description, device, unnorm_key,
                    )
                    writer.add(
                        acts,
                        episode=global_episode,
                        timestep=t,
                        task_id=task_id,
                        success=1,
                    )
                    total_stored += 1
                global_episode += 1

    return total_stored
