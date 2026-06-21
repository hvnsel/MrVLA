"""Lightweight data sources for activation collection (no simulator required)."""

# just to be used for smoke test

from __future__ import annotations

import os

from PIL import Image
from tqdm import tqdm

from .model_utils import predict_and_capture

_IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def collect_from_image_dir(
    model,
    processor,
    collector, # expects ActivationCollector instance
    writer, # expects ShardedActivationWriter instance
    image_dir: str,
    instruction: str,
    device: str,
    unnorm_key: str | None = None,
    task_id: int = 0,
    limit: int | None = None,
) -> int:
    """Run OpenVLA over every image in a folder, storing pooled activations.

    Each image is treated as its own single-step "episode". This is a smoke test for the
    hook + storage pipeline; it does not require LIBERO or a simulator.
    """
    paths = sorted(
        os.path.join(image_dir, f)
        for f in os.listdir(image_dir)
        if os.path.splitext(f)[1].lower() in _IMG_EXTS
    )
    if not paths:
        raise FileNotFoundError(f"No images found in {image_dir!r}.")
    if limit is not None:
        paths = paths[:limit]

    writer.register_task(task_id, instruction)
    count = 0
    for episode, path in enumerate(tqdm(paths, desc="images")):
        image = Image.open(path).convert("RGB")
        _action, acts = predict_and_capture(
            model, processor, collector, image, instruction, device, unnorm_key
        )
        writer.add(acts, episode=episode, timestep=0, task_id=task_id, success=-1)
        count += 1
    return count
