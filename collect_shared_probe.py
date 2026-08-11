"""Collect a SHARED probe-frame activation set across multiple OpenVLA models.

Purpose (EXPERIMENT_PLAN.md §3.1, Path B — cross-model recurrence)
------------------------------------------------------------------
To ask whether feature i of model A is "the same feature" as feature j of model B,
we correlate their activations **over the same inputs**.  The per-suite activations
already collected can't support this: each model only ever saw its own suite's
frames, so no two models share an input basis.

This script fixes that.  It:
  1. Assembles ONE fixed set of probe frames from the LIBERO demonstration HDF5s,
     pooled across suites.  Demo frames are recorded teleop observations, so they
     are **model-independent** (unlike closed-loop rollouts, which would differ per
     model and break frame alignment).
  2. Replays that identical frame set through every model in --models (the four
     fine-tuned checkpoints plus, by default, the base openvla-7b), capturing
     mean-pooled prefill activations at --layers.
  3. Saves one aligned activation matrix per model (row = shared frame, same order
     for every model) plus a shared manifest.

Downstream (`run_recurrence.py`): encode each model's probe activations with that
model's SAE and correlate columns across models.

Including the base openvla-7b is the base-inheritance reference: a feature that
recurs across the fine-tuned models only because it was inherited from the base can
be caught by checking whether it also responds to the base model's residual (encode
base probe activations through a fine-tuned SAE).

Preprocessing note: the SAME image transform (flip/resize/center-crop) is applied to
every model so the pixels are identical across models.  The default center-crop
matches the fine-tuned checkpoints; the base model sees the same crop for
comparability (a documented, deliberate choice).

Usage
-----
python collect_shared_probe.py \
    --out ./activations/shared_probe_v1 \
    --suites libero_goal,libero_spatial,libero_object,libero_10 \
    --frames-per-suite 200 \
    --layers 0,8,16,24,31
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

from mrvla.hooks import ActivationCollector
from mrvla.libero_demos import _demo_image, _find_demo_file, _resolve_image_key
from mrvla.model_utils import (
    get_hidden_dim,
    load_openvla,
    locate_decoder_layers,
    predict_and_capture,
)

# Default models: the four LIBERO fine-tunes + the base checkpoint (inheritance ref).
DEFAULT_MODELS = {
    "goal": "openvla/openvla-7b-finetuned-libero-goal",
    "spatial": "openvla/openvla-7b-finetuned-libero-spatial",
    "object": "openvla/openvla-7b-finetuned-libero-object",
    "libero10": "openvla/openvla-7b-finetuned-libero-10",
    "base": "openvla/openvla-7b",
}
DEFAULT_SUITES = ["libero_goal", "libero_spatial", "libero_object", "libero_10"]


# ---------------------------------------------------------------------------
# Phase 1: assemble a fixed, model-independent probe frame set
# ---------------------------------------------------------------------------
def build_probe_frames(suites: list[str], frames_per_suite: int, seed: int,
                       max_tasks: int | None, image_key: str | None) -> list[dict]:
    """Sample a reproducible set of demo frames spread across suites/tasks/demos.

    Returns a list of dicts, each with the RAW frame (uint8 HxWx3), the task
    instruction, and provenance metadata.  The image transform is applied later,
    identically for every model.
    """
    from libero.libero import benchmark, get_libero_path

    rng = np.random.default_rng(seed)
    datasets_root = get_libero_path("datasets")
    benchmark_dict = benchmark.get_benchmark_dict()
    frames: list[dict] = []

    for suite in suites:
        if suite not in benchmark_dict:
            raise ValueError(f"Unknown suite {suite!r}; available: {list(benchmark_dict)}")
        task_suite = benchmark_dict[suite]()
        n_tasks = task_suite.n_tasks if max_tasks is None else min(task_suite.n_tasks, max_tasks)

        # Round-robin over tasks so frames are spread, not clustered in a few tasks.
        got = 0
        task_order = list(range(n_tasks))
        rng.shuffle(task_order)
        # candidate (task, demo, t) tuples, filled lazily per task
        per_task_target = max(1, frames_per_suite // max(n_tasks, 1) + 1)

        import h5py
        for task_id in task_order:
            if got >= frames_per_suite:
                break
            task = task_suite.get_task(task_id)
            instr = task.language
            demo_path = _find_demo_file(datasets_root, suite, task)
            with h5py.File(demo_path, "r") as f:
                data_group = f["data"]
                demo_keys = sorted(data_group.keys(), key=lambda k: int(k.split("_")[-1]))
                rng.shuffle(demo_keys)
                taken_here = 0
                for demo_key in demo_keys:
                    if taken_here >= per_task_target or got >= frames_per_suite:
                        break
                    obs = data_group[f"{demo_key}/obs"]
                    key = _resolve_image_key(obs, image_key)
                    arr = obs[key]
                    T = arr.shape[0]
                    # a few strided timesteps per demo
                    n_pick = min(3, T)
                    ts = np.linspace(0, T - 1, n_pick).round().astype(int)
                    for t in ts:
                        if taken_here >= per_task_target or got >= frames_per_suite:
                            break
                        frames.append({
                            "image": np.asarray(arr[int(t)], dtype=np.uint8),
                            "instruction": instr,
                            "suite": suite, "task_id": int(task_id),
                            "demo": demo_key, "timestep": int(t),
                        })
                        got += 1
                        taken_here += 1
        print(f"[probe] {suite}: {got} frames", flush=True)

    print(f"[probe] total probe frames: {len(frames)}", flush=True)
    return frames


# ---------------------------------------------------------------------------
# Phase 2: replay the fixed frames through one model
# ---------------------------------------------------------------------------
def run_model_over_frames(model_id: str, frames: list[dict], layer_indices: list[int],
                          device: str, center_crop: bool, use_flash_attn: bool,
                          pool: str = "mean"):
    """Return acts [N_frames, L, H] float16 for one model over the shared frames.

    pool="mean"  -> mean-pooled prefill activation (the original recurrence input; matches
                    the main-study SAEs).
    pool="last"  -> the prefill's LAST-position residual = the FIRST action-token position,
                    un-pooled. This is a pure function of the input frame (no generated
                    tokens feed it, so models that would take different actions do not
                    diverge here), and it is the same residual distribution the Path A
                    action-position SAEs were trained on -- so recurrence run on this probe
                    with the ACTION-POSITION SAEs is per-feature comparable to Path A.
    """
    print(f"[probe] loading {model_id} (pool={pool}) ...", flush=True)
    model, processor = load_openvla(model_id, device=device, use_flash_attn=use_flash_attn)
    layers = locate_decoder_layers(model)
    hidden = get_hidden_dim(layers)
    collector = ActivationCollector(layers, layer_indices=layer_indices, pool=pool,
                                    dtype=torch.float16)
    acts = np.empty((len(frames), len(layer_indices), hidden), dtype=np.float16)
    try:
        for i, fr in enumerate(frames):
            pil = _demo_image(fr["image"], center_crop=center_crop)
            # unnorm_key=None: we only need activations; forward triggers the hooks.
            _action, a = predict_and_capture(
                model, processor, collector, pil, fr["instruction"], device,
                unnorm_key=None,
            )
            acts[i] = a  # [L, H]
            if i % 100 == 0 or i == len(frames) - 1:
                print(f"    {model_id}: frame {i+1}/{len(frames)}", flush=True)
    finally:
        collector.remove()
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return acts


def parse_models(spec: str | None) -> dict[str, str]:
    if not spec:
        return dict(DEFAULT_MODELS)
    out = {}
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise ValueError(f"--models entries must be key=model_id, got {part!r}")
        k, v = part.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True)
    p.add_argument("--suites", default=",".join(DEFAULT_SUITES))
    p.add_argument("--models", default=None,
                   help="Comma list of key=model_id. Default: 4 fine-tunes + base.")
    p.add_argument("--layers", default="0,8,16,24,31")
    p.add_argument("--pool", choices=["mean", "last"], default="mean",
                   help="mean = pooled prefill (original; use with the prefill SAEs). "
                        "last = first action-position residual, un-pooled (use with the "
                        "ACTION-POSITION SAEs; makes recurrence per-feature comparable to "
                        "Path A). For 'last' you typically want --layers 31.")
    p.add_argument("--frames-per-suite", type=int, default=200)
    p.add_argument("--max-tasks", type=int, default=None)
    p.add_argument("--image-key", default=None)
    p.add_argument("--center-crop", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--no-flash-attn", action="store_true")
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    suites = [s.strip() for s in args.suites.split(",") if s.strip()]
    layer_indices = [int(x) for x in args.layers.split(",") if x.strip()]
    models = parse_models(args.models)

    # Phase 1 — build the shared frame set once.
    frames = build_probe_frames(suites, args.frames_per_suite, args.seed,
                                args.max_tasks, args.image_key)
    meta = [{k: fr[k] for k in ("instruction", "suite", "task_id", "demo", "timestep")}
            for fr in frames]
    manifest = {
        "n_frames": len(frames), "suites": suites, "layers": layer_indices,
        "models": models, "center_crop": bool(args.center_crop), "seed": args.seed,
        "pool": args.pool, "meta": meta, "frames": meta,
    }
    with open(os.path.join(args.out, "probe_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[probe] wrote manifest ({len(frames)} frames, {len(models)} models)", flush=True)

    # Phase 2 — replay identical frames through every model.
    for key, model_id in models.items():
        acts = run_model_over_frames(
            model_id, frames, layer_indices, args.device,
            center_crop=args.center_crop, use_flash_attn=not args.no_flash_attn,
            pool=args.pool,
        )
        out_path = os.path.join(args.out, f"probe_{key}.npz")
        np.savez_compressed(
            out_path, acts=acts, layers=np.asarray(layer_indices, dtype=np.int32),
            model_id=model_id,
        )
        print(f"[probe] saved {out_path}  acts={acts.shape}", flush=True)

    print(f"[probe] done -> {args.out}\n"
          f"[probe] next: python run_recurrence.py --probe-dir {args.out} ...",
          flush=True)


if __name__ == "__main__":
    main()
