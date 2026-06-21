"""CLI: run OpenVLA inference and dump residual-stream activations for SAE training.

Examples
--------
Smoke test on a folder of images (no simulator needed):
    python collect_activations.py --source images \
        --model openvla/openvla-7b \
        --image-dir ./sample_images \
        --instruction "pick up the red block" \
        --unnorm-key bridge_orig \
        --out ./activations/smoke

Closed-loop LIBERO rollouts:
    python collect_activations.py --source libero \
        --model openvla/openvla-7b-finetuned-libero-spatial \
        --task-suite libero_spatial \
        --unnorm-key libero_spatial \
        --trials-per-task 20 \
        --out ./activations/libero_spatial
"""

from __future__ import annotations

import argparse

import torch

from mrvla.hooks import ActivationCollector
from mrvla.libero_collect import collect_libero
from mrvla.model_utils import get_hidden_dim, load_openvla, locate_decoder_layers
from mrvla.sources import collect_from_image_dir
from mrvla.store import ShardedActivationWriter


def parse_layers(spec: str | None, num_layers: int) -> list[int]:
    if spec is None or spec.strip().lower() == "all":
        return list(range(num_layers))
    indices = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        idx = int(part)
        if idx < 0:
            idx += num_layers
        if not 0 <= idx < num_layers:
            raise ValueError(f"Layer index {idx} out of range [0, {num_layers}).")
        indices.append(idx)
    if not indices:
        raise ValueError("No valid layer indices parsed.")
    return indices


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", choices=["images", "libero"], required=True)
    p.add_argument("--model", required=True, help="HF model id or local path to an OpenVLA checkpoint.")
    p.add_argument("--out", required=True, help="Output directory for activation shards.")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--no-flash-attn", action="store_true", help="Disable flash-attention-2.")

    # Activation capture
    p.add_argument("--layers", default="all", help='"all" or comma-separated indices, e.g. "6,12,18,24".')
    p.add_argument("--pool", choices=["mean", "last"], default="mean")
    p.add_argument("--store-dtype", choices=["float16", "float32"], default="float16")
    p.add_argument("--shard-size", type=int, default=8192)
    p.add_argument("--unnorm-key", default=None, help="Action de-normalization key (required for libero).")

    # images source
    p.add_argument("--image-dir", default=None)
    p.add_argument("--instruction", default=None)
    p.add_argument("--limit", type=int, default=None)

    # libero source
    p.add_argument("--task-suite", default="libero_spatial")
    p.add_argument("--trials-per-task", type=int, default=20)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--max-tasks", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--store-only-success", action="store_true")
    return p


def main() -> None:
    args = build_arg_parser().parse_args()

    print(f"[mrvla] loading {args.model} ...")
    model, processor = load_openvla(
        args.model, device=args.device, use_flash_attn=not args.no_flash_attn
    )

    layers = locate_decoder_layers(model)
    num_layers = len(layers)
    hidden_dim = get_hidden_dim(layers)
    layer_indices = parse_layers(args.layers, num_layers)
    print(
        f"[mrvla] found {num_layers} decoder layers (hidden={hidden_dim}); "
        f"capturing {len(layer_indices)} layers: {layer_indices}"
    )

    collector = ActivationCollector(
        layers,
        layer_indices=layer_indices,
        pool=args.pool,
        dtype=torch.float16 if args.store_dtype == "float16" else torch.float32,
    )

    extra_meta = {
        "source": args.source,
        "unnorm_key": args.unnorm_key,
        "task_suite": args.task_suite if args.source == "libero" else None,
    }

    writer = ShardedActivationWriter(
        out_dir=args.out,
        layer_indices=layer_indices,
        hidden_dim=hidden_dim,
        model_name=args.model,
        shard_size=args.shard_size,
        dtype=args.store_dtype,
        pool=args.pool,
        extra_meta=extra_meta,
    )

    try:
        if args.source == "images":
            if not args.image_dir or not args.instruction:
                raise SystemExit("--image-dir and --instruction are required for --source images")
            n = collect_from_image_dir(
                model, processor, collector, writer,
                image_dir=args.image_dir,
                instruction=args.instruction,
                device=args.device,
                unnorm_key=args.unnorm_key,
                limit=args.limit,
            )
        else:  # libero
            if not args.unnorm_key:
                raise SystemExit("--unnorm-key is required for --source libero (e.g. libero_spatial)")
            n = collect_libero(
                model, processor, collector, writer,
                task_suite_name=args.task_suite,
                unnorm_key=args.unnorm_key,
                device=args.device,
                trials_per_task=args.trials_per_task,
                max_steps=args.max_steps,
                max_tasks=args.max_tasks,
                seed=args.seed,
                store_only_success=args.store_only_success,
            )
    finally:
        collector.remove()
        writer.close()

    print(f"[mrvla] done. stored {n} timesteps across {writer._shard_idx} shards in {args.out}")


if __name__ == "__main__":
    main()
