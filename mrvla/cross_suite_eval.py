"""Cross-suite + in-distribution success-rate eval: base model vs base+LoRA.

Runs closed-loop LIBERO rollouts for two model variants across three task
suites and stores BOTH the success label per episode AND the pooled
residual-stream activations (via the existing ActivationCollector / writer),
so the same runs double as input to the post-training generality
re-classification step (extract_codes_and_metrics.py / generality_classifier.py).

  variants: base (no adapter), lora (base + trained LoRA adapter, merged)
  suites:   libero_goal (in-distribution), libero_spatial, libero_object (cross-suite)

IMPORTANT — pairing for a fair comparison:
  Both variants use the same --seed, so LiberoBenchmark hands out the same
  init states in the same order for a given (suite, trial) to both the base
  and lora runs. That's what makes the paired significance test in
  summarize_success.py meaningful (McNemar rather than an unpaired
  two-proportion test). Don't change --seed between the two variant runs.

Usage
-----
  # quick sanity pass on ghx4-interactive before committing to the full run
  python run_cross_suite_eval.py \\
      --base-model openvla/openvla-7b-finetuned-libero-goal \\
      --unnorm-key libero_goal \\
      --lora-adapter ./lora_checkpoints/run_001/final \\
      --out-root ./activations/cross_suite_eval_pilot \\
      --trials-per-task 5 --max-tasks 2

  # full run
  python run_cross_suite_eval.py \\
      --base-model openvla/openvla-7b-finetuned-libero-goal \\
      --unnorm-key libero_goal \\
      --lora-adapter ./lora_checkpoints/run_001/final \\
      --out-root ./activations/cross_suite_eval \\
      --trials-per-task 50 --seed 0
"""
from __future__ import annotations

import argparse
import json
import os

import torch

# PyTorch 2.6 changed weights_only default to True.  LoRA checkpoints saved
# via peft.save_pretrained (pre-safetensors) contain numpy arrays, which are
# blocked by the new default.  Patch torch.load globally before peft is
# imported so every internal load inside peft/transformers uses weights_only=False.
_orig_torch_load = torch.load
def _torch_load_compat(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _orig_torch_load(*args, **kwargs)
torch.load = _torch_load_compat

from mrvla.hooks import ActivationCollector
from mrvla.libero_collect import collect_libero
from mrvla.model_utils import get_hidden_dim, load_openvla, locate_decoder_layers
from mrvla.store import ShardedActivationWriter

SUITES = ["libero_goal", "libero_spatial", "libero_object"]


def load_model_variant(base_model: str, device: str, lora_adapter: str | None, no_flash_attn: bool):
    """Load the base checkpoint, optionally merging a trained LoRA adapter into it.

    merge_and_unload() (rather than leaving the PeftModel wrapper live) is
    deliberate for eval: it avoids relying on PeftModel's attribute-forwarding
    to reach the model's custom predict_action(), removes LoRA-adapter
    overhead from the timing-sensitive closed-loop rollout, and gives you the
    exact weights you'd actually ship if this intervention were adopted.
    """
    model, processor = load_openvla(base_model, device=device, use_flash_attn=not no_flash_attn)
    if lora_adapter is not None:
        from peft import PeftModel
        print(f"[eval] attaching LoRA adapter: {lora_adapter}")
        model = PeftModel.from_pretrained(model, lora_adapter)
        model = model.merge_and_unload()
        model = model.to(device)
    model.eval()
    return model, processor


def run_one(model, processor, layers, hidden_dim, layer_indices, args, variant_name, suite):
    out_dir = os.path.join(args.out_root, f"{variant_name}__{suite}")
    manifest_path = os.path.join(out_dir, "manifest.json")
    if os.path.exists(manifest_path) and not args.overwrite:
        print(f"[eval] skipping {out_dir} (manifest.json already exists; pass --overwrite to redo)")
        return out_dir

    collector = ActivationCollector(layers, layer_indices=layer_indices, pool="mean", dtype=torch.float16)
    writer = ShardedActivationWriter(
        out_dir=out_dir,
        layer_indices=layer_indices,
        hidden_dim=hidden_dim,
        model_name=f"{args.base_model}+{variant_name}",
        dtype="float16",
        pool="mean",
        extra_meta={
            "source": "libero",
            "unnorm_key": args.unnorm_key,
            "task_suite": suite,
            "variant": variant_name,
            "lora_adapter": args.lora_adapter if variant_name == "lora" else None,
            "seed": args.seed,
        },
    )
    try:
        n = collect_libero(
            model, processor, collector, writer,
            task_suite_name=suite,
            unnorm_key=args.unnorm_key,
            device=args.device,
            trials_per_task=args.trials_per_task,
            seed=args.seed,
            max_tasks=args.max_tasks,
        )
    finally:
        collector.remove()
        writer.close()
    print(f"[eval]   {variant_name} / {suite}: {n} timesteps stored -> {out_dir}")
    return out_dir


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base-model", required=True, help="e.g. openvla/openvla-7b-finetuned-libero-goal")
    p.add_argument("--unnorm-key", required=True, help="e.g. libero_goal — stays fixed across suites; it's tied to the checkpoint's own norm_stats, not the eval suite.")
    p.add_argument("--lora-adapter", required=True, help="Path to the trained LoRA adapter dir (e.g. .../final)")
    p.add_argument("--out-root", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--trials-per-task", type=int, default=50)
    p.add_argument("--max-tasks", type=int, default=None)
    p.add_argument("--seed", type=int, default=0, help="MUST match across the base and lora runs to keep trials paired.")
    p.add_argument("--layers", default="8,31", help="Comma-separated layers to capture for the reclassification step, e.g. '8,31' or 'all'.")
    p.add_argument("--suites", default=",".join(SUITES), help="Comma-separated subset of suites to run this invocation, e.g. 'libero_spatial'. Lets you split variant x suite combos across separate SLURM jobs to cut wall-clock time within a fixed GPU-hour budget — each combo is independent and skip-if-done via --overwrite.")
    p.add_argument("--variants", default="base,lora", help="Comma-separated subset of {base,lora} to run this invocation.")
    p.add_argument("--no-flash-attn", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()
    suites_to_run = [s.strip() for s in args.suites.split(",") if s.strip()]
    variants_to_run = [v.strip() for v in args.variants.split(",") if v.strip()]

    os.makedirs(args.out_root, exist_ok=True)
    results: dict[str, dict[str, str]] = {}

    all_variants = [("base", None), ("lora", args.lora_adapter)]
    for variant_name, adapter in [(n, a) for n, a in all_variants if n in variants_to_run]:
        print(f"\n[eval] === variant: {variant_name} ===")
        model, processor = load_model_variant(args.base_model, args.device, adapter, args.no_flash_attn)
        layers_module = locate_decoder_layers(model)
        hidden_dim = get_hidden_dim(layers_module)

        num_layers = len(layers_module)
        if args.layers.strip().lower() == "all":
            layer_indices = list(range(num_layers))
        else:
            layer_indices = [int(x) for x in args.layers.split(",") if x.strip()]

        for suite in suites_to_run:
            print(f"[eval] -- suite: {suite} --")
            out_dir = run_one(model, processor, layers_module, hidden_dim, layer_indices, args, variant_name, suite)
            results.setdefault(variant_name, {})[suite] = out_dir

        del model
        torch.cuda.empty_cache()

    index_path = os.path.join(args.out_root, "run_index.json")
    if os.path.exists(index_path):
        with open(index_path) as f:
            existing = json.load(f)
        for variant_name, suite_map in results.items():
            existing.setdefault(variant_name, {}).update(suite_map)
        results = existing
    with open(index_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[eval] done. Run index -> {index_path}")
    print("[eval] next: python summarize_success.py --out-root", args.out_root)


if __name__ == "__main__":
    main()