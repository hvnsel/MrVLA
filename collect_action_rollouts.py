"""Collect closed-loop LIBERO rollouts with action-position residuals AND success labels.

The missing asset: until now, internals and outcomes lived in disjoint datasets. A1 has the
7 action-position residuals but replays demonstrations, so `success` is the constant 1.
`libero_collect` has real success labels but stores the mean-pooled prefill vector, which is
not what decodes the action. This produces both at once, which is what any test relating a
per-decision internal signal to episode outcome requires.

Writes A1-compatible shards (`residual`, `episode`, `timestep`, `task_id`) plus `action`
(what the simulator actually executed) and `success`, together with the same
`head_constants.npz` A1 exports, so `mrvla/attribution.py`'s data contract is satisfied and
the existing SAE / attribution / readout code applies unchanged.

Validate on one task first -- this cannot be unit-tested without the model and simulator:

    python collect_action_rollouts.py --model openvla/openvla-7b-finetuned-libero-goal \\
        --task-suite libero_goal --unnorm-key libero_goal --no-flash-attn \\
        --max-tasks 1 --trials-per-task 2 --out /tmp/rollout_smoke

Full run (50 init states x 10 tasks = 500 episodes, ~5 GB):

    python collect_action_rollouts.py --model openvla/openvla-7b-finetuned-libero-goal \\
        --task-suite libero_goal --unnorm-key libero_goal --no-flash-attn \\
        --trials-per-task 50 --out $BASE/ROLLOUT_ACTION/goal

`--no-flash-attn` is not optional on the Delta env: flash_attn is not installed there, and
`load_openvla` defaults it ON, so omitting the flag fails at model load. Every other slurm
script in this repo passes it for the same reason.
"""

from __future__ import annotations

import argparse
import json
import os

import torch

from collect_action_activations import export_head_constants
from mrvla.action_rollout import RolloutShardWriter, rollout_action_positions
from mrvla.hooks import ActionPositionCollector
from mrvla.model_utils import get_hidden_dim, load_openvla, locate_decoder_layers


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True)
    p.add_argument("--task-suite", required=True, help="e.g. libero_goal")
    p.add_argument("--unnorm-key", required=True, help="e.g. libero_goal")
    p.add_argument("--layer", type=int, default=31)
    p.add_argument("--out", required=True)
    p.add_argument("--trials-per-task", type=int, default=50,
                   help="capped at the suite's init-state count (50 for LIBERO)")
    p.add_argument("--max-tasks", type=int, default=None)
    p.add_argument("--max-steps", type=int, default=None, help="default: the suite's cap")
    p.add_argument("--action-dim", type=int, default=7)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--camera-res", type=int, default=256)
    p.add_argument("--no-center-crop", action="store_true")
    p.add_argument("--shard-size", type=int, default=4000)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--worker-id", type=int, default=0)
    p.add_argument("--n-workers", type=int, default=1)
    p.add_argument("--no-flash-attn", action="store_true",
                   help="required on Delta: flash_attn is not in the env")
    args = p.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out, exist_ok=True)

    model, processor = load_openvla(args.model, device=device,
                                    use_flash_attn=not args.no_flash_attn)
    layers = locate_decoder_layers(model)
    d = get_hidden_dim(layers)
    if not 0 <= args.layer < len(layers):
        raise SystemExit(f"--layer {args.layer} out of range (model has {len(layers)})")

    # Only worker 0 writes the head constants: they are a property of the model, and having
    # every worker race to write the same file into a shared --out is how a truncated npz
    # gets read back later as a silent corruption.
    if args.worker_id == 0:
        hc = export_head_constants(model, args.out)
        print(f"[rollout] head constants: {hc}", flush=True)

    collector = ActionPositionCollector(layers[args.layer])
    # Per-worker shard namespace. Without it, four workers sharing --out each start their
    # counter at 0 and overwrite one another -- silently, because the analysis just globs
    # shard_*.npz and reports on whatever is left.
    writer = RolloutShardWriter(args.out, hidden_dim=d, action_dim=args.action_dim,
                                shard_size=args.shard_size,
                                prefix=f"shard_w{args.worker_id}")
    try:
        per_task = rollout_action_positions(
            model, processor, collector, writer,
            task_suite_name=args.task_suite, unnorm_key=args.unnorm_key, device=device,
            trials_per_task=args.trials_per_task, max_steps=args.max_steps,
            seed=args.seed, camera_res=args.camera_res, max_tasks=args.max_tasks,
            center_crop=not args.no_center_crop, action_dim=args.action_dim,
            worker_id=args.worker_id, n_workers=args.n_workers,
        )
    finally:
        collector.remove()

    ep = sum(v["episodes"] for v in per_task.values())
    su = sum(v["successes"] for v in per_task.values())
    writer.close(extra={
        "model": args.model, "task_suite": args.task_suite, "layer": args.layer,
        "unnorm_key": args.unnorm_key, "seed": args.seed,
        "worker_id": args.worker_id, "n_workers": args.n_workers,
        "episodes": ep, "successes": su,
        "success_rate": (su / ep) if ep else None,
        "per_task": {str(k): v for k, v in per_task.items()},
    })
    # Per-worker copy so a sharded run can be summed without reopening the shards.
    with open(os.path.join(args.out, f"rollout_summary_w{args.worker_id}.json"), "w") as f:
        json.dump({"episodes": ep, "successes": su, "per_task": per_task}, f, indent=2)

    if ep:
        print(f"\n[rollout] {writer.total} timesteps, {ep} episodes, "
              f"{su} successes ({su / ep:.3f})")
        print(f"[rollout] FAILURES = {ep - su}. That is the sample size for anything "
              f"predicting failure; below ~50 the AUROC carries no usable interval.")
    else:
        print("\n[rollout] no episodes ran")
    print(f"[rollout] wrote {args.out}")


if __name__ == "__main__":
    main()
