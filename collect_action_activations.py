"""Path A / Stage A1: collect layer-31 residuals AT THE ACTION-TOKEN POSITIONS.

Unlike collect_activations.py (which mean-pools the prefill prompt), this replays LIBERO
demonstrations and captures, per (episode, timestep), the un-pooled residual at each of
the 7 action-token decode positions, together with the emitted action token ids. It also
exports the head constants Path A attribution needs (see mrvla/attribution.py DATA
CONTRACT): the 256 action-token unembedding rows, the final-RMSNorm gain, and eps.

This is the input to the retrained SAE (Stage A2) and then the attribution/gate module.

CANNOT be unit-tested without the model; validate on the cluster with --max-demos-per-task
1 --max-tasks 1 first and check the printed shapes + that re-decoding a captured residual
reproduces the emitted token (that check IS gate level 1 on a handful of decisions).

Usage
-----
python collect_action_activations.py \
    --model openvla/openvla-7b-finetuned-libero-goal \
    --task-suite libero_goal --unnorm-key libero_goal \
    --layer 31 --out ACT_ACTION/libero_goal
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
import torch

from mrvla.hooks import ActionPositionCollector
from mrvla.libero_demos import _demo_image, _find_demo_file, _resolve_image_key
from mrvla.model_utils import build_inputs, load_openvla, locate_decoder_layers


# ---------------------------------------------------------------------------
# Head constants (once per model)
# ---------------------------------------------------------------------------
def export_head_constants(model, out_dir: str, n_bins: int = 256) -> dict:
    """Save the unembedding action rows, final-norm gain, and eps. Returns a summary.

    CAREFUL with the action-token range. The lm_head has V_full rows (32064 =
    32000 + pad_to_multiple_of), but OpenVLA decodes actions with
    ``self.vocab_size = text_config.vocab_size - pad_to_multiple_of`` (= 32000), via
    ``token_id = self.vocab_size - bin_index``, bin_index in [1, n_bins]. So the action
    tokens are ids {A - n_bins, ..., A - 1} where A = model.vocab_size (32000), NOT the
    last rows of the lm_head. Using V_full here would select the wrong 256 rows (off by
    pad_to_multiple_of) and every attribution would be wrong.
    """
    lm_head = model.get_output_embeddings()                       # Linear(d -> V_full)
    W_U = lm_head.weight.detach().float().cpu().numpy()           # [V_full, d]
    V_full = W_U.shape[0]
    A = int(model.vocab_size)                                     # action vocab (32000)
    W_U_act = W_U[A - n_bins:A]                                   # [n_bins, d], ids A-256..A-1
    act_ids = np.arange(A - n_bins, A, dtype=np.int64)

    # Final RMSNorm: language_model.model.norm (Llama). Fall back to a search.
    norm = None
    try:
        norm = model.language_model.model.norm
    except AttributeError:
        for name, mod in model.named_modules():
            if name.endswith("model.norm") and hasattr(mod, "weight"):
                norm = mod
    if norm is None or not hasattr(norm, "weight"):
        raise RuntimeError("Could not locate the final RMSNorm (expected "
                           "language_model.model.norm with a .weight).")
    g = norm.weight.detach().float().cpu().numpy()                # [d]
    eps = float(getattr(norm, "variance_epsilon", getattr(norm, "eps", 1e-5)))

    path = os.path.join(out_dir, "head_constants.npz")
    np.savez_compressed(path, W_U_act=W_U_act.astype(np.float32),
                        act_ids=act_ids, g=g.astype(np.float32),
                        eps=np.float32(eps), action_vocab=np.int64(A),
                        lm_head_vocab=np.int64(V_full), n_bins=np.int64(n_bins))
    return {"path": path, "action_vocab": A, "lm_head_vocab": V_full,
            "action_token_ids": [int(A - n_bins), int(A - 1)], "d": int(W_U.shape[1]),
            "n_action_tokens": int(n_bins), "eps": eps}


# ---------------------------------------------------------------------------
# One decision: run generate, capture residuals + emitted token ids
# ---------------------------------------------------------------------------
@torch.no_grad()
def decide_and_capture(model, processor, collector, image, instruction, device,
                       action_dim: int = 7):
    """Return (residuals [7, d] float16, token_ids [7] int64) for one image+instruction.

    Replicates OpenVLA predict_action's generate call (the 29871 append + max_new_tokens),
    but with return_dict_in_generate so we can read the emitted token ids, and with the
    ActionPositionCollector hook capturing the 7 decode-position residuals.
    """
    inputs = build_inputs(processor, image, instruction, device)
    input_ids = inputs["input_ids"]
    # Drop attention_mask: predict_action appends 29871 to input_ids but not to any mask
    # we pass, which desyncs lengths (same workaround as model_utils.predict_and_capture).
    pixel_values = inputs["pixel_values"]

    # Append the SentencePiece empty token if the prompt does not already end with it.
    if not torch.all(input_ids[:, -1] == 29871):
        empty = torch.tensor([[29871]], dtype=torch.long, device=input_ids.device)
        input_ids = torch.cat((input_ids, empty), dim=1)

    collector.reset()
    out = model.generate(
        input_ids, pixel_values=pixel_values, max_new_tokens=action_dim,
        do_sample=False, use_cache=True, return_dict_in_generate=True,
    )
    residuals = collector.stack(expected=action_dim)              # [7, d]
    token_ids = out.sequences[0, -action_dim:].detach().cpu().numpy().astype(np.int64)
    return residuals.astype(np.float16), token_ids


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True)
    p.add_argument("--task-suite", required=True)
    p.add_argument("--unnorm-key", required=True, help="kept for parity; not used for capture")
    p.add_argument("--layer", type=int, default=31, help="decoder layer to capture (default 31)")
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--no-flash-attn", action="store_true")
    p.add_argument("--max-tasks", type=int, default=None)
    p.add_argument("--max-demos-per-task", type=int, default=None)
    p.add_argument("--max-steps-per-demo", type=int, default=None)
    p.add_argument("--image-key", default=None)
    p.add_argument("--shard-size", type=int, default=4096)
    p.add_argument("--action-dim", type=int, default=7)
    args = p.parse_args()

    from libero.libero import benchmark, get_libero_path

    os.makedirs(args.out, exist_ok=True)
    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"[A1] loading {args.model} on {device}", flush=True)
    model, processor = load_openvla(args.model, device=device,
                                    use_flash_attn=not args.no_flash_attn)

    layers = locate_decoder_layers(model)
    if not (0 <= args.layer < len(layers)):
        raise ValueError(f"--layer {args.layer} out of range [0,{len(layers)})")
    collector = ActionPositionCollector(layers[args.layer], dtype=torch.float16)

    const = export_head_constants(model, args.out, n_bins=256)
    print(f"[A1] head constants: {const}", flush=True)

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite]()
    n_tasks = task_suite.n_tasks if args.max_tasks is None else min(task_suite.n_tasks, args.max_tasks)
    datasets_root = get_libero_path("datasets")

    # Buffers -> sharded npz
    res_buf, tok_buf, ep_buf, ts_buf, task_buf = [], [], [], [], []
    shard_idx = [0]
    global_episode = 0
    total = 0

    def flush():
        if not res_buf:
            return
        path = os.path.join(args.out, f"shard_{shard_idx[0]:05d}.npz")
        np.savez_compressed(
            path,
            residual=np.stack(res_buf).astype(np.float16),   # [n, 7, d]
            token_ids=np.stack(tok_buf).astype(np.int64),    # [n, 7]
            episode=np.array(ep_buf, np.int32),
            timestep=np.array(ts_buf, np.int32),
            task_id=np.array(task_buf, np.int32),
        )
        print(f"[A1]   wrote {path}  n={len(res_buf)}  residual={np.stack(res_buf).shape}",
              flush=True)
        res_buf.clear(); tok_buf.clear(); ep_buf.clear(); ts_buf.clear(); task_buf.clear()
        shard_idx[0] += 1

    try:
        for task_id in range(n_tasks):
            task = task_suite.get_task(task_id)
            instr = task.language
            demo_path = _find_demo_file(datasets_root, args.task_suite, task)
            import h5py
            with h5py.File(demo_path, "r") as f:
                data = f["data"]
                demo_keys = sorted(data.keys(), key=lambda k: int(k.split("_")[-1]))
                if args.max_demos_per_task is not None:
                    demo_keys = demo_keys[:args.max_demos_per_task]
                for dk in demo_keys:
                    obs = data[f"{dk}/obs"]
                    key = _resolve_image_key(obs, args.image_key)
                    frames = obs[key]
                    T = frames.shape[0]
                    if args.max_steps_per_demo is not None:
                        T = min(T, args.max_steps_per_demo)
                    for t in range(T):
                        pil = _demo_image(np.asarray(frames[t]))
                        res, toks = decide_and_capture(
                            model, processor, collector, pil, instr, device,
                            action_dim=args.action_dim)
                        res_buf.append(res); tok_buf.append(toks)
                        ep_buf.append(global_episode); ts_buf.append(t)
                        task_buf.append(task_id)
                        total += 1
                        if len(res_buf) >= args.shard_size:
                            flush()
                    global_episode += 1
            print(f"[A1] task {task_id} done ({instr[:40]})  total decisions={total}",
                  flush=True)
        flush()
    finally:
        collector.remove()

    manifest = {
        "model_name": args.model, "task_suite": args.task_suite, "layer": args.layer,
        "action_dim": args.action_dim, "hidden_dim": const["d"],
        "total_decisions": total, "n_shards": shard_idx[0],
        **const,
    }
    with open(os.path.join(args.out, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2, default=str)
    print(f"[A1] done. {total} decisions across {shard_idx[0]} shards -> {args.out}",
          flush=True)


if __name__ == "__main__":
    main()
