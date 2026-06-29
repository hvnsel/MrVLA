# MrVLA — OpenVLA activation collection

Run inference on [OpenVLA](https://github.com/openvla/openvla) and dump residual-stream
activations to disk so a separate script can train a Sparse Autoencoder (SAE) on them.

Activations are captured with forward hooks on the LLM decoder layers and **mean-pooled
over tokens per timestep** (matching *"Sparse Autoencoders Reveal Interpretable and
Steerable Features in VLA Models"*, [arXiv 2603.19183](https://arxiv.org/abs/2603.19183),
§A.1.3). Each saved vector carries `(episode, timestep, task_id, success)` metadata so
the downstream SAE pipeline can compute episode coverage, onset count, and run-length
for the general-vs-memorized classification.

## Layout

```
collect_activations.py     # CLI entrypoint
mrvla/
  model_utils.py           # load OpenVLA, locate decoder layers, predict+capture
  hooks.py                 # ActivationCollector (forward hooks, pooling)
  store.py                 # ShardedActivationWriter (.npz shards + manifest.json)
  sources.py               # image-folder smoke-test source
  libero_collect.py        # closed-loop LIBERO rollout collection
  libero_demos.py          # LIBERO demo-HDF5 replay (paper-faithful)
```

## Install

```bash
python -m venv .venv && .venv\Scripts\activate     # Windows
pip install -r requirements.txt
# Install a CUDA build of torch matching your driver, then optionally flash-attn.
```

## 1. Smoke test (no simulator needed)

Validate model loading + hooks + storage on a folder of images:

```bash
python collect_activations.py \
  --source images \
  --model openvla/openvla-7b \
  --image-dir ./sample_images \
  --instruction "pick up the red block" \
  --unnorm-key bridge_orig \
  --out ./activations/smoke
```

## 2. Paper-faithful collection — LIBERO demo replay

Replays the LIBERO fine-tuning demonstration HDF5s through the model. This is what
the paper uses for SAE training and the generality metrics (episode coverage, onset
count, run length), since those metrics are defined over the fine-tuning episode set.

Requires the LIBERO benchmark installed and its demo datasets downloaded
(`get_libero_path("datasets")/<suite>/*_demo.hdf5`).

```bash
python collect_activations.py \
  --source libero-demos \
  --model openvla/openvla-7b-finetuned-libero-goal \
  --task-suite libero_goal \
  --unnorm-key libero_goal \
  --layers 0,8,16,24,31 \
  --out ./activations/libero_goal_demos
```

The layer list `0,8,16,24,31` reproduces the paper's OpenVLA subset (§A.1.3: five
Llama-2 decoder layers). The paper's OpenVLA analysis uses LIBERO-Goal.

## 3. Closed-loop LIBERO rollouts (on-policy, not paper-faithful for SAE training)

Drives the simulator from initial states with the policy itself. Useful for
steering / on-policy analysis, but the observation distribution differs from the
fine-tuning data, so the paper's generality metrics will not match.

```bash
python collect_activations.py \
  --source libero \
  --model openvla/openvla-7b-finetuned-libero-spatial \
  --task-suite libero_spatial \
  --unnorm-key libero_spatial \
  --trials-per-task 20 \
  --out ./activations/libero_spatial
```

By default every decoder layer is captured. Restrict with `--layers 0,8,16,24,31` to
match the paper's OpenVLA subset and cut storage.

## Output format

`--out` directory contains:

- `shard_00000.npz`, `shard_00001.npz`, ... each with:
  - `acts`   `float16 [N, L, H]` — pooled residual stream per timestep
  - `episode`, `timestep`, `task_id`, `success`  `int32 [N]`
- `manifest.json` — model, layer indices, hidden dim, dtype, totals, task-id → text map

### Reading it back (for SAE training)

```python
import numpy as np, glob, json
manifest = json.load(open("activations/libero_spatial/manifest.json"))
for shard in sorted(glob.glob("activations/libero_spatial/shard_*.npz")):
    d = np.load(shard)
    acts = d["acts"]            # [N, L, H]
    layer3 = acts[:, 3, :]      # pick a layer to train an SAE on
```

## Storage rule of thumb

`bytes ≈ num_timesteps × num_layers × hidden_dim × 2` (float16).
At hidden=4096, 8 layers, ~200k timesteps ≈ 13 GB.
