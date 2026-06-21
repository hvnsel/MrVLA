# MrVLA — OpenVLA activation collection

Run inference on [OpenVLA](https://github.com/openvla/openvla) and dump residual-stream
activations to disk so a separate script can train a Sparse Autoencoder (SAE) on them.

Activations are captured with forward hooks on the LLM decoder layers and **mean-pooled
over tokens per timestep** (matching *"Sparse Autoencoders Reveal Interpretable and
Steerable Features in VLA Models"*). Each saved vector carries `(episode, timestep,
task_id, success)` metadata so the downstream SAE pipeline can compute episode coverage,
onset count, and run-length for the general-vs-memorized classification.

## Layout

```
collect_activations.py     # CLI entrypoint
mrvla/
  model_utils.py           # load OpenVLA, locate decoder layers, predict+capture
  hooks.py                 # ActivationCollector (forward hooks, pooling)
  store.py                 # ShardedActivationWriter (.npz shards + manifest.json)
  sources.py               # image-folder smoke-test source
  libero_collect.py        # closed-loop LIBERO rollout collection
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

## 2. Real collection — LIBERO rollouts

Requires the LIBERO benchmark installed (`pip install -e .` from the LIBERO repo).

```bash
python collect_activations.py \
  --source libero \
  --model openvla/openvla-7b-finetuned-libero-spatial \
  --task-suite libero_spatial \
  --unnorm-key libero_spatial \
  --trials-per-task 20 \
  --out ./activations/libero_spatial
```

By default every decoder layer is captured. Restrict with `--layers 6,12,18,24,30` to
mimic the paper's 8-layer subset and cut storage.

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
