"""Path A / Stage A2 prep: convert action-position residuals into SAE-trainable shards.

collect_action_activations.py (A1) writes shards of `residual [N, 7, d]` (7 action
positions per decision) plus token_ids/episode/timestep/task_id. The SAE trainer
(mrvla/train_sae.py) instead expects shards of `acts [N, L, H]` and a manifest with
`layer_indices` / `hidden_dim`. This adapter bridges the two: it flattens the 7 action
positions into 7*N independent training samples at a single captured "layer", so

    python -m mrvla.train_sae --acts-dir <this out-dir> --layer 31 --seed 0

trains an SAE on the action-position residual distribution.

The original A1 shards are left untouched -- attribution (A3/A4) reads THOSE (which keep
token_ids and task_id) and encodes them with the SAE trained here. This adapter only
produces the training corpus.

Pure numpy; unit-tested.

Usage
-----
python adapt_action_acts_for_sae.py \
    --in-dir  ACT_ACTION/goal \
    --out-dir ACT_ACTION_SAE/goal
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np


def adapt(in_dir: str, out_dir: str) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(in_dir, "manifest.json")) as f:
        man = json.load(f)
    layer = int(man.get("layer", 31))
    d = int(man.get("hidden_dim", 4096))

    shards = sorted(glob.glob(os.path.join(in_dir, "shard_*.npz")))
    if not shards:
        raise FileNotFoundError(f"No shard_*.npz in {in_dir!r}")

    total = 0
    for i, sp in enumerate(shards):
        res = np.load(sp)["residual"]                 # [n, 7, d] float16
        if res.ndim != 3:
            raise ValueError(f"{sp}: expected residual [n,7,d], got {res.shape}")
        n, seven, dd = res.shape
        if dd != d:
            raise ValueError(f"{sp}: hidden dim {dd} != manifest {d}")
        acts = res.reshape(n * seven, 1, dd).astype(np.float16)   # [7n, 1, d]
        out_path = os.path.join(out_dir, f"shard_{i:05d}.npz")
        np.savez_compressed(out_path, acts=acts)
        total += acts.shape[0]

    out_man = {
        "model_name": man.get("model_name", "unknown"),
        "layer_indices": [layer],
        "hidden_dim": d,
        "total_samples": total,
        "source": "action-position residuals (A1); 7 action slots flattened as samples",
        "source_dir": os.path.abspath(in_dir),
    }
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(out_man, f, indent=2)
    return {"n_shards": len(shards), "total_samples": total, "layer": layer, "d": d}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--in-dir", required=True, help="A1 output (residual shards)")
    p.add_argument("--out-dir", required=True, help="SAE-trainable output")
    args = p.parse_args()
    info = adapt(args.in_dir, args.out_dir)
    print(f"[adapt] {info['n_shards']} shards -> {info['total_samples']} samples "
          f"(layer {info['layer']}, d={info['d']}) -> {args.out_dir}")
    print(f"[adapt] next: python -m mrvla.train_sae --acts-dir {args.out_dir} "
          f"--layer {info['layer']} --seed 0")


if __name__ == "__main__":
    main()
