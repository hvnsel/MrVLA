import numpy as np, json, pathlib

out = pathlib.Path("activations/test/test")

# manifest
m = json.loads((out / "manifest.json").read_text())
print(json.dumps(m, indent=2))

# first shard
d = np.load(out / "shard_00000.npz")
print("\nkeys:", list(d.keys()))
print("acts   shape:", d["acts"].shape)   # expect [N, 5, 4096]
print("acts   dtype:", d["acts"].dtype)   # expect float16
print("episode    :", d["episode"])
print("timestep   :", d["timestep"])
print("task_id    :", d["task_id"])
print("success    :", d["success"])

# sanity: no NaN/Inf
acts = d["acts"].astype(np.float32)
print("\nany NaN:", np.isnan(acts).any())
print("any Inf:", np.isinf(acts).any())
print("per-layer mean L2 norm:", np.linalg.norm(acts, axis=-1).mean(axis=0))

LAYER_NAMES = {0: "layer_0", 8: "layer_8", 16: "layer_16", 24: "layer_24", 31: "layer_31"}
layer_indices = m["layer_indices"]

for ts_idx in (0, 1):
    ep = d["episode"][ts_idx]
    step = d["timestep"][ts_idx]
    print(f"\n{'='*60}")
    print(f"Timestep index {ts_idx}  |  episode={ep}  step={step}")
    print(f"{'='*60}")
    for li, layer in enumerate(layer_indices):
        vec = acts[ts_idx, li]          # [4096]
        top5_idx = np.argsort(np.abs(vec))[-5:][::-1]
        print(f"\n  Layer {layer:2d}:")
        print(f"    min={vec.min():.4f}  max={vec.max():.4f}  "
              f"mean={vec.mean():.4f}  std={vec.std():.4f}  "
              f"L2={np.linalg.norm(vec):.2f}")
        print(f"    top-5 |values| at dims: "
              + "  ".join(f"[{i}]={vec[i]:.3f}" for i in top5_idx))