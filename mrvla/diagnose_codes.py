"""Quick diagnostic: verify z sparsity, activation distribution, and episode grouping.

Run this before re-running generality_classifier to understand what's broken.
"""
import numpy as np, glob, os, sys

codes_dir = sys.argv[1] if len(sys.argv) > 1 else "E:/libero_goal_demos/codes"
layer_file = sorted(glob.glob(os.path.join(codes_dir, "layer_*.npz")))[0]
print(f"Checking: {layer_file}\n")

d = np.load(layer_file)
z        = d["z"].astype(np.float32)   # [N, F]
episode  = d["episode"]
timestep = d["timestep"]

N, F = z.shape
k_empirical = (z > 0).sum(axis=1)

print("=== Z SPARSITY (should be ~100 nonzero per row if TopK k=100) ===")
print(f"  mean nonzero per row : {k_empirical.mean():.2f}")
print(f"  min/max nonzero      : {k_empirical.min()} / {k_empirical.max()}")
print(f"  global nonzero frac  : {(z > 0).mean():.4f}  (expected ~{100/F:.4f})")

print("\n=== ACTIVATION VALUE DISTRIBUTION (nonzero entries) ===")
nz = z[z > 0]
pcts = [50, 75, 90, 95, 99, 99.9]
vals = np.percentile(nz, pcts)
print(f"  count nonzero : {len(nz):,}")
print(f"  mean          : {nz.mean():.6f}")
print(f"  std           : {nz.std():.6f}")
for p, v in zip(pcts, vals):
    print(f"  p{p:<5}        : {v:.6f}")
print(f"  max           : {nz.max():.6f}")
print(f"  fraction > 0.1: {(nz > 0.1).mean():.4f}  (paper uses tau_on=0.1)")

print("\n=== EPISODE GROUPING ===")
unique_eps, counts = np.unique(episode, return_counts=True)
print(f"  unique episodes  : {len(unique_eps)}")
print(f"  episode lengths  : mean={counts.mean():.1f}  min={counts.min()}  max={counts.max()}")
print(f"  timestep range   : {timestep.min()} – {timestep.max()}")
print(f"  sample episodes  : {unique_eps[:5]} ... {unique_eps[-5:]}")

# Expected coverage from pure chance at this sparsity
k_mean = k_empirical.mean()
ep_len = counts.mean()
p_fire_once = 1 - (1 - k_mean/F) ** ep_len
print(f"\n=== COVERAGE (theoretical if random) ===")
print(f"  if uniform rand   : P(fire>=1 per episode) = {p_fire_once:.4f}")
actual_coverage = d["coverage"]
print(f"  actual mean cov   : {actual_coverage.mean():.4f}")
print(f"  actual pct > 0.01 : {(actual_coverage > 0.01).mean():.4f}")
print(f"  actual pct > 0.10 : {(actual_coverage > 0.10).mean():.4f}")
print(f"  actual pct > 0.50 : {(actual_coverage > 0.50).mean():.4f}")
print(f"  actual pct > 0.90 : {(actual_coverage > 0.90).mean():.4f}")

# Coverage with tau_on = 0.1
print("\n=== COVERAGE IF tau_on = 0.1 APPLIED ===")
active_01 = (z > 0.1)
eps_unique = unique_eps
fired_in_ep = np.zeros(F, dtype=np.float32)
for ep in eps_unique:
    mask = (episode == ep)
    fired_in_ep += active_01[mask].any(axis=0).astype(np.float32)
cov_01 = fired_in_ep / len(eps_unique)
print(f"  coverage mean  : {cov_01.mean():.4f}")
print(f"  pct > 0.50     : {(cov_01 > 0.5).mean():.4f}")
print(f"  pct > 0.10     : {(cov_01 > 0.1).mean():.4f}")
print(f"  pct == 0 (dead): {(cov_01 == 0).mean():.4f}")
