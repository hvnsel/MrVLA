"""Plot activation strength of general SAE features within a single episode.

For one chosen episode, produces a figure per layer where each line is one
"general" feature's activation magnitude across timesteps in that episode.

Usage
-----
python -m mrvla.plot_general_features \
    --codes-dir      E:/libero_goal_demos/codes_v3 \
    --generality-dir E:/libero_goal_demos/generality_v3 \
    --out-dir        E:/libero_goal_demos/plots_general \
    --episode 0          # episode id to plot (use --list to print available ids)
"""
from __future__ import annotations

import argparse
import glob
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_layer(codes_dir: str, generality_dir: str, layer_name: str):
    codes_path = os.path.join(codes_dir, f"{layer_name}.npz")
    gen_path   = os.path.join(generality_dir, f"{layer_name}_generality.npz")

    c = np.load(codes_path)
    g = np.load(gen_path)

    z          = c["z"].astype(np.float32)   # [N, F]
    episode    = c["episode"]                # [N]
    timestep   = c["timestep"]               # [N]
    is_general = g["is_general"].astype(bool)  # [F]
    task_id    = c["task_id"] if "task_id" in c.files else None

    return z, episode, timestep, task_id, is_general


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_layer_episode(
    layer_name: str,
    z: np.ndarray,
    episode: np.ndarray,
    timestep: np.ndarray,
    task_id: np.ndarray | None,
    is_general: np.ndarray,
    episode_id: int,
    out_dir: str,
):
    general_indices = np.where(is_general)[0]
    n_gen = len(general_indices)
    if n_gen == 0:
        print(f"  [{layer_name}] no general features - skipping")
        return

    mask = (episode == episode_id)
    if not mask.any():
        print(f"  [{layer_name}] episode {episode_id} not present - skipping")
        return

    ts = timestep[mask]
    order = np.argsort(ts)
    ts = ts[order]
    z_ep = z[mask][order][:, general_indices]   # [T, n_gen]

    tid = int(task_id[mask][0]) if task_id is not None else None
    title_extra = f" (task {tid})" if tid is not None else ""

    colors = cm.tab20(np.linspace(0, 1, max(n_gen, 1)))

    fig, ax = plt.subplots(figsize=(12, 6))
    fig.suptitle(
        f"{layer_name} - general feature activations | episode {episode_id}{title_extra}",
        fontsize=12,
    )

    for i, feat_idx in enumerate(general_indices):
        ax.plot(ts, z_ep[:, i], linewidth=1.2,
                color=colors[i], label=f"feat {feat_idx}")

    ax.set_xlabel("timestep within episode")
    ax.set_ylabel("activation magnitude")
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)

    ncol_legend = 1 if n_gen <= 25 else 2
    ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5),
              fontsize=7, ncol=ncol_legend, frameon=False)

    out_path = os.path.join(
        out_dir, f"{layer_name}_episode_{episode_id:04d}.png"
    )
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [{layer_name}] {n_gen} general features, T={len(ts)} -> {out_path}",
          flush=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--codes-dir",      required=True)
    p.add_argument("--generality-dir", required=True)
    p.add_argument("--out-dir",        required=True)
    p.add_argument("--episode", type=int, default=None,
                   help="Episode id to plot. Omit with --list to see available ids.")
    p.add_argument("--list", action="store_true",
                   help="List available episode ids (with task ids) and exit.")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    layer_files = sorted(glob.glob(os.path.join(args.codes_dir, "layer_*.npz")))
    if not layer_files:
        raise FileNotFoundError(f"No layer_*.npz in {args.codes_dir!r}")

    if args.list or args.episode is None:
        d = np.load(layer_files[0])
        eps = d["episode"]
        tasks = d["task_id"] if "task_id" in d.files else None
        unique_eps = np.unique(eps)
        print(f"[plot] available episodes: {len(unique_eps)}")
        for ep in unique_eps:
            mask = (eps == ep)
            T = int(mask.sum())
            tid = int(tasks[mask][0]) if tasks is not None else -1
            print(f"  episode {int(ep):4d}  task {tid:2d}  T={T}")
        if args.episode is None:
            print("\nRe-run with --episode <id> to generate plots.")
            return

    print(f"[plot] found {len(layer_files)} layers  episode={args.episode}", flush=True)
    for fpath in layer_files:
        layer_name = os.path.basename(fpath).replace(".npz", "")
        z, episode, timestep, task_id, is_general = load_layer(
            args.codes_dir, args.generality_dir, layer_name
        )
        plot_layer_episode(layer_name, z, episode, timestep, task_id,
                           is_general, args.episode, args.out_dir)

    print(f"\n[plot] done. figures in {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
