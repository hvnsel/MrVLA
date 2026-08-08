"""Path A / §3.2b: turn feature exemplar locations into image contact sheets.

Reads identify_features.py's exemplars.json (feature -> top decisions by |phi| and by
activation, each a (task, episode, timestep) triple), locates the corresponding LIBERO demo
frames, applies the exact OpenVLA image transform, and writes one contact sheet PNG per
feature so we can eyeball what a "general" vs a "specialist" feature responds to.

Frame mapping. A1 stored `episode` as a running counter that increments once per demo, and
within a task demos are processed in sorted(demo_keys) order (mrvla/libero_demos.py). So the
demo for a decision is the one at rank = episode - min_episode(task) within that task's sorted
demo keys. We recover min_episode(task) by scanning the A1 shards' (task_id, episode) arrays,
which makes the mapping independent of the A1 run's max-demos/max-steps settings.

Usage
-----
python capture_feature_frames.py \
    --acts-dir /work/.../ACT_ACTION/goal \
    --exemplars /work/.../FEATURES/goal_k100/exemplars.json \
    --task-suite libero_goal \
    --rank-by phi \
    --out /work/.../FEATURES/goal_k100/frames
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# NOTE: mrvla.libero_demos (and h5py) are imported lazily inside main(), so this module's
# pure helpers (min_episode_per_task) are importable/testable without the LIBERO/h5py stack.


def min_episode_per_task(acts_dir: str) -> dict:
    """Smallest global episode index seen for each task_id across the A1 shards."""
    out: dict[int, int] = {}
    for sp in sorted(glob.glob(os.path.join(acts_dir, "shard_*.npz"))):
        dd = np.load(sp)
        task = dd["task_id"].astype(np.int64); ep = dd["episode"].astype(np.int64)
        for t in np.unique(task):
            m = int(ep[task == t].min())
            out[int(t)] = min(out.get(int(t), m), m)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--acts-dir", required=True, help="A1 output (for the episode->demo map)")
    p.add_argument("--exemplars", required=True, help="identify_features.py exemplars.json")
    p.add_argument("--task-suite", required=True, help="e.g. libero_goal")
    p.add_argument("--rank-by", choices=["phi", "activation"], default="phi",
                   help="which exemplar list to render (default: causal, |phi|)")
    p.add_argument("--image-key", default=None)
    p.add_argument("--cols", type=int, default=4, help="columns for the OVERALL (compact) layout")
    p.add_argument("--layout", choices=["auto", "per-task", "overall"], default="auto",
                   help="auto = per-task grid (rows=tasks) for general features, compact grid "
                        "for specialists; or force one for all features")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    with open(args.exemplars) as f:
        ex = json.load(f)
    key = "top_by_phi" if args.rank_by == "phi" else "top_by_activation"

    import h5py
    from mrvla.libero_demos import _demo_image, _find_demo_file, _resolve_image_key
    from libero.libero import benchmark, get_libero_path
    task_suite = benchmark.get_benchmark_dict()[args.task_suite]()
    datasets_root = get_libero_path("datasets")

    min_ep = min_episode_per_task(args.acts_dir)
    print(f"[cap] min episode per task: {min_ep}")

    # cache: task_id -> (demo_path, sorted_demo_keys, task_description)
    task_cache: dict[int, tuple] = {}

    def demo_info(task_id: int):
        if task_id not in task_cache:
            task = task_suite.get_task(task_id)
            path = _find_demo_file(datasets_root, args.task_suite, task)
            with h5py.File(path, "r") as fh:
                keys = sorted(fh["data"].keys(), key=lambda k: int(k.split("_")[-1]))
            task_cache[task_id] = (path, keys, task.language)
        return task_cache[task_id]

    def load_frame(task_id: int, episode: int, timestep: int):
        path, keys, _desc = demo_info(task_id)
        rank = episode - min_ep[task_id]
        if not (0 <= rank < len(keys)):
            return None
        with h5py.File(path, "r") as fh:
            obs = fh["data"][f"{keys[rank]}/obs"]
            ik = _resolve_image_key(obs, args.image_key)
            frames = obs[ik]
            if not (0 <= timestep < frames.shape[0]):
                return None
            frame = np.asarray(frames[timestep])
        return np.asarray(_demo_image(frame))

    n_tasks = task_suite.n_tasks
    per_task_cap = int(ex.get("per_task", 10))

    def suptitle(feat):
        return (f"feature {feat['feature']} · {feat['role']} · PR={feat['PR']:.2f} · "
                f"adj_breadth={feat['adjusted_breadth']:+.1f} · "
                f"fired in {feat.get('n_tasks_fired', '?')}/{n_tasks} tasks · ranked by {args.rank_by}")

    def draw(ax, rec):
        ax.set_xticks([]); ax.set_yticks([])
        if rec is None:
            return
        img = load_frame(rec["task"], rec["episode"], rec["timestep"])
        if img is not None:
            ax.imshow(img)
        ax.set_title(f"t{rec['timestep']} · z={rec['z']:.2f} φ={rec['phi']:+.2f}", fontsize=6)

    def render_per_task(feat):
        """Rows = every task in the suite; up to `per_task_cap` exemplars per row. A general
        feature fills most rows; a specialist leaves most rows empty -- both are informative."""
        pt = feat.get("per_task_by_phi", {})
        cols = per_task_cap
        fig, axes = plt.subplots(n_tasks, cols, figsize=(cols * 1.35, n_tasks * 1.5),
                                 squeeze=False)
        for g in range(n_tasks):
            recs = pt.get(str(g), [])
            for c in range(cols):
                ax = axes[g][c]
                draw(ax, recs[c] if c < len(recs) else None)
                if c == 0:
                    ax.set_ylabel(f"task {g}", fontsize=8, rotation=0, ha="right", va="center",
                                  labelpad=18)
        fig.suptitle(suptitle(feat), fontsize=11)
        fig.tight_layout(rect=[0.03, 0, 1, 0.985])
        return fig

    def render_overall(feat):
        recs = feat["top_by_phi" if args.rank_by == "phi" else "top_by_activation"]
        if not recs:
            return None
        n = len(recs); cols = min(args.cols, n); rows = (n + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.4, rows * 2.6), squeeze=False)
        axes = axes.ravel()
        for ax in axes:
            ax.set_xticks([]); ax.set_yticks([])
        for i, rec in enumerate(recs):
            draw(axes[i], rec)
            axes[i].set_title(f"task {rec['task']} · t{rec['timestep']} · "
                              f"z={rec['z']:.2f} φ={rec['phi']:+.2f}", fontsize=6)
        fig.suptitle(suptitle(feat), fontsize=11)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        return fig

    for feat in ex["features"]:
        if args.layout == "per-task":
            use_pt = True
        elif args.layout == "overall":
            use_pt = False
        else:  # auto
            use_pt = (feat["role"] == "general")
        fig = render_per_task(feat) if use_pt else render_overall(feat)
        if fig is None:
            print(f"[cap] feature {feat['feature']}: no exemplars, skipping")
            continue
        tag = "pertask" if use_pt else "overall"
        outp = os.path.join(args.out,
                            f"feat_{feat['feature']:05d}_{feat['role']}_{tag}_{args.rank_by}.png")
        fig.savefig(outp, dpi=120); plt.close(fig)
        print(f"[cap] wrote {outp}")

    print(f"[cap] done -> {args.out}")


if __name__ == "__main__":
    main()
