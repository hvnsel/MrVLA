"""Check 1 — Episode-level generality-score variance.

For every LIBERO-Goal training trajectory we compute a single generality score and
ask: is there real spread across episodes (episode-level reweighting is viable) or a
tight spike (episode-level is dead, move down a level)?

Per timestep t (hidden state -> SAE sparse code z_t over F features):
    general_mass_t = sum_j z_t[j]      for j labelled general
    total_mass_t   = sum_j z_t[j]      for all active j
    ratio_t        = general_mass_t / total_mass_t

Per trajectory i:
    generality_score_i = mean_t(ratio_t)

Two labellings are plotted side by side:
    * hard : general_mass_t uses the binary is_general split (paper threshold P>=0.5)
    * soft : general_mass_t = sum_j prob_general[j] * z_t[j]   (probability-weighted)

We also plot the per-timestep ratio distribution (not just the per-episode mean) so
that a spike can be attributed to the label distribution rather than the averaging.

Everything is read from the v3 codes / generality artefacts on the E: drive; nothing
is recomputed.

Usage
-----
python mrvla/episode_generality_variance.py \
    --codes-dir      E:/libero_goal_demos/codes_v3 \
    --generality-dir E:/libero_goal_demos/generality_v3 \
    --out-dir        E:/libero_goal_demos/check1_episode_variance \
    --layers 0,8,16,24,31
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


EPS = 1e-8


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_layer(codes_dir: str, generality_dir: str, layer_idx: int):
    codes_path = os.path.join(codes_dir, f"layer_{layer_idx:02d}.npz")
    gen_path = os.path.join(generality_dir, f"layer_{layer_idx:02d}_generality.npz")

    c = np.load(codes_path)
    g = np.load(gen_path)

    z = c["z"].astype(np.float32)              # [N, F]
    episode = c["episode"].astype(np.int64)    # [N]
    timestep = c["timestep"].astype(np.int64)  # [N]

    is_general = g["is_general"].astype(bool)  # [F]
    prob_general = g["prob_general"].astype(np.float32)  # [F]

    return z, episode, timestep, is_general, prob_general


# ---------------------------------------------------------------------------
# Score computation
# ---------------------------------------------------------------------------
def per_timestep_ratios(z: np.ndarray, is_general: np.ndarray, prob_general: np.ndarray):
    """Return (ratio_hard [N], ratio_soft [N]) per timestep.

    z is non-negative (TopK + ReLU), so total_mass > 0 whenever any feature fires.
    """
    total_mass = z.sum(axis=1)                                  # [N]
    general_mass_hard = z[:, is_general].sum(axis=1)            # [N]
    general_mass_soft = z @ prob_general                        # [N]

    denom = np.maximum(total_mass, EPS)
    ratio_hard = general_mass_hard / denom
    ratio_soft = general_mass_soft / denom
    return ratio_hard, ratio_soft, total_mass, general_mass_hard, general_mass_soft


def per_episode_mean(ratio_t: np.ndarray, episode: np.ndarray):
    """mean(ratio_t) within each episode. Returns (episode_ids [E], scores [E])."""
    ep_ids = np.unique(episode)
    # bincount-based grouped mean, robust to non-contiguous ids via searchsorted
    idx = np.searchsorted(ep_ids, episode)
    sums = np.bincount(idx, weights=ratio_t, minlength=len(ep_ids))
    counts = np.bincount(idx, minlength=len(ep_ids)).astype(np.float64)
    return ep_ids, sums / np.maximum(counts, 1.0), counts


def describe(name: str, x: np.ndarray) -> dict:
    pct = np.percentile(x, [0, 1, 5, 25, 50, 75, 95, 99, 100])
    return {
        "name": name,
        "n": int(x.size),
        "mean": float(x.mean()),
        "std": float(x.std()),
        "cv": float(x.std() / (abs(x.mean()) + EPS)),
        "min": float(pct[0]),
        "p1": float(pct[1]),
        "p5": float(pct[2]),
        "p25": float(pct[3]),
        "p50": float(pct[4]),
        "p75": float(pct[5]),
        "p95": float(pct[6]),
        "p99": float(pct[7]),
        "max": float(pct[8]),
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_layer(
    layer_idx: int,
    ep_hard: np.ndarray,
    ep_soft: np.ndarray,
    ts_hard: np.ndarray,
    ts_soft: np.ndarray,
    meta: dict,
    out_path: str,
):
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    def _hist(ax, data, title, color):
        ax.hist(data, bins=60, color=color, edgecolor="black", linewidth=0.3)
        ax.axvline(data.mean(), color="red", linestyle="--", linewidth=1.2,
                   label=f"mean={data.mean():.3f}")
        ax.axvline(np.median(data), color="orange", linestyle=":", linewidth=1.2,
                   label=f"median={np.median(data):.3f}")
        ax.set_title(f"{title}\n(std={data.std():.4f}, CV={data.std()/(abs(data.mean())+EPS):.3f})",
                     fontsize=10)
        ax.set_xlabel("generality score")
        ax.set_ylabel("count")
        ax.legend(fontsize=8)

    _hist(axes[0, 0], ep_hard, f"per-EPISODE  hard  (n={ep_hard.size})", "#4c72b0")
    _hist(axes[0, 1], ep_soft, f"per-EPISODE  soft  (n={ep_soft.size})", "#55a868")
    _hist(axes[1, 0], ts_hard, f"per-TIMESTEP hard  (n={ts_hard.size})", "#8fa8d4")
    _hist(axes[1, 1], ts_soft, f"per-TIMESTEP soft  (n={ts_soft.size})", "#9fd0b0")

    fig.suptitle(
        f"Layer {layer_idx:02d}  |  {meta['n_episodes']} episodes  |  "
        f"general features: {meta['n_general']}/{meta['n_features']}  |  "
        f"global general mass frac (hard/soft): "
        f"{meta['global_general_frac_hard']:.3f} / {meta['global_general_frac_soft']:.3f}",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--codes-dir", default="E:/libero_goal_demos/codes_v3")
    ap.add_argument("--generality-dir", default="E:/libero_goal_demos/generality_v3")
    ap.add_argument("--out-dir", default="E:/libero_goal_demos/check1_episode_variance")
    ap.add_argument("--layers", default="0,8,16,24,31")
    args = ap.parse_args()

    layers = [int(x) for x in args.layers.split(",") if x.strip() != ""]
    os.makedirs(args.out_dir, exist_ok=True)

    summary = {"codes_dir": args.codes_dir, "generality_dir": args.generality_dir,
               "layers": {}}

    for layer_idx in layers:
        print(f"\n=== layer {layer_idx:02d} ===", flush=True)
        z, episode, timestep, is_general, prob_general = load_layer(
            args.codes_dir, args.generality_dir, layer_idx)

        n_features = z.shape[1]
        n_general = int(is_general.sum())
        n_episodes = int(len(np.unique(episode)))
        print(f"  z={z.shape}  episodes={n_episodes}  "
              f"general_features={n_general}/{n_features}", flush=True)

        (ratio_hard_t, ratio_soft_t, total_mass,
         gmass_hard, gmass_soft) = per_timestep_ratios(z, is_general, prob_general)

        # global mass fraction (pooled over all timesteps, not the mean of ratios)
        tot = float(total_mass.sum())
        global_general_frac_hard = float(gmass_hard.sum() / (tot + EPS))
        global_general_frac_soft = float(gmass_soft.sum() / (tot + EPS))

        ep_ids, ep_hard, counts = per_episode_mean(ratio_hard_t, episode)
        _, ep_soft, _ = per_episode_mean(ratio_soft_t, episode)

        meta = {
            "n_features": n_features,
            "n_general": n_general,
            "n_episodes": n_episodes,
            "global_general_frac_hard": global_general_frac_hard,
            "global_general_frac_soft": global_general_frac_soft,
        }

        out_png = os.path.join(args.out_dir, f"layer_{layer_idx:02d}_episode_variance.png")
        plot_layer(layer_idx, ep_hard, ep_soft, ratio_hard_t, ratio_soft_t, meta, out_png)
        print(f"  wrote {out_png}", flush=True)

        # persist per-episode scores for downstream reweighting
        out_npz = os.path.join(args.out_dir, f"layer_{layer_idx:02d}_episode_scores.npz")
        np.savez_compressed(
            out_npz,
            episode=ep_ids.astype(np.int32),
            n_timesteps=counts.astype(np.int32),
            score_hard=ep_hard.astype(np.float32),
            score_soft=ep_soft.astype(np.float32),
        )

        stats = {
            "meta": meta,
            "episode_hard": describe("episode_hard", ep_hard),
            "episode_soft": describe("episode_soft", ep_soft),
            "timestep_hard": describe("timestep_hard", ratio_hard_t),
            "timestep_soft": describe("timestep_soft", ratio_soft_t),
        }
        summary["layers"][f"layer_{layer_idx:02d}"] = stats

        e = stats["episode_hard"]
        print(f"  per-episode HARD: mean={e['mean']:.4f} std={e['std']:.4f} "
              f"CV={e['cv']:.3f}  range=[{e['min']:.4f}, {e['max']:.4f}]  "
              f"p5..p95=[{e['p5']:.4f}, {e['p95']:.4f}]", flush=True)
        s = stats["episode_soft"]
        print(f"  per-episode SOFT: mean={s['mean']:.4f} std={s['std']:.4f} "
              f"CV={s['cv']:.3f}  range=[{s['min']:.4f}, {s['max']:.4f}]  "
              f"p5..p95=[{s['p5']:.4f}, {s['p95']:.4f}]", flush=True)

    summary_path = os.path.join(args.out_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[done] summary -> {summary_path}", flush=True)


if __name__ == "__main__":
    main()
