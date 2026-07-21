"""Map per-episode generality scores to training weights (+ control arms).

Consumes the per-episode scores written by episode_generality_variance.py
(layer_NN_episode_scores.npz; score of record: the coverage-floored,
temporally trimmed HARD score) and emits one weight vector per arm, every one
normalised so the MEAN weight is 1 — reweighting redistributes gradient mass
across episodes without shrinking the effective dataset.

Weight maps (increasing aggressiveness), all one family:
    w_i ∝ s_i ** gamma
        mild    gamma = 1   (linear in the score)
        medium  gamma = 2
        sharp   gamma = 4

Control arms, per gamma:
    uniform          w_i = 1                       (baseline; gamma-independent)
    inverted_<arm>   same power map applied to the reflected scores
                     s' = max(s) + min(s) - s      (upweights memorized —
                     should HURT if the generality story is right)
    random_<arm>     seeded permutation of the real arm's weights
                     (same marginal weight distribution, no association with
                     the episodes — does *any* reweighting help?)

Before normalisation, weights are bounded to [min_weight, max_weight] so a
near-zero score cannot silently delete an episode and a single episode cannot
dominate a batch.  (The subsequent mean-1 rescale can move weights slightly
past these bounds; they are sanity bounds on the map, not hard guarantees.)

Usage
-----
python mrvla/episode_weights.py \
    --scores-dir E:/libero_goal_demos/check1_episode_variance \
    --out-dir    E:/libero_goal_demos/episode_weights \
    --layers 8 \
    [--score-key score_hard_trimmed] [--seed 0] \
    [--min-weight 0.1] [--max-weight 10.0]
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np


GAMMAS = {"mild": 1.0, "medium": 2.0, "sharp": 4.0}


def power_weights(
    scores: np.ndarray,
    gamma: float,
    min_weight: float = 0.1,
    max_weight: float = 10.0,
) -> np.ndarray:
    """w ∝ s**gamma, bounded to [min_weight, max_weight], then mean-normalised.

    Scores are non-negative ratios; the mean-normalisation step makes the map
    scale-invariant in s, so only the *relative* spread of scores matters.
    """
    s = np.asarray(scores, dtype=np.float64)
    if (s < 0).any():
        raise ValueError("scores must be non-negative")
    w = s ** gamma
    mean = w.mean()
    if mean <= 0:
        # every score is exactly zero — degenerate; fall back to uniform
        return np.ones_like(s)
    w = w / mean                          # provisional mean-1
    w = np.clip(w, min_weight, max_weight)
    return w / w.mean()                   # re-normalise after clipping


def reflect(scores: np.ndarray) -> np.ndarray:
    """Order-reversing reflection: s' = max(s) + min(s) - s.

    Keeps the scores in the same range so the same power map applies.
    """
    s = np.asarray(scores, dtype=np.float64)
    return s.max() + s.min() - s


def effective_sample_size(w: np.ndarray) -> float:
    """ESS = (Σw)² / Σw² — #episodes an equally-weighted set would need to
    match the variance of this weighting.  Equals len(w) for uniform."""
    return float(w.sum() ** 2 / np.maximum((w ** 2).sum(), 1e-12))


def build_arms(
    scores: np.ndarray,
    seed: int = 0,
    min_weight: float = 0.1,
    max_weight: float = 10.0,
) -> dict[str, np.ndarray]:
    """Return {arm_name: weights [E]} for every arm, each mean-normalised."""
    rng = np.random.default_rng(seed)
    arms: dict[str, np.ndarray] = {
        "uniform": np.ones(len(scores), dtype=np.float64),
    }
    reflected = reflect(scores)
    for name, gamma in GAMMAS.items():
        real = power_weights(scores, gamma, min_weight, max_weight)
        arms[name] = real
        arms[f"inverted_{name}"] = power_weights(
            reflected, gamma, min_weight, max_weight)
        arms[f"random_{name}"] = rng.permutation(real)
    return arms


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scores-dir", required=True,
                    help="Dir with layer_NN_episode_scores.npz "
                         "(from episode_generality_variance.py)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--layers", default="8",
                    help="Comma-separated layer indices, e.g. '8' or '0,8,16,24,31'.")
    ap.add_argument("--score-key", default="score_hard_trimmed",
                    help="Which score column to weight on "
                         "(score_hard_trimmed | score_soft_trimmed | "
                         "score_hard | score_soft).")
    ap.add_argument("--seed", type=int, default=0,
                    help="Seed for the random-permutation control arms.")
    ap.add_argument("--min-weight", type=float, default=0.1)
    ap.add_argument("--max-weight", type=float, default=10.0)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    layers = [int(x) for x in args.layers.split(",") if x.strip() != ""]

    summary = {
        "scores_dir": args.scores_dir,
        "score_key": args.score_key,
        "gammas": GAMMAS,
        "seed": args.seed,
        "min_weight": args.min_weight,
        "max_weight": args.max_weight,
        "layers": {},
    }

    for layer_idx in layers:
        scores_path = os.path.join(
            args.scores_dir, f"layer_{layer_idx:02d}_episode_scores.npz")
        d = np.load(scores_path)
        episode = d["episode"].astype(np.int32)
        scores = d[args.score_key].astype(np.float64)
        E = len(episode)
        print(f"\n=== layer {layer_idx:02d} ===  episodes={E}  "
              f"score={args.score_key}  "
              f"mean={scores.mean():.4f} std={scores.std():.4f}", flush=True)

        arms = build_arms(scores, seed=args.seed,
                          min_weight=args.min_weight, max_weight=args.max_weight)

        layer_stats = {}
        out_npz = os.path.join(args.out_dir, f"layer_{layer_idx:02d}_weights.npz")
        payload = {"episode": episode,
                   "score": scores.astype(np.float32)}
        for name, w in arms.items():
            payload[name] = w.astype(np.float32)
            ess = effective_sample_size(w)
            layer_stats[name] = {
                "mean": float(w.mean()),
                "std": float(w.std()),
                "min": float(w.min()),
                "max": float(w.max()),
                "ess": ess,
                "ess_frac": ess / E,
            }
            print(f"  {name:<16} mean={w.mean():.4f}  std={w.std():.4f}  "
                  f"range=[{w.min():.3f}, {w.max():.3f}]  "
                  f"ESS={ess:.1f}/{E} ({ess/E:.2%})", flush=True)

        np.savez_compressed(out_npz, **payload)
        print(f"  wrote {out_npz}", flush=True)
        summary["layers"][f"layer_{layer_idx:02d}"] = {
            "n_episodes": E,
            "score_mean": float(scores.mean()),
            "score_std": float(scores.std()),
            "arms": layer_stats,
        }

    summary_path = os.path.join(args.out_dir, "weights_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[done] summary -> {summary_path}", flush=True)


if __name__ == "__main__":
    main()
