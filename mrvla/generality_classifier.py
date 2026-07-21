"""Apply the paper's logistic-regression generality classifier to SAE features.

Formula (arXiv 2603.19183, Appendix B):
    P(general | m) = sigmoid(β₀ + β₁*ō + β₂*c + β₃*ā + β₄*ℓ̄ᵣ)

    β₀ = -4.20   (intercept)
    β₁ =  1.89   (mean onset count, ō)
    β₂ =  1.80   (episode coverage, c)
    β₃ =  0.52   (mean activation magnitude when firing, ā)
    β₄ = -0.36   (relative run length, ℓ̄ᵣ = mean_run_length / mean_episode_length)

    Threshold: P >= 0.5 → "general", else "memorized"

All four metrics are recomputed from the sparse codes z stored in the
layer_NN.npz files produced by extract_codes_and_metrics.py, using a single
debounced firing state (two-threshold hysteresis: ON when z > tau_on, OFF when
z < tau_off).  Coverage, ō, ℓ̄ᵣ, and ā are all conditioned on that same state,
so the classifier inputs share one consistent definition of "fires":

  * onsets are counted on the debounced state — boundary flicker around
    tau_on no longer inflates ō or fragments runs
  * run length is episode-weighted and normalised per episode
    (ℓ̄ᵣ = mean over fired episodes of run_length_ep / T_ep)
  * ā is the mean of z over state-ON timesteps (not z > 0, not a
    re-thresholded sum)

Usage
-----
python generality_classifier.py \\
    --codes-dir E:/libero_goal_demos/codes \\
    --out-dir   E:/libero_goal_demos/generality
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np


# ---------------------------------------------------------------------------
# Classifier constants from the paper (fit on 30 hand-labeled features)
# ---------------------------------------------------------------------------
BETA = dict(
    intercept=-4.20,
    mean_onsets=1.89,
    coverage=1.80,
    mean_act_magnitude=0.52,
    rel_run_length=-0.36,
)
THRESHOLD = 0.5
TAU_ON = 0.1    # paper's activation threshold: OFF -> ON when z > TAU_ON
TAU_OFF = 0.05  # hysteresis release threshold: ON -> OFF when z < TAU_OFF


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


# ---------------------------------------------------------------------------
# Derived metric helpers
# ---------------------------------------------------------------------------
def hysteresis_state(
    z_ep: np.ndarray, tau_on: float = TAU_ON, tau_off: float = TAU_OFF
) -> np.ndarray:
    """Two-threshold (Schmitt-trigger) firing state for one episode.

    z_ep : [T, F] activations, ordered by timestep within a single episode.

    State machine per feature:
        OFF -> ON   when z > tau_on
        ON  -> OFF  when z < tau_off
        tau_off <= z <= tau_on : hold previous state (dead zone)
    Initial state is OFF.  With tau_off == tau_on this degenerates to the old
    single-threshold behaviour; tau_off < tau_on debounces boundary flicker so
    a feature hovering around tau_on no longer produces a fresh onset on every
    re-crossing.

    Returns [T, F] bool.
    """
    if tau_off > tau_on:
        raise ValueError(f"tau_off ({tau_off}) must be <= tau_on ({tau_on})")
    T = z_ep.shape[0]
    on_trig = z_ep > tau_on            # [T, F] forces state ON
    off_trig = z_ep < tau_off          # [T, F] forces state OFF
    trig = on_trig | off_trig
    # Forward-fill the most recent trigger: state[t] = on_trig at the last
    # triggering timestep <= t; OFF (-1 sentinel) before any trigger.
    idx = np.where(trig, np.arange(T)[:, None], -1)      # [T, F]
    idx = np.maximum.accumulate(idx, axis=0)             # last trigger index
    cols = np.arange(z_ep.shape[1])[None, :]
    state = np.where(idx >= 0, on_trig[np.maximum(idx, 0), cols], False)
    return state


def compute_metrics_hysteresis(
    z: np.ndarray,
    episode: np.ndarray,
    timestep: np.ndarray,
    tau_on: float = TAU_ON,
    tau_off: float = TAU_OFF,
) -> dict:
    """Compute all four classifier inputs from ONE debounced firing state.

    Every metric is derived from the same hysteresis state so their
    definitions cannot drift apart:

      coverage        : fraction of episodes with >= 1 onset of the state
      mean_onsets     : mean # of state onsets per episode, over episodes
                        where the feature fired at all
      mean_run_length : per-episode run length (active steps / onsets within
                        the episode), averaged over fired episodes —
                        episode-weighted, so flicker-heavy episodes no longer
                        dominate via their burst count
      rel_run_length  : per-episode run length normalised by THAT episode's
                        length, averaged over fired episodes
      mean_act_mag    : mean of z over timesteps where the state is ON
                        (includes dead-zone values sustained by hysteresis),
                        i.e. conditioned on the same firing definition as
                        every other metric
    """
    z = z.astype(np.float32, copy=False)
    F = z.shape[1]
    unique_eps, ep_counts = np.unique(episode, return_counts=True)
    n_eps = len(unique_eps)

    onsets_ef = np.zeros((n_eps, F), dtype=np.int32)
    active_steps_ef = np.zeros((n_eps, F), dtype=np.int32)
    run_len_ef = np.zeros((n_eps, F), dtype=np.float64)   # per-ep mean run length
    rel_run_ef = np.zeros((n_eps, F), dtype=np.float64)   # ... / T_ep
    act_sum_f = np.zeros(F, dtype=np.float64)             # Σ z over ON steps
    act_cnt_f = np.zeros(F, dtype=np.int64)               # # ON steps

    for e_idx, ep_id in enumerate(unique_eps):
        mask = (episode == ep_id)
        order = np.argsort(timestep[mask])
        z_ep = z[mask][order]                             # [T, F]
        T_ep = z_ep.shape[0]

        state = hysteresis_state(z_ep, tau_on, tau_off)   # [T, F]

        # Pad with OFF rows so onsets at t=0 count as complete bursts.
        padded = np.vstack([
            np.zeros((1, F), dtype=bool),
            state,
            np.zeros((1, F), dtype=bool),
        ])
        diff = padded[1:].astype(np.int8) - padded[:-1].astype(np.int8)
        onsets = (diff == 1).sum(axis=0)                  # [F]
        active_steps = state.sum(axis=0)                  # [F]

        onsets_ef[e_idx] = onsets
        active_steps_ef[e_idx] = active_steps
        with np.errstate(invalid="ignore"):
            rl = np.where(onsets > 0, active_steps / np.maximum(onsets, 1), 0.0)
        run_len_ef[e_idx] = rl
        rel_run_ef[e_idx] = rl / max(T_ep, 1)

        # ā accumulators: magnitude over ON steps (same state as above)
        act_sum_f += np.where(state, z_ep, 0.0).sum(axis=0)
        act_cnt_f += active_steps

    fired_in_ep = onsets_ef > 0
    n_fired = fired_in_ep.sum(axis=0)                     # [F]

    coverage = n_fired / n_eps
    mean_onsets = np.where(
        n_fired > 0,
        (onsets_ef * fired_in_ep).sum(axis=0) / np.maximum(n_fired, 1),
        0.0,
    )
    # Episode-weighted mean run length (each fired episode counts once).
    mean_run_length = np.where(
        n_fired > 0,
        (run_len_ef * fired_in_ep).sum(axis=0) / np.maximum(n_fired, 1),
        0.0,
    )
    # Per-episode-normalised relative run length: each episode's run length is
    # divided by that episode's own length before averaging, instead of
    # dividing a burst-weighted global run length by the global mean length.
    rel_run_length = np.where(
        n_fired > 0,
        (rel_run_ef * fired_in_ep).sum(axis=0) / np.maximum(n_fired, 1),
        0.0,
    )
    mean_act_mag = np.where(
        act_cnt_f > 0, act_sum_f / np.maximum(act_cnt_f, 1), 0.0
    )
    ep_mean_len = float(ep_counts.mean())

    return {
        "coverage": coverage.astype(np.float32),
        "mean_onsets": mean_onsets.astype(np.float32),
        "mean_run_length": mean_run_length.astype(np.float32),
        "rel_run_length": rel_run_length.astype(np.float32),
        "mean_act_mag": mean_act_mag.astype(np.float32),
        "ep_mean_len": ep_mean_len,
        "tau_on": float(tau_on),
        "tau_off": float(tau_off),
    }


def mean_episode_length(episode: np.ndarray) -> float:
    """Average number of timesteps per episode."""
    _, counts = np.unique(episode, return_counts=True)
    return float(counts.mean())


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------
def classify_features(
    coverage: np.ndarray,
    mean_onsets: np.ndarray,
    rel_run_length: np.ndarray,
    mean_act_mag: np.ndarray,
    ep_mean_len: float,
    verbose: bool = True,
) -> dict:
    """Return per-feature P(general) and binary label using paper's β values.

    ``rel_run_length`` is expected to already be per-episode normalised
    (see compute_metrics_hysteresis) — it is used as-is, NOT re-derived from a
    pooled run length divided by the global mean episode length.
    """
    logit = (
        BETA["intercept"]
        + BETA["mean_onsets"]     * mean_onsets
        + BETA["coverage"]        * coverage
        + BETA["mean_act_magnitude"] * mean_act_mag
        + BETA["rel_run_length"]  * rel_run_length
    )
    prob_general = sigmoid(logit)
    is_general = prob_general >= THRESHOLD

    n_features = len(coverage)
    n_general = int(is_general.sum())
    n_memorized = n_features - n_general
    frac_general = n_general / n_features

    if verbose:
        print(f"  features={n_features}  general={n_general} ({100*frac_general:.2f}%)  "
              f"memorized={n_memorized} ({100*(1-frac_general):.2f}%)")
        print(f"  paper target: ~0.45% general")
        print(f"  metric summary:")
        print(f"    coverage       mean={coverage.mean():.4f}  std={coverage.std():.4f}")
        print(f"    mean_onsets    mean={mean_onsets.mean():.4f}  std={mean_onsets.std():.4f}")
        print(f"    mean_act_mag   mean={mean_act_mag.mean():.4f}  std={mean_act_mag.std():.4f}")
        print(f"    rel_run_length mean={rel_run_length.mean():.4f}  std={rel_run_length.std():.4f}")
        print(f"    P(general)     mean={prob_general.mean():.4f}  "
              f"median={np.median(prob_general):.4f}  max={prob_general.max():.4f}")
        print(f"    mean episode length: {ep_mean_len:.1f} steps")

    return {
        "prob_general": prob_general.astype(np.float32),
        "is_general": is_general,
        "n_features": n_features,
        "n_general": n_general,
        "n_memorized": n_memorized,
        "frac_general": float(frac_general),
        "ep_mean_len": float(ep_mean_len),
        "rel_run_length": rel_run_length.astype(np.float32),
        "mean_act_mag": mean_act_mag.astype(np.float32),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--codes-dir", required=True,
                   help="Directory produced by extract_codes_and_metrics.py")
    p.add_argument("--out-dir", required=True,
                   help="Where to write per-layer results + summary.json")
    p.add_argument("--tau-on", type=float, default=TAU_ON,
                   help="Firing threshold: OFF -> ON when z > tau_on.")
    p.add_argument("--tau-off", type=float, default=TAU_OFF,
                   help="Hysteresis release: ON -> OFF when z < tau_off "
                        "(must be <= tau_on; equal reproduces single-threshold).")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Discover layers from npz files
    import glob
    layer_files = sorted(glob.glob(os.path.join(args.codes_dir, "layer_*.npz")))
    if not layer_files:
        raise FileNotFoundError(f"No layer_*.npz files in {args.codes_dir!r}")
    print(f"[gen] found {len(layer_files)} layer files", flush=True)
    print(f"[gen] β = {BETA}", flush=True)

    summary = {"beta": BETA, "threshold": THRESHOLD,
               "tau_on": args.tau_on, "tau_off": args.tau_off, "layers": {}}

    for fpath in layer_files:
        layer_name = os.path.basename(fpath).replace(".npz", "")
        print(f"\n[gen] ====== {layer_name} ======", flush=True)

        d = np.load(fpath)
        episode        = d["episode"]
        timestep       = d["timestep"]
        z              = d["z"]   # [N, F] float16

        F = z.shape[1]
        print(f"  recomputing metrics with tau_on={args.tau_on} "
              f"tau_off={args.tau_off} (hysteresis) ...", flush=True)
        m = compute_metrics_hysteresis(
            z.astype(np.float32), episode, timestep,
            tau_on=args.tau_on, tau_off=args.tau_off,
        )
        coverage = m["coverage"]
        mean_onsets = m["mean_onsets"]
        ep_mean_len = m["ep_mean_len"]

        result = classify_features(
            coverage=coverage,
            mean_onsets=mean_onsets,
            rel_run_length=m["rel_run_length"],
            mean_act_mag=m["mean_act_mag"],
            ep_mean_len=ep_mean_len,
        )

        # Save per-layer results
        out_path = os.path.join(args.out_dir, f"{layer_name}_generality.npz")
        np.savez_compressed(
            out_path,
            prob_general=result["prob_general"],
            is_general=result["is_general"].astype(np.uint8),
            coverage=coverage,
            mean_onsets=mean_onsets,
            mean_act_mag=result["mean_act_mag"],
            rel_run_length=result["rel_run_length"],
            mean_run_length=m["mean_run_length"],
            tau_on=np.float32(args.tau_on),
            tau_off=np.float32(args.tau_off),
        )
        print(f"  saved {out_path}", flush=True)

        # Top-10 most general features
        top10_idx = np.argsort(result["prob_general"])[-10:][::-1]
        print(f"  top-10 general features (by P(general)):", flush=True)
        print(f"  {'feat':>6}  {'P(gen)':>7}  {'coverage':>9}  "
              f"{'onsets':>7}  {'act_mag':>8}  {'rel_rl':>7}")
        for fi in top10_idx:
            print(f"  {fi:>6}  {result['prob_general'][fi]:>7.4f}  "
                  f"{coverage[fi]:>9.4f}  {mean_onsets[fi]:>7.3f}  "
                  f"{result['mean_act_mag'][fi]:>8.4f}  "
                  f"{result['rel_run_length'][fi]:>7.4f}")

        summary["layers"][layer_name] = {
            "n_features": result["n_features"],
            "n_general": result["n_general"],
            "n_memorized": result["n_memorized"],
            "frac_general": result["frac_general"],
            "ep_mean_len": result["ep_mean_len"],
            "top10_general_features": top10_idx.tolist(),
        }

    out_summary = os.path.join(args.out_dir, "summary.json")
    with open(out_summary, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[gen] === FINAL SUMMARY ===", flush=True)
    for layer_name, s in summary["layers"].items():
        print(f"  {layer_name}: {s['n_general']}/{s['n_features']} general "
              f"({100*s['frac_general']:.2f}%)")
    print(f"  paper target: ~0.45%")
    print(f"\n[gen] done. {out_summary}", flush=True)


if __name__ == "__main__":
    main()
