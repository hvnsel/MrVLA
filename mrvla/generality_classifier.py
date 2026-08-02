"""Faithful implementation of the generality metrics + classifier of

    Swann, McGranahan, Buurmeijer, Kennedy, Schwager,
    "Sparse Autoencoders Reveal Interpretable and Steerable Features in
    VLA Models", arXiv:2603.19183 (Stanford, 2026), Sections 3.2-3.3.

Every definition below is transcribed from the paper's equations rather than
reconstructed from prose.  Equation numbers refer to the paper.

Notation.  f_j(x_t^(e)) is the activation coefficient of SAE feature j at
timestep t of episode e; T^(e) is the length of episode e; E is the set of all
episodes; and

    E+_j = { e : exists t with f_j(x_t^(e)) > 0 }                        (p.5)

is the set of episodes in which feature j fires at least once.  NOTE that E+
is defined by f > 0, NOT by f > tau_on: coverage counts any nonzero
activation, while the onset state machine uses the threshold.  All four
metrics average over E+_j only.

    Episode coverage            c_j       = |E+_j| / |E|                  (4)

    Onset state machine         s_t = 1            if f_j(x_t) > tau_on
                                s_t = 0            if f_j(x_t) == 0       (5)
                                s_t = s_{t-1}      otherwise
                                s_0 = 0,  tau_on = 0.1

      The OFF trigger is EXACT ZERO, not a second threshold.  With a TopK SAE
      a feature is zero exactly when it drops out of the top-K, so the state
      is sticky: once on, it stays on through any nonzero dip.

    Per-episode onset count     o_j       = sum_t max(0, s_t - s_{t-1})    (6)
    Mean onset count            obar_j    = mean over E+_j of o_j          (7)

    Mean activation magnitude   abar_j    = mean over E+_j of
                                            max_t f_j(x_t^(e))            (8)

      This is the mean of per-episode PEAKS, not the mean of activations over
      active timesteps.

    Per-episode run length      r_j       = (1/o_j) sum_t s_t             (9)
    Relative run length         lbar_r,j  = mean over E+_j of
                                            r_j^(e) / T^(e)              (10)

    Classifier                  P(general | m) =
        sigma( b0 + b1*obar + b2*c + b3*abar + b4*lbar_r )               (11)

Fitted coefficients (paper Section 4.2).  One classifier per fine-tuning
dataset; the OpenVLA classifier reuses the LIBERO boundary because OpenVLA is
also fine-tuned on LIBERO.  Metrics are used UNNORMALISED so that a single
boundary applies across layers of the same model.

    LIBERO  b = (-4.20, 1.89, 1.80,  0.52, -0.36)   100%  LOO-CV on 30 labels
    DROID   b = (-1.78, 0.74, 2.36,  0.35, -1.04)   96.7% LOO-CV

The paper's fit used 30 hand-labelled features (15 general, 15 memorized) from
a single reference layer.

Reference values for validation (paper Table 2), OpenVLA / LIBERO-Goal:

    Layer 8                 1775 active features,  8 general  (99.55% memorized)
    LM avg (0,8,16,24,31)   9389 active features, 42 general  (99.55% memorized)

Note the denominator is the number of ACTIVE features (those firing at least
once), not the full dictionary width.  This module reports both.

Usage
-----
python mrvla/generality_classifier.py \
    --codes-dir E:/libero_goal_demos/codes_v4 \
    --out-dir   E:/libero_goal_demos/generality_v4 \
    --dataset   libero
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np


# ---------------------------------------------------------------------------
# Paper constants (Section 4.2)
# ---------------------------------------------------------------------------
BETA_LIBERO = dict(
    intercept=-4.20,
    mean_onsets=1.89,
    coverage=1.80,
    mean_act_magnitude=0.52,
    rel_run_length=-0.36,
)
BETA_DROID = dict(
    intercept=-1.78,
    mean_onsets=0.74,
    coverage=2.36,
    mean_act_magnitude=0.35,
    rel_run_length=-1.04,
)
BETAS = {"libero": BETA_LIBERO, "droid": BETA_DROID}

TAU_ON = 0.1        # paper Eq. (5)
THRESHOLD = 0.5     # P(general) >= 0.5 -> general

# Paper Table 2, OpenVLA / LIBERO-Goal: (n_active, n_general)
PAPER_REFERENCE = {
    "layer_08": (1775, 8),
}


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


# ---------------------------------------------------------------------------
# Eq. (5): onset state machine
# ---------------------------------------------------------------------------
def onset_state(z_ep: np.ndarray, tau_on: float = TAU_ON) -> np.ndarray:
    """Binary firing state for one episode, per paper Eq. (5).

    z_ep : [T, F] activation coefficients for a single episode, ordered by
           timestep.  Must be non-negative (TopK + ReLU).

    Transitions, per feature independently:
        s_t = 1        if z > tau_on      (ON trigger)
        s_t = 0        if z == 0          (OFF trigger -- exact zero)
        s_t = s_{t-1}  otherwise          (hold)
    with s_0 = 0.

    Implementation: the state at t is decided by the most recent TRIGGERING
    timestep u <= t; if that trigger was an ON trigger the state is 1, if it
    was an OFF trigger the state is 0, and before any trigger the state is 0.

    Returns [T, F] bool.
    """
    z_ep = np.asarray(z_ep)
    if (z_ep < 0).any():
        raise ValueError("activations must be non-negative")
    T, F = z_ep.shape
    on = z_ep > tau_on          # ON trigger
    off = z_ep <= 0.0           # OFF trigger (exact zero)
    trig = on | off
    idx = np.where(trig, np.arange(T)[:, None], -1)
    idx = np.maximum.accumulate(idx, axis=0)      # last triggering index
    cols = np.arange(F)[None, :]
    return np.where(idx >= 0, on[np.maximum(idx, 0), cols], False)


# ---------------------------------------------------------------------------
# Eqs. (4), (6)-(10): per-feature metrics
# ---------------------------------------------------------------------------
def compute_metrics(
    z: np.ndarray,
    episode: np.ndarray,
    timestep: np.ndarray,
    tau_on: float = TAU_ON,
) -> dict:
    """Compute the four classifier inputs exactly as defined in the paper.

    z        : [N, F] activation coefficients (non-negative)
    episode  : [N]    episode id per row
    timestep : [N]    timestep within episode per row

    Returns a dict with, per feature [F]:
        coverage        c        Eq. (4)   -- over episodes with any f > 0
        mean_onsets     obar     Eq. (7)
        mean_act_mag    abar     Eq. (8)   -- mean of per-episode PEAKS
        rel_run_length  lbar_r   Eq. (10)
        mean_run_length          Eq. (9) averaged over E+ (not a classifier
                                 input; reported for diagnostics)
    plus dataset-level diagnostics:
        n_episodes, ep_mean_len, is_active, n_active,
        n_active_no_onset  -- per feature, the count of episodes where the
                              feature is nonzero but never exceeds tau_on.
                              The paper asserts obar >= 1 for any feature with
                              c > 0; that holds only if this count is 0, so we
                              measure it rather than assume it.
    """
    z = np.asarray(z, dtype=np.float32)
    F = z.shape[1]
    ep_ids = np.unique(episode)
    E = len(ep_ids)

    n_eps_active = np.zeros(F, dtype=np.int64)
    n_active_no_onset = np.zeros(F, dtype=np.int64)
    sum_onsets = np.zeros(F, dtype=np.float64)
    sum_peak = np.zeros(F, dtype=np.float64)
    sum_run = np.zeros(F, dtype=np.float64)
    sum_rel = np.zeros(F, dtype=np.float64)
    ep_lengths = np.zeros(E, dtype=np.int64)

    for e_i, ep in enumerate(ep_ids):
        mask = episode == ep
        order = np.argsort(timestep[mask], kind="stable")
        z_ep = z[mask][order]                       # [T, F]
        T = z_ep.shape[0]
        ep_lengths[e_i] = T

        # E+ membership: any nonzero activation in this episode (paper p.5)
        peak = z_ep.max(axis=0)                     # [F]
        in_Eplus = peak > 0.0

        s = onset_state(z_ep, tau_on)               # [T, F]
        s_prev = np.vstack([np.zeros((1, F), dtype=bool), s[:-1]])
        onsets = (s & ~s_prev).sum(axis=0).astype(np.float64)   # Eq. (6)
        n_on = s.sum(axis=0).astype(np.float64)

        with np.errstate(divide="ignore", invalid="ignore"):
            run = np.where(onsets > 0, n_on / np.maximum(onsets, 1e-12), 0.0)  # Eq. (9)

        n_eps_active += in_Eplus
        n_active_no_onset += in_Eplus & (onsets == 0)
        sum_onsets += np.where(in_Eplus, onsets, 0.0)
        sum_peak += np.where(in_Eplus, peak, 0.0)
        sum_run += np.where(in_Eplus, run, 0.0)
        sum_rel += np.where(in_Eplus, run / T, 0.0)             # Eq. (10)

    denom = np.maximum(n_eps_active, 1).astype(np.float64)
    return {
        "coverage": (n_eps_active / E).astype(np.float32),           # Eq. (4)
        "mean_onsets": (sum_onsets / denom).astype(np.float32),      # Eq. (7)
        "mean_act_mag": (sum_peak / denom).astype(np.float32),       # Eq. (8)
        "rel_run_length": (sum_rel / denom).astype(np.float32),      # Eq. (10)
        "mean_run_length": (sum_run / denom).astype(np.float32),     # Eq. (9)
        "is_active": n_eps_active > 0,
        "n_active": int((n_eps_active > 0).sum()),
        "n_active_no_onset": n_active_no_onset,
        "n_episodes": E,
        "ep_mean_len": float(ep_lengths.mean()),
        "tau_on": float(tau_on),
    }


# ---------------------------------------------------------------------------
# Eq. (11): classifier
# ---------------------------------------------------------------------------
def classify_features(
    coverage: np.ndarray,
    mean_onsets: np.ndarray,
    mean_act_mag: np.ndarray,
    rel_run_length: np.ndarray,
    beta: dict | None = None,
    is_active: np.ndarray | None = None,
    verbose: bool = True,
) -> dict:
    """Apply Eq. (11).  Metrics are used unnormalised, as in the paper.

    ``is_active`` (features firing at least once) is used only for reporting:
    the paper's Table 2 denominators are active-feature counts, so we report
    the general fraction over active features as well as over the whole
    dictionary.  Dead features have all-zero metrics and receive
    P = sigma(beta_0), which is below threshold for both fitted models.
    """
    beta = BETA_LIBERO if beta is None else beta
    logit = (
        beta["intercept"]
        + beta["mean_onsets"] * mean_onsets
        + beta["coverage"] * coverage
        + beta["mean_act_magnitude"] * mean_act_mag
        + beta["rel_run_length"] * rel_run_length
    )
    prob = sigmoid(logit)
    is_general = prob >= THRESHOLD

    F = len(coverage)
    if is_active is None:
        is_active = np.ones(F, dtype=bool)
    n_active = int(is_active.sum())
    n_general = int(is_general.sum())
    n_general_active = int((is_general & is_active).sum())

    out = {
        "prob_general": prob.astype(np.float32),
        "is_general": is_general,
        "n_features": F,
        "n_active": n_active,
        "n_general": n_general,
        "n_general_active": n_general_active,
        "n_memorized_active": n_active - n_general_active,
        "frac_general_active": n_general_active / max(n_active, 1),
        "frac_memorized_active": 1.0 - n_general_active / max(n_active, 1),
    }
    if verbose:
        print(f"  features={F}  active={n_active}  "
              f"general={n_general_active} "
              f"({100*out['frac_general_active']:.2f}% of active)  "
              f"memorized={out['n_memorized_active']} "
              f"({100*out['frac_memorized_active']:.2f}%)")
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--codes-dir", required=True,
                   help="Directory of layer_NN.npz files from "
                        "extract_codes_and_metrics.py")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--dataset", default="libero", choices=("libero", "droid"),
                   help="Which fitted classifier to apply (paper Sec. 4.2). "
                        "OpenVLA fine-tuned on LIBERO uses the LIBERO boundary.")
    p.add_argument("--tau-on", type=float, default=TAU_ON)
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    beta = BETAS[args.dataset]

    layer_files = sorted(glob.glob(os.path.join(args.codes_dir, "layer_*.npz")))
    if not layer_files:
        raise FileNotFoundError(f"No layer_*.npz in {args.codes_dir!r}")
    print(f"[gen] {len(layer_files)} layers | classifier={args.dataset} "
          f"| beta={beta} | tau_on={args.tau_on}", flush=True)

    summary = {"beta": beta, "dataset": args.dataset, "tau_on": args.tau_on,
               "threshold": THRESHOLD, "reference": "arXiv:2603.19183 Sec 3.2-3.3",
               "layers": {}}

    for fpath in layer_files:
        layer_name = os.path.basename(fpath).replace(".npz", "")
        print(f"\n[gen] ====== {layer_name} ======", flush=True)
        d = np.load(fpath)
        z = d["z"].astype(np.float32)
        episode, timestep = d["episode"], d["timestep"]

        m = compute_metrics(z, episode, timestep, tau_on=args.tau_on)
        res = classify_features(
            coverage=m["coverage"],
            mean_onsets=m["mean_onsets"],
            mean_act_mag=m["mean_act_mag"],
            rel_run_length=m["rel_run_length"],
            beta=beta,
            is_active=m["is_active"],
        )

        n_no_onset = int((m["n_active_no_onset"] > 0).sum())
        print(f"  episodes={m['n_episodes']}  mean_len={m['ep_mean_len']:.1f}  "
              f"features with >=1 nonzero-but-sub-threshold episode: "
              f"{n_no_onset}", flush=True)

        ref = PAPER_REFERENCE.get(layer_name)
        if ref is not None:
            print(f"  [paper Table 2] OpenVLA/LIBERO-Goal {layer_name}: "
                  f"{ref[1]} general / {ref[0]} active "
                  f"({100*ref[1]/ref[0]:.2f}%)  <-- reference", flush=True)

        out_path = os.path.join(args.out_dir, f"{layer_name}_generality.npz")
        np.savez_compressed(
            out_path,
            prob_general=res["prob_general"],
            is_general=res["is_general"].astype(np.uint8),
            is_active=m["is_active"].astype(np.uint8),
            coverage=m["coverage"],
            mean_onsets=m["mean_onsets"],
            mean_act_mag=m["mean_act_mag"],
            rel_run_length=m["rel_run_length"],
            mean_run_length=m["mean_run_length"],
            n_active_no_onset=m["n_active_no_onset"],
            tau_on=np.float32(args.tau_on),
        )
        print(f"  saved {out_path}", flush=True)

        order = np.argsort(res["prob_general"])[-10:][::-1]
        print(f"  {'feat':>6}  {'P(gen)':>7}  {'cover':>7}  {'onsets':>7}  "
              f"{'peak_a':>7}  {'rel_rl':>7}")
        for fi in order:
            print(f"  {fi:>6}  {res['prob_general'][fi]:>7.4f}  "
                  f"{m['coverage'][fi]:>7.4f}  {m['mean_onsets'][fi]:>7.3f}  "
                  f"{m['mean_act_mag'][fi]:>7.4f}  "
                  f"{m['rel_run_length'][fi]:>7.4f}", flush=True)

        summary["layers"][layer_name] = {
            "n_features": res["n_features"],
            "n_active": res["n_active"],
            "n_general_active": res["n_general_active"],
            "frac_general_active": res["frac_general_active"],
            "frac_memorized_active": res["frac_memorized_active"],
            "n_episodes": m["n_episodes"],
            "ep_mean_len": m["ep_mean_len"],
            "top10_general": order.tolist(),
            "paper_reference": ref,
        }

    with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[gen] === SUMMARY (general / active features) ===")
    for name, s in summary["layers"].items():
        line = (f"  {name}: {s['n_general_active']}/{s['n_active']} "
                f"({100*s['frac_general_active']:.2f}% general, "
                f"{100*s['frac_memorized_active']:.2f}% memorized)")
        if s["paper_reference"]:
            line += f"   [paper: {s['paper_reference'][1]}/{s['paper_reference'][0]}]"
        print(line)
    print(f"\n[gen] done -> {os.path.join(args.out_dir, 'summary.json')}", flush=True)


if __name__ == "__main__":
    main()
