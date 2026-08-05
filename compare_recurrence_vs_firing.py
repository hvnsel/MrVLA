"""Do cross-model recurrence and the paper's firing metrics identify the SAME features?

This joins the two halves of the study:
  * recurrence q_cross (this work's label-free generality signal), per feature, from
    run_recurrence.py output;
  * the paper's firing statistics (episode coverage, mean onsets), per feature, from
    the per-suite codes (extract_codes_and_metrics.py: layer_NN.npz).

Same SAE, same feature indices, so the two are joined by index.  If recurrence and
coverage AGREE, recurrence validates the firing metric; if they DISAGREE, recurrence
is a different axis and the firing metric selects different features -- the paper's
central claim (firing = activity, not generality) made concrete.

Reports, per (layer, target model):
  * Spearman(q_cross, coverage) and Spearman(q_cross, mean_onsets)
  * top-K overlap: of the K most-recurrent vs K highest-coverage features, how many
    are shared (Jaccard).  Low overlap = the two metrics point at different features.

Pure numpy on saved arrays.

Usage
-----
python compare_recurrence_vs_firing.py \
    --rec-dir ./recurrence_v1 \
    --codes-map goal=./libero_goal_demos/codes_v4,spatial=...,object=...,libero10=... \
    --layers 0,8,16,24,31 --topk 100
"""

from __future__ import annotations

import argparse
import os

import numpy as np


def _spearman(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3:
        return float("nan")
    ra = np.argsort(np.argsort(a[m])).astype(float)
    rb = np.argsort(np.argsort(b[m])).astype(float)
    ra -= ra.mean(); rb -= rb.mean()
    d = np.sqrt((ra**2).sum() * (rb**2).sum())
    return float((ra*rb).sum()/d) if d > 0 else float("nan")


def _topk_overlap(a, b, active, k):
    """Jaccard overlap of the top-k active features by score a vs by score b."""
    idx = np.where(active)[0]
    ta = set(idx[np.argsort(-a[idx])[:k]])
    tb = set(idx[np.argsort(-b[idx])[:k]])
    inter = len(ta & tb)
    union = len(ta | tb)
    return inter, inter / union if union else float("nan")


def parse_map(spec):
    out = {}
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        k, v = part.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--rec-dir", required=True)
    p.add_argument("--codes-map", required=True, help="key=codes_dir pairs (layer_NN.npz)")
    p.add_argument("--layers", default="0,8,16,24,31")
    p.add_argument("--topk", type=int, default=100)
    args = p.parse_args()

    codes_map = parse_map(args.codes_map)
    layers = [int(x) for x in args.layers.split(",") if x.strip()]

    ov_hdr = f"top{args.topk} ov"
    print(f"{'layer/target':22s} {'rho(q,cover)':>13} {'rho(q,onset)':>13} "
          f"{ov_hdr:>10} {'jacc':>6}")
    print("-" * 70)
    for layer in layers:
        for key, cdir in codes_map.items():
            rec_path = os.path.join(args.rec_dir, f"layer_{layer:02d}_target_{key}.npz")
            code_path = os.path.join(cdir, f"layer_{layer:02d}.npz")
            if not (os.path.exists(rec_path) and os.path.exists(code_path)):
                continue
            rec = np.load(rec_path)
            cod = np.load(code_path)
            q = rec["q_cross"].astype(np.float64)
            active = rec["is_active"].astype(bool)
            cover = cod["coverage"].astype(np.float64)
            onset = cod["mean_onsets"].astype(np.float64)
            if len(cover) != len(q):
                print(f"L{layer:02d}/{key}: length mismatch "
                      f"(rec {len(q)} vs codes {len(cover)}) -- skipping")
                continue
            rho_c = _spearman(q[active], cover[active])
            rho_o = _spearman(q[active], onset[active])
            n_ov, jac = _topk_overlap(q, cover, active, args.topk)
            print(f"L{layer:02d}/{key:12s} {rho_c:13.3f} {rho_o:13.3f} "
                  f"{n_ov:10d} {jac:6.3f}")
    print("\nLow rho and low Jaccard => recurrence and the paper's coverage identify")
    print("DIFFERENT features: firing-based generality selects a different set than a")
    print("model-universal signal. High => recurrence largely re-finds the busy features.")


if __name__ == "__main__":
    main()
