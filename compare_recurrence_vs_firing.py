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
    """Overlap of the top-k active features by score a vs by score b.

    Returns (n_intersect, jaccard, n_expected_by_chance).  The chance level matters:
    drawing two independent top-k sets from n active features overlaps k^2/n on
    average, so an observed overlap must be read against that, not against 0.
    """
    idx = np.where(active)[0]
    n = len(idx)
    kk = min(k, n)
    ta = set(idx[np.argsort(-a[idx])[:kk]])
    tb = set(idx[np.argsort(-b[idx])[:kk]])
    inter = len(ta & tb)
    union = len(ta | tb)
    exp = (kk * kk) / n if n else float("nan")
    return inter, (inter / union if union else float("nan")), exp


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
    p.add_argument("--gen-map", default=None,
                   help="Optional key=dir pairs holding layer_NN_structural.npz (or "
                        "layer_NN_generality.npz) with the paper's paper_prob_general / "
                        "prob_general. Enables the headline rho(q_cross, P_general).")
    p.add_argument("--layers", default="0,8,16,24,31")
    p.add_argument("--topk", type=int, default=100)
    args = p.parse_args()

    codes_map = parse_map(args.codes_map)
    gen_map = parse_map(args.gen_map) if args.gen_map else {}
    layers = [int(x) for x in args.layers.split(",") if x.strip()]

    def load_pgeneral(gdir, layer):
        """Return the paper's P(general) per feature, or None."""
        for name, key in ((f"layer_{layer:02d}_structural.npz", "paper_prob_general"),
                          (f"layer_{layer:02d}_generality.npz", "prob_general")):
            path = os.path.join(gdir, name)
            if os.path.exists(path):
                d = np.load(path)
                if key in d:
                    return d[key].astype(np.float64)
        return None

    ov_hdr = f"top{args.topk}ov"
    print(f"{'layer/target':20s} {'rho(q,Pgen)':>11} {'rho(q,cover)':>12} "
          f"{'rho(q,onset)':>12} {ov_hdr:>9} {'exp':>5} {'jacc':>6}")
    print("-" * 82)
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

            pgen = load_pgeneral(gen_map[key], layer) if key in gen_map else None
            if pgen is not None and len(pgen) == len(q):
                rho_p = _spearman(q[active], pgen[active])
                n_ov, jac, exp = _topk_overlap(q, pgen, active, args.topk)
                rho_p_s = f"{rho_p:11.3f}"
            else:
                rho_p_s = f"{'n/a':>11}"
                n_ov, jac, exp = _topk_overlap(q, cover, active, args.topk)

            print(f"L{layer:02d}/{key:14s} {rho_p_s} {rho_c:12.3f} {rho_o:12.3f} "
                  f"{n_ov:9d} {exp:5.1f} {jac:6.3f}")
    print("\nrho(q,Pgen): the headline -- q_cross vs the paper's P(general). Overlap is")
    print("  vs P(general) when --gen-map is given, else vs coverage.")
    print("'exp' is the top-K overlap expected BY CHANCE (k^2/n_active): read the")
    print("  observed overlap against it, not against 0.")
    print("rho ~ 0 and overlap ~ chance => the two metrics identify DIFFERENT features:")
    print("  firing-based generality selects a different set than a model-universal one.")
    print("rho < 0 => INVERSION: features the paper ranks most general recur LEAST.")


if __name__ == "__main__":
    main()
