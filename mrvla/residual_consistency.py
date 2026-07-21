"""Step 3b — Cross-layer consistency of the confound-free episode residual.

The confound audit leaves each layer with a per-episode RESIDUAL (count-score
variance not explained by goal identity, length, mass, L0, or idle frames)
that is split-half reliable within its own layer.  This script asks the
question that decides whether that residual is a real property of the episode
or layer-local noise: do the residuals from DIFFERENT layers agree with each
other across episodes?

    consistent   -> the layers are measuring one common episode trait;
                    episode-level reweighting on the residual is defensible
    inconsistent -> each layer's "reliable residual" is idiosyncratic;
                    the episode level closes, move to timestep level

Reads the layer_NN_confound_audit_<mode>.npz files written by
confound_audit.py and reports pairwise Pearson/Spearman matrices for both the
residual (the diagnostic) and the raw score (context — raw scores can agree
merely by sharing the task-identity structure, so only the residual matrix
carries evidential weight).  Caveat: layer residuals are computed on the SAME
episodes, so a shared nuisance the audit did not model would also show up as
consistency; a positive result is strong evidence, not proof.

Usage
-----
python mrvla/residual_consistency.py \
    --audit-dir E:/libero_goal_demos/check3_confound_audit \
    --layers 0,8,24,31 \
    [--score-mode count]
"""

from __future__ import annotations

import argparse
import itertools
import json
import os

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from mrvla.confound_audit import pearson, spearman
except ImportError:  # run directly as `python mrvla/residual_consistency.py`
    from confound_audit import pearson, spearman


CONSISTENCY_ADVISORY = 0.3   # mean off-diagonal pearson above this -> consistent


def load_residuals(audit_dir: str, layers: list[int], mode: str):
    """Return (episode_ids [E], {layer: residual [E]}, {layer: score [E]})
    aligned on the episodes common to every layer file."""
    per_layer = {}
    for layer_idx in layers:
        path = os.path.join(
            audit_dir, f"layer_{layer_idx:02d}_confound_audit_{mode}.npz")
        d = np.load(path)
        per_layer[layer_idx] = (d["episode"].astype(np.int64),
                                d["score_residual"].astype(np.float64),
                                d["score"].astype(np.float64))

    common = None
    for ep, _, _ in per_layer.values():
        common = set(ep) if common is None else common & set(ep)
    common = np.array(sorted(common), dtype=np.int64)

    residuals, scores = {}, {}
    for layer_idx, (ep, resid, score) in per_layer.items():
        idx = {e: i for i, e in enumerate(ep)}
        sel = np.array([idx[e] for e in common])
        residuals[layer_idx] = resid[sel]
        scores[layer_idx] = score[sel]
    return common, residuals, scores


def correlation_matrices(vectors: dict[int, np.ndarray]):
    """Pairwise pearson/spearman across layers. Returns (layers, P, S)."""
    layers = sorted(vectors)
    n = len(layers)
    P = np.eye(n)
    S = np.eye(n)
    for (i, a), (j, b) in itertools.combinations(enumerate(layers), 2):
        P[i, j] = P[j, i] = pearson(vectors[a], vectors[b])
        S[i, j] = S[j, i] = spearman(vectors[a], vectors[b])
    return layers, P, S


def mean_off_diagonal(M: np.ndarray) -> float:
    n = M.shape[0]
    if n < 2:
        return 0.0
    mask = ~np.eye(n, dtype=bool)
    return float(M[mask].mean())


def plot_matrices(layers, P_resid, S_resid, P_raw, out_path: str, mode: str):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    panels = [
        (P_resid, "residual pearson (the diagnostic)"),
        (S_resid, "residual spearman"),
        (P_raw, "raw score pearson (context only)"),
    ]
    labels = [f"L{layer}" for layer in layers]
    for ax, (M, title) in zip(axes, panels):
        im = ax.imshow(M, vmin=-1, vmax=1, cmap="RdBu_r")
        ax.set_xticks(range(len(layers)))
        ax.set_yticks(range(len(layers)))
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)
        ax.set_title(f"{title}\nmean off-diag = {mean_off_diagonal(M):+.3f}",
                     fontsize=10)
        for i in range(len(layers)):
            for j in range(len(layers)):
                ax.text(j, i, f"{M[i, j]:+.2f}", ha="center", va="center",
                        fontsize=9,
                        color="white" if abs(M[i, j]) > 0.6 else "black")
    fig.colorbar(im, ax=axes, shrink=0.8)
    fig.suptitle(f"Cross-layer episode-residual consistency ({mode} score)",
                 fontsize=12)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--audit-dir",
                    default="E:/libero_goal_demos/check3_confound_audit")
    ap.add_argument("--layers", default="0,8,24,31",
                    help="Layers to compare (default skips the degenerate "
                         "constant-score layer 16).")
    ap.add_argument("--score-mode", default="count", choices=("mass", "count"))
    ap.add_argument("--out-dir", default=None,
                    help="Defaults to --audit-dir.")
    args = ap.parse_args()

    layers = [int(x) for x in args.layers.split(",") if x.strip() != ""]
    out_dir = args.out_dir or args.audit_dir
    os.makedirs(out_dir, exist_ok=True)

    episodes, residuals, scores = load_residuals(
        args.audit_dir, layers, args.score_mode)
    print(f"[consistency] {len(episodes)} common episodes across "
          f"layers {layers} ({args.score_mode} score)", flush=True)

    layer_order, P_resid, S_resid = correlation_matrices(residuals)
    _, P_raw, _ = correlation_matrices(scores)

    def _print_matrix(name, M):
        print(f"\n  {name}:", flush=True)
        header = "        " + "".join(f"L{layer:<7}" for layer in layer_order)
        print(header, flush=True)
        for i, layer in enumerate(layer_order):
            row = "".join(f"{M[i, j]:+.3f}  " for j in range(len(layer_order)))
            print(f"    L{layer:<3} {row}", flush=True)

    _print_matrix("residual pearson (diagnostic)", P_resid)
    _print_matrix("residual spearman", S_resid)
    _print_matrix("raw score pearson (context)", P_raw)

    mean_p = mean_off_diagonal(P_resid)
    mean_s = mean_off_diagonal(S_resid)
    consistent = mean_p >= CONSISTENCY_ADVISORY
    print(f"\n  mean off-diagonal residual r: pearson={mean_p:+.3f}  "
          f"spearman={mean_s:+.3f}", flush=True)
    print(f"  VERDICT: consistent={consistent} "
          f"(advisory threshold {CONSISTENCY_ADVISORY})", flush=True)

    out_png = os.path.join(out_dir,
                           f"residual_consistency_{args.score_mode}.png")
    plot_matrices(layer_order, P_resid, S_resid, P_raw, out_png,
                  args.score_mode)
    print(f"  wrote {out_png}", flush=True)

    out_json = os.path.join(out_dir,
                            f"residual_consistency_{args.score_mode}.json")
    with open(out_json, "w") as f:
        json.dump({
            "score_mode": args.score_mode,
            "layers": layer_order,
            "n_episodes": int(len(episodes)),
            "residual_pearson": P_resid.tolist(),
            "residual_spearman": S_resid.tolist(),
            "raw_pearson": P_raw.tolist(),
            "mean_offdiag_residual_pearson": mean_p,
            "mean_offdiag_residual_spearman": mean_s,
            "advisory_threshold": CONSISTENCY_ADVISORY,
            "consistent": bool(consistent),
        }, f, indent=2)
    print(f"  wrote {out_json}", flush=True)


if __name__ == "__main__":
    main()
