"""Path B / visualisation: show a recurrent feature firing across independently trained models.

Given the re-derived recurrent features from identify_recurrent_features.py, this makes the
recurrence LITERALLY VISIBLE: for a target-model feature, find its best-matching feature in
every other model (by activation-pattern correlation on the shared probe), then lay out the
top-activating probe frames as a grid with ROWS = MODELS. If goal's feature 312 and spatial's
matched feature light up on the SAME probe frames, that is one feature rediscovered in a
separately trained model.

Reuses the shared probe (collect_shared_probe.py): probe_{model}.npz `acts` [N, L, H] and
probe_manifest.json `meta` (per-frame suite/task/demo/timestep) for frame recovery.

Usage
-----
python capture_recurrent_frames.py \
    --probe-dir ./activations/shared_probe_v1 \
    --sae-map "goal=$BASE/SAE/goal/sae,spatial=$BASE/SAE/spatial/sae,object=...,10=..." \
    --features ./RECURRENCE/recurrent_features_goal.json \
    --target goal --layer 31 --top-frames 8 \
    --out ./RECURRENCE/frames_goal
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

# NOTE: torch / SAE / LIBERO imported lazily inside main() so best_match (pure numpy) is
# importable and testable without the cluster stack.


def best_match(z_target_col: np.ndarray, Z_other: np.ndarray):
    """Feature in Z_other [N, F] whose activation pattern best correlates with z_target_col [N].

    Returns (best_index, best_corr). Centres both, guards zero-variance columns.
    """
    t = z_target_col - z_target_col.mean()
    Zc = Z_other - Z_other.mean(axis=0, keepdims=True)
    num = t @ Zc                                            # [F]
    den = np.linalg.norm(t) * np.linalg.norm(Zc, axis=0)    # [F]
    corr = np.divide(num, den, out=np.zeros_like(num), where=den > 0)
    k = int(np.argmax(corr))
    return k, float(corr[k])


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--probe-dir", required=True)
    p.add_argument("--sae-map", required=True, help="comma list model=sae_dir")
    p.add_argument("--features", required=True, help="identify_recurrent_features.py json")
    p.add_argument("--target", required=True)
    p.add_argument("--layer", type=int, default=31)
    p.add_argument("--per-suite", type=int, default=8,
                   help="top-activating frames to show PER SUITE per model, so a feature's "
                        "activation on NON-home suites is visible (default 8)")
    p.add_argument("--which", choices=["re_derived_recurrent", "inherited_recurrent"],
                   default="re_derived_recurrent")
    p.add_argument("--image-key", default=None)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import torch  # noqa: F401
    import h5py
    from mrvla.libero_demos import _demo_image, _find_demo_file, _resolve_image_key
    from run_attribution import load_sae, sae_encode_full

    os.makedirs(args.out, exist_ok=True)
    device = args.device if __import__("torch").cuda.is_available() else "cpu"

    sae_map = dict(kv.split("=", 1) for kv in args.sae_map.split(","))
    models = list(sae_map)
    if args.target not in models:
        raise ValueError(f"target {args.target} not in sae-map keys {models}")

    manifest = json.load(open(os.path.join(args.probe_dir, "probe_manifest.json")))
    layers = list(manifest["layers"])
    lp = layers.index(args.layer)
    meta = manifest.get("meta") or manifest["frames"]      # per-frame provenance (either key)

    # encode the shared probe through every model's SAE at the layer -> Z^m [N, F]
    Z = {}
    for m in models:
        acts = np.load(os.path.join(args.probe_dir, f"probe_{m}.npz"))["acts"][:, lp, :]
        W_enc, _Wd, b_pre, k, _ck = load_sae(sae_map[m], args.layer)
        z, _l2, _mu = sae_encode_full(W_enc, b_pre, k, acts.astype(np.float32), device)
        Z[m] = z
        print(f"[rec-frames] encoded probe through {m}: Z {z.shape}", flush=True)

    feats = json.load(open(args.features))[args.which]

    from libero.libero import benchmark
    bmap = {}   # suite -> task_suite object (cache)

    def frame_image(fr):
        suite = fr["suite"]
        if suite not in bmap:
            bmap[suite] = benchmark.get_benchmark_dict()[suite]()
        task = bmap[suite].get_task(int(fr["task_id"]))
        from libero.libero import get_libero_path
        path = _find_demo_file(get_libero_path("datasets"), suite, task)
        with h5py.File(path, "r") as fh:
            obs = fh["data"][f"{fr['demo']}/obs"]
            ik = _resolve_image_key(obs, args.image_key)
            arr = np.asarray(obs[ik][int(fr["timestep"])])
        return np.asarray(_demo_image(arr))

    # group probe frames by SUITE once, so each (model, suite) block shows that model's
    # feature's top activations WITHIN that suite -- revealing whether a feature fires only
    # on its home suite or generalises across suites (the whole point of this view).
    suites = sorted({fr["suite"] for fr in meta})
    suite_idx = {s: np.array([i for i, fr in enumerate(meta) if fr["suite"] == s])
                 for s in suites}
    nps = args.per_suite
    ncols = len(suites) * nps
    ACC = "#1f4e79"

    for entry in feats:
        jt = entry["feature"]
        # matched feature per model (target maps to itself)
        matched = {args.target: (jt, 1.0)}
        for m in models:
            if m == args.target:
                continue
            matched[m] = best_match(Z[args.target][:, jt], Z[m])

        fig, axes = plt.subplots(len(models), ncols,
                                 figsize=(ncols * 0.85, len(models) * 1.5), squeeze=False)
        for ri, m in enumerate(models):
            k, corr = matched[m]
            zcol = Z[m][:, k]
            ci = 0
            for s in suites:
                idx = suite_idx[s]
                order = idx[np.argsort(zcol[idx])[::-1][:nps]]   # top-nps WITHIN this suite
                for slot in range(nps):
                    ax = axes[ri][ci]; ax.set_xticks([]); ax.set_yticks([])
                    if slot < len(order):
                        fr = meta[int(order[slot])]
                        try:
                            ax.imshow(frame_image(fr))
                        except Exception:
                            pass
                        ax.set_title(f"z={zcol[int(order[slot])]:.1f}", fontsize=5)
                    # suite header over the first cell of each block, on the top row
                    if ri == 0 and slot == 0:
                        ax.text(0.0, 1.32, s.replace("libero_", ""), transform=ax.transAxes,
                                fontsize=8.5, color=ACC, fontweight="bold")
                    # model / matched-feature label on the very first cell of the row
                    if ci == 0:
                        ax.set_ylabel(f"{m}\nfeat {k}\nr={corr:.2f}", fontsize=7,
                                      rotation=0, ha="right", va="center", labelpad=22)
                    ci += 1
        fig.suptitle(f"recurrent feature: {args.target} feat {jt}  "
                     f"(q_cross={entry.get('q_cross', float('nan')):.2f}, "
                     f"inheritance={entry.get('inheritance', float('nan')):.2f}) — "
                     f"top {nps}/suite per model; does it fire OFF its home suite?", fontsize=10)
        fig.tight_layout(rect=[0.05, 0, 1, 0.95])
        outp = os.path.join(args.out, f"recur_{args.target}_feat{jt:05d}.png")
        fig.savefig(outp, dpi=120); plt.close(fig)
        print(f"[rec-frames] wrote {outp}")

    print(f"[rec-frames] done -> {args.out}")


if __name__ == "__main__":
    main()
