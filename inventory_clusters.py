"""A4 Stage 4: do independently fine-tuned models carve the SAME inventory of causal roles?

The distributional counterpart to `inventory_recurrence.py`, and the primary route when
`action_space_geometry.py` reports a signature space too small for span-based matching -- in a
cramped space every subset spans nearly everything, so "which subspace covers this feature"
stops discriminating, while "where do the directions sit" still does.

THE QUESTION, ONE LEVEL UP. Path B matched features. Feature-level matching is bounded by SAE
non-identifiability (~60% seed reproducibility, §2.5), and its failure mode is not that roles
vanish but that they SPLIT: one model spends three features where another spends one. That is
invisible to a per-feature metric and visible in occupancy. So this asks whether the two models
partition the signature space into the same regions, and whether they weight those regions
differently.

THIS IS NOT THE RESCUE FOR PATH B -- THAT IS THE m-SWEEP. Worth stating plainly, because the
opposite is the natural assumption. Splitting comes in two kinds and they push the feature-level
metric in OPPOSITE directions:

  * SPAN-splitting -- fragments span the role without surrounding it, each with only a small
    component along it. Every fragment is a poor individual match, so `max_j cos` DROPS. This is
    the mechanism that could have produced "generals recur less", and `inventory_recurrence.py`
    is what detects it. Clustering does not: the fragments' centroid is not the role.
  * DUPLICATION-splitting -- fragments are noisy copies, each with a positive component along
    the role. Now `max_j cos` gets a best-of-N boost, so split features match BETTER, not worse
    (measured: with three noisy copies, general features score ~0.62 against ~0.54 for atomic
    specialists). Duplication therefore CANNOT explain Path B's finding, and clustering is what
    detects it, via occupancy.

So the two scripts are diagnostics for different mechanisms, and running both is what identifies
which one the real data shows. A feature-level deficit on general features plus a rising m-curve
means span-splitting; a feature-level surplus plus uneven occupancy means duplication.

THREE READINGS, EACH REPORTED:

  1. Do the inventories match?   Cluster each model's unit signatures, then assign centroids
     across models optimally. Reported against the same floors as the rest of A4 -- random
     decoders through the shared head, and the same-model different-seed ceiling -- because a
     raw centroid cosine means nothing on its own.

  2. Do they weight roles the same?   Occupancy is the share of the dictionary each model
     devotes to a matched role. "Same inventory, different multiplicities" is exactly what
     DUPLICATION-splitting looks like, and it predicts larger occupancy differences on
     general-dominated roles than on specialist ones -- a prediction that is independent of, and
     checkable against, the feature-level direction.

  3. Does the conclusion depend on k?   Sliced Wasserstein between the two signature clouds
     needs no clustering at all. If it agrees with the clustered answer, the result is not an
     artefact of choosing k; if it disagrees, the clustering is doing the work.

THE CONTRAST THAT CARRIES THE CLAIM. Per matched role, the mean confound-adjusted breadth of
its members. Path B found general FEATURES recur less. If general ROLES match as well as
specialist ones, then the dictionary does not recur but the inventory does -- and read together
with the m-sweep, that tells you whether the feature-level deficit was fragmentation or real.

Assignment is Hungarian, not greedy. Greedy lets several roles claim one popular centroid and
overstates agreement; both are printed, and the gap is itself diagnostic. This also discharges
the §3.1 step-3 commitment to report Hungarian at scale, with no scipy dependency.

Usage
-----
python inventory_clusters.py --target goal \
    --model goal=$B/ACT_ACTION_SAE/goal/sae --model spatial=$B/ACT_ACTION_SAE/spatial/sae \
    --model object=$B/ACT_ACTION_SAE/object/sae --model 10=$B/ACT_ACTION_SAE/10/sae \
    --head $B/ACT_ACTION/goal/head_constants.npz \
    --attr $B/ATTR/goal_k100/layer_31_attribution.npz \
    --seed-sae $B/ACT_ACTION_SAE/goal/sae_seed1 --k 8,16,32 --out $B/INVENTORY/goal_clusters
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

from mrvla.clustering import (
    cluster_occupancy, match_inventories, normalize_rows, sliced_wasserstein, spherical_kmeans,
)
from mrvla.inventory import random_signature_dictionary, signature_sharpness
from mrvla.stats import rankdata_average


def _ranks(x):
    """Tie-averaged ranks, centred. NOT argsort(argsort(x)): that breaks ties by array index,
    which matters here because per-channel causal mass is exactly zero for any feature that
    never fires at that slot, and base_rate is a count over a fixed denominator."""
    r = rankdata_average(x)
    return r - r.mean()


def spearman(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3:
        return float("nan")
    ra, rb = _ranks(a[m]), _ranks(b[m])
    den = np.sqrt((ra * ra).sum() * (rb * rb).sum())
    return float((ra * rb).sum() / den) if den > 0 else float("nan")


def signatures(W_dec, W_U_act, g, center=True):
    S = (W_dec * np.asarray(g, float)[None, :]) @ np.asarray(W_U_act, float).T
    return S - S.mean(axis=1, keepdims=True) if center else S


def load_decoder(sae_dir: str, layer: int):
    from run_attribution import load_sae
    _We, W_dec_t, _b, _k, ck = load_sae(sae_dir, layer)
    return W_dec_t.detach().float().cpu().numpy().astype(np.float64), ck


def cluster_breadth(labels: np.ndarray, k: int, adj: np.ndarray) -> np.ndarray:
    """[k]: mean confound-adjusted breadth of each role's member features (NaN if empty)."""
    out = np.full(k, np.nan)
    for j in range(k):
        m = (labels == j) & np.isfinite(adj)
        if m.any():
            out[j] = float(np.nanmean(adj[m]))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", action="append", required=True, help="'name=sae_dir'. Repeatable.")
    ap.add_argument("--target", required=True)
    ap.add_argument("--head", required=True)
    ap.add_argument("--attr", required=True, help="target's attribution npz (breadth ranking)")
    ap.add_argument("--seed-sae", default=None, help="second-seed SAE for the target: the CEILING")
    ap.add_argument("--base-sae", default=None, help="base-model SAE (inheritance control)")
    ap.add_argument("--layer", type=int, default=31)
    ap.add_argument("--k", default="8,16,32", help="comma-separated cluster counts to sweep")
    ap.add_argument("--n-null", type=int, default=3)
    ap.add_argument("--rng-seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    ks = [int(x) for x in args.k.split(",") if x.strip()]
    hc = np.load(args.head)
    W_U_act = hc["W_U_act"].astype(np.float64)
    g = hc["g"].astype(np.float64)

    models, cks = {}, {}
    for spec in args.model:
        name, sep, path = spec.partition("=")
        if not sep:
            name, path = os.path.basename(path or name), name
        models[name], cks[name] = load_decoder(path, args.layer)
        print(f"[clu] {name:10s} W_dec {models[name].shape}")
    if args.target not in models:
        raise SystemExit(f"--target {args.target!r} not among {list(models)}")
    others = [n for n in models if n != args.target]
    if not others:
        raise SystemExit("need at least one non-target model")

    d = models[args.target].shape[1]
    S_t = signatures(models[args.target], W_U_act, g)
    A_hat = normalize_rows(S_t)[0]
    sharp = signature_sharpness(S_t)

    from identify_features import adjusted_breadth
    A = np.load(args.attr)
    adj = adjusted_breadth(A["PR"].astype(np.float64), A["magnitude"].astype(np.float64),
                           A["base_rate"].astype(np.float64), A["is_active"].astype(bool))
    if adj.shape[0] != A_hat.shape[0]:
        raise SystemExit(f"attribution has {adj.shape[0]} features, SAE has {A_hat.shape[0]}")

    # signature clouds for every dictionary we will compare against
    clouds = {n: normalize_rows(signatures(models[n], W_U_act, g))[0] for n in others}
    rng = np.random.default_rng(args.rng_seed)
    randoms = [random_signature_dictionary(models[others[0]].shape[0], d, W_U_act, g, rng)
               for _ in range(max(1, args.n_null))]
    ceiling_cloud = None
    if args.seed_sae:
        W_seed, ck = load_decoder(args.seed_sae, args.layer)
        ceiling_cloud = normalize_rows(signatures(W_seed, W_U_act, g))[0]
        print(f"[clu] seed ceiling from {ck}")
    else:
        print("[clu] NO --seed-sae: without the same-model different-seed ceiling, centroid "
              "cosines\n[clu] have no scale. Report them as uncalibrated.")
    base_cloud = None
    if args.base_sae:
        W_base, _ = load_decoder(args.base_sae, args.layer)
        base_cloud = normalize_rows(signatures(W_base, W_U_act, g))[0]

    out: dict = {"target": args.target, "others": others, "layer": args.layer, "k_sweep": ks,
                 "per_k": {}}

    for k in ks:
        crng = np.random.default_rng(args.rng_seed)
        C_t, lab_t, _ = spherical_kmeans(A_hat, k, crng)
        occ_t = cluster_occupancy(lab_t, k)
        brd = cluster_breadth(lab_t, k, adj)

        entry = {"k": k, "target_occupancy": occ_t.tolist(),
                 "cluster_breadth": [float(v) for v in brd], "models": {}}

        print(f"\n=== k = {k} ===")
        print(f"  {'vs':12s} {'hungarian':>10s} {'greedy':>9s} {'reuse':>7s} "
              f"{'occ TVD':>9s} {'SW':>8s}")

        def compare(name, cloud, store=True):
            C_b, lab_b, _ = spherical_kmeans(cloud, k, np.random.default_rng(args.rng_seed))
            res = match_inventories(C_t, C_b)
            occ_b = cluster_occupancy(lab_b, k)
            hr = np.array([p[0] for p in res["hungarian_pairs"]])
            hcol = np.array([p[1] for p in res["hungarian_pairs"]])
            occ_diff = np.abs(occ_t[hr] - occ_b[hcol])
            tvd = float(0.5 * occ_diff.sum())
            sw = sliced_wasserstein(A_hat, cloud, np.random.default_rng(args.rng_seed))
            rec = {"hungarian_mean": res["hungarian_mean"], "greedy_mean": res["greedy_mean"],
                   "greedy_distinct_targets": res["greedy_distinct_targets"],
                   "occupancy_tvd": tvd, "sliced_wasserstein": sw,
                   "matched_similarity": res["hungarian_similarity"].tolist(),
                   "matched_occupancy_diff": occ_diff.tolist(),
                   "matched_cluster_breadth": [float(brd[i]) for i in hr]}
            print(f"  {name:12s} {res['hungarian_mean']:10.4f} {res['greedy_mean']:9.4f} "
                  f"{res['greedy_distinct_targets']:4d}/{k:<3d} {tvd:9.4f} {sw:8.4f}")
            if store:
                entry["models"][name] = rec
            return rec

        obs = [compare(n, clouds[n]) for n in others]
        null_recs = [compare(f"random{i}", R, store=False) for i, R in enumerate(randoms)]
        entry["random_floor"] = {
            "hungarian_mean": float(np.mean([r["hungarian_mean"] for r in null_recs])),
            "occupancy_tvd": float(np.mean([r["occupancy_tvd"] for r in null_recs])),
            "sliced_wasserstein": float(np.mean([r["sliced_wasserstein"] for r in null_recs]))}
        if ceiling_cloud is not None:
            entry["seed_ceiling"] = compare("seed(ceiling)", ceiling_cloud, store=False)
        if base_cloud is not None:
            entry["base_inheritance"] = compare("base", base_cloud, store=False)

        obs_h = float(np.mean([r["hungarian_mean"] for r in obs]))
        fl = entry["random_floor"]["hungarian_mean"]
        entry["observed_hungarian_mean"] = obs_h
        if "seed_ceiling" in entry:
            ce = entry["seed_ceiling"]["hungarian_mean"]
            ret = (obs_h - fl) / (ce - fl) if abs(ce - fl) > 1e-12 else float("nan")
            entry["chance_corrected_retention"] = ret
            print(f"  -> inventory match {obs_h:.4f}  floor {fl:.4f}  ceiling {ce:.4f}   "
                  f"chance-corrected retention {ret:+.3f}")
        else:
            print(f"  -> inventory match {obs_h:.4f}  floor {fl:.4f}  (no ceiling: uncalibrated)")

        # THE contrast: do general-dominated roles match as well as specialist ones?
        sims = np.mean([np.array(r["matched_similarity"]) for r in obs], axis=0)
        odiff = np.mean([np.array(r["matched_occupancy_diff"]) for r in obs], axis=0)
        cb = np.array(obs[0]["matched_cluster_breadth"], dtype=np.float64)
        r_match = spearman(cb, sims)
        r_occ = spearman(cb, odiff)
        entry["corr_cluster_breadth_vs_match"] = r_match
        entry["corr_cluster_breadth_vs_occupancy_diff"] = r_occ
        print(f"  corr(role breadth, match quality)     = {r_match:+.3f}   "
              "(Path B found this NEGATIVE at the feature level)")
        print(f"  corr(role breadth, occupancy difference) = {r_occ:+.3f}   "
              "(DUPLICATION-splitting predicts POSITIVE: general roles get different budgets)")
        out["per_k"][str(k)] = entry

    ks_done = [out["per_k"][str(k)] for k in ks]
    stable = float(np.std([e["observed_hungarian_mean"] for e in ks_done]))
    out["match_stability_across_k"] = stable
    print(f"\n[clu] inventory match across k = "
          + ", ".join(f"k{e['k']}:{e['observed_hungarian_mean']:.3f}" for e in ks_done)
          + f"   sd {stable:.4f}")
    print("[clu] A conclusion that moves with k is a conclusion about k. Cross-check it against")
    print("[clu] the sliced-Wasserstein column, which uses no clustering at all.")

    np.savez_compressed(os.path.join(args.out, f"layer_{args.layer:02d}_clusters.npz"),
                        adjusted_breadth=adj.astype(np.float32),
                        signature_sharpness=sharp.astype(np.float32))
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[clu] wrote {args.out}")


if __name__ == "__main__":
    main()
