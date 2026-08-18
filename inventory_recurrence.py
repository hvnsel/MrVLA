"""A4 Stage 2-3: does model B's dictionary EXPRESS model A's causal roles, given m features?

Path B's finding is that the most causally general features are, if anything, the LEAST
cross-model recurrent (top-decile gaps -0.017 / -0.015 / -0.008 / -0.011, all small and
negative). That was measured one-to-one: `q_causal = max_j cos(S_A_i, S_B_j)`.

The reformulation, and why it is not just a bigger number. Sparse dictionaries split and merge
-- §2.5 puts same-model seed reproducibility at only ~60% -- and splitting is not uniform across
features. Broad, diffuse, high-usage general features are the ones expected to fragment; sharp
specialists stay atomic. One-to-one matching is structurally blind to fragmentation, so
splitting ALONE could have produced the entire "generals recur less" result with no difference
in recurrence underneath.

This sweeps m and reads the SLOPE, not the level:

    general slope steep, specialist flat  -> splitting explains the null; the dictionary does
                                             not recur but the inventory does
    both steep, similar                   -> coalitions help everything; the general/specialist
                                             gap is genuinely absent, better defended than before
    both flat, general still lower        -> the published finding survives a much harder test
    observed ~ random at every m          -> the signature space cannot support cross-model
                                             claims at all (Stage 0 will have predicted this)

Every one of those is publishable, which is the point of the design.

m=1 IS the existing metric (in absolute form -- see mrvla/inventory on the sign discontinuity),
so this extends the published result along a new axis rather than replacing it. The published
signed q is printed beside it.

NULLS, ALL m-MATCHED. cos rises with m mechanically, so an unmatched null manufactures a
positive result out of dimension counting. Three are run at every m:
  * random decoders through the shared head, at the same dictionary size (chance floor)
  * the same model under a different SAE seed (the CEILING -- the most any cross-model
    comparison could reach given non-identifiability), giving chance-corrected retention
  * the base model, if given (the shared-checkpoint inheritance control)

Usage
-----
python inventory_recurrence.py --target goal \
    --model goal=$B/ACT_ACTION_SAE/goal/sae --model spatial=$B/ACT_ACTION_SAE/spatial/sae \
    --model object=$B/ACT_ACTION_SAE/object/sae --model 10=$B/ACT_ACTION_SAE/10/sae \
    --head $B/ACT_ACTION/goal/head_constants.npz \
    --attr $B/ATTR/goal_k100/layer_31_attribution.npz \
    --seed-sae $B/ACT_ACTION_SAE/goal/sae_seed1 --layer 31 --m-max 8 \
    --out $B/INVENTORY/goal
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

from mrvla.inventory import (
    anti_aligned_fraction, chance_corrected_retention, normalize_rows, omp_curve,
    random_signature_dictionary, signature_sharpness, signed_best_match,
)


def load_decoder(sae_dir: str, layer: int) -> tuple[np.ndarray, str]:
    from run_attribution import load_sae
    _We, W_dec_t, _b, _k, ck = load_sae(sae_dir, layer)
    return W_dec_t.detach().float().cpu().numpy().astype(np.float64), ck


def signatures(W_dec, W_U_act, g, center=True):
    S = (W_dec * np.asarray(g, float)[None, :]) @ np.asarray(W_U_act, float).T
    if center:
        S = S - S.mean(axis=1, keepdims=True)
    return S


def decile_curves(cos: np.ndarray, score: np.ndarray, n_dec: int = 10) -> dict:
    """Mean cos-vs-m curve per decile of `score` (the confound-adjusted breadth ranking).

    The claim lives in the slopes across deciles, so this is the primary output. Features with
    a non-finite score are dropped rather than pooled into a decile they do not belong to.
    """
    m = np.isfinite(score)
    idx = np.where(m)[0]
    order = idx[np.argsort(score[idx])]
    chunks = np.array_split(order, n_dec)
    out = {"n_deciles": len(chunks), "curves": [], "sizes": [], "slopes": []}
    for ch in chunks:
        c = cos[ch].mean(axis=0)
        out["curves"].append([float(v) for v in c])
        out["sizes"].append(int(ch.size))
        out["slopes"].append(float(c[-1] - c[0]))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", action="append", required=True, help="'name=sae_dir'. Repeatable.")
    ap.add_argument("--target", required=True, help="which model's features are the targets")
    ap.add_argument("--head", required=True, help="head_constants.npz (shared across models)")
    ap.add_argument("--attr", required=True, help="target's layer_NN_attribution.npz")
    ap.add_argument("--seed-sae", default=None,
                    help="second-seed SAE for the target: the CEILING. Without it, "
                         "chance-corrected retention cannot be computed and raw cosines are "
                         "not interpretable across m.")
    ap.add_argument("--base-sae", default=None, help="base-model SAE (inheritance control)")
    ap.add_argument("--layer", type=int, default=31)
    ap.add_argument("--m-max", type=int, default=8)
    ap.add_argument("--n-restarts", type=int, default=4)
    ap.add_argument("--n-null", type=int, default=3, help="random-dictionary draws to average")
    ap.add_argument("--positive-only", action="store_true",
                    help="restrict selection to positively-correlated features. TopK codes are "
                         "non-negative, so this is the more faithful (weaker) bound.")
    ap.add_argument("--rng-seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    hc = np.load(args.head)
    W_U_act = hc["W_U_act"].astype(np.float64)
    g = hc["g"].astype(np.float64)

    models = {}
    for spec in args.model:
        name, sep, path = spec.partition("=")
        if not sep:
            name, path = os.path.basename(path or name), name
        W, ck = load_decoder(path, args.layer)
        models[name] = W
        print(f"[inv] {name:10s} W_dec {W.shape}  {ck}")
    if args.target not in models:
        raise SystemExit(f"--target {args.target!r} is not among --model names {list(models)}")

    S_t = signatures(models[args.target], W_U_act, g)
    A_hat, A_norms = normalize_rows(S_t)
    sharp = signature_sharpness(S_t)
    d = models[args.target].shape[1]
    F_t = A_hat.shape[0]
    rng = np.random.default_rng(args.rng_seed)
    M = args.m_max

    others = [n for n in models if n != args.target]
    if not others:
        raise SystemExit("need at least one non-target model")

    # ---------------- observed ----------------
    obs, published_q, anti = [], {}, {}
    for n in others:
        B_hat, _ = normalize_rows(signatures(models[n], W_U_act, g))
        cos, _ = omp_curve(A_hat, B_hat, M, args.positive_only, args.n_restarts)
        obs.append(cos)
        q_signed, _ = signed_best_match(A_hat, B_hat)
        published_q[n] = float(np.mean(q_signed))
        anti[n] = anti_aligned_fraction(A_hat, B_hat)
        print(f"[inv] observed vs {n:10s} m=1 {cos[:, 0].mean():.4f} ... "
              f"m={M} {cos[:, -1].mean():.4f}   (published signed q = {published_q[n]:+.4f}, "
              f"anti-aligned {anti[n]:.3f})")
    OBS = np.mean(np.stack(obs), axis=0)

    # ---------------- m-matched random floor ----------------
    null_runs = []
    for n in others:
        F_b = models[n].shape[0]
        for _ in range(max(1, args.n_null)):
            B_hat = random_signature_dictionary(F_b, d, W_U_act, g, rng)
            cos, _ = omp_curve(A_hat, B_hat, M, args.positive_only, args.n_restarts)
            null_runs.append(cos)
    NULL = np.mean(np.stack(null_runs), axis=0)
    print(f"[inv] random floor            m=1 {NULL[:, 0].mean():.4f} ... "
          f"m={M} {NULL[:, -1].mean():.4f}")

    # ---------------- ceiling: same model, different SAE seed ----------------
    CEIL = None
    if args.seed_sae:
        W_seed, ck = load_decoder(args.seed_sae, args.layer)
        B_hat, _ = normalize_rows(signatures(W_seed, W_U_act, g))
        CEIL, _ = omp_curve(A_hat, B_hat, M, args.positive_only, args.n_restarts)
        print(f"[inv] seed ceiling ({os.path.basename(ck)})  m=1 {CEIL[:, 0].mean():.4f} ... "
              f"m={M} {CEIL[:, -1].mean():.4f}")
    else:
        print("[inv] NO --seed-sae: chance-corrected retention is unavailable, so raw cosines "
              "carry no scale. Report them as uninterpretable across m until a second seed "
              "exists for this suite.")

    INH = None
    if args.base_sae:
        W_base, _ = load_decoder(args.base_sae, args.layer)
        B_hat, _ = normalize_rows(signatures(W_base, W_U_act, g))
        INH, _ = omp_curve(A_hat, B_hat, M, args.positive_only, args.n_restarts)
        print(f"[inv] base inheritance        m=1 {INH[:, 0].mean():.4f} ... "
              f"m={M} {INH[:, -1].mean():.4f}")

    # ---------------- the decile contrast: the claim lives in the slopes ----------------
    from identify_features import adjusted_breadth
    A = np.load(args.attr)
    active = A["is_active"].astype(bool)
    adj = adjusted_breadth(A["PR"].astype(np.float64), A["magnitude"].astype(np.float64),
                           A["base_rate"].astype(np.float64), active)
    if adj.shape[0] != F_t:
        raise SystemExit(f"attribution has {adj.shape[0]} features, SAE has {F_t} -- "
                         "are they the same dictionary?")

    dec_obs = decile_curves(OBS, adj)
    dec_null = decile_curves(NULL, adj)
    dec_sharp = decile_curves(np.repeat(sharp[:, None], M, axis=1), adj)

    print(f"\n=== cos vs m by adjusted-breadth decile (1 = most specialist, "
          f"{dec_obs['n_deciles']} = most general) ===")
    hdr = "  decile " + "".join(f"{'m='+str(i+1):>8s}" for i in range(M)) + f"{'slope':>9s}"
    print(hdr); print("  " + "-" * (len(hdr) - 2))
    for i, c in enumerate(dec_obs["curves"]):
        print(f"  {i+1:>6d} " + "".join(f"{v:8.4f}" for v in c)
              + f"{dec_obs['slopes'][i]:+9.4f}")
    print("  " + "-" * (len(hdr) - 2))
    print("  floor  " + "".join(f"{v:8.4f}" for v in NULL.mean(axis=0)))
    if CEIL is not None:
        print("  ceil   " + "".join(f"{v:8.4f}" for v in CEIL.mean(axis=0)))

    top, bot = dec_obs["slopes"][-1], dec_obs["slopes"][0]
    print(f"\n[inv] SLOPE most-general decile {top:+.4f}   most-specialist {bot:+.4f}   "
          f"difference {top - bot:+.4f}")
    if top > bot + 0.02:
        verdict = ("general features gain MORE from coalitions than specialists -- consistent "
                   "with splitting explaining the one-to-one null")
    elif bot > top + 0.02:
        verdict = ("specialists gain more -- the opposite of the splitting hypothesis; the "
                   "published finding is not a fragmentation artefact")
    else:
        verdict = ("slopes are indistinguishable -- coalitions help both ends equally, so the "
                   "one-to-one result is not a splitting artefact")
    print(f"[inv] READING: {verdict}")
    print("[inv] Read the slope against the floor's own slope above: a rise that merely tracks "
          "the\n[inv] random floor is dimension counting, not recurrence.")

    out = {"target": args.target, "others": others, "layer": args.layer, "m_max": M,
           "n_restarts": args.n_restarts, "positive_only": bool(args.positive_only),
           "observed_mean_curve": [float(v) for v in OBS.mean(axis=0)],
           "random_floor_curve": [float(v) for v in NULL.mean(axis=0)],
           "published_signed_q": published_q, "anti_aligned_fraction": anti,
           "decile_observed": dec_obs, "decile_random": dec_null,
           "decile_mean_sharpness": [c[0] for c in dec_sharp["curves"]],
           "slope_top_decile": top, "slope_bottom_decile": bot, "verdict": verdict}
    if CEIL is not None:
        out["seed_ceiling_curve"] = [float(v) for v in CEIL.mean(axis=0)]
        ret = chance_corrected_retention(OBS.mean(axis=0), NULL.mean(axis=0),
                                         CEIL.mean(axis=0))
        out["chance_corrected_retention"] = [float(v) for v in ret]
        print("\n[inv] chance-corrected retention (1 = changing the fine-tuning suite costs "
              "nothing\n[inv] beyond changing the SAE seed; 0 = no better than random):")
        print("  " + "".join(f"  m={i+1}:{v:.3f}" for i, v in enumerate(ret)))
    if INH is not None:
        out["inheritance_curve"] = [float(v) for v in INH.mean(axis=0)]

    np.savez_compressed(os.path.join(args.out, f"layer_{args.layer:02d}_inventory.npz"),
                        observed=OBS.astype(np.float32), random_floor=NULL.astype(np.float32),
                        adjusted_breadth=adj.astype(np.float32),
                        signature_sharpness=sharp.astype(np.float32),
                        signature_norm=A_norms.astype(np.float32),
                        **({"seed_ceiling": CEIL.astype(np.float32)} if CEIL is not None else {}),
                        **({"inheritance": INH.astype(np.float32)} if INH is not None else {}))
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[inv] wrote {args.out}")
    print("[inv] NEXT: rank-residualise the decile contrast on signature sharpness (saved in "
          "the npz)\n[inv] to rule out the geometry confound -- open thread #4 in results.md.")


if __name__ == "__main__":
    main()
