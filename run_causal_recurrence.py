"""Path B, redefined causally: does a feature's CAUSAL ROLE recur across models?

The activation-based recurrence (run_recurrence.py) matches features by WHEN they fire on a
shared probe. That is a *representational* similarity, and it is confounded by
distinctiveness: features with sharp firing fingerprints (both very general AND very
specialist ones) match easily, while muddy mid-range features do not -- which is exactly the
U-shaped artefact we observed against breadth. "Two features light up together" does not mean
they DO the same thing.

This script matches features by WHAT THEY DO instead, reusing Path A's causal term. OpenVLA
decodes an action as a dot product against 256 action-bin directions, and those 256 bins are
IDENTICAL across all four fine-tunes (same action tokenisation). So every feature has a
256-dimensional causal signature in a COMMON output space:

    S[j, t]  =  < w_j , g (*) u_t >          for action bin t = 1..256

which is precisely Path A's alignment term extended from the winning bin to all bins. We then
centre each signature across the 256 bins (the same contrast trick as Path A: a feature that
lifts every action equally does not steer the action and must not earn credit), L2-normalise,
and match features across models by cosine similarity.

Why this fixes the problem
  * FUNCTIONAL, not representational -- "do these two features push the action the same way?"
  * COMMON SPACE -- the 256 bins are shared, so no probe, no frame alignment, no arbitrary
    per-SAE dictionary numbering in the target space.
  * NO distinctiveness confound -- we never look at firing patterns, so the U-shape has no
    mechanism to appear.
  * NO GPU, NO PROBE -- pure linear algebra on each model's SAE decoder + action head.
  * Directly parallel to Path A: Path A asks whether causal influence recurs across TASKS;
    this asks whether the causal ROLE recurs across MODELS.

Output schema deliberately MATCHES run_recurrence.py (q_cross / base_rate / is_active /
q_perm / inheritance), so recurrence_vs_breadth.py, compare_recurrence_groups.py and
join_pathA_pathB.py all work on it unchanged.

Usage
-----
python run_causal_recurrence.py \
    --model goal=$B/ACT_ACTION/goal=$B/ACT_ACTION_SAE/goal/sae \
    --model spatial=$B/ACT_ACTION/spatial=$B/ACT_ACTION_SAE/spatial/sae \
    --model object=$B/ACT_ACTION/object=$B/ACT_ACTION_SAE/object/sae \
    --model 10=$B/ACT_ACTION/10=$B/ACT_ACTION_SAE/10/sae \
    --layer 31 --out $B/CAUSAL_RECURRENCE
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np


# ---------------------------------------------------------------------------
# The causal signature
# ---------------------------------------------------------------------------
def causal_signature(W_dec: np.ndarray, W_U_act: np.ndarray, g: np.ndarray,
                     center: bool = True) -> np.ndarray:
    """S [F, 256] with S[j, t] = <w_j, g (*) u_t>: feature j's effect on action bin t.

    Vectorised as (W_dec * g) @ W_U_act.T. If `center`, subtract each row's mean over the 256
    bins -- the contrast trick: only a feature's ability to favour SOME actions over others
    counts, a uniform lift of every bin changes no decision and is removed.
    """
    W_dec = np.asarray(W_dec, dtype=np.float64)
    W_U_act = np.asarray(W_U_act, dtype=np.float64)
    g = np.asarray(g, dtype=np.float64)
    S = (W_dec * g[None, :]) @ W_U_act.T            # [F, 256]
    if center:
        S = S - S.mean(axis=1, keepdims=True)
    return S


def normalize_rows(S: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (unit-norm rows, original norms). Zero-norm rows stay zero (they match nothing)."""
    n = np.linalg.norm(S, axis=1)
    out = np.zeros_like(S)
    nz = n > 0
    out[nz] = S[nz] / n[nz, None]
    return out, n


def best_match_cosine(S_hat_a: np.ndarray, S_hat_b: np.ndarray, method: str = "greedy"):
    """For each row of S_hat_a, the best cosine similarity against any row of S_hat_b.

    Both are unit-norm, so the full similarity matrix is a single matrix product.
    Returns (q [Fa], idx [Fa]).
    """
    C = S_hat_a @ S_hat_b.T                          # [Fa, Fb] cosine similarities
    if method == "hungarian":
        try:
            from scipy.optimize import linear_sum_assignment
            rows, cols = linear_sum_assignment(-C)
            q = np.zeros(C.shape[0]); idx = np.full(C.shape[0], -1)
            q[rows] = C[rows, cols]; idx[rows] = cols
            return q, idx
        except Exception:
            pass
    idx = np.argmax(C, axis=1)
    return C[np.arange(C.shape[0]), idx], idx


def cross_model_causal_q(S_target: np.ndarray, S_others: list[np.ndarray],
                         method: str = "greedy") -> np.ndarray:
    """Mean over other models of the best causal-signature match for each target feature."""
    qs = [best_match_cosine(S_target, S, method=method)[0] for S in S_others]
    return np.mean(np.stack(qs, axis=0), axis=0)


def causal_q_null(S_hat_target: np.ndarray, others_heads: list[tuple],
                  rng: np.random.Generator, n_perm: int = 3,
                  method: str = "greedy") -> np.ndarray:
    """Chance floor: RANDOM decoder directions pushed through each other model's OWN head.

    others_heads : list of (F, d, W_U_act, g, center) for each other model.

    Why this null and NOT permuting the action-bin axes. Every model's signatures are produced
    through an action head, and the heads are near-identical across the fine-tunes, so ALL
    signatures live in essentially the same low-dimensional subspace of the 256-bin space.
    Permuting bin axes would rotate one model OUT of that shared subspace, destroying geometry
    that is genuinely shared rather than merely coincidental -- which deflates the floor and
    manufactures a large fake gap (empirically ~0.33 on unrelated random models). The honest
    null must destroy only FEATURE CORRESPONDENCE while PRESERVING the shared readout
    geometry: draw random unit decoder directions, push them through the same head, and match.
    If real features match no better than random directions do, the gap is correctly ~0.
    """
    qs = []
    for _ in range(max(1, n_perm)):
        for (F, d, W_U, g, center) in others_heads:
            R = rng.standard_normal((F, d))
            R /= np.linalg.norm(R, axis=1, keepdims=True).clip(min=1e-12)
            Sh, _ = normalize_rows(causal_signature(R, W_U, g, center=center))
            qs.append(best_match_cosine(S_hat_target, Sh, method=method)[0])
    return np.mean(np.stack(qs, axis=0), axis=0)


# ---------------------------------------------------------------------------
def load_model(acts_dir: str, sae_dir: str, layer: int):
    """Return (W_dec [F,d], W_U_act [256,d], g [d]) for one model."""
    from run_attribution import load_sae
    hc = np.load(os.path.join(acts_dir, "head_constants.npz"))
    W_U_act = hc["W_U_act"].astype(np.float64)
    g = hc["g"].astype(np.float64)
    _W_enc, W_dec_t, _b_pre, _k, ck = load_sae(sae_dir, layer)
    W_dec = W_dec_t.detach().float().cpu().numpy().astype(np.float64)
    return W_dec, W_U_act, g, ck


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", action="append", required=True,
                   help="label=acts_dir=sae_dir ; repeat for each model")
    p.add_argument("--base", default=None,
                   help="optional label=acts_dir=sae_dir for the BASE model (inheritance ref)")
    p.add_argument("--attr-map", default=None,
                   help="optional comma list label=attribution_npz, to copy per-feature "
                        "base_rate for the downstream activity control")
    p.add_argument("--layer", type=int, default=31)
    p.add_argument("--method", choices=["greedy", "hungarian"], default="greedy")
    p.add_argument("--n-perm", type=int, default=3)
    p.add_argument("--no-center", action="store_true",
                   help="skip the contrast centering (NOT recommended -- see module docstring)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    center = not args.no_center

    specs = {}
    for s in args.model:
        label, acts, sae = s.split("=", 2)
        specs[label] = (acts, sae)

    attr_map = {}
    if args.attr_map:
        for kv in args.attr_map.split(","):
            k, v = kv.split("=", 1)
            attr_map[k.strip()] = v.strip()

    # ---- build every model's causal signature ----
    S_raw, S_hat, norms, heads = {}, {}, {}, {}
    for label, (acts, sae) in specs.items():
        W_dec, W_U_act, g, ck = load_model(acts, sae, args.layer)
        S = causal_signature(W_dec, W_U_act, g, center=center)
        Sh, n = normalize_rows(S)
        S_raw[label], S_hat[label], norms[label] = S, Sh, n
        # (F, d, head, gain, center) -- needed to build the random-decoder null for this model
        heads[label] = (W_dec.shape[0], W_dec.shape[1], W_U_act, g, center)
        print(f"[causal] {label:10s} W_dec{W_dec.shape} -> signature{S.shape}  "
              f"median|S|={np.median(n):.3e}  dead(|S|=0)={int((n == 0).sum())}", flush=True)

    base_S_hat = None
    if args.base:
        blabel, bacts, bsae = args.base.split("=", 2)
        bW, bU, bg, _ = load_model(bacts, bsae, args.layer)
        base_S_hat, _ = normalize_rows(causal_signature(bW, bU, bg, center=center))
        print(f"[causal] base inheritance reference: {blabel}", flush=True)

    rng = np.random.default_rng(args.seed)
    summary = {"layer": args.layer, "method": args.method, "centered": center,
               "models": list(specs), "targets": {}}

    for target in specs:
        others = [S_hat[m] for m in specs if m != target]
        if not others:
            continue
        q = cross_model_causal_q(S_hat[target], others, method=args.method)
        q_perm = causal_q_null(S_hat[target], [heads[m] for m in specs if m != target],
                               rng, n_perm=args.n_perm, method=args.method)
        active = norms[target] > 0

        # inheritance: does this feature have the SAME causal role under the base model's head?
        inh = None
        if base_S_hat is not None:
            inh = np.einsum("ij,ij->i", S_hat[target], base_S_hat)   # paired cosine, per feature
        # base rate from Path A's attribution, for the downstream activity control
        br = np.zeros(len(q))
        if target in attr_map:
            A = np.load(attr_map[target])
            if A["base_rate"].shape == q.shape:
                br = A["base_rate"].astype(np.float64)
                active = active & A["is_active"].astype(bool)
            else:
                print(f"[causal] WARN {target}: attribution feature count "
                      f"{A['base_rate'].shape} != {q.shape}; base_rate not copied", flush=True)

        gap = float(np.nanmean(q[active]) - np.nanmean(q_perm[active]))
        s = {"n_active": int(active.sum()),
             "q_causal_mean": float(np.nanmean(q[active])),
             "q_causal_median": float(np.nanmedian(q[active])),
             "q_perm_mean": float(np.nanmean(q_perm[active])),
             "gap": gap}
        if inh is not None:
            s["inheritance_mean"] = float(np.nanmean(inh[active]))
        summary["targets"][target] = s
        print(f"[causal] target={target:10s} active={s['n_active']:5d}  "
              f"q_causal mean={s['q_causal_mean']:.3f} median={s['q_causal_median']:.3f}  "
              f"q_perm={s['q_perm_mean']:.3f}  gap={gap:+.3f}"
              + (f"  inherit={s['inheritance_mean']:+.3f}" if inh is not None else ""), flush=True)

        # schema-compatible with run_recurrence so downstream tools just work
        save = dict(q_cross=q.astype(np.float32), base_rate=br.astype(np.float32),
                    is_active=active.astype(np.uint8), q_perm=q_perm.astype(np.float32),
                    signature_norm=norms[target].astype(np.float32))
        if inh is not None:
            save["inheritance"] = inh.astype(np.float32)
        np.savez_compressed(
            os.path.join(args.out, f"layer_{args.layer:02d}_target_{target}.npz"), **save)

    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[causal] done -> {args.out}")
    print("[causal] q_causal = cosine match of a feature's 256-dim ACTION-BIN causal signature\n"
          "[causal] against the best-matching feature in each other model. gap = above the\n"
          "[causal] bin-permutation floor. Files are schema-compatible: feed them to\n"
          "[causal] recurrence_vs_breadth.py / compare_recurrence_groups.py / join_pathA_pathB.py.")


if __name__ == "__main__":
    main()
