"""A4 Stage 0: how many distinguishable causal roles does the action readout actually have?

THE GATE. Every Path B statistic -- q_causal, the random-decoder floor, and any subspace or
matching method built on top of them -- lives in the row space of the contrast-centered action
unembedding. Nobody has measured that space. Its dimension decides which methods are even
meaningful, so this runs before anything else in A4.

The reason to expect it to be small: the 256 action tokens are an ORDERED discretization of one
continuous value, so their unembedding rows plausibly lie near a low-dimensional curve rather
than spanning 256 directions. If the effective rank r_eff is (say) 8, then:

  * every SAE feature's causal effect on the action lives in an 8-dimensional space, and
    "2048 features" is an illusion at the output -- there are only ~r_eff distinguishable
    causal roles per decode slot, ~7*r_eff across the whole action. That is a finding about
    VLA readout geometry in its own right, and it explains why so many features' signatures
    look alike;
  * a random m-dimensional subspace already captures ~m/r_eff of any direction, so
    one-to-many matching saturates almost immediately and the m-sweep in
    `inventory_recurrence.py` cannot separate signal from arithmetic;
  * the honest route becomes distributional (where directions sit inside that small space)
    rather than span-based (which directions are covered).

This script measures r_eff, prints the saturation curve that decides it, and states the branch.

It also verifies the assumption the whole common-output-space argument rests on. LoRA here
trains only q/k/v/o_proj (`train_lora.py`), so the unembedding and the final-norm gain should be
IDENTICAL across all four fine-tuned models -- making the 256-bin space literally shared rather
than merely analogous. That is checked, not assumed: pass several --head files and any
disagreement is reported loudly.

Usage
-----
python action_space_geometry.py --head goal=$B/ACT_ACTION/goal/head_constants.npz
python action_space_geometry.py --head goal=$B/ACT_ACTION/goal/head_constants.npz \
                                --head spatial=$B/ACT_ACTION/spatial/head_constants.npz \
                                --head object=$B/ACT_ACTION/object/head_constants.npz \
                                --head 10=$B/ACT_ACTION/10/head_constants.npz \
                                --out $B/DIAGNOSTICS/action_space_geometry.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os

import numpy as np


# ---------------------------------------------------------------------------
# spectrum summaries
# ---------------------------------------------------------------------------
def effective_rank(sv: np.ndarray) -> float:
    """Participation ratio of the ENERGY spectrum: (sum s^2)^2 / sum s^4.

    The same scale-free statistic Path A uses for task breadth, applied to singular values.
    Reads as "the effective number of directions carrying the variance": equals the true rank
    for a flat spectrum, and collapses toward 1 as one direction dominates.
    """
    e = np.asarray(sv, dtype=np.float64) ** 2
    s1, s2 = e.sum(), (e * e).sum()
    return float(s1 * s1 / s2) if s2 > 0 else float("nan")


def rank_at_energy(sv: np.ndarray, frac: float) -> int:
    """Smallest k whose top-k singular directions hold `frac` of the total energy."""
    e = np.asarray(sv, dtype=np.float64) ** 2
    tot = e.sum()
    if tot <= 0:
        return 0
    return int(np.searchsorted(np.cumsum(e) / tot, frac) + 1)


def spectrum_report(M: np.ndarray) -> dict:
    """SVD summary of a [n, d] matrix: singular values plus the three rank statistics."""
    sv = np.linalg.svd(np.asarray(M, dtype=np.float64), compute_uv=False)
    return {"n_rows": int(M.shape[0]), "n_cols": int(M.shape[1]),
            "singular_values": [float(v) for v in sv[:64]],
            "effective_rank": effective_rank(sv),
            "rank_90pct_energy": rank_at_energy(sv, 0.90),
            "rank_99pct_energy": rank_at_energy(sv, 0.99),
            "condition_number": float(sv[0] / sv[-1]) if sv[-1] > 0 else float("inf")}


def contrast_center(W_U_act: np.ndarray) -> np.ndarray:
    """Subtract the mean action-token row -- the same contrast trick Path A and Path B use.

    A direction that lifts every action logit equally cannot change the argmax, so it carries
    no causal information about WHICH action is chosen. Removing the mean row is what makes
    the remaining geometry the geometry that matters, and it necessarily costs at most one
    dimension of rank.
    """
    U = np.asarray(W_U_act, dtype=np.float64)
    return U - U.mean(axis=0, keepdims=True)


# ---------------------------------------------------------------------------
# what a low ambient rank does to one-to-many matching
# ---------------------------------------------------------------------------
def random_subspace_cosine(r_eff: float, m: int) -> float:
    """Expected cos^2 between a random unit direction and a random m-dim subspace in r_eff dims.

    For an isotropic direction in R^r and a uniformly random m-dimensional subspace, the
    expected squared cosine of the projection is exactly m/r. This is the arithmetic floor
    that any one-to-many matching result has to beat: at m/r_eff near 1 a "match" means
    nothing at all, because m arbitrary directions already span the space.

    Signatures are not isotropic, so this is indicative rather than exact -- the real null is
    the m-matched random-decoder draw in inventory_recurrence.py. It is here because it is
    free and it decides whether that experiment is worth running.
    """
    if not np.isfinite(r_eff) or r_eff <= 0:
        return float("nan")
    return float(min(1.0, m / r_eff))


def saturation_curve(r_eff: float, ms=(1, 2, 3, 5, 8, 12)) -> list[dict]:
    return [{"m": int(m), "expected_cos2_random": random_subspace_cosine(r_eff, m),
             "expected_cos_random": float(np.sqrt(random_subspace_cosine(r_eff, m)))}
            for m in ms]


def _max_usable_m(r: float, ms) -> int:
    """Largest m in `ms` where a random m-subspace still explains under half of an arbitrary
    direction (cos^2 < 0.5). Beyond it, "matching" is dimension counting."""
    usable = [m for m in ms if random_subspace_cosine(r, m) < 0.5]
    return max(usable) if usable else 0


def recommend_branch(spectrum: dict, ms=(1, 2, 3, 5, 8, 12)) -> dict:
    """Which A4 design the measured geometry supports -- decided from a BRACKET, not one number.

    Rank is not a single quantity here. The energy participation ratio is pessimistic when the
    spectrum is heavy-headed (a couple of dominant directions drag it toward 1 even though many
    smaller directions carry real structure), while the 99%-energy rank is optimistic (it counts
    directions that are barely above noise). The honest reading is the interval between them,
    and the branch is only called with confidence when both ends agree.

    When they disagree we say so and defer: the analytic m/r proxy is indicative, but the real
    arbiter is the m-matched random-decoder null in inventory_recurrence.py, which measures the
    floor on the actual signature distribution rather than assuming isotropy. In that case run
    the sweep at small m WITH the empirical null, and lead with the distributional view.
    """
    r_pess = spectrum.get("effective_rank", float("nan"))
    r_opt = float(spectrum.get("rank_99pct_energy", 0)) or float("nan")
    m_pess, m_opt = _max_usable_m(r_pess, ms), _max_usable_m(r_opt, ms)
    lo, hi = min(m_pess, m_opt), max(m_pess, m_opt)

    if not (np.isfinite(r_pess) and np.isfinite(r_opt)):
        branch, why, conf = "UNKNOWN", "spectrum could not be computed", "none"
    elif lo >= 5:
        branch, conf = "m-sweep (inventory_recurrence.py), full range", "high"
        why = (f"rank bracket [{r_pess:.1f}, {r_opt:.0f}] leaves room out to m={lo} at BOTH "
               f"ends: a random m-subspace still explains under half of an arbitrary direction "
               f"there, so improvement with m can be attributed to real coalition structure "
               f"rather than to dimension counting.")
    elif hi < 2:
        branch, conf = "distributional only (inventory_clusters.py)", "high"
        why = (f"rank bracket [{r_pess:.1f}, {r_opt:.0f}] is too small for span-based matching "
               f"at either end: even m=2 random directions explain most of an arbitrary "
               f"signature. Span overlap is trivially high for every subset, so the question "
               f"must be WHERE directions sit inside this small space, not which subspace "
               f"covers them.")
    elif lo == hi:
        branch, conf = f"m-sweep, truncated at m<={lo}", "high"
        why = (f"rank bracket [{r_pess:.1f}, {r_opt:.0f}] agrees on m<={lo}. The sweep is "
               f"interpretable for small m only; beyond that the random floor exceeds cos 0.7 "
               f"and any 'match' is arithmetic. Report the truncated sweep and pair it with "
               f"the distributional view.")
    else:
        branch, conf = f"AMBIGUOUS -- run m-sweep to m<={hi} with the empirical null", "low"
        why = (f"the two rank estimates disagree about the design: the energy participation "
               f"ratio ({r_pess:.1f}) allows m<={m_pess} while the 99%-energy rank "
               f"({r_opt:.0f}) allows m<={m_opt}. The analytic m/r proxy assumes isotropy and "
               f"cannot settle it. Run the sweep to m<={hi} and let the m-matched "
               f"random-decoder null decide -- it measures the floor on the real signature "
               f"distribution. Lead with the distributional view until it does.")
    return {"r_effective_rank": r_pess, "r_99pct_energy": r_opt,
            "max_usable_m_pessimistic": m_pess, "max_usable_m_optimistic": m_opt,
            "max_usable_m": lo, "confidence": conf, "branch": branch, "why": why}


def signature_spectrum(W_dec: np.ndarray, W_U_act: np.ndarray, g: np.ndarray) -> dict:
    """Spectrum of the causal-signature matrix S [F, 256] the model's features ACTUALLY occupy.

    The W_U_act spectrum is an upper bound on the space signatures could occupy. This is the
    space they do occupy: the decoders may excite only some of the readout's directions, so
    r_eff(S) <= r_eff(centered W_U_act) and it is r_eff(S) that governs every matching
    statistic. When an SAE is available this replaces the bound as the basis for the branch.
    """
    from run_causal_recurrence import causal_signature
    S = causal_signature(np.asarray(W_dec, dtype=np.float64),
                         np.asarray(W_U_act, dtype=np.float64),
                         np.asarray(g, dtype=np.float64), center=True)
    rep = spectrum_report(S)
    rep["n_features"] = int(S.shape[0])
    return rep


# ---------------------------------------------------------------------------
# the shared-head assumption
# ---------------------------------------------------------------------------
def array_digest(a: np.ndarray) -> str:
    """Content hash of an array's exact bytes, for cross-model identity checks."""
    a = np.ascontiguousarray(np.asarray(a))
    return hashlib.sha256(a.tobytes()).hexdigest()[:16]


def compare_heads(heads: dict) -> dict:
    """Are W_U_act / g / act_ids identical across models? Returns per-key digests and a verdict.

    This is load-bearing, not bookkeeping. If the heads match, every model's causal signature is
    the SAME fixed linear map applied to a different decoder, cross-model comparison carries no
    alignment ambiguity, and the shared 256-bin output space is exact. If they differ, every
    cross-model signature number needs an alignment step that does not currently exist.
    """
    keys = ["W_U_act", "g", "act_ids"]
    digests = {k: {name: array_digest(h[k]) for name, h in heads.items() if k in h}
               for k in keys}
    identical = {k: len(set(d.values())) <= 1 for k, d in digests.items() if d}
    max_absdiff = {}
    names = list(heads)
    if len(names) > 1:
        ref = heads[names[0]]
        for k in keys:
            if k in ref and all(k in h for h in heads.values()):
                try:
                    max_absdiff[k] = float(max(
                        np.abs(np.asarray(heads[n][k], dtype=np.float64)
                               - np.asarray(ref[k], dtype=np.float64)).max()
                        for n in names[1:]))
                except ValueError:
                    max_absdiff[k] = float("nan")     # shape mismatch
    return {"digests": digests, "identical": identical, "max_abs_diff": max_absdiff}


def bin_order_note(act_ids: np.ndarray) -> dict:
    """The reversed bin axis, surfaced so downstream code cannot get the sign wrong.

    OpenVLA decodes actions with token_id = vocab_size - bin_index for bin_index in [1, 256],
    while act_ids is stored ascending. So row 0 of W_U_act is the HIGHEST bin index and row 255
    the lowest: the row axis runs opposite to the action-value axis. Anything that reads the
    bin axis as ordered -- a signed shift magnitude, "does this feature push the action up or
    down" -- must flip it, and the error is silent if it does not.
    """
    ids = np.asarray(act_ids).ravel()
    return {"n_bins": int(ids.size),
            "first_id": int(ids[0]) if ids.size else -1,
            "last_id": int(ids[-1]) if ids.size else -1,
            "ascending_ids": bool(ids.size > 1 and ids[1] > ids[0]),
            "row_axis_is_reversed_vs_bin_index": True,
            "note": "row r of W_U_act corresponds to bin_index = n_bins - r; reverse before "
                    "treating the bin axis as ordered"}


def head_divergence_impact(W_dec: np.ndarray, heads: dict, ref: str) -> dict:
    """How much does using the WRONG model's head perturb a signature?

    `max |diff|` on the unembedding is not interpretable on its own -- what matters is whether
    the difference moves the quantity A4 actually compares. So push ONE decoder through every
    model's head and report the per-feature cosine against the reference head's signature.

    Near 1.0: the heads are interchangeable and a shared-head shortcut would have been harmless.
    Materially below 1.0: signatures must be computed through each model's own head, and any
    analysis that shared one head was comparing through mismatched linear maps.
    """
    from run_causal_recurrence import causal_signature
    W_ref, g_ref = heads[ref]["W_U_act"], heads[ref]["g"]
    S_ref = causal_signature(np.asarray(W_dec, float), np.asarray(W_ref, float),
                             np.asarray(g_ref, float), center=True)
    n_ref = np.linalg.norm(S_ref, axis=1)
    out = {}
    for name, h in heads.items():
        if name == ref:
            continue
        S = causal_signature(np.asarray(W_dec, float), np.asarray(h["W_U_act"], float),
                             np.asarray(h["g"], float), center=True)
        n = np.linalg.norm(S, axis=1)
        ok = (n_ref > 0) & (n > 0)
        cos = (S_ref[ok] * S[ok]).sum(axis=1) / (n_ref[ok] * n[ok])
        out[name] = {"mean_cosine": float(cos.mean()), "min_cosine": float(cos.min()),
                     "p05_cosine": float(np.percentile(cos, 5)), "n_features": int(ok.sum())}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--head", action="append", required=True,
                    help="head_constants.npz, optionally 'name=path'. Repeatable -- pass all "
                         "four models to check the shared-head assumption.")
    ap.add_argument("--sae", action="append", default=None,
                    help="SAE dir, optionally 'name=path'. When given, the spectrum of the "
                         "actual causal-signature matrix is measured instead of relying on the "
                         "W_U_act bound -- this is the number that governs A4. Needs torch.")
    ap.add_argument("--layer", type=int, default=31)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    heads = {}
    for spec in args.head:
        name, sep, path = spec.partition("=")
        if not sep:
            name, path = os.path.basename(os.path.dirname(path or name)), name
        d = np.load(path)
        heads[name] = {k: d[k] for k in d.files}
        heads[name]["_path"] = path

    results: dict = {"models": {}}
    _W_dec_cache: list = []      # first loaded decoder, reused by the head-divergence check

    first = next(iter(heads))
    W_U_act = np.asarray(heads[first]["W_U_act"], dtype=np.float64)
    results["bin_order"] = bin_order_note(heads[first]["act_ids"])

    raw = spectrum_report(W_U_act)
    cen = spectrum_report(contrast_center(W_U_act))
    results["spectrum_raw"] = raw
    results["spectrum_contrast_centered"] = cen
    results["saturation_curve"] = saturation_curve(cen["effective_rank"])

    print(f"\n[geom] action readout geometry from {heads[first]['_path']}")
    print(f"[geom] W_U_act {W_U_act.shape}  ({raw['n_rows']} action bins in {raw['n_cols']}-dim "
          f"residual space)")
    print(f"[geom] RAW spectrum              effective rank {raw['effective_rank']:7.2f}   "
          f"90% energy at {raw['rank_90pct_energy']:3d}   99% at {raw['rank_99pct_energy']:3d}")
    print(f"[geom] CONTRAST-CENTERED         effective rank {cen['effective_rank']:7.2f}   "
          f"90% energy at {cen['rank_90pct_energy']:3d}   99% at {cen['rank_99pct_energy']:3d}")
    print(f"[geom]   ^ this is the space every causal signature lives in. It bounds the number")
    print(f"[geom]     of distinguishable causal roles per decode slot; times 7 slots for the")
    print(f"[geom]     whole action.")
    sv = np.array(cen["singular_values"][:12])
    if sv.size:
        print("[geom] top singular values (centered): "
              + "  ".join(f"{v:.3g}" for v in sv))

    print("\n[geom] what a random m-dim subspace already explains in this space:")
    for row in results["saturation_curve"]:
        bar = "#" * int(round(40 * row["expected_cos2_random"]))
        print(f"[geom]   m={row['m']:2d}  E[cos^2]={row['expected_cos2_random']:.3f}  "
              f"E[cos]={row['expected_cos_random']:.3f}  |{bar}")
    # the readout bound is what we have without an SAE; the signature spectrum supersedes it
    basis, basis_label = cen, "contrast-centered W_U_act (an UPPER BOUND on the signature space)"
    if args.sae:
        results["signature_spectra"] = {}
        for spec in args.sae:
            name, sep, path = spec.partition("=")
            if not sep:
                name, path = os.path.basename(path or name), name
            from run_attribution import load_sae
            _We, W_dec_t, _b, _k, ck = load_sae(path, args.layer)
            W_dec = W_dec_t.detach().float().cpu().numpy()
            if not _W_dec_cache:
                _W_dec_cache.append(W_dec)
            sig = signature_spectrum(W_dec, W_U_act, heads[first]["g"])
            results["signature_spectra"][name] = {"sae": ck, **sig}
            print(f"\n[geom] SIGNATURE SPACE actually occupied by {name}'s {sig['n_features']} "
                  f"features")
            print(f"[geom]   effective rank {sig['effective_rank']:7.2f}   "
                  f"90% energy at {sig['rank_90pct_energy']:3d}   "
                  f"99% at {sig['rank_99pct_energy']:3d}")
            print(f"[geom]   (<= the readout bound above; THIS is what governs A4)")
        # route on the first SAE given
        first_sig = next(iter(results["signature_spectra"].values()))
        basis, basis_label = first_sig, "the measured causal-signature spectrum"
        results["saturation_curve"] = saturation_curve(first_sig["effective_rank"])

    results["branch"] = recommend_branch(basis)
    results["branch"]["decided_from"] = basis_label
    b = results["branch"]
    print(f"\n[geom] BRANCH ({b['confidence']} confidence, from {basis_label}):")
    print(f"[geom]   {b['branch']}")
    print(f"[geom]   {b['why']}")

    if len(heads) > 1:
        cmp = compare_heads(heads)
        results["head_comparison"] = cmp
        print(f"\n[geom] shared-head check across {len(heads)} models "
              f"({', '.join(heads)}):")
        for k, ok in cmp["identical"].items():
            diff = cmp["max_abs_diff"].get(k)
            extra = ""
            if diff is not None:
                ref = np.asarray(heads[first].get(k, []), dtype=np.float64)
                scale = float(np.abs(ref).std()) if ref.size else float("nan")
                rel = diff / scale if scale > 0 else float("nan")
                extra = f"  (max |diff| = {diff:.3g}, {rel:.2f}x the entry sd)"
            print(f"[geom]   {k:10s} identical: {'YES' if ok else 'NO'}{extra}")
        if all(cmp["identical"].values()):
            print("[geom]   -> the 256-bin output space is LITERALLY shared. Cross-model "
                  "signature\n[geom]      comparison carries no alignment ambiguity, and every "
                  "difference between\n[geom]      models' signatures comes from W_dec alone.")
        else:
            print("[geom]   -> heads DIFFER, so a shared-head shortcut is invalid: each model's")
            print("[geom]      decoder must be pushed through ITS OWN head. The 256 action bins")
            print("[geom]      still mean the same commanded action in every model, so comparing")
            print("[geom]      effects-on-bins across models remains well defined -- it is the")
            print("[geom]      LINEAR MAP that differs, not the semantic space.")
            if "signature_spectra" in results:
                impact = head_divergence_impact(_W_dec_cache[0], heads, first)
                results["head_divergence_impact"] = impact
                print("[geom]   how much does the wrong head perturb a signature? (same decoder,")
                print(f"[geom]   other models' heads vs {first}'s):")
                worst = 1.0
                for name, v in impact.items():
                    print(f"[geom]     {name:10s} mean cos {v['mean_cosine']:.4f}   "
                          f"5th pct {v['p05_cosine']:.4f}   min {v['min_cosine']:.4f}")
                    worst = min(worst, v["mean_cosine"])
                if worst > 0.99:
                    print("[geom]   -> the perturbation is cosmetic; per-model heads are still")
                    print("[geom]      correct, but results would barely move either way.")
                else:
                    print("[geom]   -> the perturbation is MATERIAL. Any analysis that shared one")
                    print("[geom]      head compared through mismatched maps; re-run with per-model")
                    print("[geom]      --head arguments.")
            else:
                print("[geom]      Pass --sae to measure how much the difference actually moves a")
                print("[geom]      signature (the max |diff| above is not interpretable alone).")

    for name, h in heads.items():
        results["models"][name] = {"path": h["_path"],
                                   "shape": [int(x) for x in np.shape(h["W_U_act"])]}

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n[geom] wrote {args.out}")


if __name__ == "__main__":
    main()
