"""Encode shared-probe activations with each model's SAE and score cross-model recurrence.

Consumes the output of `collect_shared_probe.py` (per-model probe activations on a
shared frame set) plus each fine-tuned model's SAE, and computes the Path-B signal
(EXPERIMENT_PLAN.md §3.1):

  * q_cross      -- how well each fine-tuned model's features recur across the others
  * base-rate    -- reported control (recurrence should beat activity)
  * inheritance  -- if base-model probe activations are present, per feature: does it
                    respond the same way to the BASE model's residual as to its own?
                    High = the feature reads a direction already present in the base
                    (inherited, not learned during fine-tuning).  No base SAE needed:
                    we push the base residual through the fine-tuned model's own SAE.

If --seed2-map supplies a second-seed SAE for a target model, the same-model noise
floor (q_seed) and retention (q_cross / q_seed) are computed for that target; otherwise
q_cross gives a ranking and inheritance is the reference we DO have.

Usage
-----
python run_recurrence.py \
    --probe-dir ./activations/shared_probe_v1 \
    --sae-map goal=./sae/goal,spatial=./sae/spatial,object=./sae/object,libero10=./sae/libero10 \
    --base-key base \
    --layers 0,8,16,24,31 \
    --seed2-map goal=./sae/goal_seed1 \
    --out ./recurrence_v1
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
import torch

from mrvla.extract_codes_and_metrics import sae_forward
from mrvla.cross_model_recurrence import (
    base_rate,
    base_rate_residual,
    cross_model_q_permuted,
    recurrence_report,
    summarize,
)
from mrvla.structural_generality import _spearman


def parse_map(spec: str) -> dict[str, str]:
    out = {}
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise ValueError(f"--sae-map entries must be key=dir, got {part!r}")
        k, v = part.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def _sae_ckpt(sae_dir: str, layer_idx: int) -> str:
    cand = os.path.join(sae_dir, f"layer_{layer_idx:02d}", "final.pt")
    if os.path.exists(cand):
        return cand
    alt = os.path.join(sae_dir, f"layer_{layer_idx:02d}", "checkpoint.pt")
    if os.path.exists(alt):
        return alt
    hits = glob.glob(os.path.join(sae_dir, f"layer_{layer_idx:02d}*", "*.pt"))
    if hits:
        return hits[0]
    raise FileNotFoundError(f"No SAE checkpoint for layer {layer_idx} under {sae_dir}")


def encode(acts_layer: np.ndarray, sae_dir: str, layer_idx: int,
           device: str, batch_size: int) -> np.ndarray:
    """Encode [N, H] residuals with the model's layer SAE -> codes Z [N, F]."""
    ckpt = torch.load(_sae_ckpt(sae_dir, layer_idx), map_location="cpu", weights_only=False)
    return sae_forward(ckpt["W_enc"], ckpt["b_pre"], int(ckpt["config"]["k"]),
                       acts_layer.astype(np.float32), batch_size=batch_size, device=device)


def paired_column_corr(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Per-feature corr(A[:, j], B[:, j]) over rows (frames).  Returns [F]."""
    A = np.asarray(A, np.float64); B = np.asarray(B, np.float64)
    Az = A - A.mean(0, keepdims=True); Bz = B - B.mean(0, keepdims=True)
    num = (Az * Bz).sum(0)
    den = np.sqrt((Az ** 2).sum(0) * (Bz ** 2).sum(0))
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(den > 0, num / den, 0.0)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--probe-dir", required=True)
    p.add_argument("--sae-map", required=True, help="key=sae_dir pairs for fine-tuned models")
    p.add_argument("--seed2-map", default=None,
                   help="key=sae_dir pairs for a SECOND-seed SAE of the same model(s); "
                        "enables the same-model noise floor (q_seed) and retention.")
    p.add_argument("--base-key", default="base", help="probe key for the base model (inheritance ref)")
    p.add_argument("--layers", default="0,8,16,24,31")
    p.add_argument("--out", required=True)
    p.add_argument("--method", default="greedy", choices=("greedy", "hungarian"))
    p.add_argument("--n-perm", type=int, default=1,
                   help="Permutation-null repeats (chance floor for max-matching). 0 skips.")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--batch-size", type=int, default=4096)
    args = p.parse_args()
    perm_rng = np.random.default_rng(0)

    device = args.device if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out, exist_ok=True)
    sae_map = parse_map(args.sae_map)
    seed2_map = parse_map(args.seed2_map) if args.seed2_map else {}
    layer_indices = [int(x) for x in args.layers.split(",") if x.strip()]

    manifest = json.load(open(os.path.join(args.probe_dir, "probe_manifest.json")))
    probe_layers = list(manifest["layers"])

    def load_probe(key):
        d = np.load(os.path.join(args.probe_dir, f"probe_{key}.npz"))
        return d["acts"]  # [N, L, H]

    ft_keys = list(sae_map)
    acts = {k: load_probe(k) for k in ft_keys}
    base_acts = None
    if args.base_key and os.path.exists(os.path.join(args.probe_dir, f"probe_{args.base_key}.npz")):
        base_acts = load_probe(args.base_key)
        print(f"[rec] base inheritance reference: {args.base_key}", flush=True)

    summary = {"models": ft_keys, "base_key": args.base_key if base_acts is not None else None,
               "layers": {}}

    for layer_idx in layer_indices:
        lp = probe_layers.index(layer_idx)
        print(f"\n[rec] ===== layer {layer_idx} =====", flush=True)
        codes = {k: encode(acts[k][:, lp, :], sae_map[k], layer_idx, device, args.batch_size)
                 for k in ft_keys}
        # second-seed codes of the target model, for the same-model noise floor
        seed2_codes = {k: encode(acts[k][:, lp, :], seed2_map[k], layer_idx, device, args.batch_size)
                       for k in ft_keys if k in seed2_map}

        layer_out = {}
        for target in ft_keys:
            rep = recurrence_report(codes, target=target, method=args.method,
                                    seed2=seed2_codes.get(target))
            s = summarize(rep)

            # permutation null: chance floor for the best-of-F max-matching
            q_perm = None
            if args.n_perm > 0:
                others = [codes[k] for k in ft_keys if k != target]
                q_perm = cross_model_q_permuted(codes[target], others, perm_rng,
                                                n_perm=args.n_perm, method=args.method)
                act = rep["is_active"]
                s["q_perm_mean"] = float(np.nanmean(q_perm[act])) if act.any() else float("nan")
                s["q_gap_mean"] = s["q_cross_mean"] - s["q_perm_mean"]
                # chance-corrected retention: q_cross and q_seed both carry the same
                # best-of-F floor, so subtract it from both before the ratio. This is
                # the fraction of the ABOVE-CHANCE achievable matching that survives
                # changing the model -- a cleaner number than raw q_cross/q_seed.
                # Use the RATIO OF AGGREGATES, not the mean of per-feature ratios: an
                # individual feature's (q_seed_j - q_perm_j) can be tiny, so per-feature
                # ratios explode and their mean is unstable. The aggregate ratio is the
                # robust population-level estimate.
                if "q_seed" in rep:
                    num = float(np.nanmean(rep["q_cross"][act] - q_perm[act]))
                    den = float(np.nanmean(rep["q_seed"][act] - q_perm[act]))
                    s["retention_chance_corrected"] = num / den if den > 1e-6 else float("nan")

            inh_corr = None
            if base_acts is not None:
                Zb = encode(base_acts[:, lp, :], sae_map[target], layer_idx, device, args.batch_size)
                inh = paired_column_corr(codes[target], Zb)   # [F] inheritance per feature
                act = rep["is_active"]
                inh_corr = _spearman(rep["q_cross"][act], inh[act])  # does recurrence == inheritance?
                s["spearman_qcross_inheritance"] = inh_corr
                s["inheritance_mean"] = float(np.nanmean(inh[act])) if act.any() else float("nan")

            resid = base_rate_residual(rep["q_cross"], rep["base_rate"], rep["is_active"])
            save = dict(
                q_cross=rep["q_cross"], base_rate=rep["base_rate"],
                is_active=rep["is_active"].astype(np.uint8),
                qcross_baserate_residual_rank=resid,
            )
            if base_acts is not None:
                save["inheritance"] = inh
            if q_perm is not None:
                save["q_perm"] = q_perm
            if "q_seed" in rep:
                save["q_seed"] = rep["q_seed"]
                save["retention"] = rep["retention"]
            np.savez_compressed(
                os.path.join(args.out, f"layer_{layer_idx:02d}_target_{target}.npz"), **save)
            gap_str = (f"  q_perm={s['q_perm_mean']:.3f}  gap={s['q_gap_mean']:+.3f}"
                       if q_perm is not None else "")
            ret_str = ""
            if "retention_mean" in s:
                _qs = rep["q_seed"][rep["is_active"]]
                _qsm = float(np.nanmean(_qs)) if _qs.size else float("nan")
                ret_str = f"  q_seed={_qsm:.3f}  retention={s['retention_mean']:.3f}"
                if "retention_chance_corrected" in s:
                    ret_str += f"  ret_cc={s['retention_chance_corrected']:.3f}"
            print(f"  target={target:9s}  active={s['n_active']:4d}  "
                  f"q_cross mean={s['q_cross_mean']:.3f} median={s['q_cross_median']:.3f}"
                  + gap_str
                  + ret_str
                  + f"  q~baserate={s['spearman_qcross_baserate']:+.3f}"
                  + (f"  q~inherit={inh_corr:+.3f}" if inh_corr is not None else ""),
                  flush=True)
            layer_out[target] = s

        summary["layers"][f"layer_{layer_idx:02d}"] = layer_out

    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[rec] done -> {os.path.join(args.out, 'summary.json')}\n"
          f"[rec] high q_cross = recurs across models (general); high q~inherit means that "
          f"recurrence is explained by base-model inheritance, not fine-tuning.", flush=True)


if __name__ == "__main__":
    main()
