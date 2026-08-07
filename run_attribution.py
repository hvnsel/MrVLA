"""Path A stages A3 + A4: run the viability gate, then compute causal task-breadth.

Consumes:
  * Stage-A1 shards          (residual [n,7,d], token_ids [n,7], task_id [n], ...)
  * head_constants.npz       (W_U_act [256,d], act_ids, g [d], eps)   -- from A1
  * the SAE retrained on action-position residuals (A2)

Produces:
  * GATE report -- L1: does the SAE reconstruction re-decode to the same action token?
                   L2: do the frozen-r per-feature phi sum back to the true logit?
    This is the go/no-go. If L1 < 0.85 the attribution is meaningless and we stop.
  * If the gate passes: per-feature causal task-breadth
        C_j(g)  = mean over task-g decisions of |phi_j|
        PR_j    = (sum_g C_j(g))^2 / sum_g C_j(g)^2      (effective #tasks driven)
    plus the mandatory confound controls (PR vs total causal magnitude, PR vs base rate,
    and leave-one-task-out prediction of held-out causal importance).

Memory note: phi is [n_decisions x F] which is far too large to materialise (~560k x 2048
per model). phi is sparse (only k=100 features active per decision), and C_j(g) is just a
per-task mean, so we STREAM the shards and accumulate per-task sums. Nothing large is held.

Usage
-----
python run_attribution.py \
    --acts-dir ACT_ACTION/goal \
    --sae-dir  ACT_ACTION_SAE/goal/sae \
    --layer 31 --out ATTR/goal
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
import torch

from mrvla.attribution import (
    action_logits,
    contrast_direction,
    participation_ratio,
    reconstruct,
    rms,
    rmsnorm,
    total_magnitude,
)
from mrvla.structural_generality import _partial_spearman, _spearman


# ---------------------------------------------------------------------------
# SAE encode returning the per-sample normalisers the attribution formula needs
# ---------------------------------------------------------------------------
@torch.no_grad()
def sae_encode_full(W_enc, b_pre, k: int, X: np.ndarray, device: str,
                    batch_size: int = 4096):
    """Encode X [N,d] -> (z [N,F], l2 [N], mu [N]).

    Mirrors TopKSAE.forward's pre-normalisation exactly (train_sae.py `_prenorm`):
        x_shift = x - b_pre ;  mu = mean(x_shift) ;  l2 = ||x_shift - mu||
        x_norm  = (x_shift - mu) / l2 ;  z = ReLU(TopK(x_norm @ W_enc))
    Unlike extract_codes_and_metrics.sae_forward we RETURN l2 and mu, because the
    reconstruction is h_hat = l2*(z @ W_dec) + mu + b_pre and the attribution carries the
    l2 factor. Discarding them (as the Path-B encoder does) would make phi wrong by a
    per-sample scale.
    """
    N, d = X.shape
    F = W_enc.shape[1]
    z_out = np.empty((N, F), dtype=np.float32)
    l2_out = np.empty(N, dtype=np.float32)
    mu_out = np.empty(N, dtype=np.float32)
    W_enc_d = W_enc.to(device)
    b_pre_d = b_pre.to(device)

    for i in range(0, N, batch_size):
        j = min(i + batch_size, N)
        xb = torch.from_numpy(X[i:j]).to(device, dtype=torch.float32)
        x_shift = xb - b_pre_d
        mu = x_shift.mean(dim=1, keepdim=True)
        x_cent = x_shift - mu
        l2 = x_cent.norm(dim=1, keepdim=True).clamp(min=1e-8)
        x_norm = x_cent / l2
        pre = x_norm @ W_enc_d
        vals, idx = pre.topk(k, dim=1)
        z = torch.zeros_like(pre)
        z.scatter_(1, idx, torch.relu(vals))
        z_out[i:j] = z.cpu().numpy()
        l2_out[i:j] = l2.squeeze(1).cpu().numpy()
        mu_out[i:j] = mu.squeeze(1).cpu().numpy()
    return z_out, l2_out, mu_out


def load_sae(sae_dir: str, layer: int):
    cand = os.path.join(sae_dir, f"layer_{layer:02d}", "final.pt")
    if not os.path.exists(cand):
        alt = os.path.join(sae_dir, f"layer_{layer:02d}", "checkpoint.pt")
        cand = alt if os.path.exists(alt) else cand
    if not os.path.exists(cand):
        hits = glob.glob(os.path.join(sae_dir, f"layer_{layer:02d}*", "*.pt"))
        if not hits:
            raise FileNotFoundError(f"No SAE checkpoint for layer {layer} under {sae_dir}")
        cand = hits[0]
    ck = torch.load(cand, map_location="cpu", weights_only=False)
    return ck["W_enc"], ck["W_dec"], ck["b_pre"], int(ck["config"]["k"]), cand


# ---------------------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--acts-dir", required=True, help="A1 output (residual shards + head_constants)")
    p.add_argument("--sae-dir", required=True, help="SAE trained on action-position residuals")
    p.add_argument("--layer", type=int, default=31)
    p.add_argument("--out", required=True)
    p.add_argument("--gate-sample", type=int, default=2000,
                   help="decisions used for the gate (L1/L2); 0 = all")
    p.add_argument("--gate-threshold", type=float, default=0.85)
    p.add_argument("--force", action="store_true",
                   help="compute A4 even if the gate fails (for diagnosis only)")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--batch-size", type=int, default=4096)
    args = p.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out, exist_ok=True)

    hc = np.load(os.path.join(args.acts_dir, "head_constants.npz"))
    W_U_act = hc["W_U_act"].astype(np.float64)
    g_gain = hc["g"].astype(np.float64)
    eps = float(hc["eps"])
    act_ids = hc["act_ids"].astype(np.int64)
    id0 = int(act_ids[0])
    n_act, d = W_U_act.shape

    W_enc, W_dec_t, b_pre_t, k, ck_path = load_sae(args.sae_dir, args.layer)
    W_dec = W_dec_t.detach().float().cpu().numpy().astype(np.float64)   # [F, d]
    b_pre_np = b_pre_t.detach().float().cpu().numpy().astype(np.float64)
    F = W_dec.shape[0]
    print(f"[attr] SAE {ck_path}  F={F}  k={k}  d={d}  action ids {id0}..{id0+n_act-1}",
          flush=True)

    shards = sorted(glob.glob(os.path.join(args.acts_dir, "shard_*.npz")))
    if not shards:
        raise FileNotFoundError(f"no shard_*.npz in {args.acts_dir}")

    # ---------------- pass 1: the GATE, on a subsample ----------------------
    print("[attr] === GATE ===", flush=True)
    n_seen = agree = 0
    logit_corrs, true_c, recon_c, hhat_c = [], [], [], []
    for sp in shards:
        dd = np.load(sp)
        res = dd["residual"]                       # [n,7,d] float16
        toks = dd["token_ids"].astype(np.int64)    # [n,7]
        n = res.shape[0]
        Xflat = res.reshape(n * 7, d).astype(np.float32)
        z, l2, mu = sae_encode_full(W_enc, b_pre_t, k, Xflat, device, args.batch_size)
        for r in range(n * 7):
            if args.gate_sample and n_seen >= args.gate_sample:
                break
            h = Xflat[r].astype(np.float64)
            tok = int(toks[r // 7, r % 7])
            tok_row = tok - id0
            if not (0 <= tok_row < n_act):
                continue
            true_l = action_logits(h, W_U_act, g_gain, eps)
            h_hat = reconstruct(z[r], W_dec, float(l2[r]), float(mu[r]), b_pre_np)
            rec_l = action_logits(h_hat, W_U_act, g_gain, eps)
            agree += int(np.argmax(true_l) == np.argmax(rec_l))
            a = true_l - true_l.mean(); b = rec_l - rec_l.mean()
            den = np.sqrt((a * a).sum() * (b * b).sum())
            if den > 0:
                logit_corrs.append(float((a * b).sum() / den))
            # L2: frozen-r decomposition. Two DIFFERENT questions, reported separately:
            #  L2a (recon_c vs true_c)  -- does the decomposition explain the ACTUAL logit?
            #      This combines SAE reconstruction error with the linearisation.
            #  L2b (recon_c vs hhat_c)  -- is the frozen-r arithmetic itself exact?
            #      sum_j phi_j + const should equal h_hat's contrasted logit at the same
            #      frozen r, up to float error. This is a BUG CANARY: it must be ~1.000
            #      regardless of how well the SAE reconstructs.
            r_scal = rms(h, eps)
            u_c = contrast_direction(W_U_act, tok_row)
            gu = g_gain * u_c
            phi_sum = float((l2[r] / r_scal) * (z[r].astype(np.float64) @ (W_dec @ gu)))
            const = float((mu[r] + b_pre_np) @ gu / r_scal)
            recon_c.append(phi_sum + const)
            true_c.append(float(rmsnorm(h, g_gain, eps) @ u_c))
            hhat_c.append(float((h_hat @ gu) / r_scal))
            n_seen += 1
        if args.gate_sample and n_seen >= args.gate_sample:
            break

    gate1 = agree / n_seen if n_seen else float("nan")
    tc, rc, hc = np.array(true_c), np.array(recon_c), np.array(hhat_c)

    def _corr(x, y):
        a = x - x.mean(); b = y - y.mean()
        den = np.sqrt((a * a).sum() * (b * b).sum())
        return float((a * b).sum() / den) if den > 0 else float("nan")

    gate2_corr = _corr(tc, rc)          # L2a: explains the ACTUAL logit
    gate2b_corr = _corr(hc, rc)         # L2b: frozen-r arithmetic exact? (bug canary)
    gate = {
        "n_decisions": n_seen,
        "L1_action_match": gate1,
        "L1_mean_logit_corr": float(np.mean(logit_corrs)) if logit_corrs else float("nan"),
        "L1_threshold": args.gate_threshold,
        "L1_pass": bool(gate1 >= args.gate_threshold),
        "L2a_vs_true_logit_corr": gate2_corr,
        "L2a_mean_abs_discrepancy": float(np.abs(tc - rc).mean()),
        "L2b_vs_reconstruction_corr": gate2b_corr,
        "L2b_mean_abs_discrepancy": float(np.abs(hc - rc).mean()),
        "L2b_is_exact": bool(gate2b_corr > 0.999),
    }
    print(f"[attr] L1 action match      : {gate1:.4f}  (threshold {args.gate_threshold})")
    print(f"[attr] L1 mean logit corr   : {gate['L1_mean_logit_corr']:.4f}")
    print(f"[attr] L2a vs TRUE logit    : corr={gate2_corr:.4f}  "
          f"mean|diff|={gate['L2a_mean_abs_discrepancy']:.4f}   "
          f"(reconstruction + linearisation)")
    print(f"[attr] L2b vs RECONSTRUCTION: corr={gate2b_corr:.4f}  "
          f"mean|diff|={gate['L2b_mean_abs_discrepancy']:.2e}   "
          f"(frozen-r arithmetic; MUST be ~1.000 -- bug canary)")
    if not gate["L2b_is_exact"]:
        print("[attr] WARNING: L2b is not ~1.000 -- the frozen-r decomposition does not "
              "reproduce its own reconstruction. That is a BUG, not a property of the data.")
    print(f"[attr] GATE {'PASS' if gate['L1_pass'] else 'FAIL'}", flush=True)
    with open(os.path.join(args.out, "gate.json"), "w") as f:
        json.dump(gate, f, indent=2)

    if not gate["L1_pass"] and not args.force:
        print("[attr] Gate failed: SAE features do not linearly explain action selection.\n"
              "[attr] Stopping (this is a reportable methods finding). Use --force to "
              "compute A4 anyway for diagnosis.", flush=True)
        return

    # ---------------- pass 2: stream, accumulate C_j(g) ---------------------
    print("[attr] === A4: causal task-breadth ===", flush=True)
    task_sum: dict[int, np.ndarray] = {}     # task -> sum |phi| over decisions [F]
    task_n: dict[int, int] = {}
    fire_count = np.zeros(F, dtype=np.int64)
    n_total = 0
    for sp in shards:
        dd = np.load(sp)
        res = dd["residual"]; toks = dd["token_ids"].astype(np.int64)
        task_of = dd["task_id"].astype(np.int64)          # [n] per decision
        n = res.shape[0]
        Xflat = res.reshape(n * 7, d).astype(np.float32)
        z, l2, mu = sae_encode_full(W_enc, b_pre_t, k, Xflat, device, args.batch_size)
        fire_count += (z > 0).sum(axis=0).astype(np.int64)
        for r in range(n * 7):
            tok = int(toks[r // 7, r % 7]); tok_row = tok - id0
            if not (0 <= tok_row < n_act):
                continue
            h = Xflat[r].astype(np.float64)
            r_scal = rms(h, eps)
            gu = g_gain * contrast_direction(W_U_act, tok_row)
            align = W_dec @ gu                              # [F]
            phi = (l2[r] / r_scal) * z[r].astype(np.float64) * align
            t = int(task_of[r // 7])
            if t not in task_sum:
                task_sum[t] = np.zeros(F, dtype=np.float64); task_n[t] = 0
            task_sum[t] += np.abs(phi)
            task_n[t] += 1
            n_total += 1
        print(f"[attr]   {os.path.basename(sp)}: cumulative decisions={n_total}", flush=True)

    task_ids = np.array(sorted(task_sum))
    C = np.stack([task_sum[t] / max(task_n[t], 1) for t in task_ids])   # [G, F]
    PR = participation_ratio(C)
    mag = total_magnitude(C)
    base_rate = fire_count / max(n_total, 1)
    active = mag > 0

    # ---------------- confound controls ------------------------------------
    rho_mag = _spearman(PR[active], mag[active])
    rho_br = _spearman(PR[active], base_rate[active])
    # leave-one-task-out: does PR on G-1 tasks predict held-out causal importance?
    loto, loto_partial = [], []
    G = len(task_ids)
    for gi in range(G):
        keep = np.arange(G) != gi
        PR_tr = participation_ratio(C[keep])
        mag_tr = total_magnitude(C[keep])
        held = C[gi]
        m = (mag_tr > 0) & np.isfinite(PR_tr)
        if m.sum() > 3:
            loto.append(_spearman(PR_tr[m], held[m]))
            loto_partial.append(_partial_spearman(PR_tr[m], held[m], mag_tr[m]))
    summary = {
        "n_decisions": n_total, "n_tasks": int(G), "n_features": int(F),
        "n_active": int(active.sum()),
        "PR_mean": float(np.nanmean(PR[active])),
        "PR_p10": float(np.nanpercentile(PR[active], 10)),
        "PR_p90": float(np.nanpercentile(PR[active], 90)),
        "spearman_PR_vs_magnitude": rho_mag,
        "spearman_PR_vs_baserate": rho_br,
        "loto_mean_spearman": float(np.mean(loto)) if loto else float("nan"),
        "loto_mean_partial_vs_magnitude": float(np.mean(loto_partial)) if loto_partial else float("nan"),
        "gate": gate,
    }
    np.savez_compressed(os.path.join(args.out, f"layer_{args.layer:02d}_attribution.npz"),
                        C=C.astype(np.float32), task_ids=task_ids,
                        PR=PR.astype(np.float32), magnitude=mag.astype(np.float32),
                        base_rate=base_rate.astype(np.float32),
                        is_active=active.astype(np.uint8))
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[attr] decisions={n_total}  tasks={G}  active features={int(active.sum())}")
    print(f"[attr] PR mean={summary['PR_mean']:.3f}  p10={summary['PR_p10']:.3f}  "
          f"p90={summary['PR_p90']:.3f}   (1 = one task, {G} = all tasks)")
    print(f"[attr] CONTROL  PR ~ causal magnitude : {rho_mag:+.3f}")
    print(f"[attr] CONTROL  PR ~ base firing rate : {rho_br:+.3f}")
    print(f"[attr] LOTO     PR -> held-out importance : {summary['loto_mean_spearman']:+.3f}"
          f"   partial (magnitude removed): {summary['loto_mean_partial_vs_magnitude']:+.3f}")
    print(f"\n[attr] Read: a POSITIVE partial LOTO means causal BREADTH predicts held-out\n"
          f"[attr] causal importance beyond how strong the feature is overall -- i.e. task-\n"
          f"[attr] breadth is a real axis, not a restatement of magnitude.", flush=True)


if __name__ == "__main__":
    main()
