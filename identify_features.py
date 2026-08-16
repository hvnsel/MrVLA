"""Path A / §3.2b: identify the features the breadth axis flags, and mine exemplar decisions.

Path A validated the breadth AXIS (partial|both = +0.493) but did not name a feature. This
script ranks features by CONFOUND-ADJUSTED breadth -- the participation ratio residualised on
(total causal magnitude, base firing rate), the same two confounds A4 controls -- so the top
of the ranking is "general beyond activity", not merely busy. It then streams the A1 shards,
recomputes each selected feature's per-decision causal contribution phi_j, and records the
decisions with the largest |phi_j| (causal exemplars) and the largest z_j (activation
exemplars). capture_feature_frames.py turns those (task, episode, timestep) locations into
image contact sheets.

No hard "general/not" cut is claimed -- the RANKING is the object. We deliberately keep both
ends: the top (general candidates) and the bottom-among-those-that-matter (specialist
candidates), so the frames can be compared.

Usage
-----
python identify_features.py \
    --acts-dir /work/.../ACT_ACTION/goal \
    --sae-dir  /work/.../ACT_ACTION_SAE/goal/sae \
    --attr     /work/.../ATTR/goal_k100/layer_31_attribution.npz \
    --layer 31 --top 8 --exemplars 12 --per-task 10 \
    --out /work/.../FEATURES/goal_k100

--per-task 10 keeps the top-10 exemplar decisions PER TASK per feature, so
capture_feature_frames.py can lay a general feature out as a grid with 10 frames from every
task (its cross-task signature), while a specialist shows filled rows for only its 1-2 tasks.
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np

from mrvla.attribution import contrast_direction, rms

# torch and the SAE encoder are imported inside main(): `adjusted_breadth` and
# `select_general_specialist` below are pure numpy and are imported by the CPU-only
# re-analysis tools (compare_recurrence_groups, join_pathA_pathB, run_ablation's coalition
# builder), which advertise themselves as needing no GPU and should not need a torch install
# to run or to be unit-tested. Same deferral run_ablation.py already uses for load_sae.


def _ranks(x: np.ndarray) -> np.ndarray:
    r = np.argsort(np.argsort(x)).astype(np.float64)
    return r - r.mean()


def adjusted_breadth(PR, magnitude, base_rate, active):
    """PR residualised on (magnitude, base rate) in rank space, per feature.

    Returns an array [F] of the confound-adjusted breadth score (NaN for inactive
    features). Higher = drives more tasks than its strength and firing rate alone predict.
    """
    F = PR.shape[0]
    out = np.full(F, np.nan)
    m = active & np.isfinite(PR) & np.isfinite(magnitude) & np.isfinite(base_rate)
    if m.sum() < 5:
        return out
    ry = _ranks(PR[m])
    C = np.stack([_ranks(magnitude[m]), _ranks(base_rate[m])], axis=1)   # [n,2]
    beta, *_ = np.linalg.lstsq(C, ry, rcond=None)
    resid = ry - C @ beta
    out[m] = resid
    return out


def select_general_specialist(adj, magnitude, active, top):
    """Pick `top` general and `top` specialist features, BOTH drawn only from load-bearing
    features (magnitude at or above the active median), so we contrast "broad and strong"
    against "narrow and strong" rather than against weak noise.

    general    = highest adjusted breadth among eligible.
    specialist = lowest  adjusted breadth among eligible.

    Returns (general_idx_list, specialist_idx_list). Both ends are restricted to the eligible
    set: we rank ONLY the eligible indices and slice both ends of that ranking, so an
    ineligible (weak / inactive) feature can never land in either list.
    """
    med_mag = np.nanmedian(magnitude[active])
    eligible = active & (magnitude >= med_mag) & np.isfinite(adj)
    elig_idx = np.where(eligible)[0]
    elig_order = elig_idx[np.argsort(adj[elig_idx])]        # eligible only, ascending by adj
    general = elig_order[::-1][:top].tolist()               # highest adjusted breadth
    specialist = elig_order[:top].tolist()                  # lowest adjusted breadth
    return general, specialist


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--acts-dir", required=True, help="A1 output (residual shards + head_constants)")
    p.add_argument("--sae-dir", required=True)
    p.add_argument("--attr", required=True, help="run_attribution's layer_NN_attribution.npz")
    p.add_argument("--layer", type=int, default=31)
    p.add_argument("--top", type=int, default=8, help="how many general + specialist features")
    p.add_argument("--exemplars", type=int, default=12, help="overall exemplar decisions per feature")
    p.add_argument("--per-task", type=int, default=10,
                   help="top-|phi| exemplar decisions to keep PER TASK per feature (for the "
                        "per-task grid: a general feature shows this many from every task)")
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--batch-size", type=int, default=4096)
    args = p.parse_args()

    import torch

    from run_attribution import load_sae, sae_encode_full

    device = args.device if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out, exist_ok=True)

    # --- attribution summary: pick the features -----------------------------
    A = np.load(args.attr)
    PR = A["PR"].astype(np.float64)
    magnitude = A["magnitude"].astype(np.float64)
    base_rate = A["base_rate"].astype(np.float64)
    active = A["is_active"].astype(bool)
    adj = adjusted_breadth(PR, magnitude, base_rate, active)

    general, specialist = select_general_specialist(adj, magnitude, active, args.top)
    selected = general + specialist
    role = {j: "general" for j in general}
    role.update({j: "specialist" for j in specialist})
    print(f"[id] selected {len(general)} general + {len(specialist)} specialist features")
    for j in selected:
        print(f"[id]   feat {j:5d}  role={role[j]:10s} PR={PR[j]:.2f} "
              f"mag={magnitude[j]:.3e} base_rate={base_rate[j]:.3f} adj_breadth={adj[j]:+.1f}")

    # --- head constants + SAE ------------------------------------------------
    hc = np.load(os.path.join(args.acts_dir, "head_constants.npz"))
    W_U_act = hc["W_U_act"].astype(np.float64)
    g_gain = hc["g"].astype(np.float64)
    eps = float(hc["eps"])
    act_ids = hc["act_ids"].astype(np.int64)
    id0 = int(act_ids[0]); n_act, d = W_U_act.shape

    W_enc, W_dec_t, b_pre_t, k, ck_path = load_sae(args.sae_dir, args.layer)
    W_dec = W_dec_t.detach().float().cpu().numpy().astype(np.float64)     # [F, d]
    sel = np.array(selected, dtype=np.int64)
    W_dec_sel = W_dec[sel]                                                # [S, d]

    # --- stream shards, keep exemplars per selected feature -----------------
    # For each selected feature we keep three running collections of records
    # (score, task, episode, timestep, z, phi):
    #   phi_ex[s]     -- overall top-|phi|            (compact view / specialists)
    #   z_ex[s]       -- overall top-z                (what activates it)
    #   pt_ex[s][g]   -- top-|phi| WITHIN task g      (the per-task grid; a general
    #                    feature fills all G tasks, a specialist only 1-2)
    S = len(sel)
    phi_ex = [[] for _ in range(S)]
    z_ex = [[] for _ in range(S)]
    pt_ex = [dict() for _ in range(S)]
    shards = sorted(glob.glob(os.path.join(args.acts_dir, "shard_*.npz")))
    if not shards:
        raise FileNotFoundError(f"no shard_*.npz in {args.acts_dir}")

    def trim(lst, n):
        lst.sort(key=lambda r: -r[0])
        del lst[n:]

    for sp in shards:
        dd = np.load(sp)
        res = dd["residual"]; toks = dd["token_ids"].astype(np.int64)
        task_of = dd["task_id"].astype(np.int64)
        ep_of = dd["episode"].astype(np.int64)
        ts_of = dd["timestep"].astype(np.int64)
        n = res.shape[0]
        Xflat = res.reshape(n * 7, d).astype(np.float32)
        z, l2, mu = sae_encode_full(W_enc, b_pre_t, k, Xflat, device, args.batch_size)
        z_sel = z[:, sel]                                                # [n*7, S]
        for r in range(n * 7):
            di, pos = r // 7, r % 7
            tok = int(toks[di, pos]); tok_row = tok - id0
            if not (0 <= tok_row < n_act):
                continue
            zr = z_sel[r]
            if not zr.any():
                continue
            h = Xflat[r].astype(np.float64)
            r_scal = rms(h, eps)
            gu = g_gain * contrast_direction(W_U_act, tok_row)
            align = W_dec_sel @ gu                                       # [S]
            phi = (l2[r] / r_scal) * zr.astype(np.float64) * align       # [S]
            task, ep, ts = int(task_of[di]), int(ep_of[di]), int(ts_of[di])
            for s in range(S):
                if zr[s] > 0:
                    ap = abs(float(phi[s]))
                    rec = (ap, task, ep, ts, float(zr[s]), float(phi[s]))
                    phi_ex[s].append(rec)
                    z_ex[s].append((float(zr[s]),) + rec[1:])
                    pt_ex[s].setdefault(task, []).append(rec)
        for s in range(S):
            trim(phi_ex[s], args.exemplars); trim(z_ex[s], args.exemplars)
            for g in pt_ex[s]:
                trim(pt_ex[s][g], args.per_task)
        print(f"[id]   {os.path.basename(sp)} scanned", flush=True)

    # --- write ---------------------------------------------------------------
    def pack(lst):
        return [{"task": t, "episode": e, "timestep": ts, "z": zz, "phi": ph}
                for (_score, t, e, ts, zz, ph) in lst]

    out = {"layer": args.layer, "sae": ck_path, "top": args.top,
           "exemplars": args.exemplars, "per_task": args.per_task, "features": []}
    for s, j in enumerate(sel):
        per_task = {str(g): pack(pt_ex[s][g]) for g in sorted(pt_ex[s])}
        out["features"].append({
            "feature": int(j), "role": role[int(j)],
            "PR": float(PR[j]), "magnitude": float(magnitude[j]),
            "base_rate": float(base_rate[j]), "adjusted_breadth": float(adj[j]),
            "n_tasks_fired": len(per_task),
            "top_by_phi": pack(phi_ex[s]),
            "top_by_activation": pack(z_ex[s]),
            "per_task_by_phi": per_task,
        })
    with open(os.path.join(args.out, "exemplars.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"[id] wrote {os.path.join(args.out, 'exemplars.json')}  "
          f"({len(sel)} features x {args.exemplars} exemplars)")
    print(f"[id] next: python capture_feature_frames.py --acts-dir {args.acts_dir} "
          f"--exemplars {os.path.join(args.out, 'exemplars.json')} "
          f"--task-suite <suite> --out {args.out}/frames")


if __name__ == "__main__":
    main()
