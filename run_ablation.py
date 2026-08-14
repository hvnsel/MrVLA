"""Path A / behavioural: closed-loop ablation of feature coalitions, success rate per task.

THE TEST. Path A says causal task-breadth is a real axis. The behavioural prediction is not
just "general features matter" but a statement about SCOPE:

    ablating a GENERAL coalition   -> success degrades across MANY tasks
    ablating a SPECIALIST coalition-> success degrades in ITS OWN tasks, ~nowhere else
    ablating a RANDOM coalition    -> the null: how much does removing *any* k features hurt?
    ablating a FIRING-ranked one   -> the head-to-head vs the prior work's activity metric

so we log success PER TASK for every condition, on the SAME initial states, and compare the
damage profile to each coalition's own per-task causal profile C_j(g).

Conditions (choose with --conditions, default all five):
  baseline     no ablation (the ceiling; required to interpret everything else)
  general      top-N by confound-adjusted breadth
  specialist   bottom-N by confound-adjusted breadth, among load-bearing features
  random       N random load-bearing features (matched pool; the null)
  firing       top-N by base firing rate (the prior work's proxy; the head-to-head)
Plus:
  --features 12,438,901   ablate exactly these (a named coalition, or one feature)
  --individual            in addition, ablate each selected feature ON ITS OWN

Ablation = project the features' decoder directions out of the layer-31 residual on every
forward pass, for the whole episode (mrvla.hooks.ActivationAblator).

Parallelism: the job list is (condition x task); shard it across GPUs with
--worker-id/--n-workers, one process per GPU, then merge with analyze_ablation.py.

Usage (4 GPUs)
--------------
for i in 0 1 2 3; do
  python run_ablation.py --model openvla/openvla-7b-finetuned-libero-goal \\
    --task-suite libero_goal --unnorm-key libero_goal \\
    --sae-dir  $BASE/ACT_ACTION_SAE/goal/sae \\
    --attr     $BASE/ATTR/goal_k100/layer_31_attribution.npz \\
    --episodes-per-task 20 --top 5 \\
    --worker-id $i --n-workers 4 --device cuda:$i \\
    --out $BASE/ABLATION/goal &
done; wait
python analyze_ablation.py --dir $BASE/ABLATION/goal
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

from identify_features import adjusted_breadth, select_general_specialist

# NOTE: mrvla.libero_collect / mrvla.model_utils / mrvla.hooks / run_attribution are imported
# lazily inside the functions that need them, so build_coalitions() (pure numpy) stays
# importable and testable without the LIBERO + torch/transformers stack.

_DUMMY_ACTION = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]
_NUM_STEPS_WAIT = 10
_SUITE_MAX_STEPS = {"libero_spatial": 220, "libero_object": 280,
                    "libero_goal": 300, "libero_10": 520, "libero_90": 400}


# ---------------------------------------------------------------------------
# Coalition construction
# ---------------------------------------------------------------------------
def build_coalitions(attr_path: str, top: int, seed: int = 0) -> dict:
    """Return {condition_name: [feature indices]} plus per-feature bookkeeping.

    All coalitions are drawn from the LOAD-BEARING pool (magnitude >= active median) so that
    'general' and 'specialist' differ in BREADTH, not in whether they matter at all. We record
    each coalition's total magnitude so the analysis can check the two are comparable --
    otherwise "generals hurt more" could just mean "we picked stronger features".
    """
    A = np.load(attr_path)
    PR = A["PR"].astype(np.float64)
    magnitude = A["magnitude"].astype(np.float64)
    base_rate = A["base_rate"].astype(np.float64)
    active = A["is_active"].astype(bool)
    C = A["C"].astype(np.float64)                      # [G, F] per-task causal importance
    adj = adjusted_breadth(PR, magnitude, base_rate, active)

    general, specialist = select_general_specialist(adj, magnitude, active, top)

    med_mag = np.nanmedian(magnitude[active])
    pool = np.where(active & (magnitude >= med_mag) & np.isfinite(adj))[0]
    rng = np.random.default_rng(seed)
    chosen = set(general) | set(specialist)
    cand = np.array([j for j in pool if j not in chosen])
    random_coalition = rng.choice(cand, size=min(top, len(cand)), replace=False).tolist()

    # the prior work's proxy: rank by raw base firing rate (activity), among active features
    fr = np.where(active, base_rate, -np.inf)
    firing = np.argsort(fr)[::-1][:top].tolist()

    coalitions = {"general": general, "specialist": specialist,
                  "random": random_coalition, "firing": firing}

    def info(idx):
        return {
            "features": [int(j) for j in idx],
            "PR": [float(PR[j]) for j in idx],
            "magnitude": [float(magnitude[j]) for j in idx],
            "base_rate": [float(base_rate[j]) for j in idx],
            "adjusted_breadth": [float(adj[j]) for j in idx],
            "total_magnitude": float(sum(magnitude[j] for j in idx)),
            # predicted damage profile: where this coalition does its causal work
            "per_task_profile": [float(v) for v in C[:, idx].sum(axis=1)],
        }

    return {"coalitions": coalitions,
            "info": {k: info(v) for k, v in coalitions.items()},
            "n_tasks": int(C.shape[0])}


def parse_feature_specs(specs, ablate_each) -> dict:
    """Turn --features / --ablate-each into {condition_name: [feature ids]}.

    --features accepts "name=1,2,3" (a named set) or a bare "1,2,3" (named "custom"), and is
    repeatable, so several sets can be compared in ONE run against the same baseline on
    identical init states -- which is a genuinely paired comparison, unlike running them as
    separate jobs.

    --ablate-each takes a bare id list and makes one SINGLETON condition per id ("only_<id>").
    That is the right shape when each feature carries its own prediction about which task it
    should damage: a coalition confounds the features with each other and cannot be read
    per-feature.
    """
    out: dict[str, list[int]] = {}
    for spec in (specs or []):
        name, sep, ids = spec.partition("=")
        if not sep:
            name, ids = "custom", spec
        name = name.strip()
        feats = [int(x) for x in ids.split(",") if x.strip()]
        if not name:
            raise SystemExit(f"--features {spec!r}: empty condition name")
        if not feats:
            raise SystemExit(f"--features {spec!r}: no feature ids")
        if name in out:
            raise SystemExit(f"--features: duplicate condition name {name!r}")
        out[name] = feats
    for tok in (ablate_each or "").split(","):
        tok = tok.strip()
        if not tok:
            continue
        name = f"only_{int(tok)}"
        if name in out:
            raise SystemExit(f"--ablate-each: {name!r} already defined by --features")
        out[name] = [int(tok)]
    return out


# ---------------------------------------------------------------------------
# One closed-loop episode
# ---------------------------------------------------------------------------
def run_episode(model, processor, env, init_state, instruction, device, unnorm_key,
                max_steps: int, center_crop: bool = True):
    """Roll out one episode; return (success, n_steps)."""
    import torch
    from mrvla.libero_collect import (
        _get_libero_image, _invert_gripper_action, _normalize_gripper_action)
    from mrvla.model_utils import build_inputs
    torch.set_grad_enabled(False)
    env.reset()
    obs = env.set_init_state(init_state)
    success, step = False, 0
    while step < max_steps + _NUM_STEPS_WAIT:
        if step < _NUM_STEPS_WAIT:
            obs, _r, done, _i = env.step(_DUMMY_ACTION)
            step += 1
            continue
        image = _get_libero_image(obs, center_crop=center_crop)
        inputs = build_inputs(processor, image, instruction, device)
        action = model.predict_action(**inputs, unnorm_key=unnorm_key, do_sample=False)
        action = _invert_gripper_action(_normalize_gripper_action(action, binarize=True))
        obs, _r, done, _i = env.step(action.tolist())
        step += 1
        if done:
            success = True
            break
    return success, step


def make_ablator(layer_module, W_dec: np.ndarray, feature_idx, device: str):
    """ActivationAblator that projects out exactly `feature_idx` on every forward pass.

    Deterministic mode with a 0/1 indicator selects an explicit feature set (threshold 0.5).
    ablate_decode_passes=True so ALL 7 action-token passes are ablated, not just the prefill --
    our features live at the action positions.
    """
    import torch
    from mrvla.hooks import ActivationAblator
    F = W_dec.shape[0]
    ind = torch.zeros(F, dtype=torch.float32)
    ind[list(feature_idx)] = 1.0
    return ActivationAblator(
        layer=layer_module,
        decoder_dirs=torch.from_numpy(W_dec.astype(np.float32)),
        prob_memorized=ind,
        mode="deterministic",
        threshold=0.5,
        ablate_decode_passes=True,
    )


# ---------------------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True)
    p.add_argument("--task-suite", required=True)
    p.add_argument("--unnorm-key", required=True)
    p.add_argument("--sae-dir", required=True)
    p.add_argument("--attr", required=True, help="layer_NN_attribution.npz from run_attribution")
    p.add_argument("--layer", type=int, default=31)
    p.add_argument("--top", type=int, default=5, help="coalition size N")
    p.add_argument("--episodes-per-task", type=int, default=20)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--conditions", default="baseline,general,specialist,random,firing",
                   help="comma list; baseline is strongly recommended (it is the ceiling)")
    p.add_argument("--features", action="append", default=None,
                   help="NAME=id,id,... (or a bare id list -> 'custom'). Repeatable, so "
                        "several named sets run against one baseline on the same init states.")
    p.add_argument("--ablate-each", default=None,
                   help="comma list of feature ids; ablates each ONE ON ITS OWN as condition "
                        "only_<id>. Use when each feature has its own per-task prediction.")
    p.add_argument("--individual", action="store_true",
                   help="also ablate each general/specialist feature on its own")
    p.add_argument("--worker-id", type=int, default=0)
    p.add_argument("--n-workers", type=int, default=1)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--no-flash-attn", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    import torch
    from mrvla.model_utils import load_openvla, locate_decoder_layers
    from run_attribution import load_sae

    os.makedirs(args.out, exist_ok=True)
    device = args.device if torch.cuda.is_available() else "cpu"
    max_steps = args.max_steps or _SUITE_MAX_STEPS.get(args.task_suite, 300)

    built = build_coalitions(args.attr, args.top, seed=args.seed)
    coalitions = dict(built["coalitions"])
    user_sets = parse_feature_specs(args.features, args.ablate_each)
    coalitions.update(user_sets)
    if args.individual:
        for role in ("general", "specialist"):
            for j in built["coalitions"][role]:
                coalitions[f"{role}_only_{j}"] = [int(j)]

    wanted = [c.strip() for c in args.conditions.split(",") if c.strip()]
    wanted += list(user_sets)
    if args.individual:
        wanted += [k for k in coalitions if "_only_" in k]
    # de-duplicate, preserving order: a user-named set that also appears in --conditions must
    # not be scheduled (and billed for GPU time) twice.
    seen: set[str] = set()
    conditions = []
    for c in wanted:
        if c in seen or not (c == "baseline" or c in coalitions):
            continue
        seen.add(c)
        conditions.append(c)

    # manifest (worker 0 only, so all workers agree on one file)
    if args.worker_id == 0:
        with open(os.path.join(args.out, "manifest.json"), "w") as f:
            json.dump({"model": args.model, "task_suite": args.task_suite,
                       "layer": args.layer, "top": args.top,
                       "episodes_per_task": args.episodes_per_task,
                       "conditions": conditions, "coalitions": coalitions,
                       "info": built["info"], "n_tasks": built["n_tasks"],
                       "max_steps": max_steps, "seed": args.seed}, f, indent=2)

    # torch >= 2.6 defaults torch.load to weights_only=True, which rejects LIBERO's pickled
    # numpy init-state files inside benchmark.get_task_init_states. Allowlist the numpy
    # symbols an ndarray pickle needs (process-wide) rather than disabling the check for
    # every checkpoint this process loads. Must run BEFORE the benchmark is constructed.
    from mrvla.torch_compat import allow_numpy_pickles
    allow_numpy_pickles()

    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    task_suite = benchmark.get_benchmark_dict()[args.task_suite]()
    n_tasks = task_suite.n_tasks

    # ---- job list: (condition, task). Shard round-robin across workers. -----
    jobs = [(c, t) for c in conditions for t in range(n_tasks)]
    mine = [j for i, j in enumerate(jobs) if i % args.n_workers == args.worker_id]
    print(f"[abl] worker {args.worker_id}/{args.n_workers}: {len(mine)}/{len(jobs)} jobs "
          f"({len(conditions)} conditions x {n_tasks} tasks)", flush=True)
    if not mine:
        return

    print(f"[abl] loading {args.model} on {device}", flush=True)
    model, processor = load_openvla(args.model, device=device,
                                    use_flash_attn=not args.no_flash_attn)
    layers = locate_decoder_layers(model)
    layer_module = layers[args.layer]
    _We, W_dec_t, _b, _k, ck = load_sae(args.sae_dir, args.layer)
    W_dec = W_dec_t.detach().float().cpu().numpy()      # [F, d]
    print(f"[abl] SAE {ck}  W_dec {W_dec.shape}", flush=True)

    results = []
    out_path = os.path.join(args.out, f"results_w{args.worker_id:02d}.json")

    # group jobs by task so each env is built once per task
    by_task: dict[int, list[str]] = {}
    for c, t in mine:
        by_task.setdefault(t, []).append(c)

    for task_id, conds in sorted(by_task.items()):
        task = task_suite.get_task(task_id)
        instruction = task.language
        bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
        env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=256, camera_widths=256)
        env.seed(args.seed)
        init_states = task_suite.get_task_init_states(task_id)
        n_ep = min(args.episodes_per_task, len(init_states))

        for cond in conds:
            ablator = None
            if cond != "baseline":
                ablator = make_ablator(layer_module, W_dec, coalitions[cond], device)
            try:
                n_succ = 0
                for ep in range(n_ep):          # SAME init states for every condition
                    if ablator is not None:
                        ablator.reset_step()
                    ok, steps = run_episode(model, processor, env, init_states[ep],
                                            instruction, device, args.unnorm_key, max_steps)
                    n_succ += int(ok)
                    results.append({"condition": cond, "task_id": task_id, "episode": ep,
                                    "success": int(ok), "steps": int(steps)})
                print(f"[abl] task {task_id:2d} · {cond:22s} : {n_succ}/{n_ep} "
                      f"({100.0*n_succ/max(n_ep,1):.0f}%)", flush=True)
            finally:
                if ablator is not None:
                    ablator.remove()
            with open(out_path, "w") as f:      # checkpoint after every condition
                json.dump(results, f)
        env.close()

    with open(out_path, "w") as f:
        json.dump(results, f)
    print(f"[abl] worker {args.worker_id} done -> {out_path}  ({len(results)} episodes)")


if __name__ == "__main__":
    main()
