"""Additive feature steering on LIBERO: amplify one SAE feature and watch what the robot does.

WHY STEERING RATHER THAN ABLATION
Ablation asks whether a feature is NECESSARY. With a TopK k=100 code, deleting one feature of
~100 active ones removes ~1% of the reconstruction, and our single-feature ablations came back
flat against a healthy 78.9% baseline. Steering asks whether a feature is SUFFICIENT to drive
behaviour -- add alpha*w to the residual stream and see whether the predicted behaviour
appears. It is a far stronger intervention, and it is the one prior work has shown moves VLA
behaviour in this exact setting.

WHAT COMES OUT
  * success rate per (condition, task)  -- free, it is just the env's done flag; and
  * an episode VIDEO for the first --video-episodes episodes of each (condition, task), so the
    behavioural signature is directly watchable. Baseline and every steered condition replay
    the SAME init states, so videos line up frame-for-frame for side-by-side comparison.

CHOOSING ALPHA
Decoder rows are unit-norm, so alpha is in units of residual-stream norm -- which is
model- and layer-specific. An alpha borrowed from another paper's model and layer means
nothing here. Calibrate first:

    python run_steering.py ... --gamma 1.0 --calibrate-only

prints the residual scale and the implied alpha. Then pin it with --alpha for the real run, so
every worker applies the IDENTICAL intervention (with --gamma each shard would measure its own
scale and they would drift apart).

Usage
-----
# 1. calibrate (seconds, one forward pass)
python run_steering.py --model openvla/openvla-7b-finetuned-libero-goal \\
  --task-suite libero_goal --unnorm-key libero_goal \\
  --sae-dir $B/ACT_ACTION_SAE/goal/sae --features 1167 \\
  --gamma 1.0 --calibrate-only --out $B/STEER/goal

# 2. run, 7 workers (baseline + 6 steered conditions)
for i in 0 1 2 3 4 5 6; do
  python run_steering.py --model openvla/openvla-7b-finetuned-libero-goal \\
    --task-suite libero_goal --unnorm-key libero_goal \\
    --sae-dir $B/ACT_ACTION_SAE/goal/sae \\
    --features 1167,1235,1628,1999,1140,1134 \\
    --alpha 42.0 --episodes-per-task 10 --video-episodes 2 \\
    --worker-id $i --n-workers 7 --device cuda:0 \\
    --out $B/STEER/goal &
done; wait
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

# NOTE: torch / LIBERO / transformers are imported lazily inside functions so this module
# stays importable (and testable) on a CPU box without the cluster stack.

_DUMMY_ACTION = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]
_NUM_STEPS_WAIT = 10
_SUITE_MAX_STEPS = {"libero_spatial": 220, "libero_object": 280,
                    "libero_goal": 300, "libero_10": 520, "libero_90": 400}


def parse_features(spec: str) -> list[int]:
    """'1167,1235' -> [1167, 1235]. Rejects empties and duplicates: a duplicated id would
    silently double that feature's steering magnitude."""
    ids = [int(x) for x in (spec or "").split(",") if x.strip()]
    if not ids:
        raise SystemExit("--features: no feature ids given")
    if len(set(ids)) != len(ids):
        raise SystemExit(f"--features: duplicate ids in {ids}")
    return ids


def build_conditions(features: list[int], n_random: int = 0) -> dict:
    """{condition_name: description}. baseline first, then one condition per steered feature.

    `n_random` adds norm-matched RANDOM-direction controls. Off by default, but worth one
    condition: without it, "any large perturbation degrades behaviour" is an unfalsified
    explanation for every steering result -- the same hole the firing-rate ablation fell into.
    """
    conds = {"baseline": {"kind": "baseline", "features": []}}
    for j in features:
        conds[f"steer_{j}"] = {"kind": "feature", "features": [int(j)]}
    for r in range(n_random):
        conds[f"steer_random{r}"] = {"kind": "random", "features": [], "random_seed": r}
    return conds


def direction_for(cond: dict, W_dec: np.ndarray, seed: int):
    """Unit-norm [1, d] direction to add for a condition, or None for baseline."""
    if cond["kind"] == "baseline":
        return None
    if cond["kind"] == "feature":
        v = W_dec[cond["features"][0]][None, :]
    else:
        rng = np.random.default_rng(1000 + seed + cond.get("random_seed", 0))
        v = rng.standard_normal((1, W_dec.shape[1]))
    v = v / max(float(np.linalg.norm(v)), 1e-8)
    return v.astype(np.float32)


def shard(jobs: list, worker_id: int, n_workers: int) -> list:
    return [j for i, j in enumerate(jobs) if i % n_workers == worker_id]


def run_episode(model, processor, env, init_state, instruction, device, unnorm_key,
                max_steps: int, steerer=None, capture: bool = False,
                center_crop: bool = True):
    """One episode. Returns (success, n_steps, frames) where frames is [] unless `capture`."""
    import torch
    from mrvla.libero_collect import (
        _get_libero_image, _invert_gripper_action, _normalize_gripper_action)
    from mrvla.model_utils import build_inputs
    torch.set_grad_enabled(False)

    env.reset()
    obs = env.set_init_state(init_state)
    frames: list = []
    success, step = False, 0
    while step < max_steps + _NUM_STEPS_WAIT:
        if step < _NUM_STEPS_WAIT:
            obs, _r, done, _i = env.step(_DUMMY_ACTION)
            step += 1
            continue
        if capture:
            # raw agentview, flipped upright: what a human wants to watch, not the
            # centre-cropped 224 tensor the model consumes
            frames.append(np.asarray(obs["agentview_image"])[::-1].copy())
        image = _get_libero_image(obs, center_crop=center_crop)
        inputs = build_inputs(processor, image, instruction, device)
        if steerer is not None:
            steerer.reset_step()
        action = model.predict_action(**inputs, unnorm_key=unnorm_key, do_sample=False)
        action = _invert_gripper_action(_normalize_gripper_action(action, binarize=True))
        obs, _r, done, _i = env.step(action.tolist())
        step += 1
        if done:
            success = True
            break
    return success, step, frames


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True)
    p.add_argument("--task-suite", required=True)
    p.add_argument("--unnorm-key", required=True)
    p.add_argument("--sae-dir", required=True)
    p.add_argument("--layer", type=int, default=31)
    p.add_argument("--features", required=True, help="comma list of feature ids to steer")
    p.add_argument("--alpha", type=float, default=None,
                   help="absolute steering magnitude (PREFER for multi-worker runs)")
    p.add_argument("--gamma", type=float, default=None,
                   help="relative magnitude: alpha = gamma * mean||h||, measured on the "
                        "first pass. Use with --calibrate-only, then pin --alpha.")
    p.add_argument("--calibrate-only", action="store_true",
                   help="run one inference step, print the residual scale and implied alpha, "
                        "and exit without rollouts")
    p.add_argument("--episodes-per-task", type=int, default=10)
    p.add_argument("--video-episodes", type=int, default=2,
                   help="record video for the first N episodes of each (condition, task)")
    p.add_argument("--video-fps", type=int, default=10)
    p.add_argument("--video-stride", type=int, default=2)
    p.add_argument("--video-max-side", type=int, default=128)
    p.add_argument("--random-controls", type=int, default=0,
                   help="norm-matched random-direction control conditions (recommended: 1)")
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--worker-id", type=int, default=0)
    p.add_argument("--n-workers", type=int, default=1)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--no-flash-attn", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    if args.alpha is None and args.gamma is None:
        raise SystemExit("give --alpha (absolute) or --gamma (relative, with --calibrate-only)")

    import torch
    from mrvla.hooks import ActivationSteerer
    from mrvla.model_utils import load_openvla, locate_decoder_layers
    from mrvla.video import write_video
    from run_attribution import load_sae

    os.makedirs(args.out, exist_ok=True)
    device = args.device if torch.cuda.is_available() else "cpu"
    max_steps = args.max_steps or _SUITE_MAX_STEPS.get(args.task_suite, 300)

    features = parse_features(args.features)
    conds = build_conditions(features, args.random_controls)

    _W_enc, W_dec_t, _b_pre, _k, _ck = load_sae(args.sae_dir, args.layer)
    W_dec = W_dec_t.detach().float().cpu().numpy()
    F, d = W_dec.shape
    bad = [j for j in features if not (0 <= j < F)]
    if bad:
        raise SystemExit(f"feature ids out of range for dictionary of {F}: {bad}")
    print(f"[steer] SAE {args.sae_dir} layer {args.layer}  W_dec {W_dec.shape}", flush=True)

    if args.worker_id == 0:
        with open(os.path.join(args.out, "manifest.json"), "w") as f:
            json.dump({"model": args.model, "task_suite": args.task_suite,
                       "layer": args.layer, "features": features,
                       "alpha": args.alpha, "gamma": args.gamma,
                       "episodes_per_task": args.episodes_per_task,
                       "video_episodes": args.video_episodes,
                       "conditions": {k: v for k, v in conds.items()},
                       "max_steps": max_steps, "seed": args.seed}, f, indent=2)

    print(f"[steer] loading {args.model} on {device}", flush=True)
    model, processor = load_openvla(args.model, device=device,
                                    use_flash_attn=not args.no_flash_attn)
    layers = locate_decoder_layers(model)
    layer_module = layers[args.layer]

    # torch >= 2.6 rejects LIBERO's pickled numpy init states under weights_only=True
    from mrvla.torch_compat import allow_numpy_pickles
    allow_numpy_pickles()
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    task_suite = benchmark.get_benchmark_dict()[args.task_suite]()
    n_tasks = task_suite.n_tasks

    # ---- calibration: one episode-free forward pass to size alpha ------------------
    if args.calibrate_only:
        task = task_suite.get_task(0)
        bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
        env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=256, camera_widths=256)
        env.seed(args.seed)
        init_states = task_suite.get_task_init_states(0)
        probe = ActivationSteerer(layer_module,
                                  torch.from_numpy(W_dec[features[0]][None, :]),
                                  gamma=args.gamma or 1.0)
        try:
            run_episode(model, processor, env, init_states[0], task.language, device,
                        args.unnorm_key, max_steps=1)
            scale = probe.resolved_scale
            alphas = probe.resolved_alphas
        finally:
            probe.remove(); env.close()
        a = float(alphas[0]) if alphas is not None else float("nan")
        print(f"\n[steer] mean residual norm ||h|| = {scale:.3f}")
        print(f"[steer] gamma={args.gamma or 1.0}  ->  alpha = {a:.3f}")
        print(f"[steer] re-run the real job with:  --alpha {a:.3f}")
        print("[steer] (pin alpha so every worker applies the identical intervention)")
        return

    jobs = [(c, t) for c in conds for t in range(n_tasks)]
    mine = shard(jobs, args.worker_id, args.n_workers)
    print(f"[steer] worker {args.worker_id}/{args.n_workers}: {len(mine)}/{len(jobs)} jobs "
          f"({len(conds)} conditions x {n_tasks} tasks)", flush=True)
    if not mine:
        return

    vid_dir = os.path.join(args.out, "videos")
    os.makedirs(vid_dir, exist_ok=True)
    out_path = os.path.join(args.out, f"results_w{args.worker_id:02d}.json")
    results: list = []
    backend_seen = None

    by_task: dict[int, list[str]] = {}
    for c, t in mine:
        by_task.setdefault(t, []).append(c)

    for task_id, task_conds in sorted(by_task.items()):
        task = task_suite.get_task(task_id)
        instruction = task.language
        bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
        print(f"[steer] task {task_id}: building env", flush=True)
        env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=256, camera_widths=256)
        env.seed(args.seed)
        init_states = task_suite.get_task_init_states(task_id)
        n_ep = min(args.episodes_per_task, len(init_states))

        for cond_name in task_conds:
            cond = conds[cond_name]
            v = direction_for(cond, W_dec, args.seed)
            steerer = None
            if v is not None:
                steerer = ActivationSteerer(
                    layer_module, torch.from_numpy(v),
                    alphas=torch.tensor([args.alpha]) if args.alpha is not None else None,
                    gamma=args.gamma)
            try:
                n_succ = 0
                for ep in range(n_ep):        # SAME init states for every condition
                    capture = ep < args.video_episodes
                    ok, steps, frames = run_episode(
                        model, processor, env, init_states[ep], instruction, device,
                        args.unnorm_key, max_steps, steerer=steerer, capture=capture)
                    n_succ += int(ok)
                    rec = {"condition": cond_name, "task_id": task_id, "episode": ep,
                           "success": int(ok), "steps": int(steps)}
                    if capture and frames:
                        stem = os.path.join(vid_dir, f"t{task_id:02d}_e{ep:02d}_{cond_name}")
                        try:
                            path, backend = write_video(
                                frames, stem, fps=args.video_fps, stride=args.video_stride,
                                max_side=args.video_max_side)
                            rec["video"] = os.path.basename(path)
                            if backend != backend_seen:
                                backend_seen = backend
                                print(f"[steer] video backend: {backend}", flush=True)
                        except Exception as e:      # never lose a rollout to a video failure
                            print(f"[steer] WARNING video failed ({e})", flush=True)
                    results.append(rec)
                print(f"[steer] task {task_id:2d} · {cond_name:18s} : {n_succ}/{n_ep} "
                      f"({100.0 * n_succ / max(n_ep, 1):.0f}%)", flush=True)
            finally:
                if steerer is not None:
                    steerer.remove()
            with open(out_path, "w") as f:          # checkpoint after every condition
                json.dump(results, f)
        env.close()

    with open(out_path, "w") as f:
        json.dump(results, f)
    print(f"[steer] wrote {out_path} and {len(os.listdir(vid_dir))} videos", flush=True)


if __name__ == "__main__":
    main()
