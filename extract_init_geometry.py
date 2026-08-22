"""Read the INITIAL gripper and object positions out of LIBERO's init states. No rollouts.

`share` predicts how long a successful episode takes (+0.363, all ten tasks agreeing), but
collapses to +0.127 at 7/10 once controlled on the robot's early motion. That control is
suspect in a specific way: it is computed from the COMMANDED ACTIONS, which are the policy's
own behaviour. If the chain runs `share -> motion -> duration` then motion is a MEDIATOR, and
partialling it out deletes the path the effect travels along rather than a confound.

What we actually need is the initial geometry: where the gripper and the objects start. That
is fixed before the policy acts, so it cannot be a mediator, and it is the real content of
the "the object was just further away" objection.

It needs no rollouts. Resetting each env to each stored init state and reading body positions
is pure simulator, no model inference -- 500 states in well under a minute once the envs are
built.

WHICH OBJECT MATTERS. Rather than parse each task's BDDL goal to find the target, this
records the distance to the NEAREST movable object and the MEAN over all of them. Whichever
object the task is about, it is one of these, and the two bracket it. That trades a little
precision for not having to be right about ten task definitions.

EPISODE IDS MUST MATCH THE ROLLOUTS. The id is recomputed with the same rule
`collect_action_rollouts` uses -- cumulative `min(trials_per_task, len(init_states))` over
preceding tasks, plus the trial index -- so pass the SAME --trials-per-task the collection
used, or the join silently pairs each episode with another episode's geometry.

Usage
-----
python extract_init_geometry.py --task-suite libero_goal --trials-per-task 50 \\
                                --out $B/ROLLOUT_ACTION/goal/init_geometry.npz

Sanity-check on one task first -- this cannot be unit-tested without the simulator:
    ... --max-tasks 1 --trials-per-task 3
and read the printed body names. If the object list is empty or contains robot links, the
discovery fallbacks below picked wrong and every distance is meaningless.
"""

from __future__ import annotations

import argparse
import os

import numpy as np


def _eef_xpos(rs_env):
    """Gripper position, trying robosuite's conventions in order of directness."""
    for get in (lambda: np.asarray(rs_env._eef_xpos, dtype=np.float64),
                lambda: np.asarray(rs_env.sim.data.get_site_xpos("gripper0_grip_site"),
                                   dtype=np.float64),
                lambda: np.asarray(rs_env.robots[0]._hand_pos, dtype=np.float64)):
        try:
            v = get()
            if v is not None and v.shape == (3,):
                return v
        except Exception:
            continue
    raise RuntimeError("could not locate the end-effector position on this env")


def _object_bodies(rs_env) -> dict:
    """{name: xyz} for the movable objects, by whichever discovery route works.

    Printed by the caller on the first task so a wrong pick is visible immediately rather
    than turning into a column of meaningless distances.
    """
    sim = rs_env.sim
    out: dict = {}
    try:                                    # robosuite keeps the task's objects here
        for obj in rs_env.objects:
            body = getattr(obj, "root_body", None)
            if body:
                out[body] = np.asarray(sim.data.body_xpos[sim.model.body_name2id(body)],
                                       dtype=np.float64)
    except Exception:
        pass
    if out:
        return out
    try:                                    # robosuite names object bodies "<thing>_main"
        for name in sim.model.body_names:
            if name.endswith("_main") and not name.startswith("robot"):
                out[name] = np.asarray(sim.data.body_xpos[sim.model.body_name2id(name)],
                                       dtype=np.float64)
    except Exception:
        pass
    if out:
        return out
    # last resort: anything with a free joint is a movable object
    for jid in range(sim.model.njnt):
        if sim.model.jnt_type[jid] == 0:                       # mjJNT_FREE
            bid = sim.model.jnt_bodyid[jid]
            name = sim.model.body_id2name(bid)
            if name and not name.startswith("robot"):
                out[name] = np.asarray(sim.data.body_xpos[bid], dtype=np.float64)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task-suite", required=True)
    p.add_argument("--trials-per-task", type=int, default=50,
                   help="MUST match the collection, or episode ids will not line up")
    p.add_argument("--max-tasks", type=int, default=None)
    p.add_argument("--camera-res", type=int, default=256)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    from mrvla.torch_compat import allow_numpy_pickles
    allow_numpy_pickles()

    suite = benchmark.get_benchmark_dict()[args.task_suite]()
    n_tasks = suite.n_tasks if args.max_tasks is None else min(suite.n_tasks, args.max_tasks)
    n_by_task = [min(args.trials_per_task, len(suite.get_task_init_states(t)))
                 for t in range(n_tasks)]
    ep_base = np.concatenate([[0], np.cumsum(n_by_task)[:-1]]).astype(int)

    rows = []
    for task_id in range(n_tasks):
        task = suite.get_task(task_id)
        env = OffScreenRenderEnv(
            bddl_file_name=os.path.join(get_libero_path("bddl_files"),
                                        task.problem_folder, task.bddl_file),
            camera_heights=args.camera_res, camera_widths=args.camera_res)
        init_states = suite.get_task_init_states(task_id)
        for trial in range(n_by_task[task_id]):
            env.reset()
            env.set_init_state(init_states[trial])
            rs = env.env
            ee = _eef_xpos(rs)
            objs = _object_bodies(rs)
            if task_id == 0 and trial == 0:
                print(f"[geo] end-effector at {np.round(ee, 3)}")
                print(f"[geo] {len(objs)} movable bodies found: {sorted(objs)}")
                print("[geo] CHECK THIS LIST. Empty, or robot links, means the discovery "
                      "fallbacks picked wrong and every distance below is meaningless.",
                      flush=True)
            d = np.array([np.linalg.norm(x - ee) for x in objs.values()], dtype=np.float64)
            rows.append((int(ep_base[task_id]) + trial, task_id, trial,
                         float(d.min()) if d.size else np.nan,
                         float(d.mean()) if d.size else np.nan,
                         float(d.max()) if d.size else np.nan,
                         int(d.size), float(ee[0]), float(ee[1]), float(ee[2])))
        print(f"[geo] task {task_id} ({task.language[:40]}): {n_by_task[task_id]} states",
              flush=True)
        env.close()

    a = np.array(rows, dtype=np.float64)
    np.savez_compressed(
        args.out, episode=a[:, 0].astype(np.int64), task_id=a[:, 1].astype(np.int64),
        trial=a[:, 2].astype(np.int64), d_nearest=a[:, 3], d_mean=a[:, 4], d_max=a[:, 5],
        n_objects=a[:, 6].astype(np.int64), ee_xyz=a[:, 7:10],
        trials_per_task=np.int64(args.trials_per_task), task_suite=str(args.task_suite))
    print(f"\n[geo] {a.shape[0]} init states, episodes {int(a[0,0])}..{int(a[-1,0])}")
    print(f"[geo] d_nearest range {np.nanmin(a[:,3]):.3f}..{np.nanmax(a[:,3]):.3f} m")
    print(f"[geo] wrote {args.out}")


if __name__ == "__main__":
    main()
