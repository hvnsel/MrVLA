"""Aggregate run_ablation.py worker outputs: success per task, damage, and the SCOPE test.

Three questions, in order of how much they matter:

1. HOW MUCH does each coalition hurt?      overall success vs the unablated baseline.
2. HOW BROADLY?                            n_tasks_hurt, and the participation ratio of the
                                           DAMAGE profile -- the same breadth statistic we
                                           applied to phi, now applied to behavioural harm.
                                           General coalitions should have BROAD damage
                                           (high PR), specialists NARROW damage (low PR).
3. Does damage land WHERE ATTRIBUTION SAID? correlation between each coalition's per-task
                                           damage and its per-task causal profile C_j(g).
                                           This is the sharpest confirmation of Path A: the
                                           attribution predicted, in advance, which tasks
                                           would break.

Also prints the head-to-head: does our breadth-ranked coalition do more/broader damage than
the firing-ranked one (the prior work's activity proxy)?

Usage
-----
python analyze_ablation.py --dir /work/.../ABLATION/goal
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np


def participation_ratio(v: np.ndarray) -> float:
    """Effective number of tasks a damage profile spreads over (same formula as Path A)."""
    v = np.asarray(v, dtype=np.float64)
    v = np.clip(v, 0, None)
    s1, s2 = v.sum(), (v * v).sum()
    return float(s1 * s1 / s2) if s2 > 0 else float("nan")


def coverage(tot: dict, conds: list, n_tasks: int):
    """Which (condition, task) cells have no episodes, and which tasks every condition has.

    Returns (missing, complete) where `missing` is a list of (condition, task) and `complete`
    is a bool mask [n_tasks] marking tasks present in EVERY condition. Cross-condition means
    are only comparable over `complete`: a dead worker leaves different holes in different
    columns, so a plain per-column mean silently averages each condition over a different
    task set.
    """
    import numpy as _np
    missing = [(c, t) for c in conds for t in range(n_tasks) if tot[c][t] == 0]
    complete = _np.array([all(tot[c][t] > 0 for c in conds) for t in range(n_tasks)])
    return missing, complete


def _pearson(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3:
        return float("nan")
    a, b = a[m] - a[m].mean(), b[m] - b[m].mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 0 else float("nan")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dir", required=True, help="run_ablation output dir")
    p.add_argument("--out", default=None, help="summary json (default: <dir>/summary.json)")
    args = p.parse_args()

    man = json.load(open(os.path.join(args.dir, "manifest.json")))
    rows = []
    for f in sorted(glob.glob(os.path.join(args.dir, "results_w*.json"))):
        rows.extend(json.load(open(f)))
    if not rows:
        raise SystemExit(f"no results_w*.json in {args.dir}")

    n_tasks = int(man["n_tasks"])
    conds = sorted({r["condition"] for r in rows})

    # success[cond][task] = (n_success, n_episodes)
    succ: dict[str, np.ndarray] = {c: np.zeros(n_tasks) for c in conds}
    tot: dict[str, np.ndarray] = {c: np.zeros(n_tasks) for c in conds}
    for r in rows:
        succ[r["condition"]][r["task_id"]] += r["success"]
        tot[r["condition"]][r["task_id"]] += 1
    rate = {c: np.divide(succ[c], tot[c], out=np.full(n_tasks, np.nan), where=tot[c] > 0)
            for c in conds}

    has_base = "baseline" in rate
    base = rate["baseline"] if has_base else np.full(n_tasks, np.nan)

    print(f"\n[abl] {len(rows)} episodes · {n_tasks} tasks · conditions: {', '.join(conds)}")
    if not has_base:
        print("[abl] WARNING: no baseline condition -- damage is uninterpretable without it.")

    # ---------------- coverage: refuse to compare unequal task sets silently ----------
    # A dead worker leaves whole (condition, task) cells empty. Two things then go wrong
    # QUIETLY: (a) the ALL column averages each condition over a DIFFERENT subset of tasks,
    # so a condition that happens to be missing a hard task looks better than baseline; and
    # (b) a single missing baseline task makes every damage vector carry a NaN, which turns
    # damage_PR and corr(damage, attribution) into NaN for every condition at once.
    missing, complete = coverage(tot, conds, n_tasks)
    if missing:
        by_cond: dict[str, list[int]] = {}
        for c, t in missing:
            by_cond.setdefault(c, []).append(t)
        print(f"\n[abl] *** INCOMPLETE: {len(missing)} of {len(conds) * n_tasks} "
              f"(condition, task) cells have no episodes ***")
        for c, ts in sorted(by_cond.items()):
            print(f"[abl]   {c:22s} missing tasks {ts}")
        if has_base and "baseline" in by_cond:
            print("[abl]   BASELINE itself is incomplete -- every damage number below is NaN "
                  "on those tasks, and damage_PR / corr collapse to NaN entirely.")
        print("[abl]   The ALL row below averages conditions over DIFFERENT task sets and is "
              "NOT comparable across columns.")
        print("[abl]   Re-run the missing shards before interpreting anything. With "
              f"--n-workers W, cell (condition ci, task t) belongs to worker "
              f"(ci*{n_tasks} + t) mod W, using the manifest's condition order.")
        if complete.any():
            print(f"[abl]   Tasks complete in ALL conditions: {list(np.where(complete)[0])} "
                  f"-- 'ALL(common)' below uses only these and IS comparable.")
        else:
            print("[abl]   No task is complete across all conditions.")

    # ---------------- per-task table ----------------
    print("\n=== success rate per task ===")
    hdr = "task  " + "".join(f"{c[:14]:>16s}" for c in conds)
    print(hdr); print("-" * len(hdr))
    for t in range(n_tasks):
        line = f"{t:<6d}" + "".join(f"{rate[c][t]:>15.2f} " for c in conds)
        print(line)
    print("ALL   " + "".join(f"{np.nanmean(rate[c]):>15.2f} " for c in conds))
    if not complete.all():
        # same task set for every column -> the only cross-condition comparison that is valid
        # while shards are missing
        print("ALL(common)" + "".join(f"{np.nanmean(rate[c][complete]):>10.2f} " for c in conds)
              if complete.any() else "ALL(common)  -- no task complete in every condition")

    # ---------------- per-condition summary ----------------
    summary = {"n_episodes": len(rows), "n_tasks": n_tasks, "conditions": {}}
    print("\n=== damage, breadth of damage, and agreement with attribution ===")
    for c in conds:
        overall = float(np.nanmean(rate[c]))
        damage = base - rate[c] if has_base else np.full(n_tasks, np.nan)   # >0 = hurt
        dmg_pos = np.clip(damage, 0, None)
        n_hurt = int(np.sum(dmg_pos > 0.10)) if has_base else -1
        pr_dmg = participation_ratio(dmg_pos) if has_base else float("nan")
        prof = man.get("info", {}).get(c, {}).get("per_task_profile")
        corr = _pearson(dmg_pos, prof) if (prof and has_base) else float("nan")
        tot_mag = man.get("info", {}).get(c, {}).get("total_magnitude")
        summary["conditions"][c] = {
            "overall_success": overall,
            "mean_damage": float(np.nanmean(damage)) if has_base else None,
            "n_tasks_hurt_gt10pct": n_hurt,
            "damage_participation_ratio": pr_dmg,
            "corr_damage_vs_attribution_profile": corr,
            "coalition_total_magnitude": tot_mag,
            "per_task_success": [float(x) for x in rate[c]],
            "per_task_damage": [float(x) for x in damage],
        }
        if c == "baseline":
            print(f"  {c:22s} success={overall:.3f}   (the ceiling)")
        else:
            print(f"  {c:22s} success={overall:.3f}  damage={np.nanmean(damage):+.3f}  "
                  f"tasks_hurt={n_hurt}/{n_tasks}  damage_PR={pr_dmg:.2f}  "
                  f"corr(damage, attribution)={corr:+.3f}"
                  + (f"  [coalition |mag|={tot_mag:.2e}]" if tot_mag else ""))

    # ---------------- the two headline comparisons ----------------
    def g(c, k):
        return summary["conditions"].get(c, {}).get(k, float("nan"))

    print("\n=== THE TESTS ===")
    if {"general", "specialist"} <= set(conds) and has_base:
        pg, ps = g("general", "damage_participation_ratio"), g("specialist", "damage_participation_ratio")
        print(f"1. SCOPE: damage breadth  general={pg:.2f}  vs  specialist={ps:.2f}")
        print("   Path A predicts general >> specialist (broad vs narrow harm). "
              + ("PREDICTION HOLDS." if pg > ps else "prediction NOT met."))
        mg, ms = g("general", "coalition_total_magnitude"), g("specialist", "coalition_total_magnitude")
        if mg and ms:
            print(f"   (sanity: coalition magnitudes {mg:.2e} vs {ms:.2e} -- if these differ a lot,\n"
                  "    damage differences are partly strength, not breadth.)")
    if {"general", "random"} <= set(conds) and has_base:
        dg, dr = g("general", "mean_damage"), g("random", "mean_damage")
        print(f"2. NULL:  mean damage  general={dg:+.3f}  vs  random={dr:+.3f}")
        print("   Ablating 5 general features must hurt MORE than 5 arbitrary strong ones. "
              + ("HOLDS." if dg > dr else "NOT met -- the ranking may carry no behavioural signal."))
    if {"general", "firing"} <= set(conds) and has_base:
        dg, df = g("general", "mean_damage"), g("firing", "mean_damage")
        pg, pf = g("general", "damage_participation_ratio"), g("firing", "damage_participation_ratio")
        print(f"3. HEAD-TO-HEAD vs the activity metric: damage {dg:+.3f} (ours) vs {df:+.3f} (firing); "
              f"breadth {pg:.2f} vs {pf:.2f}")
        print("   Our label-free causal ranking beating the firing ranking is the thesis, "
              "demonstrated behaviourally.")

    out = args.out or os.path.join(args.dir, "summary.json")
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[abl] wrote {out}")


if __name__ == "__main__":
    main()
