"""Statistical layer for the ablation runs: paired tests, intervals, and the power bound.

`analyze_ablation.py` reports point estimates -- success rates, damage, damage breadth,
agreement with attribution. Every one of those is a number without an error bar, and the
coalition run came back a NULL (baseline 78.9%, general 75.6%, specialist 79.4%, random
78.3%). A null point estimate is not interpretable on its own: it could mean "ablating
general features does nothing" or "this design could never have seen the effect". Those are
opposite claims and only a power analysis separates them.

This script adds the missing statistics, all on files already on disk. No GPU, no rollouts.

1. PAIRED TESTS. Every condition replays the SAME init states as baseline, so episodes match
   pair-by-pair on (task, episode). McNemar uses that pairing; an unpaired proportion test
   throws it away. Reported exact (binomial) as well as chi2, because per-task cells have
   ~20 episodes and the chi2 approximation is anticonservative there.

2. INTERVALS ON DAMAGE. A CI on the paired difference, so a null reads as "damage is below
   X points" rather than "damage is zero".

3. THE POWER BOUND -- the number that makes the null reportable. Given the observed
   discordance rate and the number of pairs, what is the smallest damage this design detects
   at 80% power? Anything below that is outside the run's resolution and must not be read as
   evidence of absence. Also reports the pairs needed for a target effect, which is what
   sizes any follow-up run.

4. A NULL FOR THE SCOPE TEST. `damage_participation_ratio` is the Path A prediction
   (general = broad damage, specialist = narrow), but PR of a noisy damage vector is not
   centred anywhere meaningful -- with 20 episodes/task, sampling noise alone produces a wide
   spread of PRs. We simulate the null "damage is spread evenly over tasks" at the observed
   magnitude and per-task episode counts, and report where the observed PR falls in it. If
   the observed general-vs-specialist PR gap sits inside the null spread, the scope test did
   not resolve, and that is the honest reading.

5. A NULL FOR THE ATTRIBUTION-AGREEMENT TEST. corr(per-task damage, per-task causal profile)
   is compared against a task-permutation null and given a TWO-LEVEL bootstrap interval. Two
   levels because at 20 episodes each per-task damage carries a standard error near 0.16 --
   larger than most of the damages being correlated -- so an interval that resamples only tasks
   treats noise as data and will report a confident correlation between two noise vectors. The
   bootstrap resamples tasks and then matched episode pairs within each task.

Usage
-----
python ablation_power.py --dir /work/.../ABLATION/goal
python ablation_power.py --dir /work/.../ABLATION/goal_singles --target-effect 0.10
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np

from mrvla.stats import (
    mcnemar_exact_p,
    mcnemar_p,
    mde_paired,
    paired_diff_ci,
    required_pairs,
    wilson_interval,
)


# ---------------------------------------------------------------------------
# loading and pairing
# ---------------------------------------------------------------------------
def load_run(run_dir: str) -> tuple[list[dict], dict]:
    """Read manifest.json plus every results_w*.json shard."""
    with open(os.path.join(run_dir, "manifest.json")) as f:
        manifest = json.load(f)
    rows: list[dict] = []
    for path in sorted(glob.glob(os.path.join(run_dir, "results_w*.json"))):
        with open(path) as f:
            rows.extend(json.load(f))
    if not rows:
        raise SystemExit(f"no results_w*.json in {run_dir}")
    return rows, manifest


def success_by_key(rows: list[dict]) -> dict[str, dict[tuple[int, int], int]]:
    """{condition: {(task_id, episode): success}}. Duplicate keys keep the first row, which
    is what a re-run shard would otherwise silently double-count."""
    out: dict[str, dict[tuple[int, int], int]] = {}
    for r in rows:
        cond = out.setdefault(r["condition"], {})
        key = (int(r["task_id"]), int(r["episode"]))
        cond.setdefault(key, int(r["success"]))
    return out


def paired_counts(base: dict, cond: dict, tasks: set[int] | None = None) -> tuple[int, int, int]:
    """(b01, b10, n_pairs) over keys present in BOTH conditions.

    b10 = baseline succeeded and the ablated condition failed  -> damage.
    b01 = the ablated condition succeeded and baseline failed   -> repair.
    Restricting to shared keys is what makes this paired: a condition missing a whole task
    contributes nothing rather than being compared against a different task set.
    """
    keys = set(base) & set(cond)
    if tasks is not None:
        keys = {k for k in keys if k[0] in tasks}
    b01 = b10 = 0
    for k in keys:
        b, c = base[k], cond[k]
        if b == 1 and c == 0:
            b10 += 1
        elif b == 0 and c == 1:
            b01 += 1
    return b01, b10, len(keys)


# ---------------------------------------------------------------------------
# nulls for the two shape tests
# ---------------------------------------------------------------------------
def participation_ratio(v: np.ndarray) -> float:
    """Effective number of tasks a damage profile spreads over (Path A's PR, applied to harm)."""
    v = np.clip(np.asarray(v, dtype=np.float64), 0, None)
    s1, s2 = v.sum(), (v * v).sum()
    return float(s1 * s1 / s2) if s2 > 0 else float("nan")


def damage_pr_null(base_rate: np.ndarray, n_ep: np.ndarray, mean_damage: float,
                   n_sim: int = 2000, seed: int = 0) -> np.ndarray:
    """Null distribution of damage-PR when damage is spread EVENLY across tasks.

    Draws baseline and ablated successes independently per task from binomials at the
    observed per-task episode counts, with the ablated rate uniformly lowered by
    `mean_damage`. Any spread in the resulting PR is pure sampling noise, so the width of
    this distribution is the resolution limit of the scope test at this episode budget.

    Independent (not paired) draws make the null slightly WIDER than a paired design would
    give, so using it is conservative for "the observed PR is outside the null".

    Draws where the simulated damage is non-positive on EVERY task have no defined PR and are
    dropped. The surviving fraction is itself diagnostic: if most draws die, the damage is too
    small to register at this episode count and the scope test cannot be run at all.
    """
    rng = np.random.default_rng(seed)
    base_rate = np.asarray(base_rate, dtype=np.float64)
    n_ep = np.asarray(n_ep, dtype=np.int64)
    ok = np.isfinite(base_rate) & (n_ep > 0)
    if not ok.any():
        return np.array([])
    p_base = np.clip(base_rate[ok], 0, 1)
    p_abl = np.clip(p_base - mean_damage, 0, 1)
    n = n_ep[ok]
    sims = np.empty(n_sim, dtype=np.float64)
    for i in range(n_sim):
        b = rng.binomial(n, p_base) / n
        a = rng.binomial(n, p_abl) / n
        sims[i] = participation_ratio(b - a)
    return sims[np.isfinite(sims)]


def corr_permutation_p(damage: np.ndarray, profile: np.ndarray,
                       paired: dict | None = None, n_perm: int = 20000,
                       n_boot: int = 4000, seed: int = 0) -> tuple[float, float, tuple]:
    """(r, one-sided permutation p, bootstrap CI) for corr(per-task damage, causal profile).

    The permutation shuffles which task each causal profile entry belongs to, which is exactly
    the null "attribution does not know where damage lands".

    THE INTERVAL IS TWO-LEVEL, and it has to be. A bootstrap that resamples only TASKS treats
    each task's damage as a fixed number, when at 20 episodes a per-task damage carries a
    standard error near 0.16 -- larger than most of the damages being correlated. Such an
    interval is far too tight and will report a confident correlation between two noise vectors.
    So when `paired` is supplied (task -> (baseline successes, condition successes) as matched
    0/1 arrays), the bootstrap resamples tasks AND then resamples matched episode pairs within
    each drawn task, recomputing damage from the resampled pairs. Both sources of variation
    propagate, and the pairing is preserved because episodes are drawn as pairs.

    Without `paired` it falls back to the task-only interval, which is reported as such.
    """
    d = np.asarray(damage, dtype=np.float64)
    p = np.asarray(profile, dtype=np.float64)
    m = np.isfinite(d) & np.isfinite(p)
    keep = np.where(m)[0]
    d, p = d[m], p[m]
    if d.size < 3:
        return float("nan"), float("nan"), (float("nan"), float("nan"))

    def r_of(a, b):
        a = a - a.mean()
        b = b - b.mean()
        den = np.sqrt((a * a).sum() * (b * b).sum())
        return float((a * b).sum() / den) if den > 0 else float("nan")

    r = r_of(d, p)
    rng = np.random.default_rng(seed)
    null = np.array([r_of(d, rng.permutation(p)) for _ in range(n_perm)])
    null = null[np.isfinite(null)]
    p_val = float((null >= r).mean()) if null.size and np.isfinite(r) else float("nan")

    pairs = None
    if paired is not None:
        pairs = [paired.get(int(t)) for t in keep]
        if any(v is None or len(v[0]) < 2 for v in pairs):
            pairs = None

    boot = []
    for _ in range(n_boot):
        idx = rng.integers(0, d.size, d.size)
        if np.unique(idx).size < 3:
            continue
        if pairs is None:
            v = r_of(d[idx], p[idx])
        else:
            dd = np.empty(idx.size)
            for k, t in enumerate(idx):
                b_arr, c_arr = pairs[t]
                e = rng.integers(0, b_arr.size, b_arr.size)   # matched pairs, drawn together
                dd[k] = b_arr[e].mean() - c_arr[e].mean()
            v = r_of(dd, p[idx])
        if np.isfinite(v):
            boot.append(v)
    ci = ((float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5)))
          if len(boot) > 100 else (float("nan"), float("nan")))
    return r, p_val, ci


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", required=True, help="run_ablation.py output directory")
    ap.add_argument("--target-effect", type=float, default=0.05,
                    help="damage (in success-rate points, 0-1) a follow-up run should be "
                         "sized to detect (default 0.05)")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--power", type=float, default=0.80)
    ap.add_argument("--per-task", action="store_true",
                    help="print the per-task paired table (needed for single-feature runs, "
                         "where the prediction names a specific task)")
    ap.add_argument("--n-sim", type=int, default=2000, help="draws for the damage-PR null")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="output json (default <dir>/power.json)")
    args = ap.parse_args()

    rows, manifest = load_run(args.dir)
    n_tasks = int(manifest["n_tasks"])
    by_cond = success_by_key(rows)
    if "baseline" not in by_cond:
        raise SystemExit("no baseline condition: paired damage is undefined without it")
    base = by_cond["baseline"]
    conds = [c for c in sorted(by_cond) if c != "baseline"]

    n_base = len(base)
    base_succ = sum(base.values())
    lo, hi = wilson_interval(base_succ, n_base)
    print(f"\n[pow] {args.dir}")
    print(f"[pow] baseline: {base_succ}/{n_base} = {base_succ / max(n_base,1):.3f} "
          f"[{lo:.3f}, {hi:.3f}]   ({n_tasks} tasks)")

    per_task_base = np.full(n_tasks, np.nan)
    n_ep_task = np.zeros(n_tasks, dtype=np.int64)
    for t in range(n_tasks):
        ks = [k for k in base if k[0] == t]
        if ks:
            n_ep_task[t] = len(ks)
            per_task_base[t] = float(np.mean([base[k] for k in ks]))

    out: dict = {"dir": args.dir, "n_tasks": n_tasks,
                 "baseline": {"n": n_base, "successes": base_succ,
                              "rate": base_succ / max(n_base, 1), "ci": [lo, hi]},
                 "alpha": args.alpha, "power": args.power,
                 "target_effect": args.target_effect, "conditions": {}}

    # ---------------- pooled paired test per condition ----------------
    print("\n=== paired damage vs baseline (McNemar; + = ablation hurt) ===")
    head = (f"{'condition':24s} {'pairs':>6s} {'succ':>6s} {'damage':>8s} "
            f"{'95% CI':>16s} {'b10/b01':>9s} {'p(exact)':>9s} {'MDE@80%':>8s}")
    print(head)
    print("-" * len(head))
    for c in conds:
        cond = by_cond[c]
        b01, b10, n_pairs = paired_counts(base, cond)
        d, dlo, dhi = paired_diff_ci(b01, b10, n_pairs)
        disc = (b01 + b10) / n_pairs if n_pairs else float("nan")
        mde = mde_paired(n_pairs, disc, args.alpha, args.power) if disc > 0 else float("nan")
        p_ex = mcnemar_exact_p(b01, b10)
        rate = float(np.mean([cond[k] for k in cond])) if cond else float("nan")
        print(f"{c[:24]:24s} {n_pairs:6d} {rate:6.3f} {d:+8.3f} "
              f"[{dlo:+.3f},{dhi:+.3f}] {b10:4d}/{b01:<4d} {p_ex:9.3f} {mde:8.3f}")
        out["conditions"][c] = {
            "n_pairs": n_pairs, "success_rate": rate,
            "damage": d, "damage_ci": [dlo, dhi],
            "b10_damaged": b10, "b01_repaired": b01, "discordance_rate": disc,
            "mcnemar_p_exact": p_ex, "mcnemar_p_chi2": mcnemar_p(b01, b10),
            "mde_80pct": mde,
            "pairs_needed_for_target": required_pairs(args.target_effect, disc,
                                                      args.alpha, args.power)
            if disc > 0 else float("nan"),
        }

    # ---------------- what the design could have seen ----------------
    discs = [v["discordance_rate"] for v in out["conditions"].values()
             if np.isfinite(v["discordance_rate"])]
    disc_typ = float(np.median(discs)) if discs else float("nan")
    n_typ = int(np.median([v["n_pairs"] for v in out["conditions"].values()])) if conds else 0
    ep_task = int(np.median(n_ep_task[n_ep_task > 0])) if (n_ep_task > 0).any() else 0
    mde_pool = mde_paired(n_typ, disc_typ, args.alpha, args.power) if disc_typ > 0 else float("nan")
    mde_task = mde_paired(ep_task, disc_typ, args.alpha, args.power) if disc_typ > 0 else float("nan")
    need = required_pairs(args.target_effect, disc_typ, args.alpha, args.power) \
        if disc_typ > 0 else float("nan")
    out["design"] = {"median_pairs": n_typ, "median_discordance": disc_typ,
                     "episodes_per_task": ep_task, "mde_pooled": mde_pool,
                     "mde_per_task": mde_task, "pairs_needed_for_target": need,
                     "episodes_per_task_needed_for_target": need / max(n_tasks, 1)}
    print("\n=== what this design can resolve ===")
    print(f"  discordance rate (median over conditions) : {disc_typ:.3f}")
    print(f"  pooled  {n_typ} pairs  -> detects damage >= {mde_pool:.3f} "
          f"({100*mde_pool:.1f} points) at {100*args.power:.0f}% power")
    print(f"  per-task {ep_task} pairs -> detects damage >= {mde_task:.3f} "
          f"({100*mde_task:.1f} points) at {100*args.power:.0f}% power")
    print(f"  to detect {100*args.target_effect:.1f} points pooled: {need:.0f} pairs "
          f"= {need / max(n_tasks,1):.0f} episodes/task")
    print("  READ THIS AS: damage smaller than the pooled MDE is BELOW THE RESOLUTION of the\n"
          "  run. A non-significant condition bounds the damage (see its CI); it does not\n"
          "  show the features are inert.")

    # ---------------- scope test with its null ----------------
    info = manifest.get("info", {})
    print("\n=== scope: is the damage profile's breadth distinguishable from sampling noise? ===")
    for c in conds:
        cond = by_cond[c]
        dmg = np.full(n_tasks, np.nan)
        for t in range(n_tasks):
            ks = [k for k in set(base) & set(cond) if k[0] == t]
            if ks:
                dmg[t] = float(np.mean([base[k] for k in ks]) - np.mean([cond[k] for k in ks]))
        pr = participation_ratio(dmg)
        mean_dmg = float(np.nanmean(np.clip(dmg, 0, None)))
        null = damage_pr_null(per_task_base, n_ep_task, mean_dmg, args.n_sim, args.seed)
        entry = out["conditions"][c]
        entry["per_task_damage"] = [float(x) for x in dmg]
        entry["damage_pr"] = pr
        usable = null.size / max(args.n_sim, 1)
        entry["damage_pr_null_usable_fraction"] = usable
        if not np.isfinite(pr):
            print(f"  {c[:24]:24s} damage_PR undefined (no task shows positive damage)")
        elif null.size < 0.2 * args.n_sim:
            # too few draws produce any damage at all: at this mean damage and episode count
            # the shape of the damage profile is not a measurable quantity
            print(f"  {c[:24]:24s} damage_PR={pr:5.2f}   null UNDEFINED in "
                  f"{100*(1-usable):.0f}% of draws -- mean damage {mean_dmg:.3f} is too small "
                  f"to register at {int(np.median(n_ep_task[n_ep_task>0]))} episodes/task; "
                  f"the scope test cannot run here")
        else:
            pct = float((null <= pr).mean())
            n_lo, n_hi = float(np.percentile(null, 2.5)), float(np.percentile(null, 97.5))
            entry["damage_pr_null"] = {"mean": float(null.mean()), "p2.5": n_lo, "p97.5": n_hi,
                                       "percentile_of_observed": pct}
            verdict = "OUTSIDE null" if (pr < n_lo or pr > n_hi) else "inside null (unresolved)"
            print(f"  {c[:24]:24s} damage_PR={pr:5.2f}   null[{n_lo:4.2f}, {n_hi:5.2f}] "
                  f"mean={null.mean():5.2f}   {verdict}"
                  + (f"   [{100*(1-usable):.0f}% of draws undefined]" if usable < 0.9 else ""))

        prof = info.get(c, {}).get("per_task_profile")
        if prof and np.isfinite(dmg).sum() >= 3:
            # matched 0/1 arrays per task, so the interval can propagate episode-level noise
            paired = {}
            for t in range(n_tasks):
                ks = sorted(set(base) & set(cond) & {(t, e) for (tt, e) in base if tt == t})
                if len(ks) >= 2:
                    paired[t] = (np.array([base[k] for k in ks], dtype=np.float64),
                                 np.array([cond[k] for k in ks], dtype=np.float64))
            r, p_perm, ci = corr_permutation_p(dmg, np.asarray(prof, dtype=np.float64),
                                               paired=paired or None, seed=args.seed)
            entry["corr_damage_vs_attribution"] = {"r": r, "perm_p_one_sided": p_perm,
                                                   "bootstrap_ci": list(ci),
                                                   "bootstrap_two_level": bool(paired)}
            width = ci[1] - ci[0] if np.isfinite(ci[0]) and np.isfinite(ci[1]) else float("nan")
            note = "  (spans most of [-1,1]: uninformative)" if width > 1.2 else ""
            print(f"  {'':24s} corr(damage, attribution)={r:+.3f}  "
                  f"perm p={p_perm:.3f}  boot CI[{ci[0]:+.2f}, {ci[1]:+.2f}]{note}")

    # ---------------- per-task paired table ----------------
    if args.per_task:
        print("\n=== per-task paired damage (exact McNemar) ===")
        print("  a single-feature condition predicts damage on a NAMED task; this is where "
              "that prediction is tested.")
        for c in conds:
            cond = by_cond[c]
            print(f"\n  {c}")
            print(f"    {'task':>4s} {'base':>6s} {'abl':>6s} {'damage':>8s} "
                  f"{'b10/b01':>9s} {'p(exact)':>9s}")
            per_task = []
            for t in range(n_tasks):
                b01, b10, n_pairs = paired_counts(base, cond, tasks={t})
                if not n_pairs:
                    continue
                ks = [k for k in set(base) & set(cond) if k[0] == t]
                rb = float(np.mean([base[k] for k in ks]))
                ra = float(np.mean([cond[k] for k in ks]))
                p_ex = mcnemar_exact_p(b01, b10)
                per_task.append({"task": t, "n_pairs": n_pairs, "base_rate": rb,
                                 "abl_rate": ra, "damage": rb - ra,
                                 "b10": b10, "b01": b01, "p_exact": p_ex})
                print(f"    {t:4d} {rb:6.2f} {ra:6.2f} {rb - ra:+8.2f} "
                      f"{b10:4d}/{b01:<4d} {p_ex:9.3f}")
            out["conditions"][c]["per_task_tests"] = per_task

    dest = args.out or os.path.join(args.dir, "power.json")
    with open(dest, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[pow] wrote {dest}")


if __name__ == "__main__":
    main()
