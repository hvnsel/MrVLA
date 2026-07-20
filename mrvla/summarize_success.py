"""Summarize success rate per (variant, suite) from run_cross_suite_eval.py output.

Reports base vs. lora success rate on libero_goal (in-distribution) and
libero_spatial / libero_object (cross-suite — the actual thesis test), with a
Wilson score interval on each rate and a paired McNemar test between variants.

McNemar (not an unpaired two-proportion test) is the right test here because
run_cross_suite_eval.py uses the same --seed for both variants, so episode i
in the base run and episode i in the lora run share the same task + init
state. That means base_success[i] and lora_success[i] are genuinely paired
observations, not independent samples.

Usage
-----
    python summarize_success.py --out-root ./activations/cross_suite_eval
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os

import numpy as np


def load_episode_success(out_dir: str) -> tuple[np.ndarray, list[int]]:
    """Collapse per-timestep shard rows to one success value per episode.

    Shards store one row per timestep; success is constant within an episode,
    so the first occurrence of each episode index is enough.
    """
    ep_success: dict[int, int] = {}
    for shard_path in sorted(glob.glob(os.path.join(out_dir, "shard_*.npz"))):
        d = np.load(shard_path)
        for ep, s in zip(d["episode"], d["success"]):
            ep = int(ep)
            if ep not in ep_success:
                ep_success[ep] = int(s)
    eps = sorted(ep_success)
    return np.array([ep_success[e] for e in eps]), eps


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a binomial proportion (more reliable than
    a normal-approx interval when n is modest or the rate is near 0/1)."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1 + z ** 2 / n
    centre = p + z ** 2 / (2 * n)
    margin = z * math.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2))
    return ((centre - margin) / denom, (centre + margin) / denom)


def mcnemar_p(b01: int, b10: int) -> float:
    """Continuity-corrected McNemar test, two-sided, via the chi2(df=1) <->
    normal relationship (avoids adding a scipy dependency to the repo)."""
    if b01 + b10 == 0:
        return float("nan")
    stat = (abs(b01 - b10) - 1) ** 2 / (b01 + b10)
    z = math.sqrt(stat)
    return 2 * (1 - 0.5 * (1 + math.erf(z / math.sqrt(2))))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out-root", required=True)
    args = p.parse_args()

    with open(os.path.join(args.out_root, "run_index.json")) as f:
        index = json.load(f)

    suites = sorted({s for variant in index.values() for s in variant})

    header = f"{'suite':16s} {'base':>16s} {'lora':>16s} {'delta':>8s} {'n':>6s} {'mcnemar p':>10s}"
    print(header)
    print("-" * len(header))

    for suite in suites:
        base_dir = index["base"][suite]
        lora_dir = index["lora"][suite]
        base_succ, base_eps = load_episode_success(base_dir)
        lora_succ, lora_eps = load_episode_success(lora_dir)

        if base_eps != lora_eps:
            print(
                f"  [warn] {suite}: episode indices differ between variants "
                f"({len(base_eps)} vs {len(lora_eps)}). Trials may not be paired 1:1 — "
                f"check that --seed and --trials-per-task matched between the two runs. "
                f"Truncating to the shorter length for this comparison."
            )
            n = min(len(base_succ), len(lora_succ))
            base_succ, lora_succ = base_succ[:n], lora_succ[:n]

        n = len(base_succ)
        base_rate, lora_rate = base_succ.mean(), lora_succ.mean()
        base_lo, base_hi = wilson_interval(int(base_succ.sum()), n)
        lora_lo, lora_hi = wilson_interval(int(lora_succ.sum()), n)

        b01 = int(((base_succ == 0) & (lora_succ == 1)).sum())  # base failed, lora succeeded
        b10 = int(((base_succ == 1) & (lora_succ == 0)).sum())  # base succeeded, lora failed
        p_val = mcnemar_p(b01, b10)

        print(
            f"{suite:16s} "
            f"{base_rate:6.3f} [{base_lo:.2f},{base_hi:.2f}] "
            f"{lora_rate:6.3f} [{lora_lo:.2f},{lora_hi:.2f}] "
            f"{lora_rate - base_rate:+8.3f} {n:6d} {p_val:10.4f}"
        )
        print(f"{'':16s}  (lora fixed {b01} base failures, broke {b10} base successes)")

    print(
        "\nRead libero_goal as the sanity check (did the intervention preserve "
        "in-distribution performance) and libero_spatial / libero_object as the "
        "actual thesis test (cross-suite transfer)."
    )


if __name__ == "__main__":
    main()