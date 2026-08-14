# Paper plan — v1 scope and the v2 backlog

**Decision (recorded, standing):** ship a v1 paper now, then a v2 later. Author is an
undergrad with a PhD-application deadline; a scoped, honest v1 beats an unbounded v2 that
misses the window. This mirrors the target paper's own history — arXiv 2603.19183v1 had
steering only; v2 added the ablation experiments, DROID hardware, and the generality
classifier.

**Order of operations from here:** finish the steering run → clean up the repo → draft →
re-run the final numbers on the cleaned, known-good configuration → submit.

---

## The v1 thesis

> Swann et al. define feature "generality" from activity statistics (episode coverage, mean
> onset count, activation magnitude) and show that ablating high-generality features destroys
> policy performance. We ask whether that reflects reusable cross-task *causal* structure or
> simply how often a feature is active. We define a causal, label-free breadth metric,
> validate it, and find that ablation damage tracks **activity**, not cross-task causal reuse.

This is a measurement paper responding to a demonstration paper. Its contribution is a
selection rule that is mechanical and stated in advance, plus the controls that separate the
two explanations.

## What v1 contains (all numbers already in hand)

| Piece | Result |
|---|---|
| Sufficiency gate (pre-registered viability check) | **0.936** at k=100 |
| Causal breadth = PR over per-task attribution | mean PR ≈ 6.05 |
| Confound control: PR residualized on magnitude + base rate | built into the metric |
| Breadth ↔ held-out causal influence, LOTO | **+0.493, positive in 10/10 folds** |
| Replication | all four LIBERO suites |
| Split-half reliability of breadth | ρ 0.12–0.47, SB-corrected **0.22–0.64** (object best) |
| Rollout harness validation | baseline **78.9%** vs published ~79.2% |
| Coalition ablation (top-5 breadth) | null: general 75.6, specialist 79.4, random 78.3 |
| **Positive control**: ablate top-5 by firing rate | **2.2%** — the machinery demonstrably works |
| Single-feature ablation, pre-registered per-task predictions | null (see `ablation_goal_singles.md`) |
| Additive steering, 6 features | *pending* |
| Path B (cross-model causal-signature recurrence) | scoped negative, ~1 page, not a pillar |

The firing-rate collapse is load-bearing twice over: it is the positive control proving
ablation works, **and** it is the evidence for the thesis, since Swann's generality score is
dominated by coverage and onset count (β_c = 1.80, β_ō = 1.89 on LIBERO).

## Honest limits to state in the paper, not hide

- Breadth ranking reliability is low (SB 0.22–0.64), so top-N selection is noise-limited.
  Say so; it is why the coalition ablation is uninformative rather than negative.
- One layer (31), one model (OpenVLA-7B), one benchmark (LIBERO), n=4 fine-tunes.
- The sufficiency gate's target was changed after the original L1 formulation stalled at
  0.72–0.76. **Disclose this.** A reviewer who finds it in the code and not the text will not
  be generous.
- Path B has no positive control and no inheritance floor. Report as scoped, claim nothing.

## Explicitly OUT of v1 → the v2 backlog

- Bagged / denoised breadth selection (rank-average over task subsamples)
- Magnitude dose-response sweep: where does a single feature start to matter?
- Replicating Swann's classifier on our dictionary (their coefficients are published, and
  they sanction applying the LIBERO boundary to OpenVLA)
- Path B repairs: positive control (does grasp match grasp), inheritance floor vs base model,
  position-aware causal signature
- More layers, more seeds, more base models
- LOTO ablation / held-out generalization

---

## Clean re-run checklist ("well conditioned")

Everything below was learned the hard way this round. The final numbers in the paper should
come from a single run that satisfies all of it.

1. **Use the current branch code.** It carries four fixes that silently corrupted earlier
   runs: torch≥2.6 init-state loading, the `attention_mask` off-by-one, condition
   de-duplication, and the analyzer's coverage check.
2. **Set the render env vars** before every job:
   `MUJOCO_GL=egl`, `MUJOCO_EGL_DEVICE_ID=${CUDA_VISIBLE_DEVICES%%,*}`, same for
   `EGL_DEVICE_ID`. A worker that aborts in MuJoCo takes its whole shard with it.
3. **Pick a worker count that divides the job count evenly** (7 conditions × 10 tasks = 70 →
   7 or 10 workers, never 4).
4. **Check coverage after every run.** `analyze_ablation.py` now refuses to compare unequal
   task sets, but read the warning — an incomplete run previously looked like "ablation
   improved success."
5. **Baseline must land near 79% on goal.** That is the validated harness signature; anything
   far off means a setup regression, not a finding.
6. **Pin `--alpha` for steering.** Under `--gamma` each worker calibrates on its own first
   batch and the shards apply different interventions.
7. **Keep init states paired across conditions** (already the default) so every comparison is
   within-pair and analyzable by McNemar.
8. **Add `--random-controls 1`** to the final steering run. Without a norm-matched random
   direction, "any large perturbation changes behaviour" explains every steering result.
9. Path A metric numbers on **all four suites**; behavioural experiments on **goal** only.

## Repo cleanup before drafting

- Prune dead exploration scripts in `mrvla/` that no longer feed a result
- One README section per surviving experiment: command, output, what it shows
- Ensure every headline number is reproducible from a single documented command
- Keep `results.md` as the numbers-of-record; keep `notes/` as pre-registrations
