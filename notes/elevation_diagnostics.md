# Cheap diagnostics that raise the ceiling on the current claims

Written after an audit of `results.md` against what the code actually computes. Everything
here is **CPU-only re-analysis of artifacts already on disk** — no rollouts, no SAE
retraining, no GPU. `BASE=... ./run_diagnostics.sh` runs the lot in minutes.

The organising question was not "what else could we measure" but "where does a referee
currently get to say *that number could mean anything*". Five places, in order of how much
they cost us.

---

## 1. The ablation null has no error bar, no paired test, and no power bound

**The gap.** The coalition run came back a null — baseline 78.9%, general 75.6%, specialist
79.4%, random 78.3% — and `analyze_ablation.py` reports every one of those as a bare point
estimate. A null point estimate is not a result: "ablating general features does nothing" and
"this design could never have seen the effect" produce identical output. `results.md` does
not currently mention the null at all, which is the most exposed position in the paper.

**What was added.** `ablation_power.py`:

- **Paired McNemar** on the matched pairs the design already bought. Every condition replays
  the same init states, so episodes match on `(task, episode)`; an unpaired proportion test
  discards that. Exact binomial for per-task cells, where the discordant count is often under
  10 and the continuity-corrected chi² is conservative enough to hide real effects.
- **A CI on the damage**, so a null reads as "damage is below X points".
- **The minimum detectable effect.** The number that makes the null reportable: given the
  observed discordance rate and the pair count, the smallest damage the run resolves at 80%
  power. Damage below it is outside the run's resolution, full stop. Also reports the
  episodes/task a follow-up needs for a named target effect, which is how the next run gets
  sized instead of guessed.
- **A null for the SCOPE test.** `damage_participation_ratio` is the Path A prediction
  (general → broad harm, specialist → narrow), but the PR of a noisy damage vector is not
  centred anywhere meaningful. We simulate "damage spread evenly across tasks" at the
  observed magnitude and episode counts; if the observed PR sits inside that spread the scope
  test did not resolve. When the mean damage is too small for any simulated draw to register,
  the script says the scope test **cannot be run** rather than printing a number.
- **A null for the attribution-agreement test.** `corr(per-task damage, per-task causal
  profile)` is a correlation over G = 10 points; it now carries a task-permutation p and a
  task-level bootstrap interval.

**What it converts.** "We ablated and nothing happened" → "this design resolves damage down
to X points; general-coalition damage is bounded above by Y". A bounded null is publishable
and is the honest form of the §3.2a head-to-head. It also tells us, before spending GPU time,
whether the *single-feature* run currently in flight can possibly hit its pre-registered
per-task predictions at 20 episodes/task — `--per-task` scores exactly those named-task
predictions.

---

## 2. The prescribed Path A negative control is a no-op (methods finding)

**The gap.** `EXPERIMENT_PLAN.md` §3.2b prescribes: *"Permute task labels and recompute the
A4 partial; a 'generic causality is trivial' objection predicts it collapses to ≈0 while the
real value is +0.493."* Permuting task labels **within a feature** — independently shuffling
each column of `C` over task rows — does not collapse anything. It is very nearly the
identity on this statistic:

> LOTO already evaluates every fold. For feature *j*, holding out task *g* contributes the
> pair (PR of `C[≠g, j]`, `C[g, j]`). Permuting column *j* by π makes fold *g* contribute the
> pair that fold π(g) contributed in the real data — the same G pairs, dealt into different
> folds. The fold-level partials are all positive and similar in size, so mixing them
> reproduces the statistic.

Measured on synthetic data with real breadth structure: real **+0.164**, task-label
permutation **+0.164**. On the synthetic 2048-feature fixture: real **+0.1420**, task-label
permutation **+0.1422**. Running the control as written would have returned "the null equals
the result" and looked like the headline collapsing, when the permutation destroys nothing.

**What was added.** `permutation_null.py` with two floors that do test something, both
scoring the *identical* estimator the reported number uses (`loto_partial_both`, now factored
into `mrvla/attribution.py` and called by `run_attribution.py` — so the null cannot drift
from the result):

- **`column_shuffle`** — the one that matters. Permute feature identity independently within
  each task row. Task marginals are preserved exactly, and so is the purely mechanical
  within-column link (an evenly spread column still has a predictable held-out entry), but a
  feature no longer has an identity across tasks. Whatever survives is arithmetic, not
  biology. On the fixture: **+0.142 → −0.001, z = +15.7**. Conservative by construction:
  `base_rate` stays attached to its original index and so becomes an uninformative control,
  and partialling out a useless covariate removes less, biasing the floor upward.
- **`feature_shuffle`** — estimator floor. Permute feature identity of the held-out vector
  only. Anything but ~0 means a bug in the estimator.
- `--show-invalid-null` runs the row permutation too, so the no-op is demonstrated on the
  real data rather than asserted from a synthetic argument.

**What it converts.** "+0.493 is positive in 10/10 folds" → "+0.493 against a mechanical floor
of ≈0, p < 1/1000". It also stops the team running a control that would have looked
catastrophic.

---

## 3. "Concentration and reproducibility" has no number attached

**The gap.** §A6 answers the "of course hundreds of features influence the action" objection
with *"Influence is not the claim; **concentration and reproducibility** of influence is."*
Neither word is measured anywhere. The rebuttal is currently verbal.

**What was added.** `causal_concentration.py`:

- **`n_eff`** — the effective number of features carrying the action, the participation ratio
  applied over features instead of tasks. Same scale-free statistic Path A already uses, so
  it needs no new justification. Read against F = 2048.
- **Lorenz shares and Gini** — mass held by the top 10 / 50 / 100 / 1%, per suite and per
  task, so "concentrated" is not an artefact of averaging ten differently-shaped tasks.
- **Cross-task top-N overlap as a ratio to chance** — the *reproducibility* half. Concentration
  alone is compatible with every task recruiting its own private top-50; this asks whether it
  is the same coalition. Chance is `N²/F`, the expected intersection of two independent
  N-subsets.
- **Controls**: the same statistics for a base-firing-rate ranking (the prior work's activity
  proxy) and for a column-shuffled matrix. If causal mass concentrates no more than firing
  does, "concentration" is an activity statement — the same confound-first discipline that
  killed §2.2.

**What it converts.** A verbal rebuttal into a sentence with numbers in it: "the action is
carried by an effective N of 2048 features; the same top-50 recurs across tasks at K× chance;
firing rate concentrates less."

---

## 4. The A×B null is undefended against measurement noise — and the obvious defence is backwards

**The gap.** `corr(breadth, recurrence) = −0.127` is Path B's boundary condition. The referee
response is that both measures are noisy (SAE dictionaries are ~60% seed-reproducible) and two
unreliable measures cannot correlate. Unanswered, this is "could not measure", which analysis
commitment #4 forbids conflating with "not there".

**The correction that matters.** Substituting `r_yy = 1` for the unknown recurrence
reliability *feels* conservative but yields a **lower** bound on `|r_true|` — the wrong
direction entirely, since defending a null needs an upper bound. What defends the null is a
high **ceiling** `sqrt(r_xx · r_yy)`: if the measurements could have shown |r| up to 0.85 and
returned 0.127, the dissociation is real; if the ceiling is itself near 0.15, nothing was
measurable. The ceiling depends on `r_yy`, so **the A×B null cannot be defended without an
estimate of recurrence reliability.**

**What was added.** `reliability_ceiling.py` reports the correction in whichever direction is
actually sound:

- For the **Path A positive**, the correction is free: attenuation only ever shrinks a
  correlation, so `+0.493 / sqrt(r_xx)` is a floor needing no assumption about the other
  measure. At `r_xx = 0.72` that is **≥ +0.581**.
- For the **A×B null** with `r_yy` unknown, it refuses to fake it and inverts the question
  instead: the **breakeven reliability**, i.e. how reliable recurrence must be for the observed
  value to bound the truth below a stated threshold. At `r_xx = 0.72`, `r = −0.127`,
  threshold 0.30, that bar is **r_yy > 0.249** — a low bar, and now a concrete, cheap thing
  to go measure (`q_cross` recomputed on disjoint halves of the probe frames: no rollouts, no
  retraining).

**Prerequisite.** `r_xx` comes from `split_half_breadth.py`, which is built but whose numbers
are not yet in `results.md`. That run is stage 3 of `run_diagnostics.sh`.

---

## 5. `compare_recurrence_groups --target` could mislabel a whole per-suite table

**The gap.** Already known (`results.md` open threads): `--target` is label-only, while the
model actually analysed comes from `--rec`. `--target spatial` with the goal npz produced a
**goal** comparison filed under "spatial", and nothing downstream could detect it.

**What was added.** `--target` is now cross-checked against both `--rec` (matched exactly
against the target embedded in `layer_NN_target_<m>.npz`) and `--attr` (matched loosely, since
attribution directories are named freely). A mismatch exits with an explanation rather than
running; `--allow-mismatch` is there for deliberately unconventional naming. This catches the
realistic failure — `--rec` updated, `--attr` left behind — as well as the original one.

---

## Also changed

- `mrvla/stats.py` — the no-scipy small-sample kit (Wilson, McNemar exact + corrected, paired
  difference CI, MDE, required-n). `summarize_success.py` now imports from it instead of
  carrying its own copy, so there is one implementation of each test in the repo.
- `mrvla/attribution.py` — gained `rank_partial_both` and `loto_partial_both`, factored out of
  `run_attribution.py`. The reported statistic and its null are now literally the same
  function.
- `identify_features.py` — `torch` and the SAE encoder moved inside `main()`. They were only
  used there, and the module-level import forced a torch install on every CPU-only re-analysis
  tool that imports `adjusted_breadth` / `select_general_specialist`
  (`compare_recurrence_groups`, `join_pathA_pathB`, `run_ablation`'s coalition builder), all
  of which advertise themselves as needing no GPU.

## Not done (deliberately)

- **Signature-entropy control for B2** (open thread #3). Needs `run_causal_recurrence.py` to
  save the `[F, 256]` signature matrix `S`, not just its norms. The analysis afterwards is
  trivial and CPU-only, but it is a change to a producing script plus a re-run, so it is not
  in the "pure re-analysis" tier this pass was scoped to.
- **Four-suite Path A table** (§A5). `results.md` states Path A "reproduces across all four
  LIBERO suites" while `EXPERIMENT_PLAN.md` §3.2c lists the replication as DEFERRED and the
  per-suite table is empty. That contradiction should be resolved from the actual run status
  before the diagnostics are pointed at four suites — `run_diagnostics.sh` will silently skip
  suites whose npz is absent, which makes it a quick way to find out which ones exist.
