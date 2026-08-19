# MrVLA — Results

Living record of the major findings. Add new results under the relevant path as
they land; keep the "Headline" and "Open threads" sections at the top current.

**Question.** Do SAE features in a fine-tuned VLA (OpenVLA, Llama-2 7B, layer 31,
F=2048, TopK k=100) encode a *causal* notion of **generality** — and is that the
same thing as a feature **recurring** across independently trained models? We
redefine generality functionally and label-free as **causal influence that
recurs**, split into two axes:

- **Path A** — causal influence recurring across **tasks**, within one model.
- **Path B** — causal signatures recurring across independently fine-tuned **models**.

---

## Headline

- **Path A (positive, the headline result).** Generality is real and it is
  *causal cross-task influence*. A small set of features carry a large,
  reproducible share of the action decision across many tasks (grasp / release /
  home-pose primitives); a long tail are scene- and phase-locked specialists.
- **Path B (supporting result).** Cross-model recurrence is a **distinct axis**
  from causal generality — uncorrelated to mildly *inverse*. Recurrence is not a
  proxy for generality. Established three independent ways (firing metric,
  activation recurrence, causal-signature recurrence).
- **Net.** "General" and "recurrent-across-models" are two different properties.
  Path A is the finding; Path B is the boundary condition plus a methods
  contribution (a properly-defined causal-recurrence metric).
- **Part 2 (below).** A measurement-validity pass over our own results. Path A now
  replicates in all four suites against a zero floor; concentration is quantified;
  the ablation null is bounded rather than absent; the "generality is gripper/phase
  control" hypothesis is falsified; and Path B survives its strongest alternative
  explanation (dictionary splitting) at both feature and role level. It also finds
  four defects in the existing pipeline, one of which is still open.

---

## Path A — causal generality across tasks (within a model)

### A1. Sufficiency gate — PASS

We first had to establish that the SAE features actually *carry* the action, not
just correlate with it. The gate measures the fraction of the action margin that
the features reconstruct, as a through-origin slope
`S = Σ(c·x) / Σ(c²)` decomposed into features + bias + error summing to 1.

- **Sufficiency (features + bias) = 0.936** at k=100 — clears the 0.80 bar
  comfortably.
- This replaced an earlier brittle L1 full-reconstruction argmax-match test that
  stalled at ~0.72–0.76 even with k=256 and 300 epochs. The argmax test was the
  wrong target: we only care about the logits relevant to the 7 action tokens,
  not full-vocab reconstruction. Sufficiency isolates exactly that.

**Reading:** the features are a faithful causal handle on the action, so
attribution over them is meaningful.

### A2. Attribution and breadth

Per-feature causal contribution to a task's action:
`φ_j = (l2/r)·z_j·⟨w_j, g⊙u_contrast⟩`, with `u_contrast = u_winner − mean over 256
bins` (contrast-centered so a feature that lifts every bin equally scores zero),
`r` frozen.

Breadth of a feature = **participation ratio** over its per-task causal
contributions `C_j(g)`:
`PR = (Σ C_j)² / Σ C_j²` = effective number of tasks it influences (scale-free).

- **Mean PR ≈ 6.05 tasks.** Distribution is heavy-tailed: most features touch a
  couple of tasks, a minority spread across many.
- `adjusted_breadth` = PR rank-residualized on **magnitude** and **base_rate**,
  so "broad" cannot be an artifact of "just fires big" or "just fires often".

### A3. The core causal result (leave-one-task-out, confound-controlled)

Partial correlation between adjusted breadth and causal influence, holding the
confounds fixed, under **leave-one-task-out (LOTO)** so no single task drives it:

- **partial | both folds = +0.493**, **positive in 10/10 folds.**
- Robust to Spearman rank correlation and partial correlation (rank-residualize
  on magnitude + base_rate). Not a magnitude or base-rate artifact.

**Reading:** breadth of causal influence is a stable, real property of a feature,
not a firing-rate or amplitude shadow.

### A4. Feature interpretability

Frames captured at activation confirm the two ends are semantically what the
statistics claim:

- **General (high adjusted-breadth):** grasp events, release/open events,
  return-to-home pose — reusable manipulation *primitives*. The same feature
  fires at "grasp" across many different scenes and objects.
- **Specialist (low breadth, strong-but-narrow):** single-scene, single-phase
  detectors — e.g. one particular lid grasp in one particular layout. These are
  the memorization end.
- The general features are general *in when they fire* (across scenes) but
  specific *in what they do* (one primitive). This distinction matters for Path B.

> Fixed a selection bug here: "specialist" was initially picking ineligible weak
> features (no causal impact) instead of narrow-but-strong ones. Now selection
> ranks only causally-eligible features and slices both ends
> (`select_general_specialist`); caught by frame inspection, pinned by a
> regression test.

### A5. Replication

Path A reproduces across **all four LIBERO suites** (goal, spatial, object,
libero-10). Frames confirmed legitimate across suites. (Per-suite partial|both
numbers to be tabulated here as they are finalized.)

### A6. Answers to the obvious objections

- *"Of course hundreds of features influence the action."* — Influence is not the
  claim; **concentration and reproducibility** of influence is. A few features
  carry a large, LOTO-stable share of the *margin*, measured causally, not by
  counting nonzero gradients.
- *"Swann ablated features and it barely changed the model."* — Swann's own v2
  does find effects: real-world DROID ~65% with 4 general features per tier and
  0% with the top-4 removed; LIBERO 135 single-feature ablations. Weak *single*
  ablations are expected when influence is distributed across a coalition — which
  is exactly what breadth measures. (Coalition ablation harness built; see Open
  threads.)

---

## Path B — causal-signature recurrence across models

Setup: several independently fine-tuned models + a shared probe of frames. Ask
whether a feature's *causal role* reappears in other models, and whether that
recurrence tracks Path-A generality.

### B1. Activation-based recurrence is confounded (a distinctiveness artifact)

First-pass metric: max column-correlation of SAE codes across models on a shared
probe (`q_cross`), with a frame-shuffle permutation null and an inheritance
control (base activations through the fine-tuned SAE).

- Pooled `ret_cc ≈ 0.6`, but the action-position gap over null was only **+0.02**.
- Recurrence-vs-breadth came out **U-shaped** even after residualizing on
  base-rate and inheritance — high at both the specialist and general ends.
- The U-shape is a **distinctiveness artifact**: activation-correlation matching
  rewards features with sharp, low-entropy firing patterns (easy to match),
  independent of causal role. Not a generality signal.

### B2. Causal-signature recurrence (the properly-defined metric)

Redefine recurrence in a **common output space**: a feature's signature is how it
pushes all 256 action bins,
`S[j,t] = ⟨w_j, g⊙u_t⟩`, contrast-centered, cosine-matched across models.
Null = **random decoders through each model's own head** (calibrated so unrelated
model pairs give gap ≈ 0).

> Earlier bin-permutation null was **wrong**: all signatures share the head's
> low-dim subspace, so permuting bins rotates out of that shared geometry,
> deflating the floor and manufacturing a fake ~0.33 gap. Caught by a test;
> replaced with the random-decoder null.

Result:

- **Chance floor = 0.226.** Everything sits **above** it — shared causal
  structure across models is real.
- Recurrence **declines monotonically with breadth** — the U-shape is gone,
  confirming it was the activation-matching artifact. Specialists recur
  *slightly more*; the most general features recur *less*.
- Top-decile (most general) vs rest gaps are all small and **negative**:
  goal −0.017 (p≈1), spatial −0.015, object −0.008, libero-10 −0.011.

**Reading:** the most *causally general* features are, if anything, the *least*
cross-model recurrent. Specific primitives are more universal than diffuse
broad-influence signatures.

### B3. The A×B join

Directly correlating Path-A breadth with Path-B recurrence over features:

- **corr(breadth, recurrence) = −0.127.** Uncorrelated to mildly inverse.
- Same story from the dissociation test and the per-group comparison: general
  ≠ recurrent.

**"Recurrence ≠ generality" is now established three independent ways:** the
firing-based metric, activation recurrence, and causal-signature recurrence all
agree.

### B4. Methods hedge

SAE training is only **~60% seed-reproducible** (a feature re-derived under a new
seed matches its twin ~60% of the time). This bounds any recurrence claim and is
the reason recurrence is framed as a *distinct axis* rather than a null — one
line of caution, not the headline.

### B5. Why might Path B look this way? (candidate explanations)

1. **Usage vs existence.** Breadth measures how *broadly a feature is used* within
   one model; recurrence measures whether the *same causal object exists* in
   another. Different properties — no reason they must correlate. *(primary)*
2. **Reference-frame mismatch.** Generality is measured relative to each model's
   *own* task set, which differs across the independently fine-tuned models, so
   "broad here" need not mean "broad there". *(primary)*
3. **Distributed general solution vs universal primitive.** Broad causal
   influence may be a *diffuse, model-specific* way of solving many tasks, while
   specific primitives (grasp/release) are *universal* and thus recur. High
   adjusted-breadth selects for diffuse signatures, which are the hardest to match
   across models. *(interesting mechanistic possibility)*
4. **Signature-sharpness geometric confound.** Cosine matching may favor
   low-entropy (sharp) signatures; specialists have sharper signatures. Testable
   with a signature-entropy control. *(caveat)*
5. **Context-free signature vs contextual breadth.** The causal signature is
   measured context-free (per bin), while breadth is inherently contextual (across
   tasks), so the two need not align. *(caveat)*
6. **Deflationary.** The effect is a weak second-order one; the honest reading may
   simply be "recurrence and breadth are independent," full stop. *(deflationary)*

---

---

# Part 2 — Measurement audit, and two extensions (2026-08)

Part 1 established the two paths. Part 2 asks a different question: **which of those numbers
survive their own controls, and what were they actually measuring?** It is a measurement-validity
pass in the spirit of §2.2, applied to our own results rather than the paper's, plus two new
lines (B1 channels, A4 inventory recurrence) that were run to completion.

Everything here is on the goal suite unless stated otherwise. Tooling is in
`notes/elevation_diagnostics.md` and `notes/a4_b1_plan.md`; all of it is CPU re-analysis except
the channel pass and the second-seed SAE.

---

## P1. Path A replicates across all four suites, against a floor of zero

The A5 table is no longer empty. `partial | both` with the leave-one-task-out estimator, scored
by the same function `run_attribution.py` reports (`mrvla.attribution.loto_partial_both`), against
two permutation floors:

| suite | partial\|both | folds + | worst fold | column-shuffle floor | z | feature-shuffle floor | z |
|---|---|---|---|---|---|---|---|
| goal | **+0.493** | 10/10 | +0.399 | +0.0004 (sd 0.0094) | +52.1 | +0.0002 | +71.4 |
| spatial | **+0.449** | 10/10 | +0.333 | −0.0000 (sd 0.0094) | +47.9 | +0.0004 | +66.2 |
| object | **+0.387** | 10/10 | +0.359 | +0.0000 (sd 0.0096) | +40.2 | −0.0003 | +57.1 |
| libero-10 | **+0.535** | 10/10 | +0.399 | −0.0000 (sd 0.0096) | +55.8 | +0.0001 | +75.4 |

p < 0.001 (1000 permutations) in every cell. **Causal task-breadth predicts held-out causal
importance in all four suites, and the mechanical floor is zero.**

*The column shuffle is the floor that matters*: it permutes feature identity within each task
row, preserving each task's marginal distribution of causal mass and the purely mechanical
within-column link (an evenly spread column still has a predictable held-out entry) while
destroying a feature's identity across tasks. Whatever survives is arithmetic. The feature
shuffle is the estimator floor and must be ~0 or the estimator is biased; it is.

### P1a. The negative control §3.2b prescribed is a no-op (methods finding)

The plan called for permuting task labels and expected a collapse to ~0. Permuting task labels
*within a feature* does not collapse anything, because LOTO already evaluates every fold:
permuting column *j* makes fold *g* contribute the pair that fold π(g) contributed in the real
data — the same G pairs, dealt into different folds. Measured on our own data:

| suite | observed | task-label permutation |
|---|---|---|
| goal | +0.4926 | **+0.4840** |
| spatial | +0.4488 | **+0.4508** |
| object | +0.3866 | **+0.3843** |
| libero-10 | +0.5347 | **+0.5327** |

Running the control as written would have returned "the null equals the result" and looked like
the headline collapsing, when the permutation destroys nothing the statistic depends on.
Recorded so it is not re-attempted.

---

## P2. Concentration and reproducibility of causal influence, quantified

§A6 answers "of course hundreds of features influence the action" with *"concentration and
reproducibility of influence is the claim"*. Both words now have numbers.

| suite | effective #features (of 2048) | top-10 share | top-50 share | Gini | top-50 cross-task overlap |
|---|---|---|---|---|---|
| goal | **102.3** (5.0%) | 0.276 | 0.448 | 0.684 | 41.4/50 = **33.9× chance** |
| spatial | **105.9** (5.2%) | 0.256 | 0.436 | 0.644 | 44.8/50 = **36.7×** |
| object | **49.3** (2.4%) | 0.355 | 0.494 | 0.724 | 43.3/50 = **35.5×** |
| libero-10 | **9.6** (0.5%) | 0.588 | 0.731 | 0.889 | 42.7/50 = **35.0×** |

`n_eff` is the participation ratio applied over features instead of tasks — the same scale-free
statistic, so it needs no new justification. Chance overlap is N²/F = 1.22.

Two controls, both passed in every suite:
- **base firing rate** concentrates far less (goal n_eff 886.8, Gini 0.449 vs causal 102.3 /
  0.684), so concentration is a causal statement, not an activity one;
- **column-shuffled** causal mass gives n_eff 653.6 (goal), so the concentration comes from the
  *same* features being large across tasks, not from the marginal distribution.

Per-task `n_eff` is small in every task (goal 81.9 / 93.9 / 114.8 for min/median/max), so this is
not an averaging artefact.

> **Flag for review.** libero-10's `n_eff` = 9.6 is extreme — ten features carrying a ten-task
> long-horizon suite. The column-shuffle control (91.2) says the concentration is real rather
> than marginal-driven, but this should be eyeballed before it goes in a paper.

---

## P3. The coalition ablation is a *bounded* null, not an absent effect

`analyze_ablation.py` reported point estimates with no interval, no paired test and no power
bound, which cannot distinguish "the features are inert" from "this design could not see it".
Paired McNemar on the matched init states, with the design's minimum detectable effect:

**goal coalition run** (baseline 152/200 = 0.760 [0.696, 0.814]):

| condition | success | damage | 95% CI | b10/b01 | p (exact) |
|---|---|---|---|---|---|
| firing | 0.020 | **+0.740** | [+0.679, +0.801] | 148/0 | <0.001 |
| general | 0.710 | +0.050 | [−0.019, +0.119] | 30/20 | 0.203 |
| random | 0.755 | +0.005 | [−0.065, +0.075] | 26/25 | 1.000 |
| specialist | 0.785 | −0.025 | [−0.088, +0.038] | 18/23 | 0.533 |

**The design resolves 9.7 points pooled and 22.6 points per task.** General's +5.0 was never
detectable. The honest statement is *"general-coalition damage is below 11.9 points; anything
under 9.7 was outside this run's resolution"*. Detecting 5 points needs **79 episodes/task**
instead of 20.

The single-feature run is in the same position: pooled MDE 10.8 points, per-task 22.1, and every
pre-registered named-task prediction in `notes/ablation_goal_singles.md` was untestable at 20
episodes. `only_1134` came closest at −6.4 points [−13.3, +0.5], p = 0.108 — in the *repair*
direction.

The scope test (damage participation ratio) landed inside its sampling-noise null for every
condition, i.e. it did not resolve either. For the singles, damage PR was undefined — no task
showed positive damage at all.

---

## P4. Reliability bounds what any of this can support

**Split-half reliability of breadth** (Spearman-Brown corrected, disjoint task halves):

| suite | raw PR | adjusted breadth |
|---|---|---|
| goal | 0.221 | **0.363** |
| spatial | 0.349 | **0.273** |
| object | 0.637 | **0.469** |
| libero-10 | 0.424 | **0.337** |

The label-shuffle floor is ~0.000 in every suite, so these are real but low. **The population
axis is solid (P1); the feature-level ranking is only weakly reproducible.** Two consequences:

1. **The A×B null is not currently defensible.** With breadth reliability 0.363, the observed
   corr(breadth, recurrence) = −0.127 disattenuates to **|r_true| ≥ 0.211**, and the measurement
   ceiling is at most ±0.602. Recurrence reliability would have to exceed **0.494** for the true
   correlation to stay under 0.30. Substituting r_yy = 1 does *not* rescue it — that yields a
   lower bound on |r_true|, which argues against a null rather than for one. Until recurrence
   reliability is measured, "recurrence ≠ generality" at the feature level is a claim about noisy
   measurements. (P7 addresses this at the role level, where it is much less exposed.)
2. **Path A gets stronger for free.** Attenuation only ever weakens a correlation, so
   +0.493 implies **|r_true| ≥ 0.819**.

A practical corollary: the top-N coalition that was ablated was drawn from a ranking with
reliability ~0.36, so part of the coalition is noise, which dilutes any real effect. Selecting
targets by **split-half-stable** breadth is the cheap fix before spending 79 ep/task.

---

## P5. B1 — generality is *not* channel-localised (negative result)

OpenVLA emits 7 action tokens per decision, reusing the same 256 bins at every decode position,
so channel identity is positional. `run_attribution.py` discards that axis. Recovering it
(`run_channel_attribution.py`, 446k slot-decisions, argmax agreement with the emitted token
**0.9919**) gives a second breadth axis and tests a sharp hypothesis: *are general features
gripper/phase detectors?*

**They are not.** corr(adjusted breadth, gripper concentration), rank-residualised on magnitude
and base rate, is **+0.069**. The same partial for all seven channels:

| dx | dy | dz | droll | dpitch | dyaw | gripper |
|---|---|---|---|---|---|---|
| +0.081 | +0.083 | **+0.195** | +0.089 | +0.099 | +0.162 | +0.069 |

All small, all positive, gripper *lowest* of the group. General-vs-specialist channel profiles
differ by ≤0.07 with P(general > specialist) of 0.51–0.62. There is no channel story.

> All seven partials coming out positive is itself odd, since the seven concentrations sum to 1
> per feature and should partly cancel. That pattern is more consistent with a shared artefact
> than with seven channel-specific effects, and wants a shuffled-breadth control.

### P5a. Channel breadth is a genuine second axis

Effective number of action dimensions a feature drives: **3.21 of 7** (p10 1.33, p90 6.32), and
it correlates with task breadth at only **+0.238**. Two largely independent axes — the quadrant
map §3.3 wanted, built from data already collected.

### P5b. The gripper is a default, not a decision

Per-slot sufficiency (fraction of each channel's action margin the features additively recover):

| | dx | dy | dz | droll | dpitch | dyaw | gripper |
|---|---|---|---|---|---|---|---|
| features + bias | 0.940 | 0.934 | 0.905 | 0.949 | 0.920 | 0.937 | **0.992** |
| features alone | 0.607 | 0.833 | 0.593 | 0.903 | 0.923 | 0.776 | **−0.046** |

The gripper's margin is recovered almost entirely by the μ+b_pre bias; the features contribute
nothing. This puts a name on §2.6's *"constant default-action bias = 0.405"* — it is largely the
gripper. Necessity agrees: removing a feature changes the gripper token on 0.0006–0.0010 of
decisions it fires on, against 0.029–0.057 for dx.

*Caveat*: on a near-binary channel both the true margin and the bias term are functions of the
emitted token, so a high `recon` slope is partly tautological. The transition-conditioned figures
are the control.

---

## P6. What a projection ablation actually does

`hooks.py` ablates by projecting out the decoder direction, `h − (h·ŵ)ŵ`. Path A's φ describes
something narrower: removing the coded contribution `l2·z_j·w_j`. These are different
interventions, and the readout counterfactual (exact, no forward pass) separates them over all
446k decisions × 396 candidate features:

| | flip rate, all decisions | flip rate, given the feature fires |
|---|---|---|
| projection | 0.0229 | 0.0695 |
| coded | 0.0060 | 0.0642 |

Features are active on 9.4% of feature-decisions. Decomposing projection's 4.05M flips:
**1.15M happen when the feature fires, 2.89M when it does not — 71.5%.** Coded comes out at
−0.5%, i.e. exactly zero as it must (a consistency check on the whole pipeline).

**Most of what a rollout ablation does is act on residual structure the SAE never attributed to
the feature.** Given the feature fires, the two interventions are near-equivalent (0.0695 vs
0.0642); the large overall gap is entirely the non-firing decisions. This is a caveat on reading
any projection-ablation rollout as evidence about coded features.

### P6a. Necessity, base-rate controlled

Reported *given the feature fires* — over all decisions the rate mostly counts how often a
feature fires, which is the §2.2 confound in a new place (dx channel, goal):

| group | projection | coded |
|---|---|---|
| general | 0.0534 | 0.0569 |
| random | 0.0565 | 0.0564 |
| specialist | 0.0292 | 0.0342 |
| firing | 0.1284 | 0.1150 |

**General ≈ random > specialist.** The breadth ranking does not select features that are more
decisive per firing. Note this is not straightforwardly a failure: `adjusted_breadth` is
residualised on magnitude by construction, so it was never meant to predict per-decision
strength — it claims *breadth* of influence, which is a scope claim and is not what this table
measures. The specialist end being genuinely *less* decisive is the informative half.

---

## P7. A4 — the splitting hypothesis fails, and Path B is stronger for it

Path B matched features one-to-one (`q_causal = max_j cos`). That is structurally blind to
dictionary splitting, and splitting is not uniform: broad diffuse features are the ones expected
to fragment, so splitting alone could have manufactured "generals recur less". A4 tested this by
letting model B use *m* features to express one of model A's roles (`m` = 1 nests the published
metric).

**It buys nothing.** Cos-vs-m by adjusted-breadth decile, against the m-matched random floor:

| | m=1 | m=8 | slope |
|---|---|---|---|
| most-specialist decile | 0.3065 | 0.6052 | +0.299 |
| most-general decile | 0.2828 | 0.5881 | +0.305 |
| **random floor** | 0.2464 | 0.5684 | **+0.322** |
| seed ceiling | 0.4528 | 0.6974 | +0.245 |

Every decile rises at the same rate, and the random floor rises *faster* than any of them. The
entire gain from coalition matching is dimension counting. Chance-corrected retention is flat
across m: **0.244, 0.269, 0.272, 0.271, 0.268, 0.264, 0.259, 0.253**.

**The one-to-one null was not a fragmentation artefact.**

### P7a-bis. Replicated with each suite as the target

Re-run with spatial, object and libero-10 as the target model. The three new suites have no
second-seed SAE, so retention is unavailable there — but the claim being replicated is the
*slope* comparison, which needs no ceiling:

| target | general-decile slope | specialist-decile slope | difference | random-floor slope | general − floor |
|---|---|---|---|---|---|
| goal | +0.3053 | +0.2987 | +0.0066 | +0.3220 | **−0.0167** |
| spatial | +0.3040 | +0.2947 | +0.0093 | +0.3213 | **−0.0173** |
| object | +0.3049 | +0.2985 | +0.0064 | +0.3218 | **−0.0169** |
| libero-10 | +0.3166 | +0.3102 | +0.0064 | +0.3219 | **−0.0053** |

Two things hold in all four suites, and the second is the stronger statement:

1. The general-vs-specialist slope difference is **+0.006 to +0.009** — about 2–3% of the slope
   itself. Splitting does not preferentially help broad features anywhere.
2. **Every breadth decile in every suite rises with m more slowly than a random dictionary
   does.** The random-floor slope is +0.322 in all four suites (remarkably stable), and no
   decile reaches it. Coalition matching does not merely fail to help real dictionaries — real
   dictionaries gain *less* from it than random ones do, because the floor has more room to
   improve from a lower starting point.

The m=1 levels replicate the published direction in 3 of 4 suites (general deciles sit below
specialist ones in goal, spatial and object). **libero-10 is the exception**: its most-specialist
decile is the *lowest* at m=1 (0.2484 vs 0.2519 for the most general), so the "generals recur
less" ordering does not hold there. libero-10 is also the suite with the highest anti-aligned
fraction against every partner (0.25–0.37), and the one with the extreme concentration flagged in
P2. It behaves differently on several axes and is worth treating separately.

### P7a. The finding reproduces at the level of role inventories

Clustering each model's signatures and matching inventories by Hungarian assignment (not greedy —
greedy lets several roles claim one popular centroid; both are reported):

| k | inventory match | random floor | seed ceiling | retention | corr(role breadth, match) |
|---|---|---|---|---|---|
| 8 | 0.270 | 0.203 | 0.547 | +0.194 | **−0.810** |
| 16 | 0.266 | 0.160 | 0.540 | +0.279 | **−0.574** |
| 32 | 0.227 | 0.170 | 0.397 | +0.251 | **−0.537** |

General-dominated *roles* match across models worse than specialist-dominated ones, at every k.
The two independent methods agree on magnitude: cross-model recurrence retains ~**a quarter** of
what an SAE seed change retains (m-sweep 0.24–0.27, clusters 0.19–0.28).

*Ceiling check.* Retention divides by the same-model different-seed match, so a seed SAE trained
for a different number of epochs would bias it. Both goal SAEs are at 250 epochs, so the ceiling
is a like-for-like seed comparison and the retention figures stand as reported.

This also blunts the P4 reliability objection for this claim: it applied to a feature-level
correlation measured at reliability 0.363, whereas each point here aggregates 64–256 features.

**Reading.** *General features recur less across independently fine-tuned models. This is not an
artefact of dictionary splitting: it survives one-to-many coalition matching out to m = 8, and it
reproduces at the level of role inventories rather than individual features.* Path B moves from a
boundary condition with a methods hedge to a finding that survived the strongest available
alternative explanation.

### P7b. Three caveats on P7

1. k = 8's −0.810 is a correlation over 8 points; quote k = 16 and k = 32. None has a CI.
2. **Sliced-Wasserstein anomaly, unexplained.** Random dictionaries (0.0180–0.0182) are *closer*
   to goal's signature distribution than any real model (object 0.0186, spatial 0.0208,
   libero-10 0.0232; seed 0.0140). Distribution-level, the fine-tuned models differ from each
   other more than from noise. This wants an explanation before publication.
3. **12–37% of features' best absolute match is anti-aligned.** Since TopK codes are
   non-negative, an anti-aligned match is not the same role, so true recurrence may be lower
   than even these numbers.

---

## P8. Geometry of the action readout

Measured because every A4 statistic lives in this space and nobody had looked (`W_U_act` is
[256, 4096]):

- contrast-centred readout: effective rank **205.5**, 99% of energy in 250 directions;
- the space the goal model's 2048 features **actually occupy**: effective rank **50.8**, 99% at
  246. This is the number that governs A4, and it is comfortable — a random 12-dim subspace
  explains only 5.8% of an arbitrary direction, so the m-sweep is not dimension-counting by
  construction (it turned out to be so empirically anyway; see P7).

**The four models do not share an unembedding.** `g` and `act_ids` are identical, `W_U_act`
differs by max 0.0266 (1.64× the entry sd). This was initially assumed identical from
`train_lora.py`'s `target_modules`; that inference was wrong, because the four models are the
OpenVLA team's *published* checkpoints, not products of this repo's LoRA script. Each model's
decoder must therefore be pushed through its own head (`run_causal_recurrence.py` already did
this). The comparison remains well defined because the shared object is the 256 action **bins** —
bin *t* is the same commanded action in every model — not the map onto them. The perturbation is
cosmetic in practice: the same decoder through another model's head gives mean cos 0.9967–0.9990
(worst single feature 0.964).

---

## P9. Defects found in the existing pipeline

| defect | status |
|---|---|
| §3.2b's prescribed A4 negative control is a no-op (P1a) | corrected in `EXPERIMENT_PLAN.md` |
| `compare_recurrence_groups --target` was label-only, so a per-suite table could be silently mislabelled | fixed; now cross-checked against `--rec` and `--attr` |
| `_ranks` uses `argsort(argsort(x))`, which breaks ties by **array index** instead of averaging | **not fixed** — see below |
| `corr(damage, attribution)` bootstrap resampled tasks while treating each task's damage as fixed | fixed (two-level); the −0.770 it produced on the goal coalition was noise-on-noise |
| shared-unembedding assumption (P8) | fixed; per-model heads |

**The tie defect is live and it matters.** `base_rate` is a count over a fixed denominator, so
**10.3% of goal's features are tied**, and each tied block receives an arbitrary ordering
determined by feature index. This feeds `adjusted_breadth` — the ranking that selects every
ablation and steering target. `mrvla.stats.rankdata_average` is the correct version and all new
code uses it; the existing pipeline is deliberately left unchanged because fixing it moves
published numbers. **That should be a deliberate decision before publication, not a side effect.**

---

## P10. Where Part 2 leaves each claim

| claim | before Part 2 | after |
|---|---|---|
| Path A: breadth predicts held-out causal importance | goal only, no floor | **4/4 suites, floor ≈ 0, z = 40–56, \|r_true\| ≥ 0.82** |
| Influence is concentrated and reproducible | verbal | **n_eff 10–106 of 2048; top-50 recurs at 34–37× chance** |
| Coalition ablation | unreported null | **bounded null: damage < 11.9 pts, design resolves 9.7** |
| Generality is channel-localised (gripper/phase) | untested | **falsified: partial +0.069** |
| Breadth is a stable feature property | assumed | **reliability 0.27–0.47 — population yes, feature-level weakly** |
| Recurrence ≠ generality (feature level) | asserted | **not defensible without recurrence reliability (need > 0.494)** |
| Recurrence ≠ generality (role level) | not tested | **holds: −0.54 to −0.81, survives coalition matching** |
| Splitting explains the Path B null | untested, plausible | **falsified in all four suites: every decile rises slower than the random floor** |
| Rollout ablation tests coded features | assumed | **71.5% of its effect is on decisions the feature never fired on** |

---

## Open threads / next results to land here

- **Coalition ablation** (built, not yet run): top-N general vs specialist
  coalitions, per-task success, 5 conditions (baseline / general / specialist /
  random / firing-matched), multi-GPU sharded. Predicts general-coalition removal
  damages *many* tasks (high damage participation ratio); specialist removal
  damages *few*. Pending GPU allocation.
- **Four-suite Path A partial|both table** — fill in A5 with the per-suite numbers.
### Blocking — asymmetries a reviewer will see

- **B1 is goal-only.** Path A replicates 4/4 (P1) and the A4 m-sweep now does too
  (P7a-bis), but P5's channel result rests on goal alone. Three GPU jobs
  (`run_channels.slurm spatial|object|10`). Lower priority than it looks: B1 is a
  negative result, so a single-suite negative is a weaker claim but also a smaller one.
- **Second-seed SAEs for spatial / object / libero-10.** Without them the three new
  A4 targets report slopes but no chance-corrected retention, so "recurrence retains
  ~a quarter" is a goal-only number even though the slope conclusion is four-suite.
  One `run_sae_seed1.slurm` per suite.
- **`inventory_clusters.py` for the other three targets** — the role-level correlation
  (−0.54 to −0.81) is still goal-only. Minutes of CPU.

### Deferred — recorded, not blocking

- **Recurrence reliability is unmeasured** (P4). Decides whether the *feature-level*
  A×B null stands; the role-level result (P7a) does not depend on it, so this
  upgrades a caveat rather than unblocking a claim. Cheapest estimate: recompute
  `q_cross` on disjoint halves of the probe frames and correlate the two rankings.
- **The tie defect in `_ranks`** (P9). Real (10.3% of goal's `base_rate` tied) but
  second-order: it perturbs ordering *within* tied blocks of a ranking whose
  population-level behaviour is what every published claim rests on. Fixing it moves
  numbers, so it is a deliberate pre-publication call, not a bug to patch in passing.
- **libero-10's `n_eff` = 9.6** (P2) — extreme, control says real, wants an eyeball.
- **The sliced-Wasserstein anomaly** (P7b) — random dictionaries sit closer to goal's
  signature distribution than any real model does. Unexplained.
- **Signature-entropy control** for B2/explanation #4. `inventory_recurrence.py`
  already saves per-feature sharpness, so this is a re-analysis when wanted.

### A separate decision, not tidying

- **Re-sizing the ablation.** 79 episodes/task to detect 5 points (P3), with targets
  selected by split-half-stable breadth (P4). This is the difference between a
  *bounded null* and a positive behavioural result, and it is a real GPU spend — not
  a loose end.
- **`compare_recurrence_groups --target` bug** — *fixed.* `--target` was label-only
  while `--rec` chose the model, so a mislabelled per-suite table was undetectable.
  It is now cross-checked against both `--rec` and `--attr` and refuses to run on a
  mismatch (`--allow-mismatch` to override).
- **Recurrence reliability is not measured** — and without it the A×B null below
  cannot be defended against "both measures are too noisy to correlate". Cheapest
  estimate: `q_cross` recomputed on disjoint halves of the probe frames. See the
  diagnostics note.

### Diagnostics ready to run (CPU only, on existing artifacts)

`BASE=... ./run_diagnostics.sh` — see `notes/elevation_diagnostics.md` for what each
one is for and what it converts. In short:

| Tool | Closes |
|---|---|
| `ablation_power.py` | The coalition ablation null has no CI, no paired test and no power bound, so it is currently unreportable. Adds paired McNemar, damage intervals, the minimum detectable effect, and nulls for the damage-breadth and attribution-agreement tests. |
| `permutation_null.py` | A4's +0.493 has no floor. **The negative control the plan prescribed is a no-op** (permuting task labels within a feature leaves the statistic unchanged — LOTO already averages over every held-out task); this implements the two floors that do test something. |
| `causal_concentration.py` | §A6 claims "concentration and reproducibility of influence" with no number attached. Adds effective feature count, Lorenz/Gini shares, and cross-task top-N overlap vs chance, each against a firing-rate control. |
| `reliability_ceiling.py` | Attenuation correction. Strengthens Path A to a floor of ≥ +0.581 at reliability 0.72; shows the A×B null needs a recurrence-reliability estimate (r_yy > 0.249 suffices) before it can be claimed. |
| `split_half_breadth.py` | Breadth reliability — built, never reported; it is the input every correction above needs. |

---

*Metrics reference:* z = SAE code (post-TopK activation); φ = per-feature causal
contribution to the action; C_j(g) = feature j's causal contribution on task g;
PR = participation ratio (effective #tasks); adjusted_breadth = PR residualized on
magnitude + base_rate; q_cross = cross-model max code-correlation; S[j,t] =
256-bin causal signature; floor = random-decoder null.
