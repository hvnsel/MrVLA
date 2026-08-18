# Chapter 5 — Phase 2: auditing what those numbers mean

Phase 1 produced numbers. Phase 2 asks, of each one: **would this number look the same if the interesting thing were not true?** Every experiment in this chapter is a control, an extension designed to break a Phase 1 result, or a measurement of how much a Phase 1 result can bear.

Everything here is re-analysis of data already collected, except one GPU pass (§5.8) and one SAE retrain.

## 5.1 Why audit at all

Three specific worries motivated it.

1. **Path A's +0.493 had no floor.** It was positive in every fold, which is reassuring about consistency but says nothing about whether +0.493 is a large number *for a statistic of that shape*. Some statistics are positive on nearly any data.
2. **The ablation null was uninterpretable.** As stated at the end of Chapter 4, "we saw nothing" had two possible causes and no way to distinguish them.
3. **The A x B null rested on two noisy measurements.** §3.9 shows that unreliable measures *cannot* correlate strongly. Without knowing the reliabilities, `−0.127` might mean "these are different things" or "we cannot see anything from here".

## 5.2 Floors for Path A — and a control that does nothing

### The two valid floors

The claim is that breadth predicts held-out importance. Two different boring explanations need ruling out, so two different scrambles.

**Column shuffle** — *the one that matters*. Independently permute **feature identity within each task row** of the matrix `C`. This preserves every task's marginal distribution of causal mass, and preserves the purely mechanical fact that a column with evenly-spread values has a predictable held-out entry. What it destroys is a feature having an *identity across tasks*. Whatever survives is arithmetic, not biology.

**Feature shuffle** — the estimator floor. Permute feature identity of the held-out vector only, leaving the predictor and both control variables untouched. Nothing links predictor to target, so anything other than ~0 would mean the estimator itself is biased.

Both nulls call the identical function that produces the reported number, so there is no possibility of scoring a re-implementation that has drifted from the original.

### Result — all four suites

| Suite | `partial \| both` | Folds + | Worst fold | Column-shuffle floor | z | Feature-shuffle floor | z |
|---|---|---|---|---|---|---|---|
| goal | **+0.493** | 10/10 | +0.399 | +0.0004 (sd 0.0094) | +52.1 | +0.0002 | +71.4 |
| spatial | **+0.449** | 10/10 | +0.333 | −0.0000 (sd 0.0094) | +47.9 | +0.0004 | +66.2 |
| object | **+0.387** | 10/10 | +0.359 | +0.0000 (sd 0.0096) | +40.2 | −0.0003 | +57.1 |
| libero-10 | **+0.535** | 10/10 | +0.399 | −0.0000 (sd 0.0096) | +55.8 | +0.0001 | +75.4 |

*(`z` is how many null standard deviations the observed value sits above the floor; p < 0.001 in all cells with 1000 permutations.)*

**Reading.** The mechanical floor is zero. Path A replicates in all four suites at 40 to 75 standard deviations above chance. The A5 replication table, previously a promise, is now a measurement.

### The control the plan prescribed does nothing

The written plan called for permuting **task labels** and expected the statistic to collapse to ~0. It does not, and the reason is instructive.

LOTO already evaluates **every** fold. For feature `j`, holding out task `g` contributes the pair (breadth computed on the other 9, importance on `g`). Permuting column `j` by some permutation makes fold `g` contribute the pair that fold `perm(g)` contributed in the real data. Across all folds it is **the same set of pairs, dealt into different folds**. Since the fold-level results are all positive and similar in size, mixing them reproduces the statistic.

Measured on the real data:

| Suite | Observed | Task-label permutation |
|---|---|---|
| goal | +0.4926 | **+0.4840** |
| spatial | +0.4488 | **+0.4508** |
| object | +0.3866 | **+0.3843** |
| libero-10 | +0.5347 | **+0.5327** |

> **Why this is worth reporting rather than quietly correcting.** Had the control been run as written, it would have returned "the null equals the result" and looked like a catastrophic failure of the headline finding. It is a genuine methods trap: an intuitive permutation that destroys nothing the statistic depends on.

## 5.3 Concentration and reproducibility, quantified

The Phase 1 write-up rebutted "of course hundreds of features influence the action" with the claim that **concentration and reproducibility** of influence, not influence itself, is the point. Neither word had a number.

**What we compute.** From the per-feature total causal magnitude:

- `n_eff` — participation ratio over *features* (§3.5): the effective number of features carrying the action.
- Top-N shares and Gini (§3.11).
- **Cross-task overlap** — the average number of features shared between the top-50 sets of two different tasks, expressed as a ratio to chance. Two independent 50-subsets of 2048 features share `50^2/2048 = 1.22` by chance.

Two controls: the same statistics computed for a **base-firing-rate** ranking, and for a **column-shuffled** matrix.

| Suite | Effective #features (of 2048) | Top-10 share | Top-50 share | Gini | Top-50 cross-task overlap |
|---|---|---|---|---|---|
| goal | **102.3** (5.0%) | 0.276 | 0.448 | 0.684 | 41.4/50 = **33.9x chance** |
| spatial | **105.9** (5.2%) | 0.256 | 0.436 | 0.644 | 44.8/50 = **36.7x** |
| object | **49.3** (2.4%) | 0.355 | 0.494 | 0.724 | 43.3/50 = **35.5x** |
| libero-10 | **9.6** (0.5%) | 0.588 | 0.731 | 0.889 | 42.7/50 = **35.0x** |

Controls, both passed in every suite:

- **Firing rate concentrates far less** — for `goal`, `n_eff` 886.8 and Gini 0.449, against causal 102.3 and 0.684. So concentration is a *causal* statement, not an activity one.
- **Column-shuffled causal mass** gives `n_eff` 653.6 for `goal`. So the concentration comes from **the same features being large across tasks**, not from the shape of the marginal distribution.

Per-task `n_eff` is small in every individual task (`goal`: minimum 81.9, median 93.9, maximum 114.8), so this is not an artefact of averaging ten differently-shaped tasks.

> **Flagged for review.** libero-10's `n_eff` of 9.6 is extreme — ten features effectively carrying a ten-task long-horizon suite. The column-shuffle control (91.2) says the concentration is real rather than marginal-driven, but a number that strong should be examined directly before publication.

## 5.4 How reliable is breadth?

Split-half (§3.9): split the 10 tasks into disjoint halves, recompute PR **and** magnitude independently on each half, Spearman-correlate across features, 200 random splits, Spearman-Brown corrected. A feature-identity shuffle pins the floor.

| Suite | Raw PR reliability | Adjusted-breadth reliability |
|---|---|---|
| goal | 0.221 | **0.363** |
| spatial | 0.349 | **0.273** |
| object | 0.637 | **0.469** |
| libero-10 | 0.424 | **0.337** |

The floor is ~0.000 everywhere, so the procedure is calibrated and these are real but low.

**Reading, carefully.** These two statements are both true and not in tension:

- The **population-level axis is solid**. That is what §5.2 shows: breadth predicts held-out importance far above any floor, in every suite. Aggregated over 2048 features, a weak per-feature signal still produces a decisive population result.
- The **feature-level ranking is only weakly reproducible**. Ask "is feature 1167 more general than feature 1140?" and the answer changes substantially depending on which half of the tasks you measured on.

This bounds everything that depends on picking *individual* features — which includes the ablation and steering target selection.

## 5.5 Was the ablation null real, or was the experiment too small?

Phase 1's ablation reported point estimates and nothing else. We recompute with paired tests, intervals, and a power bound (§3.8).

**`goal` coalition run** — baseline 152/200 = 0.760, Wilson interval [0.696, 0.814]:

| Condition | Success | Damage | 95% CI | b10/b01 | p (exact) | MDE at 80% power |
|---|---|---|---|---|---|---|
| firing | 0.020 | **+0.740** | [+0.679, +0.801] | 148/0 | < 0.001 | 0.169 |
| general | 0.710 | +0.050 | [−0.019, +0.119] | 30/20 | 0.203 | 0.097 |
| random | 0.755 | +0.005 | [−0.065, +0.075] | 26/25 | 1.000 | 0.098 |
| specialist | 0.785 | −0.025 | [−0.088, +0.038] | 18/23 | 0.533 | 0.087 |

**The design's resolution**, from a median discordance rate of 0.253:

$$
  pooled (200 pairs)      MDE = 2.80 * sqrt(0.253/200) = 0.097   ->  9.7 points
  per task (20 pairs)     MDE = 2.80 * sqrt(0.253/20)  = 0.226   -> 22.6 points
  to detect 5 points      n   = 0.253 * (2.80/0.05)^2  = 793 pairs = 79 episodes/task
$$

**Reading.** General's +5.0 points was **never detectable** by this design. The honest statement is:

> General-coalition damage is below 11.9 points (the upper end of its interval); anything under 9.7 points was outside this run's resolution. The experiment does not show the features are inert.

That is a **bounded null** — a reportable result — rather than an absence of evidence.

The single-feature run is in the same position: pooled MDE 10.8 points, per-task 22.1. **Every pre-registered named-task prediction was untestable at 20 episodes per task.** The closest to an effect was one feature at −6.4 points [−13.3, +0.5], p = 0.108 — and in the *repair* direction, i.e. removing it slightly helped.

The **scope test** (does general-coalition damage spread over more tasks than specialist damage?) landed inside its sampling-noise null for every condition: it did not resolve either. For the single-feature run, damage participation ratio was undefined, because no task showed positive damage at all.

> **Discrepancy noted.** An earlier working note records the coalition baseline at 78.9%. The data files give 152/200 = 76.0%. The measured value is used throughout; the discrepancy is unexplained and is likely a partial versus complete run.

## 5.6 Can the A x B null be defended?

Applying the attenuation correction (§3.9) with the measured breadth reliability of 0.363:

| Quantity | Value | What it means |
|---|---|---|
| Observed `corr(breadth, recurrence)` | −0.127 | the raw finding |
| Measurement ceiling `sqrt(r_xx * r_yy)` | at most ±0.602 | the largest correlation these measures could ever show |
| Implied `|r_true|` lower bound | **≥ 0.211** | attenuation only shrinks, so the truth is at least this |
| Breakeven recurrence reliability | **0.494** | recurrence must be at least this reliable for `|r_true| ≤ 0.30` |

**Reading: at the feature level, the null is not currently defensible.** Recurrence reliability has never been measured. Until it is, `−0.127` is consistent with a real correlation of −0.21 or considerably larger, and the claim "recurrence is not generality" is a statement about noisy measurements rather than about models. (§5.12 addresses this at a level where the objection largely dissolves.)

The same correction applied to Path A works entirely in its favour:

$$
  |r_true|  >=  0.493 / sqrt(0.363)  =  0.819
$$

## 5.7 How many distinguishable causal roles are there? (the geometry gate)

Everything in the next three sections compares directions in the 256-bin signature space. Before running any of it, we measured how big that space actually is — because if it is tiny, then any handful of directions spans nearly all of it, matching becomes trivial, and a positive result would be dimension-counting rather than a finding.

Method: SVD (§3.12) of the contrast-centred `W_U_act`, and of the signature matrix the model's features actually occupy.

| Object | Effective rank | 90% energy | 99% energy |
|---|---|---|---|
| Contrast-centred readout `W_U_act` | 205.5 | 213 | 250 |
| **Signature space occupied by goal's 2048 features** | **50.8** | 193 | 246 |

The second row is the governing number. A random 12-dimensional subspace explains only **5.8%** of an arbitrary direction in a 50-dimensional space, so a coalition sweep out to m = 12 is interpretable. **The geometry is healthy; the experiment can mean something.**

### A correction the gate produced

The gate also checks an assumption that had been asserted rather than tested: that all four models share an unembedding matrix. **They do not.** `g` and the action token ids are identical, but `W_U_act` differs by up to 0.0266 — about 1.64 times the standard deviation of its own entries.

The assumption came from reading this project's LoRA fine-tuning script, which trains only attention projections. But the four models are the **OpenVLA team's published checkpoints**, not products of that script, and their unembeddings were modified.

**What this does and does not break.** Each model's decoder must be pushed through **its own** head. The comparison itself remains valid, because the shared object is the 256 action **bins** — bin `t` is the same commanded action in every model — not the linear map onto them. It is the map that varies, not the semantic space.

And the practical impact is small: pushing the same decoder through another model's head gives a signature cosine of **0.9967 to 0.9990** (worst single feature 0.964). Reported so that a raw weight difference is not mistaken for a meaningful one.

## 5.8 Measuring necessity exactly, without rollouts

A rollout gives one bit per episode and, as §5.5 showed, resolves nothing below ten points. But at layer 31 the readout is the *entire* remaining computation, so the counterfactual "what would the model have chosen without this feature?" can be computed **exactly, in closed form, with no forward pass at all**.

### The derivation

Write the unnormalised logits `L_t = (h (*) g) . u_t`. Removing a feature changes `h` by some multiple of its direction, so

$$
  h'      =  h  -  coeff * w_j
  L'_t    =  L_t  -  coeff * S[j, t]        S = the causal signature matrix from 4.6
$$

Three consequences make this cheap and exact:

1. **The RMSNorm factor `r` drops out of the argmax.** It is a positive number that scales every action logit equally, so it cannot change which bin wins. (It is still needed if you want logit *values*, but not for "did the choice change?".)
2. **Contrast-centring `S` is irrelevant here too** — it subtracts a constant from every bin of a row, shifting all logits equally.
3. **`S` is the same matrix Path B uses** for cross-model matching. One object serves both analyses.

So a **flip** — the feature's removal changing the emitted action — is exactly `argmax(L − coeff * S[j]) != argmax(L)`.

### Two different ablations, which are not the same intervention

This is a distinction the project had not previously drawn, and it turns out to matter.

| Name | What it removes | `coeff` | Matches |
|---|---|---|---|
| **projection** | the whole component of `h` along the feature's direction | `<h, w_j>` | what the rollout hook does |
| **coded** | only the amount the SAE attributed to that feature | `l2 * z_j` | what `phi` describes |

Projection removes everything lying along that direction — including SAE reconstruction error, the constant term, and overlap leaking in from other features. Coded removes only the feature's own coded contribution. Phase 1's rollouts used projection; Phase 1's attribution described coded. Nobody had checked whether they agree.

### Validation

Every path is verified against brute-force recomputation through the real RMSNorm and unembedding. On the real data, one further check is decisive: in coded mode a feature that did not fire has `coeff = 0` and **cannot** flip anything. Measured, coded flips on non-firing decisions come to **−0.5% of the total**, i.e. exactly zero within rounding. That single number validates the flip counting, the activity mask, and the accumulator wiring simultaneously.

Scale: **396 candidate features x 446,096 decisions = 176.7 million exact counterfactuals.**

> **Scope, stated precisely.** This is the exact **direct** effect on one decode slot at a **frozen** state. It excludes two other routes by construction: (a) within a timestep, the gripper slot attends to the six action tokens already emitted, so a feature could influence it indirectly; and (b) across timesteps, changing the action changes which state the robot next occupies. Both are real and both are invisible here. §5.10 returns to this.

## 5.9 B1 — is generality located in one action channel?

### The hypothesis

Phase 1's frame inspection said the general features look like grasp, release, and return-to-home detectors. All three are fundamentally **gripper and phase** events. That suggests a sharp, falsifiable prediction: causal breadth should concentrate in the gripper channel.

Testing it requires recovering an axis that Phase 1 discarded. The attribution code aggregates `|phi|` over tasks and, in doing so, sums over all 7 action slots. Recomputing while keeping the slot gives a per-channel causal matrix.

### Two traps, handled explicitly

**The scale trap.** Raw `|phi|` is **not comparable across slots**. `phi` carries `u_contrast`, whose length depends on where the emitted bin sits in the ordered range. The gripper is near-binary and emits extreme bins, which have the largest contrast norm — so *every* feature looks stronger there for a purely geometric reason. The fix is to work in **shares**: a feature's fraction of the total `|phi|` at that decision and slot. Both forms are computed, and a conclusion appearing only in the absolute numbers is treated as the confound rather than the finding.

**The degeneracy trap.** The gripper token is constant for most of an episode. A feature can score high there simply by dominating a low-entropy channel. So every statistic is also computed restricted to **gripper-transition timesteps** — the moments the command actually changes.

**Validation.** The recomputed argmax matches the token the model actually emitted on **99.19%** of decisions. (The 0.81% mismatch is most plausibly float16 storage of the residuals flipping near-tie decisions; it does not corrupt the flip analysis, which uses the recomputed argmax as its own consistent baseline.)

### Result: the hypothesis is falsified

`corr(adjusted breadth, gripper concentration)`, rank-residualised on magnitude and base rate: **+0.069**. The same partial for all seven channels:

| dx | dy | dz | droll | dpitch | dyaw | gripper |
|---|---|---|---|---|---|---|
| +0.081 | +0.083 | **+0.195** | +0.089 | +0.099 | +0.162 | **+0.069** |

All small, all positive, and the gripper is the **lowest of the seven**. The general/specialist channel profiles differ by at most 0.07 with `P(general > specialist)` between 0.51 and 0.62. **There is no channel story.** Generality is not localised to a channel.

> That all seven partials come out positive is itself odd, since the seven concentrations sum to 1 per feature and ought to partly cancel. That pattern is more consistent with a shared artefact than with seven channel-specific effects, and is listed as an open issue (§8.5).

### A positive by-product: channel breadth is a second axis

The participation ratio over the 7 channels gives the **effective number of action dimensions** a feature drives: mean **3.21 of 7** (10th percentile 1.33, 90th 6.32). Its correlation with task breadth is only **+0.238**.

So there are two largely independent breadth axes — across tasks, and across action dimensions — where the project previously had one.

### The gripper is a default, not a decision

Per-slot sufficiency (§4.5, computed per channel):

| | dx | dy | dz | droll | dpitch | dyaw | **gripper** |
|---|---|---|---|---|---|---|---|
| features + bias | 0.940 | 0.934 | 0.905 | 0.949 | 0.920 | 0.937 | **0.992** |
| **features alone** | 0.607 | 0.833 | 0.593 | 0.903 | 0.923 | 0.776 | **−0.046** |

Six channels are feature-driven. The gripper's action margin is recovered almost entirely by the **constant `mu + b_pre` bias**, and the features contribute nothing.

This identifies what §4.5's unexplained finding — "a constant default-action bias of 0.405" — actually *is*. It is largely the gripper.

Curiously, features put **more** `|phi|` mass on the gripper slot than on any other (0.154 of the run's total, the highest of the seven). They push hard on the gripper logits; their pushes simply do not align with which bin wins. **Loud, but not decisive.**

**Which half of this evidence is safe.** The sufficiency slope on a near-binary channel is partly tautological: both the true margin and the bias term are functions of the emitted token, so "the bias predicts the margin" comes close to being circular where only two tokens occur. The **flip rate is not tautological** and agrees: given the feature fires, removing it changes a `dx` token 0.0534 of the time and a gripper token **0.0008** — about 65 times less. On gripper-transition timesteps the gripper rate rises roughly ninefold (0.0008 to 0.0075) but remains far below translation. Lead with necessity; cite sufficiency as corroboration.

## 5.10 What a projection ablation actually does

Using both semantics from §5.8 over the full 176.7 million counterfactuals:

| | Flip rate, all decisions | Flip rate, given the feature fires |
|---|---|---|
| projection | 0.0229 | 0.0695 |
| coded | 0.0060 | 0.0642 |

Features are active on 9.4% of feature-decisions. Decomposing projection's flips:

$$
  total flips            = 0.0229 * 176,654,016  =  4.05 million
  when the feature fires = 0.0695 *  16,586,298  =  1.15 million
  when it does NOT fire  =                          2.89 million  =  71.5 %
$$

**71.5% of what a projection ablation does happens on decisions where the feature never fired.** Given the feature *does* fire, the two interventions are nearly equivalent (0.0695 versus 0.0642) — the entire gap is the non-firing decisions.

This is a direct caveat on reading any projection-based rollout ablation as evidence about *coded* features, and Phase 1's rollouts were projection-based.

### A confound caught in our own numbers

Over all decisions, coded flips showed general 0.0036 versus specialist 0.0007 — a 5.1x separation that looked like strong support for the breadth ranking. Conditioning on the feature actually firing:

| Group | projection | coded |
|---|---|---|
| general | 0.0534 | 0.0569 |
| random | 0.0565 | 0.0564 |
| specialist | 0.0292 | 0.0342 |
| firing | 0.1284 | 0.1150 |

The 5.1x collapses to 1.66x, and **random is indistinguishable from general**. Most of the apparent separation was base rate: a feature that fires twice as often has twice the opportunity to change an action, and in coded mode a feature that does not fire cannot flip anything at all. This is exactly the confound that destroyed the firing metrics in §2.2, reappearing in a new measurement.

> **Reading this carefully.** "General ≈ random" is not straightforwardly a failure of the breadth ranking. `adjusted_breadth` is *residualised on magnitude by construction* (§4.7), so it was never designed to predict per-decision strength — it claims *breadth* of influence, which is a scope claim, and per-decision flip rate is not a scope measurement. The informative half is that the **specialist** end is genuinely less decisive.

## 5.11 A4 — is the Path B null just an artefact of one-to-one matching?

### The alternative explanation worth taking seriously

Path B matched features **one-to-one**: `q = max over j of cos(signature of A's feature i, signature of B's feature j)`. That is structurally blind to the best-documented failure mode of sparse dictionaries — **splitting**. §4.11 established that dictionaries are only ~60% seed-reproducible, and the usual reason is that one fit's feature becomes two or three in another.

Crucially, splitting is **not uniform across features**. Broad, diffuse, high-usage general features are exactly the ones expected to fragment; sharp specialists stay atomic. **So splitting alone could have produced "generals recur less" with no difference in recurrence underneath.** This is the strongest innocent explanation available, and it needed testing.

### The test

Replace "does feature `i` have a twin?" with "**can model B's dictionary express feature `i`'s role using `m` features?**" — orthogonal matching pursuit (§3.13), sweeping `m` from 1 to 8. At `m = 1` this reproduces the published metric, so the experiment *extends* the existing result rather than replacing it.

**The nulls must be `m`-matched.** Cosine rises with `m` mechanically, so a null run at a different `m`, or against a dictionary of a different size, would manufacture a positive result out of dimension counting. Three controls all run at every `m`: random decoders through the corresponding model's own head; the same model under a different SAE seed (the ceiling); and, when available, the base model.

Two biases point in opposite directions and are both recorded. Unrestricted coefficients **overstate** what B can express, since TopK codes are non-negative and B can only *add* a feature's direction, never subtract it. Greedy matching pursuit **understates** it, since a locally-best first pick can lead away from the best group — on planted data where three features provably span the target, greedy recovers only 0.82 to 0.98 of it.

### Result: splitting is not the explanation

| | m=1 | m=8 | slope |
|---|---|---|---|
| most-specialist decile | 0.3065 | 0.6052 | +0.299 |
| most-general decile | 0.2828 | 0.5881 | +0.305 |
| **random floor** | 0.2464 | 0.5684 | **+0.322** |
| seed ceiling | 0.4528 | 0.6974 | +0.245 |

Every decile rises at essentially the same rate, and **the random floor rises faster than any of them**. Chance-corrected retention is flat across `m`: **0.244, 0.269, 0.272, 0.271, 0.268, 0.264, 0.259, 0.253**. Letting a model use eight features instead of one to express a role buys nothing.

### Replicated with each suite as the target

| Target | General slope | Specialist slope | Difference | Random-floor slope | General − floor |
|---|---|---|---|---|---|
| goal | +0.3053 | +0.2987 | +0.0066 | +0.3220 | **−0.0167** |
| spatial | +0.3040 | +0.2947 | +0.0093 | +0.3213 | **−0.0173** |
| object | +0.3049 | +0.2985 | +0.0064 | +0.3218 | **−0.0169** |
| libero-10 | +0.3166 | +0.3102 | +0.0064 | +0.3219 | **−0.0053** |

Two facts hold in all four suites, and the second is the stronger:

1. The general-versus-specialist slope difference is +0.006 to +0.009 — about 2–3% of the slope itself. Splitting does not preferentially help broad features anywhere.
2. **The random-floor slope is +0.322 in every suite and no decile reaches it.** Coalition matching does not merely fail to help real dictionaries; they gain *less* from it than random ones do.

**One exception, recorded.** The `m = 1` ordering reproduces the published direction in goal, spatial and object, but **libero-10 reverses it** — its most-specialist decile is the lowest. libero-10 also has the highest anti-aligned fraction against every partner (0.25 to 0.37) and the extreme concentration flagged in §5.3. It behaves differently on three separate axes and should be treated separately rather than averaged in.

## 5.12 A4 continued — comparing role inventories instead of features

If individual features do not match, perhaps the *set of roles* does. Cluster each model's signatures on the sphere (§3.14), match the cluster centres across models by Hungarian assignment (§3.13), and compare.

Three readings are produced. **Do the inventories match?** — centroid similarity against the same floor and ceiling. **Do the models weight roles the same?** — occupancy, the share of the dictionary each model devotes to a matched role, which is what duplication-style splitting would show up as. **Does the answer depend on `k`?** — cross-checked by a sliced Wasserstein distance (§3.15) that uses no clustering at all.

| k | Inventory match | Random floor | Seed ceiling | Retention | `corr(role breadth, match quality)` |
|---|---|---|---|---|---|
| 8 | 0.270 | 0.203 | 0.547 | +0.194 | **−0.810** |
| 16 | 0.266 | 0.160 | 0.540 | +0.279 | **−0.574** |
| 32 | 0.227 | 0.170 | 0.397 | +0.251 | **−0.537** |

**General-dominated roles match across models worse than specialist-dominated ones, at every `k`.** And the two independent methods agree on magnitude: cross-model recurrence retains about **a quarter** of what an SAE seed change retains (m-sweep 0.24–0.27, clusters 0.19–0.28).

This also substantially blunts §5.6's reliability objection *for this claim*. That objection applied to a feature-level correlation measured at reliability 0.363; here each point aggregates 64 to 256 features, where measurement noise is far smaller.

### An anomaly, unexplained

The sliced-Wasserstein column does something we cannot account for. **Random dictionaries sit closer to goal's signature distribution (0.0180–0.0182) than any real model does** (object 0.0186, spatial 0.0208, libero-10 0.0232; seed ceiling 0.0140). At the level of the whole distribution of directions, the four fine-tuned models differ from each other *more* than they differ from noise. This is recorded as an open problem (§8.5).

## 5.13 Defects found during Phase 2

Auditing turns up mistakes, including our own. The complete list:

| Defect | Where it came from | Status |
|---|---|---|
| The prescribed A4 negative control is a no-op | the written plan | corrected (§5.2) |
| `--target` was a label-only argument, so a per-suite comparison table could be silently mislabelled | Phase 1 tooling | fixed |
| Rank ties broken by array index instead of averaged | Phase 1 tooling, inherited | **open** (§8.3) |
| The attribution-agreement bootstrap ignored episode-level noise | introduced in Phase 2 | fixed (§5.14) |
| The shared-unembedding assumption | introduced in Phase 2 | fixed (§5.7) |
| The claim that SAEs were trained on pooled residuals | introduced in Phase 2 | **wrong, retracted** — they are un-pooled action-position residuals |

## 5.14 One Phase 2 statistic that was wrong, and how it was caught

Worth documenting because it is a good illustration of §3.10.

The first version of the "does damage land where attribution said?" interval resampled **tasks** while treating each task's damage as a fixed number. At 20 episodes, a per-task damage carries a standard error near 0.16 — larger than most of the damages being correlated. The interval was therefore far too confident, and it produced `corr(damage, attribution) = −0.770` with a 95% interval of [−0.96, −0.45] for the general coalition: an apparently decisive finding that damage lands in the *opposite* place from where attribution predicted.

The fix is a two-level bootstrap that also resamples matched episode pairs within each drawn task. Tested on planted data where damage is **pure noise** but happens by chance to correlate strongly with the profile: across every seed where the task-only interval wrongly excluded zero, the two-level interval put zero back inside, while a genuine effect still excluded it. One such seed goes from [−0.95, −0.61] to [−0.87, +0.02] — almost exactly the pattern of the real result.

So the −0.770 was noise correlated with noise. Given the per-task MDE of 22.6 points and observed per-task damages of ±5 to 20 points, every per-task damage value in that run *is* noise, which is exactly what the corrected interval now says.
