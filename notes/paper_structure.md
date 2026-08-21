# Paper structure — draft skeleton

**This is a structural plan, not prose.** Every section lists the numbers that go in it and the
computation that produced each. Numbers trace to `results.md` sections in brackets.

Working frame: **a paper about measuring causal generality in a VLA, and what generality turns
out to mean when you measure it carefully.** The positive result is real; several negative
results narrow what it can be called. That narrowing is the contribution, not an embarrassment.

Target: ~8 pages body, everything else supplementary. **Body carries ~15 numbers. The other ~30
statistical procedures live in appendices and are referenced in one sentence each.**

---

## 1. Abstract

Four sentences, in this order:

1. SAE features in OpenVLA reconstruct **0.936** of the action margin — they are a causal handle,
   not a correlate. [A1]
2. A feature's *breadth* — how evenly its causal mass spreads across tasks — predicts its causal
   importance on a **held-out** task: **+0.562 / +0.506 / +0.432 / +0.575** across four LIBERO
   suites, 10/10 folds each, against a permutation floor of zero. [P1, P1b, C1]
3. But breadth predicts where a feature **writes**, not where it **decides**: the same estimator
   with a counterfactual target retains nothing once firing opportunity is controlled. [C1a]
4. And breadth is orthogonal to cross-model recurrence (**−0.127**), so "general" and "universal"
   are different properties. [B3]

**Do not put a method name in the abstract.** No "rank partial correlation under leave-one-task-out".

---

## 2. Introduction

- The question: are there features that drive *the policy* rather than *a task*?
- Why the obvious answers fail: "fires everywhere" and "fires hard" both produce apparent
  generality mechanically. Every number in the paper is controlled for both.
- Contribution list (4 items, matching the abstract).
- One paragraph flagging that several sub-results are negative and why that is the point.

**No numbers here beyond the four headline figures.**

---

## 3. Setup

| item | value | source |
|---|---|---|
| model | OpenVLA-7B, LIBERO | — |
| suites | goal / spatial / object / libero-10, 10 tasks each | — |
| SAE | TopK, F = 2048, k = 100, layer-31 residual at action positions | — |
| decisions | ~63.7k per suite, ×7 action slots = **446k slot-decisions** | [P5] |
| validation | recomputed argmax matches emitted token at **0.9919–0.9924** | [P5] |

**Action-space note (needed for §7 to be readable):** 7 action tokens per decision, all drawn
from one shared 256-bin menu, channel identity is *positional*.

**Attribution formula** — the one equation the body needs:

$$\phi_j = \frac{\ell_2}{r}\, z_j \,\langle w_j,\ g \odot u_{\text{contrast}}\rangle,
\qquad C[g,j] = \operatorname*{mean}_{\text{decisions in task } g} |\phi_j|$$

Contrast-centred: $u$ is the emitted bin's direction minus the mean over all 256. Without it every
feature is credited for the direction all tokens share.

---

## 4. Result 1 — the features carry the action (gate, not a finding)

**Numbers:** sufficiency = **0.936** at k=100, bar was 0.80. [A1]

**Computation:** through-origin calibration slope $S = \sum(\text{true}\cdot\text{pred})/\sum \text{true}^2$
on the action margin, decomposed as `features + bias + error` summing to 1.

**One sentence of honesty, in the body:** this is a calibration slope, not $R^2$ — it asks whether
the reconstruction tracks at the right *scale* and does not penalise scatter.

**Half a page. This is a gate that was passed, not a result.**

---

## 5. Result 2 — breadth predicts held-out causal importance ★ HEADLINE

### 5.1 The two definitions

$$\mathrm{PR}_j = \frac{(\sum_g C[g,j])^2}{\sum_g C[g,j]^2} \in [1, 10]
\qquad\text{(scale-free: measures spread, not strength)}$$

Controls: $\text{magnitude}_j = \sum_g C[g,j]$, and $\text{base rate}_j$ = fraction of decisions
where *j* fired.

**Numbers for context:** mean PR = **6.05 / 7.11 / 7.72 / 6.02**. [C1]

### 5.2 The estimator — half a page, described once, used everywhere

Leave-one-task-out. Predictor and both controls come from 9 tasks; target is the 10th.
Rank-residualise predictor and target on the controls, correlate the residuals
(Frisch–Waugh–Lovell, so this equals the multiple-regression coefficient — not an approximation).

### 5.3 Main table

| suite | partial \| both | folds + | worst fold | column-shuffle floor | z |
|---|---|---|---|---|---|
| goal | **+0.562** | 10/10 | +0.399 | +0.0004 (sd 0.0094) | +52.1 |
| spatial | **+0.506** | 10/10 | +0.333 | −0.0000 (sd 0.0094) | +47.9 |
| object | **+0.432** | 10/10 | +0.359 | +0.0000 (sd 0.0096) | +40.2 |
| libero-10 | **+0.575** | 10/10 | +0.399 | −0.0000 (sd 0.0096) | +55.8 |

**Floor caveat that must appear:** floors were computed under the pre-correction control; the
corrected partials clear them by more, not less. [P1]

### 5.4 The column shuffle — the only null described in the body

One paragraph. Permutes feature identity **within each task row**, preserving each task's
marginal mass distribution and every mechanical within-column link, destroying only a feature's
identity *across* tasks. Whatever survives is arithmetic.

### 5.5 Two corrections, one sentence each, running opposite ways

| correction | direction | why | source |
|---|---|---|---|
| control-plane curvature | **−6% to −12%** | PR is capped at 10, magnitude is unbounded, so the relationship bends in rank space and a plane cannot represent it | [P1b] |
| base-rate leak | **+0.070 to +0.111** | `base_rate` was computed over all 10 tasks including the held-out one, so the control carried part of its own target and over-corrected | [C1] |

Net path: `+0.493 → +0.452 → +0.562`.

**Body gets those two sentences. Appendix B gets the basis ladder and the four-arm leak
decomposition.**

---

## 6. Result 3 — what breadth means: writes vs decides ★ THE INTERESTING ONE

This is the section that turns a correlational result into a claim with edges.

### 6.1 The counterfactual target

Delete a feature's coded contribution from one decision, check whether the emitted token changes.
Exact, no forward pass, **446k decisions per suite** — against the rollout ablation's 200 episodes.

$$\text{flip rate}_{j,g} = \frac{\#\{\text{token changed}\}}{\#\{\text{decisions where } j \text{ fired}\}}$$

### 6.2 Same estimator, new target

| suite | goal | spatial | object | 10 |
|---|---|---|---|---|
| partial | +0.260 | +0.107 | +0.127 | +0.128 |
| folds positive | 10/10 | 10/10 | 10/10 | 10/10 |
| all floors, min z | +30.5 | +13.1 | +16.2 | +15.2 |

### 6.3 The dissociation — the table the paper is for

Condition additionally on the feature's **firing count on the held-out task**. Same predictor,
same controls, same folds, same basis. Only the target differs.

| target | goal | spatial | object | 10 | pattern |
|---|---|---|---|---|---|
| attributed mass (retained fraction) | **0.51** | **0.59** | **0.55** | **0.45** | tight, ~half |
| residual after control | **+0.287** | **+0.303** | **+0.240** | **+0.264** | consistent |
| decisiveness (retained) | 0.41 | −0.49 | 0.01 | −0.41 | mean ≈ 0, sign flips |

> **Breadth predicts where a feature *writes*, beyond opportunity. It does not predict where a
> feature *decides*, beyond opportunity.**

### 6.4 Supporting numbers

- ρ(breadth, firing opportunity) = **0.74–0.86** — the two are largely the same axis, which is
  why the control is severe [C1]
- ρ(attributed breadth, decisive breadth) = **0.765 / 0.794 / 0.761 / 0.798** [C1]

### 6.5 Two limits stated in the body, not buried

1. The opportunity control is a **lower bound by construction** — breadth partly *is* firing
   across tasks — so +0.24 to +0.30 floors the non-opportunity component rather than estimating it.
2. No permutation floor exists for the *controlled* statistic. What makes the contrast readable is
   that the same control on the same folds gives a consistent positive on one target and a
   sign-flipping zero on the other.

**The ratio-artefact null goes in Appendix C**, with one body sentence: *a target that is a ratio
can manufacture a partial from denominator structure alone; a binomial null holding every
denominator fixed rules that out (z = 13–36).*

---

## 7. Result 4 — concentration

**Numbers:** [P2, P2a]

| suite | n_eff (of 2048) | top-10 share | top-50 share | Gini |
|---|---|---|---|---|
| goal | 102.3 (5.0%) | 0.276 | 0.448 | 0.684 |
| spatial | 105.9 (5.2%) | 0.256 | 0.436 | 0.644 |
| object | 49.3 (2.4%) | 0.355 | 0.494 | 0.724 |
| libero-10 | **9.6** (0.5%) | 0.588 | 0.731 | 0.889 |

**Computation:** `n_eff` is the *same* participation ratio applied over features instead of tasks
— say this explicitly, it saves introducing a second statistic.

**The control that carries the content:** base firing rate concentrates far less (goal n_eff
**886.8**, Gini 0.449 vs causal 102.3 / 0.684). So concentration is a causal statement, not an
activity one.

**Mandatory disclosure, and it is not optional:** the top-50 by causal mass overlaps the top-50 by
base rate at **41–43 / 50**, every member in roughly the top 1.5% of firing frequency. So this
establishes the **shape** of the influence distribution, **not** the recruitment of task-specific
machinery. [P2b]

**Retired, must not appear:** the cross-task reproducibility claim (34–37× chance). The coalition
is the always-on set, so the recurrence is largely mechanical. [P2b]

**libero-10's n_eff = 9.6** — jackknife [9.4, 9.8], so precisely estimated. Flag it; it is one of
three measures on which libero-10 is an outlier (see §9).

**One paragraph. Not a section with subsections.**

---

## 8. Result 5 — recurrence is a different axis

### 8.1 The metric, and why the first one was wrong

Activation-correlation recurrence gives a **U-shape** vs breadth — high at both ends. That is a
*distinctiveness artefact*: correlation-matching rewards sharp low-entropy firing patterns
regardless of causal role. [B1]

Redefine in a shared **output** space: $S[j,t] = \langle w_j, g\odot u_t\rangle$ over all 256 bins,
contrast-centred, cosine-matched across models. [B2]

**Null design is a result in itself:** the bin-permutation null is *wrong* — all signatures share
the head's low-dimensional subspace, so permuting bins rotates out of that geometry, deflating the
floor and manufacturing a fake ~0.33 gap. Replaced with random decoders through each model's own
head. **This belongs in the body**; it is a mistake others will make.

### 8.2 Numbers

- chance floor **0.226**; everything sits above it — shared causal structure across models is real
- recurrence **declines monotonically** with breadth once the artefact is removed
- top-decile (most general) vs rest: **−0.017 / −0.015 / −0.008 / −0.011**, all negative
- corr(breadth, recurrence) = **−0.127** [B3]
- **splitting hypothesis falsified** in all four suites — the strongest alternative explanation,
  and it is dead at both feature and role level [P7, P7a]

### 8.3 The hedge, one line

SAE training is ~**60%** seed-reproducible. This is why recurrence is framed as a *distinct axis*
rather than a null. [B4]

---

## 9. Methods findings — what the measurement itself taught us

Reframe negatives as contributions. Each is one short paragraph. **These are useful to the field
independent of whether our headline holds.**

| finding | number | source |
|---|---|---|
| A rollout ablation is not a coded-feature ablation. **71.5%** of a projection ablation's flips land on decisions where the feature never fired. Projection 0.0229 vs coded 0.0060 overall; 0.0695 vs 0.0642 given the feature fires. | 71.5% | [P6] |
| The standard negative control (permuting task labels within a feature) is a **no-op** — LOTO already averages over held-out tasks. | — | [P1a] |
| A linear control plane inflates a bounded-vs-unbounded partial by ~10%. | −6% to −12% | [P1b] |
| A globally computed base-rate control leaks the held-out task. Placebo confirms it is not sample composition or control noise. | +0.07 to +0.11 | [C1] |
| A projection slope of zero cannot distinguish inert features from loud misaligned ones. Adding $\text{energy}=\sum\!f^2/\sum\!t^2$ and $\cos = \text{slope}/\sqrt{\text{energy}}$ recovers the geometry. Gripper feature term is 0.27–1.07× the margin's scale at cosine **−0.39 to −0.94** — opposed, not absent. | — | [P5b] |
| Generality is **not** channel-localised — the "general features are gripper/phase detectors" hypothesis is falsified, partial **+0.069**. | +0.069 | [P5] |
| Channel breadth is a genuine **second axis**: 3.21 of 7 effective action dimensions, correlating with task breadth at only **+0.238**. | — | [P5a] |

---

## 10. Limitations — write these before a reviewer does

1. **The behavioural ablation is a bounded null and cannot be rescued.** Design resolves **9.7
   points pooled, 22.6 per task**. General-coalition damage is **below 11.9 points**; +5.0 was
   never detectable. Detecting 5 points needs **79 episodes/task**; LIBERO caps at 50. [P3]
2. **Feature-level claims are weak.** Split-half reliability of adjusted breadth = **0.363 /
   0.273 / 0.469 / 0.337**. Population-level claims only, everywhere. [P4]
3. **Breadth is task-set-relative**, not feature-intrinsic. "Feature *j* has high causal breadth
   over this suite's task distribution", never "feature *j* is a general feature". [P4a]
4. **One model family, one benchmark.**
5. **The flip counterfactual is a direct-effect lower bound** on the readout: it freezes *r* and
   the rest of the sequence. "This feature decides this token", never "determines behaviour".
6. **Two known defects left unpatched at source** — the base-rate leak and index-order tie-breaking
   in `_ranks` (10.3% of goal's base rate is tied). Both are pre-publication decisions rather than
   silent fixes. [P9]
7. **libero-10 is an outlier on three independent measures** — n_eff 9.6, gripper mass share
   0.216 vs 0.143 even, gripper feature energy 1.14× the margin. One footnote, not three. [P2, P5b, P5c]

---

## 11. Appendices

| appendix | contents | body reference |
|---|---|---|
| **A. Estimator** | rank convention and tie averaging, FWL derivation, LOTO fold construction | §5.2 |
| **B. Control adequacy** | the five-rung basis ladder (linear → quad → cubic → hinge5 → tensor4), SVD orthonormalisation, dR²(pred) = 0.080–0.118, the degrees-of-freedom placebo | §5.5 |
| **C. Nulls** | all four permutation designs, why paired shuffling matters (unpaired sd 0.014 vs 0.022 — a tighter null inflates every z), the binomial denominator null | §5.4, §6.5 |
| **D. Base-rate leak** | four-arm decomposition (shipped / rebuilt-all / rebuilt-training / placebo), ρ(A,B) = 1.00000, placebo −0.004 to −0.015 | §5.5 |
| **E. Ablation power** | McNemar exact and corrected, Wilson intervals, MDE derivation, two-level bootstrap | §10.1 |
| **F. Reliability** | split-half, Spearman–Brown, and **why disattenuation was retired** — the bound moved 0.819 → 0.750 → 0.933 as unrelated inputs changed, which is evidence the estimator is unanchored | §10.2 |
| **G. Robustness sweeps** | min-active threshold ladder (flat, 20480/20480 cells kept), per-slot breakdown, drop-gripper, transitions-only | §6 |

**Body sentence that buys all of it:** *"The result is stable under five control bases, four
permutation nulls, a denominator-threshold sweep and a per-channel breakdown (Appendices B, C, G)."*

---

## Counts, for discipline

| | body | appendix |
|---|---|---|
| numbers | ~15 | ~120 |
| statistical procedures described | 4 (rank partial, LOTO, column shuffle, calibration slope) | ~26 |
| equations | 4 (φ, C, PR, calibration slope) | the rest |

**If the body exceeds ~20 numbers, something has migrated up that belongs in an appendix.**

---

## Open decisions

1. **Framing of §6.** Is the dissociation the *headline* or a *limitation of §5*? It is the more
   novel result and the more defensible one. Recommendation: headline it, and let §5 be the setup
   that makes it interpretable.
2. **Does Path B stay in this paper?** It is a clean result with a falsified alternative, but it is
   a different question and a different method stack. It could be a second paper.
3. **P5/P6/P6a are still goal-only** — computed on the candidate feature set with `--coeff both`.
   Re-running `analyze_channels.py` against `CHANNELS/<suite>_all` extends them to four suites for
   free: CPU, minutes, artifacts already on disk. Worth doing before submission.
