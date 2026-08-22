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
- **Part 3 (below).** The first link from the decomposition to behaviour. A closed-loop
  collection that carries both the action-position residuals and the episode outcome — the two
  had never been in the same dataset — shows that **coalition churn and bias share predict how
  long a successful episode takes** (rho −0.340 and +0.285 fully controlled, p = 0.0006 and
  0.002, both clearing correction for the ten signals tried). Episode FAILURE, by contrast, is
  a degenerate target on LIBERO: duration predicts it at AUROC 1.000, and eight signals against
  it are null.
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

> **These are the LINEAR-CONTROL values and every one of them is inflated by about a tenth.**
> See P1b: the control plane cannot represent curvature, and there is curvature. The figures to
> publish are +0.452 / +0.404 / +0.362 / +0.473.

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

### P1b. The control plane was too simple, and it was inflating every suite by ~10%

`partial | both` residualises ranked breadth and the ranked held-out row on the PLANE spanned by
ranked causal magnitude and ranked base firing rate. A plane represents only confounds that are
linear in each control and additive between them. Participation ratio is capped at the task
count (10) while magnitude is unbounded, so the relationship bends in rank space — and anything
the plane cannot represent survives residualisation and is scored as signal. The bias runs
*toward* the reported result, which is why this could not be left as an assumption.

It is not a hypothetical failure mode. In `tests/test_rankbasis.py` a fixture whose **only**
structure is a curved confound yields `partial | both` = **+0.40** under linear control — the
same order as the numbers above — and under +0.12 under every enriched basis.

`control_linearity.py` refits the identical estimator under progressively richer control bases
(quadratic, cubic, piecewise-linear splines, and the full tensor product of two spline bases,
which can absorb any smooth surface, additive or not):

| suite | linear | **tensor4 (reported)** | correction | max excess | dR²(pred) | floor under tensor4 | z |
|---|---|---|---|---|---|---|---|
| goal | +0.4926 | **+0.4516** | −8.3% | +0.0460 | 0.107 | +0.0003 (sd 0.0063) | +71.9 |
| spatial | +0.4488 | **+0.4036** | −10.1% | +0.0477 | 0.080 | +0.0006 (sd 0.0067) | +60.5 |
| object | +0.3866 | **+0.3624** | −6.3% | +0.0267 | 0.106 | −0.0004 (sd 0.0070) | +52.0 |
| libero-10 | +0.5347 | **+0.4725** | −11.6% | +0.0681 | 0.118 | +0.0008 (sd 0.0067) | +70.2 |

**The curvature is real.** `dR²(pred)` is the extra variance the nonlinear terms explain when
predicting ranked breadth from the controls: 8–12% in every suite. The largest correction lands
on libero-10, the suite with both the highest linear partial and the most concentrated causal
mass (n_eff 9.6 of 2048) — the configuration with the most room for a bend to distort.

**It changes no conclusion.** Every suite still clears its floor at z = 52–72, the floor under
the enriched estimator is still zero (so the richer basis bought no bias of its own), and all
10 folds stay positive under all five bases.

*Why a drop is not automatically evidence.* Extra columns lower an in-sample partial even when
they are pure noise, so every basis is scored against a PLACEBO of the same column count drawn
at random. The placebo moves the number by ≤0.0003 at 31 columns — essentially none of the
observed 0.04–0.07 is degrees-of-freedom bookkeeping.

*Which number to publish.* `tensor4`, fixed in advance and applied uniformly. **Not** the
minimum over the ladder: that is a minimum over five noisy estimates of one quantity, so the
selection biases it downward, by +0.003 to +0.006 here. The choice barely matters in any case —
the four enriched bases agree to within 0.005 (0.015 on libero-10).

*The assumption-free backstop.* `loto_stratified` bins features into 5×5 magnitude × base-rate
quantile cells and takes a plain rank correlation inside each. Both controls are near-constant
within a cell, so no functional form is assumed anywhere and curvature cannot produce the
result:

| suite | stratified | folds + | worst fold | cells/fold |
|---|---|---|---|---|
| goal | +0.347 | 10/10 | +0.256 | 19 |
| spatial | +0.344 | 10/10 | +0.294 | 19 |
| object | +0.281 | 10/10 | +0.256 | 20 |
| libero-10 | +0.297 | 10/10 | +0.254 | 19 |

Attenuated relative to the partial by construction — discarding between-cell variance in breadth
throws away real signal — so this corroborates the sign and rules curvature out as *the*
explanation; it is not expected to reproduce the headline magnitude.

*Rank ties, for the record.* The shipped estimator breaks ties by array index rather than
averaging them, and `base_rate` is a count over a fixed denominator so 6.6–10.6% of its values
tie. Measured cost on this geometry: **~1e-5** — four orders of magnitude below anything that
matters, because the arbitrary ordering is by SAE feature index, which is initialisation order
and correlates with nothing. A hygiene fix, not a correctness threat.

*What this does not address.* Both controls are measured with sampling noise, and residualising
on a noisy control under-removes it — also biasing toward a positive partial. That requires a
reliability estimate on `mag_tr` and `base_rate` which we do not have, and it stays a stated
caveat rather than a resolved one.

## P2. Concentration of causal influence, quantified

> **DECIDED: the reproducibility half is retired; concentration ships as one paragraph.**
> The cross-task overlap column below is recorded for the audit trail and **must not be
> published** — see P2b. The coalition is the always-on set and `base_rate` is a single global
> vector, so the recurrence is largely mechanical. The column shuffle is deflated by the same
> mechanism and no longer adds anything beyond the activity control.
>
> What ships: `n_eff`, the top-k shares, per-task stability, and the activity control — the last
> of which is the only part carrying non-trivial content, namely that causal mass is **not**
> proportional to firing frequency. Read the 8.7x with the caveat that base rate is bounded in
> [0,1] while mass is not, so the two distributions have different room to skew; it says these
> are qualitatively different, not that the ratio is exact. The paragraph must disclose that the
> most causally massive features largely coincide with the most frequently firing ones, so this
> establishes the SHAPE of the influence distribution and not the recruitment of task-specific
> machinery.

§A6 answers "of course hundreds of features influence the action" with *"concentration and
reproducibility of influence is the claim"*. Only the first word survived.

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
> than marginal-driven, but this should be eyeballed before it goes in a paper. Its jackknife
> interval is [9.4, 9.8] (P2a), so the number is precisely estimated — whatever is going on is
> a stable property of the data, not sampling noise.

---

### P2a. Error bars, the top-N knob, and a chance baseline that was assumed rather than measured

Three gaps in the numbers above, all now closed (`concentration_robustness.py`, CPU).

**The dictionary is fully used, so the chance baseline was right.** `N^2/F` assumes each task
draws its top-N uniformly from all 2048 features; if far fewer were ever causally active the
pool would be smaller and the reported ratios inflated. Measured: **2048 of 2048 features carry
causal mass in every suite.** The empirical baseline — overlap after permuting feature identity
within each task row, which inherits the pool, the marginals and the ties from the data —
agrees with the analytic one throughout (1.21 vs 1.22 at N = 50). The ratios above stand.

*This also resolves the k = 100 coincidence.* `n_eff` = 102.3 on goal is not a capacity
artefact: every feature fires somewhere, so a TopK budget of 100 per decision does not pin it.
It means something stronger — that close to the same 100 features are active decision after
decision, out of 2048 available. Object (49.3) and libero-10 (9.6) land well below k, which
rules out mechanical pinning.

**Delete-one-task jackknife intervals.** Between-task variability only: the saved artefact is
task-level, so within-task sampling noise is not recoverable and these are optimistic.

| suite | n_eff | jk SE | 95% interval | Gini | top-50 share |
|---|---|---|---|---|---|
| goal | 102.3 | 4.92 | [92.6, 111.9] | 0.684 ± 0.004 | 0.448 ± 0.004 |
| spatial | 105.9 | 1.62 | [102.7, 109.0] | 0.644 ± 0.002 | 0.436 ± 0.002 |
| object | 49.3 | 0.94 | [47.4, 51.1] | 0.724 ± 0.002 | 0.494 ± 0.003 |
| libero-10 | 9.6 | 0.09 | [9.4, 9.8] | 0.889 ± 0.002 | 0.731 ± 0.003 |

Every pair separates **except goal vs spatial**, which overlap. Suites may now be compared, with
that exception stated.

**The top-N sweep, and what it does to the headline.** The ratio to chance decays monotonically
with N in every suite — goal runs 183x, 73.5x, 34.2x, 16.6x, 8.4x, 3.5x across
N = 10/25/50/100/200/400. That decay is the expected signature of a shared core with a diffuse
periphery, and the overlap exceeds chance at every N (z = 178 to 291 throughout). But it means
**"34x chance" is one point on a curve and N = 50 is an arbitrary choice**; quoting it alone is
cherry-picking, since a reviewer running N = 200 gets 8.4x. Report the curve.

**The number to lead with instead.** The union of the per-task top-N sets needs no baseline
argument at all — it is a direct count, bounded by G*N:

| N | chance union | goal | spatial | object | libero-10 | compression |
|---|---|---|---|---|---|---|
| 10 | 98 | **15** | **13** | **11** | **13** | 7.5x |
| 25 | 237 | 40 | 30 | 33 | 33 | 7.0x |
| 50 | 448 | 95 | 67 | 74 | 82 | 5.6x |

Ten tasks, each task's ten most causally important features, and the union is **11 to 15
features**. Chance (`F(1-(1-N/F)^G)`) would give 98.

*A methods note on the empirical baseline.* Permuting across all F columns scatters mass into
structurally dead features — a state no task can produce — which collapses the baseline back to
`N^2/F` and reintroduces the assumption it exists to remove. On a fixture with 300 of 2048
active it reports chance 1.20 where the truth is 8.33, inflating the ratio sevenfold. The
permutation is confined to the active support; `tests/test_concentration_robustness.py` pins the
broken variant. It happens not to matter here because all 2048 features are active, but it would
have mattered silently in any suite where they were not.

---

### P2b. The recurring coalition is the always-on features (negative result)

`coalition_identity.py`. P2's controls establish that causal mass is more *concentrated* than
firing rate, but every one of them is a statement about the SHAPE of a distribution. None
constrains the IDENTITY of the top set — mass concentrated on exactly the most frequently
firing features would produce the identical n_eff, Gini and shuffle numbers. That reading was
untested. It is now tested, and it is what the data show.

| suite | top-50 by mass ∩ top-50 by base rate | Jaccard | coalition's base-rate percentile |
|---|---|---|---|
| goal | **41/50** (chance 1.22) | 0.695 | mean 98.6, **100%** above median |
| spatial | **40/50** | 0.667 | mean 97.5, **100%** above median |
| object | **42/50** | 0.724 | mean 97.8, **100%** above median |
| libero-10 | **43/50** | 0.754 | mean 98.5, **100%** above median |

**The coalition is the always-on set.** Every member of every suite's top-50 by causal mass
sits in roughly the top 1.5% of firing frequency.

**This breaks the reproducibility claim specifically.** `base_rate` is pooled over all decisions
in the suite — it is ONE global vector, not a per-task quantity — so "top-50 by base rate" is
the same set for every task by construction. If each task's mass ranking tracks that global
firing ranking, high cross-task overlap follows mechanically. The 34x figure in P2 is therefore
substantially measuring "causal mass tracks a task-independent quantity", not "the same task
machinery is recruited across tasks". **P2's reproducibility claim cannot be reported as it
stands.**

The column shuffle does not catch this. It destroys feature identity, which shows concentration
comes from identity — but it cannot distinguish *the same task-relevant features* from *the same
always-on features*.

**What survives.** Concentration. `n_eff(mass)` = 102.3 against `n_eff(base rate)` = 886.8 is a
genuine contrast: mass is 8.7x more skewed than firing, so mean |phi|-when-firing is not
constant across features. The membership of the top set is firing-determined; the shape of the
distribution is not.

**Path A is untouched, and this strengthens it.** Overlap between the mass coalition and the
top-50 by *adjusted* breadth is **0/50 in every suite**, with the coalition sitting *below*
median adjusted breadth (39.3 / 45.5 / 45.0 / 24.2 mean percentile). Path A residualised
magnitude and base rate out from the start; this is direct evidence that the control was both
necessary and effective. P1 and P2 are not two views of one object — they are two different
objects, and P2's is the less interesting one.

*Methods note on the definition.* The coalition is the top-N by POOLED mass, a pure magnitude
criterion. Defining it instead as "features in every per-task top-N" would make the breadth
question circular, since such a feature is forced to carry mass on every task, which is close to
the definition of a high participation ratio. The strict per-task intersection is reported
separately as a description of core stability: 30 / 36 / 35 / 32 features present in every
task's top-50, union 95 / 67 / 74 / 82.

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
2. **Path A gets stronger for free.** Attenuation only ever weakens a correlation, so a
   measured value obtained with a noisy instrument implies a larger true one:
   `r_true = r_obs / sqrt(r_xx * r_yy)`, and substituting the most favourable `r_yy = 1` gives
   a lower bound.

   **Use the curvature-corrected correlation, not the original.** With P1b's +0.452 and
   `r_xx` = 0.363 the bound is **|r_true| ≥ 0.750**. (An earlier version of this line used the
   pre-correction +0.493 and reported ≥ 0.819; P1b's correction had not been propagated here.)

   **Do not publish 0.750 as a number.** The DIRECTION is a theorem — measurement noise can
   only ever weaken a correlation, so the latent relationship is stronger than the measured
   +0.452. The MAGNITUDE rests on three things that do not hold cleanly here.

   *Two biases in `r_xx`, running opposite ways and neither measured.* The base-rate leak
   inflates measured agreement, so true `r_xx` may be below 0.363 (raising the bound). But
   Spearman–Brown assumes the halves differ from the full measurement only in LENGTH, and
   halving 10 tasks to 5 also halves PR's ceiling, compressing the scale; on a toy with this
   geometry SB underestimated by a factor of 1.32, which would put `r_xx` near 0.479 and the
   bound near 0.653. Across a plausible range the bound spans **0.58 to 0.90**.

   *The true-score assumption may not hold.* Classical test theory needs a fixed true value
   with random error around it. Participation ratio is defined RELATIVE TO THE TASK SET. If
   breadth genuinely differs across task sets — which `split_half_breadth.py`'s own docstring
   raises as a possible outcome and which was never resolved — then split-half disagreement is
   real variation rather than error, and disattenuating it is not valid at all.

   *Errors are plausibly correlated.* Attenuation requires the predictor's error and the
   target's error to be independent. LOTO makes the TASKS disjoint, but both quantities come
   from the same SAE, the same decoder direction, the same unembedding and the same rollouts,
   so a poorly-learned feature direction corrupts φ on both sides identically. Correlated
   errors inflate `r_obs`, and correcting upward from an inflated value compounds rather than
   removes the error. This is the one whose direction is actively unfavourable.

   *Bookkeeping, for whoever revisits this.* `r_true ≤ 1` implies `r_yy ≥ r_obs²/r_xx` =
   0.563, which is plausible for a quantity averaged over thousands of within-task decisions,
   so the arithmetic is at least internally consistent. Do not pair the raw-PR reliability
   (0.221) with this correlation — the LOTO estimator residualises the controls internally, so
   the matched reliability is the adjusted one, and the mismatched pairing demands an
   impossible `r_yy ≥ 1.10`.

   **The check that would settle the first two.** Vary the split size (2-vs-2, 3-vs-3, 4-vs-4,
   5-vs-5) and Spearman–Brown-correct each back to full length. Flat across sizes ⇒ the
   parallel-forms model fits and the correction is defensible. Drifting upward with size ⇒ SB
   is biased here and the trend gives the true full-length value. Low at every size ⇒ breadth
   is genuinely task-set-relative and the correction must not be applied. CPU, on the existing
   npz.

A practical corollary: the top-N coalition that was ablated was drawn from a ranking with
reliability ~0.36, so part of the coalition is noise, which dilutes any real effect. Selecting
targets by **split-half-stable** breadth is the cheap fix before spending 79 ep/task.

---

### P4a. Breadth is task-set-relative, not feature-intrinsic (and Spearman-Brown does not apply)

`split_half_sweep.py`. P4 corrects a single split-half correlation to full length with
Spearman-Brown. Two things had to be checked before that number could be used: whether the
correction applies to a participation ratio at all, and whether there is a fixed "true breadth"
for it to correct toward. Neither holds.

**Spearman-Brown does not apply.** Its one-parameter model implies the same per-task reliability
backed out of every split size. On a synthetic matrix with a fixed true breadth per feature and
independent per-task draws — exactly the model it assumes — the implied value still climbs,
because PR's CEILING moves with the split size (at half-size 2 the statistic lives in [1,2] with
massive ties; at 5 it has real resolution). The measurement changes qualitatively, not merely in
length. **P4 must therefore report UNCORRECTED split-half agreement at a stated half-length.**

**And there is no fixed true score to correct toward.** A calibration matrix is built to
reproduce each suite's observed PR distribution under a classical model — per-feature Dirichlet
concentration matched by `alpha = (PR-1)/(G-PR)`, independent per-task shares — so a true score
exists in it by construction. Four signals separate the real data from it, in the same direction
in all four suites:

| | goal | spatial | object | libero-10 |
|---|---|---|---|---|
| real rho, half-size 5 (raw) | 0.126 | 0.209 | 0.467 | 0.269 |
| **calibration** rho, half-size 5 | **0.811** | **0.822** | **0.853** | **0.854** |
| rho spread ratio (real / calibration) | **5.43** | **9.79** | **5.52** | **8.85** |
| excess drift in implied reliability | −0.135 | −0.186 | −0.161 | −0.126 |
| rho curve shape | non-monotone | non-monotone | non-monotone | non-monotone |

Reference points for the spread ratio, measured on dense synthetic matrices: **~1.4** for a fixed
true score measured noisily, **~13.8** for two regimes with uncorrelated breadth. The observed
5.4–9.8 sits firmly at the task-set-relative end.

The curve shape is the sharpest signal. Both reference models give rho RISING monotonically with
split size — more tasks per half, better agreement, as any sampling-error account requires. Every
real suite instead peaks at half-size 3 or 4 and falls at 5. The shuffle floor is flat at ~0.000
throughout, so the signal being measured is real; it simply is not the signal a fixed-true-score
model produces.

**Reading — and how much weight it deserves.** Causal breadth is a property of a feature
*relative to a task distribution*, not an intrinsic property of the feature. Split-half
disagreement is real variation rather than measurement error.

Note first how much of this is DEFINITIONAL. "General" here means high causal breadth over the
suite's tasks, and a participation ratio over G tasks is relative to those G tasks by
construction. The sweep adds one empirical fact beyond that tautology: the ranking does not
survive swapping which tasks are used, *even within a single suite*. That threatens a claim
nobody made — that there is an intrinsic general/specialist trait — and not the claim Path A
actually makes, which is about PREDICTION (breadth on nine tasks predicts causal importance on a
tenth) and does not require a stable latent trait.

So this is a wording caveat rather than a threat, and it bites in one place: A4's
characterisation of the two ends describes ONE PARTICULAR SELECTION, and a different ten tasks
would select somewhat different features. The one consequence that is not merely wording is the
disattenuation, which is invalid outright.

*One alternative reading, which cannot be separated here.* The calibration assumes per-task
variation is Dirichlet; genuinely heavier-tailed per-task variation would also depress agreement
without task-set-relativity as such. Both readings lead to the same three consequences, so the
defensible claim is the narrower one: **the classical model does not fit.**

**Consequences.**

1. *Disattenuation is invalid, not merely imprecise.* Variation that is not error cannot be
   corrected away. The |r_true| ≥ 0.750 bound in P4 is retired and does not return.
2. *Wording.* Not "feature j is a general feature" but "feature j has high causal breadth over
   this suite's task distribution". A4's characterisation of the two ends survives as a
   description of the selection; the word *general* carries more than the data supports.
3. *Path A is untouched.* LOTO measures breadth-on-nine-tasks predicting causal IMPORTANCE on a
   tenth — not breadth agreeing with breadth. Different quantities and different targets. That a
   task-set-relative quantity nonetheless transfers to an unseen task is arguably a stronger
   result than if it had been intrinsic.

*Methods note on building the calibration.* Two bugs voided an earlier run of this diagnostic and
are pinned by tests. Encoding breadth as the fraction of tasks a feature touches is degenerate
here — all 2048 features carry mass in every task, so the rate is 1.0 for all of them and the
reference had no true-score variation, returning rho = −0.002. And the spread-ratio threshold was
initially set at 1.5, inside the classical fixture's own range, which flagged classical data as
task-set-relative.

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

### P5c. Path A's causal mass is NOT gripper-weighted (a confound closed)

`mrvla/channels.py` warns that the gripper is near-binary, emits extreme bins, and therefore
carries the largest `||u_contrast||` — so every feature's |phi| should be inflated at slot 6 for
a purely geometric reason. That warning has a consequence nobody had checked, and it is upstream
of everything: `run_attribution.py` pools all seven slots, so if gripper decisions carry
disproportionate mass then `C[task, feature]` — and with it Path A, and Steps 1 through 4 — is
substantially a GRIPPER measurement rather than a general one.

Read off the saved `C_slot_abs` (no rerun; the split is also printed by
`run_channel_attribution.py:350`):

| dx | dy | dz | droll | dpitch | dyaw | gripper |
|---|---|---|---|---|---|---|
| 0.1353 | 0.1331 | 0.1377 | 0.1503 | 0.1453 | 0.1446 | **0.1537** |
| 0.95x | 0.93x | 0.96x | 1.05x | 1.02x | 1.01x | **1.08x** |

An even split is 1/7 = 0.1429. **The observed range is 0.93x to 1.08x, and the gripper is 1.08x.**
Causal mass is divided almost perfectly evenly across the seven action dimensions, so Path A is
not gripper-weighted and the confound is closed.

*This also undercuts the premise of the share correction.* If the geometric inflation were
biting, the gripper's absolute share would sit well above 1/7. It does not. Either the effect is
far smaller than the docstring assumes, or it is offset elsewhere in phi — fewer active features
at gripper decisions, or a larger residual norm `r`, both of which sit in the denominator. The
practical consequence is reassuring rather than alarming: absolute and share analyses should
broadly agree, and `analyze_channels.py` already flags any sign disagreement between them
("CONTRADICT -- believe share"). If that flag never tripped on the channel run, trap 1 was a
non-issue throughout and the correction is harmless rather than load-bearing.

---

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

# Part 3 — Behaviour: does anything internal predict what the policy DOES? (2026-08)

Parts 1 and 2 characterised features and audited the measurements. Neither touched behaviour,
for a structural reason: **internals and outcomes lived in disjoint datasets.** A1 stores the
seven action-position residuals but replays demonstrations, so `success` is the constant 1.
`libero_collect` carries real success labels but stores the mean-pooled prefill vector, which
is not what decodes the action (§2.3). Part 3 builds the intersection and uses it.

Everything here is the `goal` suite.

## P11. The closed-loop collection, and two canaries that had to pass first

`collect_action_rollouts.py` drives the policy closed-loop while capturing the L31 residual at
each of the 7 decode positions, the action actually executed, and the episode outcome. Rows are
buffered per episode and committed only once the outcome is known, so every row carries a true
label.

**500 episodes (50 init states x 10 tasks), 380 successes, 120 failures — 76.0%, matching the
published baseline exactly.** 562,072 slot-decisions, ~5 GB.

Two things had to hold before any of it was interpretable, and both are results in their own
right:

| canary | value | why it mattered |
|---|---|---|
| argmax recovery | **0.997–0.999** per slot | `predict_action` returns the action, not the tokens, so the emitted bin is recovered by argmax. The executed action is stored precisely to check that recovery rather than assume it. |
| sufficiency on rollout residuals | **0.9497** (features 0.5392, bias 0.4105, error 0.0503) | The SAE was trained on demo-replay states; these are self-generated. Demo replay gave 0.9361. **The decomposition transfers to states the policy produced itself** — this could have closed the whole line and did not. |

*Methods note, recorded because it cost a full run.* Four sharded workers wrote
`shard_00000.npz` onward into one directory, each with its own counter starting at zero, and
overwrote each other. 500 episodes were collected and 135 survived; the analysis globbed what
was left and reported a complete table of AUROCs on a 27% subset with no indication anything
was missing. Shards and manifests now carry a per-worker prefix and the writer refuses to
overwrite. **The failure mode was silence, not error.**

---

## P12. Episode failure is a degenerate target on LIBERO (methods finding)

`done` fires only on success, so a failed episode always runs to the step cap:

| | p5 | p50 | p95 |
|---|---|---|---|
| failure length | 300 | 300 | 300 |
| success length | 72 | 99.5 | 200 |

**Episode duration predicts failure at AUROC 1.000, with 0.0% overlap between the two
distributions.** Length is not a confound sitting beside the label — it is very nearly a
restatement of it.

Two consequences. First, no covariate adjustment can rescue an episode-mean statistic:
residualising on duration would remove the outcome along with the confound. Second, and beyond
this project: **any episode-level failure-prediction AUROC on LIBERO is partly reading a
clock** unless its aggregation is explicitly length-matched. Worth checking against
probe-on-hidden-state monitors that aggregate over whole episodes.

The signature is visible in our own numbers. Every signal peaked at the whole-episode mean and
collapsed in fixed windows:

| signal | discrimination at `mean` | at `first10` |
|---|---|---|
| share | 0.784 | 0.521 |
| mu_t@q25 | 0.761 | 0.505 |
| margin | 0.737 | 0.574 |
| phi_total | 0.638 | 0.544 |

Five structurally different quantities behaving identically points at a shared driver, and the
driver is duration.

---

## P13. Eight signals against binary failure: all null

Scored on length-matched windows (every episode contributes exactly N timesteps), pooled and
per task, with bootstrap CIs over episodes:

| signal | best fixed-window discrimination | verdict |
|---|---|---|
| `mu_t@q25`, `mu_t@q50` | 0.51–0.57 | null; sign flips across windows |
| `share` | 0.52–0.58 | null |
| `phi_total` | 0.54–0.59 | null |
| top-2 `margin` | 0.574 | null — **the baseline fails too** |
| feature churn / period-2 returns | 0.52–0.58 | null; beaten by the action-space equivalent |
| probe(z) / probe(h), LOTO ridge | 0.626 vs null 0.630; 0.535 vs null 0.524 | null (p = 0.61, 0.43) |
| **action churn** | **0.678** | **the only real predictor found** |

At 120 failures the AUROC SE is ~0.05, so anything between 0.55 and 0.60 is unresolvable —
this is a **bounded** null, in P3's sense, not a demonstrated absence.

The honest reading: *no internal signal predicts failure better than a two-line statistic on
the robot's own commanded actions.* Note `margin` failing too — the finding is not "the
mechanistic signal loses to the scalar baseline", it is that **nothing measured before
divergence predicts failure at this sample size.**

*A defect in the probe, recorded.* Its permutation null sits at 0.630 rather than 0.5, because
out-of-fold scores are pooled across folds whose ridge fits carry different offsets. The null
captures the bias so the conclusion holds, but it eats most of the power. Scoring within task
and averaging would fix it.

---

## P14. Duration among successes: the first uncontaminated target, and two positive results

Dropping the failures removes the degeneracy — among the 380 successes, duration runs 59 to 299
steps (median 99) and has no definitional link to anything. Signals are averaged over the first
**50 timesteps** (<= the shortest success, so every episode contributes exactly 50), correlated
with duration by Spearman **within each task** and averaged. Pooling across tasks would read
"this task is slow" as signal.

### The control taxonomy, which is the methods contribution here

A control is valid if it is a common cause of predictor and outcome. It is invalid if it sits
on the path from predictor to outcome. LIBERO is deterministic from a fixed init state, so
there is no exogenous during-episode variation — **only pre-episode variables can be
confounders. Everything measured during the episode is downstream of the policy.**

| control | what it asks | valid as a confound control? |
|---|---|---|
| `phi_total` (total causal drive) | is this more than total feature activity? | yes — a co-descriptor of the same decision |
| **initial geometry** (gripper-to-object distance at the init state, extracted by resetting the sim to each stored init state; no rollouts) | is this more than how far away things started? | **yes — fixed before the policy acts** |
| action churn, path length, straightness, gripper state | — | **no.** Computed from the commanded actions, i.e. the decisions being scored. Conditioning on them estimates a direct effect when the question is a total effect. |

That distinction was reached the hard way: `share` collapses under the behavioural controls
(+0.127) and survives the exogenous one (+0.317), and the second is the one that answers the
objection.

### Results, fully controlled (magnitude + initial geometry, no mediators)

Per-task rhos are noisy at ~38 successes per task, so the test is over the ten task-level
values: Fisher-z, one-sample t against zero, df = 9.

| signal | raw rho | controlled rho | tasks agreeing | t | p | Bonferroni (10 tests) |
|---|---|---|---|---|---|---|
| **feature churn** | −0.351 | **−0.340** | 9/10 | −5.20 | **0.00056** | **clears** |
| **bias share** | +0.363 | **+0.285** | 8/10 | +4.27 | **0.0021** | **clears** |
| `mu_t` | −0.368 | −0.251 | 7/10 | −2.81 | 0.020 | no |
| task margin | +0.143 | +0.182 | 7/10 | +1.97 | 0.080 | no |
| task margin, C rows SHUFFLED | +0.105 | +0.069 | 6/10 | +0.86 | 0.41 | — |

**Two signals survive.**

- **Coalition churn.** The more the driving coalition turns over between consecutive
  timesteps, the faster the episode finishes. Cleanest of the two: no bias term, no
  bin-dependent scale, purely which features are active.
- **Bias share.** The more of the action margin the constant prior carries early, the longer
  the episode takes.

The confound is real and does not explain them: initial gripper-to-object distance
independently predicts duration at **+0.342, 10/10 tasks**, and `share` moves only +0.363 to
+0.317 under it.

*The sign test was discarding the result.* At 8/10 `share` reads p = 0.11, but the ten task
rhos are consistent in magnitude and the t-test puts it at 0.002. Both statistics are reported
because the sign test is assumption-free and the t-test uses information it throws away.

### The per-task values, which say more than either test

Fully controlled, sorted:

```
share          -0.042  -0.030 | 0.162  0.164 | 0.362  0.371  0.424  0.437  0.474  0.529
feature_churn  +0.048  -0.064 | -0.267 -0.352 -0.362 -0.393 -0.426 -0.447 -0.567 -0.567
```

**No task contradicts either signal.** The apparent dissenters sit at zero, not on the other
side: -0.042 and -0.030 for `share`, +0.048 for `feature_churn`. At ~38 successes per task the
SE is ~0.16, so all three are indistinguishable from no effect. The defensible statement is
therefore stronger than the fraction suggests — *two of ten tasks show nothing and the other
eight all point the same way* — and it is also why the sign test underperformed here: it
scored two null tasks as full votes against.

**And the effect is heterogeneous, which belongs in the record.** Where `share` is present it
ranges 0.16 to 0.53, a threefold spread, plus two tasks where it is flat. `feature_churn` is
more uniform at 0.27 to 0.57 but still varies twofold. This is not "bias share predicts
duration in this policy"; it is "in most tasks, strongly in some, not at all in two". Which
tasks are flat is worth knowing rather than averaging away.

### Retracted

**"Specialist reliance predicts efficiency" does not survive.** `mu_t` was never significant
raw (8/10, p = 0.11); it reached 9/10 only under `|mag` and `|kin`, and the latter is a
mediator column. Under the valid controls it is −0.251 at p = 0.020, which does not clear
correction for ten signals. There is a clean mechanism for the shrinkage: `mu_t` is the share
of drive from low-breadth, scene-specific features, which fire when there is scene nearby — so
it is partly an inverse-proximity readout, and proximity is what the geometry control removes.

**Task-appropriateness is null.** Matching a decision's causal profile against the task's own
`C` row versus the other nine (the first use of Path A's `C` matrix for anything but the
participation ratio) gives +0.182 against a shuffled-row floor of +0.069. Neither clears.

---

## P15. What Part 3 does and does not establish

**Does.** The action decomposition corresponds to something behavioural. Two internal
quantities predict how long a successful episode takes, surviving every control that is valid
under the causal structure, and clearing correction for the number of signals tried. Before
this, nothing in the programme connected the feature/bias split or the coalition to behaviour
at all.

**Does not.**

| | |
|---|---|
| Causal | Nothing intervenes on either signal. The correlational reading is all that is licensed. |
| Practically useful | Controlling the robot's own commanded motion leaves `share` at +0.127. That is a mediator so it does not invalidate the effect, but it bounds the incremental value over simply watching the arm — and on the failure target, action churn beat every internal signal outright. |
| Two findings, or one | `share` and `feature_churn` may be two readouts of one state — a static bias-dominated coalition in transit versus a turning-over feature-driven one in manipulation. **Untested: partial each on the other.** |
| Interpreted | `\|bias\|` is ~99% determined by which bin was emitted (Gate 0: `frac_var_from_bin` = 0.991), so `share`'s variation is denominator-driven — it is closer to inverse feature coherence scaled by a bin-dependent yardstick than to "how much the bias contributed". `|feat| / phi_total` would measure that directly and has not been run. |
| Generalised | One suite, one model, one SAE seed, simulation. |

---

---

## Open threads / next results to land here

### Part 3 (behaviour) — in priority order

- **Are `share` and `feature_churn` one finding or two?** Partial each on the other. They may
  be two readouts of a single state (static bias-dominated coalition in transit vs turning-over
  feature-driven one in manipulation). One line, minutes of CPU, and it decides whether P14
  reports one result or two.
- **Feature coherence, `|feat| / phi_total`.** Gate 0 showed `|bias|` is ~99% bin-determined,
  so `share`'s variation is denominator-driven. Coherence measures that directly, with no bias
  term and no bin-dependent scale. Should be run beside `share`, not instead of it.
- **What do high-`share` and high-churn moments look like?** `capture_feature_frames.py`
  already pulls frames at chosen (task, episode, timestep). This is the question no further
  partialling can answer — whether the signal is transit-vs-manipulation, which would make it
  real but behaviourally obvious, or something else.
- **The probe's estimator is biased** (P13): pooling out-of-fold scores across folds with
  different offsets puts its null at 0.630. Score within task and average. Cheap, and it may
  change that row from null to something.
- **Replicate on the other three suites.** Part 3 is `goal`-only. Each needs one rollout
  collection (~4 h on 4 GPUs, ~5 GB) plus CPU re-analysis.
- **Log object and gripper pose during rollouts.** The geometry control uses INITIAL distance
  only. Per-timestep pose would allow a time-varying version, and it costs nothing to store.

### Part 1–2 threads

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
