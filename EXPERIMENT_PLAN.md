# Experimental Plan: Measuring Generality in VLA SAE Features

**Status:** living plan. Section 2 records dated findings (things already tested and
ruled out); Sections 1 and 3–5 are the forward plan. The original pre-registration
(the classifier-replication study) is preserved in git history and summarised in the
Appendix — this document supersedes it after the reframe of 2026-08.

**Reference paper under study:** Swann, McGranahan, Buurmeijer, Kennedy, Schwager,
*"Sparse Autoencoders Reveal Interpretable and Steerable Features in VLA Models"*,
arXiv:2603.19183 (Stanford, 2026). Metrics/classifier: §3.2–3.3; OpenVLA results: Table 2.

**Substrate already built:** activations collected for OpenVLA fine-tuned on LIBERO
**Goal, Spatial, Object, 10**, at layers 0/8/16/24/31; TopK SAEs trained per (model,
layer) on the faithful recipe (ER=0.5→F=2048, k=100, 100 epochs); codes + firing
metrics extracted. **Caveat that shapes everything below:** the collected activations
are **mean-pooled over the prompt tokens and captured on the prefill pass only**
(`hooks.py`). They summarise the input, not the vectors that decode the action.

---

## 0. Question and philosophy

The scientific question is unchanged:

> **Does the VLA build reusable internal computations (general), or memorise specific
> situations (memorized)?**

This is a **measurement-validity** study, not a method proposal. Every phase has a
decision rule and yields a reportable result on both branches. Three principles, now
sharpened by what we found:

1. **Falsifiable at every level.** A metric that fails a validity check is a result.
2. **Confound-first.** No score is trusted until it is shown to predict something
   **beyond base firing rate** (activity). This principle already killed one metric
   family (§2.2) — it is load-bearing.
3. **Agreement over reliability.** Precise measurement is not validity. Validity comes
   from agreement across independent views (layers, seeds, models, behaviour).

---

## 1. Definition of generality (the reframe, 2026-08)

We **moved the definition off firing statistics and off human semantics.**

> **General** = a feature whose **causal influence on the policy's action recurs across
> many tasks**. **Memorized** = influence confined to one task/situation. Generality is
> a **continuous spectrum**, not a binary.
>
> **Human-interpretability is not required.** A feature may be general without any human
> being able to name what it responds to. Whether general features *also* happen to be
> interpretable is an empirical question we report, not an assumption we build in.

**Why the change.** The paper defines general as "fires across episodes for a
human-nameable event," welding together two properties that are actually independent:

- **Axis 1 — Task-breadth (causal):** does the feature *drive the action* across many
  tasks?
- **Axis 2 — Invariance (representational):** does it respond to the *same concept*
  across different appearances (scene, object instance, lighting)?

The lid feature (paper App. A.5.1) forced this split: it is **high invariance, low
breadth** — fires on all lid types/scenes, but only matters in the few lid tasks. A
grasp detector is high on both; a memorized feature is low on both; a broad control
signal with no clean concept is high-breadth/low-invariance. One number was averaging
two orthogonal axes. We measure them separately and place each feature in the quadrant.

**Consequence for existing work.** The 6 firing-based metrics (paper's coverage, mean
onsets, mean magnitude, relative run length; our group-balanced coverage, phase-
invariance) are **symptoms** of generality, not the definition. They are demoted to
*descriptors* (useful for characterising firing, e.g. clock detection) and are no longer
the generality measure.

---

## 2. What we have tested and ruled out (dated findings)

### 2.1 The paper's classifier is circular (finding, 2026-08)

The classifier is fit to 30 hand labels, but the labels are partly **defined by the same
four metrics** the classifier regresses onto (Stage-1 candidates screened by burstiness;
Stage-3 requires the global metrics to agree; ambiguous cases excluded — §3.3.2). So the
reported **100% LOO-CV measures label↔metric consistency, not construct validity**; it is
near-guaranteed by construction. This is **criterion contamination + range restriction**,
not fraud — and the paper documents the resulting failures itself (App. A.5.1: F1939
home-pose and F1381 lid both mis-classified). Reportable as a methods result.

### 2.2 Our structural firing metrics are a null (finding, 2026-08)

We built two label-free firing metrics — **group-balanced coverage** and
**phase-invariance** (`mrvla/structural_generality.py`) — and validated them by
**leave-one-group-out (LOGO)**: does the score on 9 tasks predict a feature's firing in
the held-out 10th? Raw LOGO looked strong (Spearman 0.3–0.8). **The confound control
killed it:**

- Under LIBERO's **balanced** task groups, `mean_group_rate ≡ raw episode coverage ≡
  base firing rate`. Partialling out base rate leaves nothing (partial correlation → 0,
  reported as `nan`). It was base rate all along — **and so is the paper's coverage.**
- `max_group_rate`, after controlling base rate, correlates **0.94 with concentration**
  and predicts held-out firing **negatively** (−0.05 to −0.27 across all 20 layer×suite
  cells). Controlled for activity, it is a **memorization** signal, not a general one.

**Root lesson:** a validation target of *firing* cannot isolate generality, because
firing is activity. Two equally-busy features (one general, one junk) get the same score.
The firing route is closed. (Diagnostics that proved this — saturation, unsaturated/
clock-excluded LOGO, base-rate-partial — are in `structural_generality.py` and stay as
the record.)

### 2.3 Attribution gate is red on existing data (finding, 2026-08)

The causal route (§3.2) needs the **un-pooled last-layer residual at the action-token
positions** + the action head. Existing activations are **mean-pooled + prefill-only**
(`hooks.py`), i.e. the wrong vectors, and the SAE was trained on that distribution.
**Layer-0 of the viability gate fails without a data rebuild** (re-collect action-position
residuals at L31, log actions/logits, retrain an SAE on them). Recorded so we do not
re-discover it.

### 2.4 Path B is a positive result: recurrence is a real generality signal (2026-08)

Cross-model recurrence (§3.1) was executed in full and **passed every control**, on the
shared 1000-frame probe across all 4 fine-tuned models x 5 layers (20 cells):

- **Above chance.** `gap = q_cross - q_perm` is **+0.13 to +0.26 in all 20 cells**
  (`q_cross` ~0.37-0.54 vs a permutation floor ~0.24-0.28). The permutation null shuffles
  probe-frame order, preserving each feature's marginals *and* the best-of-2048 maximum,
  so it is the correct chance baseline for max-matching.
- **Not activity, not inheritance.** `conf_R2` (rank(q_cross) regressed on
  rank(base_rate) + rank(inheritance)) is **0.001-0.113**, i.e. **89-99.9% of the
  recurrence ranking is explained by neither confound**. Inheritance is measured by
  pushing *base-model* residuals through each fine-tuned SAE (no base SAE needed).
- **Discriminating.** q_cross spreads from ~0.27 (10th pct) to ~0.69 (90th pct), so
  recurrence separates features rather than scoring them alike. This is also the answer
  to "the 4 models share a base checkpoint": mere network similarity would make all
  features recur equally.
- **Calibrated.** With a second-seed SAE (`--seed 1`) for the goal model, chance-corrected
  retention `ret_cc = mean(q_cross - q_perm) / mean(q_seed - q_perm)` is a stable
  **0.587 / 0.619 / 0.635 / 0.585 / 0.473** at L0/8/16/24/31. Changing the entire
  fine-tuning suite costs only ~40% relative to merely changing the SAE seed.

Structure: above-chance recurrence peaks mid-network (L8-L16) and is lowest at the output
layer L31 — which is simultaneously the *purest* (conf_R2 ~0, essentially no inheritance).

**Decision-rule outcome: row 1** of §3.1 (`Δ* > 0`, resolvable above the noise floor).
Row 3 (noise floor unresolvable) is ruled out: `q_seed` ~0.53-0.66, well above the ~0.25
chance floor.

### 2.5 Two findings the plan did not anticipate (2026-08)

**(a) SAE dictionaries are only ~60% reproducible across seeds.** Matching the goal
model's SAE against a second SAE trained on the *same* activations with a different seed
gives `q_seed` = **0.640 / 0.657 / 0.629 / 0.612 / 0.531** (L0/8/16/24/31), not ~1.0. Even
the same model re-analysed does not perfectly re-find its own features. This is a
standalone methods result: any conclusion from a single SAE fit inherits that
instability, and reported feature indices should be read in that light. It also supplies
the reliability estimate used in (b).

**(b) Recurrence and the paper's `P(general)` are essentially uncorrelated.** Computed on
the same SAE and feature indices, `spearman(q_cross, P_general)` lies in **-0.17 to
+0.17** across all 20 cells; top-100 overlap is 3-5x chance but with no overall
relationship. **This null survives correction for measurement reliability:** with
reliability ~0.6 the disattenuation factor is ~1.3, lifting the largest magnitude to
~0.22 — still negligible. So the paper's coverage/burstiness score tells you almost
nothing about whether an independently fine-tuned model rediscovers a feature. Given
§2.2 (that score is base rate), this is arguably the project's headline claim and had no
section in the plan.

**Outstanding commitment.** §3.1 step 3 requires reporting **greedy and Hungarian**
assignment; only greedy has been run at scale. Hungarian is implemented and needs no
re-encoding. Similarly the second seed exists for **goal only**; §3.1 asks for >=2 seeds
per (model, layer).

### 2.6 Path A is a positive result: causal task-breadth survives confound control (2026-08)

A1 (action-position collection) and A2 (SAE retrain on action-position residuals) closed
the L0 gap of §2.3. The gate then went through a documented pivot, and A4 came back
positive on the goal model.

**The gate pivot (L1 → sufficiency).** L1 (full-residual argmax re-decode) **stalled at
0.72** and would not clear 0.85: +150 epochs moved it 0.70→0.72, and raising sparsity
k=100→256 only reached 0.76 on a shallow slope (clearing 0.85 would need k≈500–1500,
abandoning sparsity). Diagnosis: L1 punishes the SAE for reconstructing the *entire* dense
last-layer residual (information for the whole 32k vocab), most of it action-irrelevant,
and argmax over tightly-packed bins is brittle. **Pivot:** gate on **sufficiency** — the
fraction of the *action-margin* logit the features additively recover — measured as a
through-origin slope `S = Σ_i c_i x_i / Σ_i c_i²`. This is the right estimand (does the
decomposition explain the *action*, not the whole residual), it is magnitude-aware where a
correlation is not, and a genuine signal clears it comfortably rather than at the margin.
Recorded per commitment #5 (no outcome-dependent switching hidden): L1 is still reported
beside sufficiency; the metric changed for a stated reason and before reading the value.

**Gate result (goal, k=100, no retraining):** sufficiency `S = 0.9361` (features+bias
recover 93.6% of the margin; threshold 0.80 → PASS), of which **features alone = 0.531**
and a **constant default-action bias = 0.405** (μ+b_pre; a small finding — part of VLA
action is a fixed lean toward rest), error ≈ 0.064; the three slopes sum to 1 by identity.
Bug canary L2b = 1.0000 (mean|diff| 2.7e-14 — the frozen-r arithmetic is exact).

**A4 result (446,096 decisions, 10 tasks, 2048 features):** PR mean 6.05, p10 2.65,
p90 9.91 — a real specialist→generalist spectrum. Raw breadth is heavily activity-entangled
(`PR ~ base rate = +0.82`), so the raw statistic is *not* the claim. The claim is the
leave-one-task-out partial predicting held-out causal importance controlling **both**
magnitude and base rate: **partial ρ = +0.493, 10/10 folds positive, min +0.399, sd 0.053.**
Causal cross-task breadth is a real, firing-independent, out-of-sample axis.

**Honest scope (what this is NOT).** No individual feature has been *identified* and no
*count* of general features exists — a count needs a non-arbitrary cutoff (bimodality test),
untested. The attribution is first-order (frozen r); the nonlinear/behavioural seal is
coalition ablation (§3.2a). All numbers are the goal suite only.

---

## 3. Method: measuring generality without firing and without labels

Generality must be validated against a target that is **not firing rate**. Three such
targets; every one keeps the confound-first discipline (must beat base rate).

### 3.1 Path B — Cross-model recurrence **(DONE 2026-08 — positive; see §2.4)**

**Idea.** A feature is general to the degree the model **rediscovers it when
independently fine-tuned on a different task suite.** Recurrence is the generality
measure — base-rate-independent (it is about the activation *pattern* matching across
models, not how often the feature fires) and label-free.

**Why first.** It runs largely on artifacts we already have (the 4 fine-tuned models
and their SAEs); it needs no action head, no decode-position residuals, and no SAE
retrain. Cheapest path to a first non-firing generality signal.

**Procedure.**
1. **Shared probe set.** Choose a fixed set of probe frames and push the *same* frames
   through each of the 4 models, capturing residuals at each layer (a small collection —
   forward passes only; the pooled activations we already have suffice for matching).
2. **Encode** each model's residuals with *that model's* SAE → activation matrices
   `Z^A [N_frames, F]`, `Z^B [N_frames, F]`, …
3. **Match features.** For feature *i* in model A, `q_i = max_j corr(Z^A_i, Z^B_j)` over
   the shared frames (report greedy **and** Hungarian assignment). High `q_i` = the
   feature has a twin in B = it recurs. `q_i` averaged over the other models is the
   **recurrence-generality score.**
4. **Confounds (mandatory).**
   - **Base rate.** Check `q` predicts beyond base firing rate (busy features may match
     trivially) via the same partial-correlation control that killed §2.2.
   - **Shared-base inheritance.** All 4 models share the base OpenVLA checkpoint, which
     inflates matches. Cancel with the differential against the **same-model /
     different-seed** noise floor:
     `Δ* = [q]_cross-model − [q]_same-model-different-seed`.
     *Requires ≥2 SAE seeds per (model, layer)* — verify these exist; if not, train the
     second seed (this was a standing commitment in the original plan §1.3).

**Decision rule.**

| Outcome | Interpretation |
|---|---|
| `Δ* > 0`, resolvable above noise floor | Recurrence is a real, base-rate-free generality signal; high-recurrence features are the "universal" ones. Positive result — proceed to characterise them (interpretable? which quadrant?). |
| `Δ* ≈ 0`, noise floor resolvable | Features are model-local; generality does not survive independent fine-tuning. Coherent negative. |
| Noise floor unresolvable (seeds match no better than chance) | SAE non-identifiability dominates at this scale — a methods result; **must not** be read as evidence about generality. |

### 3.2 Path A — Causal task-breadth (attribution) *(DONE on goal 2026-08 — gate passed via sufficiency, A4 positive; see §2.6)*

**Status: gate PASSED on goal, A4 positive (§2.6). Scope: layer 31 only.** A1–A3 built and
run; the gate was pivoted from L1 to **sufficiency** (§2.6) and passed at 0.936 on the
original k=100 SAE; A4 gave partial ρ = +0.493 (10/10 folds). Layer 31 only, because the
linear decomposition below is exact only where the residual feeds the readout directly;
earlier layers need gradient/path-patching (a separate decision). See the full design doc
and `PathA_Technical_Report.pdf` for the from-scratch derivation.

**Definition operationalised.** OpenVLA emits an action token by `logit(t) = RMSNorm(h)·u_t`
at the action position — a dot product, hence additive. With the SAE writing
`h ≈ l2·(Σ_j z_j w_j) + μ·1 + b_pre`, feature *j*'s contribution to the emitted token *t* is

  `φ_j = (l2 / r) · z_j · ⟨ w_j , g ⊙ u_contrast ⟩`,  where `r = rms(h)`, `g` = final-norm
  gain, `u_contrast = u_t − mean_s u_s` over the 256 action tokens.

Two implementation points that a naive `φ = z·⟨w,u⟩` gets wrong: (i) carry the **per-sample
`l2` factor** (our SAE normalises each sample), and (ii) attribute to the **contrast**
`u_contrast` not the raw `u_t`, so directions that lift all action logits equally get no
credit. This escapes the activity confound: a feature can fire hard (`z_j` large) yet have
`⟨w_j, g⊙u_contrast⟩ ≈ 0` and contribute nothing.

Aggregate to per-task causal importance `C_j(g) = mean over task-g decisions of |φ_j|`, then
score generality as the **participation ratio** `PR_j = (Σ_g C_j(g))² / Σ_g C_j(g)²` =
*effective number of tasks the feature drives* (1 = memorized, G = maximally general).
PR is scale-free, so it measures breadth, not strength.

**Data contract (what A1 must collect).** Per decision (episode, timestep, one of 7 action
slots): the un-pooled L31 residual `h` at the action-token position; the emitted token id;
`r = rms(h)` (or enough to recompute it). Once per model: the unembedding `W_U` [V×d], the
final-norm gain `g` [d] and eps. The SAE (retrained in A2) then supplies `z, w_j, l2, μ`.

**Viability gate (run before building A4).**
- **L0** — do we have action-position residuals + head? *Currently NO (§2.3) → A1 fixes it.*
- **L1** — feed the SAE reconstruction `ĥ` through the real frozen RMSNorm + unembedding;
  does its argmax over the 256 action tokens match the true model's? **Pass ≥ 0.85**; also
  report true-vs-reconstructed action-logit correlation (expect > 0.9).
- **L2** — do the frozen-`r` per-feature `φ_j` sum back to the true logit? Report the
  correlation and mean abs discrepancy; a poor value means RMSNorm nonlinearity is too
  strong for a clean linear decomposition.

**Confound controls on PR (non-negotiable, per commitment #2).** Rank-partial of PR against
(a) total causal magnitude `Σ_g C_j(g)` and (b) base firing rate; plus leave-one-task-out
prediction of held-out causal importance surviving both. `φ` contains `z_j`, so PR is not
perfectly activity-independent — the defensible claim is that *among firing features* causal
influence differs from firing frequency via the alignment term; the controls test it.

**Decision rules.**

| Outcome | Interpretation |
|---|---|
| Gate fails (L1) | SAE features do not linearly explain VLA action selection — a methods finding about SAE attribution in VLAs. Path A closes; Path B stands with its stated scope. |
| Gate passes, PR survives controls | A causal generality measure; combine with Path B for the breadth×recurrence map. Strongest positive. |
| Gate passes, PR collapses under controls | Causal breadth = magnitude/activity. Clean negative, same rigour as §2.2. |

**Cost:** A1 re-collection ≈ original collection (7× vectors: ~7 action positions/timestep;
~18 GB at L31 for 4 models); A2 = 4 SAE retrains (L31 only); A3 cheap. **The honest risk:**
the gate can fail after A1+A2 are paid for.

### 3.2a Path A5 — behavioural (conditional on the gate)

Define per decision `μ_t = Σ_{j not general} |φ_j| / Σ_j |φ_j|` (fraction of the decision
carried by non-general features), then run **closed-loop rollouts on held-out suites**,
logging `μ_t` with success/failure. **Design constraint against reverse causation:** measure
`μ_t` **early in the episode, before any divergence**, and test whether early `μ_t` predicts
eventual failure — otherwise `μ_t` may just be the policy noticing it is already in trouble.
Control for scene/task novelty (a third-variable confound). The **causal** upgrade uses the
existing `ActivationAblator` (`hooks.py`): project out high-`μ` features mid-forward-pass and
test whether held-out success *improves*. Correlation → "reliance on non-transferring
features predicts brittleness"; ablation-improves → "…and removing it helps" (the actual
explanation). Note vs. prior brittleness accounts (data scarcity, covariate shift, shortcut
learning, compositional/ grounding failures): those are behavioural/data-level; this is a
*mechanistic, internal, per-decision* instantiation of the shortcut-learning hypothesis.

### 3.2b Feature identification + visualisation *(NOW — the active step, 2026-08)*

Path A validated the breadth *axis*; it did not name a feature or produce a count (§2.6).
This step makes the axis concrete without reintroducing circular labels.

- **Rank, don't threshold.** Order features by **confound-adjusted breadth** = PR
  residualised on (magnitude, base rate) — the same two-covariate control A4 already uses,
  applied per feature instead of population-wide. Take the top of that ranking as *general
  candidates* and the bottom (high magnitude, low PR) as *specialist candidates*. No hard
  "general/not" cut is claimed; the ranking is the object.
- **Max-|φ| exemplars.** For each candidate, stream the A1 shards, recompute z and φ, and
  record the decisions with the largest |φ_j| (causal exemplars) — and separately the
  largest z_j (activation exemplars). Store (task, episode, timestep, z, φ).
- **Capture frames.** Re-open the LIBERO demos and pull the image frames at those
  (task, episode, timestep) triples; assemble a contact sheet per feature. Eyeball what a
  general feature vs a specialist feature actually responds to.
- **What it answers.** (i) "Show me a general feature" — the reviewer's demand. (ii) The
  §2.6 concession: does structural breadth correspond to a *meaningful* skill, or to an
  input-correlated bookkeeping direction? Exemplar frames are the first evidence either way.
- **Negative control (cheap, same machinery).** Permute task labels and recompute the A4
  partial; a "generic causality is trivial" objection predicts it collapses to ≈0 while the
  real value is +0.493. Run it to convert the verbal rebuttal into a figure.

Files: `identify_features.py` (rank + exemplar-finding, reuses run_attribution's encoder)
and `capture_feature_frames.py` (frame extraction + contact sheets, reuses
`mrvla/libero_demos.py`).

### 3.2c Replication roadmap *(DEFERRED — documented, not implemented)*

Held until §3.2b confirms the axis picks out interpretable structure. Mechanical, not
conceptual:

1. **Other three suites.** Run A1→A4 for spatial, object, libero10 (same recipe, paths
   swapped). Each SAE re-derives its own sufficiency gate — do not reuse goal's pass. Report
   every suite's `partial|both` and fold consistency. One suite is a result; four is a
   finding.
2. **Second SAE seed, all four suites.** Repeat with `--seed 1`. Given §2.5 (SAE
   dictionaries are only ~60% seed-reproducible), a single-seed feature-level result is not
   yet stable; the second seed gives Path A the same seed-noise-floor discipline Path B has
   (§3.1). Report seed agreement on the breadth ranking, not just the aggregate partial.

### 3.3 Axis 2 — Invariance (representational) *(existing data)*

Hold the concept fixed, vary appearance, measure response stability: decompose a
feature's activation variance into concept (task/phase) vs appearance (scene) with the
η²/ICC machinery already in `confound_audit.py` / `residual_consistency.py`. A feature is
invariant if its activation is explained by concept, not appearance. Needs
same-concept/different-appearance groupings (LIBERO trials, cross-suite primitives).
Gives the second axis for the breadth×invariance quadrant map.

### 3.4 Behavioral validation *(highest ceiling; needs Path A)*

Does reliance on **non-general** features (low task-breadth) predict **task failure under
cross-suite shift**? Log the attribution-based reliance per decision alongside
success/failure on held-out suites. This is the only causal test of the thesis and the
strongest possible claim; it depends on the Path-A rebuild + gate passing.

---

## 4. Execution order

```
DONE →  Path B: cross-model recurrence            → POSITIVE (§2.4)
          └─ controls: permutation null, base rate, inheritance, seed noise floor
          └─ unplanned findings: SAE seed-reproducibility ~60%; recurrence vs
             P(general) uncorrelated (§2.5)

DONE →  Path A on GOAL: attribution + sufficiency gate + A4  → POSITIVE (§2.6)
          └─ gate pivot L1→sufficiency (0.936); A4 partial|both = +0.493, 10/10 folds

NOW  →  Feature identification + visualisation (§3.2b)  ← the active step
          └─ rank features by confound-adjusted breadth (residualised PR);
             capture top max-|φ| exemplar frames per feature; eyeball what the
             general vs specialist ends actually respond to. Answers "show me a
             general feature" and tests structural-breadth vs meaningful-skill.
          └─ negative control: permute task labels, recompute A4 partial (expect ≈0)

DEFERRED (documented, NOT yet implemented — §3.2c replication roadmap):
        1. Run Path A (A1→A4) for the other 3 suites: spatial, object, libero10.
           One result → a finding only when it replicates. Same four commands per
           suite, paths swapped. Watch: each SAE re-derives its own sufficiency gate.
        2. Repeat ALL 4 suites with a SECOND SAE seed (--seed 1), so the breadth
           result carries the same seed-noise-floor discipline as Path B (§3.1, §2.5:
           SAE dictionaries are only ~60% seed-reproducible, so a single seed is not
           enough to claim a feature-level result is stable).
        + coalition ablation (§3.2a): broad vs narrow degradation; our-metric vs
          firing-metric ranking as the head-to-head behavioural test.

BACKLOG (cheap, on existing artifacts, no retraining):
        - Hungarian assignment at scale (unmet §3.1 step-3 commitment)
        - second SAE seed for spatial / object / libero10 (also serves Path B §3.1)
        - characterise the top-recurrence features (the §3.1 positive-outcome follow-up)
        - Axis 2: invariance → breadth×invariance quadrant map
```

Path A is now positive on the goal model. Before spending compute on replication
(deferred above), we first make the result *concrete*: identify the features the breadth
axis flags and look at what they do (§3.2b). Replication across suites and the second-seed
repeat are the remaining bulk of the plan, but they are mechanical and are held until the
identification step tells us the axis is picking out interpretable structure.

---

## 5. Analysis commitments (fixed in advance)

1. **Report all layers, all seeds, all four suites** — never a selected subset.
2. **Every score must beat base firing rate.** Report the base-rate-partial correlation
   alongside every raw correlation. A score that does not survive the control is reported
   as null (as in §2.2).
3. **Report the noise floor** (same-model/different-seed) beside every cross-model number.
4. **Distinguish "could not measure" from "not there"** in every null.
5. **No outcome-dependent metric switching.** If a metric changes after seeing results,
   report both.
6. **Keep the firing metrics as descriptors only** — they may characterise features
   (clocks, burstiness) but are never reported as the generality measure.

---

## Appendix — what changed from the original pre-registration

The original plan (git history) was a faithful **replication** of the paper's classifier,
with hand-labelling (its Phase 2) as "the single most valuable experiment." That plan is
superseded because:

- **Hand-labelling is out** — it can only capture human-semantic generality, and we
  redefined generality functionally (§1). It is no longer the ground truth.
- **The firing-metric family is out** as a generality measure — shown to be base rate
  (§2.2). It survives only as description.
- **Generality is now two axes** (breadth × invariance), continuous, not the paper's
  binary.
- **Validation moved to non-firing targets** (recurrence, causal attribution, behaviour),
  each gated by the confound-first control that the original plan named but had not yet
  been forced to apply.

The through-line is unchanged: a measurement-validity contribution plus reusable tooling,
publishable under every outcome.
