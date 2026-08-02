# Experimental Plan: Are SAE Generality Labels in VLAs Meaningful?

**Status:** pre-registration draft. Written *before* results are collected so that
interpretations cannot be chosen after the fact.

**Reference paper under study:** Swann, McGranahan, Buurmeijer, Kennedy, Schwager,
*"Sparse Autoencoders Reveal Interpretable and Steerable Features in VLA Models"*,
arXiv:2603.19183 (Stanford, 2026). Their metrics and classifier are Sections 3.2–3.3;
their OpenVLA results are Table 2.

---

## 0. Design philosophy

The central question is:

> **At what granularity — if any — is SAE-derived "generality" a real, usable signal in a VLA?**

This is framed as a *measurement-validity* study rather than a method proposal, which is
what makes it robust to outcome. Every phase below has a decision rule and a table
mapping each possible result to the claim it licenses. Both branches of every phase yield
a reportable finding; none of them requires the hypothesis to be true.

The framing to hold throughout: **we are not trying to show anyone is wrong.** We are
testing whether a measurement transfers and what it supports. A negative is a
transfer/validity result, not an accusation.

Three principles:

1. **Falsifiable at every level.** Each phase can kill the phase below it, and saying so
   is a result.
2. **Confound-first.** No score is used for anything until it has been decomposed against
   nuisance variables (identity, length, activity, phase).
3. **Agreement over reliability.** A quantity being *precisely measured* is not evidence
   it is *real*. Validity comes from agreement across independent views (layers, seeds,
   models, behaviour).

---

## Phase 0 — Faithful reimplementation and validation gate

**Purpose:** establish that our implementation of the paper's metrics is correct before
any conclusion rests on it. This phase exists because an earlier iteration of this project
produced a dramatic "the classifier inverts" result that was traced to *our own*
implementation diverging from the paper in three places.

### 0.1 The three corrections (already applied in `mrvla/generality_classifier.py`)

| Quantity | Paper (Eq.) | Earlier incorrect implementation |
|---|---|---|
| Mean activation magnitude `ā` | mean over active episodes of the **per-episode peak** (Eq. 8) | mean of `z` over ON timesteps |
| Onset state machine | ON when `f > τ_on`; OFF **only when `f == 0`**; else hold (Eq. 5) | single threshold, or a τ_off = 0.05 dead zone |
| Active-episode set `E⁺` | episodes with **any `f > 0`** (p. 5) | episodes where the thresholded state fired |
| Reporting denominator | **active** features (Table 2: 1775 at L8) | full dictionary width |

### 0.2 Validation runs

Run the corrected classifier on the **existing** `codes_v3` immediately — this needs no
new data and is the fastest possible check.

```
python mrvla/generality_classifier.py --codes-dir <codes> --out-dir <gen> --dataset libero
```

**Measurements:** % general over active features per layer; the coverage / onset /
peak-magnitude profile of the top-P features; count of features with nonzero-but-
sub-threshold episodes (tests the paper's `ō ≥ 1` assertion).

**Reference (paper Table 2, OpenVLA / LIBERO-Goal):** Layer 8 → **8 general / 1775 active
(0.45%)**; LM average over layers 0,8,16,24,31 → 42 / 9389 (99.55% memorized).

**Qualitative reference (paper §4.3.1):** genuine general features have **episode coverage
> 0.99**, are **bursty** (`ō` ≈ 2–4, scaling with the number of pick-and-place sub-goals),
and have **low relative run length**. Reported probabilities: F1129 = 0.91, F1902 = 0.89,
F128 = 0.92, F445 = 0.58.

### 0.3 Decision rule

| Outcome | Interpretation | Action |
|---|---|---|
| **0-A.** ~0.2–1% general; top features are high-coverage and bursty | Implementation validated; classifier transfers to our SAE | Proceed to Phase 1 |
| **0-B.** Prevalence in range, but top features are low-coverage / flickery | Classifier reproduces the *count* but not the *identity* → genuine transfer failure, and a real finding | Proceed, but this becomes a headline result; verify it is not an SAE-quality artifact in Phase 4 |
| **0-C.** 0% general, or ≫2% | Our SAE differs materially from theirs (width, sparsity, activation scale) | **Do not proceed.** Reconcile SAE config first (§1.2) — most likely the dictionary-width mismatch |

> **Note.** Our current SAE is `F = 2048`; the paper uses a 1× expansion ratio, so for
> OpenVLA (`d = 4096`) their dictionary is `F = 4096` with `k = 100` and ~43% of features
> alive. This is a known, material difference and is the first thing to check under 0-C.

---

## Phase 1 — Data and SAE training

### 1.1 Activation collection *(in progress)*
LIBERO **Goal, Spatial, Object, 10**, at layers 0, 8, 16, 24, 31 of OpenVLA.
Collecting across all four suites is what makes Phases 3–5 possible (cross-suite transfer
needs held-out suites; cross-model universality needs per-suite fine-tunes).

### 1.2 SAE configuration — **decision required**
Match the paper (`F = d = 4096`, `k = 100`, 1× expansion, per-sample normalisation,
AuxK loss, unit-norm decoder) **or** deliberately differ and treat it as a variable.
Matching makes Table 2 a direct reference point; differing makes every comparison
approximate. Recommendation: **match**, and if desired add our 2048 config as a
secondary arm to measure width sensitivity.

### 1.3 Seeds
**≥ 2 SAE seeds per (model, layer)** — mandatory, not optional. This is the noise floor
for every subsequent claim, and without it no negative result is interpretable.

### 1.4 Models
The base OpenVLA fine-tune (LIBERO-Goal) plus additional per-suite fine-tunes for
Phase 4. Note these share a base checkpoint, so they test *robustness*, not statistical
independence — state this explicitly in the writeup.

---

## Phase 2 — Feature-level generality

The feature level is where generality is *defined*; if it fails here, nothing downstream
can succeed.

### 2.1 Apply the paper's LIBERO classifier
Their β, unnormalised metrics, per §3.3.3 (they apply the LIBERO boundary to OpenVLA
because OpenVLA is also LIBERO fine-tuned).

### 2.2 Refit our own classifier *(validation arm)*
Hand-label **30–60 features** from our SAE (balanced general/memorized) following the
paper's three-stage protocol (episode-level screening → cross-episode validation →
labelling criteria, §3.3.2). Fit logistic regression on our four metrics; report LOO-CV
accuracy as they do (they achieved 100% on 30 labels).

**This is the single most valuable experiment in the plan.** It removes all dependence on
borrowed coefficients and converts "does their classifier transfer?" into "what does a
correctly-fit classifier on *our* model look like?"

### 2.3 Outcomes

| Outcome | Interpretation | Paper contribution |
|---|---|---|
| **2-A.** Our fitted β ≈ theirs; classifications agree | The classifier is stable across SAE fits and models; the generality construct is well-posed at the feature level | Positive replication; solid foundation for Phases 3–5 |
| **2-B.** β differs materially, but both boundaries select similar feature sets | Coefficients are not identifiable from 30 labels, but the decision surface is robust | Methods finding: report coefficient instability, recommend reporting selected-set profiles rather than β |
| **2-C.** β differs *and* the selected sets differ | The classifier does **not** transfer across SAE fits/models — borrowed coefficients are unsafe | Strong methods result: cross-fit non-transfer of interpretability classifiers |
| **2-D.** Hand-labelling is not reliably possible (annotators cannot separate general from memorized) | The construct itself is ill-posed at this SAE's resolution | Most fundamental result available; reframes the whole literature's premise. Requires ≥2 annotators + agreement statistic to claim |

**Required control:** inter-annotator agreement (Cohen's κ) on the hand labels. Without it,
2-D is not claimable and 2-A rests on one person's judgement.

---

## Phase 3 — The granularity descent

The organising question of the paper. Generality is defined per-feature; we test whether
it survives aggregation upward.

### 3.1 Episode level
Score each episode by general-feature content; audit with `mrvla/confound_audit.py`.

- **Score constructions:** mass-weighted `Σ z·p / Σ z` and mass-robust count
  `(1/K) Σ_{j∈A_t} p_j`.
- **Confounds:** episode length, task identity (η²), mean activation mass, mean L0,
  idle-frame fraction.
- **Validity test:** cross-layer consistency of the confound-free residual
  (`mrvla/residual_consistency.py`).

### 3.2 Frame level
The refinement the earlier iteration never tested, and the natural response to an
episode-level null.

- **Score:** per-frame count score `r_t`.
- **Primary confound: trajectory phase** φ = t/T ∈ [0,1]. If "general" frames are simply
  the home/transport frames, upweighting them is a phase curriculum that downweights the
  grasp — an intervention likely to *hurt*.
- **Variance decomposition:** Var(r) = Var_between-episode + Var_phase|episode + Var_residual.
- **Validity test:** cross-layer consistency of the phase-residual (same estimator as 3.1).

### 3.3 Outcomes

| Episode | Frame | Interpretation | Paper contribution |
|---|---|---|---|
| survives | — | Episode-level generality is real and confound-free | Constructive: proceed to weighted fine-tuning (Phase 6) |
| dissolves | survives | Generality is a *moment-level* property that averaging destroys | Strong constructive result + a clean explanation of why episode-level fails |
| dissolves | dissolves into **phase** | The frame score is a stopwatch; "generality" tracks trajectory position | Clean negative with a named mechanism; strong cautionary result |
| dissolves | dissolves into **task identity** | Both levels measure task mix, not generality | Clean negative; recommend feature/decision-level intervention only |

Note that a null here is only interpretable **given Phase 2-A** (labels are sound). If
Phase 2 gives 2-C or 2-D, a Phase 3 null is attributable to the labels rather than to
aggregation, and must be reported that way.

---

## Phase 4 — Replication and cross-model universality

### 4.1 Seed replication
Rerun Phases 2–3 across SAE seeds on the same model.

| Outcome | Interpretation |
|---|---|
| Conclusions stable across seeds | Findings are properties of the *metrics*, not of one dictionary — the strong version of every claim |
| Conclusions vary across seeds | **SAE non-identifiability dominates generality analysis.** This becomes the headline: single-fit conclusions (including prior work's, and our own) are not reliable at this scale |

### 4.2 Cross-model feature universality
Independent external criterion: do general features recur across models?

- **Matching:** identical probe frames through both SAEs; cross-correlate activations;
  `q_i = max_j corr(Z^A_i, Z^B_j)`; report greedy and Hungarian assignment.
- **Two confounds:** SAE noise floor (pulls similarity down) and shared-base inheritance
  (pulls it up). Both cancel in the differential:

  ```
  Δ* = [mean q over general − mean q over memorized]_cross-model
     − [mean q over general − mean q over memorized]_same-model-different-seed
  ```

| Outcome | Interpretation | Paper contribution |
|---|---|---|
| **Δ\* > 0** | Coverage-generality predicts cross-model recurrence — the label captures something model-independent | External validation; the paper's positive result |
| **Δ\* ≈ 0**, noise floor resolvable | Generality labels are model-local | Coherent negative: fails every external check |
| Noise floor unresolvable | SAE non-identifiability prevents cross-model comparison at this scale | Methods result **only** — must not be reported as evidence against generality |

---

## Phase 5 — Behavioural validation

The only phase that tests the hypothesis *causally*, and the one that raises the paper's
ceiling most. Memorization and fragility are decoupled in-distribution and couple only
under distribution shift — and LIBERO's four suites supply real shift, so no world model
or synthetic perturbation is needed.

### 5.1 Attribution probe
OpenVLA's action is decoded linearly from the residual stream, which the SAE decomposes as
`h_t ≈ Σ_j z_tj w_j^dec`. The contribution of feature *j* to the emitted action token
`d*` with unembedding direction `u_{d*}`:

```
φ_tj = z_tj · ⟨ w_j^dec , u_{d*} ⟩
μ_t  = Σ_{j ∉ general} |φ_tj|  /  Σ_j |φ_tj|
```

`μ_t` is the fraction of *decision-relevant* attribution carried by non-general features.
This distinguishes a memorized feature that merely **fires** from one that actually
**drives the action** — a distinction no aggregate score can make, and plausibly part of
why the aggregate scores failed.

**Gate:** if the SAE decomposition does not linearly explain the action (check
reconstruction of the action logits from `φ`), Phase 5 is not viable. Test this first;
it is cheap and it is a go/no-go.

### 5.2 Cross-suite transfer evaluation
Evaluate the LIBERO-Goal policy on held-out suites (Spatial / Object / 10), logging `μ_t`
per decision alongside success/failure.

### 5.3 Outcomes

| Outcome | Interpretation | Paper contribution |
|---|---|---|
| `μ_t` predicts transfer failure | **Memorized-feature reliance causes fragility under shift** — the project's core thesis, externally validated | Main-conference-tier positive result; motivates decision-level intervention |
| `μ_t` does not predict failure | Memorization is not the axis governing VLA transfer | Substantive negative: constrains a widely-assumed mechanism |
| Attribution probe fails the gate | Actions are not linearly attributable to SAE features | Methods finding about SAE-based attribution in VLAs; Phase 5 closes |

---

## Phase 6 — Intervention *(conditional)*

Run **only** if Phase 3 yields a validated confound-free score at some granularity, or
Phase 5 validates `μ_t`.

Weighted LoRA fine-tuning with the full arm set already implemented in
`mrvla/episode_weights.py` (uniform / mild / medium / sharp × real / inverted / random),
evaluated cross-suite. The control arms are load-bearing: **inverted** tests direction,
**random** tests whether any reweighting with the same weight distribution helps. Note
explicitly that these controls establish *whether there is signal*, not *whether the
signal is generality* — only the Phase 3 audit can do the latter.

---

## Why this yields a publishable result under every outcome

| Scenario | The paper |
|---|---|
| Everything validates (2-A, 3 survives, 4 Δ\*>0, 5 positive) | *"A validated generality signal for VLAs, with behavioural and cross-model grounding, and a training method that exploits it."* |
| Labels sound, aggregation fails (2-A, 3 dissolves) | *"Generality is a feature-level property that does not survive aggregation — here is the mechanism, the audit tooling, and the level at which intervention must occur."* |
| Classifier does not transfer (2-C) | *"Interpretability classifiers with borrowed coefficients do not transfer across SAE fits or models — diagnosis and recommended practice."* |
| Seed-unstable (4.1 varies) | *"SAE non-identifiability dominates feature-level generality analysis; single-fit conclusions are unreliable."* |
| Construct ill-posed (2-D) | *"The general/memorized dichotomy is not reliably annotatable at this resolution."* |
| Behavioural link holds (5 positive) regardless of 3 | *"Memorized-feature reliance predicts VLA transfer failure"* — strongest single claim available |

The common thread in every row: **a measurement-validity contribution plus reusable
tooling.** That is what makes the outcome irrelevant to publishability.

---

## Execution order and dependencies

```
Phase 0  (now, on existing codes)          ── gate ──┐
Phase 1  (activation collection + SAEs)              │
   ├─ 1.2 config decision  ← blocks SAE training     │
   └─ 1.3 seeds (≥2)                                 │
Phase 2  (feature level)  ← needs 1                  │
   ├─ 2.1 apply paper β                              │
   └─ 2.2 hand-label + refit  ← the key experiment   │
Phase 3  (episode → frame)  ← needs 2                │
Phase 4  (seeds, cross-model)  ← needs 1.3, 1.4      │  ─ runs parallel to 3
Phase 5  (attribution + cross-suite)  ← needs 1      │  ─ runs parallel to 3, 4
Phase 6  (intervention)  ← conditional on 3 or 5
```

Phases 3, 4 and 5 are mutually independent and should be run in parallel. Phase 5's
attribution gate (§5.1) should be tested **early** because it is cheap and it determines
whether the highest-ceiling result is available at all.

---

## Analysis commitments (fixed in advance)

1. **Report all layers and all seeds**, not a selected subset.
2. **Pre-specified thresholds:** confound-dominated if OLS R² ≥ 0.8; residual accepted as
   a stable trait only if even/odd r_SB ≥ 0.5 **and** first/second ICC > 0 (the second
   condition guards the forced-anticorrelation artifact — when the confound model fits
   well, half-residuals are algebraically forced toward r = −1).
3. **Report the noise floor** alongside every cross-fit or cross-model comparison.
4. **Report inter-annotator agreement** for any hand-labelling.
5. **No outcome-dependent metric switching.** If a metric is changed after seeing results,
   report both.
6. **Distinguish "we could not measure it" from "it is not there"** in every null.
