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

## Open threads / next results to land here

- **Coalition ablation** (built, not yet run): top-N general vs specialist
  coalitions, per-task success, 5 conditions (baseline / general / specialist /
  random / firing-matched), multi-GPU sharded. Predicts general-coalition removal
  damages *many* tasks (high damage participation ratio); specialist removal
  damages *few*. Pending GPU allocation.
- **Four-suite Path A partial|both table** — fill in A5 with the per-suite numbers.
- **Signature-entropy control** for B2/explanation #4 — to firm up "specific roles
  recur more" against the sharpness confound.
- **`compare_recurrence_groups --target` bug** — `--target` is label-only; must
  also switch `--rec` to actually change models. Fix before trusting per-suite
  group comparisons.

---

*Metrics reference:* z = SAE code (post-TopK activation); φ = per-feature causal
contribution to the action; C_j(g) = feature j's causal contribution on task g;
PR = participation ratio (effective #tasks); adjusted_breadth = PR residualized on
magnitude + base_rate; q_cross = cross-model max code-correlation; S[j,t] =
256-bin causal signature; floor = random-decoder null.
