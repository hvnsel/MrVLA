# Chapter 4 — Phase 1: building and first-testing the measures

This chapter is the experimental record of the original programme: how the two measurement paths were built, what had to be rebuilt when a measurement turned out to be invalid, and what the first round of results said.

## 4.1 The order things happened in, and why

The programme did not proceed in a straight line, and the detours are part of the record.

| Stage | What it was | Outcome |
|---|---|---|
| Firing metrics | Two label-free metrics from firing statistics | **Null** — they are base firing rate (§2.2) |
| Viability check | Does the existing data support causal analysis? | **No** — wrong vectors were collected (§4.2) |
| A1 | Re-collect the right vectors | Done |
| A2 | Retrain the SAE on them | Done |
| A3 | Gate: is causal decomposition even valid here? | **Pivoted**, then passed (§4.5) |
| A4 | Measure causal breadth, control confounds | **Positive** (§4.8) |
| Path B | Cross-model recurrence | Positive but weak, and reframed (§4.10–4.12) |
| A×B | Do the two axes agree? | **They do not** (§4.13) |
| Behaviour | Ablation and steering | Ablation **null**, steering works (§4.14) |

## 4.2 The data problem that forced a rebuild

The original activations were collected with a hook that did two things which turned out to be fatal for causal analysis:

1. it **mean-pooled** the residual vectors across all the prompt tokens, averaging them into one vector per input; and
2. it captured only the **prefill** pass — the model reading the prompt — not the passes where action tokens are actually produced.

Both are reasonable for asking *"what does the model represent about this input?"*. Both are useless for asking *"what causes this action?"*, because the vectors involved are literally not the ones the model uses to choose the action. An average over prompt positions never enters the action readout at all.

> **Why this is worth recording rather than quietly fixing.** The entire causal programme is only meaningful because it operates on the exact vectors that feed the readout. Discovering that the existing dataset did not contain them meant an expensive re-collection and an SAE retrain before a single causal number could be computed. It also means that the earlier Path B results, computed on the pooled data, describe a genuinely different object from the later action-position ones.

## 4.3 A1 — collecting the right vectors

The re-collection captures, for every decision the policy makes:

| Quantity | Shape | Meaning |
|---|---|---|
| `residual` | 7 x 4096 | layer-31 residual at each of the 7 action-token positions, un-pooled |
| `token_ids` | 7 | which action token was actually emitted at each position |
| `task_id`, `episode`, `timestep` | 1 each | bookkeeping, enough to locate any decision |

and once per model, the constants the readout needs: the 256 action rows of the unembedding `W_U_act`, the final-norm gain `g`, and `eps`.

One easily-missed detail was handled explicitly. OpenVLA's action tokens are **not** simply the last 256 rows of the unembedding matrix. The matrix is padded to a round size, and the model decodes actions using `token_id = vocab_size − bin_index` where `vocab_size` is the *unpadded* size. Selecting the last 256 rows would have picked the wrong tokens and made every attribution silently wrong.

The final dataset for the `goal` suite: **446,096 decisions** (63,728 timesteps x 7 action slots), across 10 tasks.

## 4.4 A2 — retraining the SAE

The SAE must be trained on the *same distribution* it will be used to analyse, so a new dictionary was trained on the action-position residuals: `F = 2048` features, TopK with `k = 100`, 250 epochs, at layer 31.

## 4.5 A3 — the viability gate, and the pivot

Before measuring anything, we must check that a linear decomposition of the action is *valid at all*. If the SAE's features cannot account for the model's action choice, then attributing the action to features is meaningless and the project should stop and report that.

### The first gate, and why it failed

The original gate (**L1**) was: take the SAE's reconstruction of the residual, push it through the real RMSNorm and unembedding, and ask whether it produces the **same argmax** over the 256 action tokens as the true residual. Threshold: 85% agreement.

It stalled at **0.72**. Training 150 more epochs moved it from 0.70 to 0.72. Raising sparsity from `k = 100` to `k = 256` reached only 0.76, on a slope so shallow that clearing 0.85 would have needed `k` in the many hundreds — abandoning sparsity, and with it the entire point of the SAE.

**Diagnosis.** L1 asks the SAE to reconstruct the *entire* 4096-dimensional residual well enough to win an argmax. But that residual carries information for the whole 32,000-token vocabulary, almost all of which is irrelevant to the action. The SAE was being penalised for failing at a task nobody needed it to do, and argmax over 256 tightly-packed bins is a brittle target.

### The pivot: sufficiency

The right question is narrower. We do not care whether the SAE reproduces the residual. We care whether **the features account for the part of the readout that decides the action**.

Recall from §3.2 that only the *contrast* matters — the component that distinguishes bins. Define the **action margin** of a decision as the contrast-projected logit of the token actually emitted:

$$
  true  =  RMSNorm(h) . u_contrast(emitted token)
$$

Now substitute the SAE's decomposition of `h`. Because the readout is a dot product, the margin splits **exactly** into three additive pieces:

$$
  true  =  features  +  bias  +  error

    features = sum over j of  (l2/r) * z_j * <w_j , g (*) u_contrast>
    bias     = ( mu*1 + b_pre ) . ( g (*) u_contrast ) / r
    error    = whatever the SAE failed to reconstruct
$$

where `(*)` is entry-by-entry multiplication. **Sufficiency** asks what fraction of `true` each piece supplies, measured as a straight line through the origin fitted by least squares:

$$
  S(component)  =  sum over decisions of ( true * component )
                   / sum over decisions of ( true * true )
$$

This is "the fraction of the margin this component recovers". A value of 1.0 means the component accounts for the whole margin; 0.5 means half. The three components' slopes sum to exactly 1 by construction, because they sum to `true` itself.

> **Why a slope and not a correlation.** A correlation would say whether the component *tracks* the margin, ignoring size. We need to know whether it *supplies* the margin. A component that is perfectly correlated with the true margin but ten times too small has correlation 1.0 and slope 0.1, and only the slope tells you the truth. The through-origin form also avoids dividing by small numbers, which a per-decision ratio would do.

### Result

On the `goal` suite at `k = 100`, with no retraining:

| Component | Slope | Reading |
|---|---|---|
| features + bias | **0.9361** | 93.6% of the action margin is additively recovered — the gate passes at 0.80 |
| features alone | 0.531 | just over half the margin comes from which features fired |
| constant bias | 0.405 | **a large fixed component**, independent of the input |
| error | 0.064 | what the SAE missed |
| arithmetic canary | 1.0000 | mean absolute discrepancy 2.7e-14 |

The last row is a **bug canary**, not a scientific result: it checks that the frozen-`r` arithmetic reproduces the SAE's own reconstruction exactly. At 2.7e-14 — floating-point dust — the implementation is verified.

Two things to carry forward. First, the gate passed, so attribution is meaningful. Second, and unexpectedly, **about 40% of the action margin is a constant that does not depend on the input at all** — a fixed lean toward a default action. At the time this was noted as a curiosity. Chapter 5 identifies exactly what it is.

> **A commitment honoured.** Changing a metric after seeing a bad result is a classic way to fool yourself. The record states explicitly that L1 is still reported beside sufficiency, that the metric was changed for a stated reason (L1 measures the wrong thing), and that the change was made before reading the sufficiency value.

## 4.6 A4 — the attribution formula, derived

Now we can compute, for one decision, how much each individual feature contributed to the action.

Start from the contrast-projected margin and substitute the SAE decomposition:

$$
  RMSNorm(h) . u_c  =  ( (h/r) (*) g ) . u_c
                    =  (1/r) * ( h (*) g ) . u_c
$$

and with `h = l2 * sum_j z_j w_j  +  mu*1 + b_pre`, the dot product distributes over the sum:

$$
  =  sum over j of  (l2/r) * z_j * ( (w_j (*) g) . u_c )   +   constant term
$$

Each term in that sum is one feature's contribution. So:

$$
  phi_j  =  (l2 / r) * z_j * < w_j , g (*) u_contrast >
$$

| Piece | What it is | Why it is there |
|---|---|---|
| `z_j` | how strongly feature `j` fired | a feature that did not fire contributes nothing |
| `<w_j, g (*) u_contrast>` | **alignment**: how much this feature's direction points at the winning bin rather than the average bin | a feature can fire hard and still contribute nothing if it points sideways |
| `l2 / r` | per-decision scale factors | the SAE normalises each input; `r` is the RMSNorm size. Dropping either makes every `phi` wrong by a per-decision factor |

> **Why this escapes the activity confound by construction.** The old firing metrics could not distinguish a busy feature from an important one. Here a feature can have a huge `z_j` and still have `phi_j ≈ 0`, because its direction is orthogonal to the contrast. **Firing and influencing are now separate quantities.** They are not independent — `phi` contains `z` — so the confound controls in §4.8 remain necessary, but the metric is no longer *definitionally* activity.

A computational note that also guarantees correctness: the alignment term `<w_j, g (*) u_contrast(t)>` does not depend on the decision at all, only on which token `t` was emitted. So it can be computed once for all 2048 features and all 256 tokens, giving a matrix we call the **causal signature** `S`, and every decision just looks up a column. The same matrix is central to Path B (§4.12), which is not a coincidence — it *is* the object that describes what a feature does to actions.

## 4.7 Breadth: from per-decision influence to a per-feature score

Aggregate `phi` into a **per-task causal importance**:

$$
  C_j(g)  =  average over all decisions in task g of  | phi_j |
$$

The absolute value is deliberate: we are asking how much the feature *matters*, not in which direction it pushes. This yields a matrix `C` with one row per task and one column per feature — for `goal`, 10 x 2048.

Breadth is then the participation ratio (§3.5) over the tasks:

$$
  PR_j  =  ( sum over tasks g of C_j(g) )^2  /  ( sum over tasks g of C_j(g)^2 )
$$

**1 = all influence in one task. 10 = influence spread evenly over all ten.**

### Adjusted breadth

Raw PR is entangled with the confounds — a feature that fires everywhere has nonzero `|phi|` everywhere and mechanically scores high. Measured: `PR` correlates **+0.82** with base firing rate. So raw PR is not the claim.

**Adjusted breadth** is PR rank-residualised on both **total causal magnitude** and **base firing rate** (§3.4). It answers: *does this feature spread its influence over more tasks than its strength and its firing rate alone would predict?* This adjusted score is what all downstream feature selection uses.

## 4.8 The A4 result: leave-one-task-out with confound control

A correlation computed on all 10 tasks at once could be driven by a single unusual task, and it would be measuring an in-sample fit rather than a prediction. So the test is **leave-one-task-out** (LOTO):

For each task `g` in turn:
1. Recompute breadth **and** magnitude using only the other 9 tasks.
2. Take the held-out task's causal importance `C(g)` as the target.
3. Compute the rank-partial correlation of training-breadth with held-out-importance, controlling for training-magnitude and base rate.

Recomputing magnitude per fold matters: reusing the all-10-task magnitude would leak the held-out task into the control variable.

**Result on `goal` (446,096 decisions, 10 tasks, 2048 features):**

| Quantity | Value |
|---|---|
| Mean PR | 6.05 tasks (10th percentile 2.65, 90th 9.91) |
| `partial | both controls` | **+0.493** |
| Folds positive | **10 of 10** |
| Worst fold | +0.399 |
| Standard deviation across folds | 0.053 |

**Reading.** Breadth of causal influence predicts a feature's importance on a task it was never measured on, and it does so beyond both confounds, in every fold. It is a real, out-of-sample, firing-independent property.

> **What this result explicitly is not.** It is a claim about an *axis*, not about individual features. At this stage no feature had been identified, and no *count* of general features existed — a count needs a principled cut-point, which was never established. The attribution is also first-order (`r` is held fixed), so it describes the direct readout effect, not the full nonlinear behaviour of the policy.

## 4.9 Making it concrete: identifying features and looking at frames

An axis is abstract. To check it picks out something real, we ranked features by adjusted breadth, took the extremes, found the decisions where each had the largest `|phi|`, and pulled the actual camera frames from those moments.

- **High adjusted breadth**: grasp events, release/open events, return-to-home pose. The same feature fires at "grasp" across many different scenes and objects — reusable manipulation primitives.
- **Low breadth, strong**: single-scene, single-phase detectors — one particular lid grasp in one particular layout. The memorisation end.

A distinction worth keeping: these general features are general *in when they fire* (across scenes) but specific *in what they do* (one primitive). That distinction will matter in Chapter 5.

> A bug was caught here by eye. The "specialist" end was initially selecting features that were weak *and* narrow — features with no causal impact at all, which is a different thing from narrow-but-decisive. Selection now draws both ends from features that are causally eligible in the first place. The frames revealed it; a regression test now pins it.

## 4.10 Path B — does a feature's role reappear in another model?

Path A measures breadth *within* one model. Path B asks a different question: fine-tune the same base model on a different task suite, and does the same causal role show up again?

The procedure:

1. **Shared probe.** Push the *same* set of frames through all four fine-tuned models and record the residuals.
2. **Encode** each model's residuals with *that model's own* SAE, giving an activation matrix per model.
3. **Match.** For feature `i` in model A, find `q_i = max over j of corr(activations of A's feature i, activations of B's feature j)`. A high `q_i` means feature `i` has a twin in B.

Three controls were mandatory, and all three passed:

| Control | What it rules out | Result |
|---|---|---|
| **Permutation null** — shuffle probe-frame order, preserving each feature's marginals *and* the best-of-2048 maximum | that matching is trivially easy | gap `q_cross − q_perm` = **+0.13 to +0.26 in all 20 (layer x suite) cells** |
| **Base rate + inheritance** — regress the recurrence ranking on firing rate and on what the shared base checkpoint contributes | that it is just activity, or just shared ancestry | only **0.1%–11.3%** of the ranking is explained by both together |
| **Discrimination** — does it separate features, or score them all alike? | that "everything recurs" trivially | spread from 0.27 (10th percentile) to 0.69 (90th) |

The last one also answers the obvious objection that all four models share a base checkpoint: if mere network similarity drove the result, every feature would recur equally. They do not.

**Calibration.** Raw match quality has no natural scale, so it is expressed as **chance-corrected retention** against a ceiling:

$$
  ret_cc  =  ( q_cross - q_perm )  /  ( q_seed - q_perm )
$$

where `q_seed` is how well the *same* model's dictionary matches a **re-trained copy of itself with a different random seed**. That is the most any cross-model comparison could achieve. Measured: **0.587 / 0.619 / 0.635 / 0.585 / 0.473** at layers 0 / 8 / 16 / 24 / 31. So changing the entire fine-tuning suite costs only about 40% relative to merely changing the random seed. Recurrence peaks in the middle of the network and is lowest at the output layer.

## 4.11 An unplanned finding: SAE dictionaries are only ~60% seed-reproducible

Computing the ceiling produced a result nobody was looking for. Matching the `goal` model's SAE against a second SAE trained on **exactly the same activations** with only a different random seed gives:

$$
  q_seed  =  0.640 / 0.657 / 0.629 / 0.612 / 0.531     at layers 0 / 8 / 16 / 24 / 31
$$

Not ~1.0. **A model re-analysed with the same data does not perfectly re-find its own features.** This is a standalone methods result: any conclusion drawn from a single SAE fit inherits that instability, and reported feature indices must be read in that light.

It also has a second consequence, exploited in §4.13: a measurement that is only 60% reproducible cannot correlate strongly with anything.

**Second unplanned finding, from the same machinery:** the correlation between recurrence and the reference paper's `P(general)` score, computed on identical features, lies between **−0.17 and +0.17** across all 20 cells. Given §2.2's finding that the paper's score *is* base firing rate, this says a firing-based generality score tells you essentially nothing about whether another model rediscovers a feature.

## 4.12 Redefining recurrence properly — and a null that was wrong

The activation-matching metric above has a defect that emerged on inspection: recurrence plotted against breadth came out **U-shaped**, high at *both* the specialist and general ends, even after controlling for base rate and inheritance.

**Diagnosis.** Correlation-matching over activations rewards features with sharp, low-entropy firing patterns, because those are easy to match. That is a property of a feature's *firing shape*, not of its causal role. The U-shape is a **distinctiveness artefact**.

### The fix: match in a common output space

Instead of matching *when* features fire, match *what they do to actions*. Using the causal signature from §4.6:

$$
  S[j, t]  =  < w_j , g (*) u_t >          contrast-centred across the 256 bins
$$

This is a 256-number description of how feature `j` pushes every possible action bin. Crucially, **the 256 bins mean the same thing in every model** — bin `t` is the same commanded action everywhere — so signatures are comparable across models even though feature indices are not. Match by cosine similarity.

### The null that was wrong, and why

The first null permuted the 256 bin axes. That seemed natural and was badly wrong.

Every model's signatures are produced by pushing directions through an action head, and all four heads are nearly identical. **All signatures therefore live in essentially the same low-dimensional region of the 256-bin space.** Permuting bins rotates one model *out* of that shared region, destroying geometry that is genuinely shared rather than coincidental. The floor collapses and a large fake gap appears — empirically about **0.33 even between unrelated random models**.

**The correct null** destroys only feature *correspondence* while preserving the shared readout geometry: draw **random decoder directions**, push them through the same head, and match those. If real features match no better than random directions pushed through the same head, the gap is correctly zero.

### Result

- **Chance floor = 0.226.** Everything sits above it: shared causal structure across models is real.
- The U-shape is **gone**, confirming it was the activation-matching artefact.
- Recurrence now **declines monotonically with breadth**. Specialists recur slightly *more*; the most general features recur *less*.
- Top-decile (most general) versus the rest: **goal −0.017, spatial −0.015, object −0.008, libero-10 −0.011** — all small, all negative.

## 4.13 The A x B join

Putting the two axes together on the same features:

$$
  corr( adjusted breadth , causal-signature recurrence )  =  -0.127
$$

Uncorrelated, to mildly inverse. The same conclusion emerged from a per-group comparison and a dissociation test. Combined with §4.11's independent firing-based result, **"recurrence is not generality" was established three separate ways**: via the firing metric, via activation recurrence, and via causal-signature recurrence.

At this stage the finding carried an explicit hedge: since SAE dictionaries are only ~60% seed-reproducible, a weak correlation between two noisy measurements is hard to interpret. Chapter 5 turns that hedge into a number.

## 4.14 Behaviour: ablation and steering

Everything so far is measured *inside* the model. The behavioural test asks whether it predicts what the robot does.

**Ablation** removes a feature during a rollout, using a hook that projects the feature's direction out of the residual stream on every forward pass:

$$
  h_ablated  =  h  -  ( h . V^T ) V        V = the unit decoder directions being removed
$$

Five conditions were run on `goal`, 20 episodes per task, every condition replaying the **same** initial states:

| Condition | Which 5 features | Success rate |
|---|---|---|
| baseline | none | 0.760 |
| general | highest adjusted breadth | 0.710 |
| specialist | lowest adjusted breadth, among load-bearing | 0.785 |
| random | 5 arbitrary features | 0.755 |
| firing | top 5 by raw firing rate | **0.020** |

The intended contrast — general versus specialist — produced nothing. The `firing` condition destroyed the policy, but it is not magnitude-matched to the others and plausibly removes a large share of the residual norm rather than action-specific structure.

A pre-registered follow-up ablated six named features **individually**, each carrying a specific prediction about *which task* should break. Every one of them came back null.

**Steering** adds a multiple of a feature's direction instead of removing it. Steering toward a grasp-associated feature made the gripper close **earlier** in the episode — a clear behavioural effect.

> **The state of play at the end of Phase 1.** Two internally-measured positives (Path A breadth, Path B above-chance recurrence), one clear dissociation between them, one methods finding about SAE instability — and a behavioural test that produced nothing interpretable. Whether that behavioural null meant "the features do not matter" or "the experiment was too small to tell" was, at this point, **unknown**. Answering that is where Chapter 5 begins.
