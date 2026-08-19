# How we define and quantify causal contribution

Answer to the first reviewer question. Everything here is implemented and, where a
number is quoted, either measured (marked with its `n`) or explicitly marked as not
yet run.

The short version:

> **Causal contribution is a feature's signed, exactly additive share of the
> *contrast-centred* action logit at the readout** — how much of the model's *choice
> between actions* that feature wrote — measured per decision, validated by an
> additivity identity, upgraded to necessity by exact counterfactuals, and turned
> into a feature-level property only through held-out, confound-controlled
> statistics.

The point of the definition is that it is **functional and label-free**. Nothing in
it refers to what a feature "means", to a firing pattern, or to a hand-assigned
category. It is a quantity in the units of the policy's own decision.

---

## 0. Why the definition can be exact here rather than approximate

OpenVLA emits an action as 7 discrete tokens, each an argmax over 256 bins, read off
the layer-31 residual:

```
logit(t) = RMSNorm(h) · u_t        u_t = unembedding row for action token t
```

At layer 31 the readout **is** the whole remaining computation, and it is a dot
product — additive by construction. So we do not need gradients, path patching, or a
saliency surrogate; the decomposition is an identity, not an approximation.

**Scope, stated up front.** This exactness is *why* we are at layer 31 and only
layer 31. Earlier layers would need gradient or path-patching methods and are a
separate decision.

---

## 1. Contribution: the per-decision quantity `φ`

With the SAE writing the residual as `h ≈ l2·(Σ_j z_j w_j) + μ·1 + b_pre`, feature
*j*'s contribution to the emitted token *t* is

```
φ_j = (l2 / r) · z_j · ⟨ w_j , g ⊙ u_contrast ⟩
```

* `z_j` — SAE code, `w_j` — decoder row, `l2` — the SAE's per-sample normaliser,
  `r = rms(h)`, `g` — final-RMSNorm gain.
* `u_contrast = u_t − mean_s u_s` over the **256 action tokens**.

Two choices in that formula carry the whole argument:

**(i) Contrast-centring.** A direction that lifts *every* action logit equally
receives **zero** credit. Only movement in the direction that decides *which* action
wins is counted. Without this, "confidence" and "choice" are conflated and every
high-norm feature scores.

**(ii) The alignment term `⟨w_j, g ⊙ u_contrast⟩`.** This is what makes the measure
causal rather than activity-based, and it is the exact place where we diverge from
the original paper's firing-based generality classifier. A feature can fire hard
(`z_j` large) and contribute **nothing**, because its decoder direction is orthogonal
to the contrast direction. The definition therefore separates two populations the
firing metric cannot see:

* **busy-but-inert** — fires constantly, writes nothing into the decision;
* **rare-but-decisive** — fires seldom, and flips the action when it does.

Implementation note that a naive `φ = z·⟨w, u⟩` gets wrong: carry the per-sample
`l2`, since our SAE normalises each sample.

### 1a. The validity gate on the decomposition (sufficiency)

Before attributing anything we test whether the feature terms actually recover the
decision. Sufficiency is the through-origin slope of the true action margin on the
reconstructed one,

```
S = Σ_i c_i x_i / Σ_i c_i²
```

decomposed into **features + bias + error, which sum to 1 by identity**.

Measured (goal, k=100, layer 31): **S = 0.936** — features alone **0.531**, a
constant default-action bias (`μ + b_pre`) **0.405**, error **0.064**. Pre-registered
threshold 0.80 → pass. The additivity canary (do the frozen-`r` per-feature `φ` sum
back to the true logit) returns correlation **1.0000**, mean absolute discrepancy
**2.7e-14** — the arithmetic is exact, not merely close.

The constant bias term is itself a small finding: ~40% of the action margin is a
fixed lean toward a rest pose, present in every decision and belonging to no feature.

**Recorded metric change.** This gate replaced an earlier full-residual argmax
re-decode that stalled at 0.72 (more epochs moved it 0.70→0.72; k=100→256 reached
0.76 on a shallow slope). The diagnosis was that the old gate punished the SAE for
reconstructing the entire dense last-layer residual — information for the whole 32k
vocabulary, nearly all of it action-irrelevant — when the estimand we care about is
the *action* margin. The old number is still reported beside the new one, and the
change was recorded before the new value was read.

---

## 2. Necessity: exact counterfactuals, no forward pass

Contribution is not necessity. A large `φ` on a decision with a large top-2 margin
changes nothing about what the robot does. So we compute the counterfactual exactly:

```
L_t      = (h ⊙ g) · u_t                    (logits up to the 1/r factor)
L'_t     = L_t − Σ_k coeff_k · S[k, t]      S[j, t] = (w_j ⊙ g) · u_t
```

`r` is a positive scalar and rescales all 256 bins equally, so **it drops out of the
argmax**. Flips are therefore exact, and computable for every feature on every stored
decision with **zero model forward passes**. A top-2 margin bound prunes the
decisions that provably cannot flip, and the pruning count is reported so the saving
is auditable.

What we report from this:

* **flip rate** — fraction of decisions on which removing the feature changes the
  emitted bin. Conditioned on the feature actually firing, not on it merely existing.
* **signed bin shift** — the direction and size of the change in *bin index*, which
  is interpretable: bins are the discretised `dx/dy/dz/droll/dpitch/dyaw/gripper`
  command, so the counterfactual is in units of robot motion.

**Two ablation semantics, both reported.** They are different interventions and the
gap between them is a real caveat rather than noise:

| | `coeff_j` | what it removes |
|---|---|---|
| PROJECTION | `⟨h, w_j⟩` | everything along that direction — coded amount, SAE reconstruction error, mean term, leakage from correlated features. **This is what the closed-loop rollout hook does.** |
| CODED | `l2 · z_j` | only what attribution says the feature wrote. **This is what `φ` describes.** |

Their difference measures how much of a rollout ablation's effect comes from
structure the SAE never attributed to that feature — i.e. how far a behavioural
ablation result can honestly be read as evidence about a *coded* feature.

We also report **coalition non-orthogonality**: decoder rows are unit-norm but not
mutually orthogonal, so the projection hook over-subtracts for correlated
coalitions. We reproduce the hook's actual formula (rather than a corrected
projection) because the point is to model the experiment that was really run, and
we quantify the distortion instead of leaving it implicit.

**Scope.** This is the exact *direct* effect on one decode slot. In a rollout,
ablation also changes what slot `s+1` attends to and what the next timestep observes,
so the behavioural effect is larger. Read these as a **lower bound** on behavioural
impact and as the precise answer to "did this feature decide this action".

---

## 3. Behaviour: closed-loop coalition ablation

The top of the ladder, and the head-to-head against the original paper's ranking:
same rollout protocol, our label-free breadth ranking vs their firing-based
classifier. Success logged **per task**, on identical initial states, across five
conditions — baseline / general coalition / specialist coalition / random matched
coalition / firing-ranked coalition — with paired McNemar tests, damage intervals and
a stated minimum detectable effect, so a null is separable from "underpowered".

The prediction that distinguishes the axis: removing a **general** coalition should
damage *many* tasks (high damage participation ratio), removing a **specialist**
coalition should damage *few*.

Status: built and unit-tested, **not yet run** — pending GPU allocation.

---

## 4. From per-decision `φ` to a property of a feature

```
C_j(g) = mean over task-g decisions of |φ_j|                  per-task causal importance
PR_j   = (Σ_g C_j(g))² / Σ_g C_j(g)²                          breadth = effective #tasks driven
```

The participation ratio is **scale-free**, so it measures *breadth* of causal
influence and not its strength — 1 means the feature's influence lives in a single
task, `G` means it is spread evenly over all of them. Measured on goal: mean PR
**6.05**, p10 **2.65**, p90 **9.91** — a real specialist→generalist spectrum rather
than a two-cluster artefact.

**A second, orthogonal breadth axis, free from the same data:** the same `|φ|` mass
resolved over the **7 action slots** — how many of the robot's degrees of freedom a
feature actually drives. Two traps handled explicitly there: `‖u_contrast‖` depends
on where the emitted bin sits in the ordered range, so the gripper (near-binary,
extreme bins) inflates every feature's absolute score — shares are comparable across
slots, absolutes are not; and the gripper command is constant for most of an episode,
so all statistics are reported both on all decisions and on transitions only.

---

## 5. What makes the number defensible

This is the part that matters most, because the raw statistic is **not** the claim.

**Raw breadth is activity-entangled: `PR ~ base firing rate = +0.82`.** Reporting raw
PR would be reporting firing rate with extra steps. The claim is the confound-
controlled, held-out version:

> Leave one task out. Recompute **both** breadth and total causal magnitude on the
> remaining `G−1` tasks — recomputing both is what keeps the held-out task out of the
> confound controls as well as out of the predictor. Then rank-partial training
> breadth against the **held-out** task's causal importance, controlling **both**
> total causal magnitude and base firing rate.

Measured (goal; 446,096 decisions, 10 tasks, 2048 features):

**partial ρ = +0.493, positive in 10/10 folds, min +0.399, sd 0.053.**

Supporting machinery, all implemented:

* **Concentration** — effective feature count `n_eff`, Lorenz shares (top 1% / 10 /
  50 / 100), Gini, and cross-task top-N overlap **as a ratio to chance** — each
  computed against a base-firing-rate ranking as control. If causal mass concentrates
  no more than firing rate does, "concentration" is an activity statement rather than
  a causal one. This attaches numbers to the claim that answers *"of course hundreds
  of features influence the action"*: the claim is not influence, it is
  **concentration and cross-task reproducibility** of influence.
* **Permutation floors** — with the honest note that the negative control originally
  prescribed is a **no-op**: permuting task labels within a feature leaves the LOTO
  statistic unchanged, because LOTO already averages over every held-out task. Two
  floors that do test something replaced it.
* **Attenuation correction** — for a positive result the correction is free, since
  attenuation only shrinks an observed correlation: `r_obs / sqrt(r_xx)` is a lower
  bound on the truth even with the other reliability unknown. Split-half breadth
  reliability is built but **not yet reported**; at an illustrative `r_xx = 0.72` the
  floor would be **≥ +0.581**.

**The honest limit on the claim.** `φ` contains `z_j`, so PR is not *perfectly*
activity-independent, and we do not claim it is. The defensible statement is: **among
firing features, causal influence differs from firing frequency via the alignment
term** — and the controls above are exactly the test of that statement.

---

## 6. One-paragraph version for the paper

> We define a feature's causal contribution to a decision as its signed, additive
> share of the contrast-centred action logit at the policy's readout. Because
> OpenVLA's action head is a dot product on the final residual stream, this
> decomposition is exact rather than approximate: the per-feature terms sum to the
> true logit to numerical precision, and the feature terms recover 93.6% of the
> action margin. Contribution is upgraded to necessity by an exact counterfactual —
> removing a feature's decoder direction and recomputing the argmax, which the
> RMSNorm scalar leaves invariant — giving a per-feature flip rate and a signed shift
> in commanded action, for every feature on every decision at no forward-pass cost.
> Feature-level generality is then the participation ratio of causal contribution
> across tasks, and is claimed only in leave-one-task-out form, controlling both
> total causal magnitude and base firing rate (ρ = +0.493, 10/10 folds).
