# Chapter 7 — What we claim

A result is not a claim. This chapter states exactly what the numbers in Chapter 6 do and do not support.

## 7.1 The central claim

> **Causal generality in a VLA's sparse-autoencoder features is real, concentrated, and replicable *within* a model — but it is a property of the task distribution, not a universal one. It does not predict whether a differently fine-tuned copy of the same base model contains the same feature, and the standard tools for measuring and testing it systematically misattribute it.**

## 7.2 Claims we make, with their support

### C1. Firing statistics do not measure generality

The existing operational definition fails twice: the classifier's labels are not independent of its inputs (criterion contamination), and its central statistic is algebraically base firing rate under balanced task groups. A related metric, once base rate is controlled, predicts held-out firing **negatively** in all 20 cells tested — it is a memorisation signal.

*Supported by §2.2, §6.1.*

### C2. Causal task-breadth is a valid, label-free generality measure

Breadth predicts a feature's causal importance on a task it was never measured on, beyond both causal magnitude and base firing rate, in **all four LIBERO suites**: +0.387 to +0.535, positive in 10/10 folds each, against a permutation floor of zero at 40 to 75 standard deviations. Attenuation correction puts the true relationship at `|r| >= 0.82`.

*Supported by §4.8, §5.2, §6.2.*

### C3. Causal influence is sharply concentrated, and concentrated on the same features across tasks

An effective **102 of 2048** features carry the action in `goal` (106 spatial, 49 object, 9.6 libero-10). The top 50 features hold 44% to 73% of all causal mass. The **same top-50 recurs across tasks at 34 to 37 times chance**. Causal mass is more concentrated than firing rate in every suite, and column-shuffling destroys the effect — so this is about the same features being large across tasks, not about the shape of the distribution.

*Supported by §5.3, §6.2.*

### C4. Causal generality is not localised to an action channel

The prediction that general features are gripper/phase controllers is **falsified**: the partial correlation with gripper concentration is +0.069, the lowest of the seven channels. There is, however, a second and largely independent breadth axis — features drive an effective 3.21 of 7 action dimensions, correlated with task breadth at only +0.238.

*Supported by §5.9, §6.5.*

### C5. The gripper is a learned default, not a feature-driven decision

At a **frozen state**, the gripper's action margin is recovered almost entirely by the constant `mu + b_pre` bias (sufficiency 0.992) while the features contribute nothing (−0.046), where the other six channels sit at 0.59 to 0.92. Necessity agrees and is not tautological: given a feature fires, removing it changes a `dx` token 0.0534 of the time and a gripper token 0.0008 — about 65 times less. This identifies what the previously-unexplained "constant default-action bias of 0.405" is.

*Supported by §5.9, §6.5. **Scope**: this is the direct effect at a frozen state on one decode slot. It says nothing about trajectory effects, and is fully consistent with steering a grasp feature making the gripper close earlier, since features control the arm and the gripper follows the state.*

### C6. Cross-model recurrence is real but weak, and general features recur least

Shared causal structure across independently fine-tuned models is above chance everywhere. But chance-corrected retention is only about **0.25** — changing the fine-tuning suite costs roughly three quarters of what changing an SAE random seed costs — and the most causally general features are, if anything, the least recurrent.

*Supported by §4.12, §5.11, §5.12, §6.3, §6.6.*

### C7. That dissociation is not an artefact of one-to-one matching

Dictionary splitting was the strongest innocent explanation and it fails. Letting one model express another's role as a coalition of up to eight features improves matching **no more than a random dictionary** in any suite — every breadth decile rises with `m` more slowly than the random floor does, and chance-corrected retention is flat from `m = 1` to `m = 8`. The dissociation reproduces at the level of clustered role inventories (`corr(role breadth, match)` = −0.54 to −0.81), where measurement noise is far smaller.

*Supported by §5.11, §5.12, §6.6.*

### C8. Three measurement results bound what this literature can currently claim

- Breadth rankings are only **0.27 to 0.47** reliable across disjoint task halves. The population axis is solid; individual feature rankings are weakly reproducible.
- **71.5%** of a standard projection-based ablation's effect lands on decisions where the feature never fired.
- At conventional episode budgets a coalition-ablation experiment cannot resolve damage below about **10 points**, so published null results in this design are uninterpretable without a power bound.

*Supported by §5.4, §5.5, §5.10, §6.2, §6.4, §6.5.*

## 7.3 Claims we explicitly do not make

Stating these is part of the result.

| We do **not** claim | Why not |
|---|---|
| That general features are behaviourally load-bearing | The ablation is a **bounded null**: damage below 11.9 points against a design that resolves 9.7. It bounds the effect; it does not demonstrate one |
| That specialist features are inert | Same bound, same reason |
| That we know how many general features there are | A count needs a principled cut-point; the distribution was never tested for bimodality |
| That "independently trained models fail to converge" | The four models share a base checkpoint. They are divergent branches from a common ancestor, not independent draws. The finding is about **divergence under fine-tuning**, not failed universality |
| That the surviving ~25% recurrence is *rediscovered* | It might be inherited from the shared base. The inheritance control for causal signatures needs a base-model SAE, which does not exist |
| That the feature-level A x B null is established | At reliability 0.363 it is not defensible without a recurrence-reliability estimate (§5.6). The **role-level** version (C7) does not depend on that |
| That breadth predicts per-decision decisiveness | It does not (general ≈ random given firing) — but `adjusted_breadth` is residualised on magnitude by construction, so it never claimed to |
| Anything about other model families, other layers, or real robots | One model family, one layer, simulation only |

## 7.4 Draft abstract

> Sparse autoencoders are increasingly used to identify "general" features in vision-language-action models, but generality is operationalised through firing statistics and human labels. We show this definition is circular — the labels are screened by the same metrics the classifier regresses onto — and that under balanced task groups its central statistic reduces to base firing rate.
>
> We redefine generality functionally and label-free as *causal influence that recurs*, separating two axes the standard definition conflates: breadth of causal influence across tasks within one model, and recurrence of causal signatures across separately fine-tuned models. Because OpenVLA's action readout is a dot product at the final layer, per-feature contributions to the emitted action decompose exactly, giving both first-order attribution and exact counterfactuals over 446,096 decisions without a forward pass.
>
> Causal task-breadth is real and replicates: it predicts held-out causal importance in all four LIBERO suites (partial rho = +0.39 to +0.53, 10/10 folds) against a permutation floor of zero. Influence is sharply concentrated — an effective 10 to 106 features of 2048 carry the action, with the same top-50 recurring across tasks at 34 to 37 times chance, and more concentrated than firing rate in every suite.
>
> The second axis does not follow from the first. Fine-tuning a single base checkpoint on four different task suites yields dictionaries whose causal roles align only about 25% as well as two SAE seeds on the same model do, and the features that are most causally general within a model are the ones that align worst across them. This is not an artefact of dictionary splitting: allowing a model to express another's role as a coalition of up to eight features improves matching no more than a random dictionary does in any suite, and the dissociation reproduces at the level of clustered role inventories. Generality is therefore task-distribution-relative, not universal.
>
> We also report three measurement results that bound what this literature can currently claim: breadth rankings are only 0.27 to 0.47 reliable across disjoint task halves; the standard projection-based ablation acts on decisions where the feature never fires 72% of the time; and at conventional episode budgets a coalition-ablation experiment cannot resolve damage below about ten points, making published null results uninterpretable without a power bound.

## 7.5 Why the failed experiments matter as much as the successful ones

Four experiments in this programme returned nothing, and three of those are load-bearing.

| Experiment | Outcome | Why it counts |
|---|---|---|
| Firing-based generality metrics | Null | Closed an entire methodological route and forced the causal redefinition. Without it, the project would have measured base rate and called it generality |
| B1 — generality is channel-localised | Falsified | A sharp prediction, tested and refuted. It also produced C5 and the second breadth axis as by-products |
| A4 — splitting explains the Path B null | Failed to rescue | The strongest available alternative explanation, run to completion. Its failure is what upgrades C6 from a hedged observation into a defended finding |
| Coalition ablation | Bounded null | Only interpretable *because* the power bound was computed. Without it, the same data is unreportable |

A programme that only reported its positive results would have published the firing metrics, would not have found the gripper result, and would have left the central dissociation permanently vulnerable to a one-line objection.
