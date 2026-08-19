# Chapter 2 — The scientific question

## 2.1 The intuition: general versus memorised

Suppose we look at feature 1167 in the `goal` model and find that it switches on whenever the robot is about to close its gripper — no matter which object, which layout, which task. That looks like a **general** feature: a reusable piece of machinery, a "grasp" primitive.

Now suppose feature 1140 switches on only during one specific task, when the arm is above one specific drawer in one specific scene. That looks **memorised**: a lookup-table entry for one situation.

The distinction matters for a practical reason. A policy built mostly out of general primitives should transfer to new situations. A policy built mostly out of memorised entries should be brittle — it will do well on what it was trained on and fail as soon as anything changes. Being able to *measure* which of the two a policy is doing would tell us something real about robot learning.

So: how do you measure it?

## 2.2 The existing answer, and why it does not work

The reference paper defines a general feature as one that **fires across many episodes for a human-nameable event**. In practice this is implemented as a classifier: compute a handful of statistics about *when* each feature fires, hand-label 30 features as general or memorised, fit a classifier from statistics to labels, and report its accuracy.

The statistics are things like:

- **coverage** — the fraction of episodes in which the feature fires at all;
- **mean onsets** — how many separate times it switches on within an episode;
- **mean magnitude** — how strongly it fires;
- **relative run length** — how long it stays on.

The reported accuracy is 100% under leave-one-out cross-validation. That sounds decisive. It is not, for two separate reasons.

### Problem 1: the labels and the features are not independent

To be a valid test, a classifier's **labels** must be obtained independently of its **inputs**. Here they are not. The labelling procedure screens candidates using burstiness (one of the metrics), requires the global metrics to agree before a label is assigned, and excludes ambiguous cases. So the labels are partly *defined by* the same metrics the classifier is given as input.

This is a known failure mode with a name: **criterion contamination**. Combined with **range restriction** (throwing out the ambiguous middle makes any classification easier), 100% accuracy is close to guaranteed by construction. It measures label-metric consistency, not whether the construct "general" is real.

> This is not an accusation of misconduct. The paper documents its own failures honestly — two features it discusses at length are mis-classified by its own classifier. The point is a methodological one: the reported accuracy cannot be evidence for the construct.

### Problem 2: the statistics reduce to "how often does it fire?"

This one we established ourselves, and it is the more damaging of the two.

We built two label-free firing-based metrics of our own and validated them properly, using **leave-one-group-out**: score a feature using 9 of the 10 tasks, then check whether that score predicts how the feature behaves on the held-out 10th task. Raw results looked encouraging — correlations of 0.3 to 0.8.

Then we applied the **confound control**. A confound is a third variable that could explain the result without the interesting mechanism being true. The obvious candidate here is a feature's overall **base firing rate**: how often it fires at all, across everything. A feature that fires a lot will *mechanically* appear in more episodes, more tasks, and more groups.

We therefore asked: does the score still predict held-out behaviour *once base firing rate is held fixed*? (The technique for holding a variable fixed is **partial correlation**, explained in §3.4.)

The answer was no, twice over:

- Under LIBERO's balanced task groups, one of the metrics is **algebraically identical** to base firing rate. Nothing remains after controlling for it. And by the same argument, **so is the paper's `coverage`**.
- The other metric, after controlling for base rate, correlated 0.94 with *concentration* and predicted held-out firing **negatively** in all 20 layer-by-suite cells tested (−0.05 to −0.27). Controlled for activity, it is a **memorisation** signal, not a generality one.

> **The lesson that reshaped the project.** A validation target of *firing* can never isolate generality, because firing **is** activity. Two equally busy features — one a genuine reusable primitive, one meaningless noise — receive the same score. The entire firing-statistics route is closed.

## 2.3 Our redefinition

If we cannot use firing, and we do not want human labels, what is left? **Causal influence.**

> **General** = a feature whose **causal influence on the policy's chosen action recurs across many tasks**.
> **Memorised** = a feature whose influence is confined to one task or situation.
> Generality is a **continuous spectrum**, not a binary label.

Two things this deliberately does *not* require:

- **No human interpretability.** A feature may be general without any person being able to name what it responds to. Whether general features also happen to be interpretable becomes an empirical question we report, not an assumption we build in.
- **No labels at all.** Everything is computed from the model's own behaviour.

### Splitting one word into two axes

The old definition welded together two properties that are actually independent. Making them separate is the core conceptual move.

| Axis | Question it asks | Name we use |
|---|---|---|
| **Task-breadth** (causal) | Does this feature *drive the action* across many different tasks? | **Path A** |
| **Cross-model recurrence** | Does this feature's causal role *reappear* in a separately fine-tuned model? | **Path B** |

The example that forced the split is a "lid detector" discussed in the reference paper. It fires on every kind of lid, in every scene — so by any invariance-flavoured definition it is highly general. But there are only a couple of lid tasks, so its influence on the policy's behaviour is narrow. **High invariance, low breadth.** One number was averaging two orthogonal things.

A grasp detector scores high on both. A memorised feature scores low on both. A diffuse control signal with no clean concept scores high on breadth and low on invariance. Four distinct quadrants, previously collapsed into one score.

## 2.4 The three methodological commitments

Fixed in advance, and the reason the project kept finding things:

1. **Falsifiable at every level.** A metric that fails a validity check is itself a result, publishable as such. This is why "the firing route is closed" is written up rather than buried.
2. **Confound-first.** No score is trusted until it is shown to predict something *beyond* base firing rate. This principle already killed one whole family of metrics (§2.2) and, as Chapter 5 shows, it caught a second confound years later in a completely different measurement.
3. **Agreement over reliability.** Precision is not validity. A number can be extremely repeatable and still measure the wrong thing. Confidence comes from *independent* measurements agreeing — different layers, different seeds, different models, different behaviours.

## 2.5 What a "control" is, and why this document is full of them

Most of the technical machinery in Chapters 3–5 exists to serve one purpose, so it is worth stating plainly.

Suppose you measure something and get an exciting number — say a correlation of +0.49. Before believing it means what you hope, you must rule out the boring explanations:

- **Is it just arithmetic?** Would you get +0.49 from *any* data of this shape, regardless of what the model does? → answered by a **null distribution** (§3.5): scramble the data in a way that destroys the effect but preserves everything else, recompute, and see what you get.
- **Is it just a known confound?** Would base firing rate alone produce it? → answered by a **partial correlation** (§3.4).
- **Is it just noise?** With this much data, how big a number would random chance produce? → answered by **confidence intervals** and **statistical power** (§3.6–3.8).
- **Could the experiment have detected the effect if it were there?** A null result from an experiment too small to see anything is not evidence of absence. → answered by the **minimum detectable effect** (§3.8).
- **Is the measurement reliable enough to support the claim?** Two very noisy measurements cannot correlate strongly even if the underlying truth is a perfect relationship. → answered by **reliability** and **attenuation** (§3.9).

Every one of those five appears in the results. Chapter 3 builds each tool; Chapters 4 and 5 apply them.
