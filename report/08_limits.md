# Chapter 8 — Limitations and known issues

Everything currently wrong, unresolved, or unverified. Nothing in this chapter has been fixed; it is recorded so that no reader is surprised by it later.

## 8.1 Scope limits

| Limit | Consequence |
|---|---|
| One model family (OpenVLA-7B) | Nothing here establishes that the findings hold for diffusion or flow-matching policies. The attribution method itself depends on the readout being a dot product, which flow policies do not have |
| One layer (31, the last) | The exact decomposition is only available where the residual feeds the readout directly. Earlier layers need gradient or path-patching methods, a separate decision that was never taken |
| Simulation only | LIBERO is simulated. No claim transfers to a physical robot without testing |
| Four models sharing one base checkpoint | They are siblings, not independent draws. See §7.3 |
| 10 tasks per suite | Breadth is a participation ratio over 10 items, which is a coarse statistic. This is the direct cause of the low reliability in §8.2 |

## 8.2 The reliability ceiling

Breadth reliability is 0.27 to 0.47. This is the most pervasive limitation in the programme, because it bounds every claim about *individual* features:

- The features selected for ablation and steering were chosen from a ranking that reproduces only weakly across disjoint task halves. Part of any chosen coalition is noise, which dilutes any real effect the ablation might have found.
- The feature-level A x B correlation of −0.127 cannot be interpreted as a null without a recurrence reliability estimate (§5.6).
- Population-level claims (C2, C3) are unaffected: aggregating over 2048 features turns a weak per-feature signal into a decisive population result.

**Recurrence reliability has never been measured.** The cheapest estimate would recompute the cross-model match on disjoint halves of the probe frames and correlate the two rankings — no rollouts, no retraining.

## 8.3 The rank-tie defect (open, and it touches feature selection)

Rank correlations throughout the older pipeline compute ranks by sorting twice, which breaks ties by **array index** rather than averaging them (§3.3). On continuous data the two agree, which is why it went unnoticed.

It does not agree here. `base_rate` is a count over a fixed denominator, so rarely-firing features tie with each other. Measured on `goal`: **10.3% of feature values are tied.** Each tied block therefore receives an arbitrary ordering determined by feature index, inside `adjusted_breadth` — the ranking that selects every ablation and steering target.

A correct tie-averaging implementation exists and all Phase 2 code uses it. **The older pipeline is deliberately unchanged**, because fixing it moves published numbers, and that should be a deliberate decision rather than a side effect of an audit.

## 8.4 Issues found in the audit of Phase 2's own code

A line-by-line audit of the new analysis code was performed. Computations verified correct: the per-slot sufficiency statistic is algebraically identical to the original; the attribution gather reproduces the reference implementation exactly; both counterfactual semantics match brute-force recomputation through the real RMSNorm and unembedding; the retention arithmetic, the breakeven-reliability formula and the overlap chance term were all recomputed by hand and match the printed values; and the coded-flip consistency check (zero flips on non-firing decisions) passes on the real data.

The audit also found the following, none of which has been addressed.

### A. Cluster-size confound in the role-level result — the most consequential

`corr(role breadth, match quality)` = −0.54 to −0.81 (§5.12) has **no control for cluster occupancy**. Larger clusters have more stable centroids, so they match better; they also have less noisy breadth means. Occupancy total-variation distances of 0.23 to 0.46 show the clusters are very uneven. If cluster size correlates with mean breadth, that correlation is partly or wholly a size effect. **This finding should be treated as unconfirmed until size is partialled out.**

### B. The necessity table reports a pooled rate

The group flip rates in §6.5 are computed as (total flips) / (total opportunities) across a group's 100 features. That answers *"what is the probability that a randomly chosen active feature-decision flips?"* — an estimand dominated by the busiest features in the group. The mean of per-feature rates is a different number. The conclusion "general ≈ random > specialist" is stated under the pooled estimand and could move under the other.

### C. The projection-versus-coded gap quotes the confounded column

The reported gap of 0.0168 uses all-decisions rates. The adjacent table now reports given-fires rates, where the two interventions differ by only about 0.005. The 0.0168 is not wrong, but it sits under a heading a reader will associate with the corrected table.

### D. "Transitions" marks the whole timestep

The transition control flags all seven slots of a gripper-transition timestep. So "dx on transitions" means *dx at gripper-transition timesteps*, not *dx-channel transitions*. This is intended, but the labelling invites misreading.

### E. The 0.81% argmax mismatch is unattributed

All 446,096 rows carry valid action tokens, so the 0.9919 agreement is genuine disagreement rather than filtered junk. The most plausible cause is float16 storage of the residuals flipping decisions that were near-ties. It does not corrupt the flip analysis, which uses the recomputed argmax as its own consistent baseline, but the cause is currently a guess.

### F. The damage-participation-ratio null uses a clipped mean

Negative per-task damages are counted as zero before averaging, which slightly inflates the simulated damage. The direction is conservative for rejecting the null.

### G. Latent, never triggered

The Hungarian implementation returns pairs in unsorted order when its input is transposed (more rows than columns). All current usage is square, so it has never fired.

## 8.5 Unexplained observations

Two results in the record have no explanation and should not be quoted as findings.

**The sliced-Wasserstein anomaly.** Random dictionaries sit *closer* to `goal`'s signature distribution (0.0180 to 0.0182) than any real fine-tuned model does (0.0186 to 0.0232). At the distribution level, the four models differ from each other more than from noise. We do not know why.

**All seven channel partials are positive.** The seven channel concentrations sum to 1 per feature, so their correlations with a common variable should partly cancel. Seven small positive values instead suggests a shared artefact. A shuffled-breadth control would settle it and has not been run.

## 8.6 Loose ends in the experimental record

| Item | State |
|---|---|
| B1 channel analysis | `goal` only; the other three suites are three GPU jobs |
| Second-seed SAEs | `goal` only for action-position data, so the "retains about a quarter" figure is `goal`-only even though the slope conclusion is four-suite |
| Inventory clustering | `goal` only as target |
| Inheritance control for causal signatures | Impossible without a base-model SAE, which has not been trained |
| Signature-sharpness control | Sharpness is now saved alongside the m-sweep but the decile contrast has not been residualised on it |
| Hungarian at scale for the original Path B | A standing commitment from the plan; the new implementation exists but has not been applied to the pooled-data results |
| `libero-10` anomalies | Extreme concentration (`n_eff` 9.6), highest anti-aligned fraction, and the only suite reversing the `m = 1` breadth ordering. Behaves differently on three axes and has not been investigated |
| Coalition baseline discrepancy | A working note records 78.9%; the data gives 76.0%. Unexplained |
