# Chapter 6 — Complete results

Every number produced by the programme, in one place. Phase 1 figures are as reported in the project record; Phase 2 figures were computed or recomputed directly from the stored data during the audit.

## 6.1 Measurement-validity findings (what does not work)

| Finding | Evidence | Status |
|---|---|---|
| The reference paper's classifier is circular | Labels are screened by the same metrics the classifier regresses onto; ambiguous cases excluded. 100% LOO accuracy measures label-metric consistency, not construct validity | Argument, not measurement |
| Group-balanced coverage **is** base firing rate | Algebraically identical under LIBERO's balanced task groups; partial correlation after controlling base rate is empty | Measured |
| The paper's `coverage` is base firing rate | Same argument applies | Argument |
| `max_group_rate` is a *memorisation* signal | After controlling base rate: correlates 0.94 with concentration; predicts held-out firing **negatively**, −0.05 to −0.27, in all 20 layer x suite cells | Measured |
| Pooled prefill activations cannot support causal analysis | They are not the vectors that feed the action readout | Structural |

## 6.2 Path A — causal task-breadth

### The viability gate (`goal`, k = 100)

| Component | Slope | Note |
|---|---|---|
| features + bias | **0.9361** | gate threshold 0.80 — **PASS** |
| features alone | 0.531 | |
| constant bias | 0.405 | later identified as largely the gripper (§5.9) |
| error | 0.064 | |
| frozen-`r` arithmetic canary | 1.0000 | mean abs discrepancy 2.7e-14 |
| abandoned L1 argmax gate | 0.72 | stalled; 0.76 even at k = 256 |

### The core result, four suites

| Suite | `partial \| both` | Folds + | Worst fold | Column-shuffle floor | z | p |
|---|---|---|---|---|---|---|
| goal | **+0.493** | 10/10 | +0.399 | +0.0004 | +52.1 | < 0.001 |
| spatial | **+0.449** | 10/10 | +0.333 | −0.0000 | +47.9 | < 0.001 |
| object | **+0.387** | 10/10 | +0.359 | +0.0000 | +40.2 | < 0.001 |
| libero-10 | **+0.535** | 10/10 | +0.399 | −0.0000 | +55.8 | < 0.001 |

Supporting figures (`goal`): 446,096 decisions; PR mean 6.05 (p10 2.65, p90 9.91); raw PR correlates +0.82 with base firing rate, which is why the adjusted version is used; fold standard deviation 0.053; attenuation-corrected `|r_true| >= 0.819`.

Invalid control, for the record: task-label permutation gives +0.4840 / +0.4508 / +0.3843 / +0.5327 — indistinguishable from the observed values.

### Concentration and cross-task reproducibility

| Suite | `n_eff` (of 2048) | Top-10 | Top-50 | Top-100 | Gini | Top-50 overlap | Per-task `n_eff` (min/med/max) |
|---|---|---|---|---|---|---|---|
| goal | 102.3 | 0.276 | 0.448 | 0.529 | 0.684 | 41.4/50 = 33.9x | 81.9 / 93.9 / 114.8 |
| spatial | 105.9 | 0.256 | 0.436 | 0.516 | 0.644 | 44.8/50 = 36.7x | 91.6 / 102.4 / 109.1 |
| object | 49.3 | 0.355 | 0.494 | 0.570 | 0.724 | 43.3/50 = 35.5x | 45.3 / 48.5 / 55.6 |
| libero-10 | 9.6 | 0.588 | 0.731 | 0.780 | 0.889 | 42.7/50 = 35.0x | 9.3 / 9.5 / 10.3 |

Controls (`goal`): firing-rate ranking gives `n_eff` 886.8, top-50 share 0.140, Gini 0.449. Column-shuffled causal mass gives `n_eff` 653.6, top-50 0.208, Gini 0.482. Chance overlap for top-50 of 2048 is 1.22.

### Reliability of the breadth ranking

| Suite | Raw PR (Spearman–Brown) | Adjusted breadth (Spearman–Brown) | Shuffle floor |
|---|---|---|---|
| goal | 0.221 | 0.363 | +0.002 |
| spatial | 0.349 | 0.273 | −0.000 |
| object | 0.637 | 0.469 | −0.002 |
| libero-10 | 0.424 | 0.337 | +0.001 |

## 6.3 Path B — cross-model recurrence

### Activation-based recurrence (pooled data, 20 layer x suite cells)

| Quantity | Value |
|---|---|
| `q_cross` | 0.37 – 0.54 |
| permutation floor `q_perm` | 0.24 – 0.28 |
| gap above floor | **+0.13 to +0.26 in all 20 cells** |
| confound `R^2` (base rate + inheritance) | 0.001 – 0.113 |
| discrimination (10th to 90th percentile) | 0.27 – 0.69 |
| chance-corrected retention, layers 0/8/16/24/31 | 0.587 / 0.619 / 0.635 / 0.585 / 0.473 |
| same-model different-seed `q_seed` | 0.640 / 0.657 / 0.629 / 0.612 / 0.531 |
| `corr(q_cross, paper's P(general))` | −0.17 to +0.17 across all 20 cells |

### Causal-signature recurrence (action-position)

| Quantity | Value |
|---|---|
| chance floor (random decoders through each model's own head) | 0.226 |
| top-decile (most general) minus rest — goal | −0.017 |
| — spatial | −0.015 |
| — object | −0.008 |
| — libero-10 | −0.011 |
| discarded bin-permutation null | manufactured a fake gap of ≈ 0.33 |

### The A x B join

| Quantity | Value |
|---|---|
| `corr(adjusted breadth, recurrence)` | **−0.127** |
| measurement ceiling at breadth reliability 0.363 | ±0.602 |
| implied `\|r_true\|` lower bound | ≥ 0.211 |
| recurrence reliability needed for `\|r_true\| <= 0.30` | 0.494 |

## 6.4 Behaviour — ablation and steering

### `goal` coalition run (baseline 152/200 = 0.760 [0.696, 0.814])

| Condition | Success | Damage | 95% CI | b10/b01 | p (exact) | MDE |
|---|---|---|---|---|---|---|
| firing | 0.020 | +0.740 | [+0.679, +0.801] | 148/0 | < 0.001 | 0.169 |
| general | 0.710 | +0.050 | [−0.019, +0.119] | 30/20 | 0.203 | 0.097 |
| random | 0.755 | +0.005 | [−0.065, +0.075] | 26/25 | 1.000 | 0.098 |
| specialist | 0.785 | −0.025 | [−0.088, +0.038] | 18/23 | 0.533 | 0.087 |

Design resolution: discordance rate 0.253; pooled MDE **9.7 points**; per-task MDE **22.6 points**; 79 episodes/task needed for a 5-point effect. Scope test (damage participation ratio) unresolved for every condition.

### `goal` single-feature run (baseline 134/180 = 0.744)

| Condition | Damage | 95% CI | p (exact) |
|---|---|---|---|
| only_1134 | −0.064 | [−0.133, +0.005] | 0.108 |
| only_1140 | −0.006 | [−0.077, +0.064] | 1.000 |
| only_1167 | −0.014 | [−0.096, +0.067] | 0.864 |
| only_1235 | −0.013 | [−0.088, +0.063] | 0.871 |
| only_1628 | +0.007 | [−0.076, +0.090] | 1.000 |
| only_1999 | +0.006 | [−0.078, +0.090] | 1.000 |

Pooled MDE 10.8 points, per-task 22.1. Damage participation ratio undefined for every condition — no task showed positive damage.

**Steering** (qualitative): steering toward a grasp-associated feature made the gripper close earlier in the episode.

## 6.5 B1 — action channels (`goal`)

Validation: recomputed argmax equals emitted token on **0.9919** of decisions.

### Per-channel sufficiency

| | dx | dy | dz | droll | dpitch | dyaw | gripper |
|---|---|---|---|---|---|---|---|
| features + bias | 0.9403 | 0.9343 | 0.9048 | 0.9489 | 0.9204 | 0.9365 | **0.9917** |
| features alone | 0.6066 | 0.8326 | 0.5931 | 0.9030 | 0.9227 | 0.7759 | **−0.0461** |
| absolute causal-mass share | 0.1353 | 0.1331 | 0.1377 | 0.1503 | 0.1453 | 0.1446 | 0.1537 |

### Breadth versus channel

| Quantity | Value |
|---|---|
| `corr(adjusted breadth, gripper concentration)`, partial | **+0.069** |
| per-channel partials | dx +0.081, dy +0.083, dz +0.195, droll +0.089, dpitch +0.099, dyaw +0.162, gripper +0.069 |
| channel participation ratio | mean 3.21 of 7 (p10 1.33, p90 6.32) |
| `corr(task breadth, channel breadth)` | +0.238 |
| general vs specialist profile difference | ≤ 0.07 in any channel; P(g>s) 0.506 – 0.618 |
| tie exposure of `base_rate` | **10.3% of values tied** |

### Necessity, given the feature fires

| Group | dx (projection) | dx (coded) | gripper (projection) | gripper (coded) |
|---|---|---|---|---|
| general | 0.0534 | 0.0569 | 0.0008 | 0.0007 |
| random | 0.0565 | 0.0564 | 0.0006 | 0.0007 |
| specialist | 0.0292 | 0.0342 | 0.0010 | 0.0011 |
| firing | 0.1284 | 0.1150 | 0.0197 | 0.0092 |

### Projection versus coded ablation

| Quantity | Value |
|---|---|
| projection flip rate, all decisions | 0.0229 (over 176,654,016 feature-decisions) |
| projection flip rate, given the feature fires | 0.0695 (over 16,586,298) |
| coded flip rate, all decisions | 0.0060 |
| coded flip rate, given the feature fires | 0.0642 |
| fraction of feature-decisions where the feature is active | 9.4% |
| **projection flips occurring when the feature did NOT fire** | **71.5%** |
| coded flips occurring when the feature did not fire | −0.5% (i.e. zero — consistency check) |

## 6.6 A4 — inventory-level recurrence

### Geometry gate

| Object | Effective rank | 90% energy | 99% energy |
|---|---|---|---|
| contrast-centred `W_U_act` | 205.5 | 213 | 250 |
| **signature space occupied by goal's features** | **50.8** | 193 | 246 |
| random 12-dim subspace explains | 5.8% of an arbitrary direction | | |

Head comparison across the four models: `g` and `act_ids` identical; `W_U_act` differs by max 0.0266 (1.64x its own entry standard deviation). Impact on signatures: cosine 0.9967 – 0.9990 (worst single feature 0.964).

### The m-sweep (target = goal)

| | m=1 | m=2 | m=3 | m=4 | m=5 | m=6 | m=7 | m=8 |
|---|---|---|---|---|---|---|---|---|
| most-specialist decile | 0.3065 | 0.3952 | 0.4507 | 0.4924 | 0.5268 | 0.5562 | 0.5822 | 0.6052 |
| most-general decile | 0.2828 | 0.3683 | 0.4248 | 0.4692 | 0.5051 | 0.5361 | 0.5637 | 0.5881 |
| random floor | 0.2464 | 0.3324 | 0.3924 | 0.4396 | 0.4788 | 0.5124 | 0.5420 | 0.5684 |
| seed ceiling | 0.4528 | 0.5278 | 0.5730 | 0.6070 | 0.6347 | 0.6584 | 0.6791 | 0.6974 |
| **chance-corrected retention** | 0.244 | 0.269 | 0.272 | 0.271 | 0.268 | 0.264 | 0.259 | 0.253 |

### Four-suite slope comparison

| Target | General slope | Specialist slope | Difference | Random-floor slope | General − floor |
|---|---|---|---|---|---|
| goal | +0.3053 | +0.2987 | +0.0066 | +0.3220 | −0.0167 |
| spatial | +0.3040 | +0.2947 | +0.0093 | +0.3213 | −0.0173 |
| object | +0.3049 | +0.2985 | +0.0064 | +0.3218 | −0.0169 |
| libero-10 | +0.3166 | +0.3102 | +0.0064 | +0.3219 | −0.0053 |

Anti-aligned fractions (share of features whose best absolute match points the opposite way): goal-vs-others 0.115 – 0.366; libero-10 highest against every partner (0.25 – 0.37).

### Inventory clustering (target = goal)

| k | Inventory match | Random floor | Seed ceiling | Retention | `corr(role breadth, match)` | `corr(role breadth, occupancy diff)` |
|---|---|---|---|---|---|---|
| 8 | 0.270 | 0.203 | 0.547 | +0.194 | −0.810 | −0.071 |
| 16 | 0.266 | 0.160 | 0.540 | +0.279 | −0.574 | −0.517 |
| 32 | 0.227 | 0.170 | 0.397 | +0.251 | −0.537 | −0.197 |

Stability across `k`: standard deviation 0.0192. Sliced Wasserstein: spatial 0.0208, object 0.0186, libero-10 0.0232, random 0.0180 – 0.0182, seed 0.0140.
