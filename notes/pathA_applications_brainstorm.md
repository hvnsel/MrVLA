# What the breadth axis buys us — an applications brainstorm

**Status:** brainstorm, 2026-08. Nothing here is implemented, and nothing here is a
commitment. It is an inventory of what Path A + Phase 2 actually put in our hands, an
honest reading of the one question everyone asks first ("does causal mass predict
robustness?"), and a ranked list of things worth building. Written after a survey of the
VLA steering / diagnostics literature so we do not re-invent what already exists.

Reference paper: Swann et al., *"Sparse Autoencoders Reveal Interpretable and Steerable
Features in VLA Models"*, arXiv:2603.19183v2.

---

## 0. The one-paragraph version

Path A did not just add a metric. It added a **cheap exact causal instrument** (per-feature
contribution to the emitted action over 446k decisions with **zero forward passes**, plus
exact counterfactuals) and a **label-free breadth axis** that predicts out-of-sample in 4/4
suites against a floor of zero. The paper's own conclusion proposes exactly the application
we are now in a position to deliver — *"the presence of episode-specific features can be used
to diagnose the brittleness of common VLA fine-tuning procedures... our feature metrics may
serve as a training-time proxy for generalization without requiring real-world evaluation"* —
but its metric is base firing rate (§2.2), so it cannot support that application. **We are the
only ones holding a metric that could.** That framing — *cashing the check the paper wrote* —
is the strongest available positioning, and it is available whether or not the behavioural
link ever lands.

---

## 1. Asset inventory: what we are actually holding

Be precise about this, because the applications follow from the instrument, not from the
headline.

| Asset | What it is | Why it matters for applications |
|---|---|---|
| **Exact readout decomposition** | `φ_j = (l2/r)·z_j·⟨w_j, g⊙u_contrast⟩` at L31, sufficiency 0.936 | Per-decision causal attribution **without a forward pass**. The paper needs ~150 closed-loop rollouts *per feature* (20,250 episodes for 135 features, App. F.4). We need none. |
| **Exact counterfactuals** | P6's coded-flip machinery, 446k decisions × 396 features | We can measure "what would the action have been" offline. **Any policy-edit idea can be ceiling-tested before a simulator is touched.** This is the most under-exploited asset we have. |
| **Breadth axis (label-free)** | LOTO partial +0.362 to +0.473 (tensor4 control basis), 10/10 folds, 4/4 suites, floor ≈ 0 at z = 52–72; assumption-free stratified backstop +0.281 to +0.347 | A generality ordering that needs no hand labels, no human semantics, and survives base-rate + magnitude control. |
| **Concentration profile** | `n_eff` 9.6 / 49.3 / 102.3 / 105.9 of 2048; union of per-task top-10 is **11–15 features** against a chance union of 98 | A *checkpoint-level* scalar that varies 10× across suites. Candidate model-level diagnostic. |
| **Second axis** | Channel breadth 3.21 of 7 dims, correlated with task breadth only **+0.238** | A free second dimension for a quadrant map. Currently unused. |
| **Four load-bearing negatives** | projection ablation is 71.5% off-target; ablation MDE 9.7 pts at 20 ep/task; breadth split-half agreement only 0.13–0.47; coalition = the always-on set | Each of these is a **service to the field** and each constrains every idea below. |

The ratio in row 2 is the thing to keep repeating: **rollout evaluation is the binding
constraint in robot learning, and our instrument does not consume rollouts.**

---

## 2. The honest answer on "causal mass → robustness"

The user's instinct is right: it is **not proven**, and Phase 2 gives three specific reasons
to doubt the naive version. Worth stating these before any idea that leans on the link.

**(a) Breadth does not predict per-decision decisiveness (P6a).** Given a feature fires,
removing it flips the `dx` token at rate 0.0534 for the general group, 0.0565 for random,
0.0292 for specialists. **General ≈ random.** This is not straightforwardly a failure —
`adjusted_breadth` is residualised on magnitude by construction, so it claims *scope* of
influence, not strength — but it does mean "high causal mass from general features → more
robust decision" has no per-decision support today. The informative half is that specialists
are genuinely *less* decisive.

**(b) The behavioural test is a bounded null, not a measured absence (P3).** General-coalition
damage was +0.050 [−0.019, +0.119] against a design that resolves 9.7 points pooled. The
effect was never detectable. Detecting 5 points needs **79 episodes/task**, not 20.

**(c) Breadth is task-set-relative (P4a).** There is no fixed "true breadth" for a feature.
The ranking does not survive swapping which tasks define it, *even within one suite* (real ρ
at half-size 5: 0.126 / 0.209 / 0.467 / 0.269; the classical-model calibration gives ~0.82).
So *"the fraction of this episode's causal mass carried by general features"* is not a
well-formed scalar until a reference task set is fixed. **This is the sharpest objection to
the whole robustness-proxy idea and it is ours, not a reviewer's.**

**A mechanism-level trap on top of those three.** P2b found the top-50-by-causal-mass
coalition *is* the always-on set (40–43/50 overlap with top-50 by base rate; every member
above the ~98th percentile of firing frequency). So a reliance monitor built on **raw causal
mass** would largely be computing "fraction of this decision carried by always-on features",
which is close to a constant. It has to be built on **adjusted breadth**, whose top-50 is
**0/50 overlapping** with the mass coalition. That is a concrete, non-obvious design
correction that falls straight out of Phase 2, and it would have silently killed the first
implementation.

**And the causal direction is ambiguous even if a correlation appears.** Three readings:
(i) reliance on narrow features *causes* brittleness (mechanistic shortcut learning);
(ii) novelty is a **common cause** — an unfamiliar scene both raises narrow-feature reliance
and lowers success; (iii) **reverse causation** — the episode is already going wrong, which
produces weird states, which fire narrow features. Only a design that measures early
(pre-divergence), conditions on a novelty score, and ultimately *ablates* high-reliance
features to see success **improve** separates them. Per (b), that ablation needs 79 ep/task.

> **Net:** treat "breadth predicts robustness" as an open hypothesis with a known-good test
> design and a known price, not as a finding in waiting. Several ideas below are valuable
> *whether or not* it holds, and those should be preferred.

---

## 3. Idea families

### A. Diagnostics on an already-trained checkpoint (zero rollouts)

**A1. Base-rate-matched ablation control — reinterpret the paper's headline result.**
*The cheapest high-impact experiment in the entire program.*

The paper's Table 2 is: unsteered 97.5% → memorized(4 random) 92.5% → general(4 random) 65%
→ most-general(top 4) **0%**. Their tiers come from the firing-based classifier, and their
"memorized" tier is *by construction* low-coverage = low base rate. **The tier ordering is
perfectly confounded with firing frequency.** Our §2.2 (their score ≡ base rate under
balanced groups) plus our P2b (causal mass concentrates on the always-on set) together
predict that "ablating the top-4 general features destroys the policy" may simply be
**"ablating the 4 busiest directions destroys the policy"** — which needs no generality
story at all. Note our own P3 already saw this: the **firing-matched** condition did +0.740
damage, by far the largest effect in the table.

*The control:* four features matched on base rate to the general tier but *low* on
P(general). If they damage as much, Table 2 is an activity effect.
*Cost:* one extra ablation condition. *Payoff:* reframes the field's flagship behavioural
result. *Risk:* they may not be — in which case the paper's claim strengthens and we say so.

**A2. Checkpoint concentration profile as a model-level scalar.**
`n_eff` varies 10× across same-sized suites (libero-10 at 9.6 vs goal at 102.3). Right now
that is a description. It becomes a **diagnostic** the moment it correlates with something
about the checkpoint — OOD transfer, distractor sensitivity, sample efficiency of further
fine-tuning. The missing experiment is *not* more rollouts, it is **more checkpoints**:
same suite, varied seed / data volume / LoRA rank / training length.

*Known trap:* `n_eff` differences across suites may just track task-set structure
(long-horizon vs short). Only a within-suite, across-checkpoint comparison is clean.

**A3. Rare-but-decisive feature census.**
Define: high `|φ|`-per-firing, low base rate. Path A gives this directly. **The paper's
screen cannot produce this set** — Stage-1 candidates are screened by burstiness and coverage,
so a rare feature is classified memorized regardless of what it does. (The paper's own App.
C.5 concedes this: *"some features appear to be general yet activate in such a small portion
of the dataset that they are classified as memorized."*)

This is the **discovery** use, and discovery is precisely the regime where SAEs are argued to
beat baselines — cf. *"Use Sparse Autoencoders to Discover Unknown Concepts, Not to Act on
Known Concepts"* (2506.23845). Framing our axis as a discovery instrument sidesteps the
AxBench critique entirely, because we are not competing with prompting on known concepts.
Also the natural place to look for **covariate-shift landmines**: a direction that rarely
fires but decides the action when it does is exactly where distribution shift bites.

**A4. The quadrant map we already paid for.**
Task breadth × channel breadth, correlated only +0.238 — two largely independent axes, both
computed, never plotted. Cheapest concrete figure available, and it makes "the axis picks out
structure" visible rather than asserted. Pairs with §3.3's invariance axis if that ever runs.

**A5. Layer choice for intervention (a cross-path observation).**
Path B found above-chance recurrence peaks mid-network (L8–L16) and is **lowest at L31** —
which is the only layer where Path A's decomposition is exact. So *the layer where causal
attribution is cleanest is the layer where features are least universal.* Sharp testable
prediction: **steering vectors derived at L31 transfer across checkpoints worse than
mid-layer ones.** No VLA steering paper I found tests layer-choice against transfer; the
paper itself steers PaliGemma layer 5 without justifying the choice.

---

### B. Diagnostics during fine-tuning (the highest-novelty family)

**B1. Breadth collapse over the training trajectory.** *The single most compelling answer to
"what does it buy us."*

Take checkpoints along a fine-tuning run. Compute the breadth distribution / `n_eff` / mean PR
at each. **Does causal mass migrate from broad to narrow as fine-tuning proceeds?** If it
does, that is a mechanistic account of VLA overfitting that nobody has, and it is testable
against the ground truth everyone already logs (held-out success per epoch).

The payoff if the timing works out: **an early-stopping / checkpoint-selection criterion that
consumes zero rollouts.** Rollout evaluation is *the* bottleneck — the paper says so, the
data-curation literature says so. A no-rollout proxy is worth more than another point of
success rate.

*Decision rule, fixed in advance:* does breadth-at-epoch-t predict held-out-success-at-epoch-t
better than training loss does? If not, clean negative, cheaply obtained.
*Cost:* checkpoints (which we may need to produce) + one SAE per checkpoint. No new rollouts
beyond evals already run.

**B2. Fixed reference battery — the methodological fix B1 and B3 both need.**
P4a says breadth is defined relative to a task set, so cross-checkpoint PR comparison is **not
apples-to-apples** as currently defined. Fix: define breadth over a **fixed common probe
battery** held constant across checkpoints and models, rather than over each model's own
training suite. This converts breadth from a within-model ordering into a **comparable-across-
models quantity**, which is exactly what P4a says it currently is not. Cheap, purely
definitional, and a prerequisite for B1, B3, and any μ_t monitor (§2c).

**B3. Data-recipe scoring.**
The paper's Table 1 already gestures at this — π0.5/DROID 10.81% general vs OpenVLA/
LIBERO-Goal 0.45% — and reads it as *"broader and more diverse fine-tuning data may encourage
reusable internal features."* That is the right hypothesis measured with the wrong instrument
(base rate). Redo it with breadth over a fixed battery (B2). If the ordering across data
recipes matches the ordering of held-out success, we have a **data-mixture scorer that needs
no evaluation rollouts** — directly useful given the "curated 5% coreset recovers 85–90% of
full-dataset performance" line of work.

**B4. CAFT for VLAs — highest ceiling, highest risk.**
*Concept Ablation Fine-Tuning* (2507.16795) ablates undesired-concept directions **during**
fine-tuning and cuts misaligned generalization 10× with no data changes. The VLA analogue:
**project out the lowest-breadth (most scene/episode-specific) directions during LIBERO
fine-tuning**, forcing the policy to solve the task without leaning on memorized shortcuts.
Prediction: same in-distribution success, better held-out-suite success.

Why this is *ours* specifically: CAFT needs a way to name "undesired concept" **at scale and
without labels**. Human labelling does not scale and the paper's metric is base rate. The
label-free axis is the missing input.

*Two real problems to design around.* (i) Chicken-and-egg — breadth is measured after
fine-tuning. Either run a first pass then re-fine-tune, or compute breadth on the *base*
checkpoint over the reference battery (cleaner, untested). (ii) P4a's task-set-relativity is
a direct warning: the directions being ablated must be stable across the training run, and we
have no evidence they are. *This is the idea that turns a measurement paper into an
applications paper, and it is also the one most likely to fail for a boring reason.*

---

### C. Selection instruments (what to ablate, steer, or probe)

**C1. The head-to-head, already scoped in §3.2a.** Our label-free breadth ranking vs the
paper's P(general), same ablation protocol, same suites. This is the only experiment that
licenses "our axis is *better*" rather than "our axis is *different*". The paper set it up
well for us: protocol published, and a clean monotone published effect to beat or explain
away (below P(general) ≈ 0.5, indistinguishable from unsteered at 95%; above it, sharp drop).

**Add a third arm.** *Event-Grounded SAEs* (2605.17204) rank features by behavioural
keyframe events and report the strongest causal effects on **OpenVLA** specifically —
the same model and the same benchmark family we work in. That is the most direct competitor
to our ranking that exists, it postdates our plan, and it belongs in the comparison. Their
architecture split is also a warning worth quoting: *"on OpenVLA, individual event-aligned
features measurably drop closed-loop success; on π0.5, the backbone barely responds to
single-feature edits."*

**C2. Select targets by split-half-**stable** breadth.** P4's reliability numbers make this
free, and it is the cheapest fix for the underpowered ablation: the coalition we ablated was
drawn from a ranking whose split-half agreement is 0.13–0.47 depending on suite and on whether
the (invalid, per P4a) Spearman-Brown correction is applied, so part of it was noise, diluting
any real effect. Worth stating as a **general recommendation to the field** — nobody in the VLA-SAE
literature reports feature-ranking reliability at all. Small, cheap, genuinely novel.

**C3. Coded vs projection ablation, as a standalone methods note.** P6: 71.5% of a projection
ablation's flips land on decisions where the feature **never fired**. Every SAE-ablation
result in VLA-land — including the paper's Table 2 and App. F.4 — inherits this. We can
compute the correction exactly and cheaply. Two uses: reinterpret published results, and
offer coded ablation as the correct intervention. This is a workshop paper on its own and it
requires no new compute.

---

### D. Control and success rate — what is plausible and what is not

**Lead with the honest framing.** Nothing in Path A says removing or amplifying anything makes
the policy *better*. Attribution ≠ leverage, and the paper's own limitations section says so
directly: *"meaningful top activations of a feature does not imply reliable steerability."*
Their App. F.5 adds *"the impressive robustness of VLA models to activation-level
perturbations"* — even single features at α=100 leave the policy goal-directed. AxBench
(2501.17148) found SAE steering loses to prompting and finetuning on known concepts; the
partial rebuttal (2605.31183) only recovers parity **with a supervised selection pipeline**,
which is the thing we are trying to avoid needing.

So: **single-feature additive steering for success rate is the weakest use of this axis.**
Three better bets, in order:

**D1. Offline reranking — the idea most native to our finding.**
The strongest success-rate results in the literature are not single-feature amplification;
they are **search and selection**: COAST projects latents into a fitted success subspace
(+20% sim / +40% real), TACO does test-time anti-exploration scaling (+9.1% RoboTwin, +16%
real dual-arm), Token Steering intervenes in action-token space (10.0% → 72.5%). Our φ gives
a **scoring function over candidate actions with no extra forward pass**, because the logits
decompose additively. So: score candidate action bins by *how much of their margin comes from
broad vs narrow features*, and prefer the broad one.

**The property that makes this worth doing first:** OpenVLA is argmax-decoded over 256 bins,
and P6's exact counterfactual machinery already computes what the argmax becomes under coded
edits. So **we can measure the ceiling of this idea entirely offline, over all 446k decisions,
before allocating a single GPU-hour of simulation.** Very few steering ideas can be killed or
promoted that cheaply. If the offline ceiling is small, we drop the whole steering direction
having spent CPU.

**D2. A reliance monitor (μ_t), rebuilt correctly.**
§3.2a defined `μ_t` = fraction of the decision carried by non-general features. Phase 2 did
not kill it; it just never tested it — and it made it cheap, since φ costs no forward pass, so
**μ_t is a free runtime scalar.**

*What differentiates it:* the existing VLA failure monitors are all **supervised or
distributional** — SAFECAST trains a hidden-state probe on success/failure-labelled rollouts;
FIPER (2510.09459) uses random-network-distillation OOD scores plus action-chunk entropy;
VLA-FAIL and Hide-and-Seek are in the same family. Ours would be **unsupervised and
mechanistic** — no labelled rollouts needed to construct it. That is a real gap in the
literature, and it is a defensible small result *either way*, independent of whether the
causal story holds.

*Non-negotiable design constraints, all inherited from Phase 2:*
- build it on **adjusted breadth**, never raw mass (§2, the P2b trap — else it is near-constant);
- fix the reference task battery first (B2), or μ_t is not well-defined (P4a);
- measure **early in the episode, before divergence**, or it is reverse causation;
- condition on a novelty score, or it is a common-cause artefact;
- benchmark against action entropy, logit margin, and an RND/OOD score — not against nothing.

*A free competing baseline we already have:* the sufficiency decomposition splits the action
margin into features (0.531) vs a constant default-action bias (0.405). "Fraction of this
decision carried by the default bias rather than by features" is a second runtime scalar,
available at no cost, and it is a genuinely different signal from μ_t.

**D3. Steering-layer choice (A5 above).** Moderate cost, and the prediction is sharp.

**D4. See `vlm_in_the_loop_brainstorm.md`.** A follow-on brainstorm on whether the axis can
*inform* a process that improves the policy rather than steering it directly — VLM fragility
labels joined to phi, an action-space phrasebook, and a free per-decision fragility signal
(margin / flip-susceptibility) that supersedes the mu_t monitor in D2.

---

### E. Contributions that stand alone if the behavioural link never lands

Worth being explicit about the fallback, because it is strong.

1. **A confound control the subfield now has to pass.** Any future claim that "general
   features are load-bearing" must beat a base-rate-matched control (A1). Permanent.
2. **A causal instrument that does not consume rollouts.** ~150 rollouts per feature → zero.
3. **A reliability standard for feature rankings** (C2). Nobody reports one.
4. **A power-bound template for ablation designs** (P3). Published null results in this design
   are uninterpretable without one, and now there is a worked example.
5. **The projection-vs-coded correction** (C3), which touches every prior result in the area.

---

## 4. What the literature already does (so we do not re-invent it)

| Line of work | Representative | What they get | Gap we can occupy |
|---|---|---|---|
| Latent steering for success | COAST (2605.17144) | +20% sim / +40% real, conceptor projection into success subspace | **Not SAE-based.** Fits subspaces from success/failure rollouts — i.e. supervised. Ours is label-free. |
| Test-time scaling | TACO (2512.02834) | +9.1% RoboTwin, +7.5% Simpler, +16% real | Search/selection beats steering — supports D1 over naive steering |
| Activation-guide steering | GuideVLA | +5.7% grasp on unseen objects | Averages activations from *successful* rollouts — supervised again |
| Action-token intervention | Token Steering (2606.15021) | 10.0% → 72.5%, 16.7% → 93.8% on 2 tasks | Human-in-the-loop; complementary, not competing |
| Runtime failure monitors | SAFECAST, FIPER (2510.09459), VLA-FAIL | probe / OOD+entropy based | All supervised or distributional; **no mechanistic, label-free monitor exists** → D2 |
| SAE feature ranking for VLAs | Event-Grounded SAE (2605.17204) | strongest causal effects on OpenVLA via behavioural-event grounding | **Closest competitor to Path A.** Must be an arm in C1. |
| VLA mechanistic study | Not All Features Are Created Equal (2603.19233) | visual pathway dominates; motor programs bound to scene coordinates | Independently confirms our §2.3: **mean-pooling destroys action-critical information** — vindicates the A1 rebuild |
| SAE steering skepticism | AxBench (2501.17148), rebuttal (2605.31183) | SAEs lose to prompting on known concepts; parity only with supervised selection | Argues hard for the **discovery** framing (A3), not the acting-on-known-concepts framing |

**Two convergences worth flagging as more than coincidence:**

- COAST reports that *failure modes share substantial structure across tasks while success
  representations remain largely task-specific.* That is a striking mirror of Path B: the most
  causally **general** features recur **least** across independently fine-tuned models, while
  specialists recur more. Two completely different methods, same shape of answer. Independent
  corroboration of C6/C7 from outside our pipeline, and worth citing as such.
- "Not All Features Are Created Equal" finds motor programs are **spatially bound to scene
  coordinates rather than abstract task representations.** That is a mechanistic prediction
  about what our low-breadth "specialist" features should turn out to be, and it is testable
  against our existing exemplar frames.

**And the actual gap:** none of the success-rate-improving methods above is SAE-feature-based.
Either that is an opportunity, or it is evidence that SAE features are the wrong handle for
control and the AxBench skeptics are right. D1's offline ceiling test is designed to tell us
which, cheaply, before we commit.

---

## 5. Ranked shortlist

Ranked by (payoff × probability of a clean result) / cost. "Clean result" includes clean
negatives — several of these are worth running *because* the negative is publishable.

| # | Idea | Cost | Decision rule | Why this rank |
|---|---|---|---|---|
| 1 | **Base-rate-matched ablation control** (A1) | one extra condition | If matched controls damage ≈ general tier, the paper's Table 2 is activity | Reframes the field's flagship behavioural result for almost nothing; directly extends our confound-first identity |
| 2 | **Offline reranking ceiling** (D1) | CPU only, existing data | If offline margin gain is negligible, drop the steering direction entirely | Kills or promotes the whole control program without a GPU. Best value-per-dollar in the list |
| 3 | **Coded-vs-projection methods note** (C3) | already computed | — | Done work, not yet packaged. Touches every prior ablation result in the area |
| 4 | **Fixed reference battery** (B2) | definitional | — | Prerequisite for 5, 6, and 8. Cheap and unblocks the rest |
| 5 | **Breadth collapse over training** (B1) | checkpoints + SAEs, no new rollouts | Beat train loss at predicting held-out success, or clean negative | Highest scientific novelty; the "no-rollout early stopping" payoff is the best story we have |
| 6 | **Rare-but-decisive census** (A3) | CPU | — | Cheap, discovery-flavoured, could produce the "a feature their screen could not find" moment |
| 7 | **Breadth vs P(general) vs event-grounded head-to-head** (C1) | GPU rollouts, 79 ep/task | Which ranking predicts ablation cost best | The "our axis is better" test. Expensive, and now has three arms |
| 8 | **μ_t monitor, rebuilt** (D2) | rollouts we would run anyway | AUROC vs entropy / margin / OOD baselines | Publishable either way; five design constraints must hold or it is worthless |
| 9 | **Quadrant map** (A4) | hours | — | Already-paid-for figure; makes the axis concrete |
| 10 | **CAFT-for-VLA** (B4) | training runs | ID success held, OOD success improved | Highest ceiling in the list; highest chance of failing for a boring reason |
| 11 | **Checkpoint concentration profile** (A2) | needs new checkpoints | Does `n_eff` correlate with anything about the checkpoint | Interesting, but currently a description in search of a dependent variable |
| 12 | **Layer transfer of steering vectors** (A5/D3) | moderate | L31 vectors transfer worse than mid-layer | Sharp prediction, nobody has tested it, but lower stakes |

---

## 6. Traps every idea above must respect

Carried over from Phase 2. Each of these has already bitten once.

1. **Build reliance/monitor statistics on adjusted breadth, never raw causal mass.** P2b: the
   mass coalition *is* the always-on set (40–43/50), and its overlap with the top-50 by
   adjusted breadth is **0/50**. Raw mass gives a near-constant.
2. **Fix a reference task set before comparing breadth across models or checkpoints.** P4a:
   breadth is task-set-relative, and Spearman-Brown does not apply to a participation ratio.
   Quote **uncorrected** split-half agreement at a stated half-length.
3. **Any ablation claim needs its MDE.** P3: 20 ep/task resolves 9.7 points pooled. Below that
   the design cannot distinguish inert from unmeasured.
4. **Prefer coded ablation; if projection is used, report the off-target share.** P6: 71.5%.
5. **Every new score must beat base firing rate** (commitment #2). This killed one metric
   family already and it is the reason A1 exists.
6. **Do not quote the disattenuated bounds.** P4a retired `|r_true| ≥ 0.750`; it does not
   return. Publish +0.362 to +0.473 (tensor4) and the stratified backstop.
7. **The rank-tie defect is still live** (P9/§8.3) and it touches `adjusted_breadth`, which
   selects every ablation and steering target. Any new selection code must use
   `mrvla.stats.rankdata_average`.
8. **Watch libero-10.** It is anomalous on three independent axes (`n_eff` 9.6, highest
   anti-aligned fraction, only suite reversing the m=1 breadth ordering) and should be
   reported separately rather than pooled.

---

## 7. The framing to lead with

If only one sentence survives from this document:

> The reference paper proposed using SAE feature metrics as a **rollout-free training-time
> proxy for generalization**, and then measured those features with a statistic that reduces
> to base firing rate. We built the label-free causal axis that could actually carry that
> application, together with an attribution instrument that costs no forward passes — and the
> first thing that axis says is that the field's flagship behavioural result needs a
> base-rate-matched control.

Everything in §3 is downstream of that sentence. Ideas 1–4 in the shortlist test it for very
little money, and three of the four produce a reportable result on both branches.
