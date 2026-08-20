# If SAEs can't steer, can they inform something that does?

**Status:** brainstorm, 2026-08. Companion to `pathA_applications_brainstorm.md`. Nothing
implemented. Written after reading StressDream (arXiv:2606.00267) and surveying the
VLM-supervisor / failure-recovery literature, because the honest reading of §D in the
companion doc is that **direct SAE steering is the weakest use of our instrument** — so the
question becomes whether it can feed a process that does work.

Short answer: yes, and the best version is cheaper than the one that first suggests itself,
because we are already sitting on a free per-decision fragility signal and have not looked
at it.

---

## 1. StressDream, and whether it is the right neighbour

**What it does.** Optimises the *initial noise* of a diffusion video world model so the
imagined future is stressed but still plausible. Two objectives: a **VLM semantic objective**
that reasons about the generated video and supplies gradients toward an inference-time text
prompt ("task failure"), and a **plausibility objective** (typical-set constraint under the
Gaussian prior) that stops the optimised noise drifting OOD. Uses: policy **evaluation**
(surface failure-prone actions without drawing prohibitively many nominal samples) and policy
**improvement** (favour actions that avoid the surfaced failures). Stated limitation: it
inherits the base world model — it can steer within flawed videos, or fail to imagine
outcomes absent from the world model's training distribution.

**Is it our neighbour?** Partly, and the useful part is not the method.

| | StressDream | What we would do |
|---|---|---|
| Where the search happens | pixel space (diffusion noise) | feature space (φ at L31) |
| What the VLM supplies | gradients toward a failure prompt | labels on real rollouts |
| Prerequisite | a good video world model | rollouts we already have |
| Improvement mechanism | favour robust **actions** | intervene on **mechanisms**, or rerank actions |

Three things to take from it and one to leave.

- **Take:** it is existence proof that a VLM's judgement about impending failure is a *usable
  optimisation signal*, not just a post-hoc label. That legitimises the whole "VLM as fragility
  labeller" half of the idea below.
- **Take:** it names the real problem correctly — **fragile moments are rare in nominal
  rollouts**, so you cannot just collect them, you have to manufacture them.
- **Take:** its improvement mechanism is *action selection*, which is the same slot as the
  offline reranking idea (companion §D1). Ours needs no world model and no diffusion
  optimisation, which is a genuinely competitive framing.
- **Leave:** the diffusion world model. It is a large lift, it is the source of StressDream's
  own stated limitation, and we have a cheaper way to manufacture stress (§3).

---

## 2. Verdict on the idea, split into its two halves

The proposal has two separable halves and they have very different prospects.

### Half A — VLM watches rollouts, flags fragile timesteps. **Good, and cheap.**

This is worth doing, but note the capability itself is **crowded**: AHA, *Robot Critics that
Sweat the Small Stuff* (2606.21572), DenseReward (2607.13033), SAFE (2506.09937), VLA-FAIL,
Foresight, Self-Refining VLM. Several already produce per-timestep failure labels without
manual annotation. So the contribution is **not** the labeller. The contribution is what we
join it to: nobody has a per-decision *exact causal decomposition of the action* to join those
labels against, and we do, for free.

The join turns a question we currently cannot afford to answer into one we can. Right now the
behavioural link to breadth is a bounded null (P3) because rollout-ablation gives **one bit per
episode** and needs 79 ep/task to resolve 5 points. A VLM fragility labeller gives **a label per
timestep**, each joinable to a 2048-dim φ vector. That is orders of magnitude more
information per GPU-hour, and it converts an underpowered *interventional* question into a
well-powered *observational* one — which is the regime this repo's confound machinery is
actually built for.

Immediately answerable once labels exist, all on saved artifacts:

- Is causal mass more **concentrated or more diffuse** at fragile timesteps? (`n_eff` per timestep.)
- Does the **default-action bias share** spike there? We know the margin splits into features
  0.531 / constant bias 0.405. *Is the policy coasting on the default when it is about to
  fail?* Nobody else can ask this — it requires the exact decomposition. Best hypothesis in
  the document.
- Do fragile timesteps recruit **low-breadth specialists** over broad features? (This is the
  actual test of the robustness story, done observationally.)
- Are fragile timesteps **channel-localised**? We have the seven-way split already.

### Half B — VLM decides *which features* to steer. **A stretch as stated, and rescuable.**

Right to be suspicious. Four reasons the naive version fails:

1. **Latency.** The VLA decides at 10–50 Hz. A VLM call is ~1 s. It cannot sit in the
   per-decision loop.
2. **The VLM has no idea what F1134 is.** Bridging that needs autointerp descriptions of 2048
   features, which are unreliable in LLMs and worse here — P4a says feature identity is not
   even stable across task sets.
3. **Meaning ≠ leverage.** The paper's own limitation: *"meaningful top activations of a
   feature does not imply reliable steerability."* Knowing what a feature *represents* does not
   tell you what steering it *does*.
4. **Most of the pipeline is already taken.** FPC-VLA (2509.04018) is a VLM supervisor
   triggered at keyframes that evaluates action viability and **emits natural-language
   corrections specifying direction and magnitude**. ReCoVLA, FailSafe, Steerable Policies and
   *Imagining Recovery* (2608.14822) occupy adjacent ground.

**But the fix is clean, and it comes from inverting the roles.** The mistake is asking the VLM
to do the mechanism-side reasoning. Don't. Build a **phrasebook** offline:

> For each feature, use the exact counterfactual (P6, no forward pass) to compute what coded
> intervention on it does **in action space**: *"amplifying F1134 shifts `dz` down ~4 bins on
> decisions where it fires; ablating it flips the emitted token 6.4% of the time it is active."*

That is a **mechanical description with no semantics in it**, computed exactly, offline, from
data already on disk. Then the VLM only ever has to state a desired action-space correction in
plain language — which FPC-VLA already shows it can do — and the phrasebook resolves it to
features. **The VLM never learns what a feature means; it issues a query and we do a lookup.**

That defuses all four objections: no autointerp (2), grounded in effect not meaning (3),
episode-level or keyframe-triggered rather than per-decision (1), and the novel hop is
precisely the one nobody has (4).

**Where it earns its keep, and it is narrow.** If you can adjust the action directly, why route
through features? Two real answers, and only two:

- **Staying on the policy's manifold.** Clamping the action overrides the learned priors;
  Token Steering's entire pitch is that guiding "preserves the dexterity, smoothness, and task
  priors learned by the VLA." Feature-space intervention is the version that guides.
- **Behaviours the language interface cannot reach.** The paper's §5.4 is the demonstration:
  "close the gripper" as a prompt gives 0.005 closure; steering F586 gives 0.653. If the VLM's
  correction is *"close the gripper now"*, a language-level correction **fails** and a
  feature-level one works.

So Half B is interesting, and it is defensible — but it is one hop wide, it sits downstream of
Half A, and it should not be started first.

---

## 3. The thing we are already sitting on and have not looked at

Both halves above need fragile timesteps, and fragile timesteps are rare — StressDream's whole
premise. But **we can already identify them mechanically, for free, with no VLM and no labels.**

P6 computes exact counterfactuals over 446k decisions × 396 features. That machinery gives, per
decision, the **margin between the emitted bin and its nearest competitor**, and how easily a
single feature flips it. Call it **flip-susceptibility**: a thin-margin decision is one where the
policy is barely committed. That is a fragility signal that is *mechanistic, dense, per-decision,
label-free, and already paid for.*

This **inverts the user's proposal in the direction that makes it cheap**: the VLM stops being
the generator of fragility labels and becomes the **validator** of a signal we compute ourselves.

- **The experiment:** do mechanically thin-margin decisions coincide with VLM-flagged (or
  simply observed) behaviourally fragile moments?
- **If yes:** we have a runtime failure monitor that needs **no labels, no probe training, no
  VLM at inference, and no extra forward pass** — strictly cheaper than SAFECAST (trains a
  probe on labelled rollouts) or FIPER (RND + action-chunk entropy). That is a real result.
- **If no:** thin margins are benign, which is itself worth knowing and kills a plausible idea
  for one CPU job.

This is also a **better statistic than the `μ_t` reliance monitor** in the companion doc.
`μ_t` asks "what kind of features carry this decision", which needs breadth to be meaningful
and inherits every P4a problem. Flip-susceptibility asks "how committed is the policy to this
decision", which needs nothing but the decomposition.

> **The refinement that decides whether this works at all.** A flip between adjacent bins is
> nothing — the bins are a discretisation of a continuous action, so bin 128 → 129 is a
> rounding difference while 128 → 200 is a different behaviour. Flip-susceptibility **must be
> weighted by bin distance**, not counted as a binary flip. Counting binary flips would fill
> the "fragile" set with harmless rounding ties and the statistic would look like noise for a
> reason that has nothing to do with the hypothesis. P6's flip rates as currently reported are
> binary and should not be reused unmodified.

**And it gives us a cheap substitute for StressDream's world model.** We do not need to imagine
stressed futures; we can *manufacture* stress with machinery we already have — perturbed initial
states, added distractors, and the existing ablation harness — then label the failures that
result. Same purpose, no diffusion model, and no dependence on a world model's training
distribution.

---

## 4. Where each hop stands in the literature

| Hop | Status | Who |
|---|---|---|
| VLM flags fragile timesteps / failure moments | **taken** | AHA, Robot Critics (2606.21572), DenseReward, SAFE, VLA-FAIL, Foresight |
| VLM emits NL correction with direction + magnitude | **taken** | FPC-VLA (2509.04018), ReCoVLA, FailSafe, Steerable Policies |
| VLM-guided stress generation for policy improvement | **taken** | StressDream (2606.00267) |
| Correlate SAE activations with outcomes to select steering features | **taken (LLMs)** | CorrSteer (2508.12535) — "correlation as selection heuristic, intervention as causal test" |
| **Exact per-decision causal decomposition joined to behavioural fragility labels** | **open** | — |
| **NL correction → feature intervention via an exactly-computed action-space phrasebook** | **open** | — |
| **Label-free mechanistic fragility signal (margin / flip-susceptibility) as a runtime monitor** | **open** | — |

The last three are the only places our instrument is doing work nobody else can do. Everything
above them we should consume, not rebuild.

---

## 5. Ranked, with honest odds

| Idea | Cost | Verdict |
|---|---|---|
| **Flip-susceptibility as a mechanistic fragility signal** (§3) | CPU, existing artifacts | **Do this first.** Free, novel, and it either produces the cheapest failure monitor in the literature or dies for one CPU job. Must be bin-distance weighted |
| **VLM fragility labels joined to φ** (Half A) | VLM inference over saved rollouts | **Solid.** Converts the P3 bounded null from interventional to observational. The bias-share hypothesis is the best single question in either document |
| **Manufactured stress via perturbation + existing ablation harness** (§3) | rollouts | **Sensible.** StressDream's purpose without StressDream's prerequisite |
| **Action-space phrasebook** (Half B, offline half) | CPU, existing artifacts | **Cheap and reusable.** Useful on its own as a feature catalogue even if no VLM ever calls it |
| **VLM → phrasebook → feature intervention, closed loop** (Half B, full) | rollouts + VLM | **Interesting, narrow.** Only defensible where language cannot reach the behaviour (paper §5.4) or where clamping the action would break the priors |
| **VLM learns feature semantics at runtime** | — | **Don't.** This is the version that is a stretch, and it is a stretch for four independent reasons |

---

## 6. Traps specific to this line

1. **Circularity in the labels.** If the VLM flags a timestep partly because the robot is
   *already* visibly wrong, then any feature firing at visibly-wrong states gets implicated
   regardless of causation. Labels must be **prospective** — emitted from frames before the
   visible divergence, with the future masked — and compared against **phase- and task-matched
   timesteps from successful episodes**, not against episode averages. Same reverse-causation
   discipline §3.2a already imposes on `μ_t`.
2. **Label reliability is unmeasured and must not stay that way.** Re-run the labeller on the
   same episodes and report agreement; validate against human labels on a subsample. One
   consolation: the labels are the **target**, so noise attenuates rather than inflates — a
   positive result under noisy labels is a lower bound. That is the favourable direction, unlike
   the correlated-error problem in P4.
3. **Bin distance, not binary flips** (§3). The single most likely way to get a null for the
   wrong reason.
4. **Novelty is the standing confound.** A novel scene raises fragility *and* lowers success
   independently. Condition on it or the whole thing is a novelty detector with extra steps.
5. **Everything here is L31, one model family, simulation.** The phrasebook in particular is
   defined by the argmax readout and has no direct analogue in a flow-matching policy — which
   is where π0.5 lives, and where Event-Grounded SAEs found single-feature edits barely
   register.
