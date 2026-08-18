# A4 + B1 — implementation plan and status

Two directions, sharing one piece of infrastructure. A4 asks whether independently fine-tuned
models converge on the same *inventory* of causal roles even though their dictionaries do not
align feature-for-feature. B1 asks which of the seven action channels causal breadth actually
lives in. Both run on artifacts already collected.

## Facts established from the code (these shape everything below)

1. **LoRA trains only `q_proj, k_proj, v_proj, o_proj`** (`train_lora.py:580`). The unembedding
   and the final RMSNorm gain are frozen, so they are identical across all four fine-tuned
   models and the 256-bin output space is *literally* shared — cross-model signature comparison
   carries no alignment ambiguity, and every difference between models' signatures comes from
   `W_dec` alone. `action_space_geometry.py` verifies this rather than assuming it.
2. **The same 256 action tokens are reused at all 7 decode positions.** Channel identity is
   positional. A feature's 256-bin signature is therefore semantically ambiguous until
   conditioned on the slot — which is why B1 feeds back into A4.
3. **`C` discards the slot.** `run_attribution.py:301` accumulates over rows using `r % 7` only
   to look up the token, so B1 needs a re-stream, not a re-aggregation.
4. **The rollout ablator projects out the direction** — `h - (h @ V.T) @ V` (`hooks.py:167`) —
   it does not subtract the coded contribution `l2·z_j·w_j` that `phi` describes. These are
   different interventions; both are implemented and the gap between them is a reportable
   number.
5. **The bin axis is reversed.** `token_id = vocab_size - bin_index` while `act_ids` is stored
   ascending, so row 0 of `W_U_act` is the *highest* bin. Any signed reading of the action axis
   must flip it, and the error is silent.
6. **Shards carry `episode` and `timestep`**, so phase- and transition-conditioned analysis
   needs no recollection.

## Steps

| # | File | Status |
|---|---|---|
| 0 | `action_space_geometry.py` | **done**, tested |
| 1 | `mrvla/readout.py` | **done**, tested against brute force |
| 2 | `mrvla/channels.py` + `run_channel_attribution.py` | **done**, tested |
| 3 | `analyze_channels.py` | **done**, tested |
| 4 | `inventory_recurrence.py` | **done**, tested |
| 5 | `inventory_clusters.py` | not started |

### Step 0 — the A4 gate (`action_space_geometry.py`)

Measures the dimension of the space every causal signature lives in: the SVD of the
contrast-centred `W_U_act`, and — when an SAE is passed — of the signature matrix the model's
features *actually* occupy, which is the tighter and more relevant number.

Why it comes first: if the effective rank is small, a random *m*-dimensional subspace already
explains most of any direction, one-to-many matching saturates immediately, and A4's *m*-sweep
cannot separate signal from arithmetic. The script prints the saturation curve and names the
branch.

It decides from a **bracket**, not one number. The energy participation ratio is pessimistic
when the spectrum is heavy-headed; the 99%-energy rank is optimistic. The branch is only called
with confidence when both ends agree, and otherwise the script says so and defers to the
empirical *m*-matched null in step 4, which measures the floor on the real signature
distribution instead of assuming isotropy.

**A low rank here is not a failure — it may be the headline.** If features' causal effects
occupy only ~*r* directions, then "2048 features" is an illusion at the output: there are ~*r*
distinguishable causal roles per decode slot, ~7*r* across the whole action. That is a claim
about VLA readout geometry worth making on its own, and it explains why so many signatures look
alike.

### Step 1 — exact readout counterfactuals (`mrvla/readout.py`)

At layer 31 the readout is the whole remaining computation, so removing a feature can be
evaluated exactly in logit space with **no model forward pass**:

    L_t = (h ⊙ g)·u_t        L'_t = L_t − Σ_k coeff_k · S[k,t]

`r` drops out of the argmax (a positive per-row scalar), and contrast-centring `S` shifts every
bin equally, so flips are invariant to both. The same `S` serves A4's cross-model matching.

Both ablation semantics are implemented — `projection` (matches the rollouts, needs only
`W_dec`, runs without torch) and `coded` (matches `phi`). A margin bound prunes decisions that
provably cannot flip; it is an optimisation only, and a test pins that pruned and unpruned
results are identical. `coalition_coeffs` reproduces the hook's own formula *including* its
over-subtraction on correlated directions, because the point is to model the experiment that
was run; `coalition_overlap` reports how far from orthogonal a coalition is so the distortion
is visible.

**Why this matters beyond B1:** it yields a necessity estimate over ~446k slot-decisions with a
standard error near 0.1%, against the ~9-point minimum detectable effect of the 200-episode
closed-loop ablation. It answers a narrower question — the direct effect on one decode slot,
not task success — but answers it decisively. Scope caveat, stated in the module: in a rollout
the ablation also changes what slot *s+1* attends to and what the next timestep observes, so
this is a lower bound on behavioural impact.

### Step 2 — slot-resolved attribution (`mrvla/channels.py`, `run_channel_attribution.py`)

One streaming pass producing `C_slot [7, G, F]` in absolute and decision-share form, plus exact
per-(feature, slot) flip rates and signed bin shifts for a candidate set drawn from the two ends
of the adjusted-breadth ranking (the same features the ablation and steering runs target).

Three traps handled explicitly:

- **Normalisation.** Raw `|phi|` is not comparable across slots: `phi` carries
  `u_contrast = u_t − mean_s u_s`, whose norm depends on where the emitted bin sits in the
  ordered range. The gripper is near-binary and emits extreme bins, so *every* feature looks
  stronger there for a geometric reason. Share statistics are comparable; both are computed, and
  a result appearing only in the absolute numbers is the confound, not the finding.
- **Degeneracy.** The gripper token is constant for most of an episode, so a feature can score
  high gripper share by dominating a low-entropy slot. `transition_mask` isolates the decisions
  where the command actually changes; every statistic is reported on all decisions and on
  transitions only.
- **Validation.** The recomputed argmax must equal the token the model emitted. That agreement
  rate is printed first and gates the run — below ~1.0, the residuals, head constants, or
  token-id mapping do not line up and nothing downstream is trustworthy.

Also a speedup that is really a correctness guarantee: `run_attribution` computes the alignment
term with a per-row matvec inside a Python loop over all 446k rows, but since
`u_contrast = u_t − mean_s u_s`, that term is exactly column *t* of the contrast-centred
signature matrix. The loop collapses to a column gather. A test proves the two agree against
`mrvla.attribution.attribute`.

### Step 3 — `analyze_channels.py`

Reports, in order: the per-slot sufficiency table (printed first because it gates everything —
if the decomposition recovers 95% of the gripper margin and 70% of yaw, the channels are not
equally trustworthy and a spread above 0.15 says so explicitly); `PR_chan` and its correlation
with task breadth (near zero means channel breadth is a genuinely new axis, strongly positive
means "general" just meant "touches everything" and there is one axis, not two);
`corr(adjusted_breadth, gripper concentration)` rank-residualised on magnitude and base rate
with the same estimator as the Path A headline, plus the same partial for all seven channels so
the story is not gripper-or-nothing; the general-vs-specialist channel profile with a
common-language effect size; and necessity flip rates per group per channel with Wilson
intervals, on all decisions and on gripper-transition decisions only.

Per-slot sufficiency was folded back into step 2 rather than deferred — every term
(`true_c`, `phi_sum`, the `mu + b_pre` constant) collapses to a 256-vector lookup or a row sum
of quantities the pass already computes, so it costs nothing.

**A limit of the absolute-vs-share control, stated rather than glossed.** A perfectly uniform
per-slot rescale is a monotone transform of every feature's channel profile, so it leaves rank
correlations untouched and the comparison will report AGREE however large the factor is. The
split catches *differential* distortion — inflation that depends on which features are active,
which is the realistic form, since `‖u_contrast‖` depends on the emitted bin and which bins are
emitted covaries with which features fire — and it catches every level comparison, where even a
uniform factor moves the numbers. Both behaviours are pinned by test so `AGREE` is not
over-read as "no scale confound possible".

### Step 4 — `inventory_recurrence.py`

The *m*-sweep. For each feature in model A, greedily select *m* features from B whose signatures
best span it (OMP), and sweep *m* = 1, 2, 3, 5, 8. **m=1 is exactly the published `q_causal`**,
so this extends the existing result along a new axis rather than replacing it.

The prediction that makes it worth running: if feature splitting explains the Path B null — one
model's general primitive being fragmented across several features in another — then the general
features' curve rises steeply with *m* while the specialists' stays flat, because specialists
were already atomic at m=1. Diffuse, high-usage general features are exactly the ones expected to
fragment, so splitting alone could have manufactured the entire "generals recur less" result.

Nulls, all *m*-matched (non-negotiable — cos rises with *m* mechanically, and the null must
also match the dictionary SIZE so the best-of-F selection effect is matched): random decoders
through the shared head; the same-model different-seed floor as the ceiling, giving
chance-corrected retention in the `ret_cc` style of §2.4; and the base-model inheritance control.
Reports per-breadth-decile curves and saves signature sharpness so the decile contrast can be
rank-residualised on it (open thread #4) — cosine matching plausibly favours low-entropy
signatures, and specialists have sharper ones.

**Two biases, pointing opposite ways, both stated in the code.** (i) Coefficients are
unrestricted, but TopK codes are non-negative — model B can add `w_j`, never subtract it — so
unrestricted projection *overstates* expressibility. A flat curve under an upper bound is
genuinely flat, so the null conclusion is safe; a rising curve needs the `--positive-only`
follow-up before it is believed. (ii) Greedy matching pursuit is suboptimal: on planted data
where three features provably span the target, greedy alone recovers only ~0.82–0.98 of it, so
the curve *understates* expressibility. That means a flat curve is weaker evidence of absence
than it looks. `n_restarts` narrows the second gap by re-running with each of the top-r first
picks and keeping the elementwise best per *m* — elementwise, not the best whole run, because
m=1 must stay exactly equal to the one-to-one `|max cos|` for comparability. Restarts can only
help, never hurt, but strict improvement is not guaranteed and is not claimed.

**The sign discontinuity.** `run_causal_recurrence.best_match_cosine` takes the *signed* max;
projection onto a span does not, since a span containing `−v` expresses `v`. So m=1 is
`|max cos|`, not the published signed value, and the two differ exactly on features whose best
match is anti-aligned. `anti_aligned_fraction` measures how often that happens and the published
signed q is printed beside m=1, so the size of the discontinuity is reported rather than
assumed.

**A crowding failure worth knowing about, pinned by test.** In a cramped ambient space with a
dictionary large relative to it, greedy matching latches onto spurious directions and m=1 starts
*higher* than it should — a healthy-looking match that is dimension counting rather than
correspondence. This is exactly the regime Step 0 screens for, and it is why the gate runs
first.

### Step 5 — `inventory_clusters.py`

The distributional view, and the primary route if step 0 reports low rank: cluster each model's
signatures on the sphere, match centroids across models by Hungarian on cosine against the same
floors, and compare **occupancy** — how many features per cluster. "Same inventory, different
multiplicities" is a precise form of the claim, and multiplicity differences are what splitting
*is*.

## Open dependencies

- Step 0 needs `head_constants.npz` from the cluster; it cannot be run here. Its output decides
  step 4's design, so run it first and read the printed branch.
- Step 4's ceiling needs a second SAE seed per suite; only `goal` has one today (backlog item in
  `EXPERIMENT_PLAN.md`).
- Step 2 assumes OpenVLA's action ordering `[dx, dy, dz, droll, dpitch, dyaw, gripper]` for
  labelling and for which slot the transition control uses. It is not discoverable from the
  checkpoint — `--gripper-slot` overrides it, and it is used for labelling only, never arithmetic.
