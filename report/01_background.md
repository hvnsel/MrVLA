# Chapter 1 — The machines and the vocabulary

## 1.1 What problem is the robot solving?

Imagine a robot arm above a table. On the table are objects — a bowl, a plate, a cabinet with a drawer. The robot has a camera looking at the scene, and it is given an instruction in plain English, for example *"put the bowl on the plate"*.

Roughly twenty times a second, the robot must decide what to do next. Its decision is not "pick up the bowl" — that is far too abstract for a motor. Its decision is seven numbers:

$$
  dx      how far to move the gripper left/right       (a small distance)
  dy      how far to move it forward/backward          (a small distance)
  dz      how far to move it up/down                   (a small distance)
  droll   how far to rotate about one axis             (a small angle)
  dpitch  how far to rotate about a second axis        (a small angle)
  dyaw    how far to rotate about a third axis         (a small angle)
  grip    whether the gripper should be open or closed
$$

Those seven numbers are called an **action**. The prefix *d* means "delta", i.e. a small change from where the arm is now, not an absolute position. Together the first six describe how to nudge the hand in three-dimensional space, and the seventh controls the fingers. This is called a **7-DoF action** ("degrees of freedom").

A **policy** is any function that takes the current situation and returns an action. Formally, if `o` is what the robot observes and `a` is the action:

$$
  a = pi(o)          "pi" is the traditional symbol for a policy
$$

Run the policy repeatedly — observe, act, observe, act — and you get an **episode**: a whole attempt at the task, which ends either in success (the bowl is on the plate) or failure (a time limit is reached, or something is knocked over).

## 1.2 What is a Vision-Language-Action model?

A **Vision-Language-Action model** (VLA) is a policy built out of a large language model. The idea sounds strange at first — language models predict text, they do not move robot arms — but the trick is simple: *turn the action into text*.

Concretely, in the model we study (**OpenVLA-7B**):

1. The camera image is passed through a vision encoder, producing a sequence of vectors that stand in for "what the scene looks like".
2. The instruction is turned into word-pieces (**tokens**) in the usual way for a language model.
3. Both are fed into a large transformer language model — in this case one built on Llama-2 with about 7 **billion** adjustable numbers (7B **parameters**).
4. The model then *writes out* the action, one token at a time, as if it were writing seven words.

The critical design choice is step 4. Each of the seven action numbers is **discretised**: the continuous range of possible values is chopped into **256 equal buckets**, called **bins**. Instead of outputting the number 0.037, the model outputs "bin 148". And because a language model can only emit tokens from its vocabulary, each of the 256 bins is permanently assigned to one token of the tokenizer's vocabulary — specifically the 256 least-used ones, which the model would otherwise almost never emit.

So producing one action means producing **seven tokens**, one per degree of freedom, generated one after another (**autoregressively** — each token is produced with the previous ones already visible to the model).

> **Why this matters for everything that follows.** The same 256 tokens are reused for all seven slots. There is no separate vocabulary for "up/down" versus "gripper". Which physical quantity a token refers to is determined **only by its position** in the sequence of seven. Token 148 in the first slot means a particular left/right movement; the identical token in the seventh slot means a particular gripper state. We will return to this repeatedly: *channel identity is positional*.

One more detail that will matter later. The mapping from bin to token id runs **backwards**:

$$
  token_id = vocab_size - bin_index          bin_index runs 1, 2, ..., 256
$$

so bin 1 gets the *highest* token id and bin 256 the lowest. Any analysis that treats the bin axis as ordered — for example "did this feature push the action up or down?" — has to undo that reversal, and if it does not, the error is silent.

## 1.3 LIBERO: the tasks

**LIBERO** is a standard benchmark of simulated tabletop manipulation tasks. It is divided into four **suites**, each containing **10 tasks**:

| Suite | What varies between its tasks | Example flavour |
|---|---|---|
| `spatial` | The *layout*: same objects, different positions | "pick up the black bowl between the plate and the ramekin" |
| `object` | The *objects*: same layout, different things to manipulate | "pick up the alphabet soup and place it in the basket" |
| `goal` | The *goal*: same scene, different requested outcome | "open the middle drawer of the cabinet" |
| `10` (also called LIBERO-Long) | Everything — ten long, multi-step tasks | "put both the cream cheese box and the butter in the basket" |

The authors of OpenVLA released four separately fine-tuned checkpoints, one per suite. **All four start from the same base OpenVLA model** and are then trained further on their own suite. This shared ancestry is important and we will come back to it: these four models are not independent, they are siblings.

A **rollout** is one run of the policy on one task from one starting configuration, producing an episode that succeeds or fails. Rollouts are how we test whether an intervention on the model changes its behaviour.

## 1.4 Inside the model: the residual stream

To interpret a model you have to look inside it. Here is the minimum you need.

A transformer processes a sequence of tokens through a stack of **layers** (OpenVLA has 32, numbered 0 to 31). At every layer, and for every position in the sequence, the model holds a vector of numbers. In our model that vector has **d = 4096** components. This running vector is called the **residual stream**, and we write it `h`.

The name comes from how layers work: each layer does not *replace* `h`, it *adds* to it.

$$
  h_after_layer = h_before_layer + (what this layer computed)
$$

So the residual stream is like a shared blackboard that every layer writes onto. By the time you reach the final layer, `h` contains everything the model has worked out and is about to use to choose its next token.

**Every experiment in this project reads `h` at layer 31 — the last layer — at the exact positions where the seven action tokens are being produced.** That specific choice is the single most important methodological decision in the project, and §4.3 explains why an earlier choice had to be abandoned.

## 1.5 How the model turns `h` into a chosen action bin

This is the step that makes the whole causal analysis possible, so we do it slowly.

At the end of the network, the residual vector `h` is converted into a **score for every token in the vocabulary**. Those scores are called **logits**. The token with the highest logit is the one emitted (this is called taking the **argmax** — "the argument that maximises").

Two operations happen. First a normalisation called **RMSNorm** ("root mean square normalisation"):

$$
  r        = sqrt( mean(h_i^2) + eps )        a single number: the "size" of h
  RMSNorm(h) = (h / r) * g                    divide by that size, then scale
$$

Here `eps` is a tiny constant (about 0.00001) that stops division by zero, and `g` is a fixed vector of 4096 learned numbers called the **gain**, applied component by component. In words: *shrink or stretch `h` so its typical component size is 1, then rescale each component by a learned amount*. The purpose is numerical stability during training.

Second, the **unembedding**. There is a big matrix `W_U` with one row per vocabulary token. The logit for token `t` is the **dot product** of the normalised residual with row `t`:

$$
  logit(t) = RMSNorm(h) . u_t          where u_t is row t of W_U
$$

If you have not met the dot product, it is defined in §3.1; for now, read `x . y` as "a single number measuring how much `x` and `y` point in the same direction".

We only ever care about the 256 action-token rows, which we collect into a matrix `W_U_act` of shape [256 x 4096].

> **The key structural fact.** The last operation the model performs is a **dot product**, and a dot product is **additive**: `(a + b) . u = a.u + b.u`. So if we can write `h` as a sum of pieces, we can write the logit as a sum of contributions, one per piece — and each contribution is a number we can compute exactly. This is why layer 31 is special. At any earlier layer, `h` still has to pass through many nonlinear operations before it becomes a logit, and no exact decomposition exists.

## 1.6 The problem interpretability is trying to solve

You might hope that the 4096 components of `h` each mean something — component 17 means "there is a bowl", component 892 means "the gripper is closed". They do not. Networks routinely store far more distinct concepts than they have components, by giving each concept its own **direction** in the 4096-dimensional space rather than its own axis. Directions can be packed in much more densely than axes, at the cost of slight interference between them. This phenomenon is called **superposition**.

So the concepts are there, but they are smeared across all 4096 numbers. Interpretability needs a way to un-smear them.

## 1.7 Sparse autoencoders

A **sparse autoencoder** (SAE) is the standard tool for this. The idea:

1. **Learn a dictionary.** Find a large set of directions in the 4096-dimensional space — call them `w_1, w_2, ..., w_F` — such that any real residual vector `h` from the model can be written as a sum of just a **few** of them.
2. **Encode.** Given `h`, compute a number `z_j` for each dictionary direction saying how strongly that direction is present. The vector `z = (z_1, ..., z_F)` is called the **code**, and its entries are called **feature activations**.
3. **Decode.** Rebuild an approximation of `h` by adding the directions back up, weighted by the code.

The autoencoder part means it is trained to reproduce its own input (`h` in, approximately `h` out). The **sparse** part is the whole point: we force most of the `z_j` to be exactly zero, so that only a handful of directions are used to explain any given `h`. That is what makes the pieces interpretable — a description in terms of 100 active items is legible, one in terms of 4096 is not.

In this project:

- `F = 2048` dictionary directions ("features"). Note this is *fewer* than the 4096 dimensions of `h` — an unusual choice inherited from the reference paper's recipe, so this dictionary is compressive rather than expansive.
- Sparsity is enforced by **TopK with k = 100**: compute a score for all 2048 features, keep the 100 largest, set the other 1948 to exactly zero. Simple and gives exact control over sparsity.
- The dictionary directions are stored as rows of a matrix `W_dec` of shape [2048 x 4096], each row scaled to have length 1 (**unit norm**).

The exact reconstruction the SAE performs is:

$$
  h  ~=  l2 * ( sum over j of  z_j * w_j )  +  mu * 1  +  b_pre
$$

with these pieces:

| Symbol | Shape | Meaning |
|---|---|---|
| `z_j` | number | how strongly feature `j` is active on this input (zero for all but 100 features) |
| `w_j` | 4096 numbers | the direction feature `j` writes into the residual stream (row `j` of `W_dec`) |
| `l2` | number | a per-input scale factor; this SAE normalises each input before encoding, and this undoes it |
| `mu` | number | the mean of the input's components, added back to every component (`1` means a vector of all ones) |
| `b_pre` | 4096 numbers | a fixed learned offset, the same for every input |

The terms `mu * 1 + b_pre` together form a **constant** part of the reconstruction that does not depend on which features fired. Remember it — it will turn out to matter enormously in Chapter 5.

## 1.8 What we mean by "a feature"

Throughout this document, **feature `j`** means: *the pair (direction `w_j`, activation `z_j`)*. Talking about "feature 1167" means index 1167 in this particular trained dictionary.

Three warnings about that, all of which the project has had to take seriously:

1. **Features are not neurons.** They are learned directions, not components of `h`.
2. **Feature indices are dictionary-specific.** Feature 1167 in the `goal` model's SAE has no relationship to feature 1167 in the `spatial` model's SAE. Comparing across models requires a matching procedure, and the design of that procedure is a whole research question (Chapter 4, §4.9 and §5.11).
3. **Feature identity is not fully reproducible.** Train the same SAE again with a different random starting point and you get a *different* dictionary that explains the same data roughly as well. Quantifying that instability is one of the project's findings (§4.10).
