# Chapter 3 — The mathematical toolkit

Every tool used anywhere in the project, built from the ground up. If you already know a section, skip it; the experiments in Chapters 4–5 refer back by number.

## 3.1 Vectors, dot products, length, and cosine

A **vector** is just an ordered list of numbers. The residual stream `h` is a vector with 4096 entries. We write `h_i` for the `i`-th entry.

The **dot product** of two vectors of the same length multiplies them entry by entry and adds up the results:

$$
  x . y  =  x_1*y_1 + x_2*y_2 + ... + x_n*y_n  =  sum over i of  x_i * y_i
$$

The **length** (or **norm**) of a vector is the square root of its dot product with itself — the multi-dimensional Pythagoras theorem:

$$
  ||x||  =  sqrt( x . x )  =  sqrt( x_1^2 + x_2^2 + ... + x_n^2 )
$$

A vector with length 1 is called a **unit vector**. **Normalising** a vector means dividing it by its own length, which keeps its direction and sets its length to 1.

The dot product has a geometric meaning: if `theta` is the angle between two vectors,

$$
  x . y  =  ||x|| * ||y|| * cos(theta)
$$

Rearranging gives the single most-used quantity in this project, **cosine similarity**:

$$
  cos(x, y)  =  ( x . y ) / ( ||x|| * ||y|| )
$$

Read it as **"how aligned are these two directions, ignoring how long they are?"**

| Value | Meaning |
|---|---|
| `+1` | identical direction |
| `0` | perpendicular — completely unrelated directions |
| `-1` | exactly opposite directions |

Two facts we lean on. First, if both vectors are already unit length, cosine similarity is *just the dot product*, so computing all pairwise cosines between two sets of vectors is one matrix multiplication. Second, in a high-dimensional space two *random* directions are almost always nearly perpendicular — the typical cosine between random vectors in `n` dimensions is about `1/sqrt(n)`, which for n = 256 is about 0.06. **So a cosine of 0.3 between two 256-dimensional vectors is not "weak"; it is far above what chance produces.** This is why every cosine in this report is quoted against a measured chance floor rather than judged on intuition.

## 3.2 Means, centring, and why we subtract things

The **mean** of a list is its average. **Centring** a list means subtracting its mean so the new list averages to zero.

Centring appears twice in this project and both times for the same reason: **to remove a component that carries no information about a choice.**

Suppose the model must choose between 256 action bins by picking the largest logit. Now suppose something adds the *same amount* to all 256 logits. The ranking is unchanged, so the choice is unchanged. That common component is irrelevant to the decision and should not be credited to anything.

So whenever we ask "how does this feature affect the choice?", we first subtract the average effect across all 256 bins. The resulting direction is called the **contrast direction**:

$$
  u_contrast(t)  =  u_t  -  (average of u_s over all 256 action tokens s)
$$

A feature that lifts every action bin equally has zero contrast effect and correctly scores zero.

## 3.3 Correlation, ranks, and Spearman

**Correlation** measures whether two lists of numbers move together. The standard version (**Pearson**) is the cosine similarity of the two lists after centring:

$$
  r  =  sum_i (x_i - xbar)(y_i - ybar)
        / sqrt( sum_i (x_i - xbar)^2 * sum_i (y_i - ybar)^2 )
$$

It runs from −1 (perfectly opposed) through 0 (unrelated) to +1 (perfectly aligned).

Pearson correlation assumes the relationship is a straight line and is easily distorted by a few extreme values. Our data has both problems — feature magnitudes are heavy-tailed, with a handful of features enormously larger than the rest. So we mostly use **Spearman correlation**, which is Pearson correlation applied to **ranks** instead of raw values.

**Ranks**: replace the smallest value by 1, the next by 2, and so on. A list of `(3.1, 900.0, 5.2)` becomes `(1, 3, 2)`. This throws away *how much* bigger something is and keeps only the ordering, which makes the measure immune to outliers and to any relationship that is increasing but curved.

> **A subtlety that bit us.** When two values are exactly equal ("tied"), textbook Spearman gives them the **average** of the ranks they would occupy. A common shortcut — sorting twice — instead breaks ties by whichever came first in the array, which injects an arbitrary ordering. That distinction is invisible on continuous data and very visible on counts. It is a live issue in this project; see §8.3.

## 3.4 Partial correlation: holding a variable fixed

This is the workhorse of the confound-first principle, so it gets a full explanation.

**The question.** We find that breadth correlates with importance. But busy features might score high on both simply *because* they are busy. We want: **does breadth still predict importance among features that fire equally often?**

Running the analysis only on features with identical base rates would throw away nearly all the data. The standard alternative is **residualisation**.

**The recipe**, for a control variable `c`:

1. Fit the best straight line predicting `x` from `c`. The **residual** `x_res` is what is left over: `x_res = x − (line's prediction from c)`. By construction `x_res` contains no linear information about `c`.
2. Do the same for `y`, giving `y_res`.
3. Correlate `x_res` with `y_res`.

The result is the **partial correlation of x and y controlling for c**. Read it as: *the relationship between x and y that is not explainable by c*.

We use the rank version throughout: rank all three variables first, then residualise. With two controls (`c1` and `c2`) the straight line becomes a plane, fitted by least squares, and the interpretation extends unchanged. In our code that is `rank_partial_both(y, x, c1, c2)`, and the two controls are always **causal magnitude** and **base firing rate**.

> **Why exactly those two controls?** Magnitude, because "this feature matters a lot in total" is a boring reason to look general. Base rate, because "this feature fires constantly" is the confound that destroyed the previous metric family. If a breadth score survives both, it is telling us something about *how influence is distributed across tasks* rather than about how strong or how frequent the feature is.

## 3.5 The participation ratio: turning a shape into a count

Suppose a feature has some amount of causal influence on each of 10 tasks — a list of 10 non-negative numbers. We want one number summarising **how many tasks it really spreads over**. Counting non-zeros is too crude: a feature with 9.99 units on task 1 and 0.001 on the others would count as 10.

The **participation ratio** solves this:

$$
  PR(x)  =  ( sum_i x_i )^2  /  ( sum_i x_i^2 )
$$

Work through the two extremes:

- All the mass on one task, `x = (5, 0, 0, ..., 0)`: numerator `25`, denominator `25`, so **PR = 1**.
- Mass spread evenly over 10 tasks, `x = (5, 5, ..., 5)`: numerator `50^2 = 2500`, denominator `10 * 25 = 250`, so **PR = 10**.

So PR runs from 1 (everything in one place) to `n` (perfectly even), and reads directly as **an effective count**.

**Why the formula works.** Normalise to proportions `p_i = x_i / sum(x)`. Then `PR = 1 / sum(p_i^2)`. The quantity `sum(p_i^2)` is the probability that two independent random draws land on the *same* item. If mass is concentrated, collisions are likely and PR is small; if spread out, collisions are rare and PR is large. PR is the reciprocal of a collision probability, which is exactly what "effective number of items" should mean.

**PR is scale-free**: multiplying every entry by 100 leaves PR unchanged. That is essential — it measures *breadth*, not *strength*, which is what lets us separate the two.

We use PR in four different roles, and it is worth keeping them straight:

| Applied over | Gives | Where |
|---|---|---|
| tasks, per feature | effective number of tasks a feature drives = **breadth** | §4.7 |
| features, per model | effective number of features carrying the action = **concentration** | §5.3 |
| action channels, per feature | effective number of the 7 action dimensions a feature drives | §5.9 |
| tasks, per ablation condition | effective number of tasks a removal damages | §5.5 |
| singular values | **effective rank** of a matrix | §3.13 |

## 3.6 Null distributions and permutation tests

**The question a null answers.** You measured +0.49. Would you have got +0.49 from data with no real structure?

**The method.** Take your real data and scramble it in a way that destroys the effect you claim while leaving everything else intact. Recompute the statistic. Do this a thousand times. You now have a **null distribution**: the range of values the statistic takes when the claimed effect is absent. Compare your real number to it.

If your value lands far outside, the effect is real. The **p-value** is simply the fraction of scrambled runs that reached your value or beyond.

**Everything depends on choosing the right scramble.** A permutation that destroys too much makes any result look significant; one that destroys too little makes a real result look like nothing. The project made *both* mistakes at different times and caught both:

- An early cross-model null permuted the 256 action-bin axes. That destroyed not only feature correspondence but the *shared geometry of the readout itself*, deflating the floor and manufacturing a fake gap of about 0.33 (§4.11).
- The plan's prescribed control for Path A permuted task labels within a feature. That destroyed **nothing the statistic depends on**, and reproduced the real value almost exactly (§5.2).

> **The rule of thumb.** A good null destroys exactly the correspondence you are claiming, and preserves every other structural property — marginal distributions, matrix shapes, the number of things being maximised over.

## 3.7 Confidence intervals, and the Wilson interval

If you observe 152 successes out of 200, the success rate is 0.76 — but the *true* rate could be somewhat different by luck. A **95% confidence interval** is a range constructed so that, over many repetitions of the experiment, the true value falls inside it 95% of the time.

The textbook interval `p ± 1.96 * sqrt(p(1-p)/n)` fails badly near 0 or 1. At 20 successes out of 20 it gives `1.00 ± 0.00` — claiming you know the rate perfectly from 20 trials, which is absurd.

The **Wilson score interval** fixes this by solving for which true rates are consistent with the observation, rather than assuming a symmetric spread around the estimate:

$$
  centre = p + z^2/(2n)
  margin = z * sqrt( p(1-p)/n + z^2/(4n^2) )
  interval = ( centre - margin , centre + margin ) / ( 1 + z^2/n )
$$

with `z = 1.96` for 95%. At 20/20 it gives roughly [0.84, 1.00] — asymmetric, and honest.

## 3.8 Paired data, McNemar's test, and statistical power

### Why pairing matters

Suppose you test whether removing a feature hurts the robot. You run 200 episodes normally and 200 with the feature removed, and compare success rates. But episodes differ enormously in difficulty — some starting positions are nearly impossible. Much of the difference between your two numbers is *which starting positions each condition happened to get*.

Unless you make them the same. Our ablation runs **replay the identical starting configurations** under every condition. Episode 7 of task 3 has the same initial state in the baseline and in every ablated condition. That makes the data **paired**, and pairing removes the episode-difficulty variation entirely — a large gain in sensitivity for free.

### McNemar's test

With paired binary outcomes, every pair falls into one of four boxes:

| | ablated succeeded | ablated failed |
|---|---|---|
| **baseline succeeded** | both fine | `b10` — **damage** |
| **baseline failed** | `b01` — **repair** | both failed |

The two diagonal boxes are **concordant**: both conditions did the same thing, so they say nothing about which is better. Only the **discordant** counts `b10` and `b01` carry information.

Under the null hypothesis that the ablation has no effect, each discordant pair is equally likely to fall either way — a coin flip. So with `m = b10 + b01` discordant pairs, `b10` should follow a **binomial distribution** with `m` trials and probability 0.5. **McNemar's test** is just asking how unusual the observed split is under that coin-flip model. For small `m` we compute this exactly; the common chi-squared approximation is over-conservative below about 25 discordant pairs and would throw away real power.

The **damage** and its confidence interval:

$$
  d  = ( b10 - b01 ) / n
  SE = sqrt( (b01 + b10) - (b10 - b01)^2 / n ) / n
  95% interval = d +- 1.96 * SE
$$

Note that concordant pairs contribute nothing to the uncertainty either — the whole test lives on the discordant pairs.

### Statistical power and the minimum detectable effect

**This is the concept that turned one of our uninterpretable results into a reportable one, so it deserves care.**

A null result — "we saw no effect" — has two completely different possible causes:

1. There is no effect.
2. There is an effect, but the experiment was too small to see it.

You cannot tell these apart from the result alone. What distinguishes them is **power**: the probability that the experiment would detect an effect of a given size if that effect were real. Conventionally we ask for 80% power.

Turning this around gives the **minimum detectable effect** (MDE): the smallest effect the design would catch 80% of the time. Anything smaller than the MDE is *below the resolution of the experiment*, and observing nothing there says nothing at all.

For a paired binary test, the derivation runs like this. Conditional on `m` discordant pairs, we are testing whether a coin is fair. The test rejects when

$$
  |p - 0.5| * sqrt(m)  >  z_alpha * 0.5  +  z_power * sqrt( p(1-p) )
$$

where `p` is the true probability a discordant pair goes the damage way, `z_alpha = 1.96` and `z_power = 0.84`. Writing the damage as `d = (2p − 1) * disc_rate`, where `disc_rate = m/n` is the fraction of pairs that are discordant, and taking the worst case `p ≈ 0.5` in the variance term gives the clean form:

$$
  MDE  ~=  ( z_alpha + z_power ) * sqrt( disc_rate / n )
       ~=  2.80 * sqrt( disc_rate / n )
$$

**Worked example from our data.** With `n = 200` pairs and a measured discordance rate of 0.253:

$$
  MDE = 2.80 * sqrt(0.253 / 200) = 2.80 * 0.0356 = 0.0996  ->  about 10 percentage points
$$

So a 200-episode design detects a 10-point drop and nothing smaller. Rearranging for the sample size needed to detect a target effect `d`:

$$
  n  =  disc_rate * ( ( z_alpha + z_power ) / d )^2
$$

For a 5-point effect at the same discordance rate: `n = 0.253 * (2.80/0.05)^2 = 793` pairs, i.e. **79 episodes per task** across 10 tasks instead of 20.

## 3.9 Reliability, split-half, Spearman–Brown, and attenuation

### Reliability

**Reliability** is the extent to which a measurement agrees with *itself* when repeated. It runs from 0 (pure noise) to 1 (perfectly repeatable). It is not the same as validity: a broken thermometer that always reads 20°C is perfectly reliable and completely invalid.

### Split-half

To measure the reliability of a score computed from 10 tasks, split the tasks into two disjoint halves, compute the score **independently on each half**, and correlate the two versions across features. Repeat over many random splits.

Two details that matter. **Everything derived from the tasks must be recomputed per half** — if you reuse a quantity computed on all 10 tasks, the held-out half leaks in and the reliability is inflated. And a **floor** is needed: shuffling feature identity in one half should give ≈0, confirming the procedure is calibrated.

### Spearman–Brown

A half-length measurement is *less* reliable than the full one, so the raw split-half correlation understates the truth. The **Spearman–Brown** formula corrects for length:

$$
  r_full  =  2 * rho  /  ( 1 + rho )
$$

where `rho` is the observed half-to-half correlation. Example: `rho = 0.222` gives `r_full = 0.444/1.222 = 0.363`. The formula is meaningless for `rho <= 0`, which we treat as "no reliable signal".

### Attenuation: the ceiling on any correlation

Here is the fact that makes reliability essential rather than decorative. **Noisy measurements cannot correlate strongly, even when the underlying truth is a perfect relationship.** Formally, if `x` and `y` have reliabilities `r_xx` and `r_yy`, then the correlation you observe relates to the true one by

$$
  r_observed  =  r_true  *  sqrt( r_xx * r_yy )
$$

Since `r_true` can be at most 1, this gives a hard **ceiling**:

$$
  | r_observed |  <=  sqrt( r_xx * r_yy )
$$

Two consequences, and **they point in opposite directions**, which is the part that is easy to get wrong:

- **For a positive result**, the correction is free and helps you. Attenuation only ever *shrinks* a correlation, so dividing by the ceiling gives a **lower bound** on the truth. If you measured +0.49 with a breadth reliability of 0.363, then `|r_true| >= 0.49/sqrt(0.363) = 0.82`. The true relationship is *at least* that strong.
- **For a null result**, the correction does **not** help you, and assuming otherwise is a real error. To claim a null you need an **upper** bound on `|r_true|`, and the formula gives you a lower one. Substituting `r_yy = 1` for an unknown reliability produces `|r_true| >= |r_obs|/sqrt(r_xx)` — an argument *against* the null. Defending a null requires knowing that the **ceiling was high**: the measurement had room to show a big correlation and returned a small one.

When one reliability is unknown, the honest move is to invert the question and report the **breakeven reliability** — how reliable the other measure would have to be for the observed value to bound the truth below some threshold:

$$
  require   |r_true| = |r_obs| / sqrt(r_xx * r_yy)  <=  threshold
  rearrange r_yy  >=  r_obs^2 / ( r_xx * threshold^2 )
$$

## 3.10 The bootstrap

To attach uncertainty to a statistic with no neat formula, **resample the data with replacement** many times, recompute the statistic on each resample, and read off the middle 95% of the results.

**Which unit you resample defines what uncertainty you are describing.** If you resample tasks, you are asking "what if I had drawn a different set of tasks?" If you resample episodes, you are asking "what if the same tasks had gone differently by luck?"

If both sources of variation are real, you need a **two-level bootstrap**: resample tasks, and then within each drawn task resample its episodes, recomputing the per-task quantity from the resampled episodes. Omitting the second level treats a noisy per-task number as if it were known exactly, and produces intervals that are far too confident.

> A counterintuitive consequence, verified on synthetic data: propagating extra noise does **not** always widen the interval. Extra noise in `x` pulls `corr(x, y)` toward zero, which concentrates the bootstrap replicates near zero. The two-level interval can be *narrower* while being centred nearer zero. **Coverage of zero, not width, is the property to check.**

## 3.11 Concentration: Lorenz shares and the Gini coefficient

To describe how unequally a total is split among many items, two standard measures:

**Top-N share** — the fraction of the total held by the N largest items. "The top 50 of 2048 features carry 45% of the causal mass."

**Gini coefficient** — a single summary of inequality from 0 (everything perfectly equal) to 1 (one item has everything). Sorting the values ascending as `v_1 <= ... <= v_n`:

$$
  Gini  =  2 * ( sum_i  i * v_i ) / ( n * sum_i v_i )  -  (n + 1)/n
$$

We report Gini, top-N shares and PR-based effective counts together because each is misleading alone.

## 3.12 Singular values and effective rank

A matrix maps vectors to vectors. The **singular value decomposition** (SVD) finds a set of perpendicular directions such that the matrix acts on each by simple stretching. The stretch factors are the **singular values** `s_1 >= s_2 >= ...`.

If a matrix has 256 rows but its singular values collapse to near zero after the tenth, then although the rows live in a 4096-dimensional space they really only span about ten directions. That "really only spans about" number is the **effective rank**, and we measure it with the participation ratio applied to the *energies* (squared singular values):

$$
  effective_rank  =  ( sum_i s_i^2 )^2  /  ( sum_i s_i^4 )
$$

We also report the **rank at 90% and 99% of energy**: the smallest number of directions needed to account for that share. The two disagree when the spectrum is "heavy-headed" — a couple of dominant directions drag the participation ratio down while many small directions still carry real structure. When they disagree, neither should be trusted alone.

> **Why we care.** Cross-model matching compares directions in a shared space. If that space is effectively 8-dimensional, then *any* handful of directions spans nearly all of it, "matching" becomes trivially easy, and a positive result would be dimension-counting rather than a finding. Measuring the effective rank before running the experiment tells you whether the experiment can mean anything.

## 3.13 Matching algorithms

Three related problems appear in Path B and its extensions.

**Greedy best-match.** For each item in set A, take its single best partner in B. Simple, and what the original recurrence metric does. Its weakness: several A-items can grab the *same* popular B-item, which inflates apparent agreement.

**Hungarian assignment.** Find the one-to-one pairing of A-items to B-items that maximises the *total* similarity, with no reuse. This is a classical optimisation problem with an exact polynomial-time solution (the shortest-augmenting-path algorithm). It is stricter than greedy, and the gap between the two is itself informative: a big gap means a few popular partners were absorbing many matches.

**Orthogonal matching pursuit (OMP).** Sometimes the right question is not "which single B-item matches this A-item?" but "can a *combination* of B-items reproduce it?" OMP answers this greedily:

1. Start with the target direction as the current residual.
2. Pick the B-item most correlated with the residual.
3. Project the target onto the space spanned by everything picked so far; subtract to get a new residual.
4. Repeat `m` times.

After `m` steps you have the cosine between the target and its best approximation from `m` chosen items. Sweeping `m` from 1 upward asks *how much a coalition helps*. Two properties to keep in mind:

- **Greedy is not optimal.** A locally best first pick can lead away from the best group of `m`. So the curve is a **lower bound** on what a coalition could achieve. We mitigate this with restarts (force the first pick to be each of the top few candidates and keep the best result).
- **Cosine rises with `m` automatically.** More directions span more of anything. So a rising curve proves nothing unless compared against a null run at **the same `m`** and with the same dictionary size.

## 3.14 Clustering directions on a sphere

**k-means** partitions points into `k` groups, each represented by its centre, by alternating: assign every point to its nearest centre; recompute each centre as the mean of its members; repeat.

Our points are *directions* — a feature's causal signature — where length means strength, not identity. So we normalise every point to unit length and use cosine similarity instead of straight-line distance; centres are re-normalised each round. This is **spherical k-means**. Empty clusters are re-seeded rather than dropped, so that `k` really means `k` and cluster counts remain comparable between models.

## 3.15 Comparing distributions without clustering: sliced Wasserstein

Clustering forces a choice of `k`, and a conclusion that changes with `k` is a conclusion about `k`. As a cross-check we compare two clouds of directions with no clustering at all.

The **Wasserstein distance** (or "earth mover's distance") between two distributions is the minimum work needed to reshape one into the other. In one dimension it has a simple form: sort both samples and compare them at matched quantiles. In many dimensions it is expensive — so the **sliced** version projects both clouds onto many random one-dimensional directions, computes the easy 1-D distance on each, and averages:

$$
  SW(A, B)  =  average over random directions p of  W_1( A projected on p , B projected on p )
$$

Smaller means the two clouds are distributed more alike. Like every other similarity in this report, it is only interpretable against a floor and a ceiling.
