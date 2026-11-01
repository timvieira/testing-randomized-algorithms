title: Ideas for Testing Randomized Algorithms
date: 2026-04-02
comments: true
tags: testing, sampling, algorithms, monte-carlo, program-tracing

<style>
/* Subtle scrollbars */
* {
    scrollbar-width: thin;
    scrollbar-color: #ccc transparent;
}
*::-webkit-scrollbar {
    width: 6px;
    height: 6px;
}
*::-webkit-scrollbar-track {
    background: transparent;
}
*::-webkit-scrollbar-thumb {
    background: #ccc;
    border-radius: 3px;
}
*::-webkit-scrollbar-thumb:hover {
    background: #aaa;
}
/* Let wide equations scroll on their own instead of widening the page */
mjx-container[display="true"] {
    overflow-x: auto;
    overflow-y: hidden;
    max-width: 100%;
}
/* Keep MathJax's hidden screen-reader copy from widening the page */
mjx-assistive-mml {
    right: 0 !important;
}
</style>

<div style="background: #fff3cd; border-left: 4px solid #ffc107; padding: 12px 16px; margin-bottom: 1.5em; border-radius: 4px; font-size: 0.95em;">
<b>Draft</b> — This article is a work in progress. Content may change.
</div>

Randomized algorithms are hard to test. Their outputs are, by design,
unpredictable&mdash;and that unpredictability is a *feature*, not a bug. It is what
gives them useful guarantees like unbiasedness and computational efficiency. But
it also makes them slippery: how do you check that a stochastic program is
correct when it gives different answers every time you run it?

This post sketches some ideas for how to think about testing randomized
algorithms&mdash;especially the kinds of algorithms that we use for probabilistic
inference. I'll present a concrete sampling algorithm with several plausible
implementations&mdash;some correct, some subtly broken&mdash;and develop
progressively more powerful testing strategies to tell them apart. Each
strategy catches a bug that the previous one missed.

The algorithm is a small building block. Its job is to propose the next token
inside sequential Monte Carlo (SMC) for constrained generation from language
models. Testing the block in isolation is only half the job, so at the end
we'll plug it into SMC and test the whole pipeline.

## A Tricky Little Sampling Algorithm

Suppose we have a distribution $p$ over a finite set $\mathcal{X} = \{1,
\ldots, N\}$ and a condition $C \subseteq \mathcal{X}$ (the set of "good"
elements). We want to sample from the *conditional* distribution $p(\cdot \mid
C)$. But rather than returning a plain sample, we want a *properly weighted*
sample $(x, w)$&mdash;a sample-weight pair that can be used for unbiased estimation.

The following algorithm, which we call AWRS (from [Lipkin et al., 2025](https://arxiv.org/abs/2504.05410)), does exactly this. Here is the pseudocode from the paper:

![AWRS algorithm pseudocode](../images/awrs-algorithm.png)

The idea: run rejection sampling twice. The first pass finds a conditioned
sample $x^*$ while tracking the total rejected probability mass $\psi$. The
second pass generates additional rejections to get a count $n$. The weight
$(1 - \psi) / (n + 1)$ combines these two signals&mdash;the rejected mass and
the rejection count&mdash;into an unbiased estimate of the normalizing constant
$Z$. The number of rejections before finding a valid token contains signal about
the magnitude of $Z$, and weighting by the actual probability mass removed
(rather than just counting) reduces variance.

### Four Implementations, at Least One Bug

Did we implement it correctly? That's the question we'd like our tests to
answer. To make the exercise concrete, here are four Python implementations.
They all look reasonable, but at least one has a bug. Which ones are correct?

Each variant takes a `draw` parameter that defaults to sampling from a
categorical distribution. This is a hook we'll exploit later: in production it
calls the RNG, but in testing we'll swap it for something that lets us
enumerate every possible execution.

<details class="derivation" markdown="1">
<summary>Variant A</summary>

```python
def awrs_a(p, cond, draw=sample_categorical):
    p_cond = p.copy()
    n, psi = 0, 0
    xc0 = None
    for i in range(2):
        while True:
            q = p_cond / p_cond.sum()
            x = draw(q, context=('awrs', i, xc0))
            if cond(x):
                if i == 0:
                    xc0 = x
                break
            else:
                if i == 0:
                    psi += p_cond[x]
                p_cond[x] = 0
            n += 1
    return xc0, (1 - psi) / (n + 1)
```

</details>

<details class="derivation" markdown="1">
<summary>Variant B</summary>

```python
def awrs_b(p, cond, draw=sample_categorical):
    p_cond = p.copy()
    n, psi = 0, 0
    xc0 = None
    for i in range(2):
        while True:
            q = p_cond / p_cond.sum()
            x = draw(q, context=('awrs', i, xc0))
            if cond(x):
                if i == 0:
                    xc0 = x
                break
            else:
                if i == 0:
                    psi += p_cond[x]
                p_cond[x] = 0
            n += 1
    return xc0, (1 + psi) / (n + 1)
```

</details>

<details class="derivation" markdown="1">
<summary>Variant C</summary>

```python
def awrs_c(p, cond, draw=sample_categorical):
    p_cond = p.copy()
    n, psi = 0, 0
    xc0 = None
    for i in range(2):
        while True:
            q = p_cond / p_cond.sum()
            x = draw(q, context=('awrs', i, xc0))
            if cond(x):
                if i == 0:
                    xc0 = x
                break
            else:
                psi += p_cond[x]
                p_cond[x] = 0
            n += 1
    return xc0, (1 - psi) / (n + 1)
```

</details>

<details class="derivation" markdown="1">
<summary>Variant D</summary>

```python
def awrs_d(p, cond, draw=sample_categorical):
    p_cond = p.copy()
    n, psi = 0, 0
    xc0 = None
    for i in range(2):
        while True:
            q = p_cond / p_cond.sum()
            x = draw(q, context=('awrs', i, xc0))
            if cond(x):
                if i == 0:
                    xc0 = x
                break
            else:
                if i == 0:
                    psi += q[x]
                p_cond[x] = 0
            n += 1
    return xc0, (1 - psi) / (n + 1)
```

</details>

The variants differ from one another by a line or two. I encourage you to read
them carefully and try to spot the differences before scrolling further.

<details class="derivation" markdown="1">
<summary>Hint</summary>

Compare the `else` branch and the `return` statement across the four variants.

</details>

Our job is to build testing strategies that catch the buggy ones&mdash;ideally
without needing to know which line is wrong.

But before we can test *anything*, we need to answer the most important
question in all of software testing:

### What Should This Code Do?

You cannot test code without a specification. For randomized algorithms, the
specification comes from theory&mdash;and translating theory into a checkable
property is half the battle. We'll keep coming back to this question for every
algorithm in this post.

For AWRS, the specification is: it should return a sample-weight pair $(x, w)$
such that $w \cdot \mathbf{1}[x = x']$ is an **unbiased estimator** of the
unnormalized conditional distribution for every $x' \in \mathcal{X}$:

$$
\mathop{\mathbb{E}}_{(x,w) \sim \text{AWRS}} \left[ w \cdot \mathbf{1}[x = x'] \right]
=
p(x') \cdot \mathbf{1}[x' \in C]
$$

The right-hand side is the unnormalized target&mdash;$p$ restricted to the
condition $C$, with normalizing constant $Z = \sum_{x \in C} p(x)$.

Note that you may often need to tweak definitions, narrow or broaden the scope
of theory in order to suit your code's intended use cases. This is an important
skill for turning theory into tests.

### Setting Up a Test Case

Now that we have a testable definition, we need to think about which things in
the equation we can compute and which things we need to estimate. Let's make a
concrete test case:

```python
p = np.array([0.5, 0.3, 0.15, 0.04, 0.01])  # a distribution with a long tail
cond = np.array([True, False, True, False, False])  # the "good" elements
N = len(p)

Z = p[cond].sum()
target = p * cond  # unnormalized target (not a distribution!)
```


## Strategy 1: Monte Carlo Averaging

The obvious first testing strategy: run the code "a lot of times" and compute an
average.

{% notebook testing-randomized-algorithms.ipynb cells[1:2] %}

Is this error small enough? How do you know? (Variant D has the smallest
error of the four.)

This is the central problem with Monte Carlo testing: a single number tells you
almost nothing without a reference frame. We need to know what *should* happen
as we increase the number of samples.

### Convergence Rates

A better approach is to watch the error decrease as a function of the number of
samples $M$. It should keep going down *forever*, modulo numerical precision.
And crucially, we often know the *rate* at which it should decrease.

The general picture: suppose we have a sequence of estimators $\widehat{\mu}^{(M)}$
targeting some quantity $\mu$. The mean squared error decomposes as

$$
\mathop{\mathbb{E}} \| \widehat{\mu}^{(M)} - \mu \|^2
= \| \mathop{\mathbb{E}}[\widehat{\mu}^{(M)}] - \mu \|^2
+ \mathop{\mathbb{E}} \| \widehat{\mu}^{(M)} - \mathop{\mathbb{E}}[\widehat{\mu}^{(M)}] \|^2
$$

The first term is the squared bias; the second is the variance (more precisely,
the trace of the covariance of $\widehat{\mu}^{(M)}$). Different classes of estimators
make different predictions about how these terms behave:

- **I.I.D. unbiased** (e.g., our AWRS test). The bias is zero for all $M$.
  The variance is $V/M$ where $V = \text{tr}(\text{Cov}(X))$&mdash;the sum of
  the componentwise variances of a single sample. So the mean squared error is
  exactly $V/M$.<footnote>If you prefer root mean squared
  error, the slope is $-1/2$ and the intercept is $\frac{1}{2} \log
  V$.</footnote>

- **Correlated but unbiased** (e.g., MCMC, Markov reward processes). The bias
  is still zero, but autocorrelation inflates the variance:
  $\mathop{\mathbb{E}} \| \widehat{\mu}^{(M)} - \mu \|^2 \approx V_{\text{eff}} / M$
  where $V_{\text{eff}} > V$ depends on the integrated autocorrelation time.
  The rate is still $O(1/M)$, but the constant is larger.

- **Biased but consistent** (e.g., self-normalized importance sampling). The
  bias is $O(1/M)$, so the squared bias is $O(1/M^2)$. The variance is
  $O(1/M)$.
  The MSE is still $O(1/M)$ overall, so on a log&ndash;log plot the slope looks
  the same&mdash;the bias hides in the noise at large $M$.

In all three cases, $\text{MSE} \sim c / M$ for some constant $c$, which means
$\log \text{MSE} = -\log M + \log c$: a line with slope $-1$ on a
log&ndash;log plot. The intercept $\log c$ tells us the constant.

<details class="derivation" markdown="1">
<summary>Deriving $\,V\,$ for vector-valued estimators</summary>

Our estimator is vector-valued (a distribution over $N$ elements), so we need a
scalar measure of error. The natural choice is the squared $L_2$ norm
$\| \widehat{\mu}^{(M)} - \mu \|^2$ where $\widehat{\mu}^{(M)} = \frac{1}{M} \sum_{m=1}^{M} X^{(m)}$
and $\mu = \mathop{\mathbb{E}}[X^{(m)}]$:

$$
\mathop{\mathbb{E}} \| \widehat{\mu}^{(M)} - \mu \|^2
= \mathop{\mathbb{E}} \sum_{j=1}^{N} (\widehat{\mu}^{(M)}_j - \mu_j)^2
= \sum_{j=1}^{N} \text{Var}(\widehat{\mu}^{(M)}_j)
= \frac{1}{M} \sum_{j=1}^{N} \text{Var}(X^{(m)}_j)
= \frac{V}{M}
$$

where $V = \text{tr}(\text{Cov}(X^{(m)}))$ is the trace of the covariance matrix of a single sample.
The last step uses independence; for correlated samples, the $1/M$ factor
becomes $1/M$ times an autocorrelation correction.

</details>

{% notebook testing-randomized-algorithms.ipynb cells[2:3] %}

A note on reading this plot: when we say "log&ndash;log scale" we mean the axes are
log-transformed but the tick labels show the original values (e.g., $10^2$,
$10^3$). This is what `plt.xscale('log')` does. This is different from
*literally plotting* $\log y$ vs $\log x$, where the axes are linear but the
values are logs. The line-fitting code does the latter (`np.polyfit` on
$\log x$, $\log y$), but for visual inspection, log-scaled axes are usually
clearer because you can read off the original quantities directly.

For a correct estimator, the fitted slope should be close to $-0.5$ (we're
plotting error, not squared error). Variant B fails this test plainly: its
error stops going down after a few dozen samples and the slope is positive.
**B is out.**

The other three look fine. C's slope is a little shallow, but the slope of a
single run is itself a noisy statistic&mdash;rerunning the correct variant with
other seeds gives slopes anywhere from about $-0.25$ to $-0.75$. D's curve is
indistinguishable from A's. That's not a coincidence, and we'll see why later.

### What a Single Run Can and Can't Tell Us

Watching a single long run tells us the *rate* of convergence, but it can't
distinguish the three cases above. An unbiased i.i.d. estimator, a correlated
MCMC chain, and a self-normalized importance sampler can all produce very similar
log&ndash;log plots&mdash;the slopes are the same and the constants differ in ways
that are hard to eyeball.

A pedagogical example: suppose we accidentally computed the average as
$\frac{1}{M+1} \sum_{m=1}^{M} X^{(m)}$ instead of $\frac{1}{M} \sum_{m=1}^{M}
X^{(m)}$. This estimator is *biased*&mdash;it systematically underestimates the
target&mdash;but *consistent*, because the bias factor $M/(M+1) \to 1$. The
single-run convergence plot would look nearly identical.

The same thing is happening to C. Its bias is real, but at a thousand samples
it is still smaller than the noise. A longer run would expose it eventually,
but how long is long enough? A single run has no way to tell us.

### Averaging Over Multiple Runs

To separate bias from variance, we need **multiple independent runs**. The idea:
run the estimator many times at each sample size $M$, and compare the *average*
behavior to the theoretical prediction.

Recall the decomposition from earlier:

$$
\mathop{\mathbb{E}} \| \widehat{\mu}^{(M)} - \mu \|^2
= \| \mathop{\mathbb{E}}[\widehat{\mu}^{(M)}] - \mu \|^2
+ \mathop{\mathbb{E}} \| \widehat{\mu}^{(M)} - \mathop{\mathbb{E}}[\widehat{\mu}^{(M)}] \|^2
$$

For the i.i.d. unbiased case, the variance term is $V/M$. So:

- If the observed MSE matches $V/M$, the squared bias must be near
  zero&mdash;evidence of unbiasedness.
- If the estimator has persistent bias, $\| \mathop{\mathbb{E}}[\widehat{\mu}^{(M)}] -
  \mu \|^2$ does not vanish with $M$. At large $M$, the variance $V/M$ shrinks
  but the bias dominates, causing the observed curve to *flatten out* above the
  theoretical line.
- For biased-but-consistent estimators (like self-normalized IS), the squared
  bias is $O(1/M^2)$&mdash;it vanishes
  faster than the variance. This means the MSE curve will *eventually* match
  $V/M$, but may sit slightly above it for small $M$.

{% notebook testing-randomized-algorithms.ipynb cells[3:4] %}

A and D hug the theoretical $V/M$ line, which is strong evidence that they are
properly weighted&mdash;not just consistent, but unbiased, with the right
variance. B flattens out, as before. C is the interesting one: it tracks the
line for small $M$ and then lifts off as the variance shrinks below its squared
bias. Averaging over a thousand runs makes the gap far larger than the
confidence band. **C is out.**

D, on the other hand, has now passed every test we've thrown at it.

But Monte Carlo testing is still a *blunt instrument*. Small biases hide
easily in the noise, and you need many runs to have any statistical power. For
correlated estimators, the effective variance $V_{\text{eff}}$ may be hard to
compute, making the theoretical line itself uncertain. In software&mdash;unlike
the real world&mdash;we can almost always do better.

## Strategy 2: Program Tracing

Here's the key insight: if the state space is small enough, we can *enumerate
all possible executions* of the randomized program and verify the correctness
condition *exactly* (up to floating point).

The idea is to hijack the random number generation inside the program to make it
systematically explore every possible branch. We track each execution trace's
probability, and combine all traces to compute the exact output distribution.

### The Tracer

The implementation uses a trie data structure that mirrors the branching
structure of the program's random choices. Each node in the trie corresponds to
a decision point, and its children correspond to the possible outcomes.

The beautiful thing about `TraceSWOR` is that it works as a context manager:
each `with tracer:` block explores one trace, and when it exits, that trace's
mass is zeroed out so we never revisit it. We loop until the root's mass hits
zero&mdash;meaning we've exhausted every possible execution.

<details class="derivation" markdown="1">
<summary>Tracer implementation</summary>

```python
class Node:
    """A node in the execution trie."""
    def __init__(self, idx, mass, parent, token=None):
        self.idx = idx
        self.mass = mass          # remaining probability mass
        self.parent = parent
        self.token = token        # label for visualization
        self.child_masses = None  # set on first visit
        self.active_children = {}
        self._mass = mass         # remember original mass
        self.context = None

    def sample(self):
        """Sample a child proportional to remaining mass."""
        p = self.child_masses / self.child_masses.sum()
        return np.random.choice(len(p), p=p)

    def update(self):
        """Propagate mass changes up to the root."""
        if self.parent is not None:
            self.parent.child_masses[self.idx] = self.mass
            self.parent.mass = np.sum(self.parent.child_masses)
            self.parent.update()


class Tracer:
    """
    Lazily materializes the probability tree of a generative process
    by program tracing. Works with both array-valued and dict-valued
    distributions.
    """
    def __init__(self):
        self.root = Node(idx=-1, mass=1.0, parent=None)
        self.cur = None

    def __call__(self, p, context=None):
        """Drop-in replacement for any sampling function."""
        if isinstance(p, dict):
            keys = list(p.keys())
            vals = np.array([p[k] for k in keys])
        else:
            p = np.asarray(p, dtype=float)
            keys = list(range(len(p)))
            vals = p
        vals = vals / vals.sum()
        cur = self.cur

        if cur.child_masses is None:
            cur.child_masses = cur.mass * vals
            cur.context = context

        a = cur.sample()
        if a not in cur.active_children:
            cur.active_children[a] = Node(
                idx=a, mass=cur.child_masses[a], parent=cur,
                token=keys[a],
            )
        self.cur = cur.active_children[a]
        return keys[a]


class TraceSWOR(Tracer):
    """
    Sampling without replacement meets program tracing.

    Each execution trace is explored exactly once. After a trace
    completes, its probability mass is zeroed out and propagated
    up the trie, ensuring the next trace is fresh.
    """
    def __enter__(self):
        self.cur = self.root

    def __exit__(self, *args):
        self.cur.mass = 0
        self.cur.update()
```

</details>

### Exact Verification

Now we can verify all four AWRS variants *exactly*:

{% notebook testing-randomized-algorithms.ipynb cells[5:6] %}

This is not a statistical test. It is an *exact* check (modulo floating point)
that the proper weighting condition holds.<footnote>The difference between `target` and `prob` is not zero&mdash;it rarely will be with floating point numbers. But `np.allclose` checks that they agree to within a reasonable tolerance ($\approx 10^{-8}$).</footnote> No confidence intervals, no
squinting at log&ndash;log plots. The tracer gives a definitive yes or no for each
variant. **D is out**, and A is the only survivor.

D's bug is in the `else` branch: it accumulates the *normalized* probability
`q[x]` instead of the remaining mass `p_cond[x]`. The two agree on the first
rejection, because `p_cond` still sums to one. They differ only when the first
pass rejects two or more elements. With our long-tailed `p`, that happens in
about one run in twenty, and even then the weight is off by only a couple of
percent&mdash;which is why D's single run looked just like A's.

### How Many Samples Would Monte Carlo Need?

The tracer computes more than a pass/fail bit. Collecting the second moment
alongside the first gives the exact squared bias and the exact variance $V$ of
each variant. The bias of the average stays fixed as $M$ grows while its
variance shrinks like $V/M$, so the two cross at $M^* = V / \text{bias}^2$.
Below $M^*$, the bug is buried in the noise.

{% notebook testing-randomized-algorithms.ipynb cells[6:7] %}

This is the ladder we just climbed. B surfaces after a few dozen samples, so a
single run catches it. C needs a few hundred, so it takes careful averaging. D
needs about fifteen million, and no reasonable Monte Carlo test would ever see
it. The tracer took a fraction of a second.



## Strategy 3: Testing the Whole Pipeline

AWRS is not meant to be used on its own. In [Lipkin et al., 2025](https://arxiv.org/abs/2504.05410),
it is the proposal step inside sequential Monte Carlo for generating text from
a language model subject to a constraint. Each particle asks AWRS for a next
token that satisfies the constraint, and multiplies its weight by the weight
that AWRS returns.

Program tracing is not limited to a single call of a sampling routine. It works
on anything that makes discrete random choices through a hookable interface, so
we can trace the whole pipeline at once&mdash;every AWRS call in every particle,
plus the resampling steps.

### What Should This Code Do?

Let $p$ be a language model over strings, and let $C$ be a constraint that can
be checked one token at a time. The unnormalized target is

$$
\gamma(\mathbf{x}) = p(\mathbf{x}) \cdot \mathbf{1}[\mathbf{x} \in C]
$$

and its normalizing constant $Z = \sum_{\mathbf{x}} \gamma(\mathbf{x})$ is the
probability that a string drawn from $p$ satisfies the constraint. SMC returns
$M$ weighted particles $(\mathbf{x}_m, w_m)$, and the specification is the
proper weighting condition from before, lifted from tokens to strings:

$$
\mathop{\mathbb{E}} \left[ \frac{1}{M} \sum_{m=1}^{M} w_m \cdot \mathbf{1}[\mathbf{x}_m = \mathbf{x}'] \right]
=
\gamma(\mathbf{x}')
$$

for every string $\mathbf{x}'$. Summing over $\mathbf{x}'$ gives the more
familiar statement that $\frac{1}{M} \sum_m w_m$ is an unbiased estimate of $Z$.
Checking every string is the stronger test: a bug that moves mass from one
string to another without changing the total would slip past a check on $Z$
alone.

Why should this hold? Proper weighting composes. At step $t$, AWRS returns a
token and a weight that are properly weighted for $p(\cdot \mid
\mathbf{x}_{\lt t})$ restricted to the allowed tokens. Multiplying these weights
along a string gives a properly weighted sample for the product of the
per-step targets, which is $\gamma$. Resampling preserves proper weighting as
long as every resampled particle carries the average weight.

### Setting Up a Test Case

The language model is a small, explicit distribution over strings. The
constraint says that a `b` must be followed by an `a` or by the end of the
string, so `bb` and `bc` are ruled out:

```python
lm = SimpleLM({
    (): 0.1, ('a',): 0.1, ('a', 'a'): 0.1, ('a', 'b'): 0.1,
    ('b',): 0.1, ('b', 'a'): 0.15, ('b', 'b'): 0.25, ('b', 'c'): 0.1,
})

VOCAB = ['a', 'b', 'c', EOS]

def allowed(prefix, token):
    """After a 'b', the next token must be 'a' or EOS."""
    return not (prefix[-1:] == ('b',) and token not in ('a', EOS))

target_lm = {s: prob * is_valid(s) for s, prob in lm.sequences.items()}
Z_lm = sum(target_lm.values())  # = 0.65, not 1
```

<details class="derivation" markdown="1">
<summary><span>Utilities: <code>SimpleLM</code>, <code>is_valid</code></span></summary>

```python
EOS = '<EOS>'  # end-of-sequence marker

class SimpleLM:
    """
    A sequence model defined by an explicit, finite distribution over strings.
    Maps prefixes to next-token distributions (as dicts).
    """
    def __init__(self, sequences):
        """
        sequences: dict mapping tuple-of-tokens to probability.
        E.g., {('a','a'): 0.3, ('b','b','b'): 0.6, (): 0.1}
        """
        self.sequences = sequences

    def p_next(self, prefix):
        """Return conditional distribution over next token given prefix."""
        dist = {}
        for seq, prob in self.sequences.items():
            if len(seq) < len(prefix):
                continue
            if seq[:len(prefix)] != prefix:
                continue
            if len(seq) == len(prefix):
                dist[EOS] = dist.get(EOS, 0) + prob
            else:
                token = seq[len(prefix)]
                dist[token] = dist.get(token, 0) + prob
        total = sum(dist.values())
        return {k: v / total for k, v in dist.items()}

def is_valid(string):
    return all(allowed(string[:t], string[t]) for t in range(len(string)))
```

</details>

Here is the sampler. At each step, every unfinished particle extends itself
with AWRS and multiplies in the AWRS weight. When the effective sample size
drops below `tau` times the number of live particles, it resamples. With
`tau=0` it never resamples, and the algorithm reduces to sequential importance
sampling (SIS); with `tau>0` it is sequential importance resampling (SIR).

```python
from collections import namedtuple
Particle = namedtuple('Particle', 'weight, string')

def smc(lm, M, tau, awrs, draw=sample_categorical):
    unfinished = [Particle(1, ()) for _ in range(M)]
    finished = []
    while unfinished:
        next_unfinished = []
        for particle in unfinished:
            P = lm.p_next(particle.string)
            p = np.array([P.get(v, 0) for v in VOCAB])
            i, w = awrs(p, lambda i: allowed(particle.string, VOCAB[i]), draw=draw)
            x = VOCAB[i]
            if x == EOS:
                finished.append(Particle(particle.weight * w, particle.string))
            else:
                next_unfinished.append(Particle(particle.weight * w,
                                                particle.string + (x,)))
        unfinished = next_unfinished
        if not unfinished:
            break

        W = np.array([particle.weight for particle in unfinished])
        if W.sum()**2 / (W**2).sum() < tau * len(unfinished):
            indices = [draw(W / W.sum(), context='resample') for _ in unfinished]
            unfinished = [Particle(W.sum() / len(unfinished), unfinished[k].string)
                          for k in indices]

    return finished
```

The same `draw` hook is passed down to every AWRS call and to the resampling
step, so a single tracer sees every random choice the pipeline makes.

### Sequential Importance Sampling

One particle is enough to check the proper weighting condition for SIS. We run
it with each AWRS variant, plus one more contender: *locally constrained
decoding*, which samples each token from the language model restricted to the
allowed tokens and ignores the weight. It is the obvious way to do constrained
generation.

{% notebook testing-randomized-algorithms.ipynb cells[8:9] %}

Every bug is caught through the pipeline, including D. After a `b`, two of
the four tokens are forbidden, so AWRS's first pass can reject twice in one
step&mdash;exactly the situation that triggers D's bug.

The last line is the most instructive one. Locally constrained decoding never
produces an invalid string, but it gets the probabilities wrong: it
over-represents strings whose prefixes leave little room under the constraint.
Here, it puts too much mass on `b` and `ba`, because it never pays for the
probability that `b` would have been followed by a forbidden token. This bias
is the reason AWRS returns a weight in the first place.

### Sequential Importance Resampling

Resampling is just one more random choice, made through the same `draw` hook,
so the tracer needs no changes. Two particles are enough to exercise it:

{% notebook testing-randomized-algorithms.ipynb cells[9:10] %}

The correct implementation passes and D is still caught. But notice the
number of traces: 162 with one particle, and nearly 30,000 with two.

**Pro tip:** Make sure the total mass decreases quickly, as the loop will halt
only when the total mass is zero. This requires the support to be finite. For
problems where the support is large but finite, it may be a reasonable heuristic
to stop early if the remaining mass is small, trading exactness for an unbiased
*estimate* (via techniques from [Shi et al., 2020](https://arxiv.org/abs/2002.09067)).<footnote>Traces are a combinatorial space, so this strategy is only practical for smaller problems. But "smaller problems" is exactly the regime where tests should live&mdash;small enough to verify exactly, but complex enough to exercise the algorithm's logic.</footnote>


## Some Miscellaneous Thoughts

Testing is fundamentally about **control**. Some abstractions give us more
control over a system's behavior than others, and good tests depend on choosing
abstractions that make critical behaviors observable, reproducible, and
manipulable.

The `draw` parameter in all of the algorithms above is a good example. It's a
tiny abstraction&mdash;just a function that returns a random choice&mdash;but it gives us
enormous leverage. In production, it calls the RNG. In testing, it becomes a
tracer that enumerates every possible execution. Same algorithm, radically
different observability.

A few principles I've found useful:

- **Theory gives you test cases.** The proper weighting definition, the
  unbiasedness of importance sampling, the convergence rate of Monte
  Carlo&mdash;these are all *specifications* that can be mechanically checked. If you
  don't understand the theory behind your algorithm, you don't know what to test.

- **Small exact tests beat large approximate tests.** A tracer that exhaustively
  checks a 5-element distribution is worth more than a Monte Carlo test with
  10,000 samples. The tracer gives you a definitive yes/no; the Monte Carlo test
  gives you a squiggly line that you have to interpret.

- **Inject errors to calibrate your tests.** A test that passes on correct code
  but also passes on buggy code is useless. Deliberately break the algorithm and
  verify that your test catches it.

- **Monte Carlo is the blunt instrument of last resort.** It's indispensable
  when the state space is too large to enumerate, but in software&mdash;unlike the
  real world&mdash;we can almost always find ways to do better. Freezing the RNG
  makes your code deterministic, but it is at best a regression test and
  extremely brittle.

Of course, some behaviors are only visible from "1,000 feet"&mdash;via integration
tests, stress tests, or long-horizon statistical checks. Not everything can be
checked with a microscope. But the more control you have over the randomness in
your program, the more you can check exactly rather than approximately.
