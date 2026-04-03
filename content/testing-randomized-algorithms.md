title: Testing Randomized Algorithms
date: 2026-04-02
comments: true
tags: testing, sampling, algorithms, monte-carlo, program-tracing

Randomized algorithms are hard to test. Their outputs are, by design,
unpredictable&mdash;and that unpredictability is a *feature*, not a bug. It is what
gives them useful guarantees like unbiasedness and computational efficiency. But
it also makes them slippery: how do you check that a stochastic program is
correct when it gives different answers every time you run it?

This post sketches some ideas for how to think about testing randomized
algorithms&mdash;especially the kinds of algorithms that we use for probabilistic
inference. I'll walk through a concrete sampling algorithm, develop three
progressively more powerful testing strategies, and try to make the case that the
best tests come from actually understanding the theory behind the algorithm
you're testing.

## A Tricky Little Sampling Algorithm

Suppose we have a distribution $p$ over a finite set $\mathcal{X} = \{1,
\ldots, N\}$ and a condition $C \subseteq \mathcal{X}$ (the set of "good"
elements). We want to sample from the *conditional* distribution $p(\cdot \mid
C)$. But rather than returning a plain sample, we want a *properly weighted*
sample $(x, w)$&mdash;a sample-weight pair that can be used for unbiased estimation.

The following algorithm, which we call AWRS (from [Lipkin et al., 2025](https://arxiv.org/abs/2504.05410)), does exactly this:

<details class="derivation" markdown="1">
<summary>AWRS pseudocode</summary>

**Input:** distribution $p$, condition $C$

1. Set $\psi \leftarrow 0$, $n \leftarrow 0$
2. **First pass** ($i=0$): Draw $x \sim p_{\text{cond}}$ (normalized). If $x \in C$, set $x_{c_0} \leftarrow x$ and break. Otherwise, set $\psi \leftarrow \psi + p_{\text{cond}}[x]$, zero out $p_{\text{cond}}[x]$, and increment $n$.
3. **Second pass** ($i=1$): Draw $x \sim p_{\text{cond}}$ (normalized). If $x \in C$, break. Otherwise, zero out $p_{\text{cond}}[x]$ and increment $n$.
4. **Return** $(x_{c_0},\ (1 - \psi) / (n+1))$

</details>

Here is a translation into Python. I've kept the `draw` parameter as a hook
we'll exploit later for testing:

```python
def awrs(p, cond, draw=sample_categorical):
    """
    Properly weighted sampling from a conditioned distribution.
    Returns (item, weight) pair.
    """
    p_cond = p.copy()
    n, psi = 0, 0
    xc0 = None
    for i in range(2):
        while True:
            x = draw(p_cond / p_cond.sum(), context=('awrs', i, xc0))
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

<details class="derivation" markdown="1">
<summary>Utility: <code>sample_categorical</code></summary>

```python
import numpy as np

def sample_categorical(p, context=''):
    """Sample from a categorical distribution (normalized internally)."""
    p = np.asarray(p, dtype=float)
    p = p / p.sum()
    return np.random.choice(len(p), p=p)
```

</details>

Did we implement it correctly?

Well... what is it *supposed to do*?

### Proper Weighting

AWRS is supposed to generate "properly weighted samples for the distribution
$\widetilde{p} = Z \cdot p$." Let's unpack that.

A sampler $Q$ that outputs pairs $(x, w)$ is **properly weighted** for an
unnormalized target $\widetilde{p}$ if for all $x' \in \mathcal{X}$:

$$
\mathop{\mathbb{E}}_{(x,w) \sim Q} \left[ w \cdot \mathbf{1}[x = x'] \right]
=
\widetilde{p}(x')
$$

In our case, $\widetilde{p}(x) = p(x) \cdot \mathbf{1}[x \in C]$&mdash;the
unnormalized conditional distribution with normalizing constant $Z = \sum_{x \in
C} p(x)$.<footnote>The textbook definition quantifies over all bounded measurable functions $f$, requiring $\mathop{\mathbb{E}}[w \cdot f(x)] = \sum_x \widetilde{p}(x) f(x)$. For finite $\mathcal{X}$, taking $f$ to be element-wise indicators is sufficient. You should convince yourself of this.</footnote>

Note that you may often need to tweak definitions, narrow or broaden the scope
of theory in order to suit your code's intended use cases. This is an important
skill for turning theory into tests.

### Setting Up a Test Case

Now that we have a testable definition, we need to think about which things in
the equation we can compute and which things we need to estimate. Let's make a
concrete test case:

```python
np.random.seed(0)
N = 4
p = np.random.dirichlet(np.ones(N))  # random distribution over {0,...,N-1}
cond = np.random.uniform(0, 1, size=N) < 0.5  # random subset

Z = p[cond].sum()
target = p * cond  # unnormalized target (not a distribution!)
```


## Strategy 1: Monte Carlo Averaging

The obvious first testing strategy: run the code "a lot of times" and compute an
average.

```python
np.random.seed(0)
reps = 1_000
have = np.zeros(N)

for r in range(1, 1 + reps):
    (x, w) = awrs(p, lambda x: cond[x])
    have[x] += w

print(f'error = {abs(have / reps - target).sum():.6f}')
```

Is this error small enough? How do you know?

This is the central problem with Monte Carlo testing: a single number tells you
almost nothing without a reference frame. We need to know what *should* happen
as we increase the number of samples.

### Convergence Rates

A better approach is to watch the error decrease as a function of the number of
samples. It should keep going down *forever*, modulo numerical precision. And
crucially, we know the *rate* at which it should decrease.

For i.i.d. Monte Carlo estimation, the mean squared error is
$\text{MSE} = \text{Var}(X) / M$ where $M$ is the number of samples. So the root
mean squared error should decrease as $O(1/\sqrt{M})$. On a log&ndash;log plot, this
is a line with slope $-1/2$.

{% notebook testing-randomized-algorithms.ipynb cells[1:2] %}

Why a log&ndash;log scale? Because power laws $y = c \cdot x^{a}$ become lines: $\log
y = a \log x + \log c$. The fitted slope should be close to $-0.5$, which is
exactly what we see.

### Averaging Over Multiple Runs

A single squiggly line on a log&ndash;log plot is suggestive but not rigorous. We can
do better by averaging over many independent runs and comparing to the
theoretical MSE curve.

Because we have a multidimensional target, the relevant quantity is the *trace*
of the covariance matrix: $V = \text{tr}(\text{Cov}(X))$. The theoretical MSE
of the $L_2$ error is then $V / M$.

{% notebook testing-randomized-algorithms.ipynb cells[2:3] %}

So what did we actually accomplish? We can now "eyeball" that the mean squared
error is decreasing at the theoretical rate of $1/M$. The observed curve hugs
the theoretical one, which is strong evidence that our sampler is properly
weighted.

But Monte Carlo testing is a *blunt instrument*. It is hard to beat in the real
world, but in software, we can almost always do better.

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
    by program tracing.
    """
    def __init__(self):
        self.root = Node(idx=-1, mass=1.0, parent=None)
        self.cur = None

    def __call__(self, p, context=None):
        """Drop-in replacement for sample_categorical."""
        p = np.asarray(p, dtype=float)
        cur = self.cur

        if cur.child_masses is None:
            cur.child_masses = cur.mass * p
            cur.context = context

        a = cur.sample()
        if a not in cur.active_children:
            cur.active_children[a] = Node(
                idx=a, mass=cur.child_masses[a], parent=cur, token=a,
            )
        self.cur = cur.active_children[a]
        return a


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

Now we can verify AWRS *exactly*:

```python
tracer = TraceSWOR()
prob = np.zeros(N)
while tracer.root.mass > 0:
    with tracer:
        (item, w) = awrs(p, lambda x: cond[x], draw=tracer)
        prob[item] += w * tracer.cur.mass

assert np.allclose(target, prob), f'mismatch: {target} vs {prob}'
print('✓ AWRS is properly weighted (exact verification)')
```

This is not a statistical test. It is an *exact* check (modulo floating point)
that the proper weighting condition holds. No confidence intervals, no
squinting at log&ndash;log plots. If this assertion passes, the algorithm is correct
for this test case.

I encourage you to try injecting errors into the AWRS code to see how quickly
this test catches them. For example, try changing `(1 - psi)` to `(1 + psi)`,
or removing the `if i == 0` guard. The tracer catches these immediately.

We can visualize the traces that were explored along with their context metadata:

{% notebook testing-randomized-algorithms.ipynb cells[3:4] %}

<footnote>The difference between `target` and `prob` is not zero&mdash;it rarely will be with floating point numbers. But `np.allclose` checks that they agree to within a reasonable tolerance ($\approx 10^{-8}$).</footnote>


## Strategy 3: Testing Sequential Algorithms

Program tracing is not limited to simple sampling algorithms. It works on
anything that makes discrete random choices through a hookable interface. Let's
apply it to **sequential importance sampling** (SIS) and **sequential importance
resampling** (SIR).

### A Simple Sequence Model

We need a way to generate sequences. A "language model" here is just a function
that, given a prefix, returns a distribution over the next token (including a
special end-of-sequence token). We define a target $p$ and a proposal $q$:

```python
p_lm = SimpleLM({('a', 'a'): 0.3, ('b', 'b', 'b'): 0.6, (): 0.1})
q_lm = SimpleLM({('a', 'a'): 0.2, ('b', 'b', 'b'): 0.7, (): 0.1})
```

<details class="derivation" markdown="1">
<summary>Utilities: <code>SimpleLM</code>, <code>sample_dict</code></summary>

```python
EOS = '<EOS>'  # end-of-sequence marker

class SimpleLM:
    """
    A simple sequence model defined by a probabilistic context-free grammar.
    Maps prefixes to next-token distributions (as dicts).
    """
    def __init__(self, sequences):
        """
        sequences: dict mapping tuple-of-tokens to probability.
        E.g., {('a','a'): 0.3, ('b','b','b'): 0.6, (): 0.1}
        """
        self.sequences = sequences

    def p_next(self, prefix):
        """Return distribution over next token given prefix."""
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
        return dist

def sample_dict(d, context=None):
    """Sample a key from a dict-valued distribution."""
    keys = list(d.keys())
    vals = np.array([d[k] for k in keys])
    vals = vals / vals.sum()
    idx = np.random.choice(len(keys), p=vals)
    return keys[idx]
```

</details>

### Sequential Importance Sampling

SIS generates weighted samples from $p$ using proposal $q$. At each step, it
extends each particle by sampling from $q$ and reweighting by $p/q$:

```python
from collections import namedtuple
Particle = namedtuple('Particle', 'weight, string')

def sequential_importance_sampling(p, q, M, draw=sample_dict):
    unfinished = [Particle(1, ()) for _ in range(M)]
    finished = []
    while unfinished:
        next_unfinished = []
        for particle in unfinished:
            P = p.p_next(particle.string)
            Q = q.p_next(particle.string)
            x = draw(Q)
            if x == EOS:
                finished.append(Particle(
                    particle.weight * P[x] / Q[x], particle.string))
            else:
                next_unfinished.append(Particle(
                    particle.weight * P[x] / Q[x],
                    particle.string + (x,)))
        unfinished = next_unfinished
    return finished
```

What is this code supposed to do? The theory says that $\frac{1}{M} \sum_m w_m$
is an unbiased estimate of the partition function $Z =
\sum_{\mathbf{x}} p(\mathbf{x})$&mdash;in this case, $Z = 1$ since our target
is already normalized.

We need a slightly different tracer that handles dict-valued distributions.

<details class="derivation" markdown="1">
<summary><code>DictTraceSWOR</code> implementation</summary>

```python
class DictTracer(Tracer):
    """Tracer variant that works with dict-valued distributions."""

    def __call__(self, d, context=None):
        keys = list(d.keys())
        vals = np.array([d[k] for k in keys])
        vals = vals / vals.sum()
        cur = self.cur

        if cur.child_masses is None:
            cur.child_masses = cur.mass * vals
            cur.context = context

        a = cur.sample()
        if a not in cur.active_children:
            cur.active_children[a] = Node(
                idx=a, mass=cur.child_masses[a], parent=cur, token=keys[a],
            )
        self.cur = cur.active_children[a]
        return keys[a]


class DictTraceSWOR(DictTracer):
    """Dict-based TraceSWOR."""
    def __enter__(self):
        self.cur = self.root

    def __exit__(self, *args):
        self.cur.mass = 0
        self.cur.update()
```

</details>

Now we verify SIS:

```python
tracer = DictTraceSWOR()
have = 0
while tracer.root.mass > 0:
    with tracer:
        M = 1
        finished = sequential_importance_sampling(p_lm, q_lm, M=M, draw=tracer)
        Z_hat = sum(particle.weight for particle in finished) / M
        have += Z_hat * tracer.cur.mass

assert abs(have - 1.0) < 1e-10, f'SIS is not unbiased! Z={have}'
print('✓ SIS is unbiased (exact verification)')
```

Here is the trace tree for SIS:

{% notebook testing-randomized-algorithms.ipynb cells[4:5] %}

### Sequential Importance Resampling

SIR adds a resampling step when the effective sample size drops below a
threshold. This creates more diverse particles but also more execution traces:

```python
def sequential_importance_resampling(p, q, M, tau, draw=sample_dict):
    unfinished = [Particle(1, ()) for _ in range(M)]
    finished = []
    t = 0
    while True:
        t += 1
        next_unfinished = []
        for m, particle in enumerate(unfinished):
            P = p.p_next(particle.string)
            Q = q.p_next(particle.string)
            x = draw(Q, context=(t, m))
            if x == EOS:
                finished.append(Particle(
                    particle.weight * P[x] / Q[x], particle.string))
            else:
                next_unfinished.append(Particle(
                    particle.weight * P[x] / Q[x],
                    particle.string + (x,)))
        unfinished = next_unfinished
        if not unfinished:
            break

        W = sum(particle.weight for particle in unfinished)
        W2 = sum(particle.weight**2 for particle in unfinished)
        M_eff = W**2 / W2

        if M_eff < tau * M:
            distr = {k: particle.weight / W
                     for k, particle in enumerate(unfinished)}
            indices = [draw(distr, context=('resample', t, m))
                       for m in range(M)]
            avg_weight = W / M
            unfinished = [Particle(avg_weight, unfinished[k].string)
                          for k in indices]

    return finished
```

We can trace SIR the same way:

```python
tracer = DictTraceSWOR()
have = 0
while tracer.root.mass > 0:
    with tracer:
        M = 3
        finished = sequential_importance_resampling(
            p_lm, q_lm, tau=0.75, M=M, draw=tracer)
        Z_hat = sum(particle.weight for particle in finished) / M
        have += Z_hat * tracer.cur.mass

assert abs(have - 1.0) < 1e-10
print('✓ SIR is unbiased (exact verification)')
```

Here is the trace tree for SIR&mdash;notice that with resampling we have more traces
than without. This is because there are more places that request random
decisions in the code.

{% notebook testing-randomized-algorithms.ipynb cells[5:6] %}

**Pro tip:** Make sure the total mass decreases quickly, as the loop will halt
only when the total mass is zero. This requires the support to be finite. For
problems where the support is large but finite, it may be a reasonable heuristic
to stop early if the remaining mass is small, trading exactness for an unbiased
*estimate* (via techniques from [Lew et al., 2020](https://arxiv.org/abs/2002.09067)).<footnote>Traces are a combinatorial space, so this strategy is only practical for smaller problems. But "smaller problems" is exactly the regime where tests should live&mdash;small enough to verify exactly, but complex enough to exercise the algorithm's logic.</footnote>


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
  checks a 4-element distribution is worth more than a Monte Carlo test with
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
