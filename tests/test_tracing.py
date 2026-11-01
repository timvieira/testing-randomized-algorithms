"""Tests for the program_tracing library on programs with known distributions."""

import sys
from collections import defaultdict
from math import comb
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest

from program_tracing import TraceGreedy, TraceSWOR, Tracer, traces


def distribution(program, tracer=None):
    """Exact output distribution of `program`, as {output: probability}."""
    dist = defaultdict(float)
    for prob, output in traces(program, tracer):
        dist[output] += prob
    return dict(dist)


def coin_sum(draw, n=4, p=0.3):
    return sum(draw([1 - p, p]) for _ in range(n))


def test_binomial():
    """The sum of independent coin flips is binomial."""
    dist = distribution(coin_sum)
    for k in range(5):
        assert dist[k] == pytest.approx(comb(4, k) * 0.3**k * 0.7**(4 - k))


def test_each_trace_once():
    """Every execution is visited exactly once, and their probabilities sum to one."""
    seen = []
    def program(draw):
        choices = tuple(draw([0.5, 0.25, 0.25]) for _ in range(3))
        seen.append(choices)
        return choices
    probs = [prob for prob, _ in traces(program)]
    assert len(seen) == len(set(seen)) == 27
    assert sum(probs) == pytest.approx(1)


def test_dict_distributions():
    """Dict-valued distributions return their keys."""
    def program(draw):
        return draw({'a': 1, 'b': 3}) + draw({'x': 0.5, 'y': 0.5})
    assert distribution(program) == pytest.approx({'ax': 1/8, 'ay': 1/8, 'bx': 3/8, 'by': 3/8})


def test_unnormalized_weights():
    """Weights don't need to sum to one."""
    assert distribution(lambda draw: draw([2, 6])) == pytest.approx({0: 0.25, 1: 0.75})


def test_variable_length():
    """Traces can have different lengths: a geometric draw, truncated at 5 flips."""
    def program(draw):
        n = 0
        while n < 5 and draw([0.5, 0.5]) == 1:
            n += 1
        return n
    dist = distribution(program)
    for n in range(5):
        assert dist[n] == pytest.approx(0.5**(n + 1))
    assert dist[5] == pytest.approx(0.5**5)


def test_branching_on_earlier_choices():
    """Later distributions can depend on earlier draws (a tiny Polya urn)."""
    def program(draw):
        red, blue = 1, 1
        for _ in range(3):
            if draw([red, blue]) == 0:
                red += 1
            else:
                blue += 1
        return red - 1
    # A Polya urn started at (1, 1) gives a uniform number of red draws.
    assert distribution(program) == pytest.approx({k: 1/4 for k in range(4)})


def test_zero_probability_outcomes_are_skipped():
    def program(draw):
        return draw([0.0, 1.0, 0.0])
    assert list(traces(program)) == [(1.0, 1)]


def test_no_random_choices():
    """A program that draws nothing has one trace, with probability one."""
    assert list(traces(lambda draw: 'done')) == [(1.0, 'done')]


@pytest.mark.parametrize('seed', [0, 1, 2])
def test_orders_agree(seed):
    """Sampling and greedy orders visit the same traces with the same probabilities."""
    np.random.seed(seed)
    sampled = distribution(coin_sum, TraceSWOR())
    greedy = distribution(coin_sum, TraceGreedy())
    assert sampled.keys() == greedy.keys()
    for k in sampled:
        assert sampled[k] == pytest.approx(greedy[k])


def test_greedy_starts_with_heaviest_choices():
    """Greedy's first trace takes the most probable branch at every choice."""
    (prob, output), *_ = traces(lambda draw: (draw([0.2, 0.8]), draw([0.7, 0.3])), TraceGreedy())
    assert output == (1, 0)
    assert prob == pytest.approx(0.8 * 0.7)


def test_early_stopping_leaves_the_remaining_mass():
    tracer = TraceSWOR()
    explored = 0.0
    for k, (prob, _) in enumerate(traces(coin_sum, tracer)):
        explored += prob
        if k == 4:
            break
    assert tracer.root.mass == pytest.approx(1 - explored)


def test_context_is_recorded():
    tracer = TraceSWOR()
    with tracer:
        tracer([0.5, 0.5], context='first flip')
    assert tracer.root.context == 'first flip'


def test_hidden_randomness_is_detected():
    """A program whose choices depend on randomness outside the tracer is rejected."""
    rng = np.random.default_rng(0)
    def program(draw):
        n = int(rng.integers(2, 5))   # not routed through draw
        return draw(np.ones(n))
    with pytest.raises(ValueError, match='deterministic'):
        list(traces(program))


def test_plain_tracer_follows_one_path():
    """The base Tracer is a drop-in sampler: it just makes random choices."""
    tracer = Tracer()
    tracer.cur = tracer.root
    assert tracer([0.0, 1.0]) == 1
