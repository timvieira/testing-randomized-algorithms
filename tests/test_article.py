"""
Tests for the claims made in the blog post.

Each test checks a statement in content/testing-randomized-algorithms.md,
using the code from the notebook the article is built from. If a test fails,
the article is wrong until proven otherwise.
"""

import ast
import contextlib
import io
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pytest

import program_tracing
from program_tracing import TraceGreedy, TraceSWOR, traces

ARTICLE = (ROOT / 'content' / 'testing-randomized-algorithms.md').read_text()
NOTEBOOK = ROOT / 'content' / 'testing-randomized-algorithms.ipynb'

# Notebook cells that only define things (or run fast checks), by a line they contain.
DEFINITION_MARKERS = ['def awrs_a', 'from program_tracing import', 'def exact_moments',
                      'def naive', 'def smc', 'def check_smc']


@pytest.fixture(scope='module')
def nb():
    """Namespace from running the notebook's definition cells, in order."""
    cells = json.loads(NOTEBOOK.read_text())['cells']
    ns = {}
    for cell in cells:
        src = ''.join(cell['source'])
        if cell['cell_type'] == 'code' and any(m in src for m in DEFINITION_MARKERS):
            with contextlib.redirect_stdout(io.StringIO()):
                exec(src, ns)
    return ns


def run_awrs(nb, awrs):
    return lambda draw: awrs(nb['p'], lambda x: nb['cond'][x], draw=draw)


# ---------------------------------------------------------------------------
# The article's code is the library's code
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('name', ['Node', 'Tracer', 'TraceSWOR', 'TraceGreedy'])
def test_article_shows_library_code(name):
    """The tracer code printed in the article is exactly the library's."""
    src = Path(program_tracing.tracer.__file__).read_text()
    node = next(n for n in ast.parse(src).body if getattr(n, 'name', None) == name)
    code = '\n'.join(src.splitlines()[node.lineno - 1:node.end_lineno])
    assert code in ARTICLE


def test_article_shows_notebook_variants(nb):
    """The four AWRS variants printed in the article match the notebook."""
    cell = next(''.join(c['source']) for c in json.loads(NOTEBOOK.read_text())['cells']
                if 'def awrs_a' in ''.join(c['source']))
    for name in 'abcd':
        code = re.search(rf'(def awrs_{name}\(.*?return xc0, [^\n]*)', cell, re.S).group(1)
        assert code in ARTICLE


# ---------------------------------------------------------------------------
# Strategy 2: program tracing
# ---------------------------------------------------------------------------

def test_only_variant_a_is_properly_weighted(nb):
    """'D is out, and A is the only survivor.'"""
    for name, awrs in nb['variants'].items():
        m1, _ = nb['exact_moments'](awrs)
        assert np.allclose(m1, nb['target']) == (name == 'A'), name


def test_d_differs_only_on_double_rejections(nb):
    """D's weight differs from A's only when the first pass rejects twice: 'about
    one run in twenty', and 'off by only a couple of percent'."""
    # D and A make identical random choices, so replay A's choices into D on each trace.
    differing, rel = 0.0, []
    tracer = TraceSWOR()
    while tracer.root.mass > 0:
        with tracer:
            choices = []
            def recording(p, context=None):
                choices.append(tracer(p, context=context))
                return choices[-1]
            x, w_a = nb['awrs_a'](nb['p'], lambda x: nb['cond'][x], draw=recording)
            replay = iter(choices)
            _, w_d = nb['awrs_d'](nb['p'], lambda x: nb['cond'][x], draw=lambda p, context=None: next(replay))
            if not np.isclose(w_a, w_d):
                differing += tracer.cur.mass
                rel.append(abs(w_d - w_a) / w_a)
    assert 0.03 < differing < 0.08
    assert max(rel) < 0.05


def test_monte_carlo_sample_sizes(nb):
    """The M* = V / bias^2 ladder: ~50 for B, ~400 for C, ~15 million for D."""
    mstar = {}
    for name, awrs in nb['variants'].items():
        m1, m2 = nb['exact_moments'](awrs)
        bias2 = np.sum((m1 - nb['target'])**2)
        V = np.sum(m2 - m1**2)
        mstar[name] = V / bias2 if bias2 > 1e-20 else np.inf
    assert mstar['A'] == np.inf
    assert 30 < mstar['B'] < 70
    assert 300 < mstar['C'] < 500
    assert 1.2e7 < mstar['D'] < 1.8e7


def test_naive_sampler_has_more_variance(nb):
    """Both are unbiased, but the naive weight has about 2.4 times the variance."""
    var_w = {}
    for name, sampler in [('AWRS', nb['awrs_a']), ('naive', nb['naive'])]:
        m1, m2 = nb['exact_moments'](sampler)
        assert np.allclose(m1, nb['target'])
        var_w[name] = m2.sum() - m1.sum()**2
    assert var_w['naive'] / var_w['AWRS'] == pytest.approx(2.4, abs=0.05)


def test_awrs_weights_are_at_most_one(nb):
    """'AWRS's weights are at most 1', which gives the early-stopping bound."""
    assert max(w for _, (_, w) in traces(run_awrs(nb, nb['awrs_a']))) <= 1


def test_greedy_covers_mass_faster(nb):
    """Both orders find all 196 traces, and greedy leaves less mass unexplored."""
    np.random.seed(0)
    sampled = [prob for prob, _ in traces(run_awrs(nb, nb['awrs_a']), TraceSWOR())]
    greedy = [prob for prob, _ in traces(run_awrs(nb, nb['awrs_a']), TraceGreedy())]
    assert len(sampled) == len(greedy) == 196
    assert np.all(np.cumsum(greedy) >= np.cumsum(sampled) - 1e-12)


# ---------------------------------------------------------------------------
# Strategy 3: the SMC pipeline
# ---------------------------------------------------------------------------

def test_constraint_probability(nb):
    """'Z_lm = sum(target_lm.values())  # = 0.65, not 1'"""
    assert nb['Z_lm'] == pytest.approx(0.65)


def test_sis_catches_every_bug(nb):
    """SIS with A passes; B, C, D, and locally constrained decoding fail; 162 traces."""
    samplers = {**nb['variants'], 'locally constrained': nb['locally_constrained']}
    for name, sampler in samplers.items():
        status, err, n_traces = nb['check_smc'](sampler, M=1, tau=0)
        assert (status == '✓') == (name == 'A'), name
        assert n_traces == 162


def test_locally_constrained_overweights_b(nb):
    """Locally constrained decoding 'puts too much mass on b and ba'."""
    have = defaultdict(float)
    for prob, finished in traces(lambda draw: nb['smc'](nb['lm'], 1, 0, nb['locally_constrained'], draw=draw)):
        for particle in finished:
            have[particle.string] += particle.weight * prob
    target = nb['target_lm']
    assert have[('b',)] > target[('b',)]
    assert have[('b', 'a')] > target[('b', 'a')]


@pytest.mark.timeout(300)
def test_sir_with_two_particles(nb):
    """SIR passes with A and catches D, with 'nearly 30,000' traces."""
    status, _, n_traces = nb['check_smc'](nb['variants']['A'], M=2, tau=0.75)
    assert status == '✓'
    assert 25_000 < n_traces < 31_000
    status, _, _ = nb['check_smc'](nb['variants']['D'], M=2, tau=0.75)
    assert status == '✗'


# ---------------------------------------------------------------------------
# The article's ✓ links
# ---------------------------------------------------------------------------

def test_check_marks_point_to_their_tests():
    """Each ✓ in the article links to the line where its test is defined."""
    pills = re.findall(r'<a href="[^"]*/tests/test_article\.py#L(\d+)" title="(\w+)" class="verified"', ARTICLE)
    assert len(pills) == 5
    lines = Path(__file__).read_text().splitlines()
    for lineno, name in pills:
        assert lines[int(lineno) - 1].startswith(f'def {name}('), name
