# Ideas for Testing Randomized Algorithms

Blog post on testing randomized algorithms: Monte Carlo convergence tests, and program tracing, which runs a randomized program once per execution trace so that properties like unbiasedness can be checked exactly.

**[Read the post](https://timvieira.github.io/testing-randomized-algorithms/)**

This repository also contains `program_tracing`, a small library implementing the tracer from the post.

## Installation

Install from a local clone:

```bash
pip install .              # the library (requires NumPy)
pip install -e '.[dev]'    # plus tests, lint, and the article's notebook dependencies
```

Requires Python 3.9+ and NumPy.

## Usage

Write the randomized program so that every random choice goes through a `draw` function. Passing a tracer as `draw` visits each execution exactly once, with its probability:

```python
from program_tracing import traces

def program(draw):
    flips = [draw([0.7, 0.3]) for _ in range(3)]   # draw(weights) returns an index
    return sum(flips)

dist = {}
for prob, heads in traces(program):
    dist[heads] = dist.get(heads, 0) + prob
print(dist)   # exact binomial(3, 0.3) probabilities
```

`draw` also accepts dict-valued distributions (`draw({'a': 0.2, 'b': 0.8})` returns a key) and an optional `context` label.

| Name | Description |
|---|---|
| `traces(program, tracer=None)` | Run `program(draw)` once per trace; yield `(probability, output)` |
| `TraceSWOR()` | Explore traces in random order, without replacement (the default) |
| `TraceGreedy()` | Always descend into the branch with the most unexplored mass |
| `Tracer` | Base class; override `choose(node)` for another exploration order |

Stopping early is allowed: the probability of the traces not yet explored is `tracer.root.mass`. The program must be deterministic given its draws, and every execution must terminate; the tracer raises an error if it notices a program making different choices at the same point of two traces.

## Development

```bash
make install    # pip install -e ".[dev]"
make test       # library tests and tests of the article's claims
make lint       # ruff
make coverage   # tests with an HTML coverage report
make notebook   # re-execute the article's notebook
make dev        # build, watch, and serve the article (needs the blog CLI)
make deploy     # run the tests, then build and deploy to GitHub Pages
```

The article is built with the [blog build system](https://github.com/timvieira/blog).

## License

MIT
