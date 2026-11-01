import numpy as np


class Node:
    """A node in the execution trie."""
    def __init__(self, idx, mass, parent, token=None):
        self.idx = idx
        self.mass = mass          # remaining probability mass
        self.parent = parent
        self.token = token        # label for visualization
        self.child_masses = None  # set on first visit
        self.keys = None          # outcomes of the choice made here
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
            vals = np.array([p[k] for k in keys], dtype=float)
        else:
            p = np.asarray(p, dtype=float)
            keys = list(range(len(p)))
            vals = p
        vals = vals / vals.sum()
        cur = self.cur

        if cur.child_masses is None:
            cur.child_masses = cur.mass * vals
            cur.keys = keys
            cur.context = context
        elif keys != cur.keys:
            raise ValueError('the program made a different choice at the same point '
                             'of two traces; it must be deterministic given its draws')

        a = self.choose(cur)
        if a not in cur.active_children:
            cur.active_children[a] = Node(
                idx=a, mass=cur.child_masses[a], parent=cur,
                token=keys[a],
            )
        self.cur = cur.active_children[a]
        return keys[a]

    def choose(self, node):
        """Pick which child to explore next."""
        return node.sample()


class TraceSWOR(Tracer):
    """
    Sampling without replacement meets program tracing.

    Each execution trace is explored exactly once. After a trace
    completes, its probability mass is zeroed out and propagated
    up the trie, ensuring the next trace is fresh.
    """
    def __enter__(self):
        self.cur = self.root
        return self

    def __exit__(self, *args):
        self.cur.mass = 0
        self.cur.update()


class TraceGreedy(TraceSWOR):
    """Always descend into the child with the most unexplored mass."""
    def choose(self, node):
        return int(np.argmax(node.child_masses))


def traces(program, tracer=None):
    """
    Run `program(draw)` once per execution trace.

    Yields (probability, output) pairs; the probabilities sum to one. Pass a
    `TraceGreedy()` as `tracer` to explore the most probable parts of the
    trie first. Stopping early leaves the unexplored mass in `tracer.root.mass`.
    """
    tracer = TraceSWOR() if tracer is None else tracer
    while tracer.root.mass > 0:
        with tracer:
            output = program(tracer)
            yield tracer.cur.mass, output
