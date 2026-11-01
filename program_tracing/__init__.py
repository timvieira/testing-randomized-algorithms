"""
program_tracing — run a randomized program once per execution trace.

A randomized program makes all of its random choices through a `draw`
function. Passing a tracer as `draw` explores every execution exactly once,
together with its probability, so properties like unbiasedness can be
checked exactly instead of estimated.

    from program_tracing import traces

    def program(draw):
        return draw([0.5, 0.5]) + draw([0.5, 0.5])

    for prob, output in traces(program):
        print(prob, output)        # 0.25 0, 0.25 1, 0.25 1, 0.25 2 (in some order)

The tracing strategy is from Shi et al. (2020), "Incremental Sampling
Without Replacement for Sequence Models."
"""

from program_tracing.tracer import Node, Tracer, TraceSWOR, TraceGreedy, traces

__all__ = ["Node", "Tracer", "TraceSWOR", "TraceGreedy", "traces"]
