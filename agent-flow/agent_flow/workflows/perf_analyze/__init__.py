"""Performance-analyze workflow built on ``agent_flow.AgentLayer``.

Serves a model checkpoint with ``trtllm-serve``, benchmarks and profiles
it with ``benchmark_serving.py`` (nsys + torch profiler), and synthesizes
a report whose headline is the main performance bottleneck. When the task
spec carries a ``sol`` block, a projector stage additionally derives an
analytical speed-of-light (SOL) ceiling — following the
``internal-perf-sol-analysis`` skill — between the benchmarker and the
analyzer.
All roles run on the Claude Code backend.

Public surface:

- :class:`PerfAnalyzeWorkflow` — the orchestrator for the
  benchmarker -> projector -> analyzer -> reporter pipeline (the
  projector stage is conditional).
- :class:`PromptBundle`, :data:`DEFAULT_PROMPTS`, and
  :func:`build_perf_analyze_prompts` — prompt bundle and helpers
  for the workflow's agents.
- ``STAGE_*`` constants — stage identifiers used by the checkpoint schema
  (``<workspace>/.perf_analyze_state.json``).
"""

from typing import Any

from .prompts import DEFAULT_PROMPTS, PromptBundle, build_perf_analyze_prompts
from .state import STAGE_ANALYZER, STAGE_BENCHMARKER, STAGE_PROJECTOR, STAGE_REPORTER

__all__ = [
    "DEFAULT_PROMPTS",
    "PerfAnalyzeWorkflow",
    "PromptBundle",
    "STAGE_BENCHMARKER",
    "STAGE_ANALYZER",
    "STAGE_PROJECTOR",
    "STAGE_REPORTER",
    "build_perf_analyze_prompts",
]


def __getattr__(name: str) -> Any:
    if name == "PerfAnalyzeWorkflow":
        from .workflow import PerfAnalyzeWorkflow

        return PerfAnalyzeWorkflow
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
