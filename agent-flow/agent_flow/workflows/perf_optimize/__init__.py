"""Performance-optimize workflow built on ``agent_flow.AgentLayer``.

The applying counterpart to perf-analyze: benchmarks a
``trtllm-serve`` baseline (an optional ``sol`` block in task.yaml then
enables a one-shot projector stage deriving the analytical
speed-of-light ceiling — ``sol_projection.md``, per the
``internal-perf-sol-analysis`` skill — that the analyzer weighs and the
reporter turns into a headroom-captured story), then iterates
optimization rounds — the
analyzer profiles the current build and ranks candidate optimizations
into ``roadmap.yaml`` by expected perf benefit; isolated optimizer/evaluator
pairs run the top pending items serially or concurrently; each evaluator gates attempts on code quality,
functionality, and measured gain vs expectation with a three-way
verdict (APPROVE / REJECT / PUSH_BACK), capturing an accept-evidence
nsys profile on each candidate-ready result. Parallel mode uses an Integrator;
serial mode accepts each approved candidate directly. The loop runs the configured round budget
(no agent decides when to stop; the orchestrator breaks early only when
the roadmap is exhausted or the optional improvement target is met),
stateless QA independently re-measures the final accepted state once —
and finally a reporter synthesizes the expected-vs-measured story into
``optimization_report.md`` / ``.html``. All eight roles run on the
Claude Code backend.

Public surface:

- :class:`PerfOptimizeWorkflow` — the orchestrator for the
  benchmarker -> (projector) -> [analyzer -> serial/parallel
  (optimizer <-> evaluator) items -> optional integrator] x rounds -> qa -> reporter loop.
- :class:`PromptBundle`, :data:`DEFAULT_PROMPTS`, and
  :func:`build_perf_optimize_prompts` — prompt bundle and helpers for
  the workflow's eight agents.
- ``STAGE_*`` constants — stage identifiers used by the checkpoint schema
  (``<workspace>/.perf_optimize_state.json``).
"""

from typing import Any

from .prompts import DEFAULT_PROMPTS, PromptBundle, build_perf_optimize_prompts
from .state import (
    STAGE_ANALYZER,
    STAGE_BENCHMARKER,
    STAGE_EVALUATOR,
    STAGE_INTEGRATOR,
    STAGE_OPTIMIZER,
    STAGE_OPTIMIZER_EVALUATOR,
    STAGE_PROJECTOR,
    STAGE_QA,
    STAGE_REPORTER,
)

__all__ = [
    "DEFAULT_PROMPTS",
    "PerfOptimizeWorkflow",
    "PromptBundle",
    "STAGE_ANALYZER",
    "STAGE_BENCHMARKER",
    "STAGE_EVALUATOR",
    "STAGE_INTEGRATOR",
    "STAGE_OPTIMIZER",
    "STAGE_OPTIMIZER_EVALUATOR",
    "STAGE_PROJECTOR",
    "STAGE_QA",
    "STAGE_REPORTER",
    "build_perf_optimize_prompts",
]


def __getattr__(name: str) -> Any:
    if name == "PerfOptimizeWorkflow":
        from .workflow import PerfOptimizeWorkflow

        return PerfOptimizeWorkflow
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
