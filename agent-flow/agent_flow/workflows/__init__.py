"""Multi-agent workflows built on ``agent_flow.AgentLayer``.

Each subpackage is an independently runnable workflow:

- :mod:`agent_flow.workflows.agent_team` — generic plan ↔ build harness
  (PlanDrafter ↔ PlanReviewer [↔ Human] → Coder ↔ Reviewer ↔ QA).
- :mod:`agent_flow.workflows.modeling_bringup` — domain specialization of
  ``agent_team`` for TensorRT-LLM model bring-up; only the prompt bundle
  differs from the base workflow.
- :mod:`agent_flow.workflows.perf_analyze` — read-only diagnosis of a
  ``trtllm-serve`` deployment (Benchmarker → Projector → Analyzer →
  Reporter); changes nothing, produces a bottleneck report.
- :mod:`agent_flow.workflows.perf_optimize` — the applying counterpart of
  ``perf_analyze``: it shares that workflow's task schema, prompt
  fragments and SOL projector, and adds the Optimizer ↔ Evaluator rounds,
  a stateless QA re-measurement, and a reporter.

Workflows are intentionally **not** re-exported here — import them
explicitly so the dependency is visible at the call site:

    from agent_flow.workflows.agent_team import AgentTeamWorkflow
    from agent_flow.workflows.modeling_bringup import MODELING_BRINGUP_PROMPTS
    from agent_flow.workflows.perf_analyze import PerfAnalyzeWorkflow
    from agent_flow.workflows.perf_optimize import PerfOptimizeWorkflow
"""
