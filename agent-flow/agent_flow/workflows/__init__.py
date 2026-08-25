"""Multi-agent workflows built on ``agent_flow.AgentLayer``.

Each subpackage is an independently runnable workflow:

- :mod:`agent_flow.workflows.agent_team` — generic plan ↔ build harness
  (PlanDrafter ↔ PlanReviewer [↔ Human] → Coder ↔ Reviewer ↔ QA).
- :mod:`agent_flow.workflows.modeling_bringup` — domain specialization of
  ``agent_team`` for TensorRT-LLM model bring-up; only the prompt bundle
  differs from the base workflow.

Workflows are intentionally **not** re-exported here — import them
explicitly so the dependency is visible at the call site:

    from agent_flow.workflows.agent_team import AgentTeamWorkflow
    from agent_flow.workflows.modeling_bringup import MODELING_BRINGUP_PROMPTS
"""
