"""Multi-agent team workflow built on ``agent_flow.AgentLayer``.

Public surface:

- :class:`AgentTeamWorkflow` — the orchestrator that drives the plan and
  build phases (drafter ↔ reviewer ↔ human → coder ↔ reviewer ↔ qa).
- :class:`PromptBundle` and :data:`DEFAULT_PROMPTS` — the extension point
  used by domain-specific workflows (e.g.
  ``agent_flow.workflows.modeling_bringup``) to layer additional guidance
  onto the base system prompts.
- ``STAGE_*`` constants — stage identifiers used by the checkpoint
  schema (``<workspace>/.agent_team_state.json``).

Tool factories (``build_progress_tools``, ``build_status_tools``) and
checkpoint helpers (``load_state``, ``save_state``, ``WorkflowState``)
live in their respective submodules and are imported on demand:

    from agent_flow.workflows.agent_team.progress import build_progress_tools
    from agent_flow.workflows.agent_team.state import WorkflowState
"""
from typing import Any

from .prompts import DEFAULT_PROMPTS, PromptBundle
from .state import (STAGE_CODER, STAGE_PLAN_DRAFTER, STAGE_PLAN_HUMAN,
                    STAGE_PLAN_REVIEWER, STAGE_QA, STAGE_REVIEWER)

__all__ = [
    "AgentTeamWorkflow",
    "DEFAULT_PROMPTS",
    "PromptBundle",
    "STAGE_CODER",
    "STAGE_PLAN_DRAFTER",
    "STAGE_PLAN_HUMAN",
    "STAGE_PLAN_REVIEWER",
    "STAGE_QA",
    "STAGE_REVIEWER",
]


def __getattr__(name: str) -> Any:
    if name == "AgentTeamWorkflow":
        from .workflow import AgentTeamWorkflow

        return AgentTeamWorkflow
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
