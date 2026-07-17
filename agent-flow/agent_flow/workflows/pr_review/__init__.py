"""Two-stage cross-model PR/MR review workflow built on ``agent_flow.AgentLayer``.

Public surface:

- :class:`PrReviewWorkflow` — the orchestrator that drives the two review
  stages (stage 1: Claude Code reviews ↔ Codex addresses; stage 2: Codex
  reviews ↔ Claude Code addresses), each looping until the pair converges or
  the coder stands firm on a push-back.
- :class:`PromptBundle` and :data:`DEFAULT_PROMPTS` — the extension point for
  layering domain-specific guidance onto the base reviewer/coder prompts.
- ``STAGE_*`` constants — stage identifiers used by the checkpoint schema
  (``<workspace>/.pr_review_state.json``).

Tool factories (``build_progress_tools``, ``build_discussion_tools``),
checkpoint helpers (``load_state``, ``save_state``, ``WorkflowState``), and the
``vcs`` helpers live in their respective submodules and are imported on demand:

    from agent_flow.workflows.pr_review.progress import build_progress_tools
    from agent_flow.workflows.pr_review.state import WorkflowState
"""

from typing import Any

from .prompts import DEFAULT_PROMPTS, PromptBundle
from .state import STAGE_S1_CODER, STAGE_S1_REVIEWER, STAGE_S2_CODER, STAGE_S2_REVIEWER

__all__ = [
    "DEFAULT_PROMPTS",
    "PrReviewWorkflow",
    "PromptBundle",
    "STAGE_S1_CODER",
    "STAGE_S1_REVIEWER",
    "STAGE_S2_CODER",
    "STAGE_S2_REVIEWER",
]


def __getattr__(name: str) -> Any:
    if name == "PrReviewWorkflow":
        from .workflow import PrReviewWorkflow

        return PrReviewWorkflow
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
