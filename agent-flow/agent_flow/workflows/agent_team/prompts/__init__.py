from dataclasses import dataclass

from .coder import SYSTEM_PROMPT as CODER_SYSTEM_PROMPT
from .mcp_tools import MCP_TOOLS_EXTENSIONS
from .plan_drafter import SYSTEM_PROMPT as PLAN_DRAFTER_SYSTEM_PROMPT
from .plan_reviewer import SYSTEM_PROMPT as PLAN_REVIEWER_SYSTEM_PROMPT
from .qa import SYSTEM_PROMPT as QA_SYSTEM_PROMPT
from .reviewer import SYSTEM_PROMPT as REVIEWER_SYSTEM_PROMPT


@dataclass(frozen=True)
class PromptBundle:
    """System prompts for the five agents in ``AgentTeamWorkflow``.

    Pass a custom bundle to ``AgentTeamWorkflow(..., prompts=...)`` to swap
    or extend the default prompts; use ``with_extensions`` to derive a
    bundle that appends domain-specific guidance to the defaults.

    Bundles are **transport-neutral**: they describe what each role reads
    and records, not the mechanism that moves it. ``AgentTeamWorkflow``
    appends :data:`~.mcp_tools.MCP_TOOLS_EXTENSIONS` on top of whatever
    bundle it is given when the run has in-process MCP tools enabled, and
    omits them under ``--no-mcp-tools``. Domain bundles therefore must not
    name an MCP tool in their own extensions.
    """

    plan_drafter: str
    plan_reviewer: str
    coder: str
    reviewer: str
    qa: str

    def with_extensions(
        self,
        *,
        plan_drafter: str = "",
        plan_reviewer: str = "",
        coder: str = "",
        reviewer: str = "",
        qa: str = "",
    ) -> "PromptBundle":
        """Return a new bundle with each non-empty extension appended.

        Empty / whitespace-only extensions leave the corresponding base
        prompt unchanged. Non-empty extensions are joined to the base with
        a single blank line separator.
        """

        def _append(base: str, extra: str) -> str:
            if not extra.strip():
                return base
            return base.rstrip() + "\n\n" + extra

        return PromptBundle(
            plan_drafter=_append(self.plan_drafter, plan_drafter),
            plan_reviewer=_append(self.plan_reviewer, plan_reviewer),
            coder=_append(self.coder, coder),
            reviewer=_append(self.reviewer, reviewer),
            qa=_append(self.qa, qa),
        )


DEFAULT_PROMPTS = PromptBundle(
    plan_drafter=PLAN_DRAFTER_SYSTEM_PROMPT,
    plan_reviewer=PLAN_REVIEWER_SYSTEM_PROMPT,
    coder=CODER_SYSTEM_PROMPT,
    reviewer=REVIEWER_SYSTEM_PROMPT,
    qa=QA_SYSTEM_PROMPT,
)

__all__ = [
    "CODER_SYSTEM_PROMPT",
    "DEFAULT_PROMPTS",
    "MCP_TOOLS_EXTENSIONS",
    "PLAN_DRAFTER_SYSTEM_PROMPT",
    "PLAN_REVIEWER_SYSTEM_PROMPT",
    "PromptBundle",
    "QA_SYSTEM_PROMPT",
    "REVIEWER_SYSTEM_PROMPT",
]
