from dataclasses import dataclass

from .coder import SYSTEM_PROMPT as CODER_SYSTEM_PROMPT
from .reviewer import SYSTEM_PROMPT as REVIEWER_SYSTEM_PROMPT
from .sourcing import SYSTEM_PROMPT as SOURCING_SYSTEM_PROMPT


@dataclass(frozen=True)
class PromptBundle:
    """System prompts for the two roles in ``PrReviewWorkflow``.

    The same two prompts are reused across both stages — they are role-based
    (Reviewer / Coder), not model-based — so swapping the backend per stage
    does not require new prompts. Pass a custom bundle to
    ``PrReviewWorkflow(..., prompts=...)`` to swap or extend them; use
    ``with_extensions`` to derive a bundle that appends domain-specific
    guidance to the defaults.
    """

    reviewer: str
    coder: str

    def with_extensions(
        self,
        *,
        reviewer: str = "",
        coder: str = "",
    ) -> "PromptBundle":
        """Return a new bundle with each non-empty extension appended.

        Empty / whitespace-only extensions leave the corresponding base prompt
        unchanged. Non-empty extensions are joined to the base with a single
        blank line separator.
        """

        def _append(base: str, extra: str) -> str:
            if not extra.strip():
                return base
            return base.rstrip() + "\n\n" + extra

        return PromptBundle(
            reviewer=_append(self.reviewer, reviewer),
            coder=_append(self.coder, coder),
        )


DEFAULT_PROMPTS = PromptBundle(
    reviewer=REVIEWER_SYSTEM_PROMPT,
    coder=CODER_SYSTEM_PROMPT,
)

__all__ = [
    "CODER_SYSTEM_PROMPT",
    "DEFAULT_PROMPTS",
    "PromptBundle",
    "REVIEWER_SYSTEM_PROMPT",
    "SOURCING_SYSTEM_PROMPT",
]
