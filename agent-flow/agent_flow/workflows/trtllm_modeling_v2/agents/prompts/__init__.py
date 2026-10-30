# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""trtllm_modeling_v2 prompt bundle.

Eight roles, each a system prompt built from a role module plus the shared
doctrine blocks in :mod:`._common`. :func:`build_trtllm_modeling_v2_prompts`
scopes a bundle to one run by appending the resolved launch environment —
the block that keeps agents from rediscovering how to submit a job — and any
reviewer-mode guidance.

The bundle is frozen and extended by deriving a new one
(:meth:`PromptBundle.with_extensions`), mirroring ``agent_team``'s contract,
so a caller can layer extra guidance without mutating the shared default.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Sequence

from ...infra.committer import commit_instruction
from ...run.progress import DECISIONS_BY_AGENT, EXTRA_KEYS_BY_AGENT
from ._common import decision_file_block, notes_block, scope_block, source_embargo_block
from .assembler import SYSTEM_PROMPT as ASSEMBLER_SYSTEM_PROMPT
from .improver import SYSTEM_PROMPT as IMPROVER_SYSTEM_PROMPT
from .plan_drafter import SYSTEM_PROMPT as PLAN_DRAFTER_SYSTEM_PROMPT
from .plan_reviewer import SYSTEM_PROMPT as PLAN_REVIEWER_SYSTEM_PROMPT
from .planner import SYSTEM_PROMPT as PLANNER_SYSTEM_PROMPT
from .reviewer import SYSTEM_PROMPT as REVIEWER_SYSTEM_PROMPT
from .smith import SYSTEM_PROMPT as SMITH_SYSTEM_PROMPT
from .tuner import SYSTEM_PROMPT as TUNER_SYSTEM_PROMPT

if TYPE_CHECKING:  # pragma: no cover - import cycle guard for type checkers only
    from ...infra.environment import RunEnvironment


@dataclass(frozen=True)
class PromptBundle:
    """System prompts for the roles in :class:`TrtllmModelingV2Workflow`."""

    plan_drafter: str
    plan_reviewer: str
    smith: str
    assembler: str
    reviewer: str
    # The tune stage's performance engineer: one candidate per turn.
    tuner: str = TUNER_SYSTEM_PROMPT
    # Decides the task graph; never does the work.
    planner: str = PLANNER_SYSTEM_PROMPT
    # Turns failures into scoped lessons later dispatches are shown.
    improver: str = IMPROVER_SYSTEM_PROMPT

    def with_extensions(
        self,
        *,
        plan_drafter: str = "",
        plan_reviewer: str = "",
        smith: str = "",
        assembler: str = "",
        reviewer: str = "",
        tuner: str = "",
        planner: str = "",
        improver: str = "",
    ) -> "PromptBundle":
        """Return a new bundle with each non-empty extension appended.

        Empty / whitespace-only extensions leave the corresponding base
        prompt unchanged. Non-empty ones are joined to the base with a
        single blank-line separator.
        """

        def _append(base: str, extra: str) -> str:
            if not extra.strip():
                return base
            return base.rstrip() + "\n\n" + extra

        return PromptBundle(
            plan_drafter=_append(self.plan_drafter, plan_drafter),
            plan_reviewer=_append(self.plan_reviewer, plan_reviewer),
            smith=_append(self.smith, smith),
            assembler=_append(self.assembler, assembler),
            reviewer=_append(self.reviewer, reviewer),
            tuner=_append(self.tuner, tuner),
            planner=_append(self.planner, planner),
            improver=_append(self.improver, improver),
        )

    def replace(self, **changes: str) -> "PromptBundle":
        """Return a bundle with individual prompts swapped outright."""
        return replace(self, **changes)


def _for_every_role(prompts: PromptBundle, block: str) -> PromptBundle:
    """Append one block to every role.

    Every run-scoped block goes to every role, including the ones that
    mostly read. A role that was not told the rule is a role that will
    break it — a reviewer deciding to check something on a GPU reaches for
    the same launcher as everyone else, and a reviewer not told about an
    embargo will helpfully open the embargoed file to compare against.
    """
    if not block.strip():
        return prompts
    return prompts.with_extensions(
        plan_drafter=block,
        plan_reviewer=block,
        smith=block,
        assembler=block,
        reviewer=block,
        tuner=block,
        planner=block,
        improver=block,
    )


_ROLE_PROMPTS = PromptBundle(
    plan_drafter=PLAN_DRAFTER_SYSTEM_PROMPT,
    plan_reviewer=PLAN_REVIEWER_SYSTEM_PROMPT,
    smith=SMITH_SYSTEM_PROMPT,
    assembler=ASSEMBLER_SYSTEM_PROMPT,
    reviewer=REVIEWER_SYSTEM_PROMPT,
    tuner=TUNER_SYSTEM_PROMPT,
    planner=PLANNER_SYSTEM_PROMPT,
    improver=IMPROVER_SYSTEM_PROMPT,
)

#: The roles whose decision file is the one the dispatch reserves. The tuner
#: writes the file its request names, which its own prompt describes.
DECISION_FILE_ROLES: tuple[str, ...] = tuple(r for r in DECISIONS_BY_AGENT if r != "tuner")

# Per role rather than shared: each role's enum is different, and a role
# shown another role's options will eventually pick one.
DEFAULT_PROMPTS = _ROLE_PROMPTS.with_extensions(
    **{
        role: decision_file_block(role, DECISIONS_BY_AGENT[role], EXTRA_KEYS_BY_AGENT.get(role, ()))
        for role in DECISION_FILE_ROLES
    }
)


#: Where the orchestrator snapshots the composed prompts inside the workspace.
PROMPTS_DIRNAME = "prompts"


def build_trtllm_modeling_v2_prompts(
    *,
    environment: "RunEnvironment | None" = None,
    source_embargo: "Sequence[str]" = (),
    scope: str = "full",
    commit_script: "str | None" = None,
    notes: str = "",
) -> PromptBundle:
    """Build the prompt bundle for one run.

    ``environment`` is the resolved launch environment from preflight;
    ``source_embargo``, ``scope`` and ``notes`` come from ``task.yaml``. They
    are run-scoped facts that change what a correct action is, so they are
    appended rather than left for a role to infer.

    ``commit_script`` is the installed commit helper; the two committing
    roles are told to commit through it and nothing else.
    """
    prompts = DEFAULT_PROMPTS
    prompts = _for_every_role(prompts, scope_block(scope))
    prompts = _for_every_role(prompts, source_embargo_block(tuple(source_embargo)))
    prompts = _for_every_role(prompts, notes_block(notes))
    if commit_script:
        # Only to the two roles that commit. A role handed a commit recipe
        # it has no business using is a role that will find a reason to.

        block = commit_instruction(commit_script)
        prompts = prompts.with_extensions(smith=block, assembler=block)
    if environment is None:
        return prompts
    return _for_every_role(prompts, environment.describe())


__all__ = [
    "ASSEMBLER_SYSTEM_PROMPT",
    "DEFAULT_PROMPTS",
    "PLAN_DRAFTER_SYSTEM_PROMPT",
    "PLAN_REVIEWER_SYSTEM_PROMPT",
    "PROMPTS_DIRNAME",
    "PromptBundle",
    "REVIEWER_SYSTEM_PROMPT",
    "SMITH_SYSTEM_PROMPT",
    "build_trtllm_modeling_v2_prompts",
]
