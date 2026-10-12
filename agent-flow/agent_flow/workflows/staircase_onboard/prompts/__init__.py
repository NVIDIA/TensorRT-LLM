# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Staircase-onboard prompt bundle.

Six roles, each a system prompt built from a role module plus the shared
doctrine blocks in :mod:`._common`. :func:`build_staircase_onboard_prompts`
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

from ._common import decision_file_block, scope_block, source_embargo_block
from .assembler import SYSTEM_PROMPT as ASSEMBLER_SYSTEM_PROMPT
from .auditor import SYSTEM_PROMPT as AUDITOR_SYSTEM_PROMPT
from .plan_drafter import SYSTEM_PROMPT as PLAN_DRAFTER_SYSTEM_PROMPT
from .plan_reviewer import SYSTEM_PROMPT as PLAN_REVIEWER_SYSTEM_PROMPT
from .reviewer import SYSTEM_PROMPT as REVIEWER_SYSTEM_PROMPT
from .runner import SYSTEM_PROMPT as RUNNER_SYSTEM_PROMPT
from .smith import SYSTEM_PROMPT as SMITH_SYSTEM_PROMPT

if TYPE_CHECKING:  # pragma: no cover - import cycle guard for type checkers only
    from ..environment import RunEnvironment

# Appended to the Reviewer when the run is in ``full`` mode. ``light`` is the
# default and needs no extension: the base prompt already says the gates ran
# and that re-running them buys nothing.
REVIEWER_FULL_EXTENSION = """\
## This run is in `full` reviewer mode

The operator asked for independent verification rather than the default
evidence-reading review. In addition to the checks above, re-run the boot gate
yourself with the launch prefix you were given and confirm its result matches
what the orchestrator recorded.

A mismatch between your run and the recorded one is the most valuable thing
you can find here — report it as a finding about the run, not as a defect in
the diff. Still do not re-run the accuracy gate: an hour of wall-clock to
reproduce a number within its own 0.1-0.3 point session noise is not
verification, it is delay.
"""


@dataclass(frozen=True)
class PromptBundle:
    """System prompts for the six roles in :class:`StaircaseOnboardWorkflow`."""

    plan_drafter: str
    plan_reviewer: str
    smith: str
    assembler: str
    reviewer: str
    auditor: str
    # The Runner decides nothing -- it relays a submission to a scheduler the
    # orchestrator cannot reach. Listed here because every turn needs a system
    # prompt, not because it is a seventh judgement.
    runner: str = RUNNER_SYSTEM_PROMPT

    def with_extensions(
        self,
        *,
        plan_drafter: str = "",
        plan_reviewer: str = "",
        smith: str = "",
        assembler: str = "",
        reviewer: str = "",
        auditor: str = "",
        runner: str = "",
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
            auditor=_append(self.auditor, auditor),
            runner=_append(self.runner, runner),
        )

    def replace(self, **changes: str) -> "PromptBundle":
        """Return a bundle with individual prompts swapped outright."""
        return replace(self, **changes)


def _for_every_role(prompts: PromptBundle, block: str) -> PromptBundle:
    """Append one block to every role.

    Every run-scoped block goes to every role, including the ones that
    mostly read. A role that was not told the rule is a role that will
    break it — an auditor deciding to check something on a GPU reaches for
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
        auditor=block,
        runner=block,
    )


DEFAULT_PROMPTS = PromptBundle(
    plan_drafter=PLAN_DRAFTER_SYSTEM_PROMPT,
    plan_reviewer=PLAN_REVIEWER_SYSTEM_PROMPT,
    smith=SMITH_SYSTEM_PROMPT,
    assembler=ASSEMBLER_SYSTEM_PROMPT,
    reviewer=REVIEWER_SYSTEM_PROMPT,
    auditor=AUDITOR_SYSTEM_PROMPT,
    runner=RUNNER_SYSTEM_PROMPT,
)


#: Where the orchestrator snapshots the composed prompts inside the workspace.
PROMPTS_DIRNAME = "prompts"


def build_staircase_onboard_prompts(
    *,
    environment: "RunEnvironment | None" = None,
    reviewer_mode: str = "light",
    source_embargo: "Sequence[str]" = (),
    scope: str = "full",
    decision_files: bool = False,
    commit_script: "str | None" = None,
) -> PromptBundle:
    """Build the prompt bundle for one run.

    ``environment`` is the resolved launch environment from preflight;
    ``source_embargo`` and ``scope`` come from ``task.yaml``. All three are
    run-scoped facts that change what a correct action is, so all three are
    appended rather than left for a role to infer.

    ``commit_script`` is the same idea applied to git. The baseline
    ``COMMIT_DISCIPLINE`` block states the rule and relies on the agent to
    follow it; when a helper has been installed, the rule is *enforced* and
    the two committing roles are pointed at it instead.
    """
    prompts = DEFAULT_PROMPTS
    if reviewer_mode == "full":
        prompts = prompts.with_extensions(reviewer=REVIEWER_FULL_EXTENSION)
    prompts = _for_every_role(prompts, scope_block(scope))
    prompts = _for_every_role(prompts, source_embargo_block(tuple(source_embargo)))
    if commit_script:
        # Only to the two roles that commit. A role handed a commit recipe
        # it has no business using is a role that will find a reason to.
        from ..committer import commit_instruction

        block = commit_instruction(commit_script)
        prompts = prompts.with_extensions(smith=block, assembler=block)
    if decision_files:
        # Per-role rather than shared: each role's enum is different, and a
        # role shown another role's options will eventually pick one.
        from ..progress import DECISIONS_BY_AGENT, EXTRA_KEYS_BY_AGENT

        prompts = prompts.with_extensions(
            **{
                role: decision_file_block(
                    role, DECISIONS_BY_AGENT[role], EXTRA_KEYS_BY_AGENT.get(role, ())
                )
                for role in DECISIONS_BY_AGENT
            }
        )
    if environment is None:
        return prompts
    return _for_every_role(prompts, environment.describe())


__all__ = [
    "ASSEMBLER_SYSTEM_PROMPT",
    "AUDITOR_SYSTEM_PROMPT",
    "DEFAULT_PROMPTS",
    "PLAN_DRAFTER_SYSTEM_PROMPT",
    "PLAN_REVIEWER_SYSTEM_PROMPT",
    "PROMPTS_DIRNAME",
    "PromptBundle",
    "REVIEWER_FULL_EXTENSION",
    "REVIEWER_SYSTEM_PROMPT",
    "SMITH_SYSTEM_PROMPT",
    "build_staircase_onboard_prompts",
]
