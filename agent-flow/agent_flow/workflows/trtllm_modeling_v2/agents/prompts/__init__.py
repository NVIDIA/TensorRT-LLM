# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""trtllm_modeling_v2 prompt bundle.

Every task's agent is a worker (:mod:`.worker`): one system prompt, which the
roles a task can be dispatched to (``surveyor``, ``kernel_writer``, ``assembler``,
``tuner``) end each the way its turn ends. What a task asks is said by its
tags, whose sections its dispatch carries (:mod:`.tags`). The planner has a
prompt of its own.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence

from ...infra.committer import commit_instruction
from ...run.progress import DECISIONS_BY_AGENT, EXTRA_KEYS_BY_AGENT
from ._common import decision_file_block, notes_block, scope_block, source_embargo_block
from .planner import SYSTEM_PROMPT as PLANNER_SYSTEM_PROMPT
from .tuner import SYSTEM_PROMPT as TUNER_SYSTEM_PROMPT
from .worker import role_prompt

if TYPE_CHECKING:  # pragma: no cover - import cycle guard for type checkers only
    from ...infra.environment import RunEnvironment


@dataclass(frozen=True)
class PromptBundle:
    """System prompts for the roles in :class:`TrtllmModelingV2Workflow`."""

    surveyor: str
    kernel_writer: str
    assembler: str
    tuner: str = TUNER_SYSTEM_PROMPT
    planner: str = PLANNER_SYSTEM_PROMPT

    def with_extensions(
        self,
        *,
        surveyor: str = "",
        kernel_writer: str = "",
        assembler: str = "",
        tuner: str = "",
        planner: str = "",
    ) -> "PromptBundle":
        """Return a new bundle with each non-empty extension appended after one blank line."""

        def _append(base: str, extra: str) -> str:
            if not extra.strip():
                return base
            return base.rstrip() + "\n\n" + extra

        return PromptBundle(
            surveyor=_append(self.surveyor, surveyor),
            kernel_writer=_append(self.kernel_writer, kernel_writer),
            assembler=_append(self.assembler, assembler),
            tuner=_append(self.tuner, tuner),
            planner=_append(self.planner, planner),
        )


def _for_every_role(prompts: PromptBundle, block: str) -> PromptBundle:
    """Append one block to every role, including the ones that mostly read."""
    if not block.strip():
        return prompts
    return prompts.with_extensions(
        surveyor=block,
        kernel_writer=block,
        assembler=block,
        tuner=block,
        planner=block,
    )


_ROLE_PROMPTS = PromptBundle(
    surveyor=role_prompt("surveyor"),
    kernel_writer=role_prompt("kernel_writer"),
    assembler=role_prompt("assembler"),
    tuner=TUNER_SYSTEM_PROMPT,
    planner=PLANNER_SYSTEM_PROMPT,
)

DEFAULT_PROMPTS = _ROLE_PROMPTS.with_extensions(
    **{
        role: decision_file_block(role, DECISIONS_BY_AGENT[role], EXTRA_KEYS_BY_AGENT.get(role, ()))
        for role in DECISIONS_BY_AGENT
        # The tuner writes the file its request names, which its own prompt describes.
        if role != "tuner"
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
    ``source_embargo``, ``scope`` and ``notes`` come from ``task.yaml``.
    ``commit_script`` is the installed commit helper; the two committing
    roles are told to commit through it and nothing else.
    """
    prompts = DEFAULT_PROMPTS
    prompts = _for_every_role(prompts, scope_block(scope))
    prompts = _for_every_role(prompts, source_embargo_block(tuple(source_embargo)))
    prompts = _for_every_role(prompts, notes_block(notes))
    if commit_script:
        block = commit_instruction(commit_script)
        prompts = prompts.with_extensions(kernel_writer=block, assembler=block)
    if environment is None:
        return prompts
    return _for_every_role(prompts, environment.describe())


__all__ = [
    "DEFAULT_PROMPTS",
    "PROMPTS_DIRNAME",
    "PromptBundle",
    "build_trtllm_modeling_v2_prompts",
]
