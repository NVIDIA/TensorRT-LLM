"""Modeling-bringup prompt bundle.

Builds ``MODELING_BRINGUP_PROMPTS`` by appending each ``*_extra.py``'s
``SYSTEM_PROMPT_EXTENSION`` to the corresponding base prompt shipped in
``agent_flow.workflows.agent_team.prompts``. Empty extensions leave the
base prompt unchanged, so this module is safe to import even before any
modeling-bringup-specific guidance has been written.
"""
from __future__ import annotations

from agent_flow.workflows.agent_team.prompts import (DEFAULT_PROMPTS,
                                                     PromptBundle)

from . import (coder_extra, plan_drafter_extra, plan_reviewer_extra, qa_extra,
               reviewer_extra)
from ._common import CONTAINER_BOOTSTRAP, TEST_COMMAND_CACHE


def build_modeling_bringup_prompts(*,
                                   include_slurm_environment: bool = False
                                   ) -> PromptBundle:
    """Build modeling-bringup prompts for one validated ``task.yaml``.

    Slurm/container instructions are intentionally task-scoped: they are
    appended only when the task spec contains ``slurm-environment``.
    """
    prompts = DEFAULT_PROMPTS.with_extensions(
        plan_drafter=plan_drafter_extra.SYSTEM_PROMPT_EXTENSION,
        plan_reviewer=plan_reviewer_extra.SYSTEM_PROMPT_EXTENSION,
        coder=coder_extra.SYSTEM_PROMPT_EXTENSION,
        reviewer=reviewer_extra.SYSTEM_PROMPT_EXTENSION,
        qa=qa_extra.SYSTEM_PROMPT_EXTENSION,
    )
    if not include_slurm_environment:
        return prompts

    build_phase_slurm = "\n".join([CONTAINER_BOOTSTRAP, TEST_COMMAND_CACHE])
    return prompts.with_extensions(
        plan_drafter=CONTAINER_BOOTSTRAP,
        plan_reviewer=CONTAINER_BOOTSTRAP,
        coder=build_phase_slurm,
        reviewer=build_phase_slurm,
        qa=build_phase_slurm,
    )


MODELING_BRINGUP_PROMPTS: PromptBundle = build_modeling_bringup_prompts()

__all__ = ["MODELING_BRINGUP_PROMPTS", "build_modeling_bringup_prompts"]
