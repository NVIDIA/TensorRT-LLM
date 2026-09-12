"""Staircase-bringup prompt bundle.

Four roles extend the ``agent_team`` defaults the same way ``modeling_bringup``
does. **QA is different**: its base prompt is replaced wholesale, because
``EVALUATION_CRITERIA`` — names and weights — lives in that base prompt and
``PromptBundle.with_extensions`` can only append. Inheriting it would leave a
Performance dimension (weight 1.5) scoring something staircase does not gate
on, and since PlanDrafter may not return ``DONE`` below ``min_score``, an
un-scoreable dimension deadlocks the run. See ``qa.py``.

Empty extensions leave the corresponding base prompt unchanged, so this module
is safe to import before any role guidance has been written.
"""

from __future__ import annotations

from agent_flow.workflows.agent_team.prompts import DEFAULT_PROMPTS, PromptBundle

from . import coder_extra, plan_drafter_extra, plan_reviewer_extra, qa_extra, reviewer_extra
from . import qa as staircase_qa
from ._common import get_staircase_skill_invocation, slurm_blocks


def build_staircase_prompts(
    *,
    include_slurm_environment: bool = False,
    replan_on_qa: bool = True,
) -> PromptBundle:
    """Build staircase-bringup prompts for one validated ``task.yaml``.

    Slurm/container instructions are task-scoped: appended only when the task
    spec carries a ``slurm-environment`` section.

    ``replan_on_qa`` defaults to **True** here, where ``modeling_bringup``
    defaults it to False. The Stage/Goal protocol is what carries staircase's
    one-op-per-Goal atomicity, the ``BLOCKER:`` hard conjunction that lets a
    Goal close as ``[Failed]`` only when the Coder declares it *and* the
    Reviewer independently confirms, and the gap-fix Stage insert that stands
    in for the old orchestrator's mid-run catalog dispatch. Without the flag
    none of those blocks reach any agent and the run degrades to a flat plan.
    """
    # QA's base prompt is staircase's own; the other four come from agent_team.
    base = PromptBundle(
        plan_drafter=DEFAULT_PROMPTS.plan_drafter,
        plan_reviewer=DEFAULT_PROMPTS.plan_reviewer,
        coder=DEFAULT_PROMPTS.coder,
        reviewer=DEFAULT_PROMPTS.reviewer,
        qa=staircase_qa.SYSTEM_PROMPT,
    )

    prompts = base.with_extensions(
        plan_drafter=plan_drafter_extra.SYSTEM_PROMPT_EXTENSION,
        plan_reviewer=plan_reviewer_extra.SYSTEM_PROMPT_EXTENSION,
        coder=coder_extra.SYSTEM_PROMPT_EXTENSION,
        reviewer=reviewer_extra.SYSTEM_PROMPT_EXTENSION,
        qa=qa_extra.SYSTEM_PROMPT_EXTENSION,
    )

    # Injected here rather than baked into a module-scope extension: the probe
    # spawns a session per backend, and only the roles that run gate commands
    # need the block. Returns "" while no staircase skill is installed, and
    # with_extensions no-ops on empty, so today this changes nothing.
    skill_block = get_staircase_skill_invocation()
    if skill_block:
        prompts = prompts.with_extensions(
            coder=skill_block,
            reviewer=skill_block,
            qa=skill_block,
        )

    if replan_on_qa:
        prompts = prompts.with_extensions(
            plan_drafter=plan_drafter_extra.STAGE_GOAL_EXTENSION,
            plan_reviewer=plan_reviewer_extra.STAGE_GOAL_EXTENSION,
            coder=coder_extra.STAGE_GOAL_EXTENSION,
            reviewer=reviewer_extra.STAGE_GOAL_EXTENSION,
            qa=qa_extra.STAGE_GOAL_EXTENSION,
        )

    if not include_slurm_environment:
        return prompts

    container_bootstrap, test_command_cache = slurm_blocks()
    build_phase_slurm = "\n".join([container_bootstrap, test_command_cache])
    return prompts.with_extensions(
        plan_drafter=container_bootstrap,
        plan_reviewer=container_bootstrap,
        coder=build_phase_slurm,
        reviewer=build_phase_slurm,
        qa=build_phase_slurm,
    )


STAIRCASE_PROMPTS: PromptBundle = build_staircase_prompts()

__all__ = ["STAIRCASE_PROMPTS", "build_staircase_prompts"]
