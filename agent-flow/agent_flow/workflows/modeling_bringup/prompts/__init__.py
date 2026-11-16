"""Modeling-bringup prompt bundle.

Builds ``MODELING_BRINGUP_PROMPTS`` by appending each ``*_extra.py``'s
``SYSTEM_PROMPT_EXTENSION`` to the corresponding base prompt shipped in
``agent_flow.workflows.agent_team.prompts``. Empty extensions leave the
base prompt unchanged, so this module is safe to import even before any
modeling-bringup-specific guidance has been written.
"""

from __future__ import annotations

from agent_flow.workflows.agent_team.prompts import DEFAULT_PROMPTS, PromptBundle

from . import coder_extra, plan_drafter_extra, plan_reviewer_extra, qa_extra, reviewer_extra
from ._common import CONTAINER_BOOTSTRAP, TEST_COMMAND_CACHE


def build_modeling_bringup_prompts(
    *,
    include_slurm_environment: bool = False,
    replan_on_qa: bool = False,
    concurrent: bool = False,
    max_parallel: int | None = None,
) -> PromptBundle:
    """Build modeling-bringup prompts for one validated ``task.yaml``.

    Slurm/container instructions are intentionally task-scoped: they are
    appended only when the task spec contains ``slurm-environment``.

    The Stage/Goal control-flow protocol (two-level plan schema, per-Goal
    coder turns, Reviewer-owned state table, Stage-scoped QA, replan lock
    matrix) is appended only when ``replan_on_qa`` is set — it is designed
    around the post-QA replan sub-cycle, so without ``--replan-on-qa`` the
    agents run on the base flat-plan prompts.

    The concurrent-DAG protocol (PlanDrafter emits a machine-readable
    ``## Execution Graph`` block, and coder/reviewer/qa work exactly one
    assigned node per turn with NO shared ``## Stages & Goals`` table) is
    appended only when ``concurrent`` is set. ``concurrent`` and
    ``replan_on_qa`` are mutually distinct modes: ``concurrent`` takes
    precedence, so if both are passed the concurrent extensions are wired
    and the Stage/Goal ones are not.

    On the concurrent path, ``max_parallel`` (when supplied) additionally
    appends a soft "Concurrency budget" block to the PlanDrafter prompt so it
    sizes the Execution Graph against the scheduler's ``--max-parallel`` cap.
    Ignored off the concurrent path.
    """
    prompts = DEFAULT_PROMPTS.with_extensions(
        plan_drafter=plan_drafter_extra.SYSTEM_PROMPT_EXTENSION,
        plan_reviewer=plan_reviewer_extra.SYSTEM_PROMPT_EXTENSION,
        coder=coder_extra.SYSTEM_PROMPT_EXTENSION,
        reviewer=reviewer_extra.SYSTEM_PROMPT_EXTENSION,
        qa=qa_extra.SYSTEM_PROMPT_EXTENSION,
    )
    if concurrent:
        prompts = prompts.with_extensions(
            plan_drafter=plan_drafter_extra.CONCURRENT_EXTENSION,
            plan_reviewer=plan_reviewer_extra.CONCURRENT_EXTENSION,
            coder=coder_extra.CONCURRENT_EXTENSION,
            reviewer=reviewer_extra.CONCURRENT_EXTENSION,
            qa=qa_extra.CONCURRENT_EXTENSION,
        )
        # The concurrent DAG is throttled to ``--max-parallel`` at runtime, so
        # teach the PlanDrafter (only) that budget to size the Execution Graph
        # against. Injected only when a value is supplied; the CLI always passes
        # ``args.max_parallel`` (default 8) on the ``--concurrent`` path.
        if max_parallel is not None:
            prompts = prompts.with_extensions(
                plan_drafter=plan_drafter_extra.concurrent_budget_guidance(max_parallel),
            )
    elif replan_on_qa:
        prompts = prompts.with_extensions(
            plan_drafter=plan_drafter_extra.STAGE_GOAL_EXTENSION,
            plan_reviewer=plan_reviewer_extra.STAGE_GOAL_EXTENSION,
            coder=coder_extra.STAGE_GOAL_EXTENSION,
            reviewer=reviewer_extra.STAGE_GOAL_EXTENSION,
            qa=qa_extra.STAGE_GOAL_EXTENSION,
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
