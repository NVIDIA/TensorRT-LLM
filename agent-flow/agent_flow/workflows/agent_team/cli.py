from __future__ import annotations

import argparse
from pathlib import Path

from .prompts import PromptBundle
from .state import STATE_FILENAME
from .workflow import AgentTeamWorkflow


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the planner-module -> coder -> reviewer -> qa "
        "workflow.")
    parser.add_argument("--task",
                        type=Path,
                        required=True,
                        help="Path to the task YAML file. The file is copied "
                        "verbatim into `<workspace>/task.yaml` and read by "
                        "every agent. The agent_team workflow itself "
                        "imposes no schema on the YAML; wrappers (e.g. "
                        "modeling_bringup) may enforce their own.")
    parser.add_argument("--workspace",
                        type=Path,
                        default=Path("workspace/agent-team"),
                        help="Workspace directory for shared state files "
                        "(task.yaml, plan.md, acceptance-criteria.md, "
                        "progress.yaml, status.md).")
    parser.add_argument("--num-iterations",
                        type=int,
                        default=100,
                        help="Maximum number of coder/reviewer/qa iterations.")
    parser.add_argument(
        "--coder-context-reset-interval",
        type=int,
        default=2,
        help="Recycle the coder's persistent session every N iterations. "
        "Set to 0 to disable.")
    parser.add_argument(
        "--reviewer-context-reset-interval",
        type=int,
        default=2,
        help="Recycle the reviewer's persistent session every N iterations. "
        "Set to 0 to disable.")
    parser.add_argument(
        "--min-score",
        type=float,
        default=8.0,
        help="Minimum QA weighted_score (0-10) for an APPROVE to terminate "
        "the workflow. An APPROVE below this floor is downgraded to a "
        "loop-back. Set to 0 to disable the gate (bool decision alone "
        "terminates).")
    parser.add_argument(
        "--plan-human-review",
        dest="plan_human_review_enabled",
        action="store_true",
        help="Enable the human-review stage of the plan phase. After "
        "the PlanReviewer APPROVEs the plan, the PlanDrafter pauses to "
        "ask the human for sign-off via ``ask_human`` before the build "
        "phase begins. Off by default — the plan phase runs unattended "
        "and PlanReviewer APPROVE flows straight into the build phase.")
    parser.set_defaults(plan_human_review_enabled=False)
    parser.add_argument(
        "--build-human-review",
        dest="build_human_review_enabled",
        action="store_true",
        help="Enable the Coder's ``ask_human`` escape hatch during the "
        "build phase. The Coder may pause mid-iteration to ask the "
        "human a question via stdin and integrate the reply. Off by "
        "default — the build phase is expected to be unattended once "
        "the plan is approved; only enable this when the task involves "
        "environment facts or user-only information the Coder cannot "
        "deduce.")
    parser.set_defaults(build_human_review_enabled=False)
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Wipe the workspace checkpoint and managed files "
        f"({STATE_FILENAME}, plan.md, acceptance-criteria.md, "
        "progress.yaml, status.md) and start fresh. Without this flag "
        "the workflow resumes from the checkpoint when one is present "
        "in the workspace, and starts fresh otherwise.")
    parser.add_argument(
        "--plan",
        default=None,
        help="Use a user-supplied plan as the plan-phase plan output. "
        "Value is either the plan text or a path to a file containing "
        "it; either way it is written to `<workspace>/plan.md`. "
        "Combined with --acceptance-criteria, both files are materialized "
        "and the plan phase is skipped entirely (no PlanDrafter / "
        "PlanReviewer / human review). When --acceptance-criteria is "
        "omitted, the plan phase still runs so the PlanDrafter can "
        "generate `acceptance-criteria.md`. Ignored when resuming from "
        "a checkpoint (the on-disk plan.md is preserved); pass --clean "
        "to start fresh.")
    parser.add_argument(
        "--acceptance-criteria",
        dest="acceptance_criteria",
        default=None,
        help="Use a user-supplied acceptance-criteria checklist as the "
        "plan-phase criteria output. Value is either the checklist text "
        "or a path to a file containing it; either way it is written to "
        "`<workspace>/acceptance-criteria.md`. Combined with --plan, the "
        "plan phase is skipped entirely. When --plan is omitted, the "
        "plan phase still runs so the PlanDrafter can generate "
        "`plan.md`. Ignored when resuming from a checkpoint; pass "
        "--clean to start fresh.")
    parser.add_argument(
        "--feedback",
        default=None,
        help="Append a human-feedback entry to `progress.yaml`'s "
        "`human_feedback` list before the next iteration runs. Value is "
        "either the literal feedback text or a path to a file containing "
        "it. Resume-only: requires an existing checkpoint in the workspace "
        "(fresh runs reject this flag). Typical use: stop the workflow "
        "(Ctrl-C), re-run with `--feedback \"...\"`, and the build-phase "
        "agents (coder, reviewer, qa) will read it via `read_human_feedback` "
        "on their next turn. Each invocation appends — prior entries are "
        "preserved. Also re-engages the build phase when a previously "
        "completed workflow is rerun (use --clean instead to start over).")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None,
         *,
         prompts: PromptBundle | None = None) -> None:
    """Run the agent-team workflow as a CLI."""
    args = _parse_args(argv)
    with AgentTeamWorkflow(
            workspace=args.workspace,
            num_iterations=args.num_iterations,
            coder_context_reset_interval=args.coder_context_reset_interval,
            reviewer_context_reset_interval=args.
            reviewer_context_reset_interval,
            min_score=args.min_score,
            plan_human_review_enabled=args.plan_human_review_enabled,
            build_human_review_enabled=args.build_human_review_enabled,
            clean=args.clean,
            plan=args.plan,
            acceptance_criteria=args.acceptance_criteria,
            feedback=args.feedback,
            prompts=prompts,
    ) as workflow:
        workflow.run(args.task)


if __name__ == "__main__":
    main()
