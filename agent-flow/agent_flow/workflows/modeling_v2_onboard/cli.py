# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``modeling-v2-onboard`` command line."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from agent_flow import (
    CLAUDE_CODE_DEFAULT_MODEL,
    CODEX_DEFAULT_MODEL,
    AgentLayer,
    AgentLayerConfig,
    BackendConfig,
    SessionConfig,
)
from agent_flow.workflows.modeling_bringup.task_schema import (
    TaskSchemaError,
    load_and_validate_task_yaml,
)

from .files import post_feedback
from .node_runner import AgentFactory
from .plan import PlanError
from .workflow import ModelingV2OnboardWorkflow, RunOutcome, WorkflowError

__all__ = ["main", "make_agent_factory"]


def make_agent_factory(backend: str, model: str | None) -> AgentFactory:
    """Build every agent of the run on one backend."""
    if model is None:
        model = CODEX_DEFAULT_MODEL if backend == "codex" else CLAUDE_CODE_DEFAULT_MODEL

    def factory(
        name: str,
        system_prompt: str,
        cwd: Path,
        session_mode: str,
        fork_from: str | None = None,
    ) -> AgentLayer:
        return AgentLayer(
            AgentLayerConfig(
                name=name,
                system_prompt=system_prompt,
                backend=BackendConfig(kind=backend, model=model, cwd=Path(cwd)),
                session=SessionConfig(mode=session_mode, fork_from=fork_from),
            )
        )

    return factory


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="modeling-v2-onboard",
        description=(
            "Bring up a model or a new target in ModelingV2: a planner writes plan.md, "
            "workers run its tasks in parallel, and the planner extends or revises the "
            "plan whenever a worker finishes. Rerun the same command to resume; use "
            "--feedback to steer a running run."
        ),
    )
    parser.add_argument("--task", type=Path, help="Path to task.yaml (required to run).")
    parser.add_argument(
        "--workspace",
        required=True,
        type=Path,
        help="Directory for plan.md, task results and run state; outside trtllm_repo_path.",
    )
    parser.add_argument(
        "--backend",
        choices=("claude-code", "codex"),
        default="claude-code",
        help="Agent backend for the planner and every worker (default: claude-code).",
    )
    parser.add_argument(
        "--model", default=None, help="Model for every agent (default: the backend's default)."
    )
    parser.add_argument(
        "--max-parallel",
        type=int,
        default=8,
        help="The most workers that run at the same time (default: 8).",
    )
    parser.add_argument(
        "--side-wait-minutes",
        type=float,
        default=60,
        help="Once the final task is done, how long side tasks still running may take to "
        "finish before they are stopped (default: 60).",
    )
    parser.add_argument(
        "--no-primer",
        action="store_true",
        help="Start every new_op and perf worker afresh instead of as a fork of the op "
        "primer session (the Codex backend always does).",
    )
    message = parser.add_mutually_exclusive_group()
    message.add_argument(
        "--feedback",
        metavar="TEXT",
        help="Leave a message for the planner of the run in --workspace and exit. The "
        "running orchestrator hands it to the planner's next turn, which acts on it first.",
    )
    message.add_argument(
        "--feedback-file",
        type=Path,
        metavar="PATH",
        help="Like --feedback, with the message read from a file.",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Delete this workspace's run state and the task worktrees/branches, then start over.",
    )
    args = parser.parse_args(argv)
    if args.max_parallel < 1:
        parser.error("--max-parallel must be at least 1")
    if args.side_wait_minutes < 0:
        parser.error("--side-wait-minutes cannot be negative")
    leaving_message = args.feedback is not None or args.feedback_file is not None
    if leaving_message and args.clean:
        parser.error("--feedback cannot be combined with --clean")
    if not leaving_message and args.task is None:
        parser.error("--task is required to run")
    return args


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    if args.feedback is not None or args.feedback_file is not None:
        text = args.feedback
        if text is None:
            text = args.feedback_file.read_text(encoding="utf-8")
        try:
            path = post_feedback(args.workspace, text)
        except ValueError as exc:
            print(f"error: {exc}", file=sys.stderr)
            sys.exit(2)
        print(f"message left for the planner: {path}")
        return
    # Claude Code keeps one auto-memory per repository, shared by all its
    # worktrees: the run's agents would read memories of unrelated work and of
    # earlier runs, and leave theirs to later ones. Every agent of a run starts
    # from the run's own files instead.
    os.environ["CLAUDE_CODE_DISABLE_AUTO_MEMORY"] = "1"
    try:
        task_data = load_and_validate_task_yaml(args.task)
        workflow = ModelingV2OnboardWorkflow(
            workspace=args.workspace,
            task_path=args.task,
            task_data=task_data,
            agent_factory=make_agent_factory(args.backend, args.model),
            max_parallel=args.max_parallel,
            side_wait_minutes=args.side_wait_minutes,
            primer=args.backend != "codex" and not args.no_primer,
        )
        if args.clean:
            workflow.clean()
        outcome = workflow.run()
    except (TaskSchemaError, WorkflowError, PlanError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(2)
    sys.exit(0 if outcome != RunOutcome.ABORTED else 1)


if __name__ == "__main__":
    main()
