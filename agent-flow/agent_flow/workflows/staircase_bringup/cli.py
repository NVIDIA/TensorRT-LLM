"""Console entry point for the staircase-bringup workflow.

Reuses ``agent_team``'s orchestrator, argument parser, and MCP tools
unchanged; only the prompt bundle and the task schema differ.
"""

from __future__ import annotations

import sys

from agent_flow.workflows.agent_team.cli import _parse_args
from agent_flow.workflows.agent_team.cli import main as _team_main

from .prompts import build_staircase_prompts
from .task_schema import (
    TaskSchemaError,
    has_slurm_environment,
    load_and_validate_task_yaml,
    target_relpath,
    world_size,
)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    try:
        task_data = load_and_validate_task_yaml(args.task)
    except TaskSchemaError as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(2)

    # Echo the resolved triple before anything spins up. Every gate is
    # topology-blind, so a wrong parallel segment produces a self-consistently
    # wrong run that passes smoke and accuracy while the directory name lies;
    # printing the derived path and rank count is the cheapest place to catch
    # a typo in it.
    print(f"[staircase] target: {target_relpath(task_data)}")
    print(f"[staircase] world size: {world_size(task_data)} GPU(s)")

    prompts = build_staircase_prompts(
        include_slurm_environment=has_slurm_environment(task_data),
        replan_on_qa=args.replan_on_qa,
    )
    _team_main(argv, prompts=prompts)


if __name__ == "__main__":
    main()
