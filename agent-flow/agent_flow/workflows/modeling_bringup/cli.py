from __future__ import annotations

import sys

from agent_flow.workflows.agent_team.cli import _parse_args
from agent_flow.workflows.agent_team.cli import main as _team_main

from .prompts import build_modeling_bringup_prompts
from .task_schema import (TaskSchemaError, has_slurm_environment,
                          load_and_validate_task_yaml)

MODELING_BRINGUP_PROMPTS = build_modeling_bringup_prompts()


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    try:
        task_data = load_and_validate_task_yaml(args.task)
    except TaskSchemaError as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(2)
    prompts = build_modeling_bringup_prompts(
        include_slurm_environment=has_slurm_environment(task_data))
    _team_main(argv, prompts=prompts)


if __name__ == "__main__":
    main()
