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

# agent_team defaults this to 2, which suits Goals that are a few iterations
# long and carry their whole intent in the Goal name ("onboard <op>"). A
# staircase Goal is a module, runs far longer, and its expensive content is the
# *negative* results -- which candidate was tried, what ruled it out. Recycling
# the session every other turn throws exactly those away, and the next turn
# re-tries the candidate that was already rejected. 5 is a compromise: still
# bounded, but long enough that a decision and its evidence usually outlive one
# reset. `status.md` is what must carry them across the resets that do happen.
#
# Injected into argv rather than set on the workflow, because agent_team's
# `main` re-parses argv itself and takes no override. Only applied when the
# user did not ask for a value, so an explicit flag always wins.
STAIRCASE_CODER_CONTEXT_RESET_INTERVAL = 5
_RESET_FLAG = "--coder-context-reset-interval"


def _with_staircase_defaults(argv: list[str] | None) -> list[str]:
    """Return argv with staircase's own default for the coder reset interval."""
    source = list(sys.argv[1:] if argv is None else argv)
    if any(a == _RESET_FLAG or a.startswith(f"{_RESET_FLAG}=") for a in source):
        return source
    return source + [_RESET_FLAG, str(STAIRCASE_CODER_CONTEXT_RESET_INTERVAL)]


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
    _team_main(_with_staircase_defaults(argv), prompts=prompts)


if __name__ == "__main__":
    main()
