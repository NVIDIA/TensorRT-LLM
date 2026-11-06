from __future__ import annotations

import sys
from pathlib import Path

from agent_flow.workflows.agent_team.cli import _parse_args
from agent_flow.workflows.agent_team.cli import main as _team_main

from .node_policy import policy_for_type
from .prompts import build_modeling_bringup_prompts
from .task_schema import TaskSchemaError, has_slurm_environment, load_and_validate_task_yaml

MODELING_BRINGUP_PROMPTS = build_modeling_bringup_prompts()


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    try:
        task_data = load_and_validate_task_yaml(args.task)
        prompts = build_modeling_bringup_prompts(
            include_slurm_environment=has_slurm_environment(task_data),
            replan_on_qa=args.replan_on_qa,
            concurrent=args.concurrent,
            max_parallel=args.max_parallel,
        )
    except TaskSchemaError as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(2)

    # The trtllm repo path is known only here (not in the generic agent_team
    # layer), so both --concurrent (per-node worktrees) and --clean (reset the
    # checkout) build the isolation provider here. Lazy-imported so the linear
    # no-clean path (and ``--help``) never pulls in the git provider.
    concurrent_deps: dict[str, object] = {}
    isolation = None
    if args.concurrent or args.clean:
        from agent_flow.git_worktree import GitWorktreeIsolation

        isolation = GitWorktreeIsolation(
            repo_path=Path(task_data["trtllm_repo_path"]),
            worktrees_root=Path(args.workspace) / "worktrees",
        )
    if args.concurrent:
        concurrent_deps["isolation"] = isolation
        concurrent_deps["policy_for_type"] = policy_for_type
    if args.clean:
        # --clean's trtllm-repo counterpart to agent_team's workspace-state wipe.
        import anyio

        anyio.run(isolation.clean)

    _team_main(argv, prompts=prompts, **concurrent_deps)


if __name__ == "__main__":
    main()
