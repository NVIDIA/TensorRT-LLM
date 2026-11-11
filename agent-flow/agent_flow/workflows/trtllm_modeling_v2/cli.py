# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import importlib
import logging
import os
import sys
from pathlib import Path
from typing import Callable

from .agents.turns import DEFAULT_ROLE_TURN_TIMEOUTS
from .infra.git import (
    ENV_EMAIL,
    ENV_NAME,
    CommitIdentity,
    CommitPolicy,
    CommitPolicyError,
    resolve_commit_identity,
    trial_commit,
)
from .infra.monitor.heartbeat import ORCHESTRATOR_HEARTBEAT_S, start_orchestrator_heartbeat
from .run.pipeline import DEFAULT_MAX_ATTEMPTS, DEFAULT_MAX_ROUNDS
from .run.state import STATE_FILENAME
from .spec.task_schema import TaskSchemaError, load_and_validate_task_yaml


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="trtllm-modeling-v2",
        description=(
            "Bring one checkpoint to a certified modeling_v2 target: preflight, the tasks in "
            "a fixed order, then the certification of the final commit."
        ),
    )
    parser.add_argument(
        "--task",
        type=Path,
        required=True,
        help="Path to the task YAML file. It is validated before any agent is constructed "
        "and written, resolved, to `<workspace>/task.yaml`, which every agent reads. "
        "`trtllm-modeling-v2 intake derive` writes a first one from a checkpoint.",
    )
    parser.add_argument(
        "--workspace",
        type=Path,
        default=Path("workspace/trtllm-modeling-v2"),
        help="Workspace directory for the run's shared state (task.yaml, "
        "plan.md, vocabulary.yaml,journal.md, reports/). "
        "Created on demand.",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=DEFAULT_MAX_ATTEMPTS,
        help="How many worker runs one task gets, its first included. A task that spent "
        f"them is the planner's to repair around or escalate. Default {DEFAULT_MAX_ATTEMPTS}.",
    )
    parser.add_argument(
        "--max-rounds",
        type=int,
        default=None,
        help="Ignored, with a notice: a run certifies its final commit and repairs it at most "
        f"{DEFAULT_MAX_ROUNDS - 1} time(s).",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Wipe the workspace checkpoint and managed files "
        f"({STATE_FILENAME}, plan.md, vocabulary.yaml, tasks.yaml, the verdicts, "
        "journal.md) and start fresh. Without this flag the workflow "
        "resumes from the checkpoint when one is present, and starts fresh "
        "otherwise.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run preflight and stop before the first agent turn: the spec, the checkout, "
        "the checkpoint, the contract and the agent probe. It touches no GPU (no device is "
        "queried or claimed), so it also runs on a host without one.",
    )
    parser.add_argument(
        "--agent-turn-timeout",
        type=int,
        default=None,
        help="Seconds a single role turn may take, for every role. Without it each role has its own bound ("
        + ", ".join(f"{role} {int(s) // 3600}h" for role, s in DEFAULT_ROLE_TURN_TIMEOUTS.items())
        + ").",
    )
    return parser.parse_args(argv)


def _commit_identity(task_data: dict) -> CommitIdentity:
    """The identity every commit of the run is authored and signed off under.

    The repository's configured user, read before $HOME is redirected
    (redirecting it takes the real ~/.gitconfig off git's search path), and
    proven by one trial ``git commit -s``. Exits 2 when there is no identity,
    or when this environment's git changes what a commit carries.
    """
    repo = str(task_data.get("trtllm_repo_path") or "")
    try:
        identity = resolve_commit_identity(repo)
    except CommitPolicyError as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(2)
    policy = CommitPolicy.from_task(task_data, identity=identity)
    problems = trial_commit(identity, allow_co_authors=policy.allow_co_authors)
    if problems:
        print(
            "error: a trial `git commit -s` does not meet the commit "
            f"rules under {identity.render()}:\n  - " + "\n  - ".join(problems),
            file=sys.stderr,
        )
        sys.exit(2)
    return identity


#: Open file objects whose flock this process holds, kept alive deliberately.
_HELD_LOCKS: list = []


def _claim_workspace(workspace: Path):
    """Take an exclusive lock on the workspace, or refuse to start.

    The handle is kept in a module-level list rather than returned-and-hoped,
    because `flock` is held by an OPEN FILE DESCRIPTION: let the file object
    be collected and the lock silently disappears.

    Released when the process exits, which is what a crash should do.
    """
    import fcntl

    workspace.mkdir(parents=True, exist_ok=True)
    path = workspace / ".orchestrator.lock"
    handle = path.open("w")
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        handle.close()
        print(
            f"error: another trtllm-modeling-v2 process already holds {path}.\n"
            f"       Two orchestrators on one workspace share the checkpoint and the\n"
            f"       turn records, so they overwrite each other. Stop the other one\n"
            f"       first, or use a different --workspace.",
            file=sys.stderr,
        )
        sys.exit(3)
    handle.write(f"{os.getpid()}\n")
    handle.flush()
    _HELD_LOCKS.append(handle)
    return handle


def _lazy_main(module: str) -> Callable[[list[str]], int]:
    """``module``'s ``main``, imported only when its subcommand runs."""

    def run(argv: list[str]) -> int:
        return importlib.import_module(module, __package__).main(argv)

    return run


def _feedback(argv: list[str]) -> int:
    from .run import inbox

    parser = argparse.ArgumentParser(
        prog="trtllm-modeling-v2 feedback",
        description="Leave a message for the planner of the run in a workspace. The run "
        "asks the planner about it between two tasks. A message steers the work in a few "
        f"sentences, evidence in a file it names; past {inbox.MESSAGE_MAX_CHARS} characters "
        "the planner reads its head and the path of the whole. It does not change the "
        "contract's checks, which task.yaml defines, and is no place for status reports.",
    )
    parser.add_argument("--workspace", type=Path, required=True, help="The run's workspace.")
    text = parser.add_mutually_exclusive_group(required=True)
    text.add_argument("message", nargs="?", help="The message.")
    text.add_argument("--file", type=Path, help="Read the message from this file.")
    args = parser.parse_args(argv)
    message = args.file.read_text(encoding="utf-8") if args.file else args.message
    try:
        path = inbox.post(args.workspace, message)
    except ValueError as exc:
        parser.error(str(exc))
    print(f"left for the planner, which is asked between two tasks: {path}")
    if len(message.strip()) > inbox.MESSAGE_MAX_CHARS:
        print(
            f"note: the message is {len(message.strip())} characters; the planner reads its "
            f"first {inbox.MESSAGE_MAX_CHARS} and the path of the whole"
        )
    return 0


#: ``trtllm-modeling-v2 <subcommand> ...`` for the tools that are not a run.
#: A run is ``trtllm-modeling-v2 --task ...``; a first argument named here
#: dispatches instead, and everything after it is the subcommand's own.
SUBCOMMANDS = {
    # A first spec from a checkpoint and a site, before launch: `intake derive ...`.
    "intake": _lazy_main(".spec.derive"),
    "gate": _lazy_main(".checks.gate_cli"),
    # The operator's loop of a run: wait, status.
    "monitor": _lazy_main(".infra.monitor.status"),
    # A message for the planner of a running run: `feedback --workspace W "..."`.
    "feedback": _feedback,
}


def main(argv: list[str] | None = None) -> None:
    """Run the trtllm-modeling-v2 workflow as a CLI."""
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] in SUBCOMMANDS:
        sys.exit(SUBCOMMANDS[argv[0]](argv[1:]))
    args = _parse_args(argv)
    # Validate at the CLI boundary: an invalid spec must abort before any
    # agent is constructed.
    notices: list[str] = []
    try:
        task_data = load_and_validate_task_yaml(args.task, notices)
    except TaskSchemaError as exc:
        # A refused spec still says which of its keys were ignored on the way.
        for notice in notices:
            print(f"note: {notice}", file=sys.stderr)
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(2)
    identity = _commit_identity(task_data)
    # Pinned for the rest of the process and every agent it starts, so every
    # commit the run makes is authored, committed and signed off as one user.
    os.environ[ENV_NAME], os.environ[ENV_EMAIL] = identity.name, identity.email
    os.environ.update(identity.env())

    _claim_workspace(Path(args.workspace))

    if args.max_rounds is not None:
        print(
            f"note: --max-rounds is ignored: a run certifies its final commit and repairs it "
            f"at most {DEFAULT_MAX_ROUNDS - 1} time(s); drop the flag",
            file=sys.stderr,
        )
    # A driver watching the run tells a crashed orchestrator from a busy one by it.
    heartbeat = start_orchestrator_heartbeat(args.workspace, ORCHESTRATOR_HEARTBEAT_S)

    # Claude Code's auto-memory is shared by every worktree of one repository: one
    # task's notes would reach the others, and one run's the next.
    os.environ.setdefault("CLAUDE_CODE_DISABLE_AUTO_MEMORY", "1")

    from .run.workflow import TrtllmModelingV2Workflow

    with TrtllmModelingV2Workflow(
        workspace=args.workspace,
        max_attempts=args.max_attempts,
        clean=args.clean,
        dry_run=args.dry_run,
        agent_turn_timeout=args.agent_turn_timeout,
    ) as workflow:
        try:
            delivered = workflow.run(args.task)
        except TaskSchemaError as exc:
            print(f"error: {exc}", file=sys.stderr)
            sys.exit(2)
        except KeyboardInterrupt:
            # The turns and checks in flight are not waited for: the next start runs them again.
            logging.shutdown()
            sys.stdout.flush()
            sys.stderr.flush()
            heartbeat.mark_exit()
            os._exit(130)
    # 0: delivered, or a dry run whose preflight passed. 1: anything else the run came to
    # (a failed preflight, escalated, certification-failed); the journal and the report say what.
    sys.exit(0 if delivered else 1)


if __name__ == "__main__":
    main()
