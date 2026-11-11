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
from .run.pipeline import DEFAULT_MAX_ATTEMPTS, NOTES_FILENAME
from .run.state import STATE_FILENAME
from .spec.task_schema import TaskSchemaError, load_and_validate_task_yaml


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="trtllm-modeling-v2",
        description=(
            "Onboard one checkpoint as a modeling_v2 target: preflight, the tasks in a fixed "
            "order, the onboard check on the first deployment, each further deployment, one "
            "final pass and the report. Exit status 0 means onboarded: the first deployment's "
            "boot and benchmarks passed on the commit that landed, whatever else is not done; "
            "1 anything else the run came to (`delivery_report.md` and the journal say what); "
            "2 a spec or usage error. The same command against the same workspace goes on "
            "from where the last start stopped."
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
        "them is not done: the run goes on without it, and the next start gives it fresh "
        f"ones. Default {DEFAULT_MAX_ATTEMPTS}.",
    )
    parser.add_argument(
        "--max-rounds",
        type=int,
        default=None,
        help="Ignored, with a notice: the run has one final pass and no rounds.",
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
        "the checkpoint, the anchors and the agent probe. It touches no GPU (no device is "
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
    """``feedback --workspace W "text"``: append ``text``, time-stamped, to the operator's notes."""
    from .run.journal import utc_now

    parser = argparse.ArgumentParser(
        prog="trtllm-modeling-v2 feedback",
        description=f"Append a note, with its time, to `<workspace>/{NOTES_FILENAME}`. Every "
        "later turn of the run is told to read that file first and again before it ends; "
        "no agent interprets it and it adds no task. To reach the turn in progress, stop the "
        "run and start it again with the same command: the cut turn restarts. It is "
        "information for the workers, never a reason to change an anchor or a check.",
    )
    parser.add_argument("--workspace", type=Path, required=True, help="The run's workspace.")
    text = parser.add_mutually_exclusive_group(required=True)
    text.add_argument("message", nargs="?", help="The note.")
    text.add_argument("--file", type=Path, help="Read the note from this file.")
    args = parser.parse_args(argv)
    message = (args.file.read_text(encoding="utf-8") if args.file else args.message).strip()
    if not message:
        parser.error("the note is empty")
    args.workspace.mkdir(parents=True, exist_ok=True)
    path = args.workspace / NOTES_FILENAME
    with path.open("a", encoding="utf-8") as notes:
        notes.write(f"\n## {utc_now()}\n\n{message}\n")
    print(f"appended to {path}: every later turn reads it")
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
    # A note for the workers of a run: `feedback --workspace W "..."`.
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
            "note: --max-rounds is ignored: the run has one final pass and no rounds; "
            "drop the flag",
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
            onboarded = workflow.run(args.task)
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
    # 0: onboarded, whatever else is not done; or a dry run whose preflight passed. 1: anything
    # else the run came to (a failed preflight, not onboarded): the journal and the report say.
    sys.exit(0 if onboarded else 1)


if __name__ == "__main__":
    main()
