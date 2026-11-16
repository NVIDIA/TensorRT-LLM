# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import importlib
import logging
import os
import signal
import sys
import threading
from pathlib import Path
from typing import Callable, Sequence

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
from .run.records.state import STATE_FILENAME
from .run.workflow import DEFAULT_MAX_ATTEMPTS
from .spec.task_schema import TaskSchemaError, load_and_validate_task_yaml

#: The run ended with everything left blocked: answer what is asked (`resolve`), then run it again.
EXIT_ESCALATED = 76
#: The start did not start (its preflight, its first holder or its agents): as for a spec
#: that does not load.
EXIT_NOT_STARTED = 2


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    # Here, not with the imports above: the orchestrator imported before run.workflow
    # walks into the contract -> stages -> orchestrator import cycle.
    from .run.orchestrator import DEFAULT_MAX_PARALLEL_TASKS

    parser = argparse.ArgumentParser(
        prog="trtllm-modeling-v2",
        description=(
            "Bring one checkpoint to a gated modeling_v2 target: "
            "preflight -> execution stage -> deliver. Exit 0 when the run closed, 76 when "
            "everything left is blocked and waits for an answer (`resolve`, then the same command), "
            "75 when the controller stopped for the limit of the job it runs in (the same command "
            "goes on)."
        ),
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--task",
        type=Path,
        help="Path to the task YAML file. It is validated before any agent "
        "is constructed and copied verbatim into `<workspace>/task.yaml`, "
        "which every agent reads.",
    )
    source.add_argument(
        "--checkpoint",
        type=Path,
        help="A HuggingFace checkpoint directory: derive the run's spec from its "
        "config.json, --site and --goal into `<workspace>/task.derived.yaml` (beside "
        "`derive.report.md`) and run it. A question the derivation cannot answer stops "
        "here, to be answered in the spec. Re-running the same command takes the spec "
        "as it stands, edits included; --clean derives it again.",
    )
    parser.add_argument(
        "--site",
        type=Path,
        default=None,
        help="With --checkpoint: the cluster's site.yaml, the environment, sessions, device "
        "and defaults the spec is derived with.",
    )
    parser.add_argument(
        "--goal",
        default=None,
        help="With --checkpoint: the features, members and workload in one line, e.g. "
        '"mtp disagg@tp1/tp2 tp1 tep4 dep4 isl=8192 osl=1024" (spec/derive.py). '
        "Features the checkpoint carries are built unless named `-<feature>`.",
    )
    parser.add_argument(
        "--workspace",
        type=Path,
        default=Path("workspace/trtllm-modeling-v2").absolute(),
        help="Workspace directory for the run's shared state (task.yaml, "
        "plan.md, vocabulary.yaml,journal.md, reports/). "
        "Created on demand.",
    )
    parser.add_argument(
        "--max-parallel-tasks",
        type=int,
        default=DEFAULT_MAX_PARALLEL_TASKS,
        help="How many agent turns run at once: the run's one counter. A check takes "
        "no agent, and devices are the pool's to hand out, never counted here. "
        f"Default {DEFAULT_MAX_PARALLEL_TASKS}.",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=DEFAULT_MAX_ATTEMPTS,
        help="How many attempts one task gets before it is blocked and asks the session "
        "that watches the run; `resolve retry` adds one. A turn the "
        f"controller's own limit cut spends none. Default {DEFAULT_MAX_ATTEMPTS}.",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Wipe the workspace checkpoint and managed files "
        f"({STATE_FILENAME}, plan.md, vocabulary.yaml,"
        "journal.md) and start fresh. Without this flag the workflow "
        "resumes from the checkpoint when one is present, and starts fresh "
        "otherwise.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run preflight, resolve the environment, print the plan of "
        "record, and stop before the first agent turn. The fastest way to "
        "find out whether this host can host this run.",
    )
    parser.add_argument(
        "--agent-turn-timeout",
        type=int,
        default=None,
        help="Seconds a single role turn may take, for every role. Without it each role has its own bound ("
        + ", ".join(f"{role} {int(s) // 3600}h" for role, s in DEFAULT_ROLE_TURN_TIMEOUTS.items())
        + ").",
    )
    args = parser.parse_args(argv)
    if args.checkpoint is not None and args.site is None:
        parser.error("--checkpoint needs --site <site.yaml>")
    if args.checkpoint is None and (args.goal is not None or args.site is not None):
        parser.error("--site and --goal go with --checkpoint")
    return args


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


def _derive_spec(
    checkpoint: Path, site: Path, workspace: Path, *, goal: str | None, clean: bool
) -> Path:
    """The fast path: the workspace's derived spec, derived anew when there is none or on ``clean``.

    Raises :class:`~.spec.derive.DeriveError` with every open question of a
    new derivation: the spec and its report are written first, to be answered in.
    """
    from .spec import derive as derive_mod

    path = workspace / derive_mod.SPEC_FILENAME
    if path.is_file() and not clean:
        print(f"→ {path}: the spec derived before, as it stands; --clean derives it again")
        return path
    site_data = derive_mod.read_site(site)
    derivation = derive_mod.derive_spec(
        checkpoint, site_data, derive_mod.parse_goal(goal), site_path=site
    )
    derive_mod.write_derivation(workspace, derivation, derive_mod.KEPT)
    print(f"→ derived {path} (see {workspace / derive_mod.REPORT_FILENAME})")
    for where, value, source in derivation.unstated:
        if where == "environment.kind":
            print(f"→ environment: {value} ({source}); written into the spec")
    if derivation.questions:
        raise derive_mod.DeriveError(
            f"{path} leaves {len(derivation.questions)} question(s) open; answer them in it and "
            "run the same command again:\n  - " + "\n  - ".join(derivation.questions)
        )
    return path


#: The options of a run that name a path; a subcommand's ``--workspace`` is the same one.
PATH_OPTIONS = ("--task", "--checkpoint", "--site", "--workspace")


def _whole(argv: Sequence[str], options: Sequence[str]) -> list[str]:
    """``argv`` with the path each of ``options`` names made absolute.

    Once, where the command line is read: a job runs in its pin's tree on a node, a record
    is read by another process, and the command that continues a run is said to whoever
    reads its log. None of them may carry a path relative to where a person stood.
    ``--option path`` and ``--option=path``, the option shortened as argparse takes it.
    """
    out: list[str] = []
    path_next = False
    for word in argv:
        name, equals, value = word.partition("=")
        named = len(name) > 2 and any(option.startswith(name) for option in options)
        if path_next:
            word = os.path.abspath(word)
        elif named and equals:
            word = f"{name}={os.path.abspath(value)}"
        path_next = named and not equals and not path_next
        out.append(word)
    return out


def _lazy_main(module: str) -> Callable[[list[str]], int]:
    """``module``'s ``main``, imported only when its subcommand runs."""

    def run(argv: list[str]) -> int:
        return importlib.import_module(module, __package__).main(argv)

    return run


def _feedback(argv: list[str]) -> int:
    """``feedback --workspace W "text"``: append ``text``, time-stamped, to the notes file."""
    from .run import escalation

    parser = argparse.ArgumentParser(
        prog="trtllm-modeling-v2 feedback",
        description=f"Append a note, with its time, to `<workspace>/{escalation.NOTES_FILENAME}`. "
        "Every worker's card tells it to read that file first and again before its receipt; "
        "no agent interprets it and it adds no task. To reach a turn in progress, stop the "
        "run and start it again with the same command: the cut turn runs again as the "
        "attempt it was. A requirement is changed with `resolve ... change-requirement`, "
        "which records it, not with a note.",
    )
    parser.add_argument("--workspace", type=Path, required=True, help="The run's workspace.")
    text = parser.add_mutually_exclusive_group(required=True)
    text.add_argument("message", nargs="?", help="The note.")
    text.add_argument("--file", type=Path, help="Read the note from this file.")
    args = parser.parse_args(argv)
    message = (args.file.read_text(encoding="utf-8") if args.file else args.message).strip()
    if not message:
        parser.error("the note is empty")
    path = escalation.append_note(args.workspace, message)
    print(f"appended to {path}: every later turn reads it")
    return 0


#: ``trtllm-modeling-v2 <subcommand> ...`` for the tools that are not a run.
#: A run is ``trtllm-modeling-v2 --task ...``; a first argument named here
#: dispatches instead, and everything after it is the subcommand's own.
SUBCOMMANDS = {
    # From a request to a spec, before launch: `intake derive|lint ...` (spec/INTAKE.md).
    "intake": _lazy_main(".spec.intake"),
    # One check, as the orchestrator runs it; with a run's own spec its result is recorded
    # in the run's ledger under the pin it measured (run/evidence.py).
    "gate": _lazy_main(".run.evidence"),
    # The pool's holders: `pool status|stop --workspace W`.
    "pool": _lazy_main(".infra.compute.pool"),
    # Jobs in a session's spool: `spool submit|launch|wait|status|cancel ...`.
    "spool": _lazy_main(".infra.compute.spoolctl"),
    # The operator's loop of a run: wait, status.
    "monitor": _lazy_main(".infra.monitor.status"),
    # A note for the workers of a run: `feedback --workspace W "..."`.
    "feedback": _feedback,
    # The outer session's answer to a block: `resolve --workspace W <verb> <task|run|pool> ...`.
    "resolve": _lazy_main(".run.resolve"),
}


def main(argv: list[str] | None = None) -> None:
    """Run the trtllm-modeling-v2 workflow as a CLI."""
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] in SUBCOMMANDS:
        # Its `--task` may be a task's name (`resolve`): the workspace is a path in each.
        sys.exit(SUBCOMMANDS[argv[0]](_whole(argv[1:], ("--workspace",))))
    argv = _whole(argv, PATH_OPTIONS)
    args = _parse_args(argv)
    # Validate at the CLI boundary: an invalid spec must abort before any
    # agent is constructed.
    from .spec.derive import DeriveError

    lock = None
    try:
        if args.checkpoint is not None:
            # The spec is written into the workspace, so it is claimed first.
            lock = _claim_workspace(Path(args.workspace))
            args.task = str(
                _derive_spec(
                    Path(args.checkpoint),
                    Path(args.site),
                    Path(args.workspace),
                    goal=args.goal,
                    clean=args.clean,
                )
            )
        task_data = load_and_validate_task_yaml(args.task)
    except (TaskSchemaError, DeriveError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(2)
    from .infra.compute import controller as controller_mod

    identity = _commit_identity(task_data)
    # Pinned for the rest of the process and every agent it starts, so every
    # commit the run makes is authored, committed and signed off as one user.
    os.environ[ENV_NAME], os.environ[ENV_EMAIL] = identity.name, identity.email
    os.environ.update(identity.env())

    if lock is None:
        _claim_workspace(Path(args.workspace))

    # A driver watching the run tells a crashed orchestrator from a busy one by it.
    heartbeat = start_orchestrator_heartbeat(args.workspace, ORCHESTRATOR_HEARTBEAT_S)

    # Claude Code's auto-memory is shared by every worktree of one repository: one
    # task's notes would reach the others, and one run's the next.
    os.environ.setdefault("CLAUDE_CODE_DISABLE_AUTO_MEMORY", "1")

    from .agents.roles import AgentConfigError
    from .run import orchestrator as orchestrator_mod
    from .run.workflow import TrtllmModelingV2Workflow

    def leave(code: int) -> None:
        # The limit is minutes away and the run did not return: nothing is waited for.
        logging.shutdown()
        sys.stdout.flush()
        sys.stderr.flush()
        heartbeat.mark_exit()
        os._exit(code)

    def told_to_stop(signum: int, frame: object) -> None:
        # TERM, as a job's script or the scheduler sends it: leave now, and say so where
        # whoever watches the run reads it. Without the mark a script that waits for the
        # run's watcher waits until it is killed. Nothing in flight is waited for: the next
        # start runs it again.
        heartbeat.record["why"] = "told to stop (SIGTERM)"
        try:
            leave(128 + signum)
        except BaseException:
            # The signal came during a write: flushing again raises. It still leaves.
            heartbeat.mark_exit()
            os._exit(128 + signum)

    if threading.current_thread() is threading.main_thread():
        signal.signal(signal.SIGTERM, told_to_stop)

    # A controller inside a job stops before that job's end, whoever submitted it.
    limit = controller_mod.watch_limit(
        args.workspace,
        argv,
        stop=lambda reason: orchestrator_mod.stop_run(args.workspace, reason),
        leave=leave,
    )

    with TrtllmModelingV2Workflow(
        workspace=args.workspace,
        max_parallel_tasks=args.max_parallel_tasks,
        max_attempts=args.max_attempts,
        clean=args.clean,
        dry_run=args.dry_run,
        agent_turn_timeout=args.agent_turn_timeout,
    ) as workflow:
        try:
            workflow.run(args.task)
        except AgentConfigError as exc:
            print(f"error: {exc}", file=sys.stderr)
            sys.exit(2)
        except KeyboardInterrupt:
            # The turns and checks in flight are not waited for: the next start runs them again.
            logging.shutdown()
            sys.stdout.flush()
            sys.stderr.flush()
            heartbeat.mark_exit()
            os._exit(130)
    if limit is not None and limit.finish():
        if getattr(getattr(workflow, "_state", None), "done", False):
            # The run closed while the limit was being said: nothing is left to continue.
            return
        # Stopped for the limit: to be continued, by the command that was printed. The turns
        # in flight are not waited for: what continues the run starts them again.
        leave(controller_mod.EXIT_CONTINUE)
    state = getattr(workflow, "_state", None)
    if getattr(workflow, "_budget_stop", False) and not getattr(state, "done", False):
        # The run's bound of node-hours is reached and no holder is left. It waits for an
        # answer (`POOL_CAPACITY` names the verb): the same command, run again before the
        # bound is raised, would stop here again, so this is not "to be continued". No turn
        # is waited for.
        heartbeat.record["why"] = "the run's bound of node-hours is reached (POOL_CAPACITY)"
        leave(EXIT_ESCALATED)
    if getattr(workflow, "_not_started", ""):
        # The start did not start: never a success, whoever reads the status (a script's
        # `&&`, or Slurm's state of the controller's own job, which would say COMPLETED).
        heartbeat.record["why"] = f"the start did not start: {workflow._not_started}"
        heartbeat.mark_exit()
        sys.exit(EXIT_NOT_STARTED)
    if getattr(workflow, "_escalated", False):
        # Everything left is blocked: the run's own status, for whoever watches it.
        heartbeat.record["why"] = "everything left is blocked and waits for an answer"
        heartbeat.mark_exit()
        sys.exit(EXIT_ESCALATED)
    if getattr(state, "done", False):
        heartbeat.record["why"] = f"the run closed: {getattr(state, 'verdict', '') or 'done'}"


if __name__ == "__main__":
    main()
