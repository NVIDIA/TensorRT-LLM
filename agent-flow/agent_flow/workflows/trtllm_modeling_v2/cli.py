# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import importlib
import os
import sys
from pathlib import Path
from typing import Callable, Mapping, Sequence

from .infra.committer import (
    ENV_EMAIL,
    ENV_NAME,
    CommitIdentity,
    CommitPolicy,
    CommitPolicyError,
    resolve_commit_identity,
    trial_commit,
)
from .infra.dispatch import ORCHESTRATOR_HEARTBEAT_S, start_orchestrator_heartbeat
from .infra.execution.turns import DEFAULT_ROLE_TURN_TIMEOUTS
from .run.state import STATE_FILENAME
from .run.workflow import DEFAULT_EVENT_WINDOW_S, DEFAULT_REVISION_CAP
from .spec.task_schema import TaskSchemaError, load_and_validate_task_yaml


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    # Here, not with the imports above: the orchestrator imported before run.workflow
    # walks into the contract -> stages -> orchestrator import cycle.
    from .run.orchestrator import DEFAULT_MAX_PARALLEL_TASKS

    parser = argparse.ArgumentParser(
        prog="trtllm-modeling-v2",
        description=(
            "Bring one checkpoint to a gated modeling_v2 target: "
            "preflight -> planned run -> deliver."
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
        help="A HuggingFace checkpoint directory: derive the task from its config.json, "
        "--site and --goal, write it to `<workspace>/task.intake.yaml` (beside "
        "`intake.report.md`) and run it at once. Its intake checks run in phase 0 and "
        "stop the run with every block listed. Re-running the same command resumes; "
        "--clean derives it again.",
    )
    parser.add_argument(
        "--site",
        type=Path,
        default=None,
        help="The cluster's site.yaml: with --checkpoint, the environment, sessions, "
        "device and defaults the task is derived with; with an intake --task, compiled "
        "with it into `<workspace>/task.resolved.yaml`, which the run uses.",
    )
    parser.add_argument(
        "--goal",
        default=None,
        help="With --checkpoint: the features, members and workload in one line, e.g. "
        '"mtp disagg@tp1/tp2 tp1 tep4 dep4 isl=8192 osl=1024 tune" (INTAKE.md, the fast '
        "path). Features the checkpoint carries are built unless named `-<feature>`.",
    )
    parser.add_argument(
        "--confirm",
        nargs="?",
        const="",
        default=None,
        metavar="BY",
        help="With --checkpoint: record that BY (default: $USER) confirms the derived task. "
        "The run does not wait for a confirmation; an edit of the task after its run "
        "started is refused until one is given.",
    )
    parser.add_argument(
        "--workspace",
        type=Path,
        default=Path("workspace/trtllm-modeling-v2"),
        help="Workspace directory for the run's shared state (task.yaml, "
        "plan.md, vocabulary.yaml, progress.yaml, journal.md, reports/). "
        "Created on demand.",
    )
    parser.add_argument(
        "--max-parallel-tasks",
        type=int,
        default=DEFAULT_MAX_PARALLEL_TASKS,
        help="How many agents work at once, task turns and the planner's "
        "alike; task turns leave one of them for the planner. A task keeps "
        "its place until its checks are in; checks take no agent. "
        f"Default {DEFAULT_MAX_PARALLEL_TASKS}.",
    )
    parser.add_argument(
        "--revision-cap",
        type=int,
        default=DEFAULT_REVISION_CAP,
        help="How many graph revisions the planner may make without a contract "
        f"check turning green before the run closes planner-stalled. Default "
        f"{DEFAULT_REVISION_CAP}.",
    )
    parser.add_argument(
        "--no-task-worktrees",
        dest="task_worktrees",
        action="store_false",
        help="Let tasks edit the checkout directly instead of each in its own worktree, "
        "integrated when it finishes. Only for a checkout that cannot carry worktrees.",
    )
    parser.add_argument(
        "--event-window",
        dest="event_window_s",
        type=float,
        default=DEFAULT_EVENT_WINDOW_S,
        metavar="SECONDS",
        help="While tasks, checks or a planner turn are in flight, how long a task that "
        "finished with its checks green waits to reach the planner with the events after "
        "it. A failure, a worker's report, an operator's message, a failed re-check, or nothing left "
        f"to run or start wakes the planner at once. Default {DEFAULT_EVENT_WINDOW_S:g}.",
    )
    parser.add_argument(
        "--manage-sessions",
        action="store_true",
        help="Submit, renew and release the run's compute sessions (environment.sessions) "
        "with sbatch from this host: a session with no watcher is submitted, a hand-over "
        "request gets its successor, and the run's jobs still queued are cancelled when it "
        "closes (infra/substrate/managed.py). Needs sbatch, squeue and scancel here.",
    )
    parser.add_argument(
        "--no-primers",
        dest="primers",
        action="store_false",
        help="Start every planner, kernel writer and assembler turn afresh instead of as a fork "
        "of one briefing session per role, which read what all of them read first "
        "(run/primers.py; a turn that cannot fork reads the session's summary).",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Wipe the workspace checkpoint and managed files "
        f"({STATE_FILENAME}, plan.md, vocabulary.yaml, progress.yaml, "
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
        "--wait-for-spool",
        type=float,
        default=0.0,
        metavar="MINUTES",
        help="With a session_spool environment, wait up to MINUTES for the "
        "spool to appear (its compute session is still importing the image) "
        "instead of failing validation at once.",
    )
    parser.add_argument(
        "--skip-substrate-gate",
        action="store_true",
        help="Skip preflight's substrate gate — the cold start that proves "
        "this host generates a token on the STOCK modeling path under this "
        "run's runtime block. It is the single highest-value preflight "
        "check and skipping it means the three failures it catches "
        "(MPI_ERR_SPAWN, kvCache sizing, NVRTC's missing cuda.h) resurface "
        "hours later inside an agent's turn, reading as target defects. "
        "A resumed run already skips the gate when its last pass "
        "(gates/substrate/pass.json) is under 12 h old in the same environment; "
        "use this only when the gate itself is what is broken. It is "
        "NOT the answer to an architecture the stock path does not know — "
        "declare `environment.substrate_checkpoint` for that, which keeps "
        "the two checks that do not depend on the checkpoint.",
    )
    parser.add_argument(
        "--agent-transport",
        choices=("inprocess", "external"),
        default="inprocess",
        help="How a role turn is delivered. 'inprocess' (default) spawns a "
        "CLI per turn, which requires this host to be able to authenticate "
        "one. 'external' writes the prompt to `<workspace>/dispatch/` and "
        "waits for an answer, so an already-authenticated session can take "
        "the turn instead — use it when turns fail with `EROFS ... "
        "credentials.lock` or `Not logged in`, which are host failures that "
        "present as zero-token model failures. Everything else — the stage "
        "machine, the queue, decision validation, every gate — stays "
        "here either way.",
    )
    parser.add_argument(
        "--stage",
        dest="entry_stage",
        choices=["", "tune"],
        default="",
        help="Start at a later stage. `tune` runs the performance stage (ground truth, "
        "tuner candidates) on an already-onboarded target, or re-enters a "
        "delivered run at it. Same as the `tune` subcommand. Needs a `perf:` block.",
    )
    parser.add_argument(
        "--agent-turn-timeout",
        type=int,
        default=None,
        help="Seconds a single role turn may take, in process or dispatched, for every "
        "role. Without it each role has its own bound ("
        + ", ".join(f"{role} {int(s) // 3600}h" for role, s in DEFAULT_ROLE_TURN_TIMEOUTS.items())
        + ").",
    )
    args = parser.parse_args(argv)
    if args.checkpoint is not None and args.site is None:
        parser.error("--checkpoint needs --site <site.yaml>")
    if args.checkpoint is None and (args.goal is not None or args.confirm is not None):
        parser.error("--goal and --confirm go with --checkpoint")
    return args


def _commit_identity(task_data: dict) -> CommitIdentity:
    """The identity every commit of the run is authored and signed off under.

    The repository's configured user, read before $HOME is redirected
    (redirecting it takes the real ~/.gitconfig off git's search path), and
    proven by one trial commit through the generated helper. Exits 2 when
    there is no identity, or when this environment's git changes what the
    helper writes.
    """
    repo = str(task_data.get("trtllm_repo_path") or "")
    try:
        identity = resolve_commit_identity(repo)
    except CommitPolicyError as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(2)
    policy = CommitPolicy.from_task(task_data, identity=identity)
    problems = trial_commit(policy)
    if problems:
        print(
            "error: a trial commit through the commit helper does not meet the commit "
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
            f"       Two orchestrators on one workspace share the checkpoint, the\n"
            f"       dispatch directory and the request-id counter, so they overwrite\n"
            f"       each other and both unblock on one answer. Stop the other one\n"
            f"       first, or use a different --workspace.",
            file=sys.stderr,
        )
        sys.exit(3)
    handle.write(f"{os.getpid()}\n")
    handle.flush()
    _HELD_LOCKS.append(handle)
    return handle


def _relauncher(argv: Sequence[str], env: Mapping[str, str]) -> Callable[[], None]:
    """What a reload calls: this process, replaced by the same command line on the code now on disk.

    ``argv`` is the interpreter's own command line (``sys.orig_argv``, so a
    ``python -m`` launch stays one) and ``env`` the environment the process
    was started with, not the one the run has since changed. The pid, the
    open log and the arguments stay, but for ``--clean``: the run resumes,
    it does not start over. Every module is imported again. ``--confirm``
    goes too: a confirmation is the operator's, given once, and a reload
    after the request was edited must not confirm the edit.
    """
    words = list(argv)[1:]
    kept: list[str] = []
    skip_value = False
    for word in words:
        if skip_value:
            skip_value = False
            if not word.startswith("-"):
                continue
        if word == "--clean" or word.startswith("--confirm="):
            continue
        if word == "--confirm":
            skip_value = True
            continue
        kept.append(word)
    command = [sys.executable, *kept]
    environment = dict(env)

    def reexec() -> None:
        for stream in (sys.stdout, sys.stderr):
            stream.flush()
        os.execve(sys.executable, command, environment)

    return reexec


def _resolve_intake_task(task: Path, site: Path | None, workspace: Path) -> Path:
    """A confirmed user-facing task.yaml, compiled for ``site``; any other spec as it is."""
    import yaml

    from .spec.intake.compile import compile_files
    from .spec.intake.freeze import read_confirmation, unconfirmed_reason
    from .spec.intake.schema import IntakeError, is_intake_task

    data = yaml.safe_load(task.read_text(encoding="utf-8"))
    if not is_intake_task(data):
        return task
    if site is None:
        raise IntakeError("an intake task.yaml needs --site <site.yaml>")
    reason = unconfirmed_reason(data, read_confirmation(task))
    if reason:
        raise IntakeError(reason)
    out = workspace / "task.resolved.yaml"
    compiled = compile_files(task, site, out)
    for warning in compiled.warnings:
        print(f"warning: {warning}", file=sys.stderr)
    return out


def _run_started(workspace: Path) -> bool:
    """Whether the workspace's run is past preflight, so its spec is what it started from."""
    from .run.state import STAGE_PREFLIGHT, load_state

    path = workspace / STATE_FILENAME
    if not path.is_file():
        return False
    try:
        return load_state(path).stage != STAGE_PREFLIGHT
    except (OSError, ValueError, KeyError):
        # A checkpoint this process cannot read is one it cannot call new.
        return True


def _derive_intake_task(
    checkpoint: Path,
    site: Path,
    workspace: Path,
    *,
    goal: str | None,
    confirm_by: str | None,
    clean: bool,
) -> tuple[Path, Path]:
    """The fast path: derive (or resume) the workspace's request and compile it.

    A new workspace, or ``clean``, derives the request from the checkpoint,
    ``site`` and ``goal``; otherwise the request already there is the run's,
    edits included, as long as it was derived from the same checkpoint (and
    goal, when one is given). ``confirm_by`` records a confirmation. A request
    that changed after its run started is refused unless confirmed. Returns
    the compiled spec's path and the request's.
    """
    import yaml

    from .spec.intake import derive as derive_mod
    from .spec.intake.checks import blocking, render_findings, run_checks
    from .spec.intake.compile import compile_files
    from .spec.intake.freeze import (
        confirm,
        confirmation_path,
        stamp_start,
        start_reason,
        started_path,
    )
    from .spec.intake.schema import IntakeError

    request_path = workspace / derive_mod.REQUEST_FILENAME
    if clean or not request_path.is_file():
        site_data = yaml.safe_load(site.read_text(encoding="utf-8")) or {}
        derivation = derive_mod.derive_task(
            checkpoint, site_data, derive_mod.parse_goal(goal), site_path=site
        )
        for stale in (confirmation_path(request_path), started_path(request_path)):
            stale.unlink(missing_ok=True)
        derive_mod.write_request(workspace, derivation, run_checks(derivation.task))
        print(f"→ derived {request_path} (see {workspace / derive_mod.REPORT_FILENAME})")
    else:
        existing = yaml.safe_load(request_path.read_text(encoding="utf-8")) or {}
        problem = derive_mod.resume_problem(existing, checkpoint=checkpoint, goal=goal)
        if problem:
            raise IntakeError(f"{request_path}: {problem}")
    request = yaml.safe_load(request_path.read_text(encoding="utf-8")) or {}
    if confirm_by is not None:
        findings = run_checks(request)
        if blocking(findings):
            raise IntakeError(
                f"{request_path} cannot be confirmed while a block stands:\n"
                + render_findings(blocking(findings))
            )
        by = confirm_by or os.environ.get("USER") or os.environ.get("LOGNAME") or "operator"
        stamp = confirm(request_path, by)
        print(f"→ {request_path} confirmed by {stamp['by']} at {stamp['at']}")
    started = _run_started(workspace) and not clean
    reason = start_reason(request_path, started=started)
    if reason:
        raise IntakeError(f"{request_path}: {reason}")
    if not started:
        stamp_start(request_path)
    resolved = workspace / "task.resolved.yaml"
    try:
        compiled = compile_files(request_path, site, resolved)
    except (IntakeError, TaskSchemaError) as exc:
        # Every finding of the request beside what the compile refused, in one report.
        raise IntakeError(
            f"{exc}\n\nintake checks on {request_path}:\n{render_findings(run_checks(request))}"
        ) from exc
    for warning in compiled.warnings:
        print(f"warning: {warning}", file=sys.stderr)
    return resolved, request_path


def _lazy_main(module: str) -> Callable[[list[str]], int]:
    """``module``'s ``main``, imported only when its subcommand runs."""

    def run(argv: list[str]) -> int:
        return importlib.import_module(module, __package__).main(argv)

    return run


def _metrics(argv: list[str]) -> int:
    import json

    from .run import metrics
    from .run.journal import Journal

    parser = argparse.ArgumentParser(
        prog="trtllm-modeling-v2 metrics",
        description="What a run cost and proved so far: milestones, cells, checks by gate with "
        "their device-hours, turns by role, refused graphs, operator messages, restarts.",
    )
    parser.add_argument("--workspace", type=Path, required=True, help="The run's workspace.")
    args = parser.parse_args(argv)
    opened = Journal(path=args.workspace / "journal.md").opened_at()
    path = metrics.write(args.workspace, opened=opened)
    print(json.dumps(json.loads(path.read_text(encoding="utf-8")), indent=1))
    return 0


def _feedback(argv: list[str]) -> int:
    from .run import inbox

    parser = argparse.ArgumentParser(
        prog="trtllm-modeling-v2 feedback",
        description="Leave a message for the planner of the run in a workspace. The run "
        "takes it within a loop and puts it first in the planner's next turn. A message "
        "steers the work; it relaxes no check.",
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
    print(f"left for the planner's next turn: {path}")
    return 0


#: ``trtllm-modeling-v2 <subcommand> ...`` for the tools that are not a run.
#: A run is ``trtllm-modeling-v2 --task ...``; a first argument named here
#: dispatches instead, and everything after it is the subcommand's own.
SUBCOMMANDS = {
    "gate": _lazy_main(".checks.gate_cli"),
    # The test-result cache: `tests affected|compare ...`.
    "tests": _lazy_main(".checks.testcache"),
    # Step 1: `intake derive|check|confirm|compile ...`.
    "intake": _lazy_main(".spec.intake.cli"),
    # Compute sessions: render entry scripts, status, hand-over, watchdog.
    "session": _lazy_main(".infra.substrate.cli"),
    # Jobs in a session's spool: `spool submit|launch|wait|status|cancel ...`.
    "spool": _lazy_main(".infra.substrate.spoolctl"),
    # The operator's loop of a run (wait, status, reload) and, under the external
    # transport, the turn hand-out (claim, prompt, answer, instructions).
    "driver": _lazy_main(".infra.driver.cli"),
    # A message for the planner of a running run: `feedback --workspace W "..."`.
    "feedback": _feedback,
    # What a run cost and proved so far: `metrics --workspace W` (writes metrics.json).
    "metrics": _metrics,
}


def _wait_for_spool(task: Path, timeout_s: float, poll_s: float = 15.0) -> bool:
    """Wait for a session_spool environment's spool directory; validation still owns the verdict."""
    import time

    import yaml

    try:
        env = (yaml.safe_load(task.read_text()) or {}).get("environment") or {}
    except (OSError, yaml.YAMLError):
        return False
    spool = env.get("spool") if env.get("kind") == "session_spool" else None
    if not isinstance(spool, str) or Path(spool).is_dir():
        return True
    print(f"→ waiting up to {timeout_s / 60:g} min for the spool {spool} (its session is starting)")
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        time.sleep(poll_s)
        if Path(spool).is_dir():
            return True
    return False


def main(argv: list[str] | None = None) -> None:
    """Run the trtllm-modeling-v2 workflow as a CLI."""
    # Before anything changes the environment: a reload re-executes under this one.
    relaunch = _relauncher(sys.orig_argv, os.environ) if argv is None else None
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] in SUBCOMMANDS:
        sys.exit(SUBCOMMANDS[argv[0]](argv[1:]))
    if argv and argv[0] == "tune":
        # `trtllm-modeling-v2 tune --task T --workspace W` == `--stage tune`.
        argv = [*argv[1:], "--stage", "tune"]
    args = _parse_args(argv)
    if args.wait_for_spool > 0:
        # A derived task takes its environment from the site.
        spec = args.site if args.checkpoint is not None else args.task
        _wait_for_spool(Path(spec), args.wait_for_spool * 60)

    # Validate at the CLI boundary: an invalid spec must abort before any
    # agent is constructed.
    from .spec.intake.schema import IntakeError

    lock = None
    request = None
    try:
        if args.checkpoint is not None:
            # The request is written into the workspace, so it is claimed first.
            lock = _claim_workspace(Path(args.workspace))
            resolved, request = _derive_intake_task(
                Path(args.checkpoint),
                Path(args.site),
                Path(args.workspace),
                goal=args.goal,
                confirm_by=args.confirm,
                clean=args.clean,
            )
            args.task = str(resolved)
        else:
            args.task = str(_resolve_intake_task(Path(args.task), args.site, Path(args.workspace)))
        task_data = load_and_validate_task_yaml(args.task)
    except (TaskSchemaError, IntakeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(2)
    identity = _commit_identity(task_data)
    # Pinned for the rest of the process, so every commit the run makes uses
    # the same identity.
    os.environ[ENV_NAME], os.environ[ENV_EMAIL] = identity.name, identity.email

    if lock is None:
        _claim_workspace(Path(args.workspace))

    if args.manage_sessions:
        from .infra.substrate.managed import ManagedSessions

        managed = ManagedSessions(task_data, Path(args.workspace))
        why = managed.unavailable()
        if why:
            print(f"error: --manage-sessions: {why}", file=sys.stderr)
            sys.exit(2)
        for line in managed.tick():
            print(f"→ {line}")

    transport = None
    if args.agent_transport == "external":
        from .infra.external import ExternalAgentTransport
        from .infra.external.driver import install_instructions

        transport = ExternalAgentTransport(
            args.workspace,
            timeout_s=args.agent_turn_timeout,
            heartbeat_s=ORCHESTRATOR_HEARTBEAT_S,
        )
        from .infra.driver.config import DriverConfig

        guide = install_instructions(
            DriverConfig(
                workspace=Path(args.workspace).resolve(),
                root=Path(task_data.get("trtllm_repo_path") or ".").resolve(),
                task=task_data,
            )
        )
        print(
            f"→ role turns go through {transport.root}; a driver session must "
            f"answer each `*.request.json` with a `*.result.md` and a `*.done`.\n"
            f"  Driver loop: `trtllm-modeling-v2 driver --workspace {args.workspace} wait`; "
            f"standing instructions for its subagents: {guide}"
        )
    else:
        # Turns run here, but a driver watching the run's sessions still tells
        # a crashed orchestrator from a busy one by its heartbeat.
        start_orchestrator_heartbeat(args.workspace, ORCHESTRATOR_HEARTBEAT_S)

    # Claude Code's auto-memory is shared by every worktree of one repository: one
    # task's notes would reach the others, and one run's the next.
    os.environ.setdefault("CLAUDE_CODE_DISABLE_AUTO_MEMORY", "1")

    from .run.workflow import TrtllmModelingV2Workflow

    with TrtllmModelingV2Workflow(
        workspace=args.workspace,
        max_parallel_tasks=args.max_parallel_tasks,
        revision_cap=args.revision_cap,
        clean=args.clean,
        dry_run=args.dry_run,
        skip_substrate_gate=args.skip_substrate_gate,
        agent_transport=transport,
        agent_turn_timeout=args.agent_turn_timeout,
        entry_stage=args.entry_stage,
        task_worktrees=args.task_worktrees,
        event_window_s=args.event_window_s,
        primers=args.primers,
        manage_sessions=args.manage_sessions,
    ) as workflow:
        workflow.reexec = relaunch
        # Phase 0 runs the intake checks on the request the run was derived from.
        workflow.intake_request = request
        workflow.run(args.task)


if __name__ == "__main__":
    main()
