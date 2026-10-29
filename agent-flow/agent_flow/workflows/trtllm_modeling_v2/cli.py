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
from .run.records.state import STATE_FILENAME
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
            "preflight -> execution stage -> deliver."
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
        '"mtp disagg@tp1/tp2 tp1 tep4 dep4 isl=8192 osl=1024 tune" (spec/derive.py). '
        "Features the checkpoint carries are built unless named `-<feature>`.",
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
        help="How many graph revisions the planner may make without progress (a task "
        "done, a change landed on the target or a contract check turning green) before "
        f"the run closes planner-stalled. Default {DEFAULT_REVISION_CAP}.",
    )
    parser.add_argument(
        "--event-window",
        dest="event_window_s",
        type=float,
        default=DEFAULT_EVENT_WINDOW_S,
        metavar="SECONDS",
        help="While work is in flight, batch ordinary reports, green finishes and nonblocking "
        "task failures for this many seconds. Failures blocking pending hard dependencies, "
        "operator messages, failed re-checks, or an idle graph wake the planner at once. "
        f"Default {DEFAULT_EVENT_WINDOW_S:g}.",
    )
    parser.add_argument(
        "--no-primers",
        dest="primers",
        action="store_false",
        help="Start every planner, kernel writer and assembler turn afresh instead of as a fork "
        "of one briefing session per role, which read what all of them read first "
        "(run/startup/primers.py; a turn that cannot fork reads the session's summary).",
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
    import yaml

    from .spec import derive as derive_mod

    path = workspace / derive_mod.SPEC_FILENAME
    if path.is_file() and not clean:
        print(f"→ {path}: the spec derived before, as it stands; --clean derives it again")
        return path
    site_data = yaml.safe_load(site.read_text(encoding="utf-8")) or {}
    derivation = derive_mod.derive_spec(
        checkpoint, site_data, derive_mod.parse_goal(goal), site_path=site
    )
    derive_mod.write_derivation(workspace, derivation)
    print(f"→ derived {path} (see {workspace / derive_mod.REPORT_FILENAME})")
    if derivation.questions:
        raise derive_mod.DeriveError(
            f"{path} leaves {len(derivation.questions)} question(s) open; answer them in it and "
            "run the same command again:\n  - " + "\n  - ".join(derivation.questions)
        )
    return path


def _lazy_main(module: str) -> Callable[[list[str]], int]:
    """``module``'s ``main``, imported only when its subcommand runs."""

    def run(argv: list[str]) -> int:
        return importlib.import_module(module, __package__).main(argv)

    return run


def _metrics(argv: list[str]) -> int:
    import json

    from .run.records import metrics
    from .run.records.journal import Journal

    parser = argparse.ArgumentParser(
        prog="trtllm-modeling-v2 metrics",
        description="What a run cost and proved so far: milestones, cells, checks by gate with "
        "their device-hours, turns by role, refused graphs, operator messages.",
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
        "puts it first in the planner's next turn. A message steers the work in a few "
        f"sentences (at most {inbox.MESSAGE_MAX_CHARS} characters; evidence in a file it "
        "names); it relaxes no check, and is no place for status reports.",
    )
    parser.add_argument("--workspace", type=Path, required=True, help="The run's workspace.")
    text = parser.add_mutually_exclusive_group(required=True)
    text.add_argument("message", nargs="?", help="The message.")
    text.add_argument("--file", type=Path, help="Read the message from this file.")
    parser.add_argument(
        "--now",
        action="store_true",
        help="Wake the planner for it at once; otherwise it rides with the planner's next turn.",
    )
    args = parser.parse_args(argv)
    message = args.file.read_text(encoding="utf-8") if args.file else args.message
    try:
        path = inbox.post(args.workspace, message, now=args.now)
    except ValueError as exc:
        parser.error(str(exc))
    print(f"left for the planner's {'turn, now' if args.now else 'next turn'}: {path}")
    return 0


#: ``trtllm-modeling-v2 <subcommand> ...`` for the tools that are not a run.
#: A run is ``trtllm-modeling-v2 --task ...``; a first argument named here
#: dispatches instead, and everything after it is the subcommand's own.
SUBCOMMANDS = {
    # From a request to a spec, before launch: `intake derive|lint ...` (spec/INTAKE.md).
    "intake": _lazy_main(".spec.intake"),
    "gate": _lazy_main(".checks.gate_cli"),
    # Compute sessions: render entry scripts, hand-over.
    "session": _lazy_main(".infra.compute.sessions"),
    # Jobs in a session's spool: `spool submit|launch|wait|status|cancel ...`.
    "spool": _lazy_main(".infra.compute.spoolctl"),
    # The operator's loop of a run: wait, status.
    "monitor": _lazy_main(".infra.monitor.status"),
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

    from .agents.session_model import ModelSelectionError
    from .run.workflow import TrtllmModelingV2Workflow

    with TrtllmModelingV2Workflow(
        workspace=args.workspace,
        max_parallel_tasks=args.max_parallel_tasks,
        revision_cap=args.revision_cap,
        clean=args.clean,
        dry_run=args.dry_run,
        skip_substrate_gate=args.skip_substrate_gate,
        agent_turn_timeout=args.agent_turn_timeout,
        entry_stage=args.entry_stage,
        event_window_s=args.event_window_s,
        primers=args.primers,
    ) as workflow:
        try:
            workflow.run(args.task)
        except ModelSelectionError as exc:
            print(f"error: {exc}", file=sys.stderr)
            sys.exit(2)
        except KeyboardInterrupt:
            # The turns and checks in flight are not waited for: the next start runs them again.
            logging.shutdown()
            sys.stdout.flush()
            sys.stderr.flush()
            heartbeat.mark_exit()
            os._exit(130)


if __name__ == "__main__":
    main()
