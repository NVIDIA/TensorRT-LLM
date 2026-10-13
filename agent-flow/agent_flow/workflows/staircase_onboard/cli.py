# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from .agent_home import ensure_agent_home
from .committer import (
    ENV_EMAIL,
    ENV_NAME,
    CommitIdentity,
    CommitPolicy,
    CommitPolicyError,
    resolve_commit_identity,
    trial_commit,
)
from .roles import shares_the_drafters_model
from .state import STATE_FILENAME
from .task_schema import TaskSchemaError, load_and_validate_task_yaml
from .transport import (
    DEFAULT_ROLE_TURN_TIMEOUTS,
    EXTERNAL_MAX_PARALLEL_SMITHS,
    ORCHESTRATOR_HEARTBEAT_S,
    ExternalAgentTransport,
)
from .workflow import DEFAULT_MAX_PARALLEL_SMITHS


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="staircase-onboard",
        description=(
            "Bring one checkpoint to a gated modeling_v2 target: "
            "preflight -> plan -> smith -> assemble -> deliver."
        ),
        epilog="Other commands: `staircase-onboard premerge --help` (the pre-merge pack).",
    )
    parser.add_argument(
        "--task",
        type=Path,
        required=True,
        help="Path to the task YAML file. It is validated before any agent "
        "is constructed and copied verbatim into `<workspace>/task.yaml`, "
        "which every agent reads.",
    )
    parser.add_argument(
        "--workspace",
        type=Path,
        default=Path("workspace/staircase-onboard"),
        help="Workspace directory for the run's shared state (task.yaml, "
        "plan.md, vocabulary.yaml, progress.yaml, journal.md, reports/). "
        "Created on demand.",
    )
    parser.add_argument(
        "--max-plan-rounds",
        type=int,
        default=1,
        help="How many PlanDrafter<->PlanReviewer rounds to spend before "
        "proceeding regardless. Default 1: draft, review, and at most one "
        "re-draft. Reaching the cap is not a failure — the reviewer's "
        "unresolved items are written into plan.md as open risks and the "
        "build starts. A plan review never blocks the build.",
    )
    parser.add_argument(
        "--max-assembler-turns-per-unit",
        type=int,
        default=6,
        help="Bound on the assembler's self-heal loop for one unit (the "
        "core assembly, or one feature increment). Exhausting it ends the "
        "run with the audit trail rather than looping indefinitely.",
    )
    parser.add_argument(
        "--max-parallel-smiths",
        type=int,
        default=None,
        help="Cap on concurrent smith dispatches within one dependency "
        f"batch. Default {DEFAULT_MAX_PARALLEL_SMITHS}, or "
        f"{EXTERNAL_MAX_PARALLEL_SMITHS} with --agent-transport external, where "
        "the binding limit is concurrent agent streams on one account. Catalog entries are "
        "independent by construction, so this is where a run spends extra "
        "compute to shorten wall-clock — but each dispatch is a full agent "
        "session, and the upstream rate limit is reported without any "
        "backoff, so a cap that is hit fails the whole batch together. "
        "0 means 'as many as the batch holds' and is not recommended: on "
        "a typical queue that is every entry at once.",
    )
    parser.add_argument(
        "--reviewer",
        choices=("off", "light", "full"),
        default="light",
        help="How much reviewing each assembled unit gets. 'light' "
        "(default) reads the diff, the boot-gate output and the vocabulary "
        "audit and runs nothing — the expensive evidence already exists "
        "because the gates ran. 'full' additionally re-runs the gates "
        "itself. 'off' skips the reviewer turn entirely.",
    )
    parser.add_argument(
        "--plan",
        type=Path,
        default=None,
        help="Use a pre-written plan instead of running the PlanDrafter. "
        "Combined with --vocabulary the plan phase is skipped entirely and "
        "the run starts at the smith queue. Ignored when resuming from a "
        "checkpoint; pass --clean to start fresh.",
    )
    parser.add_argument(
        "--vocabulary",
        type=Path,
        default=None,
        help="Use a pre-written vocabulary.yaml (the typed step-to-catalog "
        "mapping and dispatch queue) instead of deriving one. See --plan.",
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
        "--skip-substrate-gate",
        action="store_true",
        help="Skip preflight's substrate gate — the cold start that proves "
        "this host generates a token on the STOCK modeling path under this "
        "run's runtime block. It is the single highest-value preflight "
        "check and skipping it means the three failures it catches "
        "(MPI_ERR_SPAWN, kvCache sizing, NVRTC's missing cuda.h) resurface "
        "hours later inside an agent's turn, reading as target defects. "
        "Use it only when you have already proven the substrate in this "
        "same environment, or when the gate itself is what is broken. It is "
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
        "machine, the queue, decision validation, all three gates — stays "
        "here either way.",
    )
    parser.add_argument(
        "--overlap-units",
        action="store_true",
        help="Start the next unit's assembler in its own git worktree as soon as the "
        "current unit is UNIT_READY, while the current unit is still being gated. The "
        "early result is adopted only if it ended UNIT_READY, committed cleanly, rebases "
        "onto HEAD without conflict and fast-forwards the checkout; anything else falls "
        "back to a sequential assembly and keeps the worktree for inspection. Off by "
        "default.",
    )
    parser.add_argument(
        "--stage",
        dest="entry_stage",
        choices=["", "tune"],
        default="",
        help="Start at a later stage. `tune` runs the performance stage (ground truth, "
        "config ladder, tuner candidates) on an already-onboarded target, or re-enters a "
        "delivered run at it. Same as the `tune` subcommand. Needs a `perf:` block.",
    )
    parser.add_argument(
        "--agent-turn-timeout",
        type=int,
        default=None,
        help="Seconds a single externally-dispatched turn may take before "
        "the run fails, for every role. Without it each role has its own bound ("
        + ", ".join(f"{role} {int(s) // 3600}h" for role, s in DEFAULT_ROLE_TURN_TIMEOUTS.items())
        + "). Only meaningful with --agent-transport external.",
    )
    return parser.parse_args(argv)


def _commit_identity(task_data: dict) -> CommitIdentity:
    """The identity every commit of the run is authored and signed off under.

    The repository's configured user, read before $HOME is redirected
    (redirecting it takes the real ~/.gitconfig off git's search path), and
    proven by one trial commit through the generated helper: a commit that
    lands without a sign-off matching its author fails the target
    repository's DCO check, and finding that out after the run has made fifty
    of them is the expensive way. Exits 2 when there is no identity, or when
    this environment's git changes what the helper writes.
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

    Two orchestrators on one workspace is not a race you notice: they share
    the checkpoint file, the dispatch directory and the id counter, so they
    mint the SAME request id, overwrite each other's request JSON, and both
    unblock on one answer. Measured on this run -- a stale process that a
    `pkill` had failed to kill raced a fresh one for three seconds and the
    survivor's request was the older code's, missing the system-prompt
    pointer the driver needed.

    The handle is kept in a module-level list rather than returned-and-hoped,
    because `flock` is held by an OPEN FILE DESCRIPTION: let the file object
    be collected and the lock silently disappears. A caller that writes
    `_claim_workspace(ws)` without binding the result would take a lock that
    releases before the first agent turn -- which is worse than no lock,
    since it reads as protection.

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
            f"error: another staircase-onboard process already holds {path}.\n"
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


def _premerge(argv: list[str]) -> int:
    from .premerge.cli import main as premerge_main

    return premerge_main(argv)


def _gate(argv: list[str]) -> int:
    from .gate_cli import main as gate_main

    return gate_main(argv)


#: ``staircase-onboard <subcommand> ...`` for the tools that are not a run.
#: A run is still ``staircase-onboard --task ...``; a first argument named
#: here dispatches instead, and everything after it is the subcommand's own.
def _session(argv: list[str]) -> int:
    from .substrate.cli import main as session_main

    return session_main(argv)


def _driver(argv: list[str]) -> int:
    from .driver.cli import main as driver_main

    return driver_main(argv)


def _perf(argv: list[str]) -> int:
    """Perf tooling outside a run: the config ladder, the campaign, curve tables."""
    from .tune.cli import main as perf_main

    return perf_main(argv)


SUBCOMMANDS = {
    "premerge": _premerge,
    # Perf tooling: `perf ladder|campaign|table ...` (see tune/cli.py).
    "perf": _perf,
    "gate": _gate,
    # Compute sessions: render entry scripts, status, hand-over, watchdog.
    "session": _session,
    # The external-transport driver: wait, claim, prompt, status, instructions.
    "driver": _driver,
}


def main(argv: list[str] | None = None) -> None:
    """Run the staircase-onboard workflow as a CLI."""
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] in SUBCOMMANDS:
        sys.exit(SUBCOMMANDS[argv[0]](argv[1:]))
    if argv and argv[0] == "tune":
        # `staircase-onboard tune --task T --workspace W` == `--stage tune`: the
        # performance stage on an already-onboarded target, or a re-entry of a
        # delivered run.
        argv = [*argv[1:], "--stage", "tune"]
    args = _parse_args(argv)

    # Validate at the CLI boundary: an invalid spec must abort before any
    # agent is constructed, and the message must list every problem at
    # once so the user fixes them in a single pass.
    try:
        task_data = load_and_validate_task_yaml(args.task)
    except TaskSchemaError as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(2)
    if warning := shares_the_drafters_model(task_data):
        print(f"warning: {warning}", file=sys.stderr)
    identity = _commit_identity(task_data)
    # Pinned for the rest of the process, so the workflow resolves the same
    # identity after $HOME has been redirected below.
    os.environ[ENV_NAME], os.environ[ENV_EMAIL] = identity.name, identity.email

    # A sandboxed session's $HOME is read-only, and the first thing a spawned
    # `claude` CLI does is take a lock beside its credential store. Without
    # this every agent turn fails with `EROFS ... credentials.lock` and zero
    # tokens, which reads as a model failure rather than as "no agent ever
    # started". Returns None on an ordinary workstation, where nothing
    # changes.
    lock = _claim_workspace(Path(args.workspace))  # noqa: F841 -- held for the run

    staged = ensure_agent_home(
        Path(args.workspace),
        git_identity=(identity.name, identity.email),
    )
    if staged is not None:
        staged.apply()
        print(f"→ staged a writable agent home at {staged.root} ($HOME is not writable)")

    transport = None
    if args.agent_transport == "external":
        transport = ExternalAgentTransport(
            args.workspace,
            timeout_s=args.agent_turn_timeout,
            source_embargo=task_data.get("source_embargo") or (),
            repo_root=task_data.get("trtllm_repo_path"),
            # The driver tells a busy orchestrator from an exited one by this.
            heartbeat_s=ORCHESTRATOR_HEARTBEAT_S,
        )
        if args.max_parallel_smiths is None:
            args.max_parallel_smiths = EXTERNAL_MAX_PARALLEL_SMITHS
        from .driver.core import DriverConfig, install_instructions

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
            f"  Driver loop: `staircase-onboard driver --workspace {args.workspace} wait`; "
            f"standing instructions for its subagents: {guide}"
        )

    from .workflow import StaircaseOnboardWorkflow

    with StaircaseOnboardWorkflow(
        workspace=args.workspace,
        max_plan_rounds=args.max_plan_rounds,
        max_assembler_turns_per_unit=args.max_assembler_turns_per_unit,
        max_parallel_smiths=(
            DEFAULT_MAX_PARALLEL_SMITHS
            if args.max_parallel_smiths is None
            else args.max_parallel_smiths
        ),
        reviewer_mode=args.reviewer,
        plan=args.plan,
        vocabulary=args.vocabulary,
        clean=args.clean,
        dry_run=args.dry_run,
        skip_substrate_gate=args.skip_substrate_gate,
        agent_transport=transport,
        overlap_units=args.overlap_units,
        entry_stage=args.entry_stage,
    ) as workflow:
        workflow.run(args.task)


if __name__ == "__main__":
    main()
