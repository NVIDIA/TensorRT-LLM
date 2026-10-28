#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Submit to, wait on, inspect and cancel jobs in a compute-session spool.

The client half of :mod:`scheduler`: standard-library only and runnable by path, from any shell
with a ``python3`` (``trtllm-modeling-v2 spool`` is the same program)::

    spoolctl.py submit [--gpus N|all] [--wall 2h] [--tree DIR] <name> <command...>  # run, wait
    spoolctl.py launch [--gpus N|all] [--wall 2h] [--tree DIR] <name> <command...>  # queue
    spoolctl.py wait [--session-wait 15m] <name>
    spoolctl.py status [--json]
    spoolctl.py cancel <name>

The spool is ``--spool`` (before or after the action), else ``$SPOOL``, else ``$MV2FLOW_SPOOL``.
The owner prefix is ``--owner``, else ``$MV2FLOW_JOB_PREFIX``; when one is set a job name must
start with it, and a name whose previous result belongs to another owner is refused.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import sys
import time
import uuid
from pathlib import Path
from typing import Mapping

try:
    from .. import files as _files
    from . import scheduler as _sched
except ImportError:  # run by path: the sibling module is on sys.path[0]
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import files as _files
    import scheduler as _sched  # type: ignore[no-redef]

#: Exit code when the session is not there to run the job.
EXIT_NO_SESSION = 70
#: Exit code when a submission is refused before it is queued.
EXIT_REFUSED = 64

#: A watcher that has not heart-beat for this long is gone.
WATCHER_STALE_S = _sched.WATCHER_STALE_S


class SpoolError(RuntimeError):
    """A submission that cannot be queued, or a wait that cannot finish."""

    def __init__(self, message: str, exit_code: int = EXIT_REFUSED) -> None:
        super().__init__(message)
        self.exit_code = exit_code


def render_job(
    command: str,
    *,
    header: Mapping[str, object] | None = None,
    cwd: str | None = None,
    env: Mapping[str, str] | None = None,
) -> str:
    """The script text for one job: shebang, header, working directory, exports, command."""
    lines = ["#!/bin/bash", "set -u"]
    fields = [f"{key}={value}" for key, value in (header or {}).items() if value not in (None, "")]
    if fields:
        lines.append(f"{_sched.HEADER_TAG} " + " ".join(fields))
    if cwd:
        lines.append(f"cd {shlex.quote(cwd)} || exit 1")
    for key, value in sorted((env or {}).items()):
        lines.append(f"export {key}={shlex.quote(str(value))}")
    lines.append(command)
    return "\n".join(lines) + "\n"


def check_name(name: str, owner: str | None) -> None:
    """Refuse a name the scheduler would refuse, or one outside the owner's prefix."""
    if not _sched.NAME_PATTERN.match(name):
        raise SpoolError(f"job name {name!r} must match {_sched.NAME_PATTERN.pattern}")
    if owner:
        bare = name[len(_sched.FREE_PREFIX) :] if name.startswith(_sched.FREE_PREFIX) else name
        if not (bare == owner or bare.startswith(f"{owner}-")):
            raise SpoolError(
                f"job name {name!r} must start with this requester's prefix {owner!r} "
                f"(as '{owner}-<what>' or 'free-{owner}-<what>'); names are issued per "
                f"request so two requesters can never overwrite each other's results"
            )


def queue_job(
    spool: str | Path,
    name: str,
    script: str,
    *,
    owner: str | None = None,
    require_watcher: bool = True,
) -> None:
    """Rename ``script`` into ``pending/`` after checking the name is free to use.

    Only the owner that used a name may reuse it (its previous results go to ``history/``), and
    never while it is pending or running.
    """
    paths = _sched.SpoolPaths(spool)
    if not paths.root.is_dir():
        raise SpoolError(f"spool {paths.root} does not exist", EXIT_NO_SESSION)
    if require_watcher:
        age = _sched.watcher_age(paths.root)
        if age is None:
            raise SpoolError(
                f"no watcher heartbeat in {paths.root}: the session is not running",
                EXIT_NO_SESSION,
            )
        if age > WATCHER_STALE_S:
            raise SpoolError(
                f"the watcher heartbeat in {paths.root} is {age:.0f}s stale: the session is gone",
                EXIT_NO_SESSION,
            )
    check_name(name, owner)
    if (paths.pending / f"{name}.sh").exists() or (paths.running / f"{name}.sh").exists():
        raise SpoolError(f"job {name!r} is already queued or running in {paths.root}")
    previous = _files.read_json(paths.record(name))
    if previous is not None and (previous.get("owner") or "") != (owner or ""):
        raise SpoolError(
            f"job name {name!r} already holds a result owned by "
            f"{previous.get('owner') or '<no owner>'!r}; pick a name under your own prefix"
        )
    paths.ensure()
    paths.archive(name)
    # A cancel request of an earlier job of this name is not this one's.
    (paths.cancel / name).unlink(missing_ok=True)
    _files.write_text_atomic(paths.pending / f"{name}.sh", script)


def wait_for_job(
    spool: str | Path,
    name: str,
    *,
    token: str | None = None,
    poll_s: float = 3.0,
    timeout_s: float | None = None,
    stale_s: float = WATCHER_STALE_S,
    grace_s: float = 300.0,
    unwatched_s: float | None = None,
    submitted_at: float | None = None,
) -> dict:
    """Block until ``name`` has a receipt (matching ``token`` when given) and return its record.

    ``timeout_s`` counts from the job's START, and ``grace_s`` past it is the session's to stop the
    job and write the receipt. A spool no watcher ever heart-beat on fails at once, or
    ``unwatched_s`` after submission (a session still starting). A job re-run after a lost
    allocation is waited for through its re-run.

    Raises :class:`SpoolError`, with :data:`EXIT_NO_SESSION` when the session, not the job, ended
    the wait: the watchers stopped heart-beating for ``stale_s``, or none ever watched.
    """
    paths = _sched.SpoolPaths(spool)
    submitted = time.time() if submitted_at is None else submitted_at
    while True:
        receipt = paths.receipt(name)
        if receipt.exists():
            record = _files.read_json(paths.record(name)) or {}
            if not token or not record.get("token") or record.get("token") == token:
                try:
                    record.setdefault("rc", int(receipt.read_text(encoding="utf-8").strip()))
                    return record
                except (OSError, ValueError):
                    pass  # created but not yet written: keep waiting
        now = time.time()
        meta = _files.read_json(paths.running / f"{name}.json") or {}
        started = float(meta["started_epoch"]) if meta.get("started_epoch") else None
        age = _sched.watcher_age(paths.root, now)
        if started is not None:
            if timeout_s and now > started + timeout_s + grace_s:
                raise SpoolError(
                    f"job {name!r} ran past its {timeout_s:g}s deadline (counted from its "
                    f"start) and the session did not close it",
                    _sched.RC_WALL_LIMIT,
                )
        elif age is None and (unwatched_s is None or now > submitted + unwatched_s):
            raise SpoolError(
                f"nothing started job {name!r} and no session watches {paths.root}",
                EXIT_NO_SESSION,
            )
        if age is not None and age > stale_s:
            raise SpoolError(
                f"the session watching {paths.root} stopped heart-beating {age:.0f}s ago",
                EXIT_NO_SESSION,
            )
        time.sleep(poll_s)


def left_in_spool(spool: str | Path, name: str) -> str:
    """What becomes of ``name`` when its session stopped under it with it still in the spool."""
    paths = _sched.SpoolPaths(spool)
    for where in ("running", "pending"):
        if (paths.root / where / f"{name}.sh").exists():
            return (
                f"{name} is still in {where}/: a session that starts on this spool (the "
                f"allocation requeued, or a successor) runs it, or closes it out node_lost if it "
                f"may not run again; `spoolctl.py wait {name}` returns that. Submitting it "
                f"again would run it twice"
            )
    return ""


def cancel_job(spool: str | Path, name: str) -> Path:
    """Ask every scheduler on ``spool`` to drop ``name``; returns the request file.

    Right in any state: a scheduler refuses the job if it has not started, stops it under the
    safe-kill rule if it has, and a session not running yet sees the request when it starts.
    """
    paths = _sched.SpoolPaths(spool)
    paths.cancel.mkdir(parents=True, exist_ok=True)
    flag = paths.cancel / name
    flag.write_text(f"{time.time()}\n", encoding="utf-8")
    return flag


def status(spool: str | Path) -> dict:
    """Watcher snapshots (including stale ones), plus pending and running job names."""
    paths = _sched.SpoolPaths(spool)
    now = time.time()
    watchers = []
    for path in sorted(paths.watchers.glob("*.json")):
        record = _files.read_json(path)
        if record:
            record["age_s"] = round(now - float(record.get("alive_epoch", 0)), 1)
            watchers.append(record)
    return {
        "spool": str(paths.root),
        "watchers": watchers,
        "pending": sorted(p.name[: -len(".sh")] for p in paths.pending.glob("*.sh")),
        "running": sorted(p.name[: -len(".sh")] for p in paths.running.glob("*.sh")),
    }


def format_status(data: Mapping[str, object]) -> str:
    """The human view of :func:`status`."""
    lines = [f"spool {data['spool']}"]
    watchers = data.get("watchers") or []
    if not watchers:
        lines.append("  no watcher has ever heart-beat here")
    for record in watchers:
        stale = " STALE" if float(record.get("age_s", 0)) > WATCHER_STALE_S else ""
        handover = record.get("handover") or {}
        lines.append(
            f"  watcher {record.get('watcher')} session={record.get('session') or '-'} "
            f"job={record.get('slurm_job')} heartbeat {record.get('age_s')}s ago{stale}; "
            f"devices {','.join(record.get('devices') or []) or '-'} "
            f"free {','.join(record.get('free') or []) or '-'}"
            + (
                f" quarantined {','.join(record['quarantined'])}"
                if record.get("quarantined")
                else ""
            )
            + (
                f" in hand-over, successor {handover.get('successor') or 'none yet'}"
                if handover
                else ""
            )
        )
        for job in record.get("running") or []:
            lines.append(
                f"    RUN  {job['name']:<40} gpus={job['gpus']:<3} dev={','.join(job['devices']) or '-':<8}"
                f" {job['elapsed_s']:>8.0f}s/{job['wall_s'] or '-'} rss={job['rss_gb']}G {job['state']}"
                + (f" (re-run {job['reruns']})" if job.get("reruns") else "")
            )
        for job in record.get("pending") or []:
            lines.append(
                f"    WAIT {job['name']:<40} gpus={job['gpus']:<3} prio={job['effective_priority']}"
                f" {job['waiting_s']:>8.0f}s  "
                + (f"(re-run {job['reruns']}) " if job.get("reruns") else "")
                + str(job["reason"])
            )
    return "\n".join(lines)


def _spool_from(args: argparse.Namespace) -> str:
    spool = args.spool or os.environ.get("SPOOL") or os.environ.get("MV2FLOW_SPOOL")
    if not spool:
        raise SpoolError("no spool: pass --spool or set $SPOOL", EXIT_NO_SESSION)
    return spool


def _add_spool_option(parser: argparse.ArgumentParser) -> None:
    # Also accepted after the action; left unset there so it never hides one given before it.
    parser.add_argument("--spool", default=argparse.SUPPRESS, help=argparse.SUPPRESS)


def _add_job_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--gpus", help="device count, or 'all' for the whole node")
    parser.add_argument("--wall", help="wall limit from start (7200, 90m, 2h, HH:MM:SS); 0 = none")
    parser.add_argument("--owner", help="requester prefix (default $MV2FLOW_JOB_PREFIX)")
    parser.add_argument(
        "--tree",
        help="checkout to run against: working directory AND first PYTHONPATH entry; the "
        "receipt is refused if any process imported tensorrt_llm from elsewhere",
    )
    parser.add_argument(
        "--session-wait",
        help="how long past the session's heartbeat going stale to keep waiting for a "
        "session to come back and run the job (90s, 15m, ...; default 0)",
    )
    parser.add_argument("name")
    parser.add_argument("command", nargs=argparse.REMAINDER)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="spoolctl.py", description=__doc__.splitlines()[0])
    parser.add_argument("--spool", help="spool directory (default $SPOOL, then $MV2FLOW_SPOOL)")
    sub = parser.add_subparsers(dest="action", required=True)
    for action in ("submit", "launch"):
        _add_job_options(sub.add_parser(action))
    wait = sub.add_parser("wait")
    wait.add_argument("name")
    wait.add_argument("--token")
    wait.add_argument("--timeout", help="deadline from the job's start")
    wait.add_argument(
        "--session-wait",
        help="how long past the session's heartbeat going stale to keep waiting for a "
        "session to come back and run the job (90s, 15m, ...; default 0)",
    )
    show = sub.add_parser("status")
    show.add_argument("--json", action="store_true")
    cancel = sub.add_parser("cancel")
    cancel.add_argument("name")
    for action in sub.choices.values():
        _add_spool_option(action)
    return parser


def job_header(args: argparse.Namespace, owner: str | None, token: str) -> dict[str, object]:
    """The ``#spool:`` fields one submission's options ask for."""
    header: dict[str, object] = {}
    if args.gpus is not None:
        header["gpus"] = args.gpus
    if args.wall is not None:
        header["wall"] = int(_sched.parse_duration(args.wall))
    if owner:
        header["owner"] = owner
    if args.tree:
        header["tree"] = os.path.abspath(args.tree)
    header["token"] = token
    return header


def _submit(args: argparse.Namespace, *, wait: bool) -> int:
    spool = _spool_from(args)
    command = " ".join(arg for arg in args.command if arg != "--").strip()
    if not command:
        raise SpoolError("nothing to run: give a command after the job name")
    owner = args.owner or os.environ.get("MV2FLOW_JOB_PREFIX") or None
    token = uuid.uuid4().hex
    if args.tree and not os.path.isdir(args.tree):
        raise SpoolError(f"--tree {args.tree} is not a directory")
    cwd = (
        (os.path.abspath(args.tree) if args.tree else None)
        or os.environ.get("MV2FLOW_REPO")
        or os.getcwd()
    )
    script = render_job(command, header=job_header(args, owner, token), cwd=cwd)
    queue_job(spool, args.name, script, owner=owner)
    paths = _sched.SpoolPaths(spool)
    if not wait:
        print(
            f"queued {args.name}; log at {paths.log(args.name)}, rc at {paths.receipt(args.name)}"
        )
        return 0
    record = _wait(spool, args.name, token, None, args.session_wait)
    log = paths.log(args.name)
    if log.exists():
        sys.stdout.write(log.read_text(encoding="utf-8", errors="replace"))
    rc = int(record.get("rc", 1))
    reason = record.get("reason", "exit")
    detail = f" ({record['detail']})" if record.get("detail") else ""
    reruns = f" after {record['reruns']} re-run(s)" if record.get("reruns") else ""
    print(f"--- {args.name} rc={rc} reason={reason}{detail}{reruns}")
    return rc if 0 <= rc <= 255 else 1


def _wait(
    spool: str, name: str, token: str | None, timeout: str | None, session_wait: str | None
) -> dict:
    """The CLI's wait: through re-runs, and past a stale heartbeat for ``session_wait``."""
    try:
        extra = _sched.parse_duration(session_wait) if session_wait else 0.0
        deadline = _sched.parse_duration(timeout) if timeout else None
    except ValueError as exc:
        raise SpoolError(f"not a duration: {exc}") from exc
    try:
        return wait_for_job(
            spool, name, token=token, timeout_s=deadline, stale_s=WATCHER_STALE_S + extra
        )
    except SpoolError as exc:
        left = left_in_spool(spool, name) if exc.exit_code == EXIT_NO_SESSION else ""
        if left:
            raise SpoolError(f"{exc}; {left}", exc.exit_code) from exc
        raise


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.action in ("submit", "launch"):
            return _submit(args, wait=args.action == "submit")
        spool = _spool_from(args)
        if args.action == "wait":
            record = _wait(spool, args.name, args.token, args.timeout, args.session_wait)
            print(json.dumps(record, indent=2, sort_keys=True))
            return int(record.get("rc", 1)) if 0 <= int(record.get("rc", 1)) <= 255 else 1
        if args.action == "status":
            data = status(spool)
            print(json.dumps(data, indent=2, sort_keys=True) if args.json else format_status(data))
            return 0
        cancel_job(spool, args.name)
        print(f"cancel requested for {args.name}; the scheduler stops it under the safe-kill rule")
        return 0
    except SpoolError as exc:
        print(f"spoolctl: {exc}", file=sys.stderr)
        return exc.exit_code


__all__ = [
    "EXIT_NO_SESSION",
    "EXIT_REFUSED",
    "SpoolError",
    "WATCHER_STALE_S",
    "cancel_job",
    "check_name",
    "format_status",
    "left_in_spool",
    "queue_job",
    "render_job",
    "status",
    "wait_for_job",
]


if __name__ == "__main__":
    sys.exit(main())
