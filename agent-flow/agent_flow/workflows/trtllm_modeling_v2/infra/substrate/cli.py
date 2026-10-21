# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``trtllm-modeling-v2 session ...``: render, inspect and hand over compute sessions.

::

    trtllm-modeling-v2 session render     --task T --name N [--out PATH | --stdout]
    trtllm-modeling-v2 session status     --task T [--json]
    trtllm-modeling-v2 session requests   --task T [--json]
    trtllm-modeling-v2 session submission --task T --name N [--after JOB]
    trtllm-modeling-v2 session allocate   --task T --name N
    trtllm-modeling-v2 session ack        --task T --name N --job ID [--after JOB] [--by WHO]
    trtllm-modeling-v2 session watchdog   --task T [--timeout S] [--stale S] [--ending S]

The task file is read for its ``environment`` block only, so a session can be
rendered before the spool it will create exists.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml

from . import sessions, watchdog


class SessionConfigError(ValueError):
    """The task file's session configuration cannot be used."""


def load_session_config(path: str | Path) -> dict[str, Any]:
    """Read ``path`` and validate only what sessions need."""
    try:
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise SessionConfigError(f"cannot read {path}: {exc}") from exc
    if not isinstance(data, dict) or not isinstance(data.get("environment"), dict):
        raise SessionConfigError(f"{path} has no environment block")
    errors: list[str] = []
    sessions.validate_sessions(data["environment"], errors)
    if errors:
        raise SessionConfigError("invalid session configuration:\n  - " + "\n  - ".join(errors))
    if not sessions.sessions_from_task(data):
        raise SessionConfigError(f"{path} declares no session (environment.sessions)")
    return data


def check_session_tmpdir(task: dict[str, Any], name: str) -> None:
    """Hold the environment session ``name`` exports to preflight's ``TMPDIR`` rule."""
    from ...run.preflight import PreflightError, check_tmpdir

    session = sessions.session_by_name(task, name)
    env = {str(k): str(v) for k, v in (task["environment"].get("env") or {}).items()}
    env.update(session.env)
    try:
        check_tmpdir(env, require_set=False)
    except PreflightError as exc:
        raise SessionConfigError(f"session {name!r}: {exc}") from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="trtllm-modeling-v2 session", description=__doc__.splitlines()[0]
    )
    sub = parser.add_subparsers(dest="action", required=True)

    def add(name: str, *, named: bool, help_text: str) -> argparse.ArgumentParser:
        command = sub.add_parser(name, help=help_text)
        command.add_argument("--task", required=True, type=Path, help="the run's task.yaml")
        if named:
            command.add_argument("--name", required=True, help="session name")
        return command

    render = add("render", named=True, help_text="write the session's entry script")
    render.add_argument(
        "--out", type=Path, help="where to write it (default <spool>/session_entry.sh)"
    )
    render.add_argument("--stdout", action="store_true", help="print it instead of writing it")
    status = add("status", named=False, help_text="heartbeat, end and queue of every session")
    status.add_argument("--json", action="store_true")
    requests = add("requests", named=False, help_text="successor requests nobody acknowledged")
    requests.add_argument("--json", action="store_true")
    submission = add("submission", named=True, help_text="the broker call that starts the session")
    submission.add_argument("--after", help="job id the new allocation starts after")
    add("allocate", named=True, help_text="the broker call that starts a session's successor")
    ack = add("ack", named=True, help_text="record the job id of a requested successor")
    ack.add_argument("--job", required=True)
    ack.add_argument(
        "--after",
        help="the allocation the job was submitted to follow (default: the live watcher's)",
    )
    ack.add_argument("--by", default="driver")
    dog = add("watchdog", named=False, help_text="block until a session needs attention")
    dog.add_argument("--timeout", type=float, default=3000.0)
    dog.add_argument("--stale", type=float, default=180.0)
    dog.add_argument("--ending", type=float, default=1800.0)
    return parser


def _format_status(rows: list[dict]) -> str:
    lines = []
    for row in rows:
        beat = "never" if row["heartbeat_age_s"] is None else f"{row['heartbeat_age_s']:.0f}s ago"
        remaining = (
            "" if row["remaining_s"] is None else f" ends in {row['remaining_s'] / 3600:.1f}h"
        )
        handover = f" handover={json.dumps(row['handover'])}" if row.get("handover") else ""
        lines.append(
            f"{row['name']:<10} {row['role']:<12} job={row['slurm_job'] or '-':<10} "
            f"heartbeat {beat}{remaining} running={row['running']} pending={row['pending']}"
            f"{handover}  {row['spool']}"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        task = load_session_config(args.task)
        if args.action in ("render", "submission", "allocate"):
            check_session_tmpdir(task, args.name)
        if args.action == "render":
            if args.stdout:
                sys.stdout.write(
                    sessions.render_session_entry(task, args.name, source=str(args.task))
                )
            else:
                path = sessions.install_session_entry(
                    task, args.name, out=args.out, source=str(args.task)
                )
                print(path)
            return 0
        if args.action == "status":
            rows = sessions.session_status(task)
            print(json.dumps(rows, indent=2) if args.json else _format_status(rows))
            return 0
        if args.action == "requests":
            pending, ignored = sessions.review_requests(task)
            if args.json:
                print(json.dumps(pending, indent=2))
                return 0
            if not pending:
                print("(no successor requests)")
            for request in pending:
                print(
                    f"{request['session']}: allocation {request.get('slurm_job')} ends "
                    f"{request.get('ends_at')}; requested {request.get('requested_at')} "
                    f"({request['request_file']})"
                )
            for note in ignored:
                print(note)
            return 0
        if args.action == "submission":
            print(sessions.broker_submission(task, args.name, after_job=args.after))
            return 0
        if args.action == "allocate":
            plan = sessions.plan_successor(task, args.name)
            ack = (
                f"trtllm-modeling-v2 session ack --task {args.task} --name {plan.session} "
                f"--job <id>" + (f" --after {plan.after}" if plan.after else "")
            )
            print(f"Submit this through the broker, then record it with `{ack}`:\n")
            print(plan.call)
            print("\n" + "\n".join([plan.reason, *plan.notes]))
            return 0
        if args.action == "watchdog":
            spools = [s.spool for s in sessions.sessions_from_task(task)]
            alerts, lines = watchdog.watch(
                spools, timeout_s=args.timeout, stale_s=args.stale, ending_s=args.ending
            )
            for alert in alerts:
                print(alert.render())
            if not alerts:
                print("OK")
            print("\n".join(lines))
            return 0
        if args.action == "ack":
            session = sessions.session_by_name(task, args.name)
            print(
                sessions.acknowledge(
                    session.spool,
                    job_id=args.job,
                    by=args.by,
                    session=session.name,
                    after=args.after,
                )
            )
            return 0
    except (SessionConfigError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 2


__all__ = [
    "SessionConfigError",
    "build_parser",
    "check_session_tmpdir",
    "load_session_config",
    "main",
]
