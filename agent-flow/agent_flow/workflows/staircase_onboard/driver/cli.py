# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``staircase-onboard driver ...``: the external transport's driver loop, as commands.

::

    staircase-onboard driver --workspace WS wait [--timeout 540] [--max-inflight 8]
    staircase-onboard driver --workspace WS pending
    staircase-onboard driver --workspace WS status [--json]
    staircase-onboard driver --workspace WS claim <id> [--by NAME]
    staircase-onboard driver --workspace WS release <id>
    staircase-onboard driver --workspace WS prompt <id> [--claim] [--by NAME]
    staircase-onboard driver --workspace WS answer <id> --from FILE
    staircase-onboard driver --workspace WS instructions [--out PATH | --stdout]

The loop a driver session runs::

    wait  -> NEW <id>               claim it, hand `prompt <id>` to a subagent
          -> ORCHESTRATOR_EXITED    read its log; relaunch it (it adopts unanswered turns)
          -> SESSION_REQUEST ...    submit the successor allocation it prints, then `session ack`
          -> SESSION_STALE|ENDING|QUARANTINED ...   look at that session
          -> TIMEOUT                call wait again

``wait`` prints the event on its first line and exits 0 for every event, so a
caller reads the first word.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from . import core


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="staircase-onboard driver", description=__doc__.splitlines()[0]
    )
    parser.add_argument(
        "--workspace",
        default=os.environ.get("STAIRCASE_WORKSPACE"),
        help="the run's workspace (default $STAIRCASE_WORKSPACE)",
    )
    parser.add_argument("--root", help="working root for relative paths (default trtllm_repo_path)")
    parser.add_argument(
        "--orchestrator-job",
        help="<spool>:<job name> when the orchestrator runs as a spool job; its receipt means it exited",
    )
    parser.add_argument("--max-inflight", type=int, default=core.DEFAULT_MAX_INFLIGHT)
    sub = parser.add_subparsers(dest="action", required=True)
    wait = sub.add_parser("wait", help="block until there is something to do")
    wait.add_argument("--timeout", type=float, default=540.0)
    wait.add_argument("--poll", type=float, default=10.0)
    wait.add_argument("--no-sessions", action="store_true", help="ignore the compute sessions")
    sub.add_parser("pending", help="unclaimed, unanswered turns")
    show = sub.add_parser("status", help="every turn, the orchestrator and the sessions")
    show.add_argument("--json", action="store_true")
    take = sub.add_parser("claim", help="take a turn for this driver")
    take.add_argument("id")
    take.add_argument("--by")
    give = sub.add_parser("release", help="give a claimed turn back")
    give.add_argument("id")
    prompt = sub.add_parser("prompt", help="the subagent instruction for one turn")
    prompt.add_argument("id")
    prompt.add_argument("--claim", action="store_true", help="claim the turn first")
    prompt.add_argument("--by")
    reply = sub.add_parser("answer", help="hand a turn back with a result file")
    reply.add_argument("id")
    reply.add_argument("--from", dest="source", required=True, type=Path)
    guide = sub.add_parser("instructions", help="render the run's TURN_INSTRUCTIONS")
    guide.add_argument("--out", type=Path)
    guide.add_argument("--stdout", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.workspace:
        print("error: pass --workspace or set STAIRCASE_WORKSPACE", file=sys.stderr)
        return 2
    try:
        cfg = core.load_config(
            args.workspace,
            root=args.root,
            max_inflight=args.max_inflight,
            orchestrator_job=args.orchestrator_job,
        )
        if args.action == "wait":
            event = core.wait(
                cfg, timeout_s=args.timeout, poll_s=args.poll, watch_sessions=not args.no_sessions
            )
            print(event.render())
            print(core.format_status(core.status(cfg)))
            return 0
        if args.action == "pending":
            rows = core.pending(cfg)
            print(
                "\n".join(f"{r['id']:<20} {r.get('role')}: {r.get('what')}" for r in rows)
                or "(nothing waiting)"
            )
            return 0
        if args.action == "status":
            data = core.status(cfg)
            print(json.dumps(data, indent=2) if args.json else core.format_status(data))
            return 0
        if args.action == "claim":
            record = core.claim(cfg, args.id, by=args.by)
            print(f"claimed {args.id} for {record['by']}")
            return 0
        if args.action == "release":
            core.release(cfg, args.id)
            print(f"released {args.id}")
            return 0
        if args.action == "prompt":
            if args.claim:
                core.claim(cfg, args.id, by=args.by)
            print(core.render_prompt(cfg, args.id))
            return 0
        if args.action == "answer":
            core.answer(cfg, args.id, args.source.read_text(encoding="utf-8"))
            print(f"answered {args.id}")
            return 0
        if args.action == "instructions":
            if args.stdout:
                sys.stdout.write(core.render_instructions(cfg))
            else:
                print(core.install_instructions(cfg, args.out))
            return 0
    except core.DriverError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 2


__all__ = ["build_parser", "main"]
