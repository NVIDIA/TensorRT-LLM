# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``trtllm-modeling-v2 driver ...``: the external transport's driver loop, as commands.

::

    trtllm-modeling-v2 driver --workspace WS [--max-inflight 16] wait [--timeout 540] [--sessions-only]
    trtllm-modeling-v2 driver --workspace WS pending
    trtllm-modeling-v2 driver --workspace WS status [--json]
    trtllm-modeling-v2 driver --workspace WS claim <id> [--by NAME]
    trtllm-modeling-v2 driver --workspace WS release <id>
    trtllm-modeling-v2 driver --workspace WS prompt <id> [--claim] [--by NAME]
    trtllm-modeling-v2 driver --workspace WS answer <id> --from FILE
    trtllm-modeling-v2 driver --workspace WS instructions [--out PATH | --stdout]
    trtllm-modeling-v2 driver --workspace WS reload [--timeout 1800] [--force] [--cancel]

The loop a driver session runs::

    wait  -> NEW <id>               claim it, hand `prompt <id>` to a subagent
          -> ORCHESTRATOR_EXITED    read its log; relaunch it (it adopts turns not acted on)
          -> RELOADED               a reload re-executed the orchestrator; its loop runs again
          -> RELOAD_OVERDUE         the drain is past its deadline; it names what it waits for
          -> RELOAD_FAILED          the re-exec failed; the run goes on, on the old code
          -> SESSION_REQUEST ...    `session allocate`, submit the call it prints, run the
                                    `session ack` it prints with the new job id
          -> SESSION_STALE|ENDING|QUARANTINED ...   look at that session
          -> TIMEOUT                call wait again

A run whose turns run in process (``--agent-transport inprocess``) has no turn
to hand out; its operator runs ``wait --sessions-only``, which never returns
``NEW`` and returns every other event. A line ``note: ... archived request
...`` after the event is a successor request no live watcher stands behind,
set aside; nothing is asked of the driver for it.

``reload`` loads flow code changed since the orchestrator started, without
stopping it by hand: it starts no task, integration, check or planner turn,
waits for those in flight (worker turns go on and are adopted), checkpoints and
re-executes itself with the same pid, log and arguments. Past ``--timeout`` it
says what it still waits for and keeps waiting; with ``--force`` it goes on
without them. ``--cancel`` withdraws a request whose drain has not finished.

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
from . import reload as reload_mod


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="trtllm-modeling-v2 driver", description=__doc__.splitlines()[0]
    )
    parser.add_argument(
        "--workspace",
        default=os.environ.get("MV2FLOW_WORKSPACE"),
        help="the run's workspace (default $MV2FLOW_WORKSPACE)",
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
    wait.add_argument(
        "--sessions-only",
        action="store_true",
        help="turns run in process: never NEW; the orchestrator, reloads and the sessions",
    )
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
    again = sub.add_parser(
        "reload", help="drain the orchestrator, then re-execute it on the code now on disk"
    )
    again.add_argument(
        "--timeout",
        type=float,
        default=reload_mod.DEFAULT_DRAIN_TIMEOUT_S,
        metavar="SECONDS",
        help="how long the drain waits before it says what it still waits for "
        f"(default {reload_mod.DEFAULT_DRAIN_TIMEOUT_S:g})",
    )
    again.add_argument(
        "--force",
        action="store_true",
        help="past the timeout, re-execute without what is still in flight",
    )
    again.add_argument("--cancel", action="store_true", help="withdraw the request")
    again.add_argument("--by")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.workspace:
        print("error: pass --workspace or set MV2FLOW_WORKSPACE", file=sys.stderr)
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
                cfg,
                timeout_s=args.timeout,
                poll_s=args.poll,
                watch_sessions=not args.no_sessions,
                sessions_only=args.sessions_only,
            )
            print(event.render())
            print(core.format_status(core.status(cfg), turns=not args.sessions_only))
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
        if args.action == "reload":
            if args.cancel:
                withdrawn = core.cancel_reload(cfg)
                print(
                    f"withdrew reload {withdrawn.get('id')}"
                    if withdrawn
                    else "no reload request was waiting"
                )
                return 0
            asked = core.request_reload(cfg, force=args.force, timeout_s=args.timeout, by=args.by)
            print(
                f"reload {asked['id']} requested: the orchestrator drains, then re-executes "
                f"({'forced' if asked['force'] else 'waits'} past {asked['timeout_s']:g}s); "
                f"`wait` returns RELOADED when its loop runs again"
            )
            return 0
    except core.DriverError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 2


__all__ = ["build_parser", "main"]
