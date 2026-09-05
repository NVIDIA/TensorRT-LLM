"""Render what :mod:`agent_flow.ops.collector` gathered, as plain text.

Two views:

* the default one, for a single project: scoreboard, red gates with the reason
  their owner recorded, the reservation tables, and the message queue;
* ``--projects``, a static cross-project index: one row per project with its
  parent, start commit, final commit if it has been archived, flow state, last
  ledger row, scoreboard, allocations held and pending messages.

Plain text, not curses, and that is the point: the renderer takes a dict and
returns a string, so it is testable without a terminal, pipes into a file or a
message, and works over a connection where a full-screen program would not. A
curses view can be layered on the same dict later without moving any logic.
"""

from __future__ import annotations

import argparse
import sys
import time

from agent_flow.ops.collector import collect
from agent_flow.ops.config import add_config_argument, config_from_args

BAR_WIDTH = 24


def _age(epoch: float, now: float | None = None) -> str:
    if not epoch:
        return "-"
    delta = (time.time() if now is None else now) - epoch
    if delta < 0:
        return "just now"
    for unit, size in (("d", 86400), ("h", 3600), ("m", 60)):
        if delta >= size:
            return f"{int(delta // size)}{unit} ago"
    return "just now"


def _stamp(epoch: float) -> str:
    return time.strftime("%Y-%m-%d %H:%M", time.localtime(epoch)) if epoch else "-"


def progress_bar(green: int, total: int, width: int = BAR_WIDTH) -> str:
    if total <= 0:
        return "[" + " " * width + "] 0/0"
    filled = round(width * green / total)
    return "[" + "#" * filled + "." * (width - filled) + f"] {green}/{total}"


def render_status(status: dict) -> str:
    """The single-project view."""
    board = status["scoreboard"]
    flow = status["flow"]
    out = [
        f"{status['project']}  ({flow['state']}"
        + (f", pid {flow['pid']}" if flow["pid"] else "")
        + ")",
        f"  {progress_bar(board['green'], board['total'])} gates green"
        f"   last ledger row {_age(status['last_ledger_epoch'], status['generated_at'])}",
        "",
        status.get("narration") or "",
        "",
        "GATES",
    ]
    if not status["gates"]:
        out.append("  (no ledger rows)")
    for gate in status["gates"]:
        mark = "pass" if gate["state"] == "pass" else "FAIL"
        line = f"  {gate['id']:8s} {mark:5s} {gate['runs']:>3} runs  {gate['last_time'] or '-'}"
        if gate["last_commit"]:
            line += f"  {gate['last_commit'][:12]}"
        out.append(line)
        if gate["state"] != "pass" and gate["reason"]:
            out.append(f"           why: {gate['reason']}")
    out += ["", "ALLOCATIONS"]
    if not status["allocations"]:
        out.append("  (none declared)")
    for row in status["allocations"]:
        who = row["holder"] or "free"
        out.append(
            f"  {row['key']:14s} {row['job_id'] or '-':>10s}  {who:16s} {row['purpose'] or ''}"
        )
    out += ["", "WORKTREES"]
    if not status["worktrees"]:
        out.append("  (none declared)")
    for row in status["worktrees"]:
        out.append(f"  {row['key']:14s} {row['holder'] or 'free':16s} {row['purpose'] or ''}")
    boxes = status["mailboxes"]
    counts = boxes["counts"]
    out += [
        "",
        f"MESSAGES  {counts['pending']} pending, {counts['blocking']} blocking, "
        f"{counts['overdue']} overdue",
    ]
    overdue_ids = set(boxes["overdue_ids"])
    for rec in boxes["pending"]:
        flags = "".join(
            [
                "B" if rec.get("blocking") else " ",
                "!" if rec["id"] in overdue_ids else " ",
            ]
        )
        owed = ",".join(rec.get("owed") or rec.get("to") or [])
        first = str(rec.get("message", "")).splitlines()[0] if rec.get("message") else ""
        out.append(f"  {flags} {rec['id']:5s} -> {owed:20s} {first[:60]}")
    return "\n".join(out).rstrip() + "\n"


PROJECT_HEADER = (
    f"{'project':18s} {'state':9s} {'parent':14s} {'start':10s} {'final':10s} "
    f"{'gates':7s} {'last row':17s} {'alloc':12s} msgs"
)


def render_projects(rows: list[dict], live: str | None = None) -> str:
    """The cross-project index; ``live`` names the project to float to the top."""
    if not rows:
        return "no projects\n"
    ordered = sorted(
        rows,
        key=lambda r: (
            r["name"] != live,
            r["state"] != "running",
            -r["last_ledger_epoch"],
            r["name"],
        ),
    )
    out = [PROJECT_HEADER, "-" * len(PROJECT_HEADER)]
    for r in ordered:
        state = r["state"] + (f" {r['pid']}" if r["pid"] else "")
        out.append(
            f"{r['name']:18s} {state:9s} {(r['parent'] or '-'):14s} "
            f"{(r['start_commit'] or '-')[:10]:10s} {(r['final_commit'] or '-')[:10]:10s} "
            f"{r['passing']}/{r['gates']:<5} {_stamp(r['last_ledger_epoch']):17s} "
            f"{(','.join(r['allocations']) or '-'):12s} {r['pending_notices']}"
        )
        if not r["has_config"]:
            out.append(f"{'':18s} (no project config)")
    return "\n".join(out) + "\n"


def cmd_projects(a) -> int:
    from agent_flow.ops.config import OpsConfigError, load_config
    from agent_flow.ops.project import index, projects_root

    try:
        cfg = load_config(getattr(a, "config", None), getattr(a, "shared_config", None))
    except OpsConfigError:
        cfg = None
    try:
        root = projects_root(cfg, a.projects_root)
    except OpsConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    rows = index(root, cfg.archive_root if cfg else None)
    print(render_projects(rows, live=cfg.project_name if cfg else None), end="")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="agent_flow.ops.dashboard",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add_config_argument(p)
    p.add_argument("--projects", action="store_true", help="cross-project index instead")
    p.add_argument("--projects-root", default=None)
    p.add_argument("--narrate", action="store_true", help="ask the configured narrator")
    p.add_argument("--json", action="store_true", help="print the status dict instead")
    a = p.parse_args(argv)
    if a.projects:
        return cmd_projects(a)
    cfg = config_from_args(a)
    status = collect(cfg, narration=a.narrate)
    if a.json:
        import json

        print(json.dumps(status, indent=1, default=str))
        return 0
    print(render_status(status), end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
