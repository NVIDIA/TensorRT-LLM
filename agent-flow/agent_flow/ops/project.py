"""List and scaffold projects.

A "project" is the unit that owns a workspace, its acceptance criteria, its
notice queue and its dashboard: one directory, one project config, one ledger.
The machine-wide things (allocations, worktree slots, the container) are shared
by every project and live in the shared config, so a second project never gets
its own copy of the allocation table.

    python -m agent_flow.ops.project list
    python -m agent_flow.ops.project new my-project [--projects-root DIR]

``list`` reports, per project: whether a workflow process is alive for it, the
timestamp of the newest ledger row, and a one-line scoreboard summary. It reads
files only — no scheduler, no network — so it is safe to run anywhere.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from agent_flow.ops.config import (
    CWD_NAME,
    OpsConfig,
    OpsConfigError,
    add_config_argument,
    load_config,
)
from agent_flow.ops.ledger import ledger_rows, scoreboard

TEMPLATE = """# Project ops overlay for "{name}". The shared machine config
# (allocations, worktree slots, container) is layered underneath this file.

[project]
name = "{name}"
root = "{root}"
workspace = "workspace"
log_dir = "logs"

[roles]
names = ["coder", "reviewer"]

[roles.checkouts]
# role -> the checkout that role works from (used to infer who is acking)
# coder = "/path/to/main-checkout"
# reviewer = "/path/to/worktrees/wt-2"
"""

SUBDIRS = ("workspace", "logs", "evidence")


def projects_root(cfg: OpsConfig | None, override: str | None) -> Path:
    if override:
        return Path(override).expanduser()
    if cfg is not None and cfg.projects_root:
        return cfg.projects_root
    if os.environ.get("AGENT_FLOW_PROJECTS_ROOT"):
        return Path(os.environ["AGENT_FLOW_PROJECTS_ROOT"]).expanduser()
    raise OpsConfigError(
        "no projects root: set [projects].root in the shared config, pass "
        "--projects-root, or set $AGENT_FLOW_PROJECTS_ROOT."
    )


def flow_pid(root: Path) -> int | None:
    """Pid of a workflow process running against ``root``, if one is alive.

    Matched on the project root appearing in the command line, so it does not
    depend on the workflow's entry-point name.
    """
    try:
        out = subprocess.run(
            ["ps", "-eo", "pid=,args="], capture_output=True, text=True, timeout=20
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    me = os.getpid()
    for line in out.splitlines():
        pid, _, args = line.strip().partition(" ")
        if not pid.isdigit() or int(pid) == me:
            continue
        if str(root) in args and "agent_flow.ops.project" not in args:
            return int(pid)
    return None


def describe(root: Path) -> dict:
    """Everything ``list`` shows for one project, read from files only."""
    workspace = root / "workspace"
    rows = ledger_rows(workspace)
    board = scoreboard(rows)
    newest = max(
        (r["epoch"] for gate_rows in rows.values() for r in gate_rows),
        default=0.0,
    )
    passing = sum(1 for v in board.values() if v == "pass")
    return {
        "name": root.name,
        "root": root,
        "has_config": (root / CWD_NAME).is_file(),
        "pid": flow_pid(root),
        "last_ledger_epoch": newest,
        "gates": len(board),
        "passing": passing,
        "summary": f"{passing}/{len(board)} gates passing" if board else "no ledger rows",
    }


def find_projects(root: Path) -> list[Path]:
    if not root.is_dir():
        return []
    return sorted(
        d
        for d in root.iterdir()
        if d.is_dir() and ((d / CWD_NAME).is_file() or (d / "workspace").is_dir())
    )


def cmd_list(a) -> int:
    cfg = _optional_config(a)
    root = projects_root(cfg, a.projects_root)
    found = find_projects(root)
    if not found:
        print(f"no projects under {root}")
        return 0
    import time

    for d in found:
        info = describe(d)
        alive = f"pid {info['pid']}" if info["pid"] else "idle"
        last = (
            time.strftime("%Y-%m-%d %H:%M", time.localtime(info["last_ledger_epoch"]))
            if info["last_ledger_epoch"]
            else "-"
        )
        flag = "" if info["has_config"] else "  (no project config)"
        print(f"{info['name']:24s} {alive:12s} last row {last:16s} {info['summary']}{flag}")
    return 0


def cmd_new(a) -> int:
    cfg = _optional_config(a)
    root = projects_root(cfg, a.projects_root) / a.name
    if root.exists() and any(root.iterdir()):
        print(f"error: {root} already exists and is not empty", file=sys.stderr)
        return 3
    for sub in SUBDIRS:
        (root / sub).mkdir(parents=True, exist_ok=True)
    (root / CWD_NAME).write_text(TEMPLATE.format(name=a.name, root=root))
    print(f"created project {a.name} at {root}")
    print(f"  edit {root / CWD_NAME}, then run tools with --config {root}")
    return 0


def _optional_config(a) -> OpsConfig | None:
    """Load whatever config exists.

    The shared config is enough for these commands, and even that is optional
    when --projects-root is given.
    """
    try:
        return load_config(getattr(a, "config", None), getattr(a, "shared_config", None))
    except OpsConfigError:
        return None


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="agent_flow.ops.project",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add_config_argument(p)
    p.add_argument("--projects-root", default=None, help="directory holding the project dirs")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    s = sub.add_parser("new")
    s.add_argument("name")
    return p


def main(argv: list[str] | None = None) -> int:
    a = build_parser().parse_args(argv)
    try:
        return {"list": cmd_list, "new": cmd_new}[a.cmd](a)
    except OpsConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
