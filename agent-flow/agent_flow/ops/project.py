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
import json
import os
import subprocess
import sys
import tomllib
from pathlib import Path

from agent_flow.ops.archive import find_archives, read_manifest
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


QUEUE_NAME = "AGENT-NOTICES.jsonl"
TRAY_JSON_NAME = "TRAY-RESERVATIONS.json"


def _read_json(path: Path) -> dict:
    """A reservation table, read without its lock: viewers never block writers."""
    try:
        data = json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def project_toml(root: Path) -> dict:
    """The project's own overlay, read raw. ``{}`` when there is none."""
    try:
        return tomllib.loads((Path(root) / CWD_NAME).read_text())
    except (OSError, tomllib.TOMLDecodeError):
        return {}


def pending_notices(queue: Path) -> int:
    """Notices with no later ack, counted coarsely.

    Coarse on purpose: a notice is counted settled once ANY addressee has
    acked. The index needs one number per project without loading each
    project's role set; per-addressee detail is what
    :func:`agent_flow.ops.mailbox.status` is for.
    """
    try:
        lines = Path(queue).read_text(errors="ignore").splitlines()
    except OSError:
        return 0
    recs = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            recs.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    msgs = {r["id"]: r.get("ts", 0) for r in recs if r.get("type") == "notice" and r.get("id")}
    acked = {
        r["id"]
        for r in recs
        if r.get("type") == "ack" and r.get("id") in msgs and r.get("ts", 0) >= msgs[r["id"]]
    }
    return len(set(msgs) - acked)


def archive_manifests(archive_root: Path | None) -> dict[str, dict]:
    """``{project name: manifest}`` for every archived run under a root."""
    if not archive_root:
        return {}
    out = {}
    for folder in find_archives(Path(archive_root)):
        m = read_manifest(folder)
        if m.get("run"):
            out[str(m["run"])] = m
    return out


def holds(holder: str | None, name: str, roles: tuple[str, ...] = ()) -> bool:
    """Whether a reservation holder belongs to project ``name``.

    Matched on the project name itself or a ``<project>/<role>`` /
    ``<project>:<role>`` holder, plus any role name the caller has already
    established belongs to this project alone. Role names like "coder" are
    shared vocabulary, so attributing one to a project on the name alone would
    show a second project's allocation as this one's.
    """
    if not holder:
        return False
    holder = str(holder)
    head = holder.split("/")[0].split(":")[0]
    return holder == name or head == name or holder in roles


def held_allocations(alloc_slots: dict, name: str, roles: tuple[str, ...] = ()) -> list[str]:
    return [
        key for key, body in (alloc_slots or {}).items() if holds(body.get("holder"), name, roles)
    ]


def describe(
    root: Path,
    archives: dict[str, dict] | None = None,
    alloc_slots: dict | None = None,
    owned_roles: tuple[str, ...] = (),
) -> dict:
    """Everything the index shows for one project, read from files only.

    ``archives`` and ``alloc_slots`` are passed in rather than read here so an
    index over N projects reads the archive root and the shared allocation
    table once, not N times.
    """
    root = Path(root)
    workspace = root / "workspace"
    rows = ledger_rows(workspace)
    board = scoreboard(rows)
    newest = max(
        (r["epoch"] for gate_rows in rows.values() for r in gate_rows),
        default=0.0,
    )
    passing = sum(1 for v in board.values() if v == "pass")
    toml = project_toml(root)
    section = toml.get("project") or toml.get("run") or {}
    manifest = (archives or {}).get(root.name, {})
    pid = flow_pid(root)
    if pid:
        state = "running"
    elif manifest:
        state = "archived"
    else:
        state = "idle"
    return {
        "name": root.name,
        "root": root,
        "has_config": (root / CWD_NAME).is_file(),
        "parent": section.get("parent") or None,
        "start_commit": section.get("start_commit") or None,
        "final_commit": manifest.get("repo_head") or None,
        "archived_as": manifest.get("folder") or None,
        "pid": pid,
        "state": state,
        "last_ledger_epoch": newest,
        "gates": len(board),
        "passing": passing,
        "allocations": held_allocations(alloc_slots or {}, root.name, owned_roles),
        "pending_notices": pending_notices(root / QUEUE_NAME),
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


def _roles_of(root: Path) -> tuple[str, ...]:
    section = project_toml(root).get("roles") or {}
    names = section.get("names") or list((section.get("checkouts") or {}))
    return tuple(str(n) for n in names)


def index(
    root: Path, archive_root: Path | None = None, alloc_slots: dict | None = None
) -> list[dict]:
    """Describe every project under ``root``, cross-referenced.

    One pass over the archive root and one over the allocation table serve all
    of them. A role name is attributed to a project only when no other indexed
    project declares it, so a shared word like "coder" never shows a second
    project's allocation as this one's.
    """
    found = find_projects(Path(root))
    archives = archive_manifests(archive_root)
    roles_by_project = {d.name: _roles_of(d) for d in found}
    counts: dict[str, int] = {}
    for names in roles_by_project.values():
        for n in names:
            counts[n] = counts.get(n, 0) + 1
    slots = dict(alloc_slots or {})
    if not slots:
        for d in found:
            slots.update(_read_json(d / TRAY_JSON_NAME).get("slots") or {})
    return [
        describe(
            d,
            archives=archives,
            alloc_slots=slots,
            owned_roles=tuple(n for n in roles_by_project[d.name] if counts[n] == 1),
        )
        for d in found
    ]


def cmd_list(a) -> int:
    from agent_flow.ops.dashboard import render_projects

    cfg = _optional_config(a)
    root = projects_root(cfg, a.projects_root)
    archive_root = Path(a.archive_root).expanduser() if a.archive_root else None
    if archive_root is None and cfg is not None:
        archive_root = cfg.archive_root
    rows = index(root, archive_root)
    if not rows:
        print(f"no projects under {root}")
        return 0
    if a.json:
        print(json.dumps(rows, indent=1, default=str))
        return 0
    print(render_projects(rows))
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
    p.add_argument("--archive-root", default=None, help="where finished projects are frozen")
    sub = p.add_subparsers(dest="cmd", required=True)
    ls = sub.add_parser("list")
    ls.add_argument("--json", action="store_true")
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
