"""Freeze one project directory into a git-tracked archive of runs.

    python -m agent_flow.ops.archive freeze <name> [--source DIR] [--dest DIR]
                                       [--date YYYY-MM-DD] [--summary TEXT]
                                       [--dry-run]

The archive folder ``<dest>/run-<date>-<name>/`` holds the readable half of a
finished run: the workspace, logs, notes, the text files at the project root,
the text documents under ``handoff/``, and every evidence file small enough to
carry. Alongside them go two manifests and a README index row, so a run can be
read months later without the machine it ran on.

Why not copy everything: the directories a long run leaves behind are mostly
unreadable bulk — a source tree, checkpoints, profiler output. Evidence files
over ``evidence_max_bytes`` are therefore *listed* in ``EVIDENCE-MANIFEST.json``
with their size, mtime and original path instead of copied, and anything that
does land over ``git_max_bytes`` stays on disk but gets a per-file
``.gitignore`` entry plus a row in ``MANIFEST.json["oversize"]`` — a large file
that silently entered git history would be far more expensive to remove than to
never commit.

Two properties the tool is built around:

* **Idempotent.** Re-running for the same name refreshes the folder in place:
  unchanged files (same size and mtime) are skipped, the README row is amended
  rather than appended, and the ignore block for the folder is rewritten.
* **Symlinks are never followed.** A link inside the project can point at a
  filesystem the archive must not absorb, so links are counted and skipped.

Everything the tool copies is configurable under ``[archive]`` in the ops
config; the defaults are the layout ``agent_flow.ops.project new`` scaffolds.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from agent_flow.ops.config import OpsConfig, OpsConfigError, add_config_argument, load_config
from agent_flow.ops.ledger import gate_reasons, ledger_rows, scoreboard

#: Copied wholesale (recursively).
DEFAULT_TREES = ("workspace", "logs", "notes", "restart")
#: Copied when present at the project root.
DEFAULT_FILES = (
    "task.yaml",
    "AGENT-NOTICES.jsonl",
    "HANDOFF.md",
    "MEMENTO.md",
)
#: Globs matched at the project root.
DEFAULT_GLOBS = ("run-*.log", "*.jsonl")
#: Directory holding handoff material, of which only text documents are copied.
DEFAULT_HANDOFF_DIR = "handoff"
DEFAULT_HANDOFF_SUFFIXES = (".md", ".txt")
DEFAULT_EVIDENCE_DIR = "evidence"
DEFAULT_EVIDENCE_MAX = 2 * 1024 * 1024
DEFAULT_GIT_MAX = 5 * 1024 * 1024

SKIP_DIRS = ("__pycache__", ".git")

INDEX_START = "<!-- index -->"
INDEX_END = "<!-- /index -->"
INDEX_HEADER = (
    "| run | dates | final commit | gates green | summary |\n| --- | --- | --- | --- | --- |"
)
README_INTRO = """# Archived runs

One folder per run: `run-<date>-<name>/` holds the workspace, logs, notes, the
small evidence files and a `MANIFEST.json` describing everything else,
including the large evidence files left at their original path.

Written by `python -m agent_flow.ops.archive`.

## Runs

"""


class ArchiveSettings:
    """What to copy and where the size thresholds sit."""

    def __init__(self, cfg: OpsConfig | None = None):
        get = (lambda k, d: cfg.get("archive", k, default=d)) if cfg else (lambda k, d: d)
        self.trees = tuple(get("trees", list(DEFAULT_TREES)))
        self.files = tuple(get("files", list(DEFAULT_FILES)))
        self.globs = tuple(get("globs", list(DEFAULT_GLOBS)))
        self.handoff_dir = str(get("handoff_dir", DEFAULT_HANDOFF_DIR))
        self.handoff_suffixes = tuple(get("handoff_suffixes", list(DEFAULT_HANDOFF_SUFFIXES)))
        self.evidence_dir = str(get("evidence_dir", DEFAULT_EVIDENCE_DIR))
        self.evidence_max = int(get("evidence_max_bytes", DEFAULT_EVIDENCE_MAX))
        self.git_max = int(get("git_max_bytes", DEFAULT_GIT_MAX))
        self.workspace = str(get("workspace", "workspace"))
        self.commit = bool(get("commit", True))


def _sh(cmd: list[str], cwd: Path | None = None) -> str:
    """Run a command, returning stdout stripped, or "" on any failure."""
    try:
        r = subprocess.run(
            cmd,
            cwd=str(cwd) if cwd else None,
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return r.stdout.strip() if r.returncode == 0 else ""


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _iso(ts: float) -> str:
    return datetime.fromtimestamp(ts).isoformat(timespec="seconds")


class Copier:
    """Symlink-skipping, mtime/size-skipping file copier with a running tally."""

    def __init__(self, dry: bool = False):
        self.dry = dry
        self.copied = 0
        self.skipped = 0
        self.symlinks = 0
        self.bytes = 0

    def file(self, src: Path, dst: Path) -> bool:
        if src.is_symlink():  # never follow a link out of the project
            self.symlinks += 1
            return False
        try:
            st = src.stat()
        except OSError:
            return False
        try:
            d = dst.stat()
            if d.st_size == st.st_size and int(d.st_mtime) == int(st.st_mtime):
                self.skipped += 1
                return True
        except OSError:
            pass
        self.copied += 1
        self.bytes += st.st_size
        if not self.dry:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
        return True

    def tree(self, src: Path, dst: Path) -> None:
        if not src.is_dir() or src.is_symlink():
            return
        for root, dirs, files in os.walk(src, followlinks=False):
            rootp = Path(root)
            dirs[:] = [d for d in dirs if d not in SKIP_DIRS and not (rootp / d).is_symlink()]
            for f in sorted(files):
                s = rootp / f
                self.file(s, dst / s.relative_to(src))


def evidence_manifest(
    evidence: Path, archive: Path, cp: Copier, max_bytes: int, rel_dir: str
) -> list[dict]:
    """Every file under ``evidence``: copied when small, listed either way."""
    rows: list[dict] = []
    if not evidence.is_dir():
        return rows
    for root, dirs, files in os.walk(evidence, followlinks=False):
        rootp = Path(root)
        dirs[:] = [d for d in dirs if not (rootp / d).is_symlink()]
        for f in sorted(files):
            s = rootp / f
            if s.is_symlink():
                cp.symlinks += 1
                continue
            try:
                st = s.stat()
            except OSError:
                continue
            rel = str(s.relative_to(evidence))
            small = st.st_size <= max_bytes
            row = {
                "path": rel,
                "size": st.st_size,
                "mtime": st.st_mtime,
                "mtime_str": _iso(st.st_mtime),
                "source": str(s),
                "copied": small,
            }
            if small:
                row["sha256"] = sha256(s)
                cp.file(s, archive / rel_dir / rel)
            rows.append(row)
    return rows


def board_of(workspace: Path) -> dict:
    """Scoreboard read straight out of the archived ledger.

    Deliberately file-only: freezing a run must not need the cluster, the
    model, or any service that will outlive neither.
    """
    rows = ledger_rows(workspace)
    board = scoreboard(rows)
    reasons = gate_reasons(workspace)
    gates = []
    for gate in sorted(board):
        last = rows[gate][-1]
        gates.append(
            {
                "id": gate,
                "state": board[gate],
                "runs": len(rows[gate]),
                "last_time": last.get("time"),
                "last_run": last.get("run"),
                "last_commit": last.get("commit"),
                "reason": (reasons.get(gate) or {}).get("text"),
            }
        )
    green = sum(1 for v in board.values() if v == "pass")
    return {"gates": gates, "green": green, "total": len(board)}


def oversize_files(archive: Path, limit: int) -> list[dict]:
    out = []
    for root, dirs, files in os.walk(archive):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for f in sorted(files):
            p = Path(root) / f
            try:
                st = p.stat()
            except OSError:
                continue
            if st.st_size > limit:
                out.append(
                    {
                        "path": str(p.relative_to(archive)),
                        "size": st.st_size,
                        "mtime": _iso(st.st_mtime),
                    }
                )
    return sorted(out, key=lambda o: o["path"])


def write_gitignore(dest: Path, folder: str, oversize: list[dict], limit: int) -> None:
    """One ignore entry per oversize file, plus the blanket caches.

    The per-folder block is rewritten every time, so a re-freeze that shrinks
    a file also un-ignores it; entries belonging to other folders are kept.
    """
    gi = dest / ".gitignore"
    head = [
        "# Written by agent_flow.ops.archive. Files over the size limit are",
        "# archived on disk but never committed; each is listed in its run's",
        '# MANIFEST.json under "oversize" with size, mtime and original path.',
        f"# limit: {limit} bytes",
        "__pycache__/",
        "*.pyc",
        "",
    ]
    existing = gi.read_text().splitlines() if gi.exists() else []
    keep = [
        ln
        for ln in existing
        if ln.strip()
        and not ln.startswith("#")
        and ln not in head
        and not ln.startswith(f"{folder}/")
    ]
    block = [f"# --- {folder} (oversize, {len(oversize)} files) ---"]
    block += [f"{folder}/{o['path']}" for o in oversize]
    gi.write_text("\n".join(head + sorted(set(keep)) + [""] + block) + "\n")


def update_readme(dest: Path, folder: str, row: dict) -> None:
    """Amend this run's row in the README index, between the markers."""
    readme = dest / "README.md"
    text = readme.read_text() if readme.exists() else ""
    if INDEX_START not in text:
        text = README_INTRO + INDEX_START + "\n" + INDEX_HEADER + "\n" + INDEX_END + "\n"
    pre, rest = text.split(INDEX_START, 1)
    table, post = rest.split(INDEX_END, 1)
    header_lines = INDEX_HEADER.splitlines()
    rows = [
        ln.rstrip()
        for ln in table.strip().splitlines()
        if ln.strip() and ln.rstrip() not in header_lines
    ]
    rows = [ln for ln in rows if f"]({folder}/)" not in ln]  # amend, never duplicate
    rows.append(
        f"| [{row['name']}]({folder}/) | {row['dates']} | `{row['commit']}` | "
        f"{row['green']} | {row['summary']} |"
    )
    body = INDEX_HEADER + "\n" + "\n".join(sorted(rows)) + "\n"
    readme.write_text(pre + INDEX_START + "\n" + body + INDEX_END + post)


def ensure_repo(dest: Path, dry: bool = False) -> None:
    if (dest / ".git").is_dir():
        return
    if dry:
        return
    dest.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", "-q", "-b", "main", str(dest)], check=True)


def freeze(
    source: Path,
    dest: Path,
    name: str,
    date: str | None = None,
    summary: str = "",
    settings: ArchiveSettings | None = None,
    repo: Path | None = None,
    dry_run: bool = False,
) -> dict:
    """Copy ``source`` into ``dest`` and return the manifest.

    On ``dry_run`` nothing is written; the returned dict carries the tally
    only.
    """
    st = settings or ArchiveSettings()
    date = date or datetime.now().strftime("%Y-%m-%d")
    folder = f"run-{date}-{name}"
    archive = dest / folder
    cp = Copier(dry_run)
    t0 = time.time()

    ensure_repo(dest, dry_run)
    if not dry_run:
        archive.mkdir(parents=True, exist_ok=True)

    for tree in st.trees:
        cp.tree(source / tree, archive / tree)
    for fname in st.files:
        if (source / fname).is_file():
            cp.file(source / fname, archive / fname)
    for pat in st.globs:
        for f in sorted(source.glob(pat)):
            if f.is_file():
                cp.file(f, archive / f.name)
    handoff = source / st.handoff_dir
    if handoff.is_dir() and not handoff.is_symlink():
        for f in sorted(handoff.iterdir()):
            if f.is_file() and f.suffix.lower() in st.handoff_suffixes:
                cp.file(f, archive / st.handoff_dir / f.name)

    ev_rows = evidence_manifest(
        source / st.evidence_dir, archive, cp, st.evidence_max, st.evidence_dir
    )
    tally = {
        "files_copied": cp.copied,
        "files_unchanged": cp.skipped,
        "symlinks_skipped": cp.symlinks,
        "bytes_copied": cp.bytes,
        "evidence_files": len(ev_rows),
        "evidence_bytes": sum(r["size"] for r in ev_rows),
        "evidence_copied": sum(1 for r in ev_rows if r["copied"]),
    }
    if dry_run:
        return {"folder": folder, "archive": str(archive), "counts": tally, "dry_run": True}

    (archive / "EVIDENCE-MANIFEST.json").write_text(
        json.dumps(
            {
                "root": str(source / st.evidence_dir),
                "copy_threshold_bytes": st.evidence_max,
                "total_files": len(ev_rows),
                "total_bytes": tally["evidence_bytes"],
                "copied_files": tally["evidence_copied"],
                "files": ev_rows,
            },
            indent=1,
        )
    )

    board = board_of(archive / st.workspace)
    oversize = oversize_files(archive, st.git_max)
    logs = [
        {
            "name": f.name,
            "size": f.stat().st_size,
            "first_mtime": _iso(f.stat().st_ctime),
            "last_mtime": _iso(f.stat().st_mtime),
        }
        for f in sorted(source.glob("run-*.log"))
        if f.is_file()
    ]
    manifest = {
        "run": name,
        "folder": folder,
        "archived_at": _iso(time.time()),
        "archive_seconds": round(time.time() - t0, 1),
        "source": str(source),
        "repo": str(repo) if repo else "",
        "repo_head": _sh(["git", "-C", str(repo), "rev-parse", "HEAD"]) if repo else "",
        "repo_branch": (
            _sh(["git", "-C", str(repo), "rev-parse", "--abbrev-ref", "HEAD"]) if repo else ""
        ),
        "run_logs": logs,
        "scoreboard": board,
        "counts": tally,
        "oversize": oversize,
        "git_max_bytes": st.git_max,
    }
    (archive / "MANIFEST.json").write_text(json.dumps(manifest, indent=1))

    write_gitignore(dest, folder, oversize, st.git_max)
    dates = (
        " to ".join(dict.fromkeys([logs[0]["first_mtime"][:10], logs[-1]["last_mtime"][:10]]))
        if logs
        else date
    )
    update_readme(
        dest,
        folder,
        {
            "name": folder,
            "dates": dates,
            "commit": (manifest["repo_head"] or "?")[:12],
            "green": f"{board['green']}/{board['total']}",
            "summary": summary or f"archive of {source.name}",
        },
    )
    if st.commit:
        _sh(["git", "add", "-A", "."], cwd=dest)
        if _sh(["git", "status", "--porcelain"], cwd=dest):
            subprocess.run(
                [
                    "git",
                    "-C",
                    str(dest),
                    "commit",
                    "-q",
                    "-m",
                    f"Archive {folder}: {board['green']}/{board['total']} gates green",
                ],
                check=False,
                capture_output=True,
            )
    return manifest


def read_manifest(archive: Path) -> dict:
    """Manifest of one archived run, or ``{}`` when there is none."""
    path = Path(archive)
    if path.is_dir():
        path = path / "MANIFEST.json"
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def find_archives(dest: Path) -> list[Path]:
    """Archived run folders under ``dest``, oldest name first."""
    if not Path(dest).is_dir():
        return []
    return sorted(d for d in Path(dest).iterdir() if (d / "MANIFEST.json").is_file())


def _optional_config(a) -> OpsConfig | None:
    try:
        return load_config(getattr(a, "config", None), getattr(a, "shared_config", None))
    except OpsConfigError:
        return None


def cmd_freeze(a) -> int:
    cfg = _optional_config(a)
    source = Path(a.source).expanduser() if a.source else (cfg.project_root if cfg else None)
    dest = Path(a.dest).expanduser() if a.dest else (cfg.archive_root if cfg else None)
    if source is None or dest is None:
        print(
            "error: need a source project dir and an archive dest: pass --source/--dest "
            "or set [project].root and [project].archive_root in the config.",
            file=sys.stderr,
        )
        return 2
    source = source.resolve()
    if not source.is_dir():
        print(f"error: no such project directory: {source}", file=sys.stderr)
        return 2
    declared_repo = cfg.get("container", "repo") if cfg else None
    repo = (
        Path(a.repo).expanduser()
        if a.repo
        else (Path(declared_repo).expanduser() if declared_repo else None)
    )
    out = freeze(
        source,
        dest,
        a.name,
        date=a.date,
        summary=a.summary,
        settings=ArchiveSettings(cfg),
        repo=repo,
        dry_run=a.dry_run,
    )
    c = out["counts"]
    prefix = "[dry-run] " if a.dry_run else ""
    print(f"{prefix}{dest / out['folder']}")
    print(
        f"  {c['files_copied']} files copied ({c['bytes_copied'] / 1e6:.1f} MB), "
        f"{c['files_unchanged']} unchanged, {c['symlinks_skipped']} symlinks skipped"
    )
    print(f"  evidence: {c['evidence_files']} listed, {c['evidence_copied']} copied")
    if not a.dry_run:
        board = out["scoreboard"]
        print(f"  oversize (not committed): {len(out['oversize'])}")
        print(f"  gates green: {board['green']}/{board['total']}")
    return 0


def cmd_list(a) -> int:
    cfg = _optional_config(a)
    dest = Path(a.dest).expanduser() if a.dest else (cfg.archive_root if cfg else None)
    if dest is None:
        print("error: no archive root: pass --dest or set [project].archive_root", file=sys.stderr)
        return 2
    found = find_archives(dest)
    if not found:
        print(f"no archived runs under {dest}")
        return 0
    for d in found:
        m = read_manifest(d)
        board = m.get("scoreboard") or {}
        print(
            f"{d.name:36s} {(m.get('repo_head') or '?')[:12]:14s} "
            f"{board.get('green', '?')}/{board.get('total', '?')} green   "
            f"{m.get('archived_at', '')}"
        )
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="agent_flow.ops.archive",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add_config_argument(p)
    p.add_argument("--dest", default=None, help="archive repo (default: [project].archive_root)")
    sub = p.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("freeze")
    f.add_argument("name")
    f.add_argument("--source", default=None, help="project dir (default: [project].root)")
    f.add_argument("--date", default=None, help="date in the folder name (default: today)")
    f.add_argument("--summary", default="", help="one-line summary for the README row")
    f.add_argument("--repo", default=None, help="checkout whose HEAD is recorded")
    f.add_argument("--dry-run", action="store_true")
    sub.add_parser("list")
    return p


def main(argv: list[str] | None = None) -> int:
    a = build_parser().parse_args(argv)
    try:
        return {"freeze": cmd_freeze, "list": cmd_list}[a.cmd](a)
    except OpsConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
