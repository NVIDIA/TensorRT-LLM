# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``catalog/index.yaml``, generated from one fragment per entry.

The index is the one file every smith touched. Each appended its row and
committed the whole file, so the first smith to commit swept in its
siblings' uncommitted rows: ``HEAD`` then named entries whose files were not
in ``HEAD``, and a later smith's message named rows it did not write (F013).
Every smith of the DeepSeek-V4.1 run spent effort documenting that.

Now each entry's row is its own file::

    catalog/index.d/_header.yaml          # everything above `entries:`, and the category order
    catalog/index.d/<category>/<name>.yaml  # {path, impl, summary} of one entry

A smith writes and commits its own fragment with its entry, so no two
smiths ever commit the same file. The orchestrator regenerates
``index.yaml`` from the fragments **committed in HEAD** -- never from the
working tree, where siblings' fragments may be half-written -- and commits
that one file by itself (:func:`sync`). Generation also leaves out a
fragment whose entry wrapper is not in ``HEAD``, so the index can never name
an entry that a clone would not have.

:func:`migrate` splits an existing hand-maintained index into the header and
fragments (preserving the row order and the category dividers). :func:`sync`
does the same by itself, in a commit of its own, the first time HEAD holds a
fragment and no header, so a run that adds no catalog entry never touches
the catalog. :func:`problems` is the check: the index matches its fragments,
each fragment sits where its ``path`` says, and every fragment's entry exists.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import yaml

from .delivery import CATALOG_DIR

INDEX_FILENAME = "index.yaml"
FRAGMENTS_DIRNAME = "index.d"
HEADER_FILENAME = "_header.yaml"
#: In the workspace: serializes every sync of one run, across threads and processes.
LOCK_FILENAME = ".catalog-index.lock"
#: The comment the generated index carries just above its rows.
GENERATED_NOTICE = (
    "# GENERATED from index.d/ (python -m agent_flow.workflows.staircase_onboard.catalog_index).\n"
    "# Each entry's row is its own file, index.d/<category>/<name>.yaml: edit that, not this list.\n"
)
#: Fragment keys rendered first, in this order; any others follow, sorted.
ROW_KEYS = ("path", "impl", "summary")
#: Kept in a fragment to preserve a migrated index's row order; never rendered.
ORDER_KEY = "order"
_DIVIDER_WIDTH = 67
_GIT_TIMEOUT_S = 60

CommitFn = Callable[[str, Sequence[str]], "tuple[bool, str]"]


# ------------------------------------------------------------------ model


@dataclass(frozen=True)
class Header:
    """Everything of ``index.yaml`` that is not an entry row."""

    preamble: str = ""
    category_order: tuple[str, ...] = ()
    category_dividers: Mapping[str, str] = field(default_factory=dict)

    def to_yaml(self) -> str:
        data = {
            "preamble": self.preamble,
            "category_order": list(self.category_order),
            "category_dividers": dict(self.category_dividers),
        }
        return (
            "# The header of catalog/index.yaml: everything above `entries:`, and the category\n"
            "# order. Each entry's row is its own file in this directory.\n"
            + yaml.dump(data, Dumper=_Dumper, sort_keys=False, allow_unicode=True, width=1000)
        )

    @classmethod
    def from_yaml(cls, text: str) -> "Header":
        data = yaml.safe_load(text) or {}
        if not isinstance(data, Mapping):
            raise ValueError(f"{HEADER_FILENAME} must be a mapping")
        return cls(
            preamble=str(data.get("preamble") or ""),
            category_order=tuple(str(c) for c in data.get("category_order") or ()),
            category_dividers={
                str(k): str(v) for k, v in (data.get("category_dividers") or {}).items()
            },
        )


class _Dumper(yaml.SafeDumper):
    """Writes multi-line strings as literal blocks, so the preamble stays readable."""


def _str_representer(dumper: yaml.SafeDumper, value: str) -> yaml.Node:
    style = "|" if "\n" in value else None
    return dumper.represent_scalar("tag:yaml.org,2002:str", value, style=style)


_Dumper.add_representer(str, _str_representer)


@dataclass(frozen=True)
class Snapshot:
    """The catalog as one tree holds it: header, fragments, files, and the index text."""

    header: Header | None
    #: Fragment location (relative to the catalog) -> its row.
    fragments: Mapping[str, Mapping[str, Any]]
    #: Every file under the catalog, relative to it.
    files: frozenset[str]
    index_text: str | None


def fragment_rel(entry_path: str) -> str:
    """Where the fragment of the entry at ``entry_path`` (``norm/x.py``) lives."""
    stem = entry_path[:-3] if entry_path.endswith(".py") else entry_path
    return f"{FRAGMENTS_DIRNAME}/{stem}.yaml"


def _header_rel() -> str:
    return f"{FRAGMENTS_DIRNAME}/{HEADER_FILENAME}"


def _load_fragment(text: str) -> dict[str, Any] | None:
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError:
        return None
    return dict(data) if isinstance(data, Mapping) else None


# ---------------------------------------------------------------- snapshots


def snapshot_worktree(catalog_dir: str | Path) -> Snapshot:
    """The catalog as the working tree holds it."""
    root = Path(catalog_dir)
    files = frozenset(
        p.relative_to(root).as_posix()
        for p in root.rglob("*")
        if p.is_file() and "__pycache__" not in p.parts
    )
    header = None
    header_path = root / _header_rel()
    if header_path.is_file():
        header = Header.from_yaml(header_path.read_text(encoding="utf-8"))
    fragments: dict[str, Mapping[str, Any]] = {}
    for rel in sorted(files):
        if rel.startswith(f"{FRAGMENTS_DIRNAME}/") and rel.endswith(".yaml"):
            if rel == _header_rel():
                continue
            row = _load_fragment((root / rel).read_text(encoding="utf-8"))
            fragments[rel] = row if row is not None else {}
    index = root / INDEX_FILENAME
    return Snapshot(
        header=header,
        fragments=fragments,
        files=files,
        index_text=index.read_text(encoding="utf-8") if index.is_file() else None,
    )


def _git(repo: str | Path, *args: str, stdin: str | None = None) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "-C", str(repo), *args],
            capture_output=True,
            text=True,
            timeout=_GIT_TIMEOUT_S,
            check=False,
            input=stdin,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return completed.stdout if completed.returncode == 0 else None


def _read_blobs(repo: str | Path, blobs: Sequence[str]) -> dict[str, str]:
    """Contents of ``blobs`` by id, through one ``git cat-file --batch``."""
    if not blobs:
        return {}
    try:
        completed = subprocess.run(
            ["git", "-C", str(repo), "cat-file", "--batch"],
            input=("\n".join(blobs) + "\n").encode(),
            capture_output=True,
            timeout=_GIT_TIMEOUT_S,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return {}
    out = completed.stdout
    contents: dict[str, str] = {}
    pos = 0
    while pos < len(out):
        newline = out.index(b"\n", pos)
        header = out[pos:newline].decode().split()
        pos = newline + 1
        if len(header) < 3 or header[1] == "missing":
            continue
        size = int(header[2])
        contents[header[0]] = out[pos : pos + size].decode("utf-8", errors="replace")
        pos += size + 1
    return contents


def snapshot_at(
    repo: str | Path, rev: str = "HEAD", *, catalog_rel: str = CATALOG_DIR
) -> Snapshot | None:
    """The catalog as commit ``rev`` holds it, or None when git cannot answer."""
    listing = _git(repo, "ls-tree", "-r", "-z", rev, "--", catalog_rel)
    if listing is None:
        return None
    prefix = catalog_rel.rstrip("/") + "/"
    blobs: dict[str, str] = {}
    for record in listing.split("\0"):
        meta, _, path = record.partition("\t")
        parts = meta.split()
        if len(parts) == 3 and parts[1] == "blob" and path.startswith(prefix):
            blobs[path[len(prefix) :]] = parts[2]
    wanted = [
        rel
        for rel in blobs
        if rel == INDEX_FILENAME
        or (rel.startswith(f"{FRAGMENTS_DIRNAME}/") and rel.endswith(".yaml"))
    ]
    contents = _read_blobs(repo, [blobs[rel] for rel in wanted])
    text = {rel: contents.get(blobs[rel], "") for rel in wanted}
    header = Header.from_yaml(text[_header_rel()]) if _header_rel() in text else None
    fragments: dict[str, Mapping[str, Any]] = {}
    for rel in sorted(text):
        if rel in (INDEX_FILENAME, _header_rel()):
            continue
        row = _load_fragment(text[rel])
        fragments[rel] = row if row is not None else {}
    return Snapshot(
        header=header,
        fragments=fragments,
        files=frozenset(blobs),
        index_text=text.get(INDEX_FILENAME),
    )


# --------------------------------------------------------------- rendering


def _scalar(value: Any) -> str:
    """One YAML scalar on one line: plain when that reads back unchanged, else quoted."""
    if isinstance(value, str):
        plain = (
            value
            and value == value.strip()
            and "\n" not in value
            and ": " not in value
            and " #" not in value
            and value[0] not in "!&*-?{}[],#|>@`\"'%"
        )
        if plain:
            try:
                if yaml.safe_load(value) == value:
                    return value
            except yaml.YAMLError:
                pass
        return json.dumps(value, ensure_ascii=False)
    return json.dumps(value, ensure_ascii=False)


def _divider(category: str) -> str:
    text = f"  # ─── {category} "
    return text + "─" * max(3, _DIVIDER_WIDTH - len(text))


def _render_row(row: Mapping[str, Any]) -> list[str]:
    keys = [k for k in ROW_KEYS if k in row] + sorted(
        k for k in row if k not in ROW_KEYS and k != ORDER_KEY
    )
    lines = []
    for index, key in enumerate(keys):
        lead = "  - " if index == 0 else "    "
        # Summaries are always double-quoted, as the hand-written index had them.
        value = json.dumps(row[key], ensure_ascii=False) if key == "summary" else _scalar(row[key])
        lines.append(f"{lead}{key}: {value}")
    return lines


def renderable_rows(snapshot: Snapshot) -> list[Mapping[str, Any]]:
    """Fragments that sit where their ``path`` says and whose entry exists."""
    rows = []
    for rel, row in snapshot.fragments.items():
        path = str(row.get("path") or "")
        if path and fragment_rel(path) == rel and path in snapshot.files:
            rows.append(row)
    return rows


def render(snapshot: Snapshot) -> str:
    """The ``index.yaml`` text the snapshot's header and fragments define."""
    header = snapshot.header or Header()
    by_category: dict[str, list[Mapping[str, Any]]] = {}
    for row in renderable_rows(snapshot):
        by_category.setdefault(str(row["path"]).split("/", 1)[0], []).append(row)
    order = [c for c in header.category_order if c in by_category] + sorted(
        c for c in by_category if c not in header.category_order
    )
    out = [header.preamble.rstrip("\n") + "\n\n" if header.preamble else "", GENERATED_NOTICE]
    out.append("entries:\n")
    blocks = []
    for category in order:
        rows = sorted(
            by_category[category],
            key=lambda r: (
                0 if isinstance(r.get(ORDER_KEY), int) else 1,
                r.get(ORDER_KEY) if isinstance(r.get(ORDER_KEY), int) else 0,
                str(r["path"]),
            ),
        )
        lines = [header.category_dividers.get(category) or _divider(category)]
        for index, row in enumerate(rows):
            if index:
                lines.append("")
            lines.extend(_render_row(row))
        blocks.append("\n".join(lines))
    out.append("\n\n".join(blocks) + ("\n" if blocks else ""))
    return "".join(out)


# --------------------------------------------------------------- migration


_DIVIDER_LINE = re.compile(r"^\s*#\s*─+\s*(?P<title>.*?)\s*─+\s*$")


def split_index(text: str) -> tuple[Header, list[dict[str, Any]]]:
    """Split a hand-maintained ``index.yaml`` into its header and ordered rows."""
    lines = text.splitlines(keepends=True)
    start = next((i for i, line in enumerate(lines) if line.startswith("entries:")), None)
    if start is None:
        raise ValueError(f"{INDEX_FILENAME} has no top-level `entries:`")
    notice = set(GENERATED_NOTICE.splitlines(keepends=True))
    preamble = "".join(line for line in lines[:start] if line not in notice)
    data = yaml.safe_load(text) or {}
    rows = [dict(r) for r in data.get("entries") or () if isinstance(r, Mapping) and r.get("path")]
    order: list[str] = []
    dividers: dict[str, str] = {}
    pending_divider: str | None = None
    for line in lines[start + 1 :]:
        if _DIVIDER_LINE.match(line):
            pending_divider = line.rstrip("\n")
            continue
        match = re.match(r"^\s*-\s*path:\s*(\S+)", line)
        if match:
            category = match.group(1).strip("'\"").split("/", 1)[0]
            if category not in order:
                order.append(category)
                if pending_divider is not None:
                    dividers[category] = pending_divider
            pending_divider = None
    for row in rows:
        category = str(row["path"]).split("/", 1)[0]
        if category not in order:
            order.append(category)
    return Header(preamble=preamble, category_order=tuple(order), category_dividers=dividers), rows


def _dump_fragment(row: Mapping[str, Any]) -> str:
    return yaml.safe_dump(dict(row), sort_keys=False, allow_unicode=True, width=1000)


def migration_files(snapshot: Snapshot) -> dict[str, str]:
    """The files that turn ``snapshot`` into a fragment-generated catalog.

    The header, when there is none yet, and a fragment for every index row
    that has none: a row a smith appended by hand is adopted rather than
    dropped by the next regeneration. Row order is kept through ``order``.
    """
    if snapshot.index_text is None:
        return {}
    header, rows = split_index(snapshot.index_text)
    files: dict[str, str] = {}
    if snapshot.header is None:
        files[_header_rel()] = header.to_yaml()
    for position, row in enumerate(rows, start=1):
        rel = fragment_rel(str(row["path"]))
        if rel in snapshot.fragments or rel in files:
            continue
        fragment = {k: row[k] for k in row}
        if snapshot.header is None:
            fragment[ORDER_KEY] = position
        files[rel] = _dump_fragment(fragment)
    return files


def migrate(catalog_dir: str | Path) -> list[Path]:
    """Split the working tree's ``index.yaml`` into fragments, then regenerate it."""
    root = Path(catalog_dir)
    written = []
    for rel, text in migration_files(snapshot_worktree(root)).items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        written.append(path)
    index = root / INDEX_FILENAME
    index.write_text(render(snapshot_worktree(root)), encoding="utf-8")
    written.append(index)
    return written


# ------------------------------------------------------------------- check


def problems(snapshot: Snapshot) -> list[str]:
    """Everything that makes ``snapshot``'s index not a pure function of its fragments."""
    if snapshot.header is None:
        return [f"{FRAGMENTS_DIRNAME}/{HEADER_FILENAME} is missing: the index is not migrated"]
    found: list[str] = []
    seen: dict[str, str] = {}
    for rel, row in snapshot.fragments.items():
        path = str(row.get("path") or "")
        if not path:
            found.append(f"{rel}: no `path` (or not a YAML mapping)")
            continue
        if fragment_rel(path) != rel:
            found.append(f"{rel}: names {path}, whose fragment belongs at {fragment_rel(path)}")
        if path not in snapshot.files:
            found.append(f"{rel}: entry {path} does not exist")
        if path in seen:
            found.append(f"{rel}: {path} is also claimed by {seen[path]}")
        seen[path] = rel
    if snapshot.index_text != render(snapshot):
        found.append(
            f"{INDEX_FILENAME} is not what its fragments generate; regenerate it "
            f"(python -m agent_flow.workflows.staircase_onboard.catalog_index generate)"
        )
    return found


# -------------------------------------------------------------------- sync


@dataclass(frozen=True)
class SyncResult:
    action: str  # skipped | current | committed | deferred | failed
    detail: str = ""
    paths: tuple[str, ...] = ()


def commit_with_script(script: str | Path, repo: str | Path) -> CommitFn:
    """A commit function that goes through the workspace's ``commit.sh``."""

    def commit(message: str, paths: Sequence[str]) -> tuple[bool, str]:
        try:
            completed = subprocess.run(
                [str(script), message, *paths],
                cwd=str(repo),
                capture_output=True,
                text=True,
                timeout=900,
                check=False,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            return False, f"{type(exc).__name__}: {exc}"
        return completed.returncode == 0, (completed.stdout + completed.stderr).strip()

    return commit


def sync(
    repo: str | Path,
    *,
    commit: CommitFn,
    catalog_rel: str = CATALOG_DIR,
    lock_path: str | Path | None = None,
) -> SyncResult:
    """Make ``index.yaml`` in HEAD what HEAD's fragments generate, in one commit of its own.

    Reads HEAD, not the working tree: a sibling's fragment that is written
    but not committed is not part of HEAD, so the index never names it. Once
    HEAD has a fragment but no header, migrates first (header plus a fragment
    per row); a committed index row with no fragment is adopted. Does nothing
    while HEAD has no fragment at all. Leaves an ``index.yaml`` with
    uncommitted edits alone -- someone else is writing it.

    Idempotent: a second call on the same HEAD is ``current``. With
    ``lock_path``, calls from several threads or processes are serialized
    under an ``flock`` on that file, so two of them can never both migrate;
    the one that waited reads the HEAD the other committed. A commit that
    reports failure is checked against HEAD before anything is undone, and
    the undo touches only the files this call created and that HEAD does not
    hold.
    """
    if lock_path is None:
        return _sync(repo, commit=commit, catalog_rel=catalog_rel)
    lock = Path(lock_path)
    lock.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(lock, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        return _sync(repo, commit=commit, catalog_rel=catalog_rel)
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def _sync(repo: str | Path, *, commit: CommitFn, catalog_rel: str) -> SyncResult:
    head = snapshot_at(repo, "HEAD", catalog_rel=catalog_rel)
    if head is None:
        return SyncResult("skipped", "not a git work tree")
    if head.header is None and not head.fragments:
        # Migrated lazily, by the first fragment a smith commits: a run that
        # adds no catalog entry never touches the catalog.
        return SyncResult("skipped", "no fragment committed yet; nothing to generate")
    root = Path(repo) / catalog_rel
    index = root / INDEX_FILENAME
    current = index.read_text(encoding="utf-8") if index.is_file() else None
    if current != head.index_text:
        return SyncResult("deferred", f"{INDEX_FILENAME} has uncommitted edits; left alone")

    new_files = migration_files(head)
    fragments = dict(head.fragments)
    header = head.header
    for rel, text in new_files.items():
        if rel == _header_rel():
            header = Header.from_yaml(text)
        else:
            fragments[rel] = _load_fragment(text) or {}
    target = Snapshot(header=header, fragments=fragments, files=head.files, index_text=None)
    rendered = render(target)
    if rendered == current and not new_files:
        return SyncResult("current", "index.yaml matches its fragments")

    for rel in new_files:
        if (root / rel).exists():
            return SyncResult("deferred", f"{rel} exists uncommitted; left alone")
    for rel, text in new_files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    index.write_text(rendered, encoding="utf-8")
    paths = [f"{catalog_rel}/{rel}" for rel in new_files] + [f"{catalog_rel}/{INDEX_FILENAME}"]
    message = (
        "chore(catalog): split index.yaml into per-entry fragments under index.d/"
        if head.header is None
        else "chore(catalog): regenerate index.yaml from index.d/ fragments"
    )
    ok, output = commit(message, paths)
    if ok:
        return SyncResult("committed", output.splitlines()[0] if output else "", tuple(paths))
    after = snapshot_at(repo, "HEAD", catalog_rel=catalog_rel)
    if (
        after is not None
        and after.index_text == rendered
        and all(rel in after.files for rel in new_files)
    ):
        # The commit landed and only its report failed: nothing to undo.
        return SyncResult("committed", f"landed; the helper said: {output[-500:]}", tuple(paths))
    committed = after.files if after is not None else frozenset()
    undo = [rel for rel in new_files if rel not in committed]
    # Unstage first (a helper that resets its own commit leaves the paths
    # staged), then remove what this call created, then put the index back.
    _git(
        repo,
        "reset",
        "-q",
        "--",
        *(f"{catalog_rel}/{rel}" for rel in undo),
        f"{catalog_rel}/{INDEX_FILENAME}",
    )
    for rel in undo:
        path = root / rel
        path.unlink(missing_ok=True)
        for parent in path.parents:
            if parent == root or not parent.is_dir() or any(parent.iterdir()):
                break
            parent.rmdir()
    if current is not None and (after is None or after.index_text == head.index_text):
        index.write_text(current, encoding="utf-8")
    return SyncResult("failed", output[-2000:], tuple(paths))


# --------------------------------------------------------------- workflow


class CatalogIndexHook:
    """The orchestrator's side: sync before a turn reads the index, check at delivery."""

    def __init__(self, workflow: Any) -> None:
        self._workflow = workflow

    def _repo(self) -> str:
        return str((getattr(self._workflow, "task_data", None) or {}).get("trtllm_repo_path") or "")

    def sync(self, log: Any = None) -> SyncResult:
        """Regenerate and commit the index if HEAD's fragments moved. Never raises.

        Commits through the workspace's ``commit.sh``, so it takes the same
        lock as every agent's commit. Without that script (before preflight
        has installed it) nothing is written.
        """
        from .committer import SCRIPT_NAME

        repo = self._repo()
        script = Path(self._workflow.workspace) / "bin" / SCRIPT_NAME
        if not repo or not script.is_file():
            return SyncResult("skipped", "no repository or no commit helper yet")
        result = sync(
            repo,
            commit=commit_with_script(script, repo),
            lock_path=Path(self._workflow.workspace) / LOCK_FILENAME,
        )
        if result.action in ("committed", "failed", "deferred"):
            self._workflow.journal.event(f"catalog index {result.action}", result.detail[:500])
        return result

    def checks(self) -> list[Any]:
        """Delivery checks: the delivered index is exactly what HEAD's fragments generate."""
        from .delivery import Check

        repo = self._repo()
        head = snapshot_at(repo) if repo else None
        if head is None or (head.header is None and not head.fragments):
            return []
        found = problems(head)
        return [
            Check(
                name="catalog:index-generated",
                passed=not found,
                detail="; ".join(found[:10])
                or f"{INDEX_FILENAME} is generated from {len(head.fragments)} fragment(s)",
            )
        ]

    def extend(self, report: Any) -> Any:
        """``report`` with :meth:`checks` appended."""
        from .delivery import DeliveryReport

        return DeliveryReport(checks=(*report.checks, *self.checks()))


# ----------------------------------------------------------------------- CLI


def main(argv: Sequence[str] | None = None) -> int:
    """``migrate``, ``generate`` and ``check`` a working tree's catalog index."""
    parser = argparse.ArgumentParser(prog="staircase-catalog-index", description=main.__doc__)
    parser.add_argument("command", choices=("migrate", "generate", "check"))
    parser.add_argument("--repo", required=True, help="the repository root")
    parser.add_argument("--catalog-rel", default=CATALOG_DIR)
    args = parser.parse_args(argv)
    root = Path(args.repo) / args.catalog_rel
    if args.command == "migrate":
        for path in migrate(root):
            sys.stdout.write(f"wrote {path}\n")
        return 0
    if args.command == "generate":
        (root / INDEX_FILENAME).write_text(render(snapshot_worktree(root)), encoding="utf-8")
        sys.stdout.write(f"wrote {root / INDEX_FILENAME}\n")
        return 0
    found = problems(snapshot_worktree(root))
    for problem in found:
        sys.stdout.write(f"PROBLEM {problem}\n")
    sys.stdout.write(f"{len(found)} problem(s)\n")
    return 1 if found else 0


__all__ = [
    "CatalogIndexHook",
    "FRAGMENTS_DIRNAME",
    "GENERATED_NOTICE",
    "HEADER_FILENAME",
    "Header",
    "INDEX_FILENAME",
    "Snapshot",
    "SyncResult",
    "commit_with_script",
    "fragment_rel",
    "migrate",
    "migration_files",
    "problems",
    "render",
    "renderable_rows",
    "snapshot_at",
    "snapshot_worktree",
    "split_index",
    "sync",
]


if __name__ == "__main__":
    sys.exit(main())
