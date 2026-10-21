#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Rebuild native libraries without crashing the processes that have them mapped.

``build_wheel.py`` installs ``tensorrt_llm/libs/*.so`` by rewriting each file
**in place**. A process on any node that has the old file mapped then faults on
its next page-in of the changed pages, so a rebuild on one session can kill
another session. Two measures, both here:

1. **Fresh inodes before the build.** Every installed shared library is copied
   to a temporary file in the same directory and renamed over the original.
   Running processes keep the old, now-unlinked inode; the build's in-place
   rewrite lands on a file nobody has mapped.
2. **Announce the rebuild on every session.** A marker in each spool's
   ``rebuild/`` directory holds the admission of GPU jobs that load the
   rebuilt libraries until the build ends, so no process maps a library
   mid-rewrite: jobs against that tree, and jobs against any tree whose
   native build resolves into it. A snapshot or worktree links the
   checkout's native build (``tensorrt_llm/libs``, the bindings) rather than
   building its own (:func:`native_locations`). The marker is removed when
   the build ends and ignored once it is older than its ``max_age_s`` (a
   build that died without cleaning up).

``infra/scripts/incremental_build.sh`` drives both around ``build_wheel.py``.

Standard-library only and runnable by path::

    rebuild.py refresh-inodes --root <checkout>/tensorrt_llm
    rebuild.py begin --tree <checkout> --spool A --spool B [--note TEXT]   # prints a token
    rebuild.py end --token TOKEN --spool A --spool B
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import sys
import time
import uuid
from pathlib import Path
from typing import Iterable, Mapping, Sequence

#: Where a spool keeps rebuild announcements.
REBUILD_DIRNAME = "rebuild"

#: The package directory of a checkout; its native build lies in it.
PACKAGE_DIRNAME = "tensorrt_llm"

#: An announcement older than this is from a build that died; it holds nothing.
DEFAULT_MAX_AGE_S = 4 * 3600.0

#: Shared-library name patterns whose inodes are refreshed.
LIBRARY_PATTERNS = ("*.so", "*.so.*")


def replace_atomically(source: Path, target: Path) -> None:
    """Install ``source``'s bytes at ``target`` through a temporary file and a rename.

    The rename swaps the directory entry; a process that has ``target`` open
    or mapped keeps the old inode and never sees a partial file.
    """
    staging = target.with_name(f".{target.name}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp")
    try:
        shutil.copy2(source, staging)
        with staging.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(staging, target)
    finally:
        if staging.exists():
            staging.unlink()


def refresh_inodes(root: Path, patterns: Iterable[str] = LIBRARY_PATTERNS) -> list[Path]:
    """Give every shared library under ``root`` a fresh inode with identical contents."""
    refreshed = []
    seen: set[Path] = set()
    for pattern in patterns:
        for path in sorted(Path(root).rglob(pattern)):
            if path in seen or path.is_symlink() or not path.is_file():
                continue
            seen.add(path)
            replace_atomically(path, path)
            refreshed.append(path)
    return refreshed


def announce(
    spools: Iterable[str | Path],
    tree: str | Path,
    *,
    note: str = "",
    max_age_s: float = DEFAULT_MAX_AGE_S,
    token: str | None = None,
) -> str:
    """Write a rebuild announcement into every spool and return its token.

    It records where the rebuilt tree's native build really is
    (:func:`native_locations`), so each scheduler holds the jobs that load it
    without resolving the tree again.
    """
    token = token or f"{int(time.time())}-{uuid.uuid4().hex[:8]}"
    record = {
        "token": token,
        "tree": os.path.abspath(str(tree)),
        "native": native_locations(tree),
        "host": socket.gethostname(),
        "pid": os.getpid(),
        "started": time.time(),
        "max_age_s": max_age_s,
        "note": note,
    }
    for spool in spools:
        directory = Path(spool) / REBUILD_DIRNAME
        directory.mkdir(parents=True, exist_ok=True)
        staging = directory / f".{token}.json.tmp"
        staging.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        os.replace(staging, directory / f"{token}.json")
    return token


def retract(spools: Iterable[str | Path], token: str) -> None:
    """Remove a rebuild announcement from every spool."""
    for spool in spools:
        (Path(spool) / REBUILD_DIRNAME / f"{token}.json").unlink(missing_ok=True)


def active_announcements(spool: str | Path, now: float | None = None) -> list[dict]:
    """Announcements in ``spool`` young enough to still hold admission."""
    now = time.time() if now is None else now
    out = []
    for path in sorted((Path(spool) / REBUILD_DIRNAME).glob("*.json")):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(record, dict):
            continue
        age = now - float(record.get("started", 0))
        if age <= float(record.get("max_age_s", DEFAULT_MAX_AGE_S)):
            out.append(record)
    return out


def same_tree(a: str | None, b: str | None) -> bool:
    """Whether two paths name one directory, however each is spelled."""
    if not a or not b:
        return False
    try:
        return os.path.samefile(a, b)
    except OSError:
        return os.path.realpath(a) == os.path.realpath(b)


def native_locations(tree: str | Path) -> list[str]:
    """Where the native build a process importing from ``tree`` loads really is.

    The real path of ``tree``'s package directory, and that of every symlink
    directly in it: a snapshot or worktree links the checkout's native build
    (``tensorrt_llm/libs``, the bindings) that way rather than building its
    own, and a whole package directory may be a link too.
    """
    package = os.path.join(os.path.abspath(str(tree)), PACKAGE_DIRNAME)
    found = {os.path.realpath(package)}
    try:
        with os.scandir(package) as entries:
            for entry in entries:
                if entry.is_symlink():
                    found.add(os.path.realpath(entry.path))
    except OSError:
        pass  # no package there: it imports nothing from this tree
    return sorted(found)


def _within(path: str, root: str) -> bool:
    return path == root or path.startswith(root.rstrip(os.sep) + os.sep)


def loads_rebuilt(
    tree: str | None,
    marker: Mapping[str, object],
    locations: Sequence[str] | None = None,
) -> bool:
    """Whether a job against ``tree`` loads native libraries the rebuild ``marker`` rewrites.

    It does when ``tree`` is the rebuilt tree, and when one of its native
    locations (``locations``, else :func:`native_locations` of ``tree``) lies
    in one of the rebuilt tree's, or holds one: a snapshot whose
    ``tensorrt_llm/libs`` links into the rebuilt checkout, or a checkout whose
    package directory a rebuilt worktree's links point into. A marker written
    without its locations has them resolved here.
    """
    rebuilt_tree = marker.get("tree")
    if not tree or not rebuilt_tree:
        return False
    if same_tree(tree, str(rebuilt_tree)):
        return True
    rebuilt = marker.get("native")
    if not isinstance(rebuilt, list):
        rebuilt = native_locations(str(rebuilt_tree))
    mine = native_locations(tree) if locations is None else locations
    return any(_within(a, str(b)) or _within(str(b), a) for a in mine for b in rebuilt)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rebuild.py", description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="action", required=True)
    refresh = sub.add_parser("refresh-inodes", help="fresh inodes for every shared library")
    refresh.add_argument("--root", required=True, type=Path)
    begin = sub.add_parser("begin", help="announce a rebuild on every spool; prints the token")
    begin.add_argument("--tree", required=True)
    begin.add_argument("--spool", action="append", default=[])
    begin.add_argument("--note", default="")
    begin.add_argument("--max-age", type=float, default=DEFAULT_MAX_AGE_S)
    end = sub.add_parser("end", help="retract a rebuild announcement")
    end.add_argument("--token", required=True)
    end.add_argument("--spool", action="append", default=[])
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.action == "refresh-inodes":
        refreshed = refresh_inodes(args.root)
        print(f"rebuild: fresh inodes for {len(refreshed)} shared librar(ies) under {args.root}")
        return 0
    if args.action == "begin":
        print(announce(args.spool, args.tree, note=args.note, max_age_s=args.max_age))
        return 0
    retract(args.spool, args.token)
    return 0


__all__ = [
    "DEFAULT_MAX_AGE_S",
    "LIBRARY_PATTERNS",
    "PACKAGE_DIRNAME",
    "REBUILD_DIRNAME",
    "active_announcements",
    "announce",
    "loads_rebuilt",
    "native_locations",
    "refresh_inodes",
    "replace_atomically",
    "retract",
    "same_tree",
]


if __name__ == "__main__":
    sys.exit(main())
