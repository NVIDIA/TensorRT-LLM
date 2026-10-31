#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Rebuild native libraries without crashing the processes that have them mapped.

``build_wheel.py`` installs ``tensorrt_llm/libs/*.so`` by rewriting each file **in place**, so a
process on any node with the old file mapped faults on its next page-in: a rebuild on one session
can kill another. Two measures, which ``infra/compute/scripts/incremental_build.sh`` drives around
``build_wheel.py``:

1. **Fresh inodes before the build.** Every installed shared library is copied to a temporary
   file beside it and renamed over the original: running processes keep the old, unlinked inode,
   and the build's rewrite lands on a file nobody has mapped.
2. **Announce the rebuild on every session.** A marker in each spool's ``rebuild/`` holds the
   GPU jobs that load the rebuilt libraries (:func:`loads_rebuilt`) until the build ends, so no
   process maps a library mid-rewrite. A marker older than its ``max_age_s`` is from a build that
   died, and holds nothing.

Standard-library only and runnable by path::

    rebuild.py refresh-inodes --root <checkout>/tensorrt_llm
    rebuild.py begin --tree <checkout> --spool A --spool B [--note TEXT]   # prints a token
    rebuild.py end --token TOKEN --spool A --spool B
"""

from __future__ import annotations

import argparse
import os
import shutil
import socket
import sys
import time
import uuid
from pathlib import Path
from typing import Iterable, Mapping, Sequence

try:
    from ..files import read_json, write_json_atomic
except ImportError:  # run by path: shared file helpers live in the parent directory
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from files import read_json, write_json_atomic

#: Where a spool keeps rebuild announcements.
REBUILD_DIRNAME = "rebuild"

#: The package directory of a checkout; its native build lies in it.
PACKAGE_DIRNAME = "tensorrt_llm"

#: An announcement older than this is from a build that died; it holds nothing.
DEFAULT_MAX_AGE_S = 4 * 3600.0

#: Shared-library name patterns whose inodes are refreshed.
LIBRARY_PATTERNS = ("*.so", "*.so.*")


def replace_atomically(source: Path, target: Path) -> None:
    """Install ``source``'s bytes at ``target`` through a temporary file and a rename."""
    staging = target.with_name(f".{target.name}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp")
    try:
        shutil.copy2(source, staging)
        with staging.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(staging, target)
    finally:
        if staging.exists():
            staging.unlink()


def refresh_inodes(root: Path) -> list[Path]:
    """Give every shared library under ``root`` a fresh inode with identical contents."""
    refreshed = []
    seen: set[Path] = set()
    for pattern in LIBRARY_PATTERNS:
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
) -> str:
    """Write a rebuild announcement into every spool and return its token.

    It records the rebuilt tree's :func:`native_locations`, so no scheduler resolves them again.
    """
    token = f"{int(time.time())}-{uuid.uuid4().hex[:8]}"
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
        write_json_atomic(directory / f"{token}.json", record)
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
        record = read_json(path)
        if record is None:
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

    The real path of ``tree``'s package directory and of every symlink directly in it: a snapshot
    or worktree links the checkout's native build (``tensorrt_llm/libs``, the bindings) rather
    than building its own, and a whole package directory may be a link too.
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


def loads_rebuilt(tree: str | None, marker: Mapping[str, object], locations: Sequence[str]) -> bool:
    """Whether a job against ``tree`` loads native libraries the rebuild ``marker`` rewrites.

    It does when ``tree`` is the rebuilt tree, or one of its native ``locations`` lies in one of
    the rebuilt tree's or holds one: a snapshot whose ``tensorrt_llm/libs`` links into the rebuilt
    checkout, or a checkout whose package directory a rebuilt worktree links into.
    """
    rebuilt_tree = marker.get("tree")
    if not tree or not rebuilt_tree:
        return False
    if same_tree(tree, str(rebuilt_tree)):
        return True
    rebuilt = marker.get("native")
    if not isinstance(rebuilt, list):
        rebuilt = native_locations(str(rebuilt_tree))
    return any(_within(a, str(b)) or _within(str(b), a) for a in locations for b in rebuilt)


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
