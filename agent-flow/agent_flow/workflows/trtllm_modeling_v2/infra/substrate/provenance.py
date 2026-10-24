# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Which checkout did each process of a job import, and does the result stand?

The resolve hook (``resolvehook/sitecustomize.py``) appends one record per
process and watched module to the job's ``logs/<name>.resolve.jsonl``. This
module reads those records back and answers one question: **did every process
import the watched package from the same checkout -- and from the job's
declared tree, when it declared one?**

The answer can silently be "no": an MPI worker resolves ``tensorrt_llm``
from its working directory ahead of ``PYTHONPATH``, so a job that points only
``PYTHONPATH`` at a worktree gets the worktree on rank 0 and the main checkout
on every other rank, and its result describes neither.

Checkouts are compared by directory identity (device and inode), not by path
spelling, because one storage system can be mounted under two prefixes.

Standard-library only; the scheduler imports it by path.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping


def checkout_root(module: str, origin: str) -> str:
    """The directory ``module`` was imported from: the one that holds its top-level package."""
    path = Path(origin)
    package = path.parent if path.name == "__init__.py" else path.with_suffix("")
    root = package
    for _ in module.split("."):
        root = root.parent
    return str(root)


def _identity(path: str) -> tuple[int, int] | str:
    try:
        stat = os.stat(path)
    except OSError:
        return os.path.realpath(path)
    return (stat.st_dev, stat.st_ino)


def read_records(path: str | Path) -> list[dict]:
    """Every well-formed record in a resolve log; a missing log has none."""
    records = []
    try:
        lines = Path(path).read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return records
    for line in lines:
        try:
            record = json.loads(line)
        except ValueError:
            continue
        if isinstance(record, dict) and record.get("module"):
            records.append(record)
    return records


@dataclass
class ProvenanceVerdict:
    """Whether a job's processes agree on where they imported from."""

    ok: bool
    #: Checkout root -> how many processes imported from it, per module.
    roots: dict[str, dict[str, int]] = field(default_factory=dict)
    #: Modules a process imported without the import resolving at all.
    unresolved: list[str] = field(default_factory=list)
    message: str = ""


def check(records: Iterable[Mapping[str, object]], tree: str | None = None) -> ProvenanceVerdict:
    """Judge a job's resolve records.

    Args:
        records: Resolve-log entries (``module``, ``origin``, ``pid``, ...).
        tree: The checkout the job declared, if it declared one.

    Returns:
        A verdict that fails when any module resolved to more than one
        checkout, or to a checkout other than ``tree``.
    """
    roots: dict[str, dict[str, int]] = {}
    unresolved: list[str] = []
    for record in records:
        module = str(record.get("module"))
        origin = record.get("origin")
        if not origin:
            unresolved.append(module)
            continue
        root = checkout_root(module, str(origin))
        roots.setdefault(module, {})
        roots[module][root] = roots[module].get(root, 0) + 1

    problems = []
    expected = _identity(tree) if tree else None
    for module, seen in sorted(roots.items()):
        identities = {_identity(root) for root in seen}
        listing = ", ".join(f"{root} ({count} process(es))" for root, count in sorted(seen.items()))
        if len(identities) > 1:
            problems.append(f"{module} resolved to {len(identities)} checkouts: {listing}")
        elif expected is not None and identities != {expected}:
            problems.append(f"{module} resolved to {listing}, not to the job's tree {tree}")
    return ProvenanceVerdict(
        ok=not problems,
        roots=roots,
        unresolved=sorted(set(unresolved)),
        message="; ".join(problems),
    )


__all__ = ["ProvenanceVerdict", "check", "checkout_root", "read_records"]
