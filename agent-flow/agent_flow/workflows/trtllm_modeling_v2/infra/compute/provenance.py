# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Which checkout and which build did each process of a job run, and does the result stand?

The resolve hook (``resolvehook/sitecustomize.py``) appends one record per process and watched
module to the job's ``logs/<name>.resolve.jsonl``: where the module came from, the identity of
the native build beside it, and the role instance and world rank of the process. :func:`check`
asks whether every process imported the watched package from one checkout with one build, and
from the job's declared tree if any; :func:`missing_ranks` whether every rank the job stated
was heard from; :func:`check_pinned` whether every process of a pinned job ran the tree, and
found the libraries, that were verified against the pin's build record on its host
(``pintree.py``). The answer can silently be "no": an MPI worker resolves ``tensorrt_llm`` from
its working directory ahead of ``PYTHONPATH``, so a job that points only ``PYTHONPATH`` at a
worktree gets the worktree on rank 0, the main checkout on every other rank, and a result that
describes neither.

Checkouts are compared by directory identity (device and inode), not by path spelling: one
storage system can be mounted under two prefixes. Standard-library only; the scheduler imports
it by path.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence


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
    message: str = ""


def check(records: Iterable[Mapping[str, object]], tree: str | None = None) -> ProvenanceVerdict:
    """A verdict that fails when a module resolved to more than one checkout, or not to ``tree``."""
    roots: dict[str, dict[str, int]] = {}
    records_seen = list(records)
    for record in records_seen:
        module = str(record.get("module"))
        origin = record.get("origin")
        if not origin:
            continue
        root = checkout_root(module, str(origin))
        roots.setdefault(module, {})
        roots[module][root] = roots[module].get(root, 0) + 1

    problems = []
    builds = {str(record["build"]) for record in records_seen if record.get("build")}
    if len(builds) > 1:
        problems.append(
            f"the processes loaded {len(builds)} native builds: {', '.join(sorted(builds))}"
        )
    expected = _identity(tree) if tree else None
    for module, seen in sorted(roots.items()):
        identities = {_identity(root) for root in seen}
        listing = ", ".join(f"{root} ({count} process(es))" for root, count in sorted(seen.items()))
        if len(identities) > 1:
            problems.append(f"{module} resolved to {len(identities)} checkouts: {listing}")
        elif expected is not None and identities != {expected}:
            problems.append(f"{module} resolved to {listing}, not to the job's tree {tree}")
    return ProvenanceVerdict(ok=not problems, message="; ".join(problems))


def check_pinned(
    records: Iterable[Mapping[str, object]], trees: Iterable[Mapping[str, object]]
) -> str:
    """Why a pinned job did not run what its pin vouches for; ``""`` when it did.

    ``trees`` are the lines of the job's tree record: per host, the tree made there for the
    pin, and the identity of the build verified there (the name of the pin's build record,
    once the libraries' content had been checked against it). A job stands only if a tree was
    verified, none failed, and every process that imported the watched package imported it
    from the tree verified for its host and reports that same identity for the libraries it
    found, by their content.
    """
    trees = list(trees)
    for entry in trees:
        if not entry.get("ok"):
            why = entry.get("detail") or "the pin's tree could not be made"
            return f"{why} (on {entry.get('host')})"
    if not trees:
        return "no tree was made and verified for the job's pin"
    by_host = {str(entry.get("host")): entry for entry in trees}
    builds = sorted({(str(entry.get("build")), host) for host, entry in by_host.items()})
    if len({build for build, _ in builds}) > 1:
        # Only a pin taken where its libraries could not be read leaves this to its nodes.
        said = ", ".join(f"{host} {build}" for build, host in builds)
        return f"the nodes of the job found different libraries for its pin: {said}"
    here = os.uname().nodename
    for record in records:
        origin = record.get("origin")
        if not origin:
            continue
        module, host = str(record.get("module")), str(record.get("host"))
        entry = by_host.get(host)
        if entry is None:
            return (
                f"a process on {host} imported {module}, and no tree of the pin was verified there"
            )
        root, tree = checkout_root(module, str(origin)), str(entry.get("tree"))
        same = (
            _identity(root) == _identity(tree) if host == here else os.path.normpath(root) == tree
        )
        if not same:
            return f"{module} resolved to {root} on {host}, not to the pin's tree {tree}"
        if record.get("build") != entry.get("build"):
            return (
                f"a process on {host} (pid {record.get('pid')}) found other libraries beside "
                f"{module} than the ones verified there against the pin's build record"
            )
    return ""


def missing_ranks(
    records: Iterable[Mapping[str, object]],
    ranks: Sequence[tuple[str, int]],
    modules: Sequence[str] = (),
) -> list[str]:
    """The ``role[rank]`` a job stated (``ranks``) that no complete record came from.

    A record counts for a rank only when it is complete: of a watched module (``modules``,
    when given), naming where the module came from **and** the identity of its build, with a
    world rank that is a number. With one role stated, a record that names no role counts for
    it. A process outside any rank world (the hook writes ``-``) is rank 0 only of a role
    whose world is one rank. A malformed or unrelated record satisfies nothing.
    """
    heard: set[tuple[str, int]] = set()
    only = ranks[0][0] if len(ranks) == 1 else None
    sizes = dict(ranks)
    for record in records:
        if modules and str(record.get("module")) not in modules:
            continue
        if not record.get("origin") or not record.get("build"):
            continue
        role = str(record.get("role") or only or "")
        raw = record.get("rank")
        if isinstance(raw, int) and not isinstance(raw, bool):
            rank = raw
        elif isinstance(raw, str) and raw.isdigit():
            rank = int(raw)
        elif raw == "-" and sizes.get(role) == 1:
            rank = 0
        else:
            continue
        heard.add((role, rank))
    return [
        f"{role}[{rank}]"
        for role, size in ranks
        for rank in range(size)
        if (role, rank) not in heard
    ]


__all__ = [
    "ProvenanceVerdict",
    "check",
    "check_pinned",
    "checkout_root",
    "missing_ranks",
    "read_records",
]
