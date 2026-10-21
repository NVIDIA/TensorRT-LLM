# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``source_embargo`` entries: what they may name, and how a role prompt shows them.

``task.yaml`` names files and trees no agent may read: an implementation of
this exact checkpoint that already exists, here or in a sibling engine's
checkout, or the workflow's own worked example of the model. Every role
prompt states the embargo (``prompts._common.source_embargo_block``).

Entries are repo-relative, absolute (a tree outside the repository), or
``package:<path>``, relative to this workflow package wherever it is
installed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

PACKAGE_PREFIX = "package:"
PACKAGE_ROOT = Path(__file__).resolve().parent.parent


class EmbargoEntryError(ValueError):
    """An embargo entry that names nothing it is allowed to name."""


def package_entry_problem(entry: str) -> str:
    """Why a ``package:`` entry is unusable, or ``""``.

    It must name something strictly **inside** the workflow package:
    ``package:../perf_analyze`` resolves to a sibling workflow, and an entry
    for the package root would embargo the workflow's own skills.
    """
    root = PACKAGE_ROOT.resolve()
    target = (PACKAGE_ROOT / entry[len(PACKAGE_PREFIX) :].lstrip("/")).resolve()
    if target == root:
        return "names the whole workflow package; name a file or directory inside it"
    if root not in target.parents:
        return f"resolves to {target}, outside the workflow package {root}"
    return ""


def resolve_entry(entry: str, repo_root: str | Path | None = None) -> Path:
    """The absolute path an embargo entry names (symlinks not resolved).

    Raises :class:`EmbargoEntryError` for a ``package:`` entry that escapes
    the workflow package; the task schema reports that before a run starts.
    """
    if entry.startswith(PACKAGE_PREFIX):
        problem = package_entry_problem(entry)
        if problem:
            raise EmbargoEntryError(f"{entry!r} {problem}")
        return PACKAGE_ROOT / entry[len(PACKAGE_PREFIX) :].lstrip("/")
    raw = Path(entry)
    if raw.is_absolute() or repo_root is None:
        return raw
    return Path(repo_root) / raw


def display_entry(entry: str, repo_root: str | Path | None = None) -> str:
    """How a role prompt names an entry.

    Repo and absolute entries pass through. A ``package:`` entry keeps its
    prefix, so the prompt's embargo block can give it its own wording — it
    is the workflow's reference material, not an implementation of the
    checkpoint — followed by the path an agent would actually meet:
    repo-relative when the package sits inside the repository, else absolute.
    """
    if not entry.startswith(PACKAGE_PREFIX):
        return entry
    absolute = resolve_entry(entry, repo_root)
    shown = str(absolute)
    if repo_root is not None:
        try:
            shown = absolute.resolve().relative_to(Path(repo_root).resolve()).as_posix()
        except ValueError:
            pass
    return f"{PACKAGE_PREFIX}{shown}"


def display_entries(task_data: Mapping[str, Any]) -> list[str]:
    """``source_embargo`` as the role prompts render it."""
    repo = task_data.get("trtllm_repo_path") or None
    return [display_entry(str(e), repo) for e in (task_data.get("source_embargo") or [])]


__all__ = [
    "EmbargoEntryError",
    "PACKAGE_PREFIX",
    "PACKAGE_ROOT",
    "display_entries",
    "display_entry",
    "package_entry_problem",
    "resolve_entry",
]
