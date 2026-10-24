# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Questions to git about a checkout, asked on this host with an argv (never a shell).

Every call passes ``--no-optional-locks``: a ``git status`` otherwise refreshes
the index under ``index.lock``, and the orchestrator asks one (the target's
marker) while a task's branch is being fast-forwarded into the same checkout,
which then fails on the lock.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

# Seconds. A git query that takes longer than this is a sick host, and the
# gate's answer is `unknown` rather than a stall in front of a multi-hour run.
_GIT_TIMEOUT_S = 15


def git_output(repo_root: str | Path, *args: str) -> str | None:
    """Ask git one question about ``repo_root``. ``None`` if it cannot answer.

    Every failure mode collapses to ``None`` on purpose: a gate that dies because
    a VCS query died reports a tooling failure as a modeling failure.
    """
    try:
        completed = subprocess.run(
            ["git", "--no-optional-locks", "-C", str(repo_root), *args],
            capture_output=True,
            text=True,
            timeout=_GIT_TIMEOUT_S,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout


def git_checked(repo: str | Path, *args: str, check: bool = True) -> str:
    """Run git in ``repo`` and return its stripped stdout; raise on failure when ``check``."""
    out = subprocess.run(
        ["git", "--no-optional-locks", "-C", str(repo), *args], capture_output=True, text=True
    )
    if check and out.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)}: {out.stderr.strip()}")
    return out.stdout.strip()


#: The first git whose ``merge-tree --write-tree`` takes ``--merge-base``.
MERGE_TREE_MERGE_BASE = (2, 40)


def git_version() -> tuple[int, ...] | None:
    """This host's git, as numbers (``git version 2.39.3`` is ``(2, 39, 3)``); ``None`` without one."""
    try:
        completed = subprocess.run(
            ["git", "--version"], capture_output=True, text=True, timeout=_GIT_TIMEOUT_S
        )
    except (OSError, subprocess.SubprocessError):
        return None
    words = completed.stdout.split()
    if completed.returncode != 0 or len(words) < 3:
        return None
    numbers = []
    for part in words[2].split("."):
        if not part.isdigit():
            break
        numbers.append(int(part))
    return tuple(numbers) or None


def resolve_commit(repo: str | Path, rev: str = "HEAD") -> str | None:
    """The full commit ``rev`` names in ``repo``; ``None`` when it names none or git cannot say."""
    answer = git_output(repo, "rev-parse", "--verify", "--quiet", f"{rev}^{{commit}}") or ""
    return answer.strip() or None


__all__ = ["MERGE_TREE_MERGE_BASE", "git_checked", "git_output", "git_version", "resolve_commit"]
