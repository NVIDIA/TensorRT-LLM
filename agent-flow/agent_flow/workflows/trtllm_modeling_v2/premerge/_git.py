# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The one way this package runs git."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Mapping


class GitError(RuntimeError):
    """A git command this package depends on failed."""


def git(
    repo: str | Path,
    *args: str,
    env: Mapping[str, str] | None = None,
    input_: str | None = None,
    check: bool = True,
) -> str:
    """Run ``git -C repo <args>`` and return its stdout.

    Raises :class:`GitError` naming the command and git's own message when
    ``check`` is set and git exits non-zero.
    """
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        text=True,
        env=dict(env) if env is not None else None,
        input=input_,
    )
    if check and result.returncode != 0:
        raise GitError(f"git {' '.join(args)} failed: {result.stderr.strip()}")
    return result.stdout


def rev_parse(repo: str | Path, rev: str) -> str:
    return git(repo, "rev-parse", "--verify", f"{rev}^{{commit}}").strip()


def dirty_tracked(repo: str | Path) -> list[str]:
    """Tracked files with uncommitted changes (untracked files are ignored)."""
    out = git(repo, "status", "--porcelain", "--untracked-files=no")
    return [line for line in out.splitlines() if line.strip()]


def changed_files(repo: str | Path, base: str, head: str = "HEAD") -> list[str]:
    """Files ``head`` changes relative to ``base``, deletions excluded.

    The same set CI's pre-merge release check hands to pre-commit.
    """
    out = git(repo, "diff", "--name-only", "--diff-filter=d", base, head)
    return [line for line in out.splitlines() if line.strip()]


__all__ = ["GitError", "changed_files", "dirty_tracked", "git", "rev_parse"]
