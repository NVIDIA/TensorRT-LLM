# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""One git worktree per modeling candidate, with the main checkout's native build linked in.

A candidate is a branch ``tune/<slug>`` off the campaign's current best commit.
Python-only edits need nothing else: the spool runs a job against the worktree
by putting it first on ``PYTHONPATH`` (``#spool: tree=<worktree>``), and the
linked native artifacts make ``import tensorrt_llm`` resolve there. A
candidate that changes native code is a ``NEED_REBUILD``, not a worktree edit.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from ..overlap import native_artifacts


def _git(repo: Path, *args: str, check: bool = True) -> str:
    out = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True)
    if check and out.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {out.stderr.strip()}")
    return out.stdout.strip()


def open_candidate(repo: Path, root: Path, slug: str, base: str = "HEAD") -> Path:
    """Create (or reuse) ``root/<slug>`` on branch ``tune/<slug>`` from ``base``."""
    tree = root / slug
    if tree.exists():
        return tree
    root.mkdir(parents=True, exist_ok=True)
    branch = f"tune/{slug}"
    exists = _git(repo, "rev-parse", "--verify", "--quiet", branch, check=False)
    if exists:
        _git(repo, "worktree", "add", str(tree), branch)
    else:
        _git(repo, "worktree", "add", "-b", branch, str(tree), base)
    link_native(repo, tree)
    return tree


def link_native(repo: Path, tree: Path) -> list[str]:
    linked = []
    for rel in native_artifacts(repo):
        target = tree / rel
        if target.exists() or target.is_symlink():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        target.symlink_to(repo / rel)
        linked.append(rel)
    return linked


def candidate_commits(repo: Path, tree: Path, base: str) -> list[str]:
    return _git(tree, "rev-list", "--reverse", f"{base}..HEAD").splitlines()


def close_candidate(repo: Path, tree: Path, *, delete_branch: bool = False) -> None:
    branch = _git(tree, "rev-parse", "--abbrev-ref", "HEAD", check=False)
    _git(repo, "worktree", "remove", "--force", str(tree), check=False)
    if delete_branch and branch.startswith("tune/"):
        _git(repo, "branch", "-D", branch, check=False)
