# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""One git worktree per modeling candidate, with the main checkout's native build linked in.

A candidate is a branch ``tune/<slug>`` off the campaign's current best commit.
Python-only edits need nothing else: the spool runs a job against the worktree
by putting it first on ``PYTHONPATH`` (``#spool: tree=<worktree>``), and the
linked native artifacts make ``import tensorrt_llm`` resolve there. A
candidate that needs a native rebuild is ``BLOCKED``, not a worktree edit.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from ..infra.git import add_worktree, git_checked, link_native


def _complete(tree: Path) -> bool:
    """Whether ``tree`` is a fully created worktree (an index exists, no 'initializing' lock)."""
    gitdir = git_checked(tree, "rev-parse", "--absolute-git-dir", check=False)
    if not gitdir or not Path(gitdir, "index").exists():
        return False
    lock = Path(gitdir, "locked")
    return not (lock.exists() and "initializing" in lock.read_text(errors="replace"))


def open_candidate(
    repo: Path, root: Path, slug: str, base: str = "HEAD", prefix: str = "tune"
) -> Path:
    """Create (or reuse) ``root/<slug>`` on branch ``<prefix>/<slug>`` from ``base``.

    A run passes its own prefix: two campaigns in one repository must not share
    ``tune/best`` (git refuses a branch checked out by another worktree).
    """
    tree = root / slug
    if tree.exists():
        if _complete(tree):
            return tree
        # A worktree add killed part way (an orchestrator stopped mid-checkout) leaves the tree
        # locked "initializing" without an index. Rebuild it rather than hand it out.
        git_checked(repo, "worktree", "unlock", str(tree), check=False)
        git_checked(repo, "worktree", "remove", "--force", str(tree), check=False)
        git_checked(repo, "worktree", "prune", check=False)
        shutil.rmtree(tree, ignore_errors=True)
    root.mkdir(parents=True, exist_ok=True)
    branch = f"{prefix}/{slug}"
    exists = git_checked(repo, "rev-parse", "--verify", "--quiet", branch, check=False)
    if exists:
        add_worktree(repo, tree, str(tree), branch)
    else:
        add_worktree(repo, tree, "-b", branch, str(tree), base)
    link_native(repo, tree)
    return tree
