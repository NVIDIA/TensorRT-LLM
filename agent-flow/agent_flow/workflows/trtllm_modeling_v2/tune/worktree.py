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

import contextlib
import fcntl
import shutil
import subprocess
import time
from pathlib import Path

from ..infra.vcs import git_checked

NATIVE_ROOT = "tensorrt_llm"


def native_artifacts(repo: str | Path) -> list[str]:
    """Git-ignored entries directly under ``tensorrt_llm/`` in ``repo``: the native build."""
    out = git_checked(
        Path(repo),
        *("ls-files", "--others", "--ignored", "--exclude-standard", "--directory"),
        *("--", f"{NATIVE_ROOT}/"),
        check=False,
    )
    found = []
    for line in out.splitlines():
        parts = line.strip().rstrip("/").split("/")
        if len(parts) == 2 and parts[0] == NATIVE_ROOT and parts[1] != "__pycache__":
            if not parts[1].endswith(".egg-info"):
                found.append("/".join(parts))
    return found


@contextlib.contextmanager
def repo_lock(repo: Path):
    """Serialize worktree creation on one repository.

    Concurrent `git worktree add` on one repository (a campaign stacking a candidate
    while a tuner request opens another) fails with `Could not reset index file to
    revision 'HEAD'`.
    """
    path = Path(git_checked(repo, "rev-parse", "--git-common-dir"))
    if not path.is_absolute():
        path = repo / path
    with open(path / "mv2flow-worktree.lock", "w") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)


def add_worktree(repo: Path, tree: Path, *args: str) -> None:
    """`git worktree add` under the repo lock, retried once after cleaning up a failed attempt.

    Only the registration holds the lock (``--no-checkout``); the checkout, the slow part, runs
    outside it against the worktree's own index, so several candidates check out at once.
    """
    with repo_lock(repo):
        for attempt in (1, 2):
            out = subprocess.run(
                ["git", "-C", str(repo), "worktree", "add", "--no-checkout", *args],
                capture_output=True,
                text=True,
            )
            if out.returncode == 0:
                break
            git_checked(repo, "worktree", "remove", "--force", str(tree), check=False)
            git_checked(repo, "worktree", "prune", check=False)
            if attempt == 2:
                raise RuntimeError(f"git worktree add {' '.join(args)}: {out.stderr.strip()}")
            time.sleep(5)
    git_checked(tree, "reset", "--hard", "-q")


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
