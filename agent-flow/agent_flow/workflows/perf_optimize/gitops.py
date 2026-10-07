"""Git operations on the TensorRT-LLM checkout, owned by the orchestrator.

The perf-optimize agents edit files in ``trtllm_repo_path`` but never run
git state commands themselves — the orchestrator commits accepted roadmap
items and reverts rejected ones through these wrappers, so the repo's
history stays deterministic regardless of what an agent did in its turn.

All helpers shell out locally to ``git -C <repo> ...`` and raise
:class:`GitOpsError` (with the captured stderr) on failure, so a broken
repo aborts the run loudly instead of silently optimizing against an
inconsistent tree. Remote execution is a runtime boundary only: agents
copy source snapshots to the execution host, while Git remains local.
"""

from __future__ import annotations

import subprocess
from pathlib import Path


class GitOpsError(RuntimeError):
    """Raised when a git command fails."""


def _git(repo: str | Path, *args: str) -> str:
    """Run ``git -C repo *args`` locally and return stripped stdout."""
    cmd = ["git", "-C", str(repo), *args]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise GitOpsError(
            f"`{' '.join(cmd)}` failed with exit code {result.returncode}: "
            f"{result.stderr.strip() or result.stdout.strip()}"
        )
    return result.stdout.strip()


def is_git_repo(repo: str | Path) -> bool:
    """True iff ``repo`` is inside a git working tree."""
    try:
        return _git(repo, "rev-parse", "--is-inside-work-tree") == "true"
    except (GitOpsError, OSError):
        return False


def worktree_clean(repo: str | Path) -> bool:
    """True iff ``git status --porcelain`` reports nothing.

    Gitignored files (build artifacts in an editable TRT-LLM checkout) do
    not show up in porcelain output, so they neither block a fresh start
    nor get removed by :func:`discard_uncommitted`.

    ``--ignore-submodules=all`` because a *built* worktree breaks plain
    ``git status``: its submodule ``.git`` files point at gitdirs under
    ``<base>/.git/worktrees/<name>/modules/...`` which need not exist, so git
    exits 128 with ``fatal: not a git repository: 3rdparty/MSA/...`` while
    stdout — the actual answer — is empty. Clean tree, failed command, and an
    error naming a missing repository rather than uncommitted work.

    Observed on the real GB300 debug worktree. Narrowing to the superproject is
    also the correct scope: this guards uncommitted work that
    :func:`discard_uncommitted` would destroy, and ``reset --hard`` +
    ``clean -fd`` act on the superproject.
    """
    return _git(repo, "status", "--porcelain", "--ignore-submodules=all") == ""


def current_branch(repo: str | Path) -> str:
    return _git(repo, "rev-parse", "--abbrev-ref", "HEAD")


def rev_parse_head(repo: str | Path) -> str:
    return _git(repo, "rev-parse", "HEAD")


def create_branch(repo: str | Path, name: str) -> None:
    """Create and check out ``name`` (fails if it already exists)."""
    _git(repo, "checkout", "-b", name)


def checkout(repo: str | Path, name: str) -> None:
    _git(repo, "checkout", name)


def create_worktree(
    repo: str | Path,
    path: str | Path,
    branch: str,
    base_commit: str,
) -> None:
    """Create ``branch`` at ``base_commit`` in a new linked worktree."""
    worktree = Path(path)
    worktree.parent.mkdir(parents=True, exist_ok=True)
    _git(repo, "worktree", "add", "-b", branch, str(worktree), base_commit)


def remove_worktree(repo: str | Path, path: str | Path) -> None:
    """Remove a managed linked worktree after its useful state is integrated."""
    _git(repo, "worktree", "remove", "--force", str(path))


def reset_to(repo: str | Path, commit: str) -> None:
    """Reset a worker branch and its files to the frozen item base."""
    _git(repo, "reset", "--hard", commit)
    _git(repo, "clean", "-fd")


def fast_forward(repo: str | Path, branch: str) -> None:
    """Fast-forward the currently checked-out campaign branch."""
    _git(repo, "merge", "--ff-only", branch)


def commit_all(repo: str | Path, message: str) -> str:
    """Stage every change (``add -A``) and commit; return the new HEAD.

    ``add -A`` respects ``.gitignore``, so build outputs in an editable
    checkout are not swept into the optimization commits.

    The commit runs with ``--no-verify``: these are the orchestrator's
    bookkeeping commits on a private ``perf-optimize/*`` branch, and a
    developer checkout routinely has ``pre-commit`` hooks installed that
    reformat files and exit non-zero (aborting the commit and thereby
    crashing the workflow mid-accept). Style enforcement belongs to the
    user's eventual upstream PR, not to the optimization loop.
    """
    _git(repo, "add", "-A")
    _git(repo, "commit", "--no-verify", "-m", message)
    return rev_parse_head(repo)


def discard_uncommitted(repo: str | Path) -> None:
    """Drop every uncommitted change, tracked and untracked.

    ``clean -fd`` deliberately omits ``-x`` so gitignored files (build
    artifacts, caches) survive the revert — only the rejected attempt's
    edits and any unignored files it added are removed.
    """
    _git(repo, "reset", "--hard")
    _git(repo, "clean", "-fd")
