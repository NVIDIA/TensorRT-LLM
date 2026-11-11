# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""One worktree per task, landed in the checkout when the task finishes.

A task that changes the tree works on its own branch in its own worktree, cut from the
checkout's head when the task starts; its next attempt keeps both. When its worker says
it is ready, the branch's commits are audited against the run's commit rules
(``infra/git.py``): who made them, how they are signed, and that none touches the
workspace or a judge path. The branch is rebased onto the head, held to the task's own
check where it has one (the onboard check for `core`), and fast-forwarded in: the
checkout only moves by whole tasks, carries no merge commit, and holds exactly the commit
that was checked. Every commit that lands is labelled ``target``, ``shared`` or ``mixed``
by whether its paths are the target's own (``commit_labels.json``, for the report).
"""

from __future__ import annotations

import json
import os
import posixpath
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from ..infra.files import write_text_atomic
from ..infra.git import (
    LABEL_TARGET,
    CommitPolicy,
    audit_change,
    audit_commits,
    changed_paths,
    git_checked,
    git_output,
    link_native,
    repo_lock,
)
from ..operators.coverage import INTEGRATION_TESTS_DIR, model_ops_tests_dir

WORKTREES_DIRNAME = "worktrees"
#: Files one checkout writes at once (``checkout.workers``, git 2.32 or later).
CHECKOUT_WORKERS = 16
_CHECKOUT = ("-c", f"checkout.workers={CHECKOUT_WORKERS}")
#: A ``git worktree add`` that failed is cleaned up and tried again after this.
ADD_RETRY_S = 5.0
#: In the workspace: the label of every commit the run landed, for the report.
COMMIT_LABELS_FILENAME = "commit_labels.json"
#: A fast-forward that meets another git process's index lock is tried again.
LOCK_RETRIES, LOCK_RETRY_S = 5, 3.0
#: A finished task whose branch does not integrate as it stands goes back to
#: its worker this many times before the task fails.
REPAIR_TURNS = 1


@dataclass(frozen=True)
class Outcome:
    merged: bool
    detail: str
    #: The checkout's head once the branch is in.
    commit: str = ""
    #: The task's own check failed on the rebased commit: ``detail`` is its result line.
    checked: bool = False
    #: A rebase that stopped: the integration head it was onto, and the files
    #: that conflicted. The worker can resolve it in its worktree.
    base: str = ""
    conflicts: tuple[str, ...] = ()
    #: The worktree is not in a state to integrate (uncommitted changes, not on
    #: its branch): its worker can put it right in a short turn.
    repairable: bool = False


def own_paths(target_rel: str) -> tuple[str, ...]:
    """The target's own paths, as repo-relative prefixes; none without a target.

    Its family directory (its targets and the ``ops/`` they share) and the tests that
    mirror it, the two registries a new family adds a line to (``models/__init__.py``,
    ``_router_index.py``) and the integration tests (the accuracy test, its list entry).
    A commit inside them is the target's; any other path is code other models run.
    """
    target = target_rel.strip().strip("/")
    if not target:
        return ()
    family = posixpath.dirname(target)
    models = posixpath.dirname(family)
    return (
        f"{family}/",
        f"{model_ops_tests_dir(family)}/",
        f"{models}/__init__.py",
        f"{posixpath.dirname(models)}/_router_index.py",
        f"{INTEGRATION_TESTS_DIR}/",
    )


def load_labels(workspace: str | Path) -> list[dict]:
    """The label of every commit the run landed: ``task``, ``sha``, ``subject``, ``label``, ``outside``."""
    try:
        rows = json.loads((Path(workspace) / COMMIT_LABELS_FILENAME).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    return [row for row in rows if isinstance(row, dict)] if isinstance(rows, list) else []


class Integration:
    """The integration checkout and the task worktrees cut from it."""

    def __init__(
        self,
        repo: Path,
        workspace: Path,
        prefix: str,
        env: dict | None = None,
        log: Callable[[str], None] | None = None,
        policy: CommitPolicy | None = None,
    ) -> None:
        self.repo = Path(repo)
        self.root = Path(workspace) / WORKTREES_DIRNAME
        self.prefix = prefix
        #: Identity for the rebase's committer; the author of each commit is kept.
        self.env = env or {}
        #: Told, one line each, how every worktree was made and widened.
        self.log = log or (lambda message: None)
        #: The rules a task's commits are audited against before they are integrated;
        #: ``None``: they are not audited.
        self.policy = policy

    def branch(self, task_id: str) -> str:
        return f"{self.prefix}/{task_id}"

    def tree(self, task_id: str) -> Path:
        return self.root / task_id

    def _git(self, cwd: Path, *args: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            ["git", "-C", str(cwd), *args],
            capture_output=True,
            text=True,
            env={**os.environ, **self.env},
        )

    def open(self, task_id: str) -> Path:
        """The task's worktree, created from the checkout's head the first time.

        Only registering the worktree holds the repository lock. One whose
        checkout was cut short is made again, on the branch an earlier attempt
        left when there is one. A branch that holds nothing the checkout's
        head lacks starts again at that head (:meth:`_catch_up`), so a retry
        sees what other tasks landed since.
        """
        tree = self.tree(task_id)
        if not _checked_out(tree):
            self._create(task_id, tree)
        self._catch_up(task_id, tree)
        link_native(self.repo, tree)
        return tree

    def _catch_up(self, task_id: str, tree: Path) -> None:
        """Move a worktree to the checkout's head when its branch has nothing of its own.

        An earlier attempt that committed nothing, or whose commits all landed, has
        nothing to keep, and no rule lets its worker rebase onto the head.
        """
        ahead = git_output(self.repo, "rev-list", "--count", f"HEAD..{self.branch(task_id)}")
        head = (git_output(self.repo, "rev-parse", "HEAD") or "").strip()
        if (ahead or "").strip() != "0" or not head:
            return
        if (git_output(tree, "rev-parse", "HEAD") or "").strip() == head:
            return
        edits = git_output(tree, "status", "--porcelain", "--untracked-files=no")
        if (edits or "").strip():
            return  # edits an earlier attempt left uncommitted stay where they are
        moved = self._git(tree, *_CHECKOUT, "reset", "--hard", "-q", head)
        if moved.returncode == 0:
            self.log(f"{tree.name}: moved to the integration head {head[:10]}")

    def _create(self, task_id: str, tree: Path) -> None:
        """Register ``tree`` on the task's branch, then check it out whole.

        The branch is an earlier attempt's when one is there.
        """
        if tree.exists():
            self._discard(tree)
        self.root.mkdir(parents=True, exist_ok=True)
        branch = self.branch(task_id)
        with repo_lock(self.repo):
            for attempt in (1, 2):
                exists = git_checked(
                    self.repo, "rev-parse", "--verify", "--quiet", branch, check=False
                )
                where = (str(tree), branch) if exists else ("-b", branch, str(tree), "HEAD")
                added = self._git(self.repo, "worktree", "add", "--no-checkout", *where)
                if added.returncode == 0:
                    break
                self._discard(tree)
                if attempt == 2:
                    raise RuntimeError(
                        f"git worktree add {' '.join(where)}: {added.stderr.strip()}"
                    )
                time.sleep(ADD_RETRY_S)
        git_checked(tree, *_CHECKOUT, "reset", "--hard", "-q")
        self.log(f"{task_id}: worktree {tree} on {branch}")

    def _discard(self, tree: Path) -> None:
        """Remove what a cut-short creation of ``tree`` left: its registration and its files."""
        self._git(self.repo, "worktree", "remove", "--force", "--force", str(tree))
        shutil.rmtree(tree, ignore_errors=True)

    def integrate(self, task_id: str, check: Callable[[Path, str], str] | None = None) -> Outcome:
        """Rebase the task's branch onto the checkout's head and fast-forward it in.

        ``check(tree, commit)`` runs on the rebased, clean worktree before the checkout
        moves; what it returns, when not empty, is why the branch does not land. A branch
        that is in the checkout already (a process stopped between the fast-forward and
        its records) has nothing to integrate, and its check finds its pass on record.
        """
        tree, branch = self.tree(task_id), self.branch(task_id)
        if not (tree / ".git").exists():
            return Outcome(False, f"task {task_id} has no worktree at {tree}")
        if _rebase_in_progress(tree):
            # A rebase cut short, by a process that stopped mid-integration or
            # by a worker, leaves HEAD detached at the base; going back to the
            # branch is what makes its commits count again.
            self._git(tree, "rebase", "--abort")
        on = git_checked(tree, "symbolic-ref", "--quiet", "HEAD", check=False).strip()
        if on != f"refs/heads/{branch}":
            return Outcome(
                False,
                f"the worktree {tree} is not on its branch {branch} "
                f"(HEAD is {on or 'detached'}); check the branch out and finish there",
                repairable=True,
            )
        dirty = git_checked(tree, "status", "--porcelain", "--untracked-files=no", check=False)
        if dirty.strip():
            return Outcome(
                False,
                f"uncommitted changes in {tree}; commit them with `git commit -s` before "
                "finishing:\n" + dirty.strip(),
                repairable=True,
            )
        with repo_lock(self.repo):
            head = git_checked(self.repo, "rev-parse", "HEAD")
            count = ("rev-list", "--count", f"{head}..{branch}")
            ahead = git_checked(tree, *count, check=False).strip() or "0"
            broken = self._audit(tree, head, branch) if ahead != "0" else ""
            if broken:
                return Outcome(False, broken, repairable=True)
            # With nothing of its own, the branch is brought to the head: what its check
            # judges is then the head itself.
            rebased = self._git(tree, "rebase", head)
            if rebased.returncode != 0:
                conflicts = git_checked(
                    tree, "diff", "--name-only", "--diff-filter=U", check=False
                ).split()
                self._git(tree, "rebase", "--abort")
                return Outcome(
                    False,
                    f"branch {branch} conflicts with the integration head {head[:10]} in: "
                    + (", ".join(conflicts) or rebased.stderr.strip()[-400:]),
                    base=head,
                    conflicts=tuple(conflicts),
                )
            tip = git_checked(tree, "rev-parse", "HEAD")
        # Not under the repository's lock: a check takes as long as a benchmark does.
        failed = check(tree, tip) if check else ""
        if failed:
            return Outcome(False, failed, checked=True)
        with repo_lock(self.repo):
            merged = self._git(self.repo, "merge", "--ff-only", "--quiet", branch)
            for _ in range(LOCK_RETRIES):
                # Another git process held the checkout's index lock for a moment.
                if merged.returncode == 0 or "index.lock" not in merged.stderr:
                    break
                time.sleep(LOCK_RETRY_S)
                merged = self._git(self.repo, "merge", "--ff-only", "--quiet", branch)
            if merged.returncode != 0:
                # The checkout moved while the check ran, or git refused: nothing landed.
                return Outcome(False, f"fast-forward of {branch} failed: {merged.stderr.strip()}")
            self._label(task_id, head, branch)
        said = (
            f"integrated {branch} ({ahead} commit(s))" if ahead != "0" else "nothing to integrate"
        )
        return Outcome(True, said, tip)

    def _label(self, task_id: str, head: str, branch: str) -> None:
        """Record each commit the task landed as the target's own, shared code, or both."""
        if self.policy is None:
            return
        out = git_checked(
            self.repo, "log", "--no-merges", "--raw", "-z", "--no-renames",
            "--format=%x1e%H%x1f%s%x1f", f"{head}..{branch}", check=False,
        )  # fmt: skip
        rows = load_labels(self.root.parent)
        for record in filter(str.strip, out.split("\x1e")):
            sha, subject, raw = record.split("\x1f", 2)
            paths = [path for _, path in changed_paths(raw)]
            outside = [p for p in paths if self.policy.label([p]) != LABEL_TARGET]
            rows.append(
                dict(
                    task=task_id,
                    sha=sha,
                    subject=subject,
                    label=self.policy.label(paths),
                    outside=outside,
                )
            )
        write_text_atomic(self.root.parent / COMMIT_LABELS_FILENAME, json.dumps(rows, indent=1))

    def _audit(self, tree: Path, head: str, branch: str) -> str:
        """What in the task's own commits breaks the run's commit rules, said for its worker; "" if nothing."""
        if self.policy is None:
            return ""
        fork = git_checked(tree, "merge-base", head, branch, check=False).strip() or head
        findings = audit_commits(tree, f"{head}..{branch}", self.policy)
        said = {finding.problem for finding in findings}
        # What lands is what is audited: the net change too, which a merge commit's own
        # edits are part of and no single commit shows.
        net = audit_change(tree, fork, branch, self.policy)
        findings += [finding for finding in net if finding.problem not in said]
        if not findings:
            return ""
        return (
            f"commits on {branch} break the run's commit rules:\n"
            + "\n".join(f"  - {f.render()}" for f in findings)
            + f"\nRewrite the commits since {fork[:10]}, for this turn only despite the rule "
            f"against it: `git -C {tree} rebase --signoff {fork}` signs each one off, and "
            f"`git -C {tree} reset --soft {fork}` lets you commit again only what belongs to "
            "the task."
        )

    def close(self, task_id: str) -> None:
        """Remove an integrated task's worktree; its branch stays as the record."""
        tree = self.tree(task_id)
        if tree.exists():
            git_checked(self.repo, "worktree", "remove", "--force", str(tree), check=False)


def _rebase_in_progress(tree: Path) -> bool:
    """Whether ``tree`` holds a rebase that stopped before it finished."""
    for state in ("rebase-merge", "rebase-apply"):
        where = git_checked(tree, "rev-parse", "--git-path", state, check=False).strip()
        if where and (tree / where).exists():
            return True
    return False


def _checked_out(tree: Path) -> bool:
    """Whether ``tree`` is a worktree with every file checked out.

    ``git worktree add --no-checkout`` writes no index, and the checkout
    writes it last.
    """
    if not (tree / ".git").exists():
        return False
    gitdir = git_checked(tree, "rev-parse", "--absolute-git-dir", check=False)
    return bool(gitdir) and Path(gitdir, "index").exists()


def behind(tree: Path, repo: Path) -> tuple[str, str]:
    """``(head, count)``: the checkout's head, and how many of its commits ``tree``'s branch lacks.

    The count is ``""`` when the branch holds the head: a worker is told to rebase only
    when there is something to rebase onto.
    """
    head = (git_output(repo, "rev-parse", "HEAD") or "").strip()
    lacks = ""
    if head and git_output(tree, "merge-base", "--is-ancestor", head, "HEAD") is None:
        lacks = (git_output(tree, "rev-list", "--count", f"HEAD..{head}") or "").strip()
    return head, lacks if lacks.isdigit() and lacks != "0" else ""
