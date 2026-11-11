# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""One worktree per task, integrated into the checkout when the task finishes.

A task that changes the tree (a kernel entry, a deployment, a fix) works
on its own branch in its own worktree, cut from the checkout's head when the
task starts. When its worker finishes, the branch's commits are audited
against the run's commit rules (``infra/git.py``): who made them, how they are
signed, and that they change only the run's own paths. A branch that breaks
them goes back to its worker. The branch is then rebased onto the head as it
is then and fast-forwarded in, under one lock: the checkout only ever moves by
whole tasks and carries no merge commits. Tasks run one at a time but for
kernel entries, so a rebase rarely conflicts; one that does is aborted and
goes back to its worker once (:func:`rebase_block`).

A task's next attempt keeps its worktree and branch.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from ..infra.git import CommitPolicy, audit_commits, git_checked, git_output, link_native, repo_lock

WORKTREES_DIRNAME = "worktrees"
#: Files one checkout writes at once (``checkout.workers``, git 2.32 or later).
CHECKOUT_WORKERS = 16
_CHECKOUT = ("-c", f"checkout.workers={CHECKOUT_WORKERS}")
#: A ``git worktree add`` that failed is cleaned up and tried again after this.
ADD_RETRY_S = 5.0
#: Each integrated task's branch tip, so a task whose integration a restart
#: interrupted is not integrated a second time.
INTEGRATED_FILENAME = "integrated.json"
#: A fast-forward that meets another git process's index lock is tried again.
LOCK_RETRIES, LOCK_RETRY_S = 5, 3.0
#: A finished task whose branch does not integrate as it stands goes back to
#: its worker this many times before the task fails.
REPAIR_TURNS = 1


class TaskNotStarted(RuntimeError):
    """A task's worktree could not be prepared, so no worker ran: the attempt is not the worker's."""


@dataclass(frozen=True)
class Outcome:
    merged: bool
    detail: str
    commits: int = 0
    #: A rebase that stopped: the integration head it was onto, and the files
    #: that conflicted. The worker can resolve it in its worktree.
    base: str = ""
    conflicts: tuple[str, ...] = ()
    #: The worktree is not in a state to integrate (uncommitted changes, not on
    #: its branch): its worker can put it right in a short turn.
    repairable: bool = False


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
        self._lock = threading.Lock()

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
        head lacks starts again at that head (:meth:`_caught_up`), so a retry
        sees what other tasks landed since.
        """
        tree = self.tree(task_id)
        if _checked_out(tree):
            self._catch_up(task_id, tree)
        else:
            self._create(task_id, tree)
        link_native(self.repo, tree)
        return tree

    def _caught_up(self, task_id: str) -> str | None:
        """The integration head, when the task's branch holds no commit it lacks; else ``None``.

        An earlier attempt that committed nothing, or whose commits all landed,
        has nothing to keep, and no rule lets its worker rebase onto the head.
        """
        ahead = git_output(self.repo, "rev-list", "--count", f"HEAD..{self.branch(task_id)}")
        if (ahead or "").strip() != "0":
            return None
        return (git_output(self.repo, "rev-parse", "HEAD") or "").strip() or None

    def _catch_up(self, task_id: str, tree: Path) -> None:
        """Move a reopened worktree to the integration head when its branch has nothing else."""
        head = self._caught_up(task_id)
        if head is None or (git_output(tree, "rev-parse", "HEAD") or "").strip() == head:
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
                head = self._caught_up(task_id) if exists else None
                if head is not None:
                    self._git(self.repo, "branch", "-f", branch, head)
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

    def _integrated(self) -> dict[str, str]:
        path = self.root / INTEGRATED_FILENAME
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        return record if isinstance(record, dict) else {}

    def integrated_tip(self, task_id: str) -> str:
        """The branch tip last integrated for ``task_id``, or empty when none is recorded."""
        return str(self._integrated().get(task_id) or "")

    def _record(self, task_id: str, tip: str) -> None:
        record = {**self._integrated(), task_id: tip}
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / INTEGRATED_FILENAME).write_text(json.dumps(record, indent=1), encoding="utf-8")

    def _already_in(self, task_id: str) -> bool:
        """The task's branch went in before, and is in the checkout as it was then.

        A process that dies between a task's fast-forward and its checkpoint
        resumes the task, and its worker's decision is read again; the branch
        must not be integrated twice, from a worktree half removed.
        """
        tip = self.integrated_tip(task_id)
        if not tip:
            return False
        current = git_checked(
            self.repo, "rev-parse", "--verify", "--quiet", self.branch(task_id), check=False
        )
        if current.strip() != tip:
            return False
        return self._git(self.repo, "merge-base", "--is-ancestor", tip, "HEAD").returncode == 0

    def integrate(self, task_id: str) -> Outcome:
        """Rebase the task's branch onto the checkout's head and fast-forward it in."""
        tree, branch = self.tree(task_id), self.branch(task_id)
        if self._already_in(task_id):
            return Outcome(True, f"{branch} was integrated before the restart", 0)
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
        with self._lock, repo_lock(self.repo):
            head = git_checked(self.repo, "rev-parse", "HEAD")
            ahead = git_checked(
                tree, "rev-list", "--count", f"{head}..{branch}", check=False
            ).strip()
            if ahead in ("", "0"):
                return Outcome(True, "nothing to integrate", 0)
            broken = self._audit(tree, head, branch)
            if broken:
                return Outcome(False, broken, repairable=True)
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
            merged = self._git(self.repo, "merge", "--ff-only", "--quiet", branch)
            for _ in range(LOCK_RETRIES):
                # Another git process held the checkout's index lock for a moment.
                if merged.returncode == 0 or "index.lock" not in merged.stderr:
                    break
                time.sleep(LOCK_RETRY_S)
                merged = self._git(self.repo, "merge", "--ff-only", "--quiet", branch)
            if merged.returncode != 0:
                return Outcome(False, f"fast-forward of {branch} failed: {merged.stderr.strip()}")
            self._record(task_id, git_checked(self.repo, "rev-parse", branch))
            return Outcome(True, f"integrated {branch} ({ahead} commit(s))", int(ahead))

    def _audit(self, tree: Path, head: str, branch: str) -> str:
        """What in the task's own commits breaks the run's commit rules, said for its worker; "" if nothing."""
        if self.policy is None:
            return ""
        findings = audit_commits(tree, f"{head}..{branch}", self.policy, own=True)
        if not findings:
            return ""
        fork = git_checked(tree, "merge-base", head, branch, check=False).strip() or head
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


def _behind_note(tree: Path, repo: Path) -> str:
    """What a worker whose branch lacks the checkout's head is told; "" when it holds it."""
    head = (git_output(repo, "rev-parse", "HEAD") or "").strip()
    if not head or git_output(tree, "merge-base", "--is-ancestor", head, "HEAD") is not None:
        return ""
    behind = (git_output(tree, "rev-list", "--count", f"HEAD..{head}") or "").strip()
    if not behind.isdigit() or behind == "0":
        return ""
    return (
        f"\n\n## Your branch is behind the checkout\n\n{behind} commit(s) landed on the checkout "
        f"after your branch was cut; its head is `{head}`. Rebase onto that head before anything "
        "else, for this turn only despite the rule against it:\n\n"
        f"    git -C {tree} rebase {head}\n\n"
        "Resolve each conflict so that both your change and the landed one stand, then "
        f"`git -C {tree} add` the resolved files and `GIT_EDITOR=true git -C {tree} rebase "
        f"--continue` until the rebase is done. If the two cannot both stand, `git -C {tree} "
        "rebase --abort` and say why in your decision file."
    )


def worktree_block(tree: Path, repo: Path) -> str:
    """What a worker in its own worktree is told, appended to its turn prompt."""
    return (
        "## Your worktree\n\n"
        f"You work in `{tree}`, your own branch of the checkout `{repo}`. Edit, build, test and "
        f"commit there and nowhere else; do not touch `{repo}`. Read paths relative to your "
        "worktree where the prompt names the checkout. Commit with\n\n"
        f"    git -C {tree} add <path> [path...] && git -C {tree} commit -s -m <message>\n\n"
        f"Run a check on your work with `--tree {tree}`. When you finish, the orchestrator "
        "audits your commits and rebases them onto the checkout; uncommitted changes are not "
        "integrated and fail the task." + _behind_note(tree, repo)
    )


def repair_block(outcome: Outcome) -> str:
    """What a worker whose finished worktree could not be integrated as it stands is told."""
    return (
        "## Your finished work could not be integrated as it stands\n\n"
        f"{outcome.detail}\n\n"
        "Do not start the task again. Put the worktree right (commit what belongs to the task "
        "with `git commit -s`, drop what does not, and leave the worktree on its branch), "
        "then write your decision as you did when you finished."
    )


def rebase_block(outcome: Outcome, tree: Path) -> str:
    """What a worker whose finished branch did not rebase is told, after its worktree block."""
    files = ", ".join(f"`{f}`" for f in outcome.conflicts) or "files git did not name"
    return (
        "## Your finished work did not integrate\n\n"
        "Your task already finished, but rebasing its branch onto the checkout's head "
        f"`{outcome.base}` stopped on conflicts in {files}: another task changed them after "
        "your worktree was cut. Do not start the task again. In your worktree, and for this "
        "turn only, rebase despite the rule against it:\n\n"
        f"    git -C {tree} rebase {outcome.base}\n\n"
        "Resolve each conflict so that both your change and the other task's stand, then "
        f"`git -C {tree} add` the resolved files and `GIT_EDITOR=true git -C {tree} rebase "
        "--continue` until the rebase is done; the commits keep their messages and sign-offs. "
        "Run your task's tests again on the rebased tree, commit any fix with `git commit -s`, "
        "and write your decision as you did when you finished. If the two changes "
        f"cannot both stand, `git -C {tree} rebase --abort` and end `BLOCKED`, naming the "
        "files and why."
    )


__all__ = [
    "CHECKOUT_WORKERS",
    "Integration",
    "Outcome",
    "REPAIR_TURNS",
    "TaskNotStarted",
    "rebase_block",
    "repair_block",
    "worktree_block",
]
