# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""One worktree per task, integrated into the checkout when the task finishes.

Tasks that change the tree (``kernel``, ``assemble``, ``feature``,
``integrate``) each work on their own branch in their own worktree, cut from
the integration checkout's head when the task starts. When the worker
finishes, the branch's commits are audited against the run's commit rules
(``infra/git.py``), and one that breaks them goes back to the worker.
The branch is then rebased onto the head as it is then and fast-forwarded in,
under one lock: the integration branch only ever moves by
whole tasks, never by a half-written file, and it carries no merge commits. A
rebase that conflicts is aborted and reported with the conflicting files and
the head it was onto; the worker is then asked, in its worktree, to rebase onto
that head and resolve (:func:`rebase_block`), and the task fails only when that
does not work.

A task's next attempt keeps its worktree and branch; its worker is told how
far the branch is behind the checkout, and rebases when it needs what landed
(:func:`worktree_block`).

A worktree is a sparse checkout (:func:`cone`): the Python package, the tests
and examples a worker runs, and the directories its task ``touches``. A task
that touches native code, or runs ``exclusive``, gets the whole tree, and a
worker widens its own with ``git sparse-checkout add``. The files left out are
still in the branch: commits, the rebase and the fast-forward carry them as
they are.

No config file says a worktree is sparse. Its cone is the cone file in its
own git directory and the skip-worktree entries of its index, and each git
command run here on it carries :data:`SPARSE_ON`, which turns sparse checkout
on for that command alone: ``git sparse-checkout set --cone`` in a linked
worktree would first turn on per-worktree config in the config every worktree
of the repository shares, and fails where that file is read-only or locked.

An ``investigate`` task gets a scratch worktree that is never integrated.
"""

from __future__ import annotations

import json
import os
import posixpath
import re
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Callable, Iterable, Mapping

from ...infra.git import (
    CommitPolicy,
    audit_commits,
    git_checked,
    git_output,
    link_native,
    repo_lock,
)

#: Kinds whose work is integrated, and the kind that gets a throwaway tree.
INTEGRATED_KINDS = frozenset({"kernel", "assemble", "feature", "integrate"})
SCRATCH_KINDS = frozenset({"investigate"})
WORKTREES_DIRNAME = "worktrees"
#: What every sparse worktree checks out: the package, the tests a worker runs
#: and what they import, and the examples a gate runs and reads (the
#: quickstart, a prompt's images) from the tree it measures. Cone mode adds
#: the files directly in each one's parent directories and at the top level.
SPARSE_CONE: tuple[str, ...] = (
    "examples",
    "tensorrt_llm",
    "tests/integration",
    "tests/test_common",
    "tests/unittest/_torch",
    "tests/unittest/llmapi/apps",
    "tests/unittest/utils",
    "triton_kernels",
)
#: A task that touches these changes native code, whose build needs the whole tree.
FULL_TREE_ROOTS = frozenset({"cpp", "3rdparty"})
#: Turns sparse checkout on for one git command in a worktree, without a config
#: file: git then reads and writes the cone file in the worktree's own git
#: directory and the skip-worktree bits of its index.
SPARSE_ON = ("-c", "core.sparseCheckout=true", "-c", "core.sparseCheckoutCone=true")
#: The same for a pattern rather than a cone: ``/*`` brings every file back.
_PATTERNS_ON = ("-c", "core.sparseCheckout=true", "-c", "core.sparseCheckoutCone=false")
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
#: A finished task whose rebase conflicts goes back to its worker to resolve
#: it this many times before the task fails.
REBASE_RETRIES = 2


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

    def open(self, task_id: str, touches: Iterable[str] = (), full: bool = False) -> Path:
        """The task's worktree, created from the integration head the first time.

        It checks out :func:`cone` of ``touches``, or the whole tree when
        ``full`` or a touched path asks for it. Only registering the worktree
        holds the repository lock. A worktree opened again is widened to what
        ``touches`` asks now; one whose checkout was cut short is made again,
        on the branch an earlier attempt left when there is one. A branch that holds
        nothing the integration head lacks starts again at that head
        (:meth:`_caught_up`), so a retry sees what other tasks landed since.
        """
        tree, dirs = self.tree(task_id), cone(touches, full)
        if _checked_out(tree):
            self._widen(tree, dirs)
            self._catch_up(task_id, tree)
        else:
            self._create(task_id, tree, dirs)
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
        flags = _sparse_flags(tree)
        edits = git_output(tree, *flags, "status", "--porcelain", "--untracked-files=no")
        if (edits or "").strip():
            return  # edits an earlier attempt left uncommitted stay where they are
        moved = self._git(tree, *_CHECKOUT, *flags, "reset", "--hard", "-q", head)
        if moved.returncode == 0:
            self.log(f"{tree.name}: moved to the integration head {head[:10]}")

    def changed_paths(self, task_id: str) -> tuple[str, ...]:
        """The files the task's branch changes since it left the integration head's history."""
        listed = git_output(self.repo, "diff", "--name-only", f"HEAD...{self.branch(task_id)}")
        return tuple(sorted({line.strip() for line in (listed or "").splitlines() if line.strip()}))

    def _widen(self, tree: Path, dirs: tuple[str, ...] | None) -> None:
        """Check out in ``tree`` what ``dirs`` has beyond its cone; the whole tree for None.

        A tree whose own config makes it sparse (``git sparse-checkout set
        --cone`` where the shared config can be written) is widened through
        that config.
        """
        flags, current = _sparse(tree)
        if not current:
            return
        why = "its task now asks for the whole tree"
        if dirs is not None:
            missing = [d for d in dirs if not any(_under(d, c) for c in current)]
            if not missing:
                return
            added = self._git(
                tree, *_CHECKOUT, *flags, "sparse-checkout", "add", "--skip-checks", *missing
            )
            if added.returncode == 0:
                self.log(f"{tree.name}: worktree widened to {', '.join(missing)} {_way(flags)}")
                return
            why = f"git could not widen it: {_last_line(added.stderr)}"
        # The whole tree, also where git could not widen the cone.
        self.log(f"{tree.name}: {self._whole(tree, flags)} ({why})")

    def _whole(self, tree: Path, flags: tuple[str, ...]) -> str:
        """Check out the rest of the sparse ``tree``, leaving it whole; say what happened.

        A file the tree has on disk outside its cone is left as it is.
        """
        if not flags:
            done = self._git(tree, *_CHECKOUT, "sparse-checkout", "disable")
        else:
            # A pattern for every file checks the rest out; the cone file then
            # goes, so that git reads the tree as whole.
            done = self._git(
                tree, *_CHECKOUT, *_PATTERNS_ON, "sparse-checkout", "set", "--skip-checks", "/*"
            )
            if done.returncode == 0:
                _drop_cone(tree)
        if done.returncode != 0:
            error = _last_line(done.stderr)
            return f"worktree left sparse, git could not check out the rest: {error}"
        return f"worktree made whole {_way(flags)}"

    def _create(self, task_id: str, tree: Path, dirs: tuple[str, ...] | None) -> None:
        """Register ``tree`` on the task's branch, write its cone, then check it out.

        The tree is whole when ``dirs`` is None, and where git cannot make it
        sparse. The branch is an earlier attempt's when one is there.
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
        sparse, how = (), "worktree made whole, as its task asks"
        if dirs is not None:
            # Writes the cone file in the worktree's own git directory only, so
            # it runs beside other tasks doing the same.
            made = self._git(tree, *SPARSE_ON, "sparse-checkout", "set", "--skip-checks", *dirs)
            if made.returncode == 0:
                sparse = SPARSE_ON
                how = f"worktree made sparse {_way(sparse)}: {', '.join(dirs)}"
            else:
                _drop_cone(tree)
                error = _last_line(made.stderr)
                how = f"worktree made whole: git could not make it sparse ({error})"
        git_checked(tree, *_CHECKOUT, *sparse, "reset", "--hard", "-q")
        self.log(f"{task_id}: {how}")

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
        """Rebase the task's branch onto the integration head and fast-forward it in.

        The rebase keeps a sparse worktree sparse: what the head changed
        outside its cone stays off disk.
        """
        tree, branch = self.tree(task_id), self.branch(task_id)
        if self._already_in(task_id):
            return Outcome(True, f"{branch} was integrated before the restart", 0)
        if not (tree / ".git").exists():
            return Outcome(False, f"task {task_id} has no worktree at {tree}")
        if _rebase_in_progress(tree):
            # A rebase cut short, by a process that stopped mid-integration or
            # by a worker, leaves HEAD detached at the base; going back to the
            # branch is what makes its commits count again.
            self._git(tree, *_sparse_flags(tree), "rebase", "--abort")
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
            flags = _sparse_flags(tree)
            rebased = self._git(tree, *flags, "rebase", head)
            if rebased.returncode != 0:
                conflicts = git_checked(
                    tree, "diff", "--name-only", "--diff-filter=U", check=False
                ).split()
                self._git(tree, *flags, "rebase", "--abort")
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


def _under(path: str, root: str) -> bool:
    return path == root or path.startswith(root + "/")


def _touched_dir(path: str) -> str:
    """The directory a ``touches`` entry needs on disk, repo-relative; "" for the top level.

    A file (a last part with a suffix, as the graph reads ``touches``) needs
    the directory holding it, and a glob the directory it starts in. A path
    that leaves the repository needs nothing.
    """
    path = path.strip()
    glob = re.search(r"[*?\[]", path)
    if glob:
        path = path[: glob.start()].rpartition("/")[0] + "/"
    clean = posixpath.normpath(path) if path.strip("/") else "."
    if clean == "." or clean.startswith(("/", "..")):
        return ""
    if not path.endswith("/") and PurePosixPath(clean).suffix:
        clean = posixpath.dirname(clean)
    return clean


def cone(touches: Iterable[str] = (), full: bool = False) -> tuple[str, ...] | None:
    """The directories a task's worktree checks out, or None for the whole tree.

    :data:`SPARSE_CONE` and the directories ``touches`` names. ``full``, or a
    touched path under :data:`FULL_TREE_ROOTS`, asks for the whole tree.
    """
    dirs = {_touched_dir(path) for path in touches} - {""}
    if full or any(d.split("/")[0] in FULL_TREE_ROOTS for d in dirs):
        return None
    dirs |= set(SPARSE_CONE)
    return tuple(sorted(d for d in dirs if not any(o != d and _under(d, o) for o in dirs)))


def _sparse(tree: Path) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """What turns sparse checkout on in ``tree`` (:func:`_sparse_flags`), and its cone."""
    flags = _sparse_flags(tree)
    listed = git_output(tree, *flags, "sparse-checkout", "list") or ""
    return flags, tuple(line.strip() for line in listed.splitlines() if line.strip())


def _rebase_in_progress(tree: Path) -> bool:
    """Whether ``tree`` holds a rebase that stopped before it finished."""
    for state in ("rebase-merge", "rebase-apply"):
        where = git_checked(tree, "rev-parse", "--git-path", state, check=False).strip()
        if where and (tree / where).exists():
            return True
    return False


def _sparse_flags(tree: Path) -> tuple[str, ...]:
    """The flags that turn sparse checkout on for one git command in ``tree``.

    No flags for a tree whose own config says whether it is sparse, as one
    made sparse through ``git sparse-checkout set --cone`` does, and
    :data:`SPARSE_ON` for any other. Without a cone file, as in a whole tree,
    they change nothing.
    """
    said = git_output(tree, "config", "--show-scope", "--get", "core.sparseCheckout") or ""
    return () if said.startswith("worktree") else SPARSE_ON


def _drop_cone(tree: Path) -> None:
    """Remove ``tree``'s cone file, which git keeps in the worktree's own git directory."""
    path = git_checked(tree, "rev-parse", "--git-path", "info/sparse-checkout", check=False)
    if path:
        Path(tree, path).unlink(missing_ok=True)


def _way(flags: tuple[str, ...]) -> str:
    return "without a config" if flags else "through its config"


def _last_line(text: str) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[-1] if lines else "no message"


def _checked_out(tree: Path) -> bool:
    """Whether ``tree`` is a worktree with every file checked out.

    ``git worktree add --no-checkout`` writes no index, and the checkout
    writes it last.
    """
    if not (tree / ".git").exists():
        return False
    gitdir = git_checked(tree, "rev-parse", "--absolute-git-dir", check=False)
    return bool(gitdir) and Path(gitdir, "index").exists()


def _sparse_note(tree: Path) -> str:
    """How a worker reaches what its sparse worktree leaves out; "" for a whole tree."""
    flags, dirs = _sparse(tree)
    if not dirs:
        return ""
    listed = ", ".join(f"`{d}/`" for d in dirs)
    widen = " ".join(("git", "-C", str(tree), *flags, "sparse-checkout", "add", "<dir>"))
    note = (
        f"\n\nOnly part of the checkout is on disk in your worktree: {listed}, and the files "
        "at the top level and directly in those directories' parents. Everything else is in "
        "your branch, unchanged by what you commit: read a file of it with "
        f"`git -C {tree} show HEAD:<path>`, search it with "
        f"`git -C {tree} grep <pattern> HEAD -- <dir>`, and before you edit or build in a "
        "directory that is not on disk, check it out with\n\n"
        f"    {widen}"
    )
    if flags:
        note += (
            "\n\nKeep the `-c` options: no config file marks your worktree sparse, and without "
            "them git answers `no sparse-checkout to add to`."
        )
    return note


def _behind_note(tree: Path, repo: Path, scratch: bool) -> str:
    """What a worker whose branch lacks the checkout's head is told; "" when it holds it."""
    head = (git_output(repo, "rev-parse", "HEAD") or "").strip()
    if not head or git_output(tree, "merge-base", "--is-ancestor", head, "HEAD") is not None:
        return ""
    behind = (git_output(tree, "rev-list", "--count", f"HEAD..{head}") or "").strip()
    if not behind.isdigit() or behind == "0":
        return ""
    landed = (
        f"{behind} commit(s) landed on the checkout after your branch was cut or last rebased; "
        f"its head is `{head}`."
    )
    if not scratch:
        landed += " They reach your work when it is integrated, after your tests have run."
    return (
        f"\n\n## Your branch is behind the checkout\n\n{landed} If your task needs them (a fix "
        "it depends on, say), rebase onto that head before anything else, for this turn only "
        "despite the rule against it:\n\n"
        f"    git -C {tree} rebase {head}\n\n"
        "Resolve each conflict so that both your change and the landed one stand, then "
        f"`git -C {tree} add` the resolved files and `GIT_EDITOR=true git -C {tree} rebase "
        f"--continue` until the rebase is done. If the two cannot both stand, `git -C {tree} "
        "rebase --abort` and say why in your report."
    )


def worktree_block(tree: Path, repo: Path, scratch: bool) -> str:
    """What a worker in its own worktree is told, appended to its turn prompt."""
    if scratch:
        return (
            "## Your worktree\n\n"
            f"Experiment in `{tree}`, a scratch copy of `{repo}`. Nothing in it is integrated: "
            "write what you find to your report and decision file, and leave the checkout alone."
            + _sparse_note(tree)
            + _behind_note(tree, repo, scratch=True)
        )
    return (
        "## Your worktree\n\n"
        f"You work in `{tree}`, your own branch of the checkout `{repo}`. Edit, build, test and "
        f"commit there and nowhere else; do not touch `{repo}`. Read paths relative to your "
        "worktree where the prompt names the checkout. Commit with\n\n"
        f"    git -C {tree} add <path> [path...] && git -C {tree} commit -s -m <message>\n\n"
        f"Run a gate on your work with `--tree {tree}`. When you finish, the orchestrator "
        "checks your commits, rebases them onto the checkout and runs your task's checks "
        "there; uncommitted changes are not integrated and fail the task."
        + _sparse_note(tree)
        + _behind_note(tree, repo, scratch=False)
    )


def soft_dependency_block(states: Mapping[str, str], receipts: Path, repo: str) -> str:
    """What a task that starts beside its soft dependencies is told; "" with none.

    ``states`` is each one's status as the task starts; ``receipts`` the
    directory their receipts land in; ``repo`` the checkout.
    """
    if not states:
        return ""
    lines = []
    for dep, status in states.items():
        receipt = receipts / f"{dep}.receipt.yaml"
        if status == "done":
            lines.append(f"- `{dep}` has landed; its receipt is `{receipt}`.")
        elif status in ("failed", "abandoned", "split"):
            lines.append(
                f"- `{dep}` ended {status}: its change will not land. Go on without it, and say "
                "in your report what of your work needed it."
            )
        else:
            lines.append(
                f"- `{dep}` is {status}: its change lands while you work, and its receipt "
                f"`{receipt}` appears when it has."
            )
    head = f"$(git -C {repo} rev-parse HEAD)" if repo else "<the checkout's head>"
    return (
        "## Soft dependencies\n\n"
        "Your task starts beside these, and needs their change only where its checks run:\n\n"
        + "\n".join(lines)
        + "\n\nWrite your change against the code as it is now. When you finish, your "
        "integration waits for each of them to land or end; your branch is then rebased onto "
        "the checkout's head, which holds what landed, and your task's checks run there. If "
        "your own tests need a change that has landed meanwhile, rebase onto the checkout's "
        f"head first, for this turn only despite the rule against it: `git -C <your worktree> "
        f"rebase {head}`."
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
    "FULL_TREE_ROOTS",
    "INTEGRATED_KINDS",
    "Integration",
    "Outcome",
    "REBASE_RETRIES",
    "SCRATCH_KINDS",
    "SPARSE_CONE",
    "SPARSE_ON",
    "TaskNotStarted",
    "cone",
    "rebase_block",
    "repair_block",
    "soft_dependency_block",
    "worktree_block",
]
