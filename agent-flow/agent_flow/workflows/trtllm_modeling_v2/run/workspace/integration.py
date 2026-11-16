# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The run's trees: one whole git tree per task, landed by rebase and fast-forward.

The run asks four things of its trees: ``open(task)`` a writable tree at the run's head,
``pin(task)`` commit what is there and return the pin, ``at(pin)`` that pin as a read-only
tree, ``land(task)`` audit, rebase and fast-forward under one lock. **A job runs a pin, never
a moving tree.** A pin (:class:`Pin`) is a source commit plus a build record, the content
digest of each native library: a commit does not carry the libraries git ignores. There is
one kind of tree, made here and by ``infra/compute/pintree.py`` on a job's node.

WHERE. Trees are on the node's own disk: a task's tree on the controller's host, a pin's tree
on the node its job runs on, under one path that is the same on every node
(``pintree.local_root``; ``environment.trees.root`` states another place, which may be one
every node shares). A tree is a whole checkout with a git directory of its own
(``pintree.make_tree``), which the repository is told nothing of. So the storage every node
shares holds the repository's objects and refs, the pin files and the run's few records
(``<workspace>/trees``), and no file of any tree.

``open``. Tasks that change the tree (``kernel``, ``assemble``, ``feature``, ``integrate``)
each work on their own branch in their own tree, cut from the checkout's head when the task
starts. A task's next attempt keeps its tree and branch; its worker is told how far the branch
is behind the checkout, and rebases when it needs what landed (:func:`worktree_block`). An
``investigate`` task gets a scratch tree that never lands. A tree that is gone with its disk
(the controller moved to another node) is made again on its branch, the uncommitted edits of
the task's last pin are put back, and the worker is told what that leaves lost
(:meth:`Integration.notice`).

``pin``. What a task's tree holds, uncommitted edits included, becomes a commit no branch
moves, with the build record of the libraries the tree links; with no task, the checkout's
head is pinned. ``at`` is that commit as a detached, read-only tree under ``pins/`` on this
node, made once and used again while it is there, held by whoever reads it; it is removed
when nobody holds it and it went unused for ``pin_age_s``.

``land``. When the worker finishes, the branch's commits are audited against the run's commit
rules (``infra/git.py``), and one that breaks them goes back to the worker. The branch is then
rebased onto the head as it is then and fast-forwarded in, under one lock: the head only ever
moves by whole tasks, never by a half-written file, and it carries no merge commits. A rebase
that conflicts is aborted and reported with the conflicting files and the head it was onto;
the worker is then asked, in its tree, to rebase onto that head and resolve
(:func:`rebase_block`), and the task fails only when that does not work.

ONE BUILD, NOT REBUILT. Every tree links the checkout's native build, and nothing inside a
run rebuilds it: each pin records that build and each job is checked against the record.
Work that needs compiled code changed is not a run's to finish. :data:`NATIVE_BASE` says so
once, to a worker in its turn's prompt and in what a landing answers a branch that changed
native source.
"""

from __future__ import annotations

import contextlib
import json
import os
import socket
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Iterator, Mapping

from ...infra.compute import pintree
from ...infra.files import read_json, write_json_atomic
from ...infra.git import CommitPolicy, audit_commits, git_checked, git_output

#: Kinds whose work lands, and the kind that gets a throwaway tree.
#: The kinds whose commits land on the target: every worker's but the survey's.
INTEGRATED_KINDS = frozenset({"kernel", "weights", "routing", "state", "feature", "assemble"})
SCRATCH_KINDS: frozenset[str] = frozenset()
WORKTREES_DIRNAME = "worktrees"
#: Under the workspace: what every node must see of the run's trees.
TREES_DIRNAME = "trees"
#: Under that: per open task, the host its tree is on.
OPEN_DIRNAME = "open"
#: Files one checkout writes at once (``checkout.workers``, git 2.32 or later).
CHECKOUT_WORKERS = pintree.CHECKOUT_WORKERS
_CHECKOUT = ("-c", f"checkout.workers={CHECKOUT_WORKERS}")
#: Each landed task's branch tip, so a task whose landing a restart
#: interrupted is not landed a second time.
INTEGRATED_FILENAME = pintree.INTEGRATED_FILENAME
#: A fast-forward that meets another git process's index lock is tried again.
LOCK_RETRIES, LOCK_RETRY_S = 5, 3.0
#: A finished task whose rebase conflicts goes back to its worker to resolve
#: it this many times before the task fails.
REBASE_RETRIES = 2
#: A checkout that took at least this long wrote enough files in the index's own second for
#: the index to be worth settling (``pintree.settle_index``, up to a second); a small tree is not.
SETTLE_MIN_S = 0.3
#: What makes a file native source: its language, by suffix, or a build description by name.
NATIVE_SUFFIXES: tuple[str, ...] = (
    ".c", ".cc", ".cpp", ".cxx", ".cu", ".cuh", ".h", ".hpp", ".hxx", ".inl", ".cmake",
)  # fmt: skip
BUILD_DESCRIPTIONS: tuple[str, ...] = ("CMakeLists.txt",)
#: What a worker and the session that watches the run are told of compiled code, once: in
#: each turn's prompt (:func:`worktree_block`) and in a landing's answer (:meth:`Integration.land`).
NATIVE_BASE = (
    "Every tree of this run links one native build, the one the run started from, and every "
    "check measures it. A file's date says nothing about it: the libraries keep the dates of "
    "their build and a tree's sources the date the tree was made, and no task's change to "
    "native code lands. Nothing inside a run rebuilds it, so native code a task changed would "
    "land unbuilt and untested. Work that cannot be done without a change to native code (C++ "
    "or CUDA sources, build descriptions) therefore ends `blocked`, with what must change and "
    "why as its question: the session that watches the run rebuilds the base, and the run "
    "goes on from there."
)


#: The spec's block under ``environment``: where trees go.
TREES_FIELD = "trees"
TREES_KEYS: tuple[str, ...] = ("root",)


class TaskNotStarted(RuntimeError):
    """A task's tree could not be prepared, so no worker ran: the attempt is not the worker's."""


@dataclass(frozen=True)
class Pin:
    """A source commit and the build record of the libraries that go with it."""

    commit: str
    #: ``{library path in the tree: sha256 of its bytes}``.
    build: Mapping[str, str] = field(default_factory=dict)
    #: The pin as a file every node sees: what a job's header names.
    file: Path | None = None

    @property
    def id(self) -> str:
        return pintree.pin_id(self.commit, dict(self.build))


@dataclass(frozen=True)
class Outcome:
    """What landing a task's branch came to."""

    merged: bool
    detail: str
    commits: int = 0
    #: A rebase that stopped: the head it was onto, and the files that
    #: conflicted. The worker can resolve it in its tree.
    base: str = ""
    conflicts: tuple[str, ...] = ()
    #: The tree is not in a state to land (uncommitted changes, not on its
    #: branch): its worker can put it right in a short turn.
    repairable: bool = False


def validate_trees_block(block: Any, errors: list[str], *, where: str) -> None:
    """Validate the spec's optional ``environment.trees`` block; problems go to ``errors``."""
    if block is None:
        return
    if not isinstance(block, dict):
        errors.append(f"'{where}' must be a mapping, got {type(block).__name__}")
        return
    unknown = sorted(set(block) - set(TREES_KEYS))
    if unknown:
        errors.append(f"'{where}' has unknown key(s) {unknown}; known keys are {list(TREES_KEYS)}")
    root = block.get("root")
    if root is not None and (not isinstance(root, str) or not os.path.isabs(root)):
        errors.append(f"'{where}.root' must be an absolute path: where the run's trees go")


def place(task_data: Mapping[str, Any], workspace: Path) -> Path:
    """Where the run's trees are on this node: the place the spec states, else its own disk.

    An agent that works in a task's tree must be allowed to write there.
    """
    block = (task_data.get("environment") or {}).get(TREES_FIELD) or {}
    return Path(str(block["root"])) if block.get("root") else Path(pintree.local_root(workspace))


def make(
    task_data: Mapping[str, Any],
    *,
    repo: Path,
    workspace: Path,
    prefix: str,
    env: Mapping[str, str] | None = None,
    log: Callable[[str], None] | None = None,
    policy: CommitPolicy | None = None,
) -> "Integration":
    """The run's trees, from the spec: the checkout, the workspace and the place."""
    environment = task_data.get("environment") or {}
    block = environment.get(TREES_FIELD) or {}
    return Integration(
        repo=Path(repo),
        workspace=Path(workspace),
        root=Path(block["root"]) if block.get("root") else None,
        prefix=prefix,
        env=dict(env or {}),
        log=log,
        policy=policy,
    )


def _made(tree: Path) -> tuple[int, int] | None:
    """Which making of a pin's tree stands at ``tree`` (its marker is written last), if any."""
    try:
        found = os.stat(tree / pintree.PIN_MARKER)
    except OSError:
        return None
    return found.st_ino, found.st_mtime_ns


class Integration:
    """The checkout and the trees cut from it."""

    def __init__(
        self,
        repo: Path,
        workspace: Path,
        prefix: str,
        env: dict | None = None,
        log: Callable[[str], None] | None = None,
        policy: CommitPolicy | None = None,
        root: Path | None = None,
        pin_age_s: float | None = None,
    ) -> None:
        self.repo = Path(repo)
        #: What every node must see of the run's trees: pin files and the run's few records.
        self.shared = Path(workspace) / TREES_DIRNAME
        #: Where the trees themselves go: the place the spec states, else this node's own disk.
        self.place = Path(root) if root else Path(pintree.local_root(workspace))
        self.root = self.place / WORKTREES_DIRNAME
        #: How long a pin's tree that no job names is kept for the next job of that pin.
        self.pin_age_s = pintree.DEFAULT_PIN_AGE_S if pin_age_s is None else float(pin_age_s)
        self.prefix = prefix
        #: Identity for the rebase's committer; the author of each commit is kept.
        self.env = env or {}
        #: Told, one line each, how every tree was made.
        self.log = log or (lambda message: None)
        #: The rules a task's commits are audited against before they land;
        #: ``None``: they are not audited.
        self.policy = policy
        self._lock = threading.Lock()
        self._digests = pintree.DigestCache(self.builds / "digests.json")
        #: Whether it was said that this host cannot read the checkout's native build.
        self._unread = False
        #: Per task, what its worker must be told of a tree that was made again.
        self._notices: dict[str, str] = {}
        # The checkout says where its run keeps trees, and whoever used it last left its own
        # place there: this run's is said now, before anything of it pins a tree.
        self._config()

    @property
    def builds(self) -> Path:
        return self.shared / pintree.BUILDS_DIRNAME

    def branch(self, task_id: str) -> str:
        return f"{self.prefix}/{task_id}"

    def tree(self, task_id: str) -> Path:
        return self.root / task_id

    def notice(self, task_id: str) -> str:
        return self._notices.get(task_id, "")

    def _git(self, cwd: Path, *args: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            ["git", "-C", str(cwd), *args],
            capture_output=True,
            text=True,
            env=pintree.git_env({**os.environ, **self.env}),
        )

    def _config(self) -> dict:
        """Say in the checkout's git directory where this run keeps trees.

        A process that has only a tree's path (a worker's ``spoolctl --tree``, a gate started by
        hand) finds there where trees go on a node and where the pin files are, and pins the
        tree before its job is queued.
        """
        config = {
            "repo": str(self.repo),
            "root": str(self.place),
            "pins": str(self.shared),
            "keep_s": self.pin_age_s,
        }
        with contextlib.suppress(pintree.PinError, OSError):
            pintree.write_config(self.repo, config)
        return config

    # ------------------------------------------------------------------ open

    def open(self, task_id: str) -> Path:
        """The task's tree, a whole tree on this node's disk, cut from the checkout's head the first time.

        A tree opened again is kept; one whose checkout was cut short is made again, on the
        branch an earlier attempt left when there is one. A branch that holds nothing the head
        lacks starts again at that head (:meth:`_caught_up`), so a retry sees what other tasks
        landed since. A tree this run opened before and that is no longer here was lost with
        its disk: it is made again from the task's branch and last pin (:meth:`_made_again`).
        """
        self._config()
        tree = self.tree(task_id)
        if _checked_out(tree):
            self._catch_up(task_id, tree)
        else:
            before = read_json(self._opened(task_id))
            pinned = pintree.last_pin(self.repo, self.branch(task_id)) if before else None
            self._create(task_id, tree, keep_branch=pinned is not None, mine=bool(before))
            if before:
                self._made_again(task_id, tree, before, pinned)
        self._opened(task_id).parent.mkdir(parents=True, exist_ok=True)
        write_json_atomic(self._opened(task_id), {"host": socket.gethostname(), "tree": str(tree)})
        pintree.link_native(self.repo, tree)
        return tree

    def _opened(self, task_id: str) -> Path:
        """Where the run notes on which host a task's tree is: what tells a lost tree from a new one."""
        return self.shared / OPEN_DIRNAME / f"{task_id}.json"

    def _caught_up(self, task_id: str) -> str | None:
        """The checkout's head, when the task's branch holds no commit it lacks; else ``None``.

        An earlier attempt that committed nothing, or whose commits all landed,
        has nothing to keep, and no rule lets its worker rebase onto the head.
        """
        ahead = git_output(self.repo, "rev-list", "--count", f"HEAD..{self.branch(task_id)}")
        if (ahead or "").strip() != "0":
            return None
        return (git_output(self.repo, "rev-parse", "HEAD") or "").strip() or None

    def _catch_up(self, task_id: str, tree: Path) -> None:
        """Move a reopened tree to the checkout's head when its branch has nothing else."""
        head = self._caught_up(task_id)
        if head is None or (git_output(tree, "rev-parse", "HEAD") or "").strip() == head:
            return
        edits = git_output(tree, "status", "--porcelain", "--untracked-files=no")
        if (edits or "").strip():
            return  # edits an earlier attempt left uncommitted stay where they are
        moved = self._git(tree, *_CHECKOUT, "reset", "--hard", "-q", head)
        if moved.returncode == 0:
            self.log(f"{tree.name}: moved to the head {head[:10]}")

    def _create(
        self, task_id: str, tree: Path, keep_branch: bool = False, mine: bool = False
    ) -> None:
        """Check the whole tree out at ``tree`` on the task's branch.

        The branch is an earlier attempt's when one is there; unless ``keep_branch``, one that
        holds nothing the head lacks starts again at the head. Only the branch is written to
        the repository (under its lock); the tree's files and its git directory are here.

        A branch of that name with commits the head lacks, for a task this run never
        opened (``mine`` is false), is another run's: the same checkout and the same
        slug, as a second run has, or a run started over in a new workspace. Its commits
        are not this task's work and would land with it. It is kept under another name,
        said, and the task starts at the head.
        """
        if tree.exists():
            pintree._discard(tree)
        try:
            self.root.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise pintree.unusable(self.place, exc) from exc
        branch = self.branch(task_id)
        started = time.monotonic()
        with pintree.repo_lock(self.repo):
            exists = git_checked(self.repo, "rev-parse", "--verify", "--quiet", branch, check=False)
            head = self._caught_up(task_id) if exists and not keep_branch else None
            aside = ""
            if exists and head is None and not keep_branch and not mine:
                tip = git_checked(self.repo, "rev-parse", "--short", branch, check=False)
                aside = f"{branch}.left-{tip}"
                moved = self._git(self.repo, "branch", "-M", branch, aside)
                if moved.returncode != 0:
                    raise RuntimeError(f"git branch -M {branch}: {moved.stderr.strip()}")
                exists = ""
            if head is not None or not exists:
                made = self._git(self.repo, "branch", "-f", branch, head or "HEAD")
                if made.returncode != 0:
                    raise RuntimeError(f"git branch {branch}: {made.stderr.strip()}")
        if aside:
            self.log(
                f"{task_id}: the checkout held a branch `{branch}` that this run did not "
                f"make, with commits its head lacks: kept as `{aside}`, and the task starts "
                "at the head"
            )
        checkout = time.monotonic()
        try:
            pintree.make_tree(self.repo, tree, f"ref: refs/heads/{branch}")
        except pintree.PinError:
            pintree._discard(tree)
            raise
        except OSError as exc:
            pintree._discard(tree)
            raise pintree.unusable(self.place, exc) from exc
        if time.monotonic() - checkout >= SETTLE_MIN_S:
            pintree.settle_index(tree, wait=True)
        self.log(f"{task_id}: tree made at {tree} in {time.monotonic() - started:.1f}s")

    def _made_again(
        self, task_id: str, tree: Path, before: dict, pinned: tuple[str, list[str]] | None
    ) -> None:
        """A tree lost with its disk is back on its branch: put the last pin's edits in, and say so.

        The branch's commits were never on that disk: they are the repository's. What the
        last pin holds beyond the branch's tip (the uncommitted edits when a job was last
        queued against the tree, or the run last pinned it) is written back into the tree,
        uncommitted as it was. Anything edited after that pin is gone, and the worker hears it.
        """
        tip = git_checked(self.repo, "rev-parse", self.branch(task_id), check=False)
        restored: list[str] = []
        if pinned is not None:
            commit, files = pinned
            done = self._git(tree, "restore", "--source", commit, "--worktree", "--", ".")
            if done.returncode == 0:
                restored = files
        where = f"`{before.get('tree')}` on {before.get('host')}"
        if restored:
            shown = ", ".join(f"`{name}`" for name in restored[:12])
            more = f" and {len(restored) - 12} more" if len(restored) > 12 else ""
            back = (
                f"your branch `{self.branch(task_id)}` at `{tip[:10]}`, with the uncommitted "
                f"edits of your tree's last pin put back ({len(restored)} file(s): {shown}{more})"
            )
            lost = "every edit you made after that pin"
        else:
            back = f"your branch `{self.branch(task_id)}` at `{tip[:10]}`"
            lost = "every edit you had not committed"
        self._notices[task_id] = (
            "## Your tree was lost and made again\n\n"
            f"The disk that held your tree ({where}) is gone with its node. The tree is made "
            f"again at `{tree}` from {back}. A pin is taken each time a job is queued against "
            f"the tree. **Lost: {lost}.** Look at `git -C {tree} status` and at your notes, "
            "and redo what is missing before you go on."
        )
        self.log(
            f"{task_id}: its tree on {before.get('host')} was lost; made again at {tree} from "
            f"{tip[:10]}" + (f" and {len(restored)} file(s) of its last pin" if restored else "")
        )

    # ------------------------------------------------------------- pin, at

    def pin(self, task_id: str | None = None) -> Pin:
        """The task's tree as it is now, as a pin; with no task, the checkout's head.

        The tree's branch, index and files are left as they are: its edits are committed
        beside the branch (``pintree.freeze``). The build record is that of the libraries the
        tree links, which is the one build of the run.
        """
        if task_id is None:
            head = (git_output(self.repo, "rev-parse", "HEAD") or "").strip()
            if not head:
                raise pintree.PinError(f"git could not name the commit {self.repo} is at")
            return self.pinned(head)
        return self.pinned(pintree.freeze(self.tree(task_id), self.branch(task_id)))

    def pinned(self, commit: str) -> Pin:
        """The pin of ``commit`` on the libraries on disk: the one build every tree links."""
        with self._lock:
            build = pintree.build_record(self.repo, cache=self._digests)
        unread = pintree.unread(self.repo)
        if unread and not self._unread:
            self._unread = True
            self.log(
                f"the native build of {self.repo} cannot be read on this host: {len(unread)} "
                f"link(s) lead nowhere here (`{unread[0]}`). A pin records no library below "
                "them, and each node a job runs on records the libraries it finds there"
            )
        record = pintree.record(self._config(), commit, build, unread)
        return Pin(commit, build, pintree.write_pin(pintree.pins_root(self.shared), record))

    @contextlib.contextmanager
    def at(self, pin: Pin) -> Iterator[Path]:
        """``pin``'s commit as a detached, read-only whole tree on this node, held for the block.

        Made the first time it is asked for here, by one maker at a time, and used again
        after. Holding is a shared lock on a file beside the trees, which ends with the
        process that holds it: nothing is leased. A job of the pin on another node has its
        own tree there, at this same path. The directory is the pin's own (its commit and
        its build), so the path stands for this pin and no other for as long as it exists.
        Leaving the block removes what is done with (:meth:`prune`).
        """
        tree = pintree.pinned_tree(self.place, pin.id)
        before = _made(tree)
        started = time.monotonic()
        handle = pintree.hold(
            self.place, pin.id, repo=self.repo, pin_file=str(pin.file or ""), commit=pin.commit
        )
        try:
            if _made(tree) != before:
                again = " again (it no longer held its commit)" if before else ""
                took = time.monotonic() - started
                self.log(f"tree of {pin.commit[:10]} made{again} at {tree} in {took:.1f}s")
            yield tree
        finally:
            handle.close()
            self.prune()

    def prune(self) -> list[str]:
        """Remove the pinned trees on this node that are done with; the commits removed.

        ``pintree.sweep``: a tree stays while it is the head's, while a process here holds it,
        and until it went unused for ``pin_age_s``.
        """
        head = (git_output(self.repo, "rev-parse", "HEAD") or "").strip()
        try:
            removed = pintree.sweep(self.place, keep={head}, age_s=self.pin_age_s)
        except OSError:
            return []
        if removed:
            self.log(f"removed the trees of {', '.join(c[:10] for c in removed)}")
        return removed

    def changed_paths(self, task_id: str) -> tuple[str, ...]:
        """The files the task's branch changes since it left the head's history."""
        listed = git_output(self.repo, "diff", "--name-only", f"HEAD...{self.branch(task_id)}")
        return tuple(sorted({line.strip() for line in (listed or "").splitlines() if line.strip()}))

    def tracked(self, commit: str) -> frozenset[str]:
        """Every path ``commit``'s tree holds; empty when git cannot list it."""
        known = getattr(self, "_tracked", None)
        if known is None or known[0] != commit:
            listing = git_output(self.repo, "ls-tree", "-r", "-z", "--name-only", commit) or ""
            known = (commit, frozenset(p for p in listing.split("\0") if p))
            self._tracked = known
        return known[1]

    def commit_of(self, path: str | Path) -> str | None:
        """The commit whose pinned tree ``path`` is, on this node or, by its path, on any."""
        commit = pintree.pinned_commit(path)
        if commit:
            return commit
        path = Path(os.path.normpath(str(path)))
        if path.parent != pintree.pins_root(self.place) or not pintree._KEY.match(path.name):
            return None
        if pintree._COMMIT.match(path.name):
            return path.name
        # A pin's id names its commit by its first letters: the repository knows the rest.
        short = path.name.split("-")[0]
        full = git_output(self.repo, "rev-parse", "--verify", "--quiet", f"{short}^{{commit}}")
        return (full or "").strip() or None

    # ------------------------------------------------------------------ land

    def _integrated(self) -> dict[str, str]:
        path = self.shared / INTEGRATED_FILENAME
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        return record if isinstance(record, dict) else {}

    def integrated_tip(self, task_id: str) -> str:
        """The branch tip last landed for ``task_id``, or empty when none is recorded."""
        return str(self._integrated().get(task_id) or "")

    def _record(self, task_id: str, tip: str) -> None:
        record = {**self._integrated(), task_id: tip}
        self.shared.mkdir(parents=True, exist_ok=True)
        (self.shared / INTEGRATED_FILENAME).write_text(
            json.dumps(record, indent=1), encoding="utf-8"
        )

    def _already_in(self, task_id: str) -> bool:
        """The task's branch went in before, and is in the checkout as it was then.

        A process that dies between a task's fast-forward and its checkpoint
        resumes the task, and its worker's decision is read again; the branch
        must not land twice, from a worktree half removed.
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

    def land(self, task_id: str) -> Outcome:
        """Rebase the task's branch onto the checkout's head and fast-forward it in."""
        tree, branch = self.tree(task_id), self.branch(task_id)
        if self._already_in(task_id):
            return Outcome(True, f"{branch} was integrated before the restart", 0)
        if not (tree / ".git").exists():
            return Outcome(False, f"task {task_id} has no worktree at {tree}")
        if _rebase_in_progress(tree):
            # A rebase cut short, by a process that stopped mid-landing or
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
        with self._lock, pintree.repo_lock(self.repo):
            head = git_checked(self.repo, "rev-parse", "HEAD")
            ahead = git_checked(
                tree, "rev-list", "--count", f"{head}..{branch}", check=False
            ).strip()
            if ahead in ("", "0"):
                return Outcome(True, "nothing to integrate", 0)
            broken = self._audit(tree, head, branch) or self._unbuilt(tree, head, branch)
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
            + f"\nRewrite the commits since {fork[:10]}; the commit rules allow the rebase a "
            f"dispatch asks for: `git -C {tree} rebase --signoff {fork}` signs each one off, and "
            f"`git -C {tree} reset --soft {fork}` lets you commit again only what belongs to "
            "the task."
        )

    def _unbuilt(self, tree: Path, head: str, branch: str) -> str:
        """The native source the task's branch changes, said for its worker; "" if none."""
        names = git_checked(tree, "diff", "--name-only", "--no-renames", f"{head}...{branch}")
        native = sorted(name for name in names.splitlines() if _is_native(name.strip()))
        if not native:
            return ""
        shown = ", ".join(f"`{name}`" for name in native[:8])
        more = f" and {len(native) - 8} more" if len(native) > 8 else ""
        return f"commits on {branch} change native code ({shown}{more}). {NATIVE_BASE}"

    def close(self, task_id: str, *, landed: bool = True) -> None:
        """The task's attempt is over. Landed: remove its worktree; its branch stays as the record.

        One that did not land keeps its worktree and branch for a next attempt.
        """
        tree = self.tree(task_id)
        if landed:
            pintree._discard(tree)
            self._opened(task_id).unlink(missing_ok=True)
            self._notices.pop(task_id, None)


def _is_native(name: str) -> bool:
    """Whether a path is native source by what it is: a native language's file or a build description."""
    path = PurePosixPath(name)
    return path.suffix in NATIVE_SUFFIXES or path.name in BUILD_DESCRIPTIONS


def _rebase_in_progress(tree: Path) -> bool:
    """Whether ``tree`` holds a rebase that stopped before it finished."""
    for state in ("rebase-merge", "rebase-apply"):
        where = git_checked(tree, "rev-parse", "--git-path", state, check=False).strip()
        if where and (tree / where).exists():
            return True
    return False


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
        "it depends on, say), rebase onto that head before anything else. This is the one "
        "rebase the commit rules allow:\n\n"
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
            + _behind_note(tree, repo, scratch=True)
        )
    return (
        "## Your worktree\n\n"
        f"You work in `{tree}`, your own branch of the checkout `{repo}`. Edit, test and "
        f"commit there and nowhere else; do not touch `{repo}`. Read paths relative to your "
        "worktree where the prompt names the checkout. Commit with\n\n"
        f"    git -C {tree} add <path> [path...] && git -C {tree} commit -s -m <message>\n\n"
        f"Run a gate on your work with `--tree {tree}`: it measures a pin of what the worktree "
        "holds at that moment, uncommitted edits included, so what you edit while it runs is "
        "not what it measures. When you finish, the orchestrator "
        "checks your commits, rebases them onto the checkout and runs your task's checks "
        "there; uncommitted changes are not integrated and fail the task.\n\n"
        f"## Native code\n\n{NATIVE_BASE}" + _behind_note(tree, repo, scratch=False)
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
        "your worktree was cut. Do not start the task again. In your worktree, rebase onto "
        "that head: this is the one rebase the commit rules allow:\n\n"
        f"    git -C {tree} rebase {outcome.base}\n\n"
        "Resolve each conflict so that both your change and the other task's stand, then "
        f"`git -C {tree} add` the resolved files and `GIT_EDITOR=true git -C {tree} rebase "
        "--continue` until the rebase is done; the commits keep their messages and sign-offs. "
        "Run your task's tests again on the rebased tree, commit any fix with `git commit -s`, "
        "and write your decision as you did when you finished. If the two changes "
        f"cannot both stand, `git -C {tree} rebase --abort` and end `blocked`, naming the "
        "files and why."
    )


__all__ = [
    "CHECKOUT_WORKERS",
    "INTEGRATED_KINDS",
    "Integration",
    "NATIVE_BASE",
    "NATIVE_SUFFIXES",
    "Outcome",
    "Pin",
    "REBASE_RETRIES",
    "SCRATCH_KINDS",
    "TREES_FIELD",
    "TREES_KEYS",
    "TaskNotStarted",
    "make",
    "place",
    "rebase_block",
    "repair_block",
    "validate_trees_block",
    "worktree_block",
]
