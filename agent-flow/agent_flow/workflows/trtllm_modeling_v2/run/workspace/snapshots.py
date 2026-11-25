# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Read-only snapshots of the checkout's commits: the trees contract checks measure.

The integration checkout moves while checks run, one finished task at a time,
so a check measures a snapshot of one commit instead (:class:`Snapshots`): a
detached worktree at ``<workspace>/snapshots/<commit>``, sparse with the cone a
task worktree checks out (``integration.cone``), the checkout's native build
linked in, its files read-only. Every check of one commit shares that one
snapshot. A check leases it while it runs; once the checkout's head has moved
past its commit, a snapshot no lease holds is removed.

What a check imported is read from what its commands print
(:class:`CheckTree`): the spool's resolve hook names the file each process
imported the package from, and the engine driver names it once per engine. A
file inside a snapshot names that snapshot's commit; any other names a tree no
commit pins. A check whose commands imported more than one tree, or a tree no
commit pins, is void. What each process imported of the tree is read too (the
rank hook's reads files): a verdict holds on a later commit whose Python files
it read, and every file of the tree that is not Python, are unchanged
(``run/workflow.py:_commit_marker``).
"""

from __future__ import annotations

import contextlib
import fcntl
import json
import os
import re
import shlex
import shutil
import socket
import stat
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Any, Callable, Collection, Iterable, Iterator, Mapping

from ...checks import check_runtime
from ...checks import verdict as check_result
from ...checks.harness import workload_io
from ...infra.compute.provenance import checkout_root
from ...infra.git import git_output, link_native, repo_lock
from .integration import _CHECKOUT, SPARSE_ON, _drop_cone, _last_line, cone

SNAPSHOTS_DIRNAME = "snapshots"
#: Written last into a snapshot: one without it was cut short, and is made again.
READY_FILENAME = ".mv2flow-snapshot"
#: A lease older than this was never given back (its process died), and holds nothing.
LEASE_TTL_S = 24 * 3600.0
#: How many times a void check is measured again on the same snapshot.
VOID_RERUNS = 1
#: The spec's blocks whose words may name a file of the checkout by its absolute path.
SPEC_BLOCKS: tuple[str, ...] = ("accuracy", "runtime", "variants", "environment")

_LEASES, _LOCKS = ".leases", ".locks"
_COMMIT = re.compile(r"^[0-9a-f]{40}(?:[0-9a-f]{24})?$")
#: Beside the snapshots: per command, the directory its processes write what they read to.
READS_DIRNAME = ".reads"
#: Where the rank hook (``checks/harness/rankhook``) writes each process's reads, and of which tree.
READS_DIR_ENV = "MV2FLOW_READS_DIR"
TREE_ENV = "MV2FLOW_TREE"
#: ``[mv2flow-resolve] pid=.. rank=.. module=<module> origin=<file>``, as the resolve hook prints it.
_RESOLVED = re.compile(r"\[mv2flow-resolve\][^\n]*?\bmodule=(\S+) origin=([^\n]*)")


class SnapshotError(RuntimeError):
    """A commit's snapshot could not be made."""


class Snapshots:
    """The snapshots of one checkout's commits, kept under ``root``."""

    def __init__(
        self,
        repo: str | Path,
        root: str | Path,
        *,
        cone_paths: Iterable[str] = (),
        log: Callable[[str], None] | None = None,
    ) -> None:
        self.repo = Path(repo)
        self.root = Path(root)
        #: What each snapshot checks out: a task worktree's cone and the
        #: directories of ``cone_paths`` (the files the gates read); None for
        #: the whole tree.
        self.dirs = cone(cone_paths)
        self.log = log or (lambda message: None)
        self._tracked: tuple[str, frozenset[str]] | None = None

    def tree(self, commit: str) -> Path:
        return self.root / commit

    @contextlib.contextmanager
    def lease(self, commit: str) -> Iterator[Path]:
        """``commit``'s snapshot, made the first time it is asked for, held for the block.

        Giving the lease back prunes what no check needs any more.
        """
        tree = self.tree(commit)
        with self._locked(commit):
            if not _ready(tree, commit):
                self._make(commit, tree)
            holder = f"{socket.gethostname()}-{os.getpid()}-{uuid.uuid4().hex[:8]}"
            lease = self.root / _LEASES / commit / holder
            lease.parent.mkdir(parents=True, exist_ok=True)
            lease.touch()
        try:
            yield tree
        finally:
            lease.unlink(missing_ok=True)
            self.prune()

    def prune(self) -> list[str]:
        """Remove every snapshot no lease holds but the head's; the commits removed.

        One being made or pruned meanwhile is left to whoever holds it.
        """
        if not self.root.is_dir():
            return []
        head = (git_output(self.repo, "rev-parse", "HEAD") or "").strip()
        removed = []
        for tree in sorted(self.root.iterdir()):
            if tree.name == head or not _COMMIT.match(tree.name) or not tree.is_dir():
                continue
            with self._locked(tree.name, wait=False) as locked:
                if not locked or self._held(tree.name):
                    continue
                self._discard(tree)
                with contextlib.suppress(OSError):
                    (self.root / _LEASES / tree.name).rmdir()
            removed.append(tree.name)
        if removed:
            self.log(f"pruned the snapshots of {', '.join(c[:10] for c in removed)}")
        return removed

    def commit_of(self, path: str | Path) -> str | None:
        """The commit whose snapshot ``path`` is; ``None`` for any other directory.

        By directory identity as well as by spelling: one storage system may be
        mounted under two prefixes.
        """
        real = Path(os.path.realpath(path))
        if real.parent == Path(os.path.realpath(self.root)) and _COMMIT.match(real.name):
            return real.name
        if not self.root.is_dir():
            return None
        for tree in self.root.iterdir():
            with contextlib.suppress(OSError):
                if _COMMIT.match(tree.name) and os.path.samefile(tree, path):
                    return tree.name
        return None

    def tracked(self, commit: str) -> frozenset[str]:
        """Every path ``commit``'s tree holds; empty when git cannot list it."""
        if self._tracked is None or self._tracked[0] != commit:
            listing = git_output(self.repo, "ls-tree", "-r", "-z", "--name-only", commit) or ""
            self._tracked = (commit, frozenset(p for p in listing.split("\0") if p))
        return self._tracked[1]

    # ------------------------------------------------------------- internals

    @contextlib.contextmanager
    def _locked(self, commit: str, *, wait: bool = True) -> Iterator[bool]:
        """Hold ``commit``'s lock, across threads and processes; whether it is held.

        One maker or pruner of a snapshot at a time. Without ``wait``, a lock
        another holds is not waited for.
        """
        path = self.root / _LOCKS / f"{commit}.lock"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as handle:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX if wait else fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                yield False
                return
            try:
                yield True
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)

    def _held(self, commit: str) -> bool:
        """Whether a lease younger than :data:`LEASE_TTL_S` holds ``commit``'s snapshot."""
        now = time.time()
        held = False
        for lease in (self.root / _LEASES / commit).glob("*"):
            with contextlib.suppress(OSError):
                if now - lease.stat().st_mtime < LEASE_TTL_S:
                    held = True
                else:
                    lease.unlink()
        return held

    def _make(self, commit: str, tree: Path) -> None:
        """Check ``commit`` out at ``tree``: detached, sparse, its native build linked, read-only.

        Only registering the worktree holds the repository's lock; checking
        its files out runs beside other work.
        """
        self._discard(tree)
        self.root.mkdir(parents=True, exist_ok=True)
        with repo_lock(self.repo):
            # A registration whose files are gone would refuse the path.
            _git(self.repo, "worktree", "prune")
            added = _git(
                self.repo, "worktree", "add", "--detach", "--no-checkout", str(tree), commit
            )
        if added.returncode != 0:
            self._discard(tree)
            raise SnapshotError(f"git worktree add {commit[:10]}: {_last_line(added.stderr)}")
        sparse, how = (), "whole"
        if self.dirs is not None:
            made = _git(tree, *SPARSE_ON, "sparse-checkout", "set", "--skip-checks", *self.dirs)
            if made.returncode == 0:
                sparse, how = SPARSE_ON, f"sparse: {', '.join(self.dirs)}"
            else:
                _drop_cone(tree)
                how = f"whole, as git could not make it sparse ({_last_line(made.stderr)})"
        checked = _git(tree, *_CHECKOUT, *sparse, "reset", "--hard", "-q")
        if checked.returncode != 0:
            self._discard(tree)
            raise SnapshotError(f"checking out {commit[:10]}: {_last_line(checked.stderr)}")
        link_native(self.repo, tree)
        _read_only(tree)
        (tree / READY_FILENAME).write_text(f"{commit}\n", encoding="utf-8")
        self.log(f"snapshot of {commit[:10]} made at {tree}, {how}")

    def _discard(self, tree: Path) -> None:
        """Remove ``tree`` and its registration, whatever state a cut-short making left it in.

        The files go first, outside the repository's lock; ``git worktree
        prune`` then drops the registration of a worktree whose files are gone.
        """
        if not tree.exists():
            return
        shutil.rmtree(tree, ignore_errors=True)
        with repo_lock(self.repo):
            _git(self.repo, "worktree", "prune")


@dataclass
class CheckTree:
    """One check's tree: the commit it measures, that commit's snapshot, and what it imported.

    ``marker`` is the target's marker for ``commit``. ``uncommitted`` are the
    files a gate reads that the checkout held uncommitted when the check
    started, which no snapshot holds. ``snapshots`` names the commit of any
    other snapshot a command imported from.
    """

    commit: str
    tree: Path
    marker: str = ""
    uncommitted: tuple[str, ...] = ()
    snapshots: Snapshots | None = None
    #: What each command imported, in order: per command, the commit of each
    #: snapshot, or the path of each checkout, its processes imported from.
    commands: list[tuple[str, ...]] = field(default_factory=list)
    #: The tree's Python files the commands' processes imported, relative to it
    #: (the rank hook's ``MV2FLOW_READS`` files); ``None`` once a command that
    #: ran left none, so what the check read is not known.
    reads: set[str] | None = field(default_factory=set)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    # --------------------------------------------------------------- the spec

    def spec(self, task_data: Mapping[str, Any], tracked: Collection[str]) -> dict[str, Any]:
        """``task_data`` with the target in the snapshot.

        ``trtllm_repo_path`` names the snapshot, so whatever a gate resolves
        under it (the quickstart, a prompt's image, a relative anchor table) is
        the commit's. So does every word of :data:`SPEC_BLOCKS` naming a path
        of the checkout by its absolute path that the commit tracks (in
        ``tracked``); one it does not track, a virtualenv built in the
        checkout say, stays where it is.
        """
        repo = str(task_data.get("trtllm_repo_path") or "").rstrip("/")
        moved = {**task_data, "trtllm_repo_path": str(self.tree)}
        if not repo:
            return moved
        inside = re.compile(re.escape(repo + "/") + r"([^\s'\":;,=]+)")

        def held(rel: str) -> bool:
            rel = rel.rstrip("/")
            return rel in tracked or any(path.startswith(rel + "/") for path in tracked)

        def move(value: Any) -> Any:
            if isinstance(value, Mapping):
                return {key: move(item) for key, item in value.items()}
            if isinstance(value, list):
                return [move(item) for item in value]
            if isinstance(value, str) and repo + "/" in value:
                return inside.sub(
                    lambda m: f"{self.tree}/{m.group(1)}" if held(m.group(1)) else m.group(0), value
                )
            return value

        for block in SPEC_BLOCKS:
            if block in task_data:
                moved[block] = move(task_data[block])
        return moved

    # ------------------------------------------------------------ the commands

    def runner(self, runner: Callable[..., Any]) -> Callable[..., Any]:
        """``runner``, run from inside the snapshot (``check_runtime.in_tree``), noting what each imported."""
        inside = check_runtime.in_tree(runner, self.tree)

        def run(command: str, env: Mapping[str, str] | None = None, timeout: int = 1800) -> Any:
            reads = self.tree.parent / READS_DIRNAME / uuid.uuid4().hex
            hook = shlex.quote(str(check_runtime.RANK_HOOK_DIR))
            prefix = (
                f"export {READS_DIR_ENV}={shlex.quote(str(reads))} "
                f"{TREE_ENV}={shlex.quote(str(Path(self.tree).resolve()))} "
                f'PYTHONPATH={hook}"${{PYTHONPATH:+:$PYTHONPATH}}" && '
            )
            try:
                result = inside(prefix + command, env, timeout)
                output = f"{getattr(result, 'stdout', '')}\n{getattr(result, 'stderr', '')}"
                if self.observe(output):
                    self.read(reads)
            finally:
                shutil.rmtree(reads, ignore_errors=True)
            return result

        return run

    def read(self, directory: Path) -> None:
        """Add what one command's processes imported of the tree (``directory``'s files).

        Asked of a command that imported the package: one that left no file
        makes what the check read unknown, as its processes may have imported
        anything.
        """
        found: set[str] = set()
        files = sorted(directory.glob("*.json")) if directory.is_dir() else []
        for path in files:
            try:
                found.update(str(p) for p in json.loads(path.read_text(encoding="utf-8")))
            except (OSError, ValueError, TypeError):
                files = []
                break
        with self._lock:
            if not files or self.reads is None:
                self.reads = None
            else:
                self.reads |= found

    def observe(self, output: str) -> bool:
        """Note what one command's ``output`` says its processes imported; whether it said."""
        roots = {
            checkout_root(module, origin.strip())
            for module, origin in _RESOLVED.findall(output)
            if origin.strip() not in ("", "None")
        }
        for record in workload_io.parse_marked(output, workload_io.DRIVER_MARKER):
            if record.get("tensorrt_llm"):
                roots.add(checkout_root("tensorrt_llm", str(record["tensorrt_llm"])))
        if roots:
            with self._lock:
                self.commands.append(tuple(sorted(self._name(root) for root in roots)))
        return bool(roots)

    def _name(self, root: str) -> str:
        if os.path.realpath(root) == os.path.realpath(self.tree):
            return self.commit
        found = self.snapshots.commit_of(root) if self.snapshots is not None else None
        return found or root

    def imported(self) -> tuple[str, ...]:
        """What each command that said imported: a commit, or ``+``-joined where it was several."""
        return tuple("+".join(names) for names in self.commands)

    def void(self) -> str:
        """Why what the commands imported is not this one commit; ``""`` when it is, or none said."""
        seen = {name for names in self.commands for name in names}
        others = sorted(seen - {self.commit})
        if not others:
            return ""
        named = [
            name[:10] if _COMMIT.match(name) else f"{name} (a checkout no commit pins)"
            for name in others
        ]
        which = "some of its commands" if self.commit in seen else "its commands"
        return (
            f"{which} imported the package under test from {', '.join(named)}, not from the "
            f"snapshot of {self.commit[:10]} the check measures"
        )

    # ---------------------------------------------------------------- results

    def stamp(self, result: check_result.GateResult) -> check_result.GateResult:
        """``result`` as measured on this commit: its description, marker and imports, void or not."""
        warnings = ()
        if self.uncommitted:
            shown = ", ".join(self.uncommitted[:5])
            more = f" and {len(self.uncommitted) - 5} more" if len(self.uncommitted) > 5 else ""
            warnings = (
                f"the checkout holds {shown}{more} uncommitted; this check measured "
                f"{self.commit[:10]} without them",
            )
        return check_result.measured_on(
            result,
            tree_state=self.description,
            marker=self.marker,
            imported=self.imported(),
            void=self.void(),
            warnings=warnings,
            reads=tuple(sorted(self.reads)) if self.reads else (),
        )

    @cached_property
    def description(self) -> str:
        """The commit as a gate line names a tree (``result.describe_tree_state``)."""
        described = check_result.describe_tree_state(self.tree)
        return self.commit[:10] if described == check_result.UNKNOWN_TREE_STATE else described

    def settle(
        self,
        measure: Callable[[], check_result.GateResult],
        note: Callable[[str], None] = lambda message: None,
    ) -> check_result.GateResult:
        """``measure()`` stamped; a void result is measured again, :data:`VOID_RERUNS` times at most.

        ``note`` hears why each void one is measured again.
        """
        for attempt in range(VOID_RERUNS + 1):
            with self._lock:
                self.commands.clear()
                self.reads = set()
            result = self.stamp(measure())
            if not result.void or attempt == VOID_RERUNS:
                return result
            label = f"{result.name}@{result.member}" if result.member else result.name
            note(f"{label} on {self.commit[:10]} was void ({result.void}); measuring it again")
        return result


def void_result(name: str, member: str | None, why: str) -> check_result.GateResult:
    """A check that measured nothing because it got no tree to measure."""
    return check_result.GateResult(
        name=name, member=member, passed=False, void=why, detail=why, explained=True
    )


def _ready(tree: Path, commit: str) -> bool:
    """Whether ``tree`` is ``commit``'s snapshot, made to the end."""
    try:
        return (tree / READY_FILENAME).read_text(encoding="utf-8").strip() == commit
    except OSError:
        return False


def _read_only(tree: Path) -> None:
    """Take write permission off every file in ``tree``; directories keep it, for bytecode caches.

    A symlink is left alone: it is the checkout's native build, which a
    rebuild there rewrites.
    """
    for directory, _, names in os.walk(tree):
        for name in names:
            path = os.path.join(directory, name)
            with contextlib.suppress(OSError):
                mode = os.lstat(path).st_mode
                if stat.S_ISREG(mode) and mode & 0o222:
                    os.chmod(path, mode & ~0o222)


def _git(cwd: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(cwd), *args], capture_output=True, text=True)


__all__ = [
    "LEASE_TTL_S",
    "READY_FILENAME",
    "SNAPSHOTS_DIRNAME",
    "SPEC_BLOCKS",
    "VOID_RERUNS",
    "CheckTree",
    "SnapshotError",
    "Snapshots",
    "void_result",
]
