# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Opt-in ``git worktree`` isolation for the DAG engine.

This is the concrete, git-backed :class:`~agent_flow.orchestration.isolation.
IsolationProvider`. It gives each concurrent node its own working tree so
parallel work cannot clobber a shared checkout, and seeds a fan-in node's
worktree by git-merging its dependency branches. Worktrees created from one
repository share the same ``.git`` object store but have independent working
directories and branches, which is exactly the isolation the scheduler needs.

It lives *outside* the orchestration core on purpose: the core engine depends
only on the :class:`IsolationProvider` abstraction and must never import this
module, so workflows that want the concurrent DAG engine without git (a research
fan-out, a data-processing graph) can use :class:`~agent_flow.orchestration.
isolation.NoOpIsolation` instead. The dependency arrow points one way — this
module imports from :mod:`agent_flow.orchestration`, never the reverse.

Git is driven purely through ``asyncio.create_subprocess_exec``.
"""

from __future__ import annotations

import asyncio
import shutil
from dataclasses import dataclass, field
from pathlib import Path

from .orchestration.graph import Node
from .orchestration.isolation import IsolationError

__all__ = [
    "GitWorktreeIsolation",
    "MergeReport",
    "node_id_to_slug",
]


@dataclass
class MergeReport:
    """Outcome of git-merging a merge node's dependency branches into its worktree.

    A merge *conflict* is data, not an error: the conflicted files keep their
    conflict markers in the worktree so a later resolver (the merge node's own
    ``run_node`` in a Phase 2+ workflow) can settle them. Only a genuinely
    unexpected git failure (a bad branch ref, a missing repo) raises
    :class:`~agent_flow.orchestration.isolation.IsolationError`.

    Attributes:
        merged: Dependency branch names that merged cleanly, in merge order.
        conflicted_paths: Repo-relative paths left with conflict markers in the
            worktree. Non-empty means the worktree holds an in-progress,
            un-aborted merge that still needs resolution.
    """

    merged: list[str] = field(default_factory=list)
    conflicted_paths: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)


def node_id_to_slug(node_id: str) -> str:
    """Map a node id to a filesystem-safe worktree directory name.

    Node ids may contain path separators (e.g. ``s1/g2``) that are illegal in a
    single directory component. Path separators are replaced with ``_`` so the
    id collapses to one safe leaf directory, while the git branch keeps the raw
    id (slashes and dots are legal in refs).

    Args:
        node_id: The node's globally unique id.

    Returns:
        A string safe to use as a single path component.
    """
    return node_id.replace("/", "_").replace("\\", "_")


class GitWorktreeIsolation:
    """Isolate each node in its own ``git worktree`` under a shared root.

    Implements the :class:`~agent_flow.orchestration.isolation.IsolationProvider`
    Protocol. For ``node.isolation == "worktree"`` (the default), :meth:`acquire`
    adds a worktree at ``<worktrees_root>/<slug(node.id)>`` on branch
    ``<branch_prefix><node.id>`` seeded from ``base_ref`` (reusing an existing
    worktree/branch if one is already present, so acquire is idempotent on a
    resume). :meth:`release` removes *only* the worktree directory, keeping the
    branch — the node's durable output — alive so a later fan-in node can merge
    it; :meth:`reclaim` deletes that branch (and prunes) once no dependent needs
    it, tolerant of an already-gone branch. For ``node.isolation == "shared"``,
    the node runs directly in ``repo_path``: :meth:`acquire` returns
    ``repo_path`` and :meth:`release`/:meth:`reclaim` are no-ops. :meth:`prepare`
    git-merges a fan-in node's dependency branches into its worktree.
    """

    def __init__(
        self,
        repo_path: Path,
        worktrees_root: Path,
        branch_prefix: str = "node/",
        base_ref: str = "HEAD",
    ) -> None:
        """Configure the provider.

        Args:
            repo_path: Path to the source repository (the main checkout).
            worktrees_root: Directory under which per-node worktrees are created.
            branch_prefix: Prefix for the per-node branch name.
            base_ref: Commit-ish each new worktree branch is seeded from.
        """
        # Absolutize both paths: ``git -C <repo> worktree add <relative>`` resolves
        # against the repo, but ``acquire``'s return is used as the agent cwd
        # against the process cwd — a relative ``worktrees_root`` makes the two
        # diverge. ``.absolute()`` (not ``.resolve()``) skips symlink canonicalization.
        self.repo_path = Path(repo_path).absolute()
        self.worktrees_root = Path(worktrees_root).absolute()
        self.branch_prefix = branch_prefix
        self.base_ref = base_ref

    def _worktree_path(self, node: Node) -> Path:
        """Absolute path of the worktree directory for ``node``."""
        return self.worktrees_root / node_id_to_slug(node.id)

    def _branch(self, node: Node) -> str:
        """Branch name for ``node``'s worktree."""
        return f"{self.branch_prefix}{node.id}"

    async def acquire(self, node: Node) -> Path:
        """Provide ``node``'s working directory, creating a worktree if needed.

        Returns ``repo_path`` unchanged for shared nodes. Otherwise the call is
        idempotent — safe to re-run on a resume or re-acquire:

        * if the worktree directory already exists it is reused as-is;
        * if only the branch survives (a prior :meth:`release` removed the
          worktree but kept the branch) that branch is checked back out into a
          fresh worktree, restoring its committed content;
        * otherwise a brand-new worktree is created on branch
          ``<branch_prefix><node.id>`` seeded from ``base_ref``.

        Raises:
            IsolationError: If the ``git worktree add`` command fails.
        """
        if node.isolation == "shared":
            return self.repo_path

        self.worktrees_root.mkdir(parents=True, exist_ok=True)
        path = self._worktree_path(node)
        if path.exists():
            # Already acquired (or resumed): reuse the existing worktree.
            return path
        branch = self._branch(node)
        if await self._branch_exists(branch):
            # The durable branch outlived its worktree (post-release / resume):
            # re-check it out rather than trying to create it again.
            await self._git("worktree", "add", str(path), branch)
        else:
            await self._git("worktree", "add", "-b", branch, str(path), self.base_ref)
        return path

    async def release(self, node: Node) -> None:
        """Free ``node``'s worktree, keeping its branch; a no-op for shared nodes.

        Removes only the worktree directory. The node's branch — its durable
        output — is deliberately kept alive so a later fan-in node can merge it;
        it is deleted later by :meth:`reclaim`. Tolerates an already-removed
        worktree so a redundant release cannot fail.

        Raises:
            IsolationError: If the ``git worktree remove`` command fails for a
                reason other than the worktree already being gone.
        """
        if node.isolation == "shared":
            return

        path = self._worktree_path(node)
        if not path.exists():
            # Nothing to free: the worktree was already removed (idempotent).
            return
        await self._git("worktree", "remove", "--force", str(path))

    async def reclaim(self, node: Node) -> None:
        """Free ``node``'s durable output: delete its branch and prune worktrees.

        Called by the scheduler once no remaining dependent needs this node's
        branch. Idempotent and tolerant: deleting an already-gone (or
        never-created, e.g. a BLOCKED node) branch does not raise, and a final
        ``git worktree prune`` clears any stale administrative entries left by a
        prior removal. A no-op for shared nodes, which have no durable output.
        """
        if node.isolation == "shared":
            return

        # ``check=False``: tolerate an already-deleted or never-created branch so
        # reclaim is safe to call exactly once regardless of prior state.
        await self._git("branch", "-D", self._branch(node), check=False)
        await self._git("worktree", "prune", check=False)

    async def commit(self, node: Node, cwd: Path) -> str | None:
        """Commit ``node``'s changes to its durable output before release.

        Called by the scheduler after ``run_node`` reaches a terminal state and
        *before* :meth:`release` tears the workspace down, so the node's work is
        captured instead of being discarded (the concurrent code-loss defect this
        closes). Stages every change — new files, edits, deletions; gitignored
        build artifacts are excluded by ``git add`` — and commits it in ``cwd``:

        * a **worktree** node commits onto its own ``node/<id>`` branch, kept alive
          by :meth:`release` so a dependent can merge it;
        * a **shared** node commits onto the base branch (its ``cwd`` is
          ``repo_path``), advancing ``HEAD``. This is how a fan-in / ``merge``
          node's integration survives — and how a *later* stage's fresh worktree
          (forked from ``HEAD``) inherits it, the cross-stage code handoff.

        Returns the new commit's short hash, or ``None`` when there is nothing to
        commit (a clean tree). ``--no-verify`` skips any pre-commit hooks the
        target repo installs: a per-node commit is internal plumbing, not a
        user-authored commit, so a slow or failing hook must not block the handoff.

        Raises:
            IsolationError: If a ``git`` step fails for an unexpected reason.
        """
        await self._git("add", "-A", cwd=cwd)
        # ``diff --cached --quiet`` exits 0 when nothing is staged: skip the
        # commit rather than record an empty one.
        returncode, _stdout, _stderr = await self._run_git("diff", "--cached", "--quiet", cwd=cwd)
        if returncode == 0:
            return None
        message = f"[agent-flow] node {node.id} ({node.kind}) output"
        await self._git("commit", "--no-verify", "-m", message, cwd=cwd)
        return (await self._git("rev-parse", "--short", "HEAD", cwd=cwd)).strip()

    async def clean(self) -> None:
        """Tear down all of this provider's state and reset the repo to ``HEAD``.

        The ``--clean`` "start over" teardown: removes the per-node worktrees +
        ``<branch_prefix>*`` branches, then restores the main working tree with
        ``git reset --hard`` + ``git clean -fd``. ``-fd`` (not ``-fdx``) keeps
        gitignored build artifacts, so a rerun does not force a full rebuild.
        Every git step uses ``check=False`` so a repo with no worktrees/branches
        (a fresh checkout or a linear run) is a tolerant no-op; ``reset --hard``
        also clears any ``MERGE_HEAD`` a crashed merge node left behind.
        """
        for worktree in await self._worktrees_under_root():
            await self._git("worktree", "remove", "--force", str(worktree), check=False)
        await self._git("worktree", "prune", check=False)
        branches = await self._node_branches()
        if branches:
            await self._git("branch", "-D", *branches, check=False)
        await self._git("reset", "--hard", check=False)
        await self._git("clean", "-fd", check=False)
        shutil.rmtree(self.worktrees_root, ignore_errors=True)

    async def _worktrees_under_root(self) -> list[Path]:
        """Registered worktree paths that live under :attr:`worktrees_root`."""
        out = await self._git("worktree", "list", "--porcelain", check=False)
        root = self.worktrees_root.resolve()
        paths: list[Path] = []
        for line in out.splitlines():
            if not line.startswith("worktree "):
                continue
            wt = Path(line[len("worktree ") :].strip())
            resolved = wt.resolve()
            if resolved == root or root in resolved.parents:
                paths.append(wt)
        return paths

    async def _node_branches(self) -> list[str]:
        """Local branch names carrying this provider's ``branch_prefix``."""
        out = await self._git(
            "for-each-ref",
            "--format=%(refname:short)",
            f"refs/heads/{self.branch_prefix}",
            check=False,
        )
        return [line.strip() for line in out.splitlines() if line.strip()]

    async def prepare(self, node: Node, dep_nodes: list[Node], cwd: Path) -> MergeReport:
        """Git-merge each dependency's branch into ``node``'s worktree.

        The concrete :meth:`IsolationProvider.prepare
        <agent_flow.orchestration.isolation.IsolationProvider.prepare>` hook for a
        fan-in / ``merge`` node. Ensures ``node``'s worktree exists (acquiring it
        if it has not been yet), then merges every dependency branch —
        ``<branch_prefix><dep.id>``, the exact branch :meth:`acquire` creates for
        that dependency — into it, one at a time.

        A clean merge records the branch in :attr:`MergeReport.merged`. A merge
        that *conflicts* is not an error: the conflict markers are left in the
        worktree (the merge is **not** aborted), the unmerged paths are recorded
        in :attr:`MergeReport.conflicted_paths`, and merging stops — the worktree
        now holds an in-progress merge that a later resolver must settle before
        any further branch can be folded in. Only a genuinely unexpected git
        failure (a bad branch ref, a missing repo) raises :class:`IsolationError`.

        Args:
            node: The ``kind == "merge"`` node whose worktree receives the merges.
            dep_nodes: The dependency nodes whose branches are merged in order.
            cwd: The workspace the scheduler acquired for ``node``. Accepted for
                Protocol conformance; the worktree is derived deterministically
                from ``node`` (and self-acquired if absent), so it always matches
                the ``cwd`` the scheduler passes for a worktree node.

        Returns:
            A :class:`MergeReport` describing what merged and what conflicted.

        Raises:
            IsolationError: If a ``git merge`` fails for a reason other than a
                content conflict (e.g. an unknown branch), or if acquiring the
                worktree fails.
        """
        worktree = await self._ensure_worktree(node)
        report = MergeReport()
        for dep in dep_nodes:
            branch = self._branch(dep)
            if not await self._branch_exists(branch):
                # A reclaimed dependency branch: the dep went DONE and its branch
                # was freed once its dependents finished, but its work is already
                # integrated into the base checkout. A dependent re-running after a
                # replan reset therefore has nothing to merge for it — record the
                # skip and proceed rather than treating the absent branch as a
                # failure. (The scheduler only runs a node once its deps are DONE,
                # so an absent branch is always a reclaimed-and-integrated one.)
                report.skipped.append(branch)
                continue
            returncode, _stdout, stderr = await self._run_git(
                "merge", "--no-edit", branch, cwd=worktree
            )
            if returncode == 0:
                report.merged.append(branch)
                continue
            # Non-zero exit: a content conflict leaves unmerged paths behind,
            # whereas a real failure (bad ref, not a repo) leaves none.
            conflicted = await self._conflicted_paths(worktree)
            if not conflicted:
                raise IsolationError(
                    f"git merge {branch} failed (exit code {returncode}): {stderr.strip()}"
                )
            report.conflicted_paths.extend(conflicted)
            # Leave the in-progress merge in place for a later resolver; a further
            # ``git merge`` here would only fail with "merge already in progress".
            break
        return report

    async def _ensure_worktree(self, node: Node) -> Path:
        """Return ``node``'s worktree path, acquiring it only if it is absent.

        Shared nodes always resolve to ``repo_path``; worktree nodes reuse an
        already-acquired directory (as when the scheduler acquired it before
        seeding) and otherwise acquire a fresh one.
        """
        if node.isolation == "shared":
            return self.repo_path
        path = self._worktree_path(node)
        if not path.exists():
            return await self.acquire(node)
        return path

    async def _branch_exists(self, branch: str) -> bool:
        """Whether ``branch`` currently exists as a local ref in the repo."""
        returncode, _stdout, _stderr = await self._run_git(
            "show-ref", "--verify", "--quiet", f"refs/heads/{branch}"
        )
        return returncode == 0

    async def _conflicted_paths(self, worktree: Path) -> list[str]:
        """Repo-relative paths with unresolved conflict markers in ``worktree``."""
        out = await self._git("diff", "--name-only", "--diff-filter=U", cwd=worktree)
        return [line for line in out.splitlines() if line.strip()]

    async def _git(self, *args: str, check: bool = True, cwd: Path | None = None) -> str:
        """Run ``git -C <cwd or repo_path> <args>`` and return its stdout.

        Args:
            *args: Arguments passed to git after ``-C <location>``.
            check: When true, raise :class:`IsolationError` on a non-zero exit.
            cwd: Directory to run git in; defaults to ``repo_path``. Passing a
                worktree path targets that worktree's checkout.

        Returns:
            The command's decoded stdout.

        Raises:
            IsolationError: If ``check`` is true and git exits non-zero.
        """
        returncode, stdout, stderr = await self._run_git(*args, cwd=cwd)
        if check and returncode != 0:
            raise IsolationError(
                f"git {' '.join(args)} failed (exit code {returncode}): {stderr.strip()}"
            )
        return stdout

    async def _run_git(self, *args: str, cwd: Path | None = None) -> tuple[int, str, str]:
        """Run git and return ``(returncode, stdout, stderr)`` without raising.

        The low-level primitive behind :meth:`_git`. Callers that must inspect a
        non-zero exit (a merge conflict is exit 1 but not an error) use this to
        decide themselves; :meth:`_git` is the raise-on-failure convenience.
        """
        location = str(cwd) if cwd is not None else str(self.repo_path)
        proc = await asyncio.create_subprocess_exec(
            "git",
            "-C",
            location,
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()
        returncode = proc.returncode if proc.returncode is not None else -1
        return returncode, stdout.decode(errors="replace"), stderr.decode(errors="replace")
