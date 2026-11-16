"""Tests for the git-free isolation abstraction in the orchestration core.

Covers the generic :class:`IsolationProvider` Protocol (a minimal
acquire/release/prepare/reclaim stub conforms; a class missing ``prepare`` or
``reclaim`` does not) and :class:`NoOpIsolation`, the built-in provider for
DAG-only workflows that need no isolation: ``acquire`` returns the shared base
directory for every node regardless of its ``isolation`` mode, and
``release``/``prepare``/``reclaim`` are no-ops that return ``None``. No git, no
worktrees — the concrete git provider is tested in ``tests/test_git_worktree.py``.
"""

from __future__ import annotations

from pathlib import Path

import anyio

from agent_flow.orchestration.graph import Node
from agent_flow.orchestration.isolation import IsolationProvider, NoOpIsolation


def test_noop_conforms_to_protocol(tmp_path):
    """NoOpIsolation satisfies the runtime-checkable IsolationProvider."""
    assert isinstance(NoOpIsolation(tmp_path), IsolationProvider)


def test_minimal_acquire_release_prepare_reclaim_commit_stub_conforms():
    """Any object with acquire/release/prepare/reclaim/commit conforms to the Protocol."""

    class _Stub:
        async def acquire(self, node: Node) -> Path:
            return Path(".")

        async def release(self, node: Node) -> None:
            return None

        async def prepare(self, node: Node, dep_nodes: list[Node], cwd: Path) -> object | None:
            return None

        async def reclaim(self, node: Node) -> None:
            return None

        async def commit(self, node: Node, cwd: Path) -> object | None:
            return None

    assert isinstance(_Stub(), IsolationProvider)


def test_provider_missing_commit_does_not_conform():
    """The Protocol requires ``commit``; a class lacking it is not an instance."""

    class _NoCommit:
        async def acquire(self, node: Node) -> Path:
            return Path(".")

        async def release(self, node: Node) -> None:
            return None

        async def prepare(self, node: Node, dep_nodes: list[Node], cwd: Path) -> object | None:
            return None

        async def reclaim(self, node: Node) -> None:
            return None

    assert not isinstance(_NoCommit(), IsolationProvider)


def test_provider_missing_prepare_does_not_conform():
    """The Protocol requires ``prepare``; a class lacking it is not an instance."""

    class _NoPrepare:
        async def acquire(self, node: Node) -> Path:
            return Path(".")

        async def release(self, node: Node) -> None:
            return None

        async def reclaim(self, node: Node) -> None:
            return None

    assert not isinstance(_NoPrepare(), IsolationProvider)


def test_provider_missing_reclaim_does_not_conform():
    """The Protocol requires ``reclaim``; a class lacking it is not an instance."""

    class _NoReclaim:
        async def acquire(self, node: Node) -> Path:
            return Path(".")

        async def release(self, node: Node) -> None:
            return None

        async def prepare(self, node: Node, dep_nodes: list[Node], cwd: Path) -> object | None:
            return None

    assert not isinstance(_NoReclaim(), IsolationProvider)


def test_noop_acquire_returns_base_cwd_for_every_node(tmp_path):
    """Acquire hands back ``base_cwd`` regardless of the node's isolation mode."""
    base = tmp_path / "workspace"
    base.mkdir()
    iso = NoOpIsolation(base)

    worktree_node = Node(id="a", type="impl")  # default isolation == "worktree"
    shared_node = Node(id="b", type="impl", isolation="shared")

    assert anyio.run(iso.acquire, worktree_node) == base
    assert anyio.run(iso.acquire, shared_node) == base
    # No per-node subdirectories are created; everything shares ``base_cwd``.
    assert list(base.iterdir()) == []


def test_noop_release_is_a_noop(tmp_path):
    """Release returns None and leaves the base directory untouched."""
    base = tmp_path / "workspace"
    base.mkdir()
    (base / "keep.txt").write_text("keep\n")
    iso = NoOpIsolation(base)

    assert anyio.run(iso.release, Node(id="a", type="impl")) is None
    assert (base / "keep.txt").read_text() == "keep\n"


def test_noop_prepare_returns_none(tmp_path):
    """Prepare is a no-op returning None (nothing to seed for a DAG-only run)."""
    iso = NoOpIsolation(tmp_path)
    merge_node = Node(id="m", type="merge", kind="merge", depends_on=("a", "b"))
    dep_nodes = [Node(id="a", type="impl"), Node(id="b", type="impl")]

    result = anyio.run(iso.prepare, merge_node, dep_nodes, tmp_path)
    assert result is None


def test_noop_reclaim_is_a_noop(tmp_path):
    """Reclaim returns None and leaves the base directory untouched (no durable output)."""
    base = tmp_path / "workspace"
    base.mkdir()
    (base / "keep.txt").write_text("keep\n")
    iso = NoOpIsolation(base)

    assert anyio.run(iso.reclaim, Node(id="a", type="impl")) is None
    assert (base / "keep.txt").read_text() == "keep\n"
