# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generic per-node workspace isolation for the DAG engine.

This is the core's *only* isolation surface, and it is deliberately mechanism-
free: it defines the :class:`IsolationProvider` Protocol a scheduler talks to and
a trivial :class:`NoOpIsolation` for DAG workflows that need no isolation at all.
It knows nothing about git, worktrees, or merges — a provider decides *where* a
node runs and may seed a fan-in node's workspace, but *how* it does so lives
entirely in the (opt-in) concrete provider, never here.

Like the rest of ``orchestration`` this layer depends only on the standard
library and the pure :class:`~agent_flow.orchestration.graph.Node` model, and
must never import agent, backend, ``layers``, or any concrete-isolation
(e.g. :mod:`agent_flow.git_worktree`) code. The dependency arrow points one way:
concrete providers import this module; this module imports none of them.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

from .graph import Node

__all__ = [
    "IsolationError",
    "IsolationProvider",
    "NoOpIsolation",
]


class IsolationError(RuntimeError):
    """Raised when an isolation operation fails.

    The message should name the operation that failed and carry enough context
    (an exit code, captured stderr, a bad path) for a caller to see exactly why.
    """


@runtime_checkable
class IsolationProvider(Protocol):
    """Supplies and reclaims a working directory for a node's execution.

    Implementations decide *where* a node runs. :meth:`acquire` returns the
    directory the node should treat as its workspace; :meth:`release` frees the
    node's per-turn/working resources once its own work is finished; and
    :meth:`reclaim` frees the node's *durable* output once no remaining dependent
    still needs it. :meth:`prepare` is an optional pre-run hook for fan-in nodes.
    All are async so providers can shell out without blocking the scheduler's
    event loop.

    The split between :meth:`release` and :meth:`reclaim` exists so a node's
    durable output can outlive the node itself: a fan-in node cannot consume a
    dependency's output if that output was destroyed the moment the dependency
    finished. The scheduler therefore calls :meth:`release` on a node right after
    its run but calls :meth:`reclaim` only once every node depending on it is
    terminal.
    """

    async def acquire(self, node: Node) -> Path:
        """Return the working directory ``node`` should execute in."""
        ...

    async def release(self, node: Node) -> None:
        """Free ``node``'s working/per-turn resources; its durable output remains.

        Called right after ``node``'s run finishes. It reclaims only the
        transient workspace (e.g. a checked-out working tree); whatever durable
        output the node produced (e.g. a branch a later fan-in node will merge)
        must survive until :meth:`reclaim`.
        """
        ...

    async def reclaim(self, node: Node) -> None:
        """Free ``node``'s durable output — no remaining dependent needs it.

        Called once every node that depends on ``node`` has reached a terminal
        state (or immediately on ``node``'s own terminal state when nothing
        depends on it). Implementations should be idempotent: the scheduler
        guarantees at most one call per node, but reclaiming an already-freed or
        never-produced output must not raise.
        """
        ...

    async def prepare(self, node: Node, dep_nodes: list[Node], cwd: Path) -> object | None:
        """Seed a fan-in node's workspace from its dependencies before it runs.

        The scheduler calls this for a fan-in / ``merge`` node after
        :meth:`acquire` and before ``run_node``, passing the node, its resolved
        dependency nodes, and the workspace ``acquire`` returned (``cwd``). A
        provider may use it to fold each dependency's work into ``cwd``.

        The default semantics are a no-op that returns ``None``. Any non-``None``
        value is opaque to the scheduler and simply collected onto
        ``GraphResult.prepare_results`` keyed by node id, so a provider can
        report what it did without changing the ``run_node`` signature.
        """
        ...

    async def commit(self, node: Node, cwd: Path) -> object | None:
        """Persist ``node``'s work to its durable output before :meth:`release`.

        The scheduler calls this after ``run_node`` reaches a terminal state
        (``DONE`` or ``FAILED``) and before :meth:`release`, passing the node and
        the workspace :meth:`acquire` returned (``cwd``). A provider may use it to
        capture the node's working-tree changes into its durable output (e.g.
        commit them onto the node's branch) so a dependent can later consume them
        and nothing is lost when the transient workspace is torn down.

        The default semantics are a no-op returning ``None``; any return value is
        opaque to the scheduler.
        """
        ...


class NoOpIsolation:
    """Isolation provider that runs every node in one shared base directory.

    For DAG-only workflows (a research fan-out, a data-processing graph) that
    need the concurrent engine but no per-node isolation at all: :meth:`acquire`
    returns ``base_cwd`` for every node regardless of its ``isolation`` mode, and
    :meth:`release`, :meth:`reclaim`, and :meth:`prepare` are no-ops that return
    ``None``. It uses no git and creates nothing on disk.
    """

    def __init__(self, base_cwd: Path) -> None:
        """Configure the provider.

        Args:
            base_cwd: The single directory every node executes in.
        """
        self.base_cwd = Path(base_cwd)

    async def acquire(self, node: Node) -> Path:
        """Return the shared ``base_cwd`` for ``node``."""
        return self.base_cwd

    async def release(self, node: Node) -> None:
        """Reclaim nothing — there is no per-node workspace to tear down."""
        return None

    async def reclaim(self, node: Node) -> None:
        """Free nothing — a DAG-only run produces no durable per-node output."""
        return None

    async def prepare(self, node: Node, dep_nodes: list[Node], cwd: Path) -> object | None:
        """Seed nothing — a DAG-only run has no dependency workspaces to fold in."""
        return None

    async def commit(self, node: Node, cwd: Path) -> object | None:
        """Persist nothing — a DAG-only run produces no durable per-node output."""
        return None
