# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Concurrent ready-set scheduler for the execution graph.

This is the core of the generic DAG engine. :class:`NodeScheduler` walks the
*flattened* node set from :class:`~agent_flow.orchestration.graph.ExecutionGraph`
and runs every node whose ``depends_on`` predecessors are ``DONE``, up to
``max_parallel`` at a time. Each node's work is delegated to an injected
``run_node`` callback, so the scheduler stays completely workflow-agnostic: like
the rest of ``orchestration`` it must never import agent, backend, or ``layers``
code.

The scheduler operates on the flat node set (``graph.flatten()``) and schedules
each node by its own ``depends_on`` sibling edges **plus** nested containment: a
parent node (e.g. a Stage) is a convergence gate over its own sub-DAG, so it
becomes ready only once every child (its Goals) is DONE, and is BLOCKED if any
child fails. Children are otherwise scheduled independently by their own
``depends_on``. A fan-in node (``kind == "merge"``) is *prepared* before it runs:
the injected :class:`~agent_flow.orchestration.isolation.IsolationProvider` is
given a chance to seed the node's workspace from its dependencies (via
``IsolationProvider.prepare``) *before* ``run_node`` sees that workspace. The
scheduler stays isolation-mechanism-agnostic — it never knows or cares *how* a
provider seeds the workspace (git merge, copy, nothing).

State machine (per node):

* ``PENDING``  — not yet started; the initial state of every node.
* ``RUNNING``  — a worker holds the node; ``run_node`` is executing.
* ``DONE``     — ``run_node`` returned ``NodeOutcome(terminal_state=DONE)``.
* ``FAILED``   — ``run_node`` returned ``NodeOutcome(terminal_state=FAILED)``.
* ``BLOCKED``  — a (transitive) ``depends_on`` predecessor is ``FAILED``/``BLOCKED``;
  the node is never run.

A ``GraphState`` checkpoint is written atomically on every transition, so a crash
never loses more than the in-flight work. Constructing a new scheduler with the
same ``checkpoint_path`` resumes: already-``DONE`` and ``FAILED`` nodes are kept,
everything else is reset to ``PENDING`` and re-derived, and ``DONE`` nodes are
therefore never re-invoked.
"""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path

from .graph import ExecutionGraph, Node, NodeState
from .isolation import IsolationProvider

__all__ = [
    "GraphResult",
    "GraphState",
    "NodeOutcome",
    "NodeScheduler",
    "OnTransition",
    "RunNode",
]

SCHEMA_VERSION = 1
"""On-disk schema version for a persisted :class:`GraphState`."""


@dataclass
class NodeOutcome:
    """The result a ``run_node`` callback reports for a single node.

    Attributes:
        terminal_state: The state the node ends in — ``DONE`` or ``FAILED``.
        needs_replan: Set by the callback to signal the surrounding workflow
            that the graph should be replanned. The scheduler only surfaces this
            flag on the :class:`GraphResult`; it does not act on it here.
        info: Optional free-form detail (diagnostics, metrics) attached by the
            callback.
    """

    terminal_state: NodeState
    needs_replan: bool = False
    info: dict | None = None


# An async callback that performs a node's work in a given working directory and
# reports how it ended. The scheduler is fully parameterized by this callable and
# knows nothing about agents or backends.
RunNode = Callable[[Node, Path], Awaitable[NodeOutcome]]

# A synchronous observer invoked as ``(node_id, from_state, to_state)`` on every
# runtime state transition. Handy for logging or a live UI.
OnTransition = Callable[[str, NodeState, NodeState], None]


def _atomic_write_json(path: Path, payload: dict) -> None:
    """Write ``payload`` as JSON to ``path`` atomically (temp file + rename).

    Mirrors ``agent_flow.workflows.agent_team.state.save_state``: the JSON is
    written to a temp file in the same directory, flushed and ``fsync``-ed, then
    ``os.replace``-d onto the target so a crash never leaves a half-written
    checkpoint. The temp file is removed if anything goes wrong.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_path, path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except FileNotFoundError:
            pass
        raise


@dataclass
class GraphState:
    """Serializable snapshot of a run, sufficient to resume after a crash.

    Attributes:
        states: Per-node lifecycle state keyed by node id.
        worktrees: Per-node working-directory path (as a string) recorded when a
            node's isolation was acquired. In-flight nodes are those whose state
            is ``RUNNING``.
    """

    states: dict[str, NodeState]
    worktrees: dict[str, str] = field(default_factory=dict)

    def save(self, path: Path) -> None:
        """Atomically persist this state to ``path`` as JSON."""
        payload = {
            "version": SCHEMA_VERSION,
            "states": {node_id: state.value for node_id, state in self.states.items()},
            "worktrees": dict(self.worktrees),
        }
        _atomic_write_json(Path(path), payload)

    @classmethod
    def load(cls, path: Path) -> GraphState:
        """Reconstruct a :class:`GraphState` from a JSON checkpoint at ``path``.

        Raises:
            ValueError: If the checkpoint's schema version is unsupported.
        """
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        version = data.get("version")
        if version != SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported checkpoint version {version!r} in {path}; "
                f"expected {SCHEMA_VERSION}. Delete the file to start fresh."
            )
        states = {node_id: NodeState(value) for node_id, value in data["states"].items()}
        worktrees = {node_id: str(value) for node_id, value in data.get("worktrees", {}).items()}
        return cls(states=states, worktrees=worktrees)


@dataclass(frozen=True)
class GraphResult:
    """Terminal outcome of a :meth:`NodeScheduler.run`.

    Attributes:
        states: Final lifecycle state of every node in the graph.
        outcomes: The :class:`NodeOutcome` for each node that actually ran
            (``DONE``/``FAILED``); nodes left ``BLOCKED`` have no entry.
        prepare_results: The opaque, non-``None`` value each fan-in
            (``kind == "merge"``) node's ``IsolationProvider.prepare`` hook
            returned, keyed by node id. This is the isolation-agnostic channel by
            which a provider reports what it seeded (e.g. a merge report) without
            changing the ``run_node(node, cwd)`` signature; the scheduler treats
            every value as an opaque ``object`` and never inspects it.
    """

    states: dict[str, NodeState]
    outcomes: dict[str, NodeOutcome]
    prepare_results: dict[str, object] = field(default_factory=dict)

    @property
    def succeeded(self) -> bool:
        """True iff every node in the graph ended ``DONE``."""
        return all(state == NodeState.DONE for state in self.states.values())

    @property
    def done_ids(self) -> tuple[str, ...]:
        """Ids of nodes that completed successfully, in graph order."""
        return tuple(nid for nid, state in self.states.items() if state == NodeState.DONE)

    @property
    def failed_ids(self) -> tuple[str, ...]:
        """Ids of nodes whose ``run_node`` reported failure, in graph order."""
        return tuple(nid for nid, state in self.states.items() if state == NodeState.FAILED)

    @property
    def blocked_ids(self) -> tuple[str, ...]:
        """Ids of nodes never run because a predecessor failed, in graph order."""
        return tuple(nid for nid, state in self.states.items() if state == NodeState.BLOCKED)

    @property
    def interrupted_ids(self) -> tuple[str, ...]:
        """Ids paused mid-run (design's SUSPENDED); re-dispatch + re-attach on resume."""
        return tuple(nid for nid, s in self.states.items() if s == NodeState.INTERRUPTED)

    @property
    def paused(self) -> bool:
        """True iff the run paused with at least one node left INTERRUPTED."""
        return any(s == NodeState.INTERRUPTED for s in self.states.values())

    @property
    def needs_replan(self) -> bool:
        """True iff any node's outcome requested a replan."""
        return any(outcome.needs_replan for outcome in self.outcomes.values())


class NodeScheduler:
    """Run an :class:`ExecutionGraph` concurrently with checkpoint/resume.

    Independent nodes run in parallel up to ``max_parallel`` (an
    :class:`asyncio.Semaphore`); every node's isolation is acquired around its
    ``run_node`` call and released afterward; and a :class:`GraphState`
    checkpoint is written on each transition when ``checkpoint_path`` is set.

    Isolation is torn down in two stages so a node's durable output can outlive
    the node. ``release`` runs right after each node's ``run_node`` (freeing its
    working resources), but ``reclaim`` — which frees the node's durable output —
    is deferred until every node that ``depends_on`` it is terminal, so a fan-in
    node can still consume its dependencies' output. The scheduler tracks, per
    node, how many not-yet-terminal dependents remain, reclaims a node the moment
    that count reaches zero (a dependent-less node reclaims on its own terminal),
    reclaims each node at most once, and sweep-reclaims anything still unreclaimed
    when :meth:`run` returns.
    """

    def __init__(
        self,
        graph: ExecutionGraph,
        run_node: RunNode,
        isolation: IsolationProvider,
        *,
        on_transition: OnTransition | None = None,
        max_parallel: int = 8,
        checkpoint_path: Path | None = None,
        pause_on_failure: bool = False,
    ) -> None:
        """Configure the scheduler.

        Args:
            graph: The validated execution graph to run.
            run_node: Async callback that performs a node's work in a given
                directory and returns a :class:`NodeOutcome`.
            isolation: Provider that supplies and reclaims each node's workspace.
            on_transition: Optional ``(node_id, from_state, to_state)`` hook
                invoked on every runtime state change.
            max_parallel: Maximum number of nodes executing concurrently.
            checkpoint_path: Where to persist/resume the :class:`GraphState`;
                ``None`` disables persistence (and therefore resume).
            pause_on_failure: When ``True``, the first node that reports ``FAILED``
                stops the pass: healthy in-flight nodes are cancelled and left
                ``INTERRUPTED`` (their detached jobs survive) instead of draining
                the rest of the pass. Defaults ``False`` — the current behavior.
        """
        self._graph = graph
        self._run_node = run_node
        self._isolation = isolation
        self._on_transition = on_transition
        self._max_parallel = max_parallel
        self._checkpoint_path = Path(checkpoint_path) if checkpoint_path is not None else None
        self._pause_on_failure = pause_on_failure
        self._paused = False

        self._nodes = graph.flatten()
        # Cross-stage gating: ``depends_on`` is same-level-siblings only, so a
        # stage expresses "run after stage X" via its own ``depends_on``. That
        # dependency must gate the stage's whole subtree — not just the stage gate
        # node — so a goal is gated on its own ``depends_on`` PLUS every ancestor's.
        # Precompute each node's inherited (ancestor) ``depends_on`` once.
        self._ancestor_deps: dict[str, tuple[str, ...]] = {}
        self._index_ancestor_deps(graph.nodes, ())
        self._states: dict[str, NodeState] = {node.id: NodeState.PENDING for node in self._nodes}
        self._worktrees: dict[str, str] = {}
        self._outcomes: dict[str, NodeOutcome] = {}
        self._prepare_results: dict[str, object] = {}
        self._semaphore: asyncio.Semaphore | None = None

        # Dependency-lifetime reclamation bookkeeping. A node's durable output is
        # reclaimed only once every node that ``depends_on`` it is terminal, so
        # count each node's dependents up front and decrement as they finish.
        self._remaining_dependents: dict[str, int] = dict.fromkeys(
            (node.id for node in self._nodes), 0
        )
        for node in self._nodes:
            for dep in node.depends_on:
                if dep in self._remaining_dependents:
                    self._remaining_dependents[dep] += 1
        # Nodes whose terminal state we have already accounted for (decremented
        # their dependencies) and nodes already reclaimed — both idempotency guards.
        self._terminal_counted: set[str] = set()
        self._reclaimed: set[str] = set()

    def state(self) -> GraphState:
        """Return a snapshot of the current per-node states and worktree paths."""
        return GraphState(states=dict(self._states), worktrees=dict(self._worktrees))

    async def run(self) -> GraphResult:
        """Execute the graph to completion and return a :class:`GraphResult`.

        Resumes from ``checkpoint_path`` when it exists (already-``DONE`` and
        ``FAILED`` nodes are preserved; everything else is reset to ``PENDING``).
        Runs the ready set concurrently under a ``max_parallel`` semaphore,
        checkpointing on every transition, until no node is runnable — at which
        point every node is ``DONE``, ``FAILED``, or ``BLOCKED``.
        """
        self._resume_from_checkpoint()
        self._semaphore = asyncio.Semaphore(self._max_parallel)
        # Re-derive BLOCKED nodes (e.g. dependents of a resumed FAILED node) and
        # persist the normalized starting state.
        self._propagate_blocked()
        self._checkpoint()
        # Reclaim any node already terminal on resume whose dependents are all done.
        await self._reclaim_terminal_nodes()

        self._paused = False
        in_flight: dict[asyncio.Task[NodeOutcome], str] = {}
        try:
            while True:
                if not self._paused:
                    for node in self._ready_nodes():
                        self._transition(node.id, NodeState.RUNNING)
                        task = asyncio.create_task(self._run_one(node))
                        in_flight[task] = node.id

                if not in_flight:
                    break

                done, _ = await asyncio.wait(set(in_flight), return_when=asyncio.FIRST_COMPLETED)
                for task in done:
                    node_id = in_flight.pop(task)
                    outcome = task.result()  # re-raises if the callback raised
                    # ``run_node`` must report a terminal state; a non-terminal
                    # one (e.g. PENDING) would make the node ready again and
                    # re-run it forever, so fail fast on the contract violation.
                    if outcome.terminal_state not in (NodeState.DONE, NodeState.FAILED):
                        raise ValueError(
                            f"run_node for node {node_id!r} returned non-terminal "
                            f"terminal_state {outcome.terminal_state.name}; "
                            f"expected {NodeState.DONE.name} or {NodeState.FAILED.name}."
                        )
                    self._outcomes[node_id] = outcome
                    self._transition(node_id, outcome.terminal_state)
                    if outcome.terminal_state == NodeState.FAILED:
                        self._propagate_blocked()
                        if self._pause_on_failure:
                            self._paused = True

                # Pause-on-failure: stop the pass the moment a node fails —
                # suspend the healthy in-flight nodes (INTERRUPTED) instead of
                # draining them, then break out without dispatching more.
                if self._paused and in_flight:
                    await self._suspend_in_flight(in_flight)
                    break

                # Newly-terminal nodes (this batch, plus any just BLOCKED) may free
                # a dependency whose last dependent has now finished — reclaim them.
                await self._reclaim_terminal_nodes()
        except BaseException:
            # Do not leak orphaned tasks if a callback raised or we were cancelled.
            for task in in_flight:
                task.cancel()
            if in_flight:
                await asyncio.gather(*in_flight, return_exceptions=True)
            raise

        # Final sweep: on a genuine drain every node is terminal, so free anything
        # still holding durable output. SKIP on a pause — its DONE dependencies'
        # and INTERRUPTED nodes' branches are exactly what a resume must re-attach
        # to / merge, so reclaiming them here would silently lose them.
        if not self._paused:
            await self._sweep_reclaim()

        return GraphResult(
            states=dict(self._states),
            outcomes=dict(self._outcomes),
            prepare_results=dict(self._prepare_results),
        )

    async def _run_one(self, node: Node) -> NodeOutcome:
        """Acquire ``node``'s isolation, seed a dependent, run it, commit, release.

        The ``max_parallel`` semaphore bounds how many nodes hold isolation and
        execute ``run_node`` at once. For any node that declares ``depends_on``
        the isolation provider's ``prepare`` hook runs on its freshly acquired
        workspace *before* ``run_node`` sees it, seeding it from its dependency
        nodes' output; any non-``None`` value it returns is recorded (opaquely)
        for the :class:`GraphResult`. After ``run_node`` returns a terminal state
        and *before* ``release`` tears the workspace down, the provider's
        ``commit`` hook persists the node's work to its durable output so a
        dependent can consume it. What each hook does (or whether a provider even
        implements one) is entirely up to it; the scheduler stays mechanism-agnostic.
        """
        assert self._semaphore is not None  # set at the top of ``run``
        async with self._semaphore:
            path = await self._isolation.acquire(node)
            # Record where the node ran while it is in flight, so a crash-time
            # checkpoint knows its working directory.
            self._worktrees[node.id] = str(path)
            self._checkpoint()
            try:
                if node.depends_on:
                    # Any node with dependencies is seeded from them before it
                    # runs, so ``depends_on`` carries the dependencies' committed
                    # work (not just execution order) into this node's workspace.
                    dep_nodes = [self._graph.by_id[dep] for dep in node.depends_on]
                    prepared = await self._isolation.prepare(node, dep_nodes, path)
                    if prepared is not None:
                        self._prepare_results[node.id] = prepared
                outcome = await self._run_node(node, path)
                # Persist the node's work to its durable output on ANY terminal
                # state (DONE or FAILED), before ``release`` tears the workspace
                # down, so a dependent can consume it and nothing is lost.
                await self._isolation.commit(node, path)
                return outcome
            finally:
                await self._isolation.release(node)

    async def _suspend_in_flight(self, in_flight: dict[asyncio.Task[NodeOutcome], str]) -> None:
        """Cancel every still-running node task and mark it INTERRUPTED (suspended).

        Pause-on-failure stops the pass the moment a node fails: the healthy
        in-flight nodes are cancelled at their next await (killing the agent turn
        + any framework poll loop) but their DETACHED jobs and ``nodes/<id>/``
        state — including ``jobs.json`` — survive, so a later resume re-dispatches
        them and ``run_detached`` re-attaches the live job instead of re-running.
        """
        for task in in_flight:
            task.cancel()
        await asyncio.gather(*list(in_flight), return_exceptions=True)
        for node_id in in_flight.values():
            self._transition(node_id, NodeState.INTERRUPTED)
        in_flight.clear()

    def _index_ancestor_deps(self, nodes: tuple[Node, ...], inherited: tuple[str, ...]) -> None:
        """Record, for each node, the ``depends_on`` of all its ancestors.

        Walks the forest top-down carrying the accumulated ancestor ``depends_on``
        so :meth:`_effective_deps` can gate a node on its ancestors' cross-stage
        dependencies as well as its own siblings.
        """
        for node in nodes:
            self._ancestor_deps[node.id] = inherited
            self._index_ancestor_deps(node.children, inherited + tuple(node.depends_on))

    def _effective_deps(self, node: Node) -> tuple[str, ...]:
        """A node's own ``depends_on`` plus every ancestor stage's ``depends_on``."""
        return tuple(node.depends_on) + self._ancestor_deps[node.id]

    def _ready_nodes(self) -> list[Node]:
        """Return PENDING nodes whose effective deps AND children are all DONE.

        A node's *effective* deps are its own ``depends_on`` siblings plus every
        ancestor stage's ``depends_on`` (:meth:`_effective_deps`), so a stage's
        cross-stage dependency gates its whole subtree, not just the stage gate node.
        """
        ready: list[Node] = []
        for node in self._nodes:
            if self._states[node.id] != NodeState.PENDING:
                continue
            if not all(self._states[dep] == NodeState.DONE for dep in self._effective_deps(node)):
                continue
            if not all(self._states[child.id] == NodeState.DONE for child in node.children):
                continue
            ready.append(node)
        return ready

    def _propagate_blocked(self) -> None:
        """Block every PENDING node with a FAILED/BLOCKED effective dep or child.

        A node's effective deps include its ancestors' ``depends_on``
        (:meth:`_effective_deps`), so a failed stage dependency cascades down to
        that stage's goals rather than leaving them dangling PENDING. Iterates to a
        fixpoint so blocking cascades transitively down the graph.
        """
        blocking = (NodeState.FAILED, NodeState.BLOCKED)
        changed = True
        while changed:
            changed = False
            for node in self._nodes:
                if self._states[node.id] != NodeState.PENDING:
                    continue
                dep_blocked = any(
                    self._states[dep] in blocking for dep in self._effective_deps(node)
                )
                child_blocked = any(self._states[c.id] in blocking for c in node.children)
                if dep_blocked or child_blocked:
                    self._transition(node.id, NodeState.BLOCKED)
                    changed = True

    async def _reclaim_terminal_nodes(self) -> None:
        """Account for newly-terminal nodes and reclaim any whose dependents are done.

        Two passes over the flat node set. First, for every node that has *newly*
        reached a terminal state (``DONE``/``FAILED``/``BLOCKED``), decrement the
        remaining-dependent count of each of its dependencies — counting each node
        exactly once via :attr:`_terminal_counted`. Then reclaim every terminal
        node whose remaining-dependent count has reached zero — a dependent-less
        node therefore reclaims the moment it goes terminal. Reclaim is guarded to
        fire at most once per node.
        """
        terminal = (NodeState.DONE, NodeState.FAILED, NodeState.BLOCKED)
        # Pass 1: for each newly-terminal node, decrement its dependencies' counts.
        for node in self._nodes:
            if node.id in self._terminal_counted:
                continue
            if self._states[node.id] not in terminal:
                continue
            self._terminal_counted.add(node.id)
            for dep in node.depends_on:
                if dep in self._remaining_dependents:
                    self._remaining_dependents[dep] -= 1
        # Pass 2: reclaim every terminal node with no remaining dependents. A node
        # is only reclaimed once it is itself terminal, so a still-running
        # dependency is never freed early even if a BLOCKED dependent dropped its
        # count to zero.
        for node in self._nodes:
            if node.id in self._reclaimed:
                continue
            if self._states[node.id] not in terminal:
                continue
            if self._remaining_dependents[node.id] > 0:
                continue
            await self._reclaim(node)

    async def _sweep_reclaim(self) -> None:
        """Reclaim every still-unreclaimed node — a final sweep on ``run`` return."""
        for node in self._nodes:
            await self._reclaim(node)

    async def _reclaim(self, node: Node) -> None:
        """Free ``node``'s durable output via the isolation provider, at most once."""
        if node.id in self._reclaimed:
            return
        self._reclaimed.add(node.id)
        await self._isolation.reclaim(node)

    def _transition(self, node_id: str, new_state: NodeState) -> None:
        """Move ``node_id`` to ``new_state``, notify the hook, and checkpoint."""
        old_state = self._states[node_id]
        if old_state == new_state:
            return
        self._states[node_id] = new_state
        if self._on_transition is not None:
            self._on_transition(node_id, old_state, new_state)
        self._checkpoint()

    def _checkpoint(self) -> None:
        """Atomically persist the current :class:`GraphState`, if enabled."""
        if self._checkpoint_path is None:
            return
        self.state().save(self._checkpoint_path)

    def _resume_from_checkpoint(self) -> None:
        """Load a prior checkpoint and normalize states for a fresh run.

        ``DONE`` and ``FAILED`` nodes are preserved (so ``DONE`` nodes are never
        re-invoked); any node caught mid-flight (``RUNNING``) or otherwise
        non-terminal is reset to ``PENDING`` to be re-derived and, if runnable,
        re-run. Unknown ids in the checkpoint (graph changed) are ignored.
        """
        if self._checkpoint_path is None or not self._checkpoint_path.exists():
            return
        loaded = GraphState.load(self._checkpoint_path)
        for node_id, state in loaded.states.items():
            if node_id not in self._states:
                continue
            self._states[node_id] = (
                state if state in (NodeState.DONE, NodeState.FAILED) else (NodeState.PENDING)
            )
        for node_id, path in loaded.worktrees.items():
            if node_id in self._states:
                self._worktrees[node_id] = path
