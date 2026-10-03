"""Tests for the concurrent :class:`NodeScheduler`.

Covers the brief's cases against a stub ``run_node`` (no agents): independent
nodes run concurrently, ``depends_on`` is respected, ``max_parallel`` caps
concurrency, a ``FAILED`` node blocks its (transitive) dependents, and a
``GraphState`` checkpoint lets a fresh scheduler resume without re-running
already-``DONE`` nodes. Concurrency is proven deterministically with
``asyncio`` primitives (a barrier / an event + counter), never wall-clock
timing.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from agent_flow.orchestration.graph import ExecutionGraph, Node, NodeState
from agent_flow.orchestration.isolation import NoOpIsolation
from agent_flow.orchestration.scheduler import GraphResult, GraphState, NodeOutcome, NodeScheduler


class StubIsolation:
    """Minimal :class:`IsolationProvider` that hands out throwaway directories.

    Records acquire/release/reclaim order so tests can assert isolation is
    wrapped around every ``run_node`` call without needing a real git repository.
    """

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.acquired: list[str] = []
        self.released: list[str] = []
        self.reclaimed: list[str] = []

    async def acquire(self, node: Node) -> Path:
        self.acquired.append(node.id)
        path = self.root / node.id
        path.mkdir(parents=True, exist_ok=True)
        return path

    async def release(self, node: Node) -> None:
        self.released.append(node.id)

    async def reclaim(self, node: Node) -> None:
        self.reclaimed.append(node.id)

    async def prepare(self, node: Node, dep_nodes: list[Node], cwd: Path) -> object | None:
        return None

    async def commit(self, node: Node, cwd: Path) -> object | None:
        return None


def _impl(node_id: str, *deps: str) -> Node:
    """Build an ``impl`` node with the given id and sibling dependencies."""
    return Node(id=node_id, type="impl", depends_on=tuple(deps))


async def test_independent_nodes_run_concurrently(tmp_path):
    """Two nodes with no deps are inside ``run_node`` at the same time."""
    graph = ExecutionGraph(nodes=(_impl("a"), _impl("b")))
    barrier = asyncio.Barrier(2)
    active = 0
    peak = 0

    async def run_node(node, path):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        # Only trips once BOTH nodes are here: proves overlap without sleeps.
        await barrier.wait()
        active -= 1
        return NodeOutcome(NodeState.DONE)

    scheduler = NodeScheduler(graph, run_node, StubIsolation(tmp_path), max_parallel=8)
    # If they never overlapped, the barrier would deadlock and time out.
    result = await asyncio.wait_for(scheduler.run(), timeout=5)

    assert peak == 2
    assert result.succeeded
    assert result.states == {"a": NodeState.DONE, "b": NodeState.DONE}


async def test_max_parallel_cap_never_exceeded(tmp_path):
    """No more than ``max_parallel`` nodes execute at once, and it is reached."""
    cap = 2
    graph = ExecutionGraph(nodes=tuple(_impl(f"n{i}") for i in range(6)))
    release = asyncio.Event()
    active = 0
    peak = 0

    async def run_node(node, path):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        await release.wait()
        active -= 1
        return NodeOutcome(NodeState.DONE)

    scheduler = NodeScheduler(graph, run_node, StubIsolation(tmp_path), max_parallel=cap)
    task = asyncio.create_task(scheduler.run())

    # Let everything the semaphore permits start; the rest block on the
    # semaphore (held by the ``cap`` active workers stuck on ``release``).
    for _ in range(200):
        await asyncio.sleep(0)

    assert active == cap
    assert peak == cap

    release.set()
    result = await task

    assert peak == cap
    assert result.succeeded
    assert all(state == NodeState.DONE for state in result.states.values())


async def test_depends_on_respected(tmp_path):
    """A dependent never enters ``run_node`` before its dependency has returned."""
    graph = ExecutionGraph(nodes=(_impl("a"), _impl("b", "a")))
    events: list[tuple[str, str]] = []

    async def run_node(node, path):
        events.append(("enter", node.id))
        await asyncio.sleep(0)
        events.append(("exit", node.id))
        return NodeOutcome(NodeState.DONE)

    scheduler = NodeScheduler(graph, run_node, StubIsolation(tmp_path))
    result = await scheduler.run()

    assert result.succeeded
    assert events.index(("enter", "b")) > events.index(("exit", "a"))


async def test_parent_node_waits_for_all_children(tmp_path):
    """A parent (e.g. a Stage gate) enters ``run_node`` only after every child finishes.

    A nested node is a gate over its own sub-DAG: its Goals must all be DONE before
    the parent's convergence gate runs. Children are scheduled independently by their
    own ``depends_on``; the parent waits for all of them.
    """
    graph = ExecutionGraph(
        nodes=(
            Node(
                id="s1",
                type="stage",
                children=(Node(id="s1.g1", type="goal"), Node(id="s1.g2", type="goal")),
            ),
        )
    )
    events: list[tuple[str, str]] = []

    async def run_node(node, path):
        events.append(("enter", node.id))
        await asyncio.sleep(0)
        events.append(("exit", node.id))
        return NodeOutcome(NodeState.DONE)

    scheduler = NodeScheduler(graph, run_node, StubIsolation(tmp_path), max_parallel=8)
    result = await scheduler.run()

    assert result.succeeded
    # The parent gate starts only after BOTH goals have exited run_node.
    parent_enter = events.index(("enter", "s1"))
    assert events.index(("exit", "s1.g1")) < parent_enter
    assert events.index(("exit", "s1.g2")) < parent_enter


async def test_failed_child_blocks_parent_gate(tmp_path):
    """A FAILED child leaves its parent gate un-run and marked ``BLOCKED``.

    If a Goal fails, its Stage cannot converge, so the Stage gate must not run —
    it is blocked by the child, mirroring transitive ``depends_on`` blocking.
    """
    graph = ExecutionGraph(
        nodes=(
            Node(
                id="s1",
                type="stage",
                children=(Node(id="s1.g1", type="goal"), Node(id="s1.g2", type="goal")),
            ),
        )
    )
    ran: list[str] = []

    async def run_node(node, path):
        ran.append(node.id)
        state = NodeState.FAILED if node.id == "s1.g1" else NodeState.DONE
        return NodeOutcome(state)

    scheduler = NodeScheduler(graph, run_node, StubIsolation(tmp_path), max_parallel=8)
    result = await scheduler.run()

    assert not result.succeeded
    # The parent gate never ran; it is BLOCKED by the failed goal.
    assert "s1" not in ran
    assert result.states["s1"] == NodeState.BLOCKED
    assert result.states["s1.g1"] == NodeState.FAILED


async def test_child_goal_waits_for_ancestor_stage_dependency(tmp_path):
    """A stage's ``depends_on`` gates its whole subtree, not just the stage gate.

    Cross-stage ordering can only be expressed on the stage node (``depends_on``
    is same-level-siblings only), so a goal inside ``s2`` (with empty
    ``depends_on``) must still wait for ``s1`` because ``s2 depends_on [s1]``.
    """
    graph = ExecutionGraph(
        nodes=(
            Node(id="s1", type="stage", children=(Node(id="s1.g1", type="goal"),)),
            Node(
                id="s2",
                type="stage",
                depends_on=("s1",),
                children=(Node(id="s2.g1", type="goal"),),  # empty depends_on
            ),
        )
    )
    events: list[tuple[str, str]] = []

    async def run_node(node, path):
        events.append(("enter", node.id))
        await asyncio.sleep(0)
        events.append(("exit", node.id))
        return NodeOutcome(NodeState.DONE)

    result = await NodeScheduler(graph, run_node, StubIsolation(tmp_path), max_parallel=8).run()

    assert result.succeeded
    # s2's goal started only after the whole s1 stage finished — not at iteration 1.
    assert events.index(("enter", "s2.g1")) > events.index(("exit", "s1"))


async def test_child_goal_blocked_when_ancestor_stage_dependency_fails(tmp_path):
    """If a stage's cross-stage dependency fails, the stage's goals are BLOCKED.

    Not left dangling PENDING, and never run against a missing dependency.
    """
    graph = ExecutionGraph(
        nodes=(
            Node(id="s1", type="stage", children=(Node(id="s1.g1", type="goal"),)),
            Node(
                id="s2",
                type="stage",
                depends_on=("s1",),
                children=(Node(id="s2.g1", type="goal"),),
            ),
        )
    )
    ran: list[str] = []

    async def run_node(node, path):
        ran.append(node.id)
        return NodeOutcome(NodeState.FAILED if node.id == "s1.g1" else NodeState.DONE)

    result = await NodeScheduler(graph, run_node, StubIsolation(tmp_path), max_parallel=8).run()

    assert not result.succeeded
    # s1.g1 failed -> s1 stage blocked -> s2 AND its goal s2.g1 blocked, never run.
    assert "s2.g1" not in ran
    assert result.states["s2.g1"] == NodeState.BLOCKED
    assert result.states["s2"] == NodeState.BLOCKED


async def test_failed_node_blocks_transitive_dependents(tmp_path):
    """A ``FAILED`` node leaves its dependents un-run and marked ``BLOCKED``."""
    graph = ExecutionGraph(nodes=(_impl("a"), _impl("b", "a"), _impl("c", "b")))
    entered: list[str] = []

    async def run_node(node, path):
        entered.append(node.id)
        if node.id == "a":
            return NodeOutcome(NodeState.FAILED)
        return NodeOutcome(NodeState.DONE)

    scheduler = NodeScheduler(graph, run_node, StubIsolation(tmp_path))
    result = await scheduler.run()

    assert entered == ["a"]
    assert result.states["a"] == NodeState.FAILED
    assert result.states["b"] == NodeState.BLOCKED
    assert result.states["c"] == NodeState.BLOCKED
    assert not result.succeeded
    assert result.failed_ids == ("a",)
    assert result.blocked_ids == ("b", "c")


async def test_on_transition_reports_each_state_change(tmp_path):
    """The ``on_transition`` hook observes PENDING->RUNNING->DONE for a node."""
    graph = ExecutionGraph(nodes=(_impl("a"),))
    transitions: list[tuple[str, NodeState, NodeState]] = []

    async def run_node(node, path):
        return NodeOutcome(NodeState.DONE)

    scheduler = NodeScheduler(
        graph,
        run_node,
        StubIsolation(tmp_path),
        on_transition=lambda nid, old, new: transitions.append((nid, old, new)),
    )
    await scheduler.run()

    assert ("a", NodeState.PENDING, NodeState.RUNNING) in transitions
    assert ("a", NodeState.RUNNING, NodeState.DONE) in transitions


async def test_needs_replan_surfaced_in_result(tmp_path):
    """A ``run_node`` requesting a replan is reflected on the ``GraphResult``."""
    graph = ExecutionGraph(nodes=(_impl("a"),))

    async def run_node(node, path):
        return NodeOutcome(NodeState.DONE, needs_replan=True, info={"why": "x"})

    scheduler = NodeScheduler(graph, run_node, StubIsolation(tmp_path))
    result = await scheduler.run()

    assert result.succeeded
    assert result.needs_replan is True
    assert result.outcomes["a"].info == {"why": "x"}


def test_graphstate_roundtrip_is_atomic(tmp_path):
    """``save`` then ``load`` restores states; no temp files are left behind."""
    checkpoint = tmp_path / "state.json"
    state = GraphState(
        states={"a": NodeState.DONE, "b": NodeState.PENDING},
        worktrees={"a": "/tmp/a"},
    )
    state.save(checkpoint)

    loaded = GraphState.load(checkpoint)
    assert loaded.states == {"a": NodeState.DONE, "b": NodeState.PENDING}
    assert loaded.worktrees == {"a": "/tmp/a"}

    leftovers = [p.name for p in checkpoint.parent.iterdir() if p.name != checkpoint.name]
    assert leftovers == []


async def test_checkpoint_resume_skips_done_nodes(tmp_path):
    """A fresh scheduler on the same checkpoint does not re-run DONE nodes."""
    checkpoint = tmp_path / "state.json"
    graph = ExecutionGraph(nodes=(_impl("a"), _impl("b", "a")))

    calls_run1: list[str] = []

    async def run1(node, path):
        calls_run1.append(node.id)
        if node.id == "a":
            return NodeOutcome(NodeState.DONE)
        raise RuntimeError("crash before b completes")

    scheduler1 = NodeScheduler(
        graph, run1, StubIsolation(tmp_path / "wt1"), checkpoint_path=checkpoint
    )
    with pytest.raises(RuntimeError):
        await scheduler1.run()

    assert "a" in calls_run1
    # The crash left a durable checkpoint with ``a`` already DONE.
    persisted = GraphState.load(checkpoint)
    assert persisted.states["a"] == NodeState.DONE

    calls_run2: list[str] = []

    async def run2(node, path):
        calls_run2.append(node.id)
        return NodeOutcome(NodeState.DONE)

    scheduler2 = NodeScheduler(
        graph, run2, StubIsolation(tmp_path / "wt2"), checkpoint_path=checkpoint
    )
    result = await scheduler2.run()

    assert calls_run2 == ["b"]  # ``a`` satisfied by checkpoint, only ``b`` runs
    assert result.states["a"] == NodeState.DONE
    assert result.states["b"] == NodeState.DONE
    assert result.succeeded


async def test_non_terminal_run_node_outcome_raises(tmp_path):
    """A ``run_node`` returning a non-terminal state raises, never re-runs forever.

    ``run_node`` must report a terminal ``DONE``/``FAILED``. A contract-violating
    ``PENDING`` would otherwise transition the node back to PENDING and make it
    ready again on the next loop → an infinite re-run. The scheduler must instead
    fail fast with a clear error naming the node and the bad state.
    """
    graph = ExecutionGraph(nodes=(_impl("a"),))

    async def run_node(node, path):
        return NodeOutcome(NodeState.PENDING)

    scheduler = NodeScheduler(graph, run_node, StubIsolation(tmp_path))
    # ``wait_for`` bounds the pre-fix infinite loop so the test fails fast (with a
    # TimeoutError) rather than hanging if the guard is missing.
    with pytest.raises((ValueError, RuntimeError)) as excinfo:
        await asyncio.wait_for(scheduler.run(), timeout=5)

    message = str(excinfo.value)
    assert "a" in message
    assert "PENDING" in message


async def test_state_snapshot_reflects_completion(tmp_path):
    """``state()`` returns a snapshot with terminal states after ``run``."""
    graph = ExecutionGraph(nodes=(_impl("a"), _impl("b", "a")))

    async def run_node(node, path):
        return NodeOutcome(NodeState.DONE)

    scheduler = NodeScheduler(graph, run_node, StubIsolation(tmp_path))
    await scheduler.run()

    snapshot = scheduler.state()
    assert isinstance(snapshot, GraphState)
    assert snapshot.states == {"a": NodeState.DONE, "b": NodeState.DONE}


class _RecordingIsolation:
    """Stub :class:`IsolationProvider` that records call order and prepare's deps.

    ``prepare`` returns a generic sentinel object (the scheduler must treat it as
    opaque ``object``), so this also proves the ``prepare_results`` channel is not
    tied to any git-specific report type.
    """

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.events: list[tuple[str, str]] = []
        self.prepare_calls: dict[str, list[str]] = {}

    async def acquire(self, node: Node) -> Path:
        self.events.append(("acquire", node.id))
        path = self.root / node.id
        path.mkdir(parents=True, exist_ok=True)
        return path

    async def release(self, node: Node) -> None:
        self.events.append(("release", node.id))

    async def reclaim(self, node: Node) -> None:
        self.events.append(("reclaim", node.id))

    async def prepare(self, node: Node, dep_nodes: list[Node], cwd: Path) -> object | None:
        self.events.append(("prepare", node.id))
        self.prepare_calls[node.id] = [dep.id for dep in dep_nodes]
        return {"node": node.id, "deps": [dep.id for dep in dep_nodes]}

    async def commit(self, node: Node, cwd: Path) -> object | None:
        self.events.append(("commit", node.id))
        return None


async def test_scheduler_prepares_any_node_with_dependencies(tmp_path):
    """``prepare`` runs for ANY node that declares dependencies, not only merge nodes.

    A dependent impl node inherits its dependencies' work — the concurrent-DAG
    handoff contract that ``depends_on`` carries code.
    """
    graph = ExecutionGraph(nodes=(_impl("a"), _impl("b", "a")))
    isolation = _RecordingIsolation(tmp_path)

    async def run_node(node, path):
        return NodeOutcome(NodeState.DONE)

    result = await NodeScheduler(graph, run_node, isolation).run()

    assert result.succeeded
    # ``b`` depends on ``a`` -> prepared with [a]; ``a`` has no deps -> never prepared.
    assert isolation.prepare_calls == {"b": ["a"]}
    assert ("prepare", "a") not in isolation.events


async def test_scheduler_commits_each_node_after_run_and_before_release(tmp_path):
    """The scheduler commits every node's work after ``run_node`` and before ``release``.

    Commit happens on both ``DONE`` and ``FAILED`` terminal states, so no verified
    artifact is lost when the transient workspace is torn down (Decision B).
    """
    graph = ExecutionGraph(nodes=(_impl("ok"), _impl("bad")))
    isolation = _RecordingIsolation(tmp_path)

    async def run_node(node, path):
        isolation.events.append(("run", node.id))
        return NodeOutcome(NodeState.DONE if node.id == "ok" else NodeState.FAILED)

    await NodeScheduler(graph, run_node, isolation).run()

    # commit ran for the DONE node AND the FAILED node...
    assert ("commit", "ok") in isolation.events
    assert ("commit", "bad") in isolation.events
    # ...each strictly between its own run and its own release.
    for nid in ("ok", "bad"):
        ev = [e for e in isolation.events if e[1] == nid]
        assert ev.index(("run", nid)) < ev.index(("commit", nid)) < ev.index(("release", nid))


async def test_scheduler_calls_prepare_for_merge_nodes_only(tmp_path):
    """A merge node triggers ``prepare`` after acquire, before run_node; impl nodes never do."""
    graph = ExecutionGraph(
        nodes=(
            Node(id="a", type="impl"),
            Node(id="b", type="impl"),
            Node(id="m", type="merge", kind="merge", depends_on=("a", "b")),
        )
    )
    isolation = _RecordingIsolation(tmp_path)

    async def run_node(node, path):
        # Share the isolation timeline so acquire/prepare/run/release order is
        # asserted against a single, well-ordered event log.
        isolation.events.append(("run", node.id))
        return NodeOutcome(NodeState.DONE)

    scheduler = NodeScheduler(graph, run_node, isolation)
    result = await scheduler.run()

    assert result.succeeded
    # ``prepare`` ran only for the merge node, with the resolved dependency nodes.
    assert isolation.prepare_calls == {"m": ["a", "b"]}
    # The (opaque) value ``prepare`` returned is exposed on the GraphResult.
    assert result.prepare_results["m"] == {"node": "m", "deps": ["a", "b"]}
    assert "a" not in result.prepare_results and "b" not in result.prepare_results

    # For the merge node the sequence is acquire -> prepare -> run -> release ->
    # reclaim (``m`` has no dependents, so it is reclaimed on its own terminal).
    m_events = [ev for ev in isolation.events if ev[1] == "m"]
    assert m_events == [
        ("acquire", "m"),
        ("prepare", "m"),
        ("run", "m"),
        ("commit", "m"),
        ("release", "m"),
        ("reclaim", "m"),
    ]
    # Plain impl nodes never trigger a prepare.
    assert ("prepare", "a") not in isolation.events
    assert ("prepare", "b") not in isolation.events


async def test_runs_whole_graph_with_noop_isolation(tmp_path):
    """A whole DAG runs to completion with NoOpIsolation and a stub run_node — zero git."""
    graph = ExecutionGraph(
        nodes=(
            _impl("a"),
            _impl("b"),
            _impl("c", "a", "b"),
        )
    )
    ran_in: dict[str, Path] = {}

    async def run_node(node, path):
        ran_in[node.id] = path
        return NodeOutcome(NodeState.DONE)

    scheduler = NodeScheduler(graph, run_node, NoOpIsolation(tmp_path))
    result = await scheduler.run()

    assert result.succeeded
    assert isinstance(result, GraphResult)
    assert result.states == {
        "a": NodeState.DONE,
        "b": NodeState.DONE,
        "c": NodeState.DONE,
    }
    # Every node ran in the single shared base directory; no worktrees involved.
    assert ran_in == {"a": tmp_path, "b": tmp_path, "c": tmp_path}
    # A merge-free graph seeds nothing.
    assert result.prepare_results == {}


# --------------------------------------------------------------------------- #
# Dependency-lifetime reclamation
# --------------------------------------------------------------------------- #


class _ReclaimRecordingIsolation:
    """Stub isolation recording the order of ``release`` and ``reclaim`` calls."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.released: list[str] = []
        self.reclaimed: list[str] = []

    async def acquire(self, node: Node) -> Path:
        path = self.root / node.id
        path.mkdir(parents=True, exist_ok=True)
        return path

    async def release(self, node: Node) -> None:
        self.released.append(node.id)

    async def reclaim(self, node: Node) -> None:
        self.reclaimed.append(node.id)

    async def prepare(self, node: Node, dep_nodes: list[Node], cwd: Path) -> object | None:
        return None

    async def commit(self, node: Node, cwd: Path) -> object | None:
        return None


async def test_single_node_reclaims_on_own_terminal(tmp_path):
    """A node with no dependents is reclaimed as soon as it goes terminal."""
    graph = ExecutionGraph(nodes=(_impl("a"),))
    isolation = _ReclaimRecordingIsolation(tmp_path)

    async def run_node(node, path):
        return NodeOutcome(NodeState.DONE)

    result = await NodeScheduler(graph, run_node, isolation).run()

    assert result.succeeded
    assert isolation.reclaimed == ["a"]


async def test_leaves_reclaim_themselves_and_each_node_reclaimed_once(tmp_path):
    """Leaf dependents reclaim on their own terminal; every node is reclaimed exactly once."""
    graph = ExecutionGraph(nodes=(_impl("a"), _impl("b", "a"), _impl("c", "a")))
    isolation = _ReclaimRecordingIsolation(tmp_path)

    async def run_node(node, path):
        return NodeOutcome(NodeState.DONE)

    result = await NodeScheduler(graph, run_node, isolation).run()

    assert result.succeeded
    # a (a dependency) plus the two leaves b, c — each reclaimed once.
    assert sorted(isolation.reclaimed) == ["a", "b", "c"]
    assert len(isolation.reclaimed) == 3


async def test_dependency_reclaimed_only_after_all_dependents_terminal(tmp_path):
    """For ``a→b, a→c``, ``reclaim(a)`` fires only after BOTH b and c are terminal."""
    graph = ExecutionGraph(nodes=(_impl("a"), _impl("b", "a"), _impl("c", "a")))
    isolation = _ReclaimRecordingIsolation(tmp_path)

    gate_b = asyncio.Event()
    gate_c = asyncio.Event()
    running_b = asyncio.Event()
    running_c = asyncio.Event()

    async def run_node(node, path):
        if node.id == "b":
            running_b.set()
            await gate_b.wait()
        elif node.id == "c":
            running_c.set()
            await gate_c.wait()
        return NodeOutcome(NodeState.DONE)

    task = asyncio.create_task(NodeScheduler(graph, run_node, isolation).run())

    # ``a`` finishes first (no deps); ``b`` and ``c`` then start and block.
    await asyncio.wait_for(running_b.wait(), timeout=5)
    await asyncio.wait_for(running_c.wait(), timeout=5)
    # ``a`` is DONE but both dependents are still running -> must not be reclaimed.
    assert "a" not in isolation.reclaimed

    # Release only ``b``: ``c`` is still running, so ``a`` still must not reclaim.
    gate_b.set()
    for _ in range(100):
        await asyncio.sleep(0)
    assert "a" not in isolation.reclaimed

    # Release ``c``: now every dependent of ``a`` is terminal -> reclaim ``a``.
    gate_c.set()
    result = await asyncio.wait_for(task, timeout=5)

    assert result.succeeded
    assert "a" in isolation.reclaimed
    assert isolation.reclaimed.count("a") == 1  # reclaimed at most once
    assert sorted(isolation.reclaimed) == ["a", "b", "c"]


async def test_all_nodes_reclaimed_by_run_return_even_when_blocked(tmp_path):
    """Every node — including FAILED and BLOCKED ones — is reclaimed once by ``run`` return."""
    graph = ExecutionGraph(nodes=(_impl("a"), _impl("b", "a"), _impl("c", "b")))
    isolation = _ReclaimRecordingIsolation(tmp_path)

    async def run_node(node, path):
        if node.id == "a":
            return NodeOutcome(NodeState.FAILED)
        return NodeOutcome(NodeState.DONE)

    result = await NodeScheduler(graph, run_node, isolation).run()

    assert not result.succeeded
    assert result.states["a"] == NodeState.FAILED
    assert result.states["b"] == NodeState.BLOCKED
    assert result.states["c"] == NodeState.BLOCKED
    # All reclaimed exactly once by the time ``run`` returns.
    assert sorted(isolation.reclaimed) == ["a", "b", "c"]
    assert len(isolation.reclaimed) == 3
