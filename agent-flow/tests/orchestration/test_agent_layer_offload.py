"""The seam between the DAG scheduler and a synchronous ``AgentLayer``.

A workflow may already own a working, synchronous agent loop — perf-optimize's
optimizer ⇄ evaluator attempt loop is one — and rewriting it as a coroutine to
adopt the scheduler is a large, risky change. The alternative is to offload the
loop from an async ``run_node`` with ``anyio.to_thread.run_sync``, which is what
perf-optimize does.

That combination has two ways to be silently wrong, and both are pinned here:

* calling the blocking layer **directly** from the coroutine would stall the
  scheduler's event loop, so the nodes would serialize while still passing every
  functional assertion — only a concurrency probe catches it; and
* tearing an agent down from the wrong thread closes a client bound to another
  event loop. ``AgentLayer._aclose_self`` routes teardown through the layer's
  own ``PortalRunner`` when one was started, which is what makes the synchronous
  ``__exit__`` correct from inside a worker thread.

Nothing here touches a real backend: the layers are real, the backend is a fake.
"""

from __future__ import annotations

from pathlib import Path
from threading import Barrier, get_ident
from unittest.mock import patch

import anyio
import pytest

from agent_flow.config import AgentLayerConfig, BackendConfig, SessionConfig
from agent_flow.layers import AgentLayer
from agent_flow.orchestration import (
    ExecutionGraph,
    Node,
    NodeOutcome,
    NodeScheduler,
    NodeState,
    NoOpIsolation,
)

from ..helpers import FakeBackend


def _layer(name: str, mode: str) -> AgentLayer:
    return AgentLayer(
        AgentLayerConfig(
            name=name,
            system_prompt="you are a test agent",
            backend=BackendConfig(kind="claude-code", model="test-model"),
            session=SessionConfig(mode=mode),
            print_activity=False,
        )
    )


@pytest.mark.parametrize("mode", ["persistent", "stateless"])
def test_a_layer_runs_and_tears_down_from_inside_a_run_node(tmp_path, mode):
    """One node: build a layer, take a turn, close it — all off the event loop.

    Covers both session modes because they take different paths out of
    ``forward``: ``persistent`` hands the turn to the layer's ``PortalRunner``
    (and must tear down through it), while ``stateless`` opens its own loop with
    ``anyio.run`` on the worker thread.
    """
    backend = FakeBackend([{"text": "done"}])
    replies: list[str] = []
    loop_thread = get_ident()
    worker_threads: list[int] = []

    def blocking_item_loop() -> None:
        worker_threads.append(get_ident())
        layer = _layer("optimizer-opt-001", mode)
        try:
            replies.append(layer(" implement opt-001"))
        finally:
            # Exactly what the item loop does in its ``finally``.
            layer.__exit__(None, None, None)

    async def run_node(node: Node, cwd: Path) -> NodeOutcome:
        await anyio.to_thread.run_sync(blocking_item_loop)
        return NodeOutcome(terminal_state=NodeState.DONE)

    graph = ExecutionGraph(nodes=(Node(id="item_1", type="roadmap_item", isolation="shared"),))
    scheduler = NodeScheduler(graph, run_node, NoOpIsolation(tmp_path), max_parallel=1)

    with patch("agent_flow.layers.create_backend", return_value=backend):
        result = anyio.run(scheduler.run)

    assert result.succeeded
    assert replies == ["done"]
    assert worker_threads and worker_threads[0] != loop_thread, (
        "the agent turn must not run on the scheduler's event loop thread"
    )


def test_offloaded_item_loops_actually_overlap(tmp_path):
    """Two nodes must genuinely run at the same time, not merely both finish.

    This is the assertion that would fail if ``run_node`` ever called the
    blocking layer directly instead of offloading it: the loop would be pinned
    by the first node, the second could not reach the barrier, and the wait
    would break. Functional assertions alone cannot see that regression — the
    campaign would still produce correct results, just with no concurrency.
    """
    backend = FakeBackend([{"text": "done"}])
    barrier = Barrier(2)
    reached: list[str] = []

    def blocking_item_loop(item_id: str) -> None:
        layer = _layer(f"optimizer-{item_id}", "persistent")
        try:
            layer(f"implement {item_id}")
            # Both nodes must be inside their turn at the same moment.
            barrier.wait(timeout=10)
            reached.append(item_id)
        finally:
            layer.__exit__(None, None, None)

    async def run_node(node: Node, cwd: Path) -> NodeOutcome:
        await anyio.to_thread.run_sync(blocking_item_loop, node.id)
        return NodeOutcome(terminal_state=NodeState.DONE)

    graph = ExecutionGraph(
        nodes=(
            Node(id="item_1", type="roadmap_item", isolation="shared"),
            Node(id="item_2", type="roadmap_item", isolation="shared"),
        )
    )
    scheduler = NodeScheduler(graph, run_node, NoOpIsolation(tmp_path), max_parallel=2)

    with patch("agent_flow.layers.create_backend", return_value=backend):
        result = anyio.run(scheduler.run)

    assert result.succeeded
    assert sorted(reached) == ["item_1", "item_2"]
