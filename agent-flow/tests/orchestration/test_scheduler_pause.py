import asyncio

from agent_flow.orchestration.graph import ExecutionGraph, Node, NodeState
from agent_flow.orchestration.scheduler import NodeOutcome, NodeScheduler


class _NoIsolation:
    async def acquire(self, node):
        return __import__("pathlib").Path(".")

    async def prepare(self, node, deps, cwd):
        return None

    async def commit(self, node, cwd):
        return None

    async def release(self, node):
        return None

    async def reclaim(self, node):
        return None


class _RecordingIsolation:
    """No-op isolation that records which nodes were reclaimed."""

    def __init__(self):
        self.reclaimed = []

    async def acquire(self, node):
        return __import__("pathlib").Path(".")

    async def prepare(self, node, deps, cwd):
        return None

    async def commit(self, node, cwd):
        return None

    async def release(self, node):
        return None

    async def reclaim(self, node):
        self.reclaimed.append(node.id)


def _graph():
    # a and b are independent; a fails fast, b is a long-runner.
    return ExecutionGraph(nodes=(Node(id="a", type="s"), Node(id="b", type="s")))


def test_pause_on_failure_suspends_healthy_inflight_node():
    async def run_node(node, cwd):
        if node.id == "a":
            return NodeOutcome(terminal_state=NodeState.FAILED)
        await asyncio.sleep(3600)  # long-runner; must be cancelled, not drained
        return NodeOutcome(terminal_state=NodeState.DONE)

    sched = NodeScheduler(_graph(), run_node, _NoIsolation(), pause_on_failure=True)
    result = asyncio.run(asyncio.wait_for(sched.run(), timeout=30))

    assert result.states["a"] == NodeState.FAILED
    assert result.states["b"] == NodeState.INTERRUPTED  # suspended, not drained, not FAILED
    assert result.paused is True
    assert result.interrupted_ids == ("b",)


def test_no_pause_when_flag_off_still_drains():
    async def run_node(node, cwd):
        if node.id == "a":
            return NodeOutcome(terminal_state=NodeState.FAILED)
        return NodeOutcome(terminal_state=NodeState.DONE)

    sched = NodeScheduler(_graph(), run_node, _NoIsolation(), pause_on_failure=False)
    result = asyncio.run(sched.run())
    assert result.states["a"] == NodeState.FAILED
    assert result.states["b"] == NodeState.DONE  # drained to completion
    assert result.paused is False


def test_pause_does_not_reclaim_the_suspended_node():
    async def run_node(node, cwd):
        if node.id == "a":
            return NodeOutcome(terminal_state=NodeState.FAILED)  # triggers pause
        await asyncio.sleep(3600)  # b: long-runner → suspended
        return NodeOutcome(terminal_state=NodeState.DONE)

    iso = _RecordingIsolation()
    sched = NodeScheduler(_graph(), run_node, iso, pause_on_failure=True)
    result = asyncio.run(asyncio.wait_for(sched.run(), timeout=30))

    assert result.states["b"] == NodeState.INTERRUPTED
    assert result.paused is True
    # The suspended node's durable output must survive for resume — the pause path
    # must NOT sweep-reclaim it. (Without the fix, _sweep_reclaim records "b".)
    assert "b" not in iso.reclaimed
