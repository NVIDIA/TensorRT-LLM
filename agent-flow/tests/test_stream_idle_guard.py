"""Bounding a stream that stops arriving.

Nothing under agent-flow bounds this against a self-hosted gateway. The CLI's
byte-idle watchdog and the runtime's fetch timeout are both gated on the
endpoint being api.anthropic.com, so against any other base URL neither is
armed. Measured on one hang: 40 minutes, zero bytes read on an ESTABLISHED
socket, main thread in epoll_wait, no child process, all three CLI ceilings set
to 900s and none of them firing. A campaign in that state never fails and never
finishes.

The two properties that matter are opposite, which is why both are pinned here:
a turn that keeps producing is never interrupted however long it runs, and a
turn that goes quiet is cut. The tests use a ceiling in milliseconds rather than
the shipped 900s -- the behaviour under test is the ratio between the gap and
the ceiling, not the wall-clock value, which has its own test below.
"""

from __future__ import annotations

from types import SimpleNamespace

import anyio
import pytest

from agent_flow import layers
from agent_flow.backends.claude_code import (
    STREAM_IDLE_TIMEOUT_S,
    StreamIdleError,
    _each_before_idle,
)
from agent_flow.layers import AgentLayer

CEILING = 0.2


async def _drain(stream, seconds=CEILING):
    return [item async for item in _each_before_idle(stream, seconds)]


@pytest.mark.anyio
async def test_a_stream_that_never_speaks_is_cut():
    """The measured hang: the stream opens and then produces nothing."""

    async def never_arrives():
        await anyio.sleep_forever()
        yield  # pragma: no cover - generator marker

    with pytest.raises(StreamIdleError, match="hung"):
        await _drain(never_arrives())


@pytest.mark.anyio
async def test_the_deadline_is_per_message_not_per_turn():
    """A long turn is not a hung turn.

    Ten gaps, each over half the ceiling, run well past it in total and must
    all get through. Guards the sharp edge: a per-TURN deadline would kill real
    reporter work, which is the failure this guard exists to avoid repeating.
    """

    async def slow_but_alive():
        for i in range(10):
            await anyio.sleep(CEILING * 0.6)
            yield i

    assert await _drain(slow_but_alive()) == list(range(10))


@pytest.mark.anyio
async def test_it_cuts_on_the_gap_that_goes_quiet():
    """Items already delivered still arrive; the stall after them is what fires."""

    async def two_then_silence():
        yield "a"
        yield "b"
        await anyio.sleep_forever()

    seen = []
    with pytest.raises(StreamIdleError):
        async for item in _each_before_idle(two_then_silence(), CEILING):
            seen.append(item)

    assert seen == ["a", "b"]


@pytest.mark.anyio
async def test_a_stream_that_ends_normally_is_not_an_error():
    async def finite():
        yield 1
        yield 2

    assert await _drain(finite()) == [1, 2]


def test_the_ceiling_is_above_the_longest_real_turn():
    """Measured: the slowest reporter turn here ran well under 15 minutes.

    A ceiling at or below real work turns this guard into the very thing it
    replaced -- a timeout that fires on healthy runs.
    """
    assert STREAM_IDLE_TIMEOUT_S >= 900


def test_the_ceiling_is_overridable_without_a_code_change(monkeypatch):
    """A deployment whose turns are genuinely slower must not need a patch."""
    import importlib

    monkeypatch.setenv("AGENT_FLOW_STREAM_IDLE_TIMEOUT_S", "1800")
    import agent_flow.backends.claude_code as mod

    reloaded = importlib.reload(mod)
    try:
        assert reloaded.STREAM_IDLE_TIMEOUT_S == 1800.0
    finally:
        monkeypatch.delenv("AGENT_FLOW_STREAM_IDLE_TIMEOUT_S")
        importlib.reload(mod)


class _Layer:
    """Drives ``AgentLayer._invoke_persistent`` with everything else stubbed.

    The loop under test is small, but what it coordinates is not reachable from
    a unit test otherwise: building a real client means a real CLI subprocess.
    So the collaborators are recorded rather than run, and the assertions are
    about ORDER -- that the dead client is dropped BEFORE a replacement is
    asked for, which is the whole point of the recovery.
    """

    def __init__(self, failures: int):
        self.failures = failures
        self.events: list[str] = []
        self._backend = object()
        self.config = SimpleNamespace(system_prompt="sys")

    def _build_request(self, content):
        return SimpleNamespace(content=content, system_prompt="sys")

    async def _ensure_persistent_client(self, system_prompt):
        self.events.append("create")
        return object(), True

    async def _drop_persistent_client(self):
        self.events.append("drop")

    async def _run_with_client(self, request, client, backend, report_baseline=False):
        self.events.append("run")
        if self.failures > 0:
            self.failures -= 1
            raise StreamIdleError("no message from the backend for 900s")
        return "the answer"


def _invoke(layer):
    return AgentLayer._invoke_persistent(layer, "do the thing")


@pytest.mark.anyio
async def test_a_hung_session_is_replaced_and_the_turn_succeeds():
    layer = _Layer(failures=1)

    assert await _invoke(layer) == "the answer"
    assert layer.events == ["create", "run", "drop", "create", "run"], (
        "the dead client must be dropped before a replacement is built"
    )


@pytest.mark.anyio
async def test_the_replacement_budget_is_one():
    """A gateway that is down must end the run, not cycle sessions forever."""
    layer = _Layer(failures=99)

    with pytest.raises(StreamIdleError):
        await _invoke(layer)

    assert layer.events.count("create") == layers.STREAM_IDLE_RETRIES + 1
    assert layer.events[-1] == "drop", "the last hung client is not left running"


@pytest.mark.anyio
async def test_a_failure_that_is_not_a_hang_is_not_retried():
    """Only a hung session is worth replacing; anything else replays the same."""

    class Other(_Layer):
        async def _run_with_client(self, request, client, backend, report_baseline=False):
            self.events.append("run")
            raise RuntimeError("roadmap.yaml failed schema validation")

    layer = Other(failures=0)
    with pytest.raises(RuntimeError, match="schema validation"):
        await _invoke(layer)

    assert layer.events.count("run") == 1


@pytest.mark.anyio
async def test_a_healthy_turn_builds_one_client():
    layer = _Layer(failures=0)

    assert await _invoke(layer) == "the answer"
    assert layer.events == ["create", "run"]
