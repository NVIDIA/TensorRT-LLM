# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for the trtllm-serve startup lifecycle contract.

The contract these tests protect is that a remote poller can always tell
the three states apart:

* ``STARTING``: connect succeeds, ``/health`` answers ``503`` promptly;
* ``READY``:    connect succeeds, ``/health`` answers ``200``;
* dead:         connect is refused.

Probing is done with raw sockets rather than an HTTP client so that
"connection refused", "answered", and "accepted but never answered" stay
distinguishable -- most HTTP clients collapse the last two into one
opaque error, which is exactly the ambiguity this feature removes.
"""

import asyncio
import contextlib
import os
import pathlib
import select
import socket
import subprocess
import sys
import tempfile
import threading
import time
from contextlib import asynccontextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import NamedTuple, Optional

import pytest
from fastapi import FastAPI

from tensorrt_llm.llmapi.mpi_session import split_mpi_env
from tensorrt_llm.serve import lifecycle as lifecycle_mod
from tensorrt_llm.serve.lifecycle import (
    SERVER_STATE_HEADER,
    ServerLifecycle,
    ServerState,
    _build_off_event_loop,
    serve_with_lifecycle,
)

# Wall-clock budget a probe gets before it is recorded as a timeout. Generous
# relative to the ~ms a free event loop needs, so a failure here means the
# loop was actually blocked, not that CI was slow.
PROBE_TIMEOUT_SECONDS = 5.0

# How long the fake "engine build" blocks the worker thread. Long enough for
# many probes to land inside the STARTING window.
FAKE_BUILD_SECONDS = 5.0

PROBE_INTERVAL_SECONDS = 0.02

TIMEOUT_KEEP_ALIVE = 5

# How long the swallow-cancellation stand-in keeps ignoring cancellation:
# longer than every bound it is used against, so the task is reliably still
# alive when the assertions run, and short enough that a regression trips an
# assertion quickly rather than leaning on the timeout marker.
SWALLOW_CANCELLATION_SECONDS = 10.0

# The stand-in sleeps in slices so its deadline is actually re-evaluated.
# One long sleep would only re-check after *another* cancellation arrived,
# which never happens while a caller is hung waiting on it.
SWALLOW_SLICE_SECONDS = 0.05


def _loop_factories():
    """Both loops the server can run on; production uses uvloop."""
    factories = [pytest.param(asyncio.new_event_loop, id="asyncio")]
    try:
        import uvloop
    except ImportError:
        return factories
    return factories + [pytest.param(uvloop.new_event_loop, id="uvloop")]


LOOP_FACTORIES = _loop_factories()


# Threads that serve_with_lifecycle deliberately abandons. Abandoning a
# wedged build or teardown *is* the feature working -- but tests/unittest/
# pytest.ini enables threadleak, and its check runs as a wrapper around the
# test call, before any fixture finalizer. So the retiring has to happen
# inside the test body.
ENGINE_THREAD_PREFIX = "trtllm_engine_init"


def join_engine_init_threads(timeout: float = 60.0) -> None:
    """Wait out any build/teardown thread this test left running."""
    deadline = time.monotonic() + timeout
    for thread in list(threading.enumerate()):
        if thread.name.startswith(ENGINE_THREAD_PREFIX):
            thread.join(timeout=max(0.0, deadline - time.monotonic()))
    still_running = [
        t.name
        for t in threading.enumerate()
        if t.name.startswith(ENGINE_THREAD_PREFIX) and t.is_alive()
    ]
    assert not still_running, f"engine worker threads outlived the test: {still_running}"


def run_scenario(scenario):
    """Run an async scenario, then retire any thread it abandoned."""
    try:
        return asyncio.run(scenario())
    finally:
        join_engine_init_threads()


def run_on(loop_factory, scenario):
    """Run ``scenario()`` to completion on a fresh loop of the given kind."""
    loop = loop_factory()
    try:
        asyncio.set_event_loop(loop)
        return loop.run_until_complete(scenario())
    finally:
        with contextlib.suppress(Exception):
            loop.run_until_complete(loop.shutdown_asyncgens())
        asyncio.set_event_loop(None)
        loop.close()
        join_engine_init_threads()


class ProbeResult(NamedTuple):
    """What a remote poller observed for one request."""

    # One of: "answered", "refused", "timeout", "reset".
    outcome: str
    status: Optional[int]
    headers: dict
    elapsed: float


async def probe(
    port: int, path: str = "/health", method: str = "GET", timeout: float = PROBE_TIMEOUT_SECONDS
) -> ProbeResult:
    """Issue one raw HTTP request and classify what came back."""
    started = time.monotonic()

    def result(outcome, status=None, headers=None) -> ProbeResult:
        return ProbeResult(outcome, status, headers or {}, time.monotonic() - started)

    try:
        reader, writer = await asyncio.wait_for(asyncio.open_connection("127.0.0.1", port), timeout)
    except ConnectionRefusedError:
        return result("refused")
    except asyncio.TimeoutError:
        return result("timeout")
    except OSError:
        return result("refused")

    try:
        request = (
            f"{method} {path} HTTP/1.1\r\n"
            f"Host: 127.0.0.1:{port}\r\n"
            f"Content-Length: 0\r\n"
            f"Connection: close\r\n\r\n"
        )
        writer.write(request.encode())
        await writer.drain()

        status_line = await asyncio.wait_for(reader.readline(), timeout)
        if not status_line:
            return result("reset")
        status = int(status_line.split()[1])

        headers = {}
        while True:
            line = await asyncio.wait_for(reader.readline(), timeout)
            if line in (b"\r\n", b"\n", b""):
                break
            name, _, value = line.decode().partition(":")
            headers[name.strip().lower()] = value.strip()
        return result("answered", status, headers)
    except asyncio.TimeoutError:
        return result("timeout")
    except (ConnectionResetError, BrokenPipeError):
        return result("reset")
    finally:
        writer.close()
        try:
            await writer.wait_closed()
        except (ConnectionResetError, BrokenPipeError):
            pass


def bound_socket() -> socket.socket:
    """Bind an ephemeral port without listening, as launch_server does."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    return sock


class Teardown(NamedTuple):
    """One engine-teardown call, so overlaps between them are detectable."""

    who: str
    start: float
    end: float


def overlapping_teardowns(teardowns):
    """Return the first overlapping pair, or None if they were serialized."""
    ordered = sorted(teardowns, key=lambda t: t.start)
    for earlier, later in zip(ordered, ordered[1:]):
        if later.start < earlier.end:
            return (earlier, later, earlier.end - later.start)
    return None


class FakeServer:
    """Stand-in for OpenAIServer: an app plus the hooks the path uses.

    Its lifespan teardown and its ``shutdown_generator`` both record into
    ``teardowns``, which is what makes a second, concurrent engine shutdown
    visible to a test.
    """

    def __init__(
        self,
        register_ok=True,
        teardown_seconds=0.0,
        teardown_awaits=0.0,
        slow_route_seconds=0.0,
    ) -> None:
        self.bound_to = None
        self.registered = False
        self.teardowns = []
        self._teardown_lock = threading.Lock()
        self._register_ok = register_ok
        self._teardown_seconds = teardown_seconds
        self._teardown_awaits = teardown_awaits

        @asynccontextmanager
        async def lifespan(app):
            yield
            if self._teardown_awaits:
                # Mirrors OpenAIServer awaiting deregister_worker() before it
                # reaches the engine shutdown: on the abort path that call
                # talks to the very cluster storage whose registration just
                # failed, so it can outlast the abort bound.
                await asyncio.sleep(self._teardown_awaits)
            # Mirrors OpenAIServer: the engine teardown lives here, inline on
            # the event loop.
            self.run_teardown("lifespan")

        self.app = FastAPI(lifespan=lifespan)

        @self.app.get("/health")
        async def health():
            return {"status": "ok"}

        @self.app.post("/v1/chat/completions")
        async def chat():
            return {"status": "ok"}

        @self.app.get("/slow")
        async def slow():
            await asyncio.sleep(slow_route_seconds)
            return {"status": "ok"}

    def run_teardown(self, who: str) -> None:
        # Synchronous and slow, exactly like BaseLLM.shutdown().
        start = time.monotonic()
        time.sleep(self._teardown_seconds)
        with self._teardown_lock:
            self.teardowns.append(Teardown(who, start, time.monotonic()))

    def record_address(self, host, port) -> None:
        self.bound_to = (host, port)

    async def register_with_disagg_cluster(self) -> bool:
        self.registered = True
        return self._register_ok

    def shutdown_generator(self) -> None:
        self.run_teardown("on_startup_failure")


# --------------------------------------------------------------------------
# ServerLifecycle: the ASGI-level contract, exercised without a socket.
# --------------------------------------------------------------------------


async def call_app(app, path="/health", method="GET"):
    """Drive an ASGI app for one HTTP request; return (status, headers, body)."""
    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": method,
        "path": path,
        "raw_path": path.encode(),
        "root_path": "",
        "scheme": "http",
        "query_string": b"",
        "headers": [(b"host", b"127.0.0.1")],
        "client": ("127.0.0.1", 1234),
        "server": ("127.0.0.1", 8000),
    }
    messages = []

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message):
        messages.append(message)

    await app(scope, receive, send)

    start = next(m for m in messages if m["type"] == "http.response.start")
    body = b"".join(m.get("body", b"") for m in messages if m["type"] == "http.response.body")
    headers = {k.decode().lower(): v.decode() for k, v in start["headers"]}
    return start["status"], headers, body


class LifespanDriver:
    """Stands in for uvicorn's half of the ASGI lifespan handshake."""

    def __init__(self, app) -> None:
        self._app = app
        self._to_app: asyncio.Queue = asyncio.Queue()
        self._from_app: asyncio.Queue = asyncio.Queue()
        self.task: Optional[asyncio.Task] = None

    async def _receive(self):
        return await self._to_app.get()

    async def _send(self, message):
        await self._from_app.put(message)

    async def next_message(self) -> dict:
        return await asyncio.wait_for(self._from_app.get(), PROBE_TIMEOUT_SECONDS)

    async def startup(self) -> "LifespanDriver":
        scope = {"type": "lifespan", "asgi": {"version": "3.0"}, "state": {}}
        self.task = asyncio.create_task(self._app(scope, self._receive, self._send))
        self._to_app.put_nowait({"type": "lifespan.startup"})
        assert (await self.next_message())["type"] == "lifespan.startup.complete"
        return self

    async def shutdown(self) -> dict:
        self._to_app.put_nowait({"type": "lifespan.shutdown"})
        message = await self.next_message()
        await self.task
        return message


async def run_lifespan_startup(app) -> LifespanDriver:
    """Drive the ASGI lifespan startup handshake; return the driver."""
    return await LifespanDriver(app).startup()


def test_lifespan_startup_completes_without_an_engine():
    """Uvicorn only listens after startup completes, so it must not block."""

    async def scenario():
        lifecycle = ServerLifecycle()
        assert lifecycle.state is ServerState.STARTING
        driver = await run_lifespan_startup(lifecycle)
        assert (await driver.shutdown())["type"] == "lifespan.shutdown.complete"

    run_scenario(scenario)


@pytest.mark.parametrize(
    "path,method",
    [
        ("/health", "GET"),
        ("/v1/models", "GET"),
        ("/v1/chat/completions", "POST"),
        ("/v1/completions", "POST"),
        ("/metrics", "GET"),
        ("/definitely-not-a-route", "GET"),
        # Disagg-orchestrator-only routes. These matter more than the rest:
        # their callers parse the body without checking the status, so a
        # half-initialized handler here is cached rather than retried --
        # see Router._fetch_server_info, which now raise_for_status()es
        # precisely because this endpoint answers during STARTING.
        ("/server_info", "GET"),
        ("/steady_clock_offset", "GET"),
        ("/kv_cache_events", "POST"),
        ("/perf_metrics", "GET"),
    ],
)
def test_every_route_is_503_while_starting(path, method):
    """503, never 404 and never a half-initialized handler."""

    async def scenario():
        lifecycle = ServerLifecycle()
        status, headers, body = await call_app(lifecycle, path, method)
        assert status == 503
        assert headers[SERVER_STATE_HEADER] == ServerState.STARTING.value
        assert "retry-after" in headers
        # The body must say "not a crash" so operators reading a log of the
        # 503 do not confuse it with a dead engine.
        assert b"not a crash" in body

    run_scenario(scenario)


def test_attach_flips_to_ready_and_delegates():
    async def scenario():
        lifecycle = ServerLifecycle()
        await run_lifespan_startup(lifecycle)

        status, _, _ = await call_app(lifecycle)
        assert status == 503

        server = FakeServer()
        await lifecycle.attach(server.app)

        assert lifecycle.state is ServerState.READY
        status, _, body = await call_app(lifecycle)
        assert status == 200
        assert b"ok" in body

    run_scenario(scenario)


def test_delegate_lifespan_startup_failure_propagates():
    """A delegate whose lifespan fails must not be installed as READY."""

    async def scenario():
        lifecycle = ServerLifecycle()
        await run_lifespan_startup(lifecycle)

        async def broken_app(scope, receive, send):
            assert scope["type"] == "lifespan"
            await receive()
            await send({"type": "lifespan.startup.failed", "message": "boom"})

        with pytest.raises(RuntimeError, match="failed to start"):
            await lifecycle.attach(broken_app)
        assert lifecycle.state is ServerState.STARTING

    run_scenario(scenario)


def test_delegate_lifespan_shutdown_runs_on_outer_shutdown():
    """The delegate owns engine teardown; it must still be reached."""

    async def scenario():
        lifecycle = ServerLifecycle()
        driver = await run_lifespan_startup(lifecycle)
        shut_down = asyncio.Event()

        async def app(scope, receive, send):
            assert scope["type"] == "lifespan"
            await receive()
            await send({"type": "lifespan.startup.complete"})
            await receive()
            shut_down.set()
            await send({"type": "lifespan.shutdown.complete"})

        await lifecycle.attach(app)
        message = await driver.shutdown()
        assert message["type"] == "lifespan.shutdown.complete"
        assert shut_down.is_set()

    run_scenario(scenario)


# --------------------------------------------------------------------------
# serve_with_lifecycle: the end-to-end contract, over a real socket.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("loop_factory", LOOP_FACTORIES)
def test_blocking_build_never_leaves_a_probe_unanswered(loop_factory):
    """The regression this whole feature exists to prevent.

    ``build`` blocks its thread for seconds, exactly as engine
    initialization does. If it ran on the event loop, ``/health`` would
    accept connections and never answer them: a client-visible timeout,
    which is a third ambiguous state and strictly worse than the
    connection refused it replaces.
    """

    async def scenario():
        sock = bound_socket()
        port = sock.getsockname()[1]
        ready_at = None
        server = FakeServer()

        def build():
            time.sleep(FAKE_BUILD_SECONDS)
            return server

        serve = asyncio.create_task(
            serve_with_lifecycle(
                host="127.0.0.1",
                port=port,
                sockets=[sock],
                build=build,
                timeout_keep_alive=TIMEOUT_KEEP_ALIVE,
                on_ready=lambda srv: srv.register_with_disagg_cluster(),
            )
        )

        results = []
        deadline = time.monotonic() + FAKE_BUILD_SECONDS + 10.0
        try:
            # The bind()->listen() window is inherent to any server; what
            # matters is that it is milliseconds, not the whole engine
            # initialization. Wait it out, then hold every later probe to the
            # contract.
            listening_at = time.monotonic()
            while (await probe(port)).outcome == "refused":
                assert time.monotonic() - listening_at < 5.0, (
                    "server did not start listening promptly"
                )
                await asyncio.sleep(PROBE_INTERVAL_SECONDS)
            pre_listen_seconds = time.monotonic() - listening_at

            while time.monotonic() < deadline:
                result = await probe(port)
                results.append(result)
                if result.outcome == "answered" and result.status == 200:
                    ready_at = len(results) - 1
                    break
                await asyncio.sleep(PROBE_INTERVAL_SECONDS)
            # A few more probes after the flip, to catch a reset on
            # transition.
            for _ in range(10):
                results.append(await probe(port))
                await asyncio.sleep(PROBE_INTERVAL_SECONDS)
        finally:
            serve.cancel()
            try:
                await serve
            except asyncio.CancelledError:
                pass
            sock.close()

        assert ready_at is not None, "server never became READY"
        # The point of the change: the unreachable window is now the time to
        # start uvicorn, not the time to build the engine.
        assert pre_listen_seconds < FAKE_BUILD_SECONDS / 2, (
            f"port was refused for {pre_listen_seconds:.3f}s, which is not "
            "meaningfully shorter than the build"
        )
        # Enough probes landed inside the STARTING window to be meaningful.
        assert ready_at >= 10, f"only {ready_at} probes during STARTING"

        bad = [r for r in results if r.outcome != "answered"]
        assert not bad, f"probes that were not cleanly answered: {bad}"

        starting = results[:ready_at]
        assert all(r.status == 503 for r in starting), (
            f"non-503 during STARTING: {[r for r in starting if r.status != 503]}"
        )
        assert all(
            r.headers.get(SERVER_STATE_HEADER) == ServerState.STARTING.value for r in starting
        )
        assert all(r.status == 200 for r in results[ready_at:]), (
            "a probe after READY did not get 200: "
            f"{[r for r in results[ready_at:] if r.status != 200]}"
        )

        # No probe waited anywhere near the timeout: the loop stayed free.
        worst = max(r.elapsed for r in starting)
        assert worst < PROBE_TIMEOUT_SECONDS / 2, (
            f"slowest STARTING probe took {worst:.3f}s; the event loop was "
            "likely blocked by the build"
        )

        assert server.bound_to == ("127.0.0.1", port)
        assert server.registered

    run_on(loop_factory, scenario)


@pytest.mark.parametrize("loop_factory", LOOP_FACTORIES)
def test_build_failure_stops_listening(loop_factory):
    """A failed startup must end as the dead state: connection refused."""

    async def scenario():
        sock = bound_socket()
        port = sock.getsockname()[1]

        # Long enough that the STARTING window stays open across the
        # bind()->listen() wait below even on a loaded CI host.
        def build():
            time.sleep(FAKE_BUILD_SECONDS)
            raise RuntimeError("engine init exploded")

        serve = asyncio.create_task(
            serve_with_lifecycle(
                host="127.0.0.1",
                port=port,
                sockets=[sock],
                build=build,
                timeout_keep_alive=TIMEOUT_KEEP_ALIVE,
            )
        )

        # While the doomed build runs the server is honestly STARTING. Poll
        # out the bind()->listen() window rather than guessing it with a fixed
        # sleep: a sleep shorter than the window sees "refused" and fails the
        # assertion below for a reason that has nothing to do with the
        # contract under test. Same idiom as
        # test_blocking_build_never_leaves_a_probe_unanswered.
        listening_at = time.monotonic()
        while (await probe(port)).outcome == "refused":
            assert time.monotonic() - listening_at < 5.0, "server did not start listening promptly"
            await asyncio.sleep(PROBE_INTERVAL_SECONDS)

        during = await probe(port)
        assert during.outcome == "answered" and during.status == 503

        with pytest.raises(RuntimeError, match="engine init exploded"):
            await serve

        sock.close()
        after = await probe(port)
        assert after.outcome == "refused", (
            f"expected connection refused after a failed startup, got {after}"
        )

    run_on(loop_factory, scenario)


def test_build_is_abandoned_when_the_server_stops_first():
    """SIGTERM during a long startup must not wait out the build.

    uvicorn owns SIGINT/SIGTERM for as long as it is serving, which now
    includes the whole STARTING window. If the build were simply awaited,
    a terminating signal would only take effect once initialization
    finished -- reintroducing the very hang this feature removes.
    """

    async def scenario():
        stopped = asyncio.Event()
        release_build = threading.Event()
        build_finished = threading.Event()

        async def fake_uvicorn_serve():
            await stopped.wait()

        serve_task = asyncio.create_task(fake_uvicorn_serve())

        def build():
            release_build.wait(timeout=PROBE_TIMEOUT_SECONDS * 4)
            build_finished.set()
            return FakeServer()

        started = time.monotonic()
        build_task = asyncio.create_task(_build_off_event_loop(build, serve_task))
        await asyncio.sleep(0.2)
        stopped.set()

        with pytest.raises(RuntimeError, match="still initializing"):
            await asyncio.wait_for(build_task, PROBE_TIMEOUT_SECONDS)
        assert time.monotonic() - started < PROBE_TIMEOUT_SECONDS

        # Genuinely abandoned rather than awaited: the build is still running.
        assert not build_finished.is_set()
        # Release it while this loop is still alive, so the test does not
        # leave a thread whose late completion callback lands on a closed
        # loop inside some unrelated test later in the session.
        release_build.set()
        assert build_finished.wait(timeout=PROBE_TIMEOUT_SECONDS)
        await asyncio.sleep(0.05)

    run_scenario(scenario)


def test_refused_before_listening_and_after_shutdown():
    """Bookends the contract: an unbound port is refused, as is a closed one."""

    async def scenario():
        sock = bound_socket()
        port = sock.getsockname()[1]
        # Bound but not listening -- today's whole-startup behavior.
        assert (await probe(port)).outcome == "refused"

        server = FakeServer()
        serve = asyncio.create_task(
            serve_with_lifecycle(
                host="127.0.0.1",
                port=port,
                sockets=[sock],
                build=lambda: server,
                timeout_keep_alive=TIMEOUT_KEEP_ALIVE,
            )
        )
        for _ in range(int(PROBE_TIMEOUT_SECONDS / 0.05)):
            if (await probe(port)).outcome == "answered":
                break
            await asyncio.sleep(0.05)
        else:
            pytest.fail("server never started listening")

        serve.cancel()
        try:
            await serve
        except asyncio.CancelledError:
            pass
        sock.close()

        assert (await probe(port)).outcome == "refused"

    run_scenario(scenario)


def test_websocket_upgrade_is_refused_cleanly_while_starting():
    """Uvicorn builds a websocket scope from the Upgrade header, not routes.

    Answering such a scope with an HTTP response raises inside the protocol
    implementation and surfaces as 500 + a traceback, making STARTING noisier
    than either READY or dead.
    """

    async def scenario():
        lifecycle = ServerLifecycle()
        scope = {
            "type": "websocket",
            "asgi": {"version": "3.0"},
            "path": "/v1/anything",
            "headers": [(b"host", b"127.0.0.1")],
            "client": ("127.0.0.1", 1234),
        }
        sent = []

        async def receive():
            return {"type": "websocket.connect"}

        async def send(message):
            sent.append(message)

        await lifecycle(scope, receive, send)
        assert sent == [{"type": "websocket.close", "code": 1013}], sent
        assert not any(m["type"].startswith("http.") for m in sent), (
            "an HTTP response on a websocket scope raises inside uvicorn"
        )

    run_scenario(scenario)


def test_websocket_aborted_mid_handshake_is_left_alone():
    """A client that gives up before connecting must not be answered.

    After ``websocket.disconnect`` there is no handshake left to reject, and
    sending a close anyway raises inside uvicorn's protocol.
    """

    async def scenario():
        lifecycle = ServerLifecycle()
        scope = {
            "type": "websocket",
            "asgi": {"version": "3.0"},
            "path": "/v1/anything",
            "headers": [(b"host", b"127.0.0.1")],
            "client": ("127.0.0.1", 1234),
        }
        sent = []

        async def receive():
            return {"type": "websocket.disconnect", "code": 1006}

        async def send(message):
            sent.append(message)

        await lifecycle(scope, receive, send)
        assert sent == [], f"nothing should be sent after a disconnect: {sent}"

    run_scenario(scenario)


def test_on_ready_failure_raises_instead_of_exiting_zero():
    """A worker that cannot register must not look like a clean completion."""

    async def scenario():
        sock = bound_socket()
        port = sock.getsockname()[1]
        server = FakeServer(register_ok=False)

        with pytest.raises(RuntimeError, match="could not complete registration"):
            await serve_with_lifecycle(
                host="127.0.0.1",
                port=port,
                sockets=[sock],
                build=lambda: server,
                timeout_keep_alive=TIMEOUT_KEEP_ALIVE,
                on_ready=lambda srv: srv.register_with_disagg_cluster(),
            )
        sock.close()
        assert server.registered
        assert (await probe(port)).outcome == "refused"

    run_scenario(scenario)


def test_startup_failure_never_runs_two_engine_teardowns(monkeypatch):
    """The window that a bounded, off-loop teardown opened.

    When on_ready fails after a successful attach, uvicorn's shutdown already
    drives the delegate lifespan's teardown. If an out-of-band teardown also
    runs, both execute BaseLLM.shutdown() -- which takes no lock, so both see
    a non-None executor -- and drive ZeroMqQueue.close() from two threads,
    which the executor proxy documents as not ZMQ-safe (observed in the wild
    as a permanent hang in zmq_ctx_term).

    The abort bound is squeezed below the teardown duration so that, if the
    teardown were abandonable mid-flight, the second one would start on top
    of it.
    """
    monkeypatch.setattr(lifecycle_mod, "_ABORT_SHUTDOWN_TIMEOUT_SECONDS", 0.5)

    async def scenario():
        sock = bound_socket()
        port = sock.getsockname()[1]
        server = FakeServer(register_ok=False, teardown_seconds=3.0)

        with pytest.raises(RuntimeError, match="could not complete registration"):
            await serve_with_lifecycle(
                host="127.0.0.1",
                port=port,
                sockets=[sock],
                build=lambda: server,
                timeout_keep_alive=TIMEOUT_KEEP_ALIVE,
                on_ready=lambda srv: srv.register_with_disagg_cluster(),
                on_startup_failure=lambda srv: srv.shutdown_generator(),
            )
        sock.close()
        # Give any abandoned teardown time to land in the record.
        await asyncio.sleep(server._teardown_seconds + 2.0)

        overlap = overlapping_teardowns(server.teardowns)
        assert overlap is None, (
            f"two engine teardowns overlapped by {overlap[2]:.1f}s: "
            f"{overlap[0].who} and {overlap[1].who}"
        )
        assert [t.who for t in server.teardowns] == ["lifespan"], (
            "the delegate lifespan owns the teardown here; an extra "
            f"out-of-band one is redundant: {server.teardowns}"
        )

    run_scenario(scenario)


def test_abandon_bound_is_sane():
    """Pin the shipped value; every test that reaches it patches it.

    Both abandon tests monkeypatch _ABANDON_TIMEOUT_SECONDS, and every other
    test reaching _abandon() uses a cooperative delegate that finishes the
    moment cancellation lands, whatever the timeout -- so without this a typo
    of 15.0 for 1500.0 would pass the entire suite.
    """
    abandon = lifecycle_mod._ABANDON_TIMEOUT_SECONDS
    assert abandon > 0, "cancellation needs some time to take effect"
    assert abandon < lifecycle_mod._ABORT_SHUTDOWN_TIMEOUT_SECONDS, (
        "the abandon bound must be able to expire and still leave the abort path time to finish"
    )


def test_abort_bound_leaves_room_for_the_delegate_shutdown_bound():
    """Prefer letting the delegate finish cleanly over cancelling it.

    Correctness does not depend on this -- finish_delegate_lifespan()
    reports what actually happened whichever bound expires first, and the
    ordering is not guaranteed anyway because uvicorn's connection drain
    ahead of the lifespan shutdown is unbounded. But when uvicorn does reach
    the lifespan shutdown promptly, the delegate should get its whole bound
    to tear down of its own accord rather than being cancelled part-way.
    """
    assert (
        lifecycle_mod._ABORT_SHUTDOWN_TIMEOUT_SECONDS
        > lifecycle_mod._DELEGATE_SHUTDOWN_TIMEOUT_SECONDS
    ), (
        "the outer abort bound should exceed the delegate shutdown bound so "
        "a prompt uvicorn shutdown lets the delegate finish on its own"
    )


@pytest.mark.timeout(180)
def test_engine_is_torn_down_even_if_the_delegate_teardown_is_abandoned(monkeypatch):
    """Exactly one teardown, even when the delegate's own never completes.

    A delegate teardown that awaits (rather than blocking the loop) can be
    abandoned. If the out-of-band teardown is skipped because the delegate
    *started* one, nothing tears the engine down at all, and the only
    remaining path is LLM's atexit hook -- an unbounded BaseLLM.shutdown()
    on the main thread during interpreter shutdown. That is precisely the
    unbounded wedge this contract exists to remove.
    """
    monkeypatch.setattr(lifecycle_mod, "_DELEGATE_SHUTDOWN_TIMEOUT_SECONDS", 1.0)
    monkeypatch.setattr(lifecycle_mod, "_ABORT_SHUTDOWN_TIMEOUT_SECONDS", 3.0)
    monkeypatch.setattr(lifecycle_mod, "_TEARDOWN_TIMEOUT_SECONDS", 10.0)

    async def scenario():
        sock = bound_socket()
        port = sock.getsockname()[1]
        # Outlasts both bounds, so the delegate's teardown never reaches its
        # engine shutdown.
        server = FakeServer(register_ok=False, teardown_awaits=30.0)

        with pytest.raises(RuntimeError, match="could not complete registration"):
            await serve_with_lifecycle(
                host="127.0.0.1",
                port=port,
                sockets=[sock],
                build=lambda: server,
                timeout_keep_alive=TIMEOUT_KEEP_ALIVE,
                on_ready=lambda srv: srv.register_with_disagg_cluster(),
                on_startup_failure=lambda srv: srv.shutdown_generator(),
            )
        sock.close()

        assert [t.who for t in server.teardowns] == ["on_startup_failure"], (
            "the delegate's teardown was abandoned, so the out-of-band one "
            f"had to run and did not: {server.teardowns}"
        )
        assert overlapping_teardowns(server.teardowns) is None

    run_scenario(scenario)


@pytest.mark.timeout(180)
def test_engine_is_torn_down_when_uvicorn_never_starts_the_lifespan_shutdown(monkeypatch):
    """One open request is enough to strand the delegate before its teardown.

    uvicorn is configured without ``timeout_graceful_shutdown``, so it waits
    for in-flight requests *before* it sends ``lifespan.shutdown``. The
    delegate therefore never begins its teardown, and the out-of-band one
    has to run -- which in turn means the lifespan task, still suspended
    waiting for a shutdown message that never came, must be cancelled first
    so it cannot resume into a second teardown.
    """
    monkeypatch.setattr(lifecycle_mod, "_DELEGATE_SHUTDOWN_TIMEOUT_SECONDS", 1.0)
    monkeypatch.setattr(lifecycle_mod, "_ABORT_SHUTDOWN_TIMEOUT_SECONDS", 3.0)
    monkeypatch.setattr(lifecycle_mod, "_TEARDOWN_TIMEOUT_SECONDS", 10.0)

    async def scenario():
        sock = bound_socket()
        port = sock.getsockname()[1]
        server = FakeServer(register_ok=False, slow_route_seconds=120.0)
        in_flight = []

        async def register(srv):
            # Leave a request open, and give uvicorn a moment to accept it,
            # before the abort path starts.
            in_flight.append(asyncio.create_task(probe(port, "/slow", timeout=150.0)))
            await asyncio.sleep(1.0)
            return await srv.register_with_disagg_cluster()

        try:
            with pytest.raises(RuntimeError, match="could not complete registration"):
                await serve_with_lifecycle(
                    host="127.0.0.1",
                    port=port,
                    sockets=[sock],
                    build=lambda: server,
                    timeout_keep_alive=TIMEOUT_KEEP_ALIVE,
                    on_ready=register,
                    on_startup_failure=lambda srv: srv.shutdown_generator(),
                )
        finally:
            for task in in_flight:
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError, OSError):
                    await task
            sock.close()

        assert [t.who for t in server.teardowns] == ["on_startup_failure"], (
            "uvicorn never delivered lifespan.shutdown, so the delegate ran no "
            f"teardown and the out-of-band one had to: {server.teardowns}"
        )
        assert overlapping_teardowns(server.teardowns) is None

    run_scenario(scenario)


def test_out_of_band_teardown_runs_when_attach_fails(monkeypatch):
    """attach() failing means the lifespan teardown will never run."""
    monkeypatch.setattr(lifecycle_mod, "_TEARDOWN_TIMEOUT_SECONDS", 30.0)

    async def scenario():
        sock = bound_socket()
        port = sock.getsockname()[1]
        server = FakeServer()

        async def broken_app(scope, receive, send):
            assert scope["type"] == "lifespan"
            await receive()
            await send({"type": "lifespan.startup.failed", "message": "boom"})

        server.app = broken_app

        with pytest.raises(RuntimeError, match="failed to start"):
            await serve_with_lifecycle(
                host="127.0.0.1",
                port=port,
                sockets=[sock],
                build=lambda: server,
                timeout_keep_alive=TIMEOUT_KEEP_ALIVE,
                on_startup_failure=lambda srv: srv.shutdown_generator(),
            )
        sock.close()
        assert [t.who for t in server.teardowns] == ["on_startup_failure"], (
            "a delegate whose lifespan startup failed never runs its "
            f"teardown, so the engine would leak: {server.teardowns}"
        )

    run_scenario(scenario)


def test_wedged_out_of_band_teardown_is_bounded(monkeypatch):
    """The one teardown we do own must not hang startup forever."""
    monkeypatch.setattr(lifecycle_mod, "_TEARDOWN_TIMEOUT_SECONDS", 1.0)

    async def scenario():
        sock = bound_socket()
        port = sock.getsockname()[1]
        # Long enough to blow the bound, short enough to retire before the
        # test ends (pytest.ini enables threadleak).
        server = FakeServer(teardown_seconds=4.0)

        async def broken_app(scope, receive, send):
            await receive()
            await send({"type": "lifespan.startup.failed", "message": "boom"})

        server.app = broken_app

        started = time.monotonic()
        with pytest.raises(RuntimeError, match="failed to start"):
            await serve_with_lifecycle(
                host="127.0.0.1",
                port=port,
                sockets=[sock],
                build=lambda: server,
                timeout_keep_alive=TIMEOUT_KEEP_ALIVE,
                on_startup_failure=lambda srv: srv.shutdown_generator(),
            )
        elapsed = time.monotonic() - started
        sock.close()

        assert elapsed < server._teardown_seconds, (
            f"startup waited out the whole teardown instead of bounding it ({elapsed:.1f}s)"
        )

        # Let the abandoned teardown finish so no daemon thread outlives us.
        deadline = time.monotonic() + 30.0
        while not server.teardowns and time.monotonic() < deadline:
            await asyncio.sleep(0.1)
        assert server.teardowns, "the abandoned teardown never completed"

    run_scenario(scenario)


def test_shutdown_generator_tears_the_engine_down():
    """OpenAIServer.shutdown_generator must actually shut the engine down.

    Constructed without __init__: a real one needs a real engine. This
    covers the method the lifecycle path calls; the lifespan teardown line
    itself still has no unit coverage (it needs a live OpenAIServer).
    """
    from tensorrt_llm.serve.openai_server import OpenAIServer

    calls = []

    class FakeGenerator:
        def shutdown(self):
            calls.append("shutdown")

    server = OpenAIServer.__new__(OpenAIServer)
    server.generator = FakeGenerator()
    server.shutdown_generator()
    assert calls == ["shutdown"]


def test_failed_attach_still_routes_shutdown_into_the_delegate():
    """attach() registers the delegate lifespan *before* awaiting startup."""

    async def scenario():
        lifecycle = ServerLifecycle()
        driver = await run_lifespan_startup(lifecycle)
        shutdown_seen = asyncio.Event()

        async def half_broken_app(scope, receive, send):
            assert scope["type"] == "lifespan"
            await receive()
            await send({"type": "lifespan.startup.failed", "message": "boom"})
            # A real delegate can already own resources at this point.
            await receive()
            shutdown_seen.set()

        with pytest.raises(RuntimeError, match="failed to start"):
            await lifecycle.attach(half_broken_app)
        assert lifecycle.state is ServerState.STARTING

        # The delegate is registered even though attach() raised, so the
        # outer shutdown still reaches it.
        message = await driver.shutdown()
        assert message["type"] == "lifespan.shutdown.complete", (
            "a startup failure must not be re-raised out of shutdown, where "
            "it would mask whatever is being unwound"
        )
        assert shutdown_seen.is_set()

    run_scenario(scenario)


def test_normal_shutdown_cancels_a_lifespan_that_will_not_finish(monkeypatch):
    """The shutdown bound must cancel, not merely stop waiting.

    finish_delegate_lifespan() only runs on the abort path, so on a normal
    shutdown nothing else settles the lifespan afterwards. A task left
    suspended mid-teardown could resume later and run the engine shutdown
    alongside LLM's atexit one.
    """
    monkeypatch.setattr(lifecycle_mod, "_DELEGATE_SHUTDOWN_TIMEOUT_SECONDS", 0.5)

    async def scenario():
        lifecycle = ServerLifecycle()
        driver = await run_lifespan_startup(lifecycle)

        async def never_finishes(scope, receive, send):
            await receive()
            await send({"type": "lifespan.startup.complete"})
            await receive()
            await asyncio.sleep(3600)

        await lifecycle.attach(never_finishes)
        delegate_task = lifecycle._delegate_lifespan._task

        message = await driver.shutdown()
        assert message["type"] == "lifespan.shutdown.complete"
        assert delegate_task.done(), (
            "a lifespan that would not finish must be cancelled, not left "
            "suspended where it could resume during interpreter shutdown"
        )

    run_scenario(scenario)


def test_finish_delegate_lifespan_is_false_when_nothing_was_attached():
    """No delegate means no delegate teardown, so the caller owns it."""

    async def scenario():
        lifecycle = ServerLifecycle()
        assert await lifecycle.finish_delegate_lifespan() is False

    run_scenario(scenario)


@pytest.mark.timeout(60)
def test_out_of_band_teardown_runs_when_attach_is_never_reached():
    """A failure between building the engine and attaching must not leak it.

    ``record_address`` runs after the build and before ``attach``; blowing up
    there leaves a built engine with no lifespan to tear it down.
    """

    async def scenario():
        sock = bound_socket()
        port = sock.getsockname()[1]
        server = FakeServer()

        def explode(host, port):
            raise RuntimeError("record_address exploded")

        server.record_address = explode

        with pytest.raises(RuntimeError, match="record_address exploded"):
            await serve_with_lifecycle(
                host="127.0.0.1",
                port=port,
                sockets=[sock],
                build=lambda: server,
                timeout_keep_alive=TIMEOUT_KEEP_ALIVE,
                on_startup_failure=lambda srv: srv.shutdown_generator(),
            )
        sock.close()

        assert [t.who for t in server.teardowns] == ["on_startup_failure"], (
            "attach was never reached, so no lifespan exists to tear the "
            f"engine down: {server.teardowns}"
        )

    run_scenario(scenario)


@contextlib.asynccontextmanager
async def attached_lifespan_that_swallows_cancellation(lifecycle):
    """Attach a delegate that ignores cancellation, and clean it up after.

    Models the shape the real teardown contains twice -- cancel a
    background task, await it, swallow the CancelledError -- which is what
    makes ``Task.cancel()`` insufficient on its own.
    """
    release = asyncio.Event()
    # Genuinely self-limiting: the deadline is re-checked every slice, so the
    # task stops ignoring cancellation on its own after
    # SWALLOW_CANCELLATION_SECONDS whether or not anyone cancels it again.
    # Sleeping once for an hour instead would make the deadline unreachable
    # and leave the timeout marker as the only thing ending the run -- the
    # hang-instead-of-fail pattern these tests exist to prevent.
    deadline = time.monotonic() + SWALLOW_CANCELLATION_SECONDS

    async def swallows_cancellation(scope, receive, send):
        await receive()
        await send({"type": "lifespan.startup.complete"})
        while not release.is_set() and time.monotonic() < deadline:
            try:
                await asyncio.sleep(SWALLOW_SLICE_SECONDS)
            except asyncio.CancelledError:
                pass

    await lifecycle.attach(swallows_cancellation)
    task = lifecycle._delegate_lifespan._task
    try:
        yield task
    finally:
        # Let it go, or asyncio.run() would hang cancelling it at teardown.
        release.set()
        task.cancel()
        with contextlib.suppress(BaseException):
            await task


@pytest.mark.timeout(120)
def test_abandon_is_bounded_against_a_lifespan_that_swallows_cancellation(monkeypatch):
    """Task.cancel() delivers once; a task that catches it keeps running.

    An unbounded wait here would be the one unbounded await on a path whose
    whole purpose is boundedness -- the process would never exit and the
    startup failure would never be reported.
    """
    monkeypatch.setattr(lifecycle_mod, "_ABANDON_TIMEOUT_SECONDS", 1.0)

    async def scenario():
        lifecycle = ServerLifecycle()
        await run_lifespan_startup(lifecycle)

        async with attached_lifespan_that_swallows_cancellation(lifecycle) as task:
            started = time.monotonic()
            # Reported as "torn down" so the caller does not start a second
            # teardown alongside a task that is demonstrably still alive.
            assert await lifecycle.finish_delegate_lifespan() is True
            elapsed = time.monotonic() - started

            # Below SWALLOW_CANCELLATION_SECONDS on purpose: an unbounded
            # wait would return only once the stand-in gave up, and must not
            # be able to slip under this threshold.
            assert elapsed < SWALLOW_CANCELLATION_SECONDS / 2, (
                f"finish_delegate_lifespan() was not bounded ({elapsed:.1f}s)"
            )
            assert not task.done(), "the premise is a task that outlives cancellation"

    run_scenario(scenario)


@pytest.mark.timeout(120)
def test_finish_delegate_lifespan_propagates_caller_cancellation(monkeypatch):
    """A Ctrl-C landing here must not be mistaken for our own cancel.

    uvloop's Runner cancels the main task on KeyboardInterrupt. Swallowing
    that would let the abort path carry on as though nothing happened, and
    would defeat the deliberate ``except CancelledError: raise`` guards on
    the way out.
    """
    monkeypatch.setattr(lifecycle_mod, "_ABANDON_TIMEOUT_SECONDS", 30.0)

    async def scenario():
        lifecycle = ServerLifecycle()
        await run_lifespan_startup(lifecycle)

        async with attached_lifespan_that_swallows_cancellation(lifecycle):
            caller = asyncio.create_task(lifecycle.finish_delegate_lifespan())
            await asyncio.sleep(0.2)
            assert not caller.done(), "expected it to still be waiting"

            caller.cancel()
            with pytest.raises(asyncio.CancelledError):
                await caller

    run_scenario(scenario)


def test_cooperative_delegate_shutdown_is_bounded(monkeypatch):
    """A delegate that yields but never finishes shutting down is abandoned."""
    monkeypatch.setattr(lifecycle_mod, "_DELEGATE_SHUTDOWN_TIMEOUT_SECONDS", 1.0)

    async def scenario():
        lifecycle = ServerLifecycle()
        driver = await run_lifespan_startup(lifecycle)

        async def never_finishes(scope, receive, send):
            await receive()
            await send({"type": "lifespan.startup.complete"})
            await receive()
            await asyncio.sleep(3600)  # yields, but never completes

        await lifecycle.attach(never_finishes)
        started = time.monotonic()
        message = await driver.shutdown()
        elapsed = time.monotonic() - started

        assert message["type"] == "lifespan.shutdown.complete"
        assert elapsed < 20.0, f"shutdown bound not enforced ({elapsed:.1f}s)"

    run_scenario(scenario)


# --------------------------------------------------------------------------
# The shared test harness must survive the new contract.
#
# RemoteOpenAIServer polls /health until it returns 200. Before this feature a
# still-starting server refused the connection, so every poll raised and the
# deadline (which lived only on the exception path) was enforced. Now the
# server answers 503 instead, so the deadline has to be enforced on the
# non-exception path too -- otherwise a hung engine initialization, the exact
# failure this feature exists to make visible, would spin the harness forever
# and leak the trtllm-serve child.
# --------------------------------------------------------------------------


class _StartingForeverHandler(BaseHTTPRequestHandler):
    """Answers 503 the way a server whose engine never finishes loading does."""

    ready_after = None  # set on the subclass; None means "never ready"
    requests_served = 0

    def do_GET(self):  # noqa: N802 (BaseHTTPRequestHandler API)
        cls = type(self)
        cls.requests_served += 1
        ready = cls.ready_after is not None and cls.requests_served > cls.ready_after
        body = b"" if ready else b'{"error": {"code": "starting"}}'
        self.send_response(200 if ready else 503)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        if not ready:
            self.send_header(SERVER_STATE_HEADER, ServerState.STARTING.value)
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass  # keep pytest output readable


@contextlib.contextmanager
def health_server(ready_after=None):
    """Serve /health on an ephemeral port; 200 only after ``ready_after`` polls."""
    handler = type(
        "_Handler", (_StartingForeverHandler,), {"ready_after": ready_after, "requests_served": 0}
    )
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield httpd.server_address[1]
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=PROBE_TIMEOUT_SECONDS)


class FakePopen:
    """Just enough of subprocess.Popen for the harness's liveness check."""

    def __init__(self, returncode=None):
        self.returncode = returncode
        self.terminated = False

    def poll(self):
        return self.returncode

    def terminate(self):
        self.terminated = True
        self.returncode = -15

    def wait(self, timeout=None):
        return self.returncode


def remote_server(port, proc):
    """A RemoteOpenAIServer wired to an existing port, without launching one."""
    from llmapi.apps.openai_server import RemoteOpenAIServer

    server = RemoteOpenAIServer.__new__(RemoteOpenAIServer)
    server.host = "127.0.0.1"
    server.port = port
    server.rank = 0
    server.proc = proc
    server.extra_config_file = None
    server.log_path = None
    server.log_file = None
    return server


@pytest.mark.timeout(60)
def test_harness_times_out_against_a_server_that_is_503_forever():
    """The regression: 503 must not be an infinite, deadline-free wait."""
    proc = FakePopen()
    with health_server() as port:
        server = remote_server(port, proc)
        started = time.monotonic()
        with pytest.raises(RuntimeError, match="failed to start"):
            server.wait_for_server(timeout=2.0)
        elapsed = time.monotonic() - started

    assert elapsed < 30.0, f"deadline was not enforced (waited {elapsed:.1f}s)"
    assert proc.terminated, (
        "the trtllm-serve child must be terminated on a startup timeout, "
        "otherwise it survives the test that gave up on it"
    )


@pytest.mark.timeout(60)
def test_harness_detects_a_dead_process_behind_a_503():
    """Liveness must be checked on the 503 path, not only on the error path."""
    proc = FakePopen(returncode=1)
    with health_server() as port:
        server = remote_server(port, proc)
        started = time.monotonic()
        with pytest.raises(RuntimeError, match="exited unexpectedly"):
            # A timeout far larger than the test's patience: the only way this
            # returns quickly is via the liveness check.
            server.wait_for_server(timeout=600.0)
        elapsed = time.monotonic() - started

    assert elapsed < 30.0, f"liveness check was not reached (waited {elapsed:.1f}s)"


@pytest.mark.timeout(60)
def test_harness_returns_once_health_flips_to_200():
    """The normal path still works: 503 while starting, then ready."""
    with health_server(ready_after=3) as port:
        server = remote_server(port, FakePopen())
        server.wait_for_server(timeout=30.0)  # must not raise


# --------------------------------------------------------------------------
# Attached frontends must be visible to the caller's cleanup from the moment
# they exist, not only once the whole group signals READY.
# --------------------------------------------------------------------------


def fake_multi_frontend_llm(monkeypatch):
    """Patch in the minimum an attached-frontend spawn needs to proceed."""
    from tensorrt_llm.executor import proxy as proxy_mod

    class FakeProxy:
        def multi_frontend_attach_info(self):
            return {"endpoint": "ipc:///tmp/fake"}  # nosec B108

    class FakeLLM:
        _executor = FakeProxy()

    monkeypatch.setattr(proxy_mod, "GenerationExecutorProxy", FakeProxy)
    return FakeLLM()


# Two stand-in frontends, deliberately different.
#
# The heavy one calls the *real* child-side watchdog, so the shipped
# _watch_launcher_liveness_from_env and its call site are genuinely covered;
# it costs a full `import tensorrt_llm` (seconds and gigabytes), so exactly
# one test uses it.
#
# The light one imports nothing and is used where the property under test
# belongs to the *launcher* side -- whether spawned children outlive the
# thread that created them -- and the child is only required to stay alive.
HEAVY_FRONTEND_SRC = """
import os, sys, time
sys.path.insert(0, {trtllm_parent!r})
from tensorrt_llm.commands import serve

serve._watch_launcher_liveness_from_env()
{after_arming}
time.sleep(300)
"""

LIGHT_FRONTEND_SRC = """
import os, time
os.write(int(os.environ["TLLM_FRONTEND_READY_FD"]), b"R")
os.close(int(os.environ["TLLM_FRONTEND_READY_FD"]))
time.sleep(300)
"""


def heavy_frontend_src(after_arming: str = "") -> str:
    """Render the real-watchdog frontend, importing tensorrt_llm from this tree."""
    import tensorrt_llm

    parent = str(pathlib.Path(tensorrt_llm.__file__).resolve().parent.parent)
    return HEAVY_FRONTEND_SRC.format(trtllm_parent=parent, after_arming=after_arming)


@pytest.mark.timeout(180)
def test_frontends_survive_the_spawning_thread_exiting(monkeypatch):
    """The regression that PR_SET_PDEATHSIG introduced.

    PR_SET_PDEATHSIG fires when the *creating thread* exits, not when the
    parent process dies. Frontends are spawned from ``build_frontend`` on the
    ``trtllm_engine_init`` daemon thread, which exits as soon as startup
    finishes -- so an armed PDEATHSIG killed every child immediately after a
    perfectly healthy start, silently degrading num_serve_frontends=K to 1.

    The property is entirely on the launcher side (what
    _spawn_attached_frontends does to its children), so a light stand-in
    child is enough here; the real child-side watchdog is covered by
    test_frontend_exits_when_the_launcher_is_killed.
    """
    from tensorrt_llm.commands import serve as serve_mod

    llm = fake_multi_frontend_llm(monkeypatch)
    real_popen = subprocess.Popen

    def fake_popen(args, env=None, pass_fds=(), **kwargs):
        # Same fds and env, but a cheap stand-in instead of re-execing
        # trtllm-serve (which would need a GPU and a model).
        return real_popen(
            [sys.executable, "-c", LIGHT_FRONTEND_SRC],
            env=env,
            pass_fds=pass_fds,
            **kwargs,
        )

    monkeypatch.setattr(serve_mod.subprocess, "Popen", fake_popen)

    children = []
    failure = []

    def spawn_from_worker_thread():
        try:
            serve_mod._spawn_attached_frontends(llm, 3, children)
        except BaseException as exc:  # reported on the main thread
            failure.append(exc)

    # Exactly how production spawns them: on the engine-init thread, which
    # then exits.
    thread = threading.Thread(target=spawn_from_worker_thread, name="trtllm_engine_init")
    thread.start()
    thread.join(timeout=120)

    try:
        assert not thread.is_alive(), "spawn did not finish"
        assert not failure, f"spawn failed: {failure!r}"
        assert len(children) == 2

        # The spawning thread is now gone while this process lives on.
        time.sleep(2.0)
        codes = [child.poll() for child in children]
        assert codes == [None, None], (
            f"attached frontends died when the spawning thread exited: {codes}"
            " -- num_serve_frontends would silently degrade to 1"
        )
    finally:
        for child in children:
            child.kill()
            child.wait(timeout=10)


@pytest.mark.timeout(60)
def test_watchdog_from_env_consumes_the_pid_and_arms_the_watchdog(monkeypatch):
    """Covers the real child-side entry point's wiring.

    Stubs out the watchdog itself rather than arming a real one: a live
    watchdog thread would either leak (pytest.ini sets threadleak) or, on
    deciding the launcher is gone, call the real os._exit and take this
    process down.
    """
    from tensorrt_llm.commands import serve as serve_mod

    armed = []
    monkeypatch.setattr(serve_mod, "_watch_launcher_liveness", armed.append)
    monkeypatch.setenv("TLLM_FRONTEND_LAUNCHER_PID", "77")

    serve_mod._watch_launcher_liveness_from_env()

    assert armed == [77], "the launcher's pid must be armed"
    # Consumed, so a child that re-execs cannot arm a stale pid twice.
    assert "TLLM_FRONTEND_LAUNCHER_PID" not in os.environ


@pytest.mark.timeout(60)
def test_watchdog_from_env_is_a_no_op_without_the_pid(monkeypatch):
    """A frontend started outside a multi-frontend group must not arm."""
    from tensorrt_llm.commands import serve as serve_mod

    armed = []
    monkeypatch.setattr(serve_mod, "_watch_launcher_liveness", armed.append)
    monkeypatch.delenv("TLLM_FRONTEND_LAUNCHER_PID", raising=False)

    serve_mod._watch_launcher_liveness_from_env()

    assert armed == []


@pytest.mark.timeout(60)
def test_watchdog_from_env_ignores_a_malformed_pid(monkeypatch):
    """A pid that will not parse is not evidence of anything.

    Arming on ``int("")`` would raise out of frontend startup and fail a
    healthy group; treating it as a dead launcher would kill one.
    """
    from tensorrt_llm.commands import serve as serve_mod

    armed = []
    monkeypatch.setattr(serve_mod, "_watch_launcher_liveness", armed.append)
    monkeypatch.setenv("TLLM_FRONTEND_LAUNCHER_PID", "not-a-pid")

    serve_mod._watch_launcher_liveness_from_env()

    assert armed == []


@pytest.mark.timeout(60)
@pytest.mark.parametrize(
    "launcher_gone,expected_exits",
    [(True, [1]), (False, [])],
    ids=["launcher-gone-exits", "cannot-tell-stays-alive"],
)
def test_watcher_exits_only_on_a_definite_answer(monkeypatch, launcher_gone, expected_exits):
    """The watcher has to act on the answer, not just ask the question.

    Exiting when the check could not decide would kill a healthy frontend --
    the same false-positive-kill class as PR_SET_PDEATHSIG.
    """
    from tensorrt_llm.commands import serve as serve_mod

    exits = []
    monkeypatch.setattr(os, "_exit", exits.append)
    monkeypatch.setattr(serve_mod, "_wait_for_launcher_exit", lambda pid: launcher_gone)

    serve_mod._watch_launcher_liveness(4242)

    # Retire the thread before monkeypatch restores the real os._exit, and
    # before threadleak looks.
    deadline = time.monotonic() + 10.0
    for thread in list(threading.enumerate()):
        if thread.name.startswith("trtllm_launcher_watchdog"):
            thread.join(timeout=max(0.0, deadline - time.monotonic()))
    assert not [
        t
        for t in threading.enumerate()
        if t.name.startswith("trtllm_launcher_watchdog") and t.is_alive()
    ]

    assert exits == expected_exits


@pytest.mark.timeout(60)
@pytest.mark.parametrize("launcher_pid", [0, 1], ids=["zero", "init"])
def test_watchdog_does_not_report_an_unusable_launcher_pid_as_death(launcher_pid):
    """A pid reparenting cannot move us away from is not evidence of death.

    ``1`` is both what reparenting produces and a plausible launcher pid
    inside a container, so ``getppid() != launcher_pid`` can never become
    true; answering "the launcher died" would kill a healthy frontend --
    the same false-positive-kill class as PR_SET_PDEATHSIG. Calls the
    blocking primitive directly rather than arming a thread, so a
    regression here cannot os._exit the test process.
    """
    from tensorrt_llm.commands import serve as serve_mod

    assert serve_mod._wait_for_launcher_exit(launcher_pid) is False, (
        "an undecidable launcher pid must not be reported as launcher death"
    )


# Stands in for the launcher: it spawns the frontend as its own child, just
# as _spawn_attached_frontends does, and then does nothing. Being its parent
# is the whole of the launcher's side of the contract -- killing it is what
# makes the kernel reparent the frontend.
WATCHDOG_HOLDER_SRC = """
import os, subprocess, sys, time

life_fd = int(os.environ["FRONTEND_LIFE_FD"])
subprocess.Popen(
    [sys.executable, "-c", os.environ["FRONTEND_SRC"]],
    env={**os.environ, "TLLM_FRONTEND_LAUNCHER_PID": str(os.getpid())},
    pass_fds=(life_fd,),
)
# Hand the liveness pipe over entirely: the frontend is now its only holder,
# so the test reads EOF exactly when the frontend exits.
os.close(life_fd)
time.sleep(300)
"""


@pytest.mark.timeout(600)
def test_frontend_exits_when_the_launcher_is_killed():
    """End-to-end: SIGKILL the frontend's parent; the frontend must go.

    An orphan keeps the shared SO_REUSEPORT port bound while being unable to
    serve, which is worse than not listening at all.

    The frontend is a grandchild here, so Popen.poll() cannot see it and
    os.kill(pid, 0) would race pid reuse. Instead it inherits one end of a
    pipe this process holds the other end of: EOF is the frontend's exit,
    with no polling and no ambiguity.
    """
    with tempfile.TemporaryDirectory() as tmp:
        marker = pathlib.Path(tmp) / "armed"
        child_src = heavy_frontend_src(f"open({str(marker)!r}, 'w').write('armed')")

        life_read, life_write = os.pipe()
        holder = None
        try:
            holder = subprocess.Popen(
                [sys.executable, "-c", WATCHDOG_HOLDER_SRC],
                env={
                    # Same stripping production does before re-execing a
                    # frontend: an inherited MPI/Slurm rank identity would
                    # make the child's mpi4py try to join the launcher's job.
                    **split_mpi_env()[0],
                    "TLLM_DISABLE_MPI": "1",
                    "FRONTEND_LIFE_FD": str(life_write),
                    "FRONTEND_SRC": child_src,
                },
                pass_fds=(life_write,),
            )
            os.close(life_write)
            life_write = None

            deadline = time.monotonic() + 420
            while not marker.exists():
                assert time.monotonic() < deadline, "frontend never armed its watchdog"
                assert holder.poll() is None, (
                    f"launcher stand-in exited before the frontend armed: {holder.returncode}"
                )
                time.sleep(0.5)

            # No spurious kill: several poll intervals with a live launcher.
            readable, _, _ = select.select([life_read], [], [], 5.0)
            assert not readable, "frontend exited while its launcher was alive"

            holder.kill()
            holder.wait(timeout=30)
            killed_at = time.monotonic()

            readable, _, _ = select.select([life_read], [], [], 30.0)
            assert readable and os.read(life_read, 1) == b"", (
                "frontend survived the launcher being killed; it would sit "
                "on the shared port unable to serve"
            )
            # The poll interval is the whole of the detection latency.
            assert time.monotonic() - killed_at < 10.0
        finally:
            for fd in (life_read, life_write):
                if fd is not None:
                    with contextlib.suppress(OSError):
                        os.close(fd)
            if holder is not None and holder.poll() is None:
                holder.kill()
                holder.wait(timeout=30)


@pytest.mark.timeout(60)
def test_launcher_watchdog_returns_when_the_parent_link_changes(monkeypatch):
    """The child's half of the contract: a new ppid means the launcher is gone.

    Reparenting is the one signal that survives SIGKILL, and it is one-way,
    so the watchdog must wait on it and on nothing else.
    """
    from tensorrt_llm.commands import serve as serve_mod

    ppid = [4242]
    monkeypatch.setattr(serve_mod.os, "getppid", lambda: ppid[0])
    monkeypatch.setattr(serve_mod, "_LAUNCHER_POLL_INTERVAL_SECONDS", 0.01)
    returned = threading.Event()

    def wait():
        serve_mod._wait_for_launcher_exit(4242)
        returned.set()

    thread = threading.Thread(target=wait, daemon=True)
    thread.start()
    try:
        # While the launcher is still our parent, the child must stay put.
        assert not returned.wait(timeout=1.0), "watchdog fired while the launcher was still alive"
        # What the kernel does when the launcher dies, by any means
        # including SIGKILL: reparent us to init or the nearest subreaper.
        ppid[0] = 1
        assert returned.wait(timeout=10.0), "watchdog did not fire after reparenting"
    finally:
        thread.join(timeout=10.0)
        assert not thread.is_alive()


def test_spawned_frontends_are_visible_before_the_group_is_ready(monkeypatch):
    """An abort during the READY wait must not orphan already-spawned children.

    Each child holds the shared SO_REUSEPORT port, and the READY wait can
    last minutes (TLLM_FRONTEND_READY_TIMEOUT). If the caller's cleanup list
    were only populated on return, an abort inside that window would leave
    every child running as a listener that can never serve.
    """
    from tensorrt_llm.commands import serve as serve_mod

    llm = fake_multi_frontend_llm(monkeypatch)
    spawned = []

    class FakeChild:
        def __init__(self, *args, **kwargs):
            self.pid = 40000 + len(spawned)
            self.returncode = None
            self.terminated = False
            spawned.append(self)

        def poll(self):
            return self.returncode

        def terminate(self):
            self.terminated = True
            self.returncode = -15

        def kill(self):
            self.terminate()

        def wait(self, timeout=None):
            return self.returncode

    monkeypatch.setattr(serve_mod.subprocess, "Popen", FakeChild)
    # Bounds the test if the group somehow neither fails nor becomes ready.
    monkeypatch.setenv("TLLM_FRONTEND_READY_TIMEOUT", "5")

    children = []
    # No real child exists to hold the READY pipe open, so the group fails --
    # as "exited before signaling READY" or, if that races, on the deadline.
    # Either way the spawn is aborted, which is the situation under test.
    with pytest.raises(RuntimeError, match="(?i)attached frontend"):
        serve_mod._spawn_attached_frontends(llm, 3, children)

    assert len(spawned) == 2, "expected num_frontends - 1 children"
    assert children == spawned, (
        "the caller's cleanup list must hold every child that was created, "
        "so an abort during the READY wait can terminate them"
    )
    assert all(child.terminated for child in spawned)


def test_spawned_frontends_are_told_the_launchers_pid(monkeypatch):
    """The launcher's half of the watchdog contract.

    The child compares getppid() against this value, so a wrong or missing
    one either disarms the watchdog (orphans on the shared port) or fires it
    immediately (kills a healthy group).
    """
    from tensorrt_llm.commands import serve as serve_mod

    llm = fake_multi_frontend_llm(monkeypatch)
    envs = []

    class FakeChild:
        def __init__(self, args, env=None, **kwargs):
            envs.append(env)
            self.pid = 40000 + len(envs)
            self.returncode = None

        def poll(self):
            return self.returncode

        def terminate(self):
            self.returncode = -15

        kill = terminate

        def wait(self, timeout=None):
            return self.returncode

    monkeypatch.setattr(serve_mod.subprocess, "Popen", FakeChild)
    monkeypatch.setenv("TLLM_FRONTEND_READY_TIMEOUT", "5")

    # No real child exists to hold the READY pipe open, so the group fails;
    # the env handed to each Popen is what is under test.
    with pytest.raises(RuntimeError, match="(?i)attached frontend"):
        serve_mod._spawn_attached_frontends(llm, 3, [])

    assert len(envs) == 2
    assert [env.get("TLLM_FRONTEND_LAUNCHER_PID") for env in envs] == [str(os.getpid())] * 2


# --------------------------------------------------------------------------
# The test harness that polls /health has to fail on a dead server rather
# than sit out its (hours-long) startup deadline.
# --------------------------------------------------------------------------


@pytest.mark.timeout(120)
@pytest.mark.parametrize("exit_code", [0, 1], ids=["clean-exit", "crash"])
def test_server_harness_fails_fast_on_any_exit_during_startup(monkeypatch, exit_code):
    """During startup *any* exit is a failure, status 0 included.

    Excusing a clean exit leaves the poller hammering a refused port until
    MAX_SERVER_START_WAIT_S (two hours) elapses, so a server that died in a
    second is reported as a CI timeout two hours later.
    """
    from llmapi.apps import openai_server as harness_mod

    class ExitedServer(harness_mod.RemoteOpenAIServer):
        HEALTH_POLL_INTERVAL_S = 0.01

        def __init__(self, code):
            self.rank = 0
            self.terminated = False

            class Proc:
                def poll(self):
                    return code

            self.proc = Proc()

        def terminate(self):
            self.terminated = True

    def refused(url, timeout=None):
        raise harness_mod.requests.ConnectionError("connection refused")

    monkeypatch.setattr(harness_mod.requests, "get", refused)

    server = ExitedServer(exit_code)
    started = time.monotonic()
    with pytest.raises(RuntimeError, match=f"exited unexpectedly with code {exit_code}"):
        server._wait_for_server(url="http://127.0.0.1:1/health", timeout=5.0)
    assert time.monotonic() - started < 2.0, "the exit must be noticed on the first poll"
    assert not server.terminated
