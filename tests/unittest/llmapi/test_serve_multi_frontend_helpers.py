# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""CPU tests for trtllm-serve multi-frontend serving, against a fake launcher."""

import asyncio
import http.client
import os
import signal
import socket
import subprocess
import sys
import threading
import time
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest import mock

import msgspec
import pytest
from aiohttp import web
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.routing import Match

from tensorrt_llm._utils import AdjustedSteadyClock
from tensorrt_llm.llmapi.disagg_utils import DisaggClusterConfig, ServerRole
from tensorrt_llm.serve import _frontend_processes, openai_server
from tensorrt_llm.serve._frontend_processes import _watch_attached_frontends
from tensorrt_llm.serve.multi_frontend import (
    FORWARDED_ROUTES,
    AttachedFrontendWatchdog,
    LauncherForwarder,
    SharedClockOffset,
)
from tensorrt_llm.serve.openai_server import OpenAIServer

pytestmark = pytest.mark.cpu_only


class _FakeLauncher:
    """aiohttp app on a Unix socket standing in for the launcher's uvicorn."""

    def __init__(self, path: str, health_status: int = 200) -> None:
        self.path = path
        self.health_status = health_status
        self.health_hits = 0
        self.seen = []
        self._loop = asyncio.new_event_loop()
        self._ready = threading.Event()
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()
        assert self._ready.wait(10), "fake launcher did not start"

    def _serve(self) -> None:
        asyncio.set_event_loop(self._loop)

        async def echo(request: web.Request) -> web.Response:
            self.seen.append(
                (request.method, request.path_qs, await request.read(), dict(request.headers))
            )
            return web.json_response(
                {"path": request.path_qs}, status=201 if request.method == "POST" else 200
            )

        async def health(_: web.Request) -> web.Response:
            self.health_hits += 1
            return web.Response(status=self.health_status)

        app = web.Application()
        app.router.add_get("/health", health)
        app.router.add_route("*", "/{tail:.*}", echo)
        self._runner = web.AppRunner(app)
        self._loop.run_until_complete(self._runner.setup())
        self._loop.run_until_complete(web.UnixSite(self._runner, self.path).start())
        self._ready.set()
        self._loop.run_forever()

    def close(self) -> None:
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(10)
        self._loop.run_until_complete(self._runner.cleanup())
        self._loop.close()


def _attached_server(forwarder: LauncherForwarder) -> OpenAIServer:
    """An attached-frontend OpenAIServer with just enough state to route."""

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        yield
        await forwarder.close()

    server = object.__new__(OpenAIServer)
    server.app = FastAPI(lifespan=lifespan)
    server.generator = SimpleNamespace(
        _executor=SimpleNamespace(resource_governor_queue=None),
        args=SimpleNamespace(return_perf_metrics=False),
    )
    server.use_harmony = False
    server._enable_rl_control_endpoints = False
    server._enable_runtime_control_endpoints = False
    server.resource_governor = None
    server._launcher_forwarder = forwarder
    # The order OpenAIServer.__init__ registers them in.
    server._register_launcher_forwarding_routes()
    server.register_routes()
    return server


def test_attached_server_forwards_launcher_owned_routes(tmp_path) -> None:
    launcher = _FakeLauncher(str(tmp_path / "launcher.sock"))
    forwarder = LauncherForwarder(launcher.path)
    server = _attached_server(forwarder)

    # Every launcher-owned route resolves to the forwarder before its local handler.
    for method, template in FORWARDED_ROUTES:
        path = template.replace("{rest:path}", "b1/cancel")
        scope = {"type": "http", "method": method, "path": path, "path_params": {}}
        first = next(r for r in server.app.routes if r.matches(scope)[0] == Match.FULL)
        assert first.endpoint == forwarder.forward, f"{method} {path} is answered locally"

    body = msgspec.msgpack.encode({"requests": []})
    try:
        with TestClient(server.app) as client:
            r = client.get("/metrics?limit=3")
            assert (r.status_code, r.json()) == (200, {"path": "/metrics?limit=3"})
            # The body is replayed byte for byte, so its codec header must travel
            # with it, and so must auth headers a launcher --middleware checks.
            r = client.post(
                "/v1/messages/batches",
                content=body,
                headers={"x-trtllm-msgpack": "1", "x-api-key": "secret"},
            )
            assert r.status_code == 201
            assert client.get("/version").status_code == 200  # stays local
    finally:
        launcher.close()
    assert len(launcher.seen) == 2
    _, path, seen_body, headers = launcher.seen[-1]
    headers = {k.lower(): v for k, v in headers.items()}
    assert (path, seen_body) == ("/v1/messages/batches", body)
    assert headers["x-trtllm-msgpack"] == "1" and headers["x-api-key"] == "secret"
    assert headers["host"] != "testserver"  # hop-by-hop headers are not replayed

    with TestClient(server.app) as client:
        r = client.get("/metrics")
        assert r.status_code == 503 and "unavailable" in r.json()["error"]


async def _wait_for(predicate, timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        assert time.monotonic() < deadline, "condition not reached in time"
        await asyncio.sleep(0.01)


def test_watchdog_counts_health_failures_only_after_the_launcher_answered(tmp_path) -> None:
    path = str(tmp_path / "launcher.sock")
    deaths = []

    async def run() -> None:
        forwarder = LauncherForwarder(path)
        watchdog = AttachedFrontendWatchdog(
            forwarder,
            os.getppid(),
            deaths.append,
            ppid_interval_s=0.01,
            health_interval_s=0.02,
            health_failures=2,
        )
        task = asyncio.create_task(watchdog.run())
        launcher = None
        try:
            # The launcher starts listening after its attached frontends: ~25
            # failed probes against a socket nobody serves yet are not fatal.
            await asyncio.sleep(0.5)
            launcher = await asyncio.to_thread(_FakeLauncher, path, 503)
            await _wait_for(lambda: launcher.health_hits >= 3)
            assert not deaths and not task.done()
            launcher.health_status = 200
            await _wait_for(lambda: launcher.health_hits >= 4)
            launcher.health_status = 503
            await asyncio.wait_for(task, timeout=10)
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            await forwarder.close()
            if launcher is not None:
                await asyncio.to_thread(launcher.close)

    asyncio.run(run())
    assert len(deaths) == 1 and "failed 2 consecutive" in str(deaths[0])


def test_watchdog_notices_a_gone_launcher_process(tmp_path) -> None:
    deaths = []

    async def run() -> None:
        forwarder = LauncherForwarder(str(tmp_path / "none.sock"))
        try:
            # No process can have pid -1 as our parent: the launcher is gone.
            watchdog = AttachedFrontendWatchdog(forwarder, -1, deaths.append, ppid_interval_s=0.01)
            await asyncio.wait_for(watchdog.run(), timeout=5)
        finally:
            await forwarder.close()

    asyncio.run(run())
    assert len(deaths) == 1 and "exited" in str(deaths[0])


@pytest.mark.parametrize("by_signal", [False, True], ids=["should_exit", "sigint"])
def test_attached_frontends_reach_the_launcher_until_they_exit(
    tmp_path, monkeypatch, by_signal
) -> None:
    """Check that the launcher serves its socket while its attached frontends drain.

    Once uvicorn begins its shutdown, the launcher stops its attached frontends. Its /health
    stays truthful until they exit, so their watchdogs still see an engine that fails during
    the drain. The HTTP server finishes a request still in flight, which the attached
    frontends' drain releases. Only then does the launcher shut the engine down. A SIGINT
    reaches the HTTP server first and, once both servers stop, the handler installed before
    serving.
    """
    launcher_path = str(tmp_path / "launcher.sock")
    engine = SimpleNamespace(healthy=True)
    events, deaths, servers, statuses = [], [], [], []
    request_in_flight = threading.Event()
    release_request = asyncio.Event()
    create_uvicorn_server = openai_server.create_uvicorn_server

    def create_server(config):
        servers.append(create_uvicorn_server(config))
        return servers[-1]

    async def stop_attached_frontends(force_exit) -> None:
        events.append("attached frontends stopping")
        forwarder = LauncherForwarder(launcher_path)
        get_status = forwarder.get_status
        launcher_answered = asyncio.Event()
        probe_statuses = []

        async def watchdog_get_status(path: str) -> int | None:
            status = await get_status(path)
            probe_statuses.append(status)
            if status == 200:
                launcher_answered.set()
            return status

        watchdog = AttachedFrontendWatchdog(
            forwarder,
            os.getppid(),
            deaths.append,
            ppid_interval_s=0.01,
            health_interval_s=0.02,
            health_failures=2,
        )
        try:
            # The HTTP server stopped listening and waits for a request in flight.
            await _wait_for(lambda: not any(s.is_serving() for s in servers[0].servers))
            assert request_in_flight.is_set() and not statuses
            assert await forwarder.get_status("/health") == 200
            release_request.set()
            forwarder.get_status = watchdog_get_status
            task = asyncio.create_task(watchdog.run())
            await asyncio.wait_for(launcher_answered.wait(), timeout=5)
            engine.healthy = False
            await asyncio.wait_for(task, timeout=5)
            # The launcher answered both failed probes: its socket was still open.
            assert probe_statuses[-2:] == [503, 503]
        finally:
            await forwarder.close()
        events.append("attached frontends exited")

    async def in_flight_request() -> dict:
        request_in_flight.set()
        await release_request.wait()
        events.append("request in flight answered")
        return {}

    monkeypatch.setattr(openai_server, "create_uvicorn_server", create_server)
    monkeypatch.setenv("TRTLLM_MEDIA_STORAGE_PATH", str(tmp_path / "media"))
    generator = SimpleNamespace(
        _check_health=lambda: engine.healthy,
        shutdown=lambda: events.append("engine shut down"),
    )
    # A VisualGen server needs no model to load.
    with mock.patch.object(openai_server, "_is_visual_gen_instance", return_value=True):
        server = OpenAIServer(
            generator=generator,
            model="dummy",
            tool_parser=None,
            server_role=ServerRole.VISUAL_GEN,
            metadata_server_cfg=None,
            stop_attached_frontends=stop_attached_frontends,
        )
    server.app.add_api_route("/in_flight", in_flight_request, methods=["GET"])
    http_listener = socket.create_server(("127.0.0.1", 0))
    launcher_listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    launcher_listener.bind(launcher_path)
    launcher_listener.listen()
    listeners = [http_listener, launcher_listener]

    def request_over_http() -> None:
        connection = http.client.HTTPConnection(*http_listener.getsockname(), timeout=30)
        try:
            connection.request("GET", "/in_flight")
            statuses.append(connection.getresponse().status)
        finally:
            connection.close()

    client = threading.Thread(target=request_over_http)
    serve = server("localhost", 0, sockets=listeners)
    if by_signal:
        signals = []
        previous_handler = signal.signal(signal.SIGINT, lambda sig, frame: signals.append(sig))

        async def interrupt_once_started() -> None:
            while not (servers and servers[0].started):
                await asyncio.sleep(0.01)
            client.start()
            while not request_in_flight.is_set():
                await asyncio.sleep(0.01)
            signal.raise_signal(signal.SIGINT)

        async def run() -> None:
            interrupting = asyncio.create_task(interrupt_once_started())
            await serve
            await interrupting

        try:
            asyncio.run(asyncio.wait_for(run(), timeout=30))
        finally:
            signal.signal(signal.SIGINT, previous_handler)
            for listener in listeners:
                listener.close()
        assert signals == [signal.SIGINT]
    else:
        # Off the main thread uvicorn installs no signal handlers.
        serving = threading.Thread(target=asyncio.run, args=(serve,), daemon=True)
        serving.start()
        try:
            deadline = time.monotonic() + 30
            while not (servers and servers[0].started):
                assert serving.is_alive() and time.monotonic() < deadline, (
                    "the server did not start"
                )
                time.sleep(0.01)
            client.start()
            assert request_in_flight.wait(30), "the request did not reach the server"
            assert not events
        finally:
            # Stop the server on every path, so that a failed check cannot leave it running.
            if servers:
                servers[0].should_exit = True
            serving.join(30)
            for listener in listeners:
                listener.close()
        assert not serving.is_alive()

    client.join(30)
    assert statuses == [200]
    assert len(deaths) == 1 and "failed 2 consecutive" in str(deaths[0])
    assert events[0] == "attached frontends stopping"
    assert sorted(events[1:3]) == ["attached frontends exited", "request in flight answered"]
    assert events[3:] == ["engine shut down"]


def test_launcher_socket_server_keeps_the_graceful_shutdown_timeout(tmp_path, monkeypatch) -> None:
    """Check that the launcher's Unix socket server bounds its graceful shutdown too.

    A request forwarded by an attached frontend must not keep the launcher waiting without
    bound after the HTTP server stopped.
    """
    configs = []

    class _RecordingServer:
        def __init__(self, config) -> None:
            configs.append(config)
            self.started = True
            self.should_exit = False
            self.force_exit = False

        async def serve(self, sockets=None) -> None:
            self.should_exit = True

    async def stop_attached_frontends(force_exit) -> None:
        pass

    monkeypatch.setattr(openai_server, "create_uvicorn_server", _RecordingServer)
    monkeypatch.setattr(openai_server.uvicorn, "Server", _RecordingServer)
    monkeypatch.setenv("TRTLLM_MEDIA_STORAGE_PATH", str(tmp_path / "media"))
    # A VisualGen server needs no model to load.
    with mock.patch.object(openai_server, "_is_visual_gen_instance", return_value=True):
        server = OpenAIServer(
            generator=SimpleNamespace(),
            model="dummy",
            tool_parser=None,
            server_role=ServerRole.VISUAL_GEN,
            metadata_server_cfg=None,
            graceful_shutdown_timeout=7,
            stop_attached_frontends=stop_attached_frontends,
        )
    sockets = [SimpleNamespace(family=socket.AF_INET), SimpleNamespace(family=socket.AF_UNIX)]
    asyncio.run(server("localhost", 0, sockets=sockets))
    assert [config.timeout_graceful_shutdown for config in configs] == [7, 7]


def test_engine_shuts_down_when_stopping_the_attached_frontends_fails(
    tmp_path, monkeypatch
) -> None:
    """Check that a failed drain is reported and the lifespan still shuts the engine down."""
    events = []

    async def stop_attached_frontends(force_exit) -> None:
        raise RuntimeError("drain failed")

    monkeypatch.setenv("TRTLLM_MEDIA_STORAGE_PATH", str(tmp_path / "media"))
    generator = SimpleNamespace(
        _check_health=lambda: True,
        shutdown=lambda: events.append("engine shut down"),
    )
    # A VisualGen server needs no model to load.
    with mock.patch.object(openai_server, "_is_visual_gen_instance", return_value=True):
        server = OpenAIServer(
            generator=generator,
            model="dummy",
            tool_parser=None,
            server_role=ServerRole.VISUAL_GEN,
            metadata_server_cfg=None,
            stop_attached_frontends=stop_attached_frontends,
        )

    async def run_lifespan() -> None:
        async with server.app.router.lifespan_context(server.app):
            server._attached_frontends_stopped = asyncio.create_task(
                server._stop_attached_frontends(lambda: False)
            )

    # The drain's error is reported: it leaves the lifespan, for uvicorn to log.
    with pytest.raises(RuntimeError, match="drain failed"):
        asyncio.run(run_lifespan())
    assert events == ["engine shut down"]


@pytest.mark.parametrize(
    "graceful_shutdown_timeout_s, killed_at",
    [(30, 140.0), (None, None)],
    ids=["timeout", "no_timeout"],
)
def test_drain_kills_attached_frontends_at_the_graceful_shutdown_deadline(
    monkeypatch, graceful_shutdown_timeout_s, killed_at
) -> None:
    """Check that frontends still running the timeout plus 10 s after SIGTERM are killed.

    Without a timeout, the drain has no deadline: it waits for a frontend that drains for
    300 s, as the frontend's own server waits for its requests. Every frontend is reaped,
    including one that exited on SIGTERM.
    """
    clock = SimpleNamespace(now=100.0)
    slow_exit_s = None if killed_at is not None else 300.0

    async def sleep(seconds: float) -> None:
        clock.now += seconds
        limit = killed_at if killed_at is not None else 100.0 + slow_exit_s
        assert clock.now <= limit + 1.0, "the drain waits past its deadline"

    class _FakeFrontend:
        def __init__(self, exits_after_s: float | None) -> None:
            self.exits_after_s = exits_after_s
            self.terminated_at = self.killed_at = self.returncode = None

        def terminate(self) -> None:
            self.terminated_at = clock.now

        def kill(self) -> None:
            if self.returncode is None:
                self.killed_at = clock.now
                self.returncode = -signal.SIGKILL

        def poll(self) -> int | None:
            if (
                self.returncode is None
                and self.exits_after_s is not None
                and clock.now >= self.terminated_at + self.exits_after_s
            ):
                self.returncode = 0
            return self.returncode

    monkeypatch.setattr(_frontend_processes, "time", SimpleNamespace(monotonic=lambda: clock.now))
    monkeypatch.setattr(_frontend_processes, "asyncio", SimpleNamespace(sleep=sleep))
    exits, slow = _FakeFrontend(exits_after_s=2.0), _FakeFrontend(exits_after_s=slow_exit_s)

    asyncio.run(
        _frontend_processes._drain_attached_frontends(
            [exits, slow], graceful_shutdown_timeout_s, lambda: False
        )
    )

    assert exits.terminated_at == slow.terminated_at == 100.0
    assert (exits.killed_at, exits.returncode) == (None, 0)
    if killed_at is not None:
        assert killed_at <= slow.killed_at < killed_at + 0.2
        assert slow.returncode == -signal.SIGKILL
    else:
        assert (slow.killed_at, slow.returncode) == (None, 0)
        assert clock.now >= 100.0 + slow_exit_s


def test_drain_reaps_killed_attached_frontends_off_the_event_loop() -> None:
    """Check that the event loop keeps running while a killed frontend takes time to exit."""

    class _SlowToDieFrontend:
        """Ignores SIGTERM and exits 0.3 s after SIGKILL; wait() blocks until then."""

        def __init__(self) -> None:
            self.returncode = self.dies_at = None

        def terminate(self) -> None:
            pass

        def kill(self) -> None:
            self.dies_at = time.monotonic() + 0.3

        def poll(self) -> int | None:
            if self.dies_at is not None and time.monotonic() >= self.dies_at:
                self.returncode = -signal.SIGKILL
            return self.returncode

        def wait(self) -> int:
            while self.poll() is None:
                time.sleep(0.01)
            return self.returncode

    child = _SlowToDieFrontend()
    ticks_while_dying = []

    async def tick() -> None:
        while child.returncode is None:
            if child.dies_at is not None:
                ticks_while_dying.append(time.monotonic())
            await asyncio.sleep(0.01)

    async def main() -> None:
        await asyncio.wait_for(
            asyncio.gather(
                _frontend_processes._drain_attached_frontends([child], None, lambda: True), tick()
            ),
            timeout=10,
        )

    asyncio.run(main())
    assert child.returncode == -signal.SIGKILL
    assert ticks_while_dying, "the drain blocked the event loop while the frontend exited"


def test_force_exit_kills_attached_frontends_that_still_drain() -> None:
    """Check that a second Ctrl-C kills attached frontends before the drain deadline."""
    ignores_sigterm = (
        "import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); "
        "print(flush=True); time.sleep(60)"
    )
    child = subprocess.Popen([sys.executable, "-c", ignores_sigterm], stdout=subprocess.PIPE)
    try:
        child.stdout.readline()
        forced_at = time.monotonic() + 0.5

        def force_exit() -> bool:
            return time.monotonic() >= forced_at

        asyncio.run(_frontend_processes._drain_attached_frontends([child], None, force_exit))
        assert time.monotonic() - forced_at < 5
        assert child.returncode == -signal.SIGKILL
    finally:
        child.kill()
        child.wait()
        child.stdout.close()


def test_launcher_logs_an_attached_frontend_that_dies(monkeypatch) -> None:
    errors = []
    monkeypatch.setattr(_frontend_processes.logger, "error", errors.append)
    sleeper = [sys.executable, "-c", "import time; time.sleep(60)"]
    children = [subprocess.Popen(sleeper) for _ in range(2)]
    try:
        stopping = _watch_attached_frontends(children, interval_s=0.01)
        children[0].kill()
        deadline = time.monotonic() + 10
        while not errors and time.monotonic() < deadline:
            time.sleep(0.01)
        # Exits once the launcher terminates the group are expected.
        stopping.set()
        children[1].kill()
        time.sleep(0.1)
    finally:
        for child in children:
            child.kill()
            child.wait()
    assert len(errors) == 1
    assert "frontend 1 " in errors[0] and "signal 9" in errors[0] and "2 of 3" in errors[0]


@pytest.mark.parametrize("doing_shutdown", [False, True])
def test_on_launcher_lost_fails_fast_then_shuts_down(monkeypatch, doing_shutdown: bool) -> None:
    fatal_errors = []
    executor = SimpleNamespace(doing_shutdown=doing_shutdown, _set_fatal_error=fatal_errors.append)
    server = object.__new__(OpenAIServer)
    server.generator = SimpleNamespace(_executor=executor)
    signals, terminations = [], []
    monkeypatch.setattr(openai_server.signal, "raise_signal", signals.append)
    monkeypatch.setattr(openai_server, "_record_generator_termination", terminations.append)

    error = RuntimeError("attached frontend lost its engine: test")
    server._on_launcher_lost(error)

    assert fatal_errors == [error]
    # A shutdown already under way gets no second SIGINT (which would force-quit it).
    assert signals == ([] if doing_shutdown else [signal.SIGINT])
    assert terminations == ([] if doing_shutdown else [server.generator])


def test_shared_clock_offset_publish_and_poll(tmp_path) -> None:
    launcher, follower = SharedClockOffset(str(tmp_path)), SharedClockOffset(str(tmp_path))
    assert follower.poll() is None
    launcher.publish(-12.5)
    assert follower.poll() == -12.5
    assert follower.poll() is None  # unchanged
    launcher.publish(0.125)
    launcher.publish(0.25)
    assert follower.poll() == 0.25  # the latest value wins
    for junk in ("not a number", "nan"):
        (tmp_path / "clock_offset").write_text(junk)
        assert follower.poll() is None
    assert [p.name for p in tmp_path.iterdir()] == ["clock_offset"]


@pytest.mark.parametrize(
    "role,status",
    [("launcher", 200), ("launcher_without_ipc_dir", 500), ("single_frontend", 200)],
)
def test_set_steady_clock_offset(tmp_path, role: str, status: int) -> None:
    shared = {
        "launcher": SharedClockOffset(str(tmp_path)),
        "launcher_without_ipc_dir": SharedClockOffset(str(tmp_path / "gone")),
        "single_frontend": None,
    }[role]
    server = object.__new__(OpenAIServer)
    server._adjusted_steady_clock = AdjustedSteadyClock(time_source=lambda: 100.0)
    server._shared_clock_offset = shared

    response = asyncio.run(server.set_steady_clock_offset(offset=-7.25))

    # A non-200 makes the disaggregated server warn about misaligned timestamps.
    assert response.status_code == status
    assert server._adjusted_steady_clock.now() == 92.75
    if role == "launcher":
        assert SharedClockOffset(str(tmp_path)).poll() == -7.25


@pytest.mark.parametrize("is_launcher", [True, False])
def test_only_the_launcher_registers_the_server(monkeypatch, is_launcher: bool) -> None:
    storage, worker_cls = mock.Mock(), mock.Mock()
    monkeypatch.setattr(openai_server, "create_cluster_storage_client", storage)
    monkeypatch.setattr(openai_server, "DisaggClusterWorker", worker_cls)
    server = object.__new__(OpenAIServer)
    server._registers_server = is_launcher
    server.metadata_server = mock.Mock()
    server.disagg_cluster_config = DisaggClusterConfig("etcd://h:2379", "c", ca_cert="/ca.pem")
    server.disagg_cluster_worker = None
    server.server_role = ServerRole.CONTEXT
    server.model, server.binding_addr = "m", "http://h:8000"
    server.host, server.port = "h", 8000
    server.generator = SimpleNamespace(llm_id="h-1-0")

    server._register_server()

    if is_launcher:
        key, metadata = server.metadata_server.put.call_args.args
        assert key == "trtllm/h-1-0" and metadata["url"] == "http://h:8000"
        storage.assert_called_once_with("etcd://h:2379", "c", ca_cert="/ca.pem")
        assert server.disagg_cluster_worker is worker_cls.return_value
    else:
        server.metadata_server.put.assert_not_called()
        storage.assert_not_called()
        worker_cls.assert_not_called()
