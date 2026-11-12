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
import os
import signal
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
from tensorrt_llm.llmapi.disagg_utils import ServerRole
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
    server.disagg_cluster_config = SimpleNamespace(cluster_uri="etcd://h:2379", cluster_name="c")
    server.disagg_cluster_worker = None
    server.server_role = ServerRole.CONTEXT
    server.model, server.binding_addr = "m", "http://h:8000"
    server.host, server.port = "h", 8000
    server.generator = SimpleNamespace(llm_id="h-1-0")

    server._register_server()

    if is_launcher:
        key, metadata = server.metadata_server.put.call_args.args
        assert key == "trtllm/h-1-0" and metadata["url"] == "http://h:8000"
        assert server.disagg_cluster_worker is worker_cls.return_value
    else:
        server.metadata_server.put.assert_not_called()
        storage.assert_not_called()
        worker_cls.assert_not_called()
