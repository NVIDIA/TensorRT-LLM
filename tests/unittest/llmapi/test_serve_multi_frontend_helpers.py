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
"""CPU coverage for trtllm-serve's multi-frontend helpers.

An attached frontend forwards launcher-owned routes to the launcher over a
Unix socket and watches the launcher for engine or process death. Both are
exercised here against a fake launcher, together with the OpenAIServer hooks
that wire them in; spawning real frontends needs a GPU engine.
"""

import asyncio
import os
import signal
import threading
import time
from contextlib import asynccontextmanager
from types import SimpleNamespace

import msgspec
import pytest
from aiohttp import web
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.routing import Match

from tensorrt_llm.serve import openai_server
from tensorrt_llm.serve.multi_frontend import (
    FORWARDED_ROUTES,
    AttachedFrontendWatchdog,
    LauncherForwarder,
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
                {"method": request.method, "path": request.path_qs},
                status=201 if request.method == "POST" else 200,
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


def _attached_app(forwarder: LauncherForwarder) -> FastAPI:
    """The forwarded routes as OpenAIServer registers them, plus a local one."""

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        yield
        await forwarder.close()

    app = FastAPI(lifespan=lifespan)
    for method, path in FORWARDED_ROUTES:
        app.add_api_route(path, forwarder.forward, methods=[method])

    @app.get("/health")
    async def local_health():
        return {"local": True}

    return app


def test_forwarder_replays_launcher_owned_routes(tmp_path) -> None:
    launcher = _FakeLauncher(str(tmp_path / "launcher.sock"))
    try:
        with TestClient(_attached_app(LauncherForwarder(launcher.path))) as client:
            r = client.get("/metrics?limit=3")
            assert r.status_code == 200
            assert r.json() == {"method": "GET", "path": "/metrics?limit=3"}

            r = client.post("/v1/messages/batches/b1/cancel", json={"reason": "x"})
            assert r.status_code == 201
            assert r.json()["path"] == "/v1/messages/batches/b1/cancel"
            method, path, body, _ = launcher.seen[-1]
            assert (method, path) == ("POST", "/v1/messages/batches/b1/cancel")
            assert b'"reason"' in body

            # Routes outside FORWARDED_ROUTES stay local.
            assert client.get("/health").json() == {"local": True}
    finally:
        launcher.close()


def test_forwarder_keeps_end_to_end_headers(tmp_path) -> None:
    # The body is replayed byte for byte, so the header that says how to
    # decode it (x-trtllm-msgpack) must travel with it, and so must auth
    # headers a --middleware on the launcher checks again.
    launcher = _FakeLauncher(str(tmp_path / "launcher.sock"))
    body = msgspec.msgpack.encode({"requests": []})
    try:
        with TestClient(_attached_app(LauncherForwarder(launcher.path))) as client:
            r = client.post(
                "/v1/messages/batches",
                content=body,
                headers={
                    "content-type": "application/msgpack",
                    "x-trtllm-msgpack": "1",
                    "x-api-key": "secret",
                },
            )
            assert r.status_code == 201
    finally:
        launcher.close()
    _, path, seen_body, headers = launcher.seen[-1]
    headers = {k.lower(): v for k, v in headers.items()}
    assert path == "/v1/messages/batches"
    assert seen_body == body
    assert headers["x-trtllm-msgpack"] == "1"
    assert headers["x-api-key"] == "secret"
    assert headers["content-type"] == "application/msgpack"
    # Hop-by-hop and framing headers are the forwarder's own, not the client's.
    assert headers["host"] != "testserver"
    assert headers["content-length"] == str(len(body))


def test_forwarder_reports_unreachable_launcher(tmp_path) -> None:
    with TestClient(_attached_app(LauncherForwarder(str(tmp_path / "missing.sock")))) as client:
        r = client.get("/metrics")
        assert r.status_code == 503
        assert "unavailable" in r.json()["error"]


async def _wait_for(predicate, timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        assert time.monotonic() < deadline, "condition not reached in time"
        await asyncio.sleep(0.01)


def test_watchdog_marks_dead_after_consecutive_health_failures(tmp_path) -> None:
    launcher = _FakeLauncher(str(tmp_path / "launcher.sock"))
    deaths = []

    async def run() -> None:
        forwarder = LauncherForwarder(launcher.path)
        try:
            watchdog = AttachedFrontendWatchdog(
                forwarder,
                os.getppid(),
                deaths.append,
                ppid_interval_s=0.01,
                health_interval_s=0.02,
                health_failures=2,
            )
            task = asyncio.create_task(watchdog.run())
            await _wait_for(lambda: launcher.health_hits > 0)
            assert not deaths
            launcher.health_status = 503
            await asyncio.wait_for(task, timeout=10)
        finally:
            await forwarder.close()

    try:
        asyncio.run(run())
    finally:
        launcher.close()
    assert len(deaths) == 1
    assert "failed 2 consecutive" in str(deaths[0])


def test_watchdog_waits_for_a_starting_launcher(tmp_path) -> None:
    # The launcher starts listening only after its attached frontends are
    # ready; failures before it first answered must not kill the frontend.
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
            # Nothing listens yet: ~25 probes, any 2 of which used to be fatal.
            await asyncio.sleep(0.5)
            assert not deaths and not task.done()
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
    assert len(deaths) == 1
    assert "failed 2 consecutive" in str(deaths[0])


def test_watchdog_marks_dead_when_launcher_process_is_gone(tmp_path) -> None:
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
    assert len(deaths) == 1
    assert "exited" in str(deaths[0])


def _attached_server(forwarder: LauncherForwarder) -> OpenAIServer:
    """Just enough of an attached-frontend OpenAIServer to register routes."""
    server = object.__new__(OpenAIServer)
    server.app = FastAPI()
    server.generator = SimpleNamespace(
        _executor=SimpleNamespace(resource_governor_queue=None),
        args=SimpleNamespace(return_perf_metrics=False),
    )
    server.use_harmony = False
    server._enable_rl_control_endpoints = False
    server._enable_runtime_control_endpoints = False
    server.resource_governor = None
    server._launcher_forwarder = forwarder
    return server


def test_attached_server_forwards_before_local_handlers(tmp_path) -> None:
    forwarder = LauncherForwarder(str(tmp_path / "launcher.sock"))
    server = _attached_server(forwarder)
    # The order OpenAIServer.__init__ registers them in.
    server._register_launcher_forwarding_routes()
    server.register_routes()
    for method, template in FORWARDED_ROUTES:
        path = template.replace("{rest:path}", "b1/cancel")
        scope = {"type": "http", "method": method, "path": path, "path_params": {}}
        first = next(r for r in server.app.routes if r.matches(scope)[0] == Match.FULL)
        assert first.endpoint == forwarder.forward, f"{method} {path} is answered locally"
    # Routes that are not launcher-owned keep their local handler.
    scope = {"type": "http", "method": "GET", "path": "/health", "path_params": {}}
    first = next(r for r in server.app.routes if r.matches(scope)[0] == Match.FULL)
    assert first.endpoint == server.health


def test_single_frontend_server_registers_no_forwarding() -> None:
    server = _attached_server(None)
    before = list(server.app.routes)
    server._register_launcher_forwarding_routes()
    assert server.app.routes == before


class _FakeExecutor:
    def __init__(self, doing_shutdown: bool) -> None:
        self.doing_shutdown = doing_shutdown
        self._fatal_error = None
        self.fatal_errors = []

    def _set_fatal_error(self, error: BaseException) -> None:
        self.fatal_errors.append(error)
        self._fatal_error = error


def _server_losing_launcher(monkeypatch, executor: _FakeExecutor):
    server = object.__new__(OpenAIServer)
    server.generator = SimpleNamespace(_executor=executor)
    signals, terminations = [], []
    monkeypatch.setattr(openai_server.signal, "raise_signal", signals.append)
    monkeypatch.setattr(openai_server, "_record_generator_termination", terminations.append)
    return server, signals, terminations


@pytest.mark.parametrize("doing_shutdown", [False, True])
def test_on_launcher_lost_fails_fast_then_shuts_down(monkeypatch, doing_shutdown: bool) -> None:
    executor = _FakeExecutor(doing_shutdown)
    server, signals, terminations = _server_losing_launcher(monkeypatch, executor)

    error = RuntimeError("attached frontend lost its engine: test")
    server._on_launcher_lost(error)

    # In-flight and new requests fail with this error instead of hanging.
    assert executor.fatal_errors == [error]
    if doing_shutdown:
        # Shutdown is already under way: no second SIGINT to force-quit it.
        assert signals == [] and terminations == []
    else:
        assert signals == [signal.SIGINT]
        assert terminations == [server.generator]


def test_on_launcher_lost_keeps_an_earlier_fatal_error(monkeypatch) -> None:
    executor = _FakeExecutor(doing_shutdown=False)
    first = RuntimeError("engine error recorded first")
    executor._fatal_error = first
    server, signals, _ = _server_losing_launcher(monkeypatch, executor)

    server._on_launcher_lost(RuntimeError("attached frontend lost its engine: test"))

    assert executor.fatal_errors == [] and executor._fatal_error is first
    assert signals == [signal.SIGINT]
