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
exercised here against a fake launcher; the GPU path is covered end to end by
the serving integration tests.
"""

import asyncio
import os
import threading
from contextlib import asynccontextmanager

import pytest
from aiohttp import web
from fastapi import FastAPI
from fastapi.testclient import TestClient

from tensorrt_llm.serve.multi_frontend import (
    FORWARDED_ROUTES,
    AttachedFrontendWatchdog,
    LauncherForwarder,
)

pytestmark = pytest.mark.cpu_only


class _FakeLauncher:
    """aiohttp app on a Unix socket standing in for the launcher's uvicorn."""

    def __init__(self, path: str, health_status: int = 200) -> None:
        self.path = path
        self.health_status = health_status
        self.seen = []
        self._loop = asyncio.new_event_loop()
        self._ready = threading.Event()
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()
        assert self._ready.wait(10), "fake launcher did not start"

    def _serve(self) -> None:
        asyncio.set_event_loop(self._loop)

        async def echo(request: web.Request) -> web.Response:
            self.seen.append((request.method, request.path_qs, await request.text()))
            return web.json_response(
                {"method": request.method, "path": request.path_qs},
                status=201 if request.method == "POST" else 200,
            )

        async def health(_: web.Request) -> web.Response:
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
            method, path, body = launcher.seen[-1]
            assert (method, path) == ("POST", "/v1/messages/batches/b1/cancel")
            assert '"reason"' in body

            # Routes outside FORWARDED_ROUTES stay local.
            assert client.get("/health").json() == {"local": True}
    finally:
        launcher.close()


def test_forwarder_reports_unreachable_launcher(tmp_path) -> None:
    with TestClient(_attached_app(LauncherForwarder(str(tmp_path / "missing.sock")))) as client:
        r = client.get("/metrics")
        assert r.status_code == 503
        assert "unavailable" in r.json()["error"]


def test_watchdog_marks_dead_after_consecutive_health_failures(tmp_path) -> None:
    launcher = _FakeLauncher(str(tmp_path / "launcher.sock"), health_status=503)
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
            await asyncio.wait_for(watchdog.run(), timeout=10)
        finally:
            await forwarder.close()

    try:
        asyncio.run(run())
    finally:
        launcher.close()
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
