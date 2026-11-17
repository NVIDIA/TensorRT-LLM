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
import logging
import os
import select
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Awaitable, Callable, Iterator
from contextlib import asynccontextmanager, contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import httpx
import msgspec
import pytest
import uvicorn
from aiohttp import web
from fastapi import FastAPI, Request, Response
from fastapi.testclient import TestClient
from starlette.routing import Match
from test_common.serve.app_factory import AggConfig, isolated_serve_env

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
        assert "x-should-retry" not in r.headers


def test_forwarding_timeout_warns_that_a_delivered_post_can_still_execute(tmp_path: Path) -> None:
    payload = {"requests": []}
    seen = []

    async def run() -> None:
        launcher = FastAPI()
        created = asyncio.Event()
        release = asyncio.Event()
        completed = asyncio.Event()

        @launcher.post("/v1/messages/batches")
        async def create_batch(request: Request) -> dict:
            seen.append(await request.json())
            created.set()
            await release.wait()
            completed.set()
            return {"id": "batch-1"}

        path = str(tmp_path / "launcher.sock")
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as listener:
            listener.bind(path)
            listener.listen()
            forwarder = LauncherForwarder(path, timeout_s=1.0)
            frontend = FastAPI()
            frontend.add_api_route("/v1/messages/batches", forwarder.forward, methods=["POST"])
            server = uvicorn.Server(
                uvicorn.Config(
                    launcher,
                    http="h11",
                    loop="asyncio",
                    lifespan="off",
                    log_config=None,
                    log_level=None,
                )
            )
            serving = asyncio.create_task(server.serve(sockets=[listener]))
            try:
                await _wait_for(lambda: server.started)
                async with httpx.AsyncClient(
                    transport=httpx.ASGITransport(app=frontend), base_url="http://frontend"
                ) as client:
                    forwarding = asyncio.create_task(
                        client.post("/v1/messages/batches", json=payload)
                    )
                    await asyncio.wait_for(created.wait(), timeout=5)
                    response = await forwarding
                assert not completed.is_set()
                release.set()
                await asyncio.wait_for(completed.wait(), timeout=5)
            finally:
                release.set()
                server.should_exit = True
                await asyncio.wait_for(serving, timeout=5)
                await forwarder.close()

        assert seen == [payload]
        assert response.status_code == 504
        assert response.headers["x-should-retry"] == "false"
        assert "may still be processed" in response.json()["error"]

    asyncio.run(run())


def _capture_http_servers(monkeypatch: pytest.MonkeyPatch) -> list[uvicorn.Server]:
    servers = []
    create_server = openai_server.create_uvicorn_server

    def capture(config: uvicorn.Config) -> uvicorn.Server:
        server = create_server(config)
        servers.append(server)
        return server

    monkeypatch.setattr(openai_server, "create_uvicorn_server", capture)
    return servers


@contextmanager
def _isolated_uvicorn_logging(tmp_path: Path) -> Iterator[None]:
    def logger_state(logger: logging.Logger) -> tuple:
        return (
            logger.level,
            tuple(logger.handlers),
            tuple(logger.filters),
            logger.propagate,
            logger.disabled,
        )

    loggers = [logging.getLogger()]
    loggers.extend(
        logger
        for logger in logging.root.manager.loggerDict.values()
        if isinstance(logger, logging.Logger)
    )
    before = {logger: logger_state(logger) for logger in loggers}
    unrelated_log = tmp_path / "unrelated.log"
    unrelated_handler = logging.FileHandler(unrelated_log)

    def config_without_logging(app: FastAPI, **kwargs) -> uvicorn.Config:
        kwargs["log_config"] = None
        kwargs["log_level"] = None
        return uvicorn.Config(app, **kwargs)

    try:
        with mock.patch.object(
            openai_server, "uvicorn", SimpleNamespace(Config=config_without_logging)
        ):
            yield
    finally:
        try:
            assert {logger: logger_state(logger) for logger in loggers} == before
            assert unrelated_handler.stream is not None
            assert not unrelated_handler.stream.closed
            unrelated_handler.emit(logging.makeLogRecord({"msg": "logging remains usable"}))
            unrelated_handler.flush()
            assert unrelated_log.read_text().strip() == "logging remains usable"
        finally:
            unrelated_handler.close()


def test_attached_frontend_waits_for_launcher_http_before_accepting_requests(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    servers = _capture_http_servers(monkeypatch)

    async def run(tmp_path: Path) -> None:
        with isolated_serve_env(tmp_path) as env:
            attached = env.build_openai_server(AggConfig(multi_frontend_attached=True))
            launcher = FastAPI()
            health_seen = asyncio.Event()
            ready = asyncio.Event()
            seen = []

            @launcher.get("/health")
            async def health() -> dict:
                health_seen.set()
                await ready.wait()
                return {}

            @launcher.post("/steady_clock_offset")
            @launcher.post("/v1/messages/batches")
            async def record(request: Request) -> dict:
                seen.append((request.url.path, await request.json()))
                return {"id": "batch-1"}

            with (
                socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as listener,
                socket.socket() as frontend_socket,
            ):
                listener.bind(str(tmp_path / "multi_frontend_ipc" / "launcher.sock"))
                listener.listen()
                frontend_socket.bind(("127.0.0.1", 0))
                address = f"http://127.0.0.1:{frontend_socket.getsockname()[1]}"
                launcher_server = uvicorn.Server(
                    uvicorn.Config(
                        launcher,
                        http="h11",
                        loop="asyncio",
                        lifespan="off",
                        log_config=None,
                        log_level=None,
                    )
                )
                launcher_task = asyncio.create_task(launcher_server.serve(sockets=[listener]))
                frontend_task = asyncio.create_task(
                    attached(
                        "127.0.0.1", frontend_socket.getsockname()[1], sockets=[frontend_socket]
                    )
                )
                try:
                    await _wait_for(lambda: bool(servers))
                    frontend_server = servers[0]
                    await _wait_for(lambda: health_seen.is_set() or frontend_server.started)
                    assert not frontend_server.started
                    async with httpx.AsyncClient(base_url=address, timeout=1.0) as client:
                        with pytest.raises(httpx.ConnectError):
                            await client.post("/v1/messages/batches", json={"requests": []})
                        assert not seen
                        ready.set()
                        await _wait_for(lambda: frontend_server.started)
                        assert (await client.get("/health")).status_code == 200
                        assert (
                            await client.post("/steady_clock_offset", json=1.25)
                        ).status_code == 200
                        assert (
                            await client.post("/v1/messages/batches", json={"requests": []})
                        ).status_code == 200
                    assert seen == [
                        ("/steady_clock_offset", 1.25),
                        ("/v1/messages/batches", {"requests": []}),
                    ]
                finally:
                    ready.set()
                    if servers:
                        servers[0].should_exit = True
                    await asyncio.wait_for(frontend_task, timeout=10)
                    launcher_server.should_exit = True
                    await asyncio.wait_for(launcher_task, timeout=10)

    with (
        tempfile.TemporaryDirectory(dir="/tmp", prefix="tllm-mf-") as directory,
        _isolated_uvicorn_logging(Path(directory)),
    ):
        asyncio.run(run(Path(directory)))


def test_attached_frontend_startup_is_bounded_when_launcher_does_not_answer(
    capfd: pytest.CaptureFixture[str],
) -> None:
    async def run(tmp_path: Path) -> None:
        with isolated_serve_env(tmp_path) as env:
            attached = env.build_openai_server(
                AggConfig(
                    multi_frontend_attached=True, env=(("TLLM_FRONTEND_READY_TIMEOUT", "0.1"),)
                )
            )
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as listener:
                listener.bind(str(tmp_path / "multi_frontend_ipc" / "launcher.sock"))
                listener.listen()
                with pytest.raises((RuntimeError, SystemExit)) as failure:
                    await asyncio.wait_for(attached("127.0.0.1", 0), timeout=5)
                if isinstance(failure.value, SystemExit):
                    assert failure.value.code not in (None, 0)
                captured = capfd.readouterr()
                assert "TLLM_FRONTEND_READY_TIMEOUT" in (
                    str(failure.value) + captured.out + captured.err
                )

    with (
        tempfile.TemporaryDirectory(dir="/tmp", prefix="tllm-mf-") as directory,
        _isolated_uvicorn_logging(Path(directory)),
    ):
        asyncio.run(run(Path(directory)))


def test_attached_frontend_waiting_for_launcher_exits_on_sigterm() -> None:
    child_code = """
import asyncio
import os
import sys
from pathlib import Path
from test_common.serve.app_factory import AggConfig, isolated_serve_env

async def run():
    with isolated_serve_env(Path(sys.argv[1])) as env:
        attached = env.build_openai_server(AggConfig(
            multi_frontend_attached=True,
            env=(("TLLM_FRONTEND_READY_TIMEOUT", "300"),),
        ))
        serving = asyncio.create_task(attached("127.0.0.1", 0))
        await asyncio.sleep(0)
        os.write(int(sys.argv[2]), b"R")
        os.close(int(sys.argv[2]))
        await serving

asyncio.run(run())
"""
    with tempfile.TemporaryDirectory(dir="/tmp", prefix="tllm-mf-") as directory:
        ipc_dir = Path(directory) / "multi_frontend_ipc"
        ipc_dir.mkdir()
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as listener:
            listener.bind(str(ipc_dir / "launcher.sock"))
            listener.listen()
            read_fd, write_fd = os.pipe()
            try:
                child = subprocess.Popen(
                    [sys.executable, "-c", child_code, directory, str(write_fd)],
                    pass_fds=(write_fd,),
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                    text=True,
                )
            finally:
                os.close(write_fd)
            try:
                with os.fdopen(read_fd, "rb", buffering=0) as ready:
                    assert select.select([ready], [], [], 120)[0], "child did not reach startup"
                    assert ready.read(1) == b"R", child.communicate(timeout=5)[1]
                assert child.poll() is None
                listener.close()
                _frontend_processes._terminate_attached_frontends([child])
                stderr = child.communicate(timeout=5)[1]
                assert child.returncode == -signal.SIGTERM, stderr
            finally:
                if child.poll() is None:
                    child.kill()
                child.communicate(timeout=5)


async def _require_test_authorization(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    if request.headers.get("x-test-auth") != "allowed":
        if request.url.path == "/health":
            request.app.state.denied_health += 1
        return Response(status_code=401)
    return await call_next(request)


def test_attached_frontend_starts_with_authenticated_launcher_health(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tensorrt_llm.commands.serve import _apply_fastapi_middlewares

    servers = _capture_http_servers(monkeypatch)
    raised_signals = []
    monkeypatch.setattr(signal, "raise_signal", raised_signals.append)

    async def run(tmp_path: Path) -> None:
        with isolated_serve_env(tmp_path) as env:
            attached = env.build_openai_server(
                AggConfig(multi_frontend_attached=True, env=(("TLLM_FRONTEND_READY_TIMEOUT", "1"),))
            )
            launcher = FastAPI()
            seen = []

            @launcher.get("/health")
            async def health() -> dict:
                return {}

            @launcher.post("/v1/messages/batches")
            async def create_batch(request: Request) -> dict:
                seen.append(await request.json())
                return {"id": "batch-1"}

            for app in (launcher, attached.app):
                app.state.denied_health = 0
                _apply_fastapi_middlewares(app, [f"{__name__}._require_test_authorization"])

            with (
                socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as listener,
                socket.socket() as frontend_socket,
            ):
                listener.bind(str(tmp_path / "multi_frontend_ipc" / "launcher.sock"))
                listener.listen()
                frontend_socket.bind(("127.0.0.1", 0))
                port = frontend_socket.getsockname()[1]
                launcher_server = uvicorn.Server(
                    uvicorn.Config(
                        launcher, http="h11", loop="asyncio", log_config=None, log_level=None
                    )
                )
                launcher_task = asyncio.create_task(launcher_server.serve(sockets=[listener]))
                frontend_task = asyncio.create_task(
                    attached("127.0.0.1", port, sockets=[frontend_socket])
                )
                try:
                    await _wait_for(lambda: bool(servers))
                    frontend_server = servers[0]
                    await _wait_for(lambda: frontend_server.started or frontend_task.done())
                    assert not frontend_task.done(), frontend_task.exception()
                    async with httpx.AsyncClient(base_url=f"http://127.0.0.1:{port}") as client:
                        assert (await client.get("/health")).status_code == 401
                        headers = {"x-test-auth": "allowed"}
                        assert (await client.get("/health", headers=headers)).status_code == 200
                        await _wait_for(lambda: launcher.state.denied_health >= 4, timeout=30)
                        assert not raised_signals
                        assert attached._launcher_watchdog_task is not None
                        assert not attached._launcher_watchdog_task.done()
                        response = await client.post(
                            "/v1/messages/batches", headers=headers, json={"requests": []}
                        )
                        assert response.status_code == 200
                        assert response.json() == {"id": "batch-1"}
                    assert seen == [{"requests": []}]
                finally:
                    if servers:
                        servers[0].should_exit = True
                    await asyncio.wait_for(
                        asyncio.gather(frontend_task, return_exceptions=True), timeout=10
                    )
                    launcher_server.should_exit = True
                    await asyncio.wait_for(launcher_task, timeout=10)

    with (
        tempfile.TemporaryDirectory(dir="/tmp", prefix="tllm-mf-") as directory,
        _isolated_uvicorn_logging(Path(directory)),
    ):
        asyncio.run(run(Path(directory)))


async def _delay_test_health(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    if request.url.path == "/health":
        await asyncio.sleep(1.25)
    return await call_next(request)


def test_attached_frontend_starts_with_slow_launcher_health(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tensorrt_llm.commands.serve import _apply_fastapi_middlewares

    servers = _capture_http_servers(monkeypatch)

    async def run(tmp_path: Path) -> None:
        with isolated_serve_env(tmp_path) as env:
            attached = env.build_openai_server(
                AggConfig(multi_frontend_attached=True, env=(("TLLM_FRONTEND_READY_TIMEOUT", "5"),))
            )
            launcher = FastAPI()
            health_responses = []

            @launcher.get("/health")
            async def health() -> dict:
                health_responses.append(200)
                return {}

            _apply_fastapi_middlewares(launcher, [f"{__name__}._delay_test_health"])
            with (
                socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as listener,
                socket.socket() as frontend_socket,
            ):
                path = str(tmp_path / "multi_frontend_ipc" / "launcher.sock")
                listener.bind(path)
                listener.listen()
                frontend_socket.bind(("127.0.0.1", 0))
                port = frontend_socket.getsockname()[1]
                launcher_server = uvicorn.Server(
                    uvicorn.Config(
                        launcher, http="h11", loop="asyncio", log_config=None, log_level=None
                    )
                )
                launcher_task = asyncio.create_task(launcher_server.serve(sockets=[listener]))
                frontend_task = None
                forwarder = LauncherForwarder(path)
                try:
                    await _wait_for(lambda: launcher_server.started)
                    assert await forwarder.get_status("/health") == 200
                    frontend_task = asyncio.create_task(
                        attached("127.0.0.1", port, sockets=[frontend_socket])
                    )
                    await _wait_for(lambda: bool(servers))
                    frontend_server = servers[0]
                    await _wait_for(lambda: frontend_server.started or frontend_task.done())
                    assert not frontend_task.done(), frontend_task.exception()
                    async with httpx.AsyncClient(base_url=f"http://127.0.0.1:{port}") as client:
                        assert (await client.get("/health")).status_code == 200
                    assert len(health_responses) >= 2
                finally:
                    if servers:
                        servers[0].should_exit = True
                    if frontend_task is not None:
                        await asyncio.wait_for(
                            asyncio.gather(frontend_task, return_exceptions=True), timeout=10
                        )
                    launcher_server.should_exit = True
                    await asyncio.wait_for(launcher_task, timeout=10)
                    await forwarder.close()

    with (
        tempfile.TemporaryDirectory(dir="/tmp", prefix="tllm-mf-") as directory,
        _isolated_uvicorn_logging(Path(directory)),
    ):
        asyncio.run(run(Path(directory)))


def test_readiness_wait_notices_a_gone_launcher_process(tmp_path: Path) -> None:
    async def run() -> None:
        forwarder = LauncherForwarder(str(tmp_path / "none.sock"))
        watchdog = AttachedFrontendWatchdog(forwarder, -1, lambda error: None)
        try:
            with pytest.raises(RuntimeError, match="exited during startup"):
                await asyncio.wait_for(watchdog.wait_until_ready(300), timeout=5)
        finally:
            await forwarder.close()

    asyncio.run(run())


def test_watchdog_remembers_the_successful_startup_health_probe(tmp_path: Path) -> None:
    launcher = _FakeLauncher(str(tmp_path / "launcher.sock"))
    deaths = []

    async def run() -> None:
        forwarder = LauncherForwarder(launcher.path)
        watchdog = AttachedFrontendWatchdog(
            forwarder,
            os.getppid(),
            deaths.append,
            ppid_interval_s=0.01,
            health_interval_s=0.02,
            health_failures=2,
        )
        try:
            await watchdog.wait_until_ready(5)
            launcher.health_status = 503
            await asyncio.wait_for(watchdog.run(), timeout=5)
        finally:
            await forwarder.close()

    try:
        asyncio.run(run())
    finally:
        launcher.close()
    assert len(deaths) == 1 and "failed 2 consecutive" in str(deaths[0])


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
            # Startup probes can fail before the launcher answers; failures
            # only count after its first successful response.
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
