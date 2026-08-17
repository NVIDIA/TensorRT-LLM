# Copyright (c) 2026, NVIDIA CORPORATION.
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
"""Startup readiness of the trtllm-serve HTTP frontend.

The property under test is that a serving process is remotely
distinguishable while it initializes: the socket accepts and every endpoint
answers 503, instead of the connect being refused exactly as it would be for
a process that crashed or was never started.

CPU-only: the generator is a stub, so nothing here needs a GPU or weights.
"""

import asyncio
import socket
import threading
import time
from types import SimpleNamespace
from typing import Optional

import pytest
import requests

from tensorrt_llm.serve.openai_server import OpenAIServer
from tensorrt_llm.serve.readiness import (
    SERVER_STATE_HEADER,
    ReadinessGate,
    ReadinessMiddleware,
    ServerState,
    run_in_daemon_thread,
)

# Long enough that a poll lands mid-build on a loaded CI machine, short enough
# not to dominate the suite.
BUILD_SECONDS = 1.5


# ---------------------------------------------------------------------------
# ReadinessGate / ReadinessMiddleware
# ---------------------------------------------------------------------------
async def _call_asgi(app, scope, incoming=()):
    """Drive an ASGI app and collect what it sends."""
    received = list(incoming)
    sent = []

    async def receive():
        return received.pop(0)

    async def send(message):
        sent.append(message)

    await app(scope, receive, send)
    return sent


async def _unreachable_app(scope, receive, send):
    raise AssertionError("the gate let a request through while STARTING")


def test_gate_starts_closed():
    gate = ReadinessGate()
    assert gate.state is ServerState.STARTING
    assert not gate.is_ready

    gate.open()
    assert gate.state is ServerState.READY
    assert gate.is_ready


@pytest.mark.parametrize("path", ["/health", "/v1/completions", "/nope"])
def test_starting_answers_every_path_with_503(path):
    """A 404 would read as a real answer; routes are registered late."""
    gate = ReadinessGate()
    app = ReadinessMiddleware(_unreachable_app, gate=gate)

    sent = asyncio.run(_call_asgi(app, {"type": "http", "path": path}))

    start, body = sent
    assert start["status"] == 503
    headers = {k.decode(): v.decode() for k, v in start["headers"]}
    assert headers[SERVER_STATE_HEADER] == ServerState.STARTING.value
    # Without this clients hot-loop through a multi-minute startup.
    assert headers["retry-after"] == "1"
    assert b"still initializing" in body["body"]


def test_ready_passes_requests_through():
    gate = ReadinessGate()
    seen = []

    async def app(scope, receive, send):
        seen.append(scope["path"])

    middleware = ReadinessMiddleware(app, gate=gate)
    gate.open()

    asyncio.run(_call_asgi(middleware, {"type": "http", "path": "/health"}))
    assert seen == ["/health"]


def test_lifespan_passes_through_while_starting():
    """The wrapped app's lifespan is what drives initialization."""
    gate = ReadinessGate()
    seen = []

    async def app(scope, receive, send):
        seen.append(scope["type"])

    middleware = ReadinessMiddleware(app, gate=gate)

    asyncio.run(_call_asgi(middleware, {"type": "lifespan"}))
    assert seen == ["lifespan"]


def test_starting_closes_websocket_without_an_http_response():
    """An HTTP response on a websocket scope raises inside the protocol."""
    gate = ReadinessGate()
    app = ReadinessMiddleware(_unreachable_app, gate=gate)

    sent = asyncio.run(_call_asgi(app, {"type": "websocket"}, [{"type": "websocket.connect"}]))

    assert sent == [{"type": "websocket.close", "code": 1013}]


def test_starting_ignores_an_aborted_websocket_handshake():
    """Closing after websocket.disconnect raises inside the protocol."""
    gate = ReadinessGate()
    app = ReadinessMiddleware(_unreachable_app, gate=gate)

    sent = asyncio.run(_call_asgi(app, {"type": "websocket"}, [{"type": "websocket.disconnect"}]))

    assert sent == []


# ---------------------------------------------------------------------------
# run_in_daemon_thread
# ---------------------------------------------------------------------------
def test_daemon_thread_returns_result_and_does_not_block_the_loop():
    order = []

    def build():
        time.sleep(0.2)
        order.append("build done")
        return "engine"

    async def main():
        future = run_in_daemon_thread(build)
        # The loop stays free while the build runs; this is what lets uvicorn
        # answer /health during a multi-minute initialization.
        while not future.done():
            order.append("loop ran")
            await asyncio.sleep(0.01)
        return await future

    assert asyncio.run(main()) == "engine"
    assert "loop ran" in order
    assert "build done" in order


def test_daemon_thread_propagates_the_exception():
    async def main():
        with pytest.raises(RuntimeError, match="no weights"):
            await run_in_daemon_thread(lambda: (_ for _ in ()).throw(RuntimeError("no weights")))

    asyncio.run(main())


def test_build_thread_is_a_daemon():
    """A non-daemon thread would hold the interpreter open past SIGTERM."""
    seen = {}
    started = threading.Event()
    release = threading.Event()

    def build():
        seen["daemon"] = threading.current_thread().daemon
        started.set()
        release.wait(timeout=30)

    async def main():
        future = run_in_daemon_thread(build)
        while not started.is_set():
            await asyncio.sleep(0.01)
        release.set()
        await future

    asyncio.run(main())
    assert seen["daemon"] is True


# ---------------------------------------------------------------------------
# OpenAIServer with a deferred generator
# ---------------------------------------------------------------------------
class FakeGenerator:
    """Stands in for an LLM: only what __init__ and teardown touch."""

    def __init__(self, hf_model_dir: str):
        self.args = SimpleNamespace(
            trust_remote_code=False,
            checkpoint_format=None,
            num_postprocess_workers=0,
            return_perf_metrics=False,
            perf_metrics_output_dir=None,
            post_processor_hook=None,
            backend="pytorch",
            max_beam_width=1,
            enable_energy_metrics=False,
            enable_iter_perf_stats=False,
        )
        self.tokenizer = None
        self._hf_model_dir = hf_model_dir
        self._executor = SimpleNamespace(resource_governor_queue=None)
        self.llm_id = "fake-llm"
        self.shutdown_calls = 0

    def shutdown(self):
        self.shutdown_calls += 1

    def _check_health(self) -> bool:
        return True


@pytest.fixture
def model_dir(tmp_path):
    """An empty directory: AutoProcessor/AutoConfig fail on it locally.

    Both loads are best-effort in _init_llm, so this keeps the stub generator
    from reaching the network.
    """
    return str(tmp_path)


class ServerRunner:
    """Run OpenAIServer.__call__ on its own loop, in a thread."""

    def __init__(self, server: OpenAIServer):
        self.server = server
        self.error: Optional[BaseException] = None
        self.returned = threading.Event()
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._socket.bind(("127.0.0.1", 0))
        self.port = self._socket.getsockname()[1]
        self._thread = threading.Thread(target=self._run, name="test_serve_runner", daemon=True)

    def _run(self):
        try:
            asyncio.run(self.server("127.0.0.1", self.port, sockets=[self._socket]))
        except BaseException as e:  # noqa: BLE001 - surfaced to the test
            self.error = e
        finally:
            self.returned.set()

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self.server._request_shutdown()
        self.returned.wait(timeout=60)
        self._thread.join(timeout=60)
        self._socket.close()

    def url(self, path: str) -> str:
        return f"http://127.0.0.1:{self.port}{path}"

    def get(self, path: str, timeout: float = 10):
        return requests.get(self.url(path), timeout=timeout)

    def wait_until_listening(self, timeout: float = 30):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                return self.get(path="/health")
            except requests.ConnectionError:
                if self.returned.is_set():
                    raise AssertionError(f"server exited before listening: {self.error!r}")
                time.sleep(0.02)
        raise AssertionError("server never started listening")

    def wait_until_ready(self, timeout: float = 60):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.get("/health").status_code == 200:
                return
            time.sleep(0.05)
        raise AssertionError("server never became ready")


def make_server(factory=None, generator=None, **kwargs) -> OpenAIServer:
    return OpenAIServer(
        generator=generator,
        generator_factory=factory,
        model="fake-model",
        tool_parser=None,
        server_role=None,
        metadata_server_cfg=None,
        **kwargs,
    )


def test_requires_exactly_one_generator_source(model_dir):
    with pytest.raises(ValueError, match="exactly one"):
        make_server()
    with pytest.raises(ValueError, match="exactly one"):
        make_server(generator=FakeGenerator(model_dir), factory=lambda: FakeGenerator(model_dir))


def test_on_ready_requires_a_factory(model_dir):
    with pytest.raises(ValueError, match="requires generator_factory"):
        make_server(generator=FakeGenerator(model_dir), on_ready=lambda: None)


def test_listens_and_answers_503_while_initializing(model_dir):
    """The headline: STARTING is answerable, not connection-refused."""

    def slow_build():
        time.sleep(BUILD_SECONDS)
        return FakeGenerator(model_dir)

    server = make_server(factory=slow_build)
    with ServerRunner(server) as runner:
        started_at = time.monotonic()
        first = runner.wait_until_listening()
        listen_delay = time.monotonic() - started_at

        # Listening well before the build finishes is the whole point.
        assert listen_delay < BUILD_SECONDS
        assert first.status_code == 503
        assert first.headers[SERVER_STATE_HEADER] == ServerState.STARTING.value
        assert first.headers["retry-after"] == "1"
        assert first.json()["error"]["code"] == ServerState.STARTING.value

        # Not just /health: a 404 from a late-registered route would read to a
        # client as a real answer.
        assert runner.get("/v1/models").status_code == 503
        assert runner.get("/no/such/path").status_code == 503

        runner.wait_until_ready()
        assert runner.get("/health").status_code == 200
        assert runner.get("/v1/models").status_code == 200
        assert SERVER_STATE_HEADER not in runner.get("/health").headers


def test_ready_registers_the_generator_dependent_routes(model_dir):
    generator = FakeGenerator(model_dir)
    server = make_server(factory=lambda: generator)

    with ServerRunner(server) as runner:
        runner.wait_until_listening()
        runner.wait_until_ready()

        assert server.generator is generator
        paths = {r.path for r in server.app.routes if hasattr(r, "path")}
        assert {"/health", "/v1/completions", "/v1/chat/completions"} <= paths

    # Teardown must reach the generator built inside the lifespan.
    assert generator.shutdown_calls == 1


def test_on_ready_fires_only_once_requests_would_be_served(model_dir):
    """READY must mean "serves requests", not "listens".

    Attached frontends share one port via SO_REUSEPORT, so a frontend that
    announced itself at STARTING would take load-balanced traffic and 503 it.
    """
    observed = {}

    def on_ready():
        # Cannot poll /health from here: this runs on the serving loop, so a
        # blocking request to ourselves would deadlock.
        observed["ready_when_called"] = server._readiness.is_ready
        observed["routes"] = {r.path for r in server.app.routes if hasattr(r, "path")}

    server = make_server(factory=lambda: FakeGenerator(model_dir), on_ready=on_ready)
    with ServerRunner(server) as runner:
        runner.wait_until_listening()
        runner.wait_until_ready()

    assert observed["ready_when_called"] is True
    assert "/v1/completions" in observed["routes"]


def test_failed_initialization_stops_the_server_and_reraises(model_dir):
    """A permanent STARTING would just be a new undetectable state."""

    def failing_build():
        raise RuntimeError("could not load weights")

    server = make_server(factory=failing_build)
    with ServerRunner(server) as runner:
        runner.wait_until_listening()
        assert runner.returned.wait(timeout=60), "server kept serving"

    assert isinstance(runner.error, RuntimeError)
    assert "could not load weights" in str(runner.error)
    # Nothing is listening now, so a poller sees connection-refused: dead.
    with pytest.raises(requests.ConnectionError):
        runner.get("/health")


def test_shutdown_during_initialization_does_not_wait_for_the_build(model_dir):
    """SIGTERM mid-startup is the multi-minute hang this design removes."""
    building = threading.Event()
    release = threading.Event()
    finished = threading.Event()

    def blocked_build():
        building.set()
        release.wait(timeout=60)
        finished.set()
        return FakeGenerator(model_dir)

    server = make_server(factory=blocked_build)
    with ServerRunner(server) as runner:
        runner.wait_until_listening()
        assert building.wait(timeout=30)

        server._request_shutdown()
        stopped_within = runner.returned.wait(timeout=30)

    assert stopped_within, "shutdown waited for the build to finish"
    assert not finished.is_set(), "the build was awaited, not abandoned"
    # Let the abandoned thread finish so it does not outlive the test; in a
    # real server the process is exiting and the daemon thread just dies.
    release.set()
    finished.wait(timeout=30)


def test_eager_generator_is_ready_without_a_lifespan(model_dir):
    """The pre-existing path: routes and readiness settled in __init__."""
    generator = FakeGenerator(model_dir)
    server = make_server(generator=generator)

    assert server.generator is generator
    assert server._readiness.state is ServerState.READY
    paths = {r.path for r in server.app.routes if hasattr(r, "path")}
    assert "/v1/completions" in paths
