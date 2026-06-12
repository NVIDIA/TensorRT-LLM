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
"""Coordinator/worker disagg routing: cross-process placement contract.

CPU-only, MPI-free. Wires the real coordinator surface to a real worker-side
coordinator:

  * fake ctx/gen HTTP workers answer ``/health`` (readiness only),
  * a real ``CoordinatorServer`` (wrapping a ``DisaggCoordinatorService`` over the
    configured routers) runs in a uvicorn thread on an internal port,
  * a ``CoordinatorClient`` (what a worker holds) wraps only *stateful* routers
    in a ``CoordinatorDelegatingRouter`` whose ``get_next_server`` computes the
    routing key locally and POSTs it to the coordinator's ``/select``;
    ``finish_request`` releases coordinator-side state via ``/finish`` and the
    returned handle. *Stateless* routers (round_robin) are used as-is and place
    locally in the worker.

This proves the routing split: stateful routers (conversation, kv_cache_aware)
delegate to the coordinator via ``routing_key`` + ``get_next_server_by_key``,
while stateless routers never touch the coordinator.
"""

import asyncio
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import aiohttp
import pytest
import uvicorn

from tensorrt_llm.llmapi.disagg_utils import (CtxGenServerConfig,
                                              DisaggServerConfig, RouterConfig,
                                              ServerRole)
from tensorrt_llm.serve.coordinator_server import CoordinatorServer
from tensorrt_llm.serve.disagg_coordinator import (CoordinatorClient,
                                                   DisaggCoordinatorService)
from tensorrt_llm.serve.openai_protocol import (CompletionRequest,
                                                DisaggregatedParams)


@pytest.fixture(autouse=True)
def _reset_prometheus_registry():
    """Each coordinator builds role-prefixed Prometheus counters in the global
    default registry (via its readiness OpenAIHttpClients). In production the
    coordinator is a single process; here two coordinators share one pytest
    process, so clear the registry between tests to avoid duplicate-timeseries
    registration errors."""
    from prometheus_client import REGISTRY
    yield
    for collector in list(REGISTRY._collector_to_names):
        try:
            REGISTRY.unregister(collector)
        except Exception:
            pass


def _free_port():
    import socket
    s = socket.socket()
    # SO_REUSEADDR so a port left in TIME_WAIT by a sibling server in the same
    # suite can be rebound immediately (closes the alloc->bind race window).
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


class _FakeWorker:
    """Minimal HTTP worker: /health -> 200 (used for readiness only)."""

    def __init__(self):
        self.port = _free_port()

        class Handler(BaseHTTPRequestHandler):

            def log_message(self, *a):
                pass

            def do_GET(self):
                if self.path == "/health":
                    self.send_response(200)
                else:
                    self.send_response(404)
                self.end_headers()

        self._httpd = ThreadingHTTPServer(("127.0.0.1", self.port), Handler)
        self._thread = threading.Thread(target=self._httpd.serve_forever,
                                        daemon=True)

    @property
    def url(self):
        return f"127.0.0.1:{self.port}"

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *a):
        self._httpd.shutdown()


def _make_config(ctx_urls, gen_urls, ctx_router_type, gen_router_type):
    server_configs = [
        CtxGenServerConfig(type="ctx", hostname=u.split(":")[0],
                           port=int(u.split(":")[1])) for u in ctx_urls
    ] + [
        CtxGenServerConfig(type="gen", hostname=u.split(":")[0],
                           port=int(u.split(":")[1])) for u in gen_urls
    ]
    return DisaggServerConfig(
        server_configs=server_configs,
        ctx_router_config=RouterConfig(type=ctx_router_type,
                                       server_role=ServerRole.CONTEXT),
        gen_router_config=RouterConfig(type=gen_router_type,
                                       server_role=ServerRole.GENERATION))


def _client_factory(router, role, max_retries=1):
    from tensorrt_llm.serve.openai_client import OpenAIHttpClient
    return OpenAIHttpClient(router, role, 30, max_retries)


class _CoordinatorThread:
    """Run a CoordinatorServer (DisaggCoordinatorService) in a background thread."""

    def __init__(self, config):
        self.port = _free_port()
        self.url = f"http://127.0.0.1:{self.port}"
        # The coordinator builds its own owner routers from config.
        self._cluster = DisaggCoordinatorService(config, _client_factory)
        self._server = uvicorn.Server(
            uvicorn.Config(CoordinatorServer(self._cluster).app,
                           host="127.0.0.1", port=self.port,
                           log_level="warning"))
        self._thread = threading.Thread(target=self._server.run, daemon=True)

    def __enter__(self):
        self._thread.start()
        for _ in range(100):
            if self._server.started:
                break
            time.sleep(0.1)
        return self

    def __exit__(self, *a):
        self._server.should_exit = True
        self._thread.join(timeout=10)


async def _wait_coord_ready(url, timeout_s=30.0):
    deadline = time.time() + timeout_s
    async with aiohttp.ClientSession() as sess:
        while time.time() < deadline:
            try:
                async with sess.get(f"{url}/health", timeout=1) as r:
                    if r.status == 200:
                        return True
            except Exception:
                pass
            await asyncio.sleep(0.2)
    return False


def test_stateless_router_places_locally_in_worker():
    """A round-robin (stateless) router is NOT wrapped: the worker places locally
    with the real router and never calls the coordinator."""
    from tensorrt_llm.serve.router import (CoordinatorDelegatingRouter,
                                           RoundRobinRouter)
    with _FakeWorker() as ctx0, _FakeWorker() as gen0, _FakeWorker() as gen1:
        config = _make_config([ctx0.url], [gen0.url, gen1.url],
                              "round_robin", "round_robin")
        with _CoordinatorThread(config) as coord:
            assert asyncio.run(_wait_coord_ready(coord.url)), \
                "coordinator never became healthy"

            async def drive():
                remote = CoordinatorClient(coord.url, config)
                # Stateless -> real local router, not a delegating proxy.
                assert isinstance(remote.gen_router, RoundRobinRouter)
                assert not isinstance(remote.gen_router,
                                      CoordinatorDelegatingRouter)
                picks = []
                for _ in range(4):
                    req = CompletionRequest(model="m", prompt="hello")
                    server, _info = await remote.gen_router.get_next_server(req)
                    picks.append(server)
                    await remote.gen_router.finish_request(req)
                await remote.stop()
                return picks

            picks = asyncio.run(drive())
            assert set(picks) == {gen0.url, gen1.url}, \
                f"local round-robin should hit both gen workers, got {picks}"


def test_conversation_coordinator_sticky_by_conv_id():
    """Same conversation_id sticks to one gen worker; a stateful (conversation)
    router delegates placement to the coordinator via /select."""
    from tensorrt_llm.serve.router import CoordinatorDelegatingRouter
    with _FakeWorker() as ctx0, _FakeWorker() as gen0, _FakeWorker() as gen1:
        config = _make_config([ctx0.url], [gen0.url, gen1.url],
                              "round_robin", "conversation")
        with _CoordinatorThread(config) as coord:
            assert asyncio.run(_wait_coord_ready(coord.url))

            def _req(conv_id):
                return CompletionRequest(
                    model="m", prompt="hi",
                    disaggregated_params=DisaggregatedParams(
                        request_type="context_only", conversation_id=conv_id))

            async def drive():
                remote = CoordinatorClient(coord.url, config)
                # Stateful -> wrapped in a coordinator-delegating router.
                assert isinstance(remote.gen_router,
                                  CoordinatorDelegatingRouter)
                assert await remote.is_ready() is True
                first, _ = await remote.gen_router.get_next_server(_req("conv-A"))
                # Repeated conv-A requests must land on the same worker.
                repeats = []
                for _ in range(3):
                    s, _ = await remote.gen_router.get_next_server(_req("conv-A"))
                    repeats.append(s)
                await remote.stop()
                return first, repeats

            first, repeats = asyncio.run(drive())
            assert all(s == first for s in repeats), \
                f"conv-A must be sticky, got first={first} repeats={repeats}"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v", "-s"]))
