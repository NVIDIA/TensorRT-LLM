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
from unittest.mock import AsyncMock

import aiohttp
import pytest
import uvicorn

from tensorrt_llm.llmapi.disagg_utils import (
    CtxGenServerConfig,
    DisaggClusterConfig,
    DisaggServerConfig,
    RouterConfig,
    ServerRole,
)
from tensorrt_llm.serve.coordinator_server import CoordinatorServer
from tensorrt_llm.serve.disagg_coordinator import (
    COORDINATOR_RESERVATION_TIMEOUT_ENV,
    CoordinatorClient,
    DisaggCoordinatorService,
    coordinator_reservation_timeout,
)
from tensorrt_llm.serve.openai_protocol import (
    ChatCompletionRequest,
    CompletionRequest,
    DisaggregatedParams,
)
from tensorrt_llm.serve.router import (
    KV_CACHE_HASH_ALGO_V1,
    KV_CACHE_HASH_ALGO_V2,
    CoordinatorDelegatingRouter,
    KvCacheAwareRouter,
)
from tensorrt_llm.serve.router_utils import BlockHashMixin as SharedBlockHashMixin


@pytest.fixture(autouse=True)
def _reset_prometheus_registry():
    """Reset role-prefixed Prometheus counters.

    Tests create multiple coordinators in one process, so clear their shared
    default registry between tests to avoid duplicate-timeseries errors.
    """
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
    """Minimal HTTP worker exposing health and routing metadata."""

    def __init__(self, server_info=None):
        self.port = _free_port()
        server_info_body = json.dumps(server_info or {}).encode()

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):
                if self.path == "/health":
                    self.send_response(200)
                elif self.path == "/server_info":
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(server_info_body)))
                else:
                    self.send_response(404)
                self.end_headers()
                if self.path == "/server_info":
                    self.wfile.write(server_info_body)

        self._httpd = ThreadingHTTPServer(("127.0.0.1", self.port), Handler)
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)

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
        CtxGenServerConfig(type="ctx", hostname=u.split(":")[0], port=int(u.split(":")[1]))
        for u in ctx_urls
    ] + [
        CtxGenServerConfig(type="gen", hostname=u.split(":")[0], port=int(u.split(":")[1]))
        for u in gen_urls
    ]
    return DisaggServerConfig(
        server_configs=server_configs,
        ctx_router_config=RouterConfig(type=ctx_router_type, server_role=ServerRole.CONTEXT),
        gen_router_config=RouterConfig(type=gen_router_type, server_role=ServerRole.GENERATION),
    )


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
            uvicorn.Config(
                CoordinatorServer(self._cluster).app,
                host="127.0.0.1",
                port=self.port,
                log_level="warning",
            )
        )
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


def test_coordinator_rejects_unknown_role():
    config = _make_config([], [], "round_robin", "round_robin")
    coordinator = DisaggCoordinatorService(config, _client_factory)

    with pytest.raises(ValueError, match="Unsupported coordinator role"):
        coordinator._router_for_role("typo")


def test_kv_router_rejects_mixed_prepared_hash_algorithms():
    router = KvCacheAwareRouter(server_role=ServerRole.CONTEXT, servers=["server-a", "server-b"])
    router._prepared_ready_servers.update(router.servers)
    router._server_state["server-a"].set_hash_algo(KV_CACHE_HASH_ALGO_V1)
    router._server_state["server-b"].set_hash_algo(KV_CACHE_HASH_ALGO_V2)

    with pytest.raises(RuntimeError, match="one hash algorithm per role"):
        router.routing_key_config()


@pytest.mark.asyncio
async def test_coordinator_exposes_role_hash_algorithm():
    ctx_server = "127.0.0.1:1234"
    gen_server = "127.0.0.1:1235"
    config = _make_config([ctx_server], [gen_server], "kv_cache_aware", "kv_cache_aware")
    coordinator = DisaggCoordinatorService(config, _client_factory)
    ctx_router = coordinator.ctx_router
    gen_router = coordinator.gen_router
    assert isinstance(ctx_router, KvCacheAwareRouter)
    assert isinstance(gen_router, KvCacheAwareRouter)
    ctx_router._prepared_ready_servers.add(ctx_server)
    gen_router._prepared_ready_servers.add(gen_server)
    ctx_router._server_state[ctx_server].set_hash_algo(KV_CACHE_HASH_ALGO_V2)
    gen_router._server_state[gen_server].set_hash_algo(KV_CACHE_HASH_ALGO_V1)

    info = await coordinator.cluster_info()

    assert info["routing_key_configs"]["context"] == {
        "tokens_per_block": 32,
        "kv_cache_hash_algo": KV_CACHE_HASH_ALGO_V2,
    }
    assert info["routing_key_configs"]["generation"] == {
        "tokens_per_block": 32,
        "kv_cache_hash_algo": KV_CACHE_HASH_ALGO_V1,
    }


@pytest.mark.asyncio
async def test_coordinator_expires_stale_reservation():
    config = _make_config([], ["gen:8000"], "round_robin", "conversation")
    coordinator = DisaggCoordinatorService(
        config,
        _client_factory,
        reservation_timeout_secs=0.01,
    )

    await coordinator.select("generation", "conversation", 123, None)
    assert coordinator.gen_router._server_content_load["gen:8000"] == 1

    await asyncio.sleep(0.02)

    assert coordinator.gen_router._server_content_load["gen:8000"] == 0
    assert coordinator._reservation_tasks == {}


def test_coordinator_reservation_timeout_env(monkeypatch):
    monkeypatch.delenv(COORDINATOR_RESERVATION_TIMEOUT_ENV, raising=False)
    assert coordinator_reservation_timeout() == 180

    monkeypatch.setenv(COORDINATOR_RESERVATION_TIMEOUT_ENV, "60")
    assert coordinator_reservation_timeout() == 60


def test_coordinator_client_configures_empty_delegating_kv_router():
    config = _make_config([], [], "kv_cache_aware", "round_robin")
    client = CoordinatorClient("http://coordinator", config)
    assert isinstance(client.ctx_router, CoordinatorDelegatingRouter)
    local = client.ctx_router._local
    assert isinstance(local, KvCacheAwareRouter)
    assert local.servers == []

    client._sync_delegating_router_configs(
        {
            "routing_key_configs": {
                "context": {
                    "tokens_per_block": 64,
                    "kv_cache_hash_algo": KV_CACHE_HASH_ALGO_V2,
                }
            }
        }
    )

    request = CompletionRequest(model="m", prompt=[1, 2, 3])
    routing_key = local.routing_key(request)
    assert local.servers == []
    assert local._tokens_per_block == 64
    assert set(routing_key["block_hashes_by_algo"]) == {KV_CACHE_HASH_ALGO_V2}
    assert routing_key["num_tokens"] == 3
    assert "token_lists" not in routing_key


@pytest.mark.asyncio
async def test_coordinator_client_readiness_is_cached():
    config = _make_config([], [], "round_robin", "round_robin")
    client = CoordinatorClient("http://coordinator", config)
    client._is_ready = True

    assert await client.is_ready() is True
    assert client._session is None
    await client.stop()


@pytest.mark.asyncio
async def test_coordinator_state_sync_starts_in_background():
    config = _make_config([], [], "round_robin", "round_robin")
    client = CoordinatorClient("http://coordinator", config)
    client._state_sync_interval_s = 10
    client._await_coordinator = AsyncMock(
        return_value={
            "is_ready": True,
            "server_lists": {"context": [], "generation": []},
        }
    )
    client._sync_coordinator_state = AsyncMock()

    await client.start()

    assert client._is_ready is True
    client._sync_coordinator_state.assert_called_once_with(10)
    await client.stop()


def test_service_discovery_sets_coordinator_state_sync_interval():
    config = _make_config([], [], "round_robin", "round_robin")
    config.disagg_cluster_config = DisaggClusterConfig(
        cluster_uri="http://cluster-storage",
        heartbeat_interval_sec=7,
    )

    client = CoordinatorClient("http://coordinator", config)

    assert client._state_sync_interval_s == 7


@pytest.mark.asyncio
async def test_cluster_info_updates_readiness_and_stateless_servers():
    config = _make_config(["ctx-old:8001"], [], "round_robin", "round_robin")
    client = CoordinatorClient("http://coordinator", config)
    client.ctx_router.remove_server = AsyncMock(wraps=client.ctx_router.remove_server)
    client.ctx_router.add_server = AsyncMock(wraps=client.ctx_router.add_server)
    client.ctx_router.prepare_servers = AsyncMock()

    await client._apply_cluster_info(
        {
            "is_ready": True,
            "server_lists": {
                "context": ["ctx-new:8001"],
                "generation": [],
            },
        }
    )

    assert await client.is_ready() is True
    assert client.ctx_router.servers == ["ctx-new:8001"]
    client.ctx_router.remove_server.assert_awaited_once_with("ctx-old:8001")
    client.ctx_router.add_server.assert_awaited_once_with("ctx-new:8001")
    client.ctx_router.prepare_servers.assert_awaited_once()
    await client.stop()


def test_content_affinity_key_uses_fixed_seed():
    request = ChatCompletionRequest(model="m", messages=[{"role": "user", "content": "hello"}])

    assert KvCacheAwareRouter._content_affinity_key(request) == 7306401829117098140


def test_prefix_token_cache_retokenizes_extended_text():
    class BoundarySensitiveTokenizer:
        def __init__(self):
            self.calls = []

        def encode(self, text, add_special_tokens=False):
            assert add_special_tokens is False
            self.calls.append(text)
            return {"ab": [1], "abc": [2], "c": [3]}[text]

    tokenizer = BoundarySensitiveTokenizer()
    block_hashing = SharedBlockHashMixin()
    block_hashing._init_block_hashing()

    assert block_hashing._encode_with_prefix_cache("ab", 1, tokenizer) == [1]
    assert block_hashing._encode_with_prefix_cache("abc", 1, tokenizer) == [2]
    assert block_hashing._encode_with_prefix_cache("abc", 1, tokenizer) == [2]
    assert tokenizer.calls == ["ab", "abc"]


def test_stateless_router_places_locally_in_worker():
    """Verify stateless round-robin placement remains local.

    The worker uses the real router without calling the coordinator.
    """
    from tensorrt_llm.serve.router import CoordinatorDelegatingRouter, RoundRobinRouter

    with _FakeWorker() as ctx0, _FakeWorker() as gen0, _FakeWorker() as gen1:
        config = _make_config([ctx0.url], [gen0.url, gen1.url], "round_robin", "round_robin")
        with _CoordinatorThread(config) as coord:
            assert asyncio.run(_wait_coord_ready(coord.url)), "coordinator never became healthy"

            async def drive():
                remote = CoordinatorClient(coord.url, config)
                # Stateless -> real local router, not a delegating proxy.
                assert isinstance(remote.gen_router, RoundRobinRouter)
                assert not isinstance(remote.gen_router, CoordinatorDelegatingRouter)
                picks = []
                for _ in range(4):
                    req = CompletionRequest(model="m", prompt="hello")
                    server, _info = await remote.gen_router.get_next_server(req)
                    picks.append(server)
                    await remote.gen_router.finish_request(req)
                await remote.stop()
                return picks

            picks = asyncio.run(drive())
            assert set(picks) == {gen0.url, gen1.url}, (
                f"local round-robin should hit both gen workers, got {picks}"
            )


def test_static_stateless_router_prepares_generation_first_server_info():
    """Fleet startup prepares static local routers for generation-first."""
    from tensorrt_llm.serve.router import RoundRobinRouter

    ctx_info_endpoint = "tcp://127.0.0.1:12345"
    ctx_server_info = {"disaggregated_params": {"ctx_info_endpoint": ctx_info_endpoint}}
    with _FakeWorker(ctx_server_info) as ctx0, _FakeWorker() as gen0:
        config = _make_config([ctx0.url], [gen0.url], "round_robin", "round_robin")
        config.schedule_style = "generation_first"
        with _CoordinatorThread(config) as coord:
            assert asyncio.run(_wait_coord_ready(coord.url))

            async def drive():
                remote = CoordinatorClient(coord.url, config)
                await remote.start()
                assert isinstance(remote.ctx_router, RoundRobinRouter)
                request = CompletionRequest(model="m", prompt="hello")
                server, info = await remote.ctx_router.get_next_server(request)
                await remote.stop()
                return server, info

            server, info = asyncio.run(drive())
            assert server == ctx0.url
            assert (
                info["server_info"]["disaggregated_params"]["ctx_info_endpoint"]
                == ctx_info_endpoint
            )


@pytest.mark.asyncio
async def test_stateless_router_syncs_coordinator_server_add_remove():
    """Coordinator server lists propagate metadata topology changes."""
    config = _make_config(["ctx-old:8000"], [], "round_robin", "round_robin")
    client = CoordinatorClient("http://coordinator", config)
    client.ctx_router._fetch_server_info = AsyncMock(return_value={})

    await client._sync_stateless_routers(
        {"server_lists": {"context": ["ctx-new:8001"], "generation": []}}
    )

    assert client.ctx_router.servers == ["ctx-new:8001"]
    client.ctx_router._fetch_server_info.assert_awaited_once_with("ctx-new:8001", None)
    await client.stop()


def test_conversation_coordinator_sticky_by_conv_id():
    """Verify conversation IDs remain sticky through delegated routing.

    The stateful generation router delegates placement to coordinator
    ``/select``.
    """
    from tensorrt_llm.serve.router import CoordinatorDelegatingRouter

    with _FakeWorker() as ctx0, _FakeWorker() as gen0, _FakeWorker() as gen1:
        config = _make_config([ctx0.url], [gen0.url, gen1.url], "round_robin", "conversation")
        with _CoordinatorThread(config) as coord:
            assert asyncio.run(_wait_coord_ready(coord.url))

            def _req(conv_id, request_id):
                return CompletionRequest(
                    model="m",
                    prompt="hi",
                    disaggregated_params=DisaggregatedParams(
                        request_type="generation_only",
                        ctx_request_id=request_id,
                        conversation_id=conv_id,
                    ),
                )

            async def drive():
                remote = CoordinatorClient(coord.url, config)
                await remote.start()
                # Stateful -> wrapped in a coordinator-delegating router.
                assert isinstance(remote.gen_router, CoordinatorDelegatingRouter)
                assert await remote.is_ready() is True
                first_request = _req("conv-A", 1)
                first, _ = await remote.gen_router.get_next_server(first_request)
                await remote.gen_router.finish_request(first_request)
                # Repeated conv-A requests must land on the same worker.
                repeats = []
                for request_id in range(2, 5):
                    request = _req("conv-A", request_id)
                    s, _ = await remote.gen_router.get_next_server(request)
                    repeats.append(s)
                    await remote.gen_router.finish_request(request)
                await remote.stop()
                return first, repeats

            first, repeats = asyncio.run(drive())
            assert all(s == first for s in repeats), (
                f"conv-A must be sticky, got first={first} repeats={repeats}"
            )


def test_worker_generates_disagg_request_id_before_generation_routing():
    """Generation routing uses the ID generated by the coordinator client."""
    from tensorrt_llm.serve.router import CoordinatorDelegatingRouter

    with _FakeWorker() as ctx0, _FakeWorker() as gen0:
        config = _make_config([ctx0.url], [gen0.url], "round_robin", "conversation")
        with _CoordinatorThread(config) as coord:
            assert asyncio.run(_wait_coord_ready(coord.url))

            async def drive():
                remote = CoordinatorClient(coord.url, config)
                assert isinstance(remote.gen_router, CoordinatorDelegatingRouter)
                assigned_id = await remote.get_disagg_request_id()
                request = CompletionRequest(
                    model="m",
                    prompt="hello",
                    disaggregated_params=DisaggregatedParams(
                        request_type="generation_only",
                        ctx_request_id=assigned_id,
                        disagg_request_id=None,
                        conversation_id="conv-A",
                    ),
                )
                await remote.gen_router.get_next_server(request)
                assert request.disaggregated_params.disagg_request_id is None
                assert request.disaggregated_params.ctx_request_id == assigned_id
                await remote.gen_router.finish_request(request)
                await remote.stop()
                return assigned_id

            assigned_id = asyncio.run(drive())
            assert assigned_id > 0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v", "-s"]))
