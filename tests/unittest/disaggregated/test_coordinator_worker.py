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
    returned handle. ``round_robin`` is used as-is and places locally in the
    worker.

This proves the routing split: globally stateful routers (load_balancing,
conversation, kv_cache_aware) delegate to the coordinator via ``routing_key`` +
``get_next_server_by_key``, while round-robin routing remains local.
"""

import asyncio
import json
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

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
    ConversationParams,
    DisaggregatedParams,
)
from tensorrt_llm.serve.router import (
    KV_CACHE_HASH_ALGO_V1,
    KV_CACHE_HASH_ALGO_V2,
    CoordinatorDelegatingRouter,
    KvCacheAwareRouter,
    LoadBalancingRouter,
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


def _make_config(
    ctx_urls, gen_urls, ctx_router_type, gen_router_type, ctx_router_args=None, gen_router_args=None
):
    server_configs = [
        CtxGenServerConfig(type="ctx", hostname=u.split(":")[0], port=int(u.split(":")[1]))
        for u in ctx_urls
    ] + [
        CtxGenServerConfig(type="gen", hostname=u.split(":")[0], port=int(u.split(":")[1]))
        for u in gen_urls
    ]
    return DisaggServerConfig(
        server_configs=server_configs,
        ctx_router_config=RouterConfig(
            type=ctx_router_type, server_role=ServerRole.CONTEXT, args=ctx_router_args or {}
        ),
        gen_router_config=RouterConfig(
            type=gen_router_type, server_role=ServerRole.GENERATION, args=gen_router_args or {}
        ),
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


def _active_gen_load(coordinator) -> int:
    """Requests the generation router still counts as in flight.

    ServerState exposes no accessor for it, so the tests read the counter the
    load-balancing router increments and decrements directly.
    """
    return sum(
        state._num_active_requests for state in coordinator.gen_router._server_state.values()
    )


@pytest.mark.asyncio
async def test_concurrent_select_replaces_same_reservation_without_leaking_load():
    config = _make_config([], ["gen0:8000", "gen1:8000"], "round_robin", "load_balancing")
    coordinator = DisaggCoordinatorService(config, _client_factory, reservation_timeout_secs=60)
    assert isinstance(coordinator.gen_router, LoadBalancingRouter)

    await asyncio.gather(
        coordinator.select("generation", {}, 123, None),
        coordinator.select("generation", {}, 123, None),
    )

    assert _active_gen_load(coordinator) == 1
    await coordinator.finish("generation", 123)
    assert _active_gen_load(coordinator) == 0


@pytest.mark.asyncio
async def test_coordinator_renew_replaces_reservation_timer():
    config = _make_config([], ["gen:8000"], "round_robin", "load_balancing")
    coordinator = DisaggCoordinatorService(config, _client_factory, reservation_timeout_secs=60)

    await coordinator.select("generation", {}, 123, None)
    reservation_key = ("generation", 123)
    original_reservation = coordinator._reservation_tasks[reservation_key]

    await coordinator.renew("generation", 123)

    assert coordinator._reservation_tasks[reservation_key] is not original_reservation
    await coordinator.finish("generation", 123)
    assert coordinator._reservation_tasks == {}


@pytest.mark.asyncio
async def test_coordinator_renew_restarts_the_full_expiration_period():
    """Renewing must reset the deadline, not merely swap the expiration task.

    A renewal that kept the original deadline would still pass an
    identity-only assertion, so this drives the observable behaviour: the
    reservation has to outlive the deadline it was created with and only drop
    once a full timeout has elapsed since the renewal.
    """
    # Generous margins: the assertions sit half a timeout away from either
    # deadline so a loaded CI machine oversleeping a little cannot flip them.
    timeout = 1.0
    config = _make_config([], ["gen:8000"], "round_robin", "load_balancing")
    coordinator = DisaggCoordinatorService(
        config, _client_factory, reservation_timeout_secs=timeout
    )

    await coordinator.select("generation", {}, 123, None)
    assert _active_gen_load(coordinator) == 1

    # Renew shortly before the original deadline would have fired.
    await asyncio.sleep(timeout * 0.75)
    await coordinator.renew("generation", 123)

    # Past the original deadline, but inside the renewed one: still reserved.
    await asyncio.sleep(timeout * 0.5)
    assert ("generation", 123) in coordinator._reservation_tasks
    assert _active_gen_load(coordinator) == 1

    # Past the renewed deadline: released.
    await asyncio.sleep(timeout)
    assert coordinator._reservation_tasks == {}
    assert _active_gen_load(coordinator) == 0


def test_coordinator_compacts_route_info():
    compact = DisaggCoordinatorService._compact_route_info(
        {
            "block_hashes": [["large-hash"]],
            "hash_algo": KV_CACHE_HASH_ALGO_V2,
            "matches": [64, 32],
            "match_length": 64,
            "num_tokens": 128,
            "server_info": {
                "tokens_per_block": 32,
                "disaggregated_params": {"ctx_info_endpoint": "tcp://ctx"},
            },
        }
    )

    assert compact == {
        "match_length": 64,
        "num_tokens": 128,
        "server_info": {"disaggregated_params": {"ctx_info_endpoint": "tcp://ctx"}},
    }


def test_coordinator_reservation_timeout_env(monkeypatch):
    monkeypatch.delenv(COORDINATOR_RESERVATION_TIMEOUT_ENV, raising=False)
    assert coordinator_reservation_timeout() == 180

    monkeypatch.setenv(COORDINATOR_RESERVATION_TIMEOUT_ENV, "60")
    assert coordinator_reservation_timeout() == 60


def test_coordinator_reservation_covers_request_timeout():
    config = _make_config([], [], "round_robin", "round_robin")
    coordinator = DisaggCoordinatorService(
        config,
        _client_factory,
        reservation_timeout_secs=60,
        request_timeout_secs=300,
    )

    assert coordinator._reservation_timeout_secs == 300


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
async def test_cluster_info_updates_readiness_and_local_servers():
    config = _make_config(["ctx-old:8001"], [], "round_robin", "round_robin")
    client = CoordinatorClient("http://coordinator", config)
    client.ctx_router.remove_server = AsyncMock(wraps=client.ctx_router.remove_server)
    client.ctx_router.add_server = AsyncMock(wraps=client.ctx_router.add_server)
    client.ctx_router.prepare_servers = AsyncMock()
    client.ctx_router._fetch_server_info = AsyncMock(return_value={})

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


def test_load_balancing_router_uses_global_coordinator_state():
    """Two fleet workers choose different servers while both requests are active."""
    from tensorrt_llm.serve.router import LoadBalancingRouter

    with _FakeWorker() as ctx0, _FakeWorker() as gen0, _FakeWorker() as gen1:
        config = _make_config([ctx0.url], [gen0.url, gen1.url], "round_robin", "load_balancing")
        with _CoordinatorThread(config) as coord:
            assert asyncio.run(_wait_coord_ready(coord.url)), "coordinator never became healthy"

            def _request(request_id):
                return CompletionRequest(
                    model="m",
                    prompt="hello",
                    disaggregated_params=DisaggregatedParams(
                        request_type="generation_only",
                        ctx_request_id=request_id,
                    ),
                )

            async def drive():
                first_worker = None
                second_worker = None
                try:
                    first_worker = CoordinatorClient(coord.url, config)
                    second_worker = CoordinatorClient(coord.url, config)
                    assert isinstance(first_worker.gen_router, CoordinatorDelegatingRouter)
                    assert isinstance(first_worker.gen_router._local, LoadBalancingRouter)
                    first_request, second_request = _request(1), _request(2)
                    first, _ = await first_worker.gen_router.get_next_server(first_request)
                    second, _ = await second_worker.gen_router.get_next_server(second_request)
                    await first_worker.gen_router.finish_request(first_request)
                    await second_worker.gen_router.finish_request(second_request)
                    return first, second
                finally:
                    if first_worker is not None:
                        await first_worker.stop()
                    if second_worker is not None:
                        await second_worker.stop()

            first, second = asyncio.run(drive())
            assert {first, second} == {gen0.url, gen1.url}


def test_token_weighted_router_releases_global_load():
    """A delegated finish releases the selected context server's token load."""
    from tensorrt_llm.serve.router import LoadBalancingRouter

    with _FakeWorker() as ctx0, _FakeWorker() as ctx1, _FakeWorker() as gen0:
        config = _make_config(
            [ctx0.url, ctx1.url],
            [gen0.url],
            "load_balancing",
            "round_robin",
            ctx_router_args={"use_tokens": True},
        )
        with _CoordinatorThread(config) as coord:
            assert asyncio.run(_wait_coord_ready(coord.url)), "coordinator never became healthy"

            def _request(request_id, num_tokens):
                return CompletionRequest(
                    model="m",
                    prompt=list(range(num_tokens)),
                    disaggregated_params=DisaggregatedParams(
                        request_type="context_only",
                        disagg_request_id=request_id,
                    ),
                )

            async def drive():
                worker = CoordinatorClient(coord.url, config)
                try:
                    assert isinstance(worker.ctx_router, CoordinatorDelegatingRouter)
                    assert isinstance(worker.ctx_router._local, LoadBalancingRouter)
                    large_request = _request(1, 100)
                    small_request = _request(2, 1)
                    released_request = _request(3, 1)
                    large_server, _ = await worker.ctx_router.get_next_server(large_request)
                    await worker.ctx_router.get_next_server(small_request)
                    await worker.ctx_router.finish_request(large_request)
                    await worker.ctx_router._finish_queue.join()
                    released_server, _ = await worker.ctx_router.get_next_server(released_request)
                    await worker.ctx_router.finish_request(small_request)
                    await worker.ctx_router.finish_request(released_request)
                    return large_server, released_server
                finally:
                    await worker.stop()

            large_server, released_server = asyncio.run(drive())
            assert released_server == large_server


@pytest.mark.asyncio
async def test_delegating_router_syncs_coordinator_server_list():
    config = _make_config(["ctx-old:8000"], [], "load_balancing", "round_robin")
    client = CoordinatorClient("http://coordinator", config)

    assert isinstance(client.ctx_router, CoordinatorDelegatingRouter)
    await client._sync_router_server_lists(
        {"server_lists": {"context": ["ctx-new:8001"], "generation": []}}
    )

    assert client.ctx_router.servers == ["ctx-new:8001"]
    await client.stop()


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

    await client._sync_router_server_lists(
        {"server_lists": {"context": ["ctx-new:8001"], "generation": []}}
    )

    assert client.ctx_router.servers == ["ctx-new:8001"]
    client.ctx_router._fetch_server_info.assert_awaited_once_with("ctx-new:8001", None)
    await client.stop()


@pytest.mark.asyncio
async def test_stateless_router_keeps_old_server_when_replacement_is_unprepared():
    config = _make_config(["ctx-old:8000"], [], "round_robin", "round_robin")
    client = CoordinatorClient("http://coordinator", config)
    client.ctx_router._prepared_ready_servers.add("ctx-old:8000")
    client.ctx_router._fetch_server_info = AsyncMock(
        side_effect=RuntimeError("server info unavailable")
    )

    with pytest.raises(RuntimeError, match="Failed to prepare ctx-new:8001"):
        await client._sync_router_server_lists(
            {"server_lists": {"context": ["ctx-new:8001"], "generation": []}}
        )

    assert client.ctx_router.servers == ["ctx-old:8000"]
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
                    conversation_params=ConversationParams(conversation_id=conv_id),
                    disaggregated_params=DisaggregatedParams(
                        request_type="generation_only",
                        ctx_request_id=request_id,
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
                    conversation_params=ConversationParams(conversation_id="conv-A"),
                    disaggregated_params=DisaggregatedParams(
                        request_type="generation_only",
                        ctx_request_id=assigned_id,
                        disagg_request_id=None,
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


def test_coordinator_session_gives_up_idle_connections_before_the_server():
    """The fleet's pool must drop an idle connection before the coordinator does.

    aiohttp never probes a pooled connection before reuse, so if the SERVER
    closes first the next /select borrows a half-closed socket and fails
    instantly with BrokenPipeError/ConnectionResetError -- a 500 on whichever
    live request happened to draw that connection.
    """
    from tensorrt_llm.serve import disagg_coordinator as dc
    from tensorrt_llm.serve.coordinator_server import TIMEOUT_KEEP_ALIVE

    assert dc.COORDINATOR_KEEPALIVE_TIMEOUT_S < TIMEOUT_KEEP_ALIVE

    for url, connector_name in (
        ("unix:/tmp/trtllm_disagg_coord_8333.sock", "UnixConnector"),
        ("http://coordinator:8332", "TCPConnector"),
    ):
        with (
            patch.object(dc.aiohttp, connector_name) as connector,
            patch.object(dc.aiohttp, "ClientSession"),
        ):
            dc.make_coordinator_session(url)
        assert connector.call_args.kwargs["limit"] == 0
        assert connector.call_args.kwargs["keepalive_timeout"] == dc.COORDINATOR_KEEPALIVE_TIMEOUT_S


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("keep_alive_kwargs", "expected_timeout"),
    [({}, 10), ({"keep_alive_timeout": 3600}, 3600)],
)
async def test_coordinator_keep_alive_timeout_is_passed_to_uvicorn(
    monkeypatch, keep_alive_kwargs, expected_timeout
):
    """Both coordinator listeners honor the configured keep-alive timeout."""
    from tensorrt_llm.serve import coordinator_server

    server = object.__new__(CoordinatorServer)
    server._coordinator = AsyncMock()
    server.app = object()

    config_factory = Mock(return_value=object())
    uvicorn_server = SimpleNamespace(serve=AsyncMock())
    monkeypatch.setattr(coordinator_server.uvicorn, "Config", config_factory)
    monkeypatch.setattr(coordinator_server.uvicorn, "Server", Mock(return_value=uvicorn_server))

    await server("localhost", 8332, uds="/tmp/coord.sock", **keep_alive_kwargs)

    assert config_factory.call_count == 2  # UDS (hot path) + TCP (health)
    for call in config_factory.call_args_list:
        assert call.kwargs["timeout_keep_alive"] == expected_timeout


@pytest.mark.asyncio
@pytest.mark.parametrize("uds", [None, "/tmp/coord.sock"])
async def test_coordinator_tcp_listener_uses_the_prebound_socket(monkeypatch, uds):
    """The TCP listener must serve a socket the caller already bound.

    Letting uvicorn bind the host string makes it resolve the name, and a
    hostname whose AAAA record is link-local resolves to fe80:: with scope id 0
    -- an address the kernel always refuses, since only an interface name
    carries the scope. The standalone server and the fleet workers already
    hand uvicorn a bound socket; the coordinator has to do the same.
    """
    from tensorrt_llm.serve import coordinator_server

    server = object.__new__(CoordinatorServer)
    server._coordinator = AsyncMock()
    server.app = object()

    tcp_server = SimpleNamespace(serve=AsyncMock())
    uds_server = SimpleNamespace(serve=AsyncMock())
    # uvicorn.Server is constructed UDS first, then TCP.
    servers = [uds_server, tcp_server] if uds else [tcp_server]
    monkeypatch.setattr(coordinator_server.uvicorn, "Config", Mock(return_value=object()))
    monkeypatch.setattr(coordinator_server.uvicorn, "Server", Mock(side_effect=servers))

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    with sock:
        await server("fe80::ac31:4bff:fee6:2d43", 8332, uds=uds, sockets=[sock])

        tcp_server.serve.assert_awaited_once_with(sockets=[sock])
        if uds:
            # The UDS listener binds the path itself and takes no socket.
            uds_server.serve.assert_awaited_once_with()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v", "-s"]))
