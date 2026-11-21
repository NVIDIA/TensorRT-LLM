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
from aiohttp import web
from aiohttp.test_utils import TestServer
from fastapi import HTTPException

from tensorrt_llm.llmapi.disagg_utils import (
    CtxGenServerConfig,
    DisaggClusterConfig,
    DisaggServerConfig,
    MinimalInstances,
    RouterConfig,
    ServerRole,
)
from tensorrt_llm.serve import disagg_coordinator as coordinator_module
from tensorrt_llm.serve import router as router_module
from tensorrt_llm.serve.cluster_storage import (
    StorageItem,
    WatchEvent,
    WatchEventQueue,
    WatchEventType,
)
from tensorrt_llm.serve.coordinator_server import CoordinatorServer
from tensorrt_llm.serve.disagg_auto_scaling import DisaggClusterManager, WorkerInfo
from tensorrt_llm.serve.disagg_coordinator import (
    COORDINATOR_RESERVATION_TIMEOUT_ENV,
    CoordinatorClient,
    DisaggCoordinatorService,
    coordinator_reservation_timeout,
)
from tensorrt_llm.serve.openai_disagg_server import OpenAIDisaggServer
from tensorrt_llm.serve.openai_protocol import (
    ChatCompletionRequest,
    CompletionRequest,
    ConversationParams,
    DisaggregatedParams,
)
from tensorrt_llm.serve.router import (
    KV_CACHE_HASH_ALGO_V1,
    KV_CACHE_HASH_ALGO_V2,
    ROUTING_CONTENT_REQUIRED,
    ConversationRouter,
    CoordinatorDelegatingRouter,
    KvCacheAwareRouter,
    LoadBalancingRouter,
    RoundRobinRouter,
    ServerPreparationPending,
)
from tensorrt_llm.serve.router_utils import BlockHashMixin as SharedBlockHashMixin

pytestmark = pytest.mark.cpu_only


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


def test_coordinator_rejects_non_loopback_http_cluster_storage():
    config = _make_config([], [], "round_robin", "round_robin")
    config.hostname = "0.0.0.0"
    config.disagg_cluster_config = DisaggClusterConfig(cluster_uri="http://127.0.0.1:18000")

    with pytest.raises(ValueError, match="loopback-only"):
        DisaggCoordinatorService(config, _client_factory)


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
async def test_coordinator_publishes_worker_render_fingerprints_to_the_frontend():
    ctx_server = "127.0.0.1:1234"
    config = _make_config([ctx_server], [], "kv_cache_aware", "round_robin")
    coordinator = DisaggCoordinatorService(config, _client_factory)
    fingerprint = {
        "digest": "d",
        "compared": {"tokenizer": {"complete": True, "vocab_sha256": "big"}},
        "info": {"tool_parser": None, "reasoning_parser": "poolside_v1"},
    }
    coordinator.ctx_router._server_info = {ctx_server: {"render_fingerprint": fingerprint}}

    info = await coordinator.cluster_info()

    # Only what the match reads is published, not the whole fingerprint; the worker's
    # reasoning parser rides along for the router's decision about forwarding ids.
    assert info["render_fingerprints"]["context"] == {
        ctx_server: {
            "digest": "d",
            "compared": {"use_harmony": False, "tokenizer": {"complete": True}},
            "reasoning_parser": "poolside_v1",
        }
    }
    assert info["render_fingerprints"]["generation"] == {}

    client = CoordinatorClient("http://coordinator", config)
    local = client.ctx_router._local
    client._sync_render_fingerprints(info)
    assert local._published_render_fingerprints == info["render_fingerprints"]["context"]
    await client.stop()


@pytest.mark.asyncio
async def test_coordinator_expires_stale_reservation():
    config = _make_config(
        [],
        ["gen:8000"],
        "round_robin",
        "conversation",
        gen_router_args={"new_conversation_min_match_blocks": 0},
    )
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
    """Reuse ready state and keep a shared refresh alive across caller deadlines."""
    config = _make_config([], [], "round_robin", "round_robin")
    client = CoordinatorClient("http://coordinator", config)
    client.cluster_info = AsyncMock(side_effect=[{"is_ready": False}, {"is_ready": True}])

    assert await client.is_ready() is False
    assert await client.is_ready() is True
    assert client.cluster_info.await_count == 2
    client.cluster_info.reset_mock()

    assert await client.is_ready() is True
    client.cluster_info.assert_not_awaited()
    assert client._session is None

    client._is_ready = False
    started = asyncio.Event()
    release = asyncio.Event()

    async def fetch_when_released() -> dict:
        started.set()
        await release.wait()
        return {"is_ready": True}

    client.cluster_info.side_effect = fetch_when_released
    refresh = client._get_refresh_task()
    try:
        await asyncio.wait_for(started.wait(), timeout=30)
        # The event stays blocked until after both deadlines, regardless of runner speed.
        with patch.object(coordinator_module, "COORDINATOR_READINESS_TIMEOUT_S", 0.01):
            assert await client.is_ready() is False
            assert await client.is_ready() is False
        assert client._refresh_task is refresh
        assert not refresh.done()
        client.cluster_info.assert_awaited_once()
        release.set()
        assert await asyncio.wait_for(asyncio.shield(refresh), timeout=30) is True
        assert await client.is_ready() is True
        client.cluster_info.assert_awaited_once()
    finally:
        release.set()
        await client.stop()


@pytest.mark.asyncio
async def test_coordinator_state_sync_starts_in_background():
    """Recover from refresh failures and drain owned work before closing routers."""
    config = _make_config([], [], "round_robin", "round_robin")
    client = CoordinatorClient("http://coordinator", config)
    client._state_sync_interval_s = 10
    client._await_coordinator = AsyncMock(
        return_value={
            "is_ready": True,
            "server_lists": {"context": [], "generation": []},
        }
    )
    client.cluster_info = AsyncMock(
        side_effect=[
            RuntimeError("coordinator unavailable"),
            {"is_ready": False},
            {"is_ready": True},
            {"is_ready": True},
        ]
    )
    client._sync_router_server_lists = AsyncMock(
        side_effect=[None, None, RuntimeError("router preparation failed"), None]
    )
    readiness = []

    async def tick(interval_s: float) -> None:
        """Observe each completed iteration and stop after the recovery sequence."""
        assert interval_s == client._state_sync_interval_s
        readiness.append(client._is_ready)
        if len(readiness) == 5:
            raise asyncio.CancelledError

    # Control loop iterations without patching other tasks' asyncio.sleep.
    controlled_asyncio = SimpleNamespace(**{**vars(asyncio), "sleep": tick})
    with patch.object(coordinator_module, "asyncio", controlled_asyncio):
        try:
            await client.start()
            assert client._is_ready is True
            assert client._sync_task is not None
            with pytest.raises(asyncio.CancelledError):
                await client._sync_task
            assert readiness == [True, False, False, False, True]

            started = asyncio.Event()

            async def unfinished_fetch() -> dict:
                started.set()
                await asyncio.Event().wait()
                return {"is_ready": True}

            client.cluster_info.side_effect = unfinished_fetch
            refresh = client._get_refresh_task()
            await asyncio.wait_for(started.wait(), timeout=30)
            close_router = client.ctx_router.close

            async def close_after_refresh() -> None:
                assert refresh.cancelled()
                await close_router()

            client.ctx_router.close = close_after_refresh
        finally:
            await client.stop()
    assert refresh.cancelled()
    assert client._refresh_task is None
    assert await client.is_ready() is False


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

    async def fetch_server_info(server: str, timeout: float) -> dict:
        assert client._is_ready is False
        return {}

    client.ctx_router._fetch_server_info = AsyncMock(side_effect=fetch_server_info)
    client._is_ready = True

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


def test_router_tokenization_cache_reuses_cached_prompt_prefix():
    class CharTokenizer:
        is_fast = True

        def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
            return {
                "input_ids": [ord(c) for c in text],
                "offset_mapping": [(i, i + 1) for i in range(len(text))],
            }

    tokenizer = CharTokenizer()
    block_hashing = SharedBlockHashMixin()
    block_hashing._init_block_hashing(enable_tokenization_cache=True)
    previous = "system prompt and tools. " * 400
    for prompt in (previous, previous + "<|im_start|>user\nmore"):
        ids = block_hashing._encode_with_prefix_cache(prompt, 1, tokenizer)
        assert ids == [ord(c) for c in prompt]
    assert block_hashing._prefix_token_cache(tokenizer).hits == 1


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


_NEW_CONV_ROUTER_ARGS = {"tokens_per_block": 16, "new_conversation_min_match_blocks": 16}
_SHARED_PREFIX = list(range(1000, 1256))  # 16 blocks of 16 tokens


def _ctx_conv_request(conv_id, tokens, request_id):
    return CompletionRequest(
        model="m",
        prompt=list(tokens),
        conversation_params=ConversationParams(conversation_id=conv_id),
        disaggregated_params=DisaggregatedParams(
            request_type="context_only", disagg_request_id=request_id
        ),
    )


@pytest.mark.asyncio
async def test_coordinator_select_without_placement_reserves_nothing():
    config = _make_config(
        ["ctx0:8000", "ctx1:8000"],
        ["gen:8000"],
        "conversation",
        "round_robin",
        ctx_router_args=dict(_NEW_CONV_ROUTER_ARGS),
    )
    coordinator = DisaggCoordinatorService(config, _client_factory)
    router = coordinator.ctx_router

    server, info, req_id = await coordinator.select("context", "t::sa:a", 7, None)
    assert server is None and req_id == 7
    assert info == {ROUTING_CONTENT_REQUIRED: True}
    assert coordinator._reservation_tasks == {}
    assert sum(router._server_content_load.values()) == 0

    request = _ctx_conv_request("t::sa:a", _SHARED_PREFIX + [1] * 40, 7)
    server, _, _ = await coordinator.select("context", router.content_routing_key(request), 7, None)
    assert server in ("ctx0:8000", "ctx1:8000")
    assert ("context", 7) in coordinator._reservation_tasks
    await coordinator.finish("context", 7)
    assert coordinator._reservation_tasks == {}


def test_conversation_coordinator_places_new_conversation_by_content():
    """Delegated routing places a new sibling conversation by content.

    The sibling follows the earlier conversation, and only requests of new
    conversation_ids are hashed on the worker.
    """
    with _FakeWorker() as ctx0, _FakeWorker() as ctx1, _FakeWorker() as gen0:
        config = _make_config(
            [ctx0.url, ctx1.url],
            [gen0.url],
            "conversation",
            "round_robin",
            ctx_router_args=dict(_NEW_CONV_ROUTER_ARGS),
        )
        with _CoordinatorThread(config) as coord:
            assert asyncio.run(_wait_coord_ready(coord.url))

            async def drive():
                remote = CoordinatorClient(coord.url, config)
                await remote.start()
                router = remote.ctx_router
                assert isinstance(router, CoordinatorDelegatingRouter)
                hashed = []
                original = router._local.content_routing_key

                def counting(request):
                    hashed.append(request.conversation_params.conversation_id)
                    return original(request)

                router._local.content_routing_key = counting
                sibling_a = _ctx_conv_request("t::sa:a", _SHARED_PREFIX + [1] * 40, 1)
                other = _ctx_conv_request("t2", list(range(5000, 5300)), 2)
                sibling_b = _ctx_conv_request("t::sa:b", _SHARED_PREFIX + [2] * 40, 3)
                turn_2 = _ctx_conv_request("t::sa:a", _SHARED_PREFIX + [1] * 80, 4)
                servers = {}
                for request in (sibling_a, other, sibling_b, turn_2):
                    # All stay in flight, so least-loaded avoids busy servers.
                    servers[request.disaggregated_params.disagg_request_id] = (
                        await router.get_next_server(request)
                    )[0]
                for request in (sibling_a, other, sibling_b, turn_2):
                    await router.finish_request(request)
                await remote.stop()
                return servers, hashed

            servers, hashed = asyncio.run(drive())
            assert servers[2] != servers[1]
            assert servers[3] == servers[1]
            assert servers[4] == servers[1]
            assert hashed == ["t::sa:a", "t2", "t::sa:b"]


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


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    ("somaxconn", "expected"), [("4096\n", 4096), ("65535\n", 16384), (None, 16384)]
)
def test_coordinator_uds_backlog_is_capped_by_somaxconn(monkeypatch, somaxconn, expected):
    import io

    from tensorrt_llm.serve import coordinator_server

    real_open = open

    def fake_open(path, *args, **kwargs):
        if path != "/proc/sys/net/core/somaxconn":
            return real_open(path, *args, **kwargs)
        if somaxconn is None:
            raise FileNotFoundError(path)
        return io.StringIO(somaxconn)

    monkeypatch.setattr("builtins.open", fake_open)
    assert coordinator_server._effective_uds_backlog() == expected


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


def _context_cancellation_config(
    monkeypatch: pytest.MonkeyPatch,
    *,
    enabled: bool = True,
    ctx_urls: list[str] | None = None,
    router: str = "round_robin",
) -> DisaggServerConfig:
    """Select the existing protected-mode gates for coordinator tests."""
    monkeypatch.setenv("TRTLLM_ENABLE_KV_TRANSFER_LIFECYCLE_ENFORCEMENT", str(int(enabled)))
    monkeypatch.setenv("TRTLLM_DISAGG_NO_RETRY", "1")
    config = _make_config(ctx_urls or [], ["gen:8001"], router, "round_robin")
    config.schedule_style = "context_first"
    config.internal_request_auth_key = "secret"
    return config


@pytest.mark.asyncio
@pytest.mark.parametrize("capability", [None, False, 1, "true"])
async def test_protected_static_start_rejects_incompatible_context(
    monkeypatch: pytest.MonkeyPatch, capability: object
) -> None:
    """A reachable incompatible worker prevents protected static startup."""
    config = _context_cancellation_config(monkeypatch, ctx_urls=["ctx-good:8000", "ctx-bad:8000"])
    owner = DisaggCoordinatorService(config, Mock())
    info = {} if capability is None else {"supports_logical_context_cancellation": capability}
    owner.ctx_router._fetch_server_info = AsyncMock(
        side_effect=[{"supports_logical_context_cancellation": True}, info]
    )
    owner._wait_for_all_servers_ready = AsyncMock()
    with pytest.raises(RuntimeError, match="CTX server ctx-bad:8000"):
        await owner.start()
    assert owner.ctx_router.prepared_servers == {"ctx-good:8000"}
    owner._wait_for_all_servers_ready.assert_not_awaited()
    await owner.stop()


@pytest.mark.asyncio
async def test_protected_start_retries_transport_then_accepts_capable_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep the worker startup budget without requiring GEN capability."""
    config = _context_cancellation_config(monkeypatch, ctx_urls=["ctx:8000"])
    owner = DisaggCoordinatorService(
        config, Mock(), health_check_interval_secs=0, server_start_timeout_secs=1
    )
    owner.ctx_router._fetch_server_info = AsyncMock(
        side_effect=[
            RuntimeError("worker is still starting"),
            {"supports_logical_context_cancellation": True},
        ]
    )
    # GEN need not implement the context-side cancellation contract.
    owner.gen_router._fetch_server_info = AsyncMock(return_value={})
    owner._wait_for_all_servers_ready = AsyncMock()
    await owner.start()
    assert owner.ctx_router._fetch_server_info.await_count == 2
    assert owner.ctx_router.prepared_servers == {"ctx:8000"}
    assert (await owner.cluster_info())["context_cancellation_enabled"] is True
    owner._wait_for_all_servers_ready.assert_awaited_once()
    await owner.stop()


@pytest.mark.asyncio
async def test_protected_start_retries_non_success_server_info_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A transient HTTP response is not a definitive capability rejection."""
    config = _context_cancellation_config(monkeypatch, ctx_urls=["ctx:8000"])
    owner = DisaggCoordinatorService(
        config, Mock(), health_check_interval_secs=0, server_start_timeout_secs=1
    )
    unavailable = Mock()
    unavailable.raise_for_status.side_effect = aiohttp.ClientResponseError(
        Mock(real_url="http://ctx:8000/server_info"), (), status=503
    )
    unavailable.json = AsyncMock(return_value={"error": "starting"})
    ready = Mock()
    ready.json = AsyncMock(return_value={"supports_logical_context_cancellation": True})
    responses = []
    for response in (unavailable, ready):
        manager = AsyncMock()
        manager.__aenter__.return_value = response
        responses.append(manager)
    owner.ctx_router._session = Mock(get=Mock(side_effect=responses), close=AsyncMock())
    owner.gen_router._fetch_server_info = AsyncMock(return_value={})
    owner._wait_for_all_servers_ready = AsyncMock()
    try:
        await owner.start()
        assert owner.ctx_router._session.get.call_count == 2
        unavailable.json.assert_not_awaited()
        assert owner.ctx_router.prepared_servers == {"ctx:8000"}
    finally:
        await owner.stop()


@pytest.mark.asyncio
async def test_protected_start_uses_metadata_before_stale_static_preparation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Authoritative discovery can replace an unreachable static CTX address."""
    config = _context_cancellation_config(monkeypatch, ctx_urls=["stale:8000"])
    owner = DisaggCoordinatorService(config, Mock(), server_start_timeout_secs=1)
    metadata = {"trtllm/live": {"url": "live:8000", "server_role": "context"}}
    owner._metadata_server = metadata
    owner._metadata_config = SimpleNamespace(refresh_interval=1)
    owner.ctx_router._metadata_server = metadata
    owner.ctx_router.check_servers_health = AsyncMock(return_value=["live:8000"])
    owner.ctx_router._fetch_server_info = AsyncMock(
        return_value={"supports_logical_context_cancellation": True}
    )
    owner.gen_router._fetch_server_info = AsyncMock(return_value={})
    owner.ctx_router.start_server_monitoring = AsyncMock()
    owner.gen_router.start_server_monitoring = AsyncMock()
    owner._wait_for_all_servers_ready = AsyncMock()
    try:
        await owner.start()
        assert owner.ctx_router.servers == ["live:8000"]
        assert owner.ctx_router.prepared_servers == {"live:8000"}
        assert [call.args[0] for call in owner.ctx_router._fetch_server_info.await_args_list] == [
            "live:8000"
        ]
    finally:
        owner._metadata_server = None
        await owner.stop()


@pytest.mark.asyncio
async def test_protected_metadata_can_recover_after_empty_discovery() -> None:
    """An empty registry removes old routes and remains pollable for recovery."""
    metadata = {"trtllm/v1": {"url": "ctx:8000", "server_role": "context"}}
    router = RoundRobinRouter(
        ServerRole.CONTEXT, metadata_server=metadata, require_context_cancellation=True
    )
    router.check_servers_health = AsyncMock(side_effect=[["ctx:8000"], [], ["ctx:8000"]])
    router._fetch_server_info = AsyncMock(
        return_value={"supports_logical_context_cancellation": True}
    )
    try:
        await router.refresh_protected_context_servers()
        assert router.server_registration_id("ctx:8000") == "trtllm/v1"
        metadata.clear()
        await router.refresh_protected_context_servers()
        assert not router.servers and not router.prepared_servers
        metadata["trtllm/v2"] = {"url": "ctx:8000", "server_role": "context"}
        await router.refresh_protected_context_servers()
        assert router.servers == ["ctx:8000"]
        assert router.server_registration_id("ctx:8000") == "trtllm/v2"
    finally:
        await router.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("malformed", [[], None], ids=["array", "null"])
async def test_protected_metadata_skips_malformed_server_info(malformed: object) -> None:
    """A bad response excludes only its worker; a later valid probe can admit it."""
    servers = ["ctx-a:8000", "ctx-b:8000"]
    metadata = {f"trtllm/{server}": {"url": server, "server_role": "context"} for server in servers}
    router = RoundRobinRouter(
        ServerRole.CONTEXT, metadata_server=metadata, require_context_cancellation=True
    )
    router.check_servers_health = AsyncMock(return_value=servers)
    valid = {"supports_logical_context_cancellation": True}
    json_response = AsyncMock(side_effect=[malformed, valid, valid])
    manager = AsyncMock()
    manager.__aenter__.return_value = Mock(json=json_response)
    router._session = Mock(get=Mock(return_value=manager), close=AsyncMock())
    try:
        await router.refresh_protected_context_servers()
        assert router.prepared_servers == {servers[1]}
        assert router.servers == [servers[1]]
        await router.refresh_protected_context_servers()
        assert router.prepared_servers == set(servers)
        assert set(router.servers) == set(servers)
    finally:
        await router.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("remove_peer", [False, True], ids=["peer-joins", "peer-leaves"])
async def test_protected_kv_admission_survives_peer_update(remove_peer: bool) -> None:
    """Publishing or removing a peer must not discard a worker's pending hash state."""
    router = KvCacheAwareRouter(
        ServerRole.CONTEXT, tokens_per_block=4, require_context_cancellation=True
    )
    entered, release = asyncio.Event(), asyncio.Event()

    async def info(server: str, timeout: float) -> dict[str, bool | str]:
        if server == "pending":
            entered.set()
            await release.wait()
        return {
            "supports_logical_context_cancellation": True,
            "kv_cache_hash_algo": KV_CACHE_HASH_ALGO_V2,
        }

    router._fetch_server_info = AsyncMock(side_effect=info)
    if remove_peer:
        assert await router.add_server("peer")
    pending = asyncio.create_task(router.add_server("pending"))
    try:
        await asyncio.wait_for(entered.wait(), timeout=5)
        assert "pending" not in router.servers
        if remove_peer:
            await asyncio.wait_for(router.remove_server("peer"), timeout=5)
        else:
            assert await asyncio.wait_for(router.add_server("peer"), timeout=5)
        assert "pending" not in router.servers
        release.set()
        assert await asyncio.wait_for(pending, timeout=5)
        expected = {"pending"} if remove_peer else {"pending", "peer"}
        assert set(router.servers) == router.prepared_servers == expected
        assert router.routing_key_config() == {
            "tokens_per_block": 4,
            "kv_cache_hash_algo": KV_CACHE_HASH_ALGO_V2,
        }
    finally:
        release.set()
        pending.cancel()
        await asyncio.gather(pending, return_exceptions=True)
        await router.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("delegated", [False, True])
async def test_protected_replacement_keeps_original_load_owner(delegated: bool) -> None:
    """Completion of old work cannot decrement the replacement worker's load."""
    router = LoadBalancingRouter(ServerRole.CONTEXT, require_context_cancellation=True)
    router._fetch_server_info = AsyncMock(
        return_value={"supports_logical_context_cancellation": True}
    )
    first = CompletionRequest(model="m", prompt="first")
    second = CompletionRequest(model="m", prompt="second")
    try:
        assert await router.add_server("ctx:8000", registration_id="v1")
        old = router._server_state["ctx:8000"]
        if delegated:
            await router.get_next_server_by_key({}, req_id=1)
        else:
            await router.get_next_server(first)
        assert old._num_active_requests == 1
        assert await router.add_server("ctx:8000", registration_id="v2")
        new = router._server_state["ctx:8000"]
        assert new is not old
        if delegated:
            await router.get_next_server_by_key({}, req_id=2)
            await router.finish_request_by_id(1)
        else:
            await router.get_next_server(second)
            await router.finish_request(first)
        assert old._num_active_requests == 0
        assert new._num_active_requests == 1
        if delegated:
            await router.finish_request_by_id(2)
        else:
            await router.finish_request(second)
        assert new._num_active_requests == 0
        assert not router._request_load_state
    finally:
        await router.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("delegated", [False, True])
async def test_protected_kv_replacement_keeps_load_and_blocks_on_original_owner(
    monkeypatch: pytest.MonkeyPatch, delegated: bool
) -> None:
    """Old completions neither charge nor populate the replacement's cache."""
    router = KvCacheAwareRouter(
        ServerRole.CONTEXT, tokens_per_block=4, require_context_cancellation=True
    )
    router._fetch_server_info = AsyncMock(
        return_value={
            "supports_logical_context_cancellation": True,
            "tokens_per_block": 4,
            "kv_cache_hash_algo": KV_CACHE_HASH_ALGO_V1,
        }
    )
    # Exercise the real routing/cache bookkeeping without a remote event poll.
    monkeypatch.setattr(router, "_poll_server_on_finish", Mock())
    first = CompletionRequest(model="m", prompt=[10] * 9)
    second = CompletionRequest(model="m", prompt=[20] * 9)

    async def route(request: CompletionRequest, request_id: int) -> None:
        if delegated:
            await router.get_next_server_by_key(router.routing_key(request), req_id=request_id)
        else:
            await router.get_next_server(request)

    async def finish(request: CompletionRequest, request_id: int) -> None:
        if delegated:
            await router.finish_request_by_id(request_id)
        else:
            await router.finish_request(request)

    try:
        assert await router.add_server("ctx:8000", registration_id="v1")
        old = router._server_state["ctx:8000"]
        await route(first, 1)
        first_key = 1 if delegated else id(first)
        old_blocks, algo = router._pending_routed_blocks[first_key]
        assert old_blocks and old.num_active_requests() == 1
        assert await router.add_server("ctx:8000", registration_id="v2")
        new = router._server_state["ctx:8000"]
        assert new is not old
        await route(second, 2)
        second_key = 2 if delegated else id(second)
        new_blocks, new_algo = router._pending_routed_blocks[second_key]
        assert new_blocks and new_algo == algo
        assert set(old_blocks).isdisjoint(new_blocks)
        await finish(first, 1)
        assert old.num_active_requests() == 0 and new.num_active_requests() == 1
        assert not new._block_table(algo)
        await finish(second, 2)
        assert new.num_active_requests() == 0
        assert new._block_table(algo) == set(new_blocks)
        assert not router._request_load_state
        assert not router._req_routing_table
        assert not router._pending_routed_blocks
    finally:
        await router.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("delegated", [False, True])
async def test_kv_route_stash_failure_restores_load_and_identity_maps(
    monkeypatch: pytest.MonkeyPatch, delegated: bool
) -> None:
    """Failure before publishing routed blocks cannot leave a charged request."""
    router = KvCacheAwareRouter(ServerRole.CONTEXT, servers=["ctx:8000"], tokens_per_block=4)
    request = CompletionRequest(model="m", prompt=[10] * 9)
    monkeypatch.setattr(router, "_stash_routed_blocks", Mock(side_effect=RuntimeError("stash")))
    try:
        with pytest.raises(RuntimeError, match="stash"):
            if delegated:
                await router.get_next_server_by_key(router.routing_key(request), req_id=1)
            else:
                await router.get_next_server(request)
        assert router._server_state["ctx:8000"].num_active_requests() == 0
        assert not router._request_load_state
        assert not router._req_routing_table
        assert not router._pending_routed_blocks
    finally:
        await router.close()


@pytest.mark.asyncio
async def test_disabled_context_cancellation_accepts_legacy_workers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Disabled policy preserves workers with no capability metadata."""
    config = _context_cancellation_config(monkeypatch, enabled=False, ctx_urls=["ctx:8000"])
    owner = DisaggCoordinatorService(config, Mock())
    owner.ctx_router._fetch_server_info = AsyncMock(return_value={})
    owner.gen_router._fetch_server_info = AsyncMock(return_value={})
    owner._wait_for_all_servers_ready = AsyncMock()
    await owner.start()
    assert owner.ctx_router.prepared_servers == {"ctx:8000"}
    assert (await owner.cluster_info())["context_cancellation_enabled"] is False
    await owner.stop()


@pytest.mark.asyncio
async def test_protected_dynamic_registration_is_validated_before_routing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject bad joins and address replacements without disturbing healthy peers."""
    config = _context_cancellation_config(monkeypatch)
    owner = DisaggCoordinatorService(config, Mock())
    capable = {"ctx-good:8000": True, "ctx-bad:8000": False}

    async def info(server: str, timeout: float | None) -> dict[str, bool]:
        """Observe the externally queried worker before routing admission."""
        assert server not in owner.ctx_router.servers
        return {"supports_logical_context_cancellation": capable[server]}

    owner.ctx_router._fetch_server_info = AsyncMock(side_effect=info)
    good = WorkerInfo("good-v1", "ctx-good", 8000, ServerRole.CONTEXT)
    bad = WorkerInfo("bad-v1", "ctx-bad", 8000, ServerRole.CONTEXT)
    await owner._on_worker_event(good, WatchEventType.SET)
    await owner._on_worker_event(bad, WatchEventType.SET)
    assert owner.ctx_router.servers == ["ctx-good:8000"]
    request = CompletionRequest(model="m", prompt="hello")
    assert (await owner.ctx_router.get_next_server(request))[0] == "ctx-good:8000"
    # A heartbeat for the accepted registration does not redo preparation.
    await owner._on_worker_event(good, WatchEventType.SET)
    assert owner.ctx_router._fetch_server_info.await_count == 2
    # An observed process replacement at the same address must be checked again.
    capable["ctx-good:8000"] = False
    replacement = WorkerInfo("good-v2", "ctx-good", 8000, ServerRole.CONTEXT)
    await owner._on_worker_event(replacement, WatchEventType.SET)
    assert owner.ctx_router.servers == []
    assert owner.ctx_router.server_registration_id("ctx-good:8000") is None
    # Re-delivery after failed preparation cannot reuse the rejected registration.
    capable["ctx-good:8000"] = True
    await owner._on_worker_event(replacement, WatchEventType.SET)
    await owner._on_worker_event(good, WatchEventType.DELETE)
    assert owner.ctx_router.servers == ["ctx-good:8000"]
    await owner._on_worker_event(replacement, WatchEventType.DELETE)
    assert not owner.ctx_router.servers
    await owner.stop()


@pytest.mark.asyncio
@pytest.mark.parametrize("initial_replay", [True, False])
async def test_protected_cluster_watch_survives_incompatible_kv_registration(
    monkeypatch: pytest.MonkeyPatch, initial_replay: bool
) -> None:
    """Reject one KV layout without dropping healthy replay or live registrations."""
    config = _context_cancellation_config(monkeypatch, router="kv_cache_aware")
    config.ctx_router_config.args["tokens_per_block"] = 32
    owner = DisaggCoordinatorService(config, Mock())
    cluster_config = DisaggClusterConfig(
        cluster_uri="http://storage", cluster_name="test", minimal_instances=MinimalInstances()
    )
    events = WatchEventQueue(["/trtllm-disagg/test/workers"])
    storage = Mock()
    storage.watch = AsyncMock(return_value=events)
    storage.unwatch = AsyncMock()
    storage.stop = AsyncMock()
    manager = DisaggClusterManager(cluster_config, storage)
    observed = {name: asyncio.Event() for name in ("good", "later")}

    def registration(name: str) -> WatchEvent:
        """Build a real worker-registration event for the watch queue."""
        return WatchEvent(
            StorageItem(
                key=f"{manager.worker_key_prefix}/{name}",
                value=json.dumps(
                    {
                        "worker_id": name,
                        "host": name,
                        "port": 8000,
                        "role": ServerRole.CONTEXT.value,
                    }
                ),
            ),
            WatchEventType.SET,
        )

    first_events = [registration("bad"), registration("good")]
    storage.get_prefix = AsyncMock(
        return_value={event.storage_item.key: event.storage_item.value for event in first_events}
    )

    async def info(server: str, _timeout: float | None) -> dict:
        """Reject the first layout while exposing compatible later registrations."""
        return {
            "supports_logical_context_cancellation": True,
            "tokens_per_block": 48 if server == "bad:8000" else 32,
            "kv_cache_hash_algo": KV_CACHE_HASH_ALGO_V1,
        }

    async def on_event(worker: WorkerInfo, event_type: WatchEventType) -> None:
        """Observe completion of the real coordinator admission path."""
        await owner._on_worker_event(worker, event_type)
        if worker.worker_id in observed:
            observed[worker.worker_id].set()

    owner.ctx_router._fetch_server_info = AsyncMock(side_effect=info)
    watch_task = None
    try:
        await manager.watch_workers(get_existing_first=initial_replay, on_event=on_event)
        watch_task = manager._watch_task
        if not initial_replay:
            await events.add_events(first_events)
        await asyncio.wait_for(observed["good"].wait(), timeout=1)
        # No new event for the rejected registration is needed to watch later joins.
        await events.add_events([registration("later")])
        await asyncio.wait_for(observed["later"].wait(), timeout=1)
        assert set(owner.ctx_router.servers) == {"good:8000", "later:8000"}
        assert owner.ctx_router.prepared_servers == {"good:8000", "later:8000"}
        assert owner.ctx_router.server_registration_id("bad:8000") is None
        assert "bad:8000" not in owner.ctx_router._server_state
        assert not owner._context_admission_tasks
        assert watch_task is not None and not watch_task.done()
    finally:
        await manager.stop()
        if watch_task is not None:
            await asyncio.wait_for(asyncio.gather(watch_task, return_exceptions=True), timeout=1)
        await owner.stop()


@pytest.mark.asyncio
async def test_protected_cluster_replacement_retries_without_another_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A recovered registration becomes routable without another watch event."""
    owner = DisaggCoordinatorService(
        _context_cancellation_config(monkeypatch, router="load_balancing"),
        Mock(),
        health_check_interval_secs=0,
    )
    entered, release = asyncio.Event(), asyncio.Event()
    calls = 0

    async def info(_server: str, _timeout: float) -> dict[str, bool]:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("replacement is starting")
        if calls == 3:
            entered.set()
            await release.wait()
        return {"supports_logical_context_cancellation": True}

    owner.ctx_router._fetch_server_info = AsyncMock(side_effect=info)
    first = WorkerInfo("v1", "ctx", 8000, ServerRole.CONTEXT)
    replacement = WorkerInfo("v2", "ctx", 8000, ServerRole.CONTEXT)
    try:
        await owner._on_worker_event(first, WatchEventType.SET)
        await owner._on_worker_event(replacement, WatchEventType.SET)
        await asyncio.wait_for(entered.wait(), timeout=1)
        retry = owner._context_admission_tasks["ctx:8000"]
        assert not owner.ctx_router.servers
        assert owner.ctx_router.server_registration_id("ctx:8000") is None
        # An obsolete DELETE must not cancel the not-yet-admitted replacement.
        await owner._on_worker_event(first, WatchEventType.DELETE)
        assert owner._context_admission_tasks["ctx:8000"] is retry
        assert not retry.done()
        release.set()
        await asyncio.wait_for(asyncio.shield(retry), timeout=1)
        assert owner.ctx_router.servers == ["ctx:8000"]
        assert owner.ctx_router.prepared_servers == {"ctx:8000"}
        assert owner.ctx_router.server_registration_id("ctx:8000") == "v2"
        assert calls == 3
        assert not owner._context_admission_tasks
    finally:
        release.set()
        await owner.stop()


@pytest.mark.asyncio
async def test_protected_cluster_capability_rejection_does_not_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A successful incompatible probe is final for this delivery of SET."""
    owner = DisaggCoordinatorService(
        _context_cancellation_config(monkeypatch), Mock(), health_check_interval_secs=0
    )
    owner.ctx_router._fetch_server_info = AsyncMock(
        side_effect=[{"supports_logical_context_cancellation": True}, {}]
    )
    try:
        await owner._on_worker_event(
            WorkerInfo("v1", "ctx", 8000, ServerRole.CONTEXT), WatchEventType.SET
        )
        await owner._on_worker_event(
            WorkerInfo("v2", "ctx", 8000, ServerRole.CONTEXT), WatchEventType.SET
        )
        assert not owner.ctx_router.servers
        assert not owner._context_admission_tasks
        assert owner.ctx_router._fetch_server_info.await_count == 2
    finally:
        await owner.stop()


@pytest.mark.asyncio
@pytest.mark.parametrize("action", ["delete", "replace", "stop"])
async def test_protected_cluster_probe_is_drained_before_incarnation_changes(
    monkeypatch: pytest.MonkeyPatch, action: str
) -> None:
    """No pending probe can resurrect a deleted or superseded worker."""
    owner = DisaggCoordinatorService(
        _context_cancellation_config(monkeypatch, router="load_balancing"),
        Mock(),
        health_check_interval_secs=0,
    )
    entered, cancelled = asyncio.Event(), asyncio.Event()
    calls = 0

    async def info(_server: str, _timeout: float) -> dict[str, bool]:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("replacement is starting")
        if calls == 3:
            entered.set()
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()
        return {"supports_logical_context_cancellation": True}

    owner.ctx_router._fetch_server_info = AsyncMock(side_effect=info)
    second = WorkerInfo("v2", "ctx", 8000, ServerRole.CONTEXT)
    third = WorkerInfo("v3", "ctx", 8000, ServerRole.CONTEXT)
    try:
        await owner._on_worker_event(
            WorkerInfo("v1", "ctx", 8000, ServerRole.CONTEXT), WatchEventType.SET
        )
        await owner._on_worker_event(second, WatchEventType.SET)
        await asyncio.wait_for(entered.wait(), timeout=1)
        retry = owner._context_admission_tasks["ctx:8000"]
        if action == "delete":
            await owner._on_worker_event(second, WatchEventType.DELETE)
        elif action == "replace":
            await owner._on_worker_event(third, WatchEventType.SET)
        else:
            await owner.stop()
            await owner._on_worker_event(third, WatchEventType.SET)
        assert cancelled.is_set() and retry.cancelled()
        assert not owner._context_admission_tasks
        if action == "replace":
            assert calls == 4
            assert owner.ctx_router.server_registration_id("ctx:8000") == "v3"
        else:
            assert calls == 3
            assert not owner._context_worker_registrations
            assert not owner.ctx_router.servers
            assert not owner.ctx_router._server_state
    finally:
        await owner.stop()


@pytest.mark.asyncio
async def test_protected_metadata_join_is_not_visible_before_capability_validation() -> None:
    """Metadata publication cannot race ahead of the real preparation path."""
    metadata = {
        "trtllm/good": {"url": "good:8000", "server_role": "context"},
        "trtllm/bad": {"url": "bad:8000", "server_role": "context"},
    }
    router = RoundRobinRouter(
        ServerRole.CONTEXT, metadata_server=metadata, require_context_cancellation=True
    )
    router.check_servers_health = AsyncMock(return_value=["bad:8000", "good:8000"])
    probing_bad = asyncio.Event()
    finish_bad = asyncio.Event()
    refreshed = asyncio.Event()
    good_admitted = asyncio.Event()
    real_refresh = router.refresh_protected_context_servers
    publish = router._on_servers_updated

    def on_update(old: list[str], new: list[str]) -> None:
        publish(old, new)
        if "good:8000" in new:
            good_admitted.set()

    router._on_servers_updated = on_update

    async def refresh() -> None:
        """Signal only after the real metadata refresh finishes its routing decision."""
        await real_refresh()
        refreshed.set()

    router.refresh_protected_context_servers = refresh

    async def info(server: str, timeout: float | None) -> dict[str, bool]:
        """Observe the externally queried worker before routing admission."""
        assert server not in router.servers
        if server == "bad:8000":
            probing_bad.set()
            await finish_bad.wait()
            return {}
        return {"supports_logical_context_cancellation": True}

    router._fetch_server_info = AsyncMock(side_effect=info)
    monitor = asyncio.create_task(router._monitor_servers(60))
    try:
        await asyncio.wait_for(probing_bad.wait(), timeout=1)
        await asyncio.wait_for(good_admitted.wait(), timeout=1)
        request = CompletionRequest(model="m", prompt="hello")
        assert router.servers == ["good:8000"]
        assert (await router.get_next_server(request))[0] == "good:8000"
        finish_bad.set()
        await asyncio.wait_for(refreshed.wait(), timeout=1)
        assert not monitor.done()
        assert router.servers == ["good:8000"]
        assert router.prepared_servers == {"good:8000"}
    finally:
        finish_bad.set()
        monitor.cancel()
        await asyncio.gather(monitor, return_exceptions=True)
        await router.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("enabled,published", [(True, None), (True, False), (False, True)])
@pytest.mark.parametrize("router", ["round_robin", "load_balancing"])
async def test_fleet_cancellation_mode_mismatch_precedes_routing_sync(
    monkeypatch: pytest.MonkeyPatch, enabled: bool, published: bool | None, router: str
) -> None:
    """Both fleet router types reject mode drift before publishing routing state."""
    config = _context_cancellation_config(monkeypatch, enabled=enabled, router=router)
    client = CoordinatorClient("http://coordinator", config)
    info = {"is_ready": True, "server_lists": {"context": ["ctx:8000"], "generation": []}}
    if published is not None:
        info["context_cancellation_enabled"] = published
    client._await_coordinator = AsyncMock(return_value=info)
    client._sync_router_server_lists = AsyncMock()
    with pytest.raises(RuntimeError, match="mode differs"):
        await client.start()
    client._sync_router_server_lists.assert_not_awaited()
    assert not client._is_ready
    # A changed mode in a later coordinator snapshot also closes readiness.
    client._is_ready = True
    client.cluster_info = AsyncMock(return_value=info)
    assert await client._refresh_coordinator_state() is False
    client._sync_router_server_lists.assert_not_awaited()
    await client.stop()


@pytest.mark.asyncio
@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize(
    "router,capable",
    [
        ("round_robin", True),
        ("round_robin", False),
        ("load_balancing", True),
        ("load_balancing", False),
    ],
)
async def test_fleet_matching_cancellation_mode_prepares_before_ready(
    monkeypatch: pytest.MonkeyPatch, enabled: bool, router: str, capable: bool
) -> None:
    """Matching fleet policy permits only a fully applied coordinator snapshot."""
    config = _context_cancellation_config(monkeypatch, enabled=enabled, router=router)
    client = CoordinatorClient("http://coordinator", config)
    local = getattr(client.ctx_router, "_local", client.ctx_router)
    local._fetch_server_info = AsyncMock(
        return_value={"supports_logical_context_cancellation": True} if capable else {}
    )
    info = {"is_ready": True, "server_lists": {"context": ["ctx:8000"], "generation": []}}
    # Missing metadata remains compatible only for a disabled frontend.
    if enabled:
        info["context_cancellation_enabled"] = True
        info["context_admissions"] = {"ctx:8000": "first"}
    try:
        if enabled and not capable:
            with pytest.raises(RuntimeError, match="does not advertise supported"):
                await client._apply_cluster_info(info)
            assert not client._is_ready
            assert not client.ctx_router.servers
            assert not client.ctx_router.prepared_servers
            local._fetch_server_info.assert_awaited_once()
        else:
            await client._apply_cluster_info(info)
            assert client._is_ready
            assert client.ctx_router.servers == ["ctx:8000"]
            if enabled:
                await client._apply_cluster_info(info)
                local._fetch_server_info.assert_awaited_once()
                info["context_admissions"]["ctx:8000"] = "replacement"
                await client._apply_cluster_info(info)
                assert local._fetch_server_info.await_count == 2
                del info["context_admissions"]
                with pytest.raises(RuntimeError, match="did not publish CTX admissions"):
                    await client._apply_cluster_info(info)
                assert not client._is_ready
    finally:
        await client.stop()


@pytest.mark.asyncio
@pytest.mark.parametrize("same_url", [False, True], ids=["new-address", "replacement"])
async def test_delegated_select_rejects_unverified_grant(
    monkeypatch: pytest.MonkeyPatch, same_url: bool
) -> None:
    """Race the signer's pool refresh with a real /select and a guarded /finish."""
    config = _context_cancellation_config(monkeypatch, router="load_balancing")
    owner = DisaggCoordinatorService(config, _client_factory)
    router = owner.ctx_router
    router._fetch_server_info = AsyncMock(
        return_value={"supports_logical_context_cancellation": True}
    )
    handlers = CoordinatorServer(owner)
    old = "ctx-old:8000"
    selected = old if same_url else "ctx-new:8000"
    assert await router.add_server(old, registration_id="v1")
    paths = []

    async def dispatch(request: web.Request) -> web.Response:
        """Use the real MessagePack endpoints and interpose a superseding grant on rejection."""
        paths.append(request.path)
        if request.path == "/finish" and same_url:
            # A transport retry re-places this ID after the failed grant. Its
            # counter and timer must outlive the delayed conditional finish.
            assert await router.add_server(selected, registration_id="v3")
            await owner.select("context", {}, 41, None)
        handler = handlers.select if request.path == "/select" else handlers.finish
        response = await handler(SimpleNamespace(body=request.read))
        return web.Response(body=response.body, status=response.status_code)

    app = web.Application()
    app.router.add_post("/select", dispatch)
    app.router.add_post("/finish", dispatch)
    async with TestServer(app) as http:
        frontend = CoordinatorClient(str(http.make_url("")).rstrip("/"), config)
        edge = frontend.ctx_router
        edge._local._fetch_server_info = AsyncMock(
            return_value={"supports_logical_context_cancellation": True}
        )
        request = CompletionRequest(model="m", prompt="one")
        try:
            await edge.sync_servers([old], context_admissions=router.context_admissions())
            if not same_url:
                await router.remove_server(old)
            assert await router.add_server(selected, registration_id="v2")
            rejected_owner = router._server_state[selected]
            with pytest.raises(ServerPreparationPending) as failure:
                await edge.get_next_server(request, req_id=41)
            assert paths == ["/select", "/finish"]
            assert rejected_owner._num_active_requests == 0
            assert (("context", 41) in owner._reservation_tasks) is same_url
            assert router._server_state[selected]._num_active_requests == int(same_url)
            with pytest.raises(HTTPException) as http_error:
                OpenAIDisaggServer._handle_exception(
                    SimpleNamespace(_perf_metrics_collector=Mock()), failure.value
                )
            assert http_error.value.status_code == 503

            await owner.finish("context", 41, False)
            await edge.sync_servers([selected], context_admissions=router.context_admissions())
            server, _ = await edge.get_next_server(request, req_id=42)
            assert server == selected
            assert paths == ["/select", "/finish", "/select"]
            await owner.finish("context", 42)
            assert (
                not owner._reservation_tasks
                and router._server_state[selected]._num_active_requests == 0
            )
        finally:
            await frontend.stop()
            await owner.stop()


@pytest.mark.asyncio
async def test_delegated_probe_revokes_retiring_grants_before_waiting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A delayed old select or probe cannot restore an invalidated CTX incarnation."""
    config = _context_cancellation_config(monkeypatch, router="load_balancing")
    frontend = CoordinatorClient("http://coordinator", config)
    edge = frontend.ctx_router
    server = "ctx:8000"
    edge._local._fetch_server_info = AsyncMock(
        return_value={"supports_logical_context_cancellation": True}
    )
    await edge.sync_servers([server], context_admissions={server: "old"})
    received = asyncio.Event()
    release = asyncio.Event()

    async def delayed_probe(*args: object) -> dict:
        received.set()
        await release.wait()
        return {"supports_logical_context_cancellation": True}

    edge._local._fetch_server_info = delayed_probe
    ongoing = asyncio.create_task(edge.sync_servers([server], context_admissions={server: "new"}))
    try:
        await asyncio.wait_for(received.wait(), 2)
        edge._post_select = AsyncMock(
            return_value={"server": server, "info": {"context_admission": "old"}}
        )
        edge._finish_async = AsyncMock()
        with pytest.raises(ServerPreparationPending):
            await edge.get_next_server(CompletionRequest(model="m", prompt="one"), req_id=41)
        edge._finish_async.assert_awaited_once_with(41, False, context_admission="old")
        edge._post_select.return_value = {"server": server, "info": {}}
        with pytest.raises(ServerPreparationPending, match="did not identify"):
            await edge.get_next_server(CompletionRequest(model="m", prompt="two"), req_id=42)
        # An untagged reply does not authorize an unconditional finish of this ID.
        edge._finish_async.assert_awaited_once()
        await edge.sync_servers([], context_admissions={})
        release.set()
        with pytest.raises(ServerPreparationPending, match="superseded"):
            await ongoing
        assert edge._local.server_registration_id(server) is None
    finally:
        release.set()
        await asyncio.gather(ongoing, return_exceptions=True)
        await frontend.stop()


@pytest.mark.asyncio
async def test_rejecting_old_conversation_preserves_replacement_load() -> None:
    """A failed old grant cannot debit new work on a re-admitted conversation worker."""
    router = ConversationRouter(ServerRole.CONTEXT, require_context_cancellation=True)
    router._fetch_server_info = AsyncMock(
        return_value={"supports_logical_context_cancellation": True}
    )
    server = "ctx:8000"
    try:
        assert await router.add_server(server, registration_id="old")
        await router.get_next_server_by_key(None, req_id=41)
        assert await router.add_server(server, registration_id="new")
        await router.get_next_server_by_key(None, req_id=42)
        await router.finish_request_by_id(41, False)
        assert router._server_content_load == {server: 1}
        await router.finish_request_by_id(42, True)
        assert router._server_content_load == {server: 0}
        assert not router._coord_pending
    finally:
        await router.close()


@pytest.mark.parametrize("coordinator_type", [DisaggCoordinatorService, CoordinatorClient])
@pytest.mark.parametrize("missing", ["no_retry", "auth"])
def test_coordinator_enforces_existing_context_cancellation_prerequisites(
    monkeypatch: pytest.MonkeyPatch,
    coordinator_type: type[DisaggCoordinatorService] | type[CoordinatorClient],
    missing: str,
) -> None:
    """Owner-only processes cannot bypass existing no-retry or auth prerequisites."""
    config = _context_cancellation_config(monkeypatch)
    if missing == "no_retry":
        monkeypatch.delenv("TRTLLM_DISAGG_NO_RETRY")
    else:
        config.internal_request_auth_key = None
    with pytest.raises(ValueError, match="Context cancellation requires"):
        if coordinator_type is CoordinatorClient:
            coordinator_type("http://coordinator", config)
        else:
            coordinator_type(config, Mock())


@pytest.mark.asyncio
@pytest.mark.parametrize("discovered", [False, True], ids=["static", "metadata"])
async def test_protected_transport_startup_wait_is_bounded(
    monkeypatch: pytest.MonkeyPatch, discovered: bool
) -> None:
    """Name unadmitted addresses even if the deadline interrupts the first health probe."""
    config = _context_cancellation_config(monkeypatch, ctx_urls=[] if discovered else ["ctx:8000"])
    owner = DisaggCoordinatorService(
        config, Mock(), health_check_interval_secs=1, server_start_timeout_secs=0.01
    )
    owner.ctx_router._fetch_server_info = AsyncMock(side_effect=RuntimeError("not ready"))
    if discovered:
        entries = {"trtllm/ctx": {"url": "ctx:8000", "server_role": "context"}}
        owner._metadata_server = owner.ctx_router._metadata_server = entries
        owner._metadata_config = SimpleNamespace(refresh_interval=1)

        async def pending(_url: str) -> bool:
            await asyncio.Event().wait()
            return False

        owner.ctx_router._check_server_health = AsyncMock(side_effect=pending)
    try:
        with pytest.raises(RuntimeError, match="Timed out.*CTX servers.*ctx:8000"):
            await asyncio.wait_for(owner.start(), timeout=2)
        assert not owner.ctx_router.prepared_servers
    finally:
        await owner.stop()


@pytest.mark.asyncio
async def test_static_context_address_is_revalidated_on_registration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A dynamic registration does not inherit a static address's cached support."""
    config = _context_cancellation_config(monkeypatch, ctx_urls=["ctx:8000"])
    owner = DisaggCoordinatorService(config, Mock())
    owner.ctx_router._fetch_server_info = AsyncMock(
        return_value={"supports_logical_context_cancellation": True}
    )
    await owner.ctx_router.prepare_servers()
    owner.ctx_router._fetch_server_info.return_value = {}
    await owner._on_worker_event(
        WorkerInfo("new-process", "ctx", 8000, ServerRole.CONTEXT), WatchEventType.SET
    )
    assert owner.ctx_router.servers == []
    assert owner.ctx_router.server_registration_id("ctx:8000") is None
    await owner.stop()


@pytest.mark.parametrize("coordinator_type", [DisaggCoordinatorService, CoordinatorClient])
def test_generation_first_does_not_require_context_cancellation_capability(
    monkeypatch: pytest.MonkeyPatch,
    coordinator_type: type[DisaggCoordinatorService] | type[CoordinatorClient],
) -> None:
    """The context-first frontend contract does not change generation-first gating."""
    config = _context_cancellation_config(monkeypatch)
    config.schedule_style = "generation_first"
    config.internal_request_auth_key = None
    monkeypatch.delenv("TRTLLM_DISAGG_NO_RETRY")
    owner = (
        coordinator_type("http://coordinator", config)
        if coordinator_type is CoordinatorClient
        else coordinator_type(config, Mock())
    )
    assert not owner._context_cancellation_enabled
    assert not owner.ctx_router._require_context_cancellation
    assert "require_context_cancellation" not in config.ctx_router_config.args


@pytest.mark.asyncio
async def test_protected_start_does_not_retry_permanent_router_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """KV layout incompatibility remains an immediate startup failure."""
    config = _context_cancellation_config(
        monkeypatch, ctx_urls=["ctx:8000"], router="kv_cache_aware"
    )
    config.ctx_router_config.args["tokens_per_block"] = 32
    owner = DisaggCoordinatorService(config, Mock())
    owner.ctx_router._fetch_server_info = AsyncMock(
        return_value={"supports_logical_context_cancellation": True, "tokens_per_block": 48}
    )
    try:
        with pytest.raises(RuntimeError, match="tokens_per_block mismatch"):
            await asyncio.wait_for(owner.start(), timeout=1)
        owner.ctx_router._fetch_server_info.assert_awaited_once()
    finally:
        await owner.ctx_router.close()
        await owner.stop()


@pytest.mark.asyncio
async def test_protected_start_keeps_unready_metadata_until_health_recovers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Failed startup probes cannot erase the only registration of a starting worker."""
    config = _context_cancellation_config(monkeypatch)
    owner = DisaggCoordinatorService(
        config, Mock(), health_check_interval_secs=0, server_start_timeout_secs=1
    )
    entries = {"trtllm/ctx": {"url": "ctx:8000", "server_role": "context"}}
    metadata = Mock()
    metadata.keys.side_effect = lambda: list(entries)
    metadata.get.side_effect = entries.get
    metadata.remove.side_effect = lambda key: entries.pop(key)
    owner._metadata_server = metadata
    owner._metadata_config = SimpleNamespace(refresh_interval=1)
    owner.ctx_router._metadata_server = metadata
    owner.ctx_router._check_server_health = AsyncMock(side_effect=[False, False, True])
    owner.ctx_router._fetch_server_info = AsyncMock(
        return_value={"supports_logical_context_cancellation": True}
    )
    try:
        await owner._prepare_context_servers()
        metadata.remove.assert_not_called()
        assert owner.ctx_router._check_server_health.await_count == 3
        assert owner.ctx_router.servers == ["ctx:8000"]
        assert owner.ctx_router.prepared_servers == {"ctx:8000"}
        owner.ctx_router._fetch_server_info.assert_awaited_once()
    finally:
        owner._metadata_server = None
        await owner.stop()


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["replace", "add-context", "add-generation"])
async def test_protected_metadata_admission_rechecks_probed_registration(change: str) -> None:
    """A stale probe cannot admit a changed registration set or a conflicting role."""
    metadata = {"trtllm/v1": {"url": "ctx", "server_role": "context"}}
    router = RoundRobinRouter(
        ServerRole.CONTEXT, metadata_server=metadata, require_context_cancellation=True
    )
    router._check_server_health = AsyncMock(return_value=True)

    entered, release = asyncio.Event(), asyncio.Event()

    async def info(_server: str, _timeout: float) -> dict:
        entered.set()
        await release.wait()
        return {"supports_logical_context_cancellation": True}

    router._fetch_server_info = AsyncMock(side_effect=info)
    refresh = asyncio.create_task(router.refresh_protected_context_servers(startup=True))
    try:
        await asyncio.wait_for(entered.wait(), 1)
        if change == "replace":
            del metadata["trtllm/v1"]
        role = "generation" if change == "add-generation" else "context"
        metadata["trtllm/v2"] = {"url": "ctx", "server_role": role}
        release.set()
        await asyncio.wait_for(refresh, 1)
        assert not router.servers and not router.prepared_servers
        assert not router.server_infos()
        router._fetch_server_info = AsyncMock(return_value={})
        if change == "add-generation":
            await router.refresh_protected_context_servers(startup=True)
            router._fetch_server_info.assert_not_awaited()
        else:
            with pytest.raises(RuntimeError, match="does not advertise"):
                await router.refresh_protected_context_servers(startup=True)
            router._fetch_server_info.assert_awaited_once()
        assert not router.servers and not router.prepared_servers
    finally:
        release.set()
        await asyncio.gather(refresh, return_exceptions=True)
        await router.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("replace", [False, True], ids=["removed", "replaced"])
async def test_protected_preparation_discards_retired_worker_reply(replace: bool) -> None:
    """An old /server_info response must not republish readiness or replace current metadata."""
    router = RoundRobinRouter(ServerRole.CONTEXT, ["ctx"], require_context_cancellation=True)
    started, resume = asyncio.Event(), asyncio.Event()

    async def info(_server: str, _timeout: float) -> dict:
        if not started.is_set():
            started.set()
            await resume.wait()
            return {"supports_logical_context_cancellation": True, "version": "retired"}
        return {"supports_logical_context_cancellation": True, "version": "replacement"}

    router._fetch_server_info = AsyncMock(side_effect=info)
    old = asyncio.create_task(router.prepare_servers(["ctx"]))
    try:
        await asyncio.wait_for(started.wait(), timeout=1)
        await router.remove_server("ctx")
        if replace:
            assert await router.add_server("ctx", registration_id="v2")
        resume.set()
        await asyncio.wait_for(old, timeout=1)
        assert router.prepared_servers == ({"ctx"} if replace else set())
        assert [entry["version"] for entry in router.server_infos().values()] == (
            ["replacement"] if replace else []
        )
    finally:
        resume.set()
        await asyncio.gather(old, return_exceptions=True)
        await router.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("during", ["context", "generation"])
async def test_protected_readiness_rechecks_current_registration(
    monkeypatch: pytest.MonkeyPatch, during: str
) -> None:
    """An obsolete health result, empty router, or pending probe cannot complete startup."""
    owner = DisaggCoordinatorService(
        _context_cancellation_config(monkeypatch), Mock(), health_check_interval_secs=0
    )
    router = owner.ctx_router
    server = "ctx:8000"
    router._fetch_server_info = AsyncMock(
        return_value={"supports_logical_context_cancellation": True}
    )
    await router.add_server(server, registration_id="v1")
    seen: list[str | None] = []

    async def check_context() -> tuple[list[str], list[str]]:
        targets = list(router.servers)
        seen.append(router.server_registration_id(server))
        if len(seen) == 1 and during == "context":
            await router.remove_server(server)
        if seen[-1] == "v2" and seen.count("v2") == 1:
            raise ServerPreparationPending("capability temporarily unavailable")
        # OpenAIHttpClient.check_ready prepares the addresses its health check returned.
        await router.prepare_servers(targets)
        return targets, []

    async def check_generation() -> tuple[list[str], list[str]]:
        if len(seen) == 1 and during == "generation":
            await router.add_server(server, registration_id="v2")
        if seen[-1] is None:
            assert not await owner.is_ready()
            await router.add_server(server, registration_id="v2")
        return ["gen:8001"], []

    owner._ctx_client = Mock(check_ready=check_context, shutdown=AsyncMock())
    owner._gen_client = Mock(check_ready=check_generation, shutdown=AsyncMock())
    try:
        await asyncio.wait_for(owner._wait_for_all_servers_ready(), timeout=1)
        assert seen == (["v1", None, "v2", "v2"] if during == "context" else ["v1", "v2", "v2"])
        assert await owner.is_ready()
        assert router.prepared_servers == {server}
        assert router._fetch_server_info.await_count == 2
    finally:
        await owner.stop()


@pytest.mark.asyncio
async def test_protected_metadata_probes_only_distinct_unambiguous_context_urls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Bound startup concurrency; neither duplicate ordering nor other roles reset admission."""
    # Three eligible addresses must take two waves at this limit.
    monkeypatch.setattr(router_module, "_MAX_METADATA_HEALTH_PROBES", 2)
    entries = {
        f"trtllm/{key}": {"url": url, "server_role": role}
        for key, url, role in (
            ("a1", "a", "context"),
            ("a2", "a", "context"),
            ("b", "b", "context"),
            ("c", "c", "context"),
            ("g", "gen", "generation"),
            ("x1", "conflict", "context"),
            ("x2", "conflict", "generation"),
        )
    }
    registry = Mock(wraps=entries)
    router = LoadBalancingRouter(
        ServerRole.CONTEXT, metadata_server=registry, require_context_cancellation=True
    )
    event_loop_thread = threading.get_ident()

    def read_keys() -> list[str]:
        assert threading.get_ident() != event_loop_thread
        assert not router._lock.locked()
        return list(entries)

    registry.keys.side_effect = read_keys
    wave, release = asyncio.Event(), asyncio.Event()
    active = 0
    observed: list[str] = []

    async def health(url: str) -> bool:
        nonlocal active
        active += 1
        try:
            observed.append(url)
            assert active <= 2
            if active == 2:
                wave.set()
            await release.wait()
            return True
        finally:
            active -= 1

    router._check_server_health = AsyncMock(side_effect=health)
    router._fetch_server_info = AsyncMock(
        return_value={"supports_logical_context_cancellation": True}
    )
    refresh = asyncio.create_task(router.refresh_protected_context_servers(startup=True))
    try:
        await asyncio.wait_for(wave.wait(), timeout=1)
        release.set()
        await asyncio.wait_for(refresh, timeout=1)
        assert sorted(observed) == ["a", "b", "c"]
        assert router.prepared_servers == {"a", "b", "c"}
        # Discovery, one post-health snapshot, one post-capability snapshot for all three.
        assert registry.keys.call_count == 3
        assert registry.get.call_count == 3 * len(entries)
        state = router._server_state["a"]
        entries["trtllm/a1"] = entries.pop("trtllm/a1")
        registry.keys.reset_mock()
        await router.refresh_protected_context_servers(startup=True)
        assert registry.keys.call_count <= 2  # Shared snapshots, not one scan per admitted URL.
        assert router._server_state["a"] is state
        assert router._fetch_server_info.await_count == 3

        entries["trtllm/a3"] = dict(entries["trtllm/a1"])
        router._fetch_server_info.return_value = {}
        await router.refresh_protected_context_servers()
        assert router.prepared_servers == {"b", "c"}
        assert router._fetch_server_info.await_count == 4
    finally:
        release.set()
        refresh.cancel()
        await asyncio.gather(refresh, return_exceptions=True)
        await router.close()


@pytest.mark.asyncio
async def test_protected_metadata_layout_rejection_keeps_discovery_running() -> None:
    """A rejected KV layout cannot prevent a later compatible worker from joining."""
    metadata = {"trtllm/bad": {"url": "bad:8000", "server_role": "context"}}
    router = KvCacheAwareRouter(
        ServerRole.CONTEXT,
        metadata_server=metadata,
        tokens_per_block=32,
        require_context_cancellation=True,
    )

    async def healthy(*args: object, **kwargs: object) -> list[str]:
        """Return the unchanged registry's current health without deleting it."""
        return [entry["url"] for entry in metadata.values()]

    router.check_servers_health = AsyncMock(side_effect=healthy)
    router._fetch_server_info = AsyncMock(
        side_effect=lambda server, _: {
            "supports_logical_context_cancellation": True,
            "tokens_per_block": 48 if server == "bad:8000" else 32,
        }
    )
    refreshed = router.refresh_protected_context_servers
    complete = asyncio.Event()
    release = asyncio.Event()
    passes = 0

    async def observe_refresh() -> None:
        """Advance discovery only after the previous real refresh has completed."""
        nonlocal passes
        await refreshed()
        passes += 1
        assert "bad:8000" not in router.servers
        if passes == 1:
            metadata["trtllm/good"] = {"url": "good:8000", "server_role": "context"}
        else:
            complete.set()
            await release.wait()

    monitor = None
    try:
        with pytest.raises(RuntimeError, match="tokens_per_block mismatch"):
            await refreshed(startup=True)
        router.refresh_protected_context_servers = observe_refresh
        monitor = asyncio.create_task(router._monitor_servers(0))
        await asyncio.wait_for(complete.wait(), timeout=1)
        assert not monitor.done()
        assert passes == 2
        assert router.servers == ["good:8000"]
        assert router.prepared_servers == {"good:8000"}
        assert "bad:8000" not in router._server_state
    finally:
        release.set()
        if monitor is not None:
            monitor.cancel()
            await asyncio.gather(monitor, return_exceptions=True)
        await router.close()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v", "-s"]))
