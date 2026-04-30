import copy
import threading
from unittest import mock

import pytest

from tensorrt_llm.llmapi.disagg_utils import RouterConfig
from tensorrt_llm.runtime.kv_cache_hash import truncate_sha256_hash_to_int64
from tensorrt_llm.runtime.kv_cache_manager_v2._block_radix_tree import \
    sequence_to_blockchain_keys
from tensorrt_llm.serve.openai_protocol import (ChatCompletionRequest,
                                                CompletionRequest,
                                                DisaggregatedParams)
# yapf: disable
from tensorrt_llm.serve.router import (KV_CACHE_HASH_ALGO_V1,
                                       KV_CACHE_HASH_ALGO_V2,
                                       KV_CACHE_HASH_ALGO_V2_SHA256_64,
                                       KvCacheAwareRouter,
                                       KvCacheAwareServerState,
                                       LoadBalancingRouter, RoundRobinRouter,
                                       create_router)

# yapf: enable


# Mock class for metadata server
class MockMetadataServer:
    """Mock metadata server for testing router interactions"""

    def __init__(self):
        self.servers = {}
        self.lock = threading.Lock()

    def get(self, key):
        with self.lock:
            return self.servers.get(key)

    def put(self, key, value):
        with self.lock:
            self.servers[key] = value
            return True

    def remove(self, key):
        with self.lock:
            if key in self.servers:
                del self.servers[key]
                return True
            return False

    def add_server(self, key, url):
        with self.lock:
            self.servers[key] = url
            return True

    def keys(self, prefix=""):
        with self.lock:
            return [k for k in self.servers.keys() if k.startswith(prefix)]


@pytest.fixture
def servers():
    return ["server1", "server2", "server3"]


def get_prompt_lengths():
    return [100, 500, 10, 400, 2000, 100]


@pytest.fixture
def context_requests():

    prompt_lengths = get_prompt_lengths()
    # Create multiple CompletionRequest objects with different prompts
    return [
        CompletionRequest(model="TinyLlama",
                          prompt=["the " * length],
                          disaggregated_params=DisaggregatedParams(
                              request_type="context_only",
                              first_gen_tokens=[1000],
                              ctx_request_id=str(index),
                              encoded_opaque_state=None,
                              draft_tokens=None))
        for index, length in enumerate(prompt_lengths)
    ]


@pytest.fixture
def chat_context_requests():

    prompt_lengths = get_prompt_lengths()
    # Create multiple ChatCompletionRequest objects with different prompts
    return [
        ChatCompletionRequest(messages=[{
            "role": "user",
            "content": "the " * length
        }],
                              model="TinyLlama",
                              disaggregated_params=DisaggregatedParams(
                                  request_type="context_only",
                                  first_gen_tokens=[1000],
                                  ctx_request_id=str(index),
                                  encoded_opaque_state=None,
                                  draft_tokens=None))
        for index, length in enumerate(prompt_lengths)
    ]


@pytest.fixture
def gen_requests():

    prompt_lengths = get_prompt_lengths()
    # Create multiple ChatCompletionRequest objects with different prompts
    return [
        CompletionRequest(model="TinyLlama",
                          prompt=["the " * length],
                          disaggregated_params=DisaggregatedParams(
                              request_type="generation_only",
                              first_gen_tokens=[1000],
                              ctx_request_id=str(index),
                              encoded_opaque_state=None,
                              draft_tokens=None))
        for index, length in enumerate(prompt_lengths)
    ]


@pytest.fixture
def chat_gen_requests():

    prompt_lengths = get_prompt_lengths()

    # Create multiple ChatCompletionRequest objects with different prompts
    return [
        ChatCompletionRequest(messages=[{
            "role": "user",
            "content": "the " * length
        }],
                              model="TinyLlama",
                              disaggregated_params=DisaggregatedParams(
                                  request_type="generation_only",
                                  first_gen_tokens=[1000],
                                  ctx_request_id=str(index),
                                  encoded_opaque_state=None,
                                  draft_tokens=None))
        for index, length in enumerate(prompt_lengths)
    ]


@pytest.mark.asyncio
async def test_round_robin_router(servers, context_requests):
    router = RoundRobinRouter(server_role=None, servers=servers)
    server_sequence = [(await router.get_next_server(req))[0]
                       for req in context_requests]
    assert server_sequence == [
        "server1", "server2", "server3", "server1", "server2", "server3"
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("requests_fixture", [
    "context_requests", "chat_context_requests", "gen_requests",
    "chat_gen_requests"
])
async def test_request_balancing_router(servers, requests_fixture, request):
    router = LoadBalancingRouter(server_role=None,
                                 servers=servers,
                                 use_tokens=False)
    requests = request.getfixturevalue(requests_fixture)

    server, _ = await router.get_next_server(requests[0])
    assert server == "server1"
    server, _ = await router.get_next_server(requests[1])
    assert server == "server2"
    server, _ = await router.get_next_server(requests[2])
    assert server == "server3"

    # Similulate terminating 3rd request (on server 3)
    await router.finish_request(requests[2])

    # Now server3 is least loaded
    server, _ = await router.get_next_server(requests[3])
    assert server == "server3"

    # Simulate terminating 4th request (on server 3)
    await router.finish_request(requests[1])

    # Now server2 is least loaded
    server, _ = await router.get_next_server(requests[4])
    assert server == "server2"


@pytest.mark.asyncio
@pytest.mark.parametrize("requests_fixture", ["context_requests"])
async def test_tokens_balancing_router(servers, requests_fixture, request):
    router = LoadBalancingRouter(server_role=None,
                                 servers=servers,
                                 use_tokens=True)
    requests = request.getfixturevalue(requests_fixture)

    server_sequence = [(await router.get_next_server(req))[0]
                       for req in requests]
    # Loads at each step:
    # Step 0:
    # server1: 100
    # server2: 0
    # server3: 0

    # Step 1:
    # server1: 100
    # server2: 500
    # server3: 0

    # Step 2:
    # server1: 100
    # server2: 500
    # server3: 10

    # Step 3:
    # server1: 100
    # server2: 500
    # server3: 410

    # Step 4:
    # server1: 2100
    # server2: 500
    # server3: 410

    # Step 5:
    # server1: 2100
    # server2: 500
    # server3: 510

    assert server_sequence == [
        "server1", "server2", "server3", "server3", "server1", "server3"
    ]

    # Simulate terminating 5th request (on server 1)
    await router.finish_request(requests[4])
    server, _ = await router.get_next_server(requests[4])

    # New loads:
    #server1: 100
    #server2: 500
    #server3: 510
    assert server == "server1"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "requests_fixture",
    ["chat_context_requests", "gen_requests", "chat_gen_requests"])
async def test_gen_tokens_balancing_router(servers, requests_fixture, request):
    router = LoadBalancingRouter(server_role=None,
                                 servers=servers,
                                 use_tokens=True)
    requests = request.getfixturevalue(requests_fixture)

    # Should throw an error if trying to use tokens load balancing with gen-only requests or chat completion requests
    with pytest.raises(ValueError):
        await router.get_next_server(requests[0])


def test_v2_sha256_block_hashes_match_kv_cache_manager_v2(servers):
    tokens_per_block = 4
    token_lists = [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]
    router = KvCacheAwareRouter(server_role=None,
                                servers=servers,
                                tokens_per_block=tokens_per_block)

    block_hashes = router._compute_block_hashes(token_lists,
                                                hash_algo=KV_CACHE_HASH_ALGO_V2)
    expected_block_hashes = [
        block_key.hex()
        for token_block, block_key in sequence_to_blockchain_keys(
            tokens_per_block, None, token_lists[0][:-1]) if token_block
    ]

    assert block_hashes == [expected_block_hashes]


def test_v2_sha256_64_block_hashes_match_truncated_kv_cache_manager_v2(servers):
    tokens_per_block = 4
    token_lists = [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]
    router = KvCacheAwareRouter(server_role=None,
                                servers=servers,
                                tokens_per_block=tokens_per_block)

    block_hashes = router._compute_block_hashes(
        token_lists, hash_algo=KV_CACHE_HASH_ALGO_V2_SHA256_64)
    expected_block_hashes = [
        truncate_sha256_hash_to_int64(block_key)
        for token_block, block_key in sequence_to_blockchain_keys(
            tokens_per_block, None, token_lists[0][:-1]) if token_block
    ]

    assert block_hashes == [expected_block_hashes]
    assert all(isinstance(block_hash, int) for block_hash in block_hashes[0])


@pytest.mark.asyncio
async def test_kv_cache_aware_server_state_uses_hash_algo():
    tokens_per_block = 4
    token_lists = [[1000, 1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008]]
    router = KvCacheAwareRouter(server_role=None,
                                servers=["server1"],
                                tokens_per_block=tokens_per_block)
    v1_block_hashes = router._compute_block_hashes(
        token_lists, hash_algo=KV_CACHE_HASH_ALGO_V1)
    v2_block_hashes = router._compute_block_hashes(
        token_lists, hash_algo=KV_CACHE_HASH_ALGO_V2)
    v2_64_block_hashes = router._compute_block_hashes(
        token_lists, hash_algo=KV_CACHE_HASH_ALGO_V2_SHA256_64)
    state = KvCacheAwareServerState("server1",
                                    tokens_per_block=tokens_per_block)

    state.update_with_events([{
        "event_id": 0,
        "hash_algo": KV_CACHE_HASH_ALGO_V2,
        "data": {
            "type":
            "stored",
            "parent_hash":
            None,
            "blocks": [{
                "block_hash": block_hash
            } for block_hash in v2_block_hashes[0]],
        },
    }])

    assert state.hash_algo == KV_CACHE_HASH_ALGO_V2
    assert await state.matched_tokens(v2_block_hashes,
                                      hash_algo=KV_CACHE_HASH_ALGO_V2) == 8
    assert await state.matched_tokens(v1_block_hashes,
                                      hash_algo=KV_CACHE_HASH_ALGO_V1) == 0
    state.add_blocks(["manual-v2-block"])
    assert state.hash_algo == KV_CACHE_HASH_ALGO_V2
    assert await state.matched_tokens([["manual-v2-block"]],
                                      hash_algo=KV_CACHE_HASH_ALGO_V2) == 4
    assert await state.matched_tokens([["manual-v2-block"]],
                                      hash_algo=KV_CACHE_HASH_ALGO_V1) == 0
    state.remove_blocks(["manual-v2-block"])
    assert await state.matched_tokens([["manual-v2-block"]],
                                      hash_algo=KV_CACHE_HASH_ALGO_V2) == 0

    state.update_with_events([{
        "event_id": 1,
        "hash_algo": KV_CACHE_HASH_ALGO_V2_SHA256_64,
        "data": {
            "type":
            "stored",
            "parent_hash":
            None,
            "blocks": [{
                "block_hash": block_hash
            } for block_hash in v2_64_block_hashes[0]],
        },
    }])

    assert state.hash_algo == KV_CACHE_HASH_ALGO_V2_SHA256_64
    assert await state.matched_tokens(
        v2_64_block_hashes, hash_algo=KV_CACHE_HASH_ALGO_V2_SHA256_64) == 8


@pytest.mark.asyncio
async def test_kv_cache_aware_router_routes_to_v2_server(servers):
    tokens_per_block = 4
    token_lists = [[1000, 1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008]]
    router = KvCacheAwareRouter(server_role=None,
                                servers=servers,
                                use_tokens=False,
                                max_batch_size=32,
                                tokens_per_block=tokens_per_block)
    v2_block_hashes = router._compute_block_hashes(
        token_lists, hash_algo=KV_CACHE_HASH_ALGO_V2)
    v2_server = servers[1]
    router._server_state[v2_server].update_with_events([{
        "event_id": 0,
        "hash_algo": KV_CACHE_HASH_ALGO_V2,
        "data": {
            "type":
            "stored",
            "parent_hash":
            None,
            "blocks": [{
                "block_hash": block_hash
            } for block_hash in v2_block_hashes[0]],
        },
    }])

    request = CompletionRequest(model="TinyLlama",
                                prompt=copy.deepcopy(token_lists))
    server, info = await router.get_next_server(request)
    await router.finish_request(request)

    assert server == v2_server
    assert info["hash_algo"] == KV_CACHE_HASH_ALGO_V2
    assert info["block_hashes"] == v2_block_hashes
    assert info["matches"] == [0, 8, 0]


@pytest.mark.asyncio
async def test_kv_cache_aware_router_uses_server_info_hash_algo(servers):
    tokens_per_block = 4
    token_lists = [[1000, 1001, 1002, 1003, 1004]]
    router = KvCacheAwareRouter(server_role=None,
                                servers=[servers[0]],
                                tokens_per_block=tokens_per_block)
    router._server_info[servers[0]] = {
        "kv_cache_hash_algo": KV_CACHE_HASH_ALGO_V2
    }

    request = CompletionRequest(model="TinyLlama",
                                prompt=copy.deepcopy(token_lists))
    server, info = await router.get_next_server(request)
    await router.finish_request(request)

    assert server == servers[0]
    assert info["hash_algo"] == KV_CACHE_HASH_ALGO_V2
    assert router._server_state[servers[0]].hash_algo == KV_CACHE_HASH_ALGO_V2


@pytest.mark.asyncio
async def test_kv_cache_aware_router_finish_request_polls_events(servers):

    class MockPostResponse:

        def __init__(self, events):
            self._events = events

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            pass

        async def json(self):
            return self._events

    class MockSession:

        def __init__(self, events):
            self._events = events
            self.post_url = None

        def post(self, url):
            self.post_url = url
            return MockPostResponse(self._events)

    tokens_per_block = 4
    token_lists = [[1000, 1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008]]
    router = KvCacheAwareRouter(server_role=None,
                                servers=[servers[0]],
                                use_tokens=False,
                                max_batch_size=32,
                                tokens_per_block=tokens_per_block)
    block_hashes = router._compute_block_hashes(token_lists,
                                                hash_algo=KV_CACHE_HASH_ALGO_V1)
    events = [{
        "event_id": 0,
        "hash_algo": KV_CACHE_HASH_ALGO_V1,
        "data": {
            "type":
            "stored",
            "parent_hash":
            None,
            "blocks": [{
                "block_hash": block_hash
            } for block_hash in block_hashes[0]],
        },
    }]
    session = MockSession(events)

    request = CompletionRequest(model="TinyLlama",
                                prompt=copy.deepcopy(token_lists))
    await router.get_next_server(request)
    await router.finish_request(request, session)

    assert session.post_url == f"{servers[0]}/kv_cache_events"
    assert await router._server_state[servers[0]].matched_tokens(
        block_hashes, hash_algo=KV_CACHE_HASH_ALGO_V1) == 8


@pytest.mark.asyncio
async def test_kv_cache_aware_router_finish_request_decrements_on_poll_error(
        servers):

    class FailingSession:

        def post(self, url):
            raise RuntimeError(f"failed to post to {url}")

    tokens_per_block = 4
    token_lists = [[1000, 1001, 1002, 1003, 1004]]
    router = KvCacheAwareRouter(server_role=None,
                                servers=[servers[0]],
                                use_tokens=False,
                                max_batch_size=32,
                                tokens_per_block=tokens_per_block)

    request = CompletionRequest(model="TinyLlama",
                                prompt=copy.deepcopy(token_lists))
    await router.get_next_server(request)
    assert router._server_state[servers[0]].num_active_requests() == 1

    await router.finish_request(request, FailingSession())

    assert router._server_state[servers[0]].num_active_requests() == 0
    assert id(request) not in router._req_routing_table


@pytest.mark.asyncio
async def test_kv_cache_aware_router(servers):
    # create tokenized requests to skip tokenization
    requests = [
        CompletionRequest(model="TinyLlama", prompt=[[1000] * 100]),
        CompletionRequest(model="TinyLlama",
                          prompt=[[1000] * 50 + [1001] * 150]),
        CompletionRequest(model="TinyLlama", prompt=[[1002] * 300]),
    ]

    router = KvCacheAwareRouter(server_role=None,
                                servers=servers,
                                use_tokens=False,
                                max_batch_size=32,
                                tokens_per_block=32)
    results = [await router.get_next_server(req) for req in requests]
    servers, infos = zip(*results)
    assert servers == ("server1", "server2", "server3")

    # manually updates since no real server is involved
    for request in requests:
        await router.finish_request(request)
    for server, info in results:
        assert "block_hashes" in info and isinstance(info["block_hashes"], list)
        assert len(info["block_hashes"]) == 1 and isinstance(
            info["block_hashes"][0], list)
        router._server_state[server].add_blocks(info["block_hashes"][0])
    # req0 and req1 have a common prefix block: partial match
    assert infos[0]["block_hashes"][0][0] == infos[1]["block_hashes"][0][0]

    # no workloads, route by kv cache hits
    results = [await router.get_next_server(req) for req in reversed(requests)]
    servers, infos = zip(*results)
    assert servers == ("server3", "server2", "server1")
    # matched partial block will be counted as a whole block
    assert infos[0]["matches"] == [0, 0, 320]
    assert infos[1]["matches"] == [32, 224, 0]
    assert infos[2]["matches"] == [128, 32, 0]
    for request in requests:
        await router.finish_request(request)

    # block-wise (32/block) hit rate: 96/512, 32/512, 0/512
    another_request = CompletionRequest(model="TinyLlama",
                                        prompt=[[1000] * 500])
    dup_requests = [copy.copy(another_request) for _ in range(20)]
    another_results = [
        await router.get_next_server(req) for req in dup_requests
    ]
    servers, infos = zip(*another_results)
    # due to workload balancing, not all requests are sent to the same server
    # distribution is related to the hit rate
    counts = {server: 0 for server in servers}
    for server in servers:
        counts[server] += 1
    assert counts["server1"] > counts["server2"] > counts["server3"] > 0
    assert infos[0]["matches"] == [96, 32, 0]
    for req in dup_requests:
        await router.finish_request(req)

    # test router after block eviction on server 1&2
    # results: server3(request2), server2(request1), server1(request0)
    for server, infos in results[1:]:
        assert server in ["server1", "server2"]
        events = [{"type": "removed", "block_hashes": infos["block_hashes"][0]}]
        router._server_state[server].update_with_events(events)

    results = [await router.get_next_server(req) for req in reversed(requests)]
    servers, infos = zip(*results)
    assert servers == ("server3", "server1", "server2")


def test_create_router(servers):
    default_router = create_router(None, servers)
    assert isinstance(default_router, RoundRobinRouter)

    round_robin_router = create_router(RouterConfig(type="round_robin"),
                                       servers)
    assert isinstance(round_robin_router, RoundRobinRouter)

    router_config = RouterConfig(type="load_balancing",
                                 args={"use_tokens": False})
    requests_load_balancing_router = create_router(router_config, servers)
    assert isinstance(requests_load_balancing_router, LoadBalancingRouter)
    assert not requests_load_balancing_router._use_tokens

    router_config.args["use_tokens"] = True
    tokens_load_balancing_router = create_router(router_config, servers)
    assert isinstance(tokens_load_balancing_router, LoadBalancingRouter)
    assert tokens_load_balancing_router._use_tokens

    router_config.type = "kv_cache_aware"
    kv_cache_aware_router = create_router(router_config, servers)
    assert isinstance(kv_cache_aware_router, KvCacheAwareRouter)

    with pytest.raises(ValueError):
        create_router(RouterConfig(type="unsupported_router"), servers)


@pytest.fixture
def mock_metadata_server():
    return MockMetadataServer()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "router_class", [RoundRobinRouter, LoadBalancingRouter, KvCacheAwareRouter])
async def test_fetch_live_servers_context(mock_metadata_server, router_class):
    # Create router with mock metadata server
    router = router_class(server_role="context",
                          metadata_server=mock_metadata_server)

    # Initial check - should be no servers
    with pytest.raises(ValueError):
        servers = await router.fetch_live_servers()

    # Add a server
    server_key = "trtllm/server1"
    server_url = "http://localhost:8001"
    mock_metadata_server.add_server(server_key, {"url": server_url})

    # Fetch servers again
    servers = await router.fetch_live_servers()
    assert len(servers) == 1, "Should have one server after adding and waiting"
    assert server_key in servers, "Server key should be present"
    assert servers[
        server_key] == server_url, "Server URL should match what was added"

    # Add another server
    server_key2 = "trtllm/server2"
    server_url2 = "http://localhost:8002"
    mock_metadata_server.add_server(server_key2, {"url": server_url2})

    # Fetch servers again
    servers = await router.fetch_live_servers()
    assert len(
        servers
    ) == 2, "Should have two servers after adding second one and waiting"
    assert server_key in servers, "First server should still be present"
    assert server_key2 in servers, "Second server should be present"

    # Remove a server
    mock_metadata_server.remove(server_key)

    # Fetch servers again
    servers = await router.fetch_live_servers()
    assert len(
        servers) == 1, "Should have one server after removing one and waiting"
    assert server_key2 in servers, "Second server should still be present"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "router_class", [RoundRobinRouter, LoadBalancingRouter, KvCacheAwareRouter])
async def test_server_health_check(mock_metadata_server, router_class):
    router = router_class(server_role="context",
                          metadata_server=mock_metadata_server)

    # Add two servers
    server_key1 = "trtllm/server1"
    server_url1 = "http://localhost:8001"
    mock_metadata_server.add_server(server_key1, {"url": server_url1})

    server_key2 = "trtllm/server2"
    server_url2 = "http://localhost:8002"
    mock_metadata_server.add_server(server_key2, {"url": server_url2})

    # Mock the is_server_healthy method to simulate one server being down
    with mock.patch.object(router, '_check_server_health') as mock_is_healthy:
        # Only the second server is "healthy"
        mock_is_healthy.side_effect = lambda url, silent=False: url == server_url2

        # Fetch servers with health check
        servers = await router.fetch_live_servers()
        live_servers = await router.check_servers_health(servers)
        assert len(live_servers) == 1, "Should have one healthy server"
        assert server_url2 in live_servers, "Second server should still be present"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "router_class", [RoundRobinRouter, LoadBalancingRouter, KvCacheAwareRouter])
async def test_get_next_server_exclude_server(router_class):
    servers = ["server1", "server2", "server3"]
    router = router_class(server_role="context", servers=servers)
    exclude_server2 = {server: 0 for server in servers}
    exclude_server3 = {server: 0 for server in servers}
    for _ in range(0, 10):
        server, _ = await router.get_next_server(CompletionRequest(
            model="TinyLlama", prompt=[[10] * 10]),
                                                 exclude_server="server2")
        exclude_server2[server] += 1
        server, _ = await router.get_next_server(CompletionRequest(
            model="TinyLlama", prompt=[[10] * 10]),
                                                 exclude_server="server3")
        exclude_server3[server] += 1
    if router_class == KvCacheAwareRouter:
        # KvCacheAwareRouter is not load-balanced
        assert exclude_server2["server2"] == 0
        assert exclude_server3["server3"] == 0
    else:
        assert exclude_server2["server1"] > 0 and exclude_server2[
            "server2"] == 0 and exclude_server2["server3"] > 0, exclude_server2
        assert exclude_server3["server1"] > 0 and exclude_server3[
            "server2"] > 0 and exclude_server3["server3"] == 0, exclude_server3


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "router_class", [RoundRobinRouter, LoadBalancingRouter, KvCacheAwareRouter])
async def test_get_next_server_exclude_server_insufficient(router_class):
    servers = ["server1"]
    router = router_class(server_role="context",
                          servers=servers,
                          use_tokens=False)
    with pytest.raises(Exception):
        await router.get_next_server(CompletionRequest(model="TinyLlama",
                                                       prompt=[[10] * 10]),
                                     exclude_server=servers[0])
