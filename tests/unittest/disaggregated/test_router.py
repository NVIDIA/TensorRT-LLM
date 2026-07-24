import asyncio
import copy
import random
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import aiohttp
import msgpack
import pytest

from tensorrt_llm.llmapi.disagg_utils import RouterConfig
from tensorrt_llm.runtime.kv_cache_hash import (get_cache_salt_id,
                                                hash_v1_block_key,
                                                truncate_sha256_hash_to_int64)
from tensorrt_llm.runtime.kv_cache_manager_v2._block_radix_tree import (
    ReuseScope, sequence_to_blockchain_keys)
# yapf: disable
from tensorrt_llm.serve.openai_protocol import (ChatCompletionRequest,
                                                ChatCompletionToolsParam,
                                                CompletionRequest,
                                                ConversationParams,
                                                DisaggregatedParams,
                                                FunctionDefinition)
from tensorrt_llm.serve.router import (KV_CACHE_HASH_ALGO_V1,
                                       KV_CACHE_HASH_ALGO_V2,
                                       KV_CACHE_HASH_ALGO_V2_SHA256_64,
                                       BlockHashMixin, ConversationRouter,
                                       CoordinatorDelegatingRouter,
                                       KvCacheAwareRouter,
                                       KvCacheAwareServerState,
                                       LoadBalancingRouter, RoundRobinRouter,
                                       block_key_hasher, create_router)

# yapf: enable


def test_native_block_key_hasher_matches_python_v1():
    """Native C++ BlockKeyHasher must be bit-exact with hash_v1_block_key.

    Covers single blocks and the chained per-block pattern used by
    BlockHashMixin._compute_block_hashes.
    """
    random.seed(0)
    for _ in range(500):
        n = random.choice([1, 2, 31, 32, 33, 64, 97])
        toks = [random.randint(0, 300000) for _ in range(n)]
        parent = random.choice([None, 0, random.randint(1, 2**64 - 1)])
        ref = hash_v1_block_key(toks,
                                parent_hash=0 if parent is None else parent)
        assert block_key_hasher(toks, parent) == ref

    # chained per-block (the real _compute_block_hashes pattern)
    toks = [random.randint(0, 300000) for _ in range(2000)]
    parent = None
    for t in range(0, len(toks) - 1, 32):
        t_end = min(t + 32, len(toks) - 1)
        h = block_key_hasher(toks[t:t_end], parent)
        assert h == hash_v1_block_key(
            toks[t:t_end], parent_hash=0 if parent is None else parent)
        parent = h


def _make_mock_aiohttp_session(return_value=None):
    """Create a mock aiohttp.ClientSession whose .post() returns canned JSON."""
    if return_value is None:
        return_value = []
    mock_response = mock.AsyncMock()
    mock_response.json = mock.AsyncMock(return_value=return_value)
    mock_ctx = mock.AsyncMock()
    mock_ctx.__aenter__ = mock.AsyncMock(return_value=mock_response)
    mock_ctx.__aexit__ = mock.AsyncMock(return_value=False)
    mock_session = mock.MagicMock(spec=aiohttp.ClientSession)
    mock_session.post = mock.MagicMock(return_value=mock_ctx)
    mock_session.get = mock.MagicMock(return_value=mock_ctx)
    mock_session.close = mock.AsyncMock()
    return mock_session


@pytest.mark.asyncio
async def test_coordinator_finish_retry_is_bounded():
    local_router = RoundRobinRouter(server_role=None, servers=["server1"])
    router = CoordinatorDelegatingRouter("http://coordinator", local_router,
                                         "generation")

    def _response(status, body):
        response = mock.AsyncMock()
        response.status = status
        response.read = mock.AsyncMock(
            return_value=msgpack.packb(body, use_bin_type=True))
        context = mock.AsyncMock()
        context.__aenter__ = mock.AsyncMock(return_value=response)
        context.__aexit__ = mock.AsyncMock(return_value=False)
        return context

    session = mock.MagicMock()
    session.post = mock.MagicMock(side_effect=[
        _response(503, {"error": "temporarily unavailable"}),
        _response(503, {"error": "temporarily unavailable"}),
        _response(503, {"error": "temporarily unavailable"}),
    ])
    session.close = mock.AsyncMock()
    router._session = session

    with mock.patch("tensorrt_llm.serve.router.asyncio.sleep",
                    new_callable=mock.AsyncMock) as sleep:
        await router._finish_async(123, True)

    assert session.post.call_count == 3
    assert sleep.await_count == 2
    assert all(call.kwargs["timeout"] == 5
               for call in session.post.call_args_list)


@pytest.mark.asyncio
async def test_coordinator_finish_queue_is_bounded():
    local_router = RoundRobinRouter(server_role=None, servers=["server1"])
    router = CoordinatorDelegatingRouter("http://coordinator", local_router,
                                         "generation")
    router._finish_queue = asyncio.Queue(maxsize=1)
    router._ensure_finish_workers = mock.Mock()

    request = mock.Mock()
    await router.finish_request(request, req_id=1)
    await router.finish_request(request, req_id=2)

    assert router._finish_queue.qsize() == 1
    assert router._dropped_finishes == 1


@pytest.fixture(autouse=True)
def mock_aiohttp_session(request):
    """Auto-mock aiohttp.ClientSession so poll_events doesn't make real HTTP calls."""
    mock_session = _make_mock_aiohttp_session()
    with mock.patch('tensorrt_llm.serve.router.aiohttp.ClientSession',
                    return_value=mock_session):
        yield mock_session


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

    # First 3 requests: all servers start at 0 load, each gets a unique server
    assigned = {}
    for i in range(3):
        server, _ = await router.get_next_server(requests[i])
        assigned[i] = server
    assert len(set(assigned.values())) == 3, "All 3 servers should be used"

    # Finish 3rd request — its server drops to 0 (uniquely least loaded)
    await router.finish_request(requests[2])
    server, _ = await router.get_next_server(requests[3])
    assert server == assigned[2]

    # Finish 2nd request — its server drops to 0 (uniquely least loaded)
    await router.finish_request(requests[1])
    server, _ = await router.get_next_server(requests[4])
    assert server == assigned[1]


@pytest.mark.asyncio
@pytest.mark.parametrize("requests_fixture", ["context_requests"])
async def test_tokens_balancing_router(servers, requests_fixture, request):
    router = LoadBalancingRouter(server_role=None,
                                 servers=servers,
                                 use_tokens=True)
    requests = request.getfixturevalue(requests_fixture)

    # prompt_lengths = [100, 500, 10, 400, 2000, 100]
    server_sequence = [(await router.get_next_server(req))[0]
                       for req in requests]

    # Steps 0-1: tied loads → implementation-defined assignment.
    # Step 2+: unique least-loaded → deterministic relative to steps 0-1.
    s0, s1, s2 = server_sequence[0], server_sequence[1], server_sequence[2]
    assert len({s0, s1, s2}) == 3, "All 3 servers should be used"

    # After step 2: s0=100, s1=500, s2=10
    # Step 3: s2 uniquely least (10 < 100 < 500)
    assert server_sequence[3] == s2

    # After step 3: s0=100, s1=500, s2=410
    # Step 4: s0 uniquely least (100 < 410 < 500)
    assert server_sequence[4] == s0

    # After step 4: s0=2100, s1=500, s2=410
    # Step 5: s2 uniquely least (410 < 500 < 2100)
    assert server_sequence[5] == s2

    # Finish 5th request (2000 tokens on s0)
    await router.finish_request(requests[4])
    server, _ = await router.get_next_server(requests[4])

    # After finish: s0=100, s1=500, s2=510 → s0 uniquely least
    assert server == s0


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
            tokens_per_block, ReuseScope(), token_lists[0][:-1]) if token_block
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
            tokens_per_block, ReuseScope(), token_lists[0][:-1]) if token_block
    ]

    assert block_hashes == [expected_block_hashes]
    assert all(isinstance(block_hash, int) for block_hash in block_hashes[0])


def test_cache_aware_router_block_hashes_include_cache_salt_id(servers):
    tokens_per_block = 4
    cache_salt_id = 123
    token_lists = [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]
    router = KvCacheAwareRouter(server_role=None,
                                servers=servers,
                                tokens_per_block=tokens_per_block)

    salted_v1_block_hashes = router._compute_block_hashes(
        token_lists,
        hash_algo=KV_CACHE_HASH_ALGO_V1,
        cache_salt_id=cache_salt_id)
    expected_v1_block_hashes = []
    parent_hash = None
    for t in range(0, len(token_lists[0]) - 1, tokens_per_block):
        t_end = min(t + tokens_per_block, len(token_lists[0]) - 1)
        parent_hash = hash_v1_block_key(
            token_lists[0][t:t_end],
            parent_hash=0 if parent_hash is None else parent_hash,
            cache_salt_id=cache_salt_id)
        expected_v1_block_hashes.append(parent_hash)

    salted_v2_block_hashes = router._compute_block_hashes(
        token_lists,
        hash_algo=KV_CACHE_HASH_ALGO_V2,
        cache_salt_id=cache_salt_id)
    reuse_scope = ReuseScope(salt=cache_salt_id)
    expected_v2_block_hashes = [
        block_key.hex()
        for token_block, block_key in sequence_to_blockchain_keys(
            tokens_per_block, reuse_scope, token_lists[0][:-1]) if token_block
    ]

    salted_v2_64_block_hashes = router._compute_block_hashes(
        token_lists,
        hash_algo=KV_CACHE_HASH_ALGO_V2_SHA256_64,
        cache_salt_id=cache_salt_id)
    expected_v2_64_block_hashes = [
        truncate_sha256_hash_to_int64(block_key)
        for token_block, block_key in sequence_to_blockchain_keys(
            tokens_per_block, reuse_scope, token_lists[0][:-1]) if token_block
    ]

    assert salted_v1_block_hashes == [expected_v1_block_hashes]
    assert salted_v2_block_hashes == [expected_v2_block_hashes]
    assert salted_v2_64_block_hashes == [expected_v2_64_block_hashes]
    assert salted_v1_block_hashes != router._compute_block_hashes(
        token_lists, hash_algo=KV_CACHE_HASH_ALGO_V1)
    assert salted_v2_block_hashes != router._compute_block_hashes(
        token_lists, hash_algo=KV_CACHE_HASH_ALGO_V2)
    assert salted_v2_64_block_hashes != router._compute_block_hashes(
        token_lists, hash_algo=KV_CACHE_HASH_ALGO_V2_SHA256_64)


def test_cache_salt_id_derivation_matches_worker_path():
    from tensorrt_llm.inputs.utils import \
        get_cache_salt_id as worker_get_cache_salt_id

    assert get_cache_salt_id("abc") == 3697813978277427044
    assert get_cache_salt_id("tenant-a") == get_cache_salt_id("tenant-a")
    assert get_cache_salt_id("tenant-a") != get_cache_salt_id("tenant-b")
    assert get_cache_salt_id("tenant-a") == worker_get_cache_salt_id("tenant-a")


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
async def test_kv_cache_aware_router_routes_with_cache_salt(
        servers, monkeypatch):

    class SaltedRequest:
        model = "TinyLlama"
        cache_salt = "tenant-a"

        def __init__(self, prompt):
            self.prompt = prompt

    tokens_per_block = 4
    cache_salt_id = 123
    token_lists = [[1000, 1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008]]
    router = KvCacheAwareRouter(server_role=None,
                                servers=servers,
                                use_tokens=False,
                                max_batch_size=32,
                                tokens_per_block=tokens_per_block)
    monkeypatch.setattr("tensorrt_llm.serve.router_utils.get_cache_salt_id",
                        lambda cache_salt: cache_salt_id)
    for server in servers:
        router._server_info[server] = {
            "kv_cache_hash_algo": KV_CACHE_HASH_ALGO_V2
        }

    salted_block_hashes = router._compute_block_hashes(
        token_lists,
        hash_algo=KV_CACHE_HASH_ALGO_V2,
        cache_salt_id=cache_salt_id)
    unsalted_block_hashes = router._compute_block_hashes(
        token_lists, hash_algo=KV_CACHE_HASH_ALGO_V2)
    salted_server = servers[1]
    router._server_state[salted_server].update_with_events([{
        "event_id": 0,
        "hash_algo": KV_CACHE_HASH_ALGO_V2,
        "data": {
            "type":
            "stored",
            "parent_hash":
            None,
            "blocks": [{
                "block_hash": block_hash
            } for block_hash in salted_block_hashes[0]],
        },
    }])

    request = SaltedRequest(prompt=copy.deepcopy(token_lists))
    server, info = await router.get_next_server(request)
    await router.finish_request(request)

    assert salted_block_hashes != unsalted_block_hashes
    assert server == salted_server
    assert info["hash_algo"] == KV_CACHE_HASH_ALGO_V2
    assert info["block_hashes"] == salted_block_hashes
    assert info["matches"] == [0, 8, 0]


@pytest.mark.asyncio
async def test_kv_cache_aware_router_uses_server_info_hash_algo(servers):
    tokens_per_block = 4
    token_lists = [[1000, 1001, 1002, 1003, 1004]]
    router = KvCacheAwareRouter(server_role=None,
                                servers=[servers[0]],
                                tokens_per_block=tokens_per_block)
    # _prepare_server seeds server state from the handshake; simulate it here.
    router._server_info[servers[0]] = {
        "kv_cache_hash_algo": KV_CACHE_HASH_ALGO_V2
    }
    router._server_state[servers[0]].set_hash_algo(KV_CACHE_HASH_ALGO_V2)

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
    # Poll runs in a background task; await it before asserting.
    await router._server_state[servers[0]]._poll_task

    assert session.post_url == f"http://{servers[0]}/kv_cache_events"
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


def test_kv_cache_aware_server_state_add_blocks_set_update_fast_path():
    state = KvCacheAwareServerState("server-x", tokens_per_block=4)
    state.set_hash_algo(KV_CACHE_HASH_ALGO_V1)

    state.add_blocks([1, 2, 3], hash_algo=KV_CACHE_HASH_ALGO_V1)
    assert state._block_table(KV_CACHE_HASH_ALGO_V1) == {1, 2, 3}

    state.add_blocks((h for h in [4, 5]), hash_algo=KV_CACHE_HASH_ALGO_V1)
    assert state._block_table(KV_CACHE_HASH_ALGO_V1) == {1, 2, 3, 4, 5}

    state.add_blocks([1, 1, 2], hash_algo=KV_CACHE_HASH_ALGO_V1)
    assert state._block_table(KV_CACHE_HASH_ALGO_V1) == {1, 2, 3, 4, 5}

    state.add_blocks([], hash_algo=KV_CACHE_HASH_ALGO_V1)
    state.add_blocks(iter([]), hash_algo=KV_CACHE_HASH_ALGO_V1)
    assert state._block_table(KV_CACHE_HASH_ALGO_V1) == {1, 2, 3, 4, 5}


@pytest.mark.asyncio
async def test_kv_cache_aware_server_state_schedule_poll_coalesces():
    state = KvCacheAwareServerState("server-x", tokens_per_block=4)
    ready = asyncio.Event()

    async def slow_poll(session=None):
        await ready.wait()

    with mock.patch.object(state, "poll_and_update", side_effect=slow_poll):
        state.schedule_poll_and_update(session=None)
        first_task = state._poll_task
        state.schedule_poll_and_update(session=None)
        assert state._poll_task is first_task
        ready.set()
        await first_task


@pytest.mark.asyncio
async def test_kv_cache_aware_server_state_schedule_poll_relaunches_after_done(
):
    state = KvCacheAwareServerState("server-x", tokens_per_block=4)
    calls = 0

    async def fake_poll(session=None):
        nonlocal calls
        calls += 1

    with mock.patch.object(state, "poll_and_update", side_effect=fake_poll):
        state.schedule_poll_and_update(session=None)
        await state._poll_task
        first_task = state._poll_task
        state.schedule_poll_and_update(session=None)
        await state._poll_task
        assert state._poll_task is not first_task

    assert calls == 2


@pytest.mark.asyncio
async def test_kv_cache_aware_server_state_schedule_poll_rearms_pending():
    # A poll requested while one is in flight must re-run once more so the last
    # finish_request can never strand its events behind a coalesced poll.
    state = KvCacheAwareServerState("server-x", tokens_per_block=4)
    calls = 0
    started = asyncio.Event()
    release = asyncio.Event()

    async def slow_poll(session=None):
        nonlocal calls
        calls += 1
        started.set()
        await release.wait()

    with mock.patch.object(state, "poll_and_update", side_effect=slow_poll):
        state.schedule_poll_and_update(session=None)
        await started.wait()  # first poll is now actually in flight
        started.clear()
        # Second request arrives while the first poll is still blocked.
        state.schedule_poll_and_update(session=None)
        release.set()
        await state._poll_task

    assert calls == 2


@pytest.mark.asyncio
async def test_kv_cache_aware_server_state_cancel_poll_task():
    state = KvCacheAwareServerState("server-x", tokens_per_block=4)
    started = asyncio.Event()

    async def blocking_poll(session=None):
        started.set()
        await asyncio.Event().wait()

    with mock.patch.object(state, "poll_and_update", side_effect=blocking_poll):
        state.schedule_poll_and_update(session=None)
        await started.wait()
        await state.cancel_poll_task()

    assert state._poll_task is None
    assert state._poll_pending is False


def test_kv_cache_aware_server_state_remove_blocks_silent_on_missing():
    state = KvCacheAwareServerState("server-x", tokens_per_block=4)
    state.add_blocks([10, 20, 30], hash_algo=KV_CACHE_HASH_ALGO_V1)

    state.remove_blocks([20, 99, 100], hash_algo=KV_CACHE_HASH_ALGO_V1)
    assert state._block_table(KV_CACHE_HASH_ALGO_V1) == {10, 30}

    state.remove_blocks([777, 888], hash_algo=KV_CACHE_HASH_ALGO_V1)
    assert state._block_table(KV_CACHE_HASH_ALGO_V1) == {10, 30}

    state.remove_blocks((h for h in [10]), hash_algo=KV_CACHE_HASH_ALGO_V1)
    assert state._block_table(KV_CACHE_HASH_ALGO_V1) == {30}


@pytest.mark.asyncio
async def test_kv_cache_aware_router_applies_blocks_after_successful_finish(
        servers):
    tokens_per_block = 4
    token_lists = [[2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008]]
    router = KvCacheAwareRouter(server_role=None,
                                servers=servers,
                                use_tokens=False,
                                max_batch_size=32,
                                tokens_per_block=tokens_per_block,
                                track_routed_blocks=True)

    request = CompletionRequest(model="TinyLlama",
                                prompt=copy.deepcopy(token_lists))
    server, info = await router.get_next_server(request)
    flat_expected = [h for hl in info["block_hashes"] for h in hl]
    hash_algo = info["hash_algo"]

    assert await router._server_state[server].matched_tokens(
        info["block_hashes"], hash_algo=hash_algo) == 0

    await router.finish_request(request)

    assert router._server_state[server]._block_table(hash_algo).issuperset(
        flat_expected)
    assert await router._server_state[server].matched_tokens(
        info["block_hashes"], hash_algo=hash_algo) == 8


@pytest.mark.asyncio
async def test_kv_cache_aware_router_routed_blocks_disabled_skips_pending(
        servers):
    tokens_per_block = 4
    token_lists = [[3000, 3001, 3002, 3003, 3004, 3005, 3006, 3007, 3008]]
    router = KvCacheAwareRouter(server_role=None,
                                servers=servers,
                                use_tokens=False,
                                max_batch_size=32,
                                tokens_per_block=tokens_per_block,
                                track_routed_blocks=False)

    request = CompletionRequest(model="TinyLlama",
                                prompt=copy.deepcopy(token_lists))
    server, info = await router.get_next_server(request)

    await router.finish_request(request)

    assert await router._server_state[server].matched_tokens(
        info["block_hashes"], hash_algo=info["hash_algo"]) == 0


@pytest.mark.asyncio
async def test_kv_cache_aware_router_discards_routed_blocks_on_failure(servers):
    tokens_per_block = 4
    token_lists = [[4000, 4001, 4002, 4003, 4004, 4005, 4006, 4007, 4008]]
    router = KvCacheAwareRouter(server_role=None,
                                servers=servers,
                                use_tokens=False,
                                max_batch_size=32,
                                tokens_per_block=tokens_per_block,
                                track_routed_blocks=True)

    request = CompletionRequest(model="TinyLlama",
                                prompt=copy.deepcopy(token_lists))
    server, info = await router.get_next_server(request)
    assert await router._server_state[server].matched_tokens(
        info["block_hashes"], hash_algo=info["hash_algo"]) == 0

    await router.finish_request(request, success=False)

    assert await router._server_state[server].matched_tokens(
        info["block_hashes"], hash_algo=info["hash_algo"]) == 0


@pytest.mark.asyncio
async def test_kv_cache_aware_router_load_cap_excludes_overloaded(servers):
    tokens_per_block = 4
    token_lists = [[5000] * 16]
    router = KvCacheAwareRouter(server_role=None,
                                servers=servers,
                                use_tokens=False,
                                max_batch_size=10,
                                tokens_per_block=tokens_per_block,
                                load_cap=0.8)
    router._server_state[servers[0]]._num_active_requests = 9
    router._server_state[servers[1]]._num_active_requests = 9
    router._server_state[servers[2]]._num_active_requests = 1

    request = CompletionRequest(model="TinyLlama",
                                prompt=copy.deepcopy(token_lists))
    chosen, _ = await router.get_next_server(request)
    assert chosen == servers[2]


@pytest.mark.asyncio
async def test_kv_cache_aware_router_load_cap_fallback_when_all_overloaded(
        servers):
    router = KvCacheAwareRouter(server_role=None,
                                servers=servers,
                                use_tokens=False,
                                max_batch_size=10,
                                tokens_per_block=4,
                                load_cap=0.5)
    router._server_state[servers[0]]._num_active_requests = 8
    router._server_state[servers[1]]._num_active_requests = 6
    router._server_state[servers[2]]._num_active_requests = 9

    request = CompletionRequest(model="TinyLlama", prompt=[[100] * 16])
    chosen, _ = await router.get_next_server(request)
    assert chosen == servers[1]


@pytest.mark.asyncio
async def test_kv_cache_aware_router_load_weight_scales_load_penalty(servers):
    tokens_per_block = 4
    token_lists = [[6000] * 16]
    router = KvCacheAwareRouter(server_role=None,
                                servers=servers,
                                use_tokens=False,
                                max_batch_size=10,
                                tokens_per_block=tokens_per_block,
                                load_weight=5.0,
                                load_cap=1.0)
    block_hashes = router._compute_block_hashes(token_lists)
    router._server_state[servers[0]].add_blocks(
        [h for hl in block_hashes for h in hl])
    router._server_state[servers[0]]._num_active_requests = 7
    router._server_state[servers[1]]._num_active_requests = 1
    router._server_state[servers[2]]._num_active_requests = 1

    request = CompletionRequest(model="TinyLlama",
                                prompt=copy.deepcopy(token_lists))
    chosen, _ = await router.get_next_server(request)
    assert chosen != servers[0]


@pytest.mark.asyncio
async def test_kv_cache_aware_router_prepare_server_warns_on_tpb_mismatch_divisible(
):
    # router=32, worker=64: 64 % 32 == 0 → warning only, server stays ready
    router = KvCacheAwareRouter(server_role=None,
                                servers=["server-a"],
                                tokens_per_block=32)

    async def fake_fetch(server, timeout):
        return {"tokens_per_block": 64, "kv_cache_hash_algo": "v1_block_key"}

    with mock.patch.object(router, "_fetch_server_info",
                           side_effect=fake_fetch):
        await router._prepare_server("server-a")

    assert "server-a" in router._prepared_ready_servers
    assert router._server_info["server-a"]["tokens_per_block"] == 64


@pytest.mark.asyncio
async def test_kv_cache_aware_router_prepare_server_raises_on_tpb_not_divisible(
):
    # router=32, worker=48: 48 % 32 != 0 → RuntimeError, server removed
    router = KvCacheAwareRouter(server_role=None,
                                servers=["server-a"],
                                tokens_per_block=32)

    async def fake_fetch(server, timeout):
        return {"tokens_per_block": 48, "kv_cache_hash_algo": "v1_block_key"}

    with mock.patch.object(router, "_fetch_server_info",
                           side_effect=fake_fetch):
        with pytest.raises(RuntimeError, match="not divisible"):
            await router._prepare_server("server-a")

    assert "server-a" not in router._prepared_ready_servers
    assert "server-a" not in router._server_info


@pytest.mark.asyncio
async def test_kv_cache_aware_router_prepare_server_keeps_on_tpb_match():
    router = KvCacheAwareRouter(server_role=None,
                                servers=["server-a"],
                                tokens_per_block=32)

    async def fake_fetch(server, timeout):
        return {"tokens_per_block": 32, "kv_cache_hash_algo": "v1_block_key"}

    with mock.patch.object(router, "_fetch_server_info",
                           side_effect=fake_fetch):
        await router._prepare_server("server-a")

    assert "server-a" in router._prepared_ready_servers
    assert router._server_info["server-a"]["tokens_per_block"] == 32


@pytest.mark.asyncio
async def test_kv_cache_aware_router_prepare_server_keeps_when_worker_omits_tpb(
):
    router = KvCacheAwareRouter(server_role=None,
                                servers=["server-a"],
                                tokens_per_block=32)

    async def fake_fetch(server, timeout):
        return {"kv_cache_hash_algo": "v1_block_key"}

    with mock.patch.object(router, "_fetch_server_info",
                           side_effect=fake_fetch):
        await router._prepare_server("server-a")

    assert "server-a" in router._prepared_ready_servers


@pytest.mark.asyncio
async def test_kv_cache_aware_router_prepare_server_adopts_worker_tpb_when_unset(
):
    router = KvCacheAwareRouter(server_role=None, servers=["server-a"])
    assert router._tpb_auto is True
    assert router._tokens_per_block == 32

    async def fake_fetch(server, timeout):
        return {"tokens_per_block": 128, "kv_cache_hash_algo": "v1_block_key"}

    with mock.patch.object(router, "_fetch_server_info",
                           side_effect=fake_fetch):
        await router._prepare_server("server-a")

    assert "server-a" in router._prepared_ready_servers
    assert router._tokens_per_block == 128
    assert router._tpb_auto is False


@pytest.mark.asyncio
@pytest.mark.parametrize("worker_tpb", [32, 64, 128])
async def test_kv_cache_aware_router_auto_tpb_equals_worker(worker_tpb):
    router = KvCacheAwareRouter(server_role=None, servers=["server-a"])

    async def fake_fetch(server, timeout):
        return {
            "tokens_per_block": worker_tpb,
            "kv_cache_hash_algo": "v1_block_key"
        }

    with mock.patch.object(router, "_fetch_server_info",
                           side_effect=fake_fetch):
        await router._prepare_server("server-a")

    assert router._tokens_per_block == worker_tpb
    assert "server-a" in router._prepared_ready_servers


@pytest.mark.asyncio
async def test_kv_cache_aware_router_prepare_server_warns_on_missing_algo():
    router = KvCacheAwareRouter(server_role=None,
                                servers=["server-a"],
                                tokens_per_block=32)

    async def fake_fetch(server, timeout):
        return {"tokens_per_block": 32}

    with mock.patch("tensorrt_llm.serve.router.logger") as mock_logger:
        with mock.patch.object(router,
                               "_fetch_server_info",
                               side_effect=fake_fetch):
            await router._prepare_server("server-a")

    assert "server-a" in router._prepared_ready_servers
    assert any("did not expose kv_cache_hash_algo" in str(call.args[0])
               for call in mock_logger.warning.call_args_list)
    assert router._get_server_hash_algo("server-a") == KV_CACHE_HASH_ALGO_V1


@pytest.mark.asyncio
async def test_kv_cache_aware_router_prepare_server_raises_on_unknown_algo():
    router = KvCacheAwareRouter(server_role=None,
                                servers=["server-a"],
                                tokens_per_block=32)

    async def fake_fetch(server, timeout):
        return {"tokens_per_block": 32, "kv_cache_hash_algo": "bogus_algo"}

    with mock.patch.object(router, "_fetch_server_info",
                           side_effect=fake_fetch):
        with pytest.raises(RuntimeError, match="Unknown kv_cache_hash_algo"):
            await router._prepare_server("server-a")

    assert "server-a" not in router._prepared_ready_servers
    assert "server-a" not in router._server_info


@pytest.mark.asyncio
async def test_kv_cache_aware_router_prepare_server_persists_algo_for_lock_free_read(
):
    router = KvCacheAwareRouter(server_role=None,
                                servers=["server-a"],
                                tokens_per_block=32)

    async def fake_fetch(server, timeout):
        return {
            "tokens_per_block": 32,
            "kv_cache_hash_algo": KV_CACHE_HASH_ALGO_V2,
        }

    with mock.patch.object(router, "_fetch_server_info",
                           side_effect=fake_fetch):
        await router._prepare_server("server-a")

    # Per-request read is now sync; no lock/await on the hot path.
    assert router._get_server_hash_algo("server-a") == KV_CACHE_HASH_ALGO_V2
    assert router._server_state["server-a"].hash_algo == KV_CACHE_HASH_ALGO_V2


@pytest.mark.asyncio
async def test_kv_cache_aware_router(servers, mock_aiohttp_session):
    # create tokenized requests to skip tokenization
    # req0: [1000]*100, req1: [1000]*50+[1001]*150, req2: [1002]*300
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
                                tokens_per_block=32,
                                track_routed_blocks=False)
    results = [await router.get_next_server(req) for req in requests]
    assigned_servers, infos = zip(*results)
    # Initial routing (empty caches): all 3 should get distinct servers
    assert len(set(assigned_servers)) == 3

    # Track which server cached which request
    server_of = {i: assigned_servers[i] for i in range(3)}
    all_servers = list(router._server_state.keys())

    def matches_by_server(info):
        """Map server → matched token count from positional matches list."""
        return dict(zip(all_servers, info["matches"]))

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
    # reversed: [req2, req1, req0] — each should route to its cached server
    results = [await router.get_next_server(req) for req in reversed(requests)]
    hit_servers, hit_infos = zip(*results)
    assert hit_servers == (server_of[2], server_of[1], server_of[0])

    # matched partial block will be counted as a whole block
    # req2 ([1002]*300): only matches server_of[2] → 320 tokens
    m0 = matches_by_server(hit_infos[0])
    assert m0[server_of[2]] == 320
    assert m0[server_of[0]] == 0
    assert m0[server_of[1]] == 0
    # req1 ([1000]*50+[1001]*150): full match server_of[1] → 224, partial server_of[0] → 32
    m1 = matches_by_server(hit_infos[1])
    assert m1[server_of[1]] == 224
    assert m1[server_of[0]] == 32
    assert m1[server_of[2]] == 0
    # req0 ([1000]*100): full match server_of[0] → 128, partial server_of[1] → 32
    m2 = matches_by_server(hit_infos[2])
    assert m2[server_of[0]] == 128
    assert m2[server_of[1]] == 32
    assert m2[server_of[2]] == 0
    for request in requests:
        await router.finish_request(request)

    # block-wise (32/block) hit rate: server_of[0]=96/512, server_of[1]=32/512, server_of[2]=0/512
    another_request = CompletionRequest(model="TinyLlama",
                                        prompt=[[1000] * 500])
    dup_requests = [copy.copy(another_request) for _ in range(20)]
    another_results = [
        await router.get_next_server(req) for req in dup_requests
    ]
    dup_servers, dup_infos = zip(*another_results)
    # due to workload balancing, not all requests are sent to the same server
    # distribution follows cache hit rate
    counts = {s: 0 for s in dup_servers}
    for s in dup_servers:
        counts[s] += 1
    assert counts[server_of[0]] > counts[server_of[1]] > counts[
        server_of[2]] > 0
    dup_m = matches_by_server(dup_infos[0])
    assert dup_m[server_of[0]] == 96
    assert dup_m[server_of[1]] == 32
    assert dup_m[server_of[2]] == 0
    for req in dup_requests:
        await router.finish_request(req)

    # test router after block eviction on servers that cached req0 and req1
    # results[0] = (server_of[2], ...), results[1:] are server_of[1] and server_of[0]
    for server, info in results[1:]:
        assert server in [server_of[0], server_of[1]]
        events = [{"type": "removed", "block_hashes": info["block_hashes"][0]}]
        router._server_state[server].update_with_events(events)

    # Only server_of[2] still has cached blocks (req2)
    results = [await router.get_next_server(req) for req in reversed(requests)]
    final_servers, _ = zip(*results)
    # req2 routes to server_of[2] (full cache hit); others spread elsewhere
    assert final_servers[0] == server_of[2]
    assert len(set(final_servers)) == 3


@pytest.mark.asyncio
@pytest.mark.parametrize("api_type", ["completion", "chat"])
async def test_kv_cache_aware_router_multi_turn_conversation(
        api_type, mock_aiohttp_session):
    """Test that consecutive turns of a multi-turn conversation route to the same server due to KV cache prefix hits.

    Verifies that consecutive turns route to the same server.
    Simulates two concurrent sessions inspired by
    agentic_data/dataset_sample2000.jsonl session sess-fca58a1f44cd:
      Turn 0: 68 hash_ids (system prompt + first user input)
      Turn 1:  9 hash_ids (second user input, accumulated with turn 0)
      Turn 2:  6 hash_ids (third user input, accumulated with turn 1)

    Scaled down to 10, 3, 2 blocks for test manageability.  Each hash_id
    maps to a deterministic block of tokens (mirroring aiperf's
    HashIdRandomGenerator).  The router should prefer the server that
    already caches the conversation's prefix.
    """
    server_list = ["server1", "server2", "server3"]
    tokens_per_block = 32

    router = KvCacheAwareRouter(
        server_role=None,
        servers=server_list,
        use_tokens=False,
        max_batch_size=64,
        tokens_per_block=tokens_per_block,
    )

    # -- helpers ----------------------------------------------------------
    def hash_id_to_block(hash_id: int) -> list[int]:
        """Deterministic token block per hash_id (mirrors aiperf corpus sampling)."""
        return [(hash_id * 7 + i) % 50000 for i in range(tokens_per_block)]

    def build_tokens(hash_ids: list[int]) -> list[int]:
        tokens = []
        for hid in hash_ids:
            tokens.extend(hash_id_to_block(hid))
        # Append one extra token so the last full block is included in hashing.
        # (KvCacheManager excludes the very last token from block keys.)
        tokens.append(0)
        return tokens

    def make_request(token_ids: list[int]):
        """Create a CompletionRequest or ChatCompletionRequest with pre-tokenized IDs."""
        if api_type == "completion":
            return CompletionRequest(model="TinyLlama", prompt=[token_ids])
        else:
            # Use prompt_token_ids to skip tokenizer (no real model needed)
            return ChatCompletionRequest(
                model="TinyLlama",
                messages=[{
                    "role": "user",
                    "content": "dummy"
                }],
                prompt_token_ids=token_ids,
            )

    # -- dataset-inspired hash_ids per turn (new blocks only) -------------
    # Session A (the conversation under test)
    sess_a_turn0_hids = list(range(10))  # 10 blocks
    sess_a_turn1_hids = list(range(100, 103))  # 3 new blocks
    sess_a_turn2_hids = list(range(200, 202))  # 2 new blocks

    # Session B (competing traffic on a different server)
    sess_b_turn0_hids = list(range(500, 510))  # 10 completely different blocks

    # -- build accumulated token sequences --------------------------------
    # Turn 0: just the first turn's tokens
    sess_a_turn0_tokens = build_tokens(sess_a_turn0_hids)

    # Turn 1 accumulated: turn 0 tokens + simulated assistant reply + new user tokens
    sess_a_turn1_tokens = build_tokens(sess_a_turn0_hids + [9990, 9991] +
                                       sess_a_turn1_hids)
    # (hash_ids 9990/9991 stand in for the assistant-reply blocks)

    # Turn 2 accumulated: extends turn 1 further
    sess_a_turn2_tokens = build_tokens(sess_a_turn0_hids + [9990, 9991] +
                                       sess_a_turn1_hids + [9992, 9993] +
                                       sess_a_turn2_hids)

    sess_b_tokens = build_tokens(sess_b_turn0_hids)

    # -- Round 1: initial routing (empty caches) --------------------------
    # Route both sessions concurrently so load-balancing spreads them to
    # different servers (with equal KV cache misses, ties are broken by load).
    req_a0 = make_request(sess_a_turn0_tokens)
    server_a, info_a0 = await router.get_next_server(req_a0)
    # Do NOT finish req_a0 yet — keep its load active so session B avoids server_a

    req_b0 = make_request(sess_b_tokens)
    server_b, info_b0 = await router.get_next_server(req_b0)

    # Now finish both and populate caches
    await router.finish_request(req_a0)
    await router.finish_request(req_b0)
    router._server_state[server_a].add_blocks(info_a0["block_hashes"][0])
    router._server_state[server_b].add_blocks(info_b0["block_hashes"][0])

    # Sanity: two sessions should land on different servers
    assert server_a != server_b, "Disjoint sessions should land on different servers"

    # Verify block hashes are disjoint between sessions
    blocks_a = set(info_a0["block_hashes"][0])
    blocks_b = set(info_b0["block_hashes"][0])
    assert blocks_a.isdisjoint(
        blocks_b), "Different sessions must not share block hashes"

    # -- Round 2: turn 1 of session A (prefix extends turn 0) ------------
    req_a1 = make_request(sess_a_turn1_tokens)
    server_a1, info_a1 = await router.get_next_server(req_a1)
    await router.finish_request(req_a1)

    assert server_a1 == server_a, (
        f"Turn 1 must route to the same server as turn 0 ({server_a}) "
        f"due to KV cache prefix hit, but got {server_a1}. "
        f"Matches: {info_a1['matches']}")

    # The match count on server_a must equal the prefix overlap
    server_a_idx = list(router._server_state.keys()).index(server_a)
    expected_prefix_match = len(sess_a_turn0_hids) * tokens_per_block
    assert info_a1["matches"][server_a_idx] == expected_prefix_match, (
        f"Expected {expected_prefix_match} matched tokens on server_a, "
        f"got {info_a1['matches'][server_a_idx]}")

    # Update server_a cache with new blocks from turn 1
    router._server_state[server_a].add_blocks(info_a1["block_hashes"][0])

    # -- Round 3: turn 2 of session A (prefix extends turn 1) ------------
    req_a2 = make_request(sess_a_turn2_tokens)
    server_a2, info_a2 = await router.get_next_server(req_a2)
    await router.finish_request(req_a2)

    assert server_a2 == server_a, (
        f"Turn 2 must route to the same server as turns 0-1 ({server_a}) "
        f"due to KV cache prefix hit, but got {server_a2}. "
        f"Matches: {info_a2['matches']}")

    # Turn 2 should match all of turn 0 + turn 1 prefix blocks
    expected_full_match = (
        len(sess_a_turn0_hids) + 2 +
        len(sess_a_turn1_hids)  # turn0 + reply + turn1
    ) * tokens_per_block
    assert info_a2["matches"][server_a_idx] == expected_full_match, (
        f"Expected {expected_full_match} matched tokens on turn 2, "
        f"got {info_a2['matches'][server_a_idx]}")

    # -- Verify session B still routes to its own server ------------------
    req_b1 = make_request(sess_b_tokens)
    server_b1, info_b1 = await router.get_next_server(req_b1)
    await router.finish_request(req_b1)

    assert server_b1 == server_b, (
        f"Session B should route to its original server ({server_b}), "
        f"but got {server_b1}")


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


# ── ConversationRouter tests ──


def _make_request(conversation_id=None, prompt="the " * 100):
    params = DisaggregatedParams(request_type="context_only")
    conversation_params = None
    if conversation_id is not None:
        conversation_params = ConversationParams(
            conversation_id=conversation_id)
    return CompletionRequest(model="TinyLlama",
                             prompt=[prompt],
                             disaggregated_params=params,
                             conversation_params=conversation_params)


@pytest.mark.asyncio
async def test_conversation_router_session_affinity_and_fallbacks():
    """Session affinity, exclude-server override, and server-removal reroute."""
    servers = ["server1", "server2", "server3"]
    router = ConversationRouter(server_role=None, servers=servers)

    # Affinity: same conversation_id → same server
    req = _make_request(conversation_id="sess-A")
    first, _ = await router.get_next_server(req)
    await router.finish_request(req)
    for _ in range(3):
        req = _make_request(conversation_id="sess-A")
        s, _ = await router.get_next_server(req)
        assert s == first
        await router.finish_request(req)

    # Exclude: affinity overridden when mapped server is excluded
    req = _make_request(conversation_id="sess-A")
    s, _ = await router.get_next_server(req, exclude_server=first)
    assert s != first
    await router.finish_request(req)

    # Server removal: session re-routes to surviving server
    req = _make_request(conversation_id="sess-B")
    orig, _ = await router.get_next_server(req)
    await router.finish_request(req)
    await router.remove_server(orig)
    req = _make_request(conversation_id="sess-B")
    s, _ = await router.get_next_server(req)
    assert s != orig and s in router.servers
    await router.finish_request(req)


@pytest.mark.asyncio
async def test_conversation_router_load_balancing():
    """New sessions with distinct prompts are load-balanced across servers."""
    servers = ["server1", "server2", "server3"]
    router = ConversationRouter(server_role=None, servers=servers)

    assigned, reqs = [], []
    for i in range(3):
        req = _make_request(prompt=f"unique topic {i} " * 50)
        s, _ = await router.get_next_server(req)
        assigned.append(s)
        reqs.append(req)
    assert sorted(assigned) == sorted(servers)
    for req in reqs:
        await router.finish_request(req)

    # Session affinity survives interleaved non-session requests
    req_x = _make_request(conversation_id="sess-X", prompt="topic X")
    sx, _ = await router.get_next_server(req_x)
    await router.finish_request(req_x)
    req_x2 = _make_request(conversation_id="sess-X", prompt="topic X turn 2")
    s, _ = await router.get_next_server(req_x2)
    assert s == sx
    await router.finish_request(req_x2)


@pytest.mark.asyncio
async def test_conversation_router_prefix_and_token_id_paths():
    """Implicit prefix matching (text and token-ID paths) and hash_skip_count."""
    servers = ["server1", "server2", "server3"]
    base = "x" * 2000

    # Text prefix matching: multi-turn without conversation_id
    router = ConversationRouter(server_role=None, servers=servers)
    req1 = _make_request(prompt=base)
    s1, _ = await router.get_next_server(req1)
    await router.finish_request(req1)
    for ext in ["y" * 50, "y" * 50 + "z" * 50]:
        req = _make_request(prompt=base + ext)
        s, _ = await router.get_next_server(req)
        assert s == s1, "Extended prompt should prefix-match turn 1"
        await router.finish_request(req)

    # Token-ID path: CompletionRequest with list[list[int]]
    router2 = ConversationRouter(server_role=None, servers=servers)
    base_ids = [1000] * 2000
    dp = DisaggregatedParams(request_type="context_only")
    req_t1 = CompletionRequest(model="TinyLlama",
                               prompt=[base_ids + [0]],
                               disaggregated_params=dp)
    st1, _ = await router2.get_next_server(req_t1)
    await router2.finish_request(req_t1)

    req_t2 = CompletionRequest(model="TinyLlama",
                               prompt=[base_ids + [2000] * 50 + [0]],
                               disaggregated_params=dp)
    st2, _ = await router2.get_next_server(req_t2)
    assert st2 == st1, "Token-ID prefix should match"
    await router2.finish_request(req_t2)

    # ChatCompletionRequest with prompt_token_ids
    req_t3 = ChatCompletionRequest(model="TinyLlama",
                                   messages=[{
                                       "role": "user",
                                       "content": "dummy"
                                   }],
                                   prompt_token_ids=base_ids + [2000] * 50 +
                                   [3000] * 50 + [0],
                                   disaggregated_params=dp)
    st3, _ = await router2.get_next_server(req_t3)
    assert st3 == st1, "ChatCompletion token-ID path should match"
    await router2.finish_request(req_t3)


@pytest.mark.asyncio
async def test_conversation_router_hash_skip_count():
    """hash_skip_count strips shared system-prompt prefix.

    With tokens_per_block=128 (chars, via code-point path):
    - sys_prompt "S"*2000 → ~15 blocks, unique "A"*500 → ~4 blocks
    - Total ~19 blocks, shared ratio ~15/19 ≈ 0.79 > 0.75 threshold
    Without skip the shared prefix triggers a false implicit match.
    With skip (hash_skip_count=400 → strips 400*5=2000 chars), the
    shared prefix is removed and the remaining content differs.
    """
    servers = ["server1", "server2", "server3"]
    sys_prompt = "S" * 2000

    # Without skip: shared prefix causes false match
    r1 = ConversationRouter(server_role=None, servers=servers)
    req_a = _make_request(prompt=sys_prompt + "A" * 500)
    sa, _ = await r1.get_next_server(req_a)
    await r1.finish_request(req_a)
    req_b = _make_request(prompt=sys_prompt + "B" * 500)
    sb, _ = await r1.get_next_server(req_b)
    await r1.finish_request(req_b)
    assert sb == sa, "Without skip, shared prefix causes false match"

    # With skip: different content after prefix → no match
    r2 = ConversationRouter(server_role=None,
                            servers=servers,
                            hash_skip_count=400)
    req_a2 = _make_request(prompt=sys_prompt + "A" * 500)
    sa2, _ = await r2.get_next_server(req_a2)
    # Keep in-flight so LB prefers a different server
    req_b2 = _make_request(prompt=sys_prompt + "B" * 500)
    sb2, _ = await r2.get_next_server(req_b2)
    await r2.finish_request(req_a2)
    await r2.finish_request(req_b2)
    assert sb2 != sa2, "With skip, different content should not match"


@pytest.mark.asyncio
async def test_kv_cache_aware_router_polls_kv_cache_events(
        servers, mock_aiohttp_session):
    """finish_request must POST /kv_cache_events and apply returned events."""
    # Reconfigure the mock session to return a stored-block event.
    stored_event = [{"type": "stored", "blocks": [{"block_hash": 99999}]}]
    mock_response = mock.AsyncMock()
    mock_response.json = mock.AsyncMock(return_value=stored_event)
    mock_ctx = mock.AsyncMock()
    mock_ctx.__aenter__ = mock.AsyncMock(return_value=mock_response)
    mock_ctx.__aexit__ = mock.AsyncMock(return_value=False)
    mock_aiohttp_session.post = mock.MagicMock(return_value=mock_ctx)

    router = KvCacheAwareRouter(
        server_role=None,
        servers=servers,
        use_tokens=False,
        max_batch_size=32,
        tokens_per_block=32,
    )

    request = CompletionRequest(model="TinyLlama", prompt=[[1000] * 100])
    server, _ = await router.get_next_server(request)

    mock_aiohttp_session.post.reset_mock()
    await router.finish_request(request)
    # poll_and_update runs as a background task; yield to let it complete
    await asyncio.sleep(0)

    # /kv_cache_events was queried on the correct server
    mock_aiohttp_session.post.assert_called_once_with("http://" + server +
                                                      "/kv_cache_events")

    # Returned events were applied to the server state
    assert 99999 in router._server_state[server]._kv_cache_block_table


def test_create_router_conversation():
    router = create_router(RouterConfig(type="conversation"),
                           ["server1", "server2"])
    assert isinstance(router, ConversationRouter)


def test_block_hash_mixin_routes_through_transformers_tokenizer():
    """``BlockHashMixin._get_tokenizer`` must call ``TransformersTokenizer.from_pretrained``.

    Routing through ``TransformersTokenizer`` is what lets block-hash KV-cache
    routing inherit the post-load fixes (``maybe_fix_byte_level_tokenizer`` for
    DeepSeek-V3 Metaspace, ``_fallback_to_fast_tokenizer`` for DeepSeek-V3.2
    on transformers >= 5.x). Without this routing, ``trtllm-serve`` would
    tokenize prompts differently from the rest of TRT-LLM when computing
    block hashes for cache hits.
    """

    class _Probe(BlockHashMixin):
        pass

    probe = _Probe()
    probe._init_block_hashing()

    inner = mock.MagicMock()
    wrapper = mock.MagicMock(tokenizer=inner)
    with mock.patch(
            "tensorrt_llm.tokenizer.TransformersTokenizer.from_pretrained",
            return_value=wrapper) as routed:
        out = probe._get_tokenizer("dummy/model")

    routed.assert_called_once_with("dummy/model", trust_remote_code=True)
    # The cached tokenizer must be the raw HF tokenizer used by _tokenize,
    # not the TransformersTokenizer wrapper.
    assert out is inner
    assert probe._tokenizers["dummy/model"] is inner


@pytest.mark.asyncio
async def test_finish_request_forwards_explicit_session(servers):
    """A caller-provided session is forwarded to the kv-cache-events poll.

    finish_request(..., session=) threads the session through to
    KvCacheAwareServerState.poll_and_update (disagg router session fix), which
    must use the provided session for the kv_cache_events poll rather than its
    own self._session.
    """
    router = KvCacheAwareRouter(server_role=None,
                                servers=servers,
                                use_tokens=False,
                                max_batch_size=32,
                                tokens_per_block=32)
    request = CompletionRequest(model="TinyLlama", prompt=[[1000] * 100])
    await router.get_next_server(request)

    # Build the explicit session inline WITHOUT spec=aiohttp.ClientSession: the
    # autouse fixture has already patched aiohttp.ClientSession to a MagicMock, so
    # _make_mock_aiohttp_session's spec= would raise "Cannot spec a Mock object".
    mock_response = mock.AsyncMock()
    mock_response.json = mock.AsyncMock(return_value=[])
    mock_ctx = mock.AsyncMock()
    mock_ctx.__aenter__ = mock.AsyncMock(return_value=mock_response)
    mock_ctx.__aexit__ = mock.AsyncMock(return_value=False)
    explicit_session = mock.MagicMock()
    explicit_session.post = mock.MagicMock(return_value=mock_ctx)

    await router.finish_request(request, session=explicit_session)
    await asyncio.sleep(0)

    # The explicitly provided session is the one used to poll kv-cache events.
    explicit_session.post.assert_called_once()
    call = explicit_session.post.call_args
    posted_url = call.args[0] if call.args else call.kwargs.get("url", "")
    assert "kv_cache_events" in posted_url


# ---------------------------------------------------------------------------
# _tokenize: tools + chat_template_kwargs forwarding (PR #13232)
# ---------------------------------------------------------------------------


def _get_weather_tool() -> ChatCompletionToolsParam:
    """Build a sample tool definition matching the OpenAI schema."""
    return ChatCompletionToolsParam(function=FunctionDefinition(
        name="get_current_weather",
        description="Get the current weather for a city.",
        parameters={
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "City name"
                }
            },
            "required": ["city"],
        },
    ))


def _mock_tokenizer(token_ids=None):
    """Return a mock tokenizer with a recorded apply_chat_template.

    ``apply_chat_template`` records its kwargs and returns the supplied
    token id list.
    """
    tok = mock.MagicMock()
    tok.apply_chat_template.return_value = token_ids or [1, 2, 3, 4, 5]
    return tok


def test_router_model_type_uses_checkpoint_config(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text('{"model_type": "gpt_oss"}',
                                          encoding="utf-8")
    router = KvCacheAwareRouter(server_role=None,
                                servers=["server1"],
                                model_path=str(tmp_path))

    assert router._get_model_type() == "gpt_oss"


@pytest.mark.asyncio
async def test_gpt_oss_router_tokens_match_chat_harmony_server_input() -> None:
    """KV-cache routing must hash the same Harmony tokens used by the server."""
    from tensorrt_llm.serve.openai_server import OpenAIServer

    router = KvCacheAwareRouter(server_role=None,
                                servers=["server1"],
                                use_tokens=False,
                                max_batch_size=32,
                                tokens_per_block=32,
                                model_path="/models/gpt-oss-checkpoint")
    router_tokenizer = _mock_tokenizer(token_ids=[900, 901, 902])
    harmony_tokens = [100, 101, 102, 103]
    harmony_adapter = mock.MagicMock()
    harmony_adapter.openai_to_harmony_tokens.return_value = harmony_tokens
    harmony_adapter.get_stop_tokens.return_value = [42]
    promise = mock.MagicMock()
    promise.prompt_token_ids = []

    request = ChatCompletionRequest(
        model="my-model",
        messages=[{
            "role": "developer",
            "content": "Use tools when useful."
        }, {
            "role": "user",
            "content": "weather in Paris?"
        }],
        tools=[_get_weather_tool()],
        tool_choice="auto",
        reasoning_effort="medium",
        stream=True,
        max_completion_tokens=1,
    )
    router_request = copy.deepcopy(request)
    server_request = copy.deepcopy(request)

    server = OpenAIServer.__new__(OpenAIServer)
    server.allow_request_chat_template = False
    server.await_disconnected = mock.AsyncMock()
    server.generator = SimpleNamespace(
        args=SimpleNamespace(num_postprocess_workers=0),
        generate_async=mock.MagicMock(return_value=promise),
    )
    server.harmony_adapter = harmony_adapter
    server.model_config = SimpleNamespace(vocab_size=1000)
    server.tokenizer = SimpleNamespace(tokenizer=SimpleNamespace(
        vocab_size=1000))

    with mock.patch.object(
            router, "_get_tokenizer",
            return_value=router_tokenizer), mock.patch(
                "tensorrt_llm.serve.harmony_adapter."
                "get_harmony_adapter",
                return_value=harmony_adapter), mock.patch(
                    "tensorrt_llm.serve.router_utils."
                    "resolve_model_type_from_config",
                    return_value="gpt_oss") as resolve_model_type:
        router_token_ids = router._tokenize(router_request)[0]
        await server.chat_harmony(server_request, raw_request=None)

    server_token_ids = server.generator.generate_async.call_args.kwargs[
        "inputs"]
    assert router_token_ids == server_token_ids
    assert router_request.prompt_token_ids == harmony_tokens
    first_call, second_call = harmony_adapter.openai_to_harmony_tokens.call_args_list
    assert first_call.args == second_call.args
    assert first_call.kwargs == second_call.kwargs
    resolve_model_type.assert_called_once_with("/models/gpt-oss-checkpoint")
    router_tokenizer.apply_chat_template.assert_not_called()


def test_gpt_oss_router_respects_disable_harmony_adapter(
        monkeypatch: pytest.MonkeyPatch) -> None:
    """Router follows the same DISABLE_HARMONY_ADAPTER gate as the server."""
    monkeypatch.setenv("DISABLE_HARMONY_ADAPTER", "1")
    router = KvCacheAwareRouter(server_role=None,
                                servers=["server1"],
                                use_tokens=False,
                                max_batch_size=32,
                                tokens_per_block=32,
                                model_path="/models/gpt-oss-checkpoint")
    router_tokenizer = _mock_tokenizer(token_ids=[900, 901, 902])
    harmony_adapter = mock.MagicMock()

    request = ChatCompletionRequest(
        model="openai/gpt-oss-20b",
        messages=[{
            "role": "user",
            "content": "weather in Paris?"
        }],
        tools=[_get_weather_tool()],
    )

    with mock.patch.object(
            router, "_get_tokenizer",
            return_value=router_tokenizer), mock.patch(
                "tensorrt_llm.serve.harmony_adapter."
                "get_harmony_adapter",
                return_value=harmony_adapter), mock.patch(
                    "tensorrt_llm.serve.router_utils."
                    "resolve_model_type_from_config",
                    side_effect=AssertionError(
                        "disabled Harmony must not load model config")):
        assert router._tokenize(request) == [[900, 901, 902]]

    harmony_adapter.openai_to_harmony_tokens.assert_not_called()
    router_tokenizer.apply_chat_template.assert_called_once()


@pytest.mark.asyncio
async def test_chat_harmony_preserves_original_tool_conversion_error() -> None:
    """Harmony diagnostics must not rerun the conversion that already failed."""
    from tensorrt_llm.serve.openai_server import OpenAIServer

    original_error = RuntimeError("original tool conversion failure")
    diagnostic_error = RuntimeError("diagnostic tool conversion failure")

    class FailingTool:

        def __init__(self) -> None:
            self.calls = 0

        def model_dump(self) -> dict[str, object]:
            self.calls += 1
            if self.calls == 1:
                raise original_error
            raise diagnostic_error

    failing_tool = FailingTool()
    request = ChatCompletionRequest(
        model="my-model",
        messages=[{
            "role": "user",
            "content": "weather in Paris?"
        }],
    )
    object.__setattr__(request, "tools", [failing_tool])

    server = OpenAIServer.__new__(OpenAIServer)
    server.allow_request_chat_template = False
    server.harmony_adapter = mock.MagicMock()
    server.create_error_response = mock.MagicMock(
        return_value=str(original_error))

    response = await server.chat_harmony(request, raw_request=None)

    assert response == str(original_error)
    assert failing_tool.calls == 1
    server.create_error_response.assert_called_once_with(
        message=str(original_error), err_type="internal_error")


@pytest.mark.parametrize("router_class",
                         [KvCacheAwareRouter, ConversationRouter])
def test_tokenize_forwards_tools_and_chat_template_kwargs(router_class):
    """Regression test for PR #13232.

    ``BlockHashMixin._tokenize`` must forward the request's ``tools`` (as a
    list of dicts) and ``chat_template_kwargs`` to
    ``tokenizer.apply_chat_template``. Without this, custom tokenizers that
    render tool schemas into the prompt (e.g. DeepSeek-V3.2) produce
    truncated token ids, breaking cache-aware routing decisions and the
    ``prompt_token_ids`` handed to the worker downstream.
    """
    router = router_class(server_role=None,
                          servers=["server1"],
                          use_tokens=False,
                          max_batch_size=32,
                          tokens_per_block=32)

    tok = _mock_tokenizer()
    documents = [{"title": "Paris", "text": "Paris is in France."}]
    chat_template = "{% for message in messages %}{{ message.content }}{% endfor %}"
    with mock.patch.object(router, "_get_tokenizer", return_value=tok):
        req = ChatCompletionRequest(
            model="TinyLlama",
            messages=[{
                "role": "user",
                "content": "what's the weather in Paris?"
            }],
            tools=[_get_weather_tool()],
            documents=documents,
            chat_template=chat_template,
            chat_template_kwargs={"thinking": True},
        )
        router._tokenize(req)

    tok.apply_chat_template.assert_called_once()
    kwargs = tok.apply_chat_template.call_args.kwargs
    # tools must be forwarded as a list of dicts (model_dump), not the
    # Pydantic objects themselves.
    assert isinstance(kwargs["tools"], list) and len(kwargs["tools"]) == 1
    tool_dict = kwargs["tools"][0]
    assert isinstance(tool_dict, dict)
    assert tool_dict["type"] == "function"
    assert tool_dict["function"]["name"] == "get_current_weather"
    assert "parameters" in tool_dict["function"]
    # chat_template_kwargs must be forwarded as **kwargs (not nested).
    assert kwargs.get("thinking") is True
    assert kwargs["documents"] == documents
    assert kwargs["chat_template"] == chat_template


@pytest.mark.parametrize("router_class",
                         [KvCacheAwareRouter, ConversationRouter])
def test_tokenize_without_tools_passes_none(router_class):
    """Bare chat request: no tools, no chat_template_kwargs.

    ``apply_chat_template`` still runs but receives ``tools=None`` and no
    extra keyword arguments.
    """
    router = router_class(server_role=None,
                          servers=["server1"],
                          use_tokens=False,
                          max_batch_size=32,
                          tokens_per_block=32)

    tok = _mock_tokenizer()
    with mock.patch.object(router, "_get_tokenizer", return_value=tok):
        req = ChatCompletionRequest(model="TinyLlama",
                                    messages=[{
                                        "role": "user",
                                        "content": "hello"
                                    }])
        router._tokenize(req)

    tok.apply_chat_template.assert_called_once()
    kwargs = tok.apply_chat_template.call_args.kwargs
    assert kwargs["tools"] is None
    assert "thinking" not in kwargs
    # messages and add_generation_prompt must still flow through unchanged.
    assert kwargs["tokenize"] is False


def test_tokenize_preserves_empty_tools_list():
    """Preserve empty tools list distinct from ``None``.

    ``tools=[]`` is semantically distinct from ``tools=None``; preserve it
    so the router's call matches what the worker's own
    ``apply_chat_template`` would pass (see ``serve/openai_server.py``
    tool_dicts assignment).
    """
    router = KvCacheAwareRouter(server_role=None,
                                servers=["server1"],
                                use_tokens=False,
                                max_batch_size=32,
                                tokens_per_block=32)

    tok = _mock_tokenizer()
    with mock.patch.object(router, "_get_tokenizer", return_value=tok):
        req = ChatCompletionRequest(model="TinyLlama",
                                    messages=[{
                                        "role": "user",
                                        "content": "hi"
                                    }],
                                    tools=[])
        router._tokenize(req)

    kwargs = tok.apply_chat_template.call_args.kwargs
    assert kwargs["tools"] == []


def test_tokenize_skipped_when_prompt_token_ids_already_set():
    """Skip tokenization when ``prompt_token_ids`` is already populated.

    When the caller pre-tokenizes (``prompt_token_ids`` set), the router
    must not invoke ``apply_chat_template`` at all — the cached token ids
    are returned as-is.
    """
    router = KvCacheAwareRouter(server_role=None,
                                servers=["server1"],
                                use_tokens=False,
                                max_batch_size=32,
                                tokens_per_block=32)

    tok = _mock_tokenizer()
    with mock.patch.object(router, "_get_tokenizer",
                           return_value=tok) as get_tok:
        req = ChatCompletionRequest(
            model="TinyLlama",
            messages=[{
                "role": "user",
                "content": "irrelevant"
            }],
            tools=[_get_weather_tool()],
            chat_template_kwargs={"thinking": True},
            prompt_token_ids=[10, 20, 30],
        )
        out = router._tokenize(req)

    assert out == [[10, 20, 30]]
    get_tok.assert_not_called()
    tok.apply_chat_template.assert_not_called()


class _PrefixCacheFakeTokenizer:

    _SPECIAL = {"<bos>": 1, "<sys>": 2, "<user>": 3, "<asst>": 4, "<eot>": 5}

    def apply_chat_template(self,
                            messages,
                            add_generation_prompt=False,
                            tokenize=False,
                            return_dict=False,
                            tools=None,
                            **kwargs):
        tag = {"system": "<sys>", "user": "<user>", "assistant": "<asst>"}
        parts = ["<bos>"]
        for m in messages:
            role = m["role"] if isinstance(m, dict) else m.role
            content = m["content"] if isinstance(m, dict) else m.content
            parts.append(tag.get(role, "<user>") + str(content) + "<eot>")
        if add_generation_prompt:
            parts.append("<asst>")
        return "".join(parts)

    def encode(self, text, add_special_tokens=False):
        ids = []
        i = 0
        n = len(text)
        while i < n:
            special = None
            for token, tid in self._SPECIAL.items():
                if text.startswith(token, i):
                    special = (token, tid)
                    break
            if special is not None:
                ids.append(special[1])
                i += len(special[0])
                continue
            j = i
            while j < n and not any(
                    text.startswith(token, j) for token in self._SPECIAL):
                j += 1
            run = text[i:j]
            ids.append(9000 + len(run))
            ids.extend(ord(c) for c in run)
            i = j
        return ids


def _grow_conversation():
    convo = [{
        "role": "system",
        "content": "SYS " * 40
    }, {
        "role": "user",
        "content": "hello " * 20
    }]
    yield [dict(m) for m in convo]
    for turn in range(4):
        convo.append({"role": "assistant", "content": f"resp{turn} " * 30})
        convo.append({"role": "user", "content": f"q{turn} " * 12})
        yield [dict(m) for m in convo]


def test_prefix_cache_tokenize_matches_full_encode(servers):
    router = KvCacheAwareRouter(server_role=None,
                                servers=servers,
                                tokens_per_block=4)
    tok = _PrefixCacheFakeTokenizer()
    with mock.patch.object(router, "_get_tokenizer", return_value=tok):
        for convo in _grow_conversation():
            req = ChatCompletionRequest(model="mock", messages=convo)
            rendered = tok.apply_chat_template(
                convo,
                add_generation_prompt=req.add_generation_prompt,
                tokenize=False)
            reference = tok.encode(rendered, add_special_tokens=False)
            result = router._tokenize(req)[0]
            assert result == reference
            assert req.prompt_token_ids == reference


def test_prefix_cache_tokenize_uses_canonical_encoding(servers):
    router = KvCacheAwareRouter(server_role=None,
                                servers=servers,
                                tokens_per_block=4)
    tok = _PrefixCacheFakeTokenizer()
    encoded_lengths = []
    real_encode = tok.encode

    def _recording_encode(text, add_special_tokens=False):
        encoded_lengths.append(len(text))
        return real_encode(text, add_special_tokens=add_special_tokens)

    tok.encode = _recording_encode
    with mock.patch.object(router, "_get_tokenizer", return_value=tok):
        for convo in _grow_conversation():
            encoded_lengths.clear()
            req = ChatCompletionRequest(model="mock", messages=convo)
            rendered = tok.apply_chat_template(
                convo,
                add_generation_prompt=req.add_generation_prompt,
                tokenize=False)
            router._tokenize(req)
            assert encoded_lengths
            assert encoded_lengths == [len(rendered)]


def test_prefix_cache_tokenize_falls_back_on_divergent_prefix(servers):
    router = KvCacheAwareRouter(server_role=None,
                                servers=servers,
                                tokens_per_block=4)
    tok = _PrefixCacheFakeTokenizer()
    convo_a = [{
        "role": "system",
        "content": "A " * 30
    }, {
        "role": "user",
        "content": "alpha"
    }]
    convo_b = [{
        "role": "system",
        "content": "B " * 30
    }, {
        "role": "user",
        "content": "beta"
    }]
    with mock.patch.object(router, "_get_tokenizer", return_value=tok):
        for convo in (convo_a, convo_b, convo_a):
            req = ChatCompletionRequest(model="mock",
                                        messages=[dict(m) for m in convo])
            rendered = tok.apply_chat_template(
                [dict(m) for m in convo],
                add_generation_prompt=req.add_generation_prompt,
                tokenize=False)
            reference = tok.encode(rendered, add_special_tokens=False)
            assert router._tokenize(req)[0] == reference


@pytest.mark.asyncio
async def test_conversation_affinity_pins_across_load(servers):
    router = KvCacheAwareRouter(server_role=None,
                                servers=servers,
                                tokens_per_block=4,
                                load_weight=1.0)
    tok = mock.MagicMock()
    tok.apply_chat_template.return_value = [1, 2, 3, 4, 5, 6, 7, 8]
    convo = [{
        "role": "system",
        "content": "S"
    }, {
        "role": "user",
        "content": "u1"
    }]
    with mock.patch.object(router, "_get_tokenizer", return_value=tok):
        req1 = ChatCompletionRequest(model="m",
                                     messages=[dict(m) for m in convo])
        home, _ = await router.get_next_server(req1)
        # Saturate the home server so a fresh request would avoid it.
        for _ in range(64):
            await router._server_state[home].increment_load(
                ChatCompletionRequest(model="m",
                                      messages=[{
                                          "role": "user",
                                          "content": "x"
                                      }]))
        # Same conversation (same first-2-message content) -> pinned to home
        # even though the score now favours the idle servers.
        req2 = ChatCompletionRequest(model="m",
                                     messages=[dict(m) for m in convo] +
                                     [{
                                         "role": "assistant",
                                         "content": "a1"
                                     }, {
                                         "role": "user",
                                         "content": "u2"
                                     }])
        assert (await router.get_next_server(req2))[0] == home
        # A different conversation is free to avoid the saturated home.
        req3 = ChatCompletionRequest(model="m",
                                     messages=[{
                                         "role": "system",
                                         "content": "OTHER"
                                     }, {
                                         "role": "user",
                                         "content": "v1"
                                     }])
        assert (await router.get_next_server(req3))[0] != home
