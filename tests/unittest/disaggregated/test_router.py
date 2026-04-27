import asyncio
import copy
import threading
from unittest import mock

import aiohttp
import pytest

from tensorrt_llm.llmapi.disagg_utils import RouterConfig
from tensorrt_llm.serve.openai_protocol import (ChatCompletionRequest,
                                                CompletionRequest,
                                                DisaggregatedParams)
from tensorrt_llm.serve.router import (ConversationRouter, KvCacheAwareRouter,
                                       LoadBalancingRouter, RoundRobinRouter,
                                       create_router)


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
                                tokens_per_block=32)
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
    params = DisaggregatedParams(request_type="context_only",
                                 conversation_id=conversation_id)
    return CompletionRequest(model="TinyLlama",
                             prompt=[prompt],
                             disaggregated_params=params)


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
