import copy

import pytest

from tensorrt_llm.llmapi.disagg_utils import RouterConfig
from tensorrt_llm.serve.openai_protocol import (ChatCompletionRequest,
                                                CompletionRequest,
                                                DisaggregatedParams)
from tensorrt_llm.serve.router import (KvCacheAwareRouter, LoadBalancingRouter,
                                       RoundRobinRouter, create_router)


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
    router = RoundRobinRouter(servers)
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
    router = LoadBalancingRouter(servers, use_tokens=False)
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
    router = LoadBalancingRouter(servers, use_tokens=True)
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
    router = LoadBalancingRouter(servers, use_tokens=True)
    requests = request.getfixturevalue(requests_fixture)

    # Should throw an error if trying to use tokens load balancing with gen-only requests or chat completion requests
    with pytest.raises(ValueError):
        await router.get_next_server(requests[0])


@pytest.mark.asyncio
async def test_kv_cache_aware_router(servers):
    # create tokenized requests to skip tokenization
    requests = [
        CompletionRequest(model="TinyLlama", prompt=[[1000] * 100]),
        CompletionRequest(model="TinyLlama",
                          prompt=[[1000] * 50 + [1001] * 150]),
        CompletionRequest(model="TinyLlama", prompt=[[1002] * 300]),
    ]

    router = KvCacheAwareRouter(servers, use_tokens=False, max_batch_size=32)
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
