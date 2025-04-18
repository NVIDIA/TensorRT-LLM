import pytest

from tensorrt_llm.serve.openai_protocol import (ChatCompletionRequest,
                                                CompletionRequest,
                                                DisaggregatedParams)
from tensorrt_llm.serve.router import (LoadBalancingRouter, RoundRobinRouter,
                                       create_router)


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
    server_sequence = [
        await router.get_next_server(req) for req in context_requests
    ]
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

    server = await router.get_next_server(requests[0])
    assert server == "server1"
    server = await router.get_next_server(requests[1])
    assert server == "server2"
    server = await router.get_next_server(requests[2])
    assert server == "server3"

    # Similulate terminating 3rd request (on server 3)
    await router.finish_request(requests[2])

    # Now server3 is least loaded
    server = await router.get_next_server(requests[3])
    assert server == "server3"

    # Simulate terminating 4th request (on server 3)
    await router.finish_request(requests[1])

    # Now server2 is least loaded
    server = await router.get_next_server(requests[4])
    assert server == "server2"


@pytest.mark.asyncio
@pytest.mark.parametrize("requests_fixture", ["context_requests"])
async def test_tokens_balancing_router(servers, requests_fixture, request):
    router = LoadBalancingRouter(servers, use_tokens=True)
    requests = request.getfixturevalue(requests_fixture)

    server_sequence = [await router.get_next_server(req) for req in requests]
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
    server = await router.get_next_server(requests[4])

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


def test_create_router(servers):
    round_robin_router = create_router("round_robin", servers)
    assert isinstance(round_robin_router, RoundRobinRouter)

    requests_load_balancing_router = create_router("requests_load_balancing",
                                                   servers)
    assert isinstance(requests_load_balancing_router, LoadBalancingRouter)
    assert not requests_load_balancing_router._use_tokens

    tokens_load_balancing_router = create_router("tokens_load_balancing",
                                                 servers)
    assert isinstance(tokens_load_balancing_router, LoadBalancingRouter)
    assert tokens_load_balancing_router._use_tokens

    with pytest.raises(ValueError):
        create_router("unsupported_router", servers)
