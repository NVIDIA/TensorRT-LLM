# Copyright (c) 2025-2026, NVIDIA CORPORATION.
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

import asyncio
import re
from collections.abc import AsyncGenerator
from contextlib import AsyncExitStack
from types import SimpleNamespace
from typing import Any
from unittest import mock
from unittest.mock import AsyncMock

import aiohttp
import msgspec
import pytest
import torch
from aiohttp import web
from aiohttp.test_utils import TestServer
from openai.types.responses import ResponseErrorEvent, ResponseFailedEvent, ResponseStatus
from starlette.requests import Request

from tensorrt_llm._torch.pyexecutor.scheduler.adp_router import (
    ConversationAwareADPRouter,
    RankState,
)
from tensorrt_llm._utils import get_global_steady_clock_now_in_seconds
from tensorrt_llm.disaggregated_params import DisaggregatedParams as LlmDisaggregatedParams
from tensorrt_llm.executor.result import Logprob
from tensorrt_llm.llmapi.disagg_utils import (
    ConditionalDisaggConfig,
    DisaggClusterConfig,
    DisaggServerConfig,
    MinimalInstances,
    ServerRole,
)
from tensorrt_llm.serve.conversation_id import get_request_routing_id
from tensorrt_llm.serve.disagg_auto_scaling import DisaggClusterManager, WorkerInfo
from tensorrt_llm.serve.disagg_coordinator import DisaggCoordinatorService
from tensorrt_llm.serve.openai_client import OpenAIHttpClient
from tensorrt_llm.serve.openai_disagg_server import OpenAIDisaggServer
from tensorrt_llm.serve.openai_disagg_service import OpenAIDisaggregatedService, _SseErrorObserver
from tensorrt_llm.serve.openai_protocol import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
    ChatMessage,
    CompletionRequest,
    CompletionResponse,
    CompletionResponseChoice,
    ConversationParams,
    DisaggregatedParams,
    DisaggScheduleStyle,
    PromptTokensDetails,
    ResponsesRequest,
    ResponsesResponse,
    UsageInfo,
    _deserialize_first_gen_log_probs,
    _deserialize_first_gen_logits,
    _serialize_first_gen_log_probs,
    _serialize_first_gen_logits,
    to_llm_disaggregated_params,
)
from tensorrt_llm.serve.openai_server import OpenAIServer
from tensorrt_llm.serve.postprocess_handlers import (
    ChatPostprocArgs,
    CompletionPostprocArgs,
    chat_response_post_processor,
    completion_response_post_processor,
)
from tensorrt_llm.serve.responses_utils import _format_sse_event
from tensorrt_llm.serve.router import KvCacheAwareRouter, Router

pytestmark = pytest.mark.cpu_only


def _client_factory(*_args, **_kwargs):
    return AsyncMock()


def _make_service(schedule_style: str) -> OpenAIDisaggregatedService:
    config = DisaggServerConfig(server_configs=[], schedule_style=schedule_style)
    # The coordinator builds its own (empty) routers from config; override them
    # with mocks so tests can stub placement / readiness directly.
    cluster = DisaggCoordinatorService(config, client_factory=_client_factory)
    ctx_router = AsyncMock(spec=Router)
    gen_router = AsyncMock(spec=Router)
    cluster._ctx_router = ctx_router
    cluster._gen_router = gen_router
    service = OpenAIDisaggregatedService(config, cluster, client_factory=_client_factory)
    # Convenience handles for tests that stub placement / readiness directly.
    service._ctx_router = ctx_router
    service._gen_router = gen_router
    return service


@pytest.mark.asyncio
async def test_conditional_disagg_uses_selected_server_match_length():
    service = _make_service("context_first")
    service._config.conditional_disagg_config = ConditionalDisaggConfig(max_local_prefill_length=32)
    router = KvCacheAwareRouter(server_role=ServerRole.GENERATION, servers=[])
    router.get_next_server = AsyncMock(
        return_value=(
            "gen:8000",
            {
                "match_length": 64,
                "num_tokens": 96,
            },
        )
    )
    service._gen_router = router
    request = CompletionRequest(model="model", prompt=[1] * 96)

    server, need_context = await service._check_conditional_disagg(request, 123)

    assert server == "gen:8000"
    assert need_context is False


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("remote_tail", "num_tokens", "expected_need_context"),
    [
        (False, 32, True),
        (True, 32, False),
        (True, 33, True),
    ],
)
async def test_conditional_disagg_remote_tail_allows_cold_local_prefill(
    remote_tail, num_tokens, expected_need_context
):
    service = _make_service("context_first")
    service._config.bounded_replay_on_generation = remote_tail
    service._config.conditional_disagg_config = ConditionalDisaggConfig(max_local_prefill_length=32)
    router = KvCacheAwareRouter(server_role=ServerRole.GENERATION, servers=[])
    router.get_next_server = AsyncMock(
        return_value=(
            "gen:8000",
            {
                "match_length": 0,
                "num_tokens": num_tokens,
            },
        )
    )
    service._gen_router = router
    request = CompletionRequest(model="model", prompt=[1] * num_tokens)

    server, need_context = await service._check_conditional_disagg(request, 123)

    assert server == "gen:8000"
    assert need_context is expected_need_context


@pytest.mark.asyncio
async def test_remote_tail_cold_local_prefill_skips_context_worker(monkeypatch):
    monkeypatch.setenv("TRTLLM_V41_DECODER_BOUNDED_REPLAY", "1")
    monkeypatch.delenv("TRTLLM_DISAGG_BENCHMARK_GEN_ONLY", raising=False)
    service = _make_service("context_first")
    service._config.bounded_replay_on_generation = True
    service._config.conditional_disagg_config = ConditionalDisaggConfig(max_local_prefill_length=32)
    router = KvCacheAwareRouter(server_role=ServerRole.GENERATION, servers=[])
    router.get_next_server = AsyncMock(
        return_value=(
            "gen:8000",
            {
                "match_length": 0,
                "num_tokens": 32,
            },
        )
    )
    service._gen_router = router
    service._ctx_client = AsyncMock()
    service._gen_client = AsyncMock()
    service._gen_client.send_request = AsyncMock(
        return_value=_make_completion_response("ok", finish_reason="stop")
    )
    request = CompletionRequest(model="model", prompt=[1] * 32)

    await service._send_disagg_request_ctx_first(request)

    service._ctx_client.send_request.assert_not_awaited()
    gen_request = service._gen_client.send_request.call_args.args[0]
    assert gen_request.disaggregated_params is None


@pytest.mark.asyncio
async def test_conditional_disagg_bypass_strips_client_disagg_params(monkeypatch):
    monkeypatch.delenv("TRTLLM_DISAGG_BENCHMARK_GEN_ONLY", raising=False)
    service = _make_service("context_first")
    service._config.conditional_disagg_config = ConditionalDisaggConfig(max_local_prefill_length=32)
    router = KvCacheAwareRouter(server_role=ServerRole.GENERATION, servers=[])
    router.get_next_server = AsyncMock(
        return_value=(
            "gen:8000",
            {
                "match_length": 64,
                "num_tokens": 96,
            },
        )
    )
    service._gen_router = router
    service._gen_client = AsyncMock()
    service._gen_client.send_request = AsyncMock(
        return_value=_make_completion_response("ok", finish_reason="stop")
    )
    request = CompletionRequest(
        model="model",
        prompt=[1] * 96,
        disaggregated_params=DisaggregatedParams(
            request_type="generation_only",
            ctx_info_endpoint="tcp://attacker:5000",
            encoded_opaque_state="attacker-state",
        ),
    )

    await service._send_disagg_request_ctx_first(request)

    gen_req = service._gen_client.send_request.call_args.args[0]
    assert gen_req.disaggregated_params is None


def _make_completion_response(
    text: str,
    finish_reason: str,
    disagg_request_id: int = 42,
    prompt_token_ids=None,
    context_only=True,
    prompt_tokens=1,
    completion_tokens=1,
    cached_tokens=0,
) -> CompletionResponse:
    if prompt_token_ids is None:
        prompt_token_ids = [1, 2, 3]
    return CompletionResponse(
        model="test-model",
        usage=UsageInfo(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            prompt_tokens_details=PromptTokensDetails(cached_tokens=cached_tokens),
        ),
        prompt_token_ids=prompt_token_ids,
        choices=[
            CompletionResponseChoice(
                index=0,
                text=text,
                finish_reason=finish_reason,
                disaggregated_params=DisaggregatedParams(
                    request_type="context_only" if context_only else "generation_only",
                    disagg_request_id=disagg_request_id,
                    ctx_request_id=disagg_request_id,
                ),
            )
        ],
    )


def _make_chat_response(
    finish_reason: str,
    disagg_request_id: int = 42,
    prompt_token_ids=None,
) -> ChatCompletionResponse:
    if prompt_token_ids is None:
        prompt_token_ids = [1, 2, 3]
    return ChatCompletionResponse(
        model="test-model",
        usage=UsageInfo(prompt_tokens=1, completion_tokens=1),
        prompt_token_ids=prompt_token_ids,
        choices=[
            ChatCompletionResponseChoice(
                index=0,
                message=ChatMessage(role="assistant", content=""),
                finish_reason=finish_reason,
                disaggregated_params=DisaggregatedParams(
                    request_type="context_only",
                    disagg_request_id=disagg_request_id,
                    ctx_request_id=disagg_request_id,
                ),
            )
        ],
    )


def _make_responses_response(status: ResponseStatus = "completed") -> ResponsesResponse:
    """Build a concrete Responses API result with the requested terminal status."""
    return ResponsesResponse(
        model="test-model",
        output=[],
        parallel_tool_calls=False,
        temperature=1.0,
        tool_choice="auto",
        tools=[],
        top_p=1.0,
        background=False,
        service_tier="auto",
        status=status,
        top_logprobs=0,
        truncation="disabled",
    )


def test_remote_tail_marker_requires_generation_after_context_stop():
    service = _make_service("context_first")
    response = _make_completion_response("", finish_reason="stop")
    response.choices[0].disaggregated_params.first_gen_tokens = [17]
    response.choices[0].disaggregated_params.remote_tail_start = 64

    assert service._need_gen(response)


async def _mock_streaming_response(chunks):
    for chunk in chunks:
        await asyncio.sleep(0)
        yield chunk


def test_get_gen_request_uses_ctx_response_prompt_token_ids_for_chat():
    service = _make_service("context_first")
    ctx_prompt_token_ids = [101, 102, 103]
    request = ChatCompletionRequest(
        model="test-model",
        messages=[
            {
                "role": "user",
                "content": "hello",
            }
        ],
        prompt_token_ids=[1, 2, 3],
        conversation_params=ConversationParams(conversation_id="conv-chat"),
    )
    ctx_response = _make_chat_response(
        finish_reason="length",
        prompt_token_ids=ctx_prompt_token_ids,
    )

    gen_request = service._get_gen_request(request, ctx_response, 42)

    assert gen_request.prompt_token_ids == ctx_prompt_token_ids
    assert gen_request.disaggregated_params.request_type == "generation_only"
    assert gen_request.conversation_params.conversation_id == "conv-chat"


def test_get_ctx_request_preserves_conversation_params_on_wire():
    service = _make_service("context_first")
    request = CompletionRequest(
        model="test-model",
        prompt="hello",
        conversation_params=ConversationParams(conversation_id="conv-completion"),
    )

    ctx_request = service._get_ctx_request(request, 42)

    wire_request = ctx_request.model_dump(exclude_unset=True)
    assert wire_request["conversation_params"]["conversation_id"] == "conv-completion"


def test_get_ctx_request_isolates_subagent_affinity_from_gen():
    # The ctx request must carry the parent affinity key on its OWN
    # conversation_params copy, so stripping the gen request (which shares the
    # original object via a shallow model_copy) never mutates the ctx copy.
    service = _make_service("context_first")
    request = CompletionRequest(
        model="test-model",
        prompt="hello",
        conversation_params=ConversationParams(
            conversation_id="child", subagent_affinity_id="parent"
        ),
    )

    ctx_request = service._get_ctx_request(request, 42)
    assert ctx_request.conversation_params.subagent_affinity_id == "parent"
    # The ctx copy is a distinct object from the original conversation_params.
    assert ctx_request.conversation_params is not request.conversation_params

    # Simulate the gen request, which shares conversation_params with the original.
    gen_req = request.model_copy(update={"disaggregated_params": None})
    assert gen_req.conversation_params is request.conversation_params
    service._strip_gen_subagent_affinity(gen_req)

    assert gen_req.conversation_params.subagent_affinity_id is None
    # Stripping gen did not disturb the ctx copy's parent pin.
    assert ctx_request.conversation_params.subagent_affinity_id == "parent"


def test_strip_gen_subagent_affinity_context_scope_clears_gen():
    service = _make_service("context_first")
    service._subagent_affinity_scope = "context"
    gen_req = CompletionRequest(
        model="test-model",
        prompt="hi",
        conversation_params=ConversationParams(
            conversation_id="child", subagent_affinity_id="parent"
        ),
    )

    service._strip_gen_subagent_affinity(gen_req)

    # "context" scope: gen load-balances on its own id, no parent pin.
    assert gen_req.conversation_params.subagent_affinity_id is None
    assert gen_req.conversation_params.conversation_id == "child"


def test_strip_gen_subagent_affinity_both_scope_keeps_gen():
    service = _make_service("context_first")
    service._subagent_affinity_scope = "both"
    gen_req = CompletionRequest(
        model="test-model",
        prompt="hi",
        conversation_params=ConversationParams(
            conversation_id="child", subagent_affinity_id="parent"
        ),
    )

    service._strip_gen_subagent_affinity(gen_req)

    # "both" scope: the parent pin is preserved on the gen request too.
    assert gen_req.conversation_params.subagent_affinity_id == "parent"


def test_subagent_affinity_id_excluded_from_wire():
    # The whole rolling-deploy guarantee: subagent_affinity_id is server-private
    # and must never cross the wire in the body (an old worker's extra="forbid"
    # schema would reject an unknown field). It is forwarded as a header instead.
    params = ConversationParams(conversation_id="child", subagent_affinity_id="parent")
    dumped = params.model_dump(mode="json", exclude_unset=True)
    assert "subagent_affinity_id" not in dumped
    assert dumped["conversation_id"] == "child"

    # Same at the request level, matching OpenAIClient's wire encoding.
    request = CompletionRequest(model="m", prompt="hi", conversation_params=params)
    wire = request.model_dump(mode="json", exclude_unset=True)
    assert "subagent_affinity_id" not in wire["conversation_params"]


@pytest.mark.asyncio
@pytest.mark.parametrize("schedule_style", ["context_first", "generation_first"])
@pytest.mark.parametrize("scope", ["context", "both"])
async def test_subagent_body_id_and_affinity_at_router_dispatch(
    monkeypatch: pytest.MonkeyPatch, schedule_style: str, scope: str
) -> None:
    monkeypatch.delenv("TRTLLM_DISAGG_BENCHMARK_GEN_ONLY", raising=False)
    service = _make_service(schedule_style)
    service._subagent_affinity_scope = scope
    service._coordinator.get_disagg_request_id = AsyncMock(return_value=42)
    request = ChatCompletionRequest(
        model="test-model",
        messages=[{"role": "user", "content": "hello"}],
        conversation_params=ConversationParams(conversation_id="body-child"),
    )
    OpenAIDisaggServer._extract_conversation_id(
        request,
        SimpleNamespace(headers={"x-session-id": "header-child", "x-parent-id": "parent"}),
        "x-parent-id",
    )
    observed: dict[str, tuple[str, str]] = {}

    async def select_context(req, **kwargs):
        # Snapshot values at dispatch: generation subsequently mutates the request.
        observed["context"] = (req.conversation_params.conversation_id, get_request_routing_id(req))
        return "ctx:9000", {"server_info": {}}

    async def select_generation(req, **kwargs):
        observed["generation"] = (
            req.conversation_params.conversation_id,
            get_request_routing_id(req),
        )
        return "gen:9001", {"server_info": {}}

    async def context_response(*args, **kwargs):
        yield _make_chat_response("length").model_dump()

    async def generation_response(*args, **kwargs):
        yield _make_chat_response("stop").model_dump()

    service._ctx_router.get_next_server.side_effect = select_context
    service._gen_router.get_next_server.side_effect = select_generation
    # Isolate metrics registration across parametrized cases and other client tests.
    monkeypatch.setattr("tensorrt_llm.serve.openai_client.ClientMetricsCollector", mock.Mock())
    # Keep client-side router selection, mocking the HTTP exchange.
    service._ctx_client = OpenAIHttpClient(
        service._ctx_router, ServerRole.CONTEXT, session=mock.Mock()
    )
    service._gen_client = OpenAIHttpClient(
        service._gen_router, ServerRole.GENERATION, session=mock.Mock()
    )
    monkeypatch.setattr(service._ctx_client, "_post_with_retry", context_response)
    monkeypatch.setattr(service._gen_client, "_post_with_retry", generation_response)

    await service._send_disagg_request(request)

    service._ctx_router.get_next_server.assert_awaited_once()
    service._gen_router.get_next_server.assert_awaited_once()
    assert observed == {
        "context": ("body-child", "parent"),
        "generation": ("body-child", "parent" if scope == "both" else "body-child"),
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("schedule_style", ["context_first", "generation_first"])
@pytest.mark.parametrize("scope", ["context", "both"])
@pytest.mark.parametrize("handler_name", ["openai_chat", "chat_harmony"])
@pytest.mark.parametrize("affinity_enabled", [False, True])
@pytest.mark.parametrize("configured_role", [False, True])
async def test_subagent_affinity_http_worker_adp_placement(
    monkeypatch: pytest.MonkeyPatch,
    schedule_style: str,
    scope: str,
    handler_name: str,
    affinity_enabled: bool,
    configured_role: bool,
) -> None:
    """Exercise transport and worker ingestion with real two-rank routing.

    Only model execution/tokenization and response postprocessing are stubbed;
    the service, HTTP client, worker chat handler and ADP router execute normally.
    Rank states are deterministic CPU fixtures, not a GPU execution test.
    """
    monkeypatch.delenv("TRTLLM_DISAGG_BENCHMARK_GEN_ONLY", raising=False)
    monkeypatch.setattr("tensorrt_llm.serve.openai_client.ClientMetricsCollector", mock.Mock())

    async def no_media():
        return None, None

    monkeypatch.setattr(
        "tensorrt_llm.serve.openai_server.parse_chat_messages_coroutines",
        lambda *args, **kwargs: ([], no_media(), {}, []),
    )
    service = _make_service(schedule_style)
    service._subagent_affinity_scope = scope
    service._coordinator.get_disagg_request_id = AsyncMock(side_effect=range(42, 46))
    auth_key = "secret" if affinity_enabled else None
    observed: dict[ServerRole, list[tuple[str, str | None, int]]] = {
        ServerRole.CONTEXT: [],
        ServerRole.GENERATION: [],
    }

    def make_worker(role: ServerRole) -> web.Application:
        router = ConversationAwareADPRouter(
            SimpleNamespace(tp_rank=0, tp_size=2, has_cp_helix=False)
        )
        states = [
            RankState(rank=rank, num_active_requests=0, num_active_tokens=0) for rank in range(2)
        ]

        def generate(**kwargs):
            conversation = kwargs["conversation_params"]
            scheduling = kwargs["scheduling_params"]
            item = SimpleNamespace(
                request=SimpleNamespace(
                    py_conversation_params=conversation,
                    py_scheduling_params=scheduling,
                )
            )
            assignments, _ = router.route_requests(states, [item], max_num_active_requests=32)
            rank = next(rank for rank, items in assignments.items() if items)
            observed[role].append(
                (conversation.conversation_id, scheduling.subagent_affinity_id, rank)
            )
            return SimpleNamespace(prompt_token_ids=[1, 2, 3])

        server = object.__new__(OpenAIServer)
        server.server_role = role if configured_role else None
        server._internal_disagg_auth_key = auth_key
        server.allow_request_chat_template = False
        server.generator = SimpleNamespace(
            args=SimpleNamespace(
                num_postprocess_workers=0,
                gather_generation_logits=False,
                reasoning_parser=None,
                backend="pytorch",
                cache_transceiver_config={"backend": "UCX"},
            ),
            generate_async=generate,
        )
        server.model_config = SimpleNamespace(model_type="llama", vocab_size=1000)
        server.tokenizer = SimpleNamespace(tokenizer=SimpleNamespace(vocab_size=1000))
        server.multimodal_server_config = None
        server.tool_parser = None
        server.tool_call_id_type = "random"
        server.chat_template = None
        server.harmony_adapter = mock.Mock()
        server.harmony_adapter.get_stop_tokens.return_value = [999]
        server.await_disconnected = AsyncMock()

        async def postprocess(promise, postproc_params, raw_request, disaggregated_params):
            return _make_chat_response(
                "length" if role == ServerRole.CONTEXT else "stop",
                disagg_request_id=disaggregated_params.disagg_request_id,
            )

        server._create_chat_response = postprocess

        async def handle(http_request: web.Request) -> web.Response:
            body = msgspec.msgpack.decode(await http_request.read())
            assert "subagent_affinity_id" not in body["conversation_params"]
            request = ChatCompletionRequest.model_validate(body)
            assert (
                request.disaggregated_params.schedule_style
                == DisaggScheduleStyle[schedule_style.upper()]
            )
            raw_request = Request({"type": "http", "headers": http_request.raw_headers})
            response = await getattr(server, handler_name)(request, raw_request)
            return web.Response(
                body=response.body, status=response.status_code, content_type="application/json"
            )

        app = web.Application()
        app.router.add_post("/v1/chat/completions", handle)
        return app

    async with AsyncExitStack() as stack:
        session = await stack.enter_async_context(aiohttp.ClientSession())
        for role, instance_router, client_attr in (
            (ServerRole.CONTEXT, service._ctx_router, "_ctx_client"),
            (ServerRole.GENERATION, service._gen_router, "_gen_client"),
        ):
            worker = await stack.enter_async_context(TestServer(make_worker(role)))
            # Completion clients add the HTTP scheme to router host:port addresses.
            instance_router.get_next_server.return_value = (
                f"{worker.host}:{worker.port}",
                {"server_info": {}},
            )
            setattr(
                service,
                client_attr,
                OpenAIHttpClient(
                    instance_router,
                    role,
                    timeout_secs=5,
                    session=session,
                    internal_disagg_auth_key=auth_key,
                ),
            )
        for conversation_id in ("parent", "child-a", "child-b", "child-a"):
            request = ChatCompletionRequest(
                model="test-model",
                messages=[{"role": "user", "content": "hello"}],
                prompt_token_ids=[1, 2, 3],
                conversation_params=ConversationParams(conversation_id=conversation_id),
            )
            headers = {"x-session-id": "conflicting-header-id"}
            if conversation_id != "parent":
                headers["x-parent-id"] = "parent"
            OpenAIDisaggServer._extract_conversation_id(
                request,
                SimpleNamespace(headers=headers),
                "x-parent-id" if affinity_enabled else None,
            )
            await service._send_disagg_request(request)

    parent_placement = [("parent", None, 0)]
    child_placements = [
        ("child-a", "parent", 0),
        ("child-b", "parent", 0),
        ("child-a", "parent", 0),
    ]
    independent_placements = [("child-a", None, 1), ("child-b", None, 0), ("child-a", None, 1)]
    assert observed[ServerRole.CONTEXT] == parent_placement + (
        child_placements if affinity_enabled else independent_placements
    )
    assert observed[ServerRole.GENERATION] == parent_placement + (
        child_placements if affinity_enabled and scope == "both" else independent_placements
    )


@pytest.mark.asyncio
async def test_create_chat_response_sets_prompt_token_ids_for_context_only():
    from tensorrt_llm.serve.openai_server import OpenAIServer

    prompt_token_ids = [101, 102, 103]

    class FakePromise:
        def __init__(self) -> None:
            self.outputs = []
            self.prompt_token_ids = prompt_token_ids
            self.error = None

        async def aresult(self):
            return self

    server = OpenAIServer.__new__(OpenAIServer)
    server.generator = mock.MagicMock()
    server.generator.args.num_postprocess_workers = 0
    server._extract_metrics = mock.AsyncMock()

    post_processor = mock.Mock(
        return_value=_make_chat_response(
            finish_reason="length",
            prompt_token_ids=None,
        )
    )
    postproc_params = SimpleNamespace(post_processor=post_processor, postproc_args=object())
    disaggregated_params = SimpleNamespace(request_type="context_only", disagg_request_id=123)

    response = await server._create_chat_response(
        FakePromise(),
        postproc_params,
        raw_request=None,
        disaggregated_params=disaggregated_params,
    )

    assert response.prompt_token_ids == prompt_token_ids


@pytest.mark.asyncio
async def test_is_ready_waits_for_router_preparation():
    service = _make_service("context_first")
    cluster_manager = DisaggClusterManager(
        DisaggClusterConfig(
            cluster_uri="http://localhost:18000",
            minimal_instances=MinimalInstances(context_servers=1, generation_servers=1),
        ),
        AsyncMock(),
    )
    # Readiness now lives on the DisaggCoordinatorService the service holds.
    local = service._coordinator
    local._disagg_cluster_manager = cluster_manager

    cluster_manager._current_ctx_workers["ctx"] = WorkerInfo(
        worker_id="ctx", role=ServerRole.CONTEXT
    )
    local._ctx_router = SimpleNamespace(num_prepared_servers=0)
    local._gen_router = SimpleNamespace(num_prepared_servers=1)
    assert await service.is_ready() is False

    local._ctx_router.num_prepared_servers = 1
    assert await service.is_ready() is False

    cluster_manager._current_gen_workers["gen"] = WorkerInfo(
        worker_id="gen", role=ServerRole.GENERATION
    )
    assert await service.is_ready() is True


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True], ids=["non-streaming", "streaming"])
@pytest.mark.parametrize("schedule_style", ["context_first", "generation_first"])
async def test_send_disagg_request(monkeypatch, stream, schedule_style):
    # simulate different order of ctx and gen responses in gen first mode
    ctx_gen_delay = [
        (0.2, 0.1),
        (0.2, 0.2),
        (0.1, 0.2),
    ]
    monkeypatch.delenv("TRTLLM_DISAGG_BENCHMARK_GEN_ONLY", raising=False)
    service = _make_service(schedule_style)
    if schedule_style == "context_first":
        assert service._send_disagg_request == service._send_disagg_request_ctx_first
        assert service._schedule_style == DisaggScheduleStyle.CONTEXT_FIRST
    else:
        assert service._send_disagg_request == service._send_disagg_request_gen_first
        assert service._schedule_style == DisaggScheduleStyle.GENERATION_FIRST
    opaque_state = "opaque_state"
    for i, (ctx_delay, gen_delay) in enumerate(ctx_gen_delay):
        stream_chunks = [b"data: gen-0\n\n", b"data: gen-1\n\n"]
        service._ctx_client = AsyncMock()
        service._gen_client = AsyncMock()
        ctx_server_info = {
            "server_info": {"disaggregated_params": {"encoded_opaque_state": opaque_state}}
        }
        service._ctx_router.get_next_server = AsyncMock(return_value=("ctx:9000", ctx_server_info))
        service._gen_router.get_next_server = AsyncMock(
            return_value=("gen:9001", {"server_info": {}})
        )
        resp_text = f"response-{i}"

        async def _delayed_ctx_response(*_args, **_kwargs):
            request = _args[0]
            server, info = await service._ctx_router.get_next_server(request)
            await asyncio.sleep(ctx_delay)
            return _make_completion_response(
                resp_text,
                finish_reason="length",
                disagg_request_id=request.disaggregated_params.disagg_request_id,
                context_only=True,
                prompt_tokens=101,
                completion_tokens=0,
                cached_tokens=7,
            )

        async def _delayed_gen_response(*_args, **_kwargs):
            request = _args[0]
            server, info = await service._gen_router.get_next_server(request)
            await asyncio.sleep(gen_delay)
            if stream:
                return _mock_streaming_response(stream_chunks)
            return _make_completion_response(
                resp_text,
                finish_reason="stop",
                disagg_request_id=request.disaggregated_params.disagg_request_id,
                context_only=False,
                prompt_tokens=3,
                completion_tokens=13,
                cached_tokens=3,
            )

        service._ctx_client.send_request = AsyncMock(side_effect=_delayed_ctx_response)
        service._gen_client.send_request = AsyncMock(side_effect=_delayed_gen_response)

        request = CompletionRequest(model="test-model", prompt="hello", stream=stream)
        result = await service._send_disagg_request(request)

        ctx_req = service._ctx_client.send_request.call_args.args[0]
        assert ctx_req.disaggregated_params.request_type == "context_only"

        gen_req = service._gen_client.send_request.call_args.args[0]
        assert gen_req.disaggregated_params.request_type == "generation_only"
        if schedule_style == "generation_first":
            assert gen_req.disaggregated_params.encoded_opaque_state == opaque_state
            assert gen_req.disaggregated_params.ctx_usage is None
            print("[usage_check] generation_first: ctx_usage=None (expected)")
        else:
            print(
                f"[usage_check] context_first: gen_req ctx_usage="
                f"prompt_tokens={gen_req.disaggregated_params.ctx_usage.prompt_tokens}, "
                f"cached_tokens="
                f"{gen_req.disaggregated_params.ctx_usage.prompt_tokens_details.cached_tokens}"
            )
            assert gen_req.disaggregated_params.ctx_usage.prompt_tokens == 101
            assert gen_req.disaggregated_params.ctx_usage.prompt_tokens_details.cached_tokens == 7
        assert (
            gen_req.disaggregated_params.ctx_request_id
            == ctx_req.disaggregated_params.disagg_request_id
        )

        if stream:
            assert hasattr(result, "__aiter__")
            chunks = [chunk async for chunk in result]
            assert chunks == stream_chunks
        else:
            print(
                f"[usage_check] disagg service result: "
                f"prompt_tokens={result.usage.prompt_tokens}, "
                f"completion_tokens={result.usage.completion_tokens}, "
                f"total_tokens={result.usage.total_tokens}, "
                f"cached_tokens={result.usage.prompt_tokens_details.cached_tokens}"
            )
            assert result.model == "test-model"
            assert result.usage.prompt_tokens == 3
            assert result.usage.completion_tokens == 13
            assert result.usage.total_tokens == 16
            assert result.usage.prompt_tokens_details.cached_tokens == 3
            assert len(result.choices) == 1
            assert result.choices[0].text == resp_text
            assert result.choices[0].finish_reason == "stop"
            assert (
                result.choices[0].disaggregated_params.disagg_request_id
                == ctx_req.disaggregated_params.disagg_request_id
            )


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True], ids=["non-streaming", "streaming"])
async def test_context_only_response_finishes_hooks(stream):
    service = _make_service("context_first")
    service._ctx_client = AsyncMock()
    service._gen_client = AsyncMock()
    service._coordinator.get_disagg_request_id = AsyncMock(return_value=42)
    service._check_conditional_disagg = AsyncMock(return_value=(None, True))
    service._check_gen_only_disagg = AsyncMock(return_value=False)
    service._ctx_router.get_next_server = AsyncMock(return_value=("ctx:9000", {"server_info": {}}))
    ctx_response = _make_completion_response("done", finish_reason="stop", context_only=True)
    service._ctx_client.send_request = AsyncMock(return_value=ctx_response)
    hooks = mock.Mock()
    request = CompletionRequest(model="test-model", prompt="hello", stream=stream)

    result = await service._send_disagg_request(request, hooks)
    if stream:
        assert [chunk async for chunk in result] == [b"data: [DONE]\n\n"]
    else:
        assert result is ctx_response

    hooks.on_resp_done.assert_called_once_with("", request, ctx_response)
    service._gen_client.send_request.assert_not_awaited()


@pytest.mark.asyncio
async def test_gen_only_benchmark_streams_the_generation_worker_output_as_is(monkeypatch):
    """Gen-only benchmark mode hands the client the generation worker's stream as is.

    The context server is skipped, and an error the worker reports on the stream (an
    executor fail-fast, for example) ends the client's stream instead of being swallowed.
    """
    monkeypatch.setenv("TRTLLM_DISAGG_BENCHMARK_GEN_ONLY", "1")
    service = _make_service("context_first")
    service._coordinator.get_disagg_request_id = AsyncMock(return_value=42)
    service._ctx_client = AsyncMock()
    service._gen_client = AsyncMock()
    service._gen_router.get_next_server = AsyncMock(return_value=("gen:9001", {"server_info": {}}))
    gen_chunks = [
        b'data: {"error": {"message": "Insufficient KV cache for gen-only benchmark mode",'
        b' "type": "server_error"}}\n\n',
        b"data: [DONE]\n\n",
    ]
    service._gen_client.send_request = AsyncMock(return_value=_mock_streaming_response(gen_chunks))
    request = CompletionRequest(model="test-model", prompt="hello", stream=True)

    result = await service._send_disagg_request(request)

    service._ctx_client.send_request.assert_not_awaited()
    gen_req = service._gen_client.send_request.call_args.args[0]
    assert gen_req.disaggregated_params.request_type == "generation_only"
    assert gen_req.disaggregated_params.first_gen_tokens == [7]
    assert gen_req.ignore_eos is True
    assert [chunk async for chunk in result] == gen_chunks


@pytest.mark.asyncio
@pytest.mark.parametrize("schedule_style", ["context_first", "generation_first"])
async def test_send_disagg_request_leaves_streaming_usage_to_gen_server(schedule_style):
    service = _make_service(schedule_style)
    service._ctx_client = AsyncMock()
    service._gen_client = AsyncMock()
    service._ctx_router.get_next_server = AsyncMock(return_value=("ctx:9000", {"server_info": {}}))
    service._gen_router.get_next_server = AsyncMock(return_value=("gen:9001", {"server_info": {}}))

    async def _ctx_response(request, *_args, **_kwargs):
        return _make_completion_response(
            "",
            finish_reason="length",
            disagg_request_id=request.disaggregated_params.disagg_request_id,
            context_only=True,
            prompt_tokens=128,
            completion_tokens=0,
            cached_tokens=9,
        )

    async def _gen_response(*_args, **_kwargs):
        return _mock_streaming_response(
            [
                (
                    b'data: {"choices":[{"delta":{"content":"hello"},"index":0}],'
                    b'"model":"test-model"}\n\n'
                ),
                (
                    b'data: {"choices":[],"model":"test-model","usage":'
                    b'{"prompt_tokens":3,"completion_tokens":5,"total_tokens":8,'
                    b'"prompt_tokens_details":{"cached_tokens":3}}}\n\n'
                ),
                b"data: [DONE]\n\n",
            ]
        )

    service._ctx_client.send_request = AsyncMock(side_effect=_ctx_response)
    service._gen_client.send_request = AsyncMock(side_effect=_gen_response)

    request = CompletionRequest(model="test-model", prompt="hello", stream=True)
    result = await service._send_disagg_request(request)
    chunks = [chunk async for chunk in result]

    assert chunks[1] == (
        b'data: {"choices":[],"model":"test-model","usage":'
        b'{"prompt_tokens":3,"completion_tokens":5,"total_tokens":8,'
        b'"prompt_tokens_details":{"cached_tokens":3}}}\n\n'
    )


@pytest.mark.asyncio
async def test_context_retry_preserves_generation_reservation_id():
    service = _make_service("context_first")
    service._ctx_client = AsyncMock()
    service._gen_client = AsyncMock()
    service._coordinator.get_disagg_request_id = AsyncMock(return_value=101)
    service._check_conditional_disagg = AsyncMock(return_value=("gen:9001", True))
    service._check_gen_only_disagg = AsyncMock(return_value=False)
    service._ctx_router.get_next_server = AsyncMock(return_value=("ctx:9000", {"server_info": {}}))

    async def _ctx_response(request, *_args, **_kwargs):
        # OpenAIHttpClient regenerates this field before a successful retry.
        request.disaggregated_params.disagg_request_id = 202
        return _make_completion_response("", finish_reason="length", disagg_request_id=202)

    service._ctx_client.send_request = AsyncMock(side_effect=_ctx_response)
    service._gen_client.send_request = AsyncMock(
        return_value=_make_completion_response(
            "done", finish_reason="stop", disagg_request_id=202, context_only=False
        )
    )

    request = CompletionRequest(model="test-model", prompt="hello")
    await service._send_disagg_request(request)

    service._gen_router.get_next_server.assert_not_awaited()
    gen_call = service._gen_client.send_request.call_args
    assert gen_call.kwargs["req_id"] == 101
    assert gen_call.args[0].disaggregated_params.ctx_request_id == 202


@pytest.mark.asyncio
@pytest.mark.parametrize("schedule_style", ["context_first", "generation_first"])
async def test_gen_request_carries_orig_elapsed(monkeypatch, schedule_style):
    """Only a context-first GEN request carries orig_elapsed_s."""
    monkeypatch.delenv("TRTLLM_DISAGG_BENCHMARK_GEN_ONLY", raising=False)
    service = _make_service(schedule_style)
    service._ctx_client = AsyncMock()
    service._gen_client = AsyncMock()
    service._ctx_router.get_next_server = AsyncMock(return_value=("ctx:9000", {"server_info": {}}))
    service._gen_router.get_next_server = AsyncMock(return_value=("gen:9001", {"server_info": {}}))
    service._ctx_client.send_request = AsyncMock(
        side_effect=lambda req, *a, **k: _make_completion_response(
            "",
            finish_reason="length",
            disagg_request_id=req.disaggregated_params.disagg_request_id,
            context_only=True,
        )
    )
    service._gen_client.send_request = AsyncMock(
        side_effect=lambda req, *a, **k: _make_completion_response(
            "done",
            finish_reason="stop",
            disagg_request_id=req.disaggregated_params.disagg_request_id,
            context_only=False,
        )
    )
    hooks = mock.Mock()
    # ServerArrivalTimeMiddleware stamps arrivals on the global steady clock.
    hooks.request_arrival_time = get_global_steady_clock_now_in_seconds() - 5.0

    await service._send_disagg_request(CompletionRequest(model="test-model", prompt="hello"), hooks)

    gen_req = service._gen_client.send_request.call_args.args[0]
    elapsed = gen_req.disaggregated_params.orig_elapsed_s
    if schedule_style == "generation_first":
        assert elapsed is None
        assert (
            "orig_elapsed_s"
            not in gen_req.model_dump(mode="json", exclude_unset=True)["disaggregated_params"]
        )
        return
    assert 5.0 <= elapsed < 60.0
    # On the wire (the client sends exclude_unset=True) ...
    wire = gen_req.model_dump(mode="json", exclude_unset=True)
    assert wire["disaggregated_params"]["orig_elapsed_s"] == elapsed
    # ... and through to the LLM API params the GEN executor reads.
    assert to_llm_disaggregated_params(gen_req.disaggregated_params).orig_elapsed_s == elapsed
    # The context request does not carry it.
    ctx_req = service._ctx_client.send_request.call_args.args[0]
    assert ctx_req.disaggregated_params.orig_elapsed_s is None


def test_orig_elapsed_arrival_fallback():
    before = get_global_steady_clock_now_in_seconds()
    assert OpenAIDisaggregatedService._get_arrival_time(None) >= before
    assert OpenAIDisaggregatedService._get_arrival_time(mock.Mock()) >= before
    hooks = SimpleNamespace(request_arrival_time=12.5)
    assert OpenAIDisaggregatedService._get_arrival_time(hooks) == 12.5
    req = CompletionRequest(model="m", prompt="p")
    OpenAIDisaggregatedService._stamp_orig_elapsed(req, 0.0)  # no disaggregated_params: no-op
    assert req.disaggregated_params is None


def test_generation_postprocessor_rewrites_usage_from_disaggregated_params():
    ctx_usage = UsageInfo(
        prompt_tokens=128,
        completion_tokens=0,
        total_tokens=128,
        prompt_tokens_details=PromptTokensDetails(cached_tokens=9),
    )
    request = CompletionRequest(
        model="test-model",
        prompt=[1, 2, 3],
        disaggregated_params=DisaggregatedParams(
            request_type="generation_only",
            ctx_usage=ctx_usage,
        ),
    )
    args = CompletionPostprocArgs.from_request(request)
    args.num_prompt_tokens = 3
    rsp = SimpleNamespace(
        outputs=[
            SimpleNamespace(
                text="hello",
                token_ids=[10, 11, 12, 13, 14],
                index=0,
                disaggregated_params=LlmDisaggregatedParams(
                    request_type="generation_only",
                    ctx_usage=ctx_usage.model_dump(),
                ),
                context_logits=None,
                stop_reason=None,
                finish_reason="stop",
                length=5,
            )
        ],
        context_logits=None,
        cached_tokens=3,
    )

    response = completion_response_post_processor(rsp, args)

    print(
        f"[usage_check] completion postprocessor: "
        f"prompt_tokens={response.usage.prompt_tokens}, "
        f"completion_tokens={response.usage.completion_tokens}, "
        f"total_tokens={response.usage.total_tokens}, "
        f"cached_tokens={response.usage.prompt_tokens_details.cached_tokens}"
    )
    assert response.usage.prompt_tokens == 128
    assert response.usage.completion_tokens == 5
    assert response.usage.total_tokens == 133
    assert response.usage.prompt_tokens_details.cached_tokens == 9


def test_generation_chat_postprocessor_rewrites_usage_from_output_disaggregated_params():
    ctx_usage = {
        "prompt_tokens": 128,
        "completion_tokens": 0,
        "total_tokens": 128,
        "prompt_tokens_details": {
            "cached_tokens": 9,
        },
    }
    request = ChatCompletionRequest(
        model="test-model",
        messages=[
            {
                "role": "user",
                "content": "hello",
            }
        ],
        disaggregated_params=DisaggregatedParams(request_type="generation_only"),
    )
    args = ChatPostprocArgs.from_request(request)
    args.num_prompt_tokens = 3
    rsp = SimpleNamespace(
        outputs=[
            SimpleNamespace(
                text="hello",
                token_ids=[10, 11, 12, 13, 14],
                index=0,
                disaggregated_params=LlmDisaggregatedParams(
                    request_type="generation_only",
                    ctx_usage=ctx_usage,
                ),
                stop_reason=None,
                finish_reason="stop",
            )
        ],
        cached_tokens=3,
    )

    response = chat_response_post_processor(rsp, args)

    print(
        f"[usage_check] chat postprocessor: "
        f"prompt_tokens={response.usage.prompt_tokens}, "
        f"completion_tokens={response.usage.completion_tokens}, "
        f"total_tokens={response.usage.total_tokens}, "
        f"cached_tokens={response.usage.prompt_tokens_details.cached_tokens}"
    )
    assert response.usage.prompt_tokens == 128
    assert response.usage.completion_tokens == 5
    assert response.usage.total_tokens == 133
    assert response.usage.prompt_tokens_details.cached_tokens == 9


class TestVerifyCtxResponseDiagnostics:
    """Test enriched error messages in _verify_ctx_response (TRTLLM-11123)."""

    @pytest.mark.asyncio
    async def test_missing_disagg_params_includes_finish_reason(self):
        svc = _make_service("context_first")
        resp = _make_completion_response("", finish_reason="error", disagg_request_id=1)
        resp.choices[0].disaggregated_params = None
        with pytest.raises(ValueError, match="finish_reason='error'"):
            await svc._verify_ctx_response(resp)

    @pytest.mark.asyncio
    async def test_missing_ctx_request_id_includes_disagg_id(self):
        svc = _make_service("context_first")
        resp = _make_completion_response("", finish_reason="length", disagg_request_id=999)
        resp.choices[0].disaggregated_params.ctx_request_id = None
        with pytest.raises(ValueError, match=r"ctx_request_id is None.*999"):
            await svc._verify_ctx_response(resp)

    @pytest.mark.asyncio
    async def test_missing_disagg_request_id_includes_ctx_id(self):
        svc = _make_service("context_first")
        resp = _make_completion_response("", finish_reason="length", disagg_request_id=555)
        resp.choices[0].disaggregated_params.disagg_request_id = None
        resp.choices[0].disaggregated_params.ctx_request_id = 555
        with pytest.raises(ValueError, match=r"disagg_request_id is None.*555"):
            await svc._verify_ctx_response(resp)

    @pytest.mark.asyncio
    async def test_valid_response_passes(self):
        svc = _make_service("context_first")
        resp = _make_completion_response("ok", finish_reason="stop", disagg_request_id=42)
        result = await svc._verify_ctx_response(resp)
        assert result is resp

    @pytest.mark.asyncio
    async def test_completed_response_with_null_ctx_request_id_passes(self):
        # CTX finished early (finish_reason='stop'): no GEN handoff was set up,
        # so ctx_request_id is None. The verifier must accept it (NVBug 6245861).
        svc = _make_service("context_first")
        resp = _make_completion_response("", finish_reason="stop", disagg_request_id=42)
        resp.choices[0].disaggregated_params.ctx_request_id = None
        result = await svc._verify_ctx_response(resp)
        assert result is resp

    @pytest.mark.asyncio
    async def test_completed_response_with_null_disagg_request_id_passes(self):
        svc = _make_service("context_first")
        resp = _make_completion_response("", finish_reason="stop", disagg_request_id=42)
        resp.choices[0].disaggregated_params.disagg_request_id = None
        result = await svc._verify_ctx_response(resp)
        assert result is resp


class TestFirstGenLogProbsSerializeRoundtrip:
    """Roundtrip tests for _serialize/_deserialize_first_gen_log_probs."""

    def test_none_passthrough(self):
        assert _serialize_first_gen_log_probs(None) is None
        assert _deserialize_first_gen_log_probs(None) is None

    def test_single_token_roundtrip(self):
        original = [{10: Logprob(logprob=-0.5, rank=1)}]
        serialized = _serialize_first_gen_log_probs(original)
        recovered = _deserialize_first_gen_log_probs(serialized)
        assert len(recovered) == len(original)
        for orig_pos, rec_pos in zip(original, recovered):
            assert set(orig_pos.keys()) == set(rec_pos.keys())
            for tid in orig_pos:
                assert rec_pos[tid].logprob == orig_pos[tid].logprob
                assert rec_pos[tid].rank == orig_pos[tid].rank

    def test_multi_token_topk_roundtrip(self):
        original = [
            {
                100: Logprob(logprob=-0.1, rank=1),
                200: Logprob(logprob=-2.3, rank=2),
                300: Logprob(logprob=-5.0, rank=3),
            },
            {
                400: Logprob(logprob=-0.05, rank=1),
                500: Logprob(logprob=-3.7, rank=2),
            },
        ]
        serialized = _serialize_first_gen_log_probs(original)
        recovered = _deserialize_first_gen_log_probs(serialized)
        assert len(recovered) == len(original)
        for orig_pos, rec_pos in zip(original, recovered):
            assert set(orig_pos.keys()) == set(rec_pos.keys())
            for tid in orig_pos:
                assert rec_pos[tid].logprob == pytest.approx(orig_pos[tid].logprob)
                assert rec_pos[tid].rank == orig_pos[tid].rank

    def test_rank_none_preserved(self):
        original = [{42: Logprob(logprob=-1.0, rank=None)}]
        recovered = _deserialize_first_gen_log_probs(_serialize_first_gen_log_probs(original))
        assert recovered[0][42].rank is None

    def test_empty_list_roundtrip(self):
        original = []
        recovered = _deserialize_first_gen_log_probs(_serialize_first_gen_log_probs(original))
        assert recovered == []

    def test_serialize_rejects_non_list(self):
        with pytest.raises(ValueError, match="must be a list"):
            _serialize_first_gen_log_probs("bad")

    def test_serialize_rejects_non_dict_entry(self):
        with pytest.raises(ValueError, match="must be a dict"):
            _serialize_first_gen_log_probs(["not_a_dict"])

    def test_deserialize_rejects_non_list_entry(self):
        with pytest.raises(ValueError, match="must be a list"):
            _deserialize_first_gen_log_probs(["not_a_list"])

    def test_simple_format_roundtrip(self):
        # Simple format: each position is a plain float (sampled-token logprob).
        original = [-0.5, -1.25, -2.0]
        serialized = _serialize_first_gen_log_probs(original)
        # Serialized payload preserves floats verbatim so it remains JSON-safe.
        assert serialized == [pytest.approx(v) for v in original]
        recovered = _deserialize_first_gen_log_probs(serialized)
        assert recovered == [pytest.approx(v) for v in original]
        assert all(isinstance(v, float) for v in recovered)

    def test_simple_and_dict_formats_kept_disjoint(self):
        # Each call uses one format; mixing within a single payload is unusual
        # but the serdes round-trips them independently.
        simple = _deserialize_first_gen_log_probs(_serialize_first_gen_log_probs([-0.5]))
        dict_payload = _deserialize_first_gen_log_probs(
            _serialize_first_gen_log_probs([{1: Logprob(logprob=-0.5, rank=1)}])
        )
        assert isinstance(simple[0], float)
        assert isinstance(dict_payload[0], dict)

    def test_deserialize_rejects_missing_keys(self):
        with pytest.raises(ValueError, match="missing required keys"):
            _deserialize_first_gen_log_probs([[{"token_id": 1}]])


class TestFirstGenLogitsSerializeRoundtrip:
    """Verify generation logits survive serialize -> deserialize."""

    def test_none_passthrough(self):
        assert _serialize_first_gen_logits(None) is None
        assert _deserialize_first_gen_logits(None) is None

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
    def test_single_beam_roundtrip(self, dtype):
        original = [torch.randn(1, 128, dtype=dtype)]
        serialized = _serialize_first_gen_logits(original)
        assert isinstance(serialized, list) and len(serialized) == 1
        assert isinstance(serialized[0], dict)
        assert set(serialized[0].keys()) == {"data", "shape", "dtype"}
        restored = _deserialize_first_gen_logits(serialized)
        assert len(restored) == 1
        torch.testing.assert_close(restored[0], original[0].cpu())

    def test_bfloat16_converted_to_float16(self):
        original = [torch.randn(1, 64, dtype=torch.bfloat16)]
        serialized = _serialize_first_gen_logits(original)
        restored = _deserialize_first_gen_logits(serialized)
        expected = original[0].to(torch.float16).cpu()
        torch.testing.assert_close(restored[0], expected)

    def test_multi_beam_roundtrip(self):
        original = [
            torch.randn(1, 32, dtype=torch.float32),
            torch.randn(1, 32, dtype=torch.float32),
        ]
        serialized = _serialize_first_gen_logits(original)
        assert len(serialized) == 2
        restored = _deserialize_first_gen_logits(serialized)
        assert len(restored) == 2
        for orig, rest in zip(original, restored):
            torch.testing.assert_close(rest, orig.cpu())

    def test_deserialize_invalid_type_raises(self):
        with pytest.raises(ValueError, match="must be a dict"):
            _deserialize_first_gen_logits(["not_a_dict"])

    def test_deserialize_missing_key_raises(self):
        with pytest.raises(ValueError, match="missing required key"):
            _deserialize_first_gen_logits([{"data": "abc", "shape": [1]}])


@pytest.fixture
def generation_first_stream_service():
    service = _make_service("generation_first")
    service._ctx_client = AsyncMock()
    service._gen_client = AsyncMock()
    service._coordinator.get_disagg_request_id = AsyncMock(return_value=42)
    service._check_gen_only_disagg = AsyncMock(return_value=False)
    service._ctx_router.get_next_server = AsyncMock(return_value=("ctx:9000", {"server_info": {}}))
    return service


@pytest.mark.asyncio
@pytest.mark.parametrize("error_type", [RuntimeError, asyncio.CancelledError])
async def test_gen_first_context_failure_closes_stream(generation_first_stream_service, error_type):
    service = generation_first_stream_service
    gen_response = AsyncMock()
    service._gen_client.send_request.return_value = gen_response
    service._ctx_client.send_request.side_effect = error_type("CTX interrupted")
    request = CompletionRequest(model="test-model", prompt="hello", stream=True)

    with pytest.raises(error_type):
        await service._send_disagg_request(request)

    gen_response.aclose.assert_awaited_once_with()
    # The HTTP clients own their router accounting; the service only closes GEN.
    service._ctx_router.finish_request.assert_not_called()
    service._gen_router.finish_request.assert_not_called()


@pytest.mark.asyncio
async def test_gen_first_close_before_read_closes_stream_once(
    generation_first_stream_service,
):
    service = generation_first_stream_service
    gen_response = AsyncMock()
    service._gen_client.send_request.return_value = gen_response
    request = CompletionRequest(model="test-model", prompt="hello", stream=True)

    response = await service._send_disagg_request(request)
    await response.aclose()
    await response.aclose()

    gen_response.aclose.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_gen_first_cancellation_closes_running_stream(generation_first_stream_service):
    service = generation_first_stream_service
    gen_started = asyncio.Event()
    gen_closed = asyncio.Event()
    ctx_started = asyncio.Event()
    ctx_closed = asyncio.Event()
    blocked = asyncio.Event()

    async def gen_chunks():
        try:
            gen_started.set()
            await blocked.wait()
            yield b"data: [DONE]\n\n"
        finally:
            gen_closed.set()

    async def ctx_response(*args, **kwargs):
        ctx_started.set()
        try:
            await blocked.wait()
        finally:
            ctx_closed.set()

    service._gen_client.send_request.return_value = gen_chunks()
    service._ctx_client.send_request.side_effect = ctx_response
    request = CompletionRequest(model="test-model", prompt="hello", stream=True)
    pending = asyncio.create_task(service._send_disagg_request(request))
    try:
        await asyncio.wait_for(gen_started.wait(), timeout=1)
        await asyncio.wait_for(ctx_started.wait(), timeout=1)
    finally:
        pending.cancel()
    with pytest.raises(asyncio.CancelledError):
        await pending

    assert gen_closed.is_set()
    assert ctx_closed.is_set()


@pytest.mark.asyncio
async def test_gen_first_sse_error_after_ctx_completion(
    generation_first_stream_service: OpenAIDisaggregatedService,
) -> None:
    """Once CTX succeeds, later SSE errors remain the original caller-owned stream."""
    service = generation_first_stream_service
    release = asyncio.Event()
    expected = [b'data: {"error":{"message":"GEN failed"}}\n\n', b"data: [DONE]\n\n"]

    async def gen() -> AsyncGenerator[bytes, None]:
        await release.wait()
        for chunk in expected:
            yield chunk

    service._gen_client.send_request.return_value = gen()
    response = await service._send_disagg_request(
        CompletionRequest(model="test-model", prompt="hello", stream=True)
    )
    try:
        release.set()
        assert [chunk async for chunk in response] == expected
    finally:
        await response.aclose()


@pytest.mark.asyncio
async def test_gen_first_close_after_chunk_closes_running_stream(generation_first_stream_service):
    service = generation_first_stream_service
    gen_closed = asyncio.Event()
    blocked = asyncio.Event()

    async def gen_chunks():
        try:
            yield b'data: "first"\n\n'
            await blocked.wait()
        finally:
            gen_closed.set()

    service._gen_client.send_request.return_value = gen_chunks()
    request = CompletionRequest(model="test-model", prompt="hello", stream=True)
    response = await service._send_disagg_request(request)
    assert await asyncio.wait_for(response.__anext__(), timeout=1) == b'data: "first"\n\n'
    await response.aclose()

    assert gen_closed.is_set()


@pytest.mark.asyncio
@pytest.mark.parametrize("error_type", [RuntimeError, asyncio.CancelledError])
@pytest.mark.parametrize("ctx_outcome", ["pending", "success", "error"])
async def test_gen_first_stream_failure_observed_before_ctx_timeout(
    generation_first_stream_service: OpenAIDisaggregatedService,
    error_type: type[BaseException],
    ctx_outcome: str,
) -> None:
    """GEN failure must not wait for CTX; simultaneous CTX failure takes precedence."""
    service = generation_first_stream_service
    ctx_started, ctx_release, ctx_closed = asyncio.Event(), asyncio.Event(), asyncio.Event()
    gen_closed = asyncio.Event()
    gen_error = error_type("generation failed before publication")
    ctx_error = RuntimeError("context failed")

    async def ctx(*args: Any, **kwargs: Any) -> None:
        ctx_started.set()
        try:
            await ctx_release.wait()
            if ctx_outcome == "error":
                raise ctx_error
        finally:
            ctx_closed.set()

    async def fail_gen() -> bytes:
        await ctx_started.wait()
        if ctx_outcome != "pending":
            ctx_release.set()
        raise gen_error

    async def chunks() -> AsyncGenerator[bytes, None]:
        try:
            yield await fail_gen()
        finally:
            gen_closed.set()

    service._ctx_client.send_request.side_effect = ctx
    service._gen_client.send_request.return_value = chunks()
    expected = ctx_error if ctx_outcome == "error" else gen_error
    with pytest.raises(type(expected), match=str(expected)):
        await asyncio.wait_for(
            service._send_disagg_request(
                CompletionRequest(model="test-model", prompt="hello", stream=True)
            ),
            timeout=2,
        )

    assert ctx_closed.is_set()
    assert gen_closed.is_set()
    service._ctx_router.finish_request.assert_not_called()
    service._gen_router.finish_request.assert_not_called()


@pytest.mark.asyncio
async def test_gen_first_completed_stream_waits_for_ctx(
    generation_first_stream_service: OpenAIDisaggregatedService,
) -> None:
    """Normal early GEN completion preserves the CTX wait and buffered chunks."""
    service = generation_first_stream_service
    ctx_started, ctx_release, gen_finished = asyncio.Event(), asyncio.Event(), asyncio.Event()
    expected = [b'data: "first"\n\n', b"data: [DONE]\n\n"]

    async def ctx(*args: Any, **kwargs: Any) -> None:
        ctx_started.set()
        await ctx_release.wait()

    async def chunks() -> AsyncGenerator[bytes, None]:
        await ctx_started.wait()
        for chunk in expected:
            yield chunk
        gen_finished.set()

    service._ctx_client.send_request.side_effect = ctx
    service._gen_client.send_request.return_value = chunks()
    pending = asyncio.create_task(
        service._send_disagg_request(
            CompletionRequest(model="test-model", prompt="hello", stream=True)
        )
    )
    response = None
    try:
        await asyncio.wait_for(gen_finished.wait(), timeout=2)
        assert not pending.done()
        ctx_release.set()
        response = await asyncio.wait_for(pending, timeout=2)
        assert [chunk async for chunk in response] == expected
    finally:
        pending.cancel()
        await asyncio.gather(pending, return_exceptions=True)
        if response is not None:
            await response.aclose()


@pytest.mark.parametrize(
    "chunks",
    [
        [b'data: {"error":{"message":"failed"}}\n\n'],
        [b'data: {"error":{"message":"\xe5', b'\xa4\xb1\xe6\x95\x97"}}\r\n', b"\r\n"],
        [': keepalive\n\ndata: {"choices":[]}\n\ndata: {"error":\n', "data: {}}\n\n"],
    ],
)
def test_sse_error_observer_handles_event_boundaries(chunks: list[str | bytes]) -> None:
    """Only a complete error event signals failure, regardless of chunk boundaries."""
    observer = _SseErrorObserver()
    assert [observer.feed(chunk) for chunk in chunks] == [False] * (len(chunks) - 1) + [True]
    assert observer._pending == b""


def test_sse_error_observer_scans_large_fragmented_event_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fragmentation must not repeatedly copy or scan a growing response snapshot."""
    observer = _SseErrorObserver(responses_api=True)
    pattern = observer._EVENT_END
    scanned_bytes = 0

    def search(buffer: bytes | bytearray, start: int = 0) -> re.Match[bytes] | None:
        nonlocal scanned_bytes
        match = pattern.search(buffer, start)
        scanned_bytes += (match.end() if match else len(buffer)) - start
        return match

    monkeypatch.setattr(observer, "_EVENT_END", mock.Mock(search=search))
    stream = (
        b': keepalive\n\ndata: {"type":"response.created","response":{"instructions":"'
        + b"x" * (1024 * 1024)
        + b'"}}\r\n\r\ndata: {"type":"error","message":"failed"}\n\n'
    )
    pending = observer._pending
    chunk_size = 4096
    feeds = 0
    for start in range(0, len(stream), chunk_size):
        assert observer.feed(stream[start : start + chunk_size]) == (
            start + chunk_size >= len(stream)
        )
        feeds += 1

    # Only the delimiter's three-byte boundary overlap may be scanned again.
    assert scanned_bytes <= len(stream) + 3 * feeds
    assert observer._pending is pending
    assert not pending


@pytest.mark.parametrize("delimiter", [b"\n\n", b"\r\n\r\n", b"\n\r\n", b"\r\n\n"])
def test_sse_error_observer_preserves_split_delimiters_and_error_tail(delimiter: bytes) -> None:
    """Compaction and early error returns preserve both split and buffered events."""
    prefix = b'data: {"choices":[]}' + delimiter
    error = b'data: {"error":{"message":"failed"}}'
    tail = b'data: {"error":{}}' + delimiter
    for split in range(1, len(delimiter)):
        observer = _SseErrorObserver()
        assert not observer.feed(prefix + error + delimiter[:split])
        assert not observer.feed(b"")
        assert observer.feed(delimiter[split:] + tail)
        assert observer._pending == tail
        # An error return must not mark the remaining complete event as scanned.
        assert observer.feed(b"")
        assert observer._pending == b""
        assert not observer.feed(b"")


@pytest.mark.parametrize(
    "event",
    [
        b'data: {"choices":[{"text":"error","delta":{"error":{}}}]}\n\n',
        b'data: {"error":"generated text"}\n\n',
        b'data: {"error":\n\n',
        b"data: [DONE]\r\n\r\n",
    ],
)
def test_sse_error_observer_ignores_non_error_events(event: bytes) -> None:
    """Content, malformed payloads and DONE are not worker error envelopes."""
    observer = _SseErrorObserver()
    assert not observer.feed(event)
    assert observer._pending == b""


@pytest.mark.parametrize("event_type", ["error", "response.failed"])
def test_sse_error_observer_responses_types_do_not_change_completion_events(
    event_type: str,
) -> None:
    event = b"data: " + msgspec.json.encode({"type": event_type}) + b"\n\n"
    assert not _SseErrorObserver().feed(event)


@pytest.mark.asyncio
@pytest.mark.parametrize("fragmented", [False, True])
@pytest.mark.parametrize("protocol", ["completion", "responses_error", "responses_failed"])
async def test_gen_first_sse_error_cancels_ctx_before_done(
    generation_first_stream_service: OpenAIDisaggregatedService, fragmented: bool, protocol: str
) -> None:
    """An error event closes CTX before GEN EOF and retains every original byte."""
    service = generation_first_stream_service
    ctx_started, ctx_closed, gen_release = asyncio.Event(), asyncio.Event(), asyncio.Event()
    if protocol == "completion":
        request = CompletionRequest(model="test-model", prompt="hello", stream=True)
        event = b'data: {"error":{"message":"executor failed"}}\r\n\r\n'
        trailing = [b"data: [DONE]\n\n"]
    else:
        request = ResponsesRequest(model="test-model", input="hello", stream=True)
        snapshot = _make_responses_response("failed").model_dump()
        snapshot["error"] = {"code": "server_error", "message": "executor failed"}
        failed_event = _format_sse_event(
            ResponseFailedEvent(type="response.failed", sequence_number=3, response=snapshot)
        ).encode()
        if protocol == "responses_error":
            event = _format_sse_event(
                ResponseErrorEvent(
                    type="error",
                    sequence_number=2,
                    code="engine_error",
                    message="executor failed",
                    param=None,
                )
            ).encode()
            trailing = [failed_event]
        else:
            event = failed_event
            trailing = []
    chunks = [event[:9], event[9:-1], event[-1:]] if fragmented else [event]

    async def ctx(*args: Any, **kwargs: Any) -> None:
        ctx_started.set()
        try:
            await asyncio.Event().wait()
        finally:
            ctx_closed.set()

    async def gen() -> AsyncGenerator[bytes, None]:
        await ctx_started.wait()
        for chunk in chunks:
            yield chunk
        await gen_release.wait()
        for chunk in trailing:
            yield chunk

    service._ctx_client.send_request.side_effect = ctx
    service._gen_client.send_request.return_value = gen()
    response = await asyncio.wait_for(
        service._send_disagg_request(request),
        timeout=2,
    )
    try:
        assert ctx_closed.is_set()
        assert not gen_release.is_set()
        gen_release.set()
        assert [chunk async for chunk in response] == chunks + trailing
        service._ctx_client.send_request.assert_awaited_once()
        service._gen_client.send_request.assert_awaited_once()
    finally:
        await response.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("error_type", [RuntimeError, asyncio.CancelledError])
async def test_gen_first_sse_error_preserves_ctx_failure(
    generation_first_stream_service: OpenAIDisaggregatedService,
    error_type: type[BaseException],
) -> None:
    """An in-band GEN error must not suppress a concurrent CTX failure or cancel."""
    service = generation_first_stream_service
    ctx_started, ctx_release, gen_closed = asyncio.Event(), asyncio.Event(), asyncio.Event()
    error = error_type("CTX interrupted")

    async def ctx(*args: Any, **kwargs: Any) -> None:
        ctx_started.set()
        await ctx_release.wait()
        raise error

    async def gen() -> AsyncGenerator[bytes, None]:
        try:
            await ctx_started.wait()
            ctx_release.set()
            yield b'data: {"error":{"message":"GEN failed"}}\n\n'
            await asyncio.Event().wait()
        finally:
            gen_closed.set()

    service._ctx_client.send_request.side_effect = ctx
    service._gen_client.send_request.return_value = gen()
    with pytest.raises(error_type, match="CTX interrupted"):
        await asyncio.wait_for(
            service._send_disagg_request(
                CompletionRequest(model="test-model", prompt="hello", stream=True)
            ),
            timeout=2,
        )
    assert gen_closed.is_set()


@pytest.mark.asyncio
async def test_gen_first_sse_error_drain_preserves_caller_cancellation(
    generation_first_stream_service: OpenAIDisaggregatedService,
) -> None:
    """Caller cancellation during intentional CTX cleanup still closes both legs."""
    service = generation_first_stream_service
    ctx_started, ctx_draining = asyncio.Event(), asyncio.Event()
    ctx_closed, gen_closed = asyncio.Event(), asyncio.Event()

    async def ctx(*args: Any, **kwargs: Any) -> None:
        ctx_started.set()
        try:
            await asyncio.Event().wait()
        finally:
            ctx_draining.set()
            try:
                await asyncio.Event().wait()
            finally:
                ctx_closed.set()

    async def gen() -> AsyncGenerator[bytes, None]:
        try:
            await ctx_started.wait()
            yield b'data: {"error":{"message":"GEN failed"}}\n\n'
            await asyncio.Event().wait()
        finally:
            gen_closed.set()

    service._ctx_client.send_request.side_effect = ctx
    service._gen_client.send_request.return_value = gen()
    pending = asyncio.create_task(
        service._send_disagg_request(
            CompletionRequest(model="test-model", prompt="hello", stream=True)
        )
    )
    try:
        await asyncio.wait_for(ctx_draining.wait(), timeout=2)
    finally:
        pending.cancel()
        await asyncio.gather(pending, return_exceptions=True)
    assert pending.cancelled()
    assert ctx_closed.is_set()
    assert gen_closed.is_set()


@pytest.mark.asyncio
@pytest.mark.parametrize("error_type", [RuntimeError, asyncio.CancelledError])
async def test_gen_first_stream_failure_after_handoff(
    generation_first_stream_service: OpenAIDisaggregatedService,
    error_type: type[BaseException],
) -> None:
    """The response iterator observes GEN errors after CTX has completed."""
    service = generation_first_stream_service
    fail, gen_closed = asyncio.Event(), asyncio.Event()
    error = error_type("generation failed after handoff")

    async def chunks() -> AsyncGenerator[bytes, None]:
        try:
            yield b'data: "first"\n\n'
            await fail.wait()
            raise error
        finally:
            gen_closed.set()

    service._gen_client.send_request.return_value = chunks()
    response = await service._send_disagg_request(
        CompletionRequest(model="test-model", prompt="hello", stream=True)
    )
    try:
        assert await asyncio.wait_for(response.__anext__(), timeout=2) == b'data: "first"\n\n'
        fail.set()
        with pytest.raises(error_type, match=str(error)):
            await asyncio.wait_for(response.__anext__(), timeout=2)
        assert gen_closed.is_set()
    finally:
        await response.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("error_type", [RuntimeError, asyncio.CancelledError])
@pytest.mark.parametrize("failed_role", ["ctx", "gen"])
async def test_gen_first_failure_drains_nonstream_sibling(
    generation_first_stream_service: OpenAIDisaggregatedService,
    error_type: type[BaseException],
    failed_role: str,
) -> None:
    """Either leg failing must promptly close the other pending HTTP request."""
    service = generation_first_stream_service
    sibling_started, sibling_closed = asyncio.Event(), asyncio.Event()
    sibling_tasks = []
    error = error_type("worker interrupted")

    async def fail(*args: Any, **kwargs: Any) -> None:
        await sibling_started.wait()
        raise error

    async def pending(*args: Any, **kwargs: Any) -> None:
        sibling_tasks.append(asyncio.current_task())
        sibling_started.set()
        try:
            await asyncio.Event().wait()
        finally:
            sibling_closed.set()

    clients = {"ctx": service._ctx_client, "gen": service._gen_client}
    clients[failed_role].send_request.side_effect = fail
    clients["gen" if failed_role == "ctx" else "ctx"].send_request.side_effect = pending
    try:
        with pytest.raises(error_type) as raised:
            await asyncio.wait_for(
                service._send_disagg_request(CompletionRequest(model="test-model", prompt="hello")),
                timeout=2,
            )
        assert raised.value is error
        assert sibling_closed.is_set()
        assert all(task.done() for task in sibling_tasks)
    finally:
        for task in sibling_tasks:
            task.cancel()
        await asyncio.gather(*sibling_tasks, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("protocol", "finish_reason"),
    [
        ("completion", "stop"),
        ("chat", "tool_calls"),
        ("completion", "cancelled"),
        ("chat", "timeout"),
        ("responses", "completed"),
        ("responses", "incomplete"),
    ],
)
async def test_gen_first_completed_response_survives_ctx_error(
    generation_first_stream_service: OpenAIDisaggregatedService,
    protocol: str,
    finish_reason: str,
) -> None:
    """A late CTX error cannot rewrite an already received GEN response."""
    service = generation_first_stream_service
    gen_finished, ctx_release = asyncio.Event(), asyncio.Event()
    if protocol == "responses":
        expected = _make_responses_response(finish_reason)
        request = ResponsesRequest(model="test-model", input="hello")
    else:
        expected = (
            _make_chat_response(finish_reason)
            if protocol == "chat"
            else _make_completion_response("answer", finish_reason, context_only=False)
        )
        expected.choices[0].disaggregated_params.request_type = "generation_only"
        request = (
            ChatCompletionRequest(
                model="test-model", messages=[{"role": "user", "content": "hello"}]
            )
            if protocol == "chat"
            else CompletionRequest(model="test-model", prompt="hello")
        )

    async def ctx(*args: Any, **kwargs: Any) -> None:
        await ctx_release.wait()
        raise RuntimeError("late CTX failure")

    async def gen(
        *args: Any, **kwargs: Any
    ) -> CompletionResponse | ChatCompletionResponse | ResponsesResponse:
        gen_finished.set()
        return expected

    service._ctx_client.send_request.side_effect = ctx
    service._gen_client.send_request.side_effect = gen
    with mock.patch("tensorrt_llm.serve.openai_disagg_service.logger.warning") as warning:
        pending = asyncio.create_task(service._send_disagg_request(request))
        try:
            await asyncio.wait_for(gen_finished.wait(), timeout=2)
            assert not pending.done()
            ctx_release.set()
            response = await asyncio.wait_for(pending, timeout=2)
            assert response is expected
            if protocol == "responses":
                assert response.status == finish_reason
            else:
                assert response.choices[0].finish_reason == finish_reason
            warning.assert_called_once()
        finally:
            pending.cancel()
            await asyncio.gather(pending, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("gen_outcome", ["response", "error", "cancelled", "empty"])
async def test_gen_first_ctx_error_with_both_nonstream_tasks_done(
    generation_first_stream_service: OpenAIDisaggregatedService, gen_outcome: str
) -> None:
    """Only a concrete GEN response overrides CTX failure when both tasks finish."""
    service = generation_first_stream_service
    ctx_error = RuntimeError("CTX failure")
    expected = _make_completion_response("answer", "stop", context_only=False)
    service._ctx_client.send_request.side_effect = ctx_error
    if gen_outcome == "error":
        service._gen_client.send_request.side_effect = RuntimeError("GEN failure")
    elif gen_outcome == "cancelled":
        service._gen_client.send_request.side_effect = asyncio.CancelledError("GEN cancelled")
    else:
        service._gen_client.send_request.return_value = (
            expected if gen_outcome == "response" else None
        )

    with mock.patch("tensorrt_llm.serve.openai_disagg_service.logger.warning") as warning:
        pending = asyncio.create_task(
            service._send_disagg_request(CompletionRequest(model="test-model", prompt="hello"))
        )
        try:
            if gen_outcome == "response":
                assert await asyncio.wait_for(pending, timeout=2) is expected
                warning.assert_called_once()
            else:
                with pytest.raises(RuntimeError) as raised:
                    await asyncio.wait_for(pending, timeout=2)
                assert raised.value is ctx_error
                warning.assert_not_called()
        finally:
            pending.cancel()
            await asyncio.gather(pending, return_exceptions=True)
    service._ctx_client.send_request.assert_awaited_once()
    service._gen_client.send_request.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel_origin", ["caller", "ctx", "caller_cleanup_error"])
@pytest.mark.parametrize("responses_api", [False, True])
async def test_gen_first_completed_response_does_not_swallow_cancellation(
    generation_first_stream_service: OpenAIDisaggregatedService,
    cancel_origin: str,
    responses_api: bool,
) -> None:
    """An available GEN response cannot suppress caller or CTX cancellation."""
    service = generation_first_stream_service
    awaiting_ctx, ctx_release, ctx_closed = asyncio.Event(), asyncio.Event(), asyncio.Event()
    original_wait = asyncio.wait

    async def observe_wait(
        *args: Any, **kwargs: Any
    ) -> tuple[set[asyncio.Task], set[asyncio.Task]]:
        result = await original_wait(*args, **kwargs)
        awaiting_ctx.set()
        return result

    async def ctx(*args: Any, **kwargs: Any) -> None:
        try:
            await ctx_release.wait()
            raise asyncio.CancelledError("CTX cancelled")
        finally:
            ctx_closed.set()
            if cancel_origin == "caller_cleanup_error":
                raise RuntimeError("CTX cleanup failed")

    async def gen(*args: Any, **kwargs: Any) -> CompletionResponse | ResponsesResponse:
        if responses_api:
            return _make_responses_response()
        return _make_completion_response("answer", "stop", context_only=False)

    service._ctx_client.send_request.side_effect = ctx
    service._gen_client.send_request.side_effect = gen
    with (
        mock.patch("tensorrt_llm.serve.openai_disagg_service.logger.warning") as warning,
        mock.patch(
            "tensorrt_llm.serve.openai_disagg_service.asyncio.wait", side_effect=observe_wait
        ),
    ):
        request = (
            ResponsesRequest(model="test-model", input="hello")
            if responses_api
            else CompletionRequest(model="test-model", prompt="hello")
        )
        pending = asyncio.create_task(service._send_disagg_request(request))
        try:
            await asyncio.wait_for(awaiting_ctx.wait(), timeout=2)
            if cancel_origin != "ctx":
                pending.cancel()
            else:
                ctx_release.set()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(pending, timeout=2)
            assert ctx_closed.is_set()
            warning.assert_not_called()
        finally:
            pending.cancel()
            await asyncio.gather(pending, return_exceptions=True)


@pytest.mark.asyncio
async def test_gen_first_caller_cancellation_drains_both_nonstream_requests(
    generation_first_stream_service: OpenAIDisaggregatedService,
) -> None:
    """Caller cancellation cannot leave either worker HTTP task behind."""
    service = generation_first_stream_service
    started = [asyncio.Event(), asyncio.Event()]
    closed = [asyncio.Event(), asyncio.Event()]

    async def pending(index: int) -> None:
        started[index].set()
        try:
            await asyncio.Event().wait()
        finally:
            closed[index].set()

    async def ctx(*args: Any, **kwargs: Any) -> None:
        await pending(0)

    async def gen(*args: Any, **kwargs: Any) -> None:
        await pending(1)

    service._ctx_client.send_request.side_effect = ctx
    service._gen_client.send_request.side_effect = gen
    task = asyncio.create_task(
        service._send_disagg_request(CompletionRequest(model="test-model", prompt="hello"))
    )
    try:
        await asyncio.wait_for(asyncio.gather(*(event.wait() for event in started)), timeout=2)
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
    assert task.cancelled()
    assert all(event.is_set() for event in closed)


@pytest.mark.asyncio
async def test_gen_only_stream_keeps_caller_ownership(
    generation_first_stream_service: OpenAIDisaggregatedService,
) -> None:
    """GEN-only requests keep their lazy stream and do not acquire a CTX task."""
    service = generation_first_stream_service
    service._check_gen_only_disagg.return_value = True
    gen_response = AsyncMock()
    service._gen_client.send_request.return_value = gen_response
    response = await service._send_disagg_request(
        CompletionRequest(model="test-model", prompt="hello", stream=True)
    )
    assert response is gen_response
    service._ctx_client.send_request.assert_not_awaited()
    gen_response.__aiter__.assert_not_called()
    gen_response.aclose.assert_not_awaited()
