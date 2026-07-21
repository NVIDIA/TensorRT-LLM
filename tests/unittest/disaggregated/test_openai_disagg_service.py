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
from types import SimpleNamespace
from unittest import mock
from unittest.mock import AsyncMock

import pytest
import torch

from tensorrt_llm.disaggregated_params import DisaggregatedParams as LlmDisaggregatedParams
from tensorrt_llm.executor.result import Logprob
from tensorrt_llm.llmapi.disagg_utils import (
    ConditionalDisaggConfig,
    DisaggClusterConfig,
    DisaggServerConfig,
    MinimalInstances,
    ServerRole,
)
from tensorrt_llm.serve.disagg_auto_scaling import DisaggClusterManager, WorkerInfo
from tensorrt_llm.serve.disagg_coordinator import DisaggCoordinatorService
from tensorrt_llm.serve.openai_disagg_service import OpenAIDisaggregatedService
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
    UsageInfo,
    _deserialize_first_gen_log_probs,
    _deserialize_first_gen_logits,
    _serialize_first_gen_log_probs,
    _serialize_first_gen_logits,
)
from tensorrt_llm.serve.postprocess_handlers import (
    ChatPostprocArgs,
    CompletionPostprocArgs,
    chat_response_post_processor,
    completion_response_post_processor,
)
from tensorrt_llm.serve.router import KvCacheAwareRouter, Router


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


@pytest.mark.asyncio
async def test_generation_first_refreshes_final_endpoint_and_strips_generation():
    service = _make_service("generation_first")
    request = CompletionRequest(model="model", prompt=[1, 2, 3])
    service._coordinator.get_disagg_request_id = AsyncMock(return_value=42)
    service._ctx_router.get_next_server = AsyncMock(
        return_value=(
            "ctx:8000",
            {
                "server_info": {
                    "disaggregated_params": {
                        "ctx_info_endpoint": ["tcp://stale:1000"],
                        "ctx_endpoint_generation": "stale-generation",
                    }
                }
            },
        )
    )
    service._ctx_router.get_runtime_server_info = AsyncMock(
        return_value={
            "disaggregated_params": {
                "ctx_dp_rank": 0,
                "ctx_info_endpoint": ["tcp://final:2000"],
                "ctx_endpoint_generation": "final-generation",
            }
        }
    )
    captured_gen_request = None

    async def send_gen(gen_request, **_kwargs):
        nonlocal captured_gen_request
        captured_gen_request = gen_request
        return _make_completion_response("done", "length", context_only=False)

    service._gen_client = SimpleNamespace(send_request=AsyncMock(side_effect=send_gen))
    service._ctx_client = SimpleNamespace(send_request=AsyncMock(return_value=None))

    await service._send_disagg_request_gen_first(request)

    service._ctx_router.get_runtime_server_info.assert_awaited_once_with(
        "ctx:8000", require_generation=True
    )
    assert captured_gen_request.disaggregated_params.ctx_info_endpoint == "tcp://final:2000"
    assert not hasattr(captured_gen_request.disaggregated_params, "ctx_endpoint_generation")


@pytest.mark.asyncio
async def test_generation_first_flags_off_uses_valid_cached_endpoint():
    service = _make_service("generation_first")
    request = CompletionRequest(model="model", prompt=[1, 2, 3])
    service._coordinator.get_disagg_request_id = AsyncMock(return_value=45)
    service._ctx_router.get_next_server = AsyncMock(
        return_value=(
            "ctx:8000",
            {
                "server_info": {
                    "disaggregated_params": {
                        "ctx_info_endpoint": ["tcp://legacy:1000"],
                    }
                }
            },
        )
    )
    service._ctx_router.get_runtime_server_info = AsyncMock()
    service._ctx_client = SimpleNamespace(send_request=AsyncMock(return_value=None))
    service._gen_client = SimpleNamespace(
        send_request=AsyncMock(
            return_value=_make_completion_response("done", "length", context_only=False)
        )
    )

    await service._send_disagg_request_gen_first(request)

    service._ctx_router.get_runtime_server_info.assert_not_awaited()


@pytest.mark.asyncio
async def test_generation_first_refresh_failure_releases_ctx_reservation():
    service = _make_service("generation_first")
    request = CompletionRequest(model="model", prompt=[1, 2, 3])
    service._coordinator.get_disagg_request_id = AsyncMock(return_value=43)
    service._ctx_router.get_next_server = AsyncMock(return_value=("ctx:8000", {"server_info": {}}))
    service._ctx_router.get_runtime_server_info = AsyncMock(side_effect=RuntimeError("not final"))
    service._ctx_router.finish_request = AsyncMock()

    with pytest.raises(RuntimeError, match="not final"):
        await service._send_disagg_request_gen_first(request)

    service._ctx_router.finish_request.assert_awaited_once_with(request, success=False, req_id=43)


@pytest.mark.asyncio
async def test_generation_first_cancelled_refresh_still_releases_reservation():
    service = _make_service("generation_first")
    request = CompletionRequest(model="model", prompt=[1, 2, 3])
    service._coordinator.get_disagg_request_id = AsyncMock(return_value=44)
    service._ctx_router.get_next_server = AsyncMock(return_value=("ctx:8000", {"server_info": {}}))
    refresh_started = asyncio.Event()

    async def refresh(*_args, **_kwargs):
        refresh_started.set()
        await asyncio.Event().wait()

    service._ctx_router.get_runtime_server_info = AsyncMock(side_effect=refresh)
    cleanup_finished = asyncio.Event()

    async def finish(*_args, **_kwargs):
        await asyncio.sleep(0)
        cleanup_finished.set()

    service._ctx_router.finish_request = AsyncMock(side_effect=finish)

    service_task = asyncio.create_task(service._send_disagg_request_gen_first(request))
    await refresh_started.wait()
    service_task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await service_task

    await asyncio.wait_for(cleanup_finished.wait(), timeout=1.0)
    service._ctx_router.finish_request.assert_awaited_once_with(request, success=False, req_id=44)


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
