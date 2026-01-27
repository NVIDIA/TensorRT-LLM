import asyncio
from unittest.mock import AsyncMock

import pytest

from tensorrt_llm.llmapi.disagg_utils import DisaggServerConfig
from tensorrt_llm.serve.openai_disagg_service import OpenAIDisaggregatedService
from tensorrt_llm.serve.openai_protocol import (
    CompletionRequest,
    CompletionResponse,
    CompletionResponseChoice,
    DisaggregatedParams,
    DisaggScheduleStyle,
    UsageInfo,
)
from tensorrt_llm.serve.router import Router


def _client_factory(*_args, **_kwargs):
    return AsyncMock()


def _make_service(schedule_style: str) -> OpenAIDisaggregatedService:
    config = DisaggServerConfig(server_configs=[], schedule_style=schedule_style)
    ctx_router = AsyncMock(spec=Router)
    gen_router = AsyncMock(spec=Router)
    return OpenAIDisaggregatedService(
        config, ctx_router, gen_router, client_factory=_client_factory
    )


def _make_completion_response(
    text: str,
    finish_reason: str,
    disagg_request_id: int = 42,
    prompt_token_ids=None,
    context_only=True,
) -> CompletionResponse:
    if prompt_token_ids is None:
        prompt_token_ids = [1, 2, 3]
    return CompletionResponse(
        model="test-model",
        usage=UsageInfo(prompt_tokens=1, completion_tokens=1),
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


async def _mock_streaming_response(chunks):
    for chunk in chunks:
        await asyncio.sleep(0)
        yield chunk


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
        assert (
            gen_req.disaggregated_params.ctx_request_id
            == ctx_req.disaggregated_params.disagg_request_id
        )

        if stream:
            assert hasattr(result, "__aiter__")
            chunks = [chunk async for chunk in result]
            assert chunks == stream_chunks
        else:
            assert result.model == "test-model"
            assert result.usage.prompt_tokens == 1
            assert len(result.choices) == 1
            assert result.choices[0].text == resp_text
            assert result.choices[0].finish_reason == "stop"
            assert (
                result.choices[0].disaggregated_params.disagg_request_id
                == ctx_req.disaggregated_params.disagg_request_id
            )
