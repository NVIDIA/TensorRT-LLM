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
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import torch

from tensorrt_llm.executor.result import Logprob
from tensorrt_llm.llmapi.disagg_utils import (
    DisaggClusterConfig,
    DisaggServerConfig,
    MinimalInstances,
    ServerRole,
)
from tensorrt_llm.serve.disagg_auto_scaling import DisaggClusterManager, WorkerInfo
from tensorrt_llm.serve.openai_disagg_service import OpenAIDisaggregatedService
from tensorrt_llm.serve.openai_protocol import (
    CompletionRequest,
    CompletionResponse,
    CompletionResponseChoice,
    DisaggregatedParams,
    DisaggScheduleStyle,
    PromptTokensDetails,
    UsageInfo,
    _deserialize_first_gen_log_probs,
    _deserialize_first_gen_logits,
    _serialize_first_gen_log_probs,
    _serialize_first_gen_logits,
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


async def _mock_streaming_response(chunks):
    for chunk in chunks:
        await asyncio.sleep(0)
        yield chunk


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
    service._disagg_cluster_manager = cluster_manager

    cluster_manager._current_ctx_workers["ctx"] = WorkerInfo(
        worker_id="ctx", role=ServerRole.CONTEXT
    )
    service._ctx_router = SimpleNamespace(num_prepared_servers=0)
    service._gen_router = SimpleNamespace(num_prepared_servers=1)
    assert await service.is_ready() is False

    service._ctx_router.num_prepared_servers = 1
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
                prompt_tokens=101,
                completion_tokens=13,
                cached_tokens=101,
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
            assert result.usage.prompt_tokens == 101
            assert result.usage.completion_tokens == 13
            assert result.usage.total_tokens == 114
            assert result.usage.prompt_tokens_details.cached_tokens == 7
            assert len(result.choices) == 1
            assert result.choices[0].text == resp_text
            assert result.choices[0].finish_reason == "stop"
            assert (
                result.choices[0].disaggregated_params.disagg_request_id
                == ctx_req.disaggregated_params.disagg_request_id
            )


@pytest.mark.asyncio
@pytest.mark.parametrize("schedule_style", ["context_first", "generation_first"])
async def test_send_disagg_request_rewrites_streaming_usage(schedule_style):
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
        usage_chunk = {
            "choices": [],
            "model": "test-model",
            "usage": {
                "prompt_tokens": 128,
                "completion_tokens": 5,
                "total_tokens": 133,
                "prompt_tokens_details": {
                    "cached_tokens": 128,
                },
            },
        }
        return _mock_streaming_response(
            [
                (
                    b'data: {"choices":[{"delta":{"content":"hello"},"index":0}],'
                    b'"model":"test-model"}\n\n'
                ),
                f"data: {json.dumps(usage_chunk)}\n\n".encode(),
                b"data: [DONE]\n\n",
            ]
        )

    service._ctx_client.send_request = AsyncMock(side_effect=_ctx_response)
    service._gen_client.send_request = AsyncMock(side_effect=_gen_response)

    request = CompletionRequest(model="test-model", prompt="hello", stream=True)
    result = await service._send_disagg_request(request)
    chunks = [chunk async for chunk in result]

    usage = json.loads(chunks[1].decode().removeprefix("data: "))["usage"]
    assert usage["prompt_tokens"] == 128
    assert usage["completion_tokens"] == 5
    assert usage["total_tokens"] == 133
    assert usage["prompt_tokens_details"]["cached_tokens"] == 9


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
        resp = _make_completion_response("", finish_reason="stop", disagg_request_id=555)
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
