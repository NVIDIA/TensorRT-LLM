# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
from types import SimpleNamespace

import pytest
from openengine.v1 import (
    generation_pb2,
    input_pb2,
    kv_pb2,
    lifecycle_pb2,
    lora_pb2,
    model_pb2,
    observability_pb2,
    server_pb2,
)

from tensorrt_llm.disaggregated_params import DisaggregatedParams, DisaggScheduleStyle
from tensorrt_llm.inputs.registry import MultimodalLoraSpec
from tensorrt_llm.openengine.converters import decode_handoff, encode_handoff
from tensorrt_llm.openengine.servicer import OpenEngineServicer
from tensorrt_llm.serve.kv_event_fanout import KvEventFanout
from tensorrt_llm.serve.request_tracker import RequestTracker
from tensorrt_llm.serve.stats_fanout import StatsFanout


class _Tokenizer:
    def decode(self, token_ids: list[int], **_: object) -> str:
        return "".join(chr(96 + token_id) for token_id in token_ids)


class _Result:
    def __init__(self, handoff: DisaggregatedParams | None = None) -> None:
        self.prompt_token_ids = [1, 2]
        self.cached_tokens = 1
        self.disaggregated_params = handoff
        self.outputs = [
            SimpleNamespace(
                index=0,
                token_ids=[3, 4],
                text="cd",
                logprobs=[],
                prompt_logprobs=[],
                finish_reason="length",
                stop_reason=None,
                disaggregated_params=handoff,
            )
        ]
        self.finished = True
        self._yielded = False
        self.aborted = False

    def __aiter__(self) -> "_Result":
        return self

    async def __anext__(self) -> "_Result":
        if self._yielded:
            raise StopAsyncIteration
        self._yielded = True
        return self

    def abort(self) -> None:
        self.aborted = True


class _Llm:
    def __init__(self, handoff: DisaggregatedParams | None = None) -> None:
        self.tokenizer = _Tokenizer()
        self.args = SimpleNamespace()
        self.handoff = handoff
        self.kwargs = None
        self.result = None

    def generate_async(self, **kwargs: object) -> _Result:
        self.kwargs = kwargs
        self.result = _Result(self.handoff)
        return self.result


class _Context:
    def __init__(self, cancelled: bool = False, metadata: tuple[tuple[str, str], ...] = ()) -> None:
        self._cancelled = cancelled
        self._metadata = metadata

    def cancelled(self) -> bool:
        return self._cancelled

    def invocation_metadata(self) -> tuple[tuple[str, str], ...]:
        return self._metadata

    async def abort(self, code: object, message: str) -> None:
        raise AssertionError(f"unexpected abort {code}: {message}")


async def _collect_async(iterator):
    return [item async for item in iterator]


@pytest.mark.asyncio
async def test_aggregate_streams_deltas_finish_and_terminal_usage() -> None:
    llm = _Llm()
    tracker = RequestTracker(llm)
    servicer = OpenEngineServicer(llm, "model", server_pb2.ENGINE_ROLE_AGGREGATED, tracker)
    request = generation_pb2.GenerateRequest(
        request_id="request",
        model="model",
        prompt="hello",
        stopping=generation_pb2.StoppingOptions(max_tokens=2),
    )

    context = _Context(
        metadata=(
            (
                "traceparent",
                "00-0123456789abcdef0123456789abcdef-0123456789abcdef-01",
            ),
            ("tracestate", "vendor=value"),
            ("openengine-priority", "7"),
            ("ignored", "not-a-trace-header"),
        )
    )
    responses = [response async for response in servicer.Generate(request, context)]

    assert [response.WhichOneof("event") for response in responses] == [
        "token",
        "finished",
    ]
    assert list(responses[0].token.tokens)[0].token_id == 3
    assert responses[0].token.text == "cd"
    assert responses[-1].usage.prompt_tokens == 2
    assert responses[-1].usage.completion_tokens == 2
    assert llm.kwargs["trace_headers"] == {
        "traceparent": "00-0123456789abcdef0123456789abcdef-0123456789abcdef-01",
        "tracestate": "vendor=value",
    }
    assert 0.5 < llm.kwargs["priority"] < 1
    assert tracker.active_count == 0


@pytest.mark.asyncio
async def test_generate_cancellation_aborts_and_cleans_tracking() -> None:
    llm = _Llm()
    tracker = RequestTracker(llm)
    service = OpenEngineServicer(llm, "model", server_pb2.ENGINE_ROLE_AGGREGATED, tracker)
    request = generation_pb2.GenerateRequest(request_id="cancel", model="model", prompt="hello")

    responses = [response async for response in service.Generate(request, _Context(True))]

    assert responses == []
    assert llm.result.aborted
    assert tracker.active_count == 0


def test_generate_rejects_unsupported_cache_bypass_and_nonzero_dp_placement() -> None:
    llm = _Llm()
    service = OpenEngineServicer(
        llm, "model", server_pb2.ENGINE_ROLE_AGGREGATED, RequestTracker(llm)
    )
    request = generation_pb2.GenerateRequest(request_id="request", model="model", prompt="hello")
    request.kv.bypass_prefix_cache = True
    with pytest.raises(ValueError, match="Prefix-cache bypass"):
        service._validate_generate(request)

    request.kv.bypass_prefix_cache = False
    service._validate_generate(request, 0)
    assert service._scheduling_params(0) is None

    llm.args = SimpleNamespace(data_parallel_size=2)
    with pytest.raises(ValueError, match="attention DP"):
        service._validate_generate(request, 1)


@pytest.mark.asyncio
async def test_generate_applies_strict_attention_dp_placement() -> None:
    llm = _Llm()
    llm.args = SimpleNamespace(enable_attention_dp=True, data_parallel_size=4)
    llm._on_trt_backend = False
    service = OpenEngineServicer(
        llm, "model", server_pb2.ENGINE_ROLE_AGGREGATED, RequestTracker(llm)
    )
    request = generation_pb2.GenerateRequest(request_id="request", model="model", prompt="hello")

    context = _Context(metadata=(("openengine-target-dp-rank", "2"),))
    _ = [response async for response in service.Generate(request, context)]
    scheduling = llm.kwargs["scheduling_params"]

    assert scheduling.attention_dp_rank == 2
    assert not scheduling.attention_dp_relax


@pytest.mark.asyncio
async def test_generate_rejects_non_decimal_priority_metadata() -> None:
    llm = _Llm()
    service = OpenEngineServicer(
        llm, "model", server_pb2.ENGINE_ROLE_AGGREGATED, RequestTracker(llm)
    )
    request = generation_pb2.GenerateRequest(request_id="request", model="model", prompt="hello")
    context = _Context(metadata=(("openengine-priority", "1_000"),))

    with pytest.raises(AssertionError, match="INVALID_ARGUMENT.*base-10 integer"):
        _ = [response async for response in service.Generate(request, context)]

    assert llm.kwargs is None


def test_prefill_preserves_routing_rank_in_context_handoff() -> None:
    llm = _Llm()
    llm.args = SimpleNamespace(enable_attention_dp=True, data_parallel_size=4)
    llm._on_trt_backend = False
    service = OpenEngineServicer(llm, "model", server_pb2.ENGINE_ROLE_PREFILL, RequestTracker(llm))
    request = generation_pb2.GenerateRequest(request_id="prefill", model="model", prompt="hello")

    params = service._disaggregated_params(request, 3)

    assert params.ctx_dp_rank == 3


@pytest.mark.asyncio
async def test_generate_selects_model_owned_multimodal_lora(monkeypatch, tmp_path) -> None:
    class _Processor:
        def get_openengine_modalities(self) -> tuple[str, ...]:
            return ("audio",)

        def get_openengine_prefill_decode_modalities(self) -> tuple[str, ...]:
            return ()

        def get_required_lora_spec(self, modalities: tuple[str, ...]) -> MultimodalLoraSpec | None:
            assert modalities == ("audio",)
            return MultimodalLoraSpec("speech-lora", 1, str(tmp_path))

    async def _load_media(*_args, **_kwargs):
        return {"audio": [object()]}

    monkeypatch.setattr("tensorrt_llm.openengine.servicer.BaseMultimodalInputProcessor", _Processor)
    monkeypatch.setattr("tensorrt_llm.openengine.servicer.load_media", _load_media)
    llm = _Llm()
    llm.args = SimpleNamespace(lora_config=object())
    llm.input_processor = _Processor()
    service = OpenEngineServicer(
        llm, "model", server_pb2.ENGINE_ROLE_AGGREGATED, RequestTracker(llm)
    )
    request = generation_pb2.GenerateRequest(request_id="audio", model="model", prompt="transcribe")
    request.media.add(modality=input_pb2.MODALITY_AUDIO, raw_bytes=b"audio")

    _ = [response async for response in service.Generate(request, _Context())]

    assert llm.kwargs["lora_request"].lora_name == "speech-lora"
    assert llm.kwargs["lora_request"].lora_int_id == 1
    assert await service.loras.list() == []


@pytest.mark.asyncio
async def test_model_info_suppresses_modalities_when_required_lora_is_unavailable(
    monkeypatch, tmp_path
) -> None:
    class _Processor:
        def get_openengine_modalities(self) -> tuple[str, ...]:
            return ("audio",)

        def get_openengine_prefill_decode_modalities(self) -> tuple[str, ...]:
            return ()

        def get_required_lora_spec(self, modalities: tuple[str, ...]) -> MultimodalLoraSpec | None:
            assert modalities == ("audio",)
            return MultimodalLoraSpec("speech-lora", 1, str(tmp_path / "missing"))

    monkeypatch.setattr("tensorrt_llm.openengine.servicer.BaseMultimodalInputProcessor", _Processor)
    llm = _Llm()
    llm.input_processor = _Processor()
    service = OpenEngineServicer(
        llm, "model", server_pb2.ENGINE_ROLE_AGGREGATED, RequestTracker(llm)
    )

    info = await service.GetModelInfo(model_pb2.GetModelInfoRequest(), _Context())

    assert not info.supports_multimodal
    assert list(info.multimodal_capabilities.aggregate_modalities) == []


@pytest.mark.asyncio
async def test_context_then_generation_round_trips_handoff() -> None:
    context_handoff = DisaggregatedParams(
        request_type="context_only",
        first_gen_tokens=[7],
        ctx_request_id=100,
        disagg_request_id=101,
        ctx_info_endpoint="tcp://context:1",
        schedule_style=DisaggScheduleStyle.CONTEXT_FIRST,
        opaque_state=b"state",
    )
    context_llm = _Llm(context_handoff)
    context_service = OpenEngineServicer(
        context_llm,
        "model",
        server_pb2.ENGINE_ROLE_PREFILL,
        RequestTracker(context_llm),
    )
    context_request = generation_pb2.GenerateRequest(
        request_id="request",
        model="model",
        token_ids=input_pb2.TokenIds(ids=[1, 2]),
    )
    context_responses = [
        response async for response in context_service.Generate(context_request, _Context())
    ]
    assert len(context_responses) == 1
    assert context_responses[0].WhichOneof("event") == "prefill_ready"
    assert len(context_service._kv_session_requests) == 1
    context_load = await context_service.GetLoad(observability_pb2.GetLoadRequest(), _Context())
    assert context_load.active_kv_sessions == 0

    decode_llm = _Llm()
    decode_service = OpenEngineServicer(
        decode_llm,
        "model",
        server_pb2.ENGINE_ROLE_DECODE,
        RequestTracker(decode_llm),
    )
    decode_request = generation_pb2.GenerateRequest(
        request_id="decode",
        model="model",
        token_ids=input_pb2.TokenIds(ids=[1, 2]),
    )
    decode_request.kv.session.CopyFrom(context_responses[0].prefill_ready.kv_session)
    decode_responses = [
        response async for response in decode_service.Generate(decode_request, _Context())
    ]
    decoded = decode_llm.kwargs["disaggregated_params"]
    assert decoded.request_type == "generation_only"
    assert decoded.disagg_request_id == 101
    assert decoded.opaque_state == b"state"
    assert decoded.ctx_usage == {
        "prompt_tokens": 2,
        "completion_tokens": 2,
        "total_tokens": 4,
        "prompt_tokens_details": {"cached_tokens": 1},
    }
    terminal_usage = decode_responses[-1].usage
    assert terminal_usage.prompt_tokens == 2
    assert terminal_usage.cached_prompt_tokens == 1
    assert terminal_usage.completion_tokens == 2
    assert terminal_usage.total_tokens == 4

    abort_response = await context_service.Abort(
        lifecycle_pb2.AbortRequest(kv_session=context_responses[0].prefill_ready.kv_session),
        _Context(),
    )
    assert abort_response.status == lifecycle_pb2.ABORT_STATUS_ABORTED
    assert not context_service._kv_session_requests


@pytest.mark.asyncio
async def test_prefill_adds_context_endpoint_from_llm_discovery() -> None:
    handoff = DisaggregatedParams(
        request_type="context_only",
        first_gen_tokens=[7],
        disagg_request_id=101,
        schedule_style=DisaggScheduleStyle.CONTEXT_FIRST,
    )
    llm = _Llm(handoff)
    llm.disaggregated_params = {"ctx_info_endpoint": ["tcp://context:1234"]}
    service = OpenEngineServicer(
        llm,
        "model",
        server_pb2.ENGINE_ROLE_PREFILL,
        RequestTracker(llm),
    )
    request = generation_pb2.GenerateRequest(
        request_id="prefill",
        model="model",
        token_ids=input_pb2.TokenIds(ids=[1, 2]),
    )

    responses = [response async for response in service.Generate(request, _Context())]
    decoded = decode_handoff(responses[0].prefill_ready.kv_session)

    assert decoded.ctx_info_endpoint == "tcp://context:1234"


@pytest.mark.asyncio
async def test_prefill_session_expires_and_close_cancels_timers() -> None:
    handoff = DisaggregatedParams(
        request_type="context_only",
        disagg_request_id=101,
        schedule_style=DisaggScheduleStyle.CONTEXT_FIRST,
    )
    llm = _Llm(handoff)
    service = OpenEngineServicer(
        llm,
        "model",
        server_pb2.ENGINE_ROLE_PREFILL,
        RequestTracker(llm),
        kv_session_ttl_seconds=0.01,
    )
    request = generation_pb2.GenerateRequest(request_id="prefill", model="model", prompt="hello")

    _ = [response async for response in service.Generate(request, _Context())]
    assert service._kv_session_requests
    assert service._kv_session_timers
    await asyncio.sleep(0.02)
    assert not service._kv_session_requests
    assert not service._kv_session_timers

    service._track_kv_session("second", "request")
    timer = service._kv_session_timers["second"]
    service.close()
    assert timer.cancelled()
    assert not service._kv_session_requests


def test_decode_media_is_required_only_for_marked_handoff(monkeypatch) -> None:
    class _Processor:
        def get_openengine_modalities(self) -> tuple[str, ...]:
            return ("image", "video")

        def get_openengine_prefill_decode_modalities(self) -> tuple[str, ...]:
            return ("image", "video")

        def get_required_lora_spec(self, modalities: tuple[str, ...]) -> MultimodalLoraSpec | None:
            del modalities
            return None

    monkeypatch.setattr("tensorrt_llm.openengine.servicer.BaseMultimodalInputProcessor", _Processor)
    llm = _Llm()
    llm.input_processor = _Processor()
    service = OpenEngineServicer(llm, "model", server_pb2.ENGINE_ROLE_DECODE, RequestTracker(llm))
    session = encode_handoff(
        DisaggregatedParams(
            request_type="context_only",
            schedule_style=DisaggScheduleStyle.CONTEXT_FIRST,
            mrope_position_ids_handle={"tensor": "ids"},
            mrope_position_deltas_handle={"tensor": "deltas"},
        ),
        requires_decode_media=True,
    )
    request = generation_pb2.GenerateRequest(
        request_id="decode", model="model", token_ids=input_pb2.TokenIds(ids=[1, 2])
    )
    request.kv.session.CopyFrom(session)

    with pytest.raises(ValueError, match="must resend"):
        service._validate_generate(request)

    request.media.add(modality=input_pb2.MODALITY_IMAGE, raw_bytes=b"image")
    service._validate_generate(request)


def test_kv_batch_preserves_chain_geometry_and_source_sequence() -> None:
    llm = _Llm()
    llm.args = SimpleNamespace(kv_cache_config=SimpleNamespace(tokens_per_block=2))
    service = OpenEngineServicer(
        llm, "model", server_pb2.ENGINE_ROLE_AGGREGATED, RequestTracker(llm)
    )
    service._lora_names_by_id[7] = "adapter"
    batch = service._kv_batch(
        {
            "attention_dp_rank": 2,
            "layer_group_id": 3,
            "window_size": 4096,
            "hash_algo": "v2_sha256",
            "data": {
                "type": "stored",
                "parent_hash": "00" * 32,
                "blocks": [
                    {
                        "block_hash": "11" * 32,
                        "tokens": [{"token_id": 1}, {"token_id": 2}],
                        "mm_keys": [
                            {
                                "type": "mm_key",
                                "hash": "0123456789abcdef0011",
                                "start_offset": 4,
                            }
                        ],
                        "lora_id": 7,
                    },
                    {
                        "block_hash": "22" * 32,
                        "tokens": [{"token_id": 3}, {"token_id": 4}],
                        "lora_id": 7,
                    },
                ],
            },
        },
        sequence=9,
    )

    assert batch.sequence_number == 9
    assert batch.data_parallel_rank == 2
    assert len(batch.events) == 1
    stored = batch.events[0].block_stored
    assert len(stored.block_hashes) == 2
    assert stored.parent_block_hash.value == bytes.fromhex("00" * 32)
    assert list(stored.token_ids) == [1, 2, 3, 4]
    assert stored.block_size == 2
    assert stored.lora_id == 7
    assert stored.lora_name == "adapter"
    assert stored.group_idx == 3
    assert list(stored.extra_keys[0].values) == [
        "trt_mm_v1",
        "0",
        "81985529216486895",
        "4",
    ]


def test_kv_batch_stops_at_partial_block_and_uses_decimal_int_hashes() -> None:
    llm = _Llm()
    llm.args = SimpleNamespace(kv_cache_config=SimpleNamespace(tokens_per_block=2))
    service = OpenEngineServicer(
        llm, "model", server_pb2.ENGINE_ROLE_AGGREGATED, RequestTracker(llm)
    )
    batch = service._kv_batch(
        {
            "attention_dp_rank": 0,
            "data": {
                "type": "stored",
                "parent_hash": 2**64 - 1,
                "blocks": [
                    {
                        "block_hash": 2**63,
                        "tokens": [{"token_id": 1}, {"token_id": 2}],
                    },
                    {"block_hash": 9, "tokens": [{"token_id": 3}]},
                    {
                        "block_hash": 10,
                        "tokens": [{"token_id": 4}, {"token_id": 5}],
                    },
                ],
            },
        },
        sequence=3,
    )

    assert len(batch.events) == 1
    stored = batch.events[0].block_stored
    assert stored.parent_block_hash.encoding == "decimal_int64"
    assert stored.parent_block_hash.value == b"-1"
    assert stored.block_hashes[0].value == str(-(2**63)).encode()
    assert 9 in service._partial_block_hashes[0]


def test_kv_batch_preserves_model_owned_lora_id_zero_and_gap_reset() -> None:
    class _Processor:
        @staticmethod
        def get_model_owned_lora_identities() -> dict[str, int]:
            return {"vision-lora": 0}

    llm = _Llm()
    llm.input_processor = _Processor()
    llm.args = SimpleNamespace(kv_cache_config=SimpleNamespace(tokens_per_block=2))
    service = OpenEngineServicer(
        llm, "model", server_pb2.ENGINE_ROLE_AGGREGATED, RequestTracker(llm)
    )
    batch = service._kv_batch(
        {
            "data": {
                "type": "stored",
                "blocks": [
                    {
                        "block_hash": 2,
                        "tokens": [{"token_id": 1}, {"token_id": 2}],
                        "lora_id": 0,
                    }
                ],
            }
        },
        sequence=1,
    )
    service._partial_block_hashes[0] = {3}
    reset = service._kv_batch({"attention_dp_rank": 0, "data": {"type": "all_cleared"}}, sequence=2)

    assert batch.events[0].block_stored.lora_id == 0
    assert batch.events[0].block_stored.lora_name == "vision-lora"
    assert reset.events[0].WhichOneof("event") == "all_blocks_cleared"
    assert 0 not in service._partial_block_hashes


@pytest.mark.parametrize(
    "unsupported",
    [
        {"cache_salt": "private"},
        {"tokens": [{"token_id": 1, "token_extra_id": 1}, {"token_id": 2}]},
        {"mm_keys": [{"type": "mm_key", "hash": "not-hex"}]},
    ],
)
def test_kv_batch_fails_closed_for_unrepresentable_cache_namespace(unsupported) -> None:
    llm = _Llm()
    llm.args = SimpleNamespace(kv_cache_config=SimpleNamespace(tokens_per_block=2))
    service = OpenEngineServicer(
        llm, "model", server_pb2.ENGINE_ROLE_AGGREGATED, RequestTracker(llm)
    )
    block = {
        "block_hash": 2,
        "tokens": [{"token_id": 1}, {"token_id": 2}],
        **unsupported,
    }

    batch = service._kv_batch(
        {"data": {"type": "stored", "parent_hash": 1, "blocks": [block]}},
        sequence=1,
    )

    assert len(batch.events) == 1
    assert batch.events[0].WhichOneof("event") == "all_blocks_cleared"


@pytest.mark.asyncio
async def test_discovery_and_load_use_config_and_shared_stats(monkeypatch) -> None:
    class _Processor:
        def get_openengine_modalities(self) -> tuple[str, ...]:
            return ("image",)

        def get_openengine_prefill_decode_modalities(self) -> tuple[str, ...]:
            return ("image",)

        def get_openengine_routing_image_token_id(self) -> int:
            return 151655

        def get_required_lora_spec(self, modalities: tuple[str, ...]) -> None:
            del modalities
            return None

    monkeypatch.setattr("tensorrt_llm.openengine.servicer.BaseMultimodalInputProcessor", _Processor)
    llm = _Llm()
    llm.args = SimpleNamespace(
        model="Qwen/Qwen3-VL-2B-Instruct",
        tokenizer="Qwen/Qwen3-VL-2B-Instruct",
        tokenizer_mode="slow",
        guided_decoding_backend=None,
        lora_config=None,
        enable_lora=False,
        kv_cache_config=SimpleNamespace(tokens_per_block=64),
    )
    llm.input_processor = _Processor()
    llm.get_kv_cache_capacity = lambda: {
        "maxNumBlocks": 100,
        "tokensPerBlock": 64,
        "maxNumTokens": 6400,
    }
    stats = StatsFanout(llm)
    stats._publish(
        {
            "attentionDpRank": 1,
            "numActiveRequests": 3,
            "numQueuedRequests": 2,
            "kvCacheStats": {
                "maxNumBlocks": 100,
                "freeNumBlocks": 25,
                "tokensPerBlock": 64,
            },
            "inflightBatchingStats": {"numContextRequests": 1, "numGenRequests": 2},
        }
    )
    service = OpenEngineServicer(
        llm,
        "qwen3-vl",
        server_pb2.ENGINE_ROLE_AGGREGATED,
        RequestTracker(llm),
        stats_fanout=stats,
    )

    server_info = await service.GetServerInfo(server_pb2.GetServerInfoRequest(), _Context())
    assert server_info.schema_revision == 3
    assert server_info.minimum_client_revision == 1
    assert server_info.capacity.kv_block_size == 64
    assert server_info.capacity.total_kv_blocks == 100
    assert server_info.kv_connector.handoff_profile == "tensorrt_llm.disaggregated_params.v1"
    assert server_info.kv_connector.HasField("supports_client_bootstrap")
    assert not server_info.kv_connector.supports_client_bootstrap
    model_info = await service.GetModelInfo(model_pb2.GetModelInfoRequest(), _Context())
    assert model_info.model_id == "Qwen/Qwen3-VL-2B-Instruct"
    assert model_info.served_model_name == "qwen3-vl"
    assert list(model_info.served_model_aliases) == ["Qwen/Qwen3-VL-2B-Instruct"]
    assert model_info.tokenizer.source == "Qwen/Qwen3-VL-2B-Instruct"
    assert model_info.tokenizer.mode == "slow"
    assert model_info.multimodal_capabilities.routing_image_token_id == 151655
    assert not model_info.generation.guided_decoding.supported
    assert not model_info.supports_lora

    load = await service.GetLoad(
        observability_pb2.GetLoadRequest(include_per_rank=True), _Context()
    )
    assert load.running_requests == 3
    assert load.queued_requests == 2
    assert load.used_kv_blocks == 75
    assert load.total_kv_blocks == 100
    assert load.prefill_batch_size == 1
    assert load.decode_batch_size == 2
    assert load.ranks[0].data_parallel_rank == 1
    assert load.attributes["source"] == "shared_stats_fanout"
    assert load.attributes["kv_tokens_per_block"] == "64"
    assert load.attributes["rank.1.kv_tokens_per_block"] == "64"

    llm.args.guided_decoding_backend = "xgrammar"
    guided_info = await service.GetModelInfo(model_pb2.GetModelInfoRequest(), _Context())
    assert guided_info.generation.guided_decoding.supported
    assert (
        model_pb2.GUIDED_DECODING_MODE_STRUCTURAL_TAG
        in guided_info.generation.guided_decoding.modes
    )


@pytest.mark.asyncio
async def test_load_never_undercounts_live_tracker_from_stale_stats() -> None:
    llm = _Llm()
    stats = StatsFanout(llm)
    stats._publish({"attentionDpRank": 0, "numActiveRequests": 0})
    tracker = RequestTracker(llm)
    tracker.admit("active", _Result())
    service = OpenEngineServicer(
        llm,
        "model",
        server_pb2.ENGINE_ROLE_AGGREGATED,
        tracker,
        stats_fanout=stats,
    )

    load = await service.GetLoad(observability_pb2.GetLoadRequest(), _Context())

    assert load.running_requests == 1


@pytest.mark.asyncio
async def test_kv_event_sources_are_rank_scoped() -> None:
    llm = _Llm()
    llm.args = SimpleNamespace(
        data_parallel_size=2,
        kv_cache_config=SimpleNamespace(event_buffer_max_size=8),
    )
    fanout = KvEventFanout(llm, buffer_size=8)
    service = OpenEngineServicer(
        llm,
        "model",
        server_pb2.ENGINE_ROLE_AGGREGATED,
        RequestTracker(llm),
        kv_event_fanout=fanout,
    )

    response = await service.GetKvEventSources(kv_pb2.GetKvEventSourcesRequest(), _Context())

    assert [source.data_parallel_rank for source in response.sources] == [0, 1]


@pytest.mark.asyncio
async def test_kv_event_discovery_requires_decimal_compatible_hashes() -> None:
    llm = _Llm()
    llm.args = SimpleNamespace(
        kv_cache_config=SimpleNamespace(
            event_buffer_max_size=8,
            kv_cache_event_hash_algo="v2_sha256",
            use_kv_cache_manager_v2=True,
        )
    )
    service = OpenEngineServicer(
        llm,
        "model",
        server_pb2.ENGINE_ROLE_AGGREGATED,
        RequestTracker(llm),
        kv_event_fanout=KvEventFanout(llm),
    )

    response = await service.GetKvEventSources(kv_pb2.GetKvEventSourcesRequest(), _Context())
    assert list(response.sources) == []

    llm.args.kv_cache_config.kv_cache_event_hash_algo = "v2_sha256_64"
    response = await service.GetKvEventSources(kv_pb2.GetKvEventSourcesRequest(), _Context())
    assert len(response.sources) == 1


@pytest.mark.asyncio
async def test_direct_kv_subscription_rejects_unknown_rank() -> None:
    llm = _Llm()
    llm.args = SimpleNamespace(
        data_parallel_size=2,
        kv_cache_config=SimpleNamespace(event_buffer_max_size=8),
    )
    service = OpenEngineServicer(
        llm,
        "model",
        server_pb2.ENGINE_ROLE_AGGREGATED,
        RequestTracker(llm),
        kv_event_fanout=KvEventFanout(llm),
    )
    request = kv_pb2.SubscribeKvEventsRequest(data_parallel_ranks=[2])

    with pytest.raises(AssertionError, match="INVALID_ARGUMENT"):
        await _collect_async(service.SubscribeKvEvents(request, _Context()))


@pytest.mark.asyncio
async def test_drain_deadline_does_not_wait_forever_for_external_http() -> None:
    llm = _Llm()
    tracker = RequestTracker(llm)
    tracker.begin_external()
    service = OpenEngineServicer(
        llm,
        "model",
        server_pb2.ENGINE_ROLE_AGGREGATED,
        tracker,
        post_abort_cleanup_timeout_seconds=0.01,
    )
    request = lifecycle_pb2.DrainRequest(
        stop_accepting_new_requests=True,
        deadline_ms=1,
        abort_after_deadline=True,
    )

    responses = await asyncio.wait_for(
        _collect_async(service.Drain(request, _Context())), timeout=0.1
    )

    assert responses[-1].WhichOneof("event") == "error"
    await tracker.finish_external()


@pytest.mark.asyncio
async def test_server_info_omits_unknown_kv_geometry() -> None:
    llm = _Llm()
    service = OpenEngineServicer(
        llm, "model", server_pb2.ENGINE_ROLE_AGGREGATED, RequestTracker(llm)
    )
    info = await service.GetServerInfo(server_pb2.GetServerInfoRequest(), _Context())
    assert not info.capacity.HasField("kv_block_size")
    assert not info.capacity.HasField("total_kv_blocks")


@pytest.mark.asyncio
async def test_model_info_unknown_model_maps_to_not_found() -> None:
    llm = _Llm()
    service = OpenEngineServicer(
        llm, "model", server_pb2.ENGINE_ROLE_AGGREGATED, RequestTracker(llm)
    )

    with pytest.raises(AssertionError, match="NOT_FOUND"):
        await service.GetModelInfo(model_pb2.GetModelInfoRequest(model="other"), _Context())


@pytest.mark.asyncio
async def test_health_probe_and_unconfigured_lora_fail_explicitly(tmp_path) -> None:
    llm = _Llm()
    service = OpenEngineServicer(
        llm, "model", server_pb2.ENGINE_ROLE_AGGREGATED, RequestTracker(llm)
    )
    with pytest.raises(AssertionError, match="UNIMPLEMENTED"):
        await service.Health(lifecycle_pb2.HealthRequest(include_inference_probe=True), _Context())

    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    with pytest.raises(AssertionError, match="FAILED_PRECONDITION"):
        await service.LoadLora(
            lora_pb2.LoadLoraRequest(
                adapter=lora_pb2.LoraAdapter(
                    lora_id=1, lora_name="adapter", source_path=str(adapter_dir)
                )
            ),
            _Context(),
        )
