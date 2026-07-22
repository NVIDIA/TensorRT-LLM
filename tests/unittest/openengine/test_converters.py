# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json

import pytest
from openengine.v1 import generation_pb2, input_pb2, kv_pb2

from tensorrt_llm.disaggregated_params import DisaggregatedParams, DisaggScheduleStyle
from tensorrt_llm.executor.result import Logprob
from tensorrt_llm.openengine.converters import (
    HANDOFF_ATTRIBUTE,
    decode_handoff,
    encode_handoff,
    handoff_requires_decode_media,
    load_media,
    to_priority,
    to_sampling_params,
)


def test_context_first_handoff_round_trip_large_ids_and_binary() -> None:
    params = DisaggregatedParams(
        request_type="context_only",
        first_gen_tokens=[11, 12],
        first_gen_log_probs=[
            {
                11: Logprob(logprob=-0.25, rank=1),
                91: Logprob(logprob=-2.0, rank=2),
            },
            {12: Logprob(logprob=-0.5, rank=1)},
        ],
        ctx_request_id=2**63 - 2,
        disagg_request_id=2**63 - 1,
        ctx_dp_rank=7,
        ctx_info_endpoint="tcp://context:1234",
        draft_tokens=[21, 22],
        ctx_usage={"prompt_tokens": 9},
        conversation_id="conversation",
        schedule_style=DisaggScheduleStyle.CONTEXT_FIRST,
        opaque_state=b"\x00\xffbinary",
    )

    session = encode_handoff(params)
    payload = json.loads(session.attributes_struct[HANDOFF_ATTRIBUTE])
    assert payload["ctx_request_id"] == str(2**63 - 2)
    assert payload["disagg_request_id"] == str(2**63 - 1)

    restored = decode_handoff(session)
    assert restored.request_type == "generation_only"
    assert restored.ctx_request_id == 2**63 - 2
    assert restored.disagg_request_id == 2**63 - 1
    assert restored.opaque_state == b"\x00\xffbinary"
    assert restored.first_gen_log_probs[0][11] == Logprob(-0.25, 1)
    assert restored.first_gen_log_probs[0][91] == Logprob(-2.0, 2)
    assert restored.first_gen_log_probs[1][12] == Logprob(-0.5, 1)


def test_raw_media_handoff_omits_transient_mrope_and_requires_decode_media() -> None:
    params = DisaggregatedParams(
        request_type="context_only",
        schedule_style=DisaggScheduleStyle.CONTEXT_FIRST,
        mrope_position_ids_handle={"tensor": "ids"},
        mrope_position_deltas_handle={"tensor": "deltas"},
    )

    with pytest.raises(ValueError, match="mrope"):
        encode_handoff(params)

    session = encode_handoff(params, requires_decode_media=True)
    assert handoff_requires_decode_media(session)
    restored = decode_handoff(session)
    assert restored.mrope_position_ids_handle is None
    assert restored.mrope_position_deltas_handle is None


@pytest.mark.parametrize(
    "params, message",
    [
        (
            DisaggregatedParams(
                request_type="context_only",
                schedule_style=DisaggScheduleStyle.GENERATION_FIRST,
            ),
            "Generation-first",
        ),
        (DisaggregatedParams(request_type="context_only", first_gen_logits=[object()]), "logits"),
        (
            DisaggregatedParams(
                request_type="context_only",
                multimodal_embedding_handles=[{"handle": "encoder"}],
            ),
            "embedding handles",
        ),
    ],
)
def test_handoff_rejects_out_of_scope_topologies(params: DisaggregatedParams, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        encode_handoff(params)


def test_decode_rejects_non_decimal_identifier() -> None:
    session = kv_pb2.KvSessionRef()
    session.attributes_struct[HANDOFF_ATTRIBUTE] = json.dumps(
        {
            "schedule_style": "context_first",
            "ctx_request_id": 9007199254740993,
        }
    )
    with pytest.raises(ValueError, match="decimal string"):
        decode_handoff(session)


def test_decode_rejects_multimodal_hashes_even_without_embedding_handles() -> None:
    session = kv_pb2.KvSessionRef()
    session.attributes_struct[HANDOFF_ATTRIBUTE] = json.dumps(
        {
            "schedule_style": "context_first",
            "multimodal_hashes": [[1, 2, 3, 4, 5, 6, 7, 8]],
        }
    )

    with pytest.raises(ValueError, match="handles"):
        decode_handoff(session)


def test_decode_rejects_malformed_context_usage() -> None:
    session = kv_pb2.KvSessionRef()
    session.attributes_struct[HANDOFF_ATTRIBUTE] = json.dumps(
        {
            "schedule_style": "context_first",
            "ctx_usage": {"prompt_tokens": "2"},
        }
    )

    with pytest.raises(ValueError, match="prompt_tokens"):
        decode_handoff(session)


def test_sampling_conversion_preserves_requested_controls() -> None:
    request = generation_pb2.GenerateRequest(
        sampling=generation_pb2.SamplingParams(
            temperature=0.25,
            top_p=0.8,
            top_k=16,
            seed=2**63,
            num_sequences=2,
        ),
        stopping=generation_pb2.StoppingOptions(
            max_tokens=8,
            min_tokens=2,
            conditions=[
                generation_pb2.StopCondition(stop_text="done"),
                generation_pb2.StopCondition(stop_token_id=42),
            ],
            ignore_eos=True,
            include_stop_in_output=True,
        ),
        response=generation_pb2.ResponseOptions(
            return_output_logprobs=True,
            output_candidates=generation_pb2.CandidateTokenSelection(top_n=4),
        ),
        guided=generation_pb2.GuidedDecoding(regex="[a-z]+"),
    )
    params = to_sampling_params(request)
    assert params.max_tokens == 8
    assert params.n == 2
    assert params.best_of == 2
    assert params.stop == ["done"]
    assert params.stop_token_ids == [42]
    assert params.logprobs == 4
    assert params.guided_decoding.regex == "[a-z]+"


def test_choice_guidance_escapes_literal_regex_characters() -> None:
    request = generation_pb2.GenerateRequest()
    request.guided.choice.choices.extend(["a.b", "c+(d)"])
    params = to_sampling_params(request)
    assert params.guided_decoding.regex == r"^(?:(?:a\.b)|(?:c\+\(d\)))$"


def test_priority_mapping_is_centered_bounded_and_strictly_monotonic() -> None:
    assert 0 < to_priority(-1000) < to_priority(0)
    assert to_priority(0) == to_priority(None) == 0.5
    assert to_priority(0) < to_priority(1000) < 1


@pytest.mark.asyncio
async def test_media_preserves_per_modality_order_and_merge_inputs(monkeypatch) -> None:
    calls = []

    class _MediaIO:
        @classmethod
        def create(cls, defaults, request):
            calls.append((defaults, request))
            return cls()

        async def _run_in_executor(self, function, data):
            return function(data)

        def load_bytes(self, data):
            return data.decode()

    monkeypatch.setattr(
        "tensorrt_llm.openengine.converters.MEDIA_IO_REGISTRY",
        {"image": _MediaIO, "video": _MediaIO, "audio": _MediaIO},
    )
    options = generation_pb2.GenerateRequest().media_options
    options.update({"image": {"format": "pil"}})
    config = type("Config", (), {"media_io_kwargs": {"image": {"device": "cpu"}}})()
    media = [
        input_pb2.MediaItem(modality=input_pb2.MODALITY_IMAGE, raw_bytes=b"one"),
        input_pb2.MediaItem(modality=input_pb2.MODALITY_VIDEO, raw_bytes=b"video"),
        input_pb2.MediaItem(modality=input_pb2.MODALITY_IMAGE, raw_bytes=b"two"),
    ]

    decoded = await load_media(media, options, config)

    assert decoded == {"image": ["one", "two"], "video": ["video"]}
    assert calls[0] == ({"device": "cpu"}, {"format": "pil"})
