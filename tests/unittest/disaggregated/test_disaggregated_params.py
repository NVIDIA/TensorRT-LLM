# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from tensorrt_llm.bindings import executor as tllm
from tensorrt_llm.disaggregated_params import DisaggregatedParams
from tensorrt_llm.executor.result import GenerationResultBase, Logprob
from tensorrt_llm.inputs.multimodal import MultimodalInputLayout
from tensorrt_llm.sampling_params import SamplingParams

pytestmark = pytest.mark.cpu_only


def test_disaggregated_params_ctx_dp_rank():
    params = DisaggregatedParams()
    assert params.ctx_dp_rank is None

    params = DisaggregatedParams(ctx_dp_rank=3)
    assert params.ctx_dp_rank == 3


def test_disaggregated_params_ctx_info_endpoint():
    params = DisaggregatedParams()
    assert params.ctx_info_endpoint is None

    params = DisaggregatedParams(ctx_info_endpoint=["tcp://10.0.0.1:5000", "tcp://10.0.0.2:5000"])
    assert params.ctx_info_endpoint == ["tcp://10.0.0.1:5000", "tcp://10.0.0.2:5000"]


def test_receiver_ctx_info_endpoint_required():
    from tensorrt_llm._torch.disaggregation.native.transfer import Receiver

    with pytest.raises(ValueError, match="ctx_info_endpoint is required"):
        Receiver._extract_info_endpoint(DisaggregatedParams())
    with pytest.raises(ValueError, match="ctx_info_endpoint is required"):
        Receiver._extract_info_endpoint(DisaggregatedParams(ctx_info_endpoint=[]))
    assert (
        Receiver._extract_info_endpoint(
            DisaggregatedParams(ctx_info_endpoint="tcp://10.0.0.1:5000")
        )
        == "tcp://10.0.0.1:5000"
    )


@patch("tensorrt_llm.disaggregated_params.tllme")
def test_get_context_phase_params(mock_tllme):
    mock_ctx_params = MagicMock()
    mock_tllme.ContextPhaseParams.return_value = mock_ctx_params

    params = DisaggregatedParams(
        request_type="context_only",
        first_gen_tokens=[1, 2, 3],
        ctx_request_id=42,
        opaque_state=b"\x00\x01",
        draft_tokens=[10, 20],
        ctx_dp_rank=1,
        ctx_info_endpoint=["tcp://10.0.0.1:5000"],
    )
    result = params.get_context_phase_params()

    mock_tllme.ContextPhaseParams.assert_called_once_with(
        [1, 2, 3],  # first_gen_tokens
        42,  # request_id (ctx_request_id since disagg_request_id is None)
        b"\x00\x01",  # opaque_state
        [10, 20],  # draft_tokens
        1,  # ctx_dp_rank
        ["tcp://10.0.0.1:5000"],  # ctx_info_endpoint
    )
    assert result == mock_ctx_params


def test_to_disaggregated_params():
    from tensorrt_llm.serve.openai_protocol import to_disaggregated_params

    llm_params = DisaggregatedParams(
        request_type="context_only",
        first_gen_tokens=[1, 2],
        ctx_dp_rank=5,
        ctx_info_endpoint="tcp://10.0.0.1:5000",
        remote_tail_start=128,
        ctx_usage={
            "prompt_tokens": 10,
            "completion_tokens": 0,
            "total_tokens": 10,
            "prompt_tokens_details": {
                "cached_tokens": 4,
            },
        },
    )
    openai_params = to_disaggregated_params(llm_params)

    print(f"[usage_check] to_disaggregated_params: ctx_usage={openai_params.ctx_usage}")
    assert openai_params.request_type == "context_only"
    assert openai_params.first_gen_tokens == [1, 2]
    assert openai_params.ctx_dp_rank == 5
    assert openai_params.ctx_info_endpoint == "tcp://10.0.0.1:5000"
    assert openai_params.remote_tail_start == 128
    assert "multimodal_cache_salt" not in openai_params.model_dump()
    assert openai_params.ctx_usage.prompt_tokens == 10
    assert openai_params.ctx_usage.prompt_tokens_details.cached_tokens == 4


def test_to_llm_disaggregated_params():
    from tensorrt_llm.serve.openai_protocol import DisaggregatedParams as OpenAIDisaggregatedParams
    from tensorrt_llm.serve.openai_protocol import (
        PromptTokensDetails,
        UsageInfo,
        to_llm_disaggregated_params,
    )

    openai_params = OpenAIDisaggregatedParams(
        request_type="generation_only",
        ctx_dp_rank=2,
        ctx_info_endpoint="tcp://10.0.0.1:5000",
        remote_tail_start=256,
        ctx_usage=UsageInfo(
            prompt_tokens=10,
            completion_tokens=0,
            total_tokens=10,
            prompt_tokens_details=PromptTokensDetails(cached_tokens=4),
        ),
    )
    llm_params = to_llm_disaggregated_params(openai_params)

    print(f"[usage_check] to_llm_disaggregated_params: ctx_usage={llm_params.ctx_usage}")
    assert llm_params.request_type == "generation_only"
    assert llm_params.ctx_dp_rank == 2
    assert llm_params.ctx_info_endpoint == "tcp://10.0.0.1:5000"
    assert llm_params.remote_tail_start == 256
    assert llm_params.ctx_usage["prompt_tokens"] == 10
    assert llm_params.ctx_usage["prompt_tokens_details"]["cached_tokens"] == 4


def test_opaque_state_round_trips_through_openai_protocol():
    from tensorrt_llm.serve.openai_protocol import (
        to_disaggregated_params,
        to_llm_disaggregated_params,
    )

    openai_params = to_disaggregated_params(
        DisaggregatedParams(request_type="context_only", opaque_state=b"opaque")
    )
    assert openai_params.encoded_opaque_state == "b3BhcXVl"
    assert to_llm_disaggregated_params(openai_params).opaque_state == b"opaque"


def test_multimodal_spans_round_trip_through_openai_json():
    from tensorrt_llm.serve.openai_protocol import DisaggregatedParams as OpenAIDisaggregatedParams
    from tensorrt_llm.serve.openai_protocol import (
        to_disaggregated_params,
        to_llm_disaggregated_params,
    )

    original = DisaggregatedParams(
        request_type="context_only",
        first_gen_tokens=[17],
        opaque_state=b"opaque",
        remote_tail_start=128,
        multimodal_positions=[1, 5],
        multimodal_lengths=[2, 1],
    )
    encoded = to_disaggregated_params(original).model_dump_json()
    restored = to_llm_disaggregated_params(OpenAIDisaggregatedParams.model_validate_json(encoded))
    assert restored == original


@pytest.fixture
def handoff_llm():
    from tensorrt_llm.executor.request import GenerationRequest
    from tensorrt_llm.executor.result import GenerationResult
    from tensorrt_llm.llmapi.llm import BaseLLM

    def submit(*args, **kwargs):
        request = GenerationRequest(*args, **kwargs)
        return GenerationResult(request, disaggregated_params=request.disaggregated_params)

    llm = MagicMock()
    llm._encode_only = llm.args.mm_encoder_only = llm.args.return_perf_metrics = False
    llm._executor.is_shutdown.return_value = llm._is_encoder_decoder_model.return_value = False
    llm._prepare_sampling_params.return_value = SamplingParams(max_tokens=4)
    llm._prepare_prompt_inputs.side_effect = lambda *args, **kwargs: BaseLLM._prepare_prompt_inputs(
        llm, *args, **kwargs
    )
    llm._preprocess.side_effect = lambda *args: BaseLLM._preprocess(llm, *args)
    llm._executor.generate_async.side_effect = submit
    return llm


def _block_keys(result):
    from tensorrt_llm.runtime.kv_cache_manager_v2 import ReuseScope, sequence_to_blockchain_keys

    request = result._generation_request
    return [
        key
        for block, key in sequence_to_blockchain_keys(
            4, ReuseScope(salt=request.cache_salt), request.prompt_token_ids
        )
        if block
    ]


@pytest.mark.parametrize("caller_salt", ["tenant/가", "가" * 80], ids=["caller", "long_caller"])
def test_generation_reuses_context_block_keys_without_item_spans(handoff_llm, caller_salt):
    from tensorrt_llm.inputs.multimodal import (
        MULTIMODAL_CACHE_SALT_KEY,
        MULTIMODAL_NO_CONTENT_HASHES_KEY,
        MultimodalParams,
        isolate_mm_request,
    )
    from tensorrt_llm.llmapi.llm import BaseLLM, PreprocessedInputs
    from tensorrt_llm.serve.openai_protocol import DisaggregatedParams as WireParams
    from tensorrt_llm.serve.openai_protocol import (
        to_disaggregated_params,
        to_llm_disaggregated_params,
    )

    data = {}
    isolate_mm_request(data, "processor", ["audio"], "no item spans")
    tokens = list(range(8))
    prepared = PreprocessedInputs(
        prompt_token_ids=tokens, multimodal_params=MultimodalParams(multimodal_data=data)
    )
    context = BaseLLM.generate_async(
        handoff_llm,
        prepared,
        disaggregated_params=DisaggregatedParams(request_type="context_only"),
        cache_salt=caller_salt,
    )
    encoded = to_disaggregated_params(context.disaggregated_params).model_dump_json()
    handoff = to_llm_disaggregated_params(WireParams.model_validate_json(encoded))
    handoff.request_type = "generation_only"

    context_keys = _block_keys(context)
    assert len(context_keys) == 2
    for overrides in ({}, {"cache_salt": caller_salt}):
        generation = BaseLLM.generate_async(
            handoff_llm, tokens, disaggregated_params=handoff, **overrides
        )
        assert _block_keys(generation) == context_keys
        gen_data = generation._generation_request.multimodal_params.multimodal_data
        assert gen_data[MULTIMODAL_CACHE_SALT_KEY] == data[MULTIMODAL_CACHE_SALT_KEY]
        assert gen_data[MULTIMODAL_NO_CONTENT_HASHES_KEY]
    assert handoff.multimodal_cache_salt == context._generation_request.cache_salt
    other_caller = "b" * 225
    other = BaseLLM.generate_async(handoff_llm, prepared, cache_salt=other_caller)
    assert not set(_block_keys(other)).intersection(context_keys)
    with pytest.raises(ValueError, match="cache_salt differs"):
        BaseLLM.generate_async(
            handoff_llm, tokens, disaggregated_params=handoff, cache_salt=other_caller
        )


def test_generation_first_requires_shared_identity_without_item_spans(handoff_llm):
    from dataclasses import replace

    from tensorrt_llm.disaggregated_params import DisaggScheduleStyle
    from tensorrt_llm.inputs.multimodal import (
        MultimodalInput,
        MultimodalParams,
        isolate_mm_request,
        new_mm_cache_salt,
    )
    from tensorrt_llm.llmapi.llm import BaseLLM, PreprocessedInputs

    tokens = list(range(8))
    keys = []
    for request_id in (42, 43):
        data = {}
        isolate_mm_request(data, "processor", ["audio"], "no item spans")
        media = MultimodalParams(multimodal_data=data)
        prepared = PreprocessedInputs(prompt_token_ids=tokens, multimodal_params=media)
        context = DisaggregatedParams(
            request_type="context_only",
            schedule_style=DisaggScheduleStyle.GENERATION_FIRST,
            disagg_request_id=request_id,
        )
        with pytest.raises(ValueError, match="multimodal_cache_salt"):
            BaseLLM.generate_async(
                handoff_llm, prepared, disaggregated_params=context, cache_salt="tenant"
            )
        context.multimodal_cache_salt = new_mm_cache_salt("tenant")
        generation = replace(context, request_type="generation_only")
        gen = BaseLLM.generate_async(
            handoff_llm, tokens, disaggregated_params=generation, cache_salt="tenant"
        )
        ctx = BaseLLM.generate_async(
            handoff_llm, prepared, disaggregated_params=context, cache_salt="tenant"
        )
        gen_keys = _block_keys(gen)
        assert len(gen_keys) == 2
        assert gen_keys == _block_keys(ctx)
        local = BaseLLM.generate_async(
            handoff_llm, prepared, disaggregated_params=generation, cache_salt="tenant"
        )
        assert gen_keys == _block_keys(local)
        keys.append(gen_keys)
    assert keys[0] != keys[1]

    # Workers that already know the spans keep their per-item keys and caller salt.
    identity = MultimodalInput(
        multimodal_hashes=[[1] * 8], multimodal_positions=[1], multimodal_lengths=[3]
    )
    prepared = PreprocessedInputs(
        prompt_token_ids=tokens, multimodal_params=MultimodalParams(multimodal_input=identity)
    )
    for params in (context, generation):
        result = BaseLLM.generate_async(
            handoff_llm, prepared, disaggregated_params=params, cache_salt="tenant"
        )
        assert result._generation_request.cache_salt == "tenant"
        assert result._generation_request.multimodal_params.multimodal_input == identity


def test_embedding_handles_without_hashes_get_a_random_hash_each() -> None:
    """KV block reuse keys an item by its hash, so no two items may share one."""
    params = [
        DisaggregatedParams(multimodal_embedding_handles=[{"tensor_size": [1, 16]}] * 2)
        for _ in range(2)
    ]

    hashes = [tuple(mm_hash) for p in params for mm_hash in p.multimodal_hashes]
    assert len(set(hashes)) == 4
    for mm_hash in hashes:
        assert len(mm_hash) == 8
        assert all(isinstance(value, int) and -(2**31) <= value < 2**31 for value in mm_hash)


@patch("tensorrt_llm.disaggregated_params.tllme")
def test_get_context_phase_params_disagg_wins(mock_tllme):
    """disagg_request_id takes priority over ctx_request_id."""
    mock_tllme.ContextPhaseParams.return_value = MagicMock()

    params = DisaggregatedParams(
        request_type="context_only",
        first_gen_tokens=[1],
        ctx_request_id=200,
        disagg_request_id=100,
    )
    params.get_context_phase_params()

    # The second arg to ContextPhaseParams should be 100 (disagg), not 200 (ctx)
    call_args = mock_tllme.ContextPhaseParams.call_args
    assert call_args[0][1] == 100


@patch("tensorrt_llm.disaggregated_params.tllme")
def test_get_context_phase_params_falls_back_to_ctx(mock_tllme):
    """When disagg_request_id is None, ctx_request_id is used."""
    mock_tllme.ContextPhaseParams.return_value = MagicMock()

    params = DisaggregatedParams(
        request_type="context_only",
        first_gen_tokens=[1],
        ctx_request_id=200,
    )
    params.get_context_phase_params()

    call_args = mock_tllme.ContextPhaseParams.call_args
    assert call_args[0][1] == 200


@patch("tensorrt_llm.disaggregated_params.tllme")
def test_get_request_type_valid(mock_tllme):
    """get_request_type returns the correct enum for all 3 valid strings."""
    mock_tllme.RequestType.REQUEST_TYPE_CONTEXT_ONLY = "CTX"
    mock_tllme.RequestType.REQUEST_TYPE_GENERATION_ONLY = "GEN"
    mock_tllme.RequestType.REQUEST_TYPE_CONTEXT_AND_GENERATION = "CTX_GEN"

    assert DisaggregatedParams(request_type="context_only").get_request_type() == "CTX"
    assert DisaggregatedParams(request_type="generation_only").get_request_type() == "GEN"
    assert (
        DisaggregatedParams(request_type="context_and_generation").get_request_type() == "CTX_GEN"
    )


def test_get_request_type_invalid():
    """Invalid request_type raises ValueError at construction time."""
    with pytest.raises(ValueError, match="Unknown request type"):
        DisaggregatedParams(request_type="invalid_type")


def test_context_only_response_carries_the_first_token_logprobs_and_logits():
    """A context_only response carries the first token's logprob and logits onward.

    The context worker's response holds the first generated token's logprob and
    logits. The client folds them into the DisaggregatedParams that travel to the
    generation worker, which prepends them so its outputs cover every generated
    token.
    """
    result = GenerationResultBase(
        id=1,
        sampling_params=SamplingParams(max_tokens=4, logprobs=1, return_generation_logits=True),
    )
    result._disaggregated_params = DisaggregatedParams(
        multimodal_cache_salt="context-salt",
        multimodal_embedding_handles=[{"tensor_size": [1, 16]}],
        multimodal_input_layout=MultimodalInputLayout(
            prompt_token_ids=[1, 2, 3],
            multimodal_positions=[1],
            multimodal_lengths=[1],
            multimodal_embedding_lengths=[1],
        ),
        mrope_position_ids_handle={"tensor_size": [3, 1, 3]},
        mrope_position_deltas_handle={"tensor_size": [1, 1]},
    )
    first_token_logprob = {7: Logprob(logprob=-0.5, rank=1)}
    first_logits = torch.arange(6, dtype=torch.float32).reshape(1, 1, 6)
    context_result = SimpleNamespace(
        is_final=True,
        decoding_iter=1,
        avg_decoded_tokens_per_iter=1.0,
        context_phase_params=SimpleNamespace(
            first_gen_tokens=[7],
            req_id=1,
            opaque_state=b"",
            draft_tokens=None,
            ctx_dp_rank=0,
            disagg_info_endpoint=None,
        ),
        finish_reasons=[tllm.FinishReason.LENGTH],
        output_token_ids=[[7]],
        sequence_index=0,
        cum_log_probs=[-0.5],
        log_probs=[[first_token_logprob]],
        generation_logits=first_logits,
        per_pos_drafted=None,
        per_pos_accepted=None,
        spec_dec_totals=None,
        time_breakdown_metrics=None,
        context_logits=None,
        request_perf_metrics=None,
        additional_context_outputs=None,
        additional_generation_outputs=None,
    )

    result._handle_response(SimpleNamespace(result=context_result, has_error=lambda: False))

    params = result.disaggregated_params
    assert params.request_type == "context_only"
    assert params.first_gen_tokens == [7]
    assert params.first_gen_log_probs == [first_token_logprob]
    assert params.multimodal_cache_salt == "context-salt"
    assert params.multimodal_embedding_handles is None
    assert params.multimodal_input_layout is None
    assert params.mrope_position_ids_handle == {"tensor_size": [3, 1, 3]}
    assert params.mrope_position_deltas_handle == {"tensor_size": [1, 1]}
    (logits,) = params.first_gen_logits
    assert torch.equal(logits, first_logits[0, :1])
