# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from types import SimpleNamespace

import pytest

from tensorrt_llm.bindings import executor as tllm
from tensorrt_llm.disaggregated_params import DisaggregatedParams
from tensorrt_llm.executor.request import GenerationRequest
from tensorrt_llm.executor.result import GenerationResult
from tensorrt_llm.inputs.multimodal import MultimodalInput, MultimodalParams
from tensorrt_llm.llmapi.llm import BaseLLM
from tensorrt_llm.sampling_params import SamplingParams

MM_HASHES = [[1, 2, 3, 4, 5, 6, 7, 8]]
MM_HASH_POSITIONS = [[1, 3]]
MM_HANDLES = [{"tensor_size": [2, 8], "ipc_handle": "handle"}]


def test_disaggregated_params_accepts_multimodal_hash_positions():
    params = DisaggregatedParams(
        multimodal_embedding_handles=MM_HANDLES,
        multimodal_hashes=MM_HASHES,
        multimodal_hash_positions=MM_HASH_POSITIONS,
    )

    assert params.multimodal_hash_positions == MM_HASH_POSITIONS


def test_disaggregated_params_rejects_hash_position_count_mismatch():
    with pytest.raises(AssertionError, match="multimodal_hash_positions and multimodal_hashes"):
        DisaggregatedParams(
            multimodal_hashes=MM_HASHES,
            multimodal_hash_positions=[MM_HASH_POSITIONS[0], [5]],
        )


def _response(
    token_id: int,
    *,
    finished: bool,
    context_phase_params=None,
    mm_embedding_handles=None,
):
    finish_reason = tllm.FinishReason.END_ID if finished else tllm.FinishReason.NOT_FINISHED
    response_result = SimpleNamespace(
        is_final=finished,
        context_phase_params=context_phase_params,
        decoding_iter=0,
        cached_tokens=0,
        avg_decoded_tokens_per_iter=None,
        finish_reasons=[finish_reason],
        output_token_ids=[[token_id]],
        sequence_index=0,
        cum_log_probs=None,
        log_probs=None,
        generation_logits=None,
        additional_context_outputs=None,
        additional_generation_outputs=None,
        request_perf_metrics=None,
        time_breakdown_metrics=None,
        context_logits=None,
        mm_embedding_handles=mm_embedding_handles,
    )
    return SimpleNamespace(has_error=lambda: False, result=response_result)


def test_generation_result_preserves_hash_positions_through_epd_handoff():
    multimodal_input = MultimodalInput.from_components(
        MM_HASHES, [1], [2], mm_hash_positions=MM_HASH_POSITIONS
    )
    request = GenerationRequest(
        prompt_token_ids=[11, 999, 77, 999, 12],
        sampling_params=SamplingParams(max_tokens=4),
        multimodal_params=MultimodalParams(multimodal_input=multimodal_input),
    )
    result = GenerationResult(request)

    result._handle_response(_response(10, finished=False, mm_embedding_handles=MM_HANDLES))
    assert result.disaggregated_params.multimodal_hashes == MM_HASHES
    assert result.disaggregated_params.multimodal_hash_positions == MM_HASH_POSITIONS

    context_phase_params = SimpleNamespace(
        first_gen_tokens=[10],
        req_id=123,
        opaque_state=b"state",
        draft_tokens=None,
        ctx_dp_rank=0,
        disagg_info_endpoint="tcp://127.0.0.1:9000",
    )
    result._handle_response(_response(11, finished=True, context_phase_params=context_phase_params))

    assert result.disaggregated_params.multimodal_embedding_handles is None
    assert result.disaggregated_params.multimodal_hashes == MM_HASHES
    assert result.disaggregated_params.multimodal_hash_positions == MM_HASH_POSITIONS


def test_llm_preprocess_reconstructs_multimodal_input_with_hash_positions():
    class FakeInputProcessor:
        support_mm_disagg = True

        def get_prompt_token_ids(self, inputs, mm_handles):
            assert mm_handles == MM_HANDLES
            return [11, 999, 77, 999, 12], [2], [1]

    llm = BaseLLM.__new__(BaseLLM)
    llm.input_processor = FakeInputProcessor()
    disaggregated_params = DisaggregatedParams(
        multimodal_embedding_handles=MM_HANDLES,
        multimodal_hashes=MM_HASHES,
        multimodal_hash_positions=MM_HASH_POSITIONS,
    )

    prompt_token_ids, prompt, query_token_ids, multimodal_params = BaseLLM._preprocess(
        llm, {"prompt": "describe image"}, SamplingParams(max_tokens=1), disaggregated_params
    )

    assert prompt_token_ids == [11, 999, 77, 999, 12]
    assert prompt == "describe image"
    assert query_token_ids is None
    assert multimodal_params.multimodal_data["multimodal_embedding"] == MM_HANDLES
    assert multimodal_params.multimodal_input.multimodal_hashes == MM_HASHES
    assert multimodal_params.multimodal_input.multimodal_hash_positions == MM_HASH_POSITIONS
