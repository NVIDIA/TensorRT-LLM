from unittest.mock import MagicMock, patch

import pytest

from tensorrt_llm.disaggregated_params import DisaggregatedParams

MM_HANDLES = [
    {"tensor_size": [2, 4], "method_key": "cuda:0"},
    {"tensor_size": [3, 4], "method_key": "cuda:1"},
]
MM_HASHES = [[1, 2, 3, 4, 5, 6, 7, 8], [8, 7, 6, 5, 4, 3, 2, 1]]
MM_HASH_POSITIONS = [[1, 3], [6, 8, 10]]
MROPE_POSITION_IDS_HANDLE = {"tensor_size": [1, 2]}
MROPE_POSITION_DELTAS_HANDLE = {"tensor_size": [1]}


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
        multimodal_embedding_handles=MM_HANDLES,
        multimodal_hashes=MM_HASHES,
        multimodal_hash_positions=MM_HASH_POSITIONS,
        mrope_position_ids_handle=MROPE_POSITION_IDS_HANDLE,
        mrope_position_deltas_handle=MROPE_POSITION_DELTAS_HANDLE,
    )
    openai_params = to_disaggregated_params(llm_params)

    assert openai_params.request_type == "context_only"
    assert openai_params.first_gen_tokens == [1, 2]
    assert openai_params.ctx_dp_rank == 5
    assert openai_params.ctx_info_endpoint == "tcp://10.0.0.1:5000"
    assert openai_params.multimodal_embedding_handles == MM_HANDLES
    assert openai_params.multimodal_hashes == MM_HASHES
    assert openai_params.multimodal_hash_positions == MM_HASH_POSITIONS
    assert openai_params.mrope_position_ids_handle == MROPE_POSITION_IDS_HANDLE
    assert openai_params.mrope_position_deltas_handle == MROPE_POSITION_DELTAS_HANDLE


def test_to_llm_disaggregated_params():
    from tensorrt_llm.serve.openai_protocol import DisaggregatedParams as OpenAIDisaggregatedParams
    from tensorrt_llm.serve.openai_protocol import to_llm_disaggregated_params

    openai_params = OpenAIDisaggregatedParams(
        request_type="generation_only",
        ctx_dp_rank=2,
        ctx_info_endpoint="tcp://10.0.0.1:5000",
        multimodal_embedding_handles=MM_HANDLES,
        multimodal_hashes=MM_HASHES,
        multimodal_hash_positions=MM_HASH_POSITIONS,
        mrope_position_ids_handle=MROPE_POSITION_IDS_HANDLE,
        mrope_position_deltas_handle=MROPE_POSITION_DELTAS_HANDLE,
    )
    llm_params = to_llm_disaggregated_params(openai_params)

    assert llm_params.request_type == "generation_only"
    assert llm_params.ctx_dp_rank == 2
    assert llm_params.ctx_info_endpoint == "tcp://10.0.0.1:5000"
    assert llm_params.multimodal_embedding_handles == MM_HANDLES
    assert llm_params.multimodal_hashes == MM_HASHES
    assert llm_params.multimodal_hash_positions == MM_HASH_POSITIONS
    assert llm_params.mrope_position_ids_handle == MROPE_POSITION_IDS_HANDLE
    assert llm_params.mrope_position_deltas_handle == MROPE_POSITION_DELTAS_HANDLE


def test_disaggregated_params_conversion_roundtrip_preserves_exact_positions():
    from tensorrt_llm.serve.openai_protocol import (
        to_disaggregated_params,
        to_llm_disaggregated_params,
    )

    llm_params = DisaggregatedParams(
        request_type="context_only",
        first_gen_tokens=[1, 2],
        ctx_request_id=17,
        multimodal_embedding_handles=MM_HANDLES,
        multimodal_hashes=MM_HASHES,
        multimodal_hash_positions=MM_HASH_POSITIONS,
        mrope_position_ids_handle=MROPE_POSITION_IDS_HANDLE,
        mrope_position_deltas_handle=MROPE_POSITION_DELTAS_HANDLE,
    )

    roundtripped = to_llm_disaggregated_params(to_disaggregated_params(llm_params))

    assert roundtripped.multimodal_embedding_handles == MM_HANDLES
    assert roundtripped.multimodal_hashes == MM_HASHES
    assert roundtripped.multimodal_hash_positions == MM_HASH_POSITIONS
    assert roundtripped.mrope_position_ids_handle == MROPE_POSITION_IDS_HANDLE
    assert roundtripped.mrope_position_deltas_handle == MROPE_POSITION_DELTAS_HANDLE


def test_conversion_roundtrip_keeps_absent_positions_none():
    from tensorrt_llm.serve.openai_protocol import (
        to_disaggregated_params,
        to_llm_disaggregated_params,
    )

    llm_params = DisaggregatedParams(
        request_type="context_only",
        multimodal_embedding_handles=MM_HANDLES,
        multimodal_hashes=MM_HASHES,
    )

    roundtripped = to_llm_disaggregated_params(to_disaggregated_params(llm_params))

    assert roundtripped.multimodal_embedding_handles == MM_HANDLES
    assert roundtripped.multimodal_hashes == MM_HASHES
    assert roundtripped.multimodal_hash_positions is None


def test_mm_encoder_response_json_roundtrip_preserves_multimodal_disagg_fields():
    from tensorrt_llm.serve.openai_protocol import (
        ChatCompletionResponse,
        ChatCompletionResponseChoice,
        ChatMessage,
        UsageInfo,
        to_llm_disaggregated_params,
    )
    from tensorrt_llm.serve.openai_protocol import DisaggregatedParams as OpenAIDisaggregatedParams

    response = ChatCompletionResponse(
        id="mm-encoder-response",
        model="test-model",
        choices=[
            ChatCompletionResponseChoice(
                index=0,
                message=ChatMessage(role="assistant", content="dummy"),
                finish_reason="length",
                mm_embedding_handle=MM_HANDLES,
                disaggregated_params=OpenAIDisaggregatedParams(
                    request_type="context_only",
                    multimodal_embedding_handles=MM_HANDLES,
                    multimodal_hashes=MM_HASHES,
                    multimodal_hash_positions=MM_HASH_POSITIONS,
                    mrope_position_ids_handle=MROPE_POSITION_IDS_HANDLE,
                    mrope_position_deltas_handle=MROPE_POSITION_DELTAS_HANDLE,
                ),
            )
        ],
        usage=UsageInfo(prompt_tokens=5, completion_tokens=0, total_tokens=5),
    )

    roundtripped = ChatCompletionResponse(**response.model_dump(mode="json"))
    choice = roundtripped.choices[0]
    llm_params = to_llm_disaggregated_params(choice.disaggregated_params)

    assert choice.mm_embedding_handle == MM_HANDLES
    assert llm_params.multimodal_embedding_handles == MM_HANDLES
    assert llm_params.multimodal_hashes == MM_HASHES
    assert llm_params.multimodal_hash_positions == MM_HASH_POSITIONS
    assert llm_params.mrope_position_ids_handle == MROPE_POSITION_IDS_HANDLE
    assert llm_params.mrope_position_deltas_handle == MROPE_POSITION_DELTAS_HANDLE


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
