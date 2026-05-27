import base64
from unittest.mock import MagicMock, patch

import pytest

from tensorrt_llm.disaggregated_params import DisaggregatedParams


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
        ctx_usage={
            "prompt_tokens": 10,
            "completion_tokens": 0,
            "total_tokens": 10,
            "prompt_tokens_details": {
                "cached_tokens": 4,
            },
        },
        conversation_id="conv-abc",
    )
    openai_params = to_disaggregated_params(llm_params)

    print(f"[usage_check] to_disaggregated_params: ctx_usage={openai_params.ctx_usage}")
    assert openai_params.request_type == "context_only"
    assert openai_params.first_gen_tokens == [1, 2]
    assert openai_params.ctx_dp_rank == 5
    assert openai_params.ctx_info_endpoint == "tcp://10.0.0.1:5000"
    assert openai_params.ctx_usage.prompt_tokens == 10
    assert openai_params.ctx_usage.prompt_tokens_details.cached_tokens == 4
    assert openai_params.conversation_id == "conv-abc"


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
        ctx_usage=UsageInfo(
            prompt_tokens=10,
            completion_tokens=0,
            total_tokens=10,
            prompt_tokens_details=PromptTokensDetails(cached_tokens=4),
        ),
        conversation_id="conv-xyz",
    )
    llm_params = to_llm_disaggregated_params(openai_params)

    print(f"[usage_check] to_llm_disaggregated_params: ctx_usage={llm_params.ctx_usage}")
    assert llm_params.request_type == "generation_only"
    assert llm_params.ctx_dp_rank == 2
    assert llm_params.ctx_info_endpoint == "tcp://10.0.0.1:5000"
    assert llm_params.ctx_usage["prompt_tokens"] == 10
    assert llm_params.ctx_usage["prompt_tokens_details"]["cached_tokens"] == 4
    assert llm_params.conversation_id == "conv-xyz"


def test_disaggregated_params_conversation_id():
    """conversation_id defaults to None and survives the serve<->llm round-trip."""
    from tensorrt_llm.serve.openai_protocol import DisaggregatedParams as OpenAIDisaggregatedParams
    from tensorrt_llm.serve.openai_protocol import (
        to_disaggregated_params,
        to_llm_disaggregated_params,
    )

    assert DisaggregatedParams().conversation_id is None

    # serve -> llm -> serve preserves the conversation id end to end.
    openai_params = OpenAIDisaggregatedParams(
        request_type="context_only", conversation_id="conv-roundtrip"
    )
    llm_params = to_llm_disaggregated_params(openai_params)
    assert llm_params.conversation_id == "conv-roundtrip"
    assert to_disaggregated_params(llm_params).conversation_id == "conv-roundtrip"


def test_opaque_state_requires_valid_tag(monkeypatch):
    from tensorrt_llm.serve.openai_protocol import DisaggregatedParams as OpenAIDisaggregatedParams
    from tensorrt_llm.serve.openai_protocol import to_disaggregated_params, to_llm_disaggregated_params

    encoded = base64.b64encode(b"opaque").decode("utf-8")
    openai_params = OpenAIDisaggregatedParams(
        request_type="generation_only", encoded_opaque_state=encoded)
    with pytest.raises(ValueError, match="valid tag"):
        to_llm_disaggregated_params(openai_params)

    monkeypatch.setenv("TLLM_DISAGG_OPAQUE_STATE_HMAC_KEY", "test-key")
    signed = to_disaggregated_params(
        DisaggregatedParams(request_type="context_only", opaque_state=b"opaque"))
    assert to_llm_disaggregated_params(signed).opaque_state == b"opaque"


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
