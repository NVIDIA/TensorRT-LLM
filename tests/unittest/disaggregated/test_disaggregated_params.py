from unittest.mock import MagicMock, patch

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
        ctx_info_endpoint=["tcp://10.0.0.1:5000"],
    )
    openai_params = to_disaggregated_params(llm_params)

    assert openai_params.request_type == "context_only"
    assert openai_params.first_gen_tokens == [1, 2]
    assert openai_params.ctx_dp_rank == 5
    assert openai_params.ctx_info_endpoint == ["tcp://10.0.0.1:5000"]


def test_to_llm_disaggregated_params():
    from tensorrt_llm.serve.openai_protocol import DisaggregatedParams as OpenAIDisaggregatedParams
    from tensorrt_llm.serve.openai_protocol import to_llm_disaggregated_params

    openai_params = OpenAIDisaggregatedParams(
        request_type="generation_only",
        ctx_dp_rank=2,
        ctx_info_endpoint="tcp://10.0.0.1:5000",
    )
    llm_params = to_llm_disaggregated_params(openai_params)

    assert llm_params.request_type == "generation_only"
    assert llm_params.ctx_dp_rank == 2
    assert llm_params.ctx_info_endpoint == "tcp://10.0.0.1:5000"
