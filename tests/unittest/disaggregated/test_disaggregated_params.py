# Copyright (c) 2026, NVIDIA CORPORATION.
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

from dataclasses import FrozenInstanceError
from unittest.mock import MagicMock, patch
from uuid import UUID, uuid4

import pytest

from tensorrt_llm.disaggregated_params import DisaggregatedParams, TransceiverLifecycleAdvertisement


def _lifecycle_identity() -> dict:
    return {
        "logical_request_id": 123,
        "prefill_artifact_id": str(uuid4()),
        "artifact_version": 0,
        "handoff_attempt_uuid": str(uuid4()),
        "consumer_grant_id": str(uuid4()),
        "transfer_session_id": str(uuid4()),
    }


def _endpoint_identity() -> dict:
    return {
        "generation_endpoint_name": "generation-worker-0",
        "generation_endpoint_rank": 0,
        "generation_endpoint_incarnation": str(uuid4()),
        "context_control_endpoint": "tcp://context-worker-0:5001",
    }


def _context_transceiver_lifecycle() -> dict:
    return {
        "protocol_version": 1,
        "capabilities": [
            "ALLOCATION_GENERATION_LEASES",
            "ATTEMPT_IDENTITY",
            "CANCEL_BEFORE_CREATE_TOMBSTONES",
            "DIRECT_TRANSFER",
            "ENDPOINT_INCARNATION",
            "EXACT_WRITER_TRACKING",
            "PER_OPERATION_QUIESCENCE",
            "PUBLICATION_GATE",
            "SUBMISSION_FENCE",
            "TERMINAL_RESULT_REPLAY",
        ],
        "qualified_legacy_mode": False,
        "backend": "python",
        "instance_id": str(uuid4()),
        "world_size": 8,
        "tp_size": 4,
        "pp_size": 2,
        "cp_size": 1,
        "attention_dp": False,
    }


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


def test_disaggregated_lifecycle_identity_round_trip():
    from tensorrt_llm.serve.openai_protocol import (
        to_disaggregated_params,
        to_llm_disaggregated_params,
    )

    metadata = {
        **_lifecycle_identity(),
        **_endpoint_identity(),
        "context_transceiver_lifecycle": _context_transceiver_lifecycle(),
    }
    llm_params = DisaggregatedParams(request_type="context_only", **metadata)
    assert isinstance(
        llm_params.context_transceiver_lifecycle,
        TransceiverLifecycleAdvertisement,
    )

    openai_params = to_disaggregated_params(llm_params)
    assert {
        name: (
            getattr(openai_params, name).to_dict()
            if name == "context_transceiver_lifecycle"
            else getattr(openai_params, name)
        )
        for name in metadata
    } == metadata

    round_trip = to_llm_disaggregated_params(openai_params)
    assert {
        name: (
            getattr(round_trip, name).to_dict()
            if name == "context_transceiver_lifecycle"
            else getattr(round_trip, name)
        )
        for name in metadata
    } == metadata
    with pytest.raises(FrozenInstanceError):
        round_trip.context_transceiver_lifecycle.protocol_version = 0


def test_disaggregated_lifecycle_metadata_v0_defaults():
    from tensorrt_llm.serve.openai_protocol import DisaggregatedParams as OpenAIDisaggregatedParams

    field_names = (
        "logical_request_id",
        "prefill_artifact_id",
        "artifact_version",
        "handoff_attempt_uuid",
        "consumer_grant_id",
        "transfer_session_id",
        "generation_endpoint_name",
        "generation_endpoint_rank",
        "generation_endpoint_incarnation",
        "context_control_endpoint",
        "context_transceiver_lifecycle",
    )
    llm_params = DisaggregatedParams()
    openai_params = OpenAIDisaggregatedParams(request_type="context_only")

    assert all(getattr(llm_params, name) is None for name in field_names)
    assert all(getattr(openai_params, name) is None for name in field_names)


@pytest.mark.parametrize(
    "update,match",
    [
        ({"protocol_version": -1}, "protocol_version"),
        ({"capabilities": ["B", "A"]}, "sorted"),
        ({"qualified_legacy_mode": 1}, "qualified_legacy_mode"),
        ({"backend": "unknown"}, "backend"),
        ({"instance_id": "not-a-uuid"}, "instance_id"),
        ({"world_size": 0}, "world_size"),
        ({"attention_dp": 0}, "attention_dp"),
    ],
)
def test_context_transceiver_lifecycle_validation(update, match):
    from tensorrt_llm.serve.openai_protocol import DisaggregatedParams as OpenAIDisaggregatedParams

    lifecycle = _context_transceiver_lifecycle()
    lifecycle.update(update)

    with pytest.raises(ValueError, match=match):
        DisaggregatedParams(context_transceiver_lifecycle=lifecycle)
    with pytest.raises(ValueError, match=match):
        OpenAIDisaggregatedParams(
            request_type="generation_only",
            context_transceiver_lifecycle=lifecycle,
        )


def test_disaggregated_endpoint_identity_without_request_identity():
    from tensorrt_llm.serve.openai_protocol import DisaggregatedParams as OpenAIDisaggregatedParams

    identity = _endpoint_identity()

    llm_params = DisaggregatedParams(**identity)
    openai_params = OpenAIDisaggregatedParams(request_type="generation_only", **identity)

    assert {name: getattr(llm_params, name) for name in identity} == identity
    assert {name: getattr(openai_params, name) for name in identity} == identity


@pytest.mark.parametrize(
    "params,match",
    [
        ({"logical_request_id": 123}, "must be provided together"),
        (
            {
                **_lifecycle_identity(),
                "logical_request_id": True,
            },
            "logical_request_id must be a non-negative integer",
        ),
        (
            {
                **_lifecycle_identity(),
                "artifact_version": -1,
            },
            "artifact_version must be a non-negative integer",
        ),
        (
            {
                **_lifecycle_identity(),
                "handoff_attempt_uuid": "not-a-uuid",
            },
            "handoff_attempt_uuid must be a canonical non-nil UUID string",
        ),
        (
            {
                **_lifecycle_identity(),
                "consumer_grant_id": str(UUID(int=0)),
            },
            "consumer_grant_id must be a canonical non-nil UUID string",
        ),
    ],
)
def test_disaggregated_lifecycle_identity_validation(params, match):
    from pydantic import ValidationError

    from tensorrt_llm.serve.openai_protocol import DisaggregatedParams as OpenAIDisaggregatedParams

    with pytest.raises(ValueError, match=match):
        DisaggregatedParams(**params)
    with pytest.raises(ValidationError, match=match):
        OpenAIDisaggregatedParams(request_type="context_only", **params)


@pytest.mark.parametrize(
    "params,match",
    [
        (
            {"generation_endpoint_name": "generation-worker-0"},
            "Generation endpoint identity fields must be provided together",
        ),
        (
            {
                **_endpoint_identity(),
                "generation_endpoint_name": "",
            },
            "generation_endpoint_name must be a non-empty string",
        ),
        (
            {
                **_endpoint_identity(),
                "generation_endpoint_name": " ",
            },
            "generation_endpoint_name must be a non-empty string",
        ),
        (
            {
                **_endpoint_identity(),
                "generation_endpoint_name": 0,
            },
            "generation_endpoint_name must be a non-empty string",
        ),
        (
            {
                **_endpoint_identity(),
                "generation_endpoint_rank": True,
            },
            "generation_endpoint_rank must be a non-negative integer",
        ),
        (
            {
                **_endpoint_identity(),
                "generation_endpoint_rank": -1,
            },
            "generation_endpoint_rank must be a non-negative integer",
        ),
        (
            {
                **_endpoint_identity(),
                "generation_endpoint_rank": 1.0,
            },
            "generation_endpoint_rank must be a non-negative integer",
        ),
        (
            {
                **_endpoint_identity(),
                "generation_endpoint_incarnation": "not-a-uuid",
            },
            "generation_endpoint_incarnation must be a canonical non-nil UUID string",
        ),
        (
            {
                **_endpoint_identity(),
                "generation_endpoint_incarnation": str(UUID(int=0)),
            },
            "generation_endpoint_incarnation must be a canonical non-nil UUID string",
        ),
        (
            {
                **_endpoint_identity(),
                "generation_endpoint_incarnation": "123E4567-E89B-12D3-A456-426614174000",
            },
            "generation_endpoint_incarnation must be a canonical non-nil UUID string",
        ),
        (
            {"context_control_endpoint": ""},
            "context_control_endpoint must be a non-empty string",
        ),
        (
            {"context_control_endpoint": " "},
            "context_control_endpoint must be a non-empty string",
        ),
        (
            {"context_control_endpoint": 5001},
            "context_control_endpoint must be a non-empty string",
        ),
    ],
)
def test_disaggregated_endpoint_identity_validation(params, match):
    from pydantic import ValidationError

    from tensorrt_llm.serve.openai_protocol import DisaggregatedParams as OpenAIDisaggregatedParams

    with pytest.raises(ValueError, match=match):
        DisaggregatedParams(**params)
    with pytest.raises(ValidationError, match=match):
        OpenAIDisaggregatedParams(request_type="generation_only", **params)


def test_disaggregated_lifecycle_uuid_fields_must_be_distinct():
    identity = _lifecycle_identity()
    identity["transfer_session_id"] = identity["consumer_grant_id"]

    with pytest.raises(ValueError, match="UUID fields must be distinct"):
        DisaggregatedParams(**identity)


def test_disaggregated_lifecycle_and_endpoint_identity_is_immutable():
    from pydantic import ValidationError

    from tensorrt_llm.serve.openai_protocol import DisaggregatedParams as OpenAIDisaggregatedParams

    metadata = {**_lifecycle_identity(), **_endpoint_identity()}
    llm_params = DisaggregatedParams(**metadata)
    with pytest.raises(AttributeError, match="immutable lifecycle identity"):
        llm_params.transfer_session_id = str(uuid4())

    for field_name, value in (
        ("generation_endpoint_name", "generation-worker-1"),
        ("generation_endpoint_rank", 1),
        ("generation_endpoint_incarnation", str(uuid4())),
        ("context_control_endpoint", "tcp://context-worker-1:5001"),
    ):
        with pytest.raises(AttributeError, match="immutable lifecycle identity"):
            setattr(llm_params, field_name, value)

    openai_params = OpenAIDisaggregatedParams(request_type="context_only", **metadata)
    with pytest.raises(ValidationError, match="Field is frozen"):
        openai_params.transfer_session_id = str(uuid4())
    for field_name, value in (
        ("generation_endpoint_name", "generation-worker-1"),
        ("generation_endpoint_rank", 1),
        ("generation_endpoint_incarnation", str(uuid4())),
        ("context_control_endpoint", "tcp://context-worker-1:5001"),
    ):
        with pytest.raises(ValidationError, match="Field is frozen"):
            setattr(openai_params, field_name, value)


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
