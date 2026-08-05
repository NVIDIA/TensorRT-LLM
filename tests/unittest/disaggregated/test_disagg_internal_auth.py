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

from types import SimpleNamespace
from unittest.mock import MagicMock
from uuid import UUID

import pytest
from fastapi import HTTPException

from tensorrt_llm.serve.disagg_auth import (
    INTERNAL_DISAGG_AUTH_HEADER,
    build_internal_disagg_auth_headers,
    build_internal_disagg_lifecycle_auth_headers,
    get_internal_disagg_auth_fields,
    request_requires_internal_disagg_auth,
    validate_internal_disagg_lifecycle_request,
    validate_internal_disagg_request,
)
from tensorrt_llm.serve.disagg_lifecycle_control import (
    CONTEXT_ARTIFACT_ABORT_PATH,
    GENERATION_GRANT_ABORT_PATH,
    ContextArtifactAbortRequest,
)
from tensorrt_llm.serve.openai_protocol import CompletionRequest, DisaggregatedParams
from tensorrt_llm.serve.openai_server import OpenAIServer


def _make_request(
    *, encoded_opaque_state: str | None = None, ctx_info_endpoint: str | None = None
) -> CompletionRequest:
    return CompletionRequest(
        model="test-model",
        prompt="hello",
        stream=False,
        disaggregated_params=DisaggregatedParams(
            request_type="generation_only",
            ctx_request_id=1,
            disagg_request_id=2,
            encoded_opaque_state=encoded_opaque_state,
            ctx_info_endpoint=ctx_info_endpoint,
        ),
    )


def _with_raw_ctx_info_endpoint(
    request: CompletionRequest, ctx_info_endpoint: object
) -> CompletionRequest:
    request.disaggregated_params = request.disaggregated_params.model_copy(
        update={"ctx_info_endpoint": ctx_info_endpoint}
    )
    return request


def test_unprotected_request_does_not_require_internal_auth():
    request = _make_request()

    assert not request_requires_internal_disagg_auth(request)
    assert build_internal_disagg_auth_headers(None, request) == {}
    validate_internal_disagg_request(None, request, {})


def test_contextual_fields_alone_do_not_require_internal_auth():
    request = _make_request()
    request.disaggregated_params = request.disaggregated_params.model_copy(
        update={"ctx_dp_rank": 0, "schedule_style": 0}
    )

    assert not request_requires_internal_disagg_auth(request)
    assert build_internal_disagg_auth_headers(None, request) == {}


def test_protected_fields_come_from_protocol_metadata():
    assert set(get_internal_disagg_auth_fields()) == {
        "artifact_version",
        "consumer_grant_id",
        "context_control_endpoint",
        "context_transceiver_lifecycle",
        "encoded_opaque_state",
        "ctx_info_endpoint",
        "generation_endpoint_incarnation",
        "generation_endpoint_name",
        "generation_endpoint_rank",
        "handoff_attempt_uuid",
        "logical_request_id",
        "prefill_artifact_id",
        "transfer_session_id",
    }


def test_lifecycle_identity_fields_require_internal_auth():
    request = CompletionRequest(
        model="test-model",
        prompt="hello",
        disaggregated_params=DisaggregatedParams(
            request_type="context_only",
            logical_request_id=17,
            prefill_artifact_id=str(UUID(int=2)),
            artifact_version=0,
            handoff_attempt_uuid=str(UUID(int=3)),
            consumer_grant_id=str(UUID(int=5)),
            transfer_session_id=str(UUID(int=6)),
            ctx_dp_rank=0,
        ),
    )

    assert request_requires_internal_disagg_auth(request)
    headers = build_internal_disagg_auth_headers("secret", request)
    tampered = request.model_copy(
        update={
            "disaggregated_params": request.disaggregated_params.model_copy(
                update={"artifact_version": 1}
            )
        }
    )
    with pytest.raises(ValueError, match="Invalid internal"):
        validate_internal_disagg_request("secret", tampered, headers)

    tampered_context = request.model_copy(
        update={
            "disaggregated_params": request.disaggregated_params.model_copy(
                update={"ctx_dp_rank": 1}
            )
        }
    )
    with pytest.raises(ValueError, match="Invalid internal"):
        validate_internal_disagg_request("secret", tampered_context, headers)


@pytest.mark.parametrize(
    "completion_request",
    [
        _make_request(encoded_opaque_state="b3BhcXVl"),
        _make_request(ctx_info_endpoint="tcp://10.0.0.1:5000"),
        _make_request(
            encoded_opaque_state="b3BhcXVl",
            ctx_info_endpoint="tcp://10.0.0.1:5000",
        ),
    ],
)
def test_protected_fields_allow_missing_internal_auth_key_with_warning(
    completion_request,
):
    assert request_requires_internal_disagg_auth(completion_request)

    warning_message = (
        "In a future release the requirement to use internal_request_auth_key will be enforced"
    )
    with pytest.warns(FutureWarning, match=warning_message):
        assert build_internal_disagg_auth_headers(None, completion_request) == {}
    with pytest.warns(FutureWarning, match=warning_message):
        validate_internal_disagg_request(None, completion_request, {})


@pytest.mark.parametrize(
    "completion_request",
    [
        _make_request(encoded_opaque_state="b3BhcXVl"),
        _make_request(ctx_info_endpoint="tcp://10.0.0.1:5000"),
        _make_request(
            encoded_opaque_state="b3BhcXVl",
            ctx_info_endpoint="tcp://10.0.0.1:5000",
        ),
    ],
)
def test_protected_fields_accept_valid_internal_auth_header(completion_request):
    headers = build_internal_disagg_auth_headers("secret", completion_request)

    assert headers[INTERNAL_DISAGG_AUTH_HEADER].startswith("sha256=")
    validate_internal_disagg_request("secret", completion_request, headers)


def test_protected_fields_accept_valid_header_after_wire_roundtrip():
    request = _make_request(
        encoded_opaque_state="b3BhcXVl",
        ctx_info_endpoint="tcp://10.0.0.1:5000",
    )
    request.disaggregated_params.conversation_id = "conversation-1"
    headers = build_internal_disagg_auth_headers("secret", request)

    wire_request = CompletionRequest.model_validate_json(
        request.model_dump_json(exclude_unset=True)
    )

    validate_internal_disagg_request("secret", wire_request, headers)


def test_ctx_info_endpoint_list_sender_matches_validated_string_receiver():
    request = _with_raw_ctx_info_endpoint(
        _make_request(encoded_opaque_state="b3BhcXVl"),
        ["tcp://10.0.0.1:5000"],
    )
    headers = build_internal_disagg_auth_headers("secret", request)

    wire_request = _make_request(
        encoded_opaque_state="b3BhcXVl",
        ctx_info_endpoint="tcp://10.0.0.1:5000",
    )

    validate_internal_disagg_request("secret", wire_request, headers)


def test_unprotected_disagg_fields_do_not_invalidate_internal_auth_header():
    request = _make_request(ctx_info_endpoint="tcp://10.0.0.1:5000")
    headers = build_internal_disagg_auth_headers("secret", request)
    request.disaggregated_params.conversation_id = "conversation-1"

    validate_internal_disagg_request("secret", request, headers)


def test_opaque_state_rejects_tampered_payload():
    request = _make_request(encoded_opaque_state="b3BhcXVl")
    headers = build_internal_disagg_auth_headers("secret", request)
    request.disaggregated_params.encoded_opaque_state = "dGFtcGVyZWQ="

    with pytest.raises(ValueError, match="Invalid internal"):
        validate_internal_disagg_request("secret", request, headers)


def test_ctx_info_endpoint_rejects_tampered_payload():
    request = _make_request(ctx_info_endpoint="tcp://10.0.0.1:5000")
    headers = build_internal_disagg_auth_headers("secret", request)
    request.disaggregated_params.ctx_info_endpoint = "tcp://10.0.0.2:5000"

    with pytest.raises(ValueError, match="Invalid internal"):
        validate_internal_disagg_request("secret", request, headers)


def test_protected_fields_reject_missing_auth_header():
    request = _make_request(ctx_info_endpoint="tcp://10.0.0.1:5000")

    with pytest.raises(ValueError, match="Invalid internal"):
        validate_internal_disagg_request("secret", request, {})


def test_worker_rejects_protected_fields_without_cache_transceiver_config():
    request = _make_request(ctx_info_endpoint="tcp://10.0.0.1:5000")
    server = object.__new__(OpenAIServer)
    server.generator = type(
        "Generator",
        (),
        {
            "args": type(
                "Args",
                (),
                {
                    "cache_transceiver_config": None,
                },
            )(),
        },
    )()
    server._internal_disagg_auth_key = "secret"

    with pytest.raises(ValueError, match="cache_transceiver_config"):
        server._validate_internal_disagg_request(request, raw_request=None)


def _lifecycle_body() -> dict:
    return {
        "lifecycle_protocol_version": 1,
        "logical_request_id": 17,
        "consumer_grant_id": str(UUID(int=5)),
    }


def test_lifecycle_control_accepts_valid_auth_header():
    body = _lifecycle_body()
    headers = build_internal_disagg_lifecycle_auth_headers(
        "secret",
        CONTEXT_ARTIFACT_ABORT_PATH,
        body,
    )

    validate_internal_disagg_lifecycle_request(
        "secret",
        CONTEXT_ARTIFACT_ABORT_PATH,
        body,
        headers,
    )


def test_lifecycle_control_without_key_warns_and_remains_compatible():
    warning_message = (
        "In a future release the requirement to use internal_request_auth_key will be enforced"
    )
    with pytest.warns(FutureWarning, match=warning_message):
        headers = build_internal_disagg_lifecycle_auth_headers(
            None,
            CONTEXT_ARTIFACT_ABORT_PATH,
            _lifecycle_body(),
        )
    with pytest.warns(FutureWarning, match=warning_message):
        validate_internal_disagg_lifecycle_request(
            None,
            CONTEXT_ARTIFACT_ABORT_PATH,
            _lifecycle_body(),
            headers,
        )


def test_lifecycle_control_rejects_missing_or_tampered_auth():
    body = _lifecycle_body()
    headers = build_internal_disagg_lifecycle_auth_headers(
        "secret",
        CONTEXT_ARTIFACT_ABORT_PATH,
        body,
    )

    with pytest.raises(ValueError, match="Invalid internal"):
        validate_internal_disagg_lifecycle_request(
            "secret",
            CONTEXT_ARTIFACT_ABORT_PATH,
            body,
            {},
        )
    with pytest.raises(ValueError, match="Invalid internal"):
        validate_internal_disagg_lifecycle_request(
            "secret",
            CONTEXT_ARTIFACT_ABORT_PATH,
            {**body, "logical_request_id": 18},
            headers,
        )


def test_lifecycle_control_signature_is_bound_to_route():
    body = _lifecycle_body()
    headers = build_internal_disagg_lifecycle_auth_headers(
        "secret",
        CONTEXT_ARTIFACT_ABORT_PATH,
        body,
    )

    with pytest.raises(ValueError, match="Invalid internal"):
        validate_internal_disagg_lifecycle_request(
            "secret",
            GENERATION_GRANT_ABORT_PATH,
            body,
            headers,
        )


@pytest.mark.asyncio
async def test_lifecycle_handler_authenticates_before_state_mutation():
    request = ContextArtifactAbortRequest(
        lifecycle_protocol_version=1,
        logical_request_id=17,
        prefill_artifact_id=UUID(int=2),
        artifact_version=0,
        handoff_attempt_uuid=UUID(int=3),
        consumer_grant_id=UUID(int=5),
        transfer_session_id=UUID(int=6),
        context_endpoint_incarnation=UUID(int=7),
    )
    server = object.__new__(OpenAIServer)
    server._internal_disagg_auth_key = "secret"
    server.disagg_lifecycle_control = MagicMock()

    with pytest.raises(HTTPException) as error:
        await server.abort_context_artifact(
            request,
            SimpleNamespace(headers={}),
        )

    assert error.value.status_code == 401
    server.disagg_lifecycle_control.abort_context_artifact.assert_not_called()
