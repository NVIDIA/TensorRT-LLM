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

import pytest

from tensorrt_llm.serve.disagg_auth import (
    INTERNAL_DISAGG_AUTH_HEADER,
    build_internal_disagg_auth_headers,
    get_internal_disagg_auth_fields,
    request_requires_internal_disagg_auth,
    validate_internal_disagg_request,
)
from tensorrt_llm.serve.openai_protocol import (
    CompletionRequest,
    ConversationParams,
    DisaggregatedParams,
)
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


def test_protected_fields_come_from_protocol_metadata():
    assert set(get_internal_disagg_auth_fields()) == {
        "ctx_info_endpoint",
        "encoded_opaque_state",
    }


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
    request.conversation_params = ConversationParams(conversation_id="conversation-1")
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


def test_conversation_params_do_not_invalidate_internal_auth_header():
    request = _make_request(ctx_info_endpoint="tcp://10.0.0.1:5000")
    headers = build_internal_disagg_auth_headers("secret", request)
    request.conversation_params = ConversationParams(conversation_id="conversation-1")

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


def test_protected_fields_reject_non_ascii_auth_header():
    request = _make_request(ctx_info_endpoint="tcp://10.0.0.1:5000")
    headers = {INTERNAL_DISAGG_AUTH_HEADER: "\xff"}

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
