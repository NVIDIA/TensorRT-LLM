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

import pytest

from tensorrt_llm.llmapi.disagg_utils import ServerRole
from tensorrt_llm.serve.conversation_id import SUBAGENT_AFFINITY_HEADER
from tensorrt_llm.serve.disagg_auth import (
    INTERNAL_DISAGG_AUTH_HEADER,
    SUBAGENT_AFFINITY_AUTH_HEADER,
    build_internal_disagg_auth_headers,
    build_subagent_affinity_headers,
    get_internal_disagg_auth_fields,
    request_requires_internal_disagg_auth,
    validate_internal_disagg_request,
    validate_subagent_affinity,
)
from tensorrt_llm.serve.openai_protocol import (
    ChatCompletionRequest,
    CompletionRequest,
    ConversationParams,
    DisaggregatedParams,
)
from tensorrt_llm.serve.openai_server import OpenAIServer

pytestmark = pytest.mark.cpu_only


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


def _make_affinity_request(role: ServerRole) -> ChatCompletionRequest:
    return ChatCompletionRequest(
        model="test-model",
        messages=[{"role": "user", "content": "hello"}],
        conversation_params=ConversationParams(
            conversation_id="child", subagent_affinity_id="parent"
        ),
        disaggregated_params=DisaggregatedParams(
            request_type="context_only" if role == ServerRole.CONTEXT else "generation_only",
            disagg_request_id=42,
        ),
    )


@pytest.mark.parametrize("role", [ServerRole.CONTEXT, ServerRole.GENERATION])
def test_worker_affinity_survives_wire_roundtrip(role: ServerRole) -> None:
    request = _make_affinity_request(role)
    headers = build_subagent_affinity_headers("secret", request, role)
    wire_request = ChatCompletionRequest.model_validate_json(
        request.model_dump_json(exclude_unset=True)
    )
    assert wire_request.conversation_params.subagent_affinity_id is None
    server = object.__new__(OpenAIServer)
    server.server_role = role
    server._internal_disagg_auth_key = "secret"

    scheduling = server._get_scheduling_params(wire_request, SimpleNamespace(headers=headers))

    assert scheduling.subagent_affinity_id == "parent"
    assert wire_request.conversation_params.conversation_id == "child"


@pytest.mark.parametrize("key", [None, "secret"])
def test_aggregated_worker_ignores_affinity_header(key: str | None) -> None:
    request = _make_affinity_request(ServerRole.CONTEXT)
    headers = build_subagent_affinity_headers("secret", request, ServerRole.CONTEXT)
    request.disaggregated_params = None
    server = object.__new__(OpenAIServer)
    server.server_role = None
    server._internal_disagg_auth_key = key

    scheduling = server._get_scheduling_params(request, SimpleNamespace(headers=headers))

    assert scheduling.subagent_affinity_id is None
    assert request.conversation_params.conversation_id == "child"


@pytest.mark.parametrize("role", [ServerRole.CONTEXT, ServerRole.GENERATION])
@pytest.mark.parametrize("key", [None, "secret"])
def test_worker_rejects_unsigned_affinity(role: ServerRole, key: str | None) -> None:
    request = _make_affinity_request(role)
    with pytest.raises(ValueError, match="auth"):
        validate_subagent_affinity(key, request, role, {SUBAGENT_AFFINITY_HEADER: "parent"})


@pytest.mark.parametrize(
    "tamper", ["parent", "child", "role", "model", "request_type", "request_id", "signature", "key"]
)
def test_affinity_signature_rejects_tampering(tamper: str) -> None:
    role = ServerRole.CONTEXT
    request = _make_affinity_request(role)
    headers = build_subagent_affinity_headers("secret", request, role)
    key = "secret"
    if tamper == "parent":
        headers[SUBAGENT_AFFINITY_HEADER] = "other-parent"
    elif tamper == "child":
        request.conversation_params.conversation_id = "other-child"
    elif tamper == "role":
        role = ServerRole.GENERATION
    elif tamper == "model":
        request.model = "other-model"
    elif tamper == "request_type":
        request.disaggregated_params.request_type = "generation_only"
    elif tamper == "request_id":
        request.disaggregated_params.disagg_request_id = 43
    elif tamper == "signature":
        headers[SUBAGENT_AFFINITY_AUTH_HEADER] = "\xff"
    elif tamper == "key":
        key = "wrong-secret"

    with pytest.raises(ValueError, match="Invalid internal subagent"):
        validate_subagent_affinity(key, request, role, headers)


def test_affinity_auth_requires_key_only_when_forwarding_affinity() -> None:
    request = _make_affinity_request(ServerRole.CONTEXT)
    with pytest.raises(ValueError, match="internal_request_auth_key"):
        build_subagent_affinity_headers(None, request, ServerRole.CONTEXT)
    request.conversation_params.subagent_affinity_id = None
    assert build_subagent_affinity_headers(None, request, ServerRole.CONTEXT) == {}
    assert validate_subagent_affinity(None, request, ServerRole.CONTEXT, {}) is None


def test_affinity_signature_does_not_change_legacy_transfer_signature() -> None:
    request = _make_request(encoded_opaque_state="b3BhcXVl")
    legacy_headers = build_internal_disagg_auth_headers("secret", request)
    request.conversation_params = ConversationParams(
        conversation_id="child", subagent_affinity_id="parent"
    )
    headers = legacy_headers | build_subagent_affinity_headers(
        "secret", request, ServerRole.GENERATION
    )
    wire_request = CompletionRequest.model_validate_json(
        request.model_dump_json(exclude_unset=True)
    )
    validate_internal_disagg_request("secret", wire_request, headers)
    assert build_internal_disagg_auth_headers("secret", request) == legacy_headers
    assert (
        validate_subagent_affinity("secret", wire_request, ServerRole.GENERATION, headers)
        == "parent"
    )
