# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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
"""What a client gets back when its request body fails validation.

A 400 must carry an error object a client can read -- OpenAI's
``{"error": {"message", "type", "param", "code"}}`` on OpenAI routes and the
Anthropic envelope on /v1/messages -- and must not quote the request back:
pydantic's own rendering repeats every rejected value (an inline image is
megabytes of base64) and FastAPI's names the endpoint's source file and line.

Each server is built by the shared factory with fake generators, so these
requests meet the real route table and exception handler. The disaggregated
client does not enter its remote-service lifespan; backend and metric spies
observe validation without replacing server construction.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi.testclient import TestClient
from pydantic import TypeAdapter, ValidationError
from test_common.serve.app_factory import AggConfig, DisaggConfig, ServeTestEnv

from tensorrt_llm.executor.request import LoRARequest
from tensorrt_llm.llmapi.disagg_utils import ServerRole
from tensorrt_llm.serve import openai_disagg_server
from tensorrt_llm.serve.anthropic_protocol import AnthropicThinkingConfig
from tensorrt_llm.serve.openai_protocol import (
    TOOL_PARAM_MAX_ENUM_VALUES,
    ChatCompletionRequest,
    summarize_validation_errors,
)

pytestmark = pytest.mark.cpu_only

MODEL = "test-model"
# Long enough that an echo of it could not pass unnoticed.
PAYLOAD = "QUJD" * 50_000
_VALUE_CANARIES = ("CLIENTSTART", "CLIENTMID", "CLIENTEND")


def _rejected_value(size: int = 200_000) -> str:
    start, middle, end = _VALUE_CANARIES
    padding = size - len(start + middle + end)
    return start + "Q" * (padding // 2) + middle + "Q" * (padding - padding // 2) + end


def _chat_body(**extra):
    return {"model": MODEL, "messages": [{"role": "user", "content": "hi"}], **extra}


@pytest.fixture
def client(serve_env: ServeTestEnv) -> TestClient:
    server = serve_env.build_openai_server(AggConfig(model_name=MODEL))
    return serve_env.client(server.app)


def _error(response):
    assert response.status_code == 400
    body = response.json()
    assert set(body) == {"error"}
    return body["error"]


def test_unknown_parameter_is_named_in_the_openai_envelope(client):
    error = _error(client.post("/v1/chat/completions", json=_chat_body(stores=False)))
    assert error == {
        "message": "Unknown parameter: 'stores'.",
        "type": "invalid_request_error",
        "param": "stores",
        "code": None,
    }


def test_missing_parameter_is_named(client):
    error = _error(client.post("/v1/chat/completions", json={"model": MODEL}))
    assert error["message"] == "Missing required parameter: 'messages'."
    assert error["param"] == "messages"


def test_context_role_returns_the_openai_error_object(serve_env: ServeTestEnv) -> None:
    server = serve_env.build_openai_server(AggConfig(role=ServerRole.CONTEXT, model_name=MODEL))
    client = serve_env.client(server.app)
    resp = client.post("/v1/chat/completions", json={"model": MODEL})
    assert resp.status_code == 400
    assert resp.json() == {
        "error": {
            "message": "Missing required parameter: 'messages'.",
            "type": "invalid_request_error",
            "param": "messages",
            "code": None,
        }
    }


def test_invalid_value_names_its_location_but_not_the_value(client):
    error = _error(client.post("/v1/chat/completions", json=_chat_body(temperature="warm")))
    assert error["message"].startswith("Invalid value for 'temperature': ")
    assert "warm" not in error["message"]
    assert error["param"] == "temperature"


def test_an_oversized_parameter_name_is_cut_short(client):
    name = "x" * 200_000
    response = client.post("/v1/chat/completions", json=_chat_body(**{name: 1}))
    error = _error(response)
    shown = "x" * 64 + "..."
    assert error["message"] == f"Unknown parameter: '{shown}'."
    assert error["param"] == shown
    assert len(response.content) < 1024


def test_a_rejected_tool_is_named_by_its_place_not_its_name(client):
    name = "t" * 200_000
    enum = [f"v{i}" for i in range(TOOL_PARAM_MAX_ENUM_VALUES + 1)]
    tool = {
        "type": "function",
        "function": {
            "name": name,
            "parameters": {"type": "object", "properties": {"p": {"type": "string", "enum": enum}}},
        },
    }
    response = client.post("/v1/chat/completions", json=_chat_body(tools=[tool]))
    error = _error(response)
    assert error["param"] == "tools.0.function"
    assert error["message"].startswith("Invalid value for 'tools.0.function': ")
    assert "enum values" in error["message"]
    assert name[:64] not in response.text
    assert len(response.content) < 1024


def test_an_unknown_parameter_named_like_a_type_is_still_named(client):
    """A client key is named even when it looks like a union member's tag."""
    error = _error(client.post("/v1/chat/completions", json=_chat_body(Temperature=0.5, list=1)))
    assert error["message"] == "Unknown parameter: 'Temperature'. Unknown parameter: 'list'."
    assert error["param"] == "Temperature"


def test_the_name_of_an_unknown_or_missing_parameter_is_kept():
    errors = [
        {"type": "missing", "loc": ("body", "tools", 0, "Function"), "msg": "m"},
        {"type": "extra_forbidden", "loc": ("body", "body"), "msg": "x"},
    ]
    assert summarize_validation_errors(errors) == (
        "Missing required parameter: 'tools.0.Function'. Unknown parameter: 'body'.",
        "tools.0.Function",
    )


def test_body_that_is_not_json_has_no_parameter(client):
    error = _error(
        client.post(
            "/v1/chat/completions",
            content=b"{not json",
            headers={"content-type": "application/json"},
        )
    )
    assert error["message"].startswith("The request body is not valid JSON: ")
    assert error["param"] is None


def test_a_body_that_is_not_msgpack_has_no_parameter_and_is_not_quoted(client):
    response = client.post(
        "/v1/chat/completions",
        content=b"\xc1" + PAYLOAD.encode(),
        headers={"content-type": "application/json", "x-trtllm-msgpack": "1"},
    )
    error = _error(response)
    assert error["message"].startswith("The request body is not valid JSON: msgpack: ")
    assert error["param"] is None
    assert PAYLOAD[:64] not in response.text


def test_a_union_field_is_named_by_its_own_place_once(client):
    """A message's content is a union; every member's error points at it.

    Pydantic reports each member of the message and content unions, tagged
    with the member's type. The client is told about the field it sent.
    """
    error = _error(
        client.post(
            "/v1/chat/completions",
            json={"model": MODEL, "messages": [{"role": "user", "content": 123}]},
        )
    )
    assert error["param"] == "messages.0.content"
    assert error["message"].startswith("Invalid value for 'messages.0.content': ")
    assert error["message"].count("Invalid value") == 1
    assert "MessageParam" not in error["message"]


def test_union_members_whose_literal_rejected_the_value_are_left_out():
    errors = [
        {"type": "literal_error", "loc": ("body", "messages", 0, "System", "role"), "msg": "x"},
        {
            "type": "string_type",
            "loc": ("body", "messages", 0, "System", "content", "str"),
            "msg": "s",
        },
        {
            "type": "string_type",
            "loc": ("body", "messages", 0, "User", "content", "str"),
            "msg": "s",
        },
        {
            "type": "list_type",
            "loc": ("body", "messages", 0, "User", "content", "list[P]"),
            "msg": "l",
        },
    ]
    assert summarize_validation_errors(errors) == (
        "Invalid value for 'messages.0.content': s.",
        "messages.0.content",
    )


def test_a_literal_no_union_member_accepts_is_named_once():
    errors = [
        {"type": "literal_error", "loc": ("body", "messages", 0, "System", "role"), "msg": "a"},
        {"type": "literal_error", "loc": ("body", "messages", 0, "User", "role"), "msg": "b"},
    ]
    assert summarize_validation_errors(errors) == (
        "Invalid value for 'messages.0.role': a.",
        "messages.0.role",
    )


@pytest.mark.parametrize("field", ["thinking", "lora_request"])
@pytest.mark.parametrize("size", [32, 200_000])
def test_rejected_discriminator_and_lora_values_are_not_quoted(field: str, size: int) -> None:
    value = _rejected_value(size)
    if field == "thinking":
        annotation = AnthropicThinkingConfig
        invalid = {"type": value}
    else:
        annotation = LoRARequest
        invalid = {"lora_name": "adapter", "lora_int_id": 1, "lora_ckpt_source": value}
    with pytest.raises(ValidationError) as raised:
        TypeAdapter(annotation).validate_python(invalid)
    errors = [{**e, "loc": ("body", field, *e["loc"])} for e in raised.value.errors()]
    message, param = summarize_validation_errors(errors)
    message_size = len(message)
    quotes_value = any(canary in message for canary in _VALUE_CANARIES)
    assert message_size <= 256
    assert not quotes_value
    assert param == field
    if field == "thinking":
        assert "'enabled'" in message
        assert "'adaptive'" in message
        assert "'disabled'" in message


def test_bare_literal_alternative_does_not_hide_tool_choice_type() -> None:
    annotation = ChatCompletionRequest.model_fields["tool_choice"].annotation
    with pytest.raises(ValidationError) as raised:
        TypeAdapter(annotation).validate_python({"type": "invalid", "function": {"name": "f"}})
    errors = [{**e, "loc": ("body", "tool_choice", *e["loc"])} for e in raised.value.errors()]
    message, _ = summarize_validation_errors(errors)
    assert "tool_choice.type" in message
    assert "'function'" in message


def test_non_discriminator_literal_does_not_discard_tool_choice_mode() -> None:
    errors = [
        {
            "type": "literal_error",
            "loc": ("body", "tool_choice", "literal['none','auto','required']"),
            "msg": "literal",
        },
        {
            "type": "literal_error",
            "loc": ("body", "tool_choice", "ToolChoiceAllowed", "mode"),
            "msg": "mode",
        },
        {
            "type": "literal_error",
            "loc": ("body", "tool_choice", "ToolChoiceFunction", "type"),
            "msg": "type",
        },
    ]
    assert summarize_validation_errors(errors) == (
        "Invalid value for 'tool_choice.mode': mode.",
        "tool_choice.mode",
    )


@pytest.mark.parametrize("kind", ["value_error", "json_invalid"])
def test_validation_error_messages_have_per_error_and_joined_caps(kind: str) -> None:
    errors = [
        {"type": kind, "loc": ("body", f"field{i}"), "msg": PAYLOAD, "ctx": {"error": PAYLOAD}}
        for i in range(20)
    ]
    single, _ = summarize_validation_errors(errors[:1])
    joined, _ = summarize_validation_errors(errors)
    single_size, joined_size = len(single), len(joined)
    assert single_size <= 256
    assert joined_size <= 4096


@pytest.mark.parametrize("server_fixture", ["client", "disagg"])
@pytest.mark.parametrize("field", ["thinking", "lora_request", "response_format"])
def test_large_rejected_values_are_bounded_at_the_http_boundary(
    request: pytest.FixtureRequest,
    monkeypatch: pytest.MonkeyPatch,
    server_fixture: str,
    field: str,
) -> None:
    server = request.getfixturevalue(server_fixture)
    http: TestClient = server if server_fixture == "client" else server.client
    warning = Mock()
    monkeypatch.setattr(openai_disagg_server.logger, "warning", warning)
    value = _rejected_value()
    if field == "thinking":
        path = "/v1/messages"
        body = {**_chat_body(), "max_tokens": 16, "thinking": {"type": value}}
    else:
        path = "/v1/chat/completions"
        invalid = (
            {"lora_name": "adapter", "lora_int_id": 1, "lora_ckpt_source": value}
            if field == "lora_request"
            else {"type": "structural_tag", "format": {"type": value}}
        )
        body = _chat_body(**{field: invalid})
    response = http.post(path, json=body)
    assert response.status_code == 400
    body_size = len(response.content)
    quotes_value = any(canary in response.text for canary in _VALUE_CANARIES)
    assert body_size < 1024
    assert not quotes_value
    if server_fixture == "disagg" and field != "thinking":
        warning.assert_called_once()
        logged = warning.call_args.args[0]
        log_size = len(logged)
        quotes_value_in_log = any(canary in logged for canary in _VALUE_CANARIES)
        assert log_size < 4608
        assert not quotes_value_in_log


def test_many_errors_are_counted_not_listed(client):
    extra = {f"field{i:02d}": i for i in range(25)}
    error = _error(client.post("/v1/chat/completions", json=_chat_body(**extra)))
    assert error["message"].startswith(
        "Unknown parameter: 'field00'. Unknown parameter: 'field01'."
    )
    assert error["message"].endswith("(5 more errors not shown.)")
    assert error["param"] == "field00"


def test_the_request_and_the_server_source_are_not_quoted(client):
    response = client.post(
        "/v1/chat/completions",
        json=_chat_body(temperature=PAYLOAD, huge=PAYLOAD),
    )
    error = _error(response)
    assert PAYLOAD[:64] not in response.text
    assert len(response.content) < 16384
    assert ".py" not in response.text
    # Both rejected values are named by location; neither is quoted.
    assert "'temperature'" in error["message"]
    assert "'huge'" in error["message"]


def test_messages_route_keeps_the_anthropic_envelope_without_quoting_the_request(client):
    response = client.post(
        "/v1/messages",
        json={
            "model": MODEL,
            "max_tokens": PAYLOAD,
            "messages": [{"role": "user", "content": "hi"}],
        },
    )
    assert response.status_code == 400
    body = response.json()
    assert body["type"] == "error"
    assert body["error"]["type"] == "invalid_request_error"
    assert body["error"]["message"].startswith("Invalid value for 'max_tokens': ")
    assert PAYLOAD[:64] not in response.text
    assert ".py" not in response.text


@pytest.fixture
def disagg(serve_env: ServeTestEnv, monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """Real construction and routes, with backend and metric-call spies only."""
    server = serve_env.build_disagg_server(DisaggConfig())
    backend = AsyncMock()
    monkeypatch.setattr(server._service, "openai_chat_completion", backend)
    counter = server._perf_metrics_collector.validation_exceptions
    monkeypatch.setattr(counter, "inc", Mock(wraps=counter.inc))
    return SimpleNamespace(
        client=serve_env.client(server.app, lifespan=False),
        server=server,
        service=SimpleNamespace(openai_chat_completion=backend),
    )


def test_disaggregated_server_answers_with_the_same_envelope(disagg):
    response = disagg.client.post(
        "/v1/chat/completions", json=_chat_body(stores=False, huge=PAYLOAD)
    )

    assert _error(response) == {
        "message": "Unknown parameter: 'stores'. Unknown parameter: 'huge'.",
        "type": "invalid_request_error",
        "param": "stores",
        "code": None,
    }
    assert PAYLOAD[:64] not in response.text
    disagg.service.openai_chat_completion.assert_not_called()
    disagg.server._perf_metrics_collector.validation_exceptions.inc.assert_called_once()


def test_disaggregated_validation_log_escapes_control_characters(
    disagg: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    warning = Mock()
    monkeypatch.setattr(openai_disagg_server.logger, "warning", warning)
    _error(disagg.client.post("/v1/chat/completions", json=_chat_body(**{"field\n\x1b": 1})))
    warning.assert_called_once()
    logged = warning.call_args.args[0]
    has_controls = any(ord(char) < 32 or ord(char) == 127 for char in logged)
    assert not has_controls


def test_disaggregated_server_names_a_missing_body(disagg):
    error = _error(disagg.client.post("/v1/chat/completions"))

    assert error["message"] == "The request body is missing."
    assert error["param"] is None
    disagg.service.openai_chat_completion.assert_not_called()


def test_disaggregated_messages_route_keeps_the_anthropic_envelope(disagg):
    response = disagg.client.post(
        "/v1/messages",
        json={
            "model": MODEL,
            "max_tokens": PAYLOAD,
            "messages": [{"role": "user", "content": "hi"}],
        },
    )

    assert response.status_code == 400
    body = response.json()
    assert body["type"] == "error"
    assert body["error"]["type"] == "invalid_request_error"
    assert body["error"]["message"].startswith("Invalid value for 'max_tokens': ")
    assert PAYLOAD[:64] not in response.text
