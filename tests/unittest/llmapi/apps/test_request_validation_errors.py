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

Each server is built by its own constructor, so these requests meet the
exception handler a deployed server installs. The standard server leaves out
model loading and the full route table; the disaggregated one uses mocks for
its coordinator and request service.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

import pytest
from fastapi.testclient import TestClient

from tensorrt_llm.inputs.media_io import BaseMediaIO
from tensorrt_llm.llmapi.disagg_utils import DisaggServerConfig
from tensorrt_llm.serve import openai_disagg_server
from tensorrt_llm.serve.openai_disagg_server import OpenAIDisaggServer
from tensorrt_llm.serve.openai_protocol import (
    TOOL_PARAM_MAX_ENUM_VALUES,
    summarize_validation_errors,
)
from tensorrt_llm.serve.openai_server import OpenAIServer

pytestmark = pytest.mark.cpu_only

MODEL = "test-model"
# Long enough that an echo of it could not pass unnoticed.
PAYLOAD = "QUJD" * 50_000


def _chat_body(**extra):
    return {"model": MODEL, "messages": [{"role": "user", "content": "hi"}], **extra}


@pytest.fixture(scope="module")
def client():
    previous_media_executor = BaseMediaIO._executor
    with (
        patch.object(OpenAIServer, "_init_llm"),
        patch.object(OpenAIServer, "register_routes"),
    ):
        server = OpenAIServer(
            generator=MagicMock(),
            model=MODEL,
            tool_parser=None,
            server_role=None,
            metadata_server_cfg=None,
        )
    # The constructor publishes its media-decode pool process-wide: put the
    # previous one back and retire the server's pool, as
    # openai_server_stub.make_stub_openai_server does.
    BaseMediaIO.set_executor(previous_media_executor)
    server._media_load_executor.shutdown(wait=False)
    server.app.add_api_route("/v1/chat/completions", server.openai_chat, methods=["POST"])
    server.app.add_api_route("/v1/messages", server.anthropic_messages, methods=["POST"])
    return TestClient(server.app)


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
def disagg(monkeypatch, tmp_path):
    """A disaggregated server built by its constructor, and its request service.

    The coordinator, the request service and the metrics are mocks, so no
    worker is contacted; the app, its exception handler and its routes are the
    ones the constructor installs.
    """
    # register_routes mounts a prometheus multiprocess collector, which needs
    # the directory that startup would otherwise provide.
    monkeypatch.setenv("PROMETHEUS_MULTIPROC_DIR", str(tmp_path))
    service = Mock()
    monkeypatch.setattr(openai_disagg_server, "DisaggCoordinatorService", Mock())
    monkeypatch.setattr(
        openai_disagg_server, "OpenAIDisaggregatedService", Mock(return_value=service)
    )
    monkeypatch.setattr(openai_disagg_server, "DisaggPerfMetricsCollector", Mock())
    server = OpenAIDisaggServer(DisaggServerConfig(server_configs=[]))
    return SimpleNamespace(client=TestClient(server.app), server=server, service=service)


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
