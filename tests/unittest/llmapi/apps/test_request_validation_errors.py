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

The server is built by its own constructor, so these requests meet the
exception handler a deployed server installs; only model loading and the
full route table are left out.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

import pytest
from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.testclient import TestClient

from tensorrt_llm.serve.openai_disagg_server import OpenAIDisaggServer
from tensorrt_llm.serve.openai_protocol import ChatCompletionRequest
from tensorrt_llm.serve.openai_server import OpenAIServer

pytestmark = pytest.mark.cpu_only

MODEL = "test-model"
# Long enough that an echo of it could not pass unnoticed.
PAYLOAD = "QUJD" * 50_000


def _chat_body(**extra):
    return {"model": MODEL, "messages": [{"role": "user", "content": "hi"}], **extra}


@pytest.fixture(scope="module")
def client():
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


def test_disaggregated_server_answers_with_the_same_envelope():
    server = object.__new__(OpenAIDisaggServer)
    server._perf_metrics_collector = SimpleNamespace(validation_exceptions=Mock())
    server._val_err_n = 0

    async def chat(request: ChatCompletionRequest):
        raise AssertionError("an invalid body must not reach the route")

    app = FastAPI()
    app.add_exception_handler(RequestValidationError, server._request_validation_error)
    app.add_api_route("/v1/chat/completions", chat, methods=["POST"])

    response = TestClient(app).post(
        "/v1/chat/completions", json=_chat_body(stores=False, huge=PAYLOAD)
    )

    assert _error(response) == {
        "message": "Unknown parameter: 'stores'. Unknown parameter: 'huge'.",
        "type": "invalid_request_error",
        "param": "stores",
        "code": None,
    }
    server._perf_metrics_collector.validation_exceptions.inc.assert_called_once()
