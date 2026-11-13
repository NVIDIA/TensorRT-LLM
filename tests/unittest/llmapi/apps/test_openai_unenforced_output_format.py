# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""A requested output format on a server without a guided decoding backend.

Such a server builds no grammar, so a request with ``response_format`` (chat,
completions), ``text.format`` (Responses) or ``output_config.format``
(Anthropic Messages, served by the chat route) is served unconstrained. The
server says so in a warning, once per route.
"""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from tensorrt_llm.logger import logger
from tensorrt_llm.serve import openai_server as openai_server_module
from tensorrt_llm.serve.openai_protocol import (
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
    ChatMessage,
    CompletionResponse,
    CompletionResponseChoice,
    UsageInfo,
)
from tensorrt_llm.serve.render import chat as render_chat_module

from .openai_server_stub import make_stub_openai_server

pytestmark = pytest.mark.cpu_only

MODEL = "test-model"
SCHEMA = {"type": "object", "properties": {"answer": {"type": "string"}}, "required": ["answer"]}
ANSWER = '{"answer": "yes"}'
USAGE = UsageInfo(prompt_tokens=3, completion_tokens=4, total_tokens=7)
MESSAGES = [{"role": "user", "content": "Answer in JSON."}]

# A request per route, without a format.
REQUESTS = {
    "chat": ("/v1/chat/completions", {"messages": MESSAGES}),
    # Token ids: the stub engine has no input processor to tokenize text.
    "completions": ("/v1/completions", {"prompt": [1, 2, 3]}),
    "responses": ("/v1/responses", {"input": "Answer in JSON."}),
    "messages": ("/v1/messages", {"messages": MESSAGES, "max_tokens": 64}),
}
RESPONSE_FORMAT = {"type": "json_schema", "json_schema": {"name": "answer", "schema": SCHEMA}}
# The field each API asks for SCHEMA with.
FORMATS = {
    "chat": {"response_format": RESPONSE_FORMAT},
    "completions": {"response_format": RESPONSE_FORMAT},
    "responses": {"text": {"format": {"type": "json_schema", "name": "answer", "schema": SCHEMA}}},
    "messages": {"output_config": {"format": {"type": "json_schema", "schema": SCHEMA}}},
}
# The chat route rejects a format combined with a forced (named) tool call.
NAMED_TOOL_CHOICE = {
    "tools": [{"type": "function", "function": {"name": "lookup", "parameters": SCHEMA}}],
    "tool_choice": {"type": "function", "function": {"name": "lookup"}},
}
# Rejected by the stub server, which has no reasoning parser.
THINKING_BUDGET = {"thinking_token_budget": 16}


class _ModelConfig:
    model_type = "llama"
    vocab_size = 1024


class _Tokenizer:
    """Encodes every prompt to the same ids; rendering is stubbed separately."""

    def __init__(self):
        self.tokenizer = self
        self.vocab_size = _ModelConfig.vocab_size
        self.name_or_path = None

    def encode(self, text, **kwargs):
        return [1, 2, 3]


class _Generation:
    """A finished generation, as far as the routes read one before post-processing."""

    request_id = 1
    prompt_token_ids = [1, 2, 3]
    finished = True
    error = None

    async def aresult(self):
        return self

    def __await__(self):
        return self.aresult().__await__()


@pytest.fixture
def served(tmp_path, monkeypatch):
    """A stub-engine server without a guided decoding backend, and its client.

    Prompt rendering and each route's post-processing are stubbed;
    ``engine_calls`` records the sampling params that reach the engine.
    """

    async def render(**kwargs):
        return "rendered prompt"

    monkeypatch.setattr(render_chat_module, "async_apply_chat_template", render)
    completion = CompletionResponse(
        model=MODEL,
        choices=[CompletionResponseChoice(index=0, text=ANSWER, finish_reason="stop")],
        usage=USAGE,
    )
    monkeypatch.setattr(
        openai_server_module, "completion_response_post_processor", lambda *args: completion
    )
    responses_body = SimpleNamespace(
        prompt_token_ids=None, model_dump=lambda **kwargs: {"object": "response"}
    )
    monkeypatch.setattr(
        openai_server_module,
        "responses_api_create_response",
        AsyncMock(return_value=responses_body),
    )

    engine_calls = []

    def generate_async(*, inputs, sampling_params, **kwargs):
        engine_calls.append(sampling_params)
        return _Generation()

    server = make_stub_openai_server(
        tmp_path, MODEL, _ModelConfig(), generate_async, tokenizer=_Tokenizer()
    )
    assert server.generator.args.guided_decoding_backend is None
    server.await_disconnected = AsyncMock()
    server._extract_metrics = AsyncMock()
    server._create_chat_response = AsyncMock(
        return_value=ChatCompletionResponse(
            model=MODEL,
            choices=[
                ChatCompletionResponseChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content=ANSWER),
                    finish_reason="stop",
                )
            ],
            usage=USAGE,
        )
    )
    app = FastAPI()
    app.add_api_route("/v1/chat/completions", server.openai_chat, methods=["POST"])
    app.add_api_route("/v1/completions", server.openai_completion, methods=["POST"])
    app.add_api_route("/v1/responses", server.openai_responses, methods=["POST"])
    app.add_api_route("/v1/messages", server.anthropic_messages, methods=["POST"])
    return SimpleNamespace(client=TestClient(app), server=server, engine_calls=engine_calls)


@pytest.fixture
def format_warnings(monkeypatch):
    """The output-format warnings the server logs, deduplicated as in a fresh process."""
    # warning_once deduplicates against the process-wide key set; start from an empty one.
    monkeypatch.setattr(logger, "_appeared_keys", set())
    log = Mock()
    monkeypatch.setattr(logger, "log", log)

    def warnings():
        texts = [
            " ".join(str(part) for part in call.args[1:])
            for call in log.call_args_list
            if call.args[0] == logger.WARNING
        ]
        return [text for text in texts if "guided decoding backend" in text]

    return warnings


def _post(served, route, with_format=True):
    path, body = REQUESTS[route]
    body = {"model": MODEL, **body, **(FORMATS[route] if with_format else {})}
    response = served.client.post(path, json=body)
    assert response.status_code == 200, response.text
    return response


def test_chat_format_is_served_and_warned_about_once(served, format_warnings):
    _post(served, "chat")
    _post(served, "chat")

    # Served as before: the format still reaches the engine with each request.
    assert [params.guided_decoding.json for params in served.engine_calls] == [SCHEMA, SCHEMA]
    (warning,) = format_warnings()
    assert warning.startswith("Chat Completions API: 'response_format'")
    assert "accepted but not enforced" in warning
    assert "no guided decoding backend" in warning
    # xgrammar also handles the structural tags gpt-oss uses, which llguidance does not.
    assert "`guided_decoding_backend: xgrammar` in the --config YAML" in warning
    assert "llguidance" not in warning


@pytest.mark.parametrize(
    "route, prefix, field",
    [
        ("completions", "Completions API:", "'response_format'"),
        ("responses", "Responses API:", "'text.format'"),
        # Served by the chat route, so the chat route's warning, which names both fields.
        ("messages", "Chat Completions API:", "'output_config.format'"),
    ],
)
def test_format_on_other_routes_is_served_and_warned_about(
    served, format_warnings, route, prefix, field
):
    _post(served, route)

    (params,) = served.engine_calls
    assert params.guided_decoding.json == SCHEMA
    (warning,) = format_warnings()
    assert warning.startswith(prefix)
    assert field in warning
    assert "accepted but not enforced" in warning


def test_constraint_an_extension_derives_from_the_format_is_warned_about(served, format_warnings):
    # The kimi_k3 extension places the schema on its response channel through a structural tag.
    served.server.generator.args.reasoning_parser = "kimi_k3"

    _post(served, "chat")

    (params,) = served.engine_calls
    assert params.guided_decoding.json is None
    assert params.guided_decoding.structural_tag is not None
    (warning,) = format_warnings()
    assert warning.startswith("Chat Completions API: 'response_format'")


def test_each_route_warns_once(served, format_warnings):
    for route in ["chat", "messages", "completions", "responses"] * 2:
        _post(served, route)

    routes = [warning.split(":", 1)[0] for warning in format_warnings()]
    assert routes == ["Chat Completions API", "Completions API", "Responses API"]


@pytest.mark.parametrize("route", list(REQUESTS))
def test_request_without_a_format_logs_nothing(served, format_warnings, route):
    _post(served, route, with_format=False)

    (params,) = served.engine_calls
    assert params.guided_decoding is None
    assert format_warnings() == []


@pytest.mark.parametrize("backend", ["xgrammar", "llguidance"])
@pytest.mark.parametrize("route", list(REQUESTS))
def test_server_with_a_backend_logs_nothing(served, format_warnings, route, backend):
    served.server.generator.args.guided_decoding_backend = backend

    _post(served, route)

    (params,) = served.engine_calls
    assert params.guided_decoding.json == SCHEMA
    assert format_warnings() == []


@pytest.mark.parametrize(
    "route, rejected",
    [
        pytest.param("chat", NAMED_TOOL_CHOICE, id="chat-named-tool-choice"),
        pytest.param("chat", THINKING_BUDGET, id="chat-thinking-budget"),
        pytest.param("completions", THINKING_BUDGET, id="completions-thinking-budget"),
    ],
)
def test_rejected_request_leaves_the_warning_to_a_served_one(
    served, format_warnings, route, rejected
):
    path, body = REQUESTS[route]
    response = served.client.post(path, json={"model": MODEL, **body, **FORMATS[route], **rejected})
    assert response.status_code == 400, response.text
    assert format_warnings() == []

    _post(served, route)

    (warning,) = format_warnings()
    assert "accepted but not enforced" in warning


def _harmony_client(served, monkeypatch):
    """A client of the stub server's chat route in Harmony (gpt-oss) mode."""
    server = served.server
    server.use_harmony = True
    server.harmony_adapter = SimpleNamespace(get_stop_tokens=lambda: [7])
    monkeypatch.setattr(
        openai_server_module, "tokenize_harmony_chat_request", lambda *args, **kwargs: [1, 2, 3]
    )
    app = FastAPI()
    app.add_api_route("/v1/chat/completions", server.chat_harmony, methods=["POST"])
    return TestClient(app)


def test_harmony_rejected_request_leaves_the_warning_to_a_served_one(
    served, format_warnings, monkeypatch
):
    client = _harmony_client(served, monkeypatch)
    path, body = REQUESTS["chat"]
    body = {"model": MODEL, **body, **FORMATS["chat"]}

    response = client.post(path, json={**body, **THINKING_BUDGET})
    assert response.status_code == 400, response.text
    assert format_warnings() == []

    response = client.post(path, json=body)
    assert response.status_code == 200, response.text
    (warning,) = format_warnings()
    assert warning.startswith("Chat Completions API: 'response_format'")


def test_harmony_chat_format_is_served_and_warned_about(served, format_warnings, monkeypatch):
    path, body = REQUESTS["chat"]
    response = _harmony_client(served, monkeypatch).post(
        path, json={"model": MODEL, **body, **FORMATS["chat"]}
    )

    assert response.status_code == 200, response.text
    (params,) = served.engine_calls
    # gpt-oss places the schema on its final channel through a structural tag.
    assert params.guided_decoding.structural_tag is not None
    (warning,) = format_warnings()
    assert warning.startswith("Chat Completions API: 'response_format'")


def test_harmony_responses_format_is_served_and_warned_about(
    served, format_warnings, harmony_vocab
):
    served.server.use_harmony = True
    path, body = REQUESTS["responses"]
    response = served.client.post(path, json={"model": MODEL, **body, **FORMATS["responses"]})

    assert response.status_code == 200, response.text
    (params,) = served.engine_calls
    # As on Harmony chat, gpt-oss carries the schema in a structural tag on its final channel.
    assert params.guided_decoding.json is None
    tag = json.loads(params.guided_decoding.structural_tag)
    (final_channel,) = tag["format"]["tags"]
    assert final_channel["begin"] == "<|start|>assistant<|channel|>final<|message|>"
    assert final_channel["content"]["json_schema"] == SCHEMA
    (warning,) = format_warnings()
    assert warning.startswith("Responses API: 'text.format'")
