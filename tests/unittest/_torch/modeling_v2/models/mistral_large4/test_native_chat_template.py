# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Mistral-Large-4 serving-extension and chat-parsing wiring.

Generic mistral-common chat rendering, reasoning_effort, and response_format rules are
covered in ``tests/unittest/serve/test_mistral_native_chat.py``. This module checks the
``mistral_large_4`` model_type registration and token-id named tool_choice rejection.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from tensorrt_llm.serve.chat_utils import parse_chat_messages_coroutines
from tensorrt_llm.serve.openai_protocol import ChatCompletionRequest
from tensorrt_llm.serve.serving_extensions import apply_model_chat_extensions

pytestmark = pytest.mark.cpu_only

_WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}

_CITY_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "city",
        "schema": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
        "strict": True,
    },
}


def _request(**fields):
    fields.setdefault("messages", [{"role": "user", "content": "Hi."}])
    return ChatCompletionRequest(model="m", **fields)


@pytest.mark.parametrize("effort", ["low", "high"])
def test_top_level_reasoning_effort_reaches_the_renderer(effort):
    request = _request(reasoning_effort=effort, chat_template_kwargs={"x": 1})
    apply_model_chat_extensions(request, "mistral_large_4")
    assert request.chat_template_kwargs == {"x": 1, "reasoning_effort": effort}


def test_reasoning_effort_conflicts_and_chat_template_are_rejected():
    request = _request()
    apply_model_chat_extensions(request, "mistral_large_4")
    assert request.chat_template_kwargs is None

    with pytest.raises(ValueError, match="conflicts"):
        apply_model_chat_extensions(
            _request(reasoning_effort="high", chat_template_kwargs={"reasoning_effort": "low"}),
            "mistral_large_4",
        )
    with pytest.raises(ValueError, match="chat_template is not supported"):
        apply_model_chat_extensions(_request(chat_template="{{ messages }}"), "mistral_large_4")


@pytest.mark.parametrize(
    "response_format,passed",
    [(_CITY_FORMAT, True), ({"type": "json_object"}, True), ({"type": "text"}, False)],
)
def test_a_response_format_reaches_the_renderer(response_format, passed):
    request = _request(
        response_format=response_format,
        chat_template_kwargs={"response_format": {"type": "json_object"}},
    )
    apply_model_chat_extensions(request, "mistral_large_4")
    expected = request.response_format.model_dump(exclude_none=True) if passed else None
    assert (request.chat_template_kwargs or {}).get("response_format") == expected


def test_a_named_tool_choice_on_token_ids_is_a_bad_request(tmp_path, monkeypatch):
    from test_common.serve.app_factory import AggConfig, isolated_serve_env

    from tensorrt_llm.serve.render import chat as render_chat_module

    async def render_token_ids(**kwargs):
        return [1, 2, 3]

    monkeypatch.setattr(render_chat_module, "async_apply_chat_template", render_token_ids)

    class _Config:
        model_type = "mistral_large_4"
        vocab_size = 1024

    with isolated_serve_env(tmp_path) as env:
        server = env.build_openai_server(AggConfig(model_name="m", tool_parser="qwen3"))
        server.model_config = _Config()
        server.tokenizer = SimpleNamespace(tokenizer=SimpleNamespace(vocab_size=_Config.vocab_size))
        server.generator.args.guided_decoding_backend = "xgrammar"
        response = env.client(server.app).post(
            "/v1/chat/completions",
            json={
                "model": "m",
                "messages": [{"role": "user", "content": "Weather in Paris?"}],
                "tools": [_WEATHER_TOOL],
                "tool_choice": {"type": "function", "function": {"name": "get_weather"}},
                "max_tokens": 4,
            },
        )
    assert response.status_code == 400, response.text
    assert "renders token ids" in response.text


@pytest.mark.parametrize("model_type,kept", [("mistral_large_4", True), ("llama", False)])
def test_text_part_lists_keep_their_parts(model_type, kept):
    messages = [
        {
            "role": "system",
            "content": [
                {"type": "text", "text": "Rule one."},
                {"type": "text", "text": "Rule two."},
            ],
        },
        {"role": "user", "content": "One string."},
    ]
    conversation, media, _, _ = parse_chat_messages_coroutines(
        messages, type("Config", (), {"model_type": model_type})()
    )
    asyncio.run(media)
    assert conversation[0]["content"] == "Rule one.\nRule two."
    assert conversation[0].get("content_parts") == (["Rule one.", "Rule two."] if kept else None)
    assert "content_parts" not in conversation[1]
