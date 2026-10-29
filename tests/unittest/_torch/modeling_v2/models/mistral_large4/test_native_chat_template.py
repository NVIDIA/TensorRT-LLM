# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A mistral_large_4 chat request renders to the token ids mistral-common gives the raw
OpenAI messages.

mistral-common, not a Jinja template, renders the chat. The serving rules:

- the top-level `reasoning_effort` selects the reasoning mode; a `reasoning_effort`
  in `chat_template_kwargs` is accepted too and must agree with it;
- a request `chat_template` is rejected; `documents` and the `chat_template_kwargs`
  keys the encoder does not read are ignored;
- `add_generation_prompt` and `continue_final_message` must agree with the final
  message;
- tools reach the encoder without a null `strict`, and each text part of a content
  list reaches it as its own chunk;
- a `json_object` or named `json_schema` response format is stated in the prompt's
  model settings when the installed mistral-common has `JsonSchema`, and otherwise
  renders nothing;
- a named `tool_choice`, whose prefix is text, is a bad request on rendered token ids.

The request-rule tests run on CPU without a checkpoint. The rendering tests need
MISTRAL_LARGE4_CHECKPOINT_DIR (its tekken.json) and compare the server's rendering steps
with mistral-common encoding the raw request messages.
"""

from __future__ import annotations

import asyncio
import base64
import copy
import os
from io import BytesIO
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from tensorrt_llm.inputs.utils import apply_chat_template, async_apply_chat_template_with_mm
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


class _RecordingTokenizer:
    """Stands in for the native tokenizer and records each render it is asked for."""

    def __init__(self):
        self.calls = []

    def apply_chat_template(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        return [1]


def _render(tokenizer, messages, *, add_generation_prompt=True, **kwargs):
    return apply_chat_template(
        model_type="mistral_large_4",
        tokenizer=tokenizer,
        processor=None,
        conversation=messages,
        add_generation_prompt=add_generation_prompt,
        mm_placeholder_counts=[{} for _ in messages],
        **kwargs,
    )


def _request(**fields):
    fields.setdefault("messages", [{"role": "user", "content": "Hi."}])
    return ChatCompletionRequest(model="m", **fields)


def test_only_encoder_kwargs_reach_the_renderer():
    tokenizer = _RecordingTokenizer()
    _render(
        tokenizer,
        [{"role": "user", "content": "Hi."}],
        chat_template_kwargs={"reasoning_effort": "high", "enable_thinking": False, "foo": 1},
        documents=[{"title": "t", "text": "x"}],
    )
    ((_, kwargs),) = tokenizer.calls
    assert kwargs == {
        "tools": None,
        "continue_final_message": False,
        "tokenize": True,
        "return_dict": False,
        "return_tensors": None,
        "reasoning_effort": "high",
    }


def test_unset_reasoning_effort_is_left_to_the_checkpoint():
    tokenizer = _RecordingTokenizer()
    _render(tokenizer, [{"role": "user", "content": "Hi."}])
    ((_, kwargs),) = tokenizer.calls
    assert "reasoning_effort" not in kwargs


def test_continue_final_message_needs_a_final_assistant_message():
    tokenizer = _RecordingTokenizer()
    messages = [{"role": "user", "content": "Finish:"}, {"role": "assistant", "content": "The"}]
    _render(
        tokenizer,
        messages,
        add_generation_prompt=False,
        chat_template_kwargs={"continue_final_message": True},
    )
    assert tokenizer.calls[0][1]["continue_final_message"] is True


def test_top_level_continue_final_message_reaches_native_renderer():
    request = _request(
        messages=[{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello"}],
        add_generation_prompt=False,
        continue_final_message=True,
    )
    tokenizer = _RecordingTokenizer()
    _render(
        tokenizer,
        request.messages,
        add_generation_prompt=request.add_generation_prompt,
        chat_template_kwargs=request.chat_template_kwargs,
    )
    assert tokenizer.calls[0][1]["continue_final_message"] is True


def test_top_level_continue_final_message_rejects_generation_prompt():
    with pytest.raises(ValueError, match="Cannot set both"):
        _request(continue_final_message=True)


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("limit_field", ["max_tokens", "max_completion_tokens"])
def test_chat_rejects_oversized_output_before_generation(stream, limit_field):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from tensorrt_llm.serve.openai_server import OpenAIServer

    server = object.__new__(OpenAIServer)
    server.model = "m"
    server.allow_request_chat_template = False
    server.generator = SimpleNamespace(args=SimpleNamespace(max_seq_len=64))
    app = FastAPI()
    app.add_api_route("/v1/chat/completions", server.openai_chat, methods=["POST"])
    response = TestClient(app).post(
        "/v1/chat/completions",
        json={
            "model": "m",
            "messages": [{"role": "user", "content": "Hi"}],
            "stream": stream,
            limit_field: 999999999,
        },
    )
    assert response.status_code == 400, response.text


@pytest.mark.parametrize(
    "last_role,add_generation_prompt,continue_final_message,match",
    [
        ("assistant", True, False, "last message is from the assistant"),
        ("assistant", True, True, "Cannot set both"),
        ("user", False, True, "last message is not from the assistant"),
    ],
)
def test_generation_flags_must_agree_with_the_final_message(
    last_role, add_generation_prompt, continue_final_message, match
):
    messages = [{"role": "user", "content": "Hi."}]
    if last_role == "assistant":
        messages.append({"role": "assistant", "content": "Hello"})
    tokenizer = _RecordingTokenizer()
    with pytest.raises(ValueError, match=match):
        _render(
            tokenizer,
            messages,
            add_generation_prompt=add_generation_prompt,
            chat_template_kwargs={"continue_final_message": continue_final_message},
        )
    assert tokenizer.calls == []


def test_null_strict_is_dropped_from_tools():
    tools = [
        {"type": "function", "function": {"name": "a", "description": None, "strict": None}},
        {"type": "function", "function": {"name": "b", "strict": True}},
    ]
    original = copy.deepcopy(tools)
    tokenizer = _RecordingTokenizer()
    _render(tokenizer, [{"role": "user", "content": "Hi."}], tools=tools)
    assert tokenizer.calls[0][1]["tools"] == [
        {"type": "function", "function": {"name": "a", "description": None}},
        {"type": "function", "function": {"name": "b", "strict": True}},
    ]
    assert tools == original


def test_a_chat_template_is_rejected():
    with pytest.raises(ValueError, match="native Tekken chat template"):
        _render(_RecordingTokenizer(), [{"role": "user", "content": "Hi."}], chat_template="{{x}}")


@pytest.mark.parametrize("effort", ["low", "high"])
def test_top_level_reasoning_effort_reaches_the_renderer(effort):
    request = _request(reasoning_effort=effort, chat_template_kwargs={"x": 1})
    apply_model_chat_extensions(request, "mistral_large_4")
    assert request.chat_template_kwargs == {"x": 1, "reasoning_effort": effort}


def test_an_unset_reasoning_effort_derives_nothing():
    """The protocol's own default is not a mistral-common effort and must not leak in."""
    request = _request()
    apply_model_chat_extensions(request, "mistral_large_4")
    assert request.chat_template_kwargs is None


def test_reasoning_effort_in_kwargs_is_kept_and_must_agree():
    request = _request(chat_template_kwargs={"reasoning_effort": "low", "x": 1})
    apply_model_chat_extensions(request, "mistral_large_4")
    assert request.chat_template_kwargs == {"reasoning_effort": "low", "x": 1}

    request = _request(reasoning_effort="high", chat_template_kwargs={"reasoning_effort": "high"})
    apply_model_chat_extensions(request, "mistral_large_4")
    assert request.chat_template_kwargs == {"reasoning_effort": "high"}

    with pytest.raises(ValueError, match="conflicts"):
        apply_model_chat_extensions(
            _request(reasoning_effort="high", chat_template_kwargs={"reasoning_effort": "low"}),
            "mistral_large_4",
        )


def test_an_unknown_reasoning_effort_is_rejected():
    """mistral-common's own efforts pass; whether the checkpoint declares one is the renderer's."""
    with pytest.raises(ValueError, match="is not a reasoning effort"):
        apply_model_chat_extensions(
            _request(chat_template_kwargs={"reasoning_effort": "turbo"}), "mistral_large_4"
        )


def test_a_request_chat_template_is_rejected():
    with pytest.raises(ValueError, match="chat_template is not supported"):
        apply_model_chat_extensions(_request(chat_template="{{ messages }}"), "mistral_large_4")


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


@pytest.mark.parametrize("response_format", [_CITY_FORMAT, {"type": "json_object"}])
def test_a_response_format_renders_nothing_without_json_schema(monkeypatch, response_format):
    request_module = pytest.importorskip("mistral_common.protocol.instruct.request")
    monkeypatch.delattr(request_module, "JsonSchema", raising=False)
    tokenizer = _RecordingTokenizer()
    _render(
        tokenizer,
        [{"role": "user", "content": "Give a city as JSON."}],
        chat_template_kwargs={"response_format": response_format},
    )
    ((_, kwargs),) = tokenizer.calls
    assert "response_format" not in kwargs


def test_a_named_tool_choice_on_token_ids_is_a_bad_request(monkeypatch):
    from unittest.mock import AsyncMock

    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from tensorrt_llm.inputs import utils as input_utils
    from tensorrt_llm.serve.openai_server import OpenAIServer

    async def render_token_ids(**kwargs):
        return [1, 2, 3]

    monkeypatch.setattr(input_utils, "async_apply_chat_template", render_token_ids)

    class _Config:
        model_type = "mistral_large_4"
        vocab_size = 1024

    server = object.__new__(OpenAIServer)
    server.model = "m"
    server.allow_request_chat_template = False
    server.model_config = _Config()
    server.processor = None
    server.tokenizer = SimpleNamespace(tokenizer=SimpleNamespace(vocab_size=_Config.vocab_size))
    server.chat_template = None
    server.tool_parser = "qwen3"
    server.tool_call_id_type = "random"
    server.multimodal_server_config = None
    server.generator = SimpleNamespace(
        args=SimpleNamespace(
            gather_generation_logits=False,
            reasoning_parser=None,
            backend="pytorch",
            guided_decoding_backend="xgrammar",
            num_postprocess_workers=0,
        ),
        generate_async=None,
    )
    server.await_disconnected = AsyncMock()
    app = FastAPI()
    app.add_api_route("/v1/chat/completions", server.openai_chat, methods=["POST"])
    response = TestClient(app).post(
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


def _config(model_type):
    return type("Config", (), {"model_type": model_type})()


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
    conversation, media, _, _ = parse_chat_messages_coroutines(messages, _config(model_type))
    asyncio.run(media)
    assert conversation[0]["content"] == "Rule one.\nRule two."
    assert conversation[0].get("content_parts") == (["Rule one.", "Rule two."] if kept else None)
    assert "content_parts" not in conversation[1]


@pytest.fixture(scope="module")
def tokenizer():
    if not os.environ.get("MISTRAL_LARGE4_CHECKPOINT_DIR"):
        pytest.skip("needs checkpoint tokenizer")
    from tensorrt_llm._torch.models.checkpoints.mistral.tokenizer import MistralTokenizer

    return MistralTokenizer.from_pretrained(os.environ["MISTRAL_LARGE4_CHECKPOINT_DIR"])


def _server_ids(tokenizer, **fields):
    """The openai_chat rendering steps for one mistral_large_4 request."""
    from tensorrt_llm._torch.configs.mistral_large4 import MistralLarge4Config

    request = _request(**copy.deepcopy(fields))
    apply_model_chat_extensions(request, "mistral_large_4")
    conversation, media, counts, _ = parse_chat_messages_coroutines(
        request.messages, MistralLarge4Config(dtype="bfloat16", vocab_size=131072)
    )
    tools = None if request.tools is None else [tool.model_dump() for tool in request.tools]
    ids, _ = asyncio.run(
        async_apply_chat_template_with_mm(
            media,
            model_type="mistral_large_4",
            tokenizer=tokenizer,
            processor=None,
            conversation=conversation,
            add_generation_prompt=request.add_generation_prompt,
            mm_placeholder_counts=counts,
            tools=tools,
            documents=request.documents,
            chat_template=request.chat_template,
            chat_template_kwargs=request.chat_template_kwargs or {},
        )
    )
    return ids


def _raw_ids(tokenizer, messages, **kwargs):
    """mistral-common encoding the raw request messages."""
    return tokenizer.apply_chat_template(messages, tokenize=True, return_dict=False, **kwargs)


def _image_url(seed, size):
    pixels = np.random.default_rng(seed).integers(0, 256, size=(*size, 3), dtype=np.uint8)
    buffer = BytesIO()
    Image.fromarray(pixels, "RGB").save(buffer, "PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


def test_text_part_lists_render_part_by_part(tokenizer):
    messages = [
        {"role": "system", "content": [{"type": "text", "text": "Rule one."}] * 2},
        {
            "role": "user",
            "content": [{"type": "text", "text": "First part."}, {"type": "text", "text": "Next."}],
        },
    ]
    ids = _server_ids(tokenizer, messages=messages)
    assert ids == _raw_ids(tokenizer, messages)
    joined = [
        {"role": m["role"], "content": "\n".join(p["text"] for p in m["content"])} for m in messages
    ]
    assert ids != _raw_ids(tokenizer, joined)


def test_tools_render_as_sent(tokenizer):
    messages = [{"role": "user", "content": "Weather in Paris?"}]
    ids = _server_ids(tokenizer, messages=messages, tools=[_WEATHER_TOOL])
    assert ids == _raw_ids(tokenizer, messages, tools=[_WEATHER_TOOL])
    assert ids != _raw_ids(tokenizer, messages)


def test_top_level_reasoning_effort_renders(tokenizer):
    messages = [{"role": "user", "content": "Think."}]
    high = _server_ids(tokenizer, messages=messages, reasoning_effort="high")
    assert high == _raw_ids(tokenizer, messages, reasoning_effort="high")
    assert high != _raw_ids(tokenizer, messages, reasoning_effort="low")
    assert _server_ids(tokenizer, messages=messages) == _raw_ids(tokenizer, messages)


def test_a_reasoning_effort_the_checkpoint_does_not_declare_is_an_invalid_request(tokenizer):
    with pytest.raises(ValueError, match="reasoning_effort"):
        _server_ids(
            tokenizer, messages=[{"role": "user", "content": "Think."}], reasoning_effort="max"
        )


def test_ignored_fields_leave_the_tokens_unchanged(tokenizer):
    messages = [{"role": "user", "content": "Summarize."}]
    ids = _server_ids(
        tokenizer,
        messages=messages,
        documents=[{"title": "t", "text": "Some text."}],
        chat_template_kwargs={"enable_thinking": True},
    )
    assert ids == _raw_ids(tokenizer, messages)


def test_a_single_image_turn_renders_as_before(tokenizer):
    """The frozen MMMU prompts: one user turn of text and images, at reasoning effort low."""
    content = [
        {"type": "text", "text": "Question: "},
        {"type": "image_url", "image_url": {"url": _image_url(1, (48, 64))}},
        {"type": "text", "text": " Options: A, B."},
    ]
    messages = [{"role": "user", "content": content}]
    ids = _server_ids(
        tokenizer, messages=messages, chat_template_kwargs={"reasoning_effort": "low"}
    )
    assert ids == _raw_ids(tokenizer, messages, return_tensors=None, reasoning_effort="low")


def _direct_ids(tokenizer, messages, **kwargs):
    """mistral-common encoding the raw request messages with request-level settings."""
    from mistral_common.protocol.instruct.request import ChatCompletionRequest

    request = ChatCompletionRequest.from_openai(messages=messages, **kwargs)
    return tokenizer.mistral.encode_chat_completion(request).tokens


def test_a_response_format_renders_in_the_model_settings(tokenizer):
    try:
        from mistral_common.protocol.instruct.request import (
            JsonSchema,
            ResponseFormat,
            ResponseFormats,
        )
    except ImportError:
        pytest.skip("the installed mistral-common has no JsonSchema")

    messages = [{"role": "user", "content": "Give a city as JSON."}]
    named = _CITY_FORMAT["json_schema"]
    schema = ResponseFormat(
        type=ResponseFormats.json_schema,
        json_schema=JsonSchema(name=named["name"], schema=named["schema"], strict=True),
    )
    ids = _server_ids(tokenizer, messages=messages, response_format=_CITY_FORMAT)
    assert ids == _direct_ids(tokenizer, messages, response_format=schema)
    assert ids != _raw_ids(tokenizer, messages)
    assert _server_ids(
        tokenizer, messages=messages, response_format={"type": "json_object"}
    ) == _direct_ids(
        tokenizer, messages, response_format=ResponseFormat(type=ResponseFormats.json_object)
    )
    text = _server_ids(tokenizer, messages=messages, response_format={"type": "text"})
    assert text == _raw_ids(tokenizer, messages)


def test_a_response_format_keeps_the_image_grid(tokenizer):
    from mistral_common.protocol.instruct.request import ResponseFormat, ResponseFormats

    content = [
        {"type": "text", "text": "Describe as JSON: "},
        {"type": "image_url", "image_url": {"url": _image_url(2, (48, 64))}},
    ]
    messages = [{"role": "user", "content": content}]
    assert _direct_ids(tokenizer, messages) == _raw_ids(tokenizer, messages)
    ids = _server_ids(tokenizer, messages=messages, response_format={"type": "json_object"})
    assert ids == _direct_ids(
        tokenizer, messages, response_format=ResponseFormat(type=ResponseFormats.json_object)
    )
