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
"""A Mistral-native chat renders to the ids mistral-common gives the raw OpenAI messages.

The Mistral-native format gives the model type ``mistral_common``; mistral-common, not a
Jinja template, renders its chats. The request-rule tests use a recording tokenizer; the
rendering tests compare the chat route's steps with mistral-common on the v15 Tekken file
under LLM_MODELS_ROOT (model settings and images), and one test reads the v13 file, which
declares no model settings.
"""

import asyncio
import base64
import copy
from io import BytesIO
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image
from test_common.mistral_tekken import load_tekken

from tensorrt_llm._torch.models.checkpoints.mistral.tokenizer import MistralTokenizer
from tensorrt_llm.inputs.utils import ChatRejectedError, apply_chat_template
from tensorrt_llm.serve.chat_utils import parse_chat_messages_coroutines
from tensorrt_llm.serve.openai_protocol import ChatCompletionRequest
from tensorrt_llm.serve.render import render_conversation_with_media
from tensorrt_llm.serve.serving_extensions import apply_model_chat_extensions, get_serving_extension

pytestmark = pytest.mark.cpu_only

_MODEL_TYPE = "mistral_common"
_HI = [{"role": "user", "content": "Hi."}]
_WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
    },
}
_CITY_FORMAT = {
    "type": "json_schema",
    "json_schema": {"name": "city", "schema": {"type": "object"}, "strict": True},
}


class _RecordingTokenizer(MistralTokenizer):
    """Records each mistral-common request the renderer encodes."""

    def __init__(self):
        self.calls = []
        self._grid_only_mistral = SimpleNamespace(encode_chat_completion=self._encode)

    def _encode(self, request):
        self.calls.append(request)
        return SimpleNamespace(tokens=[1])


def _render(tokenizer, messages, *, add_generation_prompt=True, counts=None, **kwargs):
    return apply_chat_template(
        model_type=_MODEL_TYPE,
        tokenizer=tokenizer,
        processor=None,
        conversation=messages,
        add_generation_prompt=add_generation_prompt,
        mm_placeholder_counts=counts or [{} for _ in messages],
        **kwargs,
    )


def _config(model_type=_MODEL_TYPE):
    return type("Config", (), {"model_type": model_type})()


def test_the_renderer_gets_only_what_mistral_common_reads(monkeypatch):
    from tensorrt_llm._torch.models.checkpoints.mistral import tokenizer as tokenizer_module

    warnings = []
    monkeypatch.setattr(
        tokenizer_module.logger, "warning_once", lambda message, key: warnings.append(message)
    )
    tools = [
        {"type": "function", "function": {"name": "a", "description": None, "strict": None}},
        {"type": "function", "function": {"name": "b", "strict": True}},
    ]
    original = copy.deepcopy(tools)
    tokenizer = _RecordingTokenizer()
    _render(
        tokenizer,
        _HI,
        tools=tools,
        chat_template_kwargs={"reasoning_effort": "high", "enable_thinking": False, "foo": 1},
        documents=[{"title": "t", "text": "x"}],
        # A server --chat_template reaches the renderer as this.
        chat_template="{{ messages }}",
    )
    _render(tokenizer, _HI)
    request, unset = tokenizer.calls
    # What mistral-common does not read is named, not dropped silently.
    assert len(warnings) == 1
    for ignored in ("enable_thinking", "foo", "documents", "chat_template"):
        assert repr(ignored) in warnings[0]
    # A null `strict` is dropped: mistral-common refuses it.
    assert [tool.function.name for tool in request.tools] == ["a", "b"]
    assert str(getattr(request.reasoning_effort, "value", request.reasoning_effort)) == "high"
    assert tools == original
    assert unset.reasoning_effort is None  # left to the checkpoint


@pytest.mark.parametrize(
    "last_role,add_generation_prompt,continue_final_message,match",
    [
        ("assistant", False, True, None),
        ("assistant", True, False, "last message is from the assistant"),
        ("assistant", True, True, "Cannot set both"),
        ("user", False, True, "last message is not from the assistant"),
    ],
)
def test_generation_flags_must_agree_with_the_final_message(
    last_role, add_generation_prompt, continue_final_message, match
):
    messages = _HI + (
        [{"role": "assistant", "content": "Hello"}] if last_role == "assistant" else []
    )
    tokenizer = _RecordingTokenizer()
    kwargs = dict(
        add_generation_prompt=add_generation_prompt,
        chat_template_kwargs={"continue_final_message": continue_final_message},
    )
    if match is None:
        _render(tokenizer, messages, **kwargs)
        # mistral-common continues the final assistant message as a prefix.
        assert tokenizer.calls[0].messages[-1].prefix is True
        return
    # An invalid request, as when mistral-common itself rejects the chat.
    with pytest.raises(ChatRejectedError, match=match):
        _render(tokenizer, messages, **kwargs)
    assert tokenizer.calls == []


def test_the_renderer_refuses_an_hf_tokenizer_and_unloaded_images():
    from tensorrt_llm.tokenizer import TransformersTokenizer

    with pytest.raises(ValueError, match="mistral-common's MistralTokenizer"):
        _render(TransformersTokenizer(SimpleNamespace()), _HI)
    # A route that renders without loading the images gets a clear error, not an IndexError.
    tokenizer = _RecordingTokenizer()
    with pytest.raises(ValueError, match="1 images, but 0 were loaded"):
        _render(tokenizer, [{**_HI[0], "content_parts": []}], counts=[{"[IMG]": 1}])
    assert tokenizer.calls == []


_HIGH, _LOW = {"reasoning_effort": "high"}, {"reasoning_effort": "low"}


@pytest.mark.parametrize(
    "fields,expected",
    [
        ({}, None),
        ({**_HIGH, "chat_template_kwargs": {"x": 1}}, {"x": 1, **_HIGH}),
        ({**_HIGH, "chat_template_kwargs": _HIGH}, _HIGH),
        ({**_HIGH, "chat_template_kwargs": _LOW}, ValueError("conflicts")),
        ({"chat_template": "{{ messages }}"}, ValueError("chat_template is not supported")),
        (
            {"chat_template_kwargs": {"chat_template": "{{ messages }}"}},
            ValueError("chat_template is not supported"),
        ),
    ],
)
def test_reasoning_effort_and_chat_template(fields, expected):
    """An unset effort derives nothing; set ones must agree; a request template is refused."""
    request = ChatCompletionRequest(model="m", messages=_HI, **fields)
    if isinstance(expected, ValueError):
        with pytest.raises(ValueError, match=str(expected)):
            apply_model_chat_extensions(request, _MODEL_TYPE)
        return
    apply_model_chat_extensions(request, _MODEL_TYPE)
    assert request.chat_template_kwargs == expected


@pytest.mark.parametrize(
    "response_format,passed", [(_CITY_FORMAT, True), ({"type": "text"}, False)]
)
def test_a_json_response_format_reaches_the_renderer(response_format, passed):
    request = ChatCompletionRequest(
        model="m",
        messages=_HI,
        response_format=response_format,
        chat_template_kwargs={"response_format": {"type": "json_object"}},
    )
    apply_model_chat_extensions(request, _MODEL_TYPE)
    dumped = request.response_format.model_dump(exclude_none=True)
    assert request.chat_template_kwargs == ({"response_format": dumped} if passed else {})


@pytest.mark.parametrize("model_type", [_MODEL_TYPE, "llama"])
def test_chat_parsing(model_type):
    """mistral_common keeps each text part and a history call's arguments as sent."""
    native = model_type == _MODEL_TYPE
    parts = [{"type": "text", "text": "Rule one."}, {"type": "text", "text": "Rule two."}]
    conversation, media, _, _ = parse_chat_messages_coroutines(
        [{"role": "system", "content": parts}, *_HI], _config(model_type)
    )
    asyncio.run(media)
    assert conversation[0]["content"] == "Rule one.\nRule two."
    assert conversation[0].get("content_parts") == (["Rule one.", "Rule two."] if native else None)
    assert "content_parts" not in conversation[1]

    call = {"id": "c0", "type": "function", "function": {"name": "f", "arguments": "{not json"}}
    history = [*_HI, {"role": "assistant", "tool_calls": [call]}]
    if not native:
        with pytest.raises(ValueError, match="must be valid JSON"):
            parse_chat_messages_coroutines(history, _config(model_type))
        return
    conversation, media, _, _ = parse_chat_messages_coroutines(history, _config(model_type))
    asyncio.run(media)
    assert conversation[1]["tool_calls"][0]["function"]["arguments"] == "{not json"


@pytest.fixture(scope="module")
def tokenizer():
    # Model settings (reasoning effort, response format) and an image encoder.
    return load_tekken("v15")


def _server_ids(tokenizer, **fields):
    """The chat route's rendering steps for one request."""
    return _request_ids(tokenizer, ChatCompletionRequest(model="m", **copy.deepcopy(fields)))


def _request_ids(tokenizer, request):
    """The chat route's rendering steps for a built request (the Anthropic route builds one)."""
    apply_model_chat_extensions(request, _MODEL_TYPE)
    conversation, media, counts, _ = parse_chat_messages_coroutines(request.messages, _config())
    resources = SimpleNamespace(
        model_type=_MODEL_TYPE,
        tokenizer=tokenizer,
        processor=None,
        default_chat_template=None,
        extension=get_serving_extension(_MODEL_TYPE),
    )
    rendered, _ = asyncio.run(
        render_conversation_with_media(
            resources,
            media,
            conversation=conversation,
            mm_placeholder_counts=counts,
            add_generation_prompt=request.add_generation_prompt,
            tools=None if request.tools is None else [tool.model_dump() for tool in request.tools],
            documents=request.documents,
            chat_template_kwargs=request.chat_template_kwargs or {},
            injected_chat_template_kwargs=request.injected_chat_template_kwargs,
            truncate_prompt_tokens=request.truncate_prompt_tokens,
        )
    )
    assert rendered.text is None
    return rendered.token_ids


def _renders_response_formats():
    from mistral_common.protocol.instruct import request

    return hasattr(request, "JsonSchema")  # mistral-common >= 1.13.0


def _mistral_ids(tokenizer, messages, response_format=None, **kwargs):
    """mistral-common encoding the raw request messages."""
    from mistral_common.protocol.instruct.request import ChatCompletionRequest as MistralRequest

    if response_format is not None:
        from mistral_common.protocol.instruct.request import JsonSchema, ResponseFormat

        schema = None
        if response_format == "json_schema":
            schema = JsonSchema(**_CITY_FORMAT["json_schema"])
        kwargs["response_format"] = ResponseFormat(type=response_format, json_schema=schema)
    request = MistralRequest.from_openai(messages=messages, **kwargs)
    return tokenizer.mistral.encode_chat_completion(request).tokens


def _image_url(seed, size):
    pixels = np.random.default_rng(seed).integers(0, 256, size=(*size, 3), dtype=np.uint8)
    buffer = BytesIO()
    Image.fromarray(pixels, "RGB").save(buffer, "PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


_CHAT = [
    {"role": "system", "content": "Be brief."},
    {"role": "user", "content": "Hi there [INST]"},
    {"role": "assistant", "content": "Hello."},
    {"role": "user", "content": "Again."},
]
_PARTS = [
    {"role": "system", "content": [{"type": "text", "text": "Rule one."}] * 2},
    {
        "role": "user",
        "content": [{"type": "text", "text": "First."}, {"type": "text", "text": "Next."}],
    },
]
_TOOL_HISTORY = [
    {"role": "user", "content": "Weather in Paris?"},
    {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": "chatcmpl-tool-0123456789abcdef",
                "type": "function",
                "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'},
            }
        ],
    },
    {"role": "tool", "tool_call_id": "chatcmpl-tool-0123456789abcdef", "content": "Sunny."},
]
_IMAGES = [
    {"role": "system", "content": "Describe images in order."},
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "First "},
            {"type": "image_url", "image_url": {"url": _image_url(1, (48, 64))}},
            {"type": "text", "text": " then "},
            {"type": "image_url", "image_url": {"url": _image_url(2, (96, 40))}},
        ],
    },
]


def _tool_history(arguments):
    """A tool call in the history whose arguments are ``arguments`` as sent."""
    history = copy.deepcopy(_TOOL_HISTORY)
    history[1]["tool_calls"][0]["function"]["arguments"] = arguments
    return history


# Each case: the request fields, and what mistral-common is given besides the messages.
_CASES = {
    "control_tokens": ({"messages": _CHAT}, {}),
    "text_parts": ({"messages": _PARTS}, {}),
    "tools": ({"messages": _TOOL_HISTORY, "tools": [_WEATHER_TOOL]}, {"tools": [_WEATHER_TOOL]}),
    # Arguments that are not a JSON object reach mistral-common as sent, which renders them.
    "unparsable_args": ({"messages": _tool_history("{not json")}, {}),
    "ignored_fields": (
        {
            "messages": _HI,
            "documents": [{"title": "t", "text": "x"}],
            "chat_template_kwargs": {"enable_thinking": True},
        },
        {},
    ),
    "images": ({"messages": _IMAGES}, {}),
}  # fmt: skip


@pytest.mark.parametrize("case", list(_CASES))
def test_chats_render_as_mistral_common(tokenizer, case):
    if case == "images":
        pytest.importorskip("cv2")  # mistral-common's image encoder resizes with OpenCV
    fields, mistral_kwargs = _CASES[case]
    ids = _server_ids(tokenizer, **fields)
    assert ids == _mistral_ids(tokenizer, fields["messages"], **mistral_kwargs)
    assert ids[0] == tokenizer.bos_token_id
    if case == "control_tokens":  # the control token spelled in user text stays text
        assert ids.count(tokenizer.tokenizer.get_control_token("[INST]")) == 2
    if case == "text_parts":  # each part is its own chunk
        joined = [{**m, "content": "\n".join(p["text"] for p in m["content"])} for m in _PARTS]
        assert ids != _mistral_ids(tokenizer, joined)
    if case == "images":
        assert ids.count(tokenizer.tokenizer.get_control_token("[IMG_END]")) == 2


def test_a_response_format_reaches_mistral_common(monkeypatch):
    """A JSON response format is a model setting from mistral-common 1.13.0 on.

    A text format, or any format before 1.13.0 (Python 3.10), only constrains decoding:
    mistral-common keeps its default, text.
    """
    from mistral_common.protocol.instruct import request as mistral_request

    def given(*response_formats):
        recording = _RecordingTokenizer()
        for response_format in response_formats:
            _render(recording, _HI, chat_template_kwargs={"response_format": response_format})
        return [call.response_format for call in recording.calls]

    def kinds(formats):
        return [str(getattr(f.type, "value", f.type)) for f in formats]

    json_formats = (_CITY_FORMAT, {"type": "json_object"})
    assert kinds(given({"type": "text"})) == ["text"]
    if _renders_response_formats():
        schema, json_object = given(*json_formats)
        assert kinds([schema, json_object]) == ["json_schema", "json_object"]
        assert schema.json_schema.name == "city"
    monkeypatch.delattr(mistral_request, "JsonSchema", raising=False)
    assert kinds(given(*json_formats)) == ["text", "text"]


def _declared_efforts(tokenizer):
    builder = getattr(tokenizer.tokenizer, "model_settings_builder", None)
    effort = None if builder is None else builder.reasoning_effort
    return [] if effort is None else [str(getattr(v, "value", v)) for v in effort.values]


def _chat_route(tokenizer, effort):
    return _server_ids(tokenizer, messages=_HI, reasoning_effort=effort)


def _responses_route(tokenizer, effort):
    from tensorrt_llm.serve.openai_protocol import ResponsesRequest
    from tensorrt_llm.serve.responses_utils import (
        reasoning_chat_template_kwargs,
        reasoning_injected_chat_template_keys,
    )

    request = ResponsesRequest(model="m", input="Hi.", reasoning={"effort": effort})
    return _render(
        tokenizer,
        _HI,
        chat_template_kwargs=reasoning_chat_template_kwargs(request, model_type=_MODEL_TYPE),
        injected_chat_template_kwargs=reasoning_injected_chat_template_keys(request),
    )


def _anthropic_route(tokenizer, effort):
    from tensorrt_llm.serve.anthropic_adapter import convert_anthropic_request
    from tensorrt_llm.serve.anthropic_protocol import AnthropicMessagesRequest

    anthropic = AnthropicMessagesRequest(
        model="m", messages=_HI, max_tokens=16, output_config={"effort": effort}
    )
    return _request_ids(tokenizer, convert_anthropic_request(anthropic))


@pytest.mark.parametrize("effort,declared", [("high", True), ("low", False)])
@pytest.mark.parametrize(
    "route",
    [_chat_route, _responses_route, _anthropic_route],
    ids=["chat", "responses", "anthropic"],
)
def test_a_reasoning_effort_is_checked_against_the_checkpoint(tokenizer, route, effort, declared):
    """Every route passes the effort to mistral-common.

    One the checkpoint declares renders; any other is a rejected chat (a ValueError), which
    every route answers with HTTP 400.
    """
    assert (effort in _declared_efforts(tokenizer)) == declared
    if declared:
        assert route(tokenizer, effort) == _mistral_ids(tokenizer, _HI, reasoning_effort=effort)
        return
    with pytest.raises(ChatRejectedError, match="reasoning_effort"):
        route(tokenizer, effort)


def test_a_checkpoint_without_model_settings_refuses_any_reasoning_effort():
    """A v13 Tekken file (Mistral-Large-3, Ministral-3) declares no effort, so it refuses one.

    The served v13 checkpoints have no model settings; this is the one check of that file.
    """
    tokenizer = load_tekken("v13")
    assert _declared_efforts(tokenizer) == []
    with pytest.raises(ChatRejectedError, match="reasoning_effort"):
        _chat_route(tokenizer, "high")


def test_kv_cache_truncation_renders_the_chat_routes_prefix(tokenizer):
    """The message-based KV-cache truncation renders the prefix the cache holds.

    It applies the request's reasoning effort as the chat route does.
    """
    from tensorrt_llm.serve.openai_protocol import KVCacheTruncateRequest
    from tensorrt_llm.serve.resource_governor import _chat_route_request

    effort = "high"
    assert effort in _declared_efforts(tokenizer)
    request = KVCacheTruncateRequest(model="m", messages=_HI, reasoning_effort=effort)
    kwargs = _chat_route_request(request, get_serving_extension(_MODEL_TYPE)).chat_template_kwargs
    assert kwargs == {"reasoning_effort": effort}
    truncated = _server_ids(tokenizer, messages=_HI, chat_template_kwargs=kwargs)
    assert truncated == _server_ids(tokenizer, messages=_HI, reasoning_effort=effort)


def test_a_developer_message_renders_as_a_system_message(tokenizer):
    """mistral-common converts a developer message only from 1.13.1 on."""
    system = [{"role": "system", "content": "Be brief."}, *_HI]
    developer = [{**system[0], "role": "developer"}, *_HI]
    expected = _mistral_ids(tokenizer, system)
    assert _render(tokenizer, developer) == _render(tokenizer, system) == expected


def test_truncation_keeps_the_last_ids_of_a_chat_without_images(tokenizer):
    """The chat route keeps the last truncate_prompt_tokens of the ids it renders.

    The engine truncates a text prompt that way; a truncation that would cut image tokens is
    refused.
    """
    full = _server_ids(tokenizer, messages=_HI)
    assert len(full) > 1
    assert _server_ids(tokenizer, messages=_HI, truncate_prompt_tokens=1) == full[-1:]
    assert _server_ids(tokenizer, messages=_HI, truncate_prompt_tokens=len(full)) == full
    with pytest.raises(ValueError, match="carries media"):
        _server_ids(tokenizer, messages=_IMAGES, truncate_prompt_tokens=1)


@pytest.mark.parametrize(
    "reasoning,kwargs", [({"effort": "none"}, {"thinking": False}), ({}, {"thinking": True})]
)
def test_a_responses_reasoning_toggle_renders_the_checkpoint_default(tokenizer, reasoning, kwargs):
    """A reasoning toggle without a declared effort leaves the checkpoint's default."""
    from tensorrt_llm.serve.openai_protocol import ResponsesRequest
    from tensorrt_llm.serve.responses_utils import (
        reasoning_chat_template_kwargs,
        reasoning_injected_chat_template_keys,
    )

    request = ResponsesRequest(model="m", input="Hi.", reasoning=reasoning)
    assert reasoning_chat_template_kwargs(request, model_type=_MODEL_TYPE) == kwargs
    ids = _render(
        tokenizer,
        _HI,
        chat_template_kwargs=kwargs,
        injected_chat_template_kwargs=reasoning_injected_chat_template_keys(request),
    )
    assert ids == _mistral_ids(tokenizer, _HI)


def _user_with_image(size):
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe."},
                {"type": "image_url", "image_url": {"url": _image_url(3, size)}},
            ],
        }
    ]


# (height, width): a one-patch grid, and a non-square one larger than max_image_size. How
# many patches a size makes is mistral-common's; the renderer only lays out the grid.
@pytest.mark.parametrize("size", [(1, 1), (24, 2000)])
def test_image_grids_render_from_the_image_size_alone(tokenizer, size, monkeypatch):
    """A chat renders the grids mistral-common makes of the images, processing no pixel."""
    pytest.importorskip("cv2")  # the oracle resizes with OpenCV
    from mistral_common.tokens.tokenizers import image as mistral_image

    messages = _user_with_image(size)
    expected = _mistral_ids(tokenizer, messages)

    def no_pixel_work(*args, **kwargs):
        raise AssertionError("rendering processed or serialized pixels")

    monkeypatch.setattr(mistral_image, "transform_image", no_pixel_work)
    monkeypatch.setattr(Image.Image, "save", no_pixel_work)
    assert _server_ids(tokenizer, messages=messages) == expected
