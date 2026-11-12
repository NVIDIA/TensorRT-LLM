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

The Mistral-native format (``checkpoint_format`` ``mistral`` or ``mistral_large_3``) gives
the model type ``mistral_common``; mistral-common, not a Jinja template, renders its chats.
The serving rules:

- the top-level ``reasoning_effort`` selects the reasoning mode; a ``reasoning_effort`` in
  ``chat_template_kwargs`` is accepted too and must agree with it; a checkpoint that
  declares no efforts rejects one;
- a request ``chat_template`` is rejected; ``documents`` and the ``chat_template_kwargs``
  keys the encoder does not read are ignored;
- ``add_generation_prompt`` and ``continue_final_message`` must agree with the final message;
- tools reach the encoder without a null ``strict``, and each text part of a content list
  reaches it as its own chunk;
- a ``json_object`` or named ``json_schema`` response format reaches mistral-common, which
  states it in the prompt's model settings where the tokenizer file declares them;
- images are rendered from the loaded media, on every route that renders a chat.

The request-rule tests run without a checkpoint. The rendering tests use the tekken.json of
Mistral-Large-3 (v13) and Mistral-Small-4 (v15) under LLM_MODELS_ROOT, and compare the
server's rendering steps with mistral-common encoding the raw request messages.
"""

from __future__ import annotations

import asyncio
import base64
import copy
from io import BytesIO
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image
from utils.llm_data import llm_models_root

from tensorrt_llm.inputs.utils import NATIVE_MISTRAL_MODEL_TYPES, apply_chat_template
from tensorrt_llm.serve.chat_utils import parse_chat_messages_coroutines
from tensorrt_llm.serve.openai_protocol import ChatCompletionRequest
from tensorrt_llm.serve.render import render_conversation, render_conversation_with_media
from tensorrt_llm.serve.serving_extensions import apply_model_chat_extensions, get_serving_extension

pytestmark = pytest.mark.cpu_only

_MODEL_TYPE = "mistral_common"
_TEKKEN_FILES = {
    "v13": "Mistral-Large-3-675B/Mistral-Large-3-675B-Instruct-2512/tekken.json",
    "v15": "Mistral-Small-4-119B-2603/Mistral-Small-4-119B-2603/tekken.json",
}
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


class _RecordingTokenizer:
    """Stands in for the native tokenizer and records each render it is asked for."""

    def __init__(self):
        self.calls = []

    def apply_chat_template(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        return [1]


def _render(tokenizer, messages, *, add_generation_prompt=True, **kwargs):
    return apply_chat_template(
        model_type=_MODEL_TYPE,
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


def _config():
    return type("Config", (), {"model_type": _MODEL_TYPE})()


def test_the_native_format_renders_with_mistral_common():
    assert _MODEL_TYPE in NATIVE_MISTRAL_MODEL_TYPES
    assert get_serving_extension(_MODEL_TYPE).render_needs_media()
    assert not get_serving_extension("llama").render_needs_media()


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


def test_the_native_renderer_rejects_an_hf_tokenizer():
    from tensorrt_llm.tokenizer import TransformersTokenizer

    with pytest.raises(ValueError, match="mistral-common's MistralTokenizer"):
        _render(TransformersTokenizer(SimpleNamespace()), [{"role": "user", "content": "Hi."}])


def test_a_chat_rendered_without_its_loaded_images_is_refused():
    """A route that renders without loading the media gets a clear error, not an IndexError."""
    tokenizer = _RecordingTokenizer()
    with pytest.raises(ValueError, match="1 images, but 0 were loaded"):
        apply_chat_template(
            model_type=_MODEL_TYPE,
            tokenizer=tokenizer,
            processor=None,
            conversation=[{"role": "user", "content": "What is this?", "content_parts": []}],
            add_generation_prompt=True,
            mm_placeholder_counts=[{"[IMG]": 1}],
        )
    assert tokenizer.calls == []


@pytest.mark.parametrize("effort", ["low", "high"])
def test_top_level_reasoning_effort_reaches_the_renderer(effort):
    request = _request(reasoning_effort=effort, chat_template_kwargs={"x": 1})
    apply_model_chat_extensions(request, _MODEL_TYPE)
    assert request.chat_template_kwargs == {"x": 1, "reasoning_effort": effort}


def test_an_unset_reasoning_effort_derives_nothing():
    """The protocol's own default is not a mistral-common effort and must not leak in."""
    request = _request()
    apply_model_chat_extensions(request, _MODEL_TYPE)
    assert request.chat_template_kwargs is None


def test_reasoning_effort_in_kwargs_is_kept_and_must_agree():
    request = _request(chat_template_kwargs={"reasoning_effort": "low", "x": 1})
    apply_model_chat_extensions(request, _MODEL_TYPE)
    assert request.chat_template_kwargs == {"reasoning_effort": "low", "x": 1}

    request = _request(reasoning_effort="high", chat_template_kwargs={"reasoning_effort": "high"})
    apply_model_chat_extensions(request, _MODEL_TYPE)
    assert request.chat_template_kwargs == {"reasoning_effort": "high"}

    with pytest.raises(ValueError, match="conflicts"):
        apply_model_chat_extensions(
            _request(reasoning_effort="high", chat_template_kwargs={"reasoning_effort": "low"}),
            _MODEL_TYPE,
        )


def test_a_request_chat_template_is_rejected():
    with pytest.raises(ValueError, match="chat_template is not supported"):
        apply_model_chat_extensions(_request(chat_template="{{ messages }}"), _MODEL_TYPE)


@pytest.mark.parametrize(
    "response_format,passed",
    [(_CITY_FORMAT, True), ({"type": "json_object"}, True), ({"type": "text"}, False)],
)
def test_a_response_format_reaches_the_renderer(response_format, passed):
    request = _request(
        response_format=response_format,
        chat_template_kwargs={"response_format": {"type": "json_object"}},
    )
    apply_model_chat_extensions(request, _MODEL_TYPE)
    expected = request.response_format.model_dump(exclude_none=True) if passed else None
    assert (request.chat_template_kwargs or {}).get("response_format") == expected


@pytest.mark.parametrize("model_type,kept", [(_MODEL_TYPE, True), ("llama", False)])
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


@pytest.mark.parametrize("model_type,coerced", [(_MODEL_TYPE, True), ("llama", False)])
def test_a_history_tool_call_with_invalid_arguments(model_type, coerced):
    messages = [
        {"role": "user", "content": "Weather?"},
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call0",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": "{not json"},
                }
            ],
        },
    ]
    config = type("Config", (), {"model_type": model_type})()
    if not coerced:
        with pytest.raises(ValueError, match="must be valid JSON"):
            parse_chat_messages_coroutines(messages, config)
        return
    conversation, media, _, _ = parse_chat_messages_coroutines(messages, config)
    asyncio.run(media)
    assert conversation[1]["tool_calls"][0]["function"]["arguments"] == {}


@pytest.fixture(scope="module", params=sorted(_TEKKEN_FILES))
def tokenizer(request):
    from tensorrt_llm._torch.models.checkpoints.mistral.tokenizer import MistralTokenizer

    root = llm_models_root()
    path = None if root is None else root / _TEKKEN_FILES[request.param]
    if path is None or not path.is_file():
        pytest.skip(f"needs {_TEKKEN_FILES[request.param]} under LLM_MODELS_ROOT")
    return MistralTokenizer.from_pretrained(str(path))


def _resources(tokenizer):
    return SimpleNamespace(
        model_type=_MODEL_TYPE,
        tokenizer=tokenizer,
        processor=None,
        default_chat_template=None,
        extension=get_serving_extension(_MODEL_TYPE),
    )


def _server_ids(tokenizer, **fields):
    """The openai_chat rendering steps for one Mistral-native request."""
    request = _request(**copy.deepcopy(fields))
    apply_model_chat_extensions(request, _MODEL_TYPE)
    conversation, media, counts, _ = parse_chat_messages_coroutines(request.messages, _config())
    tools = None if request.tools is None else [tool.model_dump() for tool in request.tools]
    rendered, _ = asyncio.run(
        render_conversation_with_media(
            _resources(tokenizer),
            media,
            conversation=conversation,
            mm_placeholder_counts=counts,
            add_generation_prompt=request.add_generation_prompt,
            tools=tools,
            documents=request.documents,
            chat_template=request.chat_template,
            chat_template_kwargs=request.chat_template_kwargs or {},
        )
    )
    assert rendered.text is None
    return rendered.token_ids


def _raw_ids(tokenizer, messages, **kwargs):
    """mistral-common encoding the raw request messages."""
    return tokenizer.apply_chat_template(messages, tokenize=True, return_dict=False, **kwargs)


def _direct_ids(tokenizer, messages, **kwargs):
    """mistral-common encoding the raw request messages with request-level settings."""
    from mistral_common.protocol.instruct.request import ChatCompletionRequest as MistralRequest

    request = MistralRequest.from_openai(messages=messages, **kwargs)
    return tokenizer.mistral.encode_chat_completion(request).tokens


def _image_url(seed, size):
    pixels = np.random.default_rng(seed).integers(0, 256, size=(*size, 3), dtype=np.uint8)
    buffer = BytesIO()
    Image.fromarray(pixels, "RGB").save(buffer, "PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


def test_a_chat_renders_as_mistral_common_with_its_control_tokens(tokenizer):
    messages = [
        {"role": "system", "content": "Be brief."},
        {"role": "user", "content": "Hi there [INST]"},
        {"role": "assistant", "content": "Hello."},
        {"role": "user", "content": "Again."},
    ]
    ids = _server_ids(tokenizer, messages=messages)
    assert ids == _raw_ids(tokenizer, messages)
    assert ids[0] == tokenizer.bos_token_id
    # The control token spelled in user text stays text.
    inst = tokenizer.tokenizer.get_control_token("[INST]")
    assert ids.count(inst) == 2


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


def test_tools_and_their_history_render_as_sent(tokenizer):
    messages = [
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
    ids = _server_ids(tokenizer, messages=messages, tools=[_WEATHER_TOOL])
    assert ids == _raw_ids(tokenizer, messages, tools=[_WEATHER_TOOL])
    assert ids != _raw_ids(tokenizer, messages)


def test_ignored_fields_leave_the_tokens_unchanged(tokenizer):
    messages = [{"role": "user", "content": "Summarize."}]
    ids = _server_ids(
        tokenizer,
        messages=messages,
        documents=[{"title": "t", "text": "Some text."}],
        chat_template_kwargs={"enable_thinking": True},
    )
    assert ids == _raw_ids(tokenizer, messages)


def test_reasoning_effort_renders_where_the_checkpoint_declares_it(tokenizer):
    messages = [{"role": "user", "content": "Think."}]
    if not tokenizer.tokenizer.version.supports_model_settings:
        # openai_chat returns HTTP 400 for a ValueError raised while rendering.
        with pytest.raises(ValueError, match="reasoning_effort"):
            _server_ids(tokenizer, messages=messages, reasoning_effort="high")
        return
    high = _server_ids(tokenizer, messages=messages, reasoning_effort="high")
    assert high == _raw_ids(tokenizer, messages, reasoning_effort="high")
    assert high != _raw_ids(tokenizer, messages, reasoning_effort="none")
    assert _server_ids(tokenizer, messages=messages) == _raw_ids(tokenizer, messages)
    with pytest.raises(ValueError, match="reasoning_effort"):
        _server_ids(
            tokenizer, messages=messages, chat_template_kwargs={"reasoning_effort": "turbo"}
        )


def test_a_response_format_renders_as_mistral_common_renders_it(tokenizer):
    from mistral_common.protocol.instruct.request import JsonSchema, ResponseFormat, ResponseFormats

    messages = [{"role": "user", "content": "Give a city as JSON."}]
    named = _CITY_FORMAT["json_schema"]
    schema = ResponseFormat(
        type=ResponseFormats.json_schema,
        json_schema=JsonSchema(name=named["name"], schema=named["schema"], strict=True),
    )
    ids = _server_ids(tokenizer, messages=messages, response_format=_CITY_FORMAT)
    assert ids == _direct_ids(tokenizer, messages, response_format=schema)
    assert _server_ids(
        tokenizer, messages=messages, response_format={"type": "json_object"}
    ) == _direct_ids(
        tokenizer, messages, response_format=ResponseFormat(type=ResponseFormats.json_object)
    )
    text = _server_ids(tokenizer, messages=messages, response_format={"type": "text"})
    assert text == _raw_ids(tokenizer, messages)


def test_images_render_in_place_from_the_loaded_media(tokenizer):
    pytest.importorskip("cv2")  # mistral-common's image encoder resizes with OpenCV
    first, second = _image_url(1, (48, 64)), _image_url(2, (96, 40))
    messages = [
        {"role": "system", "content": "Describe images in order."},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "First "},
                {"type": "image_url", "image_url": {"url": first}},
                {"type": "text", "text": " then "},
                {"type": "image_url", "image_url": {"url": second}},
                {"type": "text", "text": "; compare them."},
            ],
        },
    ]
    ids = _server_ids(tokenizer, messages=messages)
    assert ids == _raw_ids(tokenizer, messages, return_tensors=None)
    image_end = tokenizer.tokenizer.get_control_token("[IMG_END]")
    assert ids.count(image_end) == 2


def test_the_rendering_routes_load_images_before_a_native_render(tokenizer):
    """Responses and the KV-cache governor render through the same helper as the chat route."""
    pytest.importorskip("cv2")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What is this?"},
                {"type": "image_url", "image_url": {"url": _image_url(3, (48, 64))}},
            ],
        }
    ]
    conversation, media, counts, _ = parse_chat_messages_coroutines(messages, _config())
    rendered, (mm_data, mm_embeddings) = asyncio.run(
        render_conversation_with_media(
            _resources(tokenizer),
            media,
            conversation=conversation,
            mm_placeholder_counts=counts,
            add_generation_prompt=True,
            tokenize=True,
        )
    )
    assert rendered.token_ids == _raw_ids(tokenizer, messages, return_tensors=None)
    assert len(mm_data["image"]) == 1 and mm_embeddings is None

    conversation, media, counts, _ = parse_chat_messages_coroutines(messages, _config())
    with pytest.raises(ValueError, match="were loaded for rendering"):
        asyncio.run(
            render_conversation(
                _resources(tokenizer),
                conversation=conversation,
                mm_placeholder_counts=counts,
                add_generation_prompt=True,
            )
        )
    asyncio.run(media)
