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
"""Offline tests for Responses API input preprocessing.

The Responses API accepts either a plain string or a list of structured input
items. Clients that send structured items - Codex CLI and the OpenAI SDK among
them - carry the role on each item, and losing it silently turns the caller's
question into an assistant turn.
"""

import pytest

from tensorrt_llm.inputs.utils import MultimodalDataTracker
from tensorrt_llm.serve.chat_utils import parse_chat_message_content
from tensorrt_llm.serve.openai_protocol import ResponsesRequest
from tensorrt_llm.serve.responses_utils import (
    _create_input_messages,
    _response_output_item_to_chat_completion_message,
)

# The CPU-* CI stages run pytest with -m 'cpu_only'. Without this marker every
# test in the file is deselected, which pytest reports as exit code 5 and the
# stage reports as a failure.
pytestmark = pytest.mark.cpu_only


def _message_item(role, *texts, item_id=None):
    item = {
        "type": "message",
        "role": role,
        "content": [{"type": "input_text", "text": t} for t in texts],
    }
    if item_id is not None:
        item["id"] = item_id
    return item


# ---------------------------------------------------------------------------
# Per-item conversion
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("role", ["user", "assistant", "system", "developer"])
def test_item_role_is_preserved(role):
    """Regression: the role was hardcoded to "assistant".

    With a generation prompt appended, a user question converted to an
    assistant message asks the model to continue its own turn, which produces
    fabricated context and leaked chat-template markup instead of an answer.
    """
    msg = _response_output_item_to_chat_completion_message(_message_item(role, "what is 17*23?"))
    assert msg["role"] == role
    assert msg["content"] == "what is 17*23?"


def test_all_content_parts_are_kept():
    """Regression: only content[0] survived."""
    msg = _response_output_item_to_chat_completion_message(
        _message_item("user", "first ", "second ", "third")
    )
    assert msg["content"] == "first second third"


def test_reasoning_item_is_always_assistant():
    msg = _response_output_item_to_chat_completion_message(
        {
            "type": "reasoning",
            "content": [{"type": "reasoning_text", "text": "thinking"}],
        }
    )
    assert msg == {"role": "assistant", "reasoning": "thinking"}


def test_role_defaults_to_assistant_when_absent():
    msg = _response_output_item_to_chat_completion_message(
        {
            "type": "message",
            "content": [{"type": "output_text", "text": "hi", "annotations": []}],
        }
    )
    assert msg["role"] == "assistant"


def test_empty_content_is_rejected():
    with pytest.raises(ValueError, match="empty or missing"):
        _response_output_item_to_chat_completion_message(
            {"type": "message", "role": "user", "content": []}
        )


def test_function_call_output_keeps_call_id():
    msg = _response_output_item_to_chat_completion_message(
        {
            "type": "function_call_output",
            "call_id": "call_1",
            "output": "42",
        }
    )
    assert msg == {"role": "tool", "content": "42", "tool_call_id": "call_1"}


# ---------------------------------------------------------------------------
# Whole-request conversion
# ---------------------------------------------------------------------------


def _messages(request_kwargs):
    request = ResponsesRequest(model="m", **request_kwargs)
    import asyncio

    return asyncio.run(_create_input_messages(request=request, prev_msgs=[]))


def test_string_input_becomes_a_user_message():
    assert _messages({"input": "hello"}) == [{"role": "user", "content": "hello"}]


def test_structured_input_round_trips_roles():
    """The shape Codex CLI sends: a list of message items carrying roles."""
    messages = _messages(
        {
            "instructions": "You are a helpful agent.",
            "input": [
                _message_item("user", "what is 17*23?", item_id="msg_1"),
                _message_item("assistant", "391"),
                _message_item("user", "and 2*2?"),
            ],
        }
    )
    assert [m["role"] for m in messages] == ["system", "user", "assistant", "user"]
    assert messages[0]["content"] == "You are a helpful agent."
    assert messages[1]["content"] == "what is 17*23?"
    assert messages[-1]["content"] == "and 2*2?"


def test_last_message_is_from_the_user():
    """The property that actually matters for prompt construction.

    A generation prompt is appended after these messages, so the final turn
    has to be the user's. Before the fix it was always the assistant's.
    """
    messages = _messages({"input": [_message_item("user", "ping")]})
    assert messages[-1]["role"] == "user"


def test_per_item_id_is_tolerated():
    """Clients echo items back with the id the server assigned."""
    messages = _messages({"input": [_message_item("user", "ping", item_id="msg_9")]})
    assert messages[-1] == {"role": "user", "content": "ping"}


def test_assistant_item_keeps_its_id():
    """Regression: stripping id from assistant turns broke multi-turn.

    An assistant message maps to ResponseOutputMessageParam, which requires
    both id and status. Stripping id there leaves the item matching no
    variant of the input union, so the request 422s as soon as the
    conversation contains one assistant turn - i.e. from the second reply on.
    """
    request = ResponsesRequest(
        model="m",
        input=[
            _message_item("user", "q1", item_id="msg_u"),
            {
                "id": "msg_a",
                "status": "completed",
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "a1", "annotations": []}],
            },
            _message_item("user", "q2"),
        ],
    )
    items = request.input
    assistant = items[1]
    assistant = assistant if isinstance(assistant, dict) else assistant.model_dump()
    assert assistant.get("id") == "msg_a", "assistant id must survive"
    user = items[0] if isinstance(items[0], dict) else items[0].model_dump()
    assert "id" not in user, "user id is forbidden by EasyInputMessageParam"


def test_multi_turn_conversation_round_trips():
    """Three turns, the shape a client sends on its third request."""
    messages = _messages(
        {
            "input": [
                _message_item("user", "q1"),
                {
                    "id": "m1",
                    "status": "completed",
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "a1", "annotations": []}],
                },
                _message_item("user", "q2"),
                {
                    "id": "m2",
                    "status": "completed",
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "a2", "annotations": []}],
                },
                _message_item("user", "q3"),
            ],
        }
    )
    assert [m["role"] for m in messages] == ["user", "assistant", "user", "assistant", "user"]
    assert messages[-1]["content"] == "q3"


def test_structured_input_request_is_picklable():
    """Regression: lazily-validated sequences broke postprocess workers.

    Several vendored item types declare sequence fields as Iterable[...], and
    pydantic validates those lazily into a ValidatorIterator. The request is
    pickled when handed to a postprocess worker, so a structured-input request
    failed with "cannot pickle ValidatorIterator" - and the iterator is also
    single-consumption. The lazy field here is nested at
    input[N].content[0].annotations, so a shallow walk does not catch it.
    """
    import pickle

    request = ResponsesRequest(
        model="m",
        input=[
            _message_item("user", "q1"),
            {
                "id": "m1",
                "status": "completed",
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "a1", "annotations": []}],
            },
            _message_item("user", "q2"),
        ],
    )
    pickle.dumps(request)

    def has_lazy(obj, depth=0):
        if depth > 8:
            return False
        if type(obj).__name__ == "ValidatorIterator":
            return True
        if isinstance(obj, dict):
            return any(has_lazy(v, depth + 1) for v in obj.values())
        if isinstance(obj, list):
            return any(has_lazy(v, depth + 1) for v in obj)
        return False

    assert not has_lazy(request.input)


def test_unknown_top_level_fields_are_tolerated():
    """Codex attaches client_metadata and prompt_cache_key."""
    request = ResponsesRequest(
        model="m",
        input="hi",
        client_metadata={"session_id": "s"},
        prompt_cache_key="k",
    )
    assert request.input == "hi"


# ---------------------------------------------------------------------------
# Explicit null means "unset" (litellm-style clients)
# ---------------------------------------------------------------------------
#
# Item shapes below mirror live traffic from clients that serialize every
# unset optional as null (content is synthetic).


def _echoed_assistant_message_with_nulls():
    return {
        "id": "msg_0123456789abcdef",
        "type": "message",
        "role": "assistant",
        "status": "completed",
        "phase": None,
        "content": [
            {
                "type": "output_text",
                "text": "checking the log format first.",
                "annotations": [],
                "logprobs": None,
            }
        ],
    }


def _echoed_function_call_with_nulls():
    return {
        "type": "function_call",
        "id": "fc_0123456789abcdef",
        "call_id": "call_0123456789abcdef",
        "name": "bash",
        "arguments": '{"command": "ls /app"}',
        "caller": None,
        "namespace": None,
        "status": None,
        "async_": None,
    }


def test_null_logprobs_on_echoed_output_text_is_unset():
    """Regression: null logprobs on an echoed output_text part failed validation."""
    request = ResponsesRequest(
        model="m",
        input=[
            _message_item("user", "q1"),
            _echoed_assistant_message_with_nulls(),
            _message_item("user", "q2"),
        ],
    )
    assert len(request.input) == 3


def test_null_optionals_on_echoed_function_call_are_unset():
    request = ResponsesRequest(
        model="m",
        input=[
            _message_item("user", "q"),
            _echoed_function_call_with_nulls(),
            {
                "type": "function_call_output",
                "call_id": "call_0123456789abcdef",
                "output": "app.py",
            },
        ],
    )
    call = request.input[1]
    call_id = call["call_id"] if isinstance(call, dict) else call.call_id
    assert call_id == "call_0123456789abcdef"


def test_full_multi_turn_echo_with_nulls_round_trips():
    """System + user + assistant echo + reasoning + function_call + output."""
    request = ResponsesRequest(
        model="m",
        input=[
            {"role": "system", "content": "You are a helpful assistant."},
            _message_item("user", "find the last date on each line"),
            _echoed_assistant_message_with_nulls(),
            {
                "id": "rs_0123456789abcdef",
                "type": "reasoning",
                "summary": [],
                "content": [{"type": "reasoning_text", "text": "Need to check the file."}],
            },
            _echoed_function_call_with_nulls(),
            {
                "type": "function_call_output",
                "call_id": "call_0123456789abcdef",
                "output": '{"returncode": 0}',
            },
        ],
    )
    assert len(request.input) == 6


def test_null_top_level_optionals_are_unset():
    """tool_choice/metadata/temperature: null must mean "use the default"."""
    request = ResponsesRequest(
        model="m",
        input="hi",
        tool_choice=None,
        metadata=None,
        temperature=None,
        service_tier=None,
        truncation=None,
    )
    assert request.tool_choice == "auto"
    assert request.metadata is None
    assert request.service_tier == "auto"
    assert request.truncation == "disabled"


def test_meaningful_values_survive_the_null_scrub():
    request = ResponsesRequest(
        model="m",
        input=[_message_item("user", "q")],
        temperature=0.25,
        tool_choice="none",
    )
    assert request.temperature == 0.25
    assert request.tool_choice == "none"


# ---------------------------------------------------------------------------
# Multimodal content
# ---------------------------------------------------------------------------

_IMAGE_URL = "data:image/png;base64,iVBORw0KGgo="
_QUESTION = {"type": "input_text", "text": "What is in this image?"}


def _user_item(*parts, typed=True):
    item = {"role": "user", "content": list(parts)}
    if typed:
        item["type"] = "message"
    return item


@pytest.mark.parametrize("typed", [True, False], ids=["message", "untyped"])
def test_input_image_becomes_a_chat_image_part(typed):
    """The image has to reach the chat parser, where the client placed it.

    Keeping only the text parts left the model answering about an image it
    never received.
    """
    image = {"type": "input_image", "image_url": _IMAGE_URL, "detail": "auto"}
    msg = _response_output_item_to_chat_completion_message(
        _user_item(_QUESTION, image, typed=typed)
    )
    assert msg == {
        "role": "user",
        "content": [
            {"type": "text", "text": "What is in this image?"},
            {"type": "image_url", "image_url": {"url": _IMAGE_URL, "detail": "auto"}},
        ],
    }


def _chat_text(message):
    """The prompt text the chat parser reads from a converted message."""
    return parse_chat_message_content(message, MultimodalDataTracker("llama"))["content"]


def test_untyped_input_text_becomes_chat_text_parts():
    """input_text parts are not chat parts; passed through, the chat parser rejects them."""
    msg = _response_output_item_to_chat_completion_message(
        _user_item(
            {"type": "input_text", "text": "hello"},
            {"type": "input_text", "text": "there"},
            typed=False,
        )
    )
    assert msg == {
        "role": "user",
        "content": [{"type": "text", "text": "hello"}, {"type": "text", "text": "there"}],
    }
    assert _chat_text(msg) == "hello\nthere"


@pytest.mark.parametrize(
    "content",
    [
        [{"type": "text", "text": "First"}, {"type": "text", "text": "Second"}],
        ["First", "Second"],
    ],
    ids=["text-parts", "string-parts"],
)
def test_untyped_chat_content_reads_as_in_a_chat_request(content):
    """Chat-style parts are left for the chat parser, which joins them by line.

    Converting them as Responses parts glued the text parts together and
    dropped the plain strings, which changed or erased the prompt.
    """
    item = {"role": "user", "content": content}
    msg = _response_output_item_to_chat_completion_message(item)
    assert msg == item
    assert _chat_text(msg) == "First\nSecond"
    (message,) = _messages({"input": [item]})
    assert _chat_text(message) == "First\nSecond"


def test_typed_message_keeps_plain_string_parts():
    """A plain string is a text part; skipping it erased the prompt without an error."""
    item = _user_item("first ", {"type": "input_text", "text": "second"})
    assert _response_output_item_to_chat_completion_message(item) == {
        "role": "user",
        "content": "first second",
    }
    assert _messages({"input": [item]}) == [{"role": "user", "content": "first second"}]


def test_typed_message_keeps_plain_string_parts_beside_media():
    image = {"type": "input_image", "image_url": _IMAGE_URL}
    item = _user_item("Describe it.", image)
    expected = [
        {"type": "text", "text": "Describe it."},
        {"type": "image_url", "image_url": {"url": _IMAGE_URL}},
    ]
    assert _response_output_item_to_chat_completion_message(item)["content"] == expected
    assert _messages({"input": [item]}) == [{"role": "user", "content": expected}]


@pytest.mark.parametrize(
    "part", [42, {"type": "input_video", "video_id": "v-1"}], ids=["non-dict", "unknown-dict"]
)
def test_typed_message_rejects_a_part_without_text(part):
    with pytest.raises(ValueError, match="Unsupported message content part"):
        _response_output_item_to_chat_completion_message(_user_item(_QUESTION, part))


def test_replayed_refusal_is_skipped():
    """A refusal in a replayed assistant turn has no text for the model, and must not fail the request."""
    item = {
        "type": "message",
        "role": "assistant",
        "content": [{"type": "refusal", "refusal": "I can't help with that."}],
    }
    assert _response_output_item_to_chat_completion_message(item) == {
        "role": "assistant",
        "content": "",
    }


def test_input_image_accepts_the_chat_url_object():
    image = {"type": "input_image", "image_url": {"url": _IMAGE_URL}}
    msg = _response_output_item_to_chat_completion_message(_user_item(_QUESTION, image))
    assert msg["content"][1] == {"type": "image_url", "image_url": {"url": _IMAGE_URL}}


def test_chat_style_media_parts_pass_through():
    image = {"type": "image_url", "image_url": {"url": _IMAGE_URL}}
    msg = _response_output_item_to_chat_completion_message(
        _user_item({"type": "text", "text": "Describe it."}, image, typed=False)
    )
    assert msg["content"] == [{"type": "text", "text": "Describe it."}, image]


def test_input_image_without_a_url_is_rejected():
    """A file_id cannot be resolved here, and dropping the image is worse than failing."""
    image = {"type": "input_image", "file_id": "file-123", "detail": "auto"}
    with pytest.raises(ValueError, match="file_id"):
        _response_output_item_to_chat_completion_message(_user_item(_QUESTION, image))


def test_input_file_is_rejected():
    document = {"type": "input_file", "filename": "spec.pdf", "file_data": "JVBERi0="}
    with pytest.raises(ValueError, match="input_file"):
        _response_output_item_to_chat_completion_message(_user_item(_QUESTION, document))


@pytest.mark.parametrize("image_part", [{"detail": "auto"}, {}], ids=["detail", "no-detail"])
def test_image_request_reaches_the_chat_messages(image_part):
    """The OpenAI SDK sends an untyped user item; the image must survive validation."""
    image = {"type": "input_image", "image_url": _IMAGE_URL, **image_part}
    messages = _messages({"input": [_user_item(_QUESTION, image, typed=False)]})
    assert messages == [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What is in this image?"},
                {"type": "image_url", "image_url": {"url": _IMAGE_URL, **image_part}},
            ],
        }
    ]
