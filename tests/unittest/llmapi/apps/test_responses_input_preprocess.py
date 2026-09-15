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


@pytest.mark.parametrize("role", ["user", "assistant", "system", "developer"])
@pytest.mark.parametrize("content", ["APPLE", "", "  hello\n世界  "])
def test_string_message_content_is_preserved(role: str, content: str) -> None:
    item = {"type": "message", "role": role, "content": content}
    assert _response_output_item_to_chat_completion_message(item) == {
        "role": role,
        "content": content,
    }


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


@pytest.mark.parametrize("typed", [False, True])
def test_request_preserves_string_message(typed: bool) -> None:
    item = {"role": "user", "content": "APPLE"}
    if typed:
        item["type"] = "message"
    assert _messages({"input": [item]}) == [{"role": "user", "content": "APPLE"}]


@pytest.mark.parametrize("logprobs", [None, []])
def test_returned_output_text_can_be_replayed(logprobs: list[object] | None) -> None:
    item = {
        "type": "message",
        "id": "msg_1",
        "role": "assistant",
        "status": "completed",
        "content": [
            {"type": "output_text", "text": "APPLE", "annotations": [], "logprobs": logprobs}
        ],
    }
    assert _messages({"input": [item]}) == [{"role": "assistant", "content": "APPLE"}]
    # Normalization must not mutate the caller's response object.
    assert item["content"][0]["logprobs"] is logprobs


@pytest.mark.parametrize("harmony", [False, True])
@pytest.mark.parametrize("text", ["APPLE", ""])
def test_bare_output_text_replays_as_assistant(text: str, harmony: bool) -> None:
    """Preserve assistant text and turn order through both prompt builders."""
    import asyncio

    from tensorrt_llm.serve.responses_utils import _construct_harmony_messages

    request = ResponsesRequest(
        model="m",
        input=[
            {"role": "user", "content": "Remember APPLE"},
            {"type": "output_text", "text": text, "logprobs": None},
            {"role": "user", "content": "Continue"},
        ],
    )
    if harmony:
        messages = _construct_harmony_messages(request, None)[2:]
        assert [message.author.role for message in messages] == ["user", "assistant", "user"]
        assert [message.content[0].text for message in messages] == [
            "Remember APPLE",
            text,
            "Continue",
        ]
    else:
        assert asyncio.run(_create_input_messages(request, [])) == [
            {"role": "user", "content": "Remember APPLE"},
            {"role": "assistant", "content": text},
            {"role": "user", "content": "Continue"},
        ]


def test_reasoning_summary_and_empty_reasoning_can_be_replayed() -> None:
    assert _response_output_item_to_chat_completion_message(
        {
            "type": "reasoning",
            "summary": [{"type": "summary_text", "text": "Thinking"}],
        }
    ) == {"role": "assistant", "reasoning": "Thinking"}
    assert (
        _response_output_item_to_chat_completion_message(
            {
                "type": "reasoning",
                "summary": [],
                "content": None,
            }
        )
        is None
    )


def test_replay_preserves_populated_logprobs() -> None:
    """Accept null logprobs without discarding a client's actual probabilities."""
    logprobs = [{"token": "APPLE", "logprob": -0.5, "bytes": [65], "top_logprobs": []}]
    request = ResponsesRequest(
        model="m",
        input=[
            {
                "type": "message",
                "role": "assistant",
                "id": "msg_1",
                "status": "completed",
                "content": [
                    {
                        "type": "output_text",
                        "text": "APPLE",
                        "annotations": [],
                        "logprobs": logprobs,
                    }
                ],
            }
        ],
    )
    assert request.model_dump()["input"][0]["content"][0]["logprobs"] == logprobs


@pytest.mark.parametrize("harmony", [False, True])
@pytest.mark.parametrize(
    "reasoning,expected",
    [
        ({"summary": [], "content": None}, ""),
        ({"summary": [{"type": "summary_text", "text": "Summary"}]}, "Summary"),
        (
            {
                "summary": [],
                "content": [
                    {"type": "reasoning_text", "text": "First "},
                    {"type": "reasoning_text", "text": "second"},
                ],
            },
            "First second",
        ),
        (
            {
                "summary": [{"type": "summary_text", "text": "Summary"}],
                "content": [{"type": "reasoning_text", "text": "Full reasoning"}],
            },
            "Full reasoning",
        ),
    ],
)
def test_validated_reasoning_replay(
    reasoning: dict[str, list[dict[str, str]] | None], expected: str, harmony: bool
) -> None:
    """Exercise request validation and conversion on both prompt-building paths."""
    import asyncio

    from tensorrt_llm.serve.responses_utils import _construct_harmony_messages

    request = ResponsesRequest(
        model="m",
        input=[
            {"type": "reasoning", "id": "rs_1", **reasoning},
            {"role": "user", "content": "Continue"},
        ],
    )
    if harmony:
        messages = _construct_harmony_messages(request, None)[2:]
        assert messages[-1].author.role == "user"
        assert messages[-1].content[0].text == "Continue"
        if expected:
            assert messages[0].author.role == "assistant"
            assert messages[0].channel == "analysis"
            assert messages[0].content[0].text == expected
    else:
        messages = asyncio.run(_create_input_messages(request, []))
        assert messages[-1] == {"role": "user", "content": "Continue"}
        if expected:
            assert messages[0] == {"role": "assistant", "reasoning": expected}
    assert len(messages) == (2 if expected else 1)


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
