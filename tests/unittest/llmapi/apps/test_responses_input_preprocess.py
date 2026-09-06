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


def test_reasoning_without_content_falls_back_to_the_summary():
    """The text is in `summary` whenever a summary was requested.

    Reading only `content` threw the summary away and then rejected the item
    for being empty, so a client that asked for reasoning summaries lost both
    the reasoning and the turn.
    """
    msg = _response_output_item_to_chat_completion_message(
        {
            "type": "reasoning",
            "id": "rs_1",
            "summary": [{"type": "summary_text", "text": "weighed two options"}],
        }
    )

    assert msg == {"role": "assistant", "reasoning": "weighed two options"}


def test_reasoning_with_nothing_readable_is_skipped_not_rejected():
    """A shape OpenAI emits, and one a client replays back verbatim.

    With `encrypted_content` and no summary there is no reasoning text in the
    payload to preserve. Raising here does not save anything -- the text was
    already absent -- it just fails the whole request, which ends the
    conversation rather than the turn.
    """
    msg = _response_output_item_to_chat_completion_message(
        {
            "type": "reasoning",
            "id": "rs_2",
            "summary": [],
            "encrypted_content": "gAAAAA",
        }
    )

    assert msg is None


def test_a_message_with_no_content_is_still_rejected():
    """The tolerance above is specific to reasoning.

    A message carries its text in `content` and nowhere else, so an empty one
    is malformed and silently dropping it would lose something the client
    believed it had sent.
    """
    with pytest.raises(ValueError):
        _response_output_item_to_chat_completion_message(
            {
                "type": "message",
                "role": "user",
                "content": [],
            }
        )


def test_role_without_type_gets_its_parts_translated():
    """`type` is optional on an input message; the parts still need rewriting.

    This is the API's EasyInputMessage. Returning it verbatim handed
    `input_text` to the chat-completions content parser, which knows only
    `text` and failed the request with "Unknown part type: input_text" -- so
    whether a request worked depended on a field the client may leave out.
    """
    msg = _response_output_item_to_chat_completion_message(
        {
            "role": "user",
            "content": [{"type": "input_text", "text": "hello"}],
        }
    )

    assert msg == {"role": "user", "content": [{"type": "text", "text": "hello"}]}


def test_an_assistant_turn_replayed_without_a_type_is_accepted():
    """Replaying this server's own output must not need a field it omits.

    The assistant parts come back spelled `output_text`, which is what the
    Responses API calls them.
    """
    msg = _response_output_item_to_chat_completion_message(
        {
            "role": "assistant",
            "content": [{"type": "output_text", "text": "hi"}],
        }
    )

    assert msg == {"role": "assistant", "content": [{"type": "text", "text": "hi"}]}


def test_non_text_parts_survive_the_translation():
    """Only the two Responses-only text spellings are rewritten.

    Flattening the list to a string would have been simpler and would have
    dropped the image, which the downstream parser does understand.
    """
    image = {"type": "image_url", "image_url": {"url": "http://example/x.png"}}
    msg = _response_output_item_to_chat_completion_message(
        {
            "role": "user",
            "content": [{"type": "input_text", "text": "what is this"}, image],
        }
    )

    assert msg["content"] == [{"type": "text", "text": "what is this"}, image]


def test_a_plain_string_content_is_left_alone():
    """The common shape must not be disturbed by the list handling."""
    msg = _response_output_item_to_chat_completion_message(
        {
            "role": "user",
            "content": "hello",
        }
    )

    assert msg == {"role": "user", "content": "hello"}


def test_additional_tools_item_becomes_tools():
    """Codex declares its tools as an input item, not in `tools`.

    The item carries no text, so the input-item conversion dropped it as
    unrecognised and the model was offered nothing to call -- it then narrated
    a terminal session it had invented, because narrating was the only thing
    left to do.
    """
    from tensorrt_llm.serve.openai_protocol import ResponsesRequest

    request = ResponsesRequest(
        model="m",
        input=[
            {
                "type": "additional_tools",
                "role": "developer",
                "tools": [
                    {
                        "type": "namespace",
                        "name": "functions",
                        "description": "",
                        "tools": [
                            {
                                "type": "function",
                                "name": "get_time",
                                "description": "Return the time.",
                                "parameters": {"type": "object", "properties": {}},
                            }
                        ],
                    }
                ],
            },
            {"role": "user", "content": "what time is it"},
        ],
    )

    # The item is gone from the input: it is a tool declaration, not a turn,
    # and replaying it as one would put the tool schema in the prompt as prose.
    assert [item.get("type") for item in request.input] == [None]
    assert len(request.tools) == 1
    assert request.tools[0].type == "namespace"
    assert [t.name for t in request.tools[0].tools] == ["get_time"]


def test_hoisted_tools_are_offered_to_the_template_namespaced():
    """The nested tools have to reach the prompt under qualified names.

    `_namespaced_tool_names` maps a reply's call back to its namespace, so a
    tool that never made it into `tools` would come back as an unsupported
    call even if the model somehow guessed it.
    """
    from tensorrt_llm.serve.openai_protocol import ResponsesRequest
    from tensorrt_llm.serve.responses_utils import (
        _get_chat_completion_function_tools,
        _namespaced_tool_names,
    )

    request = ResponsesRequest(
        model="m",
        input=[
            {
                "type": "additional_tools",
                "role": "developer",
                "tools": [
                    {
                        "type": "namespace",
                        "name": "collaboration",
                        "description": "",
                        "tools": [
                            {
                                "type": "function",
                                "name": "spawn_agent",
                                "description": "",
                                "parameters": {"type": "object", "properties": {}},
                            }
                        ],
                    }
                ],
            },
            {
                "role": "user",
                "content": "go",
            },
        ],
    )

    offered = [t.function.name for t in _get_chat_completion_function_tools(request.tools)]
    assert offered == ["collaboration.spawn_agent"]
    assert _namespaced_tool_names(request.tools) == {
        "collaboration.spawn_agent": ("collaboration", "spawn_agent"),
    }


def test_tools_already_in_the_tools_field_are_kept():
    """Hoisting appends; it must not discard what the client sent normally."""
    from tensorrt_llm.serve.openai_protocol import ResponsesRequest

    request = ResponsesRequest(
        model="m",
        tools=[
            {
                "type": "function",
                "name": "existing",
                "description": "",
                "parameters": {"type": "object", "properties": {}},
            }
        ],
        input=[
            {
                "type": "additional_tools",
                "role": "developer",
                "tools": [{"type": "namespace", "name": "ns", "description": "", "tools": []}],
            }
        ],
    )

    assert [getattr(t, "name", None) for t in request.tools] == ["existing", "ns"]


def test_a_request_without_the_item_is_untouched():
    """The common case must not be reshaped by the hoist."""
    from tensorrt_llm.serve.openai_protocol import ResponsesRequest

    request = ResponsesRequest(model="m", input="hello")

    assert request.input == "hello"
    assert request.tools == []


def test_function_call_output_with_content_parts_is_translated():
    """A tool result may carry parts rather than a string.

    Assigning them to `content` untouched handed `input_text` to the
    chat-completions parser, which knows only `text`, and the request failed
    with "Unknown part type: input_text".
    """
    msg = _response_output_item_to_chat_completion_message(
        {
            "type": "function_call_output",
            "call_id": "call_1",
            "output": [{"type": "input_text", "text": "42"}],
        }
    )

    assert msg == {
        "role": "tool",
        "content": [{"type": "text", "text": "42"}],
        "tool_call_id": "call_1",
    }


def test_custom_tool_call_output_with_content_parts_is_translated():
    """The custom-tool branch had the same assumption.

    This is the one that fired in practice: a tool result arrives on every
    turn after the model's first custom-tool call, so once an agent started
    using tools nearly all of its traffic was rejected -- 483 of 490 requests
    in one campaign round.
    """
    msg = _response_output_item_to_chat_completion_message(
        {
            "type": "custom_tool_call_output",
            "call_id": "call_2",
            "output": [
                {"type": "input_text", "text": "ok"},
                {"type": "input_text", "text": " done"},
            ],
        }
    )

    assert msg["role"] == "tool"
    assert msg["content"] == [{"type": "text", "text": "ok"}, {"type": "text", "text": " done"}]
    assert msg["tool_call_id"] == "call_2"


def test_a_string_tool_result_is_unchanged():
    """The simple shape must not be reshaped into parts."""
    msg = _response_output_item_to_chat_completion_message(
        {
            "type": "function_call_output",
            "call_id": "call_3",
            "output": "plain text",
        }
    )

    assert msg["content"] == "plain text"


def test_a_missing_custom_tool_result_stays_empty():
    """Absent output is still not a reason to fail the turn."""
    msg = _response_output_item_to_chat_completion_message(
        {
            "type": "custom_tool_call_output",
            "call_id": "call_4",
        }
    )

    assert msg["content"] == ""
