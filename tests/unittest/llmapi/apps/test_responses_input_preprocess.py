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

import asyncio

import pytest

from tensorrt_llm.serve.openai_protocol import ResponsesRequest
from tensorrt_llm.serve.responses_utils import (
    _create_input_messages,
    _get_chat_completion_function_tools,
    _render_developer_as_system,
    _response_output_item_to_chat_completion_message,
    _tool_resolution,
)
from tensorrt_llm.tokenizer.deepseek_v4 import DeepseekV4Tokenizer

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


@pytest.mark.parametrize("content", [[], None])
def test_empty_content_is_rejected(content):
    with pytest.raises(ValueError, match="empty or missing"):
        _response_output_item_to_chat_completion_message(
            {"type": "message", "role": "user", "content": content}
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
# Item shapes beyond a list of text parts
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "item, expected",
    [
        (
            {"type": "message", "role": "user", "content": "hello"},
            {"role": "user", "content": "hello"},
        ),
        ({"type": "message", "role": "user", "content": ""}, {"role": "user", "content": ""}),
        (
            {"type": "reasoning", "content": "thinking"},
            {"role": "assistant", "reasoning": "thinking"},
        ),
        (
            {"type": "reasoning", "summary": [{"type": "summary_text", "text": "weighed it"}]},
            {"role": "assistant", "reasoning": "weighed it"},
        ),
        ({"type": "reasoning", "summary": [], "encrypted_content": "gAAAAA"}, None),
        ({"type": "output_text", "text": "hi"}, {"role": "assistant", "content": "hi"}),
        (
            {
                "type": "agent_message",
                "content": [
                    {"type": "input_text", "text": "Payload:"},
                    {"type": "encrypted_content", "encrypted_content": "write the file"},
                    {"type": "encrypted_content", "encrypted_content": {"blob": "aGk="}},
                    {"type": "encrypted_content", "encrypted_content": "gAAAAABo" + "Zm9v" * 16},
                ],
            },
            {"role": "user", "content": "Payload:\nwrite the file"},
        ),
    ],
)
def test_item_text_is_read_from_every_shape_that_carries_it(item, expected):
    assert _response_output_item_to_chat_completion_message(item) == expected


_IMAGE = {"type": "image_url", "image_url": {"url": "http://example/x.png"}}


@pytest.mark.parametrize(
    "content, expected",
    [
        ([{"type": "input_text", "text": "hello"}], [{"type": "text", "text": "hello"}]),
        ([{"type": "output_text", "text": "hi"}], [{"type": "text", "text": "hi"}]),
        (
            [{"type": "input_text", "text": "what is this"}, _IMAGE],
            [{"type": "text", "text": "what is this"}, _IMAGE],
        ),
        ("hello", "hello"),
    ],
)
def test_untyped_message_parts_are_translated_to_chat_parts(content, expected):
    msg = _response_output_item_to_chat_completion_message({"role": "user", "content": content})
    assert msg == {"role": "user", "content": expected}


@pytest.mark.parametrize("item_type", ["function_call_output", "custom_tool_call_output"])
@pytest.mark.parametrize(
    "output, expected",
    [
        (
            [{"type": "input_text", "text": "ok"}, {"type": "input_text", "text": " done"}],
            [{"type": "text", "text": "ok"}, {"type": "text", "text": " done"}],
        ),
        ("plain text", "plain text"),
    ],
)
def test_tool_results_are_translated_to_chat_parts(item_type, output, expected):
    msg = _response_output_item_to_chat_completion_message(
        {"type": item_type, "call_id": "call_1", "output": output}
    )
    assert msg == {"role": "tool", "content": expected, "tool_call_id": "call_1"}


def test_a_missing_custom_tool_result_is_empty():
    msg = _response_output_item_to_chat_completion_message(
        {"type": "custom_tool_call_output", "call_id": "call_4"}
    )
    assert msg["content"] == ""


# ---------------------------------------------------------------------------
# Tools declared in an `additional_tools` input item
# ---------------------------------------------------------------------------


def _additional_tools_item(tools):
    return {"type": "additional_tools", "role": "developer", "tools": tools}


def _namespace(name, *functions):
    return {
        "type": "namespace",
        "name": name,
        "description": "",
        "tools": [
            {
                "type": "function",
                "name": f,
                "description": "",
                "parameters": {"type": "object", "properties": {}},
            }
            for f in functions
        ],
    }


def test_additional_tools_are_hoisted_after_the_declared_tools():
    request = ResponsesRequest(
        model="m",
        tools=[_namespace("existing", "a")],
        input=[
            _additional_tools_item([_namespace("collaboration", "spawn_agent")]),
            {"role": "user", "content": "go"},
        ],
    )

    assert [item.get("type") for item in request.input] == [None]
    assert [t.name for t in request.tools] == ["existing", "collaboration"]
    offered = [t.function.name for t in _get_chat_completion_function_tools(request.tools)]
    assert offered == ["existing.a", "collaboration.spawn_agent"]
    resolution = _tool_resolution(request.tools)
    assert resolution["collaboration.spawn_agent"] == ("collaboration", "spawn_agent", False)
    assert resolution["spawn_agent"] == ("collaboration", "spawn_agent", False)


def test_a_malformed_additional_tools_item_is_not_hoisted():
    request = {"model": "m", "input": [_additional_tools_item("not a list")], "tools": None}
    assert ResponsesRequest.hoist_additional_tools(request) is request


# ---------------------------------------------------------------------------
# developer messages
# ---------------------------------------------------------------------------


class _TemplateTokenizer:
    def __init__(self, template):
        self.template = template

    def get_chat_template(self, chat_template, tools=None):
        return self.template


_DEVELOPER_TURN = [{"role": "developer", "content": "brief"}, {"role": "user", "content": "q"}]


def _template(body):
    return _TemplateTokenizer("{% for m in messages %}" + body + "{% endfor %}")


@pytest.mark.parametrize(
    "tokenizer, role",
    [
        (_template("{% if m.role in ('system', 'user') %}{{ m.content }}{% endif %}"), "system"),
        (
            _template(
                "{% if m.role in ('system', 'developer', 'user') %}{{ m.content }}{% endif %}"
            ),
            "developer",
        ),
        # Naming the role is not handling it.
        (
            _template(
                "{% if m.role == 'developer' %}"
                "{{ raise_exception('Unsupported role: developer') }}{% endif %}{{ m.content }}"
            ),
            "system",
        ),
        (
            _TemplateTokenizer(
                "{# developer messages are not supported #}{% for m in messages %}"
                "{% if m.role != 'developer' %}{{ m.content }}{% endif %}{% endfor %}"
            ),
            "system",
        ),
        # A template that renders neither role tells nothing; the roles stay.
        (_template("{{ m.content.missing.attribute }}"), "developer"),
        (DeepseekV4Tokenizer.__new__(DeepseekV4Tokenizer), "developer"),
    ],
    ids=[
        "no_developer_branch",
        "developer_branch",
        "rejects_developer",
        "developer_only_in_a_comment",
        "renders_neither",
        "deepseek_v4",
    ],
)
def test_developer_renders_as_system_only_where_the_template_lacks_it(tokenizer, role):
    messages = _render_developer_as_system(_DEVELOPER_TURN, tokenizer, None, None)
    assert [m["role"] for m in messages] == [role, "user"]
    assert _DEVELOPER_TURN[0]["role"] == "developer"


# ---------------------------------------------------------------------------
# A multi-call assistant turn stays one assistant message, so the chat
# template can bind each tool result to its call by id
# ---------------------------------------------------------------------------


def _function_call_item(index):
    return {
        "type": "function_call",
        "id": f"fc_{index}",
        "call_id": f"call_{index}",
        "name": "exec",
        "arguments": f'{{"cmd": "step {index}"}}',
        "status": "completed",
    }


def _function_call_output_item(index):
    return {"type": "function_call_output", "call_id": f"call_{index}", "output": f"result {index}"}


def test_calls_from_one_turn_become_one_assistant_message():
    messages = _messages(
        {
            "input": [
                _message_item("user", "run the plan"),
                {
                    "id": "msg_a",
                    "status": "completed",
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "Running.", "annotations": []}],
                },
                *[_function_call_item(i) for i in range(3)],
                *[_function_call_output_item(i) for i in range(3)],
            ]
        }
    )

    assert [m["role"] for m in messages] == ["user", "assistant"] + ["tool"] * 3
    assert messages[1]["content"] == "Running."
    assert [c["id"] for c in messages[1]["tool_calls"]] == ["call_0", "call_1", "call_2"]
    assert [m["tool_call_id"] for m in messages[2:]] == ["call_0", "call_1", "call_2"]


@pytest.mark.parametrize(
    "between",
    [_function_call_output_item(0), _message_item("user", "also do this")],
    ids=["tool_result", "user_message"],
)
def test_a_turn_in_between_ends_the_assistant_message(between):
    messages = _messages({"input": [_function_call_item(0), between, _function_call_item(1)]})
    assert [len(m.get("tool_calls") or []) for m in messages] == [1, 0, 1]


def test_calls_fold_onto_a_reasoning_message():
    messages = _messages(
        {
            "input": [
                {"type": "reasoning", "content": [{"type": "reasoning_text", "text": "plan"}]},
                _function_call_item(0),
                _function_call_item(1),
            ]
        }
    )
    assert len(messages) == 1
    assert messages[0]["reasoning"] == "plan"
    assert [c["id"] for c in messages[0]["tool_calls"]] == ["call_0", "call_1"]


def test_folding_leaves_the_request_unchanged():
    request = ResponsesRequest(
        model="m",
        input=[{"role": "assistant", "content": "Running."}, _function_call_item(0)],
    )
    for _ in range(2):
        messages = asyncio.run(_create_input_messages(request=request, prev_msgs=[]))
        assert [c["id"] for c in messages[0]["tool_calls"]] == ["call_0"]
    assert "tool_calls" not in request.input[0]


def test_calls_never_fold_into_replayed_history():
    request = ResponsesRequest(model="m", input=[_function_call_item(0)])
    history = [{"role": "assistant", "content": "a stored turn"}]
    messages = asyncio.run(_create_input_messages(request=request, prev_msgs=history))

    assert messages[0] == {"role": "assistant", "content": "a stored turn"}
    assert [c["id"] for c in messages[1]["tool_calls"]] == ["call_0"]
