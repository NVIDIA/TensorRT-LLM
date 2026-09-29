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
"""Offline tests for freeform custom tools on the Responses API.

A custom tool takes one freeform string instead of JSON arguments; Codex
declares apply_patch this way, describing it as "a FREEFORM tool, so do not
wrap the patch in JSON". Reporting such a call as an ordinary function call
hands the client JSON where it expects the raw payload, and the client
rejects it - "tool apply_patch invoked with incompatible payload" - which
aborts the turn.
"""

import json
from types import SimpleNamespace

import pytest

from tensorrt_llm.serve.responses_utils import (
    CUSTOM_TOOL_INPUT_ARG,
    _custom_tool_names,
    _get_chat_completion_function_tools,
    _response_output_item_to_chat_completion_message,
    _tool_call_output_item,
)

# The CPU-* CI stages run pytest with -m 'cpu_only'. Without this marker every
# test in the file is deselected, which pytest reports as exit code 5 and the
# stage reports as a failure.
pytestmark = pytest.mark.cpu_only

PATCH = "*** Begin Patch\n*** Update File: a.txt\n-old\n+new\n*** End Patch"


def _custom_tool(name="apply_patch", description="Edit files."):
    return SimpleNamespace(type="custom", name=name, description=description)


def _call(name, parameters):
    return SimpleNamespace(name=name, parameters=parameters)


# ---------------------------------------------------------------------------
# How the tool is described to the model
# ---------------------------------------------------------------------------


def test_custom_tool_is_offered_with_a_named_string_parameter():
    """Without a named parameter the model invents its own argument names."""
    tools = _get_chat_completion_function_tools([_custom_tool()])
    assert len(tools) == 1
    params = tools[0].function.parameters
    assert params["required"] == [CUSTOM_TOOL_INPUT_ARG]
    assert params["properties"][CUSTOM_TOOL_INPUT_ARG]["type"] == "string"


def test_custom_tool_names_are_collected():
    tools = [_custom_tool(), SimpleNamespace(type="function", name="shell")]
    assert _custom_tool_names(tools) == {"apply_patch"}


def test_no_tools_yields_no_custom_names():
    assert _custom_tool_names(None) == set()


# ---------------------------------------------------------------------------
# How the call is reported back
# ---------------------------------------------------------------------------


def test_custom_tool_call_carries_the_freeform_payload():
    item = _tool_call_output_item(
        _call("apply_patch", json.dumps({CUSTOM_TOOL_INPUT_ARG: PATCH})), {"apply_patch"}
    )
    assert item.type == "custom_tool_call"
    assert item.input == PATCH
    assert item.name == "apply_patch"


def test_unknown_argument_name_is_still_forwarded():
    """A payload under an unexpected key beats dropping the call."""
    item = _tool_call_output_item(
        _call("apply_patch", json.dumps({"patch": PATCH})), {"apply_patch"}
    )
    assert item.input == PATCH


def test_non_json_arguments_are_passed_through_verbatim():
    item = _tool_call_output_item(_call("apply_patch", PATCH), {"apply_patch"})
    assert item.input == PATCH


def test_ordinary_tools_are_still_function_calls():
    item = _tool_call_output_item(_call("shell", '{"cmd": "ls"}'), {"apply_patch"})
    assert item.type == "function_call"
    assert item.arguments == '{"cmd": "ls"}'


def test_custom_and_function_calls_get_distinct_id_prefixes():
    custom = _tool_call_output_item(_call("apply_patch", PATCH), {"apply_patch"})
    function = _tool_call_output_item(_call("shell", "{}"), {"apply_patch"})
    assert custom.id.startswith("ctc_")
    assert function.id.startswith("fc_")


# ---------------------------------------------------------------------------
# How the call is replayed on the next turn
# ---------------------------------------------------------------------------


def test_custom_tool_call_replays_as_a_tool_call():
    """An unhandled item type raises, ending the conversation one turn later."""
    msg = _response_output_item_to_chat_completion_message(
        {
            "type": "custom_tool_call",
            "id": "ctc_1",
            "call_id": "call_1",
            "name": "apply_patch",
            "input": PATCH,
        }
    )
    assert msg["role"] == "assistant"
    call = msg["tool_calls"][0]
    assert call["function"]["name"] == "apply_patch"
    assert json.loads(call["function"]["arguments"]) == {CUSTOM_TOOL_INPUT_ARG: PATCH}


def test_custom_tool_call_output_replays_as_a_tool_result():
    msg = _response_output_item_to_chat_completion_message(
        {
            "type": "custom_tool_call_output",
            "call_id": "call_1",
            "output": "Done.",
        }
    )
    assert msg == {"role": "tool", "content": "Done.", "tool_call_id": "call_1"}


# ---------------------------------------------------------------------------
# Namespaced tools
# ---------------------------------------------------------------------------


def _namespace_tool(namespace="collaboration", names=("spawn_agent",)):
    inner = [
        SimpleNamespace(type="function", name=n, description=None, parameters=None) for n in names
    ]
    return SimpleNamespace(
        type="namespace", name=namespace, description="Agent collaboration.", tools=inner
    )


def test_namespaced_call_reports_the_namespace_separately():
    """Regression: every collaboration.* call came back "unsupported call".

    The client identifies a namespaced tool by its bare name plus the
    namespace field. A call named "collaboration.spawn_agent" matches
    nothing it knows, so the whole capability is unusable.
    """
    from tensorrt_llm.serve.responses_utils import _namespaced_tool_names

    tools = [_namespace_tool()]
    item = _tool_call_output_item(
        _call("collaboration.spawn_agent", "{}"), set(), _namespaced_tool_names(tools)
    )
    assert item.name == "spawn_agent"
    assert item.namespace == "collaboration"


def test_unnamespaced_call_has_no_namespace():
    item = _tool_call_output_item(_call("shell", "{}"), set(), {})
    assert item.namespace is None


def test_namespaced_custom_tool_is_recognised():
    tools = [
        SimpleNamespace(
            type="namespace",
            name="edit",
            description=None,
            tools=[SimpleNamespace(type="custom", name="apply_patch", description=None)],
        )
    ]
    assert _custom_tool_names(tools) == {"edit.apply_patch"}


def test_namespaced_call_replays_under_its_qualified_name():
    """The model was offered the qualified name, so that is what it must see."""
    msg = _response_output_item_to_chat_completion_message(
        {
            "type": "function_call",
            "call_id": "call_1",
            "name": "spawn_agent",
            "namespace": "collaboration",
            "arguments": "{}",
        }
    )
    assert msg["tool_calls"][0]["function"]["name"] == "collaboration.spawn_agent"


# ---------------------------------------------------------------------------
# Item types the client defines itself
# ---------------------------------------------------------------------------


def test_agent_message_is_replayed_as_input():
    """Codex multi-agent sessions carry these; no SDK model describes them."""
    msg = _response_output_item_to_chat_completion_message(
        {
            "type": "agent_message",
            "id": "amsg_1",
            "author": "/root/probe",
            "recipient": "/root",
            "content": [{"type": "input_text", "text": "Task finished."}],
        }
    )
    assert msg == {"role": "user", "content": "Task finished."}


def test_unknown_item_with_text_is_replayed_rather_than_refused():
    """Refusing an item fails the request, which ends the whole conversation."""
    msg = _response_output_item_to_chat_completion_message(
        {
            "type": "something_new",
            "content": [{"type": "input_text", "text": "hello"}],
        }
    )
    assert msg["content"] == "hello"


def test_unknown_item_without_text_is_dropped():
    assert (
        _response_output_item_to_chat_completion_message(
            {
                "type": "something_new",
                "id": "x_1",
            }
        )
        is None
    )


# ---------------------------------------------------------------------------
# Conversation history for a reasoning model
# ---------------------------------------------------------------------------


def test_history_records_a_custom_tool_call():
    """Regression: reading `arguments` on a custom tool call failed the request.

    History is only built when the model emits reasoning, so this stayed
    hidden behind a model with thinking disabled and appeared as an HTTP 400
    the moment a reasoning model called a freeform tool.
    """
    from tensorrt_llm.serve.responses_utils import _stored_tool_arguments, _stored_tool_name

    item = _tool_call_output_item(_call("apply_patch", PATCH), {"apply_patch"})
    assert json.loads(_stored_tool_arguments(item)) == {CUSTOM_TOOL_INPUT_ARG: PATCH}
    assert _stored_tool_name(item) == "apply_patch"


def test_history_records_a_function_call_unchanged():
    from tensorrt_llm.serve.responses_utils import _stored_tool_arguments

    item = _tool_call_output_item(_call("shell", '{"cmd": "ls"}'), set())
    assert _stored_tool_arguments(item) == '{"cmd": "ls"}'


def test_history_requalifies_a_namespaced_call():
    from tensorrt_llm.serve.responses_utils import _namespaced_tool_names, _stored_tool_name

    item = _tool_call_output_item(
        _call("collaboration.spawn_agent", "{}"), set(), _namespaced_tool_names([_namespace_tool()])
    )
    assert _stored_tool_name(item) == "collaboration.spawn_agent"


def test_a_custom_tool_inside_a_namespace_is_described_with_its_input_arg():
    """Regression: the input and output paths must agree on the schema.

    _custom_tool_names classifies a namespaced custom tool under its qualified
    name, so the output path looks for CUSTOM_TOOL_INPUT_ARG. If the namespace
    branch described the tool with an empty object schema, the model would
    never be told that argument exists and would invent its own name - the
    exact failure the top-level custom branch exists to prevent.
    """
    tools = [
        SimpleNamespace(
            type="namespace",
            name="edit",
            description="editing tools",
            tools=[SimpleNamespace(type="custom", name="apply_patch", description=None)],
        )
    ]

    fns = _get_chat_completion_function_tools(tools)
    by_name = {f.function.name: f.function.parameters for f in fns}

    assert "edit.apply_patch" in by_name
    params = by_name["edit.apply_patch"]
    assert CUSTOM_TOOL_INPUT_ARG in params["properties"]
    assert params["required"] == [CUSTOM_TOOL_INPUT_ARG]


def test_namespaced_function_tools_keep_their_own_schema():
    """The custom-tool schema must not be forced onto ordinary functions."""
    schema = {"type": "object", "properties": {"path": {"type": "string"}}}
    tools = [
        SimpleNamespace(
            type="namespace",
            name="fs",
            description=None,
            tools=[
                SimpleNamespace(type="function", name="read", description=None, parameters=schema)
            ],
        )
    ]

    fns = _get_chat_completion_function_tools(tools)
    by_name = {f.function.name: f.function.parameters for f in fns}
    assert by_name["fs.read"] == schema
