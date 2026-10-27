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

from tensorrt_llm.serve.openai_protocol import NamespacedFunctionToolParam, ResponsesRequest
from tensorrt_llm.serve.responses_utils import (
    CUSTOM_TOOL_INPUT_ARG,
    _get_chat_completion_function_tools,
    _native_tools_for_chat_template,
    _response_output_item_to_chat_completion_message,
    _template_supports_deferred_tools,
    _tool_call_output_item,
    _tool_resolution,
    _tools_for_chat_template,
)

from .render_helpers import byte_level_tokenizer, responses_prompt

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
    resolution = _tool_resolution(tools)
    assert resolution["apply_patch"][2] is True
    assert resolution["shell"][2] is False


def test_no_tools_yields_no_custom_names():
    assert _tool_resolution(None) == {}


# ---------------------------------------------------------------------------
# How the call is reported back
# ---------------------------------------------------------------------------


_CUSTOM_RES = {"apply_patch": (None, "apply_patch", True)}


def test_custom_tool_call_carries_the_freeform_payload():
    item = _tool_call_output_item(
        _call("apply_patch", json.dumps({CUSTOM_TOOL_INPUT_ARG: PATCH})), _CUSTOM_RES
    )
    assert item.type == "custom_tool_call"
    assert item.input == PATCH
    assert item.name == "apply_patch"


def test_unknown_argument_name_is_still_forwarded():
    """A payload under an unexpected key beats dropping the call."""
    item = _tool_call_output_item(_call("apply_patch", json.dumps({"patch": PATCH})), _CUSTOM_RES)
    assert item.input == PATCH


def test_non_json_arguments_are_passed_through_verbatim():
    item = _tool_call_output_item(_call("apply_patch", PATCH), _CUSTOM_RES)
    assert item.input == PATCH


def test_ordinary_tools_are_still_function_calls():
    item = _tool_call_output_item(_call("shell", '{"cmd": "ls"}'), _CUSTOM_RES)
    assert item.type == "function_call"
    assert item.arguments == '{"cmd": "ls"}'


def test_custom_and_function_calls_get_distinct_id_prefixes():
    custom = _tool_call_output_item(_call("apply_patch", PATCH), _CUSTOM_RES)
    function = _tool_call_output_item(_call("shell", "{}"), _CUSTOM_RES)
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
    tools = [_namespace_tool()]
    item = _tool_call_output_item(_call("collaboration.spawn_agent", "{}"), _tool_resolution(tools))
    assert item.name == "spawn_agent"
    assert item.namespace == "collaboration"


def test_unnamespaced_call_has_no_namespace():
    item = _tool_call_output_item(_call("shell", "{}"), {})
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
    resolution = _tool_resolution(tools)
    assert resolution["edit.apply_patch"] == ("edit", "apply_patch", True)
    # The bare spelling resolves too when unambiguous - the model writes both.
    assert resolution["apply_patch"] == ("edit", "apply_patch", True)


def test_a_namespaced_tool_keeps_its_declaration_beside_the_flattened_function():
    """A model whose renderer shows namespaces needs the declaration itself."""
    namespace = SimpleNamespace(
        type="namespace",
        name="collaboration",
        description="Agent collaboration.",
        tools=[
            SimpleNamespace(type="function", name="spawn_agent", description=None, parameters=None),
            SimpleNamespace(type="custom", name="exec", description="Run code."),
        ],
    )

    spawn, exec_ = _get_chat_completion_function_tools([namespace])
    (top_level,) = _get_chat_completion_function_tools([_custom_tool()])

    # A chat template is given the flattened function, as before.
    assert spawn.model_dump() == {
        "type": "function",
        "function": {
            "name": "collaboration.spawn_agent",
            "description": "Agent collaboration.",
            "parameters": {"type": "object", "properties": {}},
            "strict": None,
        },
    }
    declarations = [
        (tool.namespace, tool.namespace_description, tool.declared_name, tool.declared_description)
        for tool in (spawn, exec_)
    ]
    assert declarations == [
        ("collaboration", "Agent collaboration.", "spawn_agent", None),
        ("collaboration", "Agent collaboration.", "exec", "Run code."),
    ]
    assert not isinstance(top_level, NamespacedFunctionToolParam)


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
# Spellings of a namespaced tool's name
# ---------------------------------------------------------------------------


def _function_tool(name):
    return SimpleNamespace(type="function", name=name, description=None, parameters=None)


def _codex_tools():
    """Tools as the Codex CLI declares them: every one under a namespace."""
    functions = SimpleNamespace(
        type="namespace",
        name="functions",
        description="",
        tools=[_custom_tool("exec", "Run code."), _function_tool("wait")],
    )
    return [functions, _namespace_tool("collaboration", ("spawn_agent",))]


@pytest.mark.parametrize("spelling", ["functions.exec", "exec", "functions::exec"])
def test_every_spelling_of_a_namespaced_custom_tool_is_a_custom_tool_call(spelling):
    """Delivered as a function call, a custom tool's call is refused by the client.

    The model is shown "functions.exec". Models also write the bare name, and
    DeepSeek-V4.1 writes "functions::exec", the separator it was trained on.
    """
    arguments = json.dumps({CUSTOM_TOOL_INPUT_ARG: "ls -la"})
    item = _tool_call_output_item(_call(spelling, arguments), _tool_resolution(_codex_tools()))
    assert (item.type, item.namespace, item.name, item.input) == (
        "custom_tool_call",
        "functions",
        "exec",
        "ls -la",
    )


@pytest.mark.parametrize(
    "spelling", ["collaboration.spawn_agent", "spawn_agent", "collaboration::spawn_agent"]
)
def test_every_spelling_of_a_namespaced_function_is_a_function_call(spelling):
    item = _tool_call_output_item(
        _call(spelling, '{"task": "review"}'), _tool_resolution(_codex_tools())
    )
    assert (item.type, item.namespace, item.name, item.arguments) == (
        "function_call",
        "collaboration",
        "spawn_agent",
        '{"task": "review"}',
    )


@pytest.mark.parametrize(
    "spelling",
    ["missing", "functions::missing", "elsewhere::exec", "functions:exec", "functions::", "::exec"],
)
def test_a_name_no_declared_tool_answers_to_is_reported_as_written(spelling):
    item = _tool_call_output_item(_call(spelling, "{}"), _tool_resolution(_codex_tools()))
    assert (item.type, item.namespace, item.name) == ("function_call", None, spelling)


def test_a_bare_name_two_namespaces_share_resolves_to_neither():
    """Routing a call to the wrong namespace is worse than not resolving it."""
    resolution = _tool_resolution(
        [_namespace_tool("local", ("run",)), _namespace_tool("remote", ("run",))]
    )
    assert "run" not in resolution
    for namespace in ("local", "remote"):
        for separator in (".", "::"):
            assert resolution[f"{namespace}{separator}run"] == (namespace, "run", False)

    item = _tool_call_output_item(_call("run", "{}"), resolution)
    assert (item.type, item.namespace, item.name) == ("function_call", None, "run")


@pytest.mark.parametrize("top_level_first", [True, False])
def test_a_declared_name_wins_over_another_tools_spelling(top_level_first):
    """A top-level tool is what the model was shown under its exact name."""
    top_level = [_function_tool("exec"), _function_tool("functions::exec")]
    tools = top_level + _codex_tools() if top_level_first else _codex_tools() + top_level
    resolution = _tool_resolution(tools)
    assert resolution["exec"] == (None, "exec", False)
    assert resolution["functions::exec"] == (None, "functions::exec", False)
    assert resolution["functions.exec"] == ("functions", "exec", True)


# ---------------------------------------------------------------------------
# A call echoed back without its namespace
# ---------------------------------------------------------------------------


def _echoed_call(item_type, name, namespace=None):
    item = {"type": item_type, "call_id": "call_1", "name": name}
    if item_type == "custom_tool_call":
        item["input"] = "ls"
    else:
        item["arguments"] = "{}"
    if namespace is not None:
        item["namespace"] = namespace
    return item


def _replayed_name(item, tools):
    msg = _response_output_item_to_chat_completion_message(item, _tool_resolution(tools))
    return msg["tool_calls"][0]["function"]["name"]


@pytest.mark.parametrize(
    "item_type, namespace, name",
    [("custom_tool_call", "functions", "exec"), ("function_call", "collaboration", "spawn_agent")],
)
def test_a_call_echoed_without_its_namespace_is_replayed_qualified(item_type, namespace, name):
    """The model was offered the qualified name; the bare one is not on its list."""
    qualified = f"{namespace}.{name}"
    assert _replayed_name(_echoed_call(item_type, name), _codex_tools()) == qualified
    assert _replayed_name(_echoed_call(item_type, name, namespace), _codex_tools()) == qualified


@pytest.mark.parametrize(
    "tools",
    [
        [_namespace_tool("local", ("run",)), _namespace_tool("remote", ("run",))],
        [_namespace_tool("local", ("run",)), _function_tool("run")],
        [],
    ],
    ids=["two_namespaces", "top_level_tool", "no_tools"],
)
def test_an_echoed_bare_name_without_one_namespaced_owner_is_replayed_as_received(tools):
    assert _replayed_name(_echoed_call("function_call", "run"), tools) == "run"


@pytest.mark.parametrize("name", ["functions.exec", "functions::exec", "missing"])
def test_an_echoed_name_that_is_not_a_bare_declared_name_is_replayed_as_received(name):
    """Only a dropped namespace is restored; a name is never rewritten.

    The client answers a call to an undeclared name with an error, so the
    history has to keep the name the call was made under.
    """
    assert _replayed_name(_echoed_call("custom_tool_call", name), _codex_tools()) == name


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

    item = _tool_call_output_item(_call("apply_patch", PATCH), _CUSTOM_RES)
    assert json.loads(_stored_tool_arguments(item)) == {CUSTOM_TOOL_INPUT_ARG: PATCH}
    assert _stored_tool_name(item) == "apply_patch"


def test_history_records_a_function_call_unchanged():
    from tensorrt_llm.serve.responses_utils import _stored_tool_arguments

    item = _tool_call_output_item(_call("shell", '{"cmd": "ls"}'), {})
    assert _stored_tool_arguments(item) == '{"cmd": "ls"}'


def test_history_requalifies_a_namespaced_call():
    from tensorrt_llm.serve.responses_utils import _stored_tool_name

    item = _tool_call_output_item(
        _call("collaboration.spawn_agent", "{}"), _tool_resolution([_namespace_tool()])
    )
    assert _stored_tool_name(item) == "collaboration.spawn_agent"


def test_a_custom_tool_inside_a_namespace_is_described_with_its_input_arg():
    """Regression: the input and output paths must agree on the schema.

    _tool_resolution classifies a namespaced custom tool under its qualified
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


# ---------------------------------------------------------------------------
# Native (namespace / defer_loading) rendering for templates that support it
# ---------------------------------------------------------------------------


# A template that reads deferred_render_mode, the marker variable of the
# revision that renders namespace/deferred tools natively.
DEFERRED_TEMPLATE = (
    "{%- set deferred_render_mode = deferred_render_mode"
    " | default('tool_name_and_description') -%}{{ messages }}"
)
EAGER_TEMPLATE = "{{ messages }}{{ tools }}"


def _function_tool(name="get_weather", defer_loading=None, parameters=None):
    return SimpleNamespace(
        type="function",
        name=name,
        description="d",
        parameters=parameters,
        defer_loading=defer_loading,
    )


def _deferred_namespace():
    return SimpleNamespace(
        type="namespace",
        name="slack",
        description="Slack tools.",
        defer_loading=True,
        tools=[
            SimpleNamespace(type="function", name="send", description="s", parameters=None),
            SimpleNamespace(
                type="function",
                name="read",
                description="r",
                parameters=None,
                defer_loading=False,
            ),
            SimpleNamespace(type="custom", name="apply_patch", description=None),
        ],
    )


def test_template_support_is_keyed_on_deferred_render_mode():
    assert _template_supports_deferred_tools(DEFERRED_TEMPLATE)
    assert not _template_supports_deferred_tools(EAGER_TEMPLATE)
    # A native renderer with no Jinja source cannot take the structures.
    assert not _template_supports_deferred_tools(None)


def test_plain_function_tools_stay_on_the_flattened_view():
    """No namespace and no defer flag: the rendering must not change."""
    flat = [{"type": "function", "function": {"name": "get_weather"}}]
    tools = [_function_tool()]
    assert _tools_for_chat_template(tools, flat, DEFERRED_TEMPLATE) is flat


def test_namespace_is_flattened_for_a_template_without_native_support():
    """An old template would mis-render the native structures."""
    flat = [{"type": "function", "function": {"name": "slack.send"}}]
    tools = [_deferred_namespace()]
    assert _tools_for_chat_template(tools, flat, EAGER_TEMPLATE) is flat
    assert _tools_for_chat_template(tools, flat, None) is flat


def test_namespace_passes_through_natively_on_a_supporting_template():
    tools = [_deferred_namespace()]
    (entry,) = _tools_for_chat_template(tools, [], DEFERRED_TEMPLATE)
    assert entry["type"] == "namespace"
    assert entry["name"] == "slack"
    assert entry["defer_loading"] is True
    assert [m["name"] for m in entry["tools"]] == ["send", "read", "apply_patch"]


def test_defer_loading_is_forwarded_tri_state():
    """Absent must stay absent: the template reads it as "inherit"."""
    (entry,) = _native_tools_for_chat_template([_deferred_namespace()])
    send, read, _ = entry["tools"]
    assert "defer_loading" not in send
    assert read["defer_loading"] is False


def test_custom_member_keeps_the_freeform_schema_in_native_form():
    """The native description must offer the same freeform schema.

    The output path still classifies slack.apply_patch as custom and goes
    looking for the named string parameter the flattened form declares.
    """
    (entry,) = _native_tools_for_chat_template([_deferred_namespace()])
    patch = entry["tools"][2]
    assert patch["parameters"]["required"] == [CUSTOM_TOOL_INPUT_ARG]


def test_top_level_defer_flag_alone_switches_to_native():
    tools = [_function_tool(defer_loading=True)]
    (entry,) = _tools_for_chat_template(tools, [], DEFERRED_TEMPLATE)
    assert entry["function"]["defer_loading"] is True
    # The schema-less tool still presents an object schema.
    assert entry["function"]["parameters"]["type"] == "object"


def test_native_form_drops_tools_the_template_cannot_describe():
    tools = [_deferred_namespace(), SimpleNamespace(type="web_search", name=None)]
    native = _native_tools_for_chat_template(tools)
    assert [t["type"] for t in native] == ["namespace"]


def test_sdk_validated_tools_keep_defer_loading():
    """defer_loading must survive pydantic validation of the SDK types.

    The SDK declares it on function tools and members, and the namespace
    carries it as an allowed extra; losing either silently renders every
    tool eagerly.
    """
    from openai.types.responses.tool import Tool
    from pydantic import TypeAdapter

    (ns,) = TypeAdapter(list[Tool]).validate_python(
        [
            {
                "type": "namespace",
                "name": "slack",
                "description": "Slack tools.",
                "defer_loading": True,
                "tools": [
                    {"type": "function", "name": "send", "description": "s", "parameters": None},
                    {
                        "type": "function",
                        "name": "read",
                        "description": "r",
                        "parameters": None,
                        "defer_loading": False,
                    },
                ],
            }
        ]
    )
    (entry,) = _native_tools_for_chat_template([ns])
    assert entry["defer_loading"] is True
    send, read = entry["tools"]
    assert "defer_loading" not in send
    assert read["defer_loading"] is False


# Renders tools the way the native template revision does: a namespace member without its
# own defer_loading inherits the namespace's.
NATIVE_RENDER_TEMPLATE = (
    "{%- set deferred_render_mode = deferred_render_mode"
    " | default('tool_name_and_description') -%}"
    "{%- for tool in tools -%}"
    "{%- if tool.type == 'namespace' -%}"
    "{%- for member in tool.tools -%}"
    "[{{ tool.name }}.{{ member.name }}"
    " deferred={{ member.defer_loading | default(tool.defer_loading) }}]"
    "{%- endfor -%}"
    "{%- else -%}[{{ tool.function.name }}]{%- endif -%}"
    "{%- endfor -%}"
    "{%- for message in messages %}{{ message.content }}{%- endfor -%}"
)


@pytest.mark.asyncio
async def test_responses_prompt_renders_the_namespace_deferral():
    """Both deferral forms reach the prompt the Responses route renders.

    ``send`` inherits the namespace's deferral and ``read`` opts out explicitly; the
    flattened view would render both eagerly.
    """
    request = ResponsesRequest(
        model="test-model",
        input="hi",
        tools=[
            {
                "type": "namespace",
                "name": "slack",
                "description": "Slack tools.",
                "defer_loading": True,
                "tools": [
                    {"type": "function", "name": "send", "description": "s", "parameters": None},
                    {
                        "type": "function",
                        "name": "read",
                        "description": "r",
                        "parameters": None,
                        "defer_loading": False,
                    },
                ],
            }
        ],
    )
    prompt = await responses_prompt(
        request, byte_level_tokenizer(NATIVE_RENDER_TEMPLATE), model_type="test-model"
    )
    assert prompt == "[slack.send deferred=True][slack.read deferred=False]hi"
