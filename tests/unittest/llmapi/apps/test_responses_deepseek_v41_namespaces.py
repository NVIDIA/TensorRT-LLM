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
"""DeepSeek-V4.1 sees a Responses request's tool namespaces in its own form.

The checkpoint's encoder takes a namespace beside each function. It shows the
tool as ``<namespace>::<name>`` with the namespace description in front of the
tool's own, and the model writes its calls under that name. Other chat
templates are given the flattened function ``<namespace>.<name>``.

Requests are rendered by the Responses route's own path, from the request to
the prompt string. The checkpoint comparison needs the checkpoint's encoder
and is skipped without ``LLM_MODELS_ROOT``.
"""

import asyncio
import json
from types import SimpleNamespace

import pytest

from tensorrt_llm.serve.openai_protocol import ResponsesRequest
from tensorrt_llm.serve.responses_utils import (
    _create_input_tokens,
    _get_chat_completion_function_tools,
)
from tensorrt_llm.serve.tool_parser.deepseekv41_parser import DeepSeekV41Parser
from tensorrt_llm.tokenizer.deepseek_v41 import DeepseekV41Tokenizer

from ..test_deepseek_v41_tokenizer import _load_vendor_encoding

pytestmark = pytest.mark.cpu_only

# The text-only model type selects the same serving extension as the
# multimodal one, which would also import the vision model to look up its
# image placeholders.
MODEL_TYPE = "deepseek_v41_text"
INSTRUCTIONS = "You are a coding agent."
USER = {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "Fix it."}]}
REASONING = "Plan the work."
EOS = "<｜end▁of▁sentence｜>"


def _object(**properties):
    return {"type": "object", "properties": properties}


_NAMESPACES = {"multi_agent_v1": "Tools for working with sub-agents.", "functions": ""}
# (namespace, name, description, parameters); no parameters is a custom tool.
_DECLARED = [
    (None, "exec_command", "Runs a command.", _object(cmd={"type": "string"})),
    (None, "write_stdin", "Writes to a session.", _object(chars={"type": "string"})),
    (None, "apply_patch", "Edits files. This is a FREEFORM tool.", None),
    ("multi_agent_v1", "spawn_agent", "Spawns a sub-agent.", _object(message={"type": "string"})),
    ("multi_agent_v1", "send_input", "Sends input.", _object(id={"type": "string"})),
    ("multi_agent_v1", "wait", None, _object(ids={"type": "array"})),
    ("functions", "exec", "Runs code.", None),
]


def _responses_tools():
    """The tools as a Codex-like client declares them, each namespace as one tool."""
    tools, namespaces = [], {}
    for namespace, name, description, parameters in _DECLARED:
        tool = (
            {"type": "custom", "name": name}
            if parameters is None
            else {"type": "function", "name": name, "parameters": parameters, "strict": False}
        )
        if description is not None:
            tool["description"] = description
        if namespace is None:
            tools.append(tool)
            continue
        if namespace not in namespaces:
            namespaces[namespace] = {
                "type": "namespace",
                "name": namespace,
                "description": _NAMESPACES[namespace],
                "tools": [],
            }
            tools.append(namespaces[namespace])
        namespaces[namespace]["tools"].append(tool)
    return tools


def _checkpoint_tools():
    """The same tools as the checkpoint's encoder takes them: namespaces beside functions."""
    (freeform,) = _get_chat_completion_function_tools(
        [SimpleNamespace(type="custom", name="any", description=None)]
    )
    tools = []
    for namespace, name, description, parameters in _DECLARED:
        # Every function the server hands over carries `strict`, unset, and the
        # encoder renders every key it is given.
        function = {
            "name": name,
            "description": description,
            "parameters": freeform.function.parameters if parameters is None else parameters,
            "strict": None,
        }
        tool = {"type": "function", "function": function}
        if namespace is not None:
            tool["namespace"] = {"name": namespace, "description": _NAMESPACES[namespace]}
        tools.append(tool)
    return tools


def _request(*items, tools=None, thinking=False):
    return ResponsesRequest(
        model="m",
        instructions=INSTRUCTIONS,
        tools=_responses_tools() if tools is None else tools,
        input=[USER, *items],
        reasoning={"effort": "high"} if thinking else None,
        store=False,
    )


class _Recorder:
    """Stands in for the checkpoint's tokenizer and keeps the prompt it encodes."""

    all_special_tokens = []
    eos_token_id = 1
    pad_token_id = 0
    name_or_path = "recorder"

    def encode(self, text, *args, **kwargs):
        self.prompt = text
        return [0]


def _render(request, model_type=MODEL_TYPE):
    recorder = _Recorder()
    asyncio.run(
        _create_input_tokens(
            request=request,
            prev_response=None,
            prev_msgs=[],
            conversation_store=None,
            enable_store=False,
            tokenizer=DeepseekV41Tokenizer(recorder),
            model_config=SimpleNamespace(model_type=model_type),
            processor=None,
        )
    )
    return recorder.prompt


def _tool_schemas(prompt):
    block = prompt.split("### Available Tool Schemas\n\n", 1)[1].split("\n\nYou MUST", 1)[0]
    return [json.loads(line) for line in block.splitlines()]


def test_namespaced_tools_are_shown_in_the_checkpoint_form():
    schemas = _tool_schemas(_render(_request()))

    assert [schema["name"] for schema in schemas] == [
        name if namespace is None else f"{namespace}::{name}" for namespace, name, _, _ in _DECLARED
    ]
    descriptions = {schema["name"]: schema["description"] for schema in schemas}
    assert descriptions["exec_command"] == "Runs a command."
    assert descriptions["multi_agent_v1::spawn_agent"] == (
        "Tools for working with sub-agents.\nSpawns a sub-agent."
    )
    assert descriptions["multi_agent_v1::wait"] == "Tools for working with sub-agents.\n"
    # An empty namespace description adds nothing.
    assert descriptions["functions::exec"] == "Runs code."


def test_a_request_without_namespaces_renders_as_on_the_generic_path():
    """Only namespaced tools change; the generic path is a model with no extension."""
    flat_tools = [tool for tool in _responses_tools() if tool["type"] != "namespace"]
    history = [
        {"type": "function_call", "call_id": "c1", "name": "exec_command", "arguments": "{}"},
        {"type": "function_call_output", "call_id": "c1", "output": "ok"},
    ]

    for thinking in (False, True):
        flat = _request(*history, tools=flat_tools, thinking=thinking)
        assert _render(flat) == _render(flat, model_type="generic")
    assert _render(_request()) != _render(_request(), model_type="generic")


def _generated(calls, thinking):
    """A turn as the model writes it, in the checkpoint encoder's layout."""
    invokes = []
    for name, arguments in calls:
        parameters = "\n".join(
            f'<｜DSML｜ parameter name="{key}" string="{str(isinstance(value, str)).lower()}">'
            f"{value if isinstance(value, str) else json.dumps(value)}</｜DSML｜ parameter>"
            for key, value in arguments.items()
        )
        invokes.append(f'<｜DSML｜ invoke name="{name}">\n{parameters}\n</｜DSML｜ invoke>')
    calls_block = "<｜DSML｜ calls>\n" + "\n".join(invokes) + "\n</｜DSML｜ calls>"
    return (REASONING + "</think>" if thinking else "") + "\n\n" + calls_block + EOS


# Each turn: the calls as the model writes them, and as the client echoes them
# back - the name and namespace the server reported, and the type of the tool.
_TURNS = {
    "namespaced_function": [
        ("multi_agent_v1::spawn_agent", {"message": "Check the tests."}, "multi_agent_v1", False)
    ],
    "namespaced_custom_tool": [("functions::exec", {"input": "print(1)\n"}, "functions", True)],
    "two_calls": [
        ("exec_command", {"cmd": "ls"}, None, False),
        ("multi_agent_v1::wait", {"ids": ["a", "b"]}, "multi_agent_v1", False),
    ],
}


def _echo(written_name, arguments, namespace, custom, call_id):
    call = {"call_id": call_id, "name": written_name.rpartition("::")[2]}
    if namespace is not None:
        call["namespace"] = namespace
    if custom:
        return [
            {"type": "custom_tool_call", "input": arguments["input"], **call},
            {"type": "custom_tool_call_output", "call_id": call_id, "output": "done"},
        ]
    return [
        {"type": "function_call", "arguments": arguments, **call},
        {"type": "function_call_output", "call_id": call_id, "output": "done"},
    ]


def _next_turn(turn, generated, tools, thinking):
    """The next request: the generated turn as the client sends it back."""
    parsed = DeepSeekV41Parser().detect_and_parse(generated, tools)
    # The parser keeps the name the model wrote.
    assert [call.name for call in parsed.calls] == [call[0] for call in turn]
    calls, outputs = [], []
    for index, ((name, _, namespace, custom), parsed_call) in enumerate(zip(turn, parsed.calls)):
        arguments = json.loads(parsed_call.parameters) if custom else parsed_call.parameters
        call, output = _echo(name, arguments, namespace, custom, f"call_{index}")
        calls.append(call)
        outputs.append(output)
    reasoning = (
        [{"type": "reasoning", "content": [{"type": "reasoning_text", "text": REASONING}]}]
        if thinking
        else []
    )
    return _request(*reasoning, *calls, *outputs, thinking=thinking)


@pytest.mark.parametrize("thinking", [False, True], ids=["chat", "thinking"])
@pytest.mark.parametrize("turn", list(_TURNS), ids=list(_TURNS))
def test_a_replayed_turn_is_what_the_model_generated(turn, thinking):
    """The next prompt continues this one with the generated text, byte for byte.

    A call to a namespaced tool comes back from the client as a name and a
    namespace, and it must be replayed under the name the model wrote,
    ``namespace::name``. Otherwise the history differs from what the model
    generated, and prefix-cache reuse stops at the call.
    """
    first = _request(thinking=thinking)
    prompt = _render(first)
    generated = _generated([(name, arguments) for name, arguments, _, _ in _TURNS[turn]], thinking)
    tools = _get_chat_completion_function_tools(first.tools)

    next_prompt = _render(_next_turn(_TURNS[turn], generated, tools, thinking))

    assert next_prompt.startswith(prompt + generated)


@pytest.mark.parametrize("thinking", [False, True], ids=["chat", "thinking"])
def test_matches_the_checkpoint_encoder(thinking):
    """The served prompt is the checkpoint encoder's, tools and replayed calls included."""
    vendor = _load_vendor_encoding()
    turn = _TURNS["namespaced_function"] + _TURNS["namespaced_custom_tool"]
    generated = _generated([(name, arguments) for name, arguments, _, _ in turn], thinking)
    request = _next_turn(
        turn, generated, _get_chat_completion_function_tools(_request().tools), thinking
    )

    calls = [
        {
            "id": f"call_{index}",
            "type": "function",
            "namespace": namespace,
            "function": {"name": name.rpartition("::")[2], "arguments": json.dumps(arguments)},
        }
        for index, (name, arguments, namespace, _) in enumerate(turn)
    ]
    assistant = {"role": "assistant", "content": "", "tool_calls": calls}
    if thinking:
        assistant["reasoning_content"] = REASONING
    messages = [
        {"role": "system", "content": INSTRUCTIONS, "tools": _checkpoint_tools()},
        {"role": "user", "content": "Fix it."},
        assistant,
        *[{"role": "tool", "tool_call_id": call["id"], "content": "done"} for call in calls],
    ]
    expected = vendor.encode_messages(
        messages,
        thinking_mode="thinking" if thinking else "chat",
        reasoning_effort="high" if thinking else None,
    )

    assert _render(request) == expected


def test_replay_keeps_the_dotted_name_when_the_namespace_is_no_longer_declared():
    flat_tools = [tool for tool in _responses_tools() if tool["type"] != "namespace"]
    request = _request(
        {"type": "function_call", "call_id": "c1", "name": "functions.do_it", "arguments": "{}"},
        {"type": "function_call_output", "call_id": "c1", "output": "done"},
        tools=flat_tools,
    )

    assert '<｜DSML｜ invoke name="functions.do_it">' in _render(request)
