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
"""Offline tests for replaying reasoning on the Responses API.

A client that keeps no state on the server replays the reasoning of earlier
turns in ``input``. The Codex CLI asks for
``include: ["reasoning.encrypted_content"]`` and sends each reasoning item back
with that value alone, so the server writes the reasoning text into
``encrypted_content`` and reads it back from there. A turn's items (reasoning,
messages, calls) are folded back into the one assistant message the model
generated, so a template that renders earlier reasoning - DeepSeek-V4.1 once
tools are declared - reproduces the prompt the model generated from.
"""

import asyncio
import base64
import json
from types import SimpleNamespace

import pytest
from openai.types.responses.tool import FunctionTool

from tensorrt_llm.inputs.utils import MultimodalDataTracker, apply_chat_template
from tensorrt_llm.serve.chat_utils import parse_chat_message_content
from tensorrt_llm.serve.openai_protocol import ResponsesRequest
from tensorrt_llm.serve.responses_utils import (
    ResponsesStreamingProcessor,
    _create_input_messages,
    _get_chat_completion_function_tools,
    _response_output_item_to_chat_completion_message,
    create_response_non_store,
    reasoning_chat_template_kwargs,
)
from tensorrt_llm.tokenizer.deepseek_v41 import DeepseekV41Tokenizer

# The CPU-* CI stages run pytest with -m 'cpu_only'. Without this marker every
# test in the file is deselected, which pytest reports as exit code 5 and the
# stage reports as a failure.
pytestmark = pytest.mark.cpu_only

INCLUDE = ["reasoning.encrypted_content"]
PREFIX = "trtllm-reasoning-v1:"
# Edge whitespace, a blank line and non-ASCII text: the model generated all of
# it, so all of it has to come back.
REASONING = "  The user wants a listing.\n\nRun `ls` — then report.\n"
TEXT = "Listing now."


def _carrier(text):
    return PREFIX + base64.b64encode(text.encode("utf-8")).decode("ascii")


def _request(stream=True, include=INCLUDE, tools=(), input="hi"):
    # reasoning.effort turns thinking on, which the DeepSeek-V4.1 reasoning
    # parser and renderer both need.
    return ResponsesRequest(
        model="m",
        input=input,
        stream=stream,
        store=False,
        include=include,
        reasoning={"effort": "high"},
        tools=list(tools),
    )


def _generation(accumulated, diff, done):
    output = SimpleNamespace(
        index=0,
        text=accumulated,
        text_diff=diff,
        finish_reason="stop" if done else None,
        token_ids=[1, 2, 3],
        disaggregated_params=None,
    )
    return SimpleNamespace(outputs=[output], _done=done, prompt_token_ids=[1, 2], cached_tokens=0)


def _streamed_turn(request, chunks):
    """The items of one streamed DeepSeek-V4.1 turn: (output_item.done items, final output)."""
    processor = ResponsesStreamingProcessor(
        request=request,
        sampling_params=request.to_sampling_params(),
        model_name="m",
        use_harmony=False,
        reasoning_parser="deepseek_v41",
    )
    frames, accumulated, result = [], "", None
    chunks = [chunk for chunk in chunks if chunk]
    for i, chunk in enumerate(chunks):
        accumulated += chunk
        result = _generation(accumulated, chunk, i == len(chunks) - 1)
        frames.extend(processor.process_single_output(result))
    frames.append(processor.get_final_response_non_store(result))
    events = [json.loads(frame.split("data: ", 1)[1]) for frame in frames]
    assert events[-1]["type"] == "response.completed"
    done = [e["item"] for e in events if e["type"] == "response.output_item.done"]
    return done, events[-1]["response"]["output"]


def _reasoning(items):
    return [item for item in items if item["type"] == "reasoning"]


# ---------------------------------------------------------------------------
# The server writes the carrier when asked
# ---------------------------------------------------------------------------


def test_streamed_and_final_reasoning_items_carry_the_same_carrier():
    streamed, final = _streamed_turn(_request(), [REASONING[:9], REASONING[9:], "</think>", TEXT])

    (streamed_item,) = _reasoning(streamed)
    (final_item,) = _reasoning(final)
    assert streamed_item["content"][0]["text"] == REASONING
    # The format is a contract with every client holding an earlier carrier.
    assert streamed_item["encrypted_content"] == _carrier(REASONING)
    assert final_item["id"] == streamed_item["id"]
    assert final_item["encrypted_content"] == streamed_item["encrypted_content"]


def test_a_non_streamed_response_carries_the_carrier():
    request = _request(stream=False)
    response = create_response_non_store(
        generation_result=_generation(REASONING + "</think>" + TEXT, None, True),
        request=request,
        sampling_params=request.to_sampling_params(),
        model_name="m",
        use_harmony=False,
        reasoning_parser="deepseek_v41",
        num_prompt_tokens=2,
    ).model_dump()

    (item,) = _reasoning(response["output"])
    assert item["encrypted_content"] == _carrier(REASONING)


@pytest.mark.parametrize("include", [None, [], ["message.output_text.logprobs"]])
def test_no_carrier_unless_the_request_asks_for_it(include):
    streamed, final = _streamed_turn(_request(include=include), [REASONING, "</think>", TEXT])

    reasoning = _reasoning(streamed) + _reasoning(final)
    assert len(reasoning) == 2
    assert all(item["encrypted_content"] is None for item in reasoning)


# ---------------------------------------------------------------------------
# A replayed carrier restores the reasoning; anything else is not this
# model's reasoning
# ---------------------------------------------------------------------------


def _replayed_reasoning(encrypted_content, **fields):
    """A reasoning item as the Codex CLI replays it: no id and no content."""
    return {"type": "reasoning", "summary": [], "encrypted_content": encrypted_content, **fields}


def _user(text):
    return {"type": "message", "role": "user", "content": [{"type": "input_text", "text": text}]}


def _messages(request):
    return asyncio.run(_create_input_messages(request=request, prev_msgs=[]))


def test_a_replayed_carrier_restores_the_reasoning_verbatim():
    assert _response_output_item_to_chat_completion_message(
        _replayed_reasoning(_carrier(REASONING))
    ) == {"role": "assistant", "reasoning": REASONING}


def test_the_carrier_is_preferred_to_readable_content():
    item = _replayed_reasoning(
        _carrier(REASONING), content=[{"type": "reasoning_text", "text": REASONING.strip()}]
    )
    assert _response_output_item_to_chat_completion_message(item)["reasoning"] == REASONING


_FOREIGN = "gAAAAABo" + "Zm9vYmFy" * 12


@pytest.mark.parametrize(
    "item",
    [
        _replayed_reasoning(_FOREIGN),
        _replayed_reasoning(
            _FOREIGN, content=[{"type": "reasoning_text", "text": "another model's thought"}]
        ),
        _replayed_reasoning(
            _FOREIGN, summary=[{"type": "summary_text", "text": "another model's summary"}]
        ),
        _replayed_reasoning(PREFIX + "not base64!"),
        _replayed_reasoning(_carrier(REASONING).replace(PREFIX, PREFIX + "*")),
        _replayed_reasoning(PREFIX + "café"),
        _replayed_reasoning(PREFIX + base64.b64encode(b"\xff\xfe").decode("ascii")),
        {"type": "reasoning", "summary": []},
    ],
    ids=[
        "foreign",
        "foreign_with_content",
        "foreign_with_summary",
        "prefix_not_base64",
        "prefix_altered",
        "prefix_not_ascii",
        "prefix_not_utf8",
        "nothing_readable",
    ],
)
def test_reasoning_this_server_did_not_write_is_not_replayed(item):
    assert _response_output_item_to_chat_completion_message(item) is None

    messages = _messages(_request(input=[_user("q"), item, _user("next")]))
    assert [m["role"] for m in messages] == ["user", "user"]


# ---------------------------------------------------------------------------
# Round trip: emitted, replayed, rendered
# ---------------------------------------------------------------------------


def _codex_replay(items):
    """The items a client that replays by encrypted_content sends back."""
    replay = []
    for item in items:
        if item["type"] == "reasoning":
            replay.append(_replayed_reasoning(item["encrypted_content"]))
        elif item["type"] == "message":
            replay.append(
                {
                    "type": "message",
                    "role": "assistant",
                    "id": item["id"],
                    "status": "completed",
                    "content": [
                        {"type": "output_text", "text": part["text"], "annotations": []}
                        for part in item["content"]
                    ],
                }
            )
    return replay


def _conversation(messages):
    """The conversation handed to the chat template.

    For text-only messages this is what parse_chat_messages_coroutines builds,
    without the multimodal registry lookup by model type.
    """
    return [
        parse_chat_message_content(message, MultimodalDataTracker("deepseek_v41"))
        for message in messages
    ]


def test_reasoning_round_trips_into_the_conversation_the_template_gets():
    streamed, _ = _streamed_turn(_request(), [REASONING, "</think>", TEXT])

    messages = _messages(_request(input=[_user("q"), *_codex_replay(streamed), _user("next")]))

    conversation = _conversation(messages)
    assert [m["role"] for m in conversation] == ["user", "assistant", "user"]
    assert conversation[1]["reasoning_content"] == REASONING
    assert conversation[1]["content"] == TEXT


_TOOLS = [
    FunctionTool(
        type="function",
        name=name,
        description=f"{name} tool",
        parameters={"type": "object", "properties": {arg: {"type": "string"}}},
        strict=False,
    )
    for name, arg in (("shell", "cmd"), ("read_file", "path"))
]


def _chat_call(call_id, name, arguments):
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments)},
    }


# Each turn as the model generated it: one assistant message.
_TURNS = {
    "reasoning_text_call": {
        "reasoning_content": REASONING,
        "content": TEXT,
        "tool_calls": [_chat_call("call_1", "shell", {"cmd": "ls"})],
    },
    "reasoning_call_call": {
        "reasoning_content": REASONING,
        "content": "",
        "tool_calls": [
            _chat_call("call_1", "shell", {"cmd": "ls"}),
            _chat_call("call_2", "read_file", {"path": "a.txt"}),
        ],
    },
    "reasoning_text": {"reasoning_content": REASONING, "content": TEXT, "tool_calls": []},
}


class _NoTokenizer:
    """Stands in for the checkpoint tokenizer: prompts are compared as text."""

    all_special_tokens = []

    def encode(self, text, *args, **kwargs):
        raise AssertionError("the prompt is compared as text")


def _render_v41(request, messages):
    """Render as the Responses path does for DeepSeek-V4.1, as text."""
    conversation = _conversation(messages)
    return apply_chat_template(
        model_type="deepseek_v41",
        tokenizer=DeepseekV41Tokenizer(_NoTokenizer()),
        processor=None,
        conversation=conversation,
        add_generation_prompt=True,
        mm_placeholder_counts=[{} for _ in conversation],
        tools=[tool.model_dump() for tool in _get_chat_completion_function_tools(request.tools)],
        chat_template_kwargs=reasoning_chat_template_kwargs(request),
    )


@pytest.mark.parametrize("name", sorted(_TURNS))
def test_a_replayed_turn_renders_as_the_turn_the_model_generated(name):
    turn = _TURNS[name]
    question = _request(input=[_user("list the files")], tools=_TOOLS)
    prompt = _render_v41(question, _messages(question))

    # The server streams the turn's reasoning and text; its calls are
    # reported as items of their own.
    streamed, _ = _streamed_turn(
        _request(), [turn["reasoning_content"], "</think>", turn["content"]]
    )
    calls = [
        {
            "type": "function_call",
            "call_id": call["id"],
            "name": call["function"]["name"],
            "arguments": call["function"]["arguments"],
        }
        for call in turn["tool_calls"]
    ]
    results = [
        {"type": "function_call_output", "call_id": call["id"], "output": "ok"}
        for call in turn["tool_calls"]
    ] or [_user("thanks")]
    replay = _request(
        input=[_user("list the files"), *_codex_replay(streamed), *calls, *results], tools=_TOOLS
    )
    replayed_prompt = _render_v41(replay, _messages(replay))

    generated = {"role": "assistant", **turn}
    if not generated["tool_calls"]:
        del generated["tool_calls"]
    result_messages = [
        {"role": "tool", "content": "ok", "tool_call_id": call["id"]} for call in turn["tool_calls"]
    ] or [{"role": "user", "content": "thanks"}]
    original_prompt = _render_v41(
        replay, [{"role": "user", "content": "list the files"}, generated, *result_messages]
    )

    assert replayed_prompt == original_prompt
    # The previous prompt and what the model generated after it are a prefix
    # of the next prompt.
    assert replayed_prompt.startswith(
        prompt + turn["reasoning_content"] + "</think>" + turn["content"]
    )
