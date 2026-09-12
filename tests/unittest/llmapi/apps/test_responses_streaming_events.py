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
"""Offline tests for Responses API streaming event construction.

Clients track streamed content by output item: a delta is attached to the
item whose id it carries, and an item is only usable once it has been opened
with response.output_item.added and closed with response.output_item.done.
Codex CLI drops an entire turn - printing nothing - if a delta arrives with
an id it has not seen opened.
"""

import pytest

from tensorrt_llm.serve.responses_utils import (
    ResponsesStreamingEventsHelper,
    _generate_streaming_event,
)

# The CPU-* CI stages run pytest with -m 'cpu_only'. Without this marker every
# test in the file is deselected, which pytest reports as exit code 5 and the
# stage reports as a failure.
pytestmark = pytest.mark.cpu_only


def _helper():
    return ResponsesStreamingEventsHelper()


# ---------------------------------------------------------------------------
# Item identity
# ---------------------------------------------------------------------------


def test_message_item_gets_a_non_empty_id():
    """Regression: current_item_id had no writer, so every event carried "".

    A delta whose item_id is empty matches no item the client has opened, so
    the client discards the text and the user sees no reply at all.
    """
    helper = _helper()
    list(helper.get_message_output_added_events())
    assert helper.item_id
    assert helper.item_id.startswith("msg_")


def test_reasoning_item_gets_a_non_empty_id():
    helper = _helper()
    list(helper.get_reasoning_output_added_events())
    assert helper.item_id.startswith("rs_")


def test_events_share_the_open_item_id():
    helper = _helper()
    added = list(helper.get_message_output_added_events())
    item_id = helper.item_id
    assert added[0].item.id == item_id
    assert added[1].item_id == item_id
    assert helper.get_text_delta_event("hello", []).item_id == item_id


def test_a_new_item_gets_a_new_id():
    """Regression: closing an item left item_id set, so the next item reused it.

    Two output items sharing one id makes the stream ambiguous for a client
    keying its state on item_id.
    """
    helper = _helper()
    list(helper.get_message_output_added_events())
    first = helper.item_id

    # What the close path does.
    helper.is_output_item_added_sent = False
    helper.output_index_increment()

    list(helper.get_message_output_added_events())
    assert helper.item_id != first
    assert helper.item_id.startswith("msg_")


def test_added_events_are_idempotent_while_an_item_is_open():
    """Deltas may call the opener every time; only the first must emit."""
    helper = _helper()
    first = list(helper.get_message_output_added_events())
    second = list(helper.get_message_output_added_events())
    assert len(first) == 2  # output_item.added + content_part.added
    assert second == []


# ---------------------------------------------------------------------------
# Text accumulation used to close an item at end of generation
# ---------------------------------------------------------------------------


def test_text_buffer_accumulates_and_drains():
    helper = _helper()
    helper.append_text("Hello")
    helper.append_text(" world")
    assert helper.take_text() == "Hello world"
    # Draining leaves nothing behind, so a later close cannot re-emit it.
    assert helper.take_text() == ""


def test_reasoning_buffer_is_separate_from_text():
    helper = _helper()
    helper.append_text("answer")
    helper.append_reasoning("thinking")
    assert helper.take_reasoning() == "thinking"
    assert helper.take_text() == "answer"


def test_done_events_carry_the_open_item_id():
    helper = _helper()
    list(helper.get_message_output_added_events())
    item_id = helper.item_id
    assert helper.get_text_done_event("hi", []).item_id == item_id
    assert helper.get_content_part_done_event(_output_text("hi")).item_id == item_id


# ---------------------------------------------------------------------------
# Reasoning that shares a chunk with the answer
# ---------------------------------------------------------------------------


class _FakeOutput:
    """The attributes the streaming path reads from a generation output.

    `text` is the whole generation so far and `text_diff` only the newest
    chunk: the delta path parses the diff, while the done-event path re-parses
    the accumulated text. Supplying only the diff makes the second one raise,
    which is how the first version of this fake was wrong.
    """

    def __init__(self, text, text_diff, index=0):
        self.text = text
        self.text_diff = text_diff
        self.index = index


class _FakeRequest:
    tools = None


def _stream(chunks):
    """Drive the real dispatch over a sequence of generation chunks.

    Returns (reasoning deltas, text deltas) as the client would receive them.
    """
    helper = ResponsesStreamingEventsHelper()
    parsers = {}
    reasoning, text = [], []
    accumulated = ""
    for i, chunk in enumerate(chunks):
        accumulated += chunk
        events = _generate_streaming_event(
            output=_FakeOutput(accumulated, chunk),
            request=_FakeRequest(),
            finished_generation=(i == len(chunks) - 1),
            streaming_events_helper=helper,
            reasoning_parser_id="glm",
            reasoning_parser_dict=parsers,
        )
        for event in events:
            kind = getattr(event, "type", "")
            if kind == "response.reasoning_text.delta":
                reasoning.append(event.delta)
            elif kind == "response.output_text.delta":
                text.append(event.delta)
    return "".join(reasoning), "".join(text)


def test_reasoning_sharing_a_chunk_with_the_answer_is_not_dropped():
    """Regression: the reasoning half of the straddling chunk was discarded.

    A generation chunk whose raw text spans the closing think tag parses into
    both a reasoning part and a content part. The dispatch chose its branch on
    the content part alone, so the reasoning part fell into a branch that never
    ran and was never emitted -- and closing the item cleared the flag the
    done-event path is guarded on, so the done event carried the short text
    too. Measured at 56% of reasoning items across four fleets.
    """
    reasoning, text = _stream(["Let", " me check the reference.</think>I will read the file."])
    assert reasoning == "Let me check the reference."
    assert text == "I will read the file."


def test_reasoning_shorter_than_one_chunk_survives():
    """The worst case: the whole reasoning shares a chunk with the answer.

    Reasoning short enough to fit inside a single chunk was truncated to
    nothing at all, which is why the losses looked like first words.
    """
    reasoning, _ = _stream(["Sure.</think>Here it is."])
    assert reasoning == "Sure."


def test_reasoning_ending_on_a_chunk_boundary_is_unaffected():
    """The case that always worked, kept so a fix cannot regress it."""
    reasoning, text = _stream(
        ["Let", " me check the reference.", "</think>", "I will read the file."]
    )
    assert reasoning == "Let me check the reference."
    assert text == "I will read the file."


def test_reasoning_with_no_answer_after_it_is_unaffected():
    """A turn that ends in a tool call emits no text, so the other branch ran."""
    reasoning, text = _stream(["Let", " me check the reference.</think>"])
    assert reasoning == "Let me check the reference."
    assert text == ""


def _output_text(text):
    from openai.types.responses import ResponseOutputText

    return ResponseOutputText(text=text, annotations=[], type="output_text", logprobs=None)


# ---------------------------------------------------------------------------
# Order of the items in the completed-response snapshot
#
# These cover the non-streaming assembly in `_create_output_content`, but they
# live here because what they assert is a property of the stream: the snapshot
# in `response.output` has to list the items in the order the stream emitted
# them. A client that reads both must not be told two different stories about
# one generation. This is also the file the CPU CI stage runs (l0_cpu.yml).
# ---------------------------------------------------------------------------

_THINK = "Check the reference first."
_ANSWER = "The answer is 42."
_TOOL_CALL = '<tool_call>\n{"name": "read_file", "arguments": {"path": "a.txt"}}\n</tool_call>'


def _tools():
    """A real tool definition, so the tool parser accepts the parsed call."""
    from openai.types.responses.tool import FunctionTool

    return [
        FunctionTool(
            name="read_file",
            description="Read a file.",
            parameters={"type": "object", "properties": {"path": {"type": "string"}}},
            strict=False,
            type="function",
        )
    ]


def _snapshot_kinds(text, tools=None):
    """Item types of the completed-response snapshot for one generation.

    Drives the real `_create_output_content` with real parsers - `qwen3` for
    both, so text with no `<think>` is an ordinary answer rather than an
    unterminated reasoning block.
    """
    from tensorrt_llm.serve.responses_utils import _create_output_content

    items, _messages = _create_output_content(
        _FakeRequestOutput(text),
        reasoning_parser="qwen3",
        tool_parser="qwen3" if tools else None,
        tools=tools,
    )
    return items, [item.type for item in items]


class _FakeRequestOutput:
    """The one attribute `_create_output_content` reads off a result.

    It iterates `final_res.outputs` and takes `.index` and `.text` from each;
    everything else it needs it derives from the parsers.
    """

    def __init__(self, text):
        self.outputs = [_FakeOutput(text, text)]


def test_snapshot_lists_reasoning_before_the_message():
    """Regression: the snapshot appended the message first, inverting cause and effect.

    The stream emits reasoning then message, but the snapshot for the same
    generation listed message then reasoning - so a client replaying
    `response.output` read the answer before the reasoning that produced it.
    Measured on 99.99% of responses carrying both items across four fleets.
    """
    items, kinds = _snapshot_kinds(f"<think>{_THINK}</think>{_ANSWER}")
    assert kinds == ["reasoning", "message"]
    # Reordering must move the items, not relabel them.
    assert items[0].content[0].text == _THINK
    assert items[1].content[0].text == _ANSWER


def test_snapshot_lists_reasoning_then_message_then_tool_call():
    items, kinds = _snapshot_kinds(
        f"<think>{_THINK}</think>Reading it now.\n{_TOOL_CALL}", tools=_tools()
    )
    assert kinds == ["reasoning", "message", "function_call"]
    assert items[2].name == "read_file"


def test_snapshot_lists_reasoning_before_a_tool_call_with_no_message():
    """A turn that ends in a tool call emits no message item at all."""
    _items, kinds = _snapshot_kinds(f"<think>{_THINK}</think>{_TOOL_CALL}", tools=_tools())
    assert kinds == ["reasoning", "function_call"]


def test_snapshot_of_a_plain_answer_is_just_the_message():
    """No reasoning to order: the message must not be dropped or displaced."""
    items, kinds = _snapshot_kinds(_ANSWER)
    assert kinds == ["message"]
    assert items[0].content[0].text == _ANSWER


def test_snapshot_order_matches_the_streamed_order():
    """The two paths must agree; this is the invariant the bug violated.

    Both sides are driven for real - `_generate_streaming_event` for the
    stream and `_create_output_content` for the snapshot - over the same
    generation and the same reasoning parser, rather than comparing the
    snapshot against a hardcoded expectation of what the stream does.
    """
    from tensorrt_llm.serve.responses_utils import (
        ResponsesStreamingEventsHelper,
        _create_output_content,
    )

    # "glm" is reasoning_at_start, so the generation opens inside the
    # reasoning block with no <think> tag - the same setup as the streaming
    # tests above, and what the thinking chat templates actually render.
    chunks = [_THINK, "</think>", _ANSWER]

    helper = ResponsesStreamingEventsHelper()
    parsers = {}
    accumulated = ""
    streamed = []
    for i, chunk in enumerate(chunks):
        accumulated += chunk
        for event in _generate_streaming_event(
            output=_FakeOutput(accumulated, chunk),
            request=_FakeRequest(),
            finished_generation=(i == len(chunks) - 1),
            streaming_events_helper=helper,
            reasoning_parser_id="glm",
            reasoning_parser_dict=parsers,
        ):
            if getattr(event, "type", "") == "response.output_item.done":
                streamed.append(event.item.type)

    snapshot_items, _messages = _create_output_content(
        _FakeRequestOutput(accumulated), reasoning_parser="glm"
    )

    assert streamed == ["reasoning", "message"]
    assert [item.type for item in snapshot_items] == streamed
