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

import asyncio
import json

import pytest

from tensorrt_llm.executor import EngineDeadError, RequestError
from tensorrt_llm.serve.openai_protocol import ResponsesRequest
from tensorrt_llm.serve.responses_utils import (
    ResponsesStreamingEventsHelper,
    ResponsesStreamingProcessor,
    _generate_streaming_event,
    classify_stream_termination,
    guard_responses_stream,
    stream_error_event,
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
# Streams that stop before response.completed
#
# response.completed is the only event that repeats the full text, so a stream
# cut before it leaves everything the turn produced living solely in deltas.
# Until these events existed such a stream simply stopped: no error event, no
# populated response.error, nothing in the bytes or the trace to tell a reader
# the stream was cut rather than still running.
# ---------------------------------------------------------------------------


def _event_type(frame):
    if isinstance(frame, bytes):
        frame = frame.decode()
    return frame.split("\n", 1)[0][len("event: ") :]


def _event_data(frame):
    if isinstance(frame, bytes):
        frame = frame.decode()
    return json.loads(frame.split("data: ", 1)[1])


def _processor():
    """A processor built exactly as the server builds one."""
    request = ResponsesRequest(model="test-model", input="hi", stream=True)
    return ResponsesStreamingProcessor(
        request=request,
        sampling_params=request.to_sampling_params(),
        model_name="test-model",
        use_harmony=False,
    )


def _forbidden(cause, detail, events_sent):
    raise AssertionError(f"terminal events were built for a stream that did not need them: {cause}")


async def _drive(source, terminal_events, on_termination=None):
    """Run a stream through the guard, returning (frames, raised exception)."""
    seen = []
    try:
        async for chunk in guard_responses_stream(source, terminal_events, on_termination):
            seen.append(chunk)
    except BaseException as exc:  # noqa: BLE001 - the test asserts on it
        return seen, exc
    return seen, None


@pytest.mark.asyncio
async def test_a_completed_stream_is_forwarded_unchanged():
    """The happy path must stay byte-identical: same frames, nothing appended."""
    frames = [
        "event: response.created\ndata: {}\n\n",
        "event: response.output_text.delta\ndata: {}\n\n",
        "event: response.completed\ndata: {}\n\n",
    ]

    async def source():
        for frame in frames:
            yield frame

    seen, raised = await _drive(source(), _forbidden)
    assert seen == frames
    assert raised is None


@pytest.mark.asyncio
async def test_an_exception_mid_stream_ends_the_stream_with_a_terminal_event():
    processor = _processor()
    opening = processor.get_initial_responses()

    async def source():
        for frame in opening:
            yield frame
        raise RuntimeError("event construction blew up")

    seen, raised = await _drive(source(), processor.get_stream_failed_events)

    assert [_event_type(frame) for frame in seen] == [
        "response.created",
        "response.in_progress",
        "error",
        "response.failed",
    ]
    # Re-raised, not swallowed: the server still logs the fault and the trace
    # still records the response as an error rather than a clean finish.
    assert isinstance(raised, RuntimeError)


@pytest.mark.asyncio
async def test_the_terminal_event_carries_the_cause():
    processor = _processor()

    async def source():
        for frame in processor.get_initial_responses():
            yield frame
        raise ValueError("item had no id")

    seen, _ = await _drive(source(), processor.get_stream_failed_events)
    error = _event_data(seen[-2])
    assert error["type"] == "error"
    assert error["code"] == "internal_error"
    assert error["message"] == "ValueError: item had no id"


@pytest.mark.parametrize(
    "exc, cause",
    [
        (RequestError("request failed"), "engine_error"),
        (EngineDeadError(), "engine_error"),
        (ValueError("bad item"), "internal_error"),
        (asyncio.CancelledError(), "client_disconnect"),
        (GeneratorExit(), "client_disconnect"),
    ],
)
def test_causes_the_code_can_tell_apart(exc, cause):
    """An engine failure, a fault in this process, and a client hangup differ.

    Only the middle one leaves the accumulated text in memory, which is what
    makes the distinction worth recording rather than lumping into "error".
    """
    assert classify_stream_termination(exc) == cause


def test_an_upstream_transport_error_is_named_as_one():
    aiohttp = pytest.importorskip("aiohttp")
    assert classify_stream_termination(aiohttp.ClientPayloadError("cut")) == "upstream_error"


@pytest.mark.asyncio
async def test_a_client_hangup_adds_nothing_to_the_stream():
    """Nobody is left to send to, and yielding here would raise.

    An async generator that yields while GeneratorExit is propagating dies
    with "async generator ignored GeneratorExit", trading a diagnosable
    truncation for an undiagnosable one. The cause is reported out of band.
    """
    noted = []

    async def source():
        yield "event: response.created\ndata: {}\n\n"
        raise asyncio.CancelledError

    seen, raised = await _drive(
        source(), _forbidden, lambda cause, detail: noted.append((cause, detail))
    )
    assert len(seen) == 1
    assert isinstance(raised, asyncio.CancelledError)
    assert noted == [("client_disconnect", "CancelledError")]


@pytest.mark.asyncio
async def test_a_broken_reporter_does_not_replace_the_original_fault():
    """Reporting a failure must not become a second, more confusing one."""

    def broken(cause, detail, events_sent):
        raise KeyError("snapshot field missing")

    async def source():
        yield "event: response.created\ndata: {}\n\n"
        raise ValueError("the fault worth seeing")

    seen, raised = await _drive(source(), broken)
    assert len(seen) == 1
    assert isinstance(raised, ValueError)
    assert str(raised) == "the fault worth seeing"


@pytest.mark.asyncio
async def test_a_failure_after_completion_adds_nothing():
    """Regression guard for the happy path.

    aiohttp raises at end of stream when the connector closes the connection,
    which arrives after the last frame. Appending an error event there would
    corrupt a stream that completed perfectly well.
    """

    async def source():
        yield "event: response.completed\ndata: {}\n\n"
        raise RuntimeError("connection reset at end of stream")

    seen, raised = await _drive(source(), _forbidden)
    assert len(seen) == 1
    assert isinstance(raised, RuntimeError)


def test_terminal_events_continue_the_sequence_numbering():
    processor = _processor()
    opening = processor.get_initial_responses()
    assert [_event_data(frame)["sequence_number"] for frame in opening] == [0, 1]

    events = processor.get_stream_failed_events("internal_error", "ValueError: x")
    assert [_event_data(frame)["sequence_number"] for frame in events] == [2, 3]


def test_numbering_follows_events_this_processor_did_not_number():
    """With postprocessing workers the per-token events are numbered elsewhere.

    A pickled copy of this processor builds them in another process, so the
    local counter only ever saw the opening two and would put the terminal
    events back at the start of a sequence the client has already passed.
    """
    processor = _processor()
    processor.get_initial_responses()

    events = processor.get_stream_failed_events("internal_error", "ValueError: x", 610)
    assert [_event_data(frame)["sequence_number"] for frame in events] == [610, 611]


def test_the_failed_snapshot_says_failed_and_carries_no_content():
    """response.failed, not response.incomplete.

    incomplete_details.reason is a closed enum -- max_output_tokens,
    max_messages, content_filter, steered -- with no member for a server fault
    or a client hangup, so response.incomplete could only be sent with
    reason=null, which is indistinguishable from an ordinary truncation.
    """
    processor = _processor()
    processor.get_initial_responses()
    _, failed = processor.get_stream_failed_events("internal_error", "ValueError: x")

    assert _event_type(failed) == "response.failed"
    response = _event_data(failed)["response"]
    assert response["status"] == "failed"
    assert response["error"]["code"] == "server_error"
    assert response["error"]["message"] == "ValueError: x"
    # No reconstruction: the lost text stays lost, this only says so.
    assert response["output"] == []
    assert response["id"] == processor.request.request_id


@pytest.mark.asyncio
async def test_the_relayed_terminal_event_is_numbered_from_the_frames_forwarded():
    """The orchestrator forwards bytes and never sees the events it carries.

    It can still count them, including when a transport read splits the blank
    line that ends one -- without the carry every sequence number after such a
    split is one short.
    """

    async def source():
        yield b"event: response.created\ndata: {}\n\n"
        yield b"event: response.in_progress\ndata: {}\n"
        yield b"\nevent: response.output_text.delta\ndata: {}\n\n"
        raise RuntimeError("upstream cut")

    seen, raised = await _drive(source(), stream_error_event)
    assert isinstance(raised, RuntimeError)
    terminal = seen[-1]
    assert isinstance(terminal, bytes)
    assert _event_type(terminal) == "error"
    assert _event_data(terminal)["sequence_number"] == 3


@pytest.mark.asyncio
async def test_a_relay_that_never_started_reports_sequence_zero():
    """The failures that record a response with an entirely empty body.

    Nothing was forwarded, so the error event is the stream's first and the
    sequence number is exactly right rather than a guess.
    """

    async def source():
        raise RuntimeError("no context worker available")
        yield b""  # pragma: no cover - makes this an async generator

    seen, raised = await _drive(source(), stream_error_event)
    assert isinstance(raised, RuntimeError)
    assert len(seen) == 1
    assert _event_data(seen[0])["sequence_number"] == 0
    assert _event_data(seen[0])["code"] == "internal_error"


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
