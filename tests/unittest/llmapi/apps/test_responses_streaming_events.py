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
from types import SimpleNamespace

import aiohttp
import pytest
from openai.types.responses.tool import FunctionTool

from tensorrt_llm.executor import EngineDeadError, RequestError
from tensorrt_llm.serve.openai_protocol import ResponsesRequest
from tensorrt_llm.serve.responses_utils import (
    ResponsesStreamingEventsHelper,
    ResponsesStreamingProcessor,
    _create_output_content,
    _generate_streaming_event,
    classify_stream_termination,
    describe_stream_termination,
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


def _output_text(text):
    from openai.types.responses import ResponseOutputText

    return ResponseOutputText(text=text, annotations=[], type="output_text", logprobs=None)


# ---------------------------------------------------------------------------
# Reasoning and answer deltas
# ---------------------------------------------------------------------------


def _event_stream(chunks, reasoning_parser="glm"):
    """All events the real dispatch emits for a sequence of chunks."""
    helper = ResponsesStreamingEventsHelper()
    request = SimpleNamespace(tools=None, tool_choice="auto")
    parsers = {}
    events, accumulated = [], ""
    for i, chunk in enumerate(chunks):
        accumulated += chunk
        events.extend(
            _generate_streaming_event(
                output=SimpleNamespace(index=0, text=accumulated, text_diff=chunk),
                request=request,
                finished_generation=i == len(chunks) - 1,
                streaming_events_helper=helper,
                reasoning_parser_id=reasoning_parser,
                reasoning_parser_dict=parsers,
            )
        )
    return events


def _deltas(events, event_type):
    return "".join(e.delta for e in events if e.type == event_type)


@pytest.mark.parametrize(
    "chunks, reasoning, answer",
    [
        (["Let", " me check.</think>I will read it."], "Let me check.", "I will read it."),
        (["Sure.</think>Here it is."], "Sure.", "Here it is."),
        (["Let", " me check.", "</think>", "I will read it."], "Let me check.", "I will read it."),
        (["Let", " me check.</think>"], "Let me check.", ""),
    ],
)
def test_reasoning_and_answer_deltas_survive_any_chunk_boundary(chunks, reasoning, answer):
    events = _event_stream(chunks)
    assert _deltas(events, "response.reasoning_text.delta") == reasoning
    assert _deltas(events, "response.output_text.delta") == answer


def test_an_item_is_opened_before_a_whitespace_only_first_delta():
    events = _event_stream(["Thinking.</think> ", "hi"])
    types = [e.type for e in events]
    message_added = next(
        i
        for i, e in enumerate(events)
        if e.type == "response.output_item.added" and e.item.type == "message"
    )
    assert message_added < types.index("response.output_text.delta")
    assert _deltas(events, "response.output_text.delta") == " hi"


@pytest.mark.parametrize(
    "chunks, reasoning",
    [
        (["Plan the fix.", "</think>", "Done."], "Plan the fix."),
        (["All reasoning, no answer."], "All reasoning, no answer."),
    ],
)
def test_every_content_part_closes_in_order(chunks, reasoning):
    """content_part.done comes between the text done event and item done."""
    events = _event_stream(chunks)
    types = [e.type for e in events]
    assert types.count("response.content_part.added") == types.count("response.content_part.done")

    for done_type, part_type, text in (
        ("response.reasoning_text.done", "reasoning_text", reasoning),
        ("response.output_text.done", "output_text", "Done."),
    ):
        if done_type not in types:
            continue
        index = types.index(done_type)
        assert types[index + 1 : index + 3] == [
            "response.content_part.done",
            "response.output_item.done",
        ]
        part_done, item_done = events[index + 1], events[index + 2]
        assert (part_done.part.type, part_done.part.text) == (part_type, text)
        assert part_done.item_id == item_done.item.id


# ---------------------------------------------------------------------------
# Snapshot item order
# ---------------------------------------------------------------------------

_THINK = "Check the reference first."
_ANSWER = "The answer is 42."
_TOOL_CALL = '<tool_call>\n{"name": "read_file", "arguments": {"path": "a.txt"}}\n</tool_call>'


def _read_file_tool():
    return FunctionTool(
        name="read_file",
        description="Read a file.",
        parameters={"type": "object", "properties": {"path": {"type": "string"}}},
        strict=False,
        type="function",
    )


@pytest.mark.parametrize(
    "text, kinds",
    [
        (f"<think>{_THINK}</think>{_ANSWER}", ["reasoning", "message"]),
        (
            f"<think>{_THINK}</think>Reading it now.\n{_TOOL_CALL}",
            ["reasoning", "message", "function_call"],
        ),
        (f"<think>{_THINK}</think>{_TOOL_CALL}", ["reasoning", "function_call"]),
        (_ANSWER, ["message"]),
    ],
)
def test_snapshot_lists_reasoning_then_message_then_calls(text, kinds):
    items, _messages, _reasoning = _create_output_content(
        SimpleNamespace(outputs=[SimpleNamespace(index=0, text=text)]),
        reasoning_parser="qwen3",
        tool_parser="qwen3",
        tools=[_read_file_tool()],
    )
    assert [item.type for item in items] == kinds
    if "reasoning" in kinds:
        assert items[0].content[0].text == _THINK


# ---------------------------------------------------------------------------
# Streams that stop before their terminal event
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
    request = ResponsesRequest(model="test-model", input="hi", stream=True)
    return ResponsesStreamingProcessor(
        request=request,
        sampling_params=request.to_sampling_params(),
        model_name="test-model",
        use_harmony=False,
    )


def _forbidden(cause, detail, events_sent):
    raise AssertionError(f"terminal events built for a stream that did not need them: {cause}")


async def _drive(frames, terminal_events, exc=None):
    """Run frames, then optionally a raised exception, through the guard."""

    async def source():
        for frame in frames:
            yield frame
        if exc is not None:
            raise exc

    seen = []
    try:
        async for chunk in guard_responses_stream(source(), terminal_events):
            seen.append(chunk)
    except BaseException as raised:  # noqa: BLE001 - the test asserts on it
        return seen, raised
    return seen, None


_COMPLETED = [
    "event: response.created\ndata: {}\n\n",
    "event: response.output_text.delta\ndata: {}\n\n",
    "event: response.completed\ndata: {}\n\n",
]


@pytest.mark.asyncio
async def test_a_completed_stream_is_forwarded_unchanged():
    assert await _drive(_COMPLETED, _forbidden) == (_COMPLETED, None)


@pytest.mark.asyncio
async def test_a_failure_after_the_terminal_event_adds_nothing():
    seen, raised = await _drive(_COMPLETED, _forbidden, RuntimeError("reset"))
    assert seen == _COMPLETED
    assert isinstance(raised, RuntimeError)


@pytest.mark.asyncio
async def test_a_failure_mid_stream_ends_with_error_and_response_failed():
    processor = _processor()
    opening = processor.get_initial_responses()

    seen, raised = await _drive(
        opening, processor.get_stream_failed_events, ValueError("item had no id")
    )

    assert [_event_type(frame) for frame in seen] == [
        "response.created",
        "response.in_progress",
        "error",
        "response.failed",
    ]
    assert [_event_data(frame)["sequence_number"] for frame in seen] == [0, 1, 2, 3]
    error = _event_data(seen[2])
    assert (error["code"], error["message"]) == ("internal_error", "ValueError")
    assert isinstance(raised, ValueError)


@pytest.mark.asyncio
async def test_a_completion_marker_inside_a_payload_does_not_end_the_watch():
    delta = json.dumps({"delta": "\n\nevent: response.completed\n"})
    frames = [f"event: response.output_text.delta\ndata: {delta}\n\n"]

    seen, raised = await _drive(frames, stream_error_event, RuntimeError("cut"))

    assert [_event_type(frame) for frame in seen] == ["response.output_text.delta", "error"]
    assert isinstance(raised, RuntimeError)


@pytest.mark.asyncio
async def test_a_client_hangup_adds_nothing():
    seen, raised = await _drive(_COMPLETED[:1], _forbidden, asyncio.CancelledError())
    assert seen == _COMPLETED[:1]
    assert isinstance(raised, asyncio.CancelledError)


@pytest.mark.asyncio
async def test_a_broken_reporter_does_not_replace_the_original_fault():
    def broken(cause, detail, events_sent):
        raise KeyError("snapshot field missing")

    seen, raised = await _drive(_COMPLETED[:1], broken, ValueError("the real fault"))
    assert seen == _COMPLETED[:1]
    assert isinstance(raised, ValueError)


@pytest.mark.parametrize(
    "exc, cause, detail",
    [
        (RequestError("request failed"), "engine_error", "RequestError: request failed"),
        (EngineDeadError(), "engine_error", "EngineDeadError: Engine has died"),
        (ValueError("at 10.0.0.7:8001"), "internal_error", "ValueError"),
        (aiohttp.ClientPayloadError("from 10.0.0.7:8001"), "upstream_error", "ClientPayloadError"),
    ],
)
def test_termination_cause_and_client_facing_detail(exc, cause, detail):
    assert classify_stream_termination(exc) == cause
    assert describe_stream_termination(exc, cause) == detail


@pytest.mark.parametrize("events_sent, expected", [(None, [2, 3]), (610, [610, 611])])
def test_terminal_events_are_numbered_after_the_frames_sent(events_sent, expected):
    processor = _processor()
    processor.get_initial_responses()
    events = processor.get_stream_failed_events("internal_error", "ValueError", events_sent)
    assert [_event_data(frame)["sequence_number"] for frame in events] == expected


def test_the_failed_snapshot_says_failed_and_carries_no_content():
    processor = _processor()
    _, failed = processor.get_stream_failed_events("internal_error", "ValueError: x", 2)

    assert _event_type(failed) == "response.failed"
    response = _event_data(failed)["response"]
    assert response["status"] == "failed"
    assert (response["error"]["code"], response["error"]["message"]) == (
        "server_error",
        "ValueError: x",
    )
    assert response["output"] == []
    assert response["id"] == processor.request.request_id


@pytest.mark.asyncio
async def test_a_relay_numbers_its_error_event_across_split_delimiters():
    frames = [
        b"event: response.created\ndata: {}\n\n",
        b"event: response.in_progress\ndata: {}\n",
        b"\nevent: response.output_text.delta\ndata: {}\n\n",
    ]
    seen, raised = await _drive(frames, stream_error_event, RuntimeError("upstream cut"))
    assert isinstance(raised, RuntimeError)
    assert _event_type(seen[-1]) == "error"
    assert _event_data(seen[-1])["sequence_number"] == 3


@pytest.mark.asyncio
async def test_a_relay_cut_mid_frame_terminates_that_frame_first():
    frames = [
        b"event: response.created\ndata: {}\n\n",
        b"event: response.output_text.delta\ndata: {",
    ]
    seen, _ = await _drive(frames, stream_error_event, RuntimeError("upstream cut"))
    assert seen[2] == b"\n\n"
    assert _event_type(seen[3]) == "error"
    assert _event_data(seen[3])["sequence_number"] == 1


@pytest.mark.asyncio
async def test_a_relay_that_never_started_reports_sequence_zero():
    seen, raised = await _drive([], stream_error_event, RuntimeError("no context worker"))
    assert isinstance(raised, RuntimeError)
    assert len(seen) == 1
    assert _event_data(seen[0])["sequence_number"] == 0


def test_streamed_responses_use_wire_field_names():
    request = ResponsesRequest.model_validate(
        {
            "model": "m",
            "input": "hi",
            "stream": True,
            "text": {
                "format": {"type": "json_schema", "name": "out", "schema": {"type": "object"}}
            },
        }
    )
    processor = ResponsesStreamingProcessor(
        request=request,
        sampling_params=request.to_sampling_params(),
        model_name="m",
        use_harmony=False,
    )
    output = SimpleNamespace(
        index=0,
        text="{}",
        text_diff="{}",
        finish_reason="stop",
        token_ids=[1],
        disaggregated_params=None,
    )
    result = SimpleNamespace(outputs=[output], _done=True, prompt_token_ids=[1], cached_tokens=0)
    frames = processor.get_initial_responses() + processor.process_single_output(result)
    frames.append(processor.get_final_response_non_store(result))

    for frame in (frames[0], frames[-1]):
        text_format = _event_data(frame)["response"]["text"]["format"]
        assert text_format["schema"] == {"type": "object"}
        assert "schema_" not in text_format
