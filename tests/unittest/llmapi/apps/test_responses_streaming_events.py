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
import contextlib
import json
import os
import re
from types import SimpleNamespace
from unittest.mock import AsyncMock

import aiohttp
import pytest
from openai.types.responses.tool import FunctionTool
from starlette.requests import ClientDisconnect

import tensorrt_llm.serve.responses_utils as responses_utils
from tensorrt_llm.executor import EngineDeadError, RequestError
from tensorrt_llm.serve.openai_protocol import ResponsesRequest
from tensorrt_llm.serve.responses_utils import (
    ClosingStreamingResponse,
    ResponsesStreamingEventsHelper,
    ResponsesStreamingProcessor,
    _create_output_content,
    _generate_streaming_event,
    _stamp_sequence_number,
    classify_stream_termination,
    describe_stream_termination,
    guard_responses_stream,
    keep_responses_stream_alive,
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


@pytest.mark.parametrize(
    "terminal_cut", [b"event: response.completed\n", b"event: response.failed\ndata: {"]
)
@pytest.mark.asyncio
async def test_a_relay_cut_inside_its_terminal_frame_still_reports_the_error(terminal_cut):
    frames = [b"event: response.created\ndata: {}\n\n", terminal_cut]
    seen, raised = await _drive(frames, stream_error_event, RuntimeError("upstream cut"))
    assert isinstance(raised, RuntimeError)
    assert seen[:3] == [*frames, b"\n\n"]
    assert _event_type(seen[3]) == "error"


@pytest.mark.asyncio
async def test_a_terminal_frame_split_across_chunks_completes_the_stream():
    frames = [
        b"event: response.created\ndata: {}\n\n",
        b"event: response.completed\ndata: {",
        b"}\n",
        b"\n",
    ]
    seen, raised = await _drive(frames, _forbidden, RuntimeError("reset"))
    assert seen == frames
    assert isinstance(raised, RuntimeError)


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


# ---------------------------------------------------------------------------
# Keepalive while a stream is silent
# ---------------------------------------------------------------------------


def _step(text, diff, done):
    """One generation step, as the streaming post-processor receives it."""
    output = SimpleNamespace(
        index=0,
        text=text,
        text_diff=diff,
        finish_reason="stop" if done else None,
        token_ids=[1] * len(text),
        disaggregated_params=None,
    )
    return SimpleNamespace(
        outputs=[output], _done=done, prompt_token_ids=[1], cached_tokens=0, finished=done
    )


def _server_frames(steps, processor=None, fail=None, produced=None):
    """The frames the server streams: the opening pair, then each step's events.

    ``steps`` are (pause, text) pairs; a step that adds no text sends no event,
    as while a tool call is held. ``fail`` is raised instead of the last step
    finishing the response. Each frame is also appended to ``produced``.
    """
    processor = processor or _processor()
    produced = [] if produced is None else produced

    async def frames():
        for frame in processor.get_initial_responses():
            produced.append(frame)
            yield frame
        text = ""
        for i, (pause, diff) in enumerate(steps):
            await asyncio.sleep(pause)
            text += diff
            result = _step(text, diff, done=fail is None and i == len(steps) - 1)
            step_frames = processor.process_single_output(result)
            if result._done:
                step_frames.append(processor.get_final_response_non_store(result))
            for frame in step_frames:
                produced.append(frame)
                yield frame
        if fail is not None:
            raise fail

    return frames()


async def _collect(stream, sent=None, limit=500):
    """The frames of ``stream``, failing a stream that does not end instead of hanging on it."""
    sent = [] if sent is None else sent

    async def drain():
        async for frame in stream:
            sent.append(frame)
            assert len(sent) < limit, "the stream does not end"

    await asyncio.wait_for(drain(), timeout=30)
    return sent


async def _send(monkeypatch, frames, interval):
    monkeypatch.setattr(responses_utils, "RESPONSES_STREAM_KEEPALIVE_INTERVAL_S", interval)
    return await _collect(keep_responses_stream_alive(frames))


def _keepalive_positions(frames):
    """Positions of the response.in_progress events after the opening one."""
    in_progress = [i for i, f in enumerate(frames) if _event_type(f) == "response.in_progress"]
    return in_progress[1:]


def _assert_own_number_only(produced, sent, number):
    """``sent`` is ``produced`` with ``number`` as its sequence number, and byte for byte else."""
    assert _event_data(sent)["sequence_number"] == number
    if _event_data(produced)["sequence_number"] == number:
        assert sent == produced
        return
    # The one stretch that differs is the digits of the event's own number.
    prefix = len(os.path.commonprefix([produced, sent]))
    suffix = len(os.path.commonprefix([produced[prefix:][::-1], sent[prefix:][::-1]]))
    changed = (produced[prefix : len(produced) - suffix], sent[prefix : len(sent) - suffix])
    assert all(re.fullmatch(r"\d*", part) for part in changed), changed
    before, after = _event_data(produced), _event_data(sent)
    del before["sequence_number"], after["sequence_number"]
    assert before == after


@pytest.mark.asyncio
async def test_keepalives_cover_the_wait_for_the_first_token(monkeypatch):
    """Queueing and prefill send nothing; in_progress is repeated until output starts."""
    sent = await _send(monkeypatch, _server_frames([(0.4, "Hello"), (0, " world")]), 0.05)

    first_output = [_event_type(f) for f in sent].index("response.output_item.added")
    keepalives = _keepalive_positions(sent)
    assert keepalives[0] == 2
    assert len([i for i in keepalives if i < first_output]) >= 2
    assert _event_type(sent[-1]) == "response.completed"


@pytest.mark.asyncio
async def test_keepalives_cover_steps_that_send_nothing(monkeypatch):
    """Steps arrive while a tool call is held but bring no event: they do not count as output."""
    steps = [(0, "Calling the tool.")] + [(0.02, "")] * 15 + [(0, " Done.")]
    sent = await _send(monkeypatch, _server_frames(steps), 0.05)

    types = [_event_type(f) for f in sent]
    keepalives = _keepalive_positions(sent)
    assert keepalives
    assert types.index("response.output_text.delta") < keepalives[0]
    assert types[-1] == "response.completed"


@pytest.mark.asyncio
async def test_no_keepalive_while_events_flow(monkeypatch):
    """Each event restarts the wait: a stream longer than the interval but never silent is unchanged."""
    produced = []
    steps = [(0.02, f"w{i} ") for i in range(40)]
    sent = await _send(monkeypatch, _server_frames(steps, produced=produced), 0.5)

    assert sent == produced


@pytest.mark.asyncio
async def test_no_keepalive_after_the_response_ends(monkeypatch):
    """After the terminal event the response is over; a slow end of the source sends nothing."""
    produced = []

    async def frames():
        async for frame in _server_frames([(0, "hello")], produced=produced):
            yield frame
        await asyncio.sleep(0.3)  # as while the request's metrics are recorded

    sent = await _send(monkeypatch, frames(), 0.05)

    assert sent == produced
    assert _event_type(sent[-1]) == "response.completed"


@pytest.mark.asyncio
async def test_events_are_renumbered_after_a_keepalive_and_nothing_else_changes(monkeypatch):
    """Sequence numbers stay consecutive with keepalives inserted; only those digits change.

    The response carries a nested ``sequence_number`` (a tool schema), the model
    writes the field's name in its text, and the snapshot holds a float whose
    exponent form re-serializing would change: none of it may be touched.
    """
    request = ResponsesRequest(
        model="test-model",
        input="hi",
        stream=True,
        temperature=1e-05,
        metadata={"sequence_number": "9"},
        tools=[
            {
                "type": "function",
                "name": "seek",
                "parameters": {"type": "object", "examples": [{"sequence_number": 3}]},
            }
        ],
    )
    processor = ResponsesStreamingProcessor(
        request=request,
        sampling_params=request.to_sampling_params(),
        model_name="test-model",
        use_harmony=False,
    )
    produced = []
    steps = [(0.2, '{"sequence_number":'), (0.2, " 7}"), (0, " é")]
    sent = await _send(monkeypatch, _server_frames(steps, processor, produced=produced), 0.05)

    assert [_event_data(f)["sequence_number"] for f in sent] == list(range(len(sent)))
    keepalives = _keepalive_positions(sent)
    assert len(keepalives) >= 2
    for position in keepalives:
        _assert_own_number_only(produced[1], sent[position], position)
    others = [i for i in range(len(sent)) if i not in keepalives]
    assert len(others) == len(produced)
    for frame, position in zip(produced, others):
        _assert_own_number_only(frame, sent[position], position)


@pytest.mark.asyncio
async def test_a_failure_after_keepalives_still_ends_with_error_and_response_failed(monkeypatch):
    """A generation that fails during a silence ends the stream as before, numbered in place."""
    processor = _processor()
    frames = guard_responses_stream(
        _server_frames([(0, "Hello"), (0.3, "")], processor, fail=RuntimeError("engine failed")),
        processor.get_stream_failed_events,
    )
    monkeypatch.setattr(responses_utils, "RESPONSES_STREAM_KEEPALIVE_INTERVAL_S", 0.05)

    sent = []
    with pytest.raises(RuntimeError, match="engine failed"):
        await _collect(keep_responses_stream_alive(frames), sent)

    assert [_event_type(f) for f in sent[-2:]] == ["error", "response.failed"]
    assert _keepalive_positions(sent)
    assert [_event_data(f)["sequence_number"] for f in sent] == list(range(len(sent)))


@pytest.mark.asyncio
async def test_stopping_the_stream_cancels_its_pending_read(monkeypatch):
    """A client that goes away during a silence stops the source's read at once."""
    monkeypatch.setattr(responses_utils, "RESPONSES_STREAM_KEEPALIVE_INTERVAL_S", 0.05)
    cancelled = []

    async def frames():
        for frame in _processor().get_initial_responses():
            yield frame
        try:
            await asyncio.sleep(30)
        except asyncio.CancelledError:
            cancelled.append(True)
            raise
        yield "never sent"

    stream = keep_responses_stream_alive(frames())
    sent = []

    async def until_the_first_keepalive():
        async for frame in stream:
            sent.append(frame)
            if _keepalive_positions(sent):
                return

    await asyncio.wait_for(until_the_first_keepalive(), timeout=10)
    await stream.aclose()

    assert cancelled == [True]


@pytest.mark.asyncio
async def test_a_client_that_leaves_while_a_frame_waits_closes_the_source():
    """The source is closed by the time the stream is, also while its next frame waits."""
    closed = []

    async def frames():
        try:
            async for frame in _server_frames([(0, f"w{i} ") for i in range(20)]):
                yield frame
        finally:
            closed.append(True)

    stream = keep_responses_stream_alive(frames())
    async for _ in stream:
        break
    # The reader fills the queue and then waits to hand over the next frame.
    for _ in range(5):
        await asyncio.sleep(0)
    await stream.aclose()

    assert closed == [True]


@pytest.mark.asyncio
@pytest.mark.parametrize("spec_version", ["2.3", "2.4"])
async def test_a_client_that_disconnects_mid_send_closes_the_source(spec_version):
    """The response closes its stream when a send fails, on both ASGI paths.

    ASGI 2.3 cancels the blocked send; 2.4 raises from it, and Starlette then
    raises ClientDisconnect without running a background task.
    """
    closed = []

    async def frames():
        try:
            for frame in _processor().get_initial_responses():
                yield frame
            await asyncio.Event().wait()
        finally:
            closed.append(True)

    response = ClosingStreamingResponse(
        content=keep_responses_stream_alive(frames()), media_type="text/event-stream"
    )
    first_body = asyncio.Event()
    bodies = 0

    async def send(message):
        nonlocal bodies
        if message["type"] != "http.response.body":
            return
        bodies += 1
        if bodies == 1:
            first_body.set()
            return
        if spec_version == "2.4":
            raise OSError("connection reset")
        await asyncio.Event().wait()

    async def receive():
        await first_body.wait()
        await asyncio.sleep(0.05)
        return {"type": "http.disconnect"}

    scope = {"type": "http", "asgi": {"spec_version": spec_version}}
    with contextlib.suppress(ClientDisconnect):
        await asyncio.wait_for(response(scope, receive, send), timeout=5)

    assert closed == [True]


@pytest.mark.asyncio
async def test_the_source_is_read_from_one_task(monkeypatch):
    """Keepalives never move the source between tasks.

    The source is the server's generation stream; an async generator that holds
    a task-bound context (a lock, a timeout or a cancel scope) across a yield
    breaks if it is resumed from another task.
    """
    readers = set()

    async def frames():
        async for frame in _server_frames([(0.1, "one"), (0.1, " two"), (0.1, " three")]):
            readers.add(asyncio.current_task())
            yield frame

    sent = await _send(monkeypatch, frames(), 0.02)

    assert _keepalive_positions(sent)
    assert len(readers) == 1


@pytest.mark.asyncio
async def test_a_slow_client_holds_back_the_source():
    """The source is read only as fast as the client takes frames."""
    produced = []
    stream = keep_responses_stream_alive(
        _server_frames([(0, f"w{i} ") for i in range(20)], produced=produced)
    )

    sent, lead = [], 0
    async for frame in stream:
        sent.append(frame)
        lead = max(lead, len(produced) - len(sent))
        await asyncio.sleep(0.005)

    assert lead <= 3, f"the source ran {lead} frames ahead of the client"
    assert sent == produced


@pytest.mark.asyncio
async def test_a_source_cancelled_on_its_own_ends_the_stream(monkeypatch):
    """A cancellation raised inside the source ends the stream instead of leaving it to keepalives."""
    monkeypatch.setattr(responses_utils, "RESPONSES_STREAM_KEEPALIVE_INTERVAL_S", 0.02)

    async def frames():
        for frame in _processor().get_initial_responses():
            yield frame
        await asyncio.sleep(0.1)
        raise asyncio.CancelledError()

    sent = []
    with pytest.raises(asyncio.CancelledError):
        await _collect(keep_responses_stream_alive(frames()), sent, limit=50)

    assert _keepalive_positions(sent)


# Payloads as the server serializes them (compact, floats in exponent form,
# non-ASCII text raw), each with the field's name elsewhere: nested in the
# response, written by the model, or in a string ending in a backslash.
_IN_PROGRESS_DATA = (
    r'{"response":{"id":"resp_1","metadata":{"sequence_number":"9"},"temperature":1e-5,'
    r'"tools":[{"parameters":{"examples":[{"name":"x","sequence_number":3}]}}]},'
    r'"sequence_number":1,"type":"response.in_progress"}'
)
_DELTA_DATA = (
    r'{"content_index":0,"delta":"{\"sequence_number\":7}\\","item_id":"msg_1",'
    r'"logprobs":[{"logprob":-1.25e-7,"token":"é"}],"output_index":0,'
    r'"sequence_number":4,"type":"response.output_text.delta"}'
)
_TEXT_DONE_DATA = (
    r'{"content_index":0,"item_id":"msg_1","logprobs":[],"output_index":0,'
    r'"sequence_number":5,"text":"\\\",\"sequence_number\":0} é",'
    r'"type":"response.output_text.done"}'
)


@pytest.mark.parametrize("data", [_IN_PROGRESS_DATA, _DELTA_DATA, _TEXT_DONE_DATA])
@pytest.mark.parametrize("number", [0, 6, 1234])
def test_stamping_changes_only_the_events_own_number(data, number):
    event_type = json.loads(data)["type"]
    frame = f"event: {event_type}\ndata: {data}\n\n"

    stamped = _stamp_sequence_number(frame, number)

    _assert_own_number_only(frame, stamped, number)


@pytest.mark.parametrize(
    "frame",
    [
        "event: response.created\ndata: {}\n\n",
        'event: response.created\ndata: {"response":{"sequence_number":3}}\n\n',
        'event: response.created\ndata: {"sequence_number":null}\n\n',
        "event: response.created\n\n",
        "data: [DONE]\n\n",
    ],
)
def test_stamping_leaves_a_frame_without_its_own_number_alone(frame):
    assert _stamp_sequence_number(frame, 7) == frame


def _server_streaming(monkeypatch, steps):
    """An OpenAIServer whose generator streams ``steps`` as (pause, text) pairs."""
    import tensorrt_llm.serve.openai_server as server_module
    from tensorrt_llm.serve.openai_server import OpenAIServer

    server = object.__new__(OpenAIServer)
    server.model = "test-model"
    server.enable_store = False
    server._is_visual_gen = False
    server.use_harmony = False
    server.tokenizer = None
    server.model_config = None
    server.processor = None
    server.multimodal_server_config = None
    server.chat_template = None
    server.tool_parser = None
    server.metrics_collector = None
    server.conversation_store = None
    server._validate_internal_disagg_request = lambda request, raw_request: None
    server._extract_metrics = AsyncMock()

    class Generation:
        prompt_token_ids = [1, 2, 3]
        finished = False

        async def __aiter__(self):
            text = ""
            for i, (pause, diff) in enumerate(steps):
                await asyncio.sleep(pause)
                text += diff
                result = _step(text, diff, done=i == len(steps) - 1)
                self.finished = result._done
                yield result

    server.generator = SimpleNamespace(
        args=SimpleNamespace(reasoning_parser=None, num_postprocess_workers=0),
        generate_async=lambda **kwargs: Generation(),
    )

    async def preprocess(**kwargs):
        return [1, 2, 3], kwargs["request"].to_sampling_params()

    monkeypatch.setattr(server_module, "responses_api_request_preprocess", preprocess)
    return server


@pytest.mark.asyncio
async def test_the_responses_route_keeps_a_silent_stream_alive(monkeypatch):
    """The streaming route sends in_progress again before the first token and while a call is held."""
    monkeypatch.setattr(responses_utils, "RESPONSES_STREAM_KEEPALIVE_INTERVAL_S", 0.05)
    steps = [(0.3, "Let me run it.")] + [(0.02, "")] * 15 + [(0, " Done.")]
    server = _server_streaming(monkeypatch, steps)
    request = ResponsesRequest(model="test-model", input="hi", stream=True)
    raw_request = SimpleNamespace(state=SimpleNamespace(no_client_connection=True), headers={})

    response = await server.openai_responses(request, raw_request)
    sent = await _collect(response.body_iterator)

    types = [_event_type(f) for f in sent]
    keepalives = _keepalive_positions(sent)
    assert keepalives[0] == 2
    assert any(i > types.index("response.output_text.delta") for i in keepalives)
    assert types[-1] == "response.completed"
    assert [_event_data(f)["sequence_number"] for f in sent] == list(range(len(sent)))
