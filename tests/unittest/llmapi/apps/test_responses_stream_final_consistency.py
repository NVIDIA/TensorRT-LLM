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
"""Offline tests: a streamed Responses turn and its final snapshot must agree.

`response.completed` rebuilds `output` from the accumulated text in a second,
independent pass, and streaming events are numbered by whoever built them.
Three ways the two views of one generation disagreed, each measured on live
traffic before it was tested here:

* The rebuild minted fresh ids for the reasoning and message items the stream
  had already announced - 216 of 221 responses - so a client joining streamed
  items with the snapshot by id saw phantom items.
* With postprocessing workers enabled a response's frames come from two
  processes, each numbering from zero, so every streamed response carried
  sequence numbers 0,1,0,1,2,... - 221 of 221 measured.
* A stream that ended inside an unterminated tool call streamed nothing for
  it while the final rebuild kept the fragment as message text (trace
  tr_8f312e9973954fa784ae7dc8f45e9d3e, 429 characters), so the snapshot held
  text the stream never showed.
"""

import json
import pickle
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from openai.types.responses.tool import FunctionTool

from tensorrt_llm.serve.openai_protocol import ResponsesRequest
from tensorrt_llm.serve.responses_utils import (
    ResponsesStreamingProcessor,
    _create_output_content,
    stamp_sse_sequence_number,
)

# The CPU-* CI stages run pytest with -m 'cpu_only'. Without this marker every
# test in the file is deselected, which pytest reports as exit code 5 and the
# stage reports as a failure.
pytestmark = pytest.mark.cpu_only


def _event_type(frame):
    return frame.split("\n", 1)[0][len("event: ") :]


def _event_data(frame):
    return json.loads(frame.split("data: ", 1)[1])


def _exec_tool():
    return FunctionTool(
        name="exec",
        type="function",
        strict=False,
        parameters={"type": "object", "properties": {"input": {"type": "string"}}},
    )


def _processor(tools=None, reasoning_parser="glm", tool_parser=None):
    """A processor built exactly as the server builds one."""
    request = ResponsesRequest(model="test-model", input="hi", stream=True, tools=list(tools or []))
    return ResponsesStreamingProcessor(
        request=request,
        sampling_params=request.to_sampling_params(),
        model_name="test-model",
        use_harmony=False,
        reasoning_parser=reasoning_parser,
        tool_parser=tool_parser,
    )


def _generation_chunk(accumulated, diff, done):
    """One GenerationResult as the streaming loop sees it.

    The final chunk doubles as the finished result the completed-response
    builder reads, so it carries finish_reason, token ids and prompt tokens.
    """
    output = SimpleNamespace(
        index=0,
        text=accumulated,
        text_diff=diff,
        finish_reason="stop" if done else None,
        token_ids=[1, 2, 3],
        disaggregated_params=None,
    )
    return SimpleNamespace(outputs=[output], _done=done, prompt_token_ids=[1, 2], cached_tokens=0)


def _stream(processor, chunks):
    """Drive the real processor over generation chunks.

    Returns the SSE frames it produced and the finished generation result,
    which is what the server hands to get_final_response_non_store.
    """
    frames, accumulated, result = [], "", None
    for i, chunk in enumerate(chunks):
        accumulated += chunk
        result = _generation_chunk(accumulated, chunk, i == len(chunks) - 1)
        frames.extend(processor.process_single_output(result))
    return frames, result


def _streamed_items(frames, event_type="response.output_item.added"):
    return [
        (data["item"]["type"], data["item"]["id"])
        for data in map(_event_data, frames)
        if data.get("type") == event_type
    ]


def _final_output(processor, result):
    frame = processor.get_final_response_non_store(result)
    assert _event_type(frame) == "response.completed"
    return _event_data(frame)["response"]["output"]


# ---------------------------------------------------------------------------
# Item ids: the snapshot must answer to the ids the stream announced
# ---------------------------------------------------------------------------


def test_final_item_ids_match_the_streamed_item_ids():
    """The invariant, driven end to end through the real processor.

    Same shape and cure as the tool-call ids one commit earlier: the stream
    records what it announced in `output_item.added`, the rebuild reuses it.
    A client that keyed items off the streamed ids used to match nothing in
    the snapshot - 216 of 221 measured responses.
    """
    processor = _processor()
    frames, result = _stream(processor, ["Check the reference.", "</think>", "The answer is 42."])

    added = _streamed_items(frames)
    done = _streamed_items(frames, "response.output_item.done")
    assert added == done, "an item must close under the id it opened with"
    assert [item_type for item_type, _ in added] == ["reasoning", "message"]
    assert all(item_id for _, item_id in added)

    final = [(item["type"], item["id"]) for item in _final_output(processor, result)]
    assert final == added


def test_streamed_ids_are_reused_positionally_in_emission_order():
    """Two items of one type map onto the rebuild first-emitted first.

    Emission order is the only correspondence the two passes share, which is
    also how the tool-call precedent matches its calls.
    """
    items, _messages, _reasoning = _create_output_content(
        SimpleNamespace(
            outputs=[
                SimpleNamespace(index=0, text="first answer"),
                SimpleNamespace(index=1, text="second answer"),
            ]
        ),
        reasoning_parser=None,
        streamed_item_ids=[("message", "msg_first"), ("message", "msg_second")],
    )
    assert [(i.type, i.id) for i in items] == [
        ("message", "msg_first"),
        ("message", "msg_second"),
    ]


def test_reasoning_and_message_ids_are_reused_per_type():
    items, _messages, _reasoning = _create_output_content(
        SimpleNamespace(outputs=[SimpleNamespace(index=0, text="think</think>answer")]),
        reasoning_parser="glm",
        streamed_item_ids=[("reasoning", "rs_r1"), ("message", "msg_m1")],
    )
    assert [(i.type, i.id) for i in items] == [
        ("reasoning", "rs_r1"),
        ("message", "msg_m1"),
    ]


def test_a_non_streaming_rebuild_still_mints_fresh_ids_silently():
    """None means no stream ran; fresh uuids are correct and nothing warns."""
    with patch("tensorrt_llm.serve.responses_utils.logger") as mock_logger:
        items, _messages, _reasoning = _create_output_content(
            SimpleNamespace(outputs=[SimpleNamespace(index=0, text="think</think>answer")]),
            reasoning_parser="glm",
            streamed_item_ids=None,
        )
    assert items[0].id.startswith("rs_")
    assert items[1].id.startswith("msg_")
    assert not mock_logger.warning.called


def test_rebuilding_items_the_stream_never_opened_is_said_out_loud():
    """Overflow means the two views already diverged structurally.

    A fresh id cannot hide that and must not try to - that divergence is the
    unterminated-call territory below, and papering over it silently would
    bury the one signal that says the views differ.
    """
    with patch("tensorrt_llm.serve.responses_utils.logger") as mock_logger:
        items, _messages, _reasoning = _create_output_content(
            SimpleNamespace(outputs=[SimpleNamespace(index=0, text="answer")]),
            reasoning_parser=None,
            streamed_item_ids=[],
        )
    assert items[0].id.startswith("msg_")
    assert mock_logger.warning.called
    assert "message" in mock_logger.warning.call_args[0]


def test_streamed_items_the_rebuild_drops_are_said_out_loud_too():
    """The mismatch matters in both directions; the first id is still reused."""
    with patch("tensorrt_llm.serve.responses_utils.logger") as mock_logger:
        items, _messages, _reasoning = _create_output_content(
            SimpleNamespace(outputs=[SimpleNamespace(index=0, text="answer")]),
            reasoning_parser=None,
            streamed_item_ids=[("message", "msg_first"), ("message", "msg_second")],
        )
    assert [i.id for i in items] == ["msg_first"]
    assert mock_logger.warning.called


# ---------------------------------------------------------------------------
# Sequence numbers: one monotonic counter per response, stamped at the egress
# ---------------------------------------------------------------------------


def test_egress_restamp_unifies_the_two_producers_counters():
    """The postproc-worker configuration, reproduced with a real pickle.

    The worker builds the per-token frames from a pickled copy of the
    streaming processor - the same trip PostprocParams takes - whose counter
    starts at zero, while the frontend's own counter numbered the two opening
    events. The raw frames therefore restart mid-stream, and the egress
    restamp is what turns them into one monotonic sequence.
    """
    frontend = _processor()
    worker = pickle.loads(pickle.dumps(frontend))

    frames = frontend.get_initial_responses()
    body, result = _stream(worker, ["Plan.", "</think>", "Done."])
    frames += body
    frames.append(worker.get_final_response_non_store(result))

    produced = [_event_data(frame)["sequence_number"] for frame in frames]
    # The defect stated raw: two counters, one restart inside one response.
    assert produced[:3] == [0, 1, 0]

    stamped = [stamp_sse_sequence_number(frame, i) for i, frame in enumerate(frames)]
    assert [_event_data(f)["sequence_number"] for f in stamped] == list(range(len(frames)))
    # The restamp changes the number and nothing else.
    for before, after in zip(frames, stamped):
        assert _event_type(before) == _event_type(after)
        b, a = _event_data(before), _event_data(after)
        b.pop("sequence_number"), a.pop("sequence_number")
        assert b == a


def test_in_process_numbering_already_matches_the_egress_stamp():
    """With workers off one instance numbers everything; the stamp is a no-op.

    This is what keeps the restamp safe to apply unconditionally at the
    egress: the already-correct configuration comes out byte-equivalent.
    """
    processor = _processor()
    frames = processor.get_initial_responses()
    body, result = _stream(processor, ["Plan.", "</think>", "Done."])
    frames += body
    frames.append(processor.get_final_response_non_store(result))

    numbers = [_event_data(frame)["sequence_number"] for frame in frames]
    assert numbers == list(range(len(frames)))
    stamped = [stamp_sse_sequence_number(frame, i) for i, frame in enumerate(frames)]
    assert [_event_data(f)["sequence_number"] for f in stamped] == numbers


def test_the_stamp_rewrites_the_field_and_not_the_model_text():
    """The payload can quote this very protocol; only the real field changes.

    A substring replace would have rewritten the model's own output here,
    which is why the stamp parses the frame instead.
    """
    inner = 'code: {"sequence_number": 999}'
    frame = (
        "event: response.completed\ndata: "
        + json.dumps(
            {
                "type": "response.completed",
                "sequence_number": 1,
                "response": {"output_text": inner},
            },
            separators=(",", ":"),
        )
        + "\n\n"
    )
    out = stamp_sse_sequence_number(frame, 41)
    payload = _event_data(out)
    assert payload["sequence_number"] == 41
    assert payload["response"]["output_text"] == inner
    assert out.endswith("\n\n") and out.count("\n\n") == 1


def test_frames_without_a_sequence_number_pass_through_untouched():
    for frame in (
        "event: ping\n\n",
        "event: x\ndata: {}\n\n",
        "event: x\ndata: not-json\n\n",
    ):
        assert stamp_sse_sequence_number(frame, 9) == frame


# ---------------------------------------------------------------------------
# An unterminated tool call: both views carry the same fallback text
# ---------------------------------------------------------------------------

# The shape of the real 429-character fragment from trace
# tr_8f312e9973954fa784ae7dc8f45e9d3e, shortened: a GLM call whose value never
# closes, so no parser - incremental or whole-text - can read it as a call.
_UNTERMINATED = (
    "<tool_call>exec<arg_key>input</arg_key><arg_value>"
    "const cmd = await tools.exec_command({cmd: `cat metrics.json`"
)


def _message_texts(frames):
    return [
        data["item"]["content"][0]["text"]
        for data in map(_event_data, frames)
        if data.get("type") == "response.output_item.done" and data["item"]["type"] == "message"
    ]


def test_an_unterminated_call_reaches_both_views_as_the_same_text():
    """The stream must show the fallback text the final response keeps.

    Before the flush released it, the streamed view had no trace of the
    fragment while the snapshot carried it as message text - the one turn
    shape where the snapshot held text the stream never showed. Both views
    now publish the same characters; the stream splits them across two
    message items because its first message closed when the call was
    announced, chunks before anyone could know the call would never finish.
    """
    processor = _processor(tools=[_exec_tool()], reasoning_parser="glm47", tool_parser="glm47")
    frames, result = _stream(processor, ["Plan it.", "</think>Run it.", _UNTERMINATED])

    streamed_texts = _message_texts(frames)
    assert streamed_texts == ["Run it.", _UNTERMINATED]
    # No half-built call on the stream: the fragment is text, not a call.
    assert all(item_type != "function_call" for item_type, _ in _streamed_items(frames))

    final = _final_output(processor, result)
    final_messages = [item for item in final if item["type"] == "message"]
    assert [item["type"] for item in final] == ["reasoning", "message"]
    # The same characters, in the same order, on both views. The final pass
    # re-parses the whole text so its fallback is a single message; equality
    # is on the concatenated text, which is what a training-data consumer
    # reads.
    assert "".join(streamed_texts) == final_messages[0]["content"][0]["text"]
    assert _UNTERMINATED in final_messages[0]["content"][0]["text"]


def test_the_unterminated_fragment_keeps_the_streamed_ids_it_can():
    """Id linkage under the structural divergence the fragment causes.

    The stream published two message items, the rebuild derives one: the one
    it derives answers to the first streamed id, and the mismatch is warned
    about rather than hidden - it is the visible residue of a call the model
    never finished.
    """
    processor = _processor(tools=[_exec_tool()], reasoning_parser="glm47", tool_parser="glm47")
    frames, result = _stream(processor, ["Plan it.", "</think>Run it.", _UNTERMINATED])
    streamed = _streamed_items(frames)
    assert [item_type for item_type, _ in streamed] == [
        "reasoning",
        "message",
        "message",
    ]

    with patch("tensorrt_llm.serve.responses_utils.logger") as mock_logger:
        final = _final_output(processor, result)

    assert (final[0]["type"], final[0]["id"]) == streamed[0]
    assert (final[1]["type"], final[1]["id"]) == streamed[1]
    assert mock_logger.warning.called
