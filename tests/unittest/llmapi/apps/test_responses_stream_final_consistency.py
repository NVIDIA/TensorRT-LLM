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
"""A streamed Responses turn and its final snapshot must agree.

The terminal event's snapshot is built after the stream, so these tests drive
the real streaming processor and compare what it streamed with what its final
event reports.
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
    StreamedItem,
    _create_output_content,
    _whole_text_tool_calls,
    create_response_non_store,
)

pytestmark = pytest.mark.cpu_only


def _event_type(frame):
    return frame.split("\n", 1)[0][len("event: ") :]


def _event_data(frame):
    return json.loads(frame.split("data: ", 1)[1])


def _tool(name="exec", properties=("input",)):
    return FunctionTool(
        name=name,
        type="function",
        strict=False,
        parameters={"type": "object", "properties": {p: {"type": "string"} for p in properties}},
    )


def _processor(tools=(), reasoning_parser="glm", tool_parser=None, tool_choice="auto"):
    """A processor built as the server builds one."""
    request = ResponsesRequest(
        model="test-model", input="hi", stream=True, tools=list(tools), tool_choice=tool_choice
    )
    return ResponsesStreamingProcessor(
        request=request,
        sampling_params=request.to_sampling_params(),
        model_name="test-model",
        use_harmony=False,
        reasoning_parser=reasoning_parser,
        tool_parser=tool_parser,
    )


def _generation_chunk(accumulated, diff, done, finish_reason="stop"):
    """One GenerationResult as the streaming loop sees it."""
    output = SimpleNamespace(
        index=0,
        text=accumulated,
        text_diff=diff,
        finish_reason=finish_reason if done else None,
        token_ids=[1, 2, 3],
        disaggregated_params=None,
    )
    return SimpleNamespace(outputs=[output], _done=done, prompt_token_ids=[1, 2], cached_tokens=0)


def _stream(processor, chunks):
    """Drive the processor; returns its frames and the finished result."""
    frames, accumulated, result = [], "", None
    for i, chunk in enumerate(chunks):
        accumulated += chunk
        result = _generation_chunk(accumulated, chunk, i == len(chunks) - 1)
        frames.extend(processor.process_single_output(result))
    return frames, result


def _done_items(frames, item_type=None):
    return [
        data["item"]
        for data in map(_event_data, frames)
        if data.get("type") == "response.output_item.done"
        and item_type in (None, data["item"]["type"])
    ]


def _added_ids(frames):
    return [
        (data["item"]["type"], data["item"]["id"])
        for data in map(_event_data, frames)
        if data.get("type") == "response.output_item.added"
    ]


def _message_texts(items):
    return [item["content"][0]["text"] for item in items if item["type"] == "message"]


def _final_output(processor, result):
    frame = processor.get_final_response_non_store(result)
    assert _event_type(frame) == "response.completed"
    return _event_data(frame)["response"]["output"]


def _snapshot(text, streamed_item_ids, reasoning_parser=None, outputs=None):
    items, messages, _reasoning = _create_output_content(
        SimpleNamespace(outputs=outputs or [SimpleNamespace(index=0, text=text)]),
        reasoning_parser=reasoning_parser,
        streamed_item_ids=streamed_item_ids,
    )
    return items, messages


# ---------------------------------------------------------------------------
# Item ids and texts
# ---------------------------------------------------------------------------


def test_final_items_are_the_streamed_items():
    processor = _processor()
    frames, result = _stream(processor, ["Check the reference.", "</think>", " The answer. "])

    added = _added_ids(frames)
    assert [item_type for item_type, _ in added] == ["reasoning", "message"]
    assert [(i["type"], i["id"]) for i in _done_items(frames)] == added

    final = _final_output(processor, result)
    assert [(item["type"], item["id"]) for item in final] == added
    # Edge whitespace is part of the generation.
    assert _message_texts(final) == [" The answer. "]


def test_streamed_segmentation_is_repeated_verbatim():
    with patch("tensorrt_llm.serve.responses_utils.logger") as mock_logger:
        items, messages = _snapshot(
            "the answer",
            [
                StreamedItem("message", "msg_first", "the"),
                StreamedItem("message", "msg_2", " answer"),
            ],
        )
    assert [(i.id, i.content[0].text) for i in items] == [
        ("msg_first", "the"),
        ("msg_2", " answer"),
    ]
    assert messages == [{"role": "assistant", "content": "the answer"}]
    assert not mock_logger.warning.called


def test_ids_are_reused_in_emission_order_when_the_record_does_not_cover():
    """Two outputs: streaming reads only outputs[0], so the record is not used."""
    items, _ = _snapshot(
        None,
        [StreamedItem("message", "msg_first"), StreamedItem("message", "msg_second")],
        outputs=[SimpleNamespace(index=0, text="first"), SimpleNamespace(index=1, text="second")],
    )
    assert [(i.id, i.content[0].text) for i in items] == [
        ("msg_first", "first"),
        ("msg_second", "second"),
    ]


def test_a_non_streamed_snapshot_mints_fresh_ids_and_keeps_whitespace():
    with patch("tensorrt_llm.serve.responses_utils.logger") as mock_logger:
        items, _ = _snapshot("think</think> Backbone ", None, reasoning_parser="glm")
    assert [i.id[:3] for i in items] == ["rs_", "msg"]
    assert items[1].content[0].text == " Backbone "
    assert not mock_logger.warning.called


@pytest.mark.parametrize(
    "streamed",
    [
        [],
        [StreamedItem("message", "msg_first", "answer"), StreamedItem("message", "msg_2", "?")],
    ],
)
def test_a_count_mismatch_is_logged(streamed):
    with patch("tensorrt_llm.serve.responses_utils.logger") as mock_logger:
        items, _ = _snapshot("answer", streamed)
    assert [i.content[0].text for i in items] == ["answer"]
    if streamed:
        assert items[0].id == "msg_first"
    assert mock_logger.warning.called


# ---------------------------------------------------------------------------
# Sequence numbers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("postprocess_worker", [False, True])
def test_sequence_numbers_run_contiguously_from_zero(postprocess_worker):
    """A postprocessing worker gets a pickled copy before the opening pair."""
    frontend = _processor()
    producer = pickle.loads(pickle.dumps(frontend)) if postprocess_worker else frontend

    frames = frontend.get_initial_responses()
    body, result = _stream(producer, ["Plan.", "</think>", "Done."])
    frames += body + [producer.get_final_response_non_store(result)]

    assert [_event_data(frame)["sequence_number"] for frame in frames] == list(range(len(frames)))


# ---------------------------------------------------------------------------
# Tool calls
# ---------------------------------------------------------------------------

_EXEC_CALL = "<tool_call>exec<arg_key>input</arg_key><arg_value>ls</arg_value></tool_call>"
_QWEN3_CALL = '<tool_call>\n{"name": "exec", "arguments": {"input": "ls"}}\n</tool_call>'


@pytest.mark.parametrize(
    "parsers, chunks",
    [
        (("glm47", "glm47"), ["Plan it.", "</think>Run it. ", _EXEC_CALL]),
        (("glm", "qwen3"), ["Plan it.", "</think>Run it.\n", _QWEN3_CALL]),
    ],
    ids=["incremental", "whole_text"],
)
def test_final_tool_calls_are_the_streamed_ones(parsers, chunks):
    processor = _processor(tools=[_tool()], reasoning_parser=parsers[0], tool_parser=parsers[1])
    frames, result = _stream(processor, chunks)
    streamed_calls = _done_items(frames, "function_call")
    assert [(c["name"], json.loads(c["arguments"])) for c in streamed_calls] == [
        ("exec", {"input": "ls"})
    ]

    final = _final_output(processor, result)
    assert [i for i in final if i["type"] == "function_call"] == streamed_calls
    assert [i["type"] for i in final] == ["reasoning", "message", "function_call"]
    assert _message_texts(final) == _message_texts(_done_items(frames, "message"))


def test_an_unterminated_call_is_text_in_both_views():
    unterminated = "<tool_call>exec<arg_key>input</arg_key><arg_value>const cmd = `cat m.json`"
    processor = _processor(tools=[_tool()], reasoning_parser="glm47", tool_parser="glm47")
    frames, result = _stream(processor, ["Plan it.", "</think>Run it.", unterminated])

    assert _done_items(frames, "function_call") == []
    streamed = _done_items(frames)
    assert "".join(_message_texts(streamed)) == "Run it." + unterminated

    with patch("tensorrt_llm.serve.responses_utils.logger") as mock_logger:
        final = _final_output(processor, result)
    assert [(i["type"], i["id"]) for i in final] == [(i["type"], i["id"]) for i in streamed]
    assert _message_texts(final) == _message_texts(streamed)
    assert not mock_logger.warning.called


def test_tool_choice_none_keeps_the_markup_as_text_in_both_views():
    processor = _processor(
        tools=[_tool()], reasoning_parser="glm47", tool_parser="glm47", tool_choice="none"
    )
    frames, result = _stream(processor, ["Plan it.", "</think>Calling: ", _EXEC_CALL])

    assert _message_texts(_done_items(frames)) == ["Calling: " + _EXEC_CALL]
    final = _final_output(processor, result)
    assert [item["type"] for item in final] == ["reasoning", "message"]
    assert _message_texts(final) == ["Calling: " + _EXEC_CALL]


# ---------------------------------------------------------------------------
# Terminal event and status
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "finish_reason, event_type, status, details",
    [
        ("stop", "response.completed", "completed", None),
        ("length", "response.incomplete", "incomplete", {"reason": "max_output_tokens"}),
        ("timeout", "response.failed", "failed", None),
        ("cancelled", "response.completed", "cancelled", None),
    ],
)
def test_the_terminal_event_matches_the_status(finish_reason, event_type, status, details):
    processor = _processor(reasoning_parser=None)
    result = _generation_chunk("Partial answer", "Partial answer", True, finish_reason)
    frames = processor.process_single_output(result)
    assert all(_event_type(f) != "response.completed" for f in frames)

    frame = processor.get_final_response_non_store(result)
    payload = _event_data(frame)
    assert (_event_type(frame), payload["type"]) == (event_type, event_type)
    assert payload["response"]["status"] == status
    assert payload["response"]["incomplete_details"] == details
    assert _message_texts(payload["response"]["output"]) == ["Partial answer"]


@pytest.mark.parametrize(
    "finish_reason, status, details",
    [
        ("stop", "completed", None),
        ("length", "incomplete", {"reason": "max_output_tokens"}),
        # A disaggregated context worker handing off is not a token budget.
        ("not_finished", "incomplete", None),
    ],
)
def test_a_non_streamed_response_explains_an_incomplete_status(finish_reason, status, details):
    request = ResponsesRequest(model="test-model", input="hi", stream=False)
    response = create_response_non_store(
        generation_result=_generation_chunk("Partial", "Partial", True, finish_reason),
        request=request,
        sampling_params=request.to_sampling_params(),
        model_name="test-model",
        use_harmony=False,
        num_prompt_tokens=2,
    ).model_dump()
    assert (response["status"], response["incomplete_details"]) == (status, details)


@pytest.mark.parametrize(
    "streamed, whole, owed, warned",
    [
        ("Hello", "Hello world", " world", False),
        # The whole-text parse dropped the leading whitespace the stream sent.
        ("  Hello", "Hello world", " world", False),
        ("Hello ", "Hello", "", False),
        ("Hello", "Goodbye", "", True),
    ],
)
def test_the_stream_releases_what_the_whole_text_parse_reads_beyond_it(
    streamed, whole, owed, warned
):
    helper = SimpleNamespace(emitted_item_ids=[StreamedItem("message", "msg_1", streamed)])
    request = ResponsesRequest(model="test-model", input="hi", stream=True)
    with (
        patch(
            "tensorrt_llm.serve.responses_utils._apply_reasoning_parser",
            return_value=(whole, ""),
        ),
        patch(
            "tensorrt_llm.serve.responses_utils._apply_tool_parser",
            return_value=(whole, []),
        ),
        patch("tensorrt_llm.serve.responses_utils.logger") as logger,
    ):
        released, calls = _whole_text_tool_calls(
            SimpleNamespace(index=0, text=whole), request, helper, None, "qwen3", []
        )
    assert (released, calls, logger.warning.called) == (owed, [], warned)
