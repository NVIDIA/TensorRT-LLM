# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Unit tests for Responses API streaming tool call emission (TRTLLM-9605)."""

import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from tensorrt_llm.serve.responses_utils import (
    ResponsesStreamingEventsHelper,
    _generate_streaming_event,
    _should_send_done_events,
)
from tensorrt_llm.serve.tool_parser.core_types import ToolCallItem

# The CPU-* CI stages run pytest with -m 'cpu_only'. Without this marker every
# test in the file is deselected, which pytest reports as exit code 5 and the
# stage reports as a failure.
pytestmark = pytest.mark.cpu_only


def _make_mock_output(index: int = 0, text: str = "", text_diff: str = ""):
    """Create a minimal mock output with .index, .text, .text_diff."""

    class MockOutput:
        pass

    out = MockOutput()
    out.index = index
    out.text = text
    out.text_diff = text_diff
    return out


def _make_mock_request(tools: list | None = None):
    """Create a minimal ResponsesRequest-like object with .tools.

    SimpleNamespace rather than a nested class: a class body does not close over
    the enclosing function's locals (unlike a nested function), so
    ``tools = tools or []`` inside one raises NameError on the right-hand name.
    """
    return SimpleNamespace(tools=tools or [])


class TestShouldSendDoneEventsToolCalls:
    """Test that _should_send_done_events returns tool_calls when text is done due to tool calls."""

    def test_returns_tool_calls_when_text_done_due_to_tool_calls(self):
        """When full text and tool_calls exist and is_text_sent, 5th return is tool_calls."""
        output = _make_mock_output(
            index=0,
            text='Some text before <tool_call>{"name":"get_weather"}</tool_call>',
            text_diff="",
        )
        tool_calls = [
            ToolCallItem(tool_index=0, name="get_weather", parameters='{"location":"SF"}'),
        ]
        helper = ResponsesStreamingEventsHelper()
        helper.is_text_sent = True

        with (
            patch(
                "tensorrt_llm.serve.responses_utils._apply_reasoning_parser",
                return_value=('Some text before <tool_call>{"name":"get_weather"}</tool_call>', ""),
            ),
            patch(
                "tensorrt_llm.serve.responses_utils._apply_tool_parser",
                return_value=("Some text before ", tool_calls),
            ),
        ):
            result = _should_send_done_events(
                output=output,
                output_index=0,
                tool_parser_id="test_parser",
                tools=[],
                tool_parser_dict={0: None},
                streaming_events_helper=helper,
                finished_generation=False,
            )

        should_reasoning, should_text, reasoning_content, text_content, done_tool_calls = result
        assert should_text is True
        assert text_content == "Some text before "
        assert len(done_tool_calls) == 1
        assert done_tool_calls[0].name == "get_weather"
        assert done_tool_calls[0].parameters == '{"location":"SF"}'

    def test_plain_text_completion_sends_text_done_and_no_tool_calls(self):
        """The no-tool-call branch must still finish the text item.

        Asserting only that the tool-call list comes back empty would be
        asserting on the patched parser's own return value; what this branch
        actually has to get right is that plain text still reaches the client
        as a completed text item.
        """
        output = _make_mock_output(index=0, text="Just plain text", text_diff="")
        helper = ResponsesStreamingEventsHelper()
        helper.is_text_sent = True

        with (
            patch(
                "tensorrt_llm.serve.responses_utils._apply_reasoning_parser",
                return_value=("Just plain text", ""),
            ),
            patch(
                "tensorrt_llm.serve.responses_utils._apply_tool_parser",
                return_value=("Just plain text", []),
            ),
        ):
            result = _should_send_done_events(
                output=output,
                output_index=0,
                tools=[],
                streaming_events_helper=helper,
                finished_generation=True,
            )

        (_, should_send_text_done, _, text_content, done_tool_calls) = result
        # The behaviour this branch owns: generation finished with text already
        # streamed and no tool calls, so the text item must be closed out.
        assert should_send_text_done is True
        assert text_content == "Just plain text"
        assert done_tool_calls == []


class TestGenerateStreamingEventToolCalls:
    """Test that _generate_streaming_event yields output_item events for tool calls."""

    def test_emits_output_item_added_and_done_for_each_tool_call(self):
        """When done_tool_calls is non-empty, we get output_item.added and .done per tool call."""
        output = _make_mock_output(
            index=0,
            text='Hello <tool_call>{"name":"get_weather","arguments":{"location":"NYC"}}</tool_call>',
            text_diff="",
        )
        request = _make_mock_request(
            tools=[{"type": "function", "function": {"name": "get_weather"}}]
        )
        helper = ResponsesStreamingEventsHelper()
        helper.is_text_sent = True
        helper.item_id = "msg_initial"

        tool_calls = [
            ToolCallItem(tool_index=0, name="get_weather", parameters='{"location":"NYC"}'),
        ]

        # _apply_reasoning_parser: once for delta (""), once for full output in _should_send_done_events
        # _apply_tool_parser: once in _should_send_done_events with full text -> return text before tools + tool_calls
        with (
            patch(
                "tensorrt_llm.serve.responses_utils._apply_reasoning_parser",
                side_effect=[("", ""), (output.text, "")],
            ),
            patch(
                "tensorrt_llm.serve.responses_utils._apply_tool_parser",
                return_value=("Hello ", tool_calls),
            ),
            patch(
                "tensorrt_llm.serve.responses_utils._get_chat_completion_function_tools",
                return_value=[],
            ),
        ):
            events = list(
                _generate_streaming_event(
                    output=output,
                    request=request,
                    finished_generation=True,
                    streaming_events_helper=helper,
                    tool_parser_id="test",
                    tool_parser_dict={0: None},
                )
            )

        # We should have at least: text_done, content_part_done, output_item_done (text),
        # then output_item_added (tool), output_item_done (tool)
        types = [getattr(e, "type", None) for e in events]
        assert "response.output_item.added" in types
        assert "response.output_item.done" in types
        # Find the done event that has a function_call item
        from openai.types.responses import ResponseOutputItemDoneEvent

        done_events = [e for e in events if isinstance(e, ResponseOutputItemDoneEvent)]
        function_call_dones = [
            e
            for e in done_events
            if getattr(getattr(e, "item", None), "type", None) == "function_call"
        ]
        assert len(function_call_dones) >= 1
        assert function_call_dones[0].item.name == "get_weather"
        assert function_call_dones[0].item.arguments == '{"location":"NYC"}'


@pytest.mark.parametrize("suffix", ["<", "<t", "<tool_cal"])
@pytest.mark.parametrize("empty_terminal_delta", [False, True])
@pytest.mark.parametrize("with_tools", [False, True])
def test_qwen_coder_flushes_text_before_done(suffix, empty_terminal_delta, with_tools):
    from tensorrt_llm.serve.openai_protocol import ResponsesRequest

    request = ResponsesRequest(
        model="test",
        input="Hello",
        tools=[
            {"type": "function", "name": "ping", "parameters": {"type": "object", "properties": {}}}
        ]
        if with_tools
        else [],
    )
    helper = ResponsesStreamingEventsHelper()
    parsers = {}
    text = "x " + suffix
    events = list(
        _generate_streaming_event(
            _make_mock_output(text=text, text_diff=text),
            request,
            not empty_terminal_delta,
            helper,
            tool_parser_id="qwen3_coder",
            tool_parser_dict=parsers,
        )
    )
    if empty_terminal_delta:
        events += list(
            _generate_streaming_event(
                _make_mock_output(text=text, text_diff=""),
                request,
                True,
                helper,
                tool_parser_id="qwen3_coder",
                tool_parser_dict=parsers,
            )
        )
    deltas = [event for event in events if event.type == "response.output_text.delta"]
    done = [event for event in events if event.type == "response.output_text.done"]
    assert "".join(event.delta for event in deltas) == text
    assert len(done) == 1 and done[0].text == text
    assert max(events.index(event) for event in deltas) < events.index(done[0])
    assert parsers[0]._buf == ""


@pytest.mark.parametrize("chunk_size", [1, 7, 10000])
@pytest.mark.parametrize("second_call", ["", "<tool_call><function=g></function></tool_call>"])
def test_qwen_coder_recovers_unclosed_parameters_before_done(chunk_size, second_call):
    from tensorrt_llm.serve.openai_protocol import ResponsesRequest

    text = "<tool_call><function=f><parameter=x>abc</function></tool_call>" + second_call
    request = ResponsesRequest(
        model="test",
        input="Hello",
        tools=[
            {
                "type": "function",
                "name": "f",
                "parameters": {"type": "object", "properties": {"x": {"type": "string"}}},
            },
            {"type": "function", "name": "g", "parameters": {"type": "object"}},
        ],
    )
    helper = ResponsesStreamingEventsHelper()
    parsers = {}
    events = []
    for start in range(0, len(text), chunk_size):
        end = min(start + chunk_size, len(text))
        events.extend(
            _generate_streaming_event(
                _make_mock_output(text=text[:end], text_diff=text[start:end]),
                request,
                False,
                helper,
                tool_parser_id="qwen3_coder",
                tool_parser_dict=parsers,
            )
        )
    events.extend(
        _generate_streaming_event(
            _make_mock_output(text=text),
            request,
            True,
            helper,
            tool_parser_id="qwen3_coder",
            tool_parser_dict=parsers,
        )
    )
    done_calls = [
        event.item
        for event in events
        if event.type == "response.output_item.done" and event.item.type == "function_call"
    ]
    expected = [("f", {"x": "abc"})] + ([("g", {})] if second_call else [])
    assert [(call.name, json.loads(call.arguments)) for call in done_calls] == expected
    added_calls = [
        event.item
        for event in events
        if event.type == "response.output_item.added" and event.item.type == "function_call"
    ]
    assert [(call.name, json.loads(call.arguments)) for call in added_calls] == expected
    assert [call.id for call in added_calls] == [call.id for call in done_calls]
    assert not any(event.type == "response.output_text.delta" for event in events)


def test_qwen_coder_partial_preview_preserves_literal_outer_closers():
    from tensorrt_llm.serve.openai_protocol import ResponsesRequest

    request = ResponsesRequest(
        model="test",
        input="Hello",
        tools=[
            {
                "type": "function",
                "name": "f",
                "parameters": {"type": "object", "properties": {"x": {"type": "string"}}},
            }
        ],
    )
    prefix = "before<tool_call><function=f><parameter=x>a</function></tool_call>"
    text = prefix + "b</parameter></function></tool_call>"
    helper = ResponsesStreamingEventsHelper()
    parsers = {}
    events = list(
        _generate_streaming_event(
            _make_mock_output(text=prefix, text_diff=prefix),
            request,
            False,
            helper,
            tool_parser_id="qwen3_coder",
            tool_parser_dict=parsers,
        )
    )
    assert not any(event.type == "response.output_text.done" for event in events)
    events.extend(
        _generate_streaming_event(
            _make_mock_output(text=text, text_diff=text[len(prefix) :]),
            request,
            True,
            helper,
            tool_parser_id="qwen3_coder",
            tool_parser_dict=parsers,
        )
    )
    done_calls = [
        event.item
        for event in events
        if event.type == "response.output_item.done" and event.item.type == "function_call"
    ]
    assert len(done_calls) == 1
    assert json.loads(done_calls[0].arguments) == {"x": "a</function></tool_call>b"}
