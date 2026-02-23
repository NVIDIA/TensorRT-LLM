# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from unittest.mock import patch

from tensorrt_llm.serve.responses_utils import (
    ResponsesStreamingEventsHelper,
    _generate_streaming_event,
    _should_send_done_events,
)
from tensorrt_llm.serve.tool_parser.core_types import ToolCallItem


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
    """Create a minimal ResponsesRequest-like object with .tools."""

    class MockRequest:
        tools = tools or []

    return MockRequest()


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

    def test_returns_empty_tool_calls_when_no_tool_calls(self):
        """When full text but no tool_calls, 5th return is empty list."""
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

        _, _, _, _, done_tool_calls = result
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
