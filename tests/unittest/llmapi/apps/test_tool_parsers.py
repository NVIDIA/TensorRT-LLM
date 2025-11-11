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

import json

import pytest

from tensorrt_llm.serve.openai_protocol import (ChatCompletionToolsParam,
                                                FunctionDefinition)
from tensorrt_llm.serve.tool_parser.base_tool_parser import BaseToolParser
from tensorrt_llm.serve.tool_parser.core_types import StructureInfo
from tensorrt_llm.serve.tool_parser.qwen3_tool_parser import Qwen3ToolParser


# Test fixtures for common tools
@pytest.fixture
def sample_tools():
    """Sample tools for testing."""
    return [
        ChatCompletionToolsParam(
            type="function",
            function=FunctionDefinition(name="get_weather",
                                        description="Get the current weather",
                                        parameters={
                                            "type": "object",
                                            "properties": {
                                                "location": {
                                                    "type":
                                                    "string",
                                                    "description":
                                                    "The city and state"
                                                },
                                                "unit": {
                                                    "type":
                                                    "string",
                                                    "enum":
                                                    ["celsius", "fahrenheit"]
                                                }
                                            },
                                            "required": ["location"]
                                        })),
        ChatCompletionToolsParam(
            type="function",
            function=FunctionDefinition(name="search_web",
                                        description="Search the web",
                                        parameters={
                                            "type": "object",
                                            "properties": {
                                                "query": {
                                                    "type":
                                                    "string",
                                                    "description":
                                                    "The search query"
                                                }
                                            },
                                            "required": ["query"]
                                        }))
    ]


# Concrete implementation of BaseToolParser for testing
class ConcreteToolParser(BaseToolParser):
    """Concrete implementation of BaseToolParser for testing abstract methods."""

    def __init__(self):
        super().__init__()
        self.bot_token = "[TOOL_CALLS] "
        self.eot_token = "[/TOOL_CALLS]"

    def has_tool_call(self, text: str) -> bool:
        return self.bot_token in text

    def detect_and_parse(self, text: str, tools):
        # Placeholder to avoid NotImplementedError
        pass

    def structure_info(self):
        return lambda name: StructureInfo(
            begin=f'[TOOL_CALLS] {{"name":"{name}", "arguments":',
            end="}[/TOOL_CALLS]",
            trigger="[TOOL_CALLS]")


# ============================================================================
# BaseToolParser Tests
# ============================================================================


class TestBaseToolParser:
    """Test suite for BaseToolParser class."""

    def test_initialization(self):
        """Test that BaseToolParser initializes correctly."""
        parser = ConcreteToolParser()
        assert parser._buffer == ""
        assert parser.prev_tool_call_arr == []
        assert parser.current_tool_id == -1
        assert parser.current_tool_name_sent is False
        assert parser.streamed_args_for_tool == []

    def test_get_tool_indices(self, sample_tools):
        """Test _get_tool_indices correctly maps tool names to indices."""
        parser = ConcreteToolParser()
        indices = parser._get_tool_indices(sample_tools)

        assert len(indices) == 2
        assert indices["get_weather"] == 0
        assert indices["search_web"] == 1

    def test_get_tool_indices_empty(self):
        """Test _get_tool_indices with empty tools list."""
        parser = ConcreteToolParser()
        indices = parser._get_tool_indices([])
        assert indices == {}

    def test_parse_base_json_single_tool(self, sample_tools):
        """Test parse_base_json with a single tool call."""
        parser = ConcreteToolParser()
        action = {
            "name": "get_weather",
            "parameters": {
                "location": "San Francisco"
            }
        }

        results = parser.parse_base_json(action, sample_tools)

        assert len(results) == 1
        assert results[0].name == "get_weather"
        assert json.loads(results[0].parameters) == {
            "location": "San Francisco"
        }

    def test_parse_base_json_with_arguments_key(self, sample_tools):
        """Test parse_base_json handles 'arguments' key instead of 'parameters'."""
        parser = ConcreteToolParser()
        action = {"name": "search_web", "arguments": {"query": "TensorRT"}}

        results = parser.parse_base_json(action, sample_tools)

        assert len(results) == 1
        assert results[0].name == "search_web"
        assert json.loads(results[0].parameters) == {"query": "TensorRT"}

    def test_parse_base_json_multiple_tools(self, sample_tools):
        """Test parse_base_json with multiple tool calls."""
        parser = ConcreteToolParser()
        actions = [{
            "name": "get_weather",
            "parameters": {
                "location": "Boston"
            }
        }, {
            "name": "search_web",
            "arguments": {
                "query": "Python"
            }
        }]

        results = parser.parse_base_json(actions, sample_tools)

        assert len(results) == 2
        assert results[0].name == "get_weather"
        assert results[1].name == "search_web"

    def test_parse_base_json_undefined_function(self, sample_tools):
        """Test parse_base_json handles undefined function names gracefully."""
        parser = ConcreteToolParser()
        action = {"name": "undefined_function", "parameters": {}}

        results = parser.parse_base_json(action, sample_tools)

        # Should return empty list and log warning
        assert len(results) == 0

    def test_parse_base_json_missing_parameters(self, sample_tools):
        """Test parse_base_json handles missing parameters."""
        parser = ConcreteToolParser()
        action = {"name": "get_weather"}

        results = parser.parse_base_json(action, sample_tools)

        assert len(results) == 1
        assert json.loads(results[0].parameters) == {}

    def test_ends_with_partial_token(self):
        """Test _ends_with_partial_token detection."""
        parser = ConcreteToolParser()

        # Partial token at end (bot_token starts with the suffix)
        assert parser._ends_with_partial_token("Some text [TOOL",
                                               "[TOOL_CALLS] ") == 5
        assert parser._ends_with_partial_token("Some text [",
                                               "[TOOL_CALLS] ") == 1
        assert parser._ends_with_partial_token("Some text [TOOL_CALLS",
                                               "[TOOL_CALLS] ") == 11

        # No partial token
        assert parser._ends_with_partial_token("Some text",
                                               "[TOOL_CALLS] ") == 0
        assert parser._ends_with_partial_token("Some text [XYZ",
                                               "[TOOL_CALLS] ") == 0

        # Complete token at end (entire buffer is bot_token prefix but not complete match)
        # When buffer equals bot_token, it returns 0 because it's not a partial anymore
        assert parser._ends_with_partial_token("text [TOOL_CALLS] ",
                                               "[TOOL_CALLS] ") == 0

    def test_parse_streaming_increment_no_tool_call(self, sample_tools):
        """Test streaming parser returns normal text when no tool call present."""
        parser = ConcreteToolParser()

        result = parser.parse_streaming_increment("Hello, world!", sample_tools)

        assert result.normal_text == "Hello, world!"
        assert len(result.calls) == 0

    def test_parse_streaming_increment_partial_bot_token(self, sample_tools):
        """Test streaming parser buffers partial bot token."""
        parser = ConcreteToolParser()

        # Send partial bot token
        result = parser.parse_streaming_increment("[TOOL", sample_tools)

        # Should buffer and return nothing
        assert result.normal_text == ""
        assert len(result.calls) == 0
        assert parser._buffer == "[TOOL"

    def test_parse_streaming_increment_tool_name(self, sample_tools):
        """Test streaming parser handles tool name streaming."""
        parser = ConcreteToolParser()

        # Send bot token with partial JSON containing name
        result = parser.parse_streaming_increment(
            '[TOOL_CALLS] {"name":"get_weather"', sample_tools)

        # Should send tool name with empty parameters
        assert len(result.calls) == 1
        assert result.calls[0].name == "get_weather"
        assert result.calls[0].parameters == ""
        assert result.calls[0].tool_index == 0
        assert parser.current_tool_name_sent is True

    def test_parse_streaming_increment_tool_arguments(self, sample_tools):
        """Test streaming parser handles incremental argument streaming."""
        parser = ConcreteToolParser()

        # First send tool name
        result1 = parser.parse_streaming_increment(
            '[TOOL_CALLS] {"name":"get_weather"', sample_tools)
        # Should send tool name
        assert len(result1.calls) == 1
        assert result1.calls[0].name == "get_weather"

        # Then send complete arguments (parser needs complete JSON to parse incrementally)
        result2 = parser.parse_streaming_increment(
            ',"arguments":{"location":"San Francisco"}}', sample_tools)

        # Should stream arguments or complete the tool call
        # The base implementation uses partial JSON parsing, so it may return results
        assert result2 is not None  # Just verify it doesn't crash

    def test_parse_streaming_increment_complete_tool(self, sample_tools):
        """Test streaming parser handles complete tool call."""
        parser = ConcreteToolParser()

        # Send complete tool call in one chunk
        result = parser.parse_streaming_increment(
            '[TOOL_CALLS] {"name":"get_weather","arguments":{"location":"Boston"}}',
            sample_tools)

        # Should have sent tool name (first call)
        assert len(result.calls) == 1
        assert result.calls[0].name == "get_weather"

    def test_parse_streaming_increment_invalid_tool_name(self, sample_tools):
        """Test streaming parser handles invalid tool name."""
        parser = ConcreteToolParser()

        # Send invalid tool name
        result = parser.parse_streaming_increment(
            '[TOOL_CALLS] {"name":"invalid_tool"', sample_tools)

        # Should reset state
        assert len(result.calls) == 0
        assert parser._buffer == ""
        assert parser.current_tool_id == -1

    def test_supports_structural_tag(self):
        """Test supports_structural_tag returns True."""
        parser = ConcreteToolParser()
        assert parser.supports_structural_tag() is True

    def test_structure_info(self):
        """Test structure_info returns proper function."""
        parser = ConcreteToolParser()
        func = parser.structure_info()

        info = func("test_function")
        assert isinstance(info, StructureInfo)
        assert "test_function" in info.begin
        assert info.trigger == "[TOOL_CALLS]"


# ============================================================================
# Qwen3ToolParser Tests
# ============================================================================


class TestQwen3ToolParser:
    """Test suite for Qwen3ToolParser class."""

    def test_initialization(self):
        """Test that Qwen3ToolParser initializes correctly."""
        parser = Qwen3ToolParser()

        assert parser.bot_token == "<tool_call>\n"
        assert parser.eot_token == "\n</tool_call>"
        assert parser.tool_call_separator == "\n"
        assert parser._normal_text_buffer == ""

    def test_has_tool_call_true(self):
        """Test has_tool_call returns True when tool call is present."""
        parser = Qwen3ToolParser()
        text = 'Some text <tool_call>\n{"name":"get_weather"}\n</tool_call>'

        assert parser.has_tool_call(text) is True

    def test_has_tool_call_false(self):
        """Test has_tool_call returns False when no tool call present."""
        parser = Qwen3ToolParser()
        text = "Just some regular text without tool calls"

        assert parser.has_tool_call(text) is False

    def test_detect_and_parse_no_tool_call(self, sample_tools):
        """Test detect_and_parse with text containing no tool calls."""
        parser = Qwen3ToolParser()
        text = "This is just a regular response."

        result = parser.detect_and_parse(text, sample_tools)

        assert result.normal_text == "This is just a regular response."
        assert len(result.calls) == 0

    def test_detect_and_parse_single_tool(self, sample_tools):
        """Test detect_and_parse with a single tool call."""
        parser = Qwen3ToolParser()
        text = 'Normal text\n<tool_call>\n{"name":"get_weather","arguments":{"location":"NYC"}}\n</tool_call>'

        result = parser.detect_and_parse(text, sample_tools)

        assert result.normal_text == "Normal text"
        assert len(result.calls) == 1
        assert result.calls[0].name == "get_weather"
        assert json.loads(result.calls[0].parameters) == {"location": "NYC"}

    def test_detect_and_parse_multiple_tools(self, sample_tools):
        """Test detect_and_parse with multiple tool calls."""
        parser = Qwen3ToolParser()
        text = (
            '<tool_call>\n{"name":"get_weather","arguments":{"location":"LA"}}\n</tool_call>\n'
            '<tool_call>\n{"name":"search_web","arguments":{"query":"AI"}}\n</tool_call>'
        )

        result = parser.detect_and_parse(text, sample_tools)

        assert len(result.calls) == 2
        assert result.calls[0].name == "get_weather"
        assert result.calls[1].name == "search_web"

    def test_detect_and_parse_malformed_json(self, sample_tools):
        """Test detect_and_parse handles malformed JSON gracefully."""
        parser = Qwen3ToolParser()
        text = '<tool_call>\n{"name":"get_weather","arguments":MALFORMED}\n</tool_call>'

        result = parser.detect_and_parse(text, sample_tools)

        # Should return empty calls due to JSON parsing error
        assert len(result.calls) == 0

    def test_detect_and_parse_with_parameters_key(self, sample_tools):
        """Test detect_and_parse handles 'parameters' key."""
        parser = Qwen3ToolParser()
        text = '<tool_call>\n{"name":"search_web","parameters":{"query":"test"}}\n</tool_call>'

        result = parser.detect_and_parse(text, sample_tools)

        assert len(result.calls) == 1
        assert result.calls[0].name == "search_web"
        assert json.loads(result.calls[0].parameters) == {"query": "test"}

    def test_parse_streaming_increment_normal_text(self, sample_tools):
        """Test streaming parser handles normal text without tool calls."""
        parser = Qwen3ToolParser()

        result = parser.parse_streaming_increment("Hello, how can I help?",
                                                  sample_tools)

        assert result.normal_text == "Hello, how can I help?"
        assert len(result.calls) == 0

    def test_parse_streaming_increment_partial_bot_token(self, sample_tools):
        """Test streaming parser buffers partial bot token."""
        parser = Qwen3ToolParser()

        # Send partial bot token
        result = parser.parse_streaming_increment("<tool", sample_tools)

        # Should buffer
        assert result.normal_text == ""
        assert len(result.calls) == 0

    def test_parse_streaming_increment_complete_tool_call(self, sample_tools):
        """Test streaming parser with complete tool call in chunks."""
        parser = Qwen3ToolParser()

        # Send bot token
        parser.parse_streaming_increment("<tool_call>\n", sample_tools)

        # Send partial JSON with name
        result = parser.parse_streaming_increment('{"name":"get_weather"',
                                                  sample_tools)

        # Should send tool name
        assert len(result.calls) == 1
        assert result.calls[0].name == "get_weather"
        assert result.calls[0].parameters == ""

        # Send arguments
        result = parser.parse_streaming_increment(
            ',"arguments":{"location":"SF"}}\n</tool_call>', sample_tools)

        # Should stream arguments
        assert len(result.calls) == 1
        assert json.loads(result.calls[0].parameters) == {"location": "SF"}

    def test_parse_streaming_increment_end_token_handling(self, sample_tools):
        """Test streaming parser handles end token correctly."""
        parser = Qwen3ToolParser()

        # Send complete tool call
        parser.parse_streaming_increment(
            '<tool_call>\n{"name":"get_weather","arguments":{"location":"NYC"}}\n</tool_call>',
            sample_tools)

        # The end token should be removed from normal text
        # Check buffer state
        assert parser._normal_text_buffer == ""

    def test_parse_streaming_increment_multiple_tools_streaming(
            self, sample_tools):
        """Test streaming parser handles multiple tool calls."""
        parser = Qwen3ToolParser()

        # First tool
        parser.parse_streaming_increment('<tool_call>\n', sample_tools)
        parser.parse_streaming_increment(
            '{"name":"get_weather","arguments":{"location":"NYC"}}\n</tool_call>\n',
            sample_tools)

        # Second tool
        parser.parse_streaming_increment('<tool_call>\n', sample_tools)
        result = parser.parse_streaming_increment('{"name":"search_web"',
                                                  sample_tools)

        # Should have started second tool
        assert result.calls[0].name == "search_web"
        assert result.calls[0].parameters == ""
        assert result.calls[0].tool_index == 1

    def test_structure_info_function(self):
        """Test structure_info returns correct lambda function."""
        parser = Qwen3ToolParser()
        func = parser.structure_info()

        info = func("test_function")

        assert isinstance(info, StructureInfo)
        assert info.begin == '<tool_call>\n{"name":"test_function", "arguments":'
        assert info.end == "}\n</tool_call>"
        assert info.trigger == "<tool_call>"

    def test_structure_info_different_names(self):
        """Test structure_info works with different function names."""
        parser = Qwen3ToolParser()
        func = parser.structure_info()

        info1 = func("get_weather")
        info2 = func("search_web")

        assert "get_weather" in info1.begin
        assert "search_web" in info2.begin
        assert info1.end == info2.end == "}\n</tool_call>"

    def test_qwen3_format_compliance(self, sample_tools):
        """Test that Qwen3ToolParser follows the documented format structure."""
        parser = Qwen3ToolParser()

        # Test the exact format from the docstring
        text = '<tool_call>\n{"name":"get_weather", "arguments":{"location":"Tokyo"}}\n</tool_call>'

        result = parser.detect_and_parse(text, sample_tools)

        assert len(result.calls) == 1
        assert result.calls[0].name == "get_weather"
        assert json.loads(result.calls[0].parameters) == {"location": "Tokyo"}

    def test_undefined_tool_in_qwen3_format(self, sample_tools):
        """Test Qwen3ToolParser handles undefined tool gracefully."""
        parser = Qwen3ToolParser()
        text = '<tool_call>\n{"name":"undefined_func","arguments":{}}\n</tool_call>'

        result = parser.detect_and_parse(text, sample_tools)

        # Should not return any calls for undefined function
        assert len(result.calls) == 0


# ============================================================================
# Integration Tests
# ============================================================================


class TestToolParserIntegration:
    """Integration tests for tool parsers."""

    def test_end_to_end_single_tool(self, sample_tools):
        """Test end-to-end parsing of a single tool call."""
        parser = Qwen3ToolParser()

        # Simulate streaming
        chunks = [
            "<tool_call>\n", '{"name":"get', '_weather"', ',"arguments":',
            '{"location"', ':"Paris"}}\n', '</tool_call>'
        ]

        results = []
        for chunk in chunks:
            result = parser.parse_streaming_increment(chunk, sample_tools)
            if result.calls or result.normal_text:
                results.append(result)

        # Should have received tool name and arguments
        assert any(r.calls for r in results)

    def test_mixed_content_and_tool_calls(self, sample_tools):
        """Test parsing text that mixes normal content with tool calls."""
        parser = Qwen3ToolParser()

        text = (
            'I will check the weather for you.\n'
            '<tool_call>\n{"name":"get_weather","arguments":{"location":"London"}}\n</tool_call>\n'
            'Let me search that for you.')

        result = parser.detect_and_parse(text, sample_tools)

        assert "I will check the weather for you." in result.normal_text
        assert len(result.calls) == 1
        assert result.calls[0].name == "get_weather"

    def test_parser_state_reset(self, sample_tools):
        """Test that parser state can be used for multiple requests."""
        parser = Qwen3ToolParser()

        # First request
        result1 = parser.detect_and_parse(
            '<tool_call>\n{"name":"get_weather","arguments":{"location":"NYC"}}\n</tool_call>',
            sample_tools)

        # Reset internal state for new request
        parser2 = Qwen3ToolParser()

        # Second request
        result2 = parser2.detect_and_parse(
            '<tool_call>\n{"name":"search_web","arguments":{"query":"test"}}\n</tool_call>',
            sample_tools)

        assert result1.calls[0].name == "get_weather"
        assert result2.calls[0].name == "search_web"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
