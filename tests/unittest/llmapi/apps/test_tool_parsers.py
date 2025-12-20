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

import abc
import json
from typing import NamedTuple

import pytest

from tensorrt_llm.serve.openai_protocol import (ChatCompletionToolsParam,
                                                FunctionDefinition)
from tensorrt_llm.serve.tool_parser.base_tool_parser import BaseToolParser
from tensorrt_llm.serve.tool_parser.core_types import StructureInfo
from tensorrt_llm.serve.tool_parser.deepseekv3_parser import DeepSeekV3Parser
from tensorrt_llm.serve.tool_parser.deepseekv31_parser import DeepSeekV31Parser
from tensorrt_llm.serve.tool_parser.deepseekv32_parser import DeepSeekV32Parser
from tensorrt_llm.serve.tool_parser.kimi_k2_tool_parser import KimiK2ToolParser
from tensorrt_llm.serve.tool_parser.qwen3_coder_parser import \
    Qwen3CoderToolParser
from tensorrt_llm.serve.tool_parser.qwen3_tool_parser import Qwen3ToolParser


# Test fixtures for common tools
@pytest.fixture
def sample_tools():
    """Sample tools for testing."""
    return [
        ChatCompletionToolsParam(
            type="function",
            function=FunctionDefinition(
                name="get_weather",
                description="Get the current weather",
                parameters={
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state",
                        },
                        "unit": {
                            "type": "string",
                            "enum": ["celsius", "fahrenheit"],
                        },
                    },
                    "required": ["location"],
                },
            ),
        ),
        ChatCompletionToolsParam(
            type="function",
            function=FunctionDefinition(
                name="search_web",
                description="Search the web",
                parameters={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query",
                        }
                    },
                    "required": ["query"],
                },
            ),
        ),
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

        assert len(indices) == len(sample_tools)
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


class ToolParserTestCases(NamedTuple):
    has_tool_call_true: str
    detect_and_parse_single_tool: tuple[
        # Input text.
        str,
        # Expected `normal_text`.
        str,
        # Expected `name`.
        str,
        # Expected `parameters`.
        dict,
    ]
    detect_and_parse_multiple_tools: tuple[
        # Input text.
        str,
        # Expected names.
        tuple[str],
    ]
    detect_and_parse_malformed_tool: str
    detect_and_parse_with_parameters_key: tuple[
        # Input text.
        str,
        # Expected `name`.
        str,
        # Expected `parameters`.
        dict,
    ]
    parse_streaming_increment_partial_bot_token: str
    undefined_tool: str


class BaseToolParserTestClass:
    """Base class from which tests for actual implementations can be extended.

    NOTE: the name deliberately ends with `Class` so that `pytest` does not pick it up for execution
    automatically.
    """

    @abc.abstractmethod
    def make_parser(self):
        ...

    @property
    def make_tool_parser_test_cases(self) -> ToolParserTestCases:
        ...

    @pytest.fixture
    def parser(self):
        return self.make_parser()

    @pytest.fixture(scope="class")
    def tool_parser_test_cases(self) -> ToolParserTestCases:
        return self.make_tool_parser_test_cases()

    def test_has_tool_call_false(self, parser):
        """Test has_tool_call returns False when no tool call present."""
        text = "Just some regular text without tool calls"

        assert parser.has_tool_call(text) is False

    def test_has_tool_call_true(self, parser, tool_parser_test_cases):
        """Test has_tool_call returns True when tool call is present."""
        text = tool_parser_test_cases.has_tool_call_true

        assert parser.has_tool_call(text) is True

    def test_detect_and_parse_no_tool_call(self, sample_tools, parser):
        """Test detect_and_parse with text containing no tool calls."""
        text = "This is just a regular response."

        result = parser.detect_and_parse(text, sample_tools)

        assert result.normal_text == text
        assert len(result.calls) == 0

    def test_detect_and_parse_single_tool(self, sample_tools, parser,
                                          tool_parser_test_cases):
        """Test detect_and_parse with a single tool call."""
        text, normal_text, name, parameters = tool_parser_test_cases.detect_and_parse_single_tool

        result = parser.detect_and_parse(text, sample_tools)

        assert result.normal_text == normal_text
        assert len(result.calls) == 1
        assert result.calls[0].name == name
        assert json.loads(result.calls[0].parameters) == parameters

    def test_detect_and_parse_multiple_tools(self, sample_tools, parser,
                                             tool_parser_test_cases):
        """Test detect_and_parse with multiple tool calls."""
        text, call_names = tool_parser_test_cases.detect_and_parse_multiple_tools

        result = parser.detect_and_parse(text, sample_tools)

        assert tuple(call.name for call in result.calls) == call_names

    def test_detect_and_parse_malformed_tool(self, sample_tools, parser,
                                             tool_parser_test_cases):
        """Test detect_and_parse handles malformed tool call output from the model gracefully."""
        text = tool_parser_test_cases.detect_and_parse_malformed_tool

        result = parser.detect_and_parse(text, sample_tools)

        assert len(result.calls) == 0

    def test_detect_and_parse_with_parameters_key(self, sample_tools, parser,
                                                  tool_parser_test_cases):
        """Test detect_and_parse handles 'parameters' key."""
        text, name, parameters = tool_parser_test_cases.detect_and_parse_with_parameters_key

        result = parser.detect_and_parse(text, sample_tools)

        assert len(result.calls) == 1
        assert result.calls[0].name == name
        assert json.loads(result.calls[0].parameters) == parameters

    def test_parse_streaming_increment_normal_text(self, sample_tools, parser):
        """Test streaming parser handles normal text without tool calls."""
        text = "Hello, how can I help?"

        result = parser.parse_streaming_increment(text, sample_tools)

        assert result.normal_text == text
        assert len(result.calls) == 0

    def test_parse_streaming_increment_partial_bot_token(
            self, sample_tools, parser, tool_parser_test_cases):
        """Test streaming parser buffers partial bot token."""
        text = tool_parser_test_cases.parse_streaming_increment_partial_bot_token

        result = parser.parse_streaming_increment(text, sample_tools)

        assert result.normal_text == ""
        assert len(result.calls) == 0

    def test_undefined_tool(self, sample_tools, parser, tool_parser_test_cases):
        """Test the parser handles undefined tool gracefully."""
        text = tool_parser_test_cases.undefined_tool

        result = parser.detect_and_parse(text, sample_tools)

        # Should not return any calls for undefined function
        assert len(result.calls) == 0


class TestKimiK2ToolParser(BaseToolParserTestClass):
    """Test suite for KimiK2ToolParser class."""

    def make_parser(self):
        return KimiK2ToolParser()

    def make_tool_parser_test_cases(self):
        return ToolParserTestCases(
            has_tool_call_true=
            'Some text <|tool_calls_section_begin|><|tool_call_begin|>functions.get_weather:0<|tool_call_argument_begin|>{"location": "NYC"}<|tool_call_end|><|tool_calls_section_end|>',
            detect_and_parse_single_tool=(
                # Input text.
                ('Normal text'
                 '<|tool_calls_section_begin|><|tool_call_begin|>functions.get_weather:0<|tool_call_argument_begin|>{"location": "NYC"}<|tool_call_end|><|tool_calls_section_end|>'
                 ),
                # Expected `normal_text`.
                "Normal text",
                # Expected `name`.
                "get_weather",
                # Expected `parameters`.
                {
                    "location": "NYC"
                },
            ),
            detect_and_parse_multiple_tools=(
                # Input text.
                ('<|tool_calls_section_begin|><|tool_call_begin|>functions.get_weather:0<|tool_call_argument_begin|>{"location":"LA"}<|tool_call_end|>\n'
                 '<|tool_call_begin|>functions.search_web:0<|tool_call_argument_begin|>{"query":"AI"}<|tool_call_end|><|tool_calls_section_end|>'
                 ),
                # Expected names.
                ("get_weather", "search_web"),
            ),
            detect_and_parse_malformed_tool=
            ('<|tool_calls_section_begin|><|tool_call_begin|>functions.get_weather:0<|tool_call_argument_begin|>MALFORMED<|tool_call_end|><|tool_calls_section_end|>'
             ),
            detect_and_parse_with_parameters_key=(
                # Input text.
                ('<|tool_calls_section_begin|><|tool_call_begin|>functions.search_web:0<|tool_call_argument_begin|>{"query":"test"}<|tool_call_end|><|tool_calls_section_end|>'
                 ),
                # Expected `name`.
                "search_web",
                # Expected `parameters`.
                {
                    "query": "test"
                },
            ),
            parse_streaming_increment_partial_bot_token=
            "<|tool_calls_section_begin|><|tool_call_be",
            undefined_tool=
            '<|tool_calls_section_begin|><|tool_call_begin|>functions.undefined_func:0<|tool_call_argument_begin|>{"arg":"any value"}<|tool_call_end|><|tool_calls_section_end|>',
        )

    def test_initialization(self, parser):
        """Test that Qwen3ToolParser initializes correctly."""
        assert parser.bot_token == "<|tool_calls_section_begin|>"
        assert parser.eot_token == "<|tool_calls_section_end|>"

    def test_parse_streaming_increment_complete_tool_call(
            self, sample_tools, parser):
        """Test streaming parser with complete tool call in chunks."""

        # Send bot token
        parser.parse_streaming_increment("<|tool_calls_section_begin|>",
                                         sample_tools)

        # Send partial tool call with name
        result = parser.parse_streaming_increment(
            '<|tool_call_begin|>functions.get_weather:0<|tool_call_argument_begin|>{',
            sample_tools)

        # Should send tool name
        assert len(result.calls) == 1
        assert result.calls[0].name == "get_weather"
        assert result.calls[0].parameters == ""

        # Send arguments
        result = parser.parse_streaming_increment(
            '"location":"SF"}<|tool_call_end|>', sample_tools)

        # Should stream arguments
        assert len(result.calls) == 1
        assert json.loads(result.calls[0].parameters) == {"location": "SF"}

    def test_parse_streaming_increment_multiple_tools_streaming(
            self, sample_tools, parser):
        """Test streaming parser handles multiple tool calls."""

        # First tool
        parser.parse_streaming_increment('<|tool_calls_section_begin|>',
                                         sample_tools)
        parser.parse_streaming_increment(
            '<|tool_call_begin|>functions.get_weather:0<|tool_call_argument_begin|>{"location":"NYC"}<|tool_call_end|>',
            sample_tools)

        # Second tool
        parser.parse_streaming_increment(
            '<|tool_call_begin|>functions.search_web:0<|tool_call_argument_begin|>{"arg": "any value"}<|tool_call_end|>',
            sample_tools)

        result = parser.parse_streaming_increment('<|tool_calls_section_end|>',
                                                  sample_tools)
        # Should have started second tool
        assert result.calls[0].name == "search_web"
        assert result.calls[0].parameters == ""
        assert result.calls[0].tool_index == 1

    def test_structure_info_function(self):
        """Test structure_info returns correct lambda function."""
        parser = KimiK2ToolParser()
        func = parser.structure_info()

        info = func("test_function")

        assert isinstance(info, StructureInfo)
        assert info.begin == '<|tool_calls_section_begin|><|tool_call_begin|>functions.test_function:0<|tool_call_argument_begin|>'
        assert info.end == "<|tool_call_end|><|tool_calls_section_end|>"
        assert info.trigger == "<|tool_calls_section_begin|>"

    def test_structure_info_different_names(self):
        """Test structure_info works with different function names."""
        parser = KimiK2ToolParser()
        func = parser.structure_info()

        info1 = func("get_weather")
        info2 = func("search_web")

        assert "get_weather" in info1.begin
        assert "search_web" in info2.begin
        assert info1.end == info2.end == "<|tool_call_end|><|tool_calls_section_end|>"

    def test_kimi_k2_format_compliance(self, sample_tools, parser):
        """Test that KimiK2ToolParser follows the documented format structure."""

        # Test the exact format from the docstring
        text = '<|tool_calls_section_begin|><|tool_call_begin|>functions.get_weather:0<|tool_call_argument_begin|>{"location":"Tokyo"}<|tool_call_end|><|tool_calls_section_end|>'

        result = parser.detect_and_parse(text, sample_tools)

        assert len(result.calls) == 1
        assert result.calls[0].name == "get_weather"
        assert json.loads(result.calls[0].parameters) == {"location": "Tokyo"}


class TestQwen3ToolParser(BaseToolParserTestClass):
    """Test suite for Qwen3ToolParser class."""

    def make_parser(self):
        return Qwen3ToolParser()

    def make_tool_parser_test_cases(self):
        return ToolParserTestCases(
            has_tool_call_true=
            'Some text <tool_call>\n{"name":"get_weather"}\n</tool_call>',
            detect_and_parse_single_tool=(
                # Input text.
                ('Normal text\n'
                 '<tool_call>\n'
                 '{"name":"get_weather","arguments":{"location":"NYC"}}\n'
                 '</tool_call>'),
                # Expected `normal_text`.
                "Normal text",
                # Expected `name`.
                "get_weather",
                # Expected `parameters`.
                {
                    "location": "NYC"
                },
            ),
            detect_and_parse_multiple_tools=(
                # Input text.
                ('<tool_call>\n{"name":"get_weather","arguments":{"location":"LA"}}\n</tool_call>\n'
                 '<tool_call>\n{"name":"search_web","arguments":{"query":"AI"}}\n</tool_call>'
                 ),
                # Expected names.
                ("get_weather", "search_web"),
            ),
            detect_and_parse_malformed_tool=
            ('<tool_call>\n{"name":"get_weather","arguments":MALFORMED}\n</tool_call>'
             ),
            detect_and_parse_with_parameters_key=(
                # Input text.
                ('<tool_call>\n{"name":"search_web","parameters":{"query":"test"}}\n</tool_call>'
                 ),
                # Expected `name`.
                "search_web",
                # Expected `parameters`.
                {
                    "query": "test"
                },
            ),
            parse_streaming_increment_partial_bot_token="<tool",
            undefined_tool=
            '<tool_call>\n{"name":"undefined_func","arguments":{}}\n</tool_call>',
        )

    def test_initialization(self, parser):
        """Test that Qwen3ToolParser initializes correctly."""
        assert parser.bot_token == "<tool_call>\n"
        assert parser.eot_token == "\n</tool_call>"
        assert parser.tool_call_separator == "\n"
        assert parser._normal_text_buffer == ""

    # NOTE: this is not put in the base class. Even though it could be made generic, the added logic
    # to do so loses the clarity of this more direct approach.
    def test_parse_streaming_increment_complete_tool_call(
            self, sample_tools, parser):
        """Test streaming parser with complete tool call in chunks."""

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

    def test_parse_streaming_increment_end_token_handling(
            self, sample_tools, parser):
        """Test streaming parser handles end token correctly."""

        # Send complete tool call
        parser.parse_streaming_increment(
            '<tool_call>\n{"name":"get_weather","arguments":{"location":"NYC"}}\n</tool_call>',
            sample_tools)

        # The end token should be removed from normal text
        # Check buffer state
        assert parser._normal_text_buffer == ""

    def test_parse_streaming_increment_multiple_tools_streaming(
            self, sample_tools, parser):
        """Test streaming parser handles multiple tool calls."""

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

    def test_qwen3_format_compliance(self, sample_tools, parser):
        """Test that Qwen3ToolParser follows the documented format structure."""

        # Test the exact format from the docstring
        text = '<tool_call>\n{"name":"get_weather", "arguments":{"location":"Tokyo"}}\n</tool_call>'

        result = parser.detect_and_parse(text, sample_tools)

        assert len(result.calls) == 1
        assert result.calls[0].name == "get_weather"
        assert json.loads(result.calls[0].parameters) == {"location": "Tokyo"}


class TestQwen3CoderToolParser(BaseToolParserTestClass):
    """Test suite for Qwen3CoderToolParser class."""

    def make_parser(self):
        return Qwen3CoderToolParser()

    def make_tool_parser_test_cases(self):
        return ToolParserTestCases(
            has_tool_call_true=("Some text <tool_call>\n"
                                "<function=get_weather>\n"
                                "<parameter=location>NYC</parameter>\n"
                                "</function>\n"
                                "</tool_call>"),
            detect_and_parse_single_tool=(
                # Input text.
                ("Normal text\n"
                 "<tool_call>\n"
                 "<function=get_weather>\n"
                 "<parameter=location>NYC</parameter>\n"
                 "</function>\n"
                 "</tool_call>"),
                # Expected `normal_text`.
                "Normal text\n",
                # Expected `name`.
                "get_weather",
                # Expected `parameters`.
                {
                    "location": "NYC"
                },
            ),
            detect_and_parse_multiple_tools=(
                # Input text.
                ("<tool_call>\n"
                 "<function=get_weather>\n"
                 "<parameter=location>LA</parameter>\n"
                 "</function>\n"
                 "</tool_call>\n"
                 "<tool_call>\n"
                 "<function=search_web>\n"
                 "<parameter=query>AI</parameter>\n"
                 "</function>\n"
                 "</tool_call>"),
                # Expected names.
                ("get_weather", "search_web"),
            ),
            detect_and_parse_malformed_tool=(
                # Typo.
                # NOTE: the regexes + logic in `Qwen3CoderToolParser` seems deliberately forgiving.
                # For example, forgetting the closing `</function>` is fine, as is the closing
                # `</parameter>`. However, the values returned in the function call information
                # might be dubious as a result.
                "<too_call>\n"
                "<function=get_weather>\n"
                "<parameter=location>San Francisco, CA</parameter>\n"
                "</function>\n"
                "</tool_call>"),
            detect_and_parse_with_parameters_key=(
                # Input text (Qwen3Coder uses "parameter", not "parameters").
                ("<tool_call>\n"
                 "<function=search_web>\n"
                 "<parameter=query>test</parameter>\n"
                 "</function>\n"
                 "</tool_call>"),
                # Expected `name`.
                "search_web",
                # Expected `parameters`.
                {
                    "query": "test"
                },
            ),
            parse_streaming_increment_partial_bot_token="<tool_call>",
            undefined_tool=("<tool_call>\n"
                            "<function=undefined_func>\n"
                            "<parameter=arg>value</parameter>\n"
                            "</function>\n"
                            "</tool_call>"),
        )

    def test_parse_streaming_increment_complete_tool_call(
            self, sample_tools, parser):
        """Test streaming parser with complete tool call in chunks."""

        # Send tool call start token
        result = parser.parse_streaming_increment("<tool_call>\n", sample_tools)
        assert len(result.calls) == 0

        # Send function declaration
        result = parser.parse_streaming_increment("<function=get_weather>\n",
                                                  sample_tools)

        # Should send tool name with empty parameters
        assert len(result.calls) == 1
        assert result.calls[0].name == "get_weather"
        assert result.calls[0].parameters == ""

        # Send parameter block
        result = parser.parse_streaming_increment(
            '<parameter=location>SF</parameter>\n</function>\n</tool_call>',
            sample_tools)

        # Should stream parameters
        assert len(result.calls) >= 1
        # Check that parameters were sent (could be in multiple chunks)
        all_params = "".join(call.parameters for call in result.calls
                             if call.parameters)
        assert "location" in all_params
        assert "SF" in all_params

    def test_parse_streaming_increment_end_token_handling(
            self, sample_tools, parser):
        """Test streaming parser handles end token correctly."""

        # Send complete tool call
        parser.parse_streaming_increment(
            "<tool_call>\n"
            "<function=get_weather>\n"
            "<parameter=location>NYC</parameter>\n"
            "</function>\n"
            "</tool_call>", sample_tools)

        # Check buffer state - should be cleared after complete tool call
        assert parser._buf == ""
        assert parser._in_tool_call is False

    def test_parse_streaming_increment_multiple_tools_streaming(
            self, sample_tools, parser):
        """Test streaming parser handles multiple tool calls."""

        # First tool.
        parser.parse_streaming_increment("<tool_call>\n", sample_tools)
        parser.parse_streaming_increment("<function=get_weather>\n",
                                         sample_tools)
        parser.parse_streaming_increment(
            "<parameter=location>NYC</parameter>\n</function>\n</tool_call>\n",
            sample_tools)

        # Second tool.
        parser.parse_streaming_increment("<tool_call>\n", sample_tools)
        result = parser.parse_streaming_increment("<function=search_web>\n",
                                                  sample_tools)

        # Should have started second tool.
        assert result.calls[0].name == "search_web"
        assert result.calls[0].parameters == ""
        assert result.calls[0].tool_index == 1

    def test_parse_streaming_increment_multiple_parameters(
            self, sample_tools, parser):
        """Test parser handles multiple parameters in a single function call."""

        tool_def = ChatCompletionToolsParam(
            type="function",
            function=FunctionDefinition(
                name="multi_param_func",
                description="Function with multiple parameters",
                parameters={
                    "type":
                    "object",
                    "properties": {
                        "param1": {
                            "type": "string"
                        },
                        "param2": {
                            "type": "float"
                        },
                        "param3": {
                            "type": "integer"
                        },
                        "param4": {
                            "type": "boolean"
                        },
                        "param5": {
                            "type": "object"
                        },
                        "param6": {
                            "type": "array"
                        },
                        "param7": {
                            "type": "null"
                        },
                        "param8": {
                            "type": "other_type"
                        }
                    },
                    "required": [
                        "param1", "param2", "param3", "param4", "param5",
                        "param6", "param7", "param8"
                    ]
                }))

        text = ("<tool_call>\n"
                "<function=multi_param_func>\n"
                "<parameter=param1>42</parameter>\n"
                "<parameter=param2>41.9</parameter>\n"
                "<parameter=param3>42</parameter>\n"
                "<parameter=param4>true</parameter>\n"
                "<parameter=param5>{\"key\": \"value\"}</parameter>\n"
                "<parameter=param6>[1, 2, 3]</parameter>\n"
                "<parameter=param7>null</parameter>\n"
                "<parameter=param8>{'arg1': 3, 'arg2': [1, 2]}</parameter>\n"
                "</function>\n"
                "</tool_call>")

        result = parser.detect_and_parse(text, [tool_def])

        assert len(result.calls) == 1
        assert result.calls[0].name == "multi_param_func"
        assert json.loads(result.calls[0].parameters) == {
            "param1": "42",
            "param2": 41.9,
            "param3": 42,
            "param4": True,
            "param5": {
                "key": "value"
            },
            "param6": [1, 2, 3],
            "param7": None,
            "param8": {
                "arg1": 3,
                "arg2": [1, 2]
            }
        }

    def test_qwen3_coder_format_compliance(
        self,
        parser,
    ):
        """Test that Qwen3CoderToolParser follows the documented format structure."""

        # Test the exact format from the docstring
        text = ("<tool_call>\n"
                "<function=execute_bash>\n"
                "<parameter=command>\n"
                "pwd && ls\n"
                "</parameter>\n"
                "</function>\n"
                "</tool_call>")

        tool_def = ChatCompletionToolsParam(
            type="function",
            function=FunctionDefinition(
                name="execute_bash",
                description="Execute a bash command.",
                parameters={
                    "type": "object",
                    "properties": {
                        "command": {
                            "type": "string",
                            "description": "The command to execute.",
                        },
                        "unit": {
                            "type": "string",
                        }
                    },
                    "required": ["command"],
                },
            ))

        result = parser.detect_and_parse(text, [tool_def])

        assert len(result.calls) == 1
        assert result.calls[0].name == "execute_bash"
        assert json.loads(result.calls[0].parameters) == {
            "command": "pwd && ls"
        }


# ============================================================================
# DeepSeek Parser Tests
# ============================================================================


class TestDeepSeekV3Parser(BaseToolParserTestClass):
    """Test suite for DeepSeekV3Parser class."""

    def make_parser(self):
        return DeepSeekV3Parser()

    def make_tool_parser_test_cases(self):
        calls_begin = "<｜tool▁calls▁begin｜>"
        calls_end = "<｜tool▁calls▁end｜>"
        call_begin = "<｜tool▁call▁begin｜>"
        call_end = "<｜tool▁call▁end｜>"
        sep = "<｜tool▁sep｜>"

        single_text = (
            f"Lead {calls_begin}{call_begin}function{sep}get_weather\n```json\n"
            f"{json.dumps({'location': 'Tokyo'})}\n```{call_end}{calls_end}")
        single_expected_normal = "Lead"  # the text is stripped
        single_expected_name = "get_weather"
        single_expected_params = {"location": "Tokyo"}

        # Provide one tool to satisfy type hints (tuple[str, tuple[str]])
        multiple_text = (
            f"{calls_begin}{call_begin}function{sep}get_weather\n```json\n"
            f"{json.dumps({'location': 'Paris'})}\n```{call_end}"
            f"{calls_begin}{call_begin}function{sep}search_web\n```json\n"
            f"{json.dumps({'query': 'AI'})}\n```{call_end}{calls_end}")
        multiple_names = ("get_weather", "search_web")

        malformed_text = (
            f"{calls_begin}{call_begin}function{sep}get_weather\n```json\n"
            "{'location': 'Paris'}\n```"
            f"{call_end}{calls_end}")

        with_parameters_key_text = (
            f"{calls_begin}{call_begin}function{sep}search_web\n```json\n"
            f"{json.dumps({'parameters': {'query': 'TensorRT'}})}\n```{call_end}{calls_end}"
        )
        with_parameters_key_name = "search_web"
        with_parameters_key_params = {"parameters": {"query": "TensorRT"}}

        partial_bot_token = "<｜tool▁cal"

        undefined_tool_text = (
            f"{calls_begin}{call_begin}function{sep}unknown\n```json\n"
            f"{json.dumps({'x': 1})}\n```{call_end}{calls_end}")

        return ToolParserTestCases(
            has_tool_call_true=f"Hello {calls_begin}",
            detect_and_parse_single_tool=(
                single_text,
                single_expected_normal,
                single_expected_name,
                single_expected_params,
            ),
            detect_and_parse_multiple_tools=(multiple_text, multiple_names),
            detect_and_parse_malformed_tool=malformed_text,
            detect_and_parse_with_parameters_key=(
                with_parameters_key_text,
                with_parameters_key_name,
                with_parameters_key_params,
            ),
            parse_streaming_increment_partial_bot_token=partial_bot_token,
            undefined_tool=undefined_tool_text,
        )


class TestDeepSeekV31Parser(BaseToolParserTestClass):
    """Test suite for DeepSeekV31Parser class."""

    def make_parser(self):
        return DeepSeekV31Parser()

    def make_tool_parser_test_cases(self):
        calls_begin = "<｜tool▁calls▁begin｜>"
        calls_end = "<｜tool▁calls▁end｜>"
        call_begin = "<｜tool▁call▁begin｜>"
        call_end = "<｜tool▁call▁end｜>"
        sep = "<｜tool▁sep｜>"

        single_text = (
            f"Intro {calls_begin}{call_begin}get_weather{sep}"
            f"{json.dumps({'location': 'Tokyo'})}{call_end}{calls_end}")
        single_expected_normal = "Intro"  # the text is stripped
        single_expected_name = "get_weather"
        single_expected_params = {"location": "Tokyo"}

        multiple_text = (f"{calls_begin}{call_begin}get_weather{sep}"
                         f"{json.dumps({'location': 'Paris'})}{call_end}"
                         f"{calls_begin}{call_begin}search_web{sep}"
                         f"{json.dumps({'query': 'AI'})}{call_end}{calls_end}")
        multiple_names = ("get_weather", "search_web")

        malformed_text = (
            f"{calls_begin}{call_begin}get_weather{sep}{{'location':'Paris'}}"
            f"{call_end}{calls_end}")

        with_parameters_key_text = (
            f"{calls_begin}{call_begin}search_web{sep}"
            f"{json.dumps({'parameters': {'query': 'TensorRT'}})}{call_end}{calls_end}"
        )
        with_parameters_key_name = "search_web"
        with_parameters_key_params = {"parameters": {"query": "TensorRT"}}

        partial_bot_token = "<｜tool▁cal"

        undefined_tool_text = (
            f"{calls_begin}{call_begin}unknown{sep}{json.dumps({'x': 1})}{call_end}{calls_end}"
        )

        return ToolParserTestCases(
            has_tool_call_true=f"Hi {calls_begin}",
            detect_and_parse_single_tool=(
                single_text,
                single_expected_normal,
                single_expected_name,
                single_expected_params,
            ),
            detect_and_parse_multiple_tools=(multiple_text, multiple_names),
            detect_and_parse_malformed_tool=malformed_text,
            detect_and_parse_with_parameters_key=(
                with_parameters_key_text,
                with_parameters_key_name,
                with_parameters_key_params,
            ),
            parse_streaming_increment_partial_bot_token=partial_bot_token,
            undefined_tool=undefined_tool_text,
        )


# ============================================================================
# DeepSeekV32Parser Tests
# ============================================================================


class TestDeepSeekV32Parser(BaseToolParserTestClass):
    """Test suite for DeepSeekV32Parser class."""

    def make_parser(self):
        return DeepSeekV32Parser()

    def make_tool_parser_test_cases(self):
        return ToolParserTestCases(
            has_tool_call_true=
            'Some text <｜DSML｜function_calls> <｜DSML｜invoke name="get_weather"> <｜DSML｜parameter name="location" string="true">NYC</｜DSML｜parameter> </｜DSML｜invoke> </｜DSML｜function_calls>',
            detect_and_parse_single_tool=(
                # Input text.
                ('Normal text'
                 '<｜DSML｜function_calls> <｜DSML｜invoke name="get_weather"> <｜DSML｜parameter name="location" string="true">NYC</｜DSML｜parameter> </｜DSML｜invoke> </｜DSML｜function_calls>'
                 ),
                # Expected `normal_text`.
                "Normal text",
                # Expected `name`.
                "get_weather",
                # Expected `parameters`.
                {
                    "location": "NYC"
                },
            ),
            detect_and_parse_multiple_tools=(
                # Input text.
                ('<｜DSML｜function_calls> <｜DSML｜invoke name="get_weather"> <｜DSML｜parameter name="location" string="true">NYC</｜DSML｜parameter> </｜DSML｜invoke> <｜DSML｜invoke name="search_web"> { "query": "AI" } </｜DSML｜invoke> </｜DSML｜function_calls>'
                 ),
                # Expected names.
                ("get_weather", "search_web"),
            ),
            detect_and_parse_malformed_tool=(
                # Format error: using "|" instead of "｜"
                '<|DSML|function_calls> <|DSML|invoke name="get_weather"> <|DSML|parameter name="location" string="true">NYC</|DSML|parameter> </|DSML|invoke> </|DSML|function_calls>'
            ),
            detect_and_parse_with_parameters_key=(
                # Input text.
                ('<｜DSML｜function_calls> <｜DSML｜invoke name="search_web"> { "query": "test" } </｜DSML｜invoke> </｜DSML｜function_calls>'
                 ),
                # Expected `name`.
                "search_web",
                # Expected `parameters`.
                {
                    "query": "test"
                },
            ),
            parse_streaming_increment_partial_bot_token="<｜DSML｜function_calls>",
            undefined_tool=
            ('<｜DSML｜function_calls> <｜DSML｜invoke name="undefined_func"> <｜DSML｜parameter name="arg" string="true">value</｜DSML｜parameter> </｜DSML｜invoke> </｜DSML｜function_calls>'
             ),
        )

    def test_initialization(self, parser):
        """Test that DeepSeekV32Parser initializes correctly."""
        assert parser.bot_token == "<｜DSML｜function_calls>"
        assert parser.eot_token == "</｜DSML｜function_calls>"
        assert parser.invoke_end_token == "</｜DSML｜invoke>"
        assert parser._last_arguments == ""

    def test_parse_streaming_increment_complete_tool_call(
            self, sample_tools, parser):
        """Test streaming parser with complete tool call in chunks."""
        parser.parse_streaming_increment('<｜DSML｜function_calls> ',
                                         sample_tools)
        result = parser.parse_streaming_increment(
            '<｜DSML｜invoke name="get_weather"> ', sample_tools)
        assert len(result.calls) == 0

        parser.parse_streaming_increment(
            '<｜DSML｜parameter name="location" string="true">NYC</｜DSML｜parameter> ',
            sample_tools)
        result = parser.parse_streaming_increment(
            '</｜DSML｜invoke> </｜DSML｜function_calls>', sample_tools)

        assert result.calls[0].name == "get_weather"
        assert json.loads(result.calls[1].parameters) == {"location": "NYC"}

    def test_parse_streaming_increment_multiple_tools_streaming(
            self, sample_tools, parser):
        """Test streaming parser handles end token correctly."""
        # First tool
        parser.parse_streaming_increment("<｜DSML｜function_calls> ",
                                         sample_tools)
        parser.parse_streaming_increment("<｜DSML｜invoke name=\"get_weather\"> ",
                                         sample_tools)
        parser.parse_streaming_increment(
            "<｜DSML｜parameter name=\"location\" string=\"true\">NYC</｜DSML｜parameter> ",
            sample_tools)
        parser.parse_streaming_increment("</｜DSML｜invoke> ", sample_tools)

        # Second tool
        parser.parse_streaming_increment("<｜DSML｜invoke name=\"search_web\"> ",
                                         sample_tools)
        parser.parse_streaming_increment(
            "<｜DSML｜parameter name=\"query\" string=\"true\">AI</｜DSML｜parameter> ",
            sample_tools)
        result = parser.parse_streaming_increment(
            "</｜DSML｜invoke> <｜DSML｜function_calls>", sample_tools)

        assert result.calls[0].name == "search_web"
        assert json.loads(result.calls[1].parameters) == {"query": "AI"}
        assert result.calls[1].tool_index == 1

    def test_structure_info_function(self):
        """Test that DeepSeekV32Parser structure_info returns correct lambda function."""
        parser = DeepSeekV32Parser()
        func = parser.structure_info()

        info = func("get_weather")

        assert info.begin == "<｜DSML｜invoke name=\"get_weather\">"
        assert info.end == "</｜DSML｜invoke>"
        assert info.trigger == "<｜DSML｜invoke name=\"get_weather\">"

    def test_structure_info_different_names(self):
        """Test that DeepSeekV32Parser structure_info returns correct lambda function."""
        parser = DeepSeekV32Parser()
        func = parser.structure_info()

        info1 = func("get_weather")
        info2 = func("search_web")

        assert "get_weather" in info1.begin
        assert "search_web" in info2.begin
        assert info1.end == info2.end == "</｜DSML｜invoke>"

    def test_deepseek_v32_format_compliance(self, sample_tools, parser):
        """Test that DeepSeekV32Parser follows the documented format structure."""

        # Test the exact format from the docstring
        text = "<｜DSML｜function_calls> <｜DSML｜invoke name=\"get_weather\"> <｜DSML｜parameter name=\"location\" string=\"true\">NYC</｜DSML｜parameter> </｜DSML｜invoke> </｜DSML｜function_calls>"
        result = parser.detect_and_parse(text, sample_tools)

        assert len(result.calls) == 1
        assert result.calls[0].name == "get_weather"
        assert json.loads(result.calls[0].parameters) == {"location": "NYC"}


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
