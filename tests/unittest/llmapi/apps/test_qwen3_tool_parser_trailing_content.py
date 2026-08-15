# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from tensorrt_llm.serve.openai_protocol import (
    ChatCompletionToolsParam,
    FunctionDefinition,
)
from tensorrt_llm.serve.tool_parser.qwen3_tool_parser import Qwen3ToolParser

pytestmark = pytest.mark.cpu_only


def test_streaming_wrapped_form_preserves_text_after_tool_call():
    tools = [
        ChatCompletionToolsParam(
            type="function",
            function=FunctionDefinition(
                name="get_weather",
                description="Get the current weather",
                parameters={
                    "type": "object",
                    "properties": {"location": {"type": "string"}},
                    "required": ["location"],
                },
            ),
        )
    ]
    parser = Qwen3ToolParser()

    parser.parse_streaming_increment("<tool_call>\n", tools)
    parser.parse_streaming_increment(
        '{"name":"get_weather","arguments":{"location":"Paris"}}', tools
    )
    parser.parse_streaming_increment("\n</tool_call>", tools)

    result = parser.parse_streaming_increment(" It is sunny.", tools)

    assert result.normal_text == " It is sunny."
    assert result.calls == []
    assert parser._buffer == ""
