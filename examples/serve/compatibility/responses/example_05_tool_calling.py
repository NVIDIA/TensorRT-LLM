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

#!/usr/bin/env python3
"""Example 5: Tool/Function Calling.

Demonstrates tool calling with function definitions and responses.

Note: This requires a compatible model (e.g., Qwen3, GPT-OSS, Kimi K2).
"""

import json

from openai import OpenAI

# Initialize the client
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="tensorrt_llm",
)

# Get the model name from the server
models = client.models.list()
model = models.data[0].id
TOOL_CALL_SUPPORTED_MODELS = ["Qwen3", "GPT-OSS", "Kimi K2"]

print("=" * 80)
print("Example 5: Tool/Function Calling")
print("=" * 80)
print()
print(
    f"Note: Tool calling requires compatible models (e.g. {', '.join(TOOL_CALL_SUPPORTED_MODELS)})\n"
)

# Define the available tools
tools = [
    {
        "name": "get_weather",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "City and state, e.g. San Francisco, CA",
                },
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
            },
            "required": ["location"],
        },
        "type": "function",
        "description": "Get the current weather in a location",
    }
]


def get_weather(location: str, unit: str = "fahrenheit") -> dict:
    return {"location": location, "temperature": 68, "unit": unit, "conditions": "sunny"}


def process_tool_call(response) -> tuple[dict, str]:
    function_name = None
    function_arguments = None
    tool_call_id = None
    for output in response.output:
        if output.type == "function_call":
            function_name = output.name
            function_arguments = json.loads(output.arguments)
            tool_call_id = output.call_id
            break

    try:
        print(
            f"Get tool call result:\n\ttool_name: {function_name}\n\tparameters: {function_arguments})"
        )
        result = eval(f"{function_name}(**{function_arguments})")
    except Exception as e:
        print(f"Error processing tool call: {e}")
        return None, None

    return result, tool_call_id


print("Available tools:")
print(json.dumps(tools, indent=2))
print("\nUser query: What is the weather in San Francisco?\n")

try:
    # Initial request with tools
    response = client.responses.create(
        model=model,
        input="What is the weather in San Francisco?",
        tools=tools,
        tool_choice="auto",
        max_output_tokens=4096,
    )

    tool_call_result, tool_call_id = process_tool_call(response)
    call_input = [
        {
            "type": "function_call_output",
            "call_id": tool_call_id,
            "output": json.dumps(tool_call_result),
        }
    ]

    prev_response_id = response.id
    response = client.responses.create(
        model=model,
        input=call_input,
        previous_response_id=prev_response_id,
        tools=tools,
    )

    print(f"Final response: {response.output_text}")

except Exception as e:
    print(
        f"Note: Tool calling requires model support (e.g. {', '.join(TOOL_CALL_SUPPORTED_MODELS)})"
    )
    print(f"Error: {e}")
