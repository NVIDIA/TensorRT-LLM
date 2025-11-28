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
"""Example 6: Tool/Function Calling.

Demonstrates tool calling with function definitions and responses.

Note: This requires a compatible model (e.g., Llama 3.1+, Mistral Instruct).
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

print("=" * 80)
print("Example 6: Tool/Function Calling")
print("=" * 80)
print()
print("Note: Tool calling requires compatible models (e.g., Llama 3.1+)\n")

# Define the available tools
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather in a location",
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
        },
    }
]


def get_weather(location: str, unit: str = "fahrenheit") -> dict:
    return {"location": location, "temperature": 68, "unit": unit, "conditions": "sunny"}


print("Available tools:")
print(json.dumps(tools, indent=2))
print("\nUser query: What is the weather in San Francisco?\n")

try:
    # Initial request with tools
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "What is the weather in San Francisco?"}],
        tools=tools,
        tool_choice="auto",
        max_tokens=4096,
    )

    message = response.choices[0].message

    if message.tool_calls:
        print("Tool calls requested:")
        for tool_call in message.tool_calls:
            print(f"  Function: {tool_call.function.name}")
            print(f"  Arguments: {tool_call.function.arguments}")

        # Simulate function execution
        print("\nSimulating function execution...")
        function_response = get_weather(**json.loads(tool_call.function.arguments))
        print(f"Function result: {json.dumps(function_response, indent=2)}")

        # Send function result back to get final response
        messages = [
            {"role": "user", "content": "What is the weather in San Francisco?"},
            message,
            {
                "role": "tool",
                "tool_call_id": message.tool_calls[0].id,
                "content": json.dumps(function_response),
            },
        ]

        final_response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=4096,
        )

        print(f"\nFinal response: {final_response.choices[0].message.content}")
    else:
        print(f"Direct response: {message.content}")
except Exception as e:
    print("Note: Tool calling requires model support (e.g., Llama 3.1+ models)")
    print(f"Error: {e}")
