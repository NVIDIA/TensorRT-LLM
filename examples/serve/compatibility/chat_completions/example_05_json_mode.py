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
"""Example 5: JSON Mode with Schema.

Demonstrates structured output generation with JSON schema validation.

Note: This requires xgrammar support and compatible model configuration.
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
print("Example 5: JSON Mode with Schema")
print("=" * 80)
print()

# Define the JSON schema
schema = {
    "name": "city_info",
    "schema": {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "country": {"type": "string"},
            "population": {"type": "integer"},
            "famous_for": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["name", "country", "population"],
    },
}

print("Request with JSON schema:")
print(json.dumps(schema, indent=2))
print()
print("Note: JSON schema support requires xgrammar and compatible model configuration.\n")

try:
    # Create chat completion with JSON schema
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant that outputs JSON."},
            {"role": "user", "content": "Give me information about Tokyo."},
        ],
        response_format={"type": "json_schema", "json_schema": schema},
        max_tokens=4096,
    )

    print("JSON Response:")
    result = json.loads(response.choices[0].message.content)
    print(json.dumps(result, indent=2))
except Exception as e:
    print("JSON schema support requires xgrammar and proper configuration.")
    print(f"Error: {e}")
