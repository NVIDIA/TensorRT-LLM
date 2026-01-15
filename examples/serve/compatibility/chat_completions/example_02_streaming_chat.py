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
"""Example 2: Streaming Chat Completion.

Demonstrates streaming responses with real-time token delivery.
"""

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
print("Example 2: Streaming Chat Completion")
print("=" * 80)
print()

print("Prompt: Write a haiku about artificial intelligence\n")

# Create a streaming chat completion
stream = client.chat.completions.create(
    model=model,
    messages=[{"role": "user", "content": "Write a haiku about artificial intelligence"}],
    max_tokens=4096,
    temperature=0.8,
    stream=True,
)

# Print tokens as they arrive
print("Response (streaming):")
print("Assistant: ", end="", flush=True)

current_state = "none"
for chunk in stream:
    has_content = hasattr(chunk.choices[0].delta, "content") and chunk.choices[0].delta.content
    has_reasoning_content = (
        hasattr(chunk.choices[0].delta, "reasoning_content")
        and chunk.choices[0].delta.reasoning_content
    )
    if has_content:
        if current_state != "content":
            print("Content: ", end="", flush=True)
            current_state = "content"

        print(chunk.choices[0].delta.content, end="", flush=True)

    if has_reasoning_content:
        if current_state != "reasoning_content":
            print("Reasoning: ", end="", flush=True)
            current_state = "reasoning_content"

        print(chunk.choices[0].delta.reasoning_content, end="", flush=True)
print("\n")

print("Stop reason: ", chunk.choices[0].finish_reason)
