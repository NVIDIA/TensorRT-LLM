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
"""Example 2: Streaming Responses.

Demonstrates streaming responses with real-time token delivery.
"""

from openai import OpenAI


def print_streaming_responses_item(item, show_events=True):
    event_type = getattr(item, "type", "")

    if event_type == "response.created":
        if show_events:
            print(f"[Response Created: {getattr(item.response, 'id', 'unknown')}]")
    elif event_type == "response.in_progress":
        if show_events:
            print("[Response In Progress]")
    elif event_type == "response.output_item.added":
        if show_events:
            item_type = getattr(item.item, "type", "unknown")
            item_id = getattr(item.item, "id", "unknown")
            print(f"\n[Output Item Added: {item_type} (id: {item_id})]")
    elif event_type == "response.content_part.added":
        if show_events:
            part_type = getattr(item.part, "type", "unknown")
            print(f"[Content Part Added: {part_type}]")
    elif event_type == "response.reasoning_text.delta":
        print(item.delta, end="", flush=True)
    elif event_type == "response.output_text.delta":
        print(item.delta, end="", flush=True)
    elif event_type == "response.reasoning_text.done":
        if show_events:
            print(f"\n[Reasoning Text Done: {len(item.text)} chars]")
    elif event_type == "response.output_text.done":
        if show_events:
            print(f"\n[Output Text Done: {len(item.text)} chars]")
    elif event_type == "response.content_part.done":
        if show_events:
            part_type = getattr(item.part, "type", "unknown")
            print(f"[Content Part Done: {part_type}]")
    elif event_type == "response.output_item.done":
        if show_events:
            item_type = getattr(item.item, "type", "unknown")
            item_id = getattr(item.item, "id", "unknown")
            print(f"[Output Item Done: {item_type} (id: {item_id})]")
    elif event_type == "response.completed":
        if show_events:
            print("\n[Response Completed]")


# Initialize the client
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="tensorrt_llm",
)

# Get the model name from the server
models = client.models.list()
model = models.data[0].id

print("=" * 80)
print("Example 2: Streaming Responses")
print("=" * 80)
print()

print("Prompt: Write a haiku about artificial intelligence\n")

# Create a streaming responses
stream = client.responses.create(
    model=model,
    input="Write a haiku about artificial intelligence",
    max_output_tokens=4096,
    stream=True,
)

# Print tokens as they arrive
print("Response (streaming):")
print("Assistant: ", end="", flush=True)

current_state = "none"
for event in stream:
    print_streaming_responses_item(event)
