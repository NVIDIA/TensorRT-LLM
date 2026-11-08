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
"""Example 3: Multi-turn Conversation.

Demonstrates maintaining conversation context across multiple turns.
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
print("Example 3: Multi-turn Conversation")
print("=" * 80)
print()

# Start a conversation with system message
messages = [
    {"role": "system", "content": "You are an expert mathematician."},
]

# First turn: User asks a question
messages.append({"role": "user", "content": "What is 15 multiplied by 23?"})
print("USER: What is 15 multiplied by 23?")

response1 = client.chat.completions.create(
    model=model,
    messages=messages,
    max_tokens=4096,
    temperature=0,
)

assistant_reply_1 = response1.choices[0].message.content
print(f"ASSISTANT: {assistant_reply_1}\n")

# Add assistant's response to conversation history
messages.append({"role": "assistant", "content": assistant_reply_1})

# Second turn: User asks a follow-up question
messages.append({"role": "user", "content": "Now divide that result by 5"})
print("USER: Now divide that result by 5")

response2 = client.chat.completions.create(
    model=model,
    messages=messages,
    max_tokens=4096,
    temperature=0,
)

assistant_reply_2 = response2.choices[0].message.content
print(f"ASSISTANT: {assistant_reply_2}")
