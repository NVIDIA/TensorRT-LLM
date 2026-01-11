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

# First turn: User asks a question
print("USER: What is 15 multiplied by 23?")

response1 = client.responses.create(
    model=model,
    input="What is 15 multiplied by 23?",
    max_output_tokens=4096,
)

assistant_reply_1 = response1.output_text
print(f"ASSISTANT: {assistant_reply_1}\n")

# Second turn: User asks a follow-up question
print("USER: Now divide that result by 5")

# No context need to be provided for the second turn, only include the previous response id
response2 = client.responses.create(
    model=model,
    input="Now divide that result by 5",
    max_output_tokens=4096,
    previous_response_id=response1.id,
)

assistant_reply_2 = response2.output_text
print(f"ASSISTANT: {assistant_reply_2}")
