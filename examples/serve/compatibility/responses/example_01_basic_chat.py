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
"""Example 1: Basic Non-Streaming Responses.

Demonstrates a simple responses request with the OpenAI-compatible API.
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
print("Example 1: Basic Non-Streaming Responses")
print("=" * 80)
print()

# Create a simple responses request
response = client.responses.create(
    model=model,
    input="What is the capital of France?",
    max_output_tokens=4096,
)

# Print the response
print("Response:")
print(f"Content: {response.output_text}")
