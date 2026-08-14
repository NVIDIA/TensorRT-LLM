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
"""Example 7: Advanced Sampling Parameters.

Demonstrates TensorRT-LLM specific sampling parameters for fine-tuned control.
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
print("Example 7: Advanced Sampling Parameters")
print("=" * 80)
print()

print("Using TensorRT-LLM extended parameters:")
print("  - top_k: 50")
print("  - repetition_penalty: 1.1")
print("  - min_tokens: 20")
print("  - stop sequences: ['The End', '\\n\\n\\n']")
print()

# Create completion with advanced sampling parameters
response = client.chat.completions.create(
    model=model,
    messages=[{"role": "user", "content": "Write a very short story about a robot."}],
    max_tokens=4096,
    temperature=0.8,
    top_p=0.95,
    extra_body={
        "top_k": 50,
        "repetition_penalty": 1.1,
        "min_tokens": 20,
        "stop": ["The End", "\n\n\n"],
    },
)

print("Story:")
print(response.choices[0].message.content)
print(f"\nFinish reason: {response.choices[0].finish_reason}")
