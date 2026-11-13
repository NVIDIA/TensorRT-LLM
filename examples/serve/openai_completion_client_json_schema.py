### :title OpenAI Completion Client with JSON Schema

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# This example requires to specify `guided_decoding_backend` as
# `xgrammar` or `llguidance` in the config.yaml file.
import json

from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="tensorrt_llm",
)

response = client.chat.completions.create(
    model="TinyLlama-1.1B-Chat-v1.0",
    messages=[{
        "role": "system",
        "content": "you are a helpful assistant"
    }, {
        "role":
        "user",
        "content":
        f"Give me the information of the biggest city of China in the JSON format.",
    }],
    temperature=0,
    response_format={
        "type": "json",
        "schema": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string"
                },
                "population": {
                    "type": "integer"
                },
            },
            "required": ["name", "population"],
        }
    },
    # The OpenAI client has no chat_template_kwargs argument; send it in the
    # request body so the chat template actually receives it.
    extra_body={"chat_template_kwargs": {
        "enable_thinking": False
    }},
)

content = response.choices[0].message.content
try:
    response_json = json.loads(content)
    assert "name" in response_json and "population" in response_json
    print(content)
except json.JSONDecodeError as e:
    raise RuntimeError(f"Failed to decode JSON response: {content!r}") from e
