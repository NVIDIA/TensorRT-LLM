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
### OpenAI Completion Client

import os
from pathlib import Path

from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="tensorrt_llm",
)

lora_path = (Path(os.environ.get("LLM_MODELS_ROOT")) / "lora" /
             "llama-3-chinese-8b-instruct-v2-lora")
assert lora_path.exists(), f"Lora path {lora_path} does not exist"

response = client.completions.create(
    model="Llama-3.1-8B-Instruct",
    prompt="美国的首都在哪里? \n答案:",
    max_tokens=20,
    extra_body={
        "lora_request": {
            "lora_name": "llama-3-chinese-8b-instruct-v2-lora",
            "lora_int_id": 0,
            "lora_path": str(lora_path)
        }
    },
)

print(response)
