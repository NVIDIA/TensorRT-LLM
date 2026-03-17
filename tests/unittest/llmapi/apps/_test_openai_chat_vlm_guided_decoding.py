# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# Regression test for: VLM wrapper classes missing vocab_size_padded, causing
# AttributeError at server startup when guided decoding is configured.
# https://github.com/NVIDIA/TensorRT-LLM/pull/12284

import json
import os
import sys
import tempfile

import jsonschema
import openai
import pytest
import yaml
from utils.llm_data import llm_models_root

from .openai_server import RemoteOpenAIServer

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from test_llm import get_model_path

pytestmark = pytest.mark.threadleak(enabled=False)

_MODEL_NAME = "Qwen/Qwen3-VL-8B-Instruct"
_IMAGE_URL = str(llm_models_root() / "multimodals" / "test_data" / "seashore.png")

_SCHEMA = {
    "type": "object",
    "properties": {
        "subject": {"type": "string", "description": "The main subject visible in the image."},
        "setting": {"type": "string", "description": "The setting or environment of the image."},
    },
    "required": ["subject", "setting"],
    "additionalProperties": False,
}


@pytest.fixture(scope="module")
def temp_extra_llm_api_options_file():
    temp_dir = tempfile.gettempdir()
    temp_file_path = os.path.join(temp_dir, "extra_llm_api_options_vlm_guided.yaml")
    try:
        extra_llm_api_options_dict = {
            "guided_decoding_backend": "xgrammar",
            "kv_cache_config": {
                "free_gpu_memory_fraction": 0.8,
            },
            "max_num_tokens": 4096,
        }
        with open(temp_file_path, "w") as f:
            yaml.dump(extra_llm_api_options_dict, f)
        yield temp_file_path
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)


@pytest.fixture(scope="module")
def server(temp_extra_llm_api_options_file: str):
    model_path = get_model_path(_MODEL_NAME)
    args = ["--extra_llm_api_options", temp_extra_llm_api_options_file]
    with RemoteOpenAIServer(model_path, cli_args=args) as remote_server:
        yield remote_server


@pytest.fixture(scope="module")
def client(server: RemoteOpenAIServer):
    return server.get_client()


def test_vlm_guided_decoding_json_schema(client: openai.OpenAI):
    response = client.chat.completions.create(
        model=_MODEL_NAME,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": _IMAGE_URL}},
                    {"type": "text", "text": "Describe the main subject of this image."},
                ],
            }
        ],
        max_completion_tokens=256,
        temperature=0.0,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "image_description",
                "strict": True,
                "schema": _SCHEMA,
            },
        },
    )

    content = response.choices[0].message.content
    assert content is not None

    parsed = json.loads(content)
    jsonschema.validate(parsed, _SCHEMA)
