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

import os
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Generator

import openai
import pytest
import yaml

from tensorrt_llm.executor.request import LoRARequest

from ..lora_test_utils import qwen3_lora_adapter
from ..test_llm import get_model_path
from .openai_server import RemoteOpenAIServer


@pytest.fixture(scope="module", ids=["Qwen3/Qwen3-0.6B"])
def model_name() -> str:
    return "Qwen3/Qwen3-0.6B"


@pytest.fixture(scope="module")
def lora_adapter_path() -> Generator[Path, None, None]:
    with qwen3_lora_adapter() as adapter_path:
        yield adapter_path


@pytest.fixture(scope="module")
def temp_extra_llm_api_options_file(
        lora_adapter_path: Path) -> Generator[str, None, None]:
    temp_dir = tempfile.gettempdir()
    temp_file_path = os.path.join(temp_dir, "extra_llm_api_options.yaml")
    try:
        extra_llm_api_options_dict = {
            "lora_config": {
                "lora_target_modules": ['attn_q', 'attn_k', 'attn_v'],
                "max_lora_rank": 8,
                "max_loras": 4,
                "max_cpu_loras": 4,
            }
        }

        with open(temp_file_path, 'w') as f:
            yaml.dump(extra_llm_api_options_dict, f)

        yield temp_file_path
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)


@pytest.fixture(scope="module")
def server(
    model_name: str, temp_extra_llm_api_options_file: str
) -> Generator[RemoteOpenAIServer, None, None]:
    model_path = get_model_path(model_name)
    args = [
        "--backend", "pytorch", "--extra_llm_api_options",
        temp_extra_llm_api_options_file
    ]
    with RemoteOpenAIServer(model_path, port=8000,
                            cli_args=args) as remote_server:
        yield remote_server


@pytest.fixture(scope="module")
def client(server: RemoteOpenAIServer) -> openai.OpenAI:
    return server.get_client()


def test_trtllm_serve_lora_completion(client: openai.OpenAI, model_name: str,
                                      lora_adapter_path: Path) -> None:
    lora_request = LoRARequest(lora_name=lora_adapter_path.name,
                               lora_int_id=0,
                               lora_path=str(lora_adapter_path))
    response = client.completions.create(
        model=model_name,
        prompt="The capital of France is",
        max_tokens=20,
        extra_body={"lora_request": asdict(lora_request)},
    )

    assert response.choices[0].text
