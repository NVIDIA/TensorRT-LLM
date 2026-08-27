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

import openai
import pytest
import yaml

from tensorrt_llm.executor.request import LoRARequest

from ..lora_test_utils import qwen3_5_lora_adapter
from ..test_llm import get_model_path
from .openai_server import RemoteOpenAIServer

pytestmark = pytest.mark.threadleak(enabled=False)

_LORA_ADAPTER_NAME = "qwen3.5-test-lora"


@pytest.fixture(scope="module", ids=["Qwen3.5-4B"])
def model_name() -> str:
    return "Qwen3.5-4B"


@pytest.fixture(scope="module")
def lora_adapter_path():
    with qwen3_5_lora_adapter() as adapter_path:
        yield adapter_path


@pytest.fixture(scope="module")
def temp_extra_llm_api_options_file(lora_adapter_path: str):
    temp_dir = tempfile.gettempdir()
    temp_file_path = os.path.join(temp_dir, "extra_llm_api_options.yaml")
    try:
        extra_llm_api_options_dict = {
            "max_batch_size": 8,
            "lora_config": {
                "lora_dir": [lora_adapter_path],
                "lora_target_modules": ["attn_dense"],
                "max_lora_rank": 8,
                "max_loras": 4,
                "max_cpu_loras": 4,
            },
            # Disable CUDA graph
            # TODO: remove this once we have a proper fix for CUDA graph in LoRA
            "cuda_graph_config": None
        }

        with open(temp_file_path, 'w') as f:
            yaml.dump(extra_llm_api_options_dict, f)

        yield temp_file_path
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)


@pytest.fixture(scope="module")
def server(model_name: str,
           temp_extra_llm_api_options_file: str) -> RemoteOpenAIServer:
    model_path = get_model_path(model_name)
    args = []
    args.extend(["--backend", "pytorch"])
    args.extend(["--extra_llm_api_options", temp_extra_llm_api_options_file])
    with RemoteOpenAIServer(model_path, args) as remote_server:
        yield remote_server


@pytest.fixture(scope="module")
def client(server: RemoteOpenAIServer) -> openai.OpenAI:
    return server.get_client()


@pytest.mark.parametrize(
    "prompt,use_lora",
    [
        pytest.param(
            "The capital of France is",
            False,
            id="capital-prompt-base",
        ),
        pytest.param(
            "The capital of France is",
            True,
            id="capital-prompt-lora-first-load",
        ),
        pytest.param(
            "The capital of France is",
            True,
            id="capital-prompt-lora-reuse",
        ),
        pytest.param(
            "Two plus two equals",
            False,
            id="arithmetic-prompt-base",
        ),
        pytest.param(
            "Two plus two equals",
            True,
            id="arithmetic-prompt-lora-first-load",
        ),
        pytest.param(
            "Two plus two equals",
            True,
            id="arithmetic-prompt-lora-reuse",
        ),
    ],
)
def test_lora(client: openai.OpenAI, model_name: str, lora_adapter_path: str,
              prompt: str, use_lora: bool):
    base_response = None
    if use_lora:
        base_response = client.completions.create(
            model=model_name,
            prompt=prompt,
            max_tokens=20,
            temperature=0.0,
            logprobs=1,
        )
    extra_body = {}
    if use_lora:
        lora_req = LoRARequest(_LORA_ADAPTER_NAME, 1, lora_adapter_path)
        extra_body["lora_request"] = asdict(lora_req)

    response = client.completions.create(
        model=model_name,
        prompt=prompt,
        max_tokens=20,
        temperature=0.0,
        logprobs=1,
        extra_body=extra_body,
    )
    assert response.choices
    assert response.usage.completion_tokens > 0
    if use_lora:
        assert base_response is not None
        lora_choice = response.choices[0]
        base_choice = base_response.choices[0]
        assert (lora_choice.text != base_choice.text
                or lora_choice.logprobs.token_logprobs
                != base_choice.logprobs.token_logprobs)
