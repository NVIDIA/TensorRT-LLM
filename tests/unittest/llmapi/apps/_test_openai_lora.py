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
from typing import Optional

import openai
import pytest
import yaml
from utils.llm_data import llm_models_root

from tensorrt_llm.executor.request import LoRARequest

from ..test_llm import get_model_path
from .openai_server import RemoteOpenAIServer

pytestmark = pytest.mark.threadleak(enabled=False)

_CHINESE_LORA_ADAPTER = "lora/llama-3-chinese-8b-instruct-v2-lora"


def _get_lora_adapter_path(adapter_name: str) -> str:
    """Resolve LoRA adapters under LLM_MODELS_ROOT, ignoring LLM_ENGINE_DIR."""
    return str(llm_models_root() / adapter_name)


@pytest.fixture(scope="module", ids=["Llama-3.1-8B-Instruct"])
def model_name() -> str:
    return "Llama-3.1-8B-Instruct"


@pytest.fixture(scope="module")
def temp_extra_llm_api_options_file():
    temp_dir = tempfile.gettempdir()
    temp_file_path = os.path.join(temp_dir, "extra_llm_api_options.yaml")
    try:
        extra_llm_api_options_dict = {
            "lora_config": {
                "lora_dir": [_get_lora_adapter_path(_CHINESE_LORA_ADAPTER)],
                "max_lora_rank": 64,
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
    model_path = get_model_path(f"llama-3.1-model/{model_name}")
    args = []
    args.extend(["--backend", "pytorch"])
    args.extend(["--extra_llm_api_options", temp_extra_llm_api_options_file])
    with RemoteOpenAIServer(model_path, args) as remote_server:
        yield remote_server


@pytest.fixture(scope="module")
def client(server: RemoteOpenAIServer) -> openai.OpenAI:
    return server.get_client()


@pytest.mark.parametrize(
    "prompt,reference,lora_adapter_name",
    [
        pytest.param(
            "美国的首都在哪里? \n答案:",
            "华盛顿特区",
            None,
            id="chinese-prompt-base",
        ),
        pytest.param(
            "美国的首都在哪里? \n答案:",
            "华盛顿特区",
            _CHINESE_LORA_ADAPTER,
            id="chinese-prompt-lora-first-load",
        ),
        pytest.param(
            "美国的首都在哪里? \n答案:",
            "华盛顿特区",
            _CHINESE_LORA_ADAPTER,
            id="chinese-prompt-lora-reuse",
        ),
        pytest.param(
            "アメリカ合衆国の首都はどこですか? \n答え:",
            "ワシントン",
            None,
            id="japanese-prompt-base",
        ),
        pytest.param(
            "アメリカ合衆国の首都はどこですか? \n答え:",
            "ワシントン",
            _CHINESE_LORA_ADAPTER,
            id="japanese-prompt-lora-first-load",
        ),
        pytest.param(
            "アメリカ合衆国の首都はどこですか? \n答え:",
            "ワシントン",
            _CHINESE_LORA_ADAPTER,
            id="japanese-prompt-lora-reuse",
        ),
    ],
)
def test_lora(client: openai.OpenAI, model_name: str, prompt: str,
              reference: str, lora_adapter_name: Optional[str]):
    extra_body = {}
    if lora_adapter_name is not None:
        lora_req = LoRARequest(lora_adapter_name, 1,
                               _get_lora_adapter_path(lora_adapter_name))
        extra_body["lora_request"] = asdict(lora_req)

    response = client.completions.create(
        model=model_name,
        prompt=prompt,
        max_tokens=20,
        temperature=0.0,
        extra_body=extra_body,
    )
    output = response.choices[0].text
    assert reference in output, (
        f"Unexpected completion for LoRA adapter {lora_adapter_name!r}.\n"
        f"Response: {output!r}\n"
        f"Expected response to contain: {reference!r}")
