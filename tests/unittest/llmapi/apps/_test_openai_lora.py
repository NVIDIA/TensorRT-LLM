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

pytestmark = pytest.mark.threadleak(enabled=False)


@pytest.fixture(scope="module", ids=["Qwen3/Qwen3-0.6B"])
def model_name() -> str:
    return "Qwen3/Qwen3-0.6B"


@pytest.fixture(scope="module")
def lora_adapter_path() -> Generator[Path, None, None]:
    with qwen3_lora_adapter() as adapter_path:
        yield adapter_path


@pytest.fixture(scope="module")
def temp_extra_llm_api_options_file():
    temp_dir = tempfile.gettempdir()
    temp_file_path = os.path.join(temp_dir, "extra_llm_api_options.yaml")
    try:
        extra_llm_api_options_dict = {
            "lora_config": {
                "lora_target_modules": ['attn_q', 'attn_k', 'attn_v'],
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


def test_lora(client: openai.OpenAI, model_name: str,
              lora_adapter_path: Path) -> None:
    prompt = "The capital of France is"

    def complete(
        extra_body: dict[str, object] | None = None
    ) -> tuple[str, tuple[str, ...], tuple[float | None, ...]]:
        response = client.completions.create(
            model=model_name,
            prompt=prompt,
            max_tokens=20,
            temperature=0.0,
            logprobs=1,
            extra_body=extra_body or {},
        )
        choice = response.choices[0]
        assert choice.logprobs is not None
        assert choice.logprobs.tokens is not None
        assert choice.logprobs.token_logprobs is not None
        return (
            choice.text,
            tuple(choice.logprobs.tokens),
            tuple(choice.logprobs.token_logprobs),
        )

    base_output = complete()
    lora_request = LoRARequest(lora_name=lora_adapter_path.name,
                               lora_int_id=1,
                               lora_path=str(lora_adapter_path))
    extra_body = {"lora_request": asdict(lora_request)}
    first_lora_output = complete(extra_body)
    reused_lora_output = complete(extra_body)

    assert base_output[0]
    assert first_lora_output[0]
    output_changed = first_lora_output[:2] != base_output[:2]
    logprobs_changed = first_lora_output[2] != pytest.approx(base_output[2],
                                                             abs=1e-4)
    assert output_changed or logprobs_changed
    assert reused_lora_output[:2] == first_lora_output[:2]
    # Adapter reuse can shift same-token logprobs by about 0.06 due to
    # numerically equivalent kernel execution, but larger drift is suspicious.
    assert reused_lora_output[2] == pytest.approx(first_lora_output[2],
                                                  abs=0.075)
