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
import subprocess
import sys
import tempfile

import pytest
import yaml

from ..lora_test_utils import qwen3_5_lora_adapter
from .openai_server import RemoteOpenAIServer

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from test_llm import get_model_path


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
            "lora_config": {
                "lora_dir": [lora_adapter_path],
                "lora_target_modules": ["attn_dense"],
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
def server(model_name: str, temp_extra_llm_api_options_file: str):
    model_path = get_model_path(model_name)
    args = [
        "--backend", "pytorch", "--extra_llm_api_options",
        temp_extra_llm_api_options_file
    ]
    with RemoteOpenAIServer(model_path, port=8000,
                            cli_args=args) as remote_server:
        yield remote_server


@pytest.fixture(scope="module")
def example_root():
    llm_root = os.getenv("LLM_ROOT")
    return os.path.join(llm_root, "examples", "serve")


@pytest.mark.parametrize("exe, script",
                         [("python3", "openai_completion_client_for_lora.py")])
def test_trtllm_serve_examples(exe: str, script: str,
                               server: RemoteOpenAIServer, example_root: str,
                               model_name: str, lora_adapter_path: str):
    client_script = os.path.join(example_root, script)
    env = os.environ.copy()
    env["TRTLLM_LORA_MODEL"] = model_name
    env["TRTLLM_LORA_PATH"] = lora_adapter_path
    # CalledProcessError will be raised if any errors occur
    subprocess.run([exe, client_script],
                   stdout=subprocess.PIPE,
                   stderr=subprocess.PIPE,
                   text=True,
                   check=True,
                   env=env)
