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
import asyncio
import json
import threading
import time

import pytest
import torch

from tensorrt_llm.llmapi.llm_args import TorchLlmArgs
from tensorrt_llm.llmapi.llm_utils import ModelLoader
from tensorrt_llm.llmapi.utils import AsyncQueue

# isort: off
from .test_llm import llama_model_path
# isort: on


@pytest.mark.cpu_only
def test_load_hf_generation_config_dict_preserves_explicit_values(tmp_path):
    expected = {
        "temperature": 1.0,
        "top_p": 0.9,
    }
    (tmp_path / "generation_config.json").write_text(json.dumps(expected),
                                                     encoding="utf-8")

    assert ModelLoader.load_hf_generation_config_dict(tmp_path) == expected


@pytest.mark.cpu_only
def test_load_hf_generation_config_dict_returns_empty_without_file(tmp_path):
    assert ModelLoader.load_hf_generation_config_dict(tmp_path) == {}


@pytest.mark.cpu_only
def test_load_hf_generation_config_dict_returns_empty_for_malformed_json(
        tmp_path):
    (tmp_path / "generation_config.json").write_text("{", encoding="utf-8")

    assert ModelLoader.load_hf_generation_config_dict(tmp_path) == {}


@pytest.mark.cpu_only
def test_load_hf_generation_config_dict_returns_empty_for_json_array(tmp_path):
    (tmp_path / "generation_config.json").write_text("[1, 2]", encoding="utf-8")

    assert ModelLoader.load_hf_generation_config_dict(tmp_path) == {}


@pytest.mark.cpu_only
def test_LlmArgs_default_gpus_per_node():
    # default
    llm_args = TorchLlmArgs(model=llama_model_path)
    assert llm_args.gpus_per_node == torch.cuda.device_count()

    # set explicitly
    llm_args = TorchLlmArgs(model=llama_model_path, gpus_per_node=6)
    assert llm_args.gpus_per_node == 6


@pytest.mark.cpu_only
def test_AsyncQueue():
    queue = AsyncQueue()

    # put data to queue sync in a thread
    # async get data from queue in the current event loop
    # NOTE: the event loop in the two threads are different

    def put_data_to_queue():
        for i in range(10):
            time.sleep(0.1)
            queue.put(i)

    async def get_data_from_queue():
        for i in range(10):
            print(f"get: {queue.get()}")

    thread = threading.Thread(target=put_data_to_queue)
    thread.start()
    asyncio.run(get_data_from_queue())
    thread.join()
