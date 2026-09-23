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
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from tensorrt_llm._torch.auto_deploy.llm_args import \
    LlmArgs as AutoDeployLlmArgs
from tensorrt_llm.llmapi import llm as llm_module
from tensorrt_llm.llmapi import mm_encoder as mm_encoder_module
from tensorrt_llm.llmapi.llm import _TorchLLM
from tensorrt_llm.llmapi.llm_args import TorchLlmArgs
from tensorrt_llm.llmapi.llm_utils import CachedModelLoader, ModelLoader
from tensorrt_llm.llmapi.mm_encoder import MultimodalEncoder
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
def test_cached_model_loader_returns_model_dir(tmp_path):
    llm_args = TorchLlmArgs(model=str(tmp_path), gpus_per_node=1)

    model_dir = CachedModelLoader(llm_args)()

    assert model_dir == tmp_path


@pytest.mark.cpu_only
def test_cached_model_loader_returns_none_for_autodeploy(tmp_path):
    llm_args = AutoDeployLlmArgs(model=str(tmp_path))

    model_dir = CachedModelLoader(llm_args)()

    assert model_dir is None


@pytest.mark.cpu_only
def test_torch_llm_build_passes_model_dir_to_executor(monkeypatch, tmp_path):
    llm = object.__new__(_TorchLLM)
    llm.args = TorchLlmArgs(model=str(tmp_path), gpus_per_node=1)
    llm.mpi_session = None
    llm._executor_cls = MagicMock()

    monkeypatch.setattr(CachedModelLoader, "__call__", lambda self: tmp_path)
    monkeypatch.setattr(_TorchLLM, "_try_load_tokenizer", lambda self: None)
    monkeypatch.setattr(_TorchLLM, "_try_load_hf_model_config",
                        lambda self: None)
    monkeypatch.setattr(_TorchLLM,
                        "_reject_token_encoder_config_without_buckets",
                        lambda self: None)
    monkeypatch.setattr(_TorchLLM, "_try_load_generation_config",
                        lambda self: None)
    monkeypatch.setattr(_TorchLLM,
                        "_try_load_generation_config_explicit_values",
                        lambda self: {})
    monkeypatch.setattr(llm_module, "create_input_processor",
                        lambda *args, **kwargs: SimpleNamespace(tokenizer=None))
    monkeypatch.setattr(llm_module, "external_mpi_comm_available",
                        lambda world_size: False)

    llm._build_model()

    create_call = llm._executor_cls.create.call_args
    assert create_call.args == (None, )
    assert create_call.kwargs["hf_model_dir"] == tmp_path


@pytest.mark.cpu_only
def test_multimodal_encoder_build_passes_none_to_executor(
        monkeypatch, tmp_path):
    encoder = object.__new__(MultimodalEncoder)
    encoder.args = TorchLlmArgs(model=str(tmp_path), gpus_per_node=1)
    encoder.mpi_session = None
    encoder._executor_cls = MagicMock()

    monkeypatch.setattr(CachedModelLoader, "__call__", lambda self: tmp_path)
    monkeypatch.setattr(MultimodalEncoder, "_try_load_tokenizer",
                        lambda self: None)
    monkeypatch.setattr(mm_encoder_module, "create_input_processor",
                        lambda *args, **kwargs: SimpleNamespace(tokenizer=None))
    monkeypatch.setattr(mm_encoder_module, "external_mpi_comm_available",
                        lambda world_size: False)

    encoder._build_model()

    assert encoder._executor_cls.create.call_args.args == (None, )


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
