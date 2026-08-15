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
import os
import sys

import pytest
from llmapi.apps.openai_server import RemoteOpenAIServer

from tensorrt_llm.scaffolding import (ChatTask, GenerationTask, TaskStatus,
                                      TRTLLMWorker, TRTOpenaiWorker,
                                      UserMessage)

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from llmapi.test_llm import get_model_path
from utils.llm_data import llm_models_root


@pytest.fixture(scope="module")
def trtllm_model_path():
    return llm_models_root() / "Qwen3/Qwen3-0.6B"


@pytest.fixture(scope="module")
def default_prompt():
    prompt = "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?\r\n\r\n"
    return prompt


@pytest.fixture(scope="module")
def model_name():
    return "gpt_oss/gpt-oss-20b"


@pytest.fixture(scope="module", params=['pytorch'])
def backend(request):
    return request.param


@pytest.fixture(scope="module", params=[2], ids=["enable_processpool"])
def num_postprocess_workers(request):
    return request.param


@pytest.fixture(scope="module")
def server(model_name: str, backend: str, num_postprocess_workers: int):
    model_path = get_model_path(model_name)

    args = ["--backend", f"{backend}"]
    args.extend(["--num_postprocess_workers", f"{num_postprocess_workers}"])
    args.extend(["--kv_cache_free_gpu_memory_fraction", "0.5"])
    remote_server = RemoteOpenAIServer(model_path, args)
    yield remote_server
    remote_server.terminate()


def create_trtllm_worker(model_path):
    return TRTLLMWorker.init_with_new_llm(str(model_path), backend="pytorch")


def create_trtoai_worker(model_name, async_client):
    return TRTOpenaiWorker(
        async_client=async_client,
        model=model_name,
    )


def test_trtllm_worker_generation(default_prompt, trtllm_model_path):
    worker = create_trtllm_worker(trtllm_model_path)
    try:
        task = GenerationTask.create_from_prompt(default_prompt)
        task.max_tokens = 100
        status = asyncio.run(worker.run_task(task))
        assert status == TaskStatus.SUCCESS, "Generation Task is not successful with TRTLLMWorker"
    finally:
        worker.shutdown()


@pytest.mark.asyncio(loop_scope="module")
def test_trtoai_worker_generation(default_prompt, model_name, server):
    worker = create_trtoai_worker(model_name, server.get_async_client())
    try:
        task = GenerationTask.create_from_prompt(default_prompt)
        task.max_tokens = 100
        status = asyncio.run(worker.run_task(task))
        assert status == TaskStatus.SUCCESS, "Generation Task is not successful with TRTOpenaiWorker"
    finally:
        worker.shutdown()


@pytest.mark.asyncio(loop_scope="module")
def test_trtoai_worker_chat(default_prompt, model_name, server):
    worker = create_trtoai_worker(model_name, server.get_async_client())
    try:
        task = ChatTask.create_from_messages([UserMessage(default_prompt)])
        task.max_tokens = 100
        status = asyncio.run(worker.run_task(task))
        assert status == TaskStatus.SUCCESS, "Chat Task is not successful with TRTOpenaiWorker"
    finally:
        worker.shutdown()
