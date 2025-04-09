from pathlib import Path

# isort: off
from utils.llm_data import llm_models_root
# isort: on

import asyncio
import os
import sys

import pytest
from llmapi.apps.openai_server import RemoteOpenAIServer

from tensorrt_llm.scaffolding import (GenerationTask, TaskStatus, TRTLLMWorker,
                                      TRTOpenaiWorker)

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from llmapi.test_llm import get_model_path


@pytest.fixture(scope="module")
def deepseek_distill_7b_path() -> Path:
    model_dir = llm_models_root() / "DeepSeek-R1/DeepSeek-R1-Distill-Qwen-7B"
    return model_dir


@pytest.fixture(scope="module")
def default_prompt():
    prompt = "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?\r\n\r\n"
    return prompt


@pytest.fixture(scope="module")
def model_name():
    return "DeepSeek-R1/DeepSeek-R1-Distill-Qwen-7B"


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
    with RemoteOpenAIServer(model_path, args) as remote_server:
        yield remote_server


@pytest.fixture(scope="module")
def async_client(server: RemoteOpenAIServer):
    return server.get_async_client()


def create_trtoai_worker(model_name, async_client):
    return TRTOpenaiWorker(
        async_client=async_client,
        model=model_name,
        max_tokens=5,
    )


@pytest.mark.asyncio(loop_scope="module")
async def test_trtoai_worker_generation(default_prompt, model_name,
                                        async_client):
    worker = create_trtoai_worker(model_name, async_client)
    task = GenerationTask.create_from_prompt(default_prompt)
    status = await worker.run_task(task)
    assert status == TaskStatus.SUCCESS, "Generation Task is not successful with TRTOpenaiWorker"


def create_trtllm_worker(model_path):
    return TRTLLMWorker.init_with_new_llm(str(model_path),
                                          backend="pytorch",
                                          max_batch_size=1,
                                          max_num_tokens=5,
                                          temperature=0.9)


def test_trtllm_worker_generation(default_prompt, deepseek_distill_7b_path):
    worker = create_trtllm_worker(deepseek_distill_7b_path)
    task = GenerationTask.create_from_prompt(default_prompt)
    status = asyncio.run(worker.run_task(task))
    assert status == TaskStatus.SUCCESS, "Generation Task is not successful with TRTLLMWorker"
