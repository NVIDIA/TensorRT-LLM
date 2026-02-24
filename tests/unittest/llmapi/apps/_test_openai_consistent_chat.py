"""Test openai api with multiple clients"""
import asyncio
import os
import re
from tempfile import TemporaryDirectory

import openai
import pytest
from utils.util import (skip_gpu_memory_less_than_40gb, skip_num_gpus_less_than,
                        skip_nvlink_inactive)

from tensorrt_llm import LLM
from tensorrt_llm.llmapi import BuildConfig

from ..test_llm import get_model_path
from .openai_server import RemoteOpenAIServer

MESSAGES = [{
    "role": "user",
    "content": "Hello! How are you?"
}, {
    "role": "assistant",
    "content": "Hi! I am quite well, how can I help you today?"
}, {
    "role": "user",
    "content": "A song on old age?"
}]


@pytest.fixture(scope="module")
def model_name(request):
    """Parametrize model name"""
    return request.param


@pytest.fixture(scope="module")
def build_engine(model_name, request):
    """Build engine with specified quantization"""
    skip_num_gpus_less_than(4)
    tp_size = 4
    model_path = get_model_path(model_name)

    build_config = BuildConfig()
    build_config.max_batch_size = 64
    build_config.max_seq_len = 64000
    build_config.plugin_config._gemm_plugin = None

    if hasattr(request, 'param'):
        if request.param == "fp16":
            build_config.dtype = 'float16'
        elif request.param == "bf16":
            build_config.dtype = 'bfloat16'

    llm = LLM(model_path,
              tensor_parallel_size=tp_size,
              build_config=build_config)

    engine_dir = TemporaryDirectory(suffix="-engine_dir")
    with llm:
        llm.save(engine_dir.name)

    yield engine_dir.name


@pytest.fixture(scope="module")
def server(model_name: str, build_engine: str):
    os.environ["FORCE_DETERMINISTIC"] = "1"
    model_path = get_model_path(model_name)
    args = ["--tp_size", "4", "--tokenizer", model_path]
    with RemoteOpenAIServer(build_engine, args) as remote_server:
        yield remote_server

    os.environ.pop("FORCE_DETERMINISTIC")


@pytest.fixture(scope="module")
def client(server: RemoteOpenAIServer):
    return server.get_client()


@pytest.fixture(scope="module")
def async_client(server: RemoteOpenAIServer):
    return server.get_async_client()


@pytest.mark.parametrize("model_name", [
    "llama-3.1-model/Meta-Llama-3.1-70B-Instruct", "Mixtral-8x7B-Instruct-v0.1"
],
                         indirect=True)
@skip_nvlink_inactive
@skip_num_gpus_less_than(4)
@skip_gpu_memory_less_than_40gb
@pytest.mark.asyncio(loop_scope="module")
async def test_multi_consistent_sync_chat(client: openai.OpenAI,
                                          model_name: str):
    """
        RCCA: https://nvbugs/4829393
    """

    unique_content = set()

    async def send_request(messages=None):
        try:
            completion = client.chat.completions.create(
                model=model_name,
                messages=messages,
                n=1,
                max_completion_tokens=1024,
                temperature=0,
                frequency_penalty=1.0,
                stream=False,
                stop=["hello"])
            unique_content.add(completion.choices[0].message.content)
        except Exception as e:
            print(f"Error: {e}")

    tasks = []
    for _ in range(50):
        tasks.append(asyncio.create_task(send_request(MESSAGES)))
        await asyncio.sleep(1)

    await asyncio.gather(*tasks)

    print(f"Number of unique responses: {len(unique_content)}")
    assert len(unique_content) == 1, "Responses are not consistent"
    content = list(unique_content)[0]
    pattern = re.compile(r'[^a-zA-Z0-9\s\'\"]{5,}')
    assert not bool(pattern.search(content)), content


@pytest.mark.parametrize("model_name", [
    "llama-3.1-model/Meta-Llama-3.1-70B-Instruct", "Mixtral-8x7B-Instruct-v0.1"
],
                         indirect=True)
@skip_nvlink_inactive
@skip_num_gpus_less_than(4)
@skip_gpu_memory_less_than_40gb
@pytest.mark.asyncio(loop_scope="module")
async def test_multi_consistent_async_chat(async_client: openai.AsyncOpenAI,
                                           model_name: str):
    """
        RCCA: https://nvbugs/4829393
    """

    unique_content = set()

    async def send_request(messages=None):
        try:
            completion = await async_client.chat.completions.create(
                model=model_name,
                messages=messages,
                n=1,
                max_completion_tokens=1024,
                temperature=0,
                frequency_penalty=1.0,
                stream=False,
                stop=["hello"])
            unique_content.add(completion.choices[0].message.content)
        except Exception as e:
            print(f"Error: {e}")

    tasks = []
    for _ in range(50):
        tasks.append(asyncio.create_task(send_request(MESSAGES)))
        await asyncio.sleep(1)

    await asyncio.gather(*tasks)

    print(f"Number of unique responses: {len(unique_content)}")
    assert len(unique_content) == 1, "Responses are not consistent"
    content = list(unique_content)[0]
    pattern = re.compile(r'[^a-zA-Z0-9\s\'\"]{5,}')
    assert not bool(pattern.search(content)), content


@pytest.mark.parametrize("model_name", ["Mixtral-8x7B-Instruct-v0.1"],
                         indirect=True)
@pytest.mark.parametrize("build_engine", ["fp16", "bf16"], indirect=True)
@skip_nvlink_inactive
@skip_num_gpus_less_than(4)
@skip_gpu_memory_less_than_40gb
@pytest.mark.asyncio(loop_scope="module")
async def test_quant_consistency(async_client: openai.AsyncOpenAI,
                                 model_name: str, build_engine: str):
    """
        Test consistency ratio between FP16 and BF16 quantization
        RCCA: https://nvbugspro.nvidia.com/bug/4964501
    """

    async def measure_consistency(num_requests=50):
        unique_content = set()

        async def send_request(messages=None):
            try:
                completion = await async_client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    n=1,
                    max_completion_tokens=1024,
                    temperature=0,
                    frequency_penalty=1.0,
                    stream=False,
                    stop=["hello"])
                unique_content.add(completion.choices[0].message.content)
            except Exception as e:
                print(f"Error: {e}")

        tasks = []
        for _ in range(num_requests):
            tasks.append(asyncio.create_task(send_request(MESSAGES)))
            await asyncio.sleep(1)

        await asyncio.gather(*tasks)

        consistency_ratio = len(unique_content) / num_requests
        for content in unique_content:
            pattern = re.compile(r'[^\w\s\'",.!?*\-(){}\[\]]{5,}')
            assert not bool(pattern.search(content)), content

        return consistency_ratio

    if build_engine == "fp16":
        fp16_ratio = await measure_consistency()
        print(f"FP16 consistency ratio: {fp16_ratio:.4f}")
        test_quant_consistency.fp16_result = fp16_ratio
    else:  # bf16
        bf16_ratio = await measure_consistency()
        print(f"BF16 consistency ratio: {bf16_ratio:.4f}")

        fp16_ratio = getattr(test_quant_consistency, 'fp16_result', None)
        if fp16_ratio is not None:
            relative_diff = abs(fp16_ratio - bf16_ratio) / fp16_ratio * 100
            print(
                f"Consistency ratio relative difference: {relative_diff:.2f}%")

            assert relative_diff <= 5.0, (
                f"Consistency ratio difference between FP16 ({fp16_ratio:.4f}) and "
                f"BF16 ({bf16_ratio:.4f}) exceeds 5% threshold: {relative_diff:.2f}%"
            )
