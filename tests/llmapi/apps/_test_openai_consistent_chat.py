"""Test openai api with multiple clients"""
import asyncio
import os
import re
import sys
from tempfile import TemporaryDirectory

import openai
import pytest
from openai_server import RemoteOpenAIServer

from tensorrt_llm.llmapi import BuildConfig
from tensorrt_llm.llmapi.llm import LLM

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from test_llm import get_model_path

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from utils.util import (skip_gpu_memory_less_than_40gb, skip_num_gpus_less_than,
                        skip_nvlink_inactive)

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
def model_name():
    return "llama-3.1-model/Meta-Llama-3.1-70B-Instruct"


@pytest.fixture(scope="module")
def build_engine(model_name):
    """
        build engine
        trtllm-build --checkpoint_dir ./output
            --output_dir ./engine
            --max_seq_len 64000
            --max_batch_size 64
            --gemm_plugin disable
            --workers 4
    """
    skip_num_gpus_less_than(4)
    tp_size = 4
    model_path = get_model_path(model_name)
    build_config = BuildConfig()
    build_config.max_batch_size = 64
    build_config.max_seq_len = 64000
    build_config.plugin_config._gemm_plugin = None

    llm = LLM(model_path,
              tensor_parallel_size=tp_size,
              auto_parallel_world_size=tp_size,
              build_config=build_config)

    engine_dir = TemporaryDirectory(suffix="-engine_dir")
    llm.save(engine_dir.name)
    del llm

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
            completion = client.chat.completions.create(model=model_name,
                                                        messages=messages,
                                                        n=1,
                                                        max_tokens=1024,
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
                max_tokens=1024,
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
