import asyncio
import os
import re
import time

import openai
import pytest
import torch
from utils.util import skip_num_gpus_less_than, skip_nvlink_inactive

from ..test_llm import get_model_path, prompts
from .openai_server import RemoteOpenAIServer

RANK = os.environ.get("SLURM_PROCID", 0)
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
    return "llama-3.1-model/Llama-3.1-8B-Instruct"


@pytest.fixture(scope="module", params=['pytorch'], ids=["pytorch"])
def backend(request):
    return request.param


@pytest.fixture(scope="module",
                params=[(16, 1), (8, 2)],
                ids=lambda tp_pp_size: f'tp{tp_pp_size[0]}pp{tp_pp_size[1]}')
def tp_pp_size(request):
    return request.param


@pytest.fixture(scope="module")
def server(model_name: str, backend: str, tp_pp_size: tuple):
    os.environ["FORCE_DETERMINISTIC"] = "1"
    model_path = get_model_path(model_name)
    tp_size, pp_size = tp_pp_size
    device_count = torch.cuda.device_count()
    args = [
        "--tp_size",
        f"{tp_size}",
        "--pp_size",
        f"{pp_size}",
        "--gpus_per_node",
        f"{device_count}",
        "--kv_cache_free_gpu_memory_fraction",
        "0.95",
        "--backend",
        backend,
    ]
    with RemoteOpenAIServer(model_path, args, llmapi_launch=True,
                            port=8001) as remote_server:
        yield remote_server

    os.environ.pop("FORCE_DETERMINISTIC")


@pytest.fixture(scope="module")
def client(server: RemoteOpenAIServer):
    return server.get_client()


@pytest.fixture(scope="module")
def async_client(server: RemoteOpenAIServer):
    return server.get_async_client()


@skip_num_gpus_less_than(4)
def test_chat(client: openai.OpenAI, model_name: str):
    if RANK == "0":
        messages = [{
            "role": "system",
            "content": "you are a helpful assistant"
        }, {
            "role": "user",
            "content": "What is the result of 1+1? Answer in one word: "
        }]
        chat_completion = client.chat.completions.create(
            model=model_name,
            messages=messages,
            max_tokens=1,
        )
        assert chat_completion.id is not None
        assert len(chat_completion.choices) == 1
        assert chat_completion.usage.completion_tokens == 1
        message = chat_completion.choices[0].message

        print(f"Output: {message.content}")
        assert message.content == 'Two'
    else:
        time.sleep(30)
        assert True


@skip_num_gpus_less_than(4)
def test_completion(client: openai.OpenAI, model_name: str):
    if RANK == "0":
        completion = client.completions.create(
            model=model_name,
            prompt=prompts,
            max_tokens=5,
            temperature=0.0,
        )
        assert completion.choices[0].text == " D E F G H"
    else:
        time.sleep(30)
        assert True


@skip_num_gpus_less_than(4)
@pytest.mark.asyncio(loop_scope="module")
async def test_chat_streaming(async_client: openai.AsyncOpenAI,
                              model_name: str):
    if RANK == "0":
        messages = [{
            "role": "system",
            "content": "you are a helpful assistant"
        }, {
            "role": "user",
            "content": "What is the result of 1+1? Answer in one word: "
        }]
        stream = await async_client.chat.completions.create(
            model=model_name,
            messages=messages,
            max_tokens=1,
            stream=True,
        )
        async for chunk in stream:
            delta = chunk.choices[0].delta
            if delta.role:
                assert delta.role == "assistant"
            if delta.content:
                assert delta.content == "Two"
    else:
        time.sleep(30)
        assert True


@skip_num_gpus_less_than(4)
@pytest.mark.asyncio(loop_scope="module")
async def test_completion_streaming(async_client: openai.AsyncOpenAI,
                                    model_name: str):
    if RANK == "0":
        completion = await async_client.completions.create(
            model=model_name,
            prompt=prompts,
            max_tokens=5,
            temperature=0.0,
            stream=True,
        )
        str_chunk = []
        async for chunk in completion:
            str_chunk.append(chunk.choices[0].text)
        assert "".join(str_chunk) == " D E F G H"
    else:
        time.sleep(30)
        assert True


@skip_nvlink_inactive
@skip_num_gpus_less_than(4)
@pytest.mark.asyncio(loop_scope="module")
@pytest.mark.skip(reason="https://nvbugs/5112075")
async def test_multi_consistent_sync_chat(client: openai.OpenAI,
                                          model_name: str):
    """
        RCCA: https://nvbugs/4829393
    """
    if RANK == 0:
        unique_content = set()

        async def send_request(messages=None):
            try:
                completion = client.chat.completions.create(
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
    else:
        time.sleep(60)
        assert True


@skip_nvlink_inactive
@skip_num_gpus_less_than(4)
@pytest.mark.asyncio(loop_scope="module")
@pytest.mark.skip(reason="https://nvbugs/5112075")
async def test_multi_consistent_async_chat(async_client: openai.AsyncOpenAI,
                                           model_name: str):
    """
        RCCA: https://nvbugs/4829393
    """

    if RANK:
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
    else:
        time.sleep(60)
        assert True
