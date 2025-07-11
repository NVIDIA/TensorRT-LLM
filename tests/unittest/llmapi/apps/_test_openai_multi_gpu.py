import os
import tempfile

import openai
import pytest
import yaml
from utils.util import skip_single_gpu

from ..test_llm import get_model_path, prompts
from .openai_server import RemoteOpenAIServer


@pytest.fixture(scope="module")
def model_name():
    return "llama-models-v3/llama-v3-8b-instruct-hf"


@pytest.fixture(scope="module", params=["trt", "pytorch"])
def backend(request):
    return request.param


@pytest.fixture(scope="module",
                params=[True, False],
                ids=["extra_options", "no_extra_options"])
def extra_llm_api_options(request):
    return request.param


@pytest.fixture(scope="module")
def temp_extra_llm_api_options_file(request):
    temp_dir = tempfile.gettempdir()
    temp_file_path = os.path.join(temp_dir, "extra_llm_api_options.yaml")
    try:
        extra_llm_api_options_dict = {
            "enable_chunked_prefill": False,
            "kv_cache_config": {
                "enable_block_reuse": False,
                "max_tokens": 40000
            }
        }

        with open(temp_file_path, 'w') as f:
            yaml.dump(extra_llm_api_options_dict, f)

        yield temp_file_path
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)


@pytest.fixture(scope="module")
def server(model_name: str, backend: str, extra_llm_api_options: bool,
           temp_extra_llm_api_options_file: str):
    model_path = get_model_path(model_name)
    args = ["--tp_size", "2", "--max_beam_width", "1", "--backend", backend]
    if extra_llm_api_options:
        args.extend(
            ["--extra_llm_api_options", temp_extra_llm_api_options_file])
    with RemoteOpenAIServer(model_path, args) as remote_server:
        yield remote_server


@pytest.fixture(scope="module")
def client(server: RemoteOpenAIServer):
    return server.get_client()


@pytest.fixture(scope="module")
def async_client(server: RemoteOpenAIServer):
    return server.get_async_client()


@skip_single_gpu
@pytest.mark.part0
def test_chat_tp2(client: openai.OpenAI, model_name: str):
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
    assert message.content == "Two"


@skip_single_gpu
@pytest.mark.part0
def test_completion_tp2(client: openai.OpenAI, model_name: str):
    completion = client.completions.create(
        model=model_name,
        prompt=prompts,
        max_tokens=5,
        temperature=0.0,
    )
    assert completion.choices[0].text == " D E F G H"


@skip_single_gpu
@pytest.mark.asyncio(loop_scope="module")
@pytest.mark.part0
async def test_chat_streaming_tp2(async_client: openai.AsyncOpenAI,
                                  model_name: str):
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
        max_completion_tokens=1,
        stream=True,
    )
    async for chunk in stream:
        delta = chunk.choices[0].delta
        if delta.role:
            assert delta.role == "assistant"
        if delta.content:
            assert delta.content == "Two"


@skip_single_gpu
@pytest.mark.asyncio(loop_scope="module")
@pytest.mark.part0
async def test_completion_streaming_tp2(async_client: openai.AsyncOpenAI,
                                        model_name: str):
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
    pytest.skip("https://nvbugspro.nvidia.com/bug/5163855")
    assert "".join(str_chunk) == " D E F G H"
