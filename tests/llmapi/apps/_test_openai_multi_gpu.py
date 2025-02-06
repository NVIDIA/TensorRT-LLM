import os
import sys

import openai
import pytest
from openai_server import RemoteOpenAIServer

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from test_llm import get_model_path, prompts

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from utils.util import skip_single_gpu


@pytest.fixture(scope="module")
def model_name():
    return "llama-models-v3/llama-v3-8b-instruct-hf"


@pytest.fixture(scope="module", params=[None, 'pytorch'])
def backend(request):
    return request.param


@pytest.fixture(scope="module")
def server(model_name: str, backend: str):
    model_path = get_model_path(model_name)
    args = ["--tp_size", "2", "--max_beam_width", "1"]
    if backend is not None:
        args.append("--backend")
        args.append(backend)
    with RemoteOpenAIServer(model_path, args) as remote_server:
        yield remote_server


@pytest.fixture(scope="module")
def client(server: RemoteOpenAIServer):
    return server.get_client()


@pytest.fixture(scope="module")
def async_client(server: RemoteOpenAIServer):
    return server.get_async_client()


@skip_single_gpu
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
    assert message.content == 'Two'


@skip_single_gpu
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
        max_tokens=1,
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
    assert "".join(str_chunk) == " D E F G H"
