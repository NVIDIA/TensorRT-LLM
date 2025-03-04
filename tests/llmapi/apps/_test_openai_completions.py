# Adapted from
# https://github.com/vllm-project/vllm/blob/aae6927be06dedbda39c6b0c30f6aa3242b84388/tests/entrypoints/openai/test_completion.py
import os
import sys
from typing import List

import openai
import pytest
from openai_server import RemoteOpenAIServer

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from test_llm import get_model_path


@pytest.fixture(scope="module")
def model_name():
    return "llama-models-v2/TinyLlama-1.1B-Chat-v1.0"


@pytest.fixture(scope="module", params=[None, 'pytorch'])
def backend(request):
    return request.param


@pytest.fixture(scope="module",
                params=[0, 2],
                ids=["disable_processpool", "enable_processpool"])
def num_postprocess_workers(request):
    return request.param


@pytest.fixture(scope="module")
def server(model_name: str, backend: str, num_postprocess_workers: int):
    model_path = get_model_path(model_name)
    if backend == "pytorch":
        args = ["--backend", f"{backend}"]
    else:
        args = ["--max_beam_width", "4"]
    args.extend(["--num_postprocess_workers", f"{num_postprocess_workers}"])
    with RemoteOpenAIServer(model_path, args) as remote_server:
        yield remote_server


@pytest.fixture(scope="module")
def client(server: RemoteOpenAIServer):
    return server.get_client()


@pytest.fixture(scope="module")
def async_client(server: RemoteOpenAIServer):
    return server.get_async_client()


def test_single_completion(client: openai.OpenAI, model_name):
    completion = client.completions.create(
        model=model_name,
        prompt="Hello, my name is",
        max_tokens=5,
        temperature=0.0,
    )

    choice = completion.choices[0]
    assert len(choice.text) >= 5
    assert choice.finish_reason == "length"
    assert completion.id is not None
    assert completion.choices is not None and len(completion.choices) == 1
    completion_tokens = 5
    prompt_tokens = 6
    assert completion.usage == openai.types.CompletionUsage(
        completion_tokens=completion_tokens,
        prompt_tokens=prompt_tokens,
        total_tokens=prompt_tokens + completion_tokens)

    # test using token IDs
    completion = client.completions.create(
        model=model_name,
        prompt=[0, 0, 0, 0, 0],
        max_tokens=5,
        temperature=0.0,
    )
    assert len(completion.choices[0].text) >= 1


@pytest.mark.asyncio(loop_scope="module")
@pytest.mark.parametrize("echo", [True, False])
async def test_completion_streaming(async_client: openai.AsyncOpenAI,
                                    model_name: str, echo: bool):
    # place this function here to avoid OAI completion bug
    prompt = "Hello, my name is"

    single_completion = await async_client.completions.create(
        model=model_name,
        prompt=prompt,
        max_tokens=5,
        temperature=0.0,
        echo=echo,
    )
    single_output = single_completion.choices[0].text
    stream = await async_client.completions.create(
        model=model_name,
        prompt=prompt,
        max_tokens=5,
        temperature=0.0,
        stream=True,
        echo=echo,
    )
    chunks: List[str] = []
    finish_reason_count = 0
    async for chunk in stream:
        chunks.append(chunk.choices[0].text)
        if chunk.choices[0].finish_reason is not None:
            finish_reason_count += 1
    assert finish_reason_count == 1
    assert chunk.choices[0].finish_reason == "length"
    assert chunk.choices[0].text
    assert "".join(chunks) == single_output


@pytest.mark.asyncio(loop_scope="module")
@pytest.mark.parametrize("prompts",
                         [["Hello, my name is"] * 2, [[0, 0, 0, 0, 0]] * 2])
async def test_batch_completions(async_client: openai.AsyncOpenAI, model_name,
                                 prompts):
    # test simple list
    batch = await async_client.completions.create(
        model=model_name,
        prompt=prompts,
        max_tokens=5,
        temperature=0.0,
    )
    assert len(batch.choices) == 2
    assert batch.choices[0].text == batch.choices[1].text


@pytest.mark.asyncio(loop_scope="module")
@pytest.mark.parametrize("prompts",
                         [["Hello, my name is"] * 2, [[0, 0, 0, 0, 0]] * 2])
async def test_batch_completions_beam_search(async_client: openai.AsyncOpenAI,
                                             model_name, prompts, backend):
    # test beam search
    if backend == 'pytorch':
        pytest.skip("Beam search is not supported in PyTorch backend yet")
    batch = await async_client.completions.create(
        model=model_name,
        prompt=prompts,
        n=2,
        max_tokens=5,
        temperature=0.0,
        extra_body=dict(use_beam_search=True),
    )
    assert len(batch.choices) == 4
    assert batch.choices[0].text != batch.choices[
        1].text, "beam search should be different"
    assert batch.choices[0].text == batch.choices[
        2].text, "two copies of the same prompt should be the same"
    assert batch.choices[1].text == batch.choices[
        3].text, "two copies of the same prompt should be the same"


@pytest.mark.asyncio(loop_scope="module")
@pytest.mark.parametrize("prompts",
                         [["Hello, my name is"] * 2, [[0, 0, 0, 0, 0]] * 2])
async def test_batch_completions_streaming(async_client: openai.AsyncOpenAI,
                                           model_name, prompts):
    # test streaming
    batch = await async_client.completions.create(
        model=model_name,
        prompt=prompts,
        max_tokens=5,
        temperature=0.0,
        stream=True,
    )
    texts = [""] * 2
    async for chunk in batch:
        assert len(chunk.choices) == 1
        choice = chunk.choices[0]
        texts[choice.index] += choice.text
    assert texts[0] == texts[1]
