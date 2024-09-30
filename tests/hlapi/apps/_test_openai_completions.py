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
    return "llama-models-v3/llama-v3-8b-instruct-hf"


@pytest.fixture(scope="module")
def server(model_name: str):
    model_path = get_model_path(model_name)
    args = ["--max_beam_width", "4"]
    with RemoteOpenAIServer(model_path, args) as remote_server:
        yield remote_server


@pytest.fixture(scope="module")
def client(server: RemoteOpenAIServer):
    return server.get_client()


@pytest.fixture(scope="module")
def async_client(server: RemoteOpenAIServer):
    return server.get_async_client()


@pytest.mark.asyncio
@pytest.mark.parametrize("echo", [True, False])
async def test_completion_streaming(async_client: openai.AsyncOpenAI,
                                    model_name: str, echo: bool):
    prompt = "What is an LLM?"

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
    async for chunk in stream:
        chunks.append(chunk.choices[0].text)
    assert chunk.choices[0].text
    assert "".join(chunks) == single_output


def test_single_completion(client: openai.OpenAI, model_name):
    completion = client.completions.create(
        model=model_name,
        prompt="Hello, my name is",
        max_tokens=5,
        temperature=0.0,
    )

    choice = completion.choices[0]
    assert len(choice.text) >= 5
    assert completion.id is not None
    assert completion.choices is not None and len(completion.choices) == 1

    assert completion.usage == openai.types.CompletionUsage(completion_tokens=5,
                                                            prompt_tokens=6,
                                                            total_tokens=11)

    # test using token IDs
    completion = client.completions.create(
        model="llama-v3-8b-instruct-hf",
        prompt=[0, 0, 0, 0, 0],
        max_tokens=5,
        temperature=0.0,
    )
    assert len(completion.choices[0].text) >= 1


@pytest.mark.asyncio
async def test_batch_completions(async_client: openai.AsyncOpenAI, model_name):
    # test both text and token IDs
    for prompts in (["Hello, my name is"] * 2, [[0, 0, 0, 0, 0]] * 2):
        # test simple list
        batch = await async_client.completions.create(
            model=model_name,
            prompt=prompts,
            max_tokens=5,
            temperature=0.0,
        )
        assert len(batch.choices) == 2
        assert batch.choices[0].text == batch.choices[1].text

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
