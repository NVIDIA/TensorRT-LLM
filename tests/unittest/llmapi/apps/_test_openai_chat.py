# Adapted from
# https://github.com/vllm-project/vllm/blob/aae6927be06dedbda39c6b0c30f6aa3242b84388/tests/entrypoints/openai/test_chat.py
import os
import tempfile
from typing import List

import numpy as np
import openai
import pytest
import yaml

from ..test_llm import get_model_path
from .openai_server import RemoteOpenAIServer
from .utils import (invalid_logit_bias_helper, logit_bias_effect_helper,
                    make_server_with_custom_sampler_fixture)

pytestmark = pytest.mark.threadleak(enabled=False)


@pytest.fixture(scope="module", ids=["TinyLlama-1.1B-Chat"])
def model_name():
    return "llama-models-v2/TinyLlama-1.1B-Chat-v1.0"


@pytest.fixture(scope="module", params=["trt", "pytorch"])
def backend(request):
    return request.param


@pytest.fixture(scope="module",
                params=[0, 2],
                ids=["disable_processpool", "enable_processpool"])
def num_postprocess_workers(request):
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
           temp_extra_llm_api_options_file: str, num_postprocess_workers: int):
    model_path = get_model_path(model_name)
    args = ["--backend", f"{backend}"]
    if backend == "trt":
        args.extend(["--max_beam_width", "4"])
    if extra_llm_api_options:
        args.extend(
            ["--extra_llm_api_options", temp_extra_llm_api_options_file])
    args.extend(["--num_postprocess_workers", f"{num_postprocess_workers}"])
    with RemoteOpenAIServer(model_path, args) as remote_server:
        yield remote_server


@pytest.fixture(scope="module")
def client(server: RemoteOpenAIServer):
    return server.get_client()


@pytest.fixture(scope="module")
def async_client(server: RemoteOpenAIServer):
    return server.get_async_client()


def test_single_chat_session(client: openai.OpenAI, model_name: str):
    messages = [{
        "role": "system",
        "content": "you are a helpful assistant"
    }, {
        "role": "user",
        "content": "what is 1+1?"
    }]

    chat_completion = client.chat.completions.create(
        model=model_name,
        messages=messages,
        max_completion_tokens=10,
        temperature=0.0,
        logprobs=False,
    )
    assert chat_completion.id is not None
    assert len(chat_completion.choices) == 1
    message = chat_completion.choices[0].message
    assert message.content is not None
    assert message.role == "assistant"
    # test finish_reason
    finish_reason = chat_completion.choices[0].finish_reason
    completion_tokens = chat_completion.usage.completion_tokens
    if finish_reason == "length":
        assert completion_tokens == 10
    elif finish_reason == "stop":
        assert completion_tokens <= 10
    else:
        raise RuntimeError(
            f"finish_reason {finish_reason} not in [length, stop]")
    # test max_tokens
    legacy = client.chat.completions.create(
        model=model_name,
        messages=messages,
        max_tokens=10,
        temperature=0.0,
        logprobs=False,
    )
    assert legacy.choices[0].message.content \
        == chat_completion.choices[0].message.content
    # test deduced max_tokens
    chat_completion = client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=0.0,
        logprobs=False,
    )
    assert chat_completion.id is not None
    assert len(chat_completion.choices) == 1
    message = chat_completion.choices[0].message
    assert message.content is not None
    assert message.role == "assistant"
    # test logprobs
    chat_completion = client.chat.completions.create(
        model=model_name,
        messages=messages,
        max_completion_tokens=10,
        logprobs=True,
    )
    logprobs = chat_completion.choices[0].logprobs.content
    for logprob in logprobs:
        assert logprob.token is not None
        assert logprob.logprob is not None
        assert logprob.bytes is not None
        assert logprob.top_logprobs is None


def test_multi_turn_dialogue(client: openai.OpenAI, model_name: str):
    # test multi-turn dialogue
    messages = [{
        "role": "system",
        "content": "you are a helpful assistant"
    }, {
        "role": "user",
        "content": "what is 1+1?"
    }]
    messages.append({"role": "assistant", "content": "2"})
    messages.append({"role": "user", "content": "express your result in json"})
    chat_completion = client.chat.completions.create(
        model=model_name,
        messages=messages,
        max_completion_tokens=10,
    )
    message = chat_completion.choices[0].message
    assert message.content is not None and len(message.content) >= 0


def test_multiple_responses(client: openai.OpenAI, model_name: str,
                            backend: str):
    if backend == "pytorch":
        pytest.skip(
            "Multiple responses are not supported in PyTorch backend yet")

    messages = [{
        "role": "system",
        "content": "you are a helpful assistant"
    }, {
        "role": "user",
        "content": "what is 1+1?"
    }]
    # test beam search
    chat_completion = client.chat.completions.create(
        model=model_name,
        messages=messages,
        max_completion_tokens=10,
        n=2,
        temperature=0.0,
        extra_body=dict(use_beam_search=True),
    )
    assert len(chat_completion.choices) == 2
    assert chat_completion.choices[
        0].message.content != chat_completion.choices[
            1].message.content, "beam search should be different"
    # test n and best_of
    chat_completion = client.chat.completions.create(
        model=model_name,
        messages=messages,
        max_completion_tokens=10,
        n=2,
        temperature=0.0,
        extra_body=dict(best_of=4),
    )
    assert len(chat_completion.choices) == 2


@pytest.mark.asyncio(loop_scope="module")
async def test_chat_streaming(async_client: openai.AsyncOpenAI,
                              model_name: str):
    messages = [{
        "role": "system",
        "content": "you are a helpful assistant"
    }, {
        "role": "user",
        "content": "what is 1+1?"
    }]

    chat_completion = await async_client.chat.completions.create(
        model=model_name,
        messages=messages,
        max_completion_tokens=10,
        temperature=0.0,
        logprobs=True,
    )

    output = chat_completion.choices[0].message.content
    logprobs = [
        logprob_content.logprob
        for logprob_content in chat_completion.choices[0].logprobs.content
    ]
    _finish_reason = chat_completion.choices[0].finish_reason

    # test streaming
    stream = await async_client.chat.completions.create(
        model=model_name,
        messages=messages,
        max_completion_tokens=10,
        temperature=0.0,
        logprobs=True,
        stream=True,
    )
    str_chunks: List[str] = []
    logprob_chunks: List[float] = []

    finish_reason_counter = 0
    finish_reason: str = None
    async for chunk in stream:
        choice = chunk.choices[0]
        delta = choice.delta
        if logprob_chunk := choice.logprobs:
            if len(logprob_chunk.content) == 1:
                assert logprob_chunk.content[0].top_logprobs is None
                logprob_chunks.append(logprob_chunk.content[0].logprob)
            elif len(logprob_chunk.content) == 0:
                assert delta.content == ""
            else:
                raise RuntimeError("logprobs streaming error")
        if choice.finish_reason is not None:
            finish_reason_counter += 1
            finish_reason = choice.finish_reason
        if delta.role:
            assert delta.role == "assistant"
        if delta.content:
            str_chunks.append(delta.content)
    # test finish_reason
    if delta.content == "":
        assert finish_reason == "stop"
    assert finish_reason_counter == 1
    assert finish_reason == _finish_reason
    num_tokens = len(str_chunks)
    if finish_reason == "length":
        assert num_tokens == 10
    elif finish_reason == "stop":
        assert num_tokens <= 10
    else:
        raise RuntimeError(
            f"finish_reason {finish_reason} not in [length, stop]")
    # test generated tokens
    assert "".join(str_chunks) == output
    # test logprobs
    assert len(logprob_chunks) == len(logprobs)
    logprobs, logprob_chunks = np.array(logprobs), np.array(logprob_chunks)
    assert np.allclose(logprobs, logprob_chunks)


@pytest.mark.asyncio(loop_scope="module")
async def test_chat_completion_stream_options(async_client: openai.AsyncOpenAI,
                                              model_name: str):
    messages = [{
        "role": "system",
        "content": "You are a helpful assistant."
    }, {
        "role": "user",
        "content": "What is the capital of France?"
    }]

    # Test stream=True, stream_options={"include_usage": False}
    stream = await async_client.chat.completions.create(
        model=model_name,
        messages=messages,
        max_completion_tokens=10,
        temperature=0.0,
        stream=True,
        stream_options={"include_usage": False})
    async for chunk in stream:
        assert chunk.usage is None

    # Test stream=True, stream_options={"include_usage": True,
    #                                   "continuous_usage_stats": False}}
    stream = await async_client.chat.completions.create(
        model=model_name,
        messages=messages,
        max_completion_tokens=10,
        temperature=0.0,
        stream=True,
        stream_options={
            "include_usage": True,
            "continuous_usage_stats": False
        })

    async for chunk in stream:
        if chunk.choices:
            assert chunk.usage is None
        else:
            assert chunk.usage is not None
            assert chunk.usage.prompt_tokens > 0
            assert chunk.usage.completion_tokens > 0
            assert chunk.usage.total_tokens == (chunk.usage.prompt_tokens +
                                                chunk.usage.completion_tokens)
            assert chunk.choices == []

    # Test stream=False, stream_options={"include_usage": None}
    with pytest.raises(openai.BadRequestError):
        await async_client.chat.completions.create(
            model=model_name,
            messages=messages,
            max_completion_tokens=10,
            temperature=0.0,
            stream=False,
            stream_options={"include_usage": None})

    # Test stream=False, stream_options={"include_usage": True}
    with pytest.raises(openai.BadRequestError):
        await async_client.chat.completions.create(
            model=model_name,
            messages=messages,
            max_completion_tokens=10,
            temperature=0.0,
            stream=False,
            stream_options={"include_usage": True})

    # Test stream=True, stream_options={"include_usage": True,
    #                           "continuous_usage_stats": True}
    stream = await async_client.chat.completions.create(
        model=model_name,
        messages=messages,
        max_completion_tokens=10,
        temperature=0.0,
        stream=True,
        stream_options={
            "include_usage": True,
            "continuous_usage_stats": True
        },
    )
    async for chunk in stream:
        assert chunk.usage.prompt_tokens >= 0
        assert chunk.usage.completion_tokens >= 0
        assert chunk.usage.total_tokens == (chunk.usage.prompt_tokens +
                                            chunk.usage.completion_tokens)


def test_custom_role(client: openai.OpenAI, model_name: str):
    # Not sure how the model handles custom roles so we just check that
    # both string and complex message content are handled in the same way

    resp1 = client.chat.completions.create(
        model=model_name,
        messages=[{
            "role": "my-custom-role",
            "content": "what is 1+1?",
        }],  # type: ignore
        temperature=0.0,
        max_completion_tokens=16,
        seed=0)

    resp2 = client.chat.completions.create(
        model=model_name,
        messages=[{
            "role": "my-custom-role",
            "content": [{
                "type": "text",
                "text": "what is 1+1?"
            }]
        }],  # type: ignore
        temperature=0.0,
        max_completion_tokens=16,
        seed=0)

    content1 = resp1.choices[0].message.content
    content2 = resp2.choices[0].message.content
    assert content1 == content2


def test_stop_reason(client: openai.OpenAI, model_name: str, backend: str):
    if backend == "pytorch":
        pytest.skip("Stop reason is not supported in PyTorch backend yet")

    messages = [{
        "role": "system",
        "content": "you are a helpful assistant"
    }, {
        "role": "user",
        "content": "what is the result of one plus one?"
    }]

    resp = client.chat.completions.create(
        model=model_name,
        messages=messages,
        max_completion_tokens=10,
        temperature=0.0,
        stop="two",
    )
    assert resp.choices[0].finish_reason == "stop"
    assert resp.choices[0].stop_reason == "two"


server_with_custom_sampler = make_server_with_custom_sampler_fixture('chat')


@pytest.mark.asyncio(loop_scope='function')
@pytest.mark.parametrize(
    'server_with_custom_sampler',
    [
        {
            'sampler_type': "TorchSampler"
        },  # torch_sampler
        {
            'sampler_type': "TRTLLMSampler"
        },  # trtllm_sampler
    ],
    indirect=True,
    ids=['torch_sampler', 'trtllm_sampler'])
async def test_chat_completion_with_logit_bias_effect(
        server_with_custom_sampler, model_name: str) -> None:
    '''Test that logit bias affects output as expected for both samplers (chat endpoint).'''
    client = server_with_custom_sampler.get_async_client()
    await logit_bias_effect_helper(client, model_name, 'chat')


@pytest.mark.asyncio(loop_scope="module")
async def test_chat_completion_with_invalid_logit_bias(
        async_client: openai.AsyncOpenAI, model_name: str):
    """Test with invalid token IDs (non-integer keys) for chat completions"""
    await invalid_logit_bias_helper(async_client, model_name, 'chat')


def test_chat_cached_tokens(client: openai.OpenAI, model_name: str,
                            backend: str, extra_llm_api_options: bool):
    if backend == "trt":
        pytest.skip("Cached tokens is not supported in trt backend yet")

    messages = [{
        "role": "system",
        "content": "A system message"
    }, {
        "role": "user",
        "content": "Some user message"
    }]

    chat_completion = client.chat.completions.create(
        model=model_name,
        messages=messages,
        max_completion_tokens=10,
        temperature=0.0,
        logprobs=False,
    )
    expected_cached_tokens = chat_completion.usage.prompt_tokens - 1

    # We disable kv cache reuse when using extra_llm_api_options,
    # in that case, we expect cached tokens to be 0
    if extra_llm_api_options:
        expected_cached_tokens = 0

    chat_completion = client.chat.completions.create(
        model=model_name,
        messages=messages,
        max_completion_tokens=10,
        temperature=0.0,
        logprobs=False,
    )
    assert chat_completion.usage is not None
    assert chat_completion.usage.prompt_tokens_details is not None
    assert chat_completion.usage.prompt_tokens_details.cached_tokens == expected_cached_tokens


@pytest.mark.asyncio(loop_scope="module")
async def test_chat_cached_tokens_stream(async_client: openai.AsyncOpenAI,
                                         model_name: str, backend: str,
                                         extra_llm_api_options: bool):
    if backend == "trt":
        pytest.skip("Cached tokens is not supported in trt backend yet")

    messages = [{
        "role": "system",
        "content": "A system message"
    }, {
        "role": "user",
        "content": "Some user message"
    }]

    # Run the chat completion for the first time so that cached tokens are created
    chat_completion = await async_client.chat.completions.create(
        model=model_name,
        messages=messages,
        max_completion_tokens=10,
        temperature=0.0,
        logprobs=False,
    )
    expected_cached_tokens = chat_completion.usage.prompt_tokens - 1

    # We disable kv cache reuse when using extra_llm_api_options,
    # in that case, we expect cached tokens to be 0
    if extra_llm_api_options:
        expected_cached_tokens = 0

    # Test stream=True, stream_options={"include_usage": True,
    #                                   "continuous_usage_stats": False}}
    stream = await async_client.chat.completions.create(
        model=model_name,
        messages=messages,
        max_completion_tokens=10,
        temperature=0.0,
        stream=True,
        stream_options={
            "include_usage": True,
            "continuous_usage_stats": False
        })

    async for chunk in stream:
        if chunk.choices:
            assert chunk.usage is None
        else:
            assert chunk.usage is not None
            assert chunk.usage.prompt_tokens > 0
            assert chunk.usage.completion_tokens > 0
            assert chunk.usage.total_tokens == (chunk.usage.prompt_tokens +
                                                chunk.usage.completion_tokens)
            assert chunk.usage.prompt_tokens_details is not None
            assert chunk.usage.prompt_tokens_details.cached_tokens == expected_cached_tokens
            assert chunk.choices == []

    # Test stream=True, stream_options={"include_usage": True,
    #                           "continuous_usage_stats": True}
    stream = await async_client.chat.completions.create(
        model=model_name,
        messages=messages,
        max_completion_tokens=10,
        temperature=0.0,
        stream=True,
        stream_options={
            "include_usage": True,
            "continuous_usage_stats": True
        },
    )
    async for chunk in stream:
        assert chunk.usage.prompt_tokens >= 0
        assert chunk.usage.completion_tokens >= 0
        assert chunk.usage.total_tokens == (chunk.usage.prompt_tokens +
                                            chunk.usage.completion_tokens)
        assert chunk.usage.prompt_tokens_details is not None
        assert chunk.usage.prompt_tokens_details.cached_tokens == expected_cached_tokens
