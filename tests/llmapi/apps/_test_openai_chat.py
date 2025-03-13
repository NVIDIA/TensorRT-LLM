# Adapted from
# https://github.com/vllm-project/vllm/blob/aae6927be06dedbda39c6b0c30f6aa3242b84388/tests/entrypoints/openai/test_chat.py
import os
import sys
import tempfile
from typing import List

import numpy as np
import openai
import pytest
import yaml
from openai_server import RemoteOpenAIServer

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from test_llm import get_model_path


@pytest.fixture(scope="module", ids=["TinyLlama-1.1B-Chat"])
def model_name():
    return "llama-models-v2/TinyLlama-1.1B-Chat-v1.0"


@pytest.fixture(scope="module",
                params=[None, 'pytorch'],
                ids=["trt", "pytorch"])
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
    if backend == "pytorch":
        args = ["--backend", f"{backend}"]
    else:
        args = ["--max_beam_width", "4"]
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
        max_tokens=10,
        logprobs=False,
    )
    assert chat_completion.id is not None
    assert len(chat_completion.choices) == 1
    message = chat_completion.choices[0].message
    assert message.content is not None
    assert message.role == "assistant"
    # test finish_reason
    completion_tokens = chat_completion.usage.completion_tokens
    if chat_completion.choices[0].finish_reason == "length":
        assert completion_tokens == 10
    elif chat_completion.choices[0].finish_reason == "stop":
        assert completion_tokens <= 10
    else:
        raise RuntimeError("finish_reason not in [length, stop]")


def test_single_chat_session_with_logprobs(client: openai.OpenAI,
                                           model_name: str, backend: str):
    if backend == "pytorch":
        pytest.skip("Logprobs are not supported in PyTorch backend yet")

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
        max_tokens=10,
        logprobs=True,
    )
    assert chat_completion.id is not None
    assert len(chat_completion.choices) == 1
    message = chat_completion.choices[0].message
    assert message.content is not None
    assert message.role == "assistant"
    # test logprobs
    logprobs = chat_completion.choices[0].logprobs.content
    if chat_completion.choices[0].finish_reason == "length":
        assert len(logprobs) == 10
    elif chat_completion.choices[0].finish_reason == "stop":
        assert len(logprobs) <= 10
    else:
        raise RuntimeError("finish_reason not in [length, stop]")
    for logprob in logprobs:
        assert logprob.token is not None
        assert logprob.logprob is not None
        assert logprob.bytes is not None
        assert len(logprob.top_logprobs) == 0


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
        max_tokens=10,
    )
    message = chat_completion.choices[0].message
    assert message.content is not None and len(message.content) >= 0


def test_beam_search(client: openai.OpenAI, model_name: str, backend: str):
    if backend == "pytorch":
        pytest.skip("Beam search is not supported in PyTorch backend yet")

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
        max_tokens=10,
        n=2,
        temperature=0.0,
        extra_body=dict(use_beam_search=True),
    )
    assert len(chat_completion.choices) == 2
    assert chat_completion.choices[
        0].message.content != chat_completion.choices[
            1].message.content, "beam search should be different"


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
        max_tokens=10,
        temperature=0.0,
        logprobs=False,
    )
    output = chat_completion.choices[0].message.content
    _finish_reason = chat_completion.choices[0].finish_reason

    # test streaming
    stream = await async_client.chat.completions.create(
        model=model_name,
        messages=messages,
        max_tokens=10,
        temperature=0.0,
        logprobs=False,
        stream=True,
    )
    str_chunks: List[str] = []

    finish_reason_counter = 0
    finish_reason: str = None
    async for chunk in stream:
        choice = chunk.choices[0]
        delta = choice.delta
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
        raise RuntimeError("finish_reason not in [length, stop]")
    # test generated tokens
    assert "".join(str_chunks) == output


@pytest.mark.asyncio(loop_scope="module")
async def test_chat_streaming_with_logprobs(async_client: openai.AsyncOpenAI,
                                            model_name: str, backend: str):
    if backend == "pytorch":
        pytest.skip("Logprobs are not supported in PyTorch backend yet")

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
        max_tokens=10,
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
        max_tokens=10,
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
        raise RuntimeError("finish_reason not in [length, stop]")
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
        max_tokens=10,
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
        max_tokens=10,
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
            max_tokens=10,
            temperature=0.0,
            stream=False,
            stream_options={"include_usage": None})

    # Test stream=False, stream_options={"include_usage": True}
    with pytest.raises(openai.BadRequestError):
        await async_client.chat.completions.create(
            model=model_name,
            messages=messages,
            max_tokens=10,
            temperature=0.0,
            stream=False,
            stream_options={"include_usage": True})

    # Test stream=True, stream_options={"include_usage": True,
    #                           "continuous_usage_stats": True}
    stream = await async_client.chat.completions.create(
        model=model_name,
        messages=messages,
        max_tokens=10,
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
        max_tokens=10,
        temperature=0.0,
        stop="two",
    )
    assert resp.choices[0].finish_reason == "stop"
    assert resp.choices[0].stop_reason == "two"
