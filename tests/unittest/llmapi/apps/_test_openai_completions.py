# Adapted from
# https://github.com/vllm-project/vllm/blob/aae6927be06dedbda39c6b0c30f6aa3242b84388/tests/entrypoints/openai/test_completion.py

from typing import List

import numpy as np
import openai
import pytest
import yaml

from ..test_llm import get_model_path
from .openai_server import RemoteOpenAIServer
from .utils import (invalid_logit_bias_helper, logit_bias_effect_helper,
                    make_server_with_custom_sampler_fixture)


@pytest.fixture(scope="module")
def model_name():
    return "llama-models-v2/TinyLlama-1.1B-Chat-v1.0"


@pytest.fixture(scope="module", params=["trt", "pytorch"])
def backend(request):
    return request.param


@pytest.fixture(scope="module")
def temp_extra_llm_api_options_file(tmp_path_factory):
    extra_llm_api_options_dict = {
        "enable_chunked_prefill": False,
        "gather_generation_logits": True,
        "kv_cache_config": {
            "enable_block_reuse": False,
        }
    }

    temp_file_path = tmp_path_factory.mktemp(
        "config") / "extra_llm_api_options.yaml"
    with open(temp_file_path, 'w') as f:
        yaml.dump(extra_llm_api_options_dict, f)
    return temp_file_path


@pytest.fixture(scope="module",
                params=[0, 2],
                ids=["disable_processpool", "enable_processpool"])
def num_postprocess_workers(request):
    return request.param


@pytest.fixture(scope="module")
def server(model_name: str, backend: str, num_postprocess_workers: int,
           temp_extra_llm_api_options_file: str):
    model_path = get_model_path(model_name)
    args = ["--backend", f"{backend}"]
    args.extend(["--kv_cache_free_gpu_memory_fraction",
                 "0.2"])  # for co-existence with other servers
    args.extend(["--num_postprocess_workers", f"{num_postprocess_workers}"])
    if backend == "trt":
        args.extend(
            ["--extra_llm_api_options", temp_extra_llm_api_options_file])
    with RemoteOpenAIServer(model_path, args) as remote_server:
        yield remote_server


@pytest.fixture(scope="module")
def server_with_beam_search(model_name: str, backend: str,
                            num_postprocess_workers: int):
    model_path = get_model_path(model_name)
    args = ["--backend", f"{backend}"]
    args.extend(["--kv_cache_free_gpu_memory_fraction",
                 "0.2"])  # for co-existence with other servers
    args.extend(["--max_beam_width", "2"])
    args.extend(["--num_postprocess_workers", f"{num_postprocess_workers}"])
    with RemoteOpenAIServer(model_path, args) as remote_server:
        yield remote_server


@pytest.fixture(scope="module")
def client(server: RemoteOpenAIServer):
    return server.get_client()


@pytest.fixture(scope="module")
def async_client(server: RemoteOpenAIServer):
    return server.get_async_client()


@pytest.fixture(scope="module")
def async_client_with_beam_search(server_with_beam_search: RemoteOpenAIServer):
    return server_with_beam_search.get_async_client()


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
    assert completion.usage.completion_tokens == completion_tokens
    assert completion.usage.prompt_tokens == prompt_tokens
    assert completion.usage.total_tokens == prompt_tokens + completion_tokens

    # test using token IDs
    completion = client.completions.create(
        model=model_name,
        prompt=[0, 0, 0, 0, 0],
        max_tokens=5,
        temperature=0.0,
    )
    assert len(completion.choices[0].text) >= 1


def test_single_completion_with_too_long_prompt(client: openai.OpenAI,
                                                model_name):
    completion = client.completions.create(
        model=model_name,
        prompt="Hello, my name is" * 100,
        max_tokens=5,
        temperature=0.0,
    )

    print(completion)


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
async def test_batch_completions_beam_search(
        async_client_with_beam_search: openai.AsyncOpenAI, model_name, prompts):
    # test beam search
    batch = await async_client_with_beam_search.completions.create(
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


@pytest.mark.asyncio(loop_scope="module")
async def test_completion_stream_options(async_client: openai.AsyncOpenAI,
                                         model_name: str):
    prompt = "Hello, my name is"

    # Test stream=True, stream_options=
    #     {"include_usage": False, "continuous_usage_stats": False}
    stream = await async_client.completions.create(model=model_name,
                                                   prompt=prompt,
                                                   max_tokens=5,
                                                   temperature=0.0,
                                                   stream=True,
                                                   stream_options={
                                                       "include_usage":
                                                       False,
                                                       "continuous_usage_stats":
                                                       False,
                                                   })

    async for chunk in stream:
        assert chunk.usage is None

    # Test stream=True, stream_options=
    #     {"include_usage": False, "continuous_usage_stats": True}
    stream = await async_client.completions.create(model=model_name,
                                                   prompt=prompt,
                                                   max_tokens=5,
                                                   temperature=0.0,
                                                   stream=True,
                                                   stream_options={
                                                       "include_usage":
                                                       False,
                                                       "continuous_usage_stats":
                                                       True,
                                                   })
    async for chunk in stream:
        assert chunk.usage is None

    # Test stream=True, stream_options=
    #     {"include_usage": True, "continuous_usage_stats": False}
    stream = await async_client.completions.create(model=model_name,
                                                   prompt=prompt,
                                                   max_tokens=5,
                                                   temperature=0.0,
                                                   stream=True,
                                                   stream_options={
                                                       "include_usage":
                                                       True,
                                                       "continuous_usage_stats":
                                                       False,
                                                   })
    async for chunk in stream:
        if chunk.choices[0].finish_reason is None:
            assert chunk.usage is None
        else:
            assert chunk.usage is None
            final_chunk = await stream.__anext__()
            assert final_chunk.usage is not None
            assert final_chunk.usage.prompt_tokens > 0
            assert final_chunk.usage.completion_tokens > 0
            assert final_chunk.usage.total_tokens == (
                final_chunk.usage.prompt_tokens +
                final_chunk.usage.completion_tokens)
            assert final_chunk.choices == []

    # Test stream=True, stream_options=
    #     {"include_usage": True, "continuous_usage_stats": True}
    stream = await async_client.completions.create(model=model_name,
                                                   prompt=prompt,
                                                   max_tokens=5,
                                                   temperature=0.0,
                                                   stream=True,
                                                   stream_options={
                                                       "include_usage":
                                                       True,
                                                       "continuous_usage_stats":
                                                       True,
                                                   })
    async for chunk in stream:
        assert chunk.usage is not None
        assert chunk.usage.prompt_tokens > 0
        assert chunk.usage.completion_tokens > 0
        assert chunk.usage.total_tokens == (chunk.usage.prompt_tokens +
                                            chunk.usage.completion_tokens)
        if chunk.choices[0].finish_reason is not None:
            final_chunk = await stream.__anext__()
            assert final_chunk.usage is not None
            assert final_chunk.usage.prompt_tokens > 0
            assert final_chunk.usage.completion_tokens > 0
            assert final_chunk.usage.total_tokens == (
                final_chunk.usage.prompt_tokens +
                final_chunk.usage.completion_tokens)
            assert final_chunk.choices == []

    # Test stream=False, stream_options=
    #     {"include_usage": None}
    with pytest.raises(openai.BadRequestError):
        await async_client.completions.create(
            model=model_name,
            prompt=prompt,
            max_tokens=5,
            temperature=0.0,
            stream=False,
            stream_options={"include_usage": None})

    # Test stream=False, stream_options=
    #    {"include_usage": True}
    with pytest.raises(openai.BadRequestError):
        await async_client.completions.create(
            model=model_name,
            prompt=prompt,
            max_tokens=5,
            temperature=0.0,
            stream=False,
            stream_options={"include_usage": True})

    # Test stream=False, stream_options=
    #     {"continuous_usage_stats": None}
    with pytest.raises(openai.BadRequestError):
        await async_client.completions.create(
            model=model_name,
            prompt=prompt,
            max_tokens=5,
            temperature=0.0,
            stream=False,
            stream_options={"continuous_usage_stats": None})

    # Test stream=False, stream_options=
    #    {"continuous_usage_stats": True}
    with pytest.raises(openai.BadRequestError):
        await async_client.completions.create(
            model=model_name,
            prompt=prompt,
            max_tokens=5,
            temperature=0.0,
            stream=False,
            stream_options={"continuous_usage_stats": True})


def test_detokenize_single(client: openai.OpenAI, model_name):
    completion = client.completions.create(
        model=model_name,
        prompt="Hello, my name is",
        max_tokens=5,
        temperature=0.0,
        extra_body=dict(detokenize=False),
    )

    choice = completion.choices[0]
    assert choice.text == ""
    assert isinstance(choice.token_ids, list)
    assert len(choice.token_ids) > 0

    # test using token IDs
    completion = client.completions.create(
        model=model_name,
        prompt=[0, 0, 0, 0, 0],
        max_tokens=5,
        temperature=0.0,
        extra_body=dict(detokenize=True),
    )

    assert completion.choices[0].token_ids is None


@pytest.mark.asyncio(loop_scope="module")
async def test_completion_streaming(async_client: openai.AsyncOpenAI,
                                    model_name: str):
    prompt = "Hello, my name is"

    single_completion = await async_client.completions.create(
        model=model_name,
        prompt=prompt,
        max_tokens=5,
        temperature=0.0,
        extra_body=dict(detokenize=False),
    )
    single_output = single_completion.choices[0].token_ids
    stream = await async_client.completions.create(
        model=model_name,
        prompt=prompt,
        max_tokens=5,
        temperature=0.0,
        stream=True,
        extra_body=dict(detokenize=False),
    )
    tokens: List[int] = []

    async for chunk in stream:
        assert chunk.choices[0].text == ""
        assert isinstance(chunk.choices[0].token_ids, list)
        tokens.extend(chunk.choices[0].token_ids)

    assert tokens == single_output


# Use the shared fixture from utils.py
server_with_custom_sampler = make_server_with_custom_sampler_fixture(
    'completions')


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
async def test_completion_with_logit_bias_effect(
        server_with_custom_sampler: RemoteOpenAIServer,
        model_name: str) -> None:
    '''Test that logit bias affects output as expected for both samplers (completions endpoint).'''
    client = server_with_custom_sampler.get_async_client()
    await logit_bias_effect_helper(client, model_name, 'completions')


@pytest.mark.asyncio(loop_scope="module")
async def test_completion_with_invalid_logit_bias(
        async_client: openai.AsyncOpenAI, model_name: str):
    """Test with invalid token IDs (non-integer keys)"""
    await invalid_logit_bias_helper(async_client, model_name, 'completions')


def test_completion_logprobs(client: openai.OpenAI, model_name: str,
                             backend: str, num_postprocess_workers: int):
    """Test completion with logprobs enabled (non-streaming)."""
    if backend == "trt" and num_postprocess_workers > 0:
        pytest.skip("Logprobs is not supported in TRT processpool mode")

    prompt = "Hello, my name is"

    completion = client.completions.create(
        model=model_name,
        prompt=prompt,
        max_tokens=5,
        temperature=0.0,
        logprobs=1,
    )

    choice = completion.choices[0]
    assert choice.logprobs is not None

    # Verify logprobs structure
    logprobs = choice.logprobs
    assert logprobs.tokens is not None
    assert logprobs.token_logprobs is not None
    assert logprobs.text_offset is not None

    # Verify lengths match
    assert len(logprobs.tokens) == len(logprobs.token_logprobs)
    assert len(logprobs.tokens) == len(logprobs.text_offset)
    assert len(logprobs.tokens) > 0

    # Verify logprobs values are valid (negative or zero for log probabilities)
    for token_logprob in logprobs.token_logprobs:
        assert token_logprob is not None
        assert token_logprob <= 0

    # Verify text_offset is monotonically increasing
    for i in range(1, len(logprobs.text_offset)):
        assert logprobs.text_offset[i] >= logprobs.text_offset[i - 1]

    # Verify tokens are non-empty strings
    for token in logprobs.tokens:
        assert isinstance(token, str)


@pytest.mark.asyncio(loop_scope="module")
async def test_completion_logprobs_streaming(async_client: openai.AsyncOpenAI,
                                             backend: str, model_name: str,
                                             num_postprocess_workers: int):
    """Test completion with logprobs enabled (streaming)."""
    if backend == "trt" and num_postprocess_workers > 0:
        pytest.skip("Logprobs is not supported in TRT processpool mode")

    prompt = "Hello, my name is"

    # First get non-streaming result for comparison
    single_completion = await async_client.completions.create(
        model=model_name,
        prompt=prompt,
        max_tokens=5,
        temperature=0.0,
        logprobs=1,
    )
    single_logprobs = single_completion.choices[0].logprobs
    assert single_logprobs is not None

    # Now test streaming
    stream = await async_client.completions.create(
        model=model_name,
        prompt=prompt,
        max_tokens=5,
        temperature=0.0,
        logprobs=2,
        stream=True,
    )

    all_tokens: List[str] = []
    all_token_logprobs: List[float] = []

    async for chunk in stream:
        choice = chunk.choices[0]
        if choice.logprobs is not None:
            if choice.logprobs.tokens:
                all_tokens.extend(choice.logprobs.tokens)
            if choice.logprobs.token_logprobs:
                all_token_logprobs.extend(choice.logprobs.token_logprobs)

    # Verify streaming logprobs match non-streaming
    assert all_tokens == single_logprobs.tokens
    assert len(all_token_logprobs) == len(single_logprobs.token_logprobs)

    # Compare logprobs values (should be close)
    all_token_logprobs_arr = np.array(all_token_logprobs)
    single_token_logprobs_arr = np.array(single_logprobs.token_logprobs)
    assert np.allclose(all_token_logprobs_arr, single_token_logprobs_arr)

    # Verify all logprobs are valid
    for logprob in all_token_logprobs:
        assert logprob is not None
        assert logprob <= 0


def test_completion_cached_tokens(client: openai.OpenAI, model_name: str,
                                  backend: str):
    if backend == "trt":
        pytest.skip("Cached tokens is not supported in trt backend yet")

    prompt = "This is a test prompt"

    # Run the completion for the first time
    single_completion = client.completions.create(
        model=model_name,
        prompt=prompt,
        max_tokens=5,
        temperature=0.0,
    )
    expected_cached_tokens = single_completion.usage.prompt_tokens - 1

    # Run the completion for the second time
    single_completion = client.completions.create(
        model=model_name,
        prompt=prompt,
        max_tokens=5,
        temperature=0.0,
    )
    assert single_completion.usage is not None
    assert single_completion.usage.prompt_tokens_details is not None
    assert single_completion.usage.prompt_tokens_details.cached_tokens == expected_cached_tokens


@pytest.mark.asyncio(loop_scope="module")
async def test_completion_cached_tokens_stream(async_client: openai.AsyncOpenAI,
                                               model_name: str, backend: str):
    if backend == "trt":
        pytest.skip("Cached tokens is not supported in trt backend yet")

    prompt = "This is a test prompt"

    # Run the completion for the first time so that cached tokens are created
    single_completion = await async_client.completions.create(
        model=model_name,
        prompt=prompt,
        max_tokens=5,
        temperature=0.0,
    )
    expected_cached_tokens = single_completion.usage.prompt_tokens - 1

    # Test stream=True, stream_options=
    #     {"include_usage": True, "continuous_usage_stats": False}
    stream = await async_client.completions.create(model=model_name,
                                                   prompt=prompt,
                                                   max_tokens=5,
                                                   temperature=0.0,
                                                   stream=True,
                                                   stream_options={
                                                       "include_usage":
                                                       True,
                                                       "continuous_usage_stats":
                                                       False,
                                                   })
    async for chunk in stream:
        if chunk.choices[0].finish_reason is None:
            assert chunk.usage is None
        else:
            assert chunk.usage is None
            final_chunk = await stream.__anext__()
            assert final_chunk.usage is not None
            assert final_chunk.usage.prompt_tokens > 0
            assert final_chunk.usage.completion_tokens > 0
            assert final_chunk.usage.total_tokens == (
                final_chunk.usage.prompt_tokens +
                final_chunk.usage.completion_tokens)
            assert final_chunk.usage.prompt_tokens_details is not None
            assert final_chunk.usage.prompt_tokens_details.cached_tokens == expected_cached_tokens
            assert final_chunk.choices == []

    # Test stream=True, stream_options=
    #     {"include_usage": True, "continuous_usage_stats": True}
    stream = await async_client.completions.create(model=model_name,
                                                   prompt=prompt,
                                                   max_tokens=5,
                                                   temperature=0.0,
                                                   stream=True,
                                                   stream_options={
                                                       "include_usage":
                                                       True,
                                                       "continuous_usage_stats":
                                                       True,
                                                   })
    async for chunk in stream:
        assert chunk.usage is not None
        assert chunk.usage.prompt_tokens > 0
        assert chunk.usage.completion_tokens > 0
        assert chunk.usage.prompt_tokens_details is not None
        assert chunk.usage.prompt_tokens_details.cached_tokens == expected_cached_tokens
        assert chunk.usage.total_tokens == (chunk.usage.prompt_tokens +
                                            chunk.usage.completion_tokens)
        if chunk.choices[0].finish_reason is not None:
            final_chunk = await stream.__anext__()
            assert final_chunk.usage is not None
            assert final_chunk.usage.prompt_tokens > 0
            assert final_chunk.usage.completion_tokens > 0
            assert final_chunk.usage.total_tokens == (
                final_chunk.usage.prompt_tokens +
                final_chunk.usage.completion_tokens)
            assert final_chunk.usage.prompt_tokens_details is not None
            assert final_chunk.usage.prompt_tokens_details.cached_tokens == expected_cached_tokens
            assert final_chunk.choices == []
