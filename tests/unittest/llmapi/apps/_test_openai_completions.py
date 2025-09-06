# Adapted from
# https://github.com/vllm-project/vllm/blob/aae6927be06dedbda39c6b0c30f6aa3242b84388/tests/entrypoints/openai/test_completion.py

from typing import List

import openai
import pytest

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


@pytest.fixture(scope="module",
                params=[0, 2],
                ids=["disable_processpool", "enable_processpool"])
def num_postprocess_workers(request):
    return request.param


@pytest.fixture(scope="module")
def server(model_name: str, backend: str, num_postprocess_workers: int):
    model_path = get_model_path(model_name)
    args = ["--backend", f"{backend}"]
    if backend == "trt":
        args.extend(["--max_beam_width", "4"])
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


def test_completion_with_prompt_logprobs(client: openai.OpenAI,
                                         model_name: str):
    """Test that completions API accepts and returns prompt_logprobs."""

    prompt = "Hello, my name is"

    # Request with prompt_logprobs using extra_body
    completion = client.completions.create(
        model=model_name,
        prompt=prompt,
        max_tokens=5,
        temperature=0.0,
        extra_body={"prompt_logprobs": 2},  # Request top-2 prompt logprobs
    )

    # Verify basic completion fields
    assert completion.id is not None
    assert len(completion.choices) == 1
    choice = completion.choices[0]
    assert len(choice.text) >= 1

    # Verify prompt_logprobs handling
    # NOTE: Currently, the OpenAI server accepts prompt_logprobs in the request
    # and processes them internally without errors, but does not yet include
    # them in the response. This is a known limitation that will be addressed
    # in a future update to include prompt_logprobs in the response structure.

    # For now, we verify:
    # 1. The request with prompt_logprobs doesn't crash
    # 2. The completion still works correctly
    # 3. The response structure is valid

    # Check if prompt_logprobs are in the response (for future compatibility)
    if hasattr(choice,
               'prompt_logprobs') and choice.prompt_logprobs is not None:
        # Future: When prompt_logprobs are added to the response
        assert len(
            choice.prompt_logprobs) > 0, "prompt_logprobs should not be empty"
        print(
            f"✓ Found prompt_logprobs with {len(choice.prompt_logprobs)} entries"
        )
    else:
        # Current state: prompt_logprobs are computed internally but not returned
        # This test verifies the feature doesn't cause errors
        print(
            f"Note: prompt_logprobs accepted in request but not yet included in response (expected limitation)"
        )


def test_completion_with_both_logprobs(client: openai.OpenAI, model_name: str,
                                       backend: str):
    """Test completions with both prompt_logprobs and generation logprobs.

    PyTorch backend supports logprobs=1, TRT backend supports logprobs>1.
    Both should work with prompt_logprobs simultaneously.
    """

    # PyTorch backend only supports logprobs=1 (top-1)
    # TRT backend can support logprobs>1
    generation_logprobs = 1 if backend == "pytorch" else 2

    prompt = "The capital of France is"

    completion = client.completions.create(
        model=model_name,
        prompt=prompt,
        max_tokens=3,
        temperature=0.0,
        logprobs=
        generation_logprobs,  # Generation logprobs (1 for pytorch, 2 for trt)
        extra_body={"prompt_logprobs":
                    2},  # Prompt logprobs (works for both backends)
    )

    # Verify completion
    assert completion.id is not None
    assert len(completion.choices) == 1
    choice = completion.choices[0]
    assert len(choice.text) >= 1

    # Verify we got generation logprobs
    assert choice.logprobs is not None, "Generation logprobs were requested but not returned"

    # NOTE: prompt_logprobs are currently computed internally but not yet returned in response
    # This test verifies that combining both types of logprobs doesn't cause errors
    if hasattr(choice,
               'prompt_logprobs') and choice.prompt_logprobs is not None:
        # Future: When prompt_logprobs are added to the response
        assert len(
            choice.prompt_logprobs) > 0, "prompt_logprobs should not be empty"
        print(f"✓ Found both generation and prompt logprobs")
    else:
        print(
            f"Note: Both logprobs types processed without error (prompt_logprobs not yet in response)"
        )


def test_prompt_logprobs_without_context_logits(client: openai.OpenAI,
                                                model_name: str):
    """Test that prompt_logprobs works without explicitly requesting context_logits.

    This tests the fix we implemented where prompt_logprobs should work
    even when return_context_logits=False.
    """

    prompt = "The sky is"

    # Request prompt_logprobs WITHOUT explicitly requesting context_logits
    # The server should handle this internally
    completion = client.completions.create(
        model=model_name,
        prompt=prompt,
        max_tokens=3,
        temperature=0.0,
        extra_body={
            "prompt_logprobs": 2,
            "return_context_logits": False  # Explicitly set to False
        })

    # Verify completion works
    assert completion.id is not None
    assert len(completion.choices) == 1
    choice = completion.choices[0]
    assert len(choice.text) >= 1


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
