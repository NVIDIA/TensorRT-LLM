# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
from typing import List, Optional, Union

import pytest
import torch
from gc_utils import assert_resource_freed
from utils.llm_data import llm_models_root
from utils.util import similar

from tensorrt_llm import LLM
from tensorrt_llm.executor import RequestError
from tensorrt_llm.llmapi import KvCacheConfig, RequestOutput
from tensorrt_llm.llmapi.tokenizer import TransformersTokenizer
from tensorrt_llm.sampling_params import LogitsProcessor, SamplingParams

from .model_paths import default_model_name, get_model_path, llama_model_path


def check_output(
    outputs: List[RequestOutput],
    references: Union[List[str], List[List[str]]],
    *,
    similar_threshold: float = 0.8,
    finish_reasons: Optional[List[str]] = None,
    stop_reasons: Optional[List[Union[int, str]]] = None,
):
    assert len(outputs) == len(references)

    for i, (output, reference) in enumerate(zip(outputs, references)):
        if isinstance(reference, list):
            # N output
            assert len(output.outputs) == len(reference)
            for j, (out, ref) in enumerate(zip(output.outputs, reference)):
                assert similar(out.text, ref, threshold=similar_threshold)
                if finish_reasons is not None:
                    assert out.finish_reason == finish_reasons[i][j]
                if stop_reasons is not None:
                    assert out.stop_reason == stop_reasons[i][j]
        else:
            out = output.outputs[0]
            assert similar(out.text, reference, threshold=similar_threshold)
            if finish_reasons is not None:
                assert out.finish_reason == finish_reasons[i]
            if stop_reasons is not None:
                assert out.stop_reason == stop_reasons[i]


def llm_test_harness(
    model_dir: str,
    inputs: List[str],
    references: List[str],
    *,
    sampling_params: Optional[SamplingParams] = None,
    similar_threshold: float = 0.8,
    **llm_kwargs,
):
    tp_size = llm_kwargs.get("tensor_parallel_size", 1)
    pp_size = llm_kwargs.get("pipeline_parallel_size", 1)
    world_size = tp_size * pp_size
    if world_size > torch.cuda.device_count():
        pytest.skip(
            f"world_size ({world_size}) is greater than available GPUs ({torch.cuda.device_count()})"
        )

    tokenizer = llm_kwargs.pop("tokenizer", None)
    if tokenizer is None:
        tokenizer = model_dir

    with assert_resource_freed(LLM, model_dir, tokenizer, **llm_kwargs) as llm:
        outputs = llm.generate(inputs, sampling_params=sampling_params)
        print(outputs)
        check_output(outputs, references, similar_threshold=similar_threshold)


def llm_check_output(
    llm: LLM,
    inputs: List[str],
    references: List[str],
    *,
    sampling_params: Optional[SamplingParams] = None,
    similar_threshold: float = 0.8,
    finish_reasons: Optional[List[str]] = None,
    stop_reasons: Optional[List[Union[int, str]]] = None,
    **gen_kwargs,
):
    outputs = llm.generate(inputs, sampling_params=sampling_params, **gen_kwargs)
    print(outputs)
    check_output(
        outputs,
        references,
        similar_threshold=similar_threshold,
        finish_reasons=finish_reasons,
        stop_reasons=stop_reasons,
    )
    return outputs


class MyLogitsProcessor(LogitsProcessor):
    def __init__(self, biased_word_id):
        self.biased_word_id = biased_word_id

    def __call__(
        self,
        req_id: int,
        logits: torch.Tensor,
        ids: List[List[int]],
        stream_ptr: int,
        client_id: Optional[int],
    ):
        stream = None if stream_ptr is None else torch.cuda.ExternalStream(stream_ptr)
        with torch.cuda.stream(stream):
            logits[:] = float("-inf")
            logits[..., self.biased_word_id] = 0


def tinyllama_logits_processor_test_harness(backend=None, **llm_kwargs):
    tokenizer = TransformersTokenizer.from_pretrained(llama_model_path)
    biased_word_id = tokenizer.encode("Z", add_special_tokens=False)[-1]
    max_tokens = 6
    sampling_params = SamplingParams(
        max_tokens=max_tokens, logits_processor=MyLogitsProcessor(biased_word_id)
    )

    prompts = ["A B C"]
    if llm_kwargs.get("enable_chunked_prefill", None):
        prompts[0] = prompts[0] * 256
        llm_kwargs["max_num_tokens"] = 256

    # The biased token's decoded text (e.g. with or without a leading space)
    # is tokenizer-specific, so derive the expected repeated-token output
    # rather than hardcoding a particular tokenizer's rendering.
    expected_output = tokenizer.decode([biased_word_id] * max_tokens)

    llm_test_harness(
        llama_model_path,
        prompts,
        [expected_output],
        sampling_params=sampling_params,
        kv_cache_config=KvCacheConfig(free_gpu_memory_fraction=0.4),
        backend=backend,
        **llm_kwargs,
    )


def _test_llm_generate_async(model_name=default_model_name, tp_size: int = 1, tokenizer=None):
    llm = LLM(
        model=get_model_path(model_name),
        tokenizer=tokenizer,
        kv_cache_config=global_kvcache_config,
        tensor_parallel_size=tp_size,
    )

    sampling_params = SamplingParams(max_tokens=6)

    def test_async(streaming: bool):
        async def task(prompt: str):
            outputs = []
            async for output in llm.generate_async(
                prompt, streaming=streaming, sampling_params=sampling_params
            ):
                print("output", output)
                outputs.append(output.outputs[0].text)
            print(" ".join(outputs))

        async def main():
            tasks = [task(prompt) for prompt in prompts]
            await asyncio.gather(*tasks)

        asyncio.run(main())

    def test_wait(streaming: bool):
        for prompt in prompts:
            future = llm.generate_async(
                prompt, streaming=streaming, sampling_params=sampling_params
            )
            for output in future:
                print("wait", output)

    def test_non_streaming_usage_wait():
        for prompt in prompts:
            output = llm.generate_async(prompt, streaming=False, sampling_params=sampling_params)
            print(output.outputs[0].text)

    def test_future(streaming: bool):
        for prompt in prompts:
            future = llm.generate_async(
                prompt, streaming=streaming, sampling_params=sampling_params
            )
            if streaming is True:
                for output in future:
                    # Do something else and then wait for the result if needed
                    output = output.result(timeout=10)
                    print("future", output.outputs[0].text)
            else:
                # Do something else and then wait for the result if needed
                output = future.result(timeout=10)
                print("future", output.outputs[0].text)

    def test_future_async():
        async def task(prompt: str):
            future = llm.generate_async(prompt, streaming=False, sampling_params=sampling_params)
            output = await future.aresult()
            print("future", output.outputs[0].text)

        async def main():
            tasks = [task(prompt) for prompt in prompts]
            await asyncio.gather(*tasks)

        asyncio.run(main())

    test_async(streaming=True)
    test_async(streaming=False)
    test_wait(streaming=True)
    test_wait(streaming=False)
    test_future(streaming=True)
    test_future(streaming=False)
    test_future_async()
    test_non_streaming_usage_wait()


def _test_llm_capture_request_error(pytorch_backend: bool, tp_size: int = 1):
    llm = LLM(
        model=llama_model_path,
        max_num_tokens=64,
        tensor_parallel_size=tp_size,
    )

    prompt = "A " * 65  # the minimum max_num_tokens is 64
    # Both backends now consistently raise RequestError for max_num_tokens validation
    with pytest.raises(RequestError):
        llm.generate(prompt)


def run_llm_abort_request(llm: LLM, sampling_params: SamplingParams):
    # to make sure LLM run slower for canceling the request to be actually performed
    sampling_params.max_tokens = 100
    sampling_params.end_id = -1  # let it run for a while

    async def task():
        result = llm.generate_async(prompts[0], sampling_params=sampling_params, streaming=True)
        print("to abort")
        result.abort()

        print("waiting for the result")
        # Before it actually abort, we should see some outputs
        outputs = []
        async for output in result:
            print(f"get output: {output}")
            outputs.append(output)
        print(f"get {len(outputs)} remaining outputs")
        print(f"outputs: {outputs}")
        print(f"finish_reason: {outputs[-1].outputs[0].finish_reason}")
        assert 1 <= len(outputs) < 1000  # It should be aborted before the completion
        # NOTE: known issue: only the last output is finished and got the finish_reason
        assert outputs[-1].outputs[-1].finish_reason == "cancelled"

    asyncio.run(task())


sampling_params_for_aborting_request = [
    SamplingParams(),
    # n-returns
    SamplingParams(n=2, top_k=2),
    SamplingParams(n=2, top_k=2, best_of=3),
    SamplingParams(n=3, use_beam_search=True),
    SamplingParams(n=2, best_of=3, use_beam_search=True),
]


prompts = ["A B C"]


global_kvcache_config = KvCacheConfig(free_gpu_memory_fraction=0.4)


# python api does not seem to support extra tokens needed for prompt tuning + reuse.
# disable block reuse for those tests.
# TODO: Add extra tokens to prompt tuning unit tests.
global_kvcache_config_no_reuse = KvCacheConfig(
    free_gpu_memory_fraction=0.4, enable_block_reuse=False
)


cnn_dailymail_path = str(llm_models_root() / "datasets" / "cnn_dailymail")


alpaca_chinese_path = str(llm_models_root() / "datasets" / "silk-road" / "alpaca-data-gpt4-chinese")
