# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio

import pytest

from tensorrt_llm import LLM
from tensorrt_llm.executor import RequestError
from tensorrt_llm.llmapi import KvCacheConfig, SADecodingConfig
from tensorrt_llm.llmapi.llm_args import DynamicBatchConfig, SchedulerConfig
from tensorrt_llm.sampling_params import SamplingParams

from .llm_test_utils import global_kvcache_config, llm_test_harness, prompts
from .model_paths import llama_model_path

pytestmark = pytest.mark.threadleak(enabled=False)


def test_executor_results_cleanup():
    llm = LLM(model=llama_model_path, kv_cache_config=global_kvcache_config)
    sampling_params = SamplingParams(max_tokens=6)
    for i in range(20):
        llm.generate(prompts, sampling_params=sampling_params)

    num_remaining_results = len(llm._executor._results)
    print(f"result.size: {num_remaining_results}")
    assert num_remaining_results == 0


def test_oversized_prompt_rejected_with_inflated_max_seq_len():
    """Oversized prompt is rejected gracefully when max_seq_len is inflated internally.

    With speculative decoding, KV cache max_seq_len is inflated to reserve space
    for draft tokens. Admission control must use the net (user-configured) limit,
    not the inflated value. Otherwise oversized prompts crash the event loop.
    """
    max_seq_len = 256

    llm = LLM(
        model=llama_model_path,
        max_seq_len=max_seq_len,
        max_batch_size=1,
        max_num_tokens=(max_seq_len + 32),
        kv_cache_config=KvCacheConfig(enable_block_reuse=False),
        speculative_config=SADecodingConfig(max_draft_len=4),
        disable_overlap_scheduler=True,
    )

    # Prompt that fills the entire net budget -> 0 tokens left for output -> rejected
    oversized_prompt_ids = [1] * (max_seq_len + 1)
    with pytest.raises(RequestError):
        llm.generate([oversized_prompt_ids], sampling_params=SamplingParams(max_tokens=1))

    # Executor must still be alive
    result = llm.generate(["Hello"], sampling_params=SamplingParams(max_tokens=5))
    assert len(result) == 1
    assert len(result[0].outputs[0].text) > 0


def test_llm_shutdown_executor():
    llm = LLM(
        model=llama_model_path,
        kv_cache_config=global_kvcache_config,
    )

    llm.generate("A")
    llm.shutdown()

    with pytest.raises(RuntimeError):
        llm.generate("A")


def test_llm_api_jupyter_scenario():
    with LLM(
        model=llama_model_path,
        kv_cache_config=global_kvcache_config,
    ) as llm:
        sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

        async def task():
            return llm.generate(["A", "B", "C", "D"], sampling_params)

        output = asyncio.run(task())
        for token in output:
            print(token)


def test_llm_dynamic_batch_config():
    scheduler_config = SchedulerConfig(
        dynamic_batch_config=DynamicBatchConfig(
            enable_batch_size_tuning=True,
            enable_max_num_tokens_tuning=True,
            dynamic_batch_moving_average_window=128,
        )
    )
    llm_test_harness(
        llama_model_path,
        prompts,
        ["D E F G H I J K"],
        sampling_params=SamplingParams(max_tokens=9),
        scheduler_config=scheduler_config,
    )
