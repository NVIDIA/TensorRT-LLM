# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from utils.llm_data import llm_models_root
from utils.util import get_current_process_gpu_memory

from tensorrt_llm import LLM
from tensorrt_llm.llmapi import KvCacheConfig, SamplingParams
from tensorrt_llm.llmapi.llm_args import CudaGraphConfig, ExecutorMemoryType, SleepConfig


@pytest.mark.parametrize(
    "sleep_tags,restore_mode",
    [([ExecutorMemoryType.KV_CACHE], "CPU"), (None, "NONE")],
    ids=["kv_cache_cpu", "all_tags_none"],
)
def test_llm_sleep(process_gpu_memory_info_available, sleep_tags, restore_mode):
    llama_model_path = str(llm_models_root() / "Qwen3/Qwen3-0.6B")
    kv_cache_config = KvCacheConfig(
        enable_block_reuse=True,
        max_tokens=2048,
        # Reuse across CPU sleep needs host capacity, not the memlock-capped automatic tier.
        host_cache_size=512 << 20,
        use_kv_cache_manager_v2=True,
    )

    llm = LLM(
        model=llama_model_path,
        sleep_config=SleepConfig(restore_modes={ExecutorMemoryType.KV_CACHE: restore_mode}),
        kv_cache_config=kv_cache_config,
        max_seq_len=512,
        max_batch_size=4,
        max_num_tokens=512,
        cuda_graph_config=CudaGraphConfig(batch_sizes=[1, 2, 4], enable_padding=True),
        ray_worker_extension_cls="utils.sleep.V2SleepWorkerExtension",
    )

    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]

    prompts = [prompt * 20 for prompt in prompts[:3]]
    prompts = [llm.tokenizer.encode(prompt) for prompt in prompts]
    sampling_params = SamplingParams(temperature=0, max_tokens=16, return_perf_metrics=True)

    with llm:
        llm._collective_rpc("assert_v2_cache_manager")
        llm._collective_rpc("assert_v2_sleep_capacity", (restore_mode,))
        outputs = llm.generate(prompts, sampling_params)
        generated_before_sleep = [output.outputs[0].text for output in outputs]

        warm_outputs = llm.generate(prompts, sampling_params)
        assert any(
            output.outputs[0].request_perf_metrics.kv_cache_metrics.num_reused_blocks > 0
            for output in warm_outputs
        )

        memory_usage_active = get_current_process_gpu_memory(True)

        llm.release(sleep_tags)
        assert llm.get_memory_status().state == "parked"
        parked_prefixes = llm._collective_rpc("get_v2_prefix_match_lengths", (prompts,))
        if restore_mode == "NONE":
            assert all(length == 0 for lengths in parked_prefixes for length in lengths)
        else:
            assert all(any(length > 0 for length in lengths) for lengths in parked_prefixes), (
                f"CPU sleep lost reusable prefixes: {parked_prefixes}"
            )

        memory_usage_sleep = get_current_process_gpu_memory(True)
        if process_gpu_memory_info_available:
            assert memory_usage_sleep < memory_usage_active

        llm.resume(sleep_tags)
        assert llm.get_memory_status().state == "running"
        restored_prefixes = llm._collective_rpc("get_v2_prefix_match_lengths", (prompts,))
        assert restored_prefixes == parked_prefixes, (
            f"Wakeup changed prefix state: {parked_prefixes} -> {restored_prefixes}"
        )

        memory_usage_wakeup = get_current_process_gpu_memory(True)
        if process_gpu_memory_info_available:
            assert memory_usage_wakeup > memory_usage_sleep

        outputs = llm.generate(prompts, sampling_params)
        generated_after_sleep = [output.outputs[0].text for output in outputs]
        reused_blocks = [
            output.outputs[0].request_perf_metrics.kv_cache_metrics.num_reused_blocks
            for output in outputs
        ]
        if restore_mode == "NONE":
            assert all(count == 0 for count in reused_blocks)
        else:
            assert any(count > 0 for count in reused_blocks), (
                f"Prefixes survived wakeup but were not reused: {restored_prefixes}; "
                f"reported reused blocks: {reused_blocks}"
            )

    for before, after in zip(generated_before_sleep, generated_after_sleep, strict=True):
        assert before == after, "Generated result mismatch before and after sleep"


def test_llm_sleep_kv_cache_prefix_reuse():
    """Destructive KV-cache release invalidates reusable prefixes in V1."""
    llama_model_path = str(llm_models_root() / "Qwen3/Qwen3-0.6B")
    kv_cache_config = KvCacheConfig(
        enable_block_reuse=True,
        max_tokens=16384,
        tokens_per_block=32,
        use_kv_cache_manager_v2=False,
    )
    sampling_params = SamplingParams(temperature=0, max_tokens=16, return_perf_metrics=True)

    with LLM(
        model=llama_model_path,
        sleep_config=SleepConfig(restore_modes={ExecutorMemoryType.KV_CACHE: "NONE"}),
        kv_cache_config=kv_cache_config,
    ) as llm:
        prompt = llm.tokenizer.encode("The future of AI is " * 20)
        assert len(prompt) > kv_cache_config.tokens_per_block

        cold_output = llm.generate(prompt, sampling_params).outputs[0]
        before_release = llm.generate(prompt, sampling_params).outputs[0]
        assert before_release.request_perf_metrics is not None
        assert before_release.request_perf_metrics.kv_cache_metrics.num_reused_blocks > 0
        assert cold_output.token_ids == before_release.token_ids

        llm.release([ExecutorMemoryType.KV_CACHE])
        llm.resume([ExecutorMemoryType.KV_CACHE])

        after_resume = llm.generate(prompt, sampling_params).outputs[0]
        assert after_resume.request_perf_metrics is not None
        assert after_resume.request_perf_metrics.kv_cache_metrics.num_reused_blocks == 0
        assert after_resume.token_ids
        assert before_release.token_ids == after_resume.token_ids


def test_llm_sleep_discard_weights(process_gpu_memory_info_available):
    """Sleep-wakeup with NONE restore mode for model weights.

    After wakeup the weight memory is re-materialized but the original values
    are gone (NONE = no backup).  The model should still be able to run a
    forward pass without crashing — output correctness is not expected.
    """
    llama_model_path = str(llm_models_root() / "Qwen3/Qwen3-0.6B")
    kv_cache_config = KvCacheConfig(enable_block_reuse=False, max_tokens=16384)

    sleep_config = SleepConfig(
        restore_modes={
            ExecutorMemoryType.MODEL_WEIGHTS_MAIN: "NONE",
            ExecutorMemoryType.KV_CACHE: "NONE",
        }
    )

    llm = LLM(
        model=llama_model_path,
        sleep_config=sleep_config,
        kv_cache_config=kv_cache_config,
    )

    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]

    sampling_params = SamplingParams(temperature=0)

    with llm:
        outputs = llm.generate(prompts, sampling_params)
        assert all(len(output.outputs[0].text) > 0 for output in outputs)

        memory_usage_active = get_current_process_gpu_memory(True)

        llm._collective_rpc(
            "sleep",
            (
                [
                    ExecutorMemoryType.MODEL_WEIGHTS_MAIN,
                ],
            ),
        )

        memory_usage_sleep = get_current_process_gpu_memory(True)
        if process_gpu_memory_info_available:
            assert memory_usage_sleep < memory_usage_active

        llm._collective_rpc(
            "wakeup",
            (
                [
                    ExecutorMemoryType.MODEL_WEIGHTS_MAIN,
                ],
            ),
        )

        memory_usage_wakeup = get_current_process_gpu_memory(True)
        if process_gpu_memory_info_available:
            assert memory_usage_wakeup > memory_usage_sleep

        # Can generate something without crashing
        outputs = llm.generate(prompts, sampling_params)
        assert all(output.outputs[0] is not None for output in outputs)
