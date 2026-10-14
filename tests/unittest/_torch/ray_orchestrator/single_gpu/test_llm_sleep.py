# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from contextlib import nullcontext
from unittest.mock import Mock, patch

import pytest
from utils.llm_data import llm_models_root
from utils.util import get_current_process_gpu_memory

from tensorrt_llm import LLM
from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor
from tensorrt_llm.llmapi import KvCacheConfig, SamplingParams
from tensorrt_llm.llmapi.llm_args import CudaGraphConfig, ExecutorMemoryType, SleepConfig


@pytest.mark.parametrize(
    "sleep_tags,restore_mode,use_v2",
    [
        ([ExecutorMemoryType.KV_CACHE], "NONE", True),
        ([ExecutorMemoryType.KV_CACHE], "MEMSET", True),
        ([ExecutorMemoryType.KV_CACHE], "CPU", True),
        ([ExecutorMemoryType.KV_CACHE], "PINNED", True),
        (None, "NONE", True),
        (
            [ExecutorMemoryType.MODEL_ENGINE_MAIN, ExecutorMemoryType.MODEL_WEIGHTS_MAIN],
            "NONE",
            False,
        ),
    ],
    ids=["v2_none", "v2_memset", "v2_cpu", "v2_pinned", "v2_all_tags", "v1_control"],
)
def test_llm_sleep(process_gpu_memory_info_available, sleep_tags, restore_mode, use_v2):
    llama_model_path = str(llm_models_root() / "Qwen3/Qwen3-0.6B")
    kv_cache_config = KvCacheConfig(
        enable_block_reuse=use_v2, max_tokens=16384, use_kv_cache_manager_v2=use_v2
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
    sampling_params = SamplingParams(temperature=0, max_tokens=16, return_perf_metrics=True)

    with llm:
        llm._collective_rpc("assert_cache_manager_version", (use_v2, True))
        outputs = llm.generate(prompts, sampling_params)
        generated_before_sleep = [output.outputs[0].text for output in outputs]

        if use_v2:
            warm_outputs = llm.generate(prompts, sampling_params)
            assert any(
                output.outputs[0].request_perf_metrics.kv_cache_metrics.num_reused_blocks > 0
                for output in warm_outputs
            )

        memory_usage_active = get_current_process_gpu_memory(True)

        llm.release(sleep_tags)
        assert llm.get_memory_status().state == "parked"

        memory_usage_sleep = get_current_process_gpu_memory(True)
        if process_gpu_memory_info_available:
            assert memory_usage_sleep < memory_usage_active

        llm.resume(sleep_tags)
        assert llm.get_memory_status().state == "running"

        memory_usage_wakeup = get_current_process_gpu_memory(True)
        if process_gpu_memory_info_available:
            assert memory_usage_wakeup > memory_usage_sleep
        llm._collective_rpc("assert_cache_manager_version", (use_v2, True))

        outputs = llm.generate(prompts, sampling_params)
        generated_after_sleep = [output.outputs[0].text for output in outputs]
        reused_blocks = [
            output.outputs[0].request_perf_metrics.kv_cache_metrics.num_reused_blocks
            for output in outputs
        ]
        if restore_mode in ("NONE", "MEMSET"):
            assert all(count == 0 for count in reused_blocks)
        elif use_v2:
            assert any(count > 0 for count in reused_blocks)

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
    kv_cache_config = KvCacheConfig(
        enable_block_reuse=False,
        max_tokens=16384,
        use_kv_cache_manager_v2=True,
        host_cache_size=0,
    )

    sleep_config = SleepConfig(
        restore_modes={
            ExecutorMemoryType.MODEL_WEIGHTS_MAIN: "NONE",
            ExecutorMemoryType.KV_CACHE: "CPU",
        }
    )

    llm = LLM(
        model=llama_model_path,
        sleep_config=sleep_config,
        kv_cache_config=kv_cache_config,
        ray_worker_extension_cls="utils.sleep.V2SleepWorkerExtension",
    )

    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]

    sampling_params = SamplingParams(temperature=0)

    with llm:
        llm._collective_rpc("assert_v2_cache_manager")
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


@pytest.mark.parametrize("entrypoint", ["mpi", "ray"])
@pytest.mark.parametrize(
    "tag,use_v2,has_transceiver,rejected",
    [
        ("kv_cache", True, True, True),
        ("kv_cache", False, True, False),
        ("kv_cache", True, False, False),
        ("model", True, True, False),
    ],
)
def test_sleep_validates_registrations_before_closing_admission(
    entrypoint, tag, use_v2, has_transceiver, rejected
):
    from tensorrt_llm.executor.ray.gpu_worker import RayGPUWorker
    from tensorrt_llm.llmapi.llm_args import ExecutorMemoryType, SleepConfig, TorchLlmArgs

    stub = object.__new__(PyExecutor)
    stub._is_kv_manager_v2 = use_v2
    stub.kv_cache_transceiver = Mock() if has_transceiver else None
    stub.executor_request_queue = Mock()
    stub.get_memory_status = Mock(return_value={"state": "running", "parked_tags": []})
    stub.control_action = Mock(return_value=nullcontext())
    stub.invalidate_v1_prefix_cache_for_sleep = Mock()
    stub.validate_sleep_config = Mock()
    tags = [ExecutorMemoryType(tag)]
    if entrypoint == "mpi":
        run = stub.begin_sleep_transition
        mutation = stub.executor_request_queue.begin_sleep_transition
    else:
        worker = object.__new__(RayGPUWorker)
        worker.doing_shutdown = True
        worker.engine = stub
        worker.llm_args = Mock(spec=TorchLlmArgs, sleep_config=SleepConfig())
        run = worker.sleep
        mutation = stub.executor_request_queue.begin_sleep_transition

    if rejected:
        with pytest.raises(NotImplementedError, match="remote memory re-registration"):
            run(tags)
        mutation.assert_not_called()
    else:
        with (
            patch("tensorrt_llm.executor.ray.gpu_worker.logger", Mock(), create=True),
            patch("tensorrt_llm.executor.ray.gpu_worker.release_with_tag"),
            patch("torch.cuda.synchronize"),
            patch("torch.cuda.empty_cache"),
            patch("gc.collect"),
        ):
            run(tags)
        mutation.assert_called_once()
