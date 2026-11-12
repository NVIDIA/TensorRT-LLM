# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import weakref

import pytest
import torch

from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor
from tensorrt_llm._torch.pyexecutor.py_executor_creator import \
    create_py_executor
from tensorrt_llm._torch.pyexecutor.resource_manager import ResourceManagerType
from tensorrt_llm.llmapi import (CapacitySchedulerPolicy, DynamicBatchConfig,
                                 SchedulerConfig)
from tensorrt_llm.llmapi.llm_args import (CudaGraphConfig, KvCacheConfig,
                                          TorchLlmArgs)

# isort: off
from .test_llm import get_model_path
# isort: on

pytestmark = pytest.mark.threadleak(enabled=False)


def test_profile_kvcache():
    kv_cache_config = KvCacheConfig(enable_block_reuse=False,
                                    free_gpu_memory_fraction=0.9)
    cuda_graph_config = CudaGraphConfig(max_batch_size=512)
    VLM_MODEL = "Qwen3/Qwen3-VL-2B-Instruct"
    VLM_MODEL_PATH = get_model_path(VLM_MODEL)

    dynamic_batch_config = DynamicBatchConfig(
        enable_batch_size_tuning=True,
        enable_max_num_tokens_tuning=False,
        dynamic_batch_moving_average_window=128)
    scheduler_config = SchedulerConfig(
        capacity_scheduler_policy=CapacitySchedulerPolicy.GUARANTEED_NO_EVICT,
        dynamic_batch_config=dynamic_batch_config,
    )
    backend = "pytorch"
    llm_args = {
        "model": VLM_MODEL,
        "scheduler_config": scheduler_config,
        "tokenizer": None,
        "tensor_parallel_size": 1,
        "pipeline_parallel_size": 1,
        "moe_expert_parallel_size": None,
        "gpus_per_node": 1,
        "trust_remote_code": False,
        "max_num_tokens": 16384,
        "max_beam_width": 1,
        "kv_cache_config": kv_cache_config,
        "backend": backend,
        "num_postprocess_workers": 0,
        "postprocess_tokenizer_dir": VLM_MODEL,
        "reasoning_parser": None,
        "cuda_graph_config": cuda_graph_config,
    }

    torchllm_args = TorchLlmArgs(**llm_args)

    profiling_data = {"enable_mm_reqs": True}
    py_executor = create_py_executor(llm_args=torchllm_args,
                                     checkpoint_dir=VLM_MODEL_PATH,
                                     profiling_stage_data=profiling_data)
    vlm_activation_bytes_with_mm_reqs = profiling_data["activation_bytes"]
    py_executor.shutdown()
    torch.cuda.empty_cache()

    profiling_data = {"enable_mm_reqs": False}
    torchllm_args = TorchLlmArgs(**llm_args)
    py_executor_2 = create_py_executor(llm_args=torchllm_args,
                                       checkpoint_dir=VLM_MODEL_PATH,
                                       profiling_stage_data=profiling_data)
    vlm_activation_bytes_no_mm_reqs = profiling_data["activation_bytes"]
    py_executor_2.shutdown()
    torch.cuda.empty_cache()

    assert vlm_activation_bytes_with_mm_reqs > vlm_activation_bytes_no_mm_reqs, f"Activation bytes should be higher with mm reqs, but got {vlm_activation_bytes_with_mm_reqs} for mm reqs and {vlm_activation_bytes_no_mm_reqs} without mm reqs"


@pytest.mark.parametrize("release_before_shutdown", [False, True],
                         ids=["bound", "released"])
def test_pyexecutor_and_kvcache_share_execution_stream(monkeypatch,
                                                       release_before_shutdown):
    """Test that PyExecutor and KVCacheManager share the same execution_stream.

    The execution_stream is created once in create_py_executor and passed to:
    - KVCacheManager (via KvCacheCreator -> _create_kv_cache_manager)
    - PyExecutor (via create_py_executor_instance)

    Both components must use the same stream for proper synchronization. The
    profiling and serving phases must also retain the same worker thread.
    """
    monkeypatch.setenv("TRTLLM_SKIP_KV_CACHE_ESTIMATION", "0")
    worker_starts = []
    start_worker = PyExecutor.start_worker

    def record_worker_start(executor):
        start_worker(executor)
        manager = executor.resource_manager.get_resource_manager(
            ResourceManagerType.KV_CACHE_MANAGER)
        assert manager is not None
        # Do not keep the profiling KV allocation alive across teardown.
        worker_starts.append((weakref.ref(manager), executor.worker_thread))

    monkeypatch.setattr(PyExecutor, "start_worker", record_worker_start)

    MODEL = "Qwen3/Qwen3-0.6B"
    MODEL_PATH = get_model_path(MODEL)

    kv_cache_config = KvCacheConfig(enable_block_reuse=False,
                                    free_gpu_memory_fraction=0.5)

    scheduler_config = SchedulerConfig(
        capacity_scheduler_policy=CapacitySchedulerPolicy.GUARANTEED_NO_EVICT, )
    backend = "pytorch"
    llm_args = {
        "model": MODEL,
        "scheduler_config": scheduler_config,
        "tokenizer": None,
        "tensor_parallel_size": 1,
        "pipeline_parallel_size": 1,
        "moe_expert_parallel_size": None,
        "gpus_per_node": 1,
        "trust_remote_code": False,
        "max_num_tokens": 4096,
        "max_beam_width": 1,
        "kv_cache_config": kv_cache_config,
        "backend": backend,
        "num_postprocess_workers": 0,
        "postprocess_tokenizer_dir": MODEL,
        "reasoning_parser": None,
    }

    torchllm_args = TorchLlmArgs(**llm_args)

    py_executor = create_py_executor(llm_args=torchllm_args,
                                     checkpoint_dir=MODEL_PATH)

    try:
        kv_cache_manager = py_executor.resource_manager.get_resource_manager(
            ResourceManagerType.KV_CACHE_MANAGER)

        assert py_executor.execution_stream is not None, \
            "PyExecutor should have an execution_stream"
        assert kv_cache_manager is not None, \
            "KVCacheManager should exist in resource_manager"
        assert hasattr(kv_cache_manager, '_stream'), \
            "KVCacheManager should have _stream attribute"
        assert py_executor.execution_stream.cuda_stream == kv_cache_manager._stream.cuda_stream, \
            f"PyExecutor.execution_stream ({py_executor.execution_stream.cuda_stream}) " \
            f"should have the same cuda_stream pointer as KVCacheManager._stream ({kv_cache_manager._stream.cuda_stream})"
        assert py_executor.execution_stream is kv_cache_manager._stream, \
            "PyExecutor.execution_stream and KVCacheManager._stream should be the exact same stream object"

        assert len(worker_starts) >= 2, \
            "Both profiling and serving must start a worker phase"
        profiling_manager, profiling_thread = worker_starts[0]
        assert profiling_manager() is not kv_cache_manager, \
            "The first worker phase must use the profiling KV cache"
        assert worker_starts[-1][0]() is kv_cache_manager, \
            "The last worker phase must use the serving KV cache"
        assert profiling_thread is not None
        assert all(thread is profiling_thread for _, thread in worker_starts), \
            "Profiling and serving must use the same worker Thread object"
        assert py_executor.worker_thread is profiling_thread
        assert profiling_thread.is_alive(), \
            "The shared worker must remain alive until serving shuts down"
        if release_before_shutdown:
            py_executor.release_resources()
            assert py_executor.worker_thread is profiling_thread
            assert profiling_thread.is_alive(), \
                "Resource release must park the shared worker until terminal shutdown"
    finally:
        py_executor.shutdown()
        torch.cuda.empty_cache()
    assert not profiling_thread.is_alive(), \
        "Terminal shutdown must stop the shared worker"
