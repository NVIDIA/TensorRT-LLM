# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from contextlib import contextmanager
from queue import Queue
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from utils.llm_data import llm_models_root
from utils.util import get_current_process_gpu_memory

from tensorrt_llm import LLM
from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor
from tensorrt_llm.executor.utils import ErrorResponse
from tensorrt_llm.llmapi import KvCacheConfig, SamplingParams
from tensorrt_llm.llmapi.llm_args import CudaGraphConfig, ExecutorMemoryType, SleepConfig


@pytest.mark.parametrize(
    "sleep_tags,restore_mode,use_v2",
    [
        ([ExecutorMemoryType.KV_CACHE], "NONE", True),
        ([ExecutorMemoryType.KV_CACHE], "MEMSET", True),
        ([ExecutorMemoryType.KV_CACHE], "CPU", True),
        ([ExecutorMemoryType.KV_CACHE], "PINNED", True),
        (list(ExecutorMemoryType), "NONE", True),
        ([ExecutorMemoryType.KV_CACHE], "NONE", False),
    ],
    ids=["v2_none", "v2_memset", "v2_cpu", "v2_pinned", "v2_all_tags", "v1_control"],
)
def test_llm_sleep(process_gpu_memory_info_available, sleep_tags, restore_mode, use_v2):
    llama_model_path = str(llm_models_root() / "Qwen3/Qwen3-0.6B")
    kv_cache_config = KvCacheConfig(
        enable_block_reuse=True, max_tokens=16384, use_kv_cache_manager_v2=use_v2
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

        warm_outputs = llm.generate(prompts, sampling_params)
        assert any(
            output.outputs[0].request_perf_metrics.kv_cache_metrics.num_reused_blocks > 0
            for output in warm_outputs
        )

        memory_usage_active = get_current_process_gpu_memory(True)

        llm._collective_rpc(
            "sleep",
            (sleep_tags,),
        )

        memory_usage_sleep = get_current_process_gpu_memory(True)
        if process_gpu_memory_info_available:
            assert memory_usage_sleep < memory_usage_active

        llm._collective_rpc(
            "wakeup",
            (sleep_tags,),
        )

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
        else:
            assert any(count > 0 for count in reused_blocks)

    for before, after in zip(generated_before_sleep, generated_after_sleep, strict=True):
        assert before == after, "Generated result mismatch before and after sleep"


def test_llm_sleep_discard_weights(process_gpu_memory_info_available):
    """Sleep-wakeup with NONE restore mode for model weights.

    After wakeup the weight memory is re-materialized but the original values
    are gone (NONE = no backup).  The model should still be able to run a
    forward pass without crashing — output correctness is not expected.
    """
    llama_model_path = str(llm_models_root() / "Qwen3/Qwen3-0.6B")
    kv_cache_config = KvCacheConfig(
        enable_block_reuse=False, max_tokens=16384, use_kv_cache_manager_v2=True
    )

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


@pytest.mark.parametrize(
    "method,mutation", [("sleep", "release_with_tag"), ("wakeup", "materialize_with_tag")]
)
@pytest.mark.parametrize("world_size,local_failure", [(1, True), (2, True), (2, False)])
def test_ray_sleep_wakeup_failure_is_terminal(
    method, mutation, world_size, local_failure, monkeypatch
):
    from tensorrt_llm._torch.pyexecutor.executor_request_queue import (
        ExecutorRequestQueue,
        RequestAdmissionState,
    )
    from tensorrt_llm.executor.ray import gpu_worker
    from tensorrt_llm.llmapi.llm_args import ExecutorMemoryType, SleepConfig, TorchLlmArgs

    engine = object.__new__(PyExecutor)
    engine.executor_request_queue = ExecutorRequestQueue(Mock(), 4, False, 0.0)
    tags = [ExecutorMemoryType.KV_CACHE]
    if method == "wakeup":
        engine.executor_request_queue.begin_sleep_transition(tags)
        engine.executor_request_queue.complete_sleep_transition()
    engine._sleeping_memory_tags = {ExecutorMemoryType.KV_CACHE}
    engine._pp_rebalance_drain_iters = 1
    engine.enable_kv_pool_rebalance = True

    @contextmanager
    def control_action():
        expected_state = (
            RequestAdmissionState.WAKING if method == "wakeup" else RequestAdmissionState.PARKING
        )
        assert engine.get_request_admission_state() is expected_state
        yield

    engine.control_action = Mock(side_effect=control_action)
    engine.validate_sleep = Mock()
    engine.prepare_sleep = Mock()
    # finish_wakeup receives tags; model a successful peer publishing full wake.
    engine.finish_wakeup = Mock(side_effect=lambda _: engine._sleeping_memory_tags.clear())
    worker = object.__new__(gpu_worker.RayGPUWorker)
    worker.engine = engine
    worker.llm_args = Mock(spec=TorchLlmArgs, sleep_config=SleepConfig())
    monkeypatch.setattr(gpu_worker, "logger", Mock(), raising=False)
    monkeypatch.setattr(torch.cuda, "synchronize", Mock())
    monkeypatch.setattr(torch.cuda, "empty_cache", Mock())
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: world_size)

    def mutate(*_):
        expected_state = (
            RequestAdmissionState.WAKING if method == "wakeup" else RequestAdmissionState.PARKING
        )
        assert engine.get_request_admission_state() is expected_state
        with pytest.raises(RuntimeError, match="Cannot enqueue"):
            engine.executor_request_queue.enqueue_request(Mock())
        if local_failure:
            raise RuntimeError("injected allocation failure")

    action = Mock(side_effect=mutate)
    monkeypatch.setattr(gpu_worker, mutation, action)

    def agree_failure(failed, op):
        assert failed.device.type == "cpu"
        assert failed.item() == int(local_failure)
        assert op == torch.distributed.ReduceOp.MAX
        failed.fill_(1)

    agreement = Mock(side_effect=agree_failure)
    monkeypatch.setattr(torch.distributed, "all_reduce", agreement)
    message = "injected allocation failure" if local_failure else "failed on another rank"
    with pytest.raises(RuntimeError, match=message):
        getattr(worker, method)(tags)
    assert agreement.call_count == int(world_size > 1)
    assert engine.get_request_admission_state() is RequestAdmissionState.FAILED
    assert not engine._can_pause_for_rebalance()
    assert engine._pp_rebalance_drain_iters is None
    if method == "wakeup" and local_failure:
        engine.finish_wakeup.assert_not_called()
    with pytest.raises(RuntimeError, match="failed"):
        engine.executor_request_queue.enqueue_request(Mock())
    for retry in (worker.sleep, worker.wakeup):
        with pytest.raises(RuntimeError, match="previously failed"):
            retry(tags)
    engine.control_action.assert_called_once_with()
    action.assert_called_once_with(*tags)


@pytest.mark.parametrize("entrypoint", ["mpi", "ray"])
@pytest.mark.parametrize(
    "tag,use_v2,has_transceiver,rejected",
    [
        ("kv_cache", True, True, True),
        ("executor_extra", True, True, True),
        ("executor_extra", False, True, True),
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
    tags = [ExecutorMemoryType(tag)]
    if entrypoint == "mpi":
        run = stub.begin_sleep_transition
        mutation = stub.executor_request_queue.begin_sleep_transition
    else:
        worker = object.__new__(RayGPUWorker)
        worker.engine = stub
        worker.llm_args = Mock(spec=TorchLlmArgs, sleep_config=SleepConfig())
        worker._sleep = Mock()
        run = worker.sleep
        mutation = worker._sleep

    if rejected:
        with pytest.raises(NotImplementedError, match="remote memory re-registration"):
            run(tags)
        mutation.assert_not_called()
    else:
        run(tags)
        mutation.assert_called_once()


@pytest.mark.parametrize("use_ray", [False, True])
def test_submit_registers_result_before_immediate_rejection(use_ray, monkeypatch):
    from tensorrt_llm.executor import rpc_proxy_mixin
    from tensorrt_llm.executor.ray import executor as ray_executor

    module = ray_executor if use_ray else rpc_proxy_mixin
    submit = ray_executor.RayExecutor.submit if use_ray else rpc_proxy_mixin.RpcExecutorMixin.submit
    result = SimpleNamespace(queue=Queue())
    monkeypatch.setattr(module, "GenerationResult", Mock(return_value=result))
    proxy = SimpleNamespace(
        _results={},
        _get_next_client_id=lambda: 42,
        _get_logprob_params=lambda _: None,
        _handle_background_error=Mock(),
        rpc_client=Mock(),
    )
    request = SimpleNamespace(id=None, disaggregated_params=None)
    request.set_id = lambda client_id: setattr(request, "id", client_id)
    error = ErrorResponse(42, "Cannot enqueue requests while executor admission is parked", 42)

    def reject(*, need_response):
        assert not need_response
        rpc_proxy_mixin.RpcExecutorMixin.handle_responses(proxy, [error])

    proxy.rpc_client.submit.return_value.remote.side_effect = reject
    assert submit(proxy, request) is result
    assert result.queue.get_nowait() == error
    assert not proxy._results
