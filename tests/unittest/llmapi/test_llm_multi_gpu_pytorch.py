# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import gc
import os
import time
import weakref

import pytest
from utils.util import skip_ray

from tensorrt_llm import LLM
from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor
from tensorrt_llm._utils import mpi_rank
from tensorrt_llm.executor.rpc_proxy import GenerationExecutorRpcProxy
from tensorrt_llm.llmapi import KvCacheConfig
from tensorrt_llm.llmapi.mpi_session import MpiPoolSession, _process_start_time
from tensorrt_llm.sampling_params import SamplingParams

from .test_llm import (_test_llm_capture_request_error, llama_model_path,
                       llm_get_stats_async_test_harness,
                       llm_get_stats_test_harness,
                       llm_return_logprobs_test_harness,
                       tinyllama_logits_processor_test_harness)

global_kv_cache_config = KvCacheConfig(free_gpu_memory_fraction=0.4)


@pytest.mark.gpu2
def test_llm_capture_request_error():
    _test_llm_capture_request_error(pytorch_backend=True, tp_size=2)


@pytest.mark.gpu4
def test_tinyllama_logits_processor_tp2pp2():
    tinyllama_logits_processor_test_harness(backend="pytorch",
                                            tensor_parallel_size=2,
                                            pipeline_parallel_size=2)


@pytest.mark.gpu2
@pytest.mark.part0
@pytest.mark.parametrize("tp_size, pp_size", [(1, 2), (2, 1)])
def test_tinyllama_logits_processor_2gpu(tp_size: int, pp_size: int):
    tinyllama_logits_processor_test_harness(backend="pytorch",
                                            tensor_parallel_size=tp_size,
                                            pipeline_parallel_size=pp_size)


@skip_ray
@pytest.mark.gpu2
@pytest.mark.private_mpi_session
def test_llm_rpc_tp2() -> None:
    # RPC builds rank 0's engine on another thread. Reusing IPC workers would
    # leave asymmetric thread-local allreduce caches across the ranks.
    with LLM(model=llama_model_path,
             kv_cache_config=KvCacheConfig(free_gpu_memory_fraction=0.4),
             orchestrator_type="rpc",
             tensor_parallel_size=2) as llm:
        assert isinstance(llm._executor, GenerationExecutorRpcProxy)

        res = llm.generate("Tell me a joke",
                           sampling_params=SamplingParams(max_tokens=10,
                                                          end_id=-1))
        print(f"get result: {res}")

        assert len(res.outputs) == 1
        assert len(res.outputs[0].token_ids) == 10


# Each MPI process keeps only weak executor references and the observed Threads.
_rpc_executor_lifetimes = []
_rpc_original_start_worker = None


def _install_rpc_executor_lifetime_probe():
    global _rpc_original_start_worker
    assert not _rpc_executor_lifetimes
    _rpc_original_start_worker = PyExecutor.start_worker

    def record_start(executor):
        _rpc_original_start_worker(executor)
        _rpc_executor_lifetimes.append(
            (weakref.ref(executor), executor.worker_thread))

    PyExecutor.start_worker = record_start
    return mpi_rank(), os.getpid()


def _collect_rpc_executor_lifetimes():
    # Neither this probe nor the next LLM may stop the old executor for it.
    # Reclamation must work while the same MPI worker processes remain alive.
    assert len(_rpc_executor_lifetimes) >= 2
    workers = {worker for _, worker in _rpc_executor_lifetimes}
    assert None not in workers and len(workers) == 1
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        gc.collect()
        for worker in workers:
            worker.join(0.05)
        if all(ref() is None for ref, _ in _rpc_executor_lifetimes) and all(
                not worker.is_alive() for worker in workers):
            break
    assert all(ref() is None for ref, _ in _rpc_executor_lifetimes), \
        f"RPC rank {mpi_rank()} retained its previous executor"
    assert all(not worker.is_alive() for worker in workers), \
        f"RPC rank {mpi_rank()} retained its previous worker thread"
    _rpc_executor_lifetimes.clear()
    return mpi_rank(), os.getpid()


def _restore_rpc_executor_lifetime_probe():
    PyExecutor.start_worker = _rpc_original_start_worker
    return mpi_rank(), os.getpid()


def _submit_rpc_lifetime_probe(session, task):
    futures = session.submit(task)
    assert len(futures) == 2
    return sorted(future.result(timeout=120) for future in futures)


@skip_ray
@pytest.mark.gpu2
@pytest.mark.private_mpi_session
def test_llm_rpc_tp2_releases_executors_in_shared_session(monkeypatch):
    # Opt out of the cross-test IPC pool. Both RPC LLMs below explicitly share
    # this one pool; destroying a private pool per LLM would hide the leak.
    monkeypatch.setenv("TRTLLM_SKIP_KV_CACHE_ESTIMATION", "0")
    session = MpiPoolSession(n_workers=2, wait_shutdown=True)
    try:
        identities = _submit_rpc_lifetime_probe(
            session, _install_rpc_executor_lifetime_probe)
        assert [rank for rank, _ in identities] == [0, 1]
        for _ in range(2):
            with LLM(
                    model=llama_model_path,
                    kv_cache_config=KvCacheConfig(free_gpu_memory_fraction=0.4),
                    orchestrator_type="rpc",
                    tensor_parallel_size=2,
                    _mpi_session=session) as llm:
                assert isinstance(llm._executor, GenerationExecutorRpcProxy)
                assert llm._executor.mpi_session is session
                assert not llm._executor._owns_mpi_session
                result = llm.generate("Tell me a joke",
                                      sampling_params=SamplingParams(
                                          max_tokens=10, end_id=-1))
                assert len(result.outputs) == 1
                assert len(result.outputs[0].token_ids) == 10
            # Check both ranks before constructing the next LLM. Identical PIDs
            # prove that shutdown did not silently replace the worker pool.
            assert _submit_rpc_lifetime_probe(
                session, _collect_rpc_executor_lifetimes) == identities
        _submit_rpc_lifetime_probe(session,
                                   _restore_rpc_executor_lifetime_probe)
        session.shutdown()
        assert len(session._worker_identities) == 2
        assert all(_process_start_time(pid) != start
                   for pid, start in session._worker_identities), \
            "The test-owned RPC pool must exit before other GPU tests run"
    except BaseException:
        try:
            session.abandon()
        except Exception as cleanup_error:
            print(
                f"Failed to abandon RPC pool after test failure: {cleanup_error!r}"
            )
        raise


@skip_ray
@pytest.mark.gpu2
@pytest.mark.asyncio
@pytest.mark.private_mpi_session
async def test_llm_rpc_streaming_tp2() -> None:
    with LLM(model=llama_model_path,
             kv_cache_config=KvCacheConfig(free_gpu_memory_fraction=0.4),
             orchestrator_type="rpc",
             tensor_parallel_size=2) as llm:
        assert isinstance(llm._executor, GenerationExecutorRpcProxy)

        async for output in llm.generate_async("Tell me a joke",
                                               sampling_params=SamplingParams(
                                                   max_tokens=10, end_id=-1)):
            print(f"get result: {output}")


@skip_ray
@pytest.mark.gpu2
@pytest.mark.parametrize(
    "prompt_logprobs, logprobs, return_context_logits, return_generation_logits",
    [
        (None, 1, False,
         False),  # generation logprobs only (top-1, PyTorch limit)
    ])
def test_llm_return_logprobs_streaming_tp2(prompt_logprobs, logprobs,
                                           return_context_logits,
                                           return_generation_logits):
    llm_return_logprobs_test_harness(prompt_logprobs,
                                     logprobs,
                                     return_context_logits,
                                     return_generation_logits,
                                     streaming=True,
                                     backend="pytorch",
                                     tp_size=2)


@skip_ray
@pytest.mark.gpu2
@pytest.mark.parametrize(
    "return_context_logits, enable_chunked_prefill, enable_iter_req_stats",
    [
        (False, False, True),
        (False, True, True),
    ],
)
def test_llm_get_stats_pp2(return_context_logits, enable_chunked_prefill,
                           enable_iter_req_stats):
    llm_get_stats_test_harness(
        tp_size=1,
        pp_size=2,
        return_context_logits=return_context_logits,
        pytorch_backend=True,
        enable_chunked_prefill=enable_chunked_prefill,
        enable_iter_req_stats=enable_iter_req_stats,
    )


@skip_ray
@pytest.mark.gpu4
@pytest.mark.parametrize(
    "return_context_logits, enable_chunked_prefill, enable_iter_req_stats",
    [
        (False, False, True),
        (False, True, True),
    ],
)
def test_llm_get_stats_pp4(return_context_logits, enable_chunked_prefill,
                           enable_iter_req_stats):
    llm_get_stats_test_harness(
        tp_size=1,
        pp_size=4,
        return_context_logits=return_context_logits,
        pytorch_backend=True,
        enable_chunked_prefill=enable_chunked_prefill,
        enable_iter_req_stats=enable_iter_req_stats,
    )


@skip_ray
@pytest.mark.gpu2
def test_llm_get_stats_tp2():
    llm_get_stats_test_harness(tp_size=2, pytorch_backend=True)


@skip_ray
@pytest.mark.gpu2
def test_llm_get_stats_async_tp2():
    llm_get_stats_async_test_harness(tp_size=2, pytorch_backend=True)


@skip_ray
@pytest.mark.gpu2
def test_llm_get_stats_async_pp2():
    llm_get_stats_async_test_harness(pp_size=2, pytorch_backend=True)
