# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import gc
import multiprocessing
import os
import sys
import time
import uuid
import weakref
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

# Force a deterministic UCX/NIXL config regardless of what the cluster/CI
# injects; see disaggregated/test_kv_transfer.py for the full rationale.
os.environ["UCX_TLS"] = "^ib,gdr_copy"
os.environ["TRTLLM_NIXL_NUM_THREADS"] = "1"

import tensorrt_llm
import tensorrt_llm.bindings
import tensorrt_llm.bindings.executor as trtllm
from tensorrt_llm._torch.disaggregation import \
    kv_cache_transceiver as transceiver_module
from tensorrt_llm._torch.disaggregation.kv_cache_transceiver import \
    create_kv_cache_transceiver
from tensorrt_llm._torch.distributed import Distributed
from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import \
    KVCacheManagerV2
from tensorrt_llm._torch.pyexecutor.kv_cache.mamba_cache_manager import (
    MambaHybridCacheManagerV2, MixedMambaHybridCacheManager)
from tensorrt_llm._torch.pyexecutor.llm_request import (LlmRequest,
                                                        LlmRequestState)
from tensorrt_llm._torch.pyexecutor.resource_manager import KVCacheManager
from tensorrt_llm.llmapi.llm_args import CacheTransceiverConfig
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.sampling_params import SamplingParams

AttentionTypeCpp = tensorrt_llm.bindings.internal.batch_manager.AttentionType
LlmRequestType = tensorrt_llm.bindings.internal.batch_manager.LlmRequestType
DataType = tensorrt_llm.bindings.DataType
DEFAULT_KV_TRANSFER_TIMEOUT_S = (
    CacheTransceiverConfig.model_fields["kv_transfer_timeout_ms"].default /
    1000.0)
KV_TRANSFER_COMPLETION_MARGIN_S = 10.0


@pytest.mark.parametrize("transceiver_runtime", ["CPP", "auto"])
def test_cpp_transceiver_rejects_mixed_mamba_manager(transceiver_runtime):
    config = CacheTransceiverConfig(backend="NIXL",
                                    transceiver_runtime=transceiver_runtime)
    mixed_manager = object.__new__(MixedMambaHybridCacheManager)

    with pytest.raises(
            ValueError,
            match="MixedMambaHybridCacheManager requires the Python"):
        create_kv_cache_transceiver(mapping=None,
                                    dist=None,
                                    kv_cache_manager=None,
                                    attention_type=AttentionTypeCpp.DEFAULT,
                                    cache_transceiver_config=config,
                                    mamba_cache_manager=mixed_manager)


@pytest.mark.cpu_only
@pytest.mark.parametrize("runtime", [None, "CPP", "auto"])
@pytest.mark.parametrize("backend", ["NIXL", "UCX"])
@pytest.mark.parametrize("manager_cls",
                         [KVCacheManagerV2, MambaHybridCacheManagerV2])
def test_cpp_transceiver_rejects_v2_manager(monkeypatch, runtime, backend,
                                            manager_cls) -> None:
    config = CacheTransceiverConfig(backend=backend,
                                    transceiver_runtime=runtime)
    manager = object.__new__(manager_cls)
    constructor = Mock()
    monkeypatch.setattr(transceiver_module, "BindKvCacheTransceiver",
                        constructor)
    monkeypatch.delenv("TRTLLM_DISAGG_ENABLE_INFLIGHT_CANCEL", raising=False)
    monkeypatch.setattr(transceiver_module,
                        "_disagg_inflight_cancel_enabled_cache", None)

    with pytest.raises(ValueError,
                       match=r"kv_cache_config.use_kv_cache_manager_v2=False"):
        create_kv_cache_transceiver(
            mapping=None,
            dist=None,
            kv_cache_manager=manager,
            attention_type=AttentionTypeCpp.DEFAULT,
            cache_transceiver_config=config,
            mamba_cache_manager=(manager if manager_cls
                                 is MambaHybridCacheManagerV2 else None))

    constructor.assert_not_called()


@pytest.mark.cpu_only
@pytest.mark.parametrize("runtime,manager_cls", [
    ("CPP", KVCacheManager),
    (None, KVCacheManager),
    ("PYTHON", KVCacheManagerV2),
])
def test_transceiver_accepts_compatible_manager(monkeypatch, runtime,
                                                manager_cls) -> None:
    config = CacheTransceiverConfig(backend="NIXL", transceiver_runtime=runtime)
    manager = object.__new__(manager_cls)
    cpp_constructor = Mock()
    python_constructor = Mock()
    monkeypatch.setattr(transceiver_module, "BindKvCacheTransceiver",
                        cpp_constructor)
    monkeypatch.setitem(
        sys.modules, "tensorrt_llm._torch.disaggregation.transceiver",
        SimpleNamespace(KvCacheTransceiverV2=python_constructor))
    monkeypatch.delenv("TRTLLM_DISAGG_ENABLE_INFLIGHT_CANCEL", raising=False)
    monkeypatch.setattr(transceiver_module,
                        "_disagg_inflight_cancel_enabled_cache", None)

    result = create_kv_cache_transceiver(
        mapping=None,
        dist=None,
        kv_cache_manager=manager,
        attention_type=AttentionTypeCpp.DEFAULT,
        cache_transceiver_config=config)

    expected = python_constructor if runtime == "PYTHON" else cpp_constructor
    unused = cpp_constructor if runtime == "PYTHON" else python_constructor
    assert result is expected.return_value
    expected.assert_called_once()
    unused.assert_not_called()


@pytest.mark.cpu_only
@pytest.mark.parametrize("backend", ["UCX", "MPI", "MOONCAKE"])
def test_python_transceiver_backend_error_explains_v1(backend) -> None:
    config = CacheTransceiverConfig(backend=backend)

    with pytest.raises(
            ValueError,
            match=r"transceiver_runtime='CPP'.*use_kv_cache_manager_v2=False"):
        create_kv_cache_transceiver(mapping=None,
                                    dist=None,
                                    kv_cache_manager=None,
                                    attention_type=AttentionTypeCpp.DEFAULT,
                                    cache_transceiver_config=config)


def create_kv_cache_manager(mapping,
                            dtype,
                            max_tokens=256,
                            max_seq_len=256,
                            max_batch_size=1):
    return KVCacheManager(
        trtllm.KvCacheConfig(
            max_tokens=max_tokens,
            enable_block_reuse=False,
        ),
        tensorrt_llm.bindings.internal.batch_manager.CacheType.SELF,
        num_layers=1,
        num_kv_heads=1,
        head_dim=1,
        tokens_per_block=8,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
        mapping=mapping,
        dtype=dtype)


def fill_kv_cache_buffer(kv_cache_manager):
    with torch.no_grad():
        buffer = kv_cache_manager.get_buffers(0)
        random_values = torch.rand(buffer.shape,
                                   dtype=torch.float32,
                                   device=buffer.device)
        buffer.copy_(random_values)


def wait_for_transfer_completion(poll_fn,
                                 is_done_fn,
                                 timeout_s=DEFAULT_KV_TRANSFER_TIMEOUT_S +
                                 KV_TRANSFER_COMPLETION_MARGIN_S):
    """Poll until KV cache transfer completes or timeout expires.

    Args:
        poll_fn: Callable invoked each loop to drive transfer progress.
        is_done_fn: Callable returning True when transfer is complete.
        timeout_s: Seconds to wait before asserting failure.
    """
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        poll_fn()
        if is_done_fn():
            return
        time.sleep(0.01)
    poll_fn()
    assert is_done_fn(), "Timed out waiting for KV cache transfer completion"


def shutdown_transceivers(*transceivers):
    for transceiver in transceivers:
        transceiver.shutdown()


def get_context_completed_request_id(request, transceiver_runtime):
    if transceiver_runtime == "PYTHON":
        assert request.py_disaggregated_params is not None
        disagg_request_id = request.py_disaggregated_params.disagg_request_id
        assert disagg_request_id is not None
        return disagg_request_id
    return request.py_request_id


@pytest.fixture(scope="function")
def ctx_gen_kv_cache_dtype(request):
    if request.param == "ctx_fp8_gen_fp8":
        return DataType.FP8, DataType.FP8
    elif request.param == "ctx_fp16_gen_fp16":
        return DataType.HALF, DataType.HALF
    elif request.param == "ctx_bf16_gen_bf16":
        return DataType.BF16, DataType.BF16
    else:
        raise ValueError(f"Invalid config: {request.param}")


@pytest.mark.timeout(120)
@pytest.mark.parametrize(
    "ctx_gen_kv_cache_dtype",
    ["ctx_fp8_gen_fp8", "ctx_fp16_gen_fp16", "ctx_bf16_gen_bf16"],
    ids=["ctx_fp8_gen_fp8", "ctx_fp16_gen_fp16", "ctx_bf16_gen_bf16"],
    indirect=True)
@pytest.mark.parametrize("attention_type",
                         [AttentionTypeCpp.DEFAULT, AttentionTypeCpp.MLA],
                         ids=["mha", "mla"])
@pytest.mark.parametrize("backend_runtime", [("NIXL", None), ("UCX", None)],
                         ids=["NIXL", "UCX"])
def test_kv_cache_transceiver_single_process(ctx_gen_kv_cache_dtype,
                                             attention_type, backend_runtime):
    backend, transceiver_runtime = backend_runtime
    tensorrt_llm.logger.set_level("info")
    # Init kv_cache manager and cache transceiver
    mapping = Mapping(world_size=1, rank=0)
    ctx_kv_cache_dtype, gen_kv_cache_dtype = ctx_gen_kv_cache_dtype
    kv_cache_manager_ctx = create_kv_cache_manager(mapping, ctx_kv_cache_dtype)
    kv_cache_manager_gen = create_kv_cache_manager(mapping, gen_kv_cache_dtype)

    cache_transceiver_config = CacheTransceiverConfig(
        backend=backend,
        transceiver_runtime=transceiver_runtime,
        max_tokens_in_buffer=512)
    dist = Distributed.get(mapping)
    kv_cache_transceiver_ctx = create_kv_cache_transceiver(
        mapping, dist, kv_cache_manager_ctx, attention_type,
        cache_transceiver_config)

    kv_cache_transceiver_gen = create_kv_cache_transceiver(
        mapping, dist, kv_cache_manager_gen, attention_type,
        cache_transceiver_config)

    try:
        fill_kv_cache_buffer(kv_cache_manager_ctx)

        # init ctx request
        sampling_params = SamplingParams()
        ctx_request = LlmRequest(
            request_id=0,
            max_new_tokens=1,
            input_tokens=list(range(256)),
            sampling_config=tensorrt_llm.bindings.SamplingConfig(
                sampling_params._get_sampling_config()),
            is_streaming=False,
            llm_request_type=LlmRequestType.LLMREQUEST_TYPE_CONTEXT_ONLY)

        if transceiver_runtime == "PYTHON":
            disaggregated_params = tensorrt_llm.DisaggregatedParams(
                request_type="context_only",
                disagg_request_id=uuid.uuid4().int & 0x7FFFFFFFFFFFFFFF)
            ctx_request.py_disaggregated_params = disaggregated_params

        kv_cache_manager_ctx.impl.add_sequence_batch(
            [(ctx_request.py_request_id, ctx_request.prompt_len, 1)],
            [ctx_request])
        # send ctx request
        kv_cache_transceiver_ctx.respond_and_send_async(ctx_request)

        # init gen request
        gen_request = LlmRequest(
            request_id=0,
            max_new_tokens=1,
            input_tokens=list(range(256)),
            sampling_config=tensorrt_llm.bindings.SamplingConfig(
                sampling_params._get_sampling_config()),
            is_streaming=False,
            llm_request_type=LlmRequestType.LLMREQUEST_TYPE_GENERATION_ONLY,
            context_phase_params=ctx_request.context_phase_params)

        if transceiver_runtime == "PYTHON":
            disaggregated_params = tensorrt_llm.DisaggregatedParams(
                request_type="generation_only",
                disagg_request_id=ctx_request.py_disaggregated_params.
                disagg_request_id,
                ctx_request_id=ctx_request.context_phase_params.req_id,
                ctx_dp_rank=ctx_request.context_phase_params.ctx_dp_rank,
                ctx_info_endpoint=ctx_request.context_phase_params.
                disagg_info_endpoint,
                first_gen_tokens=ctx_request.context_phase_params.
                first_gen_tokens,
                draft_tokens=ctx_request.context_phase_params.draft_tokens)

            gen_request.py_disaggregated_params = disaggregated_params

        kv_cache_manager_gen.impl.add_sequence_batch(
            [(gen_request.py_request_id, gen_request.prompt_len, 1)],
            [gen_request])
        # send gen request
        kv_cache_transceiver_gen.request_and_receive_async(gen_request)

        completed_ctx_ids = set()

        def poll_transfers():
            completed, failed = (
                kv_cache_transceiver_ctx.check_context_transfer_status(1))
            assert failed == []
            completed_ctx_ids.update(completed)
            kv_cache_transceiver_gen.check_gen_transfer_status(1)

        expected_ctx_id = get_context_completed_request_id(
            ctx_request, transceiver_runtime)

        def transfers_done():
            return (expected_ctx_id in completed_ctx_ids and gen_request.state
                    == LlmRequestState.DISAGG_GENERATION_TRANS_COMPLETE)

        wait_for_transfer_completion(poll_transfers, transfers_done)

        assert torch.equal(
            kv_cache_manager_gen.get_buffers(0),
            kv_cache_manager_ctx.get_buffers(0)), "different kv-cache values"
    finally:
        shutdown_transceivers(kv_cache_transceiver_gen,
                              kv_cache_transceiver_ctx)


def _run_cpp_nixl_sync_transfer_stress():
    """Exercise repeated empty-to-nonempty sender transitions in a child."""
    request_count = 64
    prompt_len = 16
    mapping = Mapping(world_size=1, rank=0)
    dist = Distributed.get(mapping)
    manager_kwargs = {
        "max_tokens": request_count * prompt_len * 2,
        # This test validates transfer only, so no decode capacity is needed.
        "max_seq_len": prompt_len,
        "max_batch_size": request_count,
    }
    kv_cache_manager_ctx = create_kv_cache_manager(mapping, DataType.HALF,
                                                   **manager_kwargs)
    kv_cache_manager_gen = create_kv_cache_manager(mapping, DataType.HALF,
                                                   **manager_kwargs)
    cache_transceiver_config = CacheTransceiverConfig(backend="NIXL",
                                                      transceiver_runtime="CPP",
                                                      max_tokens_in_buffer=512)
    transceiver_ctx = create_kv_cache_transceiver(mapping, dist,
                                                  kv_cache_manager_ctx,
                                                  AttentionTypeCpp.DEFAULT,
                                                  cache_transceiver_config)
    transceiver_gen = create_kv_cache_transceiver(mapping, dist,
                                                  kv_cache_manager_gen,
                                                  AttentionTypeCpp.DEFAULT,
                                                  cache_transceiver_config)

    try:
        sampling_config = tensorrt_llm.bindings.SamplingConfig(
            SamplingParams()._get_sampling_config())
        ctx_requests = [
            LlmRequest(
                request_id=request_id,
                max_new_tokens=1,
                input_tokens=list(range(prompt_len)),
                sampling_config=sampling_config,
                is_streaming=False,
                llm_request_type=LlmRequestType.LLMREQUEST_TYPE_CONTEXT_ONLY)
            for request_id in range(request_count)
        ]
        kv_cache_manager_ctx.impl.add_sequence_batch(
            [(request.py_request_id, request.prompt_len, 1)
             for request in ctx_requests], ctx_requests)
        fill_kv_cache_buffer(kv_cache_manager_ctx)

        transceiver_ctx.respond_and_send_async(ctx_requests[0])
        completed_ctx_ids = set()

        def poll_context_transfers():
            completed, failed = transceiver_ctx.check_context_transfer_status(1)
            assert failed == [], f"context transfers failed: {failed}"
            completed_ctx_ids.update(completed)

        for request_index, ctx_request in enumerate(ctx_requests):
            gen_request = LlmRequest(
                request_id=request_index,
                max_new_tokens=1,
                input_tokens=list(range(prompt_len)),
                sampling_config=sampling_config,
                is_streaming=False,
                llm_request_type=LlmRequestType.LLMREQUEST_TYPE_GENERATION_ONLY,
                context_phase_params=ctx_request.context_phase_params)
            kv_cache_manager_gen.impl.add_sequence_batch(
                [(gen_request.py_request_id, gen_request.prompt_len, 1)],
                [gen_request])

            transceiver_gen.request_and_receive_sync(gen_request)

            # Queue the next response immediately so its insertion can overlap
            # the previous response's sender-side cleanup.
            if request_index + 1 < request_count:
                transceiver_ctx.respond_and_send_async(
                    ctx_requests[request_index + 1])

            assert (gen_request.state ==
                    LlmRequestState.DISAGG_GENERATION_TRANS_COMPLETE)

            ctx_block_ids = kv_cache_manager_ctx.get_cache_indices(ctx_request)
            gen_block_ids = kv_cache_manager_gen.get_cache_indices(gen_request)
            assert torch.equal(
                kv_cache_manager_ctx.get_unique_primary_pool()[ctx_block_ids],
                kv_cache_manager_gen.get_unique_primary_pool()[gen_block_ids],
            ), f"different KV-cache values for request {request_index}"

            expected_ctx_id = ctx_request.py_request_id
            wait_for_transfer_completion(poll_context_transfers,
                                         lambda request_id=expected_ctx_id:
                                         request_id in completed_ctx_ids)
    finally:
        shutdown_transceivers(transceiver_gen, transceiver_ctx)


@pytest.mark.timeout(150)
def test_cpp_nixl_sync_transfer_stress():
    """C++ NIXL sync transfers must not lose sender wakeups between requests."""
    process = multiprocessing.get_context("spawn").Process(
        target=_run_cpp_nixl_sync_transfer_stress,
        name="cpp-nixl-sync-transfer-stress")
    process.start()
    process.join(timeout=120)
    try:
        if process.is_alive():
            process.terminate()
            process.join(timeout=5)
            if process.is_alive():
                process.kill()
                process.join(timeout=5)
            pytest.fail("C++ NIXL synchronous KV transfer stress test hung")
        assert process.exitcode == 0, (
            f"C++ NIXL synchronous KV transfer child exited with "
            f"code {process.exitcode}")
    finally:
        process.close()


@pytest.mark.timeout(120)
@pytest.mark.parametrize("attention_type",
                         [AttentionTypeCpp.DEFAULT, AttentionTypeCpp.MLA],
                         ids=["mha", "mla"])
def test_cancel_request_in_transmission(attention_type):
    # Init kv_cache manager and cache transceiver
    mapping = Mapping(world_size=1, rank=0)
    dist = Distributed.get(mapping)
    ctx_kv_cache_dtype, gen_kv_cache_dtype = DataType.HALF, DataType.HALF
    kv_cache_manager_ctx = create_kv_cache_manager(mapping, ctx_kv_cache_dtype)
    kv_cache_manager_gen = create_kv_cache_manager(mapping, gen_kv_cache_dtype)

    cache_transceiver_config = CacheTransceiverConfig(backend="DEFAULT",
                                                      transceiver_runtime="CPP",
                                                      max_tokens_in_buffer=512)

    kv_cache_transceiver_ctx = create_kv_cache_transceiver(
        mapping, dist, kv_cache_manager_ctx, attention_type,
        cache_transceiver_config)

    kv_cache_transceiver_gen = create_kv_cache_transceiver(
        mapping, dist, kv_cache_manager_gen, attention_type,
        cache_transceiver_config)

    fill_kv_cache_buffer(kv_cache_manager_ctx)

    # init ctx request
    sampling_params = SamplingParams()
    ctx_request = LlmRequest(
        request_id=0,
        max_new_tokens=1,
        input_tokens=list(range(256)),
        sampling_config=tensorrt_llm.bindings.SamplingConfig(
            sampling_params._get_sampling_config()),
        is_streaming=False,
        llm_request_type=LlmRequestType.LLMREQUEST_TYPE_CONTEXT_ONLY)

    kv_cache_manager_ctx.impl.add_sequence_batch(
        [(ctx_request.py_request_id, ctx_request.prompt_len, 1)], [ctx_request])
    # send ctx request
    kv_cache_transceiver_ctx.respond_and_send_async(ctx_request)

    # wait for ctx request to be sent
    time.sleep(2)

    # cancel ctx request
    is_cancelled = kv_cache_transceiver_ctx.cancel_request(ctx_request)
    assert is_cancelled

    # init gen request
    gen_request = LlmRequest(
        request_id=0,
        max_new_tokens=1,
        input_tokens=list(range(256)),
        sampling_config=tensorrt_llm.bindings.SamplingConfig(
            sampling_params._get_sampling_config()),
        is_streaming=False,
        llm_request_type=LlmRequestType.LLMREQUEST_TYPE_GENERATION_ONLY,
        context_phase_params=ctx_request.context_phase_params)

    kv_cache_manager_gen.impl.add_sequence_batch(
        [(gen_request.py_request_id, gen_request.prompt_len, 1)], [gen_request])
    # send gen request
    kv_cache_transceiver_gen.request_and_receive_async(gen_request)

    # Block the main thread due to the async operation
    time.sleep(2)
    kv_cache_transceiver_gen.check_gen_transfer_status(0)
    assert gen_request.state == LlmRequestState.DISAGG_TRANS_ERROR


@pytest.mark.timeout(120)
def test_async_transfer_keeps_llm_request_alive():
    """Async entry points must hold a strong shared_ptr to the LlmRequest.

    A regression to raw-pointer storage in mSenderFutures /
    mRequesterFutures lets Python GC the wrapper while a C++ status check
    still dereferences it.
    """
    mapping = Mapping(world_size=1, rank=0)
    dist = Distributed.get(mapping)
    kv_cache_manager_ctx = create_kv_cache_manager(mapping, DataType.HALF)
    kv_cache_manager_gen = create_kv_cache_manager(mapping, DataType.HALF)

    cache_transceiver_config = CacheTransceiverConfig(backend="DEFAULT",
                                                      transceiver_runtime="CPP",
                                                      max_tokens_in_buffer=512)
    transceiver_ctx = create_kv_cache_transceiver(mapping, dist,
                                                  kv_cache_manager_ctx,
                                                  AttentionTypeCpp.DEFAULT,
                                                  cache_transceiver_config)
    transceiver_gen = create_kv_cache_transceiver(mapping, dist,
                                                  kv_cache_manager_gen,
                                                  AttentionTypeCpp.DEFAULT,
                                                  cache_transceiver_config)

    fill_kv_cache_buffer(kv_cache_manager_ctx)

    sampling_params = SamplingParams()
    ctx_request = LlmRequest(
        request_id=0,
        max_new_tokens=1,
        input_tokens=list(range(256)),
        sampling_config=tensorrt_llm.bindings.SamplingConfig(
            sampling_params._get_sampling_config()),
        is_streaming=False,
        llm_request_type=LlmRequestType.LLMREQUEST_TYPE_CONTEXT_ONLY)
    kv_cache_manager_ctx.impl.add_sequence_batch(
        [(ctx_request.py_request_id, ctx_request.prompt_len, 1)], [ctx_request])

    # Snapshot ctx refcount *before* submission. add_sequence_batch takes
    # std::reference_wrapper<LlmRequest> (no Python ref retained), so the
    # baseline here is clean and the +1 below isolates exactly the
    # shared_ptr captured by respond_and_send_async.
    ctx_ref = weakref.ref(ctx_request)
    baseline_ctx_refcount = sys.getrefcount(ctx_request)

    # respond_and_send_async also populates ctx_request.context_phase_params
    # as a side effect — gen_request below requires this to be non-empty.
    transceiver_ctx.respond_and_send_async(ctx_request)
    assert sys.getrefcount(ctx_request) == baseline_ctx_refcount + 1, (
        f"respond_and_send_async did not capture a shared_ptr<LlmRequest>; "
        f"refcount={sys.getrefcount(ctx_request)} "
        f"(expected baseline {baseline_ctx_refcount} + 1)")

    gen_request = LlmRequest(
        request_id=0,
        max_new_tokens=1,
        input_tokens=list(range(256)),
        sampling_config=tensorrt_llm.bindings.SamplingConfig(
            sampling_params._get_sampling_config()),
        is_streaming=False,
        llm_request_type=LlmRequestType.LLMREQUEST_TYPE_GENERATION_ONLY,
        context_phase_params=ctx_request.context_phase_params)
    kv_cache_manager_gen.impl.add_sequence_batch(
        [(gen_request.py_request_id, gen_request.prompt_len, 1)], [gen_request])

    gen_ref = weakref.ref(gen_request)
    baseline_gen_refcount = sys.getrefcount(gen_request)

    transceiver_gen.request_and_receive_async(gen_request)
    assert sys.getrefcount(gen_request) == baseline_gen_refcount + 1, (
        f"request_and_receive_async did not capture a shared_ptr<LlmRequest>; "
        f"refcount={sys.getrefcount(gen_request)} "
        f"(expected baseline {baseline_gen_refcount} + 1)")

    # Drop the external Python refs. After this, the only owners are the
    # nanobind-managed shared_ptr entries inside mSenderFutures /
    # mRequesterFutures.
    del ctx_request
    del gen_request
    gc.collect()

    assert ctx_ref() is not None, (
        "Sender-side LlmRequest was destroyed prematurely; "
        "respond_and_send_async must hold a strong shared_ptr while the "
        "sender future is in flight")
    assert gen_ref() is not None, (
        "Receiver-side LlmRequest was destroyed prematurely; "
        "request_and_receive_async must hold a strong shared_ptr while "
        "the requester future is in flight")

    # Re-acquire temporary strong refs for cleanup after proving the in-flight
    # futures owned the requests. Once a future is erased, the weakref may
    # legitimately clear.
    ctx_request = ctx_ref()
    gen_request = gen_ref()
    assert ctx_request is not None
    assert gen_request is not None

    # Drive the transfer to completion so the harness tears down cleanly.
    completed_ctx_ids = set()

    def poll_transfers():
        completed, failed = transceiver_ctx.check_context_transfer_status(1)
        assert failed == []
        completed_ctx_ids.update(completed)
        transceiver_gen.check_gen_transfer_status(1)

    def transfers_done():
        return (ctx_request.py_request_id in completed_ctx_ids
                and gen_request.state
                == LlmRequestState.DISAGG_GENERATION_TRANS_COMPLETE)

    wait_for_transfer_completion(poll_transfers, transfers_done)


def _build_ctx_request_for_timeout_test(request_id: int) -> LlmRequest:
    sampling_params = SamplingParams()
    return LlmRequest(
        request_id=request_id,
        max_new_tokens=1,
        input_tokens=list(range(256)),
        sampling_config=tensorrt_llm.bindings.SamplingConfig(
            sampling_params._get_sampling_config()),
        is_streaming=False,
        llm_request_type=LlmRequestType.LLMREQUEST_TYPE_CONTEXT_ONLY)


@pytest.mark.timeout(60)
def test_kv_transfer_timeout_warns_once_per_request(capfd):
    """Observe-only timeout WARN must fire exactly once per stuck request.

    checkContextTransferStatus emits a TLLM_LOG_WARNING when elapsed time
    exceeds kv_transfer_timeout_ms; mTimedOutSenderIds dedup suppresses
    repeat emissions across subsequent polls of the same in-flight future.
    """
    mapping = Mapping(world_size=1, rank=0)
    dist = Distributed.get(mapping)
    kv_cache_manager_ctx = create_kv_cache_manager(mapping, DataType.HALF)

    cache_transceiver_config = CacheTransceiverConfig(
        backend="DEFAULT",
        transceiver_runtime="CPP",
        max_tokens_in_buffer=512,
        kv_transfer_timeout_ms=100)
    transceiver_ctx = create_kv_cache_transceiver(mapping, dist,
                                                  kv_cache_manager_ctx,
                                                  AttentionTypeCpp.DEFAULT,
                                                  cache_transceiver_config)

    fill_kv_cache_buffer(kv_cache_manager_ctx)

    ctx_request = _build_ctx_request_for_timeout_test(request_id=42)
    kv_cache_manager_ctx.impl.add_sequence_batch(
        [(ctx_request.py_request_id, ctx_request.prompt_len, 1)], [ctx_request])

    capfd.readouterr()  # drain prior noise

    transceiver_ctx.respond_and_send_async(ctx_request)
    time.sleep(0.3)  # > kv_transfer_timeout_ms

    transceiver_ctx.check_context_transfer_status(0)
    first = capfd.readouterr()
    marker = "Context KV cache transfer for request 42 exceeded configured timeout"
    # TLLM_LOG_WARNING writes to stdout; check first.out (not first.err).
    assert marker in first.out, (
        f"Expected observe-only WARN after timeout; stdout:\n{first.out}")
    assert first.out.count(marker) == 1, (
        f"WARN should fire exactly once per poll for a single stuck request; "
        f"got {first.out.count(marker)} emissions:\n{first.out}")

    transceiver_ctx.check_context_transfer_status(0)
    second = capfd.readouterr()
    assert marker not in second.out, (
        f"WARN re-emitted on a subsequent poll; dedup broken. "
        f"stdout:\n{second.out}")

    transceiver_ctx.cancel_request(ctx_request)


@pytest.mark.timeout(60)
def test_kv_transfer_timeout_silent_when_unset(capfd):
    """Without kv_transfer_timeout_ms the observe-only WARN must stay silent.

    The elapsed-time check is gated by the optional config field; absence
    of the field must short-circuit the WARN path even on a long-running
    transfer.
    """
    mapping = Mapping(world_size=1, rank=0)
    dist = Distributed.get(mapping)
    kv_cache_manager_ctx = create_kv_cache_manager(mapping, DataType.HALF)

    cache_transceiver_config = CacheTransceiverConfig(backend="DEFAULT",
                                                      transceiver_runtime="CPP",
                                                      max_tokens_in_buffer=512)
    transceiver_ctx = create_kv_cache_transceiver(mapping, dist,
                                                  kv_cache_manager_ctx,
                                                  AttentionTypeCpp.DEFAULT,
                                                  cache_transceiver_config)

    fill_kv_cache_buffer(kv_cache_manager_ctx)

    ctx_request = _build_ctx_request_for_timeout_test(request_id=99)
    kv_cache_manager_ctx.impl.add_sequence_batch(
        [(ctx_request.py_request_id, ctx_request.prompt_len, 1)], [ctx_request])

    capfd.readouterr()

    transceiver_ctx.respond_and_send_async(ctx_request)
    time.sleep(0.3)
    transceiver_ctx.check_context_transfer_status(0)

    out = capfd.readouterr()
    # TLLM_LOG_WARNING writes to stdout; check out.out (not out.err).
    assert "exceeded configured timeout" not in out.out, (
        f"Observe-only WARN must not fire when kv_transfer_timeout_ms is "
        f"unset; stdout:\n{out.out}")

    transceiver_ctx.cancel_request(ctx_request)


@pytest.mark.timeout(120)
def test_context_transfer_bounded_poll_keeps_request_in_progress(capfd):
    """A bounded transfer poll must not make a non-ready request terminal."""
    mapping = Mapping(world_size=1, rank=0)
    dist = Distributed.get(mapping)
    kv_cache_manager_ctx = create_kv_cache_manager(mapping, DataType.HALF)
    kv_cache_manager_gen = create_kv_cache_manager(mapping, DataType.HALF)

    cache_transceiver_config = CacheTransceiverConfig(
        backend="DEFAULT",
        transceiver_runtime="CPP",
        max_tokens_in_buffer=512,
        kv_transfer_timeout_ms=100,
        kv_transfer_sender_future_timeout_ms=10)
    transceiver_ctx = create_kv_cache_transceiver(mapping, dist,
                                                  kv_cache_manager_ctx,
                                                  AttentionTypeCpp.DEFAULT,
                                                  cache_transceiver_config)
    transceiver_gen = create_kv_cache_transceiver(mapping, dist,
                                                  kv_cache_manager_gen,
                                                  AttentionTypeCpp.DEFAULT,
                                                  cache_transceiver_config)

    fill_kv_cache_buffer(kv_cache_manager_ctx)

    ctx_request = _build_ctx_request_for_timeout_test(request_id=100)
    kv_cache_manager_ctx.impl.add_sequence_batch(
        [(ctx_request.py_request_id, ctx_request.prompt_len, 1)], [ctx_request])

    transceiver_ctx.respond_and_send_async(ctx_request)

    start = time.monotonic()
    completed_request_ids, error_request_ids = (
        transceiver_ctx.check_context_transfer_status(1))
    elapsed = time.monotonic() - start

    assert elapsed < 1.0, (
        f"Bounded poll should yield back to the executor quickly; "
        f"elapsed={elapsed:.3f}s")
    assert completed_request_ids == []
    assert error_request_ids == []
    assert ctx_request.state == LlmRequestState.DISAGG_CONTEXT_TRANS_IN_PROGRESS

    sampling_params = SamplingParams()
    gen_request = LlmRequest(
        request_id=100,
        max_new_tokens=1,
        input_tokens=list(range(256)),
        sampling_config=tensorrt_llm.bindings.SamplingConfig(
            sampling_params._get_sampling_config()),
        is_streaming=False,
        llm_request_type=LlmRequestType.LLMREQUEST_TYPE_GENERATION_ONLY,
        context_phase_params=ctx_request.context_phase_params)

    kv_cache_manager_gen.impl.add_sequence_batch(
        [(gen_request.py_request_id, gen_request.prompt_len, 1)], [gen_request])

    capfd.readouterr()
    transceiver_gen.request_and_receive_async(gen_request)
    transceiver_gen.check_gen_transfer_status(0)
    out = capfd.readouterr()
    marker = "Generation KV cache transfer for request 100 exceeded configured timeout"
    assert marker not in out.out, (
        f"Generation transfer timeout WARN fired before the request had "
        f"actually timed out; stdout:\n{out.out}")

    completed_ctx_ids = set()

    def poll_transfers():
        completed, failed = transceiver_ctx.check_context_transfer_status(1)
        assert failed == []
        completed_ctx_ids.update(completed)
        transceiver_gen.check_gen_transfer_status(1)

    def transfers_done():
        return (ctx_request.py_request_id in completed_ctx_ids
                and gen_request.state
                == LlmRequestState.DISAGG_GENERATION_TRANS_COMPLETE)

    wait_for_transfer_completion(poll_transfers, transfers_done)
