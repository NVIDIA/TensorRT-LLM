import gc
import multiprocessing
import sys
import time
import uuid
import weakref

import pytest
import torch

import tensorrt_llm
import tensorrt_llm.bindings
import tensorrt_llm.bindings.executor as trtllm
from tensorrt_llm._torch.distributed import Distributed
from tensorrt_llm._torch.pyexecutor.kv_cache_transceiver import \
    create_kv_cache_transceiver
from tensorrt_llm._torch.pyexecutor.llm_request import (LlmRequest,
                                                        LlmRequestState)
from tensorrt_llm._torch.pyexecutor.mamba_cache_manager import \
    MixedMambaHybridCacheManager
from tensorrt_llm._torch.pyexecutor.resource_manager import KVCacheManager
from tensorrt_llm._torch.pyexecutor.scheduler import ScheduledRequests
from tensorrt_llm.llmapi.llm_args import CacheTransceiverConfig, KvCacheConfig
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.sampling_params import SamplingParams

AttentionTypeCpp = tensorrt_llm.bindings.internal.batch_manager.AttentionType
LlmRequestType = tensorrt_llm.bindings.internal.batch_manager.LlmRequestType
DataType = tensorrt_llm.bindings.DataType
DEFAULT_KV_TRANSFER_TIMEOUT_S = (
    CacheTransceiverConfig.model_fields["kv_transfer_timeout_ms"].default /
    1000.0)
KV_TRANSFER_COMPLETION_MARGIN_S = 10.0


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
@pytest.mark.parametrize("backend_runtime", [("NIXL", None), ("UCX", None),
                                             ("NIXL", "PYTHON")],
                         ids=["NIXL", "UCX", "PYTHON"])
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
                ctx_request_id=ctx_request.request_id,
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
            assert failed == []
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
        backend="DEFAULT", max_tokens_in_buffer=512, kv_transfer_timeout_ms=100)
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


def create_hybrid_cache_manager(mapping,
                                dtype,
                                mamba_conv_dtype=torch.float16,
                                mamba_ssm_dtype=torch.float16):
    """Create a MixedMambaHybridCacheManager for testing hybrid models.

    This manager handles both KV cache (attention layers) and Mamba cache (RNN layers).

    Args:
        mapping: The mapping configuration.
        dtype: KV cache dtype (DataType enum).
        mamba_conv_dtype: Mamba conv states dtype (torch dtype).
        mamba_ssm_dtype: Mamba SSM states dtype (torch dtype).
    """
    num_mamba_layers = 1
    num_attention_layers = 1

    # Attention at layer 0, Mamba at layer 1
    attention_layer_mask = [True, False]
    mamba_layer_mask = [False, True]

    return MixedMambaHybridCacheManager(
        # Mamba cache parameters
        mamba_d_state=16,
        mamba_d_conv=4,
        mamba_num_heads=4,
        mamba_n_groups=1,
        mamba_head_dim=64,
        mamba_num_layers=num_mamba_layers,
        mamba_layer_mask=mamba_layer_mask,
        mamba_cache_dtype=mamba_conv_dtype,
        mamba_ssm_cache_dtype=mamba_ssm_dtype,
        kv_cache_config=KvCacheConfig(
            max_tokens=256,
            enable_block_reuse=False,
        ),
        kv_cache_type=tensorrt_llm.bindings.internal.batch_manager.CacheType.
        SELF,
        num_layers=num_attention_layers,
        layer_mask=attention_layer_mask,
        num_kv_heads=1,
        head_dim=1,
        tokens_per_block=8,
        max_seq_len=256,
        max_batch_size=1,
        mapping=mapping,
        dtype=dtype,
    )


def fill_hybrid_cache_buffers(hybrid_cache_manager):
    """Fill both KV and Mamba cache buffers with random values for testing."""
    with torch.no_grad():
        kv_buffer = hybrid_cache_manager.get_buffers(0)
        random_kv = torch.rand(kv_buffer.shape,
                               dtype=torch.float32,
                               device=kv_buffer.device)
        kv_buffer.copy_(random_kv)

        conv_buffer = hybrid_cache_manager.get_conv_states(1)
        random_conv = torch.rand(conv_buffer.shape,
                                 dtype=conv_buffer.dtype,
                                 device=conv_buffer.device)
        conv_buffer.copy_(random_conv)

        ssm_buffer = hybrid_cache_manager.get_ssm_states(1)
        random_ssm = torch.rand(ssm_buffer.shape,
                                dtype=ssm_buffer.dtype,
                                device=ssm_buffer.device)
        ssm_buffer.copy_(random_ssm)


@pytest.fixture(scope="function")
def hybrid_dtypes(request):
    """
    Returns (kv_dtype, mamba_conv_dtype, mamba_ssm_dtype) based on the parametrized string.

    KV dtype: fp8, bf16
    Conv dtype: fp8, bf16, fp32
    SSM dtype: bf16, fp32
    """
    kv_dtype_str, conv_dtype_str, ssm_dtype_str = request.param

    # Map KV dtype strings to DataType enum
    kv_dtype_map = {
        "fp8": DataType.FP8,
        "bf16": DataType.BF16,
    }

    # Map Mamba dtype strings to torch dtype
    torch_dtype_map = {
        "fp8": torch.float8_e4m3fn,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }

    return kv_dtype_map[kv_dtype_str], torch_dtype_map[
        conv_dtype_str], torch_dtype_map[ssm_dtype_str]


@pytest.mark.timeout(120)
@pytest.mark.parametrize("backend", ["NIXL", "UCX"], ids=["NIXL", "UCX"])
@pytest.mark.parametrize(
    "hybrid_dtypes",
    [
        # (kv_dtype, conv_dtype, ssm_dtype)
        ("bf16", "bf16", "bf16"),
        ("bf16", "bf16", "fp32"),
        ("bf16", "fp32", "bf16"),
        ("bf16", "fp32", "fp32"),
        ("fp8", "bf16", "bf16"),
        ("fp8", "bf16", "fp32"),
        ("fp8", "fp32", "bf16"),
        ("fp8", "fp32", "fp32"),
    ],
    ids=[
        "kv_bf16-conv_bf16-ssm_bf16",
        "kv_bf16-conv_bf16-ssm_fp32",
        "kv_bf16-conv_fp32-ssm_bf16",
        "kv_bf16-conv_fp32-ssm_fp32",
        "kv_fp8-conv_bf16-ssm_bf16",
        "kv_fp8-conv_bf16-ssm_fp32",
        "kv_fp8-conv_fp32-ssm_bf16",
        "kv_fp8-conv_fp32-ssm_fp32",
    ],
    indirect=["hybrid_dtypes"],
)
def test_hybrid_cache_transceiver_single_process(backend, hybrid_dtypes):
    mapping = Mapping(world_size=1, rank=0)
    kv_dtype, mamba_conv_dtype, mamba_ssm_dtype = hybrid_dtypes

    # Create hybrid cache managers (combines KV + Mamba) for context and generation
    hybrid_cache_manager_ctx = create_hybrid_cache_manager(
        mapping, kv_dtype, mamba_conv_dtype, mamba_ssm_dtype)
    hybrid_cache_manager_gen = create_hybrid_cache_manager(
        mapping, kv_dtype, mamba_conv_dtype, mamba_ssm_dtype)

    cache_transceiver_config = CacheTransceiverConfig(backend=backend,
                                                      max_tokens_in_buffer=512)
    dist = Distributed.get(mapping)

    # Create transceivers - the hybrid manager serves as both kv_cache_manager and mamba_cache_manager
    cache_transceiver_ctx = create_kv_cache_transceiver(
        mapping,
        dist,
        hybrid_cache_manager_ctx,
        AttentionTypeCpp.DEFAULT,
        cache_transceiver_config,
        mamba_cache_manager=hybrid_cache_manager_ctx)

    cache_transceiver_gen = create_kv_cache_transceiver(
        mapping,
        dist,
        hybrid_cache_manager_gen,
        AttentionTypeCpp.DEFAULT,
        cache_transceiver_config,
        mamba_cache_manager=hybrid_cache_manager_gen)

    # Fill both KV and Mamba cache buffers with random data
    fill_hybrid_cache_buffers(hybrid_cache_manager_ctx)

    # Init ctx request
    sampling_params = SamplingParams()
    ctx_request = LlmRequest(
        request_id=0,
        max_new_tokens=1,
        input_tokens=list(range(256)),
        sampling_config=tensorrt_llm.bindings.SamplingConfig(
            sampling_params._get_sampling_config()),
        is_streaming=False,
        llm_request_type=LlmRequestType.LLMREQUEST_TYPE_CONTEXT_ONLY)

    # Prepare resources for hybrid manager (handles both KV and Mamba)
    scheduled_ctx = ScheduledRequests()
    scheduled_ctx.context_requests_last_chunk = [ctx_request]
    hybrid_cache_manager_ctx.prepare_resources(scheduled_ctx)

    # Send ctx request (sends both KV and Mamba states)
    cache_transceiver_ctx.respond_and_send_async(ctx_request)

    # Init gen request
    gen_request = LlmRequest(
        request_id=0,
        max_new_tokens=1,
        input_tokens=list(range(256)),
        sampling_config=tensorrt_llm.bindings.SamplingConfig(
            sampling_params._get_sampling_config()),
        is_streaming=False,
        llm_request_type=LlmRequestType.LLMREQUEST_TYPE_GENERATION_ONLY,
        context_phase_params=ctx_request.context_phase_params)

    # Prepare resources for hybrid manager on gen side
    scheduled_gen = ScheduledRequests()
    scheduled_gen.context_requests_last_chunk = [gen_request]
    hybrid_cache_manager_gen.prepare_resources(scheduled_gen)

    cache_transceiver_gen.request_and_receive_async(gen_request)

    completed_ctx_ids = set()

    def poll_transfers():
        completed, failed = cache_transceiver_ctx.check_context_transfer_status(
            1)
        assert failed == []
        completed_ctx_ids.update(completed)
        cache_transceiver_gen.check_gen_transfer_status(1)

    def transfers_done():
        return (ctx_request.py_request_id in completed_ctx_ids
                and gen_request.state
                == LlmRequestState.DISAGG_GENERATION_TRANS_COMPLETE)

    wait_for_transfer_completion(poll_transfers, transfers_done)

    assert torch.equal(
        hybrid_cache_manager_gen.get_buffers(0),
        hybrid_cache_manager_ctx.get_buffers(0)), "different kv-cache values"

    # The transceiver copies a single request's state between
    # independently-allocated slots on each side, so we check the
    # request's own slot instead of the full state buffer (which has
    # extra padding-dummy slots that only the ctx side touched).
    slot_ctx = hybrid_cache_manager_ctx._impl.mamba_impl.get_cache_index(
        ctx_request.py_request_id)
    slot_gen = hybrid_cache_manager_gen._impl.mamba_impl.get_cache_index(
        gen_request.py_request_id)
    assert torch.equal(
        hybrid_cache_manager_gen.get_conv_states(1)[slot_gen],
        hybrid_cache_manager_ctx.get_conv_states(1)[slot_ctx]), (
            "different mamba conv states")

    assert torch.equal(
        hybrid_cache_manager_gen.get_ssm_states(1)[slot_gen],
        hybrid_cache_manager_ctx.get_ssm_states(1)[slot_ctx]), (
            "different mamba ssm states")


@pytest.mark.timeout(120)
@pytest.mark.parametrize("backend", ["NIXL", "UCX"], ids=["NIXL", "UCX"])
def test_hybrid_cache_transceiver_cancel_request(backend):

    mapping = Mapping(world_size=1, rank=0)
    dtype = DataType.HALF

    hybrid_cache_manager_ctx = create_hybrid_cache_manager(mapping, dtype)
    hybrid_cache_manager_gen = create_hybrid_cache_manager(mapping, dtype)

    cache_transceiver_config = CacheTransceiverConfig(backend=backend,
                                                      max_tokens_in_buffer=512)
    dist = Distributed.get(mapping)

    cache_transceiver_ctx = create_kv_cache_transceiver(
        mapping,
        dist,
        hybrid_cache_manager_ctx,
        AttentionTypeCpp.DEFAULT,
        cache_transceiver_config,
        mamba_cache_manager=hybrid_cache_manager_ctx)

    cache_transceiver_gen = create_kv_cache_transceiver(
        mapping,
        dist,
        hybrid_cache_manager_gen,
        AttentionTypeCpp.DEFAULT,
        cache_transceiver_config,
        mamba_cache_manager=hybrid_cache_manager_gen)

    fill_hybrid_cache_buffers(hybrid_cache_manager_ctx)

    # Init ctx request
    sampling_params = SamplingParams()
    ctx_request = LlmRequest(
        request_id=0,
        max_new_tokens=1,
        input_tokens=list(range(256)),
        sampling_config=tensorrt_llm.bindings.SamplingConfig(
            sampling_params._get_sampling_config()),
        is_streaming=False,
        llm_request_type=LlmRequestType.LLMREQUEST_TYPE_CONTEXT_ONLY)

    scheduled_ctx = ScheduledRequests()
    scheduled_ctx.context_requests_last_chunk = [ctx_request]
    hybrid_cache_manager_ctx.prepare_resources(scheduled_ctx)

    # Send ctx request
    cache_transceiver_ctx.respond_and_send_async(ctx_request)

    # Wait for ctx request to be sent
    time.sleep(2)

    # Cancel ctx request
    is_cancelled = cache_transceiver_ctx.cancel_request(ctx_request)
    assert is_cancelled

    # Init gen request
    gen_request = LlmRequest(
        request_id=0,
        max_new_tokens=1,
        input_tokens=list(range(256)),
        sampling_config=tensorrt_llm.bindings.SamplingConfig(
            sampling_params._get_sampling_config()),
        is_streaming=False,
        llm_request_type=LlmRequestType.LLMREQUEST_TYPE_GENERATION_ONLY,
        context_phase_params=ctx_request.context_phase_params)

    scheduled_gen = ScheduledRequests()
    scheduled_gen.context_requests_last_chunk = [gen_request]
    hybrid_cache_manager_gen.prepare_resources(scheduled_gen)

    # Try to receive gen request
    cache_transceiver_gen.request_and_receive_async(gen_request)

    # Block the main thread due to the async operation
    time.sleep(2)
    assert gen_request.state == LlmRequestState.DISAGG_TRANS_ERROR
