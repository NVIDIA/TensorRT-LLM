import threading
import time
import uuid

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
    MambaHybridCacheManager
from tensorrt_llm._torch.pyexecutor.resource_manager import KVCacheManager
from tensorrt_llm._torch.pyexecutor.scheduler import ScheduledRequests
from tensorrt_llm.llmapi.llm_args import CacheTransceiverConfig, KvCacheConfig
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.sampling_params import SamplingParams

AttentionTypeCpp = tensorrt_llm.bindings.internal.batch_manager.AttentionType
LlmRequestType = tensorrt_llm.bindings.internal.batch_manager.LlmRequestType
DataType = tensorrt_llm.bindings.DataType


def create_kv_cache_manager(mapping, dtype):
    return KVCacheManager(
        trtllm.KvCacheConfig(
            max_tokens=256,
            enable_block_reuse=False,
        ),
        tensorrt_llm.bindings.internal.batch_manager.CacheType.SELF,
        num_layers=1,
        num_kv_heads=1,
        head_dim=1,
        tokens_per_block=8,
        max_seq_len=256,
        max_batch_size=1,
        mapping=mapping,
        dtype=dtype)


def fill_kv_cache_buffer(kv_cache_manager):
    with torch.no_grad():
        buffer = kv_cache_manager.get_buffers(0)
        random_values = torch.rand(buffer.shape,
                                   dtype=torch.float32,
                                   device=buffer.device)
        buffer.copy_(random_values)


# create_kv_cache_manager() configures a 256-token KV pool. Tests that need
# two concurrent generation requests must keep prompts small enough that both
# fit, hence this default. Single-request tests can override via prompt_len.
_DEFAULT_PROMPT_LEN = 64


def _make_request(request_id,
                  llm_request_type,
                  context_phase_params=None,
                  prompt_len=_DEFAULT_PROMPT_LEN):
    """Construct an ``LlmRequest`` with the shared disagg test boilerplate.

    ``prompt_len`` controls the synthetic input-token range and must fit in
    the small KV pool from :func:`create_kv_cache_manager`.
    """
    sampling_params = SamplingParams()
    kwargs = dict(
        request_id=request_id,
        max_new_tokens=1,
        input_tokens=list(range(prompt_len)),
        sampling_config=tensorrt_llm.bindings.SamplingConfig(
            sampling_params._get_sampling_config()),
        is_streaming=False,
        llm_request_type=llm_request_type,
    )
    if context_phase_params is not None:
        kwargs["context_phase_params"] = context_phase_params
    return LlmRequest(**kwargs)


def _add_sequence(kv_cache_manager, request):
    """Bind ``request`` into ``kv_cache_manager``.

    This uses the call signature every disagg transceiver test relies on.
    """
    kv_cache_manager.impl.add_sequence(request.py_request_id,
                                       request.prompt_len, 1, request)


def _drive_template_handshake_to_completion(ctx_xcvr,
                                            gen_xcvr,
                                            ctx_request_id,
                                            deadline_s=10):
    """Poll both transceivers until the template transfer completes.

    ``ctx_request_id`` must land in the sender-side ``completed_ids`` AND the
    gen-side ``mRequesterFutures`` must drain.

    The post-fix ``check_gen_transfer_status(1)`` is non-blocking (it skips
    unready futures via ``wait_for(0)``), so a single call may leave the
    template entry in ``mRequesterFutures`` if its data has not yet
    arrived. Tests that stage a "blocked" request after a template
    handshake must drain mRequesterFutures first so the blocked-side
    assertions only see their own request.
    """
    deadline = time.time() + deadline_s
    ctx_done = False
    while time.time() < deadline and (
            not ctx_done or not gen_xcvr.check_gen_transfer_complete()):
        completed_ids, error_ids = ctx_xcvr.check_context_transfer_status(1)
        assert ctx_request_id not in error_ids, (
            f"template ctx request {ctx_request_id} unexpectedly errored: "
            f"{error_ids}")
        if ctx_request_id in completed_ids:
            ctx_done = True
        gen_xcvr.check_gen_transfer_status(1)
        time.sleep(0.05)
    assert ctx_done, (
        f"template ctx request {ctx_request_id} did not complete within "
        f"{deadline_s}s; sender side may be stuck before the test can "
        f"stage its blocked request")
    assert gen_xcvr.check_gen_transfer_complete(), (
        f"gen-side mRequesterFutures still non-empty within {deadline_s}s; "
        f"the staged blocked-request assertions would otherwise observe "
        f"a spurious lingering entry from the template handshake")


@pytest.fixture
def disagg_transceiver_pair(request):
    """Build a single-process disagg ctx/gen transceiver pair.

    Used by the cancellation-flow regression tests.

    Yields ``(kv_cache_manager_ctx, kv_cache_manager_gen,
    kv_cache_transceiver_ctx, kv_cache_transceiver_gen)``.

    Pass ``(attention_type, backend)`` as the parametrize argument; the
    backend defaults to ``"DEFAULT"`` (UCX) for backend-agnostic fixes
    and can be set to ``"NIXL"`` for tests that depend on NIXL-specific
    paths (e.g. the recv buffer index manager).
    """
    param = request.param
    if isinstance(param, tuple):
        attention_type = param[0]
        backend = param[1] if len(param) > 1 else "DEFAULT"
    else:
        attention_type = param
        backend = "DEFAULT"

    tensorrt_llm.logger.set_level("info")
    mapping = Mapping(world_size=1, rank=0)
    dist = Distributed.get(mapping)
    kv_cache_manager_ctx = create_kv_cache_manager(mapping, DataType.HALF)
    kv_cache_manager_gen = create_kv_cache_manager(mapping, DataType.HALF)

    cache_transceiver_config = CacheTransceiverConfig(backend=backend,
                                                      max_tokens_in_buffer=512)

    kv_cache_transceiver_ctx = create_kv_cache_transceiver(
        mapping, dist, kv_cache_manager_ctx, attention_type,
        cache_transceiver_config)
    kv_cache_transceiver_gen = create_kv_cache_transceiver(
        mapping, dist, kv_cache_manager_gen, attention_type,
        cache_transceiver_config)

    fill_kv_cache_buffer(kv_cache_manager_ctx)

    yield (kv_cache_manager_ctx, kv_cache_manager_gen, kv_cache_transceiver_ctx,
           kv_cache_transceiver_gen)


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

    kv_cache_manager_ctx.impl.add_sequence(ctx_request.py_request_id,
                                           ctx_request.prompt_len, 1,
                                           ctx_request)
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
            first_gen_tokens=ctx_request.context_phase_params.first_gen_tokens,
            draft_tokens=ctx_request.context_phase_params.draft_tokens)

        gen_request.py_disaggregated_params = disaggregated_params

    kv_cache_manager_gen.impl.add_sequence(gen_request.py_request_id,
                                           gen_request.prompt_len, 1,
                                           gen_request)
    # send gen request
    kv_cache_transceiver_gen.request_and_receive_async(gen_request)

    kv_cache_transceiver_ctx.check_context_transfer_status(1)
    kv_cache_transceiver_gen.check_gen_transfer_status(1)

    assert torch.equal(
        kv_cache_manager_gen.get_buffers(0),
        kv_cache_manager_ctx.get_buffers(0)), "different kv-cache values"


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

    kv_cache_manager_ctx.impl.add_sequence(ctx_request.py_request_id,
                                           ctx_request.prompt_len, 1,
                                           ctx_request)
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

    kv_cache_manager_gen.impl.add_sequence(gen_request.py_request_id,
                                           gen_request.prompt_len, 1,
                                           gen_request)
    # send gen request
    kv_cache_transceiver_gen.request_and_receive_async(gen_request)

    # Block the main thread due to the async operation
    time.sleep(2)
    assert gen_request.state == LlmRequestState.DISAGG_TRANS_ERROR


@pytest.mark.timeout(120)
@pytest.mark.parametrize("disagg_transceiver_pair",
                         [AttentionTypeCpp.DEFAULT, AttentionTypeCpp.MLA],
                         ids=["mha", "mla"],
                         indirect=True)
def test_cancel_request_in_transmission_does_not_break_sender_future(
        disagg_transceiver_pair, capfd):
    """Reproduce the sender-side broken-promise on cancel-after-ready.

    Pre-fix ``CacheSender::Impl::sendResponse`` erases the ready
    ``(request, promise)`` pair on cancel without first fulfilling the
    promise, and the destructor surfaces ``future_error: Broken promise``
    on the sender future returned by ``respond_and_send_async()``.
    """
    (kv_cache_manager_ctx, kv_cache_manager_gen, kv_cache_transceiver_ctx,
     kv_cache_transceiver_gen) = disagg_transceiver_pair

    ctx_request = _make_request(0,
                                LlmRequestType.LLMREQUEST_TYPE_CONTEXT_ONLY,
                                prompt_len=256)
    _add_sequence(kv_cache_manager_ctx, ctx_request)
    kv_cache_transceiver_ctx.respond_and_send_async(ctx_request)

    # Wait for the sender to reach the ready state, then cancel it.
    time.sleep(2)
    is_cancelled = kv_cache_transceiver_ctx.cancel_request(ctx_request)
    assert is_cancelled, (
        "ctx_request must be cancellable while still ready in "
        "mReadyResponses")

    gen_request = _make_request(0,
                                LlmRequestType.LLMREQUEST_TYPE_GENERATION_ONLY,
                                ctx_request.context_phase_params,
                                prompt_len=256)
    _add_sequence(kv_cache_manager_gen, gen_request)
    kv_cache_transceiver_gen.request_and_receive_async(gen_request)

    # Sender side: the cancelled request must surface as an error rather
    # than being marked complete.
    completed_ids, error_ids = [], []
    deadline = time.time() + 10
    while time.time() < deadline and not error_ids:
        completed_ids, error_ids = (
            kv_cache_transceiver_ctx.check_context_transfer_status(1))
        if error_ids:
            break
        time.sleep(0.1)

    assert ctx_request.py_request_id not in completed_ids, (
        "cancelled ctx request must not appear in completed_ids")
    assert ctx_request.py_request_id in error_ids, (
        "cancelled ctx request must surface in error_ids")

    # Receiver side: the matching gen request must observe the cancellation
    # as a structured DISAGG_TRANS_ERROR rather than a Broken-promise
    # exception.
    deadline = time.time() + 10
    while time.time() < deadline and (gen_request.state
                                      != LlmRequestState.DISAGG_TRANS_ERROR):
        kv_cache_transceiver_gen.check_gen_transfer_status(1)
        time.sleep(0.1)

    assert gen_request.state == LlmRequestState.DISAGG_TRANS_ERROR, (
        "gen request mirroring the cancelled ctx request must end in "
        "DISAGG_TRANS_ERROR")

    captured = capfd.readouterr()
    merged = captured.out + captured.err
    assert "Broken promise" not in merged, (
        "signature #1 reproduced: cancel-after-ready left the sender "
        "promise unresolved and the destructor surfaced as Broken "
        "promise on the sender future")

    kv_cache_manager_ctx.free_resources(ctx_request)
    kv_cache_manager_gen.free_resources(gen_request)


_PROBE_TIMEOUT_S = 2.5


@pytest.mark.timeout(120)
@pytest.mark.parametrize("disagg_transceiver_pair",
                         [AttentionTypeCpp.DEFAULT, AttentionTypeCpp.MLA],
                         ids=["mha", "mla"],
                         indirect=True)
def test_check_gen_transfer_status_at_least_one_does_not_block_on_unready_future(
        disagg_transceiver_pair):
    """Reproduce the gen-side blocking hang in checkGenTransferStatus(1).

    On stock ``rc11`` the polling path called from the PyExecutor disagg
    loop unconditionally ``future.get()``s the first selected requester
    future, even when its ``wait_for(0)`` is still ``timeout``. A single
    in-flight generation request whose context-side ready signal has not
    yet arrived therefore blocks the entire decoder event loop, which is
    indistinguishable from a wedge.

    The test exercises the same shape as the wedge: drive one full
    ctx/gen handshake to completion to capture an opaque comm/cache
    state, then enqueue a generation request whose context counterpart
    has not yet been ``respond_and_send_async()``-ed and call
    ``check_gen_transfer_status(1)`` from a separate thread. The call
    must return within a bounded probe timeout larger than the
    production 1000ms sender-future wait instead of blocking indefinitely
    on the unresolved future.
    """
    (kv_cache_manager_ctx, kv_cache_manager_gen, kv_cache_transceiver_ctx,
     kv_cache_transceiver_gen) = disagg_transceiver_pair

    # Complete one normal transfer first so we can reuse its opaque
    # comm/cache state for the second (intentionally unresolved) generation
    # request.
    template_ctx_request = _make_request(
        100, LlmRequestType.LLMREQUEST_TYPE_CONTEXT_ONLY, prompt_len=256)
    _add_sequence(kv_cache_manager_ctx, template_ctx_request)
    kv_cache_transceiver_ctx.respond_and_send_async(template_ctx_request)

    template_gen_request = _make_request(
        100,
        LlmRequestType.LLMREQUEST_TYPE_GENERATION_ONLY,
        template_ctx_request.context_phase_params,
        prompt_len=256)
    _add_sequence(kv_cache_manager_gen, template_gen_request)
    kv_cache_transceiver_gen.request_and_receive_async(template_gen_request)

    _drive_template_handshake_to_completion(kv_cache_transceiver_ctx,
                                            kv_cache_transceiver_gen,
                                            template_ctx_request.py_request_id)

    opaque_state = template_ctx_request.context_phase_params.opaque_state
    assert opaque_state is not None, (
        "template ctx_request must expose its opaque comm/cache state for "
        "the staged blocked request below to reuse")

    kv_cache_manager_ctx.free_resources(template_ctx_request)
    kv_cache_manager_gen.free_resources(template_gen_request)

    # Build a generation request for a different ctx_request_id before the
    # sender has any matching ready response. This leaves a real unresolved
    # future in the C++ transceiver and reproduces the blocking pattern
    # behind signature #4.
    blocked_request_id = 101
    blocked_ctx_request = _make_request(
        blocked_request_id,
        LlmRequestType.LLMREQUEST_TYPE_CONTEXT_ONLY,
        prompt_len=256)
    _add_sequence(kv_cache_manager_ctx, blocked_ctx_request)

    blocked_context_phase_params = trtllm.ContextPhaseParams(
        list(template_ctx_request.context_phase_params.first_gen_tokens),
        blocked_request_id,
        bytes(opaque_state),
        template_ctx_request.context_phase_params.draft_tokens,
        template_ctx_request.context_phase_params.ctx_dp_rank,
        template_ctx_request.context_phase_params.disagg_info_endpoint,
    )
    blocked_gen_request = _make_request(
        blocked_request_id,
        LlmRequestType.LLMREQUEST_TYPE_GENERATION_ONLY,
        blocked_context_phase_params,
        prompt_len=256)
    _add_sequence(kv_cache_manager_gen, blocked_gen_request)
    kv_cache_transceiver_gen.request_and_receive_async(blocked_gen_request)

    # Sanity: at_least_request_num=0 must not block under any circumstance.
    start = time.time()
    kv_cache_transceiver_gen.check_gen_transfer_status(0)
    assert time.time() - start < 1.0, (
        "check_gen_transfer_status(0) must be non-blocking even when "
        "futures are unresolved")

    # Real reproducer: at_least_request_num=1 must NOT hang on an
    # unresolved future. Run it on a worker thread so that pre-fix this
    # thread stays blocked past the bounded probe timeout (the failure
    # signature), and post-fix it returns after at most the configured
    # per-iteration sender-future wait because the unready future is skipped.
    check_result = {"returned": False, "error": None}

    def call_blocking_check():
        try:
            kv_cache_transceiver_gen.check_gen_transfer_status(1)
        except BaseException as exc:  # noqa: BLE001
            check_result["error"] = exc
        finally:
            check_result["returned"] = True

    blocked_check = threading.Thread(target=call_blocking_check, daemon=True)
    blocked_check.start()
    blocked_check.join(timeout=_PROBE_TIMEOUT_S)
    blocked_during_probe = blocked_check.is_alive()

    # Allow the wedged context request to complete cleanly so the worker
    # thread can finish (in either pre- or post-fix behaviour) and we can
    # tear down the test without leaking threads.
    kv_cache_transceiver_ctx.respond_and_send_async(blocked_ctx_request)

    deadline = time.time() + 10
    completed_ids, error_ids = [], []
    while time.time() < deadline and (blocked_ctx_request.py_request_id
                                      not in completed_ids):
        completed_ids, error_ids = (
            kv_cache_transceiver_ctx.check_context_transfer_status(1))
        assert blocked_ctx_request.py_request_id not in error_ids, (
            "blocked ctx request must not error during teardown polling")
        if blocked_ctx_request.py_request_id in completed_ids:
            break
        time.sleep(0.1)
    assert blocked_ctx_request.py_request_id in completed_ids, (
        "blocked ctx request must complete on the sender side once "
        "respond_and_send_async unblocks the receiver")

    if blocked_during_probe:
        # Pre-fix path: the thread is wedged inside
        # check_gen_transfer_status(1). It will only return after the
        # sender's respond_and_send_async unblocks the receive that the
        # in-thread call is parked on.
        blocked_check.join(timeout=10)
        assert not blocked_check.is_alive(), (
            "blocked check thread must drain within 10s after the sender "
            "supplies the ready signal")
    else:
        # Post-fix path: the in-thread call already returned because it
        # skipped the unready future. Drain mRequesterFutures via
        # additional polling so subsequent assertions see an empty queue.
        deadline = time.time() + 10
        while time.time() < deadline and (
                not kv_cache_transceiver_gen.check_gen_transfer_complete()):
            kv_cache_transceiver_gen.check_gen_transfer_status(1)
            time.sleep(0.1)

    if check_result["error"] is not None:
        raise check_result["error"]
    assert check_result["returned"], (
        "blocked check thread must signal completion via check_result")
    assert not blocked_during_probe, (
        "signature #4 reproduced: check_gen_transfer_status(1) blocked on "
        "an unresolved generation future for longer than the bounded probe")
    assert kv_cache_transceiver_gen.check_gen_transfer_complete(), (
        "gen-side mRequesterFutures must drain after the blocked request "
        "is unblocked")

    assert torch.equal(
        kv_cache_manager_gen.get_buffers(0),
        kv_cache_manager_ctx.get_buffers(0)), "different kv-cache values"

    kv_cache_manager_ctx.free_resources(blocked_ctx_request)
    kv_cache_manager_gen.free_resources(blocked_gen_request)


@pytest.mark.timeout(120)
@pytest.mark.parametrize("disagg_transceiver_pair",
                         [AttentionTypeCpp.DEFAULT, AttentionTypeCpp.MLA],
                         ids=["mha", "mla"],
                         indirect=True)
def test_cancel_queued_gen_request_fulfills_receiver_future(
        disagg_transceiver_pair, capfd):
    """Reproduce the receiver-side queued-cancel broken-promise.

    Mirrors the sender-side reproducer but on
    ``CacheReceiver::Impl::cancelRequest()``. Pre-fix that function erases
    the queued ``(request, promise)`` pair without first fulfilling the
    promise, so the ``std::promise`` destructor sets
    ``future_error: Broken promise`` on the future returned by
    ``request_and_receive_async()``.

    The test holds the receiver worker thread busy with a first
    generation request that has no matching context counterpart,
    enqueues a second generation request behind it, then cancels the
    queued request. Post-fix the queued promise is fulfilled with a
    structured cancellation exception before the queue entry is erased.
    """
    (kv_cache_manager_ctx, kv_cache_manager_gen, kv_cache_transceiver_ctx,
     kv_cache_transceiver_gen) = disagg_transceiver_pair

    # Drive one full ctx/gen handshake to completion so we can reuse a
    # real opaque comm/cache state for the orphan requests below.
    template_ctx_request = _make_request(
        100, LlmRequestType.LLMREQUEST_TYPE_CONTEXT_ONLY)
    _add_sequence(kv_cache_manager_ctx, template_ctx_request)
    kv_cache_transceiver_ctx.respond_and_send_async(template_ctx_request)

    template_gen_request = _make_request(
        100, LlmRequestType.LLMREQUEST_TYPE_GENERATION_ONLY,
        template_ctx_request.context_phase_params)
    _add_sequence(kv_cache_manager_gen, template_gen_request)
    kv_cache_transceiver_gen.request_and_receive_async(template_gen_request)

    _drive_template_handshake_to_completion(kv_cache_transceiver_ctx,
                                            kv_cache_transceiver_gen,
                                            template_ctx_request.py_request_id)

    opaque_state = template_ctx_request.context_phase_params.opaque_state
    assert opaque_state is not None, (
        "template ctx_request must expose its opaque comm/cache state for "
        "the staged orphan requests below to reuse")

    kv_cache_manager_ctx.free_resources(template_ctx_request)
    kv_cache_manager_gen.free_resources(template_gen_request)

    def make_orphan_gen_request(request_id):
        ctx_phase_params = trtllm.ContextPhaseParams(
            list(template_ctx_request.context_phase_params.first_gen_tokens),
            request_id,
            bytes(opaque_state),
            template_ctx_request.context_phase_params.draft_tokens,
            template_ctx_request.context_phase_params.ctx_dp_rank,
            template_ctx_request.context_phase_params.disagg_info_endpoint,
        )
        gen_request = _make_request(
            request_id, LlmRequestType.LLMREQUEST_TYPE_GENERATION_ONLY,
            ctx_phase_params)
        _add_sequence(kv_cache_manager_gen, gen_request)
        return gen_request

    # Submit a first orphan gen request whose context counterpart will
    # never respond_and_send_async. The receiver worker dequeues it and
    # parks inside requestSync() / sendRequestInfo(), tying up the worker
    # thread.
    blocking_gen_request = make_orphan_gen_request(101)
    kv_cache_transceiver_gen.request_and_receive_async(blocking_gen_request)

    # Submit a second orphan gen request. The first one is still in
    # requestSync(), so this one stays in mRequestsQueue and is the
    # actual subject of the queued-cancel reproducer.
    queued_gen_request = make_orphan_gen_request(102)
    kv_cache_transceiver_gen.request_and_receive_async(queued_gen_request)

    # Wait briefly so the receiver worker has had time to dequeue the
    # first request and block on it, leaving the second one queued.
    time.sleep(1)

    # Cancel the queued request. Pre-fix this erases the (request,
    # promise) pair without fulfilling the promise; post-fix it
    # set_exception()s a structured kNETWORK_ERROR before erasing.
    is_cancelled = kv_cache_transceiver_gen.cancel_request(queued_gen_request)
    assert is_cancelled, (
        "queued_gen_request must still be in the receiver queue when we "
        "call cancel_request(); if this fails, the receiver worker may "
        "have dequeued faster than expected and the test setup needs to "
        "be tightened")

    # Poll the gen-side polling loop and assert the cancelled request
    # lands in DISAGG_TRANS_ERROR within a reasonable window. Pre-fix
    # this returns via a Broken-promise exception with no useful
    # diagnostic; post-fix it returns via the structured kNETWORK_ERROR
    # set by the fix.
    deadline = time.time() + 10
    while time.time() < deadline and (queued_gen_request.state
                                      != LlmRequestState.DISAGG_TRANS_ERROR):
        kv_cache_transceiver_gen.check_gen_transfer_status(1)
        time.sleep(0.1)

    assert queued_gen_request.state == LlmRequestState.DISAGG_TRANS_ERROR, (
        "queued gen request must land in DISAGG_TRANS_ERROR after "
        "cancellation rather than surfacing a Broken-promise exception")

    # Clean up the intentionally blocked request before the transceiver
    # is destroyed. Leaving it in requestSync() would make teardown race
    # the worker thread that this test deliberately parked.
    is_blocking_cancelled = (
        kv_cache_transceiver_gen.cancel_request(blocking_gen_request))
    assert is_blocking_cancelled, (
        "blocking_gen_request must be cancellable so teardown does not "
        "race the parked receiver worker")
    deadline = time.time() + 10
    while time.time() < deadline and (blocking_gen_request.state
                                      != LlmRequestState.DISAGG_TRANS_ERROR):
        kv_cache_transceiver_gen.check_gen_transfer_status(1)
        time.sleep(0.1)
    assert blocking_gen_request.state == LlmRequestState.DISAGG_TRANS_ERROR, (
        "blocking_gen_request must drain to DISAGG_TRANS_ERROR before "
        "transceiver destruction")

    kv_cache_manager_gen.free_resources(blocking_gen_request)
    kv_cache_manager_gen.free_resources(queued_gen_request)

    captured = capfd.readouterr()
    merged = captured.out + captured.err
    assert "Broken promise" not in merged, (
        "signature #5 reproduced: cancelling a queued generation request "
        "left its std::promise unresolved and the destructor surfaced "
        "as Broken promise on the consumer side")


def create_hybrid_cache_manager(mapping,
                                dtype,
                                mamba_conv_dtype=torch.float16,
                                mamba_ssm_dtype=torch.float16):
    """Create a MambaHybridCacheManager for testing hybrid models.

    This manager handles both KV cache (attention layers) and Mamba cache
    (RNN layers).

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

    return MambaHybridCacheManager(
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
    """Return dtypes for the parametrized hybrid-cache test case.

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
def test_hybrid_cache_transceiver_single_process(backend, hybrid_dtypes,
                                                 monkeypatch):
    monkeypatch.setenv("TRTLLM_USE_CPP_MAMBA", "1")
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

    # Create transceivers - MambaHybridCacheManager serves as both kv_cache_manager and mamba_cache_manager
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

    cache_transceiver_ctx.check_context_transfer_status(1)
    cache_transceiver_gen.check_gen_transfer_status(1)

    assert torch.equal(
        hybrid_cache_manager_gen.get_buffers(0),
        hybrid_cache_manager_ctx.get_buffers(0)), "different kv-cache values"

    assert torch.equal(hybrid_cache_manager_gen.get_conv_states(1),
                       hybrid_cache_manager_ctx.get_conv_states(
                           1)), "different mamba conv states"

    assert torch.equal(hybrid_cache_manager_gen.get_ssm_states(1),
                       hybrid_cache_manager_ctx.get_ssm_states(
                           1)), "different mamba ssm states"


@pytest.mark.timeout(120)
@pytest.mark.parametrize("backend", ["NIXL", "UCX"], ids=["NIXL", "UCX"])
def test_hybrid_cache_transceiver_cancel_request(backend, monkeypatch):
    monkeypatch.setenv("TRTLLM_USE_CPP_MAMBA", "1")

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
