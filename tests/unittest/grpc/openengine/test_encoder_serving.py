# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import threading
from types import SimpleNamespace

import pytest
from conftest import AbortError, FakeServicerContext
from test_control import MODEL, _inference, _servicer
from test_encoder import wire_request as request
from utils.native_encoder import model, native_result

from tensorrt_llm._torch.shared_tensor import SharedTensorContainer
from tensorrt_llm.disaggregated_params import DisaggregatedParams
from tensorrt_llm.grpc.openengine import encoding, servicer
from tensorrt_llm.grpc.openengine.bindings import encoding_pb2, lifecycle_pb2, model_pb2, server_pb2
from tensorrt_llm.grpc.openengine.capabilities import MediaCapabilities
from tensorrt_llm.grpc.openengine.control import OpenEngineControlServicer
from tensorrt_llm.grpc.openengine.errors import (
    EncoderCapacityError,
    EncoderUnavailableError,
    MediaUnavailableError,
)

pytestmark = pytest.mark.cpu_only


def encode_request():
    value = request()
    return encoding_pb2.EncodeRequest(
        request_id=value.request_id, model=value.model, prompt=value.prompt, media=value.media
    )


async def prepared_encode(*args):
    return SimpleNamespace(multimodal_params=None)


@pytest.fixture
def encoder_inference(tmp_path, monkeypatch):
    llm, _ = model()
    llm.args = SimpleNamespace(mm_encoder_only=True, guided_decoding_backend=None)
    result, _source_tensors = native_result()

    class Handle:
        error = None
        finished = True
        disaggregated_params = result.disaggregated_params

        async def __aiter__(self):
            yield self

        async def aresult(self):
            return self

        def abort(self):
            pass

    handle = Handle()

    def generate(inputs, sampling_params):
        assert sampling_params.max_tokens == 1
        return handle

    llm.generate_async = generate
    monkeypatch.setenv("TRTLLM_OPENENGINE_ENCODER_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(encoding, "_prepare", prepared_encode)
    inference = servicer.OpenEngineInferenceServicer(llm, "qwen")
    outcomes = []
    inference._metrics = SimpleNamespace(
        record_outcome=lambda *outcome: outcomes.append(outcome),
        record_result=lambda *_: pytest.fail("Encode has no generation-token accounting"),
    )
    try:
        yield inference, handle, outcomes
    finally:
        assert encoding.release_encoder_result(handle)


@pytest.mark.asyncio
@pytest.mark.parametrize("healthy", [True, False])
async def test_encoder_health_skips_language_probe_without_masking_unhealthy_engine(healthy):
    inference = _inference()
    inference.encoder_role = "encode"
    control = _servicer(inference=inference, _executor_unhealthy=not healthy)

    def unexpected_generate(*args, **kwargs):
        pytest.fail("encoder health must not submit a language-generation probe")

    control._llm.generate_async = unexpected_generate
    response = await control.Health(
        lifecycle_pb2.HealthRequest(include_inference_probe=True), FakeServicerContext()
    )
    probe = next(check for check in response.checks if check.name == "inference_probe")
    assert probe.state == lifecycle_pb2.HEALTH_STATE_READY
    assert "not applicable" in probe.message and "encoder-only" in probe.message
    assert "inference liveness was not tested" in probe.message
    assert response.state == (
        lifecycle_pb2.HEALTH_STATE_READY if healthy else lifecycle_pb2.HEALTH_STATE_NOT_READY
    )
    assert inference.active_request_count() == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "ending,epd",
    [
        ("done", False),
        ("postprocess", False),
        ("error", False),
        ("format_error", False),
        ("cancelled", False),
        ("abort_late", False),
        ("cancel_in_release", False),
        ("release_error", False),
        ("release_error", True),
        ("aggregate", False),
        ("decode", False),
    ],
)
async def test_prefill_reclaims_native_mrope_results_on_every_exit(monkeypatch, ending, epd):
    llm, _ = model()
    llm.args = SimpleNamespace(guided_decoding_backend=None)
    llm.tokenizer = None
    llm._check_health = lambda: True
    fresh, _source_tensors = native_result()
    params = fresh.disaggregated_params
    SharedTensorContainer.release_dict(params.multimodal_embedding_handles.pop())
    expected = [params.mrope_position_ids_handle, params.mrope_position_deltas_handle]
    released = []
    completed = []
    release = SharedTensorContainer.release_dict
    entered = asyncio.Event()
    allow_release = threading.Event()
    loop = asyncio.get_running_loop()

    def record(handle):
        released.append(handle)
        if ending == "cancel_in_release" and handle == expected[0]:
            loop.call_soon_threadsafe(entered.set)
            assert allow_release.wait(5)
        if ending == "release_error" and handle == expected[0]:
            raise RuntimeError("release outcome is unknown")
        release(handle)
        completed.append(handle)

    monkeypatch.setattr(SharedTensorContainer, "release_dict", record)

    class Handle:
        disaggregated_params = DisaggregatedParams(request_type="context_only")
        finished = ending != "abort_late"
        error = "native failure" if ending == "error" else None
        aborted = False

        async def __aiter__(self):
            if ending == "abort_late":
                raise asyncio.CancelledError
            if ending == "postprocess":
                self.outputs = [SimpleNamespace(disaggregated_params=params)]
            else:
                self.disaggregated_params = params
            self.finished = False
            yield self
            self.finished = True
            yield self

        async def aresult(self):
            if ending == "cancelled":
                pytest.fail(
                    "already-released exports on a text-only request must not trigger a wait"
                )
            assert self.aborted
            self.disaggregated_params = params
            return self

        def abort(self):
            self.aborted = True

    handle = Handle()
    llm.generate_async = lambda **kwargs: handle
    monkeypatch.setattr(
        servicer,
        "_prepare_input_from_request",
        lambda *args: SimpleNamespace(media=(), inputs="prompt"),
    )

    def format_result(*args):
        if ending == "format_error":
            raise ValueError("format failed")
        return []

    monkeypatch.setattr(servicer, "_format_result", format_result)
    monkeypatch.setattr(servicer, "_engine_error_response", lambda *args: None)
    context = FakeServicerContext()
    context.is_cancelled = ending == "cancelled"
    req = request()
    req.ClearField("media")
    if ending == "decode":
        req.kv.session.session_id = "123"
        req.kv.session.attributes_struct.update({"opaque_state": "a3Y="})
    elif ending != "aggregate":
        req.extra.update({"request_type": "context_only"})
    inference = servicer.OpenEngineInferenceServicer(llm, "qwen")
    if epd:
        inference.encoder_role = "consume"
    if ending == "abort_late":

        async def load_multimodal_input(*args):
            return SimpleNamespace(multimodal_params=SimpleNamespace())

        monkeypatch.setattr(servicer, "_load_prepared_input", load_multimodal_input)
    if ending == "cancel_in_release":

        async def consume():
            return [value async for value in inference.Generate(req, context)]

        rpc = asyncio.create_task(consume())
        drain = None
        try:
            await asyncio.wait_for(entered.wait(), timeout=2)
            rpc.cancel()
            with pytest.raises(asyncio.CancelledError):
                await rpc
            drain = asyncio.create_task(inference.drain_encoder_cleanup())
            await asyncio.sleep(0)
            assert not drain.done()
            assert not completed
        finally:
            allow_release.set()
            await asyncio.gather(rpc, return_exceptions=True)
            if drain is not None:
                await drain
    elif ending == "abort_late":
        with pytest.raises(asyncio.CancelledError):
            _ = [value async for value in inference.Generate(req, context)]
    else:
        _ = [value async for value in inference.Generate(req, context)]
    if ending in ("aggregate", "decode"):
        assert not released
        for handle in expected:
            release(handle)
    elif ending == "release_error":
        assert released == expected
        assert completed == expected[1:]
        assert not inference.active_request_count()
        assert inference._encoder_cleanup_failed is epd
        assert OpenEngineControlServicer(llm, "qwen", inference)._engine_is_healthy() is not epd
        release(expected[0])
        params.mrope_position_ids_handle = params.mrope_position_deltas_handle = None
        next_context = FakeServicerContext()
        if epd:
            with pytest.raises(AbortError, match="cleanup failed"):
                _ = [value async for value in inference.Generate(req, next_context)]
            assert next_context.abort_code == servicer.grpc.StatusCode.UNAVAILABLE
        else:
            req.request_id = "later-prefill"
            _ = [value async for value in inference.Generate(req, next_context)]
        assert released == expected
    else:
        assert released == expected
        assert completed == expected
    assert not inference._encoder_cleanup_tasks


@pytest.mark.asyncio
@pytest.mark.parametrize("multimodal,epd", [(False, False), (True, False), (True, True)])
async def test_prefill_cancel_waits_for_late_exports_only_for_multimodal_inputs(
    monkeypatch, multimodal, epd
):
    llm, _ = model()
    llm.args = SimpleNamespace(guided_decoding_backend=None)
    llm.tokenizer = None
    llm._check_health = lambda: True
    waits = []

    class Handle:
        error = None
        disaggregated_params = None

        def __init__(self, cancel):
            self.cancel = cancel
            self.finished = not cancel

        async def __aiter__(self):
            if self.cancel:
                raise asyncio.CancelledError
            yield self

        async def aresult(self):
            waits.append(self)
            raise asyncio.TimeoutError

        def abort(self):
            pass

    handles = iter((Handle(True), Handle(False)))
    llm.generate_async = lambda **kwargs: next(handles)

    async def prepared_input(*args):
        return SimpleNamespace(multimodal_params=SimpleNamespace()) if multimodal else "text"

    monkeypatch.setattr(servicer, "_load_prepared_input", prepared_input)
    monkeypatch.setattr(servicer, "_format_result", lambda *args: [])
    monkeypatch.setattr(servicer, "_engine_error_response", lambda *args: None)
    inference = servicer.OpenEngineInferenceServicer(llm, "qwen")
    assert not inference.encoder_role
    if epd:
        inference.encoder_role = "consume"
    req = request()
    req.ClearField("media")
    req.extra.update({"request_type": "context_only"})
    with pytest.raises(asyncio.CancelledError):
        _ = [value async for value in inference.Generate(req, FakeServicerContext())]
    assert bool(waits) is multimodal
    assert inference._encoder_cleanup_failed is epd
    assert not inference.active_request_count() and not inference._encoder_cleanup_tasks
    assert OpenEngineControlServicer(llm, "qwen", inference)._engine_is_healthy() is not epd
    if not epd:
        req.request_id = "later-prefill"
        _ = [value async for value in inference.Generate(req, FakeServicerContext())]
        assert len(waits) == int(multimodal) and not inference._encoder_cleanup_failed


@pytest.mark.asyncio
async def test_cancelled_encode_reclaims_native_exports(encoder_inference, tmp_path):
    inference, handle, outcomes = encoder_inference
    context = FakeServicerContext()
    context.is_cancelled = True
    with pytest.raises(AbortError):
        await inference.Encode(encode_request(), context)
    assert context.abort_code == encoding.grpc.StatusCode.CANCELLED
    assert outcomes == [("encode", "cancelled")]
    assert handle.disaggregated_params.multimodal_embedding_handles is None
    assert handle.disaggregated_params.mrope_position_ids_handle is None
    assert not list(tmp_path.glob("*.safetensors"))
    assert not inference._encoder_cleanup_tasks


@pytest.mark.asyncio
@pytest.mark.parametrize("role", ["", "encode", "consume"])
@pytest.mark.parametrize("failed", [False, True])
async def test_encoder_discovery_uses_existing_engine_role_only(role, failed):
    inference = _inference()
    inference.encoder_role = role
    inference._encoder_cache = SimpleNamespace(failed=failed)
    inference._media_capabilities = MediaCapabilities({"image": "<image_{}>"})
    control = _servicer(inference=inference)
    info = await control.GetModelInfo(
        model_pb2.GetModelInfoRequest(model=MODEL), FakeServicerContext()
    )
    assert control._engine_is_healthy() is not failed
    assert "trtllm_encoder_cache_release" not in info.extra
    assert "trtllm_requires_encoder_cache" not in info.extra
    server = await control.GetServerInfo(server_pb2.GetServerInfoRequest(), FakeServicerContext())
    assert server.engine_role == (
        server_pb2.ENGINE_ROLE_ENCODER if role == "encode" else server_pb2.ENGINE_ROLE_UNSPECIFIED
    )
    if role == "encode":
        assert info.generation.output_logprobs.supported is False
        assert info.generation.guided_decoding.supported is False


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel_path", ["context", "late", "during_release"])
async def test_encode_export_release_failure_withdraws_readiness(
    tmp_path, monkeypatch, cancel_path
):
    llm, _ = model()
    llm.args = SimpleNamespace(mm_encoder_only=True, guided_decoding_backend=None)
    llm.tokenizer = None
    llm._check_health = lambda: True
    result, _source_tensors = native_result()
    params = result.disaggregated_params
    expected = [
        *params.multimodal_embedding_handles,
        params.mrope_position_ids_handle,
        params.mrope_position_deltas_handle,
    ]
    released, completed = [], []
    entered, release_finished = asyncio.Event(), asyncio.Event()
    allow_release = threading.Event()
    if cancel_path != "during_release":
        allow_release.set()
    loop = asyncio.get_running_loop()
    original_release = SharedTensorContainer.release_dict

    class Handle:
        error = None
        finished = True
        disaggregated_params = params

        async def __aiter__(self):
            if cancel_path == "late":
                raise asyncio.CancelledError
            yield self

        async def aresult(self):
            return self

        def abort(self):
            pass

    handle = Handle()
    llm.generate_async = lambda **kwargs: handle
    monkeypatch.setenv("TRTLLM_OPENENGINE_ENCODER_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(encoding, "_prepare", prepared_encode)

    def release(export):
        released.append(export)
        if export == expected[0]:
            loop.call_soon_threadsafe(entered.set)
            assert allow_release.wait(5)
            raise RuntimeError("release failed")
        original_release(export)
        completed.append(export)
        if export == expected[-1]:
            loop.call_soon_threadsafe(release_finished.set)

    monkeypatch.setattr(SharedTensorContainer, "release_dict", release)
    inference = servicer.OpenEngineInferenceServicer(llm, "qwen")
    req = encode_request()
    context = FakeServicerContext()
    context.is_cancelled = cancel_path != "late"

    async def consume():
        return await inference.Encode(req, context)

    rpc = asyncio.create_task(consume())
    drain = None
    try:
        if cancel_path == "during_release":
            await asyncio.wait_for(entered.wait(), timeout=2)
            assert params.multimodal_embedding_handles is None
            assert params.mrope_position_ids_handle is None
            assert params.mrope_position_deltas_handle is None
            rpc.cancel()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(rpc, timeout=2)
            drain = asyncio.create_task(inference.drain_encoder_cleanup())
            await asyncio.sleep(0)
            assert not drain.done()
            assert not completed
            allow_release.set()
            await asyncio.wait_for(drain, timeout=5)
        elif cancel_path == "late":
            with pytest.raises(asyncio.CancelledError):
                await rpc
        else:
            with pytest.raises(AbortError):
                await rpc
        assert released == expected
        assert completed == expected[1:]
        assert inference._encoder_cleanup_failed
        control = OpenEngineControlServicer(llm, "qwen", inference)
        assert not control._engine_is_healthy()
        with pytest.raises(AbortError, match="cleanup failed"):
            await inference.Encode(req, FakeServicerContext())
        assert not inference.active_request_count() and not inference._encoder_cleanup_tasks
    finally:
        allow_release.set()
        if not rpc.done():
            rpc.cancel()
        await asyncio.gather(rpc, return_exceptions=True)
        if drain is not None:
            await drain
        await inference.drain_encoder_cleanup()
        if released:
            await asyncio.wait_for(release_finished.wait(), timeout=5)
        for export in expected:
            if export not in completed:
                original_release(export)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "cancel_path", ["rpc", "request_id", "all_requests", "reservation", "discard_failure"]
)
async def test_cancel_during_publish_reclaims_the_completed_cache(
    tmp_path, monkeypatch, cancel_path
):
    llm, _ = model()
    result, _ = native_result()
    llm.args = SimpleNamespace(mm_encoder_only=True, guided_decoding_backend=None)
    llm.tokenizer = None

    class Handle:
        disaggregated_params = result.disaggregated_params
        finished = True
        error = None

        async def __aiter__(self):
            yield self

        async def aresult(self):
            return self

        def abort(self):
            pass

    llm.generate_async = lambda **kwargs: Handle()
    monkeypatch.setenv("TRTLLM_OPENENGINE_ENCODER_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(encoding, "_prepare", prepared_encode)
    inference = servicer.OpenEngineInferenceServicer(llm, "qwen")
    if cancel_path == "reservation":
        from test_multimodal import _Frontend

        inference._frontend = _Frontend()
    publish = inference._encoder_cache.publish
    discard = inference._encoder_cache.discard
    entered, reclaimed = asyncio.Event(), asyncio.Event()
    release = threading.Event()
    loop = asyncio.get_running_loop()

    def delayed_publish(*args):
        reference = publish(*args)
        loop.call_soon_threadsafe(entered.set)
        assert release.wait(5)
        return reference

    def record_discard(reference):
        removed = discard(reference)
        loop.call_soon_threadsafe(reclaimed.set)
        if cancel_path == "discard_failure":
            raise OSError("output reclamation could not be confirmed")
        return removed

    monkeypatch.setattr(inference._encoder_cache, "publish", delayed_publish)
    monkeypatch.setattr(inference._encoder_cache, "discard", record_discard)
    req = encode_request()

    async def consume():
        return await inference.Encode(req, FakeServicerContext())

    rpc = asyncio.create_task(consume())
    try:
        await asyncio.wait_for(entered.wait(), 2)
        assert list(tmp_path.glob("*.safetensors"))
        if cancel_path == "rpc":
            rpc.cancel()
            with pytest.raises(asyncio.CancelledError):
                await rpc
        else:
            if cancel_path == "reservation":
                assert inference._frontend.reservations[0].abort()
            else:
                abort = (
                    lifecycle_pb2.AbortRequest(request_id=req.request_id)
                    if cancel_path == "request_id"
                    else lifecycle_pb2.AbortRequest(all_requests=lifecycle_pb2.AllRequests())
                )
                response = await _servicer(inference=inference).Abort(abort, FakeServicerContext())
                assert response.status == lifecycle_pb2.ABORT_STATUS_ABORTED
            assert not rpc.done()
            release.set()
            with pytest.raises(AbortError):
                await asyncio.wait_for(rpc, 2)
    finally:
        release.set()
        await asyncio.gather(rpc, return_exceptions=True)
    await asyncio.wait_for(reclaimed.wait(), 2)
    assert not list(tmp_path.glob("*.safetensors"))
    assert inference._encoder_cleanup_failed is (cancel_path == "discard_failure")
    assert inference.active_request_count() == 0
    if cancel_path == "reservation":
        assert inference._frontend.reservations[0].released


@pytest.mark.asyncio
async def test_abandoned_publication_keeps_rpc_loop_live_and_cleanup_tracked(
    encoder_inference, monkeypatch, tmp_path
):
    inference, _, _ = encoder_inference
    cache = inference._encoder_cache
    context = FakeServicerContext()
    publish, discard = cache.publish, cache.discard
    loop = asyncio.get_running_loop()
    held, unlock, unlocked = threading.Event(), threading.Event(), threading.Event()
    entered = asyncio.Event()
    discarded = []
    available = inference._media_admission._value

    def hold_publication_lock():
        with cache._lock():
            held.set()
            unlock.wait(5)
        unlocked.set()

    holder = threading.Thread(target=hold_publication_lock)

    def cancelled_publication(*args):
        reference = publish(*args)
        holder.start()
        assert held.wait(2)
        context.is_cancelled = True
        return reference

    def blocked_discard(reference):
        loop.call_soon_threadsafe(entered.set)
        result = discard(reference)
        discarded.append(result)
        return result

    monkeypatch.setattr(cache, "publish", cancelled_publication)
    monkeypatch.setattr(cache, "discard", blocked_discard)
    rpc = asyncio.create_task(inference.Encode(encode_request(), context))
    drain = None
    try:
        await asyncio.wait_for(entered.wait(), 2)
        assert not unlocked.is_set(), "discard blocked the RPC event loop"
        assert not rpc.done()
        for _ in range(2):
            rpc.cancel()
            await asyncio.sleep(0)
        drain = asyncio.create_task(inference.drain_encoder_cleanup())
        await asyncio.sleep(0)
        assert not drain.done(), "shutdown must retain pending publication cleanup"
        assert inference._encoder_cleanup_tasks
        unlock.set()
        await asyncio.wait_for(drain, 2)
        await asyncio.gather(rpc, return_exceptions=True)
        assert discarded == [True]
        assert not list(tmp_path.glob("*.safetensors"))
        assert not inference._encoder_cleanup_tasks
        assert inference.active_request_count() == 0
        assert inference._media_admission._value == available
        assert not inference._encoder_cleanup_failed
    finally:
        unlock.set()
        await asyncio.gather(rpc, return_exceptions=True)
        if drain is not None:
            await drain
        await inference.drain_encoder_cleanup()
        if holder.ident is not None:
            holder.join(timeout=2)
            assert not holder.is_alive()


@pytest.mark.asyncio
@pytest.mark.parametrize("input_kind", ["missing", "prompt", "tokens"])
async def test_unary_encode_validates_native_context_and_records_no_token_usage(
    encoder_inference, input_kind
):
    inference, _, outcomes = encoder_inference
    req = encode_request()
    if input_kind == "missing":
        req.ClearField("prompt")
    elif input_kind == "tokens":
        req.token_ids.ids.extend([1, 2])
    context = FakeServicerContext(metadata=(("traceparent", "00-native-encoder-trace"),))
    if input_kind == "missing":
        with pytest.raises(AbortError, match="requires prompt or token IDs"):
            await inference.Encode(req, context)
        assert context.abort_code == encoding.grpc.StatusCode.INVALID_ARGUMENT
        assert outcomes == [("encode", "rejected")]
    else:
        response = await inference.Encode(req, context)
        assert response.request_id == req.request_id
        assert response.encoded_media.opaque_state
        assert outcomes == [("encode", "completed")]
        from tensorrt_llm.grpc.openengine.encoder import native_reference

        assert inference._encoder_cache.discard(native_reference(response.encoded_media))
    await inference.drain_encoder_cleanup()
    assert not inference.active_request_count()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure, status, outcome",
    [
        (
            EncoderCapacityError("full"),
            encoding.grpc.StatusCode.RESOURCE_EXHAUSTED,
            "rejected",
        ),
        (
            EncoderUnavailableError("closed"),
            encoding.grpc.StatusCode.UNAVAILABLE,
            "unavailable",
        ),
        (
            MediaUnavailableError("fetch failed"),
            encoding.grpc.StatusCode.UNAVAILABLE,
            "unavailable",
        ),
        (
            ValueError("published output exceeds limit"),
            encoding.grpc.StatusCode.INVALID_ARGUMENT,
            "rejected",
        ),
        (
            RuntimeError("unexpected engine fault"),
            encoding.grpc.StatusCode.INTERNAL,
            "engine_error",
        ),
    ],
)
async def test_encode_publication_errors_preserve_status_and_metrics(
    encoder_inference, monkeypatch, failure, status, outcome
):
    inference, _, outcomes = encoder_inference

    def publish(*args):
        raise failure

    monkeypatch.setattr(inference._encoder_cache, "publish", publish)
    context = FakeServicerContext()
    with pytest.raises(AbortError):
        await inference.Encode(encode_request(), context)
    assert context.abort_code == status, context.abort_details
    assert outcomes == [("encode", outcome)]
    await inference.drain_encoder_cleanup()
    assert inference.active_request_count() == 0
    assert not inference._encoder_cleanup_failed


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure, status, outcome",
    [
        (
            EncoderCapacityError("full"),
            encoding.grpc.StatusCode.RESOURCE_EXHAUSTED,
            "rejected",
        ),
        (
            EncoderUnavailableError("closed"),
            encoding.grpc.StatusCode.UNAVAILABLE,
            "unavailable",
        ),
        (
            RuntimeError("unexpected import fault"),
            encoding.grpc.StatusCode.INTERNAL,
            "engine_error",
        ),
    ],
)
async def test_consumer_import_errors_preserve_status_and_metrics(
    tmp_path, monkeypatch, failure, status, outcome
):
    llm, _ = model()
    llm.args = SimpleNamespace(disable_mm_encoder=True, guided_decoding_backend=None)
    monkeypatch.setenv("TRTLLM_OPENENGINE_ENCODER_CACHE_DIR", str(tmp_path))
    inference = servicer.OpenEngineInferenceServicer(llm, "qwen")
    assert inference.encoder_role == "consume"
    inference._media_capabilities = MediaCapabilities({"image": "<image_{}>"})
    outcomes = []
    inference._metrics = SimpleNamespace(record_outcome=lambda *value: outcomes.append(value))

    def prepare(*args):
        raise failure

    monkeypatch.setattr(inference._encoder_cache, "prepare", prepare)
    req = request()
    req.encoded_media.opaque_state = b"native-owned-reference"
    context = FakeServicerContext()
    with pytest.raises(AbortError):
        async for _ in inference.Generate(req, context):
            pytest.fail("import failure must not generate output")
    assert context.abort_code == status, context.abort_details
    assert outcomes == [("aggregated", outcome)]
    await inference.drain_encoder_cleanup()
    assert inference.active_request_count() == 0


@pytest.mark.parametrize("invalid_role", [False, True])
def test_nixl_transport_does_not_construct_unused_shared_cache(monkeypatch, invalid_role):
    from tensorrt_llm.grpc.openengine import encoder_nixl

    llm, _ = model()
    llm.args = SimpleNamespace(
        mm_encoder_only=True,
        guided_decoding_backend=None,
        tensor_parallel_size=2 if invalid_role else 1,
    )
    closed = []
    cache = SimpleNamespace(endpoint="encoder.example:30001", close=lambda: closed.append(True))
    monkeypatch.setenv("TRTLLM_OPENENGINE_ENCODER_CACHE_DIR", "/unused-shared-cache")
    monkeypatch.setenv("TRTLLM_OPENENGINE_ENCODER_TRANSFER", "nixl")
    monkeypatch.setattr(encoder_nixl, "NixlEncoderCache", lambda *args, **kwargs: cache)
    monkeypatch.setattr(servicer, "EncoderCache", lambda *args: pytest.fail("unused shared cache"))
    if invalid_role:
        with pytest.raises(ValueError, match="supported role"):
            servicer.OpenEngineInferenceServicer(llm, "qwen")
        assert closed == [True]
    else:
        inference = servicer.OpenEngineInferenceServicer(llm, "qwen")
        assert inference._encoder_cache is cache
        assert inference.encoder_role == "encode"
        assert not closed
