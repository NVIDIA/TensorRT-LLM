# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import threading
from types import SimpleNamespace

import pytest
from conftest import AbortError, FakeServicerContext
from test_control import MODEL, _inference, _servicer
from test_encoder import model, native_result, request

from tensorrt_llm._torch.shared_tensor import SharedTensorContainer
from tensorrt_llm.disaggregated_params import DisaggregatedParams
from tensorrt_llm.grpc.openengine import servicer
from tensorrt_llm.grpc.openengine.bindings import lifecycle_pb2, model_pb2
from tensorrt_llm.grpc.openengine.capabilities import MediaCapabilities
from tensorrt_llm.grpc.openengine.control import OpenEngineControlServicer

pytestmark = pytest.mark.cpu_only


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
    "ending",
    [
        "done",
        "error",
        "format_error",
        "cancelled",
        "abort_late",
        "cancel_in_release",
        "release_error",
        "aggregate",
        "decode",
    ],
)
async def test_prefill_reclaims_native_mrope_results_on_every_exit(monkeypatch, ending):
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
        assert inference._encoder_cleanup_failed
        assert not OpenEngineControlServicer(llm, "qwen", inference)._engine_is_healthy()
        next_context = FakeServicerContext()
        with pytest.raises(AbortError, match="cleanup failed"):
            _ = [value async for value in inference.Generate(req, next_context)]
        assert next_context.abort_code == servicer.grpc.StatusCode.UNAVAILABLE
        assert released == expected
        release(expected[0])
    else:
        assert released == expected
        assert completed == expected
    assert not inference._encoder_cleanup_tasks


@pytest.mark.asyncio
@pytest.mark.parametrize("multimodal", [False, True])
async def test_prefill_cancel_waits_for_late_exports_only_for_multimodal_inputs(
    monkeypatch, multimodal
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
    req = request()
    req.ClearField("media")
    req.extra.update({"request_type": "context_only"})
    with pytest.raises(asyncio.CancelledError):
        _ = [value async for value in inference.Generate(req, FakeServicerContext())]
    assert bool(waits) is multimodal
    assert inference._encoder_cleanup_failed is multimodal
    assert not inference.active_request_count() and not inference._encoder_cleanup_tasks
    assert OpenEngineControlServicer(llm, "qwen", inference)._engine_is_healthy() is not multimodal
    if not multimodal:
        req.request_id = "later-prefill"
        _ = [value async for value in inference.Generate(req, FakeServicerContext())]
        assert not waits and not inference._encoder_cleanup_failed


@pytest.mark.asyncio
async def test_cancelled_encode_reclaims_native_exports(tmp_path, monkeypatch):
    llm, _ = model()
    result, _ = native_result()
    llm.args = SimpleNamespace(mm_encoder_only=True, guided_decoding_backend=None)
    llm.tokenizer = None

    class Handle:
        disaggregated_params = result.disaggregated_params
        finished = True
        error = None
        aborted = False

        async def __aiter__(self):
            yield self

        async def aresult(self):
            return self

        def abort(self):
            self.aborted = True

    handle = Handle()
    llm.generate_async = lambda **kwargs: handle
    monkeypatch.setenv("TRTLLM_OPENENGINE_ENCODER_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(
        servicer,
        "_prepare_input_from_request",
        lambda *args: SimpleNamespace(media=(), inputs="prompt"),
    )
    context = FakeServicerContext()
    context.is_cancelled = True
    req = request()
    req.encoder.encode_only = True
    inference = servicer.OpenEngineInferenceServicer(llm, "qwen")
    assert [value async for value in inference.Generate(req, context)] == []
    assert handle.aborted
    assert handle.disaggregated_params.multimodal_embedding_handles is None
    assert handle.disaggregated_params.mrope_position_ids_handle is None
    assert not list(tmp_path.glob("*.safetensors"))
    assert not inference._encoder_cleanup_tasks


@pytest.mark.asyncio
@pytest.mark.parametrize("role", ["", "encode", "consume"])
@pytest.mark.parametrize("backend", ["trtllm-shared-safetensors-v1", "trtllm-nixl-encoder-v1"])
async def test_encoder_capability_discovery_is_typed_and_role_specific(role, backend):
    inference = _inference()
    inference.encoder_role = role
    inference._encoder_cache = SimpleNamespace(backend=backend)
    inference._media_capabilities = MediaCapabilities({"image": "<image_{}>"})
    info = await _servicer(inference=inference).GetModelInfo(
        model_pb2.GetModelInfoRequest(model=MODEL), FakeServicerContext()
    )
    assert info.HasField("encoder") is bool(role)
    if role:
        assert info.encoder.schema_version == 1
        assert info.encoder.encoder_only is (role == "encode")
        assert info.encoder.accepts_cache is (role == "consume")
        assert info.encoder.supports_abort_cleanup is (role == "encode")
        assert info.encoder.requires_cache is (role == "consume")
        assert "trtllm_encoder_cache_release" not in info.extra
        assert "trtllm_requires_encoder_cache" not in info.extra
        assert list(info.encoder.transfer_backends) == [backend]
    if role == "encode":
        assert info.generation.output_logprobs.supported is False
        assert info.generation.guided_decoding.supported is False


@pytest.mark.asyncio
async def test_uncertain_encoder_transport_withdraws_health():
    inference = _inference()
    inference._encoder_cache = SimpleNamespace(failed=True)
    assert not _servicer(inference=inference)._engine_is_healthy()


@pytest.mark.asyncio
async def test_cancel_during_publish_reclaims_the_completed_cache(tmp_path, monkeypatch):
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
    monkeypatch.setattr(
        servicer,
        "_prepare_input_from_request",
        lambda *args: SimpleNamespace(media=(), inputs="prompt"),
    )
    inference = servicer.OpenEngineInferenceServicer(llm, "qwen")
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
        return removed

    monkeypatch.setattr(inference._encoder_cache, "publish", delayed_publish)
    monkeypatch.setattr(inference._encoder_cache, "discard", record_discard)
    req = request()
    req.encoder.encode_only = True

    async def consume():
        return [value async for value in inference.Generate(req, FakeServicerContext())]

    rpc = asyncio.create_task(consume())
    try:
        await asyncio.wait_for(entered.wait(), 2)
        assert list(tmp_path.glob("*.safetensors"))
        rpc.cancel()
        with pytest.raises(asyncio.CancelledError):
            await rpc
    finally:
        release.set()
        await asyncio.gather(rpc, return_exceptions=True)
    await asyncio.wait_for(reclaimed.wait(), 2)
    assert not list(tmp_path.glob("*.safetensors"))
    assert not inference._encoder_cleanup_failed
