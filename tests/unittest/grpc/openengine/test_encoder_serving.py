# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import threading
from types import SimpleNamespace

import pytest
from conftest import AbortError, FakeServicerContext
from test_encoder import model, native_result, request

from tensorrt_llm._torch.shared_tensor import SharedTensorContainer
from tensorrt_llm.disaggregated_params import DisaggregatedParams
from tensorrt_llm.grpc.openengine import servicer
from tensorrt_llm.grpc.openengine.control import OpenEngineControlServicer

pytestmark = pytest.mark.cpu_only


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
