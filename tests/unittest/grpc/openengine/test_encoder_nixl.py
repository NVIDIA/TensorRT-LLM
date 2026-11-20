# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import grpc
import pytest
import torch
from conftest import AbortError, FakeServicerContext
from test_control import _inference, _servicer
from utils.native_encoder import _cache as _native_cache
from utils.native_encoder import _receive, _receiver, _writes, model, native_result, request

from tensorrt_llm._torch.disaggregation import encoder_nixl as transport
from tensorrt_llm._torch.disaggregation.encoder_output import EncodedMediaRef
from tensorrt_llm.grpc.openengine import encoder_nixl as rpc
from tensorrt_llm.grpc.openengine.bindings import encoder_transfer_pb2, encoding_pb2
from tensorrt_llm.grpc.openengine.encoder import portable_reference
from tensorrt_llm.grpc.openengine.encoder_transfer import EncoderTransferServicer
from tensorrt_llm.grpc.openengine.errors import (
    ENCODER_TRANSFER_REJECTED_METADATA,
    EncoderTransferRejected,
)

pytestmark = pytest.mark.cpu_only


def _cache(endpoint=""):
    return _native_cache(endpoint, request_transfer=rpc.request_transfer)


class RpcFailure(grpc.RpcError):
    def __init__(self, message, code=grpc.StatusCode.INVALID_ARGUMENT, metadata=()):
        super().__init__(message)
        self._code = code
        self._metadata = metadata

    def details(self):
        return str(self)

    def code(self):
        return self._code

    def trailing_metadata(self):
        return self._metadata


@pytest.fixture
def pair(monkeypatch):
    producer = _cache("encoder.example:30001")
    consumer = _cache()

    class Channel:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    class Stub:
        def __init__(self, channel):
            pass

        def Transfer(self, req, timeout):
            try:
                producer.transfer(EncodedMediaRef(req.reference), req.destination)
            except transport.EncoderTransferRejected as error:
                raise RpcFailure(
                    str(error), metadata=(ENCODER_TRANSFER_REJECTED_METADATA,)
                ) from error
            return encoder_transfer_pb2.TransferEncoderCacheResponse()

    monkeypatch.setattr(rpc.grpc, "insecure_channel", lambda endpoint: Channel())
    monkeypatch.setattr(rpc.encoder_transfer_pb2_grpc, "EncoderTransferStub", Stub)
    yield producer, consumer
    for cache in (consumer, producer):
        if not cache._closed:
            cache.close()


def _native_encoder_producer(pipe):
    from concurrent.futures import ThreadPoolExecutor

    import grpc

    from tensorrt_llm.grpc.openengine.bindings import (
        encoder_transfer_pb2,
        encoder_transfer_pb2_grpc,
    )

    server = grpc.server(ThreadPoolExecutor(max_workers=2))
    port = server.add_insecure_port("127.0.0.1:0")
    producer = transport.NixlEncoderCache(
        f"127.0.0.1:{port}", "a" * 40, device="cpu", request_transfer=rpc.request_transfer
    )

    class Service(encoder_transfer_pb2_grpc.EncoderTransferServicer):
        def Transfer(self, req, context):
            producer.transfer(EncodedMediaRef(req.reference), req.destination)
            return encoder_transfer_pb2.TransferEncoderCacheResponse()

    encoder_transfer_pb2_grpc.add_EncoderTransferServicer_to_server(Service(), server)
    server.start()
    try:
        llm, _ = model()
        ref = producer.publish(native_result()[0], request(), llm)
        pipe.send((os.getpid(), ref.opaque_state))
        assert pipe.recv() == "stop"
        assert producer._used == 0
    finally:
        server.stop(0).wait()
        producer.close()
        pipe.close()


@pytest.mark.skipif(
    os.environ.get("TRTLLM_TEST_NIXL_ENCODER") != "1",
    reason="opt-in native NIXL/UCX runtime test",
)
def test_native_nixl_between_separate_processes():
    import multiprocessing

    context = multiprocessing.get_context("spawn")
    parent, child = context.Pipe()
    process = context.Process(target=_native_encoder_producer, args=(child,))
    process.start()
    child.close()
    consumer = None
    try:
        assert parent.poll(120), "encoder process did not become ready"
        producer_pid, wire = parent.recv()
        assert producer_pid != os.getpid()
        consumer = transport.NixlEncoderCache(
            "", "a" * 40, device="cpu", request_transfer=rpc.request_transfer
        )
        llm, seen = model()
        consumer.prepare(EncodedMediaRef(wire), request(), llm, None)
        expected = [
            torch.arange(8, dtype=torch.float32).reshape(2, 4),
            torch.arange(6, dtype=torch.int64).reshape(3, 1, 2),
            torch.tensor([[1]]),
        ]
        assert len(seen["tensors"]) == len(expected)
        assert all(torch.equal(a, b) for a, b in zip(seen["tensors"], expected))
        assert consumer._used == 0
        assert seen["metadata"] == {"image_grid_thw": [1, 2, 2]}
        parent.send("stop")
        process.join(20)
        assert process.exitcode == 0
    finally:
        if consumer is not None:
            consumer.close()
        if process.is_alive():
            process.terminate()
            process.join(10)
        parent.close()


@pytest.mark.parametrize(
    "failure", ["value-error", "unmarked-invalid-argument", "marked-unavailable"]
)
def test_unproven_rpc_failure_after_write_quarantines_receiver(pair, monkeypatch, failure):
    producer, consumer = pair
    llm, _ = model()
    reference = producer.publish(native_result()[0], request(), llm)
    original = producer.transfer
    errors = {
        "value-error": ValueError("reply failed after WRITE"),
        "unmarked-invalid-argument": RpcFailure("reply failed after WRITE"),
        "marked-unavailable": RpcFailure(
            "reply failed after WRITE",
            grpc.StatusCode.UNAVAILABLE,
            (ENCODER_TRANSFER_REJECTED_METADATA,),
        ),
    }

    def lose_reply(*args):
        original(*args)
        raise errors[failure]

    monkeypatch.setattr(producer, "transfer", lose_reply)
    with pytest.raises((ValueError, grpc.RpcError), match="after WRITE"):
        _receive(consumer, reference)
    assert _writes(producer) == 1 and not producer.failed
    assert producer._used == 0
    assert not consumer.failed and consumer._used > 0 and _receiver(consumer).registrations
    assert any(
        resource is _receiver(consumer)
        for entry in transport._QUARANTINED
        if isinstance(entry, tuple)
        for resource in entry
    )


def test_real_grpc_marked_rejection_releases_receiver_and_allows_a_fresh_reference():
    server = grpc.server(ThreadPoolExecutor(max_workers=2))
    port = server.add_insecure_port("127.0.0.1:0")
    producer = _cache(f"127.0.0.1:{port}")
    consumer = _cache()

    class Service(rpc.encoder_transfer_pb2_grpc.EncoderTransferServicer):
        def Transfer(self, req, context):
            try:
                producer.transfer(EncodedMediaRef(req.reference), req.destination)
            except transport.EncoderTransferRejected as error:
                context.set_trailing_metadata((ENCODER_TRANSFER_REJECTED_METADATA,))
                context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(error))
            return encoder_transfer_pb2.TransferEncoderCacheResponse()

    rpc.encoder_transfer_pb2_grpc.add_EncoderTransferServicer_to_server(Service(), server)
    server.start()
    try:
        llm, _ = model()
        first = producer.publish(native_result()[0], request(), llm)
        _receive(consumer, first)
        with pytest.raises(transport.EncoderTransferRejected) as error:
            _receive(consumer, first)
        rejection = error.value.__cause__
        assert isinstance(rejection, grpc.RpcError)
        assert rejection.code() == grpc.StatusCode.INVALID_ARGUMENT
        assert ENCODER_TRANSFER_REJECTED_METADATA in rejection.trailing_metadata()
        assert not consumer.failed and not _receiver(consumer).registrations and consumer._used == 0
        second = producer.publish(native_result()[0], request(), llm)
        actual = _receive(consumer, second)
        assert torch.equal(actual[0], torch.arange(8, dtype=torch.float32).reshape(2, 4))
        assert _writes(producer) == 2
    finally:
        consumer.close()
        producer.close()
        assert server.stop(0).wait(5)


@pytest.mark.asyncio
async def test_cancelled_encoder_transfer_keeps_native_work_alive():
    started, finish = threading.Event(), threading.Event()

    def transfer(cache, destination):
        started.set()
        assert finish.wait(10)

    inference = SimpleNamespace(_encoder_cleanup_tasks=set())
    inference.encoder_role = "encode"
    inference._encoder_cache = SimpleNamespace(transfer=transfer, backend=transport.BACKEND)
    transfer_service = EncoderTransferServicer(inference)
    task = asyncio.create_task(
        transfer_service.Transfer(
            encoder_transfer_pb2.TransferEncoderCacheRequest(), FakeServicerContext()
        )
    )
    try:
        assert await asyncio.to_thread(started.wait, 5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert len(inference._encoder_cleanup_tasks) == 1
        assert not next(iter(inference._encoder_cleanup_tasks)).done()
    finally:
        finish.set()
        await asyncio.gather(*tuple(inference._encoder_cleanup_tasks))
    assert not inference._encoder_cleanup_tasks


@pytest.mark.asyncio
@pytest.mark.parametrize("rejected", [True, False])
async def test_transfer_rejection_marker_requires_proven_prewrite_failure(rejected):
    error = EncoderTransferRejected("expired reference") if rejected else ValueError("write failed")

    def transfer(cache, destination):
        raise error

    inference = SimpleNamespace(_encoder_cleanup_tasks=set())
    inference.encoder_role = "encode"
    inference._encoder_cache = SimpleNamespace(transfer=transfer, backend=transport.BACKEND)
    context = FakeServicerContext()
    with pytest.raises(AbortError):
        await EncoderTransferServicer(inference).Transfer(
            encoder_transfer_pb2.TransferEncoderCacheRequest(), context
        )
    assert context.abort_code == (
        grpc.StatusCode.INVALID_ARGUMENT if rejected else grpc.StatusCode.UNAVAILABLE
    )
    assert context.trailing_metadata == ((ENCODER_TRANSFER_REJECTED_METADATA,) if rejected else ())


@pytest.mark.asyncio
async def test_release_rejects_deeply_nested_reference_without_changing_exports(pair):
    producer, _ = pair
    llm, _ = model()
    reference = producer.publish(native_result()[0], request(), llm)
    reserved = producer._used
    inference = _inference()
    inference.encoder_role = "encode"
    inference._encoder_cache = producer
    malformed = EncodedMediaRef(
        opaque_state=b"[" * 10000 + b"0" + b"]" * 10000,
    )
    context = FakeServicerContext()
    with pytest.raises(AbortError):
        await _servicer(inference=inference).ReleaseEncodedMedia(
            encoding_pb2.ReleaseEncodedMediaRequest(encoded_media=portable_reference(malformed)),
            context,
        )
    assert context.abort_code == grpc.StatusCode.INVALID_ARGUMENT
    assert producer._used == reserved and len(producer._exports) == 1
    assert not producer._claims and not producer.failed
    assert producer.discard(reference)
