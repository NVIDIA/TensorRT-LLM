# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import ctypes
import json
import os
from types import SimpleNamespace

import pytest
import torch
from test_encoder import model, native_result, request

from tensorrt_llm.disaggregated_params import DisaggregatedParams
from tensorrt_llm.grpc.openengine import encoder_nixl as transport
from tensorrt_llm.grpc.openengine.bindings import generation_pb2 as pb

pytestmark = pytest.mark.cpu_only


class FakeAgent:
    agents = {}

    def __init__(self, name):
        self.name = name
        self.registration = None
        self.writes = 0
        self.complete = True
        self.stopped = False
        self.agents[name] = self

    def register_memory(self, desc):
        self.registration = desc

    def deregister_memory(self, desc):
        assert self.registration is desc
        self.registration = None

    def get_local_agent_desc(self):
        return self.name.encode()

    def load_remote_agent(self, name, desc):
        assert desc == name.encode()

    def invalidate_remote_agent(self, name):
        pass

    def submit_transfer_requests(self, req):
        self.writes += 1
        peer = self.agents[req.remote_name]
        for source, target in zip(self.registration.descs, peer.registration.descs):
            ctypes.memmove(target[0], source[0], source[1])
        return SimpleNamespace(wait=lambda timeout: self.complete)

    def shutdown(self):
        self.stopped = True


@pytest.fixture
def pair(monkeypatch):
    producer = transport.NixlEncoderCache(
        "encoder.example:30001", "a" * 40, device="cpu", agent_factory=FakeAgent
    )
    consumer = transport.NixlEncoderCache("", "a" * 40, device="cpu", agent_factory=FakeAgent)

    class Channel:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    class Stub:
        def __init__(self, channel):
            pass

        def TransferEncoderCache(self, req, timeout):
            producer.transfer(req.cache, req.destination)
            return pb.TransferEncoderCacheResponse()

    monkeypatch.setattr(transport.grpc, "insecure_channel", lambda endpoint: Channel())
    monkeypatch.setattr(transport.openengine_pb2_grpc, "ControlStub", Stub)
    yield producer, consumer
    producer.close()
    consumer.close()


def test_nixl_roundtrip_retains_tensors_metadata_and_mrope(pair):
    producer, consumer = pair
    llm, seen = model()
    output, expected = native_result()
    ref = producer.publish(output, request(), llm)
    assert ref.transfer_backend == transport.BACKEND
    assert "host" not in json.loads(ref.opaque_state)
    assert not output.disaggregated_params.multimodal_embedding_handles
    _, params = consumer.prepare(ref, request(), llm, None)
    assert params.request_type == "context_and_generation"
    assert seen["metadata"] == {"image_grid_thw": [1, 2, 2]}
    assert all(torch.equal(a, b) for a, b in zip(seen["tensors"], expected))
    assert producer.agent.writes == 1
    assert producer._used == consumer._used == 0
    assert not producer._exports
    with pytest.raises(ValueError, match="already consumed"):
        consumer.prepare(ref, request(), llm, None)
    assert producer.agent.writes == 1


def test_nixl_encoder_handoff_preserves_prefill_phase(pair):
    producer, consumer = pair
    llm, seen = model()
    ref = producer.publish(native_result()[0], request(), llm)
    phase = DisaggregatedParams(request_type="context_only")
    _, params = consumer.prepare(ref, request(), llm, None, phase)
    assert params is phase
    assert params.request_type == "context_only"
    assert seen["hashes"] == [[2147483647] * 8]
    assert producer.agent.writes == 1
    assert producer._used == consumer._used == 0


def test_prefill_captures_mrope_when_native_embedding_path_bypasses_processor(pair):
    from tensorrt_llm._torch.models.modeling_qwen2vl import Qwen2VLInputProcessorBase
    from tensorrt_llm.inputs.registry import BaseMultimodalInputProcessor

    class Processor(SimpleNamespace):
        _decode_state_capture = BaseMultimodalInputProcessor._decode_state_capture
        capture_decode_state = BaseMultimodalInputProcessor.capture_decode_state
        _capture_prepared_decode = BaseMultimodalInputProcessor._capture_prepared_decode
        export_decode_state = Qwen2VLInputProcessorBase.export_decode_state

    producer, consumer = pair
    llm, _ = model()
    llm.input_processor = Processor(**vars(llm.input_processor))
    output, tensors = native_result()
    ref = producer.publish(output, request(), llm)
    preprocess = llm.preprocess

    def prepare(*args):
        inputs = preprocess(*args)
        inputs.prompt_token_ids = [11, 12]
        return inputs

    llm.preprocess = prepare
    phase = DisaggregatedParams(request_type="context_only")
    with llm.input_processor.capture_decode_state() as captured:
        consumer.prepare(ref, request(), llm, None, phase)
    assert captured == {
        "prompt_token_ids": [11, 12],
        "state": llm.input_processor.export_decode_state(
            {
                "mrope_config": {
                    "mrope_position_ids": tensors[1],
                    "mrope_position_deltas": tensors[2],
                }
            }
        ),
    }


def test_failed_native_decode_capture_reclaims_prepared_inputs(pair):
    producer, consumer = pair
    llm, _ = model()
    ref = producer.publish(native_result()[0], request(), llm)
    preprocess = llm.preprocess
    released = []

    def prepare(*args):
        inputs = preprocess(*args)
        inputs.prompt_token_ids = [11, 12]
        inputs.multimodal_params = SimpleNamespace(release_handles=lambda: released.append(True))
        return inputs

    def broken_capture(_):
        raise ValueError("invalid decode state")

    llm.preprocess = prepare
    llm.input_processor._capture_prepared_decode = broken_capture
    with pytest.raises(ValueError, match="invalid decode state"):
        consumer.prepare(
            ref, request(), llm, None, DisaggregatedParams(request_type="context_only")
        )
    assert released == [True]
    assert producer._used == consumer._used == 0
    assert not producer._exports


def _consume_worker_exports(pipe):
    from tensorrt_llm._torch.shared_tensor import SharedTensorContainer

    handles = pipe.recv()
    tensors = [SharedTensorContainer.from_dict(h).get_local_view() for h in handles]
    pipe.send((os.getpid(), [tensor.tolist() for tensor in tensors]))
    pipe.close()


def test_nixl_receive_exports_survive_until_another_worker_consumes_them(pair):
    import multiprocessing

    producer, consumer = pair
    llm, _ = model()
    output, expected = native_result()
    ref = producer.publish(output, request(), llm)
    exports = []

    def preprocess(inputs, sampling, params):
        exports.extend(
            [
                *params.multimodal_embedding_handles,
                params.mrope_position_ids_handle,
                params.mrope_position_deltas_handle,
            ]
        )
        return SimpleNamespace(multimodal_params=None)

    llm.preprocess = preprocess
    consumer.prepare(ref, request(), llm, None)
    context = multiprocessing.get_context("spawn")
    parent, child = context.Pipe()
    process = context.Process(target=_consume_worker_exports, args=(child,))
    process.start()
    child.close()
    try:
        parent.send(exports)
        assert parent.poll(60), "worker did not consume the receiver exports"
        worker_pid, actual = parent.recv()
        assert worker_pid != os.getpid()
        assert actual == [tensor.tolist() for tensor in expected]
        process.join(10)
        assert process.exitcode == 0
    finally:
        if process.is_alive():
            process.terminate()
            process.join(10)
        parent.close()
    assert producer._used == consumer._used == 0
    assert not producer._exports


def test_different_input_or_revision_is_rejected_before_transfer(pair):
    producer, consumer = pair
    llm, _ = model()
    ref = producer.publish(native_result()[0], request(), llm)
    altered = request()
    altered.prompt = "another prompt"
    with pytest.raises(ValueError, match="request inputs"):
        consumer.prepare(ref, altered, llm, None)
    consumer.model_revision = "b" * 40
    with pytest.raises(ValueError, match="model revision"):
        consumer.prepare(ref, request(), llm, None)
    assert producer.agent.writes == 0


def test_expired_reference_cannot_access_source_memory(pair, monkeypatch):
    producer, consumer = pair
    llm, _ = model()
    ref = producer.publish(native_result()[0], request(), llm)
    now = transport.time.monotonic()
    monkeypatch.setattr(transport.time, "monotonic", lambda: now + 301)
    with pytest.raises(ValueError, match="expired"):
        consumer.prepare(ref, request(), llm, None)
    assert producer.agent.writes == 0
    assert producer._used == 0


def test_unfinished_write_quarantines_both_sides(pair):
    producer, consumer = pair
    llm, _ = model()
    ref = producer.publish(native_result()[0], request(), llm)
    producer.agent.complete = False
    with pytest.raises(RuntimeError, match="did not complete"):
        consumer.prepare(ref, request(), llm, None)
    assert producer.failed and consumer.failed
    assert producer.agent.registration is not None
    assert consumer.agent.registration is not None
    assert producer._used > 0 and consumer._used > 0
    producer.close()
    consumer.close()
    assert not producer.agent.stopped and not consumer.agent.stopped


def test_lost_success_reply_retains_destination_buffers(pair, monkeypatch):
    producer, consumer = pair
    llm, _ = model()
    ref = producer.publish(native_result()[0], request(), llm)
    original = producer.transfer

    def lost_reply(*args):
        original(*args)
        raise RuntimeError("connection lost after WRITE")

    monkeypatch.setattr(producer, "transfer", lost_reply)
    with pytest.raises(RuntimeError, match="connection lost"):
        consumer.prepare(ref, request(), llm, None)
    assert not producer.failed
    assert consumer.failed and consumer.agent.registration is not None
    assert producer._used == 0


def test_bad_destination_size_does_not_consume_export(pair):
    producer, _ = pair
    llm, _ = model()
    ref = producer.publish(native_result()[0], request(), llm)
    with pytest.raises(ValueError, match="descriptors"):
        producer.transfer(ref, b'{"descs": []}')
    assert len(producer._exports) == 1
    assert not producer.failed


def test_reference_does_not_expose_source_addresses(pair):
    producer, _ = pair
    llm, _ = model()
    ref = producer.publish(native_result()[0], request(), llm)
    state = json.loads(ref.opaque_state)
    assert set(state) == {"version", "id", "endpoint", "metadata"}
    assert "ptr" not in state["metadata"]


def test_close_reclaims_undelivered_exports(pair):
    producer, _ = pair
    llm, _ = model()
    producer.publish(native_result()[0], request(), llm)
    producer.close()
    assert producer.agent.stopped
    assert producer._used == 0


def test_receiver_quota_and_dtype_validation_precede_allocation(pair):
    producer, consumer = pair
    llm, _ = model()
    ref = producer.publish(native_result()[0], request(), llm)
    state = json.loads(ref.opaque_state)
    state["metadata"]["tensors"][0]["shape"] = [2**30]
    oversized = pb.EncoderCacheRef(
        transfer_backend=transport.BACKEND, opaque_state=json.dumps(state).encode()
    )
    with pytest.raises(ValueError, match="memory limit"):
        consumer.prepare(oversized, request(), llm, None)
    assert producer.agent.writes == 0


def test_requires_explicit_revision():
    with pytest.raises(ValueError, match="pinned model revision"):
        transport.NixlEncoderCache("", "", device="cpu", agent_factory=FakeAgent)


@pytest.mark.skipif(
    os.environ.get("TRTLLM_TEST_NIXL_ENCODER") != "1",
    reason="opt-in native NIXL/UCX runtime test",
)
def test_native_nixl_over_real_grpc_loopback():
    from concurrent.futures import ThreadPoolExecutor

    import grpc

    from tensorrt_llm.grpc.openengine.bindings import openengine_pb2_grpc

    server = grpc.server(ThreadPoolExecutor(max_workers=2))
    port = server.add_insecure_port("127.0.0.1:0")
    producer = transport.NixlEncoderCache(f"127.0.0.1:{port}", "a" * 40, device="cpu")
    consumer = transport.NixlEncoderCache("", "a" * 40, device="cpu")

    class Service(openengine_pb2_grpc.ControlServicer):
        def TransferEncoderCache(self, req, context):
            producer.transfer(req.cache, req.destination)
            return pb.TransferEncoderCacheResponse()

    openengine_pb2_grpc.add_ControlServicer_to_server(Service(), server)
    server.start()
    try:
        llm, seen = model()
        output, expected = native_result()
        ref = producer.publish(output, request(), llm)
        consumer.prepare(ref, request(), llm, None)
        assert all(torch.equal(a, b) for a, b in zip(seen["tensors"], expected))
        assert producer._used == consumer._used == 0
    finally:
        server.stop(0).wait()
        consumer.close()
        producer.close()


def _native_encoder_producer(pipe):
    from concurrent.futures import ThreadPoolExecutor

    import grpc

    from tensorrt_llm.grpc.openengine.bindings import openengine_pb2_grpc

    server = grpc.server(ThreadPoolExecutor(max_workers=2))
    port = server.add_insecure_port("127.0.0.1:0")
    producer = transport.NixlEncoderCache(f"127.0.0.1:{port}", "a" * 40, device="cpu")

    class Service(openengine_pb2_grpc.ControlServicer):
        def TransferEncoderCache(self, req, context):
            producer.transfer(req.cache, req.destination)
            return pb.TransferEncoderCacheResponse()

    openengine_pb2_grpc.add_ControlServicer_to_server(Service(), server)
    server.start()
    try:
        llm, _ = model()
        ref = producer.publish(native_result()[0], request(), llm)
        pipe.send((os.getpid(), ref.SerializeToString()))
        assert pipe.recv() == "stop"
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
        consumer = transport.NixlEncoderCache("", "a" * 40, device="cpu")
        llm, seen = model()
        consumer.prepare(pb.EncoderCacheRef.FromString(wire), request(), llm, None)
        expected = [
            torch.arange(8, dtype=torch.float32).reshape(2, 4),
            torch.arange(6, dtype=torch.int64).reshape(3, 1, 2),
            torch.tensor([[1]]),
        ]
        assert all(torch.equal(a, b) for a, b in zip(seen["tensors"], expected))
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
