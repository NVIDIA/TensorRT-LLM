# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import base64
import ctypes
import json
import os
import threading
import weakref
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import grpc
import pytest
import torch
from conftest import AbortError, FakeServicerContext
from test_encoder import (
    _check_handoff_hash_initialization,
    _check_malformed_handoff_hashes,
    model,
    native_result,
    request,
)

from tensorrt_llm.disaggregated_params import DisaggregatedParams
from tensorrt_llm.grpc.openengine import encoder_nixl as transport
from tensorrt_llm.grpc.openengine.bindings import encoder_transfer_pb2
from tensorrt_llm.grpc.openengine.bindings import generation_pb2 as pb
from tensorrt_llm.grpc.openengine.encoder_transfer import EncoderTransferServicer
from tensorrt_llm.grpc.openengine.errors import (
    ENCODER_TRANSFER_REJECTED_METADATA,
    EncoderTransferRejected,
)

pytestmark = pytest.mark.cpu_only


class FakeAgent:
    agents = {}

    def __init__(self, name):
        self.name = name
        self.registration = None
        self.registrations = {}
        self.loaded = set()
        self.writes = 0
        self.complete = True
        self.stopped = False
        self.agents[name] = self

    def register_memory(self, desc):
        self.registrations[id(desc)] = desc
        self.registration = desc

    def deregister_memory(self, desc):
        assert self.registrations.pop(id(desc)) is desc
        self.registration = next(iter(self.registrations.values()), None)

    def get_local_agent_desc(self):
        return self.name.encode()

    def load_remote_agent(self, name, desc):
        assert desc == name.encode()
        assert name not in self.loaded, "peer metadata replaced before WRITE completed"
        self.loaded.add(name)

    def invalidate_remote_agent(self, name):
        self.loaded.discard(name)

    def check_remote_descs(self, name, descs):
        peer = self.agents[name]
        targets = [
            desc if isinstance(desc, tuple) else (desc.addr, desc.len, desc.device_id)
            for desc in descs.descs
        ]
        return all(
            any(
                source[0] <= target[0]
                and target[0] + target[1] <= source[0] + source[1]
                and source[2] == target[2]
                for reg in peer.registrations.values()
                for source in reg.descs
            )
            for target in targets
        )

    def submit_transfer_requests(self, req):
        self.writes += 1
        peer = self.agents[req.remote_name]
        sources = req.src_descs.descs
        targets = req.dst_descs.descs
        assert len(sources) == len(targets) > 0
        for source, target in zip(sources, targets):
            source = source if isinstance(source, tuple) else (source.addr, source.len)
            target = target if isinstance(target, tuple) else (target.addr, target.len)
            assert any(
                source[:2] == desc[:2] for reg in self.registrations.values() for desc in reg.descs
            )
            assert any(
                target[:2] == desc[:2] for reg in peer.registrations.values() for desc in reg.descs
            )
            ctypes.memmove(target[0], source[0], source[1])
        return SimpleNamespace(wait=lambda timeout: self.complete)

    def shutdown(self):
        self.stopped = True
        self.registrations.clear()
        self.registration = None
        self.loaded.clear()


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


def _cache(endpoint=""):
    agents = []

    def factory(name):
        agent = FakeAgent(name)
        agents.append(agent)
        return agent

    cache = transport.NixlEncoderCache(endpoint, "a" * 40, device="cpu", agent_factory=factory)
    agents.clear()
    cache.test_agents = agents
    return cache


def _receiver(cache):
    return cache.test_agents[-1]


def _sender(cache):
    return cache.test_agents[-1]


def _writes(cache):
    return sum(agent.writes for agent in cache.test_agents)


@pytest.mark.parametrize("supplied", [False, True])
def test_nixl_handoff_restores_hash_validation_and_missing_hash_fallback(pair, supplied):
    _check_handoff_hash_initialization(*pair, supplied)


@pytest.mark.parametrize("hashes", [[], [[1] * 7], [[1.0] * 8], ["bad"], 42, [[1] * 8] * 2])
def test_nixl_handoff_rejects_malformed_hashes_and_releases_exports(pair, monkeypatch, hashes):
    _check_malformed_handoff_hashes(*pair, hashes, monkeypatch)


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
                producer.transfer(req.cache, req.destination)
            except transport.EncoderTransferRejected as error:
                raise RpcFailure(
                    str(error), metadata=(transport.ENCODER_TRANSFER_REJECTED_METADATA,)
                ) from error
            return encoder_transfer_pb2.TransferEncoderCacheResponse()

    monkeypatch.setattr(transport.grpc, "insecure_channel", lambda endpoint: Channel())
    monkeypatch.setattr(transport.encoder_transfer_pb2_grpc, "EncoderTransferStub", Stub)
    yield producer, consumer
    for cache in (consumer, producer):
        if not cache._closed:
            cache.close()


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
    assert _writes(producer) == 1
    assert producer._used == consumer._used == 0
    assert not producer._exports
    with pytest.raises(transport.EncoderTransferRejected, match="already consumed"):
        consumer.prepare(ref, request(), llm, None)
    assert _writes(producer) == 1


def test_nixl_encoder_handoff_preserves_prefill_phase(pair):
    producer, consumer = pair
    llm, seen = model()
    ref = producer.publish(native_result()[0], request(), llm)
    phase = DisaggregatedParams(
        request_type="context_only", disagg_request_id=71, ctx_usage={"prompt_tokens": 88}
    )
    _, params = consumer.prepare(ref, request(), llm, None, phase)
    assert params.request_type == "context_only"
    assert params.disagg_request_id == phase.disagg_request_id
    assert params.ctx_usage == phase.ctx_usage
    assert seen["hashes"] == [[2147483647] * 8]
    assert _writes(producer) == 1
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
    assert _writes(producer) == 0


def test_expired_reference_cannot_access_source_memory(pair, monkeypatch):
    producer, consumer = pair
    llm, _ = model()
    ref = producer.publish(native_result()[0], request(), llm)
    now = transport.time.monotonic()
    monkeypatch.setattr(transport.time, "monotonic", lambda: now + 301)
    with pytest.raises(transport.EncoderTransferRejected, match="expired"):
        consumer.prepare(ref, request(), llm, None)
    assert _writes(producer) == 0
    assert producer._used == 0


def test_unfinished_write_quarantines_both_sides(pair, monkeypatch):
    producer, consumer = pair
    llm, _ = model()
    ref = producer.publish(native_result()[0], request(), llm)
    original = FakeAgent.submit_transfer_requests

    def incomplete(agent, req):
        original(agent, req)
        return SimpleNamespace(wait=lambda timeout: False)

    monkeypatch.setattr(FakeAgent, "submit_transfer_requests", incomplete)
    with pytest.raises(RuntimeError, match="did not complete"):
        consumer.prepare(ref, request(), llm, None)
    assert producer.failed and not consumer.failed
    assert _sender(producer).registration is not None
    assert _receiver(consumer).registration is not None
    assert producer._used > 0 and consumer._used > 0
    producer.close()
    consumer.close()
    assert not _sender(producer).stopped and not _receiver(consumer).stopped


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
    assert not consumer.failed and _receiver(consumer).registration is not None
    assert producer._used == 0


@pytest.mark.parametrize("code", [grpc.StatusCode.UNAVAILABLE, grpc.StatusCode.DEADLINE_EXCEEDED])
def test_unreachable_peer_quarantines_only_its_receive_and_late_write_remains_safe(
    pair, monkeypatch, code
):
    producer, consumer = pair
    llm, _ = model()
    reference = producer.publish(native_result()[0], request(), llm)
    pending = []

    def fail_transfer(req, timeout):
        pending.append(req)
        raise RpcFailure("encoder endpoint unavailable", code)

    with monkeypatch.context() as patch:
        patch.setattr(
            transport.encoder_transfer_pb2_grpc,
            "EncoderTransferStub",
            lambda channel: SimpleNamespace(Transfer=fail_transfer),
        )
        with pytest.raises(grpc.RpcError, match="endpoint unavailable"):
            _receive(consumer, reference)
    pinned = _receiver(consumer)
    pinned_bytes = consumer._used
    assert pinned.registrations and not pinned.stopped
    assert pinned_bytes > 0 and consumer._receive_slots == 1 and not consumer.failed

    fresh = producer.publish(native_result()[0], request(), llm)
    actual = _receive(consumer, fresh)
    assert _receiver(consumer) is not pinned and _receiver(consumer).stopped
    assert consumer._used == pinned_bytes and consumer._receive_slots == 1
    producer.transfer(pending[0].cache, pending[0].destination)
    assert torch.equal(actual[0], torch.arange(8, dtype=torch.float32).reshape(2, 4))
    consumer.close()
    assert pinned.registrations and not pinned.stopped
    assert consumer._used == pinned_bytes


@pytest.mark.parametrize("limit", ["bytes", "slots"])
def test_quarantined_receives_remain_charged_and_bound_future_allocations(pair, monkeypatch, limit):
    producer, consumer = pair
    llm, _ = model()
    reference = producer.publish(native_result()[0], request(), llm)
    original = producer.transfer

    def lost_reply(*args):
        original(*args)
        raise RpcFailure("lost reply", grpc.StatusCode.UNAVAILABLE)

    with monkeypatch.context() as patch:
        patch.setattr(producer, "transfer", lost_reply)
        with pytest.raises(grpc.RpcError, match="lost reply"):
            _receive(consumer, reference)
    pinned = _receiver(consumer)
    used, agents = consumer._used, len(consumer.test_agents)
    fresh = producer.publish(native_result()[0], request(), llm)
    if limit == "bytes":
        monkeypatch.setattr(transport, "_MAX_CACHE_BYTES", used * 2 - 1)
        message = "cache is full"
    else:
        monkeypatch.setattr(transport, "_MAX_RECEIVE_SLOTS", 1)
        message = "slots exhausted"
    with pytest.raises(RuntimeError, match=message):
        _receive(consumer, fresh)
    assert consumer._used == used and consumer._receive_slots == 1
    assert len(consumer.test_agents) == agents and not consumer.failed
    assert pinned.registrations and not pinned.stopped
    assert json.loads(fresh.opaque_state)["id"] in producer._exports


@pytest.mark.parametrize("operation", ["register_memory", "deregister_memory"])
def test_receive_registration_failure_does_not_disable_unrelated_handoffs(
    pair, monkeypatch, operation
):
    producer, consumer = pair
    llm, _ = model()
    reference = producer.publish(native_result()[0], request(), llm)
    original = getattr(FakeAgent, operation)

    def fail_receive(agent, registration):
        if "-receive-" in agent.name:
            if operation == "register_memory":
                original(agent, registration)
            raise RuntimeError("receive registration state uncertain")
        return original(agent, registration)

    with monkeypatch.context() as patch:
        patch.setattr(FakeAgent, operation, fail_receive)
        with pytest.raises(RuntimeError, match="state uncertain"):
            _receive(consumer, reference)
    pinned = _receiver(consumer)
    size = consumer._used
    assert size > 0 and pinned.registrations and not pinned.stopped
    assert not consumer.failed and consumer._receive_slots == 1
    fresh = producer.publish(native_result()[0], request(), llm)
    actual = _receive(consumer, fresh)
    assert torch.equal(actual[0], torch.arange(8, dtype=torch.float32).reshape(2, 4))
    assert consumer._used == size and consumer._receive_slots == 1


@pytest.mark.parametrize("limit", ["bytes", "slots"])
def test_exhausted_quarantine_budget_marks_cache_unhealthy_without_freeing_buffers(
    pair, monkeypatch, limit
):
    producer, consumer = pair
    llm, _ = model()
    reference = producer.publish(native_result()[0], request(), llm)
    size = producer._used
    if limit == "bytes":
        monkeypatch.setattr(transport, "_MAX_CACHE_BYTES", size)
    else:
        monkeypatch.setattr(transport, "_MAX_RECEIVE_SLOTS", 1)

    def fail_transfer(req, timeout):
        raise RpcFailure("lost peer", grpc.StatusCode.UNAVAILABLE)

    monkeypatch.setattr(
        transport.encoder_transfer_pb2_grpc,
        "EncoderTransferStub",
        lambda channel: SimpleNamespace(Transfer=fail_transfer),
    )
    with pytest.raises(grpc.RpcError, match="lost peer"):
        _receive(consumer, reference)
    pinned = _receiver(consumer)
    assert consumer.failed and consumer._used == size and pinned.registrations
    with pytest.raises(RuntimeError, match="transport unavailable"):
        _receive(consumer, reference)
    consumer.close()
    assert pinned.registrations and not pinned.stopped


def test_bad_destination_size_does_not_consume_export(pair):
    producer, _ = pair
    llm, _ = model()
    ref = producer.publish(native_result()[0], request(), llm)
    with pytest.raises(ValueError, match="descriptors"):
        producer.transfer(ref, b'{"descs": []}')
    assert len(producer._exports) == 1
    assert not producer.failed


@pytest.mark.parametrize("invalid", ["address", "device"])
def test_unregistered_destination_preserves_reference_for_retry(pair, monkeypatch, invalid):
    producer, consumer = pair
    llm, _ = model()
    reference = producer.publish(native_result()[0], request(), llm)
    original = transport._descriptors

    def unregistered(tensors, name):
        registration, source, descs = original(tensors, name)
        if "-receive-" in name:
            descs = [list(desc) for desc in descs]
            if invalid == "address":
                descs[0][0] += descs[0][1]
            else:
                descs[0][2] += 1
        return registration, source, descs

    with monkeypatch.context() as patch:
        patch.setattr(transport, "_descriptors", unregistered)
        with pytest.raises(EncoderTransferRejected, match="not registered"):
            _receive(consumer, reference)
    assert not producer.failed and not consumer.failed
    assert _writes(producer) == 0
    assert json.loads(reference.opaque_state)["id"] in producer._exports
    assert _sender(producer).stopped and not _sender(producer).registrations
    assert consumer._used == 0 and _receiver(consumer).stopped
    actual = _receive(consumer, reference)
    assert torch.equal(actual[0], torch.arange(8, dtype=torch.float32).reshape(2, 4))
    assert producer._used == consumer._used == 0


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
    assert not producer.test_agents
    assert producer._used == 0


def test_receiver_quota_and_dtype_validation_precede_allocation(pair):
    producer, consumer = pair
    llm, _ = model()
    ref = producer.publish(native_result()[0], request(), llm)
    state = json.loads(ref.opaque_state)
    state["metadata"]["tensors"][0]["shape"] = [2**30]
    oversized = pb.EncoderHandoffRef(
        transfer_backend=transport.BACKEND, opaque_state=json.dumps(state).encode()
    )
    with pytest.raises(ValueError, match="memory limit"):
        consumer.prepare(oversized, request(), llm, None)
    assert _writes(producer) == 0


def test_requires_explicit_revision():
    with pytest.raises(ValueError, match="pinned model revision"):
        transport.NixlEncoderCache("", "", device="cpu", agent_factory=FakeAgent)


@pytest.mark.parametrize(
    "state,during_claim",
    [("failed", False), ("closing", False), ("closed", False), ("failed", True), ("closing", True)],
)
def test_unhealthy_producer_rejects_before_write_without_poisoning_receiver(
    pair, monkeypatch, state, during_claim
):
    producer, consumer = pair
    llm, _ = model()
    reference = producer.publish(native_result()[0], request(), llm)
    health = producer._healthy
    checks = 0
    attribute = {"failed": "failed", "closing": "_closing", "closed": "_closed"}[state]
    with monkeypatch.context() as patch:
        if state == "closed":
            producer.close()
        elif not during_claim:
            patch.setattr(producer, attribute, True)
        else:

            def change_health(*args, **kwargs):
                nonlocal checks
                checks += 1
                if checks == 2:
                    patch.setattr(producer, attribute, True)
                return health(*args, **kwargs)

            patch.setattr(producer, "_healthy", change_health)
        with pytest.raises(transport.EncoderTransferRejected, match="transport unavailable"):
            _receive(consumer, reference)
        assert not consumer.failed
        assert consumer._used == 0 and not _receiver(consumer).registrations
        assert _writes(producer) == 0
        if during_claim:
            assert checks == 2
    consumer._healthy()


@pytest.mark.skipif(
    os.environ.get("TRTLLM_TEST_NIXL_ENCODER") != "1",
    reason="opt-in native NIXL/UCX runtime test",
)
def test_native_nixl_over_real_grpc_loopback():
    from concurrent.futures import ThreadPoolExecutor

    import grpc

    from tensorrt_llm.grpc.openengine.bindings import (
        encoder_transfer_pb2,
        encoder_transfer_pb2_grpc,
    )

    server = grpc.server(ThreadPoolExecutor(max_workers=2))
    port = server.add_insecure_port("127.0.0.1:0")
    producer = transport.NixlEncoderCache(f"127.0.0.1:{port}", "a" * 40, device="cpu")
    consumer = transport.NixlEncoderCache("", "a" * 40, device="cpu")

    class Service(encoder_transfer_pb2_grpc.EncoderTransferServicer):
        def Transfer(self, req, context):
            producer.transfer(req.cache, req.destination)
            return encoder_transfer_pb2.TransferEncoderCacheResponse()

    encoder_transfer_pb2_grpc.add_EncoderTransferServicer_to_server(Service(), server)
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

    from tensorrt_llm.grpc.openengine.bindings import (
        encoder_transfer_pb2,
        encoder_transfer_pb2_grpc,
    )

    server = grpc.server(ThreadPoolExecutor(max_workers=2))
    port = server.add_insecure_port("127.0.0.1:0")
    producer = transport.NixlEncoderCache(f"127.0.0.1:{port}", "a" * 40, device="cpu")

    class Service(encoder_transfer_pb2_grpc.EncoderTransferServicer):
        def Transfer(self, req, context):
            producer.transfer(req.cache, req.destination)
            return encoder_transfer_pb2.TransferEncoderCacheResponse()

    encoder_transfer_pb2_grpc.add_EncoderTransferServicer_to_server(Service(), server)
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
        consumer.prepare(pb.EncoderHandoffRef.FromString(wire), request(), llm, None)
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


def _receive(cache, reference):
    with cache._operation():
        return cache._receive(reference, cache._reference(reference))


def test_preflight_rejection_restores_reference_and_receiver_is_reusable(pair, monkeypatch):
    producer, consumer = pair
    llm, _ = model()
    reference = producer.publish(native_result()[0], request(), llm)
    original = FakeAgent.load_remote_agent

    def reject_peer(agent, name, desc):
        agent.loaded.add("partially-imported-under-another-name")
        raise ValueError("receiver metadata cannot be imported")

    monkeypatch.setattr(FakeAgent, "load_remote_agent", reject_peer)
    with pytest.raises(
        transport.EncoderTransferRejected, match="metadata cannot be imported"
    ) as error:
        _receive(consumer, reference)
    rejection = error.value.__cause__
    assert isinstance(rejection, grpc.RpcError)
    assert rejection.code() == grpc.StatusCode.INVALID_ARGUMENT
    assert transport.ENCODER_TRANSFER_REJECTED_METADATA in rejection.trailing_metadata()
    assert not producer.failed and not consumer.failed
    assert json.loads(reference.opaque_state)["id"] in producer._exports
    assert producer._used > 0 and _writes(producer) == 0
    assert _sender(producer).stopped and not _sender(producer).registrations
    assert not _sender(producer).loaded
    assert consumer._used == 0 and not _receiver(consumer).registrations
    assert _receiver(consumer).stopped
    monkeypatch.setattr(FakeAgent, "load_remote_agent", original)
    actual = _receive(consumer, reference)
    assert torch.equal(actual[0], torch.arange(8, dtype=torch.float32).reshape(2, 4))
    assert producer._used == consumer._used == 0


def test_preflight_constructor_moves_descriptors_then_rejects_but_retry_succeeds(pair, monkeypatch):
    from tensorrt_llm._torch.disaggregation.base import agent

    producer, consumer = pair
    llm, _ = model()
    reference = producer.publish(native_result()[0], request(), llm)
    original = agent.TransferRequest
    calls = 0

    def construct(*args):
        nonlocal calls
        calls += 1
        result = original(*args)
        if calls == 1:
            # Native construction moves the source; match that in the Python fallback too.
            args[1].descs.clear()
            raise ValueError("constructor rejected after taking descriptors")
        return result

    monkeypatch.setattr(agent, "TransferRequest", construct)
    with pytest.raises(transport.EncoderTransferRejected, match="taking descriptors"):
        _receive(consumer, reference)
    assert not producer.failed and not consumer.failed
    actual = _receive(consumer, reference)
    assert calls == 2 and _writes(producer) == 1
    assert torch.equal(actual[0], torch.arange(8, dtype=torch.float32).reshape(2, 4))


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
            (transport.ENCODER_TRANSFER_REJECTED_METADATA,),
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


def test_submit_can_post_write_then_raise_without_returning_status(pair, monkeypatch):
    producer, consumer = pair
    llm, _ = model()
    reference = producer.publish(native_result()[0], request(), llm)
    original = FakeAgent.submit_transfer_requests

    def post_then_raise(agent, req):
        original(agent, req)
        raise RuntimeError("submission failed after posting WRITE")

    monkeypatch.setattr(FakeAgent, "submit_transfer_requests", post_then_raise)
    with pytest.raises(RuntimeError, match="after posting WRITE"):
        _receive(consumer, reference)
    assert _writes(producer) == 1
    assert producer.failed and not consumer.failed
    assert producer._used > 0 and consumer._used > 0
    assert _sender(producer).registrations and _receiver(consumer).registrations
    assert not producer._exports


def _blocked_write(monkeypatch, producer, complete=True):
    entered, release = threading.Event(), threading.Event()
    original = FakeAgent.submit_transfer_requests
    statuses = []

    class Status:
        def wait(self, timeout):
            entered.set()
            assert release.wait(5), "test did not release the pending WRITE"
            return complete

    def submit(agent, req):
        original(agent, req)
        if not entered.is_set():
            status = Status()
            statuses.append(weakref.ref(status))
            return status
        return SimpleNamespace(wait=lambda timeout: True)

    monkeypatch.setattr(FakeAgent, "submit_transfer_requests", submit)
    return entered, release, statuses


def test_parallel_receives_use_isolated_agents_while_state_locks_remain_responsive(
    pair, monkeypatch
):
    producer, consumer = pair
    llm, _ = model()
    first = producer.publish(native_result()[0], request(), llm)
    entered, release, _ = _blocked_write(monkeypatch, producer)
    with ThreadPoolExecutor(max_workers=2) as workers:
        pending = workers.submit(_receive, consumer, first)
        try:
            assert entered.wait(5)
            first_receiver = _receiver(consumer)
            for cache in (producer, consumer):
                assert cache._lock.acquire(timeout=1), "state lock held across native wait/RPC"
                cache._lock.release()
            second = producer.publish(native_result()[0], request(), llm)
            progressed = workers.submit(_receive, consumer, second).result(timeout=2)
            second_receiver = _receiver(consumer)
            assert second_receiver is not first_receiver
            assert first_receiver.registrations and not first_receiver.stopped
            assert not second_receiver.registrations and second_receiver.stopped
            assert not release.is_set()
            assert torch.equal(progressed[0], torch.arange(8, dtype=torch.float32).reshape(2, 4))
        finally:
            release.set()
        pending.result(timeout=5)
    assert _writes(producer) == 2 and consumer._used == producer._used == 0
    assert not first_receiver.registrations and first_receiver.stopped


def test_receive_slots_are_reserved_during_inflight_transfers_and_returned_on_success(
    pair, monkeypatch
):
    producer, consumer = pair
    llm, _ = model()
    references = [producer.publish(native_result()[0], request(), llm) for _ in range(2)]
    monkeypatch.setattr(transport, "_MAX_RECEIVE_SLOTS", 1)
    entered, release, _ = _blocked_write(monkeypatch, producer)
    with ThreadPoolExecutor(max_workers=1) as workers:
        pending = workers.submit(_receive, consumer, references[0])
        try:
            assert entered.wait(5)
            with pytest.raises(RuntimeError, match="slots exhausted"):
                _receive(consumer, references[1])
            assert consumer._receive_slots == 1 and len(consumer.test_agents) == 1
        finally:
            release.set()
        pending.result(timeout=5)
    assert consumer._receive_slots == 0
    _receive(consumer, references[1])
    assert consumer._receive_slots == 0 and consumer._used == 0


def test_same_receiver_parallel_writes_use_isolated_producer_agents(pair, monkeypatch):
    producer, consumer = pair
    llm, _ = model()
    references = [producer.publish(native_result()[0], request(), llm) for _ in range(2)]
    peer = consumer._agent_factory(consumer.name)
    buffers, registrations, destinations = [], [], []
    for reference in references:
        specs = json.loads(reference.opaque_state)["metadata"]["tensors"]
        tensors = [
            torch.empty(spec["shape"], dtype=transport._DTYPES[spec["dtype"]]) for spec in specs
        ]
        registration, _, descriptors = transport._descriptors(tensors, consumer.name)
        peer.register_memory(registration)
        buffers.append(tensors)
        registrations.append(registration)
        destinations.append(
            transport._encode(
                {
                    "name": peer.name,
                    "agent": base64.b64encode(peer.name.encode()).decode(),
                    "memory_type": "DRAM",
                    "descs": descriptors,
                }
            )
        )
    entered, release = threading.Barrier(3), threading.Event()
    original = FakeAgent.submit_transfer_requests

    def submit(agent, req):
        original(agent, req)

        def wait(timeout):
            entered.wait(5)
            assert release.wait(5)
            return True

        return SimpleNamespace(wait=wait)

    monkeypatch.setattr(FakeAgent, "submit_transfer_requests", submit)
    try:
        with ThreadPoolExecutor(max_workers=2) as workers:
            pending = [
                workers.submit(producer.transfer, reference, destination)
                for reference, destination in zip(references, destinations)
            ]
            try:
                entered.wait(5)
                assert len(producer.test_agents) == 2
                assert all(
                    agent.loaded == {peer.name} and agent.registrations
                    for agent in producer.test_agents
                )
            finally:
                release.set()
            for future in pending:
                future.result(timeout=5)
        assert _writes(producer) == 2 and all(agent.stopped for agent in producer.test_agents)
        assert all(
            torch.equal(tensors[0], torch.arange(8, dtype=torch.float32).reshape(2, 4))
            for tensors in buffers
        )
    finally:
        for registration in registrations:
            peer.deregister_memory(registration)
        peer.shutdown()


def test_receive_quota_is_reserved_before_allocation_and_returned_on_failure(pair, monkeypatch):
    producer, consumer = pair
    llm, _ = model()
    reference = producer.publish(native_result()[0], request(), llm)
    size = producer._used
    monkeypatch.setattr(transport, "_MAX_CACHE_BYTES", size)
    original = torch.empty
    checked = []

    def allocate(*args, **kwargs):
        if kwargs.get("device") == consumer.device:
            assert consumer._used == size
            checked.append(True)
            raise MemoryError("allocation failed")
        return original(*args, **kwargs)

    monkeypatch.setattr(transport.torch, "empty", allocate)
    with pytest.raises(MemoryError, match="allocation failed"):
        _receive(consumer, reference)
    assert checked and consumer._used == 0 and not consumer.failed
    assert _writes(producer) == 0 and producer._used == size
    monkeypatch.setattr(transport.torch, "empty", original)
    _receive(consumer, reference)
    assert consumer._used == producer._used == 0


def test_close_waits_for_active_rpc_and_native_wait_on_both_sides(pair, monkeypatch):
    producer, consumer = pair
    llm, _ = model()
    reference = producer.publish(native_result()[0], request(), llm)
    entered, release, _ = _blocked_write(monkeypatch, producer)
    draining = []
    for cache in (producer, consumer):
        event, original = threading.Event(), cache._lock.wait_for

        def wait_for(predicate, timeout=None, original=original, event=event):
            event.set()
            return original(predicate, timeout)

        monkeypatch.setattr(cache._lock, "wait_for", wait_for)
        draining.append(event)
    with ThreadPoolExecutor(max_workers=3) as workers:
        pending = workers.submit(_receive, consumer, reference)
        try:
            assert entered.wait(5)
            closes = [workers.submit(cache.close) for cache in (producer, consumer)]
            assert all(event.wait(5) for event in draining)
            assert _sender(producer).registrations and _receiver(consumer).registrations
            assert not _sender(producer).stopped and not _receiver(consumer).stopped
        finally:
            release.set()
        pending.result(timeout=5)
        for future in closes:
            future.result(timeout=5)
    assert _sender(producer).stopped and _receiver(consumer).stopped
    assert producer._used == consumer._used == 0


def test_close_timeout_quarantines_active_status_and_reports_repeated_failure(pair, monkeypatch):
    producer, consumer = pair
    llm, _ = model()
    reference = producer.publish(native_result()[0], request(), llm)
    entered, release, statuses = _blocked_write(monkeypatch, producer, complete=False)

    def drain_timeout(predicate, timeout=None):
        if timeout is None:
            return predicate()
        assert timeout == transport._TIMEOUT_SECONDS + 10
        return False

    with ThreadPoolExecutor(max_workers=1) as workers:
        pending = workers.submit(_receive, consumer, reference)
        try:
            assert entered.wait(5)
            monkeypatch.setattr(producer._lock, "wait_for", drain_timeout)
            with pytest.raises(RuntimeError, match="did not drain"):
                producer.close()
            assert producer.failed and not _sender(producer).stopped
            assert _sender(producer).registrations
        finally:
            release.set()
        with pytest.raises(RuntimeError, match="did not complete"):
            pending.result(timeout=5)
    assert statuses[0]() is not None
    assert any(
        resource is statuses[0]()
        for entry in transport._QUARANTINED
        if isinstance(entry, tuple)
        for resource in entry
    )
    with pytest.raises(RuntimeError, match="shutdown failed"):
        producer.close()


def test_real_grpc_marked_rejection_releases_receiver_and_allows_a_fresh_reference():
    server = grpc.server(ThreadPoolExecutor(max_workers=2))
    port = server.add_insecure_port("127.0.0.1:0")
    producer = _cache(f"127.0.0.1:{port}")
    consumer = _cache()

    class Service(transport.encoder_transfer_pb2_grpc.EncoderTransferServicer):
        def Transfer(self, req, context):
            try:
                producer.transfer(req.cache, req.destination)
            except transport.EncoderTransferRejected as error:
                context.set_trailing_metadata((transport.ENCODER_TRANSFER_REJECTED_METADATA,))
                context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(error))
            return encoder_transfer_pb2.TransferEncoderCacheResponse()

    transport.encoder_transfer_pb2_grpc.add_EncoderTransferServicer_to_server(Service(), server)
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
        assert transport.ENCODER_TRANSFER_REJECTED_METADATA in rejection.trailing_metadata()
        assert not consumer.failed and not _receiver(consumer).registrations and consumer._used == 0
        second = producer.publish(native_result()[0], request(), llm)
        actual = _receive(consumer, second)
        assert torch.equal(actual[0], torch.arange(8, dtype=torch.float32).reshape(2, 4))
        assert _writes(producer) == 2
    finally:
        consumer.close()
        producer.close()
        assert server.stop(0).wait(5)


def test_producer_quota_is_reserved_before_cloning_concurrent_exports(pair, monkeypatch):
    producer, _ = pair
    llm, _ = model()
    outputs = [native_result(), native_result()]
    size = sum(tensor.numel() * tensor.element_size() for tensor in outputs[0][1])
    monkeypatch.setattr(transport, "_MAX_CACHE_BYTES", size)
    entered, release = threading.Event(), threading.Event()
    original = torch.Tensor.clone
    clones = []

    def clone(tensor, *args, **kwargs):
        clones.append(tuple(tensor.shape))
        if not entered.is_set():
            assert producer._used == size
            entered.set()
            assert release.wait(5), "test did not release the owned export clone"
        return original(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "clone", clone)
    with ThreadPoolExecutor(max_workers=1) as workers:
        pending = workers.submit(producer.publish, outputs[0][0], request(), llm)
        try:
            assert entered.wait(5)
            with pytest.raises(RuntimeError, match="cache is full"):
                producer.publish(outputs[1][0], request(), llm)
            assert len(clones) == 1 and producer._used == size
            assert not producer.failed
        finally:
            release.set()
        reference = pending.result(timeout=5)
    assert len(clones) == 3
    assert json.loads(reference.opaque_state)["id"] in producer._exports
    producer.close()
    assert producer._used == 0


@pytest.mark.asyncio
async def test_cancelled_encoder_transfer_keeps_native_work_alive():
    started, finish = threading.Event(), threading.Event()

    def transfer(cache, destination):
        started.set()
        assert finish.wait(10)

    inference = SimpleNamespace(_encoder_cleanup_tasks=set())
    inference.encoder_role = "encode"
    inference._encoder_cache = SimpleNamespace(transfer=transfer)
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
    inference._encoder_cache = SimpleNamespace(transfer=transfer)
    context = FakeServicerContext()
    with pytest.raises(AbortError):
        await EncoderTransferServicer(inference).Transfer(
            encoder_transfer_pb2.TransferEncoderCacheRequest(), context
        )
    assert context.abort_code == (
        grpc.StatusCode.INVALID_ARGUMENT if rejected else grpc.StatusCode.UNAVAILABLE
    )
    assert context.trailing_metadata == ((ENCODER_TRANSFER_REJECTED_METADATA,) if rejected else ())
