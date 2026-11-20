# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: F811 -- imported pytest fixture names are also test parameters

import base64
import json
import os
import threading
import weakref
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from utils.native_encoder import (
    FakeAgent,
    _check_handoff_hash_initialization,
    _check_malformed_handoff_hashes,
    _receive,
    _receiver,
    _sender,
    _writes,
    model,
    native_result,
    request,
)
from utils.native_encoder import native_pair as pair  # noqa: F401

from tensorrt_llm._torch.disaggregation import encoder_nixl as transport
from tensorrt_llm._torch.disaggregation.encoder_output import (
    EncodedMediaRef,
    EncoderCapacityError,
    EncoderTransferRejected,
    EncoderUnavailableError,
)
from tensorrt_llm.disaggregated_params import DisaggregatedParams

pytestmark = pytest.mark.cpu_only


def _consume_worker_exports(pipe):
    from tensorrt_llm._torch.shared_tensor import SharedTensorContainer

    handles = pipe.recv()
    tensors = [SharedTensorContainer.from_dict(h).get_local_view() for h in handles]
    pipe.send((os.getpid(), [tensor.tolist() for tensor in tensors]))
    pipe.close()


def _blocked_write(monkeypatch, complete=True):
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


@pytest.mark.parametrize("supplied", [False, True])
def test_nixl_handoff_restores_hash_validation_and_missing_hash_fallback(pair, supplied):
    _check_handoff_hash_initialization(*pair, supplied)


@pytest.mark.parametrize("hashes", [[], [[1] * 7], [[1.0] * 8], ["bad"], 42, [[1] * 8] * 2])
def test_nixl_handoff_rejects_malformed_hashes_before_import(pair, monkeypatch, hashes):
    _check_malformed_handoff_hashes(*pair, hashes, monkeypatch)


def test_nixl_roundtrip_retains_tensors_metadata_and_mrope(pair):
    producer, consumer = pair
    llm, seen = model()
    output, expected = native_result()
    ref = producer.publish(output, request(), llm)
    state = json.loads(ref.opaque_state)
    assert state["format"] == transport.BACKEND
    assert set(state) == {"format", "version", "id", "endpoint", "metadata"}
    assert "ptr" not in state["metadata"]
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
    altered = replace(request(), prompt="another prompt")
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


def test_bad_destination_size_does_not_consume_export(pair):
    producer, _ = pair
    llm, _ = model()
    ref = producer.publish(native_result()[0], request(), llm)
    with pytest.raises(ValueError, match="descriptors"):
        producer.transfer(ref, b'{"descs": []}')
    assert len(producer._exports) == 1
    assert not producer.failed


@pytest.mark.parametrize("invalid", ["address", "device"])
def test_unregistered_destination_consumes_claim_without_write(pair, monkeypatch, invalid):
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
    assert json.loads(reference.opaque_state)["id"] not in producer._exports
    assert _sender(producer).stopped and not _sender(producer).registrations
    assert consumer._used == 0 and _receiver(consumer).stopped
    with pytest.raises(EncoderTransferRejected, match="already consumed"):
        _receive(consumer, reference)
    reference = producer.publish(native_result()[0], request(), llm)
    actual = _receive(consumer, reference)
    assert torch.equal(actual[0], torch.arange(8, dtype=torch.float32).reshape(2, 4))
    assert producer._used == consumer._used == 0


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


def test_preflight_constructor_failure_consumes_claim_and_new_output_succeeds(pair, monkeypatch):
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
    with pytest.raises(transport.EncoderTransferRejected, match="already consumed"):
        _receive(consumer, reference)
    reference = producer.publish(native_result()[0], request(), llm)
    actual = _receive(consumer, reference)
    assert calls == 2 and _writes(producer) == 1
    assert torch.equal(actual[0], torch.arange(8, dtype=torch.float32).reshape(2, 4))


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


def test_parallel_receives_use_isolated_agents_while_state_locks_remain_responsive(
    pair, monkeypatch
):
    producer, consumer = pair
    llm, _ = model()
    first = producer.publish(native_result()[0], request(), llm)
    entered, release, _ = _blocked_write(monkeypatch)
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
    entered, release, _ = _blocked_write(monkeypatch)
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


def test_close_waits_for_active_transfer_on_both_sides(pair, monkeypatch):
    producer, consumer = pair
    llm, _ = model()
    reference = producer.publish(native_result()[0], request(), llm)
    entered, release, _ = _blocked_write(monkeypatch)
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
    entered, release, statuses = _blocked_write(monkeypatch, complete=False)

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
            with pytest.raises(EncoderCapacityError, match="cache is full"):
                producer.publish(outputs[1][0], request(), llm)
            assert len(clones) == 1 and producer._used == size
            assert not producer.failed
        finally:
            release.set()
        reference = pending.result(timeout=5)
    assert len(clones) == 3
    assert json.loads(reference.opaque_state)["id"] in producer._exports
    producer.close()
    assert producer._used == 0 and not producer.test_agents


def test_discard_reclaims_only_unclaimed_nixl_exports(pair, monkeypatch):
    producer, consumer = pair
    llm, _ = model()
    ref = producer.publish(native_result()[0], request(), llm)
    assert producer._used > 0
    value = json.loads(ref.opaque_state)
    for key, replacement in (("endpoint", "wrong.example:1234"), ("metadata", {})):
        malformed = EncodedMediaRef(
            opaque_state=json.dumps({**value, key: replacement}).encode(),
        )
        with pytest.raises(ValueError):
            producer.discard(malformed)
        assert producer._used > 0
    assert producer.discard(ref)
    assert producer._used == 0 and not producer._exports
    assert not producer.discard(ref)

    ref = producer.publish(native_result()[0], request(), llm)
    reserved = producer._used
    submit = FakeAgent.submit_transfer_requests
    checked = []

    def submit_and_discard(agent, transfer):
        status = submit(agent, transfer)
        assert not producer.discard(ref)
        assert producer._used == reserved
        assert agent.registrations
        checked.append(True)
        return status

    monkeypatch.setattr(FakeAgent, "submit_transfer_requests", submit_and_discard)
    consumer.prepare(ref, request(), llm, None)
    assert checked
    assert producer._used == 0
    assert not producer.discard(ref)


def test_discard_during_claim_prevents_rejected_transfer_from_restoring_export(pair, monkeypatch):
    producer, consumer = pair
    llm, _ = model()
    reference = producer.publish(native_result()[0], request(), llm)
    identity = json.loads(reference.opaque_state)["id"]
    reserved = producer._used

    def release_then_reject(agent, name, desc):
        assert identity in producer._claims
        value = json.loads(reference.opaque_state)
        malformed = EncodedMediaRef(
            opaque_state=json.dumps({**value, "metadata": {}}).encode(),
        )
        with pytest.raises(ValueError, match="metadata|layout"):
            producer.discard(malformed)
        assert not producer.discard(reference)
        assert not producer.discard(reference)
        assert producer._used == reserved
        assert agent.registrations
        raise ValueError("receiver preflight rejected after release")

    monkeypatch.setattr(FakeAgent, "load_remote_agent", release_then_reject)
    with pytest.raises(transport.EncoderTransferRejected, match="preflight rejected"):
        _receive(consumer, reference)
    assert not producer._exports and not producer._claims
    assert producer._used == consumer._used == 0
    assert not _sender(producer).registrations
    assert not producer.discard(reference)
    assert not producer.failed and not consumer.failed


def test_preflight_rejection_consumes_reference_and_receiver_is_reusable(pair, monkeypatch):
    producer, consumer = pair
    llm, _ = model()
    reference = producer.publish(native_result()[0], request(), llm)
    original = FakeAgent.load_remote_agent

    def reject_peer(agent, name, desc):
        agent.loaded.add("partially-imported-under-another-name")
        raise ValueError("receiver metadata cannot be imported")

    monkeypatch.setattr(FakeAgent, "load_remote_agent", reject_peer)
    with pytest.raises(transport.EncoderTransferRejected, match="metadata cannot be imported"):
        _receive(consumer, reference)
    assert not producer.failed and not consumer.failed
    assert json.loads(reference.opaque_state)["id"] not in producer._exports
    assert producer._used == 0 and _writes(producer) == 0
    assert _sender(producer).stopped and not _sender(producer).registrations
    assert not _sender(producer).loaded
    assert consumer._used == 0 and not _receiver(consumer).registrations
    assert _receiver(consumer).stopped
    monkeypatch.setattr(FakeAgent, "load_remote_agent", original)
    with pytest.raises(transport.EncoderTransferRejected, match="already consumed"):
        _receive(consumer, reference)
    reference = producer.publish(native_result()[0], request(), llm)
    actual = _receive(consumer, reference)
    assert torch.equal(actual[0], torch.arange(8, dtype=torch.float32).reshape(2, 4))
    assert producer._used == consumer._used == 0


@pytest.mark.parametrize("limit", ["bytes", "slots"])
def test_quarantined_receives_remain_charged_and_bound_future_allocations(pair, monkeypatch, limit):
    producer, consumer = pair
    llm, _ = model()
    reference = producer.publish(native_result()[0], request(), llm)
    original = producer.transfer

    def lost_reply(*args):
        original(*args)
        raise RuntimeError("lost reply")

    with monkeypatch.context() as patch:
        patch.setattr(producer, "transfer", lost_reply)
        with pytest.raises(RuntimeError, match="lost reply"):
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
    with pytest.raises(EncoderCapacityError, match=message):
        _receive(consumer, fresh)
    assert consumer._used == used and consumer._receive_slots == 1
    assert len(consumer.test_agents) == agents and not consumer.failed
    assert pinned.registrations and not pinned.stopped
    assert json.loads(fresh.opaque_state)["id"] in producer._exports


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

    def fail_transfer(endpoint, reference, destination, timeout):
        raise RuntimeError("lost peer")

    monkeypatch.setattr(consumer, "_request_transfer", fail_transfer)
    with pytest.raises(RuntimeError, match="lost peer"):
        _receive(consumer, reference)
    pinned = _receiver(consumer)
    assert consumer.failed and consumer._used == size and pinned.registrations
    with pytest.raises(EncoderUnavailableError, match="transport unavailable"):
        _receive(consumer, reference)
    consumer.close()
    assert pinned.registrations and not pinned.stopped


def test_unreachable_peer_quarantines_only_its_receive_and_late_write_remains_safe(
    pair, monkeypatch
):
    producer, consumer = pair
    llm, _ = model()
    reference = producer.publish(native_result()[0], request(), llm)
    pending = []

    def fail_transfer(endpoint, ref, destination, timeout):
        pending.append((ref, destination))
        raise TimeoutError("encoder endpoint unavailable")

    with monkeypatch.context() as patch:
        patch.setattr(consumer, "_request_transfer", fail_transfer)
        with pytest.raises(TimeoutError, match="endpoint unavailable"):
            _receive(consumer, reference)
    pinned = _receiver(consumer)
    pinned_bytes = consumer._used
    assert pinned.registrations and not pinned.stopped
    assert pinned_bytes > 0 and consumer._receive_slots == 1 and not consumer.failed

    fresh = producer.publish(native_result()[0], request(), llm)
    actual = _receive(consumer, fresh)
    assert _receiver(consumer) is not pinned and _receiver(consumer).stopped
    assert consumer._used == pinned_bytes and consumer._receive_slots == 1
    producer.transfer(*pending[0])
    assert torch.equal(actual[0], torch.arange(8, dtype=torch.float32).reshape(2, 4))
    consumer.close()
    assert pinned.registrations and not pinned.stopped
    assert consumer._used == pinned_bytes


@pytest.mark.parametrize(
    "invalid,value",
    [
        ("missing_format", None),
        ("wrong_format", "another-engine-output-v1"),
        ("missing_version", None),
        ("wrong_version", 2),
        ("oversized", None),
        ("metadata", []),
        ("tensors", 1),
        ("tensor", None),
        ("dtype", []),
        ("shape", [2**30]),
        ("embedding_metadata", ["bad"]),
        ("has_mrope", "true"),
        ("has_mrope", False),
        ("missing_hashes", None),
    ],
)
def test_native_reference_rejection_precedes_allocation_claim_and_release(
    pair, monkeypatch, invalid, value
):
    producer, consumer = pair
    llm, _ = model()
    reference = producer.publish(native_result()[0], request(), llm)
    state = json.loads(reference.opaque_state)
    identity, metadata = state["id"], state["metadata"]
    if invalid.startswith("missing_"):
        target = metadata if invalid == "missing_hashes" else state
        target.pop(invalid.removeprefix("missing_"))
    elif invalid.startswith("wrong_"):
        state[invalid.removeprefix("wrong_")] = value
    elif invalid == "metadata":
        state["metadata"] = value
    elif invalid in ("dtype", "shape"):
        metadata["tensors"][0][invalid] = value
    elif invalid == "tensor":
        metadata["tensors"][0] = value
    elif invalid != "oversized":
        metadata[invalid] = value
    payload = (
        json.dumps(state).encode()
        if invalid != "oversized"
        else b"x" * (transport._MAX_REFERENCE_BYTES + 1)
    )
    malformed = EncodedMediaRef(payload)
    destination = b"destination must not be parsed"
    decode = transport._decode

    def guarded_decode(data):
        if data == destination:
            pytest.fail("invalid reference reached receiver descriptor parsing")
        return decode(data)

    monkeypatch.setattr(transport, "_decode", guarded_decode)
    for operation in (
        lambda: consumer.prepare(malformed, request(), llm, None),
        lambda: producer.discard(malformed),
        lambda: producer.transfer(malformed, destination),
    ):
        with pytest.raises(ValueError):
            operation()
        assert identity in producer._exports and not producer._claims
        assert consumer._used == consumer._receive_slots == 0
        assert not consumer.test_agents and not producer.test_agents
    consumer.prepare(reference, request(), llm, None)
    assert producer._used == consumer._used == 0


def test_local_sender_setup_failure_is_proven_rejection_with_local_attribution(pair, monkeypatch):
    producer, consumer = pair
    llm, _ = model()
    ref = producer.publish(native_result()[0], request(), llm)

    def fail_factory(name):
        raise RuntimeError("local agent initialization failed")

    monkeypatch.setattr(producer, "_agent_factory", fail_factory)
    with pytest.raises(EncoderTransferRejected, match="local encoder sender setup.*local agent"):
        consumer.prepare(ref, request(), llm, None)
    assert producer._used == consumer._used == consumer._receive_slots == 0
    assert not producer._exports and not producer._claims and not producer.failed
    assert _receiver(consumer).stopped and not _receiver(consumer).registrations


@pytest.mark.parametrize("state", ["failed", "_closing", "_closed"])
def test_unavailable_consumer_rejects_before_receive_allocation(pair, monkeypatch, state):
    producer, consumer = pair
    llm, _ = model()
    ref = producer.publish(native_result()[0], request(), llm)
    with monkeypatch.context() as patch:
        patch.setattr(consumer, state, True)
        with pytest.raises(EncoderUnavailableError, match="transport unavailable"):
            consumer.prepare(ref, request(), llm, None)
        assert not consumer.test_agents and consumer._used == consumer._receive_slots == 0
        assert producer._used > 0 and not producer._claims
    consumer.prepare(ref, request(), llm, None)
