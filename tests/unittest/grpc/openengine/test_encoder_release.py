# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Encoder release ownership and malformed-reference handling."""

# ruff: noqa: F811 -- imported pytest fixture names also appear as test parameters

import json

import grpc
import pytest
from conftest import AbortError, FakeServicerContext
from test_control import _inference, _servicer
from test_encoder import model, native_result, request
from test_encoder_nixl import FakeAgent, _receive, _sender
from test_encoder_nixl import pair as pair

from tensorrt_llm.grpc.openengine import encoder_nixl as transport
from tensorrt_llm.grpc.openengine.bindings import generation_pb2 as pb
from tensorrt_llm.grpc.openengine.bindings import lifecycle_pb2

pytestmark = pytest.mark.cpu_only


def test_discard_reclaims_only_unclaimed_nixl_exports(pair, monkeypatch):
    producer, consumer = pair
    llm, _ = model()
    ref = producer.publish(native_result()[0], request(), llm)
    assert producer._used > 0
    value = json.loads(ref.opaque_state)
    for key, replacement in (("endpoint", "wrong.example:1234"), ("metadata", {})):
        malformed = pb.EncoderHandoffRef(
            transfer_backend=transport.BACKEND,
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
        malformed = pb.EncoderHandoffRef(
            transfer_backend=transport.BACKEND,
            opaque_state=json.dumps({**value, "metadata": {}}).encode(),
        )
        with pytest.raises(ValueError, match="metadata mismatch"):
            producer.discard(malformed)
        assert not producer._claims[identity].discard_requested
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


@pytest.mark.asyncio
async def test_abort_rejects_deeply_nested_reference_without_changing_exports(pair):
    producer, _ = pair
    llm, _ = model()
    reference = producer.publish(native_result()[0], request(), llm)
    reserved = producer._used
    inference = _inference()
    inference.encoder_role = "encode"
    inference._encoder_cache = producer
    malformed = pb.EncoderHandoffRef(
        transfer_backend=transport.BACKEND,
        opaque_state=b"[" * 10000 + b"0" + b"]" * 10000,
    )
    context = FakeServicerContext()
    with pytest.raises(AbortError):
        await _servicer(inference=inference).Abort(
            lifecycle_pb2.AbortRequest(encoder_handoff=malformed), context
        )
    assert context.abort_code == grpc.StatusCode.INVALID_ARGUMENT
    assert producer._used == reserved and len(producer._exports) == 1
    assert not producer._claims and not producer.failed
    assert producer.discard(reference)
