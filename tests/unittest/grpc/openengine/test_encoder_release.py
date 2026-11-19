# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Encoder release ownership and malformed-reference handling."""

# ruff: noqa: F811 -- imported pytest fixture names also appear as test parameters

import grpc
import pytest
from conftest import AbortError, FakeServicerContext
from test_control import _inference, _servicer
from test_encoder_nixl import pair as pair
from utils.native_encoder import model, native_result, request

from tensorrt_llm._torch.disaggregation import encoder_nixl as transport
from tensorrt_llm._torch.disaggregation.encoder_output import EncodedMediaRef
from tensorrt_llm.grpc.openengine.bindings import encoding_pb2
from tensorrt_llm.grpc.openengine.encoder import portable_reference

pytestmark = pytest.mark.cpu_only


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
        reference_format=transport.BACKEND,
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
