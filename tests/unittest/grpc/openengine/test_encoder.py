# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from tensorrt_llm._torch.disaggregation import encoder_output as oe
from tensorrt_llm.grpc.openengine.bindings import generation_pb2 as pb
from tensorrt_llm.grpc.openengine.encoder import encoder_input, validate_encoder
from tensorrt_llm.grpc.openengine.errors import UnsupportedFeatureError

pytestmark = pytest.mark.cpu_only


def wire_request():
    return pb.GenerateRequest(
        request_id="encode",
        model="qwen",
        prompt="Describe <image_1>",
        media=[pb.MediaItem(modality=pb.MODALITY_IMAGE, data_uri="data:image/png;base64,aW1hZ2U=")],
    )


def test_encoder_role_validation_fails_closed():
    req = wire_request()
    with pytest.raises(UnsupportedFeatureError, match="Encode"):
        validate_encoder(req, "encode")
    with pytest.raises(UnsupportedFeatureError):
        validate_encoder(req, "consume")
    req.encoded_media.opaque_state = b"engine-owned"
    assert validate_encoder(req, "consume") == "consume"
    with pytest.raises(UnsupportedFeatureError):
        validate_encoder(req, "")
    req.kv.session.session_id = "prefilled"
    with pytest.raises(UnsupportedFeatureError):
        validate_encoder(req, "consume")
    req.ClearField("encoded_media")
    assert validate_encoder(req, "consume") == ""


@pytest.mark.parametrize("token_input", [False, True])
def test_encoder_input_ignores_unknown_wire_fields_and_wire_order(token_input):
    req = wire_request()
    if token_input:
        req.token_ids.ids.extend([1, 2])
    equivalent = pb.GenerateRequest.FromString(req.SerializeToString())
    equivalent.media[0].ParseFromString(
        pb.MediaItem(data_uri=req.media[0].data_uri).SerializeToString()
        + pb.MediaItem(modality=pb.MODALITY_IMAGE).SerializeToString()
        + b"\xa0\x06\x01"
    )
    if token_input:
        equivalent.token_ids.ParseFromString(b"\x08\x01\x08\x02\xa0\x06\x01")
    assert req.SerializeToString() != equivalent.SerializeToString()
    assert encoder_input(req) == encoder_input(equivalent)


def test_encoder_input_preserves_order_source_and_all_media_fields():
    req = wire_request()
    req.media.add(modality=pb.MODALITY_IMAGE, raw_bytes=b"second")
    original = encoder_input(req)
    mutations = (
        lambda value: value.media.reverse(),
        lambda value: setattr(value.media[0], "url", value.media[0].data_uri),
        lambda value: setattr(value.media[0], "mime_type", "image/png"),
        lambda value: setattr(value.media[0], "uuid", "another"),
        lambda value: setattr(value.media[0], "modality", pb.MODALITY_VIDEO),
        lambda value: setattr(value.media[1], "raw_bytes", b"different"),
        lambda value: setattr(value, "prompt", value.prompt + "!"),
    )
    for mutate in mutations:
        changed = pb.GenerateRequest.FromString(req.SerializeToString())
        mutate(changed)
        assert encoder_input(changed) != original
    req.token_ids.ids.extend([1, 2])
    original = encoder_input(req)
    req.token_ids.ids.reverse()
    assert encoder_input(req) != original


@pytest.mark.parametrize("payload", [b"", b"\x00\xffopaque", b'{"format":"custom","version":9}'])
def test_native_reference_boundary_is_lossless(payload):
    from tensorrt_llm.grpc.openengine.encoder import native_reference, portable_reference

    native = oe.EncodedMediaRef(payload)
    assert native_reference(portable_reference(native)) == native
