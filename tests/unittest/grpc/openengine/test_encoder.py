# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors import safe_open

from tensorrt_llm._torch.shared_tensor import SharedTensorContainer
from tensorrt_llm.disaggregated_params import DisaggregatedParams
from tensorrt_llm.grpc.openengine import encoder as oe
from tensorrt_llm.grpc.openengine.bindings import generation_pb2 as pb
from tensorrt_llm.grpc.openengine.errors import UnsupportedFeatureError
from tensorrt_llm.grpc.openengine.preparation import _input_digest

pytestmark = pytest.mark.cpu_only


@pytest.mark.parametrize("removed_before", ["stat", "unlink"])
def test_collection_tolerates_a_concurrently_removed_file(tmp_path, monkeypatch, removed_before):
    cache = oe.EncoderCache(str(tmp_path))
    claimed = tmp_path / "claimed.tmp"
    claimed.write_bytes(b"expired")
    os.utime(claimed, (1, 1))
    retained = tmp_path / "retained.safetensors"
    retained.write_bytes(b"retained")
    expected_bytes = retained.stat().st_size
    original = getattr(Path, removed_before)
    unlink = Path.unlink
    removed = False

    def remove_first(path, *args, **kwargs):
        nonlocal removed
        if path == claimed and not removed:
            removed = True
            unlink(path)
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, removed_before, remove_first)
    assert cache._collect() == expected_bytes
    assert removed
    assert retained.is_file()
    assert not claimed.exists()


def request():
    return pb.GenerateRequest(
        request_id="encode",
        model="qwen",
        prompt="Describe <image_1>",
        media=[pb.MediaItem(modality=pb.MODALITY_IMAGE, data_uri="data:image/png;base64,aW1hZ2U=")],
    )


def model():
    values = {"model_type": "qwen3_vl"}
    processor = SimpleNamespace(
        config=SimpleNamespace(to_dict=lambda: values),
        processor=SimpleNamespace(image_processor=None),
        get_preferred_media_io_kwargs=lambda: {},
        support_mm_disagg=True,
    )
    seen = {}

    def preprocess(inputs, sampling, params):
        seen["prompt"] = inputs["prompt"]
        seen["metadata"] = params.multimodal_embedding_handles[0]["metadata"]
        seen["hashes"] = params.multimodal_hashes
        handles = [
            *params.multimodal_embedding_handles,
            params.mrope_position_ids_handle,
            params.mrope_position_deltas_handle,
        ]
        seen["tensors"] = [
            SharedTensorContainer.from_dict(h).get_local_view().clone() for h in handles
        ]
        return SimpleNamespace(multimodal_params=None)

    return SimpleNamespace(input_processor=processor, preprocess=preprocess), seen


def native_result():
    tensors = [
        torch.arange(8, dtype=torch.float32).reshape(2, 4),
        torch.arange(6, dtype=torch.int64).reshape(3, 1, 2),
        torch.tensor([[1]]),
    ]
    handles = [SharedTensorContainer.from_tensor(tensor).dump_to_dict() for tensor in tensors]
    handles[0]["metadata"] = {"image_grid_thw": [1, 2, 2]}
    params = DisaggregatedParams(
        multimodal_embedding_handles=[handles[0]],
        multimodal_hashes=[[2147483647] * 8],
        mrope_position_ids_handle=handles[1],
        mrope_position_deltas_handle=handles[2],
    )
    return SimpleNamespace(disaggregated_params=params), tensors


def test_native_exports_roundtrip_once_with_metadata_and_mrope(tmp_path):
    cache = oe.EncoderCache(str(tmp_path))
    llm, seen = model()
    result, tensors = native_result()
    ref = cache.publish(result, request(), llm)
    assert result.disaggregated_params.multimodal_embedding_handles is None
    _, params = cache.prepare(ref, request(), llm, None)
    assert params.request_type == "context_and_generation"
    assert seen["prompt"] == request().prompt
    assert seen["metadata"] == {"image_grid_thw": [1, 2, 2]}
    assert seen["hashes"] == [[2147483647] * 8]
    assert all(torch.equal(actual, expected) for actual, expected in zip(seen["tensors"], tensors))
    with pytest.raises(ValueError, match="expired or was already consumed"):
        cache.prepare(ref, request(), llm, None)
    assert not list(tmp_path.glob("*.safetensors"))
    assert not list(tmp_path.glob("*.tmp"))


def test_expiry_wrong_inputs_and_cross_host_are_rejected(tmp_path):
    cache = oe.EncoderCache(str(tmp_path))
    llm, _ = model()
    result, _ = native_result()
    ref = cache.publish(result, request(), llm)
    value = json.loads(ref.opaque_state)
    value["host"] = "another-host"
    bad = pb.EncoderHandoffRef(transfer_backend=oe.BACKEND, opaque_state=json.dumps(value).encode())
    with pytest.raises(ValueError, match="same host"):
        cache.prepare(bad, request(), llm, None)
    wrong = request()
    wrong.prompt = "Another image prompt"
    with pytest.raises(ValueError, match="does not match"):
        cache.prepare(ref, wrong, llm, None)
    cache.prepare(ref, request(), llm, None)
    result, _ = native_result()
    llm._hf_model_dir = tmp_path / "checkpoint-a"
    ref = cache.publish(result, request(), llm)
    llm._hf_model_dir = tmp_path / "checkpoint-b"
    with pytest.raises(ValueError, match="another model"):
        cache.prepare(ref, request(), llm, None)
    llm._hf_model_dir = tmp_path / "checkpoint-a"
    cache.prepare(ref, request(), llm, None)
    result, _ = native_result()
    ref = cache.publish(result, request(), llm)
    path = next(tmp_path.glob("*.safetensors"))
    os.utime(path, (1, 1))
    with pytest.raises(ValueError, match="expired"):
        cache.prepare(ref, request(), llm, None)


def test_lossy_prompt_decoding_preserves_the_handoff(tmp_path):
    cache = oe.EncoderCache(str(tmp_path))
    llm, seen = model()
    llm.tokenizer = SimpleNamespace(
        decode=lambda ids, **kwargs: "decoded prompt", encode=lambda prompt, **kwargs: [9]
    )
    req = request()
    req.token_ids.ids.extend([1, 2])
    result, _ = native_result()
    ref = cache.publish(result, req, llm)
    with pytest.raises(ValueError, match="lossless prompt token decoding"):
        cache.prepare(ref, req, llm, None)
    llm.tokenizer.encode = lambda prompt, **kwargs: [1, 2]
    cache.prepare(ref, req, llm, None)
    assert seen["prompt"] == "decoded prompt"
    assert not list(tmp_path.glob("*.safetensors"))
    assert not list(tmp_path.glob("*.tmp"))


def test_quota_counts_actual_serialized_bytes(tmp_path, monkeypatch):
    cache = oe.EncoderCache(str(tmp_path))
    llm, _ = model()
    result, tensors = native_result()
    cache.publish(result, request(), llm)
    path = next(tmp_path.glob("*.safetensors"))
    with safe_open(str(path), framework="pt") as stored:
        estimated = sum(t.numel() * t.element_size() for t in tensors) + len(
            stored.metadata()["handoff"]
        )
    monkeypatch.setattr(oe, "_MAX_CACHE_BYTES", path.stat().st_size + estimated)
    result, _ = native_result()
    with pytest.raises(RuntimeError, match="cache is full"):
        cache.publish(result, request(), llm)
    assert len(list(tmp_path.glob("*.safetensors"))) == 1
    assert not list(tmp_path.glob("*.tmp"))


def test_failed_rebuild_and_discarded_result_reclaim_all_exports(tmp_path, monkeypatch):
    llm, _ = model()
    result, _ = native_result()
    released = []
    original_from_dict = SharedTensorContainer.from_dict

    def release(handle):
        released.append(handle)
        original_from_dict(handle).get_local_view()

    monkeypatch.setattr(SharedTensorContainer, "release_dict", release)
    monkeypatch.setattr(
        SharedTensorContainer,
        "from_dict",
        lambda handle: (_ for _ in ()).throw(ValueError("broken export")),
    )
    with pytest.raises(ValueError, match="broken export"):
        oe.EncoderCache(str(tmp_path)).publish(result, request(), llm)
    assert len(released) == 3
    result, _ = native_result()
    oe.release_encoder_result(result)
    oe.release_encoder_result(result)
    assert len(released) == 6


def test_encoder_role_validation_fails_closed():
    req = request()
    req.encoder.SetInParent()
    with pytest.raises(ValueError, match="exactly one"):
        oe.validate_encoder(req, "encode")
    req.encoder.encode_only.SetInParent()
    assert oe.validate_encoder(req, "encode") == "encode"
    with pytest.raises(UnsupportedFeatureError):
        oe.validate_encoder(req, "")
    req.encoder.handoff.transfer_backend = oe.BACKEND
    assert req.encoder.WhichOneof("mode") == "handoff"
    assert oe.validate_encoder(req, "consume") == "consume"
    with pytest.raises(UnsupportedFeatureError, match="role consume"):
        oe.validate_encoder(req, "encode")
    req.ClearField("encoder")
    with pytest.raises(UnsupportedFeatureError, match="encoder handoff"):
        oe.validate_encoder(req, "consume")
    req.kv.session.session_id = "prefilled"
    assert oe.validate_encoder(req, "consume") == ""
    req.encoder.handoff.transfer_backend = oe.BACKEND
    with pytest.raises(UnsupportedFeatureError, match="must not consume an encoder handoff"):
        oe.validate_encoder(req, "consume")


@pytest.mark.parametrize(
    "field, values",
    [
        ("sampling", {"temperature": 0.0}),
        ("stopping", {"max_tokens": 1}),
        ("guided", {"json_schema": "{}"}),
        ("response", {"return_output_logprobs": False}),
    ],
)
def test_encoder_only_rejects_language_options(field, values):
    req = request()
    req.encoder.encode_only.SetInParent()
    getattr(req, field).SetInParent()
    assert oe.validate_encoder(req, "encode") == "encode"
    for name, value in values.items():
        setattr(getattr(req, field), name, value)
    with pytest.raises(UnsupportedFeatureError, match=field):
        oe.validate_encoder(req, "encode")


def test_encoder_only_rejects_lora_selection():
    req = request()
    req.encoder.encode_only.SetInParent()
    req.lora_name = "adapter"
    with pytest.raises(UnsupportedFeatureError, match="LoRA"):
        oe.validate_encoder(req, "encode")


@pytest.mark.parametrize("token_input", [False, True])
def test_shared_handoff_accepts_field_equal_inputs_with_unknown_wire_fields(tmp_path, token_input):
    req = request()
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
    cache = oe.EncoderCache(str(tmp_path))
    llm, _ = model()
    llm.tokenizer = SimpleNamespace(
        decode=lambda ids, **kwargs: "Describe <image_1>",
        encode=lambda prompt, **kwargs: [1, 2],
    )
    ref = cache.publish(native_result()[0], req, llm)
    cache.prepare(ref, equivalent, llm, None)


def test_input_identity_preserves_order_source_and_all_media_fields():
    req = request()
    req.media.add(modality=pb.MODALITY_IMAGE, raw_bytes=b"second")
    original = _input_digest(req)
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
        assert _input_digest(changed) != original
    req.token_ids.ids.extend([1, 2])
    original = _input_digest(req)
    req.token_ids.ids.reverse()
    assert _input_digest(req) != original


def test_discard_is_idempotent_and_never_releases_a_claimed_handoff(tmp_path):
    cache = oe.EncoderCache(str(tmp_path))
    llm, _ = model()
    ref = cache.publish(native_result()[0], request(), llm)
    assert cache.discard(ref)
    assert not cache.discard(ref)
    assert not list(tmp_path.glob("*.safetensors"))
    with pytest.raises(ValueError, match="already consumed"):
        cache.prepare(ref, request(), llm, None)

    ref = cache.publish(native_result()[0], request(), llm)
    preprocess = llm.preprocess

    def prepare_claimed(*args):
        assert list(tmp_path.glob("*.tmp"))
        assert not cache.discard(ref)
        assert list(tmp_path.glob("*.tmp"))
        return preprocess(*args)

    llm.preprocess = prepare_claimed
    cache.prepare(ref, request(), llm, None)
    assert not cache.discard(ref)
    assert not list(tmp_path.glob("*.tmp"))


def test_discard_rejects_foreign_and_malformed_references(tmp_path):
    cache = oe.EncoderCache(str(tmp_path))
    llm, _ = model()
    ref = cache.publish(native_result()[0], request(), llm)
    value = json.loads(ref.opaque_state)
    for key, replacement in (("namespace", "foreign"), ("host", "foreign"), ("id", "../file")):
        malformed = pb.EncoderHandoffRef(
            transfer_backend=oe.BACKEND,
            opaque_state=json.dumps({**value, key: replacement}).encode(),
        )
        with pytest.raises(ValueError):
            cache.discard(malformed)
        assert len(list(tmp_path.glob("*.safetensors"))) == 1
    assert cache.discard(ref)


@pytest.mark.parametrize("operation", ["discard", "prepare"])
def test_recursive_decode_is_invalid_without_consuming_handoff(tmp_path, monkeypatch, operation):
    cache = oe.EncoderCache(str(tmp_path))
    llm, _ = model()
    ref = cache.publish(native_result()[0], request(), llm)
    malformed = pb.EncoderHandoffRef(transfer_backend=oe.BACKEND, opaque_state=b"[" * 1024)

    def fail_decode(_):
        raise RecursionError("maximum recursion depth exceeded while decoding JSON")

    with monkeypatch.context() as patch:
        patch.setattr(oe.json, "loads", fail_decode)
        with pytest.raises(ValueError, match="invalid encoder handoff reference"):
            if operation == "discard":
                cache.discard(malformed)
            else:
                cache.prepare(malformed, request(), llm, None)
    assert len(list(tmp_path.glob("*.safetensors"))) == 1
    assert cache.discard(ref)
