# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import os
from types import SimpleNamespace

import pytest
import torch
from safetensors import safe_open

from tensorrt_llm._torch.shared_tensor import SharedTensorContainer
from tensorrt_llm.disaggregated_params import DisaggregatedParams
from tensorrt_llm.grpc.openengine import encoder as oe
from tensorrt_llm.grpc.openengine.bindings import generation_pb2 as pb
from tensorrt_llm.grpc.openengine.errors import UnsupportedFeatureError

pytestmark = pytest.mark.cpu_only


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


def test_prefill_result_reclaims_new_exports_once_and_leaves_consumed_inputs(monkeypatch):
    incoming, _ = native_result()
    params = incoming.disaggregated_params
    inputs = [params.mrope_position_ids_handle, params.mrope_position_deltas_handle]
    for handle in [*params.multimodal_embedding_handles, *inputs]:
        SharedTensorContainer.from_dict(handle).get_local_view()
    owner = oe.PrefillResultExports(params)
    released = []
    release = SharedTensorContainer.release_dict

    def record(handle):
        released.append(handle)
        release(handle)

    monkeypatch.setattr(SharedTensorContainer, "release_dict", record)
    owner.release(incoming)
    assert [params.mrope_position_ids_handle, params.mrope_position_deltas_handle] == inputs
    assert not released
    for _ in range(2):
        fresh, _ = native_result()
        SharedTensorContainer.release_dict(
            fresh.disaggregated_params.multimodal_embedding_handles[0]
        )
        released.clear()
        handles = [
            fresh.disaggregated_params.mrope_position_ids_handle,
            fresh.disaggregated_params.mrope_position_deltas_handle,
        ]
        owner.release(fresh)
        # Native result snapshots can repeat the same handles in new dictionaries.
        owner.release(
            SimpleNamespace(
                disaggregated_params=DisaggregatedParams(
                    mrope_position_ids_handle=dict(handles[0]),
                    mrope_position_deltas_handle=dict(handles[1]),
                )
            )
        )
        assert released == handles


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


def test_shared_encoder_handoff_preserves_prefill_phase(tmp_path):
    cache = oe.EncoderCache(str(tmp_path))
    llm, seen = model()
    ref = cache.publish(native_result()[0], request(), llm)
    phase = DisaggregatedParams(request_type="context_only")
    _, params = cache.prepare(ref, request(), llm, None, phase)
    assert params is phase
    assert params.request_type == "context_only"
    assert seen["hashes"] == [[2147483647] * 8]
    assert not list(tmp_path.glob("*.safetensors"))


def test_expiry_wrong_inputs_and_cross_host_are_rejected(tmp_path):
    cache = oe.EncoderCache(str(tmp_path))
    llm, _ = model()
    result, _ = native_result()
    ref = cache.publish(result, request(), llm)
    value = json.loads(ref.opaque_state)
    value["host"] = "another-host"
    bad = pb.EncoderCacheRef(transfer_backend=oe.BACKEND, opaque_state=json.dumps(value).encode())
    with pytest.raises(ValueError, match="same host"):
        cache.prepare(bad, request(), llm, None)
    wrong = request()
    wrong.prompt = "Another image prompt"
    with pytest.raises(ValueError, match="does not match"):
        cache.prepare(ref, wrong, llm, None)
    result, _ = native_result()
    llm._hf_model_dir = tmp_path / "checkpoint-a"
    ref = cache.publish(result, request(), llm)
    llm._hf_model_dir = tmp_path / "checkpoint-b"
    with pytest.raises(ValueError, match="another model"):
        cache.prepare(ref, request(), llm, None)
    result, _ = native_result()
    ref = cache.publish(result, request(), llm)
    path = next(tmp_path.glob("*.safetensors"))
    os.utime(path, (1, 1))
    with pytest.raises(ValueError, match="expired"):
        cache.prepare(ref, request(), llm, None)


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
    req.encoder.encode_only = True
    assert oe.validate_encoder(req, "encode") == "encode"
    with pytest.raises(UnsupportedFeatureError):
        oe.validate_encoder(req, "")
    req.encoder.cache.transfer_backend = oe.BACKEND
    with pytest.raises(ValueError, match="exactly one"):
        oe.validate_encoder(req, "encode")
    req.ClearField("encoder")
    with pytest.raises(UnsupportedFeatureError, match="cache reference"):
        oe.validate_encoder(req, "consume")
    req.kv.session.session_id = "123"
    assert oe.validate_encoder(req, "consume") == ""
    req.encoder.cache.transfer_backend = oe.BACKEND
    with pytest.raises(UnsupportedFeatureError, match="must not consume an encoder cache"):
        oe.validate_encoder(req, "consume")
