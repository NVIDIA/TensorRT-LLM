# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import os
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors import safe_open
from utils.native_encoder import (
    _check_handoff_hash_initialization,
    _check_malformed_handoff_hashes,
    model,
    native_result,
    request,
)

from tensorrt_llm._torch.disaggregation import encoder_output as oe
from tensorrt_llm._torch.shared_tensor import SharedTensorContainer
from tensorrt_llm.disaggregated_params import DisaggregatedParams

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


@pytest.mark.parametrize("supplied", [False, True])
def test_shared_handoff_restores_hash_validation_and_missing_hash_fallback(tmp_path, supplied):
    cache = oe.EncoderCache(str(tmp_path))
    _check_handoff_hash_initialization(cache, cache, supplied)


@pytest.mark.parametrize("hashes", [[], [[1] * 7], [[1.0] * 8], ["bad"], 42, [[1] * 8] * 2])
def test_shared_handoff_rejects_malformed_hashes_and_releases_exports(
    tmp_path, monkeypatch, hashes
):
    cache = oe.EncoderCache(str(tmp_path))
    _check_malformed_handoff_hashes(cache, cache, hashes, monkeypatch)


@pytest.mark.parametrize("location", ["top", "output", "both"])
def test_prefill_result_reclaims_new_exports_once_and_leaves_consumed_inputs(monkeypatch, location):
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
        result = SimpleNamespace(
            disaggregated_params=(fresh.disaggregated_params if location != "output" else params),
            outputs=[SimpleNamespace(disaggregated_params=fresh.disaggregated_params)]
            if location != "top"
            else [],
        )
        assert owner.has_exports(result)
        assert owner.release(result)
        assert released == handles
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


@pytest.mark.parametrize("failure", [False, True])
def test_encoder_release_reports_failure_and_deduplicates_output_exports(monkeypatch, failure):
    result, _tensors = native_result()
    params = result.disaggregated_params
    handles = [
        *params.multimodal_embedding_handles,
        params.mrope_position_ids_handle,
        params.mrope_position_deltas_handle,
    ]
    result.outputs = [
        SimpleNamespace(
            disaggregated_params=DisaggregatedParams(
                multimodal_embedding_handles=[dict(handles[0])],
                mrope_position_ids_handle=dict(handles[1]),
                mrope_position_deltas_handle=dict(handles[2]),
            )
        )
    ]
    release = SharedTensorContainer.release_dict
    released = []

    def record(handle):
        released.append(handle)
        if failure and handle == handles[0]:
            raise RuntimeError("release failed")
        release(handle)

    monkeypatch.setattr(SharedTensorContainer, "release_dict", record)
    assert oe.release_encoder_result(result) is not failure
    assert released == handles
    assert oe.release_encoder_result(result)
    assert released == handles
    if failure:
        release(handles[0])


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
    phase = DisaggregatedParams(
        request_type="context_only", disagg_request_id=71, ctx_usage={"prompt_tokens": 88}
    )
    _, params = cache.prepare(ref, request(), llm, None, phase)
    assert params.request_type == "context_only"
    assert params.disagg_request_id == phase.disagg_request_id
    assert params.ctx_usage == phase.ctx_usage
    assert seen["hashes"] == [[2147483647] * 8]
    assert not list(tmp_path.glob("*.safetensors"))


def test_expiry_wrong_inputs_and_cross_host_are_rejected(tmp_path):
    cache = oe.EncoderCache(str(tmp_path))
    llm, _ = model()
    result, _ = native_result()
    ref = cache.publish(result, request(), llm)
    value = json.loads(ref.opaque_state)
    value["host"] = "another-host"
    bad = oe.EncodedMediaRef(reference_format=oe.BACKEND, opaque_state=json.dumps(value).encode())
    with pytest.raises(ValueError, match="same host"):
        cache.prepare(bad, request(), llm, None)
    wrong = replace(request(), prompt="Another image prompt")
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
    req = replace(request(), prompt=None, token_ids=(1, 2))
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


@pytest.mark.parametrize("role", ["encode", "consume"])
@pytest.mark.parametrize("model_type", ["qwen3_vl", "nemotron_h", "qwen2_vl", ""])
def test_encoder_role_requires_a_validated_handoff_processor(role, model_type):
    llm, _ = model()
    llm.args = SimpleNamespace(
        mm_encoder_only=role == "encode", disable_mm_encoder=role == "consume"
    )
    llm.input_processor.config.model_type = model_type

    assert oe.encoder_role(llm, object()) == (role if model_type == "qwen3_vl" else "")


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
        malformed = oe.EncodedMediaRef(
            reference_format=oe.BACKEND,
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
    malformed = oe.EncodedMediaRef(reference_format=oe.BACKEND, opaque_state=b"[" * 1024)

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


@pytest.mark.parametrize("prompt,token_ids", [(None, None), ("context", (1, 2))])
def test_native_input_requires_exactly_one_context(prompt, token_ids):
    with pytest.raises(ValueError, match="prompt or token IDs"):
        oe.EncoderInput(prompt, token_ids, request().media)
