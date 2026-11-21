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
from safetensors.torch import save_file
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
    claimed = cache.root / ("a" * 32 + "-" + "b" * 32 + ".tmp")
    claimed.write_bytes(b"expired")
    os.utime(claimed, (1, 1))
    retained = cache.root / ("c" * 32 + ".safetensors")
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
def test_shared_handoff_rejects_malformed_hashes_before_import(tmp_path, monkeypatch, hashes):
    cache = oe.EncoderCache(str(tmp_path))
    _check_malformed_handoff_hashes(cache, cache, hashes, monkeypatch)


@pytest.mark.parametrize("invalid", ["metadata", "missing_hashes", "layout", "tensors"])
def test_shared_metadata_rejection_preserves_unclaimed_output(tmp_path, invalid):
    cache = oe.EncoderCache(str(tmp_path))
    llm, _ = model()
    ref = cache.publish(native_result()[0], request(), llm)
    path = cache._path(ref)
    with safe_open(str(path), framework="pt") as stored:
        tensors = {name: stored.get_tensor(name).clone() for name in stored.keys()}
        original = stored.metadata()["handoff"]
    metadata = json.loads(original)
    invalid_tensors = dict(tensors)
    if invalid == "metadata":
        metadata = []
    elif invalid == "missing_hashes":
        del metadata["hashes"]
    elif invalid == "layout":
        metadata["embedding_metadata"] = ["bad"]
    else:
        invalid_tensors.pop("mrope_ids")
    save_file(invalid_tensors, str(path), metadata={"handoff": json.dumps(metadata)})
    with pytest.raises(ValueError, match="encoder.*(metadata|hashes|layout)"):
        cache.prepare(ref, request(), llm, None)
    assert path.exists() and not list(cache.root.glob("*.tmp"))
    save_file(tensors, str(path), metadata={"handoff": original})
    cache.prepare(ref, request(), llm, None)
    assert not path.exists()


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
    assert not list(cache.root.glob("*.safetensors"))
    assert not list(cache.root.glob("*.tmp"))


def test_expiry_wrong_inputs_and_cross_host_are_rejected(tmp_path):
    cache = oe.EncoderCache(str(tmp_path))
    llm, _ = model()
    result, _ = native_result()
    ref = cache.publish(result, request(), llm)
    value = json.loads(ref.opaque_state)
    value["host"] = "another-host"
    bad = oe.EncodedMediaRef(opaque_state=json.dumps(value).encode())
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
    path = next(cache.root.glob("*.safetensors"))
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
    assert not list(cache.root.glob("*.safetensors"))
    assert not list(cache.root.glob("*.tmp"))


def test_quota_counts_actual_serialized_bytes(tmp_path, monkeypatch):
    cache = oe.EncoderCache(str(tmp_path))
    llm, _ = model()
    result, tensors = native_result()
    cache.publish(result, request(), llm)
    path = next(cache.root.glob("*.safetensors"))
    with safe_open(str(path), framework="pt") as stored:
        estimated = sum(t.numel() * t.element_size() for t in tensors) + len(
            stored.metadata()["handoff"]
        )
    monkeypatch.setattr(cache, "_max_cache_bytes", path.stat().st_size + estimated)
    result, _ = native_result()
    with pytest.raises(oe.EncoderCapacityError, match="cache is full"):
        cache.publish(result, request(), llm)
    assert len(list(cache.root.glob("*.safetensors"))) == 1
    assert not list(cache.root.glob("*.tmp"))


def test_failed_shared_publication_reclaims_original_exports(tmp_path, monkeypatch):
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
    assert not list(cache.root.glob("*.safetensors"))
    with pytest.raises(ValueError, match="already consumed"):
        cache.prepare(ref, request(), llm, None)

    ref = cache.publish(native_result()[0], request(), llm)
    preprocess = llm.preprocess

    def prepare_claimed(*args):
        assert list(cache.root.glob("*.tmp"))
        assert not cache.discard(ref)
        assert list(cache.root.glob("*.tmp"))
        return preprocess(*args)

    llm.preprocess = prepare_claimed
    cache.prepare(ref, request(), llm, None)
    assert not cache.discard(ref)
    assert not list(cache.root.glob("*.tmp"))


def test_discard_rejects_foreign_and_malformed_references(tmp_path):
    cache = oe.EncoderCache(str(tmp_path))
    llm, _ = model()
    ref = cache.publish(native_result()[0], request(), llm)
    value = json.loads(ref.opaque_state)
    for key, replacement in (("namespace", "foreign"), ("host", "foreign"), ("id", "../file")):
        malformed = oe.EncodedMediaRef(
            opaque_state=json.dumps({**value, key: replacement}).encode(),
        )
        with pytest.raises(ValueError):
            cache.discard(malformed)
        assert len(list(cache.root.glob("*.safetensors"))) == 1
    assert cache.discard(ref)


@pytest.mark.parametrize("operation", ["discard", "prepare"])
def test_recursive_decode_is_invalid_without_consuming_handoff(tmp_path, monkeypatch, operation):
    cache = oe.EncoderCache(str(tmp_path))
    llm, _ = model()
    ref = cache.publish(native_result()[0], request(), llm)
    malformed = oe.EncodedMediaRef(opaque_state=b"[" * 1024)

    def fail_decode(_):
        raise RecursionError("maximum recursion depth exceeded while decoding JSON")

    with monkeypatch.context() as patch:
        patch.setattr(oe.json, "loads", fail_decode)
        with pytest.raises(ValueError, match="invalid encoder handoff reference"):
            if operation == "discard":
                cache.discard(malformed)
            else:
                cache.prepare(malformed, request(), llm, None)
    assert len(list(cache.root.glob("*.safetensors"))) == 1
    assert cache.discard(ref)


@pytest.mark.parametrize("prompt,token_ids", [(None, None), ("context", (1, 2))])
def test_native_input_requires_exactly_one_context(prompt, token_ids):
    with pytest.raises(ValueError, match="prompt or token IDs"):
        oe.EncoderInput(prompt, token_ids, request().media)


@pytest.mark.parametrize(
    "invalid", ["missing_format", "wrong_format", "missing_version", "wrong_version", "oversized"]
)
def test_native_envelope_rejection_preserves_unclaimed_output(tmp_path, invalid):
    cache = oe.EncoderCache(str(tmp_path))
    llm, _ = model()
    reference = cache.publish(native_result()[0], request(), llm)
    value = json.loads(reference.opaque_state)
    if invalid.startswith("missing_"):
        value.pop(invalid.removeprefix("missing_"))
    elif invalid == "wrong_format":
        value["format"] = "another-engine-output-v1"
    else:
        value["version"] = 2
    payload = json.dumps(value).encode() if invalid != "oversized" else b"x" * 1025
    malformed = oe.EncodedMediaRef(payload)
    for operation in (
        lambda: cache.prepare(malformed, request(), llm, None),
        lambda: cache.discard(malformed),
    ):
        with pytest.raises(ValueError):
            operation()
        assert len(list(cache.root.glob("*.safetensors"))) == 1
    cache.prepare(reference, request(), llm, None)
    assert not list(cache.root.glob("*.safetensors"))


@pytest.mark.parametrize("directory", ["", "   "])
def test_empty_cache_path_does_not_use_working_directory(tmp_path, monkeypatch, directory):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError, match="nonempty"):
        oe.EncoderCache(directory)
    assert not list(tmp_path.iterdir())


def test_collection_only_removes_owned_files_in_private_subdirectory(tmp_path):
    checkpoint = tmp_path / ("a" * 32 + ".safetensors")
    checkpoint.write_bytes(b"checkpoint")
    os.utime(checkpoint, (1, 1))
    cache = oe.EncoderCache(tmp_path)
    unrelated = [cache.root / "model-00001-of-00002.safetensors", cache.root / "other.tmp"]
    expired = cache.root / ("b" * 32 + ".safetensors")
    claimed = cache.root / ("b" * 32 + "-" + "c" * 32 + ".tmp")
    for path in [*unrelated, expired, claimed]:
        path.write_bytes(b"old")
        os.utime(path, (1, 1))
    link = cache.root / ("d" * 32 + ".safetensors")
    link.symlink_to(checkpoint)
    fresh = cache.root / ("e" * 32 + ".safetensors")
    fresh.write_bytes(b"new")
    assert cache._collect() == len(b"new")
    assert checkpoint.read_bytes() == b"checkpoint" and link.is_symlink()
    assert all(path.read_bytes() == b"old" for path in unrelated)
    assert not expired.exists() and not claimed.exists() and fresh.is_file()
    # Construction also collects, so re-opening must preserve unrelated files.
    oe.EncoderCache(str(tmp_path))
    assert checkpoint.exists() and all(path.exists() for path in unrelated)


def test_cache_does_not_follow_a_symlinked_private_subdirectory(tmp_path):
    external = tmp_path / "external"
    external.mkdir()
    (tmp_path / ".trtllm-encoder-cache-v1").symlink_to(external, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        oe.EncoderCache(str(tmp_path))
    assert not list(external.iterdir())


@pytest.mark.parametrize("backend", ["shared", "nixl"])
def test_configured_tensor_budget_rejects_before_taking_exports(tmp_path, monkeypatch, backend):
    from utils.native_encoder import _cache

    monkeypatch.setenv("TRTLLM_ENCODER_MAX_TENSOR_BYTES", "1")
    cache = oe.EncoderCache(str(tmp_path)) if backend == "shared" else _cache("encoder:1")
    result, _ = native_result()
    params = result.disaggregated_params
    original = (
        params.multimodal_embedding_handles,
        params.mrope_position_ids_handle,
        params.mrope_position_deltas_handle,
    )
    llm, _ = model()
    try:
        with monkeypatch.context() as patch:
            patch.setattr(
                SharedTensorContainer,
                "from_dict",
                lambda *_: pytest.fail("oversized exports were imported"),
            )
            with pytest.raises(oe.EncoderCapacityError, match="tensor limit"):
                cache.publish(result, request(), llm)
        assert (
            params.multimodal_embedding_handles,
            params.mrope_position_ids_handle,
            params.mrope_position_deltas_handle,
        ) == original
    finally:
        assert oe.release_encoder_result(result)
        if backend == "nixl":
            cache.close()


@pytest.mark.parametrize("backend", ["shared", "nixl"])
@pytest.mark.parametrize("tensor,cache_bytes", [("0", "100"), ("bad", "100"), ("101", "100")])
def test_invalid_native_cache_budgets_fail_at_construction(
    tmp_path, monkeypatch, backend, tensor, cache_bytes
):
    from utils.native_encoder import _cache

    monkeypatch.setenv("TRTLLM_ENCODER_MAX_TENSOR_BYTES", tensor)
    monkeypatch.setenv("TRTLLM_ENCODER_MAX_CACHE_BYTES", cache_bytes)
    with pytest.raises(ValueError):
        if backend == "shared":
            oe.EncoderCache(str(tmp_path))
        else:
            _cache("encoder:1")
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("backend", ["shared", "nixl"])
def test_native_handoff_accepts_deepstack_output_larger_than_64_mib(tmp_path, backend):
    from utils.native_encoder import _cache, _receive

    producer = oe.EncoderCache(str(tmp_path)) if backend == "shared" else _cache("encoder:1")
    consumer = (
        producer
        if backend == "shared"
        else _cache(request_transfer=lambda _, ref, dest, timeout: producer.transfer(ref, dest))
    )
    tensor = torch.full((2304, 16384), 7, dtype=torch.bfloat16)
    assert tensor.numel() * tensor.element_size() > 64 * 1024 * 1024
    # Avoid requiring a large POSIX shared-memory mount for this CPU size case.
    handle = SharedTensorContainer.from_tensor(tensor, local=True).dump_to_dict()
    handle.update(tensor_size=list(tensor.shape), dtype=str(tensor.dtype))
    params = DisaggregatedParams(multimodal_embedding_handles=[handle], multimodal_hashes=[[1] * 8])
    result = SimpleNamespace(disaggregated_params=params)
    llm, _ = model()
    try:
        ref = producer.publish(result, request(), llm)
        if backend == "shared":
            with safe_open(str(producer._path(ref)), framework="pt") as stored:
                actual = stored.get_tensor("embedding_0")
            assert producer.discard(ref)
        else:
            actual = _receive(consumer, ref)[0]
        assert torch.equal(actual, tensor)
    finally:
        assert oe.release_encoder_result(result)
        if backend == "nixl":
            consumer.close()
            producer.close()
