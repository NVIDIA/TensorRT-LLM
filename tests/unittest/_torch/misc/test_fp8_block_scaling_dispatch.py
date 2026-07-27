# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import torch

from tensorrt_llm._torch.custom_ops.fp8_block_scaling_dispatch import (
    ActivationScaleLayout,
    BackendMeasurement,
    CacheIdentity,
    DispatchBackend,
    DispatchCache,
    DispatchEntry,
    DispatchKey,
    DispatchPolicy,
    DispatchReason,
    MatrixLayout,
    WeightScaleLayout,
    choose_fastest_correct,
    select_backend,
)
from tensorrt_llm._torch.custom_ops.fp8_block_scaling_dispatch_cache import (
    load_dispatch_cache,
    write_dispatch_cache,
)
from tensorrt_llm._torch.custom_ops.fp8_block_scaling_dispatch_runtime import (
    is_deep_gemm_compatible,
)


def _identity(*, build_id: str = "build-a") -> CacheIdentity:
    return CacheIdentity(
        sm=90,
        device_class="H200",
        trtllm_version="1.3.0rc17",
        trtllm_build_id=build_id,
        deep_gemm_version="2.5.0",
        deep_gemm_available=True,
        policy_version=1,
        backend_candidates=("sm90_trt", "sm90_deep_gemm_1d2d"),
    )


def _key(m: int, n: int = 1536, k: int = 1536) -> DispatchKey:
    return DispatchKey(
        m=m,
        n=n,
        k=k,
        activation_scale_layout=ActivationScaleLayout.LOGICAL_M_K_BLOCKS,
        weight_scale_layout=WeightScaleLayout.LOGICAL_N_K_BLOCKS,
        matrix_layout=MatrixLayout.K_MAJOR_CONTIGUOUS,
    )


def _cache(*entries: DispatchEntry, identity: CacheIdentity | None = None) -> DispatchCache:
    return DispatchCache(identity=identity or _identity(), entries=entries)


def test_small_m_guard_precedes_deep_gemm_cache_hit() -> None:
    # Given
    key = _key(512)
    cache = _cache(DispatchEntry(key=key, backend=DispatchBackend.DEEP_GEMM))

    # When
    decision = select_backend(DispatchPolicy(), key, _identity(), cache)

    # Then
    assert decision.backend is DispatchBackend.TRTLLM
    assert decision.reason is DispatchReason.SMALL_M


def test_exact_denylist_precedes_deep_gemm_cache_hit() -> None:
    # Given
    key = _key(65536, n=3072, k=3072)
    cache = _cache(DispatchEntry(key=key, backend=DispatchBackend.DEEP_GEMM))

    # When
    decision = select_backend(DispatchPolicy(), key, _identity(), cache)

    # Then
    assert decision.backend is DispatchBackend.TRTLLM
    assert decision.reason is DispatchReason.DENYLIST


def test_exact_cache_keeps_distinct_large_m_decisions() -> None:
    # Given
    key_16k = _key(16384)
    key_64k = _key(65536)
    cache = _cache(
        DispatchEntry(key=key_16k, backend=DispatchBackend.TRTLLM),
        DispatchEntry(key=key_64k, backend=DispatchBackend.DEEP_GEMM),
    )

    # When
    decision_16k = select_backend(DispatchPolicy(), key_16k, _identity(), cache)
    decision_64k = select_backend(DispatchPolicy(), key_64k, _identity(), cache)

    # Then
    assert decision_16k.backend is DispatchBackend.TRTLLM
    assert decision_64k.backend is DispatchBackend.DEEP_GEMM
    assert decision_16k.reason is DispatchReason.CACHE_HIT
    assert decision_64k.reason is DispatchReason.CACHE_HIT


def test_capture_guard_precedes_deep_gemm_cache_hit() -> None:
    # Given
    key = _key(16384)
    cache = _cache(DispatchEntry(key=key, backend=DispatchBackend.DEEP_GEMM))

    # When
    decision = select_backend(DispatchPolicy(), key, _identity(), cache, is_capturing=True)

    # Then
    assert decision.backend is DispatchBackend.TRTLLM
    assert decision.reason is DispatchReason.CAPTURE


def test_layout_revalidation_falls_back_from_cached_deep_gemm() -> None:
    # Given
    key = _key(16384)
    cache = _cache(DispatchEntry(key=key, backend=DispatchBackend.DEEP_GEMM))

    # When
    decision = select_backend(DispatchPolicy(), key, _identity(), cache, deep_gemm_compatible=False)

    # Then
    assert decision.backend is DispatchBackend.TRTLLM
    assert decision.reason is DispatchReason.UNSUPPORTED_LAYOUT


def test_k_below_block_size_is_not_deep_gemm_compatible() -> None:
    # Given
    key = _key(6889, n=3072, k=64)
    tensors = (torch.empty(0), torch.empty(0), torch.empty(0), torch.empty(0))

    # When / Then
    assert not is_deep_gemm_compatible(key, tensors)


def test_cache_identity_mismatch_falls_back_to_trtllm() -> None:
    # Given
    key = _key(16384)
    stale_cache = _cache(
        DispatchEntry(key=key, backend=DispatchBackend.DEEP_GEMM),
        identity=_identity(build_id="stale-build"),
    )

    # When
    decision = select_backend(DispatchPolicy(), key, _identity(), stale_cache)

    # Then
    assert decision.backend is DispatchBackend.TRTLLM
    assert decision.reason is DispatchReason.CACHE_IDENTITY_MISMATCH


def test_cache_miss_falls_back_to_trtllm() -> None:
    # Given
    key = _key(16384)

    # When
    decision = select_backend(DispatchPolicy(), key, _identity(), _cache())

    # Then
    assert decision.backend is DispatchBackend.TRTLLM
    assert decision.reason is DispatchReason.CACHE_MISS


def test_dispatch_cache_json_round_trip_preserves_exact_keys(tmp_path: Path) -> None:
    # Given
    cache_path = tmp_path / "fp8_dispatch.json"
    cache = _cache(
        DispatchEntry(key=_key(16384), backend=DispatchBackend.DEEP_GEMM),
        DispatchEntry(key=_key(65536), backend=DispatchBackend.TRTLLM),
    )

    # When
    write_dispatch_cache(cache_path, cache)
    loaded = load_dispatch_cache(cache_path)

    # Then
    assert loaded == cache


def test_fast_incorrect_backend_is_rejected_by_cache_builder() -> None:
    # Given
    measurements = (
        BackendMeasurement(backend=DispatchBackend.TRTLLM, correct=True, median_ms=1.0),
        BackendMeasurement(backend=DispatchBackend.DEEP_GEMM, correct=False, median_ms=0.5),
    )

    # When
    selected = choose_fastest_correct(measurements)

    # Then
    assert selected is DispatchBackend.TRTLLM
