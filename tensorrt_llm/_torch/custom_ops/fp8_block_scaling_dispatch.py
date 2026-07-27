# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic policy for correctness-gated FP8 block-scaling GEMM dispatch."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType


class DispatchBackend(str, Enum):
    TRTLLM = "trtllm"
    DEEP_GEMM = "deep_gemm"


class DispatchReason(str, Enum):
    CAPTURE = "capture"
    UNSUPPORTED_ARCH = "unsupported_arch"
    SMALL_M = "small_m"
    DENYLIST = "denylist"
    CACHE_IDENTITY_MISMATCH = "cache_identity_mismatch"
    CACHE_MISS = "cache_miss"
    BACKEND_UNAVAILABLE = "backend_unavailable"
    UNSUPPORTED_LAYOUT = "unsupported_layout"
    CACHE_HIT = "cache_hit"


class ActivationScaleLayout(str, Enum):
    LOGICAL_M_K_BLOCKS = "logical_m_k_blocks"
    TRT_PADDED_1D = "trt_padded_1d"
    TRT_TRANSPOSED_K_M = "trt_transposed_k_m"
    UNSUPPORTED = "unsupported"


class WeightScaleLayout(str, Enum):
    LOGICAL_N_K_BLOCKS = "logical_n_k_blocks"
    PACKED_UE8M0 = "packed_ue8m0"
    UNSUPPORTED = "unsupported"


class MatrixLayout(str, Enum):
    K_MAJOR_CONTIGUOUS = "k_major_contiguous"
    UNSUPPORTED = "unsupported"


@dataclass(frozen=True, slots=True)
class CacheIdentity:
    sm: int
    device_class: str
    trtllm_version: str
    trtllm_build_id: str
    deep_gemm_version: str
    deep_gemm_available: bool
    policy_version: int
    backend_candidates: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class DispatchKey:
    m: int
    n: int
    k: int
    activation_scale_layout: ActivationScaleLayout
    weight_scale_layout: WeightScaleLayout
    matrix_layout: MatrixLayout


@dataclass(frozen=True, slots=True)
class DispatchEntry:
    key: DispatchKey
    backend: DispatchBackend


@dataclass(frozen=True, slots=True)
class DispatchCache:
    identity: CacheIdentity
    entries: tuple[DispatchEntry, ...]
    _index: Mapping[DispatchKey, DispatchEntry] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        index = {entry.key: entry for entry in self.entries}
        if len(index) != len(self.entries):
            raise ValueError("FP8 dispatch cache contains duplicate exact keys")
        object.__setattr__(self, "_index", MappingProxyType(index))

    def find(self, key: DispatchKey) -> DispatchEntry | None:
        """Return the exact validated entry for a runtime input signature."""
        return self._index.get(key)


@dataclass(frozen=True, slots=True)
class DispatchPolicy:
    small_m: int = 512
    denylist: frozenset[tuple[int, int, int]] = frozenset({(65536, 3072, 3072)})
    deep_gemm_sms: frozenset[int] = frozenset({90})


@dataclass(frozen=True, slots=True)
class DispatchDecision:
    backend: DispatchBackend
    reason: DispatchReason


@dataclass(frozen=True, slots=True)
class BackendMeasurement:
    backend: DispatchBackend
    correct: bool
    median_ms: float


def choose_fastest_correct(
    measurements: tuple[BackendMeasurement, ...],
) -> DispatchBackend | None:
    """Choose the fastest backend only among correctness-gated results."""
    correct = tuple(measurement for measurement in measurements if measurement.correct)
    if not correct:
        return None
    return min(correct, key=lambda measurement: measurement.median_ms).backend


def select_static_backend(
    policy: DispatchPolicy,
    key: DispatchKey,
    sm: int,
    *,
    is_capturing: bool = False,
) -> DispatchDecision | None:
    """Apply guards that must not inspect or load the persistent cache."""
    if is_capturing:
        return DispatchDecision(DispatchBackend.TRTLLM, DispatchReason.CAPTURE)
    if sm not in policy.deep_gemm_sms:
        return DispatchDecision(DispatchBackend.TRTLLM, DispatchReason.UNSUPPORTED_ARCH)
    if key.m <= policy.small_m:
        return DispatchDecision(DispatchBackend.TRTLLM, DispatchReason.SMALL_M)
    if (key.m, key.n, key.k) in policy.denylist:
        return DispatchDecision(DispatchBackend.TRTLLM, DispatchReason.DENYLIST)
    return None


def select_backend(
    policy: DispatchPolicy,
    key: DispatchKey,
    current_identity: CacheIdentity,
    cache: DispatchCache,
    *,
    is_capturing: bool = False,
    deep_gemm_compatible: bool = True,
) -> DispatchDecision:
    """Select a backend without profiling or mutating runtime state."""
    static_decision = select_static_backend(
        policy, key, current_identity.sm, is_capturing=is_capturing
    )
    if static_decision is not None:
        return static_decision
    if cache.identity != current_identity:
        return DispatchDecision(
            DispatchBackend.TRTLLM,
            DispatchReason.CACHE_IDENTITY_MISMATCH,
        )

    entry = cache.find(key)
    if entry is None:
        return DispatchDecision(DispatchBackend.TRTLLM, DispatchReason.CACHE_MISS)
    if entry.backend is DispatchBackend.TRTLLM:
        return DispatchDecision(DispatchBackend.TRTLLM, DispatchReason.CACHE_HIT)
    if not current_identity.deep_gemm_available:
        return DispatchDecision(DispatchBackend.TRTLLM, DispatchReason.BACKEND_UNAVAILABLE)
    if not deep_gemm_compatible:
        return DispatchDecision(DispatchBackend.TRTLLM, DispatchReason.UNSUPPORTED_LAYOUT)
    return DispatchDecision(DispatchBackend.DEEP_GEMM, DispatchReason.CACHE_HIT)
