# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Runtime adapter for validated FP8 block-scaling GEMM dispatch caches."""

import hashlib
import logging
import os
import threading
from functools import lru_cache
from pathlib import Path

import torch

from tensorrt_llm import deep_gemm
from tensorrt_llm._utils import get_sm_version
from tensorrt_llm.version import __version__ as trtllm_version

from .fp8_block_scaling_dispatch import (
    ActivationScaleLayout,
    CacheIdentity,
    DispatchBackend,
    DispatchCache,
    DispatchDecision,
    DispatchKey,
    DispatchPolicy,
    DispatchReason,
    MatrixLayout,
    WeightScaleLayout,
    select_backend,
    select_static_backend,
)
from .fp8_block_scaling_dispatch_cache import load_dispatch_cache

_LOGGER = logging.getLogger(__name__)
_CACHE_PATH_ENV = "TRTLLM_FP8_BLOCK_SCALING_GEMM_DISPATCH_CACHE"
_BUILD_ID_ENV = "TRTLLM_BUILD_ID"
_SMALL_M_ENV = "TRTLLM_FP8_BLOCK_SCALING_GEMM_SMALL_M"
_DEBUG_ENV = "TRTLLM_FP8_BLOCK_SCALING_GEMM_DISPATCH_DEBUG"
_LEGACY_BACKEND_ENV = "TRTLLM_FP8_BLOCK_SCALING_GEMM_BACKEND"
_POLICY_VERSION = 1
_IDENTITY_LOCK = threading.Lock()
_IDENTITIES: dict[int, CacheIdentity] = {}


def legacy_backend_override_enabled() -> bool:
    """Preserve the existing explicit C++ backend override surface."""
    configured = os.getenv(_LEGACY_BACKEND_ENV, "").strip().lower()
    return configured not in ("", "auto")


def _device_class(device_name: str) -> str:
    for device_class in ("GH200", "H200", "H100", "B200"):
        if device_class in device_name.upper():
            return device_class
    return device_name


@lru_cache(maxsize=1)
def _trtllm_build_id() -> str:
    configured = os.getenv(_BUILD_ID_ENV)
    if configured:
        return configured

    library_path = Path(__file__).parents[2] / "libs" / "libth_common.so"
    if not library_path.is_file():
        return "unknown"

    digest = hashlib.sha256()
    with library_path.open("rb") as library:
        while chunk := library.read(1024 * 1024):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _backend_candidates(sm: int) -> tuple[str, ...]:
    if sm == 90:
        return ("sm90_trt", "sm90_deep_gemm_1d2d")
    return (f"sm{sm}_trt",)


def _deep_gemm_available(reference: torch.Tensor) -> bool:
    try:
        return bool(torch.ops.trtllm.fp8_block_scaling_gemm_deep_gemm_available(reference))
    except AttributeError:
        return False


def make_runtime_identity(reference: torch.Tensor) -> CacheIdentity:
    """Describe the exact binary, device, and backend candidate set."""
    device_index = reference.device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    with _IDENTITY_LOCK:
        cached = _IDENTITIES.get(device_index)
        if cached is not None:
            return cached

    sm = get_sm_version()
    available = _deep_gemm_available(reference)
    candidates = _backend_candidates(sm) if available else (f"sm{sm}_trt",)
    identity = CacheIdentity(
        sm=sm,
        device_class=_device_class(torch.cuda.get_device_name(device_index)),
        trtllm_version=trtllm_version,
        trtllm_build_id=_trtllm_build_id(),
        deep_gemm_version=deep_gemm.__version__,
        deep_gemm_available=available,
        policy_version=_POLICY_VERSION,
        backend_candidates=candidates,
    )
    with _IDENTITY_LOCK:
        return _IDENTITIES.setdefault(device_index, identity)


def _activation_scale_layout(
    scale: torch.Tensor,
    m: int,
    k_blocks: int,
) -> ActivationScaleLayout:
    if scale.dim() == 2 and tuple(scale.shape) == (m, k_blocks):
        return ActivationScaleLayout.LOGICAL_M_K_BLOCKS
    if scale.dim() == 2 and scale.shape[0] == k_blocks and scale.shape[1] >= m:
        return ActivationScaleLayout.TRT_TRANSPOSED_K_M
    m_padded = ((m + 3) // 4) * 4
    if scale.dim() == 1 and scale.numel() >= k_blocks * m_padded:
        return ActivationScaleLayout.TRT_PADDED_1D
    return ActivationScaleLayout.UNSUPPORTED


def make_dispatch_key(
    a: torch.Tensor,
    b: torch.Tensor,
    a_scale: torch.Tensor,
    b_scale: torch.Tensor,
) -> DispatchKey:
    """Classify a runtime call into an exact cache key."""
    m, k = a.shape
    n = b.shape[0]
    k_blocks = k // 128
    activation_layout = _activation_scale_layout(a_scale, m, k_blocks)
    weight_layout = (
        WeightScaleLayout.LOGICAL_N_K_BLOCKS
        if b_scale.dim() == 2 and tuple(b_scale.shape) == ((n + 127) // 128, k_blocks)
        else WeightScaleLayout.UNSUPPORTED
    )
    same_device = a.device == b.device == a_scale.device == b_scale.device
    matrix_layout = (
        MatrixLayout.K_MAJOR_CONTIGUOUS
        if same_device and a.stride(1) == 1 and b.stride(1) == 1
        else MatrixLayout.UNSUPPORTED
    )
    return DispatchKey(
        m=m,
        n=n,
        k=k,
        activation_scale_layout=activation_layout,
        weight_scale_layout=weight_layout,
        matrix_layout=matrix_layout,
    )


def is_deep_gemm_compatible(key: DispatchKey, tensors: tuple[torch.Tensor, ...]) -> bool:
    a, b, a_scale, b_scale = tensors
    return (
        key.k % 128 == 0
        and key.activation_scale_layout is not ActivationScaleLayout.UNSUPPORTED
        and key.weight_scale_layout is WeightScaleLayout.LOGICAL_N_K_BLOCKS
        and key.matrix_layout is MatrixLayout.K_MAJOR_CONTIGUOUS
        and a.is_cuda
        and a.dtype is torch.float8_e4m3fn
        and b.dtype is torch.float8_e4m3fn
        and a_scale.dtype is torch.float32
        and b_scale.dtype is torch.float32
        and a_scale.is_contiguous()
        and b_scale.is_contiguous()
    )


@lru_cache(maxsize=8)
def _load_cache(path: str) -> DispatchCache | None:
    try:
        return load_dispatch_cache(Path(path))
    except (OSError, KeyError, TypeError, ValueError) as error:
        _LOGGER.warning("Ignoring invalid FP8 dispatch cache %s: %s", path, error)
        return None


@lru_cache(maxsize=8)
def _warn_identity_mismatch(cache_identity: CacheIdentity, current_identity: CacheIdentity) -> None:
    _LOGGER.warning(
        "Ignoring FP8 dispatch entries built for a different runtime: cache=%s current=%s",
        cache_identity,
        current_identity,
    )


def _policy() -> DispatchPolicy:
    configured = os.getenv(_SMALL_M_ENV)
    if configured is None:
        return DispatchPolicy()
    small_m = int(configured)
    if small_m < 0:
        raise ValueError(f"{_SMALL_M_ENV} must be non-negative, got {small_m}")
    return DispatchPolicy(small_m=small_m)


@lru_cache(maxsize=1024)
def _log_decision_once(key: DispatchKey, decision: DispatchDecision) -> None:
    _LOGGER.warning(
        "FP8 block-scaling dispatch debug shape=%sx%sx%s backend=%s reason=%s",
        key.m,
        key.n,
        key.k,
        decision.backend.value,
        decision.reason.value,
    )


def _log_decision(key: DispatchKey, decision: DispatchDecision) -> None:
    if os.getenv(_DEBUG_ENV) == "1":
        _log_decision_once(key, decision)


def get_dispatch_decision(
    a: torch.Tensor,
    b: torch.Tensor,
    a_scale: torch.Tensor,
    b_scale: torch.Tensor,
) -> DispatchDecision:
    """Resolve the configured cache entry for one GEMM invocation."""
    key = make_dispatch_key(a, b, a_scale, b_scale)
    policy = _policy()
    is_capturing = torch.cuda.is_current_stream_capturing()
    static_decision = select_static_backend(
        policy, key, get_sm_version(), is_capturing=is_capturing
    )
    if static_decision is not None:
        _log_decision(key, static_decision)
        return static_decision

    cache_path = os.getenv(_CACHE_PATH_ENV)
    if cache_path is None:
        decision = DispatchDecision(DispatchBackend.TRTLLM, DispatchReason.CACHE_MISS)
        _log_decision(key, decision)
        return decision

    identity = make_runtime_identity(a)
    loaded_cache = _load_cache(cache_path)
    if loaded_cache is None:
        cache = DispatchCache(identity=identity, entries=())
    else:
        cache = loaded_cache

    decision = select_backend(
        policy,
        key,
        identity,
        cache,
        is_capturing=False,
        deep_gemm_compatible=is_deep_gemm_compatible(key, (a, b, a_scale, b_scale)),
    )
    if decision.reason is DispatchReason.CACHE_IDENTITY_MISMATCH:
        _warn_identity_mismatch(cache.identity, identity)
    _log_decision(key, decision)
    return decision
