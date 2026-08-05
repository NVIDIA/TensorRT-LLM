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
    CacheIdentity,
    DispatchBackend,
    DispatchCache,
    DispatchDecision,
    DispatchKey,
    DispatchPolicy,
    DispatchReason,
    select_backend,
    select_static_backend,
)
from .fp8_block_scaling_dispatch_cache import load_dispatch_cache
from .fp8_block_scaling_dispatch_inputs import (
    classify_activation_scale_layout,
    is_deep_gemm_compatible,
    is_deep_gemm_tensor_metadata_compatible,
    make_dispatch_key,
)

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

    try:
        return str(torch.ops.trtllm.fp8_block_scaling_gemm_runtime_build_id())
    except (AttributeError, RuntimeError):
        # Keep source-only tooling compatible with an older installed libth_common.
        pass

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


@lru_cache(maxsize=8)
def _policy(configured: str | None) -> DispatchPolicy:
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
    policy = _policy(os.getenv(_SMALL_M_ENV))
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

    entry = cache.find(key)
    deep_gemm_compatible = (
        entry is not None
        and entry.backend is DispatchBackend.DEEP_GEMM
        and is_deep_gemm_compatible(key, (a, b, a_scale, b_scale))
    )
    decision = select_backend(
        policy,
        key,
        identity,
        cache,
        is_capturing=False,
        deep_gemm_compatible=deep_gemm_compatible,
    )
    if decision.reason is DispatchReason.CACHE_IDENTITY_MISMATCH:
        _warn_identity_mismatch(cache.identity, identity)
    _log_decision(key, decision)
    return decision


def get_dispatch_backend(
    a: torch.Tensor,
    b: torch.Tensor,
    a_scale: torch.Tensor,
    b_scale: torch.Tensor,
) -> DispatchBackend:
    """Resolve a backend without allocating a diagnostic decision on the hot path."""
    if os.getenv(_DEBUG_ENV) == "1":
        return get_dispatch_decision(a, b, a_scale, b_scale).backend

    m, k = a.shape
    n = b.shape[0]
    policy = _policy(os.getenv(_SMALL_M_ENV))
    is_capturing = torch.cuda.is_current_stream_capturing()
    sm = get_sm_version()
    if (
        is_capturing
        or sm not in policy.deep_gemm_sms
        or m <= policy.small_m
        or (m, n, k) in policy.denylist
    ):
        return DispatchBackend.TRTLLM

    cache_path = os.getenv(_CACHE_PATH_ENV)
    if cache_path is None:
        return DispatchBackend.TRTLLM

    identity = make_runtime_identity(a)
    cache = _load_cache(cache_path)
    if cache is None:
        return DispatchBackend.TRTLLM
    if cache.identity != identity:
        _warn_identity_mismatch(cache.identity, identity)
        return DispatchBackend.TRTLLM
    deep_gemm_layouts = cache.deep_gemm_activation_layouts(m, n, k)
    if deep_gemm_layouts is None:
        return DispatchBackend.TRTLLM

    k_blocks = k // 128
    activation_layout = classify_activation_scale_layout(a_scale, m, k_blocks)
    if activation_layout not in deep_gemm_layouts:
        return DispatchBackend.TRTLLM
    if not identity.deep_gemm_available:
        return DispatchBackend.TRTLLM
    if k % 128 != 0:
        return DispatchBackend.TRTLLM
    if not is_deep_gemm_tensor_metadata_compatible(
        a,
        b,
        a_scale,
        b_scale,
        n,
        k_blocks,
    ):
        return DispatchBackend.TRTLLM
    return DispatchBackend.DEEP_GEMM
