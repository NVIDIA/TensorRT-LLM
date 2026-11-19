# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Availability, tile configurations, autotuning and positions for the Prims-TS GEMM callers."""

from __future__ import annotations

import hashlib
import os
from contextlib import AbstractContextManager, nullcontext
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Optional, Tuple

import torch
from torch import Tensor

if TYPE_CHECKING:
    from flashinfer.prims_ts.gemm import PrimsTsGemmConfig

# Prims-TS dense GEMM targets (FlashInfer: SM100, SM103, SM107).
_SUPPORTED_SM = (100, 103, 107)
# Upper token-count bounds of the M buckets a tactic table is keyed by.
_M_BUCKETS = (1024, 8192, 10240, 32768, 40960)

# (operand_format, epilogue, K, N) -> {M bucket: (cluster, tmem_overlap, ab_stages,
# epilogue_warps, use_tma_store)}, registered by the models that tuned them.
_TACTICS: Dict[Tuple[str, str, int, int], Dict[int, Tuple[int, bool, int, int, bool]]] = {}


def device_supported(device: torch.device) -> bool:
    """Whether ``device`` is a GPU the Prims-TS dense GEMMs target (SM100/103/107)."""
    if device.type != "cuda":
        return False
    major, minor = torch.cuda.get_device_capability(device)
    return major * 10 + minor in _SUPPORTED_SM


@lru_cache(maxsize=None)
def fp8_gemm_available() -> bool:
    """Whether FlashInfer ships the Prims-TS FP8 SwiGLU and QKV+QK-norm+RoPE GEMMs."""
    try:
        from flashinfer.prims_ts import gemm
    except ImportError:
        return False
    return all(
        callable(getattr(gemm, name, None))
        for name in ("fp8_linear_swiglu", "fp8_qkv_qknorm_rope", "PrimsTsGemmConfig")
    )


@lru_cache(maxsize=None)
def nvfp4_qkv_available() -> bool:
    """Whether FlashInfer ships the prepared Prims-TS NVFP4 QKV+QK-norm+RoPE GEMM."""
    try:
        from flashinfer.prims_ts import gemm
    except ImportError:
        return False
    return callable(getattr(gemm, "prepare_fp4_qkv_qknorm_rope", None))


def register_tactics(
    operand_format: str,
    table: Dict[Tuple[str, int, int], Dict[int, Tuple[int, bool, int, int, bool]]],
) -> None:
    """Register tuned tiles, keyed by (epilogue, K, N) then M bucket."""
    for (epilogue, k, n), buckets in table.items():
        _TACTICS[(operand_format, epilogue, k, n)] = dict(buckets)
    _gemm_config.cache_clear()


def m_bucket(m: int) -> int:
    """The tactic-table M bucket of ``m`` tokens: the smallest bound >= m, else the largest."""
    return next((bucket for bucket in _M_BUCKETS if m <= bucket), _M_BUCKETS[-1])


@lru_cache(maxsize=None)
def _gemm_config(
    arch: int,
    operand_format: str,
    epilogue: str,
    bucket: int,
    k: int,
    n: int,
    head_dim: Optional[int] = None,
    qkv_scale: bool = False,
) -> "PrimsTsGemmConfig":
    """Tile configuration for one GEMM shape; untuned shapes use the library defaults."""
    from flashinfer.prims_ts.gemm import PrimsTsGemmConfig

    qkv = epilogue == "qkv_qknorm_rope"
    kwargs = dict(
        arch=arch,
        operand_format=operand_format,
        output_format="bf16",
        epilogue=epilogue,
        has_bias=False,
        head_dim=head_dim if qkv else None,
        is_neox=False if qkv else None,
        has_qkv_scale=qkv_scale,
        tile_k=128 if operand_format == "fp8_e4m3" else 256,
    )
    tactic = _TACTICS.get((operand_format, epilogue, k, n), {}).get(bucket)
    if tactic is not None:
        cluster, overlap, stages, warps, tma = tactic
        kwargs.update(
            cluster_shape=(cluster, cluster, 1),
            tmem_overlap=overlap,
            ab_stages=stages,
            epilogue_warps=warps,
            use_tma_store=tma,
        )
    if qkv and operand_format == "nvfp4_e2m1":
        # NVFP4 writes three contiguous [M, N/3] planes with direct stores. FP8 keeps the
        # packed [M, N] output with TMA stores, which is faster for its epilogue.
        kwargs.update(separate_qkv_output=True, use_tma_store=False)
    return PrimsTsGemmConfig(**kwargs)


def config_for(
    device: torch.device,
    operand_format: str,
    epilogue: str,
    m: int,
    k: int,
    n: int,
    head_dim: Optional[int] = None,
    qkv_scale: bool = False,
) -> Optional["PrimsTsGemmConfig"]:
    """Tile configuration for an [m, k] x [n, k] GEMM on ``device``, keyed by its M bucket.

    None under ``autotune_enabled()``, for FlashInfer's autotuner to pick, except for the NVFP4
    Q/K/V planes: the tuner's configs write packed Q|K|V.
    """
    planes = epilogue == "qkv_qknorm_rope" and operand_format == "nvfp4_e2m1"
    if autotune_enabled() and not planes:
        return None
    major, minor = torch.cuda.get_device_capability(device)
    return _gemm_config(
        major * 10 + minor, operand_format, epilogue, m_bucket(m), k, n, head_dim, qkv_scale
    )


def autotune_enabled() -> bool:
    """Whether FlashInfer autotunes the Prims-TS GEMMs (``TRTLLM_FLUX3_PRIMS_TS_AUTOTUNE=1``)."""
    return os.environ.get("TRTLLM_FLUX3_PRIMS_TS_AUTOTUNE", "0") == "1"


def autotune_context(
    device: torch.device, *, tune: bool = False, load: bool = False
) -> AbstractContextManager:
    """FlashInfer autotuning of the Prims-TS GEMMs; a no-op unless ``autotune_enabled()``.

    ``tune`` profiles uncached shapes and saves the winners to one file per GPU name under
    ``TRTLLM_FLUX3_PRIMS_TS_CACHE_DIR``; ``load`` reads that file first; otherwise tuned tactics
    are only looked up. M rounds up to the tactic-table buckets, or to the comma-separated
    ``TRTLLM_FLUX3_PRIMS_TS_TUNING_BUCKETS``.
    """
    if not autotune_enabled():
        return nullcontext()
    from flashinfer import autotune

    raw = os.environ.get("TRTLLM_FLUX3_PRIMS_TS_TUNING_BUCKETS")
    buckets = tuple(sorted({int(b) for b in raw.split(",")})) if raw else _M_BUCKETS
    cache = str(_autotune_cache(device)) if tune or load else None
    return autotune(tune, cache=cache, tuning_buckets=buckets, round_up=True)


def _autotune_cache(device: torch.device) -> Path:
    root = os.environ.get("TRTLLM_FLUX3_PRIMS_TS_CACHE_DIR", "~/.cache/tensorrt_llm/flux3_prims_ts")
    folder = Path(root).expanduser() / torch.cuda.get_device_name(device).replace(" ", "_")
    folder.mkdir(parents=True, exist_ok=True)
    return folder / f"gemm-{_gemm_source_digest()}.json"


@lru_cache(maxsize=None)
def _gemm_source_digest() -> str:
    """Digest of FlashInfer's Prims-TS GEMM sources, so a kernel change drops saved tactics."""
    from flashinfer.prims_ts import gemm

    digest = hashlib.sha256()
    for source in sorted(Path(gemm.__file__).parent.glob("*.py")):
        digest.update(source.name.encode())
        digest.update(source.read_bytes())
    return digest.hexdigest()[:16]


# Never evicted: captured CUDA graphs keep reading these buffers by address.
@lru_cache(maxsize=None)
def _positions_buffer(size: int, device: torch.device) -> Tensor:
    return torch.arange(size, device=device, dtype=torch.int64)


def token_positions(rows: int, device: torch.device) -> Tensor:
    """Identity RoPE indices [rows], sliced from a power-of-two buffer per device."""
    return _positions_buffer(1 << max(rows - 1, 0).bit_length(), device)[:rows]
