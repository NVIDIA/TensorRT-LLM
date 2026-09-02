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

"""Optional attention dependencies used by DFlash's private context cache."""

from dataclasses import dataclass
from functools import lru_cache
from typing import Callable, Optional

import torch

from tensorrt_llm._torch.flashinfer_utils import IS_FLASHINFER_AVAILABLE
from tensorrt_llm._utils import get_sm_version, is_sm_100f


@dataclass(frozen=True)
class DFlashTrtllmGenOps:
    """TRTLLM-Gen operations loaded only when DFlash selects that backend."""

    append_paged_kv_cache: Callable[..., None]
    batch_context_with_kv_cache: Callable[..., None]
    batch_decode_with_kv_cache: Callable[..., None]
    get_multi_ctas_kv_counter_size: Callable[..., int]
    get_workspace_size: Callable[..., int]


@lru_cache(maxsize=1)
def get_dflash_flash_attention() -> Callable[..., torch.Tensor]:
    """Load the contiguous-cache DFlash attention implementation."""
    try:
        from flash_attn import flash_attn_with_kvcache
    except ImportError as error:
        raise RuntimeError("DFlash VANILLA attention requires the flash-attn package.") from error
    return flash_attn_with_kvcache


@lru_cache(maxsize=1)
def get_dflash_paged_append() -> Callable[..., None]:
    """Load flashinfer's paged K/V append, shared by the paged backends.

    Deliberately separate from :func:`get_dflash_trtllm_gen_ops`: the append is
    a plain scatter into an HND page pool and works wherever flashinfer does,
    while the TRTLLM-Gen FMHA kernels additionally require SM100/SM103.
    """
    if not IS_FLASHINFER_AVAILABLE:
        raise RuntimeError(
            "DFlash paged context cache requires flashinfer, which is not installed."
        )
    import flashinfer

    return flashinfer.page.append_paged_kv_cache


@lru_cache(maxsize=1)
def get_dflash_fa4_fwd() -> Callable[..., tuple]:
    """Load the FlashAttention-4 (CuTe DSL) forward."""
    try:
        from flash_attn.cute.interface import _flash_attn_fwd
    except (ImportError, OSError) as error:
        raise RuntimeError(
            "DFlash FA4 attention requires a flash-attn build with the CuTe DSL "
            "interface (flash_attn.cute)."
        ) from error
    return _flash_attn_fwd


def validate_dflash_fa4_runtime(
    *,
    dtype: torch.dtype,
    head_dim: int,
) -> None:
    """Fail before cache allocation when DFlash's shape is unsupported by FA4."""
    get_dflash_fa4_fwd()
    get_dflash_paged_append()

    if dtype not in (torch.float16, torch.bfloat16):
        raise RuntimeError(f"DFlash FA4 attention does not support activation dtype {dtype}.")

    # FA4 builds kernels for other archs too, but this backend has
    # only been validated on SM90 (H100); SM120 and friends keep VANILLA.
    sm = get_sm_version()
    if sm != 90:
        raise RuntimeError(
            f"DFlash FA4 attention backend is supported on SM90 only, got SM{sm}. "
            "Use attention_backend='VANILLA'."
        )
    # Mirrors flash_attn.cute.interface._validate_head_dims for SM90.
    if not (8 <= head_dim <= 256) or head_dim % 8 != 0:
        raise RuntimeError(f"DFlash FA4 attention does not support head_dim={head_dim} on SM90.")


def _get_trtllm_gen_unavailability_reason() -> Optional[str]:
    """Return why the DFlash TRTLLM backend cannot be initialized."""
    if not IS_FLASHINFER_AVAILABLE:
        return "flashinfer is not installed"

    from ..attention_backend.fmha.flashinfer_trtllm_gen import FlashInferTrtllmGenFmha

    missing_ops = FlashInferTrtllmGenFmha._missing_fused_nanobind_ops()
    if missing_ops:
        return "missing fused nanobind ops: " + ", ".join(missing_ops)

    sm = get_sm_version()
    if not is_sm_100f(sm):
        return f"requires SM100 or SM103, got SM{sm}"

    return None


@lru_cache(maxsize=1)
def get_dflash_trtllm_gen_ops() -> DFlashTrtllmGenOps:
    """Load TRTLLM-Gen operations after validating common prerequisites."""
    unavailable_reason = _get_trtllm_gen_unavailability_reason()
    if unavailable_reason is not None:
        raise RuntimeError(f"DFlash TRTLLM attention backend is unavailable: {unavailable_reason}.")

    import flashinfer

    from ..attention_backend.fmha.flashinfer_trtllm_gen import (
        _get_multi_ctas_kv_counter_size,
        _get_workspace_size,
    )

    return DFlashTrtllmGenOps(
        append_paged_kv_cache=flashinfer.page.append_paged_kv_cache,
        batch_context_with_kv_cache=flashinfer.prefill.trtllm_batch_context_with_kv_cache,
        batch_decode_with_kv_cache=flashinfer.decode.trtllm_batch_decode_with_kv_cache,
        get_multi_ctas_kv_counter_size=_get_multi_ctas_kv_counter_size,
        get_workspace_size=_get_workspace_size,
    )


def validate_dflash_trtllm_gen_runtime(
    *,
    dtype: torch.dtype,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    tokens_per_block: int,
    has_context_attention: bool,
) -> None:
    """Fail before cache allocation when DFlash's kernel shape is unsupported."""
    from ..attention_backend.fmha.flashinfer_trtllm_gen import FlashInferTrtllmGenFmha

    get_dflash_trtllm_gen_ops()
    if dtype not in FlashInferTrtllmGenFmha.SUPPORTED_INPUT_DTYPES:
        raise RuntimeError(f"DFlash TRTLLM attention does not support activation dtype {dtype}.")
    if num_heads <= 0 or num_kv_heads <= 0 or num_heads % num_kv_heads != 0:
        raise RuntimeError(
            "DFlash TRTLLM attention requires positive head counts with "
            f"num_heads divisible by num_kv_heads; got {num_heads} and "
            f"{num_kv_heads}."
        )
    heads_ratio = num_heads // num_kv_heads
    if heads_ratio > FlashInferTrtllmGenFmha.MAX_HEADS_RATIO_GENERATION:
        raise RuntimeError(
            "DFlash TRTLLM attention does not support a Q/KV head ratio of "
            f"{heads_ratio}; maximum is "
            f"{FlashInferTrtllmGenFmha.MAX_HEADS_RATIO_GENERATION}."
        )
    if tokens_per_block not in FlashInferTrtllmGenFmha.SUPPORTED_TOKENS_PER_BLOCK:
        supported = sorted(FlashInferTrtllmGenFmha.SUPPORTED_TOKENS_PER_BLOCK)
        raise RuntimeError(
            "DFlash TRTLLM attention does not support context-cache page size "
            f"{tokens_per_block}; supported sizes are {supported}."
        )
    if has_context_attention and head_dim in FlashInferTrtllmGenFmha.UNSUPPORTED_HEAD_SIZES_CONTEXT:
        raise RuntimeError(
            f"DFlash TRTLLM context attention does not support head dimension {head_dim}."
        )
