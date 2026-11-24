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

from tensorrt_llm._torch.cute_dsl_utils import install_cutlass_dsl_compatibility
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
        install_cutlass_dsl_compatibility()
        from flash_attn.cute.interface import _flash_attn_fwd
    except (ImportError, OSError, AttributeError) as error:
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


def dflash_trtllm_gen_unavailability_reason() -> Optional[str]:
    """Return why the DFlash TRTLLM backend cannot be initialized, or None.

    Public because ``attention_backend="AUTO"`` resolves through it: a
    drafter that prefers TRTLLM has to know whether to fall back before it
    commits, and ``get_dflash_trtllm_gen_ops`` only reports by raising.
    """
    if not IS_FLASHINFER_AVAILABLE:
        return "flashinfer is not installed"

    from tensorrt_llm._torch.attention.backends.fmha.flashinfer_trtllm_gen import (
        FlashInferTrtllmGenFmha,
    )

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
    unavailable_reason = dflash_trtllm_gen_unavailability_reason()
    if unavailable_reason is not None:
        raise RuntimeError(f"DFlash TRTLLM attention backend is unavailable: {unavailable_reason}.")

    import flashinfer

    from tensorrt_llm._torch.attention.backends.fmha.flashinfer_trtllm_gen import (
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
    from tensorrt_llm._torch.attention.backends.fmha.flashinfer_trtllm_gen import (
        FlashInferTrtllmGenFmha,
    )

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


# FP8 E4M3 storage of an MLA drafter's context latent: one per-tensor scale of
# 1.0 for the latent and for the query that reads it -- the scheme of the
# target MLA's FP8 KV cache without calibrated kv scales (quant_scale_kv and
# quant_scale_q default to 1, bmm1_scale = softmax_scale * dq_q * dq_kv,
# bmm2_scale = dq_kv). Values saturate at the format's largest finite value, as
# the target's FP8 cache writes do, instead of overflowing to NaN.
DRAFT_FP8_MAX = float(torch.finfo(torch.float8_e4m3fn).max)
DRAFT_FP8_KV_SCALE = 1.0


def to_draft_fp8(x: torch.Tensor) -> torch.Tensor:
    """``x`` as saturated FP8 E4M3 at the drafter's unit per-tensor scale."""
    return x.clamp(-DRAFT_FP8_MAX, DRAFT_FP8_MAX).to(torch.float8_e4m3fn)


@dataclass(frozen=True)
class HelixDraftShard:
    """One rank's share of a drafter's paged context KV under helix.

    Global page ``g`` of a request belongs to CP rank ``g % cp_size``, which
    stores its pages back to back: local page ``g // cp_size``. Every helper
    maps global token positions or lengths into that rank-local layout.
    """

    cp_size: int
    cp_rank: int
    page_size: int

    def local_slots(self, positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Rank-local slot of each global position, and whether this rank owns it.

        An unowned position maps to the slot this rank keeps for the token one
        or more pages away, never one of the ``page_size`` tokens around it.
        """
        page = torch.div(positions, self.page_size, rounding_mode="floor")
        slots = torch.div(page, self.cp_size, rounding_mode="floor") * self.page_size
        return slots + positions % self.page_size, page % self.cp_size == self.cp_rank

    def local_lengths(self, lengths: torch.Tensor) -> torch.Tensor:
        """Number of owned positions below each global length."""
        span = self.page_size * self.cp_size
        rem = lengths % span - self.cp_rank * self.page_size
        return torch.div(lengths, span, rounding_mode="floor") * self.page_size + rem.clamp(
            0, self.page_size
        )


def combine_helix_partials(
    out: torch.Tensor, lse: torch.Tensor, empty: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Merge per-rank partial attention over disjoint KV shards.

    Args:
        out: ``[cp, N, H, D]`` partial outputs, one per CP rank.
        lse: ``[cp, N, H]`` base-2 log-sum-exp of the matching scores, as
            TRTLLM-Gen returns it.
        empty: optional ``[cp, N]`` bool, True where a rank held no keys for
            a row; its partial is ignored whatever the kernel wrote.
    Returns:
        ``[N, H, D]`` in ``out``'s dtype.
    """
    lse = lse.float()
    if empty is not None:
        lse = lse.masked_fill(empty.unsqueeze(-1), float("-inf"))
    peak = lse.amax(dim=0, keepdim=True)
    weight = torch.exp2(lse - peak.clamp_min(torch.finfo(torch.float32).min))
    weight = weight.nan_to_num(nan=0.0)
    total = weight.sum(dim=0).clamp_min(torch.finfo(torch.float32).tiny)
    merged = (weight.unsqueeze(-1) * out.float().nan_to_num(0.0)).sum(dim=0)
    return (merged / total.unsqueeze(-1)).to(out.dtype)
