# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""MiniMax MSA-v1 Q8KV4 sparse decode attention for MiniMax-M3 generation rows (opt-in).

With ``TRTLLM_M3_MSA_V1_DECODE=1``, MsaDecodeFmha hands each sparse generation phase to
``inference.msa_v1.attention.decode.q8kv4`` (MiniMax MSA, MIT) before the Triton/CuTe sparse
decode. The kernel reads the M3 NVFP4 pool in place: packed E2M1 K/V pages of 128 tokens and
their E4M3 block scales, through the same page table, sequence lengths and top-k pages the
Triton kernel takes, and writes the BF16 output directly (no split partials or merge).

The M3 cache writer stores V block scales in the 4x4 token-quad order
(``[token // 4, scale_col, token % 4]``) while upstream MSA reads them token-major, so this
path needs an MSA build that accepts ``MINIMAX_MSA_Q8KV4_V_SCALE_LAYOUT=quad4`` (it exposes
``jit.V_SCALE_LAYOUTS``); this module sets that variable before the first JIT build and refuses
an MSA package without it.

Anything outside the kernel's contract falls back to the existing kernels, call by call: FP8 Q,
GQA 8 or 16, top-k 16, unit K/V per-tensor dequantization scales, contiguous operands, and a
plan built outside CUDA graph capture. Plans are cached in the attention metadata's
``fmha_plan_caches``, one per (batch, query length, metadata buffers), so each captured batch
size reuses the plan it built during its eager warmup.

Installing MSA in the TRT-LLM container (CUDA 13.4 nvcc selects the QMUL4 dequant path)::

    pip install --no-deps <patched MSA checkout>     # or put it on PYTHONPATH
    export CUTLASS_ROOT=<CUTLASS checkout at the MSA submodule commit>
    export TORCH_EXTENSIONS_DIR=<shared JIT cache>   # optional; cold builds take minutes
"""

from __future__ import annotations

import functools
import os
from dataclasses import dataclass
from typing import Optional

import torch

from tensorrt_llm.logger import logger

ENV_VAR = "TRTLLM_M3_MSA_V1_DECODE"
PLAN_CACHE_KEY = "minimax_m3_msa_v1_decode"

_V_SCALE_LAYOUT_ENV = "MINIMAX_MSA_Q8KV4_V_SCALE_LAYOUT"
_V_SCALE_LAYOUT = "quad4"
_PAGE = 128
_HEAD_DIM = 128
_TOPK = 16
_GQA_RATIOS = (8, 16)
_SUPPORTED_CAPABILITIES = ((10, 0), (10, 3))

_ENABLED = os.environ.get(ENV_VAR, "0").strip().lower() in ("1", "true", "on", "yes")
_logged: set = set()


def _log_once(key: str, message: str) -> None:
    if key not in _logged:
        _logged.add(key)
        logger.info(f"MiniMax-M3 MSA-v1 decode: {message}")


@functools.lru_cache(maxsize=1)
def _wrapper_class():
    """Return MSA's Q8KV4 decode wrapper class, or None when the path is off or unusable."""
    if not _ENABLED:
        return None
    configured = os.environ.setdefault(_V_SCALE_LAYOUT_ENV, _V_SCALE_LAYOUT)
    if configured != _V_SCALE_LAYOUT:
        logger.warning(
            f"{ENV_VAR}=1 ignored: the M3 cache stores V scales in 4x4 token-quad order, but "
            f"{_V_SCALE_LAYOUT_ENV}={configured!r}."
        )
        return None
    try:
        from inference.msa_v1.attention.decode.q8kv4 import (
            BatchDecodeWithPagedKVCacheWrapper,
        )
        from inference.msa_v1.attention.decode.q8kv4 import jit as msa_jit
    except ImportError as err:
        logger.warning(f"{ENV_VAR}=1 ignored: MiniMax MSA is not importable ({err}).")
        return None
    if _V_SCALE_LAYOUT not in getattr(msa_jit, "V_SCALE_LAYOUTS", ()):
        logger.warning(
            f"{ENV_VAR}=1 ignored: this MiniMax MSA build reads V scales token-major only; "
            "it needs the quad4 V-scale layout option."
        )
        return None
    return BatchDecodeWithPagedKVCacheWrapper


def is_enabled() -> bool:
    """Whether the opt-in is set and a compatible MSA package is importable."""
    return _wrapper_class() is not None


@dataclass
class _Plan:
    """One planned MSA wrapper and the token-major top-k table it is bound to."""

    wrapper: object
    topk: torch.Tensor  # [batch * q_len, num_kv_heads, 16] int32


def _unit_scale(scale: torch.Tensor, cache: dict) -> Optional[bool]:
    """Whether a single-element dequantization scale is 1.0; None if unknown inside a capture.

    Read once per tensor on the host, outside capture, and remembered by address.
    """
    key = ("unit_scale", scale.data_ptr())
    if key not in cache:
        if torch.cuda.is_current_stream_capturing():
            return None
        cache[key] = scale.numel() == 1 and float(scale.reshape(-1)[0].item()) == 1.0
    return cache[key]


def _get_or_plan(
    cache: dict,
    *,
    topk_shape: tuple,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    decode_query_len: int,
    num_q_heads: int,
    num_kv_heads: int,
    sm_scale: float,
) -> Optional[_Plan]:
    key = (
        "plan",
        int(block_table.shape[0]),
        int(block_table.shape[1]),
        block_table.data_ptr(),
        seq_lens.data_ptr(),
        int(decode_query_len),
        num_q_heads,
        num_kv_heads,
        float(sm_scale),
    )
    plan = cache.get(key)
    if plan is not None:
        return plan
    if torch.cuda.is_current_stream_capturing():
        _log_once(
            f"capture{key[1:3]}{key[5:]}",
            f"no plan for batch {key[1]} x {decode_query_len} queries before capture; this "
            "graph keeps the Triton/CuTe sparse decode.",
        )
        return None
    topk = torch.full(topk_shape, -1, dtype=torch.int32, device=block_table.device)
    wrapper = _wrapper_class()()
    wrapper.plan(
        topk,
        block_table,
        seq_lens,
        q_len_per_req=int(decode_query_len),
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        sm_scale=float(sm_scale),
    )
    plan = _Plan(wrapper=wrapper, topk=topk)
    cache[key] = plan
    _log_once(
        "planned",
        f"Q8KV4 decode on (GQA {num_q_heads // num_kv_heads}, first plan batch {key[1]} x "
        f"{decode_query_len} queries).",
    )
    return plan


@functools.lru_cache(maxsize=None)
def _device_supported(device: torch.device) -> bool:
    return torch.cuda.get_device_capability(device) in _SUPPORTED_CAPABILITIES


def try_decode(
    q: torch.Tensor,
    k_paged: torch.Tensor,
    v_paged: torch.Tensor,
    topk_idx: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    *,
    sm_scale: float,
    output: torch.Tensor,
    decode_query_len: int,
    k_block_scale: Optional[torch.Tensor],
    v_block_scale: Optional[torch.Tensor],
    k_global_scale: Optional[torch.Tensor],
    v_global_scale: Optional[torch.Tensor],
    plan_cache: dict,
) -> bool:
    """Run the MSA-v1 Q8KV4 decode when the call is inside its contract; True when it ran.

    Arguments follow minimax_m3_sparse_attn_decode:
      q: [total_q, num_heads, 128] float8_e4m3fn, contiguous.
      k_paged / v_paged: [num_pages, num_kv_heads, 128, 64] packed E2M1 (1-byte dtype).
      topk_idx: [num_kv_heads, total_q, 16] int32 logical pages (any strides), -1 padded,
        the request-local page last among the valid entries.
      block_table: [batch, max_pages] int32; seq_lens: [batch] int32 including this step's queries.
      output: [total_q, num_heads, 128] bfloat16, contiguous.
      k_block_scale / v_block_scale: [num_pages, num_kv_heads, 128, 8] E4M3 bytes, K token-major
        and V in 4x4 token-quad order.
      k_global_scale / v_global_scale: single-element FP32 dequantization scales.
      plan_cache: the attention metadata's dict for this module's plans.
    """
    if _wrapper_class() is None:
        return False
    if any(t is None for t in (k_block_scale, v_block_scale, k_global_scale, v_global_scale)):
        return False
    if not _device_supported(q.device):
        return False
    total_q, num_q_heads, head_dim = q.shape
    num_kv_heads = int(k_paged.shape[1])
    if (
        q.dtype != torch.float8_e4m3fn
        or head_dim != _HEAD_DIM
        or num_q_heads % num_kv_heads
        or num_q_heads // num_kv_heads not in _GQA_RATIOS
        or k_paged.dim() != 4
        or int(k_paged.shape[2]) != _PAGE
        or int(k_paged.shape[3]) != _HEAD_DIM // 2
        or k_paged.element_size() != 1
        or k_block_scale.element_size() != 1
        or tuple(topk_idx.shape) != (num_kv_heads, total_q, _TOPK)
        or topk_idx.dtype != torch.int32
        or output.dtype != torch.bfloat16
        or tuple(output.shape) != (total_q, num_q_heads, _HEAD_DIM)
        or not (q.is_contiguous() and output.is_contiguous())
        or not (block_table.is_contiguous() and seq_lens.is_contiguous())
        or block_table.dtype != torch.int32
        or seq_lens.dtype != torch.int32
        or total_q != int(block_table.shape[0]) * int(decode_query_len)
    ):
        return False
    unit_k = _unit_scale(k_global_scale, plan_cache)
    unit_v = _unit_scale(v_global_scale, plan_cache)
    if not (unit_k and unit_v):
        if unit_k is not None and unit_v is not None:
            _log_once("scales", "non-unit K/V dequantization scales; keeping the Triton kernel.")
        return False
    plan = _get_or_plan(
        plan_cache,
        topk_shape=(total_q, num_kv_heads, _TOPK),
        block_table=block_table,
        seq_lens=seq_lens,
        decode_query_len=decode_query_len,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        sm_scale=sm_scale,
    )
    if plan is None:
        return False
    plan.topk.copy_(topk_idx.permute(1, 0, 2))
    plan.wrapper.run(
        q,
        (k_paged.view(torch.uint8), v_paged.view(torch.uint8)),
        kv_cache_sf=(
            k_block_scale.view(torch.float8_e4m3fn),
            v_block_scale.view(torch.float8_e4m3fn),
        ),
        out=output,
    )
    return True


__all__ = ["ENV_VAR", "PLAN_CACHE_KEY", "is_enabled", "try_decode"]
