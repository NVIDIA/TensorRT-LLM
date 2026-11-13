# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The DFlash / DSpark non-causal block merge as one CUDA launch (``minimax_m31_dflash_merge_later_keys``).

A non-causal sliding-window drafter layer runs the trtllm-gen causal decode with its log-sum-exp and then adds, for
each block query ``j``, the block keys ``j < k <= j + window_right`` (``modeling_dflash``'s eager merge). The family's
CUDA op does that in one launch, within one bf16 ULP of the eager form on ``|out| >= 1/16`` (not bitwise).

``install`` registers it as the merge of the built-in DFlash windowed block attention, where the runtime takes one
(``modeling_dflash.set_dflash_merge_future_override``); operands outside the op's contract keep the eager merge.
``TRTLLM_M31_DISABLE_NATIVE_DFLASH_MERGE=1`` installs nothing. Read at import.
"""

from __future__ import annotations

import os
from typing import Optional

import torch

from tensorrt_llm._torch._experimental.modeling_v2._router_index import NoTarget
from tensorrt_llm.logger import logger

from ...native import family_ops

NATIVE_DFLASH_MERGE = os.environ.get(
    "TRTLLM_M31_DISABLE_NATIVE_DFLASH_MERGE", ""
).strip().lower() not in (
    "1",
    "true",
    "on",
    "yes",
)


def outside_contract(
    out: torch.Tensor,
    lse_log2: torch.Tensor,
    query: torch.Tensor,
    block_key: torch.Tensor,
    block_value: torch.Tensor,
) -> Optional[str]:
    """Why the CUDA merge does not take these operands, or None: bf16 CUDA ``query`` [B, blk <= 16, Hq, 128],
    ``block_key`` / ``block_value`` [B, blk, Hkv, 128] with Hkv dividing Hq, ``out`` [B * blk, Hq, 128] and a float32
    ``lse_log2`` [B * blk, Hq] on one device; unit inner strides, 8-byte aligned bf16 bases and outer strides a
    multiple of 4 elements."""
    if not query.is_cuda or query.dim() != 4 or block_key.dim() != 4 or block_value.dim() != 4:
        return f"query {tuple(query.shape)} on {query.device}"
    B, blk, num_heads, head_dim = query.shape
    num_kv_heads = block_key.shape[2]
    bf16 = (out, query, block_key, block_value)
    if any(t.dtype != torch.bfloat16 for t in bf16) or lse_log2.dtype != torch.float32:
        return f"dtypes {[t.dtype for t in bf16]} / lse {lse_log2.dtype}"
    if (
        head_dim != 128
        or blk > 16
        or num_kv_heads < 1
        or num_heads % num_kv_heads
        or tuple(block_key.shape) != (B, blk, num_kv_heads, head_dim)
        or tuple(block_value.shape) != tuple(block_key.shape)
        or tuple(out.shape) != (B * blk, num_heads, head_dim)
        or tuple(lse_log2.shape) != (B * blk, num_heads)
    ):
        return (
            f"shapes q {tuple(query.shape)} k {tuple(block_key.shape)} out {tuple(out.shape)} "
            f"lse {tuple(lse_log2.shape)}"
        )
    if any(t.device != query.device for t in (out, lse_log2, block_key, block_value)):
        return "operands on different devices"
    for t in bf16:
        if t.stride(-1) != 1 or t.data_ptr() % 8 or any(s % 4 for s in t.stride()[:-1]):
            return f"strides {t.stride()} / alignment of a {tuple(t.shape)} operand"
    return None


def merge_later_keys_(
    out: torch.Tensor,
    lse_log2: torch.Tensor,
    query: torch.Tensor,
    block_key: torch.Tensor,
    block_value: torch.Tensor,
    scale: float,
    window_right: int,
) -> bool:
    """Add each block query's later keys ``j < k <= j + window_right`` to its causal attention ``out``
    [B * blk, Hq, D] in place, given the base-2 log-sum-exp ``lse_log2`` [B * blk, Hq]. Returns whether the CUDA op
    took the operands; when it returns False nothing was written."""
    reason = outside_contract(out, lse_log2, query, block_key, block_value)
    if reason is not None:
        logger.info_once(
            f"DFlash CUDA block merge not taken ({reason}); the eager merge runs",
            key=f"dflash_merge_not_taken_{reason}",
        )
        return False
    family_ops().minimax_m31_dflash_merge_later_keys(
        out, lse_log2, query, block_key, block_value, scale, window_right
    )
    return True


def install() -> bool:
    """Register the CUDA merge with the built-in DFlash windowed block attention; returns whether it is registered.
    Process-wide; installs nothing with ``TRTLLM_M31_DISABLE_NATIVE_DFLASH_MERGE=1``, on a runtime without the
    attention's merge hook, or without the family op."""
    if not NATIVE_DFLASH_MERGE:
        return False
    from tensorrt_llm._torch.models import modeling_dflash

    register = getattr(modeling_dflash, "set_dflash_merge_future_override", None)
    if register is None:
        return False
    try:
        if not hasattr(family_ops(), "minimax_m31_dflash_merge_later_keys"):
            return False
    except NoTarget:
        return False
    register(merge_later_keys_)
    return True
