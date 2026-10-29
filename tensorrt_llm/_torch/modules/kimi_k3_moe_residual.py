# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Preserve the two BF16 rounding boundaries in a MoE residual update."""

from typing import Optional

import torch
import triton
import triton.language as tl

from tensorrt_llm.logger import logger

#: Largest token count of the fused add by default: every decode bucket (DEP16 d3 reaches 256
#: rows per rank). Context forwards pass ``max_rows=None`` (``TRTLLM_K3_FUSE_MOE_RESIDUAL_CTX``);
#: the element count is a runtime argument, so one compiled kernel serves every count.
MAX_FUSED_ROWS = 256
_BLOCK = 1024
# Offsets are int32 and the last program's masked lanes run up to BLOCK - 1 past the end, so the
# fused add takes at most this many elements.
_MAX_FUSED_NUMEL = 2**31 - _BLOCK


@triton.jit
def _moe_residual_add_kernel(routed, shared, residual, output, size, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < size
    routed_value = tl.load(routed + offsets, mask, other=0).to(tl.float32)
    shared_value = tl.load(shared + offsets, mask, other=0).to(tl.float32)
    residual_value = tl.load(residual + offsets, mask, other=0).to(tl.float32)
    moe_output = (routed_value + shared_value).to(tl.bfloat16).to(tl.float32)
    tl.store(output + offsets, residual_value + moe_output, mask)


def _fused_moe_residual_add(
    routed: torch.Tensor,
    shared: torch.Tensor,
    residual: torch.Tensor,
    max_rows: Optional[int] = MAX_FUSED_ROWS,
) -> torch.Tensor:
    """Compute residual + BF16(routed + shared), for matching [tokens, hidden] inputs.

    Inputs with more than ``max_rows`` rows (no limit if ``None``) take the two BF16 adds.
    """
    tensors = (routed, shared, residual)
    eligible = (
        routed.ndim == 2
        and 0 < routed.shape[0]
        and (max_rows is None or routed.shape[0] <= max_rows)
        and routed.numel() <= _MAX_FUSED_NUMEL
        and routed.shape[1] == 7168
        and all(
            x.shape == routed.shape
            and x.device == routed.device
            and x.is_cuda
            and x.dtype == torch.bfloat16
            and x.is_contiguous()
            for x in tensors
        )
        and not (torch.is_grad_enabled() and any(x.requires_grad for x in tensors))
    )
    # Uncapped calls with more rows than any decode bucket come from context forwards
    # (TRTLLM_K3_FUSE_MOE_RESIDUAL_CTX): log once which way the first one went.
    context = max_rows is None and routed.ndim == 2 and routed.shape[0] > MAX_FUSED_ROWS
    if not eligible:
        if context:
            logger.warning_once(
                "Kimi K3 fused MoE residual add: context inputs are not eligible (shapes "
                f"{[tuple(x.shape) for x in tensors]}, dtypes {[x.dtype for x in tensors]}, "
                f"devices {[str(x.device) for x in tensors]}, contiguous "
                f"{[x.is_contiguous() for x in tensors]}, requires_grad "
                f"{[x.requires_grad for x in tensors]}, grad mode {torch.is_grad_enabled()}); "
                "using two BF16 adds",
                key="kimi_k3_moe_residual_ctx_fallback",
            )
        return residual + (routed + shared)
    if context:
        logger.info_once(
            "Kimi K3 fused MoE residual add at context row counts (TRTLLM_K3_FUSE_MOE_RESIDUAL_CTX)",
            key="kimi_k3_moe_residual_ctx",
        )
    output = torch.empty_like(routed)
    _moe_residual_add_kernel[(triton.cdiv(routed.numel(), _BLOCK),)](
        routed, shared, residual, output, routed.numel(), _BLOCK, num_warps=4
    )
    return output
