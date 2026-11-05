# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Low-M fp32 router projection: ``logits[M, E] = float(x[M, K]) @ W[E, K]^T`` in one kernel.

Replaces the bf16->fp32 cast, the TF32 GEMM and its split-K reduce on decode
steps. One CTA per ``ROWS`` experts: the fp32 weight rows are bulk-copied into
shared memory before ``griddepcontrol.wait`` (overlapping the producer kernel
under PDL), the bf16 activations after it. Products and sums are exact fp32 FMA.

The kernels are ``torch.ops.trtllm.minimax_m3_router_gemv`` / ``_router_gemv_fp4`` / ``_router_tc``, reached
through the ``ops/moe`` entries.
"""

from typing import Optional

import torch

from ..ops.moe.m3_router_gemv import m3_router_gemv as _router_gemv_op
from ..ops.moe.m3_router_gemv_fp4 import m3_router_gemv_fp4 as _router_gemv_fp4_op
from ..ops.moe.m3_router_tc import m3_router_tc as _router_tc_op

# Opt-in dynamic shared memory left after the kernel's static reduction buffer (SM100: 227 KiB).
_MAX_DYNAMIC_SMEM = 227 * 1024 - 1024  # capacity: fixed: the device's shared-memory limit


def m3_router_gemv(
    x: torch.Tensor, weight: torch.Tensor, rows: int = 1, use_pdl: bool = False
) -> Optional[torch.Tensor]:
    """fp32 ``x @ weight.T`` for bf16 ``x`` with at most 16 rows; None when unsupported (only ``rows=1`` is built)."""
    if (
        rows != 1
        or not x.is_cuda
        or x.dtype != torch.bfloat16
        or x.dim() != 2
        or x.stride(1) != 1
        or not 1 <= x.shape[0] <= 16
        or weight.dtype != torch.float32
        or not weight.is_contiguous()
        or weight.shape[1] != x.shape[1]
        or x.shape[1] % 1024 != 0
        or weight.shape[0] % rows != 0
        or x.data_ptr() % 16 != 0
        or (x.stride(0) * 2) % 16 != 0
        or torch.cuda.get_device_capability(x.device)[0] < 10
        or rows * x.shape[1] * 4 + x.shape[0] * x.shape[1] * 2 > _MAX_DYNAMIC_SMEM
    ):
        return None
    return _router_gemv_op(x, weight, rows, use_pdl)


def m3_router_gemv_fp4(
    x: torch.Tensor, weight: torch.Tensor, sf_scale: torch.Tensor, use_pdl: bool = False
) -> Optional[tuple]:
    """Router logits plus the NVFP4 activation the routed experts consume (bit-identical to
    ``fp4_quantize(x, sf_scale, 16, False, False)``): (logits [M, E] fp32, x_fp4 [M, K / 2] uint8,
    x_sf [M * K / 16] uint8 linear). None when unsupported."""
    if (
        not x.is_cuda
        or x.dtype != torch.bfloat16
        or x.dim() != 2
        or x.stride(1) != 1
        or not 1 <= x.shape[0] <= 16
        or weight.dtype != torch.float32
        or not weight.is_contiguous()
        or weight.shape[1] != x.shape[1]
        or x.shape[1] % 1024 != 0
        or x.data_ptr() % 16 != 0
        or (x.stride(0) * 2) % 16 != 0
        or sf_scale is None
        or sf_scale.dtype != torch.float32
        or sf_scale.numel() != 1
        or torch.cuda.get_device_capability(x.device)[0] < 10
        or x.shape[1] * 4 + x.shape[0] * x.shape[1] * 2 > _MAX_DYNAMIC_SMEM
    ):
        return None
    return tuple(_router_gemv_fp4_op(x, weight, sf_scale.reshape(1), use_pdl))


def m3_router_tc(
    x: torch.Tensor,
    weight: torch.Tensor,
    sf_scale: Optional[torch.Tensor] = None,
    use_pdl: bool = False,
) -> Optional[tuple]:
    """Router logits (TF32 tensor cores, like the native TF32 GEMM) for 1..48 tokens with 128 experts, plus the
    NVFP4 activation when ``sf_scale`` is given. Returns (logits,) or (logits, x_fp4, x_sf); None when unsupported."""
    if (
        not x.is_cuda
        or x.dtype != torch.bfloat16
        or x.dim() != 2
        or x.stride(1) != 1
        or not 1 <= x.shape[0] <= 48
        or weight.dtype != torch.float32
        or weight.shape[0] != 128
        or not weight.is_contiguous()
        or weight.shape[1] != x.shape[1]
        or x.shape[1] % 1024 != 0
        or (x.stride(0) * 2) % 16 != 0
        or x.data_ptr() % 16 != 0
        or torch.cuda.get_device_capability(x.device)[0] < 10
        or 16 * (x.shape[1] // 8 + 4) * 4 + x.shape[0] * (x.shape[1] // 8 + 8) * 2
        > _MAX_DYNAMIC_SMEM
        or (sf_scale is not None and (sf_scale.dtype != torch.float32 or sf_scale.numel() != 1))
    ):
        return None
    res = _router_tc_op(x, weight, None if sf_scale is None else sf_scale.reshape(1), use_pdl)
    return tuple(res)
