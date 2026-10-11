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
"""Quantization of the K/V block means consumed by Sage block-sparse attention."""

import torch
import triton
import triton.language as tl
from torch.library import triton_op, wrap_triton
from triton.language.extra.cuda import libdevice

__all__ = ["quantize_sage_summaries"]


@triton.jit
def _quantize_sage_summaries_kernel(
    k_ptr,
    v_ptr,
    v_scale_ptr,
    quant_k_ptr,
    quant_v_ptr,
    k_scale_ptr,
    num_rows,
    blocks,
    heads,
    scale_stride,
    DIM: tl.constexpr,
    INT8: tl.constexpr,
    ROWS: tl.constexpr,
    WIDTH: tl.constexpr,
):
    rows = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    head = tl.program_id(1)
    channels = tl.arange(0, WIDTH)
    valid_rows = rows < num_rows
    valid = valid_rows[:, None] & (channels < DIM)[None, :]
    offsets = (rows[:, None] * heads + head) * DIM + channels[None, :]
    k = tl.load(k_ptr + offsets, valid, other=0).to(tl.float32)
    v = tl.load(v_ptr + offsets, valid, other=0).to(tl.float32)

    maximum = tl.maximum(tl.max(tl.abs(k), axis=1), 1e-3)
    divisor: tl.constexpr = 126.9 if INT8 else 448.0
    k_scale = tl.div_rn(maximum, divisor)
    normalized_k = tl.div_rn(k, k_scale[:, None])
    if INT8:
        quant_k = tl.minimum(tl.maximum(libdevice.nearbyint(normalized_k), -127.0), 127.0)
    else:
        quant_k = tl.minimum(tl.maximum(normalized_k, -448.0), 448.0)
    v_scale = tl.load(v_scale_ptr + head * DIM + channels, channels < DIM, other=1.0)
    quant_v = tl.minimum(tl.maximum(tl.div_rn(v, v_scale[None, :]), -448.0), 448.0)
    tl.store(quant_k_ptr + offsets, quant_k, valid)
    tl.store(quant_v_ptr + offsets, quant_v, valid)

    # Sequence b starts at scale slot b * (blocks + 1): the flat layout leaves one
    # unused slot between consecutive sequences.
    tl.store(k_scale_ptr + head * scale_stride + rows + rows // blocks, k_scale, valid_rows)


@triton_op("trtllm::quantize_sage_summaries", mutates_args={})
def quantize_sage_summaries(
    k_summary: torch.Tensor,
    v_summary: torch.Tensor,
    v_scale: torch.Tensor,
    qk_quant_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize key/value block means for the proxy routes of Sage attention.

    Every key mean gets its own scale; value means reuse the per-channel scales of
    the original V, so exact and proxy routes share one PV dequantization.

    Args:
        k_summary: Contiguous FP16/BF16 CUDA means, ``[batch, blocks, heads, head_dim]``.
        v_summary: Value means with the same shape, dtype, and device as ``k_summary``.
        v_scale: Contiguous positive FP32 ``[heads, head_dim]`` dequantization scales
            from quantizing the original V.
        qk_quant_dtype: ``torch.int8`` or ``torch.float8_e4m3fn`` for the key means.

    Returns:
        Quantized key means, FP8 E4M3 value means, and FP32 key scales of shape
        ``[heads, batch * blocks + batch - 1]``. Key scale slot ``b * (blocks + 1) + n``
        belongs to mean ``[b, n]``; the unused slot between sequences is not written.
    """
    if k_summary.ndim != 4 or min(k_summary.shape) <= 0:
        raise ValueError("k_summary must have nonempty [batch, blocks, heads, head_dim] shape")
    if k_summary.device.type != "cuda" or k_summary.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("k_summary must be a CUDA FP16 or BF16 tensor")
    if (
        v_summary.shape != k_summary.shape
        or v_summary.dtype != k_summary.dtype
        or v_summary.device != k_summary.device
    ):
        raise ValueError("key and value summaries must have matching shape, dtype, and device")
    if v_scale.shape != k_summary.shape[2:] or v_scale.dtype != torch.float32:
        raise ValueError("v_scale must be FP32 with shape [heads, head_dim]")
    if v_scale.device != k_summary.device:
        raise ValueError("v_scale and summaries must be on the same CUDA device")
    if not (k_summary.is_contiguous() and v_summary.is_contiguous() and v_scale.is_contiguous()):
        raise ValueError("summaries and v_scale must be contiguous")
    if qk_quant_dtype not in (torch.int8, torch.float8_e4m3fn):
        raise ValueError("qk_quant_dtype must be torch.int8 or torch.float8_e4m3fn")

    batch, blocks, heads, dim = k_summary.shape
    scale_stride = batch * blocks + batch - 1
    quant_k = torch.empty_like(k_summary, dtype=qk_quant_dtype)
    quant_v = torch.empty_like(v_summary, dtype=torch.float8_e4m3fn)
    k_scale = torch.empty((heads, scale_stride), device=k_summary.device, dtype=torch.float32)
    wrap_triton(_quantize_sage_summaries_kernel)[(triton.cdiv(batch * blocks, 4), heads)](
        k_summary,
        v_summary,
        v_scale,
        quant_k,
        quant_v,
        k_scale,
        batch * blocks,
        blocks,
        heads,
        scale_stride,
        DIM=dim,
        INT8=qk_quant_dtype == torch.int8,
        ROWS=4,
        WIDTH=triton.next_power_of_2(dim),
        num_warps=4,
    )
    return quant_k, quant_v, k_scale
