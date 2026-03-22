# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Triton kernel template for fused add + RMSNorm.

This kernel performs:
    add_result = x + residual
    norm_result = rmsnorm(add_result, weight, eps)

Both results are written out, matching the semantics of ``ad.fused_add_rmsnorm``.

Reference: ``custom_ops/normalization/triton_rms_norm.py`` for the unfused Triton kernel.
"""

import torch
import triton
import triton.language as tl
from torch import Tensor


@triton.jit
def fused_add_rmsnorm_kernel(
    x_ptr,
    residual_ptr,
    weight_ptr,
    norm_out_ptr,
    add_out_ptr,
    row_stride: tl.constexpr,
    eps: tl.constexpr,
    N_COLS: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Fused add + RMSNorm Triton kernel.

    Each program instance processes one row.
    """
    row_idx = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_N)
    mask = offsets < N_COLS

    # Load inputs
    row_off = row_idx * row_stride
    x = tl.load(x_ptr + row_off + offsets, mask=mask)
    res = tl.load(residual_ptr + row_off + offsets, mask=mask)
    w = tl.load(weight_ptr + offsets, mask=mask)

    # Fused add
    added = x + res

    # Store add result
    tl.store(add_out_ptr + row_off + offsets, added, mask=mask)

    # RMSNorm in float32
    added_f32 = added.to(tl.float32)
    var = tl.sum(added_f32 * added_f32, 0) * float(1.0 / N_COLS)
    normed = added_f32 / tl.sqrt(var + eps)
    normed = (w.to(tl.float32) * normed).to(added.dtype)

    # Store norm result
    tl.store(norm_out_ptr + row_off + offsets, normed, mask=mask)


def fused_add_rmsnorm(x: Tensor, residual: Tensor, weight: Tensor, eps: float = 1e-5) -> tuple:
    """Launch the fused add + RMSNorm Triton kernel.

    Args:
        x: Input tensor to add (..., hidden_size).
        residual: Residual tensor (..., hidden_size).
        weight: RMSNorm weight (hidden_size,).
        eps: Epsilon for numerical stability.

    Returns:
        Tuple of (norm_result, add_result) with the same shape as inputs.
    """
    feat_size = weight.shape[0]
    seq_len = x.numel() // x.size(-1)
    row_stride = x.stride(-2)

    BLOCK_N = triton.next_power_of_2(feat_size)

    norm_out = torch.empty_like(x)
    add_out = torch.empty_like(x)

    grid = (seq_len,)
    fused_add_rmsnorm_kernel[grid](
        x,
        residual,
        weight,
        norm_out,
        add_out,
        row_stride=row_stride,
        eps=eps,
        N_COLS=feat_size,
        BLOCK_N=BLOCK_N,
        num_warps=4,
        num_stages=3,
    )

    return norm_out, add_out
