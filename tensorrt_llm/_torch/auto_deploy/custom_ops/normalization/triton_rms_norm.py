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

import torch
import triton
import triton.language as tl
from torch import Tensor


@triton.jit
def rms_norm_kernel(
    input,
    weight,
    output,
    input_row_stride: tl.constexpr,
    eps: tl.constexpr,
    N_COLS: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Rms norm kernel.
    Forces weights to be in float32 for the kernel.
    """
    prog_id = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_N)

    w = tl.load(weight + offsets, mask=offsets < N_COLS)

    x_ptr = input + prog_id * input_row_stride
    x = tl.load(x_ptr + offsets, mask=offsets < N_COLS)
    xf = x.to(tl.float32)

    var = tl.sum(xf * xf, 0) * float(1.0 / N_COLS)
    out = xf / tl.sqrt(var + eps)
    out = (w.to(tl.float32) * out).to(x.dtype)

    out_ptr = output + prog_id * input_row_stride
    tl.store(out_ptr + offsets, out, mask=offsets < N_COLS)


def rms_norm(hidden_states: Tensor, weight: Tensor, eps: float = 1e-5):
    """Rms norm."""
    # Ensure contiguous: the Triton kernel uses the same stride for both input
    # and output pointers, but torch.empty_like always produces a contiguous
    # output. If hidden_states is non-contiguous (e.g. a split_with_sizes view),
    # input_stride != output_stride → out-of-bounds writes → cudaErrorIllegalAddress.
    if not hidden_states.is_contiguous():
        hidden_states = hidden_states.contiguous()
    feat_size = weight.shape[0]
    seq_len = hidden_states.numel() // hidden_states.size(-1)
    input_stride = hidden_states.stride(-2)

    BLOCK_N = triton.next_power_of_2(feat_size)
    out = torch.empty_like(hidden_states)

    grid = (seq_len,)
    rms_norm_kernel[grid](
        hidden_states,
        weight,
        out,
        input_row_stride=input_stride,
        eps=eps,
        N_COLS=feat_size,
        BLOCK_N=BLOCK_N,
        num_warps=4,
        num_stages=3,
    )

    return out


# Fused residual-add + RMSNorm in one kernel. Rounds the sum to the input dtype
# before the variance, so the result is bit-identical to a separate add + norm.


@triton.jit
def add_rms_norm_kernel(
    input,
    residual,
    weight,
    output,
    residual_output,
    row_stride: tl.constexpr,
    eps: tl.constexpr,
    N_COLS: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    prog_id = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_N)
    m = offsets < N_COLS
    w = tl.load(weight + offsets, mask=m)
    base = prog_id * row_stride
    x = tl.load(input + base + offsets, mask=m)
    r = tl.load(residual + base + offsets, mask=m)

    # bf16-faithful add: round sum to input dtype (matches aten.add output),
    # then cast to fp32 for the norm (matches triton_rms_norm reading a bf16 sum).
    added = (x + r).to(x.dtype)
    tl.store(residual_output + base + offsets, added, mask=m)

    af = added.to(tl.float32)
    var = tl.sum(af * af, 0) * float(1.0 / N_COLS)
    out = af / tl.sqrt(var + eps)
    out = (w.to(tl.float32) * out).to(x.dtype)
    tl.store(output + base + offsets, out, mask=m)


def add_rms_norm(hidden_states: Tensor, residual: Tensor, weight: Tensor, eps: float = 1e-5):
    """Fused (x + residual) then RMSNorm.  Returns (normed, x+residual)."""
    # Kernel shares one row_stride across in/residual/out; empty_like outputs are
    # contiguous, so non-contiguous inputs must be made contiguous (cf. rms_norm).
    if not hidden_states.is_contiguous():
        hidden_states = hidden_states.contiguous()
    if not residual.is_contiguous():
        residual = residual.contiguous()
    if hidden_states.shape != residual.shape:
        raise ValueError("hidden_states and residual must have the same shape")
    feat_size = weight.shape[0]
    seq_len = hidden_states.numel() // hidden_states.size(-1)
    row_stride = hidden_states.stride(-2)
    BLOCK_N = triton.next_power_of_2(feat_size)
    out = torch.empty_like(hidden_states)
    res_out = torch.empty_like(hidden_states)
    grid = (seq_len,)
    add_rms_norm_kernel[grid](
        hidden_states,
        residual,
        weight,
        out,
        res_out,
        row_stride=row_stride,
        eps=eps,
        N_COLS=feat_size,
        BLOCK_N=BLOCK_N,
        num_warps=4,
        num_stages=3,
    )
    return out, res_out


@torch.library.custom_op("auto_deploy::triton_fused_add_rms_norm", mutates_args=())
def triton_fused_add_rms_norm(
    input: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused residual-add + Triton RMSNorm. Returns (normed, input+residual)."""
    return add_rms_norm(input, residual, weight, eps)


@triton_fused_add_rms_norm.register_fake
def _triton_fused_add_rms_norm_fake(
    input: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float
) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.empty_like(input), torch.empty_like(input)
