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

"""Triton kernels for fused sharded QK RMSNorm with packed variance allreduce."""

from typing import Tuple

import torch
import torch.distributed as dist
import triton
import triton.language as tl
from torch import Tensor


@triton.jit
def triton_row_sumsq_kernel(
    input_ptr,
    output_ptr,
    input_row_stride: tl.constexpr,
    N_COLS: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Compute sum(x^2) over each row, returning one float32 scalar per row.

    One program per row — prog_id selects the row.
    """
    prog_id = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_N)

    x_ptr = input_ptr + prog_id * input_row_stride
    x = tl.load(x_ptr + offsets, mask=offsets < N_COLS)
    xf = x.to(tl.float32)

    row_sumsq = tl.sum(xf * xf, 0)
    tl.store(output_ptr + prog_id, row_sumsq)


@triton.jit
def triton_normalize_scale_kernel(
    input_ptr,
    global_sumsq_ptr,
    weight_ptr,
    output_ptr,
    input_row_stride: tl.constexpr,
    global_count: tl.constexpr,
    eps: tl.constexpr,
    N_COLS: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Normalize a row using a precomputed global sum-of-squares scalar.

    One program per row. Reads the global sum-of-squares for this row from
    global_sumsq_ptr[prog_id], computes rstd, then normalizes and scales.
    """
    prog_id = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_N)

    # Load precomputed global sum-of-squares for this row
    global_sumsq = tl.load(global_sumsq_ptr + prog_id).to(tl.float32)

    # Compute reciprocal standard deviation from global stats
    rstd = 1.0 / tl.sqrt(global_sumsq / global_count + eps)

    # Load input row
    x_ptr = input_ptr + prog_id * input_row_stride
    x = tl.load(x_ptr + offsets, mask=offsets < N_COLS)
    xf = x.to(tl.float32)

    # Load weight
    w = tl.load(weight_ptr + offsets, mask=offsets < N_COLS)
    wf = w.to(tl.float32)

    # Normalize and scale
    out = (xf * rstd * wf).to(x.dtype)

    out_ptr = output_ptr + prog_id * input_row_stride
    tl.store(out_ptr + offsets, out, mask=offsets < N_COLS)


def triton_fused_sharded_qk_norm(
    q: Tensor,
    k: Tensor,
    weight_q: Tensor,
    weight_k: Tensor,
    eps: float,
    world_size: int,
) -> Tuple[Tensor, Tensor]:
    """Fused sharded QK RMSNorm with a single packed allreduce.

    Computes RMSNorm for both Q and K (sharded along head dim) using ONE
    allreduce on packed [N_tokens, 2] variance stats instead of two separate
    allreduces.

    Algorithm:
        1. Compute local sum-of-squares for each row of Q and K separately.
        2. Pack into [seq_len, 2] float32 and do ONE dist.all_reduce(SUM).
        3. Normalize Q using global stats col 0, normalize K using global stats col 1.

    Args:
        q: Local Q shard, shape [..., local_q_dim].
        k: Local K shard, shape [..., local_k_dim].
        weight_q: Local Q norm weight, shape [local_q_dim].
        weight_k: Local K norm weight, shape [local_k_dim].
        eps: Small constant for numerical stability.
        world_size: Number of devices across which Q and K are sharded.

    Returns:
        Tuple of (q_norm, k_norm), each with the same shape as the respective input.
    """
    if not q.is_contiguous():
        q = q.contiguous()
    if not k.is_contiguous():
        k = k.contiguous()

    seq_len = q.numel() // q.shape[-1]
    local_q_dim = q.shape[-1]
    local_k_dim = k.shape[-1]

    BLOCK_Q = triton.next_power_of_2(local_q_dim)
    BLOCK_K = triton.next_power_of_2(local_k_dim)

    q_sumsq = torch.empty(seq_len, device=q.device, dtype=torch.float32)
    k_sumsq = torch.empty(seq_len, device=q.device, dtype=torch.float32)

    # Kernel 1a: compute local sum-of-squares for each row of Q
    triton_row_sumsq_kernel[(seq_len,)](
        q,
        q_sumsq,
        input_row_stride=q.stride(-2),
        N_COLS=local_q_dim,
        BLOCK_N=BLOCK_Q,
        num_warps=4,
    )

    # Kernel 1b: compute local sum-of-squares for each row of K
    triton_row_sumsq_kernel[(seq_len,)](
        k,
        k_sumsq,
        input_row_stride=k.stride(-2),
        N_COLS=local_k_dim,
        BLOCK_N=BLOCK_K,
        num_warps=4,
    )

    # Step 2: Pack into [seq_len, 2] and do ONE allreduce
    packed = torch.stack([q_sumsq, k_sumsq], dim=-1)  # [seq_len, 2] float32
    dist.all_reduce(packed, op=dist.ReduceOp.SUM)

    # Step 3: Normalize Q and K using their respective global stats
    q_out = torch.empty_like(q)
    k_out = torch.empty_like(k)

    triton_normalize_scale_kernel[(seq_len,)](
        q,
        packed[:, 0].contiguous(),
        weight_q,
        q_out,
        input_row_stride=q.stride(-2),
        global_count=local_q_dim * world_size,
        eps=eps,
        N_COLS=local_q_dim,
        BLOCK_N=BLOCK_Q,
        num_warps=4,
    )

    triton_normalize_scale_kernel[(seq_len,)](
        k,
        packed[:, 1].contiguous(),
        weight_k,
        k_out,
        input_row_stride=k.stride(-2),
        global_count=local_k_dim * world_size,
        eps=eps,
        N_COLS=local_k_dim,
        BLOCK_N=BLOCK_K,
        num_warps=4,
    )

    return q_out, k_out
