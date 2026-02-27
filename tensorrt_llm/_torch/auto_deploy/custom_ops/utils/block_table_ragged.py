# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Bidirectional conversion between block table and ragged tensor formats.

Block table format: [max_batch_size, max_blocks_per_seq] padded 2D tensor
Ragged format: flat 1D cache_loc + cumulative offset vector cu_num_blocks

Provides both PyTorch reference (tensorized, loop-free) and Triton kernel
implementations (one program per sequence row), registered as custom ops.
"""

import torch
import triton
import triton.language as tl

# ========================================================================================
# Triton kernels (internal)
# ========================================================================================


@triton.jit
def _block_table_to_ragged_kernel(
    block_table_ptr,
    num_blocks_ptr,
    cu_num_blocks_ptr,
    cache_loc_ptr,
    M: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Pack one row of block_table into the correct offset in cache_loc."""
    seq_id = tl.program_id(0)

    n_valid = tl.load(num_blocks_ptr + seq_id)
    dst_start = tl.load(cu_num_blocks_ptr + seq_id)

    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < n_valid

    vals = tl.load(block_table_ptr + seq_id * M + offs, mask=mask, other=0)
    tl.store(cache_loc_ptr + dst_start + offs, vals, mask=mask)


@triton.jit
def _ragged_to_block_table_kernel(
    cache_loc_ptr,
    cu_num_blocks_ptr,
    block_table_ptr,
    M: tl.constexpr,
    STRIDE_ROW,
    BLOCK_SIZE: tl.constexpr,
):
    """Unpack one segment of cache_loc into a zero-padded row of block_table."""
    seq_id = tl.program_id(0)

    src_start = tl.load(cu_num_blocks_ptr + seq_id)
    src_end = tl.load(cu_num_blocks_ptr + seq_id + 1)
    n_valid = src_end - src_start

    offs = tl.arange(0, BLOCK_SIZE)

    vals = tl.load(cache_loc_ptr + src_start + offs, mask=offs < n_valid, other=0)
    tl.store(block_table_ptr + seq_id * STRIDE_ROW + offs, vals, mask=offs < M)


@triton.jit
def _adjust_block_table_kernel(
    block_table_ptr,
    num_blocks_ptr,
    extra_idx_ptr,
    delta_ptr,
    M: tl.constexpr,
):
    """Adjust one block_table row: append (delta=+1), remove last (delta=-1), or no-op (delta=0)."""
    seq_id = tl.program_id(0)
    d = tl.load(delta_ptr + seq_id)

    if d != 0:
        n = tl.load(num_blocks_ptr + seq_id)
        if d > 0:
            extra = tl.load(extra_idx_ptr + seq_id)
            tl.store(block_table_ptr + seq_id * M + n, extra)
        tl.store(num_blocks_ptr + seq_id, n + d)


@triton.jit
def _adjust_ragged_kernel(
    cache_loc_ptr,
    temp_ptr,
    old_cu_ptr,
    new_cu_ptr,
    extra_idx_ptr,
    delta_ptr,
    BLOCK_SIZE: tl.constexpr,
):
    """Copy one sequence's segment, adjusting length by delta: +1 appends, -1 drops last."""
    seq_id = tl.program_id(0)

    old_start = tl.load(old_cu_ptr + seq_id)
    old_end = tl.load(old_cu_ptr + seq_id + 1)
    n_old = old_end - old_start

    new_start = tl.load(new_cu_ptr + seq_id)

    d = tl.load(delta_ptr + seq_id)
    n_copy = n_old + tl.minimum(d, 0)

    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < n_copy
    vals = tl.load(cache_loc_ptr + old_start + offs, mask=mask, other=0)
    tl.store(temp_ptr + new_start + offs, vals, mask=mask)

    if d > 0:
        extra = tl.load(extra_idx_ptr + seq_id)
        tl.store(temp_ptr + new_start + n_old, extra)


# ========================================================================================
# Custom ops: block table <-> ragged conversion
# ========================================================================================


@torch.library.custom_op(
    "auto_deploy::block_table_to_ragged_torch", mutates_args=("cache_loc", "cu_num_blocks")
)
def block_table_to_ragged_torch(
    block_table: torch.Tensor,
    num_blocks: torch.Tensor,
    cache_loc: torch.Tensor,
    cu_num_blocks: torch.Tensor,
    num_sequences: int,
) -> int:
    """Convert block table (padded 2D) to ragged (packed 1D) format using PyTorch."""
    N = num_sequences
    M = block_table.shape[1]

    cu_num_blocks[0] = 0
    cu_num_blocks[1 : N + 1] = torch.cumsum(num_blocks[:N], dim=0)

    col = torch.arange(M, device=block_table.device)
    mask = col.unsqueeze(0) < num_blocks[:N].unsqueeze(1)

    total = cu_num_blocks[N].item()
    cache_loc[:total] = block_table[:N][mask]

    return total


@block_table_to_ragged_torch.register_fake
def _block_table_to_ragged_torch_fake(
    block_table: torch.Tensor,
    num_blocks: torch.Tensor,
    cache_loc: torch.Tensor,
    cu_num_blocks: torch.Tensor,
    num_sequences: int,
) -> int:
    return 0


@torch.library.custom_op(
    "auto_deploy::ragged_to_block_table_torch",
    mutates_args=("block_table", "num_blocks"),
)
def ragged_to_block_table_torch(
    cache_loc: torch.Tensor,
    cu_num_blocks: torch.Tensor,
    block_table: torch.Tensor,
    num_blocks: torch.Tensor,
    num_sequences: int,
) -> None:
    """Convert ragged (packed 1D) to block table (padded 2D) format using PyTorch."""
    N = num_sequences
    M = block_table.shape[1]

    num_blocks[:N] = (cu_num_blocks[1 : N + 1] - cu_num_blocks[:N]).to(num_blocks.dtype)

    col = torch.arange(M, device=cache_loc.device)
    mask = col.unsqueeze(0) < num_blocks[:N].unsqueeze(1)

    src = cu_num_blocks[:N].unsqueeze(1) + col.unsqueeze(0)

    total = int(cu_num_blocks[N].item())
    src_safe = src.clamp(max=max(total - 1, 0))

    zero = torch.zeros(1, device=cache_loc.device, dtype=cache_loc.dtype)
    block_table[:N] = torch.where(mask, cache_loc[src_safe], zero)


@ragged_to_block_table_torch.register_fake
def _ragged_to_block_table_torch_fake(
    cache_loc: torch.Tensor,
    cu_num_blocks: torch.Tensor,
    block_table: torch.Tensor,
    num_blocks: torch.Tensor,
    num_sequences: int,
) -> None:
    pass


@torch.library.custom_op(
    "auto_deploy::block_table_to_ragged_triton", mutates_args=("cache_loc", "cu_num_blocks")
)
def block_table_to_ragged_triton(
    block_table: torch.Tensor,
    num_blocks: torch.Tensor,
    cache_loc: torch.Tensor,
    cu_num_blocks: torch.Tensor,
    num_sequences: int,
) -> int:
    """Convert block table to ragged format using Triton."""
    N = num_sequences
    M = block_table.shape[1]

    cu_num_blocks[0] = 0
    cu_num_blocks[1 : N + 1] = torch.cumsum(num_blocks[:N], dim=0)

    if N == 0:
        return 0

    BLOCK_SIZE = triton.next_power_of_2(M)
    _block_table_to_ragged_kernel[(N,)](
        block_table, num_blocks, cu_num_blocks, cache_loc, M=M, BLOCK_SIZE=BLOCK_SIZE
    )

    total = cu_num_blocks[N].item()
    return total


@block_table_to_ragged_triton.register_fake
def _block_table_to_ragged_triton_fake(
    block_table: torch.Tensor,
    num_blocks: torch.Tensor,
    cache_loc: torch.Tensor,
    cu_num_blocks: torch.Tensor,
    num_sequences: int,
) -> int:
    return 0


@torch.library.custom_op("auto_deploy::ragged_to_block_table_triton", mutates_args=("block_table",))
def ragged_to_block_table_triton(
    cache_loc: torch.Tensor,
    cu_num_blocks: torch.Tensor,
    block_table: torch.Tensor,
    num_sequences: int,
) -> None:
    """Convert ragged format to block table using Triton."""
    N = num_sequences
    M = block_table.shape[1]

    if N == 0:
        return

    BLOCK_SIZE = triton.next_power_of_2(M)
    _ragged_to_block_table_kernel[(N,)](
        cache_loc,
        cu_num_blocks,
        block_table,
        M=M,
        STRIDE_ROW=block_table.stride(0),
        BLOCK_SIZE=BLOCK_SIZE,
    )


@ragged_to_block_table_triton.register_fake
def _ragged_to_block_table_triton_fake(
    cache_loc: torch.Tensor,
    cu_num_blocks: torch.Tensor,
    block_table: torch.Tensor,
    num_sequences: int,
) -> None:
    pass


# ========================================================================================
# Custom ops: adjust block index (append / remove / no-op per sequence)
# ========================================================================================


@torch.library.custom_op(
    "auto_deploy::adjust_block_table_torch", mutates_args=("block_table", "num_blocks")
)
def adjust_block_table_torch(
    block_table: torch.Tensor,
    num_blocks: torch.Tensor,
    extra_idx: torch.Tensor,
    delta: torch.Tensor,
    num_sequences: int,
) -> None:
    """Adjust block_table per sequence: append (delta=+1), remove last (delta=-1), or no-op."""
    N = num_sequences
    if N == 0:
        return

    device = block_table.device
    seq_idx = torch.arange(N, device=device)

    append_mask = delta[:N] > 0
    row_append = seq_idx[append_mask]
    col_append = num_blocks[row_append].long()
    block_table[row_append, col_append] = extra_idx[row_append]

    nonzero_mask = delta[:N] != 0
    row_nonzero = seq_idx[nonzero_mask]
    num_blocks[row_nonzero] += delta[row_nonzero]


@adjust_block_table_torch.register_fake
def _adjust_block_table_torch_fake(
    block_table: torch.Tensor,
    num_blocks: torch.Tensor,
    extra_idx: torch.Tensor,
    delta: torch.Tensor,
    num_sequences: int,
) -> None:
    pass


@torch.library.custom_op(
    "auto_deploy::adjust_ragged_torch", mutates_args=("cache_loc", "cu_num_blocks")
)
def adjust_ragged_torch(
    cache_loc: torch.Tensor,
    cu_num_blocks: torch.Tensor,
    extra_idx: torch.Tensor,
    delta: torch.Tensor,
    num_sequences: int,
) -> int:
    """Adjust ragged cache_loc per sequence: append (+1), remove last (-1), or no-op (0)."""
    N = num_sequences
    device = cache_loc.device

    if N == 0:
        return 0

    d = delta[:N]

    old_lens = (cu_num_blocks[1 : N + 1] - cu_num_blocks[:N]).to(torch.int32)
    new_lens = old_lens + d

    new_cu = torch.zeros(N + 1, device=device, dtype=cu_num_blocks.dtype)
    new_cu[1:] = torch.cumsum(new_lens, dim=0)
    total_new = int(new_cu[N].item())

    if total_new == 0:
        cu_num_blocks[: N + 1] = new_cu
        return 0

    copy_lens = old_lens + torch.clamp(d, max=0)
    max_new_len = int(new_lens.max().item())
    col = torch.arange(max_new_len, device=device)

    copy_mask = col.unsqueeze(0) < copy_lens.unsqueeze(1)
    append_mask = (col.unsqueeze(0) == old_lens.unsqueeze(1)) & (d > 0).unsqueeze(1)

    old_cu = cu_num_blocks[:N]
    src = old_cu.unsqueeze(1) + col.unsqueeze(0)
    old_total = int(cu_num_blocks[N].item())
    src_safe = src.clamp(max=max(old_total - 1, 0))

    zero = torch.zeros(1, device=device, dtype=cache_loc.dtype)
    vals = torch.where(copy_mask, cache_loc[src_safe], zero)
    vals = torch.where(append_mask, extra_idx[:N].unsqueeze(1).expand_as(vals), vals)

    dst = new_cu[:N].unsqueeze(1) + col.unsqueeze(0)

    write_mask = copy_mask | append_mask
    temp = torch.zeros(total_new, device=device, dtype=cache_loc.dtype)
    temp[dst[write_mask].long()] = vals[write_mask]

    cache_loc[:total_new] = temp
    cu_num_blocks[: N + 1] = new_cu

    return total_new


@adjust_ragged_torch.register_fake
def _adjust_ragged_torch_fake(
    cache_loc: torch.Tensor,
    cu_num_blocks: torch.Tensor,
    extra_idx: torch.Tensor,
    delta: torch.Tensor,
    num_sequences: int,
) -> int:
    return 0


@torch.library.custom_op(
    "auto_deploy::adjust_block_table_triton", mutates_args=("block_table", "num_blocks")
)
def adjust_block_table_triton(
    block_table: torch.Tensor,
    num_blocks: torch.Tensor,
    extra_idx: torch.Tensor,
    delta: torch.Tensor,
    num_sequences: int,
) -> None:
    """Adjust block_table using Triton. In-place on block_table and num_blocks."""
    N = num_sequences
    M = block_table.shape[1]

    if N == 0:
        return

    _adjust_block_table_kernel[(N,)](block_table, num_blocks, extra_idx, delta, M=M)


@adjust_block_table_triton.register_fake
def _adjust_block_table_triton_fake(
    block_table: torch.Tensor,
    num_blocks: torch.Tensor,
    extra_idx: torch.Tensor,
    delta: torch.Tensor,
    num_sequences: int,
) -> None:
    pass


@torch.library.custom_op(
    "auto_deploy::adjust_ragged_triton", mutates_args=("cache_loc", "cu_num_blocks")
)
def adjust_ragged_triton(
    cache_loc: torch.Tensor,
    cu_num_blocks: torch.Tensor,
    extra_idx: torch.Tensor,
    delta: torch.Tensor,
    num_sequences: int,
) -> int:
    """Adjust ragged cache_loc using Triton. Uses temp buffer + copy back."""
    N = num_sequences
    device = cache_loc.device

    if N == 0:
        return 0

    d = delta[:N]
    old_lens = (cu_num_blocks[1 : N + 1] - cu_num_blocks[:N]).to(torch.int32)
    new_lens = old_lens + d

    old_cu = cu_num_blocks[: N + 1].clone()

    new_cu = torch.zeros(N + 1, device=device, dtype=cu_num_blocks.dtype)
    new_cu[1:] = torch.cumsum(new_lens, dim=0)
    total_new = int(new_cu[N].item())

    if total_new == 0:
        cu_num_blocks[: N + 1] = new_cu
        return 0

    temp = torch.zeros(total_new, device=device, dtype=cache_loc.dtype)

    max_old_len = int(old_lens.max().item())
    BLOCK_SIZE = triton.next_power_of_2(max(max_old_len, 1))

    _adjust_ragged_kernel[(N,)](
        cache_loc, temp, old_cu, new_cu, extra_idx, delta, BLOCK_SIZE=BLOCK_SIZE
    )

    cache_loc[:total_new] = temp
    cu_num_blocks[: N + 1] = new_cu

    return total_new


@adjust_ragged_triton.register_fake
def _adjust_ragged_triton_fake(
    cache_loc: torch.Tensor,
    cu_num_blocks: torch.Tensor,
    extra_idx: torch.Tensor,
    delta: torch.Tensor,
    num_sequences: int,
) -> int:
    return 0
