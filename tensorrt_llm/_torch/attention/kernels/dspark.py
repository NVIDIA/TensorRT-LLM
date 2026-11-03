# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Direct paged captured-context writes for embedded DSpark."""

import torch
import triton
import triton.language as tl


@triton.jit
def _write_context_kernel(
    values,
    positions,
    mask,
    pool,
    tables,
    capacities,
    M: tl.constexpr,
    D: tl.constexpr,
    TABLE_STRIDE: tl.constexpr,
    PAGE_STRIDE: tl.constexpr,
    TOKEN_STRIDE: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    token = tl.program_id(1)
    d = tl.arange(0, BLOCK)
    # RoPE uses frame p + 1; the managed cache stores logical token p.
    position = tl.load(positions + row * M + token) - 1
    capacity = tl.load(capacities + row)
    valid = tl.load(mask + row * M + token) & (position >= 0) & (position < capacity)
    page = tl.load(tables + row * TABLE_STRIDE + position // PAGE_SIZE, valid, other=-1)
    valid = valid & (page >= 0)
    value = tl.load(values + (row * M + token) * D + d, (d < D) & valid, other=0)
    tl.store(
        pool + page.to(tl.int64) * PAGE_STRIDE + (position % PAGE_SIZE) * TOKEN_STRIDE + d,
        value,
        (d < D) & valid,
    )


def write_dspark_context(
    values: torch.Tensor,
    positions: torch.Tensor,
    mask: torch.Tensor,
    pool: torch.Tensor,
    block_tables: torch.Tensor,
    capacities: torch.Tensor,
) -> None:
    """Write [batch, tokens, head_dim] KV, masking invalid and dummy rows before stores."""
    batch, tokens, head_dim = values.shape
    if batch == 0 or tokens == 0:
        return
    _write_context_kernel[(batch, tokens)](
        values.contiguous(),
        positions.contiguous(),
        mask.contiguous(),
        pool,
        block_tables,
        capacities,
        tokens,
        head_dim,
        block_tables.stride(0),
        pool.stride(0),
        pool.stride(1),
        pool.shape[1],
        triton.next_power_of_2(head_dim),
    )
