# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Restore the worker's full row layout after evaluating only accepted MTP tail rows."""

import torch
import triton
import triton.language as tl


@triton.jit
def _glm_rubin_mtp_tail_scatter_kernel(
    selected,
    row_ids,
    output,
    H: tl.constexpr,
    REQUESTS: tl.constexpr,
    REQUEST_TILE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    request = tl.arange(0, REQUEST_TILE)
    selected_row = tl.load(row_ids + request, request < REQUESTS, -1)
    owner = tl.sum(tl.where((request < REQUESTS) & (selected_row == row), request + 1, 0)) - 1
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    value = tl.load(selected + tl.maximum(owner, 0) * H + cols, (owner >= 0) & (cols < H), 0)
    tl.store(output + row * H + cols, value, cols < H)


@torch.library.custom_op("trtllm::glm_rubin_mtp_tail_scatter", mutates_args=())
def scatter_mtp_tail(selected: torch.Tensor, row_ids: torch.Tensor, num_rows: int) -> torch.Tensor:
    output = selected.new_empty((num_rows, selected.shape[1]))
    _glm_rubin_mtp_tail_scatter_kernel[(num_rows, triton.cdiv(selected.shape[1], 256))](
        selected,
        row_ids,
        output,
        selected.shape[1],
        selected.shape[0],
        triton.next_power_of_2(selected.shape[0]),
        256,
    )
    return output


@scatter_mtp_tail.register_fake
def _fake_scatter_mtp_tail(
    selected: torch.Tensor, row_ids: torch.Tensor, num_rows: int
) -> torch.Tensor:
    return selected.new_empty((num_rows, selected.shape[1]))
