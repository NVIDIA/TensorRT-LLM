# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Positions and validity for a DSpark backfill including the bonus frame."""

import torch
import triton
import triton.language as tl


@triton.jit
def _main_kv_backfill_inputs_kernel(
    start_ptr,
    count_ptr,
    pos_ptr,
    valid_ptr,
    WIDTH: tl.constexpr,
    WIDTH_PAD: tl.constexpr,
):
    row = tl.program_id(0)
    start = tl.load(start_ptr + row).to(tl.int64)
    count = tl.load(count_ptr + row).to(tl.int64)
    j = tl.arange(0, WIDTH_PAD)
    live = j < WIDTH
    tl.store(pos_ptr + row * WIDTH + j, start - count + 1 + j, mask=live)
    tl.store(valid_ptr + row * WIDTH + j, (j < count).to(tl.int8), mask=live)


def main_kv_backfill_inputs(
    start_pos: torch.Tensor, counts: torch.Tensor, width: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Positions and validity [G, width]; each request's accepted frames end at start_pos."""
    gens = start_pos.shape[0]
    positions = start_pos.new_empty((gens, width), dtype=torch.int64)
    valid = torch.empty((gens, width), dtype=torch.bool, device=start_pos.device)
    if gens:
        _main_kv_backfill_inputs_kernel[(gens,)](
            start_pos,
            counts,
            positions,
            valid.view(torch.int8),
            WIDTH=width,
            WIDTH_PAD=triton.next_power_of_2(width),
            num_warps=1,
        )
    return positions, valid
