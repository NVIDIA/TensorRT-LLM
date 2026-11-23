# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Inputs of a DSpark back-fill that also writes the bonus rows' main-KV frames, in one launch."""

import torch
import triton
import triton.language as tl

_BLOCK = 1024


@triton.jit
def _main_kv_backfill_inputs_kernel(
    start_ptr,
    count_ptr,
    slots_ptr,
    rows_ptr,
    main_x_ptr,
    pos_ptr,
    valid_ptr,
    slots32_ptr,
    start32_ptr,
    bonus_ptr,
    hidden,
    WIDTH: tl.constexpr,
    WIDTH_PAD: tl.constexpr,
    BLOCK: tl.constexpr,
):
    g = tl.program_id(0)
    part = tl.program_id(1)
    if part == 0:
        start = tl.load(start_ptr + g).to(tl.int64)
        count = tl.load(count_ptr + g).to(tl.int64)
        j = tl.arange(0, WIDTH_PAD)
        live = j < WIDTH
        tl.store(pos_ptr + g * WIDTH + j, start - count + 1 + j, mask=live)
        tl.store(valid_ptr + g * WIDTH + j, (j < count).to(tl.int8), mask=live)
        tl.store(slots32_ptr + g, tl.load(slots_ptr + g).to(tl.int32))
        tl.store(start32_ptr + g, start.to(tl.int32))
    row = tl.load(rows_ptr + g).to(tl.int64)
    offs = part * BLOCK + tl.arange(0, BLOCK)
    inside = offs < hidden
    tl.store(
        bonus_ptr + g * hidden + offs,
        tl.load(main_x_ptr + row * hidden + offs, mask=inside),
        mask=inside,
    )


def main_kv_backfill_inputs(
    start_pos: torch.Tensor,
    counts: torch.Tensor,
    slots: torch.Tensor,
    main_rows: torch.Tensor,
    main_x: torch.Tensor,
    width: int,
) -> tuple[torch.Tensor, ...]:
    """Window positions and validity ``[G, width]`` of the verified rows (the ``counts`` accepted
    ones end at ``start_pos``), int32 slots and start_pos, and bonus rows ``main_x[main_rows]``."""
    gens = start_pos.shape[0]
    hidden = main_x.shape[-1]
    device = main_x.device
    positions = torch.empty((gens, width), dtype=torch.int64, device=device)
    valid = torch.empty((gens, width), dtype=torch.bool, device=device)
    slots32 = torch.empty(gens, dtype=torch.int32, device=device)
    start32 = torch.empty(gens, dtype=torch.int32, device=device)
    bonus = torch.empty((gens, hidden), dtype=main_x.dtype, device=device)
    if gens:
        _main_kv_backfill_inputs_kernel[(gens, triton.cdiv(hidden, _BLOCK))](
            start_pos,
            counts,
            slots,
            main_rows,
            main_x,
            positions,
            valid.view(torch.int8),
            slots32,
            start32,
            bonus,
            hidden,
            WIDTH=width,
            WIDTH_PAD=triton.next_power_of_2(width),
            BLOCK=_BLOCK,
        )
    return positions, valid, slots32, start32, bonus
