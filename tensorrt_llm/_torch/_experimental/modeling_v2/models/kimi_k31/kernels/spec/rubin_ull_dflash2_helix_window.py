# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pack a rank's visible DFlash2 window for the existing TRTLLM MLA decode kernel.

Row ``r`` of request ``r // WIDTH`` reads its page-table row ``table_rows[r // WIDTH]``; pool
offsets are 64-bit (a page index times the page stride passes 2**31 in a large pool)."""

import triton
import triton.language as tl


@triton.jit
def pack_helix_window(
    pool,
    block,
    tables,
    table_rows,
    lengths,
    packed,
    local_lengths,
    TABLE_STRIDE: tl.constexpr,
    POOL_STRIDE: tl.constexpr,
    WIDTH: tl.constexpr,
    PAGE: tl.constexpr,
    CP: tl.constexpr,
    RANK: tl.constexpr,
    WINDOW: tl.constexpr,
    CAP: tl.constexpr,
    DEQUANTIZE: tl.constexpr,
    empty,
):
    row, tile = tl.program_id(0), tl.program_id(1)
    request, offset = row // WIDTH, row % WIDTH
    length = tl.load(lengths + request)
    end = length + offset + 1
    start = tl.maximum(0, end - WINDOW)
    cycle: tl.constexpr = CP * PAGE
    first = start // cycle * PAGE + tl.minimum(tl.maximum(start % cycle - RANK * PAGE, 0), PAGE)
    last = end // cycle * PAGE + tl.minimum(tl.maximum(end % cycle - RANK * PAGE, 0), PAGE)
    if tile == 0:
        tl.store(local_lengths + row, tl.maximum(last - first, 1))
        tl.store(empty + row, last == first)
    j = tile * 8 + tl.arange(0, 8)
    d = tl.arange(0, 1024)
    slot = first + j
    position = (slot // PAGE * CP + RANK) * PAGE + slot % PAGE
    valid = (j < last - first) & (j < CAP)
    history = valid & (position < length)
    table_row = tl.load(table_rows + request).to(tl.int64)
    page = tl.load(tables + table_row * TABLE_STRIDE + slot // PAGE, history, 0).to(tl.int64)
    old = tl.load(
        pool + page[:, None] * POOL_STRIDE + (slot % PAGE)[:, None] * 576 + d[None, :],
        history[:, None] & (d[None, :] < 576),
        0,
    )
    proposed = valid & (position >= length)
    new = tl.load(
        block + (request * WIDTH + position - length)[:, None] * 576 + d[None, :],
        proposed[:, None] & (d[None, :] < 576),
        0,
    )
    values = tl.where(proposed[:, None], new, old)
    if DEQUANTIZE:
        values = values.to(tl.float8e4nv, bitcast=True).to(tl.bfloat16)
    tl.store(
        packed + (row * CAP + j[:, None]) * 576 + d[None, :],
        values,
        (j[:, None] < CAP) & (d[None, :] < 576),
    )
