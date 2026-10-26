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
# =============================================================================
# DSpark / DFlash drafter block-table refresh: one launch per step
# =============================================================================
#
# It replaces, bit for bit, what DFlashWorker._refresh_ctx_block_tables does with the draft pool's encoded block
# offsets of this step (field 0 of the K/V pair, pool ``pool_idx``, the first ``num_seqs`` rows):
#   counts[r] = number of entries >= 0 in row r (placeholders are negative)
#   tables[r, j] = max(offset[r, j], 0) // divisor   (int32)
# Integer arithmetic only, so the result is exact; the stock path spends one launch per torch op on it.

from typing import Optional

import torch
import triton
import triton.language as tl


@triton.jit
def _refresh_kernel(
    src_ptr,  # encoded offsets, row r at src_ptr + r * src_stride_row, entry j at + j * src_stride_col
    src_stride_row,
    src_stride_col,
    tables_ptr,  # int32 [rows, width], row-major
    tables_stride,
    counts_ptr,  # int64 [rows]
    width,
    divisor,
    BLOCK: tl.constexpr,
):
    r = tl.program_id(0).to(tl.int64)
    count = tl.zeros((), dtype=tl.int64)
    for start in range(0, width, BLOCK):
        o = start + tl.arange(0, BLOCK)
        m = o < width
        v = tl.load(src_ptr + r * src_stride_row + o * src_stride_col, mask=m, other=-1).to(
            tl.int64
        )
        count += tl.sum((v >= 0).to(tl.int64), axis=0)
        # Non-negative after the clamp, so integer division is the stock floor division.
        tl.store(
            tables_ptr + r * tables_stride + o, (tl.maximum(v, 0) // divisor).to(tl.int32), mask=m
        )
    tl.store(counts_ptr + r, count)


@triton.jit
def _refresh_kernel_2d(
    src_ptr,
    src_stride_row,
    src_stride_col,
    tables_ptr,
    tables_stride,
    counts_ptr,
    width,
    divisor,
    BLOCK: tl.constexpr,
    COUNT_BLOCK: tl.constexpr,
):
    """S10c-B2: grid (rows, column chunks). Each program decodes one chunk in int32 (the offsets fit int32 and are
    non-negative after the clamp, so int32 floor division is the stock floor division); chunk 0 of a row also counts
    the row's non-placeholder entries over the whole row. No atomics: every output has exactly one writer."""
    r = tl.program_id(0).to(tl.int64)
    c = tl.program_id(1)
    o = c * BLOCK + tl.arange(0, BLOCK)
    m = o < width
    v = tl.load(src_ptr + r * src_stride_row + o * src_stride_col, mask=m, other=-1).to(tl.int32)
    tl.store(tables_ptr + r * tables_stride + o, tl.maximum(v, 0) // divisor, mask=m)
    if c == 0:
        count = tl.zeros((), dtype=tl.int64)
        for start in range(0, width, COUNT_BLOCK):
            oc = start + tl.arange(0, COUNT_BLOCK)
            vc = tl.load(
                src_ptr + r * src_stride_row + oc * src_stride_col, mask=oc < width, other=-1
            )
            count += tl.sum((vc >= 0).to(tl.int64), axis=0)
        tl.store(counts_ptr + r, count)


def refresh_block_tables(
    offsets: torch.Tensor,
    pool_idx: int,
    num_seqs: int,
    divisor: int,
    tables: torch.Tensor,
    counts: torch.Tensor,
) -> bool:
    """Decode this step's draft block offsets ``offsets[pool_idx, :num_seqs, 0]`` into ``tables[:num_seqs]`` and their
    non-placeholder counts into ``counts[:num_seqs]``, in one launch; returns False (nothing written) when the layout
    is not one the kernel reproduces, so the caller keeps the stock path."""
    if not supported(offsets, tables, counts, divisor):
        return False
    if num_seqs <= 0:
        return True
    src = offsets[pool_idx, :num_seqs, 0]
    width = src.shape[-1]
    if _parallel[0] and divisor <= 2**31 - 1 and (offsets.dtype == torch.int32 or _fits_int32(src)):
        block = min(1024, triton.next_power_of_2(width))
        _refresh_kernel_2d[(num_seqs, triton.cdiv(width, block))](
            src,
            src.stride(0),
            src.stride(1),
            tables,
            tables.stride(0),
            counts,
            width,
            divisor,
            BLOCK=block,
            COUNT_BLOCK=min(4096, triton.next_power_of_2(width)),
            num_warps=4,
        )
        return True
    _refresh_kernel[(num_seqs,)](
        src,
        src.stride(0),
        src.stride(1),
        tables,
        tables.stride(0),
        counts,
        width,
        divisor,
        BLOCK=min(1024, triton.next_power_of_2(width)),
        num_warps=1,
    )
    return True


_parallel = [False]  # S10c-B2: the 2-D int32 decode (see enable_parallel_refresh)


def enable_parallel_refresh(enabled: bool = True) -> bool:
    """S10c-B2: decode with the (row, chunk) grid in int32 instead of one int64 program per row (the same values: the
    offsets fit int32 and are non-negative after the clamp). Off by default."""
    _parallel[0] = bool(enabled)
    return _parallel[0]


def _fits_int32(src: torch.Tensor) -> bool:
    """int64 offsets are decoded in int32 only when the caller's layout guarantees they fit; without a cheap
    capture-safe check, int64 offsets keep the int64 kernel."""
    return False


def supported(
    offsets: torch.Tensor, tables: torch.Tensor, counts: torch.Tensor, divisor: Optional[int]
) -> bool:
    """[pools, rows, 2, width] int32 / int64 offsets on the tables' device; int32 row-major tables of the same width;
    int64 counts; a positive integer divisor."""
    return (
        offsets.dim() == 4
        and offsets.dtype in (torch.int32, torch.int64)
        and tables.dtype == torch.int32
        and tables.dim() == 2
        and tables.stride(1) == 1
        and tables.shape[1] == offsets.shape[-1]
        and counts.dtype == torch.int64
        and offsets.device == tables.device == counts.device
        and isinstance(divisor, int)
        and divisor >= 1
    )
