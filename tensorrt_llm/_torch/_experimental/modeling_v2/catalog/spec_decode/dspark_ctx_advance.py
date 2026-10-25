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
# DSpark / DFlash drafter context advance + read view: one launch per step
# =============================================================================
#
# It replaces, bit for bit, what DFlashWorker's generation branch does after the context store on a bound draft pool
# (_advance_ctx_len, then the advertised context length and the rebased read view of dflash_rebased_ctx_view):
#   ctx_len[slot] = min(ctx_len[slot] + num_accepted, max_ctx); ctx_len[dummy_slot] = 0
#   num_ctx       = min(ctx_len[slot], max(block_counts[row] * page - block_size, 0))
#   rebase        = min(base_pages[slot], num_ctx // page)
#   num_ctx_out   = max(num_ctx - base_pages[slot] * page, 0)                                   (int64)
#   read[row, j]  = tables[row, min(j + rebase, width - 1)]                                     (int32)
# One program per generation request, so each request's advance is read back by the same program; rows on the dummy
# slot (padding) skip the add and read 0, which the stock path's zeroing after the add makes them read too.

import torch
import triton
import triton.language as tl


@triton.jit
def _advance_kernel(
    slots_ptr,  # [n] int64
    rows_ptr,  # [n] int64: block-table rows of the requests
    nacc_ptr,  # [n] int
    ctx_len_ptr,  # [slots] int64, in / out
    counts_ptr,  # [rows] int64
    base_ptr,  # [slots] int64
    tables_ptr,  # int32 [rows, width]
    read_ptr,  # int32 [rows, width]
    out_ptr,  # [n] int64 advertised context lengths
    width,
    tables_stride,
    read_stride,
    max_ctx,
    dummy_slot,
    page,
    block_size,
    BLOCK: tl.constexpr,
):
    r = tl.program_id(0).to(tl.int64)
    slot = tl.load(slots_ptr + r)
    row = tl.load(rows_ptr + r)
    is_dummy = slot == dummy_slot
    old = tl.load(ctx_len_ptr + slot)
    new = tl.minimum(old + tl.load(nacc_ptr + r).to(tl.int64), max_ctx)
    if not is_dummy:
        tl.store(ctx_len_ptr + slot, new)
    if r == 0:
        tl.store(ctx_len_ptr + dummy_slot, tl.zeros((), dtype=tl.int64))
    new = tl.where(is_dummy, 0, new)
    allocated = tl.maximum(tl.load(counts_ptr + row) * page - block_size, 0)
    num_ctx = tl.minimum(new, allocated)
    base = tl.load(base_ptr + slot)
    rebase = tl.minimum(base, num_ctx // page)
    tl.store(out_ptr + r, tl.maximum(num_ctx - base * page, 0))
    for start in range(0, width, BLOCK):
        j = start + tl.arange(0, BLOCK)
        m = j < width
        src = tl.minimum(j + rebase, width - 1)
        v = tl.load(tables_ptr + row * tables_stride + src, mask=m)
        tl.store(read_ptr + row * read_stride + j, v, mask=m)


def advance_ctx_and_read_view(
    slots: torch.Tensor,
    rows: torch.Tensor,
    num_accepted: torch.Tensor,
    ctx_len: torch.Tensor,
    max_ctx: int,
    dummy_slot: int,
    block_counts: torch.Tensor,
    base_pages: torch.Tensor,
    tables: torch.Tensor,
    tables_read: torch.Tensor,
    page_size: int,
    block_size: int,
):
    """Advance the generation requests' context lengths and build their read view in one launch: returns the
    advertised context lengths [n] (int64) and writes ``tables_read[rows]``; None (nothing written) for a layout the
    kernel does not reproduce, so the caller keeps the stock path."""
    n = slots.shape[0]
    if not (
        slots.dim() == 1
        and rows.shape == (n,)
        and num_accepted.shape == (n,)
        and num_accepted.dtype in (torch.int32, torch.int64)
        and ctx_len.dtype == torch.int64
        and block_counts.dtype == torch.int64
        and base_pages.dtype == torch.int64
        and tables.dtype == torch.int32
        and tables_read.dtype == torch.int32
        and tables.shape == tables_read.shape
        and tables.stride(1) == 1
        and tables_read.stride(1) == 1
        and page_size >= 1
    ):
        return None
    out = torch.empty(n, dtype=torch.int64, device=ctx_len.device)
    if n == 0:
        ctx_len[dummy_slot] = 0
        return out
    width = tables.shape[1]
    _advance_kernel[(n,)](
        slots.to(torch.int64).contiguous(),
        rows.to(torch.int64).contiguous(),
        num_accepted.contiguous(),
        ctx_len,
        block_counts,
        base_pages,
        tables,
        tables_read,
        out,
        width,
        tables.stride(0),
        tables_read.stride(0),
        int(max_ctx),
        int(dummy_slot),
        int(page_size),
        int(block_size),
        BLOCK=min(1024, triton.next_power_of_2(width)),
        num_warps=4,
    )
    return out
