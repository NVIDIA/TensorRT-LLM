# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The DFlash worker's drafter page-table decode in one launch (``drafter_block_table_decode``).

When the drafter's context lives in the KV cache manager's pages, every step the worker's
``_refresh_ctx_block_tables`` (``speculative/dflash.py``) decodes the batch's rows of the
manager's attention-op table for the drafter's pool into its persistent page table and
the per-row extent:

* ``encoded = int64(src[pool, :num_seqs, 0]) - offset``;
* ``counts[r] = max over c of (c + 1) where encoded[r, c] >= 0``, else 0: the column after
  the row's last valid entry (a sliding window's evicted pages are placeholders before
  the live tail, so this is not the number of valid entries);
* ``table[r, c] = int32(max(encoded[r, c], 0) // divisor)`` for every column.

The built-in runs that as eleven torch kernels over ``[num_seqs, max_blocks_per_seq]``
int64 temporaries (the widening, a clone, the subtraction, the mask, the column product,
the row maximum, the counts copy, the clamp, the division, the narrowing, the table copy).
``decode`` runs it as one Triton launch, one program per row, reading the source row once
as int32 and writing the table row once as int32. The arithmetic is the built-in's,
element for element: int64 subtraction, the comparison, the clamp, a division of a
non-negative value by the positive divisor (truncation and floor agree there), and the
int32 narrowing, so the table rows ``[0, num_seqs)`` and ``counts[:num_seqs]`` hold the
built-in's words, every column included; nothing else is written.

Every column is decoded: the extent of a row is found from its source row, which has to be
read up to the row end anyway, and the table is written through as well as read (the
context store appends through it), so no column is left holding another request's page.

``decode`` takes the call when the worker is bound to the manager's pages and the tensors
have the layouts the kernel addresses (``unsupported_reason``); otherwise the caller runs
the built-in.

The context-K/V fork's side chain over pages decodes with ``ctx_block_tables_paged``
(``paged_ctx_block_tables``) on the steps it takes. This entry takes the worker's own
decode on every other step, such as a step of more requests than the fork takes over
pages, and only under the opt-in ``drafter_block_table_decode``.
"""

from __future__ import annotations

from typing import Optional

import torch

try:
    import triton
    import triton.language as tl

    @triton.jit(do_not_specialize=["num_cols", "offset", "divisor"])
    def _ctx_block_table_kernel(
        src,
        src_row_stride,
        table,
        table_row_stride,
        counts,
        num_cols,
        offset,
        divisor,
        BLOCK: tl.constexpr,
    ):
        row = tl.program_id(0).to(tl.int64)
        src_row = src + row * src_row_stride
        table_row = table + row * table_row_stride
        extent = tl.zeros([BLOCK], dtype=tl.int32)
        for start in range(0, num_cols, BLOCK):
            cols = start + tl.arange(0, BLOCK)
            live = cols < num_cols
            encoded = tl.load(src_row + cols, mask=live, other=0).to(tl.int64) - offset
            valid = live & (encoded >= 0)
            extent = tl.maximum(extent, tl.where(valid, cols + 1, 0))
            decoded = tl.maximum(encoded, 0) // divisor
            tl.store(table_row + cols, decoded.to(tl.int32), mask=live)
        tl.store(counts + row, tl.max(extent, axis=0).to(tl.int64))

    _KERNEL = True
except ImportError:  # pragma: no cover
    _KERNEL = False

#: Columns one program decodes per iteration, and its warps.
BLOCK = 4096
_NUM_WARPS = 8


def _source(worker, attn_metadata) -> Optional[torch.Tensor]:
    """The manager's attention-op table the built-in decodes (its ``offsets_name``), or None."""
    name = "kv_cache_block_offsets" if worker._ctx_unified_cache else "draft_kv_cache_block_offsets"
    return getattr(attn_metadata, name, None)


def unsupported_reason(worker, src: torch.Tensor, num_seqs: int) -> Optional[str]:
    """Why ``decode`` does not take this call, or None. Host-side reads only.

    Args:
        worker: the DFlash worker, bound to the manager's pages (``_ctx_block_tables`` set).
        src: the manager's attention-op table, ``[pools, rows, 2, max_blocks_per_seq]``.
        num_seqs: the batch rows to decode, at least 1.
    """
    table = worker._ctx_block_tables
    counts = worker._ctx_block_counts
    if not _KERNEL:
        return "no Triton"
    if src.dim() != 4 or src.dtype != torch.int32 or not src.is_cuda or src.stride(3) != 1:
        return "the attention-op table is not a CUDA int32 [pools, rows, 2, blocks] table"
    if (
        table.dim() != 2
        or table.dtype != torch.int32
        or not table.is_contiguous()
        or table.device != src.device
    ):
        return "the drafter page table is not a contiguous int32 table on the table's device"
    if counts is None or counts.dtype != torch.int64 or counts.stride(0) != 1:
        return "the drafter page counts are not a dense int64 vector"
    if table.shape[1] != src.shape[3]:
        return "the drafter page table is not as wide as the attention-op table"
    if not 0 <= worker._ctx_pool_idx < src.shape[0]:
        return "the drafter pool is not a table of the attention-op table"
    if num_seqs > min(src.shape[1], table.shape[0], counts.shape[0]):
        return "more rows than the tables hold"
    if int(worker._ctx_block_divisor) <= 0:
        return "the page divisor is not positive"
    return None


def decode(worker, src: torch.Tensor, num_seqs: int) -> None:
    """Decode rows ``[0, num_seqs)`` of ``src`` into the worker's page table and counts.

    The caller checked ``unsupported_reason``. One launch on the current stream; it
    allocates nothing, so it is captured into a CUDA graph as it is.
    """
    rows = src[worker._ctx_pool_idx, :, 0]
    table = worker._ctx_block_tables
    num_cols = rows.shape[1]
    block = min(BLOCK, triton.next_power_of_2(num_cols))
    _ctx_block_table_kernel[(num_seqs,)](
        rows,
        rows.stride(0),
        table,
        table.stride(0),
        worker._ctx_block_counts,
        num_cols,
        int(worker._ctx_block_offset),
        int(worker._ctx_block_divisor),
        BLOCK=block,
        num_warps=_NUM_WARPS if block >= 1024 else 4,
    )


def refresh(worker, attn_metadata, num_seqs: int, declined=None) -> Optional[bool]:
    """The worker's ``_refresh_ctx_block_tables`` in one launch, or None to run the built-in.

    None, with the built-in left to answer, when there is no table to fill or no source
    (the built-in returns False or raises), and when ``unsupported_reason`` refuses the
    call, which is then reported to ``declined`` (a callable taking the reason) if given.

    Returns:
        True once the launch is issued; None when the caller runs the built-in.
    """
    if worker._ctx_block_tables is None or num_seqs <= 0:
        return None
    src = _source(worker, attn_metadata)
    if src is None:
        return None
    reason = unsupported_reason(worker, src, num_seqs)
    if reason is not None:
        if declined is not None:
            declined(reason)
        return None
    decode(worker, src, num_seqs)
    return True
