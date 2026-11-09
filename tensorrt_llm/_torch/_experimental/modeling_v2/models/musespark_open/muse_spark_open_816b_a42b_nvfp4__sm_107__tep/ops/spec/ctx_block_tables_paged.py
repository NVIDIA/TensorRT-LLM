# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DFlash page table of a paged drafter context: the decode of the manager's block offsets, in two launches.

With the drafter's context K/V in the KV cache manager's pages (``TRTLLM`` drafter
attention), the context-K/V fork's side chain (``context_kv_fork/fork.py``) first
decodes the step's drafter page table with the worker's
``_refresh_ctx_block_tables`` (``_torch/speculative/dflash.py``), about ten launches
over every column of the table:

* the K field of the drafter pool's row of each request in the attention metadata's
  block offsets, as int64, less the pool's block offset;
* the extent, ``amax`` over ``(entry >= 0) * (column + 1)``: the last column holding a
  page, plus one;
* the entries clamped at 0 and floor-divided by the pool's block divisor, as int32.

This entry zeroes the requests' extents, then reads each entry once and writes the
table rows, one program per request and column block; each program folds its block's
extent into the request's with an atomic maximum. Integer arithmetic: bitwise the
built-in.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, OpWrapper

#: Requests one call takes.
MAX_REQUESTS = 64
#: Columns per program.
COL_BLOCK = 1024


@triton.jit
def _ctx_block_tables_paged_kernel(
    src_ptr,
    stride_src_row,
    stride_src_col,
    tables_ptr,
    stride_tables_row,
    counts_ptr,
    cols,
    block_offset,
    divisor,
    COL_BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    col = tl.program_id(1).to(tl.int64) * COL_BLOCK + tl.arange(0, COL_BLOCK).to(tl.int64)
    in_row = col < cols
    entry = (
        tl.load(src_ptr + row * stride_src_row + col * stride_src_col, mask=in_row, other=0).to(
            tl.int64
        )
        - block_offset
    )
    page = tl.maximum(entry, 0) // divisor
    tl.store(tables_ptr + row * stride_tables_row + col, page.to(tl.int32), mask=in_row)
    # The extent: the last column holding a page, plus one (0 when none does).
    extent = tl.max(tl.where(in_row & (entry >= 0), col + 1, 0), axis=0)
    tl.atomic_max(counts_ptr + row, extent, mask=extent > 0)


class CtxBlockTablesPaged(OpWrapper):
    """Decode a paged drafter context's page table and its extents in two launches.

    Arguments (``reference`` takes the same ones):
        src            [n, cols] int32: the K field of the drafter pool's rows of the
                       attention metadata's block offsets
                       (``kv_cache_block_offsets[pool_idx, :n, 0]``, or the draft
                       manager's), any row stride, column stride 1
        block_offset   the worker's ``_ctx_block_offset`` (the pool's first block)
        divisor        the worker's ``_ctx_block_divisor`` (block indices per page)
        tables         the worker's ``_ctx_block_tables`` ([slots, cols] int32); rows
                       ``:n`` are written
        counts         the worker's ``_ctx_block_counts`` ([slots] int64); ``:n`` written
    """

    ARCHS = frozenset({Arch.SM_107})

    note = """
    Metadata consumed: the attention metadata's block offsets (read) and the worker's
    ``_ctx_block_tables`` / ``_ctx_block_counts`` (written, rows ``:n``). The extents'
    zeroing, then one Triton program per request and column block (grid n x
    ceil(cols / COL_BLOCK), 4 warps), JIT-compiled once; no programmatic-dependent-launch
    attributes.

    Call site: ``ContextKvFork``'s side chain over a paged drafter context, under the
    ``paged_ctx_block_tables`` fast path, in place of ``_refresh_ctx_block_tables``'s
    tensor statements.

    Numerics: integer results, bitwise the built-in method (the reference).

    Hard limits
      n <= MAX_REQUESTS; block_offset >= 0; divisor >= 1.
    """

    def __call__(
        self,
        src: torch.Tensor,
        block_offset: int,
        divisor: int,
        tables: torch.Tensor,
        counts: torch.Tensor,
    ) -> None:
        n, cols = src.shape
        counts[:n].zero_()
        _ctx_block_tables_paged_kernel[(n, triton.cdiv(cols, COL_BLOCK))](
            src,
            src.stride(0),
            src.stride(1),
            tables,
            tables.stride(0),
            counts,
            cols,
            block_offset,
            divisor,
            COL_BLOCK=COL_BLOCK,
            num_warps=4,
        )

    def reference(
        self,
        src: torch.Tensor,
        block_offset: int,
        divisor: int,
        tables: torch.Tensor,
        counts: torch.Tensor,
    ) -> None:
        """``_refresh_ctx_block_tables``'s tensor statements."""
        n, cols = src.shape
        encoded = src.to(torch.int64).clone()
        encoded.sub_(block_offset)
        block_cols = torch.arange(1, cols + 1, dtype=torch.long, device=src.device)
        valid_cols = (encoded >= 0) * block_cols
        counts[:n].copy_(valid_cols.amax(dim=1))
        decoded = encoded.clamp_(min=0).div_(divisor, rounding_mode="floor")
        tables[:n].copy_(decoded.to(torch.int32))

    def is_valid(
        self,
        src: torch.Tensor,
        block_offset: int,
        divisor: int,
        tables: torch.Tensor,
        counts: torch.Tensor,
    ) -> None:
        if src.dim() != 2 or src.dtype != torch.int32 or src.stride(1) != 1:
            raise ValueError("src must be int32 [n, cols] with column stride 1")
        n, cols = src.shape
        if not 1 <= n <= MAX_REQUESTS:
            raise ValueError(f"1..{MAX_REQUESTS} requests, got {n}")
        if tables.dtype != torch.int32 or tables.dim() != 2 or not tables.is_contiguous():
            raise ValueError("tables must be contiguous int32 [slots, cols]")
        if tables.shape[0] < n or tables.shape[1] != cols:
            raise ValueError("tables must hold n rows of the source's columns")
        if counts.dtype != torch.int64 or counts.dim() != 1 or not counts.is_contiguous():
            raise ValueError("counts must be contiguous int64 [slots]")
        if counts.numel() < n:
            raise ValueError("counts must hold n entries")
        if block_offset < 0 or divisor < 1:
            raise ValueError("block_offset must be >= 0 and divisor >= 1")


ctx_block_tables_paged = CtxBlockTablesPaged()
