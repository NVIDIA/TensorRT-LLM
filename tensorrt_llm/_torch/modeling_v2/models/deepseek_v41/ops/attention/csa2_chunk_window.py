# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Window rows of a context chunk, in two launches.

A context chunk can be longer than the SWA window, so its own rows are attended from a step-local copy and the
rows before the chunk (the request's history) come from the SWA pool. The fused sparse attention reads one
row buffer through one index per window position; this op makes the two pieces of that:

- `old_rows`: the last `window - 1` history rows of every request, copied out of the pool (zeros where unmapped);
- `swa_idx`: for every query row and window slot, the row to read in `cat([step_rows, old_rows])` (the step's own
  row, a history row, or -1 for a position before the start, an unmapped one, or below the replay floor).

It replaces about forty small torch launches per layer (position arithmetic, page lookups, masks, `where`s).
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper

from .csa2_rows_io import csa2_row_addr

_ROW = 528  # bytes of a packed SWA row
_PAGE = 128  # rows per SWA pool page


@triton.jit
def _old_rows_kernel(
    POOL,
    TABLE,
    REQ_CS,
    OUT,
    nblk,
    table_stride,
    WINDOW: tl.constexpr,
    ROW: tl.constexpr,
    PAGE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    r = pid // (WINDOW - 1)
    j = pid % (WINDOW - 1)
    pos = tl.load(REQ_CS + r) - (WINDOW - 1) + j
    blk = pos // PAGE
    ok = (pos >= 0) & (blk < nblk)
    page = tl.load(TABLE + r * table_stride + tl.where(ok, blk, 0), mask=ok, other=-1)
    ok = ok & (page >= 0)
    src = (page.to(tl.int64) * PAGE + (pos - blk * PAGE)) * ROW
    offs = tl.arange(0, BLOCK)
    m = offs < ROW
    v = tl.load(POOL + src + offs, mask=m & ok, other=0)
    tl.store(OUT + pid.to(tl.int64) * ROW + offs, v, mask=m)


@triton.jit
def _window_idx_kernel(
    POS,
    RID,
    CS,
    FLOOR,
    TABLE,
    OUT,
    n,
    nblk,
    table_stride,
    HAS_FLOOR: tl.constexpr,
    WINDOW: tl.constexpr,
    PAGE: tl.constexpr,
):
    t = tl.program_id(0)
    j = tl.arange(0, WINDOW)
    p = tl.load(POS + t).to(tl.int64)
    r = tl.load(RID + t)
    cs = tl.load(CS + t).to(tl.int64)
    qpos = p - (WINDOW - 1) + j
    blk = qpos // PAGE
    ok = (qpos >= 0) & (blk < nblk)
    page = tl.load(TABLE + r * table_stride + tl.where(ok, blk, 0), mask=ok, other=-1)
    old_ok = ok & (page >= 0)
    old_slot = qpos - (cs - (WINDOW - 1))
    old_slot = tl.minimum(tl.maximum(old_slot, 0), WINDOW - 2)
    idx = tl.where(
        qpos >= cs, t - (p - qpos), tl.where(old_ok, n + r * (WINDOW - 1) + old_slot, -1)
    )
    idx = tl.where(qpos >= 0, idx, -1)
    if HAS_FLOOR:
        idx = tl.where(qpos >= tl.load(FLOOR + t).to(tl.int64), idx, -1)
    tl.store(OUT + t.to(tl.int64) * WINDOW + j, idx)


class Csa2ChunkWindow(OpWrapper):
    """`(old_rows [nreq * (window - 1), 528] uint8, swa_idx [n, window] int64)` of a context chunk.

    `pos` int32 `[n]` is the position of each query row, `rid` int64 `[n]` its page-table row and `cs` int32/int64
    `[n]` the first position of its request in the step. `req_cs` int64 `[nreq]` is `cs` per request (0 for a request
    without rows in the step). `swa_pool` is `[pages, 128, 528]` uint8 and `swa_table` the layer's int32
    `[nreq, blocks]` page table (-1 = unmapped). `floor` int `[n]` (optional) is the first window position a row may
    read; earlier slots are -1 (the truncated window of a bounded decoder replay).

    `old_rows[r * (window - 1) + k]` is the pool row of position `req_cs[r] - (window - 1) + k` (zeros when unmapped or
    negative). `swa_idx[i, j]` addresses `cat([step_rows, old_rows])` for the position `pos[i] - (window - 1) + j`: its
    step row `i - (window - 1 - j)` when that position is inside the step, else its history row when mapped, else -1.
    Every row has `pos >= cs` and `cs == req_cs[rid]` (the rows of a request are its chunk's tokens).
    """

    ARCHS = frozenset({Arch.SM_103})
    CELLS = (
        Cell(
            why="one request from scratch, chunk longer than the window: nothing before the chunk",
            spec=dict(reqs=[(0, 700)], window=128),
        ),
        Cell(
            why="two requests, histories across page edges (300, 0), the second one fresh",
            spec=dict(reqs=[(300, 520), (0, 200)], window=128),
        ),
        Cell(
            why="a long history, a short one and a one-row request with an unmapped history page",
            spec=dict(reqs=[(5000, 300), (130, 40), (0, 1)], window=128, unmapped=0.3),
        ),
        Cell(
            why="bounded decoder replay: window slots below the floor are masked (history 800, floor 740)",
            spec=dict(reqs=[(800, 100), (300, 60)], window=128, floor=[740, 290]),
        ),
    )
    note = "Pure index arithmetic and a row copy: every cell is compared bit for bit with the reference."

    def __call__(
        self,
        pos: torch.Tensor,
        rid: torch.Tensor,
        cs: torch.Tensor,
        req_cs: torch.Tensor,
        swa_pool: torch.Tensor,
        swa_table: torch.Tensor,
        window: int,
        floor: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        n, nreq, dev = pos.shape[0], swa_table.shape[0], pos.device
        old = torch.empty(nreq * (window - 1), _ROW, dtype=torch.uint8, device=dev)
        idx = torch.empty(n, window, dtype=torch.int64, device=dev)
        pool = swa_pool.view(-1, _ROW)
        with torch.cuda.device(dev):
            _old_rows_kernel[(nreq * (window - 1),)](
                pool,
                swa_table,
                req_cs,
                old,
                swa_table.shape[1],
                swa_table.stride(0),
                WINDOW=window,
                ROW=_ROW,
                PAGE=_PAGE,
                BLOCK=1024,
            )
            _window_idx_kernel[(n,)](
                pos,
                rid,
                cs,
                floor if floor is not None else pos,
                swa_table,
                idx,
                n,
                swa_table.shape[1],
                swa_table.stride(0),
                HAS_FLOOR=floor is not None,
                WINDOW=window,
                PAGE=_PAGE,
            )
        return old, idx

    def reference(
        self,
        pos: torch.Tensor,
        rid: torch.Tensor,
        cs: torch.Tensor,
        req_cs: torch.Tensor,
        swa_pool: torch.Tensor,
        swa_table: torch.Tensor,
        window: int,
        floor: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """The plain-torch index arithmetic (page lookups, validity masks, `where`s) this op fuses."""
        dev, n, nreq = pos.device, pos.shape[0], swa_table.shape[0]
        p = pos.long().unsqueeze(-1)
        qpos = p - (window - 1) + torch.arange(window, device=dev)
        rid2 = rid.long().unsqueeze(-1).expand_as(qpos)
        oldpos = req_cs.long().unsqueeze(-1) - (window - 1) + torch.arange(window - 1, device=dev)
        oaddr = csa2_row_addr(
            swa_table, torch.arange(nreq, device=dev).unsqueeze(-1).expand_as(oldpos), oldpos, _PAGE
        )
        flat = swa_pool.view(-1, _ROW)
        old = flat[oaddr.clamp(min=0).reshape(-1)].masked_fill((oaddr < 0).reshape(-1, 1), 0)
        cs_t = cs.long().unsqueeze(-1)
        t = torch.arange(n, device=dev).unsqueeze(-1)
        old_off = rid2 * (window - 1) + (qpos - (cs_t - (window - 1))).clamp(0, window - 2)
        old_ok = (oaddr >= 0).view(-1)[old_off.reshape(-1)].view_as(old_off)
        idx = torch.where(qpos >= cs_t, t - (p - qpos), torch.where(old_ok, n + old_off, -1))
        idx = torch.where(qpos >= 0, idx, -1)
        if floor is not None:
            idx = torch.where(qpos >= floor.long().unsqueeze(-1), idx, -1)
        return old.contiguous(), idx.contiguous()

    def is_valid(
        self,
        pos: torch.Tensor,
        rid: torch.Tensor,
        cs: torch.Tensor,
        req_cs: torch.Tensor,
        swa_pool: torch.Tensor,
        swa_table: torch.Tensor,
        window: int,
        floor: Optional[torch.Tensor] = None,
    ) -> None:
        assert pos.dtype == torch.int32 and pos.is_contiguous(), "pos is contiguous int32 [n]"
        assert rid.dtype == torch.int64 and rid.is_contiguous() and rid.shape == pos.shape, (
            "rid is int64 [n]"
        )
        assert (
            cs.dtype in (torch.int32, torch.int64) and cs.is_contiguous() and cs.shape == pos.shape
        ), "cs is [n]"
        assert (
            req_cs.dtype == torch.int64
            and req_cs.shape == (swa_table.shape[0],)
            and req_cs.is_contiguous()
        ), "req_cs is contiguous int64 [requests] (the kernel reads it with unit stride)"
        assert (
            swa_table.dtype == torch.int32 and swa_table.dim() == 2 and swa_table.stride(1) == 1
        ), "table is int32"
        assert (
            swa_pool.dtype == torch.uint8
            and swa_pool.is_contiguous()
            and swa_pool.shape[1:] == (_PAGE, _ROW)
        ), "the SWA pool is contiguous uint8 [pages, 128, 528]"
        assert window == 128, (
            "the window is the pool's page size (index arithmetic is specialised to it)"
        )
        if floor is not None:
            assert floor.shape == pos.shape and floor.is_contiguous(), "floor is [n]"


csa2_chunk_window = Csa2ChunkWindow()
