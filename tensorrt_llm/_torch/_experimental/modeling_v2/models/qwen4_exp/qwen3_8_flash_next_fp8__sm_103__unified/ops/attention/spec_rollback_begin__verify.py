# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A captured verify step's QSA rollback staging in one launch: every generation row's QSA slot
(`block_table[req, pos // tpb]`, `pos % tpb`), the pre-step contents of that slot in every
attention layer's index-K plane, and in the shared position plane.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

#: One index-K row: 128 bf16 a token (one 128-wide head).
ROW = 128
#: A position-plane row: the token's three mRoPE coordinates, int32.
POS_W = 3


# Program (l, r) resolves row r's slot and copies its row of plane l; the program past the last
# index plane copies the position triple and writes the slot.
@triton.jit(
    do_not_specialize=["rows"],
    do_not_specialize_on_alignment=["req_idx", "logical"],
)
def _begin_kernel(
    req_idx,
    logical,
    block_table,
    bt_stride,
    ptrs,
    page_strides,
    position,
    pos_s0,
    pos_s1,
    pages_out,
    within_out,
    index_out,
    position_out,
    rows,
    TPB: tl.constexpr,
    ROW_: tl.constexpr,
    PLANES: tl.constexpr,
    POS_W_: tl.constexpr,
):
    plane = tl.program_id(0)
    r = tl.program_id(1)
    pos = tl.load(logical + r).to(tl.int64)
    req = tl.load(req_idx + r).to(tl.int64)
    page = tl.load(block_table + req * bt_stride + pos // TPB).to(tl.int64)
    tok = pos % TPB
    if plane < PLANES:
        base = tl.load(ptrs + plane).to(tl.pointer_type(tl.bfloat16))
        stride = tl.load(page_strides + plane)
        offs = tl.arange(0, ROW_)
        vals = tl.load(base + page * stride + tok * ROW_ + offs)
        tl.store(index_out + (plane.to(tl.int64) * rows + r) * ROW_ + offs, vals)
    else:
        w = tl.arange(0, 4)
        wm = w < POS_W_
        v = tl.load(position + page * pos_s0 + tok * pos_s1 + w, mask=wm)
        tl.store(position_out + r * POS_W_ + w, v, mask=wm)
        tl.store(pages_out + r, page)
        tl.store(within_out + r, tok)


def _cells():
    return tuple(
        Cell(
            why=f"tep4 MTP verify, {n} request(s) x {p} rows over 12 planes",
            spec=dict(requests=n, p=p, planes=12, capture=capture),
        )
        for n, p, capture in (
            (1, 3, False),
            (4, 3, True),
            (4, 4, False),
        )
    )


class SpecRollbackBegin(OpWrapper):
    """`(pages, within, index_all, position_old) = spec_rollback_begin(req_idx, logical,
    block_table, tokens_per_block, ptrs, page_strides, position)` for a verify step's rows.

    - `req_idx` `[R]` int32 / int64 and `logical` `[R]` int64: the generation rows' request
      index (into `block_table`) and logical position (`qsa_req_idx_per_token`,
      `qsa_logical_positions`);
    - `block_table` `[S, max_pages]` int32 (unit column stride), `tokens_per_block` its page size;
    - `ptrs` / `page_strides`: `plane_table(planes)` of the attention layers' index-K planes;
    - `position` `[pages, tpb, 3]` int32 (unit innermost stride): the shared position plane.

    Returns `pages` / `within` `[R]` int64 (`block_table[req, pos // tpb]`, `pos % tpb`),
    `index_all` `[L, R, 128]` bf16 (every row's slot of every plane, plane order -- what
    `qsa_rollback_gather` returns) and `position_old` `[R, 3]` int32 (`position[pages, within]`).
    Nothing but the outputs is written. Run it before any layer of the step writes a plane.
    """

    ARCHS = frozenset({Arch.SM_103})
    CELLS: tuple[Cell, ...] = _cells()

    note = """
    Source: one Triton kernel (`_begin_kernel`), JIT-compiled per (tokens_per_block, plane
    count); grid (planes + 1, R). Copies and integer arithmetic only: bitwise the composed torch
    path.

    Preconditions: the dtypes, ranks and strides `is_valid` checks; every slot inside its pool
    (positions non-negative and inside the table) is the caller's.
    """

    def __call__(
        self, req_idx, logical, block_table, tokens_per_block: int, ptrs, page_strides, position
    ):
        rows = logical.shape[0]
        planes = ptrs.shape[0]
        dev = logical.device
        pages = torch.empty([rows], dtype=torch.int64, device=dev)
        within = torch.empty([rows], dtype=torch.int64, device=dev)
        index_all = torch.empty([planes, rows, ROW], dtype=torch.bfloat16, device=dev)
        position_old = torch.empty([rows, POS_W], dtype=torch.int32, device=dev)
        if rows:
            _begin_kernel[(planes + 1, rows)](
                req_idx,
                logical,
                block_table,
                block_table.stride(0),
                ptrs,
                page_strides,
                position,
                position.stride(0),
                position.stride(1),
                pages,
                within,
                index_all,
                position_old,
                rows,
                TPB=tokens_per_block,
                ROW_=ROW,
                PLANES=planes,
                POS_W_=POS_W,
                num_warps=1,
            )
        return pages, within, index_all, position_old

    def reference(self, req_idx, logical, block_table, tokens_per_block: int, planes, position):
        """The composed torch path; takes the planes themselves where `__call__` takes their
        `plane_table` (torch cannot dereference raw addresses)."""
        positions = logical.to(torch.long)
        pages = block_table[req_idx.to(torch.long), positions // tokens_per_block].to(torch.long)
        within = positions % tokens_per_block
        index_all = torch.stack([t[pages, within].reshape(-1, ROW) for t in planes])
        return pages, within, index_all, position[pages, within]

    def is_valid(
        self, req_idx, logical, block_table, tokens_per_block: int, ptrs, page_strides, position
    ) -> None:
        rows = logical.shape[0]
        assert logical.dtype == torch.int64 and logical.dim() == 1 and rows >= 1, (
            logical.dtype,
            logical.shape,
        )
        assert req_idx.dtype in (torch.int32, torch.int64) and tuple(req_idx.shape) == (rows,), (
            req_idx.dtype,
            req_idx.shape,
        )
        assert (
            block_table.dtype == torch.int32
            and block_table.dim() == 2
            and block_table.stride(1) == 1
        ), (
            block_table.dtype,
            block_table.stride(),
        )
        assert tokens_per_block > 0
        assert (
            ptrs.dtype == page_strides.dtype == torch.int64 and ptrs.shape == page_strides.shape
        ), "ptrs / page_strides must be plane_table's int64 tables"
        assert (
            position.dtype == torch.int32
            and position.dim() == 3
            and position.shape[2] == POS_W
            and position.stride(2) == 1
        ), (position.dtype, position.shape, position.stride())


spec_rollback_begin = SpecRollbackBegin()
