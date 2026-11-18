# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A step's QSA index-plane writes on compact index pages (`index_plane_layout`).

On plain pages the step stores every row's raw key and coordinates (`qsa_paged_index_store`),
then compresses each group it completes by reading the group's four raw keys back from the
plane (`qsa_prefill_compress`). On compact pages a page has only two raw-key rows a position
modulo 4 (one per group parity), so within one step a long chunk's later groups would overwrite
the raw keys an earlier group still has to pool. Here, in two launches:

1. `_compress_kernel`, one program a row that ends a group: the group's members come from this
   step's own rows (`token_k`, the coordinates of `coords`) where the step holds them, else from
   the plane (raw keys an earlier step stored) and the position plane; the compressed key is the
   family compression's, the same operations on the same bf16 values (bitwise), stored in the
   group's compressed-key row.
2. `_store_kernel`, one program a row: every row's coordinates into the position plane (one row
   a token, as on plain pages), and the raw key of a row that does not end a group where a later
   step may pool it: its group does not end in this step, or the request has at most
   `compress_ratio + 1` rows in the step (a decode / verify / draft-extend window, whose rows a
   speculative rollback may reopen). Those rows map to distinct plane rows, so no two programs
   write one row, and the compress launch has read every raw key before any is overwritten.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from tensorrt_llm._torch.attention.backends.sparse.qsa.index_plane_layout import index_plane_row
from tensorrt_llm._torch.attention.backends.sparse.qsa.kernels import _qsa_gemma_norm_rope


@triton.jit(do_not_specialize=["num_rows", "num_requests", "table_width", "num_pages"])
def _compress_kernel(
    token_k,
    s_k_row,
    s_k_dim,
    coords,
    s_c_row,
    s_c_axis,
    index_cache,
    s_ic_page,
    s_ic_row,
    s_ic_dim,
    position_cache,
    s_pc_page,
    s_pc_slot,
    s_pc_axis,
    request_indices,
    logical_positions,
    block_table,
    s_bt_req,
    s_bt_col,
    num_rows,
    num_requests,
    table_width,
    num_pages,
    k_norm_weight,
    cos_sin,
    cos_sin_stride,
    eps,
    HEAD_DIM: tl.constexpr,
    ROTARY_PAIRS: tl.constexpr,
    TPB: tl.constexpr,
    CR: tl.constexpr,
    ROWS: tl.constexpr,
    IS_MROPE: tl.constexpr,
    MROPE_H: tl.constexpr,
    MROPE_W: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    logical = tl.load(logical_positions + row).to(tl.int64)
    request = tl.load(request_indices + row).to(tl.int64)
    live = (
        (logical >= CR - 1)
        & ((logical + 1) % CR == 0)
        & (request >= 0)
        & (request < num_requests)
        & (logical // TPB < table_width)
    )
    if live:
        dims = tl.arange(0, HEAD_DIM)
        offsets = tl.arange(0, CR)
        members = logical - (CR - 1) + offsets
        member_rows = row - (CR - 1) + offsets
        maybe = member_rows >= 0
        member_requests = tl.load(request_indices + member_rows, mask=maybe, other=-1)
        member_logical = tl.load(logical_positions + member_rows, mask=maybe, other=-1)
        in_step = (
            maybe
            & (member_requests.to(tl.int64) == request)
            & (member_logical.to(tl.int64) == members)
        )
        member_pages = tl.load(block_table + request * s_bt_req + (members // TPB) * s_bt_col).to(
            tl.int64
        )
        cached = tl.load(
            index_cache
            + member_pages[:, None] * s_ic_page
            + index_plane_row(members, TPB, CR, ROWS)[:, None] * s_ic_row
            + dims[None, :] * s_ic_dim,
            mask=~in_step[:, None],
            other=0.0,
        )
        stepped = tl.load(
            token_k + member_rows[:, None] * s_k_row + dims[None, :] * s_k_dim,
            mask=in_step[:, None],
            other=0.0,
        )
        group = tl.where(in_step[:, None], stepped.to(index_cache.dtype.element_ty), cached)
        pooled = (tl.sum(group.to(tl.float32), axis=0) / CR).to(tl.bfloat16)

        first = logical - (CR - 1)
        first_in_step = tl.sum(tl.where(offsets == 0, in_step, False).to(tl.int32)) > 0
        first_row = row - (CR - 1)
        if first_in_step:
            first_pos_t = tl.load(coords + first_row * s_c_row).to(tl.int32)
            first_pos_h = tl.load(coords + first_row * s_c_row + s_c_axis).to(tl.int32)
            first_pos_w = tl.load(coords + first_row * s_c_row + 2 * s_c_axis).to(tl.int32)
        else:
            first_page = tl.load(block_table + request * s_bt_req + (first // TPB) * s_bt_col).to(
                tl.int64
            )
            first_base = position_cache + first_page * s_pc_page + (first % TPB) * s_pc_slot
            first_pos_t = tl.load(first_base)
            first_pos_h = tl.load(first_base + s_pc_axis)
            first_pos_w = tl.load(first_base + 2 * s_pc_axis)
        compressed = tl.reshape(
            _qsa_gemma_norm_rope(
                pooled[None, :],
                first_pos_t,
                first_pos_h,
                first_pos_w,
                cos_sin,
                cos_sin_stride,
                k_norm_weight,
                eps,
                NUM_ROWS=1,
                HEAD_DIM=HEAD_DIM,
                ROTARY_PAIRS=ROTARY_PAIRS,
                IS_MROPE=IS_MROPE,
                MROPE_H=MROPE_H,
                MROPE_W=MROPE_W,
            ),
            (HEAD_DIM,),
        )
        anchor_page = tl.load(block_table + request * s_bt_req + (logical // TPB) * s_bt_col).to(
            tl.int64
        )
        if (anchor_page >= 0) & (anchor_page < num_pages):
            tl.store(
                index_cache
                + anchor_page * s_ic_page
                + index_plane_row(logical, TPB, CR, ROWS) * s_ic_row
                + dims * s_ic_dim,
                compressed,
            )


@triton.jit(do_not_specialize=["num_rows", "num_requests", "table_width", "num_pages"])
def _store_kernel(
    token_k,
    s_k_row,
    s_k_dim,
    coords,
    s_c_row,
    s_c_axis,
    index_cache,
    s_ic_page,
    s_ic_row,
    s_ic_dim,
    position_cache,
    s_pc_page,
    s_pc_slot,
    s_pc_axis,
    request_indices,
    logical_positions,
    block_table,
    s_bt_req,
    s_bt_col,
    num_rows,
    num_requests,
    table_width,
    num_pages,
    HEAD_DIM: tl.constexpr,
    TPB: tl.constexpr,
    CR: tl.constexpr,
    ROWS: tl.constexpr,
    WINDOW: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    logical = tl.load(logical_positions + row).to(tl.int64)
    request = tl.load(request_indices + row).to(tl.int64)
    column = logical // TPB
    live = (request >= 0) & (request < num_requests) & (logical >= 0) & (column < table_width)
    page = tl.load(block_table + request * s_bt_req + column * s_bt_col, mask=live, other=-1).to(
        tl.int64
    )
    live = live & (page >= 0) & (page < num_pages)
    if live:
        axes = tl.arange(0, 4)
        coordinate = tl.load(coords + row * s_c_row + axes * s_c_axis, mask=axes < 3, other=0)
        tl.store(
            position_cache + page * s_pc_page + (logical % TPB) * s_pc_slot + axes * s_pc_axis,
            coordinate.to(position_cache.dtype.element_ty),
            mask=axes < 3,
        )
        if logical % CR != CR - 1:
            # This request's rows within WINDOW of this one (its rows are consecutive, at
            # consecutive positions): a run longer than CR + 1 rows is not a short window.
            k = tl.arange(0, WINDOW) + 1
            after_rows = row + k
            after_ok = after_rows < num_rows
            after = (
                after_ok
                & (tl.load(request_indices + after_rows, mask=after_ok, other=-1).to(tl.int64)
                   == request)
                & (tl.load(logical_positions + after_rows, mask=after_ok, other=-1).to(tl.int64)
                   == logical + k)
            )  # fmt: skip
            before_rows = row - k
            before_ok = before_rows >= 0
            before = (
                before_ok
                & (tl.load(request_indices + before_rows, mask=before_ok, other=-1).to(tl.int64)
                   == request)
                & (tl.load(logical_positions + before_rows, mask=before_ok, other=-1).to(tl.int64)
                   == logical - k)
            )  # fmt: skip
            run = 1 + tl.sum(after.to(tl.int32)) + tl.sum(before.to(tl.int32))
            short = run <= CR + 1
            # Whether this step also holds the row that ends this row's group.
            to_end = CR - 1 - logical % CR
            ends = tl.sum(tl.where(k == to_end, after, False).to(tl.int32)) > 0
            if short | ~ends:
                dims = tl.arange(0, HEAD_DIM)
                key = tl.load(token_k + row * s_k_row + dims * s_k_dim)
                tl.store(
                    index_cache
                    + page * s_ic_page
                    + index_plane_row(logical, TPB, CR, ROWS) * s_ic_row
                    + dims * s_ic_dim,
                    key.to(index_cache.dtype.element_ty),
                )


def qsa_index_compact_plane(
    token_k: torch.Tensor,
    coords: torch.Tensor,
    index_cache: torch.Tensor,
    position_cache: torch.Tensor,
    request_indices: torch.Tensor,
    logical_positions: torch.Tensor,
    block_table: torch.Tensor,
    tokens_per_block: int,
    compress_ratio: int,
    k_norm_weight: torch.Tensor,
    cos_sin: torch.Tensor,
    eps: float,
    mrope_section: tuple[int, int, int] | None,
) -> None:
    """The step's index-plane writes on compact pages: `qsa_paged_index_store` followed by
    `qsa_prefill_compress` on plain pages, to the bit (see the module docstring).

    `token_k` `[rows, 1, head_dim]` (or `[rows, head_dim]`) raw keys, `coords` `[rows, >= 3]`,
    `index_cache` `[pages, rows_a_page, 1, head_dim]` bf16 on the compact layout,
    `position_cache` `[pages, tokens_per_block, 3]` int32."""
    rows = logical_positions.shape[0]
    if rows == 0:
        return
    head_dim = index_cache.shape[-1]
    assert index_cache.dtype == torch.bfloat16 and index_cache.shape[2] == 1
    assert position_cache.dtype == torch.int32
    assert block_table.dtype == torch.int32
    rotary_pairs = cos_sin.shape[1] // 2
    section = mrope_section if mrope_section is not None else (0, 0, 0)
    k2 = token_k.reshape(token_k.shape[0], -1) if token_k.dim() == 3 else token_k
    assert k2.shape[1] == head_dim, "one index KV head"
    common = (
        k2,
        k2.stride(0),
        k2.stride(1),
        coords,
        coords.stride(0),
        coords.stride(1),
        index_cache,
        index_cache.stride(0),
        index_cache.stride(1),
        index_cache.stride(3),
        position_cache,
        position_cache.stride(0),
        position_cache.stride(1),
        position_cache.stride(2),
        request_indices,
        logical_positions,
        block_table,
        block_table.stride(0),
        block_table.stride(1),
        rows,
        block_table.shape[0],
        block_table.shape[1],
        index_cache.shape[0],
    )
    _compress_kernel[(rows,)](
        *common,
        k_norm_weight,
        cos_sin,
        cos_sin.stride(0),
        eps,
        HEAD_DIM=head_dim,
        ROTARY_PAIRS=rotary_pairs,
        TPB=tokens_per_block,
        CR=compress_ratio,
        ROWS=index_cache.shape[1],
        IS_MROPE=mrope_section is not None,
        MROPE_H=section[1],
        MROPE_W=section[2],
        num_warps=1,
    )
    _store_kernel[(rows,)](
        *common,
        HEAD_DIM=head_dim,
        TPB=tokens_per_block,
        CR=compress_ratio,
        ROWS=index_cache.shape[1],
        WINDOW=triton.next_power_of_2(compress_ratio + 1),
        num_warps=1,
    )
