# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""QSA compressed-block index scores for a verify step's rows: one key gather per request.

`qsa_paged_index_scores` scores each row on its own, so a verify step's `P` rows of one
request gather that request's compressed keys `P` times. Here one program serves a
request's `P` rows: it gathers a tile of keys once and multiplies it by the `P` rows'
index queries packed side by side (`P` x 4 heads <= 16 columns, the width the row-wise
kernel pads its 4 heads to), then reduces each row's four head columns.

Bitwise the row-wise kernel's scores: the same tile shapes, warps and dot (a score column
of an MMA depends on its own query column alone), each row's relu'd head sum over the
same thread's pair of columns and the same shuffles, and the same scale.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

from .qsa_paged_index_scores import qsa_paged_index_scores as _row_wise

#: The row-wise kernel's head padding: `P * heads` query columns fit in it.
_BLOCK_H = 16

#: This checkpoint's index geometry and the NVFP4 trunk's index page (12 index buffers of
#: 32 x 128, then the 192-element position page: page stride 49344).
_CKPT = dict(heads=4, head_dim=128, tokens_per_block=32, compress_ratio=4, width=324)


@triton.jit(do_not_specialize=["rows"])
def _qsa_index_scores_verify_kernel(
    q,
    index_cache,
    block_table,
    query_positions,
    request_indices,
    output,
    score_scale,
    rows,
    q_stride_row,
    q_stride_head,
    q_stride_dim,
    cache_stride_page,
    cache_stride_token,
    cache_stride_dim,
    bt_stride_request,
    bt_stride_page,
    out_stride_row,
    P: tl.constexpr,
    IH: tl.constexpr,
    ID: tl.constexpr,
    BLOCK_D: tl.constexpr,
    TPB: tl.constexpr,
    CR: tl.constexpr,
    MAX_BLOCKS: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_N: tl.constexpr,
    TILES_PER_PROGRAM: tl.constexpr,
    USE_PDL: tl.constexpr,
):
    if USE_PDL:
        tl.extra.cuda.gdc_wait()
        tl.extra.cuda.gdc_launch_dependents()
    first_row = tl.program_id(0) * P
    request = tl.load(request_indices + first_row).to(tl.int64)
    # The request's last row sees the most blocks (its rows sit at consecutive positions).
    last_visible = (tl.load(query_positions + first_row + P - 1) + 1) // CR
    first_tile = tl.program_id(1) * TILES_PER_PROGRAM
    if first_tile * BLOCK_N >= last_visible:
        return
    last_tile = tl.minimum(first_tile + TILES_PER_PROGRAM, tl.cdiv(last_visible, BLOCK_N))

    # Query column c is head c % IH of the group's row c // IH; columns past P rows are zero.
    columns = tl.arange(0, BLOCK_H)
    column_row = columns // IH
    dim_offsets = tl.arange(0, BLOCK_D)
    query_values = tl.load(
        q
        + (first_row + column_row).to(tl.int64)[None, :] * q_stride_row
        + (columns % IH)[None, :] * q_stride_head
        + dim_offsets[:, None] * q_stride_dim,
        mask=(dim_offsets[:, None] < ID) & (column_row < P)[None, :],
        other=0.0,
    )
    block_offsets = tl.arange(0, BLOCK_N)
    for tile in tl.range(first_tile, last_tile, num_stages=2):
        block_columns = tile * BLOCK_N + block_offsets
        valid_blocks = (block_columns < MAX_BLOCKS) & (block_columns < last_visible)
        anchor_positions = block_columns * CR + CR - 1
        physical_pages = tl.load(
            block_table + request * bt_stride_request + (anchor_positions // TPB) * bt_stride_page,
            mask=valid_blocks,
            other=0,
        ).to(tl.int64)
        keys = tl.load(
            index_cache
            + physical_pages[:, None] * cache_stride_page
            + (anchor_positions % TPB)[:, None] * cache_stride_token
            + dim_offsets[None, :] * cache_stride_dim,
            mask=valid_blocks[:, None] & (dim_offsets[None, :] < ID),
            other=0.0,
            eviction_policy="evict_first",
        )
        per_head_scores = tl.maximum(tl.dot(keys, query_values, out_dtype=tl.float32), 0.0)
        for r in tl.static_range(P):
            row = first_row + r
            visible = (tl.load(query_positions + row) + 1) // CR
            scores = tl.sum(tl.where((column_row == r)[None, :], per_head_scores, 0.0), axis=1)
            tl.store(
                output + row.to(tl.int64) * out_stride_row + block_columns,
                scores * score_scale,
                mask=valid_blocks & (block_columns < visible),
            )


def _plan(rows: int):
    """The row-wise kernel's tile plan for this many rows: (BLOCK_N, tiles a program)."""
    return (64 if rows > 4 else 8), (8 if rows > 256 else 1)


class QsaPagedIndexScoresVerify(OpWrapper):
    """`qsa_paged_index_scores` (`only_visible_blocks=True`, `context_rows=False`) for rows that
    come `rows_per_request` consecutive rows a request at consecutive positions -- an MTP
    verify step's generation rows. Returns a fresh fp32 `[rows, columns]`: each row's
    visible columns bitwise the row-wise entry's, the rest not written.

    One Triton launch over `(requests, column tiles)`; each program gathers a tile of the
    request's compressed keys once for all its rows.

    Fusion boundary: as `qsa_paged_index_scores`.
    """

    ARCHS = frozenset({Arch.SM_103})

    CELLS: tuple[Cell, ...] = (
        Cell(
            why="MTP3 verify, 1 request x 4 rows (the row-wise plan's 8-block tiles), a group "
            "end mid-request, NVFP4 trunk page",
            spec=dict(starts=(8697,), per=4, nvfp4=True),
        ),
        Cell(
            why="MTP3 verify, 32 requests x 4 rows (64-block tiles), starts at every residue "
            "mod 4, NVFP4 trunk page",
            spec=dict(starts=tuple(2000 + 197 * r for r in range(32)), per=4, nvfp4=True),
        ),
        Cell(
            why="MTP3 verify, 256 requests x 4 rows (eight tiles a program), short and long "
            "contexts, plain index plane",
            spec=dict(starts=tuple(3 + (577 * r) % 10000 for r in range(256)), per=4),
        ),
        Cell(
            why="MTP1 verify, 8 requests x 2 rows",
            spec=dict(starts=tuple(4093 + 701 * r for r in range(8)), per=2, nvfp4=True),
        ),
    )

    note = """
    Metadata consumed: as `qsa_paged_index_scores` (`qsa_block_table`,
    `qsa_logical_positions`, `qsa_req_idx_per_token`).

    Preconditions: `rows % rows_per_request == 0`; each group of `rows_per_request`
    consecutive rows is one request's, at consecutive increasing positions (the verify
    step's layout); `rows_per_request * heads <= 16`; the row-wise entry's preconditions
    (valid pages up to each row's causal bound). Not checked on the device: a group that
    breaks the layout is scored against its first row's request and its last row's bound.

    Numerics: bitwise the row-wise entry on every visible column (the cells compare them
    exactly). The tile plan (`BLOCK_N` 8 up to 4 rows else 64, eight tiles a program
    past 256 rows, two warps) is the row-wise entry's at the same row count.

    Compiled variants: one per `rows_per_request`, tile plan, table width and PDL; the row
    count is not specialized on.
    """

    def __call__(
        self,
        q: torch.Tensor,
        index_cache: torch.Tensor,
        block_table: torch.Tensor,
        query_positions: torch.Tensor,
        request_indices: torch.Tensor,
        tokens_per_block: int,
        compress_ratio: int,
        rows_per_request: int,
        use_pdl: bool = False,
    ) -> torch.Tensor:
        rows, heads, head_dim = q.shape
        columns = block_table.shape[1] * tokens_per_block // compress_ratio
        output = torch.empty((rows, columns), dtype=torch.float32, device=q.device)
        if rows == 0:
            return output
        block_n, tiles = _plan(rows)
        _qsa_index_scores_verify_kernel[
            (rows // rows_per_request, triton.cdiv(columns, block_n * tiles))
        ](
            q,
            index_cache,
            block_table,
            query_positions,
            request_indices,
            output,
            head_dim**-0.5,
            rows,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            index_cache.stride(0),
            index_cache.stride(1),
            index_cache.stride(3),
            block_table.stride(0),
            block_table.stride(1),
            output.stride(0),
            P=rows_per_request,
            IH=heads,
            ID=head_dim,
            BLOCK_D=max(16, triton.next_power_of_2(head_dim)),
            TPB=tokens_per_block,
            CR=compress_ratio,
            MAX_BLOCKS=columns,
            BLOCK_H=_BLOCK_H,
            BLOCK_N=block_n,
            TILES_PER_PROGRAM=tiles,
            USE_PDL=use_pdl,
            num_warps=2,
            launch_pdl=use_pdl,
        )
        return output

    def reference(
        self,
        q: torch.Tensor,
        index_cache: torch.Tensor,
        block_table: torch.Tensor,
        query_positions: torch.Tensor,
        request_indices: torch.Tensor,
        tokens_per_block: int,
        compress_ratio: int,
        rows_per_request: int,
        use_pdl: bool = False,
    ) -> torch.Tensor:
        """The row-wise entry on the same rows (its visible columns are the contract)."""
        return _row_wise(
            q=q,
            index_cache=index_cache,
            block_table=block_table,
            query_positions=query_positions,
            request_indices=request_indices,
            tokens_per_block=tokens_per_block,
            compress_ratio=compress_ratio,
            only_visible_blocks=True,
            context_rows=False,
        )

    def is_valid(
        self,
        q,
        index_cache,
        block_table,
        query_positions,
        request_indices,
        tokens_per_block,
        compress_ratio,
        rows_per_request,
        use_pdl=False,
    ) -> None:
        rows, heads, _ = q.shape
        assert rows_per_request >= 1 and rows % rows_per_request == 0, "whole request groups"
        assert rows_per_request * heads <= _BLOCK_H, "rows_per_request x heads <= 16"
        assert index_cache.ndim == 4 and index_cache.shape[2] == 1, "one index KV head"
        assert q.dtype == index_cache.dtype, "matching q and index dtypes"
        assert block_table.dtype == torch.int32 and block_table.ndim == 2, "int32 table"
        assert query_positions.shape == (rows,) and request_indices.shape == (rows,), (
            "one position and request a row"
        )


qsa_paged_index_scores_verify = QsaPagedIndexScoresVerify()
