# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""QSA compressed-block index scoring, read straight out of the paged index-K side cache."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.ops.jit_variants import (
    warm_variants,
)

# Approved exception: QSA's selection path is Triton called as Python, with no
# torch.ops boundary. The engine's own sparse hook (`select_qsa_paged_tokens`
# in `_torch/attention/backends/sparse/qsa/module.py`) calls this same callable.
from tensorrt_llm._torch.attention.backends.sparse.qsa import kernels as _qsa_kernels
from tensorrt_llm._torch.attention.backends.sparse.qsa.kernels import (
    triton_qsa_paged_index_scores as _triton_qsa_paged_index_scores,
)
from tensorrt_llm._torch.modules.fla.op import gdc_wait_with_memory_clobber

_INTEGER_DTYPES = (torch.int32, torch.int64)

#: At or below this many rows a call is decode-sized: the engine's own threshold
#: between its one-tile-a-program and its deeper-tile launches.
_DECODE_ROW_LIMIT = 256
#: Dot columns of one visible-split program: whole query rows of `heads` columns each
#: (`tl.dot` needs at least 16), so the rows of one request share each key gather.
_SPLIT_DOT_WIDTH = 16
#: Score columns a tile gathers keys for.
_SPLIT_BLOCK_N = 64
#: Programs a launch holds per SM, empty request slots included. One row tile (one
#: request's verify rows: a chain of dependent loads) wants a tile a program, up to the
#: 1024 tiles of a 262143-position window; wider batches stream several requests at once.
_SPLIT_PROGRAMS_PER_SM = 16
_SPLIT_PROGRAMS_PER_SM_ONE_TILE = 32
#: With `wide_split`, from this many rows four times as many programs, so the longest windows'
#: chains of column tiles do not set the launch's time. The scores do not depend on the split.
_SPLIT_WIDE_MIN_ROWS = 32
_SPLIT_PROGRAMS_PER_SM_WIDE = 64
_SPLIT_NUM_WARPS = 2
_SM_COUNT: dict = {}


# Metadata views can begin at any row of a mixed context/decode step. Their
# alignment does not change these scalar loads or the score computation.
@triton.jit(
    do_not_specialize=["num_rows", "split"],
    do_not_specialize_on_alignment=["query_positions", "request_indices"],
)
def _qsa_index_scores_visible_split_kernel(
    q,
    index_cache,
    block_table,
    query_positions,
    request_indices,
    output,
    score_scale,
    num_rows,
    split,
    q_stride_row: tl.constexpr,
    q_stride_head: tl.constexpr,
    q_stride_dim: tl.constexpr,
    cache_stride_page: tl.constexpr,
    cache_stride_token: tl.constexpr,
    cache_stride_dim: tl.constexpr,
    block_table_stride_request: tl.constexpr,
    block_table_stride_page: tl.constexpr,
    output_stride_row: tl.constexpr,
    output_stride_block: tl.constexpr,
    NUM_INDEX_HEADS: tl.constexpr,
    INDEX_HEAD_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
    TOKENS_PER_BLOCK: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,
    MAX_COMPRESSED_BLOCKS: tl.constexpr,
    ROWS_PER_TILE: tl.constexpr,
    QUERY_TILE: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_N: tl.constexpr,
    NUM_STAGES: tl.constexpr,
    USE_PDL: tl.constexpr,
):
    """Visible-only scores of one request's rows among ROWS_PER_TILE consecutive rows (in
    QUERY_TILE dot lanes). Program (t, g, s) takes the g-th distinct request of row tile t,
    in row order, and walks its column tiles s, s + split, ... up to its visible bound: the
    grid is sized by the device, not by the table, a tile's keys are gathered once for every
    row of their request in the row tile, and different requests run side by side.
    """
    if USE_PDL:
        # The memory clobber keeps loads of the predecessor's outputs below the wait.
        gdc_wait_with_memory_clobber()
        tl.extra.cuda.gdc_launch_dependents()
    row_base = tl.program_id(0) * ROWS_PER_TILE
    group_slot = tl.program_id(1)
    first_tile = tl.program_id(2)

    # One dot column per (query row, index head) pair; q is read before the metadata
    # decides whether this program has work, so the two loads overlap.
    dot_columns = tl.arange(0, BLOCK_H)
    column_row = dot_columns // NUM_INDEX_HEADS
    column_head = dot_columns % NUM_INDEX_HEADS
    rows = row_base + column_row
    present = (column_row < ROWS_PER_TILE) & (rows < num_rows)
    dim_offsets = tl.arange(0, BLOCK_D)
    query_values = tl.load(
        q
        + rows[None, :] * q_stride_row
        + dim_offsets[:, None] * q_stride_dim
        + column_head[None, :] * q_stride_head,
        mask=(dim_offsets[:, None] < INDEX_HEAD_DIM) & present[None, :],
        other=0.0,
    )

    # Each tile row (lane): its request and visible bound.
    lanes = tl.arange(0, QUERY_TILE)
    lane_rows = row_base + lanes
    lane_present = (lanes < ROWS_PER_TILE) & (lane_rows < num_rows)
    lane_requests = tl.load(request_indices + lane_rows, mask=lane_present, other=-1).to(tl.int64)
    lane_positions = tl.load(query_positions + lane_rows, mask=lane_present, other=0)
    lane_visible = tl.where(
        lane_present, tl.minimum((lane_positions + 1) // COMPRESS_RATIO, MAX_COMPRESSED_BLOCKS), 0
    )
    # A lane leads its request's group when no earlier lane holds the same request; groups
    # are numbered by their leader's lane.
    earlier_same = (
        (lane_requests[:, None] == lane_requests[None, :])
        & (lanes[:, None] < lanes[None, :])
        & lane_present[:, None]
    )
    leads = lane_present & (tl.max(earlier_same.to(tl.int32), axis=0) == 0)
    rank = tl.cumsum(leads.to(tl.int32), axis=0) - 1
    request = tl.max(tl.where(leads & (rank == group_slot), lane_requests, -1))
    lane_group = lane_present & (lane_requests == request)
    group_visible = tl.max(tl.where(lane_group, lane_visible, 0))
    # No such group (request -1) leaves group_visible at 0.
    if first_tile * BLOCK_N >= group_visible:
        return

    column_offsets = tl.arange(0, BLOCK_N)
    group_tiles = tl.cdiv(group_visible, BLOCK_N)
    for tile in tl.range(first_tile, group_tiles, split, num_stages=NUM_STAGES):
        block_columns = tile * BLOCK_N + column_offsets
        valid_blocks = block_columns < group_visible
        anchor_positions = block_columns * COMPRESS_RATIO + COMPRESS_RATIO - 1
        logical_pages = anchor_positions // TOKENS_PER_BLOCK
        token_in_page = anchor_positions % TOKENS_PER_BLOCK
        physical_pages = tl.load(
            block_table
            + request * block_table_stride_request
            + logical_pages * block_table_stride_page,
            mask=valid_blocks,
            other=0,
        ).to(tl.int64)
        keys = tl.load(
            index_cache
            + physical_pages[:, None] * cache_stride_page
            + token_in_page[:, None] * cache_stride_token
            + dim_offsets[None, :] * cache_stride_dim,
            mask=valid_blocks[:, None] & (dim_offsets[None, :] < INDEX_HEAD_DIM),
            other=0.0,
            eviction_policy="evict_first",
        )
        positive_scores = tl.maximum(tl.dot(keys, query_values, out_dtype=tl.float32), 0.0)
        # Each row's heads are adjacent dot columns: sum them per row.
        scores = tl.sum(tl.reshape(positive_scores, [BLOCK_N, QUERY_TILE, NUM_INDEX_HEADS]), axis=2)
        tl.store(
            output
            + lane_rows[None, :] * output_stride_row
            + block_columns[:, None] * output_stride_block,
            scores * score_scale,
            mask=lane_group[None, :] & (block_columns[:, None] < lane_visible[None, :]),
        )


def _sm_count(device: torch.device) -> int:
    index = device.index if device.index is not None else torch.cuda.current_device()
    if index not in _SM_COUNT:
        _SM_COUNT[index] = torch.cuda.get_device_properties(index).multi_processor_count
    return _SM_COUNT[index]


def _rows_per_tile(rows: int, query_tile: int) -> int:
    """Rows a program's dot lanes hold: the whole batch when it fits one dot tile, else a
    full tile.
    """
    return min(rows, query_tile)


def _takes_visible_split(q: torch.Tensor, only_visible_blocks: bool, prefetch: bool) -> bool:
    """The visible-split kernel serves decode-sized, visible-only calls whose head count
    divides its dot width; `prefetch` keeps the engine's PDL prefetch kernel.
    """
    heads = q.shape[1] if q.ndim == 3 else 0
    return (
        only_visible_blocks
        and not prefetch
        and q.ndim == 3
        and q.shape[0] <= _DECODE_ROW_LIMIT
        and 0 < heads <= _SPLIT_DOT_WIDTH
        and _SPLIT_DOT_WIDTH % heads == 0
    )


def _qsa_index_scores_visible_split(
    q: torch.Tensor,
    index_cache: torch.Tensor,
    block_table: torch.Tensor,
    query_positions: torch.Tensor,
    request_indices: torch.Tensor,
    tokens_per_block: int,
    compress_ratio: int,
    use_pdl: bool,
    *,
    dot_width: int = _SPLIT_DOT_WIDTH,
    rows_per_tile: int | None = None,
    block_n: int = _SPLIT_BLOCK_N,
    programs_per_sm: int | None = None,
    num_warps: int = _SPLIT_NUM_WARPS,
    num_stages: int = 2,
    wide: bool = False,
) -> torch.Tensor:
    """One launch of the visible-split kernel after the callable's own checks."""
    if index_cache.ndim != 4 or index_cache.shape[2] != 1:
        raise ValueError("QSA index cache must have shape [pages, tokens, 1, dim]")
    if block_table.ndim != 2 or block_table.dtype != torch.int32:
        raise ValueError("QSA index scoring requires a two-dimensional int32 block table")
    rows, num_index_heads, index_head_dim = q.shape
    if index_head_dim <= 0 or index_cache.shape[3] != index_head_dim:
        raise ValueError("QSA index query and cache dimensions do not match")
    if q.dtype != index_cache.dtype:
        raise ValueError("QSA index query and cache dtypes must match")
    if query_positions.shape != (rows,) or request_indices.shape != (rows,):
        raise ValueError("QSA index-scoring metadata must have one entry per row")
    if query_positions.dtype not in _INTEGER_DTYPES or request_indices.dtype not in _INTEGER_DTYPES:
        raise ValueError("QSA index-scoring metadata must use integer storage")
    if tokens_per_block <= 0 or compress_ratio < 2:
        raise ValueError("QSA index scoring requires positive pages and compression ratio >= 2")
    for t in (index_cache, block_table, query_positions, request_indices):
        if t.device != q.device:
            raise ValueError("QSA paged index scoring tensors must share one CUDA device")
    max_compressed_blocks = block_table.shape[1] * tokens_per_block // compress_ratio
    if max_compressed_blocks <= 0:
        raise ValueError("QSA index scoring requires a non-empty block table")
    output = torch.empty((rows, max_compressed_blocks), dtype=torch.float32, device=q.device)
    if rows == 0:
        return output
    query_tile = dot_width // num_index_heads
    if rows_per_tile is None:
        rows_per_tile = _rows_per_tile(rows, query_tile)
    row_tiles = triton.cdiv(rows, rows_per_tile)
    if programs_per_sm is None:
        programs_per_sm = (
            _SPLIT_PROGRAMS_PER_SM_ONE_TILE
            if row_tiles == 1
            else _SPLIT_PROGRAMS_PER_SM_WIDE
            if wide and rows >= _SPLIT_WIDE_MIN_ROWS
            else _SPLIT_PROGRAMS_PER_SM
        )
    # Every (row tile, group slot) pair gets `split` programs; slots past a tile's request
    # count exit at once.
    split = min(
        triton.cdiv(max_compressed_blocks, block_n),
        max(1, triton.cdiv(programs_per_sm * _sm_count(q.device), row_tiles * rows_per_tile)),
    )
    _qsa_index_scores_visible_split_kernel[(row_tiles, rows_per_tile, split)](
        q,
        index_cache,
        block_table,
        query_positions,
        request_indices,
        output,
        index_head_dim**-0.5,
        rows,
        split,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        index_cache.stride(0),
        index_cache.stride(1),
        index_cache.stride(3),
        block_table.stride(0),
        block_table.stride(1),
        output.stride(0),
        output.stride(1),
        NUM_INDEX_HEADS=num_index_heads,
        INDEX_HEAD_DIM=index_head_dim,
        BLOCK_D=max(16, triton.next_power_of_2(index_head_dim)),
        TOKENS_PER_BLOCK=tokens_per_block,
        COMPRESS_RATIO=compress_ratio,
        MAX_COMPRESSED_BLOCKS=max_compressed_blocks,
        # One row tile masks its rows by `num_rows` alone: compile it at the full tile so every
        # row count up to `query_tile` shares one variant.
        ROWS_PER_TILE=query_tile if row_tiles == 1 else rows_per_tile,
        QUERY_TILE=query_tile,
        BLOCK_H=dot_width,
        BLOCK_N=block_n,
        NUM_STAGES=num_stages,
        USE_PDL=use_pdl,
        num_warps=num_warps,
        launch_pdl=use_pdl,
    )
    return output


# The row count follows the batch; the metadata views begin at any row of a mixed step.
@triton.jit(
    do_not_specialize=["rows"],
    do_not_specialize_on_alignment=["query_positions", "request_indices"],
)
def _qsa_index_scores_group_kernel(
    q,
    index_cache,
    block_table,
    query_positions,
    request_indices,
    output,
    score_scale,
    rows,
    q_stride_row: tl.constexpr,
    q_stride_head: tl.constexpr,
    q_stride_dim: tl.constexpr,
    cache_stride_page: tl.constexpr,
    cache_stride_token: tl.constexpr,
    cache_stride_dim: tl.constexpr,
    block_table_stride_request: tl.constexpr,
    block_table_stride_page: tl.constexpr,
    output_stride_row: tl.constexpr,
    output_stride_block: tl.constexpr,
    NUM_INDEX_HEADS: tl.constexpr,
    INDEX_HEAD_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
    TOKENS_PER_BLOCK: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,
    MAX_COMPRESSED_BLOCKS: tl.constexpr,
    GROUP_ROWS: tl.constexpr,
    BLOCK_G: tl.constexpr,
    BLOCK_N: tl.constexpr,
    TILES_PER_PROGRAM: tl.constexpr,
    USE_PDL: tl.constexpr,
):
    """The engine's row-wise kernel (visible blocks only) over a run of ``GROUP_ROWS``
    consecutive rows a program: when the run's rows name one request (a verify step's
    rows), one key gather feeds every row's heads as the dot's columns; otherwise each
    row gathers its own request's keys. Each score is the same dot, ReLU and head sum as
    the row-wise kernel's, bit for bit."""
    BLOCK_H: tl.constexpr = BLOCK_G * NUM_INDEX_HEADS
    if USE_PDL:
        # The memory clobber keeps loads of the predecessor's outputs below the wait.
        gdc_wait_with_memory_clobber()
        tl.extra.cuda.gdc_launch_dependents()
    group = tl.program_id(0)
    slots = tl.arange(0, BLOCK_G)
    group_rows = group * GROUP_ROWS + slots
    row_ok = (slots < GROUP_ROWS) & (group_rows < rows)
    lead = tl.load(request_indices + group * GROUP_ROWS).to(tl.int64)
    requests = tl.load(request_indices + group_rows, mask=row_ok, other=0).to(tl.int64)
    requests = tl.where(row_ok, requests, lead)
    positions = tl.load(query_positions + group_rows, mask=row_ok, other=0)
    visible = tl.where(row_ok, (positions + 1) // COMPRESS_RATIO, 0)
    max_visible = tl.max(visible, axis=0)
    first_tile = tl.program_id(1) * TILES_PER_PROGRAM
    if first_tile * BLOCK_N >= max_visible:
        return
    last_tile = tl.minimum(first_tile + TILES_PER_PROGRAM, tl.cdiv(max_visible, BLOCK_N))
    shared = tl.max((requests != lead).to(tl.int32), axis=0) == 0

    dim_offsets = tl.arange(0, BLOCK_D)
    # Dot column c is head c % heads of the run's row c // heads.
    columns = tl.arange(0, BLOCK_H)
    column_slot = columns // NUM_INDEX_HEADS
    column_rows = group * GROUP_ROWS + column_slot
    column_ok = (column_slot < GROUP_ROWS) & (column_rows < rows)
    query_values = tl.load(
        q
        + column_rows[None, :] * q_stride_row
        + dim_offsets[:, None] * q_stride_dim
        + (columns % NUM_INDEX_HEADS)[None, :] * q_stride_head,
        mask=(dim_offsets[:, None] < INDEX_HEAD_DIM) & column_ok[None, :],
        other=0.0,
    )
    dim_ok = dim_offsets[None, :] < INDEX_HEAD_DIM
    block_offsets = tl.arange(0, BLOCK_N)
    for tile in tl.range(first_tile, last_tile):
        block_columns = tile * BLOCK_N + block_offsets
        valid_blocks = (block_columns < MAX_COMPRESSED_BLOCKS) & (block_columns < max_visible)
        anchor_positions = block_columns * COMPRESS_RATIO + COMPRESS_RATIO - 1
        logical_pages = anchor_positions // TOKENS_PER_BLOCK
        token_in_page = anchor_positions % TOKENS_PER_BLOCK
        if shared:
            physical_pages = tl.load(
                block_table
                + lead * block_table_stride_request
                + logical_pages * block_table_stride_page,
                mask=valid_blocks,
                other=0,
            ).to(tl.int64)
            keys = tl.load(
                index_cache
                + physical_pages[:, None] * cache_stride_page
                + token_in_page[:, None] * cache_stride_token
                + dim_offsets[None, :] * cache_stride_dim,
                mask=valid_blocks[:, None] & dim_ok,
                other=0.0,
                eviction_policy="evict_first",
            )
            per_head_scores = tl.dot(keys, query_values, out_dtype=tl.float32)
        else:
            per_head_scores = tl.zeros((BLOCK_N, BLOCK_H), dtype=tl.float32)
            for slot in tl.static_range(GROUP_ROWS):
                request = tl.sum(tl.where(slots == slot, requests, 0), axis=0)
                physical_pages = tl.load(
                    block_table
                    + request * block_table_stride_request
                    + logical_pages * block_table_stride_page,
                    mask=valid_blocks,
                    other=0,
                ).to(tl.int64)
                keys = tl.load(
                    index_cache
                    + physical_pages[:, None] * cache_stride_page
                    + token_in_page[:, None] * cache_stride_token
                    + dim_offsets[None, :] * cache_stride_dim,
                    mask=valid_blocks[:, None] & dim_ok,
                    other=0.0,
                )
                per_head_scores = tl.where(
                    (column_slot == slot)[None, :],
                    tl.dot(keys, query_values, out_dtype=tl.float32),
                    per_head_scores,
                )
        # The row-wise kernel's head sum is (h0 + h1) + (h2 + h3): pairs, then the pairs.
        pairs = tl.sum(
            tl.reshape(
                tl.maximum(per_head_scores, 0.0), (BLOCK_N, BLOCK_G, NUM_INDEX_HEADS // 2, 2)
            ),
            axis=3,
        )
        scores = tl.sum(pairs, axis=2)
        tl.store(
            output
            + group_rows[None, :] * output_stride_row
            + block_columns[:, None] * output_stride_block,
            scores * score_scale,
            mask=(block_columns[:, None] < visible[None, :]) & row_ok[None, :],
        )


#: Index-head counts the request-group kernel serves: its head sum is the row-wise kernel's
#: at up to 8 heads, two a thread then across the lanes; at 16 the row-wise kernel holds
#: four a thread.
_GROUP_HEADS = (2, 4, 8)
_GROUP_BLOCK_N = 128


def _takes_request_groups(
    q: torch.Tensor,
    only_visible_blocks: bool,
    context_rows: bool,
    prefetch: bool,
    rows_per_request: int,
) -> bool:
    """The request-group kernel serves verify-shaped, visible-only calls: runs of
    `rows_per_request` rows of one request at 2, 4 or 8 index heads. A prefill-sized
    `context_rows` call keeps the engine's query-tiled kernel."""
    return (
        rows_per_request > 1
        and only_visible_blocks
        and not prefetch
        and q.ndim == 3
        and q.shape[1] in _GROUP_HEADS
        and not (context_rows and q.shape[0] > _DECODE_ROW_LIMIT)
    )


#: With `prefill_query_tile`, a prefill-sized, visible-only context-row call at 4 index heads of
#: 128 runs the engine's query-tiled kernel at a deeper launch than the engine's (32 rows a dot
#: tile, 8 column tiles a program): each gathered key tile serves twice the rows. Bitwise the
#: engine launch's scores (a row's dot and head sum do not depend on the tile shape).
_PREFILL_QUERY_TILE = 64
_PREFILL_TILES_PER_PROGRAM = 16
_PREFILL_BLOCK_N = 64
_PREFILL_HEADS = 4
_PREFILL_HEAD_DIM = 128


def _takes_prefill_query_tile(
    q: torch.Tensor, only_visible_blocks: bool, context_rows: bool, prefetch: bool
) -> bool:
    """A context-row call the deeper query-tiled launch serves (`_PREFILL_QUERY_TILE`)."""
    return (
        context_rows
        and only_visible_blocks
        and not prefetch
        and q.ndim == 3
        and q.shape[0] > _DECODE_ROW_LIMIT
        and q.shape[1] == _PREFILL_HEADS
        and q.shape[2] == _PREFILL_HEAD_DIM
        and q.dtype == torch.bfloat16
    )


def _qsa_index_scores_prefill_query_tile(
    q: torch.Tensor,
    index_cache: torch.Tensor,
    block_table: torch.Tensor,
    query_positions: torch.Tensor,
    request_indices: torch.Tensor,
    tokens_per_block: int,
    compress_ratio: int,
) -> torch.Tensor:
    """The engine's query-tiled kernel at the deeper launch. Same contract as the engine's
    launcher with `context_rows` and `only_visible_blocks`: columns at and past a row's
    visible bound are unspecified."""
    rows = q.shape[0]
    if index_cache.ndim != 4 or index_cache.shape[2] != 1 or index_cache.shape[3] != q.shape[2]:
        raise ValueError("QSA index cache must have shape [pages, tokens, 1, dim]")
    if block_table.ndim != 2 or block_table.dtype != torch.int32:
        raise ValueError("QSA index scoring requires a two-dimensional int32 block table")
    if q.dtype != index_cache.dtype:
        raise ValueError("QSA index query and cache dtypes must match")
    if query_positions.shape != (rows,) or request_indices.shape != (rows,):
        raise ValueError("QSA index-scoring metadata must have one entry per row")
    max_blocks = block_table.shape[1] * tokens_per_block // compress_ratio
    output = torch.empty((rows, max_blocks), dtype=torch.float32, device=q.device)
    grid = (
        triton.cdiv(rows, _PREFILL_QUERY_TILE),
        triton.cdiv(max_blocks, _PREFILL_BLOCK_N * _PREFILL_TILES_PER_PROGRAM),
    )
    _qsa_kernels._qsa_paged_index_scores_qtile_kernel[grid](
        q,
        index_cache,
        block_table,
        query_positions,
        request_indices,
        output,
        q.shape[2] ** -0.5,
        rows,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        index_cache.stride(0),
        index_cache.stride(1),
        index_cache.stride(2),
        index_cache.stride(3),
        block_table.stride(0),
        block_table.stride(1),
        output.stride(0),
        output.stride(1),
        NUM_INDEX_HEADS=q.shape[1],
        INDEX_HEAD_DIM=q.shape[2],
        BLOCK_D=q.shape[2],
        TOKENS_PER_BLOCK=tokens_per_block,
        COMPRESS_RATIO=compress_ratio,
        MAX_COMPRESSED_BLOCKS=max_blocks,
        QUERY_TILE=_PREFILL_QUERY_TILE,
        BLOCK_H=_PREFILL_QUERY_TILE * q.shape[1],
        BLOCK_N=_PREFILL_BLOCK_N,
        TILES_PER_PROGRAM=_PREFILL_TILES_PER_PROGRAM,
        ONLY_VISIBLE_BLOCKS=True,
        num_warps=4,
    )
    return output


def _qsa_index_scores_request_groups(
    q: torch.Tensor,
    index_cache: torch.Tensor,
    block_table: torch.Tensor,
    query_positions: torch.Tensor,
    request_indices: torch.Tensor,
    tokens_per_block: int,
    compress_ratio: int,
    use_pdl: bool,
    rows_per_request: int,
) -> torch.Tensor:
    """One launch of the request-group kernel after the callable's own checks."""
    if index_cache.ndim != 4 or index_cache.shape[2] != 1:
        raise ValueError("QSA index cache must have shape [pages, tokens, 1, dim]")
    if block_table.ndim != 2 or block_table.dtype != torch.int32:
        raise ValueError("QSA index scoring requires a two-dimensional int32 block table")
    rows, num_index_heads, index_head_dim = q.shape
    if index_head_dim <= 0 or index_cache.shape[3] != index_head_dim:
        raise ValueError("QSA index query and cache dimensions do not match")
    if q.dtype != index_cache.dtype:
        raise ValueError("QSA index query and cache dtypes must match")
    if query_positions.shape != (rows,) or request_indices.shape != (rows,):
        raise ValueError("QSA index-scoring metadata must have one entry per row")
    if query_positions.dtype not in _INTEGER_DTYPES or request_indices.dtype not in _INTEGER_DTYPES:
        raise ValueError("QSA index-scoring metadata must use integer storage")
    if tokens_per_block <= 0 or compress_ratio < 2:
        raise ValueError("QSA index scoring requires positive pages and compression ratio >= 2")
    for t in (index_cache, block_table, query_positions, request_indices):
        if t.device != q.device:
            raise ValueError("QSA paged index scoring tensors must share one CUDA device")
    max_compressed_blocks = block_table.shape[1] * tokens_per_block // compress_ratio
    if max_compressed_blocks <= 0:
        raise ValueError("QSA index scoring requires a non-empty block table")
    output = torch.empty((rows, max_compressed_blocks), dtype=torch.float32, device=q.device)
    if rows == 0:
        return output
    # A verify step's rows of one request read the same keys: one gather a run of rows,
    # wider tiles, and a deeper column range once there are runs enough to fill the GPU.
    groups = triton.cdiv(rows, rows_per_request)
    tiles = 1 if groups <= 1 else 2 if groups <= 4 else 4
    grid = (groups, triton.cdiv(max_compressed_blocks, _GROUP_BLOCK_N * tiles))
    args = (
        q,
        index_cache,
        block_table,
        query_positions,
        request_indices,
        output,
        index_head_dim**-0.5,
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
        output.stride(1),
    )
    constexprs = dict(
        NUM_INDEX_HEADS=num_index_heads,
        INDEX_HEAD_DIM=index_head_dim,
        BLOCK_D=max(16, triton.next_power_of_2(index_head_dim)),
        TOKENS_PER_BLOCK=tokens_per_block,
        COMPRESS_RATIO=compress_ratio,
        MAX_COMPRESSED_BLOCKS=max_compressed_blocks,
        GROUP_ROWS=rows_per_request,
        # The dot's columns: every row's heads, at least the 16 the row-wise kernel pads to.
        BLOCK_G=max(triton.next_power_of_2(rows_per_request), 16 // num_index_heads, 1),
        BLOCK_N=_GROUP_BLOCK_N,
        TILES_PER_PROGRAM=tiles,
        USE_PDL=use_pdl,
        num_warps=4,
        launch_pdl=use_pdl,
    )
    # `TILES_PER_PROGRAM` follows the request count: compile every depth at the first call.
    warm_variants(
        _qsa_index_scores_group_kernel,
        (
            args[8:18],  # the strides
            q.dtype,
            q.data_ptr() % 16 == 0,
            query_positions.dtype,
            request_indices.dtype,
            num_index_heads,
            index_head_dim,
            tokens_per_block,
            compress_ratio,
            max_compressed_blocks,
            rows_per_request,
            use_pdl,
        ),
        args,
        [{"TILES_PER_PROGRAM": t} for t in (1, 2, 4)],
        **constexprs,
    )
    _qsa_index_scores_group_kernel[grid](*args, **constexprs)
    return output


_CHECKPOINT = dict(
    heads=4,
    head_dim=128,
    tokens_per_block=32,
    compress_ratio=4,
    dtype=torch.bfloat16,
    # `qsa_logical_positions` is int64, `qsa_req_idx_per_token` int32.
    positions_dtype=torch.int64,
    request_dtype=torch.int32,
    # The target's q is a head slice of its packed [T, 4 + 1 + 1, 128] index
    # scratch buffer, so its row stride is 768 rather than 512.
    q_layout="scratch_slice",
)


def _spec(**overrides) -> dict:
    """A cell spec: this checkpoint's geometry unless the cell says otherwise."""
    return {**_CHECKPOINT, **overrides}


#: The serving tables: [8, 260] on the warm-up capture path and [32, 260] in mixed
#: steps.
_WIDTH = 260


#: The same plane under an NVFP4 KV pool: the QSA position page joins each physical
#: page after the index buffers (`page_pad` elements), so the trunk's view is
#: `[pages, 32, 1, 128]` at page stride 12 x 4096 + 192 = 49344.
_TRUNK_POOL_NVFP4 = dict(pool_buffers=12, page_pad=192)

#: Core's plane (bf16 K/V) at every four-rank layout, as the stock manager allocates it:
#: the 12 attention layers' `[32, 1, 128]` INDEX_KEY buffers coalesce into one pool, layer k at
#: byte 8192 k of a 98304-byte page, so one layer's view is `[pages, 32, 1, 128]` at page stride
#: 12 x 4096 = 49152. The cells take the last layer's plane.
_TRUNK_POOL_BF16 = dict(pool_buffers=12, page_pad=0)

#: Pages in one rank's pool: the stock engine's `maxNumBlocks` at core's configuration (bf16 K/V
#: and recurrent state, KV fraction 0.6, max_batch_size 128, max_seq_len 9216, the n-gram table
#: host-resident). The largest per member; the index plane shares the attention pages.
_TEP4_POOL_PAGES = 258983
_DEP4_POOL_PAGES = 120530

#: `QSAAttentionMetadata.qsa_block_table` is `[max_num_requests, max_blocks_per_seq]`: 128 rows
#: on a tep4 or dep4 rank (the engine's max_num_requests), and ceil((max_seq_len + generation
#: headroom) / 32) padded to a multiple of 4 -- 292 at core's 9216, 516 at MTP's 16384.
_TABLE_SLOTS = 128
_CORE_TABLE_WIDTH = 292
_MTP_TABLE_WIDTH = 516

#: The longest histories each envelope serves: the last position below max_seq_len.
_CORE_LAST_POSITION = 9216 - 1
_MTP_LAST_POSITION = 16384 - 1

#: The target's AgentX server at tp1 (max_seq_len 262144): an 8200-page table (65600 score
#: columns, 1025 tiles of 64), histories near 100k tokens and to the end.
_AGENTX_TABLE_WIDTH = 8200
_AGENTX_CONTEXT_LAST = 100000 - 1
_AGENTX_LAST_POSITION = 262144 - 1


#: The tune's table: max_seq_len 262144 is 8192 pages, and MTP's headroom makes it 8196,
#: 65568 columns.
_TUNE_TABLE_WIDTH = 8196
_TUNE_LAST_POSITION = 262144 - 1
#: Windows of 0 (positions 0..2), 1, 8, 32 and 33 blocks, ~20k-94k decode contexts, and the
#: envelope's last position.
_TUNE_DEP4_POSITIONS = (
    94000, 0, 2, 3, 31, 127, 131, 20000, 34800, 49600, 64400, 65535, 79200, 88888, 131071,
    _TUNE_LAST_POSITION,
)  # fmt: skip


def _decode_positions(requests: int, last: int, rows_per_request: int = 1) -> tuple[int, ...]:
    """Each request's first row position; the first request's last row sits at `last`."""
    first = last - rows_per_request + 1
    return (first,) + tuple(1 + (7919 * g) % first for g in range(1, requests))


class QsaPagedIndexScores(OpWrapper):
    """Score every causally visible compressed block of each query row.

    One Triton launch after allocating the output: the entry's own visible-split
    kernel (`_qsa_index_scores_visible_split_kernel`) for ungrouped decode-sized
    visible-only calls, its request-group kernel (`_qsa_index_scores_group_kernel`) for
    verify-shaped ones, otherwise `tensorrt_llm._torch.attention.backends.sparse.qsa.kernels.
    triton_qsa_paged_index_scores`. For row `r` and compressed block `b` of request
    `request_indices[r]`::

        anchor    = b * compress_ratio + compress_ratio - 1
        key_b     = index_cache[block_table[req, anchor // tpb], anchor % tpb, 0]
        out[r, b] = sum_h relu(q[r, h] . key_b) * index_head_dim ** -0.5

    The block's key is whatever the compression step left at the group's LAST
    position; nothing is pooled here. Row `r` sees `(query_positions[r] + 1) //
    compress_ratio` blocks (complete groups only). Columns at or past that are
    `-inf` when `only_visible_blocks` is False and **not written** -- the fresh
    allocation's garbage -- when it is True. The column count is the block
    table's width, `block_table.shape[1] * tpb // compress_ratio`.

    Kernel choice. With `rows_per_request=1`, `only_visible_blocks`, no `prefetch`,
    at most 256 rows and an index-head count dividing 16, the visible-split kernel
    runs: a `(row tiles, request slots, split)` grid sized by the device (16 programs
    an SM, 32 when the batch is one tile) whose programs stride over each request's
    visible column tiles, one key gather serving every row of that request among
    `16 // heads` consecutive rows -- so the grid does not scale with the table's
    columns. Otherwise the engine chooses a kernel: `context_rows` is a layout
    declaration ("consecutive rows mostly share a request"), not a phase one. With
    more than 256 rows and an index-head count that divides 128 into more than one
    row (1, 2, 4, ..., 64), it selects the query-tiled kernel sharing one key gather
    across `128 // heads` rows; otherwise the row-wise kernel runs. Before the engine's
    choice, `rows_per_request` above 1 declares runs of that many rows of one request
    from row 0 (a verify step's `predicted_tokens_per_seq`): with `only_visible_blocks`,
    no `prefetch` and 2, 4 or 8 index heads, the entry's request-group kernel shares one
    key gather across each run (a run whose rows name different requests is gathered
    row by row). With `prefill_query_tile`, a visible-only context-row call of more
    than 256 rows at 4 bf16 index heads of 128 runs the query-tiled kernel at a deeper
    launch, bitwise the engine launch's scores. With `wide_split`, a visible-split call
    of `_SPLIT_WIDE_MIN_ROWS` rows or more past one row tile takes more programs an SM,
    bitwise the narrow split. Both default off. Each cell names the kernel it was
    measured on. The request-group kernel is bitwise the row-wise one; the visible-split
    and query-tiled kernels are not.

    Fusion boundary: the table walk, the gather, the dots, relu, head sum,
    scale and causal bound are inside. Producing `q` (norm, RoPE), writing and
    compressing the index cache, and everything after the scores are not.
    Returns a fresh fp32 `[rows, columns]`; nothing is written in place.
    """

    ARCHS = frozenset({Arch.SM_103})

    CELLS: tuple[Cell, ...] = (
        Cell(
            why="NVFP4 trunk plane, page stride 49344: decode MTP 4 requests x 2 rows",
            spec=_spec(
                layout="decode",
                gen_positions=(7000, 5000, 4200, 2100),
                rows_per_request=2,
                slots=32,
                pages=_WIDTH,
                only_visible_blocks=True,
                context_rows=False,
                call_rows_per_request=2,
                branch="request_group",
                **_TRUNK_POOL_NVFP4,
            ),
        ),
        Cell(
            why="decode MTP7: 32 slots x 8 rows = 256 rows, the last decode-sized count",
            spec=_spec(
                layout="decode",
                gen_positions=tuple(100 + (499 * g) % 8000 for g in range(32)),
                rows_per_request=8,
                slots=32,
                pages=_WIDTH,
                only_visible_blocks=True,
                context_rows=False,
                call_rows_per_request=8,
                branch="request_group",
            ),
        ),
        # The target's AgentX verify steps at tp1 (MTP3: 4 rows a request) over the NVFP4 trunk
        # plane and its 8200-page table, NaN on every page no request maps.
        Cell(
            why="AgentX MTP3 verify: 16 requests x 4 rows to 99999, 65600 columns",
            spec=_spec(
                layout="decode",
                gen_positions=_decode_positions(16, _AGENTX_CONTEXT_LAST, rows_per_request=4),
                rows_per_request=4,
                call_rows_per_request=4,
                slots=32,
                pages=_AGENTX_TABLE_WIDTH,
                pool_pages=60000,
                only_visible_blocks=True,
                context_rows=False,
                use_pdl=True,
                branch="request_group",
                **_TRUNK_POOL_NVFP4,
            ),
        ),
        Cell(
            why="AgentX MTP3 verify: 16 requests x 4 rows to 262143, 65600 columns",
            spec=_spec(
                layout="decode",
                gen_positions=_decode_positions(16, _AGENTX_LAST_POSITION, rows_per_request=4),
                rows_per_request=4,
                call_rows_per_request=4,
                slots=32,
                pages=_AGENTX_TABLE_WIDTH,
                pool_pages=140000,
                only_visible_blocks=True,
                context_rows=False,
                use_pdl=True,
                branch="request_group",
                **_TRUNK_POOL_NVFP4,
            ),
        ),
        Cell(
            why="AgentX MTP3 verify: 1 request x 4 rows to 262143, 65600 columns",
            spec=_spec(
                layout="decode",
                gen_positions=_decode_positions(1, _AGENTX_LAST_POSITION, rows_per_request=4),
                rows_per_request=4,
                call_rows_per_request=4,
                slots=32,
                pages=_AGENTX_TABLE_WIDTH,
                pool_pages=60000,
                only_visible_blocks=True,
                context_rows=False,
                use_pdl=True,
                branch="request_group",
                **_TRUNK_POOL_NVFP4,
            ),
        ),
        Cell(
            why="capture: 8191 rows over 8 ragged requests, visible-only",
            spec=_spec(
                layout="capture",
                lengths=(1024, 512, 2048, 256, 1500, 700, 1151, 1000),
                slots=8,
                pages=_WIDTH,
                only_visible_blocks=True,
                context_rows=True,
                branch="query_tile",
            ),
        ),
        # The four-rank layouts' sites over one rank's measured pool: the call-site flags each
        # site passes (the decode block's PDL launch, with prefetch at up to 4 rows), the
        # engine's [128, width] table with requests on seeded rows, and pages drawn over the
        # whole pool, both ends included.
        Cell(
            why="tep4 decode: one request at position 9215, prefetch, 258983-page pool",
            spec=_spec(
                layout="decode",
                gen_positions=_decode_positions(1, _CORE_LAST_POSITION),
                rows_per_request=1,
                slots=_TABLE_SLOTS,
                pages=_CORE_TABLE_WIDTH,
                pool_pages=_TEP4_POOL_PAGES,
                only_visible_blocks=True,
                context_rows=False,
                use_pdl=True,
                prefetch=True,
                branch="row_wise",
                **_TRUNK_POOL_BF16,
            ),
        ),
        Cell(
            why="tep4 decode: 128 requests x 1 row, the largest captured batch",
            spec=_spec(
                layout="decode",
                gen_positions=_decode_positions(128, _CORE_LAST_POSITION),
                rows_per_request=1,
                slots=_TABLE_SLOTS,
                pages=_CORE_TABLE_WIDTH,
                pool_pages=_TEP4_POOL_PAGES,
                only_visible_blocks=True,
                context_rows=False,
                use_pdl=True,
                branch="visible_split",
                **_TRUNK_POOL_BF16,
            ),
        ),
        Cell(
            why="tep4 MTP7 verify: 128 requests x 8 rows = 1024, table 516, to 16383",
            spec=_spec(
                layout="decode",
                gen_positions=_decode_positions(128, _MTP_LAST_POSITION, rows_per_request=8),
                rows_per_request=8,
                slots=_TABLE_SLOTS,
                pages=_MTP_TABLE_WIDTH,
                pool_pages=_TEP4_POOL_PAGES,
                only_visible_blocks=True,
                context_rows=False,
                use_pdl=True,
                branch="row_wise",
                **_TRUNK_POOL_BF16,
            ),
        ),
        Cell(
            why="tep4 context: one fresh 8192-token prompt (max_num_tokens)",
            spec=_spec(
                layout="capture",
                lengths=(8192,),
                slots=_TABLE_SLOTS,
                pages=_CORE_TABLE_WIDTH,
                pool_pages=_TEP4_POOL_PAGES,
                only_visible_blocks=True,
                context_rows=True,
                branch="query_tile",
                **_TRUNK_POOL_BF16,
            ),
        ),
        Cell(
            why="dep4 decode: 32 requests x 1 row over a 120530-page pool",
            spec=_spec(
                layout="decode",
                gen_positions=_decode_positions(32, _CORE_LAST_POSITION),
                rows_per_request=1,
                slots=_TABLE_SLOTS,
                pages=_CORE_TABLE_WIDTH,
                pool_pages=_DEP4_POOL_PAGES,
                only_visible_blocks=True,
                context_rows=False,
                use_pdl=True,
                branch="visible_split",
                **_TRUNK_POOL_BF16,
            ),
        ),
        Cell(
            why="envelope: an 8192-row chunk at positions 1024..9215, table column 287",
            spec=_spec(
                layout="decode",
                gen_positions=(_CORE_LAST_POSITION + 1 - 8192,),
                rows_per_request=8192,
                slots=_TABLE_SLOTS,
                pages=_CORE_TABLE_WIDTH,
                pool_pages=_TEP4_POOL_PAGES,
                only_visible_blocks=True,
                context_rows=True,
                branch="query_tile",
                **_TRUNK_POOL_BF16,
            ),
        ),
        # The tune's envelope (max_seq_len 262144): an 8196-page table, 65568 columns.
        Cell(
            why="tune ll MTP3 verify: 1 request x 4 rows at 84000..84003, 8196-page table",
            spec=_spec(
                layout="decode",
                gen_positions=(84000,),
                rows_per_request=4,
                slots=_TABLE_SLOTS,
                pages=_TUNE_TABLE_WIDTH,
                pool_pages=_TEP4_POOL_PAGES,
                only_visible_blocks=True,
                context_rows=False,
                branch="visible_split",
                **_TRUNK_POOL_BF16,
            ),
        ),
        Cell(
            why="tune draft: 1 row at 262143, the envelope's last position",
            spec=_spec(
                layout="decode",
                gen_positions=(_TUNE_LAST_POSITION,),
                rows_per_request=1,
                slots=_TABLE_SLOTS,
                pages=_TUNE_TABLE_WIDTH,
                pool_pages=_TEP4_POOL_PAGES,
                only_visible_blocks=True,
                context_rows=False,
                branch="visible_split",
                **_TRUNK_POOL_BF16,
            ),
        ),
        Cell(
            why="tune MTP7 verify: 4 requests x 8 rows, the first to 262143",
            spec=_spec(
                layout="decode",
                gen_positions=_decode_positions(4, _TUNE_LAST_POSITION, rows_per_request=8),
                rows_per_request=8,
                slots=_TABLE_SLOTS,
                pages=_TUNE_TABLE_WIDTH,
                pool_pages=_TEP4_POOL_PAGES,
                only_visible_blocks=True,
                context_rows=False,
                use_pdl=True,
                branch="visible_split",
                **_TRUNK_POOL_BF16,
            ),
        ),
        Cell(
            why="tune dep4 decode: 16 requests x 1 row, empty to 262143-position windows",
            spec=_spec(
                layout="decode",
                gen_positions=_TUNE_DEP4_POSITIONS,
                rows_per_request=1,
                slots=_TABLE_SLOTS,
                pages=_TUNE_TABLE_WIDTH,
                pool_pages=_DEP4_POOL_PAGES,
                only_visible_blocks=True,
                context_rows=False,
                branch="visible_split",
                **_TRUNK_POOL_BF16,
            ),
        ),
        Cell(
            why="tune context step: 2 rows of one request at 84000 (the snapshot split)",
            spec=_spec(
                layout="decode",
                gen_positions=(84000,),
                rows_per_request=2,
                slots=_TABLE_SLOTS,
                pages=_TUNE_TABLE_WIDTH,
                pool_pages=_TEP4_POOL_PAGES,
                only_visible_blocks=True,
                context_rows=True,
                branch="visible_split",
                **_TRUNK_POOL_BF16,
            ),
        ),
        Cell(
            why="small context: fresh 200- and 56-token prompts, 256 rows from position 0",
            spec=_spec(
                layout="capture",
                lengths=(200, 56),
                slots=_TABLE_SLOTS,
                pages=_TUNE_TABLE_WIDTH,
                pool_pages=_DEP4_POOL_PAGES,
                only_visible_blocks=True,
                context_rows=True,
                branch="visible_split",
                **_TRUNK_POOL_BF16,
            ),
        ),
    )

    note = """
    Source: the entry's own Triton kernel (`_qsa_index_scores_visible_split_kernel`,
    this file) and the engine's beside the wrapped callable
    (`_qsa_paged_index_scores_kernel`, `_qsa_paged_index_scores_qtile_kernel`,
    `_qsa_paged_index_scores_group_kernel`),
    JIT-compiled for the running device on first call; the cells ran that JIT
    path on sm_103.

    Visible-split kernel (decode-sized visible-only calls). Its grid is
    `(row tiles, request slots, split)`: a row tile is `min(rows, 16 // heads)`
    consecutive rows, program `(t, g, s)` takes the g-th distinct request of
    tile t (a slot past the tile's request count exits at once) and its column
    tiles s, s + split, ... of 64 blocks up to that request's bound, `split`
    fixed by the SM count (16 programs an SM, 32 for a one-tile batch) and the
    table, never by the positions -- so a captured graph follows any window
    (pinned by `test_captured_visible_split_follows_the_positions`). The dot is
    `[64, 128] x [128, rows x heads]` with fp32 accumulation, relu, a per-row
    head sum and the scale, as in the engine's kernels; the columns written are
    exactly each row's visible ones. `use_pdl` waits on its predecessor before the first read and
    releases dependents at once; `prefetch` keeps the engine's kernel (the
    visible-split one does not read before the wait).

    Metadata consumed: none -- every input is an argument. What the target
    passes: `block_table` is `QSAAttentionMetadata.qsa_block_table` (raw
    pool-page ids, int32, -1 past each sequence); `query_positions` /
    `request_indices` are `qsa_logical_positions` / `qsa_req_idx_per_token`;
    `index_cache` is the layer's bf16 index-K plane (`get_index_k_buffer`).

    Preconditions (not checked): every block-table entry the causal bound
    reaches is a valid page, and `request_indices[r]` is a row of the table --
    the loads are masked by the bound, not by the pool size, so a violation is
    an out-of-bounds read (not probed). The compression step must already have
    run: an uncompressed anchor scores as a raw index key. `rows == 0` returns
    an empty `[0, columns]` tensor without launching.

    The op validates its own metadata loudly (`ValueError`s; the test pins the
    one-entry-per-row metadata refusal), so `is_valid` adds nothing.

    `only_visible_blocks=True` really leaves garbage in the whole unscored tail
    (measured, not pinned by a test). A consumer that scans `[row_start,
    row_end)` -- `qsa_topk_select` -- is unaffected.

    fp32 operands are accepted but `tl.dot` rounds them to a tensor-core
    format: 8.1e-3 max abs against an fp32 reference on scores to 7.8 (measured),
    outside the test's band. No cell covers fp32; the side cache is bf16.

    CUDA graphs: the test captures the decode launch (`use_pdl`, `prefetch`) at
    4 rows and replays it after `q` and the anchor keys were rewritten in
    place; every replay matches an eager call bitwise. The same holds captured
    at each measured-pool decode and verify cell (1, 128 and 1024 rows at tep4,
    32 at dep4, the AgentX MTP3 verify cells' 4 and 64 rows, with their sites' flags)
    after q and every mapped key changed.

    The request-group kernel (`rows_per_request` > 1): each program takes one run of
    rows and 128 blocks a tile (1, 2 or 4 tiles a program at 1, up to 4, and more runs);
    the dot's columns are the run's rows' heads, padded to 16 as the row-wise kernel
    pads its heads, and the head sum is the row-wise kernel's pairwise order, so every
    score is bitwise the row-wise kernel's -- the test compares the bit patterns at 2,
    3, 4 and 8 rows a run, 2, 4 and 8 heads, partial last runs and runs that mix
    requests; 16 heads stay row-wise.
    The run reads up to its longest row's visible blocks and stores each row's own.

    At the measured-pool cells: at most 2.9e-6 against the reference (scores to
    11.3), relative L2 below 6.4e-8.

    `prefetch` (with `use_pdl` and `only_visible_blocks`, at generation-sized
    batches) reads the pages and compressed keys before the PDL wait and q after
    it. Sound only behind the decode step's own prologue
    (`qwen4_exp_attn_decode_prologue`), which writes no compressed key but the last
    visible block's, at a row whose position ends a group; that key is read again
    after the wait.

    `prefetch` takes effect only where the call is one tile a program -- 256 rows or
    fewer -- and is otherwise dropped silently: a 1024-row verify step passing it
    runs the plain PDL kernel.

    Every cell runs `only_visible_blocks=True` (the target never passes False),
    4 index heads x 128, compress ratio 4, 32-token pages, bf16 operands, int64
    positions and int32 request indices. The NVFP4-pool page stride (49344) is
    certified on the row-wise kernel at the tp1 cell's small pool and on the
    request-group kernel at the AgentX cells' 60000- and 140000-page pools.

    The four-rank layouts' cells run over one rank's measured pool, core's coalesced
    plane at page stride 49152 (`page_stride` is a compile-time constant, so this
    is its own kernel variant): 258983 pages at tep4, 120530 at dep4, the plane
    poisoned with NaN outside the mapped pages, which include page 0 and the
    pool's last page. The last page's elements sit near 1.27e10 elements from the
    plane's base: both kernels widen the page and request index to int64 before
    the stride product, and the cells prove it.

    Index arithmetic left in int32: `row * q_stride_row` and the output's
    `row * output_stride_row` (program ids times compile-time strides). The output
    term passes 2^31 at rows x columns >= 2^31 -- 919k rows at 2336 columns; the
    envelope's 8192 rows x 65568 columns (max_seq_len 262144) is 5.4e8. Positions
    are read at their own width (int64 here), so the fp32 2^24 bound does not apply.

    The visible-split kernel's index arithmetic: pages and request rows are widened
    to int64 before the stride products (the measured-pool cells read the pool's last
    page); `lane_rows * output_stride_row` stays int32, at most 256 x 65568 here.

    Certified: up to 8192 rows (one request's 8192-row context, fresh or ending at
    position 9215, on the query-tile kernel; 1024 verify rows on the row-wise
    kernel with 8 tiles a program), tables of 128 x 292 and 128 x 516 pages,
    positions to 16383; on the visible-split kernel also the tune's 128 x 8196
    table (65568 columns) with positions to 262143 at 1, 2, 4, 16 and 32 rows
    (empty windows, 1 and 2 rows a request, 8-row requests across two tiles, small
    context steps from position 0); on the request-group kernel 2, 4 and 8 rows a
    request, up to 256 rows, a 32 x 8200 table and positions to 262143. Not certified:
    more than 8192 rows, tables wider than 516 pages on the row-wise and query-tile
    kernels (the tune's context-sized calls), any other head geometry, compress ratio,
    page size or operand dtype, a context step's mixed request rows on the pool
    layout, the fp8 side-plane stride at a measured pool, and the `-inf` padded mode.
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
        only_visible_blocks: bool,
        context_rows: bool,
        use_pdl: bool = False,
        prefetch: bool = False,
        rows_per_request: int = 1,
        prefill_query_tile: bool = False,
        wide_split: bool = False,
    ) -> torch.Tensor:
        if rows_per_request < 1:
            raise ValueError("QSA index scoring requires rows_per_request >= 1")
        if rows_per_request == 1 and _takes_visible_split(q, only_visible_blocks, prefetch):
            return _qsa_index_scores_visible_split(
                q,
                index_cache,
                block_table,
                query_positions,
                request_indices,
                tokens_per_block,
                compress_ratio,
                use_pdl,
                wide=wide_split,
            )
        if _takes_request_groups(q, only_visible_blocks, context_rows, prefetch, rows_per_request):
            return _qsa_index_scores_request_groups(
                q,
                index_cache,
                block_table,
                query_positions,
                request_indices,
                tokens_per_block,
                compress_ratio,
                use_pdl,
                rows_per_request,
            )
        if prefill_query_tile and _takes_prefill_query_tile(
            q, only_visible_blocks, context_rows, prefetch
        ):
            return _qsa_index_scores_prefill_query_tile(
                q,
                index_cache,
                block_table,
                query_positions,
                request_indices,
                tokens_per_block,
                compress_ratio,
            )
        return _triton_qsa_paged_index_scores(
            q=q,
            index_cache=index_cache,
            block_table=block_table,
            query_positions=query_positions,
            request_indices=request_indices,
            tokens_per_block=tokens_per_block,
            compress_ratio=compress_ratio,
            only_visible_blocks=only_visible_blocks,
            context_rows=context_rows,
            use_pdl=use_pdl,
            prefetch=prefetch,
        )

    def reference(
        self,
        q: torch.Tensor,
        index_cache: torch.Tensor,
        block_table: torch.Tensor,
        query_positions: torch.Tensor,
        request_indices: torch.Tensor,
        tokens_per_block: int,
        compress_ratio: int,
        only_visible_blocks: bool,
        context_rows: bool,
        use_pdl: bool = False,
        prefetch: bool = False,
        rows_per_request: int = 1,
        prefill_query_tile: bool = False,
        wide_split: bool = False,
    ) -> torch.Tensor:
        """fp32 scores, `-inf` past each row's bound -- or NaN there, meaning
        "unspecified", when `only_visible_blocks` is True. `context_rows`,
        `rows_per_request`, `prefill_query_tile` and `wide_split` are ignored: they pick a
        kernel or a launch, not a result.

        Walked in row chunks only to bound the gathered keys, which reach
        8.7 GiB in fp32 at 8191 rows x 2080 columns.
        """
        rows, _, head_dim = q.shape
        num_columns = block_table.shape[1] * tokens_per_block // compress_ratio
        blocks = torch.arange(num_columns, device=q.device)
        anchors = blocks * compress_ratio + compress_ratio - 1
        logical_pages, in_page = anchors // tokens_per_block, anchors % tokens_per_block
        table = block_table.long()
        requests = request_indices.long()
        scores = torch.empty(rows, num_columns, dtype=torch.float32, device=q.device)
        chunk = max(1, 2**26 // max(num_columns * head_dim, 1))
        for lo in range(0, rows, chunk):
            hi = min(lo + chunk, rows)
            # Pages past a sequence are -1; their columns are masked below.
            physical = table[requests[lo:hi]][:, logical_pages].clamp_min(0)
            keys = index_cache[physical, in_page.expand(hi - lo, num_columns)][:, :, 0, :]
            per_head = torch.einsum("rhd,rbd->rbh", q[lo:hi].float(), keys.float())
            scores[lo:hi] = torch.relu(per_head).sum(-1) * (head_dim**-0.5)
        visible = blocks[None, :] < ((query_positions.long() + 1) // compress_ratio)[:, None]
        tail = float("nan") if only_visible_blocks else float("-inf")
        return scores.masked_fill(~visible, tail)


qsa_paged_index_scores = QsaPagedIndexScores()
