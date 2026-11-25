# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""QSA compressed-block index scoring over e4m3 index keys: a lossy, opt-in layout of the index
side cache in which each four-token group's anchor (the pooled key `qsa_prefill_compress` and
the decode prologues write as bf16) is stored as 128 e4m3 bytes in the first half of its slot,
and every score is a native e4m3 x e4m3 dot of that key with the e4m3-rounded query.

The plane keeps its bf16 size and layout (`[pages, tokens, 1, 128]`), so the cache manager, the
KV transfer and the rollback copies are unchanged; raw (uncompressed) keys stay bf16. Every
writer of an anchor converts the anchors it wrote (`index_anchors_to_fp8`), and every reader of
the plane's anchors scores with `qsa_index_scores_fp8`. A context worker and the generation
workers it hands requests to must both run in this layout: a transferred plane carries its
anchors as they were written.

Numerics: lossy against the bf16 keys (each anchor and the query rounded to e4m3, no scale);
exact against the fp32 reference over the rounded operands up to fp32 summation order. GB300
(dep4, MTP-3, 96 requests a rank): the index-score kernels of a step 2.19 -> 1.82 ms; a 16384-row
context step's twelve calls 14.9 ms on the bf16 query-tiled kernel, 9.5 ms on the ctx-tile kernel
here. GSM8K 95.91 / 95.98; the AgentX top-2048 block overlap with bf16 keys p50 0.976.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from tensorrt_llm._torch.modeling_v2.catalog._op import Cell, OpWrapper
from tensorrt_llm._torch.modeling_v2.models.qwen4_exp.archs import SERVED_ARCHS

#: At or below this many rows a call is decode-sized; past it a context call takes the ctx-tile
#: kernel.
_DECODE_ROW_LIMIT = 256
#: Score columns a program of the request-group kernel takes a tile.
_GROUP_BLOCK_N = 128
#: The ctx-tile kernel: 32 rows x 4 heads = 128 dot columns over 64-block column tiles, 16 column
#: tiles a program, 4 warps, Triton pipeline depth 2 (GB300, 16384 rows over 350k cached tokens,
#: ms a step of 12 calls: 8 rows 27.8, 16 rows 17.9, 32 rows x 64 blocks x 8 tiles 12.4,
#: x 16 tiles 10.4, at depth 2 9.5; 8 warps, 2 warps and 32-block tiles were slower).
_CTX_TILE_ROWS = 32
_CTX_TILE_BLOCK_N = 64
_CTX_TILE_TILES = 16
_CTX_TILE_WARPS = 4
_CTX_TILE_STAGES = 2


@triton.jit
def _anchors_to_fp8_kernel(
    cache16,
    cache8,
    block_table,
    request_indices,
    logical_positions,
    rows,
    stride16_page: tl.constexpr,
    stride16_token: tl.constexpr,
    stride8_page: tl.constexpr,
    stride8_token: tl.constexpr,
    block_table_stride_request: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    TOKENS_PER_BLOCK: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,
    BLOCK_R: tl.constexpr,
    USE_PDL: tl.constexpr,
):
    """Each anchor row's bf16 key, rounded to e4m3, into the first HEAD_DIM bytes of its slot.
    A program holds the whole key before it stores, so the overlap with its source is safe.
    USE_PDL: release the dependents first, then wait, before any load; a dependent's own wait
    covers this launch and, through it, the writer in front of it."""
    if USE_PDL:
        tl.extra.cuda.gdc_launch_dependents()
        tl.extra.cuda.gdc_wait()
    row_ids = tl.program_id(0) * BLOCK_R + tl.arange(0, BLOCK_R)
    row_ok = row_ids < rows
    logical = tl.load(logical_positions + row_ids, mask=row_ok, other=0).to(tl.int64)
    anchored = row_ok & (logical >= 0) & ((logical + 1) % COMPRESS_RATIO == 0)
    requests = tl.load(request_indices + row_ids, mask=row_ok, other=0).to(tl.int64)
    pages = tl.load(
        block_table + requests * block_table_stride_request + logical // TOKENS_PER_BLOCK,
        mask=anchored,
        other=0,
    ).to(tl.int64)
    slots = logical % TOKENS_PER_BLOCK
    dims = tl.arange(0, HEAD_DIM)
    keys = tl.load(
        cache16 + pages[:, None] * stride16_page + slots[:, None] * stride16_token + dims[None, :],
        mask=anchored[:, None],
        other=0.0,
    )
    tl.store(
        cache8 + pages[:, None] * stride8_page + slots[:, None] * stride8_token + dims[None, :],
        keys.to(tl.float8e4nv),
        mask=anchored[:, None],
    )


def _fp8_view(index_cache: torch.Tensor) -> torch.Tensor:
    """The plane as e4m3 bytes: `[pages, tokens, 1, 2 * dim]`."""
    return index_cache.view(torch.float8_e4m3fn)


def index_anchors_to_fp8(
    index_cache: torch.Tensor,
    block_table: torch.Tensor,
    request_indices: torch.Tensor,
    logical_positions: torch.Tensor,
    tokens_per_block: int,
    compress_ratio: int,
    use_pdl: bool = False,
) -> None:
    """Convert the anchors `logical_positions` names (rows with `(L + 1) % ratio == 0`), which
    a compression just wrote as bf16, to the fp8 layout. Call once per compression of a row.
    `use_pdl` launches with PDL; the kernel releases its dependents and then waits at its top."""
    rows = logical_positions.shape[0]
    if rows == 0:
        return
    cache8 = _fp8_view(index_cache)
    block_r = 4
    _anchors_to_fp8_kernel[(triton.cdiv(rows, block_r),)](
        index_cache,
        cache8,
        block_table,
        request_indices,
        logical_positions,
        rows,
        index_cache.stride(0),
        index_cache.stride(1),
        cache8.stride(0),
        cache8.stride(1),
        block_table.stride(0),
        HEAD_DIM=index_cache.shape[3],
        TOKENS_PER_BLOCK=tokens_per_block,
        COMPRESS_RATIO=compress_ratio,
        BLOCK_R=block_r,
        USE_PDL=use_pdl,
        num_warps=4,
        launch_pdl=use_pdl,
    )


@triton.jit
def _qsa_index_scores_fp8_kernel(
    q,
    cache8,
    block_table,
    query_positions,
    request_indices,
    output,
    score_scale,
    rows,
    q_stride_row: tl.constexpr,
    q_stride_head: tl.constexpr,
    cache_stride_page: tl.constexpr,
    cache_stride_token: tl.constexpr,
    block_table_stride_request: tl.constexpr,
    output_stride_row: tl.constexpr,
    NUM_INDEX_HEADS: tl.constexpr,
    INDEX_HEAD_DIM: tl.constexpr,
    TOKENS_PER_BLOCK: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,
    MAX_COMPRESSED_BLOCKS: tl.constexpr,
    GROUP_ROWS: tl.constexpr,
    BLOCK_G: tl.constexpr,
    BLOCK_N: tl.constexpr,
    TILES_PER_PROGRAM: tl.constexpr,
):
    """The request-group kernel over e4m3 keys and an e4m3 query: a run of GROUP_ROWS rows a
    program; when the run names one request one key gather feeds every row's heads, otherwise
    each row gathers its own request's keys. Scores are written at each row's visible blocks
    only."""
    BLOCK_H: tl.constexpr = BLOCK_G * NUM_INDEX_HEADS
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

    dim_offsets = tl.arange(0, INDEX_HEAD_DIM)
    columns = tl.arange(0, BLOCK_H)
    column_slot = columns // NUM_INDEX_HEADS
    column_rows = group * GROUP_ROWS + column_slot
    column_ok = (column_slot < GROUP_ROWS) & (column_rows < rows)
    query_values = tl.load(
        q
        + column_rows[None, :] * q_stride_row
        + dim_offsets[:, None]
        + (columns % NUM_INDEX_HEADS)[None, :] * q_stride_head,
        mask=column_ok[None, :],
        other=0.0,
    )
    block_offsets = tl.arange(0, BLOCK_N)
    for tile in tl.range(first_tile, last_tile):
        block_columns = tile * BLOCK_N + block_offsets
        valid_blocks = (block_columns < MAX_COMPRESSED_BLOCKS) & (block_columns < max_visible)
        anchor_positions = block_columns * COMPRESS_RATIO + COMPRESS_RATIO - 1
        logical_pages = anchor_positions // TOKENS_PER_BLOCK
        token_in_page = anchor_positions % TOKENS_PER_BLOCK
        if shared:
            physical_pages = tl.load(
                block_table + lead * block_table_stride_request + logical_pages,
                mask=valid_blocks,
                other=0,
            ).to(tl.int64)
            keys = tl.load(
                cache8
                + physical_pages[:, None] * cache_stride_page
                + token_in_page[:, None] * cache_stride_token
                + dim_offsets[None, :],
                mask=valid_blocks[:, None],
                other=0.0,
                eviction_policy="evict_first",
            )
            per_head_scores = tl.dot(keys, query_values, out_dtype=tl.float32)
        else:
            per_head_scores = tl.zeros((BLOCK_N, BLOCK_H), dtype=tl.float32)
            for slot in tl.static_range(GROUP_ROWS):
                request = tl.sum(tl.where(slots == slot, requests, 0), axis=0)
                physical_pages = tl.load(
                    block_table + request * block_table_stride_request + logical_pages,
                    mask=valid_blocks,
                    other=0,
                ).to(tl.int64)
                keys = tl.load(
                    cache8
                    + physical_pages[:, None] * cache_stride_page
                    + token_in_page[:, None] * cache_stride_token
                    + dim_offsets[None, :],
                    mask=valid_blocks[:, None],
                    other=0.0,
                )
                per_head_scores = tl.where(
                    (column_slot == slot)[None, :],
                    tl.dot(keys, query_values, out_dtype=tl.float32),
                    per_head_scores,
                )
        pairs = tl.sum(
            tl.reshape(
                tl.maximum(per_head_scores, 0.0), (BLOCK_N, BLOCK_G, NUM_INDEX_HEADS // 2, 2)
            ),
            axis=3,
        )
        scores = tl.sum(pairs, axis=2)
        tl.store(
            output + group_rows[None, :] * output_stride_row + block_columns[:, None],
            scores * score_scale,
            mask=(block_columns[:, None] < visible[None, :]) & row_ok[None, :],
        )


def _qsa_index_scores_fp8(
    q: torch.Tensor,
    index_cache: torch.Tensor,
    block_table: torch.Tensor,
    query_positions: torch.Tensor,
    request_indices: torch.Tensor,
    tokens_per_block: int,
    compress_ratio: int,
    only_visible_blocks: bool,
    context_rows: bool,
    rows_per_request: int,
) -> torch.Tensor:
    """One launch of the fp8 kernel: request groups of `rows_per_request` rows (4 rows of one
    request for a context call), one key gather a group where its rows name one request."""
    if index_cache.ndim != 4 or index_cache.shape[2] != 1:
        raise ValueError("QSA index cache must have shape [pages, tokens, 1, dim]")
    if block_table.ndim != 2 or block_table.dtype != torch.int32 or block_table.stride(1) != 1:
        raise ValueError("QSA index scoring requires a two-dimensional int32 block table")
    rows, num_index_heads, index_head_dim = q.shape
    if index_cache.shape[3] != index_head_dim or index_cache.dtype != torch.bfloat16:
        raise ValueError("fp8 index keys need a bf16-sized index plane of the query's dimension")
    if num_index_heads % 2 or num_index_heads > 8:
        raise ValueError("fp8 index scoring serves 2, 4 or 8 index heads")
    max_compressed_blocks = block_table.shape[1] * tokens_per_block // compress_ratio
    if only_visible_blocks:
        output = torch.empty((rows, max_compressed_blocks), dtype=torch.float32, device=q.device)
    else:
        output = torch.full(
            (rows, max_compressed_blocks), float("-inf"), dtype=torch.float32, device=q.device
        )
    if rows == 0:
        return output
    q8 = q.to(torch.float8_e4m3fn).contiguous()
    cache8 = _fp8_view(index_cache)
    # A context call's consecutive rows are one request's: 4 share a key gather (16 rows of
    # per-row fallback dots exceed the tensor memory a program has).
    group_rows = rows_per_request if rows_per_request > 1 else 4 if context_rows else 1
    if group_rows > 8:
        raise ValueError("fp8 index scoring serves up to 8 rows a request")
    groups = triton.cdiv(rows, group_rows)
    tiles = 1 if groups <= 1 else 2 if groups <= 4 else 4
    grid = (groups, triton.cdiv(max_compressed_blocks, _GROUP_BLOCK_N * tiles))
    _qsa_index_scores_fp8_kernel[grid](
        q8,
        cache8,
        block_table,
        query_positions,
        request_indices,
        output,
        index_head_dim**-0.5,
        rows,
        q8.stride(0),
        q8.stride(1),
        cache8.stride(0),
        cache8.stride(1),
        block_table.stride(0),
        output.stride(0),
        NUM_INDEX_HEADS=num_index_heads,
        INDEX_HEAD_DIM=index_head_dim,
        TOKENS_PER_BLOCK=tokens_per_block,
        COMPRESS_RATIO=compress_ratio,
        MAX_COMPRESSED_BLOCKS=max_compressed_blocks,
        GROUP_ROWS=group_rows,
        BLOCK_G=max(triton.next_power_of_2(group_rows), 16 // num_index_heads, 1),
        BLOCK_N=_GROUP_BLOCK_N,
        TILES_PER_PROGRAM=tiles,
        num_warps=4,
    )
    return output


@triton.jit(do_not_specialize=["rows"])
def _qsa_index_scores_fp8_ctx_tile_kernel(
    q,
    cache8,
    block_table,
    query_positions,
    request_indices,
    output,
    score_scale,
    rows,
    q_stride_row: tl.constexpr,
    q_stride_head: tl.constexpr,
    cache_stride_page: tl.constexpr,
    cache_stride_token: tl.constexpr,
    block_table_stride_request: tl.constexpr,
    output_stride_row: tl.constexpr,
    NUM_INDEX_HEADS: tl.constexpr,
    INDEX_HEAD_DIM: tl.constexpr,
    TOKENS_PER_BLOCK: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,
    MAX_COMPRESSED_BLOCKS: tl.constexpr,
    TILE_ROWS: tl.constexpr,
    BLOCK_N: tl.constexpr,
    TILES_PER_PROGRAM: tl.constexpr,
):
    """Context rows over e4m3 keys and an e4m3 query. Program (t, c) takes row tile t (rows
    `t * TILE_ROWS ..`, up to TILE_ROWS consecutive rows) and the column tiles
    `c * TILES_PER_PROGRAM ..`. The rows of one request share each key gather; a tile that
    crosses a request boundary scores its requests one after the other. Each score is `_qsa_index_scores_fp8_kernel`'s:
    the per-head fp8 dot, ReLU, the pairwise head sum, the scale, written at each row's visible
    blocks only."""
    BLOCK_H: tl.constexpr = TILE_ROWS * NUM_INDEX_HEADS
    slots = tl.arange(0, TILE_ROWS)
    first_row = tl.program_id(0) * TILE_ROWS
    count = rows - first_row
    tile_rows = first_row + slots
    row_ok = slots < count
    requests = tl.load(request_indices + tile_rows, mask=row_ok, other=-1).to(tl.int64)
    positions = tl.load(query_positions + tile_rows, mask=row_ok, other=0)
    visible = tl.where(
        row_ok, tl.minimum((positions + 1) // COMPRESS_RATIO, MAX_COMPRESSED_BLOCKS), 0
    )
    first_tile = tl.program_id(1) * TILES_PER_PROGRAM
    if first_tile * BLOCK_N >= tl.max(visible, axis=0):
        return

    dim_offsets = tl.arange(0, INDEX_HEAD_DIM)
    # Dot column c is head c % heads of the tile's row c // heads.
    columns = tl.arange(0, BLOCK_H)
    column_slot = columns // NUM_INDEX_HEADS
    query_values = tl.load(
        q
        + (first_row + column_slot)[None, :] * q_stride_row
        + dim_offsets[:, None]
        + (columns % NUM_INDEX_HEADS)[None, :] * q_stride_head,
        mask=(column_slot < count)[None, :],
        other=0.0,
    )
    # A row leads a group when it is the tile's first or names another request than the row
    # before it; a list tile has one group.
    previous = tl.load(request_indices + tile_rows - 1, mask=row_ok & (tile_rows > 0), other=-1).to(
        tl.int64
    )
    leaders = row_ok & ((slots == 0) | (requests != previous))
    pending = row_ok
    block_offsets = tl.arange(0, BLOCK_N)
    for _ in tl.range(0, tl.sum(leaders.to(tl.int32), axis=0)):
        leader = tl.min(tl.where(pending, slots, TILE_ROWS), axis=0)
        request = tl.max(tl.where(slots == leader, requests, 0), axis=0)
        group = pending & (requests == request)
        pending = pending & (requests != request)
        group_visible = tl.max(tl.where(group, visible, 0), axis=0)
        last_tile = tl.minimum(first_tile + TILES_PER_PROGRAM, tl.cdiv(group_visible, BLOCK_N))
        for tile in tl.range(first_tile, last_tile):
            block_columns = tile * BLOCK_N + block_offsets
            valid_blocks = block_columns < group_visible
            anchor_positions = block_columns * COMPRESS_RATIO + COMPRESS_RATIO - 1
            logical_pages = anchor_positions // TOKENS_PER_BLOCK
            token_in_page = anchor_positions % TOKENS_PER_BLOCK
            physical_pages = tl.load(
                block_table + request * block_table_stride_request + logical_pages,
                mask=valid_blocks,
                other=0,
            ).to(tl.int64)
            key_ptrs = (
                cache8
                + physical_pages[:, None] * cache_stride_page
                + token_in_page[:, None] * cache_stride_token
                + dim_offsets[None, :]
            )
            keys = tl.load(
                key_ptrs, mask=valid_blocks[:, None], other=0.0, eviction_policy="evict_first"
            )
            per_head_scores = tl.dot(keys, query_values, out_dtype=tl.float32)
            # The fp8 kernel's head sum: (h0 + h1) + (h2 + h3), pairs, then the pairs.
            pairs = tl.sum(
                tl.reshape(
                    tl.maximum(per_head_scores, 0.0),
                    (BLOCK_N, TILE_ROWS, NUM_INDEX_HEADS // 2, 2),
                ),
                axis=3,
            )
            scores = tl.sum(pairs, axis=2)
            tl.store(
                output + tile_rows[None, :] * output_stride_row + block_columns[:, None],
                scores * score_scale,
                mask=group[None, :] & (block_columns[:, None] < visible[None, :]),
            )


def _qsa_index_scores_fp8_ctx_tile(
    q: torch.Tensor,
    index_cache: torch.Tensor,
    block_table: torch.Tensor,
    query_positions: torch.Tensor,
    request_indices: torch.Tensor,
    tokens_per_block: int,
    compress_ratio: int,
) -> torch.Tensor:
    """A prefill-sized context call's scores (visible blocks only): the q cast and one launch of
    the ctx-tile kernel."""
    rows, num_index_heads, index_head_dim = q.shape
    max_compressed_blocks = block_table.shape[1] * tokens_per_block // compress_ratio
    output = torch.empty((rows, max_compressed_blocks), dtype=torch.float32, device=q.device)
    q8 = q.to(torch.float8_e4m3fn).contiguous()
    cache8 = _fp8_view(index_cache)
    grid = (
        triton.cdiv(rows, _CTX_TILE_ROWS),
        triton.cdiv(max_compressed_blocks, _CTX_TILE_BLOCK_N * _CTX_TILE_TILES),
    )
    _qsa_index_scores_fp8_ctx_tile_kernel[grid](
        q8,
        cache8,
        block_table,
        query_positions,
        request_indices,
        output,
        index_head_dim**-0.5,
        rows,
        q8.stride(0),
        q8.stride(1),
        cache8.stride(0),
        cache8.stride(1),
        block_table.stride(0),
        output.stride(0),
        NUM_INDEX_HEADS=num_index_heads,
        INDEX_HEAD_DIM=index_head_dim,
        TOKENS_PER_BLOCK=tokens_per_block,
        COMPRESS_RATIO=compress_ratio,
        MAX_COMPRESSED_BLOCKS=max_compressed_blocks,
        TILE_ROWS=_CTX_TILE_ROWS,
        BLOCK_N=_CTX_TILE_BLOCK_N,
        TILES_PER_PROGRAM=_CTX_TILE_TILES,
        num_warps=_CTX_TILE_WARPS,
        num_stages=_CTX_TILE_STAGES,
    )
    return output


def _takes_ctx_tile(q, only_visible_blocks: bool, context_rows: bool, rows_per_request: int):
    """Whether a call is a prefill-sized context call the ctx-tile kernel serves."""
    return (
        context_rows
        and only_visible_blocks
        and rows_per_request == 1
        and q.shape[0] > _DECODE_ROW_LIMIT
        and q.shape[1] in (2, 4, 8)
    )


def qsa_index_scores_fp8_launch(
    q: torch.Tensor,
    index_cache: torch.Tensor,
    block_table: torch.Tensor,
    query_positions: torch.Tensor,
    request_indices: torch.Tensor,
    tokens_per_block: int,
    compress_ratio: int,
    only_visible_blocks: bool,
    context_rows: bool,
    rows_per_request: int = 1,
) -> torch.Tensor:
    """The scores `[rows, max_compressed_blocks]` fp32 of `q` against the e4m3 anchors: a
    prefill-sized context call on the ctx-tile kernel (bitwise the request-group kernel's
    scores), every other call on the request-group kernel."""
    if _takes_ctx_tile(q, only_visible_blocks, context_rows, rows_per_request):
        return _qsa_index_scores_fp8_ctx_tile(
            q, index_cache, block_table, query_positions, request_indices, tokens_per_block,
            compress_ratio,
        )  # fmt: skip
    return _qsa_index_scores_fp8(
        q,
        index_cache,
        block_table,
        query_positions,
        request_indices,
        tokens_per_block,
        compress_ratio,
        only_visible_blocks,
        context_rows,
        rows_per_request,
    )


def _scores_spec(**overrides) -> dict:
    """A cell spec: the checkpoint's index geometry (4 heads x 128, 32 tokens a page, ratio 4)
    with q a head slice of the `[rows, 6, 128]` index scratch, int64 positions."""
    spec = dict(
        heads=4, head_dim=128, tokens_per_block=32, compress_ratio=4, rows_per_request=1,
        context_rows=False, only_visible_blocks=True, table=516,
    )  # fmt: skip
    spec.update(overrides)
    return spec


class QsaIndexScoresFp8(OpWrapper):
    """`qsa_paged_index_scores` over e4m3 anchors (the module docstring): per row, the visible
    blocks' scores `scale * sum_h relu(q8_h . k8)` with `q8` the e4m3-rounded query, `k8` the
    group's e4m3 anchor (`index_anchors_to_fp8`), `scale = dim ** -0.5`; with
    `only_visible_blocks` false the columns past a row's visible blocks are -inf, else they are
    left unwritten."""

    ARCHS = SERVED_ARCHS

    CELLS: tuple[Cell, ...] = (
        Cell(
            why="dep MTP-3 verify: 12 requests x 4 rows over the 8196-page table, one key "
            "gather a request",
            spec=_scores_spec(
                requests=tuple((20000 + 9973 * r, 4) for r in range(12)),
                rows_per_request=4,
                table=8196,
            ),  # fmt: skip
        ),
        Cell(
            why="dep draft rows: 52 requests x 1 row, ~2k-250k token contexts",
            spec=_scores_spec(requests=tuple((2000 + 4800 * r, 1) for r in range(52)), table=8196),
        ),
        Cell(
            why="a 2048-row context chunk of one 2048-token prompt: the ctx-tile kernel",
            spec=_scores_spec(requests=((0, 2048),), context_rows=True),
        ),
        Cell(
            why="300 context rows over two requests (a ctx tile crossing them), -inf padding off",
            spec=_scores_spec(requests=((5000, 140), (100, 160)), context_rows=True),
        ),
        Cell(
            why="decode rows with -inf past the visible blocks (not only visible)",
            spec=_scores_spec(
                requests=tuple((300 + 997 * r, 1) for r in range(8)), only_visible_blocks=False
            ),
        ),
    )

    note = """
    Call sites: the NVFP4 target under `TRTLLM_QWEN4_EXP_INDEX_KEY_FP8` (every index score call,
    context and generation, trunk and draft); `index_anchors_to_fp8` after every anchor writer.
    Preconditions: `q` bf16 `[rows, 2 | 4 | 8 heads, dim]` (unit dim stride); the plane bf16-sized
    `[pages, tokens, 1, dim]` with e4m3 anchors; int32 row-contiguous block table; positions and
    request indices `[rows]`; up to 8 rows a request group.
    """

    def __call__(
        self,
        q,
        index_cache,
        block_table,
        query_positions,
        request_indices,
        tokens_per_block: int,
        compress_ratio: int,
        only_visible_blocks: bool,
        context_rows: bool,
        rows_per_request: int = 1,
    ) -> torch.Tensor:
        return qsa_index_scores_fp8_launch(
            q,
            index_cache,
            block_table,
            query_positions,
            request_indices,
            tokens_per_block,
            compress_ratio,
            only_visible_blocks,
            context_rows,
            rows_per_request,
        )

    def reference(
        self,
        q,
        index_cache,
        block_table,
        query_positions,
        request_indices,
        tokens_per_block: int,
        compress_ratio: int,
        only_visible_blocks: bool,
        context_rows: bool,
        rows_per_request: int = 1,
    ) -> torch.Tensor:
        """float64 over the e4m3 operands, -inf past each row's visible blocks (the kernel
        leaves those columns unwritten with `only_visible_blocks`: compare visible columns)."""
        rows, heads, dim = q.shape
        blocks = block_table.shape[1] * tokens_per_block // compress_ratio
        out = torch.full((rows, blocks), float("-inf"), dtype=torch.float64, device=q.device)
        q8 = q.to(torch.float8_e4m3fn).double()
        cache8 = _fp8_view(index_cache)
        for r in range(rows):
            visible = (int(query_positions[r]) + 1) // compress_ratio
            if visible <= 0:
                continue
            anchors = torch.arange(visible, device=q.device) * compress_ratio + compress_ratio - 1
            pages = block_table[int(request_indices[r]), anchors // tokens_per_block].long()
            keys = cache8[pages, anchors % tokens_per_block, 0, :dim].double()
            per_head = torch.relu(keys @ q8[r].T)
            out[r, :visible] = per_head.sum(dim=1) * dim**-0.5
        return out

    def is_valid(
        self,
        q,
        index_cache,
        block_table,
        query_positions,
        request_indices,
        tokens_per_block: int,
        compress_ratio: int,
        only_visible_blocks: bool,
        context_rows: bool,
        rows_per_request: int = 1,
    ) -> None:
        assert q.dtype == torch.bfloat16 and q.dim() == 3 and q.shape[1] in (2, 4, 8)
        assert index_cache.dim() == 4 and index_cache.shape[2] == 1
        assert index_cache.dtype == torch.bfloat16 and index_cache.shape[3] == q.shape[2]
        assert block_table.dtype == torch.int32 and block_table.stride(1) == 1
        assert query_positions.shape == (q.shape[0],) and request_indices.shape == (q.shape[0],)
        assert 1 <= int(rows_per_request) <= 8


qsa_index_scores_fp8 = QsaIndexScoresFp8()
