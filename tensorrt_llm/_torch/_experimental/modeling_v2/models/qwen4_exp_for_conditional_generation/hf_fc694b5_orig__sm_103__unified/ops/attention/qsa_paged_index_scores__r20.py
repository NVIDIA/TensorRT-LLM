# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""QSA compressed-block index scoring, read straight out of the paged index-K side cache."""

from __future__ import annotations

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

# Approved exception: QSA's selection path is Triton called as Python, with no
# torch.ops boundary. The engine's own sparse hook (`select_qsa_paged_tokens`
# in `_torch/attention/backends/sparse/qsa/module.py`) calls this same callable.
from tensorrt_llm._torch.attention.backends.sparse.qsa.kernels import (
    triton_qsa_paged_index_scores as _triton_qsa_paged_index_scores,
)

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


#: The serving runtime dumps: an [8, 260] table on the warm-up capture path and
#: a [32, 260] one under the mmmu gate's mixed steps.
_WIDTH = 260


#: The same plane under an NVFP4 KV pool: the QSA position page joins each physical
#: page after the index buffers (`page_pad` elements), so the trunk's view is
#: `[pages, 32, 1, 128]` at page stride 12 x 4096 + 192 = 49344.
_TRUNK_POOL_NVFP4 = dict(pool_buffers=12, page_pad=192)

#: Core's plane (bf16 K/V) on every member of the 4-GPU target, as the stock manager allocates it:
#: the 12 attention layers' `[32, 1, 128]` INDEX_KEY buffers coalesce into one pool, layer k at
#: byte 8192 k of a 98304-byte page, so one layer's view is `[pages, 32, 1, 128]` at page stride
#: 12 x 4096 = 49152. Measured on the stock `Qwen4ExpHybridCacheManager` at tep4 and dep4 rank 0.
#: The cells take the last layer's plane.
_TRUNK_POOL_BF16 = dict(pool_buffers=12, page_pad=0)

#: Pages in one rank's pool: the stock engine's `maxNumBlocks` at core's configuration (bf16 K/V
#: and recurrent state, KV fraction 0.6, max_batch_size 128, max_seq_len 9216, the n-gram table
#: host-resident). The largest measured per member; the index plane shares the attention pages.
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


def _decode_positions(requests: int, last: int, rows_per_request: int = 1) -> tuple[int, ...]:
    """Each request's first row position; the first request's last row sits at `last`."""
    first = last - rows_per_request + 1
    return (first,) + tuple(1 + (7919 * g) % first for g in range(1, requests))


class QsaPagedIndexScores(OpWrapper):
    """Score every causally visible compressed block of each query row.

    Wraps `tensorrt_llm._torch.attention.backends.sparse.qsa.kernels.
    triton_qsa_paged_index_scores` (one Triton launch after allocating the
    output). For row `r` and compressed block `b` of request
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

    `context_rows` is a layout declaration ("consecutive rows mostly share a
    request"), not a phase one: with more than 256 rows and an index-head count
    that divides 128 into more than one row (1, 2, 4, ..., 64) it selects a
    query-tiled kernel sharing one key gather across `128 // heads` rows;
    otherwise the row-wise kernel runs. Each cell names the kernel it was measured on. The
    two are not bitwise equal.

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
                branch="row_wise",
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
                branch="row_wise",
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
        # The 4-GPU target's sites over one rank's measured pool: the call-site flags each
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
                branch="row_wise",
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
                branch="row_wise",
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
    )

    note = """
    Source: the engine's Triton kernels beside the wrapped callable
    (`_qsa_paged_index_scores_kernel`, `_qsa_paged_index_scores_qtile_kernel`),
    JIT-compiled for the running device on first call; the cells ran that JIT
    path on sm_103.

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

    `only_visible_blocks=True` really leaves garbage (measured, not pinned by a
    test): with a poisoned allocation freed just before the call, the whole
    unscored tail came back poisoned. At 8191 rows from position 0 over 2080 columns that is a full
    32-column band (2048..2079) for every row, 8.65M cells. A consumer that
    scans `[row_start, row_end)` -- `qsa_topk_select` -- is unaffected.

    fp32 operands are accepted but `tl.dot` rounds them to a tensor-core
    format: 8.1e-3 max abs against an fp32 reference on scores to 7.8 (measured),
    outside the test's band. No cell covers fp32; the side cache is bf16.

    CUDA graphs: the test captures the decode launch (`use_pdl`, `prefetch`) at
    4 rows and replays it after `q` and the anchor keys were rewritten in
    place; every replay matches an eager call bitwise. The same holds captured
    at each measured-pool decode and verify cell (1, 128 and 1024 rows at tep4,
    32 at dep4, with their sites' flags) after q and every mapped key changed.

    Measured on GB300 at the measured-pool cells: at most 2.9e-6 against the
    reference (scores to 11.3), relative L2 below 6.4e-8. A cell's peak is 24.5
    GiB at tep4 (the pool alone is 23.7) and 11.2 GiB at dep4, each under a second.

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
    certified on the row-wise kernel only, at the tp1 cell's small pool.

    The 4-GPU target's cells run over one rank's measured pool, core's coalesced
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

    Certified: up to 8192 rows (one request's 8192-row context, fresh or ending at
    position 9215, on the query-tile kernel; 1024 verify rows on the row-wise
    kernel with 8 tiles a program), tables of 128 x 292 and 128 x 516 pages,
    positions to 16383. Not certified: more than 8192 rows, tables wider than 516
    pages (the tune's 262144-token envelope), any other head geometry, compress
    ratio, page size or operand dtype, a context step's mixed request rows on the
    pool layout, the fp8/NVFP4 side-planes stride at a measured pool, and the
    `-inf` padded mode.
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
    ) -> torch.Tensor:
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
    ) -> torch.Tensor:
        """fp32 scores, `-inf` past each row's bound -- or NaN there, meaning
        "unspecified", when `only_visible_blocks` is True. `context_rows` is
        ignored: it picks a kernel, not a result.

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
