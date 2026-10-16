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


def _small(rows: int, requests: int, **overrides) -> dict:
    """The narrow-table cells: random positions over an 8-page table, rows
    grouped by request, each request's table row fully mapped, a contiguous q.
    Eight table rows, as in the serving dump, whatever the request count."""
    base = dict(layout="random", rows=rows, requests=requests, slots=8, pages=8)
    base.update(q_layout="packed", positions_dtype=torch.int32)
    base.update(only_visible_blocks=False, context_rows=False)
    base["branch"] = "row_wise"
    base.update(overrides)
    return _spec(**base)


#: The serving runtime dumps: an [8, 260] table on the warm-up capture path and
#: a [32, 260] one under the mmmu gate's mixed steps.
_WIDTH = 260

_FREE_HEADS = tuple(
    Cell(
        why=f"free axis, 64 rows: {h} index heads x dim {d}",
        spec=_small(64, 3, heads=h, head_dim=d),
    )
    for d in (64, 128)
    for h in (1, 2, 3, 4, 8, 16)
    if (h, d) != (4, 128)
)
_FREE_PAGING = tuple(
    Cell(
        why=f"free axis, 48 rows: compress {c} x {t} tokens per block",
        spec=_small(48, 3, compress_ratio=c, tokens_per_block=t),
    )
    for t in (16, 32, 64)
    for c in (2, 4, 8, 16)
    if (c, t) != (4, 32)
)


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
    that divides 16 into more than one row (1, 2, 4, 8) it selects a
    query-tiled kernel sharing one key gather across rows; otherwise the
    row-wise kernel runs. Each cell names the kernel it was measured on. The
    two agree to one fp32 ulp, not bitwise.

    Fusion boundary: the table walk, the gather, the dots, relu, head sum,
    scale and causal bound are inside. Producing `q` (norm, RoPE), writing and
    compressing the index cache, and everything after the scores are not.
    Returns a fresh fp32 `[rows, columns]`; nothing is written in place.
    """

    ARCHS = frozenset({Arch.SM_103})

    CELLS: tuple[Cell, ...] = (
        # ---- the target's call shapes: [*, 260] table, 2080 columns -------
        Cell(
            why="decode: 1 row deep in history, [32, 260] table, visible-only",
            spec=_spec(
                layout="decode",
                gen_positions=(6000,),
                rows_per_request=1,
                slots=32,
                pages=_WIDTH,
                only_visible_blocks=True,
                context_rows=False,
                branch="row_wise",
            ),
        ),
        Cell(
            why="decode MTP: 4 requests x 2 rows, visible-only",
            spec=_spec(
                layout="decode",
                gen_positions=(7000, 5000, 4200, 2100),
                rows_per_request=2,
                slots=32,
                pages=_WIDTH,
                only_visible_blocks=True,
                context_rows=False,
                branch="row_wise",
            ),
        ),
        Cell(
            why="decode MTP: 4 requests x 2 rows, -inf padded mode",
            spec=_spec(
                layout="decode",
                gen_positions=(7000, 5000, 4200, 2100),
                rows_per_request=2,
                slots=32,
                pages=_WIDTH,
                only_visible_blocks=False,
                context_rows=False,
                branch="row_wise",
            ),
        ),
        Cell(
            why="decode MTP: every one of 32 slots x 2 rows = 64 rows",
            spec=_spec(
                layout="decode",
                gen_positions=tuple(100 + (499 * g) % 8000 for g in range(32)),
                rows_per_request=2,
                slots=32,
                pages=_WIDTH,
                only_visible_blocks=True,
                context_rows=False,
                branch="row_wise",
            ),
        ),
        Cell(
            why="capture: 8191 rows, one request from 0, visible-only",
            spec=_spec(
                layout="capture",
                lengths=(8191,),
                slots=8,
                pages=_WIDTH,
                only_visible_blocks=True,
                context_rows=True,
                branch="query_tile",
            ),
        ),
        Cell(
            why="capture: 8191 rows, one request from 0, -inf padded",
            spec=_spec(
                layout="capture",
                lengths=(8191,),
                slots=8,
                pages=_WIDTH,
                only_visible_blocks=False,
                context_rows=True,
                branch="query_tile",
            ),
        ),
        Cell(
            why="capture: 8192 rows = 2 x 4096, visible-only",
            spec=_spec(
                layout="capture",
                lengths=(4096, 4096),
                slots=8,
                pages=_WIDTH,
                only_visible_blocks=True,
                context_rows=True,
                branch="query_tile",
            ),
        ),
        Cell(
            why="capture: 8192 rows = 2 x 4096, -inf padded",
            spec=_spec(
                layout="capture",
                lengths=(4096, 4096),
                slots=8,
                pages=_WIDTH,
                only_visible_blocks=False,
                context_rows=True,
                branch="query_tile",
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
        Cell(
            why="capture: 8191 rows over 8 ragged requests, -inf padded",
            spec=_spec(
                layout="capture",
                lengths=(1024, 512, 2048, 256, 1500, 700, 1151, 1000),
                slots=8,
                pages=_WIDTH,
                only_visible_blocks=False,
                context_rows=True,
                branch="query_tile",
            ),
        ),
        Cell(
            why="far end: 8191 rows at position 8319, all 2080 columns visible",
            spec=_spec(
                layout="far",
                rows=8191,
                slots=8,
                pages=_WIDTH,
                only_visible_blocks=False,
                context_rows=True,
                branch="query_tile",
            ),
        ),
        Cell(
            why="mixed step: 195-token chunk + 31 MTP decodes = 257 rows",
            spec=_spec(
                layout="mixed",
                requests=32,
                context_lens=(195,),
                slots=32,
                pages=_WIDTH,
                only_visible_blocks=True,
                context_rows=True,
                branch="query_tile",
            ),
        ),
        Cell(
            why="mixed step: 4 chunks + 28 MTP decodes = 1334 rows",
            spec=_spec(
                layout="mixed",
                requests=32,
                context_lens=(900, 200, 150, 28),
                slots=32,
                pages=_WIDTH,
                only_visible_blocks=True,
                context_rows=True,
                branch="query_tile",
            ),
        ),
        Cell(
            why="mixed step: 4 chunks + 28 MTP decodes = 8190 rows",
            spec=_spec(
                layout="mixed",
                requests=32,
                context_lens=(8000, 100, 30, 4),
                slots=32,
                pages=_WIDTH,
                only_visible_blocks=True,
                context_rows=True,
                branch="query_tile",
            ),
        ),
        Cell(
            why="mixed step at 162 rows: context_rows=True stays row-wise",
            spec=_spec(
                layout="mixed",
                requests=32,
                context_lens=(100,),
                slots=32,
                pages=_WIDTH,
                only_visible_blocks=True,
                context_rows=True,
                branch="row_wise",
            ),
        ),
        # ---- the branch rule's edges (8-page table) ------------------------
        Cell(
            why="1 row with context_rows=True: still row-wise", spec=_small(1, 4, context_rows=True)
        ),
        Cell(why="odd 17 rows, generation-sized grid", spec=_small(17, 4)),
        Cell(why="256 rows, context_rows=False: last decode-sized", spec=_small(256, 4)),
        Cell(
            why="256 rows, context_rows=True: at the threshold, row-wise",
            spec=_small(256, 4, context_rows=True),
        ),
        Cell(
            why="257 rows, context_rows=True: one past it, query-tiled",
            spec=_small(257, 6, context_rows=True, branch="query_tile"),
        ),
        Cell(why="257 rows, context_rows=False: prefill-sized row-wise", spec=_small(257, 6)),
        Cell(
            why="1024 rows, context_rows=True: query-tiled",
            spec=_small(1024, 6, context_rows=True, branch="query_tile"),
        ),
        Cell(why="1024 rows, context_rows=False: row-wise", spec=_small(1024, 6)),
        Cell(
            why="300 rows x 1 head: 16 rows per query tile",
            spec=_small(300, 5, heads=1, context_rows=True, branch="query_tile"),
        ),
        Cell(
            why="300 rows x 2 heads: 8 rows per query tile",
            spec=_small(300, 5, heads=2, context_rows=True, branch="query_tile"),
        ),
        Cell(
            why="300 rows x 8 heads: 2 rows per query tile",
            spec=_small(300, 5, heads=8, context_rows=True, branch="query_tile"),
        ),
        Cell(
            why="300 rows x 16 heads: one row per tile, row-wise",
            spec=_small(300, 5, heads=16, context_rows=True),
        ),
        Cell(
            why="300 rows x 3 heads: does not divide 16, row-wise",
            spec=_small(300, 5, heads=3, context_rows=True),
        ),
        Cell(
            why="300 rows x 5 heads: does not divide 16, row-wise",
            spec=_small(300, 5, heads=5, context_rows=True),
        ),
        Cell(
            why="query tile over rows shuffled across requests",
            spec=_small(320, 6, layout="shuffled", context_rows=True, branch="query_tile"),
        ),
        # ---- axes measured free --------------------------------------------
        Cell(why="fp16 q and index cache", spec=_small(64, 3, dtype=torch.float16)),
        Cell(
            why="int64 positions and int64 request indices",
            spec=_small(48, 3, positions_dtype=torch.int64, request_dtype=torch.int64),
        ),
        Cell(
            why="q as a head-dim slice of a [40, 4, 256] buffer",
            spec=_small(40, 3, q_layout="dim_slice"),
        ),
        Cell(
            why="1-page table: the narrowest legal score row",
            spec=_small(8, 2, pages=1, tokens_per_block=16),
        ),
        Cell(why="2-page table", spec=_small(8, 2, pages=2, tokens_per_block=16)),
        Cell(
            why="positions past the table: every column visible",
            spec=_small(32, 2, layout="past_table", pages=4),
        ),
        Cell(
            why="positions 0..7: rows before a complete group see nothing",
            spec=_small(8, 2, layout="arange"),
        ),
        *_FREE_HEADS,
        *_FREE_PAGING,
    )

    note = """
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

    The op validates its own metadata loudly (10 `ValueError`s, pinned by the
    test), so `is_valid` adds nothing.

    `only_visible_blocks=True` really leaves garbage: with a poisoned
    allocation freed just before the call, the whole unscored tail came back
    poisoned. At 8191 rows from position 0 over 2080 columns that is a full
    32-column band (2048..2079) for every row, 8.65M cells. A consumer that
    scans `[row_start, row_end)` -- `qsa_topk_select` -- is unaffected.

    fp32 operands are accepted but `tl.dot` rounds them to a tensor-core
    format: 8.1e-3 max abs against an fp32 reference on scores to 7.8, outside
    the test's band. No cell covers fp32; the side cache is bf16.

    CUDA graphs: captured at 8191 rows (visible-only), replayed, then replayed
    again after `q`, the cache and the table were overwritten in place; both
    matched the reference.

    Not certified: more than 8192 rows, tables wider than 260 pages, more than
    32 table rows, and any head geometry other than 4 x 128 at the 2080-column
    width. Hard limits: not probed; the largest certified extent is 8192 x
    2080 output elements.
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
