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
    )

    note = """
    Metadata consumed: none -- every input is an argument. What the target
    passes: `block_table` is `QSAMetadata.block_table` (raw
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
    place; every replay matches an eager call bitwise.

    `prefetch` (with `use_pdl` and `only_visible_blocks`, at generation-sized
    batches) reads the pages and compressed keys before the PDL wait and q after
    it. Sound only behind the decode step's own prologue
    (`qwen4_exp_attn_decode_prologue`), which writes no compressed key but the last
    visible block's, at a row whose position ends a group; that key is read again
    after the wait.

    Every cell runs `only_visible_blocks=True` (the target never passes False),
    4 index heads x 128, compress ratio 4, 32-token pages, bf16 operands, int64
    positions and int32 request indices. The NVFP4-pool page stride is
    certified on the row-wise kernel only.

    Not certified: more than 8191 rows, tables wider than 260 pages, more than
    32 table rows, any other head geometry, compress ratio, page size or
    operand dtype, a context step's mixed request rows, and the `-inf` padded
    mode. Hard limits: not probed; the largest certified extent is 8191 x 2080
    output elements.
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
