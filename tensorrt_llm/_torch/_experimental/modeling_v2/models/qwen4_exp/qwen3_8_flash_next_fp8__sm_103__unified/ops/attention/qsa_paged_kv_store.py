# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""QSA sparse path's paged K/V append: this step's rows into the HND main pool.

This target's copy of the family entry `models/qwen4_exp/ops/attention/qsa_paged_kv_store`."""

from __future__ import annotations

from typing import Tuple

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

from . import qsa_triton_kernels__stable_jit as _qsa_kernels

#: Main K/V geometry: k and v are head slices of the packed `[T, 24 + 2 + 2, 256]` QKV row;
#: `slots` x `width` is `qsa_block_table` (max_num_sequences x max_blocks_per_seq).
_CKPT = dict(
    hkv=2,
    head_dim=256,
    heads_q=24,
    tokens_per_block=32,
    slots=32,
    width=260,
    row_dtype=torch.bfloat16,
    cache_dtype=torch.bfloat16,
    request_dtype=torch.int32,
    position_dtype=torch.int64,
    hole=None,
    # Heads after v in each row: 27 for the fused projection row [q | k | v | gate | index].
    row_pad_heads=0,
)

_F8 = torch.float8_e4m3fn

_DECODE = ((0, 1), (31, 1), (32, 1), (33, 1), (100, 1), (255, 1), (7, 1), (64, 1))

#: 32 MTP7 verify blocks (8 rows each), some straddling a page edge.
_MTP7_32 = tuple((29 + 61 * i, 8) for i in range(32))

#: One rank's pool as `Qwen4ExpHybridCacheManager` lays it out: a page holds the K and V planes
#: of all 12 attention layers (24 role pages of `[hkv, 32, 256]`), so a layer's
#: `get_buffers(i, kv_layout="HND")` view steps 24 role pages a page; the cells drive layer 47
#: (`layer_slot` 11). The table is 128 x 292 at `max_seq_len` 9216.
_ENGINE = dict(_CKPT, pool="engine", role_pages=24, layer_slot=11, slots=128, width=292)

#: tp4 / tep4 rank: fused row `[q 6 | k 1 | v 1 | gate 6 | index 3]` (17 heads, stride 4352).
_TP4 = dict(_ENGINE, hkv=1, heads_q=6, row_pad_heads=9, row_stride=4352)

#: dep4 rank: tp1 row `[q 24 | k 2 | v 2 | gate 24 | index 3]` (55 heads, stride 14080), e4m3 pool.
_DEP4_FP8 = dict(_ENGINE, row_pad_heads=27, row_stride=14080, cache_dtype=torch.float8_e4m3fn)

#: tp2 / tep2 rank: fused row `[q 12 | k 1 | v 1 | gate 12 | index 3]` (29 heads, stride 7424).
_TP2 = dict(_ENGINE, hkv=1, heads_q=12, row_pad_heads=15, row_stride=7424)

#: dep4 rank over a bf16 pool.
_DEP4 = dict(_ENGINE, row_pad_heads=27, row_stride=14080)

#: dep4 with one-model MTP: the draft layer joins the trunk's manager (13 layers, 26 role pages).
_DEP4_MTP = dict(_DEP4, role_pages=26)

#: Attention pool pages per rank (`maxNumBlocks` less the recurrent slots) of the NVFP4
#: checkpoint's engines at KV fraction 0.6, `max_batch_size` 128, `max_seq_len` 9216; the FP8
#: cells reuse them at the same per-page shape.
PAGES_TEP4 = 257109
PAGES_DEP4_FP8 = 171362
PAGES_DEP4 = 119680

#: 128 decode rows: page-aligned, mid-page, the last position (9215), then a stride-577 walk.
_DECODE_128 = ((0, 1), (31, 1), (32, 1), (9215, 1)) + tuple(
    ((i * 577 + 45) % 9216, 1) for i in range(124)
)

#: An 8192-row context step: a chunk from mid-page to 9215, and a request's first 1023 tokens.
_PREFILL_8192 = ((2047, 7169), (0, 1023))

#: 128 MTP3 verify blocks (512 rows) ending at position 9215, at every offset in a page.
_MTP3_128 = tuple(((29 + 71 * i) % 9212, 4) for i in range(127)) + ((9212, 4),)


def _spec(segments, pages, **overrides) -> dict:
    """`segments` is `(first position, rows)` per request, in row order."""
    return {**_CKPT, "segments": segments, "pages": pages, **overrides}


def _engine(base: dict, member: str, segments, pages: int, **overrides) -> dict:
    """An engine-pool cell: `rows` is the step's row count, `member` whose pool `pages` is."""
    rows = sum(n for _, n in segments)
    return {
        **base,
        "member": member,
        "segments": segments,
        "pages": pages,
        "rows": rows,
        **overrides,
    }


class QsaPagedKvStore(OpWrapper):
    """Scatter every row's K and V into its request's page of the HND pool.

    One Triton launch over `(rows, Hkv)`. For row t::

        page = block_table[request_indices[t], logical_positions[t] // tokens_per_block]
        slot = logical_positions[t] % tokens_per_block
        k_cache[page, :, slot, :] = k[t]          v_cache[page, :, slot, :] = v[t]

    `logical_positions` is absolute, so prefill, decode and MTP rows are the same call. Values
    are converted once to the pool's dtype (an fp8-e4m3 pool takes no scale: readers use kv
    scale 1.0). A row that resolves to no slot (negative or out-of-table request, position or
    page column, or a table entry that is `-1` or past the pool) is skipped. Returns
    `(k_cache, v_cache)`, written in place.
    """

    ARCHS = frozenset({Arch.SM_103})

    CELLS: tuple[Cell, ...] = (
        Cell(
            why="decode: 8 rows off the fused projection row (row stride 14080), "
            "at aligned, mid and page-crossing positions, plus one row on an "
            "unallocated (-1) table entry that is skipped",
            spec=_spec(_DECODE, pages=64, row_pad_heads=27, hole="unallocated_page"),
        ),
        Cell(
            why="fp8 decode MTP7: 32 requests x 8 rows (256), blocks straddling pages",
            spec=_spec(_MTP7_32, pages=1200, slots=40, cache_dtype=_F8),
        ),
        # Engine pools at one rank's page count.
        Cell(
            why="tep4 decode, 1 row: hkv 1 off the 4352-wide row, position 9215, "
            "257109-page bf16 pool",
            spec=_engine(_TP4, "tep4", ((9215, 1),), PAGES_TEP4),
        ),
        Cell(
            why="tp4 context, 8192 rows (max_num_tokens): a mid-page chunk to 9215 and a "
            "request's first 1023",
            spec=_engine(_TP4, "tp4", _PREFILL_8192, PAGES_TEP4),
        ),
        Cell(
            why="tp2 decode, 128 rows (max_batch_size) off the 7424-wide row, aligned, mid "
            "and page-crossing",
            spec=_engine(_TP2, "tp2", _DECODE_128, PAGES_TEP4),
        ),
        Cell(
            why="dep4 MTP3 verify, 128 requests x 4 rows (512) to position 9215: hkv 2 off "
            "the 14080-wide row, 26 role pages a page (the draft layer's planes beside the "
            "trunk's)",
            spec=_engine(_DEP4_MTP, "dep4", _MTP3_128, PAGES_DEP4),
        ),
        Cell(
            why="dep4 kv_fp8 decode, 128 rows (max_batch_size): hkv 2 off the 14080-wide row "
            "into a 171362-page e4m3 pool",
            spec=_engine(_DEP4_FP8, "dep4", _DECODE_128, PAGES_DEP4_FP8),
        ),
        Cell(
            why="dep4 context, 8192 rows (max_num_tokens): hkv 2 off the 14080-wide row, a "
            "mid-page chunk to 9215 and a request's first 1023",
            spec=_engine(_DEP4, "dep4", _PREFILL_8192, PAGES_DEP4),
        ),
    )

    note = """
    Source: `qsa_triton_kernels__stable_jit._qsa_kv_store_stable_kernel` (Triton JIT).

    Preconditions:
      - destination slots are unique across the batch; two rows naming one (page, slot) both
        store, unordered. The engine's mapping cannot collide.
      - `request_indices` and `logical_positions` are contiguous (`is_valid`): the kernel gets
        no stride for them.
      - both are cast to int32 in the kernel, so positions must stay below 2^31 (the engine's
        stay below 2^14).
    Pool/table extents, the table's row stride and the row count are runtime ints, so one
    compiled kernel serves every pool and batch; CUDA-graph capturable, and a replay follows
    positions rewritten in place. Page ids are widened to int64 before the stride product.

    Not certified: duplicate destinations, multiple devices, a scaled fp8 pool, fp8 values past
    448, the packed-QKV-row layout over a bf16 pool or at one KV head, fp16 rows, geometry other
    than D 256 / Hkv 1-2 / 32-token pages, int32 positions or int64 request indices, head-major
    rows, sliced or transposed tables, `rows == 0`, unresolvable rows other than a `-1` entry
    or a negative position, positions past 9215, tables wider than 292 columns, and the FP8
    checkpoint's engines at their own page counts.
    """

    def __call__(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        request_indices: torch.Tensor,
        logical_positions: torch.Tensor,
        block_table: torch.Tensor,
        tokens_per_block: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        _qsa_kernels.qsa_paged_kv_store(
            k=k,
            v=v,
            k_cache=k_cache,
            v_cache=v_cache,
            request_indices=request_indices,
            logical_positions=logical_positions,
            block_table=block_table,
            tokens_per_block=tokens_per_block,
        )
        return k_cache, v_cache

    def reference(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        request_indices: torch.Tensor,
        logical_positions: torch.Tensor,
        block_table: torch.Tensor,
        tokens_per_block: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Both caches after the scatter, as fresh tensors; the inputs are not touched."""
        expected_k, expected_v = k_cache.clone(), v_cache.clone()
        requests = request_indices.long()
        positions = logical_positions.long()
        columns = positions.div(tokens_per_block, rounding_mode="floor")
        table_rows, table_width = block_table.shape
        addressable = (
            (requests >= 0) & (requests < table_rows) & (positions >= 0) & (columns < table_width)
        )
        pages = block_table[
            requests.clamp(0, table_rows - 1), columns.clamp(0, table_width - 1)
        ].long()
        addressable &= (pages >= 0) & (pages < k_cache.shape[0])
        page, slot = pages[addressable], (positions % tokens_per_block)[addressable]
        expected_k[page, :, slot, :] = k[addressable].to(expected_k.dtype)
        expected_v[page, :, slot, :] = v[addressable].to(expected_v.dtype)
        return expected_k, expected_v

    def is_valid(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        request_indices: torch.Tensor,
        logical_positions: torch.Tensor,
        block_table: torch.Tensor,
        tokens_per_block: int,
    ) -> None:
        # A strided view lands rows at other rows' slots, with nothing raised.
        assert request_indices.is_contiguous(), "request_indices must be a contiguous 1-D tensor"
        assert logical_positions.is_contiguous(), (
            "logical_positions must be a contiguous 1-D tensor"
        )


qsa_paged_kv_store = QsaPagedKvStore()
