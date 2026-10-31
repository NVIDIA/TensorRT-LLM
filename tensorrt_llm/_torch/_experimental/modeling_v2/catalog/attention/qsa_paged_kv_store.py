# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""QSA sparse path's paged K/V append: this step's rows into the HND main pool."""

from __future__ import annotations

from typing import Tuple

import torch

# Tier-2 sourcing: a vendored Triton callable reached as ordinary Python. QSA's
# kernel plane has no torch.ops surface.
from tensorrt_llm._torch.attention.backends.sparse.qsa import kernels as _qsa_kernels

from .._op import Arch, Cell, OpWrapper

#: This checkpoint's main K/V geometry and the engine's QSA metadata. `heads_q`
#: places k and v as head slices of the packed `[T, 24 + 2 + 2, 256]` QKV
#: buffer; `slots` x `width` is `qsa_block_table` (max_num_sequences x
#: max_blocks_per_seq). `pages` is the pool, independent of the batch.
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
    # Heads of further columns after v in each row: 27 is the target's fused input
    # projection row [q | k | v | gate | index | pad] (55 heads of 256).
    row_pad_heads=0,
)

_F8 = torch.float8_e4m3fn

_DECODE = ((0, 1), (31, 1), (32, 1), (33, 1), (100, 1), (255, 1), (7, 1), (64, 1))

#: 32 requests of one MTP7 verify block each (8 rows), at starts that put some blocks across a
#: page edge and some inside one.
_MTP7_32 = tuple((29 + 61 * i, 8) for i in range(32))

#: One rank's QSA K/V pool as the stock `Qwen4ExpHybridCacheManager` lays it out at the unified target's
#: members: a page holds the K and V role planes of all 12 attention layers (24 role pages of
#: `[hkv, 32, 256]`), so a layer's `get_buffers(i, kv_layout="HND")` view steps 24 role pages a
#: page. `pool="engine"` cells drive layer 47's view (`layer_slot` 11, the page's last planes) at
#: an engine's own page count. `slots` x `width` is the engine's `qsa_block_table`:
#: `max_num_sequences` 128 by `max_blocks_per_seq` 292 at `max_seq_len` 9216 (516 at 16384).
_ENGINE = dict(_CKPT, pool="engine", role_pages=24, layer_slot=11, slots=128, width=292)

#: A tp4 / tep4 rank: one KV head, k and v at heads 6 and 7 of the fused projection row
#: `[q 6 | k 1 | v 1 | gate 6 | index 3]` (17 heads of 256, row stride 4352).
_TP4 = dict(_ENGINE, hkv=1, heads_q=6, row_pad_heads=9, row_stride=4352)

#: A dep4 rank (attention DP, every head): the tp1 row `[q 24 | k 2 | v 2 | gate 24 | index 3]`
#: (55 heads, row stride 14080) over the kv_fp8 variant's e4m3 pool.
_DEP4_FP8 = dict(_ENGINE, row_pad_heads=27, row_stride=14080, cache_dtype=torch.float8_e4m3fn)

#: Attention pool pages of a final engine at core's runtime (KV fraction 0.6, bf16 recurrent
#: state, `max_batch_size` 128, `max_seq_len` 9216), measured on the stock engine: its reported
#: `maxNumBlocks` less the recurrent pool's slots, which that total also counts. tep4, bf16 KV:
#: 258983 - 1874. dep4, kv_fp8: 172579 - 1217.
PAGES_TEP4 = 257109
PAGES_DEP4_FP8 = 171362

#: 128 decode rows, one a request: page-aligned, mid-page, page-crossing and the last position
#: `max_seq_len` 9216 admits, then a stride-577 walk over the rest.
_DECODE_128 = ((0, 1), (31, 1), (32, 1), (9215, 1)) + tuple(
    ((i * 577 + 45) % 9216, 1) for i in range(124)
)

#: 32 decode rows, one a request, the last at position 9215.
_DECODE_32 = tuple(((i * 293 + 5) % 9216, 1) for i in range(31)) + ((9215, 1),)

#: 128 requests of one MTP7 verify block each (8 rows, 1024), the last ending at position 16383
#: (`max_seq_len` 16384 under MTP); starts at every offset in a page, so blocks straddle pages.
_MTP7_128 = tuple(((29 + 127 * i) % 16376, 8) for i in range(127)) + ((16376, 8),)

#: An 8192-row (`max_num_tokens`) context step: a chunk continuing from mid-page to position
#: 9215, and a request's first 1023 tokens.
_PREFILL_8192 = ((2047, 7169), (0, 1023))


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


def _bits(tensor: torch.Tensor) -> torch.Tensor:
    """The same storage as integers: fp8 has no comparison ops, and -0.0 != 0.0."""
    as_int = {1: torch.uint8, 2: torch.int16, 4: torch.int32}[tensor.element_size()]
    return tensor.view(as_int)


class QsaPagedKvStore(OpWrapper):
    """Scatter every row's K and V into its request's page of the HND pool.

    Wraps `tensorrt_llm._torch.attention.backends.sparse.qsa.kernels.
    triton_qsa_paged_kv_store`: one Triton launch over `(rows, Hkv)`. For row t::

        page = block_table[request_indices[t], logical_positions[t] // tokens_per_block]
        slot = logical_positions[t] % tokens_per_block
        k_cache[page, :, slot, :] = k[t]          v_cache[page, :, slot, :] = v[t]

    `logical_positions` is absolute, so prefill, decode and MTP rows are the
    same call. The store is a move, converted once to the pool's dtype
    (bitwise `Tensor.to`; an fp8-e4m3 pool saturates at 448 and takes no
    scale, so a reader treats it as kv scale 1.0). A row that resolves to no
    slot -- negative request, request past the table, negative position, page
    column past the table, or a table entry that is negative (`-1`,
    unallocated) or past the pool -- is skipped without error. Returns
    `(k_cache, v_cache)`, both written in place.

    Fusion boundary: the scatter only. The K/V projection, norm and RoPE of
    K, the QSA index plane, selection and attention are the caller's; the
    index plane is never touched.
    """

    ARCHS = frozenset({Arch.SM_103})

    CELLS: tuple[Cell, ...] = (
        # The decode and prefill targets read k/v off the fused projection row; the MTP
        # targets split them off a packed QKV row.
        Cell(
            why="decode: 8 rows off the fused projection row (row stride 14080), "
            "at aligned, mid and page-crossing positions, plus one row on an "
            "unallocated (-1) table entry that is skipped",
            spec=_spec(_DECODE, pages=64, row_pad_heads=27, hole="unallocated_page"),
        ),
        Cell(
            why="prefill 1024 rows off the fused projection row (row stride 14080)",
            spec=_spec(((0, 256),) * 4, pages=160, row_pad_heads=27),
        ),
        # MTP7 verify blocks: 8 rows per request, so one block spans up to two pages.
        # The non-default pool the forward reaches: fp8-e4m3.
        Cell(
            why="fp8 decode MTP7: 32 requests x 8 rows (256), blocks straddling pages",
            spec=_spec(_MTP7_32, pages=1200, slots=40, cache_dtype=_F8),
        ),
        # The unified target's members over the engine's own pool: one rank's page count, the
        # 24-role-page stride, layer 47's view, a 128-row table.
        Cell(
            why="tep4 decode, 1 row: hkv 1 off the 4352-wide row, position 9215, "
            "257109-page bf16 pool",
            spec=_engine(_TP4, "tep4", ((9215, 1),), PAGES_TEP4),
        ),
        Cell(
            why="tep4 decode, 128 rows (max_batch_size), aligned, mid and page-crossing",
            spec=_engine(_TP4, "tep4", _DECODE_128, PAGES_TEP4),
        ),
        Cell(
            why="tep4 MTP7 verify, 128 requests x 8 rows (1024) to position 16383, table width 516",
            spec=_engine(_TP4, "tep4", _MTP7_128, PAGES_TEP4, width=516),
        ),
        Cell(
            why="tp4 context, 8192 rows (max_num_tokens): a mid-page chunk to 9215 and a "
            "request's first 1023",
            spec=_engine(_TP4, "tp4", _PREFILL_8192, PAGES_TEP4),
        ),
        Cell(
            why="dep4 kv_fp8 decode, 32 rows: hkv 2 off the 14080-wide row into a "
            "171362-page e4m3 pool",
            spec=_engine(_DEP4_FP8, "dep4", _DECODE_32, PAGES_DEP4_FP8),
        ),
    )

    note = """
    Source: the in-tree Triton kernel `_qsa_paged_kv_store_kernel`, JIT-compiled
    on its first call; `ARCHS` is the architecture it compiled for and ran on.

    Metadata consumed: none -- every input is an argument. The target passes
    `qsa_req_idx_per_token` (int32, the row's batch index) and
    `qsa_logical_positions` (int64) sliced to the step's rows,
    `qsa_block_table` unsliced (raw pool page ids, `-1` unallocated), and the
    layer pool's K and V role views of `get_buffers(i, kv_layout="HND")`.

    Preconditions:
      - destination slots are unique across the batch: two rows naming one
        (page, slot) both store, unordered. The engine's mapping cannot
        collide.
      - `request_indices` and `logical_positions` are contiguous: the kernel
        gets no stride for them. A strided view is read through its base
        pointer and rows land at other rows' slots, silently (`is_valid`).
    Every other operand is read through its strides; the cells drive k/v as
    head slices of the packed QKV row and of the fused projection row, the
    pool's K and V role views, and a contiguous table. The callable rejects
    loudly: mismatched or non-3-D/4-D K/V shapes, head geometry,
    `k_cache.shape[2] != tokens_per_block`, row dtypes outside bf16/fp16,
    pool dtypes outside bf16/fp16/fp8-e4m3, a non-int32 or non-2-D table,
    non-integer or wrong-length metadata, CPU operands, an empty pool or
    table (not pinned by a test). `rows == 0` returns before launching.

    Strides and the pool/table extents are Triton constexprs: each distinct
    layout or pool size compiles its own kernel (once per engine, at its
    pool's page count). CUDA-graph capturable (captured at 256 rows, 32 MTP7
    verify blocks, and at 128 tep4 decode rows over the 257109-page pool); a
    replay follows positions rewritten in place.

    The engine's pool, as the stock manager lays it out at tp4 / tep4 / dep4
    (measured on the manager the engine builds): one page holds the K and V
    planes of all 12 attention layers, so a layer's K view steps 24 role
    pages a page -- 196608 elements at one KV head, 393216 at two, bf16 and
    fp8 alike -- and its page count is the engine's (`PAGES_TEP4`,
    `PAGES_DEP4_FP8`). The `pool="engine"` cells drive layer 47's view with
    the pool's last page among the touched ones: its planes lie at byte
    101099339776 of a 101 GB range at tep4 and 67382247424 of 67 GB at dep4
    kv_fp8, past 2^32 elements of the view's base. The kernel loads the page
    id as int64 before its stride product, and every such row lands bitwise.

    Hard limits: both metadata values are cast to int32 inside the kernel
    (`.to(tl.int32)`); an int64 position at `2^32 + 5` was probed once (not
    pinned by a test) and lands on slot 5 of page column 0 -- written, not
    skipped. The engine's positions stay below 2^14 at the gates' envelope
    (`max_seq_len` 16384 with MTP).

    Not certified: duplicate destinations, multiple devices, a scaled fp8
    pool (there is no scale argument), fp8 conversion of values past 448, the
    packed-QKV-row (MTP draft layer) layout over a bf16 pool or at one KV head,
    fp16 rows, head geometry other than D 256 with Hkv 1 or 2 and 32-token
    pages, int32 positions or int64 request indices, head-major rows, sliced
    or transposed tables, `rows == 0`, and of the rows that resolve to no slot
    any but an unallocated (`-1`) table entry (the reference skips each).
    Pools at a page count not driven here: tp4's own engine (tep4's count
    stands in: a tp4 rank's pool has the same per-page shape, and the stock
    engine cannot boot tp4), the MTP engine's at tep4 (its draft layer has a
    separate cache, so its trunk count differs from core's, which stands in),
    and dep4's bf16 core pool (119680 pages measured, not driven). Positions
    past 16383 (the tune's `max_seq_len` 262144) and tables wider than 516
    columns.
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
        _qsa_kernels.triton_qsa_paged_kv_store(
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
        # The kernel reads these two as flat unit-stride buffers. Measured: a
        # column of a 2-wide buffer is read through its base pointer and the
        # rows land at other rows' slots, with nothing raised.
        assert request_indices.is_contiguous(), "request_indices must be a contiguous 1-D tensor"
        assert logical_positions.is_contiguous(), (
            "logical_positions must be a contiguous 1-D tensor"
        )


qsa_paged_kv_store = QsaPagedKvStore()
