# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""QSA sparse path's paged K/V append: this step's rows into the HND main pool."""

from __future__ import annotations

from typing import Tuple

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

# Tier-2 sourcing: a vendored Triton callable reached as ordinary Python. QSA's
# kernel plane has no torch.ops surface.
from tensorrt_llm._torch.attention.backends.sparse.qsa import kernels as _qsa_kernels

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


def _spec(segments, pages, **overrides) -> dict:
    """`segments` is `(first position, rows)` per request, in row order."""
    return {**_CKPT, "segments": segments, "pages": pages, **overrides}


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
    )

    note = """
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
    layout or pool size compiles its own kernel. CUDA-graph capturable
    (captured at 256 rows, 32 MTP7 verify blocks); a replay follows positions
    rewritten in place.

    Hard limits: both metadata values are cast to int32 inside the kernel
    (`.to(tl.int32)`); an int64 position at `2^32 + 5` was probed once (not
    pinned by a test) and lands on slot 5 of page column 0 -- written, not
    skipped. The engine's positions stay below 2^14.

    Not certified: duplicate destinations, multiple devices, a scaled fp8
    pool (there is no scale argument), fp8 conversion of values past 448, the
    packed-QKV-row (MTP) layout over a bf16 pool, fp16 rows, head geometry other than
    this checkpoint's (D 256, Hkv 2, 32-token pages), int32 positions or
    int64 request indices, head-major rows, sliced or transposed tables,
    `rows == 0`, and of the rows that resolve to no slot any but an
    unallocated (`-1`) table entry (the reference skips each).
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
