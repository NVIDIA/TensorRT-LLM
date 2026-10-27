# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""QSA index side-cache store: raw index keys and position coordinates into their pages."""

from __future__ import annotations

from typing import Tuple

import torch
import triton
import triton.language as tl

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

_INTEGER_DTYPES = (torch.int32, torch.int64)


# The row count and the pool/table extents change from step to step; letting
# Triton specialize on them (== 1, divisible by 16) only multiplies compiles.
@triton.jit(do_not_specialize=["num_rows", "num_requests", "table_width", "num_pages"])
def _qsa_paged_index_store_kernel(
    token_k,
    coords,
    index_cache,
    position_cache,
    request_indices,
    logical_positions,
    block_table,
    num_rows,
    num_requests,
    table_width,
    num_pages,
    stride_k_row,
    stride_k_head,
    stride_k_dim,
    stride_c_row,
    stride_c_axis,
    stride_i_page,
    stride_i_slot,
    stride_i_head,
    stride_i_dim,
    stride_p_page,
    stride_p_slot,
    stride_p_axis,
    stride_request,
    stride_position,
    stride_t_request,
    stride_t_column,
    HKV: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
    AXES: tl.constexpr,
    BLOCK_A: tl.constexpr,
    TOKENS_PER_BLOCK: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
):
    # One program stores BLOCK_ROWS rows: the page lookup, then every head of
    # the key and the coordinate row, all through the operands' own strides.
    rows = (tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)).to(tl.int64)
    live = rows < num_rows
    request = tl.load(request_indices + rows * stride_request, mask=live, other=-1).to(tl.int64)
    position = tl.load(logical_positions + rows * stride_position, mask=live, other=-1).to(tl.int64)
    column = position // TOKENS_PER_BLOCK
    slot = position % TOKENS_PER_BLOCK
    live &= (request >= 0) & (request < num_requests) & (position >= 0) & (column < table_width)
    page = tl.load(
        block_table + request * stride_t_request + column * stride_t_column, mask=live, other=-1
    ).to(tl.int64)
    live &= (page >= 0) & (page < num_pages)

    dims = tl.arange(0, BLOCK_D)
    key_mask = live[:, None] & (dims < HEAD_DIM)[None, :]
    for head in tl.static_range(HKV):
        key = tl.load(
            token_k
            + rows[:, None] * stride_k_row
            + head * stride_k_head
            + dims[None, :] * stride_k_dim,
            mask=key_mask,
            other=0.0,
        )
        tl.store(
            index_cache
            + page[:, None] * stride_i_page
            + slot[:, None] * stride_i_slot
            + head * stride_i_head
            + dims[None, :] * stride_i_dim,
            key.to(index_cache.dtype.element_ty),
            mask=key_mask,
        )

    axes = tl.arange(0, BLOCK_A)
    axis_mask = live[:, None] & (axes < AXES)[None, :]
    coordinate = tl.load(
        coords + rows[:, None] * stride_c_row + axes[None, :] * stride_c_axis,
        mask=axis_mask,
        other=0,
    )
    tl.store(
        position_cache
        + page[:, None] * stride_p_page
        + slot[:, None] * stride_p_slot
        + axes[None, :] * stride_p_axis,
        coordinate.to(position_cache.dtype.element_ty),
        mask=axis_mask,
    )


def _launch_rows(rows: int) -> Tuple[int, int]:
    """Rows per program and warps: one row per program while the grid is small,
    a tile of rows once there are enough rows to fill the machine. Measured on
    GB300 against 1/4/8/16/32 rows per program."""
    if rows <= 256:
        return 1, 1
    return 16, 4


#: This checkpoint's index geometry and the engine's QSA metadata. `token_k` is
#: the k head of the packed `[T, 4 + 1, 128]` index projection; the coordinates
#: are the `[T, 3]` transpose of the engine's `[3, max_num_tokens]` position
#: buffer; `slots` x `width` is `qsa_block_table`. `pages` is the pool.
_CKPT = dict(
    hkv=1,
    head_dim=128,
    index_heads=4,
    tokens_per_block=32,
    axes=3,
    slots=32,
    width=260,
    key_dtype=torch.bfloat16,
    request_dtype=torch.int32,
    position_dtype=torch.int64,
    key_layout="packed",
    cache_layout="plain",
    hole=None,
)

#: The target's serving envelope: ISL 8192 + OSL 1024 puts decode rows up to position
#: 9215 (page column 287), 128 concurrent requests, and `max_num_tokens` 16384
#: rows in one prefill or mixed step.
_WIDE = dict(slots=160, width=300)
_PREFILL_16K = ((0, 8192), (0, 8192))

#: 32 requests of one MTP7 verify block each (8 rows), at starts that put some blocks across a
#: page edge and some inside one.
_MTP7_32 = tuple((29 + 61 * i, 8) for i in range(32))


def _spec(segments, pages, **overrides) -> dict:
    """`segments` is `(first position, rows)` per request, in row order."""
    return {**_CKPT, "segments": segments, "pages": pages, **overrides}


class QsaPagedIndexStore(OpWrapper):
    """Write each row's raw index key and position coordinates into its paged slot.

    For row t, with `page = block_table[request_indices[t], logical_positions[t]
    // tokens_per_block]` and `slot = logical_positions[t] % tokens_per_block`::

        index_cache[page, slot] = token_k[t]  # cast to the cache dtype
        position_cache[page, slot, :A] = position_coordinates[t]  # A = coordinates' width

    A row that resolves to no slot -- negative request, request past the
    table, negative position, page column past the table, or a table entry
    that is negative (`-1`, unallocated) or past the pool -- is skipped
    without error. The key is stored raw -- un-normed, un-rotated:
    `qsa_prefill_compress` pools raw keys and applies the Gemma norm and the
    rotation itself. Returns `(index_cache, position_cache)`, both written in
    place.

    Sourcing tier 5: one Triton launch written for this target
    (`_qsa_paged_index_store_kernel`). Nothing in the engine performs this
    store for a packed, ragged batch: the registered index-cache ops write fp8
    planes or norm and rotate on the way in, `triton_qsa_decode_pre_indexer`
    is one row per request, and `QSAIndexer.update_cache_and_compress` bundles
    the compression. Native torch needs a block-table gather, a floor-divide,
    a remainder, two indexed writes and their index casts (7 launches).

    Fusion boundary: page lookup, slot arithmetic, both scatters. The index
    projection, the query's norm and rotation, group compression, scoring,
    selection, sparse attention and the main K/V pool are the caller's.
    """

    ARCHS = frozenset({Arch.SM_103})

    CELLS: tuple[Cell, ...] = (
        Cell(
            why="decode MTP7: 32 requests x 8 rows (256), the last one-row-per-program "
            "launch, blocks straddling pages",
            spec=_spec(_MTP7_32, pages=1200, slots=40),
        ),
        Cell(
            why="prefill 16384 rows (max_num_tokens), 16 rows per program, token_k off the "
            "fused projection row",
            spec=_spec(_PREFILL_16K, pages=600, key_layout="fused_row", **_WIDE),
        ),
        Cell(
            why="NVFP4-KV side planes: index and position planes as strided views of one "
            "per-slot buffer holding 12 index-K pages and the position page; one row "
            "on an unallocated (-1) table entry is skipped",
            spec=_spec(
                ((0, 1), (31, 1), (32, 1), (33, 40)),
                pages=8,
                cache_layout="shared_page",
                layers=12,
                page_stride=49344,
                position_strides=(24672, 3, 1),
                hole="unallocated_page",
            ),
        ),
    )

    note = """
    Metadata consumed: none -- every input is an argument. The target passes
    `qsa_req_idx_per_token` (int32) and `qsa_logical_positions` (int64)
    sliced to the step's rows, `qsa_block_table` unsliced, the layer's
    `get_index_k_buffer(i)` (`[pages, tpb, 1, 128]` bf16) and the shared
    `get_qsa_position_buffer()` (`[pages, tpb, 3]` int32, rewritten
    idempotently by every sparse layer). Every operand is read through its
    own strides, all of them runtime arguments; the cells drive the target's:
    `token_k` off the packed index projection (row stride 640) and off the fused
    input projection (14080), the `[T, 3]` transpose of the position buffer as
    the coordinates, contiguous metadata and table. Under an
    NVFP4 KV pool the QSA cache manager packs each page's index-K roles and
    then its position page into one per-slot buffer, so both planes are strided
    views of one allocation: the trunk's index plane is `[pages, 32, 1, 128]`
    at page stride 49344 (12 x 4096 + 192) with the position plane
    `[pages, 32, 3]` at (24672, 3, 1); the cell checks that the other index-K
    roles of each page survive the call.

    Preconditions:
      - destination slots are unique across the batch: two rows naming one
        (page, slot) both store, unordered. The engine's mapping cannot
        collide.
    Loud rejections (`ValueError`, before launching; the test pins the
    unequal-length metadata one): metadata that is not
    one-dimensional, of unequal length, or not int32/int64; `token_k` or the
    coordinates with a row count other than the metadata's; an index plane
    whose page size, head count or head width disagrees with
    `tokens_per_block` and `token_k`; a position plane whose page size
    disagrees or that is narrower than the coordinates; a table that is not a
    2-D integer tensor. `rows == 0` returns before launching.

    Values the caller owns -- data, not metadata (measured, not pinned by a
    test): coordinates wider than the plane's dtype wrap (int64 `2^31 + 5`
    stores as -2147483643); floats truncate toward zero.

    One launch per call; HEAD_DIM, the head count, the page size and the
    coordinate width are constexprs, everything else (strides, pool and table
    extents, rows) is a runtime argument and the row count and extents are
    not specialized on, so a new batch or pool size does not recompile
    (strides still specialize on 1 and 16-divisibility). Up to 256 rows run
    one row per program (1 warp), above that 16 rows per program (4 warps).
    CUDA-graph capturable (captured at 256 rows, 32 MTP7 verify blocks); a
    replay follows positions rewritten in place.

    Hard limits: every index is widened to int64 inside the kernel before
    any arithmetic; an int64 position at `2^32 + 5` lands past the table and
    is skipped rather than truncated (read from the kernel, not pinned by a
    test). Rows are a grid axis: 2^31 programs.

    Skip certified by a cell: an unallocated (`-1`) table entry, one row of the
    NVFP4-plane cell. The other skip conditions (a negative request index among
    them) share that mask and have no cell.

    Not certified: an fp8 index plane, keys other than bf16, any index geometry
    other than 1 head x 128 at 32-token pages, decode batches other than 32 MTP7
    verify blocks, strided metadata or tables,
    stride-0 coordinates, multiple devices, concurrent streams onto one pool,
    duplicate destinations.
    """

    def __call__(
        self,
        token_k: torch.Tensor,
        position_coordinates: torch.Tensor,
        index_cache: torch.Tensor,
        position_cache: torch.Tensor,
        request_indices: torch.Tensor,
        logical_positions: torch.Tensor,
        block_table: torch.Tensor,
        tokens_per_block: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        rows = request_indices.shape[0]
        if request_indices.dim() != 1 or logical_positions.shape != request_indices.shape:
            raise ValueError(
                f"request_indices {tuple(request_indices.shape)} and logical_positions "
                f"{tuple(logical_positions.shape)} must be one-dimensional and equally long"
            )
        if request_indices.dtype not in _INTEGER_DTYPES or (
            logical_positions.dtype not in _INTEGER_DTYPES
            or block_table.dtype not in _INTEGER_DTYPES
        ):
            raise ValueError("QSA index store metadata and block table must be int32 or int64")
        if token_k.dim() != 3 or token_k.shape[0] != rows:
            raise ValueError(f"token_k {tuple(token_k.shape)} must be [{rows}, heads, dim]")
        if position_coordinates.dim() != 2 or position_coordinates.shape[0] != rows:
            raise ValueError(
                f"position_coordinates {tuple(position_coordinates.shape)} must be [{rows}, axes]"
            )
        if index_cache.dim() != 4 or index_cache.shape[1:] != (
            tokens_per_block,
            *token_k.shape[1:],
        ):
            raise ValueError(
                f"index_cache {tuple(index_cache.shape)} must be [pages, {tokens_per_block}, "
                f"{token_k.shape[1]}, {token_k.shape[2]}]"
            )
        if (
            position_cache.dim() != 3
            or position_cache.shape[1] != tokens_per_block
            or position_cache.shape[2] < position_coordinates.shape[1]
        ):
            raise ValueError(
                f"position_cache {tuple(position_cache.shape)} must be [pages, "
                f"{tokens_per_block}, >= {position_coordinates.shape[1]}]"
            )
        if block_table.dim() != 2:
            raise ValueError(f"block_table {tuple(block_table.shape)} must be two-dimensional")
        if rows == 0:
            return index_cache, position_cache
        block_rows, warps = _launch_rows(rows)
        axes = position_coordinates.shape[1]
        _qsa_paged_index_store_kernel[(triton.cdiv(rows, block_rows),)](
            token_k,
            position_coordinates,
            index_cache,
            position_cache,
            request_indices,
            logical_positions,
            block_table,
            rows,
            block_table.shape[0],
            block_table.shape[1],
            index_cache.shape[0],
            *token_k.stride(),
            *position_coordinates.stride(),
            *index_cache.stride(),
            *position_cache.stride(),
            request_indices.stride(0),
            logical_positions.stride(0),
            *block_table.stride(),
            HKV=token_k.shape[1],
            HEAD_DIM=token_k.shape[2],
            BLOCK_D=triton.next_power_of_2(token_k.shape[2]),
            AXES=axes,
            BLOCK_A=triton.next_power_of_2(axes),
            TOKENS_PER_BLOCK=tokens_per_block,
            BLOCK_ROWS=block_rows,
            num_warps=warps,
        )
        return index_cache, position_cache

    def reference(
        self,
        token_k: torch.Tensor,
        position_coordinates: torch.Tensor,
        index_cache: torch.Tensor,
        position_cache: torch.Tensor,
        request_indices: torch.Tensor,
        logical_positions: torch.Tensor,
        block_table: torch.Tensor,
        tokens_per_block: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Both planes after the store, as fresh tensors; the inputs are not touched.

        Every row's destination, and whether it has one, is decided in Python
        integers from the table read back to the host -- none of the kernel's
        arithmetic -- and the surviving rows are written with one indexed copy
        per plane.
        """
        expected_index, expected_position = index_cache.clone(), position_cache.clone()
        table = block_table.tolist()
        pool = index_cache.shape[0]
        kept, pages, slots = [], [], []
        rows = zip(request_indices.tolist(), logical_positions.tolist())
        for row, (request, position) in enumerate(rows):
            if not 0 <= request < len(table) or position < 0:
                continue
            column, slot = divmod(position, tokens_per_block)
            if column >= len(table[request]) or not 0 <= table[request][column] < pool:
                continue
            kept.append(row)
            pages.append(table[request][column])
            slots.append(slot)
        if not kept:
            return expected_index, expected_position
        device = index_cache.device
        kept_t, page_t, slot_t = (
            torch.tensor(v, dtype=torch.long, device=device) for v in (kept, pages, slots)
        )
        axes = position_coordinates.shape[-1]
        expected_index[page_t, slot_t] = token_k[kept_t].to(index_cache.dtype)
        expected_position[page_t, slot_t, :axes] = position_coordinates[kept_t].to(
            position_cache.dtype
        )
        return expected_index, expected_position


qsa_paged_index_store = QsaPagedIndexStore()
