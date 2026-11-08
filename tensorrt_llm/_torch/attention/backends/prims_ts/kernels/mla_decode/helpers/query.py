# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Logical/physical query-row geometry helpers for MLA decode."""

from dataclasses import dataclass

import cutlass
import cutlass.cute as cute
from cutlass import Int32, Int64


@dataclass(frozen=True)
class FlatQueryTileLayout:
    """Host-side layout for consecutive ``(query, head)`` rows.

    Query tokens and heads form one affine row space ordered as
    ``flat_row = query_idx * logical_num_heads_q + head_idx``.  Physical MMA
    tiles consume consecutive rows from that space, so only the final tile can
    be partial.  ``tail_rows`` is in ``[1, tile_size_q]`` and equals
    ``tile_size_q`` when the final tile is full.

    Unlike the monolithic M128 helper, this generalized PrimTS layout permits
    ``logical_num_heads_q > tile_size_q``.  That case is required by the M8,
    M16, M32, and M64 1CTA profiles: a physical tile can cover part of one
    token and the following tile continues at the next logical head row.
    """

    logical_num_heads_q: int
    logical_seq_len_q: int
    tile_size_q: int
    total_rows: int
    num_tiles: int
    tail_rows: int

    @classmethod
    def for_tile(
        cls,
        logical_num_heads_q: int,
        logical_seq_len_q: int,
        tile_size_q: int,
    ) -> "FlatQueryTileLayout":
        if logical_num_heads_q <= 0:
            raise ValueError("logical_num_heads_q must be positive")
        if logical_seq_len_q <= 0:
            raise ValueError("logical_seq_len_q must be positive")
        if tile_size_q <= 0:
            raise ValueError("tile_size_q must be positive")

        total_rows = logical_num_heads_q * logical_seq_len_q
        num_tiles = (total_rows + tile_size_q - 1) // tile_size_q
        tail_rows = total_rows - (num_tiles - 1) * tile_size_q
        return cls(
            logical_num_heads_q=logical_num_heads_q,
            logical_seq_len_q=logical_seq_len_q,
            tile_size_q=tile_size_q,
            total_rows=total_rows,
            num_tiles=num_tiles,
            tail_rows=tail_rows,
        )


@cute.jit
def query_batch_bounds(
    cu_seqlens_q,
    batch_idx,
    logical_seq_len_q: cutlass.Constexpr[int],
):
    """Return the compact-storage offset and logical Q length for a batch.

    ``cu_seqlens_q`` follows the standard cumulative-offset convention.  A
    ``None`` value selects the fixed-length specialization, where every batch
    has ``logical_seq_len_q`` rows and the batch dimension remains explicit in
    the public Q/O tensors.
    """

    if cutlass.const_expr(cu_seqlens_q is None):
        return Int32(0), Int32(logical_seq_len_q)
    q_start = Int32(cu_seqlens_q[batch_idx])
    q_end = Int32(cu_seqlens_q[Int32(batch_idx) + Int32(1)])
    return q_start, q_end - q_start


@cute.jit
def runtime_flat_query_tile_valid_rows(
    query_tile_idx,
    tile_size_q: cutlass.Constexpr[int],
    logical_num_heads_q: cutlass.Constexpr[int],
    logical_seq_len_q: cutlass.Constexpr[int],
    cu_seqlens_q=None,
    batch_idx=None,
):
    """Return the active row prefix of one physical flat-Q tile."""

    _, query_len = query_batch_bounds(
        cu_seqlens_q,
        batch_idx,
        logical_seq_len_q,
    )
    remaining_rows = query_len * Int32(logical_num_heads_q) - Int32(
        query_tile_idx
    ) * Int32(tile_size_q)
    return cute.math.max(
        Int32(0),
        cute.math.min(Int32(tile_size_q), remaining_rows),
    )


@cute.jit
def runtime_flat_query_tile_has_rows(
    query_tile_idx,
    tile_size_q: cutlass.Constexpr[int],
    logical_num_heads_q: cutlass.Constexpr[int],
    logical_seq_len_q: cutlass.Constexpr[int],
    cu_seqlens_q=None,
    batch_idx=None,
):
    """Return whether one rectangular scheduler tile owns a logical Q row."""

    return runtime_flat_query_tile_valid_rows(
        query_tile_idx,
        tile_size_q,
        logical_num_heads_q,
        logical_seq_len_q,
        cu_seqlens_q,
        batch_idx,
    ) > Int32(0)


@cute.jit
def flat_query_row_state(
    row_in_tile,
    query_tile_idx,
    tile_size_q: cutlass.Constexpr[int],
    logical_num_heads_q: cutlass.Constexpr[int],
    logical_seq_len_q: cutlass.Constexpr[int],
    cu_seqlens_q=None,
    batch_idx=None,
):
    """Map one physical flat-tile row to logical and public coordinates.

    Returns ``(storage_flat_row, logical_head, safe_local_q, storage_q,
    is_valid)``. Invalid final-tile rows receive safe control-flow coordinates
    but remain predicated from every GMEM transaction by ``is_valid``.
    """

    local_flat_query_row = Int32(query_tile_idx) * Int32(tile_size_q) + Int32(
        row_in_tile
    )
    logical_q_idx = local_flat_query_row // Int32(logical_num_heads_q)
    logical_head_idx = local_flat_query_row - logical_q_idx * Int32(logical_num_heads_q)
    q_start, q_len = query_batch_bounds(
        cu_seqlens_q,
        batch_idx,
        logical_seq_len_q,
    )
    safe_q_len = cute.math.max(q_len, Int32(1))
    safe_logical_q_idx = cute.math.min(logical_q_idx, safe_q_len - Int32(1))
    storage_q_idx = q_start + safe_logical_q_idx
    storage_flat_query_row = (
        storage_q_idx * Int32(logical_num_heads_q) + logical_head_idx
    )
    is_valid = local_flat_query_row < q_len * Int32(logical_num_heads_q)
    return (
        storage_flat_query_row,
        logical_head_idx,
        safe_logical_q_idx,
        storage_q_idx,
        is_valid,
    )


@cute.jit
def public_query_flat_row(cfg, storage_flat_query_row, batch_idx, cu_seqlens_q):
    """Return the flat physical row used by public O and LSE tensors.

    Fixed-length tensors retain an explicit batch dimension, while compact
    variable-length tensors already include the cumulative batch offset in
    ``storage_flat_query_row``.
    """

    if cutlass.const_expr(cu_seqlens_q is None):
        return Int32(batch_idx) * Int32(
            cfg.logical_seq_len_q * cfg.logical_num_heads_q
        ) + Int32(storage_flat_query_row)
    return Int32(storage_flat_query_row)


@cute.jit
def split_o_element_offset(
    cfg,
    batch_idx,
    q_idx,
    head_idx,
    split_idx,
    dim_idx,
):
    """Return a batch-dynamic partial-O offset with 64-bit-safe products.

    Batch size is intentionally absent from the compile signature, so the
    complete workspace extent cannot prove that every offset fits in Int32.
    The within-batch layout is still compile-time constant, however. Keep that
    common bounded part in 32-bit arithmetic and widen only the batch-stride
    product. Use fully widened arithmetic when a profile's per-batch layout
    alone exceeds Int32.
    """

    elements_per_batch = (
        cfg.seq_len_q * cfg.num_heads_q * cfg.num_ctas_per_seq_kv * cfg.head_dim_v
    )
    if cutlass.const_expr(elements_per_batch <= (1 << 31) - 1):
        within_batch_offset = (
            (Int32(q_idx) * Int32(cfg.num_heads_q) + Int32(head_idx))
            * Int32(cfg.num_ctas_per_seq_kv)
            + Int32(split_idx)
        ) * Int32(cfg.head_dim_v) + Int32(dim_idx)
        return Int64(batch_idx) * Int64(elements_per_batch) + Int64(within_batch_offset)

    return (
        (
            (Int64(batch_idx) * Int64(cfg.seq_len_q) + Int64(q_idx))
            * Int64(cfg.num_heads_q)
            + Int64(head_idx)
        )
        * Int64(cfg.num_ctas_per_seq_kv)
        + Int64(split_idx)
    ) * Int64(cfg.head_dim_v) + Int64(dim_idx)
