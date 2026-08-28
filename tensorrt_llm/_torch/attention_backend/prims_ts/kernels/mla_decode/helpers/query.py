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

"""groups_tokens_heads_q shape and row-mapping helpers for MLA decode."""

import cutlass
import cutlass.cute as cute
from cutlass import Int32, Int64


def groups_tokens_heads_q_capacity(logical_num_heads_q: int, tile_size_q: int) -> int:
    """Return logical Q tokens grouped into one selected Q tile.

    Capacity is a property of the selected tile and logical head count.  It is
    intentionally independent of the runtime logical Q length so one compiled
    layout has the same row geometry for short and long query sequences.
    """

    if logical_num_heads_q <= 0:
        raise ValueError("logical_num_heads_q must be positive")
    if tile_size_q <= 0:
        raise ValueError("tile_size_q must be positive")
    return max(1, tile_size_q // logical_num_heads_q)


def groups_tokens_heads_q_group_count(
    logical_seq_len_q: int, groups_tokens_heads_q_ratio: int
) -> int:
    """Return the ceil-divided number of effective query groups."""

    if logical_seq_len_q <= 0:
        raise ValueError("logical_seq_len_q must be positive")
    if groups_tokens_heads_q_ratio <= 0:
        raise ValueError("groups_tokens_heads_q_ratio must be positive")
    return (
        logical_seq_len_q + groups_tokens_heads_q_ratio - 1
    ) // groups_tokens_heads_q_ratio


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


def query_group_has_rows(
    effective_seq_group_idx: int,
    groups_tokens_heads_q_ratio: int,
    logical_seq_len_q: int,
) -> bool:
    """Return whether an effective Q group contains a logical query row."""

    if effective_seq_group_idx < 0:
        raise ValueError("effective_seq_group_idx must be non-negative")
    if groups_tokens_heads_q_ratio <= 0:
        raise ValueError("groups_tokens_heads_q_ratio must be positive")
    if logical_seq_len_q < 0:
        raise ValueError("logical_seq_len_q must be non-negative")
    return effective_seq_group_idx * groups_tokens_heads_q_ratio < logical_seq_len_q


@cute.jit
def runtime_query_group_has_rows(
    effective_seq_group_idx,
    groups_tokens_heads_q_ratio: cutlass.Constexpr[int],
    logical_seq_len_q: cutlass.Constexpr[int],
    cu_seqlens_q=None,
    batch_idx=None,
):
    """Return whether a runtime batch has data in an effective Q group."""

    _, query_len = query_batch_bounds(
        cu_seqlens_q,
        batch_idx,
        logical_seq_len_q,
    )
    return (
        Int32(effective_seq_group_idx) * Int32(groups_tokens_heads_q_ratio) < query_len
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
    """Return a partial-O offset without overflowing intermediate products.

    The workspace geometry is compile-time constant.  Keep the common, bounded
    layouts in 32-bit arithmetic and widen the final element offset; use fully
    widened arithmetic only when the flattened workspace can exceed ``Int32``.
    """

    if cutlass.const_expr(
        cfg.batch_size
        * cfg.seq_len_q
        * cfg.num_heads_q
        * cfg.num_ctas_per_seq_kv
        * cfg.head_dim_v
        <= (1 << 31) - 1
    ):
        return Int64(
            batch_idx
            * Int32(
                cfg.seq_len_q
                * cfg.num_heads_q
                * cfg.num_ctas_per_seq_kv
                * cfg.head_dim_v
            )
            + q_idx * Int32(cfg.num_heads_q * cfg.num_ctas_per_seq_kv * cfg.head_dim_v)
            + head_idx * Int32(cfg.num_ctas_per_seq_kv * cfg.head_dim_v)
            + split_idx * Int32(cfg.head_dim_v)
            + dim_idx
        )

    return (
        (
            (Int64(batch_idx) * Int64(cfg.seq_len_q) + Int64(q_idx))
            * Int64(cfg.num_heads_q)
            + Int64(head_idx)
        )
        * Int64(cfg.num_ctas_per_seq_kv)
        + Int64(split_idx)
    ) * Int64(cfg.head_dim_v) + Int64(dim_idx)


@cute.jit
def groups_tokens_heads_q_row_state(
    effective_head_idx,
    effective_seq_group_idx,
    groups_tokens_heads_q_ratio: cutlass.Constexpr[int],
    logical_num_heads_q: cutlass.Constexpr[int],
    logical_seq_len_q: cutlass.Constexpr[int],
    cu_seqlens_q=None,
    batch_idx=None,
):
    """Map one effective groups_tokens_heads_q row to logical and storage coordinates.

    Returns ``(storage_flat_row, logical_head, safe_local_q, storage_q,
    is_valid)``.  In variable-length mode, storage coordinates include the
    cumulative batch offset while causal coordinates stay batch-local.  The
    local Q index is clamped to the final real query row so padded rows can
    safely participate in K scheduling and synchronization.  ``is_valid``
    predicates public and GMEM-partial output stores.  cluster-local staging may
    retain padded rows so all participants synchronize uniformly.
    """

    effective_num_heads_q = Int32(logical_num_heads_q * groups_tokens_heads_q_ratio)
    local_flat_query_row = Int32(
        effective_seq_group_idx
    ) * effective_num_heads_q + Int32(effective_head_idx)
    logical_q_idx = local_flat_query_row // Int32(logical_num_heads_q)
    logical_head_idx = local_flat_query_row - logical_q_idx * Int32(logical_num_heads_q)
    q_start, q_len = query_batch_bounds(
        cu_seqlens_q,
        batch_idx,
        logical_seq_len_q,
    )
    # Keep invalid padded rows on a valid local coordinate for control-flow
    # and synchronization.  Their GMEM loads/stores are independently made
    # OOB/predicated by the resource that owns the transaction.
    safe_q_len = cute.math.max(q_len, Int32(1))
    safe_logical_q_idx = cute.math.min(
        logical_q_idx,
        safe_q_len - Int32(1),
    )
    storage_q_idx = q_start + safe_logical_q_idx
    storage_flat_query_row = (
        storage_q_idx * Int32(logical_num_heads_q) + logical_head_idx
    )
    is_valid = logical_q_idx < q_len
    return (
        storage_flat_query_row,
        logical_head_idx,
        safe_logical_q_idx,
        storage_q_idx,
        is_valid,
    )
