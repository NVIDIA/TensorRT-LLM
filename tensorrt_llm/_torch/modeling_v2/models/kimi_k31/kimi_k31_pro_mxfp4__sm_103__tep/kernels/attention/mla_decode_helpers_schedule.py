# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
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
"""Query-row geometry, captured-schedule and task-schedule helpers for MLA decode.

Holds the logical/physical query-row geometry helpers, the captured-schedule
helper functions, and the advisory diagnostics of MLA task-schedule construction.
"""

import warnings
from contextlib import contextmanager
from dataclasses import dataclass

import cutlass
from cutlass import Int32, cute
from cutlass.experimental.task_scheduling.task_manager import TaskManager as BaseTaskManager


@dataclass(frozen=True)
class FlatQueryTileLayout:
    """Host-side layout for consecutive ``(query, head)`` rows.

    Query tokens and heads form one affine row space ordered as
    ``flat_row = query_idx * logical_num_heads_q + head_idx``.  Physical MMA
    tiles consume consecutive rows from that space, so only the final tile can
    be partial.  ``tail_rows`` is in ``[1, tile_size_q]`` and equals
    ``tile_size_q`` when the final tile is full.

    Unlike the monolithic M128 helper, this generalized PrimTS layout permits
    ``logical_num_heads_q > tile_size_q``: a physical tile can cover part of
    one token and the following tile continues at the next logical head row.
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
    logical_seq_len_q: Int32,
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
    logical_seq_len_q: Int32,
    cu_seqlens_q=None,
    batch_idx=None,
):
    """Return the active row prefix of one physical flat-Q tile."""

    _, query_len = query_batch_bounds(
        cu_seqlens_q,
        batch_idx,
        logical_seq_len_q,
    )
    remaining_rows = query_len * Int32(logical_num_heads_q) - Int32(query_tile_idx) * Int32(
        tile_size_q
    )
    return cute.math.max(
        Int32(0),
        cute.math.min(Int32(tile_size_q), remaining_rows),
    )


@cute.jit
def runtime_flat_query_tile_has_rows(
    query_tile_idx,
    tile_size_q: cutlass.Constexpr[int],
    logical_num_heads_q: cutlass.Constexpr[int],
    logical_seq_len_q: Int32,
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
    logical_seq_len_q: Int32,
    cu_seqlens_q=None,
    batch_idx=None,
):
    """Map one physical flat-tile row to logical and public coordinates.

    Returns ``(storage_flat_row, logical_head, safe_local_q, storage_q,
    is_valid)``. Invalid final-tile rows receive safe control-flow coordinates
    but remain predicated from every GMEM transaction by ``is_valid``.
    """

    local_flat_query_row = Int32(query_tile_idx) * Int32(tile_size_q) + Int32(row_in_tile)
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
    storage_flat_query_row = storage_q_idx * Int32(logical_num_heads_q) + logical_head_idx
    is_valid = local_flat_query_row < q_len * Int32(logical_num_heads_q)
    return (
        storage_flat_query_row,
        logical_head_idx,
        safe_logical_q_idx,
        storage_q_idx,
        is_valid,
    )


def captured_loop_bounds(task_kwargs, default_start: int, default_step: int = 1):
    """Consume Task loop kwargs and return captured schedule loop bounds."""
    loop_start = task_kwargs.pop("domain_start", default_start)
    loop_step = task_kwargs.pop("step", default_step)
    loop_end = task_kwargs.pop("domain", None)
    return loop_start, loop_end, loop_step


def work_queue_tail(work_queue, *, advance_label: str = "get_and_advance_work_tile"):
    """Advance the persistent work tile at the end of a task body."""
    work_queue.wait()
    getattr(work_queue, advance_label)()
    work_queue.release()


def staged_qk_mma_k_tile(smem_k, tmem_s, iterations: int):
    """Consume one whole K stage and produce one QK score tile."""
    smem_k.wait()
    tmem_s.acquire()
    for k_subtile_idx in range(iterations):
        desc_k_base = smem_k.k_desc(k_subtile_idx=k_subtile_idx)
        tmem_s.qk_mma(desc_k_base=desc_k_base, k_subtile_idx=k_subtile_idx)
    tmem_s.commit()
    smem_k.release()


def staged_pv_mma_v_tile_per_n(
    smem_v,
    smem_p,
    tmem_o,
    *,
    iterations_pv_k: int,
    iterations_pv_n: int,
):
    """Consume P/V and publish one O pipeline token per PV N-slice."""
    smem_p.wait()
    desc_p_base = smem_p.p_desc()
    smem_v.wait()
    for pv_n_idx in range(iterations_pv_n):
        tmem_o.acquire()
        for pv_k_idx in range(iterations_pv_k):
            desc_v_base = smem_v.v_desc_n_major(pv_n_idx=pv_n_idx, pv_k_idx=pv_k_idx)
            tmem_o.pv_mma_n_major(
                desc_p_base=desc_p_base,
                desc_v_base=desc_v_base,
                pv_n_idx=pv_n_idx,
                pv_k_idx=pv_k_idx,
            )
        tmem_o.commit()
    smem_v.release()
    smem_p.release()


class MlaTaskManager(BaseTaskManager):
    """Retain schedule verification without the advisory overlap note."""


@contextmanager
def mla_build_diagnostics():
    """Hide unused-routing advisories during builds, preserving other warnings.

    MLA carries some task-local state directly rather than through returned
    routes. Suppress only that known schedule-builder advisory; compiler and
    schedule verification failures retain their normal reporting.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r".* produces '.*' but it is never routed to a downstream work call\.",
            category=UserWarning,
            module=r"cutlass\.experimental\.task_scheduling\.schedule_builder",
        )
        yield
