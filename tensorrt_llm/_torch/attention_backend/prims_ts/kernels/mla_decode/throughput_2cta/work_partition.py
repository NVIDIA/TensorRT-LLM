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

"""Runtime nonempty-prefix split-KV helpers for throughput 2CTA MLA."""

import cutlass
import cutlass.cute as cute
from cutlass import Int32


def _validate_partition_inputs(k_tile_total: int, split_kv_cap: int) -> None:
    """Validate host-side split partition inputs."""

    if k_tile_total < 0:
        raise ValueError("k_tile_total must be non-negative")
    if split_kv_cap <= 0:
        raise ValueError("split_kv_cap must be positive")


def active_split_count(k_tile_total: int, split_kv_cap: int) -> int:
    """Return the nonempty prefix under the configured-span partition."""

    _validate_partition_inputs(k_tile_total, split_kv_cap)
    if k_tile_total == 0:
        return 0
    tiles_per_split = (k_tile_total + split_kv_cap - 1) // split_kv_cap
    return (k_tile_total + tiles_per_split - 1) // tiles_per_split


def split_tile_range(
    k_tile_total: int,
    split_kv_cap: int,
    split_idx: int,
) -> tuple[int, int]:
    """Return the configured-span ``(start, count)`` for one split slot."""

    _validate_partition_inputs(k_tile_total, split_kv_cap)
    if split_idx < 0:
        raise ValueError("split_idx must be non-negative")
    tiles_per_split = (k_tile_total + split_kv_cap - 1) // split_kv_cap
    start = split_idx * tiles_per_split
    count = min(tiles_per_split, max(k_tile_total - start, 0))
    return (start, count) if count else (k_tile_total, 0)


def row_prefix_active_split_count(
    row_k_tile_total: int,
    group_k_tile_total: int,
    split_kv_cap: int,
) -> int:
    """Return how many configured-span splits intersect a row's K prefix."""

    _validate_partition_inputs(group_k_tile_total, split_kv_cap)
    if row_k_tile_total < 0:
        raise ValueError("row_k_tile_total must be non-negative")
    row_k_tile_total = min(row_k_tile_total, group_k_tile_total)
    if row_k_tile_total == 0:
        return 0

    tiles_per_split = (group_k_tile_total + split_kv_cap - 1) // split_kv_cap
    # ``row_k_tile_total`` is clipped to the group's K prefix, so its rounded
    # split count cannot exceed the group's active split count. Avoid deriving
    # that second quotient on the hot reducer path.
    return (row_k_tile_total + tiles_per_split - 1) // tiles_per_split


@cute.jit
def runtime_split_kv_cap(
    max_split_kv,
    is_var_split_kv: cutlass.Constexpr[bool],
    block_split_kvs,
    batch_idx,
):
    """Return a positive per-batch cap bounded by launch/workspace capacity."""

    max_split_kv = cute.math.max(Int32(max_split_kv), Int32(1))
    split_kv_cap = max_split_kv
    if cutlass.const_expr(is_var_split_kv):
        split_kv_cap = Int32(block_split_kvs[batch_idx])
    return cute.math.max(
        cute.math.min(split_kv_cap, max_split_kv),
        Int32(1),
    )


@cute.jit
def runtime_active_split_count(k_tile_total, split_kv_cap):
    """Device form of :func:`active_split_count`."""

    k_tile_total = cute.math.max(Int32(k_tile_total), Int32(0))
    split_kv_cap = cute.math.max(Int32(split_kv_cap), Int32(1))
    tiles_per_split = cute.math.max(
        (k_tile_total + split_kv_cap - Int32(1)) // split_kv_cap,
        Int32(1),
    )
    return (k_tile_total + tiles_per_split - Int32(1)) // tiles_per_split


@cute.jit
def runtime_split_tile_range(k_tile_total, split_kv_cap, split_idx):
    """Device form of :func:`split_tile_range`."""

    k_tile_total = cute.math.max(Int32(k_tile_total), Int32(0))
    split_idx = Int32(split_idx)
    split_kv_cap = cute.math.max(Int32(split_kv_cap), Int32(1))
    tiles_per_split = cute.math.max(
        (k_tile_total + split_kv_cap - Int32(1)) // split_kv_cap,
        Int32(1),
    )
    start = split_idx * tiles_per_split
    count = cute.math.min(
        tiles_per_split,
        cute.math.max(k_tile_total - start, Int32(0)),
    )
    start = start if count > Int32(0) else k_tile_total
    return start, count


@cute.jit
def runtime_row_prefix_active_split_count(
    row_k_tile_total,
    group_k_tile_total,
    split_kv_cap,
):
    """Device form of :func:`row_prefix_active_split_count`."""

    group_k_tile_total = cute.math.max(Int32(group_k_tile_total), Int32(0))
    row_k_tile_total = cute.math.max(
        cute.math.min(Int32(row_k_tile_total), group_k_tile_total),
        Int32(0),
    )
    split_kv_cap = cute.math.max(Int32(split_kv_cap), Int32(1))
    tiles_per_split = cute.math.max(
        (group_k_tile_total + split_kv_cap - Int32(1)) // split_kv_cap,
        Int32(1),
    )
    row_active_splits = (
        row_k_tile_total + tiles_per_split - Int32(1)
    ) // tiles_per_split
    # The row K domain was clipped to the group's prefix, making the extra
    # group-active quotient and min redundant.
    return row_active_splits
