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


"""Low-level helper intrinsics for FMHA context TS resources."""

import cutlass
import cutlass.cute as cute
from cutlass import Int32, Int64


@cute.jit
def variable_window_cta_min_start(
    cta_starts: cute.Tensor,
    *,
    batch_coord: Int32,
    seq_coord: Int32,
    q_stride: int | Int32,
    tile_size_q: cutlass.Constexpr[int],
) -> Int32:
    """Load the plan-time minimum variable-window start for one Q CTA."""
    num_seq_tiles = cute.ceil_div(q_stride, tile_size_q)
    return Int32(cta_starts[batch_coord * num_seq_tiles + seq_coord])


def bottom_right_window_left_bound(
    query_idx: int | Int32,
    q_offset: int | Int32,
    window_size_left: int,
) -> int | Int32:
    """Return the inclusive left bound for bottom-right causal attention."""
    return query_idx + q_offset - window_size_left


def bottom_right_window_tile_start(
    *,
    seq_coord: int | Int32,
    q_tile_m: int | Int32,
    kv_tile_n: int | Int32,
    q_offset: int | Int32,
    window_size_left: int,
) -> int | Int32:
    """Return the first K/V tile intersecting a bottom-right left window."""
    raw_start = (
        bottom_right_window_left_bound(
            seq_coord * q_tile_m,
            q_offset,
            window_size_left,
        )
        // kv_tile_n
    )
    if isinstance(raw_start, int):
        return max(0, raw_start)
    return cute.math.max(Int32(0), raw_start)


def bottom_right_window_max_tiles(
    *,
    q_tile_m: int,
    kv_tile_n: int,
    window_size_left: int,
) -> int:
    """Return the offset-independent maximum K/V span for one Q tile.

    The visible interval before sequence-boundary clipping is inclusive and
    has ``window_size_left + q_tile_m`` tokens.  Its alignment against a K/V
    tile can require one additional tile at each boundary, so the maximum
    intersecting span is ``ceil((length + kv_tile_n - 1) / kv_tile_n)``.
    Packed-ragged scheduling uses this bound because each request can have a
    different bottom-right Q/K offset while a task must keep one loop domain.
    """
    if q_tile_m <= 0 or kv_tile_n <= 0 or window_size_left < 0:
        raise ValueError("tile sizes must be positive and window size non-negative")
    numerator = window_size_left + q_tile_m + kv_tile_n - 1
    return (numerator + kv_tile_n - 1) // kv_tile_n


@cute.jit
def freeze_smem_descriptor(desc):
    """Copy a shared-memory descriptor through a register to prevent rematerialization."""
    return cute.arch.inline_ptx(
        "mov.b64 {$w0}, {$r0};",
        write_only_types=[Int64],
        read_only_args=[desc],
    )
