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
from cutlass import Float16, Float32, Int32, Int64, Uint32
from cutlass.cutlass_dsl import Constexpr


@cute.jit
def variable_window_cta_min_start(
    cta_starts: cute.Tensor,
    *,
    batch_coord: Int32,
    seq_coord: Int32,
    q_stride: int | Int32,
    tile_size_q: cutlass.Constexpr[int],
) -> Int32:
    """Load the per-run minimum variable-window start for one Q CTA."""
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


@cute.jit
def split_bf16_hi_lo_word(value: Float32) -> Int32:
    """Pack ``value`` as two bf16 halves: low 16 bits = truncated high part,
    high 16 bits = truncated remainder ``value - hi``."""
    bits = cutlass.Vector.from_elements((value,), Float32).bitcast(Uint32)[0]
    hi_bits = bits >> Uint32(16)
    hi = cutlass.Vector.from_elements((hi_bits << Uint32(16),), Uint32).bitcast(
        Float32
    )[0]
    lo_bits = cutlass.Vector.from_elements((value - hi,), Float32).bitcast(Uint32)[0]
    word = hi_bits | (lo_bits & Uint32(0xFFFF0000))
    return cutlass.Vector.from_elements((word,), Uint32).bitcast(Int32)[0]


@cute.jit
def expcast_e4m3_quad_relu(
    c0: Float32,
    c1: Float32,
    c2: Float32,
    c3: Float32,
    acc01: Int32,
    acc23: Int32,
    init: Constexpr[bool],
) -> tuple[Int32, Int32, Int32]:
    """ExpCast four unclamped codes into one packed E4M3 word and accumulate
    their values into two f16x2 row-sum registers.

    With ``init`` the quad starts a chain and the accumulators are ignored. The
    returned sums are then the quantized values themselves.

    ``c = 8 * (s * scale - m) + VC_EXPCAST_CODE_BIAS``

    ``cvt.rn.relu.f16x2.f32`` packs a pair and clamps underflow to code 0, an
    f16 FMA adds 1024 so the low byte of each half is the rounded code, one
    ``prmt`` gathers the four bytes, and the E4M3-pair to f16x2 conversion
    with one f16 FMA per pair accumulates the quantized values.
    """
    sum_pair = (
        "mov.b32 {$w1}, v01;\n            mov.b32 {$w2}, v23;"
        if init
        else "fma.rn.f16x2 {$w1}, v01, one2, {$r4};\n"
        "            fma.rn.f16x2 {$w2}, v23, one2, {$r5};"
    )
    return cute.arch.inline_ptx(
        """
        {
            .reg .b32 h01, h23, y01, y23, v01, v23, one2, k1024;
            .reg .b16 c01, c23;
            mov.b32 one2, 0x3C003C00;
            mov.b32 k1024, 0x64006400;
            cvt.rn.relu.f16x2.f32 h01, {$r1}, {$r0};
            cvt.rn.relu.f16x2.f32 h23, {$r3}, {$r2};
            fma.rn.f16x2 y01, h01, one2, k1024;
            fma.rn.f16x2 y23, h23, one2, k1024;
            prmt.b32 {$w0}, y01, y23, 0x6420;
            mov.b32 {c01, c23}, {$w0};
            cvt.rn.f16x2.e4m3x2 v01, c01;
            cvt.rn.f16x2.e4m3x2 v23, c23;
            """
        + sum_pair
        + """
        }""",
        write_only_types=[Int32, Int32, Int32],
        read_only_args=[c0, c1, c2, c3, acc01, acc23],
    )


@cute.jit
def f16x2_sum4(a0: Int32, a1: Int32, a2: Int32, a3: Int32) -> Float32:
    """fp32 sum of the eight fp16 halves of four f16x2 registers."""
    f = []
    for word in (a0, a1, a2, a3):
        halves = cutlass.Vector.from_elements((word,), Int32).bitcast(Float16)
        f.append(halves[0].to(Float32))
        f.append(halves[1].to(Float32))
    return ((f[0] + f[1]) + (f[2] + f[3])) + ((f[4] + f[5]) + (f[6] + f[7]))
