# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""The MXFP8 form of BF16 rows a CuTe DSL kernel already holds in registers, bit for bit
``trtllm::mxfp8_quantize(x, True)`` (``cvt_warp_fp16_to_mxfp8<bf16, 32>``).

Each thread holds 8-element vectors, so a 32-element scale block spans four consecutive lanes. Per block:
    amax  = max |x| over the block (abs.f32 per element, fmax across the four lanes)
    scale = ue8m0(amax * rcp.approx.ftz(448)), rounded toward +inf, satfinite
    q     = e4m3(x * rcp.approx.ftz(2^(scale - 127))), RN, satfinite; 0 where scale is 0 (amax <= 448 * 2^-127, an
            all-zero block included), whose 2^-127 is a denormal that rcp.approx.ftz returns +inf for
The E4M3 bytes keep element order (byte 0 = element 0, as ``__nv_fp8x2_e4m3``), 8 per thread in one 64-bit store; the
scales go to the swizzled 128x4 layout (``get_sf_out_offset_128x4``), whose 4-byte words each hold one row's scales of
four consecutive blocks, so each word is one store; the scale rows that pad the row count to a multiple of 128 are
zero, as ``quantize_mxfp8_rows`` leaves them. Rows are a multiple of 128 wide, so every word of a row holds real
scales.
"""

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int32, Int64, Uint32
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op

#: BF16 elements per thread vector; four vectors make one 32-element scale block.
VEC = 8


@dsl_user_op
def rcp_approx_ftz(x: Float32, *, loc=None, ip=None) -> Float32:
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Float32(x).ir_value(loc=loc, ip=ip)],
            "rcp.approx.ftz.f32 $0, $1;",
            "=f,f",
            has_side_effects=False,
            is_align_stack=False,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _abs_f32(x: Float32, *, loc=None, ip=None) -> Float32:
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Float32(x).ir_value(loc=loc, ip=ip)],
            "abs.f32 $0, $1;",
            "=f,f",
            has_side_effects=False,
            is_align_stack=False,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _ue8m0_rp(x: Float32, *, loc=None, ip=None) -> Uint32:
    """The UE8M0 byte of x rounded toward +inf, satfinite, in the low byte."""
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [Float32(x).ir_value(loc=loc, ip=ip)],
            "{\n\t.reg .b16 r;\n\tcvt.rp.satfinite.ue8m0x2.f32 r, 0f00000000, $1;\n\tcvt.u32.u16 $0, r;\n\t}\n",
            "=r,f",
            has_side_effects=False,
            is_align_stack=False,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _e4m3x2(lo: Float32, hi: Float32, *, loc=None, ip=None) -> Uint32:
    """E4M3 (RN, satfinite) of lo in byte 0 and of hi in byte 1."""
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [Float32(hi).ir_value(loc=loc, ip=ip), Float32(lo).ir_value(loc=loc, ip=ip)],
            "{\n\t.reg .b16 r;\n\tcvt.rn.satfinite.e4m3x2.f32 r, $1, $2;\n\tcvt.u32.u16 $0, r;\n\t}\n",
            "=r,f,f",
            has_side_effects=False,
            is_align_stack=False,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _f32_from_bits(bits: Uint32, *, loc=None, ip=None) -> Float32:
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Uint32(bits).ir_value(loc=loc, ip=ip)],
            "mov.b32 $0, $1;",
            "=f,r",
            has_side_effects=False,
            is_align_stack=False,
            loc=loc,
            ip=ip,
        )
    )


def scale_bytes(rows: int, hidden_size: int) -> int:
    """The swizzled UE8M0 scales of ``rows`` rows: whole 128-row tiles of hidden_size / 32 columns padded to 4."""
    return (rows + 127) // 128 * 128 * ((hidden_size // 32 + 3) // 4 * 4)


def _sf_word(row, k_tile, num_k_tiles):
    """The word of ``row``'s scales 4 k_tile .. 4 k_tile + 3 in get_sf_out_offset_128x4's layout ([m tiles, k tiles,
    32 (m), 4 (m), 4 (k)] bytes), counted in 4-byte words."""
    return (row // 128) * (num_k_tiles * 128) + k_tile * 128 + (row % 32) * 4 + (row % 128) // 32


def _sf_words(mSF: cute.Tensor) -> cute.Tensor:
    return cute.make_tensor(
        cute.recast_ptr(mSF.iterator, dtype=Int32), cute.make_layout((mSF.shape[0] // 4,))
    )


@cute.jit
def store_mxfp8_vector(
    out: cute.Tensor,
    mQ: cute.Tensor,
    mSF: cute.Tensor,
    row: Int32,
    v: Int32,
    num_k_tiles: cutlass.Constexpr[int],
    rcp_448: Float32,
):
    """Quantize this thread's ``VEC`` BF16 values ``out`` (vector ``v`` of ``row``) into ``mQ`` (Int64 words of 8
    E4M3 bytes, [rows, H / 8]) and the scales into ``mSF``: the first lane of every 16 writes the word of its four
    blocks' scales. Every lane of the warp calls it for the same vector slot (the block maximum and the scale word
    are warp shuffles)."""
    x = cute.make_rmem_tensor((VEC,), Float32)
    x.store(out.load().to(Float32))
    amax = _abs_f32(x[0])
    for j in cutlass.range_constexpr(1, VEC):
        amax = cute.arch.fmax(amax, _abs_f32(x[j]))
    for step in cutlass.range_constexpr(2):
        other = cute.arch.shuffle_sync_bfly(amax, offset=1 << step, mask=-1, mask_and_clamp=31)
        amax = cute.arch.fmax(other, amax)
    scale_byte = _ue8m0_rp(amax * rcp_448) & Uint32(0xFF)
    out_scale = Float32(0.0)
    if scale_byte != Uint32(0):
        out_scale = rcp_approx_ftz(_f32_from_bits(scale_byte << Uint32(23)))
    w0 = _e4m3x2(x[0] * out_scale, x[1] * out_scale) | (
        _e4m3x2(x[2] * out_scale, x[3] * out_scale) << 16
    )
    w1 = _e4m3x2(x[4] * out_scale, x[5] * out_scale) | (
        _e4m3x2(x[6] * out_scale, x[7] * out_scale) << 16
    )
    mQ[row, v] = (Int64(w1) << 32) | (Int64(w0) & Int64(0xFFFFFFFF))
    # Lanes 4 / 8 / 12 past a multiple of 16 hold the next three blocks' scales.
    scale = Int32(scale_byte)
    s1 = cute.arch.shuffle_sync_down(scale, offset=4, mask=-1, mask_and_clamp=31)
    s2 = cute.arch.shuffle_sync_down(scale, offset=8, mask=-1, mask_and_clamp=31)
    s3 = cute.arch.shuffle_sync_down(scale, offset=12, mask=-1, mask_and_clamp=31)
    if cute.arch.lane_idx() % 16 == 0:
        words = _sf_words(mSF)
        words[_sf_word(row, v // 16, num_k_tiles)] = scale | (s1 << 8) | (s2 << 16) | (s3 << 24)


@cute.jit
def zero_mxfp8_pad_rows(
    mSF: cute.Tensor,
    row: Int32,
    num_rows: Int32,
    tidx: Int32,
    num_threads: cutlass.Constexpr[int],
    num_k_tiles: cutlass.Constexpr[int],
):
    """Zero the scales of the rows that pad ``num_rows`` to a multiple of 128. They all sit in the last 128-row tile,
    whose words each hold one row's scales: the threads of the ``num_rows`` CTAs take consecutive words and zero those
    of pad rows, so the stores coalesce."""
    tile_words: cutlass.Constexpr[int] = num_k_tiles * 128
    last_tile = num_rows // 128
    rows_in_tile = num_rows - last_tile * 128
    if rows_in_tile != 0:
        words = _sf_words(mSF)
        base = last_tile * tile_words
        for word in range(row * num_threads + tidx, tile_words, num_rows * num_threads):
            w = word % 128
            # Word w of a k tile holds row (w % 4) * 32 + w // 4 of the 128-row tile.
            if (w % 4) * 32 + w // 4 >= rows_in_tile:
                words[base + word] = Int32(0)


def mxfp8_fakes(hidden_size: int, sym_m):
    """Fake (q as Int64 words [M, H / 8], scales as bytes) for compiling a kernel that writes MXFP8 rows."""
    q = cute.runtime.make_fake_compact_tensor(
        Int64, (sym_m, hidden_size // 8), stride_order=(1, 0), assumed_align=16
    )
    sf = cute.runtime.make_fake_compact_tensor(cutlass.Uint8, (cute.sym_int(),), assumed_align=16)
    return q, sf


def check_mxfp8_rows(q: torch.Tensor, sf: torch.Tensor, num_rows: int, hidden_size: int) -> None:
    if hidden_size % 128 != 0:
        raise ValueError(f"MXFP8 rows need a width that is a multiple of 128, got {hidden_size}")
    if (
        q.dtype != torch.float8_e4m3fn
        or tuple(q.shape) != (num_rows, hidden_size)
        or not q.is_contiguous()
    ):
        raise ValueError(f"q must be a contiguous float8_e4m3fn [{num_rows}, {hidden_size}]")
    need = scale_bytes(num_rows, hidden_size)
    if sf.dtype != torch.uint8 or not sf.is_contiguous() or sf.numel() < need:
        raise ValueError(f"sf must be contiguous uint8 with at least {need} bytes")
    if q.data_ptr() % 16 != 0 or sf.data_ptr() % 16 != 0:
        raise ValueError("q and sf must be 16-byte aligned")


__all__ = [
    "VEC",
    "check_mxfp8_rows",
    "mxfp8_fakes",
    "rcp_approx_ftz",
    "scale_bytes",
    "store_mxfp8_vector",
    "zero_mxfp8_pad_rows",
]
