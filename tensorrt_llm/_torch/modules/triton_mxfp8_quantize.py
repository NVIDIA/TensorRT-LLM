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
"""MXFP8 activation quantization as a Triton epilogue.

``torch.ops.trtllm.mxfp8_quantize`` is a standalone launch that re-reads an
activation the producing kernel had in registers. A Triton producer can call
:func:`mxfp8_quantize_tile` instead and emit the E4M3 values and their UE8M0
block scales itself.

Every step here reproduces ``cvt_warp_fp16_to_mxfp8`` in
``cpp/tensorrt_llm/kernels/quantization.cuh`` bit for bit, so a fused producer
and the standalone op are interchangeable:

* the block amax is taken over the *storage* dtype, so a producer holding a
  wider accumulator must round to its output dtype before calling in;
* the scale is ``amax * rcp.approx.ftz.f32(448)``, not ``amax / 448``. The two
  differ by an ulp, which the round-up below turns into a whole power of two
  whenever ``amax`` is exactly ``1.75 * 2^k``, so the approximate reciprocal is
  emitted verbatim rather than folded into a Python constant;
* the E8M0 scale rounds toward ``+inf`` onto a power of two, done here in
  integer bit arithmetic to avoid the ``exp2(ceil(log2(x)))`` idiom, which is
  correctly rounded only by luck.

A NaN block amax is the one case left outside that guarantee: the CUDA path
emits the E8M0 NaN encoding where this saturates. Everything else matches,
including the denormal scales that make the reciprocal infinite.
"""

import torch
import triton  # type: ignore[import]
import triton.language as tl  # type: ignore[import]

MXFP8_BLOCK_SIZE = 32
_E4M3_MAX = 448.0

# A jit'ed function can only reach a global that is a tl.constexpr, so the two
# constants above are mirrored for use inside the kernels.
_TL_MXFP8_BLOCK_SIZE = tl.constexpr(MXFP8_BLOCK_SIZE)
_TL_E4M3_MAX = tl.constexpr(_E4M3_MAX)

# The swizzled scale-factor layout is [numMTiles, numKTiles, 32, 4, 4], indexed
# by [mTile, kTile, outerM, innerM, innerK]; see get_sf_out_offset_128x4.
_SF_TILE_ROWS = 128
_SF_TILE_COLS = 4


def swizzled_sf_numel(num_rows: int, num_sf_cols: int) -> int:
    """Elements in the swizzled scale-factor buffer for a [num_rows, K] input.

    ``num_sf_cols`` is ``K // MXFP8_BLOCK_SIZE``. Rows pad to 128 and scale
    columns to 4, matching ``computeSwizzledLayoutSFSize``.
    """
    padded_rows = triton.cdiv(num_rows, _SF_TILE_ROWS) * _SF_TILE_ROWS
    padded_cols = triton.cdiv(num_sf_cols, _SF_TILE_COLS) * _SF_TILE_COLS
    return padded_rows * padded_cols


@triton.jit
def _rcp_approx_ftz(x):
    """``rcp.approx.ftz.f32``, the reciprocal the CUDA quantizer uses."""
    return tl.inline_asm_elementwise(
        "rcp.approx.ftz.f32 $0, $1;",
        "=f,f",
        [x],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )


@triton.jit
def mxfp8_quantize_tile(x, NUM_BLOCKS: tl.constexpr, BLOCK_ELEMS: tl.constexpr):
    """Quantize ``NUM_BLOCKS * BLOCK_ELEMS`` contiguous values of one row.

    ``x`` is float32 holding values already rounded to the producer's storage
    dtype. ``BLOCK_ELEMS`` must be :data:`MXFP8_BLOCK_SIZE`; it is a parameter
    only so the reshape stays a constexpr.

    Returns the E4M3 values in the same order and the ``NUM_BLOCKS`` UE8M0
    scale bytes.
    """
    tl.static_assert(BLOCK_ELEMS == _TL_MXFP8_BLOCK_SIZE)
    xb = tl.reshape(x, (NUM_BLOCKS, BLOCK_ELEMS))
    amax = tl.max(tl.abs(xb), axis=1)

    sf = amax * _rcp_approx_ftz(tl.full((NUM_BLOCKS,), _TL_E4M3_MAX, tl.float32))
    bits = sf.to(tl.int32, bitcast=True)
    biased_exp = (bits >> 23) & 0xFF
    mantissa = bits & 0x7FFFFF
    # Round up onto a power of two. A normal value needs the next exponent
    # whenever it has any mantissa; a denormal rounds up to 2^-127 (byte 0)
    # unless it already exceeds it, 2^-127 being mantissa 0x400000.
    sf_byte = tl.where(
        biased_exp == 0,
        tl.where(mantissa > 0x400000, 1, 0),
        biased_exp + tl.where(mantissa != 0, 1, 0),
    )
    sf_byte = tl.minimum(sf_byte, 254)

    # 1 / 2^(byte - 127) is 2^(127 - byte), exact, so build it from the
    # exponent field directly. Byte 254 lands on a denormal and comes out zero,
    # which is what the flush-to-zero reciprocal gives too.
    out_scale = ((254 - sf_byte) << 23).to(tl.float32, bitcast=True)
    # Byte 0 stands for 2^-127, also denormal, so ftz sends the reciprocal to
    # infinity rather than 2^127 and every element of the block saturates. Only
    # a vanishing amax gets here, but the two forms have to agree even so.
    out_scale = tl.where(sf_byte == 0, tl.full((NUM_BLOCKS,), float("inf"), tl.float32), out_scale)
    # An all-zero block takes scale 0, as the CUDA path does.
    out_scale = tl.where(amax != 0.0, out_scale, 0.0)

    xq = xb * out_scale[:, None]
    return tl.reshape(xq, (NUM_BLOCKS * BLOCK_ELEMS,)).to(tl.float8e4nv), sf_byte.to(tl.uint8)


@triton.jit
def swizzled_sf_offset(row, k_idx, num_k_tiles):
    """Offsets of one row's scale bytes at scale-column indices ``k_idx``."""
    inner_k = k_idx % 4
    k_tile = k_idx // 4
    inner_m = (row % 128) // 32
    outer_m = row % 32
    m_tile = row // 128
    return m_tile * num_k_tiles * 512 + k_tile * 512 + outer_m * 16 + inner_m * 4 + inner_k


@triton.jit
def _mxfp8_quantize_kernel(
    x_ptr,
    out_ptr,
    sf_ptr,
    stride_x_m,
    stride_x_k,
    stride_out_m,
    stride_out_k,
    num_k_tiles,
    CHUNK: tl.constexpr,
    NUM_BLOCKS: tl.constexpr,
):
    pid_m, pid_k = tl.program_id(0), tl.program_id(1)
    off = pid_k * CHUNK + tl.arange(0, CHUNK)
    x = tl.load(x_ptr + pid_m * stride_x_m + off * stride_x_k).to(tl.float32)
    xq, sf = mxfp8_quantize_tile(x, NUM_BLOCKS, CHUNK // NUM_BLOCKS)
    tl.store(out_ptr + pid_m * stride_out_m + off * stride_out_k, xq)
    k_idx = pid_k * NUM_BLOCKS + tl.arange(0, NUM_BLOCKS)
    tl.store(sf_ptr + swizzled_sf_offset(pid_m, k_idx, num_k_tiles), sf)


def mxfp8_quantize_triton(x: torch.Tensor, chunk: int = 128):
    """Whole-tensor MXFP8 quantize through :func:`mxfp8_quantize_tile`.

    This exists to pin the epilogue's numerics against
    ``torch.ops.trtllm.mxfp8_quantize``; production callers fuse the tile
    function into their own kernel instead of launching this one.
    """
    if x.dim() != 2 or not x.is_contiguous():
        raise ValueError("expected a contiguous 2D activation")
    m, k = x.shape
    if k % chunk or chunk % MXFP8_BLOCK_SIZE:
        raise ValueError(
            f"K ({k}) must be a multiple of chunk ({chunk}), itself "
            f"a multiple of {MXFP8_BLOCK_SIZE}"
        )
    sf_cols = k // MXFP8_BLOCK_SIZE
    out = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    # Zeroed rather than empty: the rows between m and the 128 boundary are
    # scale-factor padding that no program writes.
    sf = torch.zeros(swizzled_sf_numel(m, sf_cols), dtype=torch.uint8, device=x.device)
    _mxfp8_quantize_kernel[(m, k // chunk)](
        x,
        out,
        sf,
        x.stride(0),
        x.stride(1),
        out.stride(0),
        out.stride(1),
        triton.cdiv(sf_cols, _SF_TILE_COLS),
        CHUNK=chunk,
        NUM_BLOCKS=chunk // MXFP8_BLOCK_SIZE,
    )
    return out, sf


__all__ = [
    "MXFP8_BLOCK_SIZE",
    "mxfp8_quantize_tile",
    "mxfp8_quantize_triton",
    "swizzled_sf_numel",
    "swizzled_sf_offset",
]
