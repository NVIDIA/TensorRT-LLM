# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""TRTLLM-compatible MXFP8 quantization with configurable work per lane.

The format fixes the scale group; lane width controls its thread mapping.
The exponent-zero reciprocal follows quantization.cuh::exp2f_rcp exactly.
"""

import math

import cutlass
import cutlass.cute as cute
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import dsl_user_op

# E4M3/UE8M0 MXFP8 defines one scale per 32 values.
MXFP8_SCALE_GROUP_SIZE = 32


@dsl_user_op
def _quant_scale(amax, *, loc=None, ip=None):
    """Return the packed E8M0 byte using the native quantizer's PTX rounding."""
    bits = llvm.inline_asm(
        cutlass.Int32.mlir_type,
        [amax.ir_value()],
        "{ .reg .f32 r, s; .reg .b16 h; "
        "rcp.approx.ftz.f32 r, 0f43e00000; "
        "mul.f32 s, $1, r; "
        "cvt.rp.satfinite.ue8m0x2.f32 h, s, s; "
        "cvt.u32.u16 $0, h; and.b32 $0, $0, 255; }",
        "=r,f",
        has_side_effects=False,
        loc=loc,
        ip=ip,
    )
    return cutlass.Int32(bits)


@dsl_user_op
def _scale_reciprocal(exponent, *, loc=None, ip=None):
    """Decode reciprocal powers of two, preserving subnormal reciprocals."""
    value = llvm.inline_asm(
        cutlass.Float32.mlir_type,
        [exponent.ir_value()],
        "{ .reg .b32 e, b; .reg .pred p; "
        "sub.s32 e, 254, $1; shl.b32 b, e, 23; "
        "setp.eq.s32 p, $1, 254; selp.b32 b, 0x00400000, b, p; "
        "setp.eq.s32 p, $1, 255; selp.b32 b, 0x00200000, b, p; "
        "setp.eq.s32 p, $1, 0; selp.b32 b, 0x3f800000, b, p; "
        "mov.b32 $0, b; }",
        "=f,r",
        has_side_effects=False,
        loc=loc,
        ip=ip,
    )
    return cutlass.Float32(value)


@cute.jit
def quantize_mxfp8_chunk(
    src: cute.Tensor,
    dst: cute.Tensor,
    valid: cutlass.Boolean,
    values_per_lane: cutlass.Constexpr,
):
    """Quantize a configurable lane chunk; every warp lane must participate.

    Lane groups are derived from the MXFP8 format. Invalid lanes contribute zero
    and store nothing. A balanced tree reduces the local dependency depth.
    """
    values = cute.make_rmem_tensor((values_per_lane,), cutlass.Float32)
    raw = cute.make_rmem_tensor((values_per_lane,), src.element_type)
    raw.fill(0)
    if valid:
        cute.autovec_copy(src, raw)
    values.store(raw.load().to(cutlass.Float32))
    maxima = cute.make_rmem_tensor((values_per_lane,), cutlass.Float32)
    for i in cutlass.range_constexpr(values_per_lane):
        maxima[i] = cute.math.abs(values[i])
    for stage in cutlass.range_constexpr(int(math.log2(values_per_lane))):
        for i in cutlass.range_constexpr(values_per_lane >> (stage + 1)):
            maxima[i] = cutlass.max(maxima[2 * i], maxima[2 * i + 1])
    maximum = cutlass.max(cutlass.Float32(0), maxima[0])
    lanes_per_scale = MXFP8_SCALE_GROUP_SIZE // values_per_lane
    for stage in cutlass.range_constexpr(int(math.log2(lanes_per_scale))):
        maximum = cutlass.max(maximum, cute.arch.shuffle_sync_bfly(maximum, 1 << stage))
    exponent = _quant_scale(maximum)
    inverse = _scale_reciprocal(exponent)
    if maximum == 0:
        inverse = cutlass.Float32(0)
    packed = cute.make_rmem_tensor((values_per_lane,), cutlass.Float8E4M3FN)
    packed.store((values.load() * inverse).to(cutlass.Float8E4M3FN))
    if valid:
        cute.autovec_copy(packed, dst)
    scale = cute.make_rmem_tensor((1,), cutlass.Uint8)
    scale[0] = exponent.to(cutlass.Uint8)
    return cute.recast_tensor(scale, cutlass.Float8E8M0FNU)[0]


@cute.jit
def quantize_mxfp8_16(src: cute.Tensor, dst: cute.Tensor, valid: cutlass.Boolean):
    """Preserve the paired-lane interface for callers with 16-value slices."""
    return quantize_mxfp8_chunk(src, dst, valid, 16)
