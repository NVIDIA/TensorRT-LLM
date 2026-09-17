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
"""Strided SM100 MXFP8 packing with the native block-32 quantization arithmetic."""

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op

# Output registers 0/1: eight E4M3 bytes; register 2: one E8M0 byte.
# Inputs 3..6: four pairs of BF16; input 7: lane index. No fast-math rewrite.
_QUANT_ASM = r"""
{
 .reg .b32 a0,a1,a2,a3,m,n,lane,mask,tmp,bits;
 .reg .b16 sf,p0,p1,p2,p3;
 .reg .f32 lo,hi,amax,rcp448,scale,mult;
 .reg .pred choose_lo,is_zero;
 and.b32 a0,$3,0x7fff7fff;
 and.b32 a1,$4,0x7fff7fff;
 and.b32 a2,$5,0x7fff7fff;
 and.b32 a3,$6,0x7fff7fff;
 max.bf16x2 m,a0,a1;
 max.bf16x2 m,m,a2;
 max.bf16x2 m,m,a3;
 and.b32 lane,$7,28;
 mov.b32 mask,15;
 shl.b32 mask,mask,lane;
 shfl.sync.bfly.b32 n,m,1,31,mask;
 max.bf16x2 m,n,m;
 shfl.sync.bfly.b32 n,m,2,31,mask;
 max.bf16x2 m,n,m;
 shl.b32 bits,m,16;
 mov.b32 lo,bits;
 and.b32 bits,m,0xffff0000;
 mov.b32 hi,bits;
 setp.gt.f32 choose_lo,lo,hi;
 selp.f32 amax,lo,hi,choose_lo;
 rcp.approx.ftz.f32 rcp448,0f43e00000;
 mul.rn.f32 scale,amax,rcp448;
 cvt.rp.satfinite.ue8m0x2.f32 sf,0f00000000,scale;
 mov.b32 $2,{sf,0};
 and.b32 $2,$2,255;
 cvt.rn.bf16x2.ue8m0x2 tmp,sf;
 shl.b32 bits,tmp,16;
 mov.b32 scale,bits;
 rcp.approx.ftz.f32 mult,scale;
 setp.eq.f32 is_zero,amax,0f00000000;
 @is_zero mov.f32 mult,0f00000000;
"""
for _pair in range(4):
    _reg = _pair + 3
    _QUANT_ASM += f"""
 shl.b32 bits,${_reg},16;
 mov.b32 lo,bits;
 and.b32 bits,${_reg},0xffff0000;
 mov.b32 hi,bits;
 mul.rn.f32 lo,lo,mult;
 mul.rn.f32 hi,hi,mult;
 cvt.rn.satfinite.e4m3x2.f32 p{_pair},hi,lo;
"""
_QUANT_ASM += "mov.b32 $0,{p0,p1};\n mov.b32 $1,{p2,p3};\n}"


@dsl_user_op
def quantize_eight(p0, p1, p2, p3, lane, *, loc=None, ip=None):
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32()] * 3),
        [cutlass.Uint32(x).ir_value(loc=loc, ip=ip) for x in (p0, p1, p2, p3, lane)],
        _QUANT_ASM,
        "=&r,=&r,=&r,r,r,r,r,r",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return tuple(
        cutlass.Uint32(llvm.extractvalue(T.i32(), result, [i], loc=loc, ip=ip)) for i in range(3)
    )


@cute.jit
def sf_offset(m, k, columns):
    # Widen before multiplication, including for views beyond 2**31 elements.
    m64, k64, c64 = cutlass.Int64(m), cutlass.Int64(k), cutlass.Int64(columns)
    return (
        (m64 // 128) * ((c64 + 3) // 4) * 512
        + (k64 // 4) * 512
        + (m64 % 32) * 16
        + ((m64 % 128) // 32) * 4
        + k64 % 4
    )


@cute.kernel
def qk_kernel(x: cute.Tensor, out: cute.Tensor, sf: cute.Tensor):
    tid, _, _ = cute.arch.thread_idx()
    tile, bh, _ = cute.arch.block_idx()
    b = cutlass.Int64(bh) // x.shape[1]
    h = cutlass.Int64(bh) % x.shape[1]
    values = cute.make_rmem_tensor((8,), cutlass.BFloat16)
    pairs = cute.recast_tensor(values, cutlass.Uint32)
    encoded = cute.make_rmem_tensor((2,), cutlass.Uint32)
    octets = cute.recast_tensor(encoded, cutlass.Uint8)
    load_atom = cute.make_copy_atom(
        cute.nvgpu.CopyUniversalOp(), cutlass.BFloat16, num_bits_per_copy=128
    )
    store_atom = cute.make_copy_atom(
        cute.nvgpu.CopyUniversalOp(), cutlass.Uint8, num_bits_per_copy=64
    )
    for r in cutlass.range_constexpr(4):
        s = cutlass.Int64(tile) * 32 + r * 8 + tid // 16
        d = cutlass.Int64(tid % 16) * 8
        for i in cutlass.range_constexpr(8):
            values[i] = cutlass.BFloat16(0)
        if s < x.shape[2]:
            packet = cute.zipped_divide(x[b, h, s, None], (8,))[(None,), (d // 8,)]
            cute.copy(load_atom, packet, values)
        a, z, scale = quantize_eight(pairs[0], pairs[1], pairs[2], pairs[3], tid % 32)
        encoded[0], encoded[1] = a, z
        packet = cute.zipped_divide(out[b, h, s, None], (8,))[(None,), (d // 8,)]
        cute.copy(store_atom, octets, packet)
        if tid % 4 == 0:
            row = cutlass.Int64(bh) * out.shape[2] + s
            sf[sf_offset(row, d // 32, 4)] = cutlass.Uint8(scale)


@cute.kernel
def v_kernel(x: cute.Tensor, out: cute.Tensor, sf: cute.Tensor):
    tid, _, _ = cute.arch.thread_idx()
    tile, bh, _ = cute.arch.block_idx()
    b = cutlass.Int64(bh) // x.shape[1]
    h = cutlass.Int64(bh) % x.shape[1]
    allocator = utils.SmemAllocator()
    src = allocator.allocate_tensor(cutlass.BFloat16, cute.make_layout((4096,)), byte_alignment=128)
    dst = allocator.allocate_tensor(cutlass.Uint8, cute.make_layout((4096,)), byte_alignment=128)
    global_values = cute.make_rmem_tensor((8,), cutlass.BFloat16)
    global_octets = cute.make_rmem_tensor((8,), cutlass.Uint8)
    load_atom = cute.make_copy_atom(
        cute.nvgpu.CopyUniversalOp(), cutlass.BFloat16, num_bits_per_copy=128
    )
    store_atom = cute.make_copy_atom(
        cute.nvgpu.CopyUniversalOp(), cutlass.Uint8, num_bits_per_copy=64
    )
    # Coalesced D loads; XOR changes only shared-memory locations, not pairs.
    for r in cutlass.range_constexpr(4):
        sl = r * 8 + tid // 16
        s = cutlass.Int64(tile) * 32 + sl
        d0 = (tid % 16) * 8
        for i in cutlass.range_constexpr(8):
            global_values[i] = cutlass.BFloat16(0)
        if s < x.shape[2]:
            packet = cute.zipped_divide(x[b, h, s, None], (8,))[(None,), (cutlass.Int64(d0) // 8,)]
            cute.copy(load_atom, packet, global_values)
        address = 2 * (sl * 64 + ((d0 // 2) ^ ((sl // 8) * 8))) + d0 % 2
        packet = cute.zipped_divide(src, (8,))[(None,), (address // 8,)]
        cute.copy(load_atom, global_values, packet)
    cute.arch.sync_threads()
    values = cute.make_rmem_tensor((8,), cutlass.BFloat16)
    pairs = cute.recast_tensor(values, cutlass.Uint32)
    encoded = cute.make_rmem_tensor((2,), cutlass.Uint32)
    octets = cute.recast_tensor(encoded, cutlass.Uint8)
    for j in cutlass.range_constexpr(4):
        d = tid // 4 + 32 * j
        for i in cutlass.range_constexpr(8):
            sl = 8 * (tid % 4) + i
            address = 2 * (sl * 64 + ((d // 2) ^ ((sl // 8) * 8))) + d % 2
            values[i] = src[address]
        a, z, scale = quantize_eight(pairs[0], pairs[1], pairs[2], pairs[3], tid % 32)
        encoded[0], encoded[1] = a, z
        for i in cutlass.range_constexpr(8):
            sl = 8 * (tid % 4) + i
            address = 4 * (sl * 32 + ((d // 4) ^ ((sl // 8) * 8))) + d % 4
            dst[address] = octets[i]
        if tid % 4 == 0:
            row = cutlass.Int64(bh) * 128 + d
            columns = ((cutlass.Int64(x.shape[2]) + 127) // 128) * 4
            sf[sf_offset(row, tile, columns)] = cutlass.Uint8(scale)
    cute.arch.sync_threads()
    for r in cutlass.range_constexpr(4):
        sl = r * 8 + tid // 16
        s = cutlass.Int64(tile) * 32 + sl
        d0 = (tid % 16) * 8
        if s < x.shape[2]:
            address = 4 * (sl * 32 + ((d0 // 4) ^ ((sl // 8) * 8))) + d0 % 4
            packet = cute.zipped_divide(dst, (8,))[(None,), (address // 8,)]
            cute.copy(store_atom, packet, global_octets)
            packet = cute.zipped_divide(out[b, h, s, None], (8,))[
                (None,), (cutlass.Int64(d0) // 8,)
            ]
            cute.copy(store_atom, global_octets, packet)


@cute.jit
def launch_qk(x: cute.Tensor, out: cute.Tensor, sf: cute.Tensor, stream: cuda.CUstream):
    qk_kernel(x, out, sf).launch(
        grid=((x.shape[2] + 127) // 128 * 4, x.shape[0] * x.shape[1], 1),
        block=(128, 1, 1),
        stream=stream,
        use_pdl=False,
    )


@cute.jit
def launch_v(x: cute.Tensor, out: cute.Tensor, sf: cute.Tensor, stream: cuda.CUstream):
    v_kernel(x, out, sf).launch(
        grid=((x.shape[2] + 127) // 128 * 4, x.shape[0] * x.shape[1], 1),
        block=(128, 1, 1),
        stream=stream,
        use_pdl=False,
    )
