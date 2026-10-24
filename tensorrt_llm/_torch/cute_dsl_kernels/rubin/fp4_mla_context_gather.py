# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Canonical FP4 cache gather with MXFP8 activation and unit-scale FP8 RoPE outputs."""

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass._mlir.dialects import llvm
from cutlass.cute.runtime import make_ptr
from cutlass.cutlass_dsl import T, dsl_user_op

from ..blackwell.dense_gemm_persistent import _round_up_ue8m0, _u32_as_f32


@dsl_user_op
def _rcp(value, *, loc=None, ip=None):
    return cutlass.Float32(
        llvm.inline_asm(
            T.f32(),
            [cutlass.Float32(value).ir_value(loc=loc, ip=ip)],
            "rcp.approx.ftz.f32 $0, $1;",
            "=f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _div_full(value, denominator, *, loc=None, ip=None):
    return cutlass.Float32(
        llvm.inline_asm(
            T.f32(),
            [
                cutlass.Float32(value).ir_value(loc=loc, ip=ip),
                cutlass.Float32(denominator).ir_value(loc=loc, ip=ip),
            ],
            "div.full.f32 $0, $1, $2;",
            "=f,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@cute.jit
def _sf_offset(row, col, columns: cutlass.Constexpr):
    return (
        col % 4
        + col // 4 * 512
        + row % 32 * 16
        + row % 128 // 32 * 4
        + row // 128 * (128 * columns)
    )


@cute.jit
def _decode_fp4(code):
    mag = code & 7
    bits = ((mag // 2 + 126) << 23) | ((mag & 1) << 22)
    if mag < 2:
        bits = mag * 0x3F000000
    bits = bits.to(cutlass.Uint32) | ((code.to(cutlass.Uint32) & 8) << 28)
    return _u32_as_f32(bits)


@cute.jit
def _fp8(value):
    if value > 448.0:
        value = cutlass.Float32(448.0)
    if value < -448.0:
        value = cutlass.Float32(-448.0)
    return value.to(cutlass.Float8E4M3FN)


@cute.kernel
def _gather(
    a_ptr: cute.Pointer,
    asc_ptr: cute.Pointer,
    pe_ptr: cute.Pointer,
    kv_ptr: cute.Pointer,
    sf_ptr: cute.Pointer,
    pages_ptr: cute.Pointer,
    cu_ptr: cute.Pointer,
    offsets_ptr: cute.Pointer,
    global_ptr: cute.Pointer,
    m: cutlass.Int64,
    max_chunk: cutlass.Int64,
    table_stride: cutlass.Int64,
    num_pages: cutlass.Int64,
    page_size: cutlass.Constexpr,
    kv_s0: cutlass.Int64,
    kv_s2: cutlass.Int64,
    kv_s4: cutlass.Constexpr,
    sf_s0: cutlass.Int64,
    warps: cutlass.Constexpr,
):
    tid, _, _ = cute.arch.thread_idx()
    block, batch, _ = cute.arch.block_idx()
    lane = tid % 32
    blocks = (max_chunk + warps - 1) // warps
    # Flat pointer views retain arbitrary layer/page strides without copying.
    huge = cute.make_layout((cutlass.Int64(1 << 40),))
    asc = cute.make_tensor(asc_ptr, huge)
    pages = cute.make_tensor(pages_ptr, huge)
    cu = cute.make_tensor(cu_ptr, huge)
    offsets = cute.make_tensor(offsets_ptr, huge)
    global_scale = cute.make_tensor(global_ptr, cute.make_layout((1,)))
    if block == blocks:
        if batch == 0:
            padded_m = (m + 127) // 128 * 128
            for i in range(tid, (padded_m - m) * 16, warps * 32):
                asc[_sf_offset(m + i // 16, i % 16, 16)] = cutlass.Uint8(0)
    else:
        local = cutlass.Int64(block * warps + tid // 32)
        begin, end = cu[batch], cu[batch + 1]
        position = offsets[batch] + local
        logical_page = position // page_size
        valid = (
            (local < end - begin)
            & (local < max_chunk)
            & (position >= 0)
            & (logical_page < table_stride)
        )
        if valid:
            physical = cutlass.Int64(pages[batch * table_stride + logical_page])
            if (physical >= 0) & (physical < num_pages):
                row = position % page_size
                outrow = begin + local
                base = physical * kv_s0 + row * kv_s2
                # Keep large cache offsets in 64-bit pointer arithmetic. Scalar
                # tensor indices must remain within one row/page, below 2 GiB.
                kv_row = cute.make_tensor(kv_ptr + base, cute.make_layout((320,)))
                sf_page = cute.make_tensor(
                    sf_ptr + physical * sf_s0, cute.make_layout((page_size * 40,))
                )
                global_value = global_scale[0]
                scale = sf_page[_sf_offset(row, lane, 40)].to(cutlass.Float32)
                input_bytes = cute.make_rmem_tensor((8,), cutlass.Uint8)
                load_atom = cute.make_copy_atom(
                    cute.nvgpu.CopyUniversalOp(), cutlass.Uint8, num_bits_per_copy=64
                )
                packed_input = cute.make_tensor(
                    kv_ptr + cute.assume(base + lane * 8, 8), cute.make_layout((8,))
                )
                cute.copy(load_atom, packed_input, input_bytes)
                values = cute.make_rmem_tensor((16,), cutlass.Float32)
                amax = cutlass.Float32(1.0e-10)
                for i in cutlass.range_constexpr(16):
                    packed = input_bytes[i // 2].to(cutlass.Int32)
                    code = (packed >> ((i % 2) * 4)) & 15
                    value = (
                        _div_full(_decode_fp4(code) * scale, global_value)
                        .to(cutlass.BFloat16)
                        .to(cutlass.Float32)
                    )
                    values[i] = value
                    magnitude = value
                    if value < 0.0:
                        magnitude = -value
                    amax = cute.arch.fmax(amax, magnitude)
                for delta in cutlass.range_constexpr(3):
                    other = cute.arch.shuffle_sync_bfly(amax, 4 >> delta)
                    amax = cute.arch.fmax(amax, other)
                quant_sf = _round_up_ue8m0(amax * _rcp(cutlass.Float32(448.0)))
                inv_scale = _rcp(_u32_as_f32(quant_sf.to(cutlass.Uint32) << 23))
                packed_out = cute.make_rmem_tensor((16,), cutlass.Float8E4M3FN)
                for i in cutlass.range_constexpr(16):
                    packed_out[i] = _fp8(values[i] * inv_scale)
                output = cute.make_tensor(
                    a_ptr + cute.assume(outrow * 512 + lane * 16, 16), cute.make_layout((16,))
                )
                copy_atom = cute.make_copy_atom(
                    cute.nvgpu.CopyUniversalOp(), cutlass.Float8E4M3FN, num_bits_per_copy=128
                )
                cute.copy(copy_atom, packed_out, output)
                if lane % 8 == 0:
                    dst = _sf_offset(outrow, lane // 8 * 4, 16)
                    for i in cutlass.range_constexpr(4):
                        asc[dst + i] = quant_sf
                main_scale = sf_page[_sf_offset(row, 32 + lane // 8, 40)].to(cutlass.Float32)
                residual_scale = sf_page[_sf_offset(row, 36 + lane // 8, 40)].to(cutlass.Float32)
                main_byte = kv_row[256 + lane].to(cutlass.Int32)
                residual_byte = kv_row[288 + lane].to(cutlass.Int32)
                rope_out = cute.make_rmem_tensor((2,), cutlass.Float8E4M3FN)
                for i in cutlass.range_constexpr(2):
                    main_code = (main_byte >> (i * 4)) & 15
                    residual_code = (residual_byte >> (i * 4)) & 15
                    main = _div_full(_decode_fp4(main_code) * main_scale, global_value)
                    residual = _div_full(_decode_fp4(residual_code) * residual_scale, global_value)
                    value = (main + residual).to(cutlass.BFloat16).to(cutlass.Float32)
                    rope_out[i] = _fp8(value)
                rope = cute.make_tensor(
                    pe_ptr + cute.assume(outrow * 64 + lane * 2, 2), cute.make_layout((2,))
                )
                rope_copy = cute.make_copy_atom(
                    cute.nvgpu.CopyUniversalOp(), cutlass.Float8E4M3FN, num_bits_per_copy=16
                )
                cute.copy(rope_copy, rope_out, rope)


@cute.jit
def _launch(
    a: cute.Pointer,
    asc: cute.Pointer,
    pe: cute.Pointer,
    kv: cute.Pointer,
    sf: cute.Pointer,
    pages: cute.Pointer,
    cu: cute.Pointer,
    offsets: cute.Pointer,
    global_scale: cute.Pointer,
    m: cutlass.Int64,
    max_chunk: cutlass.Int64,
    table_stride: cutlass.Int64,
    num_pages: cutlass.Int64,
    page_size: cutlass.Constexpr,
    num_contexts: cutlass.Int64,
    kv_s0: cutlass.Int64,
    kv_s2: cutlass.Int64,
    kv_s4: cutlass.Constexpr,
    sf_s0: cutlass.Int64,
    warps: cutlass.Constexpr,
    stream: cuda.CUstream,
):
    _gather(
        a,
        asc,
        pe,
        kv,
        sf,
        pages,
        cu,
        offsets,
        global_scale,
        m,
        max_chunk,
        table_stride,
        num_pages,
        page_size,
        kv_s0,
        kv_s2,
        kv_s4,
        sf_s0,
        warps,
    ).launch(
        grid=((max_chunk + warps - 1) // warps + 1, num_contexts, 1),
        block=(warps * 32, 1, 1),
        stream=stream,
    )


_COMPILED = {}


def gather_mxfp8(
    a,
    asc,
    pe,
    kv,
    sf,
    pages,
    cu,
    offsets,
    global_scale,
    max_chunk,
    page_size=128,
    warps=4,
    *,
    page_table_stride=None,
    num_contexts=None,
):
    """Gather canonical page-128 FP4 storage; prefix lengths and page strides are dynamic."""
    from ...attention_backend.fp4_mla.fp4_mla_cutedsl_v_repack import (
        _compile_cutedsl,
        _current_cu_stream,
    )

    if (
        page_size != 128
        or kv.stride(4) != 1
        or kv.data_ptr() % 8
        or kv.stride(0) % 8
        or kv.stride(2) % 8
    ):
        raise ValueError(
            "MXFP8 gather requires canonical page-128 FP4 storage with aligned contiguous packed rows."
        )
    if warps not in (4, 8):
        raise ValueError("gather warps must be 4 or 8")
    if num_contexts is None:
        num_contexts = offsets.numel()
    if page_table_stride is None:
        page_table_stride = pages.numel() // num_contexts
    tensors = (a, asc, pe, kv, sf, pages, cu, offsets, global_scale)
    dtypes = (
        cutlass.Float8E4M3FN,
        cutlass.Uint8,
        cutlass.Float8E4M3FN,
        cutlass.Uint8,
        cutlass.Float8E4M3FN,
        cutlass.Int32,
        cutlass.Int64,
        cutlass.Int64,
        cutlass.Float32,
    )
    alignments = (16, 16, 16, 8, 1, 4, 8, 8, 4)
    pointers = tuple(
        make_ptr(dtype, tensor.data_ptr(), cute.AddressSpace.gmem, assumed_align=alignment)
        for tensor, dtype, alignment in zip(tensors, dtypes, alignments)
    )
    args = (
        *pointers,
        a.shape[0],
        max_chunk,
        page_table_stride,
        kv.shape[0],
        page_size,
        num_contexts,
        kv.stride(0),
        kv.stride(2),
        kv.stride(4),
        sf.stride(0),
        warps,
        _current_cu_stream(),
    )
    key = (a.device.index, warps)
    if key not in _COMPILED:
        _COMPILED[key] = _compile_cutedsl(_launch, *args, options="--opt-level 2")
    _COMPILED[key](*args[:13], *args[14:17], args[18], args[-1])
