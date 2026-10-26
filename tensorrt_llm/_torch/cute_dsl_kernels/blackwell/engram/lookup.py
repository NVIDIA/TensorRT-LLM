# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Engram FP8 embedding gather with fused UE8M0 dequantization."""

import functools

import cutlass
import cutlass.cute as cute
import torch
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op

try:
    from cuda.bindings import driver as cuda
except ImportError:
    from cuda import cuda

Int32, Int64, Uint32, Float32 = cutlass.Int32, cutlass.Int64, cutlass.Uint32, cutlass.Float32


@dsl_user_op
def _e4m3x2_to_f32(source, *, loc=None, ip=None):
    """Exact FP32 values of the two E4M3 bytes in the low 16 bits."""
    out = llvm.inline_asm(
        llvm.StructType.get_literal([T.f32()] * 2),
        [Int32(source).ir_value(loc=loc, ip=ip)],
        "{\n.reg .b16 s, lo, hi;\n.reg .b32 h;\ncvt.u16.u32 s, $2;\n"
        "cvt.rn.f16x2.e4m3x2 h, s;\nmov.b32 {lo, hi}, h;\n"
        "cvt.f32.f16 $0, lo;\ncvt.f32.f16 $1, hi;\n}",
        "=f,=f,r",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    return (
        Float32(llvm.extractvalue(T.f32(), out, [0], loc=loc, ip=ip)),
        Float32(llvm.extractvalue(T.f32(), out, [1], loc=loc, ip=ip)),
    )


@dsl_user_op
def _mul_rn(a, b, *, loc=None, ip=None):
    """IEEE round-to-nearest multiply that keeps subnormals."""
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Float32(a).ir_value(loc=loc, ip=ip), Float32(b).ir_value(loc=loc, ip=ip)],
            "mul.rn.f32 $0, $1, $2;",
            "=f,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def _ue8m0_to_f32(code, *, loc=None, ip=None):
    """``2 ** (code - 127)``; code 0 is the FP32 subnormal ``2 ** -127`` and 255 is NaN."""
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Int32(code).ir_value(loc=loc, ip=ip)],
            "{\n.reg .pred lo, hi;\n.reg .b32 bits;\nshl.b32 bits, $1, 23;\n"
            "setp.eq.u32 lo, $1, 0;\nsetp.eq.u32 hi, $1, 255;\n"
            "selp.b32 bits, 0x00400000, bits, lo;\nselp.b32 $0, 0x7FC00000, bits, hi;\n}",
            "=f,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


class EngramLookupKernel:
    """Gather ``[T * local_heads, dim]`` rows of an E4M3 table and dequantize them.

    Row ``r`` reads hash ``ids[r // local_heads, head_start + r % local_heads]``
    plus its head's bucket offset. Each thread owns one 16-byte column chunk of
    ``rows_per_thread`` rows and issues all of their table loads before
    converting any, so a CTA keeps that many host-memory reads in flight. A
    grid-stride loop covers the rows with at most ``max_ctas`` CTAs. With ``masked``, rows
    outside ``[vocab_start, vocab_end)`` belong to another shard and are zero.
    With ``packed``, output row ``t`` holds token ``t``'s E4M3 bytes of all local
    heads, then their UE8M0 codes.
    """

    vector_bytes = 16
    num_threads = 1024
    rows_per_thread = 4

    def __init__(
        self,
        head_start: int,
        local_heads: int,
        dim: int,
        block_size: int,
        masked: bool,
        packed: bool,
    ):
        if dim % self.vector_bytes != 0 or block_size % self.vector_bytes != 0:
            raise ValueError(
                f"Engram lookup needs dim ({dim}) and block size ({block_size}) "
                f"divisible by {self.vector_bytes}"
            )
        self.chunks = dim // self.vector_bytes
        if self.num_threads % self.chunks != 0:
            raise ValueError(
                f"Engram lookup needs {self.num_threads} threads divisible by dim / 16 ({dim})"
            )
        self.head_start = head_start
        self.local_heads = local_heads
        self.dim = dim
        self.block_size = block_size
        self.masked = masked
        self.packed = packed
        self.rows_per_wave = self.num_threads // self.chunks
        self.rows_per_iteration = self.rows_per_wave * self.rows_per_thread

    @cute.jit
    def __call__(
        self,
        weight: cute.Tensor,
        scale: cute.Tensor,
        ids: cute.Tensor,
        offsets: cute.Tensor,
        output: cute.Tensor,
        vocab_start: Int64,
        vocab_end: Int64,
        max_ctas: Int32,
        stream: cuda.CUstream,
    ):
        num_rows = output.shape[0]
        if cutlass.const_expr(self.packed):
            num_rows = num_rows * self.local_heads
        grid = (num_rows + self.rows_per_iteration - 1) // self.rows_per_iteration
        self.kernel(weight, scale, ids, offsets, output, vocab_start, vocab_end).launch(
            grid=[min(grid, max_ctas), 1, 1], block=[self.num_threads, 1, 1], stream=stream
        )

    @cute.kernel
    def kernel(
        self,
        weight: cute.Tensor,
        scale: cute.Tensor,
        ids: cute.Tensor,
        offsets: cute.Tensor,
        output: cute.Tensor,
        vocab_start: Int64,
        vocab_end: Int64,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        cta, _, _ = cute.arch.block_idx()
        grid, _, _ = cute.arch.grid_dim()
        chunk = Int64(tidx % self.chunks)
        wave_row = Int64(tidx // self.chunks)
        column = chunk * self.vector_bytes
        num_rows = Int32(output.shape[0])
        if cutlass.const_expr(self.packed):
            num_rows = Int32(output.shape[0] * self.local_heads)
        weight_base = weight.iterator.toint()
        element_bytes = output.element_type.width // 8
        step = grid * self.rows_per_iteration

        for start in cutlass.range(cta * self.rows_per_iteration, num_rows, step):
            base = Int64(start)
            local = cute.make_rmem_tensor((self.rows_per_thread,), Int64)
            for j in cutlass.range_constexpr(self.rows_per_thread):
                row = base + j * self.rows_per_wave + wave_row
                local[j] = Int64(-1)
                if row < num_rows:
                    local[j] = self.table_row(ids, offsets, row, vocab_start, vocab_end)

            words = [cute.make_rmem_tensor((4,), Uint32) for _ in range(self.rows_per_thread)]
            codes = cute.make_rmem_tensor((self.rows_per_thread,), Int32)
            for j in cutlass.range_constexpr(self.rows_per_thread):
                codes[j] = Int32(127)
                for w in cutlass.range_constexpr(4):
                    words[j][w] = Uint32(0)
                if local[j] >= 0:
                    self.load_chunk(weight_base, scale, local[j], column, words[j], codes, j)

            for j in cutlass.range_constexpr(self.rows_per_thread):
                row = base + j * self.rows_per_wave + wave_row
                if row < num_rows:
                    if cutlass.const_expr(self.packed):
                        self.store_packed(output, row, column, words[j], codes[j])
                    else:
                        self.store_chunk(
                            output, (row * self.dim + column) * element_bytes, words[j], codes[j]
                        )

    @cute.jit
    def table_row(
        self,
        ids: cute.Tensor,
        offsets: cute.Tensor,
        row: Int64,
        vocab_start: Int64,
        vocab_end: Int64,
    ) -> Int64:
        """Shard-local table row of output row ``row``, or -1 when another shard owns it."""
        head = row % self.local_heads
        index = Int64(ids[row // self.local_heads, head + self.head_start]) + Int64(offsets[head])
        if cutlass.const_expr(self.masked):
            local = Int64(-1)
            if (index >= vocab_start) & (index < vocab_end):
                local = index - vocab_start
            index = local
        return index

    @cute.jit
    def load_chunk(
        self,
        weight_base: Int64,
        scale: cute.Tensor,
        local: Int64,
        column: Int64,
        words: cute.Tensor,
        codes: cute.Tensor,
        j: cutlass.Constexpr,
    ):
        """Load one 16-byte E4M3 chunk and its scale code."""
        source = cute.make_ptr(
            Uint32,
            weight_base + local * self.dim + column,
            cute.AddressSpace.gmem,
            assumed_align=16,
        )
        cute.autovec_copy(cute.make_tensor(source, cute.make_layout(4)), words)
        codes[j] = Int32(scale[local, column // self.block_size]) & 255

    @cute.jit
    def store_chunk(self, output: cute.Tensor, byte_offset: Int64, words: cute.Tensor, code: Int32):
        """Dequantize one chunk exactly in FP32, then round it to the output dtype."""
        factor = _ue8m0_to_f32(code)
        values = cute.make_rmem_tensor((self.vector_bytes,), output.element_type)
        for pair in cutlass.range_constexpr(self.vector_bytes // 2):
            low, high = _e4m3x2_to_f32(Int32((words[pair // 2] >> (16 * (pair % 2))) & 0xFFFF))
            values[2 * pair] = _mul_rn(low, factor).to(output.element_type)
            values[2 * pair + 1] = _mul_rn(high, factor).to(output.element_type)
        target = cute.make_ptr(
            output.element_type,
            output.iterator.toint() + byte_offset,
            cute.AddressSpace.gmem,
            assumed_align=16,
        )
        cute.autovec_copy(values, cute.make_tensor(target, cute.make_layout(self.vector_bytes)))

    @cute.jit
    def store_packed(
        self, output: cute.Tensor, row: Int64, column: Int64, words: cute.Tensor, code: Int32
    ):
        token = row // self.local_heads
        head = row % self.local_heads
        values_width = self.local_heads * self.dim
        target = cute.make_ptr(
            Uint32,
            output.iterator.toint() + token * output.shape[1] + head * self.dim + column,
            cute.AddressSpace.gmem,
            assumed_align=16,
        )
        cute.autovec_copy(words, cute.make_tensor(target, cute.make_layout(4)))
        if column % self.block_size == 0:
            output[token, values_width + (head * self.dim + column) // self.block_size] = (
                cutlass.Uint8(code)
            )


_OUTPUT_DTYPES = {
    torch.bfloat16: cutlass.BFloat16,
    torch.float16: cutlass.Float16,
    torch.float32: cutlass.Float32,
    torch.uint8: cutlass.Uint8,
}


@functools.cache
def compile_engram_lookup(
    head_start: int,
    local_heads: int,
    num_heads: int,
    dim: int,
    block_size: int,
    masked: bool,
    output_dtype: torch.dtype,
    packed: bool = False,
):
    """Compile the lookup for one table geometry; tensors use TVM-FFI calling."""
    if output_dtype not in _OUTPUT_DTYPES:
        raise ValueError(f"Engram lookup does not support output dtype {output_dtype}")
    width = local_heads * dim + local_heads * dim // block_size if packed else dim
    weight = cute.runtime.make_fake_compact_tensor(
        cutlass.Uint8, (cute.sym_int(), dim), stride_order=(1, 0), assumed_align=16
    )
    scale = cute.runtime.make_fake_compact_tensor(
        cutlass.Uint8, (cute.sym_int(), dim // block_size), stride_order=(1, 0)
    )
    ids = cute.runtime.make_fake_tensor(
        Int64, (cute.sym_int(), num_heads), stride=(cute.sym_int(), cute.sym_int())
    )
    offsets = cute.runtime.make_fake_compact_tensor(Int64, (local_heads,), stride_order=(0,))
    output = cute.runtime.make_fake_compact_tensor(
        _OUTPUT_DTYPES[output_dtype], (cute.sym_int(), width), stride_order=(1, 0), assumed_align=16
    )
    kernel = EngramLookupKernel(head_start, local_heads, dim, block_size, masked, packed)
    return cute.compile(
        kernel,
        weight,
        scale,
        ids,
        offsets,
        output,
        Int64(0),
        Int64(0),
        Int32(1),
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--opt-level 2 --enable-tvm-ffi",
    )
