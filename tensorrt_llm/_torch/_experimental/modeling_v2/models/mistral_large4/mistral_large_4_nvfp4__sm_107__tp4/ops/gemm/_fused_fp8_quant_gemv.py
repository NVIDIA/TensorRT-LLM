# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Low-M K128 quantization and FP8 GEMM without global activation scratch."""

from functools import lru_cache

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import torch
from cutlass._mlir.dialects import llvm
from cutlass.cute.runtime import from_dlpack
from cutlass.cutlass_dsl import T, dsl_user_op

from tensorrt_llm._torch.cute_dsl_kernels.blackwell.cute_ptx_utils import mma_sync
from tensorrt_llm._torch.cute_dsl_kernels.blackwell.utils import (
    TRTLLM_ENABLE_PDL,
    griddepcontrol_launch_dependents,
    griddepcontrol_wait,
)


@dsl_user_op
def _exponent(amax: cutlass.Float32, *, loc=None, ip=None):
    return cutlass.Int32(
        llvm.inline_asm(
            T.i32(),
            [amax.ir_value(loc=loc, ip=ip)],
            "{ .reg .f32 rcp, raw; .reg .b32 bits, mant, inc; .reg .pred p; "
            "rcp.approx.ftz.f32 rcp, 0f43e00000; mul.f32 raw, $1, rcp; "
            "mov.b32 bits, raw; and.b32 mant, bits, 0x7fffff; "
            "setp.ne.u32 p, mant, 0; selp.u32 inc, 1, 0, p; "
            "shr.u32 bits, bits, 23; add.u32 $0, bits, inc; }",
            "=r,f",
            has_side_effects=False,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _float_bits(bits: cutlass.Int32, *, loc=None, ip=None):
    return cutlass.Float32(
        llvm.inline_asm(
            T.f32(),
            [bits.ir_value(loc=loc, ip=ip)],
            "mov.b32 $0, $1;",
            "=f,r",
            has_side_effects=False,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _scale_code(value: cutlass.Float32, *, loc=None, ip=None):
    return cutlass.Int32(
        llvm.inline_asm(
            T.i32(),
            [value.ir_value(loc=loc, ip=ip)],
            "{ .reg .b32 bits; mov.b32 bits, $1; shr.u32 $0, bits, 23; }",
            "=r,f",
            has_side_effects=False,
            loc=loc,
            ip=ip,
        )
    )


# Weight loads are volatile so they stay ahead of griddepcontrol.wait.
@dsl_user_op
def _load_words(address: cutlass.Int64, *, loc=None, ip=None):
    words = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32()] * 4),
        [address.ir_value(loc=loc, ip=ip)],
        "ld.global.nc.L1::no_allocate.v4.b32 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,l",
        has_side_effects=True,
        is_align_stack=False,
        loc=loc,
        ip=ip,
    )
    return [cutlass.Int32(llvm.extractvalue(T.i32(), words, [i], loc=loc, ip=ip)) for i in range(4)]


@dsl_user_op
def _load_byte(address: cutlass.Int64, *, loc=None, ip=None):
    return cutlass.Int32(
        llvm.inline_asm(
            T.i32(),
            [address.ir_value(loc=loc, ip=ip)],
            "ld.global.nc.u8 $0, [$1];",
            "=r,l",
            has_side_effects=True,
            loc=loc,
            ip=ip,
        )
    )


class FusedFp8QuantGemv:
    """Each CTA quantizes one K slice and computes a 16x64 output tile.

    Quantization is repeated across output tiles. Activation codes and K128
    scales stay in shared memory. Weight scales use R128c4 with each K128
    exponent replicated over four K32 slots, as in prepare_weight_scale.
    Split-K writes disjoint FP32 partials, followed by one BF16 reduction.

    Each lane streams 2x16 bytes of one weight row per K128 slot: words
    4q..4q+3 and 16+4q..16+4q+3 for quad q. MMA step s pairs word 4q+s with
    16+4q+s, a K permutation applied identically to the activation fragment,
    which therefore reads the same two 16-byte chunks from shared memory. Up to
    ``prefetch`` slots of weights are issued before the PDL wait.
    """

    def __init__(self, m: int, n: int, k: int, split_k: int, warps: int = 8, prefetch: int = 8):
        self.m, self.n, self.k = m, n, k
        self.split_k = split_k
        self.warps = warps
        self.tile_n = warps * 8
        self.slots = k // 128 // split_k
        self.prefetch = min(prefetch, self.slots)
        self.buffers = max(self.prefetch, 1)
        # MMA rows 8..15 are constant zero operands when M <= 8.
        self.rows = 8 if m <= 8 else 16

    def shared_chunk(self, sa, slot, row, chunk):
        # Odd rows rotate by four 16-byte chunks so the two rows read by each
        # quarter warp cover all banks.
        offset = (slot * self.rows + row) * 32 + (chunk + row % 2 * 4) % 8 * 4
        return cute.make_tensor(
            cute.make_ptr(
                cutlass.Int32,
                (sa.iterator + offset).toint(),
                cute.AddressSpace.smem,
                assumed_align=16,
            ),
            cute.make_layout((4,)),
        )

    def load_weights(self, w, ws, weights, scales, buffer, column, group, quad, kb):
        row = cute.domain_offset((column + group, kb * 32 + quad * 4), w).iterator.toint()
        low = _load_words(row)
        high = _load_words(row + 64)
        for i in range(4):
            weights[buffer, i] = low[i]
            weights[buffer, i + 4] = high[i]
        for i in range(2):
            nc = column + quad * 2 + i
            sf_offset = (
                (nc // 128) * (self.k // 128) * 512
                + kb * 512
                + (nc % 32) * 16
                + (nc % 128 // 32) * 4
            )
            scales[buffer, i] = _load_byte(cute.domain_offset((sf_offset,), ws).iterator.toint())

    @cute.jit
    def quantize_tile(self, x, sa, ss, slot, kb, tid):
        # Eight lanes own each complete K128 row; padded MMA rows are zero.
        if tid < 128:
            row = tid // 8
            chunk = tid % 8
            if row < ((self.m + 3) // 4) * 4:
                values = cute.make_rmem_tensor((16,), cutlass.Float32)
                values.fill(0.0)
                if row < self.m:
                    source = cute.make_tensor(
                        cute.make_ptr(
                            cutlass.BFloat16,
                            cute.domain_offset((row, kb * 128 + chunk * 16), x).iterator.toint(),
                            cute.AddressSpace.gmem,
                            assumed_align=16,
                        ),
                        cute.make_layout((16,)),
                    )
                    loaded = cute.make_rmem_tensor((16,), cutlass.BFloat16)
                    atom = cute.make_copy_atom(
                        cute.nvgpu.CopyUniversalOp(), cutlass.BFloat16, num_bits_per_copy=128
                    )
                    cute.copy(atom, source, loaded)
                    values.store(loaded.load().to(cutlass.Float32))
                amax = cutlass.Float32(1.0e-10)
                for i in cutlass.range_constexpr(16):
                    amax = cute.arch.fmax(amax, cute.arch.fmax(values[i], -values[i]))
                for shift in cutlass.range_constexpr(3):
                    amax = cute.arch.fmax(
                        amax, cute.arch.shuffle_sync_bfly(amax, offset=1 << shift)
                    )
                exponent = _exponent(amax)
                dequant = _float_bits(exponent << 23)
                quant = _float_bits((254 - exponent) << 23)
                packed = cute.make_rmem_tensor((16,), cutlass.Float8E4M3FN)
                for i in cutlass.range_constexpr(16):
                    packed[i] = (values[i] * quant).to(cutlass.Float8E4M3FN)
                cute.autovec_copy(
                    cute.recast_tensor(packed, cutlass.Int32),
                    self.shared_chunk(sa, slot, row, chunk),
                )
                if chunk == 0:
                    ss[row, slot] = dequant
            else:
                zeros = cute.make_rmem_tensor((4,), cutlass.Int32)
                zeros.fill(0)
                cute.autovec_copy(zeros, self.shared_chunk(sa, slot, row, chunk))
                if chunk == 0:
                    ss[row, slot] = cutlass.Float32(1.0)

    @cute.kernel
    def kernel(self, x: cute.Tensor, w: cute.Tensor, ws: cute.Tensor, y: cute.Tensor):
        tid, _, _ = cute.arch.thread_idx()
        bn, sk, _ = cute.arch.block_idx()
        lane = tid % 32
        warp = tid // 32
        group = lane // 4
        quad = lane % 4
        slots = self.slots
        prefetch = self.prefetch
        buffers = self.buffers
        column = bn * self.tile_n + warp * 8
        weights = cute.make_rmem_tensor((buffers, 8), cutlass.Int32)
        weight_scales = cute.make_rmem_tensor((buffers, 2), cutlass.Int32)
        # Weights are not written by the preceding kernel, so their HBM latency
        # overlaps its tail and the activation quantization below.
        for slot in cutlass.range_constexpr(prefetch):
            self.load_weights(
                w, ws, weights, weight_scales, slot, column, group, quad, sk * slots + slot
            )
        if cutlass.const_expr(TRTLLM_ENABLE_PDL):
            griddepcontrol_wait()
        allocator = utils.SmemAllocator()
        sa = allocator.allocate_tensor(
            cutlass.Int32, cute.make_layout((slots * self.rows * 32,)), byte_alignment=16
        )
        ss = allocator.allocate_tensor(
            cutlass.Float32, cute.make_layout((self.rows, slots), stride=(1, self.rows))
        )
        total = cute.make_rmem_tensor((4,), cutlass.Float32)
        total.fill(0.0)
        # Assign consecutive warps to different K128 tiles so low-M rows
        # quantize concurrently before the single consumer barrier.
        for task in range(warp, self.rows // 4 * slots, self.warps):
            slot = task % slots
            row_group = task // slots
            self.quantize_tile(x, sa, ss, slot, sk * slots + slot, row_group * 32 + lane)
        cute.arch.sync_threads()
        ar = cute.make_rmem_tensor((4,), cutlass.Int32)
        ar.fill(0)
        br = cute.make_rmem_tensor((2,), cutlass.Int32)
        for slot in cutlass.range_constexpr(slots):
            kb = sk * slots + slot
            buffer = slot % buffers
            if cutlass.const_expr(prefetch == 0):
                self.load_weights(w, ws, weights, weight_scales, 0, column, group, quad, kb)
            low = cute.make_rmem_tensor((4,), cutlass.Int32)
            high = cute.make_rmem_tensor((4,), cutlass.Int32)
            cute.autovec_copy(self.shared_chunk(sa, slot, group, quad), low)
            cute.autovec_copy(self.shared_chunk(sa, slot, group, quad + 4), high)
            if cutlass.const_expr(self.rows == 16):
                low8 = cute.make_rmem_tensor((4,), cutlass.Int32)
                high8 = cute.make_rmem_tensor((4,), cutlass.Int32)
                cute.autovec_copy(self.shared_chunk(sa, slot, group + 8, quad), low8)
                cute.autovec_copy(self.shared_chunk(sa, slot, group + 8, quad + 4), high8)
            accum = cute.make_rmem_tensor((4,), cutlass.Float32)
            accum.fill(0.0)
            for step in cutlass.range_constexpr(4):
                ar[0] = low[step]
                ar[2] = high[step]
                if cutlass.const_expr(self.rows == 16):
                    ar[1] = low8[step]
                    ar[3] = high8[step]
                br[0] = weights[buffer, step]
                br[1] = weights[buffer, step + 4]
                accum.store(
                    mma_sync(
                        cute.recast_tensor(ar, cutlass.Float8E4M3FN),
                        cute.recast_tensor(br, cutlass.Float8E4M3FN),
                        accum,
                    )
                )
            for i in cutlass.range_constexpr(self.rows // 4):
                mr = group + (i // 2) * 8
                combined_exponent = _scale_code(ss[mr, slot]) + weight_scales[buffer, i % 2] - 254
                half_exponent = combined_exponent // 2
                # Balanced powers avoid overflow in an intermediate scale product.
                scale_lo = _float_bits((half_exponent + 127) << 23)
                scale_hi = _float_bits((combined_exponent - half_exponent + 127) << 23)
                total[i] += (accum[i] * scale_lo) * scale_hi
            if cutlass.const_expr(prefetch > 0 and slot + prefetch < slots):
                self.load_weights(
                    w, ws, weights, weight_scales, buffer, column, group, quad, kb + prefetch
                )
        for i in cutlass.range_constexpr(self.rows // 4):
            mr = group + (i // 2) * 8
            nc = column + quad * 2 + i % 2
            if mr < self.m:
                if cutlass.const_expr(self.split_k == 1):
                    y[mr, nc] = total[i].to(cutlass.BFloat16)
                else:
                    y[sk, mr, nc] = total[i]

        if cutlass.const_expr(TRTLLM_ENABLE_PDL):
            griddepcontrol_launch_dependents()

    @cute.kernel
    def debug_kernel(self, x: cute.Tensor, codes: cute.Tensor, scales: cute.Tensor):
        if cutlass.const_expr(TRTLLM_ENABLE_PDL):
            griddepcontrol_wait()
        tid, _, _ = cute.arch.thread_idx()
        kb, _, _ = cute.arch.block_idx()
        allocator = utils.SmemAllocator()
        sa = allocator.allocate_tensor(
            cutlass.Int32, cute.make_layout((16 * 32,)), byte_alignment=16
        )
        ss = allocator.allocate_tensor(cutlass.Float32, cute.make_layout((16, 1), stride=(1, 16)))
        self.quantize_tile(x, sa, ss, 0, kb, tid)
        cute.arch.sync_threads()
        row, word = tid // 32, tid % 32
        for group in cutlass.range_constexpr(4):
            mr = row + group * 4
            if mr < self.m:
                codes[mr, kb * 32 + word] = self.shared_chunk(sa, 0, mr, word // 4)[word % 4]
                if word == 0:
                    scales[mr, kb] = _scale_code(ss[mr, 0]).to(cutlass.Uint8)

        if cutlass.const_expr(TRTLLM_ENABLE_PDL):
            griddepcontrol_launch_dependents()

    @cute.jit
    def debug(self, x: cute.Tensor, codes: cute.Tensor, scales: cute.Tensor, stream: cuda.CUstream):
        self.debug_kernel(x, codes, scales).launch(
            grid=(self.k // 128, 1, 1), block=(128, 1, 1), stream=stream, use_pdl=TRTLLM_ENABLE_PDL
        )

    @cute.kernel
    def reduce(self, partial: cute.Tensor, y: cute.Tensor):
        if cutlass.const_expr(TRTLLM_ENABLE_PDL):
            griddepcontrol_wait()
        tid, _, _ = cute.arch.thread_idx()
        bid, _, _ = cute.arch.block_idx()
        idx = bid * 256 + tid
        if idx < self.m * self.n:
            row, col = idx // self.n, idx % self.n
            value = cutlass.Float32(0.0)
            for sk in cutlass.range_constexpr(self.split_k):
                value += partial[sk, row, col]
            y[row, col] = value.to(cutlass.BFloat16)

        if cutlass.const_expr(TRTLLM_ENABLE_PDL):
            griddepcontrol_launch_dependents()

    @cute.jit
    def single(
        self, x: cute.Tensor, w: cute.Tensor, ws: cute.Tensor, y: cute.Tensor, stream: cuda.CUstream
    ):
        self.kernel(x, w, ws, y).launch(
            grid=(self.n // self.tile_n, 1, 1),
            block=(32 * self.warps, 1, 1),
            stream=stream,
            use_pdl=TRTLLM_ENABLE_PDL,
        )

    @cute.jit
    def __call__(
        self,
        x: cute.Tensor,
        w: cute.Tensor,
        ws: cute.Tensor,
        partial: cute.Tensor,
        y: cute.Tensor,
        stream: cuda.CUstream,
    ):
        self.kernel(x, w, ws, partial).launch(
            grid=(self.n // self.tile_n, self.split_k, 1),
            block=(32 * self.warps, 1, 1),
            stream=stream,
            use_pdl=TRTLLM_ENABLE_PDL,
        )
        self.reduce(partial, y).launch(
            grid=((self.m * self.n + 255) // 256, 1, 1),
            block=(256, 1, 1),
            stream=stream,
            use_pdl=TRTLLM_ENABLE_PDL,
        )


@lru_cache(maxsize=128)
def _compile(m: int, n: int, k: int, row_stride: int, split_k: int, warps: int, device: int):
    # Setup allocations only describe tensor layouts; the executable owns no data.
    device = torch.device("cuda", device)
    x = torch.empty_strided((m, k), (row_stride, 1), dtype=torch.bfloat16, device=device)
    w = torch.empty((n, k // 4), dtype=torch.int32, device=device)
    ws = torch.empty(((n + 127) // 128 * 128 * k // 32,), dtype=torch.uint8, device=device)
    y = torch.empty((m, n), dtype=torch.bfloat16, device=device)
    kernel = FusedFp8QuantGemv(m, n, k, split_k, warps)
    stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    if split_k == 1:
        return cute.compile(
            kernel.single,
            *(from_dlpack(t, assumed_align=16) for t in (x, w, ws, y)),
            stream,
            options="--enable-tvm-ffi",
        )
    partial = torch.empty((split_k, m, n), dtype=torch.float32, device=device)
    return cute.compile(
        FusedFp8QuantGemv(m, n, k, split_k, warps),
        *(from_dlpack(t, assumed_align=16) for t in (x, w, ws, partial, y)),
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )


def fused_fp8_quant_gemv(
    input: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    *,
    split_k: int | None = None,
    warps: int = 8,
) -> torch.Tensor:
    """SM107 BF16 to FP8 linear with replicated K128 weight scales.

    Weight scale storage is R128c4. Its four K32 scale bytes for every K128
    block must be identical; independent K32 weight scales are unsupported.
    Weights and weight scales are read before the PDL wait, so they must not be
    written by the immediately preceding kernel on the stream.
    """
    m, k = input.shape
    n = weight.shape[0]
    if warps not in (4, 8):
        raise ValueError("Fused FP8 GEMV supports four or eight warps per CTA")
    if not (1 <= m <= 16 and k % 128 == 0 and n % (warps * 8) == 0):
        raise ValueError("Fused FP8 GEMV requires M <= 16, K divisible by 128, and full N tiles")
    if input.stride(-1) != 1 or not weight.is_contiguous():
        raise ValueError("Fused FP8 GEMV requires contiguous K dimensions")
    if split_k is None:
        split_k = 8 if k >= 3072 else 1
    if (k // 128) % split_k:
        raise ValueError("Split-K must divide the number of K128 blocks")
    out = torch.empty((m, n), dtype=torch.bfloat16, device=input.device)
    fn = _compile(m, n, k, input.stride(0), split_k, warps, input.device.index)
    if split_k == 1:
        fn(input, weight.view(torch.int32), weight_scale.reshape(-1), out)
    else:
        partial = torch.empty((split_k, m, n), dtype=torch.float32, device=input.device)
        fn(input, weight.view(torch.int32), weight_scale.reshape(-1), partial, out)
    return out


def debug_quantization(input: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Export the exact shared-memory quantization for byte-level T0 checks."""
    m, k = input.shape
    codes = torch.empty((m, k), dtype=torch.float8_e4m3fn, device=input.device)
    scales = torch.empty((m, k // 128), dtype=torch.uint8, device=input.device)
    kernel = FusedFp8QuantGemv(m, 64, k, 1)
    fn = cute.compile(
        kernel.debug,
        from_dlpack(input, assumed_align=16),
        from_dlpack(codes.view(torch.int32), assumed_align=16),
        from_dlpack(scales, assumed_align=1),
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )
    fn(input, codes.view(torch.int32), scales)
    return codes, scales
