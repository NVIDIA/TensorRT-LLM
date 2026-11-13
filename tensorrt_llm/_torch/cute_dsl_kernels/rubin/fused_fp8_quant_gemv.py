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

from ..blackwell.cute_ptx_utils import mma_sync
from ..blackwell.utils import (
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


class FusedFp8QuantGemv:
    """Each CTA quantizes one K slice and computes a 16x64 output tile.

    Quantization is repeated across output tiles. Activation codes and K128
    scales stay in shared memory. Weight scales use R128c4 with each K128
    exponent replicated over four K32 slots, as in prepare_weight_scale.
    Split-K writes disjoint FP32 partials, followed by one BF16 reduction.
    """

    def __init__(self, m: int, n: int, k: int, split_k: int, warps: int = 8):
        self.m, self.n, self.k = m, n, k
        self.split_k = split_k
        self.warps = warps
        self.tile_n = warps * 8

    @cute.jit
    def quantize_tile(self, x, sa, ss, kb, tid):
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
                packed32 = cute.recast_tensor(packed, cutlass.Int32)
                for i in cutlass.range_constexpr(4):
                    sa[row, chunk * 4 + i] = packed32[i]
                if chunk == 0:
                    ss[row] = dequant
            else:
                for i in cutlass.range_constexpr(4):
                    sa[row, chunk * 4 + i] = cutlass.Int32(0)
                if chunk == 0:
                    ss[row] = cutlass.Float32(1.0)

    @cute.kernel
    def kernel(self, x: cute.Tensor, w: cute.Tensor, ws: cute.Tensor, y: cute.Tensor):
        if cutlass.const_expr(TRTLLM_ENABLE_PDL):
            griddepcontrol_wait()
        tid, _, _ = cute.arch.thread_idx()
        bn, sk, _ = cute.arch.block_idx()
        lane = tid % 32
        warp = tid // 32
        group = lane // 4
        quad = lane % 4
        allocator = utils.SmemAllocator()
        blocks = self.k // 128
        slots = blocks // self.split_k
        sa = allocator.allocate_tensor(
            cutlass.Int32, cute.make_layout((16, 32, slots), stride=(36, 1, 576))
        )
        ss = allocator.allocate_tensor(
            cutlass.Float32, cute.make_layout((16, slots), stride=(1, 16))
        )
        total = cute.make_rmem_tensor((4,), cutlass.Float32)
        total.fill(0.0)
        # Assign consecutive warps to different K128 tiles so low-M rows
        # quantize concurrently before the single consumer barrier.
        for task in range(warp, 4 * slots, self.warps):
            slot = task % slots
            row_group = task // slots
            self.quantize_tile(
                x, sa[None, None, slot], ss[None, slot], sk * slots + slot, row_group * 32 + lane
            )
        cute.arch.sync_threads()
        for slot in range(slots):
            kb = sk * slots + slot
            accum = cute.make_rmem_tensor((4,), cutlass.Float32)
            accum.fill(0.0)
            ar = cute.make_rmem_tensor((4,), cutlass.Int32)
            br = cute.make_rmem_tensor((2,), cutlass.Int32)
            nr = bn * self.tile_n + warp * 8 + group
            for step in cutlass.range_constexpr(4):
                ar[0] = sa[group, step * 8 + quad, slot]
                ar[1] = sa[group + 8, step * 8 + quad, slot]
                ar[2] = sa[group, step * 8 + quad + 4, slot]
                ar[3] = sa[group + 8, step * 8 + quad + 4, slot]
                br[0] = w[nr, kb * 32 + step * 8 + quad]
                br[1] = w[nr, kb * 32 + step * 8 + quad + 4]
                accum.store(
                    mma_sync(
                        cute.recast_tensor(ar, cutlass.Float8E4M3FN),
                        cute.recast_tensor(br, cutlass.Float8E4M3FN),
                        accum,
                    )
                )
            for i in cutlass.range_constexpr(4):
                mr = group + (i // 2) * 8
                nc = bn * self.tile_n + warp * 8 + quad * 2 + i % 2
                sf_offset = (
                    (nc // 128) * blocks * 512 + kb * 512 + (nc % 32) * 16 + (nc % 128 // 32) * 4
                )
                combined_exponent = (
                    _scale_code(ss[mr, slot]) + ws[sf_offset].to(cutlass.Int32) - 254
                )
                half_exponent = combined_exponent // 2
                # Balanced powers avoid overflow in an intermediate scale product.
                scale_lo = _float_bits((half_exponent + 127) << 23)
                scale_hi = _float_bits((combined_exponent - half_exponent + 127) << 23)
                total[i] += (accum[i] * scale_lo) * scale_hi
        for i in cutlass.range_constexpr(4):
            mr = group + (i // 2) * 8
            nc = bn * self.tile_n + warp * 8 + quad * 2 + i % 2
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
        sa = allocator.allocate_tensor(cutlass.Int32, cute.make_layout((16, 32), stride=(36, 1)))
        ss = allocator.allocate_tensor(cutlass.Float32, cute.make_layout((16,)))
        self.quantize_tile(x, sa, ss, kb, tid)
        cute.arch.sync_threads()
        row, word = tid // 32, tid % 32
        for group in cutlass.range_constexpr(4):
            mr = row + group * 4
            if mr < self.m:
                codes[mr, kb * 32 + word] = sa[mr, word]
                if word == 0:
                    scales[mr, kb] = _scale_code(ss[mr]).to(cutlass.Uint8)

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
