# SPDX-FileCopyrightText: Copyright (c) 2025 by FlashInfer team.
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CuTe DSL W4A16 dense decode kernel specialized for ``M == 1``.

Production uses one topology: one warp owns one output row of ``W``.  The
AutoTuner chooses only the number of those warps (and therefore output rows)
per CTA.  The former two-/four-output-per-warp and split-K variants were
profiling ablations; they are deliberately not production code.
"""

from __future__ import annotations

import math
import operator
import os
from collections.abc import Callable
from typing import Optional

import torch

from ...autotuner import AutoTuner, OptimizationProfile, TunableRunner, TuningConfig

try:
    import cuda.bindings.driver as cuda
    import cutlass
    import cutlass.cute as cute
    from cutlass import BFloat16, Float32, Int32
    from cutlass._mlir import ir
    from cutlass._mlir.dialects import llvm
    from cutlass.cutlass_dsl import Int64, T, Uint32, dsl_user_op

    _CUTE_AVAILABLE = True
except ImportError:
    _CUTE_AVAILABLE = False

_SF_VEC_SIZE = 16

if _CUTE_AVAILABLE:

    @dsl_user_op
    def ld_global_nc_32b_u32x8(
        base_ptr: Int64, *, loc=None, ip=None
    ) -> tuple[
        Uint32,
        Uint32,
        Uint32,
        Uint32,
        Uint32,
        Uint32,
        Uint32,
        Uint32,
    ]:
        """Load one naturally aligned 32-byte activation fragment on SM100+."""
        result = llvm.inline_asm(
            llvm.StructType.get_literal([T.i32()] * 8),
            [Int64(base_ptr).ir_value(loc=loc, ip=ip)],
            """
            {
                .reg .b64 r0, r1, r2, r3;
                ld.global.nc.v4.b64 {r0, r1, r2, r3}, [$8];
                mov.b64 {$0, $1}, r0;
                mov.b64 {$2, $3}, r1;
                mov.b64 {$4, $5}, r2;
                mov.b64 {$6, $7}, r3;
            }
            """,
            "=r,=r,=r,=r,=r,=r,=r,=r,l",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
        return (
            Uint32(llvm.extractvalue(T.i32(), result, [0], loc=loc, ip=ip)),
            Uint32(llvm.extractvalue(T.i32(), result, [1], loc=loc, ip=ip)),
            Uint32(llvm.extractvalue(T.i32(), result, [2], loc=loc, ip=ip)),
            Uint32(llvm.extractvalue(T.i32(), result, [3], loc=loc, ip=ip)),
            Uint32(llvm.extractvalue(T.i32(), result, [4], loc=loc, ip=ip)),
            Uint32(llvm.extractvalue(T.i32(), result, [5], loc=loc, ip=ip)),
            Uint32(llvm.extractvalue(T.i32(), result, [6], loc=loc, ip=ip)),
            Uint32(llvm.extractvalue(T.i32(), result, [7], loc=loc, ip=ip)),
        )

    @dsl_user_op
    def ld_global_nc_v2_u32(base_ptr: Int64, *, loc=None, ip=None) -> tuple[Uint32, Uint32]:
        """Load 64 bits through the non-coherent global-memory cache."""
        result = llvm.inline_asm(
            llvm.StructType.get_literal([T.i32(), T.i32()]),
            [Int64(base_ptr).ir_value(loc=loc, ip=ip)],
            "ld.global.nc.v2.u32 {$0, $1}, [$2];",
            "=r,=r,l",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
        return (
            Uint32(llvm.extractvalue(T.i32(), result, [0], loc=loc, ip=ip)),
            Uint32(llvm.extractvalue(T.i32(), result, [1], loc=loc, ip=ip)),
        )

    @dsl_user_op
    def f16x2_to_f32x2(packed_h2: Uint32, *, loc=None, ip=None) -> tuple[Float32, Float32]:
        """Unpack two packed float16 values into two float32 values."""
        result = llvm.inline_asm(
            ir.Type.parse("!llvm.struct<(f32, f32)>"),
            [Uint32(packed_h2).ir_value(loc=loc, ip=ip)],
            """
            {
                .reg .b16 lo, hi;
                mov.b32 {lo, hi}, $2;
                cvt.f32.f16 $0, lo;
                cvt.f32.f16 $1, hi;
            }
            """,
            "=f,=f,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
        return (
            Float32(llvm.extractvalue(T.f32(), result, [0], loc=loc, ip=ip)),
            Float32(llvm.extractvalue(T.f32(), result, [1], loc=loc, ip=ip)),
        )

    @dsl_user_op
    def cvt_e4m3_to_f32_via_f16(fp8_val: Uint32, *, loc=None, ip=None) -> Float32:
        """Convert one E4M3 value to float32 through native float16 conversion."""
        return Float32(
            llvm.inline_asm(
                T.f32(),
                [Uint32(fp8_val).ir_value(loc=loc, ip=ip)],
                """
                {
                    .reg .b16 fp8_pair;
                    .reg .b32 h2;
                    .reg .b16 lo, hi;
                    cvt.u16.u32 fp8_pair, $1;
                    cvt.rn.f16x2.e4m3x2 h2, fp8_pair;
                    mov.b32 {lo, hi}, h2;
                    cvt.f32.f16 $0, lo;
                }
                """,
                "=f,r",
                has_side_effects=False,
                is_align_stack=False,
                asm_dialect=llvm.AsmDialect.AD_ATT,
                loc=loc,
                ip=ip,
            )
        )

    @dsl_user_op
    def fp4_decode_4bytes(
        packed_u32: Uint32, *, loc=None, ip=None
    ) -> tuple[Uint32, Uint32, Uint32, Uint32]:
        """Decode four packed FP4 bytes into four float16x2 pairs."""
        result = llvm.inline_asm(
            ir.Type.parse("!llvm.struct<(i32, i32, i32, i32)>"),
            [Uint32(packed_u32).ir_value(loc=loc, ip=ip)],
            """
            {
                .reg .b8 byte0, byte1, byte2, byte3;
                mov.b32 {byte0, byte1, byte2, byte3}, $4;
                cvt.rn.f16x2.e2m1x2 $0, byte0;
                cvt.rn.f16x2.e2m1x2 $1, byte1;
                cvt.rn.f16x2.e2m1x2 $2, byte2;
                cvt.rn.f16x2.e2m1x2 $3, byte3;
            }
            """,
            "=r,=r,=r,=r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
        return (
            Uint32(llvm.extractvalue(T.i32(), result, [0], loc=loc, ip=ip)),
            Uint32(llvm.extractvalue(T.i32(), result, [1], loc=loc, ip=ip)),
            Uint32(llvm.extractvalue(T.i32(), result, [2], loc=loc, ip=ip)),
            Uint32(llvm.extractvalue(T.i32(), result, [3], loc=loc, ip=ip)),
        )

    @cute.jit
    def warp_reduce(value, op: Callable, width: cutlass.Constexpr[int] = 32):
        """Reduce values across a warp using butterfly shuffles."""
        if cutlass.const_expr(isinstance(value, cute.TensorSSA)):
            result = cute.make_rmem_tensor(value.shape, value.dtype)
            result.store(value)
            for index in cutlass.range_constexpr(cute.size(value.shape)):
                result[index] = warp_reduce(result[index], op, width)
            return result.load()

        for index in cutlass.range_constexpr(int(math.log2(width))):
            value = op(value, cute.arch.shuffle_sync_bfly(value, offset=1 << index))
        return value

    @dsl_user_op
    def _ld_global_nc_u8(base_ptr: Int64, *, loc=None, ip=None) -> Uint32:
        return Uint32(
            llvm.inline_asm(
                T.i32(),
                [Int64(base_ptr).ir_value(loc=loc, ip=ip)],
                "ld.global.nc.u8 $0, [$1];",
                "=r,l",
                has_side_effects=False,
                is_align_stack=False,
                asm_dialect=llvm.AsmDialect.AD_ATT,
                loc=loc,
                ip=ip,
            )
        )

    @dsl_user_op
    def _bf16x2_to_f32x2(word: Uint32, *, loc=None, ip=None):
        result = llvm.inline_asm(
            llvm.StructType.get_literal([T.f32(), T.f32()]),
            [Uint32(word).ir_value(loc=loc, ip=ip)],
            "{.reg .b32 lo, hi; shl.b32 lo, $2, 16; "
            "and.b32 hi, $2, 0xffff0000; mov.b32 $0, lo; mov.b32 $1, hi;}",
            "=f,=f,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
        return (
            Float32(llvm.extractvalue(T.f32(), result, [0], loc=loc, ip=ip)),
            Float32(llvm.extractvalue(T.f32(), result, [1], loc=loc, ip=ip)),
        )

    class _DenseGemmW4A16CuteM1Jit:
        """One-output-per-warp CuTe DSL W4A16-NVFP4 decode GEMV."""

        def __init__(
            self,
            n: int,
            k: int,
            tile_n: int,
            threads: int,
            sf_vec_size: int = _SF_VEC_SIZE,
        ):
            self.n = n
            self.k = k
            self.tile_n = tile_n
            self.threads = threads
            self.sf_vec_size = sf_vec_size

        @cute.jit
        def _linear_e4m3_offset(
            self,
            row: Int32,
            sf_block: Int32,
            sf_cols: Int32,
        ) -> Int64:
            """Offset into the one-time-unswizzled padded scale matrix."""
            return Int64(row) * Int64(sf_cols) + Int64(sf_block)

        @cute.jit
        def _dot16(
            self,
            x_ptr: Int64,
            w0: Float32,
            w1: Float32,
            w2: Float32,
            w3: Float32,
            w4: Float32,
            w5: Float32,
            w6: Float32,
            w7: Float32,
            w8: Float32,
            w9: Float32,
            w10: Float32,
            w11: Float32,
            w12: Float32,
            w13: Float32,
            w14: Float32,
            w15: Float32,
        ) -> Float32:
            """Unscaled 16-value BF16 x decoded-FP4 dot product in FP32."""
            a0, a1, a2, a3, b0, b1, b2, b3 = ld_global_nc_32b_u32x8(x_ptr)
            acc = Float32(0.0)
            xf0, xf1 = _bf16x2_to_f32x2(a0)
            acc += xf0 * w0
            acc += xf1 * w1
            xf0, xf1 = _bf16x2_to_f32x2(a1)
            acc += xf0 * w2
            acc += xf1 * w3
            xf0, xf1 = _bf16x2_to_f32x2(a2)
            acc += xf0 * w4
            acc += xf1 * w5
            xf0, xf1 = _bf16x2_to_f32x2(a3)
            acc += xf0 * w6
            acc += xf1 * w7
            xf0, xf1 = _bf16x2_to_f32x2(b0)
            acc += xf0 * w8
            acc += xf1 * w9
            xf0, xf1 = _bf16x2_to_f32x2(b1)
            acc += xf0 * w10
            acc += xf1 * w11
            xf0, xf1 = _bf16x2_to_f32x2(b2)
            acc += xf0 * w12
            acc += xf1 * w13
            xf0, xf1 = _bf16x2_to_f32x2(b3)
            acc += xf0 * w14
            acc += xf1 * w15
            return acc

        @cute.jit
        def _decode8(self, q_word: Uint32):
            """Decode eight packed FP4 values from one u32 into FP32."""
            d0, d1, d2, d3 = fp4_decode_4bytes(q_word)
            f0, f1 = f16x2_to_f32x2(d0)
            f2, f3 = f16x2_to_f32x2(d1)
            f4, f5 = f16x2_to_f32x2(d2)
            f6, f7 = f16x2_to_f32x2(d3)
            return f0, f1, f2, f3, f4, f5, f6, f7

        @cute.kernel
        def kernel(
            self,
            x: cute.Tensor,
            w: cute.Tensor,
            sf: cute.Tensor,
            alpha: cute.Tensor,
            out: cute.Tensor,
        ):
            bidx, _, _ = cute.arch.block_idx()
            lane_idx = cute.arch.lane_idx()
            warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())

            # One active warp owns exactly one W row / output column.  Thus
            # ``tile_n == warps_per_cta``; tuning that one number changes the
            # number of output rows per CTA without changing the kernel body.
            row = Int32(bidx) * Int32(self.tile_n) + warp_idx
            sf_cols = ((Int32(self.k // self.sf_vec_size)) + Int32(3)) // Int32(4) * Int32(4)
            cols_blocks = Int32(self.k // self.sf_vec_size)
            sf_base = sf.iterator.toint()
            x_base = x.iterator.toint()
            w_base = w.iterator.toint()
            packed_cols = Int32(self.k // 2)
            alpha_value = Float32(alpha[Int32(0)])
            acc = Float32(0.0)

            # Each lane starts at its own 16-value scale group.  A warp covers
            # 32 groups (512 K values) per loop trip; this is the K16 variant
            # that won the majority of the Nano3.5 projection sweep.
            k16_block = lane_idx
            if warp_idx < Int32(self.tile_n) and row < Int32(self.n):
                w_row_base = w_base + Int64(row) * Int64(packed_cols)
                while k16_block < cols_blocks:
                    k0 = k16_block * Int32(self.sf_vec_size)
                    scale_offset = self._linear_e4m3_offset(row, k16_block, sf_cols)
                    scale = cvt_e4m3_to_f32_via_f16(_ld_global_nc_u8(sf_base + scale_offset))
                    q0, q1 = ld_global_nc_v2_u32(w_row_base + Int64(k0 // Int32(2)))
                    weights0 = self._decode8(q0)
                    weights1 = self._decode8(q1)
                    partial = self._dot16(x_base + Int64(k0) * Int64(2), *weights0, *weights1)
                    acc += partial * scale
                    k16_block += Int32(32)

            total = warp_reduce(acc, operator.add)
            if lane_idx == Int32(0) and row < Int32(self.n):
                out[row] = BFloat16(total * alpha_value)

        @cute.jit
        def __call__(
            self,
            x: cute.Tensor,
            w: cute.Tensor,
            sf: cute.Tensor,
            alpha: cute.Tensor,
            out: cute.Tensor,
            stream,
        ):
            grid = ((self.n + self.tile_n - 1) // self.tile_n, 1, 1)
            self.kernel(x, w, sf, alpha, out).launch(
                grid=grid,
                block=[self.threads, 1, 1],
                cluster=[1, 1, 1],
                stream=stream,
            )

else:

    class _DenseGemmW4A16CuteM1Jit:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            raise ImportError("cutlass.cute (DSL) not available")


class DenseGemmW4A16CuteM1Kernel:
    """CuTe-DSL W4A16 dense M=1 decode backend.

    There is exactly one production work decomposition: one output row per
    warp.  AutoTuner changes only the number of warps/output rows per CTA.
    """

    _DEFAULT_WARPS = 16
    _TUNABLE_WARPS = (16, 28, 32)
    _THREADS_PER_WARP = 32
    _MAX_WARPS_PER_CTA = 32
    _MAX_M = 1

    @classmethod
    def is_supported(cls, m: int, k: int, n: int) -> bool:
        return (
            _CUTE_AVAILABLE and m == cls._MAX_M and k > 0 and n > 0 and k % (2 * _SF_VEC_SIZE) == 0
        )

    def __init__(
        self,
        tile_n: int | None = None,
        threads: int | None = None,
        *,
        warps: int | None = None,
    ) -> None:
        self._warps_override = warps
        self._tile_n_override = tile_n
        self._threads_override = threads
        self._compile_cache: dict = {}
        self._explicit_stream_cache: dict[tuple[int, int, int], bool] = {}

    @staticmethod
    def _expected_blockscale_shape(n: int, k: int) -> tuple[int, int]:
        """Return the padded row-major E4M3 block-scale matrix shape."""
        sf_rows = ((n + 127) // 128) * 128
        sf_cols = (((k // _SF_VEC_SIZE) + 3) // 4) * 4
        return sf_rows, sf_cols

    @classmethod
    def _validate_warps(cls, warps: int) -> int:
        warps = int(warps)
        if warps <= 0 or warps > cls._MAX_WARPS_PER_CTA:
            raise NotImplementedError(
                f"M=1 CuTe decode requires 1..{cls._MAX_WARPS_PER_CTA} warps/CTA; got {warps}"
            )
        return warps

    @classmethod
    def _threads_for_warps(cls, warps: int) -> int:
        return cls._validate_warps(warps) * cls._THREADS_PER_WARP

    @classmethod
    def _warps_from_threads(cls, threads: int) -> int:
        threads = int(threads)
        if threads <= 0 or threads % cls._THREADS_PER_WARP != 0:
            raise NotImplementedError(
                "M=1 CuTe decode thread override must be a positive multiple "
                f"of {cls._THREADS_PER_WARP}; got {threads}"
            )
        return cls._validate_warps(threads // cls._THREADS_PER_WARP)

    def _select_warps(self) -> int:
        env_warps = os.environ.get("TRTLLM_W4A16_NVFP4_M1_WARPS")
        if env_warps is not None:
            return self._validate_warps(int(env_warps))
        if self._warps_override is not None:
            return self._validate_warps(self._warps_override)

        # Kept as backward-compatible aliases.  With one output per warp,
        # tile_n is not an independent parameter: it equals warps/CTA.
        env_tile_n = os.environ.get("TRTLLM_W4A16_NVFP4_M1_TILE_N")
        if env_tile_n is not None:
            return self._validate_warps(int(env_tile_n))
        if self._tile_n_override is not None:
            return self._validate_warps(self._tile_n_override)

        env_threads = os.environ.get("TRTLLM_W4A16_NVFP4_M1_THREADS")
        if env_threads is not None:
            return self._warps_from_threads(int(env_threads))
        if self._threads_override is not None:
            return self._warps_from_threads(self._threads_override)
        return self._DEFAULT_WARPS

    def _get_compiled(self, n: int, k: int, warps: int):
        if not _CUTE_AVAILABLE:
            raise NotImplementedError("cutlass.cute (DSL) not available")
        threads = self._threads_for_warps(warps)
        key = (n, k, warps)
        if key not in self._compile_cache:
            jit_instance = _DenseGemmW4A16CuteM1Jit(
                n=n,
                k=k,
                tile_n=warps,
                threads=threads,
            )
            sf_rows, sf_cols = self._expected_blockscale_shape(n, k)
            fake_args = (
                cute.runtime.make_fake_tensor(BFloat16, (k,), stride=(1,), assumed_align=16),
                cute.runtime.make_fake_tensor(
                    cutlass.Uint8, (n, k // 2), stride=(k // 2, 1), assumed_align=16
                ),
                cute.runtime.make_fake_tensor(
                    cutlass.Uint8,
                    (sf_rows, sf_cols),
                    stride=(sf_cols, 1),
                    assumed_align=16,
                ),
                cute.runtime.make_fake_tensor(Float32, (1,), stride=(1,), assumed_align=4),
                cute.runtime.make_fake_tensor(BFloat16, (n,), stride=(1,), assumed_align=16),
            )
            self._compile_cache[key] = cute.compile(
                jit_instance,
                *fake_args,
                cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
                options="--enable-tvm-ffi",
            )
        return self._compile_cache[key]

    def __call__(
        self,
        x: torch.Tensor,
        w_fp4: torch.Tensor,
        w_blockscale_linear_u8: torch.Tensor,
        w_alpha: torch.Tensor,
        out: Optional[torch.Tensor] = None,
        *,
        tactic: int = -1,
    ) -> torch.Tensor:
        if not _CUTE_AVAILABLE:
            raise NotImplementedError("cutlass.cute (DSL) not available")
        if x.dim() != 2 or int(x.shape[0]) != self._MAX_M:
            raise ValueError(f"M=1 CuTe GEMV expects x shape [1, K], got {tuple(x.shape)}")
        if x.dtype != torch.bfloat16:
            raise TypeError(f"x dtype must be bfloat16, got {x.dtype}")
        if not x.is_cuda:
            raise ValueError("x must be a CUDA tensor")
        if not x.is_contiguous():
            raise ValueError("x must be contiguous (kernel assumes row stride K)")
        if w_fp4.dim() != 2:
            raise ValueError(f"w_fp4 must be 2D, got shape {tuple(w_fp4.shape)}")

        k = int(x.shape[1])
        n = int(w_fp4.shape[0])
        if int(w_fp4.shape[1]) * 2 != k:
            raise ValueError(
                f"w_fp4 shape {tuple(w_fp4.shape)} (K={int(w_fp4.shape[1]) * 2}) "
                f"does not match x.shape[1]={k}"
            )
        if not self.is_supported(1, k, n):
            raise NotImplementedError(f"M=1 CuTe GEMV supports K % 32 == 0; got K={k}, N={n}")

        expected_scale_shape = self._expected_blockscale_shape(n, k)
        if tuple(w_blockscale_linear_u8.shape) != expected_scale_shape:
            raise ValueError(
                "w_blockscale_linear_u8 must have padded linear scale "
                f"shape {expected_scale_shape}; got {tuple(w_blockscale_linear_u8.shape)}"
            )
        if w_blockscale_linear_u8.dtype not in (torch.uint8, torch.float8_e4m3fn):
            raise TypeError(
                "w_blockscale_linear_u8 dtype must be uint8 or float8_e4m3fn; "
                f"got {w_blockscale_linear_u8.dtype}"
            )
        if w_blockscale_linear_u8.device != x.device:
            raise ValueError("w_blockscale_linear_u8 must be on the same device as x")
        if not w_blockscale_linear_u8.is_contiguous():
            raise ValueError("w_blockscale_linear_u8 must be contiguous")

        if out is None:
            out = torch.empty(1, n, dtype=torch.bfloat16, device=x.device)
        else:
            if tuple(out.shape) != (1, n):
                raise ValueError(f"out shape {tuple(out.shape)} != expected {(1, n)}")
            if out.dtype != torch.bfloat16:
                raise TypeError(f"out dtype must be bfloat16, got {out.dtype}")
            if out.device != x.device:
                raise ValueError("out must be on the same device as x")
            if not out.is_contiguous():
                raise ValueError("out must be contiguous")

        warps = self._select_warps() if tactic == -1 else self._validate_warps(tactic)
        if os.environ.get("TRTLLM_W4A16_NVFP4_M1_TRACE_DISPATCH") == "1":
            print(
                "[w4a16-m1-config] "
                "layout=one_output_per_warp "
                f"rows_per_cta={warps} warps={warps} "
                f"threads={self._threads_for_warps(warps)} "
                f"grid_x={(n + warps - 1) // warps}",
                flush=True,
            )

        compiled = self._get_compiled(n, k, warps)
        w_fp4_u8 = w_fp4.view(torch.uint8) if w_fp4.dtype != torch.uint8 else w_fp4
        w_blockscale_u8 = (
            w_blockscale_linear_u8.view(torch.uint8)
            if w_blockscale_linear_u8.dtype != torch.uint8
            else w_blockscale_linear_u8
        )
        args = (
            x.reshape(k),
            w_fp4_u8,
            w_blockscale_u8,
            w_alpha.reshape(1),
            out.reshape(n),
        )
        key = (n, k, warps)
        requires_explicit_stream = self._explicit_stream_cache.get(key)
        if requires_explicit_stream is True:
            stream = cuda.CUstream(torch.cuda.current_stream(x.device).cuda_stream)
            compiled(*args, stream)
        elif requires_explicit_stream is False:
            compiled(*args)
        else:
            try:
                compiled(*args)
                self._explicit_stream_cache[key] = False
            except Exception as exc:
                # TVM-FFI executors inject the environment stream, whereas an
                # uncached JIT executor exposes it as a required launch argument.
                message = str(exc)
                if "CALL_MISSING_ARGS" not in message or "stream" not in message:
                    raise
                stream = cuda.CUstream(torch.cuda.current_stream(x.device).cuda_stream)
                compiled(*args, stream)
                # Cache the convention so subsequent token launches avoid a
                # Python exception and the corresponding GPU-idle launch gap.
                self._explicit_stream_cache[key] = True
        return out


# The majority winner in the five-shape Nano3.5 sweep was one output row per
# warp.  These candidates tune only the number of such rows/warps in a CTA.
_W4A16_M1_AUTOTUNE_TACTICS = DenseGemmW4A16CuteM1Kernel._TUNABLE_WARPS

_KERNEL: DenseGemmW4A16CuteM1Kernel | None = None


def _get_kernel() -> DenseGemmW4A16CuteM1Kernel:
    global _KERNEL
    if _KERNEL is None:
        _KERNEL = DenseGemmW4A16CuteM1Kernel()
    return _KERNEL


class W4A16NVFP4CuteM1Runner(TunableRunner):
    """AutoTuner runner for the one-output-per-warp CuTe M=1 kernel."""

    tuning_config = TuningConfig(
        use_cold_l2_cache=True,
        use_cuda_graph=False,
        exclude_from_cache=True,
    )

    def unique_id(self):
        # The v2 key prevents reuse of cache entries for removed topologies.
        return ("w4a16_nvfp4_cute_m1_v2",)

    def get_valid_tactics(
        self,
        inputs: list[torch.Tensor],
        profile: OptimizationProfile,
        **kwargs,
    ) -> list:
        del profile, kwargs
        x, w_fp4, _, _ = inputs
        m, k = x.shape
        n = w_fp4.shape[0]
        if not DenseGemmW4A16CuteM1Kernel.is_supported(int(m), int(k), int(n)):
            return []
        return [-1, *_W4A16_M1_AUTOTUNE_TACTICS]

    def forward(
        self,
        inputs: list[torch.Tensor],
        *,
        tactic: int = -1,
        do_preparation: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        del kwargs
        x, w_fp4, w_blockscale_linear_u8, w_alpha = inputs
        kernel = _get_kernel()
        if do_preparation:
            out = torch.empty((1, w_fp4.shape[0]), dtype=torch.bfloat16, device=x.device)
            for candidate in self.get_valid_tactics(inputs, OptimizationProfile()):
                kernel(x, w_fp4, w_blockscale_linear_u8, w_alpha, out=out, tactic=candidate)
            return out
        return kernel(x, w_fp4, w_blockscale_linear_u8, w_alpha, tactic=tactic)


_RUNNER = W4A16NVFP4CuteM1Runner()


@torch.library.custom_op(
    "trtllm::w4a16_nvfp4_cute_m1_gemv",
    mutates_args=(),
    device_types="cuda",
)
def _w4a16_nvfp4_cute_m1_gemv_tunable(
    x: torch.Tensor,
    w_fp4: torch.Tensor,
    w_blockscale_linear_u8: torch.Tensor,
    w_alpha: torch.Tensor,
) -> torch.Tensor:
    inputs = [x, w_fp4, w_blockscale_linear_u8, w_alpha]
    runner, tactic = AutoTuner.get().choose_one(
        "trtllm::w4a16_nvfp4_cute_m1_gemv",
        [_RUNNER],
        _RUNNER.tuning_config,
        inputs,
    )
    return runner(inputs, tactic=tactic)


@_w4a16_nvfp4_cute_m1_gemv_tunable.register_fake
def _(
    x: torch.Tensor,
    w_fp4: torch.Tensor,
    w_blockscale_linear_u8: torch.Tensor,
    w_alpha: torch.Tensor,
) -> torch.Tensor:
    del w_blockscale_linear_u8, w_alpha
    return torch.empty((x.shape[0], w_fp4.shape[0]), dtype=torch.bfloat16, device=x.device)


def w4a16_nvfp4_cute_m1_gemv(
    x: torch.Tensor,
    w_fp4: torch.Tensor,
    w_blockscale_linear_u8: torch.Tensor,
    w_alpha: torch.Tensor,
    *,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run the AutoTuned SM120/SM121 one-output-per-warp W4A16 GEMV."""
    if os.environ.get("TRTLLM_W4A16_NVFP4_M1_TRACE_DISPATCH") == "1":
        m, k = x.shape
        print(
            f"[w4a16-dispatch] kernel=cute_m1_gemv M={m} K={k} N={w_fp4.shape[0]}",
            flush=True,
        )
    if out is None:
        return _w4a16_nvfp4_cute_m1_gemv_tunable(x, w_fp4, w_blockscale_linear_u8, w_alpha)
    return _get_kernel()(x, w_fp4, w_blockscale_linear_u8, w_alpha, out=out)


__all__ = [
    "DenseGemmW4A16CuteM1Kernel",
    "W4A16NVFP4CuteM1Runner",
    "w4a16_nvfp4_cute_m1_gemv",
]
