# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Warp-MMA GEMM for decode-sized M (<= 64 rows) over per-tensor FP8 (W8A8) and BF16 weights.

``C[M, :] = A[M, K] @ W.T`` with register-prefetched weights in two segments, each a whole number of CTAs:

* FP8 segment ``B[n_fp8, K]`` (E4M3, FP32 per-tensor scale ``Sw[0]``); ``fp8_act`` picks the activation:
  ``"fp8"``: producer-quantized ``A8`` (scale ``Sin[0]``) on ``m16n8k32`` E4M3, output ``acc * Sin * Sw``;
  ``"quant"``: BF16 ``A16`` quantized in registers bit-exactly like ``scaleMatrixPerTensorVec``, then as
  ``"fp8"``; ``"bf16"``: ``A16`` times the weights converted exactly to BF16, ``m16n8k16``, output ``acc * Sw``.
* BF16 segment ``B2[n_bf16, K]`` with ``A16``.

Output columns are the FP8 segment's then the BF16 segment's; columns ``>= split_out`` go to ``C2``. A SwiGLU
segment stacks its rows as (8 gate, 8 up) pairs and outputs ``silu(gate) * up``.

Layout invariants:

* A CTA owns ``n_tiles`` 8-row weight tiles and splits K over ``k_warps`` warps (reduced in shared memory).
* K goes in 64-element chunks; lane ``(r, q) = (lane // 4, lane % 4)`` holds ``W[n0 + r, k + 16q : k + 16q + 16]``
  and the same 16 columns of rows ``16t + r`` / ``16t + r + 8``. These registers are the MMA fragments because
  the in-chunk reduction order is permuted identically for A and W: FP8 logical k ``4q + i`` / ``4q + 16 + i``
  -> physical ``16q + 8j + i`` / ``16q + 8j + 4 + i``; BF16 ``2q + i`` / ``2q + 8 + i`` -> ``16q + 4j + i`` /
  ``16q + 4j + 2 + i``.
* ``swap_ab`` (M <= 8, even tiles per CTA): weights are the 16-row A operand, activations the 8-column B.
* ``ksplit`` > 1: a cluster of CTAs splits K and reduces through distributed shared memory in fixed rank order.
* PDL: weights, scales, conv weights and slots load before ``griddepcontrol.wait``; activations and conv
  states (written by earlier kernels) only after it. ``early_trigger`` releases the dependent after the
  weight loads are issued; the dependent still waits for this grid to complete before reading the output.
* ``conv_channels`` > 0 (GDN ``in_proj_qkvz``, MTP target-verify): the leading FP8 output columns go
  through the causal conv1d update (``num_rows / conv_tokens`` requests; window = 3-column conv state +
  BF16-rounded GEMM outputs) with the arithmetic of Triton's ``causal_conv1d_update``; it also writes the
  per-token windows and the final conv state.
"""

from __future__ import annotations

import cutlass
import cutlass.cute as cute
from cutlass import BFloat16, Float32, Int32, Uint8, Uint32, const_expr
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op

from .utils import griddepcontrol_launch_dependents, griddepcontrol_wait

try:
    from cutlass.memory import SmemAllocator as _SmemAllocator
except ImportError:  # older CuTe DSL
    from cutlass.utils import SmemAllocator as _SmemAllocator

CHUNK = 64  # K elements per lane quad (16 per lane)
N_TILE = 8
FP8_ACTS = ("fp8", "quant", "bf16")


@dsl_user_op
def _e4m3x4_to_bf16x4(x: Uint32, *, loc=None, ip=None):
    """Four packed E4M3 -> four BF16 (exact) as two ``Uint32`` bf16x2 (element 0 in the low half)."""
    out = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32()] * 2),
        [Uint32(x).ir_value(loc=loc, ip=ip)],
        "{\n\t"
        ".reg .b16 lo, hi, h0, h1, h2, h3;\n\t"
        ".reg .b32 p01, p23;\n\t"
        ".reg .f32 f0, f1, f2, f3;\n\t"
        "mov.b32 {lo, hi}, $2;\n\t"
        "cvt.rn.f16x2.e4m3x2 p01, lo;\n\t"
        "cvt.rn.f16x2.e4m3x2 p23, hi;\n\t"
        "mov.b32 {h0, h1}, p01;\n\t"
        "mov.b32 {h2, h3}, p23;\n\t"
        "cvt.f32.f16 f0, h0;\n\t"
        "cvt.f32.f16 f1, h1;\n\t"
        "cvt.f32.f16 f2, h2;\n\t"
        "cvt.f32.f16 f3, h3;\n\t"
        "cvt.rn.bf16x2.f32 $0, f1, f0;\n\t"
        "cvt.rn.bf16x2.f32 $1, f3, f2;\n\t"
        "}\n",
        "=r,=r,r",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return (
        Uint32(llvm.extractvalue(T.i32(), out, [0], loc=loc, ip=ip)),
        Uint32(llvm.extractvalue(T.i32(), out, [1], loc=loc, ip=ip)),
    )


@dsl_user_op
def _bf16x4_to_e4m3x4(x01: Uint32, x23: Uint32, factor: Float32, *, loc=None, ip=None) -> Uint32:
    """Four BF16 (two bf16x2 words, element 0 in the low half) times ``factor`` -> four E4M3 packed with
    element 0 in the low byte: FP32 multiply then ``cvt.rn.satfinite``, the arithmetic of the static
    per-tensor quantize kernel (``scaleMatrixPerTensorVec``)."""
    out = llvm.inline_asm(
        T.i32(),
        [
            Uint32(x01).ir_value(loc=loc, ip=ip),
            Uint32(x23).ir_value(loc=loc, ip=ip),
            Float32(factor).ir_value(loc=loc, ip=ip),
        ],
        "{\n\t"
        ".reg .b32 u0, u1, u2, u3;\n\t"
        ".reg .f32 f0, f1, f2, f3;\n\t"
        ".reg .b16 l, h;\n\t"
        "shl.b32 u0, $1, 16;\n\t"
        "and.b32 u1, $1, 0xFFFF0000;\n\t"
        "shl.b32 u2, $2, 16;\n\t"
        "and.b32 u3, $2, 0xFFFF0000;\n\t"
        "mov.b32 f0, u0;\n\t"
        "mov.b32 f1, u1;\n\t"
        "mov.b32 f2, u2;\n\t"
        "mov.b32 f3, u3;\n\t"
        "mul.rn.f32 f0, f0, $3;\n\t"
        "mul.rn.f32 f1, f1, $3;\n\t"
        "mul.rn.f32 f2, f2, $3;\n\t"
        "mul.rn.f32 f3, f3, $3;\n\t"
        "cvt.rn.satfinite.e4m3x2.f32 l, f1, f0;\n\t"
        "cvt.rn.satfinite.e4m3x2.f32 h, f3, f2;\n\t"
        "mov.b32 $0, {l, h};\n\t"
        "}\n",
        "=r,r,r,f",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return Uint32(out)


@dsl_user_op
def _f32_to_e4m3(x: Float32, *, loc=None, ip=None) -> Uint32:
    """One FP32 -> E4M3 (``cvt.rn.satfinite``) in the low byte of a ``Uint32``."""
    out = llvm.inline_asm(
        T.i32(),
        [Float32(x).ir_value(loc=loc, ip=ip)],
        "{\n\t"
        ".reg .b16 h;\n\t"
        "cvt.rn.satfinite.e4m3x2.f32 h, 0f00000000, $1;\n\t"
        "cvt.u32.u16 $0, h;\n\t"
        "}\n",
        "=r,f",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return Uint32(out)


@dsl_user_op
def _rcp_rn(x: Float32, *, loc=None, ip=None) -> Float32:
    """IEEE ``1 / x`` (``div.rn.f32``), as ``1.0f / s`` in the static quantize kernel."""
    out = llvm.inline_asm(
        T.f32(),
        [Float32(x).ir_value(loc=loc, ip=ip)],
        "div.rn.f32 $0, 0f3F800000, $1;",
        "=f,f",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return Float32(out)


@dsl_user_op
def _gdn_conv_silu(e0, e1, e2, e3, w0, w1, w2, w3, *, loc=None, ip=None) -> Float32:
    """One GDN causal-conv1d output: ``silu(((0 + w0 e0) + w1 e1) + w2 e2) + w3 e3)`` rounded to BF16 (returned
    as the exact FP32 value). Inputs are FP32 holding BF16 values. Matches the SASS of Triton's
    ``_causal_conv1d_update_kernel``: BF16 products (round to nearest), FP32 sum from 0,
    ``exp(-acc)`` = ``ex2.approx.f32(-acc * log2(e))``, ``acc / (1 + e)`` = ``div.full.f32``, BF16 round."""
    out = llvm.inline_asm(
        T.f32(),
        [Float32(v).ir_value(loc=loc, ip=ip) for v in (e0, e1, e2, e3, w0, w1, w2, w3)],
        "{\n\t"
        ".reg .f32 p, acc, t, e, d, y;\n\t"
        ".reg .b16 b, z;\n\t"
        "mov.b16 z, 0;\n\t"
        "mul.rn.f32 p, $1, $5;\n\t"
        "cvt.rn.bf16.f32 b, p;\n\t"
        "mov.b32 p, {z, b};\n\t"
        "add.rn.f32 acc, 0f00000000, p;\n\t"
        "mul.rn.f32 p, $2, $6;\n\t"
        "cvt.rn.bf16.f32 b, p;\n\t"
        "mov.b32 p, {z, b};\n\t"
        "add.rn.f32 acc, acc, p;\n\t"
        "mul.rn.f32 p, $3, $7;\n\t"
        "cvt.rn.bf16.f32 b, p;\n\t"
        "mov.b32 p, {z, b};\n\t"
        "add.rn.f32 acc, acc, p;\n\t"
        "mul.rn.f32 p, $4, $8;\n\t"
        "cvt.rn.bf16.f32 b, p;\n\t"
        "mov.b32 p, {z, b};\n\t"
        "add.rn.f32 acc, acc, p;\n\t"
        "sub.rn.f32 t, 0f00000000, acc;\n\t"
        "mul.rn.f32 t, t, 0f3FB8AA3B;\n\t"
        "ex2.approx.f32 e, t;\n\t"
        "add.rn.f32 d, e, 0f3F800000;\n\t"
        "div.full.f32 y, acc, d;\n\t"
        "cvt.rn.bf16.f32 b, y;\n\t"
        "mov.b32 $0, {z, b};\n\t"
        "}\n",
        "=f,f,f,f,f,f,f,f,f",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return Float32(out)


@dsl_user_op
def _mapa_shared_cluster(smem_ptr, peer_rank, *, loc=None, ip=None) -> Int32:
    """Shared-memory address of ``smem_ptr`` in cluster CTA ``peer_rank`` (``mapa.shared::cluster``)."""
    return Int32(
        llvm.inline_asm(
            T.i32(),
            [smem_ptr.toint(loc=loc, ip=ip).ir_value(), Int32(peer_rank).ir_value(loc=loc, ip=ip)],
            "mapa.shared::cluster.u32 $0, $1, $2;",
            "=r,r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _ld_shared_cluster_f32(addr, *, loc=None, ip=None) -> Float32:
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Int32(addr).ir_value(loc=loc, ip=ip)],
            "ld.shared::cluster.f32 $0, [$1];",
            "=f,r",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


def _mma_asm(shape_types: str):
    @dsl_user_op
    def _mma(a0, a1, a2, a3, b0, b1, c0, c1, c2, c3, *, loc=None, ip=None):
        out = llvm.inline_asm(
            llvm.StructType.get_literal([T.f32()] * 4),
            [Uint32(v).ir_value(loc=loc, ip=ip) for v in (a0, a1, a2, a3, b0, b1)]
            + [Float32(v).ir_value(loc=loc, ip=ip) for v in (c0, c1, c2, c3)],
            f"mma.sync.aligned.{shape_types} "
            "{$0, $1, $2, $3}, {$4, $5, $6, $7}, {$8, $9}, {$10, $11, $12, $13};",
            "=f,=f,=f,=f,r,r,r,r,r,r,f,f,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
        return tuple(
            Float32(llvm.extractvalue(T.f32(), out, [i], loc=loc, ip=ip)) for i in range(4)
        )

    return _mma


_mma_bf16_m16n8k16 = _mma_asm("m16n8k16.row.col.f32.bf16.bf16.f32")
_mma_e4m3_m16n8k32 = _mma_asm("m16n8k32.row.col.f32.e4m3.e4m3.f32")


def _g2r_atom(dtype, streaming: bool):
    """128-bit global -> register copy atom (streaming cache hint for weights when this DSL has it)."""
    modes = getattr(cute.nvgpu, "LoadCacheMode", None)
    mode = getattr(modes, "STREAMING" if streaming else "ALWAYS", None) if modes else None
    if mode is not None:
        try:
            return cute.make_copy_atom(
                cute.nvgpu.CopyG2ROp(), dtype, num_bits_per_copy=128, load_cache_mode=mode
            )
        except TypeError:
            pass
    return cute.make_copy_atom(cute.nvgpu.CopyG2ROp(), dtype, num_bits_per_copy=128)


class LowMMmaGemmKernel:
    """``num_rows`` (<= 64) x K GEMM over an FP8 and a BF16 weight segment."""

    def __init__(
        self,
        *,
        num_rows: int,
        k_extent: int,
        k_warps: int,
        use_pdl: bool,
        n_fp8: int = 0,
        n_tiles: int = 1,
        fp8_act: str = "fp8",
        swiglu_fp8: bool = False,
        n_bf16: int = 0,
        n_tiles_bf16: int = 1,
        swiglu_bf16: bool = False,
        split_out: int = 0,
        n_valid: int = 0,
        quant_out: bool = False,
        early_trigger: bool = False,
        release_after_wait: bool = False,
        ksplit: int = 1,
        swap_ab: bool = False,
        min_blocks: int = 1,
        conv_channels: int = 0,
        conv_tokens: int = 1,
        conv_write_state: bool = True,
    ) -> None:
        """Configure one specialization (see the module doc for ``fp8_act``, ``swap_ab``, ``ksplit``, ``conv_*``).

        Args:
            n_fp8, n_bf16: segment weight rows, multiples of ``8 * n_tiles`` / ``8 * n_tiles_bf16``; either may be 0.
            split_out: output columns ``>= split_out`` go to ``C2`` (0: all to ``C``).
            n_valid: output columns ``>= n_valid`` (zero-padded weight rows) are not stored (0: all are).
            quant_out: ``C`` is uint8 E4M3, quantized with the static scale ``So[0]`` (``x * (1/So)``, satfinite).
            early_trigger: release the dependent once the weight loads are issued instead of after the stores.
            release_after_wait: release it right after the dependency wait (overrides ``early_trigger``).
            min_blocks: launch-bounds occupancy hint (CTAs per SM).
        """
        if ksplit not in (1, 2, 4, 8):
            raise ValueError(f"ksplit={ksplit} must be 1, 2, 4 or 8")
        if k_extent % (CHUNK * k_warps * ksplit):
            raise ValueError(f"K={k_extent} must be a multiple of {CHUNK * k_warps * ksplit}")
        if not 1 <= num_rows <= 64:
            raise ValueError(f"num_rows={num_rows} must be in [1, 64]")
        for tiles in (n_tiles, n_tiles_bf16):
            if tiles not in (1, 2, 4):
                raise ValueError(f"n_tiles={tiles} must be 1, 2 or 4")
        if n_fp8 % (N_TILE * n_tiles) or n_bf16 % (N_TILE * n_tiles_bf16):
            raise ValueError(
                f"segment rows {n_fp8}/{n_bf16} must be multiples of the CTA tiles "
                f"{N_TILE * n_tiles}/{N_TILE * n_tiles_bf16}"
            )
        if n_fp8 + n_bf16 == 0:
            raise ValueError("empty GEMM")
        if fp8_act not in FP8_ACTS:
            raise ValueError(f"fp8_act={fp8_act!r} must be one of {FP8_ACTS}")
        if (swiglu_fp8 and n_tiles % 2) or (swiglu_bf16 and n_tiles_bf16 % 2):
            raise ValueError("a SwiGLU segment needs an even number of tiles per CTA")
        self.num_rows = num_rows
        self.row_tiles = (num_rows + 15) // 16
        self.k_extent = k_extent
        self.k_warps = k_warps
        self.ksplit = ksplit
        self.swap_ab = swap_ab and num_rows <= 8
        self.min_blocks = min_blocks
        self.chunks_per_warp = k_extent // (CHUNK * k_warps * ksplit)
        self.n_fp8 = n_fp8
        self.n_bf16 = n_bf16
        self.n_tiles = n_tiles
        self.n_tiles_bf16 = n_tiles_bf16
        self.fp8_act = fp8_act
        self.swiglu_fp8 = swiglu_fp8
        self.swiglu_bf16 = swiglu_bf16
        self.fp8_ctas = n_fp8 // (N_TILE * n_tiles)
        self.bf16_ctas = n_bf16 // (N_TILE * n_tiles_bf16)
        self.fp8_out = n_fp8 // 2 if swiglu_fp8 else n_fp8
        self.bf16_out = n_bf16 // 2 if swiglu_bf16 else n_bf16
        self.split_out = split_out
        self.n_valid = n_valid if n_valid else self.fp8_out + self.bf16_out
        self.quant_out = quant_out
        self.use_pdl = use_pdl
        self.early_trigger = early_trigger and not release_after_wait
        self.release_after_wait = release_after_wait
        cols = 0
        if n_fp8:
            cols = N_TILE * n_tiles
        if n_bf16:
            cols = max(cols, N_TILE * n_tiles_bf16)
        self.max_cols = cols
        self.conv_channels = conv_channels
        self.conv_tokens = conv_tokens
        self.conv_write_state = conv_write_state
        if conv_channels:
            fp8_cols = N_TILE * n_tiles
            if (
                ksplit != 1
                or swiglu_fp8
                or quant_out
                or conv_channels % fp8_cols
                or conv_channels > self.fp8_out
                or (split_out and conv_channels > split_out)
                or num_rows % conv_tokens
                or (num_rows // conv_tokens) * fp8_cols > 32 * k_warps
            ):
                raise ValueError(
                    f"conv epilogue: channels={conv_channels} tokens={conv_tokens} rows={num_rows} "
                    f"cols/CTA={fp8_cols} warps={k_warps} ksplit={ksplit} needs whole FP8 CTAs, ksplit 1, no "
                    "SwiGLU / quantized output, and one (request, channel) pair per thread"
                )

    @cute.jit
    def __call__(
        self,
        gA16: cute.Tensor,
        gA8: cute.Tensor,
        gB: cute.Tensor,
        gSw: cute.Tensor,
        gSin: cute.Tensor,
        gSo: cute.Tensor,
        gB2: cute.Tensor,
        gC: cute.Tensor,
        gC2: cute.Tensor,
        gCw: cute.Tensor,
        gCs: cute.Tensor,
        gSlot: cute.Tensor,
        gWin: cute.Tensor,
        stream,
    ) -> None:
        copy_a16 = _g2r_atom(BFloat16, streaming=False)
        copy_a8 = _g2r_atom(Uint8, streaming=False)
        copy_b = _g2r_atom(Uint8, streaming=True)
        copy_b2 = _g2r_atom(BFloat16, streaming=True)
        smem = (
            (self.k_warps + (1 if self.ksplit > 1 else 0) + (1 if self.conv_channels else 0))
            * self.row_tiles
            * 16
            * self.max_cols
            * 4
        )
        grid = [(self.fp8_ctas + self.bf16_ctas) * self.ksplit, 1, 1]
        kernel = self.kernel(
            gA16,
            gA8,
            gB,
            gSw,
            gSin,
            gSo,
            gB2,
            gC,
            gC2,
            gCw,
            gCs,
            gSlot,
            gWin,
            copy_a16,
            copy_a8,
            copy_b,
            copy_b2,
        )
        if const_expr(self.ksplit > 1):
            kernel.launch(
                grid=grid,
                block=[32 * self.k_warps, 1, 1],
                cluster=[self.ksplit, 1, 1],
                smem=smem,
                stream=stream,
                use_pdl=self.use_pdl,
                min_blocks_per_mp=self.min_blocks,
            )
        else:
            kernel.launch(
                grid=grid,
                block=[32 * self.k_warps, 1, 1],
                smem=smem,
                stream=stream,
                use_pdl=self.use_pdl,
                min_blocks_per_mp=self.min_blocks,
            )

    @cute.kernel
    def kernel(
        self,
        gA16: cute.Tensor,
        gA8: cute.Tensor,
        gB: cute.Tensor,
        gSw: cute.Tensor,
        gSin: cute.Tensor,
        gSo: cute.Tensor,
        gB2: cute.Tensor,
        gC: cute.Tensor,
        gC2: cute.Tensor,
        gCw: cute.Tensor,
        gCs: cute.Tensor,
        gSlot: cute.Tensor,
        gWin: cute.Tensor,
        copy_a16: cute.CopyAtom,
        copy_a8: cute.CopyAtom,
        copy_b: cute.CopyAtom,
        copy_b2: cute.CopyAtom,
    ) -> None:
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        smem = _SmemAllocator()
        red = smem.allocate_tensor(
            Float32,
            cute.make_layout(
                (self.k_warps, self.row_tiles * 16, self.max_cols),
                stride=(self.row_tiles * 16 * self.max_cols, self.max_cols, 1),
            ),
            byte_alignment=16,
        )
        part = red
        xs = red
        if const_expr(self.conv_channels > 0):
            # BF16-rounded conv-channel GEMM outputs (the conv window), FP32 storage.
            xs = smem.allocate_tensor(
                Float32,
                cute.make_layout((self.row_tiles * 16, self.max_cols), stride=(self.max_cols, 1)),
                byte_alignment=16,
            )
        group = bidx
        kslice = 0
        if const_expr(self.ksplit > 1):
            part = smem.allocate_tensor(
                Float32,
                cute.make_layout((self.row_tiles * 16, self.max_cols), stride=(self.max_cols, 1)),
                byte_alignment=16,
            )
            # Clusters are tiles of consecutive block indices: rank = bidx % ksplit.
            group = bidx // self.ksplit
            kslice = bidx % self.ksplit
        if const_expr(self.fp8_ctas > 0 and self.bf16_ctas > 0):
            if group < self.fp8_ctas:
                self._cta(
                    gA16,
                    gA8,
                    gB,
                    gSw,
                    gSin,
                    gSo,
                    gB2,
                    gC,
                    gC2,
                    gCw,
                    gCs,
                    gSlot,
                    gWin,
                    red,
                    part,
                    xs,
                    copy_a16,
                    copy_a8,
                    copy_b,
                    copy_b2,
                    tidx,
                    group,
                    kslice,
                    True,
                )
            else:
                self._cta(
                    gA16,
                    gA8,
                    gB,
                    gSw,
                    gSin,
                    gSo,
                    gB2,
                    gC,
                    gC2,
                    gCw,
                    gCs,
                    gSlot,
                    gWin,
                    red,
                    part,
                    xs,
                    copy_a16,
                    copy_a8,
                    copy_b,
                    copy_b2,
                    tidx,
                    group - self.fp8_ctas,
                    kslice,
                    False,
                )
        elif const_expr(self.fp8_ctas > 0):
            self._cta(
                gA16,
                gA8,
                gB,
                gSw,
                gSin,
                gSo,
                gB2,
                gC,
                gC2,
                gCw,
                gCs,
                gSlot,
                gWin,
                red,
                part,
                xs,
                copy_a16,
                copy_a8,
                copy_b,
                copy_b2,
                tidx,
                group,
                kslice,
                True,
            )
        else:
            self._cta(
                gA16,
                gA8,
                gB,
                gSw,
                gSin,
                gSo,
                gB2,
                gC,
                gC2,
                gCw,
                gCs,
                gSlot,
                gWin,
                red,
                part,
                xs,
                copy_a16,
                copy_a8,
                copy_b,
                copy_b2,
                tidx,
                group,
                kslice,
                False,
            )

    @cute.jit
    def _cta(
        self,
        gA16: cute.Tensor,
        gA8: cute.Tensor,
        gB: cute.Tensor,
        gSw: cute.Tensor,
        gSin: cute.Tensor,
        gSo: cute.Tensor,
        gB2: cute.Tensor,
        gC: cute.Tensor,
        gC2: cute.Tensor,
        gCw: cute.Tensor,
        gCs: cute.Tensor,
        gSlot: cute.Tensor,
        gWin: cute.Tensor,
        red: cute.Tensor,
        part: cute.Tensor,
        xs: cute.Tensor,
        copy_a16: cute.CopyAtom,
        copy_a8: cute.CopyAtom,
        copy_b: cute.CopyAtom,
        copy_b2: cute.CopyAtom,
        tidx,
        cta,
        kslice,
        fp8: cutlass.Constexpr,
    ) -> None:
        warp = tidx // 32
        lane = tidx % 32
        r = lane // 4
        q = lane % 4

        num_rows: cutlass.Constexpr = self.num_rows
        row_tiles: cutlass.Constexpr = self.row_tiles
        k_warps: cutlass.Constexpr = self.k_warps
        chunks: cutlass.Constexpr = self.chunks_per_warp
        n_tiles: cutlass.Constexpr = self.n_tiles if fp8 else self.n_tiles_bf16
        swiglu: cutlass.Constexpr = self.swiglu_fp8 if fp8 else self.swiglu_bf16
        fp8_mma: cutlass.Constexpr = fp8 and self.fp8_act != "bf16"
        cols: cutlass.Constexpr = N_TILE * n_tiles
        out_cols: cutlass.Constexpr = cols // 2 if swiglu else cols
        out_col0: cutlass.Constexpr = 0 if fp8 else self.fp8_out

        n0 = cta * cols
        k_warp0 = (kslice * k_warps + warp) * (chunks * CHUNK)

        # Weights and scales do not depend on the producer: load them before griddepcontrol.wait.
        w8 = cute.make_rmem_tensor(cute.make_layout((n_tiles * chunks, 16), stride=(16, 1)), Uint8)
        w16 = cute.make_rmem_tensor(
            cute.make_layout((2 * n_tiles * chunks, 8), stride=(8, 1)), BFloat16
        )
        alpha = Float32(1.0)
        factor = Float32(1.0)
        if const_expr(fp8):
            gB_vec = cute.logical_divide(gB, (None, 16))
            for nt in cutlass.range_constexpr(n_tiles):
                b_row = n0 + N_TILE * nt + r
                for c in cutlass.range_constexpr(chunks):
                    k0 = k_warp0 + c * CHUNK + 16 * q
                    cute.copy(copy_b, gB_vec[b_row, (None, k0 // 16)], w8[nt * chunks + c, None])
            alpha = gSw[0]
            if const_expr(fp8_mma):
                s_in = gSin[0]
                alpha = s_in * alpha
                if const_expr(self.fp8_act == "quant"):
                    factor = _rcp_rn(s_in)
        else:
            gB2_vec = cute.logical_divide(gB2, (None, 8))
            for nt in cutlass.range_constexpr(n_tiles):
                b_row = n0 + N_TILE * nt + r
                for c in cutlass.range_constexpr(chunks):
                    k0 = k_warp0 + c * CHUNK + 16 * q
                    slot = nt * chunks + c
                    cute.copy(copy_b2, gB2_vec[b_row, (None, k0 // 8)], w16[2 * slot, None])
                    cute.copy(copy_b2, gB2_vec[b_row, (None, k0 // 8 + 1)], w16[2 * slot + 1, None])

        out_factor = Float32(1.0)
        if const_expr(self.quant_out):
            out_factor = _rcp_rn(gSo[0])

        # One (request, channel) pair per thread, indices clamped so every load is in bounds. Conv weights and
        # slots (host-copied before the step) load before the dependency wait, the conv state after it.
        conv: cutlass.Constexpr = fp8 and self.conv_channels > 0
        conv_t: cutlass.Constexpr = self.conv_tokens
        conv_pairs: cutlass.Constexpr = (num_rows // conv_t) * out_cols if conv else 1
        cv_req = Int32(0)
        cv_col = Int32(0)
        cv_ch = Int32(0)
        cv_slot = Int32(0)
        cv_s0 = Float32(0.0)
        cv_s1 = Float32(0.0)
        cv_s2 = Float32(0.0)
        cv_w0 = Float32(0.0)
        cv_w1 = Float32(0.0)
        cv_w2 = Float32(0.0)
        cv_w3 = Float32(0.0)
        if const_expr(conv):
            pair = cutlass.min(tidx, Int32(conv_pairs - 1))
            cv_req = pair // out_cols
            cv_col = pair % out_cols
            cv_ch = cutlass.min(cta * out_cols + cv_col, Int32(self.conv_channels - 1))
            cv_slot = gSlot[cv_req]
            cv_w0 = gCw[cv_ch, 0].to(Float32)
            cv_w1 = gCw[cv_ch, 1].to(Float32)
            cv_w2 = gCw[cv_ch, 2].to(Float32)
            cv_w3 = gCw[cv_ch, 3].to(Float32)

        if const_expr(self.use_pdl and self.early_trigger):
            griddepcontrol_launch_dependents()
        if const_expr(self.use_pdl):
            griddepcontrol_wait()
        if const_expr(self.use_pdl and self.release_after_wait):
            # Every launch before this one has completed here. Do not move this above the wait.
            griddepcontrol_launch_dependents()
        if const_expr(conv):
            # Written by earlier kernels, so only after the wait; these loads overlap the main loop.
            slot_c = cv_slot - cutlass.min(cv_slot, Int32(0))  # max(slot, 0)
            cv_s0 = gCs[slot_c, cv_ch, 0].to(Float32)
            cv_s1 = gCs[slot_c, cv_ch, 1].to(Float32)
            cv_s2 = gCs[slot_c, cv_ch, 2].to(Float32)

        acc = cute.make_rmem_tensor(
            cute.make_layout((n_tiles * row_tiles, 4), stride=(4, 1)), Float32
        )
        acc.fill(0.0)
        swap: cutlass.Constexpr = self.swap_ab and n_tiles % 2 == 0
        if const_expr(fp8_mma):
            if const_expr(swap):
                self._mainloop_fp8_swap(
                    acc, gA16, gA8, w8, factor, copy_a16, copy_a8, k_warp0, r, q, n_tiles
                )
            else:
                self._mainloop_fp8(
                    acc, gA16, gA8, w8, factor, copy_a16, copy_a8, k_warp0, r, q, n_tiles
                )
        else:
            if const_expr(swap):
                self._mainloop_bf16_swap(acc, gA16, w8, w16, copy_a16, k_warp0, r, q, n_tiles, fp8)
            else:
                self._mainloop_bf16(acc, gA16, w8, w16, copy_a16, k_warp0, r, q, n_tiles, fp8)

        if const_expr(swap):
            # acc[p] of tile pair p: (act row 2q, weight row 16p + r), (2q + 1, 16p + r), then + 8 rows.
            for pair in cutlass.range_constexpr(n_tiles // 2):
                col = 2 * N_TILE * pair + r
                red[warp, 2 * q, col] = acc[pair, 0]
                red[warp, 2 * q + 1, col] = acc[pair, 1]
                red[warp, 2 * q, col + N_TILE] = acc[pair, 2]
                red[warp, 2 * q + 1, col + N_TILE] = acc[pair, 3]
        else:
            for nt in cutlass.range_constexpr(n_tiles):
                for t in cutlass.range_constexpr(row_tiles):
                    col = N_TILE * nt + 2 * q
                    red[warp, 16 * t + r, col] = acc[nt * row_tiles + t, 0]
                    red[warp, 16 * t + r, col + 1] = acc[nt * row_tiles + t, 1]
                    red[warp, 16 * t + r + 8, col] = acc[nt * row_tiles + t, 2]
                    red[warp, 16 * t + r + 8, col + 1] = acc[nt * row_tiles + t, 3]
        cute.arch.sync_threads()
        if const_expr(conv):
            if cta * out_cols < self.conv_channels:
                self._conv_epilogue(
                    gC,
                    gCs,
                    gWin,
                    red,
                    xs,
                    tidx,
                    cta,
                    alpha,
                    out_cols,
                    conv_pairs,
                    cv_req,
                    cv_col,
                    cv_ch,
                    cv_slot,
                    cv_s0,
                    cv_s1,
                    cv_s2,
                    cv_w0,
                    cv_w1,
                    cv_w2,
                    cv_w3,
                )
            else:
                self._epilogue_k1(
                    gC, gC2, red, tidx, cta, alpha, out_factor, out_cols, out_col0, swiglu
                )
        elif const_expr(self.ksplit == 1):
            for idx in cutlass.range(tidx, num_rows * out_cols, 32 * k_warps):
                row = idx // out_cols
                col = idx % out_cols
                value = Float32(0.0)
                if const_expr(swiglu):
                    # Output column 8p + j of the CTA: gate in tile 2p, up in tile 2p + 1.
                    gcol = (col // N_TILE) * (2 * N_TILE) + col % N_TILE
                    gate = Float32(0.0)
                    up = Float32(0.0)
                    for w in cutlass.range_constexpr(k_warps):
                        gate = gate + red[w, row, gcol]
                    for w in cutlass.range_constexpr(k_warps):
                        up = up + red[w, row, gcol + N_TILE]
                    value = self._swiglu(gate, up, alpha)
                else:
                    for w in cutlass.range_constexpr(k_warps):
                        value = value + red[w, row, col]
                    value = value * alpha
                self._finish(gC, gC2, row, out_col0 + cta * out_cols + col, value, out_factor)
        else:
            for idx in cutlass.range(tidx, num_rows * cols, 32 * k_warps):
                row = idx // cols
                col = idx % cols
                total = Float32(0.0)
                for w in cutlass.range_constexpr(k_warps):
                    total = total + red[w, row, col]
                part[row, col] = total
            cute.arch.cluster_arrive()
            cute.arch.cluster_wait()
            # Fixed rank-order cluster sum; rank r finishes the outputs idx = r (mod ksplit).
            for idx in cutlass.range(
                kslice + tidx * self.ksplit, num_rows * out_cols, 32 * k_warps * self.ksplit
            ):
                row = idx // out_cols
                col = idx % out_cols
                value = Float32(0.0)
                if const_expr(swiglu):
                    gcol = (col // N_TILE) * (2 * N_TILE) + col % N_TILE
                    gate = self._cluster_sum(part, row, gcol)
                    up = self._cluster_sum(part, row, gcol + N_TILE)
                    value = self._swiglu(gate, up, alpha)
                else:
                    value = self._cluster_sum(part, row, col) * alpha
                self._finish(gC, gC2, row, out_col0 + cta * out_cols + col, value, out_factor)
            # Keep this CTA's shared memory alive until every peer has read it.
            cute.arch.cluster_arrive()
            cute.arch.cluster_wait()

        if const_expr(self.use_pdl and not self.early_trigger and not self.release_after_wait):
            griddepcontrol_launch_dependents()

    @cute.jit
    def _epilogue_k1(
        self,
        gC,
        gC2,
        red,
        tidx,
        cta,
        alpha,
        out_factor,
        out_cols: cutlass.Constexpr,
        out_col0: cutlass.Constexpr,
        swiglu: cutlass.Constexpr,
    ) -> None:
        """The ksplit == 1 epilogue, for CTAs of a conv-epilogue launch past the conv channels."""
        num_rows: cutlass.Constexpr = self.num_rows
        k_warps: cutlass.Constexpr = self.k_warps
        for idx in cutlass.range(tidx, num_rows * out_cols, 32 * k_warps):
            row = idx // out_cols
            col = idx % out_cols
            value = Float32(0.0)
            if const_expr(swiglu):
                gcol = (col // N_TILE) * (2 * N_TILE) + col % N_TILE
                gate = Float32(0.0)
                up = Float32(0.0)
                for w in cutlass.range_constexpr(k_warps):
                    gate = gate + red[w, row, gcol]
                for w in cutlass.range_constexpr(k_warps):
                    up = up + red[w, row, gcol + N_TILE]
                value = self._swiglu(gate, up, alpha)
            else:
                for w in cutlass.range_constexpr(k_warps):
                    value = value + red[w, row, col]
                value = value * alpha
            self._finish(gC, gC2, row, out_col0 + cta * out_cols + col, value, out_factor)

    @cute.jit
    def _conv_epilogue(
        self,
        gC,
        gCs,
        gWin,
        red,
        xs,
        tidx,
        cta,
        alpha,
        out_cols: cutlass.Constexpr,
        conv_pairs: cutlass.Constexpr,
        cv_req,
        cv_col,
        cv_ch,
        cv_slot,
        cv_s0,
        cv_s1,
        cv_s2,
        cv_w0,
        cv_w1,
        cv_w2,
        cv_w3,
    ) -> None:
        """GDN causal conv1d update of this CTA's channels: one thread per (request, channel) writes the conv
        outputs of all tokens, the windows after every token and the conv state after the last one."""
        num_rows: cutlass.Constexpr = self.num_rows
        k_warps: cutlass.Constexpr = self.k_warps
        conv_t: cutlass.Constexpr = self.conv_tokens
        for idx in cutlass.range(tidx, num_rows * out_cols, 32 * k_warps):
            row = idx // out_cols
            col = idx % out_cols
            value = Float32(0.0)
            for w in cutlass.range_constexpr(k_warps):
                value = value + red[w, row, col]
            value = value * alpha
            xs[row, col] = value.to(BFloat16).to(Float32)
        cute.arch.sync_threads()
        if tidx < conv_pairs:
            row0 = cv_req * conv_t
            ext = [cv_s0, cv_s1, cv_s2]
            for t in cutlass.range_constexpr(conv_t):
                ext.append(xs[row0 + t, cv_col])
            for t in cutlass.range_constexpr(conv_t):
                y = _gdn_conv_silu(
                    ext[t], ext[t + 1], ext[t + 2], ext[t + 3], cv_w0, cv_w1, cv_w2, cv_w3
                )
                gC[row0 + t, cv_ch] = y.to(gC.element_type)
            if cv_slot >= 0:
                for t in cutlass.range_constexpr(conv_t):
                    gWin[cv_req, t, cv_ch, 0] = ext[t + 1].to(gWin.element_type)
                    gWin[cv_req, t, cv_ch, 1] = ext[t + 2].to(gWin.element_type)
                    gWin[cv_req, t, cv_ch, 2] = ext[t + 3].to(gWin.element_type)
                if const_expr(self.conv_write_state):
                    gCs[cv_slot, cv_ch, 0] = ext[conv_t].to(gCs.element_type)
                    gCs[cv_slot, cv_ch, 1] = ext[conv_t + 1].to(gCs.element_type)
                    gCs[cv_slot, cv_ch, 2] = ext[conv_t + 2].to(gCs.element_type)

    @cute.jit
    def _mainloop_fp8(
        self,
        acc: cute.Tensor,
        gA16: cute.Tensor,
        gA8: cute.Tensor,
        w8: cute.Tensor,
        factor: Float32,
        copy_a16: cute.CopyAtom,
        copy_a8: cute.CopyAtom,
        k_warp0,
        r,
        q,
        n_tiles: cutlass.Constexpr,
    ) -> None:
        """E4M3 x E4M3 ``m16n8k32`` main loop (activations FP8, or BF16 quantized here)."""
        num_rows: cutlass.Constexpr = self.num_rows
        row_tiles: cutlass.Constexpr = self.row_tiles
        chunks: cutlass.Constexpr = self.chunks_per_warp
        quant: cutlass.Constexpr = self.fp8_act == "quant"

        a_lo = cute.make_rmem_tensor(cute.make_layout((16,)), Uint8)
        a_hi = cute.make_rmem_tensor(cute.make_layout((16,)), Uint8)
        a_lo_u32 = cute.recast_tensor(a_lo, Uint32)
        a_hi_u32 = cute.recast_tensor(a_hi, Uint32)
        x_lo0 = cute.make_rmem_tensor(cute.make_layout((8,)), BFloat16)
        x_lo1 = cute.make_rmem_tensor(cute.make_layout((8,)), BFloat16)
        x_hi0 = cute.make_rmem_tensor(cute.make_layout((8,)), BFloat16)
        x_hi1 = cute.make_rmem_tensor(cute.make_layout((8,)), BFloat16)
        x_lo0_u32 = cute.recast_tensor(x_lo0, Uint32)
        x_lo1_u32 = cute.recast_tensor(x_lo1, Uint32)
        x_hi0_u32 = cute.recast_tensor(x_hi0, Uint32)
        x_hi1_u32 = cute.recast_tensor(x_hi1, Uint32)
        gA16_vec = cute.logical_divide(gA16, (None, 8))
        gA8_vec = cute.logical_divide(gA8, (None, 16))

        for c in cutlass.range_constexpr(chunks):
            k0 = k_warp0 + c * CHUNK + 16 * q
            for t in cutlass.range_constexpr(row_tiles):
                has_hi: cutlass.Constexpr = 16 * t + 8 < num_rows
                # Rows past num_rows load the last row instead (no divergent branch around the loads and
                # the quantization); their outputs are computed but never stored.
                row_lo = self._clamp_row(16 * t + r, 16 * t + 8)
                row_hi = self._clamp_row(16 * t + r + 8, 16 * t + 16)
                if const_expr(not has_hi):
                    a_hi_u32.fill(0)
                if const_expr(quant):
                    cute.copy(copy_a16, gA16_vec[row_lo, (None, k0 // 8)], x_lo0)
                    cute.copy(copy_a16, gA16_vec[row_lo, (None, k0 // 8 + 1)], x_lo1)
                    a_lo_u32[0] = _bf16x4_to_e4m3x4(x_lo0_u32[0], x_lo0_u32[1], factor)
                    a_lo_u32[1] = _bf16x4_to_e4m3x4(x_lo0_u32[2], x_lo0_u32[3], factor)
                    a_lo_u32[2] = _bf16x4_to_e4m3x4(x_lo1_u32[0], x_lo1_u32[1], factor)
                    a_lo_u32[3] = _bf16x4_to_e4m3x4(x_lo1_u32[2], x_lo1_u32[3], factor)
                    if const_expr(has_hi):
                        cute.copy(copy_a16, gA16_vec[row_hi, (None, k0 // 8)], x_hi0)
                        cute.copy(copy_a16, gA16_vec[row_hi, (None, k0 // 8 + 1)], x_hi1)
                        a_hi_u32[0] = _bf16x4_to_e4m3x4(x_hi0_u32[0], x_hi0_u32[1], factor)
                        a_hi_u32[1] = _bf16x4_to_e4m3x4(x_hi0_u32[2], x_hi0_u32[3], factor)
                        a_hi_u32[2] = _bf16x4_to_e4m3x4(x_hi1_u32[0], x_hi1_u32[1], factor)
                        a_hi_u32[3] = _bf16x4_to_e4m3x4(x_hi1_u32[2], x_hi1_u32[3], factor)
                else:
                    cute.copy(copy_a8, gA8_vec[row_lo, (None, k0 // 16)], a_lo)
                    if const_expr(has_hi):
                        cute.copy(copy_a8, gA8_vec[row_hi, (None, k0 // 16)], a_hi)
                for nt in cutlass.range_constexpr(n_tiles):
                    w = cute.recast_tensor(w8[nt * chunks + c, None], Uint32)
                    c0 = acc[nt * row_tiles + t, 0]
                    c1 = acc[nt * row_tiles + t, 1]
                    c2 = acc[nt * row_tiles + t, 2]
                    c3 = acc[nt * row_tiles + t, 3]
                    for j in cutlass.range_constexpr(2):
                        c0, c1, c2, c3 = _mma_e4m3_m16n8k32(
                            a_lo_u32[2 * j],
                            a_hi_u32[2 * j],
                            a_lo_u32[2 * j + 1],
                            a_hi_u32[2 * j + 1],
                            w[2 * j],
                            w[2 * j + 1],
                            c0,
                            c1,
                            c2,
                            c3,
                        )
                    acc[nt * row_tiles + t, 0] = c0
                    acc[nt * row_tiles + t, 1] = c1
                    acc[nt * row_tiles + t, 2] = c2
                    acc[nt * row_tiles + t, 3] = c3

    @cute.jit
    def _mainloop_fp8_swap(
        self,
        acc: cute.Tensor,
        gA16: cute.Tensor,
        gA8: cute.Tensor,
        w8: cute.Tensor,
        factor: Float32,
        copy_a16: cute.CopyAtom,
        copy_a8: cute.CopyAtom,
        k_warp0,
        r,
        q,
        n_tiles: cutlass.Constexpr,
    ) -> None:
        """Swapped E4M3 ``m16n8k32`` main loop (num_rows <= 8): A = the weights of tiles 2p (rows 0..7) and
        2p + 1 (rows 8..15), B = activation rows 0..7 (lane group r holds row r)."""
        chunks: cutlass.Constexpr = self.chunks_per_warp
        quant: cutlass.Constexpr = self.fp8_act == "quant"
        a_row = cute.make_rmem_tensor(cute.make_layout((16,)), Uint8)
        a_row_u32 = cute.recast_tensor(a_row, Uint32)
        x0 = cute.make_rmem_tensor(cute.make_layout((8,)), BFloat16)
        x1 = cute.make_rmem_tensor(cute.make_layout((8,)), BFloat16)
        x0_u32 = cute.recast_tensor(x0, Uint32)
        x1_u32 = cute.recast_tensor(x1, Uint32)
        gA16_vec = cute.logical_divide(gA16, (None, 8))
        gA8_vec = cute.logical_divide(gA8, (None, 16))
        row = self._clamp_row(r, 8)
        for c in cutlass.range_constexpr(chunks):
            k0 = k_warp0 + c * CHUNK + 16 * q
            if const_expr(quant):
                cute.copy(copy_a16, gA16_vec[row, (None, k0 // 8)], x0)
                cute.copy(copy_a16, gA16_vec[row, (None, k0 // 8 + 1)], x1)
                a_row_u32[0] = _bf16x4_to_e4m3x4(x0_u32[0], x0_u32[1], factor)
                a_row_u32[1] = _bf16x4_to_e4m3x4(x0_u32[2], x0_u32[3], factor)
                a_row_u32[2] = _bf16x4_to_e4m3x4(x1_u32[0], x1_u32[1], factor)
                a_row_u32[3] = _bf16x4_to_e4m3x4(x1_u32[2], x1_u32[3], factor)
            else:
                cute.copy(copy_a8, gA8_vec[row, (None, k0 // 16)], a_row)
            for pair in cutlass.range_constexpr(n_tiles // 2):
                w_lo = cute.recast_tensor(w8[(2 * pair) * chunks + c, None], Uint32)
                w_hi = cute.recast_tensor(w8[(2 * pair + 1) * chunks + c, None], Uint32)
                c0 = acc[pair, 0]
                c1 = acc[pair, 1]
                c2 = acc[pair, 2]
                c3 = acc[pair, 3]
                for j in cutlass.range_constexpr(2):
                    c0, c1, c2, c3 = _mma_e4m3_m16n8k32(
                        w_lo[2 * j],
                        w_hi[2 * j],
                        w_lo[2 * j + 1],
                        w_hi[2 * j + 1],
                        a_row_u32[2 * j],
                        a_row_u32[2 * j + 1],
                        c0,
                        c1,
                        c2,
                        c3,
                    )
                acc[pair, 0] = c0
                acc[pair, 1] = c1
                acc[pair, 2] = c2
                acc[pair, 3] = c3

    @cute.jit
    def _mainloop_bf16_swap(
        self,
        acc: cute.Tensor,
        gA16: cute.Tensor,
        w8: cute.Tensor,
        w16: cute.Tensor,
        copy_a16: cute.CopyAtom,
        k_warp0,
        r,
        q,
        n_tiles: cutlass.Constexpr,
        fp8_weights: cutlass.Constexpr,
    ) -> None:
        """Swapped BF16 ``m16n8k16`` main loop (num_rows <= 8; see ``_mainloop_fp8_swap``)."""
        chunks: cutlass.Constexpr = self.chunks_per_warp
        a0 = cute.make_rmem_tensor(cute.make_layout((8,)), BFloat16)
        a1 = cute.make_rmem_tensor(cute.make_layout((8,)), BFloat16)
        a0_u32 = cute.recast_tensor(a0, Uint32)
        a1_u32 = cute.recast_tensor(a1, Uint32)
        w_lo = cute.make_rmem_tensor(cute.make_layout((8,)), Uint32)
        w_hi = cute.make_rmem_tensor(cute.make_layout((8,)), Uint32)
        gA16_vec = cute.logical_divide(gA16, (None, 8))
        row = self._clamp_row(r, 8)
        for c in cutlass.range_constexpr(chunks):
            k0 = k_warp0 + c * CHUNK + 16 * q
            cute.copy(copy_a16, gA16_vec[row, (None, k0 // 8)], a0)
            cute.copy(copy_a16, gA16_vec[row, (None, k0 // 8 + 1)], a1)
            for pair in cutlass.range_constexpr(n_tiles // 2):
                self._weights_bf16(w_lo, w8, w16, (2 * pair) * chunks + c, fp8_weights)
                self._weights_bf16(w_hi, w8, w16, (2 * pair + 1) * chunks + c, fp8_weights)
                c0 = acc[pair, 0]
                c1 = acc[pair, 1]
                c2 = acc[pair, 2]
                c3 = acc[pair, 3]
                for j in cutlass.range_constexpr(4):
                    if const_expr(j < 2):
                        c0, c1, c2, c3 = _mma_bf16_m16n8k16(
                            w_lo[2 * j],
                            w_hi[2 * j],
                            w_lo[2 * j + 1],
                            w_hi[2 * j + 1],
                            a0_u32[2 * j],
                            a0_u32[2 * j + 1],
                            c0,
                            c1,
                            c2,
                            c3,
                        )
                    else:
                        c0, c1, c2, c3 = _mma_bf16_m16n8k16(
                            w_lo[2 * j],
                            w_hi[2 * j],
                            w_lo[2 * j + 1],
                            w_hi[2 * j + 1],
                            a1_u32[2 * j - 4],
                            a1_u32[2 * j - 3],
                            c0,
                            c1,
                            c2,
                            c3,
                        )
                acc[pair, 0] = c0
                acc[pair, 1] = c1
                acc[pair, 2] = c2
                acc[pair, 3] = c3

    @cute.jit
    def _mainloop_bf16(
        self,
        acc: cute.Tensor,
        gA16: cute.Tensor,
        w8: cute.Tensor,
        w16: cute.Tensor,
        copy_a16: cute.CopyAtom,
        k_warp0,
        r,
        q,
        n_tiles: cutlass.Constexpr,
        fp8_weights: cutlass.Constexpr,
    ) -> None:
        """BF16 ``m16n8k16`` main loop: BF16 activations times BF16 weights (or E4M3 weights converted
        to BF16 exactly)."""
        num_rows: cutlass.Constexpr = self.num_rows
        row_tiles: cutlass.Constexpr = self.row_tiles
        chunks: cutlass.Constexpr = self.chunks_per_warp

        a_lo0 = cute.make_rmem_tensor(cute.make_layout((8,)), BFloat16)
        a_lo1 = cute.make_rmem_tensor(cute.make_layout((8,)), BFloat16)
        a_hi0 = cute.make_rmem_tensor(cute.make_layout((8,)), BFloat16)
        a_hi1 = cute.make_rmem_tensor(cute.make_layout((8,)), BFloat16)
        a_lo0_u32 = cute.recast_tensor(a_lo0, Uint32)
        a_lo1_u32 = cute.recast_tensor(a_lo1, Uint32)
        a_hi0_u32 = cute.recast_tensor(a_hi0, Uint32)
        a_hi1_u32 = cute.recast_tensor(a_hi1, Uint32)
        w_bf16 = cute.make_rmem_tensor(cute.make_layout((8,)), Uint32)
        gA16_vec = cute.logical_divide(gA16, (None, 8))

        for c in cutlass.range_constexpr(chunks):
            k0 = k_warp0 + c * CHUNK + 16 * q
            if const_expr(n_tiles == 1):
                self._weights_bf16(w_bf16, w8, w16, c, fp8_weights)
            for t in cutlass.range_constexpr(row_tiles):
                has_hi: cutlass.Constexpr = 16 * t + 8 < num_rows
                # Rows past num_rows load the last row instead (no divergent branch); never stored.
                row_lo = self._clamp_row(16 * t + r, 16 * t + 8)
                row_hi = self._clamp_row(16 * t + r + 8, 16 * t + 16)
                cute.copy(copy_a16, gA16_vec[row_lo, (None, k0 // 8)], a_lo0)
                cute.copy(copy_a16, gA16_vec[row_lo, (None, k0 // 8 + 1)], a_lo1)
                if const_expr(has_hi):
                    cute.copy(copy_a16, gA16_vec[row_hi, (None, k0 // 8)], a_hi0)
                    cute.copy(copy_a16, gA16_vec[row_hi, (None, k0 // 8 + 1)], a_hi1)
                else:
                    a_hi0_u32.fill(0)
                    a_hi1_u32.fill(0)
                for nt in cutlass.range_constexpr(n_tiles):
                    if const_expr(n_tiles > 1):
                        # Per row tile so that one tile's BF16 weights are live at a time.
                        self._weights_bf16(w_bf16, w8, w16, nt * chunks + c, fp8_weights)
                    c0 = acc[nt * row_tiles + t, 0]
                    c1 = acc[nt * row_tiles + t, 1]
                    c2 = acc[nt * row_tiles + t, 2]
                    c3 = acc[nt * row_tiles + t, 3]
                    for j in cutlass.range_constexpr(4):
                        # A: a0 = (row_lo, 4j..4j+1), a1 = (row_hi, same), a2/a3 = 4j+2..4j+3.
                        if const_expr(j < 2):
                            c0, c1, c2, c3 = _mma_bf16_m16n8k16(
                                a_lo0_u32[2 * j],
                                a_hi0_u32[2 * j],
                                a_lo0_u32[2 * j + 1],
                                a_hi0_u32[2 * j + 1],
                                w_bf16[2 * j],
                                w_bf16[2 * j + 1],
                                c0,
                                c1,
                                c2,
                                c3,
                            )
                        else:
                            c0, c1, c2, c3 = _mma_bf16_m16n8k16(
                                a_lo1_u32[2 * j - 4],
                                a_hi1_u32[2 * j - 4],
                                a_lo1_u32[2 * j - 3],
                                a_hi1_u32[2 * j - 3],
                                w_bf16[2 * j],
                                w_bf16[2 * j + 1],
                                c0,
                                c1,
                                c2,
                                c3,
                            )
                    acc[nt * row_tiles + t, 0] = c0
                    acc[nt * row_tiles + t, 1] = c1
                    acc[nt * row_tiles + t, 2] = c2
                    acc[nt * row_tiles + t, 3] = c3

    @cute.jit
    def _clamp_row(self, row, rows_needed: cutlass.Constexpr):
        """``row`` when every row the lane quad group touches (< ``rows_needed``) exists, else
        ``min(row, num_rows - 1)``."""
        result = row
        if const_expr(rows_needed > self.num_rows):
            result = cutlass.min(row, Int32(self.num_rows - 1))
        return result

    @cute.jit
    def _weights_bf16(
        self,
        w_bf16: cute.Tensor,
        w8: cute.Tensor,
        w16: cute.Tensor,
        slot: cutlass.Constexpr,
        fp8_weights: cutlass.Constexpr,
    ) -> None:
        """The 16 weights of register slot ``slot`` as 8 bf16x2 words (E4M3 converted exactly)."""
        if const_expr(fp8_weights):
            raw = cute.recast_tensor(w8[slot, None], Uint32)
            for i in cutlass.range_constexpr(4):
                lo, hi = _e4m3x4_to_bf16x4(raw[i])
                w_bf16[2 * i] = lo
                w_bf16[2 * i + 1] = hi
        else:
            first = cute.recast_tensor(w16[2 * slot, None], Uint32)
            second = cute.recast_tensor(w16[2 * slot + 1, None], Uint32)
            for i in cutlass.range_constexpr(4):
                w_bf16[i] = first[i]
                w_bf16[4 + i] = second[i]

    @cute.jit
    def _swiglu(self, gate: Float32, up: Float32, alpha: Float32) -> Float32:
        """``silu(gate) * up`` on the scaled sums, each rounded to BF16 first like the unfused path's
        GEMM output before its SwiGLU kernel."""
        gate = (gate * alpha).to(BFloat16).to(Float32)
        up = (up * alpha).to(BFloat16).to(Float32)
        return gate / (Float32(1.0) + cute.math.exp(-gate, fastmath=False)) * up

    @cute.jit
    def _cluster_sum(self, part: cute.Tensor, row, col) -> Float32:
        """``sum_p part_p[row, col]`` over the cluster's CTAs p = 0 .. ksplit - 1, in that order."""
        offset = cute.crd2idx((row, col), part.layout) * 4
        total = Float32(0.0)
        for p in cutlass.range_constexpr(self.ksplit):
            total = total + _ld_shared_cluster_f32(_mapa_shared_cluster(part.iterator, p) + offset)
        return total

    @cute.jit
    def _finish(
        self, gC: cute.Tensor, gC2: cute.Tensor, row, out_col, value: Float32, out_factor: Float32
    ) -> None:
        if const_expr(self.n_valid < self.fp8_out + self.bf16_out):
            if out_col < self.n_valid:
                self._store(gC, gC2, row, out_col, value, out_factor)
        else:
            self._store(gC, gC2, row, out_col, value, out_factor)

    @cute.jit
    def _store(
        self, gC: cute.Tensor, gC2: cute.Tensor, row, n, value: Float32, out_factor: Float32
    ) -> None:
        if const_expr(self.split_out == 0):
            self._put_c(gC, row, n, value, out_factor)
        else:
            if n < self.split_out:
                self._put_c(gC, row, n, value, out_factor)
            else:
                gC2[row, n - self.split_out] = value.to(gC2.element_type)

    @cute.jit
    def _put_c(self, gC: cute.Tensor, row, n, value: Float32, out_factor: Float32) -> None:
        if const_expr(self.quant_out):
            gC[row, n] = _f32_to_e4m3(value * out_factor).to(Uint8)
        else:
            gC[row, n] = value.to(gC.element_type)
