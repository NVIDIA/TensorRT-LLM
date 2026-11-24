# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Warp-MMA GEMM for decode-sized M over MXFP8 weights (BF16 activations, no activation quant).

``C[M, N] = A[M, K] (bf16) @ dequant(B[N, K] e4m3, S[N, K/32] ue8m0).T``

Layout of the work:

* A CTA owns ``n_tiles`` groups of 8 output columns (one ``m16n8`` tile column each) and splits
  K over its ``k_warps`` warps; warps reduce through shared memory at the end. Every output is
  summed in the same order whatever ``n_tiles`` is.
* K is processed in 64-element chunks. Inside a chunk the reduction order is
  permuted so that lane ``(r, q) = (lane // 4, lane % 4)`` loads one contiguous
  16-byte weight vector ``B[n0 + r, k + 16q : k + 16q + 16]`` and one 32-byte
  activation vector per row, and those registers *are* the fragments of four
  ``mma.sync.m16n8k16`` steps (logical k ``2q + t`` and ``2q + 8 + t`` of step
  ``j`` map to physical ``16q + 4j + t`` and ``16q + 4j + 2 + t``; the same
  bijection is used for A and B, so the dot product is unchanged).
* E4M3 weights times their power-of-two UE8M0 scale convert exactly to BF16 (with
  ``bf16_dequant`` by ``cvt.rn.bf16x2.e4m3x2`` and a BF16 ``mul``: the same bits).
* ``swap_ab`` (<= 8 rows, tiles in pairs): a tile pair's weights are the MMA's A operand and the
  tokens its B operand; half the MMAs, the same products at the same k positions.
* All weight vectors and scales of a warp are loaded before
  ``griddepcontrol.wait`` (programmatic dependent launch), activations after.
* With ``a_rms_norm`` the rows of A are RMS-normalized on the fly:
  ``rms(a)^-1 * (a * gamma) @ W.T`` multiplies A by ``gamma`` as it is loaded, accumulates the
  rows' sums of squares beside the MMA, and scales each output row by its inverse RMS.
* With ``index_pack`` a CTA's 32 columns are one group of a 128-wide head, and the BF16 outputs
  are packed as CSA2's cache write packs index queries (E2M1 values, a UE8M0 scale per group).
* With ``out_swiglu_limit`` B is a gated MLP's gate | up weight: a CTA's 8 columns are 4 gate
  columns and their up partners ``N/2`` further on, and ``C[M, N/2]`` is their clamped SwiGLU.
"""

from __future__ import annotations

import cutlass
import cutlass.cute as cute
from cutlass import BFloat16, Float32, Int32, Uint8, Uint32, const_expr
from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm, vector
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass.utils import SmemAllocator

from tensorrt_llm.math_utils import ceil_div

from .cute_ptx_utils import mma_sync
from .utils import griddepcontrol_launch_dependents, griddepcontrol_wait

GROUP = 32  # K elements per UE8M0 scale
CHUNK = 64  # K elements per quad of lanes (16 per lane)
N_TILE = 8
ROPE_DIM = 64  # rotated columns at the end of every head of the RoPE epilogue
PACK_HEAD = 128  # channels of an ``index_pack`` head: 64 data bytes and 4 scale bytes per row
PACK_GROUP = 32  # channels per E2M1 group and UE8M0 scale of ``index_pack``


@dsl_user_op
def _fp8x4_to_bf16x4_scaled(x: Uint32, scale: Float32, *, loc=None, ip=None) -> cute.TensorSSA:
    """Four packed E4M3 times a power-of-two scale -> four BF16 (exact), as two ``Uint32``."""
    out = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32()] * 2),
        [x.ir_value(loc=loc, ip=ip), scale.ir_value(loc=loc, ip=ip)],
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
        "mul.f32 f0, f0, $3;\n\t"
        "mul.f32 f1, f1, $3;\n\t"
        "mul.f32 f2, f2, $3;\n\t"
        "mul.f32 f3, f3, $3;\n\t"
        "cvt.rn.bf16x2.f32 $0, f1, f0;\n\t"
        "cvt.rn.bf16x2.f32 $1, f3, f2;\n\t"
        "}\n",
        "=r,=r,r,f",
        has_side_effects=False,
        is_align_stack=False,
        loc=loc,
        ip=ip,
    )
    vec = vector.from_elements(
        ir.VectorType.get([2], T.i32(), loc=loc),
        [llvm.extractvalue(T.i32(), out, [i], loc=loc, ip=ip) for i in range(2)],
        loc=loc,
        ip=ip,
    )
    return cute.TensorSSA(vec, 2, Uint32)


@dsl_user_op
def _fp8x4_to_bf16x4_s2(x: Uint32, s2: Uint32, *, loc=None, ip=None) -> cute.TensorSSA:
    """``_fp8x4_to_bf16x4_scaled`` with the scale as a BF16 pair ``s2``: exact E4M3 -> BF16, then a
    BF16 multiply. When BF16 holds the scale, both round the exact product once (the same bits)."""
    out = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32()] * 2),
        [x.ir_value(loc=loc, ip=ip), s2.ir_value(loc=loc, ip=ip)],
        "{\n\t"
        ".reg .b16 lo, hi;\n\t"
        "mov.b32 {lo, hi}, $2;\n\t"
        "cvt.rn.bf16x2.e4m3x2 $0, lo;\n\t"
        "cvt.rn.bf16x2.e4m3x2 $1, hi;\n\t"
        "mul.rn.bf16x2 $0, $0, $3;\n\t"
        "mul.rn.bf16x2 $1, $1, $3;\n\t"
        "}\n",
        "=r,=r,r,r",
        has_side_effects=False,
        is_align_stack=False,
        loc=loc,
        ip=ip,
    )
    vec = vector.from_elements(
        ir.VectorType.get([2], T.i32(), loc=loc),
        [llvm.extractvalue(T.i32(), out, [i], loc=loc, ip=ip) for i in range(2)],
        loc=loc,
        ip=ip,
    )
    return cute.TensorSSA(vec, 2, Uint32)


@dsl_user_op
def _bf16x2_scale_exact(scale: Float32, *, loc=None, ip=None) -> cute.TensorSSA:
    """An FP32 scale -> (its BF16 pair, 1 if BF16 holds it exactly or it is NaN, else 0)."""
    out = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32()] * 2),
        [scale.ir_value(loc=loc, ip=ip)],
        "{\n\t"
        ".reg .b32 back;\n\t"
        ".reg .pred eq, nan;\n\t"
        "cvt.rn.bf16x2.f32 $0, $2, $2;\n\t"
        "shl.b32 back, $0, 16;\n\t"
        "setp.eq.f32 eq, back, $2;\n\t"
        "setp.nan.f32 nan, $2, $2;\n\t"
        "or.pred eq, eq, nan;\n\t"
        "selp.u32 $1, 1, 0, eq;\n\t"
        "}\n",
        "=r,=r,f",
        has_side_effects=False,
        is_align_stack=False,
        loc=loc,
        ip=ip,
    )
    vec = vector.from_elements(
        ir.VectorType.get([2], T.i32(), loc=loc),
        [llvm.extractvalue(T.i32(), out, [i], loc=loc, ip=ip) for i in range(2)],
        loc=loc,
        ip=ip,
    )
    return cute.TensorSSA(vec, 2, Uint32)


def _f32_asm(sources, body: str, constraints: str, *, loc=None, ip=None) -> Float32:
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Float32(v).ir_value(loc=loc, ip=ip) for v in sources],
            body,
            constraints,
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _mul_rn(a: Float32, b: Float32, *, loc=None, ip=None) -> Float32:
    """IEEE round-to-nearest multiply that never flushes subnormals."""
    return _f32_asm((a, b), "mul.rn.f32 $0, $1, $2;", "=f,f,f", loc=loc, ip=ip)


@dsl_user_op
def _div_rn(a: Float32, b: Float32, *, loc=None, ip=None) -> Float32:
    return _f32_asm((a, b), "div.rn.f32 $0, $1, $2;", "=f,f,f", loc=loc, ip=ip)


@dsl_user_op
def _max_nan(a: Float32, b: Float32, *, loc=None, ip=None) -> Float32:
    """FP32 maximum that is NaN when either input is NaN."""
    return _f32_asm((a, b), "max.NaN.f32 $0, $1, $2;", "=f,f,f", loc=loc, ip=ip)


@dsl_user_op
def _f32_bits(value: Float32, *, loc=None, ip=None) -> Uint32:
    return Uint32(llvm.bitcast(T.i32(), Float32(value).ir_value(loc=loc, ip=ip)))


@dsl_user_op
def _bits_f32(bits: Uint32, *, loc=None, ip=None) -> Float32:
    return Float32(llvm.bitcast(T.f32(), Uint32(bits).ir_value(loc=loc, ip=ip)))


@dsl_user_op
def _f32x4_to_e2m1x4(v0, v1, v2, v3, *, loc=None, ip=None) -> Uint32:
    """E2M1 nibbles of four FP32 values, ``v0`` lowest."""
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [Float32(v).ir_value(loc=loc, ip=ip) for v in (v0, v1, v2, v3)],
            "{\n.reg .b8 a, b;\n.reg .b16 h;\ncvt.rn.satfinite.e2m1x2.f32 a, $2, $1;\n"
            "cvt.rn.satfinite.e2m1x2.f32 b, $4, $3;\nmov.b16 h, {a, b};\ncvt.u32.u16 $0, h;\n}",
            "=r,f,f,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


def _f32_hex(value: float) -> str:
    """PTX hexadecimal FP32 immediate (``0f...``) of ``value`` rounded to FP32."""
    import struct

    return "0f%08X" % struct.unpack("<I", struct.pack("<f", value))[0]


def _bf16x2_hex(value: float) -> str:
    """``value`` (exactly a BF16 value) as a packed BF16 pair immediate (``0x...``)."""
    import struct

    bits = struct.unpack("<I", struct.pack("<f", value))[0]
    if bits & 0xFFFF:
        raise ValueError(f"{value} is not exactly representable in BF16")
    return "0x%04X%04X" % (bits >> 16, bits >> 16)


@dsl_user_op
def _swiglu_bf16x2(
    gate: Uint32, up: Uint32, limit_f32: str, limit_bf16x2: str, *, loc=None, ip=None
) -> Uint32:
    """Two packed BF16 gate / up pairs -> their packed BF16 clamped SwiGLU, in the instructions
    ``silu_and_mul_kernel`` (``modules/swiglu.py``, with a limit) compiles to, so the bits match."""
    out = llvm.inline_asm(
        T.i32(),
        [gate.ir_value(loc=loc, ip=ip), up.ir_value(loc=loc, ip=ip)],
        "{\n\t"
        ".reg .b16 g0, g1, u0, u1;\n\t"
        ".reg .b32 lim, gm;\n\t"
        ".reg .f32 a0, a1, b0, b1, t0, t1, s0, s1, one;\n\t"
        f"mov.b32 lim, {limit_bf16x2};\n\t"
        "min.bf16x2 gm, $1, lim;\n\t"
        "mov.b32 {g0, g1}, gm;\n\t"
        "mov.b32 {u0, u1}, $2;\n\t"
        "cvt.f32.bf16 a0, g0;\n\t"
        "cvt.f32.bf16 a1, g1;\n\t"
        "cvt.f32.bf16 b0, u0;\n\t"
        "cvt.f32.bf16 b1, u1;\n\t"
        f"min.xorsign.abs.f32 b0, b0, {limit_f32};\n\t"
        f"min.xorsign.abs.f32 b1, b1, {limit_f32};\n\t"
        "mul.f32 t0, a0, 0fBFB8AA3B;\n\t"
        "mul.f32 t1, a1, 0fBFB8AA3B;\n\t"
        "ex2.approx.f32 t0, t0;\n\t"
        "ex2.approx.f32 t1, t1;\n\t"
        "add.f32 t0, t0, 0f3F800000;\n\t"
        "add.f32 t1, t1, 0f3F800000;\n\t"
        "mov.f32 one, 0f3F800000;\n\t"
        "div.full.f32 s0, one, t0;\n\t"
        "div.full.f32 s1, one, t1;\n\t"
        "mul.f32 s0, a0, s0;\n\t"
        "mul.f32 s1, a1, s1;\n\t"
        "add.f32 b0, b0, 0f00000000;\n\t"
        "add.f32 b1, b1, 0f00000000;\n\t"
        "mul.f32 s0, b0, s0;\n\t"
        "mul.f32 s1, b1, s1;\n\t"
        "cvt.rn.bf16x2.f32 $0, s1, s0;\n\t"
        "}\n",
        "=r,r,r",
        has_side_effects=False,
        is_align_stack=False,
        loc=loc,
        ip=ip,
    )
    return Uint32(out)


class LowMFp8MmaGemmKernel:
    """``num_rows`` (<= 64) x K x N MXFP8 GEMM, ``n_tiles`` 8-column tiles per CTA."""

    def __init__(
        self,
        *,
        num_rows: int,
        k_extent: int,
        k_warps: int,
        use_pdl: bool,
        split_n: int = 0,
        scale_block128: bool = False,
        group_n: int = 0,
        rope_head_dim: int = 0,
        scale_swizzled: bool = False,
        bf16_weights: bool = False,
        a_rope_head_dim: int = 0,
        a_rms_norm: bool = False,
        a_rms_eps: float = 0.0,
        n_tiles: int = 1,
        bf16_dequant: bool = False,
        swap_ab: bool = False,
        index_pack: bool = False,
        out_swiglu_limit: float = 0.0,
        release_after_wait: bool = False,
    ) -> None:
        """``scale_block128``: S is FP32 ``[N/128, K/128]`` power-of-two multipliers instead of
        UE8M0 ``[N, K/32]``. ``group_n``: B stacks groups of ``group_n`` rows, and group ``g``
        multiplies columns ``[g*K, (g+1)*K)`` of A (a grouped GEMM over ``A[M, G*K]``).
        ``rope_head_dim``: apply interleaved RoPE in FP32 to the last ``ROPE_DIM`` columns of
        every ``rope_head_dim``-wide head before the store (cos/sin table ``[P, 2, ROPE_DIM/2]``
        indexed by the row's position), like ``mla_rope_inplace`` on the output.
        ``scale_swizzled``: S is the UE8M0 ``[N, K/32]`` scales in the CUTLASS 128x4 swizzle (1-D),
        as an MXFP8 ``Linear`` stores them. ``bf16_weights``: B is BF16 ``[N, K]`` (as bytes
        ``[N, 2K]``) without scales. ``a_rope_head_dim`` (grouped only): rotate the last
        ``ROPE_DIM`` columns of every ``a_rope_head_dim``-wide head of A back by their angle as A
        is loaded (interleaved inverse RoPE in FP32, rounded to BF16, as
        ``mla_rope_inplace(inverse=True)``). ``a_rms_norm``: RMS-normalize the rows of A over K
        with the BF16 ``gamma`` [1, K] and ``a_rms_eps`` (not with groups or A RoPE). ``n_tiles``:
        8-column tiles per CTA (N a multiple of ``8 * n_tiles``); fewer CTAs load A fewer times and
        fit a large N in one wave. ``bf16_dequant``: FP8 weights times the BF16 scale pair (with
        ``scale_block128`` only in warps whose scales BF16 holds). ``swap_ab``: at <= 8 rows with an
        even ``n_tiles``, the weights are the MMA's A operand. ``index_pack`` (four tiles): C is the
        packed E2M1 data ``[M, N/2]`` and C2 the UE8M0 scales ``[M, N/32]`` of the BF16 outputs, bit
        for bit CSA2's cache-write packing of ``[M, N/128, 128]`` index queries.
        ``out_swiglu_limit`` = L > 0 (a BF16 value; one tile per CTA, plain FP8 B): B is a ``[N, K]``
        gate | up weight and C ``[M, N/2]`` its clamped SwiGLU, bit for bit the plain GEMM then
        ``silu_and_mul_kernel``. ``release_after_wait``: the dependent grid is released after this
        grid's dependency wait instead of before it."""
        if k_extent % (CHUNK * k_warps):
            raise ValueError(f"K={k_extent} must be a multiple of {CHUNK * k_warps}")
        if not 1 <= num_rows <= 64:
            raise ValueError(f"num_rows={num_rows} must be in [1, 64]")
        if split_n % N_TILE:
            raise ValueError(f"split_n={split_n} must be a multiple of {N_TILE}")
        if n_tiles not in (1, 2, 4):
            raise ValueError(f"n_tiles={n_tiles} must be 1, 2 or 4")
        if index_pack and (
            N_TILE * n_tiles != PACK_GROUP
            or split_n
            or group_n
            or bf16_weights
            or a_rms_norm
            or rope_head_dim not in (0, PACK_HEAD)
        ):
            raise ValueError("index_pack needs four tiles of an ungrouped FP8 GEMM")
        self.index_pack = index_pack
        self.n_tiles = n_tiles
        self.num_rows = num_rows
        self.row_tiles = ceil_div(num_rows, 16)
        self.k_extent = k_extent
        self.k_warps = k_warps
        self.chunks_per_warp = k_extent // (CHUNK * k_warps)
        self.use_pdl = use_pdl
        self.split_n = split_n
        self.scale_block128 = scale_block128
        self.group_n = group_n
        self.rope_head_dim = rope_head_dim
        if rope_head_dim and (rope_head_dim % N_TILE or ROPE_DIM > rope_head_dim):
            raise ValueError(f"unsupported rope head dim {rope_head_dim}")
        if group_n % (N_TILE * n_tiles):
            raise ValueError(f"group_n={group_n} must be a multiple of {N_TILE * n_tiles}")
        self.scale_swizzled = scale_swizzled
        self.sf_k_tiles = ceil_div(k_extent // GROUP, 4)
        self.bf16_weights = bf16_weights
        # 16-byte weight vectors per lane and K chunk.
        self.weight_vectors = 2 if bf16_weights else 1
        self.a_rope_head_dim = a_rope_head_dim
        # Each warp's K range must hold whole heads, so a head's last chunk is a warp's own.
        if a_rope_head_dim and (
            not group_n
            or rope_head_dim
            or a_rope_head_dim % CHUNK
            or (k_extent // k_warps) % a_rope_head_dim
        ):
            raise ValueError(f"unsupported A RoPE geometry: head {a_rope_head_dim}, K {k_extent}")
        if a_rms_norm and (group_n or a_rope_head_dim):
            raise ValueError("the A RMS norm needs ungrouped A without RoPE")
        self.a_rms_norm = a_rms_norm
        self.a_rms_eps = a_rms_eps
        self.bf16_dequant = bf16_dequant and not bf16_weights
        self.dequant_guard = self.bf16_dequant and scale_block128
        self.swap_ab = swap_ab and num_rows <= 8 and n_tiles % 2 == 0
        self.out_swiglu = out_swiglu_limit > 0.0
        if self.out_swiglu and (
            n_tiles != 1
            or split_n
            or group_n
            or rope_head_dim
            or a_rope_head_dim
            or a_rms_norm
            or bf16_weights
            or index_pack
        ):
            raise ValueError("the SwiGLU output needs one tile per CTA of a plain FP8 GEMM")
        self.out_swiglu_limit_f32 = _f32_hex(out_swiglu_limit) if self.out_swiglu else ""
        self.out_swiglu_limit_bf16x2 = _bf16x2_hex(out_swiglu_limit) if self.out_swiglu else ""
        self.release_after_wait = release_after_wait

    @cute.jit
    def __call__(
        self,
        gA: cute.Tensor,
        gB: cute.Tensor,
        gS: cute.Tensor,
        gC: cute.Tensor,
        gC2: cute.Tensor,
        gPos: cute.Tensor,
        gCosSin: cute.Tensor,
        gGamma: cute.Tensor,
        stream,
    ) -> None:
        n = cute.size(gB, mode=[0])
        copy_a = cute.make_copy_atom(
            cute.nvgpu.CopyG2ROp(),
            BFloat16,
            num_bits_per_copy=128,
            load_cache_mode=cute.nvgpu.LoadCacheMode.ALWAYS,
        )
        copy_b = cute.make_copy_atom(
            cute.nvgpu.CopyG2ROp(),
            Uint8,
            num_bits_per_copy=128,
            load_cache_mode=cute.nvgpu.LoadCacheMode.STREAMING,
        )
        self.kernel(gA, gB, gS, gC, gC2, gPos, gCosSin, gGamma, copy_a, copy_b).launch(
            grid=[cute.ceil_div(n, N_TILE * self.n_tiles), 1, 1],
            block=[32 * self.k_warps, 1, 1],
            smem=self.k_warps * self.row_tiles * 16 * (N_TILE * self.n_tiles + 1) * 4 + 16,
            stream=stream,
            use_pdl=self.use_pdl,
            min_blocks_per_mp=1,
        )

    @cute.kernel
    def kernel(
        self,
        gA: cute.Tensor,
        gB: cute.Tensor,
        gS: cute.Tensor,
        gC: cute.Tensor,
        gC2: cute.Tensor,
        gPos: cute.Tensor,
        gCosSin: cute.Tensor,
        gGamma: cute.Tensor,
        copy_a: cute.CopyAtom,
        copy_b: cute.CopyAtom,
    ) -> None:
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        warp = tidx // 32
        lane = tidx % 32
        r = lane // 4
        q = lane % 4

        num_rows: cutlass.Constexpr = self.num_rows
        row_tiles: cutlass.Constexpr = self.row_tiles
        k_warps: cutlass.Constexpr = self.k_warps
        chunks: cutlass.Constexpr = self.chunks_per_warp
        n_tiles: cutlass.Constexpr = self.n_tiles
        cols: cutlass.Constexpr = N_TILE * self.n_tiles

        n0 = bidx * cols
        k_warp0 = warp * (chunks * CHUNK)

        # ---- weights + scales of every tile (independent of the producer kernel) ----
        gB_vec = cute.logical_divide(gB, (None, 16))
        vectors: cutlass.Constexpr = self.weight_vectors
        b_regs = cute.make_rmem_tensor(
            cute.make_layout((n_tiles * chunks, 16 * vectors), stride=(16 * vectors, 1)), Uint8
        )
        s_bits = cute.make_rmem_tensor(cute.make_layout((n_tiles * chunks,)), Int32)
        s_regs = cute.recast_tensor(s_bits, Float32)
        b_vecs = cute.logical_divide(b_regs, (None, 16))
        for nt in cutlass.range_constexpr(n_tiles):
            b_row = n0 + N_TILE * nt + r
            if const_expr(self.out_swiglu):
                # Lane rows 0-3 load gate rows 4b .. 4b+3, rows 4-7 their up partners N/2 further on.
                pairs: cutlass.Constexpr = N_TILE // 2
                half_n = cute.size(gB, mode=[0]) // 2
                b_row = bidx * pairs + r + (half_n - pairs) * (r // pairs)
            for c in cutlass.range_constexpr(chunks):
                slot = nt * chunks + c
                k0 = k_warp0 + c * CHUNK + 16 * q
                if const_expr(self.bf16_weights):
                    for v in cutlass.range_constexpr(vectors):
                        cute.copy(
                            copy_b, gB_vec[b_row, (None, k0 // 8 + v)], b_vecs[slot, (None, v)]
                        )
                else:
                    cute.copy(copy_b, gB_vec[b_row, (None, k0 // 16)], b_regs[slot, None])
                if const_expr(self.bf16_weights):
                    pass
                elif const_expr(self.scale_block128):
                    s_regs[slot] = gS[b_row // 128, k0 // 128]
                elif const_expr(self.scale_swizzled):
                    # 128-row tiles of [K/128 blocks, 32 rows, 4 row groups, 4 scales].
                    block = k0 // GROUP
                    offset = (
                        (b_row // 128) * (self.sf_k_tiles * 512)
                        + (b_row % 32) * 16
                        + (b_row % 128) // 32 * 4
                        + block // 4 * 512
                        + block % 4
                    )
                    s_bits[slot] = gS[offset].to(Int32) << 23
                else:
                    # UE8M0 -> the bit pattern of 2^(s - 127).
                    s_bits[slot] = gS[b_row, k0 // GROUP].to(Int32) << 23

        # BF16 scale pairs: a UE8M0 scale's FP32 bits shifted by 16 are its BF16 bits; FP32 scales
        # round to BF16, and the warp keeps the BF16 conversion only if all of them are exact.
        s2 = cute.make_rmem_tensor(cute.make_layout((n_tiles * chunks,)), Uint32)
        s2_exact = Int32(1)
        if const_expr(self.bf16_dequant):
            for i in cutlass.range_constexpr(n_tiles * chunks):
                if const_expr(self.scale_block128):
                    pair = _bf16x2_scale_exact(s_regs[i])
                    s2[i] = pair[0]
                    s2_exact = s2_exact & Int32(pair[1])
                else:
                    s2[i] = Uint32(s_bits[i]) | (Uint32(s_bits[i]) >> 16)

        # The norm's gamma for this lane's K columns: 16 per chunk, as A.
        g_regs = cute.make_rmem_tensor(cute.make_layout((chunks, 16), stride=(16, 1)), BFloat16)
        if const_expr(self.a_rms_norm):
            gGamma_vec = cute.logical_divide(gGamma, (None, 8))
            g_vecs = cute.logical_divide(g_regs, (None, 8))
            for c in cutlass.range_constexpr(chunks):
                k0 = k_warp0 + c * CHUNK + 16 * q
                cute.copy(copy_a, gGamma_vec[0, (None, k0 // 8)], g_vecs[c, (None, 0)])
                cute.copy(copy_a, gGamma_vec[0, (None, k0 // 8 + 1)], g_vecs[c, (None, 1)])
        # Partial sums of squares of rows 16t + r (0) and 16t + r + 8 (1).
        squares = cute.make_rmem_tensor(cute.make_layout((row_tiles, 2), stride=(2, 1)), Float32)
        squares.fill(0.0)

        if const_expr(self.use_pdl and self.release_after_wait):
            # Released only once this grid's predecessors have completed.
            griddepcontrol_wait()
            griddepcontrol_launch_dependents()
        elif const_expr(self.use_pdl):
            # The dependent grid is released once the weight loads are issued; it still waits for
            # this grid's completion before reading the output.
            griddepcontrol_launch_dependents()
            griddepcontrol_wait()

        # A's inverse RoPE: positions of rows 16t + r and 16t + r + 8.
        rope_pos = cute.make_rmem_tensor(
            cute.make_layout((self.row_tiles, 2), stride=(2, 1)), Int32
        )
        if const_expr(self.a_rope_head_dim > 0):
            rope_pos.fill(0)
            for t in cutlass.range_constexpr(self.row_tiles):
                for h in cutlass.range_constexpr(2):
                    if 16 * t + r + 8 * h < self.num_rows:
                        rope_pos[t, h] = gPos[16 * t + r + 8 * h]

        gA_vec = cute.logical_divide(gA, (None, 8))
        acc = cute.make_rmem_tensor(
            cute.make_layout((n_tiles * row_tiles, 4), stride=(4, 1)), Float32
        )
        acc.fill(0.0)
        a_col0 = 0
        if const_expr(self.group_n > 0):
            a_col0 = (n0 // self.group_n) * self.k_extent
        loop_args = (acc, gA_vec, copy_a, gCosSin, b_regs, s_regs, s2, g_regs, squares, rope_pos)
        if const_expr(self.dequant_guard):
            # Warp-uniform: every lane of the warp takes the same main loop.
            if cute.arch.vote_all_sync(s2_exact == 1):
                self._main_loop(*loop_args, k_warp0, a_col0, q, r, True)
            else:
                self._main_loop(*loop_args, k_warp0, a_col0, q, r, False)
        else:
            self._main_loop(*loop_args, k_warp0, a_col0, q, r, self.bf16_dequant)

        # ---- cross-warp (split-K) reduction through shared memory ----
        smem = SmemAllocator()
        red = smem.allocate_tensor(
            Float32,
            cute.make_layout(
                (k_warps, row_tiles * 16, cols),
                stride=(row_tiles * 16 * cols, cols, 1),
            ),
            byte_alignment=16,
        )
        row_squares = smem.allocate_tensor(
            Float32,
            cute.make_layout((k_warps, row_tiles * 16), stride=(row_tiles * 16, 1)),
            byte_alignment=16,
        )
        for t in cutlass.range_constexpr(row_tiles):
            if const_expr(self.swap_ab):
                # D = (16 columns of tile pair p) x (8 tokens): lane (r, q) holds columns 16p + r
                # and 16p + 8 + r of tokens 2q and 2q + 1.
                for tp in cutlass.range_constexpr(n_tiles // 2):
                    col0 = 2 * N_TILE * tp + r
                    red[warp, 2 * q, col0] = acc[tp, 0]
                    red[warp, 2 * q + 1, col0] = acc[tp, 1]
                    red[warp, 2 * q, col0 + N_TILE] = acc[tp, 2]
                    red[warp, 2 * q + 1, col0 + N_TILE] = acc[tp, 3]
            else:
                for nt in cutlass.range_constexpr(n_tiles):
                    c0 = N_TILE * nt + 2 * q
                    red[warp, 16 * t + r, c0] = acc[nt * row_tiles + t, 0]
                    red[warp, 16 * t + r, c0 + 1] = acc[nt * row_tiles + t, 1]
                    red[warp, 16 * t + r + 8, c0] = acc[nt * row_tiles + t, 2]
                    red[warp, 16 * t + r + 8, c0 + 1] = acc[nt * row_tiles + t, 3]
            if const_expr(self.a_rms_norm):
                # The four lanes of a row hold its four 16-column slices of every chunk.
                for h in cutlass.range_constexpr(2):
                    row_sum = squares[t, h]
                    row_sum = row_sum + cute.arch.shuffle_sync_bfly(row_sum, offset=1)
                    row_sum = row_sum + cute.arch.shuffle_sync_bfly(row_sum, offset=2)
                    if q == 0:
                        row_squares[warp, 16 * t + r + 8 * h] = row_sum
        cute.arch.sync_threads()
        if const_expr(self.index_pack):
            self._pack_epilogue(red, row_squares, gC, gC2, gPos, gCosSin, n0, tidx)
        elif const_expr(self.out_swiglu):
            self._swiglu_epilogue(red, gC, bidx, tidx)
        else:
            for idx in cutlass.range(tidx, num_rows * cols, 32 * k_warps):
                row = idx // cols
                col = idx % cols
                total = self._output_value(red, row_squares, gPos, gCosSin, n0, row, col)
                self._store(gC, gC2, row, n0 + col, total)

    @cute.jit
    def _swiglu_epilogue(self, red: cute.Tensor, gC: cute.Tensor, bidx, tidx) -> None:
        """Gate column ``j`` < 4 and its up partner ``j + 4``, each summed and rounded to BF16 as the
        plain store does, then their SwiGLU into ``C[row, 4 * bidx + j]``."""
        k_warps: cutlass.Constexpr = self.k_warps
        pairs: cutlass.Constexpr = N_TILE // 2
        gate = cute.make_rmem_tensor(cute.make_layout((2,)), BFloat16)
        up = cute.make_rmem_tensor(cute.make_layout((2,)), BFloat16)
        out = cute.make_rmem_tensor(cute.make_layout((2,)), BFloat16)
        gate_u32 = cute.recast_tensor(gate, Uint32)
        up_u32 = cute.recast_tensor(up, Uint32)
        out_u32 = cute.recast_tensor(out, Uint32)
        for idx in cutlass.range(tidx, self.num_rows * pairs, 32 * k_warps):
            row = idx // pairs
            col = idx % pairs
            g = Float32(0.0)
            u = Float32(0.0)
            for w in cutlass.range_constexpr(k_warps):
                g = g + red[w, row, col]
            for w in cutlass.range_constexpr(k_warps):
                u = u + red[w, row, col + pairs]
            gate_u32[0] = Uint32(0)
            up_u32[0] = Uint32(0)
            gate[0] = g.to(BFloat16)
            up[0] = u.to(BFloat16)
            out_u32[0] = _swiglu_bf16x2(
                gate_u32[0], up_u32[0], self.out_swiglu_limit_f32, self.out_swiglu_limit_bf16x2
            )
            gC[row, bidx * pairs + col] = out[0]

    @cute.jit
    def _output_value(
        self,
        red: cute.Tensor,
        row_squares: cute.Tensor,
        gPos: cute.Tensor,
        gCosSin: cute.Tensor,
        n0,
        row,
        col,
    ) -> Float32:
        """Output ``(row, n0 + col)`` in FP32, before its rounding."""
        k_warps: cutlass.Constexpr = self.k_warps
        total = Float32(0.0)
        for w in cutlass.range_constexpr(k_warps):
            total = total + red[w, row, col]
        inv_rms = Float32(1.0)
        if const_expr(self.a_rms_norm):
            square_sum = Float32(0.0)
            for w in cutlass.range_constexpr(k_warps):
                square_sum = square_sum + row_squares[w, row]
            inv_rms = cute.math.rsqrt(
                square_sum * Float32(1.0 / self.k_extent) + Float32(self.a_rms_eps)
            )
            total = total * inv_rms
        if const_expr(self.rope_head_dim > 0):
            rope_start: cutlass.Constexpr = self.rope_head_dim - ROPE_DIM
            in_head = (n0 + col) % self.rope_head_dim
            if in_head >= rope_start:
                partner = Float32(0.0)
                for w in cutlass.range_constexpr(k_warps):
                    partner = partner + red[w, row, col ^ 1]
                if const_expr(self.a_rms_norm):
                    partner = partner * inv_rms
                rope_pair = (in_head - rope_start) // 2
                pos = gPos[row]
                cos = gCosSin[pos, 0, rope_pair]
                sin = gCosSin[pos, 1, rope_pair]
                if col % 2 == 0:
                    total = total * cos - partner * sin
                else:
                    total = total * cos + partner * sin
        return total

    @cute.jit
    def _pack_epilogue(
        self,
        red: cute.Tensor,
        row_squares: cute.Tensor,
        gData: cute.Tensor,
        gScale: cute.Tensor,
        gPos: cute.Tensor,
        gCosSin: cute.Tensor,
        n0,
        tidx,
    ) -> None:
        """Pack the CTA's group of every row as ``QuantizeScatterKernel``'s ``index`` split job."""
        num_rows: cutlass.Constexpr = self.num_rows
        rows_per_pass: cutlass.Constexpr = 4 * self.k_warps
        head = n0 // PACK_HEAD
        group = (n0 % PACK_HEAD) // PACK_GROUP
        lane8 = tidx % 8
        x = cute.make_rmem_tensor(cute.make_layout((4,)), Float32)
        for row0 in cutlass.range_constexpr(0, num_rows, rows_per_pass):
            row = row0 + tidx // 8
            # Lanes past the last row repeat it (so shuffles see the whole warp) and store nothing.
            source = row
            if source >= num_rows:
                source = num_rows - 1
            local_max = Float32(0.0)
            for j in cutlass.range_constexpr(4):
                total = self._output_value(
                    red, row_squares, gPos, gCosSin, n0, source, 4 * lane8 + j
                )
                x[j] = total.to(BFloat16).to(Float32)
                local_max = _max_nan(local_max, cute.arch.fmax(x[j], -x[j]))
            for offset in (1, 2, 4):
                local_max = _max_nan(
                    local_max, cute.arch.shuffle_sync_bfly(local_max, offset=offset)
                )
            group_nan = local_max != local_max
            clamped = local_max
            if clamped < Float32(6.0 * 2.0**-126):
                clamped = Float32(6.0 * 2.0**-126)
            # ceil(log2(amax / 6)), +inf as 128.
            bits = _f32_bits(_div_rn(clamped, Float32(6.0)))
            exponent = Int32((bits >> 23) & 255) - 127
            if (bits & Uint32(0x7FFFFF)) != 0:
                exponent = exponent + 1
            if exponent > 128:
                exponent = Int32(128)
            scale_byte = (exponent + 127) & 255
            # 2^-exponent as the UE8M0 byte's value; byte zero is the FP32 subnormal 2^-127.
            inv_byte = (127 - exponent) & 255
            inv_bits = Uint32(inv_byte) << 23
            if inv_byte == 0:
                inv_bits = Uint32(0x00400000)
            inv_scale = _bits_f32(inv_bits)
            if exponent >= 128:
                inv_scale = Float32(0.0)
            if group_nan:
                scale_byte = Int32(0)
            nibbles = _f32x4_to_e2m1x4(
                _mul_rn(x[0], inv_scale),
                _mul_rn(x[1], inv_scale),
                _mul_rn(x[2], inv_scale),
                _mul_rn(x[3], inv_scale),
            )
            if group_nan:
                nibbles = Uint32(0x7777)
            if row < num_rows:
                data_col = head * (PACK_HEAD // 2) + group * (PACK_GROUP // 2) + 2 * lane8
                gData[row, data_col] = Uint8(nibbles)
                gData[row, data_col + 1] = Uint8(nibbles >> Uint32(8))
                if lane8 == 0:
                    gScale[row, head * (PACK_HEAD // PACK_GROUP) + group] = Uint8(scale_byte)

    @cute.jit
    def _main_loop(
        self,
        acc: cute.Tensor,
        gA_vec: cute.Tensor,
        copy_a: cute.CopyAtom,
        gCosSin: cute.Tensor,
        b_regs: cute.Tensor,
        s_regs: cute.Tensor,
        s2: cute.Tensor,
        g_regs: cute.Tensor,
        squares: cute.Tensor,
        rope_pos: cute.Tensor,
        k_warp0,
        a_col0,
        q,
        r,
        fast: cutlass.Constexpr,
    ) -> None:
        """The warp's K loop; ``fast`` converts the weights with the BF16 scale pairs."""
        num_rows: cutlass.Constexpr = self.num_rows
        row_tiles: cutlass.Constexpr = self.row_tiles
        chunks: cutlass.Constexpr = self.chunks_per_warp
        n_tiles: cutlass.Constexpr = self.n_tiles
        # Rows 16t + r ("lo") and 16t + r + 8 ("hi"); k 16q + 0..7 (0) and 16q + 8..15 (1).
        a_lo0 = cute.make_rmem_tensor(cute.make_layout((8,)), BFloat16)
        a_lo1 = cute.make_rmem_tensor(cute.make_layout((8,)), BFloat16)
        a_hi0 = cute.make_rmem_tensor(cute.make_layout((8,)), BFloat16)
        a_hi1 = cute.make_rmem_tensor(cute.make_layout((8,)), BFloat16)
        a_lo0_u32 = cute.recast_tensor(a_lo0, Uint32)
        a_lo1_u32 = cute.recast_tensor(a_lo1, Uint32)
        a_hi0_u32 = cute.recast_tensor(a_hi0, Uint32)
        a_hi1_u32 = cute.recast_tensor(a_hi1, Uint32)
        a_frag = cute.make_rmem_tensor(cute.make_layout((8,)), BFloat16)
        a_frag_u32 = cute.recast_tensor(a_frag, Uint32)
        b_frag = cute.make_rmem_tensor(cute.make_layout((4,)), BFloat16)
        b_frag_u32 = cute.recast_tensor(b_frag, Uint32)
        tile = cute.make_rmem_tensor(cute.make_layout((4,)), Float32)
        w_bf16 = cute.make_rmem_tensor(cute.make_layout((8,)), Uint32)
        w_bf16b = cute.make_rmem_tensor(cute.make_layout((8,)), Uint32)

        for c in cutlass.range_constexpr(chunks):
            k0 = k_warp0 + c * CHUNK + 16 * q
            ka = a_col0 + k0
            if const_expr(n_tiles == 1):
                self._convert_weights(w_bf16, b_regs, s_regs, s2, c, fast)
            for t in cutlass.range_constexpr(row_tiles):
                row_lo = 16 * t + r
                row_hi = 16 * t + r + 8
                a_lo0_u32.fill(0)
                a_lo1_u32.fill(0)
                a_hi0_u32.fill(0)
                a_hi1_u32.fill(0)
                if row_lo < num_rows:
                    cute.copy(copy_a, gA_vec[row_lo, (None, ka // 8)], a_lo0)
                    cute.copy(copy_a, gA_vec[row_lo, (None, ka // 8 + 1)], a_lo1)
                if const_expr(16 * t + 8 < num_rows):
                    if row_hi < num_rows:
                        cute.copy(copy_a, gA_vec[row_hi, (None, ka // 8)], a_hi0)
                        cute.copy(copy_a, gA_vec[row_hi, (None, ka // 8 + 1)], a_hi1)
                if const_expr(self.a_rms_norm):
                    self._rms_scale(a_lo0, g_regs, c, 0, squares, t, 0)
                    self._rms_scale(a_lo1, g_regs, c, 8, squares, t, 0)
                    if const_expr(16 * t + 8 < num_rows):
                        self._rms_scale(a_hi0, g_regs, c, 0, squares, t, 1)
                        self._rms_scale(a_hi1, g_regs, c, 8, squares, t, 1)
                if const_expr(
                    self.a_rope_head_dim > 0 and ((c + 1) * CHUNK) % self.a_rope_head_dim == 0
                ):
                    # The head's last CHUNK columns: lane q holds rotation pairs 8q .. 8q+7.
                    self._inverse_rope(a_lo0, gCosSin, rope_pos[t, 0], 8 * q)
                    self._inverse_rope(a_lo1, gCosSin, rope_pos[t, 0], 8 * q + 4)
                    if const_expr(16 * t + 8 < num_rows):
                        self._inverse_rope(a_hi0, gCosSin, rope_pos[t, 1], 8 * q)
                        self._inverse_rope(a_hi1, gCosSin, rope_pos[t, 1], 8 * q + 4)
                if const_expr(self.swap_ab):
                    # A = the weights of tiles 2p (rows 0-7) and 2p + 1 (rows 8-15), B = the tokens
                    # (zeros past num_rows): the same physical k in the same MMA k position.
                    for tp in cutlass.range_constexpr(n_tiles // 2):
                        self._convert_weights(w_bf16, b_regs, s_regs, s2, 2 * tp * chunks + c, fast)
                        self._convert_weights(
                            w_bf16b, b_regs, s_regs, s2, (2 * tp + 1) * chunks + c, fast
                        )
                        for e in cutlass.range_constexpr(4):
                            tile[e] = acc[tp, e]
                        for j in cutlass.range_constexpr(4):
                            a_frag_u32[0] = w_bf16[2 * j]
                            a_frag_u32[1] = w_bf16b[2 * j]
                            a_frag_u32[2] = w_bf16[2 * j + 1]
                            a_frag_u32[3] = w_bf16b[2 * j + 1]
                            if const_expr(j < 2):
                                b_frag_u32[0] = a_lo0_u32[2 * j]
                                b_frag_u32[1] = a_lo0_u32[2 * j + 1]
                            else:
                                b_frag_u32[0] = a_lo1_u32[2 * j - 4]
                                b_frag_u32[1] = a_lo1_u32[2 * j - 3]
                            tile.store(mma_sync(a_frag, b_frag, tile))
                        for e in cutlass.range_constexpr(4):
                            acc[tp, e] = tile[e]
                for nt in cutlass.range_constexpr(0 if self.swap_ab else n_tiles):
                    if const_expr(n_tiles > 1):
                        # Converted per row tile, so one tile's BF16 weights are live at a time.
                        self._convert_weights(w_bf16, b_regs, s_regs, s2, nt * chunks + c, fast)
                    for e in cutlass.range_constexpr(4):
                        tile[e] = acc[nt * row_tiles + t, e]
                    for j in cutlass.range_constexpr(4):
                        # A: a0 = (row_lo, 4j..4j+1), a1 = (row_hi, same), a2/a3 = 4j+2..4j+3.
                        if const_expr(j < 2):
                            a_frag_u32[0] = a_lo0_u32[2 * j]
                            a_frag_u32[1] = a_hi0_u32[2 * j]
                            a_frag_u32[2] = a_lo0_u32[2 * j + 1]
                            a_frag_u32[3] = a_hi0_u32[2 * j + 1]
                        else:
                            a_frag_u32[0] = a_lo1_u32[2 * j - 4]
                            a_frag_u32[1] = a_hi1_u32[2 * j - 4]
                            a_frag_u32[2] = a_lo1_u32[2 * j - 3]
                            a_frag_u32[3] = a_hi1_u32[2 * j - 3]
                        b_frag_u32[0] = w_bf16[2 * j]
                        b_frag_u32[1] = w_bf16[2 * j + 1]
                        tile.store(mma_sync(a_frag, b_frag, tile))
                    for e in cutlass.range_constexpr(4):
                        acc[nt * row_tiles + t, e] = tile[e]

    @cute.jit
    def _convert_weights(
        self,
        w_bf16: cute.Tensor,
        b_regs: cute.Tensor,
        s_regs: cute.Tensor,
        s2: cute.Tensor,
        slot: cutlass.Constexpr,
        fast: cutlass.Constexpr,
    ) -> None:
        """The 16 E4M3 weights of register slot ``slot`` times their scale -> 8 BF16 pairs
        (``fast``: by the BF16 scale pair ``s2[slot]``, else the FP32 scale; the same bits)."""
        raw = cute.recast_tensor(b_regs[slot, None], Uint32)
        if const_expr(self.bf16_weights):
            for w in cutlass.range_constexpr(8):
                w_bf16[w] = raw[w]
        else:
            for w in cutlass.range_constexpr(4):
                if const_expr(fast):
                    pair = _fp8x4_to_bf16x4_s2(raw[w], s2[slot])
                else:
                    pair = _fp8x4_to_bf16x4_scaled(raw[w], s_regs[slot])
                w_bf16[2 * w] = pair[0]
                w_bf16[2 * w + 1] = pair[1]

    @cute.jit
    def _rms_scale(
        self,
        vec: cute.Tensor,
        g_regs: cute.Tensor,
        slot: cutlass.Constexpr,
        g0: cutlass.Constexpr,
        squares: cute.Tensor,
        t: cutlass.Constexpr,
        h: cutlass.Constexpr,
    ) -> None:
        """Add the squares of the eight BF16 values of ``vec`` to ``squares[t, h]`` and multiply
        them by gamma ``g_regs[slot, g0 .. g0 + 7]`` (rounded to BF16)."""
        total = squares[t, h]
        for i in cutlass.range_constexpr(8):
            x = vec[i].to(Float32)
            total = total + x * x
            vec[i] = (x * g_regs[slot, g0 + i].to(Float32)).to(BFloat16)
        squares[t, h] = total

    @cute.jit
    def _inverse_rope(self, vec: cute.Tensor, cos_sin: cute.Tensor, position, p0) -> None:
        """Rotate the four interleaved BF16 pairs of ``vec`` (pairs ``p0 .. p0 + 3`` at
        ``position``) back by their angle, in FP32."""
        for i in cutlass.range_constexpr(4):
            x0 = vec[2 * i].to(Float32)
            x1 = vec[2 * i + 1].to(Float32)
            cos = cos_sin[position, 0, p0 + i]
            sin = cos_sin[position, 1, p0 + i]
            vec[2 * i] = (x0 * cos + x1 * sin).to(BFloat16)
            vec[2 * i + 1] = (x1 * cos - x0 * sin).to(BFloat16)

    @cute.jit
    def _store(self, gC: cute.Tensor, gC2: cute.Tensor, row, n, value: Float32) -> None:
        if const_expr(self.split_n == 0):
            gC[row, n] = value.to(gC.element_type)
        else:
            if n < self.split_n:
                gC[row, n] = value.to(gC.element_type)
            else:
                gC2[row, n - self.split_n] = value.to(gC2.element_type)
