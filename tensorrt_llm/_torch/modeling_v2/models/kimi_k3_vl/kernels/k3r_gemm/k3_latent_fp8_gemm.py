# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi-K3 latent MoE projection as an FP8 GEMM with a fused epilogue (SM100/SM103/SM107).

The projection uses an FP8 E4M3 weight ``W [N, K]`` with one power-of-two (UE8M0) scale per
128x128 block (the resmoothed weight and DeepGEMM-packed scale that ``fp8_swap_ab_gemm`` call
sites hold; on SM107 the scale is rebuilt from the CuTe MXFP8 layout, see
``k3_fp8w_lowm_gemm.pack_block_scale_from_mxfp8_layout``) and a BF16 activation ``x [M, K]``:

* :func:`k3_latent_down_fp4`: ``q, sf = nvfp4_quant(bf16(x @ dequant(W)^T), global_sf)``,
  the routed-expert GEMM1 input: E2M1 codes ``[M, N / 2]`` (two per byte, low nibble first) and
  one E4M3 scale per 16 outputs ``[M, N / 16]`` in the linear layout. Replaces the activation
  quantization [+ UE8M0 packing], DeepGEMM and ``fp4_quantize``.

The op is two kernels chained with PDL (on SM107, up to :func:`lowm_route_max_m` tokens the
op instead runs the split-K GEMM of :mod:`k3_fp8w_lowm_gemm` with the same epilogue, see
below):

* :class:`K3Fp8Quant1x128Kernel` quantizes the activation to E4M3 with one UE8M0 scale per
  1x128 block, using the exact ``2^ceil(log2(amax / 448))`` scale of the Triton and CUDA
  ``fp8_quantize_1x128`` kernels, and writes the scale replicated over the four K32 slots of the
  tcgen05 scale-factor atom (``[K/128][ceil(M/128)][512 B]``, atom byte
  ``(r % 32) * 16 + (r // 32) * 4 + k32``) so the GEMM moves one contiguous 512-byte block per
  K tile.
* :class:`K3Fp8LatentGemmKernel` is a persistent MXFP8 GEMM (``tcgen05.mma.kind::mxf8f6f4`` with
  1x32 scale vectors, i.e. the 1x128 scales replicated) over 128 x ``bn`` output tiles with no
  split-K, so every output is the same FP32 accumulation chain over K as DeepGEMM's. Warp 0
  loads the activation and weight tiles with TMA plus the activation scale block with a bulk
  copy (the first weight stages do not wait on the previous grid), warp 1 issues the MMAs into
  a TMEM accumulator that is double-buffered when two fit, warp 2 writes the weight scale atoms
  (the 128x128 block exponents, read from the DeepGEMM-packed scale) and warps 4-7 run the
  epilogue on 32-column TMEM blocks: BF16 rounding, then either NVFP4 quantization with exactly
  the arithmetic of ``fp4_quantize`` (``quantize_with_block_size``) or the residual adds
  ``residual + bf16(bf16(acc) + shared)``, and 16-byte global stores.

The persistent GEMM has one 128 x ``bn`` tile per CTA and no split-K, so a decode batch
(``M <= 128``) runs only ``N / bn`` CTAs (28 for latent_down), each limited by the weight bytes
its ring keeps in flight. On SM107 the op therefore runs the split-K cluster GEMM of
``k3_fp8w_lowm_gemm`` (1x32 MXFP8 activation scales instead of 1x128) with its NVFP4 epilogue
for ``M <= lowm_route_max_m(N, K)``, at the tactic of :func:`route_tactic`; the results equal
that GEMM at the same tactic followed by ``fp4_quantize``.
"""

from __future__ import annotations

import functools

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
import cutlass.utils.blockscaled_layout as blockscaled_utils
import torch
from cutlass import Int32, Int64
from cutlass._mlir.dialects import llvm
from cutlass.cute.nvgpu import OperandMajorMode, cpasync, tcgen05
from cutlass.cute.runtime import make_ptr
from cutlass.cutlass_dsl import T, dsl_user_op

#: K extent of one pipeline stage; also the activation / weight scale block length.
CTA_K = 128
#: Output rows per CTA tile (UMMA M).
BM = 128
#: tcgen05 MXF8 scale vector length.
SF_VEC = 32
#: NVFP4 scale vector length of the output.
FP4_VEC = 16
#: Supported output-column tiles (UMMA N).
SUPPORTED_BN = (128, 192, 256)
#: Largest M the op accepts.
MAX_M = 4096

_TMA_WARP = 0
_MMA_WARP = 1
_SFB_WARP = 2
_EPI_WARP0 = 4
_THREADS = 256
_SFA_TMEM_COLS = 16

_SMEM_CAPACITY = utils.get_smem_capacity_in_bytes("sm_100")


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


# ---------------------------------------------------------------------------
# Inline PTX helpers
# ---------------------------------------------------------------------------


def _asm(res, operands, text, constraints, *, loc=None, ip=None):
    return llvm.inline_asm(
        res,
        operands,
        text,
        constraints,
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


def _asm_pure(res, operands, text, constraints, *, loc=None, ip=None):
    """Inline PTX without side effects: the compiler may schedule or drop it freely."""
    return llvm.inline_asm(
        res,
        operands,
        text,
        constraints,
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


def _iv(value, *, loc=None, ip=None):
    return Int32(value).ir_value(loc=loc, ip=ip)


def _fv(value, *, loc=None, ip=None):
    return cutlass.Float32(value).ir_value(loc=loc, ip=ip)


def _struct(res, n, ty, *, loc=None, ip=None):
    return tuple(llvm.extractvalue(ty, res, [i], loc=loc, ip=ip) for i in range(n))


@dsl_user_op
def _ldg_v4_b32(addr, *, loc=None, ip=None):
    res = _asm(
        llvm.StructType.get_literal([T.i32()] * 4),
        [Int64(addr).ir_value(loc=loc, ip=ip)],
        "ld.global.v4.b32 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,l",
        loc=loc,
        ip=ip,
    )
    return tuple(Int32(v) for v in _struct(res, 4, T.i32(), loc=loc, ip=ip))


@dsl_user_op
def _ldg_f32(addr, *, loc=None, ip=None):
    return cutlass.Float32(
        _asm(
            T.f32(),
            [Int64(addr).ir_value(loc=loc, ip=ip)],
            "ld.global.f32 $0, [$1];",
            "=f,l",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _stg_v4_b32(addr, v0, v1, v2, v3, *, loc=None, ip=None):
    _asm(
        None,
        [Int64(addr).ir_value(loc=loc, ip=ip)] + [_iv(v, loc=loc, ip=ip) for v in (v0, v1, v2, v3)],
        "st.global.v4.b32 [$0], {$1, $2, $3, $4};",
        "l,r,r,r,r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _stg_u8(addr, value, *, loc=None, ip=None):
    _asm(
        None,
        [Int64(addr).ir_value(loc=loc, ip=ip), _iv(value, loc=loc, ip=ip)],
        "st.global.u8 [$0], $1;",
        "l,r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _stg_u16(addr, value, *, loc=None, ip=None):
    _asm(
        None,
        [Int64(addr).ir_value(loc=loc, ip=ip), _iv(value, loc=loc, ip=ip)],
        "st.global.u16 [$0], $1;",
        "l,r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _sts_v4_b32(addr, v0, v1, v2, v3, *, loc=None, ip=None):
    _asm(
        None,
        [_iv(x, loc=loc, ip=ip) for x in (addr, v0, v1, v2, v3)],
        "st.shared.v4.b32 [$0], {$1, $2, $3, $4};",
        "r,r,r,r,r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _bulk_load_g2s(dst_smem, src_gmem, nbytes, mbar, *, loc=None, ip=None):
    """Bulk-copy ``nbytes`` from global memory into local SMEM, crediting ``mbar``."""
    _asm(
        None,
        [
            _iv(dst_smem, loc=loc, ip=ip),
            Int64(src_gmem).ir_value(loc=loc, ip=ip),
            _iv(nbytes, loc=loc, ip=ip),
            _iv(mbar, loc=loc, ip=ip),
        ],
        "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [$0], [$1], $2, [$3];",
        "r,l,r,r",
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _griddepcontrol_wait(*, loc=None, ip=None):
    _asm(None, [], "griddepcontrol.wait;", "", loc=loc, ip=ip)


@dsl_user_op
def _griddepcontrol_launch_dependents(*, loc=None, ip=None):
    _asm(None, [], "griddepcontrol.launch_dependents;", "", loc=loc, ip=ip)


@dsl_user_op
def _min_s32(a, b, *, loc=None, ip=None):
    return Int32(
        _asm_pure(
            T.i32(),
            [_iv(a, loc=loc, ip=ip), _iv(b, loc=loc, ip=ip)],
            "min.s32 $0, $1, $2;",
            "=r,r,r",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _abs_max_bf16x16(words, *, loc=None, ip=None):
    """|x| max of 16 BF16 values (8 packed words) as two packed BF16 magnitudes (sign garbage)."""
    body = "".join(f"max.xorsign.abs.bf16x2 m{i}, ${2 * i + 1}, ${2 * i + 2};\n" for i in range(4))
    body += (
        "max.xorsign.abs.bf16x2 m0, m0, m1;\n"
        "max.xorsign.abs.bf16x2 m2, m2, m3;\n"
        "max.xorsign.abs.bf16x2 $0, m0, m2;\n"
    )
    return Int32(
        _asm_pure(
            T.i32(),
            [_iv(w, loc=loc, ip=ip) for w in words],
            "{\n.reg .b32 m0, m1, m2, m3;\n" + body + "}",
            "=r," + ",".join(["r"] * 8),
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _max_bf16_mag(a, b, *, loc=None, ip=None):
    """Packed BF16 magnitude max of two packed words (sign garbage, masked by the consumer)."""
    return Int32(
        _asm_pure(
            T.i32(),
            [_iv(a, loc=loc, ip=ip), _iv(b, loc=loc, ip=ip)],
            "max.xorsign.abs.bf16x2 $0, $1, $2;",
            "=r,r,r",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _ue8m0_of_amax(packed_amax, *, loc=None, ip=None):
    """Biased exponent of ``2^ceil(log2(max(amax, 1e-10) / 448))`` from two packed BF16 magnitudes.

    For ``amax = 2^E * (1 + f / 128)`` (a BF16 value): ``amax / 448 <= 2^(E - 8)`` iff
    ``1 + f / 128 <= 1.75``, so the biased exponent is ``E - 8`` plus one when the 7-bit
    mantissa exceeds ``0x60``. Magnitudes below 1e-10 (only zero for real activations) use the
    exponent of 1e-10, ``2^-42`` (biased 85).
    """
    return Int32(
        _asm_pure(
            T.i32(),
            [_iv(packed_amax, loc=loc, ip=ip)],
            "{\n"
            ".reg .b32 lo, hi, a, e, mant;\n"
            ".reg .pred p, z;\n"
            "and.b32 lo, $1, 0x7fff;\n"
            "shr.u32 hi, $1, 16;\n"
            "and.b32 hi, hi, 0x7fff;\n"
            "max.u32 a, lo, hi;\n"
            "shr.u32 e, a, 7;\n"
            "and.b32 mant, a, 0x7f;\n"
            "setp.gt.u32 p, mant, 0x60;\n"
            "sub.u32 e, e, 8;\n"
            "@p add.u32 e, e, 1;\n"
            # BF16 1e-10 rounds to 0x2edc; anything at or below the FP32 1e-10 floor takes
            # its scale. 0x2edb..0x2edc straddle 1e-10, so compare against the exact bits.
            "setp.lt.u32 z, a, 0x2edc;\n"
            "@z mov.b32 e, 85;\n"
            "mov.b32 $0, e;\n"
            "}",
            "=r,r",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _bf16x2_inv_scale(ue8m0, *, loc=None, ip=None):
    """Packed BF16x2 ``2^(127 - ue8m0)``."""
    return Int32(
        _asm_pure(
            T.i32(),
            [_iv(ue8m0, loc=loc, ip=ip)],
            "{\n"
            ".reg .b32 s, t;\n"
            "sub.u32 s, 254, $1;\n"
            "shl.b32 s, s, 7;\n"
            "shl.b32 t, s, 16;\n"
            "or.b32 $0, s, t;\n"
            "}",
            "=r,r",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _quant_bf16x8_to_e4m3(w0, w1, w2, w3, inv_scale, *, loc=None, ip=None):
    """Scale eight BF16 values by the packed power-of-two ``inv_scale`` and round to E4M3.

    BF16 has the FP32 exponent range, so the BF16 multiply by a power of two is exact (no
    result is subnormal for scales from real activations) and the only rounding is the
    satfinite round-to-nearest-even conversion, as in ``clamp(x / s).to(float8_e4m3fn)``.
    """
    res = _asm_pure(
        llvm.StructType.get_literal([T.i32()] * 2),
        [_iv(x, loc=loc, ip=ip) for x in (w0, w1, w2, w3, inv_scale)],
        "{\n"
        ".reg .b32 p0, p1, p2, p3;\n"
        ".reg .b16 h0, h1, h2, h3;\n"
        "mul.rn.bf16x2 p0, $2, $6;\n"
        "mul.rn.bf16x2 p1, $3, $6;\n"
        "mul.rn.bf16x2 p2, $4, $6;\n"
        "mul.rn.bf16x2 p3, $5, $6;\n"
        "cvt.rn.satfinite.e4m3x2.bf16x2 h0, p0;\n"
        "cvt.rn.satfinite.e4m3x2.bf16x2 h1, p1;\n"
        "cvt.rn.satfinite.e4m3x2.bf16x2 h2, p2;\n"
        "cvt.rn.satfinite.e4m3x2.bf16x2 h3, p3;\n"
        "mov.b32 $0, {h0, h1};\n"
        "mov.b32 $1, {h2, h3};\n"
        "}",
        "=r,=r,r,r,r,r,r",
        loc=loc,
        ip=ip,
    )
    return tuple(Int32(v) for v in _struct(res, 2, T.i32(), loc=loc, ip=ip))


@dsl_user_op
def _rcp_approx_ftz(x, *, loc=None, ip=None):
    return cutlass.Float32(
        _asm_pure(
            T.f32(), [_fv(x, loc=loc, ip=ip)], "rcp.approx.ftz.f32 $0, $1;", "=f,f", loc=loc, ip=ip
        )
    )


@dsl_user_op
def _bf16_round_x2(a, b, *, loc=None, ip=None):
    """Round two FP32 values to BF16 (RN) and return them widened back to FP32."""
    res = _asm_pure(
        llvm.StructType.get_literal([T.f32()] * 2),
        [_fv(a, loc=loc, ip=ip), _fv(b, loc=loc, ip=ip)],
        "{\n"
        ".reg .b32 p;\n"
        ".reg .b16 l, h;\n"
        "cvt.rn.bf16x2.f32 p, $3, $2;\n"
        "mov.b32 {l, h}, p;\n"
        "cvt.f32.bf16 $0, l;\n"
        "cvt.f32.bf16 $1, h;\n"
        "}",
        "=f,=f,f,f",
        loc=loc,
        ip=ip,
    )
    return tuple(cutlass.Float32(v) for v in _struct(res, 2, T.f32(), loc=loc, ip=ip))


@dsl_user_op
def _abs_max_f32x16(vals, *, loc=None, ip=None):
    body = "".join(f"abs.f32 t{i}, ${i + 1};\n" for i in range(16))
    for step in (8, 4, 2, 1):
        body += "".join(f"max.f32 t{i}, t{i}, t{i + step};\n" for i in range(step))
    body += "mov.f32 $0, t0;\n"
    return cutlass.Float32(
        _asm_pure(
            T.f32(),
            [_fv(v, loc=loc, ip=ip) for v in vals],
            "{\n.reg .f32 " + ", ".join(f"t{i}" for i in range(16)) + ";\n" + body + "}",
            "=f," + ",".join(["f"] * 16),
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _nvfp4_scales_x2(amax0, amax1, gsf, rgsf, rcp6, *, loc=None, ip=None):
    """NVFP4 scales of two 16-element groups, as in ``cvt_warp_fp16_to_fp4`` (E4M3 scale).

    ``sf = e4m3(gsf * (amax * rcp(6)))`` and ``out_scale = amax != 0 ? rcp(f32(sf) * rcp(gsf)) : 0``
    with ``rcp`` = ``rcp.approx.ftz.f32``. Returns the two scale bytes packed in 16 bits (group
    0 in the low byte) and the two output scales.
    """
    res = _asm_pure(
        llvm.StructType.get_literal([T.i32(), T.f32(), T.f32()]),
        [_fv(x, loc=loc, ip=ip) for x in (amax0, amax1, gsf, rgsf, rcp6)],
        "{\n"
        ".reg .f32 s0, s1, f0, f1;\n"
        ".reg .b16 h, l16, h16;\n"
        ".reg .b32 hw;\n"
        ".reg .pred z0, z1;\n"
        "mul.rn.f32 s0, $3, $7;\n"
        "mul.rn.f32 s1, $4, $7;\n"
        "mul.rn.f32 s0, $5, s0;\n"
        "mul.rn.f32 s1, $5, s1;\n"
        "cvt.rn.satfinite.e4m3x2.f32 h, s1, s0;\n"
        "cvt.rn.f16x2.e4m3x2 hw, h;\n"
        "mov.b32 {l16, h16}, hw;\n"
        "cvt.f32.f16 f0, l16;\n"
        "cvt.f32.f16 f1, h16;\n"
        "mul.rn.f32 f0, f0, $6;\n"
        "mul.rn.f32 f1, f1, $6;\n"
        "rcp.approx.ftz.f32 f0, f0;\n"
        "rcp.approx.ftz.f32 f1, f1;\n"
        "setp.eq.f32 z0, $3, 0f00000000;\n"
        "setp.eq.f32 z1, $4, 0f00000000;\n"
        "selp.f32 $1, 0f00000000, f0, z0;\n"
        "selp.f32 $2, 0f00000000, f1, z1;\n"
        "cvt.u32.u16 $0, h;\n"
        "}",
        "=r,=f,=f,f,f,f,f,f",
        loc=loc,
        ip=ip,
    )
    sf = llvm.extractvalue(T.i32(), res, [0], loc=loc, ip=ip)
    o0 = llvm.extractvalue(T.f32(), res, [1], loc=loc, ip=ip)
    o1 = llvm.extractvalue(T.f32(), res, [2], loc=loc, ip=ip)
    return Int32(sf), cutlass.Float32(o0), cutlass.Float32(o1)


@dsl_user_op
def _e2m1x8(vals, scale, *, loc=None, ip=None):
    """Scale eight FP32 values (``mul.rn``) and pack them as E2M1 into 32 bits, element 0 lowest."""
    body = "".join(f"mul.rn.f32 t{i}, ${i + 1}, $9;\n" for i in range(8))
    body += "".join(
        f"cvt.rn.satfinite.e2m1x2.f32 b{j}, t{2 * j + 1}, t{2 * j};\n" for j in range(4)
    )
    body += "mov.b32 $0, {b0, b1, b2, b3};\n"
    return Int32(
        _asm_pure(
            T.i32(),
            [_fv(v, loc=loc, ip=ip) for v in vals] + [_fv(scale, loc=loc, ip=ip)],
            "{\n.reg .f32 t0, t1, t2, t3, t4, t5, t6, t7;\n.reg .b8 b0, b1, b2, b3;\n" + body + "}",
            "=r," + ",".join(["f"] * 9),
            loc=loc,
            ip=ip,
        )
    )


# ---------------------------------------------------------------------------
# Activation quantization
# ---------------------------------------------------------------------------


class K3Fp8Quant1x128Kernel:
    """``x`` BF16 ``[M, K]`` -> E4M3 ``[M, K]`` + UE8M0 1x128 scales in the UMMA scale atom layout.

    A work item is one (row, 32-element group); the four groups of a 128-element block are
    handled by four consecutive lanes, which share the block's absolute maximum through two
    shuffles. Each lane writes its own (replicated) scale byte. Rows ``M .. 128 * ceil(M/128)``
    get the neutral scale 1.0 (their FP8 rows are TMA zero-fill in the GEMM). With
    ``sf_m_major`` the scale atoms are ordered ``[ceil(M/128)][K/128][512 B]`` (the CUTLASS
    block-scaled layout) instead of ``[K/128][ceil(M/128)][512 B]``.
    """

    def __init__(
        self, k: int, use_pdl: bool, threads: int = 128, items: int = 2, sf_m_major: bool = False
    ):
        if k % CTA_K:
            raise ValueError(f"K={k} must be a multiple of {CTA_K}")
        self.k = k
        self.groups = k // SF_VEC
        self.use_pdl = use_pdl
        self.sf_m_major = sf_m_major
        self._THREADS = threads
        self._ITEMS = items

    @cute.jit
    def __call__(
        self,
        x_ptr: cute.Pointer,
        xq_ptr: cute.Pointer,
        xs_ptr: cute.Pointer,
        m: Int32,
        stream: cuda.CUstream,
    ):
        items = ((m + BM - 1) // BM) * BM * self.groups
        per_cta = self._THREADS * self._ITEMS
        self.kernel(x_ptr, xq_ptr, xs_ptr, m).launch(
            grid=((items + per_cta - 1) // per_cta, 1, 1),
            block=(self._THREADS, 1, 1),
            stream=stream,
            use_pdl=self.use_pdl,
        )

    @cute.kernel
    def kernel(self, x_ptr: cute.Pointer, xq_ptr: cute.Pointer, xs_ptr: cute.Pointer, m: Int32):
        if cutlass.const_expr(self.use_pdl):
            # The GEMM reads this grid's outputs only after its own griddepcontrol.wait; let it
            # launch now so its weight stream overlaps this kernel.
            _griddepcontrol_launch_dependents()
            _griddepcontrol_wait()
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        gdim, _, _ = cute.arch.grid_dim()
        m_tiles = (m + BM - 1) // BM
        items = m_tiles * BM * self.groups
        stride = gdim * self._THREADS
        ipt = self._ITEMS
        rX = cute.make_rmem_tensor((ipt * 16,), Int32)
        # Issue every load first; padded or out-of-range items re-read a valid row.
        for it in cutlass.range_constexpr(ipt):
            item = _min_s32(bidx * self._THREADS + tidx + it * stride, items - 1)
            r = _min_s32(item // self.groups, m - 1)
            g = item - (item // self.groups) * self.groups
            src = x_ptr.toint() + Int64(r * self.k + g * SF_VEC) * 2
            for v in cutlass.range_constexpr(4):
                w0, w1, w2, w3 = _ldg_v4_b32(src + v * 16)
                rX[it * 16 + v * 4] = w0
                rX[it * 16 + v * 4 + 1] = w1
                rX[it * 16 + v * 4 + 2] = w2
                rX[it * 16 + v * 4 + 3] = w3
        for it in cutlass.range_constexpr(ipt):
            item = bidx * self._THREADS + tidx + it * stride
            # Lanes 4i..4i+3 always hold the four groups of one 128-element block (the item
            # count and the per-pass stride are multiples of 4), so the shuffles stay inside
            # the block even for out-of-range quads.
            ws = [rX[it * 16 + i] for i in range(16)]
            amax = _max_bf16_mag(_abs_max_bf16x16(ws[:8]), _abs_max_bf16x16(ws[8:]))
            amax = _max_bf16_mag(amax, cute.arch.shuffle_sync_bfly(amax, offset=1))
            amax = _max_bf16_mag(amax, cute.arch.shuffle_sync_bfly(amax, offset=2))
            if item < items:
                r = item // self.groups
                g = item - r * self.groups
                rr = r & (BM - 1)
                if cutlass.const_expr(self.sf_m_major):
                    atom = (r >> 7) * (self.k // CTA_K) + (g >> 2)
                else:
                    atom = (g >> 2) * m_tiles + (r >> 7)
                sf_off = atom * 512 + (rr & 31) * 16 + (rr >> 5) * 4
                sf_addr = xs_ptr.toint() + Int64(sf_off + (g & 3))
                if r < m:
                    e_a = _ue8m0_of_amax(amax)
                    inv = _bf16x2_inv_scale(e_a)
                    packed = []
                    for v in cutlass.range_constexpr(4):
                        p0, p1 = _quant_bf16x8_to_e4m3(*ws[4 * v : 4 * v + 4], inv)
                        packed.extend((p0, p1))
                    dst = xq_ptr.toint() + Int64(r * self.k + g * SF_VEC)
                    _stg_v4_b32(dst, *packed[:4])
                    _stg_v4_b32(dst + 16, *packed[4:])
                    _stg_u8(sf_addr, e_a)
                else:
                    _stg_u8(sf_addr, Int32(127))


# ---------------------------------------------------------------------------
# GEMM with fused epilogues
# ---------------------------------------------------------------------------

#: Residual epilogue: per-warp SMEM staging of a 32-row x 32-column BF16 block, rows padded to
#: 80 bytes so the row-per-thread 16-byte stores are bank-conflict free.
_RES_ROW_BYTES = 80
_RES_WARP_BYTES = 32 * _RES_ROW_BYTES
#: Residual epilogue: blocks of shared / residual words loaded ahead of their use (deeper
#: prefetch measured no faster on GB300).
_RES_PREFETCH = 1

#: NVFP4-quantized output (codes + linear E4M3 scales) for the routed-expert GEMM1.
EPI_NVFP4 = "nvfp4"
#: BF16 output plus the MoE residual update ``residual + bf16(bf16(acc) + shared)``.
EPI_RESIDUAL = "residual"


@dsl_user_op
def _residual_from_bf16x2(t, shared, residual, *, loc=None, ip=None):
    """``bf16(r + bf16(t + s))`` for two BF16 columns, each add in FP32 as PyTorch does."""
    return Int32(
        _asm_pure(
            T.i32(),
            [_iv(x, loc=loc, ip=ip) for x in (t, shared, residual)],
            "{\n"
            ".reg .b32 u;\n"
            ".reg .b16 l, h;\n"
            ".reg .f32 a0, a1, b0, b1;\n"
            "mov.b32 {l, h}, $1;\n"
            "cvt.f32.bf16 a0, l;\n"
            "cvt.f32.bf16 a1, h;\n"
            "mov.b32 {l, h}, $2;\n"
            "cvt.f32.bf16 b0, l;\n"
            "cvt.f32.bf16 b1, h;\n"
            "add.rn.f32 a0, a0, b0;\n"
            "add.rn.f32 a1, a1, b1;\n"
            "cvt.rn.bf16x2.f32 u, a1, a0;\n"
            "mov.b32 {l, h}, u;\n"
            "cvt.f32.bf16 a0, l;\n"
            "cvt.f32.bf16 a1, h;\n"
            "mov.b32 {l, h}, $3;\n"
            "cvt.f32.bf16 b0, l;\n"
            "cvt.f32.bf16 b1, h;\n"
            "add.rn.f32 a0, b0, a0;\n"
            "add.rn.f32 a1, b1, a1;\n"
            "cvt.rn.bf16x2.f32 $0, a1, a0;\n"
            "}",
            "=r,r,r,r",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _f32x2_to_bf16x2(a, b, *, loc=None, ip=None):
    """Round two FP32 values to BF16 (RN), packed with ``a`` in the low half."""
    return Int32(
        _asm_pure(
            T.i32(),
            [_fv(a, loc=loc, ip=ip), _fv(b, loc=loc, ip=ip)],
            "cvt.rn.bf16x2.f32 $0, $2, $1;",
            "=r,f,f",
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _lds_v4_b32(addr, *, loc=None, ip=None):
    res = _asm(
        llvm.StructType.get_literal([T.i32()] * 4),
        [_iv(addr, loc=loc, ip=ip)],
        "ld.shared.v4.b32 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,r",
        loc=loc,
        ip=ip,
    )
    return tuple(Int32(v) for v in _struct(res, 4, T.i32(), loc=loc, ip=ip))


def smem_bytes(bn: int, stages: int, epilogue: str = EPI_NVFP4) -> int:
    """Conservative host-side mirror of the GEMM's shared-memory footprint."""
    stage = BM * CTA_K + bn * CTA_K + 512 + _ceil_div(bn, BM) * 512
    staging = 4 * _RES_WARP_BYTES if epilogue == EPI_RESIDUAL else 0
    return 1024 + stages * stage + staging + 2048


def max_stages(bn: int, epilogue: str = EPI_NVFP4) -> int:
    s = 1
    while smem_bytes(bn, s + 1, epilogue) <= _SMEM_CAPACITY:
        s += 1
    return s


class K3Fp8LatentGemmKernel:
    """MXFP8 GEMM ``[M, K] x [N, K]^T`` over a static persistent tile schedule, fused epilogue.

    CTA ``c`` runs tiles ``c, c + grid, ...``; tile ``t`` is output rows ``128 * (t % m_tiles)``
    and columns ``bn * (t // m_tiles)``, so the CTAs sharing a weight tile run together. The
    activation (E4M3, from :class:`K3Fp8Quant1x128Kernel`) and weight stages share one ring;
    the accumulator is double-buffered in TMEM when two fit, so the epilogue of one tile
    overlaps the mainloop of the next.
    """

    def __init__(
        self,
        n: int,
        k: int,
        bn: int,
        stages: int,
        epilogue: str,
        use_pdl: bool,
    ):
        if bn not in SUPPORTED_BN:
            raise ValueError(f"bn must be one of {SUPPORTED_BN}, got {bn}")
        if n % BM or k % CTA_K:
            raise ValueError(f"N={n} and K={k} must be multiples of {BM}")
        if epilogue not in (EPI_NVFP4, EPI_RESIDUAL):
            raise ValueError(f"unknown epilogue {epilogue!r}")
        if smem_bytes(bn, stages, epilogue) > _SMEM_CAPACITY:
            raise ValueError(f"{stages} stages of bn={bn} exceed shared memory")
        self.n = n
        self.k = k
        self.bn = bn
        self.stages = stages
        self.epilogue_kind = epilogue
        self.use_pdl = use_pdl
        self.kt_total = k // CTA_K
        # The last column tile may be partial (bn = 192); its missing weight rows are TMA
        # zero fill and its missing columns are not stored.
        self.n_tiles = _ceil_div(n, bn)
        # Scale-factor atoms (128 rows) per stage of the weight operand.
        self.nb = _ceil_div(bn, BM)
        # A tile spans at most this many 128-row weight scale blocks.
        self.wblocks = bn // BM + (1 if bn % BM else 0)
        self.mma_tiler = (BM, bn, CTA_K)
        self.mma_tiler_sfb = (BM, self.nb * BM, CTA_K)
        self.a_stage_bytes = BM * CTA_K
        self.b_stage_bytes = bn * CTA_K
        self.sfb_stage_bytes = self.nb * 512
        sf_cols = _SFA_TMEM_COLS + self.nb * 16
        self.acc_bufs = 2 if 2 * bn + sf_cols <= 512 else 1
        cols = self.acc_bufs * bn + sf_cols
        self.tmem_cols = 1 << (cols - 1).bit_length()
        self.sfa_tmem_off = self.acc_bufs * bn
        self.sfb_tmem_off = self.sfa_tmem_off + _SFA_TMEM_COLS
        # Weight block exponents per lane: K tiles lane, lane + 32, ...
        self.exp_regs = _ceil_div(self.kt_total, 32)
        # Prefetched weight stages issued before waiting on the previous grid.
        self.prefetch = min(stages, self.kt_total)
        self.staging_bytes = 4 * _RES_WARP_BYTES if epilogue == EPI_RESIDUAL else 16
        # Epilogue warps (warps 4 .. 4 + epi_warps - 1); subclasses may use two per TMEM lane quarter.
        self.epi_warps = 4

    @cute.jit
    def __call__(
        self,
        xq_ptr: cute.Pointer,
        xs_ptr: cute.Pointer,
        w_ptr: cute.Pointer,
        ws_ptr: cute.Pointer,
        out0_ptr: cute.Pointer,
        out1_ptr: cute.Pointer,
        aux0_ptr: cute.Pointer,
        aux1_ptr: cute.Pointer,
        m: Int32,
        num_ctas: Int32,
        stream: cuda.CUstream,
    ):
        """``nvfp4``: out0 = codes, out1 = scales, aux0 = global scale (aux1 unused).
        ``residual``: out0 = BF16 output, aux0 = shared-expert output, aux1 = residual."""
        n, k = self.n, self.k
        mA = cute.make_tensor(xq_ptr, cute.make_layout((m, k, 1), stride=(k, 1, m * k)))
        mB = cute.make_tensor(w_ptr, cute.make_layout((n, k, 1), stride=(k, 1, n * k)))
        # DeepGEMM packed UE8M0 weight scale: int32 [N, ceil(K/512)], MN-major; byte
        # (kb % 4) of word (n, kb // 4) holds the exponent of block (n // 128, kb).
        mWS = cute.make_tensor(
            ws_ptr,
            cute.make_layout((n // BM, (4, _ceil_div(k // CTA_K, 4))), stride=(BM * 4, (1, 4 * n))),
        )
        tiled_mma = sm100_utils.make_blockscaled_trivial_tiled_mma(
            cutlass.Float8E4M3FN,
            cutlass.Float8E4M3FN,
            OperandMajorMode.K,
            OperandMajorMode.K,
            cutlass.Float8E8M0FNU,
            SF_VEC,
            tcgen05.CtaGroup.ONE,
            (BM, self.bn),
        )
        a_layout = sm100_utils.make_smem_layout_a(
            tiled_mma, self.mma_tiler, cutlass.Float8E4M3FN, self.stages
        )
        b_layout = sm100_utils.make_smem_layout_b(
            tiled_mma, self.mma_tiler, cutlass.Float8E4M3FN, self.stages
        )
        sfa_layout = blockscaled_utils.make_smem_layout_sfa(
            tiled_mma, self.mma_tiler, SF_VEC, self.stages
        )
        # The weight scale operand covers whole 128-row atoms (bn rounded up to 128).
        tiled_mma_sfb = sm100_utils.make_blockscaled_trivial_tiled_mma(
            cutlass.Float8E4M3FN,
            cutlass.Float8E4M3FN,
            OperandMajorMode.K,
            OperandMajorMode.K,
            cutlass.Float8E8M0FNU,
            SF_VEC,
            tcgen05.CtaGroup.ONE,
            (BM, self.nb * BM),
        )
        sfb_layout = blockscaled_utils.make_smem_layout_sfb(
            tiled_mma_sfb, self.mma_tiler_sfb, SF_VEC, self.stages
        )
        cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout((1, 1, 1)), (tiled_mma.thr_id.shape,)
        )
        tma_atom_a, tma_tensor_a = cute.nvgpu.make_tiled_tma_atom_A(
            cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE),
            mA,
            cute.slice_(a_layout, (None, None, None, 0)),
            self.mma_tiler,
            tiled_mma,
            cluster_layout_vmnk.shape,
        )
        tma_atom_b, tma_tensor_b = cute.nvgpu.make_tiled_tma_atom_B(
            cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE),
            mB,
            cute.slice_(b_layout, (None, None, None, 0)),
            self.mma_tiler,
            tiled_mma,
            cluster_layout_vmnk.shape,
        )

        @cute.struct
        class SharedStorage:
            full: cute.struct.MemRange[cutlass.Int64, self.stages]
            empty: cute.struct.MemRange[cutlass.Int64, self.stages]
            acc_full: cute.struct.MemRange[cutlass.Int64, self.acc_bufs]
            acc_empty: cute.struct.MemRange[cutlass.Int64, self.acc_bufs]
            tmem_ready: cutlass.Int64
            tmem_free: cutlass.Int64
            tmem_holding_buf: cutlass.Int32
            sA: cute.struct.Align[
                cute.struct.MemRange[cutlass.Float8E4M3FN, cute.cosize(a_layout.outer)], 1024
            ]
            sB: cute.struct.Align[
                cute.struct.MemRange[cutlass.Float8E4M3FN, cute.cosize(b_layout.outer)], 1024
            ]
            sSFA: cute.struct.Align[
                cute.struct.MemRange[cutlass.Float8E8M0FNU, cute.cosize(sfa_layout)], 128
            ]
            sSFB: cute.struct.Align[
                cute.struct.MemRange[cutlass.Float8E8M0FNU, cute.cosize(sfb_layout)], 128
            ]
            sOut: cute.struct.Align[cute.struct.MemRange[cutlass.Uint8, self.staging_bytes], 128]

        self.shared_storage = SharedStorage
        self.kernel(
            tiled_mma,
            tiled_mma_sfb,
            tma_atom_a,
            tma_tensor_a,
            tma_atom_b,
            tma_tensor_b,
            xs_ptr,
            mWS,
            out0_ptr,
            out1_ptr,
            aux0_ptr,
            aux1_ptr,
            m,
            a_layout,
            b_layout,
            sfa_layout,
            sfb_layout,
        ).launch(
            grid=(num_ctas, 1, 1),
            block=(32 * (_EPI_WARP0 + self.epi_warps), 1, 1),
            smem=self.shared_storage.size_in_bytes(),
            min_blocks_per_mp=1,
            stream=stream,
            use_pdl=self.use_pdl,
        )

    @cute.kernel
    def kernel(
        self,
        tiled_mma: cute.TiledMma,
        tiled_mma_sfb: cute.TiledMma,
        tma_atom_a: cute.CopyAtom,
        mA: cute.Tensor,
        tma_atom_b: cute.CopyAtom,
        mB: cute.Tensor,
        xs_ptr: cute.Pointer,
        mWS: cute.Tensor,
        out0_ptr: cute.Pointer,
        out1_ptr: cute.Pointer,
        aux0_ptr: cute.Pointer,
        aux1_ptr: cute.Pointer,
        m: Int32,
        a_layout: cute.ComposedLayout,
        b_layout: cute.ComposedLayout,
        sfa_layout: cute.Layout,
        sfb_layout: cute.Layout,
    ):
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        gdim, _, _ = cute.arch.grid_dim()
        m_tiles = (m + BM - 1) // BM
        num_tiles = m_tiles * self.n_tiles

        storage = utils.SmemAllocator().allocate(self.shared_storage)
        full = storage.full.data_ptr()
        empty = storage.empty.data_ptr()

        if warp_idx == _TMA_WARP:
            cpasync.prefetch_descriptor(tma_atom_a)
            cpasync.prefetch_descriptor(tma_atom_b)
            with cute.arch.elect_one():
                for i in cutlass.range_constexpr(self.stages):
                    # Arrivals: the TMA producer (with the transaction bytes) and the SFB writer.
                    cute.arch.mbarrier_init(full + i, 2)
                    cute.arch.mbarrier_init(empty + i, 1)
                for i in cutlass.range_constexpr(self.acc_bufs):
                    cute.arch.mbarrier_init(storage.acc_full.data_ptr() + i, 1)
                    cute.arch.mbarrier_init(storage.acc_empty.data_ptr() + i, self.epi_warps)
                cute.arch.mbarrier_init(storage.tmem_ready, 1)
                cute.arch.mbarrier_init(storage.tmem_free, self.epi_warps)
        cute.arch.mbarrier_init_fence()
        cute.arch.barrier()

        if warp_idx == _TMA_WARP:
            self.producer(
                tma_atom_a,
                mA,
                tma_atom_b,
                mB,
                tiled_mma,
                storage,
                xs_ptr,
                m_tiles,
                a_layout,
                b_layout,
            )
        elif warp_idx == _SFB_WARP:
            self.sfb_writer(storage, mWS, m_tiles)
        elif warp_idx == _MMA_WARP:
            self.mma_warp(
                tiled_mma,
                tiled_mma_sfb,
                storage,
                a_layout,
                b_layout,
                sfa_layout,
                sfb_layout,
                num_tiles,
            )
        elif warp_idx >= _EPI_WARP0:
            self.epilogue(
                tiled_mma,
                storage,
                out0_ptr,
                out1_ptr,
                aux0_ptr,
                aux1_ptr,
                m,
                m_tiles,
                tidx - 32 * _EPI_WARP0,
            )

    @cute.jit
    def tile_coords(self, t: Int32, m_tiles: Int32):
        """``(n_tile, m_tile)`` of linear tile ``t``: column-major, so the CTAs running together
        share a weight tile. Every warp role maps tiles through this method."""
        n_tile = t // m_tiles
        return n_tile, t - n_tile * m_tiles

    @cute.jit
    def producer(
        self,
        tma_atom_a,
        mA,
        tma_atom_b,
        mB,
        tiled_mma,
        storage,
        xs_ptr,
        m_tiles,
        a_layout,
        b_layout,
    ):
        """TMA warp: weight + activation tiles and the activation scale block per stage."""
        full = storage.full.data_ptr()
        empty = storage.empty.data_ptr()
        bidx, _, _ = cute.arch.block_idx()
        gdim, _, _ = cute.arch.grid_dim()
        num_tiles = m_tiles * self.n_tiles
        sA = storage.sA.get_tensor(a_layout.outer, swizzle=a_layout.inner)
        sB = storage.sB.get_tensor(b_layout.outer, swizzle=b_layout.inner)
        gA = cute.local_tile(mA, cute.slice_(self.mma_tiler, (None, 0, None)), (None, None, None))
        gB = cute.local_tile(mB, cute.slice_(self.mma_tiler, (0, None, None)), (None, None, None))
        thr_mma = tiled_mma.get_slice(0)
        tAsA, tAgA = cpasync.tma_partition(
            tma_atom_a,
            0,
            cute.make_layout(1),
            cute.group_modes(sA, 0, 3),
            cute.group_modes(thr_mma.partition_A(gA), 0, 3),
        )
        tBsB, tBgB = cpasync.tma_partition(
            tma_atom_b,
            0,
            cute.make_layout(1),
            cute.group_modes(sB, 0, 3),
            cute.group_modes(thr_mma.partition_B(gB), 0, 3),
        )
        sfa_base = storage.sSFA.data_ptr().toint()
        stage_bytes = self.a_stage_bytes + self.b_stage_bytes + 512
        # Weights do not depend on the previous grid: fill the ring with the first tile's
        # weight stages, then wait for the activation.
        n_tile0, _ = self.tile_coords(bidx, m_tiles)
        if bidx < num_tiles:
            for kt in cutlass.range(self.prefetch, unroll=1):
                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive_and_expect_tx(full + kt, stage_bytes)
                cute.copy(
                    tma_atom_b,
                    tBgB[(None, n_tile0, kt, 0)],
                    tBsB[(None, kt)],
                    tma_bar_ptr=full + kt,
                )
        if cutlass.const_expr(self.use_pdl):
            _griddepcontrol_wait()
        it = Int32(0)
        for t in cutlass.range(bidx, num_tiles, gdim, unroll=1):
            n_tile, m_tile = self.tile_coords(t, m_tiles)
            for kt in cutlass.range(self.kt_total, unroll=1):
                stage = it % self.stages
                if it >= self.stages:
                    cute.arch.mbarrier_wait(empty + stage, ((it // self.stages) & 1) ^ 1)
                if it >= self.prefetch:
                    with cute.arch.elect_one():
                        cute.arch.mbarrier_arrive_and_expect_tx(full + stage, stage_bytes)
                    cute.copy(
                        tma_atom_b,
                        tBgB[(None, n_tile, kt, 0)],
                        tBsB[(None, stage)],
                        tma_bar_ptr=full + stage,
                    )
                cute.copy(
                    tma_atom_a,
                    tAgA[(None, m_tile, kt, 0)],
                    tAsA[(None, stage)],
                    tma_bar_ptr=full + stage,
                )
                with cute.arch.elect_one():
                    _bulk_load_g2s(
                        sfa_base + stage * 512,
                        xs_ptr.toint() + Int64((kt * m_tiles + m_tile) * 512),
                        512,
                        (full + stage).toint(),
                    )
                it = it + 1

    @cute.jit
    def sfb_writer(self, storage, mWS: cute.Tensor, m_tiles: Int32):
        """Write each stage's weight scale atoms from the 128x128 block exponents.

        Atom ``b`` covers tile rows ``128 b .. 128 b + 127``; its byte
        ``(r % 32) * 16 + (r // 32) * 4 + k32`` belongs to tile row ``128 b + r``. A 32-row group
        never straddles a weight block, so lane ``l`` writes one 4-byte word per group ``r // 32``.
        Rows past ``N`` reuse the last block's exponent (their weight rows are zero).
        """
        full = storage.full.data_ptr()
        empty = storage.empty.data_ptr()
        bidx, _, _ = cute.arch.block_idx()
        gdim, _, _ = cute.arch.grid_dim()
        num_tiles = m_tiles * self.n_tiles
        lane = cute.arch.lane_idx()
        nblk_w = self.n // BM
        sfb_base = storage.sSFB.data_ptr().toint()
        it = Int32(0)
        for t in cutlass.range(bidx, num_tiles, gdim, unroll=1):
            n_tile, _ = self.tile_coords(t, m_tiles)
            row0 = n_tile * self.bn
            blk0 = row0 // BM
            exps = []
            for wb in cutlass.range_constexpr(self.wblocks):
                blk = _min_s32(blk0 + wb, nblk_w - 1)
                for j in cutlass.range_constexpr(self.exp_regs):
                    e = Int32(127)
                    if lane + 32 * j < self.kt_total:
                        e = Int32(mWS[blk, lane + 32 * j])
                    exps.append(e)
            # Weight block (relative to blk0) of each (atom, 32-row group); warp-uniform.
            sel = []
            for b in cutlass.range_constexpr(self.nb):
                for g in cutlass.range_constexpr(4):
                    sel.append((row0 + b * BM + g * 32) // BM - blk0)
            for kt in cutlass.range(self.kt_total, unroll=1):
                stage = it % self.stages
                if it >= self.stages:
                    cute.arch.mbarrier_wait(empty + stage, ((it // self.stages) & 1) ^ 1)
                ek = []
                for wb in cutlass.range_constexpr(self.wblocks):
                    e = cute.arch.shuffle_sync(exps[wb * self.exp_regs], kt & 31)
                    for j in cutlass.range_constexpr(1, self.exp_regs):
                        ej = cute.arch.shuffle_sync(exps[wb * self.exp_regs + j], kt & 31)
                        if (kt >> 5) == j:
                            e = ej
                    ek.append(e)
                for b in cutlass.range_constexpr(self.nb):
                    words = []
                    for g in cutlass.range_constexpr(4):
                        e = ek[0]
                        for wb in cutlass.range_constexpr(1, self.wblocks):
                            if sel[b * 4 + g] == wb:
                                e = ek[wb]
                        words.append(e * Int32(0x01010101))
                    _sts_v4_b32(
                        sfb_base + stage * self.sfb_stage_bytes + b * 512 + lane * 16, *words
                    )
                cute.arch.fence_proxy("async.shared", space="cta")
                cute.arch.sync_warp()
                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive(full + stage)
                it = it + 1

    @cute.jit
    def mma_warp(
        self,
        tiled_mma: cute.TiledMma,
        tiled_mma_sfb: cute.TiledMma,
        storage,
        a_layout: cute.ComposedLayout,
        b_layout: cute.ComposedLayout,
        sfa_layout: cute.Layout,
        sfb_layout: cute.Layout,
        num_tiles: Int32,
    ):
        full = storage.full.data_ptr()
        empty = storage.empty.data_ptr()
        acc_full = storage.acc_full.data_ptr()
        acc_empty = storage.acc_empty.data_ptr()
        bidx, _, _ = cute.arch.block_idx()
        gdim, _, _ = cute.arch.grid_dim()
        cute.arch.alloc_tmem(self.tmem_cols, storage.tmem_holding_buf)
        cute.arch.relinquish_tmem_alloc_permit()
        with cute.arch.elect_one():
            cute.arch.mbarrier_arrive(storage.tmem_ready)
        acc_tmem_ptr = cute.arch.retrieve_tmem_ptr(
            cutlass.Float32, alignment=16, ptr_to_buffer_holding_addr=storage.tmem_holding_buf
        )
        acc_shape = tiled_mma.partition_shape_C(self.mma_tiler[:2])
        acc_layout = tiled_mma.make_fragment_C(acc_shape).layout
        tCtSFA = cute.make_tensor(
            cute.recast_ptr(acc_tmem_ptr + self.sfa_tmem_off, dtype=cutlass.Float8E8M0FNU),
            blockscaled_utils.make_tmem_layout_sfa(
                tiled_mma, self.mma_tiler, SF_VEC, cute.slice_(sfa_layout, (None, None, None, 0))
            ),
        )
        tCtSFB = cute.make_tensor(
            cute.recast_ptr(acc_tmem_ptr + self.sfb_tmem_off, dtype=cutlass.Float8E8M0FNU),
            blockscaled_utils.make_tmem_layout_sfb(
                tiled_mma_sfb,
                self.mma_tiler_sfb,
                SF_VEC,
                cute.slice_(sfb_layout, (None, None, None, 0)),
            ),
        )
        s2t_sfa, tCsSFA_s2t, tCtSFA_s2t = _s2t_partition_sf(
            storage.sSFA.get_tensor(sfa_layout), tCtSFA
        )
        s2t_sfb, tCsSFB_s2t, tCtSFB_s2t = _s2t_partition_sf(
            storage.sSFB.get_tensor(sfb_layout), tCtSFB
        )
        sA = storage.sA.get_tensor(a_layout.outer, swizzle=a_layout.inner)
        sB = storage.sB.get_tensor(b_layout.outer, swizzle=b_layout.inner)
        tCrA = tiled_mma.make_fragment_A(sA)
        tCrB = tiled_mma.make_fragment_B(sB)
        num_kphases = cute.size(tCrA, mode=[2])

        it = Int32(0)
        ti = Int32(0)
        for _t in cutlass.range(bidx, num_tiles, gdim, unroll=1):
            buf = ti % self.acc_bufs
            if ti >= self.acc_bufs:
                cute.arch.mbarrier_wait(acc_empty + buf, ((ti // self.acc_bufs) & 1) ^ 1)
            tCtAcc = cute.make_tensor(acc_tmem_ptr + buf * self.bn, acc_layout)
            tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
            for _kt in cutlass.range(self.kt_total, unroll=1):
                stage = it % self.stages
                cute.arch.mbarrier_wait(full + stage, (it // self.stages) & 1)
                cute.copy(s2t_sfa, tCsSFA_s2t[(None, None, None, None, stage)], tCtSFA_s2t)
                cute.copy(s2t_sfb, tCsSFB_s2t[(None, None, None, None, stage)], tCtSFB_s2t)
                for kp in cutlass.range_constexpr(num_kphases):
                    tiled_mma.set(tcgen05.Field.SFA, tCtSFA[(None, None, kp)].iterator)
                    tiled_mma.set(tcgen05.Field.SFB, tCtSFB[(None, None, kp)].iterator)
                    cute.gemm(
                        tiled_mma,
                        tCtAcc,
                        tCrA[(None, None, kp, stage)],
                        tCrB[(None, None, kp, stage)],
                        tCtAcc,
                    )
                    tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                with cute.arch.elect_one():
                    tcgen05.commit(empty + stage)
                it = it + 1
            with cute.arch.elect_one():
                tcgen05.commit(acc_full + buf)
            ti = ti + 1

        cute.arch.mbarrier_wait(storage.tmem_free, 0)
        cute.arch.dealloc_tmem(acc_tmem_ptr, self.tmem_cols)

    @cute.jit
    def epilogue(
        self,
        tiled_mma: cute.TiledMma,
        storage,
        out0_ptr: cute.Pointer,
        out1_ptr: cute.Pointer,
        aux0_ptr: cute.Pointer,
        aux1_ptr: cute.Pointer,
        m: Int32,
        m_tiles: Int32,
        q: Int32,
    ):
        """Thread ``q`` owns output row ``128 * m_tile + q`` of every tile; 32-column TMEM blocks."""
        acc_full = storage.acc_full.data_ptr()
        acc_empty = storage.acc_empty.data_ptr()
        bidx, _, _ = cute.arch.block_idx()
        gdim, _, _ = cute.arch.grid_dim()
        num_tiles = m_tiles * self.n_tiles
        gsf = cutlass.Float32(0.0)
        rgsf = cutlass.Float32(0.0)
        rcp6 = cutlass.Float32(0.0)
        if cutlass.const_expr(self.epilogue_kind == EPI_NVFP4):
            gsf = _ldg_f32(aux0_ptr.toint())
            rgsf = _rcp_approx_ftz(gsf)
            rcp6 = _rcp_approx_ftz(cutlass.Float32(6.0))
        cute.arch.mbarrier_wait(storage.tmem_ready, 0)
        acc_tmem_ptr = cute.arch.retrieve_tmem_ptr(
            cutlass.Float32, alignment=16, ptr_to_buffer_holding_addr=storage.tmem_holding_buf
        )
        acc_shape = tiled_mma.partition_shape_C(self.mma_tiler[:2])
        acc_layout = tiled_mma.make_fragment_C(acc_shape).layout
        tAcc0 = cute.make_tensor(acc_tmem_ptr, acc_layout)[((None, None), 0, 0)]
        cAcc = cute.make_identity_tensor((BM, self.bn))
        blk = 32
        tiled_t2r = tcgen05.make_tmem_copy(
            cute.make_copy_atom(tcgen05.Ld32x32bOp(tcgen05.Repetition(blk)), cutlass.Float32),
            cute.local_tile(tAcc0, (BM, blk), (0, 0)),
        )
        thr_t2r = tiled_t2r.get_slice(q)
        nblk = self.bn // blk
        tTR_cAcc0 = thr_t2r.partition_D(cute.local_tile(cAcc, (BM, blk), (0, 0)))
        rBuf = [cute.make_rmem_tensor(tTR_cAcc0.shape, cutlass.Float32) for _ in range(2)]

        ti = Int32(0)
        for t in cutlass.range(bidx, num_tiles, gdim, unroll=1):
            n_tile, m_tile = self.tile_coords(t, m_tiles)
            buf = ti % self.acc_bufs
            tAcc = cute.make_tensor(acc_tmem_ptr + buf * self.bn, acc_layout)[((None, None), 0, 0)]
            row = m_tile * BM + q
            col0 = n_tile * self.bn
            # Residual epilogue: the shared-expert and residual words of block j are loaded
            # _RES_PREFETCH blocks ahead (the first ones while the mainloop still runs) in the
            # coalesced layout.
            pf = {}
            if cutlass.const_expr(self.epilogue_kind == EPI_RESIDUAL):
                for j in cutlass.range_constexpr(min(_RES_PREFETCH, nblk)):
                    pf[j] = self._load_residual_block(
                        aux0_ptr, aux1_ptr, m_tile, col0 + j * blk, m, q
                    )
            cute.arch.mbarrier_wait(acc_full + buf, (ti // self.acc_bufs) & 1)
            cute.copy(
                tiled_t2r, thr_t2r.partition_S(cute.local_tile(tAcc, (BM, blk), (0, 0))), rBuf[0]
            )
            for j in cutlass.range_constexpr(nblk):
                cute.arch.fence_view_async_tmem_load()
                rAcc = rBuf[j % 2]
                if cutlass.const_expr(j + 1 < nblk):
                    cute.copy(
                        tiled_t2r,
                        thr_t2r.partition_S(cute.local_tile(tAcc, (BM, blk), (0, j + 1))),
                        rBuf[(j + 1) % 2],
                    )
                if cutlass.const_expr(self.epilogue_kind == EPI_NVFP4):
                    if row < m and col0 + j * blk < self.n:
                        self._store_nvfp4(
                            rAcc, out0_ptr, out1_ptr, row, col0 + j * blk, gsf, rgsf, rcp6
                        )
                else:
                    if cutlass.const_expr(j + _RES_PREFETCH < nblk):
                        pf[j + _RES_PREFETCH] = self._load_residual_block(
                            aux0_ptr, aux1_ptr, m_tile, col0 + (j + _RES_PREFETCH) * blk, m, q
                        )
                    self._store_residual(
                        rAcc, storage, out0_ptr, pf[j], m_tile, col0 + j * blk, m, q
                    )
            # Every TMEM load of this buffer has completed (fence above); release it.
            cute.arch.fence_view_async_tmem_load()
            cute.arch.sync_warp()
            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive(acc_empty + buf)
            ti = ti + 1
        cute.arch.sync_warp()
        with cute.arch.elect_one():
            cute.arch.mbarrier_arrive(storage.tmem_free)

    @cute.jit
    def _store_nvfp4(self, rAcc, q_ptr, sf_ptr, row, col, gsf, rgsf, rcp6):
        """32 accumulators of one row -> BF16 -> 16 bytes of E2M1 codes + two E4M3 scales."""
        vals = []
        for i in cutlass.range_constexpr(16):
            a, b = _bf16_round_x2(rAcc[2 * i], rAcc[2 * i + 1])
            vals.extend((a, b))
        amax0 = _abs_max_f32x16(vals[:16])
        amax1 = _abs_max_f32x16(vals[16:])
        sf2, os0, os1 = _nvfp4_scales_x2(amax0, amax1, gsf, rgsf, rcp6)
        w0 = _e2m1x8(vals[0:8], os0)
        w1 = _e2m1x8(vals[8:16], os0)
        w2 = _e2m1x8(vals[16:24], os1)
        w3 = _e2m1x8(vals[24:32], os1)
        _stg_v4_b32(q_ptr.toint() + Int64(row) * (self.n // 2) + Int64(col // 2), w0, w1, w2, w3)
        _stg_u16(sf_ptr.toint() + Int64(row) * (self.n // FP4_VEC) + Int64(col // FP4_VEC), sf2)

    @cute.jit
    def _load_residual_block(self, shared_ptr, residual_ptr, m_tile, col, m, q):
        """Shared / residual words of one 32-column block in the coalesced store layout.

        Warp ``q // 32`` handles rows ``32 (q // 32) + 8 i + lane // 4`` (``i < 4``), lane
        ``l`` the 16 bytes at column ``col + 8 (l % 4)``; rows / columns that are not stored
        read a clamped in-bounds address.
        """
        lane = q % 32
        r0 = m_tile * BM + (q - lane) + lane // 4
        c = _min_s32(col + (lane % 4) * 8, self.n - 8)
        words = []
        for i in cutlass.range_constexpr(4):
            r = _min_s32(r0 + 8 * i, m - 1)
            off = (Int64(r) * self.n + Int64(c)) * 2
            words.append(
                (_ldg_v4_b32(shared_ptr.toint() + off), _ldg_v4_b32(residual_ptr.toint() + off))
            )
        return words

    @cute.jit
    def _store_residual(self, rAcc, storage, out_ptr, block, m_tile, col, m, q):
        """One 32-column block: ``residual + bf16(bf16(acc) + shared)`` with coalesced stores.

        Thread ``q`` rounds its row to BF16 into the warp's SMEM staging rows; after a warp
        sync, every 4 lanes combine and store 64 contiguous bytes of one row.
        """
        lane = q % 32
        base = storage.sOut.data_ptr().toint() + (q // 32) * _RES_WARP_BYTES
        words = []
        for i in cutlass.range_constexpr(16):
            words.append(_f32x2_to_bf16x2(rAcc[2 * i], rAcc[2 * i + 1]))
        for v in cutlass.range_constexpr(4):
            _sts_v4_b32(base + lane * _RES_ROW_BYTES + v * 16, *words[4 * v : 4 * v + 4])
        cute.arch.sync_warp()
        r0 = m_tile * BM + (q - lane) + lane // 4
        c = col + (lane % 4) * 8
        for i in cutlass.range_constexpr(4):
            t = _lds_v4_b32(base + (lane // 4 + 8 * i) * _RES_ROW_BYTES + (lane % 4) * 16)
            sh, rs = block[i]
            o = [_residual_from_bf16x2(t[v], sh[v], rs[v]) for v in range(4)]
            r = r0 + 8 * i
            if r < m and c < self.n:
                _stg_v4_b32(out_ptr.toint() + (Int64(r) * self.n + Int64(c)) * 2, *o)
        # The staging rows are rewritten by the next block.
        cute.arch.sync_warp()


def _s2t_partition_sf(sSF: cute.Tensor, tSF: cute.Tensor):
    """S2T (SMEM -> TMEM) copy of a scale-factor operand, partitioned for one thread."""
    tCsSF_compact = cute.filter_zeros(sSF)
    tCtSF_compact = cute.filter_zeros(tSF)
    copy_atom = cute.make_copy_atom(
        tcgen05.Cp4x32x128bOp(tcgen05.CtaGroup.ONE), cutlass.Float8E8M0FNU
    )
    tiled_copy = tcgen05.make_s2t_copy(copy_atom, tCtSF_compact)
    thr_copy = tiled_copy.get_slice(0)
    tCsSF_s2t = tcgen05.get_s2t_smem_desc_tensor(tiled_copy, thr_copy.partition_S(tCsSF_compact))
    tCtSF_s2t = thr_copy.partition_D(tCtSF_compact)
    return tiled_copy, tCsSF_s2t, tCtSF_s2t


# ---------------------------------------------------------------------------
# Host side
# ---------------------------------------------------------------------------


@functools.cache
def _num_sms() -> int:
    return torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count


def bn_for(m: int, n: int) -> int:
    """Column tile minimizing the per-CTA work of the persistent schedule (ties: wider tile)."""
    m_tiles = _ceil_div(m, BM)
    best, best_cost = SUPPORTED_BN[-1], None
    for bn in SUPPORTED_BN:
        cost = _ceil_div(m_tiles * _ceil_div(n, bn), _num_sms()) * bn
        if best_cost is None or cost <= best_cost:
            best, best_cost = bn, cost
    return best


@functools.cache
def _compiled_quant(k: int, use_pdl: bool, sf_m_major: bool = False):
    kernel = K3Fp8Quant1x128Kernel(k, use_pdl, sf_m_major=sf_m_major)
    dummy = 1 << 20
    x_ptr = make_ptr(cutlass.BFloat16, dummy, cute.AddressSpace.gmem, assumed_align=16)
    xq_ptr = make_ptr(cutlass.Float8E4M3FN, dummy, cute.AddressSpace.gmem, assumed_align=16)
    xs_ptr = make_ptr(cutlass.Uint8, dummy, cute.AddressSpace.gmem, assumed_align=16)
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    return cute.compile(kernel, x_ptr, xq_ptr, xs_ptr, Int32(1), stream)


@functools.cache
def _compiled_gemm(n: int, k: int, bn: int, stages: int, epilogue: str, use_pdl: bool):
    kernel = K3Fp8LatentGemmKernel(n, k, bn, stages, epilogue, use_pdl)
    dummy = 1 << 20
    ptrs = [
        make_ptr(dtype, dummy, cute.AddressSpace.gmem, assumed_align=16)
        for dtype in (cutlass.Float8E4M3FN, cutlass.Uint8, cutlass.Float8E4M3FN)
        + (cutlass.Uint8,) * 5
    ]
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    return cute.compile(kernel, *ptrs, Int32(1), Int32(1), stream)


def _gmem_ptr(dtype, tensor: torch.Tensor):
    return make_ptr(dtype, tensor.data_ptr(), cute.AddressSpace.gmem, assumed_align=16)


def quantize_act_1x128(
    x: torch.Tensor,
    xq: torch.Tensor,
    xs: torch.Tensor,
    *,
    use_pdl: bool = True,
    sf_m_major: bool = False,
) -> None:
    """Launch :class:`K3Fp8Quant1x128Kernel` on the current stream."""
    m, k = x.shape
    stream = cuda.CUstream(torch.cuda.current_stream(x.device).cuda_stream)
    _compiled_quant(k, use_pdl, sf_m_major)(
        _gmem_ptr(cutlass.BFloat16, x),
        _gmem_ptr(cutlass.Float8E4M3FN, xq),
        _gmem_ptr(cutlass.Uint8, xs),
        Int32(m),
        stream,
    )


#: Per compute capability: (N, K) -> largest M for which the fused ops run the split-K GEMM of
#: ``k3_fp8w_lowm_gemm`` with the fused epilogue instead of the persistent kernel (measured on
#: Rubin, CUDA graph, cold weights: faster than the persistent kernel and than the unfused
#: CuTe MXFP8 / low-M GEMM + ``fp4_quantize`` / adds at every M up to the limit).
_LOWM_ROUTE_MAX_M = {
    (10, 7): {(3584, 7168): 256},
}
#: Per compute capability: (N, K) -> ``(min M, max M, kernel, split_k)`` ranges where the route
#: uses ``k3_fp8w_lowm_gemm.split_tactic`` (kernel "low" or "mid") instead of that module's
#: default tactic. latent_down runs on a side stream next to the routing chain: its standalone
#: optimum (up to 224 CTAs) delays the router GEMM and routing kernels, so it uses narrower
#: grids (84-112 CTAs) that measured faster per MoE layer with the routing beside them.
_LOWM_ROUTE_TACTICS = {
    (10, 7): {
        (3584, 7168): ((1, 16, "low", 3), (17, 96, "mid", 4)),
    },
}


@functools.cache
def _device_capability(device_index: int) -> tuple:
    return torch.cuda.get_device_capability(device_index)


def lowm_route_max_m(n: int, k: int) -> int:
    """Largest M for which the ops run the ``k3_fp8w_lowm_gemm`` split-K GEMM (0: never)."""
    table = _LOWM_ROUTE_MAX_M.get(_device_capability(torch.cuda.current_device()), {})
    return table.get((n, k), 0)


def route_tactic(m: int, n: int, k: int):
    """The ``k3_fp8w_lowm_gemm`` tactic of the route for M tokens (``None``: its default)."""
    from . import k3_fp8w_lowm_gemm

    table = _LOWM_ROUTE_TACTICS.get(_device_capability(torch.cuda.current_device()), {})
    for lo, hi, kernel, split_k in table.get((n, k), ()):
        if lo <= m <= hi:
            return k3_fp8w_lowm_gemm.split_tactic(m, n, k, split_k, mid=kernel == "mid")
    return None


def _run(x, w, w_scale, epilogue, out0, out1, aux0, aux1, bn, stages, use_pdl) -> None:
    m, k = x.shape
    n = w.shape[0]
    if bn is None and stages is None and m <= lowm_route_max_m(n, k):
        from . import k3_fp8w_lowm_gemm

        k3_fp8w_lowm_gemm.run_k3_fp8w_lowm_gemm_fused(
            x,
            w,
            w_scale,
            epilogue,
            out0,
            out1,
            aux0,
            aux1,
            tactic=route_tactic(m, n, k),
            use_pdl=use_pdl,
        )
        return
    if bn is None:
        bn = bn_for(m, n)
    if stages is None:
        stages = max_stages(bn, epilogue)
    m_tiles = _ceil_div(m, BM)
    num_ctas = min(m_tiles * _ceil_div(n, bn), _num_sms())
    xq = torch.empty((m, k), dtype=torch.float8_e4m3fn, device=x.device)
    xs = torch.empty((k // CTA_K) * m_tiles * 512, dtype=torch.uint8, device=x.device)
    quantize_act_1x128(x, xq, xs, use_pdl=use_pdl)
    stream = cuda.CUstream(torch.cuda.current_stream(x.device).cuda_stream)
    _compiled_gemm(n, k, bn, stages, epilogue, use_pdl)(
        _gmem_ptr(cutlass.Float8E4M3FN, xq),
        _gmem_ptr(cutlass.Uint8, xs),
        _gmem_ptr(cutlass.Float8E4M3FN, w),
        *(_gmem_ptr(cutlass.Uint8, t) for t in (w_scale, out0, out1, aux0, aux1)),
        Int32(m),
        Int32(num_ctas),
        stream,
    )


def run_k3_latent_down_fp4(
    x: torch.Tensor,
    w: torch.Tensor,
    w_scale: torch.Tensor,
    global_sf: torch.Tensor,
    q_out: torch.Tensor,
    sf_out: torch.Tensor,
    *,
    bn: int | None = None,
    stages: int | None = None,
    use_pdl: bool = True,
) -> None:
    """Quantize ``x`` and launch the NVFP4-epilogue GEMM into ``q_out`` / ``sf_out``.

    ``bn`` / ``stages`` select a tile of the persistent kernel; with neither, M up to
    :func:`lowm_route_max_m` runs the ``k3_fp8w_lowm_gemm`` split-K GEMM instead."""
    _run(x, w, w_scale, EPI_NVFP4, q_out, sf_out, global_sf, global_sf, bn, stages, use_pdl)


def _check_operands(x, w, w_scale) -> None:
    if x.dtype != torch.bfloat16 or x.dim() != 2 or not x.is_contiguous():
        raise ValueError("x must be a contiguous 2D bfloat16 tensor")
    if w.dtype != torch.float8_e4m3fn or w.dim() != 2 or not w.is_contiguous():
        raise ValueError("w must be a contiguous 2D float8_e4m3fn tensor")
    n, k = w.shape
    if x.shape[1] != k:
        raise ValueError(f"x has K={x.shape[1]}, weight has K={k}")
    if n % BM or k % CTA_K:
        raise ValueError(f"N={n} and K={k} must be multiples of {BM}")
    words = _ceil_div(k // CTA_K, 4)
    if (
        w_scale.dtype != torch.int32
        or tuple(w_scale.shape) != (n, words)
        or w_scale.stride(0) != 1
        or (words > 1 and w_scale.stride(1) != n)
    ):
        raise ValueError(
            f"w_scale must be DeepGEMM-packed UE8M0 int32 [{n}, {words}] with strides (1, {n})"
        )
    if x.data_ptr() % 16 or w.data_ptr() % 16:
        raise ValueError("x and w must be 16-byte aligned")


def _check_global_sf(global_sf) -> None:
    if (
        not isinstance(global_sf, torch.Tensor)
        or global_sf.dtype != torch.float32
        or global_sf.numel() != 1
        or not global_sf.is_cuda
    ):
        raise ValueError("global_sf must be a one-element float32 CUDA tensor")


@functools.cache
def _is_supported_device(device_index: int) -> bool:
    return torch.cuda.get_device_capability(device_index) in ((10, 0), (10, 3), (10, 7))


def _can_run(x, w, checks) -> bool:
    if x.dim() != 2 or w.dim() != 2 or not x.is_cuda:
        return False
    if not 1 <= x.shape[0] <= MAX_M:
        return False
    if not _is_supported_device(x.device.index):
        return False
    try:
        for check in checks:
            check()
    except ValueError:
        return False
    return True


def can_run(x: torch.Tensor, w: torch.Tensor, w_scale: torch.Tensor, global_sf) -> bool:
    """Whether :func:`k3_latent_down_fp4` accepts these inputs (non-raising check)."""
    return _can_run(
        x, w, (lambda: _check_operands(x, w, w_scale), lambda: _check_global_sf(global_sf))
    )


def k3_latent_down_fp4(
    x: torch.Tensor, w: torch.Tensor, w_scale: torch.Tensor, global_sf: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """NVFP4 ``(codes uint8 [M, N/2], scales uint8 [M, N/16])`` of ``bf16(x @ dequant(w)^T)``.

    ``w`` is the UE8M0-resmoothed FP8 weight and ``w_scale`` the DeepGEMM-packed UE8M0 scale
    (as held by ``fp8_swap_ab_gemm`` call sites); ``global_sf`` is the NVFP4 global scale of the
    consumer (``fp4_quantize``'s ``globalScale``). Scales use the linear layout. The activation
    is quantized with 1x128 UE8M0 scales, or with 1x32 ones where the op runs the split-K GEMM
    of ``k3_fp8w_lowm_gemm`` (``M <= lowm_route_max_m(N, K)``).
    """
    _check_operands(x, w, w_scale)
    _check_global_sf(global_sf)
    m = x.shape[0]
    n = w.shape[0]
    q_out = torch.empty((m, n // 2), dtype=torch.uint8, device=x.device)
    sf_out = torch.empty((m, n // FP4_VEC), dtype=torch.uint8, device=x.device)
    run_k3_latent_down_fp4(x, w, w_scale, global_sf, q_out, sf_out)
    return q_out, sf_out


__all__ = [
    "K3Fp8LatentGemmKernel",
    "K3Fp8Quant1x128Kernel",
    "MAX_M",
    "bn_for",
    "can_run",
    "k3_latent_down_fp4",
    "lowm_route_max_m",
    "quantize_act_1x128",
    "route_tactic",
    "run_k3_latent_down_fp4",
]
