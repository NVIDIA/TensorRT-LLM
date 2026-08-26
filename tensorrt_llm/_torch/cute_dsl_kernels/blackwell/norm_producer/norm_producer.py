# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright 2023-2026 SGLang Team
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
#
# Unified DiT norm-site producer kernel (generic-D CuTe DSL, Blackwell).
#
# One row-per-CTA kernel covering the fused pattern shared by every DiT
# norm site in the visual-gen models:
#
#   Store( (1 + scale) * Norm( x  [* gate]  [+ residual] ) + shift )
#
# with a compile-time variant dispatch over:
#   prologue    {none, residual, gate+residual}
#   norm        {LayerNorm no-affine, LayerNorm affine, RMSNorm weightless}
#   modulators  {none, per-batch [B, D] rows, per-token [B, S, D] chunk
#                views, each optionally composed inline with a [D] fp32
#                table row}
#   math mode   {fp32 composition (WAN dtype law: fp32 throughout, ONE
#                final rounding), bf16-narrow-first (LTX/Qwen/FLUX dtype
#                law: bf16 rounding at every eager op boundary)}
#   num_out     {1, 2 modulated outputs} x optional residual_out
#   store       {bf16, NVFP4 static-scale, NVFP4 deferred-scale}
#
# Provenance: the row-per-CTA scaffold (two-pass mean/var warp+CTA reduce,
# autovec 128-bit copies, symbolic-stride compile cache) is derived from
# SGLang's CuTe DSL ScaleResidualNormScaleShift kernel (Apache-2.0):
#   upstream repo: https://github.com/sgl-project/sglang
#   upstream path: python/sglang/kernels/ops/diffusion/cutedsl/
#                  scale_residual_norm_scale_shift.py (+ common/norm_fusion.py)
#   pinned SHA:    e1c4db9621f7c4203ee9becd5d5456d4e6bf54f7
# via the in-tree pertoken_adaln kernel (same package layout; reduce.py is
# imported from there rather than re-vendored).
#
# Store-form recipes:
#   - NVFP4 static: replicates tensorrt_llm::kernels::quantize_with_block_size
#     <FP16_TO_FP4, __nv_bfloat16, 16, false> BITWISE (quantization.cuh:
#     cvt_warp_fp16_to_fp4 L455-539 incl. the approximate reciprocals,
#     get_sf_out_offset_128x4 L703-741 swizzle, pad-row SF zeroing L831-859),
#     so payload+SF are byte-identical to
#     torch.ops.trtllm.fp4_quantize(y_bf16, global_scale, 16, False).
#     NOTE: the fusedAdaptiveLayerNorm CUDA kernel's own fp4 epilogue uses an
#     exact 1/6 constant and a true IEEE division instead - it can differ
#     from fp4_quantize on final-ULP boundary values; this kernel pins the
#     fp4_quantize recipe (the canonical one) instead.
#   - NVFP4 deferred: for DYNAMIC quantization where the global scale is
#     unknown at producer time. Emits a scale-invariant e2m1 payload
#     (y * (6/a) with the division PINNED to div.full.f32 - IEEE div.rn
#     differs on e2m1 RNE ties) plus raw fp32 block scales a/6 in
#     [M, D/16] row-major unswizzled. The sfc_finalize module in this
#     package (K2) later computes s = 448/max(raw) and the swizzled e4m3
#     SF tensor; the payload needs no second pass.
#
# torch.library custom op (functional, register_fake):
#   trtllm::fused_norm_producer -> List[Tensor]
# Tagged torch.Tag.needs_fixed_stride_order: inside compiled regions
# inductor otherwise passes arbitrary-strided buffers to opaque custom ops.

from typing import List, Optional

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass._mlir.dialects import llvm as _llvm

from ..pertoken_adaln.reduce import warp_reduce_sum
from ..utils import TRTLLM_ENABLE_PDL, griddepcontrol_launch_dependents, griddepcontrol_wait

WARP_SIZE = 32

TORCH_TO_CUTE_DTYPE = {
    torch.float16: cutlass.Float16,
    torch.bfloat16: cutlass.BFloat16,
    torch.float32: cutlass.Float32,
    # NVFP4 store epilogue outputs (packed e2m1 payload viewed as int32;
    # swizzled SF bytes as uint8).
    torch.int32: cutlass.Int32,
    torch.uint8: cutlass.Uint8,
}

_COMPILE_CACHE = {}

# cutlass-dsl 4.5.x segfaults compiling vec=16/32 kernels that carry fp32
# modulator or affine operands (the pertoken_adaln residual-vec16 crash
# class); 4.6.1 (the in-tree pin) compiles and passes all of them.
try:
    _CUTLASS_DSL_PRE_46 = tuple(int(x) for x in cutlass.__version__.split(".")[:2]) < (4, 6)
except Exception:
    _CUTLASS_DSL_PRE_46 = False

_REQUIRED_ALIGNMENT = 32

NORM_TYPES = ("layer", "rms")
MATH_MODES = ("fp32", "bf16")
STORE_FORMS = ("bf16", "nvfp4_static", "nvfp4_deferred")

# Elements per thread. 8 -> one LDG.128 per bf16 operand per thread,
# num_warps = D/256. Larger vec halves/quarters the CTA thread count (more
# ILP per thread, shorter reduce tree, more registers). The NVFP4 store
# forms accept 8 (16-element group amax via adjacent-lane pairing) or
# 16/32 (whole groups in-thread); the group amax is exactly associative so
# every scale and payload byte is bitwise-identical across vec.
DEFAULT_VEC = 8

# CuTe DSL specialization requires the tensor-or-scalar-sentinel operands in
# ``NormProducer`` and its nested helpers to remain unannotated: the
# decorator derives their concrete MLIR types from each compiled call.


def _to_fake_cute_arg(t: torch.Tensor | int | float) -> object:
    """Symbolic shapes/strides everywhere except the last dim (compile-time)
    so one compilation covers any B/S/M and any last-dim-divisible outer
    strides - including temb chunk views with strides (S*6*D, 6*D, 1) and
    per-batch modulation chunk rows with stride (k*D, 1)."""
    if isinstance(t, torch.Tensor):
        D = t.shape[-1]
        dtype = TORCH_TO_CUTE_DTYPE[t.dtype]
        shape = (*(cute.sym_int() for _ in range(t.ndim - 1)), D)
        stride = (*(cute.sym_int(divisibility=D) for _ in range(t.ndim - 1)), 1)
        return cute.runtime.make_fake_tensor(
            dtype, shape, stride, memspace=cute.AddressSpace.gmem, assumed_align=32
        )
    if isinstance(t, int):
        return cutlass.Int32(t)
    if isinstance(t, float):
        return cutlass.Float32(t)
    return t


@cute.arch.dsl_user_op
def _fp32x8_to_e2m1x8_packed(x0, x1, x2, x3, x4, x5, x6, x7, *, loc=None, ip=None):
    """8 fp32 -> 8 packed e2m1 codes in one Int32 (element 2k in the low
    nibble of byte k). This is the reference's fp32_vec_to_e2m1 inline asm
    (quantization.cuh L310-335) verbatim: cvt.rn.satfinite.e2m1x2.f32 packs
    its FIRST operand into the high nibble, so byte k = ($1=x_{2k} low,
    $2=x_{2k+1} high). Emitting the same instruction sequence keeps the
    payload bitwise-identical to quantize_with_block_size by construction
    and avoids a register->local-memory round-trip that a sub-byte rmem
    fragment + pointer recast would incur."""
    args = [cutlass.Float32(v).ir_value(loc=loc, ip=ip) for v in (x0, x1, x2, x3, x4, x5, x6, x7)]
    res = _llvm.inline_asm(
        cutlass.Int32.mlir_type,
        args,
        "{\n"
        ".reg .b8 byte0;\n"
        ".reg .b8 byte1;\n"
        ".reg .b8 byte2;\n"
        ".reg .b8 byte3;\n"
        "cvt.rn.satfinite.e2m1x2.f32 byte0, $2, $1;\n"
        "cvt.rn.satfinite.e2m1x2.f32 byte1, $4, $3;\n"
        "cvt.rn.satfinite.e2m1x2.f32 byte2, $6, $5;\n"
        "cvt.rn.satfinite.e2m1x2.f32 byte3, $8, $7;\n"
        "mov.b32 $0, {byte0, byte1, byte2, byte3};\n"
        "}",
        "=r,f,f,f,f,f,f,f,f",
    )
    return cutlass.Int32(res)


@cute.arch.dsl_user_op
def _div_full_f32(a, b, *, loc=None, ip=None):
    """Full-range approximate fp32 division (~2 ulp). This is the exact
    lowering the certified deferred-SFC producer uses for its 6/a
    normalization (`6.0 / a` -> div.full.f32); pinned via PTX so the
    deferred payload stays bitwise-identical to the certified codec
    regardless of compiler -prec-div defaults. NOT rcp.approx and NOT
    div.rn: a 1-ulp difference in the reciprocal flips e2m1 RNE tie
    values (~0.1% of payload bytes, measured and rejected)."""
    args = [
        cutlass.Float32(a).ir_value(loc=loc, ip=ip),
        cutlass.Float32(b).ir_value(loc=loc, ip=ip),
    ]
    res = _llvm.inline_asm(
        cutlass.Float32.mlir_type,
        args,
        "div.full.f32 $0, $1, $2;",
        "=f,f,f",
    )
    return cutlass.Float32(res)


@cute.jit
def cta_reduce_multi(vals, num_warps: cutlass.Constexpr, tidx: cutlass.Int32):
    """CTA-reduce N fp32 values in a SINGLE barrier round (the N-value
    analogue of reduce.py's cta_reduce_sum): one smem trip + two syncs for
    all sums instead of two syncs per sum. N is a trace-time constant (the
    python tuple length)."""
    n = len(vals)
    stride = num_warps + 1
    smem = cutlass.utils.SmemAllocator()
    acc = smem.allocate_tensor(cutlass.Float32, n * stride)
    warp_id = tidx >> 5
    lane_id = tidx & 31
    if lane_id == 0:
        for i in cutlass.range_constexpr(n):
            acc[i * stride + warp_id] = vals[i]
    cute.arch.sync_threads()
    if warp_id == 0:
        for i in cutlass.range_constexpr(n):
            v = acc[i * stride + lane_id] if lane_id < num_warps else cutlass.Float32(0)
            v = warp_reduce_sum(v)
            if lane_id == 0:
                acc[i * stride + num_warps] = v
    cute.arch.sync_threads()
    return tuple(acc[i * stride + num_warps] for i in range(n))


@cute.arch.dsl_user_op
def _f32_to_e4m3_byte(a, *, loc=None, ip=None):
    """fp32 -> e4m3 byte code via cvt.rn.satfinite.e4m3x2.f32 - the exact
    instruction the reference __nv_fp8_e4m3 cast emits (quantization.cuh
    L500)."""
    args = [cutlass.Float32(a).ir_value(loc=loc, ip=ip)]
    res = _llvm.inline_asm(
        cutlass.Int32.mlir_type,
        args,
        "{\n.reg .b16 t;\n.reg .b32 w;\ncvt.rn.satfinite.e4m3x2.f32 t, $1, $1;\n"
        "cvt.u32.u16 w, t;\nand.b32 $0, w, 255;\n}",
        "=r,f",
    )
    return cutlass.Int32(res)


@cute.arch.dsl_user_op
def _e4m3_byte_to_f32(b, *, loc=None, ip=None):
    """e4m3 byte code -> fp32 (exact: e4m3 -> f16 -> f32, both lossless),
    matching the reference's float(sf_fp8) decode."""
    args = [cutlass.Int32(b).ir_value(loc=loc, ip=ip)]
    res = _llvm.inline_asm(
        cutlass.Float32.mlir_type,
        args,
        "{\n.reg .b16 t, lo, hi;\n.reg .b32 h2;\ncvt.u16.u32 t, $1;\n"
        "cvt.rn.f16x2.e4m3x2 h2, t;\nmov.b32 {lo, hi}, h2;\ncvt.f32.f16 $0, lo;\n}",
        "=f,r",
    )
    return cutlass.Float32(res)


@cute.arch.dsl_user_op
def _round_bf16_rn_f32(a, *, loc=None, ip=None):
    """fp32 -> bf16 (RNE) -> fp32 round-trip pinned via PTX. A plain
    to(bf16).to(f32) SSA chain (and even a bf16 rmem store/load round-trip)
    gets folded away by the compiler, silently deleting the bf16 math
    mode's mid-chain narrows (measured: residual_out drifted up to 4 bf16
    ulps from eager); the asm boundary makes the rounding irreducible."""
    args = [cutlass.Float32(a).ir_value(loc=loc, ip=ip)]
    res = _llvm.inline_asm(
        cutlass.Float32.mlir_type,
        args,
        "{\n.reg .b16 t;\ncvt.rn.bf16.f32 t, $1;\ncvt.f32.bf16 $0, t;\n}",
        "=f,f",
    )
    return cutlass.Float32(res)


@cute.arch.dsl_user_op
def _mul_rn_f32(a, b, *, loc=None, ip=None):
    """Separately-rounded fp32 multiply (mul.rn.f32), pinned via PTX so the
    compiler cannot contract it with the following residual add into an FMA.
    Eager torch computes `x * gate` and `residual + (...)` as two rounded
    ops; FMA contraction flips ~2e-5 of residual_out bf16 values by one ulp
    (measured 19/817k at D=3072) and would break the op's bitwise
    residual_out claim."""
    args = [
        cutlass.Float32(a).ir_value(loc=loc, ip=ip),
        cutlass.Float32(b).ir_value(loc=loc, ip=ip),
    ]
    res = _llvm.inline_asm(
        cutlass.Float32.mlir_type,
        args,
        "mul.rn.f32 $0, $1, $2;",
        "=f,f,f",
    )
    return cutlass.Float32(res)


@cute.jit
def _nvfp4_scale_of_gmax(mRawSF, s_val, gmax):
    """Per-16-element-group scale derivation, no memory side effects.
    Returns (oscale, sf_byte, raw): static form follows quantization.cuh
    with the reference's approximate reciprocals VERBATIM (L497/L504);
    deferred form is the div.full.f32-pinned 6/a with raw = a/6."""
    zero_f32 = cutlass.Float32(0.0)
    if cutlass.const_expr(isinstance(mRawSF, cute.Tensor)):
        raw = gmax * cutlass.Float32(1.0 / 6.0)
        oscale_raw = _div_full_f32(cutlass.Float32(6.0), gmax)
        oscale = oscale_raw if gmax > zero_f32 else zero_f32
        return oscale, cutlass.Int32(0), raw
    rcp6 = cute.arch.rcp_approx(cutlass.Float32(6.0))
    sf_f32 = s_val * (gmax * rcp6)
    sf_byte = _f32_to_e4m3_byte(sf_f32)  # cvt.rn.satfinite (reference cast)
    sf_dec = _e4m3_byte_to_f32(sf_byte)
    oscale_raw = cute.arch.rcp_approx(sf_dec * cute.arch.rcp_approx(s_val))
    oscale = oscale_raw if gmax != zero_f32 else zero_f32
    return oscale, sf_byte, zero_f32


@cute.jit
def _nvfp4_store_scale(mSF, mRawSF, sf_byte, raw, row, kidx):
    """Store one group's scale: raw fp32 [M, D/16] row-major (deferred) or
    the swizzled-128x4 e4m3 SF byte (static, get_sf_out_offset_128x4)."""
    if cutlass.const_expr(isinstance(mRawSF, cute.Tensor)):
        mRawSF[row, kidx] = raw
    else:
        sf_off = (
            ((kidx >> 2) << 9)  # kTile * 512
            + ((row & 31) << 4)  # outerM * 16
            + (((row & 127) >> 5) << 2)  # innerM * 4
            + (kidx & 3)  # innerK
        )
        mSF[row >> 7, sf_off] = sf_byte.to(cutlass.Uint8)


@cute.jit
def _pack8(tYrY, base: cutlass.Constexpr, oscale):
    """Pack elements [base, base+8) of the fragment, scaled, into one int32
    of e2m1 codes (element 2k in the low nibble of byte k - exactly the
    fp32_vec_to_e2m1 packing). Scalar fp32 multiplies mirror the
    reference's per-element `fp2Vals[i] *= outputScale`."""
    return _fp32x8_to_e2m1x8_packed(
        tYrY[base + 0].to(cutlass.Float32) * oscale,
        tYrY[base + 1].to(cutlass.Float32) * oscale,
        tYrY[base + 2].to(cutlass.Float32) * oscale,
        tYrY[base + 3].to(cutlass.Float32) * oscale,
        tYrY[base + 4].to(cutlass.Float32) * oscale,
        tYrY[base + 5].to(cutlass.Float32) * oscale,
        tYrY[base + 6].to(cutlass.Float32) * oscale,
        tYrY[base + 7].to(cutlass.Float32) * oscale,
    )


@cute.jit
def _amax8(tYrY, base: cutlass.Constexpr):
    """Thread-local amax over 8 fragment values. The reference computes
    abs/max in bf16 then widens the final max to fp32; abs and max are
    exact in both precisions, so an fp32 max over exactly-widened values
    is identical."""
    amax = cutlass.Float32(0.0)
    for i in range(8):
        xf = tYrY[base + i].to(cutlass.Float32)
        amax = cute.arch.fmax(amax, cute.arch.fmax(xf, -xf))
    return amax


@cute.jit
def _nvfp4_quant_store(
    tYrY,  # bf16 rmem fragment: this thread's `vec` contiguous row elements
    mOut4,  # [B, S, D//8] Int32 gmem tensor (uint8 payload viewed as int32)
    mSF,  # static: [ceil(M/128), 8*D] Uint8 swizzled SF bytes; sentinel otherwise
    mRawSF,  # deferred: [M, D//16] Float32 raw block scales a/6; sentinel otherwise
    s_val,  # Float32: SFScaleVal (static global scale; 1.0 sentinel when deferred)
    vec: cutlass.Constexpr,
    tidx,
    bidx,
    bidy,
    row,  # flattened row index (b * S + s)
):
    """Quantize one thread's `vec` bf16 values to NVFP4 and store the
    payload plus per-16-group scales. vec=8 pairs adjacent lanes for the
    group amax (shfl.bfly xor 1); vec>=16 owns whole groups in-thread (max
    is exactly associative, so the group amax - and thus every scale and
    payload byte - is bitwise-identical across vec)."""
    if cutlass.const_expr(vec == 8):
        amax = _amax8(tYrY, 0)
        # All 32 lanes of every warp reach this call, so the full mask is safe.
        gmax = cute.arch.fmax(amax, cute.arch.shuffle_sync_bfly(amax, offset=1))
        oscale, sf_byte, raw = _nvfp4_scale_of_gmax(mRawSF, s_val, gmax)
        # One scale store per 16-element group (even lane of each pair).
        if tidx % 2 == 0:
            _nvfp4_store_scale(mSF, mRawSF, sf_byte, raw, row, tidx >> 1)
        mOut4[bidx, bidy, tidx] = _pack8(tYrY, 0, oscale)
    else:
        for g in cutlass.range_constexpr(vec // 16):
            gmax = cute.arch.fmax(_amax8(tYrY, 16 * g), _amax8(tYrY, 16 * g + 8))
            kidx = tidx * (vec // 16) + g
            oscale, sf_byte, raw = _nvfp4_scale_of_gmax(mRawSF, s_val, gmax)
            _nvfp4_store_scale(mSF, mRawSF, sf_byte, raw, row, kidx)
            mOut4[bidx, bidy, 2 * kidx] = _pack8(tYrY, 16 * g, oscale)
            mOut4[bidx, bidy, 2 * kidx + 1] = _pack8(tYrY, 16 * g + 8, oscale)


class NormProducer:
    """Row-per-CTA fused (gate/residual) + Norm + modulate + store kernel.

    Grid: one CTA per (batch, seq) token row; block: num_warps * 32 threads;
    each thread owns `vec` consecutive elements of the D row (128-bit
    vectorized gmem copies via cute.autovec_copy). ALL gmem loads are issued
    before the reduction so modulator loads (L2-resident for per-batch rows
    and [D] tables) overlap the CTA reduces.

    Compile-time variant axes: norm_type ("layer"/"rms"), math_mode
    ("fp32"/"bf16"), store ("bf16"/"nvfp4_static"/"nvfp4_deferred"), plus
    the operand-presence axes (gate/residual/weight/bias/tables/second
    output) which specialize through const_expr isinstance checks on the
    scalar sentinels.
    """

    @classmethod
    def make_hash_key(cls, *inputs: torch.Tensor | int | float) -> tuple[object, ...]:
        def _sig(val: torch.Tensor | int | float) -> object:
            if isinstance(val, torch.Tensor):
                return (val.dtype, val.ndim, val.shape[-1])
            return val

        return tuple(_sig(val) for val in inputs)

    def __init__(
        self,
        D: int,
        vec: int = DEFAULT_VEC,
        norm_type: str = "layer",
        math_mode: str = "fp32",
        store: str = "bf16",
        rows: int = 1,
        min_blocks: int = 0,
        use_pdl: bool = TRTLLM_ENABLE_PDL,
    ) -> None:
        assert D % (WARP_SIZE * vec) == 0, f"D={D} must be a multiple of {WARP_SIZE * vec}"
        assert rows in (1, 2, 4), f"rows={rows} not supported (1, 2 or 4)"
        assert norm_type in NORM_TYPES
        assert math_mode in MATH_MODES
        assert store in STORE_FORMS
        if store != "bf16":
            # The quant epilogue pairs adjacent lanes (vec=8) or owns whole
            # 16-element groups in-thread (vec=16/32).
            assert vec in (8, 16, 32), f"{store} store requires vec in (8,16,32), got {vec}"
        self.D = D
        self.vec = vec
        self.norm_type = norm_type
        self.math_mode = math_mode
        self.store = store
        self.rows = rows
        self.min_blocks = min_blocks
        self.use_pdl = use_pdl
        self.num_warps = D // (WARP_SIZE * vec)
        # cta_reduce_sum gathers the per-warp partials with a SINGLE warp
        # (one lane per partial), so > WARP_SIZE warps would silently drop
        # partials. The dispatch guards (D <= 8192, vec in (8, 16, 32))
        # already imply this; assert it here so a direct construction that
        # relaxes either bound fails loudly instead of corrupting the reduce.
        assert self.num_warps <= WARP_SIZE, (
            f"num_warps={self.num_warps} exceeds {WARP_SIZE}: cta_reduce_sum "
            f"reduces at most one partial per lane of warp 0 (D={D}, vec={vec})"
        )
        self.num_threads = self.num_warps * WARP_SIZE

    def _auto_min_blocks(self) -> int:
        """CTAs/SM floor for the register allocator. Measured on B200
        (D=5120/3072/4096 sweeps): 4 CTAs/SM is the sweet spot whenever the
        block size allows it (283 vs 334 us at D=5120 vec=16; 244 vs 268 us
        at D=3072 vec=8), 5+ spills; NVFP4 stores at wide blocks keep the
        entry-measured floor of 3 (3->2 CTA/SM was a 73.7%->42.9% DRAM
        cliff); otherwise leave the allocator alone."""
        if self.min_blocks > 0:
            return self.min_blocks
        if self.num_threads * 4 <= 2048:
            return 4
        return 3 if self.store != "bf16" else 1

    @cute.jit
    def __call__(
        self,
        mY,
        mY2,
        mOut4,
        mSF,
        mRawSF,
        mGS,
        mResOut,
        mRes,
        mX,
        mGateC,
        mTGate,
        mWeight,
        mBias,
        mShiftC,
        mScaleC,
        mTShift,
        mTScale,
        mShiftC2,
        mScaleC2,
        mTShift2,
        mTScale2,
        eps: cutlass.Float32 = cutlass.Float32(1e-6),
        stream: cuda.CUstream = cuda.CUstream(cuda.CUstream_flags.CU_STREAM_DEFAULT),
    ):
        B, S, _ = mX.shape
        atom_copy = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            mX.element_type,
            num_bits_per_copy=128,
        )
        t_layout = cute.make_layout(self.num_threads)
        v_layout = cute.make_layout(self.vec)
        tiled_copy = cute.make_tiled_copy_tv(atom_copy, t_layout, v_layout)

        self.kernel(
            mY,
            mY2,
            mOut4,
            mSF,
            mRawSF,
            mGS,
            mResOut,
            mRes,
            mX,
            mGateC,
            mTGate,
            mWeight,
            mBias,
            mShiftC,
            mScaleC,
            mTShift,
            mTScale,
            mShiftC2,
            mScaleC2,
            mTShift2,
            mTScale2,
            tiled_copy,
            eps,
        ).launch(
            grid=[(B * S) // self.rows, 1, 1],
            block=[self.num_threads, 1, 1],
            # Measured on B200 (fp4-static epilogue): without a bound the
            # epilogue's extra registers drop the kernel from 3 to 2 CTAs/SM
            # (theoretical occupancy 93.75% -> 62.5%, DRAM 73.7% -> 42.9%).
            # min_blocks_per_mp=3 restores the bf16 form's occupancy (same
            # idea as quantize_with_block_size's __launch_bounds__(512, 4)).
            min_blocks_per_mp=self._auto_min_blocks(),
            use_pdl=self.use_pdl,
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mY,
        mY2,
        mOut4,
        mSF,
        mRawSF,
        mGS,
        mResOut,
        mRes,
        mX,
        mGateC,
        mTGate,
        mWeight,
        mBias,
        mShiftC,
        mScaleC,
        mTShift,
        mTScale,
        mShiftC2,
        mScaleC2,
        mTShift2,
        mTScale2,
        tiled_copy: cute.TiledCopy,
        eps: cutlass.Float32,
    ):
        B, S, _ = mX.shape
        R = self.rows
        tidx, _, _ = cute.arch.thread_idx()
        bid, _, _ = cute.arch.block_idx()
        thr_copy = tiled_copy.get_slice(tidx)

        # R consecutive token rows per CTA: amortizes the CTA-reduction
        # barriers and the L2-broadcast modulator loads across rows and
        # deepens per-thread ILP (both measured as the top stalls of the
        # R=1 form at D=5120).
        rows = []
        bxs = []
        bys = []
        for r in cutlass.range_constexpr(R):
            rw = cutlass.Int32(bid * R + r)
            rows.append(rw)
            bxs.append(cutlass.Int32(rw // S))
            bys.append(cutlass.Int32(rw % S))

        if cutlass.const_expr(self.use_pdl):
            griddepcontrol_wait()

        @cute.jit
        def slice_at(mV, bx, by):
            """[B,S,D] token tensor (any D-divisible outer strides), [B,D]
            per-batch row (broadcast over S), or [D] row -> per-thread gmem
            tile + rmem fragment. local_tile preserves base-pointer
            alignment for vectorized loads."""
            if cutlass.const_expr(isinstance(mV, cute.Tensor)):
                if cutlass.const_expr(len(mV.shape) == 1):
                    gV = mV
                elif cutlass.const_expr(len(mV.shape) == 2):
                    gV = cute.local_tile(mV, tiler=(1, self.D), coord=(bx, 0))
                    gV = gV[0, None]
                else:
                    gV = cute.local_tile(mV, tiler=(1, 1, self.D), coord=(bx, by, 0))
                    gV = gV[0, 0, None]
                tVgV = thr_copy.partition_S(gV)
                tVrV = cute.make_fragment_like(tVgV, tVgV.element_type)
                return tVgV, tVrV
            return mV, mV

        @cute.jit
        def copy_if(src, dst):
            if cutlass.const_expr(isinstance(src, cute.Tensor) and isinstance(dst, cute.Tensor)):
                cute.autovec_copy(src, dst)

        fp32_mode = cutlass.const_expr(self.math_mode == "fp32")

        # Shared [D] table operands: sliced/loaded once for all R rows.
        tWg, tWr = slice_at(mWeight, 0, 0)
        tBg, tBr = slice_at(mBias, 0, 0)
        tTGg, tTGr = slice_at(mTGate, 0, 0)
        tTSHg, tTSHr = slice_at(mTShift, 0, 0)
        tTSCg, tTSCr = slice_at(mTScale, 0, 0)
        tTSH2g, tTSH2r = slice_at(mTShift2, 0, 0)
        tTSC2g, tTSC2r = slice_at(mTScale2, 0, 0)

        # Per-row operands.
        pX = [slice_at(mX, bxs[r], bys[r]) for r in range(R)]
        pResid = [slice_at(mRes, bxs[r], bys[r]) for r in range(R)]
        pG = [slice_at(mGateC, bxs[r], bys[r]) for r in range(R)]
        pSH = [slice_at(mShiftC, bxs[r], bys[r]) for r in range(R)]
        pSC = [slice_at(mScaleC, bxs[r], bys[r]) for r in range(R)]
        pSH2 = [slice_at(mShiftC2, bxs[r], bys[r]) for r in range(R)]
        pSC2 = [slice_at(mScaleC2, bxs[r], bys[r]) for r in range(R)]
        pRO = [slice_at(mResOut, bxs[r], bys[r]) for r in range(R)]
        pY = [slice_at(mY, bxs[r], bys[r]) for r in range(R)]
        pY2 = [slice_at(mY2, bxs[r], bys[r]) for r in range(R)]

        frag_t = pX[0][1]  # layout template for fp32 scratch fragments

        @cute.jit
        def round_narrow(v):
            """One eager-op-boundary rounding (bf16 math mode only): round
            the fp32 vector through the x dtype and widen back. torch eager
            elementwise bf16 ops compute in fp32 and round per op; this
            replays those narrows at the same points. Pinned per element
            via _round_bf16_rn_f32 - plain conversion chains get folded
            away (see that helper's docstring)."""
            if cutlass.const_expr(fp32_mode):
                return v
            tmp = cute.make_fragment_like(frag_t, cutlass.Float32)
            tmp.store(v)
            out = cute.make_fragment_like(frag_t, cutlass.Float32)
            for i in range(cute.size(out)):
                out[i] = _round_bf16_rn_f32(tmp[i])
            return out.load()

        @cute.jit
        def combine(tCr, tTr):
            """Modulator compose: optional [D] fp32 table + optional ts row
            (bf16 or fp32; per-batch or per-token chunk view) -> fp32 vector.
            fp32 mode composes in fp32 (WAN law: table + chunk.float());
            bf16 mode narrows the table to bf16 first, then adds in one
            rounded bf16 step (LTX law: (table.to(bf16) + ts))."""
            if cutlass.const_expr(isinstance(tTr, cute.Tensor) and isinstance(tCr, cute.Tensor)):
                if cutlass.const_expr(fp32_mode):
                    return tTr.load() + tCr.load().to(cutlass.Float32)
                tb = round_narrow(tTr.load())
                return round_narrow(tb + tCr.load().to(cutlass.Float32))
            if cutlass.const_expr(isinstance(tCr, cute.Tensor)):
                return tCr.load().to(cutlass.Float32)
            return tTr.load().to(cutlass.Float32)

        # Issue ALL gmem loads up front: x/residual rows first (they feed
        # the reductions), then the L2-resident modulator rows and tables,
        # so every load overlaps the stats reductions.
        for r in cutlass.range_constexpr(R):
            copy_if(*pX[r])
            copy_if(*pResid[r])
        for r in cutlass.range_constexpr(R):
            copy_if(*pG[r])
            copy_if(*pSH[r])
            copy_if(*pSC[r])
            copy_if(*pSH2[r])
            copy_if(*pSC2[r])
        copy_if(tTGg, tTGr)
        copy_if(tWg, tWr)
        copy_if(tBg, tBr)
        copy_if(tTSHg, tTSHr)
        copy_if(tTSCg, tTSCr)
        copy_if(tTSH2g, tTSH2r)
        copy_if(tTSC2g, tTSC2r)

        # Optional gate + residual prologue per row. fp32 mode: accumulate
        # in fp32, ONE rounding to the residual dtype (matches the eager
        # `.to(x.dtype)` before the norm). bf16 mode: replay the eager bf16
        # narrows after the gate multiply and after the residual add.
        ln_srcs = []
        for r in cutlass.range_constexpr(R):
            tXrX = pX[r][1]
            tRrR = pResid[r][1]
            tGrG = pG[r][1]
            if cutlass.const_expr(isinstance(tRrR, cute.Tensor)):
                v = tXrX.load().to(cutlass.Float32)
                if cutlass.const_expr(
                    isinstance(tGrG, cute.Tensor) or isinstance(tTGr, cute.Tensor)
                ):
                    g = combine(tGrG, tTGr)
                    if cutlass.const_expr(fp32_mode):
                        # Pin the gate multiply to mul.rn.f32 (_mul_rn_f32):
                        # eager rounds x*gate and residual+(..) separately.
                        tGf = cute.make_fragment_like(frag_t, cutlass.Float32)
                        tGf.store(g)
                        tPf = cute.make_fragment_like(frag_t, cutlass.Float32)
                        for i in range(cute.size(tPf)):
                            tPf[i] = _mul_rn_f32(tXrX[i].to(cutlass.Float32), tGf[i])
                        v = tPf.load()
                    else:
                        v = round_narrow(v * g)
                v = tRrR.load().to(cutlass.Float32) + v
                tROrRO = pRO[r][1]
                tROrRO.store(v.to(tROrRO.element_type))
                copy_if(tROrRO, pRO[r][0])
                ln_srcs.append(tROrRO)  # norm reads the ROUNDED residual
            else:
                ln_srcs.append(tXrX)

        # Stats for all R rows in ONE CTA-reduction round.
        partials = []
        for r in cutlass.range_constexpr(R):
            if cutlass.const_expr(self.norm_type == "layer"):
                s = cute.Float32(0.0)
                ss = cute.Float32(0.0)
                for idx in range(cute.size(ln_srcs[r])):
                    xf = ln_srcs[r][idx].to(cutlass.Float32)
                    s += xf
                    ss += xf * xf
                partials.append(warp_reduce_sum(s))
                partials.append(warp_reduce_sum(ss))
            else:
                ss = cute.Float32(0.0)
                for idx in range(cute.size(ln_srcs[r])):
                    xf = ln_srcs[r][idx].to(cutlass.Float32)
                    ss += xf * xf
                partials.append(warp_reduce_sum(ss))
        totals = cta_reduce_multi(tuple(partials), self.num_warps, tidx)
        stats = []
        for r in cutlass.range_constexpr(R):
            if cutlass.const_expr(self.norm_type == "layer"):
                # var = E[x^2] - mean^2, the same single-pass scheme as the
                # fusedAdaptiveLayerNorm CUDA kernel (a second reduction
                # round measured as the top stall of the two-pass form).
                mean = totals[2 * r] / self.D
                var = totals[2 * r + 1] / self.D - mean * mean
                stats.append((mean, cute.rsqrt(var + eps)))
            else:
                stats.append((cutlass.Float32(0.0), cute.rsqrt(totals[r] / self.D + eps)))

        @cute.jit
        def modulate(n, tSCr, tTSCrL, tSHr, tTSHrL):
            """y_i = n * (1 + scale_i) + shift_i. fp32 mode: all fp32.
            bf16 mode: rounded after (1 + scale), after the multiply and
            after the add - the eager op boundaries."""
            v = n
            if cutlass.const_expr(isinstance(tSCr, cute.Tensor) or isinstance(tTSCrL, cute.Tensor)):
                v = round_narrow(v * round_narrow(1.0 + combine(tSCr, tTSCrL)))
            if cutlass.const_expr(isinstance(tSHr, cute.Tensor) or isinstance(tTSHrL, cute.Tensor)):
                v = round_narrow(v + combine(tSHr, tTSHrL))
            return v

        for r in cutlass.range_constexpr(R):
            mean, rstd = stats[r]
            # Normalize + optional affine in fp32 (torch LayerNorm/F.rms_norm
            # compute fp32 internally); bf16 mode then rounds ONCE here (the
            # norm's own output narrow) before the bf16 modulate.
            n = (ln_srcs[r].load().to(cutlass.Float32) - mean) * rstd
            if cutlass.const_expr(isinstance(tWr, cute.Tensor)):
                n = n * tWr.load().to(cutlass.Float32)
            if cutlass.const_expr(isinstance(tBr, cute.Tensor)):
                n = n + tBr.load().to(cutlass.Float32)
            n = round_narrow(n)

            y = modulate(n, pSC[r][1], tTSCr, pSH[r][1], tTSHr)

            tYrY = pY[r][1]
            if cutlass.const_expr(self.store == "bf16"):
                # ONE rounding at the final store (fp32 mode); bf16 mode's y
                # is already bf16-valued so this narrows losslessly.
                tYrY.store(y.to(tYrY.element_type))
                copy_if(tYrY, pY[r][0])
            else:
                # Round y to the exact bf16 values fp4_quantize would read,
                # then quantize + store NVFP4 instead of storing bf16.
                tYb = cute.make_fragment_like(frag_t, mX.element_type)
                tYb.store(y.to(mX.element_type))
                if cutlass.const_expr(isinstance(mGS, cute.Tensor)):
                    s_val = mGS[0]
                else:
                    s_val = cutlass.Float32(1.0)
                _nvfp4_quant_store(
                    tYb, mOut4, mSF, mRawSF, s_val, self.vec, tidx, bxs[r], bys[r], rows[r]
                )
                if cutlass.const_expr(self.store == "nvfp4_static"):
                    # Pad-row SF zeroing (quantization.cuh L831-859): rows
                    # padded to a multiple of 128 in the swizzled SF layout;
                    # the CTA covering row p (p < pad count) zeroes padding
                    # row M + p. The host wrapper guarantees M >= pad count.
                    M = cutlass.Int32(B * S)
                    padM = cutlass.Int32(((M + 127) >> 7) << 7)
                    if rows[r] < padM - M:
                        n_groups = self.D // 16
                        for pz in cutlass.range_constexpr(
                            (n_groups + self.num_threads - 1) // self.num_threads
                        ):
                            gi = pz * self.num_threads + tidx
                            if gi < n_groups:
                                rp = M + rows[r]
                                pad_off = (
                                    ((gi >> 2) << 9)
                                    + ((rp & 31) << 4)
                                    + (((rp & 127) >> 5) << 2)
                                    + (gi & 3)
                                )
                                mSF[rp >> 7, pad_off] = cutlass.Uint8(0)

            # Optional second modulated output (dual form), always bf16.
            tY2r = pY2[r][1]
            if cutlass.const_expr(isinstance(tY2r, cute.Tensor)):
                y2 = modulate(n, pSC2[r][1], tTSC2r, pSH2[r][1], tTSH2r)
                tY2r.store(y2.to(tY2r.element_type))
                copy_if(tY2r, pY2[r][0])

        if cutlass.const_expr(self.use_pdl):
            griddepcontrol_launch_dependents()


# ---------------------------------------------------------------------------
# Validation + dispatch
# ---------------------------------------------------------------------------
_X_DTYPES = (torch.bfloat16,)
_ROW_DTYPES = (torch.bfloat16, torch.float32)


def _validate_bsd(
    t: torch.Tensor,
    B: int,
    S: int,
    D: int,
    name: str,
    dtypes: tuple = _X_DTYPES,
    device: Optional[torch.device] = None,
) -> None:
    if t.dtype not in dtypes:
        raise ValueError(f"{name}: unsupported dtype {t.dtype} (expected one of {dtypes})")
    if not t.is_cuda:
        raise ValueError(f"{name}: expected a CUDA tensor, got device {t.device}")
    if device is not None and t.device != device:
        raise ValueError(f"{name}: expected device {device}, got {t.device}")
    if t.shape != (B, S, D):
        raise ValueError(f"{name}: expected shape {(B, S, D)}, got {tuple(t.shape)}")
    if t.stride(-1) != 1:
        raise ValueError(f"{name}: last dim must be contiguous (stride(-1)==1)")
    for i in range(t.ndim - 1):
        if t.stride(i) % D != 0:
            raise ValueError(f"{name}: stride({i})={t.stride(i)} not divisible by D={D}")
    if t.data_ptr() % _REQUIRED_ALIGNMENT != 0:
        raise ValueError(f"{name}: data pointer must be {_REQUIRED_ALIGNMENT}-byte aligned")


def _validate_table(t: Optional[torch.Tensor], D: int, name: str, device: torch.device) -> None:
    if t is None:
        return
    if t.dtype != torch.float32:
        raise ValueError(f"{name}: expected fp32, got {t.dtype}")
    if not t.is_cuda or t.device != device:
        raise ValueError(f"{name}: expected device {device}, got {t.device}")
    if t.shape != (D,):
        raise ValueError(f"{name}: expected shape ({D},), got {tuple(t.shape)}")
    if t.stride(-1) != 1:
        raise ValueError(f"{name}: must be contiguous")
    if t.data_ptr() % _REQUIRED_ALIGNMENT != 0:
        raise ValueError(f"{name}: data pointer must be {_REQUIRED_ALIGNMENT}-byte aligned")


def _normalize_ts(
    t: Optional[torch.Tensor],
    B: int,
    S: int,
    D: int,
    name: str,
    device: torch.device,
    math_mode: str,
) -> Optional[torch.Tensor]:
    """Accept a per-token [B, S, D] chunk view, a per-batch [B, D] /
    [B, 1, D] / [1, 1, D] row, or a [D]-broadcast [1, D] row; return a
    kernel-ready [B, S, D] or [B, D] view (fail-closed on anything else)."""
    if t is None:
        return None
    if t.dtype not in _ROW_DTYPES:
        raise ValueError(f"{name}: unsupported dtype {t.dtype}")
    if math_mode == "bf16" and t.dtype != torch.bfloat16:
        raise ValueError(f"{name}: bf16 math mode requires bf16 modulator rows, got {t.dtype}")
    if not t.is_cuda or t.device != device:
        raise ValueError(f"{name}: expected device {device}, got {t.device}")
    if t.ndim == 3 and t.shape == (B, S, D):
        pass  # per-token chunk view
    else:
        if t.ndim == 3:
            if t.shape[1] != 1:
                raise ValueError(f"{name}: expected [B, S, D] or [B, 1, D], got {tuple(t.shape)}")
            t = t.squeeze(1)
        if t.ndim != 2:
            raise ValueError(f"{name}: expected 2D/3D modulator, got ndim={t.ndim}")
        if t.shape[-1] != D:
            raise ValueError(f"{name}: expected last dim {D}, got {t.shape[-1]}")
        if t.shape[0] == 1 and B > 1:
            t = t.expand(B, D)  # stride(0)==0 is D-divisible; kernel broadcasts
        if t.shape[0] != B:
            raise ValueError(f"{name}: batch {t.shape[0]} != x batch {B}")
    if t.stride(-1) != 1:
        raise ValueError(f"{name}: last dim must be contiguous (stride(-1)==1)")
    for i in range(t.ndim - 1):
        if t.stride(i) % D != 0:
            raise ValueError(f"{name}: stride({i})={t.stride(i)} not divisible by D={D}")
    if t.data_ptr() % _REQUIRED_ALIGNMENT != 0:
        raise ValueError(f"{name}: data pointer must be {_REQUIRED_ALIGNMENT}-byte aligned")
    return t


def _check_d(D: int, vec: int) -> None:
    if vec not in (8, 16, 32):
        raise ValueError(f"vec={vec} not supported: expected 8, 16 or 32")
    if D <= 0 or D % 256 != 0 or D > 8192:
        raise ValueError(f"D={D} not supported: must be a multiple of 256 and <= 8192")
    if D % (WARP_SIZE * vec) != 0:
        raise ValueError(f"D={D} not divisible by {WARP_SIZE * vec} (vec={vec})")


def _launch(
    torch_tensors: list,
    D: int,
    vec: int,
    norm_type: str,
    math_mode: str,
    store: str,
    rows: int,
    mb: int,
    eps: float,
    device: torch.device,
) -> None:
    with torch.cuda.device(device):
        stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
        hash_key = (
            device,
            vec,
            norm_type,
            math_mode,
            store,
            rows,
            mb,
            *NormProducer.make_hash_key(*torch_tensors),
        )
        compiled_fn = _COMPILE_CACHE.get(hash_key)
        if compiled_fn is None:
            if torch.cuda.get_device_capability(device) != (10, 0):
                raise ValueError("fused_norm_producer requires an SM100 GPU")
            kernel = NormProducer(D, vec, norm_type, math_mode, store, rows, min_blocks=mb)
            fake_sig_args = [_to_fake_cute_arg(t) for t in torch_tensors]
            compiled_fn = cute.compile(kernel, *fake_sig_args, options="--enable-tvm-ffi")
            _COMPILE_CACHE[hash_key] = compiled_fn
        compiled_fn(*torch_tensors, eps, stream)


@torch.library.custom_op(
    "trtllm::fused_norm_producer",
    mutates_args=(),
    tags=(torch.Tag.needs_fixed_stride_order,),
)
def fused_norm_producer(
    x: torch.Tensor,
    residual: Optional[torch.Tensor] = None,
    gate: Optional[torch.Tensor] = None,
    gate_table: Optional[torch.Tensor] = None,
    weight: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    shift: Optional[torch.Tensor] = None,
    scale: Optional[torch.Tensor] = None,
    shift_table: Optional[torch.Tensor] = None,
    scale_table: Optional[torch.Tensor] = None,
    shift2: Optional[torch.Tensor] = None,
    scale2: Optional[torch.Tensor] = None,
    shift2_table: Optional[torch.Tensor] = None,
    scale2_table: Optional[torch.Tensor] = None,
    global_scale: Optional[torch.Tensor] = None,
    norm_type: str = "layer",
    math_mode: str = "fp32",
    store: str = "bf16",
    eps: float = 1e-6,
    vec: int = DEFAULT_VEC,
    rows_per_cta: int = 1,
    min_blocks: int = 0,
) -> List[torch.Tensor]:
    """Unified DiT norm-site producer.

    Computes, with compile-time specialization on which operands are given::

        v            = x [* (gate_table + gate)] ;  prologue gate
        residual_out = (residual + v) -> x.dtype ;  iff residual is given
        n            = Norm(residual_out or x)   ;  "layer" (optionally
                                                    affine via weight/bias)
                                                    or weightless "rms"
        y            = n * (1 + (scale_table + scale)) + (shift_table + shift)
        y2           = n * (1 + (scale2_table + scale2)) + (shift2_table + shift2)

    and stores y per ``store``:

    - ``"bf16"``:           y as x.dtype.
    - ``"nvfp4_static"``:   (y_fp4 [B, S, D//2] uint8, y_sf [pad128(B*S) *
      (D//16)] uint8 swizzled) - payload+SF bitwise-identical to
      ``torch.ops.trtllm.fp4_quantize(y_bf16, global_scale, 16, False)`` on
      this op's own bf16 y. Requires ``global_scale`` (fp32 scalar tensor,
      the consumer Linear's static input_scale) and B*S >= 128-pad count.
    - ``"nvfp4_deferred"``: (y_fp4 [B, S, D//2] uint8, y_raw_sf [B*S, D//16]
      fp32 row-major raw block scales a/6) for DYNAMIC quantization - the
      global scale is unknown at producer time; finalize with
      ``sfc_finalize`` (K2) from this package. ``global_scale`` must be None.

    Modulators: ``gate/shift/scale`` accept per-token [B, S, D] chunk views
    (e.g. raw ``temb[:, :, i, :]`` views of a [B, S, 6, D] temb), per-batch
    [B, D] / [B, 1, D] rows, or [1, D]-broadcast rows; bf16 or fp32. The
    optional ``*_table`` operands are [D] fp32 rows composed with the
    matching row INLINE in registers (zero modulator materialization).

    ``math_mode`` selects the dtype law of the surrounding model:

    - ``"fp32"`` (WAN): all composition in fp32, ONE final rounding per
      stored tensor (residual_out and y round independently; the norm reads
      the ROUNDED residual_out, bit-matching the eager ``.to(x.dtype)``).
    - ``"bf16"`` (LTX/Qwen/FLUX narrow-first law): modulator combine is
      ``(table.to(bf16) + row)`` and every eager elementwise op boundary
      (gate mul, residual add, norm output, 1+scale, modulate mul/add)
      rounds to bf16, replaying the model's mid-chain narrows exactly.
      Requires bf16 modulator rows.

    Returns a list: ``[y]`` or ``[y_fp4, y_sf(or y_raw_sf)]``, then ``y2``
    if the second modulator set is given, then ``residual_out`` if
    ``residual`` is given.

    Contract: x/residual [B, S, D] bf16, stride(-1)==1, outer strides
    divisible by D, 32-byte aligned; D a multiple of 256, <= 8192;
    vec in (8, 16, 32) with D % (32*vec) == 0.

    Tuning knobs (compile-time; the defaults are safe everywhere and the
    tuned values are per-(D, form) measurements - see the PR bench table):

    - ``vec``: elements per thread. vec=16 with min_blocks=4 measured best
      for D=5120 forms (283 us vs 334 default at M=65520); vec=8 default is
      within ~1.5% for the gate+resid forms. NOTE: vec=16/32 with fp32
      modulator/affine operands requires nvidia-cutlass-dsl >= 4.6 (the
      in-tree pin) - 4.5.x segfaults compiling those configs (guarded).
    - ``rows_per_cta``: token rows per CTA (1, 2, 4; must divide B*S).
      R>1 amortizes the CTA-reduction barriers but raises register
      pressure; measured a regression for bf16 D=5120 and a small win for
      some NVFP4 forms - keep 1 unless a bench says otherwise.
    - ``min_blocks``: min CTAs/SM hint (0 = auto: 3 for NVFP4 stores per
      the measured 3->2 CTA/SM occupancy cliff, else 1). min_blocks=4 at
      vec=16 is the measured D=5120 sweet spot; 5+ spills.
    """
    if x.ndim != 3:
        raise ValueError(f"x: expected a 3D [B, S, D] tensor, got {x.ndim}D")
    B, S, D = x.shape
    if B == 0 or S == 0:
        raise ValueError(f"x: B and S must be nonzero, got shape {tuple(x.shape)}")
    if norm_type not in NORM_TYPES:
        raise ValueError(f"norm_type must be one of {NORM_TYPES}, got {norm_type!r}")
    if math_mode not in MATH_MODES:
        raise ValueError(f"math_mode must be one of {MATH_MODES}, got {math_mode!r}")
    if store not in STORE_FORMS:
        raise ValueError(f"store must be one of {STORE_FORMS}, got {store!r}")
    if rows_per_cta not in (1, 2, 4):
        raise ValueError(f"rows_per_cta={rows_per_cta} not supported (1, 2 or 4)")
    if not 0 <= min_blocks <= 32:
        raise ValueError(f"min_blocks={min_blocks} out of range [0, 32] (0 = auto)")
    if vec != 8 and _CUTLASS_DSL_PRE_46:
        has_f32_row = any(
            t is not None and t.dtype == torch.float32
            for t in (
                gate_table,
                weight,
                bias,
                shift_table,
                scale_table,
                shift2_table,
                scale2_table,
                gate,
                shift,
                scale,
                shift2,
                scale2,
            )
        )
        if has_f32_row:
            raise ValueError(
                "vec=16/32 with fp32 modulator/affine operands requires "
                "nvidia-cutlass-dsl >= 4.6 (4.5.x segfaults compiling these "
                "configs); use vec=8 or upgrade the DSL"
            )
    if (B * S) % rows_per_cta != 0:
        raise ValueError(
            f"rows_per_cta={rows_per_cta} must divide B*S={B * S} (pad-free row tiling)"
        )
    _check_d(D, vec)
    _validate_bsd(x, B, S, D, "x")
    if norm_type == "rms" and (weight is not None or bias is not None):
        raise ValueError("rms norm is weightless: weight/bias must be None")
    if bias is not None and weight is None:
        raise ValueError("bias requires weight")
    if (shift is None and shift_table is None) != (scale is None and scale_table is None):
        raise ValueError("shift and scale modulator sets must be passed together")
    has_out2 = any(t is not None for t in (shift2, scale2, shift2_table, scale2_table))
    if has_out2 and (
        (shift2 is None and shift2_table is None) or (scale2 is None and scale2_table is None)
    ):
        raise ValueError("shift2 and scale2 modulator sets must be passed together")
    if gate is not None or gate_table is not None:
        if residual is None:
            raise ValueError("gate requires residual (the gated-residual prologue)")
    if residual is not None:
        _validate_bsd(residual, B, S, D, "residual", device=x.device)
    gate = _normalize_ts(gate, B, S, D, "gate", x.device, math_mode)
    shift = _normalize_ts(shift, B, S, D, "shift", x.device, math_mode)
    scale = _normalize_ts(scale, B, S, D, "scale", x.device, math_mode)
    shift2 = _normalize_ts(shift2, B, S, D, "shift2", x.device, math_mode)
    scale2 = _normalize_ts(scale2, B, S, D, "scale2", x.device, math_mode)
    _validate_table(gate_table, D, "gate_table", x.device)
    _validate_table(weight, D, "weight", x.device)
    _validate_table(bias, D, "bias", x.device)
    _validate_table(shift_table, D, "shift_table", x.device)
    _validate_table(scale_table, D, "scale_table", x.device)
    _validate_table(shift2_table, D, "shift2_table", x.device)
    _validate_table(scale2_table, D, "scale2_table", x.device)

    M = B * S
    outputs: List[torch.Tensor] = []
    if store == "bf16":
        if global_scale is not None:
            raise ValueError("global_scale is only accepted with store='nvfp4_static'")
        y = torch.empty_like(x)
        m_y, m_out4, m_sf, m_rawsf, m_gs = y, 0, 0, 0, 1.0
        outputs.append(y)
    else:
        if x.dtype != torch.bfloat16:
            raise ValueError("NVFP4 stores require bf16 x (the reference recipe is bf16->fp4)")
        y_fp4 = torch.empty(B, S, D // 2, dtype=torch.uint8, device=x.device)
        m_y, m_out4 = 0, y_fp4.view(torch.int32)
        outputs.append(y_fp4)
        if store == "nvfp4_static":
            if (
                global_scale is None
                or global_scale.dtype != torch.float32
                or global_scale.numel() != 1
            ):
                raise ValueError("nvfp4_static requires global_scale (single-element fp32 tensor)")
            padM = (M + 127) // 128 * 128
            if M < padM - M:
                raise ValueError(f"B*S={M} too small for in-kernel SF pad zeroing")
            y_sf = torch.empty(padM * (D // 16), dtype=torch.uint8, device=x.device)
            m_sf, m_rawsf, m_gs = y_sf.view(padM // 128, 8 * D), 0, global_scale.reshape(1)
            outputs.append(y_sf)
        else:
            if global_scale is not None:
                raise ValueError("nvfp4_deferred computes scales itself: global_scale must be None")
            y_raw_sf = torch.empty(M, D // 16, dtype=torch.float32, device=x.device)
            m_sf, m_rawsf, m_gs = 0, y_raw_sf, 1.0
            outputs.append(y_raw_sf)

    if has_out2:
        y2 = torch.empty_like(x)
        outputs.append(y2)
    else:
        y2 = None
    if residual is not None:
        residual_out = torch.empty_like(x)
        outputs.append(residual_out)
    else:
        residual_out = None

    # Scalar placeholders for absent operands (CuTe DSL TVM-FFI backend does
    # not accept None); they generate no code (const_expr isinstance checks).
    torch_tensors = [
        m_y,
        0 if y2 is None else y2,
        m_out4,
        m_sf,
        m_rawsf,
        m_gs,
        0 if residual_out is None else residual_out,
        0 if residual is None else residual,
        x,
        1 if gate is None else gate,
        1 if gate_table is None else gate_table,
        1 if weight is None else weight,
        0 if bias is None else bias,
        0 if shift is None else shift,
        0 if scale is None else scale,
        0 if shift_table is None else shift_table,
        0 if scale_table is None else scale_table,
        0 if shift2 is None else shift2,
        0 if scale2 is None else scale2,
        0 if shift2_table is None else shift2_table,
        0 if scale2_table is None else scale2_table,
    ]
    _launch(
        torch_tensors, D, vec, norm_type, math_mode, store, rows_per_cta, min_blocks, eps, x.device
    )
    return outputs


@fused_norm_producer.register_fake
def _fused_norm_producer_fake(
    x: torch.Tensor,
    residual: Optional[torch.Tensor] = None,
    gate: Optional[torch.Tensor] = None,
    gate_table: Optional[torch.Tensor] = None,
    weight: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    shift: Optional[torch.Tensor] = None,
    scale: Optional[torch.Tensor] = None,
    shift_table: Optional[torch.Tensor] = None,
    scale_table: Optional[torch.Tensor] = None,
    shift2: Optional[torch.Tensor] = None,
    scale2: Optional[torch.Tensor] = None,
    shift2_table: Optional[torch.Tensor] = None,
    scale2_table: Optional[torch.Tensor] = None,
    global_scale: Optional[torch.Tensor] = None,
    norm_type: str = "layer",
    math_mode: str = "fp32",
    store: str = "bf16",
    eps: float = 1e-6,
    vec: int = DEFAULT_VEC,
    rows_per_cta: int = 1,
    min_blocks: int = 0,
) -> List[torch.Tensor]:
    B, S, D = x.shape
    M = B * S
    outputs = []
    if store == "bf16":
        outputs.append(torch.empty_like(x))
    else:
        outputs.append(x.new_empty((B, S, D // 2), dtype=torch.uint8))
        if store == "nvfp4_static":
            padM = (M + 127) // 128 * 128
            outputs.append(x.new_empty((padM * (D // 16),), dtype=torch.uint8))
        else:
            outputs.append(x.new_empty((M, D // 16), dtype=torch.float32))
    if any(t is not None for t in (shift2, scale2, shift2_table, scale2_table)):
        outputs.append(torch.empty_like(x))
    if residual is not None:
        outputs.append(torch.empty_like(x))
    return outputs
