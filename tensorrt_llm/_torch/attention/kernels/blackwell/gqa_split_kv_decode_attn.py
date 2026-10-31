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
"""Fused split-KV decode attention (CuTe DSL, SM100 / SM103), one launch per attention call.

Decode shape of one rank: one KV head (D = 256, FP8 E4M3 KV in 64-token pages, separate K / V page tables),
``group`` q heads, ``q_len`` tokens per request (causal among the draft tokens), bf16 output in the
[tokens, heads * D] layout o_proj reads, optionally times sigmoid(gate). For a KV-replica split use
gqa_kv_replica_partial_attn.py.

Work split: grid (n_splits, batch), one CTA per SM, all CTAs co-resident. CTA (s, r) streams a contiguous page range
of request r's KV [0, L) in 128-key tiles (2 pages); the q rows (q_len x group <= 32, padded to 16 / 32) are the MMA
N dimension ("swap AB": keys are the UMMA M):

* warp 0, TMA: block-table entries -> SMEM; per tile 4 boxes of K and 4 of V (64 keys x 128 B, 128B swizzle,
  EVICT_FIRST) into a 2-stage ring. Pages that may hold this call's new tokens, plus ``fresh_margin_pages`` below
  them (MTP passes write those earlier in the step), are loaded only after ``griddepcontrol.wait``; under PDL the
  prefix before them is requested without waiting. Warp 3 loads q after the wait.
* warp 1, MMA (tcgen05 kind::f8f6f4, FP32 in TMEM): GEMM1 S^T[128, N] = K . Q^T (A = K K-major, B = Q K-major) into
  one of two S buffers; GEMM2 O^T[256, N] += V^T . P^T (A = V MN-major, B = P^T MN-major), one tile behind GEMM1.
* warps 4-7, softmax (thread = key = TMEM lane): lazy rescale (the row reference max moves only when a score exceeds
  it by more than the threshold, so P stays within E4M3 range and O is rescaled in TMEM only then); P^T -> SMEM E4M3.
* epilogue: every CTA writes its partial (O^T unnormalized FP32 chunk-major, then (m, l) per row) to the workspace
  and arrives on its request's 64-bit epoch counter (grows by n_splits per launch, never reset). Each CTA then
  bulk-copies its chunk of every split's partial, combines, applies bmm2 and the gate and stores bf16.

Assumptions of the grid barrier (``counters`` row b, see split_kv_decode):

(a) Two launches that use the same counter row never overlap: the next launch reads the epoch only after its
    ``griddepcontrol.wait``, which must imply that the previous launch completed. True when every kernel
    between two launches on the stream either is a plain (non-PDL) launch or executes ``griddepcontrol.wait`` before
    it exits, as all kernels of the decode step do. Concurrent launches on different streams need separate rows.
(b) All n_splits x batch CTAs are co-resident: one CTA per SM and grid <= SMs (split_kv_decode asserts it), and no
    other kernel holds SMs while it waits for work ordered after this launch.
(c) Every launch on a row uses the same n_splits (split_kv_decode keeps one counter row per batch size).
"""

from __future__ import annotations

import cutlass
import cutlass.cute as cute
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass import Float32, Int32, Int64, Uint32, const_expr
from cutlass._mlir.dialects import llvm
from cutlass.cute.nvgpu import OperandMajorMode, cpasync, tcgen05
from cutlass.cutlass_dsl import T, dsl_user_op

try:
    from cutlass.utils import SmemAllocator
except ImportError:  # pragma: no cover - older CuTe DSL
    from cutlass.memory import SmemAllocator

FP8 = cutlass.Float8E4M3FN
PAGE = 64  # tokens per KV page
HEAD_DIM = 256
TILE_KEYS = 128  # keys per tile = UMMA M of GEMM1
SW_BYTES = 128  # bytes per swizzle row (one TMA box column)
EVICT_FIRST = 0x12F0000000000000  # CUTLASS TMA::CacheHintSm90::EVICT_FIRST
LOG2E = 1.4426950408889634
RESCALE_THRESHOLD = 8.0  # log2 units: P <= 2^8 = 256 < 448 (E4M3 max) without a P shift
P_TOP = 8.75  # log2 of the largest P (2^8.75 = 430 < 448): a P shift s lowers the lazy-rescale threshold to 8.75 - s
NUM_THREADS = 256  # warp 0 TMA, warp 1 MMA, warps 2-3 epilogue helpers, warps 4-7 softmax
SOFTMAX_WARP0 = 4
TAB_MAX = 512  # block-table entries per split staged in SMEM (K and V each)
BAR_SOFTMAX_RED = 1  # named barrier ids (0 = __syncthreads)
BAR_SOFTMAX = 2


def _asm(ret, args, text, cons, side_effects, loc, ip):
    return llvm.inline_asm(
        ret,
        args,
        text,
        cons,
        has_side_effects=side_effects,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _ex2(x, *, loc=None, ip=None) -> Float32:
    return Float32(
        _asm(
            T.f32(),
            [Float32(x).ir_value(loc=loc, ip=ip)],
            "ex2.approx.ftz.f32 $0, $1;",
            "=f,f",
            False,
            loc,
            ip,
        )
    )


@dsl_user_op
def _pack_e4m3x4(a, b, c, d, *, loc=None, ip=None) -> Uint32:
    """Four FP32 -> four E4M3 (cvt.rn.satfinite), ``a`` in the low byte."""
    out = _asm(
        T.i32(),
        [Float32(v).ir_value(loc=loc, ip=ip) for v in (a, b, c, d)],
        "{\n\t.reg .b16 lo, hi;\n\t"
        "cvt.rn.satfinite.e4m3x2.f32 lo, $2, $1;\n\t"
        "cvt.rn.satfinite.e4m3x2.f32 hi, $4, $3;\n\t"
        "mov.b32 $0, {lo, hi};\n\t}\n",
        "=r,f,f,f,f",
        False,
        loc,
        ip,
    )
    return Uint32(out)


@dsl_user_op
def _st_shared_v4(addr, a, b, c, d, *, loc=None, ip=None) -> None:
    _asm(
        None,
        [Int32(addr).ir_value(loc=loc, ip=ip)]
        + [Uint32(v).ir_value(loc=loc, ip=ip) for v in (a, b, c, d)],
        "st.shared.v4.b32 [$0], {$1, $2, $3, $4};",
        "r,r,r,r,r",
        True,
        loc,
        ip,
    )


@dsl_user_op
def _bar_red_or(pred, bar_id: int, nthreads: int, *, loc=None, ip=None) -> Int32:
    """Named barrier over ``nthreads`` threads that also ORs a predicate (``barrier.cta.red.or.pred``)."""
    return Int32(
        _asm(
            T.i32(),
            [Int32(pred).ir_value(loc=loc, ip=ip)],
            "{\n\t.reg .pred pi, po;\n\tsetp.ne.s32 pi, $1, 0;\n\t"
            f"barrier.cta.red.or.pred po, {bar_id}, {nthreads}, pi;\n\t"
            "selp.s32 $0, 1, 0, po;\n\t}\n",
            "=r,r",
            True,
            loc,
            ip,
        )
    )


@dsl_user_op
def _griddep_wait(*, loc=None, ip=None) -> None:
    _asm(None, [], "griddepcontrol.wait;", "", True, loc, ip)


@dsl_user_op
def _griddep_launch(*, loc=None, ip=None) -> None:
    _asm(None, [], "griddepcontrol.launch_dependents;", "", True, loc, ip)


@dsl_user_op
def _red_release_add_u64(addr, val, *, loc=None, ip=None) -> None:
    _asm(
        None,
        [Int64(addr).ir_value(loc=loc, ip=ip), Int64(val).ir_value(loc=loc, ip=ip)],
        "red.release.gpu.global.add.u64 [$0], $1;",
        "l,l",
        True,
        loc,
        ip,
    )


@dsl_user_op
def _ld_acquire_gpu_u64(addr, *, loc=None, ip=None) -> Int64:
    return Int64(
        _asm(
            T.i64(),
            [Int64(addr).ir_value(loc=loc, ip=ip)],
            "ld.acquire.gpu.global.u64 $0, [$1];",
            "=l,l",
            True,
            loc,
            ip,
        )
    )


@dsl_user_op
def _ld_relaxed_gpu_u64(addr, *, loc=None, ip=None) -> Int64:
    return Int64(
        _asm(
            T.i64(),
            [Int64(addr).ir_value(loc=loc, ip=ip)],
            "ld.relaxed.gpu.global.u64 $0, [$1];",
            "=l,l",
            True,
            loc,
            ip,
        )
    )


@dsl_user_op
def _st_shared_f32(addr, v, *, loc=None, ip=None) -> None:
    _asm(
        None,
        [Int32(addr).ir_value(loc=loc, ip=ip), Float32(v).ir_value(loc=loc, ip=ip)],
        "st.shared.f32 [$0], $1;",
        "r,f",
        True,
        loc,
        ip,
    )


@dsl_user_op
def _ld_shared_v4(addr, *, loc=None, ip=None):
    out = _asm(
        llvm.StructType.get_literal([T.f32()] * 4),
        [Int32(addr).ir_value(loc=loc, ip=ip)],
        "ld.shared.v4.f32 {$0, $1, $2, $3}, [$4];",
        "=f,=f,=f,=f,r",
        True,
        loc,
        ip,
    )
    return tuple(Float32(llvm.extractvalue(T.f32(), out, [i], loc=loc, ip=ip)) for i in range(4))


@dsl_user_op
def _st_global_v4(addr, a, b, c, d, *, loc=None, ip=None) -> None:
    _asm(
        None,
        [Int64(addr).ir_value(loc=loc, ip=ip)]
        + [Float32(v).ir_value(loc=loc, ip=ip) for v in (a, b, c, d)],
        "st.global.v4.f32 [$0], {$1, $2, $3, $4};",
        "l,f,f,f,f",
        True,
        loc,
        ip,
    )


@dsl_user_op
def _ld_shared_v2(addr, *, loc=None, ip=None):
    out = _asm(
        llvm.StructType.get_literal([T.f32()] * 2),
        [Int32(addr).ir_value(loc=loc, ip=ip)],
        "ld.shared.v2.f32 {$0, $1}, [$2];",
        "=f,=f,r",
        True,
        loc,
        ip,
    )
    return tuple(Float32(llvm.extractvalue(T.f32(), out, [i], loc=loc, ip=ip)) for i in range(2))


@dsl_user_op
def _bulk_g2s(dst_smem, src_gmem, nbytes, mbar_smem, *, loc=None, ip=None) -> None:
    """cp.async.bulk global -> shared (16-byte aligned, size a multiple of 16), completion on an mbarrier."""
    _asm(
        None,
        [
            Int32(dst_smem).ir_value(loc=loc, ip=ip),
            Int64(src_gmem).ir_value(loc=loc, ip=ip),
            Int32(nbytes).ir_value(loc=loc, ip=ip),
            Int32(mbar_smem).ir_value(loc=loc, ip=ip),
        ],
        "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [$0], [$1], $2, [$3];",
        "r,l,r,r",
        True,
        loc,
        ip,
    )


@dsl_user_op
def _nanosleep(ns, *, loc=None, ip=None) -> None:
    _asm(None, [Int32(ns).ir_value(loc=loc, ip=ip)], "nanosleep.u32 $0;", "r", True, loc, ip)


# Fold mode helpers: every float op has an explicit rounding mode (no contraction), so results are bitwise those of
# _qkv_rope_gate_fp8_append_kernel. Loads are volatile asm, ordered against griddepcontrol.wait (also volatile); no
# memory clobber, so a load is not pinned behind an unrelated earlier store.
@dsl_user_op
def _ld_global_b32(addr, *, loc=None, ip=None) -> Uint32:
    return Uint32(
        _asm(
            T.i32(),
            [Int64(addr).ir_value(loc=loc, ip=ip)],
            "ld.global.b32 $0, [$1];",
            "=r,l",
            True,
            loc,
            ip,
        )
    )


@dsl_user_op
def _ld_global_f32(addr, *, loc=None, ip=None) -> Float32:
    return Float32(
        _asm(
            T.f32(),
            [Int64(addr).ir_value(loc=loc, ip=ip)],
            "ld.global.f32 $0, [$1];",
            "=f,l",
            True,
            loc,
            ip,
        )
    )


@dsl_user_op
def _ld_global_s64(addr, *, loc=None, ip=None) -> Int64:
    return Int64(
        _asm(
            T.i64(),
            [Int64(addr).ir_value(loc=loc, ip=ip)],
            "ld.global.s64 $0, [$1];",
            "=l,l",
            True,
            loc,
            ip,
        )
    )


@dsl_user_op
def _ld_global_s32(addr, *, loc=None, ip=None) -> Int32:
    return Int32(
        _asm(
            T.i32(),
            [Int64(addr).ir_value(loc=loc, ip=ip)],
            "ld.global.s32 $0, [$1];",
            "=r,l",
            True,
            loc,
            ip,
        )
    )


@dsl_user_op
def _bf16_lo(u, *, loc=None, ip=None) -> Float32:
    """Low bf16 of a 32-bit word as FP32 (exact)."""
    return Float32(
        _asm(
            T.f32(),
            [Uint32(u).ir_value(loc=loc, ip=ip)],
            "{\n\t.reg .b16 l, h;\n\tmov.b32 {l, h}, $1;\n\tcvt.f32.bf16 $0, l;\n\t}\n",
            "=f,r",
            False,
            loc,
            ip,
        )
    )


@dsl_user_op
def _bf16_hi(u, *, loc=None, ip=None) -> Float32:
    """High bf16 of a 32-bit word as FP32 (exact)."""
    return Float32(
        _asm(
            T.f32(),
            [Uint32(u).ir_value(loc=loc, ip=ip)],
            "{\n\t.reg .b16 l, h;\n\tmov.b32 {l, h}, $1;\n\tcvt.f32.bf16 $0, h;\n\t}\n",
            "=f,r",
            False,
            loc,
            ip,
        )
    )


@dsl_user_op
def _rn_bf16(x, *, loc=None, ip=None) -> Float32:
    """x rounded to bf16 (round to nearest even), as FP32."""
    return Float32(
        _asm(
            T.f32(),
            [Float32(x).ir_value(loc=loc, ip=ip)],
            "{\n\t.reg .b16 h;\n\tcvt.rn.bf16.f32 h, $1;\n\tcvt.f32.bf16 $0, h;\n\t}\n",
            "=f,f",
            False,
            loc,
            ip,
        )
    )


@dsl_user_op
def _rn_bf16x2(a, b, *, loc=None, ip=None):
    """(a, b) rounded to bf16 (round to nearest even) with one packed conversion, as FP32."""
    out = _asm(
        llvm.StructType.get_literal([T.f32()] * 2),
        [Float32(a).ir_value(loc=loc, ip=ip), Float32(b).ir_value(loc=loc, ip=ip)],
        "{\n\t.reg .b32 p;\n\tcvt.rn.bf16x2.f32 p, $3, $2;\n\tshl.b32 $0, p, 16;\n\t"
        "and.b32 $1, p, 0xffff0000;\n\t}\n",
        "=f,=f,f,f",
        False,
        loc,
        ip,
    )
    return (
        Float32(llvm.extractvalue(T.f32(), out, [0], loc=loc, ip=ip)),
        Float32(llvm.extractvalue(T.f32(), out, [1], loc=loc, ip=ip)),
    )


@dsl_user_op
def _mul_rn(a, b, *, loc=None, ip=None) -> Float32:
    return Float32(
        _asm(
            T.f32(),
            [Float32(a).ir_value(loc=loc, ip=ip), Float32(b).ir_value(loc=loc, ip=ip)],
            "mul.rn.f32 $0, $1, $2;",
            "=f,f,f",
            False,
            loc,
            ip,
        )
    )


@dsl_user_op
def _add_rn(a, b, *, loc=None, ip=None) -> Float32:
    return Float32(
        _asm(
            T.f32(),
            [Float32(a).ir_value(loc=loc, ip=ip), Float32(b).ir_value(loc=loc, ip=ip)],
            "add.rn.f32 $0, $1, $2;",
            "=f,f,f",
            False,
            loc,
            ip,
        )
    )


@dsl_user_op
def _fma_rn(a, b, c, *, loc=None, ip=None) -> Float32:
    return Float32(
        _asm(
            T.f32(),
            [Float32(v).ir_value(loc=loc, ip=ip) for v in (a, b, c)],
            "fma.rn.f32 $0, $1, $2, $3;",
            "=f,f,f,f",
            False,
            loc,
            ip,
        )
    )


@dsl_user_op
def _neg(a, *, loc=None, ip=None) -> Float32:
    return Float32(
        _asm(
            T.f32(),
            [Float32(a).ir_value(loc=loc, ip=ip)],
            "neg.f32 $0, $1;",
            "=f,f",
            False,
            loc,
            ip,
        )
    )


@dsl_user_op
def _div_full(a, b, *, loc=None, ip=None) -> Float32:
    return Float32(
        _asm(
            T.f32(),
            [Float32(a).ir_value(loc=loc, ip=ip), Float32(b).ir_value(loc=loc, ip=ip)],
            "div.full.f32 $0, $1, $2;",
            "=f,f,f",
            False,
            loc,
            ip,
        )
    )


@dsl_user_op
def _rsqrt_ftz(a, *, loc=None, ip=None) -> Float32:
    return Float32(
        _asm(
            T.f32(),
            [Float32(a).ir_value(loc=loc, ip=ip)],
            "rsqrt.approx.ftz.f32 $0, $1;",
            "=f,f",
            False,
            loc,
            ip,
        )
    )


@dsl_user_op
def _e4m3x2(a, b, *, loc=None, ip=None) -> Uint32:
    """Two FP32 -> two E4M3 (cvt.rn.satfinite), ``a`` in the low byte, in the low 16 bits of the result."""
    return Uint32(
        _asm(
            T.i32(),
            [Float32(a).ir_value(loc=loc, ip=ip), Float32(b).ir_value(loc=loc, ip=ip)],
            "{\n\t.reg .b16 p;\n\tcvt.rn.satfinite.e4m3x2.f32 p, $2, $1;\n\tcvt.u32.u16 $0, p;\n\t}\n",
            "=r,f,f",
            False,
            loc,
            ip,
        )
    )


@dsl_user_op
def _selp_f32(pred, a, b, *, loc=None, ip=None) -> Float32:
    """pred != 0 ? a : b (branch-free)."""
    return Float32(
        _asm(
            T.f32(),
            [
                Int32(pred).ir_value(loc=loc, ip=ip),
                Float32(a).ir_value(loc=loc, ip=ip),
                Float32(b).ir_value(loc=loc, ip=ip),
            ],
            "{\n\t.reg .pred p;\n\tsetp.ne.s32 p, $1, 0;\n\tselp.f32 $0, $2, $3, p;\n\t}\n",
            "=f,r,f,f",
            False,
            loc,
            ip,
        )
    )


@dsl_user_op
def _selp_u32(pred, a, b, *, loc=None, ip=None) -> Uint32:
    """pred != 0 ? a : b (branch-free)."""
    return Uint32(
        _asm(
            T.i32(),
            [
                Int32(pred).ir_value(loc=loc, ip=ip),
                Uint32(a).ir_value(loc=loc, ip=ip),
                Uint32(b).ir_value(loc=loc, ip=ip),
            ],
            "{\n\t.reg .pred p;\n\tsetp.ne.s32 p, $1, 0;\n\tselp.b32 $0, $2, $3, p;\n\t}\n",
            "=r,r,r,r",
            False,
            loc,
            ip,
        )
    )


@dsl_user_op
def _st_shared_b16(addr, v, *, loc=None, ip=None) -> None:
    _asm(
        None,
        [Int32(addr).ir_value(loc=loc, ip=ip), Uint32(v).ir_value(loc=loc, ip=ip)],
        "{\n\t.reg .b16 p;\n\tcvt.u16.u32 p, $1;\n\tst.shared.b16 [$0], p;\n\t}\n",
        "r,r",
        True,
        loc,
        ip,
    )


@dsl_user_op
def _st_global_b16(addr, v, *, loc=None, ip=None) -> None:
    _asm(
        None,
        [Int64(addr).ir_value(loc=loc, ip=ip), Uint32(v).ir_value(loc=loc, ip=ip)],
        "{\n\t.reg .b16 p;\n\tcvt.u16.u32 p, $1;\n\tst.global.b16 [$0], p;\n\t}\n",
        "l,r,~{memory}",
        True,
        loc,
        ip,
    )


class GqaSplitKvDecodeAttnKernel:
    """See the module docstring. Compile-time parameters:

    q_len, group:    tokens per request and q heads per KV head of this rank (rows = q_len * group <= 32).
    has_gate:        multiply the output by sigmoid(gate) (same arithmetic as the separate sigmoid-mul kernel:
                     the bf16-rounded attention output times sigmoid(gate) in FP32).
    bmm1_from_ptr:   read the log2-domain QK scale from ``bmm1_ptr[1]`` (trtllm-gen [raw, log2] layout) after the
                     dependency wait, else use the ``bmm1_log2`` argument. Same for ``bmm2`` (``bmm2_ptr[0]``).
    use_pdl:         launched as a programmatic dependent: releases this grid's dependents at entry (safe with the
                     grid barrier: the dependent grid launches only once every CTA is resident) and requests the KV
                     pages that cannot hold this call's new tokens before ``griddepcontrol.wait``.
    fresh_margin_pages: pages below the new tokens' first page that are also read only after the wait.
    n_splits:        CTAs per request (rows <= n_splits <= 256, n_splits * batch <= SMs).
    p_shift:         E4M3 P shift in log2 units (see __init__).
    fold:            the qk-norm / RoPE / FP8 / KV-append prologue runs in this kernel (see __init__).
    """

    def __init__(
        self,
        *,
        q_len: int,
        group: int,
        has_gate: bool,
        bmm1_from_ptr: bool,
        bmm2_from_ptr: bool,
        use_pdl: bool = True,
        evict_first: bool = True,
        stages: int = 2,
        n_splits: int = 152,
        fresh_margin_pages: int = 1,
        p_shift: float = 6.0,
        fold: bool = False,
        use_mrope: bool = False,
        mrope_s1: int = 0,
        mrope_s2: int = 0,
        pos_i64: bool = False,
    ):
        self.q_len = q_len
        # fold: the kernel reads the raw QKV projection ([tokens, (q, gate) x heads, k, v] bf16) and does the work of
        # _qkv_rope_gate_fp8_append_kernel (attention/kernels/fused_qk_norm_rope_gate.py) itself: Gemma qk-norm + neox
        # RoPE (rotary dim 64, optionally interleaved mrope) + FP8 of q into SMEM (every CTA), FP8 K / V of the new
        # tokens into the paged cache (the CTA whose tiles hold them, before it loads those tiles), the FMHA scales and
        # the gate. Bit-identical to that kernel (same operation order, explicit rounding).
        self.fold = fold
        self.use_mrope = use_mrope
        self.mrope_s1 = mrope_s1
        self.mrope_s2 = mrope_s2
        self.pos_i64 = pos_i64
        if fold:
            assert has_gate and not bmm1_from_ptr and not bmm2_from_ptr
        self.q_full_count = 128 if fold else 1
        self.fold_rpw = (
            8 if q_len * group > 16 else 4
        )  # q rows (padded to the MMA N) per softmax warp
        # P = exp2(x - m_ref + p_shift) before the E4M3 conversion: the reference max maps to 2^p_shift instead of 1,
        # so small probabilities stay E4M3 normals (>= 2^-6) down to 2^-(6 + p_shift) of the max and flush to 0 only
        # below 2^-(10 + p_shift). O and l carry the same 2^p_shift (the output is unchanged); the lazy rescale moves
        # the reference once a score exceeds it by P_TOP - p_shift.
        self.p_shift = float(p_shift)
        self.rescale_threshold = min(RESCALE_THRESHOLD, P_TOP - self.p_shift)
        self.group = group
        self.rows = q_len * group
        assert 1 <= self.rows <= 32, f"q_len * group = {self.rows} must be in [1, 32]"
        self.nq = 16 if self.rows <= 16 else 32
        self.has_gate = has_gate
        self.bmm1_from_ptr = bmm1_from_ptr
        self.bmm2_from_ptr = bmm2_from_ptr
        self.use_pdl = use_pdl
        self.evict_first = evict_first
        self.stages = stages
        # pages >= (L - q_len) // 64 - fresh_margin_pages are read only after griddepcontrol.wait: with 1, pre-wait
        # reads never touch a position written in this step (MTP passes 2-6 also wrote the ~q_len + 5 positions just
        # below the new token earlier in the same step)
        self.fresh_margin_pages = fresh_margin_pages
        self.n_splits = n_splits  # CTAs per request (grid x), compile-time (fixed per batch size)
        assert self.rows <= n_splits <= 256, "combine needs rows <= splits <= 256"
        total = self.rows * HEAD_DIM
        self.chunk = 4 * ((total + 4 * n_splits - 1) // (4 * n_splits))  # outputs finalized per CTA
        # Combine TMA box row / SMEM row stride: an odd number of 16-byte quads, so lanes reading one float4 slot of
        # consecutive splits hit different bank groups. Columns read past the chunk stay inside the split's partial.
        self.chunk_pad = self.chunk + (4 if (self.chunk // 4) % 2 == 0 else 0)
        self.n_chunks = (
            total + self.chunk - 1
        ) // self.chunk  # CTAs of a request that finalize outputs
        n_vec = self.chunk // 4
        gps = 32
        while gps > NUM_THREADS // n_vec:
            gps //= 2
        self.gps = (
            gps  # lanes (power of two, one warp) sharing one float4 output slot in the combine
        )
        self.combine_maxl = (n_splits + gps - 1) // gps  # splits per combine lane
        # workspace per request (floats): O partials chunk-major [n_chunks][n_splits][chunk_pad], then (m, l, -, -)
        # row-major [rows][n_splits][4]: each finalizing CTA reads one contiguous block per tensor (bulk copies)
        self.ws_o_floats = self.n_chunks * n_splits * self.chunk_pad
        self.ws_req_floats = self.ws_o_floats + self.rows * n_splits * 4
        # TMEM: 2 S buffers (nq columns each) + O^T (2 x 128 lanes x nq columns).
        self.tmem_cols = 128 if self.nq == 32 else 64

    @cute.jit
    def __call__(
        self,
        q_ptr: cute.Pointer,
        kv_ptr: cute.Pointer,
        n_kv_rows: Int32,
        tab_ptr: cute.Pointer,
        tab_stride_b: Int32,
        tab_stride_kv: Int32,
        seqlen_ptr: cute.Pointer,
        bmm1_ptr: cute.Pointer,
        bmm1_log2: Float32,
        bmm2_ptr: cute.Pointer,
        bmm2: Float32,
        gate_ptr: cute.Pointer,
        gate_stride_tok: Int32,
        out_ptr: cute.Pointer,
        out_stride_tok: Int32,
        ws_ptr: cute.Pointer,
        cnt_ptr: cute.Pointer,
        batch: Int32,
        n_splits: Int32,
        qkv_ptr: cute.Pointer,
        qkv_stride: Int32,
        qw_ptr: cute.Pointer,
        kw_ptr: cute.Pointer,
        cs_ptr: cute.Pointer,
        max_pos: Int32,
        pos_ptr: cute.Pointer,
        kvq_ptr: cute.Pointer,
        kvd_ptr: cute.Pointer,
        host_bmm1: Float32,
        eps: Float32,
        stream,
    ):
        S = self.stages
        NQ = self.nq
        R = self.rows
        NS = self.n_splits
        CH = self.chunk_pad
        mQ = cute.make_tensor(
            q_ptr, cute.make_layout((R, HEAD_DIM, batch), stride=(HEAD_DIM, 1, R * HEAD_DIM))
        )
        mKV = cute.make_tensor(
            kv_ptr, cute.make_layout((n_kv_rows, HEAD_DIM), stride=(HEAD_DIM, 1))
        )
        mKVt = cute.make_tensor(
            kv_ptr, cute.make_layout((HEAD_DIM, n_kv_rows), stride=(1, HEAD_DIM))
        )
        mTab = cute.make_tensor(
            tab_ptr,
            cute.make_layout(
                (batch, 2, 1 << 24),  # page extent unused (no bounds)
                stride=(tab_stride_b, tab_stride_kv, 1),
            ),
        )
        mSeq = cute.make_tensor(seqlen_ptr, cute.make_layout((batch,)))
        mBmm1 = cute.make_tensor(bmm1_ptr, cute.make_layout((2,)))
        mBmm2 = cute.make_tensor(bmm2_ptr, cute.make_layout((1,)))
        # gate / out rows: token (r * q_len + t), head g -> [token, g * 256 + d]
        if const_expr(self.fold):
            # the gate of head g sits right after its q in the raw QKV projection: [token, g * 512 + 256 + d]
            mGate = cute.make_tensor(
                qkv_ptr + HEAD_DIM,
                cute.make_layout(
                    (batch * self.q_len, self.group, HEAD_DIM), stride=(qkv_stride, 2 * HEAD_DIM, 1)
                ),
            )
        else:
            mGate = cute.make_tensor(
                gate_ptr,
                cute.make_layout(
                    (batch * self.q_len, self.group * HEAD_DIM), stride=(gate_stride_tok, 1)
                ),
            )
        mOut = cute.make_tensor(
            out_ptr,
            cute.make_layout(
                (batch * self.q_len, self.group * HEAD_DIM), stride=(out_stride_tok, 1)
            ),
        )
        # workspace: ws_req_floats per request (chunk-major O partials, then (m, l) per row; addressed explicitly)
        mWs = cute.make_tensor(ws_ptr, cute.make_layout((batch * self.ws_req_floats,)))
        mCnt = cute.make_tensor(cnt_ptr, cute.make_layout((8,)))  # grid-barrier epoch per request

        mma_qk = sm100_utils.make_trivial_tiled_mma(
            FP8,
            FP8,
            OperandMajorMode.K,
            OperandMajorMode.K,
            Float32,
            tcgen05.CtaGroup.ONE,
            (TILE_KEYS, NQ),
        )
        mma_pv = sm100_utils.make_trivial_tiled_mma(
            FP8,
            FP8,
            OperandMajorMode.MN,
            OperandMajorMode.MN,
            Float32,
            tcgen05.CtaGroup.ONE,
            (128, NQ),
        )
        qk_tiler = (TILE_KEYS, NQ, HEAD_DIM)
        pv_tiler = (HEAD_DIM, NQ, TILE_KEYS)
        sK_l = sm100_utils.make_smem_layout_a(mma_qk, qk_tiler, FP8, S)
        sV_l = sm100_utils.make_smem_layout_a(mma_pv, pv_tiler, FP8, S)
        sQ_l = sm100_utils.make_smem_layout_b(mma_qk, qk_tiler, FP8, 1)
        sP_l = sm100_utils.make_smem_layout_b(mma_pv, pv_tiler, FP8, 2)
        # Same buffers seen by the TMA (one box = one page x 128 B) and by the softmax P^T stores ((n, key, buffer)).
        sK_tma_l = cute.tiled_divide(
            sm100_utils.make_smem_layout(OperandMajorMode.K, (TILE_KEYS, HEAD_DIM), FP8, S),
            (PAGE, SW_BYTES),
        )
        sV_tma_l = cute.tiled_divide(
            sm100_utils.make_smem_layout(OperandMajorMode.MN, (HEAD_DIM, TILE_KEYS), FP8, S),
            (SW_BYTES, PAGE),
        )
        sP_nk_l = sm100_utils.make_smem_layout(OperandMajorMode.MN, (NQ, TILE_KEYS), FP8, 2)
        # fold: the same Q buffer seen as (row, d) for the in-kernel q stores (K-major, as the K tile's TMA view)
        sQ_nk_l = sm100_utils.make_smem_layout(OperandMajorMode.K, (NQ, HEAD_DIM), FP8, 1)
        box_k = sm100_utils.make_smem_layout(OperandMajorMode.K, (PAGE, SW_BYTES), FP8, 1)
        box_v = sm100_utils.make_smem_layout(OperandMajorMode.MN, (SW_BYTES, PAGE), FP8, 1)

        op = cpasync.CopyBulkTensorTileG2SOp()
        tma_k, gK_tma = cpasync.make_tiled_tma_atom(op, mKV, box_k, (PAGE, SW_BYTES))
        tma_v, gV_tma = cpasync.make_tiled_tma_atom(op, mKVt, box_v, (SW_BYTES, PAGE))
        cluster_vmnk = cute.tiled_divide(cute.make_layout((1, 1, 1)), (mma_qk.thr_id.shape,))
        sQ_one = cute.slice_(sQ_l, (None, None, None, 0))
        tma_q, gQ_tma = cute.nvgpu.make_tiled_tma_atom_B(
            sm100_utils.cluster_shape_to_tma_atom_B((1, 1), mma_qk.thr_id),
            mQ,
            sQ_one,
            qk_tiler,
            mma_qk,
            cluster_vmnk.shape,
        )
        q_bytes = cute.size_in_bytes(FP8, sQ_one)
        box_ws = cute.make_layout((CH, NS), stride=(1, CH))
        box_ml = cute.make_layout((4, NS), stride=(1, 4))

        self.kernel(
            mma_qk,
            mma_pv,
            tma_k,
            gK_tma,
            tma_v,
            gV_tma,
            tma_q,
            gQ_tma,
            box_ws,
            box_ml,
            mTab,
            mSeq,
            mBmm1,
            bmm1_log2,
            mBmm2,
            bmm2,
            mGate,
            mOut,
            mWs,
            mCnt,
            sK_l,
            sV_l,
            sQ_l,
            sP_l,
            sK_tma_l,
            sV_tma_l,
            sP_nk_l,
            q_bytes,
            NS,
            sQ_nk_l,
            kv_ptr,
            qkv_ptr,
            qkv_stride,
            qw_ptr,
            kw_ptr,
            cs_ptr,
            max_pos,
            pos_ptr,
            kvq_ptr,
            kvd_ptr,
            host_bmm1,
            eps,
            batch,
        ).launch(
            grid=[NS, batch, 1],
            block=[NUM_THREADS, 1, 1],
            stream=stream,
            use_pdl=self.use_pdl,
            min_blocks_per_mp=1,
        )

    # fold-mode row layout (one warp per 256-element row): lane m = 8 * w + i holds the element pairs
    # e(k) = 64 * w + 2 * i + 16 * k, k = 0..3, i.e. the pairs of threads 32 * w + i + 8 * k of the 4-warp Triton
    # kernel. Its sum of squares then reduces in exactly Triton's tree (per thread fma(x0, x0, x1 * x1); butterfly
    # 16, 8 inside the lane, 4, 2, 1 across lanes; warps (p0 + p2) + (p1 + p3)), and the RoPE pairs (e, e + 32) of
    # the rotary dims [0, 64) are both in lane i of w = 0 (k and k + 2).
    @cute.jit
    def _fold_load_row(self, src, xw: cute.Tensor, j, lane):
        """The 4 bf16 pairs of this lane of the row at byte address ``src`` -> xw[j, 0..3]."""
        e0 = 64 * (lane // 8) + 2 * (lane % 8)
        for k in cutlass.range_constexpr(4):
            xw[j, k] = _ld_global_b32(src + 2 * (e0 + 16 * k))

    @cute.jit
    def _fold_load_static(
        self, lane, w_ptr: cute.Pointer, kvq_ptr: cute.Pointer, ww: cute.Tensor
    ) -> Float32:
        """Static operands (norm weight words of this lane's elements, kv_scale_orig_quant), requested before the
        dependency wait and only consumed after it, so their latency hides behind the predecessor."""
        e0 = 64 * (lane // 8) + 2 * (lane % 8)
        for k in cutlass.range_constexpr(4):
            ww[k] = _ld_global_b32(w_ptr.toint() + 2 * (e0 + 16 * k))
        return _ld_global_f32(kvq_ptr.toint())

    @cute.jit
    def _fold_weights(self, ww: cute.Tensor, wp1: cute.Tensor):
        """Norm weight words -> weight + 1 in FP32 (as the Triton kernel: bf16 -> f32, + 1.0)."""
        for k in cutlass.range_constexpr(4):
            wp1[2 * k] = _add_rn(_bf16_lo(ww[k]), Float32(1.0))
            wp1[2 * k + 1] = _add_rn(_bf16_hi(ww[k]), Float32(1.0))

    @cute.jit
    def _fold_cs_entry(self, idx):
        """Entry idx of sCS (token t = idx // 64, column c: cos (c < 32) or sin of rotary index c % 32) -> (t, c, mrope
        section of the index)."""
        t = idx // 64
        c = idx % 64
        ri = c % 32
        sec = Int32(0)
        if const_expr(self.use_mrope):
            if ri % 3 == 1:
                if ri < 3 * self.mrope_s1:
                    sec = Int32(1)
            if ri % 3 == 2:
                if ri < 3 * self.mrope_s2:
                    sec = Int32(2)
        return t, c, sec

    @cute.jit
    def _fold_cs_prefill(self, st, L, cs_ptr: cute.Pointer, max_pos, sCS: cute.Tensor):
        """Before the dependency wait (the cos / sin table is static): sCS for the guessed positions L - q_len + t (a
        decode token's rotary position is its cache position unless the request carries an mrope delta);
        _fold_tables reloads the entries whose real position differs."""
        QL: cutlass.Constexpr = self.q_len
        NE: cutlass.Constexpr = (QL * 64 + 127) // 128
        vals = cute.make_rmem_tensor((NE,), Float32)
        for e in cutlass.range_constexpr(NE):  # all loads in flight before any use
            idx = cutlass.min(st + 128 * e, QL * 64 - 1)
            g = cutlass.min(cutlass.max(Int64(L - QL + idx // 64), Int64(0)), Int64(max_pos) - 1)
            vals[e] = _ld_global_f32(cs_ptr.toint() + (g * 64 + idx % 64) * 4)
        for e in cutlass.range_constexpr(NE):
            idx = st + 128 * e
            if idx < QL * 64:
                sCS[idx // 64, idx % 64] = vals[e]

    @cute.jit
    def _fold_tables(
        self,
        st,
        req,
        batch,
        L,
        pos_ptr: cute.Pointer,
        cs_ptr: cute.Pointer,
        max_pos,
        sPos: cute.Tensor,
        sCS: cute.Tensor,
    ):
        """The 128 softmax threads, after the dependency wait: positions of this request's tokens (per mrope section)
        -> sPos; then every sCS entry whose position differs from the prefill's guess is reloaded (per-index mrope
        section, NaN for an out-of-table position, as the Triton kernel). The common case costs one load round."""
        QL: cutlass.Constexpr = self.q_len
        NSEC: cutlass.Constexpr = 3 if self.use_mrope else 1
        n_tok = batch * QL
        if st < NSEC * QL:
            idx = Int64(st // QL) * Int64(n_tok) + Int64(req * QL + st % QL)
            p = Int64(0)
            if const_expr(self.pos_i64):
                p = _ld_global_s64(pos_ptr.toint() + idx * 8)
            else:
                p = Int64(_ld_global_s32(pos_ptr.toint() + idx * 4))
            sPos[st // QL, st % QL] = p
        cute.arch.barrier(barrier_id=BAR_SOFTMAX, number_of_threads=128)
        NE: cutlass.Constexpr = (QL * 64 + 127) // 128
        for e in cutlass.range_constexpr(NE):
            idx = st + 128 * e
            if idx < QL * 64:
                t, c, sec = self._fold_cs_entry(idx)
                p = sPos[sec, t]
                g = cutlass.min(cutlass.max(Int64(L - QL + t), Int64(0)), Int64(max_pos) - 1)
                if p != g:
                    valid = Int32(0)
                    sp = Int64(0)
                    if p >= Int64(0):
                        if p < Int64(max_pos):
                            valid = Int32(1)
                            sp = p
                    v = _ld_global_f32(cs_ptr.toint() + (sp * 64 + c) * 4)
                    sCS[t, c] = _selp_f32(valid, v, Float32(float("nan")))
        cute.arch.barrier(barrier_id=BAR_SOFTMAX, number_of_threads=128)

    @cute.jit
    def _fold_row_fp8(
        self,
        xw: cute.Tensor,
        j,
        wp1: cute.Tensor,
        norm: cutlass.Constexpr,
        do_rope,
        sCS: cute.Tensor,
        t,
        lane,
        sc,
        eps,
        out: cute.Tensor,
    ):
        """Row j of xw (token t of the request) -> 4 FP8 pairs (out[0..3], low 16 bits): Gemma RMS norm (weight + 1 =
        wp1), neox RoPE on the rotary pairs (do_rope: lanes 0-7 of a q / K row; cos / sin from sCS), then
        e4m3(bf16(bf16(y) * sc)), all as the Triton kernel."""
        x = cute.make_rmem_tensor((8,), Float32)
        for k in cutlass.range_constexpr(4):
            x[2 * k] = _bf16_lo(xw[j, k])
            x[2 * k + 1] = _bf16_hi(xw[j, k])
        y = cute.make_rmem_tensor((8,), Float32)
        if const_expr(norm):
            t0 = _fma_rn(x[0], x[0], _mul_rn(x[1], x[1]))
            t1 = _fma_rn(x[2], x[2], _mul_rn(x[3], x[3]))
            t2 = _fma_rn(x[4], x[4], _mul_rn(x[5], x[5]))
            t3 = _fma_rn(x[6], x[6], _mul_rn(x[7], x[7]))
            b = _add_rn(
                _add_rn(t0, t2), _add_rn(t1, t3)
            )  # Triton levels 16 (k, k + 2) and 8 (k = 0, 1)
            b = _add_rn(b, cute.arch.shuffle_sync_bfly(b, offset=4))
            b = _add_rn(b, cute.arch.shuffle_sync_bfly(b, offset=2))
            b = _add_rn(b, cute.arch.shuffle_sync_bfly(b, offset=1))
            b = _add_rn(
                b, cute.arch.shuffle_sync_bfly(b, offset=16)
            )  # warps: lane 0 = (p0 + p2) + (p1 + p3)
            b = _add_rn(b, cute.arch.shuffle_sync_bfly(b, offset=8))
            tot = cute.arch.shuffle_sync(b, offset=0)
            inv = _rsqrt_ftz(_add_rn(_div_full(tot, Float32(256.0)), eps))
            for e in cutlass.range_constexpr(4):
                y[2 * e], y[2 * e + 1] = _rn_bf16x2(
                    _mul_rn(wp1[2 * e], _mul_rn(inv, x[2 * e])),
                    _mul_rn(wp1[2 * e + 1], _mul_rn(inv, x[2 * e + 1])),
                )
            # RoPE on lanes 0-7 (w = 0), branch-free so that ptxas can interleave rows: every lane computes it with
            # the cos / sin of index 2 * (lane % 8) + h + 16 k (in range) and keeps it only where do_rope
            r1 = cute.make_rmem_tensor((4,), Float32)
            r2 = cute.make_rmem_tensor((4,), Float32)
            for k in cutlass.range_constexpr(2):
                for h in cutlass.range_constexpr(2):
                    ri = 2 * (lane % 8) + h + 16 * k
                    f = y[2 * k + h]
                    s2 = y[2 * k + 4 + h]
                    cv = sCS[t, ri]
                    sv = sCS[t, 32 + ri]
                    r1[2 * k + h] = _fma_rn(cv, f, _neg(_mul_rn(sv, s2)))
                    r2[2 * k + h] = _fma_rn(sv, f, _mul_rn(cv, s2))
            for k in cutlass.range_constexpr(2):
                a0, a1 = _rn_bf16x2(r1[2 * k], r1[2 * k + 1])
                b0, b1 = _rn_bf16x2(r2[2 * k], r2[2 * k + 1])
                y[2 * k] = _selp_f32(do_rope, a0, y[2 * k])
                y[2 * k + 1] = _selp_f32(do_rope, a1, y[2 * k + 1])
                y[2 * k + 4] = _selp_f32(do_rope, b0, y[2 * k + 4])
                y[2 * k + 5] = _selp_f32(do_rope, b1, y[2 * k + 5])
        else:
            for e in cutlass.range_constexpr(8):
                y[e] = x[e]
        for k in cutlass.range_constexpr(4):
            q0, q1 = _rn_bf16x2(_mul_rn(y[2 * k], sc), _mul_rn(y[2 * k + 1], sc))
            out[k] = _e4m3x2(q0, q1)

    @cute.jit
    def _fold_q(
        self,
        sw,
        st,
        lane,
        req,
        batch,
        L,
        qkv_ptr: cute.Pointer,
        qkv_stride,
        ww: cute.Tensor,
        kvq_raw,
        cs_ptr: cute.Pointer,
        max_pos,
        pos_ptr: cute.Pointer,
        eps,
        sQ: cute.Tensor,
        sQ_nk_l: cute.ComposedLayout,
        sPos: cute.Tensor,
        sCS: cute.Tensor,
        cs_ready,
    ):
        """The softmax warps after the dependency wait: q rows sw, sw + 4, ... (row r = token t * group + head h of
        this request) -> FP8 into the GEMM1 B operand; rows R..NQ-1 (MMA padding, zero-filled by the q TMA
        otherwise) are zeroed. Row loads first, then the position / cos-sin tables (shared with warp 2's K rows via
        cs_ready), then the arithmetic."""
        NQ: cutlass.Constexpr = self.nq  # noqa: F841
        R: cutlass.Constexpr = self.rows
        QL: cutlass.Constexpr = self.q_len
        G: cutlass.Constexpr = self.group
        RPW: cutlass.Constexpr = self.fold_rpw
        e0 = 64 * (lane // 8) + 2 * (lane % 8)
        qkv_b = qkv_ptr.toint()
        xw = cute.make_rmem_tensor((RPW, 4), Uint32)
        for j in cutlass.range_constexpr(RPW):
            rc = cutlass.min(
                sw + 4 * j, R - 1
            )  # rows >= R (MMA padding) compute row R - 1, zeroed below
            tq = req * QL + rc // G
            self._fold_load_row(
                qkv_b + (Int64(tq) * Int64(qkv_stride) + Int64((rc % G) * 2 * HEAD_DIM)) * 2,
                xw,
                j,
                lane,
            )
        self._fold_tables(st, req, batch, L, pos_ptr, cs_ptr, max_pos, sPos, sCS)
        wp1 = cute.make_rmem_tensor((8,), Float32)
        self._fold_weights(ww, wp1)
        sc = _rn_bf16(kvq_raw)
        cute.arch.mbarrier_arrive(cs_ready)
        sq_b = sQ.iterator.toint()
        f8 = cute.make_rmem_tensor((4,), Uint32)
        f8_all = cute.make_rmem_tensor((RPW, 4), Uint32)
        rope_lane = Int32(0)
        if lane < 8:
            rope_lane = Int32(1)
        # all rows' arithmetic first (no volatile asm in between, so ptxas can interleave the rows' dependency chains),
        # then every store
        for j in cutlass.range_constexpr(RPW):  # RPW * 4 == NQ: every row of the operand is written
            r = sw + 4 * j
            rc = cutlass.min(r, R - 1)
            keep = Int32(0)
            if r < R:
                keep = Int32(1)
            self._fold_row_fp8(xw, j, wp1, True, rope_lane, sCS, rc // G, lane, sc, eps, f8)
            for k in cutlass.range_constexpr(4):
                f8_all[j, k] = _selp_u32(keep, f8[k], Uint32(0))
        for j in cutlass.range_constexpr(RPW):
            for k in cutlass.range_constexpr(4):
                _st_shared_b16(
                    sq_b + cute.crd2idx((sw + 4 * j, e0 + 16 * k, 0), sQ_nk_l), f8_all[j, k]
                )

    @cute.jit
    def _fold_append(
        self,
        kind: cutlass.Constexpr,
        lane,
        req,
        L,
        k_lo,
        k_hi,
        mTab: cute.Tensor,
        kv_base: cute.Pointer,
        qkv_ptr: cute.Pointer,
        qkv_stride,
        ww: cute.Tensor,
        kvq_raw,
        eps,
        sCS: cute.Tensor,
        cs_ready,
    ):
        """The new tokens of this request whose positions fall in this CTA's tiles [k_lo, k_hi): their K (kind 0:
        qk-norm + RoPE, cos / sin from the softmax warps' table) or V (kind 1) row -> FP8 at position L - q_len + t
        of the paged cache, as the Triton kernel (skipped for a negative position or block index). Each position lies
        in exactly one CTA's tiles."""
        QL: cutlass.Constexpr = self.q_len
        G: cutlass.Constexpr = self.group
        e0 = 64 * (lane // 8) + 2 * (lane % 8)
        qkv_b = qkv_ptr.toint()
        xw = cute.make_rmem_tensor((QL, 4), Uint32)
        own = cute.make_rmem_tensor((QL,), Int32)
        n_own = Int32(0)
        for t in cutlass.range_constexpr(QL):
            pos = L - QL + t
            o = Int32(0)
            if pos >= 0:
                if pos // TILE_KEYS >= k_lo:
                    if pos // TILE_KEYS < k_hi:
                        o = Int32(1)
            own[t] = o
            n_own = n_own + o
            if o != 0:
                src = (
                    qkv_b
                    + (Int64(req * QL + t) * Int64(qkv_stride) + Int64((2 * G + kind) * HEAD_DIM))
                    * 2
                )
                self._fold_load_row(src, xw, t, lane)
        wp1 = cute.make_rmem_tensor((8,), Float32)
        if const_expr(kind == 0):
            self._fold_weights(ww, wp1)
        sc = _rn_bf16(kvq_raw)
        if n_own > 0:
            if const_expr(kind == 0):
                cute.arch.mbarrier_wait(cs_ready, 0)
            f8 = cute.make_rmem_tensor((4,), Uint32)
            kv_b = kv_base.toint()
            for t in cutlass.range_constexpr(QL):
                if own[t] != 0:
                    pos = L - QL + t
                    blk = mTab[req, kind, pos // PAGE]
                    rope_lane = Int32(0)
                    if lane < 8:
                        rope_lane = Int32(1)
                    self._fold_row_fp8(xw, t, wp1, kind == 0, rope_lane, sCS, t, lane, sc, eps, f8)
                    if blk >= 0:
                        dst = kv_b + (Int64(blk) * PAGE + Int64(pos % PAGE)) * HEAD_DIM + e0
                        for k in cutlass.range_constexpr(4):
                            _st_global_b16(dst + 16 * k, f8[k])

    @cute.jit
    def _produce(
        self,
        i,
        kb0,
        kb1,
        vb0,
        vb1,
        tma_k: cute.CopyAtom,
        tKg: cute.Tensor,
        tKs: cute.Tensor,
        tma_v: cute.CopyAtom,
        tVg: cute.Tensor,
        tVs: cute.Tensor,
        full_k,
        empty_k,
        full_v,
        empty_v,
    ):
        """TMA of tile i of the ring: pages (pg0, pg0 + 1) of K (blocks kb0, kb1) and V (vb0, vb1)."""
        S: cutlass.Constexpr = self.stages
        s = i % S
        ph = (i // S) & 1
        kbs = (kb0, kb1)
        vbs = (vb0, vb1)
        cute.arch.mbarrier_wait(empty_k + s, ph ^ 1)
        with cute.arch.elect_one():
            cute.arch.mbarrier_arrive_and_expect_tx(full_k + s, 2 * PAGE * HEAD_DIM)
        for p in cutlass.range_constexpr(2):
            for kb in cutlass.range_constexpr(2):
                if const_expr(self.evict_first):
                    cute.copy(
                        tma_k,
                        tKg[(None, kbs[p], kb)],
                        tKs[(None, p, kb, s)],
                        tma_bar_ptr=full_k + s,
                        cache_policy=Int64(EVICT_FIRST),
                    )
                else:
                    cute.copy(
                        tma_k,
                        tKg[(None, kbs[p], kb)],
                        tKs[(None, p, kb, s)],
                        tma_bar_ptr=full_k + s,
                    )
        cute.arch.mbarrier_wait(empty_v + s, ph ^ 1)
        with cute.arch.elect_one():
            cute.arch.mbarrier_arrive_and_expect_tx(full_v + s, 2 * PAGE * HEAD_DIM)
        for p in cutlass.range_constexpr(2):
            for h in cutlass.range_constexpr(2):
                if const_expr(self.evict_first):
                    cute.copy(
                        tma_v,
                        tVg[(None, h, vbs[p])],
                        tVs[(None, h, p, s)],
                        tma_bar_ptr=full_v + s,
                        cache_policy=Int64(EVICT_FIRST),
                    )
                else:
                    cute.copy(
                        tma_v, tVg[(None, h, vbs[p])], tVs[(None, h, p, s)], tma_bar_ptr=full_v + s
                    )

    @cute.jit
    def _gemm2(
        self,
        j,
        mma_pv: cute.TiledMma,
        tOtO: cute.Tensor,
        tCrV: cute.Tensor,
        tCrP: cute.Tensor,
        full_v,
        p_full,
        empty_v,
        p_empty,
        o_done,
    ):
        """O^T += V^T(tile j) . P^T(tile j); releases the V stage and the P buffer, signals o_done. Returns mma_pv:
        its ACCUMULATE field is updated here, and the caller must rebind it so that the DSL carries the new value out of
        the enclosing dynamic regions."""
        S: cutlass.Constexpr = self.stages
        sj = j % S
        pb = j % 2
        cute.arch.mbarrier_wait(full_v + sj, (j // S) & 1)
        cute.arch.mbarrier_wait(p_full + pb, (j // 2) & 1)
        for dm in cutlass.range_constexpr(2):
            for kk in cutlass.range_constexpr(TILE_KEYS // 32):
                mma_pv.set(tcgen05.Field.ACCUMULATE, (j > 0) or (kk != 0))
                cute.gemm(
                    mma_pv,
                    tOtO[(None, dm, 0)],
                    tCrV[(None, dm, kk, sj)],
                    tCrP[(None, 0, kk, pb)],
                    tOtO[(None, dm, 0)],
                )
        with cute.arch.elect_one():
            tcgen05.commit(empty_v + sj)
            tcgen05.commit(p_empty + pb)
            tcgen05.commit(o_done)
        return mma_pv

    @cute.jit
    def _softmax_tile(
        self,
        i,
        pg0,
        x: cute.Tensor,
        m_ref: cute.Tensor,
        l_acc: cute.Tensor,
        alpha: cute.Tensor,
        tStS: cute.Tensor,
        tOtO: cute.Tensor,
        t2r_s,
        thr_s,
        t2r_o,
        thr_o,
        r2t_o,
        thr_ro,
        tSc: cute.Tensor,
        sRed: cute.Tensor,
        bars_sm,
        sP_base,
        sP_nk_l: cute.ComposedLayout,
        st,
        sw,
        lane,
        prefix,
        n_pages,
        scale,
    ):
        """Softmax of tile i (pages pg0, pg0 + 1; thread st = key st of the tile = TMEM lane st): S -> x, mask, lazy
        rescale of (m, l, O), P^T -> SMEM as E4M3."""
        NQ: cutlass.Constexpr = self.nq
        QL: cutlass.Constexpr = self.q_len
        G: cutlass.Constexpr = self.group
        s_full, s_empty, p_full, p_empty, o_done = bars_sm
        p_slot = st // PAGE
        p_off = st % PAGE
        sb = i % 2
        cute.arch.mbarrier_wait(s_full + sb, (i // 2) & 1)
        cute.copy(t2r_s, thr_s.partition_S(tStS[(None, None), 0, 0, sb]), x)
        cute.arch.fence_view_async_tmem_load()
        cute.arch.mbarrier_arrive(s_empty + sb)
        # Scale + mask (key position of this thread)
        page = pg0 + p_slot
        pos = page * PAGE + p_off
        dup = page >= n_pages  # second slot of the request's last, 1-page tile
        lim = prefix - pos  # >= 0: visible to every row
        thr_rows = Int32(0)
        if lim < 0:
            thr_rows = cutlass.min(-lim, QL) * G
            if -lim >= QL:
                thr_rows = Int32(NQ)
        if dup:
            thr_rows = Int32(NQ)
        exceed = Int32(0)
        for k in cutlass.range_constexpr(NQ):
            v = x[k] * scale
            if k < thr_rows:
                v = -Float32.inf
            x[k] = v
            if v > m_ref[k] + self.rescale_threshold:
                exceed = Int32(1)
        any_ex = _bar_red_or(exceed, BAR_SOFTMAX_RED, 128)
        if any_ex != 0:
            # Slow path: exact tile max per row, rescale l and O
            for k in cutlass.range_constexpr(NQ):
                mx = cute.arch.warp_redux_sync(x[k], kind="fmax")
                if lane == k:
                    sRed[sw, k] = mx
            cute.arch.barrier(barrier_id=BAR_SOFTMAX, number_of_threads=128)
            for k in cutlass.range_constexpr(NQ):
                mt = cute.arch.fmax(
                    cute.arch.fmax(sRed[0, k], sRed[1, k]), cute.arch.fmax(sRed[2, k], sRed[3, k])
                )
                mn = cute.arch.fmax(m_ref[k], mt)
                a = Float32(1.0)
                if mn != -Float32.inf:
                    if m_ref[k] == -Float32.inf:
                        a = Float32(0.0)
                    else:
                        a = _ex2(m_ref[k] - mn)
                alpha[k] = a
                l_acc[k] = l_acc[k] * a
                m_ref[k] = mn
            if i > 0:
                # O holds GEMM2 of tiles < i once GEMM2(i - 1) completed; GEMM2(i) waits for P(i).
                cute.arch.mbarrier_wait(o_done, (i - 1) & 1)
                for dm in cutlass.range_constexpr(2):
                    o_reg = cute.make_rmem_tensor_like(tSc, Float32)
                    cute.copy(t2r_o, thr_o.partition_S(tOtO[(None, None), dm, 0]), o_reg)
                    cute.arch.fence_view_async_tmem_load()
                    for k in cutlass.range_constexpr(NQ):
                        o_reg[k] = o_reg[k] * alpha[k]
                    cute.copy(r2t_o, o_reg, thr_ro.partition_D(tOtO[(None, None), dm, 0]))
                cute.arch.fence_view_async_tmem_store()
        # P = exp2(x - m), row sums, E4M3 P^T -> SMEM
        pw = cute.make_rmem_tensor((NQ // 4,), Uint32)
        for k4 in cutlass.range_constexpr(NQ // 4):
            pv = cute.make_rmem_tensor((4,), Float32)
            for e in cutlass.range_constexpr(4):
                k = 4 * k4 + e
                p = Float32(0.0)
                if x[k] != -Float32.inf:
                    if const_expr(self.p_shift != 0.0):
                        p = _ex2(x[k] - m_ref[k] + self.p_shift)
                    else:
                        p = _ex2(x[k] - m_ref[k])
                l_acc[k] = l_acc[k] + p
                pv[e] = p
            pw[k4] = _pack_e4m3x4(pv[0], pv[1], pv[2], pv[3])
        pb = i % 2
        cute.arch.mbarrier_wait(p_empty + pb, ((i // 2) & 1) ^ 1)
        for c in cutlass.range_constexpr(NQ // 16):
            off = cute.crd2idx((16 * c, st, pb), sP_nk_l)
            _st_shared_v4(sP_base + off, pw[4 * c], pw[4 * c + 1], pw[4 * c + 2], pw[4 * c + 3])
        cute.arch.fence_proxy("async.shared", space="cta")
        cute.arch.mbarrier_arrive(p_full + pb)

    @cute.kernel
    def kernel(
        self,
        mma_qk: cute.TiledMma,
        mma_pv: cute.TiledMma,
        tma_k: cute.CopyAtom,
        gK_tma: cute.Tensor,
        tma_v: cute.CopyAtom,
        gV_tma: cute.Tensor,
        tma_q: cute.CopyAtom,
        gQ_tma: cute.Tensor,
        box_ws: cute.Layout,
        box_ml: cute.Layout,
        mTab: cute.Tensor,
        mSeq: cute.Tensor,
        mBmm1: cute.Tensor,
        bmm1_log2: Float32,
        mBmm2: cute.Tensor,
        bmm2: Float32,
        mGate: cute.Tensor,
        mOut: cute.Tensor,
        mWs: cute.Tensor,
        mCnt: cute.Tensor,
        sK_l: cute.ComposedLayout,
        sV_l: cute.ComposedLayout,
        sQ_l: cute.ComposedLayout,
        sP_l: cute.ComposedLayout,
        sK_tma_l: cute.ComposedLayout,
        sV_tma_l: cute.ComposedLayout,
        sP_nk_l: cute.ComposedLayout,
        q_bytes: cutlass.Constexpr,
        n_splits: cutlass.Constexpr,
        sQ_nk_l: cute.ComposedLayout,
        kv_base: cute.Pointer,
        qkv_ptr: cute.Pointer,
        qkv_stride: Int32,
        qw_ptr: cute.Pointer,
        kw_ptr: cute.Pointer,
        cs_ptr: cute.Pointer,
        max_pos: Int32,
        pos_ptr: cute.Pointer,
        kvq_ptr: cute.Pointer,
        kvd_ptr: cute.Pointer,
        host_bmm1: Float32,
        eps: Float32,
        batch: Int32,
    ):
        S: cutlass.Constexpr = self.stages
        NQ: cutlass.Constexpr = self.nq
        R: cutlass.Constexpr = self.rows
        QL: cutlass.Constexpr = self.q_len
        G: cutlass.Constexpr = self.group
        tidx, _, _ = cute.arch.thread_idx()
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        lane = tidx % 32
        split, req, _ = cute.arch.block_idx()
        if const_expr(self.use_pdl):
            # Safe with the grid barrier below: the dependent grid launches only after every CTA of this grid ran
            # this instruction, i.e. after all of them are resident.
            _griddep_launch()

        # seq_lens / block tables are host-uploaded or written by non-PDL ops earlier in the graph: safe to read
        # before the dependency wait.
        L = mSeq[req]
        n_pages = (L + PAGE - 1) // PAGE
        T = (
            n_pages + 1
        ) // 2  # tiles: tile k = pages 2k, 2k + 1 (the last one may hold a single page)
        ts = cutlass.max(
            (T + n_splits - 1) // n_splits, 1
        )  # tiles per CTA; >= 1: padded dummy requests may have L = 0
        n_valid_splits = (T + ts - 1) // ts
        k_lo = cutlass.min(split * ts, T)  # this CTA's tiles: [k_lo, k_hi)
        k_hi = cutlass.min(k_lo + ts, T)
        prefix = L - QL  # keys < prefix are visible to every row
        # pages >= fresh_page are read only after griddepcontrol.wait (see fresh_margin_pages)
        fresh_page = cutlass.max(prefix // PAGE - self.fresh_margin_pages, 0)
        new_tile0 = (
            cutlass.max(prefix, 0) // TILE_KEYS
        )  # fold: tiles >= new_tile0 hold this call's new tokens
        # combine geometry: CTA `split` finalizes outputs [o_lo, o_hi) of the request's R x 256 (row-major), in
        # float4 slots; gps lanes (a power of two inside one warp) share slot q and cover splits jg, jg + gps, ...
        ws_o_base = mWs.iterator.toint() + req * (
            self.ws_req_floats * 4
        )  # [n_chunks][n_splits][chunk_pad]
        ws_ml_base = ws_o_base + self.ws_o_floats * 4  # [rows][n_splits][4]
        total = R * HEAD_DIM
        chunk: cutlass.Constexpr = self.chunk
        chunk_pad: cutlass.Constexpr = self.chunk_pad
        o_lo = split * chunk
        o_hi = cutlass.min(o_lo + chunk, total)
        n_vec = cutlass.max((o_hi - o_lo + 3) // 4, 0)
        gps: cutlass.Constexpr = self.gps
        q = tidx // gps
        jg = tidx % gps
        active = q < n_vec
        ov = o_lo + 4 * cutlass.min(q, cutlass.max(n_vec - 1, 0))
        row = ov // HEAD_DIM
        tok = req * QL + row // G
        col = (row % G) * HEAD_DIM + ov % HEAD_DIM

        smem = SmemAllocator()
        bars = smem.allocate_array(Int64, 4 * S + 14)
        kv_app = (
            bars + 4 * S + 12
        )  # fold: the new tokens' K / V of this CTA's tiles are in the cache (64 arrivals)
        cs_ready = bars + 4 * S + 13  # fold: sPos / sCS written by the softmax warps (128 arrivals)
        full_k = bars
        empty_k = bars + S
        full_v = bars + 2 * S
        empty_v = bars + 3 * S
        q_full = bars + 4 * S
        s_full = bars + 4 * S + 1  # [2]
        s_empty = bars + 4 * S + 3  # [2]
        p_full = bars + 4 * S + 5  # [2]
        p_empty = bars + 4 * S + 7  # [2]
        o_done = bars + 4 * S + 9
        o_full = bars + 4 * S + 10
        comb_bar = bars + 4 * S + 11
        tmem_holder = smem.allocate_array(Int32, 4)
        sTab = smem.allocate_tensor(Int32, cute.make_layout((2, TAB_MAX), stride=(TAB_MAX, 1)), 16)
        sRed = smem.allocate_tensor(
            Float32, cute.make_layout((4, 32), stride=(32, 1)), 16
        )  # per-warp row values
        sTarget = smem.allocate_tensor(Int64, cute.make_layout((2,)), 16)
        sScal = smem.allocate_tensor(Float32, cute.make_layout((4,)), 16)
        sComb = smem.allocate_tensor(Float32, box_ws, 128)  # [split][chunk] O chunks (TMA)
        sML0 = smem.allocate_tensor(Float32, box_ml, 128)  # [split][(m, l, -, -)] of row0
        sML1 = smem.allocate_tensor(Float32, box_ml, 128)  # ... of row0 + 1
        sGate = smem.allocate_tensor(Float32, cute.make_layout((256,)), 16)
        sPos = sTarget
        sCS = sRed
        if const_expr(self.fold):
            sPos = smem.allocate_tensor(
                Int64, cute.make_layout((3, 16), stride=(16, 1)), 16
            )  # [section][token]
            sCS = smem.allocate_tensor(
                Float32, cute.make_layout((16, 64), stride=(64, 1)), 16
            )  # [token][cos | sin]
        sK = smem.allocate_tensor(FP8, sK_l.outer, 1024, sK_l.inner)
        sV = smem.allocate_tensor(FP8, sV_l.outer, 1024, sV_l.inner)
        sQ = smem.allocate_tensor(FP8, sQ_l.outer, 1024, sQ_l.inner)
        sP = smem.allocate_tensor(FP8, sP_l.outer, 1024, sP_l.inner)
        sK_tma = cute.make_tensor(
            cute.recast_ptr(sK.iterator, sK_tma_l.inner, dtype=FP8), sK_tma_l.outer
        )
        sV_tma = cute.make_tensor(
            cute.recast_ptr(sV.iterator, sV_tma_l.inner, dtype=FP8), sV_tma_l.outer
        )

        if warp == 0:
            cpasync.prefetch_descriptor(tma_k)
            cpasync.prefetch_descriptor(tma_v)
            if const_expr(not self.fold):
                cpasync.prefetch_descriptor(tma_q)
        if tidx == 0:
            for i in cutlass.range_constexpr(4 * S):
                cute.arch.mbarrier_init(bars + i, 1)
            # q_full: the q TMA (1 arrival + tx), or in fold mode the 128 softmax threads that write q
            cute.arch.mbarrier_init(q_full, self.q_full_count)
            cute.arch.mbarrier_init(kv_app, 64)
            cute.arch.mbarrier_init(cs_ready, 128)
            for i in cutlass.range_constexpr(2):
                cute.arch.mbarrier_init(s_full + i, 1)
                cute.arch.mbarrier_init(s_empty + i, 128)
                cute.arch.mbarrier_init(p_full + i, 128)
                cute.arch.mbarrier_init(p_empty + i, 1)
            cute.arch.mbarrier_init(o_done, 1)
            cute.arch.mbarrier_init(o_full, 1)
            cute.arch.mbarrier_init(comb_bar, 1)
        cute.arch.mbarrier_init_fence()
        if warp == 1:
            cute.arch.alloc_tmem(self.tmem_cols, tmem_holder)
        cute.arch.sync_threads()
        tmem_ptr = cute.arch.retrieve_tmem_ptr(Float32, 16, tmem_holder)
        tStS = mma_qk.make_fragment_C(mma_qk.partition_shape_C((TILE_KEYS, NQ, 2)))
        tStS = cute.make_tensor(tmem_ptr, tStS.layout)
        s_cols = tcgen05.find_tmem_tensor_col_offset(tStS)
        tOtO = mma_pv.make_fragment_C(mma_pv.partition_shape_C((HEAD_DIM, NQ)))
        tOtO = cute.make_tensor(tmem_ptr + s_cols, tOtO.layout)

        if warp == 0:
            for j in cutlass.range(lane, 2 * (k_hi - k_lo), 32):
                pg_t = cutlass.min(2 * k_lo + j, n_pages - 1)
                sTab[0, j] = mTab[req, 0, pg_t]
                sTab[1, j] = mTab[req, 1, pg_t]
            cute.arch.sync_warp()
            gK = cute.tiled_divide(gK_tma, (PAGE, SW_BYTES))  # ((64, 128), blocks, 2)
            gV = cute.tiled_divide(gV_tma, (SW_BYTES, PAGE))  # ((128, 64), 2, blocks)
            tKs, tKg = cpasync.tma_partition(tma_k, 0, cute.make_layout(1), sK_tma, gK)
            tVs, tVg = cpasync.tma_partition(tma_v, 0, cute.make_layout(1), sV_tma, gV)
            waited = Int32(0)
            app_waited = Int32(0)
            if const_expr(not self.use_pdl):
                waited = Int32(1)  # no dependency wait: every page is requested in order
            for k in cutlass.range(k_lo, k_hi):
                pg0 = 2 * k
                if waited == 0:
                    if pg0 + 1 >= fresh_page:
                        # this tile may hold positions written by the preceding QKV kernel
                        if const_expr(self.use_pdl):
                            _griddep_wait()
                        waited = Int32(1)
                if const_expr(self.fold):
                    if app_waited == 0:
                        if k >= new_tile0:
                            # warps 2-3 wrote this CTA's new K / V rows (generic proxy) and fenced them for the
                            # async proxy; the TMA below reads them back
                            cute.arch.mbarrier_wait(kv_app, 0)
                            cute.arch.fence_proxy("async.global")
                            app_waited = Int32(1)
                jt = 2 * (k - k_lo)
                self._produce(
                    k - k_lo,
                    sTab[0, jt],
                    sTab[0, jt + 1],
                    sTab[1, jt],
                    sTab[1, jt + 1],
                    tma_k,
                    tKg,
                    tKs,
                    tma_v,
                    tVg,
                    tVs,
                    full_k,
                    empty_k,
                    full_v,
                    empty_v,
                )
        elif warp == 1:
            tCrK = mma_qk.make_fragment_A(sK)
            tCrQ = mma_qk.make_fragment_B(sQ)
            tCrV = mma_pv.make_fragment_A(sV)
            tCrP = mma_pv.make_fragment_B(sP)
            cute.arch.mbarrier_wait(q_full, 0)
            n_t = k_hi - k_lo
            for i in cutlass.range(n_t + 1):
                if i < n_t:
                    s = i % S
                    sb = i % 2
                    cute.arch.mbarrier_wait(full_k + s, (i // S) & 1)
                    cute.arch.mbarrier_wait(s_empty + sb, ((i // 2) & 1) ^ 1)
                    for kk in cutlass.range_constexpr(HEAD_DIM // 32):
                        mma_qk.set(tcgen05.Field.ACCUMULATE, kk != 0)
                        cute.gemm(
                            mma_qk,
                            tStS[(None, 0, 0, sb)],
                            tCrK[(None, 0, kk, s)],
                            tCrQ[(None, 0, kk, 0)],
                            tStS[(None, 0, 0, sb)],
                        )
                    with cute.arch.elect_one():
                        tcgen05.commit(s_full + sb)
                        tcgen05.commit(empty_k + s)
                if i >= 1:
                    mma_pv = self._gemm2(
                        i - 1, mma_pv, tOtO, tCrV, tCrP, full_v, p_full, empty_v, p_empty, o_done
                    )
            with cute.arch.elect_one():
                tcgen05.commit(o_full)
        elif warp >= SOFTMAX_WARP0:
            # Softmax (thread st = key st of the tile = TMEM lane st)
            st = tidx - SOFTMAX_WARP0 * 32
            sw = warp - SOFTMAX_WARP0
            wwq = cute.make_rmem_tensor((4,), Uint32)
            kvq_raw = Float32(0.0)
            s_qo = Float32(0.0)
            if const_expr(self.fold):
                # static operands before the wait: the cos / sin table of the guessed positions, the q-norm weight,
                # the KV scales (the dependency wait only orders the projection and the positions)
                kvq_raw = self._fold_load_static(lane, qw_ptr, kvq_ptr, wwq)
                s_qo = _ld_global_f32(kvd_ptr.toint())
                self._fold_cs_prefill(st, L, cs_ptr, max_pos, sCS)
            if const_expr(self.use_pdl):
                _griddep_wait()  # bmm1 / bmm2 scales come from the preceding QKV kernel
            scale = bmm1_log2
            if const_expr(self.bmm1_from_ptr):
                scale = mBmm1[1]
            if const_expr(self.fold):
                # bmm1 * log2(e) exactly as the Triton kernel prepares it: ((s * s) * host_bmm1) * log2e (FP32)
                scale = _mul_rn(_mul_rn(host_bmm1, _mul_rn(s_qo, s_qo)), Float32(LOG2E))
                self._fold_q(
                    sw,
                    st,
                    lane,
                    req,
                    batch,
                    L,
                    qkv_ptr,
                    qkv_stride,
                    wwq,
                    kvq_raw,
                    cs_ptr,
                    max_pos,
                    pos_ptr,
                    eps,
                    sQ,
                    sQ_nk_l,
                    sPos,
                    sCS,
                    cs_ready,
                )
                cute.arch.fence_proxy("async.shared", space="cta")
                cute.arch.mbarrier_arrive(q_full)
            atom_ld = cute.make_copy_atom(tcgen05.Ld32x32bOp(tcgen05.Repetition(NQ)), Float32)
            atom_st = cute.make_copy_atom(tcgen05.St32x32bOp(tcgen05.Repetition(NQ)), Float32)
            accS0 = tStS[(None, None), 0, 0, 0]
            t2r_s = tcgen05.make_tmem_copy(atom_ld, accS0)
            thr_s = t2r_s.get_slice(st)
            tSc = thr_s.partition_D(cute.make_identity_tensor((TILE_KEYS, NQ)))
            accO0 = tOtO[(None, None), 0, 0]
            t2r_o = tcgen05.make_tmem_copy(atom_ld, accO0)
            thr_o = t2r_o.get_slice(st)
            r2t_o = tcgen05.make_tmem_copy(atom_st, accO0)
            thr_ro = r2t_o.get_slice(st)
            x = cute.make_rmem_tensor_like(tSc, Float32)
            m_ref = cute.make_rmem_tensor_like(tSc, Float32)
            l_acc = cute.make_rmem_tensor_like(tSc, Float32)
            alpha = cute.make_rmem_tensor_like(tSc, Float32)
            for k in cutlass.range_constexpr(NQ):
                m_ref[k] = -Float32.inf
                l_acc[k] = Float32(0.0)
            sP_base = sP.iterator.toint()
            bars_sm = (s_full, s_empty, p_full, p_empty, o_done)
            n_t = k_hi - k_lo
            for i in cutlass.range(n_t):
                self._softmax_tile(
                    i,
                    2 * (k_lo + i),
                    x,
                    m_ref,
                    l_acc,
                    alpha,
                    tStS,
                    tOtO,
                    t2r_s,
                    thr_s,
                    t2r_o,
                    thr_o,
                    r2t_o,
                    thr_ro,
                    tSc,
                    sRed,
                    bars_sm,
                    sP_base,
                    sP_nk_l,
                    st,
                    sw,
                    lane,
                    prefix,
                    n_pages,
                    scale,
                )
            # Row sums over the 128 key threads: butterfly transpose-reduce, lane k ends with row k's sum
            cnt_live = NQ
            for off in (16, 8, 4, 2, 1):
                if const_expr(NQ == 16 and off == 16):
                    for k in cutlass.range_constexpr(16):
                        l_acc[k] = l_acc[k] + cute.arch.shuffle_sync_bfly(l_acc[k], offset=16)
                else:
                    half = cnt_live // 2
                    upper = (lane & off) != 0
                    for k in cutlass.range_constexpr(half):
                        lo_v = l_acc[k]
                        hi_v = l_acc[k + half]
                        send = lo_v if upper else hi_v
                        keep = hi_v if upper else lo_v
                        l_acc[k] = keep + cute.arch.shuffle_sync_bfly(send, offset=off)
                    cnt_live = half
            if lane < NQ:
                sRed[sw, lane] = l_acc[0]
            cute.arch.barrier(barrier_id=BAR_SOFTMAX, number_of_threads=128)
            # Partial -> workspace: O^T (thread st owns d = st and 128 + st), then (m, l) per row
            o0 = cute.make_rmem_tensor_like(tSc, Float32)
            o1 = cute.make_rmem_tensor_like(tSc, Float32)
            if n_t > 0:
                cute.arch.mbarrier_wait(o_full, 0)
                cute.copy(t2r_o, thr_o.partition_S(tOtO[(None, None), 0, 0]), o0)
                cute.copy(t2r_o, thr_o.partition_S(tOtO[(None, None), 1, 0]), o1)
                cute.arch.fence_view_async_tmem_load()
            else:
                for k in cutlass.range_constexpr(NQ):
                    o0[k] = Float32(0.0)
                    o1[k] = Float32(0.0)
            # transpose through SMEM (the K ring is free: o_full covers every MMA that read it) -> 16-byte stores
            stage_o = sK.iterator.toint()
            for k in cutlass.range_constexpr(R):
                _st_shared_f32(stage_o + (k * HEAD_DIM + st) * 4, o0[k])
                _st_shared_f32(stage_o + (k * HEAD_DIM + 128 + st) * 4, o1[k])
            cute.arch.barrier(barrier_id=BAR_SOFTMAX, number_of_threads=128)
            for it in cutlass.range_constexpr((R * 64 + 127) // 128):
                idx = (
                    st + it * 128
                )  # float4 index of the row-major [R][256] partial; never straddles a chunk
                if idx < R * 64:
                    v4o = _ld_shared_v4(stage_o + idx * 16)
                    ch_i = (4 * idx) // chunk
                    ch_w = (4 * idx) % chunk
                    dst = ws_o_base + ((ch_i * n_splits + split) * chunk_pad + ch_w) * 4
                    _st_global_v4(dst, v4o[0], v4o[1], v4o[2], v4o[3])
            if sw == 0:
                if lane < R:
                    lsum = sRed[0, lane] + sRed[1, lane] + sRed[2, lane] + sRed[3, lane]
                    mval = -Float32.inf
                    for k in cutlass.range_constexpr(NQ):
                        if lane == k:
                            mval = m_ref[k]
                    _st_global_v4(
                        ws_ml_base + ((lane * n_splits + split) * 4) * 4,
                        mval,
                        lsum,
                        Float32(0.0),
                        Float32(0.0),
                    )
        elif warp == 2:
            wwk = cute.make_rmem_tensor((4,), Uint32)
            kvq_k = Float32(0.0)
            if const_expr(self.fold):
                kvq_k = self._fold_load_static(
                    lane, kw_ptr, kvq_ptr, wwk
                )  # static: before the wait
            if const_expr(self.use_pdl):
                _griddep_wait()  # the gate and bmm2 come from earlier kernels
            if const_expr(self.fold):
                self._fold_append(
                    0,
                    lane,
                    req,
                    L,
                    k_lo,
                    k_hi,
                    mTab,
                    kv_base,
                    qkv_ptr,
                    qkv_stride,
                    wwk,
                    kvq_k,
                    eps,
                    sCS,
                    cs_ready,
                )
                cute.arch.fence_proxy("async.global")
                cute.arch.mbarrier_arrive(kv_app)
            # epoch target of the grid barrier: the counter grows by n_splits per launch and never resets; fewer than
            # n_splits CTAs of this launch can have arrived yet, so floor(c / n) is the previous launches' count.
            if lane == 0:
                c = _ld_relaxed_gpu_u64(mCnt.iterator.toint() + req * 8)
                ns64 = Int64(n_splits)
                sTarget[0] = (c // ns64 + 1) * ns64
                b2v = bmm2
                if const_expr(self.bmm2_from_ptr):
                    b2v = mBmm2[0]
                if const_expr(self.fold):
                    b2v = _ld_global_f32(
                        kvd_ptr.toint()
                    )  # bmm2 = kv_scale_quant_orig (Triton kernel: scales[4])
                sScal[0] = b2v
            # sigmoid(gate) of this CTA's output chunk
            if const_expr(self.has_gate):
                for e in cutlass.range(lane, o_hi - o_lo, 32):
                    og = o_lo + e
                    tok_g = req * QL + (og // HEAD_DIM) // G
                    gv = Float32(0.0)
                    if const_expr(self.fold):
                        gv = mGate[tok_g, (og // HEAD_DIM) % G, og % HEAD_DIM].to(Float32)
                    else:
                        col_g = ((og // HEAD_DIM) % G) * HEAD_DIM + og % HEAD_DIM
                        gv = mGate[tok_g, col_g].to(Float32)
                    sGate[e] = Float32(1.0) / (Float32(1.0) + _ex2(-gv * LOG2E))
        elif warp == 3:
            # q is written by the preceding QKV kernel: requested right after the dependency wait, independently of
            # how far the KV producer got.
            wwv = cute.make_rmem_tensor((4,), Uint32)
            kvq_v = Float32(0.0)
            if const_expr(self.fold):
                kvq_v = _ld_global_f32(kvq_ptr.toint())  # static: before the wait
            if const_expr(self.use_pdl):
                _griddep_wait()
            if const_expr(self.fold):
                self._fold_append(
                    1,
                    lane,
                    req,
                    L,
                    k_lo,
                    k_hi,
                    mTab,
                    kv_base,
                    qkv_ptr,
                    qkv_stride,
                    wwv,
                    kvq_v,
                    eps,
                    sCS,
                    cs_ready,
                )
                cute.arch.fence_proxy("async.global")
                cute.arch.mbarrier_arrive(kv_app)
            else:
                gQ = cute.local_tile(
                    gQ_tma,
                    cute.slice_((TILE_KEYS, NQ, HEAD_DIM), (0, None, None)),
                    (None, None, None),
                )
                tCgQ = mma_qk.get_slice(0).partition_B(gQ)
                tQs, tQg = cpasync.tma_partition(
                    tma_q,
                    0,
                    cute.make_layout(1),
                    cute.group_modes(sQ, 0, 3),
                    cute.group_modes(tCgQ, 0, 3),
                )
                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive_and_expect_tx(q_full, q_bytes)
                cute.copy(tma_q, tQg[(None, 0, 0, req)], tQs[(None, 0)], tma_bar_ptr=q_full)

        # Grid barrier over the CTAs of this request (epoch counter)
        cute.arch.sync_threads()  # partial written; epoch target, bmm2 and the gate chunk in SMEM
        if tidx == 0:
            cnt = mCnt.iterator.toint() + req * 8
            # red.release is cumulative over the CTA's partial stores ordered before it by the bar.sync above
            _red_release_add_u64(cnt, 1)
            target = sTarget[0]
            seen = _ld_acquire_gpu_u64(cnt)
            while seen < target:
                _nanosleep(32)
                seen = _ld_acquire_gpu_u64(cnt)
        cute.arch.sync_threads()

        # Combine: bulk-copy the splits' chunks to SMEM, reduce with shuffles
        if n_vec > 0:
            row0 = o_lo // HEAD_DIM
            row1 = (o_hi - 1) // HEAD_DIM  # row0 or row0 + 1 (chunk <= 256)
            if warp == 0:
                # this CTA's chunk of every split is one contiguous block (chunk-major workspace), and so are the
                # (m, l) of each of its (at most two) rows: 2-3 bulk copies issued by one lane
                cute.arch.fence_proxy(
                    "async.global"
                )  # acquire (generic proxy) -> bulk reads (async proxy)
                if lane == 0:
                    n_ml = row1 - row0 + 1
                    cute.arch.mbarrier_arrive_and_expect_tx(
                        comb_bar, n_splits * (chunk_pad + 4 * n_ml) * 4
                    )
                    _bulk_g2s(
                        sComb.iterator.toint(),
                        ws_o_base + (split * n_splits * chunk_pad) * 4,
                        n_splits * chunk_pad * 4,
                        comb_bar.toint(),
                    )
                    _bulk_g2s(
                        sML0.iterator.toint(),
                        ws_ml_base + (row0 * n_splits * 4) * 4,
                        n_splits * 16,
                        comb_bar.toint(),
                    )
                    if row1 != row0:
                        _bulk_g2s(
                            sML1.iterator.toint(),
                            ws_ml_base + (row1 * n_splits * 4) * 4,
                            n_splits * 16,
                            comb_bar.toint(),
                        )
            # spin (no suspend hint): every lane of a warp must be awake for the shuffles below
            landed = cute.arch.mbarrier_try_wait(comb_bar, 0)
            while not landed:
                landed = cute.arch.mbarrier_try_wait(comb_bar, 0)
            s_c = sComb.iterator.toint()
            s_ml0 = sML0.iterator.toint()
            s_ml1 = sML1.iterator.toint()
            rs = row - row0
            s_ml = s_ml0 + rs * (s_ml1 - s_ml0)  # branch-free row select
            mv = cute.make_rmem_tensor((self.combine_maxl,), Float32)
            lv = cute.make_rmem_tensor((self.combine_maxl,), Float32)
            ovs = cute.make_rmem_tensor((self.combine_maxl, 4), Float32)
            qc = cutlass.min(q, n_vec - 1)
            # straight-line SMEM reads (clamped split index: duplicates of the last valid split, weighted 0 below)
            for u in cutlass.range_constexpr(self.combine_maxl):
                jc = cutlass.min(jg + u * gps, n_valid_splits - 1)
                v4 = _ld_shared_v4(s_c + (jc * chunk_pad + 4 * qc) * 4)
                ml2 = _ld_shared_v2(s_ml + jc * 16)
                for e in cutlass.range_constexpr(4):
                    ovs[u, e] = v4[e]
                mv[u] = ml2[0]
                lv[u] = ml2[1]
            mloc = mv[0]
            for u in cutlass.range_constexpr(1, self.combine_maxl):
                mloc = cute.arch.fmax(mloc, mv[u])
            for lg in cutlass.range_constexpr(self.gps.bit_length() - 1):
                mloc = cute.arch.fmax(
                    mloc, cute.arch.shuffle_sync_bfly(mloc, offset=self.gps >> (lg + 1))
                )
            if mloc == -Float32.inf:
                mloc = Float32(0.0)  # no visible key in any split: every weight is exp2(-inf) = 0
            lsum = Float32(0.0)
            a0 = Float32(0.0)
            a1 = Float32(0.0)
            a2 = Float32(0.0)
            a3 = Float32(0.0)
            for u in cutlass.range_constexpr(self.combine_maxl):
                valid = Float32(jg + u * gps < n_valid_splits)
                w = _ex2(mv[u] - mloc) * valid
                lsum = lsum + w * lv[u]
                a0 = a0 + w * ovs[u, 0]
                a1 = a1 + w * ovs[u, 1]
                a2 = a2 + w * ovs[u, 2]
                a3 = a3 + w * ovs[u, 3]
            for lg in cutlass.range_constexpr(self.gps.bit_length() - 1):
                off = self.gps >> (lg + 1)
                lsum = lsum + cute.arch.shuffle_sync_bfly(lsum, offset=off)
                a0 = a0 + cute.arch.shuffle_sync_bfly(a0, offset=off)
                a1 = a1 + cute.arch.shuffle_sync_bfly(a1, offset=off)
                a2 = a2 + cute.arch.shuffle_sync_bfly(a2, offset=off)
                a3 = a3 + cute.arch.shuffle_sync_bfly(a3, offset=off)
            if active:
                if jg == 0:
                    scl = Float32(
                        0.0
                    )  # a row without any visible key (padding, L < q_len) gives 0, not NaN
                    if lsum > Float32(0.0):
                        scl = sScal[0] / lsum
                    accs = (a0, a1, a2, a3)
                    for e in cutlass.range_constexpr(4):
                        res = (accs[e] * scl).to(cutlass.BFloat16)
                        if const_expr(self.has_gate):
                            res = (res.to(Float32) * sGate[4 * q + e]).to(cutlass.BFloat16)
                        mOut[tok, col + e] = res
        cute.arch.sync_threads()
        if warp == 1:
            cute.arch.relinquish_tmem_alloc_permit()
            cute.arch.dealloc_tmem(tmem_ptr, self.tmem_cols)
