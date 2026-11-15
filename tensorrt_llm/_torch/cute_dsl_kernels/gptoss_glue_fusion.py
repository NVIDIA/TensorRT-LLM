# Copyright (c) 2025 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-FileCopyrightText: Copyright (c) 2025 by FlashInfer team.
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Modified by NVIDIA in 2026: the FlashInfer fused add+RMSNorm kernel is adapted into the two fused GPT-OSS
# glue kernels below (MoE finalize + add+RMSNorm, and add+RMSNorm + MXFP8 quantization).
#
# Attribution from FlashInfer's NOTICE file (Apache-2.0 §4(d)): FlashInfer, Copyright 2025-2026 NVIDIA,
# Copyright 2023-2026 FlashInfer community (https://flashinfer.ai/).
"""Bit-exact glue fusions for GPT-OSS layers that have no all-reduce to fuse into.

Two FlashInfer-derived CuTe DSL kernels, each opt-in behind its own environment variable (read by
``modeling_gpt_oss.py``) and each bitwise equal to the kernel sequence it replaces:

``moe_finalize_add_rmsnorm`` (``TLLM_GPTOSS_FUSE_FINALIZE_NORM=1``)
    Replaces the TRTLLM-Gen MoE finalize kernel (``moe::dev::finalize::finalizeKernel`` /
    ``finalizeKernelVecLoad``) and the next layer's ``flashinfer.norm.fused_add_rmsnorm``. Per token it
    gathers the ``top_k`` unfinalized expert rows of FC2, sums them in fp32 in expert order k = 0..top_k-1
    exactly as the finalize kernel does (``acc = fma(w_k, x_k, acc)`` from ``acc = 0``, skipping
    ``expanded_idx_to_permuted_idx == -1``, one round to bf16), and then runs FlashInfer's fused
    residual-add + RMSNorm body on that bf16 row unchanged. The finalized MoE row never goes to memory.

``fused_add_rmsnorm_mxfp8`` (``TLLM_GPTOSS_FUSE_NORM_MXFP8=1``)
    Replaces the post-attention ``fused_add_rmsnorm`` and the MoE's ``mxfp8_quantize`` (W4A8 MXFP4 x
    MXFP8): FlashInfer's body unchanged, writing fresh BF16 outputs, then the BF16 output it holds in
    registers is quantized with ``quantize_with_block_size<FP16_TO_MXFP8, bf16, 32>``'s instruction
    sequence (inline PTX, so the DSL's float lowering cannot change it).

Both kernels copy ``FusedAddRMSNormKernel.kernel`` of FlashInfer 0.6.18 (the version TensorRT-LLM pins);
the installed class source is fingerprinted before use, so the norm arithmetic (the fp32 ``h``, its
reduction tree, ``rsqrt``, ``h * rstd * w``) is FlashInfer's own code. The same technique (extend
FlashInfer's CuTe add+RMSNorm with another input while keeping its rounding points) is used upstream for
DeepSeek's shared+routed add (NVIDIA/TensorRT-LLM PR #17083).
"""

import ast
import functools
import hashlib
import inspect
import math
import os
import sys
import time
from typing import List, Optional, Tuple

import cutlass
import cutlass.cute as cute
import flashinfer
import torch
from cutlass import Float32, Int32, Int64, Uint32
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op
from flashinfer.norm.kernels import fused_add_rmsnorm as _fi_far
from flashinfer.norm.kernels.fused_add_rmsnorm import FusedAddRMSNormKernel
from flashinfer.norm.kernels.rmsnorm import RMSNormKernel
from flashinfer.norm.utils import get_ptr_as_int64, predicate_k, row_reduce_sum_multirow

FINALIZE_NORM_ENV = "TLLM_GPTOSS_FUSE_FINALIZE_NORM"
NORM_MXFP8_ENV = "TLLM_GPTOSS_FUSE_NORM_MXFP8"
SF_VEC_SIZE = 32
# FlashInfer 0.6.18 flashinfer/norm/kernels/fused_add_rmsnorm.py, class FusedAddRMSNormKernel: sha256 of its
# source text (the PyPI wheel's file is byte-identical to the v0.6.18 tag). Both kernels below copy its
# ``kernel`` body.
_FI_VERSION = "0.6.18"
_FI_CLASS_SHA256 = "fc4cdcdb9661915bacc44a9120e061eb13c80a9c58fc7e89c7fce2e7511547d9"
# How the TRTLLM-Gen finalize kernel accumulates ``data += float(w) * float(x)``: nvcc contracts it to one
# fma.rn.f32 per expert (default --fmad=true). False selects mul.rn + add.rn instead.
FINALIZE_USES_FMA = True
# Vector blocks whose 4 expert rows each thread loads before summing any (memory-level parallelism of the gather),
# and whether those loads skip L1. Arithmetic does not depend on either.
FINALIZE_BLOCKS_PER_LOAD = 6
FINALIZE_L1_NO_ALLOCATE = False

_logged = set()


def log_once(key: str, msg: str) -> None:
    if key not in _logged:
        _logged.add(key)
        print(f"[TRT-LLM] [I] {msg}", file=sys.stderr, flush=True)


# =============================================================================
# PTX: the MoE finalize gather (moe::dev::finalize::finalizeKernel[VecLoad])
# =============================================================================


@dsl_user_op
def _load_route_top4(
    row: Int64, num_rows: Int64, idx_addr: Int64, w_addr: Int64, *, loc=None, ip=None
) -> Tuple[Int32, Int32, Int32, Int32, Uint32, Uint32, Uint32, Uint32]:
    """The row's 4 permuted-row indices (-1 when the row is out of bounds) and its 4 bf16 routing weights
    as fp32 bit patterns (``static_cast<float>(bf16)``, exact)."""
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32()] * 8),
        [Int64(v).ir_value(loc=loc, ip=ip) for v in (row, num_rows, idx_addr, w_addr)],
        """
        {
            .reg .pred pv;
            .reg .b32 i0, i1, i2, i3, wa, wb;
            setp.lt.s64 pv, $8, $9;
            mov.b32 i0, 0xFFFFFFFF;
            mov.b32 i1, 0xFFFFFFFF;
            mov.b32 i2, 0xFFFFFFFF;
            mov.b32 i3, 0xFFFFFFFF;
            mov.b32 wa, 0;
            mov.b32 wb, 0;
            @pv ld.global.v4.b32 {i0, i1, i2, i3}, [$10];
            @pv ld.global.v2.b32 {wa, wb}, [$11];
            mov.b32 $0, i0;
            mov.b32 $1, i1;
            mov.b32 $2, i2;
            mov.b32 $3, i3;
            shl.b32 $4, wa, 16;
            and.b32 $5, wa, 0xFFFF0000;
            shl.b32 $6, wb, 16;
            and.b32 $7, wb, 0xFFFF0000;
        }
        """,
        "=r,=r,=r,=r,=r,=r,=r,=r,l,l,l,l",
        # Loads: keep them after griddepcontrol.wait (PDL) and never hoisted.
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    vals = [llvm.extractvalue(T.i32(), result, [i], loc=loc, ip=ip) for i in range(8)]
    return tuple(Int32(v) for v in vals[:4]) + tuple(Uint32(v) for v in vals[4:])


@functools.cache
def _finalize_asm(
    hidden: int, row_bytes: int, use_fma: bool, nblk: int, l1_no_allocate: bool
) -> str:
    """``nblk`` vector blocks (8 consecutive bf16 each) of one token's finalized MoE row, as 4 bf16x2 words per
    block.

    Expert k contributes to a block when its permuted index is not -1 and the block's column is inside the hidden
    size (outside it the kernel holds +0, as FlashInfer's zero-filled tile does). All 4 * nblk rows are loaded
    first, then each block is accumulated in k order from +0.0 and rounded once (cvt.rn.bf16x2.f32), as the
    finalize kernel does per element. Operands: $0..$(4n-1) the words; then the FC2 base address, the n block
    columns (elements), the 4 permuted indices and the 4 weights (fp32 bits).
    """
    n = nblk
    o = 4 * n
    base = f"${o}"

    def col(b):
        return f"${o + 1 + b}"

    def pidx(k):
        return f"${o + 1 + n + k}"

    def wgt(k):
        return f"${o + 5 + n + k}"

    hint = ".L1::no_allocate" if l1_no_allocate else ""
    lines = [
        "{",
        f".reg .pred pc<{n}>, q<{4 * n}>;",
        f".reg .b64 cb<{n}>, a<{4 * n}>;",
        f".reg .b32 t<{16 * n}>, u;",
        ".reg .f32 c<8>, x<8>, sc, m;",
    ]
    for b in range(n):
        lines.append(f"setp.lt.s64 pc{b}, {col(b)}, {hidden};")
        for k in range(4):
            lines.append(f"setp.ne.and.s32 q{4 * b + k}, {pidx(k)}, -1, pc{b};")
        lines += [f"shl.b64 cb{b}, {col(b)}, 1;", f"add.s64 cb{b}, cb{b}, {base};"]
        for k in range(4):
            lines.append(f"mad.wide.s32 a{4 * b + k}, {pidx(k)}, {row_bytes}, cb{b};")
    for b in range(n):
        for k in range(4):
            r = 16 * b + 4 * k
            lines.append(
                f"@q{4 * b + k} ld.global{hint}.v4.b32 {{t{r}, t{r + 1}, t{r + 2}, t{r + 3}}}, [a{4 * b + k}];"
            )
    for b in range(n):
        for j in range(8):
            lines.append(f"mov.f32 c{j}, 0f00000000;")
        for k in range(4):
            lines.append(f"mov.b32 sc, {wgt(k)};")
            for w in range(4):
                r = 16 * b + 4 * k + w
                lines.append(f"shl.b32 u, t{r}, 16;")
                lines.append(f"mov.b32 x{2 * w}, u;")
                lines.append(f"and.b32 u, t{r}, 0xFFFF0000;")
                lines.append(f"mov.b32 x{2 * w + 1}, u;")
            for j in range(8):
                if use_fma:
                    lines.append(f"@q{4 * b + k} fma.rn.f32 c{j}, sc, x{j}, c{j};")
                else:
                    lines.append(f"@q{4 * b + k} mul.rn.f32 m, sc, x{j};")
                    lines.append(f"@q{4 * b + k} add.rn.f32 c{j}, c{j}, m;")
        for w in range(4):
            lines.append(f"cvt.rn.bf16x2.f32 ${4 * b + w}, c{2 * w + 1}, c{2 * w};")
    lines.append("}")
    return "\n".join(lines)


@functools.cache
def _make_finalize_blocks(
    hidden: int, row_bytes: int, use_fma: bool, nblk: int, l1_no_allocate: bool
):
    """The gather op for one kernel configuration (the PTX takes the hidden size and row stride as immediates)."""
    asm = _finalize_asm(hidden, row_bytes, use_fma, nblk, l1_no_allocate)
    constraints = ",".join(["=r"] * (4 * nblk) + ["l"] * (1 + nblk) + ["r"] * 8)

    @dsl_user_op
    def finalize_blocks(base: Int64, cols, p, s, *, loc=None, ip=None) -> Tuple[Uint32, ...]:
        operands = [Int64(base).ir_value(loc=loc, ip=ip)]
        operands += [Int64(c).ir_value(loc=loc, ip=ip) for c in cols]
        operands += [Int32(v).ir_value(loc=loc, ip=ip) for v in p]
        operands += [Uint32(v).ir_value(loc=loc, ip=ip) for v in s]
        result = llvm.inline_asm(
            llvm.StructType.get_literal([T.i32()] * (4 * nblk)),
            operands,
            asm,
            constraints,
            # Loads: keep them after griddepcontrol.wait (PDL) and never hoisted.
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
        return tuple(
            Uint32(llvm.extractvalue(T.i32(), result, [i], loc=loc, ip=ip)) for i in range(4 * nblk)
        )

    return finalize_blocks


# =============================================================================
# PTX: the quantize_with_block_size<FP16_TO_MXFP8, bf16, 32> sequence
# =============================================================================


@dsl_user_op
def _absmax_bf16x8(w0: Uint32, w1: Uint32, w2: Uint32, w3: Uint32, *, loc=None, ip=None) -> Uint32:
    """cvt_warp_fp16_to_mxfp8: localMax = abs(elts[0]); localMax = max(localMax, abs(elts[i])), i = 1..3."""
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [Uint32(w).ir_value(loc=loc, ip=ip) for w in (w0, w1, w2, w3)],
            """
            {
                .reg .b32 am, aa;
                abs.bf16x2 am, $1;
                abs.bf16x2 aa, $2;
                max.bf16x2 am, am, aa;
                abs.bf16x2 aa, $3;
                max.bf16x2 am, am, aa;
                abs.bf16x2 aa, $4;
                max.bf16x2 am, am, aa;
                mov.b32 $0, am;
            }
            """,
            "=r,r,r,r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def _max_bf16x2(a: Uint32, b: Uint32, *, loc=None, ip=None) -> Uint32:
    """cuda_max(bf162, bf162) = __hmax2 = max.bf16x2."""
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [Uint32(a).ir_value(loc=loc, ip=ip), Uint32(b).ir_value(loc=loc, ip=ip)],
            "max.bf16x2 $0, $1, $2;",
            "=r,r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def _mxfp8_quant8(
    m: Uint32, w0: Uint32, w1: Uint32, w2: Uint32, w3: Uint32, *, loc=None, ip=None
) -> Tuple[Uint32, Uint32, Uint32]:
    """Scale and 8 E4M3 values from the 32-block's bf16x2 max ``m`` and this thread's 8 BF16 values.

    Returns (E8M0 byte, E4M3 bytes 0-3, E4M3 bytes 4-7), little-endian, element 0 lowest. Outputs are written
    only after every input is read.
    """
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32(), T.i32(), T.i32()]),
        [Uint32(v).ir_value(loc=loc, ip=ip) for v in (m, w0, w1, w2, w3)],
        """
        {
            .reg .b16 mlo, mhi, sfp, q01, q23, q45, q67;
            .reg .b32 ta, tb, sfb;
            .reg .f32 fx, fy, vmax, c448, r448, sfv, zero, sfd, osc;
            .reg .f32 v0, v1, v2, v3, v4, v5, v6, v7;
            .reg .pred pgt, pz, pn, pnz;
            // vecMax = float(cuda_max(localMax.x, localMax.y)); cuda_max(T, T) is (a > b) ? a : b.
            mov.b32 {mlo, mhi}, $3;
            cvt.u32.u16 ta, mlo;
            shl.b32 ta, ta, 16;
            mov.b32 fx, ta;
            cvt.u32.u16 tb, mhi;
            shl.b32 tb, tb, 16;
            mov.b32 fy, tb;
            setp.gt.f32 pgt, fx, fy;
            selp.f32 vmax, fx, fy, pgt;
            // SFValue = vecMax * reciprocal_approximate_ftz(448.0f)
            mov.f32 c448, 0f43E00000;
            rcp.approx.ftz.f32 r448, c448;
            mul.rn.f32 sfv, vmax, r448;
            // __nv_cvt_float_to_e8m0(SFValue, __NV_SATFINITE, cudaRoundPosInf) on sm_100f
            mov.f32 zero, 0f00000000;
            cvt.rp.satfinite.ue8m0x2.f32 sfp, zero, sfv;
            cvt.u32.u16 sfb, sfp;
            and.b32 sfb, sfb, 0xFF;
            // SFValue = float(e8m0) = 2^(e-127); e == 0 is 2^-127, e == 255 is NaN
            shl.b32 ta, sfb, 23;
            setp.eq.u32 pz, sfb, 0;
            selp.b32 ta, 0x00400000, ta, pz;
            setp.eq.u32 pn, sfb, 255;
            selp.b32 ta, 0x7FFFFFFF, ta, pn;
            mov.b32 sfd, ta;
            // outputScale = vecMax != 0.f ? reciprocal_approximate_ftz(SFValue) : 0.0f
            rcp.approx.ftz.f32 osc, sfd;
            setp.neu.f32 pnz, vmax, 0f00000000;
            selp.f32 osc, osc, 0f00000000, pnz;
            // __bfloat1622float2, * outputScale, __nv_fp8x2_e4m3(float2)
            shl.b32 ta, $4, 16;
            mov.b32 v0, ta;
            and.b32 tb, $4, 0xFFFF0000;
            mov.b32 v1, tb;
            shl.b32 ta, $5, 16;
            mov.b32 v2, ta;
            and.b32 tb, $5, 0xFFFF0000;
            mov.b32 v3, tb;
            shl.b32 ta, $6, 16;
            mov.b32 v4, ta;
            and.b32 tb, $6, 0xFFFF0000;
            mov.b32 v5, tb;
            shl.b32 ta, $7, 16;
            mov.b32 v6, ta;
            and.b32 tb, $7, 0xFFFF0000;
            mov.b32 v7, tb;
            mul.rn.f32 v0, v0, osc;
            mul.rn.f32 v1, v1, osc;
            mul.rn.f32 v2, v2, osc;
            mul.rn.f32 v3, v3, osc;
            mul.rn.f32 v4, v4, osc;
            mul.rn.f32 v5, v5, osc;
            mul.rn.f32 v6, v6, osc;
            mul.rn.f32 v7, v7, osc;
            cvt.rn.satfinite.e4m3x2.f32 q01, v1, v0;
            cvt.rn.satfinite.e4m3x2.f32 q23, v3, v2;
            cvt.rn.satfinite.e4m3x2.f32 q45, v5, v4;
            cvt.rn.satfinite.e4m3x2.f32 q67, v7, v6;
            mov.b32 $0, sfb;
            mov.b32 $1, {q01, q23};
            mov.b32 $2, {q45, q67};
        }
        """,
        "=r,=r,=r,r,r,r,r,r",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    sf = llvm.extractvalue(T.i32(), result, [0], loc=loc, ip=ip)
    lo = llvm.extractvalue(T.i32(), result, [1], loc=loc, ip=ip)
    hi = llvm.extractvalue(T.i32(), result, [2], loc=loc, ip=ip)
    return Uint32(sf), Uint32(lo), Uint32(hi)


@dsl_user_op
def _st_global_v2_b32(addr: Int64, lo: Uint32, hi: Uint32, *, loc=None, ip=None):
    llvm.inline_asm(
        None,
        [
            Int64(addr).ir_value(loc=loc, ip=ip),
            Uint32(lo).ir_value(loc=loc, ip=ip),
            Uint32(hi).ir_value(loc=loc, ip=ip),
        ],
        "st.global.v2.b32 [$0], {$1, $2};",
        "l,r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@dsl_user_op
def _st_global_b8(addr: Int64, val: Uint32, *, loc=None, ip=None):
    llvm.inline_asm(
        None,
        [Int64(addr).ir_value(loc=loc, ip=ip), Uint32(val).ir_value(loc=loc, ip=ip)],
        """
        {
            .reg .b16 sb;
            cvt.u16.u32 sb, $1;
            st.global.b8 [$0], sb;
        }
        """,
        "l,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


# =============================================================================
# Kernel 1: MoE finalize + residual add + RMSNorm
# =============================================================================


class MoeFinalizeAddRMSNormKernel(FusedAddRMSNormKernel):
    """FlashInfer's FusedAddRMSNormKernel (configuration inherited) whose ``input`` row is the MoE finalize of
    ``top_k`` gathered FC2 rows instead of a tensor in memory, writing fresh outputs.

    The gather needs 8 contiguous BF16 per thread per vector block (one 16-byte load per expert row), which is
    the geometry FlashInfer picks for BF16 at any hidden size that is a multiple of 8; it is checked here.
    """

    def __init__(
        self,
        dtype: cutlass.Numeric,
        H: int,
        Hp: int,
        top_k: int,
        use_fma: bool,
        sm_version: int,
        blocks_per_load: int = 1,
        l1_no_allocate: bool = False,
    ):
        super().__init__(dtype, H, 0.0, sm_version=sm_version)
        self.Hp = Hp
        self.top_k = top_k
        self.use_fma = use_fma
        self.l1_no_allocate = l1_no_allocate
        # The thread's vector blocks, in groups whose expert rows are all loaded before any is summed.
        self.block_groups = tuple(
            tuple(range(g, min(g + blocks_per_load, self.num_vec_blocks)))
            for g in range(0, self.num_vec_blocks, blocks_per_load)
        )
        if (
            dtype is not cutlass.BFloat16
            or self.vec_size != 8
            or top_k != 4
            or Hp < H
            or Hp % 8 != 0
        ):
            raise ValueError(
                f"MoE finalize gather needs bf16, 8 values per vector block, top_k 4 and a padded FC2 width "
                f"that is a multiple of 8 and >= H; got vec_size {self.vec_size}, top_k {top_k}, H {H}, Hp {Hp}"
            )

    def _smem_size_in_bytes(self) -> int:
        # FlashInfer's layout minus the input tile: only the residual is staged in shared memory.
        tile_bytes = (
            self.rows_per_block * self.cols_per_tile * (self.dtype.width // 8)
            if self.use_async_copy
            else 0
        )
        if self.cluster_n == 1:
            reduction_bytes = self.rows_per_block * self.warps_per_row * 4
        else:
            reduction_bytes = self.rows_per_block * self.warps_per_row * self.cluster_n * 4
        mbar_bytes = 8 if self.cluster_n > 1 else 0
        return tile_bytes + reduction_bytes + mbar_bytes + 16

    @cute.jit
    def __call__(
        self,
        mG: cute.Tensor,
        mI: cute.Tensor,
        mE: cute.Tensor,
        mR: cute.Tensor,
        mW: cute.Tensor,
        mO: cute.Tensor,
        mRo: cute.Tensor,
        M: Int64,
        eps: Float32,
        enable_pdl: cutlass.Constexpr[bool],
        stream,
    ):
        tv_shape, tv_stride = RMSNormKernel._make_tv_layout(
            self.threads_per_row,
            self.rows_per_block,
            self.vec_size,
            self.num_vec_blocks,
        )
        tv_layout = cute.make_layout(tv_shape, stride=tv_stride)
        tiler_mn = (self.rows_per_block, self.cols_per_tile)

        cluster_n = self.cluster_n

        self.kernel(mG, mI, mE, mR, mW, mO, mRo, M, eps, enable_pdl, tv_layout, tiler_mn).launch(
            grid=[cute.ceil_div(M, self.rows_per_block), cluster_n, 1],
            block=[self.num_threads, 1, 1],
            cluster=[1, cluster_n, 1] if cutlass.const_expr(cluster_n > 1) else None,
            smem=self._smem_size_in_bytes(),
            stream=stream,
            use_pdl=enable_pdl,
        )

    @cute.kernel
    def kernel(
        self,
        mG: cute.Tensor,
        mI: cute.Tensor,
        mE: cute.Tensor,
        mR: cute.Tensor,
        mW: cute.Tensor,
        mO: cute.Tensor,
        mRo: cute.Tensor,
        M: Int64,
        eps: Float32,
        enable_pdl: cutlass.Constexpr[bool],
        tv_layout: cute.Layout,
        tiler_mn: cute.Shape,
    ):
        # ---- FlashInfer 0.6.18 FusedAddRMSNormKernel.kernel, unchanged except where marked [FINALIZE] ----
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()

        # PDL: Wait for previous kernel (SM90+ only)
        if enable_pdl:
            cute.arch.griddepcontrol_wait()

        H = self.H
        cluster_n = self.cluster_n
        weight_bias = self.weight_bias
        copy_bits = self.copy_bits
        threads_per_row = tv_layout.shape[0][0]
        rows_per_block = tiler_mn[0]
        warps_per_row = max(threads_per_row // 32, 1)

        if cutlass.const_expr(cluster_n > 1):
            cluster_y = cute.arch.block_idx()[1]
        else:
            cluster_y = cutlass.const_expr(0)

        # ===== Allocate shared memory =====
        smem = cutlass.utils.SmemAllocator()

        if cutlass.const_expr(self.use_async_copy):
            # [FINALIZE] only the residual tile; the input row is computed in registers
            sR = smem.allocate_tensor(
                mR.element_type,
                cute.make_ordered_layout(tiler_mn, order=(1, 0)),
                byte_alignment=16,
            )

        if cutlass.const_expr(cluster_n == 1):
            reduction_buffer = smem.allocate_tensor(
                Float32,
                cute.make_layout((rows_per_block, warps_per_row)),
                byte_alignment=4,
            )
            mbar_ptr = None
        else:
            reduction_buffer = smem.allocate_tensor(
                Float32,
                cute.make_layout((rows_per_block, (warps_per_row, cluster_n))),
                byte_alignment=4,
            )
            mbar_ptr = smem.allocate_array(cutlass.Int64, num_elems=1)

        # ===== Initialize cluster =====
        if cutlass.const_expr(cluster_n > 1):
            if tidx == 0:
                cute.arch.mbarrier_init(mbar_ptr, 1)
            cute.arch.mbarrier_init_fence()
            cute.arch.cluster_arrive_relaxed()
            cute.arch.cluster_wait()

        # ===== Coordinate tracking and tiling =====
        idX = cute.make_identity_tensor(mR.shape)

        gR = cute.local_tile(mR, tiler_mn, (bidx, cluster_y))
        cX = cute.local_tile(idX, tiler_mn, (bidx, cluster_y))
        # [FINALIZE] fresh outputs instead of writing the input / residual in place
        gO = cute.local_tile(mO, tiler_mn, (bidx, cluster_y))
        gRo = cute.local_tile(mRo, tiler_mn, (bidx, cluster_y))

        mW_expanded_layout = cute.prepend(mW.layout, cute.make_layout((tiler_mn[0],), stride=(0,)))
        mW_2d = cute.make_tensor(mW.iterator, mW_expanded_layout)
        gW = cute.local_tile(mW_2d, tiler_mn, (0, cluster_y))

        # ===== Create TiledCopy atoms =====
        copy_atom_sync = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            mR.element_type,
            num_bits_per_copy=copy_bits,
        )
        copy_atom_store = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            mR.element_type,
            num_bits_per_copy=copy_bits,
        )

        if cutlass.const_expr(self.use_async_copy):
            copy_atom_async = cute.make_copy_atom(
                cute.nvgpu.cpasync.CopyG2SOp(),
                mR.element_type,
                num_bits_per_copy=copy_bits,
            )
            tiled_copy_load = cute.make_tiled_copy(copy_atom_async, tv_layout, tiler_mn)
        else:
            tiled_copy_load = cute.make_tiled_copy(copy_atom_sync, tv_layout, tiler_mn)

        tiled_copy_W = cute.make_tiled_copy(copy_atom_sync, tv_layout, tiler_mn)
        tiled_copy_store = cute.make_tiled_copy(copy_atom_store, tv_layout, tiler_mn)

        thr_copy_X = tiled_copy_load.get_slice(tidx)
        thr_copy_W = tiled_copy_W.get_slice(tidx)
        thr_copy_O = tiled_copy_store.get_slice(tidx)

        # Partition coordinates; the input registers have the residual fragment's shape
        tXcX = thr_copy_X.partition_S(cX)

        # Partition residual (same load tiled copy)
        tRgR = thr_copy_X.partition_S(gR)
        tRrR = cute.make_fragment_like(tRgR)
        tXrX = cute.make_fragment_like(tRgR)

        if cutlass.const_expr(self.use_async_copy):
            tRsR = thr_copy_X.partition_D(sR)

        # Partition weight (sync, separate tiled copy)
        tWgW = thr_copy_W.partition_S(gW)
        tWrW = cute.make_fragment_like(tWgW)
        tXrW = thr_copy_X.retile(tWrW)

        # Partition output destinations ([FINALIZE] gO / gRo)
        tXgO = thr_copy_O.partition_D(gO)
        tRgO = thr_copy_O.partition_D(gRo)
        tXrO = cute.make_fragment_like(tXgO)

        # ===== Bounds checking =====
        tXpX = predicate_k(tXcX, limit=H)
        tWpW = predicate_k(thr_copy_W.partition_S(cX), limit=H)
        row_coord = tXcX[(0, 0), 0, 0]
        row_in_bounds = row_coord[0] < M

        # ===== Pass 1: Load residual (FlashInfer's pipeline), finalize the input row, compute h, reduce =====
        if cutlass.const_expr(self.use_async_copy):
            if row_in_bounds:
                cute.copy(copy_atom_async, tRgR, tRsR, pred=tXpX)
            cute.arch.cp_async_commit_group()

            cute.copy(copy_atom_sync, tWgW, tWrW, pred=tWpW)
        else:
            tRrR.store(cute.zeros_like(tRrR, dtype=mR.element_type))
            if row_in_bounds:
                cute.copy(copy_atom_sync, tRgR, tRrR, pred=tXpX)

            cute.copy(copy_atom_sync, tWgW, tWrW, pred=tWpW)

        # [FINALIZE] The MoE finalize of this row into tXrX, while the residual copy is in flight. The thread's
        # values are num_vec_blocks blocks of 8 contiguous columns, flat index 8g + v in the fragment and the
        # coordinate tensor alike, so block g is words 4g..4g+3 of the fragment's storage.
        row = Int64(row_coord[0])
        p0, p1, p2, p3, s0, s1, s2, s3 = _load_route_top4(
            row,
            M,
            get_ptr_as_int64(mI, row * Int64(self.top_k)),
            get_ptr_as_int64(mE, row * Int64(self.top_k)),
        )
        g_base = get_ptr_as_int64(mG, Int64(0))
        words = cute.recast_tensor(
            cute.make_tensor(tXrX.iterator, cute.make_layout(cute.cosize(tXrX.layout))), Uint32
        )
        for gi in cutlass.range_constexpr(len(self.block_groups)):
            group = self.block_groups[gi]
            finalize_blocks = _make_finalize_blocks(
                H, self.Hp * (self.dtype.width // 8), self.use_fma, len(group), self.l1_no_allocate
            )
            cols = [Int64(tXcX[self.vec_size * g][1]) for g in group]
            ws = finalize_blocks(g_base, cols, (p0, p1, p2, p3), (s0, s1, s2, s3))
            for j in cutlass.range_constexpr(len(group)):
                for i in cutlass.range_constexpr(4):
                    words[4 * group[j] + i] = ws[4 * j + i]

        if cutlass.const_expr(self.use_async_copy):
            cute.arch.cp_async_wait_group(0)

            cute.autovec_copy(tRsR, tRrR)

        x_in = tXrX.load().to(Float32)
        r_in = tRrR.load().to(Float32)
        h = x_in + r_in

        # Write h to residual (global)
        tXrO.store(h.to(mR.element_type))
        if row_in_bounds:
            cute.copy(copy_atom_store, tXrO, tRgO, pred=tXpX)

        h_sq = h * h
        sum_sq = row_reduce_sum_multirow(
            h_sq, threads_per_row, reduction_buffer, mbar_ptr, cluster_n
        )

        mean_sq = sum_sq / Float32(H)
        rstd = cute.math.rsqrt(mean_sq + eps, fastmath=True)

        if cutlass.const_expr(cluster_n > 1):
            cute.arch.cluster_arrive_relaxed()
            cute.arch.cluster_wait()
        else:
            cute.arch.barrier()

        # ===== Pass 2: Normalize and store output =====
        w = tXrW.load().to(Float32)
        y = h * rstd * (w + Float32(weight_bias))

        tXrO.store(y.to(mR.element_type))

        if row_in_bounds:
            cute.copy(copy_atom_store, tXrO, tXgO, pred=tXpX)

        # PDL: Signal dependent kernels (SM90+ only)
        if enable_pdl:
            cute.arch.griddepcontrol_launch_dependents()


# =============================================================================
# Kernel 2: residual add + RMSNorm + MXFP8 quantization of the output
# =============================================================================


class FusedAddRMSNormMxfp8Kernel(FusedAddRMSNormKernel):
    """FlashInfer's FusedAddRMSNormKernel (configuration inherited) with fresh BF16 outputs and an MXFP8
    epilogue.

    The epilogue needs the geometry it was written for, checked here: one CTA per 4 rows (one warp per row, no
    cluster), 8 contiguous BF16 per thread per vector block, so lanes 4k..4k+3 of vector block g hold the 32
    columns of scale block 8g+k, and the tile is exactly the padded width, so every FP8 byte and every scale is
    written once.
    """

    def __init__(self, dtype: cutlass.Numeric, H: int, padded_k: int, sm_version: int):
        super().__init__(dtype, H, 0.0, sm_version=sm_version)
        self.padded_k = padded_k
        geometry = dict(
            cluster_n=self.cluster_n,
            vec_size=self.vec_size,
            threads_per_row=self.threads_per_row,
            num_threads=self.num_threads,
            cols_per_tile=self.cols_per_tile,
        )
        if (
            dtype is not cutlass.BFloat16
            or self.cluster_n != 1
            or self.vec_size != 8
            or self.threads_per_row != 32
            or self.num_threads != 128
            or self.cols_per_tile != padded_k
            or H % SF_VEC_SIZE != 0
            or padded_k % SF_VEC_SIZE != 0
            or padded_k < H
        ):
            raise ValueError(
                f"geometry {geometry} for H={H}, padded_k={padded_k} is not the one the MXFP8 epilogue "
                "handles (bf16, cluster 1, 32 threads x 8 values per row, tile == padded width)"
            )

    @cute.jit
    def __call__(
        self,
        mX: cute.Tensor,
        mR: cute.Tensor,
        mW: cute.Tensor,
        mXo: cute.Tensor,
        mRo: cute.Tensor,
        mQ: cute.Tensor,
        mS: cute.Tensor,
        M: Int64,
        eps: Float32,
        enable_pdl: cutlass.Constexpr[bool],
        stream,
    ):
        tv_shape, tv_stride = RMSNormKernel._make_tv_layout(
            self.threads_per_row,
            self.rows_per_block,
            self.vec_size,
            self.num_vec_blocks,
        )
        tv_layout = cute.make_layout(tv_shape, stride=tv_stride)
        tiler_mn = (self.rows_per_block, self.cols_per_tile)

        cluster_n = self.cluster_n

        self.kernel(mX, mR, mW, mXo, mRo, mQ, mS, M, eps, enable_pdl, tv_layout, tiler_mn).launch(
            grid=[cute.ceil_div(M, self.rows_per_block), cluster_n, 1],
            block=[self.num_threads, 1, 1],
            cluster=[1, cluster_n, 1] if cutlass.const_expr(cluster_n > 1) else None,
            smem=self._smem_size_in_bytes(),
            stream=stream,
            use_pdl=enable_pdl,
        )

    @cute.kernel
    def kernel(
        self,
        mX: cute.Tensor,
        mR: cute.Tensor,
        mW: cute.Tensor,
        mXo: cute.Tensor,
        mRo: cute.Tensor,
        mQ: cute.Tensor,
        mS: cute.Tensor,
        M: Int64,
        eps: Float32,
        enable_pdl: cutlass.Constexpr[bool],
        tv_layout: cute.Layout,
        tiler_mn: cute.Shape,
    ):
        # ---- FlashInfer 0.6.18 FusedAddRMSNormKernel.kernel, unchanged except where marked [MXFP8] ----
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()

        # PDL: Wait for previous kernel (SM90+ only)
        if enable_pdl:
            cute.arch.griddepcontrol_wait()

        H = self.H
        cluster_n = self.cluster_n
        weight_bias = self.weight_bias
        copy_bits = self.copy_bits
        threads_per_row = tv_layout.shape[0][0]
        rows_per_block = tiler_mn[0]
        warps_per_row = max(threads_per_row // 32, 1)

        if cutlass.const_expr(cluster_n > 1):
            cluster_y = cute.arch.block_idx()[1]
        else:
            cluster_y = cutlass.const_expr(0)

        # ===== Allocate shared memory =====
        smem = cutlass.utils.SmemAllocator()

        if cutlass.const_expr(self.use_async_copy):
            sX = smem.allocate_tensor(
                mX.element_type,
                cute.make_ordered_layout(tiler_mn, order=(1, 0)),
                byte_alignment=16,
            )
            sR = smem.allocate_tensor(
                mR.element_type,
                cute.make_ordered_layout(tiler_mn, order=(1, 0)),
                byte_alignment=16,
            )

        if cutlass.const_expr(cluster_n == 1):
            reduction_buffer = smem.allocate_tensor(
                Float32,
                cute.make_layout((rows_per_block, warps_per_row)),
                byte_alignment=4,
            )
            mbar_ptr = None
        else:
            reduction_buffer = smem.allocate_tensor(
                Float32,
                cute.make_layout((rows_per_block, (warps_per_row, cluster_n))),
                byte_alignment=4,
            )
            mbar_ptr = smem.allocate_array(cutlass.Int64, num_elems=1)

        # ===== Initialize cluster =====
        if cutlass.const_expr(cluster_n > 1):
            if tidx == 0:
                cute.arch.mbarrier_init(mbar_ptr, 1)
            cute.arch.mbarrier_init_fence()
            cute.arch.cluster_arrive_relaxed()
            cute.arch.cluster_wait()

        # ===== Coordinate tracking and tiling =====
        idX = cute.make_identity_tensor(mX.shape)

        gX = cute.local_tile(mX, tiler_mn, (bidx, cluster_y))
        gR = cute.local_tile(mR, tiler_mn, (bidx, cluster_y))
        cX = cute.local_tile(idX, tiler_mn, (bidx, cluster_y))
        # [MXFP8] fresh outputs instead of writing mX / mR in place
        gXo = cute.local_tile(mXo, tiler_mn, (bidx, cluster_y))
        gRo = cute.local_tile(mRo, tiler_mn, (bidx, cluster_y))

        mW_expanded_layout = cute.prepend(mW.layout, cute.make_layout((tiler_mn[0],), stride=(0,)))
        mW_2d = cute.make_tensor(mW.iterator, mW_expanded_layout)
        gW = cute.local_tile(mW_2d, tiler_mn, (0, cluster_y))

        # ===== Create TiledCopy atoms =====
        copy_atom_sync = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            mX.element_type,
            num_bits_per_copy=copy_bits,
        )
        copy_atom_store = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            mX.element_type,
            num_bits_per_copy=copy_bits,
        )

        if cutlass.const_expr(self.use_async_copy):
            copy_atom_async = cute.make_copy_atom(
                cute.nvgpu.cpasync.CopyG2SOp(),
                mX.element_type,
                num_bits_per_copy=copy_bits,
            )
            tiled_copy_load = cute.make_tiled_copy(copy_atom_async, tv_layout, tiler_mn)
        else:
            tiled_copy_load = cute.make_tiled_copy(copy_atom_sync, tv_layout, tiler_mn)

        tiled_copy_W = cute.make_tiled_copy(copy_atom_sync, tv_layout, tiler_mn)
        tiled_copy_store = cute.make_tiled_copy(copy_atom_store, tv_layout, tiler_mn)

        thr_copy_X = tiled_copy_load.get_slice(tidx)
        thr_copy_W = tiled_copy_W.get_slice(tidx)
        thr_copy_O = tiled_copy_store.get_slice(tidx)

        # Partition input
        tXgX = thr_copy_X.partition_S(gX)
        tXcX = thr_copy_X.partition_S(cX)
        tXrX = cute.make_fragment_like(tXgX)

        # Partition residual (same load tiled copy)
        tRgR = thr_copy_X.partition_S(gR)
        tRrR = cute.make_fragment_like(tRgR)

        if cutlass.const_expr(self.use_async_copy):
            tXsX = thr_copy_X.partition_D(sX)
            tRsR = thr_copy_X.partition_D(sR)

        # Partition weight (sync, separate tiled copy)
        tWgW = thr_copy_W.partition_S(gW)
        tWrW = cute.make_fragment_like(tWgW)
        tXrW = thr_copy_X.retile(tWrW)

        # Partition output destinations ([MXFP8] gXo / gRo instead of gX / gR)
        tXgO = thr_copy_O.partition_D(gXo)
        tRgO = thr_copy_O.partition_D(gRo)
        tXrO = cute.make_fragment_like(tXgO)

        # ===== Bounds checking =====
        tXpX = predicate_k(tXcX, limit=H)
        tWpW = predicate_k(thr_copy_W.partition_S(cX), limit=H)
        row_coord = tXcX[(0, 0), 0, 0]
        row_in_bounds = row_coord[0] < M

        # ===== Pass 1: Load input + residual, compute h, reduce =====
        if cutlass.const_expr(self.use_async_copy):
            if row_in_bounds:
                cute.copy(copy_atom_async, tXgX, tXsX, pred=tXpX)
                cute.copy(copy_atom_async, tRgR, tRsR, pred=tXpX)
            cute.arch.cp_async_commit_group()

            cute.copy(copy_atom_sync, tWgW, tWrW, pred=tWpW)

            cute.arch.cp_async_wait_group(0)

            cute.autovec_copy(tXsX, tXrX)
            cute.autovec_copy(tRsR, tRrR)
        else:
            tXrX.store(cute.zeros_like(tXrX, dtype=mX.element_type))
            tRrR.store(cute.zeros_like(tRrR, dtype=mR.element_type))
            if row_in_bounds:
                cute.copy(copy_atom_sync, tXgX, tXrX, pred=tXpX)
                cute.copy(copy_atom_sync, tRgR, tRrR, pred=tXpX)

            cute.copy(copy_atom_sync, tWgW, tWrW, pred=tWpW)

        x_in = tXrX.load().to(Float32)
        r_in = tRrR.load().to(Float32)
        h = x_in + r_in

        # Write h to residual (global)
        tXrO.store(h.to(mR.element_type))
        if row_in_bounds:
            cute.copy(copy_atom_store, tXrO, tRgO, pred=tXpX)

        h_sq = h * h
        sum_sq = row_reduce_sum_multirow(
            h_sq, threads_per_row, reduction_buffer, mbar_ptr, cluster_n
        )

        mean_sq = sum_sq / Float32(H)
        rstd = cute.math.rsqrt(mean_sq + eps, fastmath=True)

        if cutlass.const_expr(cluster_n > 1):
            cute.arch.cluster_arrive_relaxed()
            cute.arch.cluster_wait()
        else:
            cute.arch.barrier()

        # ===== Pass 2: Normalize and store output =====
        w = tXrW.load().to(Float32)
        y = h * rstd * (w + Float32(weight_bias))

        tXrO.store(y.to(mX.element_type))

        if row_in_bounds:
            cute.copy(copy_atom_store, tXrO, tXgO, pred=tXpX)

        # ===== [MXFP8] Pass 3: MXFP8 of the BF16 output held in tXrO (quantize_with_block_size<2>) =====
        # The thread's 96 values are 12 vector blocks of 8 contiguous columns (one 128-bit copy each), flat index
        # 8g + v in the fragment, the coordinate tensor and storage alike; tXpX holds one predicate per block
        # (flat g). Lane l of a row holds columns 8l + 256g .. +7 of block g, so lanes 4k..4k+3 hold scale block
        # 8g + k. A 32-column block is all data (col < H) or all padding (H % 32 == 0); padding is written as 0.
        words = cute.recast_tensor(
            cute.make_tensor(tXrO.iterator, cute.make_layout(cute.cosize(tXrO.layout))), Uint32
        )
        row_q = Int64(row_coord[0]) * Int64(self.padded_k)
        row_s = Int64(row_coord[0]) * Int64(self.padded_k // SF_VEC_SIZE)
        is_sf_lane = (tidx % 4) == 0
        for g in cutlass.range_constexpr(self.num_vec_blocks):
            w0 = words[4 * g + 0]
            w1 = words[4 * g + 1]
            w2 = words[4 * g + 2]
            w3 = words[4 * g + 3]
            amax = _absmax_bf16x8(w0, w1, w2, w3)
            amax = _max_bf16x2(cute.arch.shuffle_sync_bfly(amax, offset=1), amax)
            amax = _max_bf16x2(cute.arch.shuffle_sync_bfly(amax, offset=2), amax)
            sf, q_lo, q_hi = _mxfp8_quant8(amax, w0, w1, w2, w3)
            col = tXcX[self.vec_size * g][1]
            if row_in_bounds:
                q_addr = get_ptr_as_int64(mQ, row_q + Int64(col))
                s_addr = get_ptr_as_int64(mS, row_s + Int64(col // SF_VEC_SIZE))
                if tXpX[g]:
                    _st_global_v2_b32(q_addr, q_lo, q_hi)
                    if is_sf_lane:
                        _st_global_b8(s_addr, sf)
                else:
                    _st_global_v2_b32(q_addr, Uint32(0), Uint32(0))
                    if is_sf_lane:
                        _st_global_b8(s_addr, Uint32(0))

        # PDL: Signal dependent kernels (SM90+ only)
        if enable_pdl:
            cute.arch.griddepcontrol_launch_dependents()


# =============================================================================
# Host side
# =============================================================================


def padded_width(hidden: int, alignment: int) -> int:
    """mxfp8_quantize's padded_k: ((k + alignment - 1) / alignment) * alignment."""
    return (hidden + alignment - 1) // alignment * alignment


def enable_pdl(device: torch.device) -> bool:
    """What flashinfer_fused_add_rmsnorm resolves: enable_pdl=get_env_enable_pdl(), then device_support_pdl."""
    from flashinfer.utils import device_support_pdl

    from tensorrt_llm._torch.flashinfer_utils import get_env_enable_pdl

    return bool(get_env_enable_pdl() and device_support_pdl(device))


@functools.lru_cache(maxsize=16)
def sm_version_of(device: torch.device) -> int:
    major, minor = torch.cuda.get_device_capability(device)
    return major * 10 + minor


@functools.cache
def flashinfer_check() -> Optional[str]:
    """None when the installed FlashInfer runs the norm these kernels copy, else why not."""
    if flashinfer.__version__ != _FI_VERSION:
        return f"FlashInfer {flashinfer.__version__}, the kernels copy {_FI_VERSION}"
    path = inspect.getsourcefile(_fi_far)
    with open(path, encoding="utf-8") as f:
        src = f.read()
    cls = next(
        n
        for n in ast.parse(src).body
        if isinstance(n, ast.ClassDef) and n.name == "FusedAddRMSNormKernel"
    )
    text = "\n".join(src.splitlines()[cls.lineno - 1 : cls.end_lineno])
    digest = hashlib.sha256(text.encode()).hexdigest()
    if digest != _FI_CLASS_SHA256:
        return f"FusedAddRMSNormKernel source differs from {_FI_VERSION} (sha256 {digest[:12]})"
    return None


def flashinfer_uses_cute_norm(device: torch.device) -> bool:
    """FlashInfer runs its CuTe-DSL fused_add_rmsnorm (the code copied here), not its CUDA JIT kernel."""
    import flashinfer.norm as _fi_norm

    return not _fi_norm._use_cuda_norm(device)


# A failed compile, by cache key: functools.cache does not keep exceptions, so without this every caller would
# compile (seconds) and fail again.
_COMPILE_FAILURES: dict = {}


def _cached_compile(fn, key):
    if key in _COMPILE_FAILURES:
        raise RuntimeError(_COMPILE_FAILURES[key])
    try:
        return fn(*key)
    except Exception as e:
        arch = os.environ.get("CUTE_DSL_ARCH") or "the DSL default"
        _COMPILE_FAILURES[key] = f"CuTe DSL compile failed for target {arch}: {e!r}"
        raise RuntimeError(_COMPILE_FAILURES[key]) from e


@functools.cache
def _compile_finalize_norm(
    H: int,
    Hp: int,
    top_k: int,
    use_fma: bool,
    pdl: bool,
    sm_version: int,
    blocks_per_load: int,
    l1_no_allocate: bool,
):
    dtype = cutlass.BFloat16
    kernel_obj = MoeFinalizeAddRMSNormKernel(
        dtype, H, Hp, top_k, use_fma, sm_version, blocks_per_load, l1_no_allocate
    )
    sym_m = cute.sym_int(64)
    sym_p = cute.sym_int(64)
    align = math.gcd(128, H * (dtype.width // 8))

    def bf16_rows():
        return cute.runtime.make_fake_compact_tensor(
            dtype, (sym_m, H), stride_order=(1, 0), assumed_align=align
        )

    g_fake = cute.runtime.make_fake_compact_tensor(
        dtype, (sym_p, Hp), stride_order=(1, 0), assumed_align=16
    )
    i_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32, (sym_m, top_k), stride_order=(1, 0), assumed_align=16
    )
    e_fake = cute.runtime.make_fake_compact_tensor(
        dtype, (sym_m, top_k), stride_order=(1, 0), assumed_align=8
    )
    w_fake = cute.runtime.make_fake_compact_tensor(dtype, (H,), assumed_align=16)
    stream_fake = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    return cute.compile(
        kernel_obj,
        g_fake,
        i_fake,
        e_fake,
        bf16_rows(),
        w_fake,
        bf16_rows(),
        bf16_rows(),
        Int64(1),
        Float32(1e-6),
        pdl,
        stream_fake,
        options="--enable-tvm-ffi",
    )


def compiled_finalize_norm(
    H: int,
    Hp: int,
    top_k: int,
    use_fma: bool,
    pdl: bool,
    sm_version: int,
    blocks_per_load: Optional[int] = None,
    l1_no_allocate: Optional[bool] = None,
):
    key = (
        H,
        Hp,
        top_k,
        use_fma,
        pdl,
        sm_version,
        FINALIZE_BLOCKS_PER_LOAD if blocks_per_load is None else blocks_per_load,
        FINALIZE_L1_NO_ALLOCATE if l1_no_allocate is None else l1_no_allocate,
    )
    return _cached_compile(_compile_finalize_norm, key)


@functools.cache
def _compile_norm_mxfp8(H: int, padded_k: int, pdl: bool, sm_version: int):
    dtype = cutlass.BFloat16
    kernel_obj = FusedAddRMSNormMxfp8Kernel(dtype, H, padded_k, sm_version)
    sym_m = cute.sym_int(64)
    align = math.gcd(128, H * (dtype.width // 8))

    def bf16_2d():
        return cute.runtime.make_fake_compact_tensor(
            dtype, (sym_m, H), stride_order=(1, 0), assumed_align=align
        )

    w_fake = cute.runtime.make_fake_compact_tensor(dtype, (H,), assumed_align=16)
    q_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Uint8, (sym_m, padded_k), stride_order=(1, 0), assumed_align=math.gcd(128, padded_k)
    )
    s_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Uint8,
        (sym_m, padded_k // SF_VEC_SIZE),
        stride_order=(1, 0),
        assumed_align=math.gcd(128, padded_k // SF_VEC_SIZE),
    )
    stream_fake = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    return cute.compile(
        kernel_obj,
        bf16_2d(),
        bf16_2d(),
        w_fake,
        bf16_2d(),
        bf16_2d(),
        q_fake,
        s_fake,
        Int64(1),
        Float32(1e-6),
        pdl,
        stream_fake,
        options="--enable-tvm-ffi",
    )


def compiled_norm_mxfp8(H: int, padded_k: int, pdl: bool, sm_version: int):
    return _cached_compile(_compile_norm_mxfp8, (H, padded_k, pdl, sm_version))


# =============================================================================
# Ops
# =============================================================================


def finalize_norm_contract(
    gemm2_output: torch.Tensor,
    expert_weights: torch.Tensor,
    expanded_idx_to_permuted_idx: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
) -> Optional[str]:
    """None when the kernel computes exactly these operands' finalize + add+RMSNorm, else why not."""
    if residual.dim() != 2 or gemm2_output.dim() != 2:
        return f"rank: residual {residual.dim()}, gemm2 {gemm2_output.dim()}"
    m, hidden = residual.shape
    if (
        residual.dtype != torch.bfloat16
        or weight.dtype != torch.bfloat16
        or gemm2_output.dtype != torch.bfloat16
        or expert_weights.dtype != torch.bfloat16
        or expanded_idx_to_permuted_idx.dtype != torch.int32
    ):
        return "dtypes"
    if expert_weights.numel() != 4 * m or expanded_idx_to_permuted_idx.numel() != 4 * m:
        return f"top_k: {expert_weights.numel()} weights, {expanded_idx_to_permuted_idx.numel()} indices for {m} rows"
    if (
        gemm2_output.shape[1] < hidden
        or gemm2_output.shape[1] % 8 != 0
        or weight.shape != (hidden,)
    ):
        return (
            f"widths: gemm2 {tuple(gemm2_output.shape)}, weight {tuple(weight.shape)}, H {hidden}"
        )
    if m * hidden > 2**31 - 1:
        return f"{m} x {hidden} rows exceed int32 offsets"
    return None


def _contiguous_aligned(t: torch.Tensor, align: int) -> torch.Tensor:
    """``t`` itself when it is contiguous at an ``align``-byte boundary, else a fresh copy (the caching allocator
    returns 512-byte aligned blocks). A copy changes no value, so the kernel's result is the same."""
    if t.is_contiguous() and t.data_ptr() % align == 0:
        return t
    return t.clone(memory_format=torch.contiguous_format)


@torch.library.custom_op("trtllm::moe_finalize_add_rmsnorm", mutates_args=())
def moe_finalize_add_rmsnorm(
    gemm2_output: torch.Tensor,
    expert_weights: torch.Tensor,
    expanded_idx_to_permuted_idx: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> List[torch.Tensor]:
    """[normed BF16 [M, H], new residual BF16 [M, H]] from the unfinalized TRTLLM-Gen MoE triple.

    ``gemm2_output`` is [P, Hp] BF16 (Hp >= H, the padded FC2 width the finalize kernel strides by),
    ``expert_weights`` [M, 4] BF16, ``expanded_idx_to_permuted_idx`` [M, 4] (or [M * 4]) int32. Equal bit for
    bit to ``moe::dev::finalize::run`` on the triple followed by ``flashinfer.norm.fused_add_rmsnorm``.
    """
    reason = finalize_norm_contract(
        gemm2_output, expert_weights, expanded_idx_to_permuted_idx, residual, weight
    )
    if reason is not None:
        # The model gate admits only layers that meet the contract; reaching this is a bug, not a shape to serve.
        raise ValueError(
            f"{FINALIZE_NORM_ENV}: operands outside the fused kernel's contract: {reason}"
        )
    m, hidden = residual.shape
    # The compiled kernel assumes FlashInfer's 128 B for the rows, 16 B for the FC2 rows, indices and norm weight,
    # 8 B for the routing weights. Contiguous activations from the caching allocator already meet that.
    residual = _contiguous_aligned(residual, 128)
    weight = _contiguous_aligned(weight, 16)
    gemm2_output = _contiguous_aligned(gemm2_output, 16)
    idx = _contiguous_aligned(expanded_idx_to_permuted_idx.reshape(m, 4), 16)
    expert_weights = _contiguous_aligned(expert_weights.reshape(m, 4), 8)
    out = torch.empty_like(residual)
    res = torch.empty_like(residual)
    if m > 0:
        kernel = compiled_finalize_norm(
            hidden,
            gemm2_output.shape[1],
            4,
            FINALIZE_USES_FMA,
            enable_pdl(residual.device),
            sm_version_of(residual.device),
        )
        kernel(gemm2_output, idx, expert_weights, residual, weight, out, res, m, eps)
        log_once(
            "finalize-ran",
            f"{FINALIZE_NORM_ENV}: fused MoE finalize + add+RMSNorm kernel launched "
            f"(M={m}, H={hidden}, FC2 width {gemm2_output.shape[1]}, top_k 4)",
        )
    return [out, res]


@moe_finalize_add_rmsnorm.register_fake
def _(gemm2_output, expert_weights, expanded_idx_to_permuted_idx, residual, weight, eps):
    return [residual.new_empty(residual.shape), residual.new_empty(residual.shape)]


def unfused_add_rmsnorm_mxfp8(
    x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float, alignment: int
):
    """Today's two kernels on copies: the reference the fused kernel equals, and the runtime fallback."""
    norm = x.clone()
    res = residual.clone()
    torch.ops.trtllm.flashinfer_fused_add_rmsnorm(norm, res, weight, eps)
    q, sf = torch.ops.trtllm.mxfp8_quantize(norm, False, alignment=alignment)
    return norm, res, q, sf.view(norm.shape[0], -1)


@torch.library.custom_op("trtllm::fused_add_rmsnorm_mxfp8", mutates_args=())
def fused_add_rmsnorm_mxfp8(
    x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float, alignment: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """(norm BF16, residual BF16, MXFP8 E4M3 [M, K], UE8M0 scales uint8 [M, K/32]) for 2-D BF16 x, residual."""
    m, hidden = x.shape
    padded_k = padded_width(hidden, alignment)
    eligible = (
        x.dtype == torch.bfloat16
        and residual.dtype == torch.bfloat16
        and weight.dtype == torch.bfloat16
        and x.is_contiguous()
        and residual.is_contiguous()
        and weight.is_contiguous()
        and residual.shape == x.shape
        and m * padded_k <= 2**31 - 1
        # the compiled kernel assumes FlashInfer's alignment: 128 B for x and residual, 16 B for w
        and x.data_ptr() % 128 == 0
        and residual.data_ptr() % 128 == 0
        and weight.data_ptr() % 16 == 0
    )
    if not eligible:
        log_once(
            "mxfp8-fallback",
            f"{NORM_MXFP8_ENV}: runtime fallback to the unfused kernels (shape {tuple(x.shape)}, dtypes "
            f"{x.dtype}/{residual.dtype}, contiguous {x.is_contiguous()}/{residual.is_contiguous()}, "
            f"addresses {x.data_ptr() % 128}/{residual.data_ptr() % 128} mod 128)",
        )
        return unfused_add_rmsnorm_mxfp8(x, residual, weight, eps, alignment)
    norm = torch.empty_like(x)
    res = torch.empty_like(residual)
    q = torch.empty((m, padded_k), dtype=torch.float8_e4m3fn, device=x.device)
    sf = torch.empty((m, padded_k // SF_VEC_SIZE), dtype=torch.uint8, device=x.device)
    if m > 0:
        kernel = compiled_norm_mxfp8(
            hidden, padded_k, enable_pdl(x.device), sm_version_of(x.device)
        )
        kernel(x, residual, weight, norm, res, q.view(torch.uint8), sf, m, eps)
        log_once(
            "mxfp8-ran",
            f"{NORM_MXFP8_ENV}: fused add+RMSNorm+MXFP8 kernel launched (M={m}, H={hidden}, K={padded_k})",
        )
    return norm, res, q, sf


@fused_add_rmsnorm_mxfp8.register_fake
def _(x, residual, weight, eps, alignment):
    m, hidden = x.shape
    padded_k = padded_width(hidden, alignment)
    return (
        x.new_empty((m, hidden)),
        residual.new_empty((m, hidden)),
        x.new_empty((m, padded_k), dtype=torch.float8_e4m3fn),
        x.new_empty((m, padded_k // SF_VEC_SIZE), dtype=torch.uint8),
    )


# =============================================================================
# GPT-OSS integration: gates (once per layer at model build) and the layer op
# =============================================================================


def _plain_rmsnorm(norm) -> bool:
    """The RMSNorm module runs FlashInfer's plain fused add+RMSNorm (no Gemma offset, quantization or cuda.tile)."""
    return not (
        getattr(norm, "use_gemma", True)
        or getattr(norm, "is_nvfp4", True)
        or getattr(norm, "use_cuda_tile", True)
        or getattr(norm, "return_hp_output", True)
    )


def _check_layout(block, sm_version: int, norm):
    """Raise with the reason unless the GPT-OSS block runs the no-all-reduce layout over the native TRTLLM-Gen W4A8
    MXFP4 x MXFP8 MoE and FlashInfer's CuTe norm; else (device, experts, backend)."""
    from tensorrt_llm._torch.moe.fused_moe.configurable_moe import ConfigurableMoE
    from tensorrt_llm._torch.moe.fused_moe.trtllm_gen.trtllm_w4a8_mxfp4_mxfp8 import (
        TrtllmTrtllmGenW4a8Mxfp4Mxfp8Impl,
    )

    if sm_version // 10 != 10:
        raise ValueError(f"SM {sm_version}: the TRTLLM-Gen MXFP4 MoE runs on the SM100 family only")
    if block.is_tp or block.enable_attn_dp:
        raise ValueError("not the forward_normal layout (TP > 1 or attention DP)")
    if not _plain_rmsnorm(norm) or norm.weight.dtype != torch.bfloat16:
        raise ValueError("the norm is not FlashInfer's plain BF16 fused add+RMSNorm")
    device = torch.device("cuda", torch.cuda.current_device())
    if not flashinfer_uses_cute_norm(device):
        raise ValueError(
            "FlashInfer would run its CUDA JIT norm, not the CuTe-DSL kernel these copy"
        )
    reason = flashinfer_check()
    if reason:
        raise ValueError(reason)
    experts = block.mlp.experts
    if not isinstance(experts, ConfigurableMoE):
        raise ValueError(f"experts are {type(experts).__name__}, not ConfigurableMoE")
    backend = experts.backend
    if type(backend) is not TrtllmTrtllmGenW4a8Mxfp4Mxfp8Impl:
        raise ValueError(
            f"MoE backend {type(backend).__name__}, not the native TRTLLM-Gen W4A8 MXFP4/MXFP8 leaf"
        )
    if (
        experts.comm is not None
        or experts._using_load_balancer()
        or experts.apply_router_weight_on_input
    ):
        raise ValueError("MoE has a comm strategy, EPLB or router weight on input")
    return device, experts, backend


def finalize_width(hidden: int) -> int:
    """The FC2 row width of the MXFP4 runner (output_hidden_size = roundUp(valid_hidden_size, 128))."""
    return (hidden + 127) // 128 * 128


def gate_finalize_norm(block, sm_version: int) -> bool:
    """Raise with the reason unless the fused finalize + add+RMSNorm may replace this block's MoE finalize and its
    next-layer norm; else compile the kernel (cached) and return True. All layers share one configuration."""
    device, experts, backend = _check_layout(block, sm_version, block.next_layer_layernorm)
    if experts.unfinalized_combine_fn is not None:
        raise ValueError("the MoE registers its own unfinalized combine")
    if experts.routing_method.experts_per_token != 4:
        raise ValueError(
            f"top_k {experts.routing_method.experts_per_token}, the kernel gathers 4 rows"
        )
    if experts.reduce_results:
        raise ValueError("the MoE reduces its results")
    hidden = backend.hidden_size
    t0 = time.monotonic()
    compiled_finalize_norm(
        hidden, finalize_width(hidden), 4, FINALIZE_USES_FMA, enable_pdl(device), sm_version
    )
    log_once(
        "finalize-enabled",
        f"{FINALIZE_NORM_ENV}=1 active: MoE finalize + residual add + RMSNorm in one kernel (H={hidden}, FC2 width "
        f"{finalize_width(hidden)}, top_k 4, SM{sm_version}, FlashInfer {flashinfer.__version__}, compiled in "
        f"{time.monotonic() - t0:.1f} s); finalizeKernel* should be absent from nsys",
    )
    return True


def gate_norm_mxfp8(block, sm_version: int) -> int:
    """Raise with the reason unless the fused add+RMSNorm+MXFP8 may replace this block's post-attention norm and its
    MoE input quantization; else compile the kernel (cached) and return the MoE input alignment it was built for."""
    device, experts, backend = _check_layout(block, sm_version, block.post_attention_layernorm)
    alignment = int(backend._get_quant_method().input_hidden_alignment)
    hidden = block.post_attention_layernorm.weight.shape[0]
    if alignment % SF_VEC_SIZE != 0:
        raise ValueError(f"MoE input alignment {alignment}")
    t0 = time.monotonic()
    compiled_norm_mxfp8(hidden, padded_width(hidden, alignment), enable_pdl(device), sm_version)
    log_once(
        "mxfp8-enabled",
        f"{NORM_MXFP8_ENV}=1 active: post-attention add+RMSNorm writes BF16 + the MXFP8 MoE input in one kernel "
        f"(H={hidden}, K={padded_width(hidden, alignment)}, SM{sm_version}, FlashInfer {flashinfer.__version__}, "
        f"compiled in {time.monotonic() - t0:.1f} s); quantize_with_block_size<2> should be absent from nsys",
    )
    return alignment


def gate_norm_mxfp8_dep3(block, sm_version: int) -> int:
    """TLLM_GPTOSS_DEP3_GLUE=mxfp8: ``gate_norm_mxfp8`` for attention DP + EP. Raise with the reason unless the MoE
    quantizes its own tokens before a one-sided post-quant dispatch, which is where the fused kernel's MXFP8 copy
    replaces ``quantize_input``'s ``mxfp8_quantize`` (``ExternalCommMoEScheduler._forward_chunk_impl`` step 5); else
    compile the kernel (cached) and return the MoE input alignment it was built for.

    EPLB is allowed: it maps expert ids to slots and never reads the activations."""
    from tensorrt_llm._torch.moe.fused_moe.communication.nvlink_one_sided import NVLinkOneSided
    from tensorrt_llm._torch.moe.fused_moe.configurable_moe import ConfigurableMoE
    from tensorrt_llm._torch.moe.fused_moe.trtllm_gen.trtllm_w4a8_mxfp4_mxfp8 import (
        TrtllmTrtllmGenW4a8Mxfp4Mxfp8Impl,
    )

    norm = block.post_attention_layernorm
    if sm_version // 10 != 10:
        raise ValueError(f"SM {sm_version}: the TRTLLM-Gen MXFP4 MoE runs on the SM100 family only")
    if block.is_tp or not block.enable_attn_dp:
        raise ValueError("not the attention-DP layout")
    if not _plain_rmsnorm(norm) or norm.weight.dtype != torch.bfloat16:
        raise ValueError("the norm is not FlashInfer's plain BF16 fused add+RMSNorm")
    device = torch.device("cuda", torch.cuda.current_device())
    if not flashinfer_uses_cute_norm(device):
        raise ValueError(
            "FlashInfer would run its CUDA JIT norm, not the CuTe-DSL kernel these copy"
        )
    reason = flashinfer_check()
    if reason:
        raise ValueError(reason)
    experts = block.mlp.experts
    if not isinstance(experts, ConfigurableMoE):
        raise ValueError(f"experts are {type(experts).__name__}, not ConfigurableMoE")
    backend = experts.backend
    if type(backend) is not TrtllmTrtllmGenW4a8Mxfp4Mxfp8Impl:
        raise ValueError(
            f"MoE backend {type(backend).__name__}, not the native TRTLLM-Gen W4A8 MXFP4/MXFP8 leaf"
        )
    comm = experts.comm
    if not isinstance(comm, NVLinkOneSided):
        raise ValueError(f"MoE comm {type(comm).__name__}, not the one-sided NVLink all-to-all")
    if not comm.supports_post_quant_dispatch() or comm.uses_internal_dispatch_quantization():
        raise ValueError("the all-to-all does not dispatch the MoE's own quantized input")
    if experts.apply_router_weight_on_input:
        raise ValueError("router weight on input scales the activations before the quantization")
    alignment = int(backend._get_quant_method().input_hidden_alignment)
    hidden = norm.weight.shape[0]
    if alignment % SF_VEC_SIZE != 0:
        raise ValueError(f"MoE input alignment {alignment}")
    t0 = time.monotonic()
    compiled_norm_mxfp8(hidden, padded_width(hidden, alignment), enable_pdl(device), sm_version)
    log_once(
        "mxfp8-dep3-enabled",
        f"TLLM_GPTOSS_DEP3_GLUE=mxfp8 active: attention-DP post-attention add+RMSNorm writes BF16 + the MXFP8 MoE "
        f"input in one kernel ahead of the {type(comm).__name__} dispatch (H={hidden}, "
        f"K={padded_width(hidden, alignment)}, SM{sm_version}, FlashInfer {flashinfer.__version__}, compiled in "
        f"{time.monotonic() - t0:.1f} s); quantize_with_block_size<2> should be absent from nsys",
    )
    return alignment


def moe_layer_finalize_add_rmsnorm(
    moe,
    x: torch.Tensor,
    x_sf: Optional[torch.Tensor],
    router_logits: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> List[torch.Tensor]:
    """The MoE layer without its finalize kernel, then the fused finalize + add+RMSNorm: [normed, new residual].

    ``x`` is the BF16 MoE input, or with ``x_sf`` the MXFP8 one (TLLM_GPTOSS_FUSE_NORM_MXFP8). A forward the MoE
    would split into chunks keeps today's kernels (finalized MoE, then FlashInfer's fused add+RMSNorm)."""
    from tensorrt_llm._torch.utils import MxFp8QuantizedTensor

    hidden_states = x if x_sf is None else MxFp8QuantizedTensor(x, x_sf, is_sf_swizzled=False)
    if moe.calculate_num_chunks([residual.shape[0]]) > 1:
        log_once(
            "finalize-chunked",
            f"{FINALIZE_NORM_ENV}: {residual.shape[0]} tokens exceed the MoE chunk size; this forward keeps the "
            "finalize kernel and the separate norm",
        )
        out = moe.forward_impl(
            hidden_states, router_logits, do_finalize=True, output_dtype=torch.bfloat16
        )
        out = out.view(residual.shape)
        res = residual.clone()
        torch.ops.trtllm.flashinfer_fused_add_rmsnorm(out, res, weight, eps)
        return [out, res]
    gemm2_output, expert_weights, expanded_idx_to_permuted_idx = moe.forward_impl(
        hidden_states, router_logits, do_finalize=False, output_dtype=torch.bfloat16
    )
    return moe_finalize_add_rmsnorm(
        gemm2_output, expert_weights, expanded_idx_to_permuted_idx, residual, weight, eps
    )


@torch.library.custom_op("trtllm::gptoss_moe_finalize_add_rmsnorm", mutates_args=())
def gptoss_moe_finalize_add_rmsnorm(
    layer_idx: str,
    x: torch.Tensor,
    x_sf: Optional[torch.Tensor],
    router_logits: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> List[torch.Tensor]:
    """``moe_layer_finalize_add_rmsnorm`` for the MoE layer registered as ``layer_idx`` (the torch.compile path,
    like ``trtllm::moe_custom_op``): one opaque op with [M, H] outputs, so the unfinalized rows never enter the
    traced graph."""
    from tensorrt_llm._torch.moe.fused_moe.interface import extract_extra_attrs

    moe = extract_extra_attrs(layer_idx)
    return moe_layer_finalize_add_rmsnorm(moe, x, x_sf, router_logits, residual, weight, eps)


@gptoss_moe_finalize_add_rmsnorm.register_fake
def _(layer_idx, x, x_sf, router_logits, residual, weight, eps):
    return [residual.new_empty(residual.shape), residual.new_empty(residual.shape)]
