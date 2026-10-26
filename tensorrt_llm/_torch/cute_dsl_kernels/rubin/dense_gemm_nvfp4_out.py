# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 AND BSD-3-Clause
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

# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# The GEMM mainloop is derived from blackwell/dense_gemm_persistent.py, which is copied and
# modified from the CUTLASS example (public reference pinned to v4.2.0):
# https://github.com/NVIDIA/cutlass/blob/v4.2.0/examples/python/CuTeDSL/blackwell/dense_gemm_persistent.py
"""Persistent batched BF16 GEMM for Rubin (SM107) whose epilogue writes NVFP4 (DS-R1 MLA decode, opt-in).

Gate: ``TRTLLM_MLA_NVFP4_EPILOGUE`` (modules/mla.py; default 0 = off, nothing here runs).

The absorbed-MLA generation BMMs otherwise write BF16 and a second kernel re-reads and quantizes it:

* bmm_q (``q_nope^T x W_UK``) writes ``fused_q[..., :512]`` and the fused FP4 MLA RoPE kernel's Q-prefix programs
  quantize it (``fp4_mla_kernels._fp4_mla_quantize_q_groups_words``): packed Q bytes 0-255 of each 320-byte row
  (row = token * 128 + head) and scale columns 0-31 of the H128-swizzled Q-scale plane.
* bmm_o (``attn_latent^T x W_UV``) writes the [tokens, 16384] o_proj input and ``fp4_quantize``
  (C++ op or the bit-identical ``nvfp4_quantize_triton``, swizzled scales) quantizes it for the NVFP4 o_proj.

This kernel computes the same GEMM (the Blackwell CuTe DSL persistent BF16 mainloop of
``blackwell/dense_gemm_persistent.py`` that the Rubin BF16 BMM also runs: TMA, tcgen05 MMA, FP32 accumulator in
TMEM) and quantizes in the epilogue. It first rounds the accumulator to BF16 exactly as the BF16 kernels do (so the
quantizer sees the bytes the BF16 GEMM would have written), then applies the consumer's own quantizer arithmetic
instruction for instruction (inline PTX; no reassociation or contraction is possible):

* ``MODE_Q`` (``_fp4_mla_quantize_q_groups_words``): v = mul.rn(x, q_global_scale); s = max(div.full(amax(|v|), 6),
  2^-9); s8 = cvt.rn.satfinite.e4m3(s); y = mul.rn(v, rcp.approx.ftz(f32(s8))); e2m1 = cvt.rn.satfinite.e2m1x2(y).
* ``MODE_O`` (``nvfp4_quantize_triton`` / ``cvt_warp_fp16_to_fp4``): sf = mul.rn(gs, mul.rn(amax(|x|),
  rcp.approx.ftz(6))); s8 = cvt.rn.satfinite.e4m3(sf); out = rcp.approx.ftz(mul.rn(f32(s8), rcp.approx.ftz(gs)))
  (0 where amax == 0); y = mul.rn(x, out); e2m1 = cvt.rn.satfinite.e2m1x2(y).

Each epilogue thread owns one accumulator row and EPI_N consecutive columns (tcgen05 32x32b loads), i.e. whole
16-element scale groups, so no cross-thread reduction is needed. Packed E2M1 words go straight from registers to
global memory (two 16-byte stores per 64 columns) at ``m * out_stride_m + l * out_stride_l + n / 2`` bytes, and each
thread's four E4M3 scale bytes as one 32-bit store into the 128x4-swizzled scale plane at row
``m * sf_row_m + l * sf_row_l``, column ``l * sf_col_l + n / 16`` (``get_sf_out_offset_128x4``). That covers both
consumers: the Q rows (token * 128 + head, 320 B, H128 scale rows) and o_proj's monolithic [tokens, 128 * 128]
activation (scale columns head * 8 + n / 16). Rows at or beyond ``fp4_rows`` are not stored; scale rows below
``sf_rows`` are (o_proj's padding rows to the next multiple of 128 get the zero scale fp4_quantize writes, since their
A rows are TMA zero-fill).

Lineage: the idea of quantizing the FP32 accumulator per thread over whole SF blocks and the monolithic o_proj scale
layout follow NVIDIA/TensorRT-LLM PR #15370 (closed unmerged); the bit-exact quantizers reproduce the consumers'
own quantizer arithmetic. Public cutlass-dsl (4.8) APIs only. The BF16 GEMM output equals the cuBLAS
``bmm_out`` output only when both accumulate the K loop in the same order (no split-K); the unit test
(tests/unittest/_torch/modules/test_mla_nvfp4_out_epilogue.py) checks the bytes end to end against
``bmm_out`` + the shipped quantizers.
"""

from typing import Tuple, Type

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
from cutlass._mlir.dialects import llvm
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait

from ..blackwell.custom_pipeline import PipelineTmaUmma, PipelineUmmaAsync
from ..blackwell.dense_gemm_persistent import (
    PersistentDenseGemmKernel as BlackwellPersistentDenseGemmKernel,
)
from ..blackwell.utils import (
    TRTLLM_ENABLE_PDL,
    griddepcontrol_launch_dependents,
    griddepcontrol_wait,
)

MODE_Q = 0  # FP4 MLA Q prefix: fp4_mla_kernels._fp4_mla_quantize_q_groups_words
MODE_O = 1  # o_proj input: fp4_utils.nvfp4_quantize_triton (= fp4_quantize, swizzled)

SF_VEC = 16


# ----------------------------------------------------------------------------------------------- exact PTX helpers
@dsl_user_op
def _mul_rn(a, b, *, loc=None, ip=None):
    return cutlass.Float32(
        llvm.inline_asm(
            T.f32(),
            [
                cutlass.Float32(a).ir_value(loc=loc, ip=ip),
                cutlass.Float32(b).ir_value(loc=loc, ip=ip),
            ],
            "mul.rn.f32 $0, $1, $2;",
            "=f,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _div_full(a, b, *, loc=None, ip=None):
    # Triton's fp32 '/' (arith.divf) lowers to div.full.f32
    return cutlass.Float32(
        llvm.inline_asm(
            T.f32(),
            [
                cutlass.Float32(a).ir_value(loc=loc, ip=ip),
                cutlass.Float32(b).ir_value(loc=loc, ip=ip),
            ],
            "div.full.f32 $0, $1, $2;",
            "=f,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _max_f32(a, b, *, loc=None, ip=None):
    return cutlass.Float32(
        llvm.inline_asm(
            T.f32(),
            [
                cutlass.Float32(a).ir_value(loc=loc, ip=ip),
                cutlass.Float32(b).ir_value(loc=loc, ip=ip),
            ],
            "max.f32 $0, $1, $2;",
            "=f,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _absmax_f32(m, a, *, loc=None, ip=None):
    """max(m, |a|)."""
    return cutlass.Float32(
        llvm.inline_asm(
            T.f32(),
            [
                cutlass.Float32(m).ir_value(loc=loc, ip=ip),
                cutlass.Float32(a).ir_value(loc=loc, ip=ip),
            ],
            "{\n\t.reg .f32 t;\n\tabs.f32 t, $2;\n\tmax.f32 $0, $1, t;\n\t}",
            "=f,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _rcp_approx_ftz(a, *, loc=None, ip=None):
    return cutlass.Float32(
        llvm.inline_asm(
            T.f32(),
            [cutlass.Float32(a).ir_value(loc=loc, ip=ip)],
            "rcp.approx.ftz.f32 $0, $1;",
            "=f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _e4m3_bits(a, *, loc=None, ip=None):
    """cvt.rn.satfinite.e4m3 of one float; the byte in bits 0-7 (as Triton's fp32 -> float8e4nv)."""
    return cutlass.Int32(
        llvm.inline_asm(
            T.i32(),
            [cutlass.Float32(a).ir_value(loc=loc, ip=ip)],
            "{\n\t.reg .b16 t;\n\tcvt.rn.satfinite.e4m3x2.f32 t, $1, $1;\n\t"
            "cvt.u32.u16 $0, t;\n\tand.b32 $0, $0, 255;\n\t}",
            "=r,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _e4m3_bits_to_f32(bits, *, loc=None, ip=None):
    """Exact value of an E4M3 byte (bits 0-7), through the hardware f16x2 decode."""
    return cutlass.Float32(
        llvm.inline_asm(
            T.f32(),
            [cutlass.Int32(bits).ir_value(loc=loc, ip=ip)],
            "{\n\t.reg .b16 t, lo, hi;\n\t.reg .b32 h2;\n\tcvt.u16.u32 t, $1;\n\t"
            "cvt.rn.f16x2.e4m3x2 h2, t;\n\tmov.b32 {lo, hi}, h2;\n\tcvt.f32.f16 $0, lo;\n\t}",
            "=f,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _select_nonzero(val, key, *, loc=None, ip=None):
    """key != 0 (unordered, as Triton's '!=') ? val : 0."""
    return cutlass.Float32(
        llvm.inline_asm(
            T.f32(),
            [
                cutlass.Float32(val).ir_value(loc=loc, ip=ip),
                cutlass.Float32(key).ir_value(loc=loc, ip=ip),
            ],
            "{\n\t.reg .pred p;\n\tsetp.neu.f32 p, $2, 0f00000000;\n\tselp.f32 $0, $1, 0f00000000, p;\n\t}",
            "=f,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _pack_e2m1x8(p0, p1, p2, p3, p4, p5, p6, p7, *, loc=None, ip=None):
    """Eight floats -> one natural-order E2M1 word (byte k = e2m1(p[2k+1]) << 4 | e2m1(p[2k]))."""
    return cutlass.Int32(
        llvm.inline_asm(
            T.i32(),
            [cutlass.Float32(p).ir_value(loc=loc, ip=ip) for p in (p0, p1, p2, p3, p4, p5, p6, p7)],
            "{\n\t.reg .b8 b0, b1, b2, b3;\n\t"
            "cvt.rn.satfinite.e2m1x2.f32 b0, $2, $1;\n\t"
            "cvt.rn.satfinite.e2m1x2.f32 b1, $4, $3;\n\t"
            "cvt.rn.satfinite.e2m1x2.f32 b2, $6, $5;\n\t"
            "cvt.rn.satfinite.e2m1x2.f32 b3, $8, $7;\n\t"
            "mov.b32 $0, {b0, b1, b2, b3};\n\t}",
            "=r,f,f,f,f,f,f,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def _st_global_v4(ptr, w0, w1, w2, w3, *, loc=None, ip=None):
    llvm.inline_asm(
        None,
        [
            ptr.llvm_ptr,
            cutlass.Int32(w0).ir_value(loc=loc, ip=ip),
            cutlass.Int32(w1).ir_value(loc=loc, ip=ip),
            cutlass.Int32(w2).ir_value(loc=loc, ip=ip),
            cutlass.Int32(w3).ir_value(loc=loc, ip=ip),
        ],
        "st.global.v4.b32 [$0], {$1, $2, $3, $4};",
        "l,r,r,r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _st_global_b32(ptr, w, *, loc=None, ip=None):
    llvm.inline_asm(
        None,
        [ptr.llvm_ptr, cutlass.Int32(w).ir_value(loc=loc, ip=ip)],
        "st.global.b32 [$0], $1;",
        "l,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _st_global_b16(ptr, w, *, loc=None, ip=None):
    llvm.inline_asm(
        None,
        [ptr.llvm_ptr, cutlass.Int32(w).ir_value(loc=loc, ip=ip)],
        "{\n\t.reg .b16 t;\n\tcvt.u16.u32 t, $1;\n\tst.global.b16 [$0], t;\n\t}",
        "l,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


# ----------------------------------------------------------------------------------------------- the kernel
class Sm107PersistentDenseGemmNvfp4OutKernel(BlackwellPersistentDenseGemmKernel):
    """Persistent batched BF16 GEMM (C[m, n, l] = sum_k A[m, k, l] B[n, k, l]) with an NVFP4 epilogue, SM107.

    The mainloop (TMA warp, MMA warp, two TMEM accumulator stages, static persistent tile scheduler) is the
    ``blackwell/dense_gemm_persistent.py`` one, which the Rubin BF16 BMM (``rubin/dense_bf16_gemm_persistent.py``)
    also runs. Only the epilogue differs (no C shared memory, no TMA store).

    Constraints: CTA tile M 128 (1-CTA MMA M 128 or 2-CTA MMA M 256), N a multiple of the MMA tile N and of EPI_N,
    EPI_N in (32, 64).
    """

    def __init__(
        self,
        mode: int,
        use_2cta_instrs: bool,
        mma_tiler_mn: Tuple[int, int],
        cluster_shape_mn: Tuple[int, int],
        epi_n: int = 64,
        max_num_ab_stage: int = 0,
        epi_groups: int = 1,
    ):
        super().__init__(
            acc_dtype=cutlass.Float32,
            use_2cta_instrs=use_2cta_instrs,
            mma_tiler_mn=mma_tiler_mn,
            cluster_shape_mn=cluster_shape_mn,
            use_tma_store=False,
        )
        # Read only by _compute_num_tmem_alloc_cols, as in rubin/dense_bf16_gemm_persistent.py.
        self.arch = "sm_107"
        assert mode in (MODE_Q, MODE_O)
        assert epi_n in (32, 64)
        self.mode = mode
        self.epi_n = epi_n
        self._max_num_ab_stage = max_num_ab_stage
        # epi_groups 2: two 4-warp epilogue groups take alternate EPI_N subtiles of each accumulator tile (two warps
        # per SM sub-partition hide the quantizer's dependent-instruction latency); warps 8 / 9 are MMA / TMA.
        assert epi_groups in (1, 2)
        self.epi_groups = epi_groups
        if epi_groups == 2:
            self.epilogue_warp_id = (0, 1, 2, 3, 4, 5, 6, 7)
            self.mma_warp_id = 8
            self.tma_warp_id = 9
            self.threads_per_cta = 32 * len(
                (self.mma_warp_id, self.tma_warp_id, *self.epilogue_warp_id)
            )

    def __repr__(self) -> str:
        return (
            f"nvfp4out_mode{self.mode}_2cta{int(self.use_2cta_instrs)}"
            f"_tile{self.mma_tiler_mn[0]}x{self.mma_tiler_mn[1]}"
            f"_cluster{self.cluster_shape_mn[0]}x{self.cluster_shape_mn[1]}"
            f"_epi{self.epi_n}_stage{self._max_num_ab_stage}_groups{self.epi_groups}"
        )

    __str__ = __repr__

    @staticmethod
    def can_implement(
        mode: int,
        use_2cta_instrs: bool,
        mma_tiler_mn: Tuple[int, int],
        cluster_shape_mn: Tuple[int, int],
        m: int,
        n: int,
        k: int,
        batch: int,
        epi_n: int = 64,
    ) -> bool:
        if mode not in (MODE_Q, MODE_O) or epi_n not in (32, 64):
            return False
        if not BlackwellPersistentDenseGemmKernel.is_valid_mma_tiler_and_cluster_shape(
            use_2cta_instrs, mma_tiler_mn, cluster_shape_mn
        ):
            return False
        cta_tile_m = mma_tiler_mn[0] // (2 if use_2cta_instrs else 1)
        if cta_tile_m != 128:
            return False  # one TMEM lane per accumulator row (32x32b loads)
        if n % mma_tiler_mn[1] != 0 or mma_tiler_mn[1] % epi_n != 0:
            return False
        if k % 8 != 0 or m <= 0 or batch <= 0:
            return False
        # SM107: an M cluster wider than the M tiles leaves phantom CTAs (the Rubin BF16 kernel's guard)
        if (m + cta_tile_m - 1) // cta_tile_m < cluster_shape_mn[0]:
            return False
        return True

    def _setup_attributes(self):
        super()._setup_attributes()  # use_tma_store=False: no C shared memory in the stage count
        if self._max_num_ab_stage > 0 and self.num_ab_stage > self._max_num_ab_stage:
            self.num_ab_stage = self._max_num_ab_stage
            tiled_mma = self._create_tiled_mma()
            self.a_smem_layout_staged = utils.sm100.make_smem_layout_a(
                tiled_mma, self.mma_tiler, self.a_dtype, self.num_ab_stage
            )
            self.b_smem_layout_staged = utils.sm100.make_smem_layout_b(
                tiled_mma, self.mma_tiler, self.b_dtype, self.num_ab_stage
            )
        assert self.cta_tile_shape_mnk[0] == 128
        self.epi_tile = (self.cta_tile_shape_mnk[0], self.epi_n)

    @cute.jit
    def __call__(
        self,
        a: cute.Tensor,
        b: cute.Tensor,
        out_ptr: cute.Pointer,
        sf_ptr: cute.Pointer,
        scale: cute.Tensor,
        out_stride_m: cutlass.Int32,
        out_stride_l: cutlass.Int32,
        sf_row_m: cutlass.Int32,
        sf_row_l: cutlass.Int32,
        sf_col_l: cutlass.Int32,
        sf_kt512: cutlass.Int32,
        fp4_rows: cutlass.Int32,
        sf_rows: cutlass.Int32,
        max_active_clusters: cutlass.Constexpr,
        stream: cuda.CUstream,
    ):
        self.a_dtype: Type[cutlass.Numeric] = a.element_type
        self.b_dtype: Type[cutlass.Numeric] = b.element_type
        # the quantizer reads the BF16 the BF16 kernel would have stored; the helpers size for that C
        self.c_dtype: Type[cutlass.Numeric] = cutlass.BFloat16
        self.a_major_mode = utils.LayoutEnum.from_tensor(a).mma_major_mode()
        self.b_major_mode = utils.LayoutEnum.from_tensor(b).mma_major_mode()
        self.c_layout = utils.LayoutEnum.ROW_MAJOR

        if cutlass.const_expr(self.a_dtype != self.b_dtype):
            raise TypeError(f"Type must match: {self.a_dtype} != {self.b_dtype}")

        tiled_mma = self._create_tiled_mma()
        self._setup_attributes()

        atom_thr_size = cute.size(tiled_mma.thr_id.shape)

        a_op = utils.sm100.cluster_shape_to_tma_atom_A(self.cluster_shape_mn, tiled_mma.thr_id)
        a_smem_layout = cute.slice_(self.a_smem_layout_staged, (None, None, None, 0))
        tma_atom_a, tma_tensor_a = cute.nvgpu.make_tiled_tma_atom_A(
            a_op,
            a,
            a_smem_layout,
            self.mma_tiler,
            tiled_mma,
            self.cluster_layout_vmnk.shape,
        )
        b_op = utils.sm100.cluster_shape_to_tma_atom_B(self.cluster_shape_mn, tiled_mma.thr_id)
        b_smem_layout = cute.slice_(self.b_smem_layout_staged, (None, None, None, 0))
        tma_atom_b, tma_tensor_b = cute.nvgpu.make_tiled_tma_atom_B(
            b_op,
            b,
            b_smem_layout,
            self.mma_tiler,
            tiled_mma,
            self.cluster_layout_vmnk.shape,
        )
        a_copy_size = cute.size_in_bytes(self.a_dtype, a_smem_layout)
        b_copy_size = cute.size_in_bytes(self.b_dtype, b_smem_layout)
        self.num_tma_load_bytes = (a_copy_size + b_copy_size) * atom_thr_size

        # the (M, N, L) output index space: only its shape is used (tile counts)
        c_coord = cute.make_identity_tensor((a.shape[0], b.shape[0], a.shape[2]))
        self.tile_sched_params, grid = self._compute_grid(
            c_coord,
            self.cta_tile_shape_mnk,
            self.cluster_shape_mn,
            self.swizzle_size,
            self.raster_along,
            max_active_clusters,
        )

        self.kernel(
            tiled_mma,
            tma_atom_a,
            tma_tensor_a,
            tma_atom_b,
            tma_tensor_b,
            out_ptr,
            sf_ptr,
            scale,
            out_stride_m,
            out_stride_l,
            sf_row_m,
            sf_row_l,
            sf_col_l,
            sf_kt512,
            fp4_rows,
            sf_rows,
            self.cluster_layout_vmnk,
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            self.epi_tile,
            self.tile_sched_params,
        ).launch(
            grid=grid,
            block=[self.threads_per_cta, 1, 1],
            cluster=(*self.cluster_shape_mn, 1),
            stream=stream,
            use_pdl=TRTLLM_ENABLE_PDL,
        )
        return

    @cute.kernel
    def kernel(
        self,
        tiled_mma: cute.TiledMma,
        tma_atom_a: cute.CopyAtom,
        mA_mkl: cute.Tensor,
        tma_atom_b: cute.CopyAtom,
        mB_nkl: cute.Tensor,
        out_ptr: cute.Pointer,
        sf_ptr: cute.Pointer,
        scale: cute.Tensor,
        out_stride_m: cutlass.Int32,
        out_stride_l: cutlass.Int32,
        sf_row_m: cutlass.Int32,
        sf_row_l: cutlass.Int32,
        sf_col_l: cutlass.Int32,
        sf_kt512: cutlass.Int32,
        fp4_rows: cutlass.Int32,
        sf_rows: cutlass.Int32,
        cluster_layout_vmnk: cute.Layout,
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        epi_tile: cute.Tile,
        tile_sched_params: utils.PersistentTileSchedulerParams,
    ):
        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)

        if warp_idx == self.tma_warp_id:
            cpasync.prefetch_descriptor(tma_atom_a)
            cpasync.prefetch_descriptor(tma_atom_b)

        use_2cta_instrs = cute.size(tiled_mma.thr_id.shape) == 2

        bidx, bidy, bidz = cute.arch.block_idx()
        mma_tile_coord_v = bidx % cute.size(tiled_mma.thr_id.shape)
        is_leader_cta = mma_tile_coord_v == 0
        cta_rank_in_cluster = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        block_in_cluster_coord_vmnk = cluster_layout_vmnk.get_flat_coord(cta_rank_in_cluster)
        tidx, _, _ = cute.arch.thread_idx()

        @cute.struct
        class SharedStorage:
            ab_full_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_ab_stage * 2]
            acc_full_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_acc_stage * 2]
            tmem_dealloc_mbar: cutlass.Int64
            tmem_holding_buf: cutlass.Int32

        smem = utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)

        ab_pipeline_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        num_tma_producer = self.num_mcast_ctas_a + self.num_mcast_ctas_b - 1
        ab_pipeline_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, num_tma_producer
        )
        ab_producer, ab_consumer = PipelineTmaUmma.create(
            barrier_storage=storage.ab_full_mbar_ptr.data_ptr(),
            num_stages=self.num_ab_stage,
            producer_group=ab_pipeline_producer_group,
            consumer_group=ab_pipeline_consumer_group,
            tx_count=self.num_tma_load_bytes,
            cta_layout_vmnk=cluster_layout_vmnk,
        ).make_participants()

        acc_pipeline_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        num_acc_consumer_threads = len(self.epilogue_warp_id) * (2 if use_2cta_instrs else 1)
        acc_pipeline_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, num_acc_consumer_threads
        )
        acc_pipeline = PipelineUmmaAsync.create(
            barrier_storage=storage.acc_full_mbar_ptr.data_ptr(),
            num_stages=self.num_acc_stage,
            producer_group=acc_pipeline_producer_group,
            consumer_group=acc_pipeline_consumer_group,
            cta_layout_vmnk=cluster_layout_vmnk,
        )

        tmem_alloc_barrier = pipeline.NamedBarrier(
            barrier_id=self.tmem_alloc_sync_bar_id,
            num_threads=32 * len((self.mma_warp_id, *self.epilogue_warp_id)),
        )
        tmem = utils.TmemAllocator(
            storage.tmem_holding_buf.ptr,
            barrier_for_retrieve=tmem_alloc_barrier,
            allocator_warp_id=self.epilogue_warp_id[0],
            is_two_cta=use_2cta_instrs,
            two_cta_tmem_dealloc_mbar_ptr=storage.tmem_dealloc_mbar.ptr,
        )

        pipeline_init_arrive(cluster_shape_mn=self.cluster_shape_mn, is_relaxed=True)

        sA = smem.allocate_tensor(
            element_type=self.a_dtype,
            layout=a_smem_layout_staged.outer,
            byte_alignment=128,
            swizzle=a_smem_layout_staged.inner,
        )
        sB = smem.allocate_tensor(
            element_type=self.b_dtype,
            layout=b_smem_layout_staged.outer,
            byte_alignment=128,
            swizzle=b_smem_layout_staged.inner,
        )

        a_full_mcast_mask = None
        b_full_mcast_mask = None
        if cutlass.const_expr(self.is_a_mcast or self.is_b_mcast or use_2cta_instrs):
            a_full_mcast_mask = cpasync.create_tma_multicast_mask(
                cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=2
            )
            b_full_mcast_mask = cpasync.create_tma_multicast_mask(
                cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=1
            )

        # coordinates (m, n, l) of every output element, tiled and partitioned exactly as the accumulator
        mC_mnl = cute.make_identity_tensor((mA_mkl.shape[0], mB_nkl.shape[0], mA_mkl.shape[2]))
        gA_mkl = cute.local_tile(
            mA_mkl, cute.slice_(self.mma_tiler, (None, 0, None)), (None, None, None)
        )
        gB_nkl = cute.local_tile(
            mB_nkl, cute.slice_(self.mma_tiler, (0, None, None)), (None, None, None)
        )
        gC_mnl = cute.local_tile(
            mC_mnl, cute.slice_(self.mma_tiler, (None, None, 0)), (None, None, None)
        )
        k_tile_cnt = cute.size(gA_mkl, mode=[3])

        thr_mma = tiled_mma.get_slice(mma_tile_coord_v)
        tCgA = thr_mma.partition_A(gA_mkl)
        tCgB = thr_mma.partition_B(gB_nkl)
        tCgC = thr_mma.partition_C(
            gC_mnl
        )  # coordinates (m, n, l) of this CTA's accumulator elements

        a_cta_layout = cute.make_layout(cute.slice_(cluster_layout_vmnk, (0, 0, None, 0)).shape)
        tAsA, tAgA = cpasync.tma_partition(
            tma_atom_a,
            block_in_cluster_coord_vmnk[2],
            a_cta_layout,
            cute.group_modes(sA, 0, 3),
            cute.group_modes(tCgA, 0, 3),
        )
        b_cta_layout = cute.make_layout(cute.slice_(cluster_layout_vmnk, (0, None, 0, 0)).shape)
        tBsB, tBgB = cpasync.tma_partition(
            tma_atom_b,
            block_in_cluster_coord_vmnk[1],
            b_cta_layout,
            cute.group_modes(sB, 0, 3),
            cute.group_modes(tCgB, 0, 3),
        )

        tCrA = tiled_mma.make_fragment_A(sA)
        tCrB = tiled_mma.make_fragment_B(sB)
        acc_shape = tiled_mma.partition_shape_C(self.mma_tiler[:2])
        tCtAcc_fake = tiled_mma.make_fragment_C(cute.append(acc_shape, self.num_acc_stage))

        pipeline_init_wait(cluster_shape_mn=self.cluster_shape_mn)

        tile_sched = utils.StaticPersistentTileScheduler.create(
            tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
        )
        work_tile = tile_sched.initial_work_tile_info()

        # PDL: inputs are read and outputs written only after the previous grid completes
        griddepcontrol_wait()

        # ---------------- TMA load warp ----------------
        if warp_idx == self.tma_warp_id:
            while work_tile.is_valid_tile:
                cur_tile_coord = work_tile.tile_idx
                mma_tile_coord_mnl = (
                    cur_tile_coord[0] // cute.size(tiled_mma.thr_id.shape),
                    cur_tile_coord[1],
                    cur_tile_coord[2],
                )
                tAgA_slice = tAgA[(None, mma_tile_coord_mnl[0], None, mma_tile_coord_mnl[2])]
                tBgB_slice = tBgB[(None, mma_tile_coord_mnl[1], None, mma_tile_coord_mnl[2])]

                ab_producer.reset()
                peek_ab_empty_status = ab_producer.try_acquire()

                for k_tile in cutlass.range(0, k_tile_cnt, 1, unroll=1):
                    handle = ab_producer.acquire_and_advance(peek_ab_empty_status)
                    cute.copy(
                        tma_atom_a,
                        tAgA_slice[(None, handle.count)],
                        tAsA[(None, handle.index)],
                        tma_bar_ptr=handle.barrier,
                        mcast_mask=a_full_mcast_mask,
                    )
                    cute.copy(
                        tma_atom_b,
                        tBgB_slice[(None, handle.count)],
                        tBsB[(None, handle.index)],
                        tma_bar_ptr=handle.barrier,
                        mcast_mask=b_full_mcast_mask,
                    )
                    peek_ab_empty_status = cutlass.Boolean(1)
                    if handle.count + 1 < k_tile_cnt:
                        peek_ab_empty_status = ab_producer.try_acquire()

                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()

            ab_producer.tail()

        # ---------------- MMA warp ----------------
        if warp_idx == self.mma_warp_id:
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            tCtAcc_base = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)

            acc_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_acc_stage
            )

            while work_tile.is_valid_tile:
                tCtAcc = tCtAcc_base[(None, None, None, acc_producer_state.index)]

                ab_consumer.reset()
                peek_ab_full_status = cutlass.Boolean(1)
                if is_leader_cta:
                    peek_ab_full_status = ab_consumer.try_wait()

                if is_leader_cta:
                    acc_pipeline.producer_acquire(acc_producer_state)

                tiled_mma.set(tcgen05.Field.ACCUMULATE, False)

                for k_tile in cutlass.range(0, k_tile_cnt, 1, unroll=1):
                    if is_leader_cta:
                        handle = ab_consumer.wait_and_advance(peek_ab_full_status)
                        num_kblocks = cute.size(tCrA, mode=[2])
                        for kblock_idx in cutlass.range(num_kblocks, unroll_full=True):
                            kblock_crd = (None, None, kblock_idx, handle.index)
                            cute.gemm(tiled_mma, tCtAcc, tCrA[kblock_crd], tCrB[kblock_crd], tCtAcc)
                            tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                        handle.release()
                        peek_ab_full_status = cutlass.Boolean(1)
                        if handle.count + 1 < k_tile_cnt:
                            peek_ab_full_status = ab_consumer.try_wait()

                if is_leader_cta:
                    acc_pipeline.producer_commit(acc_producer_state)
                acc_producer_state.advance()

                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()

            acc_pipeline.producer_tail(acc_producer_state)

        # ---------------- epilogue warps: TMEM -> registers -> NVFP4 -> global ----------------
        if warp_idx < self.mma_warp_id:
            tmem.allocate(self.num_tmem_alloc_cols)
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            tCtAcc_base = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)

            acc_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_acc_stage
            )

            copy_atom_t2r = utils.sm100.get_tmem_load_op(
                self.cta_tile_shape_mnk,
                self.c_layout,
                self.c_dtype,
                self.acc_dtype,
                epi_tile,
                use_2cta_instrs,
            )
            tAcc_epi = cute.flat_divide(tCtAcc_base[((None, None), 0, 0, None)], epi_tile)
            tiled_copy_t2r = tcgen05.make_tmem_copy(copy_atom_t2r, tAcc_epi[(None, None, 0, 0, 0)])
            # one 4-warp group covers the 128 TMEM lanes; a second group reuses the same lane partition
            epi_tidx = tidx % 128
            epi_group = tidx // 128
            thr_copy_t2r = tiled_copy_t2r.get_slice(epi_tidx)
            tTR_tAcc_base = thr_copy_t2r.partition_S(tAcc_epi)

            gC_mnl_epi = cute.flat_divide(tCgC[((None, None), 0, 0, None, None, None)], epi_tile)
            tTR_cC = thr_copy_t2r.partition_D(gC_mnl_epi)
            tTR_rAcc = cute.make_rmem_tensor(
                tTR_cC[(None, None, None, 0, 0, 0, 0, 0)].shape, self.acc_dtype
            )
            tTR_rF = cute.make_rmem_tensor(tTR_rAcc.shape, cutlass.Float32)

            g = scale[0]
            rcp6 = _rcp_approx_ftz(cutlass.Float32(6.0))
            rcp_g = _rcp_approx_ftz(g)

            epi_n = self.epi_n
            groups = epi_n // SF_VEC

            while work_tile.is_valid_tile:
                cur_tile_coord = work_tile.tile_idx
                mma_tile_coord_mnl = (
                    cur_tile_coord[0] // cute.size(tiled_mma.thr_id.shape),
                    cur_tile_coord[1],
                    cur_tile_coord[2],
                )
                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()

                acc_stage_index = acc_consumer_state.index
                tTR_tAcc = tTR_tAcc_base[(None, None, None, None, None, acc_stage_index)]
                tTR_cC_tile = tTR_cC[(None, None, None, None, None, *mma_tile_coord_mnl)]

                acc_pipeline.consumer_wait(acc_consumer_state)

                tTR_tAcc = cute.group_modes(tTR_tAcc, 3, cute.rank(tTR_tAcc))
                tTR_cC_tile = cute.group_modes(tTR_cC_tile, 3, cute.rank(tTR_cC_tile))
                subtile_cnt = cute.size(tTR_tAcc.shape, mode=[3])

                for subtile_idx in cutlass.range(epi_group, subtile_cnt, self.epi_groups):
                    tTR_tAcc_mn = tTR_tAcc[(None, None, None, subtile_idx)]
                    cute.copy(tiled_copy_t2r, tTR_tAcc_mn, tTR_rAcc)
                    # the BF16 the BF16 kernel would store (round to nearest even), as FP32
                    tTR_rF.store(tTR_rAcc.load().to(cutlass.BFloat16).to(cutlass.Float32))

                    coord = tTR_cC_tile[(None, None, None, subtile_idx)][0]
                    row_m = coord[0]
                    col_n = coord[1]
                    head_l = coord[2]

                    if row_m < fp4_rows:
                        # rows of real tokens: quantize, store the packed words and the scale bytes
                        words = []
                        sf_word = cutlass.Int32(0)
                        for grp in cutlass.range_constexpr(groups):
                            vals = []
                            for e in cutlass.range_constexpr(SF_VEC):
                                x = tTR_rF[grp * SF_VEC + e]
                                if cutlass.const_expr(self.mode == MODE_Q):
                                    x = _mul_rn(x, g)
                                vals.append(x)
                            amax = cutlass.Float32(0.0)
                            for e in cutlass.range_constexpr(SF_VEC):
                                amax = _absmax_f32(amax, vals[e])
                            if cutlass.const_expr(self.mode == MODE_Q):
                                s = _max_f32(
                                    _div_full(amax, cutlass.Float32(6.0)),
                                    cutlass.Float32(0.001953125),
                                )
                                sbits = _e4m3_bits(s)
                                inv = _rcp_approx_ftz(_e4m3_bits_to_f32(sbits))
                            else:
                                s = _mul_rn(g, _mul_rn(amax, rcp6))
                                sbits = _e4m3_bits(s)
                                inv = _rcp_approx_ftz(_mul_rn(_e4m3_bits_to_f32(sbits), rcp_g))
                                inv = _select_nonzero(inv, amax)
                            ys = [_mul_rn(vals[e], inv) for e in range(SF_VEC)]
                            words.append(_pack_e2m1x8(*ys[0:8]))
                            words.append(_pack_e2m1x8(*ys[8:16]))
                            sf_word = sf_word | (sbits << (8 * grp))
                        byte_off = (
                            cutlass.Int64(row_m) * cutlass.Int64(out_stride_m)
                            + cutlass.Int64(head_l) * cutlass.Int64(out_stride_l)
                            + cutlass.Int64(col_n // 2)
                        )
                        for q4 in cutlass.range_constexpr(epi_n // 32):
                            _st_global_v4(
                                out_ptr + (byte_off + 16 * q4),
                                words[4 * q4],
                                words[4 * q4 + 1],
                                words[4 * q4 + 2],
                                words[4 * q4 + 3],
                            )
                        _store = Sm107PersistentDenseGemmNvfp4OutKernel._store_sf
                        _store(
                            sf_ptr,
                            row_m,
                            head_l,
                            col_n,
                            sf_row_m,
                            sf_row_l,
                            sf_col_l,
                            sf_kt512,
                            sf_word,
                            groups,
                        )
                    elif row_m < sf_rows:
                        # scale-plane padding rows (o_proj: up to the next multiple of 128): their A rows are TMA
                        # zero-fill, so the quantizer's scale byte is 0, as fp4_quantize writes; no packed bytes
                        _store = Sm107PersistentDenseGemmNvfp4OutKernel._store_sf
                        _store(
                            sf_ptr,
                            row_m,
                            head_l,
                            col_n,
                            sf_row_m,
                            sf_row_l,
                            sf_col_l,
                            sf_kt512,
                            cutlass.Int32(0),
                            groups,
                        )

                with cute.arch.elect_one():
                    acc_pipeline.consumer_release(acc_consumer_state)
                acc_consumer_state.advance()

            # PDL: dependents may launch once every epilogue store of this CTA is issued (only these warps trigger,
            # unlike the BF16 base kernel's early trigger from the TMA / MMA warps); a dependent's griddepcontrol.wait
            # still waits for this whole grid.
            griddepcontrol_launch_dependents()
            tmem.relinquish_alloc_permit()
            tmem.free(tmem_ptr)

    @staticmethod
    def _store_sf(
        sf_ptr, row_m, head_l, col_n, sf_row_m, sf_row_l, sf_col_l, sf_kt512, sf_word, groups: int
    ):
        """This thread's scale bytes (one per 16 columns) into the 128x4-swizzled plane: row
        r = m * sf_row_m + l * sf_row_l, column c = l * sf_col_l + n / 16 (get_sf_out_offset_128x4)."""
        r = row_m * sf_row_m + head_l * sf_row_l
        c = head_l * sf_col_l + col_n // SF_VEC
        sf_off = cutlass.Int64(r // 128) * cutlass.Int64(sf_kt512) + cutlass.Int64(
            (c // 4) * 512 + (r % 32) * 16 + ((r % 128) // 32) * 4 + (c % 4)
        )
        if cutlass.const_expr(groups == 4):
            _st_global_b32(sf_ptr + sf_off, sf_word)
        else:
            _st_global_b16(sf_ptr + sf_off, sf_word)

    @cute.jit
    def wrapper(
        self,
        m: cutlass.Int32,
        n: cutlass.Int32,
        k: cutlass.Int32,
        batch: cutlass.Int32,
        a_ptr: cute.Pointer,
        b_ptr: cute.Pointer,
        out_ptr: cute.Pointer,
        sf_ptr: cute.Pointer,
        scale_ptr: cute.Pointer,
        a_stride_m: cutlass.Int32,
        a_stride_l: cutlass.Int32,
        b_stride_n: cutlass.Int32,
        b_stride_l: cutlass.Int32,
        out_stride_m: cutlass.Int32,
        out_stride_l: cutlass.Int32,
        sf_row_m: cutlass.Int32,
        sf_row_l: cutlass.Int32,
        sf_col_l: cutlass.Int32,
        sf_kt512: cutlass.Int32,
        fp4_rows: cutlass.Int32,
        sf_rows: cutlass.Int32,
        max_active_clusters: cutlass.Constexpr,
        stream: cuda.CUstream,
    ):
        """A [batch, m, k] (strides a_stride_l, a_stride_m, 1), B [batch, n, k] (b_stride_l, b_stride_n, 1), both BF16.

        out_ptr / sf_ptr: Uint8 global pointers; out_stride_m / out_stride_l in bytes; the scale plane's row
        r = m * sf_row_m + l * sf_row_l and column c = l * sf_col_l + n / 16, sf_kt512 = 512 * (padded columns / 4);
        packed rows m < fp4_rows and scale rows m < sf_rows are stored. scale_ptr: one Float32 (q_global_scale for
        MODE_Q, o_proj's input_scale for MODE_O).
        """
        a = cute.make_tensor(
            a_ptr, layout=cute.make_layout((m, k, batch), stride=(a_stride_m, 1, a_stride_l))
        )
        b = cute.make_tensor(
            b_ptr, layout=cute.make_layout((n, k, batch), stride=(b_stride_n, 1, b_stride_l))
        )
        scale = cute.make_tensor(scale_ptr, layout=cute.make_layout(1))
        self(
            a,
            b,
            out_ptr,
            sf_ptr,
            scale,
            out_stride_m,
            out_stride_l,
            sf_row_m,
            sf_row_l,
            sf_col_l,
            sf_kt512,
            fp4_rows,
            sf_rows,
            max_active_clusters,
            stream,
        )


# ----------------------------------------------------------------------------------------------- host runner
# tactic: (use_2cta_instrs, mma_tiler_mn, cluster_shape_mn, epi_n, max_num_ab_stage, epi_groups)
Nvfp4OutTactic = Tuple[bool, Tuple[int, int], Tuple[int, int], int, int, int]

TACTIC_ENV = {
    MODE_Q: "TRTLLM_MLA_NVFP4_EPILOGUE_TACTIC_Q",
    MODE_O: "TRTLLM_MLA_NVFP4_EPILOGUE_TACTIC_O",
}


def parse_tactic(text: str) -> Nvfp4OutTactic:
    """'2cta,256x256,2x1,64,0[,2]' -> (True, (256, 256), (2, 1), 64, 0, 1 or 2): 2-CTA MMA, MMA tile, cluster, EPI_N,
    A/B stage cap (0: as many as fit), epilogue warp groups (default 1)."""
    parts = [p.strip() for p in text.split(",")]
    if len(parts) not in (5, 6) or parts[0] not in ("1cta", "2cta"):
        raise ValueError(f"tactic must look like '2cta,256x256,2x1,64,0[,2]', got {text!r}")
    tile = tuple(int(v) for v in parts[1].split("x"))
    cluster = tuple(int(v) for v in parts[2].split("x"))
    return (
        parts[0] == "2cta",
        tile,
        cluster,
        int(parts[3]),
        int(parts[4]),
        int(parts[5]) if len(parts) == 6 else 1,
    )


def default_tactic(
    mode: int, m: int, n: int, k: int, batch: int, overlap: bool = False
) -> Nvfp4OutTactic:
    """Static tactic (no autotuner: rank-uniform, no warmup profiling); TACTIC_ENV[mode] overrides.

    From back-to-back tactic sweeps (time per call, real layer weights):
    MODE_Q (N 512, K 128), M > 128: 2-CTA 256x256 with two epilogue warp groups; cluster 2x2 when the RoPE kernel
    runs beside it on another stream, 2x1 when it follows serially. M <= 128: 1-CTA 128x128, two groups.
    MODE_O (N 128, K 512): 1-CTA 128x128.
    """
    import os

    env = os.environ.get(TACTIC_ENV[mode], "")
    if env:
        return parse_tactic(env)
    if mode == MODE_O:
        return (False, (128, 128 if n % 128 == 0 else 64), (1, 1), 64, 0, 1)
    if m <= 128:
        return (False, (128, 128 if n % 128 == 0 else 64), (1, 1), 64, 0, 2)
    if n % 256 == 0:
        return (True, (256, 256), (2, 2) if overlap and n % 512 == 0 else (2, 1), 64, 0, 2)
    return (True, (256, 128), (2, 1), 64, 0, 2)


def candidate_tactics(mode: int, n: int, overlap: bool) -> Tuple[Nvfp4OutTactic, ...]:
    """Every tactic default_tactic can return for this mode / n (compiled before any CUDA graph capture)."""
    return tuple(dict.fromkeys(default_tactic(mode, m, n, 0, 1, overlap) for m in (1, 4096)))


class Sm107Nvfp4OutBmm:
    """Compile cache and launcher for Sm107PersistentDenseGemmNvfp4OutKernel (non-TVM-FFI path, current stream)."""

    _compiled = {}

    @classmethod
    def run(
        cls,
        mode: int,
        a,
        b,
        out,
        sf,
        scale,
        *,
        out_stride_m: int,
        out_stride_l: int,
        sf_row_m: int,
        sf_row_l: int,
        sf_col_l: int,
        sf_cols_padded: int,
        fp4_rows: int,
        sf_rows: int,
        tactic: Nvfp4OutTactic = None,
        overlap: bool = False,
    ) -> Nvfp4OutTactic:
        """a: [batch, m, k] BF16, k stride 1; b: [batch, n, k] BF16, k stride 1; out / sf: uint8 (or byte-sized)
        CUDA tensors the epilogue writes at the given byte strides; scale: one float32 on the device."""
        import torch
        from cutlass.cute.runtime import make_ptr

        batch, m, k = a.shape
        n = b.shape[1]
        if (
            b.shape[0] != batch
            or b.shape[2] != k
            or a.dtype != torch.bfloat16
            or b.dtype != torch.bfloat16
            or a.stride(2) != 1
            or b.stride(2) != 1
            or out.element_size() != 1
            or sf.element_size() != 1
            or scale.dtype != torch.float32
            or scale.numel() < 1
            or sf_cols_padded % 4 != 0
        ):
            raise ValueError(
                "Sm107Nvfp4OutBmm: unsupported operands "
                f"a{tuple(a.shape)}/{a.dtype}/{a.stride()} b{tuple(b.shape)}/{b.dtype}/{b.stride()} "
                f"out {out.dtype} sf {sf.dtype} scale {scale.dtype}"
            )
        if tactic is None:
            tactic = default_tactic(mode, m, n, k, batch, overlap)
        use_2cta, tile, cluster, epi_n, stages, groups = tactic
        if not Sm107PersistentDenseGemmNvfp4OutKernel.can_implement(
            mode, use_2cta, tile, cluster, m, n, k, batch, epi_n
        ):
            raise ValueError(
                f"Sm107Nvfp4OutBmm: tactic {tactic} cannot run mode {mode} m={m} n={n} k={k} "
                f"batch={batch}"
            )
        a_ptr = make_ptr(cutlass.BFloat16, a.data_ptr(), cute.AddressSpace.gmem, assumed_align=16)
        b_ptr = make_ptr(cutlass.BFloat16, b.data_ptr(), cute.AddressSpace.gmem, assumed_align=16)
        out_ptr = make_ptr(cutlass.Uint8, out.data_ptr(), cute.AddressSpace.gmem, assumed_align=16)
        sf_ptr = make_ptr(cutlass.Uint8, sf.data_ptr(), cute.AddressSpace.gmem, assumed_align=4)
        scale_ptr = make_ptr(
            cutlass.Float32, scale.data_ptr(), cute.AddressSpace.gmem, assumed_align=4
        )
        stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
        args = (
            m,
            n,
            k,
            batch,
            a_ptr,
            b_ptr,
            out_ptr,
            sf_ptr,
            scale_ptr,
            a.stride(1),
            a.stride(0),
            b.stride(1),
            b.stride(0),
            out_stride_m,
            out_stride_l,
            sf_row_m,
            sf_row_l,
            sf_col_l,
            512 * (sf_cols_padded // 4),
            fp4_rows,
            sf_rows,
        )
        key = (mode, tactic)
        compiled = cls._compiled.get(key)
        if compiled is None:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    f"Sm107Nvfp4OutBmm: tactic {tactic} (mode {mode}) first compiled while capturing "
                    "a CUDA graph; run a warmup forward first."
                )
            # compile every tactic this mode can pick at another batch size too, so no capture ever compiles
            for t in (tactic,) + candidate_tactics(mode, n, overlap):
                if (mode, t) not in cls._compiled:
                    cls._compiled[(mode, t)] = cls._compile(mode, t, args, stream)
            compiled = cls._compiled[key]
        compiled(*args, stream=stream)
        return tactic

    @staticmethod
    def _compile(mode: int, tactic: Nvfp4OutTactic, args, stream):
        use_2cta, tile, cluster, epi_n, stages, groups = tactic
        kernel = Sm107PersistentDenseGemmNvfp4OutKernel(
            mode, use_2cta, tile, cluster, epi_n, stages, groups
        )
        max_active_clusters = cutlass.utils.HardwareInfo().get_max_active_clusters(
            cluster[0] * cluster[1]
        )
        return cute.compile(
            kernel.wrapper, *args, max_active_clusters, stream=stream, options="--opt-level 2"
        )
