# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# H3 GateUp GEMM and SwiGLU on SM107, with a BF16 projection boundary.
#
# Native SM107 (Rubin) CuTe DSL kernel:
#   * persistent, warp-specialized tcgen05 2-CTA FP8 GEMM, 256x256x64 instructions with the SM107
#     Bkeep/Breuse collector pattern: each CTA owns 2x128 A rows and one 128-row B half per
#     k-block; B is kept in the operand collector for the second A chunk, cutting SMEM reads
#   * CLC (cluster launch control) dynamic scheduling, preferred 4x2 clusters (A and B TMA
#     multicast) with a 2x1 fallback so every SM works; slot -> tile swizzle for L2 reuse
#   * weight rows are prepacked once in setup so that every 256-wide N tile holds 128 gate rows
#     followed by the 128 matching up rows: gate column j and up column j land in TMEM columns
#     j and j+128 of the same TMEM lane (= same thread)
#   * single 512-column TMEM accumulator; 8 epilogue warps drain it quickly into registers as
#     BF16 (bf16_rn(acc*(sa*sw)), the mandatory rounding), release it to the MMA warp, then run
#     the strict SG (same PTX ops as the reference SG kernel) -> saturating E4M3 from registers
#     while the next tile's MMAs execute; packed 32B global stores
#   * BF16 mode uses ordinary 2-CTA MMA, reads the original [gate; up] weights via a
#     strided tensor map and returns BF16 SwiGLU activations
from typing import Tuple

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.rubin_helpers as sm107_helpers
import torch
from cutlass._mlir.dialects import llvm
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.cute.nvgpu.tcgen05 import CollectorOp
from cutlass.cute.runtime import make_fake_compact_tensor, make_fake_stream
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait
from cutlass.utils.gemm.sm100 import transform_partitioned_tensor_layout

K_DIM = 5376
D_HALF = 14336
N_TOT = 2 * D_HALF
HALF_TILE_N = 128  # gate (or up) columns per 256-wide MMA N tile


@dsl_user_op
def _scale_to_bf16x2(a0, a1, s_ab, *, loc=None, ip=None):
    """(acc0, acc1) -> bf16x2 {lo: bf16_rn(acc0*s), hi: bf16_rn(acc1*s)} (the mandatory BF16 GEMM rounding)."""
    return cutlass.Uint32(
        llvm.inline_asm(
            T.i32(),
            [cutlass.Float32(v).ir_value(loc=loc, ip=ip) for v in (a0, a1, s_ab)],
            "{\n.reg .f32 x0, x1;\n.reg .b64 a, s, p;\n"
            "mov.b64 a, {$1, $2};\n"
            "mov.b64 s, {$3, $3};\n"
            "mul.rn.f32x2 p, a, s;\n"
            "mov.b64 {x0, x1}, p;\n"
            "cvt.rn.bf16x2.f32 $0, x1, x0;\n}\n",
            "=r,f,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def _st_shared_zero(ptr, *, loc=None, ip=None):
    llvm.inline_asm(
        None,
        [ptr.toint(loc=loc, ip=ip).ir_value()],
        "st.shared.u32 [$0], 0;",
        "r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@dsl_user_op
def _ld_shared_volatile(ptr, *, loc=None, ip=None):
    return cutlass.Uint32(
        llvm.inline_asm(
            T.i32(),
            [ptr.toint(loc=loc, ip=ip).ir_value()],
            "ld.volatile.shared.u32 $0, [$1];",
            "=r,r",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def _sg_recip(s_g, *, loc=None, ip=None):
    """Loop-invariant part of ptxas' exact div.full.f32(y, s_g) expansion."""
    f32 = T.f32()
    res = llvm.inline_asm(
        llvm.StructType.get_literal([f32, f32]),
        [cutlass.Float32(s_g).ir_value(loc=loc, ip=ip)],
        """{
.reg .pred q0, q1;
.reg .f32 ab, sv, fv;
abs.f32 ab, $2;
setp.gt.f32 q0, ab, 0f7E800000;
setp.geu.f32 q1, ab, 0f00800000;
mov.f32 sv, $2;
mov.f32 fv, 0f3F800000;
@q0 mul.rn.f32 sv, sv, 0f3E800000;
@q0 mov.f32 fv, 0f3E800000;
@!q1 mul.rn.f32 sv, sv, 0f4B800000;
@!q1 mov.f32 fv, 0f4B800000;
rcp.approx.ftz.f32 $0, sv;
mov.f32 $1, fv;
}""",
        "=f,=f,f",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    r_sg = cutlass.Float32(llvm.extractvalue(f32, res, [0], loc=loc, ip=ip))
    f_sg = cutlass.Float32(llvm.extractvalue(f32, res, [1], loc=loc, ip=ip))
    return r_sg, f_sg


@dsl_user_op
def _swiglu4_bf16(g01, g23, u01, u23, f_sg, r_sg, zero, *, loc=None, ip=None):
    """4 BF16 gate/up pairs (packed bf16x2, element 0 in the low half) -> 4 packed E4M3 bytes.

    Matches the H3 reference SG exactly: a=f32(gate), b=f32(up);
    sig = div.full(1, 1 + ex2.approx.ftz(-a*log2e)); y = (a*sig)*b; q = div.full(y, sg);
    min.xorsign.abs(q, 448); cvt.rn.satfinite.e4m3x2.
    """
    args = [cutlass.Uint32(v).ir_value(loc=loc, ip=ip) for v in (g01, g23, u01, u23)]
    args.append(cutlass.Float32(f_sg).ir_value(loc=loc, ip=ip))
    args.append(cutlass.Float32(r_sg).ir_value(loc=loc, ip=ip))
    args.append(cutlass.Uint32(zero).ir_value(loc=loc, ip=ip))

    def one(a, b, o):
        return (
            "mul.rn.f32 t, " + a + ", 0fBFB8AA3B;\n"
            "ex2.approx.ftz.f32 t, t;\n"
            "add.rn.f32 t, t, 0f3F800000;\n"
            "setp.gt.f32 p, t, 0f7E800000;\n"
            "@p mul.rn.f32 t, t, 0f3E800000;\n"
            "rcp.approx.ftz.f32 t, t;\n"
            "@p mul.rn.f32 t, t, 0f3E800000;\n"
            "mul.rn.f32 t, " + a + ", t;\n"
            "mul.rn.f32 t, t, " + b + ";\n"
            "mul.rn.f32 t, t, $5;\n"
            "mul.rn.f32 t, $6, t;\n"
            "min.xorsign.abs.f32 " + o + ", t, 0f43E00000;\n"
        )

    asm = (
        "{\n"
        ".reg .f32 t;\n"
        ".reg .pred p;\n"
        ".reg .f32 ga<4>;\n"
        ".reg .f32 ub<4>;\n"
        ".reg .f32 y<4>;\n"
        ".reg .b16 q<2>;\n"
        ".reg .b32 z<4>;\n"
        # `zero` ($7) is loaded after the accumulator release: OR-ing it in pins the SG math
        # after the TMEM drain so the scheduler cannot stretch the drain (value unchanged).
        "or.b32 z0, $1, $7;\n"
        "or.b32 z1, $2, $7;\n"
        "or.b32 z2, $3, $7;\n"
        "or.b32 z3, $4, $7;\n"
        "shl.b32 ga0, z0, 16;\n"
        "and.b32 ga1, z0, 0xFFFF0000;\n"
        "shl.b32 ga2, z1, 16;\n"
        "and.b32 ga3, z1, 0xFFFF0000;\n"
        "shl.b32 ub0, z2, 16;\n"
        "and.b32 ub1, z2, 0xFFFF0000;\n"
        "shl.b32 ub2, z3, 16;\n"
        "and.b32 ub3, z3, 0xFFFF0000;\n"
        + one("ga0", "ub0", "y0")
        + one("ga1", "ub1", "y1")
        + one("ga2", "ub2", "y2")
        + one("ga3", "ub3", "y3")
        + "cvt.rn.satfinite.e4m3x2.f32 q0, y1, y0;\n"
        "cvt.rn.satfinite.e4m3x2.f32 q1, y3, y2;\n"
        "mov.b32 $0, {q0, q1};\n"
        "}\n"
    )
    return cutlass.Uint32(
        llvm.inline_asm(
            T.i32(),
            args,
            asm,
            "=r,r,r,r,r,f,f,r",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def _swiglu_bf16x2(gate, up, zero, *, loc=None, ip=None):
    """Evaluate plain SwiGLU in FP32 on rounded BF16 inputs and return packed BF16."""
    args = [cutlass.Uint32(v).ir_value(loc=loc, ip=ip) for v in (gate, up, zero)]
    asm = """{
.reg .b32 g, u;
.reg .b16 g_lo, g_hi, u_lo, u_hi;
.reg .f32 a<2>, b<2>, r<2>, t;
or.b32 g, $1, $3;
or.b32 u, $2, $3;
mov.b32 {g_lo, g_hi}, g;
mov.b32 {u_lo, u_hi}, u;
cvt.f32.bf16 a0, g_lo;
cvt.f32.bf16 a1, g_hi;
cvt.f32.bf16 b0, u_lo;
cvt.f32.bf16 b1, u_hi;
"""
    for i in range(2):
        asm += (
            f"mul.rn.f32 t, a{i}, 0fBFB8AA3B;\n"
            "ex2.approx.ftz.f32 t, t;\n"
            "add.rn.f32 t, t, 0f3F800000;\n"
            "div.full.f32 t, 0f3F800000, t;\n"
            f"mul.rn.f32 t, a{i}, t;\n"
            f"add.rn.f32 b{i}, b{i}, 0f00000000;\n"
            f"mul.rn.f32 r{i}, t, b{i};\n"
        )
    asm += "cvt.rn.bf16x2.f32 $0, r1, r0;\n}"
    return cutlass.Uint32(
        llvm.inline_asm(
            T.i32(),
            args,
            asm,
            "=r,r,r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


class GateUpSwiGLUBreuseKernel:
    def __init__(
        self,
        cluster_shape_mn=(4, 2),
        num_ab_stage=4,
        swizzle_size=8,
        fallback_cluster_shape_mn=(2, 1),
        mma_tile_k=128,
        skew_kb=2,
        bf16=False,
    ):
        self.acc_dtype = cutlass.Float32
        self.cta_group = tcgen05.CtaGroup.TWO
        self.bf16 = bf16
        self.mma_inst_shape = (256, 256, 16 if bf16 else 64)
        self.mma_tiler = (512, 256, mma_tile_k)  # 2 x inst M -> Bkeep/Breuse
        self.cluster_shape_mn = cluster_shape_mn
        self.fallback_cluster_shape_mn = fallback_cluster_shape_mn
        self.num_ab_stage = num_ab_stage
        self.num_tmem_alloc_cols = 512
        self.epi_tile = (128, 32)
        self.num_epi_warps = 8
        self.epilogue_warp_id = tuple(range(self.num_epi_warps))
        self.mma_warp_id = 8
        self.tma_warp_id = 9
        self.sched_warp_id = 10
        self.threads_per_cta = 32 * 11
        self.num_clc_response_bytes = 16
        self.tmem_alloc_sync_bar_id = 2
        self.tmem_dealloc_sync_bar_id = 3
        self.arch = "sm_107"
        self.swizzle_size = swizzle_size
        # k-blocks at each tile end/start where keep and reuse MMAs are issued separately so
        # each accumulator half drains while the other half still computes
        self.skew_kb = skew_kb

    def _make_tiled_mma(self, b_collector_op=CollectorOp.DISCARD):
        # No M permutation: the 512-row MMA tile is 2 atoms of 256 rows; within atom c the
        # pair's CTA v owns rows 256c + 128v + [0, 128) (keep acc c=0, reuse acc c=1).
        return sm107_helpers.make_trivial_tiled_mma(
            self.a_dtype,
            self.b_dtype,
            tcgen05.OperandMajorMode.K,
            tcgen05.OperandMajorMode.K,
            self.acc_dtype,
            self.cta_group,
            self.mma_inst_shape,
            permutation_mnk=(1, 1, 1),
            b_collector_op=b_collector_op,
        )

    @cute.jit
    def __call__(
        self,
        act: cute.Tensor,
        wgt: cute.Tensor,
        s_in: cute.Tensor,
        s_w: cute.Tensor,
        s_g: cute.Tensor,
        out: cute.Tensor,
        stream: cuda.CUstream,
    ):
        self.a_dtype = act.element_type
        self.b_dtype = wgt.element_type

        tiled_mma = self._make_tiled_mma()
        if cutlass.const_expr(self.bf16):
            # Each CTA loads one 128-row half of the paired gate/up tile.
            logical_weight_layout = cute.make_layout(
                ((HALF_TILE_N, 2, D_HALF // HALF_TILE_N), K_DIM),
                stride=((K_DIM, D_HALF * K_DIM, HALF_TILE_N * K_DIM), 1),
            )
            wgt = cute.make_tensor(wgt.iterator, logical_weight_layout)
            tiled_mma_bkeep = self._make_tiled_mma()
            tiled_mma_breuse = self._make_tiled_mma()
        else:
            tiled_mma_bkeep = self._make_tiled_mma(CollectorOp.FILL)
            tiled_mma_breuse = self._make_tiled_mma(CollectorOp.LASTUSE)

        self.cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout((*self.cluster_shape_mn, 1)), (tiled_mma.thr_id.shape,)
        )
        self.fb_cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout((*self.fallback_cluster_shape_mn, 1)), (tiled_mma.thr_id.shape,)
        )

        self.a_smem_layout_staged = utils.sm100.make_smem_layout_a(
            tiled_mma, self.mma_tiler, self.a_dtype, self.num_ab_stage
        )
        self.b_smem_layout_staged = utils.sm100.make_smem_layout_b(
            tiled_mma, self.mma_tiler, self.b_dtype, self.num_ab_stage
        )
        a_smem_layout = cute.select(
            self.a_smem_layout_staged, mode=list(range(cute.rank(self.a_smem_layout_staged) - 1))
        )
        b_smem_layout = cute.select(
            self.b_smem_layout_staged, mode=list(range(cute.rank(self.b_smem_layout_staged) - 1))
        )

        tma_atom_a, tma_tensor_a = cute.nvgpu.make_tiled_tma_atom_A(
            utils.sm100.cluster_shape_to_tma_atom_A(self.cluster_shape_mn, tiled_mma.thr_id),
            act,
            a_smem_layout,
            self.mma_tiler,
            tiled_mma,
            self.cluster_layout_vmnk.shape,
        )
        tma_atom_b, tma_tensor_b = cute.nvgpu.make_tiled_tma_atom_B(
            utils.sm100.cluster_shape_to_tma_atom_B(self.cluster_shape_mn, tiled_mma.thr_id),
            wgt,
            b_smem_layout,
            self.mma_tiler,
            tiled_mma,
            self.cluster_layout_vmnk.shape,
        )
        fb_tma_atom_a, fb_tma_tensor_a = cute.nvgpu.make_tiled_tma_atom_A(
            utils.sm100.cluster_shape_to_tma_atom_A(
                self.fallback_cluster_shape_mn, tiled_mma.thr_id
            ),
            act,
            a_smem_layout,
            self.mma_tiler,
            tiled_mma,
            self.fb_cluster_layout_vmnk.shape,
        )
        fb_tma_atom_b, fb_tma_tensor_b = cute.nvgpu.make_tiled_tma_atom_B(
            utils.sm100.cluster_shape_to_tma_atom_B(
                self.fallback_cluster_shape_mn, tiled_mma.thr_id
            ),
            wgt,
            b_smem_layout,
            self.mma_tiler,
            tiled_mma,
            self.fb_cluster_layout_vmnk.shape,
        )
        a_copy_size = cute.size_in_bytes(self.a_dtype, a_smem_layout)
        b_copy_size = cute.size_in_bytes(self.b_dtype, b_smem_layout)
        self.num_tma_load_bytes = (a_copy_size + b_copy_size) * cute.size(tiled_mma.thr_id.shape)

        # CTA tiles: 256 rows (two 128-row A chunks) x 256 N; one 2-CTA pair per 512-row MMA tile.
        m = act.shape[0]
        num_ctas_m = (m + self.mma_tiler[0] - 1) // self.mma_tiler[0] * 2
        num_ctas_mnl = (num_ctas_m, N_TOT // self.mma_tiler[1], 1)
        self.tile_sched_params = utils.ClcDynamicPersistentTileSchedulerParams(
            num_ctas_mnl,
            (*self.cluster_shape_mn, 1),
            1,
            True,
            fallback_cluster_shape_mnk=(*self.fallback_cluster_shape_mn, 1),
        )
        grid = utils.ClcDynamicPersistentTileScheduler.get_grid_shape(self.tile_sched_params)

        self.kernel(
            tiled_mma,
            tiled_mma_bkeep,
            tiled_mma_breuse,
            (tma_atom_a, fb_tma_atom_a),
            (tma_tensor_a, fb_tma_tensor_a),
            (tma_atom_b, fb_tma_atom_b),
            (tma_tensor_b, fb_tma_tensor_b),
            out,
            s_in,
            s_w,
            s_g,
            (self.cluster_layout_vmnk, self.fb_cluster_layout_vmnk),
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            self.tile_sched_params,
        ).launch(
            grid=grid,
            block=[self.threads_per_cta, 1, 1],
            cluster=(*self.cluster_shape_mn, 1),
            fallback_cluster=(*self.fallback_cluster_shape_mn, 1),
            stream=stream,
            smem_merge_branch_allocs=True,
        )

    @cute.kernel
    def kernel(
        self,
        tiled_mma: cute.TiledMma,
        tiled_mma_bkeep: cute.TiledMma,
        tiled_mma_breuse: cute.TiledMma,
        tma_atom_a: Tuple[cute.CopyAtom, cute.CopyAtom],
        mA_mk: Tuple[cute.Tensor, cute.Tensor],
        tma_atom_b: Tuple[cute.CopyAtom, cute.CopyAtom],
        mB_nk: Tuple[cute.Tensor, cute.Tensor],
        mOut: cute.Tensor,
        s_in: cute.Tensor,
        s_w: cute.Tensor,
        s_g: cute.Tensor,
        cluster_layout_vmnk: Tuple[cute.Layout, cute.Layout],
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        tile_sched_params: utils.ClcDynamicPersistentTileSchedulerParams,
    ):
        cbdim_x, cbdim_y, cbdim_z = cute.arch.block_in_cluster_dim()
        is_preferred_cluster = (
            cbdim_x == self.cluster_shape_mn[0]
            and cbdim_y == self.cluster_shape_mn[1]
            and cbdim_z == 1
        )
        if is_preferred_cluster:
            self.kernel_impl(
                tiled_mma,
                tiled_mma_bkeep,
                tiled_mma_breuse,
                tma_atom_a[0],
                mA_mk[0],
                tma_atom_b[0],
                mB_nk[0],
                mOut,
                s_in,
                s_w,
                s_g,
                cluster_layout_vmnk[0],
                a_smem_layout_staged,
                b_smem_layout_staged,
                tile_sched_params,
                self.cluster_shape_mn,
            )
        else:
            self.kernel_impl(
                tiled_mma,
                tiled_mma_bkeep,
                tiled_mma_breuse,
                tma_atom_a[1],
                mA_mk[1],
                tma_atom_b[1],
                mB_nk[1],
                mOut,
                s_in,
                s_w,
                s_g,
                cluster_layout_vmnk[1],
                a_smem_layout_staged,
                b_smem_layout_staged,
                tile_sched_params,
                self.fallback_cluster_shape_mn,
            )

    @cute.jit
    def _remap_tile(self, tile_idx, n_px):
        """Map a CLC work id to a swizzled CTA tile.

        swizzled (CTA M tile, N tile): consecutive slots walk `swizzle_size` N clusters, then M.
        """
        cm_sz = self.cluster_shape_mn[0]
        cn_sz = self.cluster_shape_mn[1]
        tx = tile_idx[0]
        ty = tile_idx[1]
        px = tx // cm_sz
        py = ty // cn_sz
        slot = px + n_px * py
        cn_in = slot % self.swizzle_size
        t = slot // self.swizzle_size
        cm = t % n_px
        cn = (t // n_px) * self.swizzle_size + cn_in
        return cm * cm_sz + (tx - px * cm_sz), cn * cn_sz + (ty - py * cn_sz)

    @cute.jit
    def kernel_impl(
        self,
        tiled_mma: cute.TiledMma,
        tiled_mma_bkeep: cute.TiledMma,
        tiled_mma_breuse: cute.TiledMma,
        tma_atom_a: cute.CopyAtom,
        mA_mk: cute.Tensor,
        tma_atom_b: cute.CopyAtom,
        mB_nk: cute.Tensor,
        mOut: cute.Tensor,
        s_in: cute.Tensor,
        s_w: cute.Tensor,
        s_g: cute.Tensor,
        cluster_layout_vmnk: cute.Layout,
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        tile_sched_params: utils.ClcDynamicPersistentTileSchedulerParams,
        cluster_shape_mn: cutlass.Constexpr,
    ):
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())

        if warp_idx == self.tma_warp_id:
            cpasync.prefetch_descriptor(tma_atom_a)
            cpasync.prefetch_descriptor(tma_atom_b)

        bidx, _, _ = cute.arch.block_idx()
        mma_tile_coord_v = bidx % cute.size(tiled_mma.thr_id.shape)
        is_leader_cta = mma_tile_coord_v == 0
        cta_rank_in_cluster = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        is_first_cta_in_cluster = cta_rank_in_cluster == 0
        block_in_cluster_coord_vmnk = cluster_layout_vmnk.get_flat_coord(cta_rank_in_cluster)
        tidx, _, _ = cute.arch.thread_idx()

        @cute.struct
        class SharedStorage:
            ab_full_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_ab_stage * 2]
            acc_full_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2]
            acc2_full_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2]
            tmem_dealloc_mbar: cutlass.Int64
            tmem_holding_buf: cutlass.Int32
            clc_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2]
            clc_response: cute.struct.Align[cute.struct.MemRange[cutlass.Int32, 4], 16]
            tinfo_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2 * 4]
            tinfo: cute.struct.MemRange[cutlass.Int32, 4 * 4]
            zero_word: cutlass.Int32

        smem = cutlass.memory.SmemAllocator()
        storage = smem.allocate(SharedStorage)
        zero_ptr = storage.zero_word.ptr
        if warp_idx == self.tma_warp_id:
            with cute.arch.elect_one():
                _st_shared_zero(zero_ptr)

        ab_producer, ab_consumer = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.ab_full_mbar_ptr.data_ptr(),
            num_stages=self.num_ab_stage,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Warp),
            tx_count=self.num_tma_load_bytes,
            cta_layout_vmnk=cluster_layout_vmnk,
            enable_multicast_signaling=True,
            defer_sync=True,
        ).make_participants()

        # keep accumulator (M atom 0) and reuse accumulator (M atom 1) are handed off separately
        acc_pipeline = pipeline.PipelineUmmaAsync.create(
            barrier_storage=storage.acc_full_mbar_ptr.data_ptr(),
            num_stages=1,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, self.num_epi_warps * 2),
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )
        acc2_pipeline = pipeline.PipelineUmmaAsync.create(
            barrier_storage=storage.acc2_full_mbar_ptr.data_ptr(),
            num_stages=1,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, self.num_epi_warps * 2),
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )

        cluster_size = cluster_shape_mn[0] * cluster_shape_mn[1]
        clc_pipeline = pipeline.PipelineClcFetchAsync.create(
            barrier_storage=storage.clc_mbar_ptr.data_ptr(),
            num_stages=1,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread,
                32 * (1 + cluster_size * 2),
            ),
            tx_count=self.num_clc_response_bytes,
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )
        clc_response_ptr = storage.clc_response.data_ptr()
        clc_consumer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, 1)

        # Decouple the eight epilogue warps from CLC.  The MMA warp publishes the
        # already-remapped coordinates of each accumulator through a four-deep CTA ring.
        tinfo_pipeline = pipeline.PipelineAsync.create(
            barrier_storage=storage.tinfo_mbar_ptr.data_ptr(),
            num_stages=4,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, self.num_epi_warps),
            defer_sync=True,
        )
        tinfo = cute.make_tensor(storage.tinfo.data_ptr(), cute.make_layout((4, 4), stride=(1, 4)))

        tmem_alloc_barrier = pipeline.NamedBarrier(
            barrier_id=self.tmem_alloc_sync_bar_id,
            num_threads=32 * (1 + self.num_epi_warps),
        )
        tmem_dealloc_barrier = pipeline.NamedBarrier(
            barrier_id=self.tmem_dealloc_sync_bar_id,
            num_threads=32 * self.num_epi_warps,
        )
        tmem = cutlass.memory.TmemAllocator(
            storage.tmem_holding_buf.ptr,
            barrier_for_retrieve=tmem_alloc_barrier,
            allocator_warp_id=self.epilogue_warp_id[0],
            is_two_cta=True,
            two_cta_tmem_dealloc_mbar_ptr=storage.tmem_dealloc_mbar.ptr,
            arch=self.arch,
        )

        pipeline_init_arrive(cluster_shape_mn=cluster_shape_mn, is_relaxed=True)

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

        a_full_mcast_mask = cpasync.create_tma_multicast_mask(
            cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=2
        )
        b_full_mcast_mask = cpasync.create_tma_multicast_mask(
            cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=1
        )

        # (bM, bK, RestM, RestK)
        gA_mk = cute.local_tile(mA_mk, cute.slice_(self.mma_tiler, (None, 0, None)), (None, None))
        # (bN, bK, RestN, RestK)
        gB_nk = cute.local_tile(mB_nk, cute.slice_(self.mma_tiler, (0, None, None)), (None, None))
        k_tile_cnt = cute.size(gA_mk, mode=[3])

        thr_mma = tiled_mma.get_slice(mma_tile_coord_v)
        tCgA = thr_mma.partition_A(gA_mk)
        tCgB = thr_mma.partition_B(gB_nk)

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
        # (MMA, MMA_M=2, MMA_N=1): keep acc = [.., 0, 0], reuse acc = [.., 1, 0]
        tCtAcc_fake = tiled_mma.make_fragment_C(acc_shape)

        pipeline_init_wait(cluster_shape_mn=cluster_shape_mn)

        # TMEM: warp 0 allocates; a CTA-wide barrier (every warp, same program point) publishes
        # the address before the MMA / epilogue warps retrieve it.
        tmem.allocate(self.num_tmem_alloc_cols)  # executed by the allocator warp only
        cute.arch.barrier()

        tile_sched = utils.ClcDynamicPersistentTileScheduler.create(
            tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim(), clc_response_ptr
        )
        work_tile = tile_sched.initial_work_tile_info()
        n_px = (
            (mOut.shape[0] + self.mma_tiler[0] - 1) // self.mma_tiler[0] * 2
            + self.cluster_shape_mn[0]
            - 1
        ) // self.cluster_shape_mn[0]

        # ---------------- TMA producer warp ----------------
        if warp_idx == self.tma_warp_id:
            while work_tile.is_valid_tile:
                m_cta, n_t = self._remap_tile(work_tile.tile_idx, n_px)
                tAgA_slice = tAgA[(None, m_cta // 2, None)]
                tBgB_slice = tBgB[(None, n_t, None)]

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

                clc_pipeline.consumer_wait(clc_consumer_state)
                work_tile = tile_sched.get_current_work()
                clc_pipeline.consumer_release(clc_consumer_state)
                clc_consumer_state.advance()
            ab_producer.tail()

        # ---------------- CLC scheduler warp (first CTA of each cluster) ----------------
        if warp_idx == self.sched_warp_id and is_first_cta_in_cluster:
            clc_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.ProducerConsumer, 1
            )
            while work_tile.is_valid_tile:
                clc_pipeline.producer_acquire(clc_producer_state)
                mbarrier_addr = clc_pipeline.producer_get_barrier(clc_producer_state)
                tile_sched.advance_to_next_work(mbarrier_addr)
                clc_producer_state.advance()
                clc_pipeline.consumer_wait(clc_consumer_state)
                work_tile = tile_sched.get_current_work()
                clc_pipeline.consumer_release(clc_consumer_state)
                clc_consumer_state.advance()
            clc_pipeline.producer_tail(clc_producer_state)

        # ---------------- MMA warp (Bkeep / Breuse) ----------------
        if warp_idx == self.mma_warp_id:
            tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            tCtAcc = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)
            tCtAcc_keep = tCtAcc[(None, 0, 0)]
            tCtAcc_reuse = tCtAcc[(None, 1, 0)]
            acc_producer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, 1)
            acc2_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, 1
            )
            tinfo_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, 4
            )
            num_kphase = self.mma_tiler[2] // self.mma_inst_shape[2]
            skew = self.skew_kb
            while work_tile.is_valid_tile:
                m_cta, n_t = self._remap_tile(work_tile.tile_idx, n_px)
                tinfo_pipeline.producer_acquire(tinfo_producer_state)
                with cute.arch.elect_one():
                    tinfo[0, tinfo_producer_state.index] = m_cta
                    tinfo[1, tinfo_producer_state.index] = n_t
                    tinfo[2, tinfo_producer_state.index] = cutlass.Int32(1)
                    tinfo_pipeline.producer_commit(tinfo_producer_state)
                tinfo_producer_state.advance()
                ab_consumer.reset()
                if is_leader_cta:
                    # ---- head: keep-only MMAs while the epilogue still drains the reuse half ----
                    acc_pipeline.producer_acquire(acc_producer_state)
                    head = []
                    for kb in cutlass.range_constexpr(skew):
                        h = ab_consumer.wait_and_advance()
                        head.append(h)
                        for kp in cutlass.range_constexpr(num_kphase):
                            tiled_mma.set(tcgen05.Field.ACCUMULATE, kb != 0 or kp != 0)
                            cute.gemm(
                                tiled_mma,
                                tCtAcc_keep,
                                tCrA[(None, 0, kp, h.index)],
                                tCrB[(None, 0, kp, h.index)],
                                tCtAcc_keep,
                            )
                    acc2_pipeline.producer_acquire(acc2_producer_state)
                    for kb in cutlass.range_constexpr(skew):
                        h = head[kb]
                        for kp in cutlass.range_constexpr(num_kphase):
                            tiled_mma.set(tcgen05.Field.ACCUMULATE, kb != 0 or kp != 0)
                            cute.gemm(
                                tiled_mma,
                                tCtAcc_reuse,
                                tCrA[(None, 1, kp, h.index)],
                                tCrB[(None, 0, kp, h.index)],
                                tCtAcc_reuse,
                            )
                        h.release()
                    # ---- middle: Bkeep/Breuse pairs share the B operand collector ----
                    tiled_mma_bkeep.set(tcgen05.Field.ACCUMULATE, True)
                    tiled_mma_breuse.set(tcgen05.Field.ACCUMULATE, True)
                    peek_ab_full_status = ab_consumer.try_wait()
                    for k_tile in cutlass.range(skew, k_tile_cnt - skew, 1, unroll=2):
                        handle = ab_consumer.wait_and_advance(peek_ab_full_status)
                        tCrA_stage = tCrA[(None, None, None, handle.index)]
                        tCrB_stage = tCrB[(None, None, None, handle.index)]
                        for k_phase in cutlass.range_constexpr(num_kphase):
                            tCrB_slice = tCrB_stage[(None, 0, k_phase)]
                            cute.gemm(
                                tiled_mma_bkeep,
                                tCtAcc_keep,
                                tCrA_stage[(None, 0, k_phase)],
                                tCrB_slice,
                                tCtAcc_keep,
                            )
                            cute.gemm(
                                tiled_mma_breuse,
                                tCtAcc_reuse,
                                tCrA_stage[(None, 1, k_phase)],
                                tCrB_slice,
                                tCtAcc_reuse,
                            )
                        handle.release()
                        peek_ab_full_status = ab_consumer.try_wait()
                    # ---- tail: finish keep first and hand it off, then finish reuse ----
                    tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                    tail = []
                    for kb in cutlass.range_constexpr(skew):
                        h = ab_consumer.wait_and_advance()
                        tail.append(h)
                        for kp in cutlass.range_constexpr(num_kphase):
                            cute.gemm(
                                tiled_mma,
                                tCtAcc_keep,
                                tCrA[(None, 0, kp, h.index)],
                                tCrB[(None, 0, kp, h.index)],
                                tCtAcc_keep,
                            )
                    acc_pipeline.producer_commit(acc_producer_state)
                    for kb in cutlass.range_constexpr(skew):
                        h = tail[kb]
                        for kp in cutlass.range_constexpr(num_kphase):
                            cute.gemm(
                                tiled_mma,
                                tCtAcc_reuse,
                                tCrA[(None, 1, kp, h.index)],
                                tCrB[(None, 0, kp, h.index)],
                                tCtAcc_reuse,
                            )
                        h.release()
                    acc2_pipeline.producer_commit(acc2_producer_state)
                acc_producer_state.advance()
                acc2_producer_state.advance()
                clc_pipeline.consumer_wait(clc_consumer_state)
                work_tile = tile_sched.get_current_work()
                clc_pipeline.consumer_release(clc_consumer_state)
                clc_consumer_state.advance()
            tinfo_pipeline.producer_acquire(tinfo_producer_state)
            with cute.arch.elect_one():
                tinfo[2, tinfo_producer_state.index] = cutlass.Int32(0)
                tinfo_pipeline.producer_commit(tinfo_producer_state)
            tinfo_producer_state.advance()
            acc_pipeline.producer_tail(acc_producer_state)
            acc2_pipeline.producer_tail(acc2_producer_state)

        # ---------------- Epilogue warps: TMEM -> BF16 regs -> release -> SG -> E4M3 -> global ----------------
        if warp_idx < self.mma_warp_id:
            tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            tCtAcc = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)

            # every warp drains its TMEM lane quarter of the keep accumulator (M atom 0) and then of
            # the reuse accumulator (M atom 1); warps 0-3 take gate/up columns [0, 64), warps 4-7
            # columns [64, 128): each thread owns one keep row and one reuse row, 64 outputs each
            hh = warp_idx // 4
            etid = tidx % 128

            # ((128,2),(256,1)) -> (128, 32, 2, 8): [lane, col, chunk, gate 0..3 | up 4..7]
            tAcc = transform_partitioned_tensor_layout(tCtAcc)
            tAcc_epi = cute.flat_divide(tAcc, self.epi_tile)
            copy_atom_t2r = cute.make_copy_atom(
                tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition.x32), self.acc_dtype
            )
            tiled_copy_t2r = tcgen05.make_tmem_copy(copy_atom_t2r, tAcc_epi[(None, None, 0, 0)])
            thr_copy_t2r = tiled_copy_t2r.get_slice(etid)
            # (T2R, T2R_M, T2R_N, EPI_M, EPI_N)
            tTR_tAcc = thr_copy_t2r.partition_S(tAcc_epi)

            tTR_rAcc = cute.make_rmem_tensor(
                thr_copy_t2r.partition_D(cute.make_identity_tensor(self.epi_tile)).shape,
                self.acc_dtype,
            )
            n_val = cute.size(tTR_rAcc)  # 32 fp32 columns per subtile
            num_sub = HALF_TILE_N // self.epi_tile[1]  # 4 gate subtiles (+4 up subtiles)
            # BF16 staging: [gate subtiles 0..3 | up subtiles 0..3] x 16 packed bf16x2
            rStage = cute.make_rmem_tensor((2 * num_sub * n_val // 2,), cutlass.Uint32)
            tTR_rO = cute.make_rmem_tensor((n_val,), mOut.element_type)
            tTR_rO32 = cute.recast_tensor(tTR_rO, cutlass.Uint32)
            store_atom = cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(), mOut.element_type, num_bits_per_copy=256
            )

            if cutlass.const_expr(self.bf16):
                s_ab = cutlass.Float32(1.0)
            else:
                s_ab = s_in[0] * s_w[0]
                r_sg, f_sg = _sg_recip(s_g[0])
            m_total = mOut.shape[0]

            acc_consumer_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, 1)
            acc2_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, 1
            )
            tinfo_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, 4
            )
            tinfo_pipeline.consumer_wait(tinfo_consumer_state)
            m_cta = tinfo[0, tinfo_consumer_state.index]
            n_t = tinfo[1, tinfo_consumer_state.index]
            valid = tinfo[2, tinfo_consumer_state.index] != 0
            cute.arch.sync_warp()
            with cute.arch.elect_one():
                tinfo_pipeline.consumer_release(tinfo_consumer_state)
            tinfo_consumer_state.advance()
            while valid:
                # 512-row MMA tile (m_cta // 2): pair CTA v = m_cta % 2 owns rows 128v + [0, 128) of
                # each 256-row M atom; keep row = atom 0, reuse row = atom 1
                row = (m_cta // 2) * 512 + (m_cta % 2) * 128 + etid
                out_col = n_t * HALF_TILE_N + hh * (HALF_TILE_N // 2)
                sub_g = hh * (num_sub // 2)

                # ---- drain: TMEM fp32 -> bf16_rn(acc * sa*sw) staged in registers ----
                acc_pipeline.consumer_wait(acc_consumer_state)
                cute.arch.sync_warp()
                for j in cutlass.range_constexpr(num_sub):
                    sub = (j // 2) * num_sub + sub_g + (j % 2)
                    cute.copy(tiled_copy_t2r, tTR_tAcc[(None, None, None, 0, sub)], tTR_rAcc)
                    for i in cutlass.range_constexpr(n_val // 2):
                        rStage[j * (n_val // 2) + i] = _scale_to_bf16x2(
                            tTR_rAcc[2 * i], tTR_rAcc[2 * i + 1], s_ab
                        )
                cute.arch.fence_view_async_tmem_load()
                with cute.arch.elect_one():
                    acc_pipeline.consumer_release(acc_consumer_state)
                acc_consumer_state.advance()
                acc2_pipeline.consumer_wait(acc2_consumer_state)
                cute.arch.sync_warp()
                for j in cutlass.range_constexpr(num_sub):
                    sub = (j // 2) * num_sub + sub_g + (j % 2)
                    cute.copy(tiled_copy_t2r, tTR_tAcc[(None, None, None, 1, sub)], tTR_rAcc)
                    for i in cutlass.range_constexpr(n_val // 2):
                        rStage[(num_sub + j) * (n_val // 2) + i] = _scale_to_bf16x2(
                            tTR_rAcc[2 * i], tTR_rAcc[2 * i + 1], s_ab
                        )
                cute.arch.fence_view_async_tmem_load()
                with cute.arch.elect_one():
                    acc2_pipeline.consumer_release(acc2_consumer_state)
                acc2_consumer_state.advance()
                zero = _ld_shared_volatile(zero_ptr)

                # ---- SwiGLU from registers (overlaps the next tile's MMAs) ----
                for c in cutlass.range_constexpr(2):
                    gRow = cute.local_tile(mOut[(row + c * 256, None)], (n_val,), (None,))
                    for s in cutlass.range_constexpr(2):
                        if cutlass.const_expr(self.bf16):
                            for i in cutlass.range_constexpr(n_val // 2):
                                g = (c * num_sub + s) * (n_val // 2) + i
                                u = (c * num_sub + 2 + s) * (n_val // 2) + i
                                tTR_rO32[i] = _swiglu_bf16x2(rStage[g], rStage[u], zero)
                        else:
                            for i in cutlass.range_constexpr(n_val // 4):
                                g = (c * num_sub + s) * (n_val // 2) + 2 * i
                                u = (c * num_sub + 2 + s) * (n_val // 2) + 2 * i
                                tTR_rO32[i] = _swiglu4_bf16(
                                    rStage[g],
                                    rStage[g + 1],
                                    rStage[u],
                                    rStage[u + 1],
                                    f_sg,
                                    r_sg,
                                    zero,
                                )
                        if row + c * 256 < m_total:
                            cute.copy(store_atom, tTR_rO, gRow[(None, out_col // n_val + s)])
                        cute.arch.sync_warp()

                tinfo_pipeline.consumer_wait(tinfo_consumer_state)
                m_cta = tinfo[0, tinfo_consumer_state.index]
                n_t = tinfo[1, tinfo_consumer_state.index]
                valid = tinfo[2, tinfo_consumer_state.index] != 0
                cute.arch.sync_warp()
                with cute.arch.elect_one():
                    tinfo_pipeline.consumer_release(tinfo_consumer_state)
                tinfo_consumer_state.advance()

            tmem_dealloc_barrier.arrive_and_wait()
            tmem.relinquish_alloc_permit()
            tmem.free(tmem_ptr)


# ----------------------------------------------------------------------------------------------
# Host side
# ----------------------------------------------------------------------------------------------
_CFG = dict(
    cluster_shape_mn=(2, 4),
    num_ab_stage=6,
    swizzle_size=7,
    fallback_cluster_shape_mn=(2, 1),
    mma_tile_k=128,
    skew_kb=3,
)
_compiled = {}


def _smem_optin_bytes(device_index):
    """Real per-block opt-in shared memory of the device we run on (never the arch table)."""
    return int(torch.cuda.get_device_properties(device_index).shared_memory_per_block_optin)


def _cache_key(device_index, bf16):
    return device_index, bf16, repr(sorted(_CFG.items()))


def precompiled_bf16(device_index):
    return _compiled.get(_cache_key(device_index, True))


def _get_compiled(device_index, smem_optin=None, *, bf16=False):
    key = _cache_key(device_index, bf16)
    fn = _compiled.get(key)
    if fn is None:
        cfg = dict(_CFG)
        element_type = cutlass.BFloat16 if bf16 else cutlass.Float8E4M3FN
        element_bytes = 2 if bf16 else 1
        if bf16:
            cfg["mma_tile_k"] = 64
        if smem_optin is None:
            smem_optin = _smem_optin_bytes(device_index)
        if cfg.get("num_ab_stage") is None:
            # A (2x128 rows) + B (128 rows) per stage; 4 KiB for barriers/alignment.
            cfg["num_ab_stage"] = max(
                2, min(16, (smem_optin - 4096) // (3 * 128 * cfg["mma_tile_k"] * element_bytes))
            )
        kern = GateUpSwiGLUBreuseKernel(**cfg, bf16=bf16)
        m_sym = cute.sym_int32()
        act = make_fake_compact_tensor(
            element_type, (m_sym, K_DIM), stride_order=(1, 0), assumed_align=256
        )
        wgt = make_fake_compact_tensor(
            element_type, (N_TOT, K_DIM), stride_order=(1, 0), assumed_align=256
        )
        s_in = make_fake_compact_tensor(cutlass.Float32, (1,), assumed_align=4)
        s_w = make_fake_compact_tensor(cutlass.Float32, (1,), assumed_align=4)
        s_g = make_fake_compact_tensor(cutlass.Float32, (1,), assumed_align=4)
        out = make_fake_compact_tensor(
            element_type, (m_sym, D_HALF), stride_order=(1, 0), assumed_align=256
        )
        stream = make_fake_stream(use_tvm_ffi_env_stream=True)
        fn = cute.compile(kern, act, wgt, s_in, s_w, s_g, out, stream, options="--enable-tvm-ffi")
        _compiled[key] = fn
    return fn


def pack_weight(weight):
    """[all gate; all up] rows -> per 256-row tile: [128 gate rows of tile t; 128 up rows of tile t]."""
    w = weight.view(2, D_HALF // HALF_TILE_N, HALF_TILE_N, K_DIM)
    return w.permute(1, 0, 2, 3).contiguous().view(N_TOT, K_DIM)
