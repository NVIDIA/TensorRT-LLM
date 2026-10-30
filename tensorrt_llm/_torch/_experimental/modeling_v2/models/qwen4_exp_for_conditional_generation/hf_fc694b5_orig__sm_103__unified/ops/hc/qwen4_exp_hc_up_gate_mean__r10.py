# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Hyper-connection mixer second half as one tcgen05 GEMM with a custom
epilogue: the up projection `low @ up^T` accumulates in tensor memory and the
epilogue applies the sigmoid gate to the normed stream and averages the
streams, so the `[T, C*H]` gate logits never reach global memory.

A CuTe DSL kernel from the tp1 target's `ops/hc/qwen4_exp_hc_up_gate_mean`,
reused here as this target's local variant. A persistent warp-specialized
kernel: one TMA warp loads `low` and the up weight, one MMA warp issues
`tcgen05.mma` into a double-buffered TMEM accumulator, four epilogue warps
read the accumulator, load the matching normed elements straight from global
memory and write the mean. An MMA tile is 128 rows by `C * BH` up-projection
outputs: the `BH`-wide hidden slice `[h0, h0 + BH)` of every stream, read
through a strided view of the unmodified `[C*H, R]` weight (no copy, no
re-layout), so the stream reduction stays inside one thread.
"""

from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import torch
import torch.nn.functional as F
from cutlass._mlir.dialects import llvm
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.cute.runtime import from_dlpack
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper
from tensorrt_llm._torch.flashinfer_utils import get_env_enable_pdl
from tensorrt_llm._torch.kernels.blackwell.custom_pipeline import PipelineTmaUmma, PipelineUmmaAsync
from tensorrt_llm._torch.kernels.blackwell.utils import make_ptr

#: Programmatic dependent launch (sm_90+), off with TRTLLM_ENABLE_PDL=0.
_PDL = get_env_enable_pdl()


@dsl_user_op
def _sigmoid(x, *, loc=None, ip=None):
    """sigmoid(x) = 0.5 * tanh(0.5 * x) + 0.5: one MUFU op (tanh.approx.f32,
    max relative error 2^-10.99) instead of ex2 + rcp."""
    return cutlass.Float32(
        llvm.inline_asm(
            T.f32(),
            [cutlass.Float32(x).ir_value(loc=loc, ip=ip)],
            "{ .reg .f32 t; mul.f32 t, $1, 0f3F000000; tanh.approx.f32 t, t; "
            "fma.rn.f32 $0, t, 0f3F000000, 0f3F000000; }",
            "=f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


# The NVFP4 epilogue (`fp4=True`) is the `torch.ops.trtllm.fp4_quantize` recipe
# (`cvt_warp_fp16_to_fp4`, e4m3 scales) instruction for instruction, every
# multiply an explicit `mul.rn.f32` so that no compiler contraction or
# reassociation can move a rounding.


def _asm(ret, args, text, constraints, loc, ip, side_effects=False):
    return llvm.inline_asm(
        ret,
        [a.ir_value(loc=loc, ip=ip) for a in args],
        text,
        constraints,
        has_side_effects=side_effects,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _rcp_approx_ftz(x, *, loc=None, ip=None):
    """`rcp.approx.ftz.f32`, the quantizer's `reciprocal_approximate_ftz`."""
    return cutlass.Float32(
        _asm(T.f32(), [cutlass.Float32(x)], "rcp.approx.ftz.f32 $0, $1;", "=f,f", loc, ip)
    )


@dsl_user_op
def _absmax16(v, *, loc=None, ip=None):
    """max |v_i| over 16 fp32 values (exact; NaN-quiet like the bf16 `__hmax2`)."""
    body = "abs.f32 m, $1; " + "".join(f"abs.f32 t, ${i}; max.f32 m, m, t; " for i in range(2, 17))
    return cutlass.Float32(
        _asm(
            T.f32(),
            [cutlass.Float32(x) for x in v],
            "{ .reg .f32 m, t; " + body + "mov.f32 $0, m; }",
            "=f" + ",f" * 16,
            loc,
            ip,
        )
    )


@dsl_user_op
def _e4m3_scale(amax, gs, rcp6, *, loc=None, ip=None):
    """The block's scale byte: e4m3(gs * (amax * rcp(6))), `cvt.rn.satfinite`."""
    return cutlass.Int32(
        _asm(
            T.i32(),
            [cutlass.Float32(amax), cutlass.Float32(gs), cutlass.Float32(rcp6)],
            "{ .reg .f32 t; .reg .b16 h; mul.rn.f32 t, $1, $3; mul.rn.f32 t, $2, t; "
            "cvt.rn.satfinite.e4m3x2.f32 h, 0f00000000, t; cvt.u32.u16 $0, h; }",
            "=r,f,f,f",
            loc,
            ip,
        )
    )


@dsl_user_op
def _e2m1_out_scale(sf, rcp_gs, amax, *, loc=None, ip=None):
    """amax != 0 ? rcp(float(e4m3 sf) * rcp(gs)) : 0, the data multiplier."""
    return cutlass.Float32(
        _asm(
            T.f32(),
            [cutlass.Int32(sf), cutlass.Float32(rcp_gs), cutlass.Float32(amax)],
            "{ .reg .b16 h, lo, hi; .reg .b32 f2; .reg .f32 s, t; .reg .pred p; "
            "cvt.u16.u32 h, $1; cvt.rn.f16x2.e4m3x2 f2, h; mov.b32 {lo, hi}, f2; "
            "cvt.f32.f16 s, lo; mul.rn.f32 t, s, $2; rcp.approx.ftz.f32 t, t; "
            "setp.neu.f32 p, $3, 0f00000000; selp.f32 $0, t, 0f00000000, p; }",
            "=f,r,f,f",
            loc,
            ip,
        )
    )


@dsl_user_op
def _e2m1x8(v, scale, *, loc=None, ip=None):
    """Eight values times `scale`, each to e2m1 (`cvt.rn.satfinite`), packed
    into 4 bytes; element `2k` in byte k's low nibble."""
    muls = "".join(f"mul.rn.f32 a{i}, ${i + 1}, $9; " for i in range(8))
    cvts = "".join(f"cvt.rn.satfinite.e2m1x2.f32 b{k}, a{2 * k + 1}, a{2 * k}; " for k in range(4))
    return cutlass.Int32(
        _asm(
            T.i32(),
            [cutlass.Float32(x) for x in v] + [cutlass.Float32(scale)],
            "{ .reg .f32 a<8>; .reg .b8 b<4>; " + muls + cvts + "mov.b32 $0, {b0, b1, b2, b3}; }",
            "=r" + ",f" * 9,
            loc,
            ip,
        )
    )


@dsl_user_op
def _st_global_v4(addr, a, b, c, d, *, loc=None, ip=None):
    """One 16-byte `st.global.v4.b32` (a row's packed e2m1 for 32 columns): a
    row-per-thread epilogue pays per store instruction, not per byte."""
    _asm(
        T.i32(),
        [cutlass.Int64(addr)] + [cutlass.Int32(x) for x in (a, b, c, d)],
        "{ st.global.v4.b32 [$1], {$2, $3, $4, $5}; mov.u32 $0, 0; }",
        "=r,l,r,r,r,r",
        loc,
        ip,
        side_effects=True,
    )


@dsl_user_op
def _st_global_b16(addr, x, *, loc=None, ip=None):
    """One 2-byte store of the low half of `x` (two e4m3 scale bytes)."""
    _asm(
        T.i32(),
        [cutlass.Int64(addr), cutlass.Int32(x)],
        "{ .reg .b16 t; cvt.u16.u32 t, $2; st.global.b16 [$1], t; mov.u32 $0, 0; }",
        "=r,l,r",
        loc,
        ip,
        side_effects=True,
    )


class HcUpGateMeanKernel:
    """`mixed[t, h] = 1/C * sum_s sigmoid((low @ up^T)[t, s*H + h]) * normed[t, s*H + h]`.

    Compile-time: `block_h` (hidden columns per tile, 32 or 64), `hc` (C),
    `hidden` (H), `rank` (R), `fp4` (also write the NVFP4 of `mixed`: packed
    e2m1 `[T, H/2]` and linear e4m3 scales `[T, H/16]` under the global scale
    `gs`). Runtime: the row count and the pointers; without `fp4` the `xq`,
    `sf` and `gs` pointers are never read.
    """

    def __init__(
        self, block_h: int, hc: int, hidden: int, rank: int, num_n_stage: int = 2, fp4: bool = False
    ):
        assert block_h in (32, 64) and hidden % block_h == 0 and rank % 64 == 0
        self.bh, self.hc, self.hidden, self.rank = block_h, hc, hidden, rank
        self.fp4 = fp4
        self.acc_dtype = cutlass.Float32
        self.io_dtype = cutlass.BFloat16
        self.mma_tiler = (128, hc * block_h, 1)
        self.epi_cols = 32
        self.cta_group = tcgen05.CtaGroup.ONE
        self.cluster_shape_mn = (1, 1)
        self.epilogue_warp_id = (0, 1, 2, 3)
        self.mma_warp_id = 4
        self.tma_warp_id = 5
        self.nrm_warp_id = 6
        self.threads_per_cta = 32 * 7
        self.num_n_stage = num_n_stage
        self.epilog_sync_bar_id = 1
        self.tmem_alloc_sync_bar_id = 2
        self.num_acc_stage = 2

    def _setup(self, a_major, b_major):
        tiled_mma = utils.sm100.make_trivial_tiled_mma(
            self.io_dtype, a_major, b_major, self.acc_dtype, self.cta_group, self.mma_tiler[:2]
        )
        inst_k = cute.size(tiled_mma.shape_mnk, mode=[2])
        self.mma_tiler = (self.mma_tiler[0], self.mma_tiler[1], inst_k * 4)
        self.cta_tile_shape_mnk = self.mma_tiler
        self.cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout((*self.cluster_shape_mn, 1)), (tiled_mma.thr_id.shape,)
        )
        self.epi_tile = (self.mma_tiler[0], self.epi_cols)
        a_one = utils.sm100.make_smem_layout_a(tiled_mma, self.mma_tiler, self.io_dtype, 1)
        b_one = utils.sm100.make_smem_layout_b(tiled_mma, self.mma_tiler, self.io_dtype, 1)
        ab_bytes = cute.size_in_bytes(self.io_dtype, a_one) + cute.size_in_bytes(
            self.io_dtype, b_one
        )
        smem_capacity = utils.get_smem_capacity_in_bytes()
        # One normed stage is the 32-column slice of every stream for one tile half.
        self.n_box = (self.mma_tiler[0], self.epi_cols)
        self.num_n_bytes = self.hc * self.n_box[0] * self.n_box[1] * 2
        self.num_ab_stage = min(
            8, (smem_capacity - 2048 - self.num_n_stage * self.num_n_bytes) // ab_bytes
        )
        self.n_pre = min(self.num_ab_stage, self.rank // self.mma_tiler[2])
        self.n_smem_layout_staged = utils.sm100.make_smem_layout_epi(
            self.io_dtype, utils.LayoutEnum.ROW_MAJOR, self.n_box, self.num_n_stage * self.hc
        )
        self.a_smem_layout_staged = utils.sm100.make_smem_layout_a(
            tiled_mma, self.mma_tiler, self.io_dtype, self.num_ab_stage
        )
        self.b_smem_layout_staged = utils.sm100.make_smem_layout_b(
            tiled_mma, self.mma_tiler, self.io_dtype, self.num_ab_stage
        )
        acc_shape = tiled_mma.partition_shape_C(self.mma_tiler[:2])
        acc_fake = tiled_mma.make_fragment_C(cute.append(acc_shape, self.num_acc_stage))
        self.num_tmem_alloc_cols = utils.get_num_tmem_alloc_cols(acc_fake, arch="sm_100")
        return tiled_mma

    @cute.jit
    def __call__(
        self,
        m: cutlass.Int32,
        low_ptr: cute.Pointer,
        up_ptr: cute.Pointer,
        normed_ptr: cute.Pointer,
        mixed: cute.Tensor,
        xq_ptr: cute.Pointer,
        sf_ptr: cute.Pointer,
        gs_ptr: cute.Pointer,
        max_active_clusters: cutlass.Constexpr,
        stream: cuda.CUstream,
    ):
        bh, hc, hid, rank = self.bh, self.hc, self.hidden, self.rank
        nt = hid // bh
        width = hc * hid
        # low [m, R] (K-major A); the up weight [C*H, R] seen as N = (BH, C, H/BH) so
        # that one N tile holds hidden slice [h0, h0+BH) of every stream.
        a = cute.make_tensor(low_ptr, cute.make_layout((m, rank, 1), stride=(rank, 1, 0)))
        b = cute.make_tensor(
            up_ptr,
            cute.make_layout(((bh, hc, nt), rank, 1), stride=((rank, hid * rank, bh * rank), 1, 0)),
        )
        # normed [m, C*H] is staged through shared memory by TMA in [128, 32] boxes;
        # mixed [m, H] is written in the accumulator's N order, its stream mode
        # with stride 0 (every stream maps onto the one output column).
        nrm = cute.make_tensor(normed_ptr, cute.make_layout((m, width), stride=(width, 1)))
        mix = cute.make_tensor(
            mixed.iterator, cute.make_layout((m, (bh, hc, nt), 1), stride=(hid, (1, 0, bh), 0))
        )
        # NVFP4 outputs (`fp4` only): 8 e2m1 codes per Int32, one e4m3 byte per 16.
        xq = cute.make_tensor(xq_ptr, cute.make_layout((m, hid // 8), stride=(hid // 8, 1)))
        xsf = cute.make_tensor(sf_ptr, cute.make_layout((m, hid // 16), stride=(hid // 16, 1)))
        gsc = cute.make_tensor(gs_ptr, cute.make_layout(1))
        k_major = utils.LayoutEnum.ROW_MAJOR.mma_major_mode()
        tiled_mma = self._setup(k_major, k_major)

        a_op = utils.sm100.cluster_shape_to_tma_atom_A(self.cluster_shape_mn, tiled_mma.thr_id)
        a_smem = cute.slice_(self.a_smem_layout_staged, (None, None, None, 0))
        tma_atom_a, tma_a = cute.nvgpu.make_tiled_tma_atom_A(
            a_op, a, a_smem, self.mma_tiler, tiled_mma, self.cluster_layout_vmnk.shape
        )
        b_op = utils.sm100.cluster_shape_to_tma_atom_B(self.cluster_shape_mn, tiled_mma.thr_id)
        b_smem = cute.slice_(self.b_smem_layout_staged, (None, None, None, 0))
        tma_atom_b, tma_b = cute.nvgpu.make_tiled_tma_atom_B(
            b_op, b, b_smem, self.mma_tiler, tiled_mma, self.cluster_layout_vmnk.shape
        )
        tma_atom_n, tma_n = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            nrm,
            cute.select(self.n_smem_layout_staged, mode=[0, 1]),
            self.n_box,
        )
        self.num_tma_load_bytes = cute.size_in_bytes(self.io_dtype, a_smem) + cute.size_in_bytes(
            self.io_dtype, b_smem
        )
        # Rastered along N: consecutive CTAs take consecutive hidden slices of the
        # same rows, so the normed reads in flight cover whole rows rather than
        # one 64-byte column block of many rows (~2% faster at 8K-16K rows).
        sched = utils.PersistentTileSchedulerParams(
            ((m + 127) // 128, nt, 1), (*self.cluster_shape_mn, 1), 1, False
        )
        grid = utils.StaticPersistentTileScheduler.get_grid_shape(sched, max_active_clusters)
        self.kernel(
            tiled_mma,
            tma_atom_a,
            tma_a,
            tma_atom_b,
            tma_b,
            tma_atom_n,
            tma_n,
            mix,
            xq,
            xsf,
            gsc,
            m,
            self.cluster_layout_vmnk,
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            self.n_smem_layout_staged,
            self.epi_tile,
            sched,
        ).launch(
            grid=grid,
            block=[self.threads_per_cta, 1, 1],
            cluster=(*self.cluster_shape_mn, 1),
            stream=stream,
            use_pdl=_PDL,
        )

    @cute.kernel
    def kernel(
        self,
        tiled_mma: cute.TiledMma,
        tma_atom_a: cute.CopyAtom,
        mA_mkl: cute.Tensor,
        tma_atom_b: cute.CopyAtom,
        mB_nkl: cute.Tensor,
        tma_atom_n: cute.CopyAtom,
        mN_mn: cute.Tensor,
        mX_mnl: cute.Tensor,
        mQ: cute.Tensor,
        mSF: cute.Tensor,
        mGS: cute.Tensor,
        m: cutlass.Int32,
        cluster_layout_vmnk: cute.Layout,
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        n_smem_layout_staged: cute.ComposedLayout,
        epi_tile: cute.Tile,
        sched_params: utils.PersistentTileSchedulerParams,
    ):
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        if warp_idx == self.tma_warp_id:
            cpasync.prefetch_descriptor(tma_atom_a)
            cpasync.prefetch_descriptor(tma_atom_b)
        if warp_idx == self.nrm_warp_id:
            cpasync.prefetch_descriptor(tma_atom_n)
        tidx, _, _ = cute.arch.thread_idx()

        @cute.struct
        class SharedStorage:
            ab_full_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_ab_stage * 2]
            acc_full_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_acc_stage * 2]
            n_full_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_n_stage * 2]
            tmem_dealloc_mbar: cutlass.Int64
            tmem_holding_buf: cutlass.Int32

        smem = utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)
        ab_producer, ab_consumer = PipelineTmaUmma.create(
            barrier_storage=storage.ab_full_mbar_ptr.data_ptr(),
            num_stages=self.num_ab_stage,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
            tx_count=self.num_tma_load_bytes,
            cta_layout_vmnk=cluster_layout_vmnk,
        ).make_participants()
        acc_pipeline = PipelineUmmaAsync.create(
            barrier_storage=storage.acc_full_mbar_ptr.data_ptr(),
            num_stages=self.num_acc_stage,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread, len(self.epilogue_warp_id)
            ),
            cta_layout_vmnk=cluster_layout_vmnk,
        )
        n_producer, n_consumer = pipeline.PipelineTmaAsync.create(
            barrier_storage=storage.n_full_mbar_ptr.data_ptr(),
            num_stages=self.num_n_stage,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread, len(self.epilogue_warp_id)
            ),
            tx_count=self.num_n_bytes,
            cta_layout_vmnk=cluster_layout_vmnk,
        ).make_participants()
        tmem_alloc_barrier = pipeline.NamedBarrier(
            barrier_id=self.tmem_alloc_sync_bar_id,
            num_threads=32 * len((self.mma_warp_id, *self.epilogue_warp_id)),
        )
        tmem = utils.TmemAllocator(
            storage.tmem_holding_buf.ptr,
            barrier_for_retrieve=tmem_alloc_barrier,
            allocator_warp_id=self.epilogue_warp_id[0],
            is_two_cta=False,
            two_cta_tmem_dealloc_mbar_ptr=storage.tmem_dealloc_mbar.ptr,
        )
        pipeline_init_arrive(cluster_shape_mn=self.cluster_shape_mn, is_relaxed=True)

        sA = smem.allocate_tensor(
            element_type=self.io_dtype,
            layout=a_smem_layout_staged.outer,
            byte_alignment=128,
            swizzle=a_smem_layout_staged.inner,
        )
        sB = smem.allocate_tensor(
            element_type=self.io_dtype,
            layout=b_smem_layout_staged.outer,
            byte_alignment=128,
            swizzle=b_smem_layout_staged.inner,
        )
        sN = smem.allocate_tensor(
            element_type=self.io_dtype,
            layout=n_smem_layout_staged.outer,
            byte_alignment=128,
            swizzle=n_smem_layout_staged.inner,
        )

        gA_mkl = cute.local_tile(
            mA_mkl, cute.slice_(self.mma_tiler, (None, 0, None)), (None, None, None)
        )
        gB_nkl = cute.local_tile(
            mB_nkl, cute.slice_(self.mma_tiler, (0, None, None)), (None, None, None)
        )
        tile_mn = cute.slice_(self.mma_tiler, (None, None, 0))
        gX_mnl = cute.local_tile(mX_mnl, tile_mn, (None, None, None))
        cX_mnl = cute.local_tile(
            cute.make_identity_tensor(mX_mnl.shape), tile_mn, (None, None, None)
        )
        # normed boxes: [128 rows, 32 columns] at (row tile, column block)
        gN_box = cute.local_tile(mN_mn, self.n_box, (None, None))
        k_tile_cnt = cute.size(gA_mkl, mode=[3])

        thr_mma = tiled_mma.get_slice(0)
        tCgA = thr_mma.partition_A(gA_mkl)
        tCgB = thr_mma.partition_B(gB_nkl)
        tCgX = thr_mma.partition_C(gX_mnl)
        tCcX = thr_mma.partition_C(cX_mnl)

        tAsA, tAgA = cpasync.tma_partition(
            tma_atom_a,
            0,
            cute.make_layout(1),
            cute.group_modes(sA, 0, 3),
            cute.group_modes(tCgA, 0, 3),
        )
        tBsB, tBgB = cpasync.tma_partition(
            tma_atom_b,
            0,
            cute.make_layout(1),
            cute.group_modes(sB, 0, 3),
            cute.group_modes(tCgB, 0, 3),
        )
        tNsN, tNgN = cpasync.tma_partition(
            tma_atom_n,
            0,
            cute.make_layout(1),
            cute.group_modes(sN, 0, 2),
            cute.group_modes(gN_box, 0, 2),
        )
        tCrA = tiled_mma.make_fragment_A(sA)
        tCrB = tiled_mma.make_fragment_B(sB)
        acc_shape = tiled_mma.partition_shape_C(self.mma_tiler[:2])
        tCtAcc_fake = tiled_mma.make_fragment_C(cute.append(acc_shape, self.num_acc_stage))

        pipeline_init_wait(cluster_shape_mn=self.cluster_shape_mn)
        tile_sched = utils.StaticPersistentTileScheduler.create(
            sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
        )
        work_tile = tile_sched.initial_work_tile_info()

        if warp_idx == self.tma_warp_id:
            # Programmatic dependent launch: the first tile's up-weight stages (static)
            # stream before the grid-dependency wait, `low` -- the previous kernel's
            # output -- after it. Every CTA owns at least one tile (grid <= tiles).
            crd = work_tile.tile_idx
            tAgA_slice = tAgA[(None, crd[0], None, crd[2])]
            tBgB_slice = tBgB[(None, crd[1], None, crd[2])]
            ab_producer.reset()
            pre = []
            for kk in cutlass.range_constexpr(self.n_pre):
                handle = ab_producer.acquire_and_advance()
                cute.copy(
                    tma_atom_b,
                    tBgB_slice[(None, kk)],
                    tBsB[(None, handle.index)],
                    tma_bar_ptr=handle.barrier,
                )
                pre.append(handle)
            cute.arch.griddepcontrol_wait()
            for kk in cutlass.range_constexpr(self.n_pre):
                handle = pre[kk]
                cute.copy(
                    tma_atom_a,
                    tAgA_slice[(None, kk)],
                    tAsA[(None, handle.index)],
                    tma_bar_ptr=handle.barrier,
                )
            for k_tile in cutlass.range(self.n_pre, k_tile_cnt, 1, unroll=1):
                handle = ab_producer.acquire_and_advance()
                cute.copy(
                    tma_atom_a,
                    tAgA_slice[(None, k_tile)],
                    tAsA[(None, handle.index)],
                    tma_bar_ptr=handle.barrier,
                )
                cute.copy(
                    tma_atom_b,
                    tBgB_slice[(None, k_tile)],
                    tBsB[(None, handle.index)],
                    tma_bar_ptr=handle.barrier,
                )
            tile_sched.advance_to_next_work()
            work_tile = tile_sched.get_current_work()
            while work_tile.is_valid_tile:
                crd = work_tile.tile_idx
                tAgA_slice = tAgA[(None, crd[0], None, crd[2])]
                tBgB_slice = tBgB[(None, crd[1], None, crd[2])]
                ab_producer.reset()
                peek = ab_producer.try_acquire()
                for k_tile in cutlass.range(0, k_tile_cnt, 1, unroll=1):
                    handle = ab_producer.acquire_and_advance(peek)
                    cute.copy(
                        tma_atom_a,
                        tAgA_slice[(None, handle.count)],
                        tAsA[(None, handle.index)],
                        tma_bar_ptr=handle.barrier,
                    )
                    cute.copy(
                        tma_atom_b,
                        tBgB_slice[(None, handle.count)],
                        tBsB[(None, handle.index)],
                        tma_bar_ptr=handle.barrier,
                    )
                    peek = cutlass.Boolean(1)
                    if handle.count + 1 < k_tile_cnt:
                        peek = ab_producer.try_acquire()
                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()
            ab_producer.tail()

        if warp_idx == self.nrm_warp_id:
            cute.arch.griddepcontrol_wait()
            halves = self.bh // self.epi_cols
            col_blocks = self.hidden // self.epi_cols
            while work_tile.is_valid_tile:
                crd = work_tile.tile_idx
                for half in cutlass.range_constexpr(halves):
                    handle = n_producer.acquire_and_advance()
                    for s in cutlass.range_constexpr(self.hc):
                        cute.copy(
                            tma_atom_n,
                            tNgN[(None, crd[0], s * col_blocks + crd[1] * halves + half)],
                            tNsN[(None, handle.index * self.hc + s)],
                            tma_bar_ptr=handle.barrier,
                        )
                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()
            n_producer.tail()

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
                peek = ab_consumer.try_wait()
                acc_pipeline.producer_acquire(acc_producer_state)
                tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
                for k_tile in cutlass.range(0, k_tile_cnt, 1, unroll=1):
                    handle = ab_consumer.wait_and_advance(peek)
                    num_kblocks = cute.size(tCrA, mode=[2])
                    for kb in cutlass.range(num_kblocks, unroll_full=True):
                        kcrd = (None, None, kb, handle.index)
                        cute.gemm(tiled_mma, tCtAcc, tCrA[kcrd], tCrB[kcrd], tCtAcc)
                        tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                    handle.release()
                    peek = cutlass.Boolean(1)
                    if handle.count + 1 < k_tile_cnt:
                        peek = ab_consumer.try_wait()
                acc_pipeline.producer_commit(acc_producer_state)
                acc_producer_state.advance()
                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()
            acc_pipeline.producer_tail(acc_producer_state)

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
                utils.LayoutEnum.ROW_MAJOR,
                self.io_dtype,
                self.acc_dtype,
                epi_tile,
                False,
            )
            tAcc_epi = cute.flat_divide(tCtAcc_base[((None, None), 0, 0, None)], epi_tile)
            tiled_copy_t2r = tcgen05.make_tmem_copy(copy_atom_t2r, tAcc_epi[(None, None, 0, 0, 0)])
            thr_copy_t2r = tiled_copy_t2r.get_slice(tidx)
            tTR_tAcc_base = thr_copy_t2r.partition_S(tAcc_epi)
            tTR_sN = thr_copy_t2r.partition_D(sN)
            tTR_gX = thr_copy_t2r.partition_D(
                cute.flat_divide(tCgX[((None, None), 0, 0, None, None, None)], epi_tile)
            )
            tTR_cX = thr_copy_t2r.partition_D(
                cute.flat_divide(tCcX[((None, None), 0, 0, None, None, None)], epi_tile)
            )
            frag_shape = tTR_gX[(None, None, None, 0, 0, 0, 0, 0)].shape
            tTR_rAcc = cute.make_rmem_tensor(frag_shape, self.acc_dtype)
            tTR_rX = cute.make_rmem_tensor(frag_shape, self.io_dtype)
            tTR_rSum = cute.make_rmem_tensor(frag_shape, self.acc_dtype)
            nfrag = cute.size(tTR_rAcc)
            halves = self.bh // self.epi_cols
            rN = cute.make_rmem_tensor(frag_shape, self.io_dtype)
            inv_hc = cutlass.Float32(1.0 / self.hc)
            if cutlass.const_expr(self.fp4):
                # A thread owns one row's `epi_cols` consecutive columns (the
                # 32dp32b TMEM load), i.e. whole 16-element scale blocks.
                gs = mGS[0]
                rcp_gs = _rcp_approx_ftz(gs)
                rcp6 = _rcp_approx_ftz(cutlass.Float32(6.0))
                rV = cute.make_rmem_tensor((nfrag,), self.acc_dtype)
                assert nfrag == 32, "the NVFP4 epilogue stores two scale blocks per row"
                q_base = mQ.iterator.toint()
                sf_base = mSF.iterator.toint()

            while work_tile.is_valid_tile:
                crd = work_tile.tile_idx
                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()

                tTR_tAcc = tTR_tAcc_base[(None, None, None, None, None, acc_consumer_state.index)]
                tTR_tAcc = cute.group_modes(tTR_tAcc, 3, cute.rank(tTR_tAcc))
                gX_t = tTR_gX[(None, None, None, None, None, *crd)]
                gX_t = cute.group_modes(gX_t, 3, cute.rank(gX_t))
                cX_t = tTR_cX[(None, None, None, None, None, *crd)]
                cX_t = cute.group_modes(cX_t, 3, cute.rank(cX_t))
                row_ok = cX_t[(0, 0, 0, 0)][0] < m

                acc_pipeline.consumer_wait(acc_consumer_state)
                for half in cutlass.range_constexpr(halves):
                    n_handle = n_consumer.wait_and_advance()
                    for s in cutlass.range_constexpr(self.hc):
                        cute.autovec_copy(
                            tTR_sN[(None, None, None, n_handle.index * self.hc + s)], rN
                        )
                        cute.copy(
                            tiled_copy_t2r,
                            tTR_tAcc[(None, None, None, s * halves + half)],
                            tTR_rAcc,
                        )
                        for i in cutlass.range_constexpr(nfrag):
                            gate = _sigmoid(tTR_rAcc[i])
                            if cutlass.const_expr(s == 0):
                                tTR_rSum[i] = gate * rN[i].to(self.acc_dtype)
                            else:
                                tTR_rSum[i] = tTR_rSum[i] + gate * rN[i].to(self.acc_dtype)
                    cute.arch.sync_warp()
                    n_handle.release()
                    for i in cutlass.range_constexpr(nfrag):
                        tTR_rX[i] = (tTR_rSum[i] * inv_hc).to(self.io_dtype)
                    if row_ok:
                        cute.autovec_copy(tTR_rX, gX_t[(None, None, None, half)])
                        if cutlass.const_expr(self.fp4):
                            row = cX_t[(0, 0, 0, 0)][0]
                            blk0 = (crd[1] * self.bh + half * self.epi_cols) // 16
                            for i in cutlass.range_constexpr(nfrag):
                                rV[i] = tTR_rX[i].to(self.acc_dtype)
                            words, sfs = [], []
                            for b in cutlass.range_constexpr(2):
                                v = [rV[16 * b + j] for j in range(16)]
                                amax = _absmax16(v)
                                sf = _e4m3_scale(amax, gs, rcp6)
                                osc = _e2m1_out_scale(sf, rcp_gs, amax)
                                words += [_e2m1x8(v[:8], osc), _e2m1x8(v[8:], osc)]
                                sfs.append(sf)
                            row64 = cutlass.Int64(row)
                            _st_global_v4(
                                q_base + row64 * (self.hidden // 2) + cutlass.Int64(blk0) * 8,
                                *words,
                            )
                            _st_global_b16(
                                sf_base + row64 * (self.hidden // 16) + cutlass.Int64(blk0),
                                sfs[0] | (sfs[1] << 8),
                            )
                with cute.arch.elect_one():
                    acc_pipeline.consumer_release(acc_consumer_state)
                acc_consumer_state.advance()

            cute.arch.barrier(
                barrier_id=self.epilog_sync_bar_id,
                number_of_threads=32 * len(self.epilogue_warp_id),
            )
            tmem.relinquish_alloc_permit()
            tmem.free(tmem_ptr)


_COMPILED: dict = {}


def _launcher(block_h: int, hc: int, hidden: int, rank: int, ptrs, fp4: bool = False):
    """Compile once per static configuration; later calls reuse the TVM-FFI entry,
    which launches on the caller's current stream (graph-capturable). `ptrs` is
    `(low, up, normed, mixed tensor, xq, sf, gs)`."""
    key = (block_h, hc, hidden, rank, fp4)
    fn = _COMPILED.get(key)
    if fn is None:
        kern = HcUpGateMeanKernel(block_h, hc, hidden, rank, fp4=fp4)
        cptrs = [
            make_ptr(cutlass.BFloat16, p, cute.AddressSpace.gmem, assumed_align=16)
            for p in ptrs[:3]
        ]
        out = from_dlpack(ptrs[3], assumed_align=16).mark_layout_dynamic(leading_dim=1)
        qptrs = [
            make_ptr(cutlass.Int32, ptrs[4], cute.AddressSpace.gmem, assumed_align=16),
            make_ptr(cutlass.Uint8, ptrs[5], cute.AddressSpace.gmem, assumed_align=2),
            make_ptr(cutlass.Float32, ptrs[6], cute.AddressSpace.gmem, assumed_align=4),
        ]
        fn = cute.compile(
            kern,
            128,
            *cptrs,
            out,
            *qptrs,
            max_active_clusters=utils.HardwareInfo().get_max_active_clusters(1),
            stream=cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
            options="--opt-level 2 --enable-tvm-ffi",
        )
        _COMPILED[key] = fn
    return fn


def hc_up_gate_mean(
    low: torch.Tensor, up: torch.Tensor, normed: torch.Tensor, hc_count: int, block_h: int
) -> torch.Tensor:
    """Launch the fused kernel; `up` is the `[R, C*H]` transposed view of the
    `[C*H, R]` up weight, as `cublas_mm(low, up)` takes it."""
    rows, rank = low.shape
    hidden = normed.shape[1] // hc_count
    mixed = torch.empty((rows, hidden), dtype=normed.dtype, device=normed.device)
    if rows > 0:
        ptrs = (low.data_ptr(), up.data_ptr(), normed.data_ptr())
        # The NVFP4 pointers are never read without `fp4`.
        unused = (mixed.data_ptr(),) * 3
        _launcher(block_h, hc_count, hidden, rank, (*ptrs, mixed, *unused))(
            rows, *ptrs, mixed, *unused
        )
    return mixed


def hc_up_gate_mean_fp4(
    low: torch.Tensor,
    up: torch.Tensor,
    normed: torch.Tensor,
    hc_count: int,
    global_scale: torch.Tensor,
    block_h: int,
):
    """`hc_up_gate_mean` plus the NVFP4 of `mixed` in the same epilogue.
    Returns `(mixed [T, H], xq [T, H/2] uint8, xsf [T*H/16] uint8)`."""
    rows, rank = low.shape
    hidden = normed.shape[1] // hc_count
    dev = normed.device
    mixed = torch.empty((rows, hidden), dtype=normed.dtype, device=dev)
    xq = torch.empty((rows, hidden // 2), dtype=torch.uint8, device=dev)
    xsf = torch.empty((rows * (hidden // 16),), dtype=torch.uint8, device=dev)
    if rows > 0:
        ptrs = (low.data_ptr(), up.data_ptr(), normed.data_ptr())
        qptrs = (xq.data_ptr(), xsf.data_ptr(), global_scale.data_ptr())
        _launcher(block_h, hc_count, hidden, rank, (*ptrs, mixed, *qptrs), fp4=True)(
            rows, *ptrs, mixed, *qptrs
        )
    return mixed, xq, xsf


# Hidden columns per tile. 32 (MMA N = 4 * 32 = 128, five A/B stages) measured
# equal to 64 (N = 256, three stages) at 8K-16K rows and faster at 32..2048 rows.
_BLOCK_H = 32


class Qwen4ExpHcUpGateMean(OpWrapper):
    """With `C = hc_count`, `H = width // C`, `low` the `[T, R]` SiLU bottleneck
    and `up` the `[R, C*H]` transposed view of the `[C*H, R]` up weight::

        mixed[t, i] = mean over s of sigmoid((low @ up)[t, s*H + i]) * normed[t, s*H + i]

    The GEMM accumulates in fp32 in tensor memory and is never rounded or
    written; sigmoid, product and stream mean are fp32 with one rounding to
    `normed`'s dtype. Returns a fresh `[T, H]` tensor. Fusion boundary: the up
    projection, the sigmoid gate and the stream mean (two launches as `cublas_mm(low,
    up)` followed by a separate gate-and-mean pass).
    """

    ARCHS = frozenset({Arch.SM_103})

    # spec: rows, hc_count (4), hidden (2560), rank (320); bf16 only.
    CELLS: tuple[Cell, ...] = (
        tuple(
            Cell(why=f"checkpoint 4x2560, rank 320, {why}", spec=dict(rows=t))
            for t, why in (
                (128, "decode 128 rows, one full tile"),
                (16384, "prefill chunk 16384 (max_num_tokens)"),
            )
        )
        + (Cell(why="ragged tail, 16320 + 70 rows (mixed step)", spec=dict(rows=16390)),)
        # The tp4 / tep4 / dep4 target: the mixer runs on the full 4 x 2560
        # residual on every rank, so only the row count moves per member.
        + tuple(
            Cell(why=why, spec=dict(rows=t))
            for t, why in (
                (1, "1 row: one decode token, the smallest captured batch; 1 of a tile's 128 rows"),
                (32, "32 rows: dep4's per-rank decode cap (128 / 4), a quarter of one tile"),
                (1024, "1024 rows: mtp7 verify at tep4, 128 requests x 8 rows"),
                (8192, "8192 rows: one prefill chunk at max_num_tokens, the envelope's largest"),
            )
        )
    )

    note = """
    Source: the tp1 target's CuTe DSL kernel, reused in tree (no trtllm op or
    FlashInfer call fuses this multi-input epilogue; cuBLAS writes the logits).
    It is JIT-compiled by `cute.compile` on the first call per
    `(block_h, C, H, R)` for the running device: sm_103 (GB300) in the cells,
    with no `CUTE_DSL_ARCH` override. The cells assert that the compiled entry
    answered. A persistent tcgen05 GEMM whose epilogue applies the gate, so the
    `[T, C*H]` gate logits never reach HBM. Metadata consumed: none. Call sites:
    the hyper-connection mixers at decode, MTP verify (4 rows per request) and
    prefill. Under tp4, tep4 and dep4 every rank mixes the full 4 x 2560
    residual, so only the row count moves per member. Preconditions (bf16):
    `low` `[T, R]` contiguous, `up` the transposed view of a contiguous
    `[C*H, R]` weight, `normed` `[T, C*H]` contiguous, 16-byte aligned
    pointers, `R % 64 == 0`, `H % 32 == 0`, `C * 32 <= 256`. Compiles once per
    `(block_h, C, H, R)` on first call (~1 s); the row count is a runtime
    argument. Graph-capturable (launches on the current stream, no host sync).
    Launched with programmatic dependent launch (TRTLLM_ENABLE_PDL=0 turns it
    off): each CTA's first tile streams its up-weight stages before
    `griddepcontrol.wait` and `low` / `normed` after it, so behind a producer
    that releases dependents early (`qwen4_exp_hc_decode_tc_first`) the weight
    read overlaps the producer.

    Certified: bf16 at the checkpoint's 4 x 2560, rank 320, at 1 and 32 rows
    (part of one tile), 128 (one full tile), 1024, 8192, 16384 and 16390 rows
    (a partial last tile); graph replay.

    Measured on GB300 against the float64 reference. The band is `rtol = eps`,
    `atol = eps * max|ref|`, with `eps` the bf16 epsilon:
    - worst element: 0.0035 (1 row) to 0.40 (8192 rows) of the band;
    - relative L2: 2.9e-6 to 8.1e-5;
    - cosine: >= 0.999999997;
    - 99.9th-percentile error: <= 3.5e-6 of max|ref|.
    Two wrong mixes land 49.6x to 93.9x the band: the gates rolled by one
    stream, and the last stream dropped. The band does not separate a
    bf16-rounded gate logit (0.28x to 0.46x), which is a legitimate
    two-launch result. Time per call, eager: 11 us at 1-1024 rows, 55 us at
    8192, 106 us at 16384.

    Not certified:
    - rows that no cell drives: zero, 2-31, 33-127 (the same one-tile path
      as 1 and 32), between the full-tile cells, and above 16390;
    - hc_count other than 4, hidden other than 2560, rank other than 320 (the
      kernel compiles for them);
    - `hc_up_gate_mean_fp4`, which this module also defines: no cell here
      drives it.
    """

    def __call__(
        self, low: torch.Tensor, up: torch.Tensor, normed: torch.Tensor, hc_count: int
    ) -> torch.Tensor:
        return hc_up_gate_mean(low, up, normed, hc_count, _BLOCK_H)

    def reference(
        self, low: torch.Tensor, up: torch.Tensor, normed: torch.Tensor, hc_count: int
    ) -> torch.Tensor:
        """float64, one rounding."""
        hidden = normed.shape[1] // hc_count
        gate = torch.sigmoid(F.linear(low.double(), up.double().t())).unflatten(
            -1, (hc_count, hidden)
        )
        x = normed.double().unflatten(-1, (hc_count, hidden))
        return (gate * x).mean(dim=-2).to(normed.dtype)

    def is_valid(
        self, low: torch.Tensor, up: torch.Tensor, normed: torch.Tensor, hc_count: int
    ) -> None:
        rows, rank = low.shape
        width = normed.shape[1]
        assert low.dtype == up.dtype == normed.dtype == torch.bfloat16, "bf16 operands"
        assert normed.shape[0] == rows and up.shape == (rank, width), "shapes"
        assert low.stride() == (rank, 1) and normed.stride() == (width, 1), "contiguous rows"
        assert up.stride() == (1, rank), "up must be the transposed view of a [C*H, R] weight"
        assert width % hc_count == 0 and (width // hc_count) % 32 == 0 and hc_count * 32 <= 256
        assert rank % 64 == 0, "rank must be a multiple of 64"
        for t in (low, up, normed):
            assert t.data_ptr() % 16 == 0, "16-byte aligned operands"


qwen4_exp_hc_up_gate_mean = Qwen4ExpHcUpGateMean()
