# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Hyper-connection mixer second half on the un-normalized residual: the up
projection as a 2-CTA tcgen05 GEMM whose epilogue gates `hc_out * rstd * (1 + w)`
-- the site's normed stream, formed in registers -- and averages the streams
(optionally also writing the routed experts' NVFP4 input).

Tier 6 (CuTe DSL written for this target). `qwen4_exp_hc_up_gate_mean`'s epilogue
on a CTA pair: `tcgen05.mma.cta_group::2` with M = 256 (each CTA 128 rows of
`low`, resident in shared memory for its row tile) and N = 128 (C streams x 32
hidden columns), each CTA loading half of the up weight's N per k block. Tiles
are walked row-tile-major over a contiguous range per pair (one `low` load per
row tile); the MMA runs up to four tiles ahead of the epilogue (four accumulator
stages).

The epilogue is two sets of eight warps that take alternate tiles. A load warp
brings each (tile, stream)'s `hc_out` box (TMA) and `(1 + w)` slice (bulk copy) to
its own shared-memory stage; a set hands a stream's stage back as soon as it has
consumed it, so the next tile's boxes load under the current tile's math. Within a
set, warp w and w + 4 read the same 128 accumulator rows and split the tile's 32
columns; the gate and the stream mean run on packed fp32 pairs (`f32x2`), and each
thread stores its 16 outputs as one 32-byte sector.
"""

from __future__ import annotations

from typing import Optional, Tuple

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
from tensorrt_llm._torch._experimental.modeling_v2.catalog.hc.qwen4_exp_hc_inject_norm import (
    _bf16x2,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.hc.qwen4_exp_hc_up_gate_mean import (
    _asm,
    _rcp_approx_ftz,
    _st_global_b16,
    _st_global_v4,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.quantization.fp4_quantize import (
    fp4_quantize,
)
from tensorrt_llm._torch.flashinfer_utils import get_env_enable_pdl
from tensorrt_llm._torch.kernels.blackwell.custom_pipeline import PipelineTmaUmma, PipelineUmmaAsync
from tensorrt_llm._torch.kernels.blackwell.utils import make_ptr

from .qwen4_exp_hc_first_rescale_tc import _lds_v4, _sts_v4

#: Programmatic dependent launch (sm_90+), off with TRTLLM_ENABLE_PDL=0.
_PDL = get_env_enable_pdl()

#: Hidden columns a tile (MMA N = C * 32 = 128) and a thread's epilogue columns.
_BLOCK_H = 32
_SUB = 16
#: Epilogue warp sets (alternate tiles) and warps per set.
_SETS = 2
_SET_WARPS = 8
#: Up-weight ring, accumulator stages, and the hc_out / (1 + w) ring (one stage per
#: (tile, stream)).
_B_STAGES = 8
_ACC_STAGES = 4
_N_STAGES = 8
#: NVFP4 block: one e4m3 scale per this many contiguous elements.
SF_VEC = 16


@dsl_user_op
def _ldg_v4_f32(addr, *, loc=None, ip=None):
    """Four fp32 values at a 16-byte aligned global address."""
    ty = T.f32()
    res = _asm(
        llvm.StructType.get_literal([ty] * 4),
        [cutlass.Int64(addr)],
        "ld.global.v4.f32 {$0, $1, $2, $3}, [$4];",
        "=f,=f,=f,=f,l",
        loc,
        ip,
    )
    return [cutlass.Float32(llvm.extractvalue(ty, res, [i], loc=loc, ip=ip)) for i in range(4)]


@dsl_user_op
def _lds_v4_f32(addr, *, loc=None, ip=None):
    """Four fp32 values at a 16-byte aligned shared address (side-effecting: ordered
    after the stage wait).
    """
    ty = T.f32()
    res = _asm(
        llvm.StructType.get_literal([ty] * 4),
        [cutlass.Int32(addr)],
        "ld.shared.v4.f32 {$0, $1, $2, $3}, [$4];",
        "=f,=f,=f,=f,r",
        loc,
        ip,
        side_effects=True,
    )
    return [cutlass.Float32(llvm.extractvalue(ty, res, [i], loc=loc, ip=ip)) for i in range(4)]


@dsl_user_op
def _bulk_g2s(dst, src, nbytes: int, mbar, *, loc=None, ip=None):
    """One non-tensor bulk copy global -> this CTA's shared memory, completing on
    `mbar`.
    """
    _asm(
        T.i32(),
        [cutlass.Int32(dst), cutlass.Int64(src), cutlass.Int32(mbar)],
        "{ cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [$1], [$2], "
        + str(nbytes)
        + ", [$3]; mov.u32 $0, 0; }",
        "=r,r,l,r",
        loc,
        ip,
        side_effects=True,
    )


@dsl_user_op
def _st_global_v8(addr, w, *, loc=None, ip=None):
    """One 32-byte `st.global.v8.b32` (sm_100+): a thread's 16 bf16 outputs, one
    full sector.
    """
    _asm(
        T.i32(),
        [cutlass.Int64(addr)] + [cutlass.Int32(x) for x in w],
        "{ st.global.v8.b32 [$1], {$2, $3, $4, $5, $6, $7, $8, $9}; mov.u32 $0, 0; }",
        "=r,l" + ",r" * 8,
        loc,
        ip,
        side_effects=True,
    )


@dsl_user_op
def _gate2(z0, z1, xw, wa, wb, hr, a0, a1, first: bool, *, loc=None, ip=None):
    """One column pair of one stream on the packed fp32 pipe (`f32x2`: each lane is
    the scalar `.rn` operation): `t = tanh.approx(z * 0.5)`, `g = fma(t, hr, hr)`,
    `xw = x * w1`, `acc = fma(g, xw, acc)` (`g * xw` for the first stream). `xw` is
    the stream's bf16x2 word (element 2j in the low half), widened exactly by bit
    shifts.
    """
    tail = "mul.rn.f32x2 a, t, x; " if first else "mov.b64 a, {$8, $9}; fma.rn.f32x2 a, t, x, a; "
    text = (
        "{ .reg .b64 z, h, t, x, w, a; .reg .f32 c, t0, t1; .reg .b32 xl, xh; "
        "mov.f32 c, 0f3F000000; mov.b64 h, {c, c}; "
        "mov.b64 z, {$2, $3}; mul.rn.f32x2 z, z, h; "
        "mov.b64 {t0, t1}, z; tanh.approx.f32 t0, t0; tanh.approx.f32 t1, t1; "
        "mov.b64 t, {t0, t1}; mov.b64 h, {$7, $7}; fma.rn.f32x2 t, t, h, h; "
        "shl.b32 xl, $4, 16; and.b32 xh, $4, 0xFFFF0000; mov.b64 x, {xl, xh}; "
        "mov.b64 w, {$5, $6}; mul.rn.f32x2 x, x, w; " + tail + "mov.b64 {$0, $1}, a; }"
    )
    ty = T.f32()
    res = _asm(
        llvm.StructType.get_literal([ty, ty]),
        [
            cutlass.Float32(z0),
            cutlass.Float32(z1),
            cutlass.Int32(xw),
            cutlass.Float32(wa),
            cutlass.Float32(wb),
            cutlass.Float32(hr),
            cutlass.Float32(a0),
            cutlass.Float32(a1),
        ],
        text,
        "=f,=f,f,f,r,f,f,f,f,f",
        loc,
        ip,
    )
    return [cutlass.Float32(llvm.extractvalue(ty, res, [i], loc=loc, ip=ip)) for i in range(2)]


@dsl_user_op
def _nvfp4_16(words, gs, rcp_gs, rcp6, *, loc=None, ip=None):
    """NVFP4 of one 16-element block given as 8 bf16x2 words (element 2k in word k's
    low half): `(e2m1 word for elements 0..7, e2m1 word for 8..15, e4m3 scale byte)`.
    `qwen4_exp_hc_up_gate_mean`'s recipe (`_absmax16`, `_e4m3_scale`,
    `_e2m1_out_scale`, `_e2m1x8`) on the same bf16-rounded values, with the max taken
    on the packed bf16 pairs, the bf16 -> fp32 unpack done by bit shifts (both exact)
    and the scaling multiplies on `f32x2` pairs (each lane `mul.rn.f32`): bitwise its
    outputs.
    """
    amax = "abs.bf16x2 m, $3; " + "".join(
        f"abs.bf16x2 t, ${4 + k}; max.bf16x2 m, m, t; " for k in range(7)
    )
    unpack = "".join(
        f"shl.b32 a{2 * k}, ${3 + k}, 16; and.b32 a{2 * k + 1}, ${3 + k}, 0xFFFF0000; "
        for k in range(8)
    )
    muls = "mov.b64 oo, {o, o}; " + "".join(
        f"mov.b64 pp, {{a{2 * k}, a{2 * k + 1}}}; mul.rn.f32x2 pp, pp, oo; "
        f"mov.b64 {{a{2 * k}, a{2 * k + 1}}}, pp; "
        for k in range(8)
    )
    cvts = "".join(f"cvt.rn.satfinite.e2m1x2.f32 b{k}, a{2 * k + 1}, a{2 * k}; " for k in range(8))
    text = (
        "{ .reg .b32 m, t, f2; .reg .b32 a<16>; .reg .f32 x, y, mx, o; .reg .b16 h, l16, h16; "
        ".reg .pred p; .reg .b8 b<8>; .reg .b64 oo, pp; "
        + amax
        + "mov.b32 {l16, h16}, m; cvt.f32.bf16 x, l16; cvt.f32.bf16 y, h16; max.f32 mx, x, y; "
        "mul.rn.f32 x, mx, $13; mul.rn.f32 x, $11, x; "
        "cvt.rn.satfinite.e4m3x2.f32 h, 0f00000000, x; cvt.u32.u16 $2, h; "
        "cvt.rn.f16x2.e4m3x2 f2, h; mov.b32 {l16, h16}, f2; cvt.f32.f16 y, l16; "
        "mul.rn.f32 y, y, $12; rcp.approx.ftz.f32 y, y; setp.neu.f32 p, mx, 0f00000000; "
        "selp.f32 o, y, 0f00000000, p; "
        + unpack
        + muls
        + cvts
        + "mov.b32 $0, {b0, b1, b2, b3}; mov.b32 $1, {b4, b5, b6, b7}; }"
    )
    ty = T.i32()
    res = _asm(
        llvm.StructType.get_literal([ty, ty, ty]),
        [cutlass.Int32(w) for w in words]
        + [cutlass.Float32(gs), cutlass.Float32(rcp_gs), cutlass.Float32(rcp6)],
        text,
        "=r,=r,=r" + ",r" * 8 + ",f,f,f",
        loc,
        ip,
    )
    return [cutlass.Int32(llvm.extractvalue(ty, res, [i], loc=loc, ip=ip)) for i in range(3)]


class HcUpGateMeanRstdKernel:
    """`mixed[t, i] = 1/C sum_s sigmoid((low @ up)[t, s*H + i]) * hc_out[t, s*H + i]
    * rstd[t, s] * w1[s*H + i]` (+ its NVFP4 with `fp4`).
    """

    def __init__(self, hc: int, hidden: int, rank: int, fp4: bool):
        assert rank % 64 == 0 and hidden % _BLOCK_H == 0 and hc == 4
        self.bh = _BLOCK_H
        self.sub = _SUB
        self.hc, self.hidden, self.rank, self.fp4 = hc, hidden, rank, fp4
        self.kb = rank // 64
        self.acc_dtype = cutlass.Float32
        self.io_dtype = cutlass.BFloat16
        self.mma_tiler = (256, hc * self.bh, 64)
        self.cta_group = tcgen05.CtaGroup.TWO
        self.cluster_shape_mn = (2, 1)
        self.epi_sets = _SETS
        self.set_warps = _SET_WARPS
        self.epi_warps = _SETS * _SET_WARPS
        self.mma_warp_id = self.epi_warps
        self.tma_warp_id = self.epi_warps + 1
        self.nrm_warp_id = self.epi_warps + 2
        self.threads_per_cta = 32 * (self.epi_warps + 3)
        self.num_b_stage = _B_STAGES
        self.num_n_stage = _N_STAGES
        self.num_acc_stage = _ACC_STAGES
        # The pair's j-th tile uses accumulator stage j % ACC and, for stream s, the
        # hc_out stage (C * j + s) % NS.
        assert _ACC_STAGES & (_ACC_STAGES - 1) == 0 and _SETS & (_SETS - 1) == 0
        assert _N_STAGES >= hc * _SETS
        self.acc_shift = _ACC_STAGES.bit_length() - 1
        self.set_shift = _SETS.bit_length() - 1
        self.epilog_sync_bar_id = 1
        self.tmem_alloc_sync_bar_id = 2
        # Named barriers 3..3 + 4 * SETS - 1: a set's epilogue warps w and w + 4 (the
        # NVFP4 hand-off).
        self.pair_bar_id0 = 3

    def _setup(self):
        k_major = utils.LayoutEnum.ROW_MAJOR.mma_major_mode()
        tiled_mma = utils.sm100.make_trivial_tiled_mma(
            self.io_dtype,
            self.io_dtype,
            k_major,
            k_major,
            self.acc_dtype,
            self.cta_group,
            self.mma_tiler[:2],
        )
        self.cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout((*self.cluster_shape_mn, 1)), (tiled_mma.thr_id.shape,)
        )
        self.n_box = (128, self.bh)
        # A stage: one stream of this CTA's tile, its hc_out box ([128, 32] bf16) and
        # (1 + w) slice ([32] fp32).
        self.num_n_bytes = 128 * self.bh * 2 + self.bh * 4
        self.a_smem_layout_staged = utils.sm100.make_smem_layout_a(
            tiled_mma, self.mma_tiler, self.io_dtype, self.kb
        )
        self.b_smem_layout_staged = utils.sm100.make_smem_layout_b(
            tiled_mma, self.mma_tiler, self.io_dtype, self.num_b_stage
        )
        self.n_smem_layout_staged = utils.sm100.make_smem_layout_epi(
            self.io_dtype, utils.LayoutEnum.ROW_MAJOR, self.n_box, self.num_n_stage
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
        hc_ptr: cute.Pointer,
        mixed: cute.Tensor,
        xq_ptr: cute.Pointer,
        sf_ptr: cute.Pointer,
        gs_ptr: cute.Pointer,
        rstd_ptr: cute.Pointer,
        w1_ptr: cute.Pointer,
        max_clusters: cutlass.Constexpr,
        stream: cuda.CUstream,
    ):
        bh, hc, hid, rank = self.bh, self.hc, self.hidden, self.rank
        nt = hid // bh
        width = hc * hid
        a = cute.make_tensor(low_ptr, cute.make_layout((m, rank, 1), stride=(rank, 1, 0)))
        # The `[C*H, R]` up weight viewed as N = (32 hidden, C streams, H/32 slices): an
        # MMA tile's N is one hidden slice of every stream.
        b = cute.make_tensor(
            up_ptr,
            cute.make_layout(((bh, hc, nt), rank, 1), stride=((rank, hid * rank, bh * rank), 1, 0)),
        )
        nrm = cute.make_tensor(hc_ptr, cute.make_layout((m, width), stride=(width, 1)))
        gsc = cute.make_tensor(gs_ptr, cute.make_layout(1))
        tiled_mma = self._setup()
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
        # Both CTAs' A (all k blocks) / B stage land on the leader's barrier.
        self.a_bytes = cute.size_in_bytes(self.io_dtype, a_smem) * self.kb * 2
        self.b_bytes = cute.size_in_bytes(self.io_dtype, b_smem) * 2
        total = ((m + 255) // 256) * nt
        ncl = cutlass.min(total, max_clusters)
        self.kernel(
            tiled_mma,
            tma_atom_a,
            tma_a,
            tma_atom_b,
            tma_b,
            tma_atom_n,
            tma_n,
            gsc,
            mixed.iterator.toint(),
            xq_ptr.toint(),
            sf_ptr.toint(),
            rstd_ptr.toint(),
            w1_ptr.toint(),
            m,
            total,
            self.cluster_layout_vmnk,
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            self.n_smem_layout_staged,
        ).launch(
            grid=[2 * ncl, 1, 1],
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
        mGS: cute.Tensor,
        mixed_base: cutlass.Int64,
        q_base: cutlass.Int64,
        sf_base: cutlass.Int64,
        rstd_base: cutlass.Int64,
        w1_base: cutlass.Int64,
        m: cutlass.Int32,
        total: cutlass.Int32,
        cluster_layout_vmnk: cute.Layout,
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        n_smem_layout_staged: cute.ComposedLayout,
    ):
        hc, hid, bh, sub = self.hc, self.hidden, self.bh, self.sub
        nt = hid // bh
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        if warp_idx == self.tma_warp_id:
            cpasync.prefetch_descriptor(tma_atom_a)
            cpasync.prefetch_descriptor(tma_atom_b)
        if warp_idx == self.nrm_warp_id:
            cpasync.prefetch_descriptor(tma_atom_n)
        tidx, _, _ = cute.arch.thread_idx()
        bid, _, _ = cute.arch.block_idx()
        nblk, _, _ = cute.arch.grid_dim()
        v = bid % 2
        is_leader = v == 0
        cl = bid // 2
        ncl = nblk // 2
        cta_rank = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        cluster_coord = cluster_layout_vmnk.get_flat_coord(cta_rank)
        # The pair's contiguous range [t0, t1) of (256-row tile, hidden slice) tiles,
        # row-tile-major.
        t0 = (cl * total) // ncl
        t1 = ((cl + 1) * total) // ncl
        m_first = t0 // nt
        m_last = (t1 - 1) // nt

        @cute.struct
        class SharedStorage:
            a_mbar: cute.struct.MemRange[cutlass.Int64, 2]
            b_mbar: cute.struct.MemRange[cutlass.Int64, self.num_b_stage * 2]
            acc_mbar: cute.struct.MemRange[cutlass.Int64, self.num_acc_stage * 2]
            n_mbar: cute.struct.MemRange[cutlass.Int64, self.num_n_stage * 2]
            tmem_dealloc_mbar: cutlass.Int64
            tmem_holding_buf: cutlass.Int32

        smem = utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)
        a_producer, a_consumer = PipelineTmaUmma.create(
            barrier_storage=storage.a_mbar.data_ptr(),
            num_stages=1,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
            tx_count=self.a_bytes,
            cta_layout_vmnk=cluster_layout_vmnk,
        ).make_participants()
        b_producer, b_consumer = PipelineTmaUmma.create(
            barrier_storage=storage.b_mbar.data_ptr(),
            num_stages=self.num_b_stage,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
            tx_count=self.b_bytes,
            cta_layout_vmnk=cluster_layout_vmnk,
        ).make_participants()
        # An accumulator stage is consumed by one set's warps in each CTA.
        acc_pipeline = PipelineUmmaAsync.create(
            barrier_storage=storage.acc_mbar.data_ptr(),
            num_stages=self.num_acc_stage,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 2 * self.set_warps),
            cta_layout_vmnk=cluster_layout_vmnk,
        )
        # hc_out / (1 + w) stages are per CTA (each CTA its own rows), one set's warps
        # consuming each.
        n_pipe = pipeline.PipelineTmaAsync.create(
            barrier_storage=storage.n_mbar.data_ptr(),
            num_stages=self.num_n_stage,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, self.set_warps),
            tx_count=self.num_n_bytes,
            cta_layout_vmnk=cute.make_layout((1, 1, 1, 1)),
        )
        n_producer = n_pipe.make_participants()[0]
        tmem_alloc_barrier = pipeline.NamedBarrier(
            barrier_id=self.tmem_alloc_sync_bar_id, num_threads=32 * (1 + self.epi_warps)
        )
        tmem = utils.TmemAllocator(
            storage.tmem_holding_buf.ptr,
            barrier_for_retrieve=tmem_alloc_barrier,
            allocator_warp_id=0,
            is_two_cta=True,
            two_cta_tmem_dealloc_mbar_ptr=storage.tmem_dealloc_mbar.ptr,
        )
        pipeline_init_arrive(cluster_shape_mn=self.cluster_shape_mn, is_relaxed=True)

        sA = smem.allocate_tensor(
            element_type=self.io_dtype,
            layout=a_smem_layout_staged.outer,
            byte_alignment=1024,
            swizzle=a_smem_layout_staged.inner,
        )
        sB = smem.allocate_tensor(
            element_type=self.io_dtype,
            layout=b_smem_layout_staged.outer,
            byte_alignment=1024,
            swizzle=b_smem_layout_staged.inner,
        )
        sN = smem.allocate_tensor(
            element_type=self.io_dtype,
            layout=n_smem_layout_staged.outer,
            byte_alignment=1024,
            swizzle=n_smem_layout_staged.inner,
        )
        # NVFP4: a set's upper warp group hands its half of a row's e2m1 / scale bytes to
        # the lower group through shared memory (two tile slots per set), which stores
        # the whole row piece (one 16-byte e2m1 store, one 2-byte scale store).
        sX = smem.allocate_tensor(
            element_type=cutlass.Int32,
            layout=cute.make_layout((2 * self.epi_sets, 128, 4), stride=(512, 4, 1)),
            byte_alignment=128,
        )
        sW = smem.allocate_tensor(
            element_type=cutlass.Float32,
            layout=cute.make_layout((bh, self.num_n_stage), stride=(1, bh)),
            byte_alignment=128,
        )

        gA_mkl = cute.local_tile(
            mA_mkl, cute.slice_(self.mma_tiler, (None, 0, None)), (None, None, None)
        )
        gB_nkl = cute.local_tile(
            mB_nkl, cute.slice_(self.mma_tiler, (0, None, None)), (None, None, None)
        )
        gN_box = cute.local_tile(mN_mn, self.n_box, (None, None))
        thr_mma = tiled_mma.get_slice(v)
        tCgA = thr_mma.partition_A(gA_mkl)
        tCgB = thr_mma.partition_B(gB_nkl)
        a_cta_layout = cute.make_layout(cute.slice_(cluster_layout_vmnk, (0, 0, None, 0)).shape)
        tAsA, tAgA = cpasync.tma_partition(
            tma_atom_a,
            cluster_coord[2],
            a_cta_layout,
            cute.group_modes(sA, 0, 3),
            cute.group_modes(tCgA, 0, 3),
        )
        b_cta_layout = cute.make_layout(cute.slice_(cluster_layout_vmnk, (0, None, 0, 0)).shape)
        tBsB, tBgB = cpasync.tma_partition(
            tma_atom_b,
            cluster_coord[1],
            b_cta_layout,
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
        a_mcast = cpasync.create_tma_multicast_mask(
            cluster_layout_vmnk, cluster_coord, mcast_mode=2
        )
        b_mcast = cpasync.create_tma_multicast_mask(
            cluster_layout_vmnk, cluster_coord, mcast_mode=1
        )
        tCrA = tiled_mma.make_fragment_A(sA)
        tCrB = tiled_mma.make_fragment_B(sB)
        acc_shape = tiled_mma.partition_shape_C(self.mma_tiler[:2])
        tCtAcc_fake = tiled_mma.make_fragment_C(cute.append(acc_shape, self.num_acc_stage))

        pipeline_init_wait(cluster_shape_mn=self.cluster_shape_mn)

        # ---------------------------------------------------------- TMA (A, B): both CTAs
        if warp_idx == self.tma_warp_id:
            cute.arch.griddepcontrol_wait()
            for mi in cutlass.range(m_first, m_last + 1, 1, unroll=1):
                a_handle = a_producer.acquire_and_advance()
                for kb in cutlass.range_constexpr(self.kb):
                    cute.copy(
                        tma_atom_a,
                        tAgA[(None, mi, kb, 0)],
                        tAsA[(None, kb)],
                        tma_bar_ptr=a_handle.barrier,
                        mcast_mask=a_mcast,
                    )
                n_lo = cutlass.max(t0 - mi * nt, 0)
                n_hi = cutlass.min(t1 - mi * nt, nt)
                for n in cutlass.range(n_lo, n_hi, 1, unroll=1):
                    for kb in cutlass.range_constexpr(self.kb):
                        handle = b_producer.acquire_and_advance()
                        cute.copy(
                            tma_atom_b,
                            tBgB[(None, n, kb, 0)],
                            tBsB[(None, handle.index)],
                            tma_bar_ptr=handle.barrier,
                            mcast_mask=b_mcast,
                        )
            a_producer.tail()
            b_producer.tail()

        # ---------------------------------------------------------- hc_out boxes, (1 + w)
        if warp_idx == self.nrm_warp_id:
            cute.arch.griddepcontrol_wait()
            sw_base = cutlass.Int32(sW.iterator.toint())
            for mi in cutlass.range(m_first, m_last + 1, 1, unroll=1):
                n_lo = cutlass.max(t0 - mi * nt, 0)
                n_hi = cutlass.min(t1 - mi * nt, nt)
                for n in cutlass.range(n_lo, n_hi, 1, unroll=1):
                    for s in cutlass.range_constexpr(hc):
                        handle = n_producer.acquire_and_advance()
                        with cute.arch.elect_one():
                            _bulk_g2s(
                                sw_base + handle.index * (bh * 4),
                                w1_base + cutlass.Int64(s * hid + n * bh) * 4,
                                bh * 4,
                                cutlass.Int32(handle.barrier.toint()),
                            )
                        cute.copy(
                            tma_atom_n,
                            tNgN[(None, 2 * mi + v, s * nt + n)],
                            tNsN[(None, handle.index)],
                            tma_bar_ptr=handle.barrier,
                        )
            n_producer.tail()

        # ---------------------------------------------------------- MMA (the leader issues)
        if warp_idx == self.mma_warp_id:
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            tCtAcc_base = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)
            acc_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_acc_stage
            )
            # Both CTAs walk the tiles (the accumulator state advances in both); the
            # leader issues.
            for mi in cutlass.range(m_first, m_last + 1, 1, unroll=1):
                n_lo = cutlass.max(t0 - mi * nt, 0)
                n_hi = cutlass.min(t1 - mi * nt, nt)
                if is_leader:
                    a_handle = a_consumer.wait_and_advance()
                    for n in cutlass.range(n_lo, n_hi, 1, unroll=1):
                        tCtAcc = tCtAcc_base[(None, None, None, acc_state.index)]
                        acc_pipeline.producer_acquire(acc_state)
                        tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
                        for kb in cutlass.range_constexpr(self.kb):
                            handle = b_consumer.wait_and_advance()
                            num_kblocks = cute.size(tCrA, mode=[2])
                            for kk in cutlass.range(num_kblocks, unroll_full=True):
                                cute.gemm(
                                    tiled_mma,
                                    tCtAcc,
                                    tCrA[(None, None, kk, kb)],
                                    tCrB[(None, None, kk, handle.index)],
                                    tCtAcc,
                                )
                                tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                            handle.release()
                        acc_pipeline.producer_commit(acc_state)
                        acc_state.advance()
                    a_handle.release()
                else:
                    for n in cutlass.range(n_lo, n_hi, 1, unroll=1):
                        acc_state.advance()
            acc_pipeline.producer_tail(acc_state)

        # ---------------------------------------------------------- epilogue (each CTA its rows)
        if warp_idx < self.epi_warps:
            cute.arch.griddepcontrol_wait()
            tmem.allocate(self.num_tmem_alloc_cols)
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            tCtAcc_base = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)
            # Set `eset` takes the pair's tiles t0 + eset, t0 + eset + SETS, ...; thread
            # (grp, lt) of a set owns row lt of the CTA's 128 and columns
            # [grp * 16, grp * 16 + 16) of the tile's 32.
            eset = warp_idx // self.set_warps
            lt = tidx % 128
            grp = (tidx // 128) % 2
            epi_tile = (128, sub)
            copy_atom_t2r = utils.sm100.get_tmem_load_op(
                (128, self.mma_tiler[1], self.mma_tiler[2]),
                utils.LayoutEnum.ROW_MAJOR,
                self.io_dtype,
                self.acc_dtype,
                epi_tile,
                True,
            )
            tAcc_epi = cute.flat_divide(tCtAcc_base[((None, None), 0, 0, None)], epi_tile)
            tiled_copy_t2r = tcgen05.make_tmem_copy(copy_atom_t2r, tAcc_epi[(None, None, 0, 0, 0)])
            thr_copy_t2r = tiled_copy_t2r.get_slice(lt)
            tTR_tAcc = thr_copy_t2r.partition_S(tAcc_epi)
            tTR_sN = thr_copy_t2r.partition_D(cute.flat_divide(sN, epi_tile))
            sw_addr = cutlass.Int32(sW.iterator.toint())
            sx_addr = cutlass.Int32(sX.iterator.toint()) + eset * (2 * 128 * 16)
            frag_shape = tTR_sN[(None, None, None, 0, 0, 0)].shape
            rAcc = cute.make_rmem_tensor(frag_shape, self.acc_dtype)
            rN = cute.make_rmem_tensor(frag_shape, self.io_dtype)
            # sigmoid(z) * r / C = fma(tanh(z / 2), r / 2C, r / 2C): the stream mean and the
            # norm's rstd folded into the gate.
            scale = cutlass.Float32(0.5 / hc)
            if cutlass.const_expr(self.fp4):
                gs = mGS[0]
                rcp_gs = _rcp_approx_ftz(gs)
                rcp6 = _rcp_approx_ftz(cutlass.Float32(6.0))

            # The rows' rstd, reloaded when the row tile changes.
            mi_cur = cutlass.Int32(-1)
            rs0 = cutlass.Float32(0.0)
            rs1 = cutlass.Float32(0.0)
            rs2 = cutlass.Float32(0.0)
            rs3 = cutlass.Float32(0.0)
            for t in cutlass.range(t0 + eset, t1, self.epi_sets, unroll=1):
                j = t - t0
                acc_state = pipeline.PipelineState(
                    self.num_acc_stage,
                    j,
                    j & (self.num_acc_stage - 1),
                    (j >> self.acc_shift) & 1,
                )
                mi = t // nt
                n = t - mi * nt
                row = mi * 256 + v * 128 + lt
                row_ok = row < m
                if mi != mi_cur:
                    rs0, rs1, rs2, rs3 = _ldg_v4_f32(
                        rstd_base + cutlass.Int64(cutlass.min(row, m - 1)) * (hc * 4)
                    )
                    mi_cur = mi
                rs = [rs0, rs1, rs2, rs3]
                col0 = n * bh + grp * sub
                acc_pipeline.consumer_wait(acc_state)
                acc = [cutlass.Float32(0.0)] * sub
                for s in cutlass.range_constexpr(hc):
                    cute.copy(
                        tiled_copy_t2r,
                        tTR_tAcc[(None, None, None, 0, 2 * s + grp, acc_state.index)],
                        rAcc,
                    )
                    # tcgen05.wait::ld: the stream's accumulator columns are in registers;
                    # after the last stream the stage goes back before the math.
                    cute.arch.fence_view_async_tmem_load()
                    if cutlass.const_expr(s == hc - 1):
                        cute.arch.sync_warp()
                        with cute.arch.elect_one():
                            acc_pipeline.consumer_release(acc_state)
                    hrs = rs[s] * scale
                    q = j * hc + s
                    if cutlass.const_expr(self.num_n_stage == hc * self.epi_sets):
                        # The set's stream-s stage is fixed; its phase flips every own tile.
                        n_state = pipeline.PipelineState(
                            self.num_n_stage, q, eset * hc + s, (j >> self.set_shift) & 1
                        )
                    else:
                        n_state = pipeline.PipelineState(
                            self.num_n_stage, q, q % self.num_n_stage, (q // self.num_n_stage) & 1
                        )
                    n_pipe.consumer_wait(n_state)
                    cute.autovec_copy(tTR_sN[(None, None, None, 0, grp, n_state.index)], rN)
                    waddr = sw_addr + (n_state.index * bh + grp * sub) * 4
                    w1 = []
                    for c4 in cutlass.range_constexpr(sub // 4):
                        w1 += _lds_v4_f32(waddr + c4 * 16)
                    xw = cute.recast_tensor(rN, cutlass.Int32)
                    for k in cutlass.range_constexpr(sub // 2):
                        acc[2 * k], acc[2 * k + 1] = _gate2(
                            rAcc[2 * k],
                            rAcc[2 * k + 1],
                            xw[k],
                            w1[2 * k],
                            w1[2 * k + 1],
                            hrs,
                            acc[2 * k],
                            acc[2 * k + 1],
                            s == 0,
                        )
                    # Every read of the stage (hc_out, (1 + w)) is performed before it is
                    # handed back.
                    cute.arch.fence_acq_rel_cta()
                    cute.arch.sync_warp()
                    n_pipe.consumer_release(n_state)  # lane 0 of each warp arrives
                words = [_bf16x2(acc[2 * k], acc[2 * k + 1]) for k in range(sub // 2)]
                if row_ok:
                    _st_global_v8(mixed_base + (cutlass.Int64(row) * hid + col0) * 2, words)
                if cutlass.const_expr(self.fp4):
                    q0, q1, sf = _nvfp4_16(words, gs, rcp_gs, rcp6)
                    slot = sx_addr + (((t // self.epi_sets) % 2) * 128 + lt) * 16
                    if grp == 1:
                        _sts_v4(slot, q0, q1, sf, sf)
                    cute.arch.barrier(
                        barrier_id=self.pair_bar_id0 + warp_idx % 4 + 4 * eset,
                        number_of_threads=64,
                    )
                    if grp == 0:
                        pq = _lds_v4(slot)
                        if row_ok:
                            row64 = cutlass.Int64(row)
                            _st_global_v4(
                                q_base + row64 * (hid // 2) + cutlass.Int64(n * bh // 2),
                                q0,
                                q1,
                                pq[0],
                                pq[1],
                            )
                            _st_global_b16(
                                sf_base + row64 * (hid // 16) + cutlass.Int64(n * bh // 16),
                                sf | (pq[2] << 8),
                            )

            cute.arch.barrier(
                barrier_id=self.epilog_sync_bar_id, number_of_threads=32 * self.epi_warps
            )
            tmem.relinquish_alloc_permit()
            tmem.free(tmem_ptr)


_COMPILED: dict = {}


def _launcher(hc: int, hidden: int, rank: int, fp4: bool, ptrs):
    """Compile once per static configuration; later calls reuse the TVM-FFI entry,
    which launches on the caller's current stream (graph-capturable). `ptrs` is
    `(low, up, hc_out, mixed tensor, xq, sf, gs, rstd, norm_w1)`.
    """
    key = (hc, hidden, rank, fp4)
    fn = _COMPILED.get(key)
    if fn is None:
        kern = HcUpGateMeanRstdKernel(hc, hidden, rank, fp4)
        gm = cute.AddressSpace.gmem
        cptrs = [
            make_ptr(cutlass.BFloat16, ptrs[0], gm, assumed_align=16),
            make_ptr(cutlass.BFloat16, ptrs[1], gm, assumed_align=16),
            make_ptr(cutlass.BFloat16, ptrs[2], gm, assumed_align=16),
            from_dlpack(ptrs[3], assumed_align=16).mark_layout_dynamic(leading_dim=1),
            make_ptr(cutlass.Int32, ptrs[4], gm, assumed_align=16),
            make_ptr(cutlass.Uint8, ptrs[5], gm, assumed_align=2),
            make_ptr(cutlass.Float32, ptrs[6], gm, assumed_align=4),
            make_ptr(cutlass.Float32, ptrs[7], gm, assumed_align=16),
            make_ptr(cutlass.Float32, ptrs[8], gm, assumed_align=16),
        ]
        fn = cute.compile(
            kern,
            128,
            *cptrs,
            max_clusters=utils.HardwareInfo().get_max_active_clusters(2),
            stream=cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
            options="--opt-level 2 --enable-tvm-ffi",
        )
        _COMPILED[key] = fn
    return fn


def hc_up_gate_mean_rstd(
    low: torch.Tensor,
    up: torch.Tensor,
    hc_out: torch.Tensor,
    rstd: torch.Tensor,
    norm_w1: torch.Tensor,
    hc_count: int,
    global_scale: Optional[torch.Tensor] = None,
):
    """`mixed [T, H]`, and with `global_scale` also `xq [T, H/2]` and `xsf [T*H/16]`
    (linear scales).
    """
    rows, rank = low.shape
    hidden = hc_out.shape[1] // hc_count
    dev = hc_out.device
    fp4 = global_scale is not None
    mixed = torch.empty((rows, hidden), dtype=hc_out.dtype, device=dev)
    xq = xsf = None
    if fp4:
        xq = torch.empty((rows, hidden // 2), dtype=torch.uint8, device=dev)
        xsf = torch.empty((rows * (hidden // SF_VEC),), dtype=torch.uint8, device=dev)
    if rows > 0:
        # The NVFP4 pointers are never read without `fp4`.
        qp = (
            (xq.data_ptr(), xsf.data_ptr(), global_scale.data_ptr())
            if fp4
            else (mixed.data_ptr(),) * 3
        )
        ptrs = (
            low.data_ptr(),
            up.data_ptr(),
            hc_out.data_ptr(),
            mixed,
            *qp,
            rstd.data_ptr(),
            norm_w1.data_ptr(),
        )
        _launcher(hc_count, hidden, rank, fp4, ptrs)(rows, *ptrs)
    return (mixed, xq, xsf) if fp4 else mixed


def _mixed_reference(low, up, hc_out, rstd, norm_w1, hc_count) -> torch.Tensor:
    """float64: the gate times the unrounded normed stream, stream mean, one rounding."""
    hidden = hc_out.shape[1] // hc_count
    gate = torch.sigmoid(F.linear(low.double(), up.double().t())).unflatten(-1, (hc_count, hidden))
    x = hc_out.double().unflatten(-1, (hc_count, hidden)) * rstd.double().unsqueeze(-1)
    x = x * norm_w1.double().unflatten(-1, (hc_count, hidden))
    return (gate * x).mean(dim=-2).to(hc_out.dtype)


def _check(low, up, hc_out, rstd, norm_w1, hc_count) -> None:
    rows, rank = low.shape
    width = hc_out.shape[1]
    assert low.dtype == up.dtype == hc_out.dtype == torch.bfloat16, "bf16 operands"
    assert hc_out.shape[0] == rows and up.shape == (rank, width), "shapes"
    assert low.stride() == (rank, 1) and hc_out.stride() == (width, 1), "contiguous rows"
    assert up.stride() == (1, rank), "up must be the transposed view of a [C*H, R] weight"
    assert rstd.dtype == torch.float32 and rstd.shape == (rows, hc_count), "rstd [T, C] fp32"
    assert rstd.is_contiguous(), "rstd contiguous"
    assert norm_w1.dtype == torch.float32 and norm_w1.shape == (width,), "norm_w1 [C*H] fp32"
    assert norm_w1.is_contiguous(), "norm_w1 contiguous"
    # A row's rstd is one 16-byte load, and the MMA's N is C * 32 = 128.
    assert hc_count == 4, "hc_count 4"
    assert width % hc_count == 0 and (width // hc_count) % _BLOCK_H == 0, "H % 32 == 0"
    assert rank % 64 == 0, "rank must be a multiple of 64"
    for t in (low, up, hc_out, rstd, norm_w1):
        assert t.data_ptr() % 16 == 0, "16-byte aligned operands"


class Qwen4ExpHcUpGateMeanRstd(OpWrapper):
    """With `C = hc_count`, `H = width // C`, `low` the `[T, R]` SiLU bottleneck,
    `up` the `[R, C*H]` transposed view of the `[C*H, R]` up weight, rstd the site's
    `[T, C]` fp32 statistic and `norm_w1 = 1 + norm_weight` in fp32 `[C*H]`::

        normed[t, s*H + i] = hc_out[t, s*H + i] * rstd[t, s] * norm_w1[s*H + i]
        mixed[t, i]        = mean over s of sigmoid((low @ up)[t, s*H + i]) * normed[t, s*H + i]

    The GEMM accumulates in fp32 in tensor memory and is never rounded or written;
    `normed` is fp32 in registers (not rounded to bf16 as a stored normed stream
    is); gate, product and stream mean are fp32 with one rounding to `hc_out`'s
    dtype. Per element and stream, in stream order: `t = tanh.approx(z * 0.5)`,
    `g = fma(t, r, r)` with `r = rstd / 2C`, `xw = hc_out * norm_w1`,
    `acc = fma(g, xw, acc)` (`g * xw` for stream 0), each a round-to-nearest fp32
    operation. Returns a fresh `[T, H]` tensor. Fusion boundary: the up projection,
    the grouped norm's scaling, the sigmoid gate and the stream mean.
    """

    ARCHS = frozenset({Arch.SM_103})

    # spec: rows. The checkpoint's geometry: hc_count 4, hidden 2560, rank 320, bf16.
    CELLS: tuple[Cell, ...] = tuple(
        Cell(why=f"checkpoint 4x2560, rank 320, {why}", spec=dict(rows=t))
        for t, why in (
            (16384, "prefill chunk 16384 (max_num_tokens)"),
            (2048, "prefill 2048 rows"),
            (16390, "ragged tail, 16320 + 70 rows (mixed step)"),
            (5, "five rows, less than one tile"),
        )
    )

    note = """
    Tier 6 (CuTe DSL written for this target): a 2-CTA tcgen05 GEMM with a
    multi-input epilogue (`qwen4_exp_hc_up_gate_mean`'s, plus the norm's
    scaling); no lower tier has one. Metadata consumed: none. Call sites:
    `PrefillTarget`'s attention / GDN and stack-final mixers at or above
    `_HC_RSTD_MIN_ROWS` rows, after `qwen4_exp_hc_first_rescale_tc`, on
    `qwen4_exp_hc_inject_rstd`'s `hc_out` and rstd; `norm_w1` precomputed once
    per site at load.

    Preconditions (bf16 activations): `low` `[T, R]` contiguous, `up` the
    transposed view of a contiguous `[C*H, R]` weight, `hc_out` `[T, C*H]`
    contiguous, rstd `[T, C]` and `norm_w1` `[C*H]` fp32 contiguous, 16-byte
    aligned pointers, `hc_count == 4`, `R % 64 == 0`, `H % 32 == 0`. Compiles
    once per `(C, H, R)` on first call; the row count is a runtime argument.
    Graph-capturable (current stream, no host sync). Launched with
    programmatic dependent launch (TRTLLM_ENABLE_PDL=0 turns it off).

    Launch: persistent, one cluster (CTA pair) per two SMs, each walking a
    contiguous row-tile-major range of (256-row tile, 32-column slice) tiles;
    608 threads a CTA (two sets of eight epilogue warps on alternate tiles, a
    weight-load, an hc_out-load and an MMA warp), about 217 KB of shared memory
    (low tile, 8-stage weight ring, 8 per-stream hc_out stages) and all 512
    tensor-memory columns (4 accumulator stages). GB300, 16 calls chained over
    distinct weights: 4096 / 8192 / 16384 rows about 24 / 45 / 89 us.

    Numerics: the gate is `tanh.approx`-based (as `qwen4_exp_hc_up_gate_mean`'s,
    max relative error 2^-10.99) with rstd / C folded into it; against the float64
    value of the unrounded normed stream the result sits in the gate band at
    about 0.1-0.25 of it (the stored-normed op: 0.3), and is not bitwise
    `qwen4_exp_hc_up_gate_mean`'s (that op gates the bf16-rounded normed). The
    fp32 sequence above is bitwise what the GPU test emulates (zero logits).

    Certified: bf16 at the checkpoint's 4 x 2560, rank 320: 16384, 2048, 5
    rows and a partial last tile (16390); gate band (`rtol = atol / max|ref| =
    eps`) against the float64 reference; the epilogue's fp32 sequence bitwise
    (5, 300, 4100 rows); graph replay.

    Not certified: zero rows, hc_count other than 4, hidden other than 2560,
    rank other than 320.
    """

    def __call__(
        self,
        low: torch.Tensor,
        up: torch.Tensor,
        hc_out: torch.Tensor,
        rstd: torch.Tensor,
        norm_w1: torch.Tensor,
        hc_count: int,
    ) -> torch.Tensor:
        return hc_up_gate_mean_rstd(low, up, hc_out, rstd, norm_w1, hc_count)

    def reference(
        self,
        low: torch.Tensor,
        up: torch.Tensor,
        hc_out: torch.Tensor,
        rstd: torch.Tensor,
        norm_w1: torch.Tensor,
        hc_count: int,
    ) -> torch.Tensor:
        """float64, one rounding."""
        return _mixed_reference(low, up, hc_out, rstd, norm_w1, hc_count)

    def is_valid(
        self,
        low: torch.Tensor,
        up: torch.Tensor,
        hc_out: torch.Tensor,
        rstd: torch.Tensor,
        norm_w1: torch.Tensor,
        hc_count: int,
    ) -> None:
        _check(low, up, hc_out, rstd, norm_w1, hc_count)


class Qwen4ExpHcUpGateMeanRstdFp4(OpWrapper):
    """`qwen4_exp_hc_up_gate_mean_rstd` at an MoE site, whose epilogue also writes
    the routed experts' NVFP4 input::

        mixed = qwen4_exp_hc_up_gate_mean_rstd(low, up, hc_out, rstd, norm_w1, C)
        xq, xsf = NVFP4(mixed)  # e2m1 pairs, e4m3 scale per 16, linear layout

    `mixed` is bitwise the bf16 entry's (same kernel, same epilogue arithmetic);
    the quantization reads that bf16-rounded value with
    `qwen4_exp_hc_up_gate_mean_fp4`'s recipe. Returns fresh `(mixed [T, H], xq
    [T, H/2] uint8, xsf [T*H/16] uint8)`. `global_scale` is the 1-element fp32
    NVFP4 global scale. Fusion boundary: the up projection, the norm's scaling,
    the sigmoid gate, the stream mean and the NVFP4 quantization of the mean.
    """

    ARCHS = frozenset({Arch.SM_103})

    # spec: rows. The checkpoint's geometry: hc_count 4, hidden 2560, rank 320, bf16.
    CELLS: tuple[Cell, ...] = tuple(
        Cell(why=f"checkpoint 4x2560, rank 320, NVFP4, {why}", spec=dict(rows=t))
        for t, why in (
            (16384, "prefill chunk 16384 (max_num_tokens)"),
            (2048, "prefill 2048 rows"),
            (16390, "ragged tail, 16320 + 70 rows (mixed step)"),
            (5, "five rows, less than one tile"),
        )
    )

    note = """
    Tier 6 (CuTe DSL written for this target): `qwen4_exp_hc_up_gate_mean_rstd`'s
    kernel compiled with its NVFP4 epilogue. Metadata consumed: none. Call
    sites: `PrefillTarget`'s 48 MoE-site mixers at or above `_HC_RSTD_MIN_ROWS`
    rows; `xq` / `xsf` feed the routed experts (linear scales), `mixed` the
    router and the shared expert. Preconditions: those of
    `qwen4_exp_hc_up_gate_mean_rstd` and `global_scale` one fp32 element on the
    device. Compiles once per `(C, H, R)`; graph-capturable.

    Each thread holds 16 of a row's 32 tile columns (one scale block); within an
    epilogue set the upper warp group hands its e2m1 word pair and scale byte to
    the lower one through shared memory, which stores the row's 16 e2m1 bytes and
    2 scale bytes in one store each. GB300, 16 calls chained over distinct
    weights: 4096 / 8192 / 16384 rows about 26 / 53 / 100 us.

    Numerics: `mixed` as the bf16 entry. The quantization is the trtllm NVFP4
    recipe instruction for instruction (`rcp.approx.ftz.f32` reciprocals,
    `cvt.rn.satfinite` to e4m3 and to e2m1, every multiply `mul.rn.f32`); the
    GPU test checks it against `fp4_quantize`'s native-torch reference on the
    returned `mixed` (scales bitwise, e2m1 codes bitwise outside the reciprocal
    tie window).

    Certified: bf16 at the checkpoint's 4 x 2560, rank 320, a calibrated
    global scale: 16384, 2048, 5 rows and a partial last tile (16390); graph
    replay.

    Not certified: zero rows, hc_count other than 4, hidden other than 2560,
    rank other than 320.
    """

    def __call__(
        self,
        low: torch.Tensor,
        up: torch.Tensor,
        hc_out: torch.Tensor,
        rstd: torch.Tensor,
        norm_w1: torch.Tensor,
        hc_count: int,
        global_scale: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return hc_up_gate_mean_rstd(low, up, hc_out, rstd, norm_w1, hc_count, global_scale)

    def reference(
        self,
        low: torch.Tensor,
        up: torch.Tensor,
        hc_out: torch.Tensor,
        rstd: torch.Tensor,
        norm_w1: torch.Tensor,
        hc_count: int,
        global_scale: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """`mixed` in float64 with one rounding, then the NVFP4 of that rounded value
        in native torch.
        """
        mixed = _mixed_reference(low, up, hc_out, rstd, norm_w1, hc_count)
        return (mixed,) + fp4_quantize.reference(mixed, global_scale, SF_VEC, False, False)

    def is_valid(
        self,
        low: torch.Tensor,
        up: torch.Tensor,
        hc_out: torch.Tensor,
        rstd: torch.Tensor,
        norm_w1: torch.Tensor,
        hc_count: int,
        global_scale: torch.Tensor,
    ) -> None:
        _check(low, up, hc_out, rstd, norm_w1, hc_count)
        assert (hc_out.shape[1] // hc_count) % SF_VEC == 0, "hidden must be whole 16-blocks"
        # The kernel loads element 0 only; a per-row scale would be applied as row 0's.
        assert global_scale.numel() == 1 and global_scale.dtype == torch.float32
        assert global_scale.is_cuda


qwen4_exp_hc_up_gate_mean_rstd = Qwen4ExpHcUpGateMeanRstd()
qwen4_exp_hc_up_gate_mean_rstd_fp4 = Qwen4ExpHcUpGateMeanRstdFp4()
