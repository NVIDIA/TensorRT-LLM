# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Hyper-connection first projection on the un-normalized residual, one tcgen05
launch: the A operand is the site's normed stream formed on chip from `hc_out`
and its per-(row, stream) rstd, so `[T, C*H]` normed never exists in memory.

Tier 6 (CuTe DSL written for this target). A 2-CTA tcgen05 GEMM (M = 256 rows a
CTA pair, K = 64 a stage, N = 352 as two MMAs of 192 + 160 columns). Per CTA: a
TMA warp lands the raw 128 x 64 `hc_out` tile (128B-swizzled) in the operand
ring; eight transform warps scale it in place by `rstd * (1 + w)`, round to bf16
and commit the stage to the pair's leader; a TMA warp streams both halves of the
first weight (2-CTA multicast); the leader's MMA warp accumulates in tensor
memory. The same eight warps then run the epilogue (TMEM -> registers -> SiLU /
2 sigmoid -> HBM): `qwen4_exp_hc_lowrank_act`'s activations without the fp32
logits round trip.

With few row tiles (`_pick_split`) each CTA pair covers `1 / split` of K and
writes fp32 partial logits `[split, T, 352]`; a Triton combine sums them in split
order and applies the activations.
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
import triton
import triton.language as tl
from cutlass._mlir.dialects import llvm
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.cute.runtime import from_dlpack
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper
from tensorrt_llm._torch._experimental.modeling_v2.catalog.hc.qwen4_exp_hc_inject_norm import (
    _bf16_hi,
    _bf16_lo,
    _bf16x2,
    _ldg_v4,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.hc.qwen4_exp_hc_up_gate_mean import (
    _asm,
    _st_global_v4,
)
from tensorrt_llm._torch.flashinfer_utils import get_env_enable_pdl
from tensorrt_llm._torch.kernels.blackwell.custom_pipeline import PipelineTmaUmma, PipelineUmmaAsync
from tensorrt_llm._torch.kernels.blackwell.utils import make_ptr

#: Programmatic dependent launch (sm_90+), off with TRTLLM_ENABLE_PDL=0.
_PDL = get_env_enable_pdl()

#: K a stage (one 128B-swizzled bf16 row of the operand tile).
BK = 64
#: Columns of the two MMAs a K step: 192 + 160 = 352 >= rank + C, the padded first
#: weight's columns the cuBLAS path multiplies (`hc_first_cols`). The 2-CTA atom needs
#: N / 2 a multiple of 16 per CTA, so two 176-column halves do not build.
NH0 = 192
NH1 = 160
NCOL = NH0 + NH1
#: Operand (A) ring depth: A is the DRAM stream (`hc_out`), B (the weight) comes from L2.
_A_STAGES = 6
_B_STAGES = 5
#: Transform warp groups: group g transforms k blocks j = g (mod groups), so that many
#: k blocks are in flight in the transform at once (its per-k-block latency is what
#: bounds a single group).
_GROUPS = 2
#: Largest K split (`_pick_split`).
_MAX_SPLIT = 8


@dsl_user_op
def _lds_v4(addr, *, loc=None, ip=None):
    """16 bytes of shared memory. Side-effecting, so never moved across the barrier
    wait that makes them valid.
    """
    ty = T.i32()
    res = _asm(
        llvm.StructType.get_literal([ty] * 4),
        [cutlass.Int32(addr)],
        "ld.shared.v4.b32 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,r",
        loc,
        ip,
        side_effects=True,
    )
    return [cutlass.Int32(llvm.extractvalue(ty, res, [i], loc=loc, ip=ip)) for i in range(4)]


@dsl_user_op
def _sts_v4(addr, a, b, c, d, *, loc=None, ip=None):
    """16 bytes to shared memory."""
    _asm(
        T.i32(),
        [cutlass.Int32(addr)] + [cutlass.Int32(v) for v in (a, b, c, d)],
        "{ st.shared.v4.b32 [$1], {$2, $3, $4, $5}; mov.u32 $0, 0; }",
        "=r,r,r,r,r,r",
        loc,
        ip,
        side_effects=True,
    )


@dsl_user_op
def _ldg_f32(addr, *, loc=None, ip=None):
    return cutlass.Float32(
        _asm(
            T.f32(),
            [cutlass.Int64(addr)],
            "ld.global.f32 $0, [$1];",
            "=f,l",
            loc,
            ip,
            side_effects=True,
        )
    )


@dsl_user_op
def _bits(x, *, loc=None, ip=None):
    return cutlass.Int32(_asm(T.i32(), [cutlass.Float32(x)], "mov.b32 $0, $1;", "=r,f", loc, ip))


@dsl_user_op
def _mul(a, b, *, loc=None, ip=None):
    """`mul.rn.f32`: never contracted into an FMA, so the rounding is the norm's."""
    return cutlass.Float32(
        _asm(
            T.f32(),
            [cutlass.Float32(a), cutlass.Float32(b)],
            "mul.rn.f32 $0, $1, $2;",
            "=f,f,f",
            loc,
            ip,
        )
    )


@dsl_user_op
def _add1(a, *, loc=None, ip=None):
    """`1 + a` in fp32 (`add.rn`): the grouped norm's `(1 + w)`."""
    return cutlass.Float32(
        _asm(T.f32(), [cutlass.Float32(a)], "add.rn.f32 $0, $1, 0f3F800000;", "=f,f", loc, ip)
    )


@dsl_user_op
def _sigmoid_fast(x, *, loc=None, ip=None):
    """1 / (1 + 2^(-x log2 e)) with ex2.approx and rcp.approx (two MUFU ops, no
    division).
    """
    return cutlass.Float32(
        _asm(
            T.f32(),
            [cutlass.Float32(x)],
            "{ .reg .f32 t; mul.f32 t, $1, 0fBFB8AA3B; ex2.approx.ftz.f32 t, t; "
            "add.f32 t, t, 0f3F800000; rcp.approx.ftz.f32 $0, t; }",
            "=f,f",
            loc,
            ip,
        )
    )


@dsl_user_op
def _st_global_v2(addr, a, b, *, loc=None, ip=None):
    """8 bytes to global memory (a row's four bf16 injection weights)."""
    _asm(
        T.i32(),
        [cutlass.Int64(addr), cutlass.Int32(a), cutlass.Int32(b)],
        "{ st.global.v2.b32 [$1], {$2, $3}; mov.u32 $0, 0; }",
        "=r,l,r,r",
        loc,
        ip,
        side_effects=True,
    )


class HcFirstRescaleKernel:
    """`low = silu(A @ first[:rank]^T / C)`, `inject = 2 sigmoid(A @ first[rank:rank+C]^T / C)`
    with `A = bf16(hc_out * rstd * (1 + w))` formed in the operand ring (`split == 1`), or the
    fp32 partial logits of K slice `sp` (`split > 1`).
    """

    def __init__(self, hc: int, hidden: int, rank: int, split: int):
        assert (hc * hidden) % BK == 0 and hidden % BK == 0 and (hc * hidden // BK) % split == 0
        assert rank + hc <= NCOL and rank % 16 == 0 and hc == 4
        self.hc, self.hidden, self.rank, self.split = hc, hidden, rank, split
        self.kt = hc * hidden // BK // split  # k blocks a CTA pair
        self.num_stages = _A_STAGES
        self.num_b_stages = _B_STAGES
        self.groups = _GROUPS
        self.acc_dtype = cutlass.Float32
        self.io_dtype = cutlass.BFloat16
        self.cta_group = tcgen05.CtaGroup.TWO
        self.cluster_shape_mn = (2, 1)
        self.mma_tiler = (256, NH0, BK)
        self.mma_tiler1 = (256, NH1, BK)
        self.tw = 8  # transform / epilogue warps
        self.mma_warp_id = 8
        self.tma_warp_id = 9
        self.tmb_warp_id = 10
        self.threads_per_cta = 32 * 11
        self.epi_bar_id = 1
        self.tmem_bar_id = 2

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
        tiled_mma1 = utils.sm100.make_trivial_tiled_mma(
            self.io_dtype,
            self.io_dtype,
            k_major,
            k_major,
            self.acc_dtype,
            self.cta_group,
            self.mma_tiler1[:2],
        )
        self.cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout((*self.cluster_shape_mn, 1)), (tiled_mma.thr_id.shape,)
        )
        self.a_smem_staged = utils.sm100.make_smem_layout_a(
            tiled_mma, self.mma_tiler, self.io_dtype, self.num_stages
        )
        self.b_smem_staged = utils.sm100.make_smem_layout_b(
            tiled_mma, self.mma_tiler, self.io_dtype, self.num_b_stages
        )
        self.b1_smem_staged = utils.sm100.make_smem_layout_b(
            tiled_mma1, self.mma_tiler1, self.io_dtype, self.num_b_stages
        )
        self.num_tmem_cols = 512
        return tiled_mma, tiled_mma1

    @cute.jit
    def __call__(
        self,
        m: cutlass.Int32,
        hc_ptr: cute.Pointer,
        w_ptr: cute.Pointer,
        rstd: cute.Tensor,
        b_ptr: cute.Pointer,
        out_ptr: cute.Pointer,
        inj_ptr: cute.Pointer,
        stream: cuda.CUstream,
    ):
        width = self.hc * self.hidden
        tiled_mma, tiled_mma1 = self._setup()
        a = cute.make_tensor(hc_ptr, cute.make_layout((m, width, 1), stride=(width, 1, 0)))
        b = cute.make_tensor(b_ptr, cute.make_layout((NH0, width, 1), stride=(width, 1, 0)))
        b1 = cute.make_tensor(
            b_ptr + NH0 * width, cute.make_layout((NH1, width, 1), stride=(width, 1, 0))
        )
        a_one = cute.slice_(self.a_smem_staged, (None, None, None, 0))
        # The raw A tile of a CTA: a plain 1-CTA TMA into the operand stage, transformed
        # in place.
        tma_atom_a, tma_a = cute.nvgpu.make_tiled_tma_atom_A(
            cpasync.CopyBulkTensorTileG2SOp(),
            a,
            a_one,
            self.mma_tiler,
            tiled_mma,
            self.cluster_layout_vmnk.shape,
        )
        b_op = utils.sm100.cluster_shape_to_tma_atom_B(self.cluster_shape_mn, tiled_mma.thr_id)
        b_one = cute.slice_(self.b_smem_staged, (None, None, None, 0))
        tma_atom_b, tma_b = cute.nvgpu.make_tiled_tma_atom_B(
            b_op, b, b_one, self.mma_tiler, tiled_mma, self.cluster_layout_vmnk.shape
        )
        b1_one = cute.slice_(self.b1_smem_staged, (None, None, None, 0))
        tma_atom_b1, tma_b1 = cute.nvgpu.make_tiled_tma_atom_B(
            b_op, b1, b1_one, self.mma_tiler1, tiled_mma1, self.cluster_layout_vmnk.shape
        )
        self.a_bytes = cute.size_in_bytes(self.io_dtype, a_one)
        # Both MMAs' B, both CTAs of the pair.
        self.b_bytes = (
            cute.size_in_bytes(self.io_dtype, b_one) + cute.size_in_bytes(self.io_dtype, b1_one)
        ) * 2
        pairs = (m + 255) // 256
        self.kernel(
            tiled_mma,
            tiled_mma1,
            tma_atom_a,
            tma_a,
            tma_atom_b,
            tma_b,
            tma_atom_b1,
            tma_b1,
            w_ptr.toint(),
            rstd.iterator.toint(),
            out_ptr.toint(),
            inj_ptr.toint(),
            m,
            self.cluster_layout_vmnk,
            self.a_smem_staged,
            self.b_smem_staged,
            self.b1_smem_staged,
        ).launch(
            grid=[2 * pairs * self.split, 1, 1],
            block=[self.threads_per_cta, 1, 1],
            cluster=(*self.cluster_shape_mn, 1),
            stream=stream,
            use_pdl=_PDL,
        )

    @cute.kernel
    def kernel(
        self,
        tiled_mma: cute.TiledMma,
        tiled_mma1: cute.TiledMma,
        tma_atom_a: cute.CopyAtom,
        mA_mkl: cute.Tensor,
        tma_atom_b: cute.CopyAtom,
        mB_nkl: cute.Tensor,
        tma_atom_b1: cute.CopyAtom,
        mB1_nkl: cute.Tensor,
        w_base: cutlass.Int64,
        rstd_base: cutlass.Int64,
        out_base: cutlass.Int64,
        inj_base: cutlass.Int64,
        m: cutlass.Int32,
        cluster_layout_vmnk: cute.Layout,
        a_smem_staged: cute.ComposedLayout,
        b_smem_staged: cute.ComposedLayout,
        b1_smem_staged: cute.ComposedLayout,
    ):
        hc, hid, rank, split = self.hc, self.hidden, self.rank, self.split
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        tidx, _, _ = cute.arch.thread_idx()
        bid, _, _ = cute.arch.block_idx()
        v = bid % 2
        is_leader = v == 0
        pair = bid // 2
        m_pair = pair // split
        sp = pair % split
        kt0 = sp * self.kt
        cta_rank = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        cluster_coord = cluster_layout_vmnk.get_flat_coord(cta_rank)
        if warp_idx == self.tma_warp_id:
            cpasync.prefetch_descriptor(tma_atom_a)
        if warp_idx == self.tmb_warp_id:
            cpasync.prefetch_descriptor(tma_atom_b)
            cpasync.prefetch_descriptor(tma_atom_b1)

        @cute.struct
        class SharedStorage:
            raw_full: cute.struct.MemRange[cutlass.Int64, self.num_stages]
            b_mbar: cute.struct.MemRange[cutlass.Int64, self.num_b_stages * 2]
            a_mbar: cute.struct.MemRange[cutlass.Int64, self.num_stages * 2]
            acc_mbar: cute.struct.MemRange[cutlass.Int64, 2]
            tmem_dealloc_mbar: cutlass.Int64
            tmem_holding_buf: cutlass.Int32

        smem = utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)
        # Raw tile landed (TMA transaction count), per CTA: the transform warps wait on it.
        raw_full = storage.raw_full.data_ptr()
        if warp_idx == self.tma_warp_id:
            with cute.arch.elect_one():
                for i in cutlass.range_constexpr(self.num_stages):
                    cute.arch.mbarrier_init(raw_full + i, 1)
        b_producer, b_consumer = PipelineTmaUmma.create(
            barrier_storage=storage.b_mbar.data_ptr(),
            num_stages=self.num_b_stages,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
            tx_count=self.b_bytes,
            cta_layout_vmnk=cluster_layout_vmnk,
        ).make_participants()
        # A operand stages: the TMA warp acquires a stage (the MMA has released it), the
        # transform warps of both CTAs commit it to the leader, the leader's MMA consumes
        # and releases it.
        a_pipe = pipeline.PipelineAsyncUmma.create(
            barrier_storage=storage.a_mbar.data_ptr(),
            num_stages=self.num_stages,
            producer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread, 2 * (self.tw // self.groups)
            ),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
            cta_layout_vmnk=cluster_layout_vmnk,
        )
        a_consumer = a_pipe.make_participants()[1]
        acc_pipeline = PipelineUmmaAsync.create(
            barrier_storage=storage.acc_mbar.data_ptr(),
            num_stages=1,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 2 * self.tw),
            cta_layout_vmnk=cluster_layout_vmnk,
        )
        tmem_alloc_barrier = pipeline.NamedBarrier(
            barrier_id=self.tmem_bar_id, num_threads=32 * (1 + self.tw)
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
            layout=a_smem_staged.outer,
            byte_alignment=1024,
            swizzle=a_smem_staged.inner,
        )
        sB = smem.allocate_tensor(
            element_type=self.io_dtype,
            layout=b_smem_staged.outer,
            byte_alignment=1024,
            swizzle=b_smem_staged.inner,
        )
        sB1 = smem.allocate_tensor(
            element_type=self.io_dtype,
            layout=b1_smem_staged.outer,
            byte_alignment=1024,
            swizzle=b1_smem_staged.inner,
        )

        gA_mkl = cute.local_tile(
            mA_mkl, cute.slice_(self.mma_tiler, (None, 0, None)), (None, None, None)
        )
        gB_nkl = cute.local_tile(
            mB_nkl, cute.slice_(self.mma_tiler, (0, None, None)), (None, None, None)
        )
        thr_mma = tiled_mma.get_slice(v)
        tCgA = thr_mma.partition_A(gA_mkl)
        tCgB = thr_mma.partition_B(gB_nkl)
        tAsA, tAgA = cpasync.tma_partition(
            tma_atom_a,
            0,
            cute.make_layout(1),
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
        gB1_nkl = cute.local_tile(
            mB1_nkl, cute.slice_(self.mma_tiler1, (0, None, None)), (None, None, None)
        )
        tCgB1 = tiled_mma1.get_slice(v).partition_B(gB1_nkl)
        tB1sB1, tB1gB1 = cpasync.tma_partition(
            tma_atom_b1,
            cluster_coord[1],
            b_cta_layout,
            cute.group_modes(sB1, 0, 3),
            cute.group_modes(tCgB1, 0, 3),
        )
        b_mcast = cpasync.create_tma_multicast_mask(
            cluster_layout_vmnk, cluster_coord, mcast_mode=1
        )
        tCrA = tiled_mma.make_fragment_A(sA)
        tCrB = tiled_mma.make_fragment_B(sB)
        tCrA1 = tiled_mma1.make_fragment_A(sA)
        tCrB1 = tiled_mma1.make_fragment_B(sB1)
        acc0_fake = tiled_mma.make_fragment_C(tiled_mma.partition_shape_C(self.mma_tiler[:2]))
        acc1_fake = tiled_mma1.make_fragment_C(tiled_mma1.partition_shape_C(self.mma_tiler1[:2]))

        pipeline_init_wait(cluster_shape_mn=self.cluster_shape_mn)

        # ------------------------------------------------------------ TMA warp (A, raw)
        if warp_idx == self.tma_warp_id:
            cute.arch.griddepcontrol_wait()
            tAgA_t = tAgA[(None, m_pair, None, 0)]
            a_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_stages
            )
            for j in cutlass.range(self.kt, unroll=1):
                ai = a_state.index
                a_pipe.producer_acquire(a_state)
                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive_and_expect_tx(raw_full + ai, self.a_bytes)
                cute.copy(
                    tma_atom_a, tAgA_t[(None, kt0 + j)], tAsA[(None, ai)], tma_bar_ptr=raw_full + ai
                )
                a_state.advance()

        # ------------------------------------------------------------ TMA warp (B)
        if warp_idx == self.tmb_warp_id:
            for j in cutlass.range(self.kt, unroll=1):
                handle = b_producer.acquire_and_advance()
                cute.copy(
                    tma_atom_b,
                    tBgB[(None, 0, kt0 + j, 0)],
                    tBsB[(None, handle.index)],
                    tma_bar_ptr=handle.barrier,
                    mcast_mask=b_mcast,
                )
                cute.copy(
                    tma_atom_b1,
                    tB1gB1[(None, 0, kt0 + j, 0)],
                    tB1sB1[(None, handle.index)],
                    tma_bar_ptr=handle.barrier,
                    mcast_mask=b_mcast,
                )
            b_producer.tail()

        # ------------------------------------------------------------ MMA warp
        if warp_idx == self.mma_warp_id:
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            tCtAcc0 = cute.make_tensor(tmem_ptr, acc0_fake.layout)
            tCtAcc1 = cute.make_tensor(tmem_ptr + NH0, acc1_fake.layout)
            acc_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, 1)
            if is_leader:
                acc_pipeline.producer_acquire(acc_state)
                tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
                tiled_mma1.set(tcgen05.Field.ACCUMULATE, False)
                for j in cutlass.range(self.kt, unroll=1):
                    b_handle = b_consumer.wait_and_advance()
                    a_handle = a_consumer.wait_and_advance()
                    num_kblocks = cute.size(tCrA, mode=[2])
                    for kk in cutlass.range(num_kblocks, unroll_full=True):
                        cute.gemm(
                            tiled_mma,
                            tCtAcc0,
                            tCrA[(None, None, kk, a_handle.index)],
                            tCrB[(None, None, kk, b_handle.index)],
                            tCtAcc0,
                        )
                        cute.gemm(
                            tiled_mma1,
                            tCtAcc1,
                            tCrA1[(None, None, kk, a_handle.index)],
                            tCrB1[(None, None, kk, b_handle.index)],
                            tCtAcc1,
                        )
                        tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                        tiled_mma1.set(tcgen05.Field.ACCUMULATE, True)
                    b_handle.release()
                    a_handle.release()
                acc_pipeline.producer_commit(acc_state)
            acc_state.advance()
            acc_pipeline.producer_tail(acc_state)

        # ------------------------------------------------------------ transform + epilogue
        if warp_idx < self.tw:
            cute.arch.griddepcontrol_wait()
            tmem.allocate(self.num_tmem_cols)
            # Before the transform loop: the MMA warp waits for this allocation, and the
            # MMA must run for the operand ring to drain.
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            # Transform mapping (128B-swizzled K-major 128 x 64 bf16 tile): within its group
            # of `gt` threads, thread t owns the 16-byte chunks at byte offsets 16 (t + gt j),
            # i.e. rows (t >> 3) + (gt / 8) j and the logical 8-column chunk
            # q = (t & 7) ^ ((t >> 3) & 7) of every row it owns. The tile is transformed in
            # place: same offsets in, same out.
            G = self.groups
            gt = 32 * self.tw // G
            nch = 1024 // gt  # chunks a thread
            grp = tidx // gt
            tg = tidx % gt
            q = (tg & 7) ^ ((tg >> 3) & 7)
            row0 = m_pair * 256 + v * 128 + (tg >> 3)
            op_addr = cutlass.Int32(sA.iterator.toint()) + tg * 16
            op_stage = cute.size_in_bytes(
                self.io_dtype, cute.slice_(a_smem_staged, (None, None, None, 0))
            )
            rows64 = [cutlass.Int64(cutlass.min(row0 + (gt // 8) * jj, m - 1)) for jj in range(nch)]
            kpb = hid // BK  # k blocks a stream
            S = self.num_stages
            # The group's next k block's scales are loaded one iteration ahead: rstd of the
            # thread's rows, 8 bf16 of the norm weight.
            rbuf = cute.make_rmem_tensor((nch,), cutlass.Float32)
            wbuf = cute.make_rmem_tensor((4,), cutlass.Int32)
            kfirst = kt0 + grp
            for jj in cutlass.range_constexpr(nch):
                rbuf[jj] = _ldg_f32(
                    rstd_base + (rows64[jj] * hc + cutlass.Int64(kfirst // kpb)) * 4
                )
            w0 = _ldg_v4(w_base + cutlass.Int64(kfirst * BK + q * 8) * 2, cutlass.Int32(1))
            for e in cutlass.range_constexpr(4):
                wbuf[e] = w0[e]
            for j in cutlass.range(grp, self.kt, G, unroll=1):
                r = [rbuf[jj] for jj in range(nch)]
                ww = [wbuf[e] for e in range(4)]
                if j + G < self.kt:
                    kn = kt0 + j + G
                    s1 = kn // kpb
                    for jj in cutlass.range_constexpr(nch):
                        rbuf[jj] = _ldg_f32(rstd_base + (rows64[jj] * hc + cutlass.Int64(s1)) * 4)
                    wn = _ldg_v4(w_base + cutlass.Int64(kn * BK + q * 8) * 2, cutlass.Int32(1))
                    for e in cutlass.range_constexpr(4):
                        wbuf[e] = wn[e]
                w1 = []
                for word in ww:
                    w1 += [_add1(_bf16_lo(word)), _add1(_bf16_hi(word))]
                ai = j % S
                cute.arch.mbarrier_wait(raw_full + ai, (j // S) % 2)
                base = op_addr + ai * op_stage
                for jj in cutlass.range_constexpr(nch):
                    words = _lds_v4(base + jj * (gt * 16))
                    out = []
                    for e in cutlass.range_constexpr(4):
                        # (x * rstd) * (1 + w), each product rounded in fp32, then bf16:
                        # `qwen4_exp_hc_inject_norm`'s normed, bitwise.
                        lo = _mul(_mul(_bf16_lo(words[e]), r[jj]), w1[2 * e])
                        hi = _mul(_mul(_bf16_hi(words[e]), r[jj]), w1[2 * e + 1])
                        out.append(_bf16x2(lo, hi))
                    _sts_v4(base + jj * (gt * 16), *out)
                cute.arch.fence_proxy("async.shared", space="cta")
                cute.arch.sync_warp()
                with cute.arch.elect_one():
                    a_pipe.sync_object_full.arrive(ai, a_pipe.producer_mask)

            # Epilogue: warp w reads TMEM lanes 32 (w % 4).. of accumulator w // 4
            # (columns [0, NH0) and [NH0, NH0 + NH1)).
            lt = tidx % 128
            half = tidx // 128
            row = m_pair * 256 + v * 128 + lt
            row64 = cutlass.Int64(row)
            acc_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, 1)
            acc_pipeline.consumer_wait(acc_state)
            inv = cutlass.Float32(1.0 / hc)
            epi_tile = (128, 16)
            for hh in cutlass.range_constexpr(2):
                if half == hh:
                    nhh = NH0 if hh == 0 else NH1
                    acc_t = cute.make_tensor(
                        tmem_ptr + hh * NH0, (acc0_fake if hh == 0 else acc1_fake).layout
                    )
                    copy_atom_t2r = utils.sm100.get_tmem_load_op(
                        (128, nhh, BK),
                        utils.LayoutEnum.ROW_MAJOR,
                        self.io_dtype,
                        self.acc_dtype,
                        epi_tile,
                        True,
                    )
                    tAcc_epi = cute.flat_divide(acc_t[((None, None), 0, 0)], epi_tile)
                    tiled_copy_t2r = tcgen05.make_tmem_copy(
                        copy_atom_t2r, tAcc_epi[(None, None, 0, 0)]
                    )
                    thr_copy_t2r = tiled_copy_t2r.get_slice(lt)
                    tTR_tAcc = thr_copy_t2r.partition_S(tAcc_epi)
                    tTR_cE = thr_copy_t2r.partition_D(cute.make_identity_tensor(epi_tile))
                    frag = cute.make_rmem_tensor(tTR_cE.shape, self.acc_dtype)
                    for c in cutlass.range_constexpr(nhh // 16):
                        col = hh * NH0 + c * 16
                        if cutlass.const_expr(col < rank + hc):
                            cute.copy(tiled_copy_t2r, tTR_tAcc[(None, None, None, 0, c)], frag)
                            if row < m:
                                if cutlass.const_expr(split > 1):
                                    pbase = (
                                        out_base
                                        + ((cutlass.Int64(sp) * m + row64) * NCOL + col) * 4
                                    )
                                    for qq in cutlass.range_constexpr(4):
                                        _st_global_v4(
                                            pbase + qq * 16,
                                            *[_bits(frag[4 * qq + e]) for e in range(4)],
                                        )
                                elif cutlass.const_expr(col + 16 <= rank):
                                    vals = []
                                    for e in cutlass.range_constexpr(16):
                                        z = frag[e] * inv
                                        vals.append(z * _sigmoid_fast(z))
                                    words = [
                                        _bf16x2(vals[2 * e], vals[2 * e + 1]) for e in range(8)
                                    ]
                                    obase = out_base + (row64 * rank + col) * 2
                                    _st_global_v4(obase, *words[0:4])
                                    _st_global_v4(obase + 16, *words[4:8])
                                else:
                                    # The chunk holding the injection logits (rank % 16 == 0).
                                    vals = []
                                    for e in cutlass.range_constexpr(hc):
                                        z = frag[e] * inv
                                        vals.append(cutlass.Float32(2.0) * _sigmoid_fast(z))
                                    _st_global_v2(
                                        inj_base + row64 * (hc * 2),
                                        _bf16x2(vals[0], vals[1]),
                                        _bf16x2(vals[2], vals[3]),
                                    )
            cute.arch.sync_warp()
            with cute.arch.elect_one():
                acc_pipeline.consumer_release(acc_state)
            cute.arch.barrier(barrier_id=self.epi_bar_id, number_of_threads=32 * self.tw)
            tmem.relinquish_alloc_permit()
            tmem.free(tmem_ptr)


@triton.jit
def _combine_act_kernel(
    part_ptr,
    low_ptr,
    inj_ptr,
    T,
    SPLIT: tl.constexpr,
    RANK: tl.constexpr,
    HC: tl.constexpr,
    NCOL: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # The K-split partial logits summed in split order (fp32), then the activations of
    # the split-1 epilogue (`tl.sigmoid` here, the ex2 / rcp form there).
    row = tl.program_id(0).to(tl.int64)
    k = tl.arange(0, BLOCK)
    mask = k < RANK + HC
    z = tl.load(part_ptr + row * NCOL + k, mask=mask, other=0.0)
    for sidx in tl.static_range(1, SPLIT):
        z += tl.load(part_ptr + (sidx * T + row) * NCOL + k, mask=mask, other=0.0)
    z = z * (1.0 / HC)
    tl.store(
        low_ptr + row * RANK + k, (z * tl.sigmoid(z)).to(low_ptr.dtype.element_ty), mask=k < RANK
    )
    ki = k - RANK
    tl.store(
        inj_ptr + row * HC + ki,
        (2.0 * tl.sigmoid(z)).to(inj_ptr.dtype.element_ty),
        mask=(ki >= 0) & (ki < HC),
    )


_COMPILED: dict = {}
_SMS: dict = {}


def _launcher(hc: int, hidden: int, rank: int, split: int, ptrs):
    """Compile once per static configuration; later calls reuse the TVM-FFI entry,
    which launches on the caller's current stream (graph-capturable). `ptrs` is
    `(hc_out, norm_weight, rstd tensor, first, out, inject)`.
    """
    key = (hc, hidden, rank, split)
    fn = _COMPILED.get(key)
    if fn is None:
        kern = HcFirstRescaleKernel(hc, hidden, rank, split)
        gm = cute.AddressSpace.gmem
        cptrs = [
            make_ptr(cutlass.BFloat16, ptrs[0], gm, assumed_align=16),
            make_ptr(cutlass.BFloat16, ptrs[1], gm, assumed_align=16),
            from_dlpack(ptrs[2], assumed_align=16).mark_layout_dynamic(leading_dim=1),
            make_ptr(cutlass.BFloat16, ptrs[3], gm, assumed_align=16),
            make_ptr(cutlass.BFloat16, ptrs[4], gm, assumed_align=16),
            make_ptr(cutlass.BFloat16, ptrs[5], gm, assumed_align=8),
        ]
        fn = cute.compile(
            kern,
            128,
            *cptrs,
            stream=cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
            options="--opt-level 2 --enable-tvm-ffi",
        )
        _COMPILED[key] = fn
    return fn


def _pick_split(rows: int, k_blocks: int, device: torch.device) -> int:
    """CTA pairs = ceil(rows / 256) * split: split K in two while the grid stays within
    one wave of SMs (and K divides), up to `_MAX_SPLIT`.
    """
    sms = _SMS.get(device.index)
    if sms is None:
        sms = _SMS[device.index] = torch.cuda.get_device_properties(device).multi_processor_count
    pairs = -(-rows // 256)
    split = 1
    while split < _MAX_SPLIT and 2 * pairs * split * 2 <= sms and k_blocks % (2 * split) == 0:
        split *= 2
    return split


def hc_first_rescale(
    hc_out: torch.Tensor,
    rstd: torch.Tensor,
    norm_weight: torch.Tensor,
    first_t: torch.Tensor,
    hc_count: int,
    rank: int,
    with_inject: bool,
):
    """`(low [T, rank], inject [T, C] or None)`; `first_t` the contiguous padded
    `[N, C*H]` first weight (N >= 352).
    """
    rows, width = hc_out.shape
    hidden = width // hc_count
    dev = hc_out.device
    low = torch.empty((rows, rank), dtype=hc_out.dtype, device=dev)
    # Written in every case (the kernel stores the injection chunk); dropped without
    # `with_inject`.
    inject = torch.empty((rows, hc_count), dtype=hc_out.dtype, device=dev)
    if rows > 0:
        split = _pick_split(rows, width // BK, dev)
        if split == 1:
            out = low
        else:
            out = torch.empty((split, rows, NCOL), dtype=torch.float32, device=dev)
        ptrs = (
            hc_out.data_ptr(),
            norm_weight.data_ptr(),
            rstd,
            first_t.data_ptr(),
            out.data_ptr(),
            inject.data_ptr(),
        )
        _launcher(hc_count, hidden, rank, split, ptrs)(rows, *ptrs)
        if split > 1:
            _combine_act_kernel[(rows,)](
                out,
                low,
                inject,
                rows,
                SPLIT=split,
                RANK=rank,
                HC=hc_count,
                NCOL=NCOL,
                BLOCK=triton.next_power_of_2(rank + hc_count),
                num_warps=4,
            )
    return low, (inject if with_inject else None)


class Qwen4ExpHcFirstRescaleTc(OpWrapper):
    """With `C = hc_count`, `H = width // C`, `first` the `[C*H, N]` transposed view
    of the padded `[N, C*H]` `[down; block_inject; 0]` weight (N >= 352)::

        normed = bf16(hc_out[t, s * H + i] * rstd[t, s] * (1 + norm_weight[s * H + i]))
        logits = normed @ first[:, : rank + C]  # fp32
        low = silu(logits[:, :rank] / C)  # [T, rank]
        inject = 2 * sigmoid(logits[:, rank : rank + C] / C)  # [T, C]

    `normed` is bitwise `qwen4_exp_hc_inject_norm`'s (the same fp32 products, one
    rounding), formed in shared memory and never written. Returns fresh contiguous
    `(low, inject)` in `hc_out`'s dtype; `inject` is `None` exactly when
    `with_inject` is false (the stack-final mixer; `low` is unchanged).

    Fusion boundary: the grouped norm's scaling, the first projection and both
    activations -- `cublas_mm(normed, first[:, :352])` plus
    `qwen4_exp_hc_lowrank_act` fed by a site that stored `normed`.
    """

    ARCHS = frozenset({Arch.SM_103})

    # spec: rows, inject (True). The checkpoint's geometry throughout: hc_count 4,
    # hidden 2560, rank 320, the padded 384-row first weight, bf16.
    CELLS: tuple[Cell, ...] = (
        Cell(
            why="prefill chunk 16384, one K slice a CTA pair",
            spec=dict(rows=16384),
        ),
        Cell(why="prefill 8192 rows, K split in two", spec=dict(rows=8192)),
        Cell(why="prefill 2048 rows, K split in eight", spec=dict(rows=2048)),
        Cell(
            why="stack-final mixer (no injection), prefill chunk 16384",
            spec=dict(rows=16384, inject=False),
        ),
        Cell(why="ragged tail, 16320 + 70 rows (mixed step)", spec=dict(rows=16390)),
        Cell(why="five rows, less than one tile", spec=dict(rows=5)),
    )

    note = """
    Tier 6 (CuTe DSL written for this target): a tcgen05 GEMM whose A operand is
    produced by warps of the same CTA from a second tensor; no lower tier has an
    operand prologue (cuBLAS needs `normed` materialized). Metadata consumed:
    none. Call sites: `PrefillTarget`'s trunk mixers at or above
    `_HC_RSTD_MIN_ROWS` rows (every attention / GDN, MoE and stack-final site),
    fed by `qwen4_exp_hc_inject_rstd`.

    Preconditions: bf16; `hc_out` `[T, C*H]` contiguous; rstd `[T, C]` fp32
    contiguous; `norm_weight` `[C*H]` contiguous; `first` the transposed view of
    a contiguous `[N, C*H]` weight, N >= 352; 16-byte aligned pointers;
    `hc_count == 4`, `H % 64 == 0`, `rank % 16 == 0`, `rank + C <= 352`.
    Compiles once per `(C, H, rank, split)` on first call; the row count is a
    runtime argument. Graph-capturable (current stream, no host sync; the split
    is a host function of the row count).

    Launch: one CTA pair a 256-row tile, the whole K (split 1), or, when that
    leaves more than half the SMs idle, `split` pairs a tile each over `1/split`
    of K (2048 rows: split 8, 8192: split 2, 16384: split 1 on 152 SMs) with a
    Triton combine. Launched with programmatic dependent launch
    (TRTLLM_ENABLE_PDL=0 turns it off): the weight stream starts before
    `griddepcontrol.wait`. GB300, 16384 rows: 82 us, against cuBLAS 77.6 +
    lowrank_act 9.3 on a stored `normed` -- the site's saving is the `normed`
    write and re-read this removes upstream (`qwen4_exp_hc_inject_rstd`).

    Numerics: `low` / `inject` are one fp32 activation of fp32 logits rounded
    once (split 1: `ex2.approx` / `rcp.approx` sigmoid; split > 1: the partials
    summed in split order, `tl.sigmoid`). Measured against the cuBLAS +
    `qwen4_exp_hc_lowrank_act` path on the same `normed`: split 1 bitwise at
    2048-16390 rows (where cuBLAS does not split K itself); otherwise the last
    bit differs on about 0.5% of `low`, inside the same band.

    Certified: bf16 at the checkpoint's 4 x 2560, rank 320: 16384 (split 1),
    8192 (split 2), 2048 (split 8), 5 rows, a partial last tile (16390), the
    stack-final form; graph replay. Against the float64 product of the
    once-rounded `normed`, gate band (`rtol = atol / max|ref| = eps`).

    Not certified: zero rows, hc_count other than 4, hidden other than 2560,
    rank other than 320.
    """

    def __call__(
        self,
        hc_out: torch.Tensor,
        rstd: torch.Tensor,
        norm_weight: torch.Tensor,
        first: torch.Tensor,
        hc_count: int,
        rank: int,
        with_inject: bool,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        return hc_first_rescale(hc_out, rstd, norm_weight, first.t(), hc_count, rank, with_inject)

    def reference(
        self,
        hc_out: torch.Tensor,
        rstd: torch.Tensor,
        norm_weight: torch.Tensor,
        first: torch.Tensor,
        hc_count: int,
        rank: int,
        with_inject: bool,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """float64: `normed` rounded once to `hc_out`'s dtype (the op's operand),
        then the projection and both activations, each output rounded once.
        """
        rows, width = hc_out.shape
        hidden = width // hc_count
        x = hc_out.double().unflatten(-1, (hc_count, hidden)) * rstd.double().unsqueeze(-1)
        normed = (x.flatten(-2) * (1.0 + norm_weight.double())).to(hc_out.dtype)
        logits = F.linear(normed.double(), first[:, : rank + hc_count].t().double())
        low = F.silu(logits[:, :rank] / hc_count).to(hc_out.dtype)
        inject = (2.0 * torch.sigmoid(logits[:, rank:] / hc_count)).to(hc_out.dtype)
        return low, (inject if with_inject else None)

    def is_valid(
        self,
        hc_out: torch.Tensor,
        rstd: torch.Tensor,
        norm_weight: torch.Tensor,
        first: torch.Tensor,
        hc_count: int,
        rank: int,
        with_inject: bool,
    ) -> None:
        rows, width = hc_out.shape
        assert hc_out.dtype == norm_weight.dtype == first.dtype == torch.bfloat16, "bf16"
        assert hc_out.stride() == (width, 1), "hc_out: contiguous rows"
        assert rstd.dtype == torch.float32 and rstd.shape == (rows, hc_count), "rstd [T, C] fp32"
        assert rstd.is_contiguous(), "rstd contiguous"
        assert norm_weight.shape == (width,) and norm_weight.is_contiguous(), "norm_weight [C*H]"
        assert first.dim() == 2 and first.shape[0] == width and first.shape[1] >= NCOL, (
            f"first [C*H, N >= {NCOL}]"
        )
        assert first.stride() == (1, width), "first must be the transposed view of [N, C*H]"
        # The epilogue stores the injection chunk as two bf16x2 words (C == 4) and
        # reads its 16-column chunks wholly as `low` or as injection.
        assert hc_count == 4, "hc_count 4"
        assert (width // hc_count) % BK == 0, f"hidden must be a multiple of {BK}"
        assert rank % 16 == 0 and rank + hc_count <= NCOL, "rank % 16 == 0, rank + C <= 352"
        assert isinstance(with_inject, bool)
        for t in (hc_out, norm_weight, first, rstd):
            assert t.data_ptr() % 16 == 0, "16-byte aligned operands"


qwen4_exp_hc_first_rescale_tc = Qwen4ExpHcFirstRescaleTc()
