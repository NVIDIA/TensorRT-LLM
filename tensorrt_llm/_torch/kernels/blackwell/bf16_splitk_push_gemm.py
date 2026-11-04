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
"""The low-M BF16 split-K GEMM (``dense_bf16_gemm_splitk``, vendored from FlashInfer) with an
all-reduce push epilogue (SM100 / SM103 / SM107).

``out[M, N] = A[M, K] @ W[N, K]^T`` for a TP partial sum that an MNNVL all-reduce reduces next:
instead of storing its BF16 tile, the owner CTA of each cluster (after the split-K reduction)
stages the tile token-major in shared memory and stores it as 16-byte vectors into the Lamport
slots of the call (the ``mnnvl_pushed_allreduce.push_slot`` layout: every rank's slot through the
multicast address for the one-shot transport, the token owner's scatter slot for the two-shot
one; ``-0`` sent as ``+0``). ``AllReduce.finish_push`` of the target then gives the all-reduce of
the GEMM output (with its fusion), bit for bit equal to this GEMM's plain output followed by the
all-reduce.

PDL: the weight warp streams before ``griddepcontrol.wait`` as in the plain kernel, but the
dependents are released by the activation warp right after its wait (not by the weight warp): the
finish kernel reads the workspace's rotation state as soon as it is resident, which is safe only
once every earlier grid has completed. The epilogue reads the state after its own wait.

Rows: up to :data:`MAX_M` = 64 with ``mma_n`` up to 64 (FlashInfer's own policy stops at 32 rows
/ ``mma_n`` 32). :func:`run_gemm` runs the same kernel without the push (plain store).
"""

from __future__ import annotations

import dataclasses
import functools
from typing import TYPE_CHECKING

import cuda.bindings.driver as _cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils.blackwell_helpers as sm100_utils
import torch as _torch
from cutlass import Int32, Int64
from cutlass.cute import experimental as cute_ext
from cutlass.cutlass_dsl import T, dsl_user_op

from .dense_bf16_gemm_splitk import AB_BUFFER_ALIGN_BYTES as _AB_BUFFER_ALIGN_BYTES
from .dense_bf16_gemm_splitk import MAILBOX_ALIGN_BYTES as _MAILBOX_ALIGN_BYTES
from .dense_bf16_gemm_splitk import MBARRIER_BYTES as _MBARRIER_BYTES
from .dense_bf16_gemm_splitk import OWNER_RANK, SplitKDenseGemmKernel, SplitKTactic, validate_tactic
from .dense_bf16_gemm_splitk import SMEM_CAPACITY_BYTES as _SMEM_CAPACITY_BYTES
from .dense_bf16_gemm_splitk import TMEM_POINTER_BYTES as _TMEM_POINTER_BYTES
from .dense_bf16_gemm_splitk import align_up as _align_up
from .dense_bf16_gemm_splitk import from_dlpack_dynamic as _from_dlpack_dynamic
from .dense_bf16_gemm_splitk import smem_bytes as _smem_bytes
from .dense_bf16_gemm_splitk import store_shared_remote_v4 as _store_shared_remote_v4
from .raw_ptx import asm as _asm
from .raw_ptx import iv as _iv
from .raw_ptx import ld_nc_u64 as _ld_nc_u64
from .raw_ptx import ld_volatile_v4, st_v4
from .raw_ptx import multimem_st_v4 as _multimem_st_v4
from .raw_ptx import sanitize_bf16x2 as _sanitize_bf16x2
from .raw_ptx import unpack as _unpack

if TYPE_CHECKING:
    from ...distributed.ops import LamportWorkspaceView

#: Alignment of the token-major staging tile.
_STAGE_ALIGN_BYTES = 16


class _NoopKernel:
    """An empty ``cute.compile`` kernel (see :func:`ensure_cute_runtime`)."""

    @cute.jit
    def __call__(self, stream: _cuda.CUstream):
        self.kernel().launch(grid=(1, 1, 1), block=(32, 1, 1), stream=stream)

    @cute.kernel
    def kernel(self):
        pass


@functools.cache
def ensure_cute_runtime() -> None:
    """Load and run one ``cute.compile`` module in this process. With the image's cutlass-dsl, the
    first ``cute.experimental`` module a process loads fails with ``cudaErrorInvalidValue`` unless
    a ``cute.compile`` module was loaded before; this makes the split-K kernels safe to be first."""
    stream = _cuda.CUstream(_torch.cuda.current_stream().cuda_stream)
    cute.compile(_NoopKernel(), stream)(stream)


def smem_bytes(tactic: SplitKTactic) -> int:
    """Shared memory of the push kernel: the plain kernel's plus the staging tile."""
    return (
        _align_up(_smem_bytes(tactic, tactic.ab_stages), _STAGE_ALIGN_BYTES)
        + tactic.mma_m * tactic.mma_n * 2
    )


#: Largest row count and the UMMA N tiles (rows per CTA) of the push kernel.
MAX_M = 64
_MMA_N = (8, 16, 32, 64)


def validate_push_tactic(tactic: SplitKTactic, m: int, n: int, k: int) -> None:
    """Reject a tactic the push kernel cannot run for ``x [m, k] @ w[n, k]^T``."""
    if m > 32 or tactic.mma_n > 32:
        # The vendored rules, with the row bound and UMMA N widened.
        if not 1 <= m <= MAX_M or tactic.mma_n not in _MMA_N:
            raise ValueError(f"unsupported rows {m} / mma_n {tactic.mma_n}")
        validate_tactic(dataclasses.replace(tactic, mma_n=32), 1, n, k)
        if _smem_bytes(tactic, tactic.ab_stages) > _SMEM_CAPACITY_BYTES:
            raise ValueError(f"tactic {tactic} exceeds shared memory")
    else:
        validate_tactic(tactic, m, n, k)
    if smem_bytes(tactic) > _SMEM_CAPACITY_BYTES:
        raise ValueError(f"tactic {tactic} with the staging tile exceeds shared memory")
    if tactic.mma_m % 8:
        raise ValueError("the push epilogue stores 8-column vectors")


@functools.cache
def _default_push_tactic(mma_n: int, n: int, k: int):
    for ab_stages in range(12, 0, -1):
        tactic = SplitKTactic(64, mma_n, 1, ab_stages)
        try:
            validate_push_tactic(tactic, mma_n, n, k)
        except ValueError:
            continue
        return tactic
    return None


def default_push_tactic(m: int, n: int, k: int):
    """The push GEMM's tactic for ``m`` rows: no K split, the UMMA N tile covering the rows, the
    deepest pipeline that fits; None above :data:`MAX_M` or when none fits."""
    if not 1 <= m <= MAX_M:
        return None
    return _default_push_tactic(max(8, 1 << (m - 1).bit_length()), n, k)


@dsl_user_op
def _lds_v4(addr, *, loc=None, ip=None):
    from cutlass._mlir.dialects import llvm

    res = _asm(
        llvm.StructType.get_literal([T.i32()] * 4),
        [_iv(addr, loc=loc, ip=ip)],
        "ld.shared.v4.b32 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,r",
        loc=loc,
        ip=ip,
    )
    return _unpack(res, T.i32(), Int32, 4, loc=loc, ip=ip)


def _lamport_cur(flags):
    """``(current buffer, bytes per buffer)`` of the workspace's rotation state."""
    f = ld_volatile_v4(flags)
    return f[0], f[2]


def _push_vector(peer_ptrs, mc_base, cur, nbytes, tok, rank, col, words, world, hidden, one_shot):
    """Store eight BF16 values (``-0`` as ``+0``) of row ``tok`` at column ``col`` into this
    rank's slot of the call in buffer ``cur`` (``mnnvl_pushed_allreduce.push_slot`` /
    ``push_v4``)."""
    w = [_sanitize_bf16x2(x) for x in words]
    row_bytes = hidden * 2
    if one_shot:
        addr = (
            mc_base
            + Int64(cur) * Int64(nbytes)
            + Int64(tok * world + rank) * row_bytes
            + Int64(col) * 2
        )
        _multimem_st_v4(addr, w[0], w[1], w[2], w[3])
    else:
        owner = _ld_nc_u64(peer_ptrs + Int64(tok % world) * 8)
        addr = (
            owner
            + Int64(cur * 2) * Int64(nbytes // 2)
            + Int64((tok // world) * world + rank) * row_bytes
            + Int64(col) * 2
        )
        st_v4(addr, w[0], w[1], w[2], w[3])


class SplitKArPushGemmKernel(SplitKDenseGemmKernel):
    """:class:`SplitKDenseGemmKernel` whose owner CTAs push the BF16 tile into the Lamport slots of
    an all-reduce of ``world`` ranks over ``hidden`` (= N) columns (see the module docstring).
    ``store_out`` also stores the tile into ``out`` as the plain kernel does; without ``push`` the
    kernel only stores (no workspace access)."""

    def __init__(
        self,
        *,
        tactic: SplitKTactic,
        use_pdl: bool,
        world: int,
        hidden: int,
        one_shot: bool,
        store_out: bool,
        push: bool = True,
    ) -> None:
        super().__init__(tactic=tactic, use_pdl=use_pdl, has_bias=False)
        self.world = world
        self.hidden = hidden
        self.one_shot = one_shot
        self.store_out = store_out
        self.push = push
        self.vecs_per_token = tactic.mma_m // 8

    @cute.experimental.jit
    def __call__(
        self,
        a: cute.Tensor,
        b: cute.Tensor,
        c: cute.Tensor,
        peer_ptrs: Int64,
        mc_base: Int64,
        flags: Int64,
        rank: Int32,
        m: Int32,
        stream: _cuda.CUstream,
    ):
        self.kernel(a, b, c, peer_ptrs, mc_base, flags, rank, m).launch(
            grid=(
                cute.ceil_div(c.layout.shape[0], self.cta_m),
                cute.ceil_div(c.layout.shape[1], self.cta_n) * self.split_k,
                c.layout.shape[2],
            ),
            block=(self.threads_per_cta, 1, 1),
            cluster=self.cluster_shape,
            smem=cute.Int64(_SMEM_CAPACITY_BYTES),
            stream=stream,
            use_pdl=self.use_pdl,
        )

    @cute.experimental.kernel
    def kernel(
        self,
        mA: cute.Tensor,
        mB: cute.Tensor,
        mC: cute.Tensor,
        peer_ptrs: Int64,
        mc_base: Int64,
        flags: Int64,
        rank: Int32,
        m: Int32,
    ):
        """The plain kernel's warp dispatch plus the staging tile and the push arguments."""
        stages = self.num_ab_stage
        ab_dtype = mA.element_type
        tiled_mma = sm100_utils.make_trivial_tiled_mma(
            ab_dtype,
            ab_dtype,
            cutlass.utils.LayoutEnum.from_tensor(mA).mma_major_mode(),
            cutlass.utils.LayoutEnum.from_tensor(mB).mma_major_mode(),
            self.acc_dtype,
            self.cta_group,
            self.mma_tiler_mn,
        )
        mnk_tiler = (self.mma_tiler_mn[0], self.mma_tiler_mn[1], self.cta_k)
        block_idx = cute.arch.block_idx()
        bidx = block_idx[0]
        split_rank = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        n_idx = block_idx[1] // self.split_k
        l_idx = block_idx[2]
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())

        sA = cute_ext.allocate(
            ab_dtype,
            cute.AddressSpace.smem,
            sm100_utils.make_smem_layout_a(tiled_mma, mnk_tiler, ab_dtype, stages),
            alignment=_AB_BUFFER_ALIGN_BYTES,
        )
        sB = cute_ext.allocate(
            ab_dtype,
            cute.AddressSpace.smem,
            sm100_utils.make_smem_layout_b(tiled_mma, mnk_tiler, ab_dtype, stages),
            alignment=_AB_BUFFER_ALIGN_BYTES,
        )
        acc_layout = cute_ext.make_tmem_layout_acc(tiled_mma, self.mma_tiler_mn, acc_stage=1)
        c_tiler_mn = (self.cta_m, self.cta_n)

        bar_full = cute_ext.allocate(
            cutlass.Int64,
            cute.AddressSpace.smem,
            cute.make_layout(stages),
            alignment=_MBARRIER_BYTES,
        ).iterator
        bar_empty = cute_ext.allocate(
            cutlass.Int64,
            cute.AddressSpace.smem,
            cute.make_layout(stages),
            alignment=_MBARRIER_BYTES,
        ).iterator
        bar_tma_epilog = cute_ext.allocate(
            cutlass.Int64,
            cute.AddressSpace.smem,
            cute.make_layout(1),
            alignment=_MBARRIER_BYTES,
        ).iterator
        bar_mma_epilog = cute_ext.allocate(
            cutlass.Int64,
            cute.AddressSpace.smem,
            cute.make_layout(1),
            alignment=_MBARRIER_BYTES,
        ).iterator
        bar_tmem_alloc = cute_ext.allocate(
            cutlass.Int64,
            cute.AddressSpace.smem,
            cute.make_layout(1),
            alignment=_MBARRIER_BYTES,
        ).iterator
        tmem_base_ptr = cute_ext.allocate(
            cutlass.Int32,
            cute.AddressSpace.smem,
            cute.make_layout(1),
            alignment=_TMEM_POINTER_BYTES,
        ).iterator
        if cutlass.const_expr(self.split_k > 1):
            mailbox = cute_ext.allocate(
                cutlass.Float32,
                cute.AddressSpace.smem,
                cute.make_layout(self.mailbox_elements),
                alignment=_MAILBOX_ALIGN_BYTES,
            )
            bar_reduce = cute_ext.allocate(
                cutlass.Int64,
                cute.AddressSpace.smem,
                cute.make_layout(1),
                alignment=_MBARRIER_BYTES,
            ).iterator
        else:
            mailbox = sA
            bar_reduce = bar_mma_epilog
        # The owner's BF16 tile, token-major: (hidden column, token) -> column + token * cta_m.
        stage = cute_ext.allocate(
            mC.element_type,
            cute.AddressSpace.smem,
            cute.make_layout((self.cta_m, self.cta_n), stride=(1, self.cta_m)),
            alignment=_STAGE_ALIGN_BYTES,
        )

        if warp_idx == 0:
            with cute.arch.elect_one():
                for i in range(stages):
                    cute.arch.mbarrier_init(bar_full + i, 2)
                    cute.arch.mbarrier_init(bar_empty + i, 1)
                cute.arch.mbarrier_init(bar_tma_epilog, 32)
                cute.arch.mbarrier_init(bar_mma_epilog, 1)
                cute.arch.mbarrier_init(bar_tmem_alloc, 160)
                if cutlass.const_expr(self.split_k > 1):
                    cute.arch.mbarrier_init(bar_reduce, 1)
        cute.arch.mbarrier_init_fence()
        if cutlass.const_expr(self.split_k > 1):
            cute.arch.cluster_arrive_relaxed()
        else:
            cute.arch.barrier()

        k_tile_count = cute.size(mA, mode=[1]) // self.cta_k // self.split_k
        k_tile_start = split_rank * k_tile_count
        if cutlass.const_expr(self.split_k > 1):
            cute.arch.cluster_wait()

        if warp_idx == 0:
            self.dma_warp(
                bar_full,
                bar_empty,
                bar_tma_epilog,
                cute.local_tile(mA, (self.cta_m, self.cta_k), (bidx, None, l_idx)),
                sA,
                cute_ext.get_cta_v_map_ab(mA, mnk_tiler, tiled_mma, "A"),
                k_tile_start,
                k_tile_count,
                True,
            )
        elif warp_idx == 1:
            self.dma_warp(
                bar_full,
                bar_empty,
                bar_tma_epilog,
                cute.local_tile(mB, (self.cta_n, self.cta_k), (n_idx, None, l_idx)),
                sB,
                cute_ext.get_cta_v_map_ab(mB, mnk_tiler, tiled_mma, "B"),
                k_tile_start,
                k_tile_count,
                False,
            )
        elif warp_idx == 2:
            self.mma_warp(
                bar_full,
                bar_empty,
                bar_mma_epilog,
                bar_tmem_alloc,
                tiled_mma,
                sA,
                sB,
                tmem_base_ptr,
                acc_layout,
                self.cta_k // cute.size(tiled_mma.shape_mnk, mode=[2]),
                k_tile_count,
            )
        elif warp_idx >= 4:
            self.push_epilog_warp(
                bar_mma_epilog,
                bar_tmem_alloc,
                tmem_base_ptr,
                acc_layout,
                cute.local_tile(mC, c_tiler_mn, (bidx, n_idx, l_idx)),
                cute.arch.thread_idx()[0] - 128,
                mC.element_type,
                cutlass.utils.LayoutEnum.from_tensor(mC),
                mailbox,
                bar_reduce,
                split_rank,
                stage,
                bidx,
                n_idx,
                peer_ptrs,
                mc_base,
                flags,
                rank,
                m,
            )

    @cute.experimental.jit
    def dma_warp(
        self,
        bar_full,
        bar_empty,
        bar_tma_epilog,
        g_tile: cute.Tensor,
        s_tile: cute.Tensor,
        cta_v_map: cute.Layout,
        k_tile_start: cutlass.Int32,
        k_tile_count: cutlass.Int32,
        is_a: cutlass.Constexpr,
    ):
        """The plain kernel's TMA producer; the activation warp releases the dependents right
        after its ``griddepcontrol.wait`` and the weight warp never does."""
        stages = self.num_ab_stage
        if cutlass.const_expr(not is_a and self.use_pdl):
            cute.arch.griddepcontrol_wait()
            cute.arch.griddepcontrol_launch_dependents()

        empty_phase = cutlass.Int32(1)
        for k_tile in cutlass.range(k_tile_count, unroll=1):
            stage = k_tile % stages
            cute.arch.mbarrier_wait(bar_empty + stage, empty_phase)
            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive_and_expect_tx(
                    bar_full + stage,
                    cute.size_in_bytes(
                        s_tile.element_type,
                        cute.slice_(s_tile.layout, (None, None, None, 0)),
                    ),
                )
            cute_ext.tma_load(
                g_tile[None, None, k_tile_start + k_tile],
                s_tile[None, None, None, stage],
                (bar_full + stage).value,
                cta_v_map=cta_v_map,
                tma_operation_type=self.tma_op,
                update_expect_tx=False,
            )
            if stage == stages - 1:
                empty_phase = empty_phase ^ 1
        self._drain_producer(bar_empty, empty_phase, k_tile_count)

    @cute.experimental.jit
    def push_epilog_warp(
        self,
        bar_mma_epilog,
        bar_tmem_alloc,
        tmem_base_ptr,
        acc_layout: cutlass.Constexpr,
        gD_tile: cute.Tensor,
        epi_tid: cutlass.Int32,
        c_dtype: cutlass.Constexpr,
        d_layout: cutlass.Constexpr,
        mailbox,
        bar_reduce,
        split_rank: cutlass.Int32,
        stage: cute.Tensor,
        bidx: cutlass.Int32,
        n_idx: cutlass.Int32,
        peer_ptrs: Int64,
        mc_base: Int64,
        flags: Int64,
        rank: Int32,
        m: Int32,
    ):
        cute.arch.mbarrier_arrive(bar_tmem_alloc)
        cute.arch.mbarrier_wait(bar_tmem_alloc, 0)
        acc_view = cute.make_tensor(
            cute.arch.retrieve_tmem_ptr(self.acc_dtype, 16, tmem_base_ptr),
            acc_layout,
        )[((None, None), 0, 0, 0)]
        epi_tile = (self.cta_m, self.cta_n)
        tiled_copy_t2r = cute.nvgpu.tcgen05.make_tmem_copy(
            sm100_utils.get_tmem_load_op(
                (self.cta_m, self.cta_n, self.cta_k),
                d_layout,
                c_dtype,
                self.acc_dtype,
                epi_tile,
                False,
            ),
            acc_view,
        )
        gD_epi = cute.flat_divide(gD_tile, epi_tile)
        rmem_layout = cute_ext.make_t2r_rmem_layout(tiled_copy_t2r, gD_epi, epi_tid)
        rAcc = cute_ext.allocate(self.acc_dtype, cute.AddressSpace.rmem, rmem_layout, alignment=32)
        rD = cute_ext.allocate(c_dtype, cute.AddressSpace.rmem, rmem_layout, alignment=32)
        thr_t2r = tiled_copy_t2r.get_slice(epi_tid)

        cute.arch.mbarrier_wait(bar_mma_epilog, 0)
        cute_ext.partition_and_copy(thr_t2r, acc_view, rAcc)
        cute.arch.fence_view_async_tmem_load()
        cute.arch.mbarrier_arrive(bar_tmem_alloc)

        if cutlass.const_expr(self.split_k > 1):
            values_per_thread = cutlass.const_expr(cute.size(rmem_layout))
            values_per_peer = cutlass.const_expr(self.epilog_threads * values_per_thread)
            if split_rank != OWNER_RANK:
                for value_idx in cutlass.range_constexpr(0, values_per_thread, 4):
                    _store_shared_remote_v4(
                        rAcc[value_idx],
                        rAcc[value_idx + 1],
                        rAcc[value_idx + 2],
                        rAcc[value_idx + 3],
                        mailbox.iterator
                        + (split_rank - Int32(1)) * values_per_peer
                        + epi_tid * values_per_thread
                        + value_idx,
                        bar_reduce,
                        Int32(OWNER_RANK),
                    )
            else:
                if epi_tid == 0:
                    cute.arch.mbarrier_arrive_and_expect_tx(
                        bar_reduce, self.expected_transaction_bytes
                    )
                cute.arch.mbarrier_wait(bar_reduce, 0)
                for peer in cutlass.range_constexpr(self.split_k - 1):
                    for value_idx in cutlass.range_constexpr(values_per_thread):
                        rAcc[value_idx] = (
                            rAcc[value_idx]
                            + mailbox[
                                peer * values_per_peer + epi_tid * values_per_thread + value_idx
                            ]
                        )

        if split_rank == OWNER_RANK:
            rD.store(rAcc.load().to(c_dtype))
            if cutlass.const_expr(self.store_out):
                cute_ext.partition_and_copy(thr_t2r, rD, gD_epi[None, None, 0, 0])
            if cutlass.const_expr(self.push):
                cute_ext.partition_and_copy(thr_t2r, rD, stage)
                # The four epilogue warps only.
                cute.arch.barrier(barrier_id=1, number_of_threads=self.epilog_threads)
                # This epilogue thread's 16-byte vectors of the tile, token-major.
                if cutlass.const_expr(self.use_pdl):
                    cute.arch.griddepcontrol_wait()
                cur, nbytes = _lamport_cur(flags)
                stage_base = stage.iterator.toint()
                vecs = self.cta_n * self.vecs_per_token
                for it in cutlass.range_constexpr(
                    (vecs + self.epilog_threads - 1) // self.epilog_threads
                ):
                    v = epi_tid + it * self.epilog_threads
                    t = v // self.vecs_per_token
                    col = (v % self.vecs_per_token) * 8
                    tok = n_idx * self.cta_n + t
                    if v < vecs:
                        if tok < m:
                            _push_vector(
                                peer_ptrs,
                                mc_base,
                                cur,
                                nbytes,
                                tok,
                                rank,
                                bidx * self.cta_m + col,
                                _lds_v4(stage_base + (t * self.cta_m + col) * 2),
                                self.world,
                                self.hidden,
                                self.one_shot,
                            )


@cute.experimental.jit
def _bmm_push(
    gemm_op: cutlass.Constexpr,
    a: cute.Tensor,
    b: cute.Tensor,
    c: cute.Tensor,
    peer_ptrs: Int64,
    mc_base: Int64,
    flags: Int64,
    rank: Int32,
    m: Int32,
    stream: _cuda.CUstream,
):
    gemm_op(
        cute.make_tensor(a.iterator, cute.select(a.layout, mode=[1, 2, 0])),
        cute.make_tensor(b.iterator, cute.select(b.layout, mode=[2, 1, 0])),
        cute.make_tensor(c.iterator, cute.select(c.layout, mode=[1, 2, 0])),
        peer_ptrs,
        mc_base,
        flags,
        rank,
        m,
        stream,
    )


def _make_layout_tensor(shape, dtype, leading_dim):
    permutation = [dim for dim in range(len(shape)) if dim != leading_dim] + [leading_dim]
    return _torch.empty(
        tuple(shape[dim] for dim in permutation), dtype=dtype, device="cuda"
    ).permute([permutation.index(dim) for dim in range(len(shape))])


@functools.cache
def _compiled(dtype, tactic, use_pdl, world, hidden, one_shot, store_out, push=True):
    kernel = SplitKArPushGemmKernel(
        tactic=tactic,
        use_pdl=use_pdl,
        world=world,
        hidden=hidden,
        one_shot=one_shot,
        store_out=store_out,
        push=push,
    )
    # The swapped operands of ``x [M, K] @ w[N, K]^T``: A = w (batch, N, K) K-major,
    # B = x^T (batch, K, M) K-major, C = out^T (batch, N, M) N-major.
    m, n, k = 8, 64, 128
    tensors = tuple(
        _from_dlpack_dynamic(_make_layout_tensor(shape, dtype, leading), leading)
        for shape, leading in (((1, n, k), 2), ((1, k, m), 1), ((1, n, m), 1))
    )
    stream = _cuda.CUstream(_torch.cuda.current_stream().cuda_stream)
    a = Int64(1 << 20)
    return cute_ext.compile(_bmm_push, kernel, *tensors, a, a, a, Int32(0), Int32(1), stream)


@functools.cache
def _unstored_out(n: int, dtype: _torch.dtype, device: _torch.device) -> _torch.Tensor:
    """``[MAX_M, n]`` storage for the C operand of the push variant, which never stores ``out``
    (the operand only carries the output's shape)."""
    return _torch.empty((MAX_M, n), dtype=dtype, device=device)


def _swapped(x, weight, out):
    return (
        _from_dlpack_dynamic(weight.unsqueeze(0), 2),
        _from_dlpack_dynamic(x.unsqueeze(0).transpose(-2, -1), 1),
        _from_dlpack_dynamic(out.unsqueeze(0).transpose(-2, -1), 1),
    )


def run_gemm(
    x: _torch.Tensor,
    weight: _torch.Tensor,
    out: _torch.Tensor,
    tactic: SplitKTactic,
    *,
    pdl: bool = True,
) -> _torch.Tensor:
    """``out = x [M, K] @ weight[N, K]^T`` (BF16, contiguous) with this kernel's store-only
    variant: the same values the push variant pushes, for ``M`` up to :data:`MAX_M`. Returns
    ``out``."""
    m, k = x.shape
    n = weight.shape[0]
    validate_push_tactic(tactic, m, n, k)
    ensure_cute_runtime()
    _compiled(x.dtype, tactic, pdl, 2, n, True, True, False)(
        *_swapped(x, weight, out),
        Int64(0),
        Int64(0),
        Int64(0),
        Int32(0),
        Int32(m),
        _cuda.CUstream(_torch.cuda.current_stream(x.device).cuda_stream),
    )
    return out


def run_gemm_push(
    x: _torch.Tensor,
    weight: _torch.Tensor,
    view: LamportWorkspaceView,
    one_shot: bool,
    tactic: SplitKTactic,
    *,
    pdl: bool = True,
) -> None:
    """Push ``x [M, K] @ weight[N, K]^T`` (BF16, contiguous) into the Lamport slots of the next
    all-reduce on ``view`` with the ``one_shot`` (else two-shot) transport; the target's
    ``AllReduce.finish_push`` completes it."""
    from .mnnvl_pushed_allreduce import check_transport

    m, k = x.shape
    n = weight.shape[0]
    validate_push_tactic(tactic, m, n, k)
    check_transport(view, m, n, one_shot)
    ensure_cute_runtime()
    out = _unstored_out(n, x.dtype, x.device)[:m]
    compiled = _compiled(x.dtype, tactic, pdl, view.world, n, one_shot, False)
    compiled(
        *_swapped(x, weight, out),
        Int64(view.peer_ptrs.data_ptr()),
        Int64(view.mc_base),
        Int64(view.flags.data_ptr()),
        Int32(view.rank),
        Int32(m),
        _cuda.CUstream(_torch.cuda.current_stream(x.device).cuda_stream),
    )


__all__ = [
    "MAX_M",
    "default_push_tactic",
    "SplitKArPushGemmKernel",
    "ensure_cute_runtime",
    "run_gemm",
    "run_gemm_push",
    "smem_bytes",
    "validate_push_tactic",
]
