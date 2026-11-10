# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A DFlash 2 convolution's ``prepare`` in one launch: the coefficient GEMM with the input-side
block convolution in its epilogue.

``DFlash2BlockConv.prepare`` projects the sublayer input onto ``2 x taps x groups`` coefficients
(``kernel_projection``, ``[2048, 8192]``: side, tap, group) and convolves the input with side 0's
(``dflash2_block_conv``, its own launch between this GEMM and the sublayer's first GEMM); side 1's
go to the sublayer's ``finish``. On the drafter's 9..32-row blocks the GEMM is flashinfer's
cluster split-K kernel at the fixed tactic of ``fixed_low_m_gemm`` (``drafter_fixed_gemms``).

This entry is that GEMM on ``batch_splitk_gemm``'s copy of the same kernel (same tiles, same
``split_k``, so the same bits: each output element is the FP32 sum of the ``split_k`` K-slice
partials in cluster-rank order, rounded once to BF16) over a row-permuted copy of the weight in
which each side-0 group's two taps are adjacent rows (``interleave``), so a CTA tile holds both
coefficients of each of its groups. After the reduction the CTA that owns a side-0 tile's values
stages them (rounded to BF16, as the GEMM output is) in shared memory and convolves the input
with them, statement for statement as ``dflash2_block_conv``: each coefficient ``base + delta``,
tap product, mask product and running sum rounded to BF16, IEEE FP32 in between, the masked tap-1
term multiplied by 0.0, row ``t - 1`` read across a block boundary and zero before row 0.
Side-0 coefficients are not stored; side-1 tiles (the original row order) store theirs as the
GEMM does. So the convolution output and the side-1 coefficients are bitwise the GEMM +
``dflash2_block_conv`` pair's.

Ownership: the reduce-scatter epilogue gives cluster rank ``r`` the values of epilogue thread
group ``r`` (``128 / split_k`` threads, whole warps); a warp's TMEM sub-partition holds a
contiguous quarter of the tile's kernel-M rows (``tmem_warp_shape`` (4, 1)), so rank ``r`` owns
kernel-M rows ``[r M / s, (r + 1) M / s)`` of every public row, an even number of whole groups.

Programmatic dependent launch: as ``batch_splitk_gemm`` (weight before the wait, activation
after it, dependents released once the weight warp has issued its last load). The epilogue
warps issue every load of the convolution (input rows ``t`` and ``t - 1``, both base taps)
after their own ``griddepcontrol.wait`` and before they wait for the mainloop, so the loads
complete under it and only the arithmetic and the stores follow the reduction.
"""

from __future__ import annotations

import functools
from typing import Tuple

import cuda.bindings.driver as _cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
import torch
from cutlass import Int32, Int64, Uint32
from cutlass.cute import experimental as cute_ext
from cutlass.cutlass_dsl import dsl_user_op

from ..allreduce.drafter_hop import (
    _add_rn,
    _asm,
    _i64,
    _keep,
    _load_x4_if,
    _mul_rn,
    _pack_bf16x2,
    _round_bf16,
    _u32,
    _unpack_bf16x2,
)
from . import batch_splitk_gemm as _bsk

#: The drafter's convolution geometry: hidden 8192, two taps, groups of 16 channels.
HIDDEN, TAPS, GROUP_SIZE = 8192, 2, 16
GROUPS = HIDDEN // GROUP_SIZE
#: Rows of the coefficient projection per side (taps x groups).
SIDE_ROWS = TAPS * GROUPS
#: Channels one epilogue item covers (one 16-byte vector).
_VECTOR = 8
_STAGE_ALIGN_BYTES = 128


@dsl_user_op
def _store_x4_if(address, enabled, w0, w1, w2, w3, *, loc=None, ip=None):
    """Plain 16-byte global store when ``enabled``; else nothing."""
    _asm(
        None,
        [_i64(address, loc, ip), _u32(enabled, loc, ip)]
        + [_u32(w, loc, ip) for w in (w0, w1, w2, w3)],
        "{\n.reg .pred p;\nsetp.ne.u32 p, $1, 0;\n@p st.global.v4.b32 [$0], {$2, $3, $4, $5};\n}",
        "l,r,r,r,r,r",
        loc,
        ip,
    )


def interleave(weight: torch.Tensor) -> torch.Tensor:
    """The kernel's weight: side 0's rows reordered ``[tap][group] -> [group][tap]``, side 1 as is.

    ``weight`` is a ``kernel_projection`` weight ``[2 * taps * groups, hidden]``. Returns a new
    contiguous tensor (the module's weight is not modified).
    """
    side0 = weight[:SIDE_ROWS].view(TAPS, GROUPS, -1).transpose(0, 1).reshape(SIDE_ROWS, -1)
    return torch.cat([side0, weight[SIDE_ROWS:]], dim=0).contiguous()


def stage_bytes(tactic: _bsk.BatchSplitKTactic) -> int:
    return tactic.mma_m * tactic.mma_n * 4


def smem_bytes(tactic: _bsk.BatchSplitKTactic, ab_stages: int | None = None) -> int:
    """``batch_splitk_gemm``'s layout plus the FP32 staging tile."""
    cursor = _bsk.smem_bytes(tactic, ab_stages)
    return _bsk._align_up(cursor, _STAGE_ALIGN_BYTES) + stage_bytes(tactic)


def fitted(tactic: _bsk.BatchSplitKTactic) -> _bsk.BatchSplitKTactic:
    """``tactic`` with as many of its stages as fit beside the staging tile (pipeline depth only:
    the stages do not change any output bit)."""
    stages = tactic.ab_stages
    while stages > 1 and smem_bytes(tactic, stages) > _bsk.SMEM_CAPACITY_BYTES:
        stages -= 1
    return _bsk.BatchSplitKTactic(tactic.mma_m, tactic.mma_n, tactic.split_k, stages)


def validate(tactic: _bsk.BatchSplitKTactic, m: int, n: int, k: int) -> None:
    """``batch_splitk_gemm.validate`` plus the epilogue's own bounds.

    Raises:
        ValueError: naming the first violated bound.
    """
    _bsk.validate(tactic, m, n, k)
    if n != 2 * SIDE_ROWS or k != HIDDEN:
        raise ValueError(f"the convolution epilogue takes a [{2 * SIDE_ROWS}, {HIDDEN}] weight")
    rows_owned = tactic.mma_m // tactic.split_k
    if rows_owned % 2 or (rows_owned // 2 * 2 * tactic.mma_n) % (128 // tactic.split_k):
        raise ValueError(f"tactic {tactic}: a rank's tile share is not whole groups per thread")
    if SIDE_ROWS % tactic.mma_m:
        raise ValueError(f"mma_m={tactic.mma_m} does not tile one side's {SIDE_ROWS} rows")
    if smem_bytes(tactic) > _bsk.SMEM_CAPACITY_BYTES:
        raise ValueError(f"tactic {tactic} with its staging tile exceeds shared memory")


class ConvCoefGemmKernel(_bsk.BatchSplitKGemmKernel):
    """``batch_splitk_gemm``'s kernel with the side-0 convolution in its epilogue."""

    def __init__(self, *, tactic: _bsk.BatchSplitKTactic, use_pdl: bool) -> None:
        super().__init__(tactic=tactic, use_pdl=use_pdl)
        self.rows_owned = tactic.mma_m // tactic.split_k
        # Items (8 channels of one group of one public row) per owner thread.
        self.items = self.rows_owned * tactic.mma_n // self.rs_group

    @cute.experimental.jit
    def __call__(
        self,
        a: cute.Tensor,
        b: cute.Tensor,
        c: cute.Tensor,
        x: Int64,
        base: Int64,
        out: Int64,
        rows: Int32,
        block: Int32,
        stream: _cuda.CUstream,
    ):
        self.kernel(a, b, c, x, base, out, rows, block).launch(
            grid=(
                cute.ceil_div(c.layout.shape[0], self.cta_m),
                cute.ceil_div(c.layout.shape[1], self.cta_n) * self.split_k,
                c.layout.shape[2],
            ),
            block=(self.threads_per_cta, 1, 1),
            cluster=self.cluster_shape,
            smem=cute.Int64(_bsk.SMEM_CAPACITY_BYTES),
            stream=stream,
            use_pdl=self.use_pdl,
        )

    @cute.experimental.kernel
    def kernel(
        self,
        mA: cute.Tensor,
        mB: cute.Tensor,
        mC: cute.Tensor,
        x: Int64,
        base: Int64,
        out: Int64,
        rows: Int32,
        block: Int32,
    ):
        # batch_splitk_gemm.BatchSplitKGemmKernel.kernel, the epilogue call aside.
        stages = self.num_ab_stage
        ab_dtype = mA.element_type
        tiled_mma = sm100_utils.make_trivial_tiled_mma(
            ab_dtype,
            ab_dtype,
            utils.LayoutEnum.from_tensor(mA).mma_major_mode(),
            utils.LayoutEnum.from_tensor(mB).mma_major_mode(),
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
            alignment=_bsk._AB_BUFFER_ALIGN_BYTES,
        )
        sB = cute_ext.allocate(
            ab_dtype,
            cute.AddressSpace.smem,
            sm100_utils.make_smem_layout_b(tiled_mma, mnk_tiler, ab_dtype, stages),
            alignment=_bsk._AB_BUFFER_ALIGN_BYTES,
        )
        acc_layout = cute_ext.make_tmem_layout_acc(tiled_mma, self.mma_tiler_mn, acc_stage=1)
        c_tiler_mn = (self.cta_m, self.cta_n)

        bar_full = cute_ext.allocate(
            cutlass.Int64,
            cute.AddressSpace.smem,
            cute.make_layout(stages),
            alignment=_bsk._MBARRIER_BYTES,
        ).iterator
        bar_empty = cute_ext.allocate(
            cutlass.Int64,
            cute.AddressSpace.smem,
            cute.make_layout(stages),
            alignment=_bsk._MBARRIER_BYTES,
        ).iterator
        bar_mma_epilog = cute_ext.allocate(
            cutlass.Int64,
            cute.AddressSpace.smem,
            cute.make_layout(1),
            alignment=_bsk._MBARRIER_BYTES,
        ).iterator
        bar_tmem_alloc = cute_ext.allocate(
            cutlass.Int64,
            cute.AddressSpace.smem,
            cute.make_layout(1),
            alignment=_bsk._MBARRIER_BYTES,
        ).iterator
        bar_spare = cute_ext.allocate(
            cutlass.Int64,
            cute.AddressSpace.smem,
            cute.make_layout(1),
            alignment=_bsk._MBARRIER_BYTES,
        ).iterator
        tmem_base_ptr = cute_ext.allocate(
            cutlass.Int32,
            cute.AddressSpace.smem,
            cute.make_layout(1),
            alignment=_bsk._TMEM_POINTER_BYTES,
        ).iterator
        if cutlass.const_expr(self.split_k > 1):
            mailbox = cute_ext.allocate(
                cutlass.Float32,
                cute.AddressSpace.smem,
                cute.make_layout(self.rs_mailbox_elements),
                alignment=_bsk._MAILBOX_ALIGN_BYTES,
            )
            bar_reduce = cute_ext.allocate(
                cutlass.Int64,
                cute.AddressSpace.smem,
                cute.make_layout(1),
                alignment=_bsk._MBARRIER_BYTES,
            ).iterator
        else:
            mailbox = sA
            bar_reduce = bar_spare
        # The owner's BF16-rounded coefficients of the tile, as FP32, kernel-M fastest.
        stage = cute_ext.allocate(
            cutlass.Float32,
            cute.AddressSpace.smem,
            cute.make_layout((self.cta_m, self.cta_n), stride=(1, self.cta_m)),
            alignment=_STAGE_ALIGN_BYTES,
        )

        if warp_idx == 0:
            with cute.arch.elect_one():
                for i in range(stages):
                    cute.arch.mbarrier_init(bar_full + i, 2)
                    cute.arch.mbarrier_init(bar_empty + i, 1)
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
            self.epilog_conv_warp(
                bar_mma_epilog,
                bar_tmem_alloc,
                tmem_base_ptr,
                acc_layout,
                cute.local_tile(mC, c_tiler_mn, (bidx, n_idx, l_idx)),
                cute.arch.thread_idx()[0] - 128,
                mC.element_type,
                utils.LayoutEnum.from_tensor(mC),
                mailbox,
                bar_reduce,
                split_rank,
                stage,
                bidx,
                n_idx,
                x,
                base,
                out,
                rows,
                block,
            )

    @cute.experimental.jit
    def epilog_conv_warp(
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
        x: Int64,
        base: Int64,
        out: Int64,
        rows: Int32,
        block: Int32,
    ):
        # batch_splitk_gemm.BatchSplitKGemmKernel.epilog_warp up to the owner's rounding.
        cute.arch.mbarrier_arrive(bar_tmem_alloc)
        cute.arch.mbarrier_wait(bar_tmem_alloc, 0)
        _bsk._tcgen05_fence(before=False)

        acc_view = cute.make_tensor(
            cute.arch.retrieve_tmem_ptr(self.acc_dtype, 16, tmem_base_ptr), acc_layout
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

        # The convolution's operands, loaded while the mainloop runs: this thread's items of the
        # tile's side-0 groups when its group owns them (predicated off, no access, otherwise).
        side0 = bidx * self.cta_m < SIDE_ROWS
        dest = Int32(0)
        lt = epi_tid
        if cutlass.const_expr(self.split_k > 1):
            dest = epi_tid // self.rs_group
            lt = epi_tid % self.rs_group
        conv_on = Uint32(side0) & Uint32(dest == split_rank)
        if cutlass.const_expr(self.use_pdl):
            # The input rows are the predecessor's output.
            cute.arch.griddepcontrol_wait()
        items = self._prefetch(conv_on, split_rank, lt, bidx, n_idx, x, base, rows)

        cute.arch.mbarrier_wait(bar_mma_epilog, 0)
        _bsk._tcgen05_fence(before=False)
        cute_ext.partition_and_copy(thr_t2r, acc_view, rAcc)
        cute.arch.fence_view_async_tmem_load()
        _bsk._tcgen05_fence(before=True)
        cute.arch.mbarrier_arrive(bar_tmem_alloc)

        if cutlass.const_expr(self.split_k > 1):
            vpt = cutlass.const_expr(cute.size(rmem_layout))
            group_values = cutlass.const_expr(self.rs_group * vpt)
            if dest != split_rank:
                j = split_rank - cutlass.select_(split_rank > dest, Int32(1), Int32(0))
                for value_idx in cutlass.range_constexpr(0, vpt, 4):
                    slot = (value_idx // 4) * (self.rs_group * 4) + lt * 4
                    _bsk._store_shared_remote_v4(
                        rAcc[value_idx],
                        rAcc[value_idx + 1],
                        rAcc[value_idx + 2],
                        rAcc[value_idx + 3],
                        mailbox.iterator + j * group_values + slot,
                        bar_reduce,
                        dest,
                    )
            else:
                if lt == 0:
                    cute.arch.mbarrier_arrive_and_expect_tx(bar_reduce, self.rs_expected_bytes)
                _bsk._mbarrier_wait_acquire_cluster(bar_reduce, Int32(0))
                for value_idx in cutlass.range_constexpr(vpt):
                    slot = (value_idx // 4) * (self.rs_group * 4) + lt * 4 + value_idx % 4
                    own = rAcc[value_idx]
                    acc = own
                    for r in cutlass.range_constexpr(self.split_k):
                        jr = cutlass.min(
                            Int32(r) - cutlass.select_(Int32(r) > split_rank, Int32(1), Int32(0)),
                            Int32(self.split_k - 2),
                        )
                        term = cutlass.select_(
                            Int32(r) == split_rank, own, mailbox[jr * group_values + slot]
                        )
                        if cutlass.const_expr(r == 0):
                            acc = term
                        else:
                            acc = acc + term
                    rAcc[value_idx] = acc
                rD.store(rAcc.load().to(c_dtype))
                if side0:
                    # The coefficients as the GEMM output holds them (BF16), widened back.
                    rAcc.store(rD.load().to(self.acc_dtype))
                    cute_ext.partition_and_copy(thr_t2r, rAcc, stage)
                else:
                    cute_ext.partition_and_copy(thr_t2r, rD, gD_epi[None, None, 0, 0])
        else:
            rD.store(rAcc.load().to(c_dtype))
            if side0:
                rAcc.store(rD.load().to(self.acc_dtype))
                cute_ext.partition_and_copy(thr_t2r, rAcc, stage)
            else:
                cute_ext.partition_and_copy(thr_t2r, rD, gD_epi[None, None, 0, 0])

        if side0:
            if dest == split_rank:
                # The owner group's staged rows are complete for every one of its threads.
                cute.arch.barrier(barrier_id=1, number_of_threads=self.rs_group)
                self._convolve(stage, items, out, block)

    def _prefetch(
        self,
        conv_on: Uint32,
        split_rank: cutlass.Int32,
        lt: cutlass.Int32,
        bidx: cutlass.Int32,
        n_idx: cutlass.Int32,
        x: Int64,
        base: Int64,
        rows: Int32,
    ):
        """This thread's items (8 channels of one owned group of one public row): coordinates and
        the input rows ``t``, ``t - 1`` and both base taps, all loads issued before any use."""
        groups_owned = self.rows_owned // 2
        m_first = split_rank * self.rows_owned
        items = []
        for i in range(self.items):
            item = lt + i * self.rs_group
            half = item % 2
            gl = (item // 2) % groups_owned
            n = item // (2 * groups_owned)
            t = n_idx * self.cta_n + n
            m = m_first + 2 * gl
            group = (bidx * self.cta_m + m) // 2
            column_bytes = Int64(group * GROUP_SIZE + half * _VECTOR) * 2
            row_bytes = Int64(t) * (HIDDEN * 2)
            on = conv_on & Uint32(t < rows)
            cur = _load_x4_if(x + row_bytes + column_bytes, on)
            prev = _load_x4_if(x + row_bytes - HIDDEN * 2 + column_bytes, on & Uint32(t >= 1))
            b0 = _load_x4_if(base + column_bytes, on)
            b1 = _load_x4_if(base + Int64(HIDDEN * 2) + column_bytes, on)
            items.append((t, m, n, row_bytes + column_bytes, Uint32(t < rows), cur, prev, b0, b1))
        return items

    def _convolve(self, stage: cute.Tensor, items, out: Int64, block: Int32):
        """``dflash2_block_conv``'s statements on the prefetched items, coefficients from the stage
        (rows past the call's are computed and not stored: no branch)."""
        for t, m, n, offset, valid, cur, prev, b0, b1 in items:
            d0 = stage[m, n]
            d1 = stage[m + 1, n]
            keep = _keep(t % block)
            words = []
            for k in range(_VECTOR // 2):
                pair = []
                for e in range(2):
                    coefficient = _round_bf16(_add_rn(_unpack_bf16x2(b0[k])[e], d0))
                    acc = _round_bf16(_mul_rn(coefficient, _unpack_bf16x2(cur[k])[e]))
                    coefficient = _round_bf16(_add_rn(_unpack_bf16x2(b1[k])[e], d1))
                    term = _round_bf16(_mul_rn(coefficient, _unpack_bf16x2(prev[k])[e]))
                    term = _round_bf16(_mul_rn(term, keep))
                    pair.append(_round_bf16(_add_rn(acc, term)))
                words.append(_pack_bf16x2(pair[0], pair[1]))
            _store_x4_if(out + offset, valid, words[0], words[1], words[2], words[3])


@cute.experimental.jit
def _bmm_conv(
    gemm_op: cutlass.Constexpr,
    a: cute.Tensor,
    b: cute.Tensor,
    c: cute.Tensor,
    x: Int64,
    base: Int64,
    out: Int64,
    rows: Int32,
    block: Int32,
    stream: _cuda.CUstream,
):
    gemm_op(
        cute.make_tensor(a.iterator, cute.select(a.layout, mode=[1, 2, 0])),
        cute.make_tensor(b.iterator, cute.select(b.layout, mode=[2, 1, 0])),
        cute.make_tensor(c.iterator, cute.select(c.layout, mode=[1, 2, 0])),
        x,
        base,
        out,
        rows,
        block,
        stream,
    )


@functools.lru_cache(maxsize=None)
def _compiled(
    tactic: _bsk.BatchSplitKTactic, use_pdl: bool, leading: Tuple[int, int, int], device: int
):
    m, n, k = 64, 8, _bsk.CTA_K
    dtype = torch.bfloat16
    with torch.cuda.device(device):
        reprs = tuple(
            _bsk._from_dlpack_dynamic(_bsk._make_layout_tensor(shape, dtype, d), d)
            for shape, d in (
                ((1, n, k), leading[0]),
                ((1, k, m), leading[1]),
                ((1, n, m), leading[2]),
            )
        )
        return cute_ext.compile(
            _bmm_conv,
            ConvCoefGemmKernel(tactic=tactic, use_pdl=use_pdl),
            *reprs,
            Int64(0),
            Int64(0),
            Int64(0),
            Int32(8),
            Int32(8),
            _cuda.CUstream(torch.cuda.current_stream().cuda_stream),
        )


def run(
    x: torch.Tensor,
    weight: torch.Tensor,
    base0: torch.Tensor,
    block_size: int,
    tactic: _bsk.BatchSplitKTactic,
    pdl: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """``prepare``'s two outputs from ``x [M, 8192]`` and an ``interleave``d weight ``[2048, 8192]``.

    Returns ``(conv, coefficients)``: the input-side convolution ``[M, 8192]`` (bitwise
    ``dflash2_block_conv(x, coef[:, 0], base0, block_size, 16)`` of the GEMM's coefficients) and
    the ``[M, 2048]`` coefficient buffer whose side-1 half holds the GEMM's side-1 coefficients in
    the original order (side 0's half is not written). ``base0`` is the input-side base kernel
    ``[2, 8192]``, contiguous. ``tactic`` is taken with its stages cut to fit (``fitted``).

    Raises:
        ValueError: for operands or a tactic the kernel does not take.
    """
    tactic = fitted(tactic)
    if not (x.dtype == weight.dtype == base0.dtype == torch.bfloat16):
        raise ValueError("the convolution epilogue takes BF16 operands only")
    if x.dim() != 2 or weight.dim() != 2 or tuple(base0.shape) != (TAPS, HIDDEN):
        raise ValueError(
            f"shapes x {tuple(x.shape)}, weight {tuple(weight.shape)}, base {tuple(base0.shape)}"
        )
    m, k = (int(d) for d in x.shape)
    n = int(weight.shape[0])
    if int(weight.shape[1]) != k:
        raise ValueError(f"shapes x {tuple(x.shape)}, weight {tuple(weight.shape)}")
    if not (x.is_contiguous() and weight.is_contiguous() and base0.is_contiguous()):
        raise ValueError("the convolution epilogue takes contiguous operands")
    if any(t.data_ptr() % 32 for t in (x, weight)) or base0.data_ptr() % 16:
        raise ValueError("the convolution epilogue takes 32-byte aligned operands (base 16)")
    if not (x.is_cuda and weight.device == x.device == base0.device):
        raise ValueError("the convolution epilogue takes CUDA operands on one device")
    validate(tactic, m, n, k)
    coefficients = torch.empty((m, n), dtype=x.dtype, device=x.device)
    conv = torch.empty((m, k), dtype=x.dtype, device=x.device)
    with torch.cuda.device(x.device):
        tensors, leading = _bsk._swapped(x, weight, coefficients)
        compiled = _compiled(tactic, bool(pdl), leading, x.get_device())
        compiled(
            *tensors,
            Int64(x.data_ptr()),
            Int64(base0.data_ptr()),
            Int64(conv.data_ptr()),
            Int32(m),
            Int32(block_size),
            _cuda.CUstream(torch.cuda.current_stream(x.device).cuda_stream),
        )
    return conv, coefficients
