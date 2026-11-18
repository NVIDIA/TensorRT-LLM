# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Rubin split-K projection with the weights staged in tensor memory (B <= 32).

``SplitKGemm`` with 128-row M tiles: each CTA of a ``split_k`` cluster owns a K slice of
one M tile. Before the PDL wait the MMA warp copies the K tiles of A (the weights) from
the TMA ring to tensor memory as they land (``tcgen05.cp``), which frees their ring
stages for the next tiles, so the whole slice can be on chip when x lands; their MMAs
then read A from tensor memory (M128 N8 K16: 11.3 cycles vs 43 from shared memory,
``notes/kdahead3.md``). B (x) is loaded for every K tile after the PDL wait, each tile
on its own barrier. N tiles of 8, 16 or 32 token columns (the accumulator's
tensor-memory columns, before the staged tiles).

The FP32 partials are reduce-scattered over the cluster in one DSMEM hop: owner rank o
(``o < owners``) receives rows [o R, (o + 1) R) (R = 128 / owners) of every rank's
partial (16-byte ``st.async``), sums them in rank order, ((p0 + p1) + p2) + ..., as
``SplitKGemm``'s owner does, rounds and stores them. The output is bit for bit
``SplitKGemm``'s with the (128, mma_n, split_k, stages) tactic (the same MMAs in the
same order; A from tensor memory equals A from shared memory)."""

import cuda.bindings.driver as _cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass import BFloat16, Float32, Int32, Uint8
from cutlass.cute import experimental as cute_ext
from cutlass.cute.arch import store_async_dsmem
from cutlass.cute.nvgpu import tcgen05

from ..attn_res.kernel import BESIDE_SMEM
from ..common.primitives import at, load_words, named_barrier, store_words
from ..kda.head import _copy_to_tmem, _mma_smem, _mma_tmem, store_bf16_canonical
from .splitk import (
    ATOM_K,
    SplitKGemm,
    SplitKTactic,
    _acquire_tensor_map,
    _fence_tcgen05,
    _smem,
    _tma_load_evict_first,
    _wait_cluster,
)
from .splitk_tmem64 import _mma_smem64, _mma_tmem64

__all__ = ["SplitKTmemGemm"]


class SplitKTmemGemm(SplitKGemm):
    """``tactic`` (128, mma_n 8 / 16 / 32, split_k, ring stages); ``k``: the reduction
    length (static: the staged loops are unrolled); ``tmem_tiles``: K tiles of 64
    staged in tensor memory (the rest stream through the ring after them);
    ``early_loads``: A tiles requested before the PDL wait (None: as the ring allows);
    ``owners``: ranks that reduce and store a 128 / owners row share."""

    def __init__(
        self, *, tactic: SplitKTactic, k: int, tmem_tiles: int, early_loads=None,
        owners: int = 0, canonical: bool = False,
    ):  # fmt: skip
        super().__init__(tactic=tactic)
        assert tactic.mma_m == 128 and tactic.mma_n in (8, 16, 32), "M128 N8-32 tiles"
        self.n = n = tactic.mma_n
        # Above N8 the MMAs take the N field of the instruction descriptor and B's
        # SW128 descriptor with its 1024-byte stride between 8-row groups.
        self.idesc = 0x08000490 | (n >> 3) << 17
        self.cta_k = 64
        self.k_tiles = k // self.cta_k // self.split_k
        assert self.k_tiles * self.cta_k * self.split_k == k
        assert 0 < tmem_tiles <= self.k_tiles and n + 32 * tmem_tiles <= 512
        assert self.k_tiles >= self.num_ab_stage
        self.tmem_tiles, self.early_loads = tmem_tiles, early_loads
        # ``canonical``: BF16 NaN results are stored as 0x7FFF, so no element is
        # 0xFFFF (MlaQuery's x sentinel, ``mla.query.X_SENT``); other values unchanged.
        self.canonical = canonical
        self.owners = owners or min(self.split_k, 8)
        assert self.owners <= self.split_k and 128 % (4 * self.owners) == 0
        self.share = 128 // self.owners  # rows each owner reduces
        # Shared memory actually used (A ring, resident B, barriers, the partial, the
        # owner mailbox, alignment), so a CTA fits beside an AttnRes CTA (the in-
        # projections after it, at its carve-out 100); N16 / N32 tactics take the ring
        # stages that keep it within 218 KB, beside three AttnRes CTAs (at 16 / 32
        # rows the AttnRes's 7 CTAs per row stack up to three per SM).
        stages = self.num_ab_stage
        used = stages * 128 * self.cta_k * 2 + self.k_tiles * n * self.cta_k * 2
        used += (2 * stages + self.k_tiles + 3) * 8 + 4 + 128 * n * 4
        used += self.split_k * n * self.share * 4
        self.smem_bytes = (used + 2560 + 1023) // 1024 * 1024
        assert self.smem_bytes <= BESIDE_SMEM

    @cute.experimental.jit
    def __call__(
        self, a: cute.Tensor, b: cute.Tensor, c: cute.Tensor, bias: cute.Tensor,
        a_map: cutlass.Int64, stream: _cuda.CUstream,
    ):  # fmt: skip
        self.kernel(a, b, c, a_map).launch(
            grid=(
                cute.ceil_div(c.layout.shape[0], self.cta_m),
                cute.ceil_div(c.layout.shape[1], self.cta_n) * self.split_k,
                c.layout.shape[2],
            ),
            block=(self.threads_per_cta, 1, 1),
            cluster=self.cluster_shape,
            smem=cute.Int64(self.smem_bytes),
            stream=stream,
            use_pdl=True,
        )

    @cute.experimental.kernel
    def kernel(
        self, mA: cute.Tensor, mB: cute.Tensor, mC: cute.Tensor, a_map: cutlass.Int64
    ):
        """A (M, K, L) and B (N, K, L) K-major, C (M, N, L) M-major."""
        stages, k_tiles = self.num_ab_stage, self.k_tiles
        ab_dtype = mA.element_type
        majors = (utils.LayoutEnum.from_tensor(t).mma_major_mode() for t in (mA, mB))
        tiled_mma = sm100_utils.make_trivial_tiled_mma(
            ab_dtype, ab_dtype, *majors, self.acc_dtype, self.cta_group,
            self.mma_tiler_mn,
        )  # fmt: skip
        mnk_tiler = (self.mma_tiler_mn[0], self.mma_tiler_mn[1], self.cta_k)
        block_idx = cute.arch.block_idx()
        bidx = block_idx[0]
        split_rank = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        n_idx = block_idx[1] // self.split_k
        l_idx = block_idx[2]
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())

        a_shape = (tiled_mma, mnk_tiler, ab_dtype, stages)
        b_shape = (tiled_mma, mnk_tiler, ab_dtype, k_tiles)
        sA = _smem(ab_dtype, None, 1024, sm100_utils.make_smem_layout_a(*a_shape))
        sB = _smem(ab_dtype, None, 1024, sm100_utils.make_smem_layout_b(*b_shape))
        acc_layout = cute_ext.make_tmem_layout_acc(
            tiled_mma, self.mma_tiler_mn, acc_stage=1
        )
        bar_full = _smem(cutlass.Int64, stages, 8).iterator
        bar_empty = _smem(cutlass.Int64, stages, 8).iterator
        b_full = _smem(cutlass.Int64, k_tiles, 8).iterator
        bar_mma_epilog = _smem(cutlass.Int64, 1, 8).iterator
        bar_tmem_alloc = _smem(cutlass.Int64, 1, 8).iterator
        bar_reduce = _smem(cutlass.Int64, 1, 8).iterator
        tmem_base_ptr = _smem(cutlass.Int32, 1, 4).iterator
        # This CTA's FP32 partial [128 rows, N] (row fastest); an owner's mailbox
        # [split rank][N][its row share].
        tile_layout = cute.make_layout((self.cta_m, self.cta_n), stride=(1, self.cta_m))
        partial = _smem(Float32, None, 128, tile_layout)
        mailbox = _smem(Float32, self.split_k * self.cta_n * self.share, 128)

        if warp_idx == 0:
            with cute.arch.elect_one():
                for i in range(stages):
                    cute.arch.mbarrier_init(bar_full + i, 1)
                    cute.arch.mbarrier_init(bar_empty + i, 1)
                for i in range(k_tiles):
                    cute.arch.mbarrier_init(b_full + i, 1)
                cute.arch.mbarrier_init(bar_mma_epilog, 1)
                cute.arch.mbarrier_init(bar_tmem_alloc, 160)
                cute.arch.mbarrier_init(bar_reduce, 1)
        cute.arch.mbarrier_init_fence()
        cute.arch.cluster_arrive_relaxed()  # every barrier before any DSMEM traffic
        k_tile_start = split_rank * k_tiles
        cute.arch.cluster_wait()

        # Warp 3 is idle; warps 4-7 run the epilogue.
        if warp_idx == 0:
            self.weight_warp(bar_full, bar_empty, sA, a_map, bidx * self.cta_m,
                             k_tile_start, k_tiles)  # fmt: skip
        elif warp_idx == 1:
            b_tile = cute.local_tile(mB, (self.cta_n, self.cta_k), (n_idx, None, l_idx))
            b_map = cute_ext.get_cta_v_map_ab(mB, mnk_tiler, tiled_mma, "B")
            self.resident_b(b_full, b_tile, sB, b_map, k_tile_start)
        elif warp_idx == 2:
            self.staged_mma_warp(
                bar_full, bar_empty, b_full, bar_mma_epilog, bar_tmem_alloc, sA, sB,
                tmem_base_ptr,
            )  # fmt: skip
        elif warp_idx >= 4:
            self.reduce_scatter(
                cute.arch.thread_idx()[0] - 128, bar_mma_epilog, bar_tmem_alloc,
                tmem_base_ptr, acc_layout, partial, mailbox, bar_reduce, split_rank,
                mC, bidx * self.cta_m, n_idx * self.cta_n, l_idx,
            )  # fmt: skip

    @cute.experimental.jit
    def weight_warp(
        self, bar_full, bar_empty, s_tile: cute.Tensor, a_map: cutlass.Int64,
        row0: cutlass.Int32, k_tile_start: cutlass.Int32, k_tile_count: cutlass.Int32,
    ):  # fmt: skip
        """``SplitKGemm.weight_warp`` (one evict_first box per K tile, the dependents'
        trigger once every tile is requested); with ``early_loads``, tiles from that
        index on are requested only after the PDL wait."""
        stages = self.num_ab_stage
        stage_bytes = cute.size_in_bytes(
            s_tile.element_type, cute.slice_(s_tile.layout, (None, None, None, 0))
        )
        early = self.k_tiles if self.early_loads is None else self.early_loads
        _acquire_tensor_map(a_map)
        empty_phase = cutlass.Int32(1)
        for k_tile in cutlass.range(k_tile_count, unroll=1):
            stage = k_tile % stages
            if cutlass.const_expr(early < self.k_tiles):  # noqa: SIM102 (static)
                if k_tile == early:
                    cute.arch.griddepcontrol_wait()
            cute.arch.mbarrier_wait(bar_empty + stage, empty_phase)
            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive_and_expect_tx(bar_full + stage, stage_bytes)
            with cute.arch.elect_one():
                k0 = (k_tile_start + k_tile) * self.cta_k
                tile = s_tile[None, None, None, stage].iterator
                for atom in cutlass.range_constexpr(self.cta_k // ATOM_K):
                    box, mbar = tile + atom * self.cta_m * ATOM_K, bar_full + stage
                    k = k0 + atom * ATOM_K
                    _tma_load_evict_first(box, mbar, a_map, k, row0)
            if stage == stages - 1:
                empty_phase = empty_phase ^ 1
        cute.arch.griddepcontrol_launch_dependents()
        self.drain(bar_empty, empty_phase, k_tile_count)

    @cute.experimental.jit
    def resident_b(
        self, b_full, g_tile: cute.Tensor, s_tile: cute.Tensor, cta_v_map: cute.Layout,
        k_tile_start: cutlass.Int32,
    ):  # fmt: skip
        """Every K tile of B (the activations) after the PDL wait, each on its own
        barrier."""
        tile_bytes = cute.size_in_bytes(
            s_tile.element_type, cute.slice_(s_tile.layout, (None, None, None, 0))
        )
        cute.arch.griddepcontrol_wait()
        for k_tile in cutlass.range(self.k_tiles, unroll=1):
            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive_and_expect_tx(b_full + k_tile, tile_bytes)
            cute_ext.tma_load(
                g_tile[None, None, k_tile_start + k_tile],
                s_tile[None, None, None, k_tile],
                (b_full + k_tile).value,
                cta_v_map=cta_v_map,
                tma_operation_type=cute_ext.OperationTypeEnum.SM90_TMA_LOAD,
                update_expect_tx=False,
            )

    @cute.experimental.jit
    def staged_mma_warp(
        self, bar_full, bar_empty, b_full, bar_mma_epilog, bar_tmem_alloc,
        sA: cute.Tensor, sB: cute.Tensor, tmem_base_ptr,
    ):  # fmt: skip
        """K tiles 0 .. tmem_tiles - 1 to tensor memory as they land (their ring
        stages refill), then every tile's MMAs in K order: A from tensor memory for
        those, from the ring for the rest (``KdaHeadWide.staged_mma_warp``). The
        accumulator is the first N columns."""
        stages, staged, k_blocks = self.num_ab_stage, self.tmem_tiles, self.cta_k // 16
        a_layout = cute.slice_(sA.layout, (None, None, None, 0))
        b_layout = cute.slice_(sB.layout, (None, None, None, 0))
        a_step = cute.size_in_bytes(sA.element_type, a_layout) // 16
        b_step = cute.size_in_bytes(sB.element_type, b_layout) // 16
        columns = self.cta_k // 2  # tensor-memory columns of one A tile
        cute.arch.alloc_tmem(512, tmem_base_ptr, is_two_cta=False)
        _fence_tcgen05(before=True)
        cute.arch.mbarrier_arrive(bar_tmem_alloc)
        cute.arch.relinquish_tmem_alloc_permit(is_two_cta=False)
        _fence_tcgen05()
        base = Int32(cute.make_tensor(tmem_base_ptr, cute.make_layout(1))[0])
        a_desc = ((sA.iterator.toint() >> 4) & 0x3FFF) | 0x10000
        b_desc = ((sB.iterator.toint() >> 4) & 0x3FFF) | 0x10000

        for k in cutlass.range_constexpr(staged):
            stage, phase = k % stages, (k // stages) % 2
            cute.arch.mbarrier_wait(bar_full + stage, phase)
            _fence_tcgen05()
            with cute.arch.elect_one():
                for kb in cutlass.range_constexpr(k_blocks):
                    source = a_desc + stage * a_step + 2 * kb
                    _copy_to_tmem(base + self.n + columns * k + 8 * kb, source)
                tcgen05.commit(bar_empty + stage, None, self.cta_group)
        for k in cutlass.range_constexpr(self.k_tiles):
            cute.arch.mbarrier_wait(b_full + k, 0)
            if cutlass.const_expr(k < staged):  # A from tensor memory
                _fence_tcgen05()
                with cute.arch.elect_one():
                    for kb in cutlass.range_constexpr(k_blocks):
                        b = b_desc + k * b_step + 2 * kb
                        accumulate = Int32(0 if k == 0 and kb == 0 else 1)
                        a = base + self.n + columns * k + 8 * kb
                        if cutlass.const_expr(self.n == 8):
                            _mma_tmem(base, a, b, accumulate)
                        else:
                            _mma_tmem64(base, a, b, accumulate, idesc=self.idesc)
            else:
                self.ring_mma(bar_full, bar_empty, a_desc, b_desc, base, k)
        with cute.arch.elect_one():
            tcgen05.commit(bar_mma_epilog, None, self.cta_group)
        cute.arch.mbarrier_wait(bar_tmem_alloc, 0)  # the release lands in phase 1
        cute.arch.mbarrier_arrive(bar_tmem_alloc)
        cute.arch.mbarrier_wait(bar_tmem_alloc, 1)
        _fence_tcgen05()
        tmem_ptr = cute.arch.retrieve_tmem_ptr(self.acc_dtype, 16, tmem_base_ptr)
        cute.arch.dealloc_tmem(tmem_ptr, 512, is_two_cta=False)

    @cute.jit
    def ring_mma(self, bar_full, bar_empty, a_desc, b_desc, base, k: cutlass.Constexpr):
        """K tile ``k``'s MMAs with A from its ring stage, which is then released."""
        stages, k_blocks = self.num_ab_stage, self.cta_k // 16
        stage, phase = k % stages, (k // stages) % 2
        a_step, b_step = 128 * self.cta_k * 2 // 16, self.n * self.cta_k * 2 // 16
        cute.arch.mbarrier_wait(bar_full + stage, phase)
        _fence_tcgen05()
        with cute.arch.elect_one():
            for kb in cutlass.range_constexpr(k_blocks):
                b = b_desc + k * b_step + 2 * kb
                accumulate = Int32(0 if k == 0 and kb == 0 else 1)
                a = a_desc + stage * a_step + 2 * kb
                if cutlass.const_expr(self.n == 8):
                    _mma_smem(base, a, b, accumulate)
                else:
                    _mma_smem64(base, a, b, accumulate, idesc=self.idesc)
            tcgen05.commit(bar_empty + stage, None, self.cta_group)

    @cute.experimental.jit
    def reduce_scatter(
        self, epi_tid: cutlass.Int32, bar_mma_epilog, bar_tmem_alloc, tmem_base_ptr,
        acc_layout: cutlass.Constexpr, partial: cute.Tensor, mailbox: cute.Tensor,
        bar_reduce, split_rank: cutlass.Int32, mC: cute.Tensor, row0: cutlass.Int32,
        col0: cutlass.Int32, l_idx: cutlass.Int32,
    ):  # fmt: skip
        """The accumulator -> ``partial``; every owner's row share of it to the
        owner's mailbox slot ``split_rank`` (16-byte chunks: 4 rows of one column);
        owners sum the slots in rank order, round and store their rows."""
        cute.arch.mbarrier_arrive(bar_tmem_alloc)
        cute.arch.mbarrier_wait(bar_tmem_alloc, 0)
        _fence_tcgen05()
        acc_view = cute.make_tensor(
            cute.arch.retrieve_tmem_ptr(self.acc_dtype, 16, tmem_base_ptr), acc_layout
        )[((None, None), 0, 0, 0)]
        epi_tile = (self.cta_m, self.cta_n)
        d_layout = utils.LayoutEnum.from_tensor(partial)
        load_op = sm100_utils.get_tmem_load_op(
            (self.cta_m, self.cta_n, self.cta_k), d_layout, BFloat16, self.acc_dtype,
            epi_tile, False,
        )  # fmt: skip
        tiled_copy_t2r = cute.nvgpu.tcgen05.make_tmem_copy(load_op, acc_view)
        part_epi = cute.flat_divide(partial, epi_tile)
        rmem_layout = cute_ext.make_t2r_rmem_layout(tiled_copy_t2r, part_epi, epi_tid)
        rAcc = cute_ext.allocate(
            self.acc_dtype, cute.AddressSpace.rmem, rmem_layout, alignment=32
        )
        thr_t2r = tiled_copy_t2r.get_slice(epi_tid)
        cute.arch.mbarrier_wait(bar_mma_epilog, 0)
        _fence_tcgen05()
        cute_ext.partition_and_copy(thr_t2r, acc_view, rAcc)
        cute.arch.fence_view_async_tmem_load()
        _fence_tcgen05(before=True)
        cute.arch.mbarrier_arrive(bar_tmem_alloc)
        cute_ext.partition_and_copy(thr_t2r, rAcc, part_epi[None, None, 0, 0])
        named_barrier(3, 128)

        share, cols = self.share, self.cta_n
        smem = cute.AddressSpace.smem
        for m in cutlass.range_constexpr(self.cta_m * cols // 4 // 128):
            i = epi_tid + 128 * m
            col, row = i // (self.cta_m // 4), 4 * (i % (self.cta_m // 4))
            owner, r = row // share, row % share
            src = partial.iterator + cute.assume(col * self.cta_m + row, divby=4)
            words = load_words(src, count=4, shared=True)
            offset = (split_rank * cols + col) * share + r
            dst = mailbox.iterator + cute.assume(offset, divby=4)
            if owner == split_rank:
                store_words(dst, words, shared=True)
            else:
                remote = cute.make_ptr(Uint8, dst.llvm_ptr, smem, assumed_align=16)
                store_async_dsmem(remote, words, bar_reduce, owner)

        if split_rank < self.owners:
            if epi_tid == 0:
                expected = (self.split_k - 1) * cols * share * 4
                cute.arch.mbarrier_arrive_and_expect_tx(bar_reduce, expected)
            named_barrier(3, 128)  # this CTA's own slot
            _wait_cluster(bar_reduce, Int32(0))
            rows, columns = mC.layout.shape[0], mC.layout.shape[1]
            for m in cutlass.range_constexpr(share * cols // 128):
                v = epi_tid + 128 * m
                col, r = v // share, v % share
                total = mailbox[col * share + r]
                for s in cutlass.range_constexpr(1, self.split_k):
                    total = total + mailbox[(s * cols + col) * share + r]
                row = row0 + split_rank * share + r
                if (row < rows) & (col0 + col < columns):
                    if cutlass.const_expr(self.canonical):
                        store_bf16_canonical(at(mC, row, col0 + col, l_idx), total)
                    else:
                        mC[row, col0 + col, l_idx] = mC.element_type(total)
