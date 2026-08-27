# Copyright (c) 2025, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
# SM90 (Hopper) forward pass for flash attention, extracted from flash_fwd.py.

from types import SimpleNamespace
from typing import Callable, Optional
from functools import partial

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, const_expr
from cutlass.cute.nvgpu import cpasync, warpgroup
from cutlass.utils import LayoutEnum
import cutlass.utils.hopper_helpers as sm90_utils_basic
from cutlass import pipeline
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait
from cutlass.base_dsl.arch import Arch

from ._compat import copy_utils
from ._compat import layout_utils
from ._compat import sm90_utils

from sol_attn._vendor.flash_attn.cute.cute_dsl_utils import assume_tensor_aligned
from sol_attn._vendor.flash_attn.cute import utils
from sol_attn._vendor.flash_attn.cute.mask import AttentionMask
from sol_attn._vendor.flash_attn.cute.softmax import Softmax, apply_score_mod_inner
from sol_attn._vendor.flash_attn.cute.seqlen_info import SeqlenInfoQK
from sol_attn._vendor.flash_attn.cute.block_info import BlockInfo
from sol_attn._vendor.flash_attn.cute.block_sparsity import BlockSparseTensors
from sol_attn._vendor.flash_attn.cute import pipeline as pipeline_custom
from sol_attn._vendor.flash_attn.cute.pack_gqa import PackGQA, pack_gqa_layout, make_packgqa_tiled_tma_atom
from sol_attn._vendor.flash_attn.cute.named_barrier import NamedBarrierFwd
from ._compat.cute_dsl_utils import ParamsBase
from sol_attn._vendor.flash_attn.cute.tile_scheduler import (
    TileSchedulerArguments,
    SingleTileScheduler,
    SingleTileLPTScheduler,
    SingleTileVarlenScheduler,
)
from sol_attn._vendor.flash_attn.cute.flash_fwd import FlashAttentionForwardBase
from . import atoms as sol_attn_atoms
from . import exact as exact_stream
from sol_attn.common import selector as sol_attn_selector


SOL_ATTN_ROUTE_MASK_BARRIER_ID = 7
SOL_ATTN_ROUTE_SUM_BARRIER_ID = 8


class SolAttnMainloopSm90(FlashAttentionForwardBase):
    def __init__(
        self,
        *args,
        sol_attn_assume_lane_group_route_reduce: bool = False,
        sol_attn_assume_full_k_exact_blocks: bool = False,
        sol_attn_tail_exact_words1: bool = False,
        sol_attn_assume_full_route_groups: bool = False,
        sol_attn_static_num_full_route_groups: int = -1,
        sol_attn_static_tail_valid_count: int = -1,
        sol_attn_tail_physical_tile16: bool = False,
        sol_attn_exact_mask_seqlen_last_only: bool = False,
        sol_attn_tail16_lane_group_route_reduce: bool = False,
        sol_attn_num_splits: int = 1,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.qk_dtype = cutlass.BFloat16
        self.pv_dtype = self.dtype
        self.sol_attn_group_size = 64
        self.sol_attn_group_words = 2
        self.mma_pv_is_rs = True
        self.sol_attn_mma_regs_override = 128
        self.sol_attn_warp_route_mask = True
        self.sol_attn_fast_route_lens = True
        self.sol_attn_early_route_mask_publish = True
        self.sol_attn_lane_group_route_reduce = True
        self.sol_attn_assume_lane_group_route_reduce = sol_attn_assume_lane_group_route_reduce
        self.sol_attn_assume_full_k_exact_blocks = sol_attn_assume_full_k_exact_blocks
        self.sol_attn_route_sum_arrive_overlap = True
        self.sol_attn_route_mask_after_scale = True
        self.sol_attn_assume_full_route_groups = sol_attn_assume_full_route_groups
        self.sol_attn_static_num_full_route_groups = sol_attn_static_num_full_route_groups
        self.sol_attn_static_tail_valid_count = sol_attn_static_tail_valid_count
        self.sol_attn_tail_exact_words1 = (
            sol_attn_tail_exact_words1 and 0 < self.sol_attn_static_tail_valid_count <= 32
        )
        self.sol_attn_tail_route_mask_words1 = False
        self.sol_attn_tail_physical_tile16 = (
            sol_attn_tail_physical_tile16 and 0 < self.sol_attn_static_tail_valid_count <= 16
        )
        self.sol_attn_exact_mask_seqlen_last_only = sol_attn_exact_mask_seqlen_last_only
        self.sol_attn_full_route_mask_seqlen_false = True
        self.sol_attn_tail16_lane_group_route_reduce = (
            sol_attn_tail16_lane_group_route_reduce and self.sol_attn_tail_physical_tile16
        )
        self.sol_attn_full_block_row_sum_prescale = False
        self.sol_attn_neutral_softmax_state = True
        self.sol_attn_assume_nonempty_rows = False
        self.sol_attn_ballot_mask = True
        self.sol_attn_approx_colmask = False
        self.sol_attn_packed_route_reduction = False
        self.sol_attn_num_splits = sol_attn_num_splits
        self.buffer_align_bytes = 1024
        self.use_tma_KV = True
        self.cluster_shape_mn = (1, 1)
        if not (self.arch >= Arch.sm_90 and self.arch <= Arch.sm_90a):
            raise AssertionError("The Hopper backend requires SM90")

    def _get_smem_layout_atom(self):
        sQ_layout_atom = warpgroup.make_smem_layout_atom(
            sm90_utils_basic.get_smem_layout_atom(
                LayoutEnum.ROW_MAJOR, self.qk_dtype, self.tile_hdim
            ),
            self.qk_dtype,
        )
        sK_layout_atom = sQ_layout_atom
        sV_layout_atom = warpgroup.make_smem_layout_atom(
            sm90_utils_basic.get_smem_layout_atom(
                LayoutEnum.ROW_MAJOR, self.pv_dtype, self.tile_hdimv
            ),
            self.pv_dtype,
        )
        sO_layout_atom = sV_layout_atom
        if not self.mma_pv_is_rs:
            sP_layout_atom = warpgroup.make_smem_layout_atom(
                sm90_utils_basic.get_smem_layout_atom(
                    LayoutEnum.ROW_MAJOR, self.pv_dtype, self.tile_n
                ),
                self.pv_dtype,
            )
        else:
            sP_layout_atom = None
        return sQ_layout_atom, sK_layout_atom, sV_layout_atom, sO_layout_atom, sP_layout_atom

    def _get_tiled_mma(self):
        tiled_mma_qk = sm90_utils.make_tiled_mma(
            cutlass.BFloat16,
            "K",
            "K",
            self.tile_n,
            source="SS",
            atom_layout_mnk=(self.tile_m // 64, 1, 1),
            b_dtype=cutlass.BFloat16,
            acc_dtype=Float32,
        )
        tiled_mma_pv = sol_attn_atoms.make_pv_mma(
            tile_m=self.tile_m,
            tile_v=self.tile_hdimv,
        )
        return tiled_mma_qk, tiled_mma_pv

    @cute.jit
    def sol_attn_qk_gemm_zero_init(
        self,
        tiled_mma: cute.TiledMma,
        shape: cute.Shape,
        tCrA: cute.Tensor,
        tCrB: cute.Tensor,
        A_idx: Optional[Int32] = None,
        B_idx: Optional[Int32] = None,
        wg_wait: int = -1,
        swap_AB: bool = False,
    ) -> cute.Tensor:
        """Run BF16 QK WGMMA directly into an FP32 accumulator."""

        return sm90_utils.gemm_zero_init(
            tiled_mma,
            shape,
            tCrA,
            tCrB,
            A_idx,
            B_idx,
            wg_wait,
            swap_AB,
        )

    def _get_shared_storage_cls(self):
        sQ_struct, sK_struct = [
            cute.struct.Align[
                cute.struct.MemRange[self.qk_dtype, cute.cosize(layout)], self.buffer_align_bytes
            ]
            for layout in (self.sQ_layout, self.sK_layout)
        ]
        sV_struct = cute.struct.Align[
            cute.struct.MemRange[self.pv_dtype, cute.cosize(self.sV_layout)],
            self.buffer_align_bytes,
        ]
        cosize_sQV = max(cute.cosize(self.sQ_layout), cute.cosize(self.sV_layout))
        sQV_struct = cute.struct.Align[cute.struct.MemRange[self.pv_dtype, cosize_sQV], 1024]
        cosize_sP = cute.cosize(self.sP_layout) if const_expr(self.sP_layout is not None) else 0
        sP_struct = cute.struct.Align[cute.struct.MemRange[self.pv_dtype, cosize_sP], 1024]
        route_mask_struct = cute.struct.Align[
            cute.struct.MemRange[Int32, 4], 16
        ]
        route_sums_struct = cute.struct.Align[
            cute.struct.MemRange[Float32, 4 * self.tile_n], 16
        ]
        # 1 stage * 2 for Q pipeline (full + empty), self.num_stages*2 for K, self.num_stages*2 for V,
        mbar_ptr_Q_struct = cute.struct.MemRange[cutlass.Int64, 1 * 2]
        mbar_ptr_K_struct = cute.struct.MemRange[cutlass.Int64, self.num_stages * 2]
        mbar_ptr_V_struct = cute.struct.MemRange[cutlass.Int64, self.num_stages * 2]

        @cute.struct
        class SharedStorageQKV:
            mbar_ptr_Q: mbar_ptr_Q_struct
            mbar_ptr_K: mbar_ptr_K_struct
            mbar_ptr_V: mbar_ptr_V_struct
            sV: sV_struct
            sQ: sQ_struct
            sK: sK_struct
            sP: sP_struct
            route_mask: route_mask_struct
            route_sums: route_sums_struct

        @cute.struct
        class SharedStorageSharedQV:
            mbar_ptr_Q: mbar_ptr_Q_struct
            mbar_ptr_K: mbar_ptr_K_struct
            mbar_ptr_V: mbar_ptr_V_struct
            sQ: sQV_struct
            sK: sK_struct
            sP: sP_struct
            route_mask: route_mask_struct
            route_sums: route_sums_struct

        return SharedStorageQKV if const_expr(not self.Q_in_regs) else SharedStorageSharedQV

    @cute.jit
    def sol_attn_reduce_route_sums_lane_group(
        self,
        acc_S_mn: cute.Tensor,
        route_sums: cute.Tensor,
        warp_in_mma: Int32,
        lane: Int32,
    ):
        """Reduce route columns using the observed SM90 accumulator lane layout."""

        for col_group in cutlass.range_constexpr(self.tile_n // 8):
            base = col_group * 4
            partial0 = Float32(acc_S_mn[base]) + Float32(acc_S_mn[base + 1])
            partial1 = Float32(acc_S_mn[base + 2]) + Float32(acc_S_mn[base + 3])
            partial0 += cute.arch.shuffle_sync_down(partial0, 16)
            partial1 += cute.arch.shuffle_sync_down(partial1, 16)
            partial0 += cute.arch.shuffle_sync_down(partial0, 8)
            partial1 += cute.arch.shuffle_sync_down(partial1, 8)
            partial0 += cute.arch.shuffle_sync_down(partial0, 4)
            partial1 += cute.arch.shuffle_sync_down(partial1, 4)
            if lane < Int32(4):
                col = Int32(8 * col_group) + lane * Int32(2)
                route_sums[warp_in_mma, col] = partial0
                route_sums[warp_in_mma, col + Int32(1)] = partial1

    @cute.jit
    def sol_attn_reduce_route_sums_guarded(
        self,
        acc_S_mn: cute.Tensor,
        route_sums: cute.Tensor,
        tScS_mn: cute.Tensor,
        q_start: Int32,
        seqlen: SeqlenInfoQK,
        warp_in_mma: Int32,
        lane: Int32,
    ):
        """Fallback route-column reduction that ignores invalid q rows."""

        for off in cutlass.range_constexpr(self.tile_n):
            partial = Float32(0.0)
            for i in cutlass.range(cute.size(acc_S_mn), unroll_full=True):
                row = tScS_mn[i][0]
                col = tScS_mn[i][1]
                valid_row = q_start + row < seqlen.seqlen_q
                if col == Int32(off) and valid_row:
                    partial += Float32(acc_S_mn[i])
            warp_sum = cute.arch.warp_reduction_sum(partial)
            if lane == Int32(0):
                route_sums[warp_in_mma, off] = warp_sum

    @cute.jit
    def sol_attn_reduce_route_sums_lane_group_tail16(
        self,
        acc_S_mn: cute.Tensor,
        route_sums: cute.Tensor,
        route_col_offset: Int32,
        warp_in_mma: Int32,
        lane: Int32,
    ):
        """Reduce a physical 16-column tail route tile using the accumulator lane layout."""

        for col_group in cutlass.range_constexpr(2):
            base = col_group * 4
            partial0 = Float32(acc_S_mn[base]) + Float32(acc_S_mn[base + 1])
            partial1 = Float32(acc_S_mn[base + 2]) + Float32(acc_S_mn[base + 3])
            partial0 += cute.arch.shuffle_sync_down(partial0, 16)
            partial1 += cute.arch.shuffle_sync_down(partial1, 16)
            partial0 += cute.arch.shuffle_sync_down(partial0, 8)
            partial1 += cute.arch.shuffle_sync_down(partial1, 8)
            partial0 += cute.arch.shuffle_sync_down(partial0, 4)
            partial1 += cute.arch.shuffle_sync_down(partial1, 4)
            if lane < Int32(4):
                col = route_col_offset + Int32(8 * col_group) + lane * Int32(2)
                route_sums[warp_in_mma, col] = partial0
                route_sums[warp_in_mma, col + Int32(1)] = partial1

    @cute.jit
    def sol_attn_reduce_route_sums_static_tail(
        self,
        acc_S_mn: cute.Tensor,
        route_sums: cute.Tensor,
        tScS_mn: cute.Tensor,
        q_start: Int32,
        route_col_offset: Int32,
        seqlen: SeqlenInfoQK,
        warp_in_mma: Int32,
        lane: Int32,
        full_q_tile: bool,
    ):
        """Reduce only the compile-time-known valid columns of a static tail route group."""

        for off in cutlass.range_constexpr(self.sol_attn_static_tail_valid_count):
            route_col = route_col_offset + Int32(off)
            partial = Float32(0.0)
            for i in cutlass.range(cute.size(acc_S_mn), unroll_full=True):
                row = tScS_mn[i][0]
                col = tScS_mn[i][1]
                valid_row = True
                if not full_q_tile:
                    valid_row = q_start + row < seqlen.seqlen_q
                if col == route_col and valid_row:
                    partial += Float32(acc_S_mn[i])
            warp_sum = cute.arch.warp_reduction_sum(partial)
            if lane == Int32(0):
                route_sums[warp_in_mma, route_col] = warp_sum

    @cute.jit
    def sol_attn_reduce_route_sums_physical16(
        self,
        acc_S_mn: cute.Tensor,
        route_sums: cute.Tensor,
        tScS_mn: cute.Tensor,
        q_start: Int32,
        route_col_offset: Int32,
        seqlen: SeqlenInfoQK,
        warp_in_mma: Int32,
        lane: Int32,
        full_q_tile: bool,
    ):
        """Reduce a physically 16-column route tile into the 64-column route_sums buffer."""

        for off in cutlass.range_constexpr(16):
            route_col = route_col_offset + Int32(off)
            partial = Float32(0.0)
            for i in cutlass.range(cute.size(acc_S_mn), unroll_full=True):
                row = tScS_mn[i][0]
                col = tScS_mn[i][1]
                valid_row = True
                if not full_q_tile:
                    valid_row = q_start + row < seqlen.seqlen_q
                if col == route_col and valid_row:
                    partial += Float32(acc_S_mn[i])
            warp_sum = cute.arch.warp_reduction_sum(partial)
            if lane == Int32(0):
                route_sums[warp_in_mma, route_col] = warp_sum

    @cute.jit
    def sol_attn_build_route_mask_from_acc(
        self,
        acc_S: cute.Tensor,
        route_sums: cute.Tensor,
        tScS_mn: cute.Tensor,
        m_block: Int32,
        group_start_n_block: Int32,
        valid_count: Int32,
        route_col_offset: Int32,
        seqlen: SeqlenInfoQK,
        batch_idx: Int32,
        head_idx: Int32,
        mGlobalThresh: cute.Tensor,
        softmax_scale_log2: Float32,
        sink_range: Int32,
        assume_full_route_group: cutlass.Constexpr[bool] = False,
        physical_route_tile_n: cutlass.Constexpr[int] = 64,
        route_mask_words_override: cutlass.Constexpr[int] = 0,
    ):
        """Build the exact mask from the distributed route QK accumulator.

        WGMMA accumulators are distributed across the 128 consumer threads.
        Each consumer warp first reduces its local contribution per route
        column, then one consumer thread combines the four warp partials into
        the CTA-local bitmask.
        """

        tidx, _, _ = cute.arch.thread_idx()
        consumer_tidx = tidx
        warp_in_mma = consumer_tidx // cute.arch.WARP_SIZE
        lane = cute.arch.lane_idx()
        acc_S_mn = layout_utils.reshape_acc_to_mn(acc_S)

        q_start = m_block * self.tile_m
        q_len_i32 = seqlen.seqlen_q - q_start
        if q_len_i32 > Int32(self.tile_m):
            q_len_i32 = Int32(self.tile_m)
        q_len = Float32(q_len_i32)

        full_q_tile = q_len_i32 == Int32(self.tile_m)
        if const_expr(self.sol_attn_tail16_lane_group_route_reduce and physical_route_tile_n == 16):
            if full_q_tile:
                self.sol_attn_reduce_route_sums_lane_group_tail16(
                    acc_S_mn,
                    route_sums,
                    route_col_offset,
                    warp_in_mma,
                    lane,
                )
            else:
                self.sol_attn_reduce_route_sums_static_tail(
                    acc_S_mn,
                    route_sums,
                    tScS_mn,
                    q_start,
                    route_col_offset,
                    seqlen,
                    warp_in_mma,
                    lane,
                    full_q_tile,
                )
        elif const_expr(physical_route_tile_n == 16):
            self.sol_attn_reduce_route_sums_physical16(
                acc_S_mn,
                route_sums,
                tScS_mn,
                q_start,
                route_col_offset,
                seqlen,
                warp_in_mma,
                lane,
                full_q_tile,
            )
        elif const_expr(self.sol_attn_assume_lane_group_route_reduce):
            self.sol_attn_reduce_route_sums_lane_group(acc_S_mn, route_sums, warp_in_mma, lane)
        elif const_expr(self.sol_attn_lane_group_route_reduce):
            if full_q_tile:
                self.sol_attn_reduce_route_sums_lane_group(acc_S_mn, route_sums, warp_in_mma, lane)
            else:
                self.sol_attn_reduce_route_sums_guarded(
                    acc_S_mn,
                    route_sums,
                    tScS_mn,
                    q_start,
                    seqlen,
                    warp_in_mma,
                    lane,
                )
        else:
            for off in cutlass.range_constexpr(self.tile_n):
                partial = Float32(0.0)
                for i in cutlass.range(cute.size(acc_S_mn), unroll_full=True):
                    row = tScS_mn[i][0]
                    col = tScS_mn[i][1]
                    valid_row = q_start + row < seqlen.seqlen_q
                    if col == Int32(off) and valid_row:
                        partial += Float32(acc_S_mn[i])
                warp_sum = cute.arch.warp_reduction_sum(partial)
                if lane == Int32(0):
                    route_sums[warp_in_mma, off] = warp_sum

        if const_expr(self.sol_attn_route_sum_arrive_overlap and self.sol_attn_warp_route_mask):
            if warp_in_mma == Int32(0):
                cute.arch.barrier(
                    barrier_id=SOL_ATTN_ROUTE_SUM_BARRIER_ID,
                    number_of_threads=self.num_mma_threads,
                )
            else:
                cute.arch.barrier_arrive(
                    barrier_id=SOL_ATTN_ROUTE_SUM_BARRIER_ID,
                    number_of_threads=self.num_mma_threads,
                )
        else:
            cute.arch.barrier(
                barrier_id=SOL_ATTN_ROUTE_SUM_BARRIER_ID,
                number_of_threads=self.num_mma_threads,
            )

        mask0 = Int32(0)
        mask1 = Int32(0)
        mask2 = Int32(0)
        mask3 = Int32(0)
        thresh = Float32(mGlobalThresh[m_block, head_idx, batch_idx])
        if const_expr(self.sol_attn_warp_route_mask):
            route_mask_words = self.sol_attn_group_words
            if const_expr(route_mask_words_override != 0):
                route_mask_words = route_mask_words_override
            if const_expr(self.sol_attn_tail_route_mask_words1 and not assume_full_route_group):
                route_mask_words = 1
            build_mask = warp_in_mma == Int32(0)
            if build_mask:
                sink_enabled = sink_range != Int32(0)
                sink_start_block = sink_range & Int32(0xFFFF)
                sink_end_block = (sink_range >> Int32(16)) & Int32(0xFFFF)
                if const_expr(self.sol_attn_packed_route_reduction):
                    off0 = lane
                    off1 = Int32(32) + lane
                    route_col0 = route_col_offset + off0
                    route_col1 = route_col_offset + off1
                    col_sum0 = Float32(route_sums[0, route_col0]) + Float32(
                        route_sums[1, route_col0]
                    )
                    col_sum1 = Float32(route_sums[0, route_col1]) + Float32(
                        route_sums[1, route_col1]
                    )
                    col_sum0 += Float32(route_sums[2, route_col0])
                    col_sum1 += Float32(route_sums[2, route_col1])
                    col_sum0 += Float32(route_sums[3, route_col0])
                    col_sum1 += Float32(route_sums[3, route_col1])
                    col_mean0 = col_sum0 * softmax_scale_log2 / q_len
                    col_mean1 = col_sum1 * softmax_scale_log2 / q_len
                    exact0 = sol_attn_selector.sol_attn_route_is_exact(
                        m_block,
                        group_start_n_block + off0,
                        col_mean0,
                        thresh,
                        True,
                    )
                    exact1 = sol_attn_selector.sol_attn_route_is_exact(
                        m_block,
                        group_start_n_block + off1,
                        col_mean1,
                        thresh,
                        True,
                    )
                    if sink_enabled:
                        exact0 = exact0 or (
                            group_start_n_block + off0 >= sink_start_block
                            and group_start_n_block + off0 < sink_end_block
                        )
                        exact1 = exact1 or (
                            group_start_n_block + off1 >= sink_start_block
                            and group_start_n_block + off1 < sink_end_block
                        )
                    word_bits0 = Int32(cute.arch.vote_ballot_sync(exact0))
                    word_bits1 = Int32(cute.arch.vote_ballot_sync(exact1))
                    if lane == Int32(0):
                        mask0 = word_bits0
                        mask1 = word_bits1
                else:
                    for word in cutlass.range_constexpr(route_mask_words):
                        off = Int32(word * 32) + lane
                        route_col = route_col_offset + off
                        valid = True
                        if const_expr(
                            not (
                                self.sol_attn_assume_full_route_groups
                                or assume_full_route_group
                            )
                        ):
                            valid = off < valid_count
                        exact = False
                        if valid:
                            col_sum = (
                                Float32(route_sums[0, route_col])
                                + Float32(route_sums[1, route_col])
                                + Float32(route_sums[2, route_col])
                                + Float32(route_sums[3, route_col])
                            )
                            col_mean = col_sum * softmax_scale_log2 / q_len
                            exact = sol_attn_selector.sol_attn_route_is_exact(
                                m_block,
                                group_start_n_block + off,
                                col_mean,
                                thresh,
                                valid,
                            )
                            if sink_enabled:
                                exact = exact or (
                                    group_start_n_block + off
                                    >= sink_start_block
                                    and group_start_n_block + off
                                    < sink_end_block
                                )
                        if const_expr(self.sol_attn_approx_colmask):
                            column_mask = -Float32.inf
                            if valid and not exact:
                                column_mask = Float32(0.0)
                            route_sums[0, route_col] = column_mask
                        if const_expr(self.sol_attn_ballot_mask):
                            word_bits = Int32(cute.arch.vote_ballot_sync(exact))
                        else:
                            word_bits = Int32(0)
                            if exact:
                                word_bits = Int32(1) << lane
                            word_bits = word_bits | cute.arch.shuffle_sync_down(
                                word_bits, 16
                            )
                            word_bits = word_bits | cute.arch.shuffle_sync_down(
                                word_bits, 8
                            )
                            word_bits = word_bits | cute.arch.shuffle_sync_down(
                                word_bits, 4
                            )
                            word_bits = word_bits | cute.arch.shuffle_sync_down(
                                word_bits, 2
                            )
                            word_bits = word_bits | cute.arch.shuffle_sync_down(
                                word_bits, 1
                            )
                        if lane == Int32(0):
                            if const_expr(word == 0):
                                mask0 = word_bits
                            elif const_expr(word == 1):
                                mask1 = word_bits
                            elif const_expr(word == 2):
                                mask2 = word_bits
                            else:
                                mask3 = word_bits
        else:
            sink_enabled = sink_range != Int32(0)
            sink_start_block = sink_range & Int32(0xFFFF)
            sink_end_block = (sink_range >> Int32(16)) & Int32(0xFFFF)
            for off in cutlass.range_constexpr(self.sol_attn_group_size):
                route_col = route_col_offset + Int32(off)
                valid = True
                if const_expr(
                    not (self.sol_attn_assume_full_route_groups or assume_full_route_group)
                ):
                    valid = Int32(off) < valid_count
                col_sum = (
                    Float32(route_sums[0, route_col])
                    + Float32(route_sums[1, route_col])
                    + Float32(route_sums[2, route_col])
                    + Float32(route_sums[3, route_col])
                )
                if valid:
                    col_mean = col_sum * softmax_scale_log2 / q_len
                    exact = sol_attn_selector.sol_attn_route_is_exact(
                        m_block,
                        group_start_n_block + Int32(off),
                        col_mean,
                        thresh,
                        valid,
                    )
                    if sink_enabled:
                        exact = exact or (
                            group_start_n_block + Int32(off)
                            >= sink_start_block
                            and group_start_n_block + Int32(off)
                            < sink_end_block
                        )
                    if exact:
                        mask0, mask1, mask2, mask3 = (
                            sol_attn_selector.sol_attn_set_exact_bit(
                                mask0, mask1, mask2, mask3, Int32(off)
                            )
                        )

        return mask0, mask1, mask2, mask3

    @cute.jit
    def sol_attn_mask_route_approx_columns(
        self,
        acc_S: cute.Tensor,
        route_sums: cute.Tensor,
        tScS_mn: cute.Tensor,
        valid_count: Int32,
        route_col_offset: Int32,
        mask0: Int32,
        mask1: Int32,
        mask2: Int32,
        mask3: Int32,
        assume_full_route_group: cutlass.Constexpr[bool] = False,
        route_mask_words_override: cutlass.Constexpr[int] = 0,
    ):
        """Keep only approximate route columns in the route score tile."""

        acc_S_mn = layout_utils.reshape_acc_to_mn(acc_S)
        if const_expr(self.sol_attn_approx_colmask):
            for i in cutlass.range(cute.size(acc_S_mn), unroll_full=True):
                col = tScS_mn[i][1]
                acc_S_mn[i] = Float32(acc_S_mn[i]) + Float32(route_sums[0, col])
        else:
            for i in cutlass.range(cute.size(acc_S_mn), unroll_full=True):
                col = tScS_mn[i][1]
                group_col = col - route_col_offset
                valid = True
                if const_expr(self.sol_attn_group_size != self.tile_n):
                    valid = group_col >= Int32(0)
                    if valid:
                        valid = group_col < valid_count
                elif const_expr(
                    not (self.sol_attn_assume_full_route_groups or assume_full_route_group)
                ):
                    valid = col < valid_count
                exact = False
                if valid:
                    route_mask_words = self.sol_attn_group_words
                    if const_expr(route_mask_words_override != 0):
                        route_mask_words = route_mask_words_override
                    if const_expr(
                        self.sol_attn_tail_route_mask_words1
                        and not assume_full_route_group
                    ):
                        route_mask_words = 1
                    exact = sol_attn_selector.sol_attn_test_exact_bit_limited_words(
                        mask0, mask1, mask2, mask3, group_col, route_mask_words
                    )
                if (not valid) or exact:
                    acc_S_mn[i] = -Float32.inf

    @cute.jit
    def sol_attn_expand_route_acc_to_full_tile(
        self,
        acc_S: cute.Tensor,
        acc_S_full_ref: cute.Tensor,
        tScS_mn: cute.Tensor,
        tScS_full_mn: cute.Tensor,
    ) -> cute.Tensor:
        """Expand a narrow physical route accumulator into a full 64-column P tile."""

        acc_S_mn = layout_utils.reshape_acc_to_mn(acc_S)
        acc_S_full = cute.make_rmem_tensor_like(acc_S_full_ref, Float32)
        acc_S_full_mn = layout_utils.reshape_acc_to_mn(acc_S_full)
        for i in cutlass.range(cute.size(acc_S_full_mn), unroll_full=True):
            row = tScS_full_mn[i][0]
            col = tScS_full_mn[i][1]
            value = -Float32.inf
            if col < Int32(16):
                for j in cutlass.range(cute.size(acc_S_mn), unroll_full=True):
                    row16 = tScS_mn[j][0]
                    col16 = tScS_mn[j][1]
                    if row == row16 and col == col16:
                        value = Float32(acc_S_mn[j])
            acc_S_full_mn[i] = value
        return acc_S_full

    @cute.jit
    def sol_attn_expand_route_prob_to_full_tile(
        self,
        acc_P: cute.Tensor,
        acc_S_full_ref: cute.Tensor,
        tScS_mn: cute.Tensor,
        tScS_full_mn: cute.Tensor,
    ) -> cute.Tensor:
        """Expand a compact route probability tile into the full PV A fragment."""

        acc_P_mn = layout_utils.reshape_acc_to_mn(acc_P)
        acc_P_full = cute.make_rmem_tensor_like(acc_S_full_ref, Float32)
        acc_P_full_mn = layout_utils.reshape_acc_to_mn(acc_P_full)
        for i in cutlass.range(cute.size(acc_P_full_mn), unroll_full=True):
            row = tScS_full_mn[i][0]
            col = tScS_full_mn[i][1]
            value = Float32(0.0)
            if col < Int32(16):
                for j in cutlass.range(cute.size(acc_P_mn), unroll_full=True):
                    row16 = tScS_mn[j][0]
                    col16 = tScS_mn[j][1]
                    if row == row16 and col == col16:
                        value = Float32(acc_P_mn[j])
            acc_P_full_mn[i] = value
        return acc_P_full

    @cute.jit
    def sol_attn_apply_route_current_lens_to_row_sum(
        self,
        acc_S: cute.Tensor,
        tScS_mn: cute.Tensor,
        group_start_n_block: Int32,
        valid_count: Int32,
        route_col_offset: Int32,
        seqlen: SeqlenInfoQK,
        softmax: Softmax,
    ):
        """Correct route approx denominator for VC tiles that are block sums."""

        acc_S_mn = layout_utils.reshape_acc_to_mn(acc_S)
        last_n_block = (
            (seqlen.seqlen_k + Int32(self.tile_n - 1)) // Int32(self.tile_n)
        ) - Int32(1)
        tail_len = seqlen.seqlen_k - last_n_block * Int32(self.tile_n)
        for r in cutlass.range(cute.size(softmax.row_sum), unroll_full=True):
            extra = Float32(0.0)
            for c in cutlass.range(cute.size(acc_S_mn.shape[1]), unroll_full=True):
                col = tScS_mn[r, c][1]
                group_col = col - route_col_offset
                valid = group_col >= Int32(0)
                if valid:
                    valid = group_col < valid_count
                if valid:
                    kv_block_idx = group_start_n_block + group_col
                    current_len = Int32(self.tile_n)
                    if kv_block_idx == last_n_block:
                        current_len = tail_len
                    extra += Float32(acc_S_mn[r, c]) * (Float32(current_len) - Float32(1.0))
            softmax.row_sum[r] += extra

    @cute.jit
    def sol_attn_apply_route_current_lens_to_row_sum_fast(
        self,
        acc_S: cute.Tensor,
        tScS_mn: cute.Tensor,
        row_sum_prev: cute.Tensor,
        row_scale: cute.Tensor,
        group_start_n_block: Int32,
        valid_count: Int32,
        route_col_offset: Int32,
        seqlen: SeqlenInfoQK,
        softmax: Softmax,
        is_first_block: cutlass.Constexpr[bool],
    ):
        """Fast denominator correction for full-length route groups."""

        last_n_block = (
            (seqlen.seqlen_k + Int32(self.tile_n - 1)) // Int32(self.tile_n)
        ) - Int32(1)
        tail_len = seqlen.seqlen_k - last_n_block * Int32(self.tile_n)
        group_end = group_start_n_block + valid_count
        full_len_group = (tail_len == Int32(self.tile_n)) or (group_end <= last_n_block)

        if full_len_group:
            block_extra = Float32(self.tile_n - 1)
            for r in cutlass.range(cute.size(softmax.row_sum), unroll_full=True):
                prev_scaled = Float32(0.0)
                if const_expr(not is_first_block):
                    prev_scaled = Float32(row_sum_prev[r]) * Float32(row_scale[r])
                route_row_sum = Float32(softmax.row_sum[r]) - prev_scaled
                softmax.row_sum[r] += route_row_sum * block_extra
        else:
            # Tail blocks need per-column current_len because the last route
            # column may represent fewer than tile_n values.
            self.sol_attn_apply_route_current_lens_to_row_sum(
                acc_S,
                tScS_mn,
                group_start_n_block,
                valid_count,
                route_col_offset,
                seqlen,
                softmax,
            )

    @cute.jit
    def __call__(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mO: cute.Tensor,
        mKC: cute.Tensor,
        mVC: cute.Tensor,
        mGlobalThresh: cute.Tensor,
        mLSE: Optional[cute.Tensor],
        softmax_scale: Float32,
        sink_range: Int32,
        stream: cuda.CUstream = None,
    ):
        """Configure and launch the Hopper Sol-Attn kernel."""

        mCuSeqlensQ = None
        mCuSeqlensK = None
        mSeqUsedQ = None
        mSeqUsedK = None
        mPageTable = None
        window_size_left = None
        window_size_right = None
        learnable_sink = None
        blocksparse_tensors = None
        piecewise_k = None
        piecewise_v = None
        aux_tensors = None
        self.varlen_q = mCuSeqlensQ is not None or mSeqUsedQ is not None

        mQ, mK, mV, mO, mKC, mVC, mGlobalThresh = [
            assume_tensor_aligned(t)
            for t in (mQ, mK, mV, mO, mKC, mVC, mGlobalThresh)
        ]
        if const_expr(piecewise_k is not None):
            piecewise_k, piecewise_v = [
                assume_tensor_aligned(t) for t in (piecewise_k, piecewise_v)
            ]
        SOL_ATTN_BTHD_TRANSPOSE = [1, 3, 2, 0]
        SOL_ATTN_BNH_TRANSPOSE = [1, 2, 0]
        mQ, mK, mV, mO, mKC, mVC = [
            layout_utils.select(t, SOL_ATTN_BTHD_TRANSPOSE)
            for t in (mQ, mK, mV, mO, mKC, mVC)
        ]
        mGlobalThresh = layout_utils.select(
            mGlobalThresh, SOL_ATTN_BNH_TRANSPOSE
        )
        if const_expr(piecewise_k is not None):
            piecewise_k, piecewise_v = [
                layout_utils.select(t, SOL_ATTN_BTHD_TRANSPOSE)
                for t in (piecewise_k, piecewise_v)
            ]
        LSE_layout_transpose = [1, 2, 0]
        mLSE = (
            layout_utils.select(mLSE, LSE_layout_transpose)
            if const_expr(mLSE is not None)
            else None
        )

        tiled_mma_qk, tiled_mma_pv = self._get_tiled_mma()
        self.num_mma_threads = tiled_mma_qk.size
        self.num_threads_per_warp_group = 128
        self.num_wg_mma = self.num_mma_threads // self.num_threads_per_warp_group
        assert self.num_wg_mma in [1, 2, 3]
        if const_expr(self.num_wg_mma != 1):
            raise NotImplementedError("SOL_ATTN SM90 path requires exactly one MMA warpgroup")
        self.num_threads = self.num_threads_per_warp_group
        self.num_producer_threads = 32
        self.num_Q_load_threads = self.num_threads_per_warp_group  # If not TMA_Q
        self.num_epilogue_threads = self.num_mma_threads
        self.num_mma_regs, self.num_producer_regs = {1: (256, 56), 2: (240, 24), 3: (160, 32)}[
            self.num_wg_mma
        ]
        self.use_block_sparsity = cutlass.const_expr(blocksparse_tensors is not None)
        self.has_piecewise_kv = cutlass.const_expr(piecewise_k is not None)
        if const_expr(self.use_block_sparsity):
            raise NotImplementedError("one-warpgroup SOL_ATTN path does not support block sparsity")
        if const_expr(self.has_piecewise_kv):
            raise NotImplementedError("one-warpgroup SOL_ATTN path does not support piecewise KV")

        self.use_scheduler_barrier = self.num_wg_mma == 2
        self.use_tma_Q = self.arch >= Arch.sm_90 and not (
            self.pack_gqa and self.tile_m % self.qhead_per_kvhead != 0
        )
        if const_expr(not self.use_tma_Q):
            raise NotImplementedError("one-warpgroup SOL_ATTN path requires TMA Q/O")
        # FP32 split partials require a direct register-to-global epilogue.
        # A BF16 split partial matches V/O dtype and can reuse the shared-memory
        # plus TMA-O epilogue.
        self.use_tma_O = (
            self.sol_attn_num_splits == 1 or mO.element_type == self.dtype
        )
        # Producer needs more registers when doing cp.async Q or KV loads
        if const_expr(self.num_wg_mma == 2 and (not self.use_tma_Q or not self.use_tma_KV)):
            self.num_mma_regs, self.num_producer_regs = 224, 40
        if const_expr(self.sol_attn_mma_regs_override is not None):
            self.num_mma_regs = self.sol_attn_mma_regs_override
        self.rescale_O_before_gemm = False
        self._setup_attributes()
        # TODO: we prob don't need most of what's in _setup_attributes
        self.sQ_layout, self.sK_layout, self.sV_layout, self.sO_layout = [
            sm90_utils.make_smem_layout(mX.element_type, LayoutEnum.ROW_MAJOR, shape, stage)
            for mX, shape, stage in [
                (mQ, (self.tile_m, self.tile_hdim), None),
                (mK, (self.tile_n, self.tile_hdim), self.num_stages),
                (mV, (self.tile_n, self.tile_hdimv), self.num_stages),
                # sO always holds the BF16 PV epilogue tile.  Split-KV's
                # global mO is an FP32 partial workspace, so derive this
                # shared-memory layout from V instead of global O.
                (mV, (self.tile_m, self.tile_hdimv), None),
            ]
        ]
        self.sP_layout = None
        if const_expr(not self.mma_pv_is_rs):
            self.sP_layout = sm90_utils.make_smem_layout(
                mV.element_type, LayoutEnum.ROW_MAJOR, (self.tile_m, self.tile_n)
            )
        SharedStorage = self._get_shared_storage_cls()

        mQ_og, mO_og = mQ, mO
        if const_expr(self.pack_gqa):
            nheads_kv = mK.shape[2]
            mQ = pack_gqa_layout(mQ, self.qhead_per_kvhead, nheads_kv, head_idx=2)
            mO = pack_gqa_layout(mO, self.qhead_per_kvhead, nheads_kv, head_idx=2)
            if const_expr(mLSE is not None):
                mLSE = pack_gqa_layout(mLSE, self.qhead_per_kvhead, nheads_kv, head_idx=1)

        # TMA
        gmem_tiled_copy_Q = cpasync.CopyBulkTensorTileG2SOp()
        gmem_tiled_copy_KV = cpasync.CopyBulkTensorTileG2SOp()  # Might multicast
        gmem_tiled_copy_O = cpasync.CopyBulkTensorTileS2GOp()
        self.tma_copy_bytes = {
            name: cute.size_in_bytes(mX.element_type, cute.select(layout, mode=[0, 1]))
            for name, mX, layout in [
                ("Q", mQ, self.sQ_layout),
                ("K", mK, self.sK_layout),
                ("V", mV, self.sV_layout),
            ]
        }
        make_tiled_tma_atom_fn = (
            partial(make_packgqa_tiled_tma_atom, qhead_per_kvhead=self.qhead_per_kvhead, head_idx=2)
            if const_expr(self.pack_gqa)
            else cpasync.make_tiled_tma_atom
        )
        tma_atom_Q, tma_tensor_Q = None, None
        if const_expr(self.use_tma_Q):
            tma_atom_Q, tma_tensor_Q = make_tiled_tma_atom_fn(
                gmem_tiled_copy_Q,
                mQ_og if const_expr(self.pack_gqa) else mQ,
                self.sQ_layout,
                (self.tile_m, self.tile_hdim),  # No mcast
            )
        tma_atom_K, tma_tensor_K = None, None
        tma_atom_V, tma_tensor_V = None, None
        tma_atom_KC, tma_tensor_KC = None, None
        tma_atom_VC, tma_tensor_VC = None, None
        tma_atom_K2, tma_tensor_K2 = None, None
        tma_atom_V2, tma_tensor_V2 = None, None
        if const_expr(self.use_tma_KV):
            tma_atom_K, tma_tensor_K = cpasync.make_tiled_tma_atom(
                gmem_tiled_copy_KV,
                mK,
                cute.select(self.sK_layout, mode=[0, 1]),
                (self.tile_n, self.tile_hdim),
                1,  # No mcast for now
            )
            tma_atom_V, tma_tensor_V = cpasync.make_tiled_tma_atom(
                gmem_tiled_copy_KV,
                mV,
                cute.select(self.sV_layout, mode=[0, 1]),
                (self.tile_n, self.tile_hdimv),
                1,  # No mcast for now
            )
            tma_atom_KC, tma_tensor_KC = cpasync.make_tiled_tma_atom(
                gmem_tiled_copy_KV,
                mKC,
                cute.select(self.sK_layout, mode=[0, 1]),
                (self.tile_n, self.tile_hdim),
                1,
            )
            tma_atom_VC, tma_tensor_VC = cpasync.make_tiled_tma_atom(
                gmem_tiled_copy_KV,
                mVC,
                cute.select(self.sV_layout, mode=[0, 1]),
                (self.tile_n, self.tile_hdimv),
                1,
            )
            if const_expr(self.has_piecewise_kv):
                tma_atom_K2, tma_tensor_K2 = cpasync.make_tiled_tma_atom(
                    gmem_tiled_copy_KV,
                    piecewise_k,
                    cute.select(self.sK_layout, mode=[0, 1]),
                    (self.tile_n, self.tile_hdim),
                    1,
                )
                tma_atom_V2, tma_tensor_V2 = cpasync.make_tiled_tma_atom(
                    gmem_tiled_copy_KV,
                    piecewise_v,
                    cute.select(self.sV_layout, mode=[0, 1]),
                    (self.tile_n, self.tile_hdimv),
                    1,
                )
        tma_atom_O, tma_tensor_O = None, None
        if const_expr(self.use_tma_O):
            mO_tma = mO_og if const_expr(self.pack_gqa) else mO
            if const_expr(self.varlen_q):
                mO_tma = copy_utils.create_ragged_tensor_for_tma(
                    mO_tma, ragged_dim=0, ptr_shift=True
                )
            tma_atom_O, tma_tensor_O = make_tiled_tma_atom_fn(
                gmem_tiled_copy_O,
                mO_tma,
                self.sO_layout,
                (self.tile_m, self.tile_hdimv),  # No mcast
            )
        if const_expr(mCuSeqlensQ is not None or mSeqUsedQ is not None):
            TileScheduler = SingleTileVarlenScheduler
        else:
            TileScheduler = (
                SingleTileScheduler
                if const_expr(not self.is_causal or self.is_local)
                else SingleTileLPTScheduler
            )
        tile_sched_args = TileSchedulerArguments(
            cute.ceil_div(cute.size(mQ.shape[0]), self.tile_m),
            cute.size(mQ.shape[2]),
            cute.size(mQ.shape[3])
            if const_expr(mCuSeqlensQ is None)
            else cute.size(mCuSeqlensQ.shape[0] - 1),
            self.sol_attn_num_splits,
            cute.size(mK.shape[0])
            if const_expr(mPageTable is None)
            else mK.shape[0] * mPageTable.shape[1],
            mQ.shape[1],
            mV.shape[1],
            total_q=cute.size(mQ.shape[0])
            if const_expr(mCuSeqlensQ is not None)
            else cute.size(mQ.shape[0]) * cute.size(mQ.shape[3]),
            tile_shape_mn=(self.tile_m, self.tile_n),
            mCuSeqlensQ=mCuSeqlensQ,
            mSeqUsedQ=mSeqUsedQ,
            qhead_per_kvhead_packgqa=self.qhead_per_kvhead if const_expr(self.pack_gqa) else 1,
            element_size=self.dtype.width // 8,
            is_persistent=False,
            lpt=self.is_causal or self.is_local,
            is_split_kv=self.sol_attn_num_splits > 1,
        )
        tile_sched_params = TileScheduler.to_underlying_arguments(tile_sched_args)
        grid_dim = TileScheduler.get_grid_shape(tile_sched_params)
        softmax_scale_log2, softmax_scale = utils.compute_softmax_scale_log2(
            softmax_scale, self.score_mod
        )
        window_size_left = Int32(window_size_left) if window_size_left is not None else None
        window_size_right = Int32(window_size_right) if window_size_right is not None else None
        fastdiv_mods = utils.compute_fastdiv_mods(
            mQ, mK, self.qhead_per_kvhead, self.pack_gqa, aux_tensors, mPageTable
        )

        self.kernel(
            tma_tensor_Q if const_expr(self.use_tma_Q) else mQ,
            tma_tensor_K if const_expr(self.use_tma_KV) else mK,
            tma_tensor_V if const_expr(self.use_tma_KV) else mV,
            tma_tensor_KC if const_expr(self.use_tma_KV) else mKC,
            tma_tensor_VC if const_expr(self.use_tma_KV) else mVC,
            tma_tensor_K2 if const_expr(self.has_piecewise_kv) else None,
            tma_tensor_V2 if const_expr(self.has_piecewise_kv) else None,
            tma_tensor_O if const_expr(self.use_tma_O) else mO,
            mGlobalThresh,
            mLSE,
            mCuSeqlensQ,
            mCuSeqlensK,
            mSeqUsedQ,
            mSeqUsedK,
            mPageTable,
            tma_atom_Q,
            tma_atom_K,
            tma_atom_V,
            tma_atom_KC,
            tma_atom_VC,
            tma_atom_K2,
            tma_atom_V2,
            tma_atom_O,
            softmax_scale_log2,
            softmax_scale,
            sink_range,
            window_size_left,
            window_size_right,
            learnable_sink,
            blocksparse_tensors,
            self.sQ_layout,
            self.sK_layout,
            self.sV_layout,
            self.sO_layout,
            self.sP_layout,
            self.gmem_tiled_copy_Q,
            self.gmem_tiled_copy_K,
            self.gmem_tiled_copy_V,
            self.gmem_tiled_copy_O,
            tiled_mma_qk,
            tiled_mma_pv,
            tile_sched_params,
            TileScheduler,
            SharedStorage,
            fastdiv_mods,
        ).launch(
            grid=grid_dim,
            block=[self.num_threads, 1, 1],
            stream=stream,
            min_blocks_per_mp=1,
        )

    @cute.kernel
    def kernel(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mKC: cute.Tensor,
        mVC: cute.Tensor,
        mK2: Optional[cute.Tensor],
        mV2: Optional[cute.Tensor],
        mO: cute.Tensor,
        mGlobalThresh: cute.Tensor,
        mLSE: Optional[cute.Tensor],
        mCuSeqlensQ: Optional[cute.Tensor],
        mCuSeqlensK: Optional[cute.Tensor],
        mSeqUsedQ: Optional[cute.Tensor],
        mSeqUsedK: Optional[cute.Tensor],
        mPageTable: Optional[cute.Tensor],
        tma_atom_Q: Optional[cute.CopyAtom],
        tma_atom_K: Optional[cute.CopyAtom],
        tma_atom_V: Optional[cute.CopyAtom],
        tma_atom_KC: Optional[cute.CopyAtom],
        tma_atom_VC: Optional[cute.CopyAtom],
        tma_atom_K2: Optional[cute.CopyAtom],
        tma_atom_V2: Optional[cute.CopyAtom],
        tma_atom_O: Optional[cute.CopyAtom],
        softmax_scale_log2: Float32,
        softmax_scale: Optional[Float32],
        sink_range: Int32,
        window_size_left: Optional[Int32],
        window_size_right: Optional[Int32],
        learnable_sink: Optional[cute.Tensor],
        blocksparse_tensors: Optional[BlockSparseTensors],
        sQ_layout: cute.ComposedLayout,
        sK_layout: cute.ComposedLayout,
        sV_layout: cute.ComposedLayout,
        sO_layout: cute.ComposedLayout,
        sP_layout: cute.ComposedLayout | None,
        gmem_tiled_copy_Q: cute.TiledCopy,
        gmem_tiled_copy_K: cute.TiledCopy,
        gmem_tiled_copy_V: cute.TiledCopy,
        gmem_tiled_copy_O: cute.TiledCopy,
        tiled_mma_qk: cute.TiledMma,
        tiled_mma_pv: cute.TiledMma,
        tile_sched_params: ParamsBase,
        TileScheduler: cutlass.Constexpr[Callable],
        SharedStorage: cutlass.Constexpr[Callable],
        fastdiv_mods=None,
    ):
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        # Prefetch tma descriptor
        if warp_idx == 0:
            for tma_atom in (
                tma_atom_Q,
                tma_atom_K,
                tma_atom_V,
                tma_atom_KC,
                tma_atom_VC,
                tma_atom_K2,
                tma_atom_V2,
                tma_atom_O,
            ):
                if const_expr(tma_atom is not None):
                    cpasync.prefetch_descriptor(tma_atom)

        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)

        # Mbarrier / pipeline init
        mbar_ptr_Q = storage.mbar_ptr_Q.data_ptr()

        ThreadCooperativeGroup = partial(pipeline.CooperativeGroup, pipeline.Agent.Thread)
        tma_warp = ThreadCooperativeGroup(1)
        load_threads = ThreadCooperativeGroup(self.num_threads_per_warp_group)
        mma_warps = ThreadCooperativeGroup(self.num_mma_threads // cute.arch.WARP_SIZE)
        if const_expr(self.use_tma_Q):
            pipeline_q = pipeline_custom.PipelineTmaAsync.create(
                barrier_storage=mbar_ptr_Q,
                num_stages=1,
                producer_group=tma_warp,
                consumer_group=mma_warps,
                tx_count=self.tma_copy_bytes["Q"],
                defer_sync=True,
            )
        else:
            pipeline_q = pipeline_custom.PipelineCpAsync.create(
                barrier_storage=mbar_ptr_Q,
                num_stages=1,
                producer_group=load_threads,
                consumer_group=mma_warps,
                defer_sync=True,
                elect_one_release=True,
                syncwarp_before_release=False,
            )

        if const_expr(self.use_tma_KV):
            pipeline_k = pipeline_custom.PipelineTmaAsync.create(
                barrier_storage=storage.mbar_ptr_K.data_ptr(),
                num_stages=self.num_stages,
                producer_group=tma_warp,
                consumer_group=mma_warps,
                tx_count=self.tma_copy_bytes["K"],
                defer_sync=True,
            )
            pipeline_v = pipeline_custom.PipelineTmaAsync.create(
                barrier_storage=storage.mbar_ptr_V.data_ptr(),
                num_stages=self.num_stages,
                producer_group=tma_warp,
                consumer_group=mma_warps,
                tx_count=self.tma_copy_bytes["V"],
                defer_sync=True,
            )
        else:
            pipeline_k = pipeline_custom.PipelineCpAsync.create(
                barrier_storage=storage.mbar_ptr_K.data_ptr(),
                num_stages=self.num_stages,
                producer_group=load_threads,
                consumer_group=mma_warps,
                defer_sync=True,
                elect_one_release=True,
                syncwarp_before_release=False,
            )
            pipeline_v = pipeline_custom.PipelineCpAsync.create(
                barrier_storage=storage.mbar_ptr_V.data_ptr(),
                num_stages=self.num_stages,
                producer_group=load_threads,
                consumer_group=mma_warps,
                defer_sync=True,
                elect_one_release=True,
                syncwarp_before_release=False,
            )

        # Cluster arrive after barrier init
        pipeline_init_arrive(cluster_shape_mn=self.cluster_shape_mn, is_relaxed=True)

        # ///////////////////////////////////////////////////////////////////////////////
        # Get shared memory buffer
        # ///////////////////////////////////////////////////////////////////////////////
        sQ = storage.sQ.get_tensor(sQ_layout.outer, swizzle=sQ_layout.inner)
        sK = storage.sK.get_tensor(sK_layout.outer, swizzle=sK_layout.inner)
        if const_expr(not self.Q_in_regs):
            sV = storage.sV.get_tensor(sV_layout.outer, swizzle=sV_layout.inner)
        else:
            sV = storage.sQ.get_tensor(
                sV_layout.outer, swizzle=sV_layout.inner, dtype=mV.element_type
            )
        # Transpose view of V to tensor with layout (head_dim_v, tile_n) for tiled mma
        sVt = layout_utils.transpose_view(sV)
        sP = None
        if const_expr(sP_layout is not None):
            sP = storage.sP.get_tensor(sP_layout.outer, swizzle=sP_layout.inner)
        # reuse sQ's data iterator
        sO = storage.sQ.get_tensor(sO_layout.outer, swizzle=sO_layout.inner, dtype=self.dtype)
        route_mask = storage.route_mask.get_tensor(
            cute.make_layout((4,))
        )
        route_sums = storage.route_sums.get_tensor(cute.make_layout((4, self.tile_n)))

        block_info = BlockInfo(
            self.tile_m,
            self.tile_n,
            self.is_causal,
            self.is_local,
            False,  # is_split_kv
            window_size_left,
            window_size_right,
            qhead_per_kvhead_packgqa=self.qhead_per_kvhead if const_expr(self.pack_gqa) else 1,
        )
        SeqlenInfoCls = partial(
            SeqlenInfoQK.create,
            seqlen_q_static=mQ.shape[0] if const_expr(not self.pack_gqa) else mQ.shape[0][1],
            seqlen_k_static=mK.shape[0]
            if const_expr(mPageTable is None)
            else mK.shape[0] * mPageTable.shape[1],
            mCuSeqlensQ=mCuSeqlensQ,
            mCuSeqlensK=mCuSeqlensK,
            mSeqUsedQ=mSeqUsedQ,
            mSeqUsedK=mSeqUsedK,
            # Don't need to pass in tile_mn because we won't access offset_padded
        )
        AttentionMaskCls = partial(
            AttentionMask,
            self.tile_m,
            self.tile_n,
            window_size_left=window_size_left,
            window_size_right=window_size_right,
            qhead_per_kvhead_packgqa=self.qhead_per_kvhead if const_expr(self.pack_gqa) else 1,
        )
        TileSchedulerCls = partial(TileScheduler.create, tile_sched_params)

        # Cluster wait before starting
        pipeline_init_wait(cluster_shape_mn=self.cluster_shape_mn)

        cute.arch.setmaxregister_increase(self.num_mma_regs)
        self.mma_one_warpgroup_sol_attn_route_tma(
            tiled_mma_qk,
            tiled_mma_pv,
            mQ,
            mK,
            mV,
            mKC,
            mVC,
            mO,
            mLSE,
            sQ,
            sK,
            sV,
            sVt,
            sP,
            sO,
            tma_atom_Q,
            tma_atom_K,
            tma_atom_V,
            tma_atom_KC,
            tma_atom_VC,
            gmem_tiled_copy_O,
            tma_atom_O,
            pipeline_q,
            pipeline_k,
            pipeline_v,
            SeqlenInfoCls,
            AttentionMaskCls,
            TileSchedulerCls,
            mGlobalThresh,
            route_mask,
            route_sums,
            softmax_scale_log2,
            softmax_scale,
            sink_range,
            block_info,
        )

    @cute.jit
    def epilogue_one_warpgroup_tma_o(
        self,
        acc_O: cute.Tensor,
        lse: cute.Tensor,
        mO: cute.Tensor,
        mLSE: Optional[cute.Tensor],
        sO: cute.Tensor,
        seqlen: SeqlenInfoQK,
        tma_atom_O: cute.CopyAtom,
        tiled_mma: cute.TiledMma,
        tidx: Int32,
        m_block: Int32,
        head_idx: Int32,
        batch_idx: Int32,
    ):
        """One-warpgroup TMA-O epilogue with an in-CTA store owner."""

        rO = cute.make_fragment_like(acc_O, self.dtype)
        rO.store(acc_O.load().to(self.dtype))
        cute.arch.barrier(
            barrier_id=int(NamedBarrierFwd.Epilogue),
            number_of_threads=self.num_epilogue_threads,
        )
        smem_copy_atom_O = utils.get_smem_store_atom(
            self.arch.major * 10 + self.arch.minor, self.dtype
        )
        smem_thr_copy_O = cute.make_tiled_copy_C(smem_copy_atom_O, tiled_mma).get_slice(tidx)
        taccOrO = smem_thr_copy_O.retile(rO)
        taccOsO = smem_thr_copy_O.partition_D(sO)
        cute.copy(smem_copy_atom_O, taccOrO, taccOsO)

        cO = cute.make_identity_tensor((self.tile_m, self.tile_hdimv))
        if const_expr(mLSE is not None):
            mLSE_cur = mLSE[None, head_idx, batch_idx]
            gLSE = cute.local_tile(mLSE_cur, (self.tile_m,), (m_block,))
            gLSE_expanded_layout = cute.append(
                gLSE.layout, cute.make_layout((self.tile_hdimv,), stride=(0,))
            )
            gLSE_expanded = cute.make_tensor(gLSE.iterator, gLSE_expanded_layout)
            thr_mma = tiled_mma.get_slice(tidx)
            taccOgLSE = layout_utils.reshape_acc_to_mn(thr_mma.partition_C(gLSE_expanded))
            taccOcO = layout_utils.reshape_acc_to_mn(thr_mma.partition_C(cO))
            t0accOcO = layout_utils.reshape_acc_to_mn(thr_mma.get_slice(0).partition_C(cO))
            if taccOcO[0][1] == 0:
                for m in cutlass.range_constexpr(cute.size(taccOgLSE.shape[1])):
                    if (
                        t0accOcO[m, 0][0]
                        < seqlen.seqlen_q - m_block * self.tile_m - taccOcO[0][0]
                    ):
                        taccOgLSE[m, 0] = lse[m]

        mO_cur = mO[None, None, head_idx, batch_idx]
        cute.arch.fence_view_async_shared()
        cute.arch.barrier(
            barrier_id=int(NamedBarrierFwd.Epilogue),
            number_of_threads=self.num_epilogue_threads,
        )
        gO = cute.local_tile(mO_cur, (self.tile_m, self.tile_hdimv), (m_block, 0))
        store_O, _, _ = copy_utils.tma_get_copy_fn(
            tma_atom_O, 0, cute.make_layout(1), sO, gO, single_stage=True
        )
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        if warp_idx == Int32(0):
            store_O()
            cute.arch.cp_async_bulk_commit_group()
            cute.arch.cp_async_bulk_wait_group(0, read=True)

    @cute.jit
    def epilogue_one_warpgroup_split_partial(
        self,
        acc_O: cute.Tensor,
        lse: cute.Tensor,
        mO: cute.Tensor,
        mLSE: cute.Tensor,
        seqlen: SeqlenInfoQK,
        tiled_mma: cute.TiledMma,
        tidx: Int32,
        m_block: Int32,
        partial_head_idx: Int32,
        batch_idx: Int32,
    ):
        """Write one normalized FP32 split partial and its natural-log LSE.

        ``mO`` and ``mLSE`` use a physical split-head dimension.  The caller
        maps ``(split, head)`` to ``partial_head_idx``; a later combine kernel
        performs the log-sum-exp weighted reduction across that dimension.
        """

        mO_cur = mO[None, None, partial_head_idx, batch_idx]
        gO = cute.local_tile(
            mO_cur, (self.tile_m, self.tile_hdimv), (m_block, 0)
        )
        copy_atom = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            Float32,
            num_bits_per_copy=32,
        )
        tiled_copy = cute.make_tiled_copy_C(copy_atom, tiled_mma)
        rO = cute.make_rmem_tensor_like(acc_O, Float32)
        rO.store(acc_O.load())
        tOrO = tiled_copy.retile(rO)
        tOgO = tiled_copy.get_slice(tidx).partition_D(gO)
        cute.autovec_copy(tOrO, tOgO)

        mLSE_cur = mLSE[None, partial_head_idx, batch_idx]
        gLSE = cute.local_tile(mLSE_cur, (self.tile_m,), (m_block,))
        gLSE_expanded_layout = cute.append(
            gLSE.layout, cute.make_layout((self.tile_hdimv,), stride=(0,))
        )
        gLSE_expanded = cute.make_tensor(
            gLSE.iterator, gLSE_expanded_layout
        )
        thr_mma = tiled_mma.get_slice(tidx)
        taccOgLSE = layout_utils.reshape_acc_to_mn(
            thr_mma.partition_C(gLSE_expanded)
        )
        cO = cute.make_identity_tensor((self.tile_m, self.tile_hdimv))
        taccOcO = layout_utils.reshape_acc_to_mn(thr_mma.partition_C(cO))
        t0accOcO = layout_utils.reshape_acc_to_mn(
            thr_mma.get_slice(0).partition_C(cO)
        )
        if taccOcO[0][1] == 0:
            for m in cutlass.range_constexpr(cute.size(taccOgLSE.shape[1])):
                if (
                    t0accOcO[m, 0][0]
                    < seqlen.seqlen_q
                    - m_block * self.tile_m
                    - taccOcO[0][0]
                ):
                    taccOgLSE[m, 0] = lse[m]

    @cute.jit
    def mma_one_warpgroup_sol_attn_route_tma(
        self,
        tiled_mma_qk: cute.TiledMma,
        tiled_mma_pv: cute.TiledMma,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mKC: cute.Tensor,
        mVC: cute.Tensor,
        mO: cute.Tensor,
        mLSE: Optional[cute.Tensor],
        sQ: cute.Tensor,
        sK: cute.Tensor,
        sV: cute.Tensor,
        sVt: cute.Tensor,
        sP: Optional[cute.Tensor],
        sO: cute.Tensor,
        tma_atom_Q: Optional[cute.CopyAtom],
        tma_atom_K: Optional[cute.CopyAtom],
        tma_atom_V: Optional[cute.CopyAtom],
        tma_atom_KC: Optional[cute.CopyAtom],
        tma_atom_VC: Optional[cute.CopyAtom],
        gmem_tiled_copy_O: cute.TiledCopy,
        tma_atom_O: Optional[cute.CopyAtom],
        pipeline_q: pipeline.PipelineAsync,
        pipeline_k: pipeline.PipelineAsync,
        pipeline_v: pipeline.PipelineAsync,
        SeqlenInfoCls: Callable,
        AttentionMaskCls: Callable,
        TileSchedulerCls: cutlass.Constexpr[Callable],
        mGlobalThresh: cute.Tensor,
        route_mask: cute.Tensor,
        route_sums: cute.Tensor,
        softmax_scale_log2: Float32,
        softmax_scale: Float32,
        sink_range: Int32,
        block_info: BlockInfo,
    ):
        """Run the fused route, approximate, and exact attention mainloop."""

        tidx, _, _ = cute.arch.thread_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        if const_expr(not (self.use_tma_Q and self.use_tma_KV)):
            if tidx == Int32(0) and warp_idx == Int32(0):
                cute.printf("SOL_ATTN one-warpgroup path requires TMA Q/KV\n")
        else:
            q_producer_phase = Int32(1)
            q_consumer_phase = Int32(0)
            kv_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_stages
            )
            kv_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_stages
            )
            tile_scheduler = TileSchedulerCls()
            work_tile = tile_scheduler.initial_work_tile_info()

            if work_tile.is_valid_tile:
                m_block, head_idx, batch_idx, split_idx = work_tile.tile_idx
                partial_head_idx = (
                    head_idx
                    + split_idx * mQ.shape[2]
                    if const_expr(self.sol_attn_num_splits > 1)
                    else head_idx
                )
                seqlen = SeqlenInfoCls(batch_idx)
                head_idx_kv = (
                    head_idx // self.qhead_per_kvhead
                    if const_expr(not self.pack_gqa)
                    else head_idx
                )
                mQ_cur = seqlen.offset_batch_Q(mQ, batch_idx, dim=3)[None, None, head_idx]
                mK_cur = seqlen.offset_batch_K(mK, batch_idx, dim=3)[
                    None, None, head_idx_kv
                ]
                mV_cur = seqlen.offset_batch_K(mV, batch_idx, dim=3)[
                    None, None, head_idx_kv
                ]
                mKC_cur = mKC[None, None, head_idx_kv, batch_idx]
                mVC_cur = mVC[None, None, head_idx_kv, batch_idx]

                gQ = cute.local_tile(mQ_cur, (self.tile_m, self.tile_hdim), (m_block, 0))
                gK = cute.local_tile(mK_cur, (self.tile_n, self.tile_hdim), (None, 0))
                gV = cute.local_tile(mV_cur, (self.tile_n, self.tile_hdimv), (None, 0))
                gKC = cute.local_tile(mKC_cur, (self.tile_n, self.tile_hdim), (None, 0))
                gVC = cute.local_tile(mVC_cur, (self.tile_n, self.tile_hdimv), (None, 0))

                load_Q, _, _ = copy_utils.tma_get_copy_fn(
                    tma_atom_Q, 0, cute.make_layout(1), gQ, sQ, single_stage=True
                )
                tma_load_K_fn, _, _ = copy_utils.tma_get_copy_fn(
                    tma_atom_K, 0, cute.make_layout(1), gK, sK
                )
                tma_load_K_fn = copy_utils.tma_producer_copy_fn(tma_load_K_fn, pipeline_k)
                tma_load_V_fn, _, _ = copy_utils.tma_get_copy_fn(
                    tma_atom_V, 0, cute.make_layout(1), gV, sV
                )
                tma_load_V_fn = copy_utils.tma_producer_copy_fn(tma_load_V_fn, pipeline_v)
                tma_load_KC_fn, _, _ = copy_utils.tma_get_copy_fn(
                    tma_atom_KC, 0, cute.make_layout(1), gKC, sK
                )
                tma_load_KC_fn = copy_utils.tma_producer_copy_fn(
                    tma_load_KC_fn, pipeline_k
                )
                tma_load_VC_fn, _, _ = copy_utils.tma_get_copy_fn(
                    tma_atom_VC, 0, cute.make_layout(1), gVC, sV
                )
                tma_load_VC_fn = copy_utils.tma_producer_copy_fn(
                    tma_load_VC_fn, pipeline_v
                )

                if warp_idx == Int32(0):
                    pipeline_q.producer_acquire_w_index_phase(0, q_producer_phase)
                    load_Q(tma_bar_ptr=pipeline_q.sync_object_full.get_barrier(0))

                pipeline_q.consumer_wait_w_index_phase(0, q_consumer_phase)
                warp_group_thread_layout = cute.make_layout(
                    1, stride=self.num_threads_per_warp_group
                )
                thr_mma_qk = tiled_mma_qk.get_slice(tidx)
                wg_mma_qk = tiled_mma_qk.get_slice(warp_group_thread_layout(Int32(0)))
                wg_mma_pv = tiled_mma_pv.get_slice(warp_group_thread_layout(Int32(0)))
                _, tSrQ, tSrK = sm90_utils.partition_fragment_ABC(
                    wg_mma_qk, (self.tile_m, self.tile_n, self.tile_hdim), sQ, sK
                )
                mma_qk_fn = partial(
                    self.sol_attn_qk_gemm_zero_init,
                    tiled_mma_qk,
                    (self.tile_m, self.tile_n),
                    tSrQ,
                    tSrK,
                )
                acc_O, tOrP, tOrVt = sm90_utils.partition_fragment_ABC(
                    wg_mma_pv, (self.tile_m, self.tile_hdimv, self.tile_n), sP, sVt
                )
                mma_pv_fn = partial(sm90_utils.gemm_w_idx, tiled_mma_pv, acc_O, tOrP, tOrVt)
                smem_copy_atom_P = utils.get_smem_store_atom(
                    self.arch.major * 10 + self.arch.minor, self.dtype
                )
                smem_thr_copy_P = cute.make_tiled_copy_C(
                    smem_copy_atom_P, tiled_mma_qk
                ).get_slice(tidx)
                tPsP = smem_thr_copy_P.partition_D(sP) if const_expr(sP is not None) else None
                smem_copy_params = SimpleNamespace(
                    smem_thr_copy_P=smem_thr_copy_P,
                    tPsP=tPsP,
                )
                acc_O.fill(0.0)
                cS_route = cute.make_identity_tensor((self.tile_m, self.tile_n))
                tScS_route_mn = layout_utils.reshape_acc_to_mn(
                    thr_mma_qk.partition_C(cS_route)
                )
                mask = AttentionMaskCls(seqlen)
                mask_fn = partial(
                    mask.apply_mask,
                    batch_idx=batch_idx,
                    head_idx=head_idx,
                    m_block=m_block,
                    thr_mma=thr_mma_qk,
                    mask_causal=self.is_causal,
                    mask_local=self.is_local,
                    aux_tensors=None,
                    fastdiv_mods=None,
                )
                score_mod_fn = None
                if const_expr(self.score_mod is not None):
                    score_mod_fn = partial(
                        self.apply_score_mod,
                        thr_mma_qk,
                        batch_idx,
                        head_idx,
                        m_block,
                        softmax_scale=softmax_scale,
                        aux_tensors=None,
                        fastdiv_mods=None,
                    )
                softmax = Softmax.create(
                    softmax_scale_log2,
                    num_rows=acc_O.shape[0][0] * acc_O.shape[1],
                    softmax_scale=softmax_scale,
                )
                if const_expr(self.sol_attn_neutral_softmax_state):
                    softmax.row_max.fill(-Float32.inf)
                    softmax.row_sum.fill(0.0)
                exact_mma_one_n_block = partial(
                    self.mma_one_n_block,
                    mma_qk_fn=mma_qk_fn,
                    pipeline_k=pipeline_k,
                    pipeline_v=pipeline_v,
                    acc_O=acc_O,
                    tOrP=tOrP,
                    smem_copy_params=smem_copy_params,
                    softmax=softmax,
                    score_mod_fn=score_mod_fn,
                    score_scale_fn=None,
                    check_inf=not self.sol_attn_assume_nonempty_rows,
                )
                n_block_min, n_block_max = block_info.get_n_block_min_max(seqlen, m_block)
                route_block_count = n_block_max - n_block_min
                if const_expr(self.sol_attn_static_num_full_route_groups >= 0):
                    num_full_route_groups = Int32(self.sol_attn_static_num_full_route_groups)
                    if const_expr(self.sol_attn_static_tail_valid_count > 0):
                        tail_valid_count = Int32(self.sol_attn_static_tail_valid_count)
                    else:
                        tail_valid_count = Int32(0)
                elif const_expr(self.sol_attn_assume_full_route_groups):
                    num_full_route_groups = cute.ceil_div(
                        route_block_count, self.sol_attn_group_size
                    )
                    tail_valid_count = Int32(0)
                else:
                    num_full_route_groups = route_block_count // Int32(self.sol_attn_group_size)
                    tail_valid_count = (
                        route_block_count
                        - num_full_route_groups * Int32(self.sol_attn_group_size)
                    )
                num_route_groups = num_full_route_groups
                if tail_valid_count > Int32(0):
                    num_route_groups += Int32(1)
                if const_expr(self.sol_attn_num_splits == 1):
                    split_group_begin = Int32(0)
                    split_num_route_groups = num_route_groups
                else:
                    groups_per_split = (
                        num_route_groups + self.sol_attn_num_splits - 1
                    ) // self.sol_attn_num_splits
                    split_group_begin = split_idx * groups_per_split
                    split_group_end = cutlass.min(
                        split_group_begin + groups_per_split, num_route_groups
                    )
                    split_num_route_groups = cutlass.max(
                        split_group_end - split_group_begin, Int32(0)
                    )
                O_should_accumulate = self.sol_attn_neutral_softmax_state
                for local_group_iter in cutlass.range(
                    split_num_route_groups, unroll=1
                ):
                    group_iter = split_group_begin + local_group_iter
                    group_start = n_block_min + group_iter * Int32(self.sol_attn_group_size)
                    route_valid_count = Int32(self.sol_attn_group_size)
                    if const_expr(not self.sol_attn_assume_full_route_groups):
                        if group_iter == num_full_route_groups and tail_valid_count > Int32(0):
                            route_valid_count = tail_valid_count
                    route_col_offset = group_start - (
                        group_start // Int32(self.tile_n)
                    ) * Int32(self.tile_n)
                    route_n_block = group_start - route_col_offset
                    route_tile = route_n_block // Int32(self.tile_n)
                    has_next_route_group = (
                        local_group_iter + Int32(1) < split_num_route_groups
                    )
                    next_route_tile = Int32(-1)
                    if has_next_route_group:
                        next_group_start = group_start + Int32(self.sol_attn_group_size)
                        next_route_tile = next_group_start // Int32(self.tile_n)

                    if warp_idx == Int32(0):
                        if local_group_iter == Int32(0):
                            pipeline_k.producer_acquire(kv_producer_state)
                            tma_load_KC_fn(
                                src_idx=route_tile,
                                producer_state=kv_producer_state,
                            )
                        else:
                            previous_group_had_exact = (
                                (route_mask[0] != Int32(0))
                                or (route_mask[1] != Int32(0))
                                or (route_mask[2] != Int32(0))
                                or (route_mask[3] != Int32(0))
                            )
                            if not previous_group_had_exact:
                                pipeline_k.producer_acquire(kv_producer_state)
                                tma_load_KC_fn(
                                    src_idx=route_tile,
                                    producer_state=kv_producer_state,
                                )
                        pipeline_v.producer_acquire(kv_producer_state)
                        tma_load_VC_fn(
                            src_idx=route_tile,
                            producer_state=kv_producer_state,
                        )
                        kv_producer_state.advance()

                    pipeline_k.consumer_wait(
                        kv_consumer_state,
                        pipeline_k.consumer_try_wait(kv_consumer_state),
                    )
                    acc_S = mma_qk_fn(B_idx=kv_consumer_state.index, wg_wait=-1)
                    warpgroup.wait_group(0)
                    pipeline_k.consumer_release(kv_consumer_state)
                    mask0, mask1, mask2, mask3 = self.sol_attn_build_route_mask_from_acc(
                        acc_S,
                        route_sums,
                        tScS_route_mn,
                        m_block,
                        group_start,
                        route_valid_count,
                        route_col_offset,
                        seqlen,
                        batch_idx,
                        head_idx,
                        mGlobalThresh,
                        softmax_scale_log2,
                        sink_range,
                        False,
                        route_mask_words_override=2,
                    )
                    exact_mask0 = mask0
                    exact_mask1 = mask1
                    exact_mask2 = mask2
                    exact_mask3 = mask3
                    first_exact_n_block = group_start
                    first_exact_exists = False
                    if tidx == Int32(0):
                        route_mask[0] = mask0
                        route_mask[1] = mask1
                        route_mask[2] = mask2
                        route_mask[3] = mask3

                    cute.arch.barrier(
                        barrier_id=SOL_ATTN_ROUTE_MASK_BARRIER_ID,
                        number_of_threads=self.num_mma_threads,
                    )
                    mask0 = route_mask[0]
                    mask1 = route_mask[1]
                    mask2 = route_mask[2]
                    mask3 = route_mask[3]

                    exact_mask0 = mask0
                    exact_mask1 = mask1
                    exact_mask2 = mask2
                    exact_mask3 = mask3

                    first_exact_exists = (
                        (mask0 != Int32(0))
                        or (mask1 != Int32(0))
                        or (mask2 != Int32(0))
                        or (mask3 != Int32(0))
                    )
                    if mask0 != Int32(0):
                        first_lowbit = mask0 & (Int32(0) - mask0)
                        first_exact_n_block += sol_attn_selector.sol_attn_bfind_b32(
                            first_lowbit
                        )
                        exact_mask0 = mask0 & (mask0 - Int32(1))
                    elif mask1 != Int32(0):
                        first_lowbit = mask1 & (Int32(0) - mask1)
                        first_exact_n_block += Int32(32) + (
                            sol_attn_selector.sol_attn_bfind_b32(first_lowbit)
                        )
                        exact_mask1 = mask1 & (mask1 - Int32(1))
                    elif mask2 != Int32(0):
                        first_lowbit = mask2 & (Int32(0) - mask2)
                        first_exact_n_block += Int32(64) + (
                            sol_attn_selector.sol_attn_bfind_b32(first_lowbit)
                        )
                        exact_mask2 = mask2 & (mask2 - Int32(1))
                    elif mask3 != Int32(0):
                        first_lowbit = mask3 & (Int32(0) - mask3)
                        first_exact_n_block += Int32(96) + (
                            sol_attn_selector.sol_attn_bfind_b32(first_lowbit)
                        )
                        exact_mask3 = mask3 & (mask3 - Int32(1))
                    if first_exact_exists and warp_idx == Int32(0):
                        pipeline_k.producer_acquire(kv_producer_state)
                        tma_load_K_fn(
                            src_idx=first_exact_n_block,
                            producer_state=kv_producer_state,
                        )

                    if const_expr(self.score_mod is not None):
                        score_mod_fn(acc_S, n_block=route_n_block, seqlen=seqlen)
                    mask_fn(
                        acc_S,
                        n_block=route_n_block,
                        mask_mod=self.mask_mod,
                        mask_seqlen=not self.sol_attn_full_route_mask_seqlen_false,
                    )
                    if const_expr(self.sol_attn_assume_full_route_groups):
                        route_has_approx = (mask0 != Int32(-1)) or (mask1 != Int32(-1))
                    else:
                        valid0 = route_valid_count
                        if valid0 > Int32(32):
                            valid0 = Int32(32)
                        valid_bits0 = Int32(0)
                        if valid0 > Int32(0):
                            valid_bits0 = Int32(-1)
                            if valid0 < Int32(32):
                                valid_bits0 = (Int32(1) << valid0) - Int32(1)
                        valid1 = route_valid_count - Int32(32)
                        if valid1 < Int32(0):
                            valid1 = Int32(0)
                        if valid1 > Int32(32):
                            valid1 = Int32(32)
                        valid_bits1 = Int32(0)
                        if valid1 > Int32(0):
                            valid_bits1 = Int32(-1)
                            if valid1 < Int32(32):
                                valid_bits1 = (Int32(1) << valid1) - Int32(1)
                        route_has_approx = (
                            ((mask0 & valid_bits0) != valid_bits0)
                            or ((mask1 & valid_bits1) != valid_bits1)
                        )
                    self.sol_attn_mask_route_approx_columns(
                        acc_S,
                        route_sums,
                        tScS_route_mn,
                        route_valid_count,
                        route_col_offset,
                        mask0,
                        mask1,
                        mask2,
                        mask3,
                        False,
                        route_mask_words_override=self.sol_attn_group_words,
                    )
                    pipeline_v.consumer_wait(
                        kv_consumer_state,
                        pipeline_v.consumer_try_wait(kv_consumer_state),
                    )
                    if route_has_approx:
                        row_sum_prev = None
                        if const_expr(
                            self.sol_attn_fast_route_lens
                            and not self.sol_attn_full_block_row_sum_prescale
                        ):
                            row_sum_prev = cute.make_fragment_like(softmax.row_sum, Float32)
                            row_sum_prev.store(softmax.row_sum.load())
                        if O_should_accumulate:
                            if const_expr(self.sol_attn_full_block_row_sum_prescale):
                                for r in cutlass.range(
                                    cute.size(softmax.row_sum), unroll_full=True
                                ):
                                    softmax.row_sum[r] *= Float32(1.0 / self.tile_n)
                            row_scale = softmax.online_softmax(
                                acc_S,
                                is_first=False,
                                check_inf=not self.sol_attn_assume_nonempty_rows,
                            )
                            softmax.rescale_O(acc_O, row_scale)
                            if const_expr(self.sol_attn_full_block_row_sum_prescale):
                                for r in cutlass.range(
                                    cute.size(softmax.row_sum), unroll_full=True
                                ):
                                    softmax.row_sum[r] *= Float32(self.tile_n)
                            elif const_expr(self.sol_attn_fast_route_lens):
                                self.sol_attn_apply_route_current_lens_to_row_sum_fast(
                                    acc_S,
                                    tScS_route_mn,
                                    row_sum_prev,
                                    row_scale,
                                    group_start,
                                    route_valid_count,
                                    route_col_offset,
                                    seqlen,
                                    softmax,
                                    False,
                                )
                            else:
                                self.sol_attn_apply_route_current_lens_to_row_sum(
                                    acc_S,
                                    tScS_route_mn,
                                    group_start,
                                    route_valid_count,
                                    route_col_offset,
                                    seqlen,
                                    softmax,
                                )
                        else:
                            row_scale = softmax.online_softmax(
                                acc_S,
                                is_first=True,
                                check_inf=not self.sol_attn_assume_nonempty_rows,
                            )
                            if const_expr(self.sol_attn_full_block_row_sum_prescale):
                                for r in cutlass.range(
                                    cute.size(softmax.row_sum), unroll_full=True
                                ):
                                    softmax.row_sum[r] *= Float32(self.tile_n)
                            elif const_expr(self.sol_attn_fast_route_lens):
                                self.sol_attn_apply_route_current_lens_to_row_sum_fast(
                                    acc_S,
                                    tScS_route_mn,
                                    row_sum_prev,
                                    row_scale,
                                    group_start,
                                    route_valid_count,
                                    route_col_offset,
                                    seqlen,
                                    softmax,
                                    True,
                                )
                            else:
                                self.sol_attn_apply_route_current_lens_to_row_sum(
                                    acc_S,
                                    tScS_route_mn,
                                    group_start,
                                    route_valid_count,
                                    route_col_offset,
                                    seqlen,
                                    softmax,
                                )
                        tOrP_acc = layout_utils.reshape_acc_to_frgA(acc_S)
                        tOrP_cur = (
                            tOrP
                            if const_expr(self.mma_pv_is_rs)
                            else cute.make_rmem_tensor_like(tOrP_acc, self.dtype)
                        )
                        utils.cvt_f16(tOrP_acc, tOrP_cur)
                        if const_expr(not self.mma_pv_is_rs):
                            tPrP = smem_copy_params.smem_thr_copy_P.retile(tOrP_cur)
                            cute.copy(
                                smem_copy_params.smem_thr_copy_P,
                                tPrP,
                                smem_copy_params.tPsP,
                            )
                            cute.arch.fence_view_async_shared()
                            cute.arch.sync_warp()
                        if O_should_accumulate:
                            sm90_utils.gemm_w_idx(
                                tiled_mma_pv,
                                acc_O,
                                tOrP_cur,
                                tOrVt,
                                zero_init=False,
                                B_idx=kv_consumer_state.index,
                                wg_wait=-1,
                            )
                        else:
                            sm90_utils.gemm_w_idx(
                                tiled_mma_pv,
                                acc_O,
                                tOrP_cur,
                                tOrVt,
                                zero_init=True,
                                B_idx=kv_consumer_state.index,
                                wg_wait=-1,
                            )
                        warpgroup.wait_group(0)
                        O_should_accumulate = True
                    pipeline_v.consumer_release(kv_consumer_state)
                    kv_consumer_state.advance()

                    last_n_block = Int32(-1)
                    if const_expr(
                        (not self.sol_attn_assume_full_k_exact_blocks)
                        or self.sol_attn_exact_mask_seqlen_last_only
                    ):
                        last_n_block = (
                            (seqlen.seqlen_k + Int32(self.tile_n - 1)) // Int32(self.tile_n)
                        ) - Int32(1)
                    if O_should_accumulate:
                        (
                            kv_producer_state,
                            kv_consumer_state,
                            O_should_accumulate,
                            _,
                        ) = exact_stream.consume_exact_blocks(
                            exact_mask0,
                            exact_mask1,
                            exact_mask2,
                            exact_mask3,
                            group_start,
                            seqlen,
                            kv_producer_state,
                            kv_consumer_state,
                            tma_load_K_fn,
                            tma_load_V_fn,
                            pipeline_k,
                            pipeline_v,
                            warp_idx == Int32(0),
                            mma_pv_fn,
                            exact_mma_one_n_block,
                            mask_fn,
                            score_mod_fn,
                            O_should_accumulate,
                            self.warp_scheduler_barrier_sync,
                            self.warp_scheduler_barrier_arrive,
                            not self.sol_attn_assume_full_k_exact_blocks,
                            False,
                            self.sol_attn_group_words,
                            last_n_block,
                            self.sol_attn_exact_mask_seqlen_last_only,
                            first_exact_n_block,
                            first_exact_exists,
                            next_route_tile,
                            tma_load_KC_fn,
                        )
                    else:
                        (
                            kv_producer_state,
                            kv_consumer_state,
                            O_should_accumulate,
                            _,
                        ) = exact_stream.consume_exact_blocks(
                            exact_mask0,
                            exact_mask1,
                            exact_mask2,
                            exact_mask3,
                            group_start,
                            seqlen,
                            kv_producer_state,
                            kv_consumer_state,
                            tma_load_K_fn,
                            tma_load_V_fn,
                            pipeline_k,
                            pipeline_v,
                            warp_idx == Int32(0),
                            mma_pv_fn,
                            exact_mma_one_n_block,
                            mask_fn,
                            score_mod_fn,
                            O_should_accumulate,
                            self.warp_scheduler_barrier_sync,
                            self.warp_scheduler_barrier_arrive,
                            not self.sol_attn_assume_full_k_exact_blocks,
                            True,
                            self.sol_attn_group_words,
                            last_n_block,
                            self.sol_attn_exact_mask_seqlen_last_only,
                            first_exact_n_block,
                            first_exact_exists,
                            next_route_tile,
                            tma_load_KC_fn,
                        )

                pipeline_q.consumer_release_w_index(0)
                final_scale = softmax.finalize(sink_val=None)
                softmax.rescale_O(acc_O, final_scale)
                if const_expr(self.use_tma_O):
                    self.epilogue_one_warpgroup_tma_o(
                        acc_O,
                        softmax.row_sum,
                        mO,
                        mLSE,
                        sO,
                        seqlen,
                        tma_atom_O,
                        tiled_mma_pv,
                        tidx,
                        m_block,
                        partial_head_idx,
                        batch_idx,
                    )
                else:
                    self.epilogue_one_warpgroup_split_partial(
                        acc_O,
                        softmax.row_sum,
                        mO,
                        mLSE,
                        seqlen,
                        tiled_mma_pv,
                        tidx,
                        m_block,
                        partial_head_idx,
                        batch_idx,
                    )

    @cute.jit
    def mma_one_n_block(
        self,
        smem_pipe_read: pipeline.PipelineState | pipeline_custom.PipelineStateSimple,
        n_block: Int32,
        mma_qk_fn: Callable,
        mma_pv_fn: Callable,
        pipeline_k: pipeline.PipelineAsync,
        pipeline_v: pipeline.PipelineAsync,
        acc_O: cute.Tensor,
        tOrP: cute.Tensor,
        smem_copy_params: SimpleNamespace,
        softmax: Softmax,
        seqlen: SeqlenInfoQK,
        scores_scale: Optional[cute.Tensor] = None,
        score_mod_fn: Optional[Callable] = None,
        score_scale_fn: Optional[Callable] = None,
        mask_fn: Optional[Callable] = None,
        last_block_mask_fn: Optional[Callable] = None,
        last_n_block: Int32 = Int32(-1),
        is_first_n_block: cutlass.Constexpr = False,
        check_inf: cutlass.Constexpr = True,
        prefetch_next: cutlass.Constexpr = False,
        next_n_block: Int32 = Int32(-1),
        kv_producer_state=None,
        load_K: Optional[Callable] = None,
        issue_load=False,
    ):
        pipeline_k.consumer_wait(smem_pipe_read, pipeline_k.consumer_try_wait(smem_pipe_read))
        acc_S = mma_qk_fn(B_idx=smem_pipe_read.index, wg_wait=-1)
        self.warp_scheduler_barrier_arrive()
        warpgroup.wait_group(0)
        pipeline_k.consumer_release(smem_pipe_read)

        # Reuse the released K stage while the current softmax and P@V run.
        if const_expr(prefetch_next):
            if issue_load and next_n_block >= Int32(0):
                pipeline_k.producer_acquire(kv_producer_state)
                load_K(src_idx=next_n_block, producer_state=kv_producer_state)

        if const_expr(score_scale_fn is not None):
            score_scale_fn(acc_S, n_block=n_block)
        if const_expr(score_mod_fn is not None):
            score_mod_fn(acc_S, n_block=n_block, seqlen=seqlen)
        if const_expr(mask_fn is not None):
            mask_fn(acc_S=acc_S, n_block=n_block)
        if const_expr(last_block_mask_fn is not None):
            if n_block == last_n_block:
                last_block_mask_fn(acc_S=acc_S, n_block=n_block)

        row_scale = softmax.online_softmax(acc_S, is_first=is_first_n_block, check_inf=check_inf)
        tOrP_acc = layout_utils.reshape_acc_to_frgA(acc_S)
        tOrP_cur = (
            tOrP
            if const_expr(self.mma_pv_is_rs)
            else cute.make_rmem_tensor_like(tOrP_acc, self.dtype)
        )
        utils.cvt_f16(tOrP_acc, tOrP_cur)
        if const_expr(not self.mma_pv_is_rs):
            tPrP = smem_copy_params.smem_thr_copy_P.retile(tOrP_cur)
            cute.copy(smem_copy_params.smem_thr_copy_P, tPrP, smem_copy_params.tPsP)
        softmax.rescale_O(acc_O, row_scale)
        if const_expr(not self.mma_pv_is_rs):
            cute.arch.fence_view_async_shared()
            cute.arch.sync_warp()

        pipeline_v.consumer_wait(smem_pipe_read, pipeline_v.consumer_try_wait(smem_pipe_read))
        self.warp_scheduler_barrier_sync()
        mma_pv_fn(B_idx=smem_pipe_read.index, wg_wait=0)
        pipeline_v.consumer_release(smem_pipe_read)
        smem_pipe_read.advance()
        return smem_pipe_read

    @cute.jit
    def mma_init(self):
        warp_group_idx = utils.canonical_warp_group_idx(sync=False)
        if const_expr(self.use_scheduler_barrier):
            if warp_group_idx == 1:
                cute.arch.barrier_arrive(
                    barrier_id=int(NamedBarrierFwd.WarpSchedulerWG1),
                    number_of_threads=2 * self.num_threads_per_warp_group,
                )

    @cute.jit
    def apply_score_mod(
        self,
        thr_mma_qk,
        batch_idx,
        head_idx,
        m_block,
        acc_S,
        n_block,
        softmax_scale,
        seqlen,
        aux_tensors: Optional[list] = None,
        fastdiv_mods=None,
    ):
        # Prepare index tensor
        cS = cute.make_identity_tensor((self.tile_m, self.tile_n))
        cS = cute.domain_offset((m_block * self.tile_m, n_block * self.tile_n), cS)
        tScS = thr_mma_qk.partition_C(cS)

        apply_score_mod_inner(
            acc_S,
            tScS,
            self.score_mod,
            batch_idx,
            head_idx,
            softmax_scale,
            self.vec_size,
            self.qk_acc_dtype,
            aux_tensors,
            fastdiv_mods,
            seqlen_info=seqlen,
            constant_q_idx=None,
            qhead_per_kvhead=self.qhead_per_kvhead if const_expr(self.pack_gqa) else 1,
        )

    def warp_scheduler_barrier_sync(self):
        if const_expr(self.use_scheduler_barrier):
            cute.arch.barrier(
                barrier_id=int(NamedBarrierFwd.WarpSchedulerWG1)
                - 1
                + utils.canonical_warp_group_idx(sync=False),
                number_of_threads=2 * self.num_threads_per_warp_group,
            )

    def warp_scheduler_barrier_arrive(self):
        if const_expr(self.use_scheduler_barrier):
            assert self.num_wg_mma in [2, 3]
            cur_wg = utils.canonical_warp_group_idx(sync=False) - 1
            if const_expr(self.num_wg_mma == 2):
                next_wg = 1 - cur_wg
            else:
                t = cur_wg + 1
                next_wg = t % self.num_wg_mma
            cute.arch.barrier_arrive(
                barrier_id=int(NamedBarrierFwd.WarpSchedulerWG1) + next_wg,
                number_of_threads=2 * self.num_threads_per_warp_group,
            )
