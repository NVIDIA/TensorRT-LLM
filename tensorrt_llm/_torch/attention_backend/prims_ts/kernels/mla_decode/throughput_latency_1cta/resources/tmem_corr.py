# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2026 by FlashInfer team.
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

"""Correction, normalization, and output-store resource."""

from dataclasses import dataclass
from typing import Optional

from cutlass.experimental import primitives as prims

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int8, Int32, Int64
from cutlass.experimental import primitives as cprims
from cutlass.experimental.task_scheduling.enums import WorkAttr
from cutlass.experimental.task_scheduling.memory import SmemAllocation
from cutlass.experimental.task_scheduling.resources import (
    StageInfo,
    producer_work,
)

from ...helpers.constants import (
    SMEM_WORD_BYTE_SHIFT,
    SMEM_WORD_BYTES,
    TCGEN05_16X256B_REGS_PER_LOAD,
    TCGEN05_16X256B_SHAPE,
    WARPGROUP_THREADS,
)


from ...helpers.layout import (
    _TASK_CACHE_LANE_IDX,
    _TASK_CACHE_TMEM_BASE_OFFSET,
    _TASK_CACHE_WARP_GRP_THREAD_IDX,
    _TASK_CACHE_WARP_IDX,
    decode_gen_task_cache,
    head_dim_cta_offset_v,
    local_q_head_idx_for_scale,
    num_fp8_output_regs,
    num_o_reg_pairs,
    num_o_repeats,
    num_o_stsm_row_blocks,
    num_o_tmem_loads_per_stage,
    num_softmax_scale_groups,
    o_stage_stsm_and_copy_offsets,
    o_stage_tmem_col_offset,
    smem_array,
)
from ...helpers.math import (
    ceil_div,
    fadd2,
    ffma2,
    fmul2,
    output_dtype,
    pack_float2_to_bf16,
    partial_output_dtype,
)
from ...helpers.ops import (
    fp8_quant_scale_rcp,
    pack_float4_to_fp8_e4m3,
    store_transposed_smem8b_x2,
    store_transposed_smem8b_x4,
    tcgen05_ld_16x32bx2_f32,
    tcgen05_panel_addr,
    tcgen05_second_panel_addr,
    tcgen05_st_16x32bx2_f32,
    vector_from_scalars,
)
from ...helpers.query import groups_tokens_heads_q_row_state, public_query_flat_row
from ...helpers.tile import (
    batch_idx_for_stage_cfg,
    cta_idx_head_dim_v_for_stage,
    cta_idx_kv_for_stage,
    cta_idx_q_for_stage,
    head_idx_for_stage,
)

from .common import (
    MlaResource,
)

# =====================================================================
# TmemCorrResource — Correction and output store to GMEM
# =====================================================================


@dataclass(kw_only=True)
class TmemCorrResource(MlaResource):
    """Fused throughput-latency 1CTA correction and output-store helper.

    Despite the historical name, this resource does not own TMEM. O is handed
    from MmaTask to CorrectionTask through TmemOResource; this helper owns SMEM
    scratch/staging and fuses correction, StoreO, and StoreLSE. This should be
    cleaned up into clearer correction/store resource boundaries later.
    """

    inst_id: cutlass.Constexpr[int] = 0
    scale_softmax_log2: Float32 = None
    output_scale: Float32 = None
    o_tensor: object = None
    lse_tensor: object = None
    acc_o_tensor: object = None
    acc_lse_tensor: object = None
    cache_seqs: object = None
    head_idx: object = None
    batch_idx: object = None
    cta_idx_q: object = None
    cta_idx_kv: object = None
    cta_idx_head_dim_v: object = None
    store_barrier_id: cutlass.Constexpr[int] = 6
    sum_barrier_id: cutlass.Constexpr[int] = 7
    _sum_alloc: cutlass.Constexpr[Optional[SmemAllocation]] = None
    _cluster_reduction_alloc: cutlass.Constexpr[Optional[SmemAllocation]] = None
    _cluster_reduction_barrier_alloc: cutlass.Constexpr[Optional[SmemAllocation]] = None
    _smem_o: object = None
    _smem_o_i32: object = None
    _sum_scratch: object = None
    _cluster_reduction_smem: object = None
    _cluster_reduction_barrier: object = None

    def get_smem_requirements(self):
        """Return correction/output SMEM scratch and optional cluster buffers."""
        if self.inst_id != 1:
            return []
        if self._alloc is None:
            self._alloc = SmemAllocation(
                name=f"{self.name}_oStage",
                size_bytes=self.cfg.o_smem_tile_bytes,
                alignment=self.cfg.stensor_align,
            )
        if self._sum_alloc is None:
            self._sum_alloc = SmemAllocation(
                name=f"{self.name}_sumScratch",
                size_bytes=self.cfg.corr_scratch_bytes,
                alignment=16,
            )
        allocs = [self._alloc, self._sum_alloc]
        if self.cfg.cluster_reduction_smem_bytes and self.acc_o_tensor is None:
            if self._cluster_reduction_alloc is None:
                self._cluster_reduction_alloc = SmemAllocation(
                    name=f"{self.name}_clusterReduction",
                    size_bytes=self.cfg.cluster_reduction_smem_bytes,
                    alignment=64,
                )
            if self._cluster_reduction_barrier_alloc is None:
                self._cluster_reduction_barrier_alloc = SmemAllocation(
                    name=f"{self.name}_clusterReductionBarrier",
                    size_bytes=8,
                    alignment=8,
                )
            allocs.append(self._cluster_reduction_alloc)
            allocs.append(self._cluster_reduction_barrier_alloc)
        return allocs

    @cute.jit
    def _init_store_state_from_context(self, context) -> None:
        """Create correction/output SMEM views and initialize cluster state."""
        if cutlass.const_expr(self.inst_id != 1):
            return
        self._smem_o = smem_array(
            context,
            self._alloc,
            output_dtype(self.cfg),
            self.cfg.o_smem_tile_bytes // self.cfg.o_dtype_bytes,
        )
        self._smem_o_i32 = smem_array(
            context,
            self._alloc,
            Int32,
            self.cfg.o_smem_tile_bytes // 4,
        )
        self._sum_scratch = smem_array(
            context,
            self._sum_alloc,
            Float32,
            self.cfg.corr_scratch_bytes // self.cfg.acc_dtype_bytes,
        )
        if cutlass.const_expr(self._cluster_reduction_alloc is not None):
            self._cluster_reduction_smem = smem_array(
                context,
                self._cluster_reduction_alloc,
                Int8,
                self.cfg.cluster_reduction_smem_bytes,
            )
            self._cluster_reduction_barrier = smem_array(
                context,
                self._cluster_reduction_barrier_alloc,
                Int64,
                1,
            )
            self._init_cluster_reduction_barrier()

    @cute.jit
    def initialize_runtime_state_internal(
        self,
        context=None,
        captured_schedule: cutlass.Constexpr[bool] = False,
    ) -> None:
        """Initialize correction/store SMEM views outside captured task bodies."""
        super().initialize_runtime_state_internal(context, captured_schedule)
        # cluster state is initialized explicitly by the kernel entry before the
        # cluster rendezvous.  This guarantees that every remote transaction
        # barrier is live before any peer can publish a partial.  Non-cluster
        # schedules retain the ordinary task-owned initialization path.
        if cutlass.const_expr(self.cfg.use_cluster_reduction != 1):
            self._init_store_state_from_context(context)

    @cute.jit
    def create_cluster_function_variables(self, context) -> None:
        """Bind cluster correction SMEM and initialize its transaction barrier."""

        if cutlass.const_expr(self.cfg.use_cluster_reduction == 1):
            self._init_store_state_from_context(context)

    @producer_work(work_attrs=WorkAttr.AUXILIARY)
    @cute.jit
    def init_store_state(self, stage_info: StageInfo) -> None:
        """Initialize correction/store task-local state for captured schedules."""
        # Producer aux work initializes the correction task's view of TMEM O.
        # Runtime SMEM views are initialized once in initialize_runtime_state_internal().
        self._init_tmem_state(stage_info)

    @cute.jit
    def _o_base_col(self):
        """Return the first TMEM column owned by the O accumulator."""

        if cutlass.const_expr(self.cfg.kernel_variant == "keeps_mma_ab"):
            return Int32(2 * self.cfg.tmem_s_cols)
        return Int32(2 * self.cfg.tmem_s_cols + 2 * self.cfg.tmem_stats_cols)

    @cute.jit
    def _init_cluster_reduction_barrier(self):
        """Initialize the transaction barrier for this CTA's reduction slice."""
        if cutlass.const_expr(self.cfg.use_cluster_reduction == 1):
            cfg = self.cfg
            warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
            cta_rank = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
            rows_per_slice = cfg.cluster_reduction_rows_per_slice
            num_slices_per_cta = ceil_div(
                cfg.cluster_reduction_slices, cfg.num_ctas_per_seq_kv
            )
            rows_per_cta = num_slices_per_cta * rows_per_slice
            first_row = cta_rank * Int32(rows_per_cta)
            remaining_rows = Int32(cfg.tile_size_q) - first_row
            valid_rows = cute.math.max(
                Int32(0),
                cute.math.min(Int32(rows_per_cta), remaining_rows),
            )
            bytes_per_row = Int32(
                cfg.head_dim_per_cta_v * cfg.partial_o_dtype_bytes + cfg.acc_dtype_bytes
            )
            expected_bytes = valid_rows * Int32(cfg.num_ctas_per_seq_kv) * bytes_per_row
            if warp_idx == Int32(cfg.correction_warp_idx):
                with cute.arch.elect_one():
                    prims.mbarrier_init(self._cluster_reduction_barrier, 1)
                    prims.mbarrier_arrive_expect_tx(
                        self._cluster_reduction_barrier, expected_bytes
                    )

    @cute.jit
    def _cluster_rows_per_cta(self):
        """Return how many Q rows this CTA owns during cluster reduction."""

        num_slices_per_cta = ceil_div(
            self.cfg.cluster_reduction_slices, self.cfg.num_ctas_per_seq_kv
        )
        return Int32(num_slices_per_cta * self.cfg.cluster_reduction_rows_per_slice)

    @cute.jit
    def _cluster_o_bytes(self):
        """Return the O bytes contributed by all split CTAs for this owner."""

        rows_per_cta = self._cluster_rows_per_cta()
        return Int64(
            Int32(self.cfg.num_ctas_per_seq_kv)
            * rows_per_cta
            * Int32(self.cfg.head_dim_per_cta_v * self.cfg.partial_o_dtype_bytes)
        )

    @cute.jit
    def _cluster_remote_smem_ptr(self, owner_rank, byte_offset, dtype):
        """Map a local cluster SMEM address to the CTA that owns the row slice."""

        local_ptr = self._cluster_local_smem_ptr(byte_offset, dtype)
        return prims.mapa(local_ptr, owner_rank)

    @cute.jit
    def _cluster_local_smem_ptr(self, byte_offset, dtype):
        """Return a typed pointer into this CTA's cluster reduction SMEM buffer."""

        return cutlass.inttoptr(
            self._cluster_reduction_smem.data_ptr().toint(Int64) + Int64(byte_offset),
            mem_space=3,
            dtype=dtype,
        )

    @cute.jit
    def _store_partial_o_to_cluster_smem(
        self,
        local_row_idx,
        cta_idx_kv,
        local_head_dim_elem_offset,
        local_head_dim_byte_offset,
        partial_vec,
    ):
        """Send one partial O vector to the CTA that owns its reduction slice."""
        rows_per_cta = self._cluster_rows_per_cta()
        owner_rank = local_row_idx // rows_per_cta
        owner_row_idx = local_row_idx - owner_rank * rows_per_cta
        elem_offset = (
            cta_idx_kv * rows_per_cta * Int32(self.cfg.head_dim_per_cta_v)
            + owner_row_idx * Int32(self.cfg.head_dim_per_cta_v)
            + local_head_dim_elem_offset
        )
        byte_offset = Int64(
            elem_offset * Int32(self.cfg.partial_o_dtype_bytes)
        ) + Int64(local_head_dim_byte_offset)
        remote_ptr = self._cluster_remote_smem_ptr(owner_rank, byte_offset, Int32)
        remote_barrier = prims.mapa(self._cluster_reduction_barrier, owner_rank)
        # Keep the vectorized publication on the public inline-PTX API so the
        # operation remains one 16-byte store instead of four scalar stores.
        cute.arch.inline_ptx(
            "st.async.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 "
            "[{$r0}], {{$r1}, {$r2}, {$r3}, {$r4}}, [{$r5}];",
            read_only_args=[
                remote_ptr.ir_value(),
                partial_vec[0],
                partial_vec[1],
                partial_vec[2],
                partial_vec[3],
                remote_barrier.ir_value(),
            ],
        )

    @cute.jit
    def _store_partial_lse_to_cluster_smem(self, local_row_idx, cta_idx_kv, lse_val):
        """Send one partial log-sum-exp value to the matching reduction owner."""
        rows_per_cta = self._cluster_rows_per_cta()
        owner_rank = local_row_idx // rows_per_cta
        owner_row_idx = local_row_idx - owner_rank * rows_per_cta
        stats_elem_offset = cta_idx_kv * rows_per_cta * Int32(
            2
        ) + owner_row_idx * Int32(2)
        byte_offset = self._cluster_o_bytes() + Int64(
            stats_elem_offset * Int32(self.cfg.acc_dtype_bytes)
        )
        remote_ptr = self._cluster_remote_smem_ptr(owner_rank, byte_offset, Int32)
        remote_barrier = prims.mapa(self._cluster_reduction_barrier, owner_rank)
        cute.arch.inline_ptx(
            "st.async.shared::cluster.mbarrier::complete_tx::bytes.b32 "
            "[{$r0}], {$r1}, [{$r2}];",
            read_only_args=[
                remote_ptr.ir_value(),
                lse_val.bitcast(Int32),
                remote_barrier.ir_value(),
            ],
        )

    @cute.jit
    def _store_o_slice_to_gmem(
        self,
        stage_info: StageInfo,
        task_cache,
        v_stage_idx: int,
        head_dim_offset: Int32,
    ):
        """Store one corrected O head-dim slice to final or split-KV output."""
        if cutlass.const_expr(self.o_tensor is None and self.acc_o_tensor is None):
            return

        cfg = self.cfg
        batch_idx = batch_idx_for_stage_cfg(self.batch_idx, cfg, stage_info)
        head_base_idx = head_idx_for_stage(self.head_idx, cfg, stage_info)
        cta_idx_q = cta_idx_q_for_stage(self.cta_idx_q, stage_info)
        cta_idx_kv = cta_idx_kv_for_stage(self.cta_idx_kv, stage_info)
        warp_grp_thread_idx = task_cache[_TASK_CACHE_WARP_GRP_THREAD_IDX]
        warp_idx = task_cache[_TASK_CACHE_WARP_IDX]
        lane_idx = task_cache[_TASK_CACHE_LANE_IDX]
        num_copy_segments = cfg.o_copy_segments_per_stage
        for copy_segment_idx in cutlass.range_constexpr(num_copy_segments):
            _, load_smem_offset, dst_row_idx, dst_col_offset = (
                o_stage_stsm_and_copy_offsets(
                    cfg,
                    warp_grp_thread_idx,
                    warp_idx,
                    lane_idx,
                    0,
                    copy_segment_idx,
                )
            )
            global_head_idx = head_base_idx + dst_row_idx
            if dst_row_idx < Int32(cfg.tile_size_q) and global_head_idx < Int32(
                cfg.num_heads_q
            ):
                storage_flat_query_row, _, _, _, valid_output_row = (
                    groups_tokens_heads_q_row_state(
                        global_head_idx,
                        cta_idx_q,
                        cfg.groups_tokens_heads_q_ratio,
                        cfg.logical_num_heads_q,
                        cfg.logical_seq_len_q,
                        cu_seqlens_q=self.cu_seqlens_q,
                        batch_idx=batch_idx,
                    )
                )
                smem_src = self._smem_o_i32.data_ptr(
                    load_smem_offset >> SMEM_WORD_BYTE_SHIFT
                )
                if cutlass.const_expr(cfg.use_cluster_reduction == 1):
                    self._store_partial_o_to_cluster_smem(
                        dst_row_idx,
                        cta_idx_kv,
                        Int32(v_stage_idx * cfg.head_dim_per_stage_v),
                        dst_col_offset,
                        smem_src.load(count=4, alignment=16),
                    )
                elif valid_output_row:
                    if cutlass.const_expr(self.acc_o_tensor is not None):
                        split_kv = Int32(cfg.num_ctas_per_seq_kv)
                        base_elem_offset = (
                            global_head_idx * Int32(split_kv * cfg.head_dim_v)
                            + cta_idx_kv * Int32(cfg.head_dim_v)
                            + head_dim_offset
                            + Int32(v_stage_idx * cfg.head_dim_per_stage_v)
                            + cta_idx_q
                            * Int32(cfg.num_heads_q * split_kv * cfg.head_dim_v)
                            + batch_idx
                            * Int32(
                                cfg.seq_len_q
                                * cfg.num_heads_q
                                * split_kv
                                * cfg.head_dim_v
                            )
                        )
                        base_ptr = self.acc_o_tensor.iterator.raw_ptr().toint(Int64)
                        element_bytes = cfg.partial_o_dtype_bytes
                    else:
                        output_query_row = public_query_flat_row(
                            cfg,
                            storage_flat_query_row,
                            batch_idx,
                            self.cu_seqlens_q,
                        )
                        base_elem_offset = (
                            Int64(output_query_row) * Int64(cfg.head_dim_v)
                            + Int64(head_dim_offset)
                            + Int64(v_stage_idx * cfg.head_dim_per_stage_v)
                        )
                        base_ptr = self.o_tensor.iterator.raw_ptr().toint(Int64)
                        element_bytes = cfg.o_dtype_bytes
                    byte_offset = Int64(base_elem_offset) * Int64(
                        element_bytes
                    ) + Int64(dst_col_offset)
                    dst_ptr = cutlass.inttoptr(
                        base_ptr + byte_offset,
                        mem_space=1,
                        dtype=Int32,
                    )
                    dst_ptr.store(
                        smem_src.load(count=4, alignment=16),
                        alignment=16,
                    )

    @cute.jit
    def _store_lse_to_gmem(
        self, stage_info: StageInfo, task_cache, final_max, reduced_sum
    ):
        """Store final LSE or split-KV partial LSE for one corrected tile."""
        if cutlass.const_expr(self.lse_tensor is None and self.acc_lse_tensor is None):
            return
        batch_idx = batch_idx_for_stage_cfg(self.batch_idx, self.cfg, stage_info)
        head_base_idx = head_idx_for_stage(self.head_idx, self.cfg, stage_info)
        cta_idx_q = cta_idx_q_for_stage(self.cta_idx_q, stage_info)
        cta_idx_kv = cta_idx_kv_for_stage(self.cta_idx_kv, stage_info)
        cta_idx_head_dim_v = cta_idx_head_dim_v_for_stage(
            self.cta_idx_head_dim_v, stage_info
        )
        warp_idx = task_cache[_TASK_CACHE_WARP_IDX]
        lane_idx = task_cache[_TASK_CACHE_LANE_IDX]
        if cutlass.const_expr(self.cfg.kernel_variant == "keeps_mma_ab"):
            should_store_lse = lane_idx < Int32(16)
            local_row_idx = warp_idx * Int32(16) + (lane_idx & Int32(0xF))
            head_idx = head_base_idx + local_row_idx
            scale_idx = Int32(0)
        else:
            should_store_lse = warp_idx == Int32(0) and lane_idx < Int32(
                4 * num_softmax_scale_groups(self.cfg)
            )
            col_group_idx = lane_idx & Int32(0x3)
            scale_idx = lane_idx >> Int32(2)
            local_row_idx = local_q_head_idx_for_scale(
                self.cfg, col_group_idx, scale_idx
            )
            head_idx = head_base_idx + local_row_idx
        if should_store_lse:
            # Swaps keeps at least four scale groups, so TileQ8 has eight
            # padded rows in the correction register footprint.  Do not let
            # those rows publish split LSE into the next logical head tile.
            if local_row_idx < Int32(self.cfg.tile_size_q) and head_idx < Int32(
                self.cfg.num_heads_q
            ):
                storage_flat_query_row, _, _, _, valid_output_row = (
                    groups_tokens_heads_q_row_state(
                        head_idx,
                        cta_idx_q,
                        self.cfg.groups_tokens_heads_q_ratio,
                        self.cfg.logical_num_heads_q,
                        self.cfg.logical_seq_len_q,
                        cu_seqlens_q=self.cu_seqlens_q,
                        batch_idx=batch_idx,
                    )
                )
                lse_sum = reduced_sum[scale_idx]
                if cutlass.const_expr(self.cfg.is_fp8_qkv()):
                    # FP8 P is quantized as 448 * softmax(P) for BMM2.  The
                    # online sum tracks the same scale, so undo it for the
                    # externally visible log-sum-exp value.
                    lse_sum = lse_sum * fp8_quant_scale_rcp()
                lse_val = (
                    cute.math.log2(lse_sum, fastmath=True)
                    + self.scale_softmax_log2 * final_max[scale_idx]
                )
                if cutlass.const_expr(self.cfg.use_cluster_reduction == 1):
                    self._store_partial_lse_to_cluster_smem(
                        local_row_idx, cta_idx_kv, lse_val
                    )
                elif cutlass.const_expr(self.acc_lse_tensor is not None):
                    if valid_output_row and cta_idx_head_dim_v == Int32(0):
                        split_kv = Int32(self.cfg.num_ctas_per_seq_kv)
                        elem_offset = (
                            batch_idx
                            * Int32(
                                self.cfg.seq_len_q * self.cfg.num_heads_q * split_kv
                            )
                            + cta_idx_q * Int32(self.cfg.num_heads_q * split_kv)
                            + head_idx * split_kv
                            + cta_idx_kv
                        )
                        (self.acc_lse_tensor.iterator.raw_ptr() + elem_offset).store(
                            lse_val
                        )
                else:
                    if valid_output_row:
                        elem_offset = public_query_flat_row(
                            self.cfg,
                            storage_flat_query_row,
                            batch_idx,
                            self.cu_seqlens_q,
                        )
                        (self.lse_tensor.iterator.raw_ptr() + elem_offset).store(
                            lse_val
                        )

    @cute.jit
    def _cluster_lse_byte_offset(self, split_idx, owner_row_idx):
        rows_per_cta = self._cluster_rows_per_cta()
        stats_elem_offset = split_idx * rows_per_cta * Int32(2) + owner_row_idx * Int32(
            2
        )
        return self._cluster_o_bytes() + Int64(
            stats_elem_offset * Int32(self.cfg.acc_dtype_bytes)
        )

    @cute.jit
    def _cluster_o_byte_offset(self, split_idx, owner_row_idx, dim_idx):
        rows_per_cta = self._cluster_rows_per_cta()
        elem_offset = (
            split_idx * rows_per_cta * Int32(self.cfg.head_dim_per_cta_v)
            + owner_row_idx * Int32(self.cfg.head_dim_per_cta_v)
            + dim_idx
        )
        return Int64(elem_offset * Int32(self.cfg.partial_o_dtype_bytes))

    @cute.jit
    def _cluster_wait_transaction_barrier(self, warp_grp_thread_idx):
        """Wait for peer cluster stores before the owner reads local DSMEM."""

        # Give every correction lane one nonblocking acquire attempt. If the
        # transaction is not already complete, only lane 0 keeps polling. The
        # correction store barrier is free after the final O-staging phase and
        # publishes lane 0's ready point to all 128 correction lanes before
        # any of them reads the distributed-SMEM partials below.
        cluster_transaction_ready = prims.mbarrier_try_wait_parity(
            self._cluster_reduction_barrier, 0, time_limit=0
        )
        if warp_grp_thread_idx == Int32(0):
            while not cluster_transaction_ready:
                cluster_transaction_ready = prims.mbarrier_try_wait_parity(
                    self._cluster_reduction_barrier, 0, time_limit=10_000_000
                )
        prims.barrier_cta_sync(
            barrier_id=self.store_barrier_id,
            thread_count=WARPGROUP_THREADS,
        )

    @cute.jit
    def publish_neutral_cluster_partial_and_reduce(
        self,
        batch_idx,
        head_base_idx,
        cta_idx_q,
        cta_idx_kv,
        cta_idx_head_dim_v,
    ):
        """Publish an empty split and execute this rank's static reduction work.

        Runtime split pruning only removes the K/V task graph.  Every launched
        cluster rank still contributes the configured partial byte count so owner
        barriers, DSMEM offsets, and row ownership remain compile-time static.
        The neutral pair ``O=0, LSE=-inf`` has no effect on online-softmax
        reduction and is safe for mutable runtime sequence lengths.
        """

        if cutlass.const_expr(self.cfg.use_cluster_reduction == 1):
            cfg = self.cfg
            warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
            is_correction_warp = warp_idx >= Int32(
                cfg.correction_warp_idx
            ) and warp_idx < Int32(cfg.correction_warp_idx + cfg.correction_num_warps)
            if is_correction_warp:
                thread_idx, _, _ = cute.arch.thread_idx()
                warp_grp_thread_idx = thread_idx - Int32(cfg.correction_warp_idx * 32)
                base_vec_offset = warp_grp_thread_idx * Int32(8)
                row_in_slice = base_vec_offset // Int32(cfg.head_dim_per_cta_v)
                dim_idx = base_vec_offset - row_in_slice * Int32(cfg.head_dim_per_cta_v)
                neutral_o = vector_from_scalars(
                    (Int32(0), Int32(0), Int32(0), Int32(0)),
                    dtype=Int32,
                )
                for slice_idx in cutlass.range_constexpr(cfg.cluster_reduction_slices):
                    local_row_idx = (
                        Int32(slice_idx * cfg.cluster_reduction_rows_per_slice)
                        + row_in_slice
                    )
                    global_head_idx = head_base_idx + local_row_idx
                    if local_row_idx < Int32(
                        cfg.tile_size_q
                    ) and global_head_idx < Int32(cfg.num_heads_q):
                        self._store_partial_o_to_cluster_smem(
                            local_row_idx,
                            cta_idx_kv,
                            dim_idx,
                            Int32(0),
                            neutral_o,
                        )
                        if dim_idx == Int32(0):
                            self._store_partial_lse_to_cluster_smem(
                                local_row_idx,
                                cta_idx_kv,
                                Float32(-Float32.inf),
                            )

                head_dim_offset = head_dim_cta_offset_v(cfg, cta_idx_head_dim_v)
                self._reduce_cluster_partials_and_store_for_tile(
                    batch_idx,
                    head_base_idx,
                    cta_idx_q,
                    cta_idx_kv,
                    cta_idx_head_dim_v,
                    warp_grp_thread_idx,
                    head_dim_offset,
                )

    @cute.jit
    def _reduce_cluster_partials_and_store(
        self, stage_info: StageInfo, task_cache, head_dim_offset
    ):
        """Reduce DSMEM split-KV partials for this CTA rank's row slices."""
        if cutlass.const_expr(self.cfg.use_cluster_reduction == 1):
            cfg = self.cfg
            batch_idx = batch_idx_for_stage_cfg(self.batch_idx, cfg, stage_info)
            head_base_idx = head_idx_for_stage(self.head_idx, cfg, stage_info)
            cta_idx_q = cta_idx_q_for_stage(self.cta_idx_q, stage_info)
            cta_idx_kv = cta_idx_kv_for_stage(self.cta_idx_kv, stage_info)
            cta_idx_head_dim_v = cta_idx_head_dim_v_for_stage(
                self.cta_idx_head_dim_v, stage_info
            )
            warp_grp_thread_idx = task_cache[_TASK_CACHE_WARP_GRP_THREAD_IDX]
            self._reduce_cluster_partials_and_store_for_tile(
                batch_idx,
                head_base_idx,
                cta_idx_q,
                cta_idx_kv,
                cta_idx_head_dim_v,
                warp_grp_thread_idx,
                head_dim_offset,
            )

    @cute.jit
    def _reduce_cluster_partials_and_store_for_tile(
        self,
        batch_idx,
        head_base_idx,
        cta_idx_q,
        cta_idx_kv,
        cta_idx_head_dim_v,
        warp_grp_thread_idx,
        head_dim_offset,
    ):
        """Reduce one configured-S cluster row-owner slice."""
        if cutlass.const_expr(self.cfg.use_cluster_reduction == 1):
            cfg = self.cfg
            rows_per_slice = cfg.cluster_reduction_rows_per_slice
            num_slices_per_cta = ceil_div(
                cfg.cluster_reduction_slices, cfg.num_ctas_per_seq_kv
            )
            rows_per_cta = num_slices_per_cta * rows_per_slice
            owner_first_row = cta_idx_kv * Int32(rows_per_cta)
            valid_rows = cute.math.max(
                Int32(0),
                cute.math.min(
                    Int32(rows_per_cta),
                    Int32(cfg.tile_size_q) - owner_first_row,
                ),
            )

            if valid_rows > Int32(0):
                self._cluster_wait_transaction_barrier(warp_grp_thread_idx)

                base_vec_offset = warp_grp_thread_idx * Int32(8)
                row_in_slice = base_vec_offset // Int32(cfg.head_dim_per_cta_v)
                dim_idx = base_vec_offset - row_in_slice * Int32(cfg.head_dim_per_cta_v)
                output_element_dtype = output_dtype(cfg)
                partial_element_dtype = partial_output_dtype(cfg)

                for slice_offset in cutlass.range_constexpr(num_slices_per_cta):
                    owner_row_idx = Int32(slice_offset * rows_per_slice) + row_in_slice
                    local_row_idx = owner_first_row + owner_row_idx
                    if local_row_idx < Int32(cfg.tile_size_q):
                        global_head_idx = head_base_idx + local_row_idx
                        if global_head_idx < Int32(cfg.num_heads_q):
                            (
                                storage_flat_query_row,
                                _,
                                _,
                                _,
                                valid_output_row,
                            ) = groups_tokens_heads_q_row_state(
                                global_head_idx,
                                cta_idx_q,
                                cfg.groups_tokens_heads_q_ratio,
                                cfg.logical_num_heads_q,
                                cfg.logical_seq_len_q,
                                cu_seqlens_q=self.cu_seqlens_q,
                                batch_idx=batch_idx,
                            )
                            lse_max = Float32(-Float32.inf)
                            local_lse = cutlass.Array(
                                Float32,
                                cfg.num_ctas_per_seq_kv,
                                space=cutlass.AddressSpace.rmem,
                            )
                            for split_idx in cutlass.range_constexpr(
                                cfg.num_ctas_per_seq_kv
                            ):
                                lse_ptr = self._cluster_local_smem_ptr(
                                    self._cluster_lse_byte_offset(
                                        Int32(split_idx), owner_row_idx
                                    ),
                                    Int32,
                                )
                                lse_val = lse_ptr.load().bitcast(Float32)
                                local_lse[split_idx] = lse_val
                                lse_max = cute.math.max(lse_max, lse_val, ftz=True)

                            lse_max = (
                                lse_max
                                if lse_max != Float32(-Float32.inf)
                                else Float32(0.0)
                            )
                            lse_sum = Float32(0.0)
                            for split_idx in cutlass.range_constexpr(
                                cfg.num_ctas_per_seq_kv
                            ):
                                lse_sum += cute.math.exp2(
                                    local_lse[split_idx] - lse_max, fastmath=True
                                )
                            global_lse = (
                                lse_max + cute.math.log2(lse_sum, fastmath=True)
                                if lse_sum != Float32(0.0) and lse_sum == lse_sum
                                else Float32(Float32.inf)
                            )

                            if (
                                valid_output_row
                                and dim_idx == Int32(0)
                                and cta_idx_head_dim_v == Int32(0)
                            ):
                                if cutlass.const_expr(self.lse_tensor is not None):
                                    lse_offset = public_query_flat_row(
                                        cfg,
                                        storage_flat_query_row,
                                        batch_idx,
                                        self.cu_seqlens_q,
                                    )
                                    (
                                        self.lse_tensor.iterator.raw_ptr() + lse_offset
                                    ).store(global_lse)

                            acc_vec = vector_from_scalars(
                                (
                                    Float32(0.0),
                                    Float32(0.0),
                                    Float32(0.0),
                                    Float32(0.0),
                                    Float32(0.0),
                                    Float32(0.0),
                                    Float32(0.0),
                                    Float32(0.0),
                                ),
                                dtype=Float32,
                            )
                            for split_idx in cutlass.range_constexpr(
                                cfg.num_ctas_per_seq_kv
                            ):
                                scale = cute.math.exp2(
                                    local_lse[split_idx] - global_lse, fastmath=True
                                )
                                partial_ptr = self._cluster_local_smem_ptr(
                                    self._cluster_o_byte_offset(
                                        Int32(split_idx), owner_row_idx, dim_idx
                                    ),
                                    partial_element_dtype,
                                )
                                partial_vec = partial_ptr.load(
                                    count=8, alignment=16
                                ).to(Float32)
                                acc_vec = acc_vec + partial_vec * scale

                            if cutlass.const_expr(self.o_tensor is not None):
                                if valid_output_row:
                                    output_query_row = public_query_flat_row(
                                        cfg,
                                        storage_flat_query_row,
                                        batch_idx,
                                        self.cu_seqlens_q,
                                    )
                                    out_elem_offset = (
                                        Int64(output_query_row) * Int64(cfg.head_dim_v)
                                        + Int64(head_dim_offset)
                                        + Int64(dim_idx)
                                    )
                                    out_byte_offset = out_elem_offset * Int64(
                                        cfg.o_dtype_bytes
                                    )
                                    if cutlass.const_expr(cfg.use_fp8_output == 1):
                                        packed_o = cutlass.Array(
                                            Int32, 2, space=cutlass.AddressSpace.rmem
                                        )
                                        packed_o[0] = pack_float4_to_fp8_e4m3(
                                            acc_vec[0],
                                            acc_vec[1],
                                            acc_vec[2],
                                            acc_vec[3],
                                        )
                                        packed_o[1] = pack_float4_to_fp8_e4m3(
                                            acc_vec[4],
                                            acc_vec[5],
                                            acc_vec[6],
                                            acc_vec[7],
                                        )
                                        out_ptr = cutlass.inttoptr(
                                            self.o_tensor.iterator.raw_ptr().toint(
                                                Int64
                                            )
                                            + out_byte_offset,
                                            mem_space=1,
                                            dtype=Int32,
                                        )
                                        out_ptr.store(packed_o.load(0, 2), alignment=8)
                                    else:
                                        out_ptr = cutlass.inttoptr(
                                            self.o_tensor.iterator.raw_ptr().toint(
                                                Int64
                                            )
                                            + out_byte_offset,
                                            mem_space=1,
                                            dtype=output_element_dtype,
                                        )
                                        out_ptr.store(
                                            acc_vec.to(output_element_dtype),
                                            alignment=16,
                                        )

    @producer_work
    @cute.jit
    def correct_loop_and_store(
        self,
        stage_info: StageInfo,
        *,
        old_max_arr,
        new_max_arr,
        sum_arr,
        inst0_new_max_arr,
        inst0_sum_arr,
        inst1_new_max_arr,
        inst1_sum_arr,
        o_stage_idx,
        tail_o_stage_idx_0,
        tail_o_stage_idx_1,
    ):
        """Rescale the running O accumulator for a loop-stage softmax update."""
        del sum_arr, inst0_new_max_arr, inst0_sum_arr
        del inst1_new_max_arr, inst1_sum_arr
        del tail_o_stage_idx_0, tail_o_stage_idx_1
        self._rescale_loop_o(
            stage_info,
            old_max_arr=old_max_arr,
            new_max_arr=new_max_arr,
            o_stage_idx=o_stage_idx,
        )

    @producer_work
    @cute.jit
    def correct_tail_and_store(
        self,
        stage_info: StageInfo,
        *,
        old_max_arr,
        new_max_arr,
        sum_arr,
        inst0_new_max_arr,
        inst0_sum_arr,
        inst1_new_max_arr,
        inst1_sum_arr,
        o_stage_idx,
        tail_o_stage_idx_0,
        tail_o_stage_idx_1,
    ):
        """Normalize final O, then store O/LSE or split-KV partials."""
        del old_max_arr, o_stage_idx
        self._normalize_and_store_tail_o(
            stage_info,
            new_max_arr=new_max_arr,
            sum_arr=sum_arr,
            inst0_new_max_arr=inst0_new_max_arr,
            inst0_sum_arr=inst0_sum_arr,
            inst1_new_max_arr=inst1_new_max_arr,
            inst1_sum_arr=inst1_sum_arr,
            tail_o_stage_idx_0=tail_o_stage_idx_0,
            tail_o_stage_idx_1=tail_o_stage_idx_1,
        )

    @cute.jit
    def _apply_loop_correction_in_tmem(
        self,
        stage_info: StageInfo,
        *,
        scale_vals,
        o_stage_idx,
    ):
        """Apply one nonidentity loop correction to the live TMEM O stage."""
        cfg = self.cfg
        task_cache = decode_gen_task_cache(stage_info)
        tmem_row_base = task_cache[_TASK_CACHE_TMEM_BASE_OFFSET]
        shape = TCGEN05_16X256B_SHAPE
        q_repeats = num_o_repeats(cfg)
        o_reg_pair_count = num_o_reg_pairs(cfg)
        num_o_tmem_loads = num_o_tmem_loads_per_stage(cfg)

        if cutlass.const_expr(cfg.kernel_variant == "keeps_mma_ab"):
            for v_stage_idx in cutlass.range_constexpr(cfg.v_head_dim_stages):
                base_addr = (
                    tmem_row_base
                    + self._o_base_col()
                    + o_stage_tmem_col_offset(cfg, o_stage_idx, v_stage_idx)
                )
                loaded = tcgen05_ld_16x32bx2_f32(
                    prims.make_tmem_ptr(base_addr, Float32),
                    num=cfg.head_dim_per_stage_v // 2,
                    offset=Int32(cfg.head_dim_per_stage_v // 2),
                )
                prims.tcgen05_wait(kind=prims.Tcgen05Wait.LOAD)
                cute.arch.fence_view_async_tmem_load()
                scaled = cutlass.Array(
                    Float32,
                    cfg.head_dim_per_stage_v // 2,
                    space=cutlass.AddressSpace.rmem,
                )
                for reg_idx in cutlass.range_constexpr(cfg.head_dim_per_stage_v // 2):
                    scaled[reg_idx] = loaded[reg_idx] * scale_vals[0]
                scaled_vec = vector_from_scalars(
                    tuple(
                        scaled[reg_idx]
                        for reg_idx in range(cfg.head_dim_per_stage_v // 2)
                    ),
                    Float32,
                )
                tcgen05_st_16x32bx2_f32(
                    prims.make_tmem_ptr(base_addr, Float32),
                    scaled_vec,
                    offset=Int32(cfg.head_dim_per_stage_v // 2),
                )
            return

        for v_stage_idx in cutlass.range_constexpr(cfg.v_head_dim_stages):
            base_addr = (
                tmem_row_base
                + self._o_base_col()
                + o_stage_tmem_col_offset(cfg, o_stage_idx, v_stage_idx)
            )
            for chunk_idx in cutlass.range_constexpr(num_o_tmem_loads):
                if cutlass.const_expr(cfg.tile_size_q == 8):
                    loaded_lo = prims.tcgen05_ld(
                        shape,
                        prims.make_tmem_ptr(
                            tcgen05_panel_addr(base_addr, chunk_idx), Float32
                        ),
                        num=1,
                    )
                    loaded_hi = prims.tcgen05_ld(
                        shape,
                        prims.make_tmem_ptr(
                            tcgen05_second_panel_addr(
                                tcgen05_panel_addr(base_addr, chunk_idx)
                            ),
                            Float32,
                        ),
                        num=1,
                    )
                else:
                    loaded = prims.tcgen05_ld(
                        shape,
                        prims.make_tmem_ptr(
                            tcgen05_panel_addr(base_addr, chunk_idx), Float32
                        ),
                        num=q_repeats,
                    )
                prims.tcgen05_wait(kind=prims.Tcgen05Wait.LOAD)
                cute.arch.fence_view_async_tmem_load()

                if cutlass.const_expr(cfg.tile_size_q == 8):
                    scaled_lo = cutlass.Array(
                        Float32, 4, space=cutlass.AddressSpace.rmem
                    )
                    scaled_hi = cutlass.Array(
                        Float32, 4, space=cutlass.AddressSpace.rmem
                    )
                    for reg_pair_idx in cutlass.range_constexpr(o_reg_pair_count):
                        scale_base = (
                            (reg_pair_idx % (2 * max(cfg.tile_size_q // 8, 1))) // 2
                        ) * 2
                        reg_base = (reg_pair_idx % 2) * 2
                        if cutlass.const_expr(reg_pair_idx < 2):
                            scaled_lo[reg_base] = (
                                loaded_lo[reg_base] * scale_vals[scale_base]
                            )
                            scaled_lo[reg_base + 1] = (
                                loaded_lo[reg_base + 1] * scale_vals[scale_base + 1]
                            )
                        else:
                            scaled_hi[reg_base] = (
                                loaded_hi[reg_base] * scale_vals[scale_base]
                            )
                            scaled_hi[reg_base + 1] = (
                                loaded_hi[reg_base + 1] * scale_vals[scale_base + 1]
                            )
                    scaled_lo_vec = vector_from_scalars(
                        (scaled_lo[0], scaled_lo[1], scaled_lo[2], scaled_lo[3]),
                        Float32,
                    )
                    scaled_hi_vec = vector_from_scalars(
                        (scaled_hi[0], scaled_hi[1], scaled_hi[2], scaled_hi[3]),
                        Float32,
                    )
                    prims.tcgen05_st(
                        shape,
                        prims.make_tmem_ptr(
                            tcgen05_panel_addr(base_addr, chunk_idx), Float32
                        ),
                        scaled_lo_vec,
                    )
                    prims.tcgen05_st(
                        shape,
                        prims.make_tmem_ptr(
                            tcgen05_second_panel_addr(
                                tcgen05_panel_addr(base_addr, chunk_idx)
                            ),
                            Float32,
                        ),
                        scaled_hi_vec,
                    )
                else:
                    scaled = cutlass.Array(
                        Float32, 4 * q_repeats, space=cutlass.AddressSpace.rmem
                    )
                    for reg_pair_idx in cutlass.range_constexpr(o_reg_pair_count):
                        scale_base = (
                            (reg_pair_idx % (2 * max(cfg.tile_size_q // 8, 1))) // 2
                        ) * 2
                        reg_base = reg_pair_idx * 2
                        scaled[reg_base] = loaded[reg_base] * scale_vals[scale_base]
                        scaled[reg_base + 1] = (
                            loaded[reg_base + 1] * scale_vals[scale_base + 1]
                        )
                    scaled_vec = vector_from_scalars(
                        tuple(scaled[reg_idx] for reg_idx in range(4 * q_repeats)),
                        Float32,
                    )
                    prims.tcgen05_st(
                        shape,
                        prims.make_tmem_ptr(
                            tcgen05_panel_addr(base_addr, chunk_idx), Float32
                        ),
                        scaled_vec,
                    )
        return

    @cute.jit
    def _rescale_loop_o(
        self,
        stage_info: StageInfo,
        *,
        old_max_arr,
        new_max_arr,
        o_stage_idx,
    ):
        """Rescale the running O accumulator after a loop softmax update."""
        cfg = self.cfg
        num_scale_groups = num_softmax_scale_groups(cfg)

        # A loop correction is exactly identity when its row maximum did not
        # change. Avoid exp2 for identity lanes, and skip collective TMEM
        # traffic only when the entire correction warp agrees.
        scale_vals = cutlass.Array(
            Float32, num_scale_groups, space=cutlass.AddressSpace.rmem
        )
        if cutlass.const_expr(cfg.kernel_variant == "keeps_mma_ab"):
            old_max = old_max_arr[0]
            new_max = new_max_arr[0]
            scale_is_identity = old_max == new_max
            scale_vals[0] = Float32(1.0)
            if not scale_is_identity:
                scale_vals[0] = cute.math.exp2(
                    self.scale_softmax_log2 * (old_max - new_max),
                    fastmath=True,
                )
            lane_scales_are_identity = scale_is_identity
        else:
            lane_scales_are_identity = cutlass.Boolean(True)
            for scale_base in cutlass.range_constexpr(0, num_scale_groups, 2):
                old_max_0 = old_max_arr[scale_base]
                old_max_1 = old_max_arr[scale_base + 1]
                new_max_0 = new_max_arr[scale_base]
                new_max_1 = new_max_arr[scale_base + 1]
                scale_0_is_identity = old_max_0 == new_max_0
                scale_1_is_identity = old_max_1 == new_max_1
                scale_vals[scale_base] = Float32(1.0)
                scale_vals[scale_base + 1] = Float32(1.0)
                max_diff_pair = fadd2(
                    (old_max_0, old_max_1),
                    (-new_max_0, -new_max_1),
                )
                scale_pair = fmul2(
                    (self.scale_softmax_log2, self.scale_softmax_log2),
                    max_diff_pair,
                )
                if not scale_0_is_identity:
                    scale_vals[scale_base] = cute.math.exp2(
                        scale_pair[0], fastmath=True
                    )
                if not scale_1_is_identity:
                    scale_vals[scale_base + 1] = cute.math.exp2(
                        scale_pair[1], fastmath=True
                    )
                lane_scales_are_identity = (
                    lane_scales_are_identity & scale_0_is_identity & scale_1_is_identity
                )

        skip_rescale = prims.vote_sync(
            cute.arch.FULL_MASK,
            lane_scales_are_identity,
            prims.VoteSync.ALL,
        )
        if not skip_rescale:
            self._apply_loop_correction_in_tmem(
                stage_info,
                scale_vals=scale_vals,
                o_stage_idx=o_stage_idx,
            )
        # Preserve the correction task's TMEM ordering point even when this
        # warp had no rescale transaction to issue.
        prims.tcgen05_wait(kind=prims.Tcgen05Wait.STORE)
        if not skip_rescale:
            cute.arch.fence_view_async_tmem_store()

    @cute.jit
    def _compute_tail_softmax_scales(
        self,
        task_cache,
        *,
        new_max_arr,
        sum_arr,
        inst0_new_max_arr,
        inst0_sum_arr,
        inst1_new_max_arr,
        inst1_sum_arr,
    ):
        """Compute final max/sum values and O normalization scales."""
        cfg = self.cfg
        num_scale_groups = num_softmax_scale_groups(cfg)
        final_max = cutlass.Array(
            Float32, num_scale_groups, space=cutlass.AddressSpace.rmem
        )
        final_sum = cutlass.Array(
            Float32, num_scale_groups, space=cutlass.AddressSpace.rmem
        )
        exp_scale0 = cutlass.Array(
            Float32, num_scale_groups, space=cutlass.AddressSpace.rmem
        )
        exp_scale1 = cutlass.Array(
            Float32, num_scale_groups, space=cutlass.AddressSpace.rmem
        )
        reduced_sum = cutlass.Array(
            Float32, num_scale_groups, space=cutlass.AddressSpace.rmem
        )
        final_scale0 = cutlass.Array(
            Float32, num_scale_groups, space=cutlass.AddressSpace.rmem
        )
        final_scale1 = cutlass.Array(
            Float32, num_scale_groups, space=cutlass.AddressSpace.rmem
        )
        warp_grp_thread_idx = task_cache[_TASK_CACHE_WARP_GRP_THREAD_IDX]
        warp_idx = task_cache[_TASK_CACHE_WARP_IDX]
        lane_idx = task_cache[_TASK_CACHE_LANE_IDX]
        # Four column groups share one softmax row through separate lane
        # subsets.  The low two thread bits select the group-local scratch lane.
        col_group_idx = warp_grp_thread_idx & Int32(0x3)

        for scale_idx in cutlass.range_constexpr(num_scale_groups):
            if cutlass.const_expr(cfg.kernel_variant == "keeps_mma_ab"):
                inst_sum = sum_arr[scale_idx]
                inst_max = new_max_arr[scale_idx]
                final_max[scale_idx] = inst_max
                exp_scale0[scale_idx] = Float32(1.0)
                exp_scale1[scale_idx] = Float32(0.0)
                final_sum[scale_idx] = inst_sum
                final_sum[scale_idx] += Float32(
                    cprims.shfl_sync(
                        thread_mask=0xFFFFFFFF,
                        val=final_sum[scale_idx],
                        offset=16,
                        mask_and_clamp=0x1F,
                        kind=cprims.Shfl.BFLY,
                    )
                )
                reduced_sum[scale_idx] = final_sum[scale_idx]
            else:
                inst0_sum = inst0_sum_arr[scale_idx]
                inst1_sum = inst1_sum_arr[scale_idx]
                inst0_max = inst0_new_max_arr[scale_idx]
                inst1_max = inst1_new_max_arr[scale_idx]
                final_max[scale_idx] = cute.math.max(inst0_max, inst1_max, ftz=True)
                exp_scale0[scale_idx] = cute.math.exp2(
                    self.scale_softmax_log2 * (inst0_max - final_max[scale_idx]),
                    fastmath=True,
                )
                exp_scale1[scale_idx] = cute.math.exp2(
                    self.scale_softmax_log2 * (inst1_max - final_max[scale_idx]),
                    fastmath=True,
                )
                final_sum[scale_idx] = (
                    inst0_sum * exp_scale0[scale_idx]
                    + inst1_sum * exp_scale1[scale_idx]
                )
                final_sum[scale_idx] += Float32(
                    cprims.shfl_sync(
                        thread_mask=0xFFFFFFFF,
                        val=final_sum[scale_idx],
                        offset=16,
                        mask_and_clamp=0x1F,
                        kind=cprims.Shfl.BFLY,
                    )
                )
                final_sum[scale_idx] += Float32(
                    cprims.shfl_sync(
                        thread_mask=0xFFFFFFFF,
                        val=final_sum[scale_idx],
                        offset=8,
                        mask_and_clamp=0x1F,
                        kind=cprims.Shfl.BFLY,
                    )
                )
                final_sum[scale_idx] += Float32(
                    cprims.shfl_sync(
                        thread_mask=0xFFFFFFFF,
                        val=final_sum[scale_idx],
                        offset=4,
                        mask_and_clamp=0x1F,
                        kind=cprims.Shfl.BFLY,
                    )
                )

        if not cutlass.const_expr(cfg.kernel_variant == "keeps_mma_ab"):
            # The two softmax pipes first reduce within their owning warp, then
            # publish one value per column group so the warpgroup can form the
            # final normalization denominator before O is stored.
            warp_store_base = warp_idx * Int32(
                4 * num_scale_groups
            ) + col_group_idx * Int32(num_scale_groups)
            if lane_idx < Int32(4):
                self._sum_scratch.store(
                    tuple(
                        final_sum[scale_idx] for scale_idx in range(num_scale_groups)
                    ),
                    warp_store_base,
                    alignment=16 if num_scale_groups >= 4 else 8,
                )
            prims.barrier_cta_sync(
                barrier_id=self.sum_barrier_id, thread_count=WARPGROUP_THREADS
            )

            reduce_base = col_group_idx * Int32(num_scale_groups)
            reduced_vec = self._sum_scratch.load(
                reduce_base,
                vector_size=num_scale_groups,
                alignment=16 if num_scale_groups == 4 else 8,
            )
            for scale_idx in cutlass.range_constexpr(num_scale_groups):
                reduced_sum[scale_idx] = reduced_vec[scale_idx]
            for warp_offset in cutlass.range_constexpr(1, 4):
                other_vec = self._sum_scratch.load(
                    reduce_base + warp_offset * Int32(4 * num_scale_groups),
                    vector_size=num_scale_groups,
                    alignment=16 if num_scale_groups == 4 else 8,
                )
                for scale_idx in cutlass.range_constexpr(num_scale_groups):
                    reduced_sum[scale_idx] += other_vec[scale_idx]

        for scale_idx in cutlass.range_constexpr(num_scale_groups):
            norm_scale = Float32(0.0)
            if reduced_sum[scale_idx] != Float32(0.0):
                norm_scale = self.output_scale / reduced_sum[scale_idx]
            final_scale0[scale_idx] = norm_scale * exp_scale0[scale_idx]
            final_scale1[scale_idx] = norm_scale * exp_scale1[scale_idx]
        return final_max, reduced_sum, final_scale0, final_scale1

    @cute.jit
    def _store_keeps_tail_o_stage(
        self,
        stage_info: StageInfo,
        task_cache,
        *,
        base_addr0,
        final_scale0,
        v_stage_idx: int,
        head_dim_offset,
    ):
        """Normalize and directly store one keeps-MMA-AB O stage."""
        cfg = self.cfg
        batch_idx = batch_idx_for_stage_cfg(self.batch_idx, cfg, stage_info)
        head_base_idx = head_idx_for_stage(self.head_idx, cfg, stage_info)
        cta_idx_q = cta_idx_q_for_stage(self.cta_idx_q, stage_info)
        cta_idx_kv = cta_idx_kv_for_stage(self.cta_idx_kv, stage_info)
        warp_idx = task_cache[_TASK_CACHE_WARP_IDX]
        lane_idx = task_cache[_TASK_CACHE_LANE_IDX]
        # Keeps-MMA-AB stores one head row per half-warp.  The low 16 lanes own
        # the row index; the upper lane bit selects the head-dim half.
        local_row_idx = warp_idx * Int32(16) + (lane_idx & Int32(0xF))
        global_head_idx = head_base_idx + local_row_idx
        storage_flat_query_row, _, _, _, valid_output_row = (
            groups_tokens_heads_q_row_state(
                global_head_idx,
                cta_idx_q,
                cfg.groups_tokens_heads_q_ratio,
                cfg.logical_num_heads_q,
                cfg.logical_seq_len_q,
                cu_seqlens_q=self.cu_seqlens_q,
                batch_idx=batch_idx,
            )
        )
        half_warp_col_offset = (lane_idx >> Int32(4)) * Int32(
            cfg.head_dim_per_stage_v // 2
        )

        # ``tcgen05.ld.sync.aligned`` must remain convergent across the warp.
        # Padded query rows still own valid physical TMEM rows, so load and
        # normalize every row and predicate only the externally visible store.
        for chunk_idx in cutlass.range_constexpr(0, cfg.head_dim_per_stage_v // 2, 8):
            o_loaded = tcgen05_ld_16x32bx2_f32(
                prims.make_tmem_ptr(base_addr0 + Int32(chunk_idx), Float32),
                num=8,
                offset=Int32(cfg.head_dim_per_stage_v // 2),
            )
            prims.tcgen05_wait(kind=prims.Tcgen05Wait.LOAD)
            cute.arch.fence_view_async_tmem_load()
            if cutlass.const_expr(
                cfg.use_fp8_output == 1
                and self.acc_o_tensor is None
                and cfg.use_cluster_reduction != 1
            ):
                final_vals = cutlass.Array(Float32, 8, space=cutlass.AddressSpace.rmem)
                packed_o = cutlass.Array(Int32, 2, space=cutlass.AddressSpace.rmem)
            else:
                packed_o = cutlass.Array(Int32, 4, space=cutlass.AddressSpace.rmem)
            for pair_idx in cutlass.range_constexpr(4):
                reg_base = pair_idx * 2
                final_pair = fmul2(
                    (final_scale0[0], final_scale0[0]),
                    (o_loaded[reg_base], o_loaded[reg_base + 1]),
                )
                if cutlass.const_expr(
                    cfg.use_fp8_output == 1
                    and self.acc_o_tensor is None
                    and cfg.use_cluster_reduction != 1
                ):
                    final_vals[reg_base] = final_pair[0]
                    final_vals[reg_base + 1] = final_pair[1]
                else:
                    packed_o[pair_idx] = pack_float2_to_bf16(
                        final_pair[0], final_pair[1]
                    )
            if cutlass.const_expr(
                cfg.use_fp8_output == 1
                and self.acc_o_tensor is None
                and cfg.use_cluster_reduction != 1
            ):
                packed_o[0] = pack_float4_to_fp8_e4m3(
                    final_vals[0],
                    final_vals[1],
                    final_vals[2],
                    final_vals[3],
                )
                packed_o[1] = pack_float4_to_fp8_e4m3(
                    final_vals[4],
                    final_vals[5],
                    final_vals[6],
                    final_vals[7],
                )
            store_col_offset = (
                head_dim_offset
                + Int32(v_stage_idx * cfg.head_dim_per_stage_v)
                + half_warp_col_offset
                + Int32(chunk_idx)
            )
            if global_head_idx < Int32(cfg.num_heads_q) and valid_output_row:
                if cutlass.const_expr(self.acc_o_tensor is not None):
                    split_kv = Int32(cfg.num_ctas_per_seq_kv)
                    base_elem_offset = (
                        global_head_idx * Int32(split_kv * cfg.head_dim_v)
                        + cta_idx_kv * Int32(cfg.head_dim_v)
                        + store_col_offset
                        + cta_idx_q * Int32(cfg.num_heads_q * split_kv * cfg.head_dim_v)
                        + batch_idx
                        * Int32(
                            cfg.seq_len_q * cfg.num_heads_q * split_kv * cfg.head_dim_v
                        )
                    )
                    base_ptr = self.acc_o_tensor.iterator.raw_ptr().toint(Int64)
                    element_bytes = cfg.partial_o_dtype_bytes
                else:
                    output_query_row = public_query_flat_row(
                        cfg,
                        storage_flat_query_row,
                        batch_idx,
                        self.cu_seqlens_q,
                    )
                    base_elem_offset = Int64(output_query_row) * Int64(
                        cfg.head_dim_v
                    ) + Int64(store_col_offset)
                    base_ptr = self.o_tensor.iterator.raw_ptr().toint(Int64)
                    element_bytes = cfg.o_dtype_bytes
                dst_ptr = cutlass.inttoptr(
                    base_ptr + Int64(base_elem_offset) * Int64(element_bytes),
                    mem_space=1,
                    dtype=Int32,
                )
                if cutlass.const_expr(
                    cfg.use_fp8_output == 1
                    and self.acc_o_tensor is None
                    and cfg.use_cluster_reduction != 1
                ):
                    dst_ptr.store(packed_o.load(0, 2), alignment=8)
                else:
                    dst_ptr.store(packed_o.load(0, 4), alignment=16)

    @cute.jit
    def _stage_tile32_tail_o_to_smem(
        self,
        task_cache,
        *,
        base_addr0,
        base_addr1,
        final_scale0,
        final_scale1,
    ):
        """Stage the tileSizeQ=32 BF16/partial O tail path through SMEM."""
        cfg = self.cfg
        shape = TCGEN05_16X256B_SHAPE
        q_repeats = num_o_repeats(cfg)
        o_reg_pair_count = num_o_reg_pairs(cfg)
        num_o_tmem_loads = num_o_tmem_loads_per_stage(cfg)
        warp_grp_thread_idx = task_cache[_TASK_CACHE_WARP_GRP_THREAD_IDX]
        warp_idx = task_cache[_TASK_CACHE_WARP_IDX]
        lane_idx = task_cache[_TASK_CACHE_LANE_IDX]

        for chunk_idx in cutlass.range_constexpr(num_o_tmem_loads):
            o0_loaded = prims.tcgen05_ld(
                shape,
                prims.make_tmem_ptr(tcgen05_panel_addr(base_addr0, chunk_idx), Float32),
                num=q_repeats,
            )
            o1_loaded = prims.tcgen05_ld(
                shape,
                prims.make_tmem_ptr(tcgen05_panel_addr(base_addr1, chunk_idx), Float32),
                num=q_repeats,
            )
            prims.tcgen05_wait(kind=prims.Tcgen05Wait.LOAD)
            cute.arch.fence_view_async_tmem_load()
            regs_o = cutlass.Array(
                Int32,
                o_reg_pair_count,
                space=cutlass.AddressSpace.rmem,
            )
            for pair_idx in cutlass.range_constexpr(o_reg_pair_count):
                scale_base = ((pair_idx % (2 * max(cfg.tile_size_q // 8, 1))) // 2) * 2
                reg_base = pair_idx * 2
                final_pair = ffma2(
                    (
                        final_scale0[scale_base],
                        final_scale0[scale_base + 1],
                    ),
                    (
                        o0_loaded[reg_base],
                        o0_loaded[reg_base + 1],
                    ),
                    fmul2(
                        (
                            final_scale1[scale_base],
                            final_scale1[scale_base + 1],
                        ),
                        (
                            o1_loaded[reg_base],
                            o1_loaded[reg_base + 1],
                        ),
                    ),
                )
                regs_o[pair_idx] = pack_float2_to_bf16(final_pair[0], final_pair[1])

            for store_row_block_idx in cutlass.range_constexpr(
                num_o_stsm_row_blocks(cfg)
            ):
                smem_offset_bytes, _, _, _ = o_stage_stsm_and_copy_offsets(
                    cfg,
                    warp_grp_thread_idx,
                    warp_idx,
                    lane_idx,
                    chunk_idx,
                    store_row_block_idx,
                )
                smem_dst = self._smem_o_i32.data_ptr(
                    smem_offset_bytes >> SMEM_WORD_BYTE_SHIFT
                )
                reg_store_offset = store_row_block_idx * TCGEN05_16X256B_REGS_PER_LOAD
                store_regs = (regs_o.data_ptr() + reg_store_offset).load(
                    count=TCGEN05_16X256B_REGS_PER_LOAD,
                    alignment=SMEM_WORD_BYTES,
                )
                prims.stmatrix(
                    smem_dst,
                    store_regs,
                    prims.MMALayout.COL,
                    shape=prims.StoreShape.M8N8,
                )

    @cute.jit
    def _store_fp8_tail_vals_to_smem(self, task_cache, final_vals):
        """Pack normalized FP8 O values and stage them into store SMEM."""
        cfg = self.cfg
        warp_grp_thread_idx = task_cache[_TASK_CACHE_WARP_GRP_THREAD_IDX]
        regs_fp8 = cutlass.Array(
            Int32,
            num_fp8_output_regs(cfg),
            space=cutlass.AddressSpace.rmem,
        )
        for packed_idx in cutlass.range_constexpr(num_fp8_output_regs(cfg)):
            val_base = packed_idx * TCGEN05_16X256B_REGS_PER_LOAD
            regs_fp8[packed_idx] = pack_float4_to_fp8_e4m3(
                final_vals[val_base],
                final_vals[val_base + 1],
                final_vals[val_base + 2],
                final_vals[val_base + 3],
            )
        if cutlass.const_expr(num_fp8_output_regs(cfg) == 2):
            store_transposed_smem8b_x2(
                self._smem_o_i32,
                regs_fp8[0],
                regs_fp8[1],
                warp_grp_thread_idx,
                cfg.tile_size_q,
                cfg.head_dim_per_stage_v,
            )
        else:
            for stsm_chunk_idx in cutlass.range_constexpr(
                num_fp8_output_regs(cfg) // 4
            ):
                reg_base = stsm_chunk_idx * TCGEN05_16X256B_REGS_PER_LOAD
                store_transposed_smem8b_x4(
                    self._smem_o_i32,
                    regs_fp8[reg_base],
                    regs_fp8[reg_base + 1],
                    regs_fp8[reg_base + 2],
                    regs_fp8[reg_base + 3],
                    warp_grp_thread_idx,
                    cfg.tile_size_q,
                    cfg.head_dim_per_stage_v,
                    stsm_idx=stsm_chunk_idx,
                )

    @cute.jit
    def _store_bf16_tail_regs_to_smem(self, task_cache, regs_o):
        """Stage normalized BF16/partial O registers into store SMEM."""
        cfg = self.cfg
        warp_grp_thread_idx = task_cache[_TASK_CACHE_WARP_GRP_THREAD_IDX]
        warp_idx = task_cache[_TASK_CACHE_WARP_IDX]
        lane_idx = task_cache[_TASK_CACHE_LANE_IDX]
        for store_chunk_idx in cutlass.range_constexpr(cfg.o_copy_segments_per_stage):
            stsm_group_idx = store_chunk_idx // num_o_stsm_row_blocks(cfg)
            copy_segment_idx = store_chunk_idx % num_o_stsm_row_blocks(cfg)
            if cutlass.const_expr(cfg.tile_size_q == 8):
                stsm_group_idx = 0
                copy_segment_idx = store_chunk_idx
            smem_offset_bytes, _, _, _ = o_stage_stsm_and_copy_offsets(
                cfg,
                warp_grp_thread_idx,
                warp_idx,
                lane_idx,
                stsm_group_idx,
                copy_segment_idx,
            )
            smem_dst = self._smem_o_i32.data_ptr(
                smem_offset_bytes >> SMEM_WORD_BYTE_SHIFT
            )
            reg_store_offset = store_chunk_idx * TCGEN05_16X256B_REGS_PER_LOAD
            store_regs = (regs_o.data_ptr() + reg_store_offset).load(
                count=TCGEN05_16X256B_REGS_PER_LOAD,
                alignment=SMEM_WORD_BYTES,
            )
            prims.stmatrix(
                smem_dst,
                store_regs,
                prims.MMALayout.COL,
                shape=prims.StoreShape.M8N8,
            )

    @cute.jit
    def _stage_generic_tail_o_to_smem(
        self,
        task_cache,
        *,
        base_addr0,
        base_addr1,
        final_scale0,
        final_scale1,
    ):
        """Stage the generic swaps-MMA-AB O tail path through SMEM."""
        cfg = self.cfg
        shape = TCGEN05_16X256B_SHAPE
        q_repeats = num_o_repeats(cfg)
        o_reg_pair_count = num_o_reg_pairs(cfg)
        num_o_tmem_loads = num_o_tmem_loads_per_stage(cfg)

        if cutlass.const_expr(
            cfg.use_fp8_output == 1
            and self.acc_o_tensor is None
            and cfg.use_cluster_reduction != 1
        ):
            final_vals = cutlass.Array(
                Float32,
                o_reg_pair_count * num_o_tmem_loads * 2,
                space=cutlass.AddressSpace.rmem,
            )
        else:
            regs_o = cutlass.Array(
                Int32,
                o_reg_pair_count * num_o_tmem_loads,
                space=cutlass.AddressSpace.rmem,
            )
        for chunk_idx in cutlass.range_constexpr(num_o_tmem_loads):
            if cutlass.const_expr(cfg.tile_size_q == 8):
                o0_loaded_lo = prims.tcgen05_ld(
                    shape,
                    prims.make_tmem_ptr(
                        tcgen05_panel_addr(base_addr0, chunk_idx), Float32
                    ),
                    num=1,
                )
                o1_loaded_lo = prims.tcgen05_ld(
                    shape,
                    prims.make_tmem_ptr(
                        tcgen05_panel_addr(base_addr1, chunk_idx), Float32
                    ),
                    num=1,
                )
                o0_loaded_hi = prims.tcgen05_ld(
                    shape,
                    prims.make_tmem_ptr(
                        tcgen05_second_panel_addr(
                            tcgen05_panel_addr(base_addr0, chunk_idx)
                        ),
                        Float32,
                    ),
                    num=1,
                )
                o1_loaded_hi = prims.tcgen05_ld(
                    shape,
                    prims.make_tmem_ptr(
                        tcgen05_second_panel_addr(
                            tcgen05_panel_addr(base_addr1, chunk_idx)
                        ),
                        Float32,
                    ),
                    num=1,
                )
            else:
                o0_loaded = prims.tcgen05_ld(
                    shape,
                    prims.make_tmem_ptr(
                        tcgen05_panel_addr(base_addr0, chunk_idx), Float32
                    ),
                    num=q_repeats,
                )
                o1_loaded = prims.tcgen05_ld(
                    shape,
                    prims.make_tmem_ptr(
                        tcgen05_panel_addr(base_addr1, chunk_idx), Float32
                    ),
                    num=q_repeats,
                )
            prims.tcgen05_wait(kind=prims.Tcgen05Wait.LOAD)
            cute.arch.fence_view_async_tmem_load()
            for pair_idx in cutlass.range_constexpr(o_reg_pair_count):
                scale_base = ((pair_idx % (2 * max(cfg.tile_size_q // 8, 1))) // 2) * 2
                if cutlass.const_expr(cfg.tile_size_q == 8):
                    reg_base = (pair_idx % 2) * 2
                    o0_pair = (
                        o0_loaded_lo[reg_base],
                        o0_loaded_lo[reg_base + 1],
                    )
                    o1_pair = (
                        o1_loaded_lo[reg_base],
                        o1_loaded_lo[reg_base + 1],
                    )
                    if cutlass.const_expr(pair_idx >= 2):
                        o0_pair = (
                            o0_loaded_hi[reg_base],
                            o0_loaded_hi[reg_base + 1],
                        )
                        o1_pair = (
                            o1_loaded_hi[reg_base],
                            o1_loaded_hi[reg_base + 1],
                        )
                else:
                    reg_base = pair_idx * 2
                    o0_pair = (
                        o0_loaded[reg_base],
                        o0_loaded[reg_base + 1],
                    )
                    o1_pair = (
                        o1_loaded[reg_base],
                        o1_loaded[reg_base + 1],
                    )
                final_pair = ffma2(
                    (
                        final_scale0[scale_base],
                        final_scale0[scale_base + 1],
                    ),
                    o0_pair,
                    fmul2(
                        (
                            final_scale1[scale_base],
                            final_scale1[scale_base + 1],
                        ),
                        o1_pair,
                    ),
                )
                if cutlass.const_expr(
                    cfg.use_fp8_output == 1
                    and self.acc_o_tensor is None
                    and cfg.use_cluster_reduction != 1
                ):
                    final_base = chunk_idx * o_reg_pair_count * 2 + pair_idx * 2
                    final_vals[final_base] = final_pair[0]
                    final_vals[final_base + 1] = final_pair[1]
                else:
                    regs_o[chunk_idx * o_reg_pair_count + pair_idx] = (
                        pack_float2_to_bf16(final_pair[0], final_pair[1])
                    )

        if cutlass.const_expr(
            cfg.use_fp8_output == 1
            and self.acc_o_tensor is None
            and cfg.use_cluster_reduction != 1
        ):
            self._store_fp8_tail_vals_to_smem(task_cache, final_vals)
        else:
            self._store_bf16_tail_regs_to_smem(task_cache, regs_o)

    @cute.jit
    def _stage_swaps_tail_o_to_smem(
        self,
        task_cache,
        *,
        base_addr0,
        base_addr1,
        final_scale0,
        final_scale1,
    ):
        """Select the swaps-MMA-AB O staging path for one V stage."""
        if cutlass.const_expr(
            self.cfg.tile_size_q == 32
            and (
                self.cfg.use_fp8_output != 1
                or self.acc_o_tensor is not None
                or self.cfg.use_cluster_reduction == 1
            )
        ):
            self._stage_tile32_tail_o_to_smem(
                task_cache,
                base_addr0=base_addr0,
                base_addr1=base_addr1,
                final_scale0=final_scale0,
                final_scale1=final_scale1,
            )
        else:
            self._stage_generic_tail_o_to_smem(
                task_cache,
                base_addr0=base_addr0,
                base_addr1=base_addr1,
                final_scale0=final_scale0,
                final_scale1=final_scale1,
            )

    @cute.jit
    def _normalize_and_store_tail_o(
        self,
        stage_info: StageInfo,
        *,
        new_max_arr,
        sum_arr,
        inst0_new_max_arr,
        inst0_sum_arr,
        inst1_new_max_arr,
        inst1_sum_arr,
        tail_o_stage_idx_0,
        tail_o_stage_idx_1,
    ):
        """Normalize final O and store O/LSE or split-KV partials."""
        if cutlass.const_expr(self.inst_id == 1):
            cfg = self.cfg
            task_cache = decode_gen_task_cache(stage_info)
            tmem_row_base = task_cache[_TASK_CACHE_TMEM_BASE_OFFSET]
            cta_idx_head_dim_v = cta_idx_head_dim_v_for_stage(
                self.cta_idx_head_dim_v, stage_info
            )
            head_dim_offset = head_dim_cta_offset_v(cfg, cta_idx_head_dim_v)
            final_max, reduced_sum, final_scale0, final_scale1 = (
                self._compute_tail_softmax_scales(
                    task_cache,
                    new_max_arr=new_max_arr,
                    sum_arr=sum_arr,
                    inst0_new_max_arr=inst0_new_max_arr,
                    inst0_sum_arr=inst0_sum_arr,
                    inst1_new_max_arr=inst1_new_max_arr,
                    inst1_sum_arr=inst1_sum_arr,
                )
            )
            self._store_lse_to_gmem(stage_info, task_cache, final_max, reduced_sum)

            for v_stage_idx in cutlass.range_constexpr(cfg.v_head_dim_stages):
                base_addr0 = (
                    tmem_row_base
                    + self._o_base_col()
                    + o_stage_tmem_col_offset(cfg, tail_o_stage_idx_0, v_stage_idx)
                )
                base_addr1 = (
                    tmem_row_base
                    + self._o_base_col()
                    + o_stage_tmem_col_offset(cfg, tail_o_stage_idx_1, v_stage_idx)
                )
                if cutlass.const_expr(cfg.kernel_variant == "keeps_mma_ab"):
                    self._store_keeps_tail_o_stage(
                        stage_info,
                        task_cache,
                        base_addr0=base_addr0,
                        final_scale0=final_scale0,
                        v_stage_idx=v_stage_idx,
                        head_dim_offset=head_dim_offset,
                    )
                else:
                    self._stage_swaps_tail_o_to_smem(
                        task_cache,
                        base_addr0=base_addr0,
                        base_addr1=base_addr1,
                        final_scale0=final_scale0,
                        final_scale1=final_scale1,
                    )
                    cute.arch.fence_view_async_shared()
                    prims.barrier_cta_sync(
                        barrier_id=self.store_barrier_id,
                        thread_count=WARPGROUP_THREADS,
                    )
                    self._store_o_slice_to_gmem(
                        stage_info, task_cache, v_stage_idx, head_dim_offset
                    )
                    prims.barrier_cta_sync(
                        barrier_id=self.store_barrier_id,
                        thread_count=WARPGROUP_THREADS,
                    )
            self._reduce_cluster_partials_and_store(
                stage_info, task_cache, head_dim_offset
            )
