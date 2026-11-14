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

from ...helpers.constants import LN_2, LOG2_E

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int8, Int32, Int64, Uint32
from cutlass.experimental import primitives as cprims
from cutlass.experimental.task_scheduling.enums import WorkAttr
from cutlass.experimental.task_scheduling.memory import SmemAllocation
from cutlass.experimental.task_scheduling.resources import (
    StageInfo,
    producer_work,
)

from ...helpers.math import final_lse
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
    neg_max_f32,
    ceil_div,
    fadd2,
    ffma2,
    fmul2,
    output_dtype,
    pack_float2_to_bf16,
    partial_output_dtype,
)
from ...helpers.ops import (
    fp8_p_scale,
    fp8_p_scale_rcp,
    pack_float4_to_fp8_e4m3,
    store_transposed_smem8b_x2,
    store_transposed_smem8b_x4,
    tcgen05_ld_16x32bx2_f32,
    tcgen05_panel_addr,
    tcgen05_second_panel_addr,
    tcgen05_st_16x32bx2_f32,
    vector_from_scalars,
)
from ...helpers.query import (
    flat_query_row_state,
    public_query_flat_row,
    split_o_element_offset,
)
from ...helpers.tile import (
    batch_idx_for_stage_cfg,
    cta_idx_head_dim_v_for_stage,
    cta_idx_kv_for_stage,
    cta_idx_q_for_stage,
    head_idx_for_stage,
    global_kv_tile_idx,
    runtime_seq_len_kv_from_task_cache,
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
    attention_sinks: object = None
    output_quant: cutlass.Constexpr = None
    output_quant_args: object = None
    o_tensor: object = None
    lse_tensor: object = None
    acc_o_tensor: object = None
    acc_lse_tensor: object = None
    cache_seqs: object = None
    page_offsets: object = None
    head_idx: object = None
    batch_idx: object = None
    cta_idx_q: object = None
    cta_idx_kv: object = None
    cta_idx_head_dim_v: object = None
    store_barrier_id: cutlass.Constexpr[int] = 6
    sum_barrier_id: cutlass.Constexpr[int] = 7
    # WS means tcgen05.mma.ws (M64 warp-specialized PV). This scratch either
    # joins the two Layout-E N-column sums (128 FP32 values, 512 bytes) or
    # relays Layout-F row factors to WS consumers (64 FP32 values, 256 bytes).
    _ws_scale_alloc: cutlass.Constexpr[Optional[SmemAllocation]] = None
    _sum_alloc: cutlass.Constexpr[Optional[SmemAllocation]] = None
    _cluster_reduction_alloc: cutlass.Constexpr[Optional[SmemAllocation]] = None
    _cluster_reduction_barrier_alloc: cutlass.Constexpr[Optional[SmemAllocation]] = None
    _smem_o: object = None
    _smem_o_i32: object = None
    _sum_scratch: object = None
    _cluster_reduction_smem: object = None
    _cluster_reduction_barrier: object = None
    _o_coalesce_alloc: object = None
    _smem_o_coalesce: object = None
    _smem_o_coalesce_bytes: object = None
    _quant_scratch_alloc: object = None
    _quant_scratch: object = None
    # Output-quantizer state carried from a work tile's start to its tail.
    _tail_quant_state: object = None
    # TMA descriptor of the public output, used by the WS TMA output stores.
    tma_o_desc: object = None
    _tail_sink: object = None
    # Cluster split finisher: the first CTA's buffer that every split's partials are pushed
    # into (cfg.cluster_split_push_bytes) and its transaction barrier.
    _split_push_alloc: cutlass.Constexpr[Optional[SmemAllocation]] = None
    _split_push_barrier_alloc: cutlass.Constexpr[Optional[SmemAllocation]] = None
    _split_push_smem: object = None
    _split_push_barrier: object = None

    def _prefetches_tail_sink(self):
        # Keep-AB fused epilogues read one sink per row-owning lane: one per
        # half-warp lane, or every lane of WS native-E (_keeps_tail_logical_head).
        return (
            self.inst_id == 1
            and self.cfg.kernel_variant == "keeps_mma_ab"
            and self.cfg.fuse_sparse_epilogue
            and self.attention_sinks is not None
        )

    def _prefetches_output_quant(self):
        # An output quantizer may load per-row inputs (such as a RoPE table)
        # when a work tile starts and publish them through a per-warp shared
        # slot before its tail needs them.
        return (
            self._ws_tma_quant_output()
            and hasattr(self.output_quant, "prefetch")
            and hasattr(self.output_quant, "stage_rotary")
        )

    def _keeps_direct_output(self):
        return (
            self.cfg.kernel_variant == "keeps_mma_ab"
            and (self.cfg.num_tokens_per_page == 1 and not self.cfg.sparse_d576_reload)
            and self.cfg.use_cluster_reduction != 1
        )

    def _coalesced_keeps_output(self):
        # Final BF16 rows only; partials, FP8 and quantized outputs keep the
        # direct per-lane stores.
        return (
            self.inst_id == 1
            and self._keeps_direct_output()
            and self.cfg.coalesce_keeps_output
            and self.output_quant is None
            and self.acc_o_tensor is None
            and self.cfg.use_fp8_output != 1
        )

    def _o_coalesce_bytes(self):
        # One D stage of sixteen rows per correction warp.
        return (
            self.cfg.correction_num_warps
            * 16
            * self.cfg.head_dim_per_stage_v
            * self.cfg.o_dtype_bytes
        )

    def _ws_coalesced_output(self):
        # Final BF16 rows of the WS Layout-E epilogue, transposed through SMEM.
        return (
            self.inst_id == 1
            and self.cfg.ws_pv
            and self.cfg.ws_native_e
            and self.cfg.ws_coalesce_cols > 0
            and self._keeps_direct_output()
            and self.output_quant is None
            and self.acc_o_tensor is None
            and self.cfg.use_fp8_output != 1
        )

    def _ws_tma_output(self):
        return self._ws_coalesced_output() and self.cfg.ws_tma_output

    def _ws_tma_quant_output(self):
        # The output quantizer stages each lane's 128 FP8 channels of a WS half
        # as one 128-byte SW128 row; the warp's 32 rows leave as one TMA box.
        return (
            self.inst_id == 1
            and self.cfg.ws_pv
            and self.cfg.ws_native_e
            and self.cfg.ws_tma_output
            and self.cfg.ws_coalesce_cols == 64
            and self._keeps_direct_output()
            and self.output_quant is not None
            and self.acc_o_tensor is None
        )

    def _ws_staged_output(self):
        return self._ws_coalesced_output() or self._ws_tma_quant_output()

    def _ws_o_coalesce_bytes(self):
        # One chunk of 32 rows per correction warp; TMA output alternates two,
        # so one can be written while the other is still being read. A chunk
        # row is 128 bytes: 64 BF16 or 128 E4M3 channels.
        return (
            (2 if self._ws_tma_output() or self._ws_tma_quant_output() else 1)
            * self.cfg.correction_num_warps
            * 32
            * self.cfg.ws_coalesce_cols
            * self.cfg.o_dtype_bytes
        )

    def get_smem_requirements(self):
        """Return correction/output SMEM scratch and optional cluster buffers."""
        allocs = list(self._correction_smem_requirements())
        if self.cfg.cluster_split_finisher:
            if self._split_push_alloc is None:
                self._split_push_alloc = SmemAllocation(
                    name=f"{self.name}_splitPush",
                    size_bytes=self.cfg.cluster_split_push_bytes,
                    alignment=16,
                )
                self._split_push_barrier_alloc = SmemAllocation(
                    name=f"{self.name}_splitPushBarrier", size_bytes=8, alignment=8
                )
            allocs += [self._split_push_alloc, self._split_push_barrier_alloc]
        return allocs

    def _correction_smem_requirements(self):
        if self.cfg.ws_pv and self.inst_id == 1:
            if self._ws_scale_alloc is None:
                self._ws_scale_alloc = SmemAllocation(
                    name=f"{self.name}_ws_scale",
                    size_bytes=512 if self.cfg.ws_native_e else 256,
                    alignment=16,
                )
            if self._ws_staged_output():
                if self._o_coalesce_alloc is None:
                    self._o_coalesce_alloc = SmemAllocation(
                        name=f"{self.name}_wsOCoalesce",
                        size_bytes=self._ws_o_coalesce_bytes(),
                        # A TMA SW128 source starts on a 1 KiB swizzle atom.
                        alignment=1024 if self.cfg.ws_tma_output else 128,
                    )
                allocs = [self._ws_scale_alloc, self._o_coalesce_alloc]
                if self._prefetches_output_quant():
                    if self._quant_scratch_alloc is None:
                        self._quant_scratch_alloc = SmemAllocation(
                            name=f"{self.name}_quantScratch",
                            size_bytes=self.cfg.correction_num_warps
                            * self.output_quant.rotary_scratch_bytes,
                            alignment=16,
                        )
                    allocs.append(self._quant_scratch_alloc)
                return allocs
            return [self._ws_scale_alloc]

        if self._coalesced_keeps_output():
            if self._o_coalesce_alloc is None:
                self._o_coalesce_alloc = SmemAllocation(
                    name=f"{self.name}_oCoalesce",
                    size_bytes=self._o_coalesce_bytes(),
                    alignment=128,
                )
            return [self._o_coalesce_alloc]
        if self.inst_id != 1 or self._keeps_direct_output():
            # Keeps-AB normalizes through warp shuffles and stores from TMEM
            # directly to GMEM. Its output and sum scratch are never used.
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
        if cutlass.const_expr(self._ws_staged_output()):
            self._smem_o_coalesce = smem_array(
                context, self._o_coalesce_alloc, Int32, self._ws_o_coalesce_bytes() // 4
            )
            if cutlass.const_expr(self._ws_tma_output() or self._ws_tma_quant_output()):
                self._smem_o_coalesce_bytes = smem_array(
                    context, self._o_coalesce_alloc, Int8, self._ws_o_coalesce_bytes()
                )
            if cutlass.const_expr(self._prefetches_output_quant()):
                self._quant_scratch = smem_array(
                    context,
                    self._quant_scratch_alloc,
                    Int8,
                    self.cfg.correction_num_warps * self.output_quant.rotary_scratch_bytes,
                )
            return
        if cutlass.const_expr(self._coalesced_keeps_output()):
            self._smem_o_coalesce = smem_array(
                context, self._o_coalesce_alloc, Int32, self._o_coalesce_bytes() // 4
            )
            return
        if cutlass.const_expr(self.inst_id != 1 or self._keeps_direct_output()):
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
        """Bind cluster correction SMEM and initialize its transaction barriers.

        The kernel entry calls this before the cluster arrival, so every barrier a peer
        completes (the split push barrier, the reduction slice barriers) is armed before any
        peer can publish into it.
        """

        if cutlass.const_expr(self._split_push_alloc is not None):
            self._init_split_push(context)
        if cutlass.const_expr(self.cfg.use_cluster_reduction == 1):
            self._init_store_state_from_context(context)

    @producer_work(work_attrs=WorkAttr.AUXILIARY)
    @cute.jit
    def init_store_state(self, stage_info: StageInfo) -> None:
        """Initialize correction/store task-local state for captured schedules."""
        # Producer aux work initializes the correction task's view of TMEM O.
        # Runtime SMEM views are initialized once in initialize_runtime_state_internal().
        self._init_tmem_state(stage_info)
        if cutlass.const_expr(self._prefetches_tail_sink()):
            # Persistent work-tile loops carry the per-tile sink; define it
            # before the loop.
            self._tail_sink = Float32(-Float32.inf)
        if cutlass.const_expr(self._prefetches_output_quant()):
            self._tail_quant_state = Int32(0)

    @producer_work(work_attrs=WorkAttr.AUXILIARY)
    @cute.jit
    def prefetch_tail_inputs(self, stage_info: StageInfo) -> None:
        """Load this work tile's row sinks before any wait for O.

        The tail otherwise issues this global load after the final PV, where
        its full L2 round trip precedes every output store.
        """
        if cutlass.const_expr(self._prefetches_tail_sink()):
            task_cache = decode_gen_task_cache(stage_info)
            logical_head = self._keeps_tail_logical_head(stage_info, task_cache)
            sink = Float32(-Float32.inf)
            if logical_head < self.cfg.logical_num_heads_q:
                sink = Float32(self.attention_sinks[logical_head])
            self._tail_sink = sink
        if cutlass.const_expr(self._prefetches_output_quant()):
            task_cache = decode_gen_task_cache(stage_info)
            self._tail_quant_state = self.output_quant.prefetch(
                self.output_quant_args, self._ws_lane_output_row(stage_info, task_cache), 0
            )

    @cute.jit
    def _keeps_tail_logical_head(self, stage_info: StageInfo, task_cache):
        """Logical query head of this lane's Keep-AB row.

        One per half-warp lane, or under WS Layout E one per lane: lane l of
        correction warp w owns row (w % 2) * 32 + l.
        """
        cfg = self.cfg
        warp_idx = task_cache[_TASK_CACHE_WARP_IDX]
        lane_idx = task_cache[_TASK_CACHE_LANE_IDX]
        local_row = warp_idx * Int32(16) + (lane_idx & Int32(15))
        if cutlass.const_expr(cfg.ws_native_e):
            local_row = (warp_idx % Int32(2)) * Int32(32) + lane_idx
        return (
            cta_idx_q_for_stage(self.cta_idx_q, stage_info) * cfg.tile_size_q
            + head_idx_for_stage(self.head_idx, cfg, stage_info)
            + local_row
        )

    @cute.jit
    def _ws_lane_output_row(self, stage_info: StageInfo, task_cache):
        """Public flat output row (token * heads + head) of this WS Layout-E lane."""
        cfg = self.cfg
        warp_idx = task_cache[_TASK_CACHE_WARP_IDX]
        lane_idx = task_cache[_TASK_CACHE_LANE_IDX]
        batch_idx = batch_idx_for_stage_cfg(self.batch_idx, cfg, stage_info)
        storage_flat_query_row, _, _, _, _ = flat_query_row_state(
            head_idx_for_stage(self.head_idx, cfg, stage_info)
            + (warp_idx % Int32(2)) * Int32(32)
            + lane_idx,
            cta_idx_q_for_stage(self.cta_idx_q, stage_info),
            cfg.tile_size_q,
            cfg.logical_num_heads_q,
            cfg.logical_seq_len_q,
            cu_seqlens_q=self.cu_seqlens_q,
            batch_idx=batch_idx,
        )
        return public_query_flat_row(
            cfg, storage_flat_query_row, batch_idx, self.cu_seqlens_q
        )

    @cute.jit
    def _o_base_col(self):
        """Return the first TMEM column owned by the O accumulator."""

        if cutlass.const_expr(self.cfg.single_kv_pipe):
            return Int32(2 * self.cfg.tmem_s_cols)
        return Int32(2 * self.cfg.tmem_s_cols + 2 * self.cfg.tmem_stats_cols)

    @cute.jit
    def _init_split_push(self, context) -> None:
        """Bind the split push buffer; the cluster's first CTA arms its transaction barrier.

        The barrier expects every split's partial O and LSE of each covered row. It runs at
        the kernel entry (``create_cluster_function_variables``), before the cluster arrival
        that every pushing thread waits on: a push that reached the barrier before its
        initialization would be lost and leave the first CTA waiting.
        """
        cfg = self.cfg
        self._split_push_smem = smem_array(
            context, self._split_push_alloc, Int8, cfg.cluster_split_push_bytes
        )
        self._split_push_barrier = smem_array(context, self._split_push_barrier_alloc, Int64, 1)
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        rank = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        if warp_idx == Int32(cfg.correction_warp_idx) and rank == Int32(0):
            with cute.arch.elect_one():
                prims.mbarrier_init(self._split_push_barrier, 1)
            prims.bar_warp_sync(cute.arch.FULL_MASK)
            with cute.arch.elect_one():
                prims.mbarrier_arrive_expect_tx(
                    self._split_push_barrier,
                    Int32(
                        cfg.num_ctas_per_seq_kv
                        * cfg.cluster_split_rows
                        * (cfg.head_dim_per_cta_v * cfg.partial_o_dtype_bytes + 4)
                    ),
                )

    @cute.jit
    def _push_split_bytes(self, byte_offset, words: tuple) -> None:
        """Store one or four words at ``byte_offset`` of the first cluster CTA's push buffer."""
        remote = prims.mapa(
            cutlass.inttoptr(
                self._split_push_smem.data_ptr().toint(Int64) + Int64(byte_offset),
                mem_space=3,
                dtype=Int32,
            ),
            Int32(0),
        )
        barrier = prims.mapa(self._split_push_barrier, Int32(0))
        if cutlass.const_expr(len(words) == 4):
            cute.arch.inline_ptx(
                "st.async.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 "
                "[{$r0}], {{$r1}, {$r2}, {$r3}, {$r4}}, [{$r5}];",
                read_only_args=[remote.ir_value(), *words, barrier.ir_value()],
            )
        else:
            cute.arch.inline_ptx(
                "st.async.shared::cluster.mbarrier::complete_tx::bytes.b32 "
                "[{$r0}], {$r1}, [{$r2}];",
                read_only_args=[remote.ir_value(), words[0], barrier.ir_value()],
            )

    @cute.jit
    def push_split_o(self, split, row, column, words: tuple) -> None:
        """Push 8 BF16 partial-O values of ``row`` from ``column`` of this CTA's V slice."""
        cfg = self.cfg
        offset = (
            (split * Int32(cfg.cluster_split_rows) + row) * Int32(cfg.head_dim_per_cta_v)
            + column
        ) * Int32(cfg.partial_o_dtype_bytes)
        self._push_split_bytes(offset, words)

    @cute.jit
    def push_split_lse(self, split, row, lse_val) -> None:
        """Push one row's partial LSE (log2 units, as the reducer reads it)."""
        cfg = self.cfg
        offset = Int32(
            cfg.num_ctas_per_seq_kv
            * cfg.cluster_split_rows
            * cfg.head_dim_per_cta_v
            * cfg.partial_o_dtype_bytes
        ) + (split * Int32(cfg.cluster_split_rows) + row) * Int32(4)
        self._push_split_bytes(offset, (lse_val.bitcast(Int32),))

    @cute.jit
    def push_neutral_split(self, split) -> None:
        """Push ``(O = 0, LSE = -inf)`` for every covered row of a split without a task graph."""
        cfg = self.cfg
        thread_idx, _, _ = cute.arch.thread_idx()
        vectors_per_row = cfg.head_dim_per_cta_v // 8
        vectors = cfg.cluster_split_rows * vectors_per_row
        for iteration in cutlass.range_constexpr(ceil_div(vectors, cfg.threads_per_cta)):
            vector = thread_idx + Int32(iteration * cfg.threads_per_cta)
            if vector < Int32(vectors):
                row = vector // Int32(vectors_per_row)
                column = (vector - row * Int32(vectors_per_row)) * Int32(8)
                self.push_split_o(split, row, column, (Int32(0), Int32(0), Int32(0), Int32(0)))
        if thread_idx < Int32(cfg.cluster_split_rows):
            self.push_split_lse(split, thread_idx, Float32(-Float32.inf))

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
                # Init and arrive can lower to different elected lanes. Order
                # the warp's initialization before registering transactions.
                prims.bar_warp_sync(cute.arch.FULL_MASK)
                with cute.arch.elect_one():
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
                    flat_query_row_state(
                        global_head_idx,
                        cta_idx_q,
                        cfg.tile_size_q,
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
                        base_elem_offset = split_o_element_offset(
                            cfg,
                            batch_idx,
                            cta_idx_q,
                            global_head_idx,
                            cta_idx_kv,
                            head_dim_offset
                            + Int32(v_stage_idx * cfg.head_dim_per_stage_v),
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
        if cutlass.const_expr(
            self.lse_tensor is None
            and self.acc_lse_tensor is None
            and self.cfg.use_cluster_reduction != 1
        ):
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
            if cutlass.const_expr(self.cfg.ws_native_e):
                should_store_lse = warp_idx < Int32(2)
                local_row_idx = (warp_idx % Int32(2)) * Int32(32) + lane_idx
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
        if cutlass.const_expr(
            (self.cfg.num_tokens_per_page == 1 and not self.cfg.sparse_d576_reload)
            and self.cfg.use_cluster_reduction != 1
            and not self.cfg.cluster_split_finisher
        ):
            should_store_lse = should_store_lse and cta_idx_head_dim_v == Int32(0)
        if should_store_lse:
            # Bound physical rows before publishing statistics so a padded
            # query row cannot write into the next logical head tile.
            if local_row_idx < Int32(self.cfg.tile_size_q) and head_idx < Int32(
                self.cfg.num_heads_q
            ):
                storage_flat_query_row, _, _, _, valid_output_row = (
                    flat_query_row_state(
                        head_idx,
                        cta_idx_q,
                        self.cfg.tile_size_q,
                        self.cfg.logical_num_heads_q,
                        self.cfg.logical_seq_len_q,
                        cu_seqlens_q=self.cu_seqlens_q,
                        batch_idx=batch_idx,
                    )
                )
                # Keep the four swap-M16 statistics in registers. A dynamic
                # Array index here otherwise creates a local-memory round trip.
                selected_sum = Float32(0)
                selected_max = Float32(0)
                for group in cutlass.range_constexpr(
                    num_softmax_scale_groups(self.cfg)
                ):
                    if scale_idx == Int32(group):
                        selected_sum = reduced_sum[group]
                        selected_max = final_max[group]
                lse_sum = selected_sum
                if cutlass.const_expr(self.cfg.is_fp8_qkv()):
                    # P and the online sum use the same FP8 quantization scale.
                    # Undo that scale for the externally visible log-sum-exp.
                    lse_sum = lse_sum * fp8_p_scale_rcp(self.cfg)
                lse_val = (
                    cute.math.log2(lse_sum, fastmath=True)
                    + self.scale_softmax_log2 * selected_max
                )
                if cutlass.const_expr(self.cfg.sparse_direct):
                    if selected_max == neg_max_f32():
                        lse_val = Float32(-Float32.inf)
                if cutlass.const_expr(self.cfg.fuse_sparse_epilogue):
                    count = Float32(selected_max != neg_max_f32())
                    lse_val = (
                        lse_val * Float32(LN_2)
                        if count > Float32(0)
                        else Float32(-Float32.inf)
                    )
                    valid_output_row = valid_output_row and cta_idx_head_dim_v == Int32(
                        0
                    )
                if cutlass.const_expr(self.cfg.use_cluster_reduction == 1):
                    self._store_partial_lse_to_cluster_smem(
                        local_row_idx, cta_idx_kv, lse_val
                    )
                elif cutlass.const_expr(self.cfg.cluster_split_finisher):
                    if valid_output_row:
                        self.push_split_lse(cta_idx_kv, local_row_idx, lse_val)
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
                elif cutlass.const_expr(self.lse_tensor is not None):
                    if valid_output_row:
                        elem_offset = public_query_flat_row(
                            self.cfg,
                            storage_flat_query_row,
                            batch_idx,
                            self.cu_seqlens_q,
                        )
                        (self.lse_tensor.iterator.raw_ptr() + elem_offset).store(
                            final_lse(lse_val, self.cfg.lse_base_on_e)
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
            is_correction_warp = (warp_idx >= Int32(cfg.correction_warp_idx)) & (
                warp_idx < Int32(cfg.correction_warp_idx + cfg.correction_num_warps)
            )
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
                            ) = flat_query_row_state(
                                global_head_idx,
                                cta_idx_q,
                                cfg.tile_size_q,
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
                            has_finite_mass = (
                                lse_sum != Float32(0.0) and lse_sum == lse_sum
                            )
                            is_fully_masked = lse_sum == Float32(0.0)
                            global_lse = (
                                lse_max + cute.math.log2(lse_sum, fastmath=True)
                                if has_finite_mass
                                else Float32(Float32.inf)
                            )
                            published_lse = (
                                Float32(-Float32.inf) if is_fully_masked else global_lse
                            )

                            sink_scale = Float32(1)
                            if cutlass.const_expr(cfg.fuse_sparse_cluster_epilogue):
                                head = storage_flat_query_row % Int32(
                                    cfg.logical_num_heads_q
                                )
                                sink = Float32(self.attention_sinks[head])
                                sink_scale = Float32(0)
                                if has_finite_mass and sink != Float32(Float32.inf):
                                    delta = sink * Float32(LOG2_E) - global_lse
                                    z = cute.math.exp2(
                                        -cute.math.abs(delta), fastmath=True
                                    )
                                    inverse = cute.math.rcp(Float32(1) + z, approx=True)
                                    sink_scale = (
                                        z * inverse if delta > Float32(0) else inverse
                                    )
                                published_lse *= Float32(LN_2)

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
                                    ).store(final_lse(published_lse, cfg.lse_base_on_e))

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

                            if cutlass.const_expr(cfg.fuse_sparse_cluster_epilogue):
                                acc_vec = acc_vec * sink_scale

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
                                    if cutlass.const_expr(
                                        self.output_quant is not None
                                    ):
                                        self.output_quant.store(
                                            self.output_quant_args,
                                            acc_vec,
                                            output_query_row,
                                            head_dim_offset + dim_idx,
                                        )
                                    elif cutlass.const_expr(cfg.use_fp8_output == 1):
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
    def ws_sum_reduce(self, stage_info, value):
        """Join the two N-column partial softmax sums for each Layout-E M64 query row.

        Threads tid and tid^64 represent the same row in the two column halves
        of the WS data-path layout. Their sums must be added once before O
        normalization. The barriers publish scratch and protect it from reuse.
        PTX: https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-data-path-layout-e"""
        scratch = smem_array(stage_info.context, self._ws_scale_alloc, Float32, 128)
        tid = cute.arch.thread_idx()[0] % Int32(128)
        scratch[tid] = value
        prims.barrier_cta_sync(barrier_id=self.store_barrier_id, thread_count=128)
        result = value + scratch[tid ^ Int32(64)]
        prims.barrier_cta_sync(barrier_id=self.store_barrier_id, thread_count=128)
        return result

    @cute.jit
    def ws_row_scale(self, stage_info, scale):
        # Existing stats arrive in Layout-F thread order. One writer per row;
        # every correction thread participates before warp-local skip logic.
        """Remap legacy Layout-F row factors to the Layout-E WS PV consumer order.

        One writer per row publishes a factor; all correction threads read the
        matching row, with barriers before reading and before scratch reuse."""
        scratch = smem_array(stage_info.context, self._ws_scale_alloc, Float32, 64)
        tid = cute.arch.thread_idx()[0] % 128
        if tid % 32 < 16:
            scratch[(tid // 32) * 16 + tid % 16] = scale
        prims.barrier_cta_sync(barrier_id=self.store_barrier_id, thread_count=128)
        result = scratch[tid % 64]
        prims.barrier_cta_sync(barrier_id=self.store_barrier_id, thread_count=128)
        return result

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

        if cutlass.const_expr(cfg.ws_pv):
            for tile in cutlass.range_constexpr(4):
                addr = tmem_row_base + self._o_base_col() + Int32(tile * 64)
                loaded = prims.tcgen05_ld(
                    prims.Tcgen05LdStShape.SHAPE_32X32B,
                    prims.make_tmem_ptr(addr, Float32),
                    num=64,
                )
                prims.tcgen05_wait(kind=prims.Tcgen05Wait.LOAD)
                cute.arch.fence_view_async_tmem_load()
                # Packed FMUL2 pairs: the same RN products in half the instructions.
                pairs = [
                    fmul2(
                        (scale_vals[0], scale_vals[0]),
                        (loaded[i].bitcast(Float32), loaded[i + 1].bitcast(Float32)),
                    )
                    for i in range(0, 64, 2)
                ]
                scaled = vector_from_scalars(
                    tuple(pair[i % 2] for pair in pairs for i in range(2)),
                    Float32,
                )
                prims.tcgen05_st(
                    prims.Tcgen05LdStShape.SHAPE_32X32B,
                    prims.make_tmem_ptr(addr, Float32),
                    scaled,
                )
            return

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
                if cutlass.const_expr(cfg.paired_sparse_correction):
                    for reg_idx in cutlass.range_constexpr(
                        0, cfg.head_dim_per_stage_v // 2, 2
                    ):
                        pair = fmul2(
                            (loaded[reg_idx], loaded[reg_idx + 1]),
                            (scale_vals[0], scale_vals[0]),
                        )
                        scaled[reg_idx], scaled[reg_idx + 1] = pair[0], pair[1]
                else:
                    for reg_idx in cutlass.range_constexpr(
                        cfg.head_dim_per_stage_v // 2
                    ):
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
    def tensor_scale_tile(self, stage_info, local_tile):
        """Map one local pipeline-instance tile to its global split-KV offset and query row."""
        cfg = self.cfg
        batch = batch_idx_for_stage_cfg(self.batch_idx, cfg, stage_info)
        length = runtime_seq_len_kv_from_task_cache(
            cfg,
            decode_gen_task_cache(stage_info),
            cta_idx_q_for_stage(self.cta_idx_q, stage_info),
            self.cu_seqlens_q,
            batch,
        )
        tile = global_kv_tile_idx(
            cfg, local_tile, length, cta_idx_kv_for_stage(self.cta_idx_kv, stage_info)
        )
        return tile * Int32(cfg.tile_size_kv), batch

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

        if cutlass.const_expr(cfg.has_fp8_primary):
            # O is still expressed in the previous source's V units. Apply
            # its descale/current descale together with the normal softmax
            # correction when this stream crosses primary -> extra. This is
            # independent of the denominator update, which uses softmax only.
            local = (stage_info.loop_offset + Int32(1)) * Int32(
                cfg.num_insts_kv
            ) + Int32(self.inst_id % cfg.num_insts_kv)
            current, batch = self.tensor_scale_tile(stage_info, local)
            ratio = self.page_offsets.tensor_scale_ratio(
                current, current - Int32(cfg.num_insts_kv * cfg.tile_size_kv), batch
            )
            for i in cutlass.range_constexpr(num_scale_groups):
                scale_vals[i] *= ratio
            lane_scales_are_identity = lane_scales_are_identity and ratio == Float32(1)

        if cutlass.const_expr(cfg.ws_pv and not cfg.ws_native_e):
            # Legacy non-WS QK produces Layout-F row factors, while WS PV/O
            # uses Layout E. Relay them before the warp-wide identity test;
            # every correction lane must participate in the scratch barriers.
            scale_vals[0] = self.ws_row_scale(stage_info, scale_vals[0])
            lane_scales_are_identity = scale_vals[0] == Float32(1.0)
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
        stage_info,
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
                if cutlass.const_expr(cfg.ws_native_e):
                    final_sum[scale_idx] = self.ws_sum_reduce(
                        stage_info, final_sum[scale_idx]
                    )
                else:
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
                inst0_max = inst0_new_max_arr[scale_idx]
                inst1_sum = (
                    Float32(0.0)
                    if cutlass.const_expr(cfg.one_insts_kv_swap)
                    else inst1_sum_arr[scale_idx]
                )
                inst1_max = (
                    inst0_max
                    if cutlass.const_expr(cfg.one_insts_kv_swap)
                    else inst1_new_max_arr[scale_idx]
                )
                final_max[scale_idx] = cute.math.max(inst0_max, inst1_max, ftz=True)
                exp_scale0[scale_idx] = cute.math.exp2(
                    self.scale_softmax_log2 * (inst0_max - final_max[scale_idx]),
                    fastmath=True,
                )
                exp_scale1[scale_idx] = cute.math.exp2(
                    self.scale_softmax_log2 * (inst1_max - final_max[scale_idx]),
                    fastmath=True,
                )
                if cutlass.const_expr(cfg.one_insts_kv_swap):
                    exp_scale1[scale_idx] = Float32(0.0)
                final_sum[scale_idx] = (
                    inst0_sum * exp_scale0[scale_idx]
                    + inst1_sum * exp_scale1[scale_idx]
                )
                if cutlass.const_expr(not cfg.compact_softmax_stats):
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

        if cutlass.const_expr(cfg.kernel_variant != "keeps_mma_ab"):
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
            if cutlass.const_expr(cfg.fuse_sparse_epilogue):
                # Only row-owning lanes compute sink factors; broadcast to peers.
                norm_lane = (
                    lane_idx & Int32(15)
                    if cfg.kernel_variant == "keeps_mma_ab"
                    else col_group_idx
                )
                norm_lanes = 16 if cfg.kernel_variant == "keeps_mma_ab" else 4
                if cutlass.const_expr(cfg.ws_native_e):
                    norm_lane = lane_idx
                    norm_lanes = 32
                if lane_idx < Int32(norm_lanes):
                    logical_head = (
                        cta_idx_q_for_stage(self.cta_idx_q, stage_info)
                        * cfg.tile_size_q
                        + head_idx_for_stage(self.head_idx, cfg, stage_info)
                        + (
                            (
                                (warp_idx % Int32(2)) * Int32(32) + norm_lane
                                if cfg.ws_native_e
                                else warp_idx * Int32(16) + norm_lane
                            )
                            if cfg.kernel_variant == "keeps_mma_ab"
                            else local_q_head_idx_for_scale(
                                cfg, col_group_idx, scale_idx
                            )
                        )
                    )
                    count = Float32(final_max[scale_idx] != neg_max_f32())
                    norm_scale = Float32(0)
                    if logical_head < cfg.logical_num_heads_q and count > Float32(0):
                        sink = Float32(-Float32.inf)
                        if cutlass.const_expr(self._prefetches_tail_sink()):
                            sink = self._tail_sink
                        elif cutlass.const_expr(self.attention_sinks is not None):
                            sink = Float32(self.attention_sinks[logical_head])
                        if sink == Float32(-Float32.inf):
                            norm_scale = self.output_scale / reduced_sum[scale_idx]
                        elif sink != Float32(Float32.inf):
                            delta = (
                                sink * Float32(LOG2_E)
                                - self.scale_softmax_log2 * final_max[scale_idx]
                            )
                            attention_rescale = cute.math.exp2(
                                -cute.math.max(delta, Float32(0)), approx=True
                            )
                            sink_mass = cute.math.exp2(
                                cute.math.min(delta, Float32(0)), approx=True
                            )
                            if cutlass.const_expr(cfg.is_fp8_qkv()):
                                sink_mass *= fp8_p_scale(cfg)
                            norm_scale = (
                                self.output_scale
                                * attention_rescale
                                * cute.math.rcp(
                                    reduced_sum[scale_idx] * attention_rescale
                                    + sink_mass,
                                    approx=True,
                                )
                            )
                norm_scale = Float32(
                    cprims.shfl_sync(
                        thread_mask=0xFFFFFFFF,
                        val=norm_scale,
                        offset=norm_lane,
                        mask_and_clamp=0x1F,
                        kind=cprims.Shfl.IDX,
                    )
                )
            final_scale0[scale_idx] = norm_scale * exp_scale0[scale_idx]
            final_scale1[scale_idx] = norm_scale * exp_scale1[scale_idx]
        if cutlass.const_expr(cfg.has_fp8_primary):
            last0, batch = self.tensor_scale_tile(
                stage_info, stage_info.loop_end * Int32(cfg.num_insts_kv)
            )
            scale0 = self.page_offsets.tensor_scale(last0, batch)
            scale1 = self.page_offsets.tensor_scale(
                last0 + Int32(cfg.tile_size_kv), batch
            )
            for i in cutlass.range_constexpr(num_scale_groups):
                final_scale0[i] *= scale0
                final_scale1[i] *= scale1
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
        quant_staged=None,
    ):
        """Normalize and directly store one keeps-MMA-AB O stage."""
        cfg = self.cfg
        if cutlass.const_expr(cfg.ws_pv and v_stage_idx % 2 != 0):
            return
        store_scale = final_scale0[0]
        if cutlass.const_expr(cfg.ws_pv and not cfg.ws_native_e):
            store_scale = self.ws_row_scale(stage_info, store_scale)
        batch_idx = batch_idx_for_stage_cfg(self.batch_idx, cfg, stage_info)
        head_base_idx = head_idx_for_stage(self.head_idx, cfg, stage_info)
        cta_idx_q = cta_idx_q_for_stage(self.cta_idx_q, stage_info)
        cta_idx_kv = cta_idx_kv_for_stage(self.cta_idx_kv, stage_info)
        warp_idx = task_cache[_TASK_CACHE_WARP_IDX]
        lane_idx = task_cache[_TASK_CACHE_LANE_IDX]
        # Keeps-MMA-AB stores one head row per half-warp.  The low 16 lanes own
        # the row index; the upper lane bit selects the head-dim half.
        local_row_idx = warp_idx * Int32(16) + (lane_idx & Int32(0xF))
        if cutlass.const_expr(cfg.ws_pv):
            local_row_idx = (warp_idx % Int32(2)) * Int32(32) + lane_idx
        global_head_idx = head_base_idx + local_row_idx
        storage_flat_query_row, _, _, _, valid_output_row = flat_query_row_state(
            global_head_idx,
            cta_idx_q,
            cfg.tile_size_q,
            cfg.logical_num_heads_q,
            cfg.logical_seq_len_q,
            cu_seqlens_q=self.cu_seqlens_q,
            batch_idx=batch_idx,
        )
        half_warp_col_offset = (lane_idx >> Int32(4)) * Int32(
            cfg.head_dim_per_stage_v // 2
        )

        if cutlass.const_expr(cfg.ws_pv):
            half_warp_col_offset = (warp_idx // Int32(2)) * Int32(128)

        if cutlass.const_expr(self._ws_tma_quant_output()):
            self._store_ws_half_quant_via_tma(
                stage_info,
                task_cache,
                base_addr0=base_addr0,
                store_scale=store_scale,
                row=public_query_flat_row(
                    cfg, storage_flat_query_row, batch_idx, self.cu_seqlens_q
                ),
                col_base=head_dim_offset
                + Int32(v_stage_idx * cfg.head_dim_per_stage_v)
                + half_warp_col_offset,
                v_stage_idx=v_stage_idx,
                quant_staged=quant_staged,
            )
            return

        if cutlass.const_expr(self.output_quant is not None):
            # Non-WS Layout F gives each lane half a D128 stage; block128
            # spans lanes separated by16. WS Layout E gives a lane128 values
            # in a D256 PV pass, so block32/block128 are both lane-local.
            # Match the TMEM load shape and extent to the selected layout.
            lane_columns = 128 if cfg.ws_pv else cfg.head_dim_per_stage_v // 2
            width = min(self.output_quant.block_size, lane_columns)
            for stripe in cutlass.range_constexpr(0, lane_columns, width):
                values = cutlass.Array(Float32, width, space=cutlass.AddressSpace.rmem)
                if cutlass.const_expr(cfg.ws_pv):
                    loaded = prims.tcgen05_ld(
                        prims.Tcgen05LdStShape.SHAPE_32X32B,
                        prims.make_tmem_ptr(base_addr0 + Int32(stripe), Float32),
                        num=width,
                    )
                else:
                    loaded = tcgen05_ld_16x32bx2_f32(
                        prims.make_tmem_ptr(base_addr0 + Int32(stripe), Float32),
                        num=width,
                        offset=Int32(cfg.head_dim_per_stage_v // 2),
                    )
                prims.tcgen05_wait(kind=prims.Tcgen05Wait.LOAD)
                cute.arch.fence_view_async_tmem_load()
                for i in cutlass.range_constexpr(width):
                    values[i] = loaded[i] * store_scale
                if global_head_idx < Int32(cfg.num_heads_q) and valid_output_row:
                    row = public_query_flat_row(
                        cfg, storage_flat_query_row, batch_idx, self.cu_seqlens_q
                    )
                    self.output_quant.store(
                        self.output_quant_args,
                        values,
                        row,
                        head_dim_offset
                        + v_stage_idx * cfg.head_dim_per_stage_v
                        + half_warp_col_offset
                        + stripe,
                        lane_stride=1 if cfg.ws_pv else 16,
                        # WS consumes two D128 descriptor slices per PV pass;
                        # its last even stage owns the final64 rotary channels.
                        rotary=v_stage_idx + (2 if cfg.ws_pv else 1)
                        == cfg.v_head_dim_stages,
                    )
            return

        if cutlass.const_expr(self._ws_coalesced_output()):
            self._store_keeps_tail_o_ws_coalesced(
                stage_info,
                task_cache,
                base_addr0=base_addr0,
                store_scale=store_scale,
                col_base=head_dim_offset
                + Int32(v_stage_idx * cfg.head_dim_per_stage_v)
                + half_warp_col_offset,
            )
            return

        if cutlass.const_expr(self._coalesced_keeps_output()):
            self._store_keeps_tail_o_stage_coalesced(
                stage_info,
                task_cache,
                base_addr0=base_addr0,
                final_scale0=final_scale0,
                v_stage_idx=v_stage_idx,
                head_dim_offset=head_dim_offset,
            )
            return

        store_col_base = (
            head_dim_offset
            + Int32(v_stage_idx * cfg.head_dim_per_stage_v)
            + half_warp_col_offset
        )
        if cutlass.const_expr(self.acc_o_tensor is not None):
            base_elem_offset = split_o_element_offset(
                cfg,
                batch_idx,
                cta_idx_q,
                global_head_idx,
                cta_idx_kv,
                store_col_base,
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
            base_elem_offset = Int64(output_query_row) * Int64(cfg.head_dim_v) + Int64(
                store_col_base
            )
            base_ptr = self.o_tensor.iterator.raw_ptr().toint(Int64)
            element_bytes = cfg.o_dtype_bytes
        row_address = base_ptr + base_elem_offset * Int64(element_bytes)
        valid_store = (global_head_idx < Int32(cfg.num_heads_q)) & valid_output_row

        # Compute the row address once, outside the validity guard. Keeping
        # only each store predicated lets subsequent TMEM loads overlap it.
        # ``tcgen05.ld.sync.aligned`` must remain convergent across the warp.
        # Padded query rows still own valid physical TMEM rows, so load and
        # normalize every row and predicate only the externally visible store.
        batched_loads = []
        if cutlass.const_expr(not cfg.ws_pv):
            # Every chunk's TMEM load is in flight before the one wait.
            for chunk_idx in cutlass.range_constexpr(0, cfg.head_dim_per_stage_v // 2, 8):
                batched_loads.append(
                    tcgen05_ld_16x32bx2_f32(
                        prims.make_tmem_ptr(base_addr0 + Int32(chunk_idx), Float32),
                        num=8,
                        offset=Int32(cfg.head_dim_per_stage_v // 2),
                    )
                )
            prims.tcgen05_wait(kind=prims.Tcgen05Wait.LOAD)
            cute.arch.fence_view_async_tmem_load()
        for chunk_idx in cutlass.range_constexpr(
            0,
            128 if cfg.ws_pv else cfg.head_dim_per_stage_v // 2,
            32 if cfg.ws_pv else 8,
        ):
            if cutlass.const_expr(cfg.ws_pv):
                o_loaded = prims.tcgen05_ld(
                    prims.Tcgen05LdStShape.SHAPE_32X32B,
                    prims.make_tmem_ptr(base_addr0 + Int32(chunk_idx), Float32),
                    num=32,
                )
                prims.tcgen05_wait(kind=prims.Tcgen05Wait.LOAD)
                cute.arch.fence_view_async_tmem_load()
            else:
                o_loaded = batched_loads[chunk_idx // 8]
            # Load32 columns per lane, then retain the same aligned16B
            # stores and exact BF16 rounding in four8-value subchunks.
            for store_sub in cutlass.range_constexpr(0, 32 if cfg.ws_pv else 8, 8):
                if cutlass.const_expr(
                    cfg.use_fp8_output == 1
                    and self.acc_o_tensor is None
                    and cfg.use_cluster_reduction != 1
                ):
                    final_vals = cutlass.Array(
                        Float32, 8, space=cutlass.AddressSpace.rmem
                    )
                    packed_o = cutlass.Array(Int32, 2, space=cutlass.AddressSpace.rmem)
                else:
                    packed_o = cutlass.Array(Int32, 4, space=cutlass.AddressSpace.rmem)
                for pair_idx in cutlass.range_constexpr(4):
                    reg_base = pair_idx * 2
                    final_pair = fmul2(
                        (store_scale, store_scale),
                        (
                            o_loaded[store_sub + reg_base],
                            o_loaded[store_sub + reg_base + 1],
                        ),
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
                dst_ptr = cutlass.inttoptr(
                    row_address + Int64((chunk_idx + store_sub) * element_bytes),
                    mem_space=1,
                    dtype=Int32,
                )
                if cutlass.const_expr(
                    cfg.use_fp8_output == 1
                    and self.acc_o_tensor is None
                    and cfg.use_cluster_reduction != 1
                ):
                    if valid_store:
                        dst_ptr.store(packed_o.load(0, 2), alignment=8)
                elif cutlass.const_expr(cfg.cluster_split_finisher):
                    if valid_store:
                        self.push_split_o(
                            cta_idx_kv,
                            local_row_idx,
                            store_col_base - head_dim_offset + Int32(chunk_idx + store_sub),
                            (packed_o[0], packed_o[1], packed_o[2], packed_o[3]),
                        )
                else:
                    if valid_store:
                        dst_ptr.store(packed_o.load(0, 4), alignment=16)

    @cute.jit
    def _store_keeps_tail_o_ws_coalesced(
        self,
        stage_info: StageInfo,
        task_cache,
        *,
        base_addr0,
        store_scale,
        col_base,
    ):
        """Normalize one WS Layout-E O half and store whole row segments via SMEM.

        Lane l of correction warp w owns head row (w % 2) * 32 + l and 128
        columns from ``col_base``, so direct stores touch 32 lines per
        instruction. Each chunk of ``ws_coalesce_cols`` columns is transposed
        through the warp's SMEM slice; each store then writes 512 bytes of
        consecutive rows. Normalization and BF16 rounding match the direct path.
        """
        cfg = self.cfg
        chunk_cols = cfg.ws_coalesce_cols
        row_bytes = chunk_cols * cfg.o_dtype_bytes
        units = row_bytes // 16
        rows_per_store = 32 // units
        stores = 32 // rows_per_store
        batch_idx = batch_idx_for_stage_cfg(self.batch_idx, cfg, stage_info)
        head_base_idx = head_idx_for_stage(self.head_idx, cfg, stage_info)
        cta_idx_q = cta_idx_q_for_stage(self.cta_idx_q, stage_info)
        warp_idx = task_cache[_TASK_CACHE_WARP_IDX]
        lane_idx = task_cache[_TASK_CACHE_LANE_IDX]
        warp_base = self._smem_o_coalesce.data_ptr().toint(Int32) + warp_idx * Int32(
            32 * row_bytes
        )
        row_base = (warp_idx % Int32(2)) * Int32(32)
        unit = lane_idx % Int32(units)

        # A row's 16B units are XOR-rotated by (row >> shift) & (units - 1), so
        # both the row-major writes and the row-segment reads of each
        # quarter-warp phase hit all 32 banks.
        swizzle_shift = int(units != 8)
        swizzle_mask = units - 1

        if cutlass.const_expr(self._ws_tma_output()):
            # The warp's 32 rows form one TMA box over (channel, head, query),
            # whose SW128 shared layout is exactly the swizzled slice above.
            self._store_ws_half_via_tma(
                stage_info,
                task_cache,
                base_addr0=base_addr0,
                store_scale=store_scale,
                col_base=col_base,
            )
            return

        # Row addresses depend only on the staged row, not on the chunk.
        o_base = self.o_tensor.iterator.raw_ptr().toint(Int64)
        row_addresses = []
        row_valid = []
        for store_idx in cutlass.range_constexpr(stores):
            local_row = Int32(store_idx * rows_per_store) + lane_idx // Int32(units)
            global_head_idx = head_base_idx + row_base + local_row
            storage_flat_query_row, _, _, _, valid_output_row = flat_query_row_state(
                global_head_idx,
                cta_idx_q,
                cfg.tile_size_q,
                cfg.logical_num_heads_q,
                cfg.logical_seq_len_q,
                cu_seqlens_q=self.cu_seqlens_q,
                batch_idx=batch_idx,
            )
            output_query_row = public_query_flat_row(
                cfg, storage_flat_query_row, batch_idx, self.cu_seqlens_q
            )
            elem = Int64(output_query_row) * Int64(cfg.head_dim_v) + Int64(
                col_base + unit * Int32(16 // cfg.o_dtype_bytes)
            )
            row_addresses.append(o_base + elem * Int64(cfg.o_dtype_bytes))
            row_valid.append((global_head_idx < Int32(cfg.num_heads_q)) & valid_output_row)

        write_base = warp_base + lane_idx * Int32(row_bytes)
        write_swizzle = (lane_idx >> Int32(swizzle_shift)) & Int32(swizzle_mask)
        for chunk in cutlass.range_constexpr(0, 128, chunk_cols):
            o_loaded = prims.tcgen05_ld(
                prims.Tcgen05LdStShape.SHAPE_32X32B,
                prims.make_tmem_ptr(base_addr0 + Int32(chunk), Float32),
                num=chunk_cols,
            )
            prims.tcgen05_wait(kind=prims.Tcgen05Wait.LOAD)
            cute.arch.fence_view_async_tmem_load()
            for u in cutlass.range_constexpr(units):
                packed_o = cutlass.Array(Int32, 4, space=cutlass.AddressSpace.rmem)
                for pair_idx in cutlass.range_constexpr(4):
                    reg_base = u * 8 + pair_idx * 2
                    final_pair = fmul2(
                        (store_scale, store_scale),
                        (o_loaded[reg_base], o_loaded[reg_base + 1]),
                    )
                    packed_o[pair_idx] = pack_float2_to_bf16(final_pair[0], final_pair[1])
                dst = cutlass.inttoptr(
                    write_base + ((Int32(u) ^ write_swizzle) << Int32(4)),
                    mem_space=3,
                    dtype=Int32,
                )
                dst.store(packed_o.load(0, 4), alignment=16)
            cute.arch.sync_warp()
            staged = []
            for store_idx in cutlass.range_constexpr(stores):
                local_row = Int32(store_idx * rows_per_store) + lane_idx // Int32(units)
                src = cutlass.inttoptr(
                    warp_base
                    + local_row * Int32(row_bytes)
                    + (
                        (
                            unit
                            ^ ((local_row >> Int32(swizzle_shift)) & Int32(swizzle_mask))
                        )
                        << Int32(4)
                    ),
                    mem_space=3,
                    dtype=Int32,
                )
                staged.append(src.load(count=4, alignment=16))
            for store_idx in cutlass.range_constexpr(stores):
                dst_ptr = cutlass.inttoptr(
                    row_addresses[store_idx] + Int64(chunk * cfg.o_dtype_bytes),
                    mem_space=1,
                    dtype=Int32,
                )
                if row_valid[store_idx]:
                    dst_ptr.store(staged[store_idx], alignment=16)
            # The next chunk overwrites this warp's slice.
            cute.arch.sync_warp()

    @cute.jit
    def _store_ws_half_via_tma(
        self,
        stage_info: StageInfo,
        task_cache,
        *,
        base_addr0,
        store_scale,
        col_base,
    ):
        """Store one WS Layout-E O half as 32-row x 64-channel TMA boxes.

        Each correction warp alternates two 4 KiB SW128 slices: lane l writes
        row l, then one lane issues the tensor store. A slice is rewritten only
        after the store issued from it two chunks earlier has read it.
        """
        cfg = self.cfg
        batch_idx = batch_idx_for_stage_cfg(self.batch_idx, cfg, stage_info)
        head_base_idx = head_idx_for_stage(self.head_idx, cfg, stage_info)
        cta_idx_q = cta_idx_q_for_stage(self.cta_idx_q, stage_info)
        warp_idx = task_cache[_TASK_CACHE_WARP_IDX]
        lane_idx = task_cache[_TASK_CACHE_LANE_IDX]
        slice_bytes = 32 * 128
        warp_offset = warp_idx * Int32(2 * slice_bytes)
        warp_base = self._smem_o_coalesce.data_ptr().toint(Int32) + warp_offset
        storage_flat_query_row, _, _, _, _ = flat_query_row_state(
            head_base_idx + (warp_idx % Int32(2)) * Int32(32),
            cta_idx_q,
            cfg.tile_size_q,
            cfg.logical_num_heads_q,
            cfg.logical_seq_len_q,
            cu_seqlens_q=self.cu_seqlens_q,
            batch_idx=batch_idx,
        )
        first_row = public_query_flat_row(
            cfg, storage_flat_query_row, batch_idx, self.cu_seqlens_q
        )
        first_head = first_row % Int32(cfg.logical_num_heads_q)
        first_query = first_row // Int32(cfg.logical_num_heads_q)
        write_base = lane_idx * Int32(128)
        write_swizzle = lane_idx & Int32(7)
        for chunk in cutlass.range_constexpr(0, 128, 64):
            slice_offset = (chunk // 64) * slice_bytes
            # Keep at most the other slice's store in flight.
            prims.cp_async_bulk_wait_group(1, read=True)
            cute.arch.sync_warp()
            o_loaded = prims.tcgen05_ld(
                prims.Tcgen05LdStShape.SHAPE_32X32B,
                prims.make_tmem_ptr(base_addr0 + Int32(chunk), Float32),
                num=64,
            )
            prims.tcgen05_wait(kind=prims.Tcgen05Wait.LOAD)
            cute.arch.fence_view_async_tmem_load()
            for u in cutlass.range_constexpr(8):
                packed_o = cutlass.Array(Int32, 4, space=cutlass.AddressSpace.rmem)
                for pair_idx in cutlass.range_constexpr(4):
                    reg_base = u * 8 + pair_idx * 2
                    final_pair = fmul2(
                        (store_scale, store_scale),
                        (o_loaded[reg_base], o_loaded[reg_base + 1]),
                    )
                    packed_o[pair_idx] = pack_float2_to_bf16(final_pair[0], final_pair[1])
                dst = cutlass.inttoptr(
                    warp_base
                    + Int32(slice_offset)
                    + write_base
                    + ((Int32(u) ^ write_swizzle) << Int32(4)),
                    mem_space=3,
                    dtype=Int32,
                )
                dst.store(packed_o.load(0, 4), alignment=16)
            cute.arch.fence_view_async_shared()
            cute.arch.sync_warp()
            if prims.elect_sync():
                prims.cp_async_bulk_tensor_global_shared_cta(
                    self.tma_o_desc,
                    self._smem_o_coalesce_bytes.subview(warp_offset + Int32(slice_offset)),
                    (col_base + Int32(chunk), first_head, first_query),
                )
            prims.cp_async_bulk_commit_group()

    @cute.jit
    def _store_ws_half_quant_via_tma(
        self,
        stage_info: StageInfo,
        task_cache,
        *,
        base_addr0,
        store_scale,
        row,
        col_base,
        v_stage_idx: int,
        quant_staged=None,
    ):
        """Quantize one WS Layout-E O half through the output quantizer; store it by TMA.

        Lane l of correction warp w owns output row ``row`` (head (w % 2) * 32 + l)
        and 128 channels from ``col_base``. The quantizer writes each 1x32 block's
        E4M3 bytes into the lane's 128-byte row of the warp's SW128 slice
        (``shared_output`` is the slice's 1 KiB-aligned shared address) and stores the
        block scales itself. One lane then stores the 32 rows as one TMA box of
        (channel, head in group, token, group). The two halves alternate two slices.
        """
        cfg = self.cfg
        quant = self.output_quant
        heads_per_group = cutlass.const_expr(quant.num_heads // quant.num_groups)
        warp_idx = task_cache[_TASK_CACHE_WARP_IDX]
        lane_idx = task_cache[_TASK_CACHE_LANE_IDX]
        slice_bytes = 32 * 128
        slice_offset = warp_idx * Int32(2 * slice_bytes) + Int32(
            (v_stage_idx // 2) * slice_bytes
        )
        slice_base = self._smem_o_coalesce.data_ptr().toint(Int32) + slice_offset
        # Keep at most the other half's store in flight.
        prims.cp_async_bulk_wait_group(1, read=True)
        cute.arch.sync_warp()
        rotary = cutlass.const_expr(v_stage_idx + 2 == cfg.v_head_dim_stages)
        rotary_scratch = None
        if cutlass.const_expr(rotary and quant_staged is not None):
            rotary_scratch = self._quant_scratch.data_ptr().toint(Int32) + warp_idx * Int32(
                quant.rotary_scratch_bytes
            )
            quant.stage_rotary(quant_staged, rotary_scratch)
        # A quantizer that stores four blocks' scales at once gets them as one
        # word per lane and half instead of a byte per block.
        packs_scales = cutlass.const_expr(hasattr(quant, "store_scales"))
        scale_word = Uint32(0)
        for stripe in cutlass.range_constexpr(0, 128, quant.block_size):
            loaded = prims.tcgen05_ld(
                prims.Tcgen05LdStShape.SHAPE_32X32B,
                prims.make_tmem_ptr(base_addr0 + Int32(stripe), Float32),
                num=quant.block_size,
            )
            prims.tcgen05_wait(kind=prims.Tcgen05Wait.LOAD)
            cute.arch.fence_view_async_tmem_load()
            values = cutlass.Array(
                Float32, quant.block_size, space=cutlass.AddressSpace.rmem
            )
            for i in cutlass.range_constexpr(0, quant.block_size, 2):
                pair = fmul2((store_scale, store_scale), (loaded[i], loaded[i + 1]))
                values[i], values[i + 1] = pair[0], pair[1]
            if cutlass.const_expr(packs_scales):
                scale = quant.store(
                    self.output_quant_args,
                    values,
                    row,
                    col_base + Int32(stripe),
                    lane_stride=1,
                    rotary=rotary,
                    shared_output=slice_base,
                    rotary_scratch=rotary_scratch,
                    scale_out=True,
                )
                scale_word = scale_word | (
                    Uint32(scale) << Uint32(8 * (stripe // quant.block_size))
                )
            else:
                quant.store(
                    self.output_quant_args,
                    values,
                    row,
                    col_base + Int32(stripe),
                    lane_stride=1,
                    rotary=rotary,
                    shared_output=slice_base,
                    rotary_scratch=rotary_scratch,
                )
        if cutlass.const_expr(packs_scales):
            quant.store_scales(self.output_quant_args, row, col_base, scale_word)
        cute.arch.fence_view_async_shared()
        cute.arch.sync_warp()
        first_row = Int64(row) - Int64(lane_idx)
        token = Int32(first_row // quant.num_heads)
        first_head = Int32(first_row % quant.num_heads)
        if prims.elect_sync():
            prims.cp_async_bulk_tensor_global_shared_cta(
                self.tma_o_desc,
                self._smem_o_coalesce_bytes.subview(slice_offset),
                (
                    col_base,
                    first_head % Int32(heads_per_group),
                    token,
                    first_head // Int32(heads_per_group),
                ),
            )
        prims.cp_async_bulk_commit_group()

    def _tail_staged_stripes(self):
        """Lane tail stripes (of eight 32-channel ones) held as BF16 words across the O release.

        The WS quantized tail otherwise quantizes every stripe while it holds the O stage,
        so the next tile's first PV waits for the whole epilogue. Staged stripes are only
        loaded and rounded before the release and quantized after it
        (``cfg.ws_tail_staged_stripes``; 0 keeps the current tail).
        """
        if not (
            self._ws_tma_quant_output()
            and hasattr(self.output_quant, "round_words")
            and hasattr(self.output_quant, "store_words")
            and hasattr(self.output_quant, "store_scales")
            and self.output_quant.block_size == 32
            and self.cfg.kernel_variant == "keeps_mma_ab"
            and self.cfg.v_head_dim_stages == 4
            and self.cfg.head_dim_per_stage_v == 128
        ):
            return 0
        return self.cfg.ws_tail_staged_stripes

    @producer_work
    @cute.jit
    def correct_tail_and_stage(
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
        """``correct_tail_and_store`` up to the O release; ``store_staged_tail`` finishes it."""
        del old_max_arr, o_stage_idx
        if cutlass.const_expr(self._tail_staged_stripes() == 0):
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
        else:
            self._stage_ws_quant_tail(
                stage_info,
                new_max_arr=new_max_arr,
                sum_arr=sum_arr,
                inst0_new_max_arr=inst0_new_max_arr,
                inst0_sum_arr=inst0_sum_arr,
                inst1_new_max_arr=inst1_new_max_arr,
                inst1_sum_arr=inst1_sum_arr,
                tail_o_stage_idx_0=tail_o_stage_idx_0,
            )

    @producer_work
    @cute.jit
    def store_staged_tail(self, stage_info: StageInfo):
        """Quantize and store the stripes ``correct_tail_and_stage`` held, after the O release."""
        if cutlass.const_expr(self._tail_staged_stripes() > 0):
            halves, quant_staged, task_cache, head_dim_offset = self._tail_staged
            quant = self.output_quant
            for half in halves:
                (
                    h,
                    slice_offset,
                    slice_base,
                    row,
                    col_base,
                    rotary,
                    rotary_scratch,
                    words,
                ) = half[:8]
                scale_word = half[8]
                if cutlass.const_expr(len(words) > 0):
                    if cutlass.const_expr(len(words) == 4):
                        # No stripe of this half wrote its slice before the release.
                        prims.cp_async_bulk_wait_group(1, read=True)
                        cute.arch.sync_warp()
                        if cutlass.const_expr(rotary and quant_staged is not None):
                            rotary_scratch = self._quant_scratch.data_ptr().toint(
                                Int32
                            ) + task_cache[_TASK_CACHE_WARP_IDX] * Int32(
                                quant.rotary_scratch_bytes
                            )
                            quant.stage_rotary(quant_staged, rotary_scratch)
                    for stripe, stripe_words in words:
                        scale = quant.store_words(
                            self.output_quant_args,
                            stripe_words,
                            row,
                            col_base + Int32(stripe * quant.block_size),
                            lane_stride=1,
                            rotary=rotary,
                            shared_output=slice_base,
                            rotary_scratch=rotary_scratch,
                            scale_out=True,
                        )
                        scale_word = scale_word | (Uint32(scale) << Uint32(8 * stripe))
                    self._finish_ws_quant_half(
                        task_cache,
                        row=row,
                        col_base=col_base,
                        slice_offset=slice_offset,
                        scale_word=scale_word,
                    )
            self._reduce_cluster_partials_and_store(
                stage_info, task_cache, head_dim_offset
            )

    @cute.jit
    def _finish_ws_quant_half(
        self, task_cache, *, row, col_base, slice_offset, scale_word
    ):
        """Store a quantized WS half's block scales and its 32-row TMA box."""
        quant = self.output_quant
        heads_per_group = cutlass.const_expr(quant.num_heads // quant.num_groups)
        lane_idx = task_cache[_TASK_CACHE_LANE_IDX]
        quant.store_scales(self.output_quant_args, row, col_base, scale_word)
        cute.arch.fence_view_async_shared()
        cute.arch.sync_warp()
        first_row = Int64(row) - Int64(lane_idx)
        token = Int32(first_row // quant.num_heads)
        first_head = Int32(first_row % quant.num_heads)
        if prims.elect_sync():
            prims.cp_async_bulk_tensor_global_shared_cta(
                self.tma_o_desc,
                self._smem_o_coalesce_bytes.subview(slice_offset),
                (
                    col_base,
                    first_head % Int32(heads_per_group),
                    token,
                    first_head // Int32(heads_per_group),
                ),
            )
        prims.cp_async_bulk_commit_group()

    @cute.jit
    def _stage_ws_quant_tail(
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
    ):
        """The O-reading part of the WS quantized tail.

        Lane l of correction warp w owns output row (w % 2) * 32 + l and, per half h,
        128 channels from h * 256 + (w // 2) * 128: stripes 4h..4h+3 of 32 channels.
        The last ``_tail_staged_stripes()`` stripes are loaded, scaled and rounded to
        BF16 words; the others are quantized into the half's staging slice now, as
        ``_store_ws_half_quant_via_tma`` does. A half without staged stripes is stored
        now. ``store_staged_tail`` quantizes the rest after the O release, with the same
        operations in the same order per value, so the output bytes do not change.
        """
        cfg = self.cfg
        quant = self.output_quant
        staged_from = cutlass.const_expr(8 - self._tail_staged_stripes())
        task_cache = decode_gen_task_cache(stage_info)
        quant_staged = None
        if cutlass.const_expr(self._prefetches_output_quant()):
            quant_staged = quant.prefetch(
                self.output_quant_args,
                self._ws_lane_output_row(stage_info, task_cache),
                1,
                state=self._tail_quant_state,
            )
        tmem_row_base = task_cache[_TASK_CACHE_TMEM_BASE_OFFSET]
        cta_idx_head_dim_v = cta_idx_head_dim_v_for_stage(
            self.cta_idx_head_dim_v, stage_info
        )
        head_dim_offset = head_dim_cta_offset_v(cfg, cta_idx_head_dim_v)
        final_max, reduced_sum, final_scale0, _ = self._compute_tail_softmax_scales(
            task_cache,
            stage_info,
            new_max_arr=new_max_arr,
            sum_arr=sum_arr,
            inst0_new_max_arr=inst0_new_max_arr,
            inst0_sum_arr=inst0_sum_arr,
            inst1_new_max_arr=inst1_new_max_arr,
            inst1_sum_arr=inst1_sum_arr,
        )
        if cutlass.const_expr(cfg.cluster_split_finisher):
            # As in _normalize_and_store_tail_o: no partial is pushed before the cluster's
            # first CTA armed its push barrier.
            prims.barrier_cluster_wait()
        self._store_lse_to_gmem(stage_info, task_cache, final_max, reduced_sum)
        store_scale = final_scale0[0]
        warp_idx = task_cache[_TASK_CACHE_WARP_IDX]
        row = self._ws_lane_output_row(stage_info, task_cache)
        slice_bytes = 32 * 128
        halves = []
        for h in cutlass.range_constexpr(2):
            v_stage_idx = 2 * h
            base_addr0 = (
                tmem_row_base
                + self._o_base_col()
                + o_stage_tmem_col_offset(cfg, tail_o_stage_idx_0, v_stage_idx)
            )
            col_base = (
                head_dim_offset
                + Int32(v_stage_idx * cfg.head_dim_per_stage_v)
                + (warp_idx // Int32(2)) * Int32(128)
            )
            slice_offset = warp_idx * Int32(2 * slice_bytes) + Int32(h * slice_bytes)
            slice_base = self._smem_o_coalesce.data_ptr().toint(Int32) + slice_offset
            rotary = cutlass.const_expr(v_stage_idx + 2 == cfg.v_head_dim_stages)
            direct = [s for s in range(4) if 4 * h + s < staged_from]
            rotary_scratch = None
            if cutlass.const_expr(len(direct) > 0):
                # Keep at most the other half's store in flight.
                prims.cp_async_bulk_wait_group(1, read=True)
                cute.arch.sync_warp()
                if cutlass.const_expr(rotary and quant_staged is not None):
                    rotary_scratch = self._quant_scratch.data_ptr().toint(
                        Int32
                    ) + warp_idx * Int32(quant.rotary_scratch_bytes)
                    quant.stage_rotary(quant_staged, rotary_scratch)
            scale_word = Uint32(0)
            words = []
            for stripe in cutlass.range_constexpr(4):
                loaded = prims.tcgen05_ld(
                    prims.Tcgen05LdStShape.SHAPE_32X32B,
                    prims.make_tmem_ptr(
                        base_addr0 + Int32(stripe * quant.block_size), Float32
                    ),
                    num=quant.block_size,
                )
                prims.tcgen05_wait(kind=prims.Tcgen05Wait.LOAD)
                cute.arch.fence_view_async_tmem_load()
                values = cutlass.Array(
                    Float32, quant.block_size, space=cutlass.AddressSpace.rmem
                )
                for i in cutlass.range_constexpr(0, quant.block_size, 2):
                    pair = fmul2((store_scale, store_scale), (loaded[i], loaded[i + 1]))
                    values[i], values[i + 1] = pair[0], pair[1]
                if cutlass.const_expr(4 * h + stripe >= staged_from):
                    words.append((stripe, quant.round_words(values)))
                else:
                    scale = quant.store(
                        self.output_quant_args,
                        values,
                        row,
                        col_base + Int32(stripe * quant.block_size),
                        lane_stride=1,
                        rotary=rotary,
                        shared_output=slice_base,
                        rotary_scratch=rotary_scratch,
                        scale_out=True,
                    )
                    scale_word = scale_word | (Uint32(scale) << Uint32(8 * stripe))
            if cutlass.const_expr(len(words) == 0):
                self._finish_ws_quant_half(
                    task_cache,
                    row=row,
                    col_base=col_base,
                    slice_offset=slice_offset,
                    scale_word=scale_word,
                )
            halves.append(
                (
                    h,
                    slice_offset,
                    slice_base,
                    row,
                    col_base,
                    rotary,
                    rotary_scratch,
                    words,
                    scale_word,
                )
            )
        self._tail_staged = (halves, quant_staged, task_cache, head_dim_offset)

    @producer_work(work_attrs=WorkAttr.AUXILIARY)
    @cute.jit
    def drain_output_stores(self, stage_info: StageInfo) -> None:
        """Wait for the WS TMA output stores before the CTA retires."""
        del stage_info
        if cutlass.const_expr(self._ws_tma_output() or self._ws_tma_quant_output()):
            prims.cp_async_bulk_wait_group(0)

    @cute.jit
    def _store_keeps_tail_o_stage_coalesced(
        self,
        stage_info: StageInfo,
        task_cache,
        *,
        base_addr0,
        final_scale0,
        v_stage_idx: int,
        head_dim_offset,
    ):
        """Normalize one Keep-AB O stage and store full 128B lines via SMEM.

        Each lane owns half of one head row, so direct stores touched 32 lines
        per instruction. The warp's sixteen rows are transposed through a
        private SMEM slice; each store then writes two contiguous rows.
        """
        cfg = self.cfg
        batch_idx = batch_idx_for_stage_cfg(self.batch_idx, cfg, stage_info)
        head_base_idx = head_idx_for_stage(self.head_idx, cfg, stage_info)
        cta_idx_q = cta_idx_q_for_stage(self.cta_idx_q, stage_info)
        warp_idx = task_cache[_TASK_CACHE_WARP_IDX]
        lane_idx = task_cache[_TASK_CACHE_LANE_IDX]
        half_cols = cfg.head_dim_per_stage_v // 2
        row_bytes = cfg.head_dim_per_stage_v * cfg.o_dtype_bytes
        chunks_per_row = row_bytes // 16
        load_cols = 32 if half_cols % 32 == 0 else 8
        o_blocks = []
        for block_idx in cutlass.range_constexpr(0, half_cols, load_cols):
            o_blocks.append(
                tcgen05_ld_16x32bx2_f32(
                    prims.make_tmem_ptr(base_addr0 + Int32(block_idx), Float32),
                    num=load_cols,
                    offset=Int32(half_cols),
                )
            )
        prims.tcgen05_wait(kind=prims.Tcgen05Wait.LOAD)
        cute.arch.fence_view_async_tmem_load()

        warp_base = self._smem_o_coalesce.data_ptr().toint(Int32) + warp_idx * Int32(
            16 * row_bytes
        )
        row = lane_idx & Int32(0xF)
        half = lane_idx >> Int32(4)
        # XOR the 16B chunk with the row's low bits: each quarter-warp phase of
        # the row-major writes and the two-row reads hits all 32 banks once.
        for chunk_idx in cutlass.range_constexpr(0, half_cols, 8):
            o_block = o_blocks[chunk_idx // load_cols]
            packed_o = cutlass.Array(Int32, 4, space=cutlass.AddressSpace.rmem)
            for pair_idx in cutlass.range_constexpr(4):
                reg_base = chunk_idx % load_cols + pair_idx * 2
                final_pair = fmul2(
                    (final_scale0[0], final_scale0[0]),
                    (o_block[reg_base], o_block[reg_base + 1]),
                )
                packed_o[pair_idx] = pack_float2_to_bf16(final_pair[0], final_pair[1])
            chunk = half * Int32(half_cols // 8) + Int32(chunk_idx // 8)
            dst = cutlass.inttoptr(
                warp_base + row * Int32(row_bytes) + ((chunk ^ (row & 7)) << 4),
                mem_space=3,
                dtype=Int32,
            )
            dst.store(packed_o.load(0, 4), alignment=16)
        cute.arch.sync_warp()

        rows_per_store = 32 // chunks_per_row
        chunk = lane_idx % Int32(chunks_per_row)
        # Read every staged row and form every address before the predicated
        # stores. A store branch holding its load or address math kept them
        # adjacent, serializing one SMEM round trip per store.
        staged_rows = []
        for store_idx in cutlass.range_constexpr(16 // rows_per_store):
            local_row = Int32(store_idx * rows_per_store) + lane_idx // Int32(
                chunks_per_row
            )
            src = cutlass.inttoptr(
                warp_base
                + local_row * Int32(row_bytes)
                + ((chunk ^ (local_row & 7)) << 4),
                mem_space=3,
                dtype=Int32,
            )
            staged_rows.append(src.load(count=4, alignment=16))
        for store_idx in cutlass.range_constexpr(16 // rows_per_store):
            local_row = Int32(store_idx * rows_per_store) + lane_idx // Int32(
                chunks_per_row
            )
            values = staged_rows[store_idx]
            global_head_idx = head_base_idx + warp_idx * Int32(16) + local_row
            storage_flat_query_row, _, _, _, valid_output_row = flat_query_row_state(
                global_head_idx,
                cta_idx_q,
                cfg.tile_size_q,
                cfg.logical_num_heads_q,
                cfg.logical_seq_len_q,
                cu_seqlens_q=self.cu_seqlens_q,
                batch_idx=batch_idx,
            )
            # Invalid rows form an unused address from safe coordinates.
            output_query_row = public_query_flat_row(
                cfg, storage_flat_query_row, batch_idx, self.cu_seqlens_q
            )
            col = (
                head_dim_offset
                + Int32(v_stage_idx * cfg.head_dim_per_stage_v)
                + chunk * Int32(16 // cfg.o_dtype_bytes)
            )
            elem = Int64(output_query_row) * Int64(cfg.head_dim_v) + Int64(col)
            dst_ptr = cutlass.inttoptr(
                self.o_tensor.iterator.raw_ptr().toint(Int64)
                + elem * Int64(cfg.o_dtype_bytes),
                mem_space=1,
                dtype=Int32,
            )
            if global_head_idx < Int32(cfg.num_heads_q) and valid_output_row:
                dst_ptr.store(values, alignment=16)
        # The next stage overwrites this warp's slice.
        cute.arch.sync_warp()

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
                if cutlass.const_expr(cfg.one_insts_kv_swap):
                    o1_loaded_lo = o0_loaded_lo
                else:
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
                if cutlass.const_expr(cfg.one_insts_kv_swap):
                    o1_loaded_hi = o0_loaded_hi
                else:
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
                if cutlass.const_expr(cfg.one_insts_kv_swap):
                    o1_loaded = o0_loaded
                else:
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
                if cutlass.const_expr(cfg.one_insts_kv_swap):
                    # Only O0 exists: normalize it directly, without a second
                    # TMEM load or the two-stream weighted merge.
                    final_pair = fmul2(
                        (final_scale0[scale_base], final_scale0[scale_base + 1]),
                        o0_pair,
                    )
                else:
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
    def _store_swap_quantized_stage(
        self,
        stage_info,
        task_cache,
        base_addr0,
        base_addr1,
        final_scale0,
        final_scale1,
        v_stage_idx,
        head_dim_offset,
    ):
        """Transpose FP32 O in two half-Q passes using the existing O scratch.

        Half as many FP32 rows fit in the same buffer as the ordinary BF16
        stage. After the transpose, each lane owns a contiguous vector, so
        rotation pairs stay local and quantization needs only warp reductions.
        """
        cfg = self.cfg
        tid = task_cache[_TASK_CACHE_WARP_GRP_THREAD_IDX]
        loaded0 = prims.tcgen05_ld(
            "32x32b", prims.make_tmem_ptr(base_addr0, Float32), num=cfg.tile_size_q
        )
        if cutlass.const_expr(cfg.one_insts_kv_swap):
            loaded1 = loaded0
        else:
            loaded1 = prims.tcgen05_ld(
                "32x32b", prims.make_tmem_ptr(base_addr1, Float32), num=cfg.tile_size_q
            )
        prims.tcgen05_wait(kind=prims.Tcgen05Wait.LOAD)
        cute.arch.fence_view_async_tmem_load()
        batch = batch_idx_for_stage_cfg(self.batch_idx, cfg, stage_info)
        head_base = head_idx_for_stage(self.head_idx, cfg, stage_info)
        q_tile = cta_idx_q_for_stage(self.cta_idx_q, stage_info)
        rows_per_pass = cfg.tile_size_q // 2
        width = rows_per_pass  # 128 channels * rows / 128 correction threads.
        threads_per_row = 128 // width
        for half in cutlass.range_constexpr(2):
            for i in cutlass.range_constexpr(rows_per_pass):
                h = half * rows_per_pass + i
                scale_idx = (h // 8) * 2 + h % 2
                source_lane = (h % 8) // 2
                s0 = cprims.shfl_sync(
                    thread_mask=0xFFFFFFFF,
                    val=final_scale0[scale_idx],
                    offset=source_lane,
                    mask_and_clamp=0x1F,
                    kind=cprims.Shfl.IDX,
                )
                value = loaded0[h] * s0
                if cutlass.const_expr(not cfg.one_insts_kv_swap):
                    s1 = cprims.shfl_sync(
                        thread_mask=0xFFFFFFFF,
                        val=final_scale1[scale_idx],
                        offset=source_lane,
                        mask_and_clamp=0x1F,
                        kind=cprims.Shfl.IDX,
                    )
                    value += loaded1[h] * s1
                self._smem_o_i32[i * 128 + tid] = value.bitcast(Int32)
            prims.barrier_cta_sync(
                self.store_barrier_id, thread_count=WARPGROUP_THREADS
            )
            h = head_base + half * rows_per_pass + tid // threads_per_row
            flat_row, _, _, _, valid = flat_query_row_state(
                h,
                q_tile,
                cfg.tile_size_q,
                cfg.logical_num_heads_q,
                cfg.logical_seq_len_q,
                self.cu_seqlens_q,
                batch,
            )
            values = self._smem_o_i32.load(
                tid * width, vector_size=width, alignment=width * 4
            ).bitcast(Float32)
            if h < cfg.num_heads_q and valid:
                row = public_query_flat_row(cfg, flat_row, batch, self.cu_seqlens_q)
                self.output_quant.store(
                    self.output_quant_args,
                    values,
                    row,
                    head_dim_offset
                    + v_stage_idx * cfg.head_dim_per_stage_v
                    + (tid % threads_per_row) * width,
                    rotary=v_stage_idx + 1 == cfg.v_head_dim_stages,
                )
            prims.barrier_cta_sync(
                self.store_barrier_id, thread_count=WARPGROUP_THREADS
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
            # The quantizer's tail inputs load now and are used only by the last
            # O half, so their latency overlaps the tail's other work.
            quant_staged = None
            if cutlass.const_expr(self._prefetches_output_quant()):
                quant_staged = self.output_quant.prefetch(
                    self.output_quant_args,
                    self._ws_lane_output_row(stage_info, task_cache),
                    1,
                    state=self._tail_quant_state,
                )
            tmem_row_base = task_cache[_TASK_CACHE_TMEM_BASE_OFFSET]
            cta_idx_head_dim_v = cta_idx_head_dim_v_for_stage(
                self.cta_idx_head_dim_v, stage_info
            )
            head_dim_offset = head_dim_cta_offset_v(cfg, cta_idx_head_dim_v)
            final_max, reduced_sum, final_scale0, final_scale1 = (
                self._compute_tail_softmax_scales(
                    task_cache,
                    stage_info,
                    new_max_arr=new_max_arr,
                    sum_arr=sum_arr,
                    inst0_new_max_arr=inst0_new_max_arr,
                    inst0_sum_arr=inst0_sum_arr,
                    inst1_new_max_arr=inst1_new_max_arr,
                    inst1_sum_arr=inst1_sum_arr,
                )
            )
            if cutlass.const_expr(cfg.cluster_split_finisher):
                # Pairs the kernel's arrival after setup: the first CTA's push barrier is
                # armed before any partial is pushed into it.
                prims.barrier_cluster_wait()
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
                        quant_staged=quant_staged,
                    )
                elif cutlass.const_expr(
                    self.output_quant is not None and cfg.fuse_sparse_epilogue
                ):
                    self._store_swap_quantized_stage(
                        stage_info,
                        task_cache,
                        base_addr0,
                        base_addr1,
                        final_scale0,
                        final_scale1,
                        v_stage_idx,
                        head_dim_offset,
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
