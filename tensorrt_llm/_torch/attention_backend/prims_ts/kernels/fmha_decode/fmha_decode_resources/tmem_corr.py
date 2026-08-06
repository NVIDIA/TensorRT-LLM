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

"""``TmemCorrResource`` — correction and output resource.

LOOP stages rescale an in-flight O tile when the running max changes;
TAIL stages combine the two BMM2 instances, normalize by the final
denominator, and either store the final O tile or write partial O for a
split-KV reduction.
"""

from dataclasses import dataclass

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, Int64, Uint32

from cutlass.experimental import primitives as prims
from cutlass.experimental.task_scheduling.enums import WorkAttr
from cutlass.experimental.task_scheduling.memory import (
    ResourceContext,
    SmemAllocation,
    TmemAllocation,
)
from cutlass.experimental.task_scheduling.resources import (
    StageInfo,
    producer_work,
)

from ..fmha_decode_config import FmhaDecodeConfig
from ...placeholder_helpers import _placeholder_smem_array
from .helpers_common import (
    Constexpr,
    DecodeGenResourceBase,
    ResourceVars,
    fadd2,
    ffma2,
    fmul2,
    _TASK_CACHE_LANE_IDX,
    _TASK_CACHE_TMEM_BASE_OFFSET,
    _TASK_CACHE_WARP_GRP_THREAD_IDX,
    _TASK_CACHE_WARP_IDX,
    _decode_gen_task_cache,
    _keeps_col_base,
    _keeps_row_idx,
    _keeps_tcgen05_ld,
    _keeps_tcgen05_st,
    _attention_sink_head_stride,
    _local_head_from_q_output_row,
    _logical_head_batch,
    _logical_q_group_idx,
    _q_group_token_base,
    _q_physical_output_row,
    _q_physical_output_row_from_logical,
    _q_row_is_valid_for_seq,
    _q_tile_output_row_base,
    _q_tile_valid_rows_for_seq,
    _neg_max_f32,
    _pack_float2_to_bf16,
    _pack_float2_to_fp16,
)
from .helpers_kv_tile_idx import (
    _load_runtime_seq_len_kv,
    _logical_cta_kv_idx,
    _runtime_execution_splits_kv,
)
from .helpers_output import (
    _copy_transposed_smem8b_to_gmem,
    _fp16_o_reorg_offsets,
    _load_partial_o_vec8_as_f32,
    _store_transposed_smem8b,
)
from .helpers_softmax import (
    _attention_sink_for_local_head,
    _attention_sink_for_scale_idx,
    _pack_float4_to_fp8_e4m3,
)
from .smem_p import SmemPResource
from .tmem_softmax_stats import TmemSoftmaxLocalResource


@dataclass(kw_only=True)
class TmemCorrResource(DecodeGenResourceBase):
    """Correction and output resource.

    Loop stages rescale an in-flight O tile when the running softmax max
    changes. Tail stages combine the two BMM2 instances, normalize by the final
    denominator, and either store the final O tile or write partial results for
    a split-KV reduction.
    """

    inst_id: Constexpr[int] = 0
    cfg: Constexpr[FmhaDecodeConfig] = None
    softmax_local0_ref: Constexpr[TmemSoftmaxLocalResource] = None
    softmax_local1_ref: Constexpr[TmemSoftmaxLocalResource] = None
    scale_softmax_log2: Float32 = None
    output_scale: Float32 = None
    o_ptr: cute.Pointer = None
    partial_o_ptr: cute.Pointer = None
    partial_stats_ptr: cute.Pointer = None
    split_kv_counter_ptr: cute.Pointer = None
    attention_sinks_ptr: cute.Pointer = None
    seqlens_kv: cute.Pointer = None
    max_seq_len_kv: Constexpr[int] = 0
    seq_len_q: Int32 = None
    q_token_offset: Int32 = None
    num_heads_kv: Int32 = None
    h_r: Int32 = None
    h_k_idx: Int32 = None
    b_idx: Int32 = None
    q_group_idx: Int32 = None
    active_splits_kv: Int32 = None
    static_full_split_prefix: Constexpr[bool] = False
    smem_p0_ref: Constexpr[SmemPResource] = None
    smem_p1_ref: Constexpr[SmemPResource] = None
    tmem_o_ref: object = None
    store_barrier_id: Constexpr[int] = 6
    sum_barrier_id: Constexpr[int] = 7
    _alloc: Constexpr[SmemAllocation | None] = None
    _sum_alloc: Constexpr[SmemAllocation | None] = None
    _gmem_reducer_rank_alloc: Constexpr[SmemAllocation | None] = None
    _smem_base_o_i32: cutlass.Array = None
    _sum_scratch: cutlass.Array = None
    _gmem_reducer_rank: cutlass.Array = None
    _cluster_partial_o_alloc: Constexpr[SmemAllocation | None] = None
    _cluster_partial_stats_alloc: Constexpr[SmemAllocation | None] = None
    _cluster_mbarrier_alloc: Constexpr[SmemAllocation | None] = None
    _cluster_partial_o_i32: cutlass.Array = None
    _cluster_partial_stats: cutlass.Array = None
    _cluster_mbarrier: cutlass.Array = None

    def _init_placeholder_state(self) -> None:
        """Create placeholder SMEM views for correction and split reduction."""
        o_stage_dtype_bytes = (
            2
            if self.cfg.use_split_kv and self.cfg.use_fp8_output
            else self.cfg.o_dtype_bytes
        )
        o_entries = max(
            self.cfg.tile_size_q * self.cfg.headdim * o_stage_dtype_bytes // 4,
            1,
        )
        # Keeps reduces each row's denominator in registers (plus the q64
        # xor-16 lane pair) and never enters either Swaps scratch reducer.
        # Retain a one-element placeholder for construction-time tracing, but
        # do not reserve physical SMEM for it.
        sum_entries = max(self.cfg.correction_sum_scratch_entries, 1)
        cluster_o_entries = max(
            self.cfg.cluster_max_runtime_partial_rows * self.cfg.headdim * 2 // 4,
            1,
        )
        cluster_stats_entries = max(
            self.cfg.cluster_max_runtime_partial_rows * 2,
            1,
        )
        self._smem_base_o_i32 = _placeholder_smem_array(Int32, o_entries)
        self._sum_scratch = _placeholder_smem_array(Float32, sum_entries)
        self._gmem_reducer_rank = _placeholder_smem_array(Int32, 1)
        self._cluster_partial_o_i32 = _placeholder_smem_array(Int32, cluster_o_entries)
        self._cluster_partial_stats = _placeholder_smem_array(
            Float32, cluster_stats_entries
        )
        self._cluster_mbarrier = _placeholder_smem_array(cutlass.Int64, 1)

    def _owns_final_epilogue(self) -> bool:
        """Whether this correction instance owns final normalization/output."""
        return (self.cfg.num_insts_kv == 1 and self.inst_id == 0) or (
            self.cfg.num_insts_kv != 1 and self.inst_id == 1
        )

    @cute.jit
    def _runtime_splits_kv(self, stage_info: StageInfo) -> Int32:
        """Return the useful producer/reducer fanout for this logical tile."""
        cfg = self.cfg
        if cutlass.const_expr(self.static_full_split_prefix):
            return Int32(cfg.splits_kv)
        # Nonpersistent split-KV launches already derived this cluster-uniform
        # prefix at kernel entry to prune the physical CTA suffix. Reuse that
        # value in correction instead of loading seq_lens and repeating the
        # integer partition in every correction lane.
        if cutlass.const_expr(self.active_splits_kv is not None):
            return self.active_splits_kv
        seq_len_kv = _load_runtime_seq_len_kv(
            self.seqlens_kv,
            self.max_seq_len_kv,
            stage_info,
            self.h_k_idx,
            self.b_idx,
        )
        logical_q_group_idx = _logical_q_group_idx(cfg, stage_info, self.q_group_idx)
        return _runtime_execution_splits_kv(
            cfg,
            seq_len_kv,
            self.seq_len_q,
            _q_group_token_base(cfg, logical_q_group_idx),
        )

    def _can_alias_o_smem(self) -> bool:
        """Whether final O staging can reuse the two contiguous P buffers."""
        # Persistent CTAs can start the next work tile while correction is still
        # draining the final O staging path. Keep O staging separate from SmemP
        # there so the next tile's P producer cannot overwrite the copy source.
        o_stage_dtype_bytes = (
            2
            if self.cfg.use_split_kv and self.cfg.use_fp8_output
            else self.cfg.o_dtype_bytes
        )
        required_o_bytes = self.cfg.tile_size_q * self.cfg.headdim * o_stage_dtype_bytes
        available_p_bytes = 2 * self.cfg.smem_p_tile_bytes
        return (
            self.smem_p0_ref is not None
            and self.smem_p1_ref is not None
            and not self.cfg.use_persistent_scheduler
            and available_p_bytes >= required_o_bytes
        )

    def get_smem_requirements(self) -> list[SmemAllocation]:
        """Allocate correction SMEM scratch, staging, and cluster partial buffers."""
        if not self._owns_final_epilogue():
            return []
        o_stage_dtype_bytes = (
            2
            if self.cfg.use_split_kv and self.cfg.use_fp8_output
            else self.cfg.o_dtype_bytes
        )
        needs_o_staging = not self.cfg.use_keeps_mma_ab and not self._can_alias_o_smem()
        if needs_o_staging and self._alloc is None:
            # O staging is only needed for the final correction/output path and
            # lives on inst1, which owns the final two-instance reduction.
            self._alloc = SmemAllocation(
                name=f"{self.name}",
                size_bytes=self.cfg.tile_size_q
                * self.cfg.headdim
                * o_stage_dtype_bytes,
                alignment=self.cfg.stensor_align,
            )
        sum_entries = self.cfg.correction_sum_scratch_entries
        if self._sum_alloc is None and sum_entries > 0:
            self._sum_alloc = SmemAllocation(
                name=f"{self.name}_sumScratch",
                size_bytes=sum_entries * 4,
                alignment=16,
            )
        if (
            self.cfg.use_split_kv
            and not self.cfg.use_separate_reduction_kernel
            and not self.cfg.supports_cluster_smem_reduction
            and self._gmem_reducer_rank_alloc is None
        ):
            self._gmem_reducer_rank_alloc = SmemAllocation(
                name=f"{self.name}_gmemReducerRank",
                size_bytes=4,
                alignment=4,
            )
        if self.cfg.supports_cluster_smem_reduction:
            # Owner-CTA distributed-SMEM staging for the cluster reducer: every
            # split's partial O (fp16) and stats (float2) for the rows this CTA
            # owns. Peers write into these via prims.mapa; the owner reads them
            # locally instead of round-tripping the partials through GMEM.
            max_partial_rows = self.cfg.cluster_max_runtime_partial_rows
            if self._cluster_partial_o_alloc is None:
                self._cluster_partial_o_alloc = SmemAllocation(
                    name=f"{self.name}_clusterPartialO",
                    size_bytes=max_partial_rows * self.cfg.headdim * 2,
                    alignment=16,
                )
            if self._cluster_partial_stats_alloc is None:
                self._cluster_partial_stats_alloc = SmemAllocation(
                    name=f"{self.name}_clusterPartialStats",
                    size_bytes=max_partial_rows * 2 * 4,
                    alignment=16,
                )
            if self._cluster_mbarrier_alloc is None:
                # One transaction mbarrier per owner CTA; peers async-store into
                # this CTA's partial buffers and signal it.
                self._cluster_mbarrier_alloc = SmemAllocation(
                    name=f"{self.name}_clusterTransactionBarrier",
                    size_bytes=8,
                    alignment=8,
                )
        allocs = []
        if self._alloc is not None:
            allocs.append(self._alloc)
        if self._sum_alloc is not None:
            allocs.append(self._sum_alloc)
        if self._gmem_reducer_rank_alloc is not None:
            allocs.append(self._gmem_reducer_rank_alloc)
        if self._cluster_partial_o_alloc is not None:
            allocs.append(self._cluster_partial_o_alloc)
        if self._cluster_partial_stats_alloc is not None:
            allocs.append(self._cluster_partial_stats_alloc)
        if self._cluster_mbarrier_alloc is not None:
            allocs.append(self._cluster_mbarrier_alloc)
        return allocs

    def get_tmem_requirements(self) -> list[TmemAllocation]:
        """Correction consumes TMEM O but does not allocate new TMEM."""
        return []

    @cute.jit
    def _split_reduction_rows_per_cta(
        self, total_rows: Int32, splits_kv: Int32
    ) -> Int32:
        """Return the slice-aligned row capacity owned by each split reducer."""
        rows_per_slice = Int32(self.cfg.split_reduction_rows_per_slice)
        num_slices = (total_rows + rows_per_slice - Int32(1)) // rows_per_slice
        slices_per_cta = (num_slices + splits_kv - Int32(1)) // splits_kv
        return slices_per_cta * rows_per_slice

    @cute.jit
    def _split_reduction_row_is_owned(
        self, cta_idx_kv: Int32, row_idx: Int32, total_rows: Int32, splits_kv: Int32
    ) -> bool:
        """Test whether this split CTA owns one reducer row."""
        rows_per_cta = self._split_reduction_rows_per_cta(total_rows, splits_kv)
        row_start = cta_idx_kv * rows_per_cta
        row_end = cute.math.min(row_start + rows_per_cta, total_rows)
        return row_idx >= row_start and row_idx < row_end

    @cute.jit
    def _multi_cta_counter_q_groups(self, h_r: Int32) -> Int32:
        """Return the number of split-KV counter groups in the output tile."""
        cfg = self.cfg
        if cutlass.const_expr(cfg.groups_tokens_heads_q):
            return Int32(
                (cfg.max_seq_len_q + cfg.q_tokens_per_cta - 1) // cfg.q_tokens_per_cta
            )
        if cutlass.const_expr(
            cfg.max_seq_len_q > 1
            and not cfg.groups_tokens_heads_q
            and cfg.heads_q_per_kv != 0
        ):
            return Int32(
                ((cfg.heads_q_per_kv + cfg.tile_size_q - 1) // cfg.tile_size_q)
                * cfg.max_seq_len_q
            )
        return (h_r + Int32(cfg.tile_size_q - 1)) // Int32(cfg.tile_size_q)

    @cute.jit
    def _gmem_partial_row_offset(
        self,
        logical_kv_idx: Int32,
        cta_idx_kv: Int32,
        row_idx: Int32,
    ) -> Int64:
        """Return a split/row workspace offset using 64-bit arithmetic."""

        return (
            Int64(logical_kv_idx) * Int64(self.cfg.max_splits_kv) + Int64(cta_idx_kv)
        ) * Int64(self.h_r) + Int64(row_idx)

    @cute.jit
    def _cluster_reduction_rows_per_cta(self, splits_kv: Int32) -> Int32:
        """Return the cluster owner row-band height for active splits."""
        return self._split_reduction_rows_per_cta(
            Int32(self.cfg.tile_size_q), splits_kv
        )

    @cute.jit
    def _cluster_reduction_cta_for_row(self, row_idx: Int32, splits_kv: Int32) -> Int32:
        """Map an output row to its owning cluster split CTA."""
        return row_idx // self._cluster_reduction_rows_per_cta(splits_kv)

    @cute.jit
    def _cluster_reduction_local_row(self, row_idx: Int32, splits_kv: Int32) -> Int32:
        """Return the row index inside the owning cluster reducer band."""
        rows_per_cta = self._cluster_reduction_rows_per_cta(splits_kv)
        return row_idx - (row_idx // rows_per_cta) * rows_per_cta

    @cute.jit
    def _cluster_reduction_row_is_owned(
        self, cta_idx_kv: Int32, row_idx: Int32, splits_kv: Int32
    ) -> bool:
        """Test whether this split CTA owns one cluster reducer row."""
        return self._split_reduction_row_is_owned(
            cta_idx_kv, row_idx, Int32(self.cfg.tile_size_q), splits_kv
        )

    @cute.jit
    def _cluster_partial_row_idx(
        self, split_idx: Int32, local_row_idx: Int32, splits_kv: Int32
    ) -> Int32:
        """Linearize a split/local-row pair in cluster partial buffers."""
        return (
            split_idx * self._cluster_reduction_rows_per_cta(splits_kv) + local_row_idx
        )

    @cute.jit
    def _cluster_partial_o_i32_offset(
        self, row_idx: Int32, col_offset_bytes: Int32
    ) -> Int32:
        """Return the int32 offset for a cluster partial-O fragment."""
        return (row_idx * Int32(self.cfg.headdim * 2) + col_offset_bytes) >> Int32(2)

    @cute.jit
    def _cluster_partial_stats_offset(self, row_idx: Int32) -> Int32:
        """Return the float offset for a cluster partial max/sum pair."""
        _ = self
        return row_idx * Int32(2)

    @cute.jit
    def _cluster_init_transaction_barrier(
        self, mbarrier, transaction_bytes: Constexpr[int]
    ) -> None:
        """Initialize the owner CTA mbarrier for peer async partial stores."""
        _ = self
        tidx, _, _ = cute.arch.thread_idx()
        if tidx == Int32(0):
            # One lane initializes the owner CTA's transaction mbarrier. The
            # expected byte count is the total peer partial-O + stats traffic
            # this owner receives before it can reduce local distributed SMEM.
            prims.mbarrier_init(mbarrier, 1)
            prims.mbarrier_arrive_expect_tx(mbarrier, transaction_bytes)

    @cute.jit
    def _cluster_wait_transaction_barrier(
        self,
        mbarrier,
        warp_grp_thread_idx: Int32,
        barrier_id: Constexpr[int],
        barrier_threads: Constexpr[int],
    ) -> None:
        """Wait for all peer async cluster partial stores before local reduction."""
        _ = self
        # Give every correction lane one nonblocking acquire attempt. When the
        # short transaction is already complete this preserves the all-lane
        # fast path; otherwise only lane 0 continues polling. The existing
        # correction-group barrier publishes lane 0's ready point before any
        # reducer lane reads the distributed-SMEM partials.
        cluster_transaction_ready = prims.mbarrier_try_wait_parity(
            mbarrier, 0, time_limit=0
        )
        if warp_grp_thread_idx == Int32(0):
            while not cluster_transaction_ready:
                cluster_transaction_ready = prims.mbarrier_try_wait_parity(
                    mbarrier, 0, time_limit=10_000_000
                )
        prims.barrier_cta_sync(barrier_id, thread_count=barrier_threads)

    @cute.jit
    def _cluster_complete_inactive_row_bytes(
        self,
        mbarrier,
        cta_idx_kv: Int32,
        splits_kv: Int32,
        logical_q_group_idx: Int32,
        warp_grp_thread_idx: Int32,
    ) -> None:
        """Complete transaction bytes suppressed for inactive owner rows.

        The barrier is initialized with the physical owner-band upper bound.
        Partial O and stats stores are issued only for valid token/head rows, so
        one correction lane reports the remaining bytes before the owner waits.
        """
        cfg = self.cfg
        rows_per_cta = self._cluster_reduction_rows_per_cta(splits_kv)
        valid_tile_rows = _q_tile_valid_rows_for_seq(
            cfg,
            self.h_r,
            logical_q_group_idx,
            self.seq_len_q,
        )
        owner_row_start = cta_idx_kv * rows_per_cta
        valid_owner_rows = cute.math.min(
            cute.math.max(valid_tile_rows - owner_row_start, Int32(0)),
            rows_per_cta,
        )
        bytes_per_row = Int32(cfg.headdim * 2 + 8)
        actual_transaction_bytes = splits_kv * valid_owner_rows * bytes_per_row
        completion_bytes = (
            Int32(cfg.cluster_transaction_bytes) - actual_transaction_bytes
        )
        if warp_grp_thread_idx == Int32(0) and completion_bytes > Int32(0):
            prims.mbarrier_complete_tx(mbarrier, completion_bytes)

    @cute.jit
    def _cluster_store_async_vec4_i32(self, dst_ptr, vals, mbarrier) -> None:
        """Async-store one packed partial-O vector into owner distributed SMEM."""
        _ = self
        # Peer CTA writes one 16-byte partial-O vector into owner distributed
        # SMEM and charges the bytes to the owner's transaction mbarrier.
        # Keep the vectorized publication on the public inline-PTX API so the
        # operation remains one 16-byte store instead of four scalar stores.
        cute.arch.inline_ptx(
            "st.async.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 "
            "[{$r0}], {{$r1}, {$r2}, {$r3}, {$r4}}, [{$r5}];",
            read_only_args=[
                dst_ptr.ir_value(),
                vals[0],
                vals[1],
                vals[2],
                vals[3],
                mbarrier.ir_value(),
            ],
        )

    @cute.jit
    def _cluster_store_async_vec2_f32(
        self, dst_ptr, val0: Float32, val1: Float32, mbarrier
    ) -> None:
        """Async-store one partial max/sum pair into owner distributed SMEM."""
        _ = self
        # Peer CTA writes one float2 (max, sum) stats record and signals the
        # same owner mbarrier used by the matching partial-O vectors.
        cute.arch.inline_ptx(
            "st.async.shared::cluster.mbarrier::complete_tx::bytes.v2.f32 "
            "[{$r0}], {{$r1}, {$r2}}, [{$r3}];",
            read_only_args=[
                dst_ptr.ir_value(),
                val0,
                val1,
                mbarrier.ir_value(),
            ],
        )

    @cute.jit
    def _safe_norm_rcp(self, sum_val: Float32) -> Float32:
        """Clamp the softmax denominator before approximate reciprocal."""
        _ = self
        return cute.math.rcp(
            cute.math.max(sum_val, Float32(1.0e-12), ftz=True), approx=True
        )

    @cute.jit
    def _separate_partial_norm_scale(self, sum_val: Float32) -> Float32:
        """Return the public output-domain scale for normalized partial O."""

        # FlashInfer exposes bmm2_scale for every output dtype. Fold it into
        # every normalized partial so the shared reducer only merges states.
        return self.output_scale * self._safe_norm_rcp(sum_val)

    @cute.jit
    def _separate_partial_lse(self, max_val: Float32, sum_val: Float32) -> Float32:
        """Convert one local max/sum pair to the shared log2-LSE contract."""

        lse_val = Float32(-Float32.inf)
        if sum_val > Float32(0.0):
            lse_val = self.scale_softmax_log2 * max_val + cute.math.log2(
                sum_val, fastmath=True
            )
        return lse_val

    @cute.jit
    def _pack_separate_partial_o_pair(self, val0: Float32, val1: Float32) -> Int32:
        """Pack normalized separate partial O in the selected 16-bit type."""

        if cutlass.const_expr(self.cfg.use_bf16_separate_partial_o):
            return _pack_float2_to_bf16(val0, val1)
        return _pack_float2_to_fp16(val0, val1)

    @cute.jit
    def _swaps_o_stage_base_addr(
        self,
        tmem_row_base: Int32,
        o_base_col: Constexpr[int],
        o_stage_idx: Int32,
    ) -> Int32:
        """Return the TMEM base for one logical Swaps O stage."""
        return (
            tmem_row_base
            + Int32(o_base_col)
            + o_stage_idx * Int32(self.cfg.tmem_o_stage_cols)
        )

    @cute.jit
    def _online_softmax_correction_scale(
        self,
        old_max: Float32,
        new_max: Float32,
    ) -> tuple[Float32, cutlass.Boolean]:
        """Return the exact max-change scale and whether it is identity."""
        scale_is_identity = old_max == new_max
        scale = Float32(1.0)
        if not scale_is_identity:
            scale = cute.math.exp2(
                self.scale_softmax_log2 * (old_max - new_max),
                fastmath=True,
            )
        return scale, scale_is_identity

    @cute.jit
    def _warp_can_skip_o_correction(
        self,
        lane_scales_are_identity: cutlass.Boolean,
    ) -> cutlass.Boolean:
        """Return true only when every lane can leave its O fragment unchanged."""
        _ = self
        return prims.vote_sync(
            cute.arch.FULL_MASK,
            lane_scales_are_identity,
            prims.VoteSync.ALL,
        )

    @cute.jit
    def _swaps_load_o_stage_chunks(
        self,
        base_addr: Int32,
        *,
        q_repeats: Constexpr[int],
        num_o_chunks: Constexpr[int],
        output_f32_regs: Constexpr[int],
    ) -> cutlass.Array:
        """Load all 64-column chunks from one Swaps O stage into registers."""
        cfg = self.cfg
        o_vals = cutlass.Array(
            Float32, output_f32_regs, space=cutlass.AddressSpace.rmem
        )
        for chunk_idx in cutlass.range_constexpr(num_o_chunks):
            loaded = prims.tcgen05_ld(
                "16x256b",
                prims.make_tmem_ptr(
                    base_addr + Int32(cfg.swaps_o_chunk_tmem_offset(chunk_idx)),
                    Float32,
                ),
                num=q_repeats,
            )
            for reg_idx in cutlass.range_constexpr(4 * q_repeats):
                o_vals[chunk_idx * 4 * q_repeats + reg_idx] = loaded[reg_idx]
        prims.tcgen05_wait(kind=prims.Tcgen05Wait.LOAD)
        cute.arch.fence_view_async_tmem_load()
        return o_vals

    @cute.jit
    def _swaps_load_two_o_stage_chunks(
        self,
        base_addr0: Int32,
        base_addr1: Int32,
        *,
        q_repeats: Constexpr[int],
        num_o_chunks: Constexpr[int],
        output_f32_regs: Constexpr[int],
    ) -> tuple[cutlass.Array, cutlass.Array]:
        """Load matching chunks from the two final Swaps O stages."""
        cfg = self.cfg
        o0_vals = cutlass.Array(
            Float32, output_f32_regs, space=cutlass.AddressSpace.rmem
        )
        o1_vals = cutlass.Array(
            Float32, output_f32_regs, space=cutlass.AddressSpace.rmem
        )
        for chunk_idx in cutlass.range_constexpr(num_o_chunks):
            o0_loaded = prims.tcgen05_ld(
                "16x256b",
                prims.make_tmem_ptr(
                    base_addr0 + Int32(cfg.swaps_o_chunk_tmem_offset(chunk_idx)),
                    Float32,
                ),
                num=q_repeats,
            )
            o1_loaded = prims.tcgen05_ld(
                "16x256b",
                prims.make_tmem_ptr(
                    base_addr1 + Int32(cfg.swaps_o_chunk_tmem_offset(chunk_idx)),
                    Float32,
                ),
                num=q_repeats,
            )
            for reg_idx in cutlass.range_constexpr(4 * q_repeats):
                o0_vals[chunk_idx * 4 * q_repeats + reg_idx] = o0_loaded[reg_idx]
                o1_vals[chunk_idx * 4 * q_repeats + reg_idx] = o1_loaded[reg_idx]
        prims.tcgen05_wait(kind=prims.Tcgen05Wait.LOAD)
        cute.arch.fence_view_async_tmem_load()
        return o0_vals, o1_vals

    @cute.jit
    def _swaps_store_scaled_o_chunk(
        self,
        base_addr: Int32,
        chunk_idx: Constexpr[int],
        chunk_scaled: cutlass.Array,
        *,
        q_repeats: Constexpr[int],
    ) -> None:
        """Store one scaled Swaps O chunk back to TMEM."""
        cfg = self.cfg
        if cutlass.const_expr(q_repeats == 4):
            scaled_vec = cutlass.Vector.from_elements(
                (
                    chunk_scaled[0],
                    chunk_scaled[1],
                    chunk_scaled[2],
                    chunk_scaled[3],
                    chunk_scaled[4],
                    chunk_scaled[5],
                    chunk_scaled[6],
                    chunk_scaled[7],
                    chunk_scaled[8],
                    chunk_scaled[9],
                    chunk_scaled[10],
                    chunk_scaled[11],
                    chunk_scaled[12],
                    chunk_scaled[13],
                    chunk_scaled[14],
                    chunk_scaled[15],
                ),
                Float32,
            )
        elif cutlass.const_expr(q_repeats == 2):
            scaled_vec = cutlass.Vector.from_elements(
                (
                    chunk_scaled[0],
                    chunk_scaled[1],
                    chunk_scaled[2],
                    chunk_scaled[3],
                    chunk_scaled[4],
                    chunk_scaled[5],
                    chunk_scaled[6],
                    chunk_scaled[7],
                ),
                Float32,
            )
        else:
            scaled_vec = cutlass.Vector.from_elements(
                (
                    chunk_scaled[0],
                    chunk_scaled[1],
                    chunk_scaled[2],
                    chunk_scaled[3],
                ),
                Float32,
            )
        prims.tcgen05_st(
            "16x256b",
            prims.make_tmem_ptr(
                base_addr + Int32(cfg.swaps_o_chunk_tmem_offset(chunk_idx)),
                Float32,
            ),
            scaled_vec,
        )

    @cute.jit
    def _swaps_rescale_o_stage_in_tmem(
        self,
        base_addr: Int32,
        scale_vals: cutlass.Array,
        skip_correction: cutlass.Boolean,
        *,
        q_repeats: Constexpr[int],
        num_o_chunks: Constexpr[int],
        output_f32_regs: Constexpr[int],
    ) -> None:
        """Apply online-softmax correction scales to one Swaps O stage."""
        if not skip_correction:
            o_vals = self._swaps_load_o_stage_chunks(
                base_addr,
                q_repeats=q_repeats,
                num_o_chunks=num_o_chunks,
                output_f32_regs=output_f32_regs,
            )
            for chunk_idx in cutlass.range_constexpr(num_o_chunks):
                chunk_scaled = cutlass.Array(
                    Float32, 4 * q_repeats, space=cutlass.AddressSpace.rmem
                )
                for reg_pair_idx in cutlass.range_constexpr(2 * q_repeats):
                    global_pair_idx = chunk_idx * (2 * q_repeats) + reg_pair_idx
                    scale_base = (reg_pair_idx // 2) * 2
                    reg_base = global_pair_idx * 2
                    scaled_pair = fmul2(
                        (
                            scale_vals[scale_base],
                            scale_vals[scale_base + 1],
                        ),
                        (o_vals[reg_base], o_vals[reg_base + 1]),
                    )
                    chunk_scaled[reg_pair_idx * 2] = scaled_pair[0]
                    chunk_scaled[reg_pair_idx * 2 + 1] = scaled_pair[1]
                self._swaps_store_scaled_o_chunk(
                    base_addr,
                    chunk_idx,
                    chunk_scaled,
                    q_repeats=q_repeats,
                )
        # Keep the correction task's TMEM ordering point on the no-op path.
        prims.tcgen05_wait(kind=prims.Tcgen05Wait.STORE)
        if not skip_correction:
            # FlashInfer's established TMEM path keeps this view fence after
            # real stores. Avoid emitting its duplicate wait on a no-op path.
            cute.arch.fence_view_async_tmem_store()

    @cute.jit
    def _zero_o_vec8(self) -> cutlass.Array:
        """Return a zero-initialized 8-element O accumulator fragment."""
        _ = self
        output_vals = cutlass.Array(Float32, 8, space=cutlass.AddressSpace.rmem)
        for elem_idx in cutlass.range_constexpr(8):
            output_vals[elem_idx] = Float32(0.0)
        return output_vals

    @cute.jit
    def _fold_split_o_vec8(
        self,
        output_vals: cutlass.Array,
        sum_val: Float32,
        old_max_val: Float32,
        max_val: Float32,
        local_max: Float32,
        local_sum: Float32,
        loaded_partial_regs: cutlass.Array,
    ) -> tuple[cutlass.Array, Float32, Float32, Float32]:
        """Fold one split-KV partial into the online-softmax reducer state."""
        cfg = self.cfg
        new_max = cute.math.max(max_val, local_max, ftz=True)
        corr_scale0 = cute.math.exp2(
            self.scale_softmax_log2 * (old_max_val - new_max),
            fastmath=True,
        )
        corr_scale1 = cute.math.exp2(
            self.scale_softmax_log2 * (local_max - new_max),
            fastmath=True,
        )
        partial_vals = _load_partial_o_vec8_as_f32(
            loaded_partial_regs,
            cfg.use_bf16_output and not cfg.use_fp8_output,
        )
        sum_val = sum_val * corr_scale0 + local_sum * corr_scale1
        for pair_idx in cutlass.range_constexpr(4):
            val_base = pair_idx * 2
            folded_pair = ffma2(
                (corr_scale1, corr_scale1),
                (partial_vals[val_base], partial_vals[val_base + 1]),
                fmul2(
                    (corr_scale0, corr_scale0),
                    (output_vals[val_base], output_vals[val_base + 1]),
                ),
            )
            output_vals[val_base] = folded_pair[0]
            output_vals[val_base + 1] = folded_pair[1]
        return output_vals, sum_val, new_max, new_max

    @cute.jit
    def _store_final_o_vec8(
        self,
        final_o_dst,
        output_vals: cutlass.Array,
        norm_scale: Float32,
    ) -> None:
        """Pack one contiguous 8-element output fragment to the final O dtype."""
        cfg = self.cfg
        if cutlass.const_expr(cfg.use_fp8_output):
            final_pairs = cutlass.Array(Float32, 8, space=cutlass.AddressSpace.rmem)
            for pair_idx in cutlass.range_constexpr(4):
                val_base = pair_idx * 2
                pair = fmul2(
                    (norm_scale, norm_scale),
                    (output_vals[val_base], output_vals[val_base + 1]),
                )
                final_pairs[val_base] = pair[0]
                final_pairs[val_base + 1] = pair[1]
            final_fp8_regs = cutlass.Array(Int32, 2, space=cutlass.AddressSpace.rmem)
            final_fp8_regs[0] = _pack_float4_to_fp8_e4m3(
                final_pairs[0],
                final_pairs[1],
                final_pairs[2],
                final_pairs[3],
            )
            final_fp8_regs[1] = _pack_float4_to_fp8_e4m3(
                final_pairs[4],
                final_pairs[5],
                final_pairs[6],
                final_pairs[7],
            )
            final_o_dst.store(
                final_fp8_regs.data_ptr().load(count=2, alignment=4),
                alignment=8,
            )
        else:
            final_regs = cutlass.Array(Int32, 4, space=cutlass.AddressSpace.rmem)
            for reg_idx in cutlass.range_constexpr(4):
                pair = fmul2(
                    (norm_scale, norm_scale),
                    (
                        output_vals[reg_idx * 2],
                        output_vals[reg_idx * 2 + 1],
                    ),
                )
                if cutlass.const_expr(cfg.use_bf16_output):
                    final_regs[reg_idx] = _pack_float2_to_bf16(pair[0], pair[1])
                else:
                    final_regs[reg_idx] = _pack_float2_to_fp16(pair[0], pair[1])
            final_o_dst.store(
                final_regs.data_ptr().load(count=4, alignment=4),
                alignment=16,
            )

    @cute.jit
    def _store_softmax_normalized_o_vec8(
        self,
        logical_h_k_idx: Int32,
        logical_b_idx: Int32,
        reduce_row_idx: Int32,
        reduce_col_idx: Int32,
        output_vals: cutlass.Array,
        sum_val: Float32,
        max_val: Float32,
    ) -> None:
        """Normalize one O fragment, apply attention sink, and store to GMEM."""
        cfg = self.cfg
        attention_sink_h_r = _attention_sink_head_stride(cfg, self.h_r)
        attention_sink_head_idx = _local_head_from_q_output_row(
            cfg, self.h_r, reduce_row_idx
        )
        sum_val += _attention_sink_for_local_head(
            cfg,
            self.attention_sinks_ptr,
            self.scale_softmax_log2,
            max_val,
            logical_h_k_idx,
            attention_sink_h_r,
            self.num_heads_kv,
            attention_sink_head_idx,
        )
        # ``output_scale`` is the public bmm2_scale. Fold it into the final
        # normalization for every output dtype; split partials reach this
        # helper only after the cross-CTA reduction has completed.
        norm_scale = self.output_scale * self._safe_norm_rcp(sum_val)
        physical_dst_row_idx = _q_physical_output_row_from_logical(
            cfg,
            self.h_r,
            self.num_heads_kv,
            logical_b_idx,
            logical_h_k_idx,
            reduce_row_idx,
            self.q_token_offset,
        )
        dst_row_base = Int64(physical_dst_row_idx) * Int64(
            cfg.headdim * cfg.o_dtype_bytes
        )
        dst_offset = dst_row_base + Int64(reduce_col_idx) * Int64(cfg.o_dtype_bytes)
        final_o_dst = cutlass.inttoptr(
            self.o_ptr.toint() + dst_offset,
            mem_space=1,
            dtype=Int32,
        )
        self._store_final_o_vec8(final_o_dst, output_vals, norm_scale)

    @cute.jit
    def _warp_reduce_col_group_sum_pair(
        self, sum_pair: tuple[Float32, Float32]
    ) -> tuple[Float32, Float32]:
        """Reduce two scale sums across lanes in one correction column group."""
        _ = self
        for shfl_idx in cutlass.range_constexpr(3):
            shfl_mask = 16 >> shfl_idx
            other_pair = (
                Float32(
                    prims.shfl_sync(
                        thread_mask=0xFFFFFFFF,
                        val=sum_pair[0],
                        offset=shfl_mask,
                        mask_and_clamp=0x1F,
                        kind=prims.Shfl.BFLY,
                    )
                ),
                Float32(
                    prims.shfl_sync(
                        thread_mask=0xFFFFFFFF,
                        val=sum_pair[1],
                        offset=shfl_mask,
                        mask_and_clamp=0x1F,
                        kind=prims.Shfl.BFLY,
                    )
                ),
            )
            sum_pair = fadd2(sum_pair, other_pair)
        return sum_pair

    def _resolve_o_smem_offset(self) -> int | None:
        """Return the aliased SmemP offset used for final O staging, if valid."""
        # Non-persistent kernels can reuse the two contiguous SmemP allocations
        # as final O staging. Persistent kernels allocate separate O staging to
        # avoid overlap with the next work tile's P producer.
        if not self._can_alias_o_smem():
            return None
        if self.smem_p0_ref is None or self.smem_p1_ref is None:
            return None
        p0_alloc = getattr(self.smem_p0_ref, "_alloc", None)
        p1_alloc = getattr(self.smem_p1_ref, "_alloc", None)
        if p0_alloc is None or p1_alloc is None:
            return None
        lo_alloc, hi_alloc = (p0_alloc, p1_alloc)
        if hi_alloc.offset < lo_alloc.offset:
            lo_alloc, hi_alloc = hi_alloc, lo_alloc
        if hi_alloc.offset != lo_alloc.offset + lo_alloc.size_bytes:
            raise ValueError(
                "TmemCorrResource expected contiguous SmemP allocations for O aliasing"
            )
        return lo_alloc.offset

    @cute.jit
    def create_function_variables(
        self, context: ResourceContext | None = None
    ) -> ResourceVars:
        """Bind epilogue SMEM buffers and initialize cluster reduction barriers."""
        alias_offset = self._resolve_o_smem_offset()
        o_stage_dtype_bytes = (
            2
            if self.cfg.use_split_kv and self.cfg.use_fp8_output
            else self.cfg.o_dtype_bytes
        )
        if cutlass.const_expr(
            context is not None
            and context.smem_base is not None
            and (alias_offset is not None or self._alloc is not None)
        ):
            # SMEM O staging is addressed as int32 because both stmatrix output
            # and vectorized GMEM stores move packed 16-byte fragments.
            self._smem_base_o_i32 = cutlass.Array(
                context.smem_base.data_ptr()
                + (alias_offset if alias_offset is not None else self._alloc.offset),
                dtype=Int32,
                shape=(
                    self.cfg.tile_size_q * self.cfg.headdim * o_stage_dtype_bytes // 4,
                ),
                addrspace=3,
            )
        if cutlass.const_expr(
            context is not None
            and context.smem_base is not None
            and self._sum_alloc is not None
        ):
            # Sum scratch combines the four correction warps' denominator
            # partials before normalization.
            self._sum_scratch = cutlass.Array(
                context.smem_base.data_ptr() + self._sum_alloc.offset,
                dtype=Float32,
                shape=(self._sum_alloc.size_bytes // 4,),
                addrspace=3,
            )
        if cutlass.const_expr(
            context is not None
            and context.smem_base is not None
            and self._gmem_reducer_rank_alloc is not None
        ):
            self._gmem_reducer_rank = cutlass.Array(
                context.smem_base.data_ptr() + self._gmem_reducer_rank_alloc.offset,
                dtype=Int32,
                shape=(1,),
                addrspace=3,
            )
        if cutlass.const_expr(
            context is not None
            and context.smem_base is not None
            and self._cluster_partial_o_alloc is not None
        ):
            # cluster distributed-SMEM partial-O staging, addressed as int32 because
            # peers deliver packed 16-byte fp16 fragments via prims.mapa.
            self._cluster_partial_o_i32 = cutlass.Array(
                context.smem_base.data_ptr() + self._cluster_partial_o_alloc.offset,
                dtype=Int32,
                shape=(self._cluster_partial_o_alloc.size_bytes // 4,),
                addrspace=3,
            )
        if cutlass.const_expr(
            context is not None
            and context.smem_base is not None
            and self._cluster_partial_stats_alloc is not None
        ):
            # cluster distributed-SMEM stats staging (float2 max/sum per owned row
            # and split).
            self._cluster_partial_stats = cutlass.Array(
                context.smem_base.data_ptr() + self._cluster_partial_stats_alloc.offset,
                dtype=Float32,
                shape=(self._cluster_partial_stats_alloc.size_bytes // 4,),
                addrspace=3,
            )
        if cutlass.const_expr(
            context is not None
            and context.smem_base is not None
            and self._cluster_mbarrier_alloc is not None
        ):
            # Per-owner transaction mbarrier, initialised once to expect the full
            # cross-split partial byte count before any peer async-stores into it
            # (the cluster_arrive/cluster_wait at kernel start makes it visible).
            self._cluster_mbarrier = cutlass.Array(
                context.smem_base.data_ptr() + self._cluster_mbarrier_alloc.offset,
                dtype=cutlass.Int64,
                shape=(1,),
                addrspace=3,
            )
            self._cluster_init_transaction_barrier(
                self._cluster_mbarrier.data_ptr(),
                self.cfg.cluster_transaction_bytes,
            )
        return {}

    @producer_work(work_attrs=WorkAttr.AUXILIARY)
    @cute.jit
    def init_epilogue_state(self, stage_info: StageInfo) -> None:
        """Preserve the correction init schedule slot after eager SMEM binding."""
        # ProdAuxWork: function variables are materialized before TaskManager.run()
        # so cluster mbarriers are visible before peer async stores. Keep this as
        # a captured-schedule placeholder for existing task structure.
        return

    @cute.jit
    def _sync_gmem_split_reducers(
        self,
        counter_group_idx: Int32,
        splits_kv: Int32,
        warp_grp_thread_idx: Int32,
    ) -> Int32:
        """Return the arrival-ranked owner index for GMEM split-KV reduction."""
        cfg = self.cfg
        # All correction lanes must finish publishing this CTA's partial O/stats
        # before lane 0 participates in the global completion counter.
        prims.barrier_cta_sync(
            self.store_barrier_id,
            thread_count=cfg.correction_barrier_threads,
        )
        counter_ptr = cutlass.inttoptr(
            self.split_kv_counter_ptr.toint()
            + cutlass.Int64(counter_group_idx * Int32(4)),
            mem_space=1,
            dtype=Uint32,
        )
        if warp_grp_thread_idx == Int32(0):
            inc_limit = prims.mov_b32(splits_kv - Int32(1), target_type=Uint32)
            # Global wrapping INC counts split-CTA completion. Map arrivals in
            # reverse order so the last publisher becomes owner 0 instead of
            # handing reduction back to a fixed logical CTA. Non-owners can
            # leave immediately. If multiple row-slice owners are needed, only
            # the earlier owner arrivals wait for the last arrival to publish
            # every split and wrap the counter to zero.
            old_complete_u32 = prims.atomicrmw(
                prims.AtomicOp.INC,
                counter_ptr,
                inc_limit,
                mem_order=prims.MemOrder.ACQ_REL,
                syncscope=prims.MemScope.GPU,
            )
            old_complete = prims.mov_b32(old_complete_u32, target_type=Int32)
            reducer_cta_idx = splits_kv - Int32(1) - old_complete
            if reducer_cta_idx < Int32(
                cfg.cluster_reduction_num_owner_ctas
            ) and old_complete < splits_kv - Int32(1):
                while prims.load_ext(
                    counter_ptr,
                    order=prims.MemOrder.ACQUIRE,
                    scope=prims.MemScope.GPU,
                ) != Uint32(0):
                    pass
            self._gmem_reducer_rank.store(reducer_cta_idx, 0, alignment=4)
        # Broadcast the arrival-ranked owner and, for owner CTAs, make the
        # all-arrived state visible before reading split partials.
        prims.barrier_cta_sync(
            self.store_barrier_id,
            thread_count=cfg.correction_barrier_threads,
        )
        return self._gmem_reducer_rank.load(0, alignment=4)

    @cute.jit
    def _reduce_and_store_gmem_split_o_vec8(
        self,
        logical_h_k_idx: Int32,
        logical_b_idx: Int32,
        logical_kv_idx: Int32,
        splits_kv: Int32,
        reduce_row_idx: Int32,
        reduce_col_idx: Int32,
        valid_reduce_row,
        *,
        full_prefix: Constexpr[bool],
    ) -> None:
        """Reduce one GMEM split-KV O fragment and write the final output."""
        cfg = self.cfg
        reduction_splits_kv = splits_kv
        if cutlass.const_expr(full_prefix):
            reduction_splits_kv = Int32(cfg.splits_kv)
        output_vals = self._zero_o_vec8()
        sum_val = Float32(0.0)
        old_max_val = _neg_max_f32()
        max_val = _neg_max_f32()
        partial_max = cutlass.Array(Float32, 4, space=cutlass.AddressSpace.rmem)
        partial_sum = cutlass.Array(Float32, 4, space=cutlass.AddressSpace.rmem)
        partial_regs = cutlass.Array(Int32, 16, space=cutlass.AddressSpace.rmem)
        for split_base_i in cutlass.range_constexpr(0, cfg.max_splits_kv, 4):
            split_base = Int32(split_base_i)
            # Load up to four split partials first. Keeping the stats and O
            # loads grouped lets the fold loop below operate on registers only.
            for jj in cutlass.range_constexpr(4):
                if cutlass.const_expr(full_prefix):
                    split_idx = Int32(split_base_i + jj)
                    valid_split_idx = cutlass.const_expr(
                        split_base_i + jj < cfg.splits_kv
                    )
                else:
                    split_idx = split_base + Int32(jj)
                    valid_split_idx = split_idx < reduction_splits_kv
                    if cutlass.const_expr(cfg.max_splits_kv % 4 != 0):
                        split_idx = cute.math.min(
                            split_idx, reduction_splits_kv - Int32(1)
                        )
                if valid_split_idx and valid_reduce_row:
                    stats_offset = (
                        (logical_kv_idx * Int32(cfg.max_splits_kv) + split_idx)
                        * self.h_r
                        + reduce_row_idx
                    ) * Int32(2)
                    stats_src = cutlass.inttoptr(
                        self.partial_stats_ptr.toint()
                        + cutlass.Int64(stats_offset * Int32(4)),
                        mem_space=1,
                        dtype=Float32,
                    )
                    stats_pair = stats_src.load(count=2, alignment=8)
                    partial_max[jj] = stats_pair[0]
                    partial_sum[jj] = stats_pair[1]
                    partial_o_offset = (
                        (logical_kv_idx * Int32(cfg.max_splits_kv) + split_idx)
                        * self.h_r
                        + reduce_row_idx
                    ) * Int32(cfg.headdim * 2) + (reduce_col_idx << Int32(1))
                    partial_o_src = cutlass.inttoptr(
                        self.partial_o_ptr.toint() + cutlass.Int64(partial_o_offset),
                        mem_space=1,
                        dtype=Int32,
                    )
                    loaded_partial_regs = partial_o_src.load(count=4, alignment=16)
                    for reg_idx in cutlass.range_constexpr(4):
                        partial_regs[jj * 4 + reg_idx] = loaded_partial_regs[reg_idx]

            # Fold the loaded partials with the online log-sum-exp recurrence:
            # update max/sum and rescale the accumulated O vector each time a
            # split contributes a larger max.
            for jj in cutlass.range_constexpr(4):
                valid_split_for_apply = split_base + Int32(jj) < reduction_splits_kv
                if cutlass.const_expr(full_prefix):
                    valid_split_for_apply = cutlass.const_expr(
                        split_base_i + jj < cfg.splits_kv
                    )
                if valid_split_for_apply and valid_reduce_row:
                    regs_base = jj * 4
                    loaded_partial_regs = cutlass.Vector.from_elements(
                        (
                            partial_regs[regs_base],
                            partial_regs[regs_base + 1],
                            partial_regs[regs_base + 2],
                            partial_regs[regs_base + 3],
                        ),
                        Int32,
                    )
                    output_vals, sum_val, old_max_val, max_val = (
                        self._fold_split_o_vec8(
                            output_vals,
                            sum_val,
                            old_max_val,
                            max_val,
                            partial_max[jj],
                            partial_sum[jj],
                            loaded_partial_regs,
                        )
                    )

        if valid_reduce_row:
            self._store_softmax_normalized_o_vec8(
                logical_h_k_idx,
                logical_b_idx,
                reduce_row_idx,
                reduce_col_idx,
                output_vals,
                sum_val,
                max_val,
            )

    @cute.jit
    def _reduce_fused_gmem_partial_segment(
        self,
        logical_h_k_idx: Int32,
        logical_b_idx: Int32,
        logical_kv_idx: Int32,
        q_row_offset: Int32,
        logical_q_group_idx: Int32,
        reducer_cta_idx: Int32,
        splits_kv: Int32,
        warp_grp_thread_idx: Int32,
        rows_per_reducer: Int32,
        segment_idx: Int32,
        *,
        full_prefix: Constexpr[bool],
    ) -> None:
        """Reduce one correction-warpgroup slice from fused GMEM partials."""
        cfg = self.cfg
        partial_row_bytes = Int32(cfg.headdim * 2)
        reduce_base_offset = warp_grp_thread_idx * Int32(16) + Int32(
            segment_idx * cfg.split_reduction_slice_bytes
        )
        reduce_tile_row_idx = reduce_base_offset // partial_row_bytes
        reduce_local_row_idx = reducer_cta_idx * rows_per_reducer + reduce_tile_row_idx
        reduce_output_row_idx = q_row_offset + reduce_local_row_idx
        reduce_col_idx = (reduce_base_offset % partial_row_bytes) >> Int32(1)
        valid_reduce_row = (
            reduce_tile_row_idx < rows_per_reducer
            and _q_row_is_valid_for_seq(
                cfg,
                self.h_r,
                logical_q_group_idx,
                reduce_local_row_idx,
                self.seq_len_q,
            )
        )
        self._reduce_and_store_gmem_split_o_vec8(
            logical_h_k_idx,
            logical_b_idx,
            logical_kv_idx,
            splits_kv,
            reduce_output_row_idx,
            reduce_col_idx,
            valid_reduce_row,
            full_prefix=full_prefix,
        )

    @cute.jit
    def _reduce_fused_gmem_partials_impl(
        self,
        logical_h_k_idx: Int32,
        logical_b_idx: Int32,
        logical_kv_idx: Int32,
        q_row_offset: Int32,
        logical_q_group_idx: Int32,
        counter_group_idx: Int32,
        splits_kv: Int32,
        warp_grp_thread_idx: Int32,
        *,
        full_prefix: Constexpr[bool],
    ) -> None:
        """Elect fused-GMEM owners and run static or contracted geometry."""
        cfg = self.cfg
        reduction_splits_kv = splits_kv
        if cutlass.const_expr(full_prefix):
            reduction_splits_kv = Int32(cfg.splits_kv)
        reducer_cta_idx = self._sync_gmem_split_reducers(
            counter_group_idx,
            reduction_splits_kv,
            warp_grp_thread_idx,
        )

        if reducer_cta_idx < Int32(cfg.cluster_reduction_num_owner_ctas):
            if cutlass.const_expr(full_prefix):
                rows_per_reducer = Int32(cfg.cluster_reduction_rows_per_cta)
                for segment_idx in cutlass.range_constexpr(
                    cfg.split_reduction_slices_per_cta
                ):
                    self._reduce_fused_gmem_partial_segment(
                        logical_h_k_idx,
                        logical_b_idx,
                        logical_kv_idx,
                        q_row_offset,
                        logical_q_group_idx,
                        reducer_cta_idx,
                        reduction_splits_kv,
                        warp_grp_thread_idx,
                        rows_per_reducer,
                        Int32(segment_idx),
                        full_prefix=True,
                    )
            else:
                rows_per_reducer = self._split_reduction_rows_per_cta(
                    Int32(cfg.tile_size_q), reduction_splits_kv
                )
                runtime_segments = rows_per_reducer // Int32(
                    cfg.split_reduction_rows_per_slice
                )
                for segment_idx in cutlass.range(runtime_segments, unroll=1):
                    self._reduce_fused_gmem_partial_segment(
                        logical_h_k_idx,
                        logical_b_idx,
                        logical_kv_idx,
                        q_row_offset,
                        logical_q_group_idx,
                        reducer_cta_idx,
                        reduction_splits_kv,
                        warp_grp_thread_idx,
                        rows_per_reducer,
                        segment_idx,
                        full_prefix=False,
                    )

    @cute.jit
    def _reduce_fused_gmem_partials(
        self,
        logical_h_k_idx: Int32,
        logical_b_idx: Int32,
        logical_kv_idx: Int32,
        q_row_offset: Int32,
        logical_q_group_idx: Int32,
        counter_group_idx: Int32,
        splits_kv: Int32,
        warp_grp_thread_idx: Int32,
    ) -> None:
        """Select the full-prefix or contracted fused-GMEM reducer."""
        cfg = self.cfg
        if cutlass.const_expr(self.static_full_split_prefix):
            self._reduce_fused_gmem_partials_impl(
                logical_h_k_idx,
                logical_b_idx,
                logical_kv_idx,
                q_row_offset,
                logical_q_group_idx,
                counter_group_idx,
                Int32(cfg.splits_kv),
                warp_grp_thread_idx,
                full_prefix=True,
            )
            return
        # The producer prefix is CTA-uniform. The full path therefore keeps
        # every correction-barrier participant on the same static schedule.
        if splits_kv == Int32(cfg.splits_kv):
            self._reduce_fused_gmem_partials_impl(
                logical_h_k_idx,
                logical_b_idx,
                logical_kv_idx,
                q_row_offset,
                logical_q_group_idx,
                counter_group_idx,
                Int32(cfg.splits_kv),
                warp_grp_thread_idx,
                full_prefix=True,
            )
        else:
            self._reduce_fused_gmem_partials_impl(
                logical_h_k_idx,
                logical_b_idx,
                logical_kv_idx,
                q_row_offset,
                logical_q_group_idx,
                counter_group_idx,
                splits_kv,
                warp_grp_thread_idx,
                full_prefix=False,
            )

    @cute.jit
    def _reduce_cluster_partial_segment(
        self,
        logical_h_k_idx: Int32,
        logical_b_idx: Int32,
        q_row_offset: Int32,
        logical_q_group_idx: Int32,
        cta_idx_kv: Int32,
        splits_kv: Int32,
        warp_grp_thread_idx: Int32,
        rows_per_cta: Int32,
        segment_idx: Int32,
        *,
        full_prefix: Constexpr[bool],
    ) -> None:
        """Reduce one correction-warpgroup slice from cluster partial SMEM."""
        cfg = self.cfg
        reduction_splits_kv = splits_kv
        if cutlass.const_expr(full_prefix):
            reduction_splits_kv = Int32(cfg.splits_kv)
        row_bytes = Int32(cfg.headdim * 2)
        reduce_base_offset = (
            warp_grp_thread_idx * Int32(16)
            + cta_idx_kv * rows_per_cta * row_bytes
            + Int32(segment_idx * cfg.split_reduction_slice_bytes)
        )
        reduce_row_idx = reduce_base_offset // row_bytes
        reduce_col_idx = (reduce_base_offset % row_bytes) >> Int32(1)
        valid_reduce_row = self._cluster_reduction_row_is_owned(
            cta_idx_kv, reduce_row_idx, reduction_splits_kv
        ) and _q_row_is_valid_for_seq(
            cfg,
            self.h_r,
            logical_q_group_idx,
            reduce_row_idx,
            self.seq_len_q,
        )
        local_row_idx = self._cluster_reduction_local_row(
            reduce_row_idx, reduction_splits_kv
        )
        output_vals = self._zero_o_vec8()
        sum_val = Float32(0.0)
        old_max_val = _neg_max_f32()
        max_val = _neg_max_f32()
        partial_max = cutlass.Array(Float32, 4, space=cutlass.AddressSpace.rmem)
        partial_sum = cutlass.Array(Float32, 4, space=cutlass.AddressSpace.rmem)
        partial_regs = cutlass.Array(Int32, 16, space=cutlass.AddressSpace.rmem)
        for split_base_i in cutlass.range_constexpr(0, cfg.max_splits_kv, 4):
            split_base = Int32(split_base_i)
            # Load up to four split partials first. The full-prefix
            # specialization resolves every slot predicate at compile time;
            # contracted prefixes retain the runtime bound and safe clamp.
            for jj in cutlass.range_constexpr(4):
                if cutlass.const_expr(full_prefix):
                    split_idx = Int32(split_base_i + jj)
                    valid_split_idx = cutlass.const_expr(
                        split_base_i + jj < cfg.splits_kv
                    )
                else:
                    split_idx = split_base + Int32(jj)
                    valid_split_idx = split_idx < splits_kv
                    if cutlass.const_expr(cfg.max_splits_kv % 4 != 0):
                        split_idx = cute.math.min(split_idx, splits_kv - Int32(1))
                if valid_split_idx and valid_reduce_row:
                    partial_row_idx = self._cluster_partial_row_idx(
                        split_idx, local_row_idx, reduction_splits_kv
                    )
                    stats_pair = (
                        self._cluster_partial_stats.subview(
                            self._cluster_partial_stats_offset(partial_row_idx)
                        )
                        .data_ptr()
                        .load(count=2, alignment=8)
                    )
                    partial_max[jj] = stats_pair[0]
                    partial_sum[jj] = stats_pair[1]
                    loaded_partial_regs = (
                        self._cluster_partial_o_i32.subview(
                            self._cluster_partial_o_i32_offset(
                                partial_row_idx,
                                reduce_col_idx << Int32(1),
                            )
                        )
                        .data_ptr()
                        .load(count=4, alignment=16)
                    )
                    for reg_idx in cutlass.range_constexpr(4):
                        partial_regs[jj * 4 + reg_idx] = loaded_partial_regs[reg_idx]

            # Preserve online-softmax arithmetic order after grouped loads.
            for jj in cutlass.range_constexpr(4):
                valid_split_for_apply = split_base + Int32(jj) < splits_kv
                if cutlass.const_expr(full_prefix):
                    valid_split_for_apply = cutlass.const_expr(
                        split_base_i + jj < cfg.splits_kv
                    )
                if valid_split_for_apply and valid_reduce_row:
                    regs_base = jj * 4
                    loaded_partial_regs = cutlass.Vector.from_elements(
                        (
                            partial_regs[regs_base],
                            partial_regs[regs_base + 1],
                            partial_regs[regs_base + 2],
                            partial_regs[regs_base + 3],
                        ),
                        Int32,
                    )
                    output_vals, sum_val, old_max_val, max_val = (
                        self._fold_split_o_vec8(
                            output_vals,
                            sum_val,
                            old_max_val,
                            max_val,
                            partial_max[jj],
                            partial_sum[jj],
                            loaded_partial_regs,
                        )
                    )
        if valid_reduce_row:
            self._store_softmax_normalized_o_vec8(
                logical_h_k_idx,
                logical_b_idx,
                q_row_offset + reduce_row_idx,
                reduce_col_idx,
                output_vals,
                sum_val,
                max_val,
            )

    @cute.jit
    def _reduce_cluster_partials_impl(
        self,
        logical_h_k_idx: Int32,
        logical_b_idx: Int32,
        q_row_offset: Int32,
        logical_q_group_idx: Int32,
        cta_idx_kv: Int32,
        splits_kv: Int32,
        warp_grp_thread_idx: Int32,
        *,
        full_prefix: Constexpr[bool],
    ) -> None:
        """Wait for cluster publication and run static or contracted geometry."""
        cfg = self.cfg
        reduction_splits_kv = splits_kv
        if cutlass.const_expr(full_prefix):
            reduction_splits_kv = Int32(cfg.splits_kv)
        # Split CTAs always publish their partials, but only CTAs with a
        # physical output-row band participate in the owner-side wait and
        # reduction. Ownership is allocated in correction-warpgroup slices,
        # so structurally empty ranks never wait on an unused local barrier.
        if cta_idx_kv < Int32(cfg.cluster_reduction_num_owner_ctas):
            self._cluster_complete_inactive_row_bytes(
                self._cluster_mbarrier.data_ptr(),
                cta_idx_kv,
                reduction_splits_kv,
                logical_q_group_idx,
                warp_grp_thread_idx,
            )
            self._cluster_wait_transaction_barrier(
                self._cluster_mbarrier.data_ptr(),
                warp_grp_thread_idx,
                self.store_barrier_id,
                cfg.correction_barrier_threads,
            )
            if cutlass.const_expr(full_prefix):
                rows_per_cta = Int32(cfg.cluster_reduction_rows_per_cta)
                for segment_idx in cutlass.range_constexpr(
                    cfg.split_reduction_slices_per_cta
                ):
                    self._reduce_cluster_partial_segment(
                        logical_h_k_idx,
                        logical_b_idx,
                        q_row_offset,
                        logical_q_group_idx,
                        cta_idx_kv,
                        reduction_splits_kv,
                        warp_grp_thread_idx,
                        rows_per_cta,
                        Int32(segment_idx),
                        full_prefix=True,
                    )
            else:
                rows_per_cta = self._cluster_reduction_rows_per_cta(reduction_splits_kv)
                runtime_segments = rows_per_cta // Int32(
                    cfg.split_reduction_rows_per_slice
                )
                for segment_idx in cutlass.range(runtime_segments, unroll=1):
                    self._reduce_cluster_partial_segment(
                        logical_h_k_idx,
                        logical_b_idx,
                        q_row_offset,
                        logical_q_group_idx,
                        cta_idx_kv,
                        reduction_splits_kv,
                        warp_grp_thread_idx,
                        rows_per_cta,
                        segment_idx,
                        full_prefix=False,
                    )

    @cute.jit
    def _stage_fp16_o_regs_to_smem(
        self,
        cfg: Constexpr[FmhaDecodeConfig],
        regs_o: cutlass.Array,
        warp_grp_thread_idx: Int32,
        warp_idx: Int32,
        lane_idx: Int32,
        *,
        output_pair_regs: Constexpr[int],
    ) -> None:
        """Stage packed FP16/BF16 O fragments through the STSM layout.

        Correction lanes hold O as packed register pairs, but final GMEM stores
        need contiguous row segments. This first operation writes the register
        fragments into the swizzled SMEM layout before a second operation
        reloads contiguous vectors.
        """
        if cutlass.const_expr(output_pair_regs < 4):
            smem_offset_bytes, _, _, _ = _fp16_o_reorg_offsets(
                cfg, warp_grp_thread_idx, warp_idx, lane_idx, 0, 0
            )
            smem_dst = self._smem_base_o_i32.subview(
                smem_offset_bytes >> Int32(2)
            ).data_ptr()
            prims.stmatrix(
                smem_dst,
                regs_o.data_ptr().load(count=output_pair_regs, alignment=4),
                prims.MMALayout.COL,
                shape=prims.StoreShape.M8N8,
            )
        else:
            for stsm_group_idx in cutlass.range_constexpr(output_pair_regs // 4):
                smem_offset_bytes, _, _, _ = _fp16_o_reorg_offsets(
                    cfg,
                    warp_grp_thread_idx,
                    warp_idx,
                    lane_idx,
                    stsm_group_idx,
                    0,
                )
                smem_dst = self._smem_base_o_i32.subview(
                    smem_offset_bytes >> Int32(2)
                ).data_ptr()
                prims.stmatrix(
                    smem_dst,
                    (regs_o.data_ptr() + stsm_group_idx * 4).load(
                        count=4,
                        alignment=4,
                    ),
                    prims.MMALayout.COL,
                    shape=prims.StoreShape.M8N8,
                )
        cute.arch.fence_view_async_shared()
        # The copy-out phase below reloads SMEM written by all correction
        # warps, so synchronize the correction warpgroup after the STSM stores.
        prims.barrier_cta_sync(
            self.store_barrier_id,
            thread_count=cfg.correction_barrier_threads,
        )

    @cute.jit
    def _copy_staged_fp16_o_segments(
        self,
        cfg: Constexpr[FmhaDecodeConfig],
        logical_kv_idx: Int32,
        cta_idx_kv: Int32,
        splits_kv: Int32,
        row_offset: Int32,
        logical_h_k_idx: Int32,
        logical_b_idx: Int32,
        logical_q_group_idx: Int32,
        warp_grp_thread_idx: Int32,
        warp_idx: Int32,
        lane_idx: Int32,
        *,
        num_copy_segments: Constexpr[int],
        copy_to_partial: Constexpr[bool],
        enable_cluster: Constexpr[bool],
        full_prefix: Constexpr[bool],
    ) -> None:
        """Copy STSM-reorganized FP16/BF16 O fragments to partial or final GMEM.

        Each 2 KiB segment is covered by 128 lanes x 16 bytes. For split-KV the
        destination is either GMEM partial storage or owner-CTA distributed SMEM
        for cluster reduction; otherwise it is the final output tensor.
        """
        publication_splits_kv = splits_kv
        if cutlass.const_expr(enable_cluster and full_prefix):
            publication_splits_kv = Int32(cfg.splits_kv)
        for copy_segment_idx in cutlass.range_constexpr(num_copy_segments):
            _, load_smem_offset, dst_row_idx, dst_col_offset = _fp16_o_reorg_offsets(
                cfg,
                warp_grp_thread_idx,
                warp_idx,
                lane_idx,
                0,
                copy_segment_idx,
            )
            if cutlass.const_expr(copy_to_partial):
                partial_o_row_idx = row_offset + dst_row_idx
                valid_partial_row = _q_row_is_valid_for_seq(
                    cfg,
                    self.h_r,
                    logical_q_group_idx,
                    dst_row_idx,
                    self.seq_len_q,
                )
                if valid_partial_row:
                    smem_src = self._smem_base_o_i32.subview(
                        load_smem_offset >> Int32(2)
                    ).data_ptr()
                    if cutlass.const_expr(enable_cluster):
                        # Peers async-store partial O into the owner CTA's SMEM
                        # and signal its transaction mbarrier.
                        cluster_owner = self._cluster_reduction_cta_for_row(
                            dst_row_idx, publication_splits_kv
                        )
                        cluster_local = self._cluster_reduction_local_row(
                            dst_row_idx, publication_splits_kv
                        )
                        partial_o_dst = prims.mapa(
                            self._cluster_partial_o_i32.subview(
                                self._cluster_partial_o_i32_offset(
                                    self._cluster_partial_row_idx(
                                        cta_idx_kv,
                                        cluster_local,
                                        publication_splits_kv,
                                    ),
                                    dst_col_offset,
                                )
                            ).data_ptr(),
                            cluster_owner,
                        )
                        self._cluster_store_async_vec4_i32(
                            partial_o_dst,
                            smem_src.load(count=4, alignment=16),
                            prims.mapa(
                                self._cluster_mbarrier.data_ptr(),
                                cluster_owner,
                            ),
                        )
                    else:
                        partial_o_row_base = self._gmem_partial_row_offset(
                            logical_kv_idx,
                            cta_idx_kv,
                            partial_o_row_idx,
                        ) * Int64(cfg.headdim * 2)
                        partial_o_dst = cutlass.inttoptr(
                            self.partial_o_ptr.toint()
                            + partial_o_row_base
                            + Int64(dst_col_offset),
                            mem_space=1,
                            dtype=Int32,
                        )
                        partial_o_dst.store(
                            smem_src.load(count=4, alignment=16),
                            alignment=16,
                        )
            else:
                if _q_row_is_valid_for_seq(
                    cfg,
                    self.h_r,
                    logical_q_group_idx,
                    dst_row_idx,
                    self.seq_len_q,
                ):
                    smem_src = self._smem_base_o_i32.subview(
                        load_smem_offset >> Int32(2)
                    ).data_ptr()
                    physical_dst_row_idx = _q_physical_output_row(
                        cfg,
                        self.h_r,
                        self.num_heads_kv,
                        logical_b_idx,
                        logical_h_k_idx,
                        logical_q_group_idx,
                        dst_row_idx,
                        self.q_token_offset,
                    )
                    dst_row_base = Int64(physical_dst_row_idx) * Int64(
                        cfg.headdim * cfg.o_dtype_bytes
                    )
                    dst_ptr = cutlass.inttoptr(
                        self.o_ptr.toint() + dst_row_base + Int64(dst_col_offset),
                        mem_space=1,
                        dtype=Int32,
                    )
                    dst_ptr.store(smem_src.load(count=4, alignment=16), alignment=16)

    @cute.jit
    def _copy_staged_fp16_o_to_partial(
        self,
        cfg: Constexpr[FmhaDecodeConfig],
        logical_kv_idx: Int32,
        cta_idx_kv: Int32,
        splits_kv: Int32,
        row_offset: Int32,
        logical_q_group_idx: Int32,
        warp_grp_thread_idx: Int32,
        warp_idx: Int32,
        lane_idx: Int32,
        *,
        num_copy_segments: Constexpr[int],
        enable_cluster: Constexpr[bool],
        full_prefix: Constexpr[bool],
    ) -> None:
        """Publish staged partial O to GMEM, or owner CTA SMEM for cluster reduction."""
        self._copy_staged_fp16_o_segments(
            cfg,
            logical_kv_idx,
            cta_idx_kv,
            splits_kv,
            row_offset,
            Int32(0),
            Int32(0),
            logical_q_group_idx,
            warp_grp_thread_idx,
            warp_idx,
            lane_idx,
            num_copy_segments=num_copy_segments,
            copy_to_partial=True,
            enable_cluster=enable_cluster,
            full_prefix=full_prefix,
        )

    @cute.jit
    def _copy_staged_fp16_o_to_output(
        self,
        cfg: Constexpr[FmhaDecodeConfig],
        logical_h_k_idx: Int32,
        logical_b_idx: Int32,
        logical_q_group_idx: Int32,
        warp_grp_thread_idx: Int32,
        warp_idx: Int32,
        lane_idx: Int32,
        *,
        num_copy_segments: Constexpr[int],
    ) -> None:
        """Copy staged FP16/BF16 O fragments to the final output tensor."""
        self._copy_staged_fp16_o_segments(
            cfg,
            Int32(0),
            Int32(0),
            Int32(1),
            Int32(0),
            logical_h_k_idx,
            logical_b_idx,
            logical_q_group_idx,
            warp_grp_thread_idx,
            warp_idx,
            lane_idx,
            num_copy_segments=num_copy_segments,
            copy_to_partial=False,
            enable_cluster=False,
            full_prefix=False,
        )

    @cute.jit
    def _stage_and_copy_swaps_partial_o(
        self,
        cfg: Constexpr[FmhaDecodeConfig],
        regs_partial_o: cutlass.Array,
        logical_kv_idx: Int32,
        cta_idx_kv: Int32,
        splits_kv: Int32,
        row_offset: Int32,
        logical_q_group_idx: Int32,
        warp_grp_thread_idx: Int32,
        warp_idx: Int32,
        lane_idx: Int32,
        *,
        output_pair_regs: Constexpr[int],
        num_copy_segments: Constexpr[int],
        enable_cluster: Constexpr[bool],
        full_prefix: Constexpr[bool],
    ) -> None:
        """Stage Swaps partial O through SMEM and publish it for split-KV."""
        # Operation 1: reformat per-lane O registers into contiguous row
        # fragments via SMEM. Operation 2: publish those fragments to the
        # selected split-KV staging backend.
        self._stage_fp16_o_regs_to_smem(
            cfg,
            regs_partial_o,
            warp_grp_thread_idx,
            warp_idx,
            lane_idx,
            output_pair_regs=output_pair_regs,
        )
        self._copy_staged_fp16_o_to_partial(
            cfg,
            logical_kv_idx,
            cta_idx_kv,
            splits_kv,
            row_offset,
            logical_q_group_idx,
            warp_grp_thread_idx,
            warp_idx,
            lane_idx,
            num_copy_segments=num_copy_segments,
            enable_cluster=enable_cluster,
            full_prefix=full_prefix,
        )

    @cute.jit
    def _store_split_stats_from_scale_arrays(
        self,
        cfg: Constexpr[FmhaDecodeConfig],
        logical_kv_idx: Int32,
        cta_idx_kv: Int32,
        splits_kv: Int32,
        row_offset: Int32,
        logical_q_group_idx: Int32,
        warp_idx: Int32,
        lane_idx: Int32,
        final_max: cutlass.Array,
        reduced_sum: cutlass.Array,
        *,
        num_scale_groups: Constexpr[int],
        enable_cluster: Constexpr[bool],
        full_prefix: Constexpr[bool],
    ) -> None:
        """Publish split-KV state for Swaps tail epilogues.

        Separate-GMEM writes one FP32 log2-LSE value per row. Fused GMEM/cluster
        writes a float2(max, sum) record.
        """
        publication_splits_kv = splits_kv
        if cutlass.const_expr(enable_cluster and full_prefix):
            publication_splits_kv = Int32(cfg.splits_kv)
        # Only one warp publishes per-row state. The lane mapping keeps each
        # record next to the matching partial-O row.
        if warp_idx == Int32(0) and lane_idx < Int32(4 * num_scale_groups):
            stats_idx = lane_idx >> Int32(2)
            quad_thread_idx = lane_idx & Int32(3)
            stats_row_base = (
                quad_thread_idx * Int32(2)
                + ((stats_idx >> Int32(1)) * Int32(8))
                + (stats_idx & Int32(1))
            )
            stats_row_idx = row_offset + stats_row_base
            if _q_row_is_valid_for_seq(
                cfg,
                self.h_r,
                logical_q_group_idx,
                stats_row_base,
                self.seq_len_q,
            ):
                if cutlass.const_expr(cfg.use_separate_reduction_kernel):
                    lse_row = self._gmem_partial_row_offset(
                        logical_kv_idx,
                        cta_idx_kv,
                        stats_row_idx,
                    )
                    lse_ptr = cutlass.inttoptr(
                        self.partial_stats_ptr.toint() + lse_row * Int64(4),
                        mem_space=1,
                        dtype=Float32,
                    )
                    lse_ptr.store(
                        self._separate_partial_lse(
                            final_max[stats_idx], reduced_sum[stats_idx]
                        ),
                        alignment=4,
                    )
                elif cutlass.const_expr(enable_cluster):
                    cluster_owner = self._cluster_reduction_cta_for_row(
                        stats_row_base, publication_splits_kv
                    )
                    cluster_local = self._cluster_reduction_local_row(
                        stats_row_base, publication_splits_kv
                    )
                    stats_ptr = prims.mapa(
                        self._cluster_partial_stats.subview(
                            self._cluster_partial_stats_offset(
                                self._cluster_partial_row_idx(
                                    cta_idx_kv,
                                    cluster_local,
                                    publication_splits_kv,
                                )
                            )
                        ).data_ptr(),
                        cluster_owner,
                    )
                    self._cluster_store_async_vec2_f32(
                        stats_ptr,
                        final_max[stats_idx],
                        reduced_sum[stats_idx],
                        prims.mapa(
                            self._cluster_mbarrier.data_ptr(),
                            cluster_owner,
                        ),
                    )
                else:
                    stats_base = (
                        (logical_kv_idx * Int32(cfg.max_splits_kv) + cta_idx_kv)
                        * self.h_r
                        + stats_row_idx
                    ) * Int32(2)
                    stats_ptr = cutlass.inttoptr(
                        self.partial_stats_ptr.toint()
                        + cutlass.Int64(stats_base * Int32(4)),
                        mem_space=1,
                        dtype=Float32,
                    )
                    stats_ptr.store(
                        cutlass.Vector.from_elements(
                            (final_max[stats_idx], reduced_sum[stats_idx]),
                            Float32,
                        ),
                        alignment=8,
                    )

    @cute.jit
    def _publish_and_reduce_cluster_swaps_partials_impl(
        self,
        regs_partial_o: cutlass.Array,
        logical_h_k_idx: Int32,
        logical_b_idx: Int32,
        logical_kv_idx: Int32,
        q_row_offset: Int32,
        logical_q_group_idx: Int32,
        cta_idx_kv: Int32,
        splits_kv: Int32,
        warp_grp_thread_idx: Int32,
        warp_idx: Int32,
        lane_idx: Int32,
        final_max: cutlass.Array,
        reduced_sum: cutlass.Array,
        *,
        output_pair_regs: Constexpr[int],
        num_copy_segments: Constexpr[int],
        num_scale_groups: Constexpr[int],
        full_prefix: Constexpr[bool],
    ) -> None:
        """Publish and reduce one static or contracted cluster split prefix."""
        cfg = self.cfg
        publication_splits_kv = splits_kv
        if cutlass.const_expr(full_prefix):
            publication_splits_kv = Int32(cfg.splits_kv)
        self._stage_and_copy_swaps_partial_o(
            cfg,
            regs_partial_o,
            logical_kv_idx,
            cta_idx_kv,
            publication_splits_kv,
            q_row_offset,
            logical_q_group_idx,
            warp_grp_thread_idx,
            warp_idx,
            lane_idx,
            output_pair_regs=output_pair_regs,
            num_copy_segments=num_copy_segments,
            enable_cluster=True,
            full_prefix=full_prefix,
        )
        self._store_split_stats_from_scale_arrays(
            cfg,
            logical_kv_idx,
            cta_idx_kv,
            publication_splits_kv,
            q_row_offset,
            logical_q_group_idx,
            warp_idx,
            lane_idx,
            final_max,
            reduced_sum,
            num_scale_groups=num_scale_groups,
            enable_cluster=True,
            full_prefix=full_prefix,
        )
        self._reduce_cluster_partials_impl(
            logical_h_k_idx,
            logical_b_idx,
            q_row_offset,
            logical_q_group_idx,
            cta_idx_kv,
            publication_splits_kv,
            warp_grp_thread_idx,
            full_prefix=full_prefix,
        )

    @cute.jit
    def _publish_and_reduce_cluster_swaps_partials(
        self,
        regs_partial_o: cutlass.Array,
        logical_h_k_idx: Int32,
        logical_b_idx: Int32,
        logical_kv_idx: Int32,
        q_row_offset: Int32,
        logical_q_group_idx: Int32,
        cta_idx_kv: Int32,
        splits_kv: Int32,
        warp_grp_thread_idx: Int32,
        warp_idx: Int32,
        lane_idx: Int32,
        final_max: cutlass.Array,
        reduced_sum: cutlass.Array,
        *,
        output_pair_regs: Constexpr[int],
        num_copy_segments: Constexpr[int],
        num_scale_groups: Constexpr[int],
    ) -> None:
        """Select full-prefix or contracted cluster publication and reduction."""
        cfg = self.cfg
        if cutlass.const_expr(self.static_full_split_prefix):
            self._publish_and_reduce_cluster_swaps_partials_impl(
                regs_partial_o,
                logical_h_k_idx,
                logical_b_idx,
                logical_kv_idx,
                q_row_offset,
                logical_q_group_idx,
                cta_idx_kv,
                Int32(cfg.splits_kv),
                warp_grp_thread_idx,
                warp_idx,
                lane_idx,
                final_max,
                reduced_sum,
                output_pair_regs=output_pair_regs,
                num_copy_segments=num_copy_segments,
                num_scale_groups=num_scale_groups,
                full_prefix=True,
            )
            return
        # Runtime pruning produces one prefix for the complete physical
        # cluster. Every correction lane and every active rank takes the same
        # branch before any distributed-SMEM address or mbarrier operation.
        if splits_kv == Int32(cfg.splits_kv):
            self._publish_and_reduce_cluster_swaps_partials_impl(
                regs_partial_o,
                logical_h_k_idx,
                logical_b_idx,
                logical_kv_idx,
                q_row_offset,
                logical_q_group_idx,
                cta_idx_kv,
                Int32(cfg.splits_kv),
                warp_grp_thread_idx,
                warp_idx,
                lane_idx,
                final_max,
                reduced_sum,
                output_pair_regs=output_pair_regs,
                num_copy_segments=num_copy_segments,
                num_scale_groups=num_scale_groups,
                full_prefix=True,
            )
        else:
            self._publish_and_reduce_cluster_swaps_partials_impl(
                regs_partial_o,
                logical_h_k_idx,
                logical_b_idx,
                logical_kv_idx,
                q_row_offset,
                logical_q_group_idx,
                cta_idx_kv,
                splits_kv,
                warp_grp_thread_idx,
                warp_idx,
                lane_idx,
                final_max,
                reduced_sum,
                output_pair_regs=output_pair_regs,
                num_copy_segments=num_copy_segments,
                num_scale_groups=num_scale_groups,
                full_prefix=False,
            )

    @cute.jit
    def _keeps_tail_epilogue(
        self,
        stage_info: StageInfo,
        *,
        tail_o_stage_idx_0: Int32,
        tail_o_stage_idx_1: Int32,
        inst0_new_max_arr: cutlass.Array,
        inst0_sum_arr: cutlass.Array,
        inst1_new_max_arr: cutlass.Array,
        inst1_sum_arr: cutlass.Array,
        tmem_row_base: Int32,
        o_base_col: Constexpr[int],
        output_pair_regs: Constexpr[int],
        keeps_o_ldst_offset: Constexpr[int],
        row_idx: Int32,
        col_base: Int32,
        warp_grp_thread_idx: Int32,
    ) -> None:
        """Finish the KeepsMmaAb tail stage and optional split-KV reduction."""
        cfg = self.cfg
        # Tail stats are already final per-instance sums and maxima. Combine
        # inst0/inst1 first, then either publish split-KV partials or normalize
        # directly to the output tensor.
        inst0_sum_0 = inst0_sum_arr[0]
        inst0_new_max_0 = inst0_new_max_arr[0]
        inst1_sum_0 = inst1_sum_arr[0]
        inst1_new_max_0 = inst1_new_max_arr[0]

        if cutlass.const_expr(cfg.tile_size_q == 64):
            # The q64 16dp32bitx2 layout keeps the two 64-K half-row sums
            # separate throughout the online recurrence.  Pair them only at
            # finalization, after which both lanes use the same denominator
            # while continuing to own disjoint O-column halves.
            inst0_sum_0 += Float32(
                prims.shfl_sync(
                    thread_mask=0xFFFFFFFF,
                    val=inst0_sum_0,
                    offset=16,
                    mask_and_clamp=0x1F,
                    kind=prims.Shfl.BFLY,
                )
            )
            if cutlass.const_expr(cfg.num_insts_kv != 1):
                inst1_sum_0 += Float32(
                    prims.shfl_sync(
                        thread_mask=0xFFFFFFFF,
                        val=inst1_sum_0,
                        offset=16,
                        mask_and_clamp=0x1F,
                        kind=prims.Shfl.BFLY,
                    )
                )

        if cutlass.const_expr(
            cfg.use_fp8_qkv
            and cfg.use_fp8_output
            and cfg.tile_size_q == 128
            and cfg.headdim == 128
            and cfg.num_insts_kv == 2
            and cfg.max_seq_len_q == 1
            and cfg.has_static_dense_full_kv_tiles
        ):
            final_max_0 = cute.math.max(inst0_new_max_0, inst1_new_max_0, ftz=True)
            exp_scale0_0 = cute.math.exp2(
                self.scale_softmax_log2 * (inst0_new_max_0 - final_max_0),
                fastmath=True,
            )
            exp_scale1_0 = cute.math.exp2(
                self.scale_softmax_log2 * (inst1_new_max_0 - final_max_0),
                fastmath=True,
            )
            reduced_sum_0 = inst0_sum_0 * exp_scale0_0 + inst1_sum_0 * exp_scale1_0
        else:
            uses_inst0 = inst0_new_max_0 != _neg_max_f32()
            uses_inst1 = False
            if cutlass.const_expr(cfg.num_insts_kv != 1):
                uses_inst1 = inst1_new_max_0 != _neg_max_f32()
            final_max_0 = _neg_max_f32()
            if uses_inst0:
                final_max_0 = inst0_new_max_0
            if uses_inst1:
                final_max_0 = cute.math.max(final_max_0, inst1_new_max_0, ftz=True)

            exp_scale0_0 = Float32(0.0)
            exp_scale1_0 = Float32(0.0)
            reduced_sum_0 = Float32(0.0)
            if uses_inst0:
                exp_scale0_0 = cute.math.exp2(
                    self.scale_softmax_log2 * (inst0_new_max_0 - final_max_0),
                    fastmath=True,
                )
                reduced_sum_0 += inst0_sum_0 * exp_scale0_0
            if uses_inst1:
                exp_scale1_0 = cute.math.exp2(
                    self.scale_softmax_log2 * (inst1_new_max_0 - final_max_0),
                    fastmath=True,
                )
                reduced_sum_0 += inst1_sum_0 * exp_scale1_0

        logical_h_k_idx, logical_b_idx = _logical_head_batch(
            stage_info, self.h_k_idx, self.b_idx
        )
        logical_q_group_idx = _logical_q_group_idx(cfg, stage_info, self.q_group_idx)
        # The physical MMA tile may end with structural padding. Scratch rows
        # omit that padding, so advance Q groups by the complete token/head rows
        # represented by the Q tensor map rather than by ``tile_size_q``.
        q_row_offset = _q_tile_output_row_base(cfg, logical_q_group_idx)
        logical_output_row_idx = q_row_offset + row_idx
        valid_output_row = _q_row_is_valid_for_seq(
            cfg,
            self.h_r,
            logical_q_group_idx,
            row_idx,
            self.seq_len_q,
        )
        if cutlass.const_expr(cfg.use_split_kv):
            logical_kv_idx = logical_b_idx * self.num_heads_kv + logical_h_k_idx
            splits_kv = self._runtime_splits_kv(stage_info)
            cta_idx_kv = _logical_cta_kv_idx(cfg, stage_info)
            base_addr0 = tmem_row_base + Int32(
                o_base_col + tail_o_stage_idx_0 * cfg.tmem_o_stage_cols
            )
            base_addr1 = tmem_row_base + Int32(
                o_base_col + tail_o_stage_idx_1 * cfg.tmem_o_stage_cols
            )
            if cutlass.const_expr(cfg.num_insts_kv == 1):
                base_addr1 = base_addr0
            partial_dst_col_offset = col_base * Int32(2)
            partial_norm_scale = Float32(1.0)
            if cutlass.const_expr(cfg.use_separate_reduction_kernel):
                partial_norm_scale = self._separate_partial_norm_scale(reduced_sum_0)
            regs_o_chunk = cutlass.Array(Int32, 4, space=cutlass.AddressSpace.rmem)
            partial_o_row_base = self._gmem_partial_row_offset(
                logical_kv_idx,
                cta_idx_kv,
                logical_output_row_idx,
            ) * Int64(cfg.headdim * 2)
            for store_idx in cutlass.range_constexpr(output_pair_regs // 4):
                chunk_col = store_idx * 8
                # Load O0/O1 from the tail TMEM slots, combine with
                # max-correction scales, then store normalized 16-bit O for
                # standalone reduction or unnormalized O for fused reduction.
                o0_vals = _keeps_tcgen05_ld(
                    cfg,
                    prims.make_tmem_ptr(base_addr0 + Int32(chunk_col), Float32),
                    num=8,
                    offset=keeps_o_ldst_offset,
                )
                if cutlass.const_expr(cfg.num_insts_kv != 1):
                    o1_vals = _keeps_tcgen05_ld(
                        cfg,
                        prims.make_tmem_ptr(base_addr1 + Int32(chunk_col), Float32),
                        num=8,
                        offset=keeps_o_ldst_offset,
                    )
                prims.tcgen05_wait(kind=prims.Tcgen05Wait.LOAD)
                for chunk_idx in cutlass.range_constexpr(4):
                    reg_base = chunk_idx * 2
                    if cutlass.const_expr(cfg.num_insts_kv == 1):
                        partial_pair = fmul2(
                            (exp_scale0_0, exp_scale0_0),
                            (
                                o0_vals[reg_base],
                                o0_vals[reg_base + 1],
                            ),
                        )
                    else:
                        partial_pair = ffma2(
                            (exp_scale1_0, exp_scale1_0),
                            (o1_vals[reg_base], o1_vals[reg_base + 1]),
                            fmul2(
                                (exp_scale0_0, exp_scale0_0),
                                (
                                    o0_vals[reg_base],
                                    o0_vals[reg_base + 1],
                                ),
                            ),
                        )
                    if cutlass.const_expr(cfg.use_separate_reduction_kernel):
                        partial_pair = fmul2(
                            (partial_norm_scale, partial_norm_scale), partial_pair
                        )
                        regs_o_chunk[chunk_idx] = self._pack_separate_partial_o_pair(
                            partial_pair[0], partial_pair[1]
                        )
                    elif cutlass.const_expr(cfg.use_bf16_output):
                        regs_o_chunk[chunk_idx] = _pack_float2_to_bf16(
                            partial_pair[0], partial_pair[1]
                        )
                    else:
                        regs_o_chunk[chunk_idx] = _pack_float2_to_fp16(
                            partial_pair[0], partial_pair[1]
                        )
                partial_o_dst = cutlass.inttoptr(
                    self.partial_o_ptr.toint()
                    + partial_o_row_base
                    + Int64(partial_dst_col_offset)
                    + Int64(store_idx * 16),
                    mem_space=1,
                    dtype=Int32,
                )
                if valid_output_row:
                    partial_o_dst.store(
                        regs_o_chunk.data_ptr().load(count=4, alignment=4),
                        alignment=16,
                    )

            if col_base == Int32(0) and valid_output_row:
                # One state record per output row accompanies all partial-O
                # vector chunks for that row.
                if cutlass.const_expr(cfg.use_separate_reduction_kernel):
                    stats_row = self._gmem_partial_row_offset(
                        logical_kv_idx,
                        cta_idx_kv,
                        logical_output_row_idx,
                    )
                    stats_ptr = cutlass.inttoptr(
                        self.partial_stats_ptr.toint() + stats_row * Int64(4),
                        mem_space=1,
                        dtype=Float32,
                    )
                    stats_ptr.store(
                        self._separate_partial_lse(final_max_0, reduced_sum_0),
                        alignment=4,
                    )
                else:
                    stats_row = (
                        logical_kv_idx * Int32(cfg.max_splits_kv) + cta_idx_kv
                    ) * self.h_r + logical_output_row_idx
                    stats_ptr = cutlass.inttoptr(
                        self.partial_stats_ptr.toint()
                        + cutlass.Int64(stats_row * Int32(8)),
                        mem_space=1,
                        dtype=Float32,
                    )
                    stats_ptr.store(
                        cutlass.Vector.from_elements(
                            (final_max_0, reduced_sum_0), Float32
                        ),
                        alignment=8,
                    )

            if cutlass.const_expr(cfg.use_separate_reduction_kernel):
                # Separate reducer kernel will consume the partial O/stats
                # written above; this CTA has no in-kernel reduction work.
                return

            # In-kernel GMEM reducer: all split CTAs publish partials, then
            # synchronize before each owner CTA reduces its 2-KiB slice band.
            counter_q_groups = self._multi_cta_counter_q_groups(self.h_r)
            counter_group_idx = logical_kv_idx * counter_q_groups + logical_q_group_idx
            self._reduce_fused_gmem_partials(
                logical_h_k_idx,
                logical_b_idx,
                logical_kv_idx,
                q_row_offset,
                logical_q_group_idx,
                counter_group_idx,
                splits_kv,
                warp_grp_thread_idx,
            )
            return

        attention_sink_h_r = _attention_sink_head_stride(cfg, self.h_r)
        attention_sink_head_idx = _local_head_from_q_output_row(
            cfg, self.h_r, logical_output_row_idx
        )
        reduced_sum_0 += _attention_sink_for_local_head(
            cfg,
            self.attention_sinks_ptr,
            self.scale_softmax_log2,
            final_max_0,
            logical_h_k_idx,
            attention_sink_h_r,
            self.num_heads_kv,
            attention_sink_head_idx,
        )
        # Apply public bmm2_scale in the final direct-output normalization.
        # The split-KV branch above returns before reaching this point, so its
        # partial O remains unscaled for the selected reducer.
        norm_scale_0 = self.output_scale * self._safe_norm_rcp(reduced_sum_0)
        # Direct-output path: fold attention sinks into the denominator, then
        # bake normalization and per-instance max correction into the O scales.
        final_scale0_0 = norm_scale_0 * exp_scale0_0
        final_scale1_0 = norm_scale_0 * exp_scale1_0

        base_addr0 = tmem_row_base + Int32(
            o_base_col + tail_o_stage_idx_0 * cfg.tmem_o_stage_cols
        )
        base_addr1 = tmem_row_base + Int32(
            o_base_col + tail_o_stage_idx_1 * cfg.tmem_o_stage_cols
        )
        if cutlass.const_expr(cfg.num_insts_kv == 1):
            base_addr1 = base_addr0
        physical_dst_row_idx = _q_physical_output_row(
            cfg,
            self.h_r,
            self.num_heads_kv,
            logical_b_idx,
            logical_h_k_idx,
            logical_q_group_idx,
            row_idx,
            self.q_token_offset,
        )
        dst_row_base = Int64(physical_dst_row_idx) * Int64(
            cfg.headdim * cfg.o_dtype_bytes
        )
        dst_col_offset = Int64(col_base) * Int64(cfg.o_dtype_bytes)
        if cutlass.const_expr(
            cfg.use_fp8_output and cfg.tile_size_q == 128 and cfg.headdim == 128
        ):
            regs_o_chunk = cutlass.Array(Int32, 4, space=cutlass.AddressSpace.rmem)
            for store_idx in cutlass.range_constexpr(output_pair_regs // 8):
                chunk_col = store_idx * 16
                # FP8 output path: load a wider O chunk, apply final scales,
                # pack four values per register, and store directly to GMEM.
                o0_vals = _keeps_tcgen05_ld(
                    cfg,
                    prims.make_tmem_ptr(base_addr0 + Int32(chunk_col), Float32),
                    num=16,
                    offset=keeps_o_ldst_offset,
                )
                if cutlass.const_expr(cfg.num_insts_kv != 1):
                    o1_vals = _keeps_tcgen05_ld(
                        cfg,
                        prims.make_tmem_ptr(base_addr1 + Int32(chunk_col), Float32),
                        num=16,
                        offset=keeps_o_ldst_offset,
                    )
                prims.tcgen05_wait(kind=prims.Tcgen05Wait.LOAD)
                for packed_idx in cutlass.range_constexpr(4):
                    reg_base = packed_idx * 4
                    if cutlass.const_expr(cfg.num_insts_kv == 1):
                        final_pair0 = fmul2(
                            (final_scale0_0, final_scale0_0),
                            (o0_vals[reg_base], o0_vals[reg_base + 1]),
                        )
                        final_pair1 = fmul2(
                            (final_scale0_0, final_scale0_0),
                            (
                                o0_vals[reg_base + 2],
                                o0_vals[reg_base + 3],
                            ),
                        )
                    else:
                        final_pair0 = ffma2(
                            (final_scale1_0, final_scale1_0),
                            (o1_vals[reg_base], o1_vals[reg_base + 1]),
                            fmul2(
                                (final_scale0_0, final_scale0_0),
                                (o0_vals[reg_base], o0_vals[reg_base + 1]),
                            ),
                        )
                        final_pair1 = ffma2(
                            (final_scale1_0, final_scale1_0),
                            (
                                o1_vals[reg_base + 2],
                                o1_vals[reg_base + 3],
                            ),
                            fmul2(
                                (final_scale0_0, final_scale0_0),
                                (
                                    o0_vals[reg_base + 2],
                                    o0_vals[reg_base + 3],
                                ),
                            ),
                        )
                    regs_o_chunk[packed_idx] = _pack_float4_to_fp8_e4m3(
                        final_pair0[0],
                        final_pair0[1],
                        final_pair1[0],
                        final_pair1[1],
                    )
                dst_ptr = cutlass.inttoptr(
                    self.o_ptr.toint()
                    + dst_row_base
                    + dst_col_offset
                    + Int64(store_idx * 16),
                    mem_space=1,
                    dtype=Int32,
                )
                if valid_output_row:
                    dst_ptr.store(
                        regs_o_chunk.data_ptr().load(count=4, alignment=4),
                        alignment=16,
                    )
            return

        regs_o_chunk = cutlass.Array(Int32, 4, space=cutlass.AddressSpace.rmem)
        for store_idx in cutlass.range_constexpr(output_pair_regs // 4):
            chunk_col = store_idx * 8
            # FP16/BF16 or generic FP8 path: load one output chunk from each
            # tail O stage, combine the two instances, then pack to output type.
            o0_vals = _keeps_tcgen05_ld(
                cfg,
                prims.make_tmem_ptr(base_addr0 + Int32(chunk_col), Float32),
                num=8,
                offset=keeps_o_ldst_offset,
            )
            if cutlass.const_expr(cfg.num_insts_kv != 1):
                o1_vals = _keeps_tcgen05_ld(
                    cfg,
                    prims.make_tmem_ptr(base_addr1 + Int32(chunk_col), Float32),
                    num=8,
                    offset=keeps_o_ldst_offset,
                )
            prims.tcgen05_wait(kind=prims.Tcgen05Wait.LOAD)
            if cutlass.const_expr(cfg.use_fp8_output):
                for packed_idx in cutlass.range_constexpr(2):
                    reg_base = packed_idx * 4
                    if cutlass.const_expr(cfg.num_insts_kv == 1):
                        final_pair0 = fmul2(
                            (final_scale0_0, final_scale0_0),
                            (o0_vals[reg_base], o0_vals[reg_base + 1]),
                        )
                        final_pair1 = fmul2(
                            (final_scale0_0, final_scale0_0),
                            (
                                o0_vals[reg_base + 2],
                                o0_vals[reg_base + 3],
                            ),
                        )
                    else:
                        final_pair0 = ffma2(
                            (final_scale1_0, final_scale1_0),
                            (o1_vals[reg_base], o1_vals[reg_base + 1]),
                            fmul2(
                                (final_scale0_0, final_scale0_0),
                                (o0_vals[reg_base], o0_vals[reg_base + 1]),
                            ),
                        )
                        final_pair1 = ffma2(
                            (final_scale1_0, final_scale1_0),
                            (
                                o1_vals[reg_base + 2],
                                o1_vals[reg_base + 3],
                            ),
                            fmul2(
                                (final_scale0_0, final_scale0_0),
                                (
                                    o0_vals[reg_base + 2],
                                    o0_vals[reg_base + 3],
                                ),
                            ),
                        )
                    regs_o_chunk[packed_idx] = _pack_float4_to_fp8_e4m3(
                        final_pair0[0],
                        final_pair0[1],
                        final_pair1[0],
                        final_pair1[1],
                    )
            else:
                for chunk_idx in cutlass.range_constexpr(4):
                    reg_base = chunk_idx * 2
                    if cutlass.const_expr(cfg.num_insts_kv == 1):
                        final_pair = fmul2(
                            (final_scale0_0, final_scale0_0),
                            (o0_vals[reg_base], o0_vals[reg_base + 1]),
                        )
                    else:
                        final_pair = ffma2(
                            (final_scale1_0, final_scale1_0),
                            (o1_vals[reg_base], o1_vals[reg_base + 1]),
                            fmul2(
                                (final_scale0_0, final_scale0_0),
                                (o0_vals[reg_base], o0_vals[reg_base + 1]),
                            ),
                        )
                    if cutlass.const_expr(cfg.use_bf16_output):
                        regs_o_chunk[chunk_idx] = _pack_float2_to_bf16(
                            final_pair[0], final_pair[1]
                        )
                    else:
                        regs_o_chunk[chunk_idx] = _pack_float2_to_fp16(
                            final_pair[0], final_pair[1]
                        )
            dst_ptr = cutlass.inttoptr(
                self.o_ptr.toint()
                + dst_row_base
                + dst_col_offset
                + Int64(store_idx * (8 if cfg.use_fp8_output else 16)),
                mem_space=1,
                dtype=Int32,
            )
            if cutlass.const_expr(cfg.use_fp8_output):
                if valid_output_row:
                    dst_ptr.store(
                        regs_o_chunk.data_ptr().load(count=2, alignment=4),
                        alignment=8,
                    )
            else:
                if valid_output_row:
                    dst_ptr.store(
                        regs_o_chunk.data_ptr().load(count=4, alignment=4),
                        alignment=16,
                    )
        return

    @cute.jit
    def _swaps_wide_tail_epilogue(
        self,
        stage_info: StageInfo,
        *,
        tail_o_stage_idx_0: Int32,
        tail_o_stage_idx_1: Int32,
        inst0_new_max_arr: cutlass.Array,
        inst0_sum_arr: cutlass.Array,
        inst1_new_max_arr: cutlass.Array,
        inst1_sum_arr: cutlass.Array,
        tmem_row_base: Int32,
        o_base_col: Constexpr[int],
        q_repeats: Constexpr[int],
        num_scale_groups: Constexpr[int],
        output_pair_regs: Constexpr[int],
        output_f32_regs: Constexpr[int],
        num_o_chunks: Constexpr[int],
    ) -> None:
        """Finish the SwapsMmaAb TileSizeQ 16/32 tail stage."""
        cfg = self.cfg
        task_cache = _decode_gen_task_cache(stage_info)
        warp_grp_thread_idx = task_cache[_TASK_CACHE_WARP_GRP_THREAD_IDX]
        # Tail ProdWork: combine inst0 and inst1 stats, reduce the
        # final denominator across correction warps, then normalize
        # and output the final O tile.
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

        warp_idx = task_cache[_TASK_CACHE_WARP_IDX]
        lane_idx = task_cache[_TASK_CACHE_LANE_IDX]
        col_group_idx = warp_grp_thread_idx & Int32(0x3)
        for scale_base in cutlass.range_constexpr(0, num_scale_groups, 2):
            final_sum_pair_arr = cutlass.Array(
                Float32, 2, space=cutlass.AddressSpace.rmem
            )
            # Merge the two softmax instances with the standard
            # log-sum-exp correction. Some tail stages can be inactive,
            # so skip entries whose max is still the sentinel value.
            for pair_idx in cutlass.range_constexpr(2):
                scale_idx = scale_base + pair_idx
                inst0_max = inst0_new_max_arr[scale_idx]
                inst1_max = inst1_new_max_arr[scale_idx]
                uses_inst0 = inst0_max != _neg_max_f32()
                uses_inst1 = inst1_max != _neg_max_f32()

                final_max_val = _neg_max_f32()
                if uses_inst0:
                    final_max_val = inst0_max
                if uses_inst1:
                    final_max_val = cute.math.max(final_max_val, inst1_max, ftz=True)

                exp_scale0_val = Float32(0.0)
                exp_scale1_val = Float32(0.0)
                final_sum_val = Float32(0.0)
                if uses_inst0:
                    exp_scale0_val = cute.math.exp2(
                        self.scale_softmax_log2 * (inst0_max - final_max_val),
                        fastmath=True,
                    )
                    final_sum_val += inst0_sum_arr[scale_idx] * exp_scale0_val
                if uses_inst1:
                    exp_scale1_val = cute.math.exp2(
                        self.scale_softmax_log2 * (inst1_max - final_max_val),
                        fastmath=True,
                    )
                    final_sum_val += inst1_sum_arr[scale_idx] * exp_scale1_val

                final_max[scale_idx] = final_max_val
                exp_scale0[scale_idx] = exp_scale0_val
                exp_scale1[scale_idx] = exp_scale1_val
                final_sum_pair_arr[pair_idx] = final_sum_val

            final_sum_pair = self._warp_reduce_col_group_sum_pair(
                (final_sum_pair_arr[0], final_sum_pair_arr[1])
            )
            final_sum[scale_base] = final_sum_pair[0]
            final_sum[scale_base + 1] = final_sum_pair[1]

        warp_store_base = warp_idx * Int32(
            4 * num_scale_groups
        ) + col_group_idx * Int32(num_scale_groups)
        if lane_idx < Int32(4):
            # Store one warp partial per column group. The following CTA
            # barrier and reload combine the four correction warps for the
            # final denominator.
            if cutlass.const_expr(num_scale_groups == 8):
                self._sum_scratch.store(
                    (
                        final_sum[0],
                        final_sum[1],
                        final_sum[2],
                        final_sum[3],
                        final_sum[4],
                        final_sum[5],
                        final_sum[6],
                        final_sum[7],
                    ),
                    warp_store_base,
                    alignment=16,
                )
            else:
                self._sum_scratch.store(
                    (
                        final_sum[0],
                        final_sum[1],
                        final_sum[2],
                        final_sum[3],
                    ),
                    warp_store_base,
                    alignment=16,
                )
        prims.barrier_cta_sync(
            self.sum_barrier_id,
            thread_count=cfg.correction_barrier_threads,
        )

        reduce_base = col_group_idx * Int32(num_scale_groups)
        reduced_vec = self._sum_scratch.load(
            reduce_base,
            vector_size=num_scale_groups,
            alignment=16,
        )
        for scale_idx in cutlass.range_constexpr(num_scale_groups):
            reduced_sum[scale_idx] = reduced_vec[scale_idx]
        for warp_offset in cutlass.range_constexpr(1, 4):
            # Combine the four warp partials for this column group.
            other_vec = self._sum_scratch.load(
                reduce_base + warp_offset * Int32(4 * num_scale_groups),
                vector_size=num_scale_groups,
                alignment=16,
            )
            for scale_base in cutlass.range_constexpr(0, num_scale_groups, 2):
                reduced_pair = fadd2(
                    (
                        reduced_sum[scale_base],
                        reduced_sum[scale_base + 1],
                    ),
                    (other_vec[scale_base], other_vec[scale_base + 1]),
                )
                reduced_sum[scale_base] = reduced_pair[0]
                reduced_sum[scale_base + 1] = reduced_pair[1]
        if cutlass.const_expr(not cfg.use_split_kv):
            logical_h_k_idx, _ = _logical_head_batch(
                stage_info, self.h_k_idx, self.b_idx
            )
            logical_q_group_idx = _logical_q_group_idx(
                cfg, stage_info, self.q_group_idx
            )
            for scale_idx in cutlass.range_constexpr(num_scale_groups):
                # Attention sinks add to the denominator before the
                # final normalization scale is computed.
                reduced_sum[scale_idx] += _attention_sink_for_scale_idx(
                    cfg,
                    self.attention_sinks_ptr,
                    self.scale_softmax_log2,
                    final_max[scale_idx],
                    logical_h_k_idx,
                    self.h_r,
                    self.num_heads_kv,
                    logical_q_group_idx,
                    col_group_idx,
                    scale_idx,
                )

        base_addr0 = self._swaps_o_stage_base_addr(
            tmem_row_base, o_base_col, tail_o_stage_idx_0
        )
        base_addr1 = self._swaps_o_stage_base_addr(
            tmem_row_base, o_base_col, tail_o_stage_idx_1
        )
        o0_vals, o1_vals = self._swaps_load_two_o_stage_chunks(
            base_addr0,
            base_addr1,
            q_repeats=q_repeats,
            num_o_chunks=num_o_chunks,
            output_f32_regs=output_f32_regs,
        )

        if cutlass.const_expr(cfg.use_split_kv):
            # Split-KV tail: separate reduction stores normalized 16-bit O
            # plus log2-LSE; fused GMEM/cluster retains unnormalized O plus
            # max/sum state.
            logical_h_k_idx, logical_b_idx = _logical_head_batch(
                stage_info, self.h_k_idx, self.b_idx
            )
            logical_q_group_idx = _logical_q_group_idx(
                cfg, stage_info, self.q_group_idx
            )
            q_row_offset = _q_tile_output_row_base(cfg, logical_q_group_idx)
            logical_kv_idx = logical_b_idx * self.num_heads_kv + logical_h_k_idx
            splits_kv = self._runtime_splits_kv(stage_info)
            cta_idx_kv = _logical_cta_kv_idx(cfg, stage_info)

            if cutlass.const_expr(cfg.use_separate_reduction_kernel):
                for scale_idx in cutlass.range_constexpr(num_scale_groups):
                    norm_scale = self._separate_partial_norm_scale(
                        reduced_sum[scale_idx]
                    )
                    final_scale0[scale_idx] = norm_scale * exp_scale0[scale_idx]
                    final_scale1[scale_idx] = norm_scale * exp_scale1[scale_idx]

            regs_partial_o = cutlass.Array(
                Int32, output_pair_regs, space=cutlass.AddressSpace.rmem
            )
            for pair_idx in cutlass.range_constexpr(output_pair_regs):
                # Separate reduction includes this split's reciprocal sum;
                # fused reduction delays normalization until the final merge.
                scale_base = ((pair_idx % (2 * q_repeats)) // 2) * 2
                reg_base = pair_idx * 2
                partial_scale0 = (
                    (final_scale0[scale_base], final_scale0[scale_base + 1])
                    if cutlass.const_expr(cfg.use_separate_reduction_kernel)
                    else (exp_scale0[scale_base], exp_scale0[scale_base + 1])
                )
                partial_scale1 = (
                    (final_scale1[scale_base], final_scale1[scale_base + 1])
                    if cutlass.const_expr(cfg.use_separate_reduction_kernel)
                    else (exp_scale1[scale_base], exp_scale1[scale_base + 1])
                )
                partial_pair = ffma2(
                    partial_scale0,
                    (o0_vals[reg_base], o0_vals[reg_base + 1]),
                    fmul2(
                        partial_scale1,
                        (o1_vals[reg_base], o1_vals[reg_base + 1]),
                    ),
                )
                if cutlass.const_expr(cfg.use_separate_reduction_kernel):
                    regs_partial_o[pair_idx] = self._pack_separate_partial_o_pair(
                        partial_pair[0], partial_pair[1]
                    )
                elif cutlass.const_expr(cfg.use_bf16_output and not cfg.use_fp8_output):
                    regs_partial_o[pair_idx] = _pack_float2_to_bf16(
                        partial_pair[0], partial_pair[1]
                    )
                else:
                    regs_partial_o[pair_idx] = _pack_float2_to_fp16(
                        partial_pair[0], partial_pair[1]
                    )

            num_copy_segments = max(
                (cfg.tile_size_q * cfg.headdim * 2 + 2047) // 2048,
                1,
            )
            if cutlass.const_expr(cfg.supports_cluster_smem_reduction):
                self._publish_and_reduce_cluster_swaps_partials(
                    regs_partial_o,
                    logical_h_k_idx,
                    logical_b_idx,
                    logical_kv_idx,
                    q_row_offset,
                    logical_q_group_idx,
                    cta_idx_kv,
                    splits_kv,
                    warp_grp_thread_idx,
                    warp_idx,
                    lane_idx,
                    final_max,
                    reduced_sum,
                    output_pair_regs=output_pair_regs,
                    num_copy_segments=num_copy_segments,
                    num_scale_groups=num_scale_groups,
                )
                return

            self._stage_and_copy_swaps_partial_o(
                cfg,
                regs_partial_o,
                logical_kv_idx,
                cta_idx_kv,
                splits_kv,
                q_row_offset,
                logical_q_group_idx,
                warp_grp_thread_idx,
                warp_idx,
                lane_idx,
                output_pair_regs=output_pair_regs,
                num_copy_segments=num_copy_segments,
                enable_cluster=False,
                full_prefix=False,
            )

            self._store_split_stats_from_scale_arrays(
                cfg,
                logical_kv_idx,
                cta_idx_kv,
                splits_kv,
                q_row_offset,
                logical_q_group_idx,
                warp_idx,
                lane_idx,
                final_max,
                reduced_sum,
                num_scale_groups=num_scale_groups,
                enable_cluster=False,
                full_prefix=False,
            )

            if cutlass.const_expr(cfg.use_separate_reduction_kernel):
                # The standalone reducer consumes the partial O/stats above;
                # do not enter the in-kernel completion/election protocol.
                return

            counter_q_groups = self._multi_cta_counter_q_groups(self.h_r)
            counter_group_idx = logical_kv_idx * counter_q_groups + logical_q_group_idx
            self._reduce_fused_gmem_partials(
                logical_h_k_idx,
                logical_b_idx,
                logical_kv_idx,
                q_row_offset,
                logical_q_group_idx,
                counter_group_idx,
                splits_kv,
                warp_grp_thread_idx,
            )
            return

        for scale_idx in cutlass.range_constexpr(num_scale_groups):
            # Direct final-output scales include both the reciprocal softmax
            # denominator and public bmm2_scale. Split partials returned above
            # use only the unnormalized ``exp_scale*`` arrays.
            norm_scale = self.output_scale * self._safe_norm_rcp(reduced_sum[scale_idx])
            final_scale0[scale_idx] = norm_scale * exp_scale0[scale_idx]
            final_scale1[scale_idx] = norm_scale * exp_scale1[scale_idx]

        regs_o = cutlass.Array(
            Int32,
            cfg.num_fp8_output_regs if cfg.use_fp8_output else output_pair_regs,
            space=cutlass.AddressSpace.rmem,
        )
        if cutlass.const_expr(cfg.use_fp8_output):
            # Fold and pack two adjacent pairs immediately. Keeping a second
            # full FP32 output array live makes D256 exceed the correction
            # task's register budget before the transposed STSM.
            for packed_idx in cutlass.range_constexpr(cfg.num_fp8_output_regs):
                pair_idx0 = packed_idx * 2
                pair_idx1 = pair_idx0 + 1
                scale_base0 = ((pair_idx0 % (2 * q_repeats)) // 2) * 2
                scale_base1 = ((pair_idx1 % (2 * q_repeats)) // 2) * 2
                reg_base0 = pair_idx0 * 2
                reg_base1 = pair_idx1 * 2
                final_pair0 = ffma2(
                    (final_scale0[scale_base0], final_scale0[scale_base0 + 1]),
                    (o0_vals[reg_base0], o0_vals[reg_base0 + 1]),
                    fmul2(
                        (
                            final_scale1[scale_base0],
                            final_scale1[scale_base0 + 1],
                        ),
                        (o1_vals[reg_base0], o1_vals[reg_base0 + 1]),
                    ),
                )
                final_pair1 = ffma2(
                    (final_scale0[scale_base1], final_scale0[scale_base1 + 1]),
                    (o0_vals[reg_base1], o0_vals[reg_base1 + 1]),
                    fmul2(
                        (
                            final_scale1[scale_base1],
                            final_scale1[scale_base1 + 1],
                        ),
                        (o1_vals[reg_base1], o1_vals[reg_base1 + 1]),
                    ),
                )
                # bmm2_scale is already folded into ``final_scale*`` above.
                regs_o[packed_idx] = _pack_float4_to_fp8_e4m3(
                    final_pair0[0],
                    final_pair0[1],
                    final_pair1[0],
                    final_pair1[1],
                )
        else:
            for pair_idx in cutlass.range_constexpr(output_pair_regs):
                # non-split-KV path: combine inst0/inst1 O and pack directly
                # to the final output dtype.
                scale_base = ((pair_idx % (2 * q_repeats)) // 2) * 2
                reg_base = pair_idx * 2
                final_pair = ffma2(
                    (final_scale0[scale_base], final_scale0[scale_base + 1]),
                    (o0_vals[reg_base], o0_vals[reg_base + 1]),
                    fmul2(
                        (
                            final_scale1[scale_base],
                            final_scale1[scale_base + 1],
                        ),
                        (o1_vals[reg_base], o1_vals[reg_base + 1]),
                    ),
                )
                if cutlass.const_expr(cfg.use_bf16_output):
                    regs_o[pair_idx] = _pack_float2_to_bf16(
                        final_pair[0], final_pair[1]
                    )
                else:
                    regs_o[pair_idx] = _pack_float2_to_fp16(
                        final_pair[0], final_pair[1]
                    )

        warp_idx = task_cache[_TASK_CACHE_WARP_IDX]
        lane_idx = task_cache[_TASK_CACHE_LANE_IDX]
        if cutlass.const_expr(cfg.use_fp8_output):
            # FP8 final output is staged with transposed 8-bit
            # STSM and then copied to GMEM as contiguous vectors. This two-step
            # layout is required because correction registers are in MMA
            # fragment order, not output row-major order.
            _store_transposed_smem8b(
                self._smem_base_o_i32,
                regs_o.data_ptr().load(
                    count=cfg.num_fp8_output_regs,
                    alignment=4,
                ),
                warp_grp_thread_idx,
                cfg.tile_size_q,
                cfg.headdim,
                cfg.num_fp8_output_regs,
            )
            cute.arch.fence_view_async_shared()
            prims.barrier_cta_sync(
                self.store_barrier_id,
                thread_count=cfg.correction_barrier_threads,
            )

            logical_h_k_idx, logical_b_idx = _logical_head_batch(
                stage_info, self.h_k_idx, self.b_idx
            )
            logical_q_group_idx = _logical_q_group_idx(
                cfg, stage_info, self.q_group_idx
            )
            _copy_transposed_smem8b_to_gmem(
                self._smem_base_o_i32,
                self.o_ptr,
                cfg,
                logical_h_k_idx,
                logical_b_idx,
                logical_q_group_idx,
                self.h_r,
                self.num_heads_kv,
                self.seq_len_q,
                self.q_token_offset,
                warp_grp_thread_idx,
                cfg.fp8_copy_can_use_full_tile_fast_path,
            )
            if cutlass.const_expr(cfg.use_persistent_scheduler):
                prims.barrier_cta_sync(
                    self.store_barrier_id,
                    thread_count=cfg.correction_barrier_threads,
                )
            return

        logical_h_k_idx, logical_b_idx = _logical_head_batch(
            stage_info, self.h_k_idx, self.b_idx
        )
        logical_q_group_idx = _logical_q_group_idx(cfg, stage_info, self.q_group_idx)
        self._stage_fp16_o_regs_to_smem(
            cfg,
            regs_o,
            warp_grp_thread_idx,
            warp_idx,
            lane_idx,
            output_pair_regs=output_pair_regs,
        )
        num_copy_segments = max(
            (cfg.tile_size_q * cfg.headdim * cfg.o_dtype_bytes + 2047) // 2048,
            1,
        )
        self._copy_staged_fp16_o_to_output(
            cfg,
            logical_h_k_idx,
            logical_b_idx,
            logical_q_group_idx,
            warp_grp_thread_idx,
            warp_idx,
            lane_idx,
            num_copy_segments=num_copy_segments,
        )
        if cutlass.const_expr(cfg.use_persistent_scheduler):
            prims.barrier_cta_sync(
                self.store_barrier_id,
                thread_count=cfg.correction_barrier_threads,
            )
        return

    @cute.jit
    def _swaps_q8_tail_epilogue(
        self,
        stage_info: StageInfo,
        *,
        tail_o_stage_idx_0: Int32,
        tail_o_stage_idx_1: Int32,
        inst0_new_max_arr: cutlass.Array,
        inst0_sum_arr: cutlass.Array,
        inst1_new_max_arr: cutlass.Array,
        inst1_sum_arr: cutlass.Array,
        tmem_row_base: Int32,
        o_base_col: Constexpr[int],
    ) -> None:
        """Finish the SwapsMmaAb TileSizeQ 8 tail stage."""
        cfg = self.cfg
        task_cache = _decode_gen_task_cache(stage_info)
        warp_grp_thread_idx = task_cache[_TASK_CACHE_WARP_GRP_THREAD_IDX]
        # Tile-Q=8 tail ProdWork: combine the two final O stages,
        # normalize, and either emit final O or reduce split-KV
        # partials.
        num_scale_groups = 2
        final_max = cutlass.Array(Float32, 2, space=cutlass.AddressSpace.rmem)
        final_sum = cutlass.Array(Float32, 2, space=cutlass.AddressSpace.rmem)
        exp_scale0 = cutlass.Array(Float32, 2, space=cutlass.AddressSpace.rmem)
        exp_scale1 = cutlass.Array(Float32, 2, space=cutlass.AddressSpace.rmem)
        reduced_sum = cutlass.Array(Float32, 2, space=cutlass.AddressSpace.rmem)
        final_scale0 = cutlass.Array(Float32, 2, space=cutlass.AddressSpace.rmem)
        final_scale1 = cutlass.Array(Float32, 2, space=cutlass.AddressSpace.rmem)

        # Compute per-instance exp corrections relative to the final
        # max for the two scale groups.
        for scale_idx in cutlass.range_constexpr(num_scale_groups):
            inst0_max = inst0_new_max_arr[scale_idx]
            inst1_max = inst1_new_max_arr[scale_idx]
            uses_inst0 = inst0_max != _neg_max_f32()
            uses_inst1 = inst1_max != _neg_max_f32()
            final_max[scale_idx] = _neg_max_f32()
            if uses_inst0:
                final_max[scale_idx] = inst0_max
            if uses_inst1:
                final_max[scale_idx] = cute.math.max(
                    final_max[scale_idx], inst1_max, ftz=True
                )
            exp_scale0[scale_idx] = Float32(0.0)
            exp_scale1[scale_idx] = Float32(0.0)
            if uses_inst0:
                exp_scale0[scale_idx] = cute.math.exp2(
                    self.scale_softmax_log2 * (inst0_max - final_max[scale_idx]),
                    fastmath=True,
                )
            if uses_inst1:
                exp_scale1[scale_idx] = cute.math.exp2(
                    self.scale_softmax_log2 * (inst1_max - final_max[scale_idx]),
                    fastmath=True,
                )
            reduced_sum[scale_idx] = Float32(0.0)
            final_scale0[scale_idx] = Float32(0.0)
            final_scale1[scale_idx] = Float32(0.0)

        final_sums = ffma2(
            (exp_scale0[0], exp_scale0[1]),
            (inst0_sum_arr[0], inst0_sum_arr[1]),
            fmul2(
                (exp_scale1[0], exp_scale1[1]),
                (inst1_sum_arr[0], inst1_sum_arr[1]),
            ),
        )
        for scale_idx in cutlass.range_constexpr(num_scale_groups):
            final_sum[scale_idx] = final_sums[scale_idx]

        task_cache = _decode_gen_task_cache(stage_info)
        warp_grp_thread_idx = task_cache[_TASK_CACHE_WARP_GRP_THREAD_IDX]
        warp_idx = task_cache[_TASK_CACHE_WARP_IDX]
        lane_idx = task_cache[_TASK_CACHE_LANE_IDX]
        col_group_idx = warp_grp_thread_idx & Int32(0x3)

        # Reduce within each warp across lanes that share the same
        # column group, then combine the 4 warp partials through
        # shared memory.
        final_sum_pair = self._warp_reduce_col_group_sum_pair(
            (final_sum[0], final_sum[1])
        )
        for scale_idx in cutlass.range_constexpr(num_scale_groups):
            final_sum[scale_idx] = final_sum_pair[scale_idx]

        warp_store_base = warp_idx * Int32(8) + col_group_idx * Int32(2)
        if lane_idx < Int32(4):
            self._sum_scratch.store(
                (final_sum[0], final_sum[1]),
                warp_store_base,
                alignment=8,
            )
        prims.barrier_cta_sync(
            self.sum_barrier_id,
            thread_count=cfg.correction_barrier_threads,
        )

        reduce_base = col_group_idx * Int32(2)
        reduced_pair = self._sum_scratch.load(
            reduce_base,
            vector_size=2,
            alignment=8,
        )
        for scale_idx in cutlass.range_constexpr(num_scale_groups):
            reduced_sum[scale_idx] = reduced_pair[scale_idx]
        for warp_offset in cutlass.range_constexpr(1, 4):
            other_pair = self._sum_scratch.load(
                reduce_base + warp_offset * Int32(8),
                vector_size=2,
                alignment=8,
            )
            reduced_pair = fadd2(
                (reduced_sum[0], reduced_sum[1]),
                (other_pair[0], other_pair[1]),
            )
            for scale_idx in cutlass.range_constexpr(num_scale_groups):
                reduced_sum[scale_idx] = reduced_pair[scale_idx]

        if cutlass.const_expr(not cfg.use_split_kv):
            logical_h_k_idx, _ = _logical_head_batch(
                stage_info, self.h_k_idx, self.b_idx
            )
            logical_q_group_idx = _logical_q_group_idx(
                cfg, stage_info, self.q_group_idx
            )
            for scale_idx in cutlass.range_constexpr(num_scale_groups):
                reduced_sum[scale_idx] += _attention_sink_for_scale_idx(
                    cfg,
                    self.attention_sinks_ptr,
                    self.scale_softmax_log2,
                    final_max[scale_idx],
                    logical_h_k_idx,
                    self.h_r,
                    self.num_heads_kv,
                    logical_q_group_idx,
                    col_group_idx,
                    scale_idx,
                )

        base_addr0 = self._swaps_o_stage_base_addr(
            tmem_row_base, o_base_col, tail_o_stage_idx_0
        )
        base_addr1 = self._swaps_o_stage_base_addr(
            tmem_row_base, o_base_col, tail_o_stage_idx_1
        )
        output_pair_regs = cfg.num_fp16_output_regs
        output_f32_regs = output_pair_regs * 2
        num_o_chunks = cfg.headdim // 64
        o0_vals, o1_vals = self._swaps_load_two_o_stage_chunks(
            base_addr0,
            base_addr1,
            q_repeats=1,
            num_o_chunks=num_o_chunks,
            output_f32_regs=output_f32_regs,
        )

        if cutlass.const_expr(cfg.use_split_kv):
            # Publish this CTA's partial output and statistics. Standalone
            # reduction returns after publication; cluster peers reduce through
            # DSMEM, while fused GMEM elects the final producer CTA.
            logical_h_k_idx, logical_b_idx = _logical_head_batch(
                stage_info, self.h_k_idx, self.b_idx
            )
            logical_q_group_idx = _logical_q_group_idx(
                cfg, stage_info, self.q_group_idx
            )
            q_row_offset = _q_tile_output_row_base(cfg, logical_q_group_idx)
            logical_kv_idx = logical_b_idx * self.num_heads_kv + logical_h_k_idx
            splits_kv = self._runtime_splits_kv(stage_info)
            cta_idx_kv = _logical_cta_kv_idx(cfg, stage_info)

            if cutlass.const_expr(cfg.use_separate_reduction_kernel):
                for scale_idx in cutlass.range_constexpr(num_scale_groups):
                    norm_scale = self._separate_partial_norm_scale(
                        reduced_sum[scale_idx]
                    )
                    final_scale0[scale_idx] = norm_scale * exp_scale0[scale_idx]
                    final_scale1[scale_idx] = norm_scale * exp_scale1[scale_idx]

            # Store normalized 16-bit O for the standalone reducer, or preserve
            # unnormalized 16-bit O for fused GMEM/cluster reduction.
            regs_partial_o = cutlass.Array(
                Int32,
                cfg.num_fp16_output_regs,
                space=cutlass.AddressSpace.rmem,
            )
            partial_scale0_pair = (
                (final_scale0[0], final_scale0[1])
                if cutlass.const_expr(cfg.use_separate_reduction_kernel)
                else (exp_scale0[0], exp_scale0[1])
            )
            partial_scale1_pair = (
                (final_scale1[0], final_scale1[1])
                if cutlass.const_expr(cfg.use_separate_reduction_kernel)
                else (exp_scale1[0], exp_scale1[1])
            )
            for reg_idx in cutlass.range_constexpr(cfg.num_fp16_output_regs):
                reg_base = reg_idx * 2
                partial_pair = ffma2(
                    partial_scale0_pair,
                    (o0_vals[reg_base], o0_vals[reg_base + 1]),
                    fmul2(
                        partial_scale1_pair,
                        (o1_vals[reg_base], o1_vals[reg_base + 1]),
                    ),
                )
                if cutlass.const_expr(cfg.use_separate_reduction_kernel):
                    regs_partial_o[reg_idx] = self._pack_separate_partial_o_pair(
                        partial_pair[0], partial_pair[1]
                    )
                elif cutlass.const_expr(cfg.use_bf16_output and not cfg.use_fp8_output):
                    regs_partial_o[reg_idx] = _pack_float2_to_bf16(
                        partial_pair[0], partial_pair[1]
                    )
                else:
                    regs_partial_o[reg_idx] = _pack_float2_to_fp16(
                        partial_pair[0], partial_pair[1]
                    )

            num_partial_o_segments = max(
                (cfg.tile_size_q * cfg.headdim * 2 + 2047) // 2048,
                1,
            )
            if cutlass.const_expr(cfg.supports_cluster_smem_reduction):
                self._publish_and_reduce_cluster_swaps_partials(
                    regs_partial_o,
                    logical_h_k_idx,
                    logical_b_idx,
                    logical_kv_idx,
                    q_row_offset,
                    logical_q_group_idx,
                    cta_idx_kv,
                    splits_kv,
                    warp_grp_thread_idx,
                    warp_idx,
                    lane_idx,
                    final_max,
                    reduced_sum,
                    output_pair_regs=output_pair_regs,
                    num_copy_segments=num_partial_o_segments,
                    num_scale_groups=2,
                )
                return

            self._stage_and_copy_swaps_partial_o(
                cfg,
                regs_partial_o,
                logical_kv_idx,
                cta_idx_kv,
                splits_kv,
                q_row_offset,
                logical_q_group_idx,
                warp_grp_thread_idx,
                warp_idx,
                lane_idx,
                output_pair_regs=output_pair_regs,
                num_copy_segments=num_partial_o_segments,
                enable_cluster=False,
                full_prefix=False,
            )

            self._store_split_stats_from_scale_arrays(
                cfg,
                logical_kv_idx,
                cta_idx_kv,
                splits_kv,
                q_row_offset,
                logical_q_group_idx,
                warp_idx,
                lane_idx,
                final_max,
                reduced_sum,
                num_scale_groups=2,
                enable_cluster=False,
                full_prefix=False,
            )

            if cutlass.const_expr(cfg.use_separate_reduction_kernel):
                # The standalone reducer consumes the partial O/stats. Do not
                # also execute the fused in-kernel counter/cluster reduction.
                return

            counter_q_groups = self._multi_cta_counter_q_groups(self.h_r)
            counter_group_idx = logical_kv_idx * counter_q_groups + logical_q_group_idx
            self._reduce_fused_gmem_partials(
                logical_h_k_idx,
                logical_b_idx,
                logical_kv_idx,
                q_row_offset,
                logical_q_group_idx,
                counter_group_idx,
                splits_kv,
                warp_grp_thread_idx,
            )
            return

        for scale_idx in cutlass.range_constexpr(num_scale_groups):
            # Direct final-output scales include both the reciprocal softmax
            # denominator and public bmm2_scale. Split partials returned above
            # use only the unnormalized ``exp_scale*`` arrays.
            norm_scale = self.output_scale * self._safe_norm_rcp(reduced_sum[scale_idx])
            final_scale0[scale_idx] = norm_scale * exp_scale0[scale_idx]
            final_scale1[scale_idx] = norm_scale * exp_scale1[scale_idx]

        regs_o = cutlass.Array(
            Int32,
            cfg.num_fp8_output_regs if cfg.use_fp8_output else cfg.num_fp16_output_regs,
            space=cutlass.AddressSpace.rmem,
        )
        # Form the final O registers from the two tail stages using the
        # normalized instance scales.
        if cutlass.const_expr(cfg.use_fp8_output):
            for packed_idx in cutlass.range_constexpr(cfg.num_fp8_output_regs):
                pair_idx0 = packed_idx * 2
                pair_idx1 = pair_idx0 + 1
                src_idx0 = pair_idx0 * 2
                src_idx1 = pair_idx1 * 2
                final_pair0 = ffma2(
                    (final_scale0[0], final_scale0[1]),
                    (o0_vals[src_idx0], o0_vals[src_idx0 + 1]),
                    fmul2(
                        (final_scale1[0], final_scale1[1]),
                        (o1_vals[src_idx0], o1_vals[src_idx0 + 1]),
                    ),
                )
                final_pair1 = ffma2(
                    (final_scale0[0], final_scale0[1]),
                    (o0_vals[src_idx1], o0_vals[src_idx1 + 1]),
                    fmul2(
                        (final_scale1[0], final_scale1[1]),
                        (o1_vals[src_idx1], o1_vals[src_idx1 + 1]),
                    ),
                )
                regs_o[packed_idx] = _pack_float4_to_fp8_e4m3(
                    final_pair0[0],
                    final_pair0[1],
                    final_pair1[0],
                    final_pair1[1],
                )
        else:
            for reg_idx in cutlass.range_constexpr(cfg.num_fp16_output_regs):
                reg_base = reg_idx * 2
                final_pair = ffma2(
                    (final_scale0[0], final_scale0[1]),
                    (o0_vals[reg_base], o0_vals[reg_base + 1]),
                    fmul2(
                        (final_scale1[0], final_scale1[1]),
                        (o1_vals[reg_base], o1_vals[reg_base + 1]),
                    ),
                )
                if cutlass.const_expr(cfg.use_bf16_output):
                    regs_o[reg_idx] = _pack_float2_to_bf16(final_pair[0], final_pair[1])
                else:
                    regs_o[reg_idx] = _pack_float2_to_fp16(final_pair[0], final_pair[1])

        task_cache = _decode_gen_task_cache(stage_info)
        warp_grp_thread_idx = task_cache[_TASK_CACHE_WARP_GRP_THREAD_IDX]
        if cutlass.const_expr(cfg.use_fp8_output):
            # FP8 output uses transposed 8-bit staging before the final
            # vectorized GMEM copy.
            _store_transposed_smem8b(
                self._smem_base_o_i32,
                regs_o.data_ptr().load(
                    count=cfg.num_fp8_output_regs,
                    alignment=4,
                ),
                warp_grp_thread_idx,
                cfg.tile_size_q,
                cfg.headdim,
                cfg.num_fp8_output_regs,
            )
            cute.arch.fence_view_async_shared()
            prims.barrier_cta_sync(
                self.store_barrier_id,
                thread_count=cfg.correction_barrier_threads,
            )
            logical_h_k_idx, logical_b_idx = _logical_head_batch(
                stage_info, self.h_k_idx, self.b_idx
            )
            logical_q_group_idx = _logical_q_group_idx(
                cfg, stage_info, self.q_group_idx
            )
            _copy_transposed_smem8b_to_gmem(
                self._smem_base_o_i32,
                self.o_ptr,
                cfg,
                logical_h_k_idx,
                logical_b_idx,
                logical_q_group_idx,
                self.h_r,
                self.num_heads_kv,
                self.seq_len_q,
                self.q_token_offset,
                warp_grp_thread_idx,
                cfg.fp8_copy_can_use_full_tile_fast_path,
            )
            if cutlass.const_expr(cfg.use_persistent_scheduler):
                prims.barrier_cta_sync(
                    self.store_barrier_id,
                    thread_count=cfg.correction_barrier_threads,
                )
        else:
            warp_idx = task_cache[_TASK_CACHE_WARP_IDX]
            lane_idx = task_cache[_TASK_CACHE_LANE_IDX]
            logical_h_k_idx, logical_b_idx = _logical_head_batch(
                stage_info, self.h_k_idx, self.b_idx
            )
            logical_q_group_idx = _logical_q_group_idx(
                cfg, stage_info, self.q_group_idx
            )
            self._stage_fp16_o_regs_to_smem(
                cfg,
                regs_o,
                warp_grp_thread_idx,
                warp_idx,
                lane_idx,
                output_pair_regs=cfg.num_fp16_output_regs,
            )
            num_copy_segments = max(
                (cfg.tile_size_q * cfg.headdim * cfg.o_dtype_bytes + 2047) // 2048,
                1,
            )
            self._copy_staged_fp16_o_to_output(
                cfg,
                logical_h_k_idx,
                logical_b_idx,
                logical_q_group_idx,
                warp_grp_thread_idx,
                warp_idx,
                lane_idx,
                num_copy_segments=num_copy_segments,
            )
            if cutlass.const_expr(cfg.use_persistent_scheduler):
                prims.barrier_cta_sync(
                    self.store_barrier_id,
                    thread_count=cfg.correction_barrier_threads,
                )

    @producer_work
    @cute.jit
    def correction_loop_epilogue(
        self,
        stage_info: StageInfo,
        *,
        o_stage_idx: Int32,
        tail_o_stage_idx_0: Int32,
        tail_o_stage_idx_1: Int32,
        old_max_arr: cutlass.Array,
        new_max_arr: cutlass.Array,
    ) -> None:
        """Rescale the live O stage before later BMM2 waves accumulate into it."""
        # ProdWork: correction owns the live O TMEM stage after loop stats
        # arrive. Rescale it in place so later PV waves accumulate in the
        # updated online-softmax frame.
        cfg = self.cfg
        # Resolve the live TMEM O stage and default SwapsMmaAb column base. The
        # KeepsMmaAb path overrides the base because O is allocated by TmemO.
        task_cache = _decode_gen_task_cache(stage_info)
        tmem_row_base = task_cache[_TASK_CACHE_TMEM_BASE_OFFSET]
        o_base_col = 2 * cfg.tmem_s_cols + 2 * cfg.tmem_stats_cols

        if cutlass.const_expr(cfg.use_keeps_mma_ab):
            # KeepsMmaAb has one softmax scale group for this path. Compute the
            # rescale once, then apply it independently to every P-by-V
            # head-dimension slice in TMEM.
            o_base_col = self.tmem_o_ref._alloc.offset
            corr_chunk_regs = cfg.keeps_loop_correction_chunk_regs

            old_max_0 = old_max_arr[0]
            new_max_0 = new_max_arr[0]
            scale_0, scale_is_identity = self._online_softmax_correction_scale(
                old_max_0,
                new_max_0,
            )
            skip_correction = self._warp_can_skip_o_correction(scale_is_identity)
            scale_pair = (scale_0, scale_0)
            scaled_chunk = cutlass.Array(
                Float32, corr_chunk_regs, space=cutlass.AddressSpace.rmem
            )
            if not skip_correction:
                # Each correction warp owns disjoint Keeps rows. When every
                # row retains its running maximum, the scale is exactly one
                # and the in-place TMEM rescale is a no-op.
                for (
                    head_dim_stage_offset,
                    keeps_o_ldst_offset,
                    num_corr_chunks,
                ) in cfg.keeps_loop_correction_stage_layout:
                    stage_base_addr = tmem_row_base + Int32(
                        o_base_col
                        + o_stage_idx * cfg.tmem_o_stage_cols
                        + head_dim_stage_offset
                    )
                    for chunk_idx in cutlass.range_constexpr(num_corr_chunks):
                        # Load one O chunk, multiply by
                        # exp(old_max-new_max), and store it before the next PV
                        # wave accumulates into this head-dimension slice.
                        chunk_col = chunk_idx * corr_chunk_regs
                        chunk_addr = stage_base_addr + Int32(chunk_col)
                        loaded = _keeps_tcgen05_ld(
                            cfg,
                            prims.make_tmem_ptr(chunk_addr, Float32),
                            num=corr_chunk_regs,
                            offset=keeps_o_ldst_offset,
                        )
                        prims.tcgen05_wait(kind=prims.Tcgen05Wait.LOAD)
                        for reg_pair_idx in cutlass.range_constexpr(
                            corr_chunk_regs // 2
                        ):
                            reg_base = reg_pair_idx * 2
                            scaled_pair = fmul2(
                                scale_pair,
                                (loaded[reg_base], loaded[reg_base + 1]),
                            )
                            scaled_chunk[reg_base] = scaled_pair[0]
                            scaled_chunk[reg_base + 1] = scaled_pair[1]
                        _keeps_tcgen05_st(
                            cfg,
                            prims.make_tmem_ptr(chunk_addr, Float32),
                            scaled_chunk.data_ptr().load(
                                count=corr_chunk_regs, alignment=4
                            ),
                            offset=keeps_o_ldst_offset,
                        )
                        prims.tcgen05_wait(kind=prims.Tcgen05Wait.STORE)
            if skip_correction:
                # Preserve the correction task's TMEM ordering when this warp
                # issues no correction transaction.
                prims.tcgen05_wait(kind=prims.Tcgen05Wait.STORE)
            return

        if cutlass.const_expr(cfg.tile_size_q == 16):
            q_repeats = max(cfg.tile_size_q // 8, 1)
            num_scale_groups = cfg.num_softmax_scale_groups
            output_pair_regs = cfg.num_fp16_output_regs
            output_f32_regs = output_pair_regs * 2
            num_o_chunks = cfg.headdim // 64
            scale_vals = cutlass.Array(
                Float32, num_scale_groups, space=cutlass.AddressSpace.rmem
            )
            lane_scales_are_identity = cutlass.Boolean(True)
            for scale_base in cutlass.range_constexpr(0, num_scale_groups, 2):
                old_max_0 = old_max_arr[scale_base]
                old_max_1 = old_max_arr[scale_base + 1]
                new_max_0 = new_max_arr[scale_base]
                new_max_1 = new_max_arr[scale_base + 1]
                scale_0_is_identity = old_max_0 == new_max_0
                scale_1_is_identity = old_max_1 == new_max_1
                scale_0 = Float32(1.0)
                scale_1 = Float32(1.0)
                # Preserve the packed arithmetic used by the original path;
                # only the exponentials and TMEM transaction are conditional.
                max_diff_pair = fadd2(
                    (old_max_0, old_max_1),
                    (-new_max_0, -new_max_1),
                )
                scale_pair = fmul2(
                    (self.scale_softmax_log2, self.scale_softmax_log2),
                    max_diff_pair,
                )
                if not scale_0_is_identity:
                    scale_0 = cute.math.exp2(scale_pair[0], fastmath=True)
                if not scale_1_is_identity:
                    scale_1 = cute.math.exp2(scale_pair[1], fastmath=True)
                scale_vals[scale_base] = scale_0
                scale_vals[scale_base + 1] = scale_1
                lane_scales_are_identity = (
                    lane_scales_are_identity & scale_0_is_identity & scale_1_is_identity
                )

            skip_correction = self._warp_can_skip_o_correction(lane_scales_are_identity)

            base_addr = self._swaps_o_stage_base_addr(
                tmem_row_base, o_base_col, o_stage_idx
            )
            self._swaps_rescale_o_stage_in_tmem(
                base_addr,
                scale_vals,
                skip_correction,
                q_repeats=q_repeats,
                num_o_chunks=num_o_chunks,
                output_f32_regs=output_f32_regs,
            )
            return

        if cutlass.const_expr(cfg.tile_size_q == 32):
            # FlashInfer keeps Q32 straight-line: the warp vote and dynamic
            # TMEM gate regress the wider Swaps pipeline on B200.
            q_repeats = max(cfg.tile_size_q // 8, 1)
            num_scale_groups = cfg.num_softmax_scale_groups
            output_pair_regs = cfg.num_fp16_output_regs
            output_f32_regs = output_pair_regs * 2
            num_o_chunks = cfg.headdim // 64
            scale_vals = cutlass.Array(
                Float32, num_scale_groups, space=cutlass.AddressSpace.rmem
            )
            for scale_base in cutlass.range_constexpr(0, num_scale_groups, 2):
                old_max_0 = old_max_arr[scale_base]
                old_max_1 = old_max_arr[scale_base + 1]
                new_max_0 = new_max_arr[scale_base]
                new_max_1 = new_max_arr[scale_base + 1]
                max_diff_pair = fadd2((old_max_0, old_max_1), (-new_max_0, -new_max_1))
                scale_pair = fmul2(
                    (self.scale_softmax_log2, self.scale_softmax_log2),
                    max_diff_pair,
                )
                scale_vals[scale_base] = cute.math.exp2(scale_pair[0], fastmath=True)
                scale_vals[scale_base + 1] = cute.math.exp2(
                    scale_pair[1], fastmath=True
                )

            base_addr = self._swaps_o_stage_base_addr(
                tmem_row_base, o_base_col, o_stage_idx
            )
            self._swaps_rescale_o_stage_in_tmem(
                base_addr,
                scale_vals,
                cutlass.Boolean(False),
                q_repeats=q_repeats,
                num_o_chunks=num_o_chunks,
                output_f32_regs=output_f32_regs,
            )
            return

        # Tile-Q=8 loop ProdWork: rescale the current O stage in place when
        # the row max changes.
        old_max_0 = old_max_arr[0]
        old_max_1 = old_max_arr[1]
        new_max_0 = new_max_arr[0]
        new_max_1 = new_max_arr[1]
        scale_0_is_identity = old_max_0 == new_max_0
        scale_1_is_identity = old_max_1 == new_max_1
        scale_0 = Float32(1.0)
        scale_1 = Float32(1.0)
        max_diff_pair = fadd2((old_max_0, old_max_1), (-new_max_0, -new_max_1))
        scale_pair = fmul2(
            (self.scale_softmax_log2, self.scale_softmax_log2), max_diff_pair
        )
        if not scale_0_is_identity:
            scale_0 = cute.math.exp2(scale_pair[0], fastmath=True)
        if not scale_1_is_identity:
            scale_1 = cute.math.exp2(scale_pair[1], fastmath=True)
        skip_correction = self._warp_can_skip_o_correction(
            scale_0_is_identity & scale_1_is_identity
        )

        scale_vals = cutlass.Array(Float32, 2, space=cutlass.AddressSpace.rmem)
        scale_vals[0] = scale_0
        scale_vals[1] = scale_1
        base_addr = self._swaps_o_stage_base_addr(
            tmem_row_base, o_base_col, o_stage_idx
        )
        output_pair_regs = cfg.num_fp16_output_regs
        output_f32_regs = output_pair_regs * 2
        num_o_chunks = cfg.headdim // 64
        self._swaps_rescale_o_stage_in_tmem(
            base_addr,
            scale_vals,
            skip_correction,
            q_repeats=1,
            num_o_chunks=num_o_chunks,
            output_f32_regs=output_f32_regs,
        )

    @producer_work
    @cute.jit
    def correction_tail_epilogue(
        self,
        stage_info: StageInfo,
        *,
        o_stage_idx: Int32,
        tail_o_stage_idx_0: Int32,
        tail_o_stage_idx_1: Int32,
        old_max_arr: cutlass.Array,
        new_max_arr: cutlass.Array,
    ) -> None:
        """Normalize final O stages and store or publish the output tile."""
        _ = o_stage_idx
        _ = old_max_arr
        _ = new_max_arr
        cfg = self.cfg
        # ProdWork: consume final per-instruction softmax stats, normalize the
        # tail O stages, then route the tile to direct output or the active
        # split-KV reduction path.
        # Read inst-specific softmax stats from SoftmaxLocal resource refs.
        # These arrays were populated when the softmax-local tail stats were
        # consumed by CorrectionTask.
        inst0_new_max_arr = self.softmax_local0_ref._inst_new_max_arr
        inst0_sum_arr = self.softmax_local0_ref._inst_sum_arr
        inst1_new_max_arr = inst0_new_max_arr
        inst1_sum_arr = inst0_sum_arr
        if cutlass.const_expr(
            cfg.num_insts_kv != 1 and self.softmax_local1_ref is not None
        ):
            inst1_new_max_arr = self.softmax_local1_ref._inst_new_max_arr
            inst1_sum_arr = self.softmax_local1_ref._inst_sum_arr
        task_cache = _decode_gen_task_cache(stage_info)
        tmem_row_base = task_cache[_TASK_CACHE_TMEM_BASE_OFFSET]
        o_base_col = 2 * cfg.tmem_s_cols + 2 * cfg.tmem_stats_cols
        warp_grp_thread_idx = task_cache[_TASK_CACHE_WARP_GRP_THREAD_IDX]

        if cutlass.const_expr(cfg.use_keeps_mma_ab):
            # KeepsMmaAb finalization is owned by the last active K/V instance.
            # Earlier instances publish stats but do not write the final output.
            if cutlass.const_expr(
                (cfg.num_insts_kv == 1 and self.inst_id == 0)
                or (cfg.num_insts_kv != 1 and self.inst_id == 1)
            ):
                # Derive the per-lane output row/column ownership before
                # entering the common Keeps tail helper.
                o_base_col = self.tmem_o_ref._alloc.offset
                output_f32_regs = cfg.keeps_output_f32_regs
                output_pair_regs = output_f32_regs // 2
                keeps_o_ldst_offset = cfg.headdim // 2
                lane_idx = task_cache[_TASK_CACHE_LANE_IDX]
                row_idx = _keeps_row_idx(cfg, warp_grp_thread_idx)
                col_base = _keeps_col_base(cfg, lane_idx, cfg.headdim // 2)
                self._keeps_tail_epilogue(
                    stage_info,
                    tail_o_stage_idx_0=tail_o_stage_idx_0,
                    tail_o_stage_idx_1=tail_o_stage_idx_1,
                    inst0_new_max_arr=inst0_new_max_arr,
                    inst0_sum_arr=inst0_sum_arr,
                    inst1_new_max_arr=inst1_new_max_arr,
                    inst1_sum_arr=inst1_sum_arr,
                    tmem_row_base=tmem_row_base,
                    o_base_col=o_base_col,
                    output_pair_regs=output_pair_regs,
                    keeps_o_ldst_offset=keeps_o_ldst_offset,
                    row_idx=row_idx,
                    col_base=col_base,
                    warp_grp_thread_idx=warp_grp_thread_idx,
                )
            return

        if cutlass.const_expr(self.inst_id != 1):
            # SwapsMmaAb tail uses instance 1 to combine inst0/inst1 final O
            # stages. Instance 0 exits after publishing its stats.
            return

        if cutlass.const_expr(cfg.tile_size_q in (16, 32)):
            # Wide tile-Q Swaps path has multiple softmax scale groups and may
            # span several 64-column O chunks.
            q_repeats = max(cfg.tile_size_q // 8, 1)
            num_scale_groups = cfg.num_softmax_scale_groups
            output_pair_regs = cfg.num_fp16_output_regs
            output_f32_regs = output_pair_regs * 2
            num_o_chunks = cfg.headdim // 64
            self._swaps_wide_tail_epilogue(
                stage_info,
                tail_o_stage_idx_0=tail_o_stage_idx_0,
                tail_o_stage_idx_1=tail_o_stage_idx_1,
                inst0_new_max_arr=inst0_new_max_arr,
                inst0_sum_arr=inst0_sum_arr,
                inst1_new_max_arr=inst1_new_max_arr,
                inst1_sum_arr=inst1_sum_arr,
                tmem_row_base=tmem_row_base,
                o_base_col=o_base_col,
                q_repeats=q_repeats,
                num_scale_groups=num_scale_groups,
                output_pair_regs=output_pair_regs,
                output_f32_regs=output_f32_regs,
                num_o_chunks=num_o_chunks,
            )
            return

        # Tile-Q=8 Swaps path has exactly two scale groups and a compact tail
        # helper specialized for that register layout.
        self._swaps_q8_tail_epilogue(
            stage_info,
            tail_o_stage_idx_0=tail_o_stage_idx_0,
            tail_o_stage_idx_1=tail_o_stage_idx_1,
            inst0_new_max_arr=inst0_new_max_arr,
            inst0_sum_arr=inst0_sum_arr,
            inst1_new_max_arr=inst1_new_max_arr,
            inst1_sum_arr=inst1_sum_arr,
            tmem_row_base=tmem_row_base,
            o_base_col=o_base_col,
        )
        return
