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

"""TMEM score resources for QK MMA and online softmax."""

from dataclasses import dataclass
from typing import ClassVar, Optional

from cutlass.experimental import primitives as prims

import cutlass
import cutlass.cute as cute
from cutlass import Boolean, Float32, Int32, Int64, Uint32
from cutlass.experimental import primitives as cprims
from cutlass.experimental.task_scheduling.enums import WorkAttr
from cutlass.experimental.task_scheduling.memory import SmemAllocation, TmemAllocation
from cutlass.experimental.task_scheduling.resources import (
    MemoryResource,
    StageInfo,
    TaskLocalVariable,
    consumer_work,
    producer_work,
)

from ...helpers.constants import (
    OCTET_LANES,
    QUAD_LANE_MASK,
    QUAD_LANE_SHIFT,
    SCORE_ROWS_PER_Q_PAIR,
    SCORE_TOKENS_PER_QK_GROUP,
    SMEM_WORD_BYTES,
    TCGEN05_16X256B_SHAPE,
    TCGEN05_DESC_NEXT_K_BLOCK_UNITS,
    TCGEN05_DESC_WRAPPED_K_BLOCK_UNITS,
    WARP_LANES,
    WARPGROUP_THREADS,
    WARPGROUP_WARPS,
)


from ...helpers.layout import (
    _TASK_CACHE_LANE_IDX,
    _TASK_CACHE_TMEM_BASE_OFFSET,
    _TASK_CACHE_WARP_GRP_THREAD_IDX,
    _TASK_CACHE_WARP_IDX,
    decode_gen_task_cache,
    num_q_repeats,
    num_s_regs_per_thread,
    num_softmax_scale_groups,
    q_p_desc_k_block_wrap_units,
    q_stage_smem_element_offset,
    smem_array,
    softmax_scratch_words,
)
from ...helpers.math import (
    ffma2,
    float_to_u32_for_atomic_max,
    init_softmax_scratch_u32,
    mma_k_step_for_qkv,
    mma_kind_for_qkv,
    neg_max_f32,
    qkv_dtype,
    smem_atomic_max_u32,
    u32_to_float_for_atomic_max,
)
from ...helpers.mask import MaskType
from ...helpers.ops import (
    float_to_u32_bits,
    freeze_smem_descriptor,
    softmax_sum_state_ptr,
    tcgen05_ld_16x32bx2_f32,
    tcgen05_second_panel_addr,
    u32_bits_to_float,
)
from ...helpers.stage import MlaStage
from ...helpers.tile import (
    batch_idx_for_stage_cfg,
    cta_idx_kv_for_stage,
    cta_idx_q_for_stage,
    global_kv_tile_idx,
    head_idx_for_stage,
    runtime_seq_len_kv_for_effective_head,
    runtime_seq_len_kv_from_task_cache,
    softmax_kv_tile_idx,
)

from .common import (
    MlaResource,
)

# =====================================================================
# TmemSResource — S scores in TMEM, UmmaProducerAsync pipeline
# =====================================================================


@dataclass(kw_only=True)
class TmemSResource(MlaResource):
    """TMEM score resource plus task-local online-softmax state."""

    _task_local_specs: ClassVar[tuple[tuple, ...]] = (
        ("old_max_arr", cutlass.Array, None, "Previous running softmax maxima."),
        ("sum_arr", cutlass.Array, None, "Running softmax denominators."),
        ("new_max_arr", cutlass.Array, None, "Current running softmax maxima."),
        ("local_sum_arr", cutlass.Array, None, "Local denominator contributions."),
        ("s_arr", cutlass.Array, None, "Loaded S scores for the current tile."),
    )
    inst_id: cutlass.Constexpr[int] = 0
    scale_softmax_log2: Float32 = None
    p_ref: Optional[MemoryResource] = None
    global_ref: Optional[MemoryResource] = None
    cache_seqs: object = None
    head_idx: object = None
    batch_idx: object = None
    cta_idx_q: object = None
    cta_idx_kv: object = None
    sync_barrier_id: cutlass.Constexpr[int] = 0
    q_desc_current: object = None
    q_desc_rope_current: object = None
    new_max_state: object = None
    sum_state: object = None
    _scratch_alloc: cutlass.Constexpr[Optional[SmemAllocation]] = None
    _softmax_scratch: object = None
    _p_local_sum_arr: object = None
    old_max_arr: cutlass.Constexpr[TaskLocalVariable] = (
        TaskLocalVariable.uninitialized()
    )
    sum_arr: cutlass.Constexpr[TaskLocalVariable] = TaskLocalVariable.uninitialized()
    new_max_arr: cutlass.Constexpr[TaskLocalVariable] = (
        TaskLocalVariable.uninitialized()
    )
    local_sum_arr: cutlass.Constexpr[TaskLocalVariable] = (
        TaskLocalVariable.uninitialized()
    )
    s_arr: cutlass.Constexpr[TaskLocalVariable] = TaskLocalVariable.uninitialized()

    @cute.jit
    def _reset_softmax_state(self):
        """Reset per-resource softmax max and sum state arrays."""
        num_scale_groups = num_softmax_scale_groups(self.cfg)
        self.new_max_state = cutlass.Array(
            Float32,
            num_scale_groups,
            space=cutlass.AddressSpace.rmem,
        )
        self.sum_state = cutlass.Array(
            Float32,
            num_scale_groups,
            space=cutlass.AddressSpace.rmem,
        )
        for idx in cutlass.range_constexpr(num_scale_groups):
            self.new_max_state[idx] = neg_max_f32()
            self.sum_state[idx] = Float32(0.0)

    def get_smem_requirements(self):
        """Return the softmax scratch SMEM allocation."""
        if self._scratch_alloc is None:
            self._scratch_alloc = SmemAllocation(
                name=f"{self.name}_softmaxScratch",
                size_bytes=softmax_scratch_words(self.cfg) * SMEM_WORD_BYTES,
                alignment=16,
            )
        return [self._scratch_alloc]

    def get_tmem_requirements(self):
        """Return the TMEM allocation for score tiles."""
        if self._tmem_alloc is None:
            num_stages = (
                self.pipeline_config.num_stages
                if self.cfg.kernel_variant == "keeps_mma_ab"
                and self.pipeline_config is not None
                else 1
            )
            self._tmem_alloc = TmemAllocation(
                name=f"{self.name}",
                num_columns=self.cfg.tmem_s_cols * num_stages,
            )
        return [self._tmem_alloc]

    @cute.jit
    def _make_initial_softmax_vars(self):
        """Create fresh softmax state arrays for the current work tile."""
        self._reset_softmax_state()
        num_scale_groups = num_softmax_scale_groups(self.cfg)
        num_s_regs = num_s_regs_per_thread(self.cfg)
        self._p_local_sum_arr = cutlass.Array(
            Float32,
            num_scale_groups,
            space=cutlass.AddressSpace.rmem,
        )
        old_max_arr = cutlass.Array(
            Float32, num_scale_groups, space=cutlass.AddressSpace.rmem
        )
        sum_arr = cutlass.Array(
            Float32, num_scale_groups, space=cutlass.AddressSpace.rmem
        )
        new_max_arr = cutlass.Array(
            Float32, num_scale_groups, space=cutlass.AddressSpace.rmem
        )
        local_sum_arr = cutlass.Array(
            Float32,
            num_scale_groups,
            space=cutlass.AddressSpace.rmem,
        )
        s_arr = cutlass.Array(Float32, num_s_regs, space=cutlass.AddressSpace.rmem)
        for idx in cutlass.range_constexpr(num_scale_groups):
            old_max_arr[idx] = neg_max_f32()
            sum_arr[idx] = Float32(0.0)
            new_max_arr[idx] = neg_max_f32()
            local_sum_arr[idx] = Float32(0.0)
            self._p_local_sum_arr[idx] = Float32(0.0)
        for idx in cutlass.range_constexpr(num_s_regs):
            s_arr[idx] = neg_max_f32()
        return old_max_arr, sum_arr, new_max_arr, local_sum_arr, s_arr

    @consumer_work(
        work_attrs=WorkAttr.AUXILIARY,
        returns=(old_max_arr, sum_arr, new_max_arr, local_sum_arr, s_arr),
    )
    @cute.jit
    def init_softmax_state(self, stage_info: StageInfo):
        """Create softmax arrays and scratch state for the first work tile."""
        # Consumer aux work belongs to the softmax task.  It creates the local
        # arrays that will be threaded through update_softmax(),
        # materialize_p(), and update_softmax_sum().
        context = stage_info.context
        self._init_tmem_state(stage_info)
        self._softmax_scratch = smem_array(
            context,
            self._scratch_alloc,
            Uint32,
            softmax_scratch_words(self.cfg),
        )
        self.q_desc_current = Int64(0)
        self.q_desc_rope_current = Int64(0)
        if cutlass.const_expr(self.cfg.kernel_variant == "keeps_mma_ab"):
            thread_idx = cute.arch.thread_idx()[0]
            state_ptr = self._softmax_scratch.data_ptr(thread_idx)
            state_ptr.store(
                float_to_u32_bits(neg_max_f32()),
                is_volatile=True,
                alignment=4,
            )
            softmax_sum_state_ptr(state_ptr).store(
                float_to_u32_bits(Float32(0.0)),
                is_volatile=True,
                alignment=4,
            )
        return self._make_initial_softmax_vars()

    @consumer_work(
        work_attrs=WorkAttr.AUXILIARY,
        returns=(old_max_arr, sum_arr, new_max_arr, local_sum_arr, s_arr),
    )
    @cute.jit
    def init_softmax_work_tile_state(self, stage_info: StageInfo):
        """Create fresh softmax arrays for each persistent work tile."""
        # Persistent schedules reuse the same task graph for multiple work
        # tiles, so the consumer-side softmax state must be reset per tile.
        del stage_info
        return self._make_initial_softmax_vars()

    @producer_work(work_attrs=WorkAttr.AUXILIARY)
    @cute.jit
    def reset_softmax_work_tile_state(self, stage_info: StageInfo) -> None:
        """Reset per-resource softmax state for a persistent work tile."""
        # Producer aux work mirrors the consumer reset for schedules where this
        # resource later produces QK scores into TMEM.
        del stage_info
        self._make_initial_softmax_vars()

    @producer_work(work_attrs=WorkAttr.AUXILIARY)
    @cute.jit
    def set_q_desc(self, stage_info: StageInfo, *, q_desc, q_desc_rope):
        """Cache the Q descriptor once while the Q SMEM stage is live."""
        # q_desc() returns descriptors from the SmemQ consumer side.  This
        # producer aux method stores them in the score resource so each QK MMA
        # call can reuse them without adding another schedule edge.
        del stage_info
        self.q_desc_current = q_desc
        self.q_desc_rope_current = q_desc_rope

    @producer_work
    @cute.jit
    def qk_mma(
        self,
        stage_info: StageInfo,
        *,
        kv_desc,
        k_subtile_idx: cutlass.Constexpr[int],
    ):
        """Issue the staged QK MMA for one 128-wide MLA head-dim slice."""
        # Producer work for TmemSResource: the MMA warp writes score fragments
        # into the acquired TMEM score stage.  The paired softmax task consumes
        # this TMEM stage through update_softmax().
        cfg = self.cfg
        qk_stage_idx = k_subtile_idx
        q_stage_offset_bytes = Int32(
            q_stage_smem_element_offset(cfg, qk_stage_idx) * cfg.qkv_dtype_bytes
        )
        q_desc = freeze_smem_descriptor(self.q_desc_current)
        if cutlass.const_expr(cfg.is_fp8_qkv() and cfg.rope_dim == 64):
            if cutlass.const_expr(
                qk_stage_idx == cfg.latent_dim // cfg.head_dim_per_stage_kv
            ):
                q_desc = freeze_smem_descriptor(self.q_desc_rope_current)
        kv_desc = freeze_smem_descriptor(kv_desc)
        if cutlass.const_expr(
            not cfg.is_fp8_qkv()
            or cfg.rope_dim != 64
            or qk_stage_idx != cfg.latent_dim // cfg.head_dim_per_stage_kv
        ):
            q_desc = q_desc + (q_stage_offset_bytes >> 4)

        task_cache = decode_gen_task_cache(stage_info)
        stage_col_offset = Int32(0)
        if cutlass.const_expr(cfg.kernel_variant == "keeps_mma_ab"):
            stage_col_offset = stage_info.stage_idx * Int32(cfg.tmem_s_cols)
        tmem_ptr = prims.make_tmem_ptr(
            task_cache[_TASK_CACHE_TMEM_BASE_OFFSET]
            + Int32(self._tmem_alloc.offset)
            + stage_col_offset,
            Float32,
        )
        if cutlass.const_expr(cfg.kernel_variant == "keeps_mma_ab"):
            idesc = cprims.Tcgen05InstrDesc.build(
                c_dtype=Float32,
                a_dtype=qkv_dtype(cfg),
                b_dtype=qkv_dtype(cfg),
                n_dim=cfg.tile_size_kv,
                m_dim=cfg.tile_size_q,
            )
        else:
            idesc = cprims.Tcgen05InstrDesc.build(
                c_dtype=Float32,
                a_dtype=qkv_dtype(cfg),
                b_dtype=qkv_dtype(cfg),
                n_dim=cfg.tile_size_q,
                m_dim=cfg.tile_size_kv,
            )
        k_block_count = cfg.qk_head_stage_width(qk_stage_idx) // mma_k_step_for_qkv(cfg)
        if prims.elect_sync():
            scale_d = Boolean(qk_stage_idx != 0)
            for k_block in cutlass.range_constexpr(k_block_count):
                if cutlass.const_expr(cfg.kernel_variant == "keeps_mma_ab"):
                    prims.tcgen05_mma(
                        mma_kind_for_qkv(cfg),
                        prims.CTAGroup.CTA_1,
                        tmem_ptr,
                        q_desc,
                        kv_desc,
                        idesc,
                        scale_d,
                    )
                else:
                    prims.tcgen05_mma(
                        mma_kind_for_qkv(cfg),
                        prims.CTAGroup.CTA_1,
                        tmem_ptr,
                        kv_desc,
                        q_desc,
                        idesc,
                        scale_d,
                    )
                scale_d = Boolean(True)
                if cutlass.const_expr(k_block + 1 < k_block_count):
                    if cutlass.const_expr(k_block == 3):
                        kv_desc = kv_desc + Int32(TCGEN05_DESC_WRAPPED_K_BLOCK_UNITS)
                        q_desc = q_desc + Int32(q_p_desc_k_block_wrap_units(cfg))
                    else:
                        kv_desc = kv_desc + Int32(TCGEN05_DESC_NEXT_K_BLOCK_UNITS)
                        q_desc = q_desc + Int32(TCGEN05_DESC_NEXT_K_BLOCK_UNITS)

    @consumer_work(returns=(old_max_arr, sum_arr, new_max_arr, local_sum_arr, s_arr))
    @cute.jit
    def update_softmax(
        self,
        stage_info: StageInfo,
        *,
        old_max_arr,
        sum_arr,
        new_max_arr,
        local_sum_arr,
        s_arr,
        section: cutlass.Constexpr[MlaStage],
    ):
        """Read S from TMEM and update the grouped-head softmax state."""
        # Consumer work for TmemSResource: softmax waits for the QK MMA score
        # stage, loads S from TMEM, applies masks, and returns updated local
        # softmax state to the captured schedule.
        cfg = self.cfg
        task_cache = decode_gen_task_cache(stage_info)
        warp_grp_thread_idx = task_cache[_TASK_CACHE_WARP_GRP_THREAD_IDX]
        if cutlass.const_expr(self.cfg.kernel_variant == "keeps_mma_ab"):
            warp_grp_thread_idx = cute.arch.thread_idx()[0]
        q_repeats = num_q_repeats(cfg)
        num_scale_groups = num_softmax_scale_groups(cfg)
        old_max_vals = cutlass.Array(
            Float32, num_scale_groups, space=cutlass.AddressSpace.rmem
        )
        sum_vals = cutlass.Array(
            Float32, num_scale_groups, space=cutlass.AddressSpace.rmem
        )
        new_max_vals = cutlass.Array(
            Float32, num_scale_groups, space=cutlass.AddressSpace.rmem
        )
        local_max_vals = cutlass.Array(
            Float32, num_scale_groups, space=cutlass.AddressSpace.rmem
        )
        s_vals = cutlass.Array(
            Float32, num_s_regs_per_thread(cfg), space=cutlass.AddressSpace.rmem
        )

        for idx in cutlass.range_constexpr(num_scale_groups):
            if cutlass.const_expr(cfg.kernel_variant == "keeps_mma_ab"):
                state_ptr = self._softmax_scratch.data_ptr(warp_grp_thread_idx)
                old_max_vals[idx] = u32_bits_to_float(
                    state_ptr.load(is_volatile=True, alignment=4)
                )
                sum_vals[idx] = u32_bits_to_float(
                    softmax_sum_state_ptr(state_ptr).load(is_volatile=True, alignment=4)
                )
            else:
                old_max_vals[idx] = new_max_arr[idx]
                sum_vals[idx] = sum_arr[idx]
            new_max_vals[idx] = old_max_vals[idx]
            local_max_vals[idx] = neg_max_f32()
        for idx in cutlass.range_constexpr(num_s_regs_per_thread(cfg)):
            s_vals[idx] = neg_max_f32()

        # Normalize both score layouts into one local S register array before
        # masking: keeps-MMA-AB reads a single TMEM panel, swaps-MMA-AB reads
        # two 16x256b panels.
        stage_col_offset = Int32(0)
        if cutlass.const_expr(cfg.kernel_variant == "keeps_mma_ab"):
            stage_col_offset = stage_info.stage_idx * Int32(cfg.tmem_s_cols)
        base_addr = (
            task_cache[_TASK_CACHE_TMEM_BASE_OFFSET]
            + Int32(self._tmem_alloc.offset)
            + stage_col_offset
        )
        if cutlass.const_expr(cfg.kernel_variant == "keeps_mma_ab"):
            loaded = tcgen05_ld_16x32bx2_f32(
                prims.make_tmem_ptr(base_addr, Float32),
                num=cfg.tile_size_kv // 2,
                offset=Int32(cfg.tile_size_kv // 2),
            )
            prims.tcgen05_wait(kind=prims.Tcgen05Wait.LOAD)
            for reg_idx in cutlass.range_constexpr(num_s_regs_per_thread(cfg)):
                s_vals[reg_idx] = loaded[reg_idx]
        else:
            loaded0 = prims.tcgen05_ld(
                TCGEN05_16X256B_SHAPE,
                prims.make_tmem_ptr(base_addr, Float32),
                num=q_repeats,
            )
            loaded1 = prims.tcgen05_ld(
                TCGEN05_16X256B_SHAPE,
                prims.make_tmem_ptr(tcgen05_second_panel_addr(base_addr), Float32),
                num=q_repeats,
            )
            prims.tcgen05_wait(kind=prims.Tcgen05Wait.LOAD)
            for repeat_idx in cutlass.range_constexpr(q_repeats):
                ld_base = repeat_idx * 4
                s_vals[ld_base + 0] = loaded0[ld_base + 0]
                s_vals[ld_base + 1] = loaded0[ld_base + 1]
                s_vals[ld_base + 2] = loaded0[ld_base + 2]
                s_vals[ld_base + 3] = loaded0[ld_base + 3]
                s_vals[q_repeats * 4 + ld_base + 0] = loaded1[ld_base + 0]
                s_vals[q_repeats * 4 + ld_base + 1] = loaded1[ld_base + 1]
                s_vals[q_repeats * 4 + ld_base + 2] = loaded1[ld_base + 2]
                s_vals[q_repeats * 4 + ld_base + 3] = loaded1[ld_base + 3]

        batch_idx = batch_idx_for_stage_cfg(self.batch_idx, cfg, stage_info)
        cta_idx_q = cta_idx_q_for_stage(self.cta_idx_q, stage_info)
        cta_idx_kv = cta_idx_kv_for_stage(self.cta_idx_kv, stage_info)
        seq_len_kv = runtime_seq_len_kv_from_task_cache(
            cfg,
            task_cache,
            cta_idx_q,
            self.cu_seqlens_q,
            batch_idx,
        )
        local_tile_idx = softmax_kv_tile_idx(cfg, stage_info, self.inst_id)
        tile_idx = global_kv_tile_idx(cfg, local_tile_idx, seq_len_kv, cta_idx_kv)
        tile_offset_k = tile_idx * Int32(cfg.tile_size_kv)
        next_tile_offset_k = tile_offset_k + Int32(cfg.tile_size_kv)
        should_apply_dense_mask = (
            (seq_len_kv % Int32(cfg.tile_size_kv)) != Int32(0)
        ) or (next_tile_offset_k > seq_len_kv)
        needs_row_causal_mask = cutlass.const_expr(
            cfg.mask_type == MaskType.CAUSAL.value
            and cfg.groups_tokens_heads_q_ratio > 1
        )
        if cutlass.const_expr(needs_row_causal_mask):
            min_seq_len_kv = seq_len_kv - Int32(cfg.groups_tokens_heads_q_ratio - 1)
            should_apply_dense_mask = should_apply_dense_mask or (
                next_tile_offset_k > min_seq_len_kv
            )
        if should_apply_dense_mask:
            # The CTA domain already captures dense and non-grouped causal
            # visibility. Only grouped causal rows need a narrower row limit.
            warp_idx = task_cache[_TASK_CACHE_WARP_IDX]
            lane_idx = task_cache[_TASK_CACHE_LANE_IDX]
            if cutlass.const_expr(cfg.kernel_variant == "keeps_mma_ab"):
                local_col_base = (lane_idx >> Int32(4)) * Int32(cfg.tile_size_kv // 2)
                local_row_idx = warp_idx * Int32(16) + (lane_idx & Int32(0xF))
                effective_head_idx = (
                    head_idx_for_stage(self.head_idx, cfg, stage_info) + local_row_idx
                )
                row_seq_len_kv = seq_len_kv
                if cutlass.const_expr(needs_row_causal_mask):
                    row_seq_len_kv = runtime_seq_len_kv_for_effective_head(
                        cfg,
                        self.cache_seqs,
                        batch_idx,
                        cta_idx_q,
                        effective_head_idx,
                        self.cu_seqlens_q,
                    )
                for reg_idx in cutlass.range_constexpr(num_s_regs_per_thread(cfg)):
                    token_idx = tile_offset_k + local_col_base + Int32(reg_idx)
                    if token_idx >= row_seq_len_kv:
                        s_vals[reg_idx] = neg_max_f32()
            else:
                local_idx_k0 = warp_idx * Int32(WARP_LANES) + (
                    lane_idx >> Int32(QUAD_LANE_SHIFT)
                )
            if cutlass.const_expr(cfg.kernel_variant != "keeps_mma_ab"):
                # Score TMEM is read as two panels: each repeat contributes
                # the first two 8-token groups, then the second panel carries
                # the next two groups.
                head_idx = head_idx_for_stage(self.head_idx, cfg, stage_info)
                local_q_pair = (lane_idx & Int32(QUAD_LANE_MASK)) * Int32(
                    SCORE_ROWS_PER_Q_PAIR
                )
                for repeat_idx in cutlass.range_constexpr(q_repeats):
                    effective_head_idx_0 = (
                        head_idx
                        + Int32(repeat_idx * SCORE_TOKENS_PER_QK_GROUP)
                        + local_q_pair
                    )
                    effective_head_idx_1 = effective_head_idx_0 + Int32(1)
                    seq_len_kv_0 = seq_len_kv
                    seq_len_kv_1 = seq_len_kv
                    if cutlass.const_expr(needs_row_causal_mask):
                        seq_len_kv_0 = runtime_seq_len_kv_for_effective_head(
                            cfg,
                            self.cache_seqs,
                            batch_idx,
                            cta_idx_q,
                            effective_head_idx_0,
                            self.cu_seqlens_q,
                        )
                        seq_len_kv_1 = runtime_seq_len_kv_for_effective_head(
                            cfg,
                            self.cache_seqs,
                            batch_idx,
                            cta_idx_q,
                            effective_head_idx_1,
                            self.cu_seqlens_q,
                        )
                    s_base = repeat_idx * 4
                    s_second_panel_base = q_repeats * 4 + s_base
                    token_idx = tile_offset_k + local_idx_k0
                    if token_idx >= seq_len_kv_0:
                        s_vals[s_base + 0] = neg_max_f32()
                    if token_idx >= seq_len_kv_1:
                        s_vals[s_base + 1] = neg_max_f32()
                    token_idx = (
                        tile_offset_k + local_idx_k0 + Int32(SCORE_TOKENS_PER_QK_GROUP)
                    )
                    if token_idx >= seq_len_kv_0:
                        s_vals[s_base + 2] = neg_max_f32()
                    if token_idx >= seq_len_kv_1:
                        s_vals[s_base + 3] = neg_max_f32()
                    token_idx = (
                        tile_offset_k
                        + local_idx_k0
                        + Int32(2 * SCORE_TOKENS_PER_QK_GROUP)
                    )
                    if token_idx >= seq_len_kv_0:
                        s_vals[s_second_panel_base + 0] = neg_max_f32()
                    if token_idx >= seq_len_kv_1:
                        s_vals[s_second_panel_base + 1] = neg_max_f32()
                    token_idx = (
                        tile_offset_k
                        + local_idx_k0
                        + Int32(3 * SCORE_TOKENS_PER_QK_GROUP)
                    )
                    if token_idx >= seq_len_kv_0:
                        s_vals[s_second_panel_base + 2] = neg_max_f32()
                    if token_idx >= seq_len_kv_1:
                        s_vals[s_second_panel_base + 3] = neg_max_f32()

        if cutlass.const_expr(cfg.kernel_variant == "keeps_mma_ab"):
            # Keeps-MMA-AB keeps the softmax state in scratch words indexed by
            # CTA thread, so only a warp-level max is needed here.
            local_max = old_max_vals[0]
            for reg_idx in cutlass.range_constexpr(num_s_regs_per_thread(cfg)):
                local_max = cute.math.max(local_max, s_vals[reg_idx], ftz=True)
            local_max = cute.math.max(
                local_max,
                Float32(
                    cprims.shfl_sync(
                        thread_mask=0xFFFFFFFF,
                        val=local_max,
                        offset=16,
                        mask_and_clamp=0x1F,
                        kind=cprims.Shfl.BFLY,
                    )
                ),
                ftz=True,
            )
            new_max_vals[0] = local_max
            self.new_max_state[0] = local_max
        else:
            # Swaps-MMA-AB reduces per-scale maxima first within each warp and
            # then across the four softmax columns through SMEM atomics.
            for scale_idx in cutlass.range_constexpr(num_scale_groups):
                s_base = Int32((scale_idx // 2) * 4 + (scale_idx & 1))
                s_stride = Int32(q_repeats * 4)
                local_max = cute.math.max(
                    cute.math.max(s_vals[s_base + 0], s_vals[s_base + 2], ftz=True),
                    cute.math.max(
                        s_vals[s_base + s_stride],
                        s_vals[s_base + s_stride + Int32(2)],
                        ftz=True,
                    ),
                    ftz=True,
                )
                local_max = cute.math.max(local_max, old_max_vals[scale_idx], ftz=True)
                local_max = cute.math.max(
                    local_max,
                    Float32(
                        cprims.shfl_sync(
                            thread_mask=0xFFFFFFFF,
                            val=local_max,
                            offset=16,
                            mask_and_clamp=0x1F,
                            kind=cprims.Shfl.BFLY,
                        )
                    ),
                    ftz=True,
                )
                local_max = cute.math.max(
                    local_max,
                    Float32(
                        cprims.shfl_sync(
                            thread_mask=0xFFFFFFFF,
                            val=local_max,
                            offset=8,
                            mask_and_clamp=0x1F,
                            kind=cprims.Shfl.BFLY,
                        )
                    ),
                    ftz=True,
                )
                local_max_vals[scale_idx] = local_max

            # This is state initialization, not schedule dispatch: the first
            # softmax tile in a work tile must clear the shared atomic scratch.
            # The sync covers one four-warp softmax group.
            if cutlass.const_expr(section == MlaStage.Head):
                init_softmax_scratch_u32(
                    self._softmax_scratch,
                    warp_grp_thread_idx,
                    WARPGROUP_WARPS * num_softmax_scale_groups(cfg),
                )
                prims.barrier_cta_sync(
                    barrier_id=self.sync_barrier_id, thread_count=WARPGROUP_THREADS
                )
            elif cutlass.const_expr(section == MlaStage.Loop):
                if stage_info.loop_offset == stage_info.loop_start:
                    init_softmax_scratch_u32(
                        self._softmax_scratch,
                        warp_grp_thread_idx,
                        WARPGROUP_WARPS * num_softmax_scale_groups(cfg),
                    )
                    prims.barrier_cta_sync(
                        barrier_id=self.sync_barrier_id,
                        thread_count=WARPGROUP_THREADS,
                    )

            lane_idx = task_cache[_TASK_CACHE_LANE_IDX]
            col_group_idx = lane_idx & Int32(QUAD_LANE_MASK)
            if lane_idx < Int32(OCTET_LANES):
                atomic_reduce_base = col_group_idx * Int32(num_scale_groups)
                for scale_idx in cutlass.range_constexpr(num_scale_groups):
                    smem_atomic_max_u32(
                        self._softmax_scratch.data_ptr(
                            atomic_reduce_base + Int32(scale_idx)
                        ),
                        float_to_u32_for_atomic_max(local_max_vals[scale_idx]),
                    )
            prims.barrier_cta_sync(
                barrier_id=self.sync_barrier_id, thread_count=WARPGROUP_THREADS
            )

            reduced_max_ptr = self._softmax_scratch.data_ptr(
                col_group_idx * Int32(num_scale_groups)
            )
            reduced_max = reduced_max_ptr.load(
                count=num_scale_groups,
                alignment=16 if num_scale_groups == 4 else 8,
            )
            for scale_idx in cutlass.range_constexpr(num_scale_groups):
                new_max_vals[scale_idx] = u32_to_float_for_atomic_max(
                    reduced_max[scale_idx]
                )
                self.new_max_state[scale_idx] = new_max_vals[scale_idx]
        for scale_idx in cutlass.range_constexpr(num_scale_groups):
            old_max_arr[scale_idx] = old_max_vals[scale_idx]
            sum_arr[scale_idx] = sum_vals[scale_idx]
            new_max_arr[scale_idx] = new_max_vals[scale_idx]
        for idx in cutlass.range_constexpr(num_s_regs_per_thread(cfg)):
            s_arr[idx] = s_vals[idx]
        return old_max_arr, sum_arr, new_max_arr, local_sum_arr, s_arr

    @consumer_work(
        work_attrs=WorkAttr.AUXILIARY,
        returns=(old_max_arr, sum_arr, new_max_arr, local_sum_arr, s_arr),
    )
    @cute.jit
    def update_softmax_sum(
        self,
        stage_info: StageInfo,
        *,
        old_max_arr,
        sum_arr,
        new_max_arr,
        local_sum_arr,
        s_arr,
    ):
        """Apply the online-softmax sum correction after P is materialized."""
        # This consumer aux step runs after SmemP/TmemP has produced P and
        # published local_sum_arr.  It updates the running denominator for the
        # next score tile without consuming a new score payload.
        if cutlass.const_expr(self.cfg.kernel_variant == "keeps_mma_ab"):
            state_idx = cute.arch.thread_idx()[0]
            state_ptr = self._softmax_scratch.data_ptr(state_idx)
            old_max = old_max_arr[0]
            new_max = u32_bits_to_float(state_ptr.load(is_volatile=True, alignment=4))
            updated_sum = u32_bits_to_float(
                softmax_sum_state_ptr(state_ptr).load(is_volatile=True, alignment=4)
            )
            prims.barrier_cta_sync(
                barrier_id=self.sync_barrier_id,
                thread_count=WARPGROUP_THREADS,
            )
            self.new_max_state[0] = new_max
            self.sum_state[0] = updated_sum
            old_max_arr[0] = old_max
            sum_arr[0] = updated_sum
            new_max_arr[0] = new_max
            return old_max_arr, sum_arr, new_max_arr, local_sum_arr, s_arr
        for scale_base in cutlass.range_constexpr(
            0, num_softmax_scale_groups(self.cfg), 2
        ):
            old_max_0 = old_max_arr[scale_base]
            old_max_1 = old_max_arr[scale_base + 1]
            new_max_0 = new_max_arr[scale_base]
            new_max_1 = new_max_arr[scale_base + 1]
            local_sum_0 = local_sum_arr[scale_base]
            local_sum_1 = local_sum_arr[scale_base + 1]
            if self.p_ref is not None:
                local_sum_0 = self._p_local_sum_arr[scale_base]
                local_sum_1 = self._p_local_sum_arr[scale_base + 1]
            sum_0 = sum_arr[scale_base]
            sum_1 = sum_arr[scale_base + 1]

            exp_scale0 = Float32(0.0)
            exp_scale1 = Float32(0.0)
            if (old_max_0 != neg_max_f32()) and (new_max_0 != neg_max_f32()):
                exp_scale0 = cute.math.exp2(
                    self.scale_softmax_log2 * (old_max_0 - new_max_0),
                    fastmath=True,
                )
            if (old_max_1 != neg_max_f32()) and (new_max_1 != neg_max_f32()):
                exp_scale1 = cute.math.exp2(
                    self.scale_softmax_log2 * (old_max_1 - new_max_1),
                    fastmath=True,
                )
            updated_sums = ffma2(
                (exp_scale0, exp_scale1),
                (sum_0, sum_1),
                (local_sum_0, local_sum_1),
            )
            self.sum_state[scale_base] = updated_sums[0]
            self.sum_state[scale_base + 1] = updated_sums[1]
            sum_arr[scale_base] = updated_sums[0]
            sum_arr[scale_base + 1] = updated_sums[1]
            old_max_arr[scale_base] = old_max_0
            old_max_arr[scale_base + 1] = old_max_1
            new_max_arr[scale_base] = new_max_0
            new_max_arr[scale_base + 1] = new_max_1
            local_sum_arr[scale_base] = local_sum_0
            local_sum_arr[scale_base + 1] = local_sum_1
        return old_max_arr, sum_arr, new_max_arr, local_sum_arr, s_arr


# =====================================================================
# TmemSKeepsResource — Keeps-MMA-AB S scores in TMEM
# =====================================================================


@dataclass(kw_only=True)
class TmemSKeepsResource(TmemSResource):
    """Keeps-MMA-AB score resource with a single softmax scale group."""
