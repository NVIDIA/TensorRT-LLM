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

"""``TmemSResource`` — BMM1 accumulator / softmax input.

Producer: QK MMA → S in TMEM. Consumer: load S to registers, maintain
running row max/sum, apply optional causal / sliding-window / sink masks,
publish softmax stats for ``SmemPResource``.
"""

from dataclasses import dataclass
from typing import ClassVar

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, Uint32
from cutlass.experimental import primitives as prims

from cutlass.experimental.task_scheduling.enums import WorkAttr
from cutlass.experimental.task_scheduling.memory import (
    ResourceContext,
    SmemAllocation,
    TmemAllocation,
)
from cutlass.experimental.task_scheduling.resources import (
    MemoryResource,
    StageInfo,
    TaskLocalVariable,
    consumer_work,
    producer_work,
)

from ..fmha_decode_config import FmhaDecodeConfig
from ...placeholder_helpers import (
    _placeholder_local_array,
    _placeholder_smem_array,
)
from .helpers_common import (
    Constexpr,
    DecodeGenResourceBase,
    ResourceVars,
    ffma2,
    _TASK_CACHE_LANE_IDX,
    _TASK_CACHE_KV_RAW_TILE_BASE,
    _TASK_CACHE_KV_VALID_TILE_END,
    _TASK_CACHE_KV_WINDOW_START,
    _TASK_CACHE_TMEM_BASE_OFFSET,
    _TASK_CACHE_WARP_GRP_THREAD_IDX,
    _TASK_CACHE_WARP_IDX,
    _clamp_valid_tile_idx,
    _decode_gen_task_cache,
    _freeze_smem_descriptor,
    _is_last_loop_iteration,
    _keeps_col_base,
    _keeps_row_idx,
    _keeps_tcgen05_ld,
    _logical_q_group_idx,
    _mma_k_step,
    _mma_kind_for_qkv,
    _neg_max_f32,
    _softmax_scale_pair_width,
    _q_row_is_valid_for_seq,
    _q_row_token_and_local_head,
    _q_group_token_base,
    _softmax_tile_idx,
)
from .helpers_kv_tile_idx import (
    _kv_tile_is_fully_unmasked_for_q_group,
    _load_runtime_seq_len_kv,
    _runtime_split_kv_global_tile_idx,
    _num_skipped_kv_tiles,
    _runtime_clamp_valid_tile_idx,
    _runtime_total_kv_tiles,
    _sliding_window_start_idx,
    _static_split_kv_global_tile_idx,
)
from .helpers_softmax import (
    _float_to_u32_for_atomic_max,
    _init_softmax_scratch_u32,
    _smem_atomic_max_u32,
    _u32_to_float_for_atomic_max,
    _wspro_reduce_max4,
)


def _qk_mma_operand_contract_for_config(
    cfg: FmhaDecodeConfig,
) -> tuple[bool, int, int]:
    """Return ``(Q-is-A, M, N)`` for the selected BMM1 orientation."""
    if cfg.use_keeps_mma_ab:
        return True, cfg.tile_size_q, cfg.tile_size_kv
    return False, cfg.tile_size_kv, cfg.tile_size_q


@dataclass(kw_only=True)
class TmemSResource(DecodeGenResourceBase):
    """TMEM score resource for BMM1 and softmax.

    Producers run K x Q^T MMA into TMEM S. Consumers load S into registers,
    compute running row maxima, and carry the softmax state used by P and
    correction.
    """

    _task_local_specs: ClassVar[tuple[tuple, ...]] = (
        ("old_max_arr", cutlass.Array, None, "Previous running softmax maximum."),
        ("sum_arr", cutlass.Array, None, "Running softmax denominator."),
        ("new_max_arr", cutlass.Array, None, "Current running softmax maximum."),
        ("s_arr", cutlass.Array, None, "Loaded S scores for the current tile."),
    )
    inst_id: Constexpr[int] = 0
    cfg: Constexpr[FmhaDecodeConfig] = None
    scale_softmax_log2: Float32 = None
    seqlens_kv: cute.Pointer | None = None
    max_seq_len_kv: Int32 = None
    seq_len_q: Int32 = None
    h_r: Int32 | None = None
    q_group_idx: Int32 | None = None
    q_ref: Constexpr[MemoryResource | None] = None
    _p_local_sum_arr: cutlass.Array | None = None
    _global_sum_arr: cutlass.Array | None = None
    _alloc: Constexpr[TmemAllocation | None] = None
    sync_barrier_id: Constexpr[int] = 0
    _scratch_alloc: Constexpr[SmemAllocation | None] = None
    _softmax_scratch_u32: cutlass.Array = None
    old_max_arr: Constexpr[TaskLocalVariable] = TaskLocalVariable.uninitialized()
    sum_arr: Constexpr[TaskLocalVariable] = TaskLocalVariable.uninitialized()
    new_max_arr: Constexpr[TaskLocalVariable] = TaskLocalVariable.uninitialized()
    s_arr: Constexpr[TaskLocalVariable] = TaskLocalVariable.uninitialized()

    def _init_placeholder_state(self) -> None:
        """Create placeholder register and scratch state for softmax."""
        num_scale_groups = self.cfg.num_softmax_scale_groups
        num_s_regs = self.cfg.num_s_regs_per_thread
        self.old_max_arr.default = _placeholder_local_array(Float32, num_scale_groups)
        self.sum_arr.default = _placeholder_local_array(Float32, num_scale_groups)
        self.new_max_arr.default = _placeholder_local_array(Float32, num_scale_groups)
        self.s_arr.default = _placeholder_local_array(Float32, num_s_regs)
        self._p_local_sum_arr = _placeholder_local_array(Float32, num_scale_groups)
        self._global_sum_arr = _placeholder_local_array(Float32, num_scale_groups)
        scratch_entries = (
            4 * num_scale_groups if self.cfg.use_keeps_mma_ab else self.cfg.tile_size_q
        )
        self._softmax_scratch_u32 = _placeholder_smem_array(Uint32, scratch_entries)

    @cute.jit
    def store_p_local_sum(self, scale_idx: int, value: Float32) -> None:
        """Publish the P producer's local denominator contribution."""
        self._p_local_sum_arr[scale_idx] = value

    @cute.jit
    def load_p_local_sum(self, scale_idx: int) -> Float32:
        """Load the local denominator published with the current P tile."""
        return self._p_local_sum_arr[scale_idx]

    @cute.jit
    def store_global_sum(self, scale_idx: int, value: Float32) -> None:
        """Publish the FP8 cross-warp denominator correction."""
        self._global_sum_arr[scale_idx] = value

    @cute.jit
    def load_global_sum(self, scale_idx: int) -> Float32:
        """Load the FP8 denominator after cross-warp correction."""
        return self._global_sum_arr[scale_idx]

    def get_smem_requirements(self) -> list[SmemAllocation]:
        """Allocate softmax scratch used for CTA-wide max reductions."""
        if self._scratch_alloc is None:
            scratch_entries = (
                4 * self.cfg.num_softmax_scale_groups
                if self.cfg.use_keeps_mma_ab
                else self.cfg.tile_size_q
            )
            self._scratch_alloc = SmemAllocation(
                name=f"{self.name}_softmaxScratch",
                size_bytes=scratch_entries * 4,
                alignment=16,
            )
        return [self._scratch_alloc]

    def get_tmem_requirements(self) -> list[TmemAllocation]:
        """Allocate TMEM S score columns for QK MMA output."""
        if self._alloc is None:
            num_stages = (
                self.pipeline_config.num_stages
                if (
                    self.pipeline_config is not None
                    and self.cfg.use_keeps_mma_ab
                    and self.cfg.num_insts_kv == 1
                )
                else 1
            )
            self._alloc = TmemAllocation(
                name=f"{self.name}",
                num_columns=self.cfg.tmem_s_cols * num_stages,
            )
        return [self._alloc]

    @cute.jit
    def _q_desc_for_head_dim_stage(
        self,
        q_desc: prims.Tcgen05SmemDesc,
        head_dim_stage_idx: Constexpr[int],
    ) -> tuple[prims.Tcgen05SmemDesc, Constexpr[int]]:
        """Select the staged-Q descriptor slice consumed by this BMM1 call."""
        cfg = self.cfg
        if cutlass.const_expr(cfg.head_dim_per_stage_kv != 0):
            q_stage_offset = Int32(
                head_dim_stage_idx
                * cfg.head_dim_kv_stage
                * cfg.tile_size_q
                * cfg.q_dtype_bytes
                // 16
            )
            q_desc = q_desc + q_stage_offset
        return q_desc, head_dim_stage_idx

    @cute.jit
    def _advance_qk_descs_after_mma_k(
        self,
        k_desc: prims.Tcgen05SmemDesc,
        q_desc: prims.Tcgen05SmemDesc,
        *,
        crosses_64b_chunk: Constexpr[bool],
    ) -> tuple[prims.Tcgen05SmemDesc, prims.Tcgen05SmemDesc]:
        """Advance K/Q descriptors to the next 16-wide MMA-K slice.

        16-bit layouts are staged as 64-column chunks. Crossing that chunk
        boundary uses the large descriptor jump; all other steps advance by one
        MMA-K slice.
        """
        cfg = self.cfg
        if cutlass.const_expr(not cfg.use_fp8_qkv and crosses_64b_chunk):
            k_desc = k_desc + Int32(1018)
            if cutlass.const_expr(cfg.tile_size_q >= 16):
                q_desc = q_desc + Int32(8 * cfg.tile_size_q - 6)
            else:
                q_desc = q_desc + Int32(58)
        else:
            k_desc = k_desc + Int32(2)
            q_desc = q_desc + Int32(2)
        return k_desc, q_desc

    @cute.jit
    def _stage_slot_offset_from_slot(self, slot: Int32) -> Int32:
        """Map a logical S pipeline slot to a TMEM column offset."""
        cfg = self.cfg
        if cutlass.const_expr(not cfg.use_keeps_mma_ab or cfg.num_insts_kv != 1):
            return Int32(0)
        return slot * Int32(cfg.tmem_s_cols)

    @cute.jit
    def _qk_head_stage_slot_offset(self, stage_info: StageInfo) -> Int32:
        """Return the TMEM S slot used by HEAD QK MMA."""
        if cutlass.const_expr(stage_info.stage_idx is not None):
            return self._stage_slot_offset_from_slot(Int32(stage_info.stage_idx))
        return self._stage_slot_offset_from_slot(Int32(0))

    @cute.jit
    def _qk_loop_stage_slot_offset(self, stage_info: StageInfo) -> Int32:
        """Return the producer TMEM S slot for a LOOP QK MMA wave."""
        if cutlass.const_expr(stage_info.stage_idx is not None):
            return self._stage_slot_offset_from_slot(Int32(stage_info.stage_idx))
        return self._stage_slot_offset_from_slot(
            (stage_info.loop_offset + Int32(1)) % Int32(2)
        )

    @cute.jit
    def _softmax_loop_stage_slot_offset(self, stage_info: StageInfo) -> Int32:
        """Return the consumer TMEM S slot read by LOOP softmax."""
        if cutlass.const_expr(stage_info.stage_idx is not None):
            return self._stage_slot_offset_from_slot(Int32(stage_info.stage_idx))
        return self._stage_slot_offset_from_slot(stage_info.loop_offset % Int32(2))

    @cute.jit
    def _create_initial_task_locals(
        self, context: ResourceContext | None = None
    ) -> ResourceVars:
        """Bind softmax scratch and initialize running max/sum state."""
        if cutlass.const_expr(
            context is not None
            and context.smem_base is not None
            and self._scratch_alloc is not None
        ):
            # Shared scratch holds encoded per-scale-group maxima for the
            # four softmax warps before they are reduced back to registers.
            scratch_ptr = context.smem_base.data_ptr() + self._scratch_alloc.offset
            self._softmax_scratch_u32 = cutlass.Array(
                scratch_ptr,
                dtype=Uint32,
                shape=(
                    (
                        4 * self.cfg.num_softmax_scale_groups
                        if self.cfg.use_keeps_mma_ab
                        else self.cfg.tile_size_q
                    ),
                ),
                addrspace=3,
            )

        num_scale_groups = self.cfg.num_softmax_scale_groups
        num_s_regs = self.cfg.num_s_regs_per_thread
        # Cross-resource mutable arrays are stored as instance attributes, not
        # consumer vars, so SmemP and TmemSoftmaxGlobal can update them in
        # place between schedule steps.
        self._p_local_sum_arr = cutlass.Array(
            Float32, num_scale_groups, space=cutlass.AddressSpace.rmem
        )
        self._global_sum_arr = cutlass.Array(
            Float32, num_scale_groups, space=cutlass.AddressSpace.rmem
        )
        result = {
            "old_max_arr": cutlass.Array(
                Float32, num_scale_groups, space=cutlass.AddressSpace.rmem
            ),
            "sum_arr": cutlass.Array(
                Float32, num_scale_groups, space=cutlass.AddressSpace.rmem
            ),
            "new_max_arr": cutlass.Array(
                Float32, num_scale_groups, space=cutlass.AddressSpace.rmem
            ),
            "s_arr": cutlass.Array(
                Float32, num_s_regs, space=cutlass.AddressSpace.rmem
            ),
        }
        for idx in cutlass.range_constexpr(num_scale_groups):
            # Initialize running max/sum state for the first K/V tile.
            result["old_max_arr"][idx] = _neg_max_f32()
            result["sum_arr"][idx] = Float32(0.0)
            result["new_max_arr"][idx] = _neg_max_f32()
            self._p_local_sum_arr[idx] = Float32(0.0)
            self._global_sum_arr[idx] = Float32(0.0)
        for idx in cutlass.range_constexpr(num_s_regs):
            # Invalid lanes start at -inf so masks and empty tiles naturally
            # contribute zero probability.
            result["s_arr"][idx] = _neg_max_f32()
        if cutlass.const_expr(not self.cfg.use_fp8_qkv):
            result["packed_p_arr"] = cutlass.Array(
                Int32, self.cfg.num_packed_p_regs, space=cutlass.AddressSpace.rmem
            )
            for idx in cutlass.range_constexpr(self.cfg.num_packed_p_regs):
                result["packed_p_arr"][idx] = Int32(0)
        return result

    @cute.jit
    def _create_work_tile_task_locals(
        self, context: ResourceContext | None = None
    ) -> ResourceVars:
        """Create per-work-tile softmax state for persistent scheduling."""
        _ = context
        # Reinitialize the softmax state for each persistent-scheduler work
        # tile while preserving the resource-level scratch allocation.
        num_scale_groups = self.cfg.num_softmax_scale_groups
        num_s_regs = self.cfg.num_s_regs_per_thread
        result = {
            "old_max_arr": cutlass.Array(
                Float32, num_scale_groups, space=cutlass.AddressSpace.rmem
            ),
            "sum_arr": cutlass.Array(
                Float32, num_scale_groups, space=cutlass.AddressSpace.rmem
            ),
            "new_max_arr": cutlass.Array(
                Float32, num_scale_groups, space=cutlass.AddressSpace.rmem
            ),
            "s_arr": cutlass.Array(
                Float32, num_s_regs, space=cutlass.AddressSpace.rmem
            ),
        }
        for idx in cutlass.range_constexpr(num_scale_groups):
            result["old_max_arr"][idx] = _neg_max_f32()
            result["sum_arr"][idx] = Float32(0.0)
            result["new_max_arr"][idx] = _neg_max_f32()
            self._p_local_sum_arr[idx] = Float32(0.0)
            self._global_sum_arr[idx] = Float32(0.0)
        for idx in cutlass.range_constexpr(num_s_regs):
            result["s_arr"][idx] = _neg_max_f32()
        if cutlass.const_expr(not self.cfg.use_fp8_qkv):
            result["packed_p_arr"] = cutlass.Array(
                Int32, self.cfg.num_packed_p_regs, space=cutlass.AddressSpace.rmem
            )
            for idx in cutlass.range_constexpr(self.cfg.num_packed_p_regs):
                result["packed_p_arr"][idx] = Int32(0)
        return result

    @consumer_work(
        work_attrs=WorkAttr.AUXILIARY,
        returns=(old_max_arr, sum_arr, new_max_arr, s_arr),
    )
    @cute.jit
    def init_softmax_state(
        self, stage_info: StageInfo
    ) -> tuple[cutlass.Array, cutlass.Array, cutlass.Array, cutlass.Array]:
        """Initialize and return the softmax task-local state tuple."""
        # ConsAuxWork: seed the running max/sum state and local S buffers before
        # the softmax task consumes any QK score tile.
        result = self._create_initial_task_locals(stage_info.context)
        return (
            result["old_max_arr"],
            result["sum_arr"],
            result["new_max_arr"],
            result["s_arr"],
        )

    @producer_work
    @cute.jit
    def qk_mma_head(
        self,
        stage_info: StageInfo,
        *,
        q_desc: prims.Tcgen05SmemDesc,
        kv_desc: prims.Tcgen05SmemDesc,
        head_dim_stage_idx: Constexpr[int],
    ) -> None:
        """Issue HEAD QK MMA into the initial S slot."""
        # ProdWork: HEAD produces the first score tile, overwriting the initial
        # S stage before any loop softmax work has consumed it.
        self._qk_mma(
            stage_info,
            q_desc=q_desc,
            kv_desc=kv_desc,
            stage_slot_offset=self._qk_head_stage_slot_offset(stage_info),
            head_dim_stage_idx=head_dim_stage_idx,
        )

    @producer_work
    @cute.jit
    def qk_mma_loop(
        self,
        stage_info: StageInfo,
        *,
        q_desc: prims.Tcgen05SmemDesc,
        kv_desc: prims.Tcgen05SmemDesc,
        head_dim_stage_idx: Constexpr[int],
    ) -> None:
        """Issue LOOP QK MMA into the next producer S slot."""
        # ProdWork: LOOP produces the next score tile into the stage that
        # softmax will consume for this steady-state iteration.
        self._qk_mma(
            stage_info,
            q_desc=q_desc,
            kv_desc=kv_desc,
            stage_slot_offset=self._qk_loop_stage_slot_offset(stage_info),
            head_dim_stage_idx=head_dim_stage_idx,
        )

    @producer_work
    @cute.jit
    def qk_mma_head_from_q_ref(
        self,
        stage_info: StageInfo,
        *,
        kv_desc: prims.Tcgen05SmemDesc,
        head_dim_stage_idx: Constexpr[int],
    ) -> None:
        """Issue guarded persistent HEAD QK without a routed Q descriptor."""
        assert self.q_ref is not None
        self._qk_mma(
            stage_info,
            q_desc=self.q_ref.current_consumer_q_desc(),
            kv_desc=kv_desc,
            stage_slot_offset=self._qk_head_stage_slot_offset(stage_info),
            head_dim_stage_idx=head_dim_stage_idx,
        )

    @producer_work
    @cute.jit
    def qk_mma_loop_from_q_ref(
        self,
        stage_info: StageInfo,
        *,
        kv_desc: prims.Tcgen05SmemDesc,
        head_dim_stage_idx: Constexpr[int],
    ) -> None:
        """Issue guarded persistent LOOP QK without a routed Q descriptor."""
        assert self.q_ref is not None
        self._qk_mma(
            stage_info,
            q_desc=self.q_ref.current_consumer_q_desc(),
            kv_desc=kv_desc,
            stage_slot_offset=self._qk_loop_stage_slot_offset(stage_info),
            head_dim_stage_idx=head_dim_stage_idx,
        )

    @cute.jit
    def _qk_mma(
        self,
        stage_info: StageInfo,
        *,
        q_desc: prims.Tcgen05SmemDesc,
        kv_desc: prims.Tcgen05SmemDesc,
        stage_slot_offset: Int32,
        head_dim_stage_idx: Constexpr[int],
    ) -> None:
        """Issue BMM1 with the selected Keeps/Swaps MMA orientation.

        Issues all 16-wide MMA-K slices for the current staged head-dim tile.
        Descriptor stage selection and per-slice jumps are centralized in the
        local descriptor helpers below.
        """
        cfg = self.cfg
        k_desc = _freeze_smem_descriptor(kv_desc)
        q_desc = _freeze_smem_descriptor(q_desc)
        q_desc, head_dim_stage_idx = self._q_desc_for_head_dim_stage(
            q_desc, head_dim_stage_idx
        )

        # TMEM destination: addrspace-6 pointer from base + alloc offset.
        # cutlass's tcgen05_alloc returns the base in tmem_ptr_i32, so add the
        # per-resource column offset before issuing MMA.
        task_cache = _decode_gen_task_cache(stage_info)
        tmem_col = prims.make_tmem_ptr(
            task_cache[_TASK_CACHE_TMEM_BASE_OFFSET]
            + Int32(self._alloc.offset)
            + stage_slot_offset,
            Float32,
        )

        q_is_a, mma_m, mma_n = _qk_mma_operand_contract_for_config(cfg)
        idesc = prims.Tcgen05InstrDesc.build(
            c_dtype=Float32,
            a_dtype=cfg.q_dtype,
            b_dtype=cfg.q_dtype,
            n_dim=mma_n,
            m_dim=mma_m,
        )

        if cutlass.const_expr(cfg.head_dim_per_stage_kv == 0):
            if prims.elect_sync():
                scale_d = False
                for ki in cutlass.range_constexpr(cfg.headdim // _mma_k_step(cfg)):
                    # Keeps computes Q x K^T (A=Q, B=K); Swaps computes the
                    # transposed K x Q^T tile (A=K, B=Q). The first
                    # instruction overwrites S and later slices accumulate.
                    if cutlass.const_expr(q_is_a):
                        a_desc, b_desc = q_desc, k_desc
                    else:
                        a_desc, b_desc = k_desc, q_desc
                    prims.tcgen05_mma(
                        _mma_kind_for_qkv(cfg),
                        prims.CTAGroup.CTA_1,
                        tmem_col,
                        a_desc,
                        b_desc,
                        idesc,
                        scale_d,
                    )
                    scale_d = True
                    if cutlass.const_expr(ki + 1 < cfg.headdim // _mma_k_step(cfg)):
                        k_desc, q_desc = self._advance_qk_descs_after_mma_k(
                            k_desc,
                            q_desc,
                            crosses_64b_chunk=cfg.headdim == 128 and ki == 3,
                        )
        else:
            mma_k_steps = cfg.head_dim_kv_stage // _mma_k_step(cfg)
            if prims.elect_sync():
                # Peel the first MMA so overwrite-vs-accumulate remains a
                # compile-time value rather than loop-carried state.
                if cutlass.const_expr(q_is_a):
                    first_a_desc, first_b_desc = q_desc, k_desc
                else:
                    first_a_desc, first_b_desc = k_desc, q_desc
                prims.tcgen05_mma(
                    _mma_kind_for_qkv(cfg),
                    prims.CTAGroup.CTA_1,
                    tmem_col,
                    first_a_desc,
                    first_b_desc,
                    idesc,
                    cutlass.Boolean(head_dim_stage_idx != 0),
                )

            # Derive every remaining descriptor from the immutable roots. At
            # each 64-column boundary the recurrence replaces its ordinary +2
            # step with +1018 for K and +(8 * TileQ - 6), or +58, for Q; the
            # closed form therefore adds each boundary jump minus that +2.
            # Keeping descriptors out of iter_args avoids staged-D256 spills.
            for ki in cutlass.range(1, mma_k_steps, 1, unroll=1):
                if cutlass.const_expr(cfg.use_fp8_qkv):
                    k_desc_offset = ki * Int32(2)
                    q_desc_offset = ki * Int32(2)
                else:
                    chunk_idx = (ki * Int32(_mma_k_step(cfg))) // Int32(64)
                    k_desc_offset = ki * Int32(2) + chunk_idx * Int32(1016)
                    q_chunk_extra = (
                        8 * cfg.tile_size_q - 8
                        if cutlass.const_expr(cfg.tile_size_q >= 16)
                        else 56
                    )
                    q_desc_offset = ki * Int32(2) + chunk_idx * Int32(q_chunk_extra)
                iter_k_desc = k_desc + k_desc_offset
                iter_q_desc = q_desc + q_desc_offset
                if prims.elect_sync():
                    if cutlass.const_expr(q_is_a):
                        iter_a_desc, iter_b_desc = iter_q_desc, iter_k_desc
                    else:
                        iter_a_desc, iter_b_desc = iter_k_desc, iter_q_desc
                    prims.tcgen05_mma(
                        _mma_kind_for_qkv(cfg),
                        prims.CTAGroup.CTA_1,
                        tmem_col,
                        iter_a_desc,
                        iter_b_desc,
                        idesc,
                        cutlass.Boolean(True),
                    )

    @cute.jit
    def _load_mask_reduce_keeps(
        self,
        stage_info: StageInfo,
        s_vals: cutlass.Array,
        tile_offset_k: Int32,
        element_mask_end_idx: Int32,
        window_start_idx: Int32,
        seq_len_kv: Int32,
        logical_q_group_idx: Int32,
        is_valid_effective_tile: cutlass.Boolean,
        is_masked_final_wave: cutlass.Boolean,
        *,
        apply_boundary_mask: Constexpr[bool],
    ) -> Float32:
        """Load one Keeps fragment and reduce its row maximum.

        The caller chooses the masked/unmasked path before TMEM load. Keeping the
        score fragment out of the branch condition avoids carrying 64/128 live
        S registers through a post-load control-flow edge.
        """
        cfg = self.cfg
        num_s_regs = cfg.num_s_regs_per_thread
        task_cache = _decode_gen_task_cache(stage_info)
        base_addr = (
            task_cache[_TASK_CACHE_TMEM_BASE_OFFSET]
            + Int32(self._alloc.offset)
            + self._softmax_loop_stage_slot_offset(stage_info)
        )
        for load_atom_idx in cutlass.range_constexpr(num_s_regs // 32):
            atom_col = load_atom_idx * 32
            loaded = _keeps_tcgen05_ld(
                cfg,
                prims.make_tmem_ptr(base_addr + Int32(atom_col), Float32),
                num=32,
                offset=cfg.tile_size_kv // 2,
            )
            prims.tcgen05_wait(kind=prims.Tcgen05Wait.LOAD)
            for atom_reg_idx in cutlass.range_constexpr(32):
                s_vals[atom_col + atom_reg_idx] = loaded[atom_reg_idx]

        warp_grp_thread_idx = task_cache[_TASK_CACHE_WARP_GRP_THREAD_IDX]
        lane_idx = task_cache[_TASK_CACHE_LANE_IDX]
        tile_row_idx = _keeps_row_idx(cfg, warp_grp_thread_idx)
        col_base = _keeps_col_base(cfg, lane_idx, num_s_regs)

        if cutlass.const_expr(apply_boundary_mask):
            # Runtime native no-split paging uses the absolute effective tile
            # index. For active rows, an invalid tile therefore begins at or
            # beyond the CTA's causal union; each row's upper mask suppresses
            # the complete tile. Inactive rows are safe only when absent or
            # guaranteed row-independent and discarded at publication.
            per_row_paged_upper_mask_covers_invalid_tile = (
                self.seqlens_kv is not None
                and cfg.use_paged_kv
                and not cfg.use_split_kv
                and cfg.uses_per_row_causal_mask
                and not cfg.use_sliding_window_causal
                and (cfg.q_tiles_are_full or cfg.uses_guarded_grouped_keeps_output_rows)
            )
            if cutlass.const_expr(not per_row_paged_upper_mask_covers_invalid_tile):
                if not (
                    is_valid_effective_tile
                    and tile_offset_k < seq_len_kv
                    and not is_masked_final_wave
                ):
                    for reg_idx in cutlass.range_constexpr(num_s_regs):
                        s_vals[reg_idx] = _neg_max_f32()

            # A per-row causal endpoint is always <= seq_len_kv and is
            # applied by the loop below. Avoid emitting a second,
            # mathematically redundant upper-bound pass for grouped Q.
            if cutlass.const_expr(not cfg.uses_per_row_causal_mask):
                for reg_idx in cutlass.range_constexpr(num_s_regs):
                    token_idx = tile_offset_k + col_base + Int32(reg_idx)
                    if token_idx >= element_mask_end_idx:
                        s_vals[reg_idx] = _neg_max_f32()
                    if cutlass.const_expr(cfg.use_sliding_window_causal):
                        if token_idx < window_start_idx:
                            s_vals[reg_idx] = _neg_max_f32()

            if cutlass.const_expr(cfg.uses_per_row_causal_mask):
                q_token_idx, _ = _q_row_token_and_local_head(
                    cfg,
                    self.h_r,
                    logical_q_group_idx,
                    tile_row_idx,
                )
                causal_end = seq_len_kv - self.seq_len_q + q_token_idx + Int32(1)
                causal_start = Int32(0)
                if cutlass.const_expr(cfg.use_sliding_window_causal):
                    causal_start = cute.math.max(
                        causal_end - Int32(cfg.attention_window_size), Int32(0)
                    )
                causal_start_rel = causal_start - tile_offset_k - col_base
                causal_end_rel = causal_end - tile_offset_k - col_base
                for reg_idx in cutlass.range_constexpr(num_s_regs):
                    if cutlass.const_expr(cfg.use_sliding_window_causal):
                        if (
                            Int32(reg_idx) < causal_start_rel
                            or Int32(reg_idx) >= causal_end_rel
                        ):
                            s_vals[reg_idx] = _neg_max_f32()
                    else:
                        if Int32(reg_idx) >= causal_end_rel:
                            s_vals[reg_idx] = _neg_max_f32()

        if cutlass.const_expr(cfg.q_score_rows_need_mask):
            if not _q_row_is_valid_for_seq(
                cfg,
                self.h_r,
                logical_q_group_idx,
                tile_row_idx,
                self.seq_len_q,
            ):
                for reg_idx in cutlass.range_constexpr(num_s_regs):
                    s_vals[reg_idx] = _neg_max_f32()

        max_chains = cutlass.Array(Float32, 4, space=cutlass.AddressSpace.rmem)
        for chain_idx in cutlass.range_constexpr(4):
            max_chains[chain_idx] = _neg_max_f32()
        for reg_base in cutlass.range_constexpr(0, num_s_regs, 4):
            for chain_idx in cutlass.range_constexpr(4):
                max_chains[chain_idx] = cute.math.max(
                    max_chains[chain_idx],
                    s_vals[reg_base + chain_idx],
                    ftz=True,
                )
        tile_max = cute.math.max(
            cute.math.max(max_chains[0], max_chains[1], ftz=True),
            cute.math.max(max_chains[2], max_chains[3], ftz=True),
            ftz=True,
        )
        if cutlass.const_expr(cfg.tile_size_q == 64):
            return cute.math.max(
                tile_max,
                Float32(
                    prims.shfl_sync(
                        thread_mask=0xFFFFFFFF,
                        val=tile_max,
                        offset=16,
                        mask_and_clamp=0x1F,
                        kind=prims.Shfl.BFLY,
                    )
                ),
                ftz=True,
            )
        return tile_max

    @cute.jit
    def _compute_softmax_loop_keeps(
        self,
        stage_info: StageInfo,
        *,
        old_max_arr: cutlass.Array,
        sum_arr: cutlass.Array,
        new_max_arr: cutlass.Array,
        s_arr: cutlass.Array,
    ) -> tuple[object, object, object, object]:
        """Load and reduce the row-major Keeps S fragment.

        TQ128 assigns one complete Q row to each warp-group thread.  TQ64
        assigns one row to a lane pair: lanes ``xor 16`` own the low/high
        64-column halves.  This path deliberately avoids the Swaps scratch
        reduction, whose 16x256b register mapping is unrelated to Keeps.
        """
        cfg = self.cfg
        num_s_regs = cfg.num_s_regs_per_thread
        old_max = new_max_arr[0]
        running_sum = sum_arr[0]
        s_vals = cutlass.Array(Float32, num_s_regs, space=cutlass.AddressSpace.rmem)
        use_runtime_paged_dense_load = (
            self.seqlens_kv is not None
            and cfg.use_paged_kv
            and not cfg.use_sliding_window_causal
            and cfg.tile_size_q in (64, 128)
        )
        use_preload_mask_split = (
            use_runtime_paged_dense_load or cfg.uses_per_row_causal_mask
        )
        if cutlass.const_expr(not use_preload_mask_split):
            for reg_idx in cutlass.range_constexpr(num_s_regs):
                s_vals[reg_idx] = _neg_max_f32()

        task_cache = _decode_gen_task_cache(stage_info)
        if cutlass.const_expr(self.seqlens_kv is None):
            seq_len_kv = Int32(self.max_seq_len_kv)
        else:
            seq_len_kv = _load_runtime_seq_len_kv(
                self.seqlens_kv,
                self.max_seq_len_kv,
                stage_info,
                Int32(0),
                Int32(0),
            )

        logical_q_group_idx = _logical_q_group_idx(cfg, stage_info, self.q_group_idx)
        q_token_base = _q_group_token_base(cfg, logical_q_group_idx)
        element_mask_end_idx = seq_len_kv
        if cutlass.const_expr(cfg.uses_uniform_causal_mask):
            element_mask_end_idx = seq_len_kv - self.seq_len_q + q_token_base + Int32(1)
        use_runtime_kv_domain = (
            self.seqlens_kv is not None or cfg.uses_runtime_q_kv_union
        )
        local_tile_idx = _softmax_tile_idx(cfg, stage_info, self.inst_id)
        if cutlass.const_expr(not use_runtime_kv_domain):
            effective_tile_idx = _static_split_kv_global_tile_idx(
                cfg, stage_info, local_tile_idx
            )
            effective_total_kv_tiles = Int32(cfg.total_kv_tiles)
            tile_idx = _clamp_valid_tile_idx(cfg, effective_tile_idx)
            tile_idx = tile_idx + Int32(cfg.static_num_skipped_kv_tiles)
            window_start_idx = Int32(cfg.static_window_start_idx)
        elif cutlass.const_expr(cfg.use_paged_kv and not cfg.use_split_kv):
            effective_tile_idx = (
                Int32(task_cache[_TASK_CACHE_KV_RAW_TILE_BASE]) + local_tile_idx
            )
            effective_total_kv_tiles = Int32(task_cache[_TASK_CACHE_KV_VALID_TILE_END])
            tile_idx = effective_tile_idx
            window_start_idx = Int32(task_cache[_TASK_CACHE_KV_WINDOW_START])
        else:
            effective_tile_idx = _runtime_split_kv_global_tile_idx(
                cfg,
                stage_info,
                local_tile_idx,
                seq_len_kv,
                self.seq_len_q,
                q_token_base,
            )
            effective_total_kv_tiles = _runtime_total_kv_tiles(
                cfg, seq_len_kv, self.seq_len_q, q_token_base
            )
            tile_idx = _runtime_clamp_valid_tile_idx(
                cfg,
                effective_tile_idx,
                seq_len_kv,
                self.seq_len_q,
                q_token_base,
            )
            tile_idx = tile_idx + _num_skipped_kv_tiles(
                cfg, seq_len_kv, self.seq_len_q, q_token_base
            )
            window_start_idx = _sliding_window_start_idx(
                cfg, seq_len_kv, self.seq_len_q, q_token_base
            )

        tile_offset_k = tile_idx * Int32(cfg.tile_size_kv)
        is_valid_effective_tile = effective_tile_idx < effective_total_kv_tiles
        is_masked_final_wave = False
        if cutlass.const_expr(not use_runtime_kv_domain and not cfg.use_split_kv):
            if cutlass.const_expr(cfg.has_odd_kv_tail and self.inst_id == 1):
                is_masked_final_wave = _is_last_loop_iteration(stage_info)

        if cutlass.const_expr(use_preload_mask_split):
            # Select the complete unmasked/masked TMEM load+max path before any S
            # registers are materialized. The shared predicate covers the
            # intersection of all active grouped-Q causal/window intervals.
            tile_has_valid_scores = (
                is_valid_effective_tile
                and (tile_offset_k < seq_len_kv)
                and not is_masked_final_wave
            )
            tile_is_unmasked = _kv_tile_is_fully_unmasked_for_q_group(
                cfg,
                tile_offset_k,
                seq_len_kv,
                self.seq_len_q,
                q_token_base,
                tile_has_valid_scores,
            )
            tile_max = _neg_max_f32()
            if tile_is_unmasked:
                tile_max = self._load_mask_reduce_keeps(
                    stage_info,
                    s_vals,
                    tile_offset_k,
                    element_mask_end_idx,
                    window_start_idx,
                    seq_len_kv,
                    logical_q_group_idx,
                    is_valid_effective_tile,
                    is_masked_final_wave,
                    apply_boundary_mask=False,
                )
            else:
                tile_max = self._load_mask_reduce_keeps(
                    stage_info,
                    s_vals,
                    tile_offset_k,
                    element_mask_end_idx,
                    window_start_idx,
                    seq_len_kv,
                    logical_q_group_idx,
                    is_valid_effective_tile,
                    is_masked_final_wave,
                    apply_boundary_mask=True,
                )

            new_max = cute.math.max(old_max, tile_max, ftz=True)
            old_max_arr[0] = old_max
            sum_arr[0] = running_sum
            new_max_arr[0] = new_max
            for reg_idx in cutlass.range_constexpr(num_s_regs):
                s_arr[reg_idx] = s_vals[reg_idx]
            return old_max_arr, sum_arr, new_max_arr, s_arr

        should_load_s = (
            is_valid_effective_tile
            and (tile_offset_k < seq_len_kv)
            and not is_masked_final_wave
        )
        if should_load_s:
            base_addr = (
                task_cache[_TASK_CACHE_TMEM_BASE_OFFSET]
                + Int32(self._alloc.offset)
                + self._softmax_loop_stage_slot_offset(stage_info)
            )
            # Keep each intrinsic result at the native 32-register atom. TQ128
            # uses four consecutive atoms and TQ64 uses two atoms with the same
            # half-split offset, avoiding a monolithic x64/x128 LLVM intrinsic
            # result.
            load_atom_regs = 32
            for load_atom_idx in cutlass.range_constexpr(num_s_regs // load_atom_regs):
                atom_col = load_atom_idx * load_atom_regs
                loaded = _keeps_tcgen05_ld(
                    cfg,
                    prims.make_tmem_ptr(base_addr + Int32(atom_col), Float32),
                    num=load_atom_regs,
                    offset=cfg.tile_size_kv // 2,
                )
                prims.tcgen05_wait(kind=prims.Tcgen05Wait.LOAD)
                for atom_reg_idx in cutlass.range_constexpr(load_atom_regs):
                    s_vals[atom_col + atom_reg_idx] = loaded[atom_reg_idx]

        warp_grp_thread_idx = task_cache[_TASK_CACHE_WARP_GRP_THREAD_IDX]
        lane_idx = task_cache[_TASK_CACHE_LANE_IDX]
        tile_row_idx = _keeps_row_idx(cfg, warp_grp_thread_idx)
        col_base = _keeps_col_base(cfg, lane_idx, num_s_regs)

        # Tail/uniform-causal/window masks use each register's true logical K
        # column. For q64, the paired lane owns the complementary 64-column half.
        for reg_idx in cutlass.range_constexpr(num_s_regs):
            token_idx = tile_offset_k + col_base + Int32(reg_idx)
            # The per-row causal pass below subsumes seq_len_kv's upper bound.
            if cutlass.const_expr(not cfg.uses_per_row_causal_mask):
                if token_idx >= element_mask_end_idx:
                    s_vals[reg_idx] = _neg_max_f32()
            if cutlass.const_expr(cfg.use_sliding_window_causal):
                if token_idx < window_start_idx:
                    s_vals[reg_idx] = _neg_max_f32()

        if cutlass.const_expr(cfg.uses_per_row_causal_mask):
            q_token_idx, _ = _q_row_token_and_local_head(
                cfg,
                self.h_r,
                logical_q_group_idx,
                tile_row_idx,
            )
            causal_end = seq_len_kv - self.seq_len_q + q_token_idx + Int32(1)
            causal_start = Int32(0)
            if cutlass.const_expr(cfg.use_sliding_window_causal):
                causal_start = cute.math.max(
                    causal_end - Int32(cfg.attention_window_size), Int32(0)
                )
            for reg_idx in cutlass.range_constexpr(num_s_regs):
                token_idx = tile_offset_k + col_base + Int32(reg_idx)
                if token_idx < causal_start or token_idx >= causal_end:
                    s_vals[reg_idx] = _neg_max_f32()

        if cutlass.const_expr(cfg.q_score_rows_need_mask):
            if not _q_row_is_valid_for_seq(
                cfg,
                self.h_r,
                logical_q_group_idx,
                tile_row_idx,
                self.seq_len_q,
            ):
                for reg_idx in cutlass.range_constexpr(num_s_regs):
                    s_vals[reg_idx] = _neg_max_f32()

        # Four independent max chains avoid one long dependency chain across
        # 64/128 score registers.
        max_chains = cutlass.Array(Float32, 4, space=cutlass.AddressSpace.rmem)
        for chain_idx in cutlass.range_constexpr(4):
            max_chains[chain_idx] = _neg_max_f32()
        for reg_base in cutlass.range_constexpr(0, num_s_regs, 4):
            for chain_idx in cutlass.range_constexpr(4):
                max_chains[chain_idx] = cute.math.max(
                    max_chains[chain_idx],
                    s_vals[reg_base + chain_idx],
                    ftz=True,
                )
        tile_max = cute.math.max(
            cute.math.max(max_chains[0], max_chains[1], ftz=True),
            cute.math.max(max_chains[2], max_chains[3], ftz=True),
            ftz=True,
        )
        if cutlass.const_expr(cfg.tile_size_q == 64):
            tile_max = cute.math.max(
                tile_max,
                Float32(
                    prims.shfl_sync(
                        thread_mask=0xFFFFFFFF,
                        val=tile_max,
                        offset=16,
                        mask_and_clamp=0x1F,
                        kind=prims.Shfl.BFLY,
                    )
                ),
                ftz=True,
            )
        new_max = cute.math.max(old_max, tile_max, ftz=True)

        old_max_arr[0] = old_max
        sum_arr[0] = running_sum
        new_max_arr[0] = new_max
        for reg_idx in cutlass.range_constexpr(num_s_regs):
            s_arr[reg_idx] = s_vals[reg_idx]
        return old_max_arr, sum_arr, new_max_arr, s_arr

    @consumer_work(returns=(old_max_arr, sum_arr, new_max_arr, s_arr))
    @cute.jit
    def compute_softmax_loop(
        self,
        stage_info: StageInfo,
        *,
        old_max_arr: cutlass.Array,
        sum_arr: cutlass.Array,
        new_max_arr: cutlass.Array,
        s_arr: cutlass.Array,
    ) -> tuple[object, object, object, object]:
        """Load S from TMEM and materialize the running softmax state.

        Operation order: load BMM1 scores, apply tail/window masks, reduce the
        row max through shared scratch, and return the old/new max payload that
        correction consumes.
        """
        cfg = self.cfg
        if cutlass.const_expr(cfg.use_keeps_mma_ab):
            return self._compute_softmax_loop_keeps(
                stage_info,
                old_max_arr=old_max_arr,
                sum_arr=sum_arr,
                new_max_arr=new_max_arr,
                s_arr=s_arr,
            )
        # ConsWork: consume the committed S tile, update the running max state,
        # and forward masked S registers to the P producer.
        # Start from the previously published running max/sum and a fresh
        # local S buffer for this tile.
        num_scale_groups = cfg.num_softmax_scale_groups
        q_repeats = max(cfg.tile_size_q // 8, 1)
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
            Float32, cfg.num_s_regs_per_thread, space=cutlass.AddressSpace.rmem
        )
        use_runtime_paged_dense_load = (
            self.seqlens_kv is not None
            and cfg.use_paged_kv
            and not cfg.use_split_kv
            and cfg.max_seq_len_q == 1
            and not cfg.use_sliding_window_causal
        )
        for idx in cutlass.range_constexpr(num_scale_groups):
            old_max_vals[idx] = new_max_arr[idx]
            sum_vals[idx] = sum_arr[idx]
            new_max_vals[idx] = new_max_arr[idx]
            local_max_vals[idx] = _neg_max_f32()
        if cutlass.const_expr(not use_runtime_paged_dense_load):
            for idx in cutlass.range_constexpr(cfg.num_s_regs_per_thread):
                s_vals[idx] = _neg_max_f32()
        task_cache = _decode_gen_task_cache(stage_info)
        if cutlass.const_expr(self.seqlens_kv is None):
            seq_len_kv = Int32(self.max_seq_len_kv)
        else:
            # Variable-seqlen kernels read the active sequence length for
            # the logical batch carried by the work tile.
            seq_len_kv = _load_runtime_seq_len_kv(
                self.seqlens_kv,
                self.max_seq_len_kv,
                stage_info,
                Int32(0),
                Int32(0),
            )
        logical_q_group_idx = _logical_q_group_idx(cfg, stage_info, self.q_group_idx)
        q_token_base = _q_group_token_base(cfg, logical_q_group_idx)
        element_mask_end_idx = seq_len_kv
        if cutlass.const_expr(cfg.uses_uniform_causal_mask):
            element_mask_end_idx = seq_len_kv - self.seq_len_q + q_token_base + Int32(1)
        use_runtime_kv_domain = (
            self.seqlens_kv is not None or cfg.uses_runtime_q_kv_union
        )
        local_tile_idx = _softmax_tile_idx(cfg, stage_info, self.inst_id)
        if cutlass.const_expr(not use_runtime_kv_domain):
            # Static path: compute the effective global tile and any
            # sliding-window prefix skip at compile time.
            effective_tile_idx = _static_split_kv_global_tile_idx(
                cfg, stage_info, local_tile_idx
            )
            effective_total_kv_tiles = Int32(cfg.total_kv_tiles)
            tile_idx = _clamp_valid_tile_idx(cfg, effective_tile_idx)
            tile_idx = tile_idx + Int32(cfg.static_num_skipped_kv_tiles)
            window_start_idx = Int32(cfg.static_window_start_idx)
        elif cutlass.const_expr(cfg.use_paged_kv and not cfg.use_split_kv):
            # Non-split native paging can consume the task's affine raw tile
            # geometry directly. Split-KV retains its existing resolver because
            # that path benchmarks faster with the general softmax mapping.
            effective_tile_idx = (
                Int32(task_cache[_TASK_CACHE_KV_RAW_TILE_BASE]) + local_tile_idx
            )
            effective_total_kv_tiles = Int32(task_cache[_TASK_CACHE_KV_VALID_TILE_END])
            tile_idx = effective_tile_idx
            window_start_idx = Int32(task_cache[_TASK_CACHE_KV_WINDOW_START])
        else:
            # Runtime path: compute the same values from the batch-specific
            # sequence length.
            effective_tile_idx = _runtime_split_kv_global_tile_idx(
                cfg,
                stage_info,
                local_tile_idx,
                seq_len_kv,
                self.seq_len_q,
                q_token_base,
            )
            effective_total_kv_tiles = _runtime_total_kv_tiles(
                cfg, seq_len_kv, self.seq_len_q, q_token_base
            )
            tile_idx = _runtime_clamp_valid_tile_idx(
                cfg,
                effective_tile_idx,
                seq_len_kv,
                self.seq_len_q,
                q_token_base,
            )
            tile_idx = tile_idx + _num_skipped_kv_tiles(
                cfg, seq_len_kv, self.seq_len_q, q_token_base
            )
            window_start_idx = _sliding_window_start_idx(
                cfg, seq_len_kv, self.seq_len_q, q_token_base
            )
        tile_offset_k = tile_idx * Int32(cfg.tile_size_kv)
        is_valid_effective_tile = effective_tile_idx < effective_total_kv_tiles
        is_masked_final_wave = False
        if cutlass.const_expr(not use_runtime_kv_domain and not cfg.use_split_kv):
            if cutlass.const_expr(cfg.has_odd_kv_tail and self.inst_id == 1):
                # The second instance in an odd tail is a prefetch duplicate
                # and must not contribute to softmax.
                is_masked_final_wave = _is_last_loop_iteration(stage_info)

        should_load_s = True
        if cutlass.const_expr(not use_runtime_paged_dense_load):
            should_load_s = (
                is_valid_effective_tile
                and (tile_offset_k < seq_len_kv)
                and not is_masked_final_wave
            )
        if should_load_s:
            # ConsWork: load the S tile produced by BMM1 from TMEM into
            # registers. Two TMEM rows cover the two K subtiles. Invalid
            # odd-tail waves intentionally skip the load and leave S at
            # -inf so the later P path contributes zero probability.
            base_addr = (
                task_cache[_TASK_CACHE_TMEM_BASE_OFFSET]
                + Int32(self._alloc.offset)
                + self._softmax_loop_stage_slot_offset(stage_info)
            )

            shape = "16x256b"
            loaded0 = prims.tcgen05_ld(
                shape,
                prims.make_tmem_ptr(base_addr, Float32),
                num=q_repeats,
            )
            loaded1 = prims.tcgen05_ld(
                shape,
                prims.make_tmem_ptr(base_addr + Int32(16 << 16), Float32),
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

        if cutlass.const_expr(use_runtime_paged_dense_load):
            if not (
                is_valid_effective_tile
                and (tile_offset_k < seq_len_kv)
                and not is_masked_final_wave
            ):
                for idx in cutlass.range_constexpr(cfg.num_s_regs_per_thread):
                    s_vals[idx] = _neg_max_f32()

        next_tile_offset_k = tile_offset_k + Int32(cfg.tile_size_kv)
        # Determine whether this tile crosses the active right endpoint or the
        # start of the causal sliding window. Dense full tiles skip per-element
        # masking.
        if cutlass.const_expr(
            not use_runtime_kv_domain and not cfg.uses_uniform_causal_mask
        ):
            has_static_tail_mask = (cfg.static_seq_len_kv % cfg.tile_size_kv) != 0
            has_static_window_prefix_mask = (
                cfg.use_sliding_window_causal
                and (cfg.static_window_start_idx % cfg.tile_size_kv) != 0
            )
            if cutlass.const_expr(
                not has_static_tail_mask and not has_static_window_prefix_mask
            ):
                should_apply_dense_mask = False
            else:
                should_apply_dense_mask = next_tile_offset_k > seq_len_kv
            if cutlass.const_expr(has_static_window_prefix_mask):
                should_apply_dense_mask = should_apply_dense_mask or (
                    (tile_offset_k <= window_start_idx)
                    and (next_tile_offset_k > window_start_idx)
                )
        else:
            # Runtime tails and the one-endpoint ungrouped causal path need
            # element masking only on the tile that crosses the right bound.
            should_apply_dense_mask = next_tile_offset_k > element_mask_end_idx
            if cutlass.const_expr(cfg.use_sliding_window_causal):
                window_start_remainder = window_start_idx % Int32(cfg.tile_size_kv)
                should_apply_dense_mask = should_apply_dense_mask or (
                    (window_start_remainder != Int32(0))
                    and (tile_offset_k <= window_start_idx)
                    and (next_tile_offset_k > window_start_idx)
                )
        if should_apply_dense_mask:
            # Mask invalid S registers to -inf so they produce zero P and
            # do not affect row max or row sum. This keeps the schedule
            # shape fixed even when only part of the K/V tile is valid.
            warp_idx = task_cache[_TASK_CACHE_WARP_IDX]
            lane_idx = task_cache[_TASK_CACHE_LANE_IDX]
            local_idx_k0 = warp_idx * Int32(32) + (lane_idx >> Int32(2))
            for repeat_idx in cutlass.range_constexpr(q_repeats):
                for token_group_idx in cutlass.range_constexpr(4):
                    token_idx = (
                        tile_offset_k + local_idx_k0 + Int32(token_group_idx * 8)
                    )
                    if cutlass.const_expr(token_group_idx < 2):
                        s_base = repeat_idx * 4 + token_group_idx * 2
                    else:
                        s_base = (
                            q_repeats * 4 + repeat_idx * 4 + (token_group_idx - 2) * 2
                        )
                    if token_idx >= element_mask_end_idx:
                        s_vals[s_base + 0] = _neg_max_f32()
                        s_vals[s_base + 1] = _neg_max_f32()
                    if cutlass.const_expr(cfg.use_sliding_window_causal):
                        if token_idx < window_start_idx:
                            s_vals[s_base + 0] = _neg_max_f32()
                            s_vals[s_base + 1] = _neg_max_f32()

        if cutlass.const_expr(cfg.uses_per_row_causal_mask):
            tile_has_valid_scores = (
                is_valid_effective_tile
                and (tile_offset_k < seq_len_kv)
                and not is_masked_final_wave
            )
            if not _kv_tile_is_fully_unmasked_for_q_group(
                cfg,
                tile_offset_k,
                seq_len_kv,
                self.seq_len_q,
                q_token_base,
                tile_has_valid_scores,
            ):
                # Grouped causal decode has a distinct causal/window bound for
                # every Q token. Only boundary tiles need this row-exact pass.
                warp_idx = task_cache[_TASK_CACHE_WARP_IDX]
                lane_idx = task_cache[_TASK_CACHE_LANE_IDX]
                col_group_idx = lane_idx & Int32(0x3)
                local_idx_k0 = warp_idx * Int32(32) + (lane_idx >> Int32(2))
                for scale_idx in cutlass.range_constexpr(num_scale_groups):
                    repeat_idx = scale_idx // 2
                    pair_idx = scale_idx % 2
                    tile_row_idx = (
                        Int32(repeat_idx * 8)
                        + col_group_idx * Int32(2)
                        + Int32(pair_idx)
                    )
                    q_token_idx, _ = _q_row_token_and_local_head(
                        cfg,
                        self.h_r,
                        logical_q_group_idx,
                        tile_row_idx,
                    )
                    causal_end = seq_len_kv - self.seq_len_q + q_token_idx + Int32(1)
                    causal_start = Int32(0)
                    if cutlass.const_expr(cfg.use_sliding_window_causal):
                        causal_start = cute.math.max(
                            causal_end - Int32(cfg.attention_window_size), Int32(0)
                        )
                    for token_group_idx in cutlass.range_constexpr(4):
                        token_idx = (
                            tile_offset_k + local_idx_k0 + Int32(token_group_idx * 8)
                        )
                        s_idx = (
                            repeat_idx * 4
                            + pair_idx
                            + (token_group_idx & 1) * 2
                            + (token_group_idx >> 1) * q_repeats * 4
                        )
                        if token_idx < causal_start or token_idx >= causal_end:
                            s_vals[s_idx] = _neg_max_f32()

        warp_grp_thread_idx = task_cache[_TASK_CACHE_WARP_GRP_THREAD_IDX]
        if cutlass.const_expr(cfg.q_score_rows_need_mask):
            # Structural grouped padding and the final partial token/head band
            # must not enter max/sum or P. The Swaps TMEM layout assigns one
            # logical Q row to each (column-group, scale-group) pair.
            lane_idx = task_cache[_TASK_CACHE_LANE_IDX]
            col_group_idx = lane_idx & Int32(0x3)
            for scale_idx in cutlass.range_constexpr(num_scale_groups):
                repeat_idx = scale_idx // 2
                pair_idx = scale_idx % 2
                tile_row_idx = (
                    col_group_idx * Int32(2) + Int32(pair_idx) + Int32(repeat_idx * 8)
                )
                if not _q_row_is_valid_for_seq(
                    cfg,
                    self.h_r,
                    logical_q_group_idx,
                    tile_row_idx,
                    self.seq_len_q,
                ):
                    s_base = repeat_idx * 4 + pair_idx
                    s_base_hi = q_repeats * 4 + s_base
                    s_vals[s_base + 0] = _neg_max_f32()
                    s_vals[s_base + 2] = _neg_max_f32()
                    s_vals[s_base_hi + 0] = _neg_max_f32()
                    s_vals[s_base_hi + 2] = _neg_max_f32()

        lane_idx = task_cache[_TASK_CACHE_LANE_IDX]
        for scale_idx in cutlass.range_constexpr(num_scale_groups):
            # Reduce this lane's S registers to one candidate per scale group.
            repeat_idx = scale_idx // 2
            pair_idx = scale_idx % 2
            s_base = repeat_idx * 4 + pair_idx
            s_base_hi = q_repeats * 4 + s_base
            local_max = cute.math.max(
                cute.math.max(s_vals[s_base + 0], s_vals[s_base + 2], ftz=True),
                cute.math.max(s_vals[s_base_hi + 0], s_vals[s_base_hi + 2], ftz=True),
                ftz=True,
            )
            local_max = cute.math.max(local_max, old_max_vals[scale_idx], ftz=True)
            local_max_vals[scale_idx] = local_max

        local_row_idx = (lane_idx >> Int32(2)) & Int32(0x3)
        if cutlass.const_expr(num_scale_groups > 2):
            # Transpose each four-scale block across four strided warp rows.
            # Every lane then owns one reduced scale group per block.
            for scale_base in cutlass.range_constexpr(0, num_scale_groups, 4):
                local_max_vals[scale_base] = _wspro_reduce_max4(
                    local_max_vals[scale_base],
                    local_max_vals[scale_base + 1],
                    local_max_vals[scale_base + 2],
                    local_max_vals[scale_base + 3],
                    local_row_idx,
                )
        else:
            # Two scale groups use the compact partial reduction.
            for scale_idx in cutlass.range_constexpr(num_scale_groups):
                local_max = cute.math.max(
                    local_max_vals[scale_idx],
                    Float32(
                        prims.shfl_sync(
                            thread_mask=0xFFFFFFFF,
                            val=local_max_vals[scale_idx],
                            offset=16,
                            mask_and_clamp=0x1F,
                            kind=prims.Shfl.BFLY,
                        )
                    ),
                    ftz=True,
                )
                local_max_vals[scale_idx] = cute.math.max(
                    local_max,
                    Float32(
                        prims.shfl_sync(
                            thread_mask=0xFFFFFFFF,
                            val=local_max,
                            offset=8,
                            mask_and_clamp=0x1F,
                            kind=prims.Shfl.BFLY,
                        )
                    ),
                    ftz=True,
                )

        if stage_info.loop_offset == stage_info.loop_start:
            # Softmax consumes S in the loop stage. Persistent CTAs reuse this
            # scratch across work tiles, so reset it at the first loop iteration
            # before the running max atomics. The barrier prevents a lane from
            # atomically updating a slot another lane is still reinitializing.
            _init_softmax_scratch_u32(
                self._softmax_scratch_u32, warp_grp_thread_idx, cfg.tile_size_q
            )
            prims.barrier_cta_sync(self.sync_barrier_id, thread_count=128)

        col_group_idx = lane_idx & Int32(0x3)
        atomic_reduce_base = col_group_idx * Int32(num_scale_groups)
        if cutlass.const_expr(num_scale_groups > 2):
            # Two row groups publish one partial per distributed scale group.
            for scale_base in cutlass.range_constexpr(0, num_scale_groups, 4):
                scale_idx = local_row_idx + Int32(scale_base)
                _smem_atomic_max_u32(
                    self._softmax_scratch_u32.data_ptr()
                    + atomic_reduce_base
                    + scale_idx,
                    _float_to_u32_for_atomic_max(local_max_vals[scale_base]),
                )
        elif lane_idx < Int32(8):
            # The compact fallback publishes both scale groups from eight lanes.
            for scale_idx in cutlass.range_constexpr(num_scale_groups):
                _smem_atomic_max_u32(
                    self._softmax_scratch_u32.data_ptr()
                    + atomic_reduce_base
                    + Int32(scale_idx),
                    _float_to_u32_for_atomic_max(local_max_vals[scale_idx]),
                )
        # Wait for every SMEM atomic max to finish before reloading the
        # reduced maxima. Without this barrier, the vector reload below can
        # race a late writer from another softmax warp.
        prims.barrier_cta_sync(self.sync_barrier_id, thread_count=128)

        reduce_base = col_group_idx * Int32(num_scale_groups)
        reduced_max_ptr = self._softmax_scratch_u32.data_ptr() + reduce_base
        # Reload the reduced max as one aligned vector, then decode back to
        # float. Keeping this reload vectorized avoids the scalar LDS shape.
        reduced_max = reduced_max_ptr.load(
            count=num_scale_groups,
            alignment=16 if num_scale_groups >= 4 else 8,
        )
        for scale_idx in cutlass.range_constexpr(num_scale_groups):
            # Decode the CTA-wide maxima back into the running softmax
            # state carried by this resource.
            new_max_vals[scale_idx] = _u32_to_float_for_atomic_max(
                reduced_max[scale_idx]
            )

        for scale_idx in cutlass.range_constexpr(num_scale_groups):
            old_max_arr[scale_idx] = old_max_vals[scale_idx]
            sum_arr[scale_idx] = sum_vals[scale_idx]
            new_max_arr[scale_idx] = new_max_vals[scale_idx]
        for idx in cutlass.range_constexpr(cfg.num_s_regs_per_thread):
            # Forward the loaded/masked S registers to SmemP.compute_p.
            s_arr[idx] = s_vals[idx]
        return old_max_arr, sum_arr, new_max_arr, s_arr

    @consumer_work(
        returns=sum_arr,
        work_attrs=WorkAttr.AUXILIARY,
    )
    @cute.jit
    def reduce_sums(
        self,
        stage_info: StageInfo,
        *,
        old_max_arr: cutlass.Array,
        new_max_arr: cutlass.Array,
        sum_arr: cutlass.Array,
    ) -> ResourceVars:
        """Fold local P sums into the running online-softmax denominators."""
        cfg = self.cfg
        # ConsTailWork: denominator update runs after P has been materialized,
        # so the resource-owned local sum matches the P payload consumed by BMM2.
        if cutlass.const_expr(cfg.use_fp8_qkv):
            # FP8 uses TmemSoftmaxGlobal to update sums after P
            # quantization, so this stage only copies the corrected sums
            # back into the running state. This keeps the denominator
            # consistent with the quantized P actually consumed by BMM2.
            for scale_idx in cutlass.range_constexpr(cfg.num_softmax_scale_groups):
                sum_arr[scale_idx] = self.load_global_sum(scale_idx)
            return sum_arr
        num_scale_groups = cfg.num_softmax_scale_groups
        for scale_base in cutlass.range_constexpr(0, num_scale_groups, 2):
            # Running sum recurrence:
            # sum_new = sum_old * exp(old_max - new_max) + local_sum.
            # local_sum comes from SmemP, after P has been produced, so the
            # denominator update stays ordered after P materialization.
            # Gather one pair of scale groups so the rescale and sum update can
            # use paired arithmetic and publish both groups together.
            old_max = cutlass.Array(Float32, 2, space=cutlass.AddressSpace.rmem)
            new_max = cutlass.Array(Float32, 2, space=cutlass.AddressSpace.rmem)
            local_sum = cutlass.Array(Float32, 2, space=cutlass.AddressSpace.rmem)
            sum_vals = cutlass.Array(Float32, 2, space=cutlass.AddressSpace.rmem)
            exp_scale = cutlass.Array(Float32, 2, space=cutlass.AddressSpace.rmem)
            pair_width = _softmax_scale_pair_width(num_scale_groups, scale_base)
            # KeepsMmaAb has one scale group. Initialize the unused packed-FMA
            # lane explicitly, then load and publish only the live lane so no
            # out-of-bounds task-local access can become LLVM poison.
            for pair_idx in cutlass.range_constexpr(2):
                old_max[pair_idx] = _neg_max_f32()
                new_max[pair_idx] = _neg_max_f32()
                local_sum[pair_idx] = Float32(0.0)
                sum_vals[pair_idx] = Float32(0.0)
                exp_scale[pair_idx] = Float32(0.0)
            for pair_idx in cutlass.range_constexpr(pair_width):
                scale_idx = scale_base + pair_idx
                old_max[pair_idx] = old_max_arr[scale_idx]
                new_max[pair_idx] = new_max_arr[scale_idx]
                local_sum[pair_idx] = self.load_p_local_sum(scale_idx)
                sum_vals[pair_idx] = sum_arr[scale_idx]

            # Dense full-tile FP16/BF16 paths never see -inf max sentinels, so
            # they can compute exp(old-new) directly. General paths guard the
            # sentinel to keep empty/masked groups at zero contribution.
            if cutlass.const_expr(
                cfg.has_static_dense_full_kv_tiles
                and cfg.tile_size_q in (16, 32)
                and not cfg.use_keeps_mma_ab
                and not cfg.use_fp8_qkv
                and cfg.q_tiles_are_full
            ):
                for pair_idx in cutlass.range_constexpr(pair_width):
                    exp_scale[pair_idx] = cute.math.exp2(
                        self.scale_softmax_log2
                        * (old_max[pair_idx] - new_max[pair_idx]),
                        fastmath=True,
                    )
            else:
                for pair_idx in cutlass.range_constexpr(pair_width):
                    if (old_max[pair_idx] != _neg_max_f32()) and (
                        new_max[pair_idx] != _neg_max_f32()
                    ):
                        exp_scale[pair_idx] = cute.math.exp2(
                            self.scale_softmax_log2
                            * (old_max[pair_idx] - new_max[pair_idx]),
                            fastmath=True,
                        )
            updated_sums = ffma2(
                (exp_scale[0], exp_scale[1]),
                (sum_vals[0], sum_vals[1]),
                (local_sum[0], local_sum[1]),
            )
            # Publish the updated running denominator for the next softmax tile
            # and for tail correction normalization.
            for pair_idx in cutlass.range_constexpr(pair_width):
                sum_arr[scale_base + pair_idx] = updated_sums[pair_idx]
        return sum_arr
