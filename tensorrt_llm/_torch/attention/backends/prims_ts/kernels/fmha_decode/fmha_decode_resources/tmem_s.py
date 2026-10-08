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
from cutlass import BFloat16, Boolean, Float32, Int32, Uint32
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

from ...._block_sparse.prepared import _PREPARED_ROUTE_IS_FULL_FLAG
from ..fmha_decode_config import CAUSAL, FmhaDecodeConfig
from ..fmha_decode_constants import (
    INT32_SCORE_BIAS,
    INT32_SCORE_SEED_MMA_K,
    INT32_SCORE_SEED_TILE_LBO,
    INT32_SCORE_SEED_TILE_SBO,
    INT32_SCORE_SEED_TILE_WORDS,
    SOFTMAX_RESCALE_THRESHOLD_LOG2,
)
from ...tcgen05_compat import tcgen05_mma_ws
from ...placeholder_helpers import (
    _placeholder_local_array,
    _placeholder_smem_array,
)
from .helpers_common import (
    _TASK_CACHE_LANE_IDX,
    _TASK_CACHE_KV_RAW_TILE_BASE,
    _TASK_CACHE_KV_VALID_TILE_END,
    _TASK_CACHE_KV_WINDOW_START,
    _TASK_CACHE_TMEM_BASE_OFFSET,
    _TASK_CACHE_WARP_GRP_THREAD_IDX,
    _TASK_CACHE_WARP_IDX,
    Constexpr,
    DecodeGenResourceBase,
    DescriptorValue,
    ResourceVars,
    _clamp_valid_tile_idx,
    _decode_gen_task_cache,
    _freeze_smem_descriptor,
    _is_last_loop_iteration,
    _keeps_col_base,
    _keeps_row_idx,
    _keeps_score_col,
    _keeps_spatial_half,
    _keeps_tcgen05_ld,
    _keeps_tcgen05_st,
    _logical_head_batch,
    _logical_q_group_idx,
    _mma_k_step_qk,
    _mma_kind_for_qk,
    _neg_max_f32,
    _qk_accumulator_dtype,
    _softmax_scale_pair_width,
    _swaps_routed_coordinate,
    _q_row_is_valid_for_seq,
    _q_row_token_and_local_head,
    _q_group_token_base,
    _route_is_proxy,
    _softmax_tile_idx,
    ffma2,
    fmul2,
)
from .smem_block_sparse_metadata import (
    _swaps_forwards_packed_route_full,
)
from .sage_scales import SageKScalesResource, SageScaleTensors, load_q_scale
from .helpers_kv_tile_idx import (
    resolve_keeps_tile_context,
    _kv_tile_is_fully_unmasked_for_q_group,
    _load_runtime_seq_len_kv,
    _num_skipped_kv_tiles,
    _runtime_clamp_valid_tile_idx,
    _runtime_split_kv_global_tile_idx,
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


def _swaps_uses_origin0_k32_full_guard(cfg: FmhaDecodeConfig) -> bool:
    """Whether one staged origin can prove this warp's K32 slice valid."""

    return (
        cfg.kv_block_size >= 32
        and not cfg.uses_prepared_score_keep_words
        and not cfg.uses_uniform_causal_mask
        and not cfg.uses_per_row_causal_mask
    )


def _swaps_token_word_covers_kv_tail(cfg: FmhaDecodeConfig) -> bool:
    """Whether SWAP's prepared token word covers the logical KV tail."""

    return (
        cfg.uses_prepared_score_keep_words
        and not cfg.uses_uniform_causal_mask
        and not cfg.uses_per_row_causal_mask
    )


def _swaps_uses_token_only_score_validity(cfg: FmhaDecodeConfig) -> bool:
    """Whether prepared token words replace SWAP's atom-origin guard."""

    return (
        cfg.use_block_sparse
        and _swaps_token_word_covers_kv_tail(cfg)
        and cfg.tile_size_q < 64
        and cfg.use_persistent_scheduler
        and (cfg.kv_block_size >= 16 or cfg.use_parallel_sparse_kv_loads)
    )


@cute.jit
def _dense_fragment_keep_word(
    rows_are_active: Boolean,
    visible_start: Int32,
    visible_end: Int32,
    *,
    fragment_regs: cutlass.Constexpr[int],
) -> Uint32:
    """Return the keep word of one dense K32 fragment.

    ``visible_start`` and ``visible_end`` are the visible token range relative
    to the fragment's first column. Columns outside ``[start, end)`` are
    masked; an inactive tile or Q row masks the whole fragment.
    """
    keep_word = Uint32(0)
    if rows_are_active:
        first_kept = cute.math.max(visible_start, Int32(0))
        end_kept = cute.math.min(visible_end, Int32(fragment_regs))
        if first_kept < end_kept:
            # Both shift amounts stay strictly below the register width:
            # 1 <= end_kept <= fragment_regs and 0 <= first_kept < end_kept.
            keep_word = (Uint32(0xFFFFFFFF) >> (Int32(fragment_regs) - end_kept)) & (
                Uint32(0xFFFFFFFF) << first_kept
            )
    return keep_word


@cute.jit
def _sparse_effective_keep_word(
    q_row_is_valid: Boolean,
    fragment_origin: Int32,
    fragment_valid: Int32,
    token_word: Uint32,
    seq_len_kv: Int32,
    causal_end: Int32,
    *,
    apply_causal_mask: cutlass.Constexpr[bool],
    apply_token_mask: cutlass.Constexpr[bool],
) -> Uint32:
    """Fold route, KV-tail, causal, and token predicates for one K32 fragment."""

    keep_word = Uint32(0)
    if q_row_is_valid and fragment_valid != Int32(0):
        visible_end = seq_len_kv
        if cutlass.const_expr(apply_causal_mask):
            visible_end = cute.math.min(visible_end, causal_end)
        visible_tokens = visible_end - fragment_origin
        if visible_tokens >= Int32(32):
            keep_word = Uint32(0xFFFFFFFF)
            if cutlass.const_expr(apply_token_mask):
                keep_word = token_word
        elif visible_tokens > Int32(0):
            # Keep the shift strictly below 32; shifting a 32-bit value by its
            # width is undefined in PTX and LLVM.
            keep_word = (Uint32(1) << visible_tokens) - Uint32(1)
            if cutlass.const_expr(apply_token_mask):
                keep_word = keep_word & token_word
    return keep_word


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

    _rts_internal_consumer_var_names: ClassVar[tuple[str, ...]] = (
        "old_max_arr",
        "sum_arr",
        "new_max_arr",
        "s_arr",
    )
    _task_local_specs: ClassVar[tuple[tuple, ...]] = (
        (
            "old_max_arr",
            cutlass.Array,
            None,
            "Previous softmax anchor (normally the running row maximum).",
        ),
        ("sum_arr", cutlass.Array, None, "Running softmax denominator."),
        (
            "new_max_arr",
            cutlass.Array,
            None,
            "Current softmax anchor (normally the running row maximum).",
        ),
        ("s_arr", cutlass.Array, None, "Loaded S scores for the current tile."),
        (
            "sage_q_scale",
            Float32,
            Float32(1.0),
            "Sage Q scale of the lane's row, loaded once per work tile.",
        ),
    )
    inst_id: Constexpr[int] = 0
    cfg: Constexpr[FmhaDecodeConfig] = None
    scale_softmax_log2: Float32 = None
    seqlens_kv: cute.Pointer | None = None
    max_seq_len_kv: Int32 = None
    seq_len_q: Int32 = None
    h_r: Int32 | None = None
    h_k_idx: Int32 | None = None
    b_idx: Int32 | None = None
    q_group_idx: Int32 | None = None
    scale_tensors: SageScaleTensors | None = None
    q_ref: Constexpr[MemoryResource | None] = None
    page_offsets_ref: Constexpr[MemoryResource | None] = None
    _p_local_sum_arr: cutlass.Array | None = None
    _global_sum_arr: cutlass.Array | None = None
    _alloc: Constexpr[TmemAllocation | None] = None
    sync_barrier_id: Constexpr[int] = 0
    _scratch_alloc: Constexpr[SmemAllocation | None] = None
    _softmax_scratch_u32: cutlass.Array = None
    # Constant BF16 operand tile of the MMA step that seeds INT32 scores. One
    # instance owns the tile; the other reads it through this reference.
    score_seed_owner: Constexpr["TmemSResource | None"] = None
    _seed_alloc: Constexpr[SmemAllocation | None] = None
    # The instance's ``sfK`` resources. Proxy routes read
    # ``sage_summary_k_scales``, which is ``sage_k_scales`` unless the summary
    # scales have their own K block size.
    sage_k_scales: SageKScalesResource | None = None
    sage_summary_k_scales: SageKScalesResource | None = None
    old_max_arr: Constexpr[TaskLocalVariable] = TaskLocalVariable.uninitialized()
    sum_arr: Constexpr[TaskLocalVariable] = TaskLocalVariable.uninitialized()
    new_max_arr: Constexpr[TaskLocalVariable] = TaskLocalVariable.uninitialized()
    s_arr: Constexpr[TaskLocalVariable] = TaskLocalVariable.uninitialized()
    sage_q_scale: Constexpr[TaskLocalVariable] = TaskLocalVariable.uninitialized()

    def _init_placeholder_state(self) -> None:
        """Create placeholder register and scratch state for softmax."""
        num_scale_groups = self.cfg.num_softmax_scale_groups
        num_s_regs = self.cfg.softmax_score_fragment_regs
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
        allocations = [self._scratch_alloc]
        if self.cfg.uses_int32_scores and self.score_seed_owner is None:
            if self._seed_alloc is None:
                self._seed_alloc = SmemAllocation(
                    name=f"{self.name}_scoreSeed",
                    size_bytes=self._seed_mma_tile_bytes(),
                    alignment=128,
                )
            allocations.append(self._seed_alloc)
        return allocations

    def _seed_mma_tile_bytes(self) -> int:
        """Return the byte size of the seeding MMA's shared operand tile.

        A reads the first M rows and B the first N rows of the same tile.
        """
        _, mma_m, mma_n = _qk_mma_operand_contract_for_config(self.cfg)
        return max(mma_m, mma_n) * INT32_SCORE_SEED_MMA_K * 2

    @cute.jit
    def _seed_mma_tile_words(self, context: ResourceContext) -> cutlass.Array:
        """Return the seeding MMA's operand tile as a 32-bit SMEM array."""
        owner = self if self.score_seed_owner is None else self.score_seed_owner
        return cutlass.Array(
            context.smem_base.data_ptr() + owner._seed_alloc.offset,
            dtype=Uint32,
            shape=(owner._seed_mma_tile_bytes() // 4,),
            addrspace=3,
        )

    @cute.jit
    def create_function_variables(
        self, context: ResourceContext | None = None
    ) -> ResourceVars:
        """Write the score-seed operand tile; the prologue barrier publishes it."""
        if cutlass.const_expr(
            self.cfg.uses_int32_scores
            and context is not None
            and context.smem_base is not None
        ):
            self.fill_score_seed_tiles(context)
        return {}

    @cute.jit
    def fill_score_seed_tiles(self, context: ResourceContext) -> None:
        """Fill the seeding MMA's constant operand tile with the whole CTA.

        The fence makes the generic-proxy stores visible to the tensor core.
        """
        assert self.score_seed_owner is None, "only the owning instance fills the tile"
        words = self._seed_mma_tile_words(context)
        num_words = self._seed_mma_tile_bytes() // 4
        num_threads = self.cfg.threads_per_cta
        tidx, _, _ = cute.arch.thread_idx()
        even_word, odd_word = INT32_SCORE_SEED_TILE_WORDS
        for round_idx in cutlass.range_constexpr(
            (num_words + num_threads - 1) // num_threads
        ):
            word_idx = tidx + Int32(round_idx * num_threads)
            if word_idx < Int32(num_words):
                word = Uint32(even_word)
                if (word_idx & Int32(1)) != Int32(0):
                    word = Uint32(odd_word)
                words[word_idx] = word
        cute.arch.fence_view_async_shared()

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
        if cutlass.const_expr(cfg.qk_dtype_bytes != 1 and crosses_64b_chunk):
            k_desc = k_desc + Int32(8 * cfg.tile_size_kv - 6)
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
        num_s_regs = self.cfg.softmax_score_fragment_regs
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
        num_s_regs = self.cfg.softmax_score_fragment_regs
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

    @cute.jit
    def _qk_tmem_col(self, stage_info: StageInfo, stage_slot_offset: Int32):
        """Return the TMEM pointer of one producer S slot.

        cutlass's tcgen05_alloc returns the base in tmem_ptr_i32, so add the
        per-resource column offset before issuing MMA.
        """
        task_cache = _decode_gen_task_cache(stage_info)
        return prims.make_tmem_ptr(
            task_cache[_TASK_CACHE_TMEM_BASE_OFFSET]
            + Int32(self._alloc.offset)
            + stage_slot_offset,
            Float32,
        )

    @cute.jit
    def _seed_scores(self, stage_info: StageInfo, stage_slot_offset: Int32) -> None:
        """Write ``INT32_SCORE_BIAS`` into the acquired S slot with one BF16 MMA step.

        The step reads the constant operand tile and no accumulator; the INT8
        K steps accumulate onto the seed's bit pattern in issue order.
        """
        cfg = self.cfg
        assert cfg.uses_int32_scores
        # The bias binade relies on ``|score| <= 2**21``, which holds for
        # 128-wide INT8 dot products.
        assert cfg.headdim == 128
        if prims.elect_sync():
            tmem_col = self._qk_tmem_col(stage_info, stage_slot_offset)
            _, mma_m, mma_n = _qk_mma_operand_contract_for_config(cfg)
            tile_desc = prims.Tcgen05SmemDesc.build(
                self._seed_mma_tile_words(stage_info.context),
                leading_byte_offset=INT32_SCORE_SEED_TILE_LBO,
                stride_byte_offset=INT32_SCORE_SEED_TILE_SBO,
                layout=prims.Tcgen05SmemSwizzle.NONE,
            )
            idesc = prims.Tcgen05InstrDesc.build(
                c_dtype=Float32,
                a_dtype=BFloat16,
                b_dtype=BFloat16,
                n_dim=mma_n,
                m_dim=mma_m,
            )
            if cutlass.const_expr(cfg.uses_ws_2x2_datapath):
                tcgen05_mma_ws(
                    prims.Tcgen05MMAKind.F16,
                    tmem_col,
                    tile_desc,
                    tile_desc,
                    idesc,
                    False,
                )
            else:
                prims.tcgen05_mma(
                    prims.Tcgen05MMAKind.F16,
                    prims.CTAGroup.CTA_1,
                    tmem_col,
                    tile_desc,
                    tile_desc,
                    idesc,
                    False,
                )

    @producer_work
    @cute.jit
    def seed_scores_head(self, stage_info: StageInfo) -> None:
        """Seed the initial S slot before HEAD QK awaits its K tile."""
        self._seed_scores(stage_info, self._qk_head_stage_slot_offset(stage_info))

    @producer_work
    @cute.jit
    def seed_scores_loop(self, stage_info: StageInfo) -> None:
        """Seed the next producer S slot before LOOP QK awaits its K tile."""
        self._seed_scores(stage_info, self._qk_loop_stage_slot_offset(stage_info))

    @producer_work
    @cute.jit
    def qk_mma_head(
        self,
        stage_info: StageInfo,
        *,
        q_desc: prims.Tcgen05SmemDesc,
        kv_desc: DescriptorValue,
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
        kv_desc: DescriptorValue,
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
        kv_desc: DescriptorValue,
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
        kv_desc: DescriptorValue,
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
        kv_desc: DescriptorValue,
        stage_slot_offset: Int32,
        head_dim_stage_idx: Constexpr[int],
    ) -> None:
        """Issue BMM1 with the selected Keeps/Swaps MMA orientation.

        Issues all 16-wide MMA-K slices for the current staged head-dim tile.
        Descriptor stage selection and per-slice jumps are centralized in the
        local descriptor helpers below.
        """
        cfg = self.cfg
        if cutlass.const_expr(cfg.store_transformed_kv_in_tmem):
            k_desc = prims.make_tmem_ptr(kv_desc, Int32)
        else:
            k_desc = _freeze_smem_descriptor(kv_desc)
        q_desc = _freeze_smem_descriptor(q_desc)
        q_desc, head_dim_stage_idx = self._q_desc_for_head_dim_stage(
            q_desc, head_dim_stage_idx
        )

        tmem_col = self._qk_tmem_col(stage_info, stage_slot_offset)

        q_is_a, mma_m, mma_n = _qk_mma_operand_contract_for_config(cfg)
        # Q and K use the effective QK precision after any KV transformation.
        idesc = prims.Tcgen05InstrDesc.build(
            c_dtype=_qk_accumulator_dtype(cfg),
            a_dtype=cfg.qk_dtype,
            b_dtype=cfg.qk_dtype,
            n_dim=mma_n,
            m_dim=mma_m,
        )

        if cutlass.const_expr(cfg.head_dim_per_stage_kv == 0):
            if prims.elect_sync():
                # INT32 scores accumulate onto the seeded bias; FP32 scores
                # overwrite S with the first slice.
                scale_d = cfg.uses_int32_scores
                for ki in cutlass.range_constexpr(cfg.headdim // _mma_k_step_qk(cfg)):
                    # Keeps computes Q x K^T (A=Q, B=K); Swaps computes the
                    # transposed K x Q^T tile (A=K, B=Q). Later slices
                    # accumulate.
                    if cutlass.const_expr(q_is_a):
                        a_desc, b_desc = q_desc, k_desc
                    else:
                        a_desc, b_desc = k_desc, q_desc
                    if cutlass.const_expr(cfg.uses_ws_2x2_datapath):
                        tcgen05_mma_ws(
                            _mma_kind_for_qk(cfg),
                            tmem_col,
                            a_desc,
                            b_desc,
                            idesc,
                            scale_d,
                        )
                    else:
                        prims.tcgen05_mma(
                            _mma_kind_for_qk(cfg),
                            prims.CTAGroup.CTA_1,
                            tmem_col,
                            a_desc,
                            b_desc,
                            idesc,
                            scale_d,
                        )
                    scale_d = True
                    if cutlass.const_expr(ki + 1 < cfg.headdim // _mma_k_step_qk(cfg)):
                        if cutlass.const_expr(cfg.store_transformed_kv_in_tmem):
                            k_desc = prims.make_tmem_ptr(
                                kv_desc + Int32((ki + 1) * _mma_k_step_qk(cfg) // 4),
                                Int32,
                            )
                            q_desc = q_desc + Int32(2)
                        else:
                            k_desc, q_desc = self._advance_qk_descs_after_mma_k(
                                k_desc,
                                q_desc,
                                crosses_64b_chunk=cfg.headdim == 128 and ki == 3,
                            )
        else:
            assert not cfg.uses_int32_scores, (
                "staged head-dim BMM1 does not seed scores"
            )
            mma_k_steps = cfg.head_dim_kv_stage // _mma_k_step_qk(cfg)
            if prims.elect_sync():
                # Peel the first MMA so overwrite-vs-accumulate remains a
                # compile-time value rather than loop-carried state.
                if cutlass.const_expr(q_is_a):
                    first_a_desc, first_b_desc = q_desc, k_desc
                else:
                    first_a_desc, first_b_desc = k_desc, q_desc
                prims.tcgen05_mma(
                    _mma_kind_for_qk(cfg),
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
                if cutlass.const_expr(cfg.store_transformed_kv_in_tmem):
                    iter_k_desc = prims.make_tmem_ptr(
                        kv_desc + ki * Int32(_mma_k_step_qk(cfg) // 4), Int32
                    )
                    q_desc_offset = ki * Int32(2)
                elif cutlass.const_expr(cfg.qk_dtype_bytes == 1):
                    k_desc_offset = ki * Int32(2)
                    q_desc_offset = ki * Int32(2)
                    iter_k_desc = k_desc + k_desc_offset
                else:
                    chunk_idx = (ki * Int32(_mma_k_step_qk(cfg))) // Int32(64)
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
                        _mma_kind_for_qk(cfg),
                        prims.CTAGroup.CTA_1,
                        tmem_col,
                        iter_a_desc,
                        iter_b_desc,
                        idesc,
                        cutlass.Boolean(True),
                    )

    @cute.jit
    def _resolve_keeps_tile_context(self, stage_info: StageInfo):
        """Resolve one score tile's logical position and boundary-mask state."""
        return resolve_keeps_tile_context(
            self.cfg,
            stage_info,
            inst_id=self.inst_id,
            seqlens_kv=self.seqlens_kv,
            max_seq_len_kv=self.max_seq_len_kv,
            seq_len_q=self.seq_len_q,
            q_group_idx=self.q_group_idx,
        )

    @cute.jit
    def _reduce_keeps_row_max(self, s_vals: cutlass.Array) -> Float32:
        """Reduce one Keeps score row while preserving its lane ownership."""

        cfg = self.cfg
        max_chains = cutlass.Array(Float32, 4, space=cutlass.AddressSpace.rmem)
        for chain_idx in cutlass.range_constexpr(4):
            max_chains[chain_idx] = _neg_max_f32()
        for reg_base in cutlass.range_constexpr(0, cfg.num_s_regs_per_thread, 4):
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
            # A Q64 row is split across lanes xor 16; Q128 already owns the
            # complete row locally and therefore needs no cross-lane combine.
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
        return tile_max

    @cute.jit
    def _publish_keeps_softmax_state(
        self,
        s_vals: cutlass.Array,
        tile_max: Float32,
        old_max: Float32,
        running_sum: Float32,
        old_max_arr: cutlass.Array,
        sum_arr: cutlass.Array,
        new_max_arr: cutlass.Array,
        s_arr: cutlass.Array,
    ) -> None:
        """Publish a masked Keeps row and its updated softmax anchor."""

        old_max_arr[0] = old_max
        sum_arr[0] = running_sum
        new_max_arr[0] = self._softmax_anchor(old_max, tile_max)
        for reg_idx in cutlass.range_constexpr(self.cfg.num_s_regs_per_thread):
            s_arr[reg_idx] = s_vals[reg_idx]

    @cute.jit
    def _proxy_score_shifts(
        self, route_is_proxy: cutlass.Boolean
    ) -> tuple[Float32, Float32]:
        """Return a route's ``(proxy_max_shift, tail_shift)`` in score units.

        A proxy route carries its token mass in the logit: the tile maximum
        moves by the full block's ``log2`` mass and the ragged final summary's
        score by its ``log2`` shortfall, both divided by ``c`` so they apply
        to scores before the anchor. Exact routes get zero shifts. Callers
        pass a warp-uniform route kind so the branch stays uniform.
        """
        proxy_max_shift = Float32(0.0)
        tail_shift = Float32(0.0)
        if route_is_proxy:
            inv_scale_softmax_log2 = cute.math.rcp(self.scale_softmax_log2)
            _, tail_log2_delta = self.cfg.proxy_tail_summary
            proxy_max_shift = (
                Float32(self.cfg.proxy_log2_block_mass) * inv_scale_softmax_log2
            )
            tail_shift = Float32(tail_log2_delta) * inv_scale_softmax_log2
        return proxy_max_shift, tail_shift

    @cute.jit
    def _softmax_anchor(self, old_max: Float32, tile_max: Float32) -> Float32:
        """Return the exponent reference max for the tile's P pass.

        Online softmax only requires a common finite reference for P, the
        running sum, and O; it does not require the exact row maximum.
        Profiles that defer anchor updates keep the previous reference while
        the tile raises it by less than ``SOFTMAX_RESCALE_THRESHOLD_LOG2``
        log2 units, so correction can skip the in-place TMEM O rescale. The
        16-bit P path represents the bounded values above one, and the
        numerator and denominator stay in the same scale frame. Larger jumps
        still rebase to keep P comfortably in range.
        """
        new_max = cute.math.max(old_max, tile_max, ftz=True)
        if cutlass.const_expr(self.cfg.defers_softmax_anchor_updates):
            if old_max != _neg_max_f32():
                max_delta_log2 = self.scale_softmax_log2 * (old_max - new_max)
                if max_delta_log2 >= Float32(-SOFTMAX_RESCALE_THRESHOLD_LOG2):
                    new_max = old_max
        return new_max

    @cute.jit
    def _q_token_kv_block_sparse_keeps_kv128_membership_word(
        self,
        stage_info: StageInfo,
        logical_q_group_idx: Int32,
        warp_grp_thread_idx: Int32,
        tile_row_idx: Int32,
    ) -> Uint32:
        """Return a lane-local 16/32-page keep word for Q64/Q128 with KV128."""

        cfg = self.cfg
        assert cfg.tile_size_q in (64, 128)
        assert cfg.tile_size_kv == 128
        assert cfg.uses_q_token_kv_block_sparse_page_membership
        assert self.page_offsets_ref is not None
        q_token_idx, _ = _q_row_token_and_local_head(
            cfg,
            self.h_r,
            logical_q_group_idx,
            tile_row_idx,
        )
        membership_bit = Uint32(1) << q_token_idx
        local_tile_idx = _softmax_tile_idx(cfg, stage_info, self.inst_id)
        lane_idx = warp_grp_thread_idx & Int32(31)
        col_base = _keeps_col_base(
            cfg,
            lane_idx,
            cfg.softmax_score_fragment_regs,
        )
        keep_word = Uint32(0)
        page_span = min(cfg.num_tokens_per_page, cfg.num_s_regs_per_thread)
        pages_per_lane = cfg.num_s_regs_per_thread // page_span
        for page_vector_idx in cutlass.range_constexpr(pages_per_lane // 4):
            memberships = (
                self.page_offsets_ref.q_token_kv_block_sparse_page_memberships4(
                    stage_info,
                    local_tile_idx,
                    col_base // Int32(cfg.num_tokens_per_page)
                    + Int32(page_vector_idx * 4),
                )
            )
            for vector_elem_idx in cutlass.range_constexpr(4):
                local_page_idx = page_vector_idx * 4 + vector_elem_idx
                page_is_member = Uint32(
                    (memberships[vector_elem_idx] & membership_bit) != Uint32(0)
                )
                keep_word = keep_word | (page_is_member << Uint32(local_page_idx))
        if cutlass.const_expr(pages_per_lane < 4):
            for local_page_idx in cutlass.range_constexpr(pages_per_lane):
                membership = (
                    self.page_offsets_ref.q_token_kv_block_sparse_page_membership(
                        stage_info,
                        local_tile_idx,
                        (col_base + Int32(local_page_idx * page_span))
                        // Int32(cfg.num_tokens_per_page),
                    )
                )
                member = Uint32((membership & membership_bit) != Uint32(0))
                keep_word = keep_word | (member << Uint32(local_page_idx))
        return keep_word

    @cute.jit
    def _load_keeps_fragment_impl(
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
        membership_keep_word: Uint32,
        *,
        apply_boundary_mask: Constexpr[bool],
    ) -> None:
        """Load one complete-row Keeps score tile with a compile-time mask policy.

        The caller chooses the masked/unmasked path before TMEM load. Keeping the
        score fragment out of the branch condition avoids carrying 64/128 live
        S registers through a post-load control-flow edge. Max reduction is a
        separate operation because the later P-materialization reload only
        needs the masked scores.
        """
        cfg = self.cfg
        task_cache = _decode_gen_task_cache(stage_info)
        num_s_regs = cfg.num_s_regs_per_thread
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
                    token_idx = tile_offset_k + _keeps_score_col(
                        cfg,
                        warp_grp_thread_idx,
                        reg_idx,
                        col_base,
                    )
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
                causal_start_rel = causal_start - tile_offset_k
                causal_end_rel = causal_end - tile_offset_k
                if cutlass.const_expr(cfg.uses_q_token_kv_block_sparse_page_membership):
                    assert cfg.tile_size_kv == 128
                    # Membership removes complete future pages; only tokens
                    # beyond the causal endpoint within its page need masking.
                    causal_tail_tokens = causal_end & Int32(cfg.num_tokens_per_page - 1)
                    causal_tail_page_rel = causal_end_rel - causal_tail_tokens
                    page_span = min(num_s_regs, cfg.num_tokens_per_page)
                    for local_page_idx in cutlass.range_constexpr(
                        num_s_regs // page_span
                    ):
                        page_score_col = _keeps_score_col(
                            cfg,
                            warp_grp_thread_idx,
                            local_page_idx * page_span,
                            col_base,
                        )
                        page_origin = (
                            page_score_col // Int32(cfg.num_tokens_per_page)
                        ) * Int32(cfg.num_tokens_per_page)
                        page_is_causal_tail = (
                            causal_tail_tokens != Int32(0)
                            and page_origin == causal_tail_page_rel
                        )
                        # A KV128 block may span two lane-local 64-column halves.
                        # The second half's first register can already be masked.
                        for token_in_span in cutlass.range_constexpr(
                            0 if cfg.num_tokens_per_page > num_s_regs else 1, page_span
                        ):
                            token_offset = (
                                page_score_col - page_origin + Int32(token_in_span)
                            )
                            valid = not (
                                page_is_causal_tail
                                and token_offset >= causal_tail_tokens
                            )
                            reg = local_page_idx * page_span + token_in_span
                            s_vals[reg] = cutlass.select_(
                                valid, s_vals[reg], _neg_max_f32()
                            )
                else:
                    for reg_idx in cutlass.range_constexpr(num_s_regs):
                        score_col = _keeps_score_col(
                            cfg,
                            warp_grp_thread_idx,
                            reg_idx,
                            col_base,
                        )
                        if cutlass.const_expr(cfg.use_sliding_window_causal):
                            if score_col < causal_start_rel:
                                s_vals[reg_idx] = _neg_max_f32()
                        if score_col >= causal_end_rel:
                            s_vals[reg_idx] = _neg_max_f32()

        if cutlass.const_expr(cfg.uses_q_token_kv_block_sparse_page_membership):
            # Q64 lanes own a contiguous 64-column half; Q128 lanes own the
            # complete 128-column row. The preloaded 16/32-page keep word
            # avoids carrying SMEM values through this dynamic
            # masked/unmasked loader specialization.
            assert cfg.tile_size_kv == 128
            page_span = min(num_s_regs, cfg.num_tokens_per_page)
            for local_page_idx in cutlass.range_constexpr(num_s_regs // page_span):
                page_is_member = (
                    (membership_keep_word >> Uint32(local_page_idx)) & Uint32(1)
                ) != Uint32(0)
                for token_in_page in cutlass.range_constexpr(page_span):
                    membership_reg_idx = local_page_idx * page_span + token_in_page
                    s_vals[membership_reg_idx] = cutlass.select_(
                        page_is_member,
                        s_vals[membership_reg_idx],
                        _neg_max_f32(),
                    )

        if cutlass.const_expr(cfg.q_score_rows_need_mask):
            if not _q_row_is_valid_for_seq(
                cfg,
                self.h_r,
                logical_q_group_idx,
                tile_row_idx,
                self.seq_len_q,
            ):
                for invalid_q_reg_idx in cutlass.range_constexpr(num_s_regs):
                    s_vals[invalid_q_reg_idx] = _neg_max_f32()

    @cute.jit
    def _load_keeps_fragment(
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
        tile_is_unmasked: cutlass.Boolean,
    ) -> None:
        """Select the masked or unmasked fragment loader before LDTM.

        ``tile_is_unmasked`` is runtime state, whereas the implementation's
        mask policy remains constexpr. Keeping the branch outside the loader
        lets the unmasked specialization erase boundary-mask instructions and
        avoids carrying the loaded score registers through a post-LDTM branch.
        """
        membership_keep_word = Uint32(0xFFFFFFFF)
        if cutlass.const_expr(self.cfg.uses_q_token_kv_block_sparse_page_membership):
            task_cache = _decode_gen_task_cache(stage_info)
            warp_grp_thread_idx = task_cache[_TASK_CACHE_WARP_GRP_THREAD_IDX]
            tile_row_idx = _keeps_row_idx(self.cfg, warp_grp_thread_idx)
            membership_keep_word = (
                self._q_token_kv_block_sparse_keeps_kv128_membership_word(
                    stage_info,
                    logical_q_group_idx,
                    warp_grp_thread_idx,
                    tile_row_idx,
                )
            )
        if tile_is_unmasked:
            self._load_keeps_fragment_impl(
                stage_info,
                s_vals,
                tile_offset_k,
                element_mask_end_idx,
                window_start_idx,
                seq_len_kv,
                logical_q_group_idx,
                is_valid_effective_tile,
                is_masked_final_wave,
                membership_keep_word,
                apply_boundary_mask=False,
            )
        else:
            self._load_keeps_fragment_impl(
                stage_info,
                s_vals,
                tile_offset_k,
                element_mask_end_idx,
                window_start_idx,
                seq_len_kv,
                logical_q_group_idx,
                is_valid_effective_tile,
                is_masked_final_wave,
                membership_keep_word,
                apply_boundary_mask=True,
            )

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
        if cutlass.const_expr(cfg.streams_tmem_p_fragments):
            # Streamed profiles share the fragment max pass with block-sparse
            # routes; dense tiles describe their visible range as keep words.
            return self._compute_softmax_loop_keeps_fragments(
                stage_info,
                old_max_arr=old_max_arr,
                sum_arr=sum_arr,
                new_max_arr=new_max_arr,
                s_arr=s_arr,
                use_sparse=False,
            )
        task_cache = _decode_gen_task_cache(stage_info)
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

        (
            seq_len_kv,
            logical_q_group_idx,
            element_mask_end_idx,
            tile_offset_k,
            window_start_idx,
            is_valid_effective_tile,
            is_masked_final_wave,
            tile_is_unmasked,
            tile_has_valid_scores,
        ) = self._resolve_keeps_tile_context(stage_info)

        if cutlass.const_expr(use_preload_mask_split):
            # Select the complete unmasked/masked TMEM load+max path before any S
            # registers are materialized. The shared predicate covers the
            # intersection of all active grouped-Q causal/window intervals.
            tile_max = _neg_max_f32()
            self._load_keeps_fragment(
                stage_info,
                s_vals,
                tile_offset_k,
                element_mask_end_idx,
                window_start_idx,
                seq_len_kv,
                logical_q_group_idx,
                is_valid_effective_tile,
                is_masked_final_wave,
                tile_is_unmasked,
            )
            tile_max = self._reduce_keeps_row_max(s_vals)

            self._publish_keeps_softmax_state(
                s_vals,
                tile_max,
                old_max,
                running_sum,
                old_max_arr,
                sum_arr,
                new_max_arr,
                s_arr,
            )
            return old_max_arr, sum_arr, new_max_arr, s_arr

        if tile_has_valid_scores:
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
            token_idx = tile_offset_k + _keeps_score_col(
                cfg, warp_grp_thread_idx, reg_idx, col_base
            )
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
                token_idx = tile_offset_k + _keeps_score_col(
                    cfg, warp_grp_thread_idx, reg_idx, col_base
                )
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

        tile_max = self._reduce_keeps_row_max(s_vals)
        self._publish_keeps_softmax_state(
            s_vals,
            tile_max,
            old_max,
            running_sum,
            old_max_arr,
            sum_arr,
            new_max_arr,
            s_arr,
        )
        return old_max_arr, sum_arr, new_max_arr, s_arr

    @cute.jit
    def _apply_q_token_kv_block_sparse_swaps_page_membership_mask(
        self,
        stage_info: StageInfo,
        s_vals: cutlass.Array,
        task_cache: cutlass.Array,
        logical_q_group_idx: Int32,
        local_tile_idx: Int32,
    ) -> None:
        """Mask Swaps scores whose page is absent from a grouped Q row."""

        cfg = self.cfg
        assert not cfg.use_keeps_mma_ab
        assert cfg.uses_q_token_kv_block_sparse_page_membership
        assert self.page_offsets_ref is not None
        num_scale_groups = cfg.num_softmax_scale_groups
        q_repeats = max(cfg.tile_size_q // 8, 1)
        warp_idx = task_cache[_TASK_CACHE_WARP_IDX]
        lane_idx = task_cache[_TASK_CACHE_LANE_IDX]
        col_group_idx = lane_idx & Int32(0x3)
        local_idx_k0 = warp_idx * Int32(32) + (lane_idx >> Int32(2))

        # Each held page-4 slot has a separate query-membership byte; locator
        # values never carry membership bits. One lookup serves the pair of Q
        # rows represented by a Swaps score register pair; causal masking still
        # handles the zero-to-three-token tail within a member page.
        for scale_idx in cutlass.range_constexpr(num_scale_groups):
            repeat_idx = scale_idx // 2
            pair_idx = scale_idx % 2
            tile_row_idx = (
                Int32(repeat_idx * 8) + col_group_idx * Int32(2) + Int32(pair_idx)
            )
            q_token_idx, _ = _q_row_token_and_local_head(
                cfg,
                self.h_r,
                logical_q_group_idx,
                tile_row_idx,
            )
            membership_bit = Uint32(1) << q_token_idx
            for token_group_idx in cutlass.range_constexpr(4):
                local_token_idx = local_idx_k0 + Int32(token_group_idx * 8)
                page_frag = local_token_idx // Int32(cfg.num_tokens_per_page)
                membership = (
                    self.page_offsets_ref.q_token_kv_block_sparse_page_membership(
                        stage_info,
                        local_tile_idx,
                        page_frag,
                    )
                )
                s_idx = (
                    repeat_idx * 4
                    + pair_idx
                    + (token_group_idx & 1) * 2
                    + (token_group_idx >> 1) * q_repeats * 4
                )
                if (membership & membership_bit) == Uint32(0):
                    s_vals[s_idx] = _neg_max_f32()

    @cute.jit
    def _compute_softmax_loop_swaps(
        self,
        stage_info: StageInfo,
        *,
        old_max_arr: cutlass.Array,
        sum_arr: cutlass.Array,
        new_max_arr: cutlass.Array,
        s_arr: cutlass.Array,
        sparse_origin0: Int32,
        sparse_origin1: Int32,
        sparse_origin2: Int32,
        sparse_origin3: Int32,
        sparse_token_word: Uint32,
        sparse_route_flags: Uint32,
        use_sparse: Constexpr[bool],
    ) -> tuple[object, object, object, object]:
        """Load SWAP S from TMEM and materialize the running softmax state.

        Operation order: load BMM1 scores, apply tail/window masks, reduce the
        row max through shared scratch, and return the old/new max payload that
        correction consumes.
        """
        cfg = self.cfg
        assert not cfg.use_keeps_mma_ab
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
        should_load_s = True
        if cutlass.const_expr(not use_sparse):
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
                effective_total_kv_tiles = Int32(
                    task_cache[_TASK_CACHE_KV_VALID_TILE_END]
                )
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

            if cutlass.const_expr(not use_runtime_paged_dense_load):
                should_load_s = (
                    is_valid_effective_tile
                    and (tile_offset_k < seq_len_kv)
                    and not is_masked_final_wave
                )
        else:
            # Sparse routes always have a committed S tile. Invalid atoms were
            # zero-filled by TMA and are suppressed below by either the staged
            # origin predicate or the prepared token word.
            use_runtime_kv_domain = False
            effective_tile_idx = Int32(0)
            effective_total_kv_tiles = Int32(1)
            tile_offset_k = Int32(0)
            window_start_idx = Int32(0)
            is_masked_final_wave = cutlass.Boolean(False)
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

        route_is_proxy = cutlass.Boolean(False)
        if cutlass.const_expr(use_sparse and cfg.use_block_sparse_proxy_routes):
            route_is_proxy = _route_is_proxy(sparse_route_flags.bitcast(Int32))
        if cutlass.const_expr(use_sparse):
            # Route, KV-tail, uniform-causal, and token validity depend only on
            # K, so one predicate masks the adjacent pair of Q-row registers.
            should_apply_sparse_mask = cutlass.Boolean(True)
            if cutlass.const_expr(
                _swaps_forwards_packed_route_full(cfg)
                or _swaps_uses_origin0_k32_full_guard(cfg)
            ):
                if cutlass.const_expr(_swaps_forwards_packed_route_full(cfg)):
                    # Prepare already proved structural fullness for the
                    # complete KV128 route and staging replicated the summary
                    # for this Softmax warp's logical K32 slice.
                    k32_is_full = cute.arch.make_warp_uniform(
                        cutlass.Boolean(
                            (sparse_route_flags & Uint32(_PREPARED_ROUTE_IS_FULL_FLAG))
                            != Uint32(0)
                        )
                    )
                else:
                    # One origin covers this warp's K32 slice, so it can
                    # bypass all four lane-local K8 predicates. B16 stays on
                    # the straight-line path: its two-origin guard is not
                    # cheaper after code generation.
                    k32_is_full = cute.arch.make_warp_uniform(
                        cutlass.Boolean(
                            sparse_origin0 >= Int32(0)
                            and sparse_origin0 <= seq_len_kv - Int32(32)
                        )
                    )
                should_apply_sparse_mask = cutlass.Boolean(not k32_is_full)
            if should_apply_sparse_mask:
                lane_k_offset = Int32(task_cache[_TASK_CACHE_LANE_IDX]) >> Int32(2)
                token_word_covers_kv_tail = _swaps_token_word_covers_kv_tail(cfg)
                for token_group_idx in cutlass.range_constexpr(4):
                    atom_origin, logical_k = _swaps_routed_coordinate(
                        cfg,
                        lane_k_offset,
                        sparse_origin0,
                        sparse_origin1,
                        sparse_origin2,
                        sparse_origin3,
                        token_group_idx=token_group_idx,
                    )
                    # Prepared words zero absent atoms and the logical KV
                    # tail. Qualified profiles can therefore omit the local
                    # atom-origin guard, independently of the K/V issuer warp.
                    score_is_valid = cutlass.Boolean(True)
                    if not route_is_proxy:
                        if cutlass.const_expr(
                            not _swaps_uses_token_only_score_validity(cfg)
                        ):
                            score_is_valid = cutlass.Boolean(atom_origin >= Int32(0))
                        if cutlass.const_expr(not token_word_covers_kv_tail):
                            score_is_valid = cutlass.Boolean(
                                score_is_valid and logical_k < seq_len_kv
                            )
                        if cutlass.const_expr(cfg.uses_uniform_causal_mask):
                            score_is_valid = cutlass.Boolean(
                                score_is_valid and logical_k < element_mask_end_idx
                            )
                    if cutlass.const_expr(cfg.uses_prepared_score_keep_words):
                        token_bit_idx = Int32(token_group_idx * 8) + lane_k_offset
                        token_is_valid = (
                            (sparse_token_word >> token_bit_idx) & Uint32(1)
                        ) != Uint32(0)
                        score_is_valid = cutlass.Boolean(
                            score_is_valid and token_is_valid
                        )
                    if not score_is_valid:
                        for repeat_idx in cutlass.range_constexpr(q_repeats):
                            if cutlass.const_expr(token_group_idx < 2):
                                s_base = repeat_idx * 4 + token_group_idx * 2
                            else:
                                s_base = (
                                    q_repeats * 4
                                    + repeat_idx * 4
                                    + (token_group_idx - 2) * 2
                                )
                            s_vals[s_base + 0] = _neg_max_f32()
                            s_vals[s_base + 1] = _neg_max_f32()

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
        if cutlass.const_expr(use_sparse):
            should_apply_dense_mask = False
        elif cutlass.const_expr(
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

        if cutlass.const_expr(cfg.uses_q_token_kv_block_sparse_page_membership):
            self._apply_q_token_kv_block_sparse_swaps_page_membership_mask(
                stage_info,
                s_vals,
                task_cache,
                logical_q_group_idx,
                local_tile_idx,
            )

        if cutlass.const_expr(cfg.uses_per_row_causal_mask):
            apply_per_row_causal_mask = cutlass.Boolean(True)
            if cutlass.const_expr(not use_sparse):
                tile_has_valid_scores = (
                    is_valid_effective_tile
                    and (tile_offset_k < seq_len_kv)
                    and not is_masked_final_wave
                )
                apply_per_row_causal_mask = cutlass.Boolean(
                    not _kv_tile_is_fully_unmasked_for_q_group(
                        cfg,
                        tile_offset_k,
                        seq_len_kv,
                        self.seq_len_q,
                        q_token_base,
                        tile_has_valid_scores,
                    )
                )
            if apply_per_row_causal_mask:
                # Grouped causal decode has a distinct causal/window bound for
                # every Q token. Sparse routes always use their logical K;
                # dense routes retain the boundary-tile fast path above.
                causal_warp_idx = task_cache[_TASK_CACHE_WARP_IDX]
                causal_lane_idx = task_cache[_TASK_CACHE_LANE_IDX]
                causal_col_group_idx = causal_lane_idx & Int32(0x3)
                causal_local_idx_k0 = causal_warp_idx * Int32(32) + (
                    causal_lane_idx >> Int32(2)
                )
                for scale_idx in cutlass.range_constexpr(num_scale_groups):
                    repeat_idx = scale_idx // 2
                    pair_idx = scale_idx % 2
                    tile_row_idx = (
                        Int32(repeat_idx * 8)
                        + causal_col_group_idx * Int32(2)
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
                            tile_offset_k
                            + causal_local_idx_k0
                            + Int32(token_group_idx * 8)
                        )
                        if cutlass.const_expr(use_sparse):
                            _, token_idx = _swaps_routed_coordinate(
                                cfg,
                                causal_lane_idx >> Int32(2),
                                sparse_origin0,
                                sparse_origin1,
                                sparse_origin2,
                                sparse_origin3,
                                token_group_idx=token_group_idx,
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
        # Proxy mass (``_proxy_score_shifts``): the ragged final summary's
        # scores shift before the reduction so the P pass sees its true mass.
        proxy_max_shift = Float32(0.0)
        if cutlass.const_expr(use_sparse and cfg.use_block_sparse_proxy_routes):
            tail_summary_idx, tail_log2_delta = cfg.proxy_tail_summary
            route_is_proxy_uniform = cute.arch.make_warp_uniform(route_is_proxy)
            proxy_max_shift, tail_shift = self._proxy_score_shifts(
                route_is_proxy_uniform
            )
            if cutlass.const_expr(tail_log2_delta != 0.0):
                if route_is_proxy_uniform:
                    for token_group_idx in cutlass.range_constexpr(4):
                        atom_origin, logical_summary = _swaps_routed_coordinate(
                            cfg,
                            Int32(lane_idx) >> Int32(2),
                            sparse_origin0,
                            sparse_origin1,
                            sparse_origin2,
                            sparse_origin3,
                            token_group_idx=token_group_idx,
                        )
                        # Invalid atoms never hold the tail.
                        if atom_origin >= Int32(0) and logical_summary == Int32(
                            tail_summary_idx
                        ):
                            for tail_repeat_idx in cutlass.range_constexpr(q_repeats):
                                tail_s_base = tail_repeat_idx * 4 + token_group_idx * 2
                                if cutlass.const_expr(token_group_idx >= 2):
                                    tail_s_base = (
                                        q_repeats * 4
                                        + tail_repeat_idx * 4
                                        + (token_group_idx - 2) * 2
                                    )
                                s_vals[tail_s_base + 0] = (
                                    s_vals[tail_s_base + 0] + tail_shift
                                )
                                s_vals[tail_s_base + 1] = (
                                    s_vals[tail_s_base + 1] + tail_shift
                                )
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
            if cutlass.const_expr(use_sparse and cfg.use_block_sparse_proxy_routes):
                # The mass enters before the anchor, so the P range is unchanged.
                local_max = local_max + proxy_max_shift
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
        """Load S from TMEM and materialize the running softmax state."""

        if cutlass.const_expr(self.cfg.use_keeps_mma_ab):
            return self._compute_softmax_loop_keeps(
                stage_info,
                old_max_arr=old_max_arr,
                sum_arr=sum_arr,
                new_max_arr=new_max_arr,
                s_arr=s_arr,
            )
        return self._compute_softmax_loop_swaps(
            stage_info,
            old_max_arr=old_max_arr,
            sum_arr=sum_arr,
            new_max_arr=new_max_arr,
            s_arr=s_arr,
            sparse_origin0=Int32(-1),
            sparse_origin1=Int32(-1),
            sparse_origin2=Int32(-1),
            sparse_origin3=Int32(-1),
            sparse_token_word=Uint32(0xFFFFFFFF),
            sparse_route_flags=Uint32(0),
            use_sparse=False,
        )

    @consumer_work(work_attrs=WorkAttr.AUXILIARY, returns=sage_q_scale)
    @cute.jit
    def load_sage_q_scale(self, stage_info: StageInfo) -> Float32:
        """Load ``sfQ`` of the lane's Q row, fixed for the whole work tile."""
        cfg = self.cfg
        assert cfg.use_sage_attention
        task_cache = _decode_gen_task_cache(stage_info)
        warp_grp_thread_idx = Int32(task_cache[_TASK_CACHE_WARP_GRP_THREAD_IDX])
        kv_head_idx, batch_idx = _logical_head_batch(
            stage_info, self.h_k_idx, self.b_idx
        )
        q_token_idx, local_head_idx = _q_row_token_and_local_head(
            cfg,
            self.h_r,
            _logical_q_group_idx(cfg, stage_info, self.q_group_idx),
            _keeps_row_idx(cfg, warp_grp_thread_idx),
        )
        return load_q_scale(
            cfg,
            self.scale_tensors.q_scale_ptr.toint(),
            self.scale_tensors.q_scale_head_stride,
            kv_head_idx=kv_head_idx,
            local_head_idx=local_head_idx,
            batch_idx=batch_idx,
            q_token_idx=q_token_idx,
        )

    @consumer_work(returns=("old_max_arr", "sum_arr", "new_max_arr", "s_arr"))
    @cute.jit
    def compute_sage_softmax_loop(
        self,
        stage_info: StageInfo,
        *,
        old_max_arr: cutlass.Array,
        sum_arr: cutlass.Array,
        new_max_arr: cutlass.Array,
        s_arr: cutlass.Array,
        sage_q_scale: Float32,
        sage_scale_arr: cutlass.Array,
        sage_summary_scale_arr: cutlass.Array,
    ) -> tuple[object, object, object, object]:
        """Consume S with per-group Sage scales and publish the dequantized max."""

        assert self.cfg.use_sage_attention
        return self._compute_softmax_loop_keeps_fragments(
            stage_info,
            old_max_arr=old_max_arr,
            sum_arr=sum_arr,
            new_max_arr=new_max_arr,
            s_arr=s_arr,
            use_sparse=False,
            sage_q_scale=sage_q_scale,
            sage_scale_arr=sage_scale_arr,
            sage_summary_scale_arr=sage_summary_scale_arr,
        )

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
        # so local_sum_arr represents the exact P payload consumed by BMM2.
        if cutlass.const_expr(cfg.use_fp8_pv):
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
                and not cfg.use_fp8_pv
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

    @cute.jit
    def _compute_softmax_loop_keeps_fragments(
        self,
        stage_info: StageInfo,
        *,
        old_max_arr: cutlass.Array,
        sum_arr: cutlass.Array,
        new_max_arr: cutlass.Array,
        s_arr: cutlass.Array,
        use_sparse: Constexpr[bool],
        sparse_origin0: Int32 | None = None,
        sparse_origin1: Int32 | None = None,
        sparse_route_flags: Int32 | None = None,
        sparse_token_word0: Uint32 | None = None,
        sparse_token_word1: Uint32 | None = None,
        sparse_token_word2: Uint32 | None = None,
        sparse_token_word3: Uint32 | None = None,
        sage_q_scale: Float32 | None = None,
        sage_scale_arr: cutlass.Array | None = None,
        sage_summary_scale_arr: cutlass.Array | None = None,
    ) -> tuple[object, object, object, object]:
        """Mask streamed K32 score fragments in place and reduce their max.

        Every fragment gets one keep word. Block-sparse routes derive it from
        their two K64 atom origins, validity flags and prepared token words;
        dense tiles derive it from the tile's visible token range (sequence
        end, uniform or per-row causal end, sliding-window start) and the Q
        row's validity, and leave the route arguments unset. Masked
        fragments are written back to TMEM so the P pass can reload them
        without any mask logic. With Sage attention the scores stay quantized
        unless the geometry has one scale per score: each fragment folds its
        group maxima with ``sfK`` and ``sfQ`` scales the tile maximum once.
        Biased INT32 scores (``INT32_SCORE_BIAS``) share this path; the bias
        leaves with the group maxima.
        """
        cfg = self.cfg
        assert cfg.streams_tmem_p_fragments
        num_fragments = cfg.num_softmax_score_fragments
        fragment_regs = cfg.softmax_score_fragment_regs
        # The seven-slot softmax metadata ABI carries exactly four token words.
        assert num_fragments == 4 and fragment_regs == 32
        task_cache = _decode_gen_task_cache(stage_info)
        keep_words = cutlass.Array(
            Uint32, num_fragments, space=cutlass.AddressSpace.rmem
        )
        warp_grp_thread_idx = Int32(task_cache[_TASK_CACHE_WARP_GRP_THREAD_IDX])
        tile_row_idx = _keeps_row_idx(cfg, warp_grp_thread_idx)
        if cutlass.const_expr(use_sparse):
            token_words = (
                sparse_token_word0,
                sparse_token_word1,
                sparse_token_word2,
                sparse_token_word3,
            )
            logical_q_group_idx = _logical_q_group_idx(
                cfg, stage_info, self.q_group_idx
            )
            q_token_idx, _ = _q_row_token_and_local_head(
                cfg,
                self.h_r,
                logical_q_group_idx,
                tile_row_idx,
            )
            q_row_is_valid = _q_row_is_valid_for_seq(
                cfg,
                self.h_r,
                logical_q_group_idx,
                tile_row_idx,
                self.seq_len_q,
            )
            seq_len_kv = _load_runtime_seq_len_kv(
                self.seqlens_kv,
                self.max_seq_len_kv,
                stage_info,
                Int32(0),
                Int32(0),
            )
            causal_end = seq_len_kv - self.seq_len_q + q_token_idx + Int32(1)
            origin0 = Int32(sparse_origin0)
            origin1 = Int32(sparse_origin1)
            valid0 = sparse_route_flags & Int32(1)
            valid1 = (sparse_route_flags >> Int32(1)) & Int32(1)
            fragments_per_origin = cfg.softmax_fragments_per_route_atom
            for fragment_idx in cutlass.range_constexpr(num_fragments):
                atom_offset = Int32(
                    (fragment_idx % fragments_per_origin) * fragment_regs
                )
                fragment_origin = origin0 + atom_offset
                fragment_valid = valid0
                if cutlass.const_expr(fragment_idx >= fragments_per_origin):
                    fragment_origin = origin1 + atom_offset
                    fragment_valid = valid1
                if cutlass.const_expr(cfg.trusts_prepared_score_words):
                    prepared_keep_word = Uint32(0)
                    if q_row_is_valid:
                        prepared_keep_word = Uint32(token_words[fragment_idx])
                    keep_words[fragment_idx] = prepared_keep_word
                else:
                    keep_words[fragment_idx] = _sparse_effective_keep_word(
                        q_row_is_valid,
                        fragment_origin,
                        fragment_valid,
                        Uint32(token_words[fragment_idx]),
                        seq_len_kv,
                        causal_end,
                        apply_causal_mask=cfg.mask_type == CAUSAL,
                        apply_token_mask=cfg.uses_prepared_score_keep_words,
                    )

        else:
            (
                seq_len_kv,
                logical_q_group_idx,
                element_mask_end_idx,
                tile_offset_k,
                window_start_idx,
                _is_valid_effective_tile,
                _is_masked_final_wave,
                tile_is_unmasked,
                rows_are_active,
            ) = self._resolve_keeps_tile_context(stage_info)
            if cutlass.const_expr(cfg.q_score_rows_need_mask):
                rows_are_active = cutlass.Boolean(
                    rows_are_active
                    and _q_row_is_valid_for_seq(
                        cfg,
                        self.h_r,
                        logical_q_group_idx,
                        tile_row_idx,
                        self.seq_len_q,
                    )
                )
            visible_start = Int32(0)
            visible_end = element_mask_end_idx
            if cutlass.const_expr(cfg.uses_per_row_causal_mask):
                q_token_idx, _ = _q_row_token_and_local_head(
                    cfg,
                    self.h_r,
                    logical_q_group_idx,
                    tile_row_idx,
                )
                visible_end = seq_len_kv - self.seq_len_q + q_token_idx + Int32(1)
                visible_start = _sliding_window_start_idx(
                    cfg, seq_len_kv, self.seq_len_q, q_token_idx
                )
            elif cutlass.const_expr(cfg.use_sliding_window_causal):
                visible_start = window_start_idx
            # A tile that is unmasked for the whole Q group has all-ones keep
            # words on every active row, so only masked tiles build them.
            warp_scores_are_unmasked = cute.arch.vote_all_sync(
                cutlass.Boolean(tile_is_unmasked and rows_are_active)
            )
        if cutlass.const_expr(use_sparse):
            warp_scores_are_unmasked = cutlass.Boolean(True)
            for fragment_idx in cutlass.range_constexpr(num_fragments):
                warp_scores_are_unmasked = cutlass.Boolean(
                    warp_scores_are_unmasked
                    and keep_words[fragment_idx] == Uint32(0xFFFFFFFF)
                )
            # The load/store branch must be uniform for each participating warp.
            warp_scores_are_unmasked = cute.arch.vote_all_sync(warp_scores_are_unmasked)

        # Proxy mass (``_proxy_score_shifts``): the ragged final summary's
        # score shifts before the fold and the TMEM write-back, so the P pass
        # reloads the shifted score.
        proxy_max_shift = Float32(0.0)
        tail_shift = Float32(0.0)
        tail_fragment_mask = Int32(0)
        tail_lane: Constexpr[int] = 0
        shifts_tail: Constexpr[bool] = False
        route_is_proxy = cutlass.Boolean(False)
        # A mixed plan quantizes proxy summaries with their own K block size,
        # so exact and proxy tiles read different ``sfK`` geometries; the
        # CTA-uniform route kind selects one per tile.
        mixed_scales: Constexpr[bool] = use_sparse and cfg.sage_mixed_k_geometry
        # The unrolled unmasked pass serves the exact geometry only.
        exact_dequantized: Constexpr[bool] = cfg.sage_scores_dequantized_for(
            proxy=False
        )
        if cutlass.const_expr(use_sparse and cfg.use_block_sparse_proxy_routes):
            # A warp-uniform route kind keeps the fold and the load/store
            # branch below on the uniform datapath.
            route_is_proxy = cute.arch.make_warp_uniform(
                _route_is_proxy(sparse_route_flags)
            )
            tail_summary_idx, tail_log2_delta = cfg.proxy_tail_summary
            tail_lane = tail_summary_idx % fragment_regs
            shifts_tail = tail_log2_delta != 0.0
            proxy_max_shift, tail_shift = self._proxy_score_shifts(route_is_proxy)
            if cutlass.const_expr(shifts_tail and cfg.use_sage_attention):
                # Sage scores are quantized: the shift is divided by the row's
                # ``sfQ`` here and by the tail group's ``sfK`` in the fragment.
                tail_shift = tail_shift * cute.math.rcp(sage_q_scale, approx=True)
            if cutlass.const_expr(shifts_tail):
                # One mask bit per fragment marks the one holding the ragged
                # final summary, whose half and fragment are compile-time
                # (``proxy_static_tail_fragment``). A thread holds it on a
                # proxy route of the last summary group, in the tail's half;
                # with several groups, ``origin0`` identifies the last one.
                tail_half, tail_fragment = cfg.proxy_static_tail_fragment
                holds_tail = cutlass.Boolean(
                    route_is_proxy
                    and _keeps_spatial_half(cfg, warp_grp_thread_idx)
                    == Int32(tail_half)
                )
                if cutlass.const_expr(cfg.num_proxy_groups > 1):
                    # The half's first atom is atom ``tail_half`` of the group.
                    tail_group_origin0 = (
                        cfg.num_proxy_groups - 1
                    ) * cfg.tile_size_kv + tail_half * cfg.block_sparse_kv_atom_size
                    holds_tail = cutlass.Boolean(
                        holds_tail and origin0 == Int32(tail_group_origin0)
                    )
                if holds_tail:
                    tail_fragment_mask = Int32(1 << tail_fragment)
                tail_fragment_mask = cute.arch.make_warp_uniform(tail_fragment_mask)
                # The shifted tail score must reach TMEM, so the route takes
                # the store branch.
                warp_scores_are_unmasked = cute.arch.vote_all_sync(
                    cutlass.Boolean(
                        warp_scores_are_unmasked and tail_fragment_mask == Int32(0)
                    )
                )
        if cutlass.const_expr(mixed_scales):
            # Proxy tiles take the rolled pass, so the summary geometry's fold
            # is emitted once rather than per unrolled fragment.
            warp_scores_are_unmasked = cutlass.Boolean(
                warp_scores_are_unmasked and not route_is_proxy
            )

        score_tmem_addr = (
            task_cache[_TASK_CACHE_TMEM_BASE_OFFSET]
            + Int32(self._alloc.offset)
            + self._softmax_loop_stage_slot_offset(stage_info)
        )
        max_chains = cutlass.Array(Float32, 4, space=cutlass.AddressSpace.rmem)
        max_identity = _neg_max_f32()
        if cutlass.const_expr(cfg.use_sage_attention):
            max_identity = Float32(float("-inf"))
        for chain_idx in cutlass.range_constexpr(4):
            max_chains[chain_idx] = max_identity
        scales_view = None
        summary_view = None
        if cutlass.const_expr(cfg.use_sage_attention):
            scales_view = self.sage_k_scales.open(stage_info, sage_scale_arr, None)
            if cutlass.const_expr(mixed_scales):
                summary_view = self.sage_summary_k_scales.open(
                    stage_info, sage_summary_scale_arr, None
                )

        if warp_scores_are_unmasked:
            for fragment_idx in cutlass.range_constexpr(num_fragments):
                if cutlass.const_expr(cfg.use_sage_attention):
                    # Issued ahead of the score load so an SMEM read hides
                    # behind the TMEM wait.
                    fragment_scales = self.sage_k_scales.fragment(
                        scales_view, Int32(fragment_idx)
                    )
                loaded = _keeps_tcgen05_ld(
                    cfg,
                    prims.make_tmem_ptr(
                        score_tmem_addr + Int32(fragment_idx * fragment_regs),
                        Float32,
                    ),
                    num=fragment_regs,
                    offset=cfg.tile_size_kv // 2,
                )
                prims.tcgen05_wait(kind=prims.Tcgen05Wait.LOAD)
                if cutlass.const_expr(exact_dequantized):
                    scores = cutlass.Array(
                        Float32, fragment_regs, space=cutlass.AddressSpace.rmem
                    )
                    for score_idx in cutlass.range_constexpr(fragment_regs):
                        scores[score_idx] = Float32(loaded[score_idx])
                    self._dequantize_fragment_scores(
                        scores, fragment_scales, cfg.sage_k_groups_per_fragment
                    )
                    self.sage_k_scales.advance(scales_view)
                    for score_idx in cutlass.range_constexpr(fragment_regs):
                        chain_idx: Constexpr[int] = score_idx % 4
                        max_chains[chain_idx] = cute.math.max(
                            max_chains[chain_idx], Float32(scores[score_idx]), ftz=True
                        )
                    _keeps_tcgen05_st(
                        cfg,
                        prims.make_tmem_ptr(
                            score_tmem_addr + Int32(fragment_idx * fragment_regs),
                            Float32,
                        ),
                        scores.data_ptr().load(count=fragment_regs, alignment=4),
                        offset=cfg.tile_size_kv // 2,
                    )
                elif cutlass.const_expr(cfg.use_sage_attention):
                    self._fold_sage_fragment_max(
                        max_chains,
                        loaded,
                        fragment_scales=fragment_scales,
                        chain_base=fragment_idx * cfg.sage_k_groups_per_fragment,
                        groups=cfg.sage_k_groups_per_fragment,
                    )
                    self.sage_k_scales.advance(scales_view)
                else:
                    for score_idx in cutlass.range_constexpr(fragment_regs):
                        chain_idx: Constexpr[int] = score_idx % 4
                        max_chains[chain_idx] = cute.math.max(
                            max_chains[chain_idx],
                            Float32(loaded[score_idx]),
                            ftz=True,
                        )
            if cutlass.const_expr(exact_dequantized):
                prims.tcgen05_wait(kind=prims.Tcgen05Wait.STORE)
                cute.arch.fence_view_async_tmem_store()
        else:
            if cutlass.const_expr(not use_sparse):
                lane_idx = Int32(task_cache[_TASK_CACHE_LANE_IDX])
                col_base = _keeps_col_base(cfg, lane_idx, num_fragments * fragment_regs)
                for fragment_idx in cutlass.range_constexpr(num_fragments):
                    fragment_token_base = tile_offset_k + _keeps_score_col(
                        cfg,
                        warp_grp_thread_idx,
                        fragment_idx * fragment_regs,
                        col_base,
                    )
                    keep_words[fragment_idx] = _dense_fragment_keep_word(
                        rows_are_active,
                        visible_start - fragment_token_base,
                        visible_end - fragment_token_base,
                        fragment_regs=fragment_regs,
                    )
            # The masked pass is one rolled loop per route kind. A mixed
            # geometry selects the loop per tile: a per-fragment selection
            # would be if-converted and run both folds on every tile.
            if cutlass.const_expr(mixed_scales):
                if route_is_proxy:
                    self._mask_score_fragments(
                        score_tmem_addr,
                        keep_words,
                        max_chains,
                        proxy_kind=True,
                        scales_view=summary_view,
                        may_hold_tail=shifts_tail,
                        tail_fragment_mask=tail_fragment_mask,
                        tail_shift=tail_shift,
                        tail_lane=tail_lane,
                    )
                else:
                    self._mask_score_fragments(
                        score_tmem_addr,
                        keep_words,
                        max_chains,
                        proxy_kind=False,
                        scales_view=scales_view,
                        may_hold_tail=shifts_tail,
                        tail_fragment_mask=tail_fragment_mask,
                        tail_shift=tail_shift,
                        tail_lane=tail_lane,
                    )
            else:
                self._mask_score_fragments(
                    score_tmem_addr,
                    keep_words,
                    max_chains,
                    proxy_kind=False,
                    scales_view=scales_view,
                    may_hold_tail=shifts_tail,
                    tail_fragment_mask=tail_fragment_mask,
                    tail_shift=tail_shift,
                    tail_lane=tail_lane,
                )
            prims.tcgen05_wait(kind=prims.Tcgen05Wait.STORE)
            cute.arch.fence_view_async_tmem_store()

        tile_max = cute.math.max(
            cute.math.max(max_chains[0], max_chains[1], ftz=True),
            cute.math.max(max_chains[2], max_chains[3], ftz=True),
            ftz=True,
        )
        if cutlass.const_expr(cfg.use_sage_attention):
            # The chains hold ``gmax * sfK``; ``sfQ`` applies once here. Only
            # the ``-inf`` identity maps to the finite sentinel: a real
            # pre-``sfQ`` maximum of ``-FLT_MAX`` is an ordinary score.
            if tile_max == Float32(float("-inf")):
                tile_max = _neg_max_f32()
            else:
                tile_max = tile_max * sage_q_scale
        if cutlass.const_expr(use_sparse and cfg.use_block_sparse_proxy_routes):
            # The mass enters before the anchor, so the P range is unchanged; a
            # fully masked route's sentinel absorbs the shift in FP32.
            tile_max = tile_max + proxy_max_shift
        old_max = new_max_arr[0]
        new_max = self._softmax_anchor(old_max, tile_max)
        old_max_arr[0] = old_max
        new_max_arr[0] = new_max
        return old_max_arr, sum_arr, new_max_arr, s_arr

    @cute.jit
    def _mask_score_fragments(
        self,
        score_tmem_addr: Int32,
        keep_words: cutlass.Array,
        max_chains: cutlass.Array,
        *,
        proxy_kind: Constexpr[bool],
        scales_view,
        may_hold_tail: Constexpr[bool],
        tail_fragment_mask: Int32,
        tail_shift: Float32,
        tail_lane: Constexpr[int],
    ) -> None:
        """Mask, fold and write back every fragment with one route kind's scales.

        ``scales_view`` is the opened view of the resource ``proxy_kind``
        selects; the resource is read from ``self`` because a local would be
        flattened as a loop-carried value. The keep words rotate down one
        entry per fragment, so the body reads ``keep_words[0]``.

        Cleared keep bits become ``-inf`` with Sage, else ``-FLT_MAX``. The
        fragment holding a proxy route's ragged final summary adds the mass
        shortfall to that score; Sage divides the shift (already divided by
        ``sfQ``) by the tail group's ``sfK`` with an approximate reciprocal,
        ample for a shift far below the score resolution. A route kind whose
        scores are dequantized writes the dequantized scores back.
        """
        cfg = self.cfg
        num_fragments = cfg.num_softmax_score_fragments
        fragment_regs = cfg.softmax_score_fragment_regs
        groups: Constexpr[int] = cfg.sage_k_groups_per_fragment_for(proxy=proxy_kind)
        dequantize_scores: Constexpr[bool] = cfg.sage_scores_dequantized_for(
            proxy=proxy_kind
        )
        for fragment in cutlass.range(num_fragments, unroll=1):
            fragment_scales = None
            if cutlass.const_expr(cfg.use_sage_attention and proxy_kind):
                fragment_scales = self.sage_summary_k_scales.fragment(
                    scales_view, Int32(fragment)
                )
            elif cutlass.const_expr(cfg.use_sage_attention):
                fragment_scales = self.sage_k_scales.fragment(
                    scales_view, Int32(fragment)
                )
            keep_word = Uint32(keep_words[0])
            fragment_addr = score_tmem_addr + Int32(fragment) * Int32(fragment_regs)
            loaded = _keeps_tcgen05_ld(
                cfg,
                prims.make_tmem_ptr(fragment_addr, Float32),
                num=fragment_regs,
                offset=cfg.tile_size_kv // 2,
            )
            prims.tcgen05_wait(kind=prims.Tcgen05Wait.LOAD)
            masked_scores = cutlass.Array(
                Float32, fragment_regs, space=cutlass.AddressSpace.rmem
            )
            for score_idx in cutlass.range_constexpr(fragment_regs):
                score = Float32(loaded[score_idx])
                score_is_kept = ((keep_word >> Int32(score_idx)) & Uint32(1)) != Uint32(
                    0
                )
                if not score_is_kept:
                    if cutlass.const_expr(cfg.use_sage_attention):
                        score = Float32(float("-inf"))
                    else:
                        score = _neg_max_f32()
                if cutlass.const_expr(may_hold_tail and score_idx == tail_lane):
                    if ((tail_fragment_mask >> Int32(fragment)) & Int32(1)) != Int32(0):
                        lane_shift = tail_shift
                        if cutlass.const_expr(cfg.use_sage_attention):
                            tail_group: Constexpr[int] = tail_lane // (
                                fragment_regs // groups
                            )
                            lane_shift = tail_shift * cute.math.rcp(
                                Float32(fragment_scales[tail_group]), approx=True
                            )
                        score = score + lane_shift
                masked_scores[score_idx] = score
                if cutlass.const_expr(not cfg.use_sage_attention):
                    chain_idx: Constexpr[int] = score_idx % 4
                    max_chains[chain_idx] = cute.math.max(
                        max_chains[chain_idx], score, ftz=True
                    )
            if cutlass.const_expr(dequantize_scores):
                self._dequantize_fragment_scores(masked_scores, fragment_scales, groups)
                for score_idx in cutlass.range_constexpr(fragment_regs):
                    chain_idx: Constexpr[int] = score_idx % 4
                    max_chains[chain_idx] = cute.math.max(
                        max_chains[chain_idx],
                        Float32(masked_scores[score_idx]),
                        ftz=True,
                    )
            elif cutlass.const_expr(cfg.use_sage_attention):
                self._fold_sage_fragment_max(
                    max_chains,
                    masked_scores,
                    fragment_scales=fragment_scales,
                    chain_base=0,
                    groups=groups,
                )
            _keeps_tcgen05_st(
                cfg,
                prims.make_tmem_ptr(fragment_addr, Float32),
                masked_scores.data_ptr().load(count=fragment_regs, alignment=4),
                offset=cfg.tile_size_kv // 2,
            )
            if cutlass.const_expr(cfg.use_sage_attention and proxy_kind):
                self.sage_summary_k_scales.advance(scales_view)
            elif cutlass.const_expr(cfg.use_sage_attention):
                self.sage_k_scales.advance(scales_view)
            for entry in cutlass.range_constexpr(num_fragments - 1):
                keep_words[entry] = Uint32(keep_words[entry + 1])

    @cute.jit
    def _dequantize_fragment_scores(
        self,
        scores: cutlass.Array,
        fragment_scales: cutlass.Array,
        groups: Constexpr[int],
    ) -> None:
        """Replace one fragment's quantized scores by ``(s - bias) * sfK`` in place.

        Biased INT32 scores fold ``-bias * sfK`` into the same packed FMA. A
        masked ``-inf`` score stays ``-inf`` under any positive scale.
        """
        cfg = self.cfg
        fragment_regs = cfg.softmax_score_fragment_regs
        group_regs = fragment_regs // groups
        group_addends = None
        if cutlass.const_expr(cfg.uses_int32_scores):
            group_addends = cutlass.Array(
                Float32, groups, space=cutlass.AddressSpace.rmem
            )
            neg_bias = Float32(-INT32_SCORE_BIAS)
            for group_base in cutlass.range_constexpr(0, groups - groups % 2, 2):
                group_addends[group_base], group_addends[group_base + 1] = fmul2(
                    (neg_bias, neg_bias),
                    (
                        Float32(fragment_scales[group_base]),
                        Float32(fragment_scales[group_base + 1]),
                    ),
                )
            if cutlass.const_expr(groups % 2 == 1):
                group_addends[groups - 1] = neg_bias * Float32(
                    fragment_scales[groups - 1]
                )
        for pair_base in cutlass.range_constexpr(0, fragment_regs, 2):
            group0: Constexpr[int] = pair_base // group_regs
            group1: Constexpr[int] = (pair_base + 1) // group_regs
            pair = (Float32(scores[pair_base]), Float32(scores[pair_base + 1]))
            scales = (
                Float32(fragment_scales[group0]),
                Float32(fragment_scales[group1]),
            )
            if cutlass.const_expr(cfg.uses_int32_scores):
                pair = cute.arch.fma_packed_f32x2(
                    pair,
                    scales,
                    (Float32(group_addends[group0]), Float32(group_addends[group1])),
                )
            else:
                pair = fmul2(pair, scales)
            scores[pair_base] = pair[0]
            scores[pair_base + 1] = pair[1]

    @cute.jit
    def _fold_sage_fragment_max(
        self,
        max_chains: cutlass.Array,
        scores,
        *,
        fragment_scales: cutlass.Array,
        chain_base: Constexpr[int],
        groups: Constexpr[int],
    ) -> None:
        """Fold one fragment's dequantized group maxima into the max chains.

        Each scale group is reduced on the quantized scores, then
        ``(group_max - bias) * sfK_g`` is folded; the caller applies ``sfQ``
        to the tile maximum. Groups of four or more scores reduce over four
        chains; masked ``-inf`` scores need no special case. Group maxima
        leave in pairs through a packed bias add (exact on the biased scores'
        unit spacing) and a packed scale multiply. Group ``g`` folds into
        chain ``(chain_base + g) % 4``.
        """
        cfg = self.cfg
        group_regs = cfg.softmax_score_fragment_regs // groups
        width: Constexpr[int] = 1 if groups == 1 else 2
        assert groups % width == 0
        for group_base in cutlass.range_constexpr(0, groups, width):
            maxima: tuple = ()
            for elem in cutlass.range_constexpr(width):
                first: Constexpr[int] = (group_base + elem) * group_regs
                if cutlass.const_expr(group_regs < 4):
                    group_max = Float32(scores[first])
                    for score_elem in cutlass.range_constexpr(1, group_regs):
                        group_max = cute.math.max(
                            group_max, Float32(scores[first + score_elem]), ftz=True
                        )
                else:
                    chains = cutlass.Array(Float32, 4, space=cutlass.AddressSpace.rmem)
                    for chain_idx in cutlass.range_constexpr(4):
                        chains[chain_idx] = Float32(scores[first + chain_idx])
                    for score_elem in cutlass.range_constexpr(4, group_regs):
                        chain_idx: Constexpr[int] = score_elem % 4
                        chains[chain_idx] = cute.math.max(
                            chains[chain_idx],
                            Float32(scores[first + score_elem]),
                            ftz=True,
                        )
                    group_max = cute.math.max(
                        cute.math.max(chains[0], chains[1], ftz=True),
                        cute.math.max(chains[2], chains[3], ftz=True),
                        ftz=True,
                    )
                maxima += (group_max,)
            if cutlass.const_expr(width == 1):
                scaled_one = Float32(maxima[0])
                if cutlass.const_expr(cfg.uses_int32_scores):
                    scaled_one = scaled_one - Float32(INT32_SCORE_BIAS)
                scaled = (scaled_one * Float32(fragment_scales[group_base]),)
            else:
                scaled = maxima
                if cutlass.const_expr(cfg.uses_int32_scores):
                    scaled = cute.arch.add_packed_f32x2(
                        scaled, (Float32(-INT32_SCORE_BIAS), Float32(-INT32_SCORE_BIAS))
                    )
                scaled = fmul2(
                    scaled,
                    (
                        Float32(fragment_scales[group_base]),
                        Float32(fragment_scales[group_base + 1]),
                    ),
                )
            for elem in cutlass.range_constexpr(width):
                scaled_max = Float32(scaled[elem])
                fold_chain: Constexpr[int] = (chain_base + group_base + elem) % 4
                max_chains[fold_chain] = cute.math.max(
                    max_chains[fold_chain], scaled_max, ftz=True
                )

    @consumer_work(returns=("old_max_arr", "sum_arr", "new_max_arr", "s_arr"))
    @cute.jit
    def compute_sage_block_sparse_softmax_loop(
        self,
        stage_info: StageInfo,
        *,
        old_max_arr: cutlass.Array,
        sum_arr: cutlass.Array,
        new_max_arr: cutlass.Array,
        s_arr: cutlass.Array,
        sparse_origin0: Int32,
        sparse_origin1: Int32,
        sparse_route_flags: Int32,
        sparse_token_word0: Uint32,
        sparse_token_word1: Uint32,
        sparse_token_word2: Uint32,
        sparse_token_word3: Uint32,
        sage_q_scale: Float32,
        sage_scale_arr: cutlass.Array,
        sage_summary_scale_arr: cutlass.Array,
    ) -> tuple[object, object, object, object]:
        """Consume one routed S payload with per-group Sage scales."""

        assert self.cfg.use_block_sparse and self.cfg.use_sage_attention
        return self._compute_softmax_loop_keeps_fragments(
            stage_info,
            old_max_arr=old_max_arr,
            sum_arr=sum_arr,
            new_max_arr=new_max_arr,
            s_arr=s_arr,
            use_sparse=True,
            sparse_origin0=sparse_origin0,
            sparse_origin1=sparse_origin1,
            sparse_route_flags=sparse_route_flags,
            sparse_token_word0=sparse_token_word0,
            sparse_token_word1=sparse_token_word1,
            sparse_token_word2=sparse_token_word2,
            sparse_token_word3=sparse_token_word3,
            sage_q_scale=sage_q_scale,
            sage_scale_arr=sage_scale_arr,
            sage_summary_scale_arr=sage_summary_scale_arr,
        )

    @consumer_work(returns=("old_max_arr", "sum_arr", "new_max_arr", "s_arr"))
    @cute.jit
    def compute_block_sparse_softmax_loop(
        self,
        stage_info: StageInfo,
        *,
        old_max_arr: cutlass.Array,
        sum_arr: cutlass.Array,
        new_max_arr: cutlass.Array,
        s_arr: cutlass.Array,
        sparse_origin0: Int32,
        sparse_origin1: Int32,
        sparse_route_flags: Int32,
        sparse_token_word0: Uint32,
        sparse_token_word1: Uint32,
        sparse_token_word2: Uint32,
        sparse_token_word3: Uint32,
    ) -> tuple[object, object, object, object]:
        """Consume S plus one explicitly routed, register-resident payload."""

        assert self.cfg.use_block_sparse
        if cutlass.const_expr(self.cfg.use_keeps_mma_ab):
            # Every block-sparse Keeps profile streams K32 fragments.
            return self._compute_softmax_loop_keeps_fragments(
                stage_info,
                old_max_arr=old_max_arr,
                sum_arr=sum_arr,
                new_max_arr=new_max_arr,
                s_arr=s_arr,
                use_sparse=True,
                sparse_origin0=sparse_origin0,
                sparse_origin1=sparse_origin1,
                sparse_route_flags=sparse_route_flags,
                sparse_token_word0=sparse_token_word0,
                sparse_token_word1=sparse_token_word1,
                sparse_token_word2=sparse_token_word2,
                sparse_token_word3=sparse_token_word3,
            )
        # SWAP reuses the Keeps seven-slot task ABI: all four origins remain
        # logical KV atom bases, but origin2 occupies the flags slot and
        # origin3 is bit-preserved in word0. Word1 carries the logical K32 token
        # mask and word2 optionally carries the prepared route-full summary.
        return self._compute_softmax_loop_swaps(
            stage_info,
            old_max_arr=old_max_arr,
            sum_arr=sum_arr,
            new_max_arr=new_max_arr,
            s_arr=s_arr,
            sparse_origin0=Int32(sparse_origin0),
            sparse_origin1=Int32(sparse_origin1),
            sparse_origin2=Int32(sparse_route_flags),
            sparse_origin3=sparse_token_word0.bitcast(Int32),
            sparse_token_word=Uint32(sparse_token_word1),
            sparse_route_flags=Uint32(sparse_token_word2),
            use_sparse=True,
        )
