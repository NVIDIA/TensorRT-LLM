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

"""``SmemPResource`` — P operand staging for BMM2.

Producer (Softmax): converts S in registers to P and publishes per-lane local
sums back through ``TmemSResource``. Keeps Q64/Q128 overlays P on consumed S
columns in TMEM; Swaps retains the SMEM operand layout. Consumer (MmaTask)
publishes the corresponding TMEM address or SMEM descriptor.
"""

from dataclasses import dataclass
from typing import ClassVar

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, Int64, Uint32
from cutlass.experimental import primitives as prims

from cutlass.experimental.task_scheduling.memory import (
    ResourceContext,
    SmemAllocation,
    TmemAllocation,
)
from cutlass.experimental.task_scheduling.resources import (
    WorkAttr,
    StageInfo,
    TaskLocalVariable,
    consumer_work,
    producer_work,
)

from ..fmha_decode_config import FmhaDecodeConfig
from ..fmha_decode_constants import (
    FP8_P_QUANT_LOG2_SCALE,
    INT32_SCORE_BIAS,
)
from ...placeholder_helpers import _placeholder_smem_array
from .helpers_common import (
    _route_is_proxy,
    Constexpr,
    DecodeGenResourceBase,
    ResourceVars,
    fadd2,
    ffma2,
    _TASK_CACHE_LANE_IDX,
    _TASK_CACHE_WARP_GRP_THREAD_IDX,
    _TASK_CACHE_WARP_IDX,
    _decode_gen_task_cache,
    _is_last_loop_iteration,
    _keeps_col_base,
    _keeps_row_idx,
    _keeps_tcgen05_ld,
    _keeps_tcgen05_st,
    _named_barrier_arrive,
    _neg_max_f32,
    _pack_float2_to_bf16,
    _pack_float2_to_fp16,
    _wait_for_mbarrier_phase,
)
from .helpers_output import (
    _keeps_p_smem_block_offset_bytes,
    _p_stsm_smem_offset_bytes,
    _store_transposed_smem8b,
    _store_transposed_smem8b_x2,
    _store_transposed_smem8b_x4,
)
from .helpers_softmax import (
    _compute_fp8_p_regs_and_local_sums,
    _compute_p_values_and_local_sums,
    _ex2_emulation_packed_f32x2,
    _pack_float4_to_fp8_e4m3,
    _pack_float4_to_fp8_e4m3_inline,
)
from .sage_scales import SageKScalesResource
from .tmem_s import TmemSResource

# Tunable: number of score pairs per streamed fragment whose exponentials run
# as FMA polynomials instead of MUFU. The MUFU issue rate bounds the fragment
# otherwise, while the FMA pipe is nearly idle in the softmax warps. Larger
# shares grow the fragment body and the softmax warps become instruction-fetch
# bound again, so one quarter of the 16 pairs is the measured optimum.
KV_TILE_256_EX2_EMULATED_PAIRS = 4


def _pair_uses_ex2_emulation(pair_idx: int, pairs_per_fragment: int) -> bool:
    """Spread the emulated pairs evenly across a fragment's score pairs."""
    count = KV_TILE_256_EX2_EMULATED_PAIRS
    pairs = pairs_per_fragment
    return ((pair_idx + 1) * count) // pairs != (pair_idx * count) // pairs


@dataclass(kw_only=True)
class SmemPResource(DecodeGenResourceBase):
    """P operand resource consumed by BMM2.

    Softmax producers convert S to P, store it in the profile's TMEM or SMEM
    layout, and publish local sums back to TmemS. Most profiles use the generic
    full/empty P pipeline. Streamed profiles instead publish four independently ready
    K32 TMEM fragments; BMM2 consumes those fragments in order, while the
    matching TmemO full barrier prevents the next QK from overwriting aliased P.
    """

    _task_local_specs: ClassVar[tuple[tuple, ...]] = (
        (
            "p_desc_0_slot",
            prims.Tcgen05SmemDesc,
            prims.Tcgen05SmemDesc(0),
            "P descriptor for VP MMA call 0.",
        ),
        (
            "p_desc_1_slot",
            prims.Tcgen05SmemDesc,
            prims.Tcgen05SmemDesc(0),
            "P descriptor for VP MMA call 1.",
        ),
        (
            "p_tmem_addr_0_slot",
            Int32,
            Int32(0),
            "TMEM P address for VP MMA call 0.",
        ),
        (
            "p_tmem_addr_1_slot",
            Int32,
            Int32(0),
            "TMEM P address for VP MMA call 1.",
        ),
    )
    inst_id: Constexpr[int] = 0
    cfg: Constexpr[FmhaDecodeConfig] = None
    scale_softmax_log2: Float32 = None
    use_variable_seqlens_kv: Constexpr[bool] = False
    tmem_s_ref: Constexpr[TmemSResource] = None
    tmem_o_ref: Constexpr[object] = None
    # The instance's ``sfK`` resources, shared with the S resource; proxy
    # routes read ``sage_summary_k_scales``, everything else ``sage_k_scales``.
    sage_k_scales: SageKScalesResource | None = None
    sage_summary_k_scales: SageKScalesResource | None = None
    _alloc: Constexpr[SmemAllocation | None] = None
    _fragment_ready_alloc: Constexpr[SmemAllocation | None] = None
    _tmem_alloc: Constexpr[TmemAllocation | None] = None
    _tmem_base_addr: Int32 = None
    _smem_base_p: cutlass.Array = None
    _smem_base_p_i32: cutlass.Array = None
    _fragment_ready: cutlass.Array = None
    p_desc_0_slot: Constexpr[TaskLocalVariable] = TaskLocalVariable.uninitialized()
    p_desc_1_slot: Constexpr[TaskLocalVariable] = TaskLocalVariable.uninitialized()
    p_tmem_addr_0_slot: Constexpr[TaskLocalVariable] = TaskLocalVariable.uninitialized()
    p_tmem_addr_1_slot: Constexpr[TaskLocalVariable] = TaskLocalVariable.uninitialized()

    def _init_placeholder_state(self) -> None:
        """Create placeholder P storage state."""
        self._tmem_base_addr = Int32(0)
        self._smem_base_p = _placeholder_smem_array(
            self.cfg.pv_dtype,
            self.cfg.smem_p_tile_bytes // self.cfg.pv_dtype_bytes,
        )
        self._smem_base_p_i32 = _placeholder_smem_array(
            Int32, self.cfg.smem_p_tile_bytes // 4
        )
        self._fragment_ready = _placeholder_smem_array(
            Int64, self.cfg.num_softmax_score_fragments
        )

    def get_smem_requirements(self) -> list[SmemAllocation]:
        """Allocate P storage or the streamed fragment-ready barriers."""
        if self.cfg.streams_tmem_p_fragments:
            if self._fragment_ready_alloc is None:
                self._fragment_ready_alloc = SmemAllocation(
                    name=f"{self.name}_fragmentReady",
                    size_bytes=self.cfg.num_softmax_score_fragments * 8,
                    alignment=16,
                )
            return [self._fragment_ready_alloc]
        if self.cfg.uses_tmem_p:
            return []
        if self._alloc is None:
            self._alloc = SmemAllocation(
                name=f"{self.name}",
                size_bytes=self.cfg.smem_p_tile_bytes,
                alignment=self.cfg.stensor_align,
            )
        return [self._alloc]

    @cute.jit
    def _bind_fragment_ready(self, context: ResourceContext | None = None) -> None:
        """Bind the one-way streamed P-ready barriers from the SMEM context."""
        if cutlass.const_expr(
            self.cfg.streams_tmem_p_fragments
            and context is not None
            and context.smem_base is not None
            and self._fragment_ready_alloc is not None
        ):
            self._fragment_ready = cutlass.Array(
                context.smem_base.data_ptr() + self._fragment_ready_alloc.offset,
                dtype=Int64,
                shape=(self.cfg.num_softmax_score_fragments,),
                addrspace=3,
            )

    @cute.jit
    def create_function_variables(
        self, context: ResourceContext | None = None
    ) -> ResourceVars:
        """Bind and initialize the streamed per-fragment ready barriers."""
        self._bind_fragment_ready(context)
        if cutlass.const_expr(self.cfg.streams_tmem_p_fragments):
            tidx, _, _ = cute.arch.thread_idx()
            producer_warps = (
                self.cfg.softmax0_num_warps
                if self.inst_id == 0
                else self.cfg.softmax1_num_warps
            )
            if tidx == Int32(0):
                for fragment_idx in cutlass.range_constexpr(
                    self.cfg.num_softmax_score_fragments
                ):
                    prims.mbarrier_init(
                        self._fragment_ready.data_ptr() + fragment_idx,
                        producer_warps,
                    )
        return {}

    @cute.jit
    def initialize_runtime_state_internal(
        self,
        context: ResourceContext | None = None,
        captured_schedule: bool = False,
    ) -> None:
        """Initialize generic resource state and bind fragment barriers."""
        super().initialize_runtime_state_internal(context, captured_schedule)
        self._bind_fragment_ready(context)

    def get_tmem_requirements(self) -> list[TmemAllocation]:
        """Allocate the TMEM columns occupied by the Keeps P operand."""
        if not self.cfg.uses_tmem_p:
            return []
        if self._tmem_alloc is None:
            self._tmem_alloc = TmemAllocation(
                name=f"{self.name}_tmem",
                num_columns=self.cfg.tmem_p_cols_per_inst,
            )
        return [self._tmem_alloc]

    @cute.jit
    def _create_initial_task_locals(
        self, context: ResourceContext | None = None
    ) -> ResourceVars:
        """Bind P storage and initialize operand task locals."""
        self._bind_fragment_ready(context)
        if cutlass.const_expr(
            not self.cfg.uses_tmem_p
            and context is not None
            and context.smem_base is not None
        ):
            # P is materialized in SMEM because BMM2 consumes it as a tcgen05
            # SMEM operand.
            smem_base_ptr = context.smem_base.data_ptr() + self._alloc.offset
            self._smem_base_p = cutlass.Array(
                smem_base_ptr,
                dtype=self.cfg.pv_dtype,
                shape=(self.cfg.smem_p_tile_bytes // self.cfg.pv_dtype_bytes,),
                addrspace=3,
            )
            self._smem_base_p_i32 = cutlass.Array(
                smem_base_ptr,
                dtype=Int32,
                shape=(self.cfg.smem_p_tile_bytes // 4,),
                addrspace=3,
            )
        if cutlass.const_expr(
            self.cfg.uses_tmem_p
            and context is not None
            and context.tmem_ptr_i32 is not None
        ):
            self._tmem_base_addr = context.tmem_ptr_i32.load()
        return {
            "p_desc_0": cutlass.Int64(0),
            "p_desc_1": cutlass.Int64(0),
            "p_tmem_addr_0": Int32(0),
            "p_tmem_addr_1": Int32(0),
        }

    @cute.jit
    def _create_work_tile_task_locals(
        self, context: ResourceContext | None = None
    ) -> ResourceVars:
        """Provide P operand slots for one work tile."""
        _ = context
        return {
            "p_desc_0": cutlass.Int64(0),
            "p_desc_1": cutlass.Int64(0),
            "p_tmem_addr_0": Int32(0),
            "p_tmem_addr_1": Int32(0),
        }

    @producer_work(work_attrs=WorkAttr.AUXILIARY)
    @cute.jit
    def init_compute_state(self, stage_info: StageInfo) -> None:
        """Initialize producer-side P registers and local sums."""
        # ProdAuxWork: bind the SMEM P tile and reset producer-local P/sum
        # state before the softmax producer starts writing this work tile.
        self._create_initial_task_locals(stage_info.context)

    @consumer_work(work_attrs=WorkAttr.AUXILIARY)
    @cute.jit
    def init_descriptor_state(self, stage_info: StageInfo) -> None:
        """Initialize consumer-side P operand placeholders."""
        # ConsAuxWork: mirror the P storage binding on the BMM2 side so operand
        # work can publish a valid descriptor or TMEM address for this tile.
        self._create_initial_task_locals(stage_info.context)

    @cute.jit
    def _exponent_addend(
        self,
        new_max: Float32,
        route_is_proxy: cutlass.Boolean,
        *,
        guards_masked_rows: Constexpr[bool] = True,
    ) -> Float32:
        """Return ``addend`` of one row's P exponent ``exp2(c * s + addend)``.

        - ``-c * max`` anchors the row; a fully masked row (``-FLT_MAX``
          maximum) anchors on zero so its scores exponentiate to zero, not NaN.
          Callers whose masked rows are skipped or discarded downstream pass
          ``guards_masked_rows=False``.
        - Byte-wide P adds ``log2(448)``, its static FP8 quantization scale.
        - A proxy route adds ``log2`` of its block mass, as the max pass does,
          so each summary probability carries the tokens it stands for.
        """
        cfg = self.cfg
        safe_new_max = new_max
        if cutlass.const_expr(guards_masked_rows):
            if safe_new_max == _neg_max_f32():
                safe_new_max = Float32(0.0)
        # Not ``cute.math.fma``: the fused form makes ptxas spill in the callers.
        addend = Float32(-self.scale_softmax_log2 * safe_new_max)
        if cutlass.const_expr(self.cfg.use_fp8_pv):
            addend += Float32(FP8_P_QUANT_LOG2_SCALE)
        if cutlass.const_expr(cfg.use_block_sparse_proxy_routes):
            if route_is_proxy:
                addend += Float32(cfg.proxy_log2_block_mass)
        return addend

    @producer_work
    @cute.jit
    def compute_p_fragments(
        self,
        stage_info: StageInfo,
        *,
        new_max_arr: cutlass.Array,
    ) -> None:
        """Stream every ordinary K32 fragment from one rolled loop."""
        self._compute_p_fragments_impl(
            stage_info,
            new_max_arr=new_max_arr,
            route_is_proxy=cutlass.Boolean(False),
        )

    @producer_work
    @cute.jit
    def compute_sage_p_fragments(
        self,
        stage_info: StageInfo,
        *,
        new_max_arr: cutlass.Array,
        sage_q_scale: Float32,
        sage_scale_arr: cutlass.Array,
        sage_summary_scale_arr: cutlass.Array,
    ) -> None:
        """Stream every K32 fragment with per-group Sage dequantization scales."""
        assert self.cfg.use_sage_attention
        self._compute_p_fragments_impl(
            stage_info,
            new_max_arr=new_max_arr,
            route_is_proxy=cutlass.Boolean(False),
            sage_q_scale=sage_q_scale,
            sage_scale_arr=sage_scale_arr,
            sage_summary_scale_arr=sage_summary_scale_arr,
        )

    @producer_work
    @cute.jit
    def compute_proxy_route_p_fragments(
        self,
        stage_info: StageInfo,
        *,
        new_max_arr: cutlass.Array,
        route_flags: Int32,
    ) -> None:
        """Stream every exact or proxy K32 fragment from one rolled loop."""
        assert self.cfg.use_block_sparse_proxy_routes
        self._compute_p_fragments_impl(
            stage_info,
            new_max_arr=new_max_arr,
            route_is_proxy=_route_is_proxy(route_flags),
        )

    @producer_work
    @cute.jit
    def compute_sage_proxy_route_p_fragments(
        self,
        stage_info: StageInfo,
        *,
        new_max_arr: cutlass.Array,
        route_flags: Int32,
        sage_q_scale: Float32,
        sage_scale_arr: cutlass.Array,
        sage_summary_scale_arr: cutlass.Array,
    ) -> None:
        """Stream exact or proxy K32 fragments with per-group Sage scales."""
        assert self.cfg.use_block_sparse_proxy_routes and self.cfg.use_sage_attention
        self._compute_p_fragments_impl(
            stage_info,
            new_max_arr=new_max_arr,
            route_is_proxy=_route_is_proxy(route_flags),
            sage_q_scale=sage_q_scale,
            sage_scale_arr=sage_scale_arr,
            sage_summary_scale_arr=sage_summary_scale_arr,
        )

    @cute.jit
    def _compute_p_fragments_impl(
        self,
        stage_info: StageInfo,
        *,
        new_max_arr: cutlass.Array,
        route_is_proxy: cutlass.Boolean,
        sage_q_scale: Float32 | None = None,
        sage_scale_arr: cutlass.Array | None = None,
        sage_summary_scale_arr: cutlass.Array | None = None,
    ) -> None:
        """Reload, exponentiate, and publish all K32 fragments in a rolled loop.

        The fragment index is a runtime loop variable, so the exponentiation
        body exists once in the instruction stream and only the TMEM column
        offset and the fragment barrier depend on it. Unrolling the fragments
        would replicate that body for every fragment and both softmax
        instances and leave the softmax warps instruction-fetch bound. The max
        pass has already written masked (and mass-shifted) scores back to
        TMEM, so the reload needs no mask or route logic beyond the proxy
        addend. Sage attention changes only the per-group multipliers
        ``c * sfQ * sfK_g``, biased INT32 scores only the per-group addends.
        """
        cfg = self.cfg
        assert cfg.streams_tmem_p_fragments
        assert self._tmem_alloc.offset == self.tmem_s_ref._alloc.offset
        # One FP32 score per column; two 16-bit or four FP8 probabilities per
        # packed column.
        fragment_regs = cfg.softmax_score_fragment_regs
        fragment_cols = cfg.fragment_p_packed_cols

        exponent_addend = self._exponent_addend(new_max_arr[0], route_is_proxy)
        tmem_base = self._tmem_base_addr + Int32(self._tmem_alloc.offset)
        tidx, _, _ = cute.arch.thread_idx()
        publishes_fragment = (tidx & Int32(31)) == Int32(0)

        # ``c * sfQ`` is one factor per lane and tile: the only multiplier of a
        # route kind whose max pass returns dequantized scores (no bias),
        # otherwise folded into the kind's raw ``sfK`` words.
        exact_dequantized: Constexpr[bool] = cfg.sage_scores_dequantized_for(
            proxy=False
        )
        summary_dequantized: Constexpr[bool] = cfg.sage_scores_dequantized_for(
            proxy=True
        )
        exact_scales_in_smem: Constexpr[bool] = cfg.sage_k_scales_in_smem_for(
            cfg.sage_k_groups_per_fragment
        )
        # A mixed geometry selects the resource per fragment on the CTA-uniform
        # route kind, unless proxy scores are dequantized and the exact words
        # sit in registers: the exact resource then serves both kinds, routing
        # ones on a proxy tile.
        unified_scales: Constexpr[bool] = (
            cfg.sage_mixed_k_geometry
            and summary_dequantized
            and not exact_scales_in_smem
        )
        mixed_scales: Constexpr[bool] = cfg.sage_mixed_k_geometry and not unified_scales
        exp_scale = None
        scales_view = None
        summary_view = None
        score_bias = None
        row_multipliers = None
        row_addends = None
        if cutlass.const_expr(cfg.use_sage_attention):
            exp_scale = self.scale_softmax_log2 * sage_q_scale
            if cutlass.const_expr(exact_dequantized or summary_dequantized):
                row_multipliers = cutlass.Array(
                    Float32, 1, space=cutlass.AddressSpace.rmem
                )
                row_addends = cutlass.Array(Float32, 1, space=cutlass.AddressSpace.rmem)
                row_multipliers[0] = exp_scale
                row_addends[0] = exponent_addend
            if cutlass.const_expr(not exact_dequantized):
                scales_view = self.sage_k_scales.open(
                    stage_info, sage_scale_arr, exp_scale
                )
            if cutlass.const_expr(mixed_scales and not summary_dequantized):
                summary_view = self.sage_summary_k_scales.open(
                    stage_info, sage_summary_scale_arr, exp_scale
                )
            if cutlass.const_expr(unified_scales and cfg.uses_int32_scores):
                score_bias = Float32(INT32_SCORE_BIAS)
                if route_is_proxy:
                    score_bias = Float32(0.0)

        # A fragment is scaled before its first exponential, and the next
        # fragment's TMEM load issues once the scale FFMAs have consumed the
        # scores, so it reuses their registers and hides behind the
        # exponentials. The running sum stays a packed pair until the loop ends.
        num_fragments = cfg.num_softmax_score_fragments
        last_fragment = Int32(num_fragments - 1)
        total_sum_pair = (Float32(0.0), Float32(0.0))
        s_arr = cutlass.Array(Float32, fragment_regs, space=cutlass.AddressSpace.rmem)
        pending_scores = cutlass.Array(
            Float32, fragment_regs, space=cutlass.AddressSpace.rmem
        )
        self._load_score_fragment(tmem_base, Int32(0), pending_scores)
        for fragment_idx in cutlass.range(cfg.num_softmax_score_fragments, unroll=1):
            fragment = Int32(fragment_idx)
            prims.tcgen05_wait(kind=prims.Tcgen05Wait.LOAD)
            for score_idx in cutlass.range_constexpr(fragment_regs):
                s_arr[score_idx] = pending_scores[score_idx]

            if cutlass.const_expr(mixed_scales):
                # Each route kind scales the fragment in place with its own
                # resource, so only the scores cross the branch.
                if route_is_proxy:
                    if cutlass.const_expr(summary_dequantized):
                        self._scale_fragment_pairs(
                            s_arr, row_addends, row_multipliers, groups=1
                        )
                    else:
                        self._scale_fragment_with(
                            self.sage_summary_k_scales,
                            summary_view,
                            fragment,
                            s_arr,
                            exponent_addend,
                        )
                else:
                    if cutlass.const_expr(exact_dequantized):
                        self._scale_fragment_pairs(
                            s_arr, row_addends, row_multipliers, groups=1
                        )
                    else:
                        self._scale_fragment_with(
                            self.sage_k_scales,
                            scales_view,
                            fragment,
                            s_arr,
                            exponent_addend,
                        )
            elif cutlass.const_expr(exact_dequantized):
                self._scale_fragment_pairs(
                    s_arr, row_addends, row_multipliers, groups=1
                )
            else:
                fragment_multipliers = None
                if cutlass.const_expr(cfg.use_sage_attention):
                    fragment_multipliers = self.sage_k_scales.fragment(
                        scales_view, fragment
                    )
                    self.sage_k_scales.advance(scales_view)
                group_multipliers, group_addends = self._fragment_exponent_terms(
                    fragment_multipliers,
                    exponent_addend,
                    groups=cfg.sage_k_groups_per_fragment,
                    score_bias=score_bias,
                )
                self._scale_fragment_pairs(
                    s_arr,
                    group_addends,
                    group_multipliers,
                    groups=cfg.sage_k_groups_per_fragment,
                )
            # The last iteration reloads its own fragment so the loop body
            # stays branch-free; the wait after the loop retires it.
            next_fragment = fragment + Int32(1)
            if next_fragment > last_fragment:
                next_fragment = last_fragment
            self._load_score_fragment(tmem_base, next_fragment, pending_scores)
            local_sum_pair = self._exponentiate_fragment_pairs(s_arr)
            total_sum_pair = cute.arch.add_packed_f32x2(total_sum_pair, local_sum_pair)

            if cutlass.const_expr(cfg.use_fp8_pv):
                packed_regs = cutlass.Array(
                    Int32, fragment_cols, space=cutlass.AddressSpace.rmem
                )
                for col_idx in cutlass.range_constexpr(fragment_cols):
                    value_base = col_idx * 4
                    packed_regs[col_idx] = _pack_float4_to_fp8_e4m3_inline(
                        Float32(s_arr[value_base]),
                        Float32(s_arr[value_base + 1]),
                        Float32(s_arr[value_base + 2]),
                        Float32(s_arr[value_base + 3]),
                    )
                packed_p = packed_regs.data_ptr().load(count=fragment_cols, alignment=4)
            else:
                packed_p = (
                    s_arr.data_ptr()
                    .load(count=fragment_regs, alignment=4)
                    .to(cfg.pv_dtype)
                    .bitcast(Int32)
                )
            # Fence the packed TMEM store before one lane per warp arrives, so
            # the MMA warp can start this fragment's PV k-slice.
            _keeps_tcgen05_st(
                cfg,
                prims.make_tmem_ptr(
                    tmem_base + fragment * Int32(cfg.fragment_p_packed_cols), Int32
                ),
                packed_p,
                offset=cfg.tmem_p_cols_per_inst,
            )
            cute.arch.fence_view_async_tmem_store()
            prims.tcgen05_fence(prims.Tcgen05Fence.BEFORE_THREAD_SYNC)
            if publishes_fragment:
                prims.mbarrier_arrive(self._fragment_ready.data_ptr() + fragment)
        prims.tcgen05_wait(kind=prims.Tcgen05Wait.LOAD)
        total_sum = Float32(total_sum_pair[0] + total_sum_pair[1])
        self.tmem_s_ref.store_p_local_sum(0, total_sum)

    @cute.jit
    def _scale_fragment_with(
        self,
        scales: Constexpr[SageKScalesResource],
        view,
        fragment: Int32,
        s_arr: cutlass.Array,
        exponent_addend: Float32,
    ) -> None:
        """Turn one fragment's scores into log2 exponents with one route kind's resource."""
        groups = scales.groups
        fragment_multipliers = scales.fragment(view, fragment)
        scales.advance(view)
        group_multipliers, group_addends = self._fragment_exponent_terms(
            fragment_multipliers, exponent_addend, groups=groups
        )
        self._scale_fragment_pairs(
            s_arr, group_addends, group_multipliers, groups=groups
        )

    @cute.jit
    def _load_score_fragment(
        self, tmem_base: Int32, fragment: Int32, scores: cutlass.Array
    ) -> None:
        """Issue the TMEM load of one K32 score fragment into ``scores``.

        The caller waits for the load before reading ``scores``. The copy runs
        before that wait, so it must stay a constant-indexed register rename
        without optimization barriers, or it would read the destination early.
        """
        cfg = self.cfg
        fragment_regs = cfg.softmax_score_fragment_regs
        loaded = _keeps_tcgen05_ld(
            cfg,
            prims.make_tmem_ptr(tmem_base + fragment * Int32(fragment_regs), Float32),
            num=fragment_regs,
            offset=cfg.tile_size_kv // 2,
        )
        for score_idx in cutlass.range_constexpr(fragment_regs):
            scores[score_idx] = loaded[score_idx]

    @cute.jit
    def _scale_fragment_pairs(
        self,
        s_arr: cutlass.Array,
        group_addends: cutlass.Array,
        group_multipliers: cutlass.Array,
        groups: Constexpr[int],
    ) -> None:
        """Turn one fragment of scores into log2 exponents in place.

        Each score takes the multiplier and addend of its compile-time scale
        group (``groups`` per fragment); the two scores of a pair differ in
        group only for the one-token K block.
        """
        fragment_regs = self.cfg.softmax_score_fragment_regs
        group_regs = fragment_regs // groups
        for value_idx in cutlass.range_constexpr(0, fragment_regs, 2):
            group0: Constexpr[int] = value_idx // group_regs
            group1: Constexpr[int] = (value_idx + 1) // group_regs
            s_arr[value_idx], s_arr[value_idx + 1] = cute.arch.fma_packed_f32x2(
                (Float32(s_arr[value_idx]), Float32(s_arr[value_idx + 1])),
                (
                    Float32(group_multipliers[group0]),
                    Float32(group_multipliers[group1]),
                ),
                (Float32(group_addends[group0]), Float32(group_addends[group1])),
            )

    @cute.jit
    def _fragment_exponent_terms(
        self,
        fragment_multipliers: cutlass.Array | None,
        exponent_addend: Float32,
        groups: Constexpr[int],
        score_bias: Float32 | None = None,
    ) -> tuple[cutlass.Array, cutlass.Array]:
        """Return one fragment's exponent multipliers and addends per scale group.

        The multipliers are the resource's ``c * sfQ * sfK_g`` under Sage, else
        the softmax scale. Biased INT32 scores subtract ``bias * multiplier``
        from the row's addend; ``score_bias`` replaces the constant bias when
        given (zero on a tile whose scores come back dequantized).
        """
        cfg = self.cfg
        neg_bias = Float32(-INT32_SCORE_BIAS)
        if cutlass.const_expr(score_bias is not None):
            neg_bias = -score_bias
        group_multipliers = cutlass.Array(
            Float32, groups, space=cutlass.AddressSpace.rmem
        )
        group_addends = cutlass.Array(Float32, groups, space=cutlass.AddressSpace.rmem)
        if cutlass.const_expr(cfg.use_sage_attention):
            for group_idx in cutlass.range_constexpr(groups):
                group_multipliers[group_idx] = Float32(fragment_multipliers[group_idx])
        else:
            group_multipliers[0] = self.scale_softmax_log2
        if cutlass.const_expr(cfg.uses_int32_scores):
            # One rounding per group instead of one conversion per score; the
            # error stays below one quantized score unit.
            bias_pair = (neg_bias, neg_bias)
            for group_base in cutlass.range_constexpr(0, groups - groups % 2, 2):
                group_addends[group_base], group_addends[group_base + 1] = (
                    cute.arch.fma_packed_f32x2(
                        bias_pair,
                        (
                            Float32(group_multipliers[group_base]),
                            Float32(group_multipliers[group_base + 1]),
                        ),
                        (exponent_addend, exponent_addend),
                    )
                )
            if cutlass.const_expr(groups % 2 == 1):
                group_addends[groups - 1] = Float32(
                    cute.math.fma(
                        neg_bias,
                        Float32(group_multipliers[groups - 1]),
                        exponent_addend,
                    )
                )
        else:
            for group_idx in cutlass.range_constexpr(groups):
                group_addends[group_idx] = exponent_addend
        return group_multipliers, group_addends

    @cute.jit
    def _exponentiate_fragment_pairs(
        self, s_arr: cutlass.Array
    ) -> tuple[Float32, Float32]:
        """Turn one fragment of log2 exponents into probabilities in place.

        Returns the fragment's probability sum as a packed pair. Eight
        independent chains keep the denominator update off one long dependency
        chain; the first four pairs seed them, since an add to zero is a real
        FADD. A configurable subset of pairs runs its exponentials on the FMA
        pipe.
        """
        pairs_per_fragment = self.cfg.softmax_score_fragment_regs // 2
        assert pairs_per_fragment >= 4
        sum_chains = cutlass.Array(Float32, 8, space=cutlass.AddressSpace.rmem)
        for pair_idx in cutlass.range_constexpr(pairs_per_fragment):
            value_idx = pair_idx * 2
            p0 = Float32(s_arr[value_idx])
            p1 = Float32(s_arr[value_idx + 1])
            if cutlass.const_expr(
                _pair_uses_ex2_emulation(pair_idx, pairs_per_fragment)
            ):
                p0, p1 = _ex2_emulation_packed_f32x2(p0, p1)
            else:
                p0 = Float32(cute.math.exp2(p0, fastmath=True))
                p1 = Float32(cute.math.exp2(p1, fastmath=True))
            s_arr[value_idx] = p0
            s_arr[value_idx + 1] = p1
            chain_idx = (pair_idx & 3) * 2
            if cutlass.const_expr(pair_idx < 4):
                sum_chains[chain_idx] = p0
                sum_chains[chain_idx + 1] = p1
            else:
                sum_chains[chain_idx], sum_chains[chain_idx + 1] = (
                    cute.arch.add_packed_f32x2(
                        (sum_chains[chain_idx], sum_chains[chain_idx + 1]),
                        (p0, p1),
                    )
                )
        sum01 = cute.arch.add_packed_f32x2(
            (sum_chains[0], sum_chains[1]),
            (sum_chains[2], sum_chains[3]),
        )
        sum23 = cute.arch.add_packed_f32x2(
            (sum_chains[4], sum_chains[5]),
            (sum_chains[6], sum_chains[7]),
        )
        return cute.arch.add_packed_f32x2(sum01, sum23)

    @cute.jit
    def _compute_keeps_p(
        self,
        stage_info: StageInfo,
        *,
        new_max_arr: cutlass.Array,
        s_arr: cutlass.Array,
        route_is_proxy: cutlass.Boolean,
    ) -> None:
        """Materialize one complete-row Keeps probability tile.

        TQ128 gives each warp-group thread a complete 128-column row. TQ64
        gives paired lanes the low/high 64-column halves of one row. Each lane
        writes disjoint packed blocks into the TMEM or SMEM layout consumed by
        BMM2. Streamed profiles, including every block-sparse Keeps profile,
        produce P through the rolled fragment loop instead.
        """
        cfg = self.cfg
        # Every block-sparse Keeps profile streams P; only dense complete rows
        # reach this path.
        assert not cfg.streams_tmem_p_fragments and not cfg.use_block_sparse
        task_cache = _decode_gen_task_cache(stage_info)
        warp_grp_thread_idx = task_cache[_TASK_CACHE_WARP_GRP_THREAD_IDX]
        lane_idx = task_cache[_TASK_CACHE_LANE_IDX]
        num_s_regs = cfg.num_s_regs_per_thread
        vector_elements = cfg.keeps_p_smem_vector_elements
        num_vector_blocks = num_s_regs // vector_elements
        row_idx = _keeps_row_idx(cfg, warp_grp_thread_idx)
        col_base = _keeps_col_base(cfg, lane_idx, num_s_regs)
        p_tmem_stage_base = Int32(0)
        if cutlass.const_expr(cfg.uses_tmem_p):
            p_tmem_stage_base = (
                self._tmem_base_addr
                + Int32(self._tmem_alloc.offset)
                + stage_info.stage_idx * cfg.tmem_s_cols
            )

        exponent_addend = self._exponent_addend(new_max_arr[0], route_is_proxy)

        # Preserve four independent modulo-4 sum chains as two packed pairs,
        # without keeping a second 16-value P array live beside the S row.
        local_sum_pair_01 = (Float32(0.0), Float32(0.0))
        local_sum_pair_23 = (Float32(0.0), Float32(0.0))
        # Each vector block is exactly 16 bytes after conversion. Compute and
        # pack adjacent pairs directly into their final register payload.
        packed_p_regs = cfg.num_packed_p_regs if cfg.uses_two_inst_tmem_p else 4
        packed_p = cutlass.Array(Int32, packed_p_regs, space=cutlass.AddressSpace.rmem)
        for block_idx in cutlass.range_constexpr(num_vector_blocks):
            s_base = block_idx * vector_elements
            packed_base = block_idx * 4 if cfg.uses_two_inst_tmem_p else 0
            if cutlass.const_expr(cfg.use_fp8_pv):
                for packed_idx in cutlass.range_constexpr(4):
                    val_base = packed_idx * 4
                    scaled_pair_01 = ffma2(
                        (
                            s_arr[s_base + val_base],
                            s_arr[s_base + val_base + 1],
                        ),
                        (self.scale_softmax_log2, self.scale_softmax_log2),
                        (exponent_addend, exponent_addend),
                    )
                    p_pair_01 = (
                        cute.math.exp2(scaled_pair_01[0], fastmath=True),
                        cute.math.exp2(scaled_pair_01[1], fastmath=True),
                    )
                    local_sum_pair_01 = fadd2(local_sum_pair_01, p_pair_01)
                    scaled_pair_23 = ffma2(
                        (
                            s_arr[s_base + val_base + 2],
                            s_arr[s_base + val_base + 3],
                        ),
                        (self.scale_softmax_log2, self.scale_softmax_log2),
                        (exponent_addend, exponent_addend),
                    )
                    p_pair_23 = (
                        cute.math.exp2(scaled_pair_23[0], fastmath=True),
                        cute.math.exp2(scaled_pair_23[1], fastmath=True),
                    )
                    local_sum_pair_23 = fadd2(local_sum_pair_23, p_pair_23)
                    packed_p[packed_base + packed_idx] = (
                        _pack_float4_to_fp8_e4m3_inline(
                            p_pair_01[0],
                            p_pair_01[1],
                            p_pair_23[0],
                            p_pair_23[1],
                        )
                    )
            else:
                for packed_idx in cutlass.range_constexpr(4):
                    val_base = packed_idx * 2
                    scaled_pair = ffma2(
                        (
                            s_arr[s_base + val_base],
                            s_arr[s_base + val_base + 1],
                        ),
                        (self.scale_softmax_log2, self.scale_softmax_log2),
                        (exponent_addend, exponent_addend),
                    )
                    p_pair = (
                        cute.math.exp2(scaled_pair[0], fastmath=True),
                        cute.math.exp2(scaled_pair[1], fastmath=True),
                    )
                    if cutlass.const_expr(packed_idx % 2 == 0):
                        local_sum_pair_01 = fadd2(local_sum_pair_01, p_pair)
                    else:
                        local_sum_pair_23 = fadd2(local_sum_pair_23, p_pair)
                    if cutlass.const_expr(cfg.use_bf16_pv):
                        packed_p[packed_base + packed_idx] = _pack_float2_to_bf16(
                            p_pair[0], p_pair[1]
                        )
                    else:
                        packed_p[packed_base + packed_idx] = _pack_float2_to_fp16(
                            p_pair[0], p_pair[1]
                        )

            if cutlass.const_expr(cfg.uses_two_inst_tmem_p):
                # Retain the complete packed row and publish it once below.
                pass
            elif cutlass.const_expr(cfg.uses_tmem_p):
                # Each register packs two 16-bit P values. The q64 TMEM store shape
                # maps paired half-warps onto the low/high 32-column halves of
                # the 64-column UInt32 P tile.
                p_tmem_addr = p_tmem_stage_base + Int32(block_idx * 4)
                _keeps_tcgen05_st(
                    cfg,
                    prims.make_tmem_ptr(p_tmem_addr, Int32),
                    packed_p.data_ptr().load(count=4, alignment=4),
                    offset=cfg.num_packed_p_regs,
                )
            else:
                logical_col = col_base + Int32(s_base)
                smem_offset_bytes = _keeps_p_smem_block_offset_bytes(
                    cfg, row_idx, logical_col
                )
                smem_dst = self._smem_base_p_i32.subview(
                    smem_offset_bytes >> Int32(2)
                ).data_ptr()
                smem_dst.store(
                    packed_p.data_ptr().load(count=4, alignment=4), alignment=16
                )
        if cutlass.const_expr(cfg.uses_two_inst_tmem_p):
            # FP8 publishes a complete row with one x16/x32 STTM. FP16/BF16
            # uses x16 slices to limit Softmax register pressure. This is the
            # complete-row Q128/KV128 path; KV256 publishes K32 fragments.
            assert cfg.num_packed_p_regs in (16, 32, 64)
            regs_per_store = cfg.num_packed_p_regs if cfg.use_fp8_pv else 16
            assert cfg.num_packed_p_regs % regs_per_store == 0
            for store_idx in cutlass.range_constexpr(
                cfg.num_packed_p_regs // regs_per_store
            ):
                packed_offset = store_idx * regs_per_store
                _keeps_tcgen05_st(
                    cfg,
                    prims.make_tmem_ptr(
                        p_tmem_stage_base + Int32(packed_offset), Int32
                    ),
                    (packed_p.data_ptr() + packed_offset).load(
                        count=regs_per_store, alignment=4
                    ),
                    # Separate the paired Softmax destinations by one packed
                    # row (the half-row split for x16/x32 TMEM layouts).
                    offset=cfg.num_packed_p_regs,
                )
            if cutlass.const_expr(cfg.ordered_softmax_early_release):
                # Hand the baton over as soon as this group's TMEM store has
                # issued: the partner's exp2/pack/TMEM store touch only its own
                # registers and TMEM region, so it need not wait for this
                # store to drain, the async fence, or the pipeline commit.
                # The exp2 phases stay serialized (shared MUFU), but the
                # store-drain + commit tail overlaps the partner's wakeup.
                _named_barrier_arrive(
                    cfg.resolved_softmax_order_barrier_threads,
                    barrier_id=cfg.softmax_order_barrier_id + 1 - self.inst_id,
                )
        local_sum_pair0 = fadd2(local_sum_pair_01, local_sum_pair_23)
        local_sum = local_sum_pair0[0] + local_sum_pair0[1]
        self.tmem_s_ref.store_p_local_sum(0, local_sum)

        # Publish the selected memory view before the task-level P pipeline
        # exposes this stage to BMM2.
        if cutlass.const_expr(cfg.uses_tmem_p):
            cute.arch.fence_view_async_tmem_store()
            if cutlass.const_expr(cfg.uses_staged_one_inst_tmem_p):
                # Synchronize the D256 producer warp group after its TMEM stores
                # are visible.
                prims.barrier_cta_sync(4 + self.inst_id, thread_count=128)
        else:
            # Each producer thread orders its own SMEM P stores with an
            # async-proxy fence before its own AsyncUmma producer-commit
            # mbarrier arrive. The full barrier counts all 128 softmax
            # threads, so the commit itself is the warp-group visibility
            # point and no extra named barrier is needed here.
            cute.arch.fence_view_async_shared()

    @cute.jit
    def _compute_p_impl(
        self,
        stage_info: StageInfo,
        *,
        new_max_arr: cutlass.Array,
        s_arr: cutlass.Array,
        route_is_proxy: cutlass.Boolean,
    ) -> None:
        """Compute P from S, stage its BMM2 operand, and publish local sums."""
        cfg = self.cfg
        if cutlass.const_expr(cfg.use_keeps_mma_ab):
            self._compute_keeps_p(
                stage_info,
                new_max_arr=new_max_arr,
                s_arr=s_arr,
                route_is_proxy=route_is_proxy,
            )
            return
        # ProdWork: transform the softmax S registers into the P operand layout
        # expected by BMM2, while recording the per-scale local sums consumed by
        # the denominator update.
        # Decode the scheduler cache once so every store path uses the same
        # warp/lane ownership for SMEM offsets and STSM swizzles.
        task_cache = _decode_gen_task_cache(stage_info)
        warp_grp_thread_idx = task_cache[_TASK_CACHE_WARP_GRP_THREAD_IDX]
        lane_idx = Int32(task_cache[_TASK_CACHE_LANE_IDX])
        # Static launches whose KV tiles split evenly between the instances
        # exponentiate unmasked tiles; a row that stays at -FLT_MAX there is
        # discarded by reduce_sums and the correction gates.
        static_even_tiles = (
            not self.use_variable_seqlens_kv
            and cfg.total_kv_tiles > 0
            and cfg.total_kv_tiles % cfg.num_insts_kv == 0
        )
        guards_fp8_rows = not (static_even_tiles and cfg.q_tiles_are_full)
        if cutlass.const_expr(cfg.tile_size_q == 32 and cfg.use_fp8_pv):
            # Tile-Q=32 FP8 fast path: compute E4M3 P registers in the
            # same order consumed by the STSM helper, while also capturing
            # one local denominator sum per softmax scale group.
            packed_p = cutlass.Array(
                Int32, cfg.num_packed_p_regs, space=cutlass.AddressSpace.rmem
            )
            local_sum = cutlass.Array(
                Float32,
                cfg.num_softmax_scale_groups,
                space=cutlass.AddressSpace.rmem,
            )
            for scale_pair_idx in cutlass.range_constexpr(4):
                scale_base = scale_pair_idx * 2
                q32_s_base = scale_pair_idx * 4
                p_lo, p_hi, sum_lo, sum_hi = _compute_fp8_p_regs_and_local_sums(
                    self.scale_softmax_log2,
                    self._exponent_addend(
                        new_max_arr[scale_base],
                        route_is_proxy,
                        guards_masked_rows=guards_fp8_rows,
                    ),
                    self._exponent_addend(
                        new_max_arr[scale_base + 1],
                        route_is_proxy,
                        guards_masked_rows=guards_fp8_rows,
                    ),
                    s_arr[q32_s_base],
                    s_arr[q32_s_base + 1],
                    s_arr[q32_s_base + 2],
                    s_arr[q32_s_base + 3],
                    s_arr[q32_s_base + 16],
                    s_arr[q32_s_base + 17],
                    s_arr[q32_s_base + 18],
                    s_arr[q32_s_base + 19],
                )
                packed_p[scale_pair_idx] = p_lo
                packed_p[scale_pair_idx + 4] = p_hi
                local_sum[scale_base] = sum_lo
                local_sum[scale_base + 1] = sum_hi
            # Publish the local denominator contribution before committing P so
            # TmemS can update the online-softmax sum after P materialization.
            for scale_idx in cutlass.range_constexpr(cfg.num_softmax_scale_groups):
                self.tmem_s_ref.store_p_local_sum(scale_idx, local_sum[scale_idx])
            # Store the low and high K halves separately. The x4 helper writes
            # transposed E4M3 bytes into the SMEM layout expected by BMM2.
            _store_transposed_smem8b_x4(
                self._smem_base_p_i32,
                packed_p[0],
                packed_p[1],
                packed_p[2],
                packed_p[3],
                warp_grp_thread_idx,
                cfg.tile_size_q,
                cfg.tile_size_kv,
            )
            _store_transposed_smem8b_x4(
                self._smem_base_p_i32,
                packed_p[4],
                packed_p[5],
                packed_p[6],
                packed_p[7],
                warp_grp_thread_idx,
                cfg.tile_size_q,
                cfg.tile_size_kv,
                1,
            )
            # The P producer writes SMEM directly with STSM helpers. Fence
            # and synchronize the warpgroup before the TS pipeline commits
            # the stage to the BMM2 consumer.
            cute.arch.fence_view_async_shared()
            prims.barrier_cta_sync(4 + self.inst_id, thread_count=128)
            return

        if cutlass.const_expr(cfg.tile_size_q == 16 and cfg.use_fp8_pv):
            # Tile-Q=16 FP8 fast path: each helper call handles two softmax
            # scale groups and returns the low/high K halves already packed for
            # STSM. This avoids keeping all 16 FP32 P values live and packing
            # them in a separate pass.
            packed_p = cutlass.Array(
                Int32, cfg.num_packed_p_regs, space=cutlass.AddressSpace.rmem
            )
            local_sum = cutlass.Array(
                Float32,
                cfg.num_softmax_scale_groups,
                space=cutlass.AddressSpace.rmem,
            )
            for scale_pair_idx in cutlass.range_constexpr(2):
                scale_base = scale_pair_idx * 2
                s_base = scale_pair_idx * 4
                p_lo, p_hi, sum_lo, sum_hi = _compute_fp8_p_regs_and_local_sums(
                    self.scale_softmax_log2,
                    self._exponent_addend(
                        new_max_arr[scale_base],
                        route_is_proxy,
                        guards_masked_rows=guards_fp8_rows,
                    ),
                    self._exponent_addend(
                        new_max_arr[scale_base + 1],
                        route_is_proxy,
                        guards_masked_rows=guards_fp8_rows,
                    ),
                    s_arr[s_base],
                    s_arr[s_base + 1],
                    s_arr[s_base + 2],
                    s_arr[s_base + 3],
                    s_arr[s_base + 8],
                    s_arr[s_base + 9],
                    s_arr[s_base + 10],
                    s_arr[s_base + 11],
                )
                packed_p[scale_pair_idx] = p_lo
                packed_p[scale_pair_idx + 2] = p_hi
                local_sum[scale_base] = sum_lo
                local_sum[scale_base + 1] = sum_hi

            for scale_idx in cutlass.range_constexpr(cfg.num_softmax_scale_groups):
                self.tmem_s_ref.store_p_local_sum(scale_idx, local_sum[scale_idx])
            _store_transposed_smem8b(
                self._smem_base_p_i32,
                packed_p.data_ptr().load(count=cfg.num_packed_p_regs, alignment=4),
                warp_grp_thread_idx,
                cfg.tile_size_q,
                cfg.tile_size_kv,
                cfg.num_packed_p_regs,
            )
            cute.arch.fence_view_async_shared()
            prims.barrier_cta_sync(4 + self.inst_id, thread_count=128)
            return

        if cutlass.const_expr(cfg.tile_size_q in (16, 32)):
            # Generic tile-Q 16/32 path: compute P scalars for each
            # softmax scale group, accumulate local sums, then pack/store
            # to the BMM2 SMEM layout.
            q_repeats = max(cfg.tile_size_q // 8, 1)
            num_s_regs = cfg.num_s_regs_per_thread
            num_scale_groups = cfg.num_softmax_scale_groups
            p_vals = cutlass.Array(Float32, num_s_regs, space=cutlass.AddressSpace.rmem)
            local_sums = cutlass.Array(
                Float32, num_scale_groups, space=cutlass.AddressSpace.rmem
            )
            for idx in cutlass.range_constexpr(num_s_regs):
                p_vals[idx] = Float32(0.0)
            for idx in cutlass.range_constexpr(num_scale_groups):
                local_sums[idx] = Float32(0.0)

            for scale_idx in cutlass.range_constexpr(num_scale_groups):
                # Convert each softmax scale group from S to P. Masked rows have
                # new_max == -inf and keep their initialized zero P/local_sum.
                new_max = new_max_arr[scale_idx]
                exponent_addend = self._exponent_addend(new_max, route_is_proxy)
                if new_max != _neg_max_f32():
                    repeat_idx = scale_idx // 2
                    pair_idx = scale_idx % 2
                    generic_s_base = repeat_idx * 4 + pair_idx
                    for k_pair_idx in cutlass.range_constexpr(4):
                        if cutlass.const_expr(k_pair_idx < 2):
                            s_idx = generic_s_base + k_pair_idx * 2
                        else:
                            s_idx = (
                                generic_s_base + q_repeats * 4 + (k_pair_idx - 2) * 2
                            )
                        p_val = cute.math.exp2(
                            s_arr[s_idx] * self.scale_softmax_log2 + exponent_addend,
                            fastmath=True,
                        )
                        p_vals[s_idx] = p_val
                        local_sums[scale_idx] += p_val
            # Hand off denominator contributions through TmemS. P remains a pure
            # MMA operand in SMEM; sums are not reloaded from the P tile.
            for scale_idx in cutlass.range_constexpr(num_scale_groups):
                self.tmem_s_ref.store_p_local_sum(scale_idx, local_sums[scale_idx])

            if cutlass.const_expr(cfg.use_fp8_pv):
                # FP8 P is packed four values per register and stored with
                # transposed 8-bit helpers so BMM2 sees the tcgen05 layout.
                packed_p = cutlass.Array(
                    Int32, cfg.num_packed_p_regs, space=cutlass.AddressSpace.rmem
                )
                for packed_idx in cutlass.range_constexpr(cfg.num_packed_p_regs):
                    val_base = packed_idx * 4
                    packed_p[packed_idx] = _pack_float4_to_fp8_e4m3(
                        p_vals[val_base],
                        p_vals[val_base + 1],
                        p_vals[val_base + 2],
                        p_vals[val_base + 3],
                    )
                _store_transposed_smem8b(
                    self._smem_base_p_i32,
                    packed_p.data_ptr().load(count=cfg.num_packed_p_regs, alignment=4),
                    warp_grp_thread_idx,
                    cfg.tile_size_q,
                    cfg.tile_size_kv,
                    cfg.num_packed_p_regs,
                )
                # Inline byte stores need an explicit CTA barrier before the
                # pipeline stage can be observed by the BMM2 consumer.
                cute.arch.fence_view_async_shared()
                prims.barrier_cta_sync(4 + self.inst_id, thread_count=128)
                return

            # FP16/BF16 P uses stmatrix stores. Each stmatrix group writes
            # one 8x8 fragment to the swizzled SMEM tile consumed by UMMA.
            regs_p = cutlass.Array(
                Int32,
                cfg.num_packed_p_regs,
                space=cutlass.AddressSpace.rmem,
            )
            for pair_idx in cutlass.range_constexpr(cfg.num_packed_p_regs):
                val_base = pair_idx * 2
                if cutlass.const_expr(cfg.use_bf16_pv):
                    regs_p[pair_idx] = _pack_float2_to_bf16(
                        p_vals[val_base], p_vals[val_base + 1]
                    )
                else:
                    regs_p[pair_idx] = _pack_float2_to_fp16(
                        p_vals[val_base], p_vals[val_base + 1]
                    )
            warp_idx = task_cache[_TASK_CACHE_WARP_IDX]
            lane_idx = task_cache[_TASK_CACHE_LANE_IDX]
            for stsm_group_idx in cutlass.range_constexpr(cfg.num_packed_p_regs // 4):
                smem_offset_bytes = _p_stsm_smem_offset_bytes(
                    warp_idx, lane_idx, stsm_group_idx, cfg.tile_size_q
                )
                smem_dst = (
                    self._smem_base_p_i32.subview((smem_offset_bytes >> 2))
                ).data_ptr()
                prims.stmatrix(
                    smem_dst,
                    (regs_p.data_ptr() + stsm_group_idx * 4).load(count=4, alignment=4),
                    prims.MMALayout.COL,
                    shape=prims.StoreShape.M8N8,
                )
            cute.arch.fence_view_async_shared()
            return

        if cutlass.const_expr(cfg.use_fp8_pv):
            # Tile-Q=8 FP8 path: compute packed P and local sums directly
            # from the eight S registers owned by this lane.
            packed_p = cutlass.Array(
                Int32, cfg.num_packed_p_regs, space=cutlass.AddressSpace.rmem
            )
            local_sum = cutlass.Array(
                Float32,
                cfg.num_softmax_scale_groups,
                space=cutlass.AddressSpace.rmem,
            )
            packed_p[0], packed_p[1], local_sum[0], local_sum[1] = (
                _compute_fp8_p_regs_and_local_sums(
                    self.scale_softmax_log2,
                    self._exponent_addend(
                        new_max_arr[0],
                        route_is_proxy,
                        guards_masked_rows=not static_even_tiles,
                    ),
                    self._exponent_addend(
                        new_max_arr[1],
                        route_is_proxy,
                        guards_masked_rows=not static_even_tiles,
                    ),
                    s_arr[0],
                    s_arr[1],
                    s_arr[2],
                    s_arr[3],
                    s_arr[4],
                    s_arr[5],
                    s_arr[6],
                    s_arr[7],
                )
            )
            if cutlass.const_expr(
                not self.use_variable_seqlens_kv
                and not cfg.uses_runtime_q_kv_union
                and not cfg.use_split_kv
                and cfg.has_odd_kv_tail
                and self.inst_id == 1
            ):
                # In the static nonsplit profile the final inst1 wave is only
                # structural padding. Publish an exact zero contribution so
                # the paired instance cannot perturb final normalization.
                if _is_last_loop_iteration(stage_info):
                    packed_p[0] = Int32(0)
                    packed_p[1] = Int32(0)
                    local_sum[0] = Float32(0.0)
                    local_sum[1] = Float32(0.0)
            # Publish sums through TmemS; packed E4M3 P bytes use the
            # transposed SMEM tile consumed by BMM2.
            for scale_idx in cutlass.range_constexpr(cfg.num_softmax_scale_groups):
                self.tmem_s_ref.store_p_local_sum(scale_idx, local_sum[scale_idx])
            _store_transposed_smem8b_x2(
                self._smem_base_p_i32,
                packed_p[0],
                packed_p[1],
                warp_grp_thread_idx,
                cfg.tile_size_q,
                cfg.tile_size_kv,
            )
        else:
            # Tile-Q=8 16-bit path: compute P scalars, accumulate local
            # sums, pack to 16-bit, and store a matrix tile into SMEM.
            local_sum = cutlass.Array(
                Float32,
                cfg.num_softmax_scale_groups,
                space=cutlass.AddressSpace.rmem,
            )
            p_vals = cutlass.Array(Float32, 8, space=cutlass.AddressSpace.rmem)
            for scale_idx in cutlass.range_constexpr(cfg.num_softmax_scale_groups):
                local_sum[scale_idx] = Float32(0.0)
            for p_idx in cutlass.range_constexpr(8):
                p_vals[p_idx] = Float32(0.0)
            exponent_addends = cutlass.Array(
                Float32,
                cfg.num_softmax_scale_groups,
                space=cutlass.AddressSpace.rmem,
            )
            # Both tile-Q=8 16-bit paths below discard or skip -FLT_MAX rows.
            for scale_idx in cutlass.range_constexpr(cfg.num_softmax_scale_groups):
                exponent_addends[scale_idx] = self._exponent_addend(
                    new_max_arr[scale_idx], route_is_proxy, guards_masked_rows=False
                )
            if cutlass.const_expr(
                not self.use_variable_seqlens_kv
                and not cfg.use_split_kv
                and cfg.total_kv_tiles > 0
                and (cfg.total_kv_tiles % cfg.num_insts_kv) == 0
            ):
                # The straight-line helper may synthesize P for an entirely
                # masked sparse instance whose maximum stayed at -inf. This is
                # intentionally safe: reduce_sums' guarded rescale and the
                # correction path's uses_instN gate both key on that sentinel
                # and discard the instance before its P/O contribution is visible.
                p_result = _compute_p_values_and_local_sums(
                    self.scale_softmax_log2,
                    exponent_addends[0],
                    exponent_addends[1],
                    s_arr[0],
                    s_arr[1],
                    s_arr[2],
                    s_arr[3],
                    s_arr[4],
                    s_arr[5],
                    s_arr[6],
                    s_arr[7],
                )
                for p_idx in cutlass.range_constexpr(8):
                    p_vals[p_idx] = p_result[p_idx]
                for scale_idx in cutlass.range_constexpr(cfg.num_softmax_scale_groups):
                    local_sum[scale_idx] = p_result[8 + scale_idx]
            else:
                # General tile-Q=8 path preserves masked S entries as zero P
                # contribution by skipping groups whose new max stayed -inf.
                for scale_idx in cutlass.range_constexpr(cfg.num_softmax_scale_groups):
                    exponent_addend = exponent_addends[scale_idx]
                    if new_max_arr[scale_idx] != _neg_max_f32():
                        for pair_idx in cutlass.range_constexpr(2):
                            p_base = scale_idx + pair_idx * 2
                            scaled_pair = ffma2(
                                (s_arr[p_base], s_arr[p_base + 4]),
                                (
                                    self.scale_softmax_log2,
                                    self.scale_softmax_log2,
                                ),
                                (exponent_addend, exponent_addend),
                            )
                            p_pair = (
                                cute.math.exp2(scaled_pair[0], fastmath=True),
                                cute.math.exp2(scaled_pair[1], fastmath=True),
                            )
                            p_vals[p_base] = p_pair[0]
                            p_vals[p_base + 4] = p_pair[1]
                            local_sum[scale_idx] += p_pair[0]
                            local_sum[scale_idx] += p_pair[1]
            # Pack the P scalars to match the dtype consumed by BMM2.
            regs_p = cutlass.Array(
                Int32, cfg.num_packed_p_regs, space=cutlass.AddressSpace.rmem
            )
            if cutlass.const_expr(cfg.use_bf16_pv):
                for reg_idx in cutlass.range_constexpr(cfg.num_packed_p_regs):
                    val_base = reg_idx * 2
                    regs_p[reg_idx] = _pack_float2_to_bf16(
                        p_vals[val_base], p_vals[val_base + 1]
                    )
            else:
                for reg_idx in cutlass.range_constexpr(cfg.num_packed_p_regs):
                    val_base = reg_idx * 2
                    regs_p[reg_idx] = _pack_float2_to_fp16(
                        p_vals[val_base], p_vals[val_base + 1]
                    )
            # Publish the denominator contribution after P has been computed,
            # before the SMEM fence exposes the P tile to the downstream MMA.
            for scale_idx in cutlass.range_constexpr(cfg.num_softmax_scale_groups):
                self.tmem_s_ref.store_p_local_sum(scale_idx, local_sum[scale_idx])
            warp_idx = task_cache[_TASK_CACHE_WARP_IDX]
            lane_idx = task_cache[_TASK_CACHE_LANE_IDX]
            # Compute the stmatrix destination matching the P descriptor
            # swizzle and store the register fragment.
            slice_idx = warp_idx // Int32(2)
            warp_idx_in_slice = warp_idx % Int32(2)
            mtx_idx = lane_idx // Int32(8)
            thr_row_idx = lane_idx % Int32(8)
            mtx_col_idx = warp_idx_in_slice * Int32(4) + (mtx_idx % Int32(4))
            smem_offset_bytes = (
                slice_idx * Int32(8 * 128)
                + thr_row_idx * Int32(128)
                + ((mtx_col_idx ^ thr_row_idx) * Int32(16))
            )
            smem_dst = (
                self._smem_base_p_i32.subview((smem_offset_bytes >> 2))
            ).data_ptr()
            prims.stmatrix(
                smem_dst,
                regs_p.data_ptr().load(count=4, alignment=4),
                prims.MMALayout.COL,
                shape=prims.StoreShape.M8N8,
            )
        cute.arch.fence_view_async_shared()
        if cutlass.const_expr(cfg.use_fp8_pv):
            # FP8 P uses inline STSM stores. Synchronize the producer
            # warpgroup before the UMMA-consumer pipeline is committed so
            # BMM2 cannot observe a partially written P tile.
            prims.barrier_cta_sync(4 + self.inst_id, thread_count=128)

    @producer_work
    @cute.jit
    def compute_p(
        self,
        stage_info: StageInfo,
        *,
        new_max_arr: cutlass.Array,
        s_arr: cutlass.Array,
    ) -> None:
        """Compute an exact/dense P tile without typed-route metadata."""

        self._compute_p_impl(
            stage_info,
            new_max_arr=new_max_arr,
            s_arr=s_arr,
            route_is_proxy=cutlass.Boolean(False),
        )

    @producer_work
    @cute.jit
    def compute_proxy_route_p(
        self,
        stage_info: StageInfo,
        *,
        new_max_arr: cutlass.Array,
        s_arr: cutlass.Array,
        keeps_route_flags: Int32,
        swaps_route_flags: Uint32,
    ) -> None:
        """Read the route kind from the active Keeps/SWAP metadata view and compute P.

        Keeps carries the kind in its Int32 route flags, SWAP in its
        bit-preserving Uint32 flags word. The kind selects only the exponent
        addend; the max pass has already applied the proxy mass to the scores.
        """

        assert self.cfg.use_block_sparse_proxy_routes
        if cutlass.const_expr(self.cfg.use_keeps_mma_ab):
            route_is_proxy = _route_is_proxy(keeps_route_flags)
        else:
            route_is_proxy = _route_is_proxy(swaps_route_flags.bitcast(Int32))
        self._compute_p_impl(
            stage_info,
            new_max_arr=new_max_arr,
            s_arr=s_arr,
            route_is_proxy=route_is_proxy,
        )

    @consumer_work(
        returns=(
            p_desc_0_slot,
            p_desc_1_slot,
            p_tmem_addr_0_slot,
            p_tmem_addr_1_slot,
        )
    )
    @cute.jit
    def p_operands(
        self, stage_info: StageInfo
    ) -> tuple[
        prims.Tcgen05SmemDesc,
        prims.Tcgen05SmemDesc,
        Int32,
        Int32,
    ]:
        """Publish the stage-specific P operand consumed by BMM2."""
        cfg = self.cfg
        p_desc_0 = prims.Tcgen05SmemDesc(0)
        p_desc_1 = prims.Tcgen05SmemDesc(0)
        p_tmem_addr_0 = Int32(0)
        p_tmem_addr_1 = Int32(0)
        if cutlass.const_expr(cfg.uses_tmem_p):
            # ConsWork: select the physical TMEM stage paired with the P
            # pipeline token that was just waited. The allocation aliases the
            # stats-free columns of the corresponding S stage.
            p_stage_cols = cfg.tmem_s_cols
            if cutlass.const_expr(cfg.streams_tmem_p_fragments):
                # A streamed profile's four pipeline stages are K32 fragments of one P
                # operand, not four independent full S/P stages.
                p_stage_cols = cfg.fragment_p_packed_cols
            p_tmem_addr = self._tmem_base_addr + Int32(
                self._tmem_alloc.offset + stage_info.stage_idx * p_stage_cols
            )
            if cutlass.const_expr(self.inst_id == 0):
                p_tmem_addr_0 = p_tmem_addr
            else:
                p_tmem_addr_1 = p_tmem_addr
        else:
            # ConsWork: build the SMEM descriptor for P. Only the descriptor
            # slot corresponding to this resource instance is populated; the
            # MmaTask receives both slots and selects the active one for BMM2.
            p_desc = prims.Tcgen05SmemDesc.build(
                self._smem_base_p,
                leading_byte_offset=Int32(cfg.tile_size_q * 128),
                stride_byte_offset=1024,
                layout=prims.Tcgen05SmemSwizzle.SWIZZLE_128B,
            )
            if cutlass.const_expr(self.inst_id == 0):
                p_desc_0 = p_desc
            else:
                p_desc_1 = p_desc
        return p_desc_0, p_desc_1, p_tmem_addr_0, p_tmem_addr_1

    @consumer_work(returns=p_tmem_addr_0_slot)
    @cute.jit
    def wait_p_fragment(
        self,
        stage_info: StageInfo,
        *,
        fragment_idx: Constexpr[int],
    ) -> Int32:
        """Wait for and return the next streamed P-fragment TMEM address."""
        cfg = self.cfg
        _ = stage_info
        assert cfg.streams_tmem_p_fragments
        _wait_for_mbarrier_phase(
            self._fragment_ready.data_ptr() + Int32(fragment_idx),
            self.tmem_s_ref.producer_state.phase,
        )
        prims.tcgen05_fence(prims.Tcgen05Fence.AFTER_THREAD_SYNC)

        p_tmem_addr = self._tmem_base_addr + Int32(
            self._tmem_alloc.offset + fragment_idx * cfg.fragment_p_packed_cols
        )
        return p_tmem_addr
