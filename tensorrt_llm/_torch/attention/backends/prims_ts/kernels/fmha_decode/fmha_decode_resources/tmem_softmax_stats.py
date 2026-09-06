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

"""Softmax stats resources for FMHA decode TS kernel.

Holds ``TmemSoftmaxLocalResource`` (per-loop ``old_max``/``new_max``/``sum``
arrays exchanged with the correction warps) and ``TmemSoftmaxGlobalResource``
(FP8 sum-correction helper that reapplies running-max correction after P
quantization). ``TmemStatsDoneResource`` carries the overwrite credit for TMEM
columns shared by S and the local stats payload.
"""

from dataclasses import dataclass
from typing import ClassVar

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32

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
from ...placeholder_helpers import _placeholder_local_array
from .helpers_common import (
    Constexpr,
    DecodeGenResourceBase,
    ResourceVars,
    fadd2,
    ffma2,
    fmul2,
    _named_barrier_arrive,
    _named_barrier_sync,
    _TASK_CACHE_TMEM_BASE_OFFSET,
    _TASK_CACHE_WARP_GRP_THREAD_IDX,
    _decode_gen_task_cache,
    _neg_max_f32,
    _softmax_scale_pair_width,
)
from .tmem_s import TmemSResource


@dataclass(kw_only=True)
class TmemStatsDoneResource(MemoryResource):
    """Barrier returned after Correction loads stats from aliased TMEM."""

    is_barrier: Constexpr[bool] = True


@dataclass(kw_only=True)
class TmemSoftmaxOrderResource(MemoryResource):
    """CTA named-barrier baton for deterministic P0/P1 ordering.

    The barrier is only enabled for profiles where softmax0 and softmax1 share
    producer-side hazards. It serializes the P0/P1 publication order without
    changing the TS resource graph.
    """

    cfg: Constexpr[FmhaDecodeConfig] = None
    is_barrier: Constexpr[bool] = True

    @consumer_work
    @cute.jit
    def prime_softmax1(self, stage_info: StageInfo) -> None:
        """Prime the softmax1 side of the ordered P0/P1 baton."""
        _ = stage_info
        if cutlass.const_expr(self.cfg.uses_ordered_softmax_barrier):
            # ConsWork: seed the first named barrier so softmax0 is allowed to
            # publish before softmax1 enters its ordered slot.
            _named_barrier_arrive(
                self.cfg.resolved_softmax_order_barrier_threads,
                barrier_id=self.cfg.softmax_order_barrier_id,
            )

    @producer_work
    @cute.jit
    def wait_softmax0(self, stage_info: StageInfo) -> None:
        """Wait until softmax0 is allowed to publish before softmax1."""
        _ = stage_info
        if cutlass.const_expr(self.cfg.uses_ordered_softmax_barrier):
            # ProdWork: softmax0 waits on the baton before publishing P/stats,
            # preserving the expected P0 -> P1 order for shared resources.
            _named_barrier_sync(
                self.cfg.resolved_softmax_order_barrier_threads,
                barrier_id=self.cfg.softmax_order_barrier_id,
            )

    @producer_work
    @cute.jit
    def release_softmax1(self, stage_info: StageInfo) -> None:
        """Signal that softmax1 may publish its P payload."""
        _ = stage_info
        if cutlass.const_expr(
            self.cfg.uses_ordered_softmax_barrier
            and not self.cfg.ordered_softmax_early_release
        ):
            # ProdWork: softmax0 hands the baton to softmax1 after its P/stats
            # payload is visible to downstream tasks.
            _named_barrier_arrive(
                self.cfg.resolved_softmax_order_barrier_threads,
                barrier_id=self.cfg.softmax_order_barrier_id + 1,
            )

    @consumer_work
    @cute.jit
    def wait_softmax1(self, stage_info: StageInfo) -> None:
        """Wait until softmax1 has completed its ordered publication slot."""
        _ = stage_info
        if cutlass.const_expr(self.cfg.uses_ordered_softmax_barrier):
            # ConsWork: softmax1 waits until softmax0's publication for this
            # iteration is visible, preserving the P0 -> P1 order. The baton
            # is a two-party softmax0/softmax1 protocol; correction does not
            # participate.
            _named_barrier_sync(
                self.cfg.resolved_softmax_order_barrier_threads,
                barrier_id=self.cfg.softmax_order_barrier_id + 1,
            )

    @consumer_work
    @cute.jit
    def release_softmax0(self, stage_info: StageInfo) -> None:
        """Release the baton so the next softmax0 publication can proceed."""
        _ = stage_info
        if cutlass.const_expr(
            self.cfg.uses_ordered_softmax_barrier
            and not self.cfg.ordered_softmax_early_release
        ):
            # ConsWork: softmax1 completes the baton cycle after its own
            # publication and allows the next softmax0 slot to enter the
            # ordered region.
            _named_barrier_arrive(
                self.cfg.resolved_softmax_order_barrier_threads,
                barrier_id=self.cfg.softmax_order_barrier_id,
            )


@dataclass(kw_only=True)
class TmemSoftmaxLocalResource(DecodeGenResourceBase):
    """TMEM-local softmax statistics exchanged with correction.

    Loop stats carry old/new maxima for in-place O correction. Tail stats carry
    final sums and maxima for output normalization.
    """

    _task_local_specs: ClassVar[tuple[tuple, ...]] = (
        (
            "old_max_arr",
            cutlass.Array,
            None,
            "Previous softmax maximum read from TMEM.",
        ),
        ("new_max_arr", cutlass.Array, None, "Current softmax maximum read from TMEM."),
        ("sum_arr", cutlass.Array, None, "Softmax denominator read from TMEM."),
        (
            "inst_old_max_arr",
            cutlass.Array,
            None,
            "Per-instruction previous softmax maximum.",
        ),
        (
            "inst_new_max_arr",
            cutlass.Array,
            None,
            "Per-instruction current softmax maximum.",
        ),
        ("inst_sum_arr", cutlass.Array, None, "Per-instruction softmax denominator."),
    )
    inst_id: Constexpr[int] = 0
    cfg: Constexpr[FmhaDecodeConfig] = None
    _alloc: Constexpr[TmemAllocation | None] = None
    _smem_alloc: Constexpr[SmemAllocation | None] = None
    _inst_new_max_arr: cutlass.Array | None = None
    _inst_sum_arr: cutlass.Array | None = None
    old_max_arr: Constexpr[TaskLocalVariable] = TaskLocalVariable.uninitialized()
    new_max_arr: Constexpr[TaskLocalVariable] = TaskLocalVariable.uninitialized()
    sum_arr: Constexpr[TaskLocalVariable] = TaskLocalVariable.uninitialized()
    inst_old_max_arr: Constexpr[TaskLocalVariable] = TaskLocalVariable.uninitialized()
    inst_new_max_arr: Constexpr[TaskLocalVariable] = TaskLocalVariable.uninitialized()
    inst_sum_arr: Constexpr[TaskLocalVariable] = TaskLocalVariable.uninitialized()

    def _init_placeholder_state(self) -> None:
        """Create placeholder arrays for softmax-local stat handoff."""
        num_sg = self.cfg.num_softmax_scale_groups
        self.old_max_arr.default = _placeholder_local_array(Float32, num_sg)
        self.new_max_arr.default = _placeholder_local_array(Float32, num_sg)
        self.sum_arr.default = _placeholder_local_array(Float32, num_sg)
        self.inst_old_max_arr.default = _placeholder_local_array(Float32, num_sg)
        self.inst_new_max_arr.default = _placeholder_local_array(Float32, num_sg)
        self.inst_sum_arr.default = _placeholder_local_array(Float32, num_sg)
        self._inst_new_max_arr = _placeholder_local_array(Float32, num_sg)
        self._inst_sum_arr = _placeholder_local_array(Float32, num_sg)

    def _stats_smem_rows(self) -> int:
        """One SMEM stats row per softmax warp-group thread."""
        num_warps = (
            self.cfg.softmax0_num_warps
            if self.inst_id == 0
            else self.cfg.softmax1_num_warps
        )
        return num_warps * 32

    def _stats_smem_row_elems(self) -> int:
        """FP32 payload elements per row: two stat halves per scale group."""
        return self.cfg.num_softmax_scale_groups * 2

    def _stats_smem_alignment(self) -> int:
        return min(16, self._stats_smem_row_elems() * 4)

    def get_smem_requirements(self) -> list[SmemAllocation]:
        """Allocate the SMEM stats ring when stats cannot own TMEM columns."""
        if not self.cfg.keeps_stats_via_smem:
            return []
        if self._smem_alloc is None:
            stage_bytes = self._stats_smem_rows() * self._stats_smem_row_elems() * 4
            self._smem_alloc = SmemAllocation(
                name=f"{self.name}_smem",
                size_bytes=self.pipeline_config.num_stages * stage_bytes,
                alignment=16,
            )
        return [self._smem_alloc]

    def get_tmem_requirements(self) -> list[TmemAllocation]:
        """Allocate TMEM columns for softmax stat handoff records."""
        if self.cfg.keeps_stats_via_smem:
            # Do not expose a fictitious TMEM access to the allocator or the
            # exhaustive dependency checker. The constexpr store/load paths
            # below use only the SMEM ring for this profile.
            return []
        if self._alloc is None:
            self._alloc = TmemAllocation(
                name=f"{self.name}",
                num_columns=self.cfg.tmem_stats_cols,
            )
        return [self._alloc]

    @cute.jit
    def _create_initial_task_locals(
        self, context: ResourceContext | None = None
    ) -> ResourceVars:
        """Initialize loop and tail-visible softmax stat arrays."""
        num_sg = self.cfg.num_softmax_scale_groups
        # Retain final per-instance stats alongside the loop-carried values;
        # the correction task receives them explicitly for the tail merge.
        self._inst_new_max_arr = cutlass.Array(
            Float32, num_sg, space=cutlass.AddressSpace.rmem
        )
        self._inst_sum_arr = cutlass.Array(
            Float32, num_sg, space=cutlass.AddressSpace.rmem
        )
        result = {
            "old_max_arr": cutlass.Array(
                Float32, num_sg, space=cutlass.AddressSpace.rmem
            ),
            "new_max_arr": cutlass.Array(
                Float32, num_sg, space=cutlass.AddressSpace.rmem
            ),
            "sum_arr": cutlass.Array(Float32, num_sg, space=cutlass.AddressSpace.rmem),
            "inst_old_max_arr": cutlass.Array(
                Float32, num_sg, space=cutlass.AddressSpace.rmem
            ),
            "inst_new_max_arr": self._inst_new_max_arr,
            "inst_sum_arr": self._inst_sum_arr,
        }
        for idx in cutlass.range_constexpr(num_sg):
            # Initialize both loop stats and final instance stats to the
            # neutral softmax state.
            result["old_max_arr"][idx] = _neg_max_f32()
            result["new_max_arr"][idx] = _neg_max_f32()
            result["sum_arr"][idx] = Float32(0.0)
            result["inst_old_max_arr"][idx] = _neg_max_f32()
            self._inst_new_max_arr[idx] = _neg_max_f32()
            self._inst_sum_arr[idx] = Float32(0.0)
        return result

    @cute.jit
    def _create_work_tile_task_locals(
        self, context: ResourceContext | None = None
    ) -> ResourceVars:
        """Create per-work-tile softmax stat arrays for persistent scheduling."""
        _ = context
        num_sg = self.cfg.num_softmax_scale_groups
        # Tail-visible instance stats are per work tile, so persistent kernels
        # must allocate fresh local arrays when the scheduler advances.
        self._inst_new_max_arr = cutlass.Array(
            Float32, num_sg, space=cutlass.AddressSpace.rmem
        )
        self._inst_sum_arr = cutlass.Array(
            Float32, num_sg, space=cutlass.AddressSpace.rmem
        )
        result = {
            "old_max_arr": cutlass.Array(
                Float32, num_sg, space=cutlass.AddressSpace.rmem
            ),
            "new_max_arr": cutlass.Array(
                Float32, num_sg, space=cutlass.AddressSpace.rmem
            ),
            "sum_arr": cutlass.Array(Float32, num_sg, space=cutlass.AddressSpace.rmem),
            "inst_old_max_arr": cutlass.Array(
                Float32, num_sg, space=cutlass.AddressSpace.rmem
            ),
            "inst_new_max_arr": self._inst_new_max_arr,
            "inst_sum_arr": self._inst_sum_arr,
        }
        for idx in cutlass.range_constexpr(num_sg):
            result["old_max_arr"][idx] = _neg_max_f32()
            result["new_max_arr"][idx] = _neg_max_f32()
            result["sum_arr"][idx] = Float32(0.0)
            result["inst_old_max_arr"][idx] = _neg_max_f32()
            self._inst_new_max_arr[idx] = _neg_max_f32()
            self._inst_sum_arr[idx] = Float32(0.0)
        return result

    @consumer_work(
        work_attrs=WorkAttr.AUXILIARY,
        returns=(
            old_max_arr,
            new_max_arr,
            sum_arr,
            inst_old_max_arr,
            inst_new_max_arr,
            inst_sum_arr,
        ),
    )
    @cute.jit
    def init_stats_state(
        self, stage_info: StageInfo
    ) -> tuple[
        cutlass.Array,
        cutlass.Array,
        cutlass.Array,
        cutlass.Array,
        cutlass.Array,
        cutlass.Array,
    ]:
        """Initialize and return correction-side softmax stat state."""
        # ConsAuxWork: allocate correction-visible stat arrays that carry
        # loop old/new maxima and tail per-instance sum/max payloads.
        result = self._create_initial_task_locals(stage_info.context)
        return (
            result["old_max_arr"],
            result["new_max_arr"],
            result["sum_arr"],
            result["inst_old_max_arr"],
            result["inst_new_max_arr"],
            result["inst_sum_arr"],
        )

    @cute.jit
    def _stats_ptr(self, stage_info: StageInfo):
        """Return the TMEM pointer for this resource's stats handoff slot."""
        task_cache = _decode_gen_task_cache(stage_info)
        stats_base = (
            task_cache[_TASK_CACHE_TMEM_BASE_OFFSET]
            + Int32(self._alloc.offset)
            + stage_info.stage_idx * self.cfg.tmem_s_cols
        )
        return prims.make_tmem_ptr(stats_base, Float32)

    @cute.jit
    def _stats_vector(self, first_arr: cutlass.Array, second_arr: cutlass.Array):
        """Pack the two stats halves into the tcgen05 vector store layout."""
        cfg = self.cfg
        if cutlass.const_expr(cfg.num_softmax_scale_groups == 8):
            return cutlass.Vector.from_elements(
                (
                    first_arr[0],
                    first_arr[1],
                    first_arr[2],
                    first_arr[3],
                    first_arr[4],
                    first_arr[5],
                    first_arr[6],
                    first_arr[7],
                    second_arr[0],
                    second_arr[1],
                    second_arr[2],
                    second_arr[3],
                    second_arr[4],
                    second_arr[5],
                    second_arr[6],
                    second_arr[7],
                ),
                Float32,
            )
        elif cutlass.const_expr(cfg.num_softmax_scale_groups == 4):
            return cutlass.Vector.from_elements(
                (
                    first_arr[0],
                    first_arr[1],
                    first_arr[2],
                    first_arr[3],
                    second_arr[0],
                    second_arr[1],
                    second_arr[2],
                    second_arr[3],
                ),
                Float32,
            )
        elif cutlass.const_expr(cfg.num_softmax_scale_groups == 1):
            return cutlass.Vector.from_elements(
                (first_arr[0], second_arr[0]),
                Float32,
            )
        else:
            return cutlass.Vector.from_elements(
                (
                    first_arr[0],
                    first_arr[1],
                    second_arr[0],
                    second_arr[1],
                ),
                Float32,
            )

    @cute.jit
    def _stats_smem_ptr(self, stage_info: StageInfo):
        """Return this thread's SMEM stats slot for the current stage.

        Softmax thread ``i`` and correction thread ``i`` address the same
        row, matching the identity lane mapping of the TMEM 32x32b path.
        """
        context = stage_info.context
        assert context is not None and context.smem_base is not None
        task_cache = _decode_gen_task_cache(stage_info)
        row_idx = task_cache[_TASK_CACHE_WARP_GRP_THREAD_IDX]
        row_elems = self._stats_smem_row_elems()
        stage_elems = self._stats_smem_rows() * row_elems
        base_ptr = context.smem_base.data_ptr() + self._smem_alloc.offset
        view = cutlass.Array(
            base_ptr,
            dtype=Float32,
            shape=(self.pipeline_config.num_stages * stage_elems,),
            addrspace=3,
        )
        elem_offset = stage_info.stage_idx * stage_elems + row_idx * row_elems
        return view.subview(elem_offset).data_ptr()

    @cute.jit
    def _store_stats_vector(
        self,
        stage_info: StageInfo,
        first_arr: cutlass.Array,
        second_arr: cutlass.Array,
    ) -> None:
        """Store one two-part softmax-stats payload for correction."""
        stats = self._stats_vector(first_arr, second_arr)
        if cutlass.const_expr(self.cfg.keeps_stats_via_smem):
            # SMEM handoff: the softmax-local mbarrier pipeline already
            # orders this generic-proxy store against correction's load, so
            # the payload needs no TMEM traffic, waits, or stats-done credit.
            self._stats_smem_ptr(stage_info).store(
                stats, alignment=self._stats_smem_alignment()
            )
        else:
            stats_ptr = self._stats_ptr(stage_info)
            # Store the two stat halves as one vector payload so correction
            # observes old/new or sum/new pairs from the same producer fire.
            prims.tcgen05_st(
                "32x32b",
                stats_ptr,
                stats,
            )
            cute.arch.fence_view_async_tmem_store()

    @cute.jit
    def _load_stats_vector(self, stage_info: StageInfo):
        """Load one two-part softmax-stats payload written by softmax."""
        if cutlass.const_expr(self.cfg.keeps_stats_via_smem):
            return self._stats_smem_ptr(stage_info).load(
                count=self.cfg.num_softmax_scale_groups * 2,
                alignment=self._stats_smem_alignment(),
            )
        stats_ptr = self._stats_ptr(stage_info)
        # Reload exactly the payload shape written by _store_stats_vector; the
        # TMEM view fence makes the read visible to scalar consumers.
        loaded = prims.tcgen05_ld(
            "32x32b",
            stats_ptr,
            num=self.cfg.num_softmax_scale_groups * 2,
        )
        cute.arch.fence_view_async_tmem_load()
        return loaded

    @cute.jit
    def _return_stats_state(
        self,
        old_max_arr: cutlass.Array,
        new_max_arr: cutlass.Array,
        sum_arr: cutlass.Array,
        inst_old_max_arr: cutlass.Array,
        inst_new_max_arr: cutlass.Array,
        inst_sum_arr: cutlass.Array,
    ) -> tuple[object, object, object, object, object, object]:
        """Return the TS task-local stats tuple in scheduler order."""
        return (
            old_max_arr,
            new_max_arr,
            sum_arr,
            inst_old_max_arr,
            inst_new_max_arr,
            inst_sum_arr,
        )

    @producer_work
    @cute.jit
    def store_loop_old_new_stats(
        self,
        stage_info: StageInfo,
        *,
        old_max_arr: cutlass.Array,
        new_max_arr: cutlass.Array,
        sum_arr: cutlass.Array,
    ) -> None:
        """Loop producer handoff for old/new maxima used by O rescale."""
        _ = sum_arr
        # ProdWork: loop old/new max handoff is skipped for a single K/V tile because
        # correction's HEAD phase only drains the resource token; there is no
        # previous O accumulator to rescale yet.
        skip_initial_handoff = stage_info.loop_end == Int32(1)
        if not skip_initial_handoff:
            # Publish old/new maxima so correction can rescale the live O stage
            # before the next PV wave accumulates into it.
            self._store_stats_vector(stage_info, old_max_arr, new_max_arr)

    @producer_work
    @cute.jit
    def store_loop_sum_new_stats(
        self,
        stage_info: StageInfo,
        *,
        old_max_arr: cutlass.Array,
        new_max_arr: cutlass.Array,
        sum_arr: cutlass.Array,
    ) -> None:
        """Loop LastIter producer handoff for final denominator and max."""
        _ = old_max_arr
        # ProdWork: LastIter loop payload carries the final denominator and max for
        # correction's TAIL normalization.
        # Pair sum with new max because tail correction needs both to normalize
        # the final O stages.
        self._store_stats_vector(stage_info, sum_arr, new_max_arr)

    @producer_work
    @cute.jit
    def store_tail_sum_new_stats(
        self,
        stage_info: StageInfo,
        *,
        old_max_arr: cutlass.Array,
        new_max_arr: cutlass.Array,
        sum_arr: cutlass.Array,
    ) -> None:
        """Tail producer handoff for final denominator and max."""
        _ = old_max_arr
        # ProdWork: Keeps-MMA-AB publishes final sums after the loop because its P path
        # avoids peeling the full P body into a LastIter loop guard.
        # Use the same sum/new-max payload shape as the loop LastIter handoff
        # so tail correction has one consumer path for both schedule variants.
        self._store_stats_vector(stage_info, sum_arr, new_max_arr)

    @consumer_work(
        returns=(
            old_max_arr,
            new_max_arr,
            sum_arr,
            inst_old_max_arr,
            inst_new_max_arr,
            inst_sum_arr,
        ),
    )
    @cute.jit
    def load_head_stats(
        self,
        stage_info: StageInfo,
        *,
        old_max_arr: cutlass.Array,
        new_max_arr: cutlass.Array,
        sum_arr: cutlass.Array,
        inst_old_max_arr: cutlass.Array,
        inst_new_max_arr: cutlass.Array,
        inst_sum_arr: cutlass.Array,
    ) -> tuple[object, object, object, object, object, object]:
        """Drain the initial stats token without reading TMEM."""
        _ = stage_info
        # ConsWork: HEAD only drains the initial resource token. Reading TMEM
        # here would observe stale data on the single-tile path.
        return self._return_stats_state(
            old_max_arr,
            new_max_arr,
            sum_arr,
            inst_old_max_arr,
            inst_new_max_arr,
            inst_sum_arr,
        )

    @consumer_work(
        returns=(
            old_max_arr,
            new_max_arr,
            sum_arr,
            inst_old_max_arr,
            inst_new_max_arr,
            inst_sum_arr,
        ),
    )
    @cute.jit
    def load_loop_stats(
        self,
        stage_info: StageInfo,
        *,
        old_max_arr: cutlass.Array,
        new_max_arr: cutlass.Array,
        sum_arr: cutlass.Array,
        inst_old_max_arr: cutlass.Array,
        inst_new_max_arr: cutlass.Array,
        inst_sum_arr: cutlass.Array,
    ) -> tuple[object, object, object, object, object, object]:
        """Load loop old/new maxima for correction's in-place O rescale."""
        cfg = self.cfg
        # ConsWork: load the old/new max payload that correction uses to
        # rescale the live O accumulator before the next PV MMA wave.
        loaded = self._load_stats_vector(stage_info)
        for scale_idx in cutlass.range_constexpr(cfg.num_softmax_scale_groups):
            loaded_new_max = loaded[cfg.num_softmax_scale_groups + scale_idx]
            new_max_arr[scale_idx] = loaded_new_max
            # Loop stats carry old max for in-place O rescaling.
            loaded_old_max = loaded[scale_idx]
            old_max_arr[scale_idx] = loaded_old_max
            inst_old_max_arr[scale_idx] = loaded_old_max
            inst_new_max_arr[scale_idx] = loaded_new_max
        return self._return_stats_state(
            old_max_arr,
            new_max_arr,
            sum_arr,
            inst_old_max_arr,
            inst_new_max_arr,
            inst_sum_arr,
        )

    @consumer_work(
        returns=(
            old_max_arr,
            new_max_arr,
            sum_arr,
            inst_old_max_arr,
            inst_new_max_arr,
            inst_sum_arr,
        ),
    )
    @cute.jit
    def load_tail_stats(
        self,
        stage_info: StageInfo,
        *,
        old_max_arr: cutlass.Array,
        new_max_arr: cutlass.Array,
        sum_arr: cutlass.Array,
        inst_old_max_arr: cutlass.Array,
        inst_new_max_arr: cutlass.Array,
        inst_sum_arr: cutlass.Array,
    ) -> tuple[object, object, object, object, object, object]:
        """Load tail final sums/maxima and mirror them into instance stats."""
        cfg = self.cfg
        # ConsWork: load the final denominator/max payload and copy it into the
        # per-instance arrays consumed by correction_tail_epilogue.
        loaded = self._load_stats_vector(stage_info)
        for scale_idx in cutlass.range_constexpr(cfg.num_softmax_scale_groups):
            loaded_new_max = loaded[cfg.num_softmax_scale_groups + scale_idx]
            loaded_sum = loaded[scale_idx]
            new_max_arr[scale_idx] = loaded_new_max
            sum_arr[scale_idx] = loaded_sum
            # Tail stats are mirrored into instance-local arrays so the final
            # reducer can combine the two K/V instruction streams.
            inst_new_max_arr[scale_idx] = loaded_new_max
            inst_sum_arr[scale_idx] = loaded_sum
        return self._return_stats_state(
            old_max_arr,
            new_max_arr,
            sum_arr,
            inst_old_max_arr,
            inst_new_max_arr,
            inst_sum_arr,
        )


@dataclass(kw_only=True)
class TmemSoftmaxGlobalResource(DecodeGenResourceBase):
    """FP8 softmax sum correction helper.

    FP8 P is quantized before the running sum is finalized. This resource
    applies the max correction after P production and publishes the corrected
    sums back through TmemS.
    """

    inst_id: Constexpr[int] = 0
    cfg: Constexpr[FmhaDecodeConfig] = None
    scale_softmax_log2: Float32 = None
    sum_barrier_id: Constexpr[int] = 2
    local_ref: Constexpr[MemoryResource | None] = None
    p_ref: Constexpr[MemoryResource | None] = None
    tmem_s_ref: Constexpr[TmemSResource] = None

    def get_smem_requirements(self) -> list[SmemAllocation]:
        """Global softmax correction uses no private SMEM allocation."""
        return []

    def get_tmem_requirements(self) -> list[TmemAllocation]:
        """Global softmax correction reuses peer resource state."""
        return []

    @producer_work
    @cute.jit
    def global_correction(
        self,
        stage_info: StageInfo,
        *,
        old_max_arr: cutlass.Array,
        new_max_arr: cutlass.Array,
        sum_arr: cutlass.Array,
    ) -> None:
        """Apply FP8 P-quantization denominator correction through TmemS."""
        cfg = self.cfg
        if cutlass.const_expr(not cfg.use_fp8_qkv):
            return

        num_scale_groups = cfg.num_softmax_scale_groups
        for scale_base in cutlass.range_constexpr(0, num_scale_groups, 2):
            # ProdWork: FP8 P is quantized before the running denominator is finalized.
            # Apply the same max correction as reduce_sums and publish the
            # corrected sums through TmemS for the next stage.
            # Gather old/new max, previous sum, and quantized-P local sum for
            # one pair of scale groups.
            old_max = cutlass.Array(Float32, 2, space=cutlass.AddressSpace.rmem)
            new_max = cutlass.Array(Float32, 2, space=cutlass.AddressSpace.rmem)
            sum_vals = cutlass.Array(Float32, 2, space=cutlass.AddressSpace.rmem)
            local_sum = cutlass.Array(Float32, 2, space=cutlass.AddressSpace.rmem)
            exp_scale = cutlass.Array(Float32, 2, space=cutlass.AddressSpace.rmem)
            pair_width = _softmax_scale_pair_width(num_scale_groups, scale_base)
            # KeepsMmaAb tracks one scale group. Pad the inactive packed-FMA
            # lane with neutral values, then touch only live task-local lanes.
            for pair_idx in cutlass.range_constexpr(2):
                old_max[pair_idx] = Float32(0.0)
                new_max[pair_idx] = Float32(0.0)
                sum_vals[pair_idx] = Float32(0.0)
                local_sum[pair_idx] = Float32(0.0)
            for pair_idx in cutlass.range_constexpr(pair_width):
                scale_idx = scale_base + pair_idx
                old_max[pair_idx] = old_max_arr[scale_idx]
                new_max[pair_idx] = new_max_arr[scale_idx]
                sum_vals[pair_idx] = sum_arr[scale_idx]
                if self.p_ref is not None:
                    local_sum[pair_idx] = self.tmem_s_ref.load_p_local_sum(scale_idx)

            # Convert the max delta into the online-softmax rescale factor that
            # brings the previous denominator into the new max frame.
            max_diff_pair = fadd2((old_max[0], old_max[1]), (-new_max[0], -new_max[1]))
            scale_pair = fmul2(
                (self.scale_softmax_log2, self.scale_softmax_log2), max_diff_pair
            )
            for pair_idx in cutlass.range_constexpr(2):
                exp_scale[pair_idx] = cute.math.exp2(
                    scale_pair[pair_idx], fastmath=True
                )
            updated_sums = ffma2(
                (exp_scale[0], exp_scale[1]),
                (sum_vals[0], sum_vals[1]),
                (local_sum[0], local_sum[1]),
            )

            # Publish corrected sums both through this resource return value and
            # through TmemS, which reduce_sums copies for FP8.
            for pair_idx in cutlass.range_constexpr(pair_width):
                scale_idx = scale_base + pair_idx
                sum_arr[scale_idx] = updated_sums[pair_idx]
                self.tmem_s_ref.store_global_sum(scale_idx, updated_sums[pair_idx])
