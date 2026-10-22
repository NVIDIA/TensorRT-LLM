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

"""TMEM resources for local and global softmax statistics."""

from dataclasses import dataclass
from typing import ClassVar, Optional

from cutlass.experimental import primitives as prims

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32
from cutlass.experimental.task_scheduling.enums import WorkAttr
from cutlass.experimental.task_scheduling.memory import TmemAllocation
from cutlass.experimental.task_scheduling.resources import (
    StageInfo,
    TaskLocalVariable,
    consumer_work,
    producer_work,
)

from ...helpers.constants import (
    TCGEN05_32B_SHAPE,
)

from ...helpers.layout import (
    _TASK_CACHE_TMEM_BASE_OFFSET,
    decode_gen_task_cache,
    num_softmax_scale_groups,
)
from ...helpers.math import (
    neg_max_f32,
)
from ...helpers.ops import (
    vector_from_scalars,
)

from .common import (
    MlaResource,
)

# =====================================================================
# TmemSoftmaxLocalResource — Local softmax stats in TMEM
# =====================================================================


@dataclass(kw_only=True)
class TmemSoftmaxLocalResource(MlaResource):
    """TMEM scratch resource for local softmax max/sum statistics."""

    _task_local_specs: ClassVar[tuple[tuple, ...]] = (
        ("old_max_arr", cutlass.Array, None, "Previous softmax maxima read from TMEM."),
        ("new_max_arr", cutlass.Array, None, "Current softmax maxima read from TMEM."),
        ("sum_arr", cutlass.Array, None, "Softmax denominators read from TMEM."),
        ("inst0_old_max_arr", cutlass.Array, None, "Instance 0 previous maxima."),
        ("inst0_new_max_arr", cutlass.Array, None, "Instance 0 current maxima."),
        ("inst0_sum_arr", cutlass.Array, None, "Instance 0 denominator sums."),
        ("inst1_old_max_arr", cutlass.Array, None, "Instance 1 previous maxima."),
        ("inst1_new_max_arr", cutlass.Array, None, "Instance 1 current maxima."),
        ("inst1_sum_arr", cutlass.Array, None, "Instance 1 denominator sums."),
    )
    inst_id: cutlass.Constexpr[int] = 0
    tmem_alias_ref: Optional[MlaResource] = None
    old_max_arr: cutlass.Constexpr[TaskLocalVariable] = (
        TaskLocalVariable.uninitialized()
    )
    new_max_arr: cutlass.Constexpr[TaskLocalVariable] = (
        TaskLocalVariable.uninitialized()
    )
    sum_arr: cutlass.Constexpr[TaskLocalVariable] = TaskLocalVariable.uninitialized()
    inst0_old_max_arr: cutlass.Constexpr[TaskLocalVariable] = (
        TaskLocalVariable.uninitialized()
    )
    inst0_new_max_arr: cutlass.Constexpr[TaskLocalVariable] = (
        TaskLocalVariable.uninitialized()
    )
    inst0_sum_arr: cutlass.Constexpr[TaskLocalVariable] = (
        TaskLocalVariable.uninitialized()
    )
    inst1_old_max_arr: cutlass.Constexpr[TaskLocalVariable] = (
        TaskLocalVariable.uninitialized()
    )
    inst1_new_max_arr: cutlass.Constexpr[TaskLocalVariable] = (
        TaskLocalVariable.uninitialized()
    )
    inst1_sum_arr: cutlass.Constexpr[TaskLocalVariable] = (
        TaskLocalVariable.uninitialized()
    )

    def get_tmem_requirements(self):
        """Return the TMEM allocation for local softmax statistics."""
        if self.tmem_alias_ref is not None:
            return []
        if self._tmem_alloc is None:
            num_stages = (
                self.pipeline_config.num_stages
                if self.cfg.kernel_variant == "keeps_mma_ab"
                and self.pipeline_config is not None
                else 1
            )
            self._tmem_alloc = TmemAllocation(
                name=f"{self.name}",
                num_columns=self.cfg.tmem_stats_cols * num_stages,
            )
        return [self._tmem_alloc]

    @cute.jit
    def _stats_base_addr(self, stage_info: StageInfo, task_cache):
        """Return the TMEM address that stores this stage's softmax stats."""
        if cutlass.const_expr(self.tmem_alias_ref is not None):
            return (
                task_cache[_TASK_CACHE_TMEM_BASE_OFFSET]
                + Int32(self.tmem_alias_ref._tmem_alloc.offset)
                + stage_info.stage_idx * Int32(self.cfg.tmem_s_cols)
            )
        stats_stage_offset = Int32(0)
        if cutlass.const_expr(self.cfg.kernel_variant == "keeps_mma_ab"):
            stats_stage_offset = stage_info.stage_idx * Int32(self.cfg.tmem_stats_cols)
        return (
            task_cache[_TASK_CACHE_TMEM_BASE_OFFSET]
            + Int32(self._tmem_alloc.offset)
            + stats_stage_offset
        )

    @cute.jit
    def _make_initial_stats_vars(self):
        """Create and initialize softmax statistic arrays."""
        num_scale_groups = num_softmax_scale_groups(self.cfg)
        old_max_arr = cutlass.Array(
            Float32, num_scale_groups, space=cutlass.AddressSpace.rmem
        )
        new_max_arr = cutlass.Array(
            Float32, num_scale_groups, space=cutlass.AddressSpace.rmem
        )
        sum_arr = cutlass.Array(
            Float32, num_scale_groups, space=cutlass.AddressSpace.rmem
        )
        inst0_old_max_arr = cutlass.Array(
            Float32,
            num_scale_groups,
            space=cutlass.AddressSpace.rmem,
        )
        inst0_new_max_arr = cutlass.Array(
            Float32,
            num_scale_groups,
            space=cutlass.AddressSpace.rmem,
        )
        inst0_sum_arr = cutlass.Array(
            Float32,
            num_scale_groups,
            space=cutlass.AddressSpace.rmem,
        )
        inst1_old_max_arr = cutlass.Array(
            Float32,
            num_scale_groups,
            space=cutlass.AddressSpace.rmem,
        )
        inst1_new_max_arr = cutlass.Array(
            Float32,
            num_scale_groups,
            space=cutlass.AddressSpace.rmem,
        )
        inst1_sum_arr = cutlass.Array(
            Float32,
            num_scale_groups,
            space=cutlass.AddressSpace.rmem,
        )
        for idx in cutlass.range_constexpr(num_scale_groups):
            old_max_arr[idx] = neg_max_f32()
            new_max_arr[idx] = neg_max_f32()
            sum_arr[idx] = Float32(0.0)
            inst0_old_max_arr[idx] = neg_max_f32()
            inst0_new_max_arr[idx] = neg_max_f32()
            inst0_sum_arr[idx] = Float32(0.0)
            inst1_old_max_arr[idx] = neg_max_f32()
            inst1_new_max_arr[idx] = neg_max_f32()
            inst1_sum_arr[idx] = Float32(0.0)
        return (
            old_max_arr,
            new_max_arr,
            sum_arr,
            inst0_old_max_arr,
            inst0_new_max_arr,
            inst0_sum_arr,
            inst1_old_max_arr,
            inst1_new_max_arr,
            inst1_sum_arr,
        )

    @consumer_work(
        work_attrs=WorkAttr.AUXILIARY,
        returns=(
            old_max_arr,
            new_max_arr,
            sum_arr,
            inst0_old_max_arr,
            inst0_new_max_arr,
            inst0_sum_arr,
            inst1_old_max_arr,
            inst1_new_max_arr,
            inst1_sum_arr,
        ),
    )
    @cute.jit
    def init_stats_state(self, stage_info: StageInfo):
        """Create statistic variables for the first work tile."""
        # Consumer aux work creates the arrays that hold stats loaded from
        # TMEM.  The arrays are later populated by the loop/tail stat loads.
        self._init_tmem_state(stage_info)
        return self._make_initial_stats_vars()

    @consumer_work(
        work_attrs=WorkAttr.AUXILIARY,
        returns=(
            old_max_arr,
            new_max_arr,
            sum_arr,
            inst0_old_max_arr,
            inst0_new_max_arr,
            inst0_sum_arr,
            inst1_old_max_arr,
            inst1_new_max_arr,
            inst1_sum_arr,
        ),
    )
    @cute.jit
    def init_stats_work_tile_state(self, stage_info: StageInfo):
        """Create statistic variables for each persistent work tile."""
        # Persistent CTAs reuse the same resource instance, so local stats must
        # be reset when the work tile changes.
        del stage_info
        return self._make_initial_stats_vars()

    @producer_work
    @cute.jit
    def store_loop_stats(
        self,
        stage_info: StageInfo,
        *,
        old_max_arr,
        new_max_arr,
        sum_arr,
        inst_idx: cutlass.Constexpr[int],
    ):
        """Store loop softmax statistics into TMEM for correction."""
        # Producer work for stats: softmax writes old/new max or sum/new max
        # snapshots into the acquired TMEM stats stage.  Correction consumes the
        # same stage through the matching loop/tail load function.
        cfg = self.cfg
        num_scale_groups = num_softmax_scale_groups(cfg)
        task_cache = decode_gen_task_cache(stage_info)
        stats_base = self._stats_base_addr(stage_info, task_cache)
        stats_ptr = prims.make_tmem_ptr(stats_base, Float32)
        skip_loop_stats_store = cutlass.const_expr(inst_idx == 0) and (
            stage_info.loop_end == Int32(1)
        )
        if not skip_loop_stats_store:
            if cutlass.const_expr(inst_idx == 0):
                stats = vector_from_scalars(
                    tuple(old_max_arr[idx] for idx in range(num_scale_groups))
                    + tuple(new_max_arr[idx] for idx in range(num_scale_groups)),
                    Float32,
                )
                prims.tcgen05_st(
                    TCGEN05_32B_SHAPE,
                    stats_ptr,
                    stats,
                )
                prims.tcgen05_wait(kind=prims.Tcgen05Wait.STORE)
                cute.arch.fence_view_async_tmem_store()
            elif cutlass.const_expr(inst_idx == 1):
                stats = vector_from_scalars(
                    tuple(sum_arr[idx] for idx in range(num_scale_groups))
                    + tuple(new_max_arr[idx] for idx in range(num_scale_groups)),
                    Float32,
                )
                prims.tcgen05_st(
                    TCGEN05_32B_SHAPE,
                    stats_ptr,
                    stats,
                )
                prims.tcgen05_wait(kind=prims.Tcgen05Wait.STORE)
                cute.arch.fence_view_async_tmem_store()

    @cute.jit
    def _stats_tuple(
        self,
        old_max_arr,
        new_max_arr,
        sum_arr,
        inst0_old_max_arr,
        inst0_new_max_arr,
        inst0_sum_arr,
        inst1_old_max_arr,
        inst1_new_max_arr,
        inst1_sum_arr,
    ):
        """Return the task-local stat arrays in schedule order."""
        return (
            old_max_arr,
            new_max_arr,
            sum_arr,
            inst0_old_max_arr,
            inst0_new_max_arr,
            inst0_sum_arr,
            inst1_old_max_arr,
            inst1_new_max_arr,
            inst1_sum_arr,
        )

    @cute.jit
    def _load_stats_payload(self, stage_info: StageInfo):
        """Load the raw stat payload from the current TMEM stats stage."""
        cfg = self.cfg
        task_cache = decode_gen_task_cache(stage_info)
        stats_base = self._stats_base_addr(stage_info, task_cache)
        stats_ptr = prims.make_tmem_ptr(stats_base, Float32)
        loaded = prims.tcgen05_ld(
            TCGEN05_32B_SHAPE,
            stats_ptr,
            num=num_softmax_scale_groups(cfg) * 2,
        )
        prims.tcgen05_wait(kind=prims.Tcgen05Wait.LOAD)
        cute.arch.fence_view_async_tmem_load()
        return loaded

    @cute.jit
    def _store_loop_stats_payload(
        self,
        loaded,
        old_max_arr,
        new_max_arr,
        sum_arr,
        inst0_old_max_arr,
        inst0_new_max_arr,
        inst0_sum_arr,
        inst1_old_max_arr,
        inst1_new_max_arr,
        inst1_sum_arr,
    ):
        """Update task-local arrays from loop old/new-max stats."""
        cfg = self.cfg
        for scale_idx in cutlass.range_constexpr(num_softmax_scale_groups(cfg)):
            loaded_new_max = loaded[num_softmax_scale_groups(cfg) + scale_idx]
            new_max_arr[scale_idx] = loaded_new_max
            loaded_old_max = loaded[scale_idx]
            old_max_arr[scale_idx] = loaded_old_max
            if cutlass.const_expr(self.inst_id == 0):
                inst0_old_max_arr[scale_idx] = loaded_old_max
                inst0_new_max_arr[scale_idx] = loaded_new_max
            else:
                inst1_old_max_arr[scale_idx] = loaded_old_max
                inst1_new_max_arr[scale_idx] = loaded_new_max
        return self._stats_tuple(
            old_max_arr,
            new_max_arr,
            sum_arr,
            inst0_old_max_arr,
            inst0_new_max_arr,
            inst0_sum_arr,
            inst1_old_max_arr,
            inst1_new_max_arr,
            inst1_sum_arr,
        )

    @cute.jit
    def _store_tail_stats_payload(
        self,
        loaded,
        old_max_arr,
        new_max_arr,
        sum_arr,
        inst0_old_max_arr,
        inst0_new_max_arr,
        inst0_sum_arr,
        inst1_old_max_arr,
        inst1_new_max_arr,
        inst1_sum_arr,
    ):
        """Update task-local arrays from tail sum/new-max stats."""
        cfg = self.cfg
        for scale_idx in cutlass.range_constexpr(num_softmax_scale_groups(cfg)):
            loaded_new_max = loaded[num_softmax_scale_groups(cfg) + scale_idx]
            loaded_sum = loaded[scale_idx]
            new_max_arr[scale_idx] = loaded_new_max
            sum_arr[scale_idx] = loaded_sum
            if cutlass.const_expr(self.inst_id == 0):
                inst0_new_max_arr[scale_idx] = loaded_new_max
                inst0_sum_arr[scale_idx] = loaded_sum
            else:
                inst1_new_max_arr[scale_idx] = loaded_new_max
                inst1_sum_arr[scale_idx] = loaded_sum
        return self._stats_tuple(
            old_max_arr,
            new_max_arr,
            sum_arr,
            inst0_old_max_arr,
            inst0_new_max_arr,
            inst0_sum_arr,
            inst1_old_max_arr,
            inst1_new_max_arr,
            inst1_sum_arr,
        )

    @consumer_work(
        returns=(
            old_max_arr,
            new_max_arr,
            sum_arr,
            inst0_old_max_arr,
            inst0_new_max_arr,
            inst0_sum_arr,
            inst1_old_max_arr,
            inst1_new_max_arr,
            inst1_sum_arr,
        ),
    )
    @cute.jit
    def load_initial_stats(
        self,
        stage_info: StageInfo,
        *,
        old_max_arr,
        new_max_arr,
        sum_arr,
        inst0_old_max_arr,
        inst0_new_max_arr,
        inst0_sum_arr,
        inst1_old_max_arr,
        inst1_new_max_arr,
        inst1_sum_arr,
    ):
        """Return initial stats before any loop-stage TMEM payload exists."""
        del stage_info
        return self._stats_tuple(
            old_max_arr,
            new_max_arr,
            sum_arr,
            inst0_old_max_arr,
            inst0_new_max_arr,
            inst0_sum_arr,
            inst1_old_max_arr,
            inst1_new_max_arr,
            inst1_sum_arr,
        )

    @consumer_work(
        returns=(
            old_max_arr,
            new_max_arr,
            sum_arr,
            inst0_old_max_arr,
            inst0_new_max_arr,
            inst0_sum_arr,
            inst1_old_max_arr,
            inst1_new_max_arr,
            inst1_sum_arr,
        ),
    )
    @cute.jit
    def load_loop_stats(
        self,
        stage_info: StageInfo,
        *,
        old_max_arr,
        new_max_arr,
        sum_arr,
        inst0_old_max_arr,
        inst0_new_max_arr,
        inst0_sum_arr,
        inst1_old_max_arr,
        inst1_new_max_arr,
        inst1_sum_arr,
    ):
        """Load loop old/new-max statistics for O rescaling."""
        loaded = self._load_stats_payload(stage_info)
        return self._store_loop_stats_payload(
            loaded,
            old_max_arr=old_max_arr,
            new_max_arr=new_max_arr,
            sum_arr=sum_arr,
            inst0_old_max_arr=inst0_old_max_arr,
            inst0_new_max_arr=inst0_new_max_arr,
            inst0_sum_arr=inst0_sum_arr,
            inst1_old_max_arr=inst1_old_max_arr,
            inst1_new_max_arr=inst1_new_max_arr,
            inst1_sum_arr=inst1_sum_arr,
        )

    @consumer_work(
        returns=(
            old_max_arr,
            new_max_arr,
            sum_arr,
            inst0_old_max_arr,
            inst0_new_max_arr,
            inst0_sum_arr,
            inst1_old_max_arr,
            inst1_new_max_arr,
            inst1_sum_arr,
        ),
    )
    @cute.jit
    def load_tail_stats(
        self,
        stage_info: StageInfo,
        *,
        old_max_arr,
        new_max_arr,
        sum_arr,
        inst0_old_max_arr,
        inst0_new_max_arr,
        inst0_sum_arr,
        inst1_old_max_arr,
        inst1_new_max_arr,
        inst1_sum_arr,
    ):
        """Load tail sum/new-max statistics for final normalization."""
        loaded = self._load_stats_payload(stage_info)
        return self._store_tail_stats_payload(
            loaded,
            old_max_arr=old_max_arr,
            new_max_arr=new_max_arr,
            sum_arr=sum_arr,
            inst0_old_max_arr=inst0_old_max_arr,
            inst0_new_max_arr=inst0_new_max_arr,
            inst0_sum_arr=inst0_sum_arr,
            inst1_old_max_arr=inst1_old_max_arr,
            inst1_new_max_arr=inst1_new_max_arr,
            inst1_sum_arr=inst1_sum_arr,
        )


# =====================================================================
# TmemSoftmaxGlobalResource — Global softmax dependency marker
# =====================================================================


@dataclass(kw_only=True)
class TmemSoftmaxGlobalResource(MlaResource):
    """Named dependency edge from local softmax stats to correction."""

    inst_id: cutlass.Constexpr[int] = 0

    @producer_work
    @cute.jit
    def track_global(self, stage_info: StageInfo):
        """No-op placeholder for the global softmax dependency edge."""
        # This resource has no payload.  Its producer work gives TS a named
        # dependency edge between the two softmax instances and correction.
        del stage_info
