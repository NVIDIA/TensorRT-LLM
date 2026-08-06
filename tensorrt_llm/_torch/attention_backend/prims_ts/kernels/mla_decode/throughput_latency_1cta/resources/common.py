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

"""Common resource state and persistent-work throttling for MLA decode."""

from dataclasses import dataclass
from typing import ClassVar, Optional

import cutlass
import cutlass.cute as cute
from cutlass import Int32
from cutlass.experimental.task_scheduling.memory import SmemAllocation, TmemAllocation
from cutlass.experimental.task_scheduling.resources import (
    MemoryResource,
    StageInfo,
    TaskLocalVariable,
    consumer_work,
    producer_work,
)

from ..config import MlaConfig

# BF16 tcgen05 SMEM descriptors in this schedule use 128B-row swizzled tiles.
# The stride is one swizzle group, and the 16 KiB leading offset selects the
# second 64-wide K block inside the staged 128-token K/V tile.
TCGEN05_BF16_SWIZZLE_STRIDE_BYTES = 1024
TCGEN05_BF16_SECOND_K_BLOCK_OFFSET_BYTES = 16384
TCGEN05_BF16_K_BLOCK_WIDTH = 64


def _install_task_local_specs(resource: object, specs: tuple[tuple, ...]) -> None:
    """Install TaskLocalVariable fields declared by resource classes."""
    for spec in specs:
        field_name, dtype, default, docs = spec[:4]
        runtime_slot_name = spec[4] if len(spec) > 4 else None
        object.__setattr__(
            resource,
            field_name,
            TaskLocalVariable(
                dtype=dtype,
                default=default,
                docs=docs,
                runtime_slot_name=runtime_slot_name,
            ),
        )


# =====================================================================
# MlaResource — Shared SMEM/TMEM allocation helpers
# =====================================================================


@dataclass(kw_only=True)
class MlaResource(MemoryResource):
    """Base resource that owns common MLA config and task-local declarations."""

    cfg: cutlass.Constexpr[MlaConfig] = None
    cu_seqlens_q: object = None
    _alloc: cutlass.Constexpr[Optional[SmemAllocation]] = None
    _tmem_alloc: cutlass.Constexpr[Optional[TmemAllocation]] = None
    _tmem_base_addr: object = None
    _task_local_specs: ClassVar[tuple[tuple, ...]] = ()

    def __post_init__(self) -> None:
        _install_task_local_specs(self, self._task_local_specs)

    def get_smem_requirements(self):
        """Return SMEM allocations required by this resource."""
        return []

    def get_tmem_requirements(self):
        """Return TMEM allocations required by this resource."""
        return []

    @cute.jit
    def _init_tmem_state(self, stage_info: StageInfo) -> None:
        """Initialize shared resource state from the TS allocation context."""
        context = stage_info.context
        if cutlass.const_expr(
            context is not None
            and context.tmem_ptr_i32 is not None
            and self._tmem_alloc is not None
        ):
            self._tmem_base_addr = Int32(context.tmem_ptr_i32.load()) + Int32(
                self._tmem_alloc.offset
            )


# =====================================================================
# ScheduleTokenThrottleResource — Dynamic scheduler pacing marker
# =====================================================================


@dataclass(kw_only=True)
class ScheduleTokenThrottleResource(MemoryResource):
    """No-op named throttle resource for dynamic persistent schedule token pacing."""

    # Producer side: the load task marks that its schedule token slot can be reused.
    # There is no data payload; the resource exists to make the ordering edge
    # visible to TS.
    @producer_work
    @cute.jit
    def publish_schedule_token(self, stage_info: StageInfo):
        """Signal that the load task has yielded its schedule token slot."""
        del stage_info

    # Consumer side: the scheduler task consumes the marker before it advances
    # to the next CLC work tile.
    @consumer_work
    @cute.jit
    def consume_schedule_token(self, stage_info: StageInfo):
        """Wait-side marker before the scheduler reuses a schedule token slot."""
        del stage_info
