# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""PDL resources with explicit compiler ordering for predecessor memory."""

from dataclasses import dataclass

import cutlass.cute as cute
from cutlass.experimental.task_scheduling.resources import (
    PdlLaunchBarrier as _PdlLaunchBarrier,
    PdlWaitBarrier as _PdlWaitBarrier,
    StageInfo,
    consumer_work,
    producer_work,
)


@dataclass(kw_only=True)
class PdlWaitBarrier(_PdlWaitBarrier):
    """Keep predecessor-dependent loads behind the grid wait."""

    @consumer_work
    @cute.jit
    def wait_griddep(self, stage_info: StageInfo) -> None:
        # The NVVM intrinsic carries no-memory effects. The CuTe PTX wrapper
        # also supplies a compiler memory clobber, so routed-metadata loads
        # cannot move ahead of predecessor completion.
        cute.arch.griddepcontrol_wait()


@dataclass(kw_only=True)
class PdlLaunchBarrier(_PdlLaunchBarrier):
    """Keep the launch hint ordered with the preceding PDL wait and loads."""

    @producer_work
    @cute.jit
    def launch_griddep(self, stage_info: StageInfo) -> None:
        cute.arch.griddepcontrol_launch_dependents()
