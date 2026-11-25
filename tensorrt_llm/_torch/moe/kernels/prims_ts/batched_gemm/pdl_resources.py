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
        # The NVVM intrinsic has no memory effects. The CuTe PTX wrapper also
        # supplies a compiler memory clobber, so predecessor-dependent loads
        # cannot move ahead of predecessor completion.
        cute.arch.griddepcontrol_wait()


@dataclass(kw_only=True)
class PdlLaunchBarrier(_PdlLaunchBarrier):
    """Keep the launch hint ordered with the preceding PDL wait and loads."""

    @producer_work
    @cute.jit
    def launch_griddep(self, stage_info: StageInfo) -> None:
        cute.arch.griddepcontrol_launch_dependents()
