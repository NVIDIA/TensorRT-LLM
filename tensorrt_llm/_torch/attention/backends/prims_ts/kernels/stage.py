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

"""Kernel-local schedule-section tag for FMHA decode/context work bodies.

Some FMHA work methods need to know which schedule section (head/loop/tail) a
call belongs to.  Rather than depend on the task-scheduling framework's
``ScheduleStageType``, the FMHA schedules pass this small kernel-local enum
explicitly as a compile-time constant (the ``section`` work argument), so bodies
branch on it with ``cutlass.const_expr(section == FmhaStage.Loop)``.
"""

import enum


class FmhaStage(enum.Enum):
    """Schedule section of a single FMHA decode/context work call."""

    Head = 0
    Loop = 1
    Tail = 2
