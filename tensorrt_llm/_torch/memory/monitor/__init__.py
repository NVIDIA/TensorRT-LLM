# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""GPU memory monitor: where did the device memory go?

``memory_monitor_config`` in the LLM arguments, or ``TLLM_MEM_MONITOR``,
selects a mode (see :func:`mode`). By default only an out-of-memory error
that reaches the executor's error handling produces a report, and nothing
is recorded or hooked in the meantime; full mode also records startup,
reports when it completes and watches the whole process for out-of-memory
errors. The monitor follows one executor per process.

Usage::

    from tensorrt_llm._torch.memory.monitor import capture, render_capture

    snapshot = capture(tag="after_weights")
    if snapshot is not None:
        print(render_capture(snapshot))

Reading memory never changes it: the monitor never calls ``empty_cache()``,
never synchronises, and never resets peak stats --
``_util.configure_kv_cache_capacity`` resets the global peak, and a second
resetter would corrupt its measurements. In full mode it does configure
PyTorch's allocation-history recording during startup (see ``tagging``).
"""

from .capability import Capability
from .capture import MonitorCapture, capture, render_capture
from .schema import Basis, Category, Entry, Placement, Source
from .settings import (
    ENV_VAR,
    RANK_ENV_VAR,
    STACKS_ENV_VAR,
    Mode,
    configure,
    current_rank,
    full_mode,
    full_mode_in,
    full_mode_requested,
    mode,
    oom_reports_enabled,
    selected_ranks,
    setting,
    stacks_requested,
)

__all__ = [
    "ENV_VAR",
    "RANK_ENV_VAR",
    "STACKS_ENV_VAR",
    "Basis",
    "Capability",
    "Category",
    "Entry",
    "Mode",
    "MonitorCapture",
    "Placement",
    "Source",
    "capture",
    "configure",
    "current_rank",
    "full_mode",
    "full_mode_in",
    "full_mode_requested",
    "mode",
    "oom_reports_enabled",
    "render_capture",
    "selected_ranks",
    "setting",
    "stacks_requested",
]
