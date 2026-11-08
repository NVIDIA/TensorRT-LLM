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

"""Per-case timing primitives.

The benchmark runs three timing variants over the same MoE module:

* :mod:`bench_moe.timing.autotune`   -- untimed pre-pass that warms the kernel
  caches before any iteration is timed.
* :mod:`bench_moe.timing.eager`      -- eager ``moe.forward`` timing via
  ``torch.cuda.Event`` plus an optional Kineto profiler pass for kernel
  breakdown.
* :mod:`bench_moe.timing.cuda_graph` -- CUDA Graph capture+replay timing with
  graph-internal external CUDA events.
* :mod:`bench_moe.timing.cupti`      -- optional CUPTI activity tracking used
  by the CUDA Graph path to get a kernel-level breakdown for replays that
  Kineto cannot see.
"""

from .autotune import _run_autotune
from .cuda_graph import _time_moe_forward_cuda_graph
from .cupti import (
    _build_cuda_graph_kernel_stats_cupti,
    _CuptiContext,
    _demangle_names,
    _try_init_cupti,
)
from .eager import (
    _kernel_times_to_summary_list,
    _l2_flush_buffer,
    _parse_profiler_events_moe,
    _time_moe_forward_eager,
)
from .nsys import _NsysProfiler, measured_range

__all__ = [
    "_CuptiContext",
    "_NsysProfiler",
    "_build_cuda_graph_kernel_stats_cupti",
    "_demangle_names",
    "_kernel_times_to_summary_list",
    "_l2_flush_buffer",
    "_parse_profiler_events_moe",
    "_run_autotune",
    "_time_moe_forward_cuda_graph",
    "_time_moe_forward_eager",
    "_try_init_cupti",
    "measured_range",
]
