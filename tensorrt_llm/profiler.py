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
"""Compatibility shim for ``tensorrt_llm.profiler``.

Will be removed once all usages are migrated to
``tensorrt_llm.observability.profiling``.

DO NOT ADD ANYTHING TO THIS FILE.
"""

import warnings

from tensorrt_llm.observability.profiling import (  # noqa: F401
    MemUnitType,
    Timer,
    bytes_to_target_unit,
    device_memory_info,
    elapsed_time_in_sec,
    host_memory_info,
    print_device_memory_usage,
    print_host_memory_usage,
    print_memory_usage,
    pynvml_context,
    reset,
    start,
    stop,
    summary,
)

warnings.warn(
    "tensorrt_llm.profiler has moved to "
    "tensorrt_llm.observability.profiling. The old path still works for "
    "now and will be removed in a future release.",
    FutureWarning,
    stacklevel=2,
)

__all__ = [
    "MemUnitType",
    "Timer",
    "bytes_to_target_unit",
    "device_memory_info",
    "elapsed_time_in_sec",
    "host_memory_info",
    "print_device_memory_usage",
    "print_host_memory_usage",
    "print_memory_usage",
    "pynvml_context",
    "reset",
    "start",
    "stop",
    "summary",
]
