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
"""Serialization for the per-iteration PEFT (LoRA) cache statistics.

The payload comes from ``BasePeftCacheManager.get_and_reset_iteration_stats()``,
which reports adapter ownership transitions and page-cache pressure for the
window since the previous drain.
"""

from typing import Any, Optional

# Transition counters, cleared by the drain that reports them.
_PEFT_COUNTER_KEYS = (
    ("requestsPaused", "requests_paused"),
    ("requestsResumed", "requests_resumed"),
    ("requestsTerminated", "requests_terminated"),
    ("tasksReleasedDevice", "tasks_released_device"),
    ("tasksReleasedHost", "tasks_released_host"),
    ("tasksEvictedDevice", "tasks_evicted_device"),
    ("pagesEvictedDevice", "pages_evicted_device"),
    ("tasksEvictedHost", "tasks_evicted_host"),
    ("pagesEvictedHost", "pages_evicted_host"),
)

# Gauges, sampled at drain time.
_PEFT_GAUGE_KEYS = (
    ("devicePagesTotal", "device_pages_total"),
    ("devicePagesAvailable", "device_pages_available"),
    ("hostPagesTotal", "host_pages_total"),
    ("hostPagesAvailable", "host_pages_available"),
    ("deviceTasksInProgress", "device_tasks_in_progress"),
    ("deviceTasksDone", "device_tasks_done"),
    ("activeTasks", "active_tasks"),
    ("pausedTasks", "paused_tasks"),
)


def serialize_peft_cache_iteration_stats(stats: Any) -> dict:
    """Render one ``PeftCacheIterationStats`` as JSON-ready camelCase fields."""
    return {
        json_key: getattr(stats, attr)
        for json_key, attr in (*_PEFT_COUNTER_KEYS, *_PEFT_GAUGE_KEYS)
    }


def append_peft_cache_iteration_stats(stats_dict: dict, peft_iter_stats: Optional[Any]) -> None:
    """Attach ``peftCacheIterationStats`` to an iteration-stats dict.

    A ``None`` payload means either no PEFT manager (no LoRA in this
    deployment) or an iteration on which stats were not drained, so the key is
    left absent rather than zero-filled -- absent and "nothing happened" are
    different states for a consumer.
    """
    if peft_iter_stats is None:
        return
    stats_dict["peftCacheIterationStats"] = serialize_peft_cache_iteration_stats(peft_iter_stats)
