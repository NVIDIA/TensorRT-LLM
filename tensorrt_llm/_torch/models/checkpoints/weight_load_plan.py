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
from enum import Enum
from typing import Sequence, TypeAlias


class WeightLoadPolicy(str, Enum):
    """Storage-to-model placement policy selected for checkpoint loading."""

    DIRECT_RANK_READ = "direct_rank_read"
    SHARED_HOST_PRODUCER = "shared_host_producer"
    RANK_COOPERATIVE_STREAM = "rank_cooperative_stream"
    SINGLE_PRODUCER_PAGE_CACHE_PREFETCH = "single_producer_page_cache_prefetch"
    GPU_BROADCAST = "gpu_broadcast"
    LEGACY_FALLBACK = "legacy_fallback"


WeightLoadPlan: TypeAlias = tuple[WeightLoadPolicy, ...]

DEFAULT_WEIGHT_LOAD_PLAN: WeightLoadPlan = (
    WeightLoadPolicy.DIRECT_RANK_READ,
    WeightLoadPolicy.SHARED_HOST_PRODUCER,
    WeightLoadPolicy.GPU_BROADCAST,
    WeightLoadPolicy.LEGACY_FALLBACK,
)


def normalize_weight_load_plan(
    value: WeightLoadPolicy | str | Sequence[WeightLoadPolicy | str] | None,
) -> WeightLoadPlan:
    """Normalize one strict policy or an ordered fallback sequence."""
    if value is None:
        return DEFAULT_WEIGHT_LOAD_PLAN
    if isinstance(value, WeightLoadPolicy):
        values = [value]
    elif isinstance(value, str):
        values = [item.strip() for item in value.split(",")]
    else:
        values = list(value)

    if not values or any(not item for item in values):
        raise ValueError("Weight-load plan must contain at least one policy")

    try:
        plan = tuple(WeightLoadPolicy(item) for item in values)
    except ValueError as error:
        supported = ", ".join(policy.value for policy in WeightLoadPolicy)
        raise ValueError(f"Unsupported weight-load policy; expected one of: {supported}") from error

    if len(set(plan)) != len(plan):
        raise ValueError("Weight-load plan must not contain duplicate policies")
    return plan
