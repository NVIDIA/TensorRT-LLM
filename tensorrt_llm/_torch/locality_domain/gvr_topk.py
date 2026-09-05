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
"""Pure row-sharding policy for Rubin locality-domain GVR Top-K."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

# These are conservative prototype thresholds, not Rubin architectural
# constants. They only reject workloads where two extra launches plus the
# fork/join events are clearly unlikely to amortize. Retune them from an R200
# BS/ISL sweep before enabling locality-domain GVR by default.
GVR_LOCALITY_MIN_TOTAL_SCORE_ELEMENTS = 1 << 20
GVR_LOCALITY_MIN_SCORE_ELEMENTS_PER_SM = 1 << 12


@dataclass(frozen=True, slots=True)
class GvrTopKRowShard:
    """One contiguous request-aligned row slice assigned to a locality domain."""

    partition_id: int
    request_start: int
    request_end: int
    row_start: int
    row_end: int
    num_sms: int

    @property
    def num_requests(self) -> int:
        return self.request_end - self.request_start

    @property
    def num_rows(self) -> int:
        return self.row_end - self.row_start


@dataclass(frozen=True, slots=True)
class GvrTopKRowShardPlan:
    """A two-domain decode plan whose row slices are disjoint and exhaustive."""

    shards: tuple[GvrTopKRowShard, GvrTopKRowShard]
    topology: tuple[tuple[int, int], tuple[int, int]]


def _validate_topology(
    topology: Sequence[tuple[int, int]],
) -> tuple[tuple[int, int], tuple[int, int]]:
    if len(topology) != 2:
        raise ValueError(f"GVR locality sharding requires 2 partitions, got {len(topology)}")

    normalized = tuple(
        (int(partition_sms), int(total_sms)) for partition_sms, total_sms in topology
    )
    if any(partition_sms <= 0 for partition_sms, _ in normalized):
        raise ValueError(f"partition SM counts must be positive, got {normalized}")
    if any(total_sms <= 0 or partition_sms > total_sms for partition_sms, total_sms in normalized):
        raise ValueError(f"invalid locality-domain topology {normalized}")
    if normalized[0][1] != normalized[1][1]:
        raise ValueError(f"locality domains disagree on the full-device SM count: {normalized}")
    if normalized[0][0] + normalized[1][0] > normalized[0][1]:
        raise ValueError(
            f"locality-domain partitions overlap the full-device topology: {normalized}"
        )
    return normalized[0], normalized[1]


def is_gvr_topk_locality_workload_large_enough(
    *,
    num_rows: int,
    next_n: int,
    score_width: int,
    top_k: int,
    min_total_score_elements: int = GVR_LOCALITY_MIN_TOTAL_SCORE_ELEMENTS,
) -> bool:
    """Apply the topology-independent part of the provisional gain gate."""
    return (
        next_n > 0
        and num_rows >= 2 * next_n
        and num_rows % next_n == 0
        and score_width > top_k
        and num_rows * score_width >= min_total_score_elements
    )


def plan_gvr_topk_row_shards(
    *,
    num_rows: int,
    next_n: int,
    score_width: int,
    top_k: int,
    topology: Sequence[tuple[int, int]],
    min_total_score_elements: int = GVR_LOCALITY_MIN_TOTAL_SCORE_ELEMENTS,
    min_score_elements_per_sm: int = GVR_LOCALITY_MIN_SCORE_ELEMENTS_PER_SM,
) -> GvrTopKRowShardPlan | None:
    """Plan two proportional, request-aligned decode row slices.

    ``num_rows`` contains ``next_n`` consecutive rows for every request. The
    split is therefore made in request space and then converted back to rows;
    this preserves the leaf GVR mapping ``request = local_row // next_n``.

    The workload checks are deliberately capture-stable: they use tensor
    geometry and the engine score-width envelope, never device KV lengths.
    They are a provisional overhead guard rather than a claim of measured
    Rubin speedup.
    """
    if num_rows < 0:
        raise ValueError(f"num_rows must be non-negative, got {num_rows}")
    if next_n < 1:
        raise ValueError(f"next_n must be positive, got {next_n}")
    if num_rows % next_n:
        raise ValueError(f"num_rows {num_rows} must be divisible by next_n {next_n}")
    if score_width < 0:
        raise ValueError(f"score_width must be non-negative, got {score_width}")
    if top_k < 1:
        raise ValueError(f"top_k must be positive, got {top_k}")
    if min_total_score_elements < 0 or min_score_elements_per_sm < 0:
        raise ValueError("GVR locality workload thresholds must be non-negative")

    normalized_topology = _validate_topology(topology)
    if not is_gvr_topk_locality_workload_large_enough(
        num_rows=num_rows,
        next_n=next_n,
        score_width=score_width,
        top_k=top_k,
        min_total_score_elements=min_total_score_elements,
    ):
        return None
    num_requests = num_rows // next_n

    sms_0 = normalized_topology[0][0]
    sms_1 = normalized_topology[1][0]
    partition_sms = sms_0 + sms_1
    # Nearest-integer proportional split, clamped so both domains receive at
    # least one complete request. This also handles asymmetric public splits.
    split_request = (num_requests * sms_0 + partition_sms // 2) // partition_sms
    split_request = min(max(split_request, 1), num_requests - 1)
    split_row = split_request * next_n

    shards = (
        GvrTopKRowShard(0, 0, split_request, 0, split_row, sms_0),
        GvrTopKRowShard(
            1,
            split_request,
            num_requests,
            split_row,
            num_rows,
            sms_1,
        ),
    )
    if any(
        shard.num_rows * score_width < min_score_elements_per_sm * shard.num_sms for shard in shards
    ):
        return None
    return GvrTopKRowShardPlan(shards=shards, topology=normalized_topology)


__all__ = [
    "GVR_LOCALITY_MIN_SCORE_ELEMENTS_PER_SM",
    "GVR_LOCALITY_MIN_TOTAL_SCORE_ELEMENTS",
    "GvrTopKRowShard",
    "GvrTopKRowShardPlan",
    "is_gvr_topk_locality_workload_large_enough",
    "plan_gvr_topk_row_shards",
]
