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
"""Conservation checks over a ledger reading.

A check earns its place only if some input could fail it. ``used = total -
free`` cannot: it restates its own definition and would hold against
corrupted data. Each check below therefore names the independent sources
whose disagreement it would catch.

Checks report rather than raise. A monitor that aborts the process it is
monitoring is worse than one that prints a suspicious number.
"""

from dataclasses import dataclass

__all__ = ["CheckResult", "check_kv_pool_conservation", "check_kv_quota"]


@dataclass(frozen=True)
class CheckResult:
    name: str
    passed: bool
    detail: str


def check_kv_pool_conservation(rows: list[dict]) -> list[CheckResult]:
    """Check C: a pool group's free and evictable slots fit inside its total.

    Non-vacuous because the three come from separate subsystems: ``free``
    from the slot allocator's free list (``storage/core.cpp:95``),
    ``evictable`` from the eviction controller's LRU container
    (``evictionController.cpp:238``), and ``total`` from the pool group's
    configured capacity (``storage/core.cpp:725``). Nothing in the code
    constrains the first two to sum within the third; double counting a block
    as both free and evictable breaks it.

    Deliberately reads only the stored fields. ``StorageStatistics`` also
    offers ``available()`` and ``unavailable()``, but they are derived from
    these same three, so a check written against them would pass by
    construction.

    ``SlotCount`` is ``int64_t``, so a violation shows up as a negative
    in-use count rather than a wrapped enormous one.
    """
    results = []
    for row in rows:
        total = row["total_slots"]
        free = row["free_slots"]
        evictable = row["evictable_slots"]
        in_use = total - free - evictable
        results.append(
            CheckResult(
                name=f"C:kv_pool_group[{row['pool_group_id']}]",
                passed=in_use >= 0,
                detail=(f"total={total} free={free} evictable={evictable} -> in_use={in_use}"),
            )
        )
    return results


def check_kv_quota(rows: list[dict], quota_bytes: int, tolerance: float = 0.05) -> CheckResult:
    """Check D: pool capacity against the quota it was sized from.

    The two are expected to differ: pools round up to an allocation
    granularity, so capacity lands at or slightly above quota. A capacity
    *below* quota, or far above it, means the sizing and the allocation
    disagree.

    Tolerance is one-sided on purpose -- rounding can only add.
    """
    capacity = sum(row["capacity_bytes"] for row in rows)
    if quota_bytes <= 0:
        return CheckResult(
            name="D:kv_quota",
            passed=True,
            detail=f"no quota to compare against (quota={quota_bytes})",
        )

    ratio = capacity / quota_bytes
    passed = 1.0 <= ratio <= 1.0 + tolerance
    return CheckResult(
        name="D:kv_quota",
        passed=passed,
        detail=(
            f"capacity={capacity} quota={quota_bytes} ratio={ratio:.4f} "
            f"(expected 1.0..{1.0 + tolerance:.2f}; rounding only adds)"
        ),
    )
