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
"""Conservation checks over a monitor reading.

A check earns its place only if some input could fail it. ``used = total -
free`` cannot: it restates its own definition and would hold against
corrupted data. Each check below therefore names the independent sources
whose disagreement it would catch.

Checks report rather than raise. A monitor that aborts the process it is
monitoring is worse than one that prints a suspicious number.

An informational result describes the reading without judging it; reports
list it as ``info`` and leave it out of the pass/fail count.
"""

from dataclasses import dataclass

from .names import UNTAGGED

__all__ = [
    "CheckResult",
    "check_kv_pool_conservation",
    "check_tag_coverage",
]


@dataclass(frozen=True)
class CheckResult:
    """One check's outcome."""

    name: str
    passed: bool
    detail: str
    #: Describes the reading rather than testing it; never counted as a pass.
    informational: bool = False


def check_kv_pool_conservation(rows: list[dict], name: str = "kv_pool_group") -> list[CheckResult]:
    """Check C: a pool group's free and evictable slots fit inside its total.

    Non-vacuous because the three come from separate subsystems: ``free``
    from the slot allocator (``SlotAllocator::numFreeSlots``), ``evictable``
    from the eviction controller's LRU container, and ``total`` from the pool
    group's capacity (``PoolGroupBase::numSlots``). Nothing in the code
    constrains the first two to sum within the third; double counting a block
    as both free and evictable breaks it.

    ``name`` labels the results, ``C:<name>[<pool group>]``, so the pools of
    several managers (target, draft, cross) stay apart.

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
                name=f"C:{name}[{row['pool_group_id']}]",
                passed=in_use >= 0,
                detail=(f"total={total} free={free} evictable={evictable} -> in_use={in_use}"),
            )
        )
    return results


def check_tag_coverage(by_tag: dict[str, int], torch_live_bytes: int) -> list[CheckResult]:
    """B: how much of PyTorch's live memory the startup tags account for.

    Informational. In a report both figures come from one allocator
    snapshot -- the tags split its live blocks -- so ``tagged <= live``
    holds by construction there and cannot flag a mislabelled block; the
    comparison only means something when ``torch_live_bytes`` is read
    separately. The second result says how much live memory no tag claims:
    activations are never tagged, so it is expected to be non-zero, and a
    tag breakdown read without it looks complete when it is not.
    """
    if not by_tag:
        return [
            CheckResult(
                name="B:tag_coverage",
                passed=True,
                detail="tagging unsupported or no window recorded; not reported as zero",
                informational=True,
            )
        ]

    tagged = sum(size for label, size in by_tag.items() if label != UNTAGGED)
    untagged = by_tag.get(UNTAGGED, 0)

    results = [
        CheckResult(
            name="B:tag_coverage",
            passed=tagged <= torch_live_bytes,
            detail=f"tagged={tagged} torch_live={torch_live_bytes}",
            informational=True,
        )
    ]

    total = tagged + untagged
    share = (untagged / total) if total else 0.0
    results.append(
        CheckResult(
            name="B:tag_untagged_share",
            passed=True,
            detail=(
                f"untagged={untagged} of {total} ({share:.1%}); expected non-zero in steady state"
            ),
            informational=True,
        )
    )
    return results
