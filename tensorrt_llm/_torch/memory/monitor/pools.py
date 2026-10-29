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
"""Which PyTorch pool a block lives in.

``memory_reserved()`` covers every pool -- the default one, CUDA graph
pools and ``torch.cuda.MemPool``s -- and they are not interchangeable. Free
space in the default pool can serve the next allocation; free space in a
graph pool is held for that graph's replay and serves nothing else. Calling
both "free, reusable" overstates the headroom by the size of every captured
graph.

Snapshot segments carry ``segment_pool_id``, but graph pools and MemPools
draw ids from the same counter (measured on torch 2.14: graph handles
``(0, 1)``, ``(0, 2)``, then ``MemPool().id == (0, 3)``), so the id alone
does not say which is which. MemPools are created in a few known places,
which register their ids here; every other private pool in this code base
comes from graph capture and is reported as one.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "DEFAULT_POOL",
    "PoolUsage",
    "classify",
    "clear_registered_pools",
    "pool_usage",
    "register_pool",
]

DEFAULT_POOL = (0, 0)

#: MemPool id -> what it is for.
_mem_pools: dict[tuple[int, int], str] = {}


def register_pool(pool_id, label: str) -> None:
    """Record that ``pool_id`` is a ``torch.cuda.MemPool`` serving ``label``.

    Cheap enough to call unconditionally from the code creating the pool.
    """
    try:
        _mem_pools[tuple(pool_id)] = label
    except TypeError:
        pass


def clear_registered_pools() -> None:
    """Forget every registration. For tests."""
    _mem_pools.clear()


def classify(pool_id) -> tuple[str, str]:
    """``(kind, label)`` with kind one of ``default``, ``mempool``, ``graph``."""
    key = tuple(pool_id)
    if key == DEFAULT_POOL:
        return "default", "default pool"
    if key in _mem_pools:
        return "mempool", _mem_pools[key]
    return "graph", "CUDA graph pool"


@dataclass(frozen=True)
class PoolUsage:
    pool_id: tuple[int, int]
    kind: str
    label: str
    reserved: int
    allocated: int

    @property
    def free(self) -> int:
        return max(self.reserved - self.allocated, 0)


def pool_usage(segments) -> list[PoolUsage]:
    """Reserved and allocated bytes per pool, from snapshot segments."""
    totals: dict[tuple[int, int], list[int]] = {}
    for segment in segments:
        key = tuple(segment.get("segment_pool_id", DEFAULT_POOL))
        acc = totals.setdefault(key, [0, 0])
        acc[0] += segment["total_size"]
        acc[1] += segment["allocated_size"]
    usage = []
    for key, (reserved, allocated) in totals.items():
        kind, label = classify(key)
        usage.append(PoolUsage(key, kind, label, reserved, allocated))
    usage.sort(key=lambda p: -p.reserved)
    return usage
