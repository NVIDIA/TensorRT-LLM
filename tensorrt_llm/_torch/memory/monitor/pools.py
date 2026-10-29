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
does not say which is which. torch makes private pools in exactly two ways
-- graph capture and ``torch.cuda.MemPool`` -- so a pool is classified as a
MemPool if a live ``MemPool`` object has its id, and as a graph pool
otherwise. That holds for MemPools created by code that never registers
here, such as the external library behind GMS weight sharing. Registration
only supplies a readable name.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "DEFAULT_POOL",
    "PoolUsage",
    "classify",
    "classify_all",
    "clear_registered_pools",
    "is_released",
    "mark_materialized",
    "mark_released",
    "pool_usage",
    "register_pool",
]

DEFAULT_POOL = (0, 0)

#: MemPool id -> what it is for.
_mem_pools: dict[tuple[int, int], str] = {}
#: MemPool id -> the sleep-mode tag whose memory it holds.
_sleep_tags: dict[tuple[int, int], str] = {}
#: Sleep-mode tags whose physical memory is currently released. torch keeps
#: accounting for those pools -- measured: after release_with_tag the device
#: got 512 MiB back while memory_reserved() did not move -- so without this
#: a sleeping process looks larger than it is and the residual goes negative.
_released_tags: set[str] = set()


def register_pool(pool_id, label: str, sleep_tag: str | None = None) -> None:
    """Record that ``pool_id`` is a ``torch.cuda.MemPool`` serving ``label``.

    ``sleep_tag`` marks a sleep-mode pool, whose memory can be released and
    restored by tag. Cheap enough to call unconditionally from the code
    creating the pool.
    """
    try:
        key = tuple(pool_id)
    except TypeError:
        return
    _mem_pools[key] = label
    if sleep_tag is not None:
        _sleep_tags[key] = sleep_tag


def mark_released(tags) -> None:
    _released_tags.update(tags)


def mark_materialized(tags) -> None:
    _released_tags.difference_update(tags)


def is_released(pool_id) -> bool:
    tag = _sleep_tags.get(tuple(pool_id))
    return tag is not None and tag in _released_tags


def clear_registered_pools() -> None:
    """Forget every registration. For tests."""
    _mem_pools.clear()
    _sleep_tags.clear()
    _released_tags.clear()


def _live_mempool_ids() -> set[tuple[int, int]]:
    """Ids of every live ``torch.cuda.MemPool``.

    Walks the garbage collector's objects, which costs on the order of
    100 ms per million objects, so it runs only when a report finds a pool
    it cannot name otherwise.
    """
    import gc

    import torch

    ids = set()
    for obj in gc.get_objects():
        if isinstance(obj, torch.cuda.MemPool):
            try:
                ids.add(tuple(obj.id))
            except Exception:  # noqa: BLE001
                continue
    return ids


def classify(pool_id, live_mempools: set | frozenset = frozenset()) -> tuple[str, str]:
    """``(kind, label)`` with kind one of ``default``, ``mempool``, ``graph``."""
    key = tuple(pool_id)
    if key == DEFAULT_POOL:
        return "default", "default pool"
    if key in _mem_pools:
        return "mempool", _mem_pools[key]
    if key in live_mempools:
        return "mempool", "unnamed MemPool"
    return "graph", "CUDA graph pool"


def classify_all(pool_ids) -> dict[tuple[int, int], tuple[str, str]]:
    """Classify several pools, discovering live MemPools only if needed."""
    keys = {tuple(p) for p in pool_ids}
    unknown = [k for k in keys if k != DEFAULT_POOL and k not in _mem_pools]
    live = _live_mempool_ids() if unknown else frozenset()
    return {k: classify(k, live) for k in keys}


@dataclass(frozen=True)
class PoolUsage:
    pool_id: tuple[int, int]
    kind: str
    label: str
    reserved: int
    allocated: int
    #: A sleep-mode pool whose physical memory is currently released.
    released: bool = False

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
    kinds = classify_all(totals)
    usage = []
    for key, (reserved, allocated) in totals.items():
        kind, label = kinds[key]
        usage.append(PoolUsage(key, kind, label, reserved, allocated, is_released(key)))
    usage.sort(key=lambda p: -p.reserved)
    return usage
