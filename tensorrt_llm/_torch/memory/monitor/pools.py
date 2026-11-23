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
draw ids from the same counter (graph handles ``(0, 1)``, ``(0, 2)``, then
``MemPool().id == (0, 3)``), so the id alone does not say which is which. torch makes private pools in exactly two ways
-- graph capture and ``torch.cuda.MemPool`` -- so a pool is classified as a
MemPool if a live ``MemPool`` object has its id, and as a graph pool
otherwise. That holds for MemPools created by code that never registers
here, such as the external library behind GMS weight sharing. Registration
only supplies a readable name.

Looking for live ``MemPool`` objects walks the garbage collector's heap, so
each pool is classified once, when first seen, and remembered. Pool ids are
never reused and a ``MemPool`` exists before any of its segments, so the
first answer stays right -- even after the ``MemPool`` object is collected
while its blocks are still in use. :func:`prime` classifies every pool that
exists when startup completes, so the walk happens before serving rather
than inside the first report; it runs again only for a pool created later.
If the walk fails, the new pools are reported as unclassified for that
report and classified on the next one.
"""

from __future__ import annotations

import gc
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterable, Iterator

import torch

from tensorrt_llm.logger import logger

__all__ = [
    "DEFAULT_POOL",
    "PoolUsage",
    "classify",
    "classify_all",
    "clear_registered_pools",
    "forget_released_tags",
    "is_released",
    "is_tag_released",
    "mark_materialized",
    "mark_released",
    "pool_usage",
    "prime",
    "register_pool",
    "released_tags_pinned",
]

DEFAULT_POOL = (0, 0)

#: MemPool id -> what it is for.
_mem_pools: dict[tuple[int, int], str] = {}
#: MemPool id -> the sleep-mode tag whose memory it holds.
_sleep_tags: dict[tuple[int, int], str] = {}
#: Sleep-mode tags whose physical memory is currently released. torch keeps
#: counting a released pool in ``memory_reserved()`` while the GPU no longer
#: backs it, so without this a sleeping process looks larger than it is and
#: the residual goes negative.
_released_tags: set[str] = set()
#: Sleep-mode tags the monitored executor allocated memory under. A tag
#: released by another executor in the process says nothing about the
#: monitored executor's memory unless that executor has memory under it too:
#: ``releaseWithTag`` releases every blob of the tag in the process.
_monitored_sleep_tags: set[str] = set()
#: Guards ``_released_tags``: the sleep path changes it on its own thread
#: while a report reads it.
_released_lock = threading.Lock()
#: Per thread, the released set a report pinned for its whole reading.
_pinned = threading.local()
#: Pools already classified as graph or unregistered MemPool.
_seen: dict[tuple[int, int], tuple[str, str]] = {}


def register_pool(pool_id: Iterable[int], label: str, sleep_tag: str | None = None) -> None:
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
        # Imported here: the providers package imports this module.
        from .providers import unmonitored_executor_current

        if not unmonitored_executor_current():
            _monitored_sleep_tags.add(sleep_tag)


def mark_released(tags: Iterable[str]) -> None:
    """Record that sleep mode has released the memory of ``tags``."""
    with _released_lock:
        _released_tags.update(tags)


def mark_materialized(tags: Iterable[str]) -> None:
    """Record that the memory of ``tags`` is backed again."""
    with _released_lock:
        _released_tags.difference_update(tags)


def _released() -> frozenset[str]:
    """The released tags: the pinned set inside :func:`released_tags_pinned`, else the current one."""
    pinned = getattr(_pinned, "tags", None)
    if pinned is not None:
        return pinned
    with _released_lock:
        return frozenset(_released_tags)


@contextmanager
def released_tags_pinned() -> Iterator[None]:
    """Within this block, every release query on this thread reads one snapshot.

    A report reads the release state for the pools, the tags and the KV
    cache rows; sleep or wakeup on another thread in between would let one
    of them count a pool the others leave out, and the rows would no longer
    add up to the process total.
    """
    previous = getattr(_pinned, "tags", None)
    _pinned.tags = _released() if previous is None else previous
    try:
        yield
    finally:
        _pinned.tags = previous


def forget_released_tags() -> None:
    """Forget which tags are released.

    For when the executor that released them shuts down: its pools are gone,
    and a tag left marked would make the next executor's pools under the
    same tag look released.
    """
    with _released_lock:
        _released_tags.clear()
    _monitored_sleep_tags.clear()


def is_tag_released(tag: str) -> bool:
    """Whether sleep mode has released the monitored executor's memory under ``tag``.

    False for a tag the monitored executor has no sleep-mode memory under,
    such as ``kv_cache`` of an executor without ``sleep_config`` while
    another executor in the process sleeps.
    """
    return tag in _monitored_sleep_tags and tag in _released()


def is_released(pool_id: Iterable[int]) -> bool:
    """Whether ``pool_id`` is a sleep-mode pool whose memory is released."""
    tag = _sleep_tags.get(tuple(pool_id))
    return tag is not None and tag in _released()


def clear_registered_pools() -> None:
    """Forget every registration. For tests."""
    _mem_pools.clear()
    _sleep_tags.clear()
    _monitored_sleep_tags.clear()
    with _released_lock:
        _released_tags.clear()
    _seen.clear()


UNCLASSIFIED = ("unclassified", "unclassified")


def _live_mempool_ids() -> set[tuple[int, int]]:
    """Ids of every live ``torch.cuda.MemPool``.

    Walks the garbage collector's objects, which costs on the order of
    100 ms per million objects, so it runs only for a pool not seen before.
    The type is read with ``type()``, which never touches the object: a dead
    ``weakref.proxy`` raises on ``isinstance``, and a class with a custom
    ``__getattribute__`` may warn or raise on attribute access.
    """
    mem_pool = torch.cuda.MemPool
    ids = set()
    objects = gc.get_objects()
    try:
        for obj in objects:
            if not issubclass(type(obj), mem_pool):
                continue
            try:
                ids.add(tuple(obj.id))
            except (AttributeError, RuntimeError, TypeError):
                continue
    finally:
        # The list holds every tracked object; let them go promptly.
        del objects
    return ids


def classify(
    pool_id: Iterable[int], live_mempools: set | frozenset = frozenset()
) -> tuple[str, str]:
    """``(kind, label)`` with kind one of ``default``, ``mempool``, ``graph``."""
    key = tuple(pool_id)
    if key == DEFAULT_POOL:
        return "default", "default pool"
    if key in _mem_pools:
        return "mempool", _mem_pools[key]
    if key in live_mempools:
        return "mempool", "unnamed MemPool"
    return "graph", "CUDA graph pool"


def classify_all(pool_ids: Iterable) -> dict[tuple[int, int], tuple[str, str]]:
    """Classify several pools, looking for live MemPools only for new ones.

    If looking fails, the new pools are ``unclassified`` and are not
    remembered, so the next call tries again.
    """
    keys = {tuple(p) for p in pool_ids}
    new = [k for k in keys if k != DEFAULT_POOL and k not in _mem_pools and k not in _seen]
    if new:
        try:
            live = _live_mempool_ids()
        except Exception as exc:  # a diagnostic must not break the report
            logger.debug(f"memory monitor: could not classify pools ({type(exc).__name__}: {exc})")
            return {k: _seen.get(k) or (UNCLASSIFIED if k in new else classify(k)) for k in keys}
        for key in new:
            _seen[key] = classify(key, live)
    return {k: _seen.get(k) or classify(k) for k in keys}


def prime() -> None:
    """Classify every pool that exists now, so later reports need not walk.

    Meant for when startup has completed: CUDA graphs and MemPools made
    during startup are classified here instead of inside the first report,
    which may run on the executor thread in the middle of an OOM.
    """
    if not torch.cuda.is_available():
        return
    try:
        segments = torch.cuda.memory_snapshot()
    except RuntimeError as exc:
        logger.debug(f"memory monitor: no allocator snapshot to classify ({exc})")
        return
    classify_all(segment.get("segment_pool_id", DEFAULT_POOL) for segment in segments)


@dataclass(frozen=True)
class PoolUsage:
    """Reserved and allocated bytes of one PyTorch pool."""

    pool_id: tuple[int, int]
    kind: str
    label: str
    reserved: int
    allocated: int
    #: A sleep-mode pool whose physical memory is currently released.
    released: bool = False
    #: The sleep-mode tag whose memory the pool holds, if it is a sleep pool.
    sleep_tag: str | None = None

    @property
    def free(self) -> int:
        """Reserved but not handed out."""
        return max(self.reserved - self.allocated, 0)


def pool_usage(segments: Iterable[dict]) -> list[PoolUsage]:
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
        usage.append(
            PoolUsage(key, kind, label, reserved, allocated, is_released(key), _sleep_tags.get(key))
        )
    usage.sort(key=lambda p: -p.reserved)
    return usage
