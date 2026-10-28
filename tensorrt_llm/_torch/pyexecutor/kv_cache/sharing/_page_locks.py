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
"""Per-cache block locks on the manager thread. A STATE lock holds block 0 and defers every commit,
which snapshots its private page; writes may lock STATE past committed history."""

from __future__ import annotations

import enum
import types
from typing import TYPE_CHECKING, Mapping, NamedTuple, Optional, Sequence

import numpy as np

from . import _manager
from ._types import HeldPages, HoldReason, _integer

if TYPE_CHECKING:
    import weakref

    import torch

    from tensorrt_llm.runtime.kv_cache_manager_v2 import _KVCache

_NO_BLOCKS = np.zeros(0, dtype=np.int64)
_NO_BLOCKS.flags.writeable = False


class LockMode(enum.Enum):
    """How a lock holds its blocks: per block, one writer or any number of readers."""

    READ = "read"
    WRITE = "write"


class PageLock:
    """A lock on blocks of one runtime cache, and the event, if any, its holder waits for before
    any access to their pages. Holds at most that one event, dropped once it reports completion."""

    __slots__ = ("_kv", "_mode", "_blocks", "_ready", "_released", "_holds")

    def __init__(
        self,
        kv: _KVCache,
        mode: LockMode,
        blocks: dict[int, np.ndarray],
        ready: Optional[torch.cuda.Event],
    ) -> None:
        self._kv: _KVCache | None = kv
        self._mode = mode
        self._blocks = blocks
        self._ready: torch.cuda.Event | None = ready
        self._released = False
        # Whether it holds a block, which ``PageLocks.holding`` counts.
        self._holds = any(map(len, blocks.values()))

    @property
    def mode(self) -> LockMode:
        """``LockMode.READ`` or ``LockMode.WRITE``."""
        return self._mode

    @property
    def blocks(self) -> Mapping[int, np.ndarray]:
        """Layer-group index -> sorted, distinct, read-only ``int64`` block ordinals."""
        return types.MappingProxyType(self._blocks)

    @property
    def released(self) -> bool:
        """Whether ``PageLocks.unlock`` ended the lock."""
        return self._released

    def is_ready(self) -> bool:
        """Whether the lock is held and its event, if any, has completed; asks the event without
        waiting, and not again once it reported completion."""
        if self._released:
            return False
        if self._ready is not None and self._ready.query():
            self._ready = None
        return self._ready is None


class CheckedBlocks(NamedTuple):
    """The blocks ``PageLocks.lock`` would take of a cache in a mode, as it takes them, and the
    registry's version when ``PageLocks.check`` found so."""

    kv: object
    mode: LockMode
    blocks: dict[int, np.ndarray]
    version: int


class _CacheLocks:
    """One runtime cache's locks, whether it is closed at its last unlock, and the pages per pool
    group it added to its hold reason's counts when they were last counted."""

    __slots__ = ("kv", "locks", "close_at_last_unlock", "held", "counted_in")

    def __init__(self, kv: _KVCache) -> None:
        self.kv = kv
        self.locks: list[PageLock] = []
        self.close_at_last_unlock = False
        self.held: dict[int, int] = {}
        # The registry's counts ``held`` is part of, or ``None``.
        self.counted_in: dict[int, int] | None = None


def _ordinals(layer_group: int, values: Sequence[int]) -> np.ndarray:
    """``values`` as sorted, distinct, read-only ``int64`` block ordinals."""
    array = np.asarray(values)
    if array.size == 0:
        return _NO_BLOCKS
    if array.ndim != 1 or not np.issubdtype(array.dtype, np.integer):
        raise ValueError(
            f"layer group {layer_group}: block ordinals must be integers in one dimension, got "
            f"{array.dtype} {array.shape}"
        )
    if array.min() < 0:
        raise ValueError(f"layer group {layer_group}: block ordinals must be non-negative")
    unique = np.unique(array.astype(np.int64))
    unique.flags.writeable = False
    return unique


class PageLocks:
    """The locks on the runtime caches of one manager, which is referenced weakly, and the locked
    caches handed over to be closed at their last unlock."""

    def __init__(
        self,
        manager: weakref.ref,
        recurrent: Sequence[bool] = (),
        pool_group_of: Optional[Sequence[int]] = None,
    ) -> None:
        self._manager_ref = manager
        # By ``id`` of the runtime cache, which the entry holds, so the id stays its own.
        self._caches: dict[int, _CacheLocks] = {}
        # The same entries, for the manager's lender hooks (``_PageLockState.locked_caches``).
        self.locked_caches: Mapping[int, object] = self._caches
        # Whether a lock can cap a cache's history: only a lock on a windowed layer group's block,
        # which the lender's layout may not have.
        self.caps_history = True
        self.version = 0
        # Unreleased locks that hold a block, of any cache.
        self.holding = 0
        # The pool group of each layer group, as given or as the manager answers at the first need.
        self._pool_group_of: list[int] | None = (
            None if pool_group_of is None else list(pool_group_of)
        )
        # The layer groups of recurrent state, by their flags in layer-group order.
        self._recurrent = frozenset(lg for lg, flag in enumerate(recurrent) if flag)
        # Pages per pool group of open caches' locks and of caches closing at their last unlock, as
        # last counted; held caches locked, unlocked or freed since then wait in ``_dirty``.
        self._locked_pages: dict[int, int] = {}
        self._deferred_pages: dict[int, int] = {}
        self._dirty: dict[int, _CacheLocks] = {}
        self._held_version = 0
        self._held: HeldPages | None = None
        # Caches kept out of reuse, by ``id``, each entry holding its cache until the cache closes.
        self.unreusable: dict[int, object] = {}

    def lock(
        self,
        kv: _KVCache,
        blocks: Mapping[int, Sequence[int]],
        mode: LockMode,
        level: Optional[int] = None,
        *,
        ready: Optional[torch.cuda.Event] = None,
        ordered: bool = False,
        checked: Optional[CheckedBlocks] = None,
        committed: Optional[Sequence[int]] = None,
    ) -> PageLock:
        """Lock an open cache's blocks (``level=None``): per block one writer or many readers,
        writes only private uncommitted blocks (STATE row 0 aside); refusals raise, changing
        nothing. ``ordered``: sorted, distinct, non-negative, read-only ``int64`` arrays in no
        lease's view, taken as is; ``checked``: as ``check`` states."""
        if level is not None:
            raise ValueError(f"only level=None, in place, is supported, got level={level!r}")
        if not isinstance(mode, LockMode):
            raise TypeError(f"mode must be a LockMode, got {type(mode).__name__}")
        manager = self._manager_ref()
        if manager is None:
            raise RuntimeError("the KV cache manager is gone")
        if _manager.is_closed(kv):
            raise ValueError("the cache is closed")
        cache = self._caches.get(id(kv))
        if (
            checked is not None
            and checked.kv is kv
            and checked.mode is mode
            and checked.version == self.version
        ):
            if cache is not None and cache.close_at_last_unlock:
                raise ValueError("the cache was freed and stays open only for the locks it holds")
            wanted = checked.blocks
            if mode is LockMode.WRITE:
                self._check_write_past_committed(kv, wanted, self._pool_groups())
        elif (
            ordered
            and checked is None
            and cache is None
            and (mode is LockMode.READ or (committed is not None and not any(committed)))
        ):
            # No lock holds the cache and a write has no committed block to start past, so only
            # the layer-group count can refuse ordered blocks.
            if self._pool_group_of is None:
                self._pool_group_of = _manager.layer_group_pool_indices(manager)
            wanted = dict(blocks)
            if wanted and max(wanted) >= len(self._pool_group_of):
                raise ValueError(
                    f"layer group {max(wanted)}: the manager has {len(self._pool_group_of)} "
                    "layer groups"
                )
        else:
            # A checked write may have been invalidated by a lock change and a commit. Its
            # caller's committed snapshot no longer establishes which blocks remain private.
            live_committed = (
                None
                if checked is not None and checked.kv is kv and mode is LockMode.WRITE
                else committed
            )
            wanted = self._wanted(manager, kv, blocks, mode, ordered, live_committed)
        if cache is None:
            cache = self._caches[id(kv)] = _CacheLocks(kv)
        lock = PageLock(kv, mode, wanted, ready)
        cache.locks.append(lock)
        self.version += 1
        self.holding += lock._holds
        if not cache.close_at_last_unlock:
            self._dirty[id(cache)] = cache
        return lock

    def lock_fresh(
        self,
        kv: _KVCache,
        blocks: dict[int, np.ndarray],
        mode: LockMode,
        ready: Optional[torch.cuda.Event],
    ) -> PageLock | None:
        """``lock`` of a fast lend's ``blocks``, the dict taken as is: ``ordered`` blocks of the
        manager's layer groups, a write's past its committed ones, which nothing refuses while no
        lock holds ``kv``; else ``None``, locking nothing, for ``lock`` to decide."""
        caches = self._caches
        if _manager.is_closed(kv):
            return None
        if id(kv) in caches:
            return None
        lock = PageLock(kv, mode, blocks, ready)
        cache = caches[id(kv)] = _CacheLocks(kv)
        cache.locks.append(lock)
        self.version += 1
        self.holding += lock._holds
        if lock._holds:
            self._dirty[id(cache)] = cache
        return lock

    def refusal(
        self, kv: _KVCache, blocks: Mapping[int, Sequence[int]], mode: LockMode
    ) -> str | None:
        """Why ``lock`` would refuse ``blocks`` of ``kv`` in ``mode`` now, or ``None``; changes
        nothing."""
        found = self.check(kv, blocks, mode)
        return found if isinstance(found, str) else None

    def check(
        self,
        kv: _KVCache,
        blocks: Mapping[int, Sequence[int]],
        mode: LockMode,
        *,
        ordered: bool = False,
        committed: Optional[Sequence[int]] = None,
    ) -> str | CheckedBlocks:
        """Why ``lock`` would refuse ``blocks`` of ``kv`` in ``mode`` now, or, for its ``checked``,
        what it would take of them, valid while no lock, unlock or commit comes between; changes
        nothing. ``committed``: ``kv``'s ``_manager.committed_by_group``, if read."""
        manager = self._manager_ref()
        if manager is None:
            return "the KV cache manager is gone"
        if _manager.is_closed(kv):
            return "the cache is closed"
        try:
            wanted = self._wanted(manager, kv, blocks, mode, ordered, committed)
        except ValueError as error:
            return str(error)
        return CheckedBlocks(kv, mode, wanted, self.version)

    def _wanted(
        self,
        manager: object,
        kv: _KVCache,
        blocks: Mapping[int, Sequence[int]],
        mode: LockMode,
        ordered: bool = False,
        committed: Optional[Sequence[int]] = None,
    ) -> dict[int, np.ndarray]:
        """``blocks`` as checked ordinals per layer group, ``ordered`` ones taken as they are;
        ``ValueError`` for what ``lock`` refuses. ``committed`` as for ``check``."""
        if self._pool_group_of is None:
            self._pool_group_of = _manager.layer_group_pool_indices(manager)
        cache = self._caches.get(id(kv))
        if cache is not None and cache.close_at_last_unlock:
            raise ValueError("the cache was freed and stays open only for the locks it holds")
        if ordered:
            # Keyed by ``int`` layer-group indices, as the lender keys what it locks.
            wanted = dict(blocks)
            if wanted and max(wanted) >= len(self._pool_group_of):
                raise ValueError(
                    f"layer group {max(wanted)}: the manager has {len(self._pool_group_of)} "
                    "layer groups"
                )
        else:
            wanted = {}
            for layer_group, values in blocks.items():
                lg = _integer("a layer-group index", layer_group, 0)
                if lg >= len(self._pool_group_of):
                    raise ValueError(
                        f"layer group {lg}: the manager has {len(self._pool_group_of)} layer groups"
                    )
                wanted[lg] = _ordinals(lg, values)
        # With no committed token, every block is past the committed whole blocks.
        if mode is LockMode.WRITE and (committed is None or any(committed)):
            self._check_write_past_committed(kv, wanted, self._pool_group_of, committed)
        for held in cache.locks if cache is not None else ():
            if LockMode.WRITE not in (mode, held.mode):
                continue
            for lg, ordinals in wanted.items():
                both = np.intersect1d(ordinals, held._blocks.get(lg, _NO_BLOCKS))
                if len(both):
                    raise ValueError(
                        f"layer group {lg}: blocks {both[:8]} are locked for "
                        f"{held.mode.value}, which a {mode.value} lock conflicts with"
                    )
        return wanted

    def _check_write_past_committed(
        self,
        kv: _KVCache,
        wanted: Mapping[int, np.ndarray],
        pool_group_of: Sequence[int],
        committed: Optional[Sequence[int]] = None,
    ) -> None:
        """Refuse writes to whole committed blocks, reading current counts unless given a snapshot."""
        count = len(pool_group_of)
        whole = (
            _manager.committed_blocks(kv, count)
            if committed is None
            else _manager.committed_blocks(kv, count, committed)
        )
        for lg, ordinals in wanted.items():
            held = whole[lg]
            earlier = [int(ordinal) for ordinal in ordinals if ordinal < held]
            if lg not in self._recurrent and earlier:
                raise ValueError(
                    f"layer group {lg}: a write lock covers only blocks past the {held} "
                    f"committed whole blocks, got blocks {earlier[:8]}"
                )

    def unlock(self, lock: PageLock) -> bool:
        """Release idempotently, retrying a held-back commit before a last unlock closes the cache;
        whether it retried one. ``ValueError`` for an unreleased lock not held here."""
        if lock._released:
            return False
        key = id(lock._kv)
        cache = self._caches.get(key)
        # A lock equals only itself.
        if cache is None or lock not in cache.locks:
            raise ValueError("the lock is not held in this registry")
        self.version += 1
        self.holding -= lock._holds
        lock._released = True
        lock._ready = None
        lock._kv = None
        locks = cache.locks
        cache.locks = [] if len(locks) == 1 else [held for held in locks if held is not lock]
        if not cache.close_at_last_unlock:
            self._dirty[id(cache)] = cache
        manager = self._manager_ref()
        kv = cache.kv
        retried = False
        try:
            if manager is not None and _manager.has_held_commit(manager, kv):
                retried = True
                _manager.commit_held_back(manager, kv)
        finally:
            if not cache.locks:
                if cache.close_at_last_unlock:
                    _manager.close_cache(kv)
                    self.unreusable.pop(key, None)
                del self._caches[key]
                self._forget(cache)
        return retried

    def is_locked(self, kv: _KVCache, begin: int = 0, end: Optional[int] = None) -> bool:
        """Whether a lock holds a ``kv`` block in ``[begin, end)`` (``None``: no end) in any group,
        or a row of recurrent state whatever the range."""
        cache = self._caches.get(id(kv))
        if cache is None:
            return False
        for lock in cache.locks:
            for lg, ordinals in lock._blocks.items():
                if lg in self._recurrent and len(ordinals):
                    return True
                first = np.searchsorted(ordinals, begin)
                if first < len(ordinals) and (end is None or ordinals[first] < end):
                    return True
        return False

    def holds_any(self) -> bool:
        """Whether a lock holds a block of any cache, open or handed over by ``defer_close``."""
        return self.holding > 0

    def lowest_locked_blocks(self, kv: _KVCache, floors: Sequence[int]) -> dict[int, int]:
        """Layer-group index -> the lowest block ordinal of ``kv`` a lock holds there at or past
        ``floors[lg]``, for each layer group where a lock holds one."""
        lowest: dict[int, int] = {}
        cache = self._caches.get(id(kv))
        for lock in cache.locks if cache is not None else ():
            for lg, ordinals in lock._blocks.items():
                at = int(np.searchsorted(ordinals, floors[lg]))
                if at < len(ordinals):
                    lowest[lg] = min(lowest.get(lg, int(ordinals[at])), int(ordinals[at]))
        return lowest

    def defer_close(self, kv: _KVCache) -> bool:
        """Whether a lock keeps ``kv`` open, held here and closed at its last unlock, its pages
        counted as ``DEFERRED_CLOSE`` until then; asked at a free or shutdown instead of a close."""
        if not self.is_locked(kv):
            self.unreusable.pop(id(kv), None)
            return False
        cache = self._caches[id(kv)]
        if cache.close_at_last_unlock:
            return True
        cache.close_at_last_unlock = True
        self.version += 1
        self._dirty[id(cache)] = cache
        return True

    def holds_open_caches(self) -> bool:
        """Whether a cache handed over by ``defer_close`` is still open."""
        return any(cache.close_at_last_unlock for cache in self._caches.values())

    def keep_out_of_reuse(self, kv: _KVCache) -> None:
        """Keep ``kv`` from committing anything more for reuse until it closes."""
        self.unreusable[id(kv)] = kv

    def held_page_counts(self) -> HeldPages:
        """The pages the caches here hold by reason, the same value while no count changes; only
        caches locked, unlocked or freed since the last call are counted again."""
        if self._dirty:
            self._count_dirty()
        held = self._held
        if held is not None and held.version == self._held_version:
            return held
        locked = dict(self._locked_pages)
        deferred = dict(self._deferred_pages)
        unreclaimable = dict(locked)
        for group, pages in deferred.items():
            unreclaimable[group] = unreclaimable.get(group, 0) + pages
        by_reason = {}
        if locked:
            by_reason[HoldReason.LOCKED] = types.MappingProxyType(locked)
        if deferred:
            by_reason[HoldReason.DEFERRED_CLOSE] = types.MappingProxyType(deferred)
        # TODO: pages shared with other caches cannot be told apart, so deduct stays an upper bound.
        held = self._held = HeldPages(
            types.MappingProxyType(deferred),
            types.MappingProxyType(unreclaimable),
            types.MappingProxyType(by_reason),
            self._held_version,
        )
        return held

    def _pool_groups(self) -> list[int]:
        """The pool group of each layer group, asked of the manager at the first need."""
        if self._pool_group_of is None:
            self._pool_group_of = _manager.layer_group_pool_indices(self._manager_ref())
        return self._pool_group_of

    def _locked_pages_of(self, cache: _CacheLocks) -> dict[int, int]:
        """Per pool group, the distinct blocks ``cache``'s locks hold; one lock's are distinct."""
        pool_group_of = self._pool_groups()
        counts: dict[int, int] = {}
        locks = cache.locks
        if len(locks) == 1:
            for lg, ordinals in locks[0]._blocks.items():
                if len(ordinals):
                    group = pool_group_of[lg]
                    counts[group] = counts.get(group, 0) + len(ordinals)
            return counts
        by_layer_group: dict[int, list[np.ndarray]] = {}
        for lock in locks:
            for lg, ordinals in lock._blocks.items():
                if len(ordinals):
                    by_layer_group.setdefault(lg, []).append(ordinals)
        for lg, arrays in by_layer_group.items():
            pages = len(arrays[0]) if len(arrays) == 1 else len(np.unique(np.concatenate(arrays)))
            group = pool_group_of[lg]
            counts[group] = counts.get(group, 0) + pages
        return counts

    def _count_dirty(self) -> None:
        """Count again the pages of each held cache locked, unlocked or freed since the last count,
        a cache closing at its last unlock's as every page it holds."""
        dirty, self._dirty = self._dirty, {}
        for cache in dirty.values():
            if cache.close_at_last_unlock:
                self._recount(cache, self._deferred_pages, self._deferred_pages_of(cache))
            else:
                self._recount(cache, self._locked_pages, self._locked_pages_of(cache))

    def _forget(self, cache: _CacheLocks) -> None:
        """Drop a cache whose last lock ended from the counts, at once where it was counted, so no
        closed cache stays referenced here until a query."""
        self._dirty.pop(id(cache), None)
        if cache.counted_in is not None:
            self._recount(cache, None, {})

    def _deferred_pages_of(self, cache: _CacheLocks) -> dict[int, int]:
        """Per pool group, every page of the runtime cache ``cache`` closes at its last unlock."""
        held: dict[int, int] = {}
        for lg, group in enumerate(self._pool_groups()):
            pages = _manager.held_pages(cache.kv, lg, recurrent=lg in self._recurrent)
            if pages:
                held[group] = held.get(group, 0) + pages
        return held

    def _recount(
        self, cache: _CacheLocks, counts: dict[int, int] | None, held: dict[int, int]
    ) -> None:
        """Count ``held`` as ``cache``'s pages in ``counts`` (``None``: nowhere) in place of what it
        counted before; the version moves if that changes anything."""
        before, was = cache.held, cache.counted_in
        if before == held and (was is counts or not held):
            return
        if was is not None:
            for group, pages in before.items():
                left = was[group] - pages
                if left:
                    was[group] = left
                else:
                    del was[group]
        if counts is not None:
            for group, pages in held.items():
                counts[group] = counts.get(group, 0) + pages
        cache.held = held if counts is not None else {}
        cache.counted_in = counts if held and counts is not None else None
        self._held_version += 1
