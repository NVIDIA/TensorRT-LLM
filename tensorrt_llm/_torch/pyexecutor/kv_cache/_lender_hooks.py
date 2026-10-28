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
"""The only way managers, the scheduler and the executor call a lender (``_ManagerHooks``), and the
writers of their lending records; a lender reaches a manager only via ``sharing/_manager.py``."""

from __future__ import annotations

from typing import TYPE_CHECKING, Mapping, Protocol, Sequence, runtime_checkable

import numpy as np

from tensorrt_llm.logger import logger

if TYPE_CHECKING:
    from tensorrt_llm.runtime.kv_cache_manager_v2 import _KVCache

    from ..llm_request import LlmRequest
    from ..scheduler.scheduler_v2 import KVCacheV2Scheduler
    from .kv_cache_manager_v2 import KVCacheManagerV2


class _PageLockState(Protocol):
    """Live lock state a batch reads per request without another call: ``holding`` counts the
    unreleased locks holding a block; a cache whose ``id`` is not in ``locked_caches`` has no lock
    holding a block of it or capping its history; ``caps_history`` says whether any lock can cap a
    history; ``unreusable`` caches, by ``id``, commit nothing more for reuse."""

    holding: int
    locked_caches: Mapping[int, object]
    caps_history: bool
    unreusable: Mapping[int, object]


# What ``commit_gate`` answers: commit; hold the commit back for a retry at the unlock; or commit
# nothing more for reuse, stopping as usual.
COMMIT, HOLD, REFUSE = 0, 1, 2


@runtime_checkable
class _ManagerHooks(Protocol):
    """Lender callbacks on the manager thread for cache lifetime, page locks and history limits.

    Call order. Free, the cache out of the map: ``_on_free``, then ``_defer_close`` instead of a
    close. In-place shrink: ``_on_shrink`` right after. Early index-slot release: ``_parking_row``,
    the buffers for the cache's beam-0 page indices from then on. Right after a suspended cache
    resumed: ``_parked_row`` for those buffers if its slot was released early, then ``_on_resume``.
    Reuse reset: ``_before_reset``, the reuse tree's clearing, ``_on_reset``. Shutdown:
    ``_on_shutdown``, ``_defer_close`` per mapped cache, ``_keeps_pools``, ``_on_caches_closed``
    once the runtime's shutdown returned. ``_is_locked`` and ``_history_cap`` answer before a
    suspension, a commit or a history move, unless ``_page_lock_state``, live state a batch reads
    per request without a call, shows no lock on the cache or none capping a history.

    The ``_on_*`` notifications raise nothing: they log their errors, some of which stop the lender
    for good. ``_before_reset`` raises ``RuntimeError`` before any reset takes place. The queries
    raise into the manager whatever goes wrong."""

    def _on_free(self, request_id: int, kv_cache: _KVCache) -> None: ...

    def _on_shrink(self, request_id: int, kv_cache: _KVCache) -> None: ...

    def _on_resume(self, request_id: int) -> None: ...

    def _parking_row(self, request_id: int, kv_cache: _KVCache) -> Sequence[memoryview] | None: ...

    def _parked_row(self, request_id: int) -> Sequence[memoryview] | None: ...

    def _before_reset(self) -> None: ...

    def _on_reset(self) -> None: ...

    def _on_shutdown(self) -> None: ...

    def _on_caches_closed(self) -> None: ...

    def _is_locked(
        self, kv_cache: _KVCache | None, begin: int = 0, end: int | None = None
    ) -> bool: ...

    def _defer_close(self, kv_cache: _KVCache) -> bool: ...

    def _keeps_pools(self) -> bool: ...

    def _history_cap(self, kv_cache: _KVCache) -> int | None: ...

    def _page_lock_state(self) -> _PageLockState: ...


def generation_lock_state(manager: KVCacheManagerV2) -> _PageLockState | None:
    """The lender's live lock state, or None without a lender; queries propagate errors."""
    return None if manager._sharing is None else manager._sharing._page_lock_state()


def history_lock_state(manager: KVCacheManagerV2) -> _PageLockState | None:
    """The lender's live lock state where a lock can cap a cache's history, else None, when
    ``lent_history`` caps nothing; queries propagate errors."""
    if manager._sharing is None:
        return None
    state = manager._sharing._page_lock_state()
    return state if state.caps_history else None


def on_create(manager: KVCacheManagerV2, request_id: int) -> None:
    """Discard commits held for an earlier cache allocation of this request."""
    if manager._sharing is None or not manager._held_commits:
        return
    for key, held in tuple(manager._held_commits.items()):
        if held[1].py_request_id == request_id:
            del manager._held_commits[key]


def on_free(manager: KVCacheManagerV2, request_id: int, kv_cache: _KVCache) -> None:
    """Tell the lender, if any, that the request's cache left the map."""
    if manager._sharing is not None:
        manager._sharing._on_free(request_id, kv_cache)


def defer_close(manager: KVCacheManagerV2, kv_cache: _KVCache) -> bool:
    """Whether the lender keeps the cache open, to close it at its last unlock."""
    return manager._sharing is not None and manager._sharing._defer_close(kv_cache)


def on_shrink(manager: KVCacheManagerV2, request_id: int, kv_cache: _KVCache) -> None:
    """Tell the lender, if any, that the request's cache may have shrunk in place."""
    if manager._sharing is not None:
        manager._sharing._on_shrink(request_id, kv_cache)


def on_resume(manager: KVCacheManagerV2, request_id: int) -> None:
    """Tell the lender, if any, that the request's cache resumed."""
    if manager._sharing is not None:
        manager._sharing._on_resume(request_id)


def parking_row(
    manager: KVCacheManagerV2, request_id: int, kv_cache: _KVCache
) -> Sequence[memoryview] | None:
    """Lender-owned buffers, one per pool, for the cache's beam-0 page indices after the request's
    index slot's early release; None if no lender gives them, the cache keeping them itself."""
    if manager._sharing is None:
        return None
    return manager._sharing._parking_row(request_id, kv_cache)


def parked_row(manager: KVCacheManagerV2, request_id: int) -> Sequence[memoryview] | None:
    """The buffers ``parking_row`` gave for the request, to bind again after its cache resumed;
    None without them."""
    return None if manager._sharing is None else manager._sharing._parked_row(request_id)


def before_reset(manager: KVCacheManagerV2) -> None:
    """Let the lender, if any, refuse a reset of the manager's reuse state: ``RuntimeError``."""
    if manager._sharing is not None:
        manager._sharing._before_reset()


def on_reset(manager: KVCacheManagerV2) -> None:
    """Tell the lender, if any, that the manager cleared its reuse tree."""
    if manager._sharing is not None:
        manager._sharing._on_reset()


def on_shutdown(manager: KVCacheManagerV2) -> None:
    """Tell the lender, if any, that the manager shuts down."""
    if manager._sharing is not None:
        manager._sharing._on_shutdown()


def keeps_pools(manager: KVCacheManagerV2) -> bool:
    """Whether the lender keeps the pools allocated past the manager's shutdown."""
    return manager._sharing is not None and manager._sharing._keeps_pools()


def on_caches_closed(manager: KVCacheManagerV2) -> None:
    """Notify the lender, if any, after runtime shutdown returns."""
    if manager._sharing is not None:
        manager._sharing._on_caches_closed()


def check_suspendable(manager: KVCacheManagerV2, kv_cache: _KVCache) -> None:
    """Refuse suspension while a lender locks the cache."""
    if manager._sharing is not None and manager._sharing._is_locked(kv_cache):
        raise RuntimeError(
            f"the KV cache of request {kv_cache.id} cannot be suspended while a lender locks "
            "its blocks"
        )


def commit_gate(
    manager: KVCacheManagerV2, kv_cache: _KVCache, request: LlmRequest, end: int
) -> int:
    """``REFUSE`` for a cache the lender keeps out of reuse; ``HOLD``, registering a retry, where
    locks prevent committing or advancing history through ``end``; else ``COMMIT``. While neither a
    lock nor a refusal names the cache, the lender is asked nothing."""
    if manager._sharing is None:
        return COMMIT
    state = manager._sharing._page_lock_state()
    key = id(kv_cache)
    if state.unreusable and key in state.unreusable:
        return REFUSE
    if not state.holding or key not in state.locked_caches:
        return COMMIT
    tokens_per_block = manager.tokens_per_block
    begin = kv_cache.num_committed_tokens // tokens_per_block
    locked = manager._sharing._is_locked(kv_cache, begin, -(-end // tokens_per_block))
    if not locked and manager._lent_history(kv_cache, end) >= end:
        return COMMIT
    held = manager._held_commits.get(key)
    if held is not None and held[1] is request:
        end = held[2]
        fetch_touched_suffix = held[3]
    else:
        fetch_touched_suffix = False
    manager._held_commits[key] = (kv_cache, request, end, fetch_touched_suffix)
    logger.warning(
        f"KVCacheManagerV2: request {kv_cache.id}: commit through token {end} skipped while "
        "a lender locks blocks it covers or its window would pass"
    )
    return HOLD


def retry_held_commit(manager: KVCacheManagerV2, kv_cache: _KVCache) -> None:
    """Retry after unlock, before deferred close, unless the request now uses another cache."""
    held = manager._held_commits.pop(id(kv_cache), None)
    if held is None:
        return
    request, end, fetch_touched_suffix = held[1:]
    current = manager.kv_cache_map.get(request.py_request_id)
    if current is None or current is kv_cache:
        stopped = manager._retry_held_commit(request, kv_cache, end, fetch_touched_suffix)
        retried = manager._held_commits.get(id(kv_cache))
        if fetch_touched_suffix and retried is not None and retried[1] is request:
            manager._held_commits[id(kv_cache)] = (*retried[:3], True)
        if stopped and current is kv_cache and manager.conversation_manager is not None:
            manager.conversation_manager.save_drop_plan(request, kv_cache)


def drop_held_commit(manager: KVCacheManagerV2, kv_cache: _KVCache, start: int) -> None:
    """Drop the cache's held-back commit when it covers computed tokens from ``start`` on, which a
    fetch from ``start`` may overwrite; the request then commits from the bounds it resumes at."""
    held = manager._held_commits.get(id(kv_cache))
    if held is None:
        return
    end = manager._held_commit_drop_end(held)
    if end <= start:
        manager._held_commits[id(kv_cache)] = (held[0], held[1], held[2], True)
        return
    del manager._held_commits[id(kv_cache)]
    logger.warning(
        f"KVCacheManagerV2: request {kv_cache.id}: commit through token "
        f"{end} dropped, as a fetch from token {start} may "
        "overwrite what it covers"
    )


def lent_history(manager: KVCacheManagerV2, kv_cache: _KVCache, history: int) -> int:
    """Cap history at locked pages without rewinding the current history; while no lock holds a
    block of the cache, or no lock can cap a history, nothing caps it."""
    if manager._sharing is None:
        return history
    state = manager._sharing._page_lock_state()
    if not state.caps_history or not state.holding or id(kv_cache) not in state.locked_caches:
        return history
    cap = manager._sharing._history_cap(kv_cache)
    return history if cap is None else max(min(history, cap), kv_cache.history_length)


def holds_page_locks(manager: KVCacheManagerV2, request_id: int | None) -> bool:
    """Whether the request, or any cache when request_id is None, holds lender locks."""
    if manager._sharing is None:
        return False
    if request_id is None:
        return manager._sharing._is_locked(None)
    kv_cache = manager.kv_cache_map.get(request_id)
    return kv_cache is not None and manager._sharing._is_locked(kv_cache)


def scheduler_holds_page_locks(scheduler: KVCacheV2Scheduler, req: LlmRequest | None) -> bool:
    """Check both target and draft managers before reclaiming pages."""
    from .kv_cache_manager_v2 import KVCacheManagerV2

    rid = None if req is None else req.py_request_id
    if scheduler.kv_cache_manager._holds_page_locks(rid):
        return True
    draft = scheduler.draft_kv_cache_manager
    return isinstance(draft, KVCacheManagerV2) and draft._holds_page_locks(rid)


def fresh_page_record(
    manager: KVCacheManagerV2, kv_cache: _KVCache
) -> dict[int, np.ndarray] | None:
    """The cache's base page of each block, per pool, as the fresh-page fill records the pages it
    gave; ``None`` with the fill off."""
    if manager._fresh_page_fill is None:
        return None
    count = int(kv_cache.num_blocks)
    return {
        pool: np.frombuffer(
            kv_cache.get_base_page_indices(pool), dtype=np.int32, count=count
        ).copy()
        for pool in range(int(manager.num_pools))
    }


def fill_fresh_pages_past(
    manager: KVCacheManagerV2, request_id: int, record: dict[int, np.ndarray] | None
) -> None:
    """Run the fresh-page fill on the request's pages that ``record``, from ``fresh_page_record``,
    lacks: the fill takes the recorded pages as given and leaves them alone."""
    if record is None:
        return
    manager._fresh_pages_filled[int(request_id)] = record
    manager._fill_fresh_kv_pages(request_id)
