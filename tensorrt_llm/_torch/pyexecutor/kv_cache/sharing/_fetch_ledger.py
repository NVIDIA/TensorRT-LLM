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
"""A staging lender's fetch ledger: per request, the latest fetch, what fetches delivered, the
prefix a fetch kept and a cache whose committed tail a failed copy may have mixed, and the
resume bounds they give."""

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

import numpy as np

from . import _manager, _ranges
from ._types import ResumeBounds

if TYPE_CHECKING:
    from tensorrt_llm.runtime.kv_cache_manager_v2 import _KVCache

    from ...llm_request import LlmRequest
    from ..kv_cache_manager_v2 import KVCacheManagerV2
    from ._layout import ManagerLayout
    from ._lender import _LentBlockEntries, _QueuedCopy, _StagingLease


@dataclass(eq=False)
class _FetchRecord:
    """One write lease's fetch into one cache: ``COPY_PENDING`` once the copy of its marked entries
    is queued without error, ``COPY_DONE`` once that copy has completed, or at once where its marks
    queue no copy and the lease has none. ``_FetchLedger._fetch_state`` derives its ``_FetchState``
    from ``marks_queued``, ``copy`` and whether ``_FetchLedger._latest_fetches`` holds it."""

    start: int
    kv: object
    marks_queued: bool = False
    copy: _QueuedCopy | None = None


class _FetchState(enum.Enum):
    """A fetch's state, which ``_FetchLedger._fetch_state`` derives.
    ``_FetchLedger._latest_fetches`` holds each request's latest fetch until it is abandoned, the
    request is freed, or resume_bounds, while the lender serves the manager and is not stopped,
    finds the request holding another cache. ``Staging._apply_marks`` takes it out
    (``_take_out_fetch``) before queuing the copy and puts it back (``_put_back_fetch``) with
    ``marks_queued`` set once the copy is queued without error."""

    # In _latest_fetches, marks not queued: resume_bounds is None and a new fetch into the cache fails
    # at the call. Staging._apply_marks sets marks_queued, or abandons it where the copy fails or
    # the call raises; _abandon_fetch abandons it too.
    AWAITING_MARKS = "awaiting_marks"
    # In _latest_fetches, its marked entries' copy queued without error and pending: as
    # AWAITING_MARKS for resume_bounds and a new fetch. The copy's completion makes it COPY_DONE.
    COPY_PENDING = "copy_pending"
    # In _latest_fetches, its copy complete or none queued: resume_bounds answers, and a new fetch into
    # the cache may start. The request's next fetch replaces it.
    COPY_DONE = "copy_done"
    # Out of _latest_fetches before its marks were queued: none of its entries count.
    ABANDONED = "abandoned"
    # Out of _latest_fetches after its marks were queued: the request's next fetch replaced it, the
    # request's free dropped it, or resume_bounds dropped it as a record of a cache the request no
    # longer holds. Replacing it drops none of its entries from the cache's deliveries.
    RETIRED = "retired"


@dataclass(eq=False)
class _Deliveries:
    """What the fetches into one cache delivered: per layer group, by block ordinal, the entries
    whose last copy from staging was queued without error and that no shrink freed since;
    ``lowest_start`` is the lowest fetch start."""

    kv: object
    lowest_start: int
    # Index by local layer group, then logical block ordinal (not LeaseView tuple positions).
    # Recorded when a copy queues successfully; resume bounds also wait for its completion.
    delivered_block_masks: list[np.ndarray]
    # (tokens computed before, max_position, the end of the last run below it), last computed
    resume_bounds_memo: tuple[int, int, int] | None = None


# TODO: each per-request record keeps its own reference to the cache it describes, so every
# reader compares it again, and a record of a replaced cache keeps that cache until the record is
# dropped or replaced.
class _FetchLedger:
    """A staging lender's per-request fetch records and the resume_bounds they give. ``Staging``
    changes the records through these methods alone and reads them through these methods, apart
    from the fields of the ``_Deliveries`` that ``_deliveries_of`` returns."""

    def __init__(
        self, layout: ManagerLayout, copy_done: Callable[[_QueuedCopy | None], bool]
    ) -> None:
        self._layout = layout
        # Whether a copy has completed, as the lender answers: each event asked at most once per
        # round of its progress.
        self._copy_done = copy_done
        # Per-request fetch records, each with the cache it describes: the request's free drops all
        # four, and resume_bounds drops one of a cache the request no longer holds.
        self._latest_fetches: dict[int, _FetchRecord] = {}  # the latest fetch into each request
        self._deliveries: dict[int, _Deliveries] = {}
        # Requests whose history was past the committed tokens after a fetch's grow: the cache it
        # grew, what the latest such fetch kept of the prefix computed before it, and the history
        # its grow left.
        self._kept_prefixes: dict[int, tuple[object, int, int]] = {}
        # Requests whose committed tail block a failed copy may have left with the fetch's bytes in
        # some pools and the bytes it held before in others: the cache. ResumeBounds is empty for it.
        self._mixed_tail_caches: dict[int, object] = {}

    def _record_fetch(
        self, request_id: int, kv: _KVCache, start: int, kept: int, position: int, committed: int
    ) -> _FetchRecord:
        """Record the request's latest fetch, from ``start`` into ``kv``, and, where its grow left
        the history at ``position`` past the ``committed`` tokens, ``kept``: what the fetch keeps of
        the prefix computed before it. Returns the fetch's record."""
        if position > committed:
            self._kept_prefixes[request_id] = (kv, kept, position)
        fetch = _FetchRecord(start, kv)
        self._latest_fetches[request_id] = fetch
        return fetch

    def _abandon_fetch(self, lease: _StagingLease) -> None:
        """Drop the write lease's fetch record: it delivered nothing, and resume_bounds counts what
        earlier fetches into the cache delivered."""
        if lease._fetch is not None and self._latest_fetches.get(lease._request_id) is lease._fetch:
            del self._latest_fetches[lease._request_id]

    def _unsettled_fetch_refusal(self, request_id: int, kv: _KVCache) -> str | None:
        """Why the request's earlier fetch into ``kv`` keeps a new one from starting, or ``None``:
        it has not settled on this rank, where the copy its marks queued may still be pending."""
        previous = self._latest_fetches.get(request_id)
        if previous is not None and previous.kv is kv and not self._fetch_settled(previous):
            return f"request {request_id} already has an unsettled fetch"
        return None

    def _fetch_settled(self, fetch: _FetchRecord) -> bool:
        """The fetch's arrived entries are marked and their copy into the request's pages is
        done."""
        return fetch.marks_queued and self._copy_done(fetch.copy)

    def _fetch_state(self, request_id: int, fetch: _FetchRecord) -> _FetchState:
        """The state of ``fetch``, a fetch into request ``request_id``, from ``_latest_fetches`` and
        ``_fetch_settled``; see ``_FetchState``."""
        if self._latest_fetches.get(request_id) is not fetch:
            return _FetchState.RETIRED if fetch.marks_queued else _FetchState.ABANDONED
        if self._fetch_settled(fetch):
            return _FetchState.COPY_DONE
        return _FetchState.COPY_PENDING if fetch.marks_queued else _FetchState.AWAITING_MARKS

    def _take_out_fetch(self, lease: _StagingLease, fetch: _FetchRecord | None) -> bool:
        """Take ``fetch``, the write lease's, out of the latest fetches before its marks' copy is
        queued, if it is still the request's latest; returns whether it was."""
        current = fetch is not None and self._latest_fetches.get(lease._request_id) is fetch
        if current:
            del self._latest_fetches[lease._request_id]
        return current

    def _invalidate_delivered_blocks(self, lease: _StagingLease, copied: list[np.ndarray]) -> None:
        """Stop counting the delivered entries ``copied`` selects, before their copy is queued: one
        that fails partway can leave a row holding each fetch's bytes in different pools.
        ``_record_delivered_blocks`` counts them again once the copy is queued without error."""
        delivered = self._deliveries.get(lease._request_id)
        if delivered is None or delivered.kv is not lease._kv:
            return
        for lg, block_ordinals, mask in zip(
            lease._block_entries.layer_group_indices, lease._block_entries.block_ordinals, copied
        ):
            blocks = delivered.delivered_block_masks[lg]
            entries = block_ordinals[mask]
            entries = entries[entries < len(blocks)]
            if blocks[entries].any():
                blocks[entries] = False
                delivered.resume_bounds_memo = None

    def _record_mixed_tail(self, request_id: int, kv: _KVCache) -> None:
        """Record ``kv`` as the request's cache whose committed tail block a failed copy may have
        left with the fetch's bytes in some pools and the bytes it held before in others."""
        self._mixed_tail_caches[request_id] = kv

    def _put_back_fetch(
        self, lease: _StagingLease, fetch: _FetchRecord, copied: list[np.ndarray]
    ) -> None:
        """Put ``fetch`` back as the request's latest once its marks' copy is queued without error,
        with ``marks_queued`` set and the lease's copy, and add the entries ``copied`` selects to
        what fetches delivered into the cache."""
        fetch.copy = lease._copy
        fetch.marks_queued = True
        self._latest_fetches[lease._request_id] = fetch
        self._record_delivered_blocks(lease._request_id, fetch, lease._block_entries, copied)

    def _record_delivered_blocks(
        self,
        request_id: int,
        fetch: _FetchRecord,
        entries: _LentBlockEntries,
        copied: list[np.ndarray],
    ) -> None:
        """Add a fetch's copied entries to what earlier fetches into the same cache delivered, so a
        fetch split into consecutive leases counts as one."""
        delivered = self._deliveries.get(request_id)
        if delivered is None or delivered.kv is not fetch.kv:
            empty = [np.zeros(0, dtype=bool) for _ in range(self._layout.num_layer_groups)]
            delivered = _Deliveries(fetch.kv, fetch.start, empty)
            self._deliveries[request_id] = delivered
        delivered.lowest_start = min(delivered.lowest_start, fetch.start)
        for lg, block_ordinals, mask in zip(
            entries.layer_group_indices, entries.block_ordinals, copied
        ):
            got = block_ordinals[mask]
            if not len(got):
                continue
            blocks = delivered.delivered_block_masks[lg]
            if int(got.max()) >= len(blocks):
                blocks = np.concatenate([blocks, np.zeros(int(got.max()) + 1 - len(blocks), bool)])
            blocks[got] = True
            delivered.delivered_block_masks[lg] = blocks
        delivered.resume_bounds_memo = None

    def _deliveries_of(self, request_id: int) -> _Deliveries | None:
        """What fetches delivered into the request's cache, or ``None``; the record may describe
        a cache the request no longer holds."""
        return self._deliveries.get(request_id)

    def _drop_deliveries(self, request_id: int) -> None:
        """Drop what fetches delivered into the request's cache."""
        self._deliveries.pop(request_id, None)

    def _invalidate_delivered_blocks_past_capacity(
        self, delivered: _Deliveries, kv: _KVCache
    ) -> None:
        """Forget delivered entries at or past the cache's block count: a shrink freed their pages,
        and a regrow brings pages without their contents."""
        kept = _manager.num_blocks(kv)
        for blocks in delivered.delivered_block_masks:
            if blocks[kept:].any():
                blocks[kept:] = False
                delivered.resume_bounds_memo = None

    def _drop_records(self, request_id: int) -> None:
        """Drop the request's four records, at its free."""
        for records in (
            self._latest_fetches,
            self._deliveries,
            self._kept_prefixes,
            self._mixed_tail_caches,
        ):
            records.pop(int(request_id), None)

    def _computed_tokens(self, request_id: int, kv: _KVCache) -> int:
        """What resume_bounds counts as computed besides delivered entries: the tokens up to the
        history, but only the committed ones and what the latest fetch kept while the history stays
        where that fetch's grow left it past them (a windowed fetch moves it to its end)."""
        state = _manager.cache_state(kv)
        advanced = self._kept_prefixes.get(request_id)
        # Past where that fetch left it, the history moved as the request resumed and computed on.
        if advanced is None or advanced[0] is not kv or state.history > advanced[2]:
            return max(state.committed, state.history)
        return max(state.committed, advanced[1])

    def _prefix_kept_by_fetch(self, request_id: int, kv: _KVCache, start: int) -> int:
        """What a fetch from ``start`` keeps of what was computed before it, since it may overwrite
        every block from ``start`` on. It never passes the history, which no shrink goes below;
        delivered entries past it keep their own record."""
        return min(self._computed_tokens(request_id, kv), start)

    def _resume_bounds(
        self,
        request: LlmRequest,
        request_id: int,
        manager: KVCacheManagerV2,
        kv: _KVCache,
        history: int,
    ) -> ResumeBounds | None:
        """``Staging.resume_bounds`` of a lender that serves and has not stopped, for the request's
        cache ``kv`` at a history of ``history`` tokens."""
        # A record of a replaced cache (a restart) is moot.
        fetch = self._latest_fetches.get(request_id)
        if fetch is not None and fetch.kv is not kv:
            del self._latest_fetches[request_id]
            fetch = None
        delivered = self._deliveries.get(request_id)
        if delivered is not None and delivered.kv is not kv:
            del self._deliveries[request_id]
            delivered = None
        advanced = self._kept_prefixes.get(request_id)
        if advanced is not None and advanced[0] is not kv:
            del self._kept_prefixes[request_id]
        tainted = self._mixed_tail_caches.get(request_id)
        if tainted is not None and tainted is not kv:
            del self._mixed_tail_caches[request_id]
            tainted = None
        if fetch is not None and not self._fetch_settled(fetch):
            return None
        if tainted is not None:
            # Empty: the request drops its cache and computes from 0.
            return ResumeBounds(max_position=0, min_position=max(history, 1))
        known = self._computed_tokens(request_id, kv)
        if delivered is None:
            # Nothing delivered: what was computed and kept counts, resumed no lower than the
            # history, which a windowed fetch moved to its end.
            usable, floor = self._clip_to_bidirectional_runs(request, known)
            return ResumeBounds(max_position=usable, min_position=max(floor, history))
        # A shrink the manager did not report still shows as blocks past the capacity.
        self._invalidate_delivered_blocks_past_capacity(delivered, kv)
        if delivered.resume_bounds_memo is None or delivered.resume_bounds_memo[0] != known:
            usable = self._max_resume_position(manager, delivered, known)
            delivered.resume_bounds_memo = (
                known,
                *self._clip_to_bidirectional_runs(request, usable),
            )
        _, usable, floor = delivered.resume_bounds_memo
        return ResumeBounds(
            max_position=usable,
            min_position=max(
                floor, int(self._min_resume_position(manager, history, delivered.lowest_start))
            ),
        )

    def _clip_to_bidirectional_runs(self, request: LlmRequest, usable: int) -> tuple[int, int]:
        """The end lowered to the start of a run of multimodal tokens it falls strictly inside, and
        the lowest floor that then leaves no position strictly inside a run below that end: the end
        of the last run at or below it. Runs are those the scheduler keeps within one chunk."""
        runs = _manager.bidirectional_runs(request)
        for b, e in runs:
            if b < usable < e:
                usable = b
        return int(usable), max((e for _, e in runs if e <= usable), default=0)

    def _min_resume_position(self, manager: KVCacheManagerV2, history: int, origin: int) -> int:
        """The lowest start that needs no restart: the history where a chunk resumed below it may
        end below it and raise, or a window has released blocks at it (they have no pages), else
        the smaller of the lowest start of a delivered fetch and the history."""
        if _ranges._floor_follows_history(manager):
            return history
        for lg in range(self._layout.num_layer_groups):
            stale_beg, stale_end = _ranges._stale_range(manager, self._layout, lg, history)
            if stale_end > stale_beg:
                return history
        return min(origin, history)

    def _max_resume_position(
        self, manager: KVCacheManagerV2, delivered: _Deliveries, known: int
    ) -> int:
        """The largest start ``P >= known`` where every layer group has what it reads among the
        blocks below ``known``, computed before the fetches, and delivered entries: full attention
        every block below ``P``, a window its sinks and in-window blocks."""
        # TODO: a fetch none of whose entries a window reads delivers none, so the usable end stays at
        # the blocks computed before the fetches, and the interval can come out empty although the
        # fetch covered all that a resume at its end reads.
        tpb = int(self._layout.tokens_per_block)
        first = known // tpb  # the blocks below were computed
        last = max((len(blocks) for blocks in delivered.delivered_block_masks), default=0)
        if last <= first:
            return known
        # Non-monotonic in P under windows, so each is checked. Blocks below known behind a window's
        # history have no pages, but no start at or above the floor reads them.
        top = last  # no start past a full-attention group's first missing block passes that group
        missing = []  # per layer group: missing[j], the undelivered blocks of [first, first + j)
        for lg, blocks in enumerate(delivered.delivered_block_masks):
            have = np.zeros(last - first, dtype=bool)
            mine = blocks[first:last]
            have[: len(mine)] = mine
            missing.append(np.concatenate([[0], np.cumsum(~have)]))
            if self._layout.windows[lg] is None and not have.all():
                top = min(top, first + int(np.argmin(have)))

        def gap(lg: int, a: int, b: int) -> bool:
            a, b = max(a, first), min(b, last)
            return b > a and missing[lg][b - first] - missing[lg][a - first] > 0

        # The largest start first: a fetch whose leases all landed is usable at its end at once.
        for end_block in range(top, first, -1):
            for lg in range(len(delivered.delivered_block_masks)):
                stale_beg, stale_end = _ranges._stale_range(
                    manager, self._layout, lg, end_block * tpb
                )
                if stale_end > stale_beg:
                    bad = gap(lg, first, min(stale_beg, end_block)) or gap(lg, stale_end, end_block)
                else:
                    bad = gap(lg, first, end_block)
                if bad:
                    break
            else:
                return end_block * tpb
        return known
