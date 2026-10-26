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
"""Per-request fetch deliveries and safe resume intervals."""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from heapq import merge
from typing import TYPE_CHECKING, Callable, Sequence

import numpy as np

from . import _manager, _ranges
from ._types import ResumeBounds

if TYPE_CHECKING:
    from tensorrt_llm.runtime.kv_cache_manager_v2 import _KVCache

    from ..kv_cache_manager_v2 import KVCacheManagerV2
    from ._layout import ManagerLayout
    from ._lender import _LeaseBase, _LentBlockEntries, _QueuedCopy


# Read-only masks of whole deliveries' blocks by run, shared by records, which write a mask only
# through ``_writable_mask``; emptied when the masks hold ``_RUN_MASK_BYTES``.
_run_masks: dict[tuple[int, int], np.ndarray] = {}
_RUN_MASK_BYTES = 1 << 22
_run_mask_bytes = 0
# The run of a layer group that lent no block.
_NO_RUN = (0, 0)
# Builds a named tuple as the tuple it is, without its Python-level constructor.
_new_tuple = tuple.__new__


def _run_mask(first: int, last: int) -> np.ndarray:
    """The shared read-only mask of ``last`` blocks, true from ``first`` on."""
    global _run_mask_bytes
    found = _run_masks.get((first, last))
    if found is None:
        if _run_mask_bytes + last > _RUN_MASK_BYTES:
            _run_masks.clear()
            _run_mask_bytes = 0
        found = np.zeros(last, dtype=bool)
        found[first:] = True
        found.flags.writeable = False
        _run_masks[first, last] = found
        _run_mask_bytes += last
    return found


def _writable_mask(masks: list[np.ndarray], lg: int) -> np.ndarray:
    """``masks[lg]`` to write, first replaced by a copy of its own where it is read-only."""
    blocks = masks[lg]
    if not blocks.flags.writeable:
        blocks = masks[lg] = blocks.copy()
    return blocks


@dataclass(eq=False, slots=True)
class _FetchRecord:
    """One fetch into a cache, with mark/copy completion and its declared token extent."""

    start: int
    kv: object
    marks_queued: bool = False
    copy: _QueuedCopy | None = None
    # Where the rows of recurrent state the lease lends stand, its end; None where it lends none.
    state_end: int | None = None
    # Where the fetch ends; None where unknown.
    end: int | None = None
    entries: _LentBlockEntries | None = None
    # The request's prompt length and multimodal runs at the lend, which its return's bounds read.
    prompt: int = 0
    runs: list[tuple[int, int]] | None = None


class _FetchState(enum.Enum):
    """Derived lifecycle of the request's latest fetch."""

    # In _latest_fetches with marks not queued: poll_resume_bounds is None and a new fetch fails.
    # _apply_marks sets marks_queued, or abandons it (failed copy, raise) as _abandon_fetch does.
    AWAITING_MARKS = "awaiting_marks"
    # In _latest_fetches, its marked entries' copy queued without error and pending: as
    # AWAITING_MARKS for poll_resume_bounds and a new fetch. The copy's completion makes it COPY_DONE.
    COPY_PENDING = "copy_pending"
    # In _latest_fetches, its copy complete or none queued: poll_resume_bounds answers, and a new fetch into
    # the cache may start. The request's next fetch replaces it.
    COPY_DONE = "copy_done"
    # Out of _latest_fetches before its marks were queued: none of its entries count.
    ABANDONED = "abandoned"
    # Out of _latest_fetches after its marks were queued: replaced by the next fetch, or dropped at
    # the free or by poll_resume_bounds for a replaced cache. Its entries stay delivered.
    RETIRED = "retired"


@dataclass(eq=False, slots=True)
class _Deliveries:
    """Delivered rows and valid token prefixes for one cache; state holds at one endpoint."""

    kv: object
    lowest_start: int
    # By local layer group (not view position), then block ordinal; set once a copy queues without
    # error, the bounds also awaiting its completion. Recurrent state records nothing here.
    delivered_block_masks: list[np.ndarray]
    # (tokens computed before per layer group, max_position, lowest safe position), last computed
    resume_bounds_memo: tuple[tuple[int, ...], int, int] | None = None
    # Where the recurrent state the latest fetch delivered stands, or None: none delivered, or a
    # later fetch lent its rows and did not deliver them all.
    state_end: int | None = None
    # Settled fetch extent, including fetches whose margin rows were all absent.
    end: int = 0
    # Partial physical pages: (layer group, logical block) -> valid token end.
    partial_ends: dict[tuple[int, int], int] = field(default_factory=dict)


# TODO: each per-request record holds its own cache reference, so every reader compares it again and
# a record of a replaced cache keeps that cache until the record is dropped or replaced.
class _FetchLedger:
    """Per-request fetch state and resume bounds, changed through these methods."""

    def __init__(
        self, layout: ManagerLayout, copy_done: Callable[[_QueuedCopy | None], bool]
    ) -> None:
        self._layout = layout
        recurrent = getattr(layout, "recurrent", ())
        # The layer groups of recurrent state, which hold one row each, valid at one position.
        self._recurrent_groups = frozenset(lg for lg, flag in enumerate(recurrent) if flag)
        groups = getattr(layout, "num_layer_groups", len(recurrent))
        self._paged_groups = tuple(lg for lg in range(groups) if lg not in self._recurrent_groups)
        # What the closed form of a whole delivery's bounds reads; it applies without Helix and
        # where every window's stale range is the sinkless window formula.
        self._tokens_per_block = int(getattr(layout, "tokens_per_block", 1))
        # The highest history the window formula reads, as the manager reads it: ``_MAX_TOKENS``
        # less one block.
        self._longest = _manager._MAX_TOKENS - self._tokens_per_block
        windows = tuple(getattr(layout, "windows", ()))
        sinks = tuple(getattr(layout, "sink_blocks", ()))
        self._window_of = tuple(
            None
            if window is None or lg in self._recurrent_groups
            else (int(window), _ranges._read_slack(layout, lg))
            for lg, window in enumerate(windows)
        )
        self._stale_windows = tuple(int(window) for window in windows if window is not None)
        self._closed_form = (
            getattr(layout, "cp", (0, 1))[1] == 1
            and len(windows) == groups
            and (
                not self._stale_windows
                or (
                    bool(getattr(layout, "stale_ranges_by_formula", False))
                    and all(
                        lg < len(sinks) and not sinks[lg]
                        for lg, window in enumerate(windows)
                        if window is not None
                    )
                )
            )
        )
        # Whether a copy has completed, as the lender answers: each event asked at most once per
        # round of its progress.
        self._copy_done = copy_done
        # Per-request fetch records, each with the cache it describes: the request's free drops all
        # four, and poll_resume_bounds drops one of a cache the request no longer holds.
        self._latest_fetches: dict[int, _FetchRecord] = {}  # the latest fetch into each request
        # What fetches delivered into each request's cache, as ``_deliveries`` reads it once the
        # whole deliveries in ``_pending`` are written into it.
        self._delivered: dict[int, _Deliveries] = {}
        # Whole deliveries ``_defer_whole_delivery`` settled and no reader has needed yet, by
        # request: ``[fetch, spans, resume bounds memo]``; the request's free drops them.
        self._pending: dict[int, list] = {}
        # Requests a fetch left with history past a group's committed tokens: the cache, per layer
        # group the computed prefix the latest such fetch kept, and the history it left.
        self._kept_prefixes: dict[int, tuple[object, tuple[int, ...], int]] = {}
        # Requests whose committed tail block a failed copy may have left with the fetch's bytes in
        # some pools and the bytes it held before in others: the cache. ResumeBounds is empty for it.
        self._mixed_tail_caches: dict[int, object] = {}

    @property
    def _deliveries(self) -> dict[int, _Deliveries]:
        """What fetches delivered into each request's cache, by request: ``_delivered``, with every
        whole delivery ``_defer_whole_delivery`` holds written into it first."""
        if self._pending:
            self._write_pending()
        return self._delivered

    def _write_pending(self) -> None:
        """Write the records of the whole deliveries ``_defer_whole_delivery`` holds; one that
        raises leaves the rest held."""
        pending = self._pending
        while pending:
            request_id = next(iter(pending))
            fetch, spans, memo = pending.pop(request_id)
            self._record_whole_delivery(request_id, fetch, spans, memo)

    def _record_fetch(
        self,
        request_id: int,
        kv: _KVCache,
        start: int,
        kept: Sequence[int],
        position: int,
        committed: Sequence[int],
        state_end: int | None = None,
        end: int | None = None,
        prompt: int = 0,
        runs: list[tuple[int, int]] | None = None,
    ) -> _FetchRecord:
        """Record and return fetch ``[start, end)`` into ``kv``; growth leaving the history at
        ``position`` past committed tokens keeps ``kept`` until computation passes ``position``."""
        if position > min(committed, default=0):
            self._kept_prefixes[request_id] = (kv, tuple(kept), position)
        fetch = _FetchRecord(start, kv, state_end=state_end, end=end, prompt=prompt, runs=runs)
        self._latest_fetches[request_id] = fetch
        return fetch

    def _record_next_fetch(
        self,
        request_id: int,
        kv: _KVCache,
        start: int,
        position: int,
        committed: Sequence[int],
        state_end: int | None = None,
        end: int | None = None,
    ) -> _FetchRecord | None:
        """``_record_fetch`` of a fetch that grows nothing, reading ``_prefix_kept_by_fetch`` only
        where kept; ``None``, recording nothing, while an earlier fetch into ``kv`` is unsettled."""
        previous = self._latest_fetches.get(request_id)
        if previous is not None and previous.kv is kv and not self._fetch_settled(previous):
            return None
        if position > min(committed, default=0):
            kept = self._prefix_kept_by_fetch(request_id, kv, start)
            self._kept_prefixes[request_id] = (kv, kept, position)
        # Neither marks queued nor a copy yet.
        fetch = _FetchRecord(start, kv, False, None, state_end, end)
        self._latest_fetches[request_id] = fetch
        return fetch

    def _has_records(self, request_id: int) -> bool:
        """Whether a record names the request: a fetch, deliveries, a kept prefix or a mixed tail."""
        return (
            request_id in self._latest_fetches
            or request_id in self._delivered
            or request_id in self._pending
            or request_id in self._kept_prefixes
            or request_id in self._mixed_tail_caches
        )

    def _abandon_fetch(self, lease: _LeaseBase) -> None:
        """Drop the write lease's fetch record: it delivered nothing, and poll_resume_bounds counts what
        earlier fetches into the cache delivered."""
        if lease._fetch is not None and self._latest_fetches.get(lease._request_id) is lease._fetch:
            del self._latest_fetches[lease._request_id]

    def _unsettled_fetch_refusal(
        self, request_id: int, kv: _KVCache, entries: _LentBlockEntries | None = None
    ) -> str | None:
        """Refuse another fetch, or a read overlapping an unsettled fetch's rows."""
        previous = self._latest_fetches.get(request_id)
        if previous is not None and previous.kv is kv and not self._fetch_settled(previous):
            if entries is not None and previous.entries is not None:
                pending = dict(
                    zip(previous.entries.layer_group_indices, previous.entries.block_ordinals)
                )
                if not any(
                    np.intersect1d(rows, pending.get(lg, ())).size
                    for lg, rows in zip(entries.layer_group_indices, entries.block_ordinals)
                ):
                    return None
            return f"request {request_id} already has an unsettled fetch"
        return None

    def _unheld_block(
        self,
        request_id: int,
        kv: _KVCache,
        entries: _LentBlockEntries,
        top: int,
        history: int | None = None,
    ) -> tuple[int, int] | None:
        """First unreadable ``(layer group, block)`` below ``top``, or None: attention needs
        computed or delivered bytes; a cache with deliveries needs a complete STATE snapshot at
        ``top``, one without keeps its computed state. ``history``: ``kv``'s, if read."""
        delivered = self._deliveries.get(request_id)
        if delivered is not None and delivered.kv is not kv:
            delivered = None
        tpb = int(self._layout.tokens_per_block)
        tainted = self._mixed_tail_caches.get(request_id) is kv
        if delivered is not None or tainted:
            committed = (
                _manager.committed_by_group(kv, self._layout.num_layer_groups) if tainted else ()
            )
            for lg, ordinals in zip(entries.layer_group_indices, entries.block_ordinals):
                if not len(ordinals):
                    continue
                if lg in self._recurrent_groups:
                    if delivered is not None and delivered.state_end != top:
                        return lg, 0
                elif tainted:
                    tail, partial = divmod(committed[lg], tpb)
                    if partial and tail in ordinals:
                        return lg, tail
        if history is None:
            history = _manager.cache_state(kv).history
        advanced = self._kept_prefixes.get(request_id)
        if history >= top and (advanced is None or advanced[0] is not kv or history > advanced[2]):
            return None
        computed = self._computed_tokens(request_id, kv)
        if self._known(computed) >= top:
            return None
        rank, size = getattr(self._layout, "cp", (0, 1))
        page = tpb // size
        for lg, ordinals in zip(entries.layer_group_indices, entries.block_ordinals):
            if lg in self._recurrent_groups:
                continue
            for b in ordinals.tolist():
                first = b * tpb + rank * page  # the first token of this rank's page
                if first >= top:
                    continue
                valid = int(computed[lg])
                if delivered is not None:
                    blocks = delivered.delivered_block_masks[lg]
                    if b < len(blocks) and blocks[b]:
                        valid = max(valid, delivered.partial_ends.get((lg, b), first + page))
                if valid < min(first + page, top):
                    return lg, b
        return None

    def _fetch_settled(self, fetch: _FetchRecord) -> bool:
        """The fetch's arrived entries are marked and their copy into the pages is done."""
        return fetch.marks_queued and self._copy_done(fetch.copy)

    def _fetch_state(self, request_id: int, fetch: _FetchRecord) -> _FetchState:
        """The state of ``fetch``, a fetch into request ``request_id``, from ``_latest_fetches`` and
        ``_fetch_settled``; see ``_FetchState``."""
        if self._latest_fetches.get(request_id) is not fetch:
            return _FetchState.RETIRED if fetch.marks_queued else _FetchState.ABANDONED
        if self._fetch_settled(fetch):
            return _FetchState.COPY_DONE
        return _FetchState.COPY_PENDING if fetch.marks_queued else _FetchState.AWAITING_MARKS

    def _take_out_fetch(self, lease: _LeaseBase, fetch: _FetchRecord | None) -> bool:
        """Take ``fetch``, the write lease's, out of the latest fetches before its marks' copy is
        queued, if it is still the request's latest; returns whether it was."""
        current = fetch is not None and self._latest_fetches.get(lease._request_id) is fetch
        if current:
            del self._latest_fetches[lease._request_id]
        return current

    def _invalidate_delivered_blocks(
        self, lease: _LeaseBase, copied: list[np.ndarray] | None
    ) -> None:
        """Stop counting the delivered entries ``copied`` selects (``None``: all the lease's) before
        their copy is queued, as one failing partway can leave a row with each fetch's bytes in
        different pools; ``_record_delivered_blocks`` recounts them once it queues without error."""
        delivered = self._deliveries.get(lease._request_id)
        if delivered is None or delivered.kv is not lease._kv:
            return
        lent = lease._block_entries
        for i, (lg, block_ordinals) in enumerate(
            zip(lent.layer_group_indices, lent.block_ordinals)
        ):
            if lg in self._recurrent_groups:
                if len(block_ordinals) and (copied is None or copied[i].any()):
                    delivered.state_end = None
                    delivered.resume_bounds_memo = None
                continue
            blocks = delivered.delivered_block_masks[lg]
            entries = block_ordinals if copied is None else block_ordinals[copied[i]]
            entries = entries[entries < len(blocks)]
            for key in tuple(delivered.partial_ends):
                if key[0] == lg and np.any(entries == key[1]):
                    delivered.partial_ends.pop(key)
            if blocks[entries].any():
                _writable_mask(delivered.delivered_block_masks, lg)[entries] = False
                delivered.resume_bounds_memo = None

    def _record_mixed_tail(self, request_id: int, kv: _KVCache) -> None:
        """Record ``kv`` as the request's cache whose committed tail block a failed copy may have
        left with the fetch's bytes in some pools and the bytes it held before in others."""
        self._mixed_tail_caches[request_id] = kv

    def _put_back_fetch(
        self, lease: _LeaseBase, fetch: _FetchRecord, copied: list[np.ndarray]
    ) -> None:
        """Once its marks' copy is queued without error, put ``fetch`` back as the request's latest,
        ``marks_queued`` set and with the lease's copy, and add the entries ``copied`` selects to
        the cache's deliveries."""
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
        """Add a fetch's copied entries to the cache's deliveries, so consecutive leases count as
        one fetch; recurrent state stands at its end if every state row arrived, else nowhere."""
        delivered = self._deliveries.get(request_id)
        if delivered is None or delivered.kv is not fetch.kv:
            empty = [np.zeros(0, dtype=bool) for _ in range(self._layout.num_layer_groups)]
            delivered = _Deliveries(fetch.kv, fetch.start, empty)
            self._deliveries[request_id] = delivered
        delivered.lowest_start = min(delivered.lowest_start, fetch.start)
        tpb = int(self._layout.tokens_per_block)
        rank, size = getattr(self._layout, "cp", (0, 1))
        if fetch.end is not None:
            delivered.end = max(delivered.end, fetch.end)
        if fetch.state_end is not None:
            states = [
                mask
                for lg, mask in zip(entries.layer_group_indices, copied)
                if lg in self._recurrent_groups
            ]
            whole = bool(states) and all(len(mask) and mask.all() for mask in states)
            delivered.state_end = fetch.state_end if whole else None
        for lg, block_ordinals, mask in zip(
            entries.layer_group_indices, entries.block_ordinals, copied
        ):
            if lg in self._recurrent_groups:
                continue
            got = block_ordinals[mask]
            if not len(got):
                continue
            blocks = delivered.delivered_block_masks[lg]
            if int(got.max()) >= len(blocks):
                blocks = np.concatenate([blocks, np.zeros(int(got.max()) + 1 - len(blocks), bool)])
            elif not blocks.flags.writeable:
                blocks = blocks.copy()
            blocks[got] = True
            delivered.delivered_block_masks[lg] = blocks
            if fetch.end is not None:
                tail = fetch.end // tpb
                page_end = tail * tpb + (rank + 1) * (tpb // size)
                if fetch.end < page_end and np.any(got == tail):
                    delivered.partial_ends[lg, tail] = fetch.end
        delivered.resume_bounds_memo = None

    def _defer_whole_delivery(
        self, request_id: int, fetch: _FetchRecord, spans: Sequence[tuple[int, int] | None]
    ) -> None:
        """``_record_whole_delivery`` of ``fetch``, settled now and held in ``_pending`` until a
        read of ``_deliveries`` writes it; ``_whole_delivery_bounds`` gives its bounds meanwhile."""
        fetch.copy = None
        fetch.marks_queued = True
        self._pending[request_id] = [fetch, spans, None]

    def _defer_first_delivery(
        self,
        request_id: int,
        fetch: _FetchRecord,
        kv: _KVCache,
        spans: Sequence[tuple[int, int] | None],
    ) -> bool:
        """If ``fetch`` is the request's latest and nothing was delivered into ``kv``, its cache,
        yet, settle it as a whole delivery of ``spans`` (held in the closed form, else recorded):
        True; else False, changing nothing."""
        if (
            self._latest_fetches.get(request_id) is not fetch
            or self._delivered_kv(request_id) is kv
        ):
            return False
        if self._closed_form:
            self._defer_whole_delivery(request_id, fetch, spans)
        else:
            self._record_whole_delivery(request_id, fetch, spans)
        return True

    def _delivered_kv(self, request_id: int) -> object | None:
        """The cache the request's deliveries describe, a held whole delivery's included, or
        ``None`` without deliveries; writes nothing."""
        pending = self._pending.get(request_id)
        if pending is not None:
            return pending[0].kv
        delivered = self._delivered.get(request_id)
        return None if delivered is None else delivered.kv

    def _record_whole_delivery(
        self,
        request_id: int,
        fetch: _FetchRecord,
        spans: Sequence[tuple[int, int] | None],
        memo: tuple[tuple[int, ...], int, int] | None = None,
    ) -> None:
        """Settle ``fetch``, the request's latest, all lent rows arrived needing no copy, into a
        cache without deliveries: ``_put_back_fetch``'s records written directly, ``memo`` their
        bounds memo, ``spans`` as ``_LentBlockEntries.spans``. No Helix."""
        fetch.copy = None
        fetch.marks_queued = True
        # This record replaces any held for the request.
        self._pending.pop(request_id, None)
        tpb = self._tokens_per_block
        end = fetch.end
        tail = None if end is None or not end % tpb else end // tpb
        masks = []
        partial_ends = {}
        for lg, span in enumerate(spans):
            run = _NO_RUN if span is None or span[1] <= span[0] else span
            mask = _run_masks.get(run)
            masks.append(_run_mask(*run) if mask is None else mask)
            if tail is not None and run[0] <= tail < run[1]:
                partial_ends[lg, tail] = end
        # ``spans`` holds every layer group, so it lent a row of each group of recurrent state.
        self._delivered[request_id] = _Deliveries(
            fetch.kv,
            fetch.start,
            masks,
            memo,
            fetch.state_end if self._recurrent_groups else None,
            0 if end is None else max(0, end),
            partial_ends,
        )

    def _whole_delivery_bounds(
        self,
        prompt: int,
        runs: Sequence[tuple[int, int]],
        request_id: int,
        manager: KVCacheManagerV2,
        kv: _KVCache,
        state: _manager.CacheState,
        spans: Sequence[tuple[int, int] | None],
        blocks: int | None = None,
    ) -> ResumeBounds | None:
        """``_resume_bounds`` from the runs while the only deliveries are the latest fetch's whole
        delivery of ``spans``, its runs ending at the fetch's end block; ``None`` where other rules
        may apply: deliveries past capacity, an endpoint block runs and computed prefix miss, a sink
        before a read window, a multimodal run at the endpoint. No Helix; keeps a held one held."""
        if not self._closed_form:
            return self._whole_delivery_bounds_by_ranges(
                prompt, runs, request_id, manager, kv, state, spans, blocks
            )
        held = self._pending.get(request_id)
        if held is None:
            delivered = self._deliveries[request_id]
            delivered_end = delivered.end
            lowest_start = delivered.lowest_start
            state_end = delivered.state_end
        else:
            # The values ``_record_whole_delivery`` writes of the fetch.
            settled = held[0]
            delivered_end = 0 if settled.end is None else max(0, settled.end)
            lowest_start = settled.start
            state_end = settled.state_end if self._recurrent_groups else None
        kept_prefixes = self._kept_prefixes
        if kept_prefixes:
            advanced = kept_prefixes.get(request_id)
            if advanced is not None and advanced[0] is not kv:
                del kept_prefixes[request_id]
        history = state.history
        mixed = self._mixed_tail_caches
        if mixed:
            tainted = mixed.get(request_id)
            if tainted is not None:
                if tainted is kv:
                    return _new_tuple(ResumeBounds, (0, max(history, 1)))
                del mixed[request_id]
        tpb = self._tokens_per_block
        if delivered_end > (_manager.num_blocks(kv) if blocks is None else blocks) * tpb:
            # A shrink the manager did not report: the rules drop what lies past the capacity.
            return None
        computed = self._computed_tokens(request_id, kv, state)
        known = self._known(computed)
        cap = max(known, min(delivered_end, prompt))
        first, last = known // tpb, -(-cap // tpb)
        recurrent = self._recurrent_groups
        windows = self._window_of
        longest = self._longest
        # Per windowed group: its window, slack and the end of the blocks behind its read window at
        # ``cap`` (the formula's stale range there starts at block 0).
        read_ends = []
        for lg, span in enumerate(spans):
            if lg in recurrent:
                continue
            covered_from = computed[lg] // tpb
            if covered_from < first:
                covered_from = first
            terms = windows[lg]
            if terms is not None:
                window, slack = terms
                behind = min(cap + slack, longest) + 1 - window
                read_end = behind // tpb if behind > 0 else 0
                read_ends.append((lg, window, slack, read_end))
                if read_end > covered_from:
                    covered_from = read_end
            if covered_from < last and (span is None or span[0] > covered_from or span[1] < last):
                return None
        usable = cap
        floor = 0
        if runs:
            # ``_clip_to_bidirectional_runs``: an endpoint strictly inside a run is the rules' case.
            for b, e in runs:
                if b < usable < e:
                    return None
            floor = max((e for _, e in runs if e <= usable), default=0)
        # ``_min_resume_position``; recurrent state needs none, as it holds at one position.
        lowest = None
        if not recurrent:
            lowest = history
            if not _ranges._floor_follows_history(manager):
                lowest = min(lowest_start, history)
                for window in self._stale_windows:
                    if min(history, longest) + 1 - window >= tpb:
                        lowest = history
                        break
        if usable > known and read_ends:
            if lowest is None or lowest >= usable:
                # A floor a block missing behind a read window sets is at most ``usable``, which
                # that window passed, so it cannot raise the bounds; the rules compute it if asked.
                if held is None:
                    delivered.resume_bounds_memo = None
                else:
                    held[2] = None
                if lowest is not None:
                    return _new_tuple(ResumeBounds, (usable, lowest))
                end = state_end
                if end is None or usable < end:
                    return _new_tuple(ResumeBounds, (0, max(history, 1)))
                return _new_tuple(ResumeBounds, (end, end))
            end_block = -(-usable // tpb)
            # No read window at ``usable`` passed the one partial page, the fetch's end block: each
            # group's one candidate is its last block missing behind its window, below its runs.
            for lg, window, slack, read_end in read_ends:
                valid = computed[lg]
                beg, end = valid // tpb, min(end_block, read_end)
                if end <= beg:
                    continue
                span = spans[lg]
                lent_from, lent_to = span if span is not None and span[1] > span[0] else (0, 0)
                if end > lent_to:
                    missing = end - 1
                elif beg < min(end, lent_from):
                    missing = min(end, lent_from) - 1
                else:
                    continue
                # The last history whose read window still reads the block, by the formula.
                keeping = (missing + 1) * tpb + window - 2
                last_read = (_manager._MAX_TOKENS if keeping >= longest else keeping) - slack
                if valid < last_read and last_read + 1 > floor:
                    floor = last_read + 1
        if held is None:
            delivered.resume_bounds_memo = computed, usable, floor
        else:
            held[2] = computed, usable, floor
        if recurrent:
            end = state_end
            if end is None or usable < end:
                return _new_tuple(ResumeBounds, (0, max(history, 1)))
            return _new_tuple(ResumeBounds, (end, end))
        return _new_tuple(ResumeBounds, (usable, max(floor, lowest)))

    def _whole_delivery_bounds_by_ranges(
        self,
        prompt: int,
        runs: Sequence[tuple[int, int]],
        request_id: int,
        manager: KVCacheManagerV2,
        kv: _KVCache,
        state: _manager.CacheState,
        spans: Sequence[tuple[int, int] | None],
        blocks: int | None = None,
    ) -> ResumeBounds | None:
        """``_whole_delivery_bounds`` for a layout whose windows the formula does not give, with
        each stale range and last history reading a block asked of ``_ranges``."""
        delivered = self._deliveries[request_id]
        advanced = self._kept_prefixes.get(request_id)
        if advanced is not None and advanced[0] is not kv:
            del self._kept_prefixes[request_id]
        tainted = self._mixed_tail_caches.get(request_id)
        if tainted is not None and tainted is not kv:
            del self._mixed_tail_caches[request_id]
            tainted = None
        history = state.history
        if tainted is not None:
            return ResumeBounds(max_position=0, min_position=max(history, 1))
        layout = self._layout
        tpb = int(layout.tokens_per_block)
        kept = _manager.num_blocks(kv) if blocks is None else blocks
        # The record's masks and partial pages end at the block that holds its end.
        if delivered.end > kept * tpb:
            return None
        computed = self._computed_tokens(request_id, kv, state)
        known = self._known(computed)
        cap = max(known, min(delivered.end, prompt))
        first, last = known // tpb, -(-cap // tpb)
        windowed = []
        for lg, span in enumerate(spans):
            if lg in self._recurrent_groups:
                continue
            covered_from = max(first, int(computed[lg]) // tpb)
            if layout.windows[lg] is not None:
                windowed.append(lg)
                read_beg, read_end = _ranges._read_stale_range(manager, layout, lg, cap)
                if min(read_beg, last) > covered_from:
                    return None
                covered_from = max(covered_from, read_end)
            if covered_from < last and (span is None or span[0] > covered_from or span[1] < last):
                return None
        usable = cap
        clipped, floor = self._clip_to_bidirectional_runs(runs, usable)
        if clipped != usable:
            return None
        if usable > known:
            end_block = -(-usable // tpb)
            for lg in windowed:
                computed_blocks = int(computed[lg]) // tpb
                read_beg, read_end = _ranges._read_stale_range(manager, layout, lg, usable)
                beg, end = max(computed_blocks, read_beg), min(end_block, read_end)
                if end <= beg:
                    continue
                span = spans[lg]
                lent_from, lent_to = span if span is not None and span[1] > span[0] else (0, 0)
                if end > lent_to:
                    missing = end - 1
                elif beg < min(end, lent_from):
                    missing = min(end, lent_from) - 1
                else:
                    missing = -1
                candidates = {missing} if missing >= 0 else set()
                candidates.update(
                    ordinal
                    for group, ordinal in delivered.partial_ends
                    if group == lg and beg <= ordinal < end
                )
                for ordinal in sorted(candidates, reverse=True):
                    last_read = _ranges._last_history_reading(manager, layout, lg, ordinal)
                    valid = max(int(computed[lg]), delivered.partial_ends.get((lg, ordinal), 0))
                    if valid < last_read:
                        floor = max(floor, last_read + 1)
                        break
        delivered.resume_bounds_memo = computed, usable, floor
        if self._recurrent_groups:
            end = delivered.state_end
            if end is None or usable < end:
                return ResumeBounds(max_position=0, min_position=max(history, 1))
            return ResumeBounds(max_position=end, min_position=end)
        return ResumeBounds(
            max_position=usable,
            min_position=max(
                floor, int(self._min_resume_position(manager, history, delivered.lowest_start))
            ),
        )

    def _deliveries_of(self, request_id: int) -> _Deliveries | None:
        """What fetches delivered into the request's cache, or ``None``; the record may describe
        a cache the request no longer holds."""
        return self._deliveries.get(request_id)

    def _drop_deliveries(self, request_id: int) -> None:
        """Drop what fetches delivered into the request's cache, a whole delivery held included."""
        self._pending.pop(request_id, None)
        self._delivered.pop(request_id, None)

    def _invalidate_delivered_blocks_past_capacity(
        self, delivered: _Deliveries, kv: _KVCache
    ) -> None:
        """Forget delivered entries at or past the cache's block count: a shrink freed their pages,
        and a regrow brings pages without their contents."""
        kept = _manager.num_blocks(kv)
        capacity_end = kept * int(self._layout.tokens_per_block)
        if (
            delivered.end <= capacity_end
            and all(len(blocks) <= kept for blocks in delivered.delivered_block_masks)
            and all(ordinal < kept for _, ordinal in delivered.partial_ends)
        ):
            return
        end = min(delivered.end, capacity_end)
        if end != delivered.end:
            delivered.end = end
            delivered.resume_bounds_memo = None
        delivered.partial_ends = {
            key: end for key, end in delivered.partial_ends.items() if key[1] < kept
        }
        for lg, blocks in enumerate(delivered.delivered_block_masks):
            if blocks[kept:].any():
                _writable_mask(delivered.delivered_block_masks, lg)[kept:] = False
                delivered.resume_bounds_memo = None

    def _drop_records(self, request_id: int) -> None:
        """Drop the request's four records, at its free, a whole delivery held included."""
        for records in (
            self._latest_fetches,
            self._pending,
            self._delivered,
            self._kept_prefixes,
            self._mixed_tail_caches,
        ):
            records.pop(int(request_id), None)

    def _computed_tokens(
        self, request_id: int, kv: _KVCache, state: _manager.CacheState | None = None
    ) -> tuple[int, ...]:
        """Computed tokens per layer group, fetched rows excluded: with the committed ones, the kept
        prefix while the history stays where the latest fetch grew it, else the history. ``state``:
        ``kv``'s, if read."""
        if state is None:
            state = _manager.cache_state(kv)
        history = state.history
        advanced = self._kept_prefixes.get(request_id)
        # Past where that fetch left it, the history moved as the request resumed and computed on.
        moved = advanced is None or advanced[0] is not kv or history > advanced[2]
        if not state.committed:
            return (history,) * self._layout.num_layer_groups if moved else tuple(advanced[1])
        committed = _manager.committed_by_group(kv, self._layout.num_layer_groups)
        if moved:
            return tuple(max(tokens, history) for tokens in committed)
        return tuple(max(tokens, kept) for tokens, kept in zip(committed, advanced[1]))

    def _prefix_kept_by_fetch(self, request_id: int, kv: _KVCache, start: int) -> tuple[int, ...]:
        """Keep each group's computed prefix up to start; a fetch may overwrite later uncommitted rows.
        Delivered rows past that prefix keep separate records."""
        if start == 0:
            return (0,) * self._layout.num_layer_groups
        return tuple(min(tokens, start) for tokens in self._computed_tokens(request_id, kv))

    def _known(self, computed: Sequence[int]) -> int:
        """The tokens every layer group but those of recurrent state computed, of ``computed``, one
        count per layer group."""
        paged = self._paged_groups
        if not paged:
            return max(computed, default=0)
        return min([computed[lg] for lg in paged])

    def _resume_bounds(
        self,
        prompt: int,
        runs: Sequence[tuple[int, int]],
        request_id: int,
        manager: KVCacheManagerV2,
        kv: _KVCache,
        state: _manager.CacheState,
    ) -> ResumeBounds | None:
        """``poll_resume_bounds`` of a serving, unstopped lender for the request's cache ``kv`` in
        ``state``, given its ``prompt`` length and multimodal ``runs``."""
        history = state.history
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
        computed = self._computed_tokens(request_id, kv, state)
        known = self._known(computed)
        if delivered is None:
            # Nothing delivered: what was computed and kept counts, resumed no lower than the
            # history, which a windowed fetch moved to its end.
            usable, floor = self._clip_to_bidirectional_runs(runs, known)
            return ResumeBounds(max_position=usable, min_position=max(floor, history))
        # A shrink the manager did not report still shows as blocks past the capacity.
        self._invalidate_delivered_blocks_past_capacity(delivered, kv)
        if delivered.resume_bounds_memo is None or delivered.resume_bounds_memo[0] != computed:
            cap = max(known, min(delivered.end, prompt))
            while True:
                usable = self._max_resume_position(manager, delivered, computed, cap)
                clipped, floor = self._clip_to_bidirectional_runs(runs, usable)
                if clipped == usable or clipped <= known:
                    usable = clipped
                    break
                # Moving back into a window can require blocks the larger endpoint did not read.
                cap = clipped
            floor = max(floor, self._delivered_resume_floor(manager, delivered, computed, usable))
            delivered.resume_bounds_memo = computed, usable, floor
        _, usable, floor = delivered.resume_bounds_memo
        if self._recurrent_groups:
            # Recurrent state holds at one position alone: where the delivered state stands, every
            # block below it there; anything else leaves the request to compute from 0.
            end = delivered.state_end
            if end is None or usable < end:
                return ResumeBounds(max_position=0, min_position=max(history, 1))
            return ResumeBounds(max_position=end, min_position=end)
        return ResumeBounds(
            max_position=usable,
            min_position=max(
                floor, int(self._min_resume_position(manager, history, delivered.lowest_start))
            ),
        )

    def _clip_to_bidirectional_runs(
        self, runs: Sequence[tuple[int, int]], usable: int
    ) -> tuple[int, int]:
        """The end lowered to the start of a multimodal run it lies strictly inside, and the lowest
        floor leaving no position below it inside a run: the end of the last run at or below it.
        ``runs``: those the scheduler keeps within one chunk."""
        if not runs:
            return int(usable), 0
        for b, e in runs:
            if b < usable < e:
                usable = b
        return int(usable), max((e for _, e in runs if e <= usable), default=0)

    def _min_resume_position(self, manager: KVCacheManagerV2, history: int, origin: int) -> int:
        """The lowest start that needs no restart: the history where a chunk resumed below it may
        end below it and raise, or a window has released blocks at it (they have no pages); else the
        lower of the history and ``origin``, a delivered fetch's lowest start."""
        if _ranges._floor_follows_history(manager):
            return history
        for lg in range(self._layout.num_layer_groups):
            stale_beg, stale_end = _ranges._stale_range(manager, self._layout, lg, history)
            if stale_end > stale_beg:
                return history
        return min(origin, history)

    def _max_resume_position(
        self,
        manager: KVCacheManagerV2,
        delivered: _Deliveries,
        computed: Sequence[int],
        cap: int | None = None,
    ) -> int:
        """Largest endpoint whose read rows are computed or delivered, partial pages to their valid
        end; a gap a window passed does not count. Helix checks this rank's page alone."""
        tpb = int(self._layout.tokens_per_block)
        known = self._known(computed)
        masks = delivered.delivered_block_masks
        groups = [lg for lg in range(len(masks)) if lg not in self._recurrent_groups]
        if not groups:
            return known if cap is None else cap
        cap = max((len(masks[lg]) for lg in groups), default=0) * tpb if cap is None else cap
        rank, size = getattr(self._layout, "cp", (0, 1))
        if size > 1:
            page = tpb // size
            usable = cap
            for lg in groups:
                b = int(computed[lg]) // tpb
                while b * tpb + rank * page < usable:
                    page_end = b * tpb + (rank + 1) * page
                    valid = int(computed[lg])
                    if b < len(masks[lg]) and masks[lg][b]:
                        valid = max(valid, delivered.partial_ends.get((lg, b), page_end))
                    if valid < min(usable, page_end):
                        usable = max(known, b * tpb + rank * page, valid)
                        break
                    b += 1
            return usable
        first, last = known // tpb, -(-cap // tpb)
        if last <= first:
            return known
        if all(self._layout.windows[lg] is None for lg in groups):
            return self._max_resume_without_windows(delivered, computed, groups, known, cap)
        top = cap
        missing = {}
        for lg in groups:
            have = np.zeros(last - first, dtype=bool)
            mine = masks[lg][first:last]
            have[: len(mine)] = mine
            for (group, ordinal), _ in delivered.partial_ends.items():
                if group == lg and first <= ordinal < last:
                    have[ordinal - first] = False
            have[: max(0, int(computed[lg]) // tpb - first)] = True
            missing[lg] = np.concatenate([[0], np.cumsum(~have)])
            if self._layout.windows[lg] is None and not have.all():
                top = min(top, (first + int(np.argmin(have)) + 1) * tpb)

        def gap(lg: int, a: int, b: int) -> bool:
            a, b = max(a, first), min(b, last)
            if b <= a:
                return False
            prefix = missing[lg]
            count = prefix[b - first] - prefix[a - first]
            tail = (candidate - 1) // tpb
            valid = max(int(computed[lg]), delivered.partial_ends.get((lg, tail), 0))
            if a <= tail < b and candidate <= valid:
                count -= prefix[tail + 1 - first] - prefix[tail - first]
            return count > 0

        # A partial end can be the only valid endpoint, even after a bidirectional rewind.
        extra = {top}
        extra.update(end for end in delivered.partial_ends.values() if known < end <= top)
        extra.update(int(end) for end in computed if known < end <= top)
        previous = None
        for candidate in merge(
            range(top // tpb * tpb, known, -tpb), sorted(extra, reverse=True), reverse=True
        ):
            if candidate == previous or candidate <= known:
                continue
            previous = candidate
            end_block = -(-candidate // tpb)
            for lg in groups:
                stale_beg, stale_end = _ranges._read_stale_range(
                    manager, self._layout, lg, candidate
                )
                if stale_end > stale_beg:
                    bad = gap(lg, first, min(stale_beg, end_block)) or gap(lg, stale_end, end_block)
                else:
                    bad = gap(lg, first, end_block)
                if bad:
                    break
            else:
                return candidate
        return known

    def _max_resume_without_windows(
        self,
        delivered: _Deliveries,
        computed: Sequence[int],
        groups: Sequence[int],
        known: int,
        cap: int,
    ) -> int:
        """``_max_resume_position`` in ``(known, cap]`` without windows or Helix: usable where each
        group has every block before the endpoint and its last up to it, so each group's first
        missing block (partial pages count) bounds it; the largest candidate within all wins."""
        tpb = int(self._layout.tokens_per_block)
        masks = delivered.delivered_block_masks
        partial_ends = delivered.partial_ends
        first, last = known // tpb, -(-cap // tpb)
        top, bound = cap, None
        for lg in groups:
            start = max(first, int(computed[lg]) // tpb)
            missing = None
            if start < last:
                segment = masks[lg][start:last]
                if len(segment):
                    at = int(segment.argmin())
                    if not segment[at]:
                        missing = start + at
                if missing is None and len(segment) < last - start:
                    missing = start + len(segment)
            for group, ordinal in partial_ends:
                if (
                    group == lg
                    and start <= ordinal < last
                    and (missing is None or ordinal < missing)
                ):
                    missing = ordinal
            if missing is None:
                continue
            top = min(top, (missing + 1) * tpb)
            # An endpoint inside the missing block reads it up to the endpoint alone.
            valid = max(int(computed[lg]), partial_ends.get((lg, missing), 0))
            group_bound = max(missing * tpb, valid)
            bound = group_bound if bound is None else min(bound, group_bound)
        limit = top if bound is None else min(top, bound)
        # The scan's candidates: whole blocks up to ``top``, ``top`` itself, partial ends and
        # computed ends, each past ``known``.
        best = limit // tpb * tpb
        if best <= known:
            best = known
        for end in (top, *partial_ends.values(), *computed):
            end = int(end)
            if known < end <= limit and end > best:
                best = end
        return best

    def _delivered_resume_floor(
        self,
        manager: KVCacheManagerV2,
        delivered: _Deliveries,
        computed: Sequence[int],
        usable: int,
    ) -> int:
        """The floor of the safe interval ending at ``usable``: a missing block behind a read window
        stays behind it at every position the interval permits, rewinds included. Each layer group
        computed the blocks below its ``computed[lg]`` tokens."""
        if usable <= self._known(computed):
            return 0
        tpb = int(self._layout.tokens_per_block)
        floor = 0
        for lg, blocks in enumerate(delivered.delivered_block_masks):
            if lg in self._recurrent_groups or self._layout.windows[lg] is None:
                continue
            first = int(computed[lg]) // tpb
            stale_beg, stale_end = _ranges._read_stale_range(manager, self._layout, lg, usable)
            beg, end = max(first, stale_beg), min(-(-usable // tpb), stale_end)
            if end <= beg:
                continue
            gaps = np.flatnonzero(~blocks[beg : min(end, len(blocks))])
            missing = beg + int(gaps[-1]) if len(gaps) else -1
            if end > len(blocks):
                missing = end - 1
            candidates = {missing} if missing >= 0 else set()
            candidates.update(
                ordinal
                for group, ordinal in delivered.partial_ends
                if group == lg and beg <= ordinal < end
            )
            for ordinal in sorted(candidates, reverse=True):
                last_read = _ranges._last_history_reading(manager, self._layout, lg, ordinal)
                valid = max(int(computed[lg]), delivered.partial_ends.get((lg, ordinal), 0))
                if valid < last_read:
                    floor = max(floor, last_read + 1)
                    break
        return floor
