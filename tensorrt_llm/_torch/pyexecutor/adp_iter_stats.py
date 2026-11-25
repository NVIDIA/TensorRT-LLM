# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Attention-DP iteration stats fanout state."""

from __future__ import annotations

import json
import re
from dataclasses import replace
from typing import Dict, List, Optional, Set

from tensorrt_llm.bindings.executor import (
    InflightBatchingStats,
    IterationStats,
    KvCacheStats,
    RequestStats,
    SpecDecodingStats,
)
from tensorrt_llm.logger import logger

from .iteration_stats_json import iteration_stats_row
from .scheduler.adp_router import RankIterStatsPayload, RankState
from .scheduler.rank_state import RankKvCacheStats, RankSchedulingCounters, RankSpecDecodingStats

_ITERATION_STATS_SCALAR_FIELDS = (
    "timestamp",
    "iter",
    "iter_latency_ms",
    "new_active_requests_queue_latency_ms",
    "num_new_active_requests",
    "num_active_requests",
    "num_queued_requests",
    "num_completed_requests",
    "max_num_active_requests",
    "gpu_mem_usage",
    "cpu_mem_usage",
    "pinned_mem_usage",
)

_ITERATION_STATS_OPTIONAL_FIELDS = ("static_batching_stats",)

# Payload counters that are all zero on a rank with nothing scheduled.
_SCHEDULED_WORK_FIELDS = (
    "num_context_requests",
    "num_ctx_tokens",
    "num_ctx_kv_tokens",
    "num_gen_requests",
    "num_gen_kv_tokens",
    "num_paused_requests",
    "num_paused_kv_tokens",
)
# The scheduled counts that give a rank a row of its own in a queue snapshot.
_ROW_WORK_FIELDS = (
    "num_context_requests",
    "num_ctx_tokens",
    "num_gen_requests",
    "num_gen_kv_tokens",
)


# The /metrics JSON key of each InflightBatchingStats field taken from a rank's payload; the other
# fields of a rank's row come from rank 0's stats.
_PAYLOAD_IFB_KEYS = {
    "num_context_requests": "numContextRequests",
    "num_ctx_tokens": "numCtxTokens",
    "num_ctx_kv_tokens": "numCtxKvTokens",
    "num_gen_requests": "numGenRequests",
    "num_gen_kv_tokens": "numGenKvTokens",
    "num_paused_requests": "numPausedRequests",
    "num_paused_kv_tokens": "numPausedKvTokens",
    "num_queued_context_requests": "numQueuedContextRequests",
    "num_queued_ctx_tokens": "numQueuedCtxTokens",
    "num_queued_gen_requests": "numQueuedGenRequests",
    "num_queued_gen_kv_tokens": "numQueuedGenKvTokens",
}
# The /metrics JSON key of each scheduling counter a rank's payload carries.
_SCHEDULING_COUNTER_KEYS = {
    "num_skipped_forward_ctx_chunks": "numSkippedForwardCtxChunks",
    "num_dropped_ctx_chunks": "numDroppedCtxChunks",
    "num_reverted_ctx_tokens": "numRevertedCtxTokens",
    "num_deferred_gen_receives": "numDeferredGenReceives",
    "gen_receive_ready_wait_ms": "genReceiveReadyWaitMs",
    "num_gen_init_cap_holds": "numGenInitCapHolds",
    "num_gen_init_transfer_budget_holds": "numGenInitTransferBudgetHolds",
    "num_ctx_decoder_hold_passes": "numCtxDecoderHoldPasses",
    "num_ctx_decoder_held_requests": "numCtxDecoderHeldRequests",
}
_NO_SCHEDULING_COUNTERS = RankSchedulingCounters()
# The /metrics JSON key of each KvCacheStats field, in the native serializer's (sorted) key order.
_KV_CACHE_STATS_KEYS = {
    "alloc_new_blocks": "allocNewBlocks",
    "alloc_total_blocks": "allocTotalBlocks",
    "cache_hit_rate": "cacheHitRate",
    "free_num_blocks": "freeNumBlocks",
    "host_slab_bytes": "hostSlabBytes",
    "max_num_blocks": "maxNumBlocks",
    "missed_blocks": "missedBlocks",
    "num_page_table_full_builds": "numPageTableFullBuilds",
    "num_page_table_reuses": "numPageTableReuses",
    "num_page_table_row_updates": "numPageTableRowUpdates",
    "page_table_full_build_ms": "pageTableFullBuildMs",
    "reused_blocks": "reusedBlocks",
    "tokens_per_block": "tokensPerBlock",
    "used_num_blocks": "usedNumBlocks",
}
# A rank's kvCacheStats text as json.dumps writes it: repr of each value, in key order.
_KV_CACHE_STATS_FORMAT = (
    "{" + ", ".join(f'"{key}": %r' for key in _KV_CACHE_STATS_KEYS.values()) + "}"
)
# The /metrics JSON key of each SpecDecodingStats field, in the native serializer's (sorted) key order.
_SPECDEC_STATS_KEYS = {
    "acceptance_length": "acceptanceLength",
    "draft_overhead": "draftOverhead",
    "iter_latency_ms": "iterLatencyMS",
    "num_accepted_tokens": "numAcceptedTokens",
    "num_draft_tokens": "numDraftTokens",
    "num_requests_with_draft_tokens": "numRequestsWithDraftTokens",
}
_SPECDEC_STATS_FORMAT = (
    "{" + ", ".join(f'"{key}": %r' for key in _SPECDEC_STATS_KEYS.values()) + "}"
)
# Marks for the parts of a rank row that differ between ranks other than 0, and their row text.
_IFB_MARK, _KV_MARK, _SPECDEC_MARK, _RANK_MARK, _QUEUED_MARK = (
    "@adp-row-inflight-batching-stats@",
    "@adp-row-kv-cache-stats@",
    "@adp-row-specdec-stats@",
    "@adp-row-attention-dp-rank@",
    "@adp-row-num-queued-requests@",
)
_IFB_TEXT, _KV_TEXT, _SPECDEC_TEXT, _RANK_TEXT, _QUEUED_TEXT = (
    json.dumps(_IFB_MARK),
    json.dumps(_KV_MARK),
    json.dumps(_SPECDEC_MARK),
    json.dumps(_RANK_MARK),
    json.dumps(_QUEUED_MARK),
)
_MARK_TEXTS = re.compile(
    "("
    + "|".join(
        re.escape(text) for text in (_IFB_TEXT, _KV_TEXT, _SPECDEC_TEXT, _RANK_TEXT, _QUEUED_TEXT)
    )
    + ")"
)


def _snapshot_kv_cache_stats(stats: KvCacheStats | None) -> RankKvCacheStats | None:
    """Capture a rank's KV counters independently of the mutable native object.

    Args:
        stats: Local KV snapshot, or None when this iteration has no measurement.

    Returns:
        Immutable scalar snapshot, or None when unavailable.
    """
    if stats is None:
        return None
    return RankKvCacheStats(*(getattr(stats, name) for name in RankKvCacheStats._fields))


def _restore_kv_cache_stats(snapshot: RankKvCacheStats) -> KvCacheStats:
    """Create a native KV stats object from one rank's captured values.

    Args:
        snapshot: Immutable counters captured with the rank's iteration payload.

    Returns:
        A fresh native stats object.
    """
    stats = KvCacheStats()
    for name, value in zip(snapshot._fields, snapshot):
        setattr(stats, name, value)
    return stats


def _snapshot_specdec_stats(stats: SpecDecodingStats | None) -> RankSpecDecodingStats | None:
    """Capture a rank's speculative-decoding counters.

    Args:
        stats: Local speculative-decoding stats, or None when they are unavailable.

    Returns:
        Immutable counter snapshot, or None when unavailable.
    """
    if stats is None:
        return None
    return RankSpecDecodingStats(
        num_draft_tokens=stats.num_draft_tokens,
        num_accepted_tokens=stats.num_accepted_tokens,
        num_requests_with_draft_tokens=stats.num_requests_with_draft_tokens,
    )


def _restore_specdec_stats(
    snapshot: RankSpecDecodingStats, rank0_stats: SpecDecodingStats | None
) -> SpecDecodingStats:
    """Create a native speculative-decoding stats object for one rank's row.

    The counters and the acceptance length derived from them are the rank's
    own. The draft timings keep rank 0's values, like the row's other timings.

    Args:
        snapshot: Counters captured with the rank's iteration payload.
        rank0_stats: Rank 0's speculative-decoding stats for the same iteration.

    Returns:
        A fresh native stats object.
    """
    stats = SpecDecodingStats()
    for name, value in zip(snapshot._fields, snapshot):
        setattr(stats, name, value)
    num_requests = snapshot.num_requests_with_draft_tokens
    stats.acceptance_length = (
        (snapshot.num_accepted_tokens + num_requests) / num_requests if num_requests else 0.0
    )
    stats.iter_latency_ms = rank0_stats.iter_latency_ms if rank0_stats is not None else 0.0
    stats.draft_overhead = rank0_stats.draft_overhead if rank0_stats is not None else 0.0
    return stats


def _kv_cache_stats_text(snapshot: RankKvCacheStats | None) -> str:
    """A rank's ``kvCacheStats`` row text, as serializing its restored stats would write it."""
    if snapshot is None:
        return "null"
    return _KV_CACHE_STATS_FORMAT % tuple(getattr(snapshot, name) for name in _KV_CACHE_STATS_KEYS)


def _specdec_stats_text(
    snapshot: RankSpecDecodingStats | None, rank0_stats: SpecDecodingStats | None
) -> str:
    """A rank's ``specDecodingStats`` row text, as serializing its restored stats would write it."""
    if snapshot is None:
        return "null"
    stats = _restore_specdec_stats(snapshot, rank0_stats)
    return _SPECDEC_STATS_FORMAT % tuple(getattr(stats, name) for name in _SPECDEC_STATS_KEYS)


class ADPIterStatsBuffer:
    """Owns pending Attention-DP IterationStats payloads and fanout.

    The executor loop thread owns this buffer. Do not mutate it from
    background request/stat threads without adding external synchronization.
    """

    def __init__(self) -> None:
        # All ranks: local per-iteration payloads waiting to piggyback on the
        # next ADP rank-state allgather.
        self._payloads: Dict[int, RankIterStatsPayload] = {}
        # Iteration IDs whose pending payload is an explicit zero placeholder,
        # not real measured local stats.
        self._synthetic_iters: Set[int] = set()
        # Rank 0 only: full IterationStats objects waiting for ADP fanout.
        self._rank0_iter_stats: Dict[int, IterationStats] = {}
        # Rank 0 only: whether the pending record is a queue snapshot (see queue()).
        self._rank0_is_queue_snapshot: Dict[int, bool] = {}
        # All ranks: pending iterations queued as an all-rank idle heartbeat.
        self._idle_heartbeat_iters: Set[int] = set()
        # What the latest finalize() call did, for the executor on rank 0:
        # whether it settled an iteration at all, the rank states if that
        # iteration was an idle heartbeat (see make_idle_heartbeat_rows),
        # whether it was the queue snapshot of a busy engine with nothing
        # scheduled on any rank, every rank's queued counters in rank order
        # (RankIterStatsPayload._QUEUED_FIELDS), and whether rank 0's row is
        # among the rows (then it is the first).
        self.last_finalize_settled = False
        self.last_idle_rank_states: Optional[List[RankState]] = None
        self.last_fanout_is_busy_queue_snapshot = False
        self.last_fanout_queued_counters: tuple = ()
        self.last_fanout_has_rank0_row = False
        # Rank 0 only: per-request stats preserved for compatibility.
        # RequestStats remain rank-0-owned under Attention-DP.
        self._rank0_req_stats: Dict[int, Optional[List[RequestStats]]] = {}
        # Rank 0 only: KV iteration stats captured with pending IterationStats.
        self._rank0_kv_iter_stats: Dict[int, Optional[Dict[int, object]]] = {}
        # Rank 0 only: per-loop CPU and GPU timings captured with the
        # IterationStats. Broadcast to all rank rows at fanout (see
        # _make_rank_iter_stats / finalize).
        self._rank0_host_step_time_ms: Dict[int, Optional[float]] = {}
        self._rank0_prev_device_step_time_ms: Dict[int, Optional[float]] = {}
        self._rank0_gpu_forward_time_ms: Dict[int, Optional[float]] = {}
        self._oldest_iter: Optional[int] = None

    @staticmethod
    def make_payload(stats: IterationStats) -> RankIterStatsPayload:
        """Pack local IterationStats fields for ADP allgather."""
        ifb = stats.inflight_batching_stats
        return RankIterStatsPayload(
            has_iter_stats=1,
            iter_stats_iter=stats.iter,
            num_context_requests=ifb.num_context_requests,
            num_ctx_tokens=ifb.num_ctx_tokens,
            num_ctx_kv_tokens=ifb.num_ctx_kv_tokens,
            num_gen_requests=ifb.num_gen_requests,
            num_gen_kv_tokens=ifb.num_gen_kv_tokens,
            num_paused_requests=ifb.num_paused_requests,
            num_paused_kv_tokens=ifb.num_paused_kv_tokens,
            num_queued_context_requests=ifb.num_queued_context_requests,
            num_queued_ctx_tokens=ifb.num_queued_ctx_tokens,
            num_queued_gen_requests=ifb.num_queued_gen_requests,
            num_queued_gen_kv_tokens=ifb.num_queued_gen_kv_tokens,
            kv_cache_stats=_snapshot_kv_cache_stats(stats.kv_cache_stats),
            cross_kv_cache_stats=_snapshot_kv_cache_stats(stats.cross_kv_cache_stats),
            specdec_stats=_snapshot_specdec_stats(stats.specdec_stats),
            scheduling_counters=RankSchedulingCounters(
                *(getattr(ifb, name) for name in RankSchedulingCounters._fields)
            ),
        )

    def queue(
        self,
        stats: IterationStats,
        req_stats: Optional[List[RequestStats]] = None,
        *,
        kv_iter_stats: Optional[Dict[int, object]] = None,
        is_rank0: bool,
        host_step_time_ms: Optional[float] = None,
        prev_device_step_time_ms: Optional[float] = None,
        gpu_forward_time_ms: Optional[float] = None,
        is_queue_snapshot: bool = False,
        idle_heartbeat: bool = False,
    ) -> None:
        """Queue local stats; rank 0 also keeps objects needed for fanout.

        ``is_queue_snapshot`` marks a record taken without a forward pass: it
        reports the queues and has no measured duration.

        ``idle_heartbeat`` marks the queue snapshot taken while every rank is
        idle (the caller's all-rank decision). It supersedes an idle
        heartbeat that is still pending: an idle engine has nothing new to
        say about an earlier moment, and a second pending snapshot would keep
        the buffer from ever draining. Every rank queues it in the same
        iteration, so the pending sets stay aligned.
        """
        # An idle heartbeat is always a queue snapshot.
        is_queue_snapshot = is_queue_snapshot or idle_heartbeat
        payload = self.make_payload(stats)
        iter_id = payload.iter_stats_iter
        if idle_heartbeat:
            for pending_iter in list(self._idle_heartbeat_iters):
                if pending_iter != iter_id:
                    self._discard(pending_iter)

        if iter_id in self._payloads and iter_id not in self._synthetic_iters:
            logger.warning(
                f"Replacing duplicate attention-DP IterationStats payload for iter {iter_id}"
            )

        self._payloads[iter_id] = payload
        self._synthetic_iters.discard(iter_id)
        self._note_payload_insert(iter_id)
        if idle_heartbeat:
            self._idle_heartbeat_iters.add(iter_id)
        else:
            self._idle_heartbeat_iters.discard(iter_id)

        if is_rank0:
            self._rank0_iter_stats[iter_id] = stats
            self._rank0_is_queue_snapshot[iter_id] = is_queue_snapshot
            self._rank0_req_stats[iter_id] = req_stats
            self._rank0_kv_iter_stats[iter_id] = kv_iter_stats
            self._rank0_host_step_time_ms[iter_id] = host_step_time_ms
            self._rank0_prev_device_step_time_ms[iter_id] = prev_device_step_time_ms
            self._rank0_gpu_forward_time_ms[iter_id] = gpu_forward_time_ms

    def next_payload(self) -> Optional[RankIterStatsPayload]:
        """Return the oldest pending stats payload to piggyback."""
        if self._oldest_iter is None:
            return None
        return self._payloads[self._oldest_iter]

    @staticmethod
    def rank0_sent_payload(all_rank_states: List[RankState]) -> bool:
        """Whether rank 0 piggybacked a payload on the exchange behind these states.

        Rank 0 may then hold more pending iterations, and the one it sent may
        still wait for a peer's payload. Rank 0 is the TP group's first rank,
        and every rank of the group gets the same answer; under helix CP
        another TP group may get a different one.

        Args:
            all_rank_states: Rank states gathered by one TP-group exchange.

        Returns:
            True if another exchange can make progress on rank 0's stats.
        """
        return any(s.rank == 0 and s.iter_stats.has_iter_stats for s in all_rank_states)

    def _note_payload_insert(self, iter_id: int) -> None:
        if self._oldest_iter is None or iter_id < self._oldest_iter:
            self._oldest_iter = iter_id

    def _recompute_oldest_iter(self) -> None:
        self._oldest_iter = min(self._payloads) if self._payloads else None

    def _ensure_zero_payload(self, iter_id: int) -> None:
        """Add a zero-work payload for a missing iteration."""
        if iter_id in self._payloads:
            return
        # This placeholder may precede an already queued measured iteration.
        # Reading the live cache manager here could put newer KV values before
        # that queued snapshot. Leave KV statistics absent for this iteration.
        self._payloads[iter_id] = RankIterStatsPayload(
            has_iter_stats=1,
            iter_stats_iter=iter_id,
        )
        self._synthetic_iters.add(iter_id)
        self._note_payload_insert(iter_id)

    def _discard(self, iter_id: int, *, recompute_oldest: bool = True) -> None:
        self._payloads.pop(iter_id, None)
        self._synthetic_iters.discard(iter_id)
        self._rank0_iter_stats.pop(iter_id, None)
        self._rank0_is_queue_snapshot.pop(iter_id, None)
        self._idle_heartbeat_iters.discard(iter_id)
        self._rank0_req_stats.pop(iter_id, None)
        self._rank0_kv_iter_stats.pop(iter_id, None)
        self._rank0_host_step_time_ms.pop(iter_id, None)
        self._rank0_prev_device_step_time_ms.pop(iter_id, None)
        self._rank0_gpu_forward_time_ms.pop(iter_id, None)
        if recompute_oldest and iter_id == self._oldest_iter:
            self._recompute_oldest_iter()

    def _drop_before(self, iter_id: int) -> None:
        changed = False
        for pending_iter in list(self._payloads):
            if pending_iter >= iter_id:
                continue
            self._discard(pending_iter, recompute_oldest=False)
            changed = True
        if changed:
            self._recompute_oldest_iter()

    def _clear_through(self, iter_id: int) -> None:
        changed = False
        for pending_iter in list(self._payloads):
            if pending_iter > iter_id:
                continue
            self._discard(pending_iter, recompute_oldest=False)
            changed = True
        if changed:
            self._recompute_oldest_iter()

    @staticmethod
    def _make_rank_iter_stats(
        rank0_stats: IterationStats,
        rank_state: RankState,
    ) -> IterationStats:
        """Build one IterationStats row for an ADP rank.

        Attention-DP emits one stats row per rank so downstream FPM consumers
        can see scheduling distribution and diagnose load imbalance. Scheduled
        and queued fields are rank-local: a rank's queued fields count the
        requests it admitted and has not scheduled yet, and rank 0's also
        count the pre-admission queues, which no rank owns yet. The queued
        load of the engine is the sum over the ranks' rows.
        """
        rank = rank_state.rank
        payload = rank_state.iter_stats
        source_ifb = rank0_stats.inflight_batching_stats

        stats = IterationStats()
        for attr in _ITERATION_STATS_SCALAR_FIELDS:
            setattr(stats, attr, getattr(rank0_stats, attr))

        # KV snapshots belong to this rank and iteration. Separate KV iteration
        # deltas retain their existing rank-0-only export ownership.
        # Native setters reject None; default-constructed fields are already absent.
        if payload.kv_cache_stats is not None:
            stats.kv_cache_stats = _restore_kv_cache_stats(payload.kv_cache_stats)
        if payload.cross_kv_cache_stats is not None:
            stats.cross_kv_cache_stats = _restore_kv_cache_stats(payload.cross_kv_cache_stats)
        for attr in _ITERATION_STATS_OPTIONAL_FIELDS:
            nested_stats = getattr(rank0_stats, attr)
            if nested_stats is not None:
                setattr(stats, attr, nested_stats)

        if payload.specdec_stats is not None:
            stats.specdec_stats = _restore_specdec_stats(
                payload.specdec_stats, rank0_stats.specdec_stats
            )

        ifb = InflightBatchingStats()
        ifb.num_context_requests = payload.num_context_requests
        ifb.num_ctx_tokens = payload.num_ctx_tokens
        ifb.num_ctx_kv_tokens = payload.num_ctx_kv_tokens
        ifb.num_gen_requests = payload.num_gen_requests
        ifb.num_gen_kv_tokens = payload.num_gen_kv_tokens
        ifb.num_paused_requests = payload.num_paused_requests
        ifb.num_paused_kv_tokens = payload.num_paused_kv_tokens
        ifb.num_scheduled_requests = ifb.num_context_requests + ifb.num_gen_requests
        ifb.num_queued_context_requests = payload.num_queued_context_requests
        ifb.num_queued_ctx_tokens = payload.num_queued_ctx_tokens
        ifb.num_queued_gen_requests = payload.num_queued_gen_requests
        ifb.num_queued_gen_kv_tokens = payload.num_queued_gen_kv_tokens
        stats.num_queued_requests = ifb.num_queued_context_requests + ifb.num_queued_gen_requests
        counters = payload.scheduling_counters or _NO_SCHEDULING_COUNTERS
        for name, value in zip(RankSchedulingCounters._fields, counters):
            setattr(ifb, name, value)

        if source_ifb is not None:
            ifb.micro_batch_id = source_ifb.micro_batch_id
            ifb.avg_num_decoded_tokens_per_iter = source_ifb.avg_num_decoded_tokens_per_iter

        if rank != 0:
            stats.num_completed_requests = 0
            stats.num_new_active_requests = 0
            stats.new_active_requests_queue_latency_ms = 0.0

        stats.inflight_batching_stats = ifb
        return stats

    def finalize(
        self,
        all_rank_states: List[RankState],
        *,
        is_rank0: bool,
        scheduler_mode: Optional[str] = None,
    ) -> List[str]:
        """Align payloads; once every rank is ready, rank 0 returns each rank's ``/metrics`` JSON
        row (``iteration_stats_row`` of that rank's stats) in rank order."""
        self.last_finalize_settled = False
        self.last_idle_rank_states = None
        self.last_fanout_is_busy_queue_snapshot = False
        self.last_fanout_queued_counters = ()
        self.last_fanout_has_rank0_row = False
        pending_states = [s for s in all_rank_states if s.iter_stats.has_iter_stats]
        if not pending_states:
            return []

        rank0_state = next((s for s in all_rank_states if s.rank == 0), None)
        if rank0_state is None or not rank0_state.iter_stats.has_iter_stats:
            logger.debug("Waiting for rank 0 attention-DP IterationStats payload before fanout")
            return []

        # Rank 0 owns the stats queue consumed by get_stats(), so converge all
        # ranks to rank 0's pending iteration. Ranks without local work for
        # that iteration contribute an explicit zero payload on the next
        # piggyback allgather instead of forcing a mixed-iteration skip.
        iter_stats_iter = rank0_state.iter_stats.iter_stats_iter
        self._drop_before(iter_stats_iter)
        self._ensure_zero_payload(iter_stats_iter)

        matching_states = [
            s
            for s in all_rank_states
            if (s.iter_stats.has_iter_stats and s.iter_stats.iter_stats_iter == iter_stats_iter)
        ]
        if len(matching_states) != len(all_rank_states):
            logger.debug(
                "Waiting for attention-DP IterationStats payloads for rank 0 "
                f"iter {iter_stats_iter}: received "
                f"{len(matching_states)}/{len(all_rank_states)} matching "
                "rank payloads"
            )
            return []

        rows: List[str] = []
        if is_rank0:
            rank0_stats = self._rank0_iter_stats.get(iter_stats_iter)
            if rank0_stats is None:
                logger.warning(
                    "Skipping attention-DP IterationStats fanout on "
                    f"rank 0: pending IterationStats object is missing for "
                    f"iter {iter_stats_iter}"
                )
                self._clear_through(iter_stats_iter)
                self.last_finalize_settled = True
                return []
            states = sorted(matching_states, key=lambda s: s.rank)
            queued_counters = tuple(
                tuple(getattr(s.iter_stats, name) for name in RankIterStatsPayload._QUEUED_FIELDS)
                for s in states
            )
            any_queued = any(any(counters) for counters in queued_counters)
            no_scheduled_work = not any(
                getattr(s.iter_stats, name) for s in states for name in _SCHEDULED_WORK_FIELDS
            )
            is_queue_snapshot = self._rank0_is_queue_snapshot.get(iter_stats_iter, False)
            is_idle_heartbeat = iter_stats_iter in self._idle_heartbeat_iters
            # A peer's measured record can share the iteration with rank 0's
            # snapshot; such a fanout is exported like any measured one.
            self.last_fanout_is_busy_queue_snapshot = (
                is_queue_snapshot and not is_idle_heartbeat and no_scheduled_work
            )
            self.last_fanout_queued_counters = queued_counters
            idle_with_empty_queues = is_idle_heartbeat and not any_queued
            exported_states = states
            if is_queue_snapshot and not idle_with_empty_queues:
                # An idle heartbeat with empty queues reports every rank. Any other queue
                # snapshot has a row only for the ranks with work to report: scheduled or
                # queued on that rank.
                exported_states = [
                    s
                    for s, counters in zip(states, queued_counters)
                    if any(getattr(s.iter_stats, name) for name in _ROW_WORK_FIELDS)
                    or any(counters)
                ]
            self.last_fanout_has_rank0_row = any(s.rank == 0 for s in exported_states)
            rows = self._rank_rows(
                rank0_stats,
                exported_states,
                self._rank0_req_stats.get(iter_stats_iter),
                self._rank0_kv_iter_stats.get(iter_stats_iter),
                # Per-loop timings exist on rank 0 only; every rank's row carries them (ADP ranks
                # step in lockstep).
                timings=dict(
                    host_step_time_ms=self._rank0_host_step_time_ms.get(iter_stats_iter),
                    prev_device_step_time_ms=self._rank0_prev_device_step_time_ms.get(
                        iter_stats_iter
                    ),
                    scheduler_mode=scheduler_mode,
                    gpu_forward_time_ms=self._rank0_gpu_forward_time_ms.get(iter_stats_iter),
                ),
                is_queue_snapshot=is_queue_snapshot,
            )
            if idle_with_empty_queues and no_scheduled_work:
                self.last_idle_rank_states = states

        self._clear_through(iter_stats_iter)
        self.last_finalize_settled = True
        return rows

    def make_idle_heartbeat_rows(
        self, rank0_stats: IterationStats, scheduler_mode: Optional[str] = None
    ) -> List[str]:
        """Repeat the last idle-heartbeat fanout with a fresh rank-0 snapshot.

        While every rank stays idle the peers are parked in the request
        broadcast, so their rows cannot change. Rank 0 can therefore publish
        the next heartbeat locally, one row per rank as in finalize(), without
        waking them for another exchange. Peer rows reuse the payloads of the
        last exchanged heartbeat; rank 0's row and the shared scalars come
        from ``rank0_stats``.

        Args:
            rank0_stats: Rank 0's current queue snapshot.
            scheduler_mode: The rows' ``schedulerMode``, as in finalize().

        Returns:
            One ``/metrics`` JSON row per rank, or an empty list when the
            latest finalize() call did not fan out an idle heartbeat.
        """
        rank_states = self.last_idle_rank_states
        if rank_states is None:
            return []
        rank0_payload = self.make_payload(rank0_stats)
        return self._rank_rows(
            rank0_stats,
            [replace(s, iter_stats=rank0_payload) if s.rank == 0 else s for s in rank_states],
            None,
            None,
            timings=dict(
                host_step_time_ms=0.0,
                prev_device_step_time_ms=0.0,
                scheduler_mode=scheduler_mode,
                gpu_forward_time_ms=0.0,
            ),
            is_queue_snapshot=True,
        )

    def _rank_rows(
        self,
        rank0_stats: IterationStats,
        states: List[RankState],
        req_stats: Optional[List[RequestStats]],
        kv_iter_stats: Optional[Dict[int, object]],
        timings: Dict[str, object],
        is_queue_snapshot: bool,
    ) -> List[str]:
        """Rows of ranks other than 0 differ only in their payload counts (with the
        ``numQueuedRequests`` those give), KV cache stats, speculative-decoding stats and rank:
        the first is built in full and the others are spliced into its text.
        ``is_queue_snapshot`` marks every row (see queue())."""
        rows: List[str] = []
        template = None
        for state in states:
            if state.rank == 0:
                stats = self._make_rank_iter_stats(rank0_stats, state)
                row = iteration_stats_row(
                    stats,
                    req_stats,
                    kv_iter_stats,
                    0,
                    **timings,
                    is_queue_snapshot=is_queue_snapshot,
                )
                rows.append(json.dumps(row))
                continue
            if template is None:
                stats = self._make_rank_iter_stats(rank0_stats, state)
                row = iteration_stats_row(
                    stats,
                    attention_dp_rank=state.rank,
                    **timings,
                    is_queue_snapshot=is_queue_snapshot,
                )
                ifb = row["inflightBatchingStats"]
                row["inflightBatchingStats"] = _IFB_MARK
                row["kvCacheStats"] = _KV_MARK
                row["specDecodingStats"] = _SPECDEC_MARK
                row["attentionDpRank"] = _RANK_MARK
                row["numQueuedRequests"] = _QUEUED_MARK
                template = (_MARK_TEXTS.split(json.dumps(row)), ifb)
            pieces, ifb = template
            payload = state.iter_stats
            for name, key in _PAYLOAD_IFB_KEYS.items():
                ifb[key] = getattr(payload, name)
            counters = payload.scheduling_counters or _NO_SCHEDULING_COUNTERS
            for name, key in _SCHEDULING_COUNTER_KEYS.items():
                ifb[key] = getattr(counters, name)
            ifb["numScheduledRequests"] = payload.num_context_requests + payload.num_gen_requests
            fills = {
                _IFB_TEXT: json.dumps(ifb),
                _KV_TEXT: _kv_cache_stats_text(payload.kv_cache_stats),
                _SPECDEC_TEXT: _specdec_stats_text(
                    payload.specdec_stats, rank0_stats.specdec_stats
                ),
                _RANK_TEXT: str(state.rank),
                _QUEUED_TEXT: str(
                    payload.num_queued_context_requests + payload.num_queued_gen_requests
                ),
            }
            rows.append("".join(fills.get(piece, piece) for piece in pieces))
        return rows
