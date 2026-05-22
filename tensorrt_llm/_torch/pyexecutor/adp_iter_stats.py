# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Attention-DP iteration stats fanout state."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Set

from tensorrt_llm.bindings.executor import InflightBatchingStats, IterationStats, RequestStats
from tensorrt_llm.logger import logger

from .scheduler.adp_router import RankIterStatsPayload, RankState

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

_ITERATION_STATS_OPTIONAL_FIELDS = (
    "kv_cache_stats",
    "cross_kv_cache_stats",
    "static_batching_stats",
    "specdec_stats",
)


@dataclass
class ADPIterStatsRecord:
    """Append-ready stats row produced by Attention-DP fanout."""

    stats: IterationStats
    req_stats: Optional[List[RequestStats]]
    kv_iter_stats: Optional[Dict[int, object]]
    attention_dp_rank: int


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
        # Rank 0 only: per-request stats preserved for compatibility.
        # RequestStats remain rank-0-owned under Attention-DP.
        self._rank0_req_stats: Dict[int, Optional[List[RequestStats]]] = {}
        # Rank 0 only: KV iteration stats captured with pending IterationStats.
        self._rank0_kv_iter_stats: Dict[int, Optional[Dict[int, object]]] = {}
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
        )

    def queue(
        self,
        stats: IterationStats,
        req_stats: Optional[List[RequestStats]] = None,
        *,
        kv_iter_stats: Optional[Dict[int, object]] = None,
        is_rank0: bool,
    ) -> None:
        """Queue local stats; rank 0 also keeps objects needed for fanout."""
        payload = self.make_payload(stats)
        iter_id = payload.iter_stats_iter

        if iter_id in self._payloads and iter_id not in self._synthetic_iters:
            logger.warning(
                f"Replacing duplicate attention-DP IterationStats payload for iter {iter_id}"
            )

        self._payloads[iter_id] = payload
        self._synthetic_iters.discard(iter_id)
        self._note_payload_insert(iter_id)

        if is_rank0:
            self._rank0_iter_stats[iter_id] = stats
            self._rank0_req_stats[iter_id] = req_stats
            self._rank0_kv_iter_stats[iter_id] = kv_iter_stats

    def next_payload(self) -> Optional[RankIterStatsPayload]:
        """Return the oldest pending stats payload to piggyback."""
        if self._oldest_iter is None:
            return None
        return self._payloads[self._oldest_iter]

    def _note_payload_insert(self, iter_id: int) -> None:
        if self._oldest_iter is None or iter_id < self._oldest_iter:
            self._oldest_iter = iter_id

    def _recompute_oldest_iter(self) -> None:
        self._oldest_iter = min(self._payloads) if self._payloads else None

    def _ensure_zero_payload(self, iter_id: int) -> None:
        """Add a zero payload when this rank had no work for an iteration."""
        if iter_id in self._payloads:
            return
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
        self._rank0_req_stats.pop(iter_id, None)
        self._rank0_kv_iter_stats.pop(iter_id, None)
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
        fields are rank-local. Queued fields remain rank-0/global because the
        executor request queue lives on rank 0.
        """
        rank = rank_state.rank
        payload = rank_state.iter_stats
        source_ifb = rank0_stats.inflight_batching_stats

        stats = IterationStats()
        for attr in _ITERATION_STATS_SCALAR_FIELDS:
            setattr(stats, attr, getattr(rank0_stats, attr))

        # Optional nested stats are copied when present. KV iteration deltas
        # are attached separately and remain rank-0-only to avoid double-logging
        # global KV-cache deltas.
        for attr in _ITERATION_STATS_OPTIONAL_FIELDS:
            nested_stats = getattr(rank0_stats, attr)
            if nested_stats is not None:
                setattr(stats, attr, nested_stats)

        ifb = InflightBatchingStats()
        ifb.num_context_requests = payload.num_context_requests
        ifb.num_ctx_tokens = payload.num_ctx_tokens
        ifb.num_ctx_kv_tokens = payload.num_ctx_kv_tokens
        ifb.num_gen_requests = payload.num_gen_requests
        ifb.num_gen_kv_tokens = payload.num_gen_kv_tokens
        ifb.num_paused_requests = payload.num_paused_requests
        ifb.num_paused_kv_tokens = payload.num_paused_kv_tokens
        ifb.num_scheduled_requests = ifb.num_context_requests + ifb.num_gen_requests

        if source_ifb is not None:
            ifb.micro_batch_id = source_ifb.micro_batch_id
            ifb.avg_num_decoded_tokens_per_iter = source_ifb.avg_num_decoded_tokens_per_iter
            if rank == 0:
                ifb.num_queued_context_requests = source_ifb.num_queued_context_requests
                ifb.num_queued_ctx_tokens = source_ifb.num_queued_ctx_tokens
                ifb.num_queued_gen_requests = source_ifb.num_queued_gen_requests
                ifb.num_queued_gen_kv_tokens = source_ifb.num_queued_gen_kv_tokens

        if rank != 0:
            stats.num_queued_requests = 0
            stats.num_completed_requests = 0
            stats.num_new_active_requests = 0
            stats.new_active_requests_queue_latency_ms = 0.0

        stats.inflight_batching_stats = ifb
        return stats

    def finalize(
        self, all_rank_states: List[RankState], *, is_rank0: bool
    ) -> List[ADPIterStatsRecord]:
        """Align payloads and return per-rank rows once all ranks are ready."""
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

        records: List[ADPIterStatsRecord] = []
        if is_rank0:
            rank0_stats = self._rank0_iter_stats.get(iter_stats_iter)
            if rank0_stats is None:
                logger.warning(
                    "Skipping attention-DP IterationStats fanout on "
                    f"rank 0: pending IterationStats object is missing for "
                    f"iter {iter_stats_iter}"
                )
                self._clear_through(iter_stats_iter)
                return []

            req_stats = self._rank0_req_stats.get(iter_stats_iter)
            kv_iter_stats = self._rank0_kv_iter_stats.get(iter_stats_iter)

            for rank_state in sorted(matching_states, key=lambda s: s.rank):
                rank = rank_state.rank
                records.append(
                    ADPIterStatsRecord(
                        stats=self._make_rank_iter_stats(rank0_stats, rank_state),
                        req_stats=req_stats if rank == 0 else None,
                        kv_iter_stats=kv_iter_stats if rank == 0 else None,
                        attention_dp_rank=rank,
                    )
                )

        self._clear_through(iter_stats_iter)
        return records
