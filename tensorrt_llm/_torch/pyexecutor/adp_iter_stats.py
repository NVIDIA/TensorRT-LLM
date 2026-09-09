# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Attention-DP iteration stats fanout state."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Set

from tensorrt_llm.bindings.executor import IterationStats, RequestStats
from tensorrt_llm.logger import logger

from .scheduler.adp_router import RankIterStatsPayload, RankState


@dataclass
class ADPIterStatsRecord:
    """Append-ready stats row produced by Attention-DP fanout.

    All rank rows share the rank-0 ``IterationStats`` object. The compact
    rank payload is applied only when the row is serialized, keeping the
    executor loop from cloning nested nanobind stats objects per ADP rank.
    """

    stats: IterationStats
    rank_iter_stats: RankIterStatsPayload
    req_stats: Optional[List[RequestStats]]
    kv_iter_stats: Optional[Dict[int, object]]
    attention_dp_rank: int
    # Per-loop CPU wall and GPU forward time captured on rank 0 alongside
    # the IterationStats at queue time. Currently broadcast unchanged to all
    # rank rows during fanout (true per-rank timing would require widening
    # the rank-state allgather payload). Under steady-state ADP all ranks
    # step in lockstep, so this is a reasonable approximation.
    host_step_time_ms: Optional[float] = None
    prev_device_step_time_ms: Optional[float] = None
    gpu_forward_time_ms: Optional[float] = None


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
    ) -> None:
        """Queue local stats; rank 0 also keeps objects needed for fanout."""
        payload = self.make_payload(stats)
        self.queue_payload(payload)
        iter_id = payload.iter_stats_iter

        if is_rank0:
            self._rank0_iter_stats[iter_id] = stats
            self._rank0_req_stats[iter_id] = req_stats
            self._rank0_kv_iter_stats[iter_id] = kv_iter_stats
            self._rank0_host_step_time_ms[iter_id] = host_step_time_ms
            self._rank0_prev_device_step_time_ms[iter_id] = prev_device_step_time_ms
            self._rank0_gpu_forward_time_ms[iter_id] = gpu_forward_time_ms

    def queue_payload(self, payload: RankIterStatsPayload) -> None:
        """Queue a compact payload without constructing full iteration stats."""
        iter_id = payload.iter_stats_iter

        if iter_id in self._payloads and iter_id not in self._synthetic_iters:
            logger.warning(
                f"Replacing duplicate attention-DP IterationStats payload for iter {iter_id}"
            )

        self._payloads[iter_id] = payload
        self._synthetic_iters.discard(iter_id)
        self._note_payload_insert(iter_id)

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
            host_step_time_ms = self._rank0_host_step_time_ms.get(iter_stats_iter)
            prev_device_step_time_ms = self._rank0_prev_device_step_time_ms.get(iter_stats_iter)
            gpu_forward_time_ms = self._rank0_gpu_forward_time_ms.get(iter_stats_iter)

            for rank_state in sorted(matching_states, key=lambda s: s.rank):
                rank = rank_state.rank
                records.append(
                    ADPIterStatsRecord(
                        stats=rank0_stats,
                        rank_iter_stats=rank_state.iter_stats,
                        req_stats=req_stats if rank == 0 else None,
                        kv_iter_stats=kv_iter_stats if rank == 0 else None,
                        attention_dp_rank=rank,
                        host_step_time_ms=host_step_time_ms,
                        prev_device_step_time_ms=prev_device_step_time_ms,
                        gpu_forward_time_ms=gpu_forward_time_ms,
                    )
                )

        self._clear_through(iter_stats_iter)
        return records
