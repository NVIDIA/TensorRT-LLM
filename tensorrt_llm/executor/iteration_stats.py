# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Owned iteration observations and client-side report assembly."""

from __future__ import annotations

import json
from dataclasses import dataclass, field, fields
from types import SimpleNamespace


@dataclass(slots=True)
class InflightBatchingSnapshot:
    num_scheduled_requests: int = 0
    num_context_requests: int = 0
    num_gen_requests: int = 0
    num_paused_requests: int = 0
    num_ctx_tokens: int = 0
    micro_batch_id: int = 0
    avg_num_decoded_tokens_per_iter: float = 0.0
    num_ctx_kv_tokens: int = 0
    num_gen_kv_tokens: int = 0
    num_queued_context_requests: int = 0
    num_queued_ctx_tokens: int = 0
    num_queued_gen_requests: int = 0
    num_queued_gen_kv_tokens: int = 0
    num_paused_kv_tokens: int = 0


@dataclass(slots=True)
class KVCacheStatsSnapshot:
    max_num_blocks: int = 0
    free_num_blocks: int = 0
    used_num_blocks: int = 0
    tokens_per_block: int = 0
    alloc_total_blocks: int = 0
    alloc_new_blocks: int = 0
    reused_blocks: int = 0
    missed_blocks: int = 0
    cache_hit_rate: float = 0.0


@dataclass(slots=True)
class SpecDecodingSnapshot:
    num_draft_tokens: int = 0
    num_accepted_tokens: int = 0
    num_requests_with_draft_tokens: int = 0
    iter_latency_ms: float = 0.0


@dataclass(slots=True)
class DisServingRequestSnapshot:
    kv_cache_transfer_ms: float = 0.0
    kv_cache_size: int = 0


@dataclass(slots=True)
class RequestStatsSnapshot:
    id: int = 0
    stage: str = "QUEUED"
    context_prefill_position: int = 0
    num_generated_tokens: int = 0
    avg_num_decoded_tokens_per_iter: float = 0.0
    scheduled: bool = False
    paused: bool = False
    dis_serving_stats: DisServingRequestSnapshot | None = None
    alloc_total_blocks_per_request: int = 0
    alloc_new_blocks_per_request: int = 0
    reused_blocks_per_request: int = 0
    missed_blocks_per_request: int = 0
    kv_cache_hit_rate_per_request: float = 0.0


@dataclass(slots=True)
class IterationStatsSnapshot:
    """Scalar state owned by one batch, sealed by handing its frame to the buffer."""

    timestamp: str = ""
    iter: int = 0
    iter_latency_ms: float = 0.0
    new_active_requests_queue_latency_ms: float = 0.0
    num_new_active_requests: int = 0
    num_active_requests: int = 0
    num_queued_requests: int = 0
    num_completed_requests: int = 0
    max_num_active_requests: int = 0
    gpu_mem_usage: int = 0
    cpu_mem_usage: int = 0
    pinned_mem_usage: int = 0
    kv_cache_stats: KVCacheStatsSnapshot | None = None
    inflight_batching_stats: InflightBatchingSnapshot = field(
        default_factory=InflightBatchingSnapshot
    )
    specdec_stats: SpecDecodingSnapshot | None = None


@dataclass(frozen=True, slots=True)
class RankStatsSnapshot:
    rank: int
    num_context_requests: int
    num_ctx_tokens: int
    num_ctx_kv_tokens: int
    num_gen_requests: int
    num_gen_kv_tokens: int
    num_paused_requests: int
    num_paused_kv_tokens: int

    @classmethod
    def capture(cls, rank: int, payload: object) -> RankStatsSnapshot:
        return cls(
            rank=rank,
            num_context_requests=payload.num_context_requests,
            num_ctx_tokens=payload.num_ctx_tokens,
            num_ctx_kv_tokens=payload.num_ctx_kv_tokens,
            num_gen_requests=payload.num_gen_requests,
            num_gen_kv_tokens=payload.num_gen_kv_tokens,
            num_paused_requests=payload.num_paused_requests,
            num_paused_kv_tokens=payload.num_paused_kv_tokens,
        )


@dataclass(slots=True)
class IterationStatsFrame:
    """Transport record containing no requests, native bindings, or CUDA events."""

    stats: IterationStatsSnapshot
    req_stats: list[RequestStatsSnapshot] | None = None
    kv_iter_stats: object | None = None
    attention_dp_rank: int | None = None
    host_step_time_ms: float | None = None
    prev_device_step_time_ms: float | None = None
    scheduler_mode: str = "non_overlap"
    gpu_forward_time_ms: float | None = None
    rank_payloads: tuple[RankStatsSnapshot, ...] = ()
    rank: int | None = None
    sequence: int = 0
    dropped_frames: int = 0


def _camel_case(name: str) -> str:
    first, *rest = name.split("_")
    return first + "".join(word.capitalize() for word in rest)


def _scalar_fields(snapshot: object) -> dict:
    return {_camel_case(f.name): getattr(snapshot, f.name) for f in fields(snapshot)}


def _request_report(snapshot: RequestStatsSnapshot) -> dict:
    report = _scalar_fields(snapshot)
    # The public JSON schema maps stages without an explicit entry to QUEUED.
    if snapshot.stage not in (
        "QUEUED",
        "CONTEXT_IN_PROGRESS",
        "GENERATION_IN_PROGRESS",
        "GENERATION_COMPLETE",
    ):
        report["stage"] = "QUEUED"
    disagg = snapshot.dis_serving_stats
    report["disServingStats"] = (
        {"kvCacheTransferMS": disagg.kv_cache_transfer_ms, "kvCacheSize": disagg.kv_cache_size}
        if disagg is not None
        else None
    )
    return report


def _iteration_report(snapshot: IterationStatsSnapshot) -> dict:
    # These field names and defaults are the executor's public JSON contract.
    report = {
        "timestamp": snapshot.timestamp,
        "iter": snapshot.iter,
        "iterLatencyMS": snapshot.iter_latency_ms,
        "newActiveRequestsQueueLatencyMS": snapshot.new_active_requests_queue_latency_ms,
        "numNewActiveRequests": snapshot.num_new_active_requests,
        "numActiveRequests": snapshot.num_active_requests,
        "numQueuedRequests": snapshot.num_queued_requests,
        "numCompletedRequests": snapshot.num_completed_requests,
        "maxNumActiveRequests": snapshot.max_num_active_requests,
        "maxBatchSizeStatic": 0,
        "maxBatchSizeTunerRecommended": 0,
        "maxBatchSizeRuntime": 0,
        "maxNumTokensStatic": 0,
        "maxNumTokensTunerRecommended": 0,
        "maxNumTokensRuntime": 0,
        "gpuMemUsage": snapshot.gpu_mem_usage,
        "cpuMemUsage": snapshot.cpu_mem_usage,
        "pinnedMemUsage": snapshot.pinned_mem_usage,
        "kvCacheStats": (
            _scalar_fields(snapshot.kv_cache_stats) if snapshot.kv_cache_stats is not None else None
        ),
        "staticBatchingStats": {
            "numScheduledRequests": 0,
            "numContextRequests": 0,
            "numCtxTokens": 0,
            "numGenTokens": 0,
            "emptyGenSlots": 0,
        },
        "inflightBatchingStats": _scalar_fields(snapshot.inflight_batching_stats),
        "specDecodingStats": None,
    }
    spec = snapshot.specdec_stats
    if spec is not None:
        count = spec.num_requests_with_draft_tokens
        report["specDecodingStats"] = {
            "numDraftTokens": spec.num_draft_tokens,
            "numAcceptedTokens": spec.num_accepted_tokens,
            "numRequestsWithDraftTokens": count,
            "acceptanceLength": (spec.num_accepted_tokens + count) / count if count else 0.0,
            "iterLatencyMS": spec.iter_latency_ms,
            "draftOverhead": (
                spec.iter_latency_ms / snapshot.iter_latency_ms
                if snapshot.iter_latency_ms > 0
                else 0.0
            ),
        }
    return report


def _materialize_frame(frame: IterationStatsFrame) -> list[dict]:
    base = _iteration_report(frame.stats)
    base["schedulerMode"] = frame.scheduler_mode
    base["statsSequence"] = frame.sequence
    base["statsDroppedFrames"] = frame.dropped_frames
    for key, value in (
        ("hostStepTimeMS", frame.host_step_time_ms),
        ("prevDeviceStepTimeMS", frame.prev_device_step_time_ms),
        ("gpuForwardTimeMS", frame.gpu_forward_time_ms),
    ):
        if value is not None:
            base[key] = value
    if frame.rank is not None:
        base["rank"] = frame.rank

    extras = {}
    if frame.req_stats:
        extras["requestStats"] = [_request_report(row) for row in frame.req_stats]
    if frame.kv_iter_stats is not None:
        from .._torch.pyexecutor.kv_cache_stats import append_kv_cache_iteration_stats

        append_kv_cache_iteration_stats(extras, frame.kv_iter_stats)

    if not frame.rank_payloads:
        base.update(extras)
        base["attentionDpRank"] = frame.attention_dp_rank or 0
        return [base]

    result = []
    for payload in frame.rank_payloads:
        row = dict(base)
        row["attentionDpRank"] = payload.rank
        ifb = dict(base["inflightBatchingStats"])
        ifb.update(
            numContextRequests=payload.num_context_requests,
            numCtxTokens=payload.num_ctx_tokens,
            numCtxKvTokens=payload.num_ctx_kv_tokens,
            numGenRequests=payload.num_gen_requests,
            numGenKvTokens=payload.num_gen_kv_tokens,
            numPausedRequests=payload.num_paused_requests,
            numPausedKvTokens=payload.num_paused_kv_tokens,
            numScheduledRequests=payload.num_context_requests + payload.num_gen_requests,
        )
        if payload.rank != 0:
            for key in (
                "numQueuedContextRequests",
                "numQueuedCtxTokens",
                "numQueuedGenRequests",
                "numQueuedGenKvTokens",
            ):
                ifb[key] = 0
            for key in (
                "numQueuedRequests",
                "numCompletedRequests",
                "numNewActiveRequests",
                "newActiveRequestsQueueLatencyMS",
            ):
                row[key] = 0
        else:
            row.update(extras)
        row["inflightBatchingStats"] = ifb
        result.append(row)
    return result


def prepare_stats_batch(stats_batch: list) -> list:
    """Prepare RPC data without materializing Python executor observations."""
    result = []
    for entry in stats_batch:
        if isinstance(entry, (IterationStatsFrame, str, dict)):
            result.append(entry)
        elif isinstance(entry, tuple):
            from .base_worker import BaseWorker

            result.append(BaseWorker._stats_serializer(entry))
        else:
            raise TypeError(f"Unsupported iteration statistics record: {type(entry).__name__}")
    return result


def materialize_stats_batch(stats_batch: list) -> list[dict]:
    """Build public reports in the client process after receiving observations."""
    result = []
    for entry in stats_batch:
        if isinstance(entry, IterationStatsFrame):
            result.extend(_materialize_frame(entry))
        elif isinstance(entry, str):
            report = json.loads(entry)
            if not isinstance(report, dict):
                raise TypeError("Iteration statistics JSON must contain an object")
            result.append(report)
        elif isinstance(entry, dict):
            result.append(entry)
        elif isinstance(entry, tuple):
            from .base_worker import BaseWorker

            result.append(json.loads(BaseWorker._stats_serializer(entry)))
        else:
            raise TypeError(f"Unsupported iteration statistics record: {type(entry).__name__}")
    return result


_LEGACY_KV_FIELDS = (
    "primary_max_num_blocks",
    "primary_free_num_blocks",
    "primary_used_num_blocks",
    "primary_evictable_num_blocks",
    "primary_peak_free_num_blocks",
    "primary_peak_used_num_blocks",
    "primary_peak_evictable_num_blocks",
    "secondary_max_num_blocks",
    "secondary_free_num_blocks",
    "secondary_used_num_blocks",
    "secondary_evictable_num_blocks",
    "secondary_peak_free_num_blocks",
    "secondary_peak_used_num_blocks",
    "secondary_peak_evictable_num_blocks",
    "iter_alloc_total_blocks",
    "iter_alloc_new_blocks",
    "iter_reused_blocks",
    "iter_full_reused_blocks",
    "iter_partial_reused_blocks",
    "iter_missed_blocks",
    "iter_cache_hit_rate",
    "iter_gen_alloc_blocks",
    "iter_onboard_blocks",
    "iter_onboard_bytes",
    "iter_offload_blocks",
    "iter_offload_bytes",
    "iter_intra_device_copy_blocks",
    "iter_intra_device_copy_bytes",
    "iter_host_dropped_blocks",
    "iter_host_dropped_bytes",
)


def capture_legacy_kv_iteration_stats(stats: dict | None) -> dict | None:
    """Copy legacy native counter rows before transporting them across processes."""
    if stats is None:
        return None
    return {
        key: SimpleNamespace(**{name: getattr(value, name) for name in _LEGACY_KV_FIELDS})
        for key, value in stats.items()
    }
