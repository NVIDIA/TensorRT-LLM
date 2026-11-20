# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Executor-loop logger exposure for already collected performance records."""

from collections.abc import Callable, Mapping
from datetime import datetime

from .perf import (
    KV_MOVEMENT_WINDOW_FIELDS,
    MetricKey,
    MetricValue,
    PerfMetricsRecord,
    PoolRows,
    block_count,
    kv_cache_utilization,
    kv_pool_rows,
    pool_counts,
    total_pool_count,
)


def _format_count(value: object) -> str:
    number = block_count(value)
    return "N/A" if number is None else str(number)


def _pool_counts(
    rows: PoolRows | None, groups: tuple[int, ...] | None, field: str
) -> tuple[str, str]:
    if rows is None or groups is None:
        return "N/A", "N/A"
    counts = pool_counts(rows, groups, field)
    vector = "/".join(_format_count(value) for value in counts.values()) if groups else "[]"
    return _format_count(total_pool_count(counts)), vector


def _kv_diagnostics(snapshot: Mapping[MetricKey, MetricValue], *, movement_fresh: bool) -> str:
    fields = {"num_paused_requests": _format_count(snapshot.get("numPausedRequests"))}
    allocation = snapshot.get("kvCacheStats")
    for name, source in (
        ("kv_alloc_total_blocks", "allocTotalBlocks"),
        ("kv_alloc_new_blocks", "allocNewBlocks"),
    ):
        fields[name] = _format_count(
            allocation.get(source) if isinstance(allocation, Mapping) else None
        )

    hot_pools, movements = kv_pool_rows(snapshot)
    groups = (
        None
        if hot_pools is None and movements is None
        else tuple(sorted(set(hot_pools or ()) | set(movements or ())))
    )
    fields["kv_pool_group_ids"] = (
        "N/A" if groups is None else "/".join(map(str, groups)) if groups else "[]"
    )
    for name, source in (
        ("kv_free_blocks", "freeNumBlocks"),
        ("kv_evictable_blocks", "evictableNumBlocks"),
    ):
        fields[name], fields[f"{name}_by_pg"] = _pool_counts(hot_pools, groups, source)
    for _, name, source in KV_MOVEMENT_WINDOW_FIELDS:
        fields[name], fields[f"{name}_by_pg"] = _pool_counts(
            movements if movement_fresh else None, groups, source
        )
    return ", ".join(f"{name} = {value}" for name, value in fields.items())


class LoggerExposure:
    """Format collected observations without accessing the runtime or CUDA."""

    def __init__(self, write: Callable[[str], None]) -> None:
        self._write = write
        self.reset()

    def reset(self) -> None:
        """Discard fetched baselines and exposed snapshots for a new context."""
        self._previous_fetched = 0
        self._previous_rank_snapshot: Mapping[MetricKey, MetricValue] | None = None

    def emit(self, record: PerfMetricsRecord) -> None:
        """Expose one collected loop in the existing single-line text format."""
        if record.kind != "loop":
            return
        values = record.per_iter
        if values["loopSequence"] == 0:
            self._previous_fetched = 0
        fetched = values["numFetchedRequestsTotal"]
        num_fetched = max(0, fetched - self._previous_fetched)
        self._previous_fetched = fetched
        device = values.get("prevDeviceStepTimeMS")
        device_text = "N/A" if device is None else f"{device}ms"
        snapshot = record.per_rank
        kv_stats = snapshot.get("kvCacheStats", {})
        if not isinstance(kv_stats, Mapping):
            kv_stats = {}
        utilization = kv_cache_utilization(kv_stats)
        kv_text = "N/A" if utilization is None else f"{utilization:.3f}"
        kv_reuse_text = ""
        if "reusedBlocks" in kv_stats:
            hit_rate = kv_stats.get("cacheHitRate")
            hit_text = f"{hit_rate:.3f}" if isinstance(hit_rate, (int, float)) else "N/A"
            kv_reuse_text = (
                f"kv_reused_blocks = {_format_count(kv_stats['reusedBlocks'])}, "
                f"kv_missed_blocks = {_format_count(kv_stats.get('missedBlocks'))}, "
                f"kv_hit_rate = {hit_text}, "
            )
        snapshot_fresh = bool(snapshot) and snapshot is not self._previous_rank_snapshot
        movement_fresh = snapshot_fresh and isinstance(
            snapshot.get("kvCacheIterationStatsByPoolGroup"), Mapping
        )
        diagnostic_text = _kv_diagnostics(snapshot, movement_fresh=movement_fresh)
        capture_time = snapshot.get("captureTime")
        capture_text = "N/A" if capture_time is None else str(capture_time)
        timestamp = datetime.fromtimestamp(values["timestamp"]).strftime("%Y-%m-%d %H:%M:%S")
        self._write(
            f"iter = {values['iter']}, "
            f"global_rank = {values['globalRank']}, "
            f"rank = {values['rank']}, "
            f"num_scheduled_requests = {values['numScheduledRequests']}, "
            f"adp_dummy_ctx_tokens = {values['adpDummyCtxTokens']}, "
            f"adp_dummy_gen_tokens = {values['adpDummyGenTokens']}, "
            f"kv_cache_util = {kv_text}, "
            f"{kv_reuse_text}"
            f"currank_total_requests = {values['numFetchedRequestsTotal']}/{values['numFetchedRequestsAllRanks']}, "
            f"host_step_time = {values['hostStepTimeMS']}ms, "
            f"prev_device_step_time = {device_text}, "
            f"timestamp = {timestamp}, "
            f"states = {values['states']}, "
            f"num_active = {values['numActiveRequests']}, "
            f"num_fetched = {num_fetched}, "
            f"num_dummy = {values['numDummyRequests']}, "
            f"num_unscheduled = {values['numUnscheduledRequests']}, "
            f"loop_sequence = {values['loopSequence']}, "
            f"batch_iter = {_format_count(snapshot.get('batchIter'))}, "
            f"kv_capture_iter = {_format_count(snapshot.get('captureIter'))}, "
            f"kv_capture_time = {capture_text}, "
            f"kv_snapshot_fresh = {str(snapshot_fresh).lower()}, "
            f"kv_movement_fresh = {str(movement_fresh).lower()}, "
            f"{diagnostic_text}"
        )
        self._previous_rank_snapshot = snapshot
