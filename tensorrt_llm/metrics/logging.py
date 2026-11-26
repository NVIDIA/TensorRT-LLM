# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Executor-loop logger exposure for already collected performance records."""

from collections.abc import Callable, Mapping, Sequence
from datetime import datetime

from .perf import PerfMetricsRecord, kv_cache_utilization


def _format_pool_list(values: Sequence[float | None]) -> str:
    items = (
        f"{pool_id}:{'N/A' if value is None else f'{value:.3f}'}"
        for pool_id, value in enumerate(values)
    )
    return f"[{';'.join(items)}]"


def format_kv_pool_fields(
    pool_utilization: Mapping[str, Sequence[float | None] | None],
    gpu_pool_gib: Sequence[float] | None = None,
) -> str:
    """Format per-pool KV cache fields for the iteration log.

    Returns ``kv_cache_gpu_pool_gib = [id:GiB;...], `` when ``gpu_pool_gib`` is given, then
    ``kv_cache_{tier}_pool_util = [id:util;...], `` per tier (``N/A`` for a missing tier).
    Ids are tier-local pool group ids. The text ends with ``", "`` so it can be spliced in
    before the next field.
    """
    fields = []
    if gpu_pool_gib is not None:
        fields.append(f"kv_cache_gpu_pool_gib = {_format_pool_list(gpu_pool_gib)}")
    for tier, values in pool_utilization.items():
        text = "N/A" if values is None else _format_pool_list(values)
        fields.append(f"kv_cache_{tier}_pool_util = {text}")
    return "".join(f"{field}, " for field in fields)


class LoggerExposure:
    """Format collected observations without accessing the runtime or CUDA."""

    def __init__(self, write: Callable[[str], None]) -> None:
        self._write = write
        self.reset()

    def reset(self) -> None:
        """Start counting fetched requests for a new profiling context."""
        self._previous_fetched = 0

    def emit(self, record: PerfMetricsRecord, kv_pool_text: str = "") -> None:
        """Expose one collected loop in the existing single-line text format.

        ``kv_pool_text`` (from ``format_kv_pool_fields``) is spliced in after the KV cache
        fields. It is kept out of the record so it never reaches iteration stats.
        """
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
        kv_stats = record.per_rank.get("kvCacheStats", {})
        utilization = kv_cache_utilization(kv_stats)
        kv_text = "N/A" if utilization is None else f"{utilization:.3f}"
        kv_reuse_text = ""
        if "reusedBlocks" in kv_stats:
            kv_reuse_text = (
                f"kv_reused_blocks = {kv_stats['reusedBlocks']}, "
                f"kv_missed_blocks = {kv_stats['missedBlocks']}, "
                f"kv_hit_rate = {kv_stats['cacheHitRate']:.3f}, "
            )
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
            f"{kv_pool_text}"
            f"currank_total_requests = {values['numFetchedRequestsTotal']}/{values['numFetchedRequestsAllRanks']}, "
            f"host_step_time = {values['hostStepTimeMS']}ms, "
            f"prev_device_step_time = {device_text}, "
            f"timestamp = {timestamp}, "
            f"states = {values['states']}, "
            f"num_active = {values['numActiveRequests']}, "
            f"num_fetched = {num_fetched}, "
            f"num_dummy = {values['numDummyRequests']}, "
            f"num_unscheduled = {values['numUnscheduledRequests']}"
        )
