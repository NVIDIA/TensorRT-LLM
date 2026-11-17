# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Executor-loop logger exposure for already collected performance records."""

from collections.abc import Callable
from datetime import datetime

from .perf import PerfMetricsRecord, kv_cache_utilization


class LoggerExposure:
    """Format collected observations without accessing the runtime or CUDA."""

    def __init__(self, write: Callable[[str], None]) -> None:
        self._write = write
        self.reset()

    def reset(self) -> None:
        """Start counting fetched requests for a new profiling context."""
        self._previous_fetched = 0

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
