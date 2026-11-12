# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Executor-loop logger exposure for already collected performance records."""

import json
from collections.abc import Callable
from datetime import datetime

from .enums import MetricNames
from .perf import PerfMetricsRecord, kv_cache_utilization


def _json_keys(value: object) -> object:
    if isinstance(value, dict):
        return {
            (key.value if isinstance(key, MetricNames) else key): _json_keys(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_json_keys(item) for item in value]
    return value


class LoggerExposure:
    """Format collected observations without accessing the runtime or CUDA."""

    def __init__(self, write: Callable[[str], None]) -> None:
        self._write = write
        self._last_rank_snapshot: object = None

    def emit(self, record: PerfMetricsRecord) -> None:
        """Expose a loop in the existing text format, with rank detail on change."""
        if record.kind == "rank":
            if record.per_rank is self._last_rank_snapshot:
                return
            self._last_rank_snapshot = record.per_rank
        if record.kind != "loop":
            self._write(json.dumps(_json_keys(record.categories()), separators=(",", ":")))
            return
        values = record.per_iter
        device = values.get("prevDeviceStepTimeMS")
        device_text = "N/A" if device is None else f"{device}ms"
        utilization = kv_cache_utilization(record.per_rank.get("kvCacheStats", {}))
        kv_text = "N/A" if utilization is None else f"{utilization:.3f}"
        timestamp = datetime.fromtimestamp(values["timestamp"]).strftime("%Y-%m-%d %H:%M:%S")
        self._write(
            f"iter = {values['iter']}, "
            f"global_rank = {values['globalRank']}, "
            f"rank = {values['rank']}, "
            f"num_scheduled_requests = {values['numScheduledRequests']}, "
            f"adp_dummy_ctx_tokens = {values['adpDummyCtxTokens']}, "
            f"adp_dummy_gen_tokens = {values['adpDummyGenTokens']}, "
            f"kv_cache_util = {kv_text}, "
            f"currank_total_requests = {values['numFetchedRequestsTotal']}/{values['numFetchedRequestsAllRanks']}, "
            f"host_step_time = {values['hostStepTimeMS']}ms, "
            f"prev_device_step_time = {device_text}, "
            f"timestamp = {timestamp}, "
            f"states = {values['states']}, "
            f"num_active = {values['numActiveRequests']}, "
            f"num_fetched = {values['numFetchedRequests']}, "
            f"num_dummy = {values['numDummyRequests']}, "
            f"num_unscheduled = {values['numUnscheduledRequests']}"
        )
        # Detailed snapshots refresh at their collection cadence, not on every log line.
        if record.per_rank:
            self.emit(PerfMetricsRecord("rank", per_rank=record.per_rank))
