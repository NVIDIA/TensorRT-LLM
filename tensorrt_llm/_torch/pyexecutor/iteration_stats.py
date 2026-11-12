# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Named transport records for finalized executor iteration statistics."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Literal, NamedTuple

from .kv_cache_stats import append_kv_cache_iteration_stats

if TYPE_CHECKING:
    from tensorrt_llm.bindings.executor import IterationStats, RequestStats

    from .kv_cache_stats import KVCacheV2IterationStatsReport


class IterationStatsRecord(NamedTuple):
    """Tuple-compatible record buffered by the executor and read by the worker.

    Append optional fields at the end to preserve historical tuple layouts.
    The record owns references to its observations; it does not copy native stats.
    """

    iteration_stats: IterationStats
    request_stats: list[RequestStats] | None = None
    kv_iteration_stats: dict[int, object] | KVCacheV2IterationStatsReport | None = None
    attention_dp_rank: int | None = None
    host_step_time_ms: float | None = None
    prev_device_step_time_ms: float | None = None
    scheduler_mode: Literal["overlap", "non_overlap"] | None = None
    gpu_forward_time_ms: float | None = None
    executor_loop: dict[str, object] | None = None
    kv_transfer_outcomes: tuple[int, int, int] | None = None

    @classmethod
    def from_tuple(cls, stats: tuple) -> IterationStatsRecord:
        """Normalize legacy tuples, ignoring trailing fields from newer producers."""
        if isinstance(stats, cls):
            return stats
        return cls(*stats[: len(cls._fields)])

    def to_dict(self, *, tp_rank: int | None = None) -> dict:
        """Serialize observations using the executor's existing JSON field names.

        Gathered rank rows carry ``rank``; worker rows carry ``attentionDpRank``
        (defaulting to zero for consumers that require it).
        """
        stats = json.loads(self.iteration_stats.to_json_str())
        if tp_rank is None:
            stats["attentionDpRank"] = (
                0 if self.attention_dp_rank is None else self.attention_dp_rank
            )
        else:
            stats["rank"] = tp_rank
        if self.request_stats:
            stats["requestStats"] = [json.loads(req.to_json_str()) for req in self.request_stats]
        append_kv_cache_iteration_stats(stats, self.kv_iteration_stats)

        # Host time covers one loop; the ping-pong device time lags it by one
        # loop. GPU forward time belongs to this batch. Scheduler mode tells
        # consumers whether iterLatencyMS includes overlapping loops.
        optional_fields = (
            ("hostStepTimeMS", self.host_step_time_ms),
            ("prevDeviceStepTimeMS", self.prev_device_step_time_ms),
            ("gpuForwardTimeMS", self.gpu_forward_time_ms),
            ("schedulerMode", self.scheduler_mode),
            ("executorLoop", self.executor_loop),
        )
        for name, value in optional_fields:
            if value is not None:
                stats[name] = value
        if self.kv_transfer_outcomes is not None:
            stats["kvTransferOutcomes"] = dict(
                zip(("success", "error", "timeout"), self.kv_transfer_outcomes)
            )
        return stats
