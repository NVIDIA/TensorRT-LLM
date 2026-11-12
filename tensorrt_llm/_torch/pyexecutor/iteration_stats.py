# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Named transport records for finalized executor iteration statistics."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, NamedTuple

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
    kv_transfer_outcomes: tuple[int, int, int] | None = None
    executor_loop: dict[str, object] | None = None

    @classmethod
    def from_tuple(cls, stats: tuple) -> IterationStatsRecord:
        """Normalize legacy tuples, ignoring trailing fields from newer producers."""
        if isinstance(stats, cls):
            return stats
        return cls(*stats[: len(cls._fields)])
