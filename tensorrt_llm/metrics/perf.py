# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Backend-independent collection records shared by metric exposures.

Collection owns source snapshots. Exposures receive immutable values and cannot
read runtime managers or consume reset-on-read statistics.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Literal, Protocol, TypeAlias

from .enums import MetricNames

FINISH_REASON_LABEL = "finished_reason"
MetricKey: TypeAlias = str | MetricNames
MetricScalar: TypeAlias = str | int | float | bool | None
MetricValue: TypeAlias = (
    MetricScalar | Mapping[MetricKey, "MetricValue"] | tuple["MetricValue", ...]
)
PoolRows: TypeAlias = dict[int, Mapping[MetricKey, MetricValue]]
KV_MOVEMENT_WINDOW_FIELDS = (
    ("offload", "kv_offload_blocks", "iterOffloadBlocks"),
    ("onboard", "kv_onboard_blocks", "iterOnboardBlocks"),
    ("host_drop", "kv_host_dropped_blocks", "iterHostDroppedBlocks"),
)

# Unrecognized fields remain per-iteration so old/new producers can interoperate.
_RANK_FIELDS = frozenset(
    {
        "kvCacheStats",
        "kvTransferOutcomes",
        "kvCacheNumConversations",
        "conversationActivity",
        "crossKvCacheStats",
        "kvCacheIterationStats",
        "kvCacheIterationStatsByLifecycle",
        "kvCacheIterationStatsByPoolGroup",
        "kvCacheIterationStatsByColdPoolGroup",
        "kvCacheStorageStats",
        "kvCacheLevelTiers",
        "iterSuspendedRequests",
        "iterResumedRequests",
        "iterDiskPrefetchBlocks",
        "iterCachedTokensByLevel",
        "iterConnectorCachedTokens",
        "gpuMemUsage",
        "cpuMemUsage",
        "pinnedMemUsage",
        "maxNumActiveRequests",
        "maxBatchSizeStatic",
        "maxBatchSizeRuntime",
        "maxNumTokensRuntime",
    }
)


def _freeze(value: object) -> MetricValue:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"Unsupported metric value type: {type(value).__name__}")


def _freeze_fields(values: Mapping) -> Mapping[MetricKey, MetricValue]:
    return MappingProxyType({key: _freeze(value) for key, value in values.items()})


def _thaw(value: MetricValue) -> object:
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


def block_count(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


def _pool_id(value: object) -> int | None:
    if isinstance(value, str) and value.isdecimal():
        return int(value)
    return block_count(value)


def kv_pool_rows(
    snapshot: Mapping[MetricKey, MetricValue],
) -> tuple[PoolRows | None, PoolRows | None]:
    storage = snapshot.get("kvCacheStorageStats")
    hot_pools: PoolRows | None = None
    if isinstance(storage, (tuple, list)):
        hot_pools = {}
        for row in storage:
            if not isinstance(row, Mapping):
                hot_pools = None
                break
            level = _pool_id(row.get("cacheLevel"))
            if level is None:
                hot_pools = None
                break
            if level != 0:
                continue
            group = _pool_id(row.get("poolGroupId"))
            if group is None or group in hot_pools:
                hot_pools = None
                break
            hot_pools[group] = row

    report = snapshot.get("kvCacheIterationStatsByPoolGroup")
    movements: PoolRows | None = None
    if isinstance(report, Mapping):
        movements = {}
        for group_id, row in report.items():
            group = _pool_id(group_id)
            if group is None or group in movements or not isinstance(row, Mapping):
                movements = None
                break
            movements[group] = row
    return hot_pools, movements


def pool_counts(rows: PoolRows | None, groups: Iterable[int], field: str) -> dict[int, int | None]:
    return {
        group: block_count(rows.get(group, {}).get(field)) if rows is not None else None
        for group in groups
    }


def total_pool_count(counts: Mapping[int, int | None]) -> int | None:
    if any(value is None for value in counts.values()):
        return None
    return sum(value for value in counts.values() if value is not None)


def kv_cache_utilization(stats: Mapping) -> float | None:
    """Return non-reclaimable utilization, preferring an explicit used count.

    Basic snapshots can supply only capacity and free blocks. Missing or
    non-positive capacity is unavailable, rather than measured zero utilization.
    """
    capacity = stats.get("maxNumBlocks", 0)
    if not isinstance(capacity, (int, float)) or capacity <= 0:
        return None
    used = stats.get("usedNumBlocks")
    if used is None:
        free = stats.get("freeNumBlocks")
        if free is None:
            return None
        used = capacity - free
    return used / capacity


def _empty_fields() -> Mapping[MetricKey, MetricValue]:
    return MappingProxyType({})


@dataclass(frozen=True, slots=True)
class PerfMetricsRecord:
    """One owned observation, partitioned by measurement scope.

    The kind distinguishes request completions, loop-boundary observations and
    completed batch records. A batch's ``requestStats`` is diagnostic data, not a
    second stream of request completions. Existing wire field names retain their
    units (for example ``hostStepTimeMS`` is milliseconds).
    """

    kind: Literal["request", "loop", "iteration", "rank"]
    per_req: Mapping[MetricKey, MetricValue] = field(default_factory=_empty_fields)
    per_iter: Mapping[MetricKey, MetricValue] = field(default_factory=_empty_fields)
    per_rank: Mapping[MetricKey, MetricValue] = field(default_factory=_empty_fields)

    def __post_init__(self) -> None:
        # Collection factories already own these proxies. Also protect direct
        # construction from mutable dict/list inputs without copying cached snapshots.
        if not isinstance(self.per_req, MappingProxyType):
            object.__setattr__(self, "per_req", _freeze_fields(self.per_req))
        if not isinstance(self.per_iter, MappingProxyType):
            object.__setattr__(self, "per_iter", _freeze_fields(self.per_iter))
        if not isinstance(self.per_rank, MappingProxyType):
            object.__setattr__(self, "per_rank", _freeze_fields(self.per_rank))

    def iteration_view(self) -> Mapping[MetricKey, MetricValue]:
        """Expose a read-only flat view without copying nested snapshots."""
        return MappingProxyType({**self.per_req, **self.per_iter, **self.per_rank})

    def iteration_json(self) -> str:
        """Encode immutable snapshots directly into the legacy wire shape."""

        def encode_mapping(value: object) -> dict:
            if isinstance(value, Mapping):
                return dict(value)
            raise TypeError(f"Unsupported metric JSON type: {type(value).__name__}")

        return json.dumps(self.iteration_view(), default=encode_mapping)

    def iteration_stats(self) -> dict:
        """Materialize a fresh legacy-compatible transport dictionary."""
        return {
            key: _thaw(value)
            for group in (self.per_req, self.per_iter, self.per_rank)
            for key, value in group.items()
        }

    def request_metrics(self) -> dict:
        """Materialize candidate completion metrics without changing enum keys."""
        return {key: _thaw(value) for key, value in self.per_req.items()}

    def categories(self) -> dict:
        """Materialize categories for structured logging and loop transport."""
        return {
            "kind": self.kind,
            "per_req": _thaw(self.per_req),
            "per_iter": _thaw(self.per_iter),
            "per_rank": _thaw(self.per_rank),
        }


class KvCacheSnapshot(Protocol):
    max_num_blocks: int
    free_num_blocks: int
    reused_blocks: int
    missed_blocks: int
    cache_hit_rate: float


class PerfMetricsCollector:
    """Collect observations before selecting logging or Prometheus exposure.

    An executor owns one instance on its loop thread. Stateless adapters also use
    the collection methods to normalize legacy dictionaries. No Prometheus, torch,
    CUDA, MPI or logging dependency is needed here.
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        """Begin a resource phase without invalidating exposure references."""
        self.latest_loop: PerfMetricsRecord | None = None
        self.latest_rank: Mapping[MetricKey, MetricValue] = _empty_fields()
        self._previous_fetched = 0
        self._loop_sequence = 0
        self._last_rank_poll: int | None = None

    @staticmethod
    def collect_request(metrics: Mapping) -> PerfMetricsRecord:
        """Take an owned snapshot of one completed output candidate."""
        return PerfMetricsRecord("request", per_req=_freeze_fields(metrics))

    @staticmethod
    def collect_iteration(stats: Mapping) -> PerfMetricsRecord:
        """Partition a source batch record without reading runtime state."""
        request = {"requestStats": stats["requestStats"]} if "requestStats" in stats else {}
        rank = {key: value for key, value in stats.items() if key in _RANK_FIELDS}
        iteration = {
            key: value
            for key, value in stats.items()
            if key not in _RANK_FIELDS and key != "requestStats"
        }
        return PerfMetricsRecord(
            "iteration", _freeze_fields(request), _freeze_fields(iteration), _freeze_fields(rank)
        )

    def collect_rank(self, stats: Mapping, *, capture_iter: int) -> PerfMetricsRecord:
        """Remember a source-owned rank snapshot for subsequent loop exposure."""
        self.latest_rank = _freeze_fields(
            {**stats, "captureIter": capture_iter, "captureTime": time.time()}
        )
        return PerfMetricsRecord("rank", per_rank=self.latest_rank)

    def collect_loop(
        self,
        values: Mapping,
        *,
        read_kv_cache: Callable[[], KvCacheSnapshot] | None = None,
        kv_refresh_interval: int = 16,
    ) -> PerfMetricsRecord:
        """Capture one loop; poll basic KV only for a logger-only executor.

        With batch collection enabled the caller supplies no reader: logging
        reuses ``collect_rank`` snapshots instead of querying the manager again.
        """
        iteration = dict(values)
        sequence = self._loop_sequence
        self._loop_sequence += 1
        iteration["loopSequence"] = sequence
        fetched = iteration["numFetchedRequestsTotal"]
        iteration["numFetchedRequests"] = max(0, fetched - self._previous_fetched)
        self._previous_fetched = fetched
        iteration.setdefault("timestamp", time.time())
        loop_id = iteration["iter"]
        if read_kv_cache is not None and (
            self._last_rank_poll is None or sequence - self._last_rank_poll >= kv_refresh_interval
        ):
            stats = read_kv_cache()
            self.collect_rank(
                {
                    "rank": iteration["rank"],
                    "globalRank": iteration["globalRank"],
                    "captureLoopSequence": sequence,
                    "kvCacheStats": {
                        "maxNumBlocks": stats.max_num_blocks,
                        "freeNumBlocks": stats.free_num_blocks,
                        "usedNumBlocks": stats.max_num_blocks - stats.free_num_blocks,
                        "reusedBlocks": stats.reused_blocks,
                        "missedBlocks": stats.missed_blocks,
                        "cacheHitRate": stats.cache_hit_rate,
                    },
                },
                capture_iter=loop_id,
            )
            self._last_rank_poll = sequence
        record = PerfMetricsRecord(
            "loop", per_iter=_freeze_fields(iteration), per_rank=self.latest_rank
        )
        self.latest_loop = record
        return record
