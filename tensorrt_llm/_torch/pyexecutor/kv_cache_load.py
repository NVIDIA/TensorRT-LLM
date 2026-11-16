# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Internal scheduler-owned routing snapshots; dictionaries cross worker RPCs."""

import time
from typing import TypedDict


class RankKvLoad(TypedDict):
    rank: int
    usedKvBlocks: int
    totalKvBlocks: int
    timestampUnixNanos: int
    runningRequests: int


class KvLoadSnapshot(TypedDict):
    timestampUnixNanos: int
    usedKvBlocks: int
    totalKvBlocks: int
    ranks: list[RankKvLoad]


def aggregate_load(ranks: list[RankKvLoad]) -> KvLoadSnapshot:
    """Aggregate actual rank counters without guessing remote capacities."""
    return {
        "timestampUnixNanos": min(rank["timestampUnixNanos"] for rank in ranks),
        "usedKvBlocks": sum(rank["usedKvBlocks"] for rank in ranks),
        "totalKvBlocks": sum(rank["totalKvBlocks"] for rank in ranks),
        "ranks": ranks,
    }


class KvLoadTracker:
    """Own scheduler-thread sampling and publish complete load snapshots to RPC readers."""

    def __init__(self, manager, rank: int, enabled: bool) -> None:
        self.manager = manager
        self.rank = rank
        self.enabled = enabled
        self.sample_interval_seconds = 0.1
        self.last_sample_monotonic = 0.0
        self.local_sample: RankKvLoad | dict = {}
        self.snapshot: KvLoadSnapshot | dict = {}
        self.rank_timestamps: tuple[tuple[int, int], ...] = ()
        self.idle_sampled = False

    def sample(self, capacity: dict, force: bool = False) -> RankKvLoad | dict:
        """Read CPU-side primary pool counters at most once per sampling interval."""
        now = time.monotonic()
        if (
            not force
            and self.local_sample
            and now - self.last_sample_monotonic < self.sample_interval_seconds
        ):
            return self.local_sample
        manager = self.manager
        if manager is None:
            return {}
        block_counts = getattr(type(manager), "get_primary_block_counts", None)
        if callable(block_counts):
            used, total = block_counts(manager)
        else:
            stats = manager.get_kv_cache_stats()
            total = int(
                getattr(stats, "max_num_blocks", 0)
                or getattr(stats, "primary_max_num_blocks", 0)
                or 0
            )
            used = int(
                getattr(stats, "used_num_blocks", 0)
                or getattr(stats, "primary_used_num_blocks", 0)
                or 0
            )
        if total <= 0:
            total = int(capacity.get("maxNumBlocks", 0))
        if total <= 0 or used < 0 or used > total:
            return {}
        sample: RankKvLoad = {
            "rank": self.rank,
            "usedKvBlocks": used,
            "totalKvBlocks": total,
            "timestampUnixNanos": time.time_ns(),
            "runningRequests": 0,
        }
        self.local_sample = sample
        self.last_sample_monotonic = now
        return sample

    def begin_step(
        self, active_requests: list, attention_dp: bool, capacity: dict
    ) -> RankKvLoad | dict:
        """Refresh at idle entry and return the sample to piggyback on ADP state."""
        if not self.enabled:
            return {}
        force = not active_requests and not self.idle_sampled
        self.idle_sampled = not active_requests
        sample = self.sample(capacity, force=force)
        if not attention_dp:
            self.record_local(sample)
        return sample

    def record_local(self, sample: RankKvLoad | dict) -> None:
        if sample and self.rank_timestamps != ((sample["timestampUnixNanos"], 0),):
            self.rank_timestamps = ((sample["timestampUnixNanos"], 0),)
            self.snapshot = aggregate_load([sample])

    def record_rank_states(self, states) -> None:
        """Publish a new ADP snapshot when blocks or active counts change."""
        timestamps = tuple(
            (state.iter_stats.kv_load_timestamp_ns, state.num_active_requests) for state in states
        )
        if not timestamps or not all(timestamp > 0 for timestamp, _ in timestamps):
            return
        if timestamps == self.rank_timestamps:
            return
        ranks: list[RankKvLoad] = [
            {
                "rank": state.rank,
                "runningRequests": state.num_active_requests,
                "usedKvBlocks": state.iter_stats.kv_used_blocks,
                "totalKvBlocks": state.iter_stats.kv_total_blocks,
                "timestampUnixNanos": state.iter_stats.kv_load_timestamp_ns,
            }
            for state in states
        ]
        self.rank_timestamps = timestamps
        self.snapshot = aggregate_load(ranks)
