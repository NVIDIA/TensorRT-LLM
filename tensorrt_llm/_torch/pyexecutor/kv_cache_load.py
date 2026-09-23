# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Internal scheduler-owned routing snapshots; dictionaries cross worker RPCs."""

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
