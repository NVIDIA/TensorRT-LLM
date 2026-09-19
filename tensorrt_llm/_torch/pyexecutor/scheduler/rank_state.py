# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Attention-DP rank-state transport types."""

from __future__ import annotations

from dataclasses import MISSING, dataclass, field, fields, replace


@dataclass
class RankIterStatsPayload:
    """Per-rank iteration payload piggybacked on the ADP allgather."""

    has_iter_stats: int = 0
    iter_stats_iter: int = -1
    num_context_requests: int = 0
    num_ctx_tokens: int = 0
    num_ctx_kv_tokens: int = 0
    num_gen_requests: int = 0
    num_gen_kv_tokens: int = 0
    num_paused_requests: int = 0
    num_paused_kv_tokens: int = 0
    kv_used_blocks: int = -1
    kv_total_blocks: int = -1
    kv_load_timestamp_ns: int = 0

    def serialize(self, include_kv_cache_load: bool = False) -> list[int]:
        """Serialize to a flat list for allgather transport."""
        values = [
            self.has_iter_stats,
            self.iter_stats_iter,
            self.num_context_requests,
            self.num_ctx_tokens,
            self.num_ctx_kv_tokens,
            self.num_gen_requests,
            self.num_gen_kv_tokens,
            self.num_paused_requests,
            self.num_paused_kv_tokens,
        ]
        if include_kv_cache_load:
            values.extend(
                [
                    self.kv_used_blocks,
                    self.kv_total_blocks,
                    self.kv_load_timestamp_ns,
                ]
            )
        return values

    @classmethod
    def deserialize(cls, data: list[int]) -> RankIterStatsPayload:
        """Deserialize a flat payload, filling omitted trailing defaults."""
        values = list(data)
        payload_fields = fields(cls)
        if len(values) > len(payload_fields):
            raise ValueError(
                f"RankIterStatsPayload has {len(values)} fields, expected at most "
                f"{len(payload_fields)}"
            )
        for field_info in payload_fields[len(values) :]:
            if field_info.default is MISSING and field_info.default_factory is MISSING:
                raise ValueError(
                    f"RankIterStatsPayload is missing required field {field_info.name}"
                )
        return cls(*values)


@dataclass
class RankState:
    """Per-rank state exchanged before attention-DP request assignment."""

    rank: int
    num_active_requests: int = 0
    num_active_tokens: int = 0
    iter_stats: RankIterStatsPayload = field(default_factory=RankIterStatsPayload)

    def copy_iter_stats_from(self, iter_stats_payload: RankIterStatsPayload | None) -> None:
        if iter_stats_payload is not None:
            self.iter_stats = replace(iter_stats_payload)

    def serialize(self, include_kv_cache_load: bool = False) -> list[int]:
        """Serialize to a flat list for allgather transport."""
        return [
            self.rank,
            self.num_active_requests,
            self.num_active_tokens,
            *self.iter_stats.serialize(include_kv_cache_load),
        ]

    @classmethod
    def deserialize(cls, data: list[int]) -> RankState:
        """Deserialize from a flat list received via allgather."""
        values = list(data)
        prefix_field_count = 3
        rank_state_fields = fields(cls)[:prefix_field_count]
        max_field_count = prefix_field_count + len(fields(RankIterStatsPayload))
        if not values:
            raise ValueError("RankState payload is missing required field rank")
        if len(values) > max_field_count:
            raise ValueError(
                f"RankState payload has {len(values)} fields, expected at most {max_field_count}"
            )
        rank_values = values[:prefix_field_count]
        for field_info in rank_state_fields[len(rank_values) :]:
            if field_info.default is not MISSING:
                rank_values.append(field_info.default)
            elif field_info.default_factory is not MISSING:
                rank_values.append(field_info.default_factory())
            else:
                raise ValueError(f"RankState payload is missing required field {field_info.name}")
        return cls(
            rank=rank_values[0],
            num_active_requests=rank_values[1],
            num_active_tokens=rank_values[2],
            iter_stats=RankIterStatsPayload.deserialize(values[prefix_field_count:]),
        )
