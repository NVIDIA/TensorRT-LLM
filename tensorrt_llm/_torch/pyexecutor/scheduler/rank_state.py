# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Attention-DP rank-state transport types."""

from __future__ import annotations

from dataclasses import MISSING, dataclass, field, fields, replace
from typing import ClassVar, NamedTuple


class RankKvCacheStats(NamedTuple):
    """Immutable rank-local KV snapshot containing only transportable scalars."""

    max_num_blocks: int
    free_num_blocks: int
    used_num_blocks: int
    tokens_per_block: int
    alloc_total_blocks: int
    alloc_new_blocks: int
    reused_blocks: int
    missed_blocks: int
    cache_hit_rate: float

    @classmethod
    def wire_size(cls) -> int:
        """Return the number of scalar fields in a KV snapshot.

        Returns:
            Number of fields in the named tuple's wire schema.
        """
        return len(cls._fields)

    @classmethod
    def deserialize(cls, data: list[int | float]) -> RankKvCacheStats:
        """Restore a KV snapshot from its scalar wire fields.

        Args:
            data: Eight block counters followed by the floating-point hit rate.

        Returns:
            The immutable KV snapshot.

        Raises:
            ValueError: The snapshot has an unexpected number of fields.
        """
        if len(data) != cls.wire_size():
            raise ValueError(f"KV snapshot requires {cls.wire_size()} fields")
        return cls(
            int(data[0]),
            int(data[1]),
            int(data[2]),
            int(data[3]),
            int(data[4]),
            int(data[5]),
            int(data[6]),
            int(data[7]),
            float(data[8]),
        )


@dataclass
class RankIterStatsPayload:
    """Per-rank iteration payload piggybacked on the ADP allgather."""

    # Legacy scalar widths are independent of the optional snapshot fields.
    _ITER_STATS_FIELD_COUNT: ClassVar[int] = 9
    _KV_LOAD_FIELD_COUNT: ClassVar[int] = 3
    _SNAPSHOT_WIRE_VERSION: ClassVar[int] = -1
    _SNAPSHOT_HEADER_SIZE: ClassVar[int] = 4

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
    kv_cache_stats: RankKvCacheStats | None = None
    cross_kv_cache_stats: RankKvCacheStats | None = None

    def serialize(self, include_kv_cache_load: bool = False) -> list[int | float]:
        """Serialize to a flat list for allgather transport."""
        values: list[int | float] = [
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
        if self.kv_cache_stats is not None or self.cross_kv_cache_stats is not None:
            # The negative version cannot be confused with has_iter_stats (0/1).
            # Presence flags frame both snapshots and the independent load fields.
            values = [
                self._SNAPSHOT_WIRE_VERSION,
                int(include_kv_cache_load),
                int(self.kv_cache_stats is not None),
                int(self.cross_kv_cache_stats is not None),
                *values,
            ]
            for snapshot in (self.kv_cache_stats, self.cross_kv_cache_stats):
                if snapshot is not None:
                    values.extend(snapshot)
        return values

    @classmethod
    def snapshot_payload_size(cls, data: list[int | float]) -> int:
        """Read the size of a versioned snapshot payload from its header.

        Args:
            data: Payload header followed by scalar fields and optional rank-state tail.

        Returns:
            Number of fields belonging to the iteration payload.

        Raises:
            ValueError: The header is incomplete or contains an unsupported version or flag.
        """
        if (
            len(data) < cls._SNAPSHOT_HEADER_SIZE
            or data[0] != cls._SNAPSHOT_WIRE_VERSION
            or any(flag not in (0, 1) for flag in data[1 : cls._SNAPSHOT_HEADER_SIZE])
        ):
            raise ValueError("Invalid KV snapshot payload header")
        return (
            cls._SNAPSHOT_HEADER_SIZE
            + cls._ITER_STATS_FIELD_COUNT
            + int(data[1]) * cls._KV_LOAD_FIELD_COUNT
            + int(data[2] + data[3]) * RankKvCacheStats.wire_size()
        )

    @classmethod
    def deserialize(cls, data: list[int | float]) -> RankIterStatsPayload:
        """Deserialize a flat payload, filling omitted trailing defaults."""
        values = list(data)
        if values and values[0] < 0:
            expected_size = cls.snapshot_payload_size(values)
            if len(values) != expected_size:
                raise ValueError(f"KV snapshot payload requires {expected_size} fields")
            scalar_end = (
                cls._SNAPSHOT_HEADER_SIZE
                + cls._ITER_STATS_FIELD_COUNT
                + int(values[1]) * cls._KV_LOAD_FIELD_COUNT
            )
            result = cls(*values[cls._SNAPSHOT_HEADER_SIZE : scalar_end])
            cursor = scalar_end
            if values[2]:
                result.kv_cache_stats = RankKvCacheStats.deserialize(
                    values[cursor : cursor + RankKvCacheStats.wire_size()]
                )
                cursor += RankKvCacheStats.wire_size()
            if values[3]:
                result.cross_kv_cache_stats = RankKvCacheStats.deserialize(
                    values[cursor : cursor + RankKvCacheStats.wire_size()]
                )
            return result
        scalar_field_count = cls._ITER_STATS_FIELD_COUNT + cls._KV_LOAD_FIELD_COUNT
        if len(values) > scalar_field_count:
            raise ValueError(
                f"RankIterStatsPayload has {len(values)} fields, expected at most "
                f"{scalar_field_count}"
            )
        return cls(*values)


@dataclass
class RankState:
    """Per-rank state exchanged before attention-DP request assignment."""

    rank: int
    num_active_requests: int = 0
    num_active_tokens: int = 0
    num_retiring_requests: int = 0
    iter_stats: RankIterStatsPayload = field(default_factory=RankIterStatsPayload)
    num_context_requests: int = 0
    num_generation_requests: int = 0
    num_connector_transfers: int = 0
    connector_poll_failed: bool = False

    def copy_iter_stats_from(self, iter_stats_payload: RankIterStatsPayload | None) -> None:
        if iter_stats_payload is not None:
            self.iter_stats = replace(iter_stats_payload)

    def serialize(self, include_kv_cache_load: bool = False) -> list[int | float]:
        """Serialize to a flat list for allgather transport."""
        return [
            self.rank,
            self.num_active_requests,
            self.num_active_tokens,
            self.num_retiring_requests,
            *self.iter_stats.serialize(include_kv_cache_load),
            self.num_context_requests,
            self.num_generation_requests,
            self.num_connector_transfers,
            int(self.connector_poll_failed),
        ]

    @classmethod
    def deserialize(cls, data: list[int | float]) -> RankState:
        """Deserialize from a flat list received via allgather."""
        values = list(data)
        prefix_field_count = 4
        rank_state_fields = fields(cls)[:prefix_field_count]
        if not values:
            raise ValueError("RankState payload is missing required field rank")
        if len(values) > prefix_field_count and values[prefix_field_count] < 0:
            stats_field_count = RankIterStatsPayload.snapshot_payload_size(
                values[prefix_field_count:]
            )
            if len(values) != prefix_field_count + stats_field_count + 4:
                raise ValueError("KV snapshot rank state requires four trailing fields")
        else:
            # Legacy messages have optional load telemetry and two or four tail
            # fields. Keep their framing independent of the Python dataclass size.
            telemetry_field_count = (
                RankIterStatsPayload._ITER_STATS_FIELD_COUNT
                + RankIterStatsPayload._KV_LOAD_FIELD_COUNT
            )
            telemetry_payload_size = prefix_field_count + telemetry_field_count + 2
            stats_field_count = (
                telemetry_field_count
                if len(values) >= telemetry_payload_size
                else RankIterStatsPayload._ITER_STATS_FIELD_COUNT
            )
        max_field_count = prefix_field_count + stats_field_count + 4
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
        stats_end = prefix_field_count + stats_field_count
        return cls(
            rank=rank_values[0],
            num_active_requests=rank_values[1],
            num_active_tokens=rank_values[2],
            num_retiring_requests=rank_values[3],
            iter_stats=RankIterStatsPayload.deserialize(values[prefix_field_count:stats_end]),
            num_context_requests=values[stats_end] if len(values) > stats_end else 0,
            num_generation_requests=values[stats_end + 1] if len(values) > stats_end + 1 else 0,
            num_connector_transfers=values[stats_end + 2] if len(values) > stats_end + 2 else 0,
            connector_poll_failed=bool(values[stats_end + 3])
            if len(values) > stats_end + 3
            else False,
        )
