# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Receiver-owned destinations and bounce regions for one transfer piece."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping

import numpy as np

from tensorrt_llm._torch.disaggregation.base import CacheKind, Chunk
from tensorrt_llm._torch.disaggregation.resource.page import KVCachePageTable
from tensorrt_llm._torch.disaggregation.resource.utils import (
    get_layer_byte_ranges,
    get_physical_pool,
)

_ADDRESS_LIMIT = 1 << 63


def _checked_interval(start: int, size: int) -> tuple[int, int]:
    end = start + size
    if start < 0 or size <= 0 or end > _ADDRESS_LIMIT:
        raise ValueError(f"invalid receive region start={start} size={size}")
    return start, end


def _destination_intervals(chunk: Chunk, page_table: KVCachePageTable) -> list[tuple[int, int]]:
    if len(chunk.block_ids_per_layer_groups) != len(page_table.layer_groups):
        raise ValueError("receive piece and page table have different layer-group counts")
    if len(chunk.kind_per_layer_group) != len(page_table.layer_groups):
        raise ValueError("receive piece and page table have different cache-kind counts")

    intervals: list[tuple[int, int]] = []
    for group_idx, (layer_group, block_ids) in enumerate(
        zip(page_table.layer_groups, chunk.block_ids_per_layer_groups)
    ):
        if chunk.kind_per_layer_group[group_idx] != layer_group.kind:
            raise ValueError(f"cache kind differs for layer group {group_idx}")
        if (
            not isinstance(block_ids, np.ndarray)
            or block_ids.dtype != np.int64
            or block_ids.ndim != 1
        ):
            raise ValueError(f"layer group {group_idx} must contain one-dimensional int64 ids")
        if np.any(block_ids < -1):
            raise ValueError(f"layer group {group_idx} has invalid negative ids")

        if layer_group.kind == CacheKind.PAGED:
            valid_ids = np.unique(block_ids[block_ids >= 0])
            for pool_idx in {view.pool_idx for view in layer_group.pool_views}:
                pool = get_physical_pool(page_table, group_idx, pool_idx)
                for block_id in valid_ids:
                    block = int(block_id)
                    if block >= pool.num_slots:
                        raise ValueError(f"block id {block} is outside layer group {group_idx}")
                    start = pool.base_address + block * pool.slot_stride_bytes
                    intervals.append(_checked_interval(start, pool.slot_bytes))
        elif layer_group.kind == CacheKind.STATE:
            if block_ids.size > 1:
                raise ValueError(f"STATE layer group {group_idx} has more than one slot id")
            if block_ids.size == 0 or block_ids[0] < 0:
                continue
            slot = int(block_ids[0])
            for view in layer_group.pool_views:
                pool = get_physical_pool(page_table, group_idx, view.pool_idx)
                if slot >= pool.num_slots:
                    raise ValueError(f"slot id {slot} is outside layer group {group_idx}")
                offsets, bytes_per_layer = get_layer_byte_ranges(view)
                for offset in offsets.values():
                    start = pool.base_address + slot * pool.slot_stride_bytes + offset
                    intervals.append(_checked_interval(start, bytes_per_layer))
        else:
            raise ValueError(f"unsupported cache kind for layer group {group_idx}")

    intervals.sort()
    # Merge only overlapping intervals. Intervals that merely touch (adjacent slots, or
    # adjacent per-layer state ranges) deliberately stay separate: every mapper emits a
    # fragment inside one slot, so a fragment that runs from one slot into the next is a
    # malformed result and must be rejected, not accepted because the neighbour happens to
    # belong to the same request.
    merged: list[tuple[int, int]] = []
    for start, end in intervals:
        if merged and start < merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


@dataclass(frozen=True)
class ReceiveBounds:
    """The destinations and per-writer source region advertised for one receive piece."""

    _starts: bytes = field(repr=False)
    _ends: bytes = field(repr=False)
    _writer_spans: Mapping[int, tuple[int, int] | None] = field(repr=False)

    @classmethod
    def from_chunk(
        cls,
        chunk: Chunk,
        page_table: KVCachePageTable | None,
        writer_spans: dict[int, tuple[int, int] | None],
    ) -> ReceiveBounds:
        """Snapshot the local receive regions before publishing them to writers."""
        spans: dict[int, tuple[int, int] | None] = {}
        for rank, span in writer_spans.items():
            if span is None:
                spans[rank] = None
                continue
            base, capacity = map(int, span)
            _checked_interval(base, capacity)
            spans[rank] = (base, capacity)

        if not any(span is not None for span in spans.values()):
            intervals: list[tuple[int, int]] = []
        else:
            if page_table is None:
                raise ValueError("a bounced receive requires a local page table")
            intervals = _destination_intervals(chunk, page_table)

        starts = np.array([start for start, _ in intervals], dtype=np.uint64).tobytes()
        ends = np.array([end for _, end in intervals], dtype=np.uint64).tobytes()
        return cls(starts, ends, MappingProxyType(spans))

    def has_writer(self, peer_rank: int) -> bool:
        """Whether ``peer_rank`` is one of the writers this piece was published to."""
        return peer_rank in self._writer_spans

    def validate_result(
        self,
        peer_rank: int,
        succeeded: bool,
        dst_ptrs: np.ndarray | None,
        sizes: np.ndarray | None,
        src_base: int | None,
    ) -> tuple[np.ndarray | None, np.ndarray | None, int | None]:
        """Return a complete result with its locally recorded source address."""
        if peer_rank not in self._writer_spans:
            raise ValueError(f"writer {peer_rank} is not assigned to this receive piece")

        tail = (dst_ptrs, sizes, src_base)
        if all(value is None for value in tail):
            return None, None, None
        if not succeeded or any(value is None for value in tail):
            raise ValueError("result has an incomplete or unexpected bounce tail")

        writer_span = self._writer_spans[peer_rank]
        if writer_span is None:
            raise ValueError("result has a bounce tail without a local receive region")
        expected_base, capacity = writer_span
        if not isinstance(src_base, (int, np.integer)) or int(src_base) != expected_base:
            raise ValueError(f"writer {peer_rank} reported a different bounce source")
        if (
            not isinstance(dst_ptrs, np.ndarray)
            or not isinstance(sizes, np.ndarray)
            or dst_ptrs.dtype != np.int64
            or sizes.dtype != np.int64
            or dst_ptrs.ndim != 1
            or sizes.ndim != 1
            or dst_ptrs.size == 0
            or dst_ptrs.size != sizes.size
        ):
            raise ValueError(
                "bounce fragment addresses and sizes must be equal-length int64 arrays"
            )
        if np.any(dst_ptrs < 0) or np.any(sizes <= 0):
            raise ValueError("bounce fragments require non-negative addresses and positive sizes")
        # Vectorized total check: this runs on the listener thread under the session lock and a
        # TP-mismatched result can carry hundreds of thousands of fragments. Every size is
        # positive and at most ``capacity`` (< 2**63), so a running uint64 sum exceeds
        # ``capacity`` strictly before it could wrap around; the prefix comparison is exact.
        if (
            dst_ptrs.size > capacity
            or np.any(sizes > capacity)
            or np.any(np.cumsum(sizes, dtype=np.uint64) > np.uint64(capacity))
        ):
            raise ValueError("bounce fragments exceed the writer's receive region")

        starts = np.frombuffer(self._starts, dtype=np.uint64)
        ends = np.frombuffer(self._ends, dtype=np.uint64)
        addresses = dst_ptrs.view(np.uint64)
        indices = np.searchsorted(starts, addresses, side="right") - 1
        if np.any(indices < 0):
            raise ValueError("bounce fragment lies outside this receive piece")
        limits = ends[indices]
        if np.any(addresses >= limits):
            raise ValueError("bounce fragment lies outside this receive piece")
        if np.any(sizes.view(np.uint64) > limits - addresses):
            raise ValueError(
                "bounce fragment starts inside a receive slot but runs past its end; "
                "a fragment may not span adjacent slots"
            )

        # The result arrays may outlive the decoded message while scatter runs on another thread.
        checked_ptrs = dst_ptrs.copy()
        checked_sizes = sizes.copy()
        checked_ptrs.setflags(write=False)
        checked_sizes.setflags(write=False)
        return checked_ptrs, checked_sizes, expected_base
