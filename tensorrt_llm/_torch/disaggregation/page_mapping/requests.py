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
"""Questions about a rank's whole pages, and the per-request rules the cache manager answers.

Memory to register, bounce sizing and the bytes a chunk moves follow from the pages; how much of a
request's history moves, which buffers it leaves out and whether first tokens travel are the cache
manager's (``KVCacheManagerV2.get_transfer_length`` and siblings).
"""

from __future__ import annotations

from typing import Dict, FrozenSet, List, Optional, Sequence

import numpy as np

from . import units
from .rank_pages import BufferRef, RankPages


def memory_registration_descs(pages: RankPages, device_id: int) -> List[tuple]:
    """Deduplicated ``(ptr, size, device_id, name)`` of every allocation backing the cache."""
    unique: Dict[tuple, int] = {}
    for g in range(pages.num_layer_groups):
        for region in units.registration_regions(pages, g):
            if region.within_known and any(
                base <= region.base_address < base + size for base, size in unique
            ):
                continue
            unique.setdefault((region.base_address, region.size), len(unique))
    return [
        (ptr, size, device_id, f"kv_cache_memory_pool{idx}") for (ptr, size), idx in unique.items()
    ]


def has_replicated_buffers(pages: Optional[RankPages]) -> bool:
    """Whether any buffer is moved by one elected sender rather than by every rank."""
    if pages is None:
        return False
    return any(
        all(seg.split_axis is None for seg in buffer.segments)
        for g in range(pages.num_layer_groups)
        for buffer in pages.buffers(g)
    )


def bounce_unit_bytes_per_layer_group(pages: RankPages) -> List[Optional[int]]:
    """Bytes per unit the bounce buffer reserves for each layer group (None: sized separately)."""
    return [units.bounce_unit_bytes(pages, g) for g in range(pages.num_layer_groups)]


def receiver_extra_bytes(
    sender_pages: Optional[RankPages],
    receiver_pages: RankPages,
    units_per_layer_group: Sequence[np.ndarray],
) -> Optional[int]:
    """Bytes outside the per-ordinal bounce sizing landing in the receiver's units.

    ``None`` when some layer group needs the sender's pages to size it and none are known.
    """
    total = 0
    for g in range(receiver_pages.num_layer_groups):
        group_units = units_per_layer_group[g]
        if group_units is None or group_units.size == 0:
            continue
        if units.bounce_unit_bytes(receiver_pages, g) is not None:
            continue
        if sender_pages is None:
            return None
        total += units.receiver_extra_bytes(sender_pages, receiver_pages, g, group_units)
    return total


def chunk_transfer_bytes(
    pages: Optional[RankPages],
    units_per_layer_group: Sequence[Optional[np.ndarray]],
    excluded: FrozenSet[BufferRef] = frozenset(),
) -> int:
    """Local bytes a chunk's buffers move: what the transfer writes, never physical slot bytes.

    Buffers that never travel occupy slot offsets no listed buffer covers, and excluded buffers do
    not move.
    """
    if pages is None:
        return 0
    total = 0
    for g, group_units in enumerate(units_per_layer_group):
        if group_units is None or group_units.size == 0:
            continue
        for i, buffer in enumerate(pages.buffers(g)):
            if (g, i) not in excluded:
                total += units.transfer_bytes(pages.kind(g), buffer.size, group_units)
    return total


def has_whole_request_groups(pages: Optional[RankPages]) -> bool:
    """Whether a layer group's units can only move once the whole prompt is done."""
    if pages is None:
        return False
    return any(not units.sends_incrementally(pages.kind(g)) for g in range(pages.num_layer_groups))


def excluded_buffers(manager, pages: Optional[RankPages], request) -> FrozenSet[BufferRef]:
    """``(layer group, buffer)`` of every buffer this request does not move, as the cache manager
    names them (``KVCacheManagerV2.get_untransferred_buffers``)."""
    if pages is None:
        return frozenset()
    left_out = manager.get_untransferred_buffers(request)
    if not left_out:
        return frozenset()
    return frozenset(
        (g, i)
        for g in range(pages.num_layer_groups)
        for i, buffer in enumerate(pages.buffers(g))
        if (buffer.layer, buffer.role) in left_out
    )
