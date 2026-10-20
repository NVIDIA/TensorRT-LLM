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
"""Whole-table questions, answered space by space by each space's spec.

Memory to register, bounce sizing, bytes a chunk moves, and the request-wide rules (how much history
moves, which regions are left out, whether first tokens travel, where a received request resumes).
"""

from __future__ import annotations

from typing import Dict, FrozenSet, List, Optional, Sequence, Tuple

import numpy as np

from .page_table import PageTable
from .registry import spec_for, specs_in


def memory_registration_descs(page_table: PageTable, device_id: int) -> List[tuple]:
    """Deduplicated ``(ptr, size, device_id, name)`` of every allocation backing the cache."""
    unique: Dict[tuple, int] = {}
    for space_index, space in enumerate(page_table.spaces):
        for region in spec_for(space).registration_regions(page_table, space_index):
            if region.within_known and any(
                base <= region.base_address < base + size for base, size in unique
            ):
                continue
            unique.setdefault((region.base_address, region.size), len(unique))
    return [
        (ptr, size, device_id, f"kv_cache_memory_pool{idx}") for (ptr, size), idx in unique.items()
    ]


def has_replicated_regions(page_table: Optional[PageTable]) -> bool:
    """Whether any region is moved by one elected sender rather than by every rank."""
    if page_table is None:
        return False
    return any(region.replicated for space in page_table.spaces for region in space.regions)


def bounce_unit_bytes_per_space(page_table: PageTable) -> List[Optional[int]]:
    """Bytes per unit the bounce buffer reserves for each space (None: sized separately)."""
    return [
        spec_for(space).bounce_unit_bytes(page_table, space_index)
        for space_index, space in enumerate(page_table.spaces)
    ]


def receiver_extra_bytes(
    sender_page_table: Optional[PageTable],
    receiver_page_table: PageTable,
    units_per_space: Sequence[np.ndarray],
) -> Optional[int]:
    """Bytes outside the per-ordinal bounce sizing landing in the receiver's units.

    ``None`` when some space needs the sender's page table to size it and none is known.
    """
    total = 0
    for space_index, space in enumerate(receiver_page_table.spaces):
        units = units_per_space[space_index]
        if units is None or units.size == 0:
            continue
        spec = spec_for(space)
        if spec.bounce_unit_bytes(receiver_page_table, space_index) is not None:
            continue
        if sender_page_table is None:
            return None
        total += spec.receiver_extra_bytes(
            sender_page_table, receiver_page_table, space_index, units
        )
    return total


def chunk_transfer_bytes(
    page_table: Optional[PageTable],
    units_per_space: Sequence[Optional[np.ndarray]],
    excluded: FrozenSet[Tuple[int, int]] = frozenset(),
) -> int:
    """Local bytes a chunk's regions move: what the transfer writes, never physical slot bytes.

    Ignored-role buffers occupy slot offsets no region covers, and excluded regions do not move.
    """
    if page_table is None:
        return 0
    total = 0
    for space_index, units in enumerate(units_per_space):
        if units is None or units.size == 0:
            continue
        space = page_table.spaces[space_index]
        spec = spec_for(space)
        for region_index, region in enumerate(space.regions):
            if (space_index, region_index) not in excluded:
                total += spec.transfer_bytes(space, region.nbytes, units)
    return total


def whole_request_specs(page_table: Optional[PageTable]) -> List[str]:
    """Specs with a space whose units can only move once the whole prompt is done."""
    if page_table is None:
        return []
    return sorted(
        {
            space.spec
            for space in page_table.spaces
            if not spec_for(space).sends_incrementally(space)
        }
    )


def transfer_length(page_table: Optional[PageTable], request) -> int:
    """Prompt tokens whose history the request moves: the least every spec allows."""
    specs = specs_in(page_table)
    if not specs:
        return request.prompt_len
    return min(spec.transfer_length(request) for spec in specs)


def resume_after_receive(page_table: Optional[PageTable], request) -> int:
    """Where a request whose whole prompt arrived continues: the earliest point every spec allows."""
    specs = specs_in(page_table)
    if not specs:
        return request.prompt_len
    return min(spec.resume_after_receive(request) for spec in specs)


def sends_first_tokens(page_table: Optional[PageTable], request) -> bool:
    return all(spec.sends_first_tokens(request) for spec in specs_in(page_table))


def excluded_regions(
    manager, page_table: Optional[PageTable], request
) -> FrozenSet[Tuple[int, int]]:
    """``(space, region)`` of every region this request does not move."""
    if page_table is None:
        return frozenset()
    return frozenset(
        (space_index, region_index)
        for space_index, space in enumerate(page_table.spaces)
        for region_index in spec_for(space).excluded_regions(
            manager, page_table, space_index, request
        )
    )
