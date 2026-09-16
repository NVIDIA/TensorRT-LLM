# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Model entry layouts, passed separately from KVCM's borrowed HostSourceView.

KVCM owns host locations, token coverage, and storage protection. EntryLayout
only describes the model's stored entries, byte offsets, strides, and scales.
The future ensure_resident() path takes both alongside logical selections.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class EntryComponent:
    """One byte span per entry, such as K, V, residuals, or quantization scales.

    Offsets and strides are bytes, relative to a host pool's page slot. Use more
    than one component for head-major data or a separate scale region. The model
    supplies this layout; buffer size alone does not reveal the entry stride.
    """

    name: str
    pool_index: int
    offset: int
    stride: int
    size: int

    def __post_init__(self) -> None:
        if (
            not self.name
            or min(self.pool_index, self.offset) < 0
            or self.size <= 0
            or self.stride < self.size
        ):
            raise ValueError("Components need a name, nonnegative offsets, and stride >= size > 0")


@dataclass(frozen=True)
class EntryLayout:
    """One KVCM layer's selected-entry layout within a lifecycle's pool group.

    entries_per_page counts stored entries, not necessarily input tokens. For
    compression by 4, a 16-token page holds 4 entries; positions are already in
    compressed-entry units. Scale bytes travel with the entry they describe.
    Pool sizes are bytes per slot from KVCM's storage layout. Component offsets
    include KVCM's buffer offset within that slot (including coalesced layers).
    """

    layer_id: int
    life_cycle_id: int
    pool_group_index: int
    entries_per_page: int
    pool_slot_bytes: tuple[int, ...]
    components: tuple[EntryComponent, ...]

    def __post_init__(self) -> None:
        if (
            min(self.layer_id, self.life_cycle_id, self.pool_group_index) < 0
            or self.entries_per_page <= 0
        ):
            raise ValueError("Layout IDs must be nonnegative and entries_per_page positive")
        if not self.components or not self.pool_slot_bytes or min(self.pool_slot_bytes) <= 0:
            raise ValueError("Layout needs components and positive pool slot sizes")
        if len({c.name for c in self.components}) != len(self.components):
            raise ValueError("Component names must be unique")
        for c in self.components:
            if c.pool_index >= len(self.pool_slot_bytes):
                raise ValueError("Component refers to an unknown pool")
            if (
                c.offset + (self.entries_per_page - 1) * c.stride + c.size
                > self.pool_slot_bytes[c.pool_index]
            ):
                raise ValueError("Component extends beyond its pool slot")
