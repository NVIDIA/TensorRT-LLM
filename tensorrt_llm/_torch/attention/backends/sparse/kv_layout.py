# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Entry layouts and host views over existing per-request KvCache objects.

These views describe storage; they do not allocate KV, hold pages, copy bytes,
change page residency, or choose victims. Owners must keep tables and storage
alive and stable through GPU use. Build views before graph capture.

The planned HiSparse path takes logical selections, layout/host views, and its
own mutable GPU-cache state directly.
"""

from dataclasses import dataclass

import torch

from .selection import check_tensor


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


@dataclass(frozen=True)
class HostStorageView:
    """Host pool offsets, separate from ordinary GPU page indices.

    request_ids: CUDA int64 [requests], identifies current table rows.
    page_slots: CUDA int64 [requests, pages], host slot IDs; -1 means absent.
        A page uses the same slot ID across its lifecycle's pools.
    valid_entries: CUDA int32 [requests, pages], completed host entries in each
        page, as a prefix. A copy in flight is not valid. Full, uncommitted pages
        are allowed. Task 4 supplies retained copies and their protection.
    pool_bytes: actual allocated byte capacity of each host pool. This view
        stores offsets rather than CPU pointers; the transfer backend owns and
        registers the corresponding pinned memory and keeps it alive.
    """

    layout: EntryLayout
    request_ids: torch.Tensor
    page_slots: torch.Tensor
    valid_entries: torch.Tensor
    pool_bytes: tuple[int, ...]

    def __post_init__(self) -> None:
        if (
            self.request_ids.ndim != 1
            or self.page_slots.ndim != 2
            or self.page_slots.shape[0] != self.request_ids.numel()
        ):
            raise ValueError("Host page tables need one row per request")
        device = self.request_ids.device
        check_tensor(self.request_ids, self.request_ids.shape, torch.int64, device)
        check_tensor(self.page_slots, self.page_slots.shape, torch.int64, device)
        check_tensor(self.valid_entries, self.page_slots.shape, torch.int32, device)
        if len(self.pool_bytes) != len(self.layout.pool_slot_bytes) or min(self.pool_bytes) < 0:
            raise ValueError("Provide a nonnegative byte capacity for each host pool")
