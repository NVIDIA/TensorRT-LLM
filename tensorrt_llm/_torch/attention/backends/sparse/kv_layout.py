# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Entry layouts and separate host/GPU views over existing per-request KvCache objects.

These views describe storage; they do not allocate KV, hold pages, copy bytes,
change page residency, or choose victims. Owners must keep tables and storage
alive and stable through GPU use. Build views and outputs before graph capture.
"""

from dataclasses import dataclass

import torch

from .selection import SelectedEntries, check_tensor


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


@dataclass(frozen=True)
class GpuCacheView:
    """Per-entry GPU mapping for one layer/lifecycle, separate from Page.cacheLevel.

    request_ids: CUDA int64 [requests], current owners of mapping rows.
    entry_indices: CUDA int32 [requests, logical_capacity], maps logical positions
        to attention-ready GPU entry indices; -1 means missing. The cache owner
        publishes entries only after copies finish and protects them through use.
        An entry hit does not make the rest of its page GPU-resident.
    """

    layer_id: int
    life_cycle_id: int
    request_ids: torch.Tensor
    entry_indices: torch.Tensor

    def __post_init__(self) -> None:
        if (
            min(self.layer_id, self.life_cycle_id) < 0
            or self.request_ids.ndim != 1
            or self.entry_indices.ndim != 2
            or self.entry_indices.shape[0] != self.request_ids.numel()
        ):
            raise ValueError("GPU entry tables need nonnegative IDs and one row per request")
        check_tensor(self.request_ids, self.request_ids.shape, torch.int64, self.request_ids.device)
        check_tensor(
            self.entry_indices, self.entry_indices.shape, torch.int32, self.request_ids.device
        )


@dataclass(frozen=True)
class ResolvedEntries:
    """Caller-owned CUDA outputs, in the same row/column order as the selection.

    valid: bool [queries, top_k], logical selection validity.
    host_valid: bool [queries, top_k], all components have a completed host copy.
    host_offsets: int64 [queries, top_k, components], bytes from each component's
        pool base; -1 if host data is unavailable. Sizes come from EntryLayout.
    gpu_indices: int32 [queries, top_k], ready entry indices or -1 on a miss.
    A valid selection can miss both stores: the caller must arrange backup/fetch
    or report failure, never silently remove it from attention.
    """

    valid: torch.Tensor
    host_valid: torch.Tensor
    host_offsets: torch.Tensor
    gpu_indices: torch.Tensor


def resolve_entries(
    selection: SelectedEntries,
    host: HostStorageView,
    gpu: GpuCacheView,
    out: ResolvedEntries,
) -> None:
    """Resolve on GPU into preallocated outputs, with no copies or residency changes.

    Table rows follow SelectionContext.request_ids. Stored IDs must also match;
    a reused row with an old ID is a storage miss, not someone else's KV. Missing
    storage never changes the logical selection. No CPU read of GPU values occurs.
    """
    from .selection_kernels import resolve_entries_kernel

    ctx, layout = selection.context, host.layout
    if (ctx.layer_id, ctx.life_cycle_id) != (layout.layer_id, layout.life_cycle_id) or (
        gpu.layer_id,
        gpu.life_cycle_id,
    ) != (layout.layer_id, layout.life_cycle_id):
        raise ValueError(
            "Selection, host layout, and GPU mapping must name the same layer/lifecycle"
        )
    device = selection.positions.device
    if device.type != "cuda":
        raise ValueError("Selection resolution requires CUDA tensors")
    nreq = ctx.request_ids.numel()
    for tensor in (host.request_ids, gpu.request_ids):
        check_tensor(tensor, (nreq,), torch.int64, device)
    for tensor in (host.page_slots, host.valid_entries, gpu.entry_indices):
        if tensor.device != device:
            raise ValueError("All tables must be on the selection device")
    shape = selection.positions.shape
    check_tensor(out.valid, shape, torch.bool, device)
    check_tensor(out.host_valid, shape, torch.bool, device)
    check_tensor(out.gpu_indices, shape, torch.int32, device)
    check_tensor(out.host_offsets, (*shape, len(layout.components)), torch.int64, device)
    if any(
        not t.is_contiguous()
        for t in (out.valid, out.host_valid, out.host_offsets, out.gpu_indices)
    ):
        raise ValueError("Resolution outputs must be contiguous")
    if selection.positions.numel() == 0:
        return
    components = tuple((c.pool_index, c.offset, c.stride, c.size) for c in layout.components)
    resolve_entries_kernel[(shape[0], (shape[1] + 127) // 128)](
        selection.positions,
        ctx.request_indices,
        ctx.request_ids,
        ctx.valid_lengths,
        selection.valid_mask,
        host.request_ids,
        host.page_slots,
        host.valid_entries,
        gpu.request_ids,
        gpu.entry_indices,
        out.valid,
        out.host_valid,
        out.host_offsets,
        out.gpu_indices,
        shape[1],
        nreq,
        host.page_slots.shape[1],
        gpu.entry_indices.shape[1],
        selection.positions.stride(0),
        selection.positions.stride(1),
        ctx.request_indices.stride(0),
        ctx.request_ids.stride(0),
        ctx.valid_lengths.stride(0),
        selection.valid_mask.stride(0) if selection.valid_mask is not None else 0,
        selection.valid_mask.stride(1) if selection.valid_mask is not None else 0,
        host.request_ids.stride(0),
        *host.page_slots.stride(),
        *host.valid_entries.stride(),
        gpu.request_ids.stride(0),
        *gpu.entry_indices.stride(),
        ENTRIES_PER_PAGE=layout.entries_per_page,
        COMPONENTS=components,
        SLOT_BYTES=layout.pool_slot_bytes,
        POOL_BYTES=host.pool_bytes,
        HAS_MASK=selection.valid_mask is not None,
        BLOCK=128,
    )
