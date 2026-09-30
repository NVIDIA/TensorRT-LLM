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
"""Turning a `KvCacheLayout` into addresses Mooncake can transfer.

Mooncake's batch APIs take, per key, a list of `(address, size)` buffers, which
is the shape of a V2 page: a layer group's regions each contribute one byte
range at `base + stride * page_index`, and their concatenation in region order
is the page's payload.

Region order is therefore the value's serialization, stable for a given model
and parallel layout because `build_kv_cache_layout_v2` derives it from the
allocator's own aggregation. `bytes_per_page` goes into the key namespace so a
geometry change cannot be read as a valid page.
"""

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from ..kv_cache_layout import KvCacheLayout, KvCacheRegion
from .gpudirect import reservation_start

__all__ = ["PageAddressing", "mapping_origin", "merge_intervals", "split_at_boundaries"]


def split_at_boundaries(
    start: int, size: int, boundary: Optional[int], origin: int = 0
) -> List[Tuple[int, int]]:
    """Cut `[start, start + size)` wherever it crosses a mapping boundary.

    Keeps registrations and transfer buffers inside a single GPU pool mapping,
    for the reason given on `KvCacheLayout.gpu_pool_mapping_bytes`. The pieces
    are consecutive, so a page's payload is the same concatenation either way.

    Boundaries sit at `origin + k * boundary` rather than at every multiple of
    `boundary`: the mappings tile a pool from the base of its reservation, and
    nothing aligns that base to the mapping size.

    Args:
        start: First byte.
        size: Length in bytes.
        boundary: Mapping size, or None to leave the range whole.
        origin: An address a boundary falls on, at or below `start`.

    Returns:
        Consecutive `(address, size)` pieces covering the input exactly.
    """
    if not boundary or size <= 0:
        return [(start, size)]
    pieces: List[Tuple[int, int]] = []
    address, remaining = start, size
    while remaining > 0:
        # Distance to the next boundary at or above `address`.
        to_boundary = boundary - (address - origin) % boundary
        take = min(remaining, to_boundary)
        pieces.append((address, take))
        address += take
        remaining -= take
    return pieces


def mapping_origin(address: int, boundary: Optional[int]) -> int:
    """The address the mapping boundaries around `address` are counted from.

    The base of the reservation the pool was mapped into, as the driver reports
    it. Where that cannot be read there is nothing to place the boundaries
    against, so the multiples of `boundary` are assumed, which is what this
    connector did before it split at all. A wrong assumption is not silent: the
    same origin decides the registrations, so a piece that still spans two
    mappings fails `register_buffer` at startup with the diagnosis attached,
    rather than moving a page's bytes to the wrong place.
    """
    if not boundary:
        return 0
    start = reservation_start(address)
    return 0 if start is None else start


def merge_intervals(intervals: Iterable[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """Collapse `(start, end)` byte ranges into a minimal disjoint cover.

    A range may not be registered twice, but several regions routinely live
    inside one pool allocation: sliding-window layer groups share it, and a
    non-uniform slot such as MiniMax-M3's index-K beside K/V splits one pool
    into several regions. Merging first spares the caller that distinction.
    """
    ordered = sorted((int(start), int(end)) for start, end in intervals if end > start)
    merged: List[Tuple[int, int]] = []
    for start, end in ordered:
        if merged and start <= merged[-1][1]:
            previous_start, previous_end = merged[-1]
            merged[-1] = (previous_start, max(previous_end, end))
        else:
            merged.append((start, end))
    return merged


class PageAddressing:
    """Resolves `(layer group, page index)` to the byte ranges of that page."""

    def __init__(self, layout: KvCacheLayout):
        self._layout = layout
        self._regions: Dict[int, Tuple[KvCacheRegion, ...]] = {}
        self._origins: Dict[int, Tuple[int, ...]] = {}
        self._bytes_per_page: Dict[int, int] = {}
        self._num_slots: Dict[int, int] = {}
        boundary = layout.gpu_pool_mapping_bytes
        for group in layout.groups:
            if not group.regions:
                raise ValueError(
                    f"layer group {group.layer_group_id} has no KV regions; there "
                    "is nothing for the connector to transfer"
                )
            self._regions[group.layer_group_id] = group.regions
            # Asked once per region here rather than per transfer: a region's
            # reservation is fixed for the lifetime of the pools.
            self._origins[group.layer_group_id] = tuple(
                mapping_origin(region.base, boundary) for region in group.regions
            )
            self._bytes_per_page[group.layer_group_id] = group.bytes_per_page
            # Regions of a group come from the same pool group and so share a
            # slot count. Disagreement would make the page index ambiguous.
            slot_counts = {region.num_slots for region in group.regions}
            if len(slot_counts) != 1:
                raise ValueError(
                    f"layer group {group.layer_group_id} mixes slot counts "
                    f"{sorted(slot_counts)}; page indices would be ambiguous"
                )
            self._num_slots[group.layer_group_id] = slot_counts.pop()

    @property
    def layout(self) -> KvCacheLayout:
        """The layout this addressing was built from."""
        return self._layout

    @property
    def mapping_bytes(self) -> Optional[int]:
        """Size of one GPU pool mapping, or None when the layout omits it."""
        return self._layout.gpu_pool_mapping_bytes

    @property
    def mapping_origins(self) -> Tuple[int, ...]:
        """The reservation bases the mapping boundaries are counted from.

        Ascending and distinct. A zero means that pool's base could not be read
        from the driver and the boundaries are being assumed; see
        `mapping_origin`.
        """
        origins = {origin for group in self._origins.values() for origin in group}
        return tuple(sorted(origins))

    @property
    def layer_group_ids(self) -> Tuple[int, ...]:
        """Layer group ids covered, in layout order."""
        return tuple(group.layer_group_id for group in self._layout.groups)

    @property
    def tokens_per_block(self) -> int:
        """Tokens held by one page."""
        return self._layout.tokens_per_block

    def bytes_per_page(self, layer_group_id: int) -> int:
        """Total payload size of one page of `layer_group_id`."""
        return self._bytes_per_page[layer_group_id]

    def num_slots(self, layer_group_id: int) -> int:
        """Number of page slots addressable in `layer_group_id`."""
        return self._num_slots[layer_group_id]

    def buffers(self, layer_group_id: int, page_index: int) -> Tuple[List[int], List[int]]:
        """Addresses and sizes of one page, in the order they concatenate.

        A region whose bytes for this page cross a GPU pool mapping boundary
        contributes several consecutive buffers rather than one, which leaves
        the payload and its order intact.

        Args:
            layer_group_id: Layer group the page index is scoped to.
            page_index: Page slot index within that group.

        Returns:
            Parallel lists of device addresses and byte counts.
        """
        regions = self._regions[layer_group_id]
        num_slots = self._num_slots[layer_group_id]
        if not 0 <= page_index < num_slots:
            raise IndexError(
                f"page index {page_index} out of range [0, {num_slots}) for layer "
                f"group {layer_group_id}"
            )
        boundary = self._layout.gpu_pool_mapping_bytes
        addresses: List[int] = []
        sizes: List[int] = []
        for region, origin in zip(regions, self._origins[layer_group_id]):
            start = region.base + region.stride * page_index
            for address, size in split_at_boundaries(start, region.size, boundary, origin):
                addresses.append(address)
                sizes.append(size)
        return addresses, sizes

    def registration_ranges(self) -> List[Tuple[int, int]]:
        """Byte ranges to hand to `register_buffer`, deduplicated and merged.

        A region's slots are strided rather than packed, so its range spans
        from the first slot to the end of the last. Registering the whole span
        is what makes every slot's address valid for RDMA, and merging keeps a
        shared pool from being registered once per region.

        Merged spans are then cut at GPU pool mapping boundaries, measured from
        the reservation each pool was mapped into, since a registration covering
        two mappings is refused. That costs one call per mapping, roughly a
        millisecond each, and is what lets the pools be registered on the
        dma-buf path rather than only through `nvidia_peermem`.
        """
        # Grouped by origin so each span is cut against its own reservation.
        # Regions in different reservations cannot overlap, so merging within a
        # group is the same cover as merging all the spans together.
        spans_by_origin: Dict[int, List[Tuple[int, int]]] = {}
        for layer_group_id, regions in self._regions.items():
            for region, origin in zip(regions, self._origins[layer_group_id]):
                span_end = region.base + region.stride * (region.num_slots - 1) + region.size
                spans_by_origin.setdefault(origin, []).append((region.base, span_end))

        boundary = self._layout.gpu_pool_mapping_bytes
        ranges: List[Tuple[int, int]] = []
        for origin, spans in spans_by_origin.items():
            for start, end in merge_intervals(spans):
                ranges.extend(
                    (address, address + size)
                    for address, size in split_at_boundaries(start, end - start, boundary, origin)
                )
        return sorted(ranges)

    def describe(self) -> str:
        """A one-line summary for startup logs."""
        parts: Sequence[str] = [
            f"lg{group.layer_group_id}("
            f"layers={len(group.layer_ids)}, "
            f"regions={len(group.regions)}, "
            f"bytes/page={group.bytes_per_page}, "
            f"slots={self._num_slots[group.layer_group_id]}, "
            f"window={group.window_size})"
            for group in self._layout.groups
        ]
        return f"tokens_per_block={self.tokens_per_block}, " + ", ".join(parts)
