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

A V2 GPU pool is a virtual reservation mapped with many `cuMemCreate`
allocations, and GPUDirect cannot register a range that crosses two of them.
Given an `AllocationLookup`, registrations and transfer buffers are cut at
those boundaries.
"""

from bisect import bisect_right
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from ..kv_cache_layout import KvCacheLayout, KvCacheRegion

__all__ = ["PageAddressing"]

#: Resolves an address to `(base, size)` of the allocation mapped there, or
#: None when it cannot. `gpudirect.allocation_range` asks the driver.
AllocationLookup = Callable[[int], Optional[Tuple[int, int]]]


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


def allocation_spans(
    start: int, end: int, lookup: AllocationLookup
) -> Optional[List[Tuple[int, int]]]:
    """Cut `[start, end)` at the boundaries of the allocations it covers.

    Args:
        start: First byte.
        end: One past the last byte.
        lookup: Resolves an address to its allocation.

    Returns:
        Consecutive `(start, end)` pieces, one per allocation touched, or None
        if any address cannot be resolved.
    """
    spans: List[Tuple[int, int]] = []
    cursor = start
    while cursor < end:
        allocation = lookup(cursor)
        if allocation is None:
            return None
        base, size = allocation
        stop = min(base + size, end)
        if not base <= cursor < stop:
            # An answer not containing `cursor` would loop or skip bytes.
            return None
        spans.append((cursor, stop))
        cursor = stop
    return spans


def split_at_cuts(start: int, size: int, cuts: Sequence[int]) -> List[Tuple[int, int]]:
    """Cut `[start, start + size)` at every address in `cuts` strictly inside it.

    Args:
        start: First byte.
        size: Length in bytes.
        cuts: Ascending addresses at which an allocation begins.

    Returns:
        Consecutive `(address, size)` pieces covering the input exactly.
    """
    end = start + size
    pieces: List[Tuple[int, int]] = []
    address = start
    index = bisect_right(cuts, start)
    while index < len(cuts) and cuts[index] < end:
        pieces.append((address, cuts[index] - address))
        address = cuts[index]
        index += 1
    pieces.append((address, end - address))
    return pieces


class PageAddressing:
    """Resolves `(layer group, page index)` to the byte ranges of that page."""

    def __init__(self, layout: KvCacheLayout, allocation_lookup: Optional[AllocationLookup] = None):
        """Index `layout`, and with `allocation_lookup` its allocation boundaries.

        Args:
            layout: The KV cache layout to address.
            allocation_lookup: Resolves an address to its allocation. Without
                it every range is left whole, which suffices for host staging.
        """
        self._layout = layout
        self._regions: Dict[int, Tuple[KvCacheRegion, ...]] = {}
        self._bytes_per_page: Dict[int, int] = {}
        self._num_slots: Dict[int, int] = {}
        for group in layout.groups:
            if not group.regions:
                raise ValueError(
                    f"layer group {group.layer_group_id} has no KV regions; there "
                    "is nothing for the connector to transfer"
                )
            self._regions[group.layer_group_id] = group.regions
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

        self._registration_ranges: List[Tuple[int, int]] = []
        self._unresolved: List[Tuple[int, int]] = []
        # Interior allocation boundaries, ascending, consulted per transfer.
        self._cuts: List[int] = []
        for start, end in self._pool_spans():
            pieces = (
                allocation_spans(start, end, allocation_lookup)
                if allocation_lookup is not None
                else None
            )
            if pieces is None:
                if allocation_lookup is not None:
                    self._unresolved.append((start, end))
                pieces = [(start, end)]
            self._registration_ranges.extend(pieces)
            self._cuts.extend(piece_start for piece_start, _ in pieces[1:])

    def _pool_spans(self) -> List[Tuple[int, int]]:
        """Merged byte spans the regions occupy, from first slot to last."""
        spans: List[Tuple[int, int]] = []
        for regions in self._regions.values():
            for region in regions:
                span_end = region.base + region.stride * (region.num_slots - 1) + region.size
                spans.append((region.base, span_end))
        return merge_intervals(spans)

    @property
    def layout(self) -> KvCacheLayout:
        """The layout this addressing was built from."""
        return self._layout

    @property
    def unresolved_ranges(self) -> Tuple[Tuple[int, int], ...]:
        """Spans whose allocations the lookup could not resolve, left whole."""
        return tuple(self._unresolved)

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

        A region crossing an allocation boundary contributes one buffer per
        allocation, in order.

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
        addresses: List[int] = []
        sizes: List[int] = []
        for region in regions:
            start = region.base + region.stride * page_index
            for address, size in split_at_cuts(start, region.size, self._cuts):
                addresses.append(address)
                sizes.append(size)
        return addresses, sizes

    def registration_ranges(self) -> List[Tuple[int, int]]:
        """Byte ranges to hand to `register_buffer`, one per allocation.

        Each region spans from its first slot to the end of its last, so every
        slot is registered. Spans are merged, so a shared pool is registered
        once, then cut at allocation boundaries. A span the lookup could not
        resolve is returned whole; see `unresolved_ranges`.
        """
        return list(self._registration_ranges)

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
