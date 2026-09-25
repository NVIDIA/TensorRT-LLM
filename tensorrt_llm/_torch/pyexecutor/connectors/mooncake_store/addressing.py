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

Mooncake's batch APIs take, per key, a list of `(address, size)` buffers. That
is exactly the shape of a V2 page: a layer group's regions each contribute one
byte range at `base + stride * page_index`, and the concatenation of those
ranges in region order is the page's payload.

Region order is therefore the value's serialization, and it is stable for a
given model and parallel layout because `build_kv_cache_layout_v2` derives it
from the allocator's own aggregation. `bytes_per_page` goes into
the key namespace to keep a geometry change from being read as a valid page.

A layer group yields two pages rather than one. Most roles hold bytes belonging
to a single attention shard; roles the manager declares replicated hold bytes
identical on every shard. They are addressed separately so the connector can
key the replicated page once for the whole TP group. Groups with no replicated
role report an empty second page, which the worker skips.
"""

from typing import Dict, Iterable, List, Sequence, Tuple

from ..kv_cache_layout import KvCacheLayout, KvCacheRegion

__all__ = ["PageAddressing", "merge_intervals"]


def merge_intervals(intervals: Iterable[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """Collapse `(start, end)` byte ranges into a minimal disjoint cover.

    Registration is per range and a range may not be registered twice, but
    several regions routinely live inside one pool allocation: sliding-window
    layer groups share it, and separating replicated roles from shard-specific
    ones (MiniMax-M3's index-K sitting beside K/V) splits one pool into
    interleaved regions of both classes. Merging first means the caller does
    not have to know which case it is in.
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
        self._replicated_regions: Dict[int, Tuple[KvCacheRegion, ...]] = {}
        self._bytes_per_page: Dict[int, int] = {}
        self._replicated_bytes_per_page: Dict[int, int] = {}
        self._num_slots: Dict[int, int] = {}
        for group in layout.groups:
            if not group.regions:
                raise ValueError(
                    f"layer group {group.layer_group_id} has no KV regions; there "
                    "is nothing for the connector to transfer"
                )
            self._regions[group.layer_group_id] = group.regions
            self._replicated_regions[group.layer_group_id] = group.replicated_regions
            self._bytes_per_page[group.layer_group_id] = group.bytes_per_page
            self._replicated_bytes_per_page[group.layer_group_id] = (
                group.replicated_bytes_per_page
            )
            # Every region of a group is drawn from the same pool group, so they
            # share a slot count; disagreement would mean the page index space is
            # not the single space the layout documents. Replicated regions are
            # indexed by that same space, so they are held to it too.
            slot_counts = {
                region.num_slots for region in (*group.regions, *group.replicated_regions)
            }
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
    def layer_group_ids(self) -> Tuple[int, ...]:
        """Layer group ids covered, in layout order."""
        return tuple(group.layer_group_id for group in self._layout.groups)

    @property
    def tokens_per_block(self) -> int:
        """Tokens held by one page."""
        return self._layout.tokens_per_block

    def bytes_per_page(self, layer_group_id: int) -> int:
        """Payload size of one shard-specific page of `layer_group_id`."""
        return self._bytes_per_page[layer_group_id]

    def replicated_bytes_per_page(self, layer_group_id: int) -> int:
        """Payload size of one replicated page of `layer_group_id`, or 0."""
        return self._replicated_bytes_per_page[layer_group_id]

    def has_replicated(self, layer_group_id: int) -> bool:
        """Whether `layer_group_id` holds any replicated-role bytes."""
        return bool(self._replicated_regions[layer_group_id])

    def num_slots(self, layer_group_id: int) -> int:
        """Number of page slots addressable in `layer_group_id`."""
        return self._num_slots[layer_group_id]

    def buffers(self, layer_group_id: int, page_index: int) -> Tuple[List[int], List[int]]:
        """Addresses and sizes of one shard-specific page, in concatenation order.

        Args:
            layer_group_id: Layer group the page index is scoped to.
            page_index: Page slot index within that group.

        Returns:
            Parallel lists of device addresses and byte counts.
        """
        return self._locate(self._regions[layer_group_id], layer_group_id, page_index)

    def replicated_buffers(
        self, layer_group_id: int, page_index: int
    ) -> Tuple[List[int], List[int]]:
        """Addresses and sizes of one replicated page, in concatenation order.

        These bytes are identical on every attention shard, so the same page is
        described here on every rank even though each rank names its own copy.

        Args:
            layer_group_id: Layer group the page index is scoped to.
            page_index: Page slot index within that group.

        Returns:
            Parallel lists of device addresses and byte counts. Both are empty
            when the group holds no replicated roles.
        """
        return self._locate(self._replicated_regions[layer_group_id], layer_group_id, page_index)

    def _locate(
        self, regions: Sequence[KvCacheRegion], layer_group_id: int, page_index: int
    ) -> Tuple[List[int], List[int]]:
        num_slots = self._num_slots[layer_group_id]
        if not 0 <= page_index < num_slots:
            raise IndexError(
                f"page index {page_index} out of range [0, {num_slots}) for layer "
                f"group {layer_group_id}"
            )
        addresses = [region.base + region.stride * page_index for region in regions]
        sizes = [region.size for region in regions]
        return addresses, sizes

    def registration_ranges(self) -> List[Tuple[int, int]]:
        """Byte ranges to hand to `register_buffer`, deduplicated and merged.

        A region's slots are strided rather than packed, so the range covering it
        is the whole span from the first slot to the end of the last. Registering
        the span is what makes every slot's address valid for RDMA, and merging
        keeps a shared pool from being registered once per region. Both region
        classes are covered: replicated bytes are transferred like any other, so
        leaving them unregistered would fail every transfer that touches them.
        """
        spans: List[Tuple[int, int]] = []
        for layer_group_id, regions in self._regions.items():
            for region in (*regions, *self._replicated_regions[layer_group_id]):
                span_end = region.base + region.stride * (region.num_slots - 1) + region.size
                spans.append((region.base, span_end))
        return merge_intervals(spans)

    def describe(self) -> str:
        """A one-line summary for startup logs."""
        parts: Sequence[str] = [
            f"lg{group.layer_group_id}("
            f"layers={len(group.layer_ids)}, "
            f"regions={len(group.regions)}, "
            f"bytes/page={group.bytes_per_page}, "
            f"replicated_regions={len(group.replicated_regions)}, "
            f"replicated_bytes/page={group.replicated_bytes_per_page}, "
            f"slots={self._num_slots[group.layer_group_id]}, "
            f"window={group.window_size})"
            for group in self._layout.groups
        ]
        return f"tokens_per_block={self.tokens_per_block}, " + ", ".join(parts)
