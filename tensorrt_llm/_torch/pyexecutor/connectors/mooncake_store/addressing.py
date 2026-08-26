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
"""Turning a ``KvCacheLayout`` into addresses Mooncake can transfer.

Mooncake's batch APIs take, per key, a list of ``(address, size)`` buffers. That
is exactly the shape of a V2 page: a layer group's regions each contribute one
byte range at ``base + stride * page_index``, and the concatenation of those
ranges in region order is the page's payload.

Region order is therefore load-bearing -- it is the value's serialization -- and
``build_kv_cache_layout_v2`` derives it from the allocator's own aggregation, so
it is stable for a given model and parallel layout. ``bytes_per_page`` goes into
the key namespace to keep a geometry change from being read as a valid page.
"""

from typing import Dict, Iterable, List, Sequence, Tuple

from ..kv_cache_layout import KvCacheLayout, KvCacheRegion

__all__ = ["PageAddressing", "merge_intervals"]


def merge_intervals(intervals: Iterable[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """Collapse ``(start, end)`` byte ranges into a minimal disjoint cover.

    Registration is per range and a range may not be registered twice, but
    several regions routinely live inside one pool allocation: sliding-window
    layer groups share it, and a non-uniform slot (MiniMax-M3's index-K sitting
    beside K/V) splits one pool into several regions. Merging first means the
    caller does not have to know which case it is in.
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
    """Resolves ``(layer group, page index)`` to the byte ranges of that page."""

    def __init__(self, layout: KvCacheLayout):
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
            # Every region of a group is drawn from the same pool group, so they
            # share a slot count; disagreement would mean the page index space is
            # not the single space the layout documents.
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
    def layer_group_ids(self) -> Tuple[int, ...]:
        """Layer group ids covered, in layout order."""
        return tuple(group.layer_group_id for group in self._layout.groups)

    @property
    def tokens_per_block(self) -> int:
        """Tokens held by one page."""
        return self._layout.tokens_per_block

    def bytes_per_page(self, layer_group_id: int) -> int:
        """Total payload size of one page of ``layer_group_id``."""
        return self._bytes_per_page[layer_group_id]

    def num_slots(self, layer_group_id: int) -> int:
        """Number of page slots addressable in ``layer_group_id``."""
        return self._num_slots[layer_group_id]

    def buffers(self, layer_group_id: int, page_index: int) -> Tuple[List[int], List[int]]:
        """Addresses and sizes of one page, in the order they concatenate.

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
        addresses = [region.base + region.stride * page_index for region in regions]
        sizes = [region.size for region in regions]
        return addresses, sizes

    def registration_ranges(self) -> List[Tuple[int, int]]:
        """Byte ranges to hand to ``register_buffer``, deduplicated and merged.

        A region's slots are strided rather than packed, so the range covering it
        is the whole span from the first slot to the end of the last. Registering
        the span is what makes every slot's address valid for RDMA, and merging
        keeps a shared pool from being registered once per region.
        """
        spans: List[Tuple[int, int]] = []
        for regions in self._regions.values():
            for region in regions:
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
            f"slots={self._num_slots[group.layer_group_id]}, "
            f"window={group.window_size})"
            for group in self._layout.groups
        ]
        return f"tokens_per_block={self.tokens_per_block}, " + ", ".join(parts)
