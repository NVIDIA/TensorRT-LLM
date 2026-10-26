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
"""A rank's pages as its peers read them: the cache manager's page layout, and where its pools are.

Nothing here describes a cache: the layout is the manager's (``KVCacheManagerV2.get_page_layout``).
This module only indexes it the way the transfer walks it -- one list of buffers per layer group,
each layer group indexed per request by unit ids (block ordinals or a state slot) -- and carries it
to peers (``to_dict``).
"""

from __future__ import annotations

import dataclasses
from functools import cached_property
from typing import Dict, List, Tuple

import numpy as np

from tensorrt_llm._torch.disaggregation.base import CacheKind
from tensorrt_llm._torch.disaggregation.base.region import MemRegionGroup, SpecRegion
from tensorrt_llm._torch.pyexecutor.kv_cache.layout_builder import DevicePool, device_pools
from tensorrt_llm._torch.pyexecutor.kv_cache.page_layout import (
    BufferDesc,
    LayerGroupDesc,
    PageLayout,
    RankTopology,
    Segment,
    ShardDesc,
)
from tensorrt_llm._utils import nvtx_range

BufferRef = Tuple[int, int]
"""``(layer group, buffer index)`` within one rank's pages."""

PoolKey = Tuple[int, int]
"""``(pool group, pool index)`` of one device pool."""

_KINDS = {"block": CacheKind.PAGED, "slot": CacheKind.STATE}


@dataclasses.dataclass(frozen=True, eq=False)
class RankPages:
    """A rank's page layout and its device pools.

    Layer group ``g`` is ``layout.layer_groups[g]``, the manager's own group order, so the per
    request unit lists line up with the manager's groups. Its transferred buffers, in slot order,
    are ``buffers(g)``; a buffer is matched with a peer's by ``(global layer, role)``.
    """

    layout: PageLayout
    pools: Tuple[DevicePool, ...]

    @classmethod
    def of_manager(cls, manager) -> "RankPages":
        layout = manager.get_page_layout()
        return cls(layout, tuple(p for group in device_pools(manager).values() for p in group))

    # -- reading the layout --------------------------------------------------------------------

    @property
    def num_layer_groups(self) -> int:
        return len(self.layout.layer_groups)

    def group(self, g: int) -> LayerGroupDesc:
        return self.layout.layer_groups[g]

    def kind(self, g: int) -> CacheKind:
        """PAGED: the group's unit ids are blocks by ordinal; STATE: one slot per request."""
        return _KINDS[self.group(g).unit]

    @cached_property
    def _buffers(self) -> Tuple[Tuple[BufferDesc, ...], ...]:
        out = []
        for desc in self.layout.layer_groups:
            layers = set(desc.layers)
            out.append(
                tuple(
                    sorted(
                        (
                            b
                            for b in self.layout.buffers
                            if b.transferred
                            and b.pool_group == desc.pool_group
                            and b.layer in layers
                        ),
                        key=lambda b: (b.pool_index, b.slot_offset),
                    )
                )
            )
        return tuple(out)

    def buffers(self, g: int) -> Tuple[BufferDesc, ...]:
        """Layer group ``g``'s transferred buffers, in slot order. A buffer that never travels
        keeps its place in the slot but is not listed."""
        return self._buffers[g]

    @cached_property
    def _pools(self) -> Dict[PoolKey, DevicePool]:
        return {(p.group, p.index): p for p in self.pools}

    def pool(self, key: PoolKey) -> DevicePool:
        return self._pools[key]

    def group_pools(self, g: int) -> List[DevicePool]:
        """The pools holding a layer group's transferred buffers, in pool order. A pool holding only
        buffers that never travel is left out: nothing reads or writes it remotely."""
        keys = {(b.pool_group, b.pool_index) for b in self.buffers(g)}
        return [self.pool(key) for key in sorted(keys)]

    @nvtx_range("RankPages.slots")
    def slots(self, key: PoolKey, unit_ids: np.ndarray) -> SpecRegion:
        """The slot of each valid unit id in one pool, ``slot_bytes`` long. Negative ids
        (``BAD_PAGE_INDEX``) are dropped; which bytes inside a slot move is the transfer group's
        run plan."""
        pool = self.pool(key)
        ptrs = pool.base + pool.slot_bytes * unit_ids[unit_ids >= 0]
        return SpecRegion(memory=MemRegionGroup(ptrs=ptrs, bytes_per_region=pool.slot_bytes))

    # -- the wire ------------------------------------------------------------------------------

    def to_dict(self) -> dict:
        layout = self.layout
        segment_rows: Dict[Tuple[Segment, ...], int] = {}
        buffers = []
        for b in layout.buffers:
            index = segment_rows.setdefault(b.segments, len(segment_rows))
            buffers.append(
                [
                    b.layer,
                    b.role,
                    b.pool_group,
                    b.pool_index,
                    b.slot_offset,
                    b.size,
                    index,
                    b.transferred,
                    b.expansion,
                ]
            )
        return {
            "tokens_per_page": layout.tokens_per_page,
            "pool_slot_bytes": [[g, list(w)] for g, w in layout.pool_slot_bytes.items()],
            "layer_groups": [
                [
                    d.kind,
                    d.kept_window,
                    d.num_sink_blocks,
                    d.pool_group,
                    list(d.layers),
                    [d.shard.count, d.shard.index],
                    d.read_window,
                    list(d.model_layers),
                    list(d.grid),
                    d.unit,
                    d.blocks_by_cp,
                ]
                for d in layout.layer_groups
            ],
            # Most buffers share a layout, so each distinct one is sent once.
            "segments": [
                [[s.nbytes, [list(a) for a in s.axes], s.split_axis] for s in segments]
                for segments in segment_rows
            ],
            "buffers": buffers,
            "topology": dataclasses.asdict(layout.topology),
            "total_kv_heads_by_layer": [[k, v] for k, v in layout.total_kv_heads_by_layer.items()],
            "dtype": layout.dtype,
            "quant_mode": layout.quant_mode,
            "pools": [[p.group, p.index, p.base, p.slot_bytes, p.num_slots] for p in self.pools],
        }

    @staticmethod
    def from_dict(data: dict) -> "RankPages":
        segment_rows = [
            tuple(
                Segment(int(n), tuple((str(a), int(e)) for a, e in axes), split)
                for n, axes, split in row
            )
            for row in data["segments"]
        ]
        layout = PageLayout(
            tokens_per_page=int(data["tokens_per_page"]),
            pool_slot_bytes={int(g): tuple(w) for g, w in data["pool_slot_bytes"]},
            layer_groups=tuple(
                LayerGroupDesc(
                    kind=kind,
                    kept_window=kept,
                    num_sink_blocks=sinks,
                    pool_group=pool_group,
                    layers=tuple(layers),
                    shard=ShardDesc(*shard),
                    read_window=read,
                    model_layers=tuple(model_layers),
                    grid=tuple(grid),
                    unit=unit,
                    blocks_by_cp=bool(blocks_by_cp),
                )
                for (
                    kind,
                    kept,
                    sinks,
                    pool_group,
                    layers,
                    shard,
                    read,
                    model_layers,
                    grid,
                    unit,
                    blocks_by_cp,
                ) in data["layer_groups"]
            ),
            buffers=tuple(
                BufferDesc(
                    layer=layer,
                    role=role,
                    pool_group=pool_group,
                    pool_index=pool_index,
                    slot_offset=offset,
                    size=size,
                    segments=segment_rows[index],
                    transferred=bool(transferred),
                    expansion=expansion,
                )
                for (
                    layer,
                    role,
                    pool_group,
                    pool_index,
                    offset,
                    size,
                    index,
                    transferred,
                    expansion,
                ) in data["buffers"]
            ),
            topology=RankTopology(**data["topology"]),
            total_kv_heads_by_layer={int(k): int(v) for k, v in data["total_kv_heads_by_layer"]},
            dtype=data["dtype"],
            quant_mode=data["quant_mode"],
        )
        pools = tuple(DevicePool(*map(int, row)) for row in data["pools"])
        return RankPages(layout, pools)
