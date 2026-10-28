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

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from tensorrt_llm._torch.disaggregation.base import CacheKind
from tensorrt_llm._torch.pyexecutor.kv_cache.page_layout import (
    PageLayout,
    layout_from_dict,
    layout_to_dict,
)


@dataclass
class PhysicalPool:
    """Physical storage for a V2 pool's slots.

    ``slot_bytes`` spans all layers and roles coalesced into one slot.
    The slot address is ``base_address + slot * slot_stride_bytes``;
    where each buffer lies within the slot is the page layout's
    (``KVCachePageTable.layout``).
    """

    base_address: int  # uint64
    slot_bytes: int
    num_slots: int
    slot_stride_bytes: Optional[int] = None

    def __post_init__(self) -> None:
        if self.slot_stride_bytes is None:
            self.slot_stride_bytes = self.slot_bytes
        if self.slot_stride_bytes < self.slot_bytes:
            raise ValueError("slot_stride_bytes must be greater than or equal to slot_bytes")

    def to_dict(self) -> dict:
        return {
            "base_address": int(self.base_address),
            "slot_bytes": int(self.slot_bytes),
            "num_slots": int(self.num_slots),
            "slot_stride_bytes": int(self.slot_stride_bytes),
        }

    @staticmethod
    def from_dict(data: dict) -> "PhysicalPool":
        return PhysicalPool(
            base_address=int(data["base_address"]),
            slot_bytes=int(data["slot_bytes"]),
            num_slots=int(data["num_slots"]),
            slot_stride_bytes=(
                int(data["slot_stride_bytes"])
                if data.get("slot_stride_bytes") is not None
                else None
            ),
        )


@dataclass
class PhysicalPoolGroup:
    pools: List[PhysicalPool]

    def to_dict(self) -> dict:
        return {"pools": [p.to_dict() for p in self.pools]}

    @classmethod
    def from_dict(cls, data: dict) -> "PhysicalPoolGroup":
        return cls(pools=[PhysicalPool.from_dict(p) for p in data.get("pools", [])])


@dataclass(frozen=True)
class LocalLayer:
    """Mapping between a local/internal layer id and a collision-free global layer id."""

    local_layer_id: int
    global_layer_id: int

    def to_dict(self) -> dict:
        return {
            "local_layer_id": int(self.local_layer_id),
            "global_layer_id": int(self.global_layer_id),
        }

    @staticmethod
    def from_dict(data: dict) -> "LocalLayer":
        return LocalLayer(
            local_layer_id=int(data["local_layer_id"]),
            global_layer_id=int(data["global_layer_id"]),
        )


@dataclass
class LayerGroup:
    """Base class for one life cycle / layer-group.

    Shared structure:
      - kind: how this group's region IDs are interpreted. PAGED extracts per-block base pointers
        and aligns on window/beam/SWA; STATE extracts per-layer pointers within one slot and has no
        block alignment.
      - pool_group_idx: index into KVCachePageTable.pool_groups
      - local_layers: local ↔ global layer ID mapping
    """

    pool_group_idx: int
    kind: CacheKind = CacheKind.PAGED
    local_layers: List[LocalLayer] = field(default_factory=list)

    def to_dict(self) -> dict:
        raise NotImplementedError

    @classmethod
    def from_dict(cls, data: dict) -> "LayerGroup":
        kind = int(data["kind"])
        if kind == CacheKind.PAGED:
            return AttentionLayerGroup.from_dict(data)
        elif kind == CacheKind.STATE:
            return MambaLayerGroup.from_dict(data)
        raise ValueError(f"Unknown layer group kind: {kind}")


@dataclass
class AttentionLayerGroup(LayerGroup):
    """Layer group for attention KV cache."""

    kv_head_num_per_rank: int = 0
    sliding_window_size: Optional[int] = None

    def to_dict(self) -> dict:
        return {
            "kind": int(self.kind),
            "pool_group_idx": int(self.pool_group_idx),
            "kv_head_num_per_rank": int(self.kv_head_num_per_rank),
            "sliding_window_size": self.sliding_window_size,
            "local_layers": [ll.to_dict() for ll in self.local_layers],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "AttentionLayerGroup":
        return cls(
            pool_group_idx=int(data["pool_group_idx"]),
            local_layers=[LocalLayer.from_dict(x) for x in data.get("local_layers", [])],
            kv_head_num_per_rank=int(data["kv_head_num_per_rank"]),
            sliding_window_size=data.get("sliding_window_size"),
        )


@dataclass
class MambaLayerGroup(LayerGroup):
    """Layer group for recurrent (STATE life cycle) states.

    One slot per request, addressed by the same slot id in every pool of
    the group's pool group; each slot packs its coalesced layers and roles
    together, as the page layout describes.
    """

    def __post_init__(self) -> None:
        self.kind = CacheKind.STATE

    def to_dict(self) -> dict:
        return {
            "kind": int(self.kind),
            "pool_group_idx": int(self.pool_group_idx),
            "local_layers": [ll.to_dict() for ll in self.local_layers],
            "slot_major_layout": True,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "MambaLayerGroup":
        if data.get("slot_major_layout", True) is not True:
            raise ValueError(
                "Python transceiver does not support the layer-major recurrent-state "
                "layout of KV cache manager V1; use a V2 slot-major peer."
            )
        return cls(
            pool_group_idx=int(data["pool_group_idx"]),
            local_layers=[LocalLayer.from_dict(x) for x in data["local_layers"]],
        )


@dataclass
class KVCachePageTable:
    """A rank's pools and layer groups, indexed for transfer, and the page layout (``layout``) the
    manager described them by: where each buffer lies in its pool's slot, from which
    ``layout_mapping`` relates two ranks' pages."""

    tokens_per_block: int
    layer_groups: List[LayerGroup]
    pool_groups: List[PhysicalPoolGroup]  # indexed by LayerGroup.pool_group_idx
    layout: Optional[PageLayout] = None

    def to_dict(self) -> dict:
        return {
            "tokens_per_block": int(self.tokens_per_block),
            "layer_groups": [lg.to_dict() for lg in self.layer_groups],
            "pool_groups": [pg.to_dict() for pg in self.pool_groups],
            "layout": None if self.layout is None else layout_to_dict(self.layout),
        }

    @staticmethod
    def from_dict(data: dict) -> "KVCachePageTable":
        layout = data.get("layout")
        return KVCachePageTable(
            tokens_per_block=int(data["tokens_per_block"]),
            layer_groups=[LayerGroup.from_dict(lg) for lg in data.get("layer_groups", [])],
            pool_groups=[PhysicalPoolGroup.from_dict(pg) for pg in data.get("pool_groups", [])],
            layout=None if layout is None else layout_from_dict(layout),
        )
