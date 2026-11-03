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
"""Persistent-state regions and their layout on KV cache manager V2 storage.

Model and attention-backend code declares what persistent memory exists as
regions. A region names its planes, one entry per model layer it covers, and
one life cycle that says when its memory is allocated, released early and
handed off to the shared prefix index. The cache manager lays every region out
on the V2 storage primitives and records which region owns each storage layer;
it never needs to know what a region holds.

V2 storage offers two allocation shapes, both backed by the same paged pools:

* a span of slots, one slot per token block, of which a window may stay live
  (an attention life cycle), and
* one slot per request, earlier states of which leave only as snapshots in the
  prefix index (an SSM life cycle).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Dict, Iterable, Optional, Protocol, Sequence, Tuple, Union, runtime_checkable

from tensorrt_llm.runtime.kv_cache_manager_v2 import (
    AttentionLayerConfig,
    BufferConfig,
    DataRole,
    LayerId,
    ReusePolicy,
    SsmLayerConfig,
)

LayerConfig = Union[AttentionLayerConfig, SsmLayerConfig]


class AllocationUnit(StrEnum):
    """How a region's memory grows over a request's life."""

    TOKEN_BLOCK = "token_block"
    """One slot per token block: the region grows with the sequence."""

    REQUEST = "request"
    """One slot per request, independent of sequence length."""


class HandOff(StrEnum):
    """When a region's committed memory passes to the shared prefix index."""

    NEVER = "never"
    """Request-private: the memory is never published for reuse."""

    BLOCK_BOUNDARIES = "block_boundaries"
    """Each full token block is published when the request commits it."""

    DECLARED_POINTS = "declared_points"
    """The live slot is snapshotted into the index at points declared per request."""


_ALLOWED_HAND_OFF = {
    AllocationUnit.TOKEN_BLOCK: (HandOff.NEVER, HandOff.BLOCK_BOUNDARIES),
    AllocationUnit.REQUEST: (HandOff.NEVER, HandOff.DECLARED_POINTS),
}


@dataclass(frozen=True)
class RegionLifeCycle:
    """When a region's memory is allocated, released early, and handed off.

    Early release follows from the allocation unit: a token-block region with a
    ``window`` releases the blocks that fall out of it, one without a window
    releases nothing before the request ends, and a per-request region keeps
    only its latest state live.

    Attributes:
        allocation: How the region's memory grows.
        hand_off: When committed memory passes to the prefix index.
        window: Token-block regions only; the number of most recent tokens whose
            blocks stay allocated. ``None`` keeps every block.
    """

    allocation: AllocationUnit
    hand_off: HandOff
    window: Optional[int] = None

    def __post_init__(self) -> None:
        if self.hand_off not in _ALLOWED_HAND_OFF[self.allocation]:
            raise ValueError(
                f"A {self.allocation} region cannot hand off at {self.hand_off}; "
                f"allowed: {[str(h) for h in _ALLOWED_HAND_OFF[self.allocation]]}"
            )
        if self.window is not None:
            if self.allocation != AllocationUnit.TOKEN_BLOCK:
                raise ValueError("Only token-block regions can declare a window")
            if self.window <= 0:
                raise ValueError(f"window must be positive, got {self.window}")

    @property
    def reuse_policy(self) -> ReusePolicy:
        """The V2 reuse policy that implements this hand-off."""
        return ReusePolicy.PRIVATE if self.hand_off == HandOff.NEVER else ReusePolicy.REQUIRED


@dataclass(frozen=True)
class Plane:
    """One kind of data a region stores per model layer.

    Attributes:
        role: The buffer role the plane is stored and addressed under.
        page_bytes: Bytes of this plane in one slot: one token block's worth for
            a token-block region, one request's worth for a per-request region.
        coalescing_group: Planes may share a physical pool only when their
            groups and sizes match.
    """

    role: DataRole
    page_bytes: int
    coalescing_group: int = 0

    def __post_init__(self) -> None:
        if self.page_bytes <= 0:
            raise ValueError(
                f"Plane {self.role} must have a positive size; omit a plane the "
                f"region does not store instead of declaring it empty"
            )


@dataclass(frozen=True)
class RegionLayer:
    """The planes a region stores for one rank-local model layer.

    Attributes:
        model_layer: Rank-local model layer index, as used by the manager's
            per-layer views.
        planes: The planes stored for this layer, in declaration order.
    """

    model_layer: int
    planes: Tuple[Plane, ...]

    def __post_init__(self) -> None:
        if not self.planes:
            raise ValueError(f"Region layer {self.model_layer} declares no planes")
        roles = [plane.role for plane in self.planes]
        if len(set(roles)) != len(roles):
            raise ValueError(f"Region layer {self.model_layer} repeats a plane role: {roles}")


@dataclass(frozen=True)
class Region:
    """A unit of persistent state with one life cycle.

    Attributes:
        name: Identifies the region in the ownership map; unique per manager.
        life_cycle: When the region's memory is allocated, released and handed off.
        layers: The model layers the region covers, each with its planes.
    """

    name: str
    life_cycle: RegionLifeCycle
    layers: Tuple[RegionLayer, ...]

    def __post_init__(self) -> None:
        model_layers = [layer.model_layer for layer in self.layers]
        if len(set(model_layers)) != len(model_layers):
            raise ValueError(f"Region {self.name!r} covers a model layer twice: {model_layers}")

    def layer_config(self, layer_id: int, layer: RegionLayer) -> LayerConfig:
        """Return the V2 layer config storing ``layer`` under ``layer_id``."""
        buffers = [
            BufferConfig(
                role=plane.role,
                size=plane.page_bytes,
                coalescing_group=plane.coalescing_group,
            )
            for plane in layer.planes
        ]
        life_cycle = self.life_cycle
        if life_cycle.allocation == AllocationUnit.REQUEST:
            return SsmLayerConfig(
                layer_id=LayerId(layer_id),
                buffers=buffers,
                reuse_policy=life_cycle.reuse_policy,
            )
        return AttentionLayerConfig(
            layer_id=LayerId(layer_id),
            buffers=buffers,
            sliding_window_size=life_cycle.window,
            num_sink_tokens=None,
            reuse_policy=life_cycle.reuse_policy,
        )


@runtime_checkable
class RegionProvider(Protocol):
    """Model- or backend-side code that declares persistent-state regions."""

    def declare_regions(self) -> Sequence[Region]:
        """Return this provider's regions; called once while the manager is built."""
        ...


@dataclass(frozen=True)
class StorageOwner:
    """The region a storage layer belongs to and the model layer it serves."""

    region: str
    model_layer: int


@dataclass(frozen=True)
class RegionLayout:
    """Which region owns each storage layer of the manager's V2 layer list.

    Storage layer ids below ``num_model_layers`` coincide with rank-local model
    layer indices. A model layer covered by more than one region is stored under
    its own id only for the first region that declares it; each later region
    gets a storage layer appended after the model layers, so storage ids at or
    above ``num_model_layers`` never index per-model-layer tables.

    Attributes:
        num_model_layers: Number of rank-local model layers.
        owners: Storage layer id to owning region, for region storage only.
        regions: The laid-out regions, in declaration order.
    """

    num_model_layers: int
    owners: Dict[int, StorageOwner] = field(default_factory=dict)
    regions: Tuple[Region, ...] = ()

    def owner(self, layer_id: int) -> Optional[StorageOwner]:
        """Return the region storing ``layer_id``, or None for a manager-built layer."""
        return self.owners.get(int(layer_id))

    def is_model_layer(self, layer_id: int) -> bool:
        """Whether ``layer_id`` also indexes the manager's per-model-layer tables."""
        return 0 <= int(layer_id) < self.num_model_layers

    def region(self, name: str) -> Region:
        """Return the region declared under ``name``."""
        for region in self.regions:
            if region.name == name:
                return region
        raise KeyError(f"No region named {name!r}")

    def layer_ids(self, region: str) -> Tuple[LayerId, ...]:
        """Storage layer ids of ``region``, in ascending order."""
        return tuple(
            LayerId(layer_id)
            for layer_id, owner in sorted(self.owners.items())
            if owner.region == region
        )

    def model_layer_of(self, layer_id: int) -> Optional[int]:
        """Model layer served by ``layer_id``: itself for a model layer id, else its owner's."""
        owner = self.owner(layer_id)
        if owner is not None:
            return owner.model_layer
        return int(layer_id) if self.is_model_layer(layer_id) else None

    def renumbered(self, retained_layer_ids: Sequence[int]) -> "RegionLayout":
        """Return the layout after the manager compacts its layer list.

        Args:
            retained_layer_ids: For each new storage id, the old id it came from.
        """
        old_to_new = {old: new for new, old in enumerate(retained_layer_ids)}
        num_model_layers = sum(1 for old in retained_layer_ids if old < self.num_model_layers)
        owners = {}
        for old, owner in self.owners.items():
            if old not in old_to_new:
                continue
            model_layer = old_to_new.get(owner.model_layer)
            if model_layer is None:
                raise ValueError(
                    f"Region {owner.region!r} serves model layer {owner.model_layer}, "
                    "which the manager dropped"
                )
            owners[old_to_new[old]] = StorageOwner(owner.region, model_layer)
        return RegionLayout(num_model_layers=num_model_layers, owners=owners, regions=self.regions)


def lay_out_regions(
    regions: Iterable[Region], model_layers: Sequence[LayerConfig]
) -> Tuple[Tuple[LayerConfig, ...], RegionLayout]:
    """Lay regions out over the manager's per-model-layer configs.

    Args:
        regions: The regions to store, in declaration order. Their order fixes
            which region keeps a shared model layer's id and the order of the
            appended storage layers.
        model_layers: The manager's config for each rank-local model layer,
            indexed by layer id.

    Returns:
        The V2 layer list that replaces ``model_layers`` -- region storage in
        place of the layers it covers, then the appended region storage -- and
        the ownership map of that list.
    """
    regions = tuple(regions)
    names = [region.name for region in regions]
    if len(set(names)) != len(names):
        raise ValueError(f"Region names must be unique, got {names}")
    num_model_layers = len(model_layers)
    layers = list(model_layers)
    appended = []
    owners: Dict[int, StorageOwner] = {}
    claimed = set()
    for region in regions:
        for region_layer in region.layers:
            model_layer = region_layer.model_layer
            if not 0 <= model_layer < num_model_layers:
                raise ValueError(
                    f"Region {region.name!r} covers model layer {model_layer}, but this "
                    f"rank has {num_model_layers} model layers"
                )
            if model_layer not in claimed:
                claimed.add(model_layer)
                layer_id = model_layer
                layers[layer_id] = region.layer_config(layer_id, region_layer)
            else:
                layer_id = num_model_layers + len(appended)
                appended.append(region.layer_config(layer_id, region_layer))
            owners[layer_id] = StorageOwner(region.name, model_layer)
    layout = RegionLayout(num_model_layers=num_model_layers, owners=owners, regions=regions)
    return tuple(layers + appended), layout


__all__ = [
    "AllocationUnit",
    "HandOff",
    "LayerConfig",
    "Plane",
    "Region",
    "RegionLayer",
    "RegionLayout",
    "RegionLifeCycle",
    "RegionProvider",
    "StorageOwner",
    "lay_out_regions",
]
