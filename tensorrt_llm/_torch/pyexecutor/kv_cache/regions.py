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

Model and attention-backend code declares persistent memory as regions. A
region names its planes, one entry per model layer it covers, and one life
cycle that says how its memory is allocated and when it is handed off to the
shared prefix index. The cache manager lays each region out as a native
``RegionConfig``: the region's buffers get a life cycle of their own, separate
from the life cycles the manager derives from its layer configs, even when the
two have equal parameters.

Only per-request regions that are never handed off are supported: one slot per
living KV cache and beam, released when the cache closes. Such a region may
supply a fixed slot count; its buffers then get a pool group of exactly that
many GPU slots, which the manager's ratio-based sizing leaves alone.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import (
    TYPE_CHECKING,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    Union,
    runtime_checkable,
)

from strenum import StrEnum

from tensorrt_llm.runtime.kv_cache_manager_v2 import (
    AttentionLayerConfig,
    BufferConfig,
    DataRole,
    LayerId,
    RegionAllocationUnit,
    RegionBuffer,
    RegionConfig,
    ReusePolicy,
    SsmLayerConfig,
)

if TYPE_CHECKING:
    from tensorrt_llm.runtime.kv_cache_manager_v2 import KVCacheManager

LayerConfig = Union[AttentionLayerConfig, SsmLayerConfig]


class AllocationUnit(StrEnum):
    """How a region's memory grows over a request's life."""

    TOKEN_BLOCK = "token_block"  # nosec B105 - an allocation unit name, not a password
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
    """How a region's memory is allocated and when it is handed off.

    Attributes:
        allocation: How the region's memory grows.
        hand_off: When committed memory passes to the prefix index.
    """

    allocation: AllocationUnit
    hand_off: HandOff

    def __post_init__(self) -> None:
        if self.hand_off not in _ALLOWED_HAND_OFF[self.allocation]:
            raise ValueError(
                f"A {self.allocation} region cannot hand off at {self.hand_off}; "
                f"allowed: {[str(h) for h in _ALLOWED_HAND_OFF[self.allocation]]}"
            )

    @property
    def reuse_policy(self) -> ReusePolicy:
        """The V2 reuse policy that implements this hand-off."""
        return ReusePolicy.PRIVATE if self.hand_off == HandOff.NEVER else ReusePolicy.REQUIRED


@dataclass(frozen=True)
class Plane:
    """One kind of data a region stores per model layer.

    Attributes:
        role: The buffer role the plane is stored and addressed under.
        page_bytes: Bytes of this plane in one slot: one request's worth for a
            per-request region.
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
        name: Identifies the region; unique per manager.
        life_cycle: How the region's memory is allocated and handed off.
        layers: The model layers the region covers, each with its planes.
        fixed_slot_count: GPU slots of the region's pool group, which the
            manager then keeps at exactly this size. The declaring code computes
            it for every owner it supports, including the headroom
            ``KvCache.resume`` needs below ``max_util_for_resume``. ``None``
            sizes the region by ratio like the manager's other groups.
    """

    name: str
    life_cycle: RegionLifeCycle
    layers: Tuple[RegionLayer, ...]
    fixed_slot_count: Optional[int] = None

    def __post_init__(self) -> None:
        model_layers = [layer.model_layer for layer in self.layers]
        if len(set(model_layers)) != len(model_layers):
            raise ValueError(f"Region {self.name!r} covers a model layer twice: {model_layers}")
        if self.fixed_slot_count is not None and self.fixed_slot_count <= 0:
            raise ValueError(
                f"Region {self.name!r} must have a positive fixed_slot_count, "
                f"got {self.fixed_slot_count}"
            )

    def bytes_per_slot(self) -> int:
        """Bytes of one slot across the region's layers."""
        return sum(plane.page_bytes for layer in self.layers for plane in layer.planes)

    def fixed_bytes(self) -> int:
        """GPU bytes of the region's fixed pool; 0 for a region sized by ratio."""
        if self.fixed_slot_count is None:
            return 0
        return self.fixed_slot_count * self.bytes_per_slot()


@runtime_checkable
class RegionProvider(Protocol):
    """Model- or backend-side code that declares persistent-state regions."""

    def declare_regions(self) -> Sequence[Region]:
        """Return this provider's regions.

        Called once while the manager is built, after its per-layer views
        exist and before it costs its memory budget.
        """
        ...


@dataclass(frozen=True)
class RegionLayout:
    """The declared regions and which region owns each of their buffers.

    Attributes:
        regions: The declared regions, in declaration order.
        owners: ``(layer id, role)`` of every region buffer to the name of its
            region. Empty until the regions are laid out.
    """

    regions: Tuple[Region, ...] = ()
    owners: Mapping[Tuple[int, DataRole], str] = field(default_factory=dict)

    def __contains__(self, name: object) -> bool:
        """Whether a region named ``name`` is declared."""
        return any(region.name == name for region in self.regions)

    def fixed_bytes(self) -> int:
        """GPU bytes of the declared regions' fixed pools."""
        return sum(region.fixed_bytes() for region in self.regions)

    def region(self, name: str) -> Region:
        """Return the region declared under ``name``."""
        for region in self.regions:
            if region.name == name:
                return region
        raise KeyError(f"No region named {name!r}")

    def owner(self, layer_id: int, role: DataRole) -> Optional[Region]:
        """Return the region storing ``role`` on ``layer_id``; None for a layer-config buffer."""
        name = self.owners.get((int(layer_id), role))
        return None if name is None else self.region(name)

    def layer_ids(self, region: str) -> Tuple[LayerId, ...]:
        """Storage layer ids holding buffers of ``region``, in ascending order."""
        layer_ids = {layer_id for (layer_id, _), name in self.owners.items() if name == region}
        return tuple(LayerId(layer_id) for layer_id in sorted(layer_ids))

    def roles(self, region: str, layer_id: int) -> Tuple[DataRole, ...]:
        """Buffer roles ``region`` stores on ``layer_id``, in declaration order."""
        return tuple(
            role
            for (owner_layer, role), name in self.owners.items()
            if owner_layer == int(layer_id) and name == region
        )

    def group_ids(self, region: str, cache: "KVCacheManager") -> Tuple[int, ...]:
        """The V2 layer group of ``region`` in ``cache``; empty for an undeclared region."""
        if region not in self:
            return ()
        return (int(cache.get_region_group_id(region)),)

    def renumbered(self, retained_layer_ids: Sequence[int]) -> "RegionLayout":
        """Return the layout after the manager compacts its layer list.

        Args:
            retained_layer_ids: For each new storage id, the old id it came from.
        """
        old_to_new = {old: new for new, old in enumerate(retained_layer_ids)}
        renumber: Dict[int, int] = {}
        for region in self.regions:
            for layer in region.layers:
                new = old_to_new.get(layer.model_layer)
                if new is None:
                    raise ValueError(
                        f"Region {region.name!r} covers layer {layer.model_layer}, "
                        "which the manager dropped"
                    )
                renumber[layer.model_layer] = new
        regions = tuple(
            Region(
                name=region.name,
                life_cycle=region.life_cycle,
                layers=tuple(
                    RegionLayer(renumber[layer.model_layer], layer.planes)
                    for layer in region.layers
                ),
                fixed_slot_count=region.fixed_slot_count,
            )
            for region in self.regions
        )
        owners = {(renumber[old], role): name for (old, role), name in self.owners.items()}
        return RegionLayout(regions=regions, owners=owners)


def lay_out_regions(
    regions: Iterable[Region], model_layers: Sequence[LayerConfig]
) -> Tuple[List[RegionConfig], RegionLayout]:
    """Lay regions out as native region configs beside the manager's layer configs.

    Args:
        regions: The regions to store, in declaration order, which is also the
            order of their life cycles after the layer life cycles.
        model_layers: The manager's config for each rank-local layer, indexed by
            layer id.

    Returns:
        One ``RegionConfig`` per region and the ownership of their buffers.

    Raises:
        ValueError: If region names repeat, a region covers a layer this rank
            does not have, a region stores a role the layer's config already
            stores, or a region's life cycle has no native region yet.
    """
    regions = tuple(regions)
    names = [region.name for region in regions]
    if len(set(names)) != len(names):
        raise ValueError(f"Region names must be unique, got {names}")
    layer_roles = {
        int(layer.layer_id): {buffer.role for buffer in layer.buffers} for layer in model_layers
    }
    configs: List[RegionConfig] = []
    owners: Dict[Tuple[int, DataRole], str] = {}
    for region in regions:
        life_cycle = region.life_cycle
        if (life_cycle.allocation, life_cycle.hand_off) != (AllocationUnit.REQUEST, HandOff.NEVER):
            raise ValueError(
                f"Region {region.name!r} is a {life_cycle.allocation} region handed off at "
                f"{life_cycle.hand_off}; only request regions that are never handed off are "
                "supported"
            )
        buffers = []
        for region_layer in region.layers:
            layer_id = region_layer.model_layer
            if layer_id not in layer_roles:
                raise ValueError(
                    f"Region {region.name!r} covers layer {layer_id}, which this rank does not have"
                )
            for plane in region_layer.planes:
                if plane.role in layer_roles[layer_id] or (layer_id, plane.role) in owners:
                    raise ValueError(
                        f"Region {region.name!r} stores {plane.role} on layer {layer_id}, "
                        "which another region or the layer's config already stores"
                    )
                owners[layer_id, plane.role] = region.name
                buffers.append(
                    RegionBuffer(
                        layer_id=LayerId(layer_id),
                        buffer=BufferConfig(
                            role=plane.role,
                            size=plane.page_bytes,
                            coalescing_group=plane.coalescing_group,
                        ),
                    )
                )
        configs.append(
            RegionConfig(
                name=region.name,
                buffers=buffers,
                allocation_unit=RegionAllocationUnit.REQUEST,
                reuse_policy=life_cycle.reuse_policy,
                fixed_slot_count=region.fixed_slot_count,
            )
        )
    return configs, RegionLayout(regions=regions, owners=owners)


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
    "lay_out_regions",
]
