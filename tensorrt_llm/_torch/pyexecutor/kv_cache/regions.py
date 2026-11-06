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
on the V2 storage primitives and records which region owns each buffer; it
never needs to know what a region holds.

V2 storage offers two allocation shapes, both backed by the same paged pools:

* a span of slots, one slot per token block, of which a window may stay live
  (an attention life cycle), and
* one slot per request, earlier states of which leave only as snapshots in the
  prefix index (an SSM life cycle).

A life cycle binds to a buffer, not to a layer. Every region covering a model
layer stores its planes as buffers of that layer's one V2 layer config, each
buffer with its region's reuse policy, so a layer covered by regions of several
life cycles spans several V2 layer groups while keeping its id.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import (
    TYPE_CHECKING,
    Dict,
    FrozenSet,
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

from tensorrt_llm._torch.disaggregation.resource.page import MapperKind, RoleLayout
from tensorrt_llm.runtime.kv_cache_manager_v2 import (
    AttentionLayerConfig,
    BufferConfig,
    DataRole,
    LayerId,
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
        bytes_per_token: What one token costs in this layer when the manager
            converts a memory budget into token capacity. Declared by
            token-block regions; the manager does not read it yet.
    """

    model_layer: int
    planes: Tuple[Plane, ...]
    bytes_per_token: int = 0

    def __post_init__(self) -> None:
        if not self.planes:
            raise ValueError(f"Region layer {self.model_layer} declares no planes")
        roles = [plane.role for plane in self.planes]
        if len(set(roles)) != len(roles):
            raise ValueError(f"Region layer {self.model_layer} repeats a plane role: {roles}")


@dataclass(frozen=True)
class RegionTransfer:
    """How a region's planes move between disaggregated workers.

    Attributes:
        mapper_kinds: Layout each transferred plane role is written in, which
            decides how it is resharded across tensor-parallel ranks.
        role_layouts: Resharding geometry of the roles whose mapper kind needs it.
        rebuilt_roles: Roles never transferred; the receiving worker rebuilds
            them from the transferred planes or starts them afresh.
    """

    mapper_kinds: Mapping[DataRole, MapperKind] = field(default_factory=dict)
    role_layouts: Mapping[DataRole, RoleLayout] = field(default_factory=dict)
    rebuilt_roles: FrozenSet[DataRole] = frozenset()


@dataclass(frozen=True)
class Region:
    """A unit of persistent state with one life cycle.

    Attributes:
        name: Identifies the region in the ownership map; unique per manager.
        life_cycle: When the region's memory is allocated, released and handed off.
        layers: The model layers the region covers, each with its planes.
        transfer: How the region's planes move between disaggregated workers.
        resumes_at_block_boundaries: Another request can resume from this
            region's committed memory only at a token-block boundary, so every
            declared hand-off point of the manager is moved onto one.
            Declared by token-block regions; the manager does not read it yet.
        positioned_by_history: The region's live window is placed by the
            request's history length, so the history length must follow every
            completed context chunk. Declared by token-block regions; the
            manager does not read it yet.
    """

    name: str
    life_cycle: RegionLifeCycle
    layers: Tuple[RegionLayer, ...]
    transfer: RegionTransfer = field(default_factory=RegionTransfer)
    resumes_at_block_boundaries: bool = False
    positioned_by_history: bool = False

    def __post_init__(self) -> None:
        model_layers = [layer.model_layer for layer in self.layers]
        if len(set(model_layers)) != len(model_layers):
            raise ValueError(f"Region {self.name!r} covers a model layer twice: {model_layers}")


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
    """Which region owns each buffer of the manager's V2 layer list.

    Region storage takes the ids of the model layers it covers, so every
    storage id also indexes the manager's per-model-layer tables. A layer
    covered by several regions holds the buffers of all of them; ownership is
    therefore recorded per buffer, ``(layer id, role)``.

    Attributes:
        regions: The declared regions, in declaration order.
        owners: The ``(layer id, role)`` of every buffer that region storage
            holds, to the name of its region, layer by layer in buffer order.
            Empty until the regions are laid out.
    """

    regions: Tuple[Region, ...] = ()
    owners: Mapping[Tuple[int, DataRole], str] = field(default_factory=dict)

    def __contains__(self, name: object) -> bool:
        """Whether a region named ``name`` is declared."""
        return any(region.name == name for region in self.regions)

    def region(self, name: str) -> Region:
        """Return the region declared under ``name``."""
        for region in self.regions:
            if region.name == name:
                return region
        raise KeyError(f"No region named {name!r}")

    def owner(self, layer_id: int, role: DataRole) -> Optional[Region]:
        """Return the region storing ``role`` on ``layer_id``; None for a manager-built buffer."""
        name = self.owners.get((int(layer_id), role))
        return None if name is None else self.region(name)

    def layer_ids(self, region: str) -> Tuple[LayerId, ...]:
        """Storage layer ids holding buffers of ``region``, in ascending order."""
        layer_ids = {layer_id for (layer_id, _), name in self.owners.items() if name == region}
        return tuple(LayerId(layer_id) for layer_id in sorted(layer_ids))

    def roles(self, region: str, layer_id: int) -> Tuple[DataRole, ...]:
        """Buffer roles ``region`` stores on ``layer_id``, in buffer order."""
        return tuple(
            role
            for (owner_layer, role), name in self.owners.items()
            if owner_layer == int(layer_id) and name == region
        )

    def group_ids(self, region: str, cache: "KVCacheManager") -> Tuple[int, ...]:
        """V2 layer group ids of ``region``'s buffers in ``cache``, ascending.

        Each group is one life cycle; a region whose buffers share a life
        cycle across its layers has a single group.
        """
        return tuple(
            sorted(
                {
                    int(cache.get_buffer_group_id(LayerId(layer_id), role))
                    for (layer_id, role), name in self.owners.items()
                    if name == region
                }
            )
        )

    def renumbered(self, retained_layer_ids: Sequence[int]) -> "RegionLayout":
        """Return the layout after the manager compacts its layer list.

        Args:
            retained_layer_ids: For each new storage id, the old id it came from.
        """
        old_to_new = {old: new for new, old in enumerate(retained_layer_ids)}
        owners = {}
        for (old, role), name in self.owners.items():
            new = old_to_new.get(old)
            if new is None:
                raise ValueError(
                    f"Region {name!r} stores {role} on layer {old}, which the manager dropped"
                )
            owners[new, role] = name
        return RegionLayout(regions=self.regions, owners=owners)


def _layer_config(layer_id: int, entries: Sequence[Tuple[Region, RegionLayer]]) -> LayerConfig:
    """Merge the planes every region stores on one model layer into its V2 layer config.

    The layer takes the life cycle of the first region that covers it, whose
    buffers inherit the layer's reuse policy; a buffer of a region whose policy
    differs overrides it, which places the buffer in that life cycle's group.
    """
    first = entries[0][0].life_cycle
    for region, _ in entries[1:]:
        life_cycle = region.life_cycle
        if life_cycle.allocation != first.allocation:
            raise ValueError(
                f"Regions on model layer {layer_id} mix {first.allocation} and "
                f"{life_cycle.allocation} allocation: one layer is one V2 layer config"
            )
        if life_cycle.window != first.window:
            raise ValueError(
                f"Token-block regions on model layer {layer_id} declare different windows "
                f"({first.window} and {life_cycle.window}): a window is a layer property"
            )
    layer_policy = first.reuse_policy
    buffers: List[BufferConfig] = []
    roles: Dict[DataRole, str] = {}
    for region, region_layer in entries:
        policy = region.life_cycle.reuse_policy
        for plane in region_layer.planes:
            if plane.role in roles:
                raise ValueError(
                    f"Regions {roles[plane.role]!r} and {region.name!r} both store "
                    f"{plane.role} on model layer {layer_id}"
                )
            roles[plane.role] = region.name
            buffers.append(
                BufferConfig(
                    role=plane.role,
                    size=plane.page_bytes,
                    coalescing_group=plane.coalescing_group,
                    reuse_policy=None if policy == layer_policy else policy,
                )
            )
    if first.allocation == AllocationUnit.REQUEST:
        return SsmLayerConfig(
            layer_id=LayerId(layer_id), buffers=buffers, reuse_policy=layer_policy
        )
    return AttentionLayerConfig(
        layer_id=LayerId(layer_id),
        buffers=buffers,
        sliding_window_size=first.window,
        num_sink_tokens=None,
        reuse_policy=layer_policy,
    )


def lay_out_regions(
    regions: Iterable[Region], model_layers: Sequence[LayerConfig]
) -> Tuple[Tuple[LayerConfig, ...], RegionLayout]:
    """Lay regions out over the manager's per-model-layer configs.

    Args:
        regions: The regions to store, in declaration order. The order fixes
            each covered layer's buffer order and its own reuse policy (the
            first covering region's).
        model_layers: The manager's config for each rank-local model layer,
            indexed by layer id.

    Returns:
        The V2 layer list that replaces ``model_layers`` -- a covered layer's
        config holds the planes of every region covering it, an uncovered
        layer keeps the manager's config -- and the ownership of its buffers.

    Raises:
        ValueError: If region names repeat, a region covers a layer this rank
            does not have, two regions store one role on a layer, or the
            regions on one layer differ in allocation unit or window.
    """
    regions = tuple(regions)
    names = [region.name for region in regions]
    if len(set(names)) != len(names):
        raise ValueError(f"Region names must be unique, got {names}")
    num_model_layers = len(model_layers)
    covering: Dict[int, List[Tuple[Region, RegionLayer]]] = {}
    for region in regions:
        for region_layer in region.layers:
            model_layer = region_layer.model_layer
            if not 0 <= model_layer < num_model_layers:
                raise ValueError(
                    f"Region {region.name!r} covers model layer {model_layer}, but this "
                    f"rank has {num_model_layers} model layers"
                )
            covering.setdefault(model_layer, []).append((region, region_layer))
    layers = list(model_layers)
    owners: Dict[Tuple[int, DataRole], str] = {}
    for layer_id in sorted(covering):
        entries = covering[layer_id]
        layers[layer_id] = _layer_config(layer_id, entries)
        for region, region_layer in entries:
            for plane in region_layer.planes:
                owners[layer_id, plane.role] = region.name
    return tuple(layers), RegionLayout(regions=regions, owners=owners)


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
    "RegionTransfer",
    "lay_out_regions",
]
