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
"""KV cache layout description handed to a KV connector under KVCacheManagerV2.

V2 allocates *pool groups*, each holding ``num_slots`` slots. A slot holds the
coalesced buffers of one *layer group* (a life cycle), and a buffer is keyed by
``BufferId(layer_id, role)``. A connector therefore cannot be handed a single
pool tensor: there is one slot address space per pool, and one index space per
layer group.

Instead it is handed :class:`KvCacheLayout` -- a description of the byte ranges
that repeat per page slot. The addressing contract is V2's own, taken verbatim
from ``AggregatedPageDesc``::

    (base + stride * i + Range(0, size) for i in aggregated_page_indices)

where ``i`` comes from ``_KVCache.get_aggregated_page_indices(layer_group_id)``.

Because ranges are described rather than implied, this covers MLA (a pool simply
has no VALUE buffer), sliding-window attention and hybrid models (one layer group
per window size) without any of them being special cases.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, Iterable, Iterator, List, Optional, Tuple

import torch

from tensorrt_llm._utils import TensorWrapper, binding_to_torch_dtype, convert_to_torch_tensor

if TYPE_CHECKING:
    from ..kv_cache_manager_v2 import KVCacheManagerV2

__all__ = [
    "KvCacheBufferRef",
    "KvCacheLayerGroupLayout",
    "KvCacheLayout",
    "KvCacheRegion",
    "build_kv_cache_layout_v2",
    "valid_page_slots",
]


def valid_page_slots(page_indices: Iterable[int]) -> Iterator[Tuple[int, int]]:
    """Yield ``(block_ordinal, page_slot)`` for the entries that address a page.

    A page-index list holds one entry per block ordinal, so entry ``i`` describes
    tokens ``[i * tokens_per_block, (i + 1) * tokens_per_block)``. A block with no
    page in the layer group -- past the end of a sliding window, or never
    allocated there -- holds ``BAD_PAGE_INDEX`` in place so that alignment
    survives, which makes the list unsafe to index with directly.

    Iterating here keeps the ordinal, which a caller needs to find the token range
    a page covers, and drops the entries that address no page, which no transfer
    may be built against.
    """
    for ordinal, slot in enumerate(page_indices):
        if slot >= 0:
            yield ordinal, slot


@dataclass(frozen=True)
class KvCacheBufferRef:
    """One ``(layer, role)`` buffer covered by a region, in memory order."""

    #: Global model layer index -- the same index space the per-layer connector
    #: hooks (``wait_for_layer_load`` / ``save_kv_layer``) receive.
    layer_id: int
    #: Native role name as the cache manager spells it, e.g. "key" / "value".
    #: Deliberately not an enum: roles are an open vocabulary on the manager
    #: side, and a connector should not need updating when one is added.
    role: str
    #: Page expansion factor for heterogeneous tokens-per-block layers.
    expansion: int = 1


@dataclass(frozen=True)
class KvCacheRegion:
    """A contiguous byte range that repeats once per page slot.

    The data for page slot ``i`` lives at ``base + stride * i``, for ``size``
    bytes. ``size`` is not necessarily equal to ``stride``: a region covers one
    run of adjacent buffers within a slot, and a slot may hold several runs.
    """

    base: int
    size: int
    stride: int
    num_slots: int
    buffers: Tuple[KvCacheBufferRef, ...]

    def address_of(self, slot_id: int) -> int:
        """Device address of this region for ``slot_id``."""
        if not 0 <= slot_id < self.num_slots:
            raise IndexError(f"slot_id {slot_id} out of range [0, {self.num_slots})")
        return self.base + self.stride * slot_id

    def as_tensor(self, dtype: torch.dtype = torch.uint8) -> torch.Tensor:
        """A strided ``[num_slots, size // itemsize]`` view; row ``i`` is slot ``i``.

        Defaults to ``uint8``. A region may span several roles whose element
        types differ, and a connector that only moves bytes should not have to
        care; callers that want a typed view can pass ``dtype`` explicitly.
        """
        itemsize = torch.tensor([], dtype=dtype).element_size()
        if self.size % itemsize or self.stride % itemsize:
            raise ValueError(
                f"region size {self.size} and stride {self.stride} must both be "
                f"multiples of {dtype} itemsize {itemsize}"
            )
        return convert_to_torch_tensor(
            TensorWrapper(
                self.base,
                dtype,
                shape=(self.num_slots, self.size // itemsize),
                strides=(self.stride // itemsize, 1),
            )
        )

    def slot_tensor(self, slot_id: int, dtype: torch.dtype = torch.uint8) -> torch.Tensor:
        """A ``[size // itemsize]`` view of one page slot.

        The guarded form of ``as_tensor(dtype)[slot_id]``. A page index that
        addresses no page is ``BAD_PAGE_INDEX`` (-1), which the strided view
        accepts as a subscript and resolves to its last row -- a live page
        belonging to some other request. Addressing a slot through here raises
        ``IndexError`` instead, so a page index reaches device memory only once
        it has been checked. Pair it with ``valid_page_slots`` to filter a whole
        list at the source.
        """
        itemsize = torch.tensor([], dtype=dtype).element_size()
        if self.size % itemsize:
            raise ValueError(
                f"region size {self.size} must be a multiple of {dtype} itemsize {itemsize}"
            )
        return convert_to_torch_tensor(
            TensorWrapper(
                self.address_of(slot_id),
                dtype,
                shape=(self.size // itemsize,),
                strides=(1,),
            )
        )


@dataclass(frozen=True)
class KvCacheLayerGroupLayout:
    """One layer group -- the unit that page indices are scoped to."""

    layer_group_id: int
    #: Global model layer indices belonging to this group.
    layer_ids: Tuple[int, ...]
    #: Attention window for this group, or None for full attention.
    window_size: Optional[int]
    regions: Tuple[KvCacheRegion, ...]

    @property
    def bytes_per_page(self) -> int:
        """Total bytes this group occupies for a single page slot."""
        return sum(region.size for region in self.regions)


@dataclass(frozen=True)
class KvCacheLayout:
    """What a connector is handed in place of a single KV cache pool tensor."""

    tokens_per_block: int
    groups: Tuple[KvCacheLayerGroupLayout, ...]
    #: Element type of the KV data, for typed views over a region.
    dtype: torch.dtype = torch.uint8

    def group(self, layer_group_id: int) -> KvCacheLayerGroupLayout:
        for group in self.groups:
            if group.layer_group_id == layer_group_id:
                return group
        raise KeyError(f"no layer group {layer_group_id} in layout")

    def group_of_layer(self, layer_id: int) -> KvCacheLayerGroupLayout:
        """The layer group owning a global model layer index."""
        for group in self.groups:
            if layer_id in group.layer_ids:
                return group
        raise KeyError(f"layer {layer_id} is not covered by this layout")

    def as_single_pool_tensor(self) -> Optional[torch.Tensor]:
        """A ``[num_slots, num_layers, kv_factor, block_size]`` view, or None.

        This is the shape a single-pool KV cache manager hands
        ``register_kv_caches``, so a connector written against that signature
        keeps working when the same cache is described as a layout. Returns None
        when the cache cannot be described that way -- several layer groups,
        several regions (block scales, or layers of differing size), or a page
        expansion factor that breaks the uniform grid.

        Dimension 1 is the region's buffers in memory order, and V2 lays those
        out layer-major and ascending by construction: the storage config walks
        ``config.layers`` in order and appends each layer's buffers to the
        coalesced buffer, then assigns offsets by walking that list. So the
        dimension is indexed by layer exactly as V1's single pool is, where
        ``getPoolLayerIdx(i) == i``. The order is not re-derived here; if that
        construction ever changes, this is what changes with it.
        """
        if len(self.groups) != 1 or len(self.groups[0].regions) != 1:
            return None
        group = self.groups[0]
        region = group.regions[0]
        num_layers = len(group.layer_ids)
        if not num_layers or len(region.buffers) % num_layers:
            return None
        kv_factor = len(region.buffers) // num_layers
        if any(buffer.expansion != 1 for buffer in region.buffers):
            return None
        itemsize = torch.tensor([], dtype=self.dtype).element_size()
        if region.size % (itemsize * len(region.buffers)):
            return None
        block_size = region.size // itemsize // len(region.buffers)
        return region.as_tensor(self.dtype).unflatten(1, (num_layers, kv_factor, block_size))


def _global_layer_ids(manager: "KVCacheManagerV2", local_layer_ids) -> List[int]:
    """Map V2-internal layer ids to global model layer indices.

    ``pp_layers`` is the local-to-global table the manager already keeps. Models
    that map several internal layers onto one model layer (the sparse-attention
    virtual-layer path) have no single global index per internal layer, so they
    are rejected rather than silently mislabelled.
    """
    if hasattr(manager, "_layer_attn_to_layer_id"):
        raise NotImplementedError(
            "KV connector layout is not supported for managers with virtual "
            "attention layers (sparse attention): an internal layer does not map "
            "to a single model layer index."
        )
    pp_layers = manager.pp_layers
    return [int(pp_layers[int(lid)]) for lid in local_layer_ids]


def _window_size(init_config, local_layer_id: int) -> Optional[int]:
    layers = init_config.layers
    if local_layer_id >= len(layers):
        raise ValueError(f"no layer config for internal layer {local_layer_id}")
    window = getattr(layers[local_layer_id], "window_size", None)
    return None if window is None else int(window)


def build_kv_cache_layout_v2(manager: "KVCacheManagerV2") -> KvCacheLayout:
    """Describe a ``KVCacheManagerV2``'s page regions for a KV connector.

    Every region is reported with ``desc.base`` verbatim and no tier predicate,
    so the addresses are device addresses only because tiers below GPU are
    rejected while a connector is attached
    (``PyExecutor._reject_non_gpu_cache_tiers``). Reads only the manager's
    public layout API -- ``layer_grouping``, ``all_buffer_ids``,
    ``get_aggregated_pages``, ``pool_group_descs`` -- so it assumes nothing
    about dimension order, kv factor, or pool count.
    """
    impl = manager.impl
    init_config = impl.init_config

    # A pool group's slot count applies to every layer group drawn from it.
    # Note LayerGroupId and PoolGroupIndex are distinct index spaces; the
    # variants of a pool group name the layer groups it backs.
    slots_by_group: Dict[int, int] = {}
    for pool_group in impl.pool_group_descs:
        for variant in pool_group.slot_desc.variants:
            slots_by_group[int(variant.layer_group_id)] = int(pool_group.num_slots)

    buffers_by_layer: Dict[int, List] = {}
    for buffer_id in impl.all_buffer_ids:
        buffers_by_layer.setdefault(int(buffer_id.layer_id), []).append(buffer_id)

    groups: List[KvCacheLayerGroupLayout] = []
    for layer_group_id, local_layer_ids in enumerate(impl.layer_grouping):
        local_layer_ids = [int(lid) for lid in local_layer_ids]
        if not local_layer_ids:
            continue

        num_slots = slots_by_group[layer_group_id]
        global_by_local = dict(zip(local_layer_ids, _global_layer_ids(manager, local_layer_ids)))

        buffer_ids = [b for lid in local_layer_ids for b in buffers_by_layer.get(lid, ())]

        regions: List[KvCacheRegion] = []
        for desc in impl.get_aggregated_pages(buffer_ids):
            if int(desc.layer_group_id) != layer_group_id:
                continue
            regions.append(
                KvCacheRegion(
                    base=int(desc.base),
                    size=int(desc.size),
                    stride=int(desc.stride),
                    num_slots=num_slots,
                    buffers=tuple(
                        KvCacheBufferRef(
                            layer_id=global_by_local[int(b.id.layer_id)],
                            role=str(b.id.role),
                            expansion=int(b.expansion),
                        )
                        for b in desc.buffers
                    ),
                )
            )

        groups.append(
            KvCacheLayerGroupLayout(
                layer_group_id=layer_group_id,
                layer_ids=tuple(global_by_local[lid] for lid in local_layer_ids),
                window_size=_window_size(init_config, local_layer_ids[0]),
                regions=tuple(regions),
            )
        )

    # A region may span roles whose element types differ; the KV dtype is the
    # one a typed view over K/V data needs, and anything else stays uint8.
    try:
        dtype = binding_to_torch_dtype(manager.dtype)
    except (AssertionError, KeyError, TypeError):
        dtype = torch.uint8

    return KvCacheLayout(
        tokens_per_block=int(manager.tokens_per_block),
        groups=tuple(groups),
        dtype=dtype,
    )
