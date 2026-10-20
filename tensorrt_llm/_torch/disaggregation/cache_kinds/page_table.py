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
"""A rank's cache memory as the transfer sees it.

The page table has four parts, none of them specific to a cache kind:

* :class:`Pool` -- where memory is: a base address and how a slot is addressed.
* :class:`UnitSpace` -- memory that one per-request list of unit ids indexes (a life cycle's blocks,
  a request's state slot). ``kind`` names the plugin that owns its selection rules; ``params`` is
  free-form data only that plugin reads.
* :class:`Region` -- one piece of memory inside a unit's slot: what it is (global layer id + role,
  the key both sides match on) and where it is (pool + offset).
* :class:`BufferDecl` -- how a region's bytes are laid out and split across ranks.

The mapping algorithm reads only regions and their declarations, so a new cache kind adds rows to
this table without changing its shape. It is exchanged between peers as is (``to_dict``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import prod
from typing import Any, Dict, List, Optional, Tuple

from tensorrt_llm._torch.disaggregation.base import CacheKind


@dataclass(frozen=True)
class BufferDecl:
    """One buffer of a region, as this rank holds it.

    ``axes`` lists the buffer's dimensions outer to inner as ``(name, extent)``; dimensions inside
    ``split_axis`` may be left out, they only make up the bytes of one unit. ``split_axis`` is the
    dimension the buffer is sharded along over ``domain``'s grid, or ``None`` when every rank of the
    grid holds the same bytes. A region is its buffers back to back.

    Examples (``kv`` heads on this rank, ``tpb`` tokens per block)::

        HND key:     axes=(("head", kv),),                split_axis="head"
        NHD key:     axes=(("token", tpb), ("head", kv)), split_axis="head"
        conv x|B|C:  one decl per section, axes=(("byte", section_bytes),), split_axis="byte"
        index-key:   axes=(),                             split_axis=None
    """

    nbytes: int
    domain: str
    axes: Tuple[Tuple[str, int], ...] = ()
    split_axis: Optional[str] = None

    def __post_init__(self) -> None:
        names = [name for name, _ in self.axes]
        if self.split_axis is not None and self.split_axis not in names:
            raise ValueError(f"split axis {self.split_axis!r} is not one of the axes {names}")
        if self.nbytes <= 0:
            raise ValueError(f"buffer must hold at least one byte, got {self.nbytes}")

    @property
    def axis_names(self) -> Tuple[str, ...]:
        return tuple(name for name, _ in self.axes)

    @property
    def _split_index(self) -> int:
        return self.axis_names.index(self.split_axis)

    @property
    def rows(self) -> int:
        """How many times the split axis repeats inside the buffer (1 when it is outermost)."""
        if self.split_axis is None:
            return 1
        return prod(extent for _, extent in self.axes[: self._split_index])

    @property
    def units(self) -> int:
        """This rank's extent along the split axis; a replicated buffer is a single unit."""
        if self.split_axis is None:
            return 1
        return self.axes[self._split_index][1]

    def unit_bytes(self, context: str) -> int:
        """Bytes of one unit of the split axis within one row.

        Taken from the registered byte size rather than element counts, so sub-byte dtypes (NVFP4)
        stay exact; a unit that does not end on a byte boundary cannot be sliced.
        """
        denominator = self.rows * self.units
        if denominator <= 0 or self.nbytes % denominator != 0:
            raise ValueError(
                f"{context}: buffer of {self.nbytes} bytes is not evenly divisible into "
                f"{self.rows} rows x {self.units} {self.split_axis!r} units (slice not byte-aligned)"
            )
        return self.nbytes // denominator

    def to_dict(self) -> dict:
        return {
            "nbytes": int(self.nbytes),
            "domain": self.domain,
            "axes": [[name, int(extent)] for name, extent in self.axes],
            "split_axis": self.split_axis,
        }

    @staticmethod
    def from_dict(data: dict) -> "BufferDecl":
        return BufferDecl(
            nbytes=int(data["nbytes"]),
            domain=str(data["domain"]),
            axes=tuple((str(name), int(extent)) for name, extent in data.get("axes") or ()),
            split_axis=data.get("split_axis"),
        )


@dataclass
class Pool:
    """Affine addressing of one allocation over slots.

    ``slot_bytes`` is the payload of one slot and ``num_slots`` the number of slots. A unit's slot
    starts at ``base_address + unit * slot_stride_bytes``; region offsets are relative to it, and the
    allocation spans ``num_slots * slot_stride_bytes`` bytes. The strides default to dense storage
    (``slot_stride_bytes == slot_bytes``, ``layer_stride_bytes == num_slots * slot_stride_bytes``).
    """

    base_address: int  # uint64
    slot_bytes: int
    num_slots: int
    slot_stride_bytes: Optional[int] = None
    layer_stride_bytes: Optional[int] = None

    def __post_init__(self) -> None:
        if self.slot_stride_bytes is None:
            self.slot_stride_bytes = self.slot_bytes
        if self.layer_stride_bytes is None:
            self.layer_stride_bytes = self.num_slots * self.slot_stride_bytes
        if self.slot_stride_bytes < self.slot_bytes:
            raise ValueError("slot_stride_bytes must be greater than or equal to slot_bytes")
        if self.layer_stride_bytes < self.slot_bytes:
            raise ValueError("layer_stride_bytes must be greater than or equal to slot_bytes")

    def to_dict(self) -> dict:
        return {
            "base_address": int(self.base_address),
            "slot_bytes": int(self.slot_bytes),
            "num_slots": int(self.num_slots),
            "slot_stride_bytes": int(self.slot_stride_bytes),
            "layer_stride_bytes": int(self.layer_stride_bytes),
        }

    @staticmethod
    def from_dict(data: dict) -> "Pool":
        return Pool(
            base_address=int(data["base_address"]),
            slot_bytes=int(data["slot_bytes"]),
            num_slots=int(data["num_slots"]),
            slot_stride_bytes=data.get("slot_stride_bytes"),
            layer_stride_bytes=data.get("layer_stride_bytes"),
        )


RegionKey = Tuple[int, str]
"""``(global layer id, role)``: what a region is, the same on both sides."""


@dataclass(frozen=True)
class Region:
    """One piece of memory inside a unit's slot.

    ``layer`` is a global layer id: unique per (layer, role) within a kind and equal on every PP
    split, so the two sides match regions by ``(kind, layer, role)``. ``local_layer`` is the
    manager's own id of the layer, kept for the plugin that described the region.
    """

    pool: int
    layer: int
    role: str
    offset: int
    buffers: Tuple[BufferDecl, ...]
    local_layer: int = -1

    def __post_init__(self) -> None:
        if not self.buffers:
            raise ValueError(f"region {self.key} declares no buffers")
        if self.offset < 0:
            raise ValueError(f"region {self.key} has negative offset {self.offset}")

    @property
    def key(self) -> RegionKey:
        return (self.layer, self.role)

    @property
    def nbytes(self) -> int:
        return sum(b.nbytes for b in self.buffers)

    @property
    def replicated(self) -> bool:
        """Every rank of the domain holds the same bytes, so one elected rank sends them."""
        return all(b.split_axis is None for b in self.buffers)


@dataclass
class UnitSpace:
    """Memory one per-request list of unit ids indexes.

    ``unit`` says how the ids are read: PAGED ids are blocks by ordinal (entry ``i`` covers tokens
    ``[i * tokens_per_block, ...)``), STATE ids are one slot per request. ``kind`` names the plugin
    whose rules select a request's units; ``params`` holds what those rules need (window size,
    tokens per block, ...) and must be msgpack-able.
    """

    kind: str
    unit: CacheKind
    regions: List[Region] = field(default_factory=list)
    params: Dict[str, Any] = field(default_factory=dict)

    @property
    def pools(self) -> List[int]:
        return sorted({r.pool for r in self.regions})

    @property
    def layers(self) -> List[int]:
        return sorted({r.layer for r in self.regions})

    def to_dict(self) -> dict:
        layouts: Dict[Tuple[BufferDecl, ...], int] = {}
        rows = []
        for r in self.regions:
            index = layouts.setdefault(r.buffers, len(layouts))
            rows.append(
                [int(r.pool), int(r.layer), str(r.role), int(r.offset), index, r.local_layer]
            )
        return {
            "kind": self.kind,
            "unit": int(self.unit),
            "params": dict(self.params),
            # Most layers share a layout, so declarations are sent once per distinct layout.
            "layouts": [[b.to_dict() for b in buffers] for buffers in layouts],
            "regions": rows,
        }

    @staticmethod
    def from_dict(data: dict) -> "UnitSpace":
        layouts = [
            tuple(BufferDecl.from_dict(b) for b in buffers) for buffers in data.get("layouts", [])
        ]
        return UnitSpace(
            kind=str(data["kind"]),
            unit=CacheKind(int(data["unit"])),
            params=dict(data.get("params") or {}),
            regions=[
                Region(
                    pool=int(pool),
                    layer=int(layer),
                    role=str(role),
                    offset=int(offset),
                    buffers=layouts[int(index)],
                    local_layer=int(local_layer),
                )
                for pool, layer, role, offset, index, local_layer in data.get("regions", [])
            ],
        )


@dataclass
class PageTable:
    """Everything a rank's cache holds: pools, and unit spaces of regions inside them.

    For a ``KVCacheManagerV2`` the space index is the life-cycle id, so per-request unit lists line
    up with the manager's own groups.
    """

    pools: List[Pool] = field(default_factory=list)
    spaces: List[UnitSpace] = field(default_factory=list)

    def pool_of(self, region: Region) -> Pool:
        return self.pools[region.pool]

    def to_dict(self) -> dict:
        return {
            "pools": [p.to_dict() for p in self.pools],
            "spaces": [s.to_dict() for s in self.spaces],
        }

    @staticmethod
    def from_dict(data: dict) -> "PageTable":
        return PageTable(
            pools=[Pool.from_dict(p) for p in data.get("pools", [])],
            spaces=[UnitSpace.from_dict(s) for s in data.get("spaces", [])],
        )
