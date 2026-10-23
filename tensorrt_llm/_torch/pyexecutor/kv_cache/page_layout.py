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
"""One rank's page layout and transfer geometry as immutable values; standard library only."""

from __future__ import annotations

import collections.abc
import dataclasses
import enum
import numbers
import types
from dataclasses import dataclass, field
from typing import Literal, Mapping

__all__ = [
    "BufferDesc",
    "LayerGroupDesc",
    "MapperKind",
    "PageLayout",
    "RankTopology",
    "RoleLayout",
    "ShardDesc",
]

_QUANT_MODES = ("fp8", "nvfp4", "fp8_ds_mla")


def _integer(value: int, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, numbers.Integral) or value < minimum:
        requirement = "positive" if minimum == 1 else "non-negative"
        raise ValueError(f"{name} must be a {requirement} integer, got {value!r}")
    return int(value)


def _frozen(mapping: dict) -> Mapping:
    return types.MappingProxyType(mapping)


# The default of an ``unchecked`` constructor's mapping argument.
_NO_ENTRIES: Mapping = _frozen({})


class MapperKind(enum.IntEnum):
    """How a role's bytes split across its layer group's grid of ranks (``LayerGroupDesc``).

    Attributes:
        INDEXED: Head-major layout; a head range is one contiguous byte range.
        HND: Alias of INDEXED for the [head, token, dimension] layout.
        REPLICATED: Identical bytes on every tensor-parallel rank.
        NHD: Token-major [token, head, dimension] layout; a head range is a slice within every
            token.
        SECTIONED: Consecutive byte sections, each sharded independently, such as recurrent
            convolution state.

    Raises:
        ValueError: If constructed from a value that names no mapper kind."""

    INDEXED = 0
    HND = INDEXED
    REPLICATED = 1
    NHD = 2
    SECTIONED = 3


@dataclass(frozen=True)
class RoleLayout:
    """Resharding geometry of one role; both fields None means no declared head axis.

    Attributes:
        section_bytes: Byte sizes of consecutive sections for a SECTIONED role, normalized to a
            tuple. When given, their sum must equal the buffer's size.
        bytes_per_head: Total bytes for one rank-local head, derived as size/head count for
            attention INDEXED/NHD roles. NHD offsets remain token-major, not one contiguous head
            slice. For other INDEXED roles, size must be divisible by this value; the quotient is
            the per-rank head count.

    Raises:
        ValueError: If a section size or bytes_per_head is not a positive integer, including a
            boolean. BufferDesc checks that the geometry fits its size."""

    section_bytes: tuple[int, ...] | None = None
    bytes_per_head: int | None = None

    def __post_init__(self) -> None:
        if self.section_bytes is not None:
            object.__setattr__(
                self,
                "section_bytes",
                tuple(_integer(n, "section size", 1) for n in self.section_bytes),
            )
        if self.bytes_per_head is not None:
            object.__setattr__(
                self, "bytes_per_head", _integer(self.bytes_per_head, "bytes_per_head", 1)
            )

    @classmethod
    def unchecked(
        cls, section_bytes: tuple[int, ...] | None = None, bytes_per_head: int | None = None
    ) -> RoleLayout:
        """The constructor's role layout of these values, without its checks and conversions;
        see ``PageLayout.unchecked``."""
        layout = object.__new__(cls)
        layout.__dict__.update(section_bytes=section_bytes, bytes_per_head=bytes_per_head)
        return layout


def _check_size(geometry: RoleLayout, size: int) -> None:
    if geometry.section_bytes is not None and sum(geometry.section_bytes) != size:
        raise ValueError(f"sections {geometry.section_bytes} do not sum to {size} bytes")
    if geometry.bytes_per_head is not None and size % geometry.bytes_per_head:
        raise ValueError(
            f"{size} bytes is not a whole number of {geometry.bytes_per_head}-byte heads"
        )


@dataclass(frozen=True)
class ShardDesc:
    """Which share of a layer group's content this rank holds.

    The default (count=1, index=0) is the whole content, named alike on every rank.

    Attributes:
        count: Positive number of content shards.
        index: Zero-based index of this rank's share, less than count.

    Raises:
        ValueError: If count is not a positive integer, index is not a non-negative integer, either
            is a boolean, or index lies outside count."""

    count: int = 1
    index: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "count", _integer(self.count, "shard count", 1))
        object.__setattr__(self, "index", _integer(self.index, "shard index"))
        if self.index >= self.count:
            raise ValueError(f"shard {self.index} of {self.count}")

    @classmethod
    def unchecked(cls, count: int = 1, index: int = 0) -> ShardDesc:
        """The constructor's shard of these values, without its checks and conversions; see
        ``PageLayout.unchecked``."""
        shard = object.__new__(cls)
        shard.__dict__.update(count=count, index=index)
        return shard


@dataclass(frozen=True)
class LayerGroupDesc:
    """One local layer group's windows, pools, global layers and content shard.

    A block held only by the kept window may remain unwritten without preventing
    resumption when it lies outside the read window.

    A group's transferred buffers split over one grid of ranks, set by its kind:
    attention groups over the tensor-parallel ranks of one attention data-parallel
    group; state groups over the tensor-parallel ranks, over none under attention
    data parallelism, and over tensor times context ranks, context-minor, under
    Helix. ``shard`` is this rank's share; each buffer's ``mapper`` and
    ``geometry`` say how it splits.

    Attributes:
        kind: "attention" for attention KV data, or "state" for recurrent state.
        kept_window: Positive number of tokens whose blocks the manager retains; None means full
            attention.
        num_sink_blocks: Non-negative number of sink blocks retained by the manager.
        pool_group: Non-negative index of the group's device pool group.
        layers: Global layer ids in this local group, normalized to a tuple.
        shard: The share of the group's content held on this rank.
        read_window: Positive number of tokens attention reads, defaulting to kept_window; None
            means full attention. A declared read window must lie within the kept window.

    Raises:
        ValueError: If an index or layer id is not a non-negative integer, a window is not a
            positive integer, any such value is a boolean, or read_window exceeds kept_window or is
            given when kept_window is None."""

    kind: Literal["attention", "state"]
    kept_window: int | None
    num_sink_blocks: int
    pool_group: int
    layers: tuple[int, ...]
    shard: ShardDesc = ShardDesc()
    read_window: int | None = None

    def __post_init__(self) -> None:
        if self.kept_window is not None:
            object.__setattr__(self, "kept_window", _integer(self.kept_window, "kept_window", 1))
        object.__setattr__(
            self, "num_sink_blocks", _integer(self.num_sink_blocks, "num_sink_blocks")
        )
        object.__setattr__(self, "pool_group", _integer(self.pool_group, "pool_group"))
        object.__setattr__(self, "layers", tuple(_integer(n, "layer") for n in self.layers))
        if self.read_window is None:
            object.__setattr__(self, "read_window", self.kept_window)
        else:
            object.__setattr__(self, "read_window", _integer(self.read_window, "read_window", 1))
        if self.read_window is not None and (
            self.kept_window is None or self.read_window > self.kept_window
        ):
            raise ValueError(
                f"a read window of {self.read_window} tokens is not within the kept window of "
                f"{self.kept_window} tokens"
            )

    @classmethod
    def unchecked(
        cls,
        kind: Literal["attention", "state"],
        kept_window: int | None,
        num_sink_blocks: int,
        pool_group: int,
        layers: tuple[int, ...],
        shard: ShardDesc = ShardDesc(),
        read_window: int | None = None,
    ) -> LayerGroupDesc:
        """The constructor's layer group of these values, without its checks and conversions; a
        read window of None still defaults to the kept window. See ``PageLayout.unchecked``."""
        group = object.__new__(cls)
        group.__dict__.update(
            kind=kind,
            kept_window=kept_window,
            num_sink_blocks=num_sink_blocks,
            pool_group=pool_group,
            layers=layers,
            shard=shard,
            read_window=kept_window if read_window is None else read_window,
        )
        return group


@dataclass(frozen=True)
class BufferDesc:
    """One role of one global layer within a page's pool slot.

    Attributes:
        layer: Non-negative global layer id.
        role: Role name, normalized to a string.
        pool_group: Non-negative device pool group index.
        pool_index: Non-negative pool index within pool_group.
        slot_offset: Non-negative byte offset within that pool's slot.
        size: Positive byte size of the buffer.
        mapper: How this role's bytes split across tensor-parallel ranks.
        geometry: RoleLayout; declared section sizes must sum to size, and declared bytes per head
            must divide size.
        transferred: Whether peers transfer this role. False marks local-only buffers, which still
            accompany copies of whole slots.
        expansion: Positive number of sub-pages within one page, each of
            PageLayout.tokens_per_page / expansion of the buffer's tokens; consumers accessing bytes
            within the buffer must support it.

    Raises:
        ValueError: If an index is not a non-negative integer, size or expansion is not a positive
            integer, any such value is a boolean, mapper names no mapper kind, geometry is not a
            RoleLayout, or geometry does not fit size."""

    layer: int
    role: str
    pool_group: int
    pool_index: int
    slot_offset: int
    size: int
    mapper: MapperKind = MapperKind.INDEXED
    geometry: RoleLayout = RoleLayout()
    transferred: bool = True
    expansion: int = 1

    def __post_init__(self) -> None:
        for name in ("layer", "pool_group", "pool_index", "slot_offset"):
            object.__setattr__(self, name, _integer(getattr(self, name), name))
        object.__setattr__(self, "size", _integer(self.size, "size", 1))
        object.__setattr__(self, "expansion", _integer(self.expansion, "expansion", 1))
        object.__setattr__(self, "role", str(self.role))
        object.__setattr__(self, "mapper", MapperKind(_integer(self.mapper, "mapper")))
        if not isinstance(self.geometry, RoleLayout):
            raise ValueError(f"geometry must be a role layout, got {self.geometry!r}")
        _check_size(self.geometry, self.size)

    @classmethod
    def unchecked(
        cls,
        layer: int,
        role: str,
        pool_group: int,
        pool_index: int,
        slot_offset: int,
        size: int,
        mapper: MapperKind = MapperKind.INDEXED,
        geometry: RoleLayout = RoleLayout(),
        transferred: bool = True,
        expansion: int = 1,
    ) -> BufferDesc:
        """The constructor's buffer of these values, without its checks and conversions; see
        ``PageLayout.unchecked``."""
        buffer = object.__new__(cls)
        buffer.__dict__.update(
            layer=layer,
            role=role,
            pool_group=pool_group,
            pool_index=pool_index,
            slot_offset=slot_offset,
            size=size,
            mapper=mapper,
            geometry=geometry,
            transferred=transferred,
            expansion=expansion,
        )
        return buffer


@dataclass(frozen=True)
class RankTopology:
    """This rank's parallel sizes and ranks, as native RankInfo carries them.

    Attributes:
        tp_size: Positive tensor-parallel group size.
        tp_rank: Zero-based tensor-parallel rank, less than tp_size.
        pp_size: Positive pipeline-parallel group size.
        pp_rank: Zero-based pipeline-parallel rank, less than pp_size.
        cp_size: Positive context-parallel group size.
        cp_rank: Zero-based context-parallel rank, less than cp_size.
        dp_size: Positive attention data-parallel group size.
        dp_rank: Zero-based attention data-parallel rank, less than dp_size.
        attention_dp: Whether attention uses data parallelism; each tensor-parallel rank then
            occupies its own attention data-parallel rank.

    Raises:
        ValueError: If a size is not a positive integer, a rank is not a non-negative integer, any
            such value is a boolean, or a rank lies outside its size."""

    tp_size: int = 1
    tp_rank: int = 0
    pp_size: int = 1
    pp_rank: int = 0
    cp_size: int = 1
    cp_rank: int = 0
    dp_size: int = 1
    dp_rank: int = 0
    attention_dp: bool = False

    def __post_init__(self) -> None:
        for name in ("tp", "pp", "cp", "dp"):
            size = _integer(getattr(self, f"{name}_size"), f"{name}_size", 1)
            rank = _integer(getattr(self, f"{name}_rank"), f"{name}_rank")
            if rank >= size:
                raise ValueError(f"{name} rank {rank} of {size}")
            object.__setattr__(self, f"{name}_size", size)
            object.__setattr__(self, f"{name}_rank", rank)

    @classmethod
    def unchecked(
        cls,
        tp_size: int = 1,
        tp_rank: int = 0,
        pp_size: int = 1,
        pp_rank: int = 0,
        cp_size: int = 1,
        cp_rank: int = 0,
        dp_size: int = 1,
        dp_rank: int = 0,
        attention_dp: bool = False,
    ) -> RankTopology:
        """The constructor's topology of these values, without its checks and conversions; see
        ``PageLayout.unchecked``."""
        topology = object.__new__(cls)
        topology.__dict__.update(
            tp_size=tp_size,
            tp_rank=tp_rank,
            pp_size=pp_size,
            pp_rank=pp_rank,
            cp_size=cp_size,
            cp_rank=cp_rank,
            dp_size=dp_size,
            dp_rank=dp_rank,
            attention_dp=attention_dp,
        )
        return topology


@dataclass(frozen=True)
class PageLayout:
    """How one rank's pages are built and where the rank stands, as immutable values.

    Values contain no addresses or manager ownership. A page uses one slot from each pool of its
    device pool group, in pool order. Buffers locate their bytes within those slots. Layer groups
    follow the manager's local order, indexed by LentLayerGroupRows.layer_group_index; their layer
    ids are global. Mapping inputs are copied into read-only mappings, and layer groups and buffers
    into tuples. Equal layouts hash alike and can serve as mapping keys.

    Region maps index Lender.regions, where staging regions precede device regions. A staging slot
    concatenates the group's pools' slots in pool order. Regions absent from a lender have no map
    entry; both maps are empty outside a lender.

    Attributes:
        tokens_per_page: Positive token count of one page. Under Helix, one manager block spans
            topology.cp_size pages, one per context-parallel rank.
        pool_slot_bytes: Device pool group index -> positive slot byte widths of its pools, in pool
            order.
        layer_groups: Tuple describing each local layer group's kind, windows, sinks, device pool
            group, global layer ids and content shard.
        buffers: Tuple of every role buffer of every layer within a page.
        topology: This rank's parallel sizes and ranks.
        total_kv_heads_by_layer: Global layer id -> positive KV head count over all tensor-parallel
            ranks, only for attention layers whose count the manager knows.
        dtype: Name of the manager's tensorrt_llm.bindings.DataType member for KV data, such as
            "HALF", "BF16", "FP8" or "NVFP4"; None if undeclared.
        quant_mode: None for unquantized KV, else "fp8", "nvfp4" or "fp8_ds_mla" (DeepSeek-V4's FP8
            format with its scales within the page).
        device_region_index: (pool group, pool index) -> device region index in Lender.regions.
        staging_region_index: Pool group -> staging region index in Lender.regions; empty when the
            lender stages nothing.

    Raises:
        ValueError: If tokens_per_page, a slot width or a KV head count is not a positive integer; a
            group, pool, layer or region index is not a non-negative integer; any such value is a
            boolean; a buffer's expansion does not divide tokens_per_page; or quant_mode is not
            listed above."""

    tokens_per_page: int
    pool_slot_bytes: Mapping[int, tuple[int, ...]]
    layer_groups: tuple[LayerGroupDesc, ...]
    buffers: tuple[BufferDesc, ...]
    topology: RankTopology = RankTopology()
    total_kv_heads_by_layer: Mapping[int, int] = field(default_factory=dict)
    dtype: str | None = None
    quant_mode: Literal["fp8", "nvfp4", "fp8_ds_mla"] | None = None
    device_region_index: Mapping[tuple[int, int], int] = field(default_factory=dict)
    staging_region_index: Mapping[int, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "tokens_per_page", _integer(self.tokens_per_page, "tokens_per_page", 1)
        )
        pools = {
            _integer(g, "pool group"): tuple(_integer(w, "slot width", 1) for w in widths)
            for g, widths in self.pool_slot_bytes.items()
        }
        object.__setattr__(self, "pool_slot_bytes", _frozen(pools))
        heads = {
            _integer(layer, "layer"): _integer(count, "KV heads", 1)
            for layer, count in self.total_kv_heads_by_layer.items()
        }
        object.__setattr__(self, "total_kv_heads_by_layer", _frozen(heads))
        device = {
            (_integer(g, "pool group"), _integer(p, "pool index")): _integer(i, "region index")
            for (g, p), i in self.device_region_index.items()
        }
        object.__setattr__(self, "device_region_index", _frozen(device))
        staging = {
            _integer(g, "pool group"): _integer(i, "region index")
            for g, i in self.staging_region_index.items()
        }
        object.__setattr__(self, "staging_region_index", _frozen(staging))
        object.__setattr__(self, "layer_groups", tuple(self.layer_groups))
        object.__setattr__(self, "buffers", tuple(self.buffers))
        if self.quant_mode is not None and self.quant_mode not in _QUANT_MODES:
            raise ValueError(
                f"quant_mode must be None or one of {_QUANT_MODES}, got {self.quant_mode!r}"
            )
        for buffer in self.buffers:
            if self.tokens_per_page % buffer.expansion:
                raise ValueError(
                    f"buffer ({buffer.layer}, {buffer.role}): expansion {buffer.expansion} does not "
                    f"divide {self.tokens_per_page} tokens per page"
                )

    @classmethod
    def unchecked(
        cls,
        tokens_per_page: int,
        pool_slot_bytes: Mapping[int, tuple[int, ...]],
        layer_groups: tuple[LayerGroupDesc, ...],
        buffers: tuple[BufferDesc, ...],
        topology: RankTopology = RankTopology(),
        total_kv_heads_by_layer: Mapping[int, int] = _NO_ENTRIES,
        dtype: str | None = None,
        quant_mode: Literal["fp8", "nvfp4", "fp8_ds_mla"] | None = None,
        device_region_index: Mapping[tuple[int, int], int] = _NO_ENTRIES,
        staging_region_index: Mapping[int, int] = _NO_ENTRIES,
    ) -> PageLayout:
        """The constructor's layout of these values, without its checks and conversions.

        Mappings become read-only copies and layer groups and buffers tuples, as in the
        constructor; every other value, including those inside layer groups and buffers, is kept
        as given. For values the constructor keeps unchanged, such as another layout's fields, the
        result equals the constructor's. Values it would refuse or convert are not detected: they
        stay as given, the layout then breaks the constraints documented above, and code reading
        it may fail or misbehave. Build layouts from values of unknown origin with the constructor.
        """
        layout = object.__new__(cls)
        layout.__dict__.update(
            tokens_per_page=tokens_per_page,
            pool_slot_bytes=_frozen(dict(pool_slot_bytes)),
            layer_groups=tuple(layer_groups),
            buffers=tuple(buffers),
            topology=topology,
            total_kv_heads_by_layer=_frozen(dict(total_kv_heads_by_layer)),
            dtype=dtype,
            quant_mode=quant_mode,
            device_region_index=_frozen(dict(device_region_index)),
            staging_region_index=_frozen(dict(staging_region_index)),
        )
        return layout

    def __hash__(self) -> int:
        values = (getattr(self, f.name) for f in dataclasses.fields(self))
        return hash(
            tuple(
                tuple(sorted(value.items()))
                if isinstance(value, collections.abc.Mapping)
                else value
                for value in values
            )
        )
