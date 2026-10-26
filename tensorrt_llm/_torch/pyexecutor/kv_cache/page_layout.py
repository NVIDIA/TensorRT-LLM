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
"""One rank's page layout and transfer geometry as immutable values, and which ranks of an
instance hold the same bytes; standard library only."""

from __future__ import annotations

import collections.abc
import dataclasses
import numbers
import types
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal, Mapping

__all__ = [
    "GRID_DIMENSIONS",
    "KIND_DEFAULTS",
    "BufferDesc",
    "LayerGroupDesc",
    "PageLayout",
    "RankTopology",
    "Segment",
    "ShardDesc",
    "grid_position",
    "held_span",
    "replica_ranks",
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


@dataclass(frozen=True)
class Segment:
    """A run of a buffer's bytes and how it splits across its layer group's grid of ranks.

    A buffer is its segments back to back. ``axes`` lists the segment's dimensions on this rank,
    outer to inner, as ``(name, extent)``; dimensions inside ``split_axis`` may be left out, since
    they only make up the bytes of one unit of it. ``split_axis`` names the dimension sharded over
    the layer group's grid of ranks (see ``LayerGroupDesc``), or is None when every rank of the grid
    holds the same bytes.

    Examples, with ``h`` heads on this rank and ``t`` tokens per page::

        head-major K:   Segment(n, (("head", h),), "head")
        token-major K:  Segment(n, (("token", t), ("head", h)), "head")
        conv x | B | C: one Segment(s, (("byte", s),), "byte") per section of s bytes
        index-K:        Segment(n)

    Attributes:
        nbytes: Positive byte size of the segment.
        axes: Dimensions outer to inner, normalized to a tuple of (name, positive extent) pairs.
        split_axis: Name of the sharded dimension, one of ``axes``; None when not sharded.

    Raises:
        ValueError: If nbytes or an extent is not a positive integer or is a boolean, axis names
            repeat, split_axis names no axis, or the axes up to split_axis do not divide nbytes
            into whole units."""

    nbytes: int
    axes: tuple[tuple[str, int], ...] = ()
    split_axis: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "nbytes", _integer(self.nbytes, "segment size", 1))
        axes = tuple(
            (str(name), _integer(extent, f"extent of axis {name!r}", 1))
            for name, extent in self.axes
        )
        object.__setattr__(self, "axes", axes)
        names = [name for name, _ in axes]
        if len(set(names)) != len(names):
            raise ValueError(f"axis names repeat: {names}")
        if self.split_axis is None:
            return
        object.__setattr__(self, "split_axis", str(self.split_axis))
        if self.split_axis not in names:
            raise ValueError(f"split axis {self.split_axis!r} is not one of the axes {names}")
        if self.nbytes % (self.rows * self.split_extent):
            raise ValueError(
                f"{self.nbytes} bytes is not a whole number of {self.rows} x {self.split_extent} "
                f"units of axis {self.split_axis!r}"
            )

    @property
    def split_extent(self) -> int | None:
        """The extent of ``split_axis`` on this rank, or None when not sharded."""
        if self.split_axis is None:
            return None
        return dict(self.axes)[self.split_axis]

    @property
    def rows(self) -> int:
        """How many times the split axis repeats within the segment: the product of the extents
        outside it, 1 when it is outermost or the segment is not sharded."""
        if self.split_axis is None:
            return 1
        rows = 1
        for name, extent in self.axes:
            if name == self.split_axis:
                return rows
            rows *= extent
        return rows

    @property
    def unit_bytes(self) -> int:
        """Bytes of one unit of the split axis within one row; the whole segment when not
        sharded."""
        if self.split_axis is None:
            return self.nbytes
        return self.nbytes // (self.rows * self.split_extent)

    @classmethod
    def unchecked(
        cls,
        nbytes: int,
        axes: tuple[tuple[str, int], ...] = (),
        split_axis: str | None = None,
    ) -> Segment:
        """The constructor's segment of these values, without its checks and conversions; see
        ``PageLayout.unchecked``."""
        segment = object.__new__(cls)
        segment.__dict__.update(nbytes=nbytes, axes=axes, split_axis=split_axis)
        return segment


def _check_segments(segments: tuple[Segment, ...], size: int) -> None:
    if not segments:
        raise ValueError("a buffer has at least one segment")
    for segment in segments:
        if not isinstance(segment, Segment):
            raise ValueError(f"segments must be Segment values, got {segment!r}")
    total = sum(segment.nbytes for segment in segments)
    if total != size:
        raise ValueError(f"segments of {total} bytes do not make up a {size}-byte buffer")


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


GRID_DIMENSIONS = ("tp", "cp")
"""Parallel dimensions a layer group's grid may name, in this order: "tp" is the tensor-parallel
ranks of one attention data-parallel group, "cp" the context-parallel ranks. Every grid starts
with "tp"; ranks along a dimension a grid leaves out hold the same bytes, unless the group's blocks
split by context-parallel rank (``blocks_by_cp``)."""

KIND_DEFAULTS = {
    "attention": (("tp",), "block", True),
    "state": (("tp", "cp"), "slot", False),
}
"""kind -> the (grid, unit, blocks_by_cp) a layer group of that kind defaults to."""


def _group_traits(
    kind: str, grid: tuple[str, ...] | None, unit: str | None, blocks_by_cp: bool | None
) -> tuple[tuple[str, ...], str, bool]:
    if kind not in KIND_DEFAULTS:
        raise ValueError(f"kind must be one of {sorted(KIND_DEFAULTS)}, got {kind!r}")
    default_grid, default_unit, default_blocks_by_cp = KIND_DEFAULTS[kind]
    grid = default_grid if grid is None else tuple(str(d) for d in grid)
    if not grid or grid != GRID_DIMENSIONS[: len(grid)]:
        raise ValueError(f"grid must be a leading part of {GRID_DIMENSIONS}, got {grid}")
    unit = default_unit if unit is None else str(unit)
    if unit not in ("block", "slot"):
        raise ValueError(f'unit must be "block" or "slot", got {unit!r}')
    if blocks_by_cp is None:
        blocks_by_cp = default_blocks_by_cp and unit == "block"
    if blocks_by_cp and unit != "block":
        raise ValueError("only blocks split by context-parallel rank; a slot unit is never split")
    return grid, unit, bool(blocks_by_cp)


def held_span(units: int, size: int, rank: int, total: int) -> tuple[int, int]:
    """The half-open range of a ``total``-unit axis that rank ``rank`` of ``size`` holds, ``units``
    long from ``rank * total // size``. Ranks hold consecutive ranges in rank order; when
    ``units * size`` exceeds ``total``, several ranks hold the same range."""
    start = rank * total // size
    return start, start + units


def grid_position(topology: RankTopology, grid: Sequence[str]) -> tuple[int, int]:
    """``(size, rank)`` of this rank in the grid that ``grid`` names: the ranks of each dimension,
    outer to inner, one flat rank with the last dimension minor. Under attention data parallelism
    every rank serves its own requests and holds its buffers whole: the grid is this rank alone."""
    if topology.attention_dp:
        return 1, 0
    size, rank = 1, 0
    for dimension in grid:
        n, r = (
            (topology.tp_size, topology.tp_rank)
            if dimension == "tp"
            else (topology.cp_size, topology.cp_rank)
        )
        size, rank = size * n, rank * n + r
    return size, rank


@dataclass(frozen=True)
class LayerGroupDesc:
    """One local layer group's windows, pools, global layers and content shard.

    A block held only by the kept window may remain unwritten without preventing
    resumption when it lies outside the read window.

    A group's split buffers spread over the grid of ranks ``grid`` names
    (``grid_position``); each buffer's ``segments`` say how it splits, and ``shard``
    is this rank's share. A request's units in the group are ``unit``: blocks by
    ordinal, each of ``tokens_per_page`` tokens, or one slot per request.

    Attributes:
        kind: "attention" for attention KV data, or "state" for recurrent state; a label for
            readers, which the fields below default from.
        kept_window: Positive number of tokens whose blocks the manager retains; None means full
            attention.
        num_sink_blocks: Non-negative number of sink blocks retained by the manager.
        pool_group: Non-negative index of the group's device pool group.
        layers: Global layer ids in this local group, normalized to a tuple.
        shard: The share of the group's content held on this rank.
        read_window: Positive number of tokens attention reads, defaulting to kept_window; None
            means full attention. A declared read window must lie within the kept window.
        model_layers: The model layer each of ``layers`` belongs to, in the same order, normalized
            to a tuple; None gives ``layers`` themselves. Layer ids numbering several caches of one
            model layer apart, such as one per attention type, all name that model layer.
        grid: The parallel dimensions the group's split buffers spread over, outer to inner, from
            ``GRID_DIMENSIONS``, which it starts; normalized to a tuple. Defaults to ``("tp",)`` for attention and
            ``("tp", "cp")`` for state.
        unit: "block" when a request's units are blocks by ordinal, "slot" when a request holds
            one slot. Defaults to "block" for attention and "slot" for state.
        blocks_by_cp: Whether, under Helix, context-parallel rank ``r`` holds blocks
            ``r::cp_size`` of the request, each rank its own pages. Defaults to True for "block"
            units; always False for "slot" units.

    Raises:
        ValueError: If an index, layer id or model layer is not a non-negative integer, a window is
            not a positive integer, any such value is a boolean, read_window exceeds kept_window or
            is given when kept_window is None, model_layers does not name one model layer per
            layer, grid is not a leading part of ``GRID_DIMENSIONS``, unit is
            neither "block" nor "slot", or blocks_by_cp is set for "slot" units."""

    kind: Literal["attention", "state"]
    kept_window: int | None
    num_sink_blocks: int
    pool_group: int
    layers: tuple[int, ...]
    shard: ShardDesc = ShardDesc()
    read_window: int | None = None
    model_layers: tuple[int, ...] | None = None
    grid: tuple[str, ...] | None = None
    unit: Literal["block", "slot"] | None = None
    blocks_by_cp: bool | None = None

    def __post_init__(self) -> None:
        grid, unit, blocks_by_cp = _group_traits(self.kind, self.grid, self.unit, self.blocks_by_cp)
        object.__setattr__(self, "grid", grid)
        object.__setattr__(self, "unit", unit)
        object.__setattr__(self, "blocks_by_cp", blocks_by_cp)
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
        if self.model_layers is None:
            object.__setattr__(self, "model_layers", self.layers)
        else:
            model_layers = tuple(_integer(n, "model layer") for n in self.model_layers)
            if len(model_layers) != len(self.layers):
                raise ValueError(
                    f"{len(model_layers)} model layers for the {len(self.layers)} layers "
                    f"{self.layers}"
                )
            object.__setattr__(self, "model_layers", model_layers)

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
        model_layers: tuple[int, ...] | None = None,
        grid: tuple[str, ...] | None = None,
        unit: Literal["block", "slot"] | None = None,
        blocks_by_cp: bool | None = None,
    ) -> LayerGroupDesc:
        """The constructor's layer group of these values, without its checks and conversions; a
        read window of None still defaults to the kept window, model layers of None to the
        layers, and grid, unit and blocks_by_cp of None to the kind's. See
        ``PageLayout.unchecked``."""
        default_grid, default_unit, default_blocks_by_cp = KIND_DEFAULTS[kind]
        group = object.__new__(cls)
        group.__dict__.update(
            kind=kind,
            kept_window=kept_window,
            num_sink_blocks=num_sink_blocks,
            pool_group=pool_group,
            layers=layers,
            shard=shard,
            read_window=kept_window if read_window is None else read_window,
            model_layers=layers if model_layers is None else model_layers,
            grid=default_grid if grid is None else grid,
            unit=default_unit if unit is None else unit,
            blocks_by_cp=default_blocks_by_cp if blocks_by_cp is None else blocks_by_cp,
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
        segments: The buffer's bytes as consecutive segments, each saying how it splits across
            the layer group's grid of ranks; normalized to a tuple whose sizes sum to size.
            Defaults to one segment of the whole buffer that every rank holds alike.
        transferred: Whether peers transfer this role. False marks local-only buffers, which still
            accompany copies of whole slots.
        expansion: Positive number of sub-pages within one page, each of
            PageLayout.tokens_per_page / expansion of the buffer's tokens; consumers accessing bytes
            within the buffer must support it.

    Raises:
        ValueError: If an index is not a non-negative integer, size or expansion is not a positive
            integer, any such value is a boolean, or the segments do not make up size."""

    layer: int
    role: str
    pool_group: int
    pool_index: int
    slot_offset: int
    size: int
    segments: tuple[Segment, ...] = ()
    transferred: bool = True
    expansion: int = 1

    def __post_init__(self) -> None:
        for name in ("layer", "pool_group", "pool_index", "slot_offset"):
            object.__setattr__(self, name, _integer(getattr(self, name), name))
        object.__setattr__(self, "size", _integer(self.size, "size", 1))
        object.__setattr__(self, "expansion", _integer(self.expansion, "expansion", 1))
        object.__setattr__(self, "role", str(self.role))
        segments = tuple(self.segments) or (Segment(self.size),)
        _check_segments(segments, self.size)
        object.__setattr__(self, "segments", segments)

    @classmethod
    def unchecked(
        cls,
        layer: int,
        role: str,
        pool_group: int,
        pool_index: int,
        slot_offset: int,
        size: int,
        segments: tuple[Segment, ...] = (),
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
            segments=segments,
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


def _same_group(layout: PageLayout, desc: LayerGroupDesc) -> LayerGroupDesc | None:
    """The layer group of ``layout`` holding exactly ``desc``'s layers, of its kind, or None."""
    layers = frozenset(desc.layers)
    for other in layout.layer_groups:
        if other.kind == desc.kind and frozenset(other.layers) == layers:
            return other
    return None


def _attention_head_ranges(
    layout: PageLayout, desc: LayerGroupDesc
) -> tuple[tuple[int, str, int, int], ...] | None:
    """The transferred nonreplicated buffers' head ranges, or None without their geometry: total
    heads of each layer and buffers split by head alone."""
    size, rank = grid_position(layout.topology, desc.grid)
    ranges = []
    for buffer in layout.buffers:
        if (
            buffer.pool_group != desc.pool_group
            or buffer.layer not in desc.layers
            or not buffer.transferred
            or all(segment.split_axis is None for segment in buffer.segments)
        ):
            continue
        total = layout.total_kv_heads_by_layer.get(buffer.layer)
        (segment, *rest) = buffer.segments
        if not total or rest or segment.split_axis != "head":
            return None
        start, _ = held_span(segment.split_extent, size, rank, total)
        ranges.append((buffer.layer, buffer.role, start, segment.split_extent))
    return tuple(sorted(ranges)) if ranges else None


def replica_ranks(
    ranks: Sequence[PageLayout],
    rank: int,
    group: int,
    *,
    replicated: bool = False,
) -> tuple[int, ...]:
    """The flat ranks of one instance holding the same bytes as layer group ``group`` of rank
    ``rank``, read from every rank's published layout.

    Ranks hold the same bytes of a layer group when they serve the same requests (the same
    attention data-parallel rank); hold the same pages (under Helix a group whose blocks split by
    context-parallel rank, ``blocks_by_cp``, holds other pages on each); hold the same layers, in a layer
    group of the same kind; and hold the same share of the group's content: a replicated buffer is
    the same on every such rank. Other buffers match when their published content shards are equal,
    or when every transferred buffer's head range is equal. Per-buffer head ranges also recognize
    replicas when a group contains layers with different replication factors. Without head
    geometry, the published content shard is the only basis for matching.

    Args:
        ranks: Every rank's layout of one instance, by flat rank.
        rank: The flat rank whose layer group is matched.
        group: The index of the layer group in that rank's layout.
        replicated: Match the group's replicated buffers; otherwise its other buffers.

    Returns:
        The matching flat ranks, ascending, ``rank`` among them.

    Raises:
        IndexError: ``rank`` or ``group`` names no rank or layer group.
    """
    layout = ranks[rank]
    desc = layout.layer_groups[group]
    topology = layout.topology
    paged_by_context = desc.blocks_by_cp
    head_ranges = None if replicated else _attention_head_ranges(layout, desc)
    out = []
    for other_rank in range(len(ranks)):
        other = ranks[other_rank]
        other_topology = other.topology
        if other_topology.dp_rank != topology.dp_rank:
            continue
        if paged_by_context and other_topology.cp_rank != topology.cp_rank:
            continue
        match = _same_group(other, desc)
        if match is None:
            continue
        if (
            not replicated
            and match.shard != desc.shard
            and (head_ranges is None or _attention_head_ranges(other, match) != head_ranges)
        ):
            continue
        out.append(other_rank)
    return tuple(out)
