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
"""One rank's page layout as immutable values, and which ranks of an instance hold the same bytes;
standard library only.

A buffer holds the local pieces of one or more tensors. A tensor names its axes and their extents
over all ranks together; each axis says how its indices spread over the parallel dimensions
(``Axis.split``: whole on every rank, a contiguous run per rank, or every n-th index per rank) and
the byte stride of its local index. Which global indices a rank holds is therefore a progression
per axis (``held``), and how two ranks' pages correspond follows from intersecting them
(``layout_mapping``)."""

from __future__ import annotations

import collections.abc
import dataclasses
import math
import numbers
import types
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal, Mapping

__all__ = [
    "GRID_DIMENSIONS",
    "Axis",
    "BufferDesc",
    "LayerGroupDesc",
    "PageLayout",
    "Progression",
    "RankTopology",
    "ShardDesc",
    "Tensor",
    "content_shard",
    "grid_place",
    "head_major",
    "held",
    "held_bytes",
    "layout_from_dict",
    "layout_to_dict",
    "owned",
    "replica_ranks",
    "replicated",
    "sectioned",
    "token_major",
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

GRID_DIMENSIONS = ("tp", "cp")
"""Parallel dimensions an axis may split over (``Axis.split``): "tp" the tensor-parallel ranks of
one attention data-parallel group, "cp" the context-parallel ranks. Pipeline ranks hold other
layers and data-parallel ranks other requests, so neither splits a page."""


@dataclass(frozen=True)
class Progression:
    """The indices ``start, start + step, ..., start + (count - 1) * step``.

    Attributes:
        start: First index.
        step: Positive distance between consecutive indices.
        count: Non-negative number of indices.
    """

    start: int
    step: int
    count: int

    @property
    def last(self) -> int:
        """The last index; ``start - step`` when empty."""
        return self.start + (self.count - 1) * self.step

    def __len__(self) -> int:
        return self.count

    def __iter__(self):
        return iter(range(self.start, self.start + self.count * self.step, self.step))

    def __contains__(self, index: object) -> bool:
        if not isinstance(index, int) or self.count == 0:
            return False
        offset = index - self.start
        return 0 <= offset <= (self.count - 1) * self.step and offset % self.step == 0

    def intersect(self, other: Progression) -> Progression | None:
        """The indices both hold, themselves a progression, or None when they share none."""
        if self.count == 0 or other.count == 0:
            return None
        lo, hi = max(self.start, other.start), min(self.last, other.last)
        if lo > hi:
            return None
        g = math.gcd(self.step, other.step)
        if (other.start - self.start) % g:
            return None
        lcm = self.step // g * other.step
        # x = self.start + self.step * t with x = other.start (mod other.step).
        m = other.step // g
        t = 0 if m == 1 else (other.start - self.start) // g * pow(self.step // g, -1, m) % m
        first = lo + (self.start + self.step * t - lo) % lcm
        if first > hi:
            return None
        return Progression(first, lcm, (hi - first) // lcm + 1)


def grid_place(topology: RankTopology, dims: Sequence[str]) -> tuple[int, int]:
    """``(size, rank)`` of this rank in the grid of the parallel dimensions ``dims``, outer to
    inner, the last minor. Under attention data parallelism each tensor-parallel rank serves its own
    requests and holds its buffers whole, so "tp" counts this rank alone."""
    size, rank = 1, 0
    for dim in dims:
        if dim == "tp":
            n, r = (1, 0) if topology.attention_dp else (topology.tp_size, topology.tp_rank)
        elif dim == "cp":
            n, r = topology.cp_size, topology.cp_rank
        else:
            raise ValueError(f"{dim!r} is no grid dimension; one of {GRID_DIMENSIONS}")
        size, rank = size * n, rank * n + r
    return size, rank


@dataclass(frozen=True)
class Axis:
    """One dimension of a tensor: its extent over all ranks together, which of its indices a rank
    holds, and the byte stride between consecutive indices a rank holds.

    A rank at ``r`` of ``n`` in the grid of the ``split`` dimensions holds every index when
    ``split`` is empty; ``held`` indices from ``r * extent // n`` when split contiguously (ranks
    hold overlapping or equal runs when ``held * n`` exceeds ``extent``); or indices
    ``r, r + n, ...`` below ``extent`` when ``cyclic``. Local index ``k`` (the k-th index held, in
    order) lies at byte ``k * stride`` of its tensor.

    Attributes:
        name: The axis' name; peers match tensors' axes on it.
        extent: Positive number of indices over all ranks together.
        stride: Positive bytes between consecutive local indices.
        split: Parallel dimensions from ``GRID_DIMENSIONS`` the axis spreads over, outer to inner;
            empty when every rank holds every index. Normalized to a tuple.
        held: Positive number of indices a rank holds of a contiguous split, at most extent;
            required for a contiguous split, None otherwise.
        cyclic: Whether a split rank holds every n-th index rather than a contiguous run.

    Raises:
        ValueError: If a number is not a positive integer or is a boolean, split names a dimension
            twice or one outside ``GRID_DIMENSIONS``, held is missing from a contiguous split or
            given for another, or held exceeds extent.
    """

    name: str
    extent: int
    stride: int
    split: tuple[str, ...] = ()
    held: int | None = None
    cyclic: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError(f"an axis needs a name, got {self.name!r}")
        object.__setattr__(self, "extent", _integer(self.extent, f"extent of {self.name}", 1))
        object.__setattr__(self, "stride", _integer(self.stride, f"stride of {self.name}", 1))
        split = tuple(str(dim) for dim in self.split)
        if len(set(split)) != len(split) or not set(split) <= set(GRID_DIMENSIONS):
            raise ValueError(f"axis {self.name} splits over {split}, not over {GRID_DIMENSIONS}")
        object.__setattr__(self, "split", split)
        object.__setattr__(self, "cyclic", bool(self.cyclic))
        contiguous = bool(split) and not self.cyclic
        if self.cyclic and not split:
            raise ValueError(f"axis {self.name} is cyclic but not split")
        if contiguous != (self.held is not None):
            raise ValueError(
                f"axis {self.name}: a contiguous split, and only one, says how many indices a "
                f"rank holds (held={self.held!r}, split={split}, cyclic={self.cyclic})"
            )
        if self.held is not None:
            held_ = _integer(self.held, f"indices held of {self.name}", 1)
            if held_ > self.extent:
                raise ValueError(f"axis {self.name}: {held_} held of {self.extent} indices")
            object.__setattr__(self, "held", held_)

    @classmethod
    def unchecked(
        cls,
        name: str,
        extent: int,
        stride: int,
        split: tuple[str, ...] = (),
        held: int | None = None,
        cyclic: bool = False,
    ) -> Axis:
        """The constructor's axis of these values, without its checks and conversions; see
        ``PageLayout.unchecked``."""
        axis = object.__new__(cls)
        axis.__dict__.update(
            name=name, extent=extent, stride=stride, split=split, held=held, cyclic=cyclic
        )
        return axis


@dataclass(frozen=True)
class Tensor:
    """A tensor whose local piece a buffer holds: its axes, and where the piece starts.

    Attributes:
        axes: The tensor's axes, normalized to a tuple; their order is the order peers list them
            in, not the order of their bytes, which their strides give.
        offset: Non-negative byte offset of the local piece within its buffer.
        name: Tells the tensors of one buffer apart; peers match tensors on it.
        dtype: The type of the tensor's elements by name, such as ``"bfloat16"``, or None where
            the layout states none; peers whose elements differ in type do not correspond
            however their bytes line up.

    Raises:
        ValueError: If there is no axis, two axes share a name, a parallel dimension splits two
            axes, or offset is not a non-negative integer.
    """

    axes: tuple[Axis, ...]
    offset: int = 0
    name: str = ""
    dtype: str | None = None

    def __post_init__(self) -> None:
        axes = tuple(self.axes)
        if not axes or not all(isinstance(a, Axis) for a in axes):
            raise ValueError(f"a tensor has one or more Axis values, got {axes!r}")
        names = [a.name for a in axes]
        if len(set(names)) != len(names):
            raise ValueError(f"axis names repeat: {names}")
        dims = [dim for a in axes for dim in a.split]
        if len(set(dims)) != len(dims):
            raise ValueError(f"a parallel dimension splits two axes: {dims}")
        object.__setattr__(self, "axes", axes)
        object.__setattr__(self, "offset", _integer(self.offset, "tensor offset"))
        object.__setattr__(self, "name", str(self.name))
        if self.dtype is not None:
            object.__setattr__(self, "dtype", str(self.dtype))

    @property
    def split(self) -> bool:
        """Whether some axis spreads over parallel dimensions."""
        return any(a.split for a in self.axes)

    @classmethod
    def unchecked(
        cls, axes: tuple[Axis, ...], offset: int = 0, name: str = "", dtype: str | None = None
    ) -> Tensor:
        """The constructor's tensor of these values, without its checks and conversions; see
        ``PageLayout.unchecked``."""
        tensor = object.__new__(cls)
        tensor.__dict__.update(axes=tuple(axes), offset=offset, name=name, dtype=dtype)
        return tensor


def held(axis: Axis, topology: RankTopology) -> Progression:
    """The global indices of ``axis`` that the rank at ``topology`` holds, in local order."""
    if not axis.split:
        return Progression(0, 1, axis.extent)
    n, r = grid_place(topology, axis.split)
    if axis.cyclic:
        return Progression(r, n, len(range(r, axis.extent, n)))
    start = r * axis.extent // n
    if start + axis.held > axis.extent:
        raise ValueError(
            f"axis {axis.name}: rank {r} of {n} holds {axis.held} indices from {start}, past "
            f"its extent of {axis.extent}"
        )
    return Progression(start, 1, axis.held)


def _footprint(tensor: Tensor) -> int | None:
    """Bytes from its buffer's start to the end of a tensor's local piece, or None when a cyclic
    axis leaves the count of its indices to the rank."""
    end = tensor.offset + 1
    for a in tensor.axes:
        if a.cyclic:
            return None
        end += ((a.held if a.split else a.extent) - 1) * a.stride
    return end


def replicated(size: int) -> tuple[Tensor, ...]:
    """A buffer of ``size`` bytes that every rank holds alike."""
    return (Tensor((Axis("byte", size, 1),)),)


def owned(size: int, ranks: int, split: Sequence[str] = ("tp",)) -> tuple[Tensor, ...]:
    """A buffer of ``size`` bytes that is this rank's own share of ``ranks`` ranks along ``split``,
    in a split the manager does not describe further: it corresponds only to the same rank's."""
    return (
        Tensor((Axis("shard", ranks, size, tuple(split), 1), Axis("byte", size, 1)), name="own"),
    )


def head_major(
    size: int,
    heads: int,
    total_heads: int,
    split: Sequence[str] = ("tp",),
    *,
    dtype: str | None = None,
) -> tuple[Tensor, ...]:
    """A buffer of ``heads`` heads of ``size // heads`` bytes each ([head, ...], HND), of
    ``total_heads`` split contiguously over ``split``, of elements of type ``dtype``."""
    if heads <= 0 or size % heads:
        raise ValueError(f"{size} bytes is not a whole number of {heads} heads")
    head = size // heads
    return (
        Tensor(
            (Axis("head", total_heads, head, tuple(split), heads), Axis("byte", head, 1)),
            dtype=dtype,
        ),
    )


def token_major(
    size: int,
    tokens: int,
    heads: int,
    total_heads: int,
    split: Sequence[str] = ("tp",),
    *,
    dtype: str | None = None,
) -> tuple[Tensor, ...]:
    """A buffer of ``tokens`` tokens, each ``heads`` heads of equal bytes ([token, head, ...],
    NHD), of ``total_heads`` split contiguously over ``split``, of elements of type ``dtype``."""
    if tokens <= 0 or heads <= 0 or size % (tokens * heads):
        raise ValueError(f"{size} bytes is not a whole number of {tokens} x {heads} token heads")
    head = size // (tokens * heads)
    return (
        Tensor(
            (
                Axis("token", tokens, heads * head),
                Axis("head", total_heads, head, tuple(split), heads),
                Axis("byte", head, 1),
            ),
            dtype=dtype,
        ),
    )


def sectioned(
    section_bytes: Sequence[int],
    ranks: int,
    split: Sequence[str] = ("tp",),
    *,
    dtype: str | None = None,
) -> tuple[Tensor, ...]:
    """Consecutive sections of a buffer, each the local ``1 / ranks`` of its own tensor, split
    contiguously by byte over the ``ranks`` ranks of ``split`` (Mamba2 conv state ``[x | B | C]``,
    GDN ``[Q | K | V]``), of elements of type ``dtype``."""
    out, offset = [], 0
    for i, n in enumerate(section_bytes):
        out.append(
            Tensor(
                (Axis("byte", n * ranks, 1, tuple(split), n),),
                offset=offset,
                name=f"s{i}",
                dtype=dtype,
            )
        )
        offset += n
    return tuple(out)


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

    Each buffer's ``tensors`` say how it splits over the ranks; ``shard`` is the share
    of the group's content this rank holds, by the indices its transferred buffers
    hold (``content_shard``), so ranks with equal shares hold equal bytes.

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
        model_layers: The model layer each of ``layers`` belongs to, in the same order, normalized
            to a tuple; None gives ``layers`` themselves. Layer ids numbering several caches of one
            model layer apart, such as one per attention type, all name that model layer.

    Raises:
        ValueError: If an index, layer id or model layer is not a non-negative integer, a window is
            not a positive integer, any such value is a boolean, read_window exceeds kept_window or
            is given when kept_window is None, or model_layers does not name one model layer per
            layer."""

    kind: Literal["attention", "state"]
    kept_window: int | None
    num_sink_blocks: int
    pool_group: int
    layers: tuple[int, ...]
    shard: ShardDesc = ShardDesc()
    read_window: int | None = None
    model_layers: tuple[int, ...] | None = None

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
    ) -> LayerGroupDesc:
        """The constructor's layer group of these values, without its checks and conversions; a
        read window of None still defaults to the kept window, and model layers of None to the
        layers. See ``PageLayout.unchecked``."""
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
        tensors: The tensors whose local pieces the buffer holds, and how they spread over ranks;
            normalized to a tuple. Defaults to the whole buffer, alike on every rank.
        transferred: Whether peers transfer this role. False marks local-only buffers, which still
            accompany copies of whole slots.
        expansion: Positive number of sub-pages within one page, each of
            PageLayout.tokens_per_page / expansion of the buffer's tokens; consumers accessing bytes
            within the buffer must support it.

    Raises:
        ValueError: If an index is not a non-negative integer, size or expansion is not a positive
            integer, any such value is a boolean, two tensors share a name, a tensor's piece
            reaches past size, or the pieces of several tensors do not lie back to back over the
            whole buffer."""

    layer: int
    role: str
    pool_group: int
    pool_index: int
    slot_offset: int
    size: int
    tensors: tuple[Tensor, ...] = ()
    transferred: bool = True
    expansion: int = 1

    def __post_init__(self) -> None:
        for name in ("layer", "pool_group", "pool_index", "slot_offset"):
            object.__setattr__(self, name, _integer(getattr(self, name), name))
        object.__setattr__(self, "size", _integer(self.size, "size", 1))
        object.__setattr__(self, "expansion", _integer(self.expansion, "expansion", 1))
        object.__setattr__(self, "role", str(self.role))
        tensors = tuple(self.tensors) or replicated(self.size)
        if not all(isinstance(t, Tensor) for t in tensors):
            raise ValueError(f"tensors must be Tensor values, got {tensors!r}")
        names = [t.name for t in tensors]
        if len(set(names)) != len(names):
            raise ValueError(f"buffer ({self.layer}, {self.role}): tensor names repeat: {names}")
        ends = [_footprint(t) for t in tensors]
        for t, end in zip(tensors, ends):
            if end is not None and end > self.size:
                raise ValueError(
                    f"buffer ({self.layer}, {self.role}): tensor {t.name!r} reaches byte {end} of "
                    f"a {self.size}-byte buffer"
                )
        if len(tensors) > 1 and None not in ends:
            pieces = sorted((t.offset, end) for t, end in zip(tensors, ends))
            bounds = [0, *(b for piece in pieces for b in piece), self.size]
            if any(a != b for a, b in zip(bounds[::2], bounds[1::2])):
                raise ValueError(
                    f"buffer ({self.layer}, {self.role}): sections {pieces} do not cover its "
                    f"{self.size} bytes"
                )
        object.__setattr__(self, "tensors", tensors)

    @property
    def split(self) -> bool:
        """Whether some tensor spreads over parallel dimensions; else every rank holds it alike."""
        return any(t.split for t in self.tensors)

    @classmethod
    def unchecked(
        cls,
        layer: int,
        role: str,
        pool_group: int,
        pool_index: int,
        slot_offset: int,
        size: int,
        tensors: tuple[Tensor, ...] = (),
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
            tensors=tuple(tensors) or replicated(size),
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
        rebuilds_untransferred: Whether some buffers peers do not transfer derive from the others,
            which the manager rebuilds after the native transfer imports a request's pages
            (``KVCacheManagerV2._page_derived_roles``). The lender, which invokes no rebuild, then
            lends every buffer, and where the source holds none of the derived ones, the importer
            invokes the rebuild (``attach_lender``).
        request_rebuilt_roles: Roles whose buffers this rank, receiving, rebuilds for some requests
            rather than receiving them (``KVCacheManagerV2._page_request_rebuilt_roles``), so a
            peer may hold none of them for a model layer it holds.

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
    rebuilds_untransferred: bool = False
    request_rebuilt_roles: frozenset[str] = frozenset()

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
        object.__setattr__(self, "rebuilds_untransferred", bool(self.rebuilds_untransferred))
        object.__setattr__(
            self, "request_rebuilt_roles", frozenset(str(r) for r in self.request_rebuilt_roles)
        )
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
        rebuilds_untransferred: bool = False,
        request_rebuilt_roles: frozenset[str] = frozenset(),
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
            rebuilds_untransferred=rebuilds_untransferred,
            request_rebuilt_roles=frozenset(request_rebuilt_roles),
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


def _split_holdings(buffers: Sequence[BufferDesc], topology: RankTopology) -> tuple:
    """The indices the rank at ``topology`` holds of every split axis of the transferred buffers
    among ``buffers``, as ((layer, role, tensor, axis), progression) pairs in that key order."""
    out = []
    for buffer in buffers:
        if not buffer.transferred or not buffer.split:
            continue
        for tensor in buffer.tensors:
            for axis in tensor.axes:
                if axis.split:
                    key = (buffer.layer, buffer.role, tensor.name, axis.name)
                    out.append((key, held(axis, topology)))
    return tuple(sorted(out, key=lambda item: item[0]))


def _held_indices(layout: PageLayout, desc: LayerGroupDesc) -> tuple | None:
    """The indices this rank holds of every split axis of the group's transferred buffers, by
    (layer, role, tensor, axis); None when no transferred buffer splits."""
    buffers = [
        b for b in layout.buffers if b.pool_group == desc.pool_group and b.layer in desc.layers
    ]
    return _split_holdings(buffers, layout.topology) or None


def content_shard(buffers: Sequence[BufferDesc], topology: RankTopology) -> ShardDesc:
    """The share of a layer group's content the rank at ``topology`` holds, read from the group's
    ``buffers``: ranks of its grid hold the same share exactly when they hold the same indices of
    every split axis of the transferred buffers, and shares are numbered in the order of the
    indices they hold. The whole content when every rank holds alike.

    The numbering names the same indices under every rank count that divides, or is divided by,
    each split axis's extent, as attention heads and TP sizes do. Where an axis's extent and rank
    count do not divide one another, each rank of the dimensions the group splits over is a share
    of its own, so that no other rank count can name other indices alike."""

    def share(place: RankTopology) -> tuple:
        return tuple(
            (key, (p.start, p.step, p.count)) for key, p in _split_holdings(buffers, place)
        )

    mine = share(topology)
    if not mine:
        return ShardDesc()
    axes = [a for b in buffers if b.transferred for t in b.tensors for a in t.axes if a.split]
    counts = [grid_place(topology, a.split)[0] for a in axes]
    if any(a.extent % n and n % a.extent for a, n in zip(axes, counts)):
        dims = [dim for dim in GRID_DIMENSIONS if any(dim in a.split for a in axes)]
        size, rank = grid_place(topology, dims)
        return ShardDesc(size, rank) if size > 1 else ShardDesc()
    shares = {
        share(
            dataclasses.replace(
                topology,
                tp_rank=tp_rank,
                cp_rank=cp_rank,
                dp_rank=tp_rank if topology.attention_dp else topology.dp_rank,
            )
        )
        for tp_rank in range(topology.tp_size)
        for cp_rank in range(topology.cp_size)
    }
    if len(shares) == 1:
        return ShardDesc()
    return ShardDesc(len(shares), sorted(shares).index(mine))


def held_bytes(tensor: Tensor, topology: RankTopology) -> int:
    """Bytes of the tensor's local piece on the rank at ``topology``: one per index it holds of
    every axis together."""
    return math.prod(held(axis, topology).count for axis in tensor.axes)


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
    attention data-parallel rank); hold the same pages (under Helix an attention group's pages split
    by context-parallel rank, while a state group's rows do not); hold the same layers, in a layer
    group of the same kind; and hold the same share of the group's content: a buffer every rank
    holds alike is the same on every such rank, and split buffers match when the ranks hold the
    same indices of every split axis. Without split buffers, the published content shard is the
    only basis for matching.

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
    paged_by_context = desc.kind == "attention"
    held_indices = None if replicated else _held_indices(layout, desc)
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
        if not replicated:
            other_held = _held_indices(other, match)
            if held_indices is None and other_held is None:
                if match.shard != desc.shard:
                    continue
            elif other_held != held_indices:
                continue
        out.append(other_rank)
    return tuple(out)


def _axis_to_dict(axis: Axis) -> dict:
    return {
        "name": axis.name,
        "extent": axis.extent,
        "stride": axis.stride,
        "split": list(axis.split),
        "held": axis.held,
        "cyclic": axis.cyclic,
    }


def layout_to_dict(layout: PageLayout) -> dict:
    """``layout`` as plain values (dicts, lists, ints, strings) for a peer, which
    ``layout_from_dict`` reads back; the lender's region maps stay local and are left out."""
    return {
        "tokens_per_page": layout.tokens_per_page,
        "pool_slot_bytes": {int(g): list(w) for g, w in layout.pool_slot_bytes.items()},
        "layer_groups": [
            {
                "kind": d.kind,
                "kept_window": d.kept_window,
                "num_sink_blocks": d.num_sink_blocks,
                "pool_group": d.pool_group,
                "layers": list(d.layers),
                "shard": [d.shard.count, d.shard.index],
                "read_window": d.read_window,
                "model_layers": list(d.model_layers),
            }
            for d in layout.layer_groups
        ],
        "buffers": [
            {
                "layer": b.layer,
                "role": b.role,
                "pool_group": b.pool_group,
                "pool_index": b.pool_index,
                "slot_offset": b.slot_offset,
                "size": b.size,
                "tensors": [
                    {
                        "axes": [_axis_to_dict(a) for a in t.axes],
                        "offset": t.offset,
                        "name": t.name,
                        "dtype": t.dtype,
                    }
                    for t in b.tensors
                ],
                "transferred": b.transferred,
                "expansion": b.expansion,
            }
            for b in layout.buffers
        ],
        "topology": dataclasses.asdict(layout.topology),
        "total_kv_heads_by_layer": {int(k): v for k, v in layout.total_kv_heads_by_layer.items()},
        "dtype": layout.dtype,
        "quant_mode": layout.quant_mode,
        "rebuilds_untransferred": layout.rebuilds_untransferred,
        "request_rebuilt_roles": sorted(layout.request_rebuilt_roles),
    }


def layout_from_dict(data: Mapping) -> PageLayout:
    """The layout ``layout_to_dict`` wrote, checked as the constructors check it."""
    return PageLayout(
        tokens_per_page=data["tokens_per_page"],
        pool_slot_bytes={int(g): tuple(w) for g, w in data["pool_slot_bytes"].items()},
        layer_groups=tuple(
            LayerGroupDesc(
                kind=d["kind"],
                kept_window=d["kept_window"],
                num_sink_blocks=d["num_sink_blocks"],
                pool_group=d["pool_group"],
                layers=tuple(d["layers"]),
                shard=ShardDesc(*d["shard"]),
                read_window=d["read_window"],
                model_layers=tuple(d["model_layers"]),
            )
            for d in data["layer_groups"]
        ),
        buffers=tuple(
            BufferDesc(
                layer=b["layer"],
                role=b["role"],
                pool_group=b["pool_group"],
                pool_index=b["pool_index"],
                slot_offset=b["slot_offset"],
                size=b["size"],
                tensors=tuple(
                    Tensor(
                        tuple(
                            Axis(
                                a["name"],
                                a["extent"],
                                a["stride"],
                                tuple(a["split"]),
                                a["held"],
                                a["cyclic"],
                            )
                            for a in t["axes"]
                        ),
                        t["offset"],
                        t["name"],
                        t.get("dtype"),
                    )
                    for t in b["tensors"]
                ),
                transferred=b["transferred"],
                expansion=b["expansion"],
            )
            for b in data["buffers"]
        ),
        topology=RankTopology(**data["topology"]),
        total_kv_heads_by_layer={int(k): v for k, v in data["total_kv_heads_by_layer"].items()},
        dtype=data["dtype"],
        quant_mode=data["quant_mode"],
        rebuilds_untransferred=data["rebuilds_untransferred"],
        request_rebuilt_roles=frozenset(data["request_rebuilt_roles"]),
    )
