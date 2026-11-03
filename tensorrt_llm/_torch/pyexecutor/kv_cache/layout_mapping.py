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
"""How two ranks' pages correspond, read from their page layouts alone.

Every buffer is the local pieces of tensors whose axes say which global indices a rank holds
(``page_layout.held``). For a source rank and a destination rank this module gives, per buffer
both hold, the byte runs the source sends: the indices both hold, of which the source is the
elected sender, as runs that are contiguous on both sides. Nothing here knows a cache kind or a
model.

Pairing and election: ranks split each parallel dimension into equal shares in rank order, and a
destination rank pairs with the source ranks whose share overlaps its own (``fan``); a
destination hears from no other rank, so the native transfer contacts exactly these. An index of
an axis split over source dimensions is sent by one of the paired source ranks holding it; a
source dimension no axis of the tensor splits holds the same bytes on each of its ranks, and the
destination takes them from one rank of its fan. Of several such ranks, the one at position
``key`` modulo their count sends, ``key`` being the destination's attention data-parallel rank,
so that destinations of several data-parallel groups spread over the sources. A destination
holding an index no paired source rank holds, as an uneven head split or a cyclic split between
different rank counts can make, is refused rather than left without it.

Token positions: runs are fixed per peer and only add each page's address, except for a buffer
holding an axis of token positions (``AxisKind.positional``), whose place of a position depends
on each side. Its runs are those of one position, and ``page_runs`` moves them to
the places of the positions a page has written within a request's transfer window
(``TransferWindow``); a transfer applies ``page_runs`` to every buffer, getting the fixed runs back
for all others. Which slots of a request pair in one transfer is ``pair_pages``.

Kinds: each axis kind registers the indices a source rank is elected to send (``owned_indices``)
and, if positional, how its runs move with position (``position_rule``); each layer-group kind
registers which slots pair (``pair_slots``). A new kind in ``page_layout`` registers its rules
here and leaves every function above unchanged.
"""

from __future__ import annotations

import functools
import itertools
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .page_layout import (
    GRID_DIMENSIONS,
    Axis,
    AxisKind,
    BufferDesc,
    Contiguous,
    Cyclic,
    GroupKind,
    Paged,
    PageLayout,
    PerRequest,
    Progression,
    RankTopology,
    Tensor,
    grid_place,
    held,
)

__all__ = [
    "BufferRuns",
    "PagePairs",
    "PositionRule",
    "TransferWindow",
    "check_compatible",
    "fan",
    "layer_group_of",
    "map_buffers",
    "merge_runs",
    "owned_indices",
    "page_runs",
    "pair_pages",
    "pair_slots",
    "paired_ordinals",
    "position_rule",
    "sent_bytes",
]


def fan(src_size: int, dst_size: int, dst_rank: int) -> range:
    """The source ranks, of ``src_size`` along one parallel dimension, whose equal share of it
    overlaps the share of destination rank ``dst_rank`` of ``dst_size``."""
    return range(dst_rank * src_size // dst_size, -(-(dst_rank + 1) * src_size // dst_size))


@dataclass(frozen=True)
class TransferWindow:
    """What one transfer of a request covers, as the request's token positions.

    Attributes:
        start: First token position the transfer moves.
        end: Token position the transfer reaches, exclusive.
        reused: Token positions from the request's start that the destination already holds, as
            a prefix it reuses; their pages are not paired.
    """

    start: int
    end: int
    reused: int = 0


class PositionRule:
    """How the runs a buffer moves of a page depend on the token positions the page has written:
    the runs are those of one position, moved to each position's place on either side.

    Subclasses are frozen dataclasses, so that buffers whose runs move by equal rules share one
    computation per page.
    """

    @property
    def kept(self) -> int:
        """The most positions of one page that a transfer moves."""
        raise NotImplementedError

    def positions(self, page: int, window: TransferWindow) -> np.ndarray:
        """The token positions of page ``page`` (its position among a request's pages, from 0)
        that a transfer over ``window`` moves, as int64."""
        raise NotImplementedError

    def offsets(self, positions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """The byte offset of each position's place from its buffer's position-0 runs, on the
        source and on the destination."""
        raise NotImplementedError


@dataclass(frozen=True, eq=False)
class BufferRuns:
    """The bytes a source rank sends of one buffer the destination rank also holds: run ``i`` is
    ``sizes[i]`` bytes from ``src_offsets[i]`` of the source page's pool slot to ``dst_offsets[i]``
    of the destination's. For a buffer of token positions, these are the runs of the position at
    index 0 on both sides, and ``page_runs`` gives what a page moves.

    Attributes:
        src: The source rank's buffer.
        dst: The destination rank's buffer of the same layer and role.
        src_offsets: int64 offsets within the source buffer's pool slot.
        dst_offsets: int64 offsets within the destination buffer's pool slot.
        sizes: int64 run lengths in bytes.
        by_position: How the runs it moves of a page depend on the token positions the page has
            written (``page_runs``): None when they do not, and equal for buffers whose runs
            depend on them alike.
    """

    src: BufferDesc
    dst: BufferDesc
    src_offsets: np.ndarray
    dst_offsets: np.ndarray
    sizes: np.ndarray
    by_position: PositionRule | None = None

    @property
    def fixed(self) -> bool:
        """Whether it moves the same runs of every page, which ``page_runs`` then returns."""
        return self.by_position is None

    @property
    def page_bytes(self) -> int:
        """The most bytes it moves of one page."""
        sent = int(self.sizes.sum())
        return sent if self.by_position is None else sent * self.by_position.kept


def page_runs(
    runs: BufferRuns, page: int, window: TransferWindow
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The runs that move page ``page`` (its position among a request's pages, from 0) in a
    transfer over ``window``: ``runs``' own for a buffer laid out alike for every position; else
    those of each position the buffer's rule moves (``PositionRule``), from its place on the
    source to its place on the destination."""
    rule = runs.by_position
    if rule is None:
        return runs.src_offsets, runs.dst_offsets, runs.sizes
    src_at, dst_at = rule.offsets(rule.positions(page, window))
    src = src_at[:, None] + runs.src_offsets[None, :]
    dst = dst_at[:, None] + runs.dst_offsets[None, :]
    sizes = np.broadcast_to(runs.sizes, src.shape).ravel()
    return merge_runs(src.ravel(), dst.ravel(), sizes)


def _dims(topology: RankTopology) -> dict[str, tuple[int, int]]:
    """(size, rank) of the rank along each parallel dimension."""
    return {dim: grid_place(topology, (dim,)) for dim in GRID_DIMENSIONS}


def _fans(src: RankTopology, dst: RankTopology) -> dict[str, range]:
    src_dims, dst_dims = _dims(src), _dims(dst)
    return {
        dim: fan(src_dims[dim][0], dst_dims[dim][0], dst_dims[dim][1]) for dim in GRID_DIMENSIONS
    }


def _flat(coords: Sequence[int], sizes: Sequence[int]) -> int:
    rank = 0
    for c, n in zip(coords, sizes):
        rank = rank * n + c
    return rank


def _elected(candidates: Sequence[int], key: int) -> int:
    return candidates[key % len(candidates)]


@functools.singledispatch
def owned_indices(
    kind: AxisKind,
    axis: Axis,
    src: RankTopology,
    fans: dict[str, range],
    key: int,
    wanted: Progression,
) -> list[Progression]:
    """The runs of indices of ``axis``, of kind ``kind``, that the source rank at ``src`` sends:
    of the indices it holds, those it is the elected sender of among the source ranks paired with
    the destination (``fans``: each parallel dimension's paired source ranks; ``key``: breaks ties
    as ``map_buffers`` says). A kind registers its own; one that spreads over no dimension, as
    the default, sends every index it holds.

    Raises:
        ValueError: If the destination holds (``wanted``) an index no paired rank holds.
    """
    if kind.split:
        raise ValueError(
            f"axis {axis.name!r}: kind {kind.name!r} spreads over {kind.split} but registers no "
            "owned_indices"
        )
    return [held(axis, src)]


def _paired(split: Sequence[str], src: RankTopology, fans: dict[str, range]) -> list[int]:
    """The source ranks paired with the destination, by flat place in the grid of ``split``."""
    sizes = [_dims(src)[dim][0] for dim in split]
    return sorted({_flat(c, sizes) for c in itertools.product(*(fans[dim] for dim in split))})


@owned_indices.register(Cyclic)
def _owned_cyclic(kind, axis, src, fans, key, wanted):
    # One rank holds each index: index i lies on rank i modulo n.
    n, r = grid_place(src, kind.split)
    paired = _paired(kind.split, src, fans)
    for rank in range(n):
        if rank not in paired and rank < axis.extent:
            _refuse_unpaired(axis, Progression(rank, n, len(range(rank, axis.extent, n))), wanted)
    return [held(axis, src)] if r in paired else []


@owned_indices.register(Contiguous)
def _owned_contiguous(kind, axis, src, fans, key, wanted):
    n, r = grid_place(src, kind.split)
    paired = _paired(kind.split, src, fans)
    runs = [(rank * axis.extent // n, rank * axis.extent // n + kind.held) for rank in range(n)]
    bounds = sorted({0, axis.extent} | {min(b, axis.extent) for run in runs for b in run})
    out: list[Progression] = []
    for lo, hi in zip(bounds, bounds[1:]):
        candidates = [rank for rank in paired if runs[rank][0] <= lo and hi <= runs[rank][1]]
        if not candidates:
            _refuse_unpaired(axis, Progression(lo, 1, hi - lo), wanted)
            continue
        if _elected(candidates, key) != r:
            continue
        if out and out[-1].last + 1 == lo:
            out[-1] = Progression(out[-1].start, 1, out[-1].count + hi - lo)
        else:
            out.append(Progression(lo, 1, hi - lo))
    return out


def _refuse_unpaired(axis: Axis, indices: Progression, wanted: Progression) -> None:
    """Raise if the destination holds any of ``indices``, which no paired source rank holds."""
    missing = indices.intersect(wanted)
    if missing is not None:
        raise ValueError(
            f"axis {axis.name!r} (extent {axis.extent}, split over {axis.split}): index "
            f"{missing.start} of the destination is held by no source rank it pairs with"
        )


def _sends_replicas(tensor: Tensor, src: RankTopology, fans: dict[str, range], key: int) -> bool:
    """Whether the source rank is the elected one along the dimensions no axis of the tensor
    splits, whose ranks hold its bytes alike."""
    free = [dim for dim in GRID_DIMENSIONS if all(dim not in a.split for a in tensor.axes)]
    if not free:
        return True
    dims = _dims(src)
    mine = tuple(dims[dim][1] for dim in free)
    return _elected(list(itertools.product(*(fans[dim] for dim in free))), key) == mine


def _tensor_runs(
    src_buffer: BufferDesc,
    src_tensor: Tensor,
    src: RankTopology,
    dst_buffer: BufferDesc,
    dst_tensor: Tensor,
    dst: RankTopology,
    fans: dict[str, range],
    key: int,
):
    """(source offsets, destination offsets, run size) of what the source sends of one tensor, a
    list of them when its boxes run in different sizes, or None when it sends nothing."""
    dst_axes = {a.name: a for a in dst_tensor.axes}
    if set(dst_axes) != {a.name for a in src_tensor.axes}:
        raise ValueError(
            f"buffer ({src_buffer.layer}, {src_buffer.role}) tensor {src_tensor.name!r}: axes "
            f"{[a.name for a in src_tensor.axes]} against {list(dst_axes)}"
        )
    if not _sends_replicas(src_tensor, src, fans, key):
        return None
    # Per axis, the runs of indices the source sends that the destination holds.
    regions = []
    for src_axis in src_tensor.axes:
        dst_axis = dst_axes[src_axis.name]
        if src_axis.extent != dst_axis.extent:
            raise ValueError(
                f"buffer ({src_buffer.layer}, {src_buffer.role}) tensor {src_tensor.name!r} axis "
                f"{src_axis.name}: extent {src_axis.extent} against {dst_axis.extent}"
            )
        src_held, dst_held = held(src_axis, src), held(dst_axis, dst)
        parts = []
        for owned in owned_indices(src_axis.kind, src_axis, src, fans, key, dst_held):
            part = owned.intersect(src_held)
            part = None if part is None else part.intersect(dst_held)
            if part is not None:
                parts.append(part)
        if not parts:
            return None
        regions.append((src_axis, dst_axis, src_held, dst_held, parts))
    out = [
        _box_runs(src_buffer, src_tensor, dst_buffer, dst_tensor, regions, choice)
        for choice in itertools.product(*(r[4] for r in regions))
    ]
    sizes = {size for _, _, size in out}
    if len(sizes) == 1:
        return (
            np.concatenate([o[0] for o in out]),
            np.concatenate([o[1] for o in out]),
            sizes.pop(),
        )
    return out


def _box_runs(
    src_buffer: BufferDesc,
    src_tensor: Tensor,
    dst_buffer: BufferDesc,
    dst_tensor: Tensor,
    regions: list,
    choice: tuple[Progression, ...],
) -> tuple[np.ndarray, np.ndarray, int]:
    """The runs of one box of indices, one progression per axis."""
    src_base = src_buffer.slot_offset + src_tensor.offset
    dst_base = dst_buffer.slot_offset + dst_tensor.offset
    loops = []
    for (src_axis, dst_axis, src_held, dst_held, _), region in zip(regions, choice):
        src_base += (region.start - src_held.start) // src_held.step * src_axis.stride
        dst_base += (region.start - dst_held.start) // dst_held.step * dst_axis.stride
        if region.count > 1:
            loops.append(
                (
                    region.count,
                    region.step // src_held.step * src_axis.stride,
                    region.step // dst_held.step * dst_axis.stride,
                )
            )
    # Fold into one run every loop that continues the run on both sides.
    size = 1
    while True:
        for i, (count, src_step, dst_step) in enumerate(loops):
            if src_step == size and dst_step == size:
                size *= count
                del loops[i]
                break
        else:
            break
    src_offsets = np.array([src_base], dtype=np.int64)
    dst_offsets = np.array([dst_base], dtype=np.int64)
    # Outermost destination loop first, so runs come in destination order.
    for count, src_step, dst_step in sorted(loops, key=lambda loop: -loop[2]):
        index = np.arange(count, dtype=np.int64)
        src_offsets = (src_offsets[:, None] + index[None, :] * src_step).ravel()
        dst_offsets = (dst_offsets[:, None] + index[None, :] * dst_step).ravel()
    return src_offsets, dst_offsets, size


def merge_runs(
    src_offsets: np.ndarray, dst_offsets: np.ndarray, sizes: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The runs with each that continues the one before it on both sides merged into it, in source
    order."""
    if src_offsets.size == 0:
        return src_offsets, dst_offsets, sizes
    order = np.argsort(src_offsets, kind="stable")
    src_offsets, dst_offsets, sizes = src_offsets[order], dst_offsets[order], sizes[order]
    starts = np.ones(src_offsets.size, dtype=bool)
    starts[1:] = (src_offsets[1:] != src_offsets[:-1] + sizes[:-1]) | (
        dst_offsets[1:] != dst_offsets[:-1] + sizes[:-1]
    )
    first = np.flatnonzero(starts)
    merged = np.add.reduceat(sizes, first)
    return src_offsets[first], dst_offsets[first], merged


def _pinned(tensor: Tensor) -> Tensor:
    """The tensor with its axis of token positions narrowed to index 0, so that its runs are those
    of one position."""
    if tensor.positional is None:
        return tensor
    axes = tuple(
        Axis.unchecked(a.name, 1, a.stride) if a.kind.positional else a for a in tensor.axes
    )
    return Tensor.unchecked(axes, tensor.offset, tensor.name, tensor.dtype)


@functools.singledispatch
def position_rule(
    kind: AxisKind, src_axis: Axis, dst_axis: Axis, tokens_per_page: int
) -> PositionRule:
    """How the runs of a buffer whose axis of token positions is ``src_axis`` on the source and
    ``dst_axis`` on the destination move with position, for pages of ``tokens_per_page`` tokens.
    A positional axis kind registers its own.

    Raises:
        ValueError: If ``kind`` registers no rule.
    """
    raise ValueError(f"axis {src_axis.name!r}: kind {kind.name!r} registers no position_rule")


def _by_position(
    src: PageLayout,
    src_buffer: BufferDesc,
    dst: PageLayout,
    dst_buffer: BufferDesc,
    groups: tuple[dict[tuple[int, int], int], dict[tuple[int, int], int]],
) -> PositionRule | None:
    """How a buffer both ranks transfer moves with token position, or None when it holds no axis
    of token positions; ``groups`` are both layouts' ``layer_group_of``.

    Raises:
        ValueError: If one side's buffer has an axis of token positions and the other's has none,
            the axes are named or kinded apart, or the axis lies in a layer group holding one
            slot per request.
    """
    src_axis, dst_axis = src_buffer.positional, dst_buffer.positional
    if src_axis is None and dst_axis is None:
        return None
    where = f"buffer ({src_buffer.layer}, {src_buffer.role})"
    if (
        src_axis is None
        or dst_axis is None
        or src_axis.name != dst_axis.name
        or type(src_axis.kind) is not type(dst_axis.kind)
    ):
        raise ValueError(
            f"{where}: token positions {src_axis and (src_axis.name, src_axis.kind.name)!r} "
            f"against {dst_axis and (dst_axis.name, dst_axis.kind.name)!r}"
        )
    for page, buffer, group_of in ((src, src_buffer, groups[0]), (dst, dst_buffer, groups[1])):
        group = page.layer_groups[group_of[buffer.pool_group, buffer.layer]]
        if group.kind.per_request:
            raise ValueError(
                f"{where}: a {src_axis.kind.name} of token positions in a {group.kind} layer "
                "group, which holds one slot per request"
            )
    return position_rule(src_axis.kind, src_axis, dst_axis, src.tokens_per_page)


def map_buffers(src: PageLayout, dst: PageLayout, *, key: int | None = None) -> list[BufferRuns]:
    """What the rank of ``src`` sends the rank of ``dst``: for each buffer both transfer (same
    global layer and role), the runs of its tensors the source is the elected sender of. Buffers
    that move nothing are left out. A buffer of token positions gives the runs of one position, to
    be moved with ``page_runs``.

    Args:
        src: The source rank's page layout.
        dst: The destination rank's page layout.
        key: Breaks ties between source ranks holding the same bytes; the destination's attention
            data-parallel rank by default.

    Raises:
        ValueError: If two matched buffers' tensors or axes differ in name or extent other than an
            axis of token positions', one has such an axis the other has not, such an axis lies in
            a group holding one slot per request, or the destination holds an index no source rank
            it pairs with holds.
    """
    key = dst.topology.dp_rank if key is None else key
    fans = _fans(src.topology, dst.topology)
    groups = (layer_group_of(src), layer_group_of(dst))
    dst_buffers = {(b.layer, b.role): b for b in dst.buffers if b.transferred}
    out = []
    for src_buffer in src.buffers:
        dst_buffer = dst_buffers.get((src_buffer.layer, src_buffer.role))
        if dst_buffer is None or not src_buffer.transferred:
            continue
        dst_tensors = {t.name: t for t in dst_buffer.tensors}
        if set(dst_tensors) != {t.name for t in src_buffer.tensors}:
            raise ValueError(
                f"buffer ({src_buffer.layer}, {src_buffer.role}): tensors "
                f"{[t.name for t in src_buffer.tensors]} against {list(dst_tensors)}"
            )
        rule = _by_position(src, src_buffer, dst, dst_buffer, groups)
        parts = []
        for src_tensor in src_buffer.tensors:
            runs = _tensor_runs(
                src_buffer,
                _pinned(src_tensor),
                src.topology,
                dst_buffer,
                _pinned(dst_tensors[src_tensor.name]),
                dst.topology,
                fans,
                key,
            )
            if isinstance(runs, list):
                parts.extend(runs)
            elif runs is not None:
                parts.append(runs)
        if not parts:
            continue
        src_offsets = np.concatenate([p[0] for p in parts])
        dst_offsets = np.concatenate([p[1] for p in parts])
        sizes = np.concatenate([np.full(p[0].size, p[2], dtype=np.int64) for p in parts])
        out.append(
            BufferRuns(
                src_buffer,
                dst_buffer,
                *merge_runs(src_offsets, dst_offsets, sizes),
                by_position=rule,
            )
        )
    return out


def _axes_summary(tensor: Tensor) -> dict[str, object]:
    """Each axis' extent by name, or its kind's name for an axis of token positions."""
    return {a.name: a.kind.name if a.kind.positional else a.extent for a in tensor.axes}


@dataclass(frozen=True, eq=False)
class PagePairs:
    """The slots one transfer of a request pairs in a layer group: source slot ``src_slots[i]``
    sends into destination slot ``dst_slots[i]``.

    Attributes:
        src_slots: int64 slot indices on the source.
        dst_slots: int64 slot indices on the destination.
        pages: int64 position of each pair's page among the request's pages, from 0, for runs
            that move by token position; None for a group holding one slot per request.
    """

    src_slots: np.ndarray
    dst_slots: np.ndarray
    pages: np.ndarray | None = None


def paired_ordinals(src_ids: np.ndarray, dst_ids: np.ndarray, start: int = 0) -> np.ndarray:
    """The indices of two block tables, both indexed by page and -1 where a side holds no block,
    at which both hold one, from index ``start`` on.

    A block moves only where both sides hold it, so a sender that has not evicted yet, a receiver
    that already reused a prefix, a receiver with a wider speculative window, or a sender chunk
    that blanks everything outside its range all fall out of the same intersection.

    Raises:
        ValueError: If the tables differ in length.
    """
    if src_ids.size != dst_ids.size:
        raise ValueError(
            f"positional block tables differ in length: src={src_ids.size} dst={dst_ids.size}; "
            "both sides must span the same prompt"
        )
    mask = (src_ids >= 0) & (dst_ids >= 0)
    if start > 0:
        mask[:start] = False
    return np.flatnonzero(mask)


@functools.singledispatch
def pair_slots(
    kind: GroupKind,
    src: PageLayout,
    src_ids: np.ndarray,
    dst: PageLayout,
    dst_ids: np.ndarray,
    window: TransferWindow,
) -> PagePairs:
    """The slots of a layer group of kind ``kind`` that one transfer over ``window`` pairs, from
    the source's slot ids ``src_ids`` and the destination's ``dst_ids`` for the request. A
    layer-group kind registers its own.

    Raises:
        ValueError: If ``kind`` registers no pairing.
    """
    raise ValueError(f"layer-group kind {kind.name!r} registers no pair_slots")


_NO_SLOTS = np.zeros(0, dtype=np.int64)


@pair_slots.register(Paged)
def _pair_paged(kind, src, src_ids, dst, dst_ids, window):
    src_ids = np.asarray(src_ids, dtype=np.int64)
    dst_ids = np.asarray(dst_ids, dtype=np.int64)
    # Helix: the destination's CP rank r of n holds pages r, r + n, ... of the source's, which
    # holds them all; reuse is refused under Helix.
    cp_size, cp_rank = dst.topology.cp_size, dst.topology.cp_rank
    helix = cp_size > 1 and src.topology.cp_size == 1
    if helix:
        src_ids = src_ids[cp_rank::cp_size]
    paired = paired_ordinals(src_ids, dst_ids, window.reused // src.tokens_per_page)
    pages = paired * cp_size + cp_rank if helix else paired
    return PagePairs(src_ids[paired], dst_ids[paired], pages)


@pair_slots.register(PerRequest)
def _pair_per_request(kind, src, src_ids, dst, dst_ids, window):
    if len(src_ids) == 0 or len(dst_ids) == 0:
        return PagePairs(_NO_SLOTS, _NO_SLOTS)
    return PagePairs(
        np.asarray(src_ids[:1], dtype=np.int64), np.asarray(dst_ids[:1], dtype=np.int64)
    )


def pair_pages(
    src: PageLayout,
    src_group: int,
    src_ids: np.ndarray,
    dst: PageLayout,
    dst_group: int,
    dst_ids: np.ndarray,
    window: TransferWindow,
) -> PagePairs:
    """The slots one transfer over ``window`` pairs between layer group ``src_group`` of ``src``,
    holding the request in slots ``src_ids``, and ``dst_group`` of ``dst``, holding it in
    ``dst_ids``: as the source group's kind registers (``pair_slots``).

    Raises:
        ValueError: If the groups differ in kind, or as the kind's pairing does.
    """
    kind = src.layer_groups[src_group].kind
    if dst.layer_groups[dst_group].kind != kind:
        raise ValueError(
            f"layer group {src_group} of {kind} against {dst_group} of "
            f"{dst.layer_groups[dst_group].kind}"
        )
    return pair_slots(kind, src, src_ids, dst, dst_ids, window)


def layer_group_of(layout: PageLayout) -> dict[tuple[int, int], int]:
    """(pool group, global layer) -> index of the layer group holding that layer's buffers there.

    Raises:
        ValueError: If a layer lies in two layer groups of one pool group.
    """
    out: dict[tuple[int, int], int] = {}
    for index, group in enumerate(layout.layer_groups):
        for layer in group.layers:
            if out.setdefault((group.pool_group, layer), index) != index:
                raise ValueError(
                    f"layer {layer} lies in layer groups {out[group.pool_group, layer]} and "
                    f"{index} of pool group {group.pool_group}"
                )
    return out


def _transferred(layout: PageLayout) -> dict[tuple[int, str], BufferDesc]:
    return {(b.layer, b.role): b for b in layout.buffers if b.transferred}


def check_compatible(a: PageLayout, b: PageLayout) -> None:
    """Check that two ranks' pages correspond, whichever sends.

    Each role of a model layer both ranks transfer goes by the same global layers on both, so ranks
    pair a model layer's buffers with its own however their pipeline stages split the model. A
    role of a model layer both ranks hold, through any of its buffers whether they move or not,
    that one rank alone transfers, whichever sends, is refused unless that rank rebuilds it per
    request (``PageLayout.request_rebuilt_roles``: CSA2's generation workers rebuild the sliding
    caches a context-only worker keeps none of past its split), rather than left unwritten.
    On the layers both hold, every buffer either rank transfers the other transfers too, but for
    such a rebuilt role, as the same tensors: the same names, axes and extents over all ranks,
    however each splits them, as each axis' kind judges (``AxisKind.corresponds``), and elements of
    the same type where the tensors state one, in pages
    of as many sub-pages, in layer groups of the same kind. The KV data is of one element type and
    quantization. Pages pair one to one, so where both transfer buffers of paged groups (not
    ``GroupKind.per_request``) their pages hold as many tokens.

    How many tokens each side keeps of a windowed group is not compared: retention margins (reuse
    backoff, draft tokens) legitimately differ between the sides, and a transfer moves the blocks
    both still hold. Each side's attention window apart from those margins is not in the layout.

    Raises:
        ValueError: Naming every difference found.
    """
    problems = []
    if a.dtype != b.dtype:
        problems.append(f"KV elements of type {a.dtype} against {b.dtype}")
    if a.quant_mode != b.quant_mode:
        problems.append(f"KV quantization {a.quant_mode} against {b.quant_mode}")
    layers = {x.layer for x in a.buffers} & {x.layer for x in b.buffers}
    ta, tb = _transferred(a), _transferred(b)
    group_a, group_b = layer_group_of(a), layer_group_of(b)
    paged = False
    for key in sorted(k for k in set(ta) | set(tb) if k[0] in layers):
        x, y = ta.get(key), tb.get(key)
        if x is None or y is None:
            if key[1] not in (a if y is None else b).request_rebuilt_roles:
                problems.append(f"buffer {key} is transferred by one rank only")
            continue
        if x.expansion != y.expansion:
            problems.append(f"buffer {key}: {x.expansion} sub-pages against {y.expansion}")
        xs, ys = {t.name: t for t in x.tensors}, {t.name: t for t in y.tensors}
        if set(xs) != set(ys):
            problems.append(f"buffer {key}: tensors {sorted(xs)} against {sorted(ys)}")
        for name in sorted(set(xs) & set(ys)):
            tx, ty = xs[name], ys[name]
            ax, ay = {a.name: a for a in tx.axes}, {a.name: a for a in ty.axes}
            if set(ax) != set(ay) or not all(
                ax[n].kind.corresponds(ax[n], ay[n]) and ay[n].kind.corresponds(ay[n], ax[n])
                for n in ax
            ):
                problems.append(
                    f"buffer {key} tensor {name!r}: axes {_axes_summary(tx)} against "
                    f"{_axes_summary(ty)}"
                )
            if tx.dtype != ty.dtype:
                problems.append(
                    f"buffer {key} tensor {name!r}: elements of type {tx.dtype} against {ty.dtype}"
                )
        index_a, index_b = (
            group_a.get((x.pool_group, x.layer)),
            group_b.get((y.pool_group, y.layer)),
        )
        if index_a is None or index_b is None:
            problems.append(f"buffer {key} lies in no layer group of one rank")
            continue
        ga, gb = a.layer_groups[index_a], b.layer_groups[index_b]
        paged |= not ga.kind.per_request and not gb.kind.per_request
        if ga.kind != gb.kind:
            problems.append(f"buffer {key}: a layer group of {ga.kind} against one of {gb.kind}")
        model_a = ga.model_layers[ga.layers.index(x.layer)]
        model_b = gb.model_layers[gb.layers.index(y.layer)]
        if model_a != model_b:
            # A layer named alike on both sides but of different model layers, as when one side
            # names its layers by pipeline stage, would move one layer's bytes into another's.
            problems.append(f"buffer {key}: of model layer {model_a} against {model_b}")
    # A side naming a model layer otherwise, as one naming its layers by what its pipeline stage
    # holds, shares none of its ids with the other and would leave the layer unwritten.
    names_a, names_b = _model_layer_names(a, group_a), _model_layer_names(b, group_b)
    for model, role in sorted(set(names_a) & set(names_b)):
        if names_a[model, role] != names_b[model, role]:
            problems.append(
                f"model layer {model}'s {role} buffers in layers {sorted(names_a[model, role])} "
                f"against {sorted(names_b[model, role])}"
            )
    both = _held_model_layers(a, group_a) & _held_model_layers(b, group_b)
    for model, role in sorted(set(names_a) ^ set(names_b)):
        holder = a if (model, role) in names_a else b
        if model in both and role not in holder.request_rebuilt_roles:
            problems.append(
                f"model layer {model}'s {role} buffers are transferred by one rank only, as a "
                "cache attached by what a pipeline stage holds or a role one peer lacks is; only "
                "a rank rebuilding them per request, as CSA2's replay does "
                "(_page_request_rebuilt_roles), may hold them alone"
            )
    if paged and a.tokens_per_page != b.tokens_per_page:
        problems.append(f"pages of {a.tokens_per_page} tokens against {b.tokens_per_page}")
    if problems:
        raise ValueError("page layouts do not correspond: " + "; ".join(problems))


def _model_layer_names(
    layout: PageLayout, groups: dict[tuple[int, int], int]
) -> dict[tuple[int, str], set[int]]:
    """(Model layer, role) -> the global layers its transferred buffers of that role lie in."""
    names: dict[tuple[int, str], set[int]] = {}
    for buffer in layout.buffers:
        index = groups.get((buffer.pool_group, buffer.layer))
        if not buffer.transferred or index is None:
            continue
        group = layout.layer_groups[index]
        model = group.model_layers[group.layers.index(buffer.layer)]
        names.setdefault((model, buffer.role), set()).add(buffer.layer)
    return names


def _held_model_layers(layout: PageLayout, groups: dict[tuple[int, int], int]) -> set[int]:
    """The model layers of every buffer lying in a layer group, transferred or not."""
    held = set()
    for buffer in layout.buffers:
        index = groups.get((buffer.pool_group, buffer.layer))
        if index is not None:
            group = layout.layer_groups[index]
            held.add(group.model_layers[group.layers.index(buffer.layer)])
    return held


def sent_bytes(src: PageLayout, dst: PageLayout, *, key: int | None = None) -> dict[int, int]:
    """The most bytes the rank of ``src`` sends the rank of ``dst`` into one page of each of the
    destination's layer groups, a per-request group's slot counting as its page: ``map_buffers``'
    runs, summed by the index of the destination's layer group, a buffer of token positions' at
    the most positions a page moves (``BufferRuns.page_bytes``). Groups it sends nothing of are
    left out.

    Raises:
        ValueError: As ``map_buffers`` and ``layer_group_of`` do.
    """
    group_of = layer_group_of(dst)
    out: dict[int, int] = {}
    for runs in map_buffers(src, dst, key=key):
        group = group_of[runs.dst.pool_group, runs.dst.layer]
        out[group] = out.get(group, 0) + runs.page_bytes
    return out
