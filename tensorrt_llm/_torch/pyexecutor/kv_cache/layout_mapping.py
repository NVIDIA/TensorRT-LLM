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

A page layout says, for every buffer, which global indices of each tensor axis a rank holds
(``page_layout.held``). From a source's and a destination's layouts this module derives what the
source sends; nothing here names a cache, a model or an axis kind:

- ``fan``: which source ranks a destination hears from, by each dimension's shares of ranks.
- ``owned_indices``: which indices of an axis a source rank sends, from the indices every rank
  holds (``AxisKind.indices``).
- ``map_buffers``: the byte runs of every buffer both ranks transfer, once per pair of ranks.
- ``page_runs``: what one page moves; for an axis of token positions, every position both sides
  hold (``AxisKind.positions``) moved to its own index on each side (``AxisKind.index_of``).
- ``pair_pages``: which slots of a request pair in one transfer, as the layer group's kind says.
- ``check_compatible``: whether two ranks' layouts may pair at all.

Example: K of one layer, 8 KV heads of 256 bytes, head-major, from TP2 to TP4. Destination rank 1
holds heads 2..3. It hears from source rank 0 alone (``fan(2, 4, 1) == range(0, 1)``), which holds
heads 0..3 and sends heads 2..3 as one run of 512 bytes.

Election: where several paired source ranks hold an index (heads every rank keeps, or a dimension
no axis of the tensor splits), the one at position ``key`` modulo their count sends it, ``key``
being the destination's attention data-parallel rank, so that the destinations of several
data-parallel groups spread over the sources. A destination holding an index no paired source rank
holds, as an uneven head split can make, is refused rather than left without it.

Adding a kind: an axis kind in ``page_layout`` says which indices each rank holds, or which token
positions a page holds and at which index; this module derives the rest. Only a kind whose ranks
hold overlapping indices that are not runs registers its own ``owned_indices``. A layer-group kind
registers which slots pair (``pair_slots``).
"""

from __future__ import annotations

import dataclasses
import functools
import itertools
from dataclasses import dataclass
from typing import Mapping, NamedTuple, Sequence

import numpy as np

from .page_layout import (
    GRID_DIMENSIONS,
    Axis,
    AxisKind,
    BufferDesc,
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
    "Pairing",
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
    "sent_bytes",
]


def fan(src_size: int, dst_size: int, dst_rank: int) -> range:
    """The source ranks, of ``src_size`` along one parallel dimension, whose equal share of it
    overlaps the share of destination rank ``dst_rank`` of ``dst_size``.

    ``fan(2, 4, 1) == range(0, 1)``: TP4 rank 1 holds a quarter lying within TP2 rank 0's half.
    ``fan(4, 2, 1) == range(2, 4)``: TP2 rank 1 holds the halves of TP4 ranks 2 and 3.
    """
    return range(dst_rank * src_size // dst_size, -(-(dst_rank + 1) * src_size // dst_size))


@dataclass(frozen=True)
class TransferWindow:
    """What one transfer of a request covers, in the request's token positions.

    A whole prompt of 100 tokens of which the destination reuses the first 32 is
    ``TransferWindow(0, 100, reused=32)``; the first of two pieces of it is
    ``TransferWindow(0, 50, reused=32)``.

    Attributes:
        start: First token position the transfer moves.
        end: Token position the transfer reaches, exclusive.
        reused: Token positions from the request's start that the destination already holds, as
            a prefix it reuses; their pages are not paired.
    """

    start: int
    end: int
    reused: int = 0


@dataclass(frozen=True)
class Pairing:
    """A source rank as it sends to one destination rank.

    Attributes:
        src: The source rank's place.
        fans: For each dimension of ``GRID_DIMENSIONS``, the source ranks along it that the
            destination pairs with (``fan``).
        key: Picks the sender among paired source ranks holding the same index: the one at
            ``key`` modulo their count (``elect``).
    """

    src: RankTopology
    fans: Mapping[str, range]
    key: int

    @classmethod
    def between(cls, src: RankTopology, dst: RankTopology, key: int) -> Pairing:
        """The source rank at ``src`` as it sends to the destination rank at ``dst``."""
        fans = {}
        for dim in GRID_DIMENSIONS:
            src_size, _ = grid_place(src, (dim,))
            dst_size, dst_rank = grid_place(dst, (dim,))
            fans[dim] = fan(src_size, dst_size, dst_rank)
        return cls(src, fans, key)

    def place(self, split: Sequence[str]) -> tuple[int, int]:
        """``(size, rank)`` of the source rank in the grid of the dimensions ``split``."""
        return grid_place(self.src, split)

    def paired(self, split: Sequence[str]) -> list[int]:
        """The source ranks the destination pairs with, by place in the grid of ``split``,
        ascending."""
        sizes = [grid_place(self.src, (dim,))[0] for dim in split]
        places = itertools.product(*(self.fans[dim] for dim in split))
        return sorted({_flat(coords, sizes) for coords in places})

    def at(self, split: Sequence[str], rank: int) -> RankTopology:
        """The source's topology moved to place ``rank`` of the grid of ``split``, to ask a kind
        what another source rank holds."""
        sizes = [grid_place(self.src, (dim,))[0] for dim in split]
        changes = {}
        for dim, size in reversed(list(zip(split, sizes))):
            rank, coord = divmod(rank, size)
            if size > 1:
                changes[f"{dim}_rank"] = coord
        return dataclasses.replace(self.src, **changes)

    def elect(self, candidates: Sequence):
        """The one of ``candidates`` that sends: the one at ``key`` modulo their count."""
        return candidates[self.key % len(candidates)]

    def sends_replicas(self, tensor: Tensor) -> bool:
        """Whether this source rank sends ``tensor`` along the dimensions none of its axes splits,
        whose ranks hold it alike: whether it is the elected one of the paired ranks there."""
        free = [dim for dim in GRID_DIMENSIONS if all(dim not in a.split for a in tensor.axes)]
        if not free:
            return True
        mine = tuple(grid_place(self.src, (dim,))[1] for dim in free)
        return self.elect(list(itertools.product(*(self.fans[dim] for dim in free)))) == mine


def _flat(coords: Sequence[int], sizes: Sequence[int]) -> int:
    rank = 0
    for c, n in zip(coords, sizes):
        rank = rank * n + c
    return rank


@functools.singledispatch
def owned_indices(
    kind: AxisKind, axis: Axis, pairing: Pairing, wanted: Progression
) -> list[Progression]:
    """The runs of ``axis``' indices that the source rank sends, derived from the indices each
    source rank holds (``AxisKind.indices``): of those it holds, the ones it is elected for among
    the paired source ranks holding them.

    8 heads split contiguously from TP4 to TP2: destination rank 0 hears from source ranks 0 and 1,
    holding heads 0..1 and 2..3, and each sends its own. Where ranks hold overlapping runs, as 6
    heads over 4 ranks of 2 heads each, every index goes to one of its holders (``Pairing.elect``);
    where each index lies on one rank alone, as ``Cyclic`` places them, that rank sends it.

    A kind whose ranks hold overlapping indices that are not runs registers its own.

    Args:
        kind: ``axis.kind``, on whose type this dispatches.
        axis: The source's axis.
        pairing: The source rank and the destination it sends to.
        wanted: The indices of ``axis`` the destination holds.

    Raises:
        ValueError: If the destination holds an index no paired source rank holds, or the ranks
            of ``kind`` hold overlapping indices that are not runs.
    """
    if not kind.split:
        return [held(axis, pairing.src)]
    n, me = pairing.place(kind.split)
    paired = pairing.paired(kind.split)
    holdings = [kind.indices(axis, pairing.at(kind.split, rank)) for rank in range(n)]
    if all(h.step == 1 or h.count <= 1 for h in holdings):
        return _elect_runs(axis, holdings, paired, me, pairing, wanted)
    if all(a.intersect(b) is None for a, b in itertools.combinations(holdings, 2)):
        for rank in range(n):
            if rank not in paired:
                _refuse_unpaired(axis, holdings[rank], wanted)
        return [holdings[me]] if me in paired else []
    raise ValueError(
        f"axis {axis.name!r}: the ranks of kind {kind.name!r} hold overlapping indices that are "
        "not runs, so which rank sends each is the kind's to say: it registers owned_indices"
    )


def _elect_runs(
    axis: Axis,
    holdings: Sequence[Progression],
    paired: Sequence[int],
    me: int,
    pairing: Pairing,
    wanted: Progression,
) -> list[Progression]:
    """``owned_indices`` where every rank holds one run: each stretch between the runs' bounds goes
    to the elected one of the paired ranks whose runs cover it."""
    runs = [(h.start, h.start + h.count) for h in holdings]
    bounds = sorted({0, axis.extent} | {min(b, axis.extent) for run in runs for b in run})
    out: list[Progression] = []
    for lo, hi in zip(bounds, bounds[1:]):
        holders = [rank for rank in paired if runs[rank][0] <= lo and hi <= runs[rank][1]]
        if not holders:
            _refuse_unpaired(axis, Progression(lo, 1, hi - lo), wanted)
            continue
        if pairing.elect(holders) != me:
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


@dataclass(frozen=True)
class PositionRule:
    """How a buffer's runs move with the token positions a page holds: its axis of token positions
    on the source and on the destination. A position both sides hold of a page
    (``AxisKind.positions``) moves from its index on the source to its index on the destination
    (``AxisKind.index_of``).

    In pages of 32 tokens, of page 1 (positions 32..63) written up to position 50: a source axis
    holding 34..49 and a destination axis holding 32..49 move 34..49, position 40 from the index
    the source's kind gives it to the index the destination's kind gives it.

    Attributes:
        src: The source's axis of token positions.
        dst: The destination's axis of the same name.
        tokens_per_page: Token positions of one page.
    """

    src: Axis
    dst: Axis
    tokens_per_page: int

    def positions(self, page: int, window: TransferWindow) -> np.ndarray:
        """The token positions of page ``page`` (its position among a request's pages, from 0)
        that both sides hold once the request has written every position below ``window.end``,
        as int64."""
        src = self.src.kind.positions(self.src, page, window.end, self.tokens_per_page)
        dst = self.dst.kind.positions(self.dst, page, window.end, self.tokens_per_page)
        return np.intersect1d(np.asarray(src, dtype=np.int64), np.asarray(dst, dtype=np.int64))

    @property
    def kept(self) -> int:
        """The most positions of one page a transfer moves: those both hold of a page written
        whole."""
        return int(self.positions(0, TransferWindow(0, self.tokens_per_page)).size)

    def offsets(self, positions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Each position's byte offset from index 0 of the axis, on the source and on the
        destination."""
        src_index = self.src.kind.index_of(self.src, positions, self.tokens_per_page)
        dst_index = self.dst.kind.index_of(self.dst, positions, self.tokens_per_page)
        return src_index * self.src.stride, dst_index * self.dst.stride


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
        by_position: How the runs move with the token positions a page holds (``page_runs``):
            None for a buffer laid out alike for every position, and equal for buffers whose runs
            move alike.
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
    those of every position both sides hold, each from its index on the source to its index on
    the destination (``PositionRule``)."""
    rule = runs.by_position
    if rule is None:
        return runs.src_offsets, runs.dst_offsets, runs.sizes
    src_at, dst_at = rule.offsets(rule.positions(page, window))
    src = src_at[:, None] + runs.src_offsets[None, :]
    dst = dst_at[:, None] + runs.dst_offsets[None, :]
    sizes = np.broadcast_to(runs.sizes, src.shape).ravel()
    return merge_runs(src.ravel(), dst.ravel(), sizes)


class _AxisPart(NamedTuple):
    """One axis of a tensor as the source sends it."""

    src_axis: Axis
    dst_axis: Axis
    src_held: Progression  # the indices the source holds, in its local order
    dst_held: Progression  # the indices the destination holds, in its local order
    sent: list[Progression]  # runs of indices the source sends that the destination holds


def _tensor_runs(
    src_buffer: BufferDesc,
    src_tensor: Tensor,
    dst_buffer: BufferDesc,
    dst_tensor: Tensor,
    pairing: Pairing,
    dst: RankTopology,
) -> list[tuple[np.ndarray, np.ndarray, int]]:
    """What the source sends of one tensor: per box of indices it sends (one run of sent indices
    per axis), the source offsets, destination offsets and size of its runs; none when it sends
    nothing of the tensor."""
    where = f"buffer ({src_buffer.layer}, {src_buffer.role}) tensor {src_tensor.name!r}"
    dst_axes = {a.name: a for a in dst_tensor.axes}
    if set(dst_axes) != {a.name for a in src_tensor.axes}:
        raise ValueError(
            f"{where}: axes {[a.name for a in src_tensor.axes]} against {list(dst_axes)}"
        )
    if not pairing.sends_replicas(src_tensor):
        return []
    parts = []
    for src_axis in src_tensor.axes:
        dst_axis = dst_axes[src_axis.name]
        if src_axis.extent != dst_axis.extent:
            raise ValueError(
                f"{where} axis {src_axis.name}: extent {src_axis.extent} against {dst_axis.extent}"
            )
        src_held, dst_held = held(src_axis, pairing.src), held(dst_axis, dst)
        sent = []
        for owned in owned_indices(src_axis.kind, src_axis, pairing, dst_held):
            part = owned.intersect(src_held)
            part = None if part is None else part.intersect(dst_held)
            if part is not None:
                sent.append(part)
        if not sent:
            return []
        parts.append(_AxisPart(src_axis, dst_axis, src_held, dst_held, sent))
    return [
        _box_runs(src_buffer, src_tensor, dst_buffer, dst_tensor, parts, box)
        for box in itertools.product(*(p.sent for p in parts))
    ]


def _box_runs(
    src_buffer: BufferDesc,
    src_tensor: Tensor,
    dst_buffer: BufferDesc,
    dst_tensor: Tensor,
    parts: Sequence[_AxisPart],
    box: Sequence[Progression],
) -> tuple[np.ndarray, np.ndarray, int]:
    """The runs of one box of indices, ``box[i]`` the indices of axis ``parts[i]``: a loop per
    axis of more than one index, the loops whose steps continue the run on both sides folded into
    one run."""
    src_base = src_buffer.slot_offset + src_tensor.offset
    dst_base = dst_buffer.slot_offset + dst_tensor.offset
    loops = []  # (count, source step, destination step) in bytes
    for part, indices in zip(parts, box):
        src_base += (
            (indices.start - part.src_held.start) // part.src_held.step * part.src_axis.stride
        )
        dst_base += (
            (indices.start - part.dst_held.start) // part.dst_held.step * part.dst_axis.stride
        )
        if indices.count > 1:
            loops.append(
                (
                    indices.count,
                    indices.step // part.src_held.step * part.src_axis.stride,
                    indices.step // part.dst_held.step * part.dst_axis.stride,
                )
            )
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
    order: runs (0 -> 100, 8 bytes) and (8 -> 108, 8 bytes) become (0 -> 100, 16 bytes)."""
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


def _position_rule(
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
            the axes differ in name or kind, or the axis lies in a layer group holding one slot
            per request.
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
    return PositionRule(src_axis, dst_axis, src.tokens_per_page)


def map_buffers(src: PageLayout, dst: PageLayout, *, key: int | None = None) -> list[BufferRuns]:
    """What the rank of ``src`` sends the rank of ``dst``: for each buffer both transfer (same
    global layer and role), the runs of its tensors the source is the elected sender of. Buffers
    it sends nothing of are left out. A buffer of token positions gives the runs of one position,
    which ``page_runs`` moves to each position of a page.

    Example: K of one layer, 8 heads of 256 bytes head-major, from TP2 rank 0 to TP4 rank 1: one
    ``BufferRuns`` of a single run, 512 bytes from byte ``s + 512`` of the source's slot to byte
    ``d`` of the destination's, ``s`` and ``d`` the buffer's offset in each slot.

    Args:
        src: The source rank's page layout.
        dst: The destination rank's page layout.
        key: Breaks ties between source ranks holding the same bytes; the destination's attention
            data-parallel rank by default.

    Raises:
        ValueError: If two matched buffers' tensors or axes differ in name, or in extent other
            than an axis of token positions', one has an axis of token positions the other has
            not, such an axis lies in a group holding one slot per request, or the destination
            holds an index no source rank it pairs with holds.
    """
    pairing = Pairing.between(
        src.topology, dst.topology, dst.topology.dp_rank if key is None else key
    )
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
        rule = _position_rule(src, src_buffer, dst, dst_buffer, groups)
        parts = [
            part
            for src_tensor in src_buffer.tensors
            for part in _tensor_runs(
                src_buffer,
                _pinned(src_tensor),
                dst_buffer,
                _pinned(dst_tensors[src_tensor.name]),
                pairing,
                dst.topology,
            )
        ]
        if not parts:
            continue
        src_offsets = np.concatenate([p[0] for p in parts])
        dst_offsets = np.concatenate([p[1] for p in parts])
        sizes = np.concatenate([np.full(p[0].size, p[2], dtype=np.int64) for p in parts])
        runs = merge_runs(src_offsets, dst_offsets, sizes)
        out.append(BufferRuns(src_buffer, dst_buffer, *runs, by_position=rule))
    return out


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
    at which both hold one, from index ``start`` on: tables ``[10, 11, -1, 13]`` and
    ``[20, -1, 22, 23]`` pair at indices 0 and 3.

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
    the source's slot ids ``src_ids`` and the destination's ``dst_ids`` for the request. Each
    layer-group kind registers its own.

    Raises:
        ValueError: If ``kind`` registers no pairing.
    """
    raise ValueError(f"layer-group kind {kind.name!r} registers no pair_slots")


_NO_SLOTS = np.zeros(0, dtype=np.int64)


@pair_slots.register(Paged)
def _pair_paged(
    kind: Paged,
    src: PageLayout,
    src_ids: np.ndarray,
    dst: PageLayout,
    dst_ids: np.ndarray,
    window: TransferWindow,
) -> PagePairs:
    """A slot per page: pages both hold, past the destination's reused prefix. A Helix
    destination at CP rank ``r`` of ``n`` holds pages ``r, r + n, ...`` of a source holding them
    all; reuse is refused under Helix."""
    src_ids = np.asarray(src_ids, dtype=np.int64)
    dst_ids = np.asarray(dst_ids, dtype=np.int64)
    cp_size, cp_rank = dst.topology.cp_size, dst.topology.cp_rank
    helix = cp_size > 1 and src.topology.cp_size == 1
    if helix:
        src_ids = src_ids[cp_rank::cp_size]
    paired = paired_ordinals(src_ids, dst_ids, window.reused // src.tokens_per_page)
    pages = paired * cp_size + cp_rank if helix else paired
    return PagePairs(src_ids[paired], dst_ids[paired], pages)


@pair_slots.register(PerRequest)
def _pair_per_request(
    kind: PerRequest,
    src: PageLayout,
    src_ids: np.ndarray,
    dst: PageLayout,
    dst_ids: np.ndarray,
    window: TransferWindow,
) -> PagePairs:
    """One slot per request: the first slot of each side, when both hold one."""
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
    ``dst_ids``, as the groups' kind registers (``pair_slots``).

    Example: attention pages ``[10, 11, 12]`` and ``[20, 21, 22]`` with the first page reused
    pair slots 11 -> 21 and 12 -> 22, pages 1 and 2.

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


def check_compatible(a: PageLayout, b: PageLayout) -> None:
    """Check that two ranks' pages correspond, whichever sends. Refused, each naming the
    difference:

    - KV of another element type or quantization.
    - On the layers both hold, a buffer one rank alone transfers, unless that rank rebuilds its
      role per request (``PageLayout.request_rebuilt_roles``: CSA2's generation workers rebuild
      the sliding caches a context-only worker keeps none of past its split); otherwise buffers
      of other sub-pages, tensors, axes, extents, element types, layer-group kinds or model
      layers. Axes correspond as their kinds judge (``AxisKind.corresponds``): heads split over
      different rank counts do, and so may axes of token positions each side sizes as it keeps.
    - A model layer's role whose buffers the two name by other global layers, or that one rank
      alone transfers of a model layer both hold, which would move one layer's bytes into
      another's or leave it unwritten.
    - Pages of different token counts where both transfer buffers of paged groups.

    How many tokens each side keeps of a windowed group is not compared: retention margins (reuse
    backoff, draft tokens) legitimately differ between the sides, and a transfer moves the blocks
    both still hold.

    Raises:
        ValueError: Naming every difference found.
    """
    groups = (layer_group_of(a), layer_group_of(b))
    problems = _kv_problems(a, b) + _buffer_problems(a, b, groups) + _naming_problems(a, b, groups)
    if _pages_pair(a, b, groups) and a.tokens_per_page != b.tokens_per_page:
        problems.append(f"pages of {a.tokens_per_page} tokens against {b.tokens_per_page}")
    if problems:
        raise ValueError("page layouts do not correspond: " + "; ".join(problems))


def _kv_problems(a: PageLayout, b: PageLayout) -> list[str]:
    problems = []
    if a.dtype != b.dtype:
        problems.append(f"KV elements of type {a.dtype} against {b.dtype}")
    if a.quant_mode != b.quant_mode:
        problems.append(f"KV quantization {a.quant_mode} against {b.quant_mode}")
    return problems


def _transferred(layout: PageLayout) -> dict[tuple[int, str], BufferDesc]:
    return {(b.layer, b.role): b for b in layout.buffers if b.transferred}


def _shared_buffers(a: PageLayout, b: PageLayout):
    """``(layer, role)``, ``a``'s buffer and ``b``'s, None where that rank does not transfer it,
    of every buffer either transfers on a layer both hold, in key order."""
    layers = {x.layer for x in a.buffers} & {x.layer for x in b.buffers}
    ta, tb = _transferred(a), _transferred(b)
    for key in sorted(k for k in set(ta) | set(tb) if k[0] in layers):
        yield key, ta.get(key), tb.get(key)


def _group_pair(a: PageLayout, x: BufferDesc, b: PageLayout, y: BufferDesc, groups):
    """The layer groups of ``a``'s buffer ``x`` and ``b``'s buffer ``y``, None where one lies in
    none."""
    index_a = groups[0].get((x.pool_group, x.layer))
    index_b = groups[1].get((y.pool_group, y.layer))
    if index_a is None or index_b is None:
        return None
    return a.layer_groups[index_a], b.layer_groups[index_b]


def _buffer_problems(a: PageLayout, b: PageLayout, groups) -> list[str]:
    """The differences of the buffers either rank transfers on the layers both hold."""
    problems = []
    for key, x, y in _shared_buffers(a, b):
        if x is None or y is None:
            if key[1] not in (a if y is None else b).request_rebuilt_roles:
                problems.append(f"buffer {key} is transferred by one rank only")
            continue
        if x.expansion != y.expansion:
            problems.append(f"buffer {key}: {x.expansion} sub-pages against {y.expansion}")
        problems += _tensor_problems(key, x, y)
        pair = _group_pair(a, x, b, y, groups)
        if pair is None:
            problems.append(f"buffer {key} lies in no layer group of one rank")
            continue
        ga, gb = pair
        if ga.kind != gb.kind:
            problems.append(f"buffer {key}: a layer group of {ga.kind} against one of {gb.kind}")
        model_a = ga.model_layers[ga.layers.index(x.layer)]
        model_b = gb.model_layers[gb.layers.index(y.layer)]
        if model_a != model_b:
            # A layer named alike on both sides but of different model layers, as when one side
            # names its layers by pipeline stage, would move one layer's bytes into another's.
            problems.append(f"buffer {key}: of model layer {model_a} against {model_b}")
    return problems


def _tensor_problems(key: tuple[int, str], x: BufferDesc, y: BufferDesc) -> list[str]:
    """The differences of two ranks' tensors of one buffer: names, axes as their kinds judge
    (``AxisKind.corresponds``), element types."""
    problems = []
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
    return problems


def _axes_summary(tensor: Tensor) -> dict[str, object]:
    """Each axis' extent by name, or its kind's name for an axis of token positions."""
    return {a.name: a.kind.name if a.kind.positional else a.extent for a in tensor.axes}


def _pages_pair(a: PageLayout, b: PageLayout, groups) -> bool:
    """Whether both ranks transfer a buffer of paged groups (not ``GroupKind.per_request``), whose
    pages then pair one to one."""
    for _, x, y in _shared_buffers(a, b):
        pair = None if x is None or y is None else _group_pair(a, x, b, y, groups)
        if pair is not None and not pair[0].kind.per_request and not pair[1].kind.per_request:
            return True
    return False


def _naming_problems(a: PageLayout, b: PageLayout, groups) -> list[str]:
    """A model layer's role whose transferred buffers the ranks name by other global layers, or
    that one rank alone transfers of a model layer both hold, unless that rank rebuilds it per
    request."""
    problems = []
    names_a, names_b = _model_layer_names(a, groups[0]), _model_layer_names(b, groups[1])
    for model, role in sorted(set(names_a) & set(names_b)):
        if names_a[model, role] != names_b[model, role]:
            problems.append(
                f"model layer {model}'s {role} buffers in layers {sorted(names_a[model, role])} "
                f"against {sorted(names_b[model, role])}"
            )
    both = _held_model_layers(a, groups[0]) & _held_model_layers(b, groups[1])
    for model, role in sorted(set(names_a) ^ set(names_b)):
        holder = a if (model, role) in names_a else b
        if model in both and role not in holder.request_rebuilt_roles:
            problems.append(
                f"model layer {model}'s {role} buffers are transferred by one rank only, as a "
                "cache attached by what a pipeline stage holds or a role one peer lacks is; only "
                "a rank rebuilding them per request, as CSA2's replay does "
                "(_page_request_rebuilt_roles), may hold them alone"
            )
    return problems


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
    out = set()
    for buffer in layout.buffers:
        index = groups.get((buffer.pool_group, buffer.layer))
        if index is not None:
            group = layout.layer_groups[index]
            out.add(group.model_layers[group.layers.index(buffer.layer)])
    return out


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
