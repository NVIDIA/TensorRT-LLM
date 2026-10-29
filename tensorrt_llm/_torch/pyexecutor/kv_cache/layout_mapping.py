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
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .page_layout import (
    GRID_DIMENSIONS,
    Axis,
    BufferDesc,
    PageLayout,
    Progression,
    RankTopology,
    Tensor,
    grid_place,
    held,
)

__all__ = [
    "BufferRuns",
    "check_compatible",
    "fan",
    "layer_group_of",
    "map_buffers",
    "merge_runs",
    "sent_bytes",
]


def fan(src_size: int, dst_size: int, dst_rank: int) -> range:
    """The source ranks, of ``src_size`` along one parallel dimension, whose equal share of it
    overlaps the share of destination rank ``dst_rank`` of ``dst_size``."""
    return range(dst_rank * src_size // dst_size, -(-(dst_rank + 1) * src_size // dst_size))


@dataclass(frozen=True, eq=False)
class BufferRuns:
    """The bytes a source rank sends of one buffer the destination rank also holds: run ``i`` is
    ``sizes[i]`` bytes from ``src_offsets[i]`` of the source page's pool slot to ``dst_offsets[i]``
    of the destination's.

    Attributes:
        src: The source rank's buffer.
        dst: The destination rank's buffer of the same layer and role.
        src_offsets: int64 offsets within the source buffer's pool slot.
        dst_offsets: int64 offsets within the destination buffer's pool slot.
        sizes: int64 run lengths in bytes.
    """

    src: BufferDesc
    dst: BufferDesc
    src_offsets: np.ndarray
    dst_offsets: np.ndarray
    sizes: np.ndarray


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


def _owned(
    axis: Axis, src: RankTopology, fans: dict[str, range], key: int, wanted: Progression
) -> list[Progression]:
    """The runs of indices of ``axis`` that the source rank sends: the whole axis when it does not
    split it, else the indices it is elected for among the paired ranks holding them.

    Raises:
        ValueError: If the destination holds (``wanted``) an index no paired rank holds.
    """
    if not axis.split:
        return [Progression(0, 1, axis.extent)]
    n, r = grid_place(src, axis.split)
    sizes = [_dims(src)[dim][0] for dim in axis.split]
    paired = sorted(
        {_flat(c, sizes) for c in itertools.product(*(fans[dim] for dim in axis.split))}
    )
    if axis.cyclic:
        # One rank holds each index: index i lies on rank i modulo n.
        for rank in range(n):
            if rank not in paired and rank < axis.extent:
                _refuse_unpaired(
                    axis, Progression(rank, n, len(range(rank, axis.extent, n))), wanted
                )
        return [held(axis, src)] if r in paired else []
    runs = [(rank * axis.extent // n, rank * axis.extent // n + axis.held) for rank in range(n)]
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
        for owned in _owned(src_axis, src, fans, key, dst_held):
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


def map_buffers(src: PageLayout, dst: PageLayout, *, key: int | None = None) -> list[BufferRuns]:
    """What the rank of ``src`` sends the rank of ``dst``: for each buffer both transfer (same
    global layer and role), the runs of its tensors the source is the elected sender of. Buffers
    that move nothing are left out.

    Args:
        src: The source rank's page layout.
        dst: The destination rank's page layout.
        key: Breaks ties between source ranks holding the same bytes; the destination's attention
            data-parallel rank by default.

    Raises:
        ValueError: If two matched buffers' tensors or axes differ in name or extent, or the
            destination holds an index no source rank it pairs with holds.
    """
    key = dst.topology.dp_rank if key is None else key
    fans = _fans(src.topology, dst.topology)
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
        parts = []
        for src_tensor in src_buffer.tensors:
            runs = _tensor_runs(
                src_buffer,
                src_tensor,
                src.topology,
                dst_buffer,
                dst_tensors[src_tensor.name],
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
        out.append(BufferRuns(src_buffer, dst_buffer, *merge_runs(src_offsets, dst_offsets, sizes)))
    return out


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


def _reads_within(read: int | None, kept: int | None) -> bool:
    """Whether a window of ``read`` tokens lies within one of ``kept`` tokens; None is unbounded."""
    return kept is None or (read is not None and read <= kept)


def check_compatible(a: PageLayout, b: PageLayout) -> None:
    """Check that two ranks' pages correspond, whichever sends.

    On the layers both hold, every buffer either rank transfers the other transfers too, as the
    same tensors: the same names, axes and extents over all ranks, however each splits them, and
    elements of the same type where the tensors state one, in pages of as many sub-pages. The KV
    data is of one element type and quantization. Pages pair one to one, so where both transfer
    attention buffers their pages hold as many tokens; a layer group keeps what the other's reads
    (its read window within the other's kept window), with as many sink blocks, and is of the same
    kind.

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
    groups_compared = set()
    for key in sorted(k for k in set(ta) | set(tb) if k[0] in layers):
        x, y = ta.get(key), tb.get(key)
        if x is None or y is None:
            problems.append(f"buffer {key} is transferred by one rank only")
            continue
        if x.expansion != y.expansion:
            problems.append(f"buffer {key}: {x.expansion} sub-pages against {y.expansion}")
        xs, ys = {t.name: t for t in x.tensors}, {t.name: t for t in y.tensors}
        if set(xs) != set(ys):
            problems.append(f"buffer {key}: tensors {sorted(xs)} against {sorted(ys)}")
        for name in sorted(set(xs) & set(ys)):
            tx, ty = xs[name], ys[name]
            ex = {axis.name: axis.extent for axis in tx.axes}
            ey = {axis.name: axis.extent for axis in ty.axes}
            if ex != ey:
                problems.append(f"buffer {key} tensor {name!r}: axes {ex} against {ey}")
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
        paged |= ga.kind == "attention" and gb.kind == "attention"
        if (ga, gb) in groups_compared:
            continue
        groups_compared.add((ga, gb))
        where = f"layer {key[0]}"
        if ga.kind != gb.kind:
            problems.append(f"{where}: a layer group of {ga.kind} against one of {gb.kind}")
        elif not (
            _reads_within(ga.read_window, gb.kept_window)
            and _reads_within(gb.read_window, ga.kept_window)
        ):
            problems.append(
                f"{where}: reads {ga.read_window} of {ga.kept_window} kept tokens against "
                f"{gb.read_window} of {gb.kept_window}"
            )
        if ga.num_sink_blocks != gb.num_sink_blocks:
            problems.append(
                f"{where}: {ga.num_sink_blocks} sink blocks against {gb.num_sink_blocks}"
            )
    if paged and a.tokens_per_page != b.tokens_per_page:
        problems.append(f"pages of {a.tokens_per_page} tokens against {b.tokens_per_page}")
    if problems:
        raise ValueError("page layouts do not correspond: " + "; ".join(problems))


def sent_bytes(src: PageLayout, dst: PageLayout, *, key: int | None = None) -> dict[int, int]:
    """Bytes the rank of ``src`` sends the rank of ``dst`` into one page of each of the
    destination's layer groups, a state group's slot counting as its page: ``map_buffers``' runs,
    summed by the index of the destination's layer group. Groups it sends nothing of are left out.

    Raises:
        ValueError: As ``map_buffers`` and ``layer_group_of`` do.
    """
    group_of = layer_group_of(dst)
    out: dict[int, int] = {}
    for runs in map_buffers(src, dst, key=key):
        group = group_of[runs.dst.pool_group, runs.dst.layer]
        out[group] = out.get(group, 0) + int(runs.sizes.sum())
    return out
