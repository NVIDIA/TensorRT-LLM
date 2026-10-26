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
"""The one mapping algorithm: from two ranks' page layouts to the byte runs that move.

1. Buffers match by ``(global layer, role)`` (:func:`match_buffers`).
2. For every segment of a matched pair, the two ranks' spans along the split axis intersect
   (:func:`~.topology.shard_span`); an empty intersection means the two ranks are not paired.
3. The intersection becomes byte runs -- one per row of the axes outside the split axis.
4. Ranks holding identical bytes elect one sender (:func:`~.topology.elects_sender`).

Buffers of the same ``(layer group, pool)`` pair and election are merged into one
:class:`TransferGroup`, so per request the transfer extracts unit addresses once per group, not
once per buffer. Groups are cached per peer; per request only unit base addresses are added.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, FrozenSet, List, Optional, Sequence, Set, Tuple

import numpy as np

from tensorrt_llm._torch.disaggregation.base.region import (
    MemRegionGroup,
    RegionMapperBase,
    SpecRegion,
    SpecRegionPair,
)
from tensorrt_llm._torch.pyexecutor.kv_cache.page_layout import BufferDesc, Segment
from tensorrt_llm._utils import nvtx_range

from . import units
from .rank_pages import BufferRef, PoolKey, RankPages
from .topology import ShardGrid, elects_sender, replication_factor, shard_grid, shard_span

if TYPE_CHECKING:
    from tensorrt_llm._torch.disaggregation.native.rank_info import RankInfo

Run = Tuple[int, int, int]
"""``(src offset, dst offset, length)`` in bytes, relative to a unit's slot base."""

MatchedPair = Tuple[int, BufferDesc, int, BufferDesc]
"""``(local layer group, local buffer, peer layer group, peer buffer)`` of one matched buffer."""


def axis_names(segment: Segment) -> Tuple[str, ...]:
    return tuple(name for name, _ in segment.axes)


def split_units(segment: Segment) -> int:
    """This rank's extent along the split axis; a segment every rank holds alike is one unit."""
    return segment.split_extent or 1


def replicated(buffer: BufferDesc) -> bool:
    """Every rank of the shard grid holds the same bytes, so one elected rank sends them."""
    return all(segment.split_axis is None for segment in buffer.segments)


def _context(buffer: BufferDesc) -> str:
    return f"buffer (layer={buffer.layer}, role={buffer.role!r})"


# ---------------------------------------------------------------------------
# Which buffers pair up
# ---------------------------------------------------------------------------


def _buffer_index(pages: RankPages) -> Dict[Tuple[int, str], BufferRef]:
    """``{(layer, role): (layer group, buffer)}``; a rank holds each (layer, role) once."""
    index: Dict[Tuple[int, str], BufferRef] = {}
    for g in range(pages.num_layer_groups):
        for i, buffer in enumerate(pages.buffers(g)):
            key = (buffer.layer, buffer.role)
            if key in index:
                raise ValueError(
                    f"{_context(buffer)} appears in layer groups {index[key][0]} and {g}"
                )
            index[key] = (g, i)
    return index


def _group_traits(pages: RankPages, g: int) -> Tuple[Tuple[str, ...], str, bool]:
    desc = pages.group(g)
    return desc.grid, desc.unit, desc.blocks_by_cp


def match_buffers(
    self_pages: Optional[RankPages], peer_pages: Optional[RankPages]
) -> Dict[BufferRef, BufferRef]:
    """Each local buffer paired with the peer buffer of the same layer and role.

    Buffers only one side holds are a legal skip (PP slices, one-sided MTP layers); a role that
    must not be one-sided is rejected by :func:`validate_peer_compatible` instead.
    """
    if self_pages is None or peer_pages is None:
        return {}
    peer_index = _buffer_index(peer_pages)
    mapping: Dict[BufferRef, BufferRef] = {}
    for g in range(self_pages.num_layer_groups):
        for i, buffer in enumerate(self_pages.buffers(g)):
            ref = peer_index.get((buffer.layer, buffer.role))
            if ref is None:
                continue
            local, peer = _group_traits(self_pages, g), _group_traits(peer_pages, ref[0])
            if local != peer:
                raise ValueError(
                    f"{_context(buffer)} lies in layer groups that split differently "
                    f"(local grid/unit/blocks_by_cp={local}, peer={peer})"
                )
            mapping[(g, i)] = ref
    return mapping


def matched_pairs(
    self_pages: RankPages, peer_pages: RankPages, mapping: Dict[BufferRef, BufferRef]
) -> List[MatchedPair]:
    return [
        (g, self_pages.buffers(g)[i], pg, peer_pages.buffers(pg)[pi])
        for (g, i), (pg, pi) in mapping.items()
    ]


# ---------------------------------------------------------------------------
# Byte runs of one segment pair, one buffer pair
# ---------------------------------------------------------------------------


def _check_same_layout(src: Segment, dst: Segment, context: str) -> None:
    if src.split_axis != dst.split_axis or axis_names(src) != axis_names(dst):
        raise ValueError(
            f"{context}: layout differs (local axes={axis_names(src)} split={src.split_axis!r}, "
            f"peer axes={axis_names(dst)} split={dst.split_axis!r})"
        )


def _segment_runs(
    src: Segment,
    dst: Segment,
    grid: Sequence[str],
    self_ri: "RankInfo",
    peer_ri: "RankInfo",
    context: str,
) -> List[Run]:
    """Runs of one segment pair, relative to each segment's start; empty when not paired."""
    _check_same_layout(src, dst, context)
    src_grid = shard_grid(grid, self_ri)
    dst_grid = shard_grid(grid, peer_ri)
    src_units, dst_units = split_units(src), split_units(dst)
    virtual = min(src_units * src_grid.size, dst_units * dst_grid.size)
    src_lo, src_hi = shard_span(src_units, src_grid, virtual)
    dst_lo, dst_hi = shard_span(dst_units, dst_grid, virtual)
    lo, hi = max(src_lo, dst_lo), min(src_hi, dst_hi)
    if lo >= hi:
        return []
    if (lo, hi) == (src_lo, src_hi) == (dst_lo, dst_hi):
        # Both ranks hold exactly the same shard: the segment moves whole.
        if src.nbytes != dst.nbytes:
            raise ValueError(
                f"{context}: cache region size mismatch: local={src.nbytes}, peer={dst.nbytes} "
                "(a sharded role without declared geometry cannot be resharded)"
            )
        return [(0, 0, src.nbytes)]
    if src.rows != dst.rows:
        raise ValueError(
            f"{context}: {src.split_axis!r} slicing requires equal row counts on both sides "
            f"(local={src.rows}, peer={dst.rows})"
        )
    if src.unit_bytes != dst.unit_bytes:
        raise ValueError(
            f"{context}: bytes per {src.split_axis!r} unit mismatch: local={src.unit_bytes}, "
            f"peer={dst.unit_bytes}"
        )
    unit_bytes = src.unit_bytes
    src_row = src_units * unit_bytes
    dst_row = dst_units * unit_bytes
    length = (hi - lo) * unit_bytes
    return [
        (
            row * src_row + (lo - src_lo) * unit_bytes,
            row * dst_row + (lo - dst_lo) * unit_bytes,
            length,
        )
        for row in range(src.rows)
    ]


def buffer_runs(
    src: BufferDesc,
    dst: BufferDesc,
    grid: Sequence[str],
    self_ri: "RankInfo",
    peer_ri: "RankInfo",
) -> List[Run]:
    """Slot-relative runs of one matched buffer pair, whose segments are split over the grid of
    ranks ``grid`` names; empty when the two ranks are not paired."""
    context = _context(src)
    if len(src.segments) != len(dst.segments):
        raise ValueError(
            f"{context}: segment count mismatch: local={len(src.segments)}, "
            f"peer={len(dst.segments)}"
        )
    runs: List[Run] = []
    src_off, dst_off = src.slot_offset, dst.slot_offset
    for i, (s_seg, d_seg) in enumerate(zip(src.segments, dst.segments)):
        for s, d, n in _segment_runs(
            s_seg, d_seg, grid, self_ri, peer_ri, f"{context} segment {i}"
        ):
            runs.append((src_off + s, dst_off + d, n))
        src_off += s_seg.nbytes
        dst_off += d_seg.nbytes
    return runs


def _merge(runs: List[Run]) -> List[Run]:
    """Merge consecutive runs that are contiguous on both sides."""
    merged: List[Run] = []
    for s, d, n in runs:
        if merged:
            ps, pd, pn = merged[-1]
            if ps + pn == s and pd + pn == d:
                merged[-1] = (ps, pd, pn + n)
                continue
        merged.append((s, d, n))
    return merged


def _election(src: BufferDesc, dst: BufferDesc, grid: Sequence[str], self_ri, peer_ri) -> int:
    return replication_factor(
        split_units(src.segments[0]),
        shard_grid(grid, self_ri),
        split_units(dst.segments[0]),
        shard_grid(grid, peer_ri),
    )


def _cp_election(desc, self_pages: RankPages, peer_pages: RankPages) -> int:
    """How many local context-parallel ranks hold the bytes one peer rank pairs with
    (``topology.paired_ranks``), for a group whose grid leaves out "cp" and whose blocks do not
    split by context-parallel rank; 1 otherwise."""
    topology, peer_topology = self_pages.layout.topology, peer_pages.layout.topology
    if "cp" in desc.grid or desc.blocks_by_cp or topology.attention_dp:
        return 1
    return max(1, topology.cp_size // peer_topology.cp_size)


# ---------------------------------------------------------------------------
# Transfer groups: runs merged per (layer group, pool) pair and election
# ---------------------------------------------------------------------------


class RunPlan(RegionMapperBase):
    """Slot-relative byte runs, ready to expand over unit base addresses.

    ``map`` takes one base address per unit on each side (``RankPages.slots``) and returns one
    :class:`SpecRegionPair` per distinct run length, so the fragment arrays stay uniform-size like
    the transfer path expects.
    """

    def __init__(self, runs: List[Run]) -> None:
        self.runs = runs
        groups: Dict[int, List[Run]] = defaultdict(list)
        for run in runs:
            groups[run[2]].append(run)
        self._groups = [
            (
                np.array([s for s, _, _ in group], dtype=np.int64),
                np.array([d for _, d, _ in group], dtype=np.int64),
                length,
            )
            for length, group in groups.items()
        ]

    @property
    def is_empty(self) -> bool:
        return not self.runs

    @nvtx_range("RunPlan.map")
    def map(self, src_regions: SpecRegion, dst_regions: SpecRegion) -> List[SpecRegionPair]:
        src_ptrs = np.asarray(src_regions.memory.ptrs, dtype=np.int64)
        dst_ptrs = np.asarray(dst_regions.memory.ptrs, dtype=np.int64)
        if src_ptrs.size != dst_ptrs.size:
            raise ValueError(
                f"Number of regions of src({src_ptrs.size}) and dst({dst_ptrs.size}) must match"
            )
        return [
            SpecRegionPair(
                src=SpecRegion(
                    memory=MemRegionGroup(
                        ptrs=np.add.outer(src_ptrs, src_offs).ravel(), bytes_per_region=length
                    ),
                    spec=src_regions.spec,
                ),
                dst=SpecRegion(
                    memory=MemRegionGroup(
                        ptrs=np.add.outer(dst_ptrs, dst_offs).ravel(), bytes_per_region=length
                    ),
                    spec=dst_regions.spec,
                ),
            )
            for src_offs, dst_offs, length in self._groups
        ]


@dataclass
class TransferGroup:
    """Matched buffers of one local/peer ``(layer group, pool)`` pair that share a sender election."""

    self_layer_group: int
    peer_layer_group: int
    self_pool: PoolKey
    peer_pool: PoolKey
    grid: Tuple[str, ...]
    """The parallel dimensions the local layer group's buffers split over."""
    owner_factor: int
    """How many local ranks of the grid hold these bytes; one of them is elected to send."""
    cp_factor: int
    """How many local context-parallel ranks, outside a grid without "cp", hold these bytes for
    one peer rank; one of them is elected to send."""
    plan: RunPlan
    buffers: Tuple[int, ...]
    """Local buffer indices (within ``self_layer_group``) the group moves."""


def plan_transfers(
    self_ri: "RankInfo",
    peer_ri: "RankInfo",
    self_pages: RankPages,
    peer_pages: RankPages,
    excluded: FrozenSet[BufferRef] = frozenset(),
) -> List[TransferGroup]:
    """Every transfer group from this rank to one peer rank, skipping ``excluded`` buffers."""
    buckets: Dict[tuple, List[Tuple[int, int, List[Run]]]] = defaultdict(list)
    for (g, i), (pg, pi) in match_buffers(self_pages, peer_pages).items():
        if (g, i) in excluded:
            continue
        src, dst = self_pages.buffers(g)[i], peer_pages.buffers(pg)[pi]
        grid = self_pages.group(g).grid
        runs = buffer_runs(src, dst, grid, self_ri, peer_ri)
        if not runs:
            continue
        factor = _election(src, dst, grid, self_ri, peer_ri)
        cp_factor = _cp_election(self_pages.group(g), self_pages, peer_pages)
        pool, peer_pool = (src.pool_group, src.pool_index), (dst.pool_group, dst.pool_index)
        buckets[(g, pg, pool, peer_pool, grid, factor, cp_factor)].append(
            (src.slot_offset, i, runs)
        )

    groups = []
    for key in sorted(buckets):
        g, pg, pool, peer_pool, grid, factor, cp_factor = key
        # Local physical order, so buffers contiguous on both sides merge into one run.
        items = sorted(buckets[key])
        groups.append(
            TransferGroup(
                self_layer_group=g,
                peer_layer_group=pg,
                self_pool=pool,
                peer_pool=peer_pool,
                grid=grid,
                owner_factor=factor,
                cp_factor=cp_factor,
                plan=RunPlan(_merge([run for _, _, runs in items for run in runs])),
                buffers=tuple(i for _, i, _ in items),
            )
        )
    return groups


def owns_transfer(group: TransferGroup, self_ri: "RankInfo", peer_ri: "RankInfo") -> bool:
    """Whether this rank is the elected sender of ``group`` to ``peer_ri``: elected among the
    ranks of its grid holding the same bytes, and among its context-parallel ranks when the grid
    leaves them out."""
    if group.owner_factor > 1 and not elects_sender(
        shard_grid(group.grid, self_ri), peer_ri.dp_rank, group.owner_factor
    ):
        return False
    cp = ShardGrid(self_ri.cp_size, self_ri.cp_rank)
    return group.cp_factor <= 1 or elects_sender(cp, peer_ri.dp_rank, group.cp_factor)


# ---------------------------------------------------------------------------
# Peer compatibility
# ---------------------------------------------------------------------------


def check_peer_compatible(self_ri: "RankInfo", peer_ri: "RankInfo") -> bool:
    """Unit-level gate at peer registration; False (logged) when the two ranks' units cannot
    correspond."""
    return units.check_rank_compatible(self_ri, peer_ri)


def _hosted_and_replicated(pages: RankPages) -> Tuple[Set[int], Set[Tuple[int, str]]]:
    hosted: Set[int] = set()
    replicated_keys: Set[Tuple[int, str]] = set()
    for g in range(pages.num_layer_groups):
        for buffer in pages.buffers(g):
            hosted.add(buffer.layer)
            if replicated(buffer):
                replicated_keys.add((buffer.layer, buffer.role))
    return hosted, replicated_keys


def validate_peer_compatible(
    self_ri: "RankInfo",
    peer_ri: "RankInfo",
    self_pages: Optional[RankPages],
    peer_pages: Optional[RankPages],
) -> None:
    """Layout-level gate on both ranks' pages; raises ``ValueError`` naming the mismatch.

    Matched buffers declare the same layouts, and a replicated role is held by both sides on every
    layer both host (matching would otherwise drop a one-sided role and leave the receiver with
    zeroed state). State buffers must reshard (:func:`units.validate_state`).
    """
    if self_pages is None or peer_pages is None:
        return
    pairs = matched_pairs(self_pages, peer_pages, match_buffers(self_pages, peer_pages))
    for _, buffer, _, peer_buffer in pairs:
        context = _context(buffer)
        if len(buffer.segments) != len(peer_buffer.segments):
            raise ValueError(
                f"{context}: segment count differs (local={len(buffer.segments)}, "
                f"peer={len(peer_buffer.segments)})"
            )
        for i, (a, b) in enumerate(zip(buffer.segments, peer_buffer.segments)):
            _check_same_layout(a, b, f"{context} segment {i}")

    hosted, replicated_keys = _hosted_and_replicated(self_pages)
    peer_hosted, peer_replicated = _hosted_and_replicated(peer_pages)
    shared = hosted & peer_hosted
    differing = sorted(k for k in replicated_keys ^ peer_replicated if k[0] in shared)
    if differing:
        raise ValueError(
            f"replicated roles differ on overlapping layers (layer, role): {differing}"
        )
    units.validate_state(self_pages, self_ri, peer_ri, pairs)


def notify_peer_registered(
    self_ri: "RankInfo", peer_ri: "RankInfo", self_pages: RankPages, groups: List[TransferGroup]
) -> None:
    """Warn once when buffers split differently on the two sides move one fragment per row."""
    peer_pages = peer_ri.pages
    if peer_pages is None:
        return
    peer_index = _buffer_index(peer_pages)
    fragments, splits = 0, (None, None)
    for group in groups:
        for i in group.buffers:
            buffer = self_pages.buffers(group.self_layer_group)[i]
            pg, pi = peer_index[(buffer.layer, buffer.role)]
            peer_buffer = peer_pages.buffers(pg)[pi]
            for src, dst in zip(buffer.segments, peer_buffer.segments):
                if src.rows == 1:
                    continue
                runs = _segment_runs(src, dst, group.grid, self_ri, peer_ri, _context(buffer))
                if len(runs) > 1:
                    fragments += len(runs)
                    splits = (src.split_extent, dst.split_extent)
    units.warn_row_fragments(fragments, *splits)
