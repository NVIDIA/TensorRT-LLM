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
"""The one mapping algorithm: from two sides' regions to the byte runs that move.

1. Regions match by ``(spec, global layer id, role)`` (:func:`match_regions`).
2. For every buffer of a matched pair, the two ranks' spans along the split axis intersect
   (:func:`~.topology.shard_span`); an empty intersection means the two ranks are not paired.
3. The intersection becomes byte runs -- one per row of the axes outside the split axis.
4. Ranks holding identical bytes elect one sender (:func:`~.topology.elects_sender`).

Regions of the same ``(layer_group, pool)`` pair and election are merged into one
:class:`TransferGroup`, so per request the transfer extracts unit addresses once per group, not
once per region. Groups are cached per peer; per request only unit base addresses are added.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, FrozenSet, List, Optional, Sequence, Set, Tuple

import numpy as np

from tensorrt_llm import logger
from tensorrt_llm._torch.disaggregation.base.region import (
    MemRegionGroup,
    RegionMapperBase,
    SpecRegion,
    SpecRegionPair,
)
from tensorrt_llm._utils import nvtx_range

from .page_table import BufferDecl, PageTable, Region, RegionKey
from .registry import specs_in
from .spec import MatchedPair
from .topology import domain_of, elects_sender, replication_factor, shard_grid, shard_span

if TYPE_CHECKING:
    from tensorrt_llm._torch.disaggregation.native.rank_info import RankInfo

RegionRef = Tuple[int, int]
"""``(layer_group index, region index)`` within one page table."""

Run = Tuple[int, int, int]
"""``(src offset, dst offset, length)`` in bytes, relative to a unit's slot base."""


# ---------------------------------------------------------------------------
# Which regions pair up
# ---------------------------------------------------------------------------


def _region_index(page_table: PageTable) -> Dict[Tuple[str, int, str], RegionRef]:
    """``{(spec, layer, role): (layer_group, region)}``; a spec holds each (layer, role) once."""
    index: Dict[Tuple[str, int, str], RegionRef] = {}
    for gi, layer_group in enumerate(page_table.layer_groups):
        for ri, region in enumerate(layer_group.regions):
            key = (layer_group.spec, region.layer, region.role)
            if key in index:
                raise ValueError(
                    f"{layer_group.spec} region (layer={region.layer}, role={region.role!r}) appears in "
                    f"layer groups {index[key][0]} and {gi}; a spec holds each (layer, role) once"
                )
            index[key] = (gi, ri)
    return index


def check_page_table(page_table: PageTable) -> None:
    """Structural checks of a freshly built table: known specs, unique regions, valid pools."""
    specs_in(page_table)
    _region_index(page_table)
    for layer_group in page_table.layer_groups:
        for region in layer_group.regions:
            if not 0 <= region.pool < len(page_table.pools):
                raise ValueError(
                    f"{layer_group.spec} region {region.key} names missing pool {region.pool}"
                )


def match_regions(
    self_page_table: Optional[PageTable], peer_page_table: Optional[PageTable]
) -> Dict[RegionRef, RegionRef]:
    """Each local region paired with the peer region of the same spec, layer and role.

    Regions only one side holds are a legal skip (PP slices, one-sided MTP layers); a role that
    must not be one-sided is rejected by :func:`validate_peer_compatible` instead.
    """
    if self_page_table is None or peer_page_table is None:
        return {}
    specs_in(self_page_table)
    specs_in(peer_page_table)
    peer_index = _region_index(peer_page_table)
    mapping: Dict[RegionRef, RegionRef] = {}
    for gi, layer_group in enumerate(self_page_table.layer_groups):
        for ri, region in enumerate(layer_group.regions):
            ref = peer_index.get((layer_group.spec, region.layer, region.role))
            if ref is None:
                continue
            peer_layer_group = peer_page_table.layer_groups[ref[0]]
            if peer_layer_group.kind != layer_group.kind:
                raise ValueError(
                    f"{layer_group.spec} region {region.key} is a {layer_group.kind.name} unit locally but "
                    f"{peer_layer_group.kind.name} on the peer"
                )
            mapping[(gi, ri)] = ref
    return mapping


def matched_pairs(
    self_page_table: PageTable, peer_page_table: PageTable, mapping: Dict[RegionRef, RegionRef]
) -> List[MatchedPair]:
    pairs = []
    for (gi, ri), (pgi, pri) in mapping.items():
        layer_group, peer_layer_group = (
            self_page_table.layer_groups[gi],
            peer_page_table.layer_groups[pgi],
        )
        pairs.append(
            (layer_group, layer_group.regions[ri], peer_layer_group, peer_layer_group.regions[pri])
        )
    return pairs


# ---------------------------------------------------------------------------
# Byte runs of one buffer pair, one region pair
# ---------------------------------------------------------------------------


def _check_same_layout(src: BufferDecl, dst: BufferDecl, context: str) -> None:
    if src.split_axis != dst.split_axis or src.axis_names != dst.axis_names:
        raise ValueError(
            f"{context}: layout differs (local axes={src.axis_names} split={src.split_axis!r}, "
            f"peer axes={dst.axis_names} split={dst.split_axis!r})"
        )


def _buffer_runs(
    src: BufferDecl,
    dst: BufferDecl,
    domain: str,
    self_ri: "RankInfo",
    peer_ri: "RankInfo",
    context: str,
) -> List[Run]:
    """Runs of one buffer pair, relative to each buffer's start; empty when not paired."""
    _check_same_layout(src, dst, context)
    src_grid = shard_grid(domain, self_ri)
    dst_grid = shard_grid(domain, peer_ri)
    virtual = min(src.units * src_grid.size, dst.units * dst_grid.size)
    src_lo, src_hi = shard_span(src.units, src_grid, virtual)
    dst_lo, dst_hi = shard_span(dst.units, dst_grid, virtual)
    lo, hi = max(src_lo, dst_lo), min(src_hi, dst_hi)
    if lo >= hi:
        return []
    if (lo, hi) == (src_lo, src_hi) == (dst_lo, dst_hi):
        # Both ranks hold exactly the same shard: the buffer moves whole.
        if src.nbytes != dst.nbytes:
            raise ValueError(
                f"{context}: cache region size mismatch: local={src.nbytes}, peer={dst.nbytes} "
                "(a sharded role without declared geometry cannot be resharded)"
            )
        return [(0, 0, src.nbytes)]
    if src.rows != dst.rows:
        raise ValueError(
            f"{context}: {src.split_axis!r} slicing requires equal row counts on both sides "
            f"(local={src.rows}, peer={dst.rows}); token-major layouts need equal tokens_per_block"
        )
    unit_bytes = src.unit_bytes(f"{context} (local)")
    peer_unit_bytes = dst.unit_bytes(f"{context} (peer)")
    if unit_bytes != peer_unit_bytes:
        raise ValueError(
            f"{context}: bytes per {src.split_axis!r} unit mismatch: local={unit_bytes}, "
            f"peer={peer_unit_bytes}"
        )
    src_row = src.units * unit_bytes
    dst_row = dst.units * unit_bytes
    length = (hi - lo) * unit_bytes
    return [
        (
            row * src_row + (lo - src_lo) * unit_bytes,
            row * dst_row + (lo - dst_lo) * unit_bytes,
            length,
        )
        for row in range(src.rows)
    ]


def region_runs(
    src: Region, dst: Region, domain: str, self_ri: "RankInfo", peer_ri: "RankInfo", context: str
) -> List[Run]:
    """Slot-relative runs of one matched region pair, whose buffers are split over ``domain``;
    empty when the two ranks are not paired."""
    if len(src.buffers) != len(dst.buffers):
        raise ValueError(
            f"{context}: buffer count mismatch: local={len(src.buffers)}, peer={len(dst.buffers)}"
        )
    runs: List[Run] = []
    src_off, dst_off = src.offset, dst.offset
    for i, (s_buf, d_buf) in enumerate(zip(src.buffers, dst.buffers)):
        for s, d, n in _buffer_runs(
            s_buf, d_buf, domain, self_ri, peer_ri, f"{context} buffer {i}"
        ):
            runs.append((src_off + s, dst_off + d, n))
        src_off += s_buf.nbytes
        dst_off += d_buf.nbytes
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


def _election(src: Region, dst: Region, domain: str, self_ri: "RankInfo", peer_ri: "RankInfo"):
    first, peer_first = src.buffers[0], dst.buffers[0]
    return replication_factor(
        first.units,
        shard_grid(domain, self_ri),
        peer_first.units,
        shard_grid(domain, peer_ri),
    )


# ---------------------------------------------------------------------------
# Transfer groups: runs merged per (layer group, pool) pair and election
# ---------------------------------------------------------------------------


class RunPlan(RegionMapperBase):
    """Slot-relative byte runs, ready to expand over unit base addresses.

    ``map`` takes one base address per unit on each side (``KVRegionExtractorV1.extract``) and
    returns one :class:`SpecRegionPair` per distinct run length, so the fragment arrays stay
    uniform-size like the transfer path expects.
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

    @property
    def bytes_per_unit(self) -> int:
        return sum(n for _, _, n in self.runs)

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
    """Matched regions of one local/peer ``(layer_group, pool)`` pair that share a sender election."""

    self_layer_group: int
    peer_layer_group: int
    self_pool: int
    peer_pool: int
    domain: str
    owner_factor: int
    """How many local ranks hold these bytes; one of them is elected to send."""
    plan: RunPlan
    regions: Tuple[int, ...]
    """Local region indices (within ``self_layer_group``) the group moves."""


def plan_transfers(
    self_ri: "RankInfo",
    peer_ri: "RankInfo",
    self_page_table: PageTable,
    peer_page_table: PageTable,
    excluded: FrozenSet[RegionRef] = frozenset(),
) -> List[TransferGroup]:
    """Every transfer group from this rank to one peer rank, skipping ``excluded`` regions."""
    mapping = match_regions(self_page_table, peer_page_table)
    buckets: Dict[tuple, List[Tuple[int, int, List[Run]]]] = defaultdict(list)
    for (gi, ri), (pgi, pri) in mapping.items():
        if (gi, ri) in excluded:
            continue
        layer_group, peer_layer_group = (
            self_page_table.layer_groups[gi],
            peer_page_table.layer_groups[pgi],
        )
        src, dst = layer_group.regions[ri], peer_layer_group.regions[pri]
        context = f"{layer_group.spec} region (layer={src.layer}, role={src.role!r})"
        domain = domain_of(layer_group.kind)
        runs = region_runs(src, dst, domain, self_ri, peer_ri, context)
        if not runs:
            continue
        factor = _election(src, dst, domain, self_ri, peer_ri)
        buckets[(gi, pgi, src.pool, dst.pool, domain, factor)].append((src.offset, ri, runs))

    groups = []
    for key in sorted(buckets):
        gi, pgi, pool, peer_pool, domain, factor = key
        # Local physical order, so regions contiguous on both sides merge into one run.
        items = sorted(buckets[key])
        groups.append(
            TransferGroup(
                self_layer_group=gi,
                peer_layer_group=pgi,
                self_pool=pool,
                peer_pool=peer_pool,
                domain=domain,
                owner_factor=factor,
                plan=RunPlan(_merge([run for _, _, runs in items for run in runs])),
                regions=tuple(ri for _, ri, _ in items),
            )
        )
    return groups


def owns_transfer(group: TransferGroup, self_ri: "RankInfo", peer_ri: "RankInfo") -> bool:
    """Whether this rank is the elected sender of ``group`` to ``peer_ri``."""
    if group.owner_factor <= 1:
        return True
    return elects_sender(shard_grid(group.domain, self_ri), peer_ri.dp_rank, group.owner_factor)


# ---------------------------------------------------------------------------
# Peer compatibility
# ---------------------------------------------------------------------------


def check_peer_compatible(self_ri: "RankInfo", peer_ri: "RankInfo") -> bool:
    """Topology-level gate of the specs both sides run; False when any rejects the peer."""
    specs = specs_in(self_ri.page_table)
    specs += [s for s in specs_in(peer_ri.page_table) if s not in specs]
    return all(spec.check_rank_compatible(self_ri, peer_ri) for spec in specs)


def _hosted_and_replicated(
    page_table: PageTable, spec_name: str
) -> Tuple[Set[int], Set[RegionKey]]:
    hosted: Set[int] = set()
    replicated: Set[RegionKey] = set()
    for layer_group in page_table.layer_groups:
        if layer_group.spec != spec_name:
            continue
        for region in layer_group.regions:
            hosted.add(region.layer)
            if region.replicated:
                replicated.add(region.key)
    return hosted, replicated


def validate_peer_compatible(
    self_ri: "RankInfo",
    peer_ri: "RankInfo",
    self_page_table: Optional[PageTable],
    peer_page_table: Optional[PageTable],
) -> None:
    """Layout-level gate on both page tables; raises ``ValueError`` naming the mismatch.

    Generic over every spec: both sides run the same specs, matched regions declare the same
    layouts, and a replicated role is held by both sides on every layer both host (matching would
    otherwise drop a one-sided role and leave the receiver with zeroed state). Each spec then checks
    its own matched regions (:meth:`CacheSpec.validate_peer`).
    """
    if self_page_table is None or peer_page_table is None:
        return
    self_specs, peer_specs = specs_in(self_page_table), specs_in(peer_page_table)
    if self_specs and peer_specs and set(self_specs) != set(peer_specs):
        raise ValueError(
            "cache specs differ: local="
            f"{sorted(s.name for s in self_specs)}, peer={sorted(s.name for s in peer_specs)}"
        )
    mapping = match_regions(self_page_table, peer_page_table)
    pairs = matched_pairs(self_page_table, peer_page_table, mapping)
    for layer_group, region, _, peer_region in pairs:
        context = f"{layer_group.spec} region (layer={region.layer}, role={region.role!r})"
        if len(region.buffers) != len(peer_region.buffers):
            raise ValueError(
                f"{context}: buffer count differs (local={len(region.buffers)}, "
                f"peer={len(peer_region.buffers)})"
            )
        for i, (a, b) in enumerate(zip(region.buffers, peer_region.buffers)):
            _check_same_layout(a, b, f"{context} buffer {i}")

    for spec in self_specs:
        hosted, replicated = _hosted_and_replicated(self_page_table, spec.name)
        peer_hosted, peer_replicated = _hosted_and_replicated(peer_page_table, spec.name)
        shared = hosted & peer_hosted
        differing = sorted(k for k in replicated ^ peer_replicated if k[0] in shared)
        if differing:
            raise ValueError(
                f"{spec.name}: replicated roles differ on overlapping layers "
                f"(layer, role): {differing}"
            )
        spec.validate_peer(self_ri, peer_ri, [p for p in pairs if p[0].spec == spec.name])


def log_incompatible(spec_name: str, reason: str, **details) -> None:
    text = ", ".join(f"{k}={v!r}" for k, v in details.items())
    logger.warning(
        "%s", f"{spec_name}: incompatible peer: {reason}" + (f"; {text}" if text else "")
    )


def notify_peer_registered(
    self_ri: "RankInfo", peer_ri: "RankInfo", self_page_table: PageTable, groups: Sequence
) -> None:
    for spec in specs_in(self_page_table):
        spec.on_peer_registered(
            self_ri,
            peer_ri,
            [
                g
                for g in groups
                if self_page_table.layer_groups[g.self_layer_group].spec == spec.name
            ],
        )
