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
"""Conformance checks a cache kind must pass before it ships.

``check_page_table_round_trip``
    The kind's page table survives the wire (``to_dict`` / ``from_dict``).

``check_resharding``
    The end-to-end property of the P/D transfer, checked byte by byte against an independent model
    of sharding: on every receiving rank, every byte of every region the senders also hold is
    written exactly once, by the elected sender, from the source byte that holds the same
    ``(layer, role, buffer, row, global unit, offset in unit)``. A rank's local unit ``u`` of a split
    axis is global unit ``rank * G // grid + u`` (``G`` the model's true extent: heads, channels,
    ...; ranks beyond ``G`` duplicate units, as GQA heads do).

Both take plain page tables and ``RankInfo``-like objects, so a new kind can be checked on a CPU
with synthetic tables before any manager exists.
"""

from __future__ import annotations

from collections import Counter
from typing import Callable, Dict, List, Sequence, Tuple

from ..correspondence import owns_transfer, plan_transfers
from ..page_table import BufferDecl, PageTable, Region
from ..registry import specs_in
from ..topology import rank_overlap, shard_grid

GlobalUnits = Callable[[Region, int, BufferDecl], int]
"""The model's true extent of buffer ``index`` of a region along its split axis."""

ByteId = Tuple[int, str, int, int, int, int]
"""``(global layer id, role, buffer index, row, global unit, offset in unit)``."""

Address = Tuple[int, int]
"""``(pool, offset within a unit's slot)``."""


def check_page_table_round_trip(page_table: PageTable) -> None:
    restored = PageTable.from_dict(page_table.to_dict())
    if restored != page_table:
        raise AssertionError("page table does not survive to_dict/from_dict")
    specs_in(restored)


def _byte_ids(region: Region, ri, global_units: GlobalUnits) -> Dict[Address, ByteId]:
    """Slot address -> identity of every byte the region holds on one rank."""
    ids: Dict[Address, ByteId] = {}
    offset = region.offset
    for index, decl in enumerate(region.buffers):
        grid = shard_grid(decl.domain, ri)
        if decl.split_axis is None:
            for b in range(decl.nbytes):
                ids[(region.pool, offset + b)] = (region.layer, region.role, index, 0, 0, b)
        else:
            unit_bytes = decl.unit_bytes("conformance")
            first = (grid.rank * global_units(region, index, decl)) // grid.size
            for b in range(decl.nbytes):
                row, rest = divmod(b, decl.units * unit_bytes)
                unit, within = divmod(rest, unit_bytes)
                ids[(region.pool, offset + b)] = (
                    region.layer,
                    region.role,
                    index,
                    row,
                    first + unit,
                    within,
                )
        offset += decl.nbytes
    return ids


def _space_byte_ids(page_table: PageTable, space: int, ri, global_units) -> Dict[Address, ByteId]:
    ids: Dict[Address, ByteId] = {}
    for region in page_table.spaces[space].regions:
        ids.update(_byte_ids(region, ri, global_units))
    return ids


def check_resharding(
    senders: Sequence,
    receivers: Sequence,
    global_units: GlobalUnits,
    *,
    sender_dp_rank: int = 0,
) -> int:
    """Check every receiver gets each byte exactly once from the right source byte.

    ``senders`` / ``receivers`` are ``RankInfo`` objects with ``page_table`` set, one per rank of
    each instance. Senders are the ranks a receiver addresses (``rank_overlap``), as at runtime.
    Returns the number of bytes checked.
    """
    by_rank = {ri.instance_rank: ri for ri in senders}
    checked = 0
    for dst_ri in receivers:
        dst_pt = dst_ri.page_table
        writes: Counter = Counter()
        expected: Dict[Tuple[int, Address], ByteId] = {}
        overlap = rank_overlap(dst_ri, senders[0], sender_dp_rank)
        for flat in overlap.ranks:
            src_ri = by_rank[flat]
            src_pt = src_ri.page_table
            src_layers = {
                (space.kind, region.key) for space in src_pt.spaces for region in space.regions
            }
            for group in plan_transfers(src_ri, dst_ri, src_pt, dst_pt):
                if not owns_transfer(group, src_ri, dst_ri):
                    continue
                src_ids = _space_byte_ids(src_pt, group.self_space, src_ri, global_units)
                dst_ids = _space_byte_ids(dst_pt, group.peer_space, dst_ri, global_units)
                for s, d, n in group.plan.runs:
                    for b in range(n):
                        src_id = src_ids[(group.self_pool, s + b)]
                        dst_id = dst_ids[(group.peer_pool, d + b)]
                        if src_id != dst_id:
                            raise AssertionError(
                                f"rank {flat} -> {dst_ri.instance_rank} space {group.peer_space}: "
                                f"byte {d + b} of pool {group.peer_pool} gets {src_id}, expected "
                                f"{dst_id}"
                            )
                        writes[(group.peer_space, (group.peer_pool, d + b))] += 1
            for space_index, space in enumerate(dst_pt.spaces):
                for region in space.regions:
                    if (space.kind, region.key) not in src_layers:
                        continue
                    for address, byte_id in _byte_ids(region, dst_ri, global_units).items():
                        expected[(space_index, address)] = byte_id
        missing = [k for k in expected if writes[k] == 0]
        repeated = [k for k, count in writes.items() if count > 1]
        if missing or repeated:
            raise AssertionError(
                f"receiver rank {dst_ri.instance_rank}: {len(missing)} bytes never written "
                f"(first {missing[:3]}), {len(repeated)} written more than once "
                f"(first {repeated[:3]})"
            )
        checked += len(expected)
    return checked


def all_ranks(make_rank: Callable[..., object], tp: int, pp: int = 1, cp: int = 1) -> List:
    """``make_rank(pp_rank=, tp_rank=, cp_rank=)`` for every rank of one instance, CP-minor."""
    return [
        make_rank(pp_rank=p, tp_rank=t, cp_rank=c)
        for p in range(pp)
        for t in range(tp)
        for c in range(cp)
    ]
