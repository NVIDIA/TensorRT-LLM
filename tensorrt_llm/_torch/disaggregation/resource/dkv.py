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
"""How the context ranks of a distributed KV (DKV) group send their KV (``KVCacheManagerV2(dkv=True)``).

Under DKV of ``N`` ranks a manager block is a DKV block of ``N`` physical pages, and a
pool of a DKV life cycle holds global page ``g`` in one of two ways:

- sharded (the ``page_split`` layout): only rank ``g % N`` holds it, in that rank's slot of DKV block ``g // N``;
- expanded: every rank holds every page of the block, as sub-page ``g % N`` of each buffer of its
  own slot. The page table describes such a pool page by page (``split_expanded_dkv_views``).

A context rank writes the pages ``g`` with ``g % N == rank`` straight from its registered pool; an
expanded view is split the same way. To a plain TP generation server it writes them to every rank
(fan-out, ``PeerRegistrar.dkv_fan_out``), so each page goes once to each receiver. To a Helix CP
generation server of ``M`` ranks, ``M`` a multiple or a divisor of ``N``, it writes each page to
the Helix rank that stores it (``PeerRegistrar.dkv_helix``; every pair keeps the replicated
overlap, and a pair that shares no page writes no paged KV): Helix stores global page ``g`` on
CP rank ``g % M`` as local page ``g // M`` (``helix_local_len``, the partial last page included),
so DKV rank ``r`` and Helix rank ``h`` share the pages with ``g % N == r`` and ``g % M == h``
(``dkv_helix_page_ids``):

- ``M == N``: the DKV's own layout. Rank ``r`` writes Helix rank ``r`` alone, its pages land in
  order, and contiguous runs merge (``merge_runs``);
- ``M == k * N``: Helix rank ``h`` stores pages of DKV rank ``h % N`` only, and rank ``r`` writes
  Helix ranks ``r, r + N, ...``, each every ``M``-th page;
- ``N == k * M``: Helix rank ``h`` stores the pages of the ``k`` DKV ranks ``h, h + M, ...``
  interleaved, and rank ``r`` writes Helix rank ``r % M`` alone.

Per global page the chunk carries this rank's slot of DKV block ``g // N``
(``Chunk.block_ids_per_layer_groups``), from the slots the group exchanges every step
(``Dkv.peer_pages``).
"""

import math
from dataclasses import dataclass
from typing import Dict, FrozenSet, Optional, Sequence, Tuple

import numpy as np

from tensorrt_llm._torch.disaggregation.resource.page import (
    BUFFER_ENTRY_DTYPE,
    KVCachePageTable,
    PhysicalPool,
    PoolView,
)
from tensorrt_llm._torch.disaggregation.resource.utils import get_physical_pool
from tensorrt_llm._torch.pyexecutor.kv_cache.dkv import Dkv


def manager_dkv(manager) -> Optional[Dkv]:
    """The DKV of a cache manager (``KVCacheManagerV2(dkv=True)``), or None."""
    dkv = getattr(manager, "dkv", None)
    return dkv if isinstance(dkv, Dkv) else None


@dataclass(frozen=True)
class DkvView:
    """How a view of a DKV life cycle holds global page ``g``: in this rank's slot when it owns
    the page (sharded, ``pages_per_slot == 0``), or at page ``slot * pages_per_slot + g % N`` of the
    page-by-page pool (expanded)."""

    pages_per_slot: int = 0

    @property
    def sharded(self) -> bool:
        return self.pages_per_slot == 0


@dataclass(frozen=True)
class DkvLayout:
    """A DKV context rank's transfer layout: its group and every view of a DKV life cycle."""

    size: int
    rank: int
    tokens_per_block: int
    life_cycles: FrozenSet[int]
    views: Dict[Tuple[int, int], DkvView]


def split_expanded_dkv_views(manager, page_table: KVCachePageTable) -> None:
    """Describe the expanded pools of the DKV life cycles page by page, in place.

    A buffer with a page expansion holds one sub-page per physical page of a DKV block. Its
    view gets a pool of one physical page per slot over the same memory (appended to the pool
    group), and one page per layer, like the generation side's view of the same buffers.
    """
    life_cycles = set(manager.dkv_life_cycles)
    for pg_desc in manager.impl.pool_group_descs:
        for variant in pg_desc.slot_desc.variants:
            lg_id = int(variant.layer_group_id)
            if lg_id not in life_cycles:
                continue
            lg = page_table.layer_groups[lg_id]
            pools = page_table.pool_groups[lg.pool_group_idx].pools
            for view_idx, view in enumerate(lg.pool_views):
                buffer = variant.coalesced_buffers[view.pool_idx]
                expansions = {
                    int(manager.impl.get_page_index_converter(b.layer_id, b.role).expansion)
                    for b in buffer.buffer_ids
                }
                if len(expansions) != 1:
                    raise ValueError(
                        f"DKV: pool {view.pool_idx} of layer group {lg_id} mixes page "
                        f"expansions {sorted(expansions)}"
                    )
                expansion = expansions.pop()
                if expansion == 1:
                    continue
                buffer_bytes = int(buffer.single_buffer_size)
                pool = pools[view.pool_idx]
                if buffer_bytes % expansion or pool.slot_bytes % (buffer_bytes // expansion):
                    raise ValueError(
                        f"DKV: pool {view.pool_idx} of layer group {lg_id} does not hold "
                        f"whole pages ({buffer_bytes} B buffers of {expansion} pages)"
                    )
                page = buffer_bytes // expansion
                pools.append(
                    PhysicalPool(
                        base_address=pool.base_address,
                        slot_bytes=page,
                        num_slots=pool.num_slots * (pool.slot_bytes // page),
                    )
                )
                entries = np.array(view.buffer_entries, dtype=BUFFER_ENTRY_DTYPE)
                entries["size"] = page
                lg.pool_views[view_idx] = PoolView(
                    pool_idx=len(pools) - 1,
                    buffer_entries=entries,
                    pool_role=view.pool_role,
                    mapper_kind=view.mapper_kind,
                    bytes_per_layer=page,
                    section_bytes=view.section_bytes,
                    bytes_per_head=view.bytes_per_head,
                )


def build_dkv_layout(manager, page_table: KVCachePageTable) -> Optional[DkvLayout]:
    """The transfer layout of a DKV manager's page table, or None without a dkv."""
    dkv = manager_dkv(manager)
    if dkv is None:
        return None
    life_cycles = frozenset(int(lc) for lc in manager.dkv_life_cycles)
    views: Dict[Tuple[int, int], DkvView] = {}
    for lg_id in life_cycles:
        lg = page_table.layer_groups[lg_id]
        group_pools = page_table.pool_groups[lg.pool_group_idx].pools
        for view_idx, view in enumerate(lg.pool_views):
            pool = get_physical_pool(page_table, lg_id, view.pool_idx)
            # An expanded view's per-page pool shares its base with the pool it was split from.
            whole = [
                p
                for p in group_pools
                if p is not pool
                and p.base_address == pool.base_address
                and p.slot_bytes > pool.slot_bytes
                and pool.slot_bytes == view.bytes_per_layer
            ]
            views[(lg_id, view_idx)] = DkvView(
                pages_per_slot=whole[0].slot_bytes // pool.slot_bytes if whole else 0
            )
    return DkvLayout(
        size=int(dkv.size),
        rank=int(dkv.rank),
        tokens_per_block=int(manager.tokens_per_block),
        life_cycles=life_cycles,
        views=views,
    )


def own_page_slots(own_blocks: Sequence[int], size: int, num_pages: int) -> np.ndarray:
    """Per global page ``g < num_pages``: this rank's slot of DKV block ``g // N`` (``-1`` where
    it has none), from this rank's slots per DKV block."""
    blocks = np.arange(num_pages, dtype=np.int64) // size
    own = np.full(num_pages, -1, dtype=np.int64)
    held = blocks < len(own_blocks)
    own[held] = np.asarray(own_blocks, dtype=np.int64)[blocks[held]]
    return own


def dkv_page_ids(
    view: DkvView,
    size: int,
    rank: int,
    own: np.ndarray,
    dst: np.ndarray,
    dst_start_page: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """``(source ids, destination ids)`` of the pages this rank sends: those it owns
    (``g % N == rank``) that both sides hold. Source ids index the view's pool."""
    if own.size != dst.size:
        raise ValueError(
            f"DKV page tables differ in length: src={own.size} dst={dst.size}; both sides "
            "must span the same prompt"
        )
    pages = np.arange(own.size, dtype=np.int64)
    mask = (own >= 0) & (dst >= 0) & (pages % size == rank)
    if dst_start_page > 0:
        mask[:dst_start_page] = False
    pages = pages[mask]
    if view.sharded:
        return own[pages], dst[pages]
    return own[pages] * view.pages_per_slot + pages % size, dst[pages]


def dkv_helix_shares_pages(size: int, rank: int, helix_size: int, helix_rank: int) -> bool:
    """Whether DKV rank ``rank`` of ``size`` owns a page Helix CP rank ``helix_rank`` of
    ``helix_size`` stores: some global page ``g`` with ``g % size == rank`` and
    ``g % helix_size == helix_rank``, which is the two ranks agreeing modulo
    ``gcd(size, helix_size)``."""
    common = math.gcd(size, helix_size)
    return rank % common == helix_rank % common


def dkv_helix_page_ids(
    view: DkvView,
    size: int,
    rank: int,
    own: np.ndarray,
    dst: np.ndarray,
    dst_start_page: int = 0,
    *,
    helix_size: Optional[int] = None,
    helix_rank: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """``(source ids, destination ids)`` of the pages DKV rank ``rank`` of ``size`` sends to
    Helix CP rank ``helix_rank`` of ``helix_size`` (by default ``rank`` of ``size``).

    Those are the global pages ``g`` with ``g % size == rank`` and ``g % helix_size ==
    helix_rank`` that both sides hold: one every ``lcm(size, helix_size)`` pages, none when the
    two ranks share no page (``dkv_helix_shares_pages``).

    Args:
        view: how this rank's view of the DKV life cycle holds a page.
        size: the DKV group's size ``N``.
        rank: this rank's place in the DKV group.
        own: per global page of the prompt, this rank's slot of DKV block ``g // N``
            (``own_page_slots``), ``-1`` where it has none; ``[num_pages]`` int64.
        dst: the Helix rank's own table, entry ``j`` = global page ``helix_rank + j * helix_size``,
            ``-1`` where it holds none; int64.
        dst_start_page: the first global page the receiver takes.
        helix_size: the Helix group's CP size ``M``.
        helix_rank: the receiver's Helix CP rank.

    Returns:
        Source ids, which index the view's pool, and destination ids, both int64 and in global
        page order.
    """
    helix_size = size if helix_size is None else helix_size
    helix_rank = rank if helix_rank is None else helix_rank
    stored = len(range(helix_rank, own.size, helix_size))
    if stored != dst.size:
        raise ValueError(
            f"Helix rank {helix_rank} of {helix_size} stores {stored} of the prompt's {own.size} "
            f"pages, its table {dst.size}; both sides must span the same prompt"
        )
    step = math.lcm(size, helix_size)
    first = next((g for g in range(rank, step, size) if g % helix_size == helix_rank), None)
    if first is None:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    pages = np.arange(first, own.size, step, dtype=np.int64)
    local = pages // helix_size
    mask = (own[pages] >= 0) & (dst[local] >= 0) & (pages >= dst_start_page)
    pages, local = pages[mask], local[mask]
    if view.sharded:
        return own[pages], dst[local]
    return own[pages] * view.pages_per_slot + rank, dst[local]


def merge_runs(
    src: np.ndarray, dst: np.ndarray, sizes: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Copies ``(src, dst, sizes)`` merged where consecutive ones (in source order) continue each
    other on both sides: one write per contiguous run."""
    if src.size < 2:
        return src, dst, sizes
    order = np.argsort(src, kind="stable")
    src, dst, sizes = src[order], dst[order], sizes[order]
    starts = np.ones(src.size, dtype=bool)
    starts[1:] = (src[1:] != src[:-1] + sizes[:-1]) | (dst[1:] != dst[:-1] + sizes[:-1])
    index = np.flatnonzero(starts)
    return src[index], dst[index], np.add.reduceat(sizes, index)
