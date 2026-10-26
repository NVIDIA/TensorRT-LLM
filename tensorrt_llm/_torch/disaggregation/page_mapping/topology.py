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
"""Rank topology: which ranks talk to each other and where a rank stands in a layer group's grid.

Each layer group names the parallel dimensions its split buffers spread over (``LayerGroupDesc.grid``);
:func:`shard_grid` places a rank in that grid from its published topology. Which bytes a rank holds
and which rank sends them follow from the grid in ``correspondence``.

Every range here is ``page_layout.held_span``: a rank's slice of a byte axis, its peer TP and CP
ranks (:func:`rank_overlap`) and, in the lender, its share of KV heads. Peer ranks are chosen from
the topology alone, since a receiver knows one sender rank's layout only; they cover every pair
whose bytes intersect, for any grid ``page_layout`` allows, and a paired rank with nothing to send
still reports.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Sequence, Tuple

from tensorrt_llm._torch.pyexecutor.kv_cache.page_layout import grid_position, held_span

if TYPE_CHECKING:
    from tensorrt_llm._torch.disaggregation.native.rank_info import RankInfo


@dataclass(frozen=True)
class ShardGrid:
    """``size`` ranks split a layer group's content; this rank is number ``rank`` of them."""

    size: int
    rank: int

    def __post_init__(self) -> None:
        if self.size < 1 or not 0 <= self.rank < self.size:
            raise ValueError(f"invalid shard grid: size={self.size}, rank={self.rank}")


def shard_grid(grid: Sequence[str], ri: "RankInfo") -> ShardGrid:
    """Where ``ri`` stands in the grid of ranks ``grid`` names (``page_layout.grid_position``)."""
    size, rank = grid_position(ri.pages.layout.topology, grid)
    return ShardGrid(size=size, rank=rank)


def shard_span(units: int, grid: ShardGrid, virtual_units: int) -> Tuple[int, int]:
    """Half-open range of a ``virtual_units``-long axis held by ``grid.rank``.

    ``virtual_units`` is the smaller of the two sides' ``units * grid.size``. The side with fewer
    copies of each unit owns disjoint, consecutive ranges; the side that duplicates units (GQA heads
    on more ranks than heads, an MLA latent on every rank) maps several ranks onto the same range.
    """
    return held_span(units, grid.size, grid.rank, virtual_units)


def replication_factor(
    self_units: int, self_grid: ShardGrid, peer_units: int, peer_grid: ShardGrid
) -> int:
    """How many local ranks hold the same bytes one peer rank needs.

    Exactly one of them sends; see :func:`elects_sender`.
    """
    return max(1, (self_units * self_grid.size) // max(1, peer_units * peer_grid.size))


def elects_sender(self_grid: ShardGrid, peer_dp_rank: int, factor: int) -> bool:
    """Pick one sender among ``factor`` duplicate holders, rotated by the receiver's DP rank.

    Rotating by the destination DP rank spreads the duplicated traffic across local ranks when the
    generation side runs several DP groups (mirrors the C++ ``needSendCache`` pairing).
    """
    return self_grid.rank % factor == peer_dp_rank % factor


# ---------------------------------------------------------------------------
# Which peer ranks this rank exchanges KV with
# ---------------------------------------------------------------------------


@dataclass
class PeerOverlap:
    overlap_pp_size: int = 0
    overlap_tp_size: int = 0
    overlap_cp_size: int = 0
    duplicate_head_factor: int = 1
    peer_duplicate_head_factor: int = 1
    ranks: List[int] = field(default_factory=list)


def paired_ranks(self_size: int, self_rank: int, peer_size: int) -> range:
    """The peer ranks of a dimension of ``peer_size`` ranks that rank ``self_rank`` of
    ``self_size`` pairs with: those whose one-unit :func:`held_span` on the shorter side's axis
    meets its own. One side's size must divide the other's."""
    total = min(self_size, peer_size)
    span = held_span(1, self_size, self_rank, total)
    peers = [p for p in range(peer_size) if held_span(1, peer_size, p, total) == span]
    return range(peers[0], peers[-1] + 1)


def _block_copies(ri: "RankInfo") -> int:
    """How many copies of a block's content the rank's grid holds, as ``units * grid size`` of its
    most finely split block buffer (a buffer every rank holds alike counts one unit); 0 for a rank
    without block units (an attention-free hybrid PP stage)."""
    pages = ri.pages
    if pages is None:
        return 0
    copies = 0
    for g, desc in enumerate(pages.layout.layer_groups):
        if desc.unit != "block":
            continue
        size, _ = grid_position(pages.layout.topology, desc.grid)
        for buffer in pages.buffers(g):
            copies = max(copies, (buffer.segments[0].split_extent or 1) * size)
    return copies


def head_duplication(self_ri: "RankInfo", peer_ri: "RankInfo") -> Tuple[int, int]:
    """``(local, peer)`` duplication factors of block content between two ranks: how many ranks of
    one side hold the bytes one rank of the other side needs (GQA heads on more ranks than heads,
    an MLA latent on every rank).

    A rank without block units reports no copies; no duplication applies to it.
    """
    a, b = _block_copies(self_ri), _block_copies(peer_ri)
    if a == 0 or b == 0:
        return 1, 1
    return max(1, a // b), max(1, b // a)


def rank_overlap(self_ri: "RankInfo", peer_ri: "RankInfo", peer_dp_rank: int) -> PeerOverlap:
    """The peer ranks (in ``peer_dp_rank``'s DP group) whose layers and shards overlap ours."""
    self_start_layer = sum(self_ri.layer_num_per_pp[: self_ri.pp_rank])
    self_end_layer = self_start_layer + self_ri.layer_num_per_pp[self_ri.pp_rank]

    pre = 0
    tgt_pp_ranks: List[int] = []
    for p in range(peer_ri.pp_size):
        peer_start_layer = pre
        peer_end_layer = peer_start_layer + peer_ri.layer_num_per_pp[p]
        if self_start_layer < peer_end_layer and self_end_layer > peer_start_layer:
            tgt_pp_ranks.append(p)
        pre += peer_ri.layer_num_per_pp[p]
    if not tgt_pp_ranks:
        return PeerOverlap()

    peer_start_pp = tgt_pp_ranks[0]
    overlap_pp_size = len(tgt_pp_ranks)

    self_tp_per_dp = self_ri.tp_size_per_dp_group
    peer_tp_per_dp = peer_ri.tp_size_per_dp_group
    tp_ranks = paired_ranks(self_tp_per_dp, self_ri.tp_rank % self_tp_per_dp, peer_tp_per_dp)
    # TP ranks of the peer's DP group ``peer_dp_rank``.
    tp_ranks = range(
        tp_ranks.start + peer_dp_rank * peer_tp_per_dp,
        tp_ranks.stop + peer_dp_rank * peer_tp_per_dp,
    )
    cp_ranks = paired_ranks(self_ri.cp_size, self_ri.cp_rank, peer_ri.cp_size)

    ranks: List[int] = []
    for pp in range(peer_start_pp, peer_start_pp + overlap_pp_size):
        for tp in tp_ranks:
            for cp in cp_ranks:
                # CP-minor flat rank (ppRank*(TP*CP) + tpRank*CP + cpRank), matching Mapping and
                # the C++ transceiver; loop nesting mirrors the layout so ranks come out sorted.
                ranks.append(pp * peer_ri.tp_size * peer_ri.cp_size + tp * peer_ri.cp_size + cp)

    dup_head, peer_dup_head = head_duplication(self_ri, peer_ri)
    return PeerOverlap(
        overlap_pp_size=overlap_pp_size,
        overlap_tp_size=len(tp_ranks),
        overlap_cp_size=len(cp_ranks),
        duplicate_head_factor=dup_head,
        peer_duplicate_head_factor=peer_dup_head,
        ranks=ranks,
    )
