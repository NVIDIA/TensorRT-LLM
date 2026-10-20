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
"""Rank topology: which ranks talk to each other and how a shard domain splits a buffer.

A cache kind never reads ``tp_size`` / ``cp_rank`` itself. It names the *shard domain* its buffers
are split over (``ATTENTION_TP`` or ``STATE_TP``), and this module turns a ``RankInfo`` into the
domain's :class:`ShardGrid`. Everything else -- which bytes a rank holds, which rank sends them --
follows from the grid in ``correspondence``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Dict, List, Tuple

if TYPE_CHECKING:
    from tensorrt_llm._torch.disaggregation.native.rank_info import RankInfo

ATTENTION_TP = "attn_tp"
"""Attention tensor parallelism inside one attention-DP group (KV heads, index-key caches)."""

STATE_TP = "state_tp"
"""The grid recurrent state is sharded over: TP, or TP x CP under helix, or 1 under attention DP."""


@dataclass(frozen=True)
class ShardGrid:
    """``size`` ranks split a domain; this rank is number ``rank`` of them."""

    size: int
    rank: int

    def __post_init__(self) -> None:
        if self.size < 1 or not 0 <= self.rank < self.size:
            raise ValueError(f"invalid shard grid: size={self.size}, rank={self.rank}")


def attention_grid(ri: "RankInfo") -> ShardGrid:
    size = ri.tp_size_per_dp_group
    return ShardGrid(size=size, rank=ri.tp_rank % size)


def state_grid(ri: "RankInfo") -> ShardGrid:
    """Recurrent state is replicated under attention DP and sharded over TP x CP under helix.

    Helix repurposes CP ranks as TP for the recurrent layers, so the grid is ``tp * cp`` with a
    CP-minor flat rank (matching ``Mapping``).
    """
    if ri.attention is not None and ri.attention.enable_attention_dp:
        return ShardGrid(size=1, rank=0)
    if ri.cp_size > 1:
        return ShardGrid(size=ri.tp_size * ri.cp_size, rank=ri.tp_rank * ri.cp_size + ri.cp_rank)
    return ShardGrid(size=ri.tp_size, rank=ri.tp_rank)


_GRIDS: Dict[str, Callable[["RankInfo"], ShardGrid]] = {
    ATTENTION_TP: attention_grid,
    STATE_TP: state_grid,
}


def shard_grid(domain: str, ri: "RankInfo") -> ShardGrid:
    try:
        return _GRIDS[domain](ri)
    except KeyError:
        raise ValueError(f"unknown shard domain {domain!r}; known: {sorted(_GRIDS)}") from None


def shard_span(units: int, grid: ShardGrid, virtual_units: int) -> Tuple[int, int]:
    """Half-open range of a ``virtual_units``-long axis held by ``grid.rank``.

    ``virtual_units`` is the smaller of the two sides' ``units * grid.size``. The side with fewer
    copies of each unit owns disjoint, consecutive ranges; the side that duplicates units (GQA heads
    on more ranks than heads, an MLA latent on every rank) maps several ranks onto the same range.
    """
    start = (grid.rank * virtual_units) // grid.size
    return start, start + units


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


def _find_overlap(self_val: int, peer_val: int, self_rank: int, peer_rank=None):
    if self_val <= peer_val:
        overlap = peer_val // self_val
        start = self_rank * overlap + (peer_rank * peer_val if peer_rank is not None else 0)
    else:
        ratio = self_val // peer_val
        start = (self_rank // ratio) + (peer_rank * peer_val if peer_rank is not None else 0)
        overlap = 1
    return overlap, start, start + overlap


def head_duplication(self_ri: "RankInfo", peer_ri: "RankInfo") -> Tuple[int, int]:
    """``(local, peer)`` attention-head duplication factors between two ranks.

    A rank with no attention cache (attention-free hybrid PP stage) reports zero heads; no head
    arithmetic applies to it.
    """
    a, b = self_ri.attention, peer_ri.attention
    if a is None or b is None or a.kv_heads_per_rank == 0 or b.kv_heads_per_rank == 0:
        return 1, 1
    self_heads = a.kv_heads_per_rank * self_ri.tp_size_per_dp_group
    peer_heads = b.kv_heads_per_rank * peer_ri.tp_size_per_dp_group
    return max(1, self_heads // peer_heads), max(1, peer_heads // self_heads)


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
    overlap_tp_size, peer_start_tp, peer_end_tp = _find_overlap(
        self_tp_per_dp, peer_tp_per_dp, self_ri.tp_rank % self_tp_per_dp, peer_dp_rank
    )
    overlap_cp_size, peer_start_cp, peer_end_cp = _find_overlap(
        self_ri.cp_size, peer_ri.cp_size, self_ri.cp_rank
    )

    ranks: List[int] = []
    for pp in range(peer_start_pp, peer_start_pp + overlap_pp_size):
        for tp in range(peer_start_tp, peer_end_tp):
            for cp in range(peer_start_cp, peer_end_cp):
                # CP-minor flat rank (ppRank*(TP*CP) + tpRank*CP + cpRank), matching Mapping and
                # the C++ transceiver; loop nesting mirrors the layout so ranks come out sorted.
                ranks.append(pp * peer_ri.tp_size * peer_ri.cp_size + tp * peer_ri.cp_size + cp)

    dup_head, peer_dup_head = head_duplication(self_ri, peer_ri)
    return PeerOverlap(
        overlap_pp_size=overlap_pp_size,
        overlap_tp_size=overlap_tp_size,
        overlap_cp_size=overlap_cp_size,
        duplicate_head_factor=dup_head,
        peer_duplicate_head_factor=peer_dup_head,
        ranks=ranks,
    )
