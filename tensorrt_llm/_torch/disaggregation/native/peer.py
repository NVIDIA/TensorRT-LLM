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

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from tensorrt_llm import logger
from tensorrt_llm._torch.disaggregation.native.auxiliary import AuxTransferLayout
from tensorrt_llm._torch.disaggregation.native.rank_info import RankInfo
from tensorrt_llm._torch.disaggregation.native.runs import RunsMapper
from tensorrt_llm._torch.pyexecutor.kv_cache.cache_layout.layout_mapping import (
    BufferRuns,
    check_compatible,
    fan,
    layer_group_of,
    map_buffers,
    merge_runs,
    sent_bytes,
)
from tensorrt_llm._torch.pyexecutor.kv_cache.cache_layout.page_layout import PageLayout
from tensorrt_llm._torch.pyexecutor.kv_cache.cache_layout.page_table import ManagerLayout

# (layer group index, pool index within the layer group's pool group)
LGPoolKey = Tuple[int, int]
# A pool of this rank and the peer rank's pool its bytes land in.
PoolPair = Tuple[LGPoolKey, LGPoolKey]


@dataclass
class PeerOverlap:
    overlap_pp_size: int = 0
    overlap_tp_size: int = 0
    overlap_cp_size: int = 0
    ranks: List[int] = field(default_factory=list)


class PeerRegistrar:
    def __init__(self, self_rank_info: RankInfo):
        self._ri = self_rank_info
        self._peer_ri_cache: Dict[str, RankInfo] = {}
        # (peer_key, excluded buffers) -> what this rank sends that peer rank, by pool pair.
        self._kv_map_cache: Dict[tuple, Dict[PoolPair, RunsMapper]] = {}
        # peer_key -> what this rank sends that peer rank, buffer by buffer, by pool pair.
        self._buffer_runs_cache: Dict[str, Dict[PoolPair, List[BufferRuns]]] = {}
        self._overlap_cache: Dict[str, PeerOverlap] = {}
        # peer_key of a rank writing this rank's pages -> what it sends per page of each of this
        # rank's layer groups.
        self._writer_shares: Dict[str, Dict[int, int]] = {}
        self._aux_transfer_layout_cache: Dict[str, AuxTransferLayout] = {}

    def register(self, peer_name: str, peer_rank: int, peer_ri: RankInfo):
        self.check_peer_compatible(peer_ri)
        key = self._unique_key(peer_name, peer_rank)
        self._peer_ri_cache[key] = peer_ri
        self._aux_transfer_layout_cache.pop(key, None)
        self._warn_fragmented_pages(peer_ri)

    def _warn_fragmented_pages(self, peer_ri: RankInfo) -> None:
        """Warn when the paged buffers this rank sends the peer move in more fragments than there
        are buffers: the two sides order a buffer's bytes differently, such as token-major KV over
        different head counts, and every page then costs that many transfer descriptors."""
        own = _layout_of(self._ri.page_table)
        maps = self.get_kv_maps(peer_ri)
        runs = self._buffer_runs_cache[
            self._unique_key(peer_ri.instance_name, peer_ri.instance_rank)
        ]
        buffers = fragments = 0
        for pair, mapper in maps.items():
            if own.layer_groups[pair[0][0]].kind == "attention":
                buffers += len(runs[pair])
                fragments += mapper.runs_per_slot
        if fragments > buffers:
            logger.warning_once(
                f"Disaggregated KV transfer to {peer_ri.instance_name} moves each page in "
                f"{fragments} fragments for {buffers} buffers, as the two sides order the buffers' "
                "bytes differently (such as token-major KV over different head counts); every page "
                "costs that many NIXL descriptors per peer, so long-context transfers may have high "
                "latency.",
                key=f"native-fragmented-pages-{fragments}-{buffers}",
            )

    def unregister(self, peer_name: str, peer_rank: int):
        key = self._unique_key(peer_name, peer_rank)
        if key in self._peer_ri_cache:
            del self._peer_ri_cache[key]
        self._aux_transfer_layout_cache.pop(key, None)
        # Clean up kv_map_cache entries for this peer
        keys_to_remove = [k for k in self._kv_map_cache if k[0] == key]
        for k in keys_to_remove:
            del self._kv_map_cache[k]
        self._buffer_runs_cache.pop(key, None)

    def get_peer_rank_info(self, peer_name: str, peer_rank: int):
        return self._peer_ri_cache[self._unique_key(peer_name, peer_rank)]

    def get_aux_transfer_layout(
        self, peer_name: str, peer_rank: int
    ) -> Optional[AuxTransferLayout]:
        return self._aux_transfer_layout_cache.get(self._unique_key(peer_name, peer_rank))

    def cache_aux_transfer_layout(
        self, peer_name: str, peer_rank: int, layout: AuxTransferLayout
    ) -> None:
        self._aux_transfer_layout_cache[self._unique_key(peer_name, peer_rank)] = layout

    @property
    def self_rank_info(self) -> RankInfo:
        return self._ri

    def _unique_key(self, name: str, rank: int) -> str:
        return name + str(rank)

    def validate_bounded_replay_compatible(self, peer_ri: RankInfo) -> None:
        """Require the same generation replay mode on both peers."""
        if self._ri.bounded_replay_on_generation != peer_ri.bounded_replay_on_generation:
            raise ValueError(
                "PeerRegistrar: bounded_replay_on_generation must match between peers "
                f"(local={self._ri.bounded_replay_on_generation}, "
                f"peer={peer_ri.bounded_replay_on_generation})."
            )

    def check_peer_compatible(self, peer_ri: RankInfo) -> None:
        """Check that this rank and the peer rank can exchange pages: the same generation replay
        mode, context parallelism on at most one side (Helix pairs a context instance without it
        with a generation instance using it), and page layouts that correspond
        (``layout_mapping.check_compatible``). Ranks holding different layers are allowed: only
        the layers both hold move.

        Raises:
            ValueError: Naming what differs.
        """
        self.validate_bounded_replay_compatible(peer_ri)
        if self._ri.cp_size > 1 and peer_ri.cp_size > 1:
            raise ValueError(
                "PeerRegistrar: context parallelism on both sides is unsupported "
                f"(local cp_size={self._ri.cp_size}, peer cp_size={peer_ri.cp_size})"
            )
        check_compatible(_layout_of(self._ri.page_table), _layout_of(peer_ri.page_table))

        self_layers = sum(self._ri.layer_num_per_pp)
        peer_layers = sum(peer_ri.layer_num_per_pp)
        if self_layers != peer_layers:
            # Allow mismatch when one side has speculative (e.g. MTP) layers
            # that the other side doesn't. Only layers that exist on both
            # sides transfer (layout_mapping.map_buffers).
            logger.warning(
                "PeerRegistrar: layer count differs "
                f"(local={self_layers}, peer={peer_layers}), "
                "allowing partial layer transfer."
            )

    def add_writer(self, writer: RankInfo) -> Dict[int, int]:
        """Record what a peer rank writing this rank's pages sends it: its share, the bytes it
        sends into one page of each of this rank's layer groups, a state group's slot counting as
        its page (``layout_mapping.sent_bytes``), from the writer's own page layout.

        Raises:
            ValueError: If the writer's pages do not correspond to this rank's, or the ranks it
                pairs with leave this rank's short (``map_buffers``).
        """
        own = _layout_of(self._ri.page_table)
        theirs = _layout_of(writer.page_table)
        check_compatible(theirs, own)
        share = sent_bytes(theirs, own, key=self._ri.dp_rank)
        self._writer_shares[self._unique_key(writer.instance_name, writer.instance_rank)] = share
        return share

    def writer_share(self, peer_ri: RankInfo, peer_rank: int) -> Dict[int, int]:
        """The share ``add_writer`` recorded for rank ``peer_rank`` of the peer's instance."""
        return self._writer_shares[self._unique_key(peer_ri.instance_name, peer_rank)]

    def get_kv_maps(
        self, peer_ri: RankInfo, excluded: frozenset = frozenset()
    ) -> Dict[PoolPair, RunsMapper]:
        """What this rank sends the peer rank: per pool of this rank and pool of the peer's its
        bytes land in, the runs of the buffers there that ``layout_mapping`` elects this rank to
        send, relative to each side's pool slot. Pairs it sends nothing of are absent.

        Args:
            peer_ri: Peer rank info.
            excluded: Buffers, as (global layer id, role), this transfer leaves out.
        """
        peer_key = self._unique_key(peer_ri.instance_name, peer_ri.instance_rank)
        cache_key = (peer_key, excluded)
        if cache_key in self._kv_map_cache:
            return self._kv_map_cache[cache_key]

        runs = self._buffer_runs_cache.get(peer_key)
        if runs is None:
            own = _layout_of(self._ri.page_table)
            peer = _layout_of(peer_ri.page_table)
            own_group, peer_group = layer_group_of(own), layer_group_of(peer)
            runs = defaultdict(list)
            # Ties between ranks holding the same bytes rotate by the destination's DP rank.
            for r in map_buffers(own, peer, key=peer_ri.dp_rank):
                pair = (
                    (own_group[r.src.pool_group, r.src.layer], r.src.pool_index),
                    (peer_group[r.dst.pool_group, r.dst.layer], r.dst.pool_index),
                )
                runs[pair].append(r)
            runs = dict(runs)
            self._buffer_runs_cache[peer_key] = runs

        maps: Dict[PoolPair, RunsMapper] = {}
        for pair, buffer_runs in runs.items():
            parts = [r for r in buffer_runs if (r.src.layer, r.src.role) not in excluded]
            if parts:
                maps[pair] = RunsMapper(
                    *merge_runs(
                        np.concatenate([r.src_offsets for r in parts]),
                        np.concatenate([r.dst_offsets for r in parts]),
                        np.concatenate([r.sizes for r in parts]),
                    )
                )
        self._kv_map_cache[cache_key] = maps
        return maps

    def get_peer_overlap(self, peer_rank_info: RankInfo, peer_dp_rank: int) -> PeerOverlap:
        peer_ri = peer_rank_info
        key = self._unique_key(peer_ri.instance_name, peer_dp_rank)
        if key in self._overlap_cache:
            return self._overlap_cache[key]

        # compute pp overlap and target layers
        self_start_layer = sum(self._ri.layer_num_per_pp[: self._ri.pp_rank])
        self_end_layer = self_start_layer + self._ri.layer_num_per_pp[self._ri.pp_rank]

        pre = 0
        tgt_pp_ranks: List[int] = []
        for p in range(peer_ri.pp_size):
            peer_start_layer = pre
            peer_end_layer = peer_start_layer + peer_ri.layer_num_per_pp[p]
            if self_start_layer < peer_end_layer and self_end_layer > peer_start_layer:
                tgt_pp_ranks.append(p)
            pre += peer_ri.layer_num_per_pp[p]

        if tgt_pp_ranks == []:
            targets = PeerOverlap()
            self._overlap_cache[key] = targets
            return targets

        peer_start_pp = tgt_pp_ranks[0]
        overlap_pp_size = len(tgt_pp_ranks)
        peer_end_pp = peer_start_pp + overlap_pp_size

        # The peer ranks whose share of each dimension overlaps this rank's (layout_mapping.fan),
        # the same pairing map_buffers elects senders within; tensor ranks count within the
        # peer's data-parallel group peer_dp_rank.
        self_tp_per_dp = self._ri.tp_size_per_dp_group
        peer_tp_per_dp = peer_ri.tp_size_per_dp_group
        tp_fan = fan(peer_tp_per_dp, self_tp_per_dp, self._ri.tp_rank % self_tp_per_dp)
        cp_fan = fan(peer_ri.cp_size, self._ri.cp_size, self._ri.cp_rank)
        tp_offset = peer_dp_rank * peer_tp_per_dp

        ranks: List[int] = []
        for pp in range(peer_start_pp, peer_end_pp):
            for tp in range(tp_offset + tp_fan.start, tp_offset + tp_fan.stop):
                for cp in cp_fan:
                    # CP-minor flat rank (ppRank*(TP*CP) + tpRank*CP +
                    # cpRank), matching Mapping and the C++ transceiver; a
                    # TP-minor formula mis-routes when tp > 1 and cp > 1.
                    # Loop nesting mirrors the layout so ranks come out in
                    # ascending order.
                    ranks.append(pp * peer_ri.tp_size * peer_ri.cp_size + tp * peer_ri.cp_size + cp)

        targets = PeerOverlap(
            overlap_pp_size=overlap_pp_size,
            overlap_tp_size=len(tp_fan),
            overlap_cp_size=len(cp_fan),
            ranks=ranks,
        )
        self._overlap_cache[key] = targets
        return targets

    def should_send_aux(self, peer_rank_info: RankInfo) -> bool:
        # to ensure the transfer aux is not duplicated

        # TP: only the first rank in each peer-TP-sized group sends aux
        ratio = max(1, self._ri.tp_size_per_dp_group // peer_rank_info.tp_size_per_dp_group)
        self_tp_rank_in_dp_group = self._ri.tp_rank % self._ri.tp_size_per_dp_group
        should_send_in_tp = self_tp_rank_in_dp_group % ratio == 0

        # PP: only the first self-PP rank whose layers overlap with the peer's PP rank sends aux.
        # All tp/pp ranks have the same aux data, so pick the first overlapping one to avoid duplication.
        peer_start_layer = sum(peer_rank_info.layer_num_per_pp[: peer_rank_info.pp_rank])
        peer_end_layer = peer_start_layer + peer_rank_info.layer_num_per_pp[peer_rank_info.pp_rank]
        offset = 0
        for p, n in enumerate(self._ri.layer_num_per_pp):
            if offset < peer_end_layer and offset + n > peer_start_layer:
                return should_send_in_tp and p == self._ri.pp_rank
            offset += n
        return False


def _layout_of(page_table: Optional[ManagerLayout]) -> PageLayout:
    """The page layout of a rank's page table."""
    if page_table is None:
        raise ValueError(
            "PeerRegistrar: a rank info carries no page table; build it from the manager "
            "(CacheLayoutAdapter.page_table)"
        )
    return page_table.page_layout
