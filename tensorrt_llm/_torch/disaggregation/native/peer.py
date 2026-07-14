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

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np

from tensorrt_llm import logger
from tensorrt_llm._torch.disaggregation.base.region import RegionMapperBase
from tensorrt_llm._torch.disaggregation.native.mixers.attention.peer import AttentionPolicy
from tensorrt_llm._torch.disaggregation.native.rank_info import RankInfo
from tensorrt_llm._torch.disaggregation.resource.kv_extractor import KVRegionExtractorV1
from tensorrt_llm._torch.disaggregation.resource.page import (
    AttentionLayerGroup,
    MapperKind,
    PoolView,
)
from tensorrt_llm._torch.disaggregation.resource.utils import (
    get_layer_byte_ranges,
    get_layer_to_layer_group,
    get_num_buffer_entries,
    get_pool_view_global_layer_ids,
    get_pool_view_num_layers,
)

# Type alias for (lg_idx, pool_idx) pair
LGPoolKey = Tuple[int, int]


@dataclass
class PeerOverlap:
    overlap_pp_size: int = 0
    overlap_tp_size: int = 0
    overlap_cp_size: int = 0
    duplicate_head_factor: int = 1
    peer_duplicate_head_factor: int = 1
    ranks: List[int] = field(default_factory=list)


class PeerRegistrar:
    def __init__(self, self_rank_info: RankInfo, self_extractor: KVRegionExtractorV1):
        self._ri = self_rank_info
        self._attention_policy = AttentionPolicy(self_rank_info)
        self._peer_ri_cache: Dict[str, RankInfo] = {}
        self._kv_map_cache: Dict[
            tuple, RegionMapperBase
        ] = {}  # key: (peer_key, self_lg_pool_key, peer_lg_pool_key)
        self._self_ext_cache = self_extractor
        self._peer_ext_cache: Dict[str, KVRegionExtractorV1] = {}
        self._overlap_cache: Dict[str, PeerOverlap] = {}
        self._lg_pool_mapping_cache: Dict[
            str, Dict[LGPoolKey, LGPoolKey]
        ] = {}  # peer_key -> {(self_lg, self_pi) -> (peer_lg, peer_pi)}

    def register(self, peer_name: str, peer_rank: int, peer_ri: RankInfo):
        assert self._self_ext_cache is not None
        if not self._check_peer_compatible(peer_ri):
            raise ValueError(
                f"PeerRegistrar.register: peer {peer_name} (rank={peer_rank}) is incompatible with local rank."
            )
        key = self._unique_key(peer_name, peer_rank)
        self._peer_ri_cache[key] = peer_ri
        peer_ri = self.get_peer_rank_info(peer_name, peer_rank)
        extractor = KVRegionExtractorV1(peer_ri.page_table)
        self._peer_ext_cache[key] = extractor

        head_match, _ = self._attention_policy.head_match(peer_ri)
        if not head_match:
            self_page_table = self._self_ext_cache.page_table
            nhd_fragments_per_token = sum(
                len(
                    self_page_table.layer_groups[layer_group_id].pool_views[pool_idx].buffer_entries
                )
                for layer_group_id, pool_idx in self.get_pool_mapping(peer_ri)
                if self_page_table.layer_groups[layer_group_id].pool_views[pool_idx].mapper_kind
                == MapperKind.NHD
            )
            if nhd_fragments_per_token:
                local_heads = self._ri.attention.kv_heads_per_rank
                peer_heads = peer_ri.attention.kv_heads_per_rank
                logger.warning_once(
                    "NHD head-mismatched disaggregated KV transfer has no "
                    "contiguous staging path and will emit approximately "
                    f"{nhd_fragments_per_token} NIXL descriptors per transferred "
                    "token per peer, excluding block-level replicated pools "
                    f"(local_kv_heads={local_heads}, peer_kv_heads={peer_heads}). "
                    "Long-context TEP/DEP transfers may have high latency.",
                    key=(
                        "native-nhd-head-mismatch-"
                        f"{local_heads}-{peer_heads}-{nhd_fragments_per_token}"
                    ),
                )

    def peer_extractor(self, peer_name: str, peer_rank: int) -> KVRegionExtractorV1:
        return self._peer_ext_cache[self._unique_key(peer_name, peer_rank)]

    @property
    def self_extractor(self) -> KVRegionExtractorV1:
        assert self._self_ext_cache is not None
        return self._self_ext_cache

    def unregister(self, peer_name: str, peer_rank: int):
        key = self._unique_key(peer_name, peer_rank)
        if key in self._peer_ri_cache:
            del self._peer_ri_cache[key]
        if key in self._peer_ext_cache:
            del self._peer_ext_cache[key]
        # Clean up kv_map_cache entries for this peer
        keys_to_remove = [k for k in self._kv_map_cache if k[0] == key]
        for k in keys_to_remove:
            del self._kv_map_cache[k]
        if key in self._lg_pool_mapping_cache:
            del self._lg_pool_mapping_cache[key]

    def get_peer_rank_info(self, peer_name: str, peer_rank: int):
        return self._peer_ri_cache[self._unique_key(peer_name, peer_rank)]

    @property
    def self_rank_info(self) -> RankInfo:
        return self._ri

    def _unique_key(self, name: str, rank: int) -> str:
        return name + str(rank)

    def _check_peer_compatible(self, peer_ri: RankInfo) -> bool:
        if not self._attention_policy.check_peer_compatible(peer_ri):
            return False

        self_layers = sum(self._ri.layer_num_per_pp)
        peer_layers = sum(peer_ri.layer_num_per_pp)
        if self_layers != peer_layers:
            # Allow mismatch when one side has speculative (e.g. MTP) layers
            # that the other side doesn't. The pool_mapping logic will only
            # transfer layers that exist on both sides.
            logger.warning(
                "PeerRegistrar: layer count differs "
                f"(local={self_layers}, peer={peer_layers}), "
                "allowing partial layer transfer."
            )

        return True

    def get_pool_mapping(self, peer_ri: RankInfo) -> Dict[LGPoolKey, LGPoolKey]:
        """Get mapping from (self_lg_idx, self_pool_idx) -> (peer_lg_idx, peer_pool_idx).

        Two-step matching:
        1. Find peer layer_group via layer_to_layer_group (global_layer_id -> lg_idx).
        2. Within the matched peer layer_group, find the unique peer pool whose
           ``PoolView.pool_role`` equals self's and whose global_layer_ids
           overlap.

        Layer-overlap is required: a peer pool with the same pool_role but
        zero layer overlap with self is *not* a match — the two pools cover
        disjoint layers and have nothing to transfer.
        """
        key = self._unique_key(peer_ri.instance_name, peer_ri.instance_rank)
        if key in self._lg_pool_mapping_cache:
            return self._lg_pool_mapping_cache[key]

        mapping: Dict[LGPoolKey, LGPoolKey] = {}
        self_pt = self._self_ext_cache.page_table
        peer_pt = peer_ri.page_table

        if self_pt is None or peer_pt is None:
            self._lg_pool_mapping_cache[key] = mapping
            return mapping
        if not self_pt.layer_groups or not peer_pt.layer_groups:
            self._lg_pool_mapping_cache[key] = mapping
            return mapping

        peer_layer_to_group = get_layer_to_layer_group(peer_pt)

        for self_lg_idx, self_lg in enumerate(self_pt.layer_groups):
            if not isinstance(self_lg, AttentionLayerGroup):
                continue
            for self_pi, self_pv in enumerate(self_lg.pool_views):
                # Every view carries buffer_entries, so a view's exact layer
                # set always comes from its entries (a view may cover a
                # subset of the LG when V2 splits an LG into multiple pools
                # by buffer-size class, or when a role class exists only on
                # some layers, e.g. sparse-layer index-K).
                pv_global_ids = get_pool_view_global_layer_ids(self_pv, self_lg)
                if not pv_global_ids:
                    continue

                # Step 1: find peer layer_group via any overlapping global_layer_id.
                peer_lg_idx = next(
                    (peer_layer_to_group[g] for g in pv_global_ids if g in peer_layer_to_group),
                    None,
                )
                if peer_lg_idx is None:
                    continue
                peer_lg = peer_pt.layer_groups[peer_lg_idx]

                # Step 2: pick the first peer pool with the same pool_role
                # whose layers overlap self's (zero-overlap pools cover
                # disjoint layers — nothing to transfer).
                #
                # Uniqueness assumption: at most one peer pool can match on
                # both ``pool_role`` (frozenset equality) and layer overlap.
                # We do *not* assume ``pool_role`` is unique within a peer LG
                # — V2 may split an LG into multiple same-role pools by
                # buffer-size class (e.g. VSWA). What we rely on is that both
                # peers run the same pool-grouping logic, so for every self_pv
                # there is exactly one peer pool with the same role *and* an
                # overlapping layer set; other same-role peer pools cover
                # disjoint layers and fall out via the overlap filter.
                self_layer_set = set(pv_global_ids)
                matched_peer_pi = None
                for peer_pi, peer_pv in enumerate(peer_lg.pool_views):
                    if peer_pv.pool_role != self_pv.pool_role:
                        continue
                    peer_global_ids = get_pool_view_global_layer_ids(peer_pv, peer_lg)
                    if not set(peer_global_ids) & self_layer_set:
                        continue
                    if peer_pv.mapper_kind != self_pv.mapper_kind:
                        raise ValueError(
                            "PeerRegistrar.get_pool_mapping: incompatible mapper "
                            f"kinds for pool role {sorted(self_pv.pool_role)} "
                            f"(local={self_pv.mapper_kind.name}, "
                            f"peer={peer_pv.mapper_kind.name}, peer_pool={peer_pi})"
                        )
                    matched_peer_pi = peer_pi
                    break

                if matched_peer_pi is not None:
                    mapping[(self_lg_idx, self_pi)] = (peer_lg_idx, matched_peer_pi)

        self._lg_pool_mapping_cache[key] = mapping
        return mapping

    def get_kv_map(
        self,
        peer_ri: RankInfo,
        self_pool_key: LGPoolKey,
        peer_pool_key: LGPoolKey,
    ) -> RegionMapperBase:
        """Get mapper for a specific pool pair.

        Args:
            peer_ri: Peer rank info.
            self_pool_key: (self_lg_idx, self_pool_idx).
            peer_pool_key: (peer_lg_idx, peer_pool_idx).
        """
        peer_key = self._unique_key(peer_ri.instance_name, peer_ri.instance_rank)
        cache_key = (peer_key, self_pool_key, peer_pool_key)
        if cache_key in self._kv_map_cache:
            return self._kv_map_cache[cache_key]

        self_pt = self._self_ext_cache.page_table
        peer_pt = peer_ri.page_table
        assert self_pt is not None
        assert peer_pt is not None
        self_lg_idx, self_pi = self_pool_key
        peer_lg_idx, peer_pi = peer_pool_key
        self_lg = self_pt.layer_groups[self_lg_idx]
        peer_lg = peer_pt.layer_groups[peer_lg_idx]
        self_pv = self_lg.pool_views[self_pi]
        peer_pv = peer_lg.pool_views[peer_pi]

        assert self._ri.attention is not None
        if self_pv.mapper_kind != peer_pv.mapper_kind:
            raise ValueError(
                "PeerRegistrar.get_kv_map: incompatible mapper kinds "
                f"(local={self_pv.mapper_kind.name}, peer={peer_pv.mapper_kind.name})"
            )

        # Every view is entries-driven: resolve the overlap layers to
        # slot-relative byte offsets on each side from the views' buffer
        # entries. Layer selection is explicit, so mappers never assume a
        # uniform layer stride (other role classes may interleave), and no
        # convention about global-id/byte-offset ordering is needed.
        self_num_layers = get_pool_view_num_layers(self_pv)
        peer_num_layers = get_pool_view_num_layers(peer_pv)
        self_global_ids = get_pool_view_global_layer_ids(self_pv, self_lg)
        peer_global_ids = get_pool_view_global_layer_ids(peer_pv, peer_lg)
        overlapping_layers = sorted(set(self_global_ids) & set(peer_global_ids))

        self_starts, self_bytes_per_layer = get_layer_byte_ranges(self_pv)
        peer_starts, peer_bytes_per_layer = get_layer_byte_ranges(peer_pv)
        self_g2l = {ll.global_layer_id: ll.local_layer_id for ll in self_lg.local_layers}
        peer_g2l = {ll.global_layer_id: ll.local_layer_id for ll in peer_lg.local_layers}
        self_layer_offsets = np.array(
            [self_starts[self_g2l[gid]] for gid in overlapping_layers], dtype=np.int64
        )
        peer_layer_offsets = np.array(
            [peer_starts[peer_g2l[gid]] for gid in overlapping_layers], dtype=np.int64
        )
        # Per-layer buffer count (K and V are separate buffers within a
        # layer's region); head-mismatch mappers slice heads inside each.
        self_buffers_per_layer = self._get_buffers_per_layer(
            self_pv,
            self_num_layers,
            layer_group_id=self_lg_idx,
            pool_idx=self_pi,
        )
        peer_buffers_per_layer = self._get_buffers_per_layer(
            peer_pv,
            peer_num_layers,
            layer_group_id=peer_lg_idx,
            pool_idx=peer_pi,
        )

        mapper = self._attention_policy.build_kv_mapper(
            peer_ri=peer_ri,
            mapper_kind=self_pv.mapper_kind,
            self_layer_offsets=self_layer_offsets,
            peer_layer_offsets=peer_layer_offsets,
            self_bytes_per_layer=self_bytes_per_layer,
            peer_bytes_per_layer=peer_bytes_per_layer,
            self_buffers_per_layer=self_buffers_per_layer,
            peer_buffers_per_layer=peer_buffers_per_layer,
        )

        self._kv_map_cache[cache_key] = mapper
        return mapper

    @staticmethod
    def _get_buffers_per_layer(
        pool_view: PoolView,
        num_layers: int,
        *,
        layer_group_id: int,
        pool_idx: int,
    ) -> int:
        num_entries = get_num_buffer_entries(pool_view)
        if num_entries == 0:
            return 1
        if num_layers <= 0 or num_entries % num_layers != 0:
            raise ValueError(
                "PoolView buffer entries are not evenly distributed across layers: "
                f"layer_group={layer_group_id}, pool={pool_idx}, "
                f"entries={num_entries}, layers={num_layers}"
            )
        return num_entries // num_layers

    @staticmethod
    def _find_overlap(self_val, peer_val, self_rank, peer_rank=None):
        if self_val <= peer_val:
            overlap = peer_val // self_val
            start = self_rank * overlap + (peer_rank * peer_val if peer_rank is not None else 0)
            end = start + overlap
        else:
            ratio = self_val // peer_val
            start = (self_rank // ratio) + (peer_rank * peer_val if peer_rank is not None else 0)
            overlap = 1
            end = start + overlap

        return overlap, start, end

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

        self_tp_per_dp = self._ri.tp_size_per_dp_group
        peer_tp_per_dp = peer_ri.tp_size_per_dp_group
        self_tp_rank_in_dp = self._ri.tp_rank % self_tp_per_dp

        overlap_tp_size, peer_start_tp, peer_end_tp = self._find_overlap(
            self_tp_per_dp, peer_tp_per_dp, self_tp_rank_in_dp, peer_dp_rank
        )
        overlap_cp_size, peer_start_cp, peer_end_cp = self._find_overlap(
            self._ri.cp_size, peer_ri.cp_size, self._ri.cp_rank
        )

        ranks: List[int] = []
        for pp in range(peer_start_pp, peer_end_pp):
            for cp in range(peer_start_cp, peer_end_cp):
                for tp in range(peer_start_tp, peer_end_tp):
                    ranks.append(pp * peer_ri.tp_size * peer_ri.cp_size + cp * peer_ri.tp_size + tp)

        dup_head, peer_dup_head = self._attention_policy.duplicate_head_factors(peer_ri)

        targets = PeerOverlap(
            overlap_pp_size=overlap_pp_size,
            overlap_tp_size=overlap_tp_size,
            overlap_cp_size=overlap_cp_size,
            duplicate_head_factor=dup_head,
            peer_duplicate_head_factor=peer_dup_head,
            ranks=ranks,
        )
        self._overlap_cache[key] = targets
        return targets

    def should_send_kv(self, peer_overlap: PeerOverlap, peer_rank_info: RankInfo) -> bool:
        dup_head_factor = peer_overlap.duplicate_head_factor
        if dup_head_factor <= 1:
            return True
        self_tp_rank_in_dp_group = self._ri.tp_rank % self._ri.tp_size_per_dp_group
        return (peer_rank_info.dp_rank % dup_head_factor) == (
            self_tp_rank_in_dp_group % dup_head_factor
        )

    def _owns_tp_fan_in(self, peer_rank_info: RankInfo) -> bool:
        """Elect one owner when replicated bytes fan in across TP ranks.

        A peer with fewer TP shards receives identical replicated data from
        several local ranks. Rotate the elected owner by the destination's
        DP rank (mirroring ``should_send_kv``'s head-duplication pairing) so
        that with a multi-DP-group generation side the extra replicated
        traffic spreads across local ranks instead of always landing on the
        first rank of each fan-in group.
        """
        ratio = max(
            1,
            self._ri.tp_size_per_dp_group // peer_rank_info.tp_size_per_dp_group,
        )
        self_tp_rank = self._ri.tp_rank % self._ri.tp_size_per_dp_group
        return self_tp_rank % ratio == peer_rank_info.dp_rank % ratio

    def should_send_pool(
        self,
        peer_overlap: PeerOverlap,
        peer_rank_info: RankInfo,
        layer_group_id: int,
        pool_idx: int,
    ) -> bool:
        """Return whether this rank owns the transfer of one view pair.

        ``pool_idx`` indexes the layer group's ``pool_views`` list (one view
        per role class; several views may share a physical pool). Each view
        is kind-homogeneous, so ownership is a single per-view decision:
        replicated views use one sender per fan-in group, sharded views
        retain head-duplication routing.
        """
        layer_group = self._self_ext_cache.page_table.layer_groups[layer_group_id]
        pool_view = layer_group.pool_views[pool_idx]
        if pool_view.mapper_kind == MapperKind.REPLICATED:
            return self._owns_tp_fan_in(peer_rank_info)
        return self.should_send_kv(peer_overlap, peer_rank_info)

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
