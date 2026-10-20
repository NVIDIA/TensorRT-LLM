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

from typing import Dict, FrozenSet, List, Optional, Tuple

from tensorrt_llm import logger
from tensorrt_llm._torch.disaggregation.cache_specs import (
    PeerOverlap,
    RegionRef,
    TransferGroup,
    check_peer_compatible,
    notify_peer_registered,
    owns_transfer,
    plan_transfers,
    rank_overlap,
    validate_peer_compatible,
)
from tensorrt_llm._torch.disaggregation.native.auxiliary import AuxTransferLayout
from tensorrt_llm._torch.disaggregation.native.rank_info import RankInfo
from tensorrt_llm._torch.disaggregation.resource.kv_extractor import KVRegionExtractorV1

__all__ = ["PeerOverlap", "PeerRegistrar"]


class PeerRegistrar:
    """Per-peer bookkeeping for the native transfer.

    Holds what each registered peer advertised (rank info, page table, aux layout) and caches the
    answers ``cache_specs`` computes for it: the transfer groups (matched regions with their byte
    plans and sender election) and which peer ranks overlap this one. No cache-spec logic lives
    here.
    """

    def __init__(self, self_rank_info: RankInfo, self_extractor: KVRegionExtractorV1):
        self._ri = self_rank_info
        self._peer_ri_cache: Dict[str, RankInfo] = {}
        # key: (peer_key, excluded regions)
        self._groups_cache: Dict[Tuple[str, FrozenSet[RegionRef]], List[TransferGroup]] = {}
        self._self_ext_cache = self_extractor
        self._peer_ext_cache: Dict[str, KVRegionExtractorV1] = {}
        self._overlap_cache: Dict[str, PeerOverlap] = {}
        self._aux_transfer_layout_cache: Dict[str, AuxTransferLayout] = {}

    def register(self, peer_name: str, peer_rank: int, peer_ri: RankInfo):
        assert self._self_ext_cache is not None
        if not self._check_peer_compatible(peer_ri):
            raise ValueError(
                f"PeerRegistrar.register: peer {peer_name} (rank={peer_rank}) is incompatible with local rank."
            )
        key = self._unique_key(peer_name, peer_rank)
        self._peer_ri_cache[key] = peer_ri
        self._aux_transfer_layout_cache.pop(key, None)
        peer_ri = self.get_peer_rank_info(peer_name, peer_rank)
        self._peer_ext_cache[key] = KVRegionExtractorV1(peer_ri.page_table)
        notify_peer_registered(
            self._ri, peer_ri, self._self_ext_cache.page_table, self.transfer_groups(peer_ri)
        )

    def peer_extractor(self, peer_name: str, peer_rank: int) -> KVRegionExtractorV1:
        return self._peer_ext_cache[self._unique_key(peer_name, peer_rank)]

    @property
    def self_extractor(self) -> KVRegionExtractorV1:
        assert self._self_ext_cache is not None
        return self._self_ext_cache

    def unregister(self, peer_name: str, peer_rank: int):
        key = self._unique_key(peer_name, peer_rank)
        self._peer_ri_cache.pop(key, None)
        self._peer_ext_cache.pop(key, None)
        self._aux_transfer_layout_cache.pop(key, None)
        for k in [k for k in self._groups_cache if k[0] == key]:
            del self._groups_cache[k]

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

    def _check_peer_compatible(self, peer_ri: RankInfo) -> bool:
        self.validate_bounded_replay_compatible(peer_ri)
        if not check_peer_compatible(self._ri, peer_ri):
            return False
        # Layout gates raise ValueError with a field-level diagnostic instead of returning False,
        # so the precise mismatch reaches the caller of register().
        self_page_table = (
            self._self_ext_cache.page_table if self._self_ext_cache is not None else None
        )
        validate_peer_compatible(self._ri, peer_ri, self_page_table, peer_ri.page_table)

        self_layers = sum(self._ri.layer_num_per_pp)
        peer_layers = sum(peer_ri.layer_num_per_pp)
        if self_layers != peer_layers:
            # One side may hold speculative (e.g. MTP) layers the other lacks; only layers both
            # sides hold are transferred.
            logger.warning(
                "PeerRegistrar: layer count differs "
                f"(local={self_layers}, peer={peer_layers}), "
                "allowing partial layer transfer."
            )
        return True

    def transfer_groups(
        self, peer_ri: RankInfo, excluded: FrozenSet[RegionRef] = frozenset()
    ) -> List[TransferGroup]:
        """Every transfer group to one peer rank (see ``cache_specs.plan_transfers``), cached."""
        key = (self._unique_key(peer_ri.instance_name, peer_ri.instance_rank), frozenset(excluded))
        groups = self._groups_cache.get(key)
        if groups is None:
            assert self._self_ext_cache.page_table is not None
            assert peer_ri.page_table is not None
            groups = plan_transfers(
                self._ri,
                peer_ri,
                self._self_ext_cache.page_table,
                peer_ri.page_table,
                excluded=key[1],
            )
            self._groups_cache[key] = groups
        return groups

    def get_peer_overlap(self, peer_rank_info: RankInfo, peer_dp_rank: int) -> PeerOverlap:
        key = self._unique_key(peer_rank_info.instance_name, peer_dp_rank)
        overlap = self._overlap_cache.get(key)
        if overlap is None:
            overlap = rank_overlap(self._ri, peer_rank_info, peer_dp_rank)
            self._overlap_cache[key] = overlap
        return overlap

    def owns(self, group: TransferGroup, peer_rank_info: RankInfo) -> bool:
        """Whether this rank is the elected sender of ``group`` to ``peer_rank_info``.

        Local ranks holding identical bytes (replicated roles, duplicated KV heads, fan-in from a
        larger TP) elect one sender each; ranks holding a unique shard always send.
        """
        return owns_transfer(group, self._ri, peer_rank_info)

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
