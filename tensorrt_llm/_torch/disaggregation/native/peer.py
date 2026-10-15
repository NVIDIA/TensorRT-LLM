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
"""Framework-side peer bookkeeping for the native transfer runtime.

Who the peers are, their page-table extractors, the aux transfer layouts and
aux ownership live here. How KV bytes pair between this rank and a peer is the
:class:`TransferPlanner`'s business; the registrar only gates registration on
it and hands it out to the sender and receiver.
"""

from typing import Dict, Optional

from tensorrt_llm._torch.disaggregation.base.planner import PeerIncompatibleError, TransferPlanner
from tensorrt_llm._torch.disaggregation.native.auxiliary import AuxTransferLayout
from tensorrt_llm._torch.disaggregation.native.rank_info import RankInfo
from tensorrt_llm._torch.disaggregation.resource.kv_extractor import KVRegionExtractorV1


class PeerRegistrar:
    def __init__(
        self,
        self_rank_info: RankInfo,
        self_extractor: KVRegionExtractorV1,
        planner: TransferPlanner,
    ):
        self._ri = self_rank_info
        self._self_ext_cache = self_extractor
        self._planner = planner
        self._peer_ri_cache: Dict[str, RankInfo] = {}
        self._peer_ext_cache: Dict[str, KVRegionExtractorV1] = {}
        self._aux_transfer_layout_cache: Dict[str, AuxTransferLayout] = {}

    @property
    def planner(self) -> TransferPlanner:
        return self._planner

    def register(self, peer_name: str, peer_rank: int, peer_ri: RankInfo):
        """Admit a peer. Raises :class:`PeerIncompatibleError` (a ``ValueError``) otherwise."""
        if peer_ri.planner_id != self._planner.planner_id:
            raise PeerIncompatibleError(
                f"PeerRegistrar.register: peer {peer_name} (rank={peer_rank}) plans transfers "
                f"with '{peer_ri.planner_id}', this rank with '{self._planner.planner_id}'; "
                "both sides of a disaggregated deployment must run the same model planner"
            )
        if peer_ri.describer_id != self._ri.describer_id:
            raise PeerIncompatibleError(
                f"PeerRegistrar.register: peer {peer_name} (rank={peer_rank}) describes its cache "
                f"with '{peer_ri.describer_id}', this rank with '{self._ri.describer_id}'; "
                "both sides of a disaggregated deployment must run the same cache describer"
            )
        self._planner.check_peer(self._ri, peer_ri)
        key = self._unique_key(peer_name, peer_rank)
        self._peer_ri_cache[key] = peer_ri
        self._aux_transfer_layout_cache.pop(key, None)
        self._peer_ext_cache[key] = KVRegionExtractorV1(peer_ri.page_table)

    def peer_extractor(self, peer_name: str, peer_rank: int) -> KVRegionExtractorV1:
        return self._peer_ext_cache[self._unique_key(peer_name, peer_rank)]

    @property
    def self_extractor(self) -> KVRegionExtractorV1:
        assert self._self_ext_cache is not None
        return self._self_ext_cache

    def unregister(self, peer_name: str, peer_rank: int):
        key = self._unique_key(peer_name, peer_rank)
        peer_ri = self._peer_ri_cache.pop(key, None)
        self._peer_ext_cache.pop(key, None)
        self._aux_transfer_layout_cache.pop(key, None)
        if peer_ri is not None:
            self._planner.forget_peer(peer_ri)

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
