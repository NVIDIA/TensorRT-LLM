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
"""Disaggregated KV transfer plan for DeepSeek targets that keep MLA latent KV in one pool.

Under attention-DP every context rank has a generation twin with the same
dp_rank. MLA keeps one latent-K region per layer in one physical pool, so a
block's bytes for all local layers are one contiguous slot. The plan is then
plain arithmetic: pair block ordinals both sides hold, address the slots, and
copy. Blocks that are consecutive on both sides move as one fragment.

Which deployments may talk to each other is the ``COVERS`` matrix below: pairs
of (context target, generation target) ids. A target never names this planner;
``modeling_v2.transfers`` finds the planner from the target id, and ``describe``
publishes that id so the peer's planner can look the pair up too. Anything not
in the matrix is refused at registration; the generic default planner stays
untouched for the models that need pool matching, head re-sharding or
recurrent state.
"""

from __future__ import annotations

from typing import FrozenSet, Optional, Tuple

import numpy as np

from tensorrt_llm._torch.disaggregation.base.backend import Chunk, RecvReqInfo
from tensorrt_llm._torch.disaggregation.base.planner import (
    PeerIncompatibleError,
    PeerTargets,
    TransferPlan,
    TransferPlanner,
)
from tensorrt_llm._torch.disaggregation.native.default_planner import DefaultTransferPlanner
from tensorrt_llm._torch.disaggregation.native.rank_info import RankInfo
from tensorrt_llm._torch.disaggregation.resource.kv_extractor import KVRegionExtractorV1
from tensorrt_llm._torch.disaggregation.resource.page import (
    CacheKind,
    KVCachePageTable,
    PhysicalPool,
)
from tensorrt_llm._torch.disaggregation.resource.utils import get_physical_pool

_R1_DEP4 = "deepseek_v3/r1_0528_nvfp4__sm_103__dep4"


def _sole_pool(page_table: Optional[KVCachePageTable], side: str) -> PhysicalPool:
    """The one PAGED pool this target keeps its latent KV in, or a refusal."""
    if page_table is None:
        raise PeerIncompatibleError(f"{side}: no page table published")
    if len(page_table.layer_groups) != 1 or page_table.layer_groups[0].kind != CacheKind.PAGED:
        raise PeerIncompatibleError(
            f"{side}: expected exactly one PAGED layer group, "
            f"got {[lg.kind.name for lg in page_table.layer_groups]}"
        )
    lg = page_table.layer_groups[0]
    pools = page_table.pool_groups[lg.pool_group_idx].pools
    if len(lg.pool_views) != 1 or len(pools) != 1:
        raise PeerIncompatibleError(
            f"{side}: expected one pool for the latent KV, "
            f"got {len(lg.pool_views)} views over {len(pools)} pools"
        )
    return get_physical_pool(page_table, 0, lg.pool_views[0].pool_idx)


class DeepseekMlaLatentTransferPlanner(TransferPlanner):
    planner_id = "modeling_v2.deepseek_mla_latent"

    #: (context target, generation target) pairs this planner knows how to move
    #: bytes between. The same pair is listed once; the planner serves both
    #: roles, so a pair is read from the sender's side or the receiver's side.
    COVERS: FrozenSet[Tuple[str, str]] = frozenset({(_R1_DEP4, _R1_DEP4)})

    def __init__(
        self, self_rank_info: RankInfo, self_extractor: KVRegionExtractorV1, *, self_target: str
    ):
        self._ri = self_rank_info
        self._target = self_target
        self._page_table = self_extractor.page_table
        self._pool = _sole_pool(self._page_table, "this rank")

    def describe(self) -> bytes:
        return self._target.encode()

    def _pair(self, peer_ri: RankInfo) -> Tuple[str, str]:
        """The (self, peer) target pair, or a refusal if the matrix lacks it."""
        peer_target = peer_ri.planner_blob.decode()
        if (self._target, peer_target) in self.COVERS or (peer_target, self._target) in self.COVERS:
            return self._target, peer_target
        raise PeerIncompatibleError(
            f"no transfer plan for the pair ({self._target!r}, {peer_target!r}); "
            f"covered pairs: {sorted(self.COVERS)}"
        )

    # ---- TransferPlanner protocol -------------------------------------------------

    def check_peer(self, self_ri: RankInfo, peer_ri: RankInfo) -> None:
        self._pair(peer_ri)
        peer_pool = _sole_pool(peer_ri.page_table, "peer")
        self_layers = len(self._page_table.layer_groups[0].local_layers)
        peer_layers = len(peer_ri.page_table.layer_groups[0].local_layers)
        if self_layers != peer_layers:
            raise PeerIncompatibleError(
                f"layer count differs: this rank {self_layers}, peer {peer_layers}"
            )
        if peer_ri.page_table.tokens_per_block != self._page_table.tokens_per_block:
            raise PeerIncompatibleError(
                f"tokens_per_block differs: this rank {self._page_table.tokens_per_block}, "
                f"peer {peer_ri.page_table.tokens_per_block}"
            )
        if peer_pool.slot_bytes != self._pool.slot_bytes:
            raise PeerIncompatibleError(
                f"latent KV slot differs: this rank {self._pool.slot_bytes} bytes, "
                f"peer {peer_pool.slot_bytes} bytes"
            )

    def targets(self, self_ri: RankInfo, peer_ri: RankInfo, peer_dp_rank: int) -> PeerTargets:
        # Every covered pair is attention-DP with pp = cp = 1 and tp == dp, so
        # the peer's flat rank is its dp_rank, and one writer per destination
        # means the fan-in split is trivially uniform.
        return PeerTargets(ranks=[peer_dp_rank], uniform_writers=True)

    def plan(
        self,
        self_ri: RankInfo,
        peer_ri: RankInfo,
        chunk: Chunk,
        req_info: RecvReqInfo,
        prompt_len: int,
    ) -> TransferPlan:
        del prompt_len  # cp == 1: a partial piece needs no projection
        if (0, 0) in chunk.excluded_pool_views:  # the sole latent pool is layer group 0, view 0
            return TransferPlan.empty()
        tpb = self._page_table.tokens_per_block
        dst_start_token = req_info.dst_start_token
        dst_start_block = dst_start_token // tpb if dst_start_token is not None else 0
        src_ids, dst_ids = DefaultTransferPlanner._pair_ordinals(
            np.asarray(chunk.block_ids_per_layer_groups[0], dtype=np.int64),
            np.asarray(req_info.block_ids_per_layer_groups[0], dtype=np.int64),
            dst_start_block=dst_start_block,
        )
        if src_ids.size == 0:
            return TransferPlan.empty()

        src_pool, dst_pool = self._pool, _sole_pool(peer_ri.page_table, "peer")
        slot = src_pool.slot_bytes
        # Consecutive slots are consecutive bytes only when the stride is the slot.
        mergeable = src_pool.slot_stride_bytes == slot and dst_pool.slot_stride_bytes == slot
        if mergeable and src_ids.size > 1:
            breaks = np.flatnonzero((np.diff(src_ids) != 1) | (np.diff(dst_ids) != 1)) + 1
            starts = np.concatenate(([0], breaks))
            lengths = np.diff(np.concatenate((starts, [src_ids.size])))
        else:
            starts = np.arange(src_ids.size)
            lengths = np.ones(src_ids.size, dtype=np.int64)
        return TransferPlan(
            src_ptrs=src_pool.base_address + src_ids[starts] * src_pool.slot_stride_bytes,
            dst_ptrs=dst_pool.base_address + dst_ids[starts] * dst_pool.slot_stride_bytes,
            sizes=lengths.astype(np.int64) * slot,
        )

    def receiver_payload_bytes(
        self, self_ri: RankInfo, peer_ri: RankInfo, req_info: RecvReqInfo
    ) -> Optional[int]:
        return 0  # no recurrent state rides along with the paged blocks
