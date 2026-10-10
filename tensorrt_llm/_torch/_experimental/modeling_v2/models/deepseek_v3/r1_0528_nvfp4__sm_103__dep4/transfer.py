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
"""Disaggregated KV transfer plan for the dep4 DeepSeek-R1 target.

Under attention-DP dep4 every context rank has exactly one generation twin, the
rank with the same dp_rank. MLA keeps one latent-K region per layer in one
physical pool, so a block's bytes are one contiguous slot. The plan is then
plain arithmetic: pair block ordinals both sides hold, address the slots, and
copy. Blocks that are consecutive on both sides move as one fragment.

This planner accepts exactly one peer shape -- the same target on four ranks --
and refuses anything else; the generic default planner stays untouched for the
models that need pool matching, head re-sharding or recurrent state.
"""

from __future__ import annotations

from typing import AbstractSet, List, Optional, Tuple

import numpy as np

from tensorrt_llm._torch.disaggregation.base.backend import TokenRange
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

_DP_SIZE = 4


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


class DeepseekR1Dep4TransferPlanner(TransferPlanner):
    planner_id = "deepseek_r1_0528_nvfp4_sm103_dep4"

    def __init__(self, self_rank_info: RankInfo, self_extractor: KVRegionExtractorV1):
        self._ri = self_rank_info
        self._page_table = self_extractor.page_table
        self._pool = _sole_pool(self._page_table, "this rank")

    # ---- TransferPlanner protocol -------------------------------------------------

    def check_peer(self, self_ri: RankInfo, peer_ri: RankInfo) -> None:
        for ri, side in ((self_ri, "this rank"), (peer_ri, "peer")):
            if ri.dp_size != _DP_SIZE or ri.tp_size != _DP_SIZE:
                raise PeerIncompatibleError(
                    f"{side}: dp_size={ri.dp_size} tp_size={ri.tp_size}; this target runs dep{_DP_SIZE} only"
                )
            if ri.pp_size != 1 or ri.cp_size != 1:
                raise PeerIncompatibleError(
                    f"{side}: pp_size={ri.pp_size} cp_size={ri.cp_size}; this target runs pp=1 cp=1 only"
                )
            if ri.attention is None or not ri.attention.enable_attention_dp:
                raise PeerIncompatibleError(f"{side}: attention DP is off")
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
        # pp = cp = 1 and tp == dp, so the peer's flat rank is its dp_rank, and
        # one writer per destination means the fan-in split is trivially uniform.
        return PeerTargets(ranks=[peer_dp_rank], uniform_writers=True)

    def plan(
        self,
        self_ri: RankInfo,
        peer_ri: RankInfo,
        src_block_ids_per_lg: List[np.ndarray],
        dst_block_ids_per_lg: List[np.ndarray],
        token_range: TokenRange,
        prompt_len: int,
        dst_start_token: Optional[int],
        excluded_pool_views: AbstractSet[Tuple[int, int]],
    ) -> TransferPlan:
        del token_range, prompt_len  # cp == 1: a partial piece needs no projection
        if (0, 0) in excluded_pool_views:  # the sole latent pool is layer group 0, view 0
            return TransferPlan.empty()
        tpb = self._page_table.tokens_per_block
        dst_start_block = dst_start_token // tpb if dst_start_token is not None else 0
        src_ids, dst_ids = DefaultTransferPlanner._pair_ordinals(
            np.asarray(src_block_ids_per_lg[0], dtype=np.int64),
            np.asarray(dst_block_ids_per_lg[0], dtype=np.int64),
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
        self, self_ri: RankInfo, peer_ri: RankInfo, dst_block_ids_per_lg: List[np.ndarray]
    ) -> Optional[int]:
        return 0  # no recurrent state rides along with the paged blocks
