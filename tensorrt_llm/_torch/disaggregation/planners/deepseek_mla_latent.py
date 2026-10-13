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
"""Disaggregated KV transfer plan for DeepSeek deployments that keep MLA latent KV in one pool.

Under attention-DP every context rank has a generation twin with the same
dp_rank. MLA keeps one latent-K region per layer in one physical pool, so a
block's bytes for all local layers are one contiguous slot. The plan is then
plain arithmetic: pair the rows both sides hold by block ordinal and copy
slot for slot. Rows that are consecutive in memory on both sides move as one
fragment.

The user selects this planner by name (see ``disaggregation.planners``);
``supports_deployment`` then checks that the plan holds for the deployment: a
DeepSeek-V3 architecture under attention-DP with no pipeline or context
parallelism. The latent KV is never sharded across ranks, so the width of
either side does not matter and a dep4 context may feed a dep8 generation.
Which implementation computed the KV does not matter either: the built-in
``DeepseekV3ForCausalLM`` and a modeling_v2 target write the same bytes. The
framework only connects ranks that bound the same ``planner_id``, so a peer
has already passed the same check for itself; ``check_peer`` then compares the
latent geometry both sides published. The generic default planner stays
untouched for the models that need pool matching, head re-sharding or
recurrent state.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Sequence

import numpy as np

from tensorrt_llm._torch.disaggregation.base.backend import RecvReqInfo, TokenRange
from tensorrt_llm._torch.disaggregation.base.planner import (
    LayerGroupRows,
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

if TYPE_CHECKING:
    from tensorrt_llm.mapping import Mapping


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
    planner_id = "trtllm.deepseek_mla_latent"

    @classmethod
    def supports_deployment(cls, architecture: str, mapping: "Mapping") -> bool:
        """Whether this plan holds for ``architecture`` sharded as ``mapping`` says."""
        return (
            architecture == "DeepseekV3ForCausalLM"
            and mapping.pp_size == 1
            and mapping.cp_size == 1
            and bool(mapping.enable_attention_dp)
        )

    def __init__(self, self_rank_info: RankInfo, self_extractor: KVRegionExtractorV1):
        self._ri = self_rank_info
        self._page_table = self_extractor.page_table
        self._pool = _sole_pool(self._page_table, "this rank")

    # ---- TransferPlanner protocol -------------------------------------------------

    def check_peer(self, self_ri: RankInfo, peer_ri: RankInfo) -> None:
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
        # Every supported deployment is attention-DP with pp = cp = 1 and
        # tp == dp, so the peer's flat rank is its dp_rank, and one writer per
        # destination means the fan-in split is trivially uniform.
        return PeerTargets(ranks=[peer_dp_rank], uniform_writers=True)

    def pair(
        self,
        self_ri: RankInfo,
        peer_ri: RankInfo,
        self_rows: Sequence[LayerGroupRows],
        peer_rows: Sequence[LayerGroupRows],
        *,
        token_range: TokenRange,
        prompt_len: int,
        dst_start_token: Optional[int],
    ) -> TransferPlan:
        del token_range, prompt_len  # cp == 1: a partial piece needs no projection
        # The sole latent pool is layer group 0, view 0; the describer leaves it
        # out of a piece that excludes it.
        src = next((r for r in self_rows if (r.layer_group_index, r.view_index) == (0, 0)), None)
        if src is None:
            return TransferPlan.empty()
        dst = next(r for r in peer_rows if (r.layer_group_index, r.view_index) == (0, 0))
        tpb = self._page_table.tokens_per_block
        dst_start_block = dst_start_token // tpb if dst_start_token is not None else 0
        src_sel, dst_sel = DefaultTransferPlanner._pair_rows(
            src, dst, dst_start_block=dst_start_block
        )
        if src_sel.size == 0:
            return TransferPlan.empty()

        src_ptrs, dst_ptrs, slot = src.addresses[src_sel], dst.addresses[dst_sel], src.row_bytes
        # Consecutive rows are consecutive bytes only when they sit one slot apart on both sides.
        if src_sel.size > 1:
            breaks = np.flatnonzero((np.diff(src_ptrs) != slot) | (np.diff(dst_ptrs) != slot)) + 1
            starts = np.concatenate(([0], breaks))
            lengths = np.diff(np.concatenate((starts, [src_sel.size])))
        else:
            starts = np.zeros(1, dtype=np.int64)
            lengths = np.ones(1, dtype=np.int64)
        return TransferPlan(
            src_ptrs=src_ptrs[starts],
            dst_ptrs=dst_ptrs[starts],
            sizes=lengths.astype(np.int64) * slot,
        )

    def receiver_payload_bytes(
        self, self_ri: RankInfo, peer_ri: RankInfo, req_info: RecvReqInfo
    ) -> Optional[int]:
        return 0  # no recurrent state rides along with the paged blocks
