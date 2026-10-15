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
"""The DeepSeek MLA-latent planner must produce the default planner's bytes for the deployments it supports."""

import numpy as np
import pytest

from tensorrt_llm._torch.disaggregation.base import CacheKind, Chunk, RecvReqInfo, TokenRange
from tensorrt_llm._torch.disaggregation.base.planner import (
    PeerIncompatibleError,
    TransferPlan,
    plan_transfer,
)
from tensorrt_llm._torch.disaggregation.native.default_describer import DefaultCacheDescriber
from tensorrt_llm._torch.disaggregation.native.default_planner import DefaultTransferPlanner
from tensorrt_llm._torch.disaggregation.native.mixers.attention.spec import AttentionInfo
from tensorrt_llm._torch.disaggregation.native.rank_info import RankInfo
from tensorrt_llm._torch.disaggregation.planners.deepseek_mla_latent import (
    DeepseekMlaLatentTransferPlanner,
)
from tensorrt_llm._torch.disaggregation.resource.kv_extractor import KVRegionExtractorV1
from tensorrt_llm._torch.disaggregation.resource.page import (
    BUFFER_ENTRY_DTYPE,
    AttentionLayerGroup,
    KVCachePageTable,
    LocalLayer,
    PhysicalPool,
    PhysicalPoolGroup,
    PoolView,
)

pytestmark = pytest.mark.cpu_only

NUM_LAYERS = 6
LAYER_BYTES = 512  # one latent-K region per layer per block
SLOT_BYTES = NUM_LAYERS * LAYER_BYTES
TPB = 16


def _mla_page_table(base: int, *, num_layers: int = NUM_LAYERS, extra_pool: bool = False):
    """One PAGED group, one K-only pool: the shape RankInfo.from_kv_cache_manager builds for MLA."""
    entries = np.array(
        [(i, i * LAYER_BYTES, LAYER_BYTES) for i in range(num_layers)], dtype=BUFFER_ENTRY_DTYPE
    )
    views = [PoolView(pool_idx=0, buffer_entries=entries)]
    pools = [PhysicalPool(base_address=base, slot_bytes=num_layers * LAYER_BYTES, num_slots=256)]
    if extra_pool:
        views.append(PoolView(pool_idx=1, buffer_entries=entries, pool_role=frozenset({"mtp"})))
        pools.append(
            PhysicalPool(base_address=base + (1 << 30), slot_bytes=SLOT_BYTES, num_slots=256)
        )
    lg = AttentionLayerGroup(
        pool_group_idx=0,
        kv_head_num_per_rank=1,
        sliding_window_size=None,
        local_layers=[LocalLayer(local_layer_id=i, global_layer_id=i) for i in range(num_layers)],
        pool_views=views,
    )
    return KVCachePageTable(
        tokens_per_block=TPB, layer_groups=[lg], pool_groups=[PhysicalPoolGroup(pools=pools)]
    )


def _dep4_rank_info(name: str, dp_rank: int, page_table, *, dp_size: int = 4, pp_size: int = 1):
    return RankInfo(
        planner_id=DeepseekMlaLatentTransferPlanner.planner_id,
        instance_name=name,
        instance_rank=dp_rank,
        tp_size=dp_size,
        tp_rank=dp_rank,
        pp_size=pp_size,
        pp_rank=0,
        dp_size=dp_size,
        dp_rank=dp_rank,
        layer_num_per_pp=[len(page_table.layer_groups[0].local_layers)],
        sender_endpoints=[],
        self_endpoint="",
        transfer_engine_info=b"",
        attention=AttentionInfo(
            kv_heads_per_rank=1,
            tokens_per_block=TPB,
            dims_per_head=512,
            element_bytes=1,
            enable_attention_dp=True,
            is_mla=True,
        ),
        page_table=page_table,
    )


def _pair(dp_rank: int = 2):
    self_pt, peer_pt = _mla_page_table(0x1000_0000), _mla_page_table(0x9000_0000)
    self_ri = _dep4_rank_info("ctx", dp_rank, self_pt)
    peer_ri = _dep4_rank_info("gen", dp_rank, peer_pt)
    ext = KVRegionExtractorV1(self_pt)
    return (
        DeepseekMlaLatentTransferPlanner(self_ri, ext),
        DefaultTransferPlanner(self_ri, ext),
        self_ri,
        peer_ri,
    )


def _fragments_at_slot_granularity(plan: TransferPlan) -> set:
    """Split merged runs into per-slot pieces so plans can be compared as sets."""
    out = set()
    for s, d, n in zip(plan.src_ptrs.tolist(), plan.dst_ptrs.tolist(), plan.sizes.tolist()):
        assert n % SLOT_BYTES == 0
        for k in range(n // SLOT_BYTES):
            out.add((s + k * SLOT_BYTES, d + k * SLOT_BYTES, SLOT_BYTES))
    return out


def _request(src, dst, dst_start_token=None):
    """Sender-side chunk and receiver-side request for one single-layer-group transfer."""
    prompt_len = len(src) * TPB
    chunk = Chunk(
        block_ids_per_layer_groups=[np.array(src, dtype=np.int64)],
        kind_per_layer_group=[CacheKind.PAGED],
        token_range=TokenRange(0, prompt_len),
        is_last=True,
    )
    req_info = RecvReqInfo(
        sender_req_id=1,
        instance_name="gen",
        instance_rank=2,
        block_ids_per_layer_groups=[np.array(dst, dtype=np.int64)],
        unique_rid=1,
        dst_start_token=dst_start_token,
    )
    return chunk, req_info, prompt_len


@pytest.mark.parametrize(
    "src,dst,dst_start_token",
    [
        ([3, 4, 5, 6], [10, 11, 12, 13], None),  # all consecutive both sides
        ([3, 4, 9, 10], [10, 11, 12, 13], None),  # src breaks a run
        ([3, 4, 5, 6], [10, 11, 20, 21], None),  # dst breaks a run
        ([3, -1, 5, 6], [10, 11, -1, 13], None),  # holes on both sides
        ([3, 4, 5, 6], [10, 11, 12, 13], 2 * TPB),  # receiver prefix reuse
        ([3, 4, 5, 6], [-1, -1, -1, -1], None),  # peer holds nothing
    ],
    ids=["contiguous", "src_break", "dst_break", "holes", "dst_start", "nothing_to_send"],
)
def test_plan_matches_default_planner_byte_for_byte(src, dst, dst_start_token):
    target, default, self_ri, peer_ri = _pair()
    chunk, req_info, prompt_len = _request(src, dst, dst_start_token)

    describer = DefaultCacheDescriber()
    ours = plan_transfer(describer, target, self_ri, peer_ri, chunk, req_info, prompt_len)
    ref = plan_transfer(describer, default, self_ri, peer_ri, chunk, req_info, prompt_len)

    assert _fragments_at_slot_granularity(ours) == _fragments_at_slot_granularity(ref)
    assert int(ours.sizes.sum()) == int(ref.sizes.sum())


def test_contiguous_blocks_merge_into_one_fragment():
    target, _, self_ri, peer_ri = _pair()
    plan = plan_transfer(
        DefaultCacheDescriber(), target, self_ri, peer_ri, *_request([3, 4, 5, 6], [10, 11, 12, 13])
    )
    assert plan.sizes.tolist() == [4 * SLOT_BYTES]


def test_targets_is_the_matching_dp_rank_only():
    target, default, self_ri, peer_ri = _pair(dp_rank=2)
    ours, ref = target.targets(self_ri, peer_ri, 2), default.targets(self_ri, peer_ri, 2)
    assert ours.ranks == ref.ranks == [2]
    assert ours.uniform_writers is True


def test_check_peer_refuses_a_peer_whose_latent_geometry_differs():
    target, _, self_ri, peer_ri = _pair()
    target.check_peer(self_ri, peer_ri)
    with pytest.raises(PeerIncompatibleError, match="pool"):
        target.check_peer(
            self_ri, _dep4_rank_info("gen", 2, _mla_page_table(0x9000_0000, extra_pool=True))
        )
    with pytest.raises(PeerIncompatibleError, match="layer"):
        target.check_peer(
            self_ri, _dep4_rank_info("gen", 2, _mla_page_table(0x9000_0000, num_layers=5))
        )
