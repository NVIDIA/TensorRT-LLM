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

"""Which pools ``PeerRegistrar.get_kv_maps`` pairs.

A rank's pools pair with a peer rank's by where each buffer's runs go: the pool (and layer group)
a buffer lies in on this rank and the pool its counterpart lies in on the peer. Nothing matches
pools by role or by kind.
"""

from collections import defaultdict

import pytest
from layout_test_utils import native_page_table

from tensorrt_llm._torch.disaggregation.native.peer import PeerRegistrar
from tensorrt_llm._torch.disaggregation.native.rank_info import RankInfo
from tensorrt_llm._torch.disaggregation.resource.kv_extractor import KVRegionExtractorV1
from tensorrt_llm._torch.pyexecutor.kv_cache.page_layout import (
    BufferDesc,
    LayerGroupDesc,
    PageLayout,
    RankTopology,
    head_major,
    replicated,
)

pytestmark = pytest.mark.cpu_only

HEAD_BYTES = 64


def kv(heads=2, total=2, pool=0):
    """K and V of ``heads`` heads of ``total``, in pool ``pool`` of their layer group."""
    size = heads * HEAD_BYTES
    return [(role, pool, size, head_major(size, heads, total)) for role in ("key", "value")]


def page_table(groups, topology=RankTopology()):
    """A native page table of ``groups``.

    Per layer group (kind, pool group, layers, roles), each role (name, pool, size, tensors) lies
    layer after layer within its pool's slot.
    """
    ends = defaultdict(int)
    descs, buffers = [], []
    for kind, g, layers, roles in groups:
        descs.append(LayerGroupDesc(kind, None, 0, g, tuple(layers)))
        for layer in layers:
            for role, pool, size, tensors in roles:
                buffers.append(BufferDesc(layer, role, g, pool, ends[g, pool], size, tensors))
                ends[g, pool] += size
    widths = defaultdict(dict)
    for (g, pool), width in ends.items():
        widths[g][pool] = width
    layout = PageLayout(
        tokens_per_page=16,
        pool_slot_bytes={g: tuple(w[p] for p in range(len(w))) for g, w in widths.items()},
        layer_groups=tuple(descs),
        buffers=tuple(buffers),
        topology=topology,
    )
    bases = {
        g: [0x100000 * (g + 1) + 0x10000 * p for p in range(len(w))]
        for g, w in layout.pool_slot_bytes.items()
    }
    return native_page_table(layout, bases)


def rank(name, table, *, tp_size=1, tp_rank=0, layers=(2,)):
    return RankInfo(
        instance_name=name,
        instance_rank=tp_rank,
        tp_size=tp_size,
        tp_rank=tp_rank,
        pp_size=len(layers),
        pp_rank=0,
        dp_size=1,
        dp_rank=0,
        cp_size=1,
        cp_rank=0,
        device_id=0,
        layer_num_per_pp=list(layers),
        self_endpoint="",
        transfer_engine_info=b"",
        enable_attention_dp=False,
        aux_meta=None,
        page_table=table,
        sender_endpoints=[],
    )


def kv_maps(own, peer, excluded=frozenset()):
    """{pool pair: bytes per slot} of what ``own``'s rank sends ``peer``'s."""
    maps = PeerRegistrar(own, KVRegionExtractorV1(own.page_table)).get_kv_maps(peer, excluded)
    return {pair: mapper.bytes_per_slot for pair, mapper in maps.items()}


def test_a_pool_pairs_with_the_peer_pool_its_buffers_land_in():
    """K/V and their block scales in two pools pair across.

    They do even where the peer orders the two pools the other way round.
    """
    scales = [(f"{role}_block_scale", 1, 16, head_major(16, 2, 2)) for role in ("key", "value")]
    peer_scales = [(role, 0, size, t) for role, _, size, t in scales]
    own = rank("ctx", page_table([("attention", 0, (0, 1), kv() + scales)]))
    peer = rank("gen", page_table([("attention", 0, (0, 1), kv(pool=1) + peer_scales)]))
    assert kv_maps(own, peer) == {((0, 0), (0, 1)): 512, ((0, 1), (0, 0)): 64}


def test_a_replicated_index_cache_is_sent_by_one_of_the_ranks_fanning_in():
    """TP2 into TP1: both ranks send their K/V heads, one the index-K every rank holds alike."""
    index = [("index_key", 1, 32, replicated(32))]
    peer = rank("gen", page_table([("attention", 0, (0, 1), kv(4, 4) + index)]))
    sent = [
        kv_maps(
            rank(
                "ctx",
                page_table(
                    [("attention", 0, (0, 1), kv(2, 4) + index)],
                    RankTopology(tp_size=2, tp_rank=r),
                ),
                tp_size=2,
                tp_rank=r,
            ),
            peer,
        )
        for r in range(2)
    ]
    assert sent == [{((0, 0), (0, 0)): 512, ((0, 1), (0, 1)): 64}, {((0, 0), (0, 0)): 512}]


def test_only_the_layers_both_ranks_hold_move():
    own = rank("ctx", page_table([("attention", 0, (10, 11), kv())]))
    peer = rank("gen", page_table([("attention", 0, (11, 12), kv())]), layers=(1, 2))
    assert kv_maps(own, peer) == {((0, 0), (0, 0)): 256}


def test_a_layer_group_pairs_with_every_peer_group_holding_its_layers():
    """Each layer's bytes go to the pool of the peer's layer group holding it.

    Here the peer keeps layers 0 and 1 in layer groups of their own.
    """
    own = rank("ctx", page_table([("attention", 0, (0, 1), kv())]))
    peer = rank("gen", page_table([("attention", 0, (0,), kv()), ("attention", 1, (1,), kv())]))
    assert kv_maps(own, peer) == {((0, 0), (0, 0)): 256, ((0, 0), (1, 0)): 256}


def test_buffers_a_transfer_leaves_out_move_nothing():
    index = [("index_key", 1, 32, replicated(32))]
    own = rank("ctx", page_table([("attention", 0, (0, 1), kv() + index)]))
    peer = rank("gen", page_table([("attention", 0, (0, 1), kv() + index)]))
    excluded = frozenset({(1, "key"), (1, "value"), (0, "index_key"), (1, "index_key")})
    assert kv_maps(own, peer, excluded) == {((0, 0), (0, 0)): 256}


def test_a_layer_in_two_layer_groups_of_one_pool_group_is_refused():
    own = rank("ctx", page_table([("attention", 0, (0,), kv()), ("attention", 0, (0, 1), [])]))
    peer = rank("gen", page_table([("attention", 0, (0, 1), kv())]))
    with pytest.raises(ValueError, match="layer 0 lies in layer groups 0 and 1"):
        kv_maps(own, peer)
