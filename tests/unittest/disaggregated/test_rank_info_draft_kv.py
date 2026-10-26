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
"""RankInfo's optional draft-pool geometry (disagg drafter context window).

Wire round trip, the unchanged target-only wire format, and the draft view.
"""

from dataclasses import asdict
from types import SimpleNamespace

import msgpack
import numpy as np
import pytest

from tensorrt_llm._torch.disaggregation.native import rank_info as rank_info_module
from tensorrt_llm._torch.disaggregation.native.mixers.attention.spec import AttentionInfo
from tensorrt_llm._torch.disaggregation.native.rank_info import RankInfo
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


def _attention(heads, tpb, dim):
    return AttentionInfo(
        kv_heads_per_rank=heads,
        tokens_per_block=tpb,
        dims_per_head=dim,
        element_bytes=2,
        enable_attention_dp=True,
        is_mla=False,
    )


def _page_table(base, tpb, buf):
    entries = np.array([(0, 0, buf), (0, buf, buf)], dtype=BUFFER_ENTRY_DTYPE)
    return KVCachePageTable(
        tokens_per_block=tpb,
        layer_groups=[
            AttentionLayerGroup(
                pool_group_idx=0,
                kv_head_num_per_rank=4,
                sliding_window_size=2048,
                local_layers=[LocalLayer(0, 0)],
                pool_views=[PoolView(pool_idx=0, buffer_entries=entries, bytes_per_layer=2 * buf)],
            )
        ],
        pool_groups=[
            PhysicalPoolGroup(
                pools=[PhysicalPool(base_address=base, slot_bytes=2 * buf, num_slots=9)]
            )
        ],
    )


def _rank_info(with_draft: bool) -> RankInfo:
    ri = RankInfo(
        instance_name="ctx",
        instance_rank=1,
        tp_size=2,
        tp_rank=1,
        pp_size=1,
        pp_rank=0,
        layer_num_per_pp=[62],
        sender_endpoints=["tcp://a:1", "tcp://a:2"],
        self_endpoint="tcp://a:3",
        transfer_engine_info=b"\x01\x02",
        dp_size=2,
        dp_rank=1,
        attention=_attention(8, 64, 128),
        page_table=_page_table(0x10000, 64, 4096),
    )
    if with_draft:
        ri.draft_attention = _attention(4, 32, 128)
        ri.draft_page_table = _page_table(0x90000, 32, 2048)
    return ri


def _legacy_bytes(ri: RankInfo) -> bytes:
    """What a rank built before the draft fields serialized: no draft keys at all."""
    data = asdict(ri)
    data.pop("draft_attention")
    data.pop("draft_page_table")
    data.pop("draft_receive_disabled")
    data["attention"] = ri.attention.to_dict()
    data["aux_meta"] = None
    data["page_table"] = ri.page_table.to_dict()
    return msgpack.packb(data)


def test_target_only_rank_info_serializes_exactly_as_before():
    ri = _rank_info(with_draft=False)
    data = ri.to_bytes()
    assert data == _legacy_bytes(ri)
    unpacked = msgpack.unpackb(data, strict_map_key=False)
    assert "draft_attention" not in unpacked and "draft_page_table" not in unpacked
    assert "draft_receive_disabled" not in unpacked
    restored = RankInfo.from_bytes(data)
    assert restored.draft_attention is None and restored.draft_page_table is None
    assert not restored.has_draft_kv
    assert restored.draft_view() is None


def test_legacy_peer_bytes_parse_without_draft_fields():
    legacy = _legacy_bytes(_rank_info(with_draft=False))
    restored = RankInfo.from_bytes(legacy)
    assert restored.draft_view() is None
    assert restored.page_table.to_dict() == _rank_info(with_draft=False).page_table.to_dict()


def test_draft_fields_round_trip():
    ri = _rank_info(with_draft=True)
    restored = RankInfo.from_bytes(ri.to_bytes())
    assert restored.has_draft_kv
    assert restored.draft_attention == ri.draft_attention
    assert restored.draft_page_table.to_dict() == ri.draft_page_table.to_dict()
    # The target geometry is untouched by the draft set.
    assert restored.attention == ri.attention
    assert restored.page_table.to_dict() == ri.page_table.to_dict()


def test_draft_view_swaps_only_the_geometry():
    ri = _rank_info(with_draft=True)
    view = RankInfo.from_bytes(ri.to_bytes()).draft_view()
    assert view.attention == ri.draft_attention
    assert view.page_table.to_dict() == ri.draft_page_table.to_dict()
    assert not view.has_draft_kv
    for name in (
        "instance_name",
        "instance_rank",
        "tp_size",
        "tp_rank",
        "pp_size",
        "pp_rank",
        "dp_size",
        "dp_rank",
        "layer_num_per_pp",
        "sender_endpoints",
        "self_endpoint",
        "transfer_engine_info",
    ):
        assert getattr(view, name) == getattr(ri, name), name
    assert view.tp_size_per_dp_group == ri.tp_size_per_dp_group == 1


def test_rank_info_built_without_init_has_no_draft_set():
    ri = object.__new__(RankInfo)
    assert not ri.has_draft_kv
    assert ri.draft_view() is None


def _manager(heads, tpb, head_dim, layers):
    mapping = SimpleNamespace(
        rank=0,
        tp_size=2,
        tp_rank=0,
        pp_size=1,
        pp_rank=0,
        dp_size=1,
        cp_size=1,
        cp_rank=0,
        enable_attention_dp=True,
    )
    return SimpleNamespace(
        mapping=mapping,
        num_kv_heads_per_layer=[heads] * layers,
        pp_layers=list(range(layers)),
        tokens_per_block=tpb,
        head_dim=head_dim,
        dtype=None,
        kv_factor=2,
    )


@pytest.mark.parametrize("with_draft", [False, True])
def test_from_kv_cache_manager_reads_the_draft_pool_geometry(monkeypatch, with_draft):
    tables = {}

    def build(manager):
        tables[id(manager)] = object()
        return tables[id(manager)]

    monkeypatch.setattr(rank_info_module, "build_page_table_from_manager", build)
    monkeypatch.setattr(rank_info_module, "get_size_in_bytes", lambda n, _dtype: 2 * n)
    target = _manager(heads=8, tpb=64, head_dim=128, layers=62)
    draft = _manager(heads=4, tpb=32, head_dim=128, layers=1) if with_draft else None

    ri = RankInfo.from_kv_cache_manager("ctx", target, 0, draft_kv_cache_manager=draft)

    assert ri.attention == AttentionInfo(8, 64, 128, 2, True, False)
    assert ri.page_table is tables[id(target)]
    if with_draft:
        assert ri.draft_attention == AttentionInfo(4, 32, 128, 2, True, False)
        assert ri.draft_page_table is tables[id(draft)]
    else:
        assert ri.draft_attention is None and ri.draft_page_table is None
