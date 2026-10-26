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

import dataclasses
from types import SimpleNamespace

import numpy as np
import pytest
from pages_test_utils import pages

from tensorrt_llm import bindings
from tensorrt_llm._torch.disaggregation.native.auxiliary import AuxBufferMeta
from tensorrt_llm._torch.disaggregation.native.rank_info import RankInfo
from tensorrt_llm._torch.disaggregation.page_mapping import RankPages, ShardGrid, shard_grid
from tensorrt_llm._torch.pyexecutor.kv_cache.layout_builder import _topology, kv_dtype

pytestmark = pytest.mark.cpu_only


def test_rank_info_construction():
    ri = RankInfo(
        instance_name="gen_0",
        instance_rank=0,
        tp_size=2,
        tp_rank=0,
        pp_size=1,
        pp_rank=0,
        layer_num_per_pp=[32],
        sender_endpoints=["tcp://10.0.0.1:5000"],
        self_endpoint="tcp://10.0.0.1:5001",
        transfer_engine_info=b"\x00\x01\x02",
    )
    assert ri.instance_name == "gen_0"
    assert ri.tp_size == 2
    assert ri.pp_size == 1
    assert ri.layer_num_per_pp == [32]
    assert ri.sender_endpoints == ["tcp://10.0.0.1:5000"]
    assert ri.bounded_replay_on_generation is False


@pytest.mark.parametrize("bounded_replay_on_generation", [False, True])
def test_rank_info_msgpack_roundtrip(bounded_replay_on_generation: bool) -> None:
    ri = RankInfo(
        instance_name="gen_0",
        instance_rank=0,
        tp_size=2,
        tp_rank=0,
        pp_size=1,
        pp_rank=0,
        layer_num_per_pp=[32],
        sender_endpoints=["tcp://10.0.0.1:5000"],
        self_endpoint="tcp://10.0.0.1:5001",
        transfer_engine_info=b"\x00\x01\x02",
        bounded_replay_on_generation=bounded_replay_on_generation,
    )
    data = ri.to_bytes()
    restored = RankInfo.from_bytes(data)
    assert restored.instance_name == ri.instance_name
    assert restored.tp_size == ri.tp_size
    assert restored.transfer_engine_info == ri.transfer_engine_info
    assert restored.aux_meta is None
    assert restored.bounded_replay_on_generation is bounded_replay_on_generation


def test_rank_info_roundtrip_with_aux_meta():
    meta = AuxBufferMeta(
        ptrs=np.array([0x4000, 0x5000], dtype=np.int64),
        size=np.array([1024, 2048], dtype=np.int64),
        item_sizes=np.array([64, 128], dtype=np.int64),
        device="cpu",
    )
    ri = RankInfo(
        instance_name="gen_0",
        instance_rank=0,
        tp_size=1,
        tp_rank=0,
        pp_size=1,
        pp_rank=0,
        layer_num_per_pp=[32],
        sender_endpoints=["tcp://10.0.0.1:5000"],
        self_endpoint="tcp://10.0.0.1:5001",
        transfer_engine_info=b"",
        aux_meta=meta,
    )
    data = ri.to_bytes()
    restored = RankInfo.from_bytes(data)
    assert restored.aux_meta is not None
    np.testing.assert_array_equal(restored.aux_meta.ptrs, [0x4000, 0x5000])
    np.testing.assert_array_equal(restored.aux_meta.size, [1024, 2048])
    np.testing.assert_array_equal(restored.aux_meta.item_sizes, [64, 128])
    assert restored.aux_meta.device == "cpu"


def _mapping(**overrides) -> SimpleNamespace:
    fields = dict(
        rank=0,
        tp_size=1,
        tp_rank=0,
        pp_size=1,
        pp_rank=0,
        dp_size=1,
        cp_size=1,
        cp_rank=0,
        enable_attention_dp=False,
    )
    fields.update(overrides)
    return SimpleNamespace(**fields)


def _pages_of_topology(manager) -> RankPages:
    """The pages of a manager holding no layer group: only its place among the ranks."""
    return pages([], [], topology=_topology(manager))


def test_from_kv_cache_manager_carries_bounded_replay(monkeypatch) -> None:
    monkeypatch.setattr(RankPages, "of_manager", staticmethod(lambda _manager: None))
    manager = SimpleNamespace(mapping=_mapping(), pp_layers=[0, 1, 2])

    info = RankInfo.from_kv_cache_manager("ctx", manager, device_id=0)
    assert info.layer_num_per_pp == [3]
    assert info.bounded_replay_on_generation is False

    manager.bounded_replay_on_generation = True
    info = RankInfo.from_kv_cache_manager("ctx", manager, device_id=0)
    assert info.bounded_replay_on_generation is True


def test_from_kv_cache_manager_preserves_attention_dp_on_attention_free_stage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(RankPages, "of_manager", staticmethod(_pages_of_topology))
    manager = SimpleNamespace(
        mapping=_mapping(rank=1, tp_size=2, tp_rank=1, enable_attention_dp=True),
        pp_layers=[0, 1],
    )

    info = RankInfo.from_kv_cache_manager("ctx", manager, device_id=0)

    assert (info.dp_size, info.dp_rank) == (2, 1)
    assert info.pages.layout.topology.attention_dp
    assert info.tp_size_per_dp_group == 1
    assert shard_grid(("tp", "cp"), info) == ShardGrid(size=1, rank=0)
    restored = RankInfo.from_bytes(info.to_bytes())
    assert restored.tp_size_per_dp_group == 1

    manager.mapping = _mapping(rank=1, tp_size=2, tp_rank=1)
    info = RankInfo.from_kv_cache_manager("ctx", manager, device_id=0)
    assert info.tp_size_per_dp_group == 2
    assert shard_grid(("tp", "cp"), info) == ShardGrid(size=2, rank=1)


@pytest.mark.parametrize(
    ("dtype", "expected_name"),
    [(bindings.DataType.NVFP4, "NVFP4"), (bindings.DataType.HALF, "HALF")],
)
def test_rank_info_carries_cache_dtype(dtype, expected_name):
    manager = SimpleNamespace(dtype=dtype)
    assert kv_dtype(manager) == expected_name
    rank_pages = pages([], [])
    rank_pages = RankPages(
        dataclasses.replace(rank_pages.layout, dtype=kv_dtype(manager)), rank_pages.pools
    )
    rank_info = RankInfo(
        instance_name="ctx",
        instance_rank=0,
        tp_size=2,
        tp_rank=0,
        pp_size=1,
        pp_rank=0,
        layer_num_per_pp=[1],
        sender_endpoints=[],
        self_endpoint="",
        transfer_engine_info=b"",
        pages=rank_pages,
    )

    restored = RankInfo.from_bytes(rank_info.to_bytes())
    assert restored.pages.layout.dtype == expected_name
