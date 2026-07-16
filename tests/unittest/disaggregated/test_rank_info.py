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

from types import SimpleNamespace

import numpy as np
import pytest

from tensorrt_llm._torch.disaggregation.native import rank_info as rank_info_module
from tensorrt_llm._torch.disaggregation.native.auxiliary import AuxBufferMeta
from tensorrt_llm._torch.disaggregation.native.rank_info import RankInfo
from tensorrt_llm.bindings import DataType


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
        server_endpoint="tcp://10.0.0.1:5000",
        self_endpoint="tcp://10.0.0.1:5001",
        transfer_engine_info=b"\x00\x01\x02",
    )
    assert ri.instance_name == "gen_0"
    assert ri.tp_size == 2
    assert ri.pp_size == 1
    assert ri.layer_num_per_pp == [32]
    assert ri.sender_endpoints == ["tcp://10.0.0.1:5000"]


def test_rank_info_msgpack_roundtrip():
    ri = RankInfo(
        instance_name="gen_0",
        instance_rank=0,
        tp_size=2,
        tp_rank=0,
        pp_size=1,
        pp_rank=0,
        layer_num_per_pp=[32],
        sender_endpoints=["tcp://10.0.0.1:5000"],
        server_endpoint="tcp://10.0.0.1:5000",
        self_endpoint="tcp://10.0.0.1:5001",
        transfer_engine_info=b"\x00\x01\x02",
    )
    data = ri.to_bytes()
    restored = RankInfo.from_bytes(data)
    assert restored.instance_name == ri.instance_name
    assert restored.tp_size == ri.tp_size
    assert restored.transfer_engine_info == ri.transfer_engine_info
    assert restored.aux_meta is None


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
        server_endpoint="tcp://10.0.0.1:5000",
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


@pytest.mark.parametrize(
    ("dtype", "expected_element_bytes", "expected_type"),
    [(DataType.NVFP4, 0.5, float), (DataType.HALF, 2, int)],
)
def test_rank_info_represents_cache_element_bytes(
    monkeypatch, dtype, expected_element_bytes, expected_type
):
    monkeypatch.setattr(rank_info_module, "build_page_table_from_manager", lambda _manager: None)
    manager = SimpleNamespace(
        mapping=SimpleNamespace(
            rank=0,
            tp_size=2,
            tp_rank=0,
            pp_size=1,
            pp_rank=0,
            dp_size=1,
            cp_size=1,
            cp_rank=0,
            enable_attention_dp=False,
        ),
        pp_layers=[0],
        num_kv_heads_per_layer=[4],
        tokens_per_block=64,
        head_dim=128,
        dtype=dtype,
        kv_factor=2,
    )

    rank_info = RankInfo.from_kv_cache_manager("ctx", manager, device_id=0)

    assert rank_info.attention.element_bytes == expected_element_bytes
    assert isinstance(rank_info.attention.element_bytes, expected_type)

    restored = RankInfo.from_bytes(rank_info.to_bytes())
    assert restored.attention.element_bytes == expected_element_bytes
    assert isinstance(restored.attention.element_bytes, expected_type)
