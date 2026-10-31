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

from tensorrt_llm import bindings
from tensorrt_llm._torch.disaggregation.native import rank_info as rank_info_module
from tensorrt_llm._torch.disaggregation.native.auxiliary import AuxBufferMeta
from tensorrt_llm._torch.disaggregation.native.mixers.ssm.peer import MambaPolicy
from tensorrt_llm._torch.disaggregation.native.rank_info import RankInfo

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


def test_from_kv_cache_manager_uses_first_nonzero_kv_head_count(monkeypatch) -> None:
    monkeypatch.setattr(rank_info_module, "build_page_table_from_manager", lambda _: None)
    mapping = SimpleNamespace(
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
    manager = SimpleNamespace(
        mapping=mapping,
        num_kv_heads_per_layer=[0, 8, 0],
        pp_layers=[0, 1, 2],
        tokens_per_block=32,
        head_dim=128,
        dtype=bindings.DataType.HALF,
        kv_factor=2,
    )

    info = RankInfo.from_kv_cache_manager("ctx", manager, device_id=0)

    assert info.attention.kv_heads_per_rank == 8
    assert info.bounded_replay_on_generation is False

    manager.bounded_replay_on_generation = True
    info = RankInfo.from_kv_cache_manager("ctx", manager, device_id=0)
    assert info.bounded_replay_on_generation is True


def test_from_kv_cache_manager_preserves_attention_dp_on_attention_free_stage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(rank_info_module, "build_page_table_from_manager", lambda _: None)
    mapping = SimpleNamespace(
        rank=1,
        tp_size=2,
        tp_rank=1,
        pp_size=1,
        pp_rank=0,
        dp_size=1,
        cp_size=1,
        cp_rank=0,
        enable_attention_dp=True,
    )
    manager = SimpleNamespace(
        mapping=mapping,
        num_kv_heads_per_layer=[0, 0],
        pp_layers=[0, 1],
        tokens_per_block=32,
        head_dim=128,
        dtype=bindings.DataType.HALF,
        kv_factor=2,
    )

    info = RankInfo.from_kv_cache_manager("ctx", manager, device_id=0)

    assert info.attention is not None
    assert info.attention.kv_heads_per_rank == 0
    assert info.attention.enable_attention_dp
    assert MambaPolicy._mamba_tp(info) == (1, 0)


@pytest.mark.parametrize(
    ("dtype", "expected_element_bytes", "expected_type"),
    [(bindings.DataType.NVFP4, 0.5, float), (bindings.DataType.HALF, 2, int)],
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


def _mamba_rank_info(tp_size=1, tp_rank=0, cp_size=1, cp_rank=0, attention_dp=None):
    # MambaPolicy reads tp_size / tp_rank / cp_size / cp_rank /
    # attention.enable_attention_dp.
    attention = None if attention_dp is None else SimpleNamespace(enable_attention_dp=attention_dp)
    return SimpleNamespace(
        tp_size=tp_size, tp_rank=tp_rank, cp_size=cp_size, cp_rank=cp_rank, attention=attention
    )


class TestMambaHelixCP:
    """MambaPolicy under decode-CP (helix): shard grid and pairing."""

    def test_mamba_tp_helix_grid_is_cp_minor(self):
        # cp == 1 keeps the plain TP grid; cp > 1 flattens tp*cp CP-minor.
        assert MambaPolicy._mamba_tp(_mamba_rank_info()) == (1, 0)
        assert MambaPolicy._mamba_tp(_mamba_rank_info(cp_size=32, cp_rank=5)) == (32, 5)
        ri = _mamba_rank_info(tp_size=2, tp_rank=1, cp_size=4, cp_rank=3)
        assert MambaPolicy._mamba_tp(ri) == (8, 7)

    def test_mamba_tp_helix_attention_dp_shards_over_cp(self):
        # Helix + attention DP (K3 helix_a2a): the CP group shares every
        # request, so the state stays head-sharded across it; must match the
        # executor's pool sizing in config_utils.mamba_effective_tp_size.
        from tensorrt_llm._torch.pyexecutor.config_utils import mamba_effective_tp_size

        ri = _mamba_rank_info(cp_size=16, cp_rank=5, attention_dp=True)
        assert MambaPolicy._mamba_tp(ri) == (16, 5)
        mapping = SimpleNamespace(
            enable_attention_dp=True, tp_size=1, cp_size=16, has_cp_helix=lambda: True
        )
        assert mamba_effective_tp_size(mapping) == 16
        # Attention DP without CP still replicates.
        assert MambaPolicy._mamba_tp(_mamba_rank_info(tp_size=4, tp_rank=2, attention_dp=True)) == (
            1,
            0,
        )

    def test_equal_width_pairs_only_matching_rank(self):
        # Regression lock for the fan-in collapse: with equal shard widths the
        # mapper is a whole-slot copy, so only the same-index sender may pair;
        # any other sender would overwrite the receiver slot with the wrong heads.
        for sender_rank in range(2):
            for gen_cp_rank in range(2):
                sender_ri = _mamba_rank_info(tp_size=2, tp_rank=sender_rank)
                peer_ri = _mamba_rank_info(cp_size=2, cp_rank=gen_cp_rank)
                paired = MambaPolicy.is_paired(sender_ri, peer_ri)
                if sender_rank == gen_cp_rank:
                    assert paired
                else:
                    assert not paired

    def test_narrow_to_wide_pairing_follows_covering_tree(self):
        # Sender grid 2, receiver grid 4: receiver g pairs with sender g // 2.
        for sender_rank in range(2):
            for gen_cp_rank in range(4):
                sender_ri = _mamba_rank_info(tp_size=2, tp_rank=sender_rank)
                peer_ri = _mamba_rank_info(cp_size=4, cp_rank=gen_cp_rank)
                paired = MambaPolicy.is_paired(sender_ri, peer_ri)
                if gen_cp_rank // 2 == sender_rank:
                    assert paired, (sender_rank, gen_cp_rank)
                else:
                    assert not paired, (sender_rank, gen_cp_rank)

    def test_is_paired_unpaired_vs_paired(self):
        # Unpaired: sender rank 0 is not paired with receiver cp_rank 1
        sender_ri = _mamba_rank_info(tp_size=2, tp_rank=0)
        peer_ri = _mamba_rank_info(cp_size=2, cp_rank=1)
        assert not MambaPolicy.is_paired(sender_ri, peer_ri)
        # Paired sender
        paired_ri = _mamba_rank_info(tp_size=2, tp_rank=1)
        assert MambaPolicy.is_paired(paired_ri, peer_ri)
