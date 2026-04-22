# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Unit tests for ``DistConfig`` (CPU-only)."""

from tensorrt_llm._torch.auto_deploy.utils.dist_config import DistConfig
from tensorrt_llm.mapping import Mapping


def test_defaults():
    cfg = DistConfig()
    assert cfg.world_size == 1
    assert cfg.rank == 0
    assert cfg.tp_size == 1
    assert cfg.pp_size == 1
    assert cfg.moe_tp_size == 1
    assert cfg.moe_ep_size == 1
    assert cfg.moe_cluster_size == 1
    assert cfg.enable_attention_dp is False
    assert cfg.allreduce_strategy == "NCCL"


def test_serialize_deserialize_roundtrip():
    original = DistConfig(
        world_size=16,
        rank=7,
        tp_size=4,
        pp_size=2,
        moe_tp_size=2,
        moe_ep_size=2,
        moe_cluster_size=1,
        enable_attention_dp=True,
        allreduce_strategy="CUSTOM",
    )
    restored = DistConfig.deserialize(original.serialize())
    assert restored == original


def test_from_dict_ignores_unknown_keys():
    cfg = DistConfig.from_dict(
        {
            "world_size": 4,
            "tp_size": 4,
            "moe_ep_size": 4,
            "not_a_real_field": "ignore_me",
            "another_extra": 123,
        }
    )
    assert cfg.world_size == 4
    assert cfg.rank == 0


def test_from_mapping_to_mapping_roundtrip():
    m = Mapping(
        world_size=8,
        rank=5,
        tp_size=4,
        pp_size=2,
        moe_tp_size=2,
        moe_ep_size=2,
        moe_cluster_size=1,
        enable_attention_dp=True,
    )
    dist = DistConfig.from_mapping(m)
    m2 = dist.to_mapping()
    assert m2.world_size == m.world_size
    assert m2.rank == m.rank
    assert m2.tp_size == m.tp_size
    assert m2.pp_size == m.pp_size
    assert m2.moe_tp_size == m.moe_tp_size
    assert m2.moe_ep_size == m.moe_ep_size
    assert m2.moe_cluster_size == m.moe_cluster_size
    assert m2.enable_attention_dp == m.enable_attention_dp


def test_tp_rank_property():
    assert DistConfig(world_size=8, rank=3, tp_size=4, moe_ep_size=4).tp_rank == 3
    assert DistConfig(world_size=8, rank=5, tp_size=4, moe_ep_size=4).tp_rank == 1


def test_moe_ep_rank_property():
    cfg = DistConfig(world_size=8, rank=3, tp_size=8, moe_ep_size=4, moe_tp_size=2)
    assert cfg.tp_rank == 3
    assert cfg.moe_ep_rank == 3
    assert cfg.moe_ep_rank == cfg.tp_rank % cfg.moe_ep_size


def test_allreduce_strategy_default():
    assert DistConfig().allreduce_strategy == "NCCL"


def test_invalid_rank_raises():
    import pytest

    with pytest.raises(ValueError, match="rank.*must be < world_size"):
        DistConfig(world_size=4, rank=5, tp_size=4, moe_ep_size=4)


def test_invalid_moe_grid_raises():
    import pytest

    with pytest.raises(ValueError, match="moe_tp_size.*must equal tp_size"):
        DistConfig(world_size=8, rank=0, tp_size=8, moe_tp_size=2, moe_ep_size=2)
