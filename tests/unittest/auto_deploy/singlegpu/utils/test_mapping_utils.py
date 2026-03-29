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

"""Tests for tensorrt_llm._torch.auto_deploy.utils.mapping_utils."""

from tensorrt_llm._torch.auto_deploy.utils.dist_config import DistConfig
from tensorrt_llm._torch.auto_deploy.utils.mapping_utils import (
    deserialize_mapping,
    print_grid,
    print_rank,
    serialize_dist_config,
)
from tensorrt_llm.mapping import Mapping


def _valid_dist_config() -> DistConfig:
    """Values consistent with Mapping validation (tp*pp == world_size, MoE product == tp)."""
    return DistConfig(
        world_size=8,
        rank=3,
        tp_size=2,
        pp_size=4,
        moe_tp_size=2,
        moe_ep_size=1,
        moe_cluster_size=1,
        enable_attention_dp=False,
        allreduce_strategy="NCCL",
    )


def test_deserialize_mapping_with_distconfig_json():
    dc = _valid_dist_config()
    dc = dc.model_copy(update={"allreduce_strategy": "CUSTOM"})
    payload = dc.serialize()
    assert "allreduce_strategy" in payload

    m = deserialize_mapping(payload)
    assert isinstance(m, Mapping)
    assert m.world_size == dc.world_size
    assert m.rank == dc.rank


def test_serialize_dist_config_roundtrip():
    dc = _valid_dist_config()
    dc = dc.model_copy(update={"allreduce_strategy": "CUSTOM"})

    s = serialize_dist_config(dc)
    back = DistConfig.deserialize(s)

    assert back.model_dump() == dc.model_dump()


def test_print_grid_format():
    dc = _valid_dist_config()
    out = print_grid(dc)
    assert "TP" in out
    assert "MoE_TP" in out
    assert "MoE_EP" in out
    assert str(dc.tp_size) in out
    assert str(dc.moe_tp_size) in out
    assert str(dc.moe_ep_size) in out


def test_print_rank_format():
    dc = _valid_dist_config()
    out = print_rank(dc)
    assert "rank" in out
    assert str(dc.rank) in out
    assert str(dc.moe_tp_rank) in out
    assert str(dc.moe_ep_rank) in out


def test_deserialize_mapping_preserves_fields():
    dc = _valid_dist_config()
    m = deserialize_mapping(dc.serialize())

    assert m.tp_size == dc.tp_size
    assert m.pp_size == dc.pp_size
    assert m.moe_tp_size == dc.moe_tp_size
    assert m.moe_ep_size == dc.moe_ep_size
    assert m.moe_cluster_size == dc.moe_cluster_size
    assert m.world_size == dc.world_size
    assert m.rank == dc.rank
    assert m.enable_attention_dp == dc.enable_attention_dp
