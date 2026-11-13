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
"""A rank's place and the parallel dimensions recurrent state splits over, as a page table reads
them."""

from types import SimpleNamespace

import pytest

from tensorrt_llm._torch.disaggregation.base.role_layout import RankTopology, grid_place
from tensorrt_llm._torch.disaggregation.native.mixers.ssm.peer import MambaPolicy
from tensorrt_llm._torch.pyexecutor.kv_cache.page_table import rank_topology, state_split

pytestmark = pytest.mark.cpu_only


@pytest.mark.parametrize("attention_dp", [False, True])
def test_rank_topology_reads_the_rank_s_place_from_its_mapping(attention_dp):
    mapping = SimpleNamespace(
        tp_size=4,
        tp_rank=3,
        pp_size=2,
        pp_rank=1,
        cp_size=2,
        cp_rank=1,
        dp_size=1,
        enable_attention_dp=attention_dp,
    )

    topology = rank_topology(mapping)

    assert (topology.tp_size, topology.tp_rank, topology.pp_size, topology.pp_rank) == (4, 3, 2, 1)
    assert (topology.cp_size, topology.cp_rank, topology.attention_dp) == (2, 1, attention_dp)
    # Under attention data parallelism each TP rank holds its buffers whole.
    assert grid_place(topology, ("tp",)) == ((1, 0) if attention_dp else (4, 3))
    assert rank_topology(None) == RankTopology()


@pytest.mark.parametrize("attention_dp", [False, True])
@pytest.mark.parametrize("cp_size", [1, 2])
@pytest.mark.parametrize("tp_size", [1, 2, 4])
def test_recurrent_state_splits_as_the_native_transfer_shards_it(tp_size, cp_size, attention_dp):
    for tp_rank in range(tp_size):
        for cp_rank in range(cp_size):
            mapping = SimpleNamespace(
                tp_size=tp_size,
                tp_rank=tp_rank,
                cp_size=cp_size,
                cp_rank=cp_rank,
                enable_attention_dp=attention_dp,
            )
            manager = SimpleNamespace(mapping=mapping, _has_cp_helix=cp_size > 1)
            rank_info = SimpleNamespace(
                tp_size=tp_size,
                tp_rank=tp_rank,
                cp_size=cp_size,
                cp_rank=cp_rank,
                attention=SimpleNamespace(enable_attention_dp=attention_dp),
            )
            topology = rank_topology(mapping)

            split = state_split(manager)

            assert grid_place(topology, split) == MambaPolicy._mamba_tp(rank_info), topology
