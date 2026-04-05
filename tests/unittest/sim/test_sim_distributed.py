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
"""Tests for SimDistributed."""

import pytest

from tensorrt_llm._torch.pyexecutor.sim_distributed import SimDistributed
from tensorrt_llm.mapping import Mapping


class TestSimDistributed:

    def test_tp1_defaults(self):
        mapping = Mapping(world_size=1, tp_size=1, rank=0)
        dist = SimDistributed(mapping)
        assert dist.rank == 0
        assert dist.tp_size == 1
        assert dist.pp_size == 1
        assert dist.world_size == 1
        assert dist.is_first_pp_rank is True
        assert dist.is_last_pp_rank is True

    def test_tp2_mapping(self):
        mapping = Mapping(world_size=2, tp_size=2, rank=0)
        dist = SimDistributed(mapping)
        assert dist.rank == 0
        assert dist.tp_size == 2
        assert dist.pp_size == 1
        assert dist.world_size == 2
        assert dist.has_tp is True

    def test_barrier_is_noop(self):
        mapping = Mapping(world_size=2, tp_size=2, rank=0)
        dist = SimDistributed(mapping)
        dist.barrier()

    def test_tp_barrier_is_noop(self):
        mapping = Mapping(world_size=2, tp_size=2, rank=0)
        dist = SimDistributed(mapping)
        dist.tp_barrier()

    def test_broadcast_returns_obj(self):
        mapping = Mapping(world_size=2, tp_size=2, rank=0)
        dist = SimDistributed(mapping)
        obj = {"hello": "world"}
        assert dist.broadcast(obj) is obj

    def test_allgather_wraps_in_list(self):
        mapping = Mapping(world_size=2, tp_size=2, rank=0)
        dist = SimDistributed(mapping)
        result = dist.allgather(42)
        assert result == [42]

    def test_tp_broadcast_returns_obj(self):
        mapping = Mapping(world_size=2, tp_size=2, rank=0)
        dist = SimDistributed(mapping)
        assert dist.tp_broadcast("data") == "data"

    def test_cp_broadcast_returns_obj(self):
        mapping = Mapping(world_size=2, tp_size=2, rank=0)
        dist = SimDistributed(mapping)
        assert dist.cp_broadcast("data") == "data"

    def test_tp_allgather_wraps_in_list(self):
        mapping = Mapping(world_size=2, tp_size=2, rank=0)
        dist = SimDistributed(mapping)
        assert dist.tp_allgather("x") == ["x"]

    def test_cp_allgather_wraps_in_list(self):
        mapping = Mapping(world_size=2, tp_size=2, rank=0)
        dist = SimDistributed(mapping)
        assert dist.cp_allgather("x") == ["x"]

    def test_allreduce_returns_obj(self):
        mapping = Mapping(world_size=2, tp_size=2, rank=0)
        dist = SimDistributed(mapping)
        assert dist.allreduce(99) == 99

    def test_recv_object_raises(self):
        mapping = Mapping(world_size=1, tp_size=1, rank=0)
        dist = SimDistributed(mapping)
        with pytest.raises(NotImplementedError):
            dist.recv_object(0)

    def test_send_object_raises(self):
        mapping = Mapping(world_size=1, tp_size=1, rank=0)
        dist = SimDistributed(mapping)
        with pytest.raises(NotImplementedError):
            dist.send_object("data", 1)
