# SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from typing import List


class Mapping(object):
    '''
    A node with 8 GPUs, tp_size = 4, cp_size = 1, pp_size = 2

    2 tp groups:

    - [0, 1, 2, 3]
    - [4, 5, 6, 7]

    4 pp groups:

    - [0, 4]
    - [1, 5]
    - [2, 6]
    - [3, 7]

    A node with 8 GPUs, tp_size = 4, cp_size = 2, pp_size = 1

    2 tp groups:

    - [0, 1, 2, 3]
    - [4, 5, 6, 7]

    4 cp groups:

    - [0, 4]
    - [1, 5]
    - [2, 6]
    - [3, 7]

    A node with 8 GPUs, moe_tp_size = 2, moe_ep_size = 4

    4 moe_tp groups:

    - [0, 4]
    - [1, 5]
    - [2, 6]
    - [3, 7]

    2 moe_ep groups:

    - [0, 1, 2, 3]
    - [4, 5, 6, 7]

    2 nodes with 16 GPUs, moe_tp_size = 2, moe_ep_size = 4, pp_size = 2

    8 moe_tp groups:

    - [0 4]
    - [1 5]
    - [2 6]
    - [3 7]
    - [8 12]
    - [9 13]
    - [10 14]
    - [11 15]

    4 moe_ep groups:

    - [0, 1, 2, 3]
    - [4, 5, 6, 7]
    - [8, 9, 10, 11]
    - [12, 13, 14, 15]

    8 pp groups:

    - [0 8]
    - [1 9]
    - [2 10]
    - [3 11]
    - [4 12]
    - [5 13]
    - [6 14]
    - [7 15]

    2 nodes with 8 GPUs, tp_size 2, pp_size 2, cp_size 2

    4 tp groups:
    - [0, 1]
    - [2, 3]
    - [4, 5]
    - [6, 7]

    4 pp groups:
    - [0, 4]
    - [1, 5]
    - [2, 6]
    - [3, 7]

    4 cp groups:
    - [0, 2]
    - [1, 3]
    - [4, 6]
    - [5, 7]
    '''

    def __init__(
            self,
            world_size=1,
            rank=0,
            gpus_per_node=8,
            *,
            cp_size=1,
            cp_config=None,
            tp_size=1,
            pp_size=1,
            moe_tp_size=-1,  # -1 means no moe
            moe_ep_size=-1,  # -1 means no moe
            auto_parallel=False,
            enable_attention_dp=False):
        # set default values for non-moe cases
        # or where only one MOE parallelism size is specified
        if moe_tp_size == -1 and moe_ep_size == -1:
            moe_tp_size = tp_size
            moe_ep_size = 1

        elif moe_tp_size == -1:
            moe_tp_size = tp_size // moe_ep_size

        elif moe_ep_size == -1:
            moe_ep_size = tp_size // moe_tp_size

        if auto_parallel:
            if tp_size != 1 or pp_size != 1 or tp_size != 1:
                raise ValueError(
                    f"When auto parallel is enabled, tp_size, pp_size, cp_size must be 1, but got {tp_size}, {pp_size}, {cp_size}."
                )
        else:
            if tp_size * pp_size * cp_size != world_size:
                raise ValueError(
                    f"world_size must equal to tp_size * pp_size * cp_size, but got {world_size} != {tp_size} * {pp_size} * {cp_size}."
                )

        moe_tp_ep_size = moe_tp_size * moe_ep_size
        if moe_tp_ep_size != tp_size:
            raise ValueError(
                f"tp_size must equal to moe_tp_size * moe_ep_size, but got {tp_size} != {moe_tp_size} * {moe_ep_size}"
            )

        if moe_ep_size != 1 and cp_size > 1:
            raise NotImplementedError("CP don't support MoE tp/ep yet")

        self.tp_size = tp_size
        self.cp_size = cp_size
        self.cp_config = cp_config if cp_config is not None else {}
        self.pp_size = pp_size
        self.moe_tp_size = moe_tp_size
        self.moe_ep_size = moe_ep_size
        self.auto_parallel = auto_parallel
        self.world_size = world_size
        self.rank = rank
        self.gpus_per_node = gpus_per_node
        self.enable_attention_dp = enable_attention_dp
        self.pp_groups = []
        self.cp_groups = []
        self.tp_groups = []
        self.moe_tp_groups = []
        self.moe_ep_groups = []

        # init pp group
        for i in range(tp_size * cp_size):
            ranks = range(i, world_size, tp_size * cp_size)
            self.pp_groups.append(list(ranks))

        # init cp group
        for i in range(pp_size):
            for j in range(tp_size):
                ranks = range(i * tp_size * cp_size + j,
                              (i + 1) * tp_size * cp_size + j, tp_size)
                self.cp_groups.append(list(ranks))

        # init tp group
        for i in range(pp_size):
            for j in range(cp_size):
                ranks = range(i * tp_size * cp_size + j * tp_size,
                              i * tp_size * cp_size + (j + 1) * tp_size)
                self.tp_groups.append(list(ranks))

        # init moe tp group
        for i in range(pp_size):
            for j in range(moe_ep_size):
                ranks = range(i * moe_tp_ep_size + j, (i + 1) * moe_tp_ep_size,
                              moe_ep_size)
                self.moe_tp_groups.append(list(ranks))

        # init moe ep group
        for i in range(pp_size):
            for j in range(moe_tp_size):
                ranks = range(i * moe_tp_ep_size + j * moe_ep_size,
                              i * moe_tp_ep_size + (j + 1) * moe_ep_size)
                self.moe_ep_groups.append(list(ranks))

    def __eq__(self, other):
        if not isinstance(other, Mapping):
            return NotImplemented

        return (self.world_size == other.world_size and self.rank == other.rank
                and self.gpus_per_node == other.gpus_per_node
                and self.cp_size == other.cp_size
                and self.tp_size == other.tp_size
                and self.pp_size == other.pp_size
                and self.moe_tp_size == other.moe_tp_size
                and self.moe_ep_size == other.moe_ep_size
                and self.auto_parallel == other.auto_parallel)

    def __hash__(self):
        return (hash(self.world_size) ^ hash(self.rank)
                ^ hash(self.gpus_per_node) ^ hash(self.cp_size)
                ^ hash(self.tp_size) ^ hash(self.pp_size)
                ^ hash(self.moe_tp_size) ^ hash(self.moe_ep_size)
                ^ hash(self.auto_parallel))

    @property
    def rank(self):
        return self._rank

    @rank.setter
    def rank(self, rank: int):
        if not isinstance(rank, int) or rank < 0 or rank >= self.world_size:
            raise ValueError(
                f"Rank should be an integer between 0 and {self.world_size-1}, but got {rank}."
            )
        self._rank = rank

    @property
    def tp_rank(self):
        return 0 if self.auto_parallel else self.rank % self.tp_size

    @property
    def pp_rank(self):
        return 0 if self.auto_parallel else self.rank // (self.tp_size *
                                                          self.cp_size)

    @property
    def cp_rank(self):
        return 0 if self.auto_parallel else self.rank % (
            self.tp_size * self.cp_size) // self.tp_size

    @property
    def moe_tp_rank(self):
        return self.tp_rank // self.moe_ep_size

    @property
    def moe_ep_rank(self):
        return self.tp_rank % self.moe_ep_size

    @property
    def tp_group(self):
        return self.tp_groups[self.pp_rank * self.cp_size + self.cp_rank]

    @property
    def pp_group(self):
        return self.pp_groups[self.cp_rank * self.tp_size + self.tp_rank]

    @property
    def cp_group(self):
        return self.cp_groups[self.pp_rank * self.tp_size + self.tp_rank]

    @property
    def moe_tp_group(self):
        return self.moe_tp_groups[self.pp_rank * self.moe_ep_size +
                                  self.moe_ep_rank]

    @property
    def moe_ep_group(self):
        return self.moe_ep_groups[self.pp_rank * self.moe_tp_size +
                                  self.moe_tp_rank]

    @property
    def node_rank(self):
        return self.rank // self.gpus_per_node

    @property
    def local_rank(self):
        return self.rank % self.gpus_per_node

    def has_cp(self):
        return self.cp_size > 1

    def get_node_rank(self, rank: int):
        return rank // self.gpus_per_node

    def get_local_rank(self, rank: int):
        return rank % self.gpus_per_node

    def has_tp(self):
        return self.tp_size > 1

    def is_last_pp_rank(self):
        return self.pp_rank == self.pp_size - 1

    def is_first_pp_rank(self):
        return self.pp_rank == 0

    def has_pp(self):
        return self.pp_size > 1

    def prev_pp_rank(self):
        p = self.rank - self.tp_size * self.cp_size
        if p < 0:
            p = p + self.world_size
        return p

    def next_pp_rank(self):
        p = self.rank + self.tp_size * self.cp_size
        if p >= self.world_size:
            p = p - self.world_size
        return p

    def has_moe_tp(self):
        return self.moe_tp_size > 1

    def has_moe_ep(self):
        return self.moe_ep_size > 1

    def pp_layers(self, num_layers: int) -> List[int]:
        layers_per_pipeline_stage = num_layers // self.pp_size
        layers_range = range(self.pp_rank * layers_per_pipeline_stage,
                             (self.pp_rank + 1) * layers_per_pipeline_stage)
        return list(layers_range)

    def ep_experts(self, num_experts: int) -> List[int]:
        assert self.cp_size == 1
        experts_per_rank = num_experts // self.moe_ep_size
        experts_range = range(self.moe_ep_rank * experts_per_rank,
                              (self.moe_ep_rank + 1) * experts_per_rank)
        return list(experts_range)

    @classmethod
    def from_dict(cls, mapping: dict):
        return cls(**mapping)

    def to_dict(self):
        return {
            'world_size': self.world_size,
            'rank': self.rank,
            'gpus_per_node': self.gpus_per_node,
            'cp_size': self.cp_size,
            'tp_size': self.tp_size,
            'pp_size': self.pp_size,
            'moe_tp_size': self.moe_tp_size,
            'moe_ep_size': self.moe_ep_size,
            'auto_parallel': self.auto_parallel,
        }
