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
    '''

    def __init__(
            self,
            world_size=1,
            rank=0,
            gpus_per_node=8,
            cp_size=1,
            tp_size=1,
            pp_size=1,
            moe_tp_size=-1,  # -1 means no moe
            moe_ep_size=-1):  # -1 means no moe
        # set default values for non-moe cases
        if moe_tp_size == -1:
            moe_tp_size = tp_size
            moe_ep_size = 1

        if pp_size * cp_size * tp_size != world_size:
            raise ValueError(
                f"world_size must equal to pp_size * cp_size * tp_size, but got {world_size} != {pp_size} * {cp_size} * {tp_size}"
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
        self.pp_size = pp_size
        self.moe_tp_size = moe_tp_size
        self.moe_ep_size = moe_ep_size
        self.world_size = world_size
        self.rank = rank
        self.gpus_per_node = gpus_per_node

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

        self.pp_rank = self.rank // (self.tp_size * self.cp_size)
        self.cp_rank = self.rank % (self.tp_size * self.cp_size) // self.tp_size
        self.tp_rank = self.rank % (self.tp_size * self.cp_size) % self.tp_size
        self.moe_tp_rank = self.tp_rank // self.moe_ep_size
        self.moe_ep_rank = self.tp_rank % self.moe_ep_size

        self.tp_group = self.tp_groups[self.pp_rank * self.cp_size +
                                       self.cp_rank]
        self.cp_group = self.cp_groups[self.pp_rank * self.tp_size +
                                       self.tp_rank]
        self.pp_group = self.pp_groups[self.cp_rank * self.tp_size +
                                       self.tp_rank]
        self.moe_tp_group = self.moe_tp_groups[self.pp_rank * moe_ep_size +
                                               self.moe_ep_rank]
        self.moe_ep_group = self.moe_ep_groups[self.pp_rank * moe_tp_size +
                                               self.moe_tp_rank]

        self.node_rank = self.rank // self.gpus_per_node
        self.local_rank = self.rank % self.gpus_per_node

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
            'moe_ep_size': self.moe_ep_size
        }
