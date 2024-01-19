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
    A node with 8 GPUs, tp_size = 4, pp_size = 2

    2 tp groups:

    - [0, 1, 2, 3]
    - [4, 5, 6, 7]

    4 pp groups:

    - [0, 4]
    - [1, 5]
    - [2, 6]
    - [3, 7]
    '''

    def __init__(self,
                 world_size=1,
                 rank=0,
                 gpus_per_node=8,
                 tp_size=1,
                 pp_size=1):
        self.tp_size = tp_size
        self.pp_size = pp_size
        self.world_size = world_size
        self.rank = rank
        self.gpus_per_node = gpus_per_node

        if pp_size * tp_size != world_size:
            raise ValueError("world_size must equal to pp_size * tp_size")
        self.pp_groups = []
        self.tp_groups = []

        # init pp group
        for i in range(tp_size):
            ranks = range(i, world_size, tp_size)
            self.pp_groups.append(list(ranks))

        # init tp group
        for i in range(pp_size):
            ranks = range(i * tp_size, (i + 1) * tp_size)
            self.tp_groups.append(list(ranks))

        self.pp_rank = self.rank // self.tp_size
        self.tp_rank = self.rank % self.tp_size

        self.tp_group = self.tp_groups[self.pp_rank]
        self.pp_group = self.pp_groups[self.tp_rank]

    def has_tp(self):
        return self.tp_size > 1

    def is_last_pp_rank(self):
        return self.pp_rank == self.pp_size - 1

    def is_first_pp_rank(self):
        return self.pp_rank == 0

    def has_pp(self):
        return self.pp_size > 1

    def prev_pp_rank(self):
        p = self.rank - self.tp_size
        if p < 0:
            p = p + self.world_size
        return p

    def next_pp_rank(self):
        p = self.rank + self.tp_size
        if p >= self.world_size:
            p = p - self.world_size
        return p

    def pp_layers(self, num_layers: int) -> List[int]:
        layers_per_pipeline_stage = num_layers // self.pp_size
        layers_range = range(self.pp_rank * layers_per_pipeline_stage,
                             (self.pp_rank + 1) * layers_per_pipeline_stage)
        return list(layers_range)

    def ep_experts(self, num_experts: int) -> List[int]:
        experts_per_rank = num_experts // self.tp_size
        experts_range = range(self.tp_rank * experts_per_rank,
                              (self.tp_rank + 1) * experts_per_rank)
        return list(experts_range)
