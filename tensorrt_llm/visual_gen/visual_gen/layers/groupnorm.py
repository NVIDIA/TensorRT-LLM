# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import torch
import torch.distributed as dist


class GroupNormParallel(torch.nn.Module):
    def __init__(self, module, chunk_dim):
        super().__init__()
        self.module = module
        self.chunk_dim = chunk_dim

    def forward(self, hidden_states):
        shape = hidden_states.shape
        N, C, G = shape[0], shape[1], self.module.num_groups
        assert C % G == 0
        hidden_states = hidden_states.reshape(N, G, -1)

        mean = hidden_states.mean(-1, keepdim=True).to(torch.float32)
        dist.all_reduce(mean)

        mean = mean / dist.get_world_size()

        var = (
            ((hidden_states - mean.to(hidden_states.dtype)) ** 2)
            .mean(-1, keepdim=True)
            .to(torch.float32)
        )

        dist.all_reduce(var)
        var = var / dist.get_world_size()

        hidden_states = (hidden_states - mean.to(hidden_states.dtype)) / (
            var.to(hidden_states.dtype) + self.module.eps
        ).sqrt()
        hidden_states = hidden_states.view(shape)

        new_shape = [1 for _ in shape]
        new_shape[1] = -1

        return hidden_states * self.module.weight.view(new_shape) + self.module.bias.view(new_shape)
