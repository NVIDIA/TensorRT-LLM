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


class ParallelVaeAttentionBlock(torch.nn.Module):
    def __init__(self, module, chunk_dim):
        super().__init__()
        self.module = module
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.chunk_dim = chunk_dim

    def forward(self, hidden_states):
        gathered_tensors = [torch.zeros_like(hidden_states) for _ in range(self.world_size)]
        hidden_states = hidden_states.contiguous()
        dist.all_gather(gathered_tensors, hidden_states)
        combined_tensor = torch.cat(gathered_tensors, dim=self.chunk_dim)

        forward_output = self.module(combined_tensor)

        chunk_sizes = [t.size(self.chunk_dim) for t in gathered_tensors]

        start_idx = sum(chunk_sizes[: self.rank])
        end_idx = start_idx + chunk_sizes[self.rank]

        local_output = torch.narrow(forward_output, self.chunk_dim, start_idx, end_idx - start_idx)

        return local_output
