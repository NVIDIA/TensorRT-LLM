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
from visual_gen.layers.groupnorm import GroupNormParallel


def prepare_data(rank, world_size, chunk_dim, x_shape):
    local_device = "cuda:" + str(rank)

    x = torch.randn(x_shape, dtype=torch.float32, device=local_device, requires_grad=False)
    dist.broadcast(x, src=0)
    local_x = torch.chunk(x, dist.get_world_size(), dim=chunk_dim)[rank]

    return x, local_x


def test_groupnorm_parallel(rank, world_size, chunk_dim):
    local_device = "cuda:" + str(rank)
    x, local_x = prepare_data(rank, world_size, chunk_dim, [1, 1024, 512, 512])
    groupnorm = torch.nn.GroupNorm(num_groups=32, num_channels=1024, eps=1e-6, affine=True)
    parallel_groupnorm = GroupNormParallel(groupnorm, chunk_dim=chunk_dim).to(local_device)
    ref_output = groupnorm(x)
    local_output = parallel_groupnorm(local_x)

    gather_list = [torch.empty_like(local_output) for _ in range(dist.get_world_size())]
    dist.all_gather(gather_list, local_output)
    output = torch.cat(gather_list, dim=chunk_dim).detach()

    max_diff = torch.max(torch.abs(output - ref_output))
    max_output = torch.max(torch.abs(output))
    max_ref_output = torch.max(torch.abs(ref_output))

    assert max_diff < 0.01, f"max_diff: {max_diff} but expected less than 0.01"
    if rank == 0:
        print(f"max_diff: {max_diff}, max_output: {max_output}, max_ref_output: {max_ref_output}")


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    test_groupnorm_parallel(rank, world_size, chunk_dim=2)

    dist.destroy_process_group()
