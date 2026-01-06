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
from diffusers.models.autoencoders.autoencoder_kl_wan import WanAttentionBlock

from visual_gen.layers.vae_attention import ParallelVaeAttentionBlock


def prepare_data(rank, world_size, chunk_dim, x_shape, cache_x_shape=None):

    local_device = "cuda:" + str(rank)

    x = torch.randn(x_shape, dtype=torch.float32, device=local_device, requires_grad=False)
    dist.broadcast(x, src=0)
    local_x = torch.chunk(x, dist.get_world_size(), dim=chunk_dim)[rank]

    return x, local_x


def prepare_model(rank):
    local_device = "cuda:" + str(rank)
    vae_attention_block = WanAttentionBlock(dim=384).to(local_device).to(torch.float32)

    weight_to_qkv = vae_attention_block.to_qkv.weight
    weight_proj = vae_attention_block.proj.weight
    dist.broadcast(weight_to_qkv, src=0)
    dist.broadcast(weight_proj, src=0)
    vae_attention_block.to_qkv.weight = weight_to_qkv
    vae_attention_block.proj.weight = weight_proj

    bias_to_qkv = vae_attention_block.to_qkv.bias
    bias_proj = vae_attention_block.proj.bias
    dist.broadcast(bias_to_qkv, src=0)
    dist.broadcast(bias_proj, src=0)
    vae_attention_block.to_qkv.bias = bias_to_qkv
    vae_attention_block.proj.bias = bias_proj

    return vae_attention_block


def test_vae_attention_block_parallel(rank, world_size, chunk_dim):
    local_device = "cuda:" + str(rank)
    x, local_x = prepare_data(rank, world_size, chunk_dim, [1, 384, 1, 128, 96])
    vae_attention_block = prepare_model(rank)
    ref_output = vae_attention_block(x)

    parallel_vae_attention_block = ParallelVaeAttentionBlock(vae_attention_block, chunk_dim=chunk_dim).to(local_device)
    local_output = parallel_vae_attention_block(local_x).contiguous()

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

    test_vae_attention_block_parallel(rank, world_size, chunk_dim=3)
    test_vae_attention_block_parallel(rank, world_size, chunk_dim=4)

    dist.destroy_process_group()
