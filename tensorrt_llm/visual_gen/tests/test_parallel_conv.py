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
from diffusers.models.autoencoders.autoencoder_kl_wan import WanCausalConv3d
from visual_gen.layers.conv import ConvParallelStride1, ConvParallelStride2


def prepare_data(rank, world_size, chunk_dim, x_shape, cache_x_shape=None):
    local_device = "cuda:" + str(rank)

    x = torch.randn(x_shape, dtype=torch.float32, device=local_device, requires_grad=False)
    dist.broadcast(x, src=0)
    local_x = torch.chunk(x, dist.get_world_size(), dim=chunk_dim)[rank]
    if cache_x_shape is not None:
        cache_x = torch.randn(
            cache_x_shape, dtype=torch.float32, device=local_device, requires_grad=False
        )
        dist.broadcast(cache_x, src=0)
        local_cache = torch.chunk(cache_x, dist.get_world_size(), dim=chunk_dim)[rank]
    else:
        cache_x = None
        local_cache = None

    return x, cache_x, local_x, local_cache


def prepare_model_conv3d(rank):
    local_device = "cuda:" + str(rank)

    wan_conv_3d = (
        WanCausalConv3d(96, 96, kernel_size=3, stride=1, padding=1)
        .to(local_device)
        .to(torch.float32)
    )
    conv_weight = wan_conv_3d.weight
    conv_bias = wan_conv_3d.bias

    dist.broadcast(conv_weight, src=0)
    dist.broadcast(conv_bias, src=0)

    wan_conv_3d.weight = conv_weight
    wan_conv_3d.bias = conv_bias

    return wan_conv_3d


def prepare_model_conv2d(rank):
    local_device = "cuda:" + str(rank)
    conv_2d = (
        torch.nn.Conv2d(96, 96, kernel_size=3, stride=1, padding=1)
        .to(local_device)
        .to(torch.float32)
    )

    conv_weight = conv_2d.weight
    conv_bias = conv_2d.bias

    dist.broadcast(conv_weight, src=0)
    dist.broadcast(conv_bias, src=0)

    conv_2d.weight = conv_weight
    conv_2d.bias = conv_bias

    return conv_2d


def prepare_model_conv2d_stride2(rank):
    local_device = "cuda:" + str(rank)
    conv_module = (
        torch.nn.Conv2d(96, 96, kernel_size=3, stride=2).to(local_device).to(torch.float32)
    )

    conv_weight = conv_module.weight
    conv_bias = conv_module.bias

    dist.broadcast(conv_weight, src=0)
    dist.broadcast(conv_bias, src=0)

    conv_module.weight = conv_weight
    conv_module.bias = conv_bias

    return conv_module


def test_wan_conv3d_parallel(rank, world_size, chunk_dim, adj_groups):
    x, cache_x, local_x, local_cache = prepare_data(
        rank, world_size, chunk_dim, [1, 96, 4, 832, 480], [1, 96, 2, 832, 480]
    )
    wan_conv_3d = prepare_model_conv3d(rank)
    ref_output = wan_conv_3d(x, cache_x).detach()

    wan_conv_3d_parallel = ConvParallelStride1(wan_conv_3d, chunk_dim, adj_groups)
    local_output = wan_conv_3d_parallel(local_x, local_cache).contiguous()

    gather_list = [torch.empty_like(local_output) for _ in range(dist.get_world_size())]
    dist.all_gather(gather_list, local_output)
    output = torch.cat(gather_list, dim=chunk_dim).detach()

    max_diff = torch.max(torch.abs(output - ref_output))
    max_output = torch.max(torch.abs(output))
    max_ref_output = torch.max(torch.abs(ref_output))

    assert max_diff < 0.01, f"max_diff: {max_diff} but expected less than 0.01"
    if rank == 0:
        print(f"max_diff: {max_diff}, max_output: {max_output}, max_ref_output: {max_ref_output}")


def test_conv2d_parallel(rank, world_size, chunk_dim, adj_groups):
    x, cache_x, local_x, local_cache = prepare_data(rank, world_size, chunk_dim, [1, 96, 832, 480])
    conv_2d = prepare_model_conv2d(rank)
    ref_output = conv_2d(x).detach()

    conv_2d_parallel = ConvParallelStride1(conv_2d, chunk_dim, adj_groups)
    local_output = conv_2d_parallel.forward(local_x, local_cache).contiguous()

    gather_list = [torch.empty_like(local_output) for _ in range(dist.get_world_size())]
    dist.all_gather(gather_list, local_output)
    output = torch.cat(gather_list, dim=chunk_dim).detach()

    max_diff = torch.max(torch.abs(output - ref_output))
    max_output = torch.max(torch.abs(output))
    max_ref_output = torch.max(torch.abs(ref_output))

    assert max_diff < 0.01, f"max_diff: {max_diff} but expected less than 0.01"
    if rank == 0:
        print(f"max_diff: {max_diff}, max_output: {max_output}, max_ref_output: {max_ref_output}")


def test_conv2d_parallel_stride2(rank, world_size, chunk_dim):
    conv_module = prepare_model_conv2d_stride2(rank)
    conv_parallel = ConvParallelStride2(conv_module, chunk_dim, pad_before_conv=(0, 1, 0, 1))
    x, _, local_x, _ = prepare_data(rank, world_size, chunk_dim, (4, 96, 832, 480))

    pad_module = torch.nn.ZeroPad2d((0, 1, 0, 1))
    ref_output = conv_module(pad_module(x))
    local_output = conv_parallel(local_x)

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
    adj_groups = [dist.new_group([i, i + 1]) for i in range(dist.get_world_size() - 1)]

    test_wan_conv3d_parallel(rank, world_size, chunk_dim=3, adj_groups=adj_groups)
    test_wan_conv3d_parallel(rank, world_size, chunk_dim=4, adj_groups=adj_groups)
    test_conv2d_parallel(rank, world_size, chunk_dim=2, adj_groups=adj_groups)
    test_conv2d_parallel(rank, world_size, chunk_dim=3, adj_groups=adj_groups)
    test_conv2d_parallel_stride2(rank, world_size, chunk_dim=2)
    test_conv2d_parallel_stride2(rank, world_size, chunk_dim=3)

    dist.destroy_process_group()
