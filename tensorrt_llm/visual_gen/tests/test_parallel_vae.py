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
from diffusers import AutoencoderKLWan
from visual_gen.models.vaes.wan_vae import ditWanAutoencoderKL


def prepare_data(rank, world_size, chunk_dim, x_shape, cache_x_shape=None):
    local_device = "cuda:" + str(rank)
    x = torch.randn(x_shape, dtype=torch.float32, device=local_device, requires_grad=False)
    dist.broadcast(x, src=0)
    local_x = torch.chunk(x, dist.get_world_size(), dim=chunk_dim)[rank]

    return x, local_x


def test_parallel_vae(rank, world_size, split_dim, vae, device):
    if split_dim == "height":
        chunk_dim = 3
    elif split_dim == "width":
        chunk_dim = 4
    else:
        raise ValueError(f"Invalid split_dim: {split_dim}")

    visual_gen_vae = ditWanAutoencoderKL.from_pretrained(
        model_id, subfolder="vae", torch_dtype=torch.float32
    )
    visual_gen_vae.parallel_vae(split_dim=split_dim)
    visual_gen_vae.to(device)

    encoder_x, local_encoder_x = prepare_data(rank, world_size, chunk_dim, (1, 3, 65, 832, 640))
    encode_output_ref = vae.encode(encoder_x).latent_dist.mode()
    encode_output = visual_gen_vae.encode(encoder_x).latent_dist.mode()

    max_diff = torch.max(torch.abs(encode_output - encode_output_ref))
    max_output = torch.max(torch.abs(encode_output))
    max_output_ref = torch.max(torch.abs(encode_output_ref))

    assert max_diff < 0.01, f"max_diff: {max_diff} but expected less than 0.01"
    if rank == 0:
        print(f"max_diff: {max_diff}, max_output: {max_output}, max_ref_output: {max_output_ref}")

    decoder_x, local_decoder_x = prepare_data(rank, world_size, chunk_dim, (1, 16, 17, 104, 80))
    decode_output_ref = vae.decode(decoder_x, return_dict=False)[0]
    decode_output = visual_gen_vae.decode(decoder_x, return_dict=False)[0]

    max_diff = torch.max(torch.abs(decode_output - decode_output_ref))
    max_output = torch.max(torch.abs(decode_output))
    max_output_ref = torch.max(torch.abs(decode_output_ref))

    assert max_diff < 0.01, f"max_diff: {max_diff} but expected less than 0.01"
    if rank == 0:
        print(f"max_diff: {max_diff}, max_output: {max_output}, max_output_ref: {max_output_ref}")


if __name__ == "__main__":
    torch.set_grad_enabled(False)

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = "cuda:" + str(rank)
    adj_groups = [dist.new_group([i, i + 1]) for i in range(dist.get_world_size() - 1)]

    model_id = "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers"
    vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32).to(
        device
    )

    test_parallel_vae(rank, world_size, "height", vae, device)
    test_parallel_vae(rank, world_size, "width", vae, device)

    dist.destroy_process_group()
