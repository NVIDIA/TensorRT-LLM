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

try:
    from hyimage.models.vae import load_vae
except ImportError:
    print("HyImage is not found, skipping test")
    exit(0)

import torch
import torch.distributed as dist
from huggingface_hub import hf_hub_download

from visual_gen.models.vaes.Hunyuan_vae import ditHunyuanVAE2D


def prepare_data(rank, world_size, chunk_dim, x_shape, cache_x_shape=None):

    local_device = "cuda:" + str(rank)
    x = torch.randn(x_shape, dtype=torch.float32, device=local_device, requires_grad=False)
    dist.broadcast(x, src=0)
    local_x = torch.chunk(x, dist.get_world_size(), dim=chunk_dim)[rank]

    return x, local_x


def test_parallel_vae(rank, world_size, split_dim, vae, device, vae_path):
    if split_dim == "height":
        chunk_dim = 2
    elif split_dim == "width":
        chunk_dim = 3
    else:
        raise ValueError(f"Invalid split_dim: {split_dim}")

    config = ditHunyuanVAE2D.load_config(vae_path)
    visual_gen_hunyuan_vae = ditHunyuanVAE2D.from_config(config)
    visual_gen_hunyuan_vae.load_checkpoint(vae_path)
    visual_gen_hunyuan_vae.parallel_vae(split_dim=split_dim)
    visual_gen_hunyuan_vae.to(device)

    decoder_x, local_decoder_x = prepare_data(rank, world_size, chunk_dim, (1, 64, 64, 64))
    decode_output_ref = vae.decode(decoder_x, return_dict=False)[0]
    decode_output = visual_gen_hunyuan_vae.decode(decoder_x, return_dict=False)[0]

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

    if rank == 0:
        hf_hub_download(
            repo_id="tencent/HunyuanImage-2.1",
            filename="vae/vae_2_1/pytorch_model.ckpt",
            local_dir="./",
        )
        hf_hub_download(
            repo_id="tencent/HunyuanImage-2.1",
            filename="vae/vae_2_1/config.json",
            local_dir="./",
        )

    dist.barrier()

    vae_path = "vae/vae_2_1/"
    vae = load_vae(device, vae_path)

    test_parallel_vae(rank, world_size, "height", vae, device, vae_path)
    test_parallel_vae(rank, world_size, "width", vae, device, vae_path)

    dist.destroy_process_group()
