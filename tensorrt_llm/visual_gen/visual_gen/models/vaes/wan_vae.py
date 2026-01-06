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

from typing import List, Tuple

import torch
import torch.distributed as dist
from diffusers import AutoencoderKLWan
from diffusers.configuration_utils import register_to_config
from diffusers.models.autoencoders.autoencoder_kl_wan import WanAttentionBlock, WanCausalConv3d
from diffusers.models.autoencoders.vae import DecoderOutput

from visual_gen.configs.parallel import VAEParallelConfig, init_dist
from visual_gen.layers.conv import ConvParallelStride1, ConvParallelStride2
from visual_gen.layers.vae_attention import ParallelVaeAttentionBlock
from visual_gen.utils.logger import get_logger

logger = get_logger(__name__)


class ditWanAutoencoderKL(AutoencoderKLWan):
    """Implementation of parallelized WanAutoencoderKL."""

    @register_to_config
    def __init__(
        self,
        base_dim: int = 96,
        z_dim: int = 16,
        dim_mult: Tuple[int] = [1, 2, 4, 4],
        num_res_blocks: int = 2,
        attn_scales: List[float] = [],
        temperal_downsample: List[bool] = [False, True, True],
        dropout: float = 0.0,
        latents_mean: List[float] = [
            -0.7571,
            -0.7089,
            -0.9113,
            0.1075,
            -0.1745,
            0.9653,
            -0.1517,
            1.5508,
            0.4134,
            -0.0715,
            0.5517,
            -0.3632,
            -0.1922,
            -0.9497,
            0.2503,
            -0.2921,
        ],
        latents_std: List[float] = [
            2.8184,
            1.4541,
            2.3275,
            2.6558,
            1.2196,
            1.7708,
            2.6052,
            2.0743,
            3.2687,
            2.1526,
            2.8652,
            1.5579,
            1.6382,
            1.1253,
            2.8251,
            1.9160,
        ],
    ) -> None:

        super().__init__(
            base_dim=base_dim,
            z_dim=z_dim,
            dim_mult=dim_mult,
            num_res_blocks=num_res_blocks,
            attn_scales=attn_scales,
            temperal_downsample=temperal_downsample,
            dropout=dropout,
            latents_mean=latents_mean,
            latents_std=latents_std,
        )

        init_dist(device_type="cuda")
        if dist.is_initialized():
            self.adj_groups = [dist.new_group([i, i + 1]) for i in range(dist.get_world_size() - 1)]
            self.input_chunk_dim = 3
            self.conv3d_chunk_dim = 3
            self.conv2d_chunk_dim = 2
            self.transformer_chunk_dim = 3

    def replace_module(self, model, old_module, new_module, name):
        new_module.name = name
        parent = model
        attrs = name.split(".")
        for attr in attrs[:-1]:
            parent = getattr(parent, attr)
        setattr(parent, attrs[-1], new_module)

    def replace_conv3d(self, model):
        conv3d_modules_to_replace = []
        for name, module in model.named_modules():
            if isinstance(module, WanCausalConv3d) and module.kernel_size[1] > 1:
                conv3d_modules_to_replace.append((name, module))

        for name, module in conv3d_modules_to_replace:
            new_module = ConvParallelStride1(module, self.conv3d_chunk_dim, self.adj_groups)
            self.replace_module(model, module, new_module, name)

    def replace_transformer(self, model):
        transformer_modules_to_replace = []
        for name, module in model.named_modules():
            if isinstance(module, WanAttentionBlock):
                transformer_modules_to_replace.append((name, module))

        for name, module in transformer_modules_to_replace:
            new_module = ParallelVaeAttentionBlock(module, self.transformer_chunk_dim)
            self.replace_module(model, module, new_module, name)

    def replace_conv2d(self, model):
        conv2d_modules_to_replace = []
        for name, module in model.named_modules():
            if ".resample" in name and isinstance(module, torch.nn.Conv2d):
                conv2d_modules_to_replace.append((name, module))

        for name, module in conv2d_modules_to_replace:
            new_module = ConvParallelStride1(module, self.conv2d_chunk_dim, self.adj_groups)
            self.replace_module(model, module, new_module, name)

    def replace_conv2d_stride2(self, model):
        conv2d_modules_to_replace = []
        for name, module in model.named_modules():
            if ".resample" in name and isinstance(module, torch.nn.Sequential) and len(module) == 2:
                if isinstance(module[0], torch.nn.ZeroPad2d) and isinstance(module[1], torch.nn.Conv2d):
                    conv2d_modules_to_replace.append((name, module))
        for name, module in conv2d_modules_to_replace:
            new_module = ConvParallelStride2(module[1], self.conv2d_chunk_dim, pad_before_conv=module[0].padding)
            self.replace_module(model, module, new_module, name)

    def parallel_decoder(self):
        self.replace_conv3d(self.decoder)
        self.replace_transformer(self.decoder)
        self.replace_conv2d(self.decoder)

    def parallel_encoder(self):
        self.replace_conv3d(self.encoder)
        self.replace_transformer(self.encoder)
        self.replace_conv2d_stride2(self.encoder)

    def parallel_vae(self, split_dim="height"):
        if not dist.is_initialized():
            logger.debug("Not in parallel environment, disable vae parallel")
            return

        logger.info(f"Parallelizing VAE with split dim: {VAEParallelConfig.parallel_vae_split_dim}")

        if split_dim == "height":
            self.input_chunk_dim = 3
            self.conv3d_chunk_dim = 3
            self.conv2d_chunk_dim = 2
            self.transformer_chunk_dim = 3
        elif split_dim == "width":
            self.input_chunk_dim = 4
            self.conv3d_chunk_dim = 4
            self.conv2d_chunk_dim = 3
            self.transformer_chunk_dim = 4
        else:
            raise ValueError(f"Invalid split_dim: {split_dim}")

        self.parallel_decoder()
        self.parallel_encoder()

    def _decode(self, latents, return_dict=True):
        if VAEParallelConfig.disable_parallel_vae:
            return super()._decode(latents, return_dict=return_dict)

        assert latents.shape[self.input_chunk_dim] % dist.get_world_size() == 0, (
            f"latents's chunk dimension {self.input_chunk_dim} must be divisible by world size {dist.get_world_size()}, "
            f"but got {latents.shape}"
        )
        latents = latents.chunk(dist.get_world_size(), dim=self.input_chunk_dim)[dist.get_rank()]
        local_video = super()._decode(latents, return_dict=False)[0]
        gather_list = [torch.empty_like(local_video) for _ in range(dist.get_world_size())]
        dist.all_gather(gather_list, local_video)
        video = torch.cat(gather_list, dim=self.input_chunk_dim)
        if not return_dict:
            return (video,)
        return DecoderOutput(sample=video)

    def _encode(self, video):
        if VAEParallelConfig.disable_parallel_vae:
            return super()._encode(video)

        assert video.shape[self.input_chunk_dim] % dist.get_world_size() == 0, (
            f"video's chunk dimension {self.input_chunk_dim} must be divisible by world size {dist.get_world_size()}, "
            f"but got {video.shape}"
        )
        video = video.chunk(dist.get_world_size(), dim=self.input_chunk_dim)[dist.get_rank()]
        local_latents = super()._encode(video)
        gather_list = [torch.empty_like(local_latents) for _ in range(dist.get_world_size())]
        dist.all_gather(gather_list, local_latents)
        latents = torch.cat(gather_list, dim=self.input_chunk_dim)
        return latents
