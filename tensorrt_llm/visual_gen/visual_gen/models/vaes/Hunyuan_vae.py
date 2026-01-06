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

from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.distributed as dist
from diffusers.configuration_utils import register_to_config
from hyimage.models.vae.hunyuanimage_vae import AttnBlock, DecoderOutput, HunyuanVAE2D

from visual_gen.configs.parallel import VAEParallelConfig, init_dist
from visual_gen.layers.conv import ConvParallelStride1
from visual_gen.layers.groupnorm import GroupNormParallel
from visual_gen.layers.vae_attention import ParallelVaeAttentionBlock
from visual_gen.utils.logger import get_logger

logger = get_logger(__name__)

PRECISION_TO_TYPE = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "fp8": torch.float8_e4m3fn,
}


class ditHunyuanVAE2D(HunyuanVAE2D):
    """Implementation of parallelized HunyuanVAE2D."""

    @register_to_config
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        latent_channels: int,
        block_out_channels: Tuple[int, ...],
        layers_per_block: int,
        ffactor_spatial: int,
        sample_size: int,
        sample_tsize: int,
        scaling_factor: float = None,
        shift_factor: Optional[float] = None,
        downsample_match_channel: bool = True,
        upsample_match_channel: bool = True,
        **kwargs,
    ):

        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            latent_channels=latent_channels,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            ffactor_spatial=ffactor_spatial,
            sample_size=sample_size,
            sample_tsize=sample_tsize,
            scaling_factor=scaling_factor,
            shift_factor=shift_factor,
            downsample_match_channel=downsample_match_channel,
            upsample_match_channel=upsample_match_channel,
            **kwargs,
        )

        init_dist(device_type="cuda")
        if dist.is_initialized():
            self.adj_groups = [dist.new_group([i, i + 1]) for i in range(dist.get_world_size() - 1)]
            self.chunk_dim = 2

    def load_checkpoint(self, vae_path, vae_precision: str = None):
        ckpt_path = Path(vae_path) / "pytorch_model.ckpt"
        if not ckpt_path.exists():
            ckpt_path = Path(vae_path) / "pytorch_model.pt"

        ckpt = torch.load(ckpt_path, map_location="cpu")
        if "state_dict" in ckpt:
            ckpt = ckpt["state_dict"]
        vae_ckpt = {}
        for k, v in ckpt.items():
            if k.startswith("vae."):
                vae_ckpt[k.replace("vae.", "")] = v
        self.load_state_dict(vae_ckpt)

        if vae_precision is not None:
            self = self.to(dtype=PRECISION_TO_TYPE[vae_precision])

    def replace_module(self, model, old_module, new_module, name):
        new_module.name = name
        parent = model
        attrs = name.split(".")
        for attr in attrs[:-1]:
            parent = getattr(parent, attr)
        setattr(parent, attrs[-1], new_module)

    def replace_conv2d_block(self, model):
        conv2d_modules_to_replace = []
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Conv2d) and module.kernel_size[1] > 1:
                conv2d_modules_to_replace.append((name, module))

        for name, module in conv2d_modules_to_replace:
            new_module = ConvParallelStride1(module, self.chunk_dim, self.adj_groups)
            self.replace_module(model, module, new_module, name)

    def replace_groupnorm_block(self, model):
        groupnorm_modules_to_replace = []
        for name, module in model.named_modules():
            # ignore the groupnorm in the attn block, we will replace the whole attn block
            if isinstance(module, torch.nn.GroupNorm) and "attn" not in name:
                groupnorm_modules_to_replace.append((name, module))

        for name, module in groupnorm_modules_to_replace:
            new_module = GroupNormParallel(module, self.chunk_dim)
            self.replace_module(model, module, new_module, name)

    def replace_attn_block(self, model):
        attn_modules_to_replace = []
        for name, module in model.named_modules():
            if isinstance(module, AttnBlock):
                attn_modules_to_replace.append((name, module))

        for name, module in attn_modules_to_replace:
            new_module = ParallelVaeAttentionBlock(module, self.chunk_dim)
            self.replace_module(model, module, new_module, name)

    def parallel_decoder(self):
        self.replace_conv2d_block(self.decoder)
        self.replace_groupnorm_block(self.decoder)
        self.replace_attn_block(self.decoder)

    def parallel_vae(self, split_dim="height"):
        if not dist.is_initialized():
            logger.debug("Not in parallel environment, disable vae parallel")
            return

        logger.info(f"Parallelizing VAE with split dim: {VAEParallelConfig.parallel_vae_split_dim}")

        if split_dim == "height":
            self.chunk_dim = 2
        elif split_dim == "width":
            self.chunk_dim = 3
        else:
            raise ValueError(f"Invalid split_dim: {split_dim}")

        logger.info("only support parallel decoding for now")
        self.parallel_decoder()

    def decode(self, z: torch.Tensor, return_dict: bool = True, generator=None):
        """
        override the decode method to support parallel decoding
        """
        original_ndim = z.ndim
        if original_ndim == 5:
            z = z.squeeze(2)

        def _decode(z):
            # print("spatial tiling: ", self.use_spatial_tiling)
            if self.use_spatial_tiling and (
                z.shape[-1] > self.tile_latent_min_size or z.shape[-2] > self.tile_latent_min_size
            ):
                return self.spatial_tiled_decode(z)

            assert z.shape[self.chunk_dim] % dist.get_world_size() == 0, (
                f"z's chunk dimension {self.chunk_dim} must be divisible by world size {dist.get_world_size()}, "
                f"but got {z.shape}"
            )
            z = z.chunk(dist.get_world_size(), dim=self.chunk_dim)[dist.get_rank()]
            local_video = self.decoder(z).contiguous()
            gather_list = [torch.empty_like(local_video) for _ in range(dist.get_world_size())]
            dist.all_gather(gather_list, local_video)
            video = torch.cat(gather_list, dim=self.chunk_dim)
            return video

        if self.use_slicing and z.shape[0] > 1:
            decoded_slices = [_decode(z_slice) for z_slice in z.split(1)]
            decoded = torch.cat(decoded_slices)
        else:
            decoded = _decode(z)

        if original_ndim == 5:
            decoded = decoded.unsqueeze(2)

        if not return_dict:
            return (decoded,)

        return DecoderOutput(sample=decoded)
