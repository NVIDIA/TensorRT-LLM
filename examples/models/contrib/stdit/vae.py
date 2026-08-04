# Copyright 2024 HPC-AI Technology Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# reference: https://github.com/hpcaitech/Open-Sora/blob/main/opensora/models/vae/vae.py

import os
from typing import Tuple, Union

import torch
import torch.nn.functional as F
from diffusers.models import AutoencoderKL
from einops import rearrange
from transformers import PretrainedConfig, PreTrainedModel
from utils import load_checkpoint


class VideoAutoencoderKL(torch.nn.Module):

    def __init__(
        self,
        from_pretrained=None,
        micro_batch_size=None,
        cache_dir=None,
        local_files_only=False,
        subfolder=None,
        scaling_factor=0.18215,
    ):
        super().__init__()
        self.module = AutoencoderKL.from_pretrained(
            from_pretrained,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            subfolder=subfolder,
        )
        self.out_channels = self.module.config.latent_channels
        self.patch_size = (1, 8, 8)
        self.micro_batch_size = micro_batch_size
        self.scaling_factor = scaling_factor

    def encode(self, x):
        # x: (B, C, T, H, W)
        B = x.shape[0]
        x = rearrange(x, "B C T H W -> (B T) C H W")

        if self.micro_batch_size is None:
            x = self.module.encode(x).latent_dist.sample().mul_(
                self.scaling_factor)
        else:
            # NOTE: cannot be used for training
            bs = self.micro_batch_size
            x_out = []
            for i in range(0, x.shape[0], bs):
                x_bs = x[i:i + bs]
                x_bs = self.module.encode(x_bs).latent_dist.sample().mul_(
                    self.scaling_factor)
                x_out.append(x_bs)
            x = torch.cat(x_out, dim=0)
        x = rearrange(x, "(B T) C H W -> B C T H W", B=B)
        return x

    def decode(self, x, **kwargs):
        # x: (B, C, T, H, W)
        B = x.shape[0]
        x = rearrange(x, "B C T H W -> (B T) C H W")
        if self.micro_batch_size is None:
            x = self.module.decode(x / self.scaling_factor).sample
        else:
            # NOTE: cannot be used for training
            bs = self.micro_batch_size
            x_out = []
            for i in range(0, x.shape[0], bs):
                x_bs = x[i:i + bs]
                x_bs = self.module.decode(x_bs / self.scaling_factor).sample
                x_out.append(x_bs)
            x = torch.cat(x_out, dim=0)
        x = rearrange(x, "(B T) C H W -> B C T H W", B=B)
        return x

    def get_latent_size(self, input_size):
        latent_size = []
        for i in range(3):
            # assert (
            #     input_size[i] is None or input_size[i] % self.patch_size[i] == 0
            # ), "Input size must be divisible by patch size"
            latent_size.append(
                input_size[i] //
                self.patch_size[i] if input_size[i] is not None else None)
        return latent_size

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def dtype(self):
        return next(self.parameters()).dtype


def cast_tuple(t, length=1):
    return t if isinstance(t, tuple) else ((t, ) * length)


def divisible_by(num, den):
    return (num % den) == 0


def is_odd(n):
    return not divisible_by(n, 2)


def pad_at_dim(t, pad, dim=-1):
    dims_from_right = (-dim - 1) if dim < 0 else (t.ndim - dim - 1)
    zeros = (0, 0) * dims_from_right
    return F.pad(t, (*zeros, *pad), mode="constant")


def exists(v):
    return v is not None


class DiagonalGaussianDistribution:
    """Stripped version of https://github.com/richzhang/PerceptualSimilarity/tree/master/models"""

    def __init__(
        self,
        parameters,
        deterministic=False,
    ):
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = torch.zeros_like(self.mean).to(
                device=self.parameters.device, dtype=self.mean.dtype)

    def sample(self):
        # torch.randn: standard normal distribution
        x = self.mean + self.std * torch.randn(self.mean.shape).to(
            device=self.parameters.device, dtype=self.mean.dtype)
        return x

    def kl(self, other=None):
        if self.deterministic:
            return torch.Tensor([0.0])
        else:
            if other is None:  # SCH: assumes other is a standard normal distribution
                return 0.5 * torch.sum(
                    torch.pow(self.mean, 2) + self.var - 1.0 - self.logvar,
                    dim=[1, 2, 3, 4])
            else:
                return 0.5 * torch.sum(
                    torch.pow(self.mean - other.mean, 2) / other.var +
                    self.var / other.var - 1.0 - self.logvar + other.logvar,
                    dim=[1, 2, 3, 4],
                )

    def nll(self, sample, dims=[1, 2, 3, 4]):
        if self.deterministic:
            return torch.Tensor([0.0])
        logtwopi = torch.log(torch.Tensor([2.0 * torch.pi]))
        return 0.5 * torch.sum(logtwopi + self.logvar +
                               torch.pow(sample - self.mean, 2) / self.var,
                               dim=dims)

    def mode(self):
        return self.mean


class CausalConv3d(torch.nn.Module):

    def __init__(
        self,
        chan_in,
        chan_out,
        kernel_size: Union[int, Tuple[int, int, int]],
        pad_mode="constant",
        strides=None,  # allow custom stride
        **kwargs,
    ):
        super().__init__()
        kernel_size = cast_tuple(kernel_size, 3)
        time_kernel_size, height_kernel_size, width_kernel_size = kernel_size
        assert is_odd(height_kernel_size) and is_odd(width_kernel_size)
        dilation = kwargs.pop("dilation", 1)
        stride = strides[0] if strides is not None else kwargs.pop("stride", 1)
        self.pad_mode = pad_mode
        time_pad = dilation * (time_kernel_size - 1) + (1 - stride)
        height_pad = height_kernel_size // 2
        width_pad = width_kernel_size // 2
        self.time_pad = time_pad
        self.time_causal_padding = (width_pad, width_pad, height_pad,
                                    height_pad, time_pad, 0)
        stride = strides if strides is not None else (stride, 1, 1)
        dilation = (dilation, 1, 1)
        self.conv = torch.nn.Conv3d(chan_in,
                                    chan_out,
                                    kernel_size,
                                    stride=stride,
                                    dilation=dilation,
                                    **kwargs)

    def forward(self, x):
        x = F.pad(x, self.time_causal_padding, mode=self.pad_mode)
        x = self.conv(x)
        return x


class ResBlock(torch.nn.Module):

    def __init__(
        self,
        in_channels,  # SCH: added
        filters,
        conv_fn,
        activation_fn=torch.nn.SiLU,
        use_conv_shortcut=False,
        num_groups=32,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.filters = filters
        self.activate = activation_fn()
        self.use_conv_shortcut = use_conv_shortcut

        # SCH: MAGVIT uses GroupNorm by default
        self.norm1 = torch.nn.GroupNorm(num_groups, in_channels)
        self.conv1 = conv_fn(in_channels,
                             self.filters,
                             kernel_size=(3, 3, 3),
                             bias=False)
        self.norm2 = torch.nn.GroupNorm(num_groups, self.filters)
        self.conv2 = conv_fn(self.filters,
                             self.filters,
                             kernel_size=(3, 3, 3),
                             bias=False)
        if in_channels != filters:
            if self.use_conv_shortcut:
                self.conv3 = conv_fn(in_channels,
                                     self.filters,
                                     kernel_size=(3, 3, 3),
                                     bias=False)
            else:
                self.conv3 = conv_fn(in_channels,
                                     self.filters,
                                     kernel_size=(1, 1, 1),
                                     bias=False)

    def forward(self, x):
        residual = x
        x = self.norm1(x)
        x = self.activate(x)
        x = self.conv1(x)
        x = self.norm2(x)
        x = self.activate(x)
        x = self.conv2(x)
        if self.in_channels != self.filters:  # SCH: ResBlock X->Y
            residual = self.conv3(residual)
        return x + residual


def get_activation_fn(activation):
    if activation == "relu":
        activation_fn = torch.nn.ReLU
    elif activation == "swish":
        activation_fn = torch.nn.SiLU
    else:
        raise NotImplementedError
    return activation_fn


class Encoder(torch.nn.Module):
    """Encoder Blocks."""

    def __init__(
        self,
        in_out_channels=4,
        latent_embed_dim=512,  # num channels for latent vector
        filters=128,
        num_res_blocks=4,
        channel_multipliers=(1, 2, 2, 4),
        temporal_downsample=(False, True, True),
        num_groups=32,  # for nn.GroupNorm
        activation_fn="swish",
    ):
        super().__init__()
        self.filters = filters
        self.num_res_blocks = num_res_blocks
        self.num_blocks = len(channel_multipliers)
        self.channel_multipliers = channel_multipliers
        self.temporal_downsample = temporal_downsample
        self.num_groups = num_groups
        self.embedding_dim = latent_embed_dim

        self.activation_fn = get_activation_fn(activation_fn)
        self.activate = self.activation_fn()
        self.conv_fn = CausalConv3d
        self.block_args = dict(
            conv_fn=self.conv_fn,
            activation_fn=self.activation_fn,
            use_conv_shortcut=False,
            num_groups=self.num_groups,
        )

        # first layer conv
        self.conv_in = self.conv_fn(
            in_out_channels,
            filters,
            kernel_size=(3, 3, 3),
            bias=False,
        )

        # ResBlocks and conv downsample
        self.block_res_blocks = torch.nn.ModuleList([])
        self.conv_blocks = torch.nn.ModuleList([])

        filters = self.filters
        prev_filters = filters  # record for in_channels
        for i in range(self.num_blocks):
            filters = self.filters * self.channel_multipliers[i]
            block_items = torch.nn.ModuleList([])
            for _ in range(self.num_res_blocks):
                block_items.append(
                    ResBlock(prev_filters, filters, **self.block_args))
                prev_filters = filters  # update in_channels
            self.block_res_blocks.append(block_items)

            if i < self.num_blocks - 1:
                if self.temporal_downsample[i]:
                    t_stride = 2 if self.temporal_downsample[i] else 1
                    s_stride = 1
                    self.conv_blocks.append(
                        self.conv_fn(prev_filters,
                                     filters,
                                     kernel_size=(3, 3, 3),
                                     strides=(t_stride, s_stride, s_stride)))
                    prev_filters = filters  # update in_channels
                else:
                    # if no t downsample, don't add since this does nothing for pipeline models
                    self.conv_blocks.append(
                        torch.nn.Identity(prev_filters))  # Identity
                    prev_filters = filters  # update in_channels

        # last layer res block
        self.res_blocks = torch.nn.ModuleList([])
        for _ in range(self.num_res_blocks):
            self.res_blocks.append(
                ResBlock(prev_filters, filters, **self.block_args))
            prev_filters = filters  # update in_channels

        # MAGVIT uses Group Normalization
        self.norm1 = torch.nn.GroupNorm(self.num_groups, prev_filters)

        self.conv2 = self.conv_fn(prev_filters,
                                  self.embedding_dim,
                                  kernel_size=(1, 1, 1),
                                  padding="same")

    def forward(self, x):
        x = self.conv_in(x)

        for i in range(self.num_blocks):
            for j in range(self.num_res_blocks):
                x = self.block_res_blocks[i][j](x)
            if i < self.num_blocks - 1:
                x = self.conv_blocks[i](x)
        for i in range(self.num_res_blocks):
            x = self.res_blocks[i](x)

        x = self.norm1(x)
        x = self.activate(x)
        x = self.conv2(x)
        return x


class Decoder(torch.nn.Module):
    """Decoder Blocks."""

    def __init__(
        self,
        in_out_channels=4,
        latent_embed_dim=512,
        filters=128,
        num_res_blocks=4,
        channel_multipliers=(1, 2, 2, 4),
        temporal_downsample=(False, True, True),
        num_groups=32,  # for nn.GroupNorm
        activation_fn="swish",
    ):
        super().__init__()
        self.filters = filters
        self.num_res_blocks = num_res_blocks
        self.num_blocks = len(channel_multipliers)
        self.channel_multipliers = channel_multipliers
        self.temporal_downsample = temporal_downsample
        self.num_groups = num_groups
        self.embedding_dim = latent_embed_dim
        self.s_stride = 1

        self.activation_fn = get_activation_fn(activation_fn)
        self.activate = self.activation_fn()
        self.conv_fn = CausalConv3d
        self.block_args = dict(
            conv_fn=self.conv_fn,
            activation_fn=self.activation_fn,
            use_conv_shortcut=False,
            num_groups=self.num_groups,
        )

        filters = self.filters * self.channel_multipliers[-1]
        prev_filters = filters

        # last conv
        self.conv1 = self.conv_fn(self.embedding_dim,
                                  filters,
                                  kernel_size=(3, 3, 3),
                                  bias=True)

        # last layer res block
        self.res_blocks = torch.nn.ModuleList([])
        for _ in range(self.num_res_blocks):
            self.res_blocks.append(ResBlock(filters, filters,
                                            **self.block_args))

        # ResBlocks and conv upsample
        self.block_res_blocks = torch.nn.ModuleList([])
        self.num_blocks = len(self.channel_multipliers)
        self.conv_blocks = torch.nn.ModuleList([])
        # reverse to keep track of the in_channels, but append also in a reverse direction
        for i in reversed(range(self.num_blocks)):
            filters = self.filters * self.channel_multipliers[i]
            # resblock handling
            block_items = torch.nn.ModuleList([])
            for _ in range(self.num_res_blocks):
                block_items.append(
                    ResBlock(prev_filters, filters, **self.block_args))
                prev_filters = filters  # SCH: update in_channels
            self.block_res_blocks.insert(0, block_items)  # SCH: append in front

            # conv blocks with upsampling
            if i > 0:
                if self.temporal_downsample[i - 1]:
                    t_stride = 2 if self.temporal_downsample[i - 1] else 1
                    # SCH: T-Causal Conv 3x3x3, f -> (t_stride * 2 * 2) * f, depth to space t_stride x 2 x 2
                    self.conv_blocks.insert(
                        0,
                        self.conv_fn(prev_filters,
                                     prev_filters * t_stride * self.s_stride *
                                     self.s_stride,
                                     kernel_size=(3, 3, 3)),
                    )
                else:
                    self.conv_blocks.insert(
                        0,
                        torch.nn.Identity(prev_filters),
                    )

        self.norm1 = torch.nn.GroupNorm(self.num_groups, prev_filters)

        self.conv_out = self.conv_fn(filters, in_out_channels, 3)

    def forward(self, x):
        x = self.conv1(x)
        for i in range(self.num_res_blocks):
            x = self.res_blocks[i](x)
        for i in reversed(range(self.num_blocks)):
            for j in range(self.num_res_blocks):
                x = self.block_res_blocks[i][j](x)
            if i > 0:
                t_stride = 2 if self.temporal_downsample[i - 1] else 1
                x = self.conv_blocks[i - 1](x)
                x = rearrange(
                    x,
                    "B (C ts hs ws) T H W -> B C (T ts) (H hs) (W ws)",
                    ts=t_stride,
                    hs=self.s_stride,
                    ws=self.s_stride,
                )

        x = self.norm1(x)
        x = self.activate(x)
        x = self.conv_out(x)
        return x


class VAE_Temporal(torch.nn.Module):

    def __init__(
        self,
        in_out_channels=4,
        latent_embed_dim=4,
        embed_dim=4,
        filters=128,
        num_res_blocks=4,
        channel_multipliers=(1, 2, 2, 4),
        temporal_downsample=(True, True, False),
        num_groups=32,  # for nn.GroupNorm
        activation_fn="swish",
    ):
        super().__init__()

        self.time_downsample_factor = 2**sum(temporal_downsample)
        # self.time_padding = self.time_downsample_factor - 1
        self.patch_size = (self.time_downsample_factor, 1, 1)
        self.out_channels = in_out_channels

        # NOTE: following MAGVIT, conv in bias=False in encoder first conv
        self.encoder = Encoder(
            in_out_channels=in_out_channels,
            latent_embed_dim=latent_embed_dim * 2,
            filters=filters,
            num_res_blocks=num_res_blocks,
            channel_multipliers=channel_multipliers,
            temporal_downsample=temporal_downsample,
            num_groups=num_groups,  # for nn.GroupNorm
            activation_fn=activation_fn,
        )
        self.quant_conv = CausalConv3d(2 * latent_embed_dim, 2 * embed_dim, 1)

        self.post_quant_conv = CausalConv3d(embed_dim, latent_embed_dim, 1)
        self.decoder = Decoder(
            in_out_channels=in_out_channels,
            latent_embed_dim=latent_embed_dim,
            filters=filters,
            num_res_blocks=num_res_blocks,
            channel_multipliers=channel_multipliers,
            temporal_downsample=temporal_downsample,
            num_groups=num_groups,  # for nn.GroupNorm
            activation_fn=activation_fn,
        )

    def get_latent_size(self, input_size):
        latent_size = []
        for i in range(3):
            if input_size[i] is None:
                lsize = None
            elif i == 0:
                time_padding = (0 if
                                (input_size[i] % self.time_downsample_factor
                                 == 0) else self.time_downsample_factor -
                                input_size[i] % self.time_downsample_factor)
                lsize = (input_size[i] + time_padding) // self.patch_size[i]
            else:
                lsize = input_size[i] // self.patch_size[i]
            latent_size.append(lsize)
        return latent_size

    def encode(self, x):
        time_padding = x.shape[2] % self.time_downsample_factor
        if time_padding != 0:
            time_padding = self.time_downsample_factor - time_padding
        x = pad_at_dim(x, (time_padding, 0), dim=2)
        encoded_feature = self.encoder(x)
        moments = self.quant_conv(encoded_feature).to(x.dtype)
        posterior = DiagonalGaussianDistribution(moments)
        return posterior

    def decode(self, z, num_frames=None):
        time_padding = num_frames % self.time_downsample_factor
        if time_padding != 0:
            time_padding = self.time_downsample_factor - time_padding
        z = self.post_quant_conv(z)
        x = self.decoder(z)
        x = x[:, :, time_padding:]
        return x

    def forward(self, x, sample_posterior=True):
        posterior = self.encode(x)
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()
        recon_video = self.decode(z, num_frames=x.shape[2])
        return recon_video, posterior, z


VAE_MODELS = {
    "VideoAutoencoderKL": VideoAutoencoderKL,
    "VAE_Temporal_SD": VAE_Temporal,
}


class VideoAutoencoderPipelineConfig(PretrainedConfig):
    model_type = "VideoAutoencoderPipeline"

    def __init__(
        self,
        spatial_vae_config=None,
        temporal_vae_config=None,
        from_pretrained=None,
        freeze_vae_2d=False,
        cal_loss=False,
        micro_frame_size=None,
        shift=0.0,
        scale=1.0,
        **kwargs,
    ):
        self.spatial_vae_config = spatial_vae_config
        self.temporal_vae_config = temporal_vae_config
        self.from_pretrained = from_pretrained
        self.freeze_vae_2d = freeze_vae_2d
        self.cal_loss = cal_loss
        self.micro_frame_size = micro_frame_size
        self.shift = shift
        self.scale = scale
        super().__init__(**kwargs)


class VideoAutoencoderPipeline(PreTrainedModel):
    config_class = VideoAutoencoderPipelineConfig

    def __init__(self, config: VideoAutoencoderPipelineConfig):
        super().__init__(config=config)
        vae_type = config.spatial_vae_config.pop('type')
        self.spatial_vae = VAE_MODELS[vae_type](**config.spatial_vae_config)

        vae_type = config.temporal_vae_config.pop('type')
        pretrained_path = config.temporal_vae_config.pop(
            'from_pretrained'
        ) if 'from_pretrained' in config.temporal_vae_config else None
        self.temporal_vae = VAE_MODELS[vae_type](**config.temporal_vae_config)
        if pretrained_path is not None:
            load_checkpoint(self.temporal_vae, pretrained_path)

        self.cal_loss = config.cal_loss
        self.micro_frame_size = config.micro_frame_size
        self.micro_z_frame_size = self.temporal_vae.get_latent_size(
            [config.micro_frame_size, None, None])[0]
        if config.freeze_vae_2d:
            for param in self.spatial_vae.parameters():
                param.requires_grad = False
        self.out_channels = self.temporal_vae.out_channels
        # normalization parameters
        scale = torch.tensor(config.scale)
        shift = torch.tensor(config.shift)
        if len(scale.shape) > 0:
            scale = scale[None, :, None, None, None]
        if len(shift.shape) > 0:
            shift = shift[None, :, None, None, None]
        self.register_buffer("scale", scale)
        self.register_buffer("shift", shift)

    def encode(self, x):
        x_z = self.spatial_vae.encode(x)

        if self.micro_frame_size is None:
            posterior = self.temporal_vae.encode(x_z)
            z = posterior.sample()
        else:
            z_list = []
            for i in range(0, x_z.shape[2], self.micro_frame_size):
                x_z_bs = x_z[:, :, i:i + self.micro_frame_size]
                posterior = self.temporal_vae.encode(x_z_bs)
                z_list.append(posterior.sample())
            z = torch.cat(z_list, dim=2)

        if self.cal_loss:
            return z, posterior, x_z
        else:
            return (z - self.shift) / self.scale

    def decode(self, z, num_frames=None):
        if not self.cal_loss:
            z = z * self.scale.to(z.dtype) + self.shift.to(z.dtype)

        if self.micro_frame_size is None:
            x_z = self.temporal_vae.decode(z, num_frames=num_frames)
            x = self.spatial_vae.decode(x_z)
        else:
            x_z_list = []
            for i in range(0, z.size(2), self.micro_z_frame_size):
                z_bs = z[:, :, i:i + self.micro_z_frame_size]
                x_z_bs = self.temporal_vae.decode(z_bs,
                                                  num_frames=min(
                                                      self.micro_frame_size,
                                                      num_frames))
                x_z_list.append(x_z_bs)
                num_frames -= self.micro_frame_size
            x_z = torch.cat(x_z_list, dim=2)
            x = self.spatial_vae.decode(x_z)

        if self.cal_loss:
            return x, x_z
        else:
            return x

    def forward(self, x):
        assert self.cal_loss, "This method is only available when cal_loss is True"
        z, posterior, x_z = self.encode(x)
        x_rec, x_z_rec = self.decode(z, num_frames=x_z.shape[2])
        return x_rec, x_z_rec, z, posterior, x_z

    def get_latent_size(self, input_size):
        if self.micro_frame_size is None or input_size[0] is None:
            return self.temporal_vae.get_latent_size(
                self.spatial_vae.get_latent_size(input_size))
        else:
            sub_input_size = [
                self.micro_frame_size, input_size[1], input_size[2]
            ]
            sub_latent_size = self.temporal_vae.get_latent_size(
                self.spatial_vae.get_latent_size(sub_input_size))
            sub_latent_size[0] = sub_latent_size[0] * (input_size[0] //
                                                       self.micro_frame_size)
            remain_temporal_size = [
                input_size[0] % self.micro_frame_size, None, None
            ]
            if remain_temporal_size[0] > 0:
                remain_size = self.temporal_vae.get_latent_size(
                    remain_temporal_size)
                sub_latent_size[0] += remain_size[0]
            return sub_latent_size

    def get_temporal_last_layer(self):
        return self.temporal_vae.decoder.conv_out.conv.weight

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def dtype(self):
        return next(self.parameters()).dtype


def get_vae(
    micro_batch_size=4,
    micro_frame_size=17,
    from_pretrained=None,
    local_files_only=False,
    freeze_vae_2d=False,
    cal_loss=False,
    force_huggingface=False,
):
    spatial_vae_config = dict(
        type="VideoAutoencoderKL",
        from_pretrained="PixArt-alpha/pixart_sigma_sdxlvae_T5_diffusers",
        subfolder="vae",
        micro_batch_size=micro_batch_size,
        local_files_only=local_files_only,
    )
    temporal_vae_config = dict(
        type="VAE_Temporal_SD",
        from_pretrained=None,
        in_out_channels=4,
        latent_embed_dim=4,
        embed_dim=4,
        filters=128,
        num_res_blocks=4,
        channel_multipliers=(1, 2, 2, 4),
        temporal_downsample=(False, True, True),
    )
    shift = (-0.10, 0.34, 0.27, 0.98)
    scale = (3.85, 2.32, 2.33, 3.06)
    kwargs = dict(
        spatial_vae_config=spatial_vae_config,
        temporal_vae_config=temporal_vae_config,
        freeze_vae_2d=freeze_vae_2d,
        cal_loss=cal_loss,
        micro_frame_size=micro_frame_size,
        shift=shift,
        scale=scale,
    )

    if force_huggingface or (from_pretrained is not None
                             and not os.path.exists(from_pretrained)):
        model = VideoAutoencoderPipeline.from_pretrained(
            from_pretrained, **kwargs)
    else:
        config = VideoAutoencoderPipelineConfig(**kwargs)
        model = VideoAutoencoderPipeline(config)
        if from_pretrained:
            load_checkpoint(model, from_pretrained)
    return model
