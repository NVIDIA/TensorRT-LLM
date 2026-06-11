# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#
# Copyright 2025 The Wan Team and The HuggingFace Team. All rights reserved.
#
# This module is adapted from HuggingFace diffusers'
# diffusers.models.autoencoders.autoencoder_kl_wan implementation.

from __future__ import annotations

import json
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKLOutput
from diffusers.models.autoencoders.vae import DecoderOutput, DiagonalGaussianDistribution

CACHE_T = 2
LAYOUT_MODES = ("channels_last", "contiguous")
WAN_VAE_LATENTS_MEAN = [
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
]
WAN_VAE_LATENTS_STD = [
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
]


def _validate_layout_mode(layout_mode: str) -> str:
    if layout_mode not in LAYOUT_MODES:
        raise ValueError(
            f"Unsupported Wan VAE layout_mode={layout_mode!r}; expected one of {LAYOUT_MODES}"
        )
    return layout_mode


def _contiguous_if_needed(x: torch.Tensor) -> torch.Tensor:
    if x.is_contiguous():
        return x
    return x.contiguous()


def _channels_last_3d_if_needed(x: torch.Tensor) -> torch.Tensor:
    if x.is_contiguous(memory_format=torch.channels_last_3d):
        return x
    return x.to(memory_format=torch.channels_last_3d)


def _channels_last_2d_if_needed(x: torch.Tensor) -> torch.Tensor:
    if x.is_contiguous(memory_format=torch.channels_last):
        return x
    return x.to(memory_format=torch.channels_last)


def _to_device_if_needed(x: torch.Tensor, device: torch.device) -> torch.Tensor:
    if x.device == device:
        return x
    return x.to(device)


def _format_5d_activation(x: torch.Tensor, layout_mode: str) -> torch.Tensor:
    if layout_mode == "channels_last":
        return _channels_last_3d_if_needed(x)
    return _contiguous_if_needed(x)


def _format_4d_activation(x: torch.Tensor, layout_mode: str) -> torch.Tensor:
    if layout_mode == "channels_last":
        return _channels_last_2d_if_needed(x)
    return _contiguous_if_needed(x)


def _format_conv_weight(module: nn.Module, layout_mode: str) -> None:
    if isinstance(module, nn.Conv3d):
        memory_format = (
            torch.channels_last_3d if layout_mode == "channels_last" else torch.contiguous_format
        )
        module.to(memory_format=memory_format)
    elif isinstance(module, nn.Conv2d):
        memory_format = (
            torch.channels_last if layout_mode == "channels_last" else torch.contiguous_format
        )
        module.to(memory_format=memory_format)


class LayoutAwareModule:
    layout_mode: str

    def set_layout_mode(self, layout_mode: str) -> None:
        self.layout_mode = _validate_layout_mode(layout_mode)


def _activation(name: str):
    if name == "silu":
        return F.silu
    raise ValueError(f"Unsupported Wan VAE activation: {name}")


@dataclass
class WanVAEConfig:
    base_dim: int = 96
    decoder_base_dim: int | None = None
    z_dim: int = 16
    dim_mult: list[int] | None = None
    num_res_blocks: int = 2
    attn_scales: list[float] | None = None
    temperal_downsample: list[bool] | None = None
    dropout: float = 0.0
    latents_mean: list[float] | None = None
    latents_std: list[float] | None = None
    is_residual: bool = False
    in_channels: int = 3
    out_channels: int = 3
    patch_size: int | None = None
    scale_factor_temporal: int | None = 4
    scale_factor_spatial: int | None = 8

    def __post_init__(self) -> None:
        if self.decoder_base_dim is None:
            self.decoder_base_dim = self.base_dim
        if self.dim_mult is None:
            self.dim_mult = [1, 2, 4, 4]
        if self.attn_scales is None:
            self.attn_scales = []
        if self.temperal_downsample is None:
            self.temperal_downsample = [False, True, True]
        if self.latents_mean is None:
            self.latents_mean = WAN_VAE_LATENTS_MEAN.copy()
        if self.latents_std is None:
            self.latents_std = WAN_VAE_LATENTS_STD.copy()

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> "WanVAEConfig":
        valid_names = {field.name for field in fields(cls)}
        kwargs = {key: value for key, value in config_dict.items() if key in valid_names}
        return cls(**kwargs)

    @classmethod
    def from_json_file(cls, path: str | Path) -> "WanVAEConfig":
        with open(path, encoding="utf-8") as config_file:
            return cls.from_dict(json.load(config_file))

    def get(self, name: str, default: Any = None) -> Any:
        return getattr(self, name, default)

    @property
    def public_video_channels(self) -> int:
        if self.patch_size is None:
            return self.in_channels
        patch_area = self.patch_size * self.patch_size
        if self.in_channels % patch_area != 0:
            raise ValueError(
                f"in_channels={self.in_channels} is not divisible by patch_size^2={patch_area}"
            )
        return self.in_channels // patch_area


class AvgDown3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, factor_t: int, factor_s: int = 1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.factor_t = factor_t
        self.factor_s = factor_s
        self.factor = self.factor_t * self.factor_s * self.factor_s

        if in_channels * self.factor % out_channels != 0:
            raise ValueError("AvgDown3D channel grouping must divide evenly")
        self.group_size = in_channels * self.factor // out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pad_t = (self.factor_t - x.shape[2] % self.factor_t) % self.factor_t
        if pad_t > 0:
            x = F.pad(x, (0, 0, 0, 0, pad_t, 0))
        batch_size, channels, frames, height, width = x.shape
        x = x.reshape(
            batch_size,
            channels,
            frames // self.factor_t,
            self.factor_t,
            height // self.factor_s,
            self.factor_s,
            width // self.factor_s,
            self.factor_s,
        )
        x = x.permute(0, 1, 3, 5, 7, 2, 4, 6).contiguous()
        x = x.reshape(
            batch_size,
            channels * self.factor,
            frames // self.factor_t,
            height // self.factor_s,
            width // self.factor_s,
        )
        x = x.reshape(
            batch_size,
            self.out_channels,
            self.group_size,
            frames // self.factor_t,
            height // self.factor_s,
            width // self.factor_s,
        )
        return x.mean(dim=2)


class DupUp3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, factor_t: int, factor_s: int = 1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.factor_t = factor_t
        self.factor_s = factor_s
        self.factor = self.factor_t * self.factor_s * self.factor_s

        if out_channels * self.factor % in_channels != 0:
            raise ValueError("DupUp3D channel repeat count must divide evenly")
        self.repeats = out_channels * self.factor // in_channels

    def forward(self, x: torch.Tensor, first_chunk: bool = False) -> torch.Tensor:
        x = x if self.repeats == 1 else x.repeat_interleave(self.repeats, dim=1)
        x = x.reshape(
            x.size(0),
            self.out_channels,
            self.factor_t,
            self.factor_s,
            self.factor_s,
            x.size(2),
            x.size(3),
            x.size(4),
        )
        x = x.permute(0, 1, 5, 2, 6, 3, 7, 4).contiguous()
        x = x.reshape(
            x.size(0),
            self.out_channels,
            x.size(2) * self.factor_t,
            x.size(4) * self.factor_s,
            x.size(6) * self.factor_s,
        )
        if first_chunk:
            x = x[:, :, self.factor_t - 1 :, :, :]
        return x


class WanCausalConv3d(nn.Conv3d, LayoutAwareModule):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int, int],
        stride: int | tuple[int, int, int] = 1,
        padding: int | tuple[int, int, int] = 0,
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        self.layout_mode = "contiguous"
        self._padding = (
            0,
            0,
            0,
            0,
            2 * self.padding[0],
            0,
        )
        self.padding = (0, self.padding[1], self.padding[2])

    def forward(self, x: torch.Tensor, cache_x: torch.Tensor | None = None) -> torch.Tensor:
        if self.layout_mode == "channels_last":
            x = _channels_last_3d_if_needed(x)
        padding = list(self._padding)
        if cache_x is not None and self._padding[4] > 0:
            cache_x = _to_device_if_needed(cache_x, x.device)
            if self.layout_mode == "channels_last":
                cache_x = _channels_last_3d_if_needed(cache_x)
            x = torch.cat([cache_x, x], dim=2)
            padding[4] -= cache_x.shape[2]
        if any(padding):
            x = F.pad(x, padding)
        if self.layout_mode == "channels_last":
            x = _channels_last_3d_if_needed(x)
        else:
            x = _contiguous_if_needed(x)
        x = super().forward(x)
        if self.layout_mode == "channels_last":
            return _channels_last_3d_if_needed(x)
        return x


class WanConv2d(nn.Conv2d, LayoutAwareModule):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.layout_mode = "contiguous"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.layout_mode == "channels_last":
            x = _channels_last_2d_if_needed(x)
        else:
            x = _contiguous_if_needed(x)
        x = super().forward(x)
        if self.layout_mode == "channels_last":
            return _channels_last_2d_if_needed(x)
        return x


class WanRMSNorm(nn.Module):
    def __init__(
        self,
        dim: int,
        channel_first: bool = True,
        images: bool = True,
        bias: bool = False,
    ) -> None:
        super().__init__()
        broadcastable_dims = (1, 1, 1) if not images else (1, 1)
        shape = (dim, *broadcastable_dims) if channel_first else (dim,)

        self.channel_first = channel_first
        self.scale = dim**0.5
        self.gamma = nn.Parameter(torch.ones(shape))
        self.bias = nn.Parameter(torch.zeros(shape)) if bias else 0.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm_dim = 1 if self.channel_first else -1
        return F.normalize(x, dim=norm_dim) * self.scale * self.gamma + self.bias


class WanUpsample(nn.Upsample):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x.float()).type_as(x)


class WanResample(nn.Module, LayoutAwareModule):
    def __init__(self, dim: int, mode: str, upsample_out_dim: int | None = None) -> None:
        super().__init__()
        self.layout_mode = "contiguous"
        self.dim = dim
        self.mode = mode
        if upsample_out_dim is None:
            upsample_out_dim = dim // 2

        if mode == "upsample2d":
            self.resample = nn.Sequential(
                WanUpsample(scale_factor=(2.0, 2.0), mode="nearest-exact"),
                WanConv2d(dim, upsample_out_dim, 3, padding=1),
            )
        elif mode == "upsample3d":
            self.resample = nn.Sequential(
                WanUpsample(scale_factor=(2.0, 2.0), mode="nearest-exact"),
                WanConv2d(dim, upsample_out_dim, 3, padding=1),
            )
            self.time_conv = WanCausalConv3d(dim, dim * 2, (3, 1, 1), padding=(1, 0, 0))
        elif mode == "downsample2d":
            self.resample = nn.Sequential(
                nn.ZeroPad2d((0, 1, 0, 1)), WanConv2d(dim, dim, 3, stride=(2, 2))
            )
        elif mode == "downsample3d":
            self.resample = nn.Sequential(
                nn.ZeroPad2d((0, 1, 0, 1)), WanConv2d(dim, dim, 3, stride=(2, 2))
            )
            self.time_conv = WanCausalConv3d(dim, dim, (3, 1, 1), stride=(2, 1, 1))
        else:
            self.resample = nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        feat_cache: list[torch.Tensor | str | None] | None = None,
        feat_idx: list[int] | None = None,
    ) -> torch.Tensor:
        batch_size, channels, frames, height, width = x.size()
        if feat_cache is not None and feat_idx is None:
            raise ValueError("feat_idx is required when feat_cache is provided")

        if self.mode == "upsample3d" and feat_cache is not None and feat_idx is not None:
            idx = feat_idx[0]
            if feat_cache[idx] is None:
                feat_cache[idx] = "Rep"
                feat_idx[0] += 1
            else:
                cache_x = x[:, :, -CACHE_T:, :, :]
                if (
                    cache_x.shape[2] < 2
                    and feat_cache[idx] is not None
                    and feat_cache[idx] != "Rep"
                ):
                    cache_x = torch.cat(
                        [
                            _to_device_if_needed(
                                feat_cache[idx][:, :, -1, :, :].unsqueeze(2), cache_x.device
                            ),
                            cache_x,
                        ],
                        dim=2,
                    )
                if cache_x.shape[2] < 2 and feat_cache[idx] == "Rep":
                    cache_x = torch.cat([torch.zeros_like(cache_x), cache_x], dim=2)
                if feat_cache[idx] == "Rep":
                    x = self.time_conv(x)
                else:
                    x = self.time_conv(x, feat_cache[idx])
                feat_cache[idx] = cache_x
                feat_idx[0] += 1

                x = x.reshape(batch_size, 2, channels, frames, height, width)
                x = torch.stack((x[:, 0, :, :, :, :], x[:, 1, :, :, :, :]), 3)
                x = x.reshape(batch_size, channels, frames * 2, height, width)

        frames = x.shape[2]
        x_4d = x.permute(0, 2, 1, 3, 4).reshape(batch_size * frames, channels, height, width)
        x = _format_4d_activation(x_4d, self.layout_mode)
        x = self.resample(x)
        x_5d = x.reshape(batch_size, frames, x.size(1), x.size(2), x.size(3)).permute(0, 2, 1, 3, 4)
        x = _format_5d_activation(x_5d, self.layout_mode)

        if self.mode == "downsample3d" and feat_cache is not None and feat_idx is not None:
            idx = feat_idx[0]
            if feat_cache[idx] is None:
                feat_cache[idx] = x
                feat_idx[0] += 1
            else:
                cache_x = x[:, :, -1:, :, :]
                x = self.time_conv(torch.cat([feat_cache[idx][:, :, -1:, :, :], x], dim=2))
                feat_cache[idx] = cache_x
                feat_idx[0] += 1
        return x


class WanResidualBlock(nn.Module, LayoutAwareModule):
    def __init__(
        self, in_dim: int, out_dim: int, dropout: float = 0.0, non_linearity: str = "silu"
    ):
        super().__init__()
        self.layout_mode = "contiguous"
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.nonlinearity = _activation(non_linearity)
        self.norm1 = WanRMSNorm(in_dim, images=False)
        self.conv1 = WanCausalConv3d(in_dim, out_dim, 3, padding=1)
        self.norm2 = WanRMSNorm(out_dim, images=False)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = WanCausalConv3d(out_dim, out_dim, 3, padding=1)
        self.conv_shortcut = (
            WanCausalConv3d(in_dim, out_dim, 1) if in_dim != out_dim else nn.Identity()
        )

    def forward(
        self,
        x: torch.Tensor,
        feat_cache: list[torch.Tensor | str | None] | None = None,
        feat_idx: list[int] | None = None,
    ) -> torch.Tensor:
        if feat_cache is not None and feat_idx is None:
            raise ValueError("feat_idx is required when feat_cache is provided")

        residual = self.conv_shortcut(x)
        x = self.nonlinearity(self.norm1(x))

        if feat_cache is not None and feat_idx is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -CACHE_T:, :, :]
            if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                cache_x = torch.cat(
                    [
                        _to_device_if_needed(
                            feat_cache[idx][:, :, -1, :, :].unsqueeze(2), cache_x.device
                        ),
                        cache_x,
                    ],
                    dim=2,
                )
            x = self.conv1(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.conv1(x)

        x = self.dropout(self.nonlinearity(self.norm2(x)))

        if feat_cache is not None and feat_idx is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -CACHE_T:, :, :]
            if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                cache_x = torch.cat(
                    [
                        _to_device_if_needed(
                            feat_cache[idx][:, :, -1, :, :].unsqueeze(2), cache_x.device
                        ),
                        cache_x,
                    ],
                    dim=2,
                )
            x = self.conv2(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.conv2(x)

        return _format_5d_activation(x + residual, self.layout_mode)


class WanAttentionBlock(nn.Module, LayoutAwareModule):
    def __init__(self, dim: int):
        super().__init__()
        self.layout_mode = "contiguous"
        self.dim = dim
        self.norm = WanRMSNorm(dim)
        self.to_qkv = WanConv2d(dim, dim * 3, 1)
        self.proj = WanConv2d(dim, dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        batch_size, channels, frames, height, width = x.size()
        x = x.permute(0, 2, 1, 3, 4).reshape(batch_size * frames, channels, height, width)
        x = _format_4d_activation(x, self.layout_mode)
        x = self.norm(x)

        qkv = self.to_qkv(x)
        qkv = qkv.reshape(batch_size * frames, 1, channels * 3, -1)
        qkv = qkv.permute(0, 1, 3, 2).contiguous()
        q, k, v = qkv.chunk(3, dim=-1)
        x = F.scaled_dot_product_attention(q, k, v)
        x = x.squeeze(1).permute(0, 2, 1).reshape(batch_size * frames, channels, height, width)
        x = self.proj(x)
        x = x.reshape(batch_size, frames, channels, height, width).permute(0, 2, 1, 3, 4)
        return _format_5d_activation(x + residual, self.layout_mode)


class WanMidBlock(nn.Module, LayoutAwareModule):
    def __init__(
        self, dim: int, dropout: float = 0.0, non_linearity: str = "silu", num_layers: int = 1
    ):
        super().__init__()
        self.layout_mode = "contiguous"
        resnets = [WanResidualBlock(dim, dim, dropout, non_linearity)]
        attentions = []
        for _ in range(num_layers):
            attentions.append(WanAttentionBlock(dim))
            resnets.append(WanResidualBlock(dim, dim, dropout, non_linearity))
        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)
        self.gradient_checkpointing = False

    def forward(
        self,
        x: torch.Tensor,
        feat_cache: list[torch.Tensor | str | None] | None = None,
        feat_idx: list[int] | None = None,
    ) -> torch.Tensor:
        x = self.resnets[0](x, feat_cache=feat_cache, feat_idx=feat_idx)
        for attn, resnet in zip(self.attentions, self.resnets[1:]):
            x = attn(x)
            x = resnet(x, feat_cache=feat_cache, feat_idx=feat_idx)
        return _format_5d_activation(x, self.layout_mode)


class WanResidualDownBlock(nn.Module, LayoutAwareModule):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        dropout: float,
        num_res_blocks: int,
        temperal_downsample: bool = False,
        down_flag: bool = False,
    ):
        super().__init__()
        self.layout_mode = "contiguous"
        self.avg_shortcut = AvgDown3D(
            in_dim,
            out_dim,
            factor_t=2 if temperal_downsample else 1,
            factor_s=2 if down_flag else 1,
        )
        resnets = []
        for _ in range(num_res_blocks):
            resnets.append(WanResidualBlock(in_dim, out_dim, dropout))
            in_dim = out_dim
        self.resnets = nn.ModuleList(resnets)
        self.downsampler = None
        if down_flag:
            mode = "downsample3d" if temperal_downsample else "downsample2d"
            self.downsampler = WanResample(out_dim, mode=mode)

    def forward(
        self,
        x: torch.Tensor,
        feat_cache: list[torch.Tensor | str | None] | None = None,
        feat_idx: list[int] | None = None,
    ) -> torch.Tensor:
        residual = x
        for resnet in self.resnets:
            x = resnet(x, feat_cache=feat_cache, feat_idx=feat_idx)
        if self.downsampler is not None:
            x = self.downsampler(x, feat_cache=feat_cache, feat_idx=feat_idx)
        return _format_5d_activation(x + self.avg_shortcut(residual), self.layout_mode)


class WanEncoder3d(nn.Module, LayoutAwareModule):
    def __init__(
        self,
        in_channels: int = 3,
        dim: int = 128,
        z_dim: int = 4,
        dim_mult: list[int] | None = None,
        num_res_blocks: int = 2,
        attn_scales: list[float] | None = None,
        temperal_downsample: list[bool] | None = None,
        dropout: float = 0.0,
        non_linearity: str = "silu",
        is_residual: bool = False,
    ):
        super().__init__()
        self.layout_mode = "contiguous"
        dim_mult = [1, 2, 4, 4] if dim_mult is None else dim_mult
        attn_scales = [] if attn_scales is None else attn_scales
        temperal_downsample = (
            [False, True, True] if temperal_downsample is None else temperal_downsample
        )
        self.nonlinearity = _activation(non_linearity)

        dims = [dim * value for value in [1] + dim_mult]
        scale = 1.0
        self.conv_in = WanCausalConv3d(in_channels, dims[0], 3, padding=1)
        self.down_blocks = nn.ModuleList([])
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            if is_residual:
                self.down_blocks.append(
                    WanResidualDownBlock(
                        in_dim,
                        out_dim,
                        dropout,
                        num_res_blocks,
                        temperal_downsample=temperal_downsample[i]
                        if i != len(dim_mult) - 1
                        else False,
                        down_flag=i != len(dim_mult) - 1,
                    )
                )
            else:
                for _ in range(num_res_blocks):
                    self.down_blocks.append(WanResidualBlock(in_dim, out_dim, dropout))
                    if scale in attn_scales:
                        self.down_blocks.append(WanAttentionBlock(out_dim))
                    in_dim = out_dim
                if i != len(dim_mult) - 1:
                    mode = "downsample3d" if temperal_downsample[i] else "downsample2d"
                    self.down_blocks.append(WanResample(out_dim, mode=mode))
                    scale /= 2.0

        self.mid_block = WanMidBlock(out_dim, dropout, non_linearity, num_layers=1)
        self.norm_out = WanRMSNorm(out_dim, images=False)
        self.conv_out = WanCausalConv3d(out_dim, z_dim, 3, padding=1)
        self.gradient_checkpointing = False

    def forward(
        self,
        x: torch.Tensor,
        feat_cache: list[torch.Tensor | str | None] | None = None,
        feat_idx: list[int] | None = None,
    ) -> torch.Tensor:
        if feat_cache is not None and feat_idx is None:
            raise ValueError("feat_idx is required when feat_cache is provided")

        if feat_cache is not None and feat_idx is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -CACHE_T:, :, :]
            if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                cache_x = torch.cat(
                    [
                        _to_device_if_needed(
                            feat_cache[idx][:, :, -1, :, :].unsqueeze(2), cache_x.device
                        ),
                        cache_x,
                    ],
                    dim=2,
                )
            x = self.conv_in(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.conv_in(x)

        for layer in self.down_blocks:
            if feat_cache is not None:
                x = layer(x, feat_cache=feat_cache, feat_idx=feat_idx)
            else:
                x = layer(x)

        x = self.mid_block(x, feat_cache=feat_cache, feat_idx=feat_idx)
        x = self.nonlinearity(self.norm_out(x))
        if feat_cache is not None and feat_idx is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -CACHE_T:, :, :]
            if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                cache_x = torch.cat(
                    [
                        _to_device_if_needed(
                            feat_cache[idx][:, :, -1, :, :].unsqueeze(2), cache_x.device
                        ),
                        cache_x,
                    ],
                    dim=2,
                )
            x = self.conv_out(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.conv_out(x)
        return _format_5d_activation(x, self.layout_mode)


class WanResidualUpBlock(nn.Module, LayoutAwareModule):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_res_blocks: int,
        dropout: float = 0.0,
        temperal_upsample: bool = False,
        up_flag: bool = False,
        non_linearity: str = "silu",
    ):
        super().__init__()
        self.layout_mode = "contiguous"
        self.avg_shortcut = None
        if up_flag:
            self.avg_shortcut = DupUp3D(
                in_dim, out_dim, factor_t=2 if temperal_upsample else 1, factor_s=2
            )

        current_dim = in_dim
        resnets = []
        for _ in range(num_res_blocks + 1):
            resnets.append(WanResidualBlock(current_dim, out_dim, dropout, non_linearity))
            current_dim = out_dim
        self.resnets = nn.ModuleList(resnets)

        self.upsampler = None
        if up_flag:
            mode = "upsample3d" if temperal_upsample else "upsample2d"
            self.upsampler = WanResample(out_dim, mode=mode, upsample_out_dim=out_dim)
        self.gradient_checkpointing = False

    def forward(
        self,
        x: torch.Tensor,
        feat_cache: list[torch.Tensor | str | None] | None = None,
        feat_idx: list[int] | None = None,
        first_chunk: bool = False,
    ) -> torch.Tensor:
        residual = x
        for resnet in self.resnets:
            x = resnet(x, feat_cache=feat_cache, feat_idx=feat_idx)
        if self.upsampler is not None:
            x = self.upsampler(x, feat_cache=feat_cache, feat_idx=feat_idx)
        if self.avg_shortcut is not None:
            x = x + self.avg_shortcut(residual, first_chunk=first_chunk)
        return _format_5d_activation(x, self.layout_mode)


class WanUpBlock(nn.Module, LayoutAwareModule):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_res_blocks: int,
        dropout: float = 0.0,
        upsample_mode: str | None = None,
        non_linearity: str = "silu",
    ):
        super().__init__()
        self.layout_mode = "contiguous"
        current_dim = in_dim
        resnets = []
        for _ in range(num_res_blocks + 1):
            resnets.append(WanResidualBlock(current_dim, out_dim, dropout, non_linearity))
            current_dim = out_dim
        self.resnets = nn.ModuleList(resnets)
        self.upsamplers = None
        if upsample_mode is not None:
            self.upsamplers = nn.ModuleList([WanResample(out_dim, mode=upsample_mode)])
        self.gradient_checkpointing = False

    def forward(
        self,
        x: torch.Tensor,
        feat_cache: list[torch.Tensor | str | None] | None = None,
        feat_idx: list[int] | None = None,
        first_chunk: bool | None = None,
    ) -> torch.Tensor:
        del first_chunk
        for resnet in self.resnets:
            x = resnet(x, feat_cache=feat_cache, feat_idx=feat_idx)
        if self.upsamplers is not None:
            x = self.upsamplers[0](x, feat_cache=feat_cache, feat_idx=feat_idx)
        return _format_5d_activation(x, self.layout_mode)


class WanDecoder3d(nn.Module, LayoutAwareModule):
    def __init__(
        self,
        dim: int = 128,
        z_dim: int = 4,
        dim_mult: list[int] | None = None,
        num_res_blocks: int = 2,
        attn_scales: list[float] | None = None,
        temperal_upsample: list[bool] | None = None,
        dropout: float = 0.0,
        non_linearity: str = "silu",
        out_channels: int = 3,
        is_residual: bool = False,
    ):
        super().__init__()
        self.layout_mode = "contiguous"
        dim_mult = [1, 2, 4, 4] if dim_mult is None else dim_mult
        attn_scales = [] if attn_scales is None else attn_scales
        del attn_scales
        temperal_upsample = [False, True, True] if temperal_upsample is None else temperal_upsample
        self.nonlinearity = _activation(non_linearity)

        dims = [dim * value for value in [dim_mult[-1]] + dim_mult[::-1]]
        self.conv_in = WanCausalConv3d(z_dim, dims[0], 3, padding=1)
        self.mid_block = WanMidBlock(dims[0], dropout, non_linearity, num_layers=1)
        self.up_blocks = nn.ModuleList([])
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            if i > 0 and not is_residual:
                in_dim = in_dim // 2
            up_flag = i != len(dim_mult) - 1
            upsample_mode = None
            if up_flag and temperal_upsample[i]:
                upsample_mode = "upsample3d"
            elif up_flag:
                upsample_mode = "upsample2d"

            if is_residual:
                up_block = WanResidualUpBlock(
                    in_dim=in_dim,
                    out_dim=out_dim,
                    num_res_blocks=num_res_blocks,
                    dropout=dropout,
                    temperal_upsample=temperal_upsample[i] if up_flag else False,
                    up_flag=up_flag,
                    non_linearity=non_linearity,
                )
            else:
                up_block = WanUpBlock(
                    in_dim=in_dim,
                    out_dim=out_dim,
                    num_res_blocks=num_res_blocks,
                    dropout=dropout,
                    upsample_mode=upsample_mode,
                    non_linearity=non_linearity,
                )
            self.up_blocks.append(up_block)

        self.norm_out = WanRMSNorm(out_dim, images=False)
        self.conv_out = WanCausalConv3d(out_dim, out_channels, 3, padding=1)
        self.gradient_checkpointing = False

    def forward(
        self,
        x: torch.Tensor,
        feat_cache: list[torch.Tensor | str | None] | None = None,
        feat_idx: list[int] | None = None,
        first_chunk: bool = False,
    ) -> torch.Tensor:
        if feat_cache is not None and feat_idx is None:
            raise ValueError("feat_idx is required when feat_cache is provided")

        if feat_cache is not None and feat_idx is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -CACHE_T:, :, :]
            if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                cache_x = torch.cat(
                    [
                        _to_device_if_needed(
                            feat_cache[idx][:, :, -1, :, :].unsqueeze(2), cache_x.device
                        ),
                        cache_x,
                    ],
                    dim=2,
                )
            x = self.conv_in(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.conv_in(x)

        x = self.mid_block(x, feat_cache=feat_cache, feat_idx=feat_idx)
        for up_block in self.up_blocks:
            x = up_block(x, feat_cache=feat_cache, feat_idx=feat_idx, first_chunk=first_chunk)

        x = self.nonlinearity(self.norm_out(x))
        if feat_cache is not None and feat_idx is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -CACHE_T:, :, :]
            if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                cache_x = torch.cat(
                    [
                        _to_device_if_needed(
                            feat_cache[idx][:, :, -1, :, :].unsqueeze(2), cache_x.device
                        ),
                        cache_x,
                    ],
                    dim=2,
                )
            x = self.conv_out(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.conv_out(x)
        return _format_5d_activation(x, self.layout_mode)


def patchify(x: torch.Tensor, patch_size: int) -> torch.Tensor:
    if patch_size == 1:
        return x
    if x.dim() != 5:
        raise ValueError(f"Invalid input shape: {x.shape}")
    batch_size, channels, frames, height, width = x.shape
    if height % patch_size != 0 or width % patch_size != 0:
        raise ValueError(
            f"Height ({height}) and width ({width}) must be divisible by patch_size ({patch_size})"
        )
    x = x.reshape(
        batch_size,
        channels,
        frames,
        height // patch_size,
        patch_size,
        width // patch_size,
        patch_size,
    )
    x = x.permute(0, 1, 6, 4, 2, 3, 5).contiguous()
    return x.reshape(
        batch_size,
        channels * patch_size * patch_size,
        frames,
        height // patch_size,
        width // patch_size,
    )


def unpatchify(x: torch.Tensor, patch_size: int) -> torch.Tensor:
    if patch_size == 1:
        return x
    if x.dim() != 5:
        raise ValueError(f"Invalid input shape: {x.shape}")
    batch_size, channels_patches, frames, height, width = x.shape
    channels = channels_patches // (patch_size * patch_size)
    x = x.reshape(batch_size, channels, patch_size, patch_size, frames, height, width)
    x = x.permute(0, 1, 4, 5, 3, 6, 2).contiguous()
    return x.reshape(batch_size, channels, frames, height * patch_size, width * patch_size)


class WanVAE(nn.Module):
    """Wan VAE implementation compatible with diffusers AutoencoderKLWan."""

    def __init__(
        self,
        config: WanVAEConfig | dict[str, Any] | None = None,
        layout_mode: str = "contiguous",
    ) -> None:
        super().__init__()
        if config is None:
            config = WanVAEConfig()
        if isinstance(config, dict):
            config = WanVAEConfig.from_dict(config)
        self.config = config
        self.z_dim = config.z_dim
        self.temperal_downsample = config.temperal_downsample
        self.temperal_upsample = config.temperal_downsample[::-1]

        self.encoder = WanEncoder3d(
            in_channels=config.in_channels,
            dim=config.base_dim,
            z_dim=config.z_dim * 2,
            dim_mult=config.dim_mult,
            num_res_blocks=config.num_res_blocks,
            attn_scales=config.attn_scales,
            temperal_downsample=config.temperal_downsample,
            dropout=config.dropout,
            is_residual=config.is_residual,
        )
        self.quant_conv = WanCausalConv3d(config.z_dim * 2, config.z_dim * 2, 1)
        self.post_quant_conv = WanCausalConv3d(config.z_dim, config.z_dim, 1)
        self.decoder = WanDecoder3d(
            dim=config.decoder_base_dim,
            z_dim=config.z_dim,
            dim_mult=config.dim_mult,
            num_res_blocks=config.num_res_blocks,
            attn_scales=config.attn_scales,
            temperal_upsample=self.temperal_upsample,
            dropout=config.dropout,
            out_channels=config.out_channels,
            is_residual=config.is_residual,
        )

        self.spatial_compression_ratio = config.scale_factor_spatial
        self.use_slicing = False
        self.use_tiling = False
        self.layout_mode = _validate_layout_mode(layout_mode)
        self.set_layout_mode(self.layout_mode)
        self._cached_conv_counts = {
            "decoder": sum(
                isinstance(module, WanCausalConv3d) for module in self.decoder.modules()
            ),
            "encoder": sum(
                isinstance(module, WanCausalConv3d) for module in self.encoder.modules()
            ),
        }
        self.clear_cache()

    @classmethod
    def from_config_file(cls, path: str | Path, layout_mode: str = "contiguous") -> "WanVAE":
        return cls(WanVAEConfig.from_json_file(path), layout_mode=layout_mode)

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype

    def set_layout_mode(self, layout_mode: str) -> None:
        self.layout_mode = _validate_layout_mode(layout_mode)
        for module in self.modules():
            if module is self:
                continue
            if isinstance(module, LayoutAwareModule):
                module.set_layout_mode(self.layout_mode)
            _format_conv_weight(module, self.layout_mode)

    def load_diffusers_state_dict(
        self,
        state_dict: dict[str, torch.Tensor],
        strict: bool = True,
    ) -> Any:
        return self.load_state_dict(state_dict, strict=strict)

    def enable_tiling(self, *_args: Any, **_kwargs: Any) -> None:
        raise NotImplementedError("WanVAE does not support tiled encode/decode yet")

    def clear_cache(self) -> None:
        self._conv_num = self._cached_conv_counts["decoder"]
        self._conv_idx = [0]
        self._feat_map: list[torch.Tensor | str | None] = [None] * self._conv_num
        self._enc_conv_num = self._cached_conv_counts["encoder"]
        self._enc_conv_idx = [0]
        self._enc_feat_map: list[torch.Tensor | str | None] = [None] * self._enc_conv_num

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        _, _, num_frame, _, _ = x.shape

        if self.use_tiling:
            raise NotImplementedError("WanVAE tiled encode is not implemented")

        self.clear_cache()
        if self.config.patch_size is not None:
            x = patchify(x, patch_size=self.config.patch_size)
        x = _format_5d_activation(x, self.layout_mode)

        num_chunks = 1 + (num_frame - 1) // 4
        out_chunks: list[torch.Tensor] = []
        for i in range(num_chunks):
            self._enc_conv_idx = [0]
            if i == 0:
                out_chunk = self.encoder(
                    x[:, :, :1, :, :], feat_cache=self._enc_feat_map, feat_idx=self._enc_conv_idx
                )
            else:
                out_chunk = self.encoder(
                    x[:, :, 1 + 4 * (i - 1) : 1 + 4 * i, :, :],
                    feat_cache=self._enc_feat_map,
                    feat_idx=self._enc_conv_idx,
                )
            out_chunks.append(out_chunk)

        if len(out_chunks) == 1:
            out = out_chunks[0]
        else:
            out = torch.cat(out_chunks, dim=2)

        enc = self.quant_conv(out)
        enc = _format_5d_activation(enc, self.layout_mode)
        self.clear_cache()
        return enc

    def encode(
        self,
        x: torch.Tensor,
        return_dict: bool = True,
    ) -> AutoencoderKLOutput | tuple[DiagonalGaussianDistribution]:
        if self.use_slicing and x.shape[0] > 1:
            encoded_slices = [self._encode(x_slice) for x_slice in x.split(1)]
            h = torch.cat(encoded_slices)
        else:
            h = self._encode(x)
        posterior = DiagonalGaussianDistribution(h)
        if not return_dict:
            return (posterior,)
        return AutoencoderKLOutput(latent_dist=posterior)

    def _decode(
        self, z: torch.Tensor, return_dict: bool = True
    ) -> DecoderOutput | tuple[torch.Tensor]:
        _, _, num_frame, _, _ = z.shape

        if self.use_tiling:
            raise NotImplementedError("WanVAE tiled decode is not implemented")

        self.clear_cache()
        z = _format_5d_activation(z, self.layout_mode)
        x = self.post_quant_conv(z)
        out_chunks: list[torch.Tensor] = []
        for i in range(num_frame):
            self._conv_idx = [0]
            out_chunk = self.decoder(
                x[:, :, i : i + 1, :, :],
                feat_cache=self._feat_map,
                feat_idx=self._conv_idx,
                first_chunk=i == 0,
            )
            out_chunks.append(out_chunk)

        if len(out_chunks) == 1:
            out = out_chunks[0]
        else:
            out = torch.cat(out_chunks, dim=2)

        if self.config.patch_size is not None:
            out = unpatchify(out, patch_size=self.config.patch_size)
        out = _format_5d_activation(out, self.layout_mode)
        out = torch.clamp(out, min=-1.0, max=1.0)

        self.clear_cache()
        if not return_dict:
            return (out,)
        return DecoderOutput(sample=out)

    def decode(
        self, z: torch.Tensor, return_dict: bool = True
    ) -> DecoderOutput | tuple[torch.Tensor]:
        if self.use_slicing and z.shape[0] > 1:
            decoded_slices = [self._decode(z_slice).sample for z_slice in z.split(1)]
            decoded = torch.cat(decoded_slices)
        else:
            decoded = self._decode(z).sample

        if not return_dict:
            return (decoded,)
        return DecoderOutput(sample=decoded)

    def forward(
        self,
        sample: torch.Tensor,
        sample_posterior: bool = False,
        return_dict: bool = True,
        generator: torch.Generator | None = None,
    ) -> DecoderOutput | tuple[torch.Tensor]:
        posterior = self.encode(sample).latent_dist
        z = posterior.sample(generator=generator) if sample_posterior else posterior.mode()
        return self.decode(z, return_dict=return_dict)
