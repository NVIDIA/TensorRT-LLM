# SPDX-FileCopyrightText: Copyright (c) 2025–2026 Lightricks Ltd.
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: LicenseRef-LTX-2

import math
from typing import Tuple

import torch
from einops import rearrange
from torch import nn

from .convolution import make_conv_nd
from .enums import PaddingModeType


class SpaceToDepthDownsample(nn.Module):
    """Spatial/temporal downsampling via conv + space-to-depth rearrangement.

    Matches the reference LTX-2 encoder architecture: a stride-1 conv
    operates at the original spatial resolution, producing
    ``out_channels // prod(stride)`` channels, followed by a
    space-to-depth rearrange that folds spatial/temporal dimensions
    into the channel axis to yield ``out_channels``.

    When ``residual=True`` the skip connection applies the same
    space-to-depth rearrangement to the input, groups the resulting
    channels, and mean-pools over each group to match ``out_channels``.
    """

    def __init__(
        self,
        dims: int | Tuple[int, int],
        in_channels: int,
        out_channels: int,
        stride: Tuple[int, int, int],
        residual: bool = False,
        spatial_padding_mode: PaddingModeType = PaddingModeType.ZEROS,
    ):
        super().__init__()
        self.stride = stride
        self.out_channels = out_channels
        self.residual = residual
        self.group_size = in_channels * math.prod(stride) // out_channels
        self.conv = make_conv_nd(
            dims=dims,
            in_channels=in_channels,
            out_channels=out_channels // math.prod(stride),
            kernel_size=3,
            stride=1,
            causal=True,
            spatial_padding_mode=spatial_padding_mode,
        )

    def forward(self, x: torch.Tensor, causal: bool = True) -> torch.Tensor:
        if self.stride[0] == 2:
            x = torch.cat([x[:, :, :1, :, :], x], dim=2)

        if self.residual:
            x_in = rearrange(
                x,
                "b c (d p1) (h p2) (w p3) -> b (c p1 p2 p3) d h w",
                p1=self.stride[0],
                p2=self.stride[1],
                p3=self.stride[2],
            )
            x_in = rearrange(x_in, "b (c g) d h w -> b c g d h w", g=self.group_size)
            x_in = x_in.mean(dim=2)

        x = self.conv(x, causal=causal)
        x = rearrange(
            x,
            "b c (d p1) (h p2) (w p3) -> b (c p1 p2 p3) d h w",
            p1=self.stride[0],
            p2=self.stride[1],
            p3=self.stride[2],
        )

        if self.residual:
            x = x + x_in

        return x


class DepthToSpaceUpsample(nn.Module):
    def __init__(
        self,
        dims: int | Tuple[int, int],
        in_channels: int,
        stride: Tuple[int, int, int],
        residual: bool = False,
        out_channels_reduction_factor: int = 1,
        spatial_padding_mode: PaddingModeType = PaddingModeType.ZEROS,
    ):
        super().__init__()
        self.stride = stride
        self.out_channels = math.prod(stride) * in_channels // out_channels_reduction_factor
        self.conv = make_conv_nd(
            dims=dims,
            in_channels=in_channels,
            out_channels=self.out_channels,
            kernel_size=3,
            stride=1,
            causal=True,
            spatial_padding_mode=spatial_padding_mode,
        )
        self.residual = residual
        self.out_channels_reduction_factor = out_channels_reduction_factor

    def forward(self, x: torch.Tensor, causal: bool = True) -> torch.Tensor:
        if self.residual:
            x_in = rearrange(
                x,
                "b (c p1 p2 p3) d h w -> b c (d p1) (h p2) (w p3)",
                p1=self.stride[0],
                p2=self.stride[1],
                p3=self.stride[2],
            )
            num_repeat = math.prod(self.stride) // self.out_channels_reduction_factor
            x_in = x_in.repeat(1, num_repeat, 1, 1, 1)
            if self.stride[0] == 2:
                x_in = x_in[:, :, 1:, :, :]

        x = self.conv(x, causal=causal)
        x = rearrange(
            x,
            "b (c p1 p2 p3) d h w -> b c (d p1) (h p2) (w p3)",
            p1=self.stride[0],
            p2=self.stride[1],
            p3=self.stride[2],
        )
        if self.stride[0] == 2:
            x = x[:, :, 1:, :, :]
        if self.residual:
            x = x + x_in
        return x
