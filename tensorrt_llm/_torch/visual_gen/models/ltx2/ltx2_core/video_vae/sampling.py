# Ported from https://github.com/Lightricks/LTX-2
# packages/ltx-core/src/ltx_core/model/video_vae/sampling.py
# Decoder-only: SpaceToDepthDownsample removed (encoder-only).

import math
from typing import Tuple

import torch
from einops import rearrange
from torch import nn

from .convolution import make_conv_nd
from .enums import PaddingModeType


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
        self.out_channels = (
            math.prod(stride) * in_channels // out_channels_reduction_factor
        )
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
