# SPDX-FileCopyrightText: Copyright (c) 2025–2026 Lightricks Ltd.
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: LicenseRef-LTX-2

from typing import Set, Tuple

import torch

from ..normalization import NormType
from .attention import AttnBlock
from .causal_conv_2d import make_conv2d
from .causality_axis import CausalityAxis
from .resnet import ResnetBlock


class Upsample(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        with_conv: bool,
        causality_axis: CausalityAxis = CausalityAxis.HEIGHT,
    ) -> None:
        super().__init__()
        self.with_conv = with_conv
        self.causality_axis = causality_axis
        if self.with_conv:
            self.conv = make_conv2d(
                in_channels,
                in_channels,
                kernel_size=3,
                stride=1,
                causality_axis=causality_axis,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
            match self.causality_axis:
                case CausalityAxis.NONE:
                    pass
                case CausalityAxis.HEIGHT:
                    x = x[:, :, 1:, :]
                case CausalityAxis.WIDTH:
                    x = x[:, :, :, 1:]
                case CausalityAxis.WIDTH_COMPATIBILITY:
                    pass
                case _:
                    raise ValueError(f"Invalid causality_axis: {self.causality_axis}")
        return x


def build_upsampling_path(
    *,
    ch: int,
    ch_mult: Tuple[int, ...],
    num_resolutions: int,
    num_res_blocks: int,
    resolution: int,
    temb_channels: int,
    dropout: float,
    norm_type: NormType,
    causality_axis: CausalityAxis,
    attn_resolutions: Set[int],
    resamp_with_conv: bool,
    initial_block_channels: int,
) -> tuple[torch.nn.ModuleList, int]:
    up_modules = torch.nn.ModuleList()
    block_in = initial_block_channels
    curr_res = resolution // (2 ** (num_resolutions - 1))

    for level in reversed(range(num_resolutions)):
        stage = torch.nn.Module()
        stage.block = torch.nn.ModuleList()
        stage.attn = torch.nn.ModuleList()
        block_out = ch * ch_mult[level]

        for _ in range(num_res_blocks + 1):
            stage.block.append(
                ResnetBlock(
                    in_channels=block_in,
                    out_channels=block_out,
                    temb_channels=temb_channels,
                    dropout=dropout,
                    norm_type=norm_type,
                    causality_axis=causality_axis,
                )
            )
            block_in = block_out
            if curr_res in attn_resolutions:
                stage.attn.append(AttnBlock(block_in, norm_type=norm_type))

        if level != 0:
            stage.upsample = Upsample(block_in, resamp_with_conv, causality_axis=causality_axis)
            curr_res *= 2

        up_modules.insert(0, stage)

    return up_modules, block_in
