# SPDX-FileCopyrightText: Copyright (c) 2025-2026 Lightricks Ltd.
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: LicenseRef-LTX-2

import torch

from ....ltx2.ltx2_core.audio_vae.attention import AttnBlock
from ....ltx2.ltx2_core.audio_vae.causality_axis import CausalityAxis
from ....ltx2.ltx2_core.audio_vae.resnet import ResnetBlock
from ....ltx2.ltx2_core.normalization import NormType


class Downsample(torch.nn.Module):
    """Strided-convolution (or average-pool) downsampling with causal padding.

    The convolution is a plain ``Conv2d`` with ``padding=0``; the asymmetric pad
    is applied by hand because torch convolutions cannot express a pad that is
    larger on one side of an axis, which is exactly what causality needs.
    """

    def __init__(
        self,
        in_channels: int,
        with_conv: bool,
        causality_axis: CausalityAxis = CausalityAxis.HEIGHT,
    ) -> None:
        super().__init__()
        self.with_conv = with_conv
        self.causality_axis = causality_axis

        if self.causality_axis != CausalityAxis.NONE and not self.with_conv:
            raise ValueError("causality is only supported when `with_conv=True`.")

        if self.with_conv:
            self.conv = torch.nn.Conv2d(
                in_channels, in_channels, kernel_size=3, stride=2, padding=0
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.with_conv:
            return torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)

        # F.pad tuple order is (left, right, top, bottom).
        match self.causality_axis:
            case CausalityAxis.NONE:
                pad = (0, 1, 0, 1)
            case CausalityAxis.WIDTH:
                pad = (2, 0, 0, 1)
            case CausalityAxis.HEIGHT:
                pad = (0, 1, 2, 0)
            case CausalityAxis.WIDTH_COMPATIBILITY:
                pad = (1, 0, 0, 1)
            case _:
                raise ValueError(f"Invalid causality_axis: {self.causality_axis}")

        x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
        return self.conv(x)


def build_downsampling_path(
    *,
    ch: int,
    ch_mult: tuple[int, ...],
    num_res_blocks: int,
    resolution: int,
    dropout: float,
    norm_type: NormType,
    causality_axis: CausalityAxis,
    attn_resolutions: set[int],
    resamp_with_conv: bool,
) -> tuple[torch.nn.ModuleList, int]:
    """Build the encoder downsampling path; mirror of :func:`build_upsampling_path`."""
    down_modules = torch.nn.ModuleList()
    curr_res = resolution
    in_ch_mult = (1, *ch_mult)
    block_in = ch

    num_resolutions = len(ch_mult)
    for level in range(num_resolutions):
        stage = torch.nn.Module()
        stage.block = torch.nn.ModuleList()
        stage.attn = torch.nn.ModuleList()
        block_in = ch * in_ch_mult[level]
        block_out = ch * ch_mult[level]

        for _ in range(num_res_blocks):
            stage.block.append(
                ResnetBlock(
                    in_channels=block_in,
                    out_channels=block_out,
                    temb_channels=0,
                    dropout=dropout,
                    norm_type=norm_type,
                    causality_axis=causality_axis,
                )
            )
            block_in = block_out
            if curr_res in attn_resolutions:
                stage.attn.append(AttnBlock(block_in, norm_type=norm_type))

        if level != num_resolutions - 1:
            stage.downsample = Downsample(block_in, resamp_with_conv, causality_axis=causality_axis)
            curr_res //= 2

        down_modules.append(stage)

    return down_modules, block_in
