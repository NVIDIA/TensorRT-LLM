# SPDX-FileCopyrightText: Copyright (c) 2025–2026 Lightricks Ltd.
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: LicenseRef-LTX-2

from typing import Tuple

import torch

from ..normalization import NormType, build_normalization_layer
from .causal_conv_2d import make_conv2d
from .causality_axis import CausalityAxis

LRELU_SLOPE = 0.1


class ResBlock1(torch.nn.Module):
    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        dilation: Tuple[int, int, int] = (1, 3, 5),
    ):
        super().__init__()
        self.convs1 = torch.nn.ModuleList(
            [
                torch.nn.Conv1d(
                    channels,
                    channels,
                    kernel_size,
                    1,
                    dilation=dilation[0],
                    padding="same",
                ),
                torch.nn.Conv1d(
                    channels,
                    channels,
                    kernel_size,
                    1,
                    dilation=dilation[1],
                    padding="same",
                ),
                torch.nn.Conv1d(
                    channels,
                    channels,
                    kernel_size,
                    1,
                    dilation=dilation[2],
                    padding="same",
                ),
            ]
        )
        self.convs2 = torch.nn.ModuleList(
            [
                torch.nn.Conv1d(
                    channels,
                    channels,
                    kernel_size,
                    1,
                    dilation=1,
                    padding="same",
                ),
                torch.nn.Conv1d(
                    channels,
                    channels,
                    kernel_size,
                    1,
                    dilation=1,
                    padding="same",
                ),
                torch.nn.Conv1d(
                    channels,
                    channels,
                    kernel_size,
                    1,
                    dilation=1,
                    padding="same",
                ),
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for conv1, conv2 in zip(self.convs1, self.convs2, strict=True):
            xt = torch.nn.functional.leaky_relu(x, LRELU_SLOPE)
            xt = conv1(xt)
            xt = torch.nn.functional.leaky_relu(xt, LRELU_SLOPE)
            xt = conv2(xt)
            x = xt + x
        return x


class ResBlock2(torch.nn.Module):
    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        dilation: Tuple[int, int] = (1, 3),
    ):
        super().__init__()
        self.convs = torch.nn.ModuleList(
            [
                torch.nn.Conv1d(
                    channels,
                    channels,
                    kernel_size,
                    1,
                    dilation=dilation[0],
                    padding="same",
                ),
                torch.nn.Conv1d(
                    channels,
                    channels,
                    kernel_size,
                    1,
                    dilation=dilation[1],
                    padding="same",
                ),
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for conv in self.convs:
            xt = torch.nn.functional.leaky_relu(x, LRELU_SLOPE)
            xt = conv(xt)
            x = xt + x
        return x


class ResnetBlock(torch.nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int | None = None,
        conv_shortcut: bool = False,
        dropout: float = 0.0,
        temb_channels: int = 512,
        norm_type: NormType = NormType.GROUP,
        causality_axis: CausalityAxis = CausalityAxis.HEIGHT,
    ) -> None:
        super().__init__()
        self.causality_axis = causality_axis

        if self.causality_axis != CausalityAxis.NONE and norm_type == NormType.GROUP:
            raise ValueError("Causal ResnetBlock with GroupNorm is not supported.")

        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = build_normalization_layer(in_channels, normtype=norm_type)
        self.non_linearity = torch.nn.SiLU()
        self.conv1 = make_conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            causality_axis=causality_axis,
        )
        if temb_channels > 0:
            self.temb_proj = torch.nn.Linear(temb_channels, out_channels)
        self.norm2 = build_normalization_layer(out_channels, normtype=norm_type)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = make_conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            causality_axis=causality_axis,
        )
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = make_conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=3,
                    stride=1,
                    causality_axis=causality_axis,
                )
            else:
                self.nin_shortcut = make_conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=1,
                    causality_axis=causality_axis,
                )

    def forward(
        self,
        x: torch.Tensor,
        temb: torch.Tensor | None = None,
    ) -> torch.Tensor:
        h = self.norm1(x)
        h = self.non_linearity(h)
        h = self.conv1(h)
        if temb is not None:
            h = h + self.temb_proj(self.non_linearity(temb))[:, :, None, None]
        h = self.norm2(h)
        h = self.non_linearity(h)
        h = self.dropout(h)
        h = self.conv2(h)
        if self.in_channels != self.out_channels:
            x = self.conv_shortcut(x) if self.use_conv_shortcut else self.nin_shortcut(x)
        return x + h
