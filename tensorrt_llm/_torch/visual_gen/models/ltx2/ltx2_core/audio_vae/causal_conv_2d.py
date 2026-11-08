# SPDX-FileCopyrightText: Copyright (c) 2025–2026 Lightricks Ltd.
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: LicenseRef-LTX-2

import torch
import torch.nn.functional as F

from .causality_axis import CausalityAxis


class CausalConv2d(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int],
        stride: int = 1,
        dilation: int | tuple[int, int] = 1,
        groups: int = 1,
        bias: bool = True,
        causality_axis: CausalityAxis = CausalityAxis.HEIGHT,
    ) -> None:
        super().__init__()
        self.causality_axis = causality_axis
        kernel_size = torch.nn.modules.utils._pair(kernel_size)
        dilation = torch.nn.modules.utils._pair(dilation)
        pad_h = (kernel_size[0] - 1) * dilation[0]
        pad_w = (kernel_size[1] - 1) * dilation[1]

        match self.causality_axis:
            case CausalityAxis.NONE:
                self.padding = (
                    pad_w // 2,
                    pad_w - pad_w // 2,
                    pad_h // 2,
                    pad_h - pad_h // 2,
                )
            case CausalityAxis.WIDTH | CausalityAxis.WIDTH_COMPATIBILITY:
                self.padding = (
                    pad_w,
                    0,
                    pad_h // 2,
                    pad_h - pad_h // 2,
                )
            case CausalityAxis.HEIGHT:
                self.padding = (
                    pad_w // 2,
                    pad_w - pad_w // 2,
                    pad_h,
                    0,
                )
            case _:
                raise ValueError(f"Invalid causality_axis: {causality_axis}")

        self.conv = torch.nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=0,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pad(x, self.padding)
        return self.conv(x)


def make_conv2d(
    in_channels: int,
    out_channels: int,
    kernel_size: int | tuple[int, int],
    stride: int = 1,
    dilation: int = 1,
    groups: int = 1,
    bias: bool = True,
    causality_axis: CausalityAxis = CausalityAxis.HEIGHT,
) -> CausalConv2d:
    return CausalConv2d(
        in_channels,
        out_channels,
        kernel_size,
        stride,
        dilation,
        groups,
        bias,
        causality_axis,
    )
