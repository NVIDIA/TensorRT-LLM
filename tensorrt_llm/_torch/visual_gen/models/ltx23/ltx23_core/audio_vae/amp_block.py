# SPDX-FileCopyrightText: Copyright (c) 2025-2026 Lightricks Ltd.
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: LicenseRef-LTX-2

import torch
from torch import nn

from .activations import Activation1d, SnakeBeta


class AMPBlock1(nn.Module):
    """BigVGAN anti-aliased multi-periodicity residual block."""

    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        dilation: tuple[int, int, int] = (1, 3, 5),
    ) -> None:
        super().__init__()

        def get_padding(dilation_value: int) -> int:
            return (kernel_size * dilation_value - dilation_value) // 2

        self.convs1 = nn.ModuleList(
            [
                nn.Conv1d(
                    channels,
                    channels,
                    kernel_size,
                    1,
                    dilation=dilation_value,
                    padding=get_padding(dilation_value),
                )
                for dilation_value in dilation
            ]
        )
        self.convs2 = nn.ModuleList(
            [
                nn.Conv1d(
                    channels,
                    channels,
                    kernel_size,
                    1,
                    dilation=1,
                    padding=get_padding(1),
                )
                for _ in dilation
            ]
        )
        self.acts1 = nn.ModuleList(
            [Activation1d(SnakeBeta(channels)) for _ in self.convs1]
        )
        self.acts2 = nn.ModuleList(
            [Activation1d(SnakeBeta(channels)) for _ in self.convs2]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for conv1, conv2, act1, act2 in zip(
            self.convs1, self.convs2, self.acts1, self.acts2, strict=True
        ):
            residual = conv2(act2(conv1(act1(x))))
            x = x + residual
        return x
