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

        def convs(dilations: tuple[int, ...]) -> nn.ModuleList:
            return nn.ModuleList(
                [
                    nn.Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=d,
                        padding=(kernel_size * d - d) // 2,
                    )
                    for d in dilations
                ]
            )

        def acts(n: int) -> nn.ModuleList:
            return nn.ModuleList([Activation1d(SnakeBeta(channels)) for _ in range(n)])

        self.convs1 = convs(dilation)
        self.convs2 = convs((1,) * len(dilation))
        self.acts1 = acts(len(self.convs1))
        self.acts2 = acts(len(self.convs2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for conv1, conv2, act1, act2 in zip(
            self.convs1, self.convs2, self.acts1, self.acts2, strict=True
        ):
            residual = conv2(act2(conv1(act1(x))))
            x = x + residual
        return x
