# SPDX-FileCopyrightText: Copyright (c) 2025–2026 Lightricks Ltd.
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: LicenseRef-LTX-2

from enum import Enum

import torch
from torch import nn


class NormType(Enum):
    GROUP = "group"
    PIXEL = "pixel"


class PixelNorm(nn.Module):
    """Per-pixel (per-location) RMS normalization layer."""

    def __init__(self, dim: int = 1, eps: float = 1e-8) -> None:
        super().__init__()
        self.dim = dim
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean_sq = torch.mean(x**2, dim=self.dim, keepdim=True)
        rms = torch.sqrt(mean_sq + self.eps)
        return x / rms


def build_normalization_layer(
    in_channels: int, *, num_groups: int = 32, normtype: NormType = NormType.GROUP
) -> nn.Module:
    if normtype == NormType.GROUP:
        return torch.nn.GroupNorm(
            num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True
        )
    if normtype == NormType.PIXEL:
        return PixelNorm(dim=1, eps=1e-6)
    raise ValueError(f"Invalid normalization type: {normtype}")
