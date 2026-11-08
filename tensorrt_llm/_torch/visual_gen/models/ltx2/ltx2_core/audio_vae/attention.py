# SPDX-FileCopyrightText: Copyright (c) 2025–2026 Lightricks Ltd.
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: LicenseRef-LTX-2

import torch

from ..normalization import NormType, build_normalization_layer


class AttnBlock(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        norm_type: NormType = NormType.GROUP,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.norm = build_normalization_layer(in_channels, normtype=norm_type)
        self.q = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.k = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.v = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.proj_out = torch.nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h_ = self.norm(x)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        b, c, h, w = q.shape
        q = q.reshape(b, c, h * w).permute(0, 2, 1).contiguous()
        k = k.reshape(b, c, h * w).contiguous()
        w_ = torch.bmm(q, k) * (int(c) ** (-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        v = v.reshape(b, c, h * w).contiguous()
        w_ = w_.permute(0, 2, 1).contiguous()
        h_ = torch.bmm(v, w_).reshape(b, c, h, w).contiguous()
        h_ = self.proj_out(h_)
        return x + h_
