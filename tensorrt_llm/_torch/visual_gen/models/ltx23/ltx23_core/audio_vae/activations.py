# SPDX-FileCopyrightText: Copyright (c) 2025-2026 Lightricks Ltd.
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: LicenseRef-LTX-2

import math

import torch
import torch.nn.functional as F
from torch import nn


def _sinc(x: torch.Tensor) -> torch.Tensor:
    return torch.where(
        x == 0,
        torch.tensor(1.0, device=x.device, dtype=x.dtype),
        torch.sin(math.pi * x) / math.pi / x,
    )


def kaiser_sinc_filter1d(
    cutoff: float, half_width: float, kernel_size: int
) -> torch.Tensor:
    even = kernel_size % 2 == 0
    half_size = kernel_size // 2
    delta_f = 4 * half_width
    amplitude = 2.285 * (half_size - 1) * math.pi * delta_f + 7.95
    if amplitude > 50.0:
        beta = 0.1102 * (amplitude - 8.7)
    elif amplitude >= 21.0:
        beta = 0.5842 * (amplitude - 21) ** 0.4 + 0.07886 * (amplitude - 21.0)
    else:
        beta = 0.0
    window = torch.kaiser_window(kernel_size, beta=beta, periodic=False)
    time = (
        torch.arange(-half_size, half_size) + 0.5
        if even
        else torch.arange(kernel_size) - half_size
    )
    if cutoff == 0:
        filter_ = torch.zeros_like(time)
    else:
        filter_ = 2 * cutoff * window * _sinc(2 * cutoff * time)
        filter_ /= filter_.sum()
    return filter_.view(1, 1, kernel_size)


class LowPassFilter1d(nn.Module):
    def __init__(
        self,
        cutoff: float = 0.5,
        half_width: float = 0.6,
        stride: int = 1,
        padding: bool = True,
        padding_mode: str = "replicate",
        kernel_size: int = 12,
    ) -> None:
        super().__init__()
        if cutoff < 0:
            raise ValueError("Minimum cutoff must be larger than zero.")
        if cutoff > 0.5:
            raise ValueError("A cutoff above 0.5 does not make sense.")
        self.kernel_size = kernel_size
        self.even = kernel_size % 2 == 0
        self.pad_left = kernel_size // 2 - int(self.even)
        self.pad_right = kernel_size // 2
        self.stride = stride
        self.padding = padding
        self.padding_mode = padding_mode
        self.register_buffer(
            "filter", kaiser_sinc_filter1d(cutoff, half_width, kernel_size)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, n_channels, _ = x.shape
        if self.padding:
            x = F.pad(x, (self.pad_left, self.pad_right), mode=self.padding_mode)
        return F.conv1d(
            x,
            self.filter.expand(n_channels, -1, -1),
            stride=self.stride,
            groups=n_channels,
        )


class UpSample1d(nn.Module):
    def __init__(
        self,
        ratio: int = 2,
        kernel_size: int | None = None,
        persistent: bool = True,
        window_type: str = "kaiser",
    ) -> None:
        super().__init__()
        self.ratio = ratio
        self.stride = ratio

        if window_type == "hann":
            rolloff = 0.99
            lowpass_filter_width = 6
            width = math.ceil(lowpass_filter_width / rolloff)
            self.kernel_size = 2 * width * ratio + 1
            self.pad = width
            self.pad_left = 2 * width * ratio
            self.pad_right = self.kernel_size - ratio
            time_axis = (torch.arange(self.kernel_size) / ratio - width) * rolloff
            time_clamped = time_axis.clamp(-lowpass_filter_width, lowpass_filter_width)
            window = (
                torch.cos(time_clamped * math.pi / lowpass_filter_width / 2) ** 2
            )
            sinc_filter = (
                torch.sinc(time_axis) * window * rolloff / ratio
            ).view(1, 1, -1)
        else:
            self.kernel_size = (
                int(6 * ratio // 2) * 2 if kernel_size is None else kernel_size
            )
            self.pad = self.kernel_size // ratio - 1
            self.pad_left = (
                self.pad * self.stride + (self.kernel_size - self.stride) // 2
            )
            self.pad_right = (
                self.pad * self.stride
                + (self.kernel_size - self.stride + 1) // 2
            )
            sinc_filter = kaiser_sinc_filter1d(
                cutoff=0.5 / ratio,
                half_width=0.6 / ratio,
                kernel_size=self.kernel_size,
            )

        self.register_buffer("filter", sinc_filter, persistent=persistent)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, n_channels, _ = x.shape
        x = F.pad(x, (self.pad, self.pad), mode="replicate")
        filt = self.filter.to(dtype=x.dtype, device=x.device).expand(
            n_channels, -1, -1
        )
        x = self.ratio * F.conv_transpose1d(
            x, filt, stride=self.stride, groups=n_channels
        )
        return x[..., self.pad_left : -self.pad_right]


class DownSample1d(nn.Module):
    def __init__(self, ratio: int = 2, kernel_size: int | None = None) -> None:
        super().__init__()
        self.ratio = ratio
        self.kernel_size = (
            int(6 * ratio // 2) * 2 if kernel_size is None else kernel_size
        )
        self.lowpass = LowPassFilter1d(
            cutoff=0.5 / ratio,
            half_width=0.6 / ratio,
            stride=ratio,
            kernel_size=self.kernel_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lowpass(x)


class Activation1d(nn.Module):
    def __init__(
        self,
        activation: nn.Module,
        up_ratio: int = 2,
        down_ratio: int = 2,
        up_kernel_size: int = 12,
        down_kernel_size: int = 12,
    ) -> None:
        super().__init__()
        self.act = activation
        self.upsample = UpSample1d(up_ratio, up_kernel_size)
        self.downsample = DownSample1d(down_ratio, down_kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.downsample(self.act(self.upsample(x)))


class SnakeBeta(nn.Module):
    def __init__(
        self,
        in_features: int,
        alpha: float = 1.0,
        alpha_trainable: bool = True,
        alpha_logscale: bool = True,
    ) -> None:
        super().__init__()
        self.alpha_logscale = alpha_logscale
        self.alpha = nn.Parameter(
            torch.zeros(in_features)
            if alpha_logscale
            else torch.ones(in_features) * alpha
        )
        self.alpha.requires_grad = alpha_trainable
        self.beta = nn.Parameter(
            torch.zeros(in_features)
            if alpha_logscale
            else torch.ones(in_features) * alpha
        )
        self.beta.requires_grad = alpha_trainable
        self.eps = 1e-9

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        alpha = self.alpha.unsqueeze(0).unsqueeze(-1)
        beta = self.beta.unsqueeze(0).unsqueeze(-1)
        if self.alpha_logscale:
            alpha = torch.exp(alpha)
            beta = torch.exp(beta)
        return x + torch.sin(x * alpha).pow(2) / (beta + self.eps)
