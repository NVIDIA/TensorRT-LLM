# SPDX-FileCopyrightText: Copyright (c) 2025-2026 Lightricks Ltd.
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: LicenseRef-LTX-2

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from torch import nn

from .activations import UpSample1d

if TYPE_CHECKING:
    from .vocoder import Vocoder


class _STFTFn(nn.Module):
    def __init__(self, filter_length: int, hop_length: int, win_length: int) -> None:
        super().__init__()
        self.hop_length = hop_length
        self.win_length = win_length
        n_freqs = filter_length // 2 + 1
        self.register_buffer("forward_basis", torch.zeros(n_freqs * 2, 1, filter_length))

    def forward(self, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if y.dim() == 2:
            y = y.unsqueeze(1)
        left_pad = max(0, self.win_length - self.hop_length)
        y = F.pad(y, (left_pad, 0))
        spec = F.conv1d(y, self.forward_basis, stride=self.hop_length)
        n_freqs = spec.shape[1] // 2
        real, imag = spec[:, :n_freqs], spec[:, n_freqs:]
        magnitude = torch.sqrt(real**2 + imag**2)
        phase = torch.atan2(imag.float(), real.float()).to(real.dtype)
        return magnitude, phase


class MelSTFT(nn.Module):
    def __init__(
        self,
        filter_length: int,
        hop_length: int,
        win_length: int,
        n_mel_channels: int,
    ) -> None:
        super().__init__()
        self.stft_fn = _STFTFn(filter_length, hop_length, win_length)
        n_freqs = filter_length // 2 + 1
        self.register_buffer("mel_basis", torch.zeros(n_mel_channels, n_freqs))

    def mel_spectrogram(
        self, y: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        magnitude, phase = self.stft_fn(y)
        energy = torch.norm(magnitude, dim=1)
        mel = torch.matmul(self.mel_basis.to(magnitude.dtype), magnitude)
        log_mel = torch.log(torch.clamp(mel, min=1e-5))
        return log_mel, magnitude, phase, energy


class VocoderWithBWE(nn.Module):
    def __init__(
        self,
        vocoder: "Vocoder",
        bwe_generator: "Vocoder",
        mel_stft: MelSTFT,
        input_sampling_rate: int,
        output_sampling_rate: int,
        hop_length: int,
    ) -> None:
        super().__init__()
        self.vocoder = vocoder
        self.bwe_generator = bwe_generator
        self.mel_stft = mel_stft
        self.input_sampling_rate = input_sampling_rate
        self.output_sampling_rate = output_sampling_rate
        self.hop_length = hop_length
        with torch.device("cpu"):
            self.resampler = UpSample1d(
                ratio=output_sampling_rate // input_sampling_rate,
                persistent=False,
                window_type="hann",
            )

    def _compute_mel(self, audio: torch.Tensor) -> torch.Tensor:
        batch, n_channels, _ = audio.shape
        flat = audio.reshape(batch * n_channels, -1)
        mel, _, _, _ = self.mel_stft.mel_spectrogram(flat)
        return mel.reshape(batch, n_channels, mel.shape[1], mel.shape[2])

    def forward(self, mel_spec: torch.Tensor) -> torch.Tensor:
        input_dtype = mel_spec.dtype
        with torch.autocast(device_type=mel_spec.device.type, dtype=torch.float32):
            x = self.vocoder(mel_spec.float())
            length_low_rate = x.shape[-1]
            output_length = length_low_rate * self.output_sampling_rate // self.input_sampling_rate

            remainder = length_low_rate % self.hop_length
            if remainder:
                x = F.pad(x, (0, self.hop_length - remainder))

            mel = self._compute_mel(x)
            residual = self.bwe_generator(mel.transpose(2, 3))
            skip = self.resampler(x)
            assert residual.shape == skip.shape, f"residual {residual.shape} != skip {skip.shape}"
            return torch.clamp(residual + skip, -1, 1)[..., :output_length].to(input_dtype)
